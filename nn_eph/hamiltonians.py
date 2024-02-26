import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

# os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, lax
from jax import numpy as jnp


@dataclass
class kq:
    omega_q: Sequence
    e_k: Sequence
    g_kq: Any
    max_n_phonons: Any = jnp.inf

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_k = walker[0]
        phonon_occ = walker[1]
        n_sites = jnp.array(lattice.shape)

        overlap = wave.calc_overlap(elec_k, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(
            elec_k, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )

        # diagonal
        energy = (
            jnp.sum(jnp.array(self.omega_q) * phonon_occ)
            + jnp.array(self.e_k)[elec_k]
            + 0.0j
        )

        ratios = jnp.zeros((2 * len(lattice.sites),)) + 0.0j

        # e_ph coupling
        # carry = (energy, ratios)
        def scanned_fun(carry, kp):
            qc = tuple((jnp.array(elec_k) - kp) % n_sites)
            qd = tuple((kp - jnp.array(elec_k)) % n_sites)

            kp_i = lattice.get_site_num(kp)
            qc_i = lattice.get_site_num(qc)
            qd_i = lattice.get_site_num(qd)

            new_elec_k = tuple(kp)

            new_phonon_occ = phonon_occ.at[qc].add(1)
            # ratio = (jnp.sum(phonon_occ) < self.max_n_phonons) * wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[qc] + 1)**0.5
            phonon_pos = tuple(qc)
            phonon_change = 1
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * wave.calc_overlap_ratio(
                    elec_k,
                    new_elec_k,
                    phonon_pos,
                    phonon_change,
                    parameters,
                    lattice,
                    overlap,
                    new_phonon_occ,
                )
                / (phonon_occ[qc] + 1) ** 0.5
            )
            carry[0] -= (
                jnp.array(self.g_kq)[kp_i, qc_i] * (phonon_occ[qc] + 1) ** 0.5 * ratio
            )
            carry[1] = carry[1].at[2 * kp_i].set(ratio)

            new_phonon_occ = phonon_occ.at[qd].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            # ratio = (phonon_occ[qd])**0.5 * wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice) / overlap
            phonon_pos = tuple(qd)
            phonon_change = -1 * (phonon_occ[qd] > 0)
            ratio = (phonon_occ[qd]) ** 0.5 * wave.calc_overlap_ratio(
                elec_k,
                new_elec_k,
                phonon_pos,
                phonon_change,
                parameters,
                lattice,
                overlap,
                new_phonon_occ,
            )
            carry[0] -= (
                jnp.array(self.g_kq)[kp_i, qc_i] * (phonon_occ[qd]) ** 0.5 * ratio
            )
            carry[1] = carry[1].at[2 * kp_i + 1].set(ratio)

            return carry, (qc, qd)

        [energy, ratios], (qc, qd) = lax.scan(
            scanned_fun, [energy, ratios], jnp.array(lattice.sites)
        )

        qc = jnp.stack(qc, axis=-1)
        qd = jnp.stack(qd, axis=-1)

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print('\nwalker: {}', walker)
        # jax.debug.print('overlap: {}', overlap)
        ##jax.debug.print('overlap_gradient: {}', overlap_gradient)
        # jax.debug.print('weight: {}', weight)
        # jax.debug.print('ratios: {}', ratios)
        # jax.debug.print('energy: {}', energy)
        # jax.debug.print('random_number: {}', random_number)
        # jax.debug.print('new_ind: {}', new_ind)
        # jax.debug.print('qc: {}', qc)

        walker[0] = tuple(jnp.array(lattice.sites)[new_ind // 2])
        walker[1] = lax.cond(
            new_ind % 2 == 0,
            lambda x: x.at[tuple(qc[new_ind // 2])].add(1),
            lambda x: x.at[tuple(qd[new_ind // 2])].add(-1),
            walker[1],
        )
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update_lr(
        self, walker, parameters, wave, lattice, random_number
    ):
        elec_k = walker[0]
        phonon_occ = walker[1]
        n_sites = jnp.array(lattice.shape)

        overlap = wave.calc_overlap(elec_k, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(
            elec_k, phonon_occ, parameters, lattice
        )
        prob = (
            jnp.abs(jnp.exp(overlap)) ** 2
            + (jnp.abs(overlap_gradient * jnp.exp(overlap)) ** 2).sum()
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )

        # diagonal
        energy = (
            jnp.sum(jnp.array(self.omega_q) * phonon_occ)
            + jnp.array(self.e_k)[elec_k]
            + 0.0j
        )

        ratios = jnp.zeros((2 * len(lattice.sites),)) + 0.0j

        # e_ph coupling
        # carry = (energy, ratios)
        def scanned_fun(carry, kp):
            qc = tuple((jnp.array(elec_k) - kp) % n_sites)
            qd = tuple((kp - jnp.array(elec_k)) % n_sites)

            kp_i = lattice.get_site_num(kp)
            qc_i = lattice.get_site_num(qc)

            new_elec_k = tuple(kp)

            new_phonon_occ = phonon_occ.at[qc].add(1)
            # ratio = (jnp.sum(phonon_occ) < self.max_n_phonons) * wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[qc] + 1)**0.5
            # phonon_pos = tuple(qc)
            # phonon_change = 1
            new_overlap = wave.calc_overlap(
                new_elec_k, new_phonon_occ, parameters, lattice
            )
            overlap_ratio = jnp.exp(new_overlap - overlap)
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_elec_k, new_phonon_occ, parameters, lattice
            )
            new_prob = (
                jnp.abs(jnp.exp(new_overlap)) ** 2
                + (jnp.abs(new_overlap_gradient * jnp.exp(new_overlap)) ** 2).sum()
            )
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * overlap_ratio
                / (phonon_occ[qc] + 1) ** 0.5
            )
            carry[0] -= (
                jnp.array(self.g_kq)[kp_i, qc_i] * (phonon_occ[qc] + 1) ** 0.5 * ratio
            )
            prob_ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * new_prob**0.5
                / prob**0.5
                / (phonon_occ[qc] + 1) ** 0.5
            )
            carry[1] = carry[1].at[2 * kp_i].set(prob_ratio)

            new_phonon_occ = phonon_occ.at[qd].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            # ratio = (phonon_occ[qd])**0.5 * wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice) / overlap
            # phonon_pos = tuple(qd)
            # phonon_change = -1 * (phonon_occ[qd] > 0)
            new_overlap = wave.calc_overlap(
                new_elec_k, new_phonon_occ, parameters, lattice
            )
            overlap_ratio = jnp.exp(new_overlap - overlap)
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_elec_k, new_phonon_occ, parameters, lattice
            )
            new_prob = (
                jnp.abs(jnp.exp(new_overlap)) ** 2
                + (jnp.abs(new_overlap_gradient * jnp.exp(new_overlap)) ** 2).sum()
            )
            ratio = (phonon_occ[qd]) ** 0.5 * overlap_ratio
            carry[0] -= (
                jnp.array(self.g_kq)[kp_i, qc_i] * (phonon_occ[qd]) ** 0.5 * ratio
            )
            prob_ratio = (phonon_occ[qd]) ** 0.5 * new_prob**0.5 / prob**0.5
            carry[1] = carry[1].at[2 * kp_i + 1].set(prob_ratio)

            return carry, (qc, qd)

        [energy, ratios], (qc, qd) = lax.scan(
            scanned_fun, [energy, ratios], jnp.array(lattice.sites)
        )

        qc = jnp.stack(qc, axis=-1)
        qd = jnp.stack(qd, axis=-1)

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print('\nwalker: {}', walker)
        # jax.debug.print('overlap: {}', overlap)
        ##jax.debug.print('overlap_gradient: {}', overlap_gradient)
        # jax.debug.print('weight: {}', weight)
        # jax.debug.print('ratios: {}', ratios)
        # jax.debug.print('energy: {}', energy)
        # jax.debug.print('random_number: {}', random_number)
        # jax.debug.print('new_ind: {}', new_ind)
        # jax.debug.print('qc: {}', qc)

        walker[0] = tuple(jnp.array(lattice.sites)[new_ind // 2])
        walker[1] = lax.cond(
            new_ind % 2 == 0,
            lambda x: x.at[tuple(qc[new_ind // 2])].add(1),
            lambda x: x.at[tuple(qd[new_ind // 2])].add(-1),
            walker[1],
        )
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega_q, self.e_k, self.g_kq, self.max_n_phonons))


@dataclass
class kq_ne_np:
    omega_nu_q: Sequence
    e_n_k: Sequence
    g_mn_nu_kq: Any
    max_n_phonons: Any = jnp.inf

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_n_k = walker[0]
        elec_n = walker[0][0]
        elec_k = walker[0][1]
        phonon_occ = walker[1]
        n_sites = jnp.array(lattice.shape)
        n_e_bands = len(self.e_n_k)
        n_p_bands = len(self.omega_nu_q)

        overlap = wave.calc_overlap(elec_n_k, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(
            elec_n_k, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )

        # diagonal
        energy = (
            jnp.sum(jnp.array(self.omega_nu_q) * phonon_occ)
            + jnp.array(self.e_n_k)[elec_n][elec_k]
            + 0.0j
        )

        # e_ph coupling
        # carry = (energy, ratios, m, nu)
        def scanned_fun(carry, kp):
            m = carry[2]
            nu = carry[3]
            qc = tuple((jnp.array(elec_k) - kp) % n_sites)
            qd = tuple((kp - jnp.array(elec_k)) % n_sites)

            kp_i = lattice.get_site_num(kp)
            qc_i = lattice.get_site_num(qc)
            qd_i = lattice.get_site_num(qd)

            new_elec_n_k = (m, tuple(kp))

            new_phonon_occ = phonon_occ.at[(nu, *qc)].add(1)
            phonon_pos = (nu, tuple(qc))
            phonon_change = 1
            # ratio = (jnp.sum(phonon_occ) < self.max_n_phonons) * wave.calc_overlap(new_elec_n_k,
            #                                                                       new_phonon_occparameters, lattice) / overlap /     (phonon_occ[(nu, *qc)] + 1)**0.5
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * wave.calc_overlap_ratio(
                    elec_n_k,
                    new_elec_n_k,
                    phonon_pos,
                    phonon_change,
                    parameters,
                    lattice,
                    overlap,
                    new_phonon_occ,
                )
                / (phonon_occ[(nu, *qc)] + 1) ** 0.5
            )
            carry[0] -= (
                jnp.array(self.g_mn_nu_kq)[elec_n, m, nu, kp_i, qc_i]
                * (phonon_occ[(nu, *qc)] + 1) ** 0.5
                * ratio
            )
            carry[1] = carry[1].at[m, nu, 2 * kp_i].set(ratio)

            new_phonon_occ = phonon_occ.at[(nu, *qd)].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            phonon_pos = (nu, tuple(qd))
            phonon_change = -1 * (phonon_occ[(nu, *qd)] > 0)
            ratio = (phonon_occ[(nu, *qd)]) ** 0.5 * wave.calc_overlap_ratio(
                elec_n_k,
                new_elec_n_k,
                phonon_pos,
                phonon_change,
                parameters,
                lattice,
                overlap,
                new_phonon_occ,
            )
            carry[0] -= (
                jnp.array(self.g_mn_nu_kq)[elec_n, m, nu, kp_i, qc_i]
                * (phonon_occ[(nu, *qd)]) ** 0.5
                * ratio
            )
            carry[1] = carry[1].at[m, nu, 2 * kp_i + 1].set(ratio)

            return carry, (qc, qd)

        def outer_scanned_fun(carry, m_nu):
            m = m_nu // n_p_bands
            nu = m_nu % n_p_bands
            [carry[0], carry[1], _, _], (qc, qd) = lax.scan(
                scanned_fun, [carry[0], carry[1], m, nu], jnp.array(lattice.sites)
            )
            return [carry[0], carry[1], qc, qd], m_nu

        ratios = (
            jnp.zeros(
                (
                    n_e_bands,
                    n_p_bands,
                    2 * len(lattice.sites),
                )
            )
            + 0.0j
        )
        qc = tuple(
            [
                jnp.zeros(len(lattice.sites), dtype=jnp.int32)
                for _ in range(len(lattice.shape))
            ]
        )
        qd = tuple(
            [
                jnp.zeros(len(lattice.sites), dtype=jnp.int32)
                for _ in range(len(lattice.shape))
            ]
        )
        [energy, ratios, qc, qd], _ = lax.scan(
            outer_scanned_fun,
            [energy, ratios, qc, qd],
            jnp.arange(n_e_bands * n_p_bands),
        )
        qc = jnp.stack(qc, axis=-1)
        qd = jnp.stack(qd, axis=-1)

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios.reshape(-1)))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print('\nwalker: {}', walker)
        # jax.debug.print('overlap: {}', overlap)
        ##jax.debug.print('overlap_gradient: {}', overlap_gradient)
        # jax.debug.print('weight: {}', weight)
        # jax.debug.print('ratios: {}', ratios)
        # jax.debug.print('energy: {}', energy)
        # jax.debug.print('random_number: {}', random_number)
        # jax.debug.print('new_ind: {}', new_ind)
        # jax.debug.print('qc: {}', qc)

        new_e_band = new_ind // (2 * len(lattice.sites) * n_p_bands)
        p_band = (new_ind % (2 * len(lattice.sites) * n_p_bands)) // (
            2 * len(lattice.sites)
        )
        new_ind = new_ind % (2 * len(lattice.sites))
        walker[0] = (new_e_band, tuple(jnp.array(lattice.sites)[new_ind // 2]))
        walker[1] = lax.cond(
            new_ind % 2 == 0,
            lambda x: x.at[(p_band, *tuple(qc[new_ind // 2]))].add(1),
            lambda x: x.at[(p_band, *tuple(qd[new_ind // 2]))].add(-1),
            walker[1],
        )
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega_nu_q, self.e_n_k, self.g_mn_nu_kq, self.max_n_phonons))


@dataclass
class holstein:
    omega: float
    g: float
    max_n_phonons: any = jnp.inf

    @partial(jit, static_argnums=(0, 3, 4))
    def local_correlation_functions(self, walker, parameters, wave, lattice):
        elec_pos = walker[0]
        phonon_occ = walker[1]
        overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)

        # ke
        # carry: [ energy ]
        def scanned_fun(carry, x):
            new_overlap = wave.calc_overlap(tuple(x), phonon_occ, parameters, lattice)
            ratio = jnp.exp(new_overlap - overlap)
            carry[0] -= ratio
            return carry, ratio

        ke = 0.0j
        [ke], _ = lax.scan(scanned_fun, [ke], lattice.get_nearest_neighbors(elec_pos))

        # phonon_numbers
        phonon_numbers = phonon_occ.copy()

        # e_ph correlation
        # shift phonon_occ by elec_pos
        input_ar = phonon_occ.reshape(-1, *lattice.shape)
        for ax in range(len(lattice.shape)):
            for phonon_type in range(phonon_occ.shape[0]):
                input_ar = input_ar.at[phonon_type].set(
                    input_ar[phonon_type].take(
                        elec_pos[ax] + jnp.arange(lattice.shape[ax]),
                        axis=ax,
                        mode="wrap",
                    )
                )
        shifted_phonon_occ = jnp.stack([*input_ar], axis=-1).reshape(phonon_occ.shape)

        # calculate local phonon position at each site
        # carry: [ counter ]
        def scanned_fun(carry, x):
            pos = tuple(x)
            phonon_pos = 0.0j

            new_phonon_occ = shifted_phonon_occ.at[pos].add(1)
            ratio = (
                (jnp.sum(shifted_phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (shifted_phonon_occ[pos] + 1) ** 0.5
            )
            phonon_pos += (shifted_phonon_occ[pos] + 1) ** 0.5 * ratio

            new_phonon_occ = shifted_phonon_occ.at[pos].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (shifted_phonon_occ[pos]) ** 0.5 * jnp.exp(
                wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            phonon_pos += (shifted_phonon_occ[pos]) ** 0.5 * ratio

            carry[0] = carry[0] + 1
            return carry, phonon_pos

        counter = 0
        _, pos_corr = lax.scan(scanned_fun, [counter], jnp.array(lattice.sites))
        pos_corr = pos_corr.reshape(shifted_phonon_occ.shape)

        average_phonon_pos = jnp.average(pos_corr)

        return ke, phonon_numbers, pos_corr, average_phonon_pos

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_pos = walker[0]
        phonon_occ = walker[1]

        overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(
            elec_pos, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )

        # diagonal
        energy = self.omega * jnp.sum(phonon_occ) + 0.0j

        # electron hops
        nearest_neighbors = lattice.get_nearest_neighbors(elec_pos)

        # carry: [ energy ]
        def scanned_fun(carry, x):
            new_overlap = wave.calc_overlap(tuple(x), phonon_occ, parameters, lattice)
            ratio = jnp.exp(new_overlap - overlap)
            carry[0] -= ratio
            return carry, ratio

        [energy], hop_ratios = lax.scan(
            scanned_fun, [energy], lattice.get_nearest_neighbors(elec_pos)
        )

        # e_ph coupling
        e_ph_ratios = jnp.zeros((2,)) + 0.0j
        new_phonon_occ = phonon_occ.at[elec_pos].add(1)
        ratio_0 = (
            (jnp.sum(phonon_occ) < self.max_n_phonons)
            * jnp.exp(
                wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[elec_pos] + 1) ** 0.5
        )
        energy -= self.g * (phonon_occ[elec_pos] + 1) ** 0.5 * ratio_0
        e_ph_ratios = e_ph_ratios.at[0].set(ratio_0)

        new_phonon_occ = phonon_occ.at[elec_pos].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_1 = (phonon_occ[elec_pos]) ** 0.5 * jnp.exp(
            wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) - overlap
        )
        energy -= self.g * (phonon_occ[elec_pos]) ** 0.5 * ratio_1
        e_ph_ratios = e_ph_ratios.at[1].set(ratio_1)

        cumulative_ratios = jnp.cumsum(
            jnp.abs(jnp.concatenate((hop_ratios, e_ph_ratios)))
        )
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print('walker: {}', walker)
        # jax.debug.print('overlap: {}', overlap)
        # jax.debug.print('ratios: {}', jnp.concatenate((hop_ratios, e_ph_ratios)))
        # jax.debug.print('energy: {}', energy)
        # jax.debug.print('weight: {}', weight)
        # jax.debug.print('random_number: {}', random_number)
        # jax.debug.print('new_ind: {}', new_ind)

        # update
        walker[0] = jnp.array(walker[0])
        walker[0] = lax.cond(
            new_ind < lattice.coord_num,
            lambda w: tuple(nearest_neighbors[new_ind]),
            lambda w: elec_pos,
            0,
        )
        walker[1] = lax.cond(
            new_ind >= lattice.coord_num,
            lambda w: w.at[elec_pos].add(1 - 2 * ((new_ind - lattice.coord_num))),
            lambda w: w,
            walker[1],
        )

        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])
        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)
        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega, self.g, self.max_n_phonons))


@dataclass
class long_range:
    omega: float
    g: float
    zeta: float
    max_n_phonons: any = jnp.inf

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_pos = walker[0]
        phonon_occ = walker[1]

        overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(
            elec_pos, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )

        # diagonal
        energy = self.omega * jnp.sum(phonon_occ) + 0.0j

        # electron hops
        nearest_neighbors = lattice.get_nearest_neighbors(elec_pos)
        # carry: [ energy ]

        def scanned_fun(carry, x):
            new_overlap = wave.calc_overlap(tuple(x), phonon_occ, parameters, lattice)
            ratio = jnp.exp(new_overlap - overlap)
            carry[0] -= ratio
            return carry, ratio

        [energy], hop_ratios = lax.scan(
            scanned_fun, [energy], lattice.get_nearest_neighbors(elec_pos)
        )

        # e_ph coupling
        e_ph_ratios = jnp.zeros(2 * lattice.n_sites) + 0.0j

        # carry: [ energy, ratios, counter ]
        def scanned_fun(carry, x):
            pos = tuple(x)
            dist = lattice.get_distance(elec_pos, x)

            new_phonon_occ = phonon_occ.at[pos].add(1)
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (phonon_occ[pos] + 1) ** 0.5
            )
            carry[0] -= (
                self.g
                * jnp.exp(-dist / self.zeta)
                * (phonon_occ[pos] + 1) ** 0.5
                * ratio
                / (1 + dist**2) ** 1.5
            )
            carry[1] = carry[1].at[2 * carry[2]].set(ratio)

            new_phonon_occ = phonon_occ.at[pos].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[pos]) ** 0.5 * jnp.exp(
                wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[0] -= (
                self.g
                * jnp.exp(-dist / self.zeta)
                * (phonon_occ[pos]) ** 0.5
                * ratio
                / (1 + dist**2) ** 1.5
            )
            carry[1] = carry[1].at[2 * carry[2] + 1].set(ratio)

            carry[2] = carry[2] + 1
            return carry, x

        counter = 0
        [energy, e_ph_ratios, _], _ = lax.scan(
            scanned_fun, [energy, e_ph_ratios, counter], jnp.array(lattice.sites)
        )

        cumulative_ratios = jnp.cumsum(
            jnp.abs(jnp.concatenate((hop_ratios, e_ph_ratios)))
        )
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print('walker: {}', walker)
        # jax.debug.print('overlap: {}', overlap)
        # jax.debug.print('ratios: {}', jnp.concatenate((hop_ratios, e_ph_ratios)))
        # jax.debug.print('energy: {}', energy)
        # jax.debug.print('weight: {}', weight)
        # jax.debug.print('random_number: {}', random_number)
        # jax.debug.print('new_ind: {}', new_ind)

        # update
        walker[0] = jnp.array(walker[0])
        walker[0] = lax.cond(
            new_ind < lattice.coord_num,
            lambda w: tuple(nearest_neighbors[new_ind]),
            lambda w: elec_pos,
            0,
        )
        phonon_pos = tuple(jnp.array(lattice.sites)[(new_ind - lattice.coord_num) // 2])
        phonon_change = 1 - 2 * ((new_ind - lattice.coord_num) % 2)
        walker[1] = lax.cond(
            new_ind >= lattice.coord_num,
            lambda w: w.at[phonon_pos].add(phonon_change),
            lambda w: w,
            walker[1],
        )

        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])
        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)
        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega, self.g, self.zeta, self.max_n_phonons))


@dataclass
class bond_ssh:
    omega: float
    g: float
    max_n_phonons: any = jnp.inf

    @partial(jit, static_argnums=(0, 3, 4))
    def local_correlation_functions(self, walker, parameters, wave, lattice):
        elec_pos = walker[0]
        phonon_occ = walker[1]
        overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)

        # ke
        # carry: [ energy ]
        def scanned_fun(carry, x):
            new_overlap = wave.calc_overlap(tuple(x), phonon_occ, parameters, lattice)
            ratio = jnp.exp(new_overlap - overlap)
            carry[0] -= ratio
            return carry, ratio

        ke = 0.0j
        [ke], _ = lax.scan(scanned_fun, [ke], lattice.get_nearest_neighbors(elec_pos))

        # phonon_numbers
        phonon_numbers = phonon_occ[0].copy()

        # e_ph correlation
        # shift phonon_occ by elec_pos
        input_ar = phonon_occ.reshape(-1, *lattice.shape)
        for ax in range(len(lattice.shape)):
            for phonon_type in range(phonon_occ.shape[0]):
                input_ar = input_ar.at[phonon_type].set(
                    input_ar[phonon_type].take(
                        elec_pos[ax] + jnp.arange(lattice.shape[ax]),
                        axis=ax,
                        mode="wrap",
                    )
                )
        # shifted_phonon_occ = jnp.stack([*input_ar], axis=-1)
        shifted_phonon_occ = input_ar

        # calculate local phonon position at each site
        # carry: [ counter ]
        def scanned_fun(carry, x):
            pos = tuple(x)
            phonon_pos = 0.0j

            new_phonon_occ = shifted_phonon_occ.at[(0, *pos)].add(1)
            ratio = (
                (jnp.sum(shifted_phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (shifted_phonon_occ[(0, *pos)] + 1) ** 0.5
            )
            phonon_pos += (shifted_phonon_occ[(0, *pos)] + 1) ** 0.5 * ratio

            new_phonon_occ = shifted_phonon_occ.at[(0, *pos)].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (shifted_phonon_occ[(0, *pos)]) ** 0.5 * jnp.exp(
                wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            phonon_pos += (shifted_phonon_occ[(0, *pos)]) ** 0.5 * ratio

            carry[0] = carry[0] + 1
            return carry, phonon_pos

        counter = 0
        _, pos_corr = lax.scan(scanned_fun, [counter], jnp.array(lattice.sites))
        pos_corr = pos_corr.reshape(lattice.shape)

        average_phonon_pos = jnp.average(pos_corr)

        return ke, phonon_numbers, pos_corr, average_phonon_pos

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_pos = walker[0]
        phonon_occ = walker[1].reshape(len(lattice.shape), *lattice.shape)

        overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(
            elec_pos, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )

        # diagonal
        energy = self.omega * jnp.sum(phonon_occ) + 0.0j

        # electron hops bare and dressed
        nearest_neighbors = lattice.get_nearest_neighbors(elec_pos)
        neighboring_bonds = lattice.get_neighboring_bonds(elec_pos)

        # carry: [ energy ]
        def scanned_fun(carry, x):
            new_elec_pos = tuple(nearest_neighbors[x])
            # bare
            new_overlap = wave.calc_overlap(
                new_elec_pos, phonon_occ, parameters, lattice
            )
            ratio_0 = jnp.exp(new_overlap - overlap)
            carry[0] -= ratio_0

            bond = tuple(neighboring_bonds[x])
            # create phonon
            new_phonon_occ = phonon_occ.at[bond].add(1)
            ratio_1 = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (phonon_occ[bond] + 1) ** 0.5
            )
            carry[0] -= self.g * (phonon_occ[bond] + 1) ** 0.5 * ratio_1

            # destroy phonon
            new_phonon_occ = phonon_occ.at[bond].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio_2 = (phonon_occ[bond]) ** 0.5 * jnp.exp(
                wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[0] -= self.g * (phonon_occ[bond]) ** 0.5 * ratio_2

            return carry, (ratio_0, ratio_1, ratio_2)

        [energy], ratios = lax.scan(
            scanned_fun, [energy], jnp.arange(lattice.coord_num)
        )

        ratios = jnp.stack(ratios, axis=-1)
        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print('walker: {}', walker)
        # jax.debug.print('overlap: {}', overlap)
        # jax.debug.print('ratios: {}', ratios)
        # jax.debug.print('energy: {}', energy)
        # jax.debug.print('weight: {}', weight)
        # jax.debug.print('random_number: {}', random_number)
        # jax.debug.print('new_ind: {}', new_ind)

        # update
        new_elec_pos = tuple(nearest_neighbors[new_ind // 3])
        phonon_change = 1 - 2 * (new_ind % 3 - 1)
        bond = tuple(neighboring_bonds[new_ind // 3])
        walker[0] = new_elec_pos
        walker[1] = lax.cond(
            new_ind % 3 > 0,
            lambda w: w.at[bond].add(phonon_change),
            lambda w: w,
            phonon_occ,
        ).reshape(walker[1].shape)

        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])
        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)
        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega, self.g, self.max_n_phonons))


@dataclass
class ssh:
    omega: float
    g: float
    max_n_phonons: any = jnp.inf

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_pos = walker[0]
        phonon_occ = walker[1].reshape(len(lattice.shape), *lattice.shape)

        overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(
            elec_pos, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )

        # diagonal
        energy = self.omega * jnp.sum(phonon_occ) + 0.0j

        # electron hops bare and dressed
        nearest_neighbors = lattice.get_nearest_neighbor_modes(elec_pos)

        # carry: [ energy ]
        def scanned_fun(carry, x):
            new_elec_pos = tuple(nearest_neighbors[x][1:])
            dir = nearest_neighbors[x][0]

            # bare
            new_overlap = wave.calc_overlap(
                new_elec_pos, phonon_occ, parameters, lattice
            )
            ratio_0 = jnp.exp(new_overlap - overlap)
            carry[0] -= ratio_0

            hop_sign = jnp.array(lattice.hop_signs)[x]

            # create phonon on old site
            new_phonon_occ = phonon_occ.at[(dir, *elec_pos)].add(1)
            ratio_1 = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (phonon_occ[dir][elec_pos] + 1) ** 0.5
            )
            carry[0] += (
                self.g * hop_sign * (phonon_occ[dir][elec_pos] + 1) ** 0.5 * ratio_1
            )

            # destroy phonon on old site
            new_phonon_occ = phonon_occ.at[(dir, *elec_pos)].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio_2 = (phonon_occ[dir][elec_pos]) ** 0.5 * jnp.exp(
                wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[0] += self.g * hop_sign * (phonon_occ[dir][elec_pos]) ** 0.5 * ratio_2

            # create phonon on new site
            new_phonon_occ = phonon_occ.at[(dir, *new_elec_pos)].add(1)
            ratio_3 = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (phonon_occ[dir][new_elec_pos] + 1) ** 0.5
            )
            carry[0] -= (
                self.g * hop_sign * (phonon_occ[dir][new_elec_pos] + 1) ** 0.5 * ratio_3
            )

            # destroy phonon on new site
            new_phonon_occ = phonon_occ.at[(dir, *new_elec_pos)].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio_4 = (phonon_occ[dir][new_elec_pos]) ** 0.5 * jnp.exp(
                wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[0] -= (
                self.g * hop_sign * (phonon_occ[dir][new_elec_pos]) ** 0.5 * ratio_4
            )

            return carry, (ratio_0, ratio_1, ratio_2, ratio_3, ratio_4)

        [energy], ratios = lax.scan(
            scanned_fun, [energy], jnp.arange(lattice.coord_num)
        )

        # ratios = jnp.concatenate(ratios)
        ratios = jnp.stack(ratios, axis=-1)
        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # import jax
        # jax.debug.print("\nwalker: {}", walker)
        # jax.debug.print("overlap: {}", overlap)
        ## jax.debug.print('overlap_gradient: {}', overlap_gradient)
        # jax.debug.print("nearest_neighbors: {}", nearest_neighbors)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}\n", new_ind)

        # update
        new_elec_pos = tuple(nearest_neighbors[new_ind // 5][1:])
        dir = nearest_neighbors[new_ind // 5][0]
        phonon_change = 1 - 2 * (((new_ind % 5) - 1) % 2)
        phonon_pos = lax.cond(
            new_ind % 5 < 3,
            lambda w: (dir, *elec_pos),
            lambda w: (dir, *new_elec_pos),
            0,
        )
        walker[0] = new_elec_pos
        walker[1] = lax.cond(
            new_ind % 5 > 0,
            lambda w: w.at[phonon_pos].add(phonon_change),
            lambda w: w,
            phonon_occ,
        ).reshape(walker[1].shape)
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        # jax.debug.print('new_walker: {}\n', walker)

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega, self.g, self.max_n_phonons))


if __name__ == "__main__":
    import lattices
    import models
    import wavefunctions

    l_x, l_y, l_z = 3, 3, 3
    n_sites = l_x * l_y * l_z
    # n_bands = 2
    lattice = lattices.three_dimensional_grid(l_x, l_y, l_z)

    np.random.seed(0)
    parameters = jnp.array(np.random.rand(len(lattice.shell_distances)))
    wave = wavefunctions.merrifield(parameters.size)
    phonon_occ = jnp.array(
        [
            [[np.random.randint(2) for _ in range(l_x)] for _ in range(l_y)]
            for _ in range(l_z)
        ]
    )
    walker = [(0, 0, 0), phonon_occ]
    random_number = 0.9
    ham = holstein(1.0, 1.0)
    (
        energy,
        qp_weight,
        overlap_gradient,
        weight,
        walker,
        overlap,
    ) = ham.local_energy_and_update(walker, parameters, wave, lattice, random_number)
    print(energy)

    # model_r = models.MLP([5, 1])
    # model_phi = models.MLP([5, 1])
    # model_input = jnp.zeros((1 + n_bands)*n_sites)
    # nn_parameters_r = model_r.init(random.PRNGKey(0), model_input, mutable=True)
    # nn_parameters_phi = model_phi.init(random.PRNGKey(1), model_input, mutable=True)
    # n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters_r)) + sum(x.size for x in tree_util.tree_leaves(nn_parameters_phi))
    # parameters = [ nn_parameters_r, nn_parameters_phi ]
    # lattice_shape = (n_bands, *lattice.shape)
    # wave = wavefunctions.nn_complex_n(model_r.apply, model_phi.apply, n_nn_parameters, lattice_shape)
#
# walker = [ (0, (0,)), jnp.zeros(lattice.shape) ]
#
# omega_q = tuple(1. for _ in range(n_sites))
# e_n_k = (tuple(-2. * np.cos(2. * np.pi * k / n_sites) for k in range(n_sites)),
#         tuple(-2. * np.cos(2. * np.pi * k / n_sites) for k in range(n_sites)))
# g_mn_kq = ((tuple(tuple(1./n_sites**0.5 for _ in range(n_sites)) for _ in range(n_sites)),
#            tuple(tuple(1./n_sites**0.5 for _ in range(n_sites)) for _ in range(n_sites))),
#           (tuple(tuple(1./n_sites**0.5 for _ in range(n_sites)) for _ in range(n_sites)),
#            tuple(tuple(1./n_sites**0.5 for _ in range(n_sites)) for _ in range(n_sites))))
#
# ham = kq_ne(omega_q, e_n_k, g_mn_kq)
# random_number = 0.9
# energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker, parameters, wave, lattice, random_number)
