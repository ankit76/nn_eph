import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Sequence, Tuple

# os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, lax
from jax import numpy as jnp


@dataclass
class kq:
    """Bipolaron Hamiltonian for a single electron and phonon band in k-space.

    Attributes
    ----------
    omega_q : Sequence
        Phonon frequencies.
    e_k : Sequence
        Electronic energies.
    g_kq : Sequence
        Electron-phonon coupling.
    u : float
        On-site Coulomb repulsion.
    max_n_phonons : Any, optional
        Maximum number of phonons per site. The default is jnp.inf.
    """

    omega_q: Sequence
    e_k: Sequence
    g_kq: Sequence
    u: float
    max_n_phonons: Any = jnp.inf

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(
        self,
        walker: List,
        parameters: Any,
        wave: Any,
        lattice: Any,
        random_number: float,
    ) -> Tuple[float, float, Any, float, List, float]:
        """Calculate the local energy and update the walker.

        Parameters
        ----------
        walker : List
            Walker with electron momenta and phonon occupations as [List[Tuple], jnp.ndarray].
        parameters : Any
            Wave function parameters.
        wave : Any
            Wave function.
        lattice : Any
            Lattice.
        random_number : float
            Random number for mc update.

        Returns
        -------
        energy : float
            Local energy.
        qp_weight : float
            Quasiparticle weight.
        overlap_gradient : Any
            Overlap gradient.
        weight : float
            Weight.
        walker : List[List[tuple], jnp.ndarray]
            Updated walker.
        overlap : float
            Overlap.
        """

        elec_k = walker[0]
        phonon_occ = walker[1]
        n_sites = lattice.n_sites
        lattice_shape = jnp.array(lattice.shape)
        total_k = (jnp.array(elec_k[0]) + jnp.array(elec_k[1])) % lattice_shape

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
            + self.u / n_sites
            + jnp.array(self.e_k)[elec_k[0]]
            + jnp.array(self.e_k)[elec_k[1]]
            + 0.0j
        )
        # import jax

        # jax.debug.print("diagonal: {}", energy)

        ratios = jnp.zeros((5 * n_sites,)) + 0.0j

        # ee and eph scattering
        # carry: (energy, ratios, ham_elements)
        def scanned_fun(carry, kp):
            kp_i = lattice.get_site_num(kp)
            # ee
            kpp = (total_k - kp) % lattice_shape
            new_elec_k = [tuple(kp), tuple(kpp)]
            k_transfer = jnp.linalg.norm((kp - jnp.array(elec_k[0])) % lattice_shape)
            new_overlap = wave.calc_overlap(new_elec_k, phonon_occ, parameters, lattice)
            ratio = (k_transfer > 0.0) * jnp.exp(new_overlap - overlap)
            carry[0] += ratio * self.u / n_sites
            carry[1] = carry[1].at[5 * kp_i].set(ratio)
            carry[2] = carry[2].at[5 * kp_i].set(self.u / n_sites)

            # eph
            # elec_0
            qc_0 = tuple((jnp.array(elec_k[0]) - kp) % lattice_shape)
            qd_0 = tuple((kp - jnp.array(elec_k[0])) % lattice_shape)
            zero_q = 1.0 - (jnp.linalg.norm(jnp.array(qc_0)) == 0.0) / 2.0
            q_i = lattice.get_site_num(qc_0)
            new_elec_k = [tuple(kp), elec_k[1]]

            new_phonon_occ = phonon_occ.at[qc_0].add(1)
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (phonon_occ[qc_0] + 1) ** 0.5
            )
            carry[0] -= (
                ratio * jnp.array(self.g_kq)[kp_i, q_i] * (phonon_occ[qc_0] + 1) ** 0.5
            )
            carry[1] = carry[1].at[5 * kp_i + 1].set(ratio * zero_q)
            carry[2] = (
                carry[2]
                .at[5 * kp_i + 1]
                .set(jnp.array(self.g_kq)[kp_i, q_i] * (phonon_occ[qc_0] + 1) ** 0.5)
            )

            new_phonon_occ = phonon_occ.at[qd_0].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[qd_0]) ** 0.5 * jnp.exp(
                wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[0] -= (
                ratio * jnp.array(self.g_kq)[kp_i, q_i] * (phonon_occ[qd_0]) ** 0.5
            )
            carry[1] = carry[1].at[5 * kp_i + 2].set(ratio * zero_q)
            carry[2] = (
                carry[2]
                .at[5 * kp_i + 2]
                .set(jnp.array(self.g_kq)[kp_i, q_i] * (phonon_occ[qd_0]) ** 0.5)
            )

            # elec_1
            qc_1 = tuple((jnp.array(elec_k[1]) - kp) % lattice_shape)
            qd_1 = tuple((kp - jnp.array(elec_k[1])) % lattice_shape)
            zero_q = 1.0 - (jnp.linalg.norm(jnp.array(qc_1)) == 0.0) / 2.0
            q_i = lattice.get_site_num(qc_1)
            new_elec_k = [elec_k[0], tuple(kp)]

            new_phonon_occ = phonon_occ.at[qc_1].add(1)
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (phonon_occ[qc_1] + 1) ** 0.5
            )
            carry[0] -= (
                ratio * jnp.array(self.g_kq)[kp_i, q_i] * (phonon_occ[qc_1] + 1) ** 0.5
            )
            carry[1] = carry[1].at[5 * kp_i + 3].set(ratio * zero_q)
            carry[2] = (
                carry[2]
                .at[5 * kp_i + 3]
                .set(jnp.array(self.g_kq)[kp_i, q_i] * (phonon_occ[qc_1] + 1) ** 0.5)
            )

            new_phonon_occ = phonon_occ.at[qd_1].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[qd_1]) ** 0.5 * jnp.exp(
                wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[0] -= (
                ratio * jnp.array(self.g_kq)[kp_i, q_i] * (phonon_occ[qd_1]) ** 0.5
            )
            carry[1] = carry[1].at[5 * kp_i + 4].set(ratio * zero_q)
            carry[2] = (
                carry[2]
                .at[5 * kp_i + 4]
                .set(jnp.array(self.g_kq)[kp_i, q_i] * (phonon_occ[qd_1]) ** 0.5)
            )

            return carry, (qc_0, qd_0, qc_1, qd_1)

        ham_elements = 0.0 * ratios
        [energy, ratios, ham_elements], (qc_0, qd_0, qc_1, qd_1) = lax.scan(
            scanned_fun, [energy, ratios, ham_elements], jnp.array(lattice.sites)
        )

        qc_0 = jnp.stack(qc_0, axis=-1)
        qd_0 = jnp.stack(qd_0, axis=-1)
        qc_1 = jnp.stack(qc_1, axis=-1)
        qd_1 = jnp.stack(qd_1, axis=-1)

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # import jax
        #
        # jax.debug.print("\nwalker: {}", walker)
        # jax.debug.print("overlap: {}", overlap)
        # # jax.debug.print('overlap_gradient: {}', overlap_gradient)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}", new_ind)
        # jax.debug.print("ham_elements: {}", ham_elements)

        # ee
        new_elec_ee_0 = jnp.array(lattice.sites)[new_ind // 5]
        new_elec_ee_1 = (total_k - jnp.array(new_elec_ee_0)) % lattice_shape
        new_elec_eph = jnp.array(lattice.sites)[new_ind // 5]
        new_elec_0 = (
            (new_ind % 5 == 0) * new_elec_ee_0
            + ((new_ind % 5 == 1) + (new_ind % 5 == 2)) * new_elec_eph
            + (new_ind % 5 > 2) * jnp.array(elec_k[0])
        )
        new_elec_1 = (
            (new_ind % 5 == 0) * new_elec_ee_1
            + ((new_ind % 5 == 3) + (new_ind % 5 == 4)) * new_elec_eph
            + ((new_ind % 5 == 1) + (new_ind % 5 == 2)) * jnp.array(elec_k[1])
        )
        walker[0] = [tuple(new_elec_0), tuple(new_elec_1)]

        new_phonon_occ_0 = phonon_occ.at[tuple(qc_0[new_ind // 5])].add(1)
        new_phonon_occ_1 = phonon_occ.at[tuple(qd_0[new_ind // 5])].add(-1)
        new_phonon_occ_1 = jnp.where(new_phonon_occ_1 < 0, 0, new_phonon_occ_1)
        new_phonon_occ_2 = phonon_occ.at[tuple(qc_1[new_ind // 5])].add(1)
        new_phonon_occ_3 = phonon_occ.at[tuple(qd_1[new_ind // 5])].add(-1)
        new_phonon_occ_3 = jnp.where(new_phonon_occ_3 < 0, 0, new_phonon_occ_3)
        new_phonon_occ = (
            (new_ind % 5 == 0) * phonon_occ
            + (new_ind % 5 == 1) * new_phonon_occ_0
            + (new_ind % 5 == 2) * new_phonon_occ_1
            + (new_ind % 5 == 3) * new_phonon_occ_2
            + (new_ind % 5 == 4) * new_phonon_occ_3
        )
        walker[1] = new_phonon_occ
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

        # jax.debug.print("new_walker: {}", walker)

        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega_q, self.e_k, self.g_kq, self.u))


@dataclass
class kq_np:
    """Bipolaron Hamiltonian for a single electron and n phonon bands in k-space.

    Parameters
    ----------
    omega_nu_q : Sequence
        Phonon frequencies.
    e_k : Sequence
        Electronic energies.
    g_nu_kq : Sequence
        Electron-phonon coupling.
    u : float
        On-site Coulomb repulsion.
    max_n_phonons : Any, optional
        Maximum number of phonons per site. The default is jnp.inf.
    """

    omega_nu_q: Sequence
    e_k: Sequence
    g_nu_kq: Sequence
    u: float
    max_n_phonons: Any = jnp.inf

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_k = walker[0]
        phonon_occ = walker[1]
        n_p_bands = len(self.omega_nu_q)
        n_sites = lattice.n_sites
        lattice_shape = jnp.array(lattice.shape)
        total_k = (jnp.array(elec_k[0]) + jnp.array(elec_k[1])) % lattice_shape

        overlap = wave.calc_overlap(elec_k, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(
            elec_k, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )

        # diagonal
        energy = (
            jnp.sum(jnp.array(self.omega_nu_q) * phonon_occ)
            + self.u / n_sites
            + jnp.array(self.e_k)[elec_k[0]]
            + jnp.array(self.e_k)[elec_k[1]]
            + 0.0j
        )

        # loop over phonon bands
        # carry: (energy, ratios, kp, qc_0, qd_0, qc_1, qd_1)
        def inner_scanned_fun(carry, nu):
            kp, qc_0, qd_0, qc_1, qd_1 = (
                carry[2],
                carry[3],
                carry[4],
                carry[5],
                carry[6],
            )
            kp_i = lattice.get_site_num(kp)
            # elec_0
            q_i = lattice.get_site_num(qc_0)
            zero_q = 1.0 - (jnp.linalg.norm(jnp.array(qc_0)) == 0.0) / 2.0
            new_elec_k = [tuple(kp), elec_k[1]]

            new_phonon_occ = phonon_occ.at[(nu, *qc_0)].add(1)
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (phonon_occ[(nu, *qc_0)] + 1) ** 0.5
            )
            carry[0] -= (
                ratio
                * jnp.array(self.g_nu_kq)[nu, kp_i, q_i]
                * (phonon_occ[(nu, *qc_0)] + 1) ** 0.5
            )
            carry[1] = carry[1].at[4 * nu].set(ratio * zero_q)

            new_phonon_occ = phonon_occ.at[(nu, *qd_0)].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[(nu, *qd_0)]) ** 0.5 * jnp.exp(
                wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[0] -= (
                ratio
                * jnp.array(self.g_nu_kq)[nu, kp_i, q_i]
                * (phonon_occ[(nu, *qd_0)]) ** 0.5
            )
            carry[1] = carry[1].at[4 * nu + 1].set(ratio * zero_q)

            # elec_1
            q_i = lattice.get_site_num(qc_1)
            zero_q = 1.0 - (jnp.linalg.norm(jnp.array(qc_1)) == 0.0) / 2.0
            new_elec_k = [elec_k[0], tuple(kp)]

            new_phonon_occ = phonon_occ.at[(nu, *qc_1)].add(1)
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (phonon_occ[(nu, *qc_1)] + 1) ** 0.5
            )
            carry[0] -= (
                ratio
                * jnp.array(self.g_nu_kq)[nu, kp_i, q_i]
                * (phonon_occ[(nu, *qc_1)] + 1) ** 0.5
            )
            carry[1] = carry[1].at[4 * nu + 2].set(ratio * zero_q)

            new_phonon_occ = phonon_occ.at[(nu, *qd_1)].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[(nu, *qd_1)]) ** 0.5 * jnp.exp(
                wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[0] -= (
                ratio
                * jnp.array(self.g_nu_kq)[nu, kp_i, q_i]
                * (phonon_occ[(nu, *qd_1)]) ** 0.5
            )
            carry[1] = carry[1].at[4 * nu + 3].set(ratio * zero_q)

            return carry, nu

        # ee and eph scattering
        # carry: (energy, ratios)
        def scanned_fun(carry, kp):
            kp_i = lattice.get_site_num(kp)
            # ee
            kpp = (total_k - kp) % lattice_shape
            new_elec_k = [tuple(kp), tuple(kpp)]
            k_transfer = jnp.linalg.norm((kp - jnp.array(elec_k[0])) % lattice_shape)
            new_overlap = wave.calc_overlap(new_elec_k, phonon_occ, parameters, lattice)
            ratio = (k_transfer > 0.0) * jnp.exp(new_overlap - overlap)
            carry[0] += ratio * self.u / n_sites
            carry[1] = carry[1].at[kp_i].set(ratio)

            # eph
            eph_ratios = jnp.zeros((4 * n_p_bands,)) + 0.0j
            qc_0 = tuple((jnp.array(elec_k[0]) - kp) % lattice_shape)
            qd_0 = tuple((kp - jnp.array(elec_k[0])) % lattice_shape)
            qc_1 = tuple((jnp.array(elec_k[1]) - kp) % lattice_shape)
            qd_1 = tuple((kp - jnp.array(elec_k[1])) % lattice_shape)
            [carry[0], eph_ratios, _, _, _, _, _], _ = lax.scan(
                inner_scanned_fun,
                [carry[0], eph_ratios, kp, qc_0, qd_0, qc_1, qd_1],
                jnp.arange(n_p_bands),
            )
            # carry[1] = (
            #     carry[1]
            #     .at[(1 + 4 * n_p_bands) * kp_i + 1 : (1 + 4 * n_p_bands) * (kp_i + 1)]
            #     .set(eph_ratios)
            # )

            return carry, (qc_0, qd_0, qc_1, qd_1, eph_ratios)

        ee_ratios = jnp.zeros((n_sites,)) + 0.0j
        [energy, ee_ratios], (qc_0, qd_0, qc_1, qd_1, eph_ratios) = lax.scan(
            scanned_fun, [energy, ee_ratios], jnp.array(lattice.sites)
        )

        qc_0 = jnp.stack(qc_0, axis=-1)
        qd_0 = jnp.stack(qd_0, axis=-1)
        qc_1 = jnp.stack(qc_1, axis=-1)
        qd_1 = jnp.stack(qd_1, axis=-1)
        eph_ratios = jnp.stack(eph_ratios, axis=0).reshape(-1)
        ratios = jnp.concatenate([ee_ratios, eph_ratios])

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # import jax
        #
        # jax.debug.print("\nwalker: {}", walker)
        # jax.debug.print("overlap: {}", overlap)
        # # jax.debug.print('overlap_gradient: {}', overlap_gradient)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}", new_ind)
        # # jax.debug.print("qc_0: {}", qc_0)
        # # jax.debug.print("qd_0: {}", qd_0)
        # # jax.debug.print("qc_1: {}", qc_1)
        # # jax.debug.print("qd_1: {}", qd_1)

        # ee
        ee = new_ind < n_sites
        eph = new_ind >= n_sites
        new_ind_eph = new_ind - n_sites
        cd_index = (new_ind_eph % (4 * n_p_bands)) % 4
        new_elec_ee_0 = jnp.array(lattice.sites)[new_ind % n_sites]
        new_elec_ee_1 = (total_k - jnp.array(new_elec_ee_0)) % lattice_shape
        new_elec_eph = jnp.array(lattice.sites)[new_ind_eph // (4 * n_p_bands)]
        new_elec_0 = (
            ee * new_elec_ee_0
            + eph * ((cd_index == 0) + (cd_index == 1)) * new_elec_eph
            + eph * (cd_index > 1) * jnp.array(elec_k[0])
        )
        new_elec_1 = (
            ee * new_elec_ee_1
            + eph * ((cd_index == 2) + (cd_index == 3)) * new_elec_eph
            + eph * (cd_index < 2) * jnp.array(elec_k[1])
        )
        walker[0] = [tuple(new_elec_0), tuple(new_elec_1)]

        ph_k = new_ind_eph // (4 * n_p_bands)
        ph_band = (new_ind_eph % (4 * n_p_bands)) // 4
        new_phonon_occ_0 = phonon_occ.at[(ph_band, *qc_0[ph_k])].add(1)
        new_phonon_occ_1 = phonon_occ.at[(ph_band, *qd_0[ph_k])].add(-1)
        new_phonon_occ_1 = jnp.where(new_phonon_occ_1 < 0, 0, new_phonon_occ_1)
        new_phonon_occ_2 = phonon_occ.at[(ph_band, *qc_1[ph_k])].add(1)
        new_phonon_occ_3 = phonon_occ.at[(ph_band, *qd_1[ph_k])].add(-1)
        new_phonon_occ_3 = jnp.where(new_phonon_occ_3 < 0, 0, new_phonon_occ_3)
        new_phonon_occ = (
            ee * phonon_occ
            + eph * (cd_index == 0) * new_phonon_occ_0
            + eph * (cd_index == 1) * new_phonon_occ_1
            + eph * (cd_index == 2) * new_phonon_occ_2
            + eph * (cd_index == 3) * new_phonon_occ_3
        )
        walker[1] = new_phonon_occ
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

        # jax.debug.print("ee: {}", ee)
        # jax.debug.print("eph: {}", eph)
        # jax.debug.print("ph_band: {}", ph_band)
        # jax.debug.print("ph_k: {}", ph_k)
        # jax.debug.print("cd_index: {}", cd_index)
        # jax.debug.print("new_walker: {}", walker)

        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega_nu_q, self.e_k, self.g_nu_kq, self.u))


@dataclass
class holstein_1d:
    omega: float
    g: float
    u: float
    triplet: bool = False

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_pos = walker[0]
        phonon_occ = walker[1]
        n_sites = phonon_occ.shape[0]

        overlap = wave.calc_overlap_map(elec_pos, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_map_gradient(
            elec_pos, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )
        # jax.debug.print('\nwalker: {}', walker)
        # jax.debug.print('overlap: {}', overlap)

        # diagonal
        energy = (
            self.omega * jnp.sum(phonon_occ)
            + self.u * (elec_pos[0][0] == elec_pos[1][0])
            + 0.0j
        )

        # electron hops
        new_elec_pos = [((elec_pos[0][0] + 1) % n_sites,), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_0 = lax.cond(
            n_sites > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        )
        energy -= ratio_0

        new_elec_pos = [((elec_pos[0][0] - 1) % n_sites,), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_1 = lax.cond(
            n_sites > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        )
        energy -= ratio_1

        new_elec_pos = [elec_pos[0], ((elec_pos[1][0] + 1) % n_sites,)]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_2 = lax.cond(
            n_sites > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        )
        energy -= ratio_2

        new_elec_pos = [elec_pos[0], ((elec_pos[1][0] - 1) % n_sites,)]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_3 = lax.cond(
            n_sites > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        )
        energy -= ratio_3

        # e_ph coupling
        new_phonon_occ = phonon_occ.at[elec_pos[0]].add(1)
        ratio_4 = (
            jnp.exp(
                wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[elec_pos[0]] + 1) ** 0.5
        )
        energy -= self.g * (phonon_occ[elec_pos[0]] + 1) ** 0.5 * ratio_4

        new_phonon_occ = phonon_occ.at[elec_pos[0]].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_5 = (phonon_occ[elec_pos[0]]) ** 0.5 * jnp.exp(
            wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice)
            - overlap
        )
        energy -= self.g * (phonon_occ[elec_pos[0]]) ** 0.5 * ratio_5

        new_phonon_occ = phonon_occ.at[elec_pos[1]].add(1)
        ratio_6 = (
            jnp.exp(
                wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[elec_pos[1]] + 1) ** 0.5
        )
        energy -= self.g * (phonon_occ[elec_pos[1]] + 1) ** 0.5 * ratio_6

        new_phonon_occ = phonon_occ.at[elec_pos[1]].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_7 = (phonon_occ[elec_pos[1]]) ** 0.5 * jnp.exp(
            wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice)
            - overlap
        )
        energy -= self.g * (phonon_occ[elec_pos[1]]) ** 0.5 * ratio_7

        cumulative_ratios = jnp.cumsum(
            jnp.abs(
                jnp.array(
                    [
                        ratio_0,
                        ratio_1,
                        ratio_2,
                        ratio_3,
                        ratio_4,
                        ratio_5,
                        ratio_6,
                        ratio_7,
                    ]
                )
            )
        )
        # jax.debug.print('ratios: {}', jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3, ratio_4, ratio_5, ratio_6, ratio_7 ]))
        # jax.debug.print('energy: {}', energy)
        weight = 1 / cumulative_ratios[-1]
        # jax.debug.print('weight: {}', weight)
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )
        # jax.debug.print('random_number: {}', random_number)
        # jax.debug.print('new_ind: {}', new_ind)

        walker_copy = jnp.array(walker[0])
        elec_pos = jnp.array(elec_pos)
        ind_0 = jnp.array([1, -1, 1, -1])[new_ind]
        walker_copy = lax.cond(
            new_ind < 4,
            lambda x: x.at[new_ind // 2, 0].set(
                (elec_pos[new_ind // 2][0] + ind_0) % n_sites
            ),
            lambda x: x,
            walker_copy,
        )
        walker[0] = [(walker_copy[0][0],), (walker_copy[1][0],)]

        walker[1] = lax.cond(
            new_ind > 5,
            lambda x: x.at[elec_pos[1]].add(-2 * new_ind + 13),
            lambda x: x,
            walker[1],
        )
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega, self.g, self.triplet))


@dataclass
class long_range_1d:
    omega: float
    g: float
    zeta: float
    u: float
    triplet: bool = False

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_pos = walker[0]
        phonon_occ = walker[1]
        n_sites = lattice.shape[0]

        overlap = wave.calc_overlap_map(elec_pos, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_map_gradient(
            elec_pos, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )
        # jax.debug.print('\nwalker: {}', walker)
        # jax.debug.print('overlap: {}', overlap)

        # diagonal
        energy = (
            self.omega * jnp.sum(phonon_occ)
            + self.u * (elec_pos[0][0] == elec_pos[1][0])
            + 0.0j
        )
        # 2 e hop terms, n_sites phonon terms
        ratios = jnp.zeros((4 + 2 * n_sites,)) + 0.0j

        # electron hops
        new_elec_pos = [((elec_pos[0][0] + 1) % n_sites,), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_0 = lax.cond(
            n_sites > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        )
        energy -= ratio_0
        ratios = ratios.at[0].set(ratio_0)

        new_elec_pos = [((elec_pos[0][0] - 1) % n_sites,), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_1 = lax.cond(
            n_sites > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        )
        energy -= ratio_1
        ratios = ratios.at[1].set(ratio_1)

        new_elec_pos = [elec_pos[0], ((elec_pos[1][0] + 1) % n_sites,)]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_2 = lax.cond(
            n_sites > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        )
        energy -= ratio_2
        ratios = ratios.at[2].set(ratio_2)

        new_elec_pos = [elec_pos[0], ((elec_pos[1][0] - 1) % n_sites,)]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_3 = lax.cond(
            n_sites > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        )
        energy -= ratio_3
        ratios = ratios.at[3].set(ratio_3)

        # e_ph coupling
        def scanned_fun(carry, x):
            pos = x
            dist_0 = jnp.min(
                jnp.array(
                    [(x - elec_pos[0][0]) % n_sites, (elec_pos[0][0] - x) % n_sites]
                )
            )
            dist_1 = jnp.min(
                jnp.array(
                    [(x - elec_pos[1][0]) % n_sites, (elec_pos[1][0] - x) % n_sites]
                )
            )

            new_phonon_occ = phonon_occ.at[pos].add(1)
            ratio = (
                jnp.exp(
                    wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (phonon_occ[pos] + 1) ** 0.5
            )
            carry[0] -= (
                self.g
                * jnp.exp(-dist_0 / self.zeta)
                * (phonon_occ[pos] + 1) ** 0.5
                * ratio
                / (1 + dist_0**2) ** 1.5
            )
            carry[0] -= (
                self.g
                * jnp.exp(-dist_1 / self.zeta)
                * (phonon_occ[pos] + 1) ** 0.5
                * ratio
                / (1 + dist_1**2) ** 1.5
            )
            carry[1] = carry[1].at[2 * x + 4].set(ratio)

            new_phonon_occ = phonon_occ.at[pos].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[pos]) ** 0.5 * jnp.exp(
                wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[0] -= (
                self.g
                * jnp.exp(-dist_0 / self.zeta)
                * (phonon_occ[pos]) ** 0.5
                * ratio
                / (1 + dist_0**2) ** 1.5
            )
            carry[0] -= (
                self.g
                * jnp.exp(-dist_1 / self.zeta)
                * (phonon_occ[pos]) ** 0.5
                * ratio
                / (1 + dist_1**2) ** 1.5
            )
            carry[1] = carry[1].at[2 * x + 5].set(ratio)

            return carry, x

        [energy, ratios], _ = lax.scan(
            scanned_fun, [energy, ratios], jnp.arange(n_sites)
        )

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))

        # jax.debug.print('ratios: {}', jnp.array(ratios))
        # jax.debug.print('energy: {}', energy)
        weight = 1 / cumulative_ratios[-1]
        # jax.debug.print('weight: {}', weight)
        # jax.debug.print('random_number: {}', random_number)
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )
        # jax.debug.print('new_ind: {}', new_ind)

        walker_copy = jnp.array(walker[0])
        elec_pos = jnp.array(elec_pos)
        ind_0 = jnp.array([1, -1, 1, -1])[new_ind]
        walker_copy = lax.cond(
            new_ind < 4,
            lambda x: x.at[new_ind // 2, 0].set(
                (elec_pos[new_ind // 2][0] + ind_0) % n_sites
            ),
            lambda x: x,
            walker_copy,
        )
        walker[0] = [(walker_copy[0][0],), (walker_copy[1][0],)]

        walker[1] = lax.cond(
            new_ind > 3,
            lambda x: x.at[(new_ind - 4) // 2].add(1 - 2 * ((new_ind - 4) % 2)),
            lambda x: x,
            walker[1],
        )
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega, self.g, self.zeta, self.triplet))


@dataclass
class ssh:
    omega: float
    g: float
    u: float
    max_n_phonons: any = jnp.inf

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_pos = walker[0]
        phonon_occ = walker[1].reshape(len(lattice.shape), *lattice.shape)

        overlap = wave.calc_overlap_map(elec_pos, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_map_gradient(
            elec_pos, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )

        # diagonal
        energy = (
            self.omega * jnp.sum(phonon_occ)
            + self.u * jnp.product(jnp.array(elec_pos[0]) == jnp.array(elec_pos[1]))
            + 0.0j
        )

        # electron hops bare and dressed
        nearest_neighbors_0 = lattice.get_nearest_neighbor_modes(elec_pos[0])
        nearest_neighbors_1 = lattice.get_nearest_neighbor_modes(elec_pos[1])

        # carry: [ energy, ratios_0, ratios_1 ]
        def scanned_fun(carry, x):
            hop_sign = jnp.array(lattice.hop_signs)[x]

            # elec 0
            new_elec_pos = [tuple(nearest_neighbors_0[x][1:]), elec_pos[1]]
            dir = nearest_neighbors_0[x][0]

            # bare
            new_overlap = wave.calc_overlap_map(
                new_elec_pos, phonon_occ, parameters, lattice
            )
            ratio = jnp.exp(new_overlap - overlap)
            carry[1] = carry[1].at[x * 5].set(ratio)
            carry[0] -= ratio

            # create phonon on old site
            new_phonon_occ = phonon_occ.at[(dir, *elec_pos[0])].add(1)
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap_map(
                        new_elec_pos, new_phonon_occ, parameters, lattice
                    )
                    - overlap
                )
                / (phonon_occ[(dir, *elec_pos[0])] + 1) ** 0.5
            )
            carry[1] = carry[1].at[x * 5 + 1].set(ratio)
            carry[0] += (
                self.g * hop_sign * (phonon_occ[(dir, *elec_pos[0])] + 1) ** 0.5 * ratio
            )

            # destroy phonon on old site
            new_phonon_occ = phonon_occ.at[(dir, *elec_pos[0])].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[(dir, *elec_pos[0])]) ** 0.5 * jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[1] = carry[1].at[x * 5 + 2].set(ratio)
            carry[0] += (
                self.g * hop_sign * (phonon_occ[(dir, *elec_pos[0])]) ** 0.5 * ratio
            )

            # create phonon on new site
            new_phonon_occ = phonon_occ.at[(dir, *new_elec_pos[0])].add(1)
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap_map(
                        new_elec_pos, new_phonon_occ, parameters, lattice
                    )
                    - overlap
                )
                / (phonon_occ[(dir, *new_elec_pos[0])] + 1) ** 0.5
            )
            carry[1] = carry[1].at[x * 5 + 3].set(ratio)
            carry[0] -= (
                self.g
                * hop_sign
                * (phonon_occ[(dir, *new_elec_pos[0])] + 1) ** 0.5
                * ratio
            )

            # destroy phonon on new site
            new_phonon_occ = phonon_occ.at[(dir, *new_elec_pos[0])].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[(dir, *new_elec_pos[0])]) ** 0.5 * jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[1] = carry[1].at[x * 5 + 4].set(ratio)
            carry[0] -= (
                self.g * hop_sign * (phonon_occ[(dir, *new_elec_pos[0])]) ** 0.5 * ratio
            )

            # elec 1
            new_elec_pos = [elec_pos[0], tuple(nearest_neighbors_1[x][1:])]
            dir = nearest_neighbors_1[x][0]

            # bare
            new_overlap = wave.calc_overlap_map(
                new_elec_pos, phonon_occ, parameters, lattice
            )
            ratio = jnp.exp(new_overlap - overlap)
            carry[2] = carry[2].at[x * 5].set(ratio)
            carry[0] -= ratio

            # create phonon on old site
            new_phonon_occ = phonon_occ.at[(dir, *elec_pos[1])].add(1)
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap_map(
                        new_elec_pos, new_phonon_occ, parameters, lattice
                    )
                    - overlap
                )
                / (phonon_occ[(dir, *elec_pos[1])] + 1) ** 0.5
            )
            carry[2] = carry[2].at[x * 5 + 1].set(ratio)
            carry[0] += (
                self.g * hop_sign * (phonon_occ[(dir, *elec_pos[1])] + 1) ** 0.5 * ratio
            )

            # destroy phonon on old site
            new_phonon_occ = phonon_occ.at[(dir, *elec_pos[1])].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[(dir, *elec_pos[1])]) ** 0.5 * jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[2] = carry[2].at[x * 5 + 2].set(ratio)
            carry[0] += (
                self.g * hop_sign * (phonon_occ[(dir, *elec_pos[1])]) ** 0.5 * ratio
            )

            # create phonon on new site
            new_phonon_occ = phonon_occ.at[(dir, *new_elec_pos[1])].add(1)
            ratio = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(
                    wave.calc_overlap_map(
                        new_elec_pos, new_phonon_occ, parameters, lattice
                    )
                    - overlap
                )
                / (phonon_occ[(dir, *new_elec_pos[1])] + 1) ** 0.5
            )
            carry[2] = carry[2].at[x * 5 + 3].set(ratio)
            carry[0] -= (
                self.g
                * hop_sign
                * (phonon_occ[(dir, *new_elec_pos[1])] + 1) ** 0.5
                * ratio
            )

            # destroy phonon on new site
            new_phonon_occ = phonon_occ.at[(dir, *new_elec_pos[1])].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[(dir, *new_elec_pos[1])]) ** 0.5 * jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[2] = carry[2].at[x * 5 + 4].set(ratio)
            carry[0] -= (
                self.g * hop_sign * (phonon_occ[(dir, *new_elec_pos[1])]) ** 0.5 * ratio
            )

            return carry, x

        ratios_0 = jnp.zeros((5 * lattice.coord_num,)) + 0.0j
        ratios_1 = jnp.zeros((5 * lattice.coord_num,)) + 0.0j
        [energy, ratios_0, ratios_1], _ = lax.scan(
            scanned_fun, [energy, ratios_0, ratios_1], jnp.arange(lattice.coord_num)
        )

        ratios = jnp.concatenate([ratios_0, ratios_1])
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
        elec_no = new_ind // (5 * lattice.coord_num)
        new_ind_elec = new_ind % (5 * lattice.coord_num)
        new_elec_pos = tuple(
            jnp.array([nearest_neighbors_0, nearest_neighbors_1])[elec_no][
                new_ind_elec // 5
            ][1:]
        )
        dir = jnp.array([nearest_neighbors_0, nearest_neighbors_1])[elec_no][
            new_ind_elec // 5
        ][0]
        walker[0] = lax.cond(
            elec_no == 0,
            lambda x: [new_elec_pos, elec_pos[1]],
            lambda x: [elec_pos[0], new_elec_pos],
            0.0,
        )
        phonon_change = 1 - 2 * (((new_ind_elec % 5) - 1) % 2)
        phonon_pos = lax.cond(
            new_ind_elec % 5 < 3,
            lambda w: (dir, *jnp.array(elec_pos)[elec_no]),
            lambda w: (dir, *new_elec_pos),
            0,
        )
        walker[1] = lax.cond(
            new_ind_elec % 5 > 0,
            lambda w: w.at[phonon_pos].add(phonon_change),
            lambda w: w,
            phonon_occ,
        ).reshape(walker[1].shape)
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])
        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        # jax.debug.print('new_walker: {}', walker)

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega, self.g, self.u, self.max_n_phonons))


@dataclass
class bond_ssh_2d:
    omega: float
    g: float
    u: float

    def tup_eq(self, pos0, pos1):
        return (pos0[0] == pos1[0]) * (pos0[1] == pos1[1])

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_pos = walker[0]
        phonon_occ = walker[1]
        l_y, l_x = lattice.shape

        overlap = wave.calc_overlap_map(elec_pos, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_map_gradient(
            elec_pos, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )
        # jax.debug.print('\nwalker:\n{}\n', walker)
        # jax.debug.print('overlap: {}', overlap)

        # diagonal
        energy = (
            self.omega * jnp.sum(phonon_occ)
            + self.u * self.tup_eq(elec_pos[0], elec_pos[1])
            + 0.0j
        )

        # electron hops
        # electron 0
        # right
        # bare
        new_elec_pos = [(elec_pos[0][0], (elec_pos[0][1] + 1) % l_x), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_0_x = lax.cond(
            l_x > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0, new_overlap
        )
        energy -= ratio_0_x

        bond_pos_0_x = (1, elec_pos[0][0], elec_pos[0][1] * (1 - (l_x == 2)))
        # create phonon
        new_phonon_occ = phonon_occ.at[bond_pos_0_x].add(1)
        ratio_0_x_cp = (
            jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[bond_pos_0_x] + 1) ** 0.5
        )
        energy -= self.g * (phonon_occ[bond_pos_0_x] + 1) ** 0.5 * ratio_0_x_cp

        # destroy phonon
        new_phonon_occ = phonon_occ.at[bond_pos_0_x].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_0_x_dp = (phonon_occ[bond_pos_0_x]) ** 0.5 * jnp.exp(
            wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
            - overlap
        )
        energy -= self.g * (phonon_occ[bond_pos_0_x]) ** 0.5 * ratio_0_x_dp

        # left
        # bare
        new_elec_pos = [(elec_pos[0][0], (elec_pos[0][1] - 1) % l_x), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_1_x = lax.cond(
            l_x > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0, new_overlap
        )
        energy -= ratio_1_x

        bond_pos_1_x = (1, elec_pos[0][0], new_elec_pos[0][1] * (1 - (l_x == 2)))
        # create phonon
        new_phonon_occ = phonon_occ.at[bond_pos_1_x].add(1)
        ratio_1_x_cp = lax.cond(
            l_x > 2,
            lambda x: jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[bond_pos_1_x] + 1) ** 0.5,
            lambda x: 0.0,
            0.0,
        )
        energy -= self.g * (phonon_occ[bond_pos_1_x] + 1) ** 0.5 * ratio_1_x_cp

        # destroy phonon
        new_phonon_occ = phonon_occ.at[bond_pos_1_x].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_1_x_dp = lax.cond(
            l_x > 2,
            lambda x: (phonon_occ[bond_pos_1_x]) ** 0.5
            * jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            ),
            lambda x: 0.0,
            0.0,
        )
        energy -= self.g * (phonon_occ[bond_pos_1_x]) ** 0.5 * ratio_1_x_dp

        # up
        # bare
        new_elec_pos = [((elec_pos[0][0] - 1) % l_y, elec_pos[0][1]), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_0_y = lax.cond(
            l_y > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0, new_overlap
        )
        energy -= ratio_0_y

        bond_pos_0_y = (0, new_elec_pos[0][0] * (1 - (l_y == 2)), elec_pos[0][1])
        # create phonon
        new_phonon_occ = phonon_occ.at[bond_pos_0_y].add(1)
        ratio_0_y_cp = (
            jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[bond_pos_0_y] + 1) ** 0.5
        )
        energy -= self.g * (phonon_occ[bond_pos_0_y] + 1) ** 0.5 * ratio_0_y_cp

        # destroy phonon
        new_phonon_occ = phonon_occ.at[bond_pos_0_y].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_0_y_dp = (phonon_occ[bond_pos_0_y]) ** 0.5 * jnp.exp(
            wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
            - overlap
        )
        energy -= self.g * (phonon_occ[bond_pos_0_y]) ** 0.5 * ratio_0_y_dp

        # down
        # bare
        new_elec_pos = [((elec_pos[0][0] + 1) % l_y, elec_pos[0][1]), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_1_y = lax.cond(
            l_y > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0, new_overlap
        )
        energy -= ratio_1_y

        bond_pos_1_y = (0, elec_pos[0][0] * (1 - (l_y == 2)), elec_pos[0][1])
        # create phonon
        new_phonon_occ = phonon_occ.at[bond_pos_1_y].add(1)
        ratio_1_y_cp = lax.cond(
            l_y > 2,
            lambda x: jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[bond_pos_1_y] + 1) ** 0.5,
            lambda x: 0.0,
            0.0,
        )
        energy -= self.g * (phonon_occ[bond_pos_1_y] + 1) ** 0.5 * ratio_1_y_cp

        # destroy phonon
        new_phonon_occ = phonon_occ.at[bond_pos_1_y].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_1_y_dp = lax.cond(
            l_y > 2,
            lambda x: (phonon_occ[bond_pos_1_y]) ** 0.5
            * jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            ),
            lambda x: 0.0,
            0.0,
        )
        energy -= self.g * (phonon_occ[bond_pos_1_y]) ** 0.5 * ratio_1_y_dp

        ratios_0 = [
            ratio_0_x,
            ratio_0_x_cp,
            ratio_0_x_dp,
            ratio_1_x,
            ratio_1_x_cp,
            ratio_1_x_dp,
            ratio_0_y,
            ratio_0_y_cp,
            ratio_0_y_dp,
            ratio_1_y,
            ratio_1_y_cp,
            ratio_1_y_dp,
        ]
        bond_pos_0 = [bond_pos_0_x, bond_pos_1_x, bond_pos_0_y, bond_pos_1_y]

        # electron 1
        # right
        # bare
        new_elec_pos = [elec_pos[0], (elec_pos[1][0], (elec_pos[1][1] + 1) % l_x)]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_0_x = lax.cond(
            l_x > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0, new_overlap
        )
        energy -= ratio_0_x

        bond_pos_0_x = (1, elec_pos[1][0], elec_pos[1][1] * (1 - (l_x == 2)))
        # create phonon
        new_phonon_occ = phonon_occ.at[bond_pos_0_x].add(1)
        ratio_0_x_cp = (
            jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[bond_pos_0_x] + 1) ** 0.5
        )
        energy -= self.g * (phonon_occ[bond_pos_0_x] + 1) ** 0.5 * ratio_0_x_cp

        # destroy phonon
        new_phonon_occ = phonon_occ.at[bond_pos_0_x].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_0_x_dp = (phonon_occ[bond_pos_0_x]) ** 0.5 * jnp.exp(
            wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
            - overlap
        )
        energy -= self.g * (phonon_occ[bond_pos_0_x]) ** 0.5 * ratio_0_x_dp

        # left
        # bare
        new_elec_pos = [elec_pos[0], (elec_pos[1][0], (elec_pos[1][1] - 1) % l_x)]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_1_x = lax.cond(
            l_x > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0, new_overlap
        )
        energy -= ratio_1_x

        bond_pos_1_x = (1, elec_pos[1][0], new_elec_pos[1][1] * (1 - (l_x == 2)))
        # create phonon
        new_phonon_occ = phonon_occ.at[bond_pos_1_x].add(1)
        ratio_1_x_cp = lax.cond(
            l_x > 2,
            lambda x: jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[bond_pos_1_x] + 1) ** 0.5,
            lambda x: 0.0,
            0.0,
        )
        energy -= self.g * (phonon_occ[bond_pos_1_x] + 1) ** 0.5 * ratio_1_x_cp

        # destroy phonon
        new_phonon_occ = phonon_occ.at[bond_pos_1_x].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_1_x_dp = lax.cond(
            l_x > 2,
            lambda x: (phonon_occ[bond_pos_1_x]) ** 0.5
            * jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            ),
            lambda x: 0.0,
            0.0,
        )
        energy -= self.g * (phonon_occ[bond_pos_1_x]) ** 0.5 * ratio_1_x_dp

        # up
        # bare
        new_elec_pos = [elec_pos[0], ((elec_pos[1][0] - 1) % l_y, elec_pos[1][1])]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_0_y = lax.cond(
            l_y > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0, new_overlap
        )
        energy -= ratio_0_y

        bond_pos_0_y = (0, new_elec_pos[1][0] * (1 - (l_y == 2)), elec_pos[1][1])
        # create phonon
        new_phonon_occ = phonon_occ.at[bond_pos_0_y].add(1)
        ratio_0_y_cp = (
            jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[bond_pos_0_y] + 1) ** 0.5
        )
        energy -= self.g * (phonon_occ[bond_pos_0_y] + 1) ** 0.5 * ratio_0_y_cp

        # destroy phonon
        new_phonon_occ = phonon_occ.at[bond_pos_0_y].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_0_y_dp = (phonon_occ[bond_pos_0_y]) ** 0.5 * jnp.exp(
            wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
            - overlap
        )
        energy -= self.g * (phonon_occ[bond_pos_0_y]) ** 0.5 * ratio_0_y_dp

        # down
        # bare
        new_elec_pos = [elec_pos[0], ((elec_pos[1][0] + 1) % l_y, elec_pos[1][1])]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_1_y = lax.cond(
            l_y > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0, new_overlap
        )
        energy -= ratio_1_y

        bond_pos_1_y = (0, elec_pos[1][0] * (1 - (l_y == 2)), elec_pos[1][1])
        # create phonon
        new_phonon_occ = phonon_occ.at[bond_pos_1_y].add(1)
        ratio_1_y_cp = lax.cond(
            l_y > 2,
            lambda x: jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            / (phonon_occ[bond_pos_1_y] + 1) ** 0.5,
            lambda x: 0.0,
            0.0,
        )
        energy -= self.g * (phonon_occ[bond_pos_1_y] + 1) ** 0.5 * ratio_1_y_cp

        # destroy phonon
        new_phonon_occ = phonon_occ.at[bond_pos_1_y].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio_1_y_dp = lax.cond(
            l_y > 2,
            lambda x: (phonon_occ[bond_pos_1_y]) ** 0.5
            * jnp.exp(
                wave.calc_overlap_map(new_elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            ),
            lambda x: 0.0,
            0.0,
        )
        energy -= self.g * (phonon_occ[bond_pos_1_y]) ** 0.5 * ratio_1_y_dp

        ratios_1 = [
            ratio_0_x,
            ratio_0_x_cp,
            ratio_0_x_dp,
            ratio_1_x,
            ratio_1_x_cp,
            ratio_1_x_dp,
            ratio_0_y,
            ratio_0_y_cp,
            ratio_0_y_dp,
            ratio_1_y,
            ratio_1_y_cp,
            ratio_1_y_dp,
        ]
        bond_pos_1 = [bond_pos_0_x, bond_pos_1_x, bond_pos_0_y, bond_pos_1_y]

        cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array(ratios_0 + ratios_1)))
        bond_pos = jnp.array(bond_pos_0 + bond_pos_1)
        # jax.debug.print('bond_pos:\n{}\n', bond_pos)
        # jax.debug.print('ratios: {}', jnp.array(ratios_0 + ratios_1))
        # jax.debug.print('energy: {}', energy)
        weight = 1 / cumulative_ratios[-1]
        # jax.debug.print('weight: {}', weight)
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )
        # jax.debug.print('random_number: {}', random_number)
        # jax.debug.print('new_ind: {}', new_ind)

        # electron hop
        walker_copy = jnp.array(walker[0])
        elec_pos = jnp.array(elec_pos)
        elec_0_1 = new_ind // 12
        rel_new_ind = new_ind % 12
        ind_0 = 1 - rel_new_ind // 6
        ind_1 = jnp.array([1, -1, -1, 1])[rel_new_ind // 3]
        l_x_l_y = (rel_new_ind // 3 < 2) * l_x + (rel_new_ind // 3 > 1) * l_y
        walker_copy = walker_copy.at[elec_0_1, ind_0].set(
            (elec_pos[elec_0_1][ind_0] + ind_1) % l_x_l_y
        )
        walker[0] = lax.cond(
            elec_0_1 == 0,
            lambda x: [(walker_copy[0][0], walker_copy[0][1]), walker[0][1]],
            lambda x: [walker[0][0], (walker_copy[1][0], walker_copy[1][1])],
            0.0,
        )

        # eph
        ind_0 = bond_pos[new_ind // 3]
        ind_1 = jnp.array([0, 1, -1])[new_ind % 3]
        walker[1] = walker[1].at[(*ind_0,)].add(ind_1)
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])
        # jax.debug.print('new_walker:\n{}\n', walker)

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega, self.g, self.u))


@dataclass
class long_range_2d:
    omega: float
    g: float
    zeta: float
    u: float
    triplet: bool = False

    def tup_eq(self, pos0, pos1):
        return (pos0[0] == pos1[0]) * (pos0[1] == pos1[1])

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
        elec_pos = walker[0]
        phonon_occ = walker[1]
        (l_y, l_x) = phonon_occ.shape
        n_sites = l_y * l_x
        lattice_sites = jnp.array(lattice.sites)

        overlap = wave.calc_overlap_map(elec_pos, phonon_occ, parameters, lattice)
        overlap_gradient = wave.calc_overlap_map_gradient(
            elec_pos, phonon_occ, parameters, lattice
        )
        qp_weight = lax.cond(
            jnp.sum(phonon_occ) == 0, lambda x: 1.0, lambda x: 0.0, 0.0
        )
        # jax.debug.print('\nwalker: {}', walker)
        # jax.debug.print('overlap: {}', overlap)

        # diagonal
        energy = (
            self.omega * jnp.sum(phonon_occ)
            + self.u * self.tup_eq(elec_pos[0], elec_pos[1])
            + 0.0j
        )

        # 8 e hop terms, n_sites phonon terms
        ratios = jnp.zeros((8 + 2 * n_sites,)) + 0.0j

        # electron hops
        # elctron 1
        new_elec_pos = [(elec_pos[0][0], (elec_pos[0][1] + 1) % l_x), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_0_x = lax.cond(
            l_x > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        ) * (1 - self.triplet * self.tup_eq(new_elec_pos[0], new_elec_pos[1]))
        energy -= ratio_0_x
        ratios = ratios.at[0].set(ratio_0_x)

        new_elec_pos = [(elec_pos[0][0], (elec_pos[0][1] - 1) % l_x), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_1_x = lax.cond(
            l_x > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        ) * (1 - self.triplet * self.tup_eq(new_elec_pos[0], new_elec_pos[1]))
        energy -= ratio_1_x
        ratios = ratios.at[1].set(ratio_1_x)

        new_elec_pos = [((elec_pos[0][0] + 1) % l_y, elec_pos[0][1]), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_0_y = lax.cond(
            l_y > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        ) * (1 - self.triplet * self.tup_eq(new_elec_pos[0], new_elec_pos[1]))
        energy -= ratio_0_y
        ratios = ratios.at[2].set(ratio_0_y)

        new_elec_pos = [((elec_pos[0][0] - 1) % l_y, elec_pos[0][1]), elec_pos[1]]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_1_y = lax.cond(
            l_y > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        ) * (1 - self.triplet * self.tup_eq(new_elec_pos[0], new_elec_pos[1]))
        energy -= ratio_1_y
        ratios = ratios.at[3].set(ratio_1_y)

        # elec 2
        new_elec_pos = [elec_pos[0], (elec_pos[1][0], (elec_pos[1][1] + 1) % l_x)]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_0_x = lax.cond(
            l_x > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        ) * (1 - self.triplet * self.tup_eq(new_elec_pos[0], new_elec_pos[1]))
        energy -= ratio_0_x
        ratios = ratios.at[4].set(ratio_0_x)

        new_elec_pos = [elec_pos[0], (elec_pos[1][0], (elec_pos[1][1] - 1) % l_x)]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_1_x = lax.cond(
            l_x > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        ) * (1 - self.triplet * self.tup_eq(new_elec_pos[0], new_elec_pos[1]))
        energy -= ratio_1_x
        ratios = ratios.at[5].set(ratio_1_x)

        new_elec_pos = [elec_pos[0], ((elec_pos[1][0] + 1) % l_y, elec_pos[1][1])]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_0_y = lax.cond(
            l_y > 1, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        ) * (1 - self.triplet * self.tup_eq(new_elec_pos[0], new_elec_pos[1]))
        energy -= ratio_0_y
        ratios = ratios.at[6].set(ratio_0_y)

        new_elec_pos = [elec_pos[0], ((elec_pos[1][0] - 1) % l_y, elec_pos[1][1])]
        new_overlap = wave.calc_overlap_map(
            new_elec_pos, phonon_occ, parameters, lattice
        )
        ratio_1_y = lax.cond(
            l_y > 2, lambda x: jnp.exp(x - overlap), lambda x: 0.0 * x, new_overlap
        ) * (1 - self.triplet * self.tup_eq(new_elec_pos[0], new_elec_pos[1]))
        energy -= ratio_1_y
        ratios = ratios.at[7].set(ratio_1_y)

        # e_ph coupling
        def scanned_fun(carry, x):
            pos = x
            counter = carry[2]
            dist_0 = lattice.get_distance(elec_pos[0], x)
            dist_1 = lattice.get_distance(elec_pos[1], x)

            new_phonon_occ = phonon_occ.at[pos[0], pos[1]].add(1)
            ratio = (
                jnp.exp(
                    wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice)
                    - overlap
                )
                / (phonon_occ[pos[0], pos[1]] + 1) ** 0.5
            )
            carry[0] -= (
                self.g
                * jnp.exp(-dist_0 / self.zeta)
                * (phonon_occ[pos[0], pos[1]] + 1) ** 0.5
                * ratio
                / (1 + dist_0**2) ** 1.5
            )
            carry[0] -= (
                self.g
                * jnp.exp(-dist_1 / self.zeta)
                * (phonon_occ[pos[0], pos[1]] + 1) ** 0.5
                * ratio
                / (1 + dist_1**2) ** 1.5
            )
            carry[1] = carry[1].at[2 * carry[2] + 8].set(ratio)

            new_phonon_occ = phonon_occ.at[pos[0], pos[1]].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            ratio = (phonon_occ[pos[0], pos[1]]) ** 0.5 * jnp.exp(
                wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice)
                - overlap
            )
            carry[0] -= (
                self.g
                * jnp.exp(-dist_0 / self.zeta)
                * (phonon_occ[pos[0], pos[1]]) ** 0.5
                * ratio
                / (1 + dist_0**2) ** 1.5
            )
            carry[0] -= (
                self.g
                * jnp.exp(-dist_0 / self.zeta)
                * (phonon_occ[pos[0], pos[1]]) ** 0.5
                * ratio
                / (1 + dist_0**2) ** 1.5
            )
            carry[1] = carry[1].at[2 * carry[2] + 9].set(ratio)

            carry[2] = carry[2] + 1

            return carry, x

        counter = 0
        [energy, ratios, _], _ = lax.scan(
            scanned_fun, [energy, ratios, counter], lattice_sites
        )

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))

        # jax.debug.print('ratios: {}', jnp.array(ratios))
        # jax.debug.print('energy: {}', energy)
        weight = 1 / cumulative_ratios[-1]
        # jax.debug.print('weight: {}', weight)
        # jax.debug.print('random_number: {}', random_number)
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )
        # jax.debug.print('new_ind: {}', new_ind)

        lattice_shape = jnp.array((l_y, l_x))
        # electron 1
        walker[0][0] = jnp.array(walker[0][0])
        walker[0][0] = lax.cond(
            new_ind < 4,
            lambda w: w.at[1 - new_ind // 2].set(
                (elec_pos[0][1 - new_ind // 2] + 1 - 2 * (new_ind % 2))
                % lattice_shape[1 - new_ind // 2]
            ),
            lambda w: w,
            walker[0][0],
        )
        walker[0][0] = (walker[0][0][0], walker[0][0][1])
        # electron 2
        walker[0][1] = jnp.array(walker[0][1])
        walker[0][1] = lax.cond(
            (new_ind > 3) & (new_ind < 8),
            lambda w: w.at[1 - (new_ind - 4) // 2].set(
                (elec_pos[1][1 - (new_ind - 4) // 2] + 1 - 2 * ((new_ind - 4) % 2))
                % lattice_shape[1 - (new_ind - 4) // 2]
            ),
            lambda w: w,
            walker[0][1],
        )
        walker[0][1] = (walker[0][1][0], walker[0][1][1])

        # phonon
        walker[1] = lax.cond(
            new_ind >= 8,
            lambda w: w.at[(*(lattice_sites[(new_ind - 8) // 2]),)].add(
                1 - 2 * ((new_ind - 8) % 2)
            ),
            lambda w: w,
            walker[1],
        )
        walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    def __hash__(self):
        return hash((self.omega, self.g, self.zeta, self.triplet))


if __name__ == "__main__":
    phonon_occ = jnp.array([[0, 0, 0], [0, 0, 0]])
    elec_pos = [(0, 0), (0, 1)]
    ham = ssh(1.0, 1.0, 1.0)
    print(ham.energy_diagonal(elec_pos, phonon_occ))
