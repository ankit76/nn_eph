from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
from jax import jit, lax
from jax import numpy as jnp
from jax import vmap

from nn_eph.wavefunctions_n import t_projected_state, wave_function


@dataclass
class heisenberg:
    j: float = 1.0

    @partial(jit, static_argnums=(0, 2))
    def get_marshall_sign(self, walker: jax.Array, lattice):
        return 1.0  # lattice.get_marshall_sign(walker)

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(
        self,
        walker: jax.Array,
        parameters: Any,
        wave: wave_function,
        lattice,
        random_number: float,
    ):
        n_bonds = len(lattice.bonds)

        walker_data = wave.build_walker_data(walker, parameters, lattice)
        sign = self.get_marshall_sign(walker, lattice)
        overlap = walker_data["log_overlap"] + jnp.log(sign + 0.0j)
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)
        qp_weight = 0.0

        def z_ene(bond):
            neighbors = lattice.get_neighboring_sites(bond)
            return self.j * walker[(*neighbors[0],)] * walker[(*neighbors[1],)]

        # diagonal
        energy = jnp.sum(vmap(z_ene)(jnp.array(lattice.bonds))) + 0.0j

        # spin flips
        def flip(bond):
            neighbors = lattice.get_neighboring_sites(bond)
            i1 = neighbors[0]
            i2 = neighbors[1]

            # s1+ s2-
            new_walker = walker.at[(*i1,)].set(0.5)
            new_walker = new_walker.at[(*i2,)].set(-0.5)
            new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
            # new_overlap = (
            #     (walker[i1] == -0.5)
            #     * (walker[i2] == 0.5)
            #     * wave.calc_overlap(new_walker, parameters, lattice)
            # )
            new_overlap = new_walker_data["log_overlap"]
            new_sign = self.get_marshall_sign(new_walker, lattice)
            ratio_1 = (
                (walker[(*i1,)] == -0.5)
                * (walker[(*i2,)] == 0.5)
                * jnp.exp(new_overlap - overlap)
                * new_sign
            )

            # s1- s2+
            new_walker = walker.at[(*i1,)].set(-0.5)
            new_walker = new_walker.at[(*i2,)].set(0.5)
            new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
            # new_overlap = (
            #     (walker[i1] == 0.5)
            #     * (walker[i2] == -0.5)
            #     * wave.calc_overlap(new_walker, parameters, lattice)
            # )
            new_overlap = new_walker_data["log_overlap"]
            new_sign = self.get_marshall_sign(new_walker, lattice)
            ratio_2 = (
                (walker[(*i1,)] == 0.5)
                * (walker[(*i2,)] == -0.5)
                * jnp.exp(new_overlap - overlap)
                * new_sign
            )

            return ratio_1, ratio_2

        ratios = vmap(flip)(jnp.array(lattice.bonds))
        energy += self.j * jnp.sum(jnp.concatenate(ratios)) / 2

        # update walker
        cumulative_ratios = jnp.cumsum(jnp.abs(jnp.concatenate(ratios)))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print("\nwalker: {}", walker)
        # jax.debug.print("overlap: {}", overlap)
        # # jax.debug.print('overlap_gradient: {}', overlap_gradient)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}\n", new_ind)

        bond = new_ind % n_bonds
        neighbors = lattice.get_neighboring_sites(jnp.array(lattice.bonds)[bond])
        i1 = neighbors[0]
        i2 = neighbors[1]
        walker = walker.at[(*i1,)].set(-walker[(*i1,)])
        walker = walker.at[(*i2,)].set(-walker[(*i2,)])

        # jax.debug.print("new_walker: {}", walker)

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    @partial(jit, static_argnums=(0, 3, 4))
    def sf_q(
        self, walker, parameters: Any, wave: wave_function, lattice: Any
    ) -> jnp.array:
        sz_i = walker
        sz_q = jnp.abs(jnp.fft.fftn(sz_i, norm="ortho")) ** 2
        return jnp.array([sz_q, sz_q])

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update_lr(
        self,
        walker: jax.Array,
        parameters: Any,
        wave: t_projected_state,
        lattice,
        random_number: float,
        parameters_copy: Any,
    ):
        """Calculate the local energy of a walker and update it using lr sampling

        Parameters
        ----------
        walker : Sequence
            walker spin configuration
        parameters : Any
            Parameters of the wavefunction
        wave : Any
            Wave function
        lattice : Any
            Lattice
        random_number : float
            Random number used for MC update
        parameters_copy : Any
            Workaround for nans in local energy gradients

        Returns
        -------
        Tuple
            Tuple of local energy, scalar property, overlap_gradient_ratio, weight, updated_walker, overlap
        """
        n_bonds = len(lattice.bonds)

        walker_data = wave.build_walker_data(walker, parameters, lattice)
        sign = self.get_marshall_sign(walker, lattice)
        overlap = walker_data["log_overlap"] + jnp.log(sign + 0.0j)
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)
        prob = (
            jnp.abs(jnp.exp(overlap)) ** 2
            + (jnp.abs(overlap_gradient * jnp.exp(overlap)) ** 2).sum()
        )
        qp_weight = 0.0

        def z_ene(bond):
            neighbors = lattice.get_neighboring_sites(bond)
            return self.j * walker[neighbors[0]] * walker[neighbors[1]]

        # diagonal
        energy = jnp.sum(vmap(z_ene)(jnp.array(lattice.bonds))) + 0.0j

        # assuming this will always be used with t_projected_state
        # calculating overlap with k=0 state
        overlaps = jnp.exp(walker_data["log_overlaps"])
        overlap_0 = jnp.sum(overlaps) * sign
        k_factors = jnp.exp(-jnp.array(wave.trans_factors))
        spin_i = walker.reshape(-1)
        spin_q = jnp.sum(k_factors * spin_i) / jnp.sqrt(lattice.n_sites)
        vector_property = (
            jnp.array([spin_q, spin_q, 1.0]) * overlap_0 / jnp.exp(overlap)
        )  # these are < w | S_q | psi_0 > / < w | psi_q >, < w | N_q | psi_0 > / < w | psi_q >, and < w | psi_0 > / < w | psi_q >

        # spin flips
        # carry: energy
        def flip(carry, bond):
            neighbors = lattice.get_neighboring_sites(bond)
            i1 = neighbors[0]
            i2 = neighbors[1]

            # s1+ s2-
            new_walker = walker.at[(*i1,)].set(0.5)
            new_walker = new_walker.at[(*i2,)].set(-0.5)
            new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
            # new_overlap = (
            #     (walker[(*i1,)] == -0.5)
            #     * (walker[(*i2,)] == 0.5)
            #     * wave.calc_overlap(new_walker, parameters, lattice)
            # )
            new_overlap = new_walker_data["log_overlap"]
            new_sign = self.get_marshall_sign(new_walker, lattice)
            ratio_1 = (
                (walker[(*i1,)] == -0.5)
                * (walker[(*i2,)] == 0.5)
                * jnp.exp(new_overlap - overlap)
                * new_sign
            )
            carry += self.j * ratio_1 / 2
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = jnp.exp(overlap) * ratio_1
            new_prob = (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_1 = new_prob / prob

            # s1- s2+
            new_walker = walker.at[(*i1,)].set(-0.5)
            new_walker = new_walker.at[(*i2,)].set(0.5)
            new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
            # new_overlap = (
            #     (walker[(*i1,)] == 0.5)
            #     * (walker[(*i2,)] == -0.5)
            #     * wave.calc_overlap(new_walker, parameters, lattice)
            # )
            new_overlap = new_walker_data["log_overlap"]
            new_sign = self.get_marshall_sign(new_walker, lattice)
            ratio_2 = (
                (walker[(*i1,)] == 0.5)
                * (walker[(*i2,)] == -0.5)
                * jnp.exp(new_overlap - overlap)
                * new_sign
            )
            carry += self.j * ratio_2 / 2
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = jnp.exp(overlap) * ratio_2
            new_prob = (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_2 = new_prob / prob

            return carry, (prob_ratio_1, prob_ratio_2)

        # ratios = vmap(flip)(jnp.array(lattice.bonds))
        energy, prob_ratios = lax.scan(flip, energy, jnp.array(lattice.bonds))
        # energy += self.j * jnp.sum(jnp.concatenate(ratios)) / 2
        ratios = jnp.concatenate((prob_ratios[0], prob_ratios[1]))

        # update walker
        cumulative_ratios = jnp.cumsum(jnp.abs(ratios) ** 0.5)
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # import jax
        # jax.debug.print("\nwalker: {}", walker)
        # jax.debug.print("overlap: {}", overlap)
        # # jax.debug.print('overlap_gradient: {}', overlap_gradient)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}\n", new_ind)

        bond = new_ind % n_bonds
        neighbors = lattice.get_neighboring_sites(jnp.array(lattice.bonds)[bond])
        i1 = neighbors[0]
        i2 = neighbors[1]
        walker = walker.at[(*i1,)].set(-walker[(*i1,)])
        walker = walker.at[(*i2,)].set(-walker[(*i2,)])

        return (
            energy,
            vector_property,
            overlap_gradient,
            weight,
            walker,
            jnp.exp(overlap),
        )

    def __hash__(self):
        return hash((self.j,))


@dataclass
class heisenberg_bond:
    omega: float
    g: float
    j: float = 1.0
    max_n_phonons: Any = jnp.inf

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(
        self,
        walker: jax.Array,
        parameters: dict,
        wave: wave_function,
        lattice,
        random_number: float,
    ):
        spins = walker[0]
        phonons = walker[1:]
        n_bonds = len(lattice.bonds)

        walker_data = wave.build_walker_data(walker, parameters, lattice)
        overlap = walker_data["log_overlap"]
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)

        # qp_weight
        # ignoring max_n_phonons, i = i_r
        bond_i = lattice.bonds[1]
        neighbors = lattice.get_neighboring_sites(bond_i)
        i1 = neighbors[0]
        i2 = neighbors[1]
        # bond_i_r = lattice.bonds[n_bonds // 2]
        # qp_weight = spins[i1]
        qp_weight = spins[(*i1,)] * spins[(*i2,)]

        new_walker = walker.at[(0, *i1)].set(0.5)
        new_walker = new_walker.at[(0, *i2)].set(-0.5)
        new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
        new_overlap = new_walker_data["log_overlap"]
        ratio_2 = (
            (spins[(*i1,)] == -0.5)
            * (spins[(*i2,)] == 0.5)
            * jnp.exp(new_overlap - overlap)
        )
        qp_weight += ratio_2 / 2

        new_walker = walker.at[(0, *i1)].set(-0.5)
        new_walker = new_walker.at[(0, *i2)].set(0.5)
        new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
        new_overlap = new_walker_data["log_overlap"]
        ratio_5 = (
            (spins[(*i1,)] == 0.5)
            * (spins[(*i2,)] == -0.5)
            * jnp.exp(new_overlap - overlap)
        )
        qp_weight += ratio_5 / 2

        ## c c
        # new_phonons = phonons.at[bond_i].add(1)
        # new_phonons = new_phonons.at[bond_i_r].add(1)
        # new_overlap = wave.calc_overlap([spins, new_phonons], parameters, lattice)
        # qp_weight += (phonons[bond_i]+1)**0.5 * (phonons[bond_i_r]+1)**0.5 * new_overlap / overlap

        ## c a
        # new_phonons = phonons.at[bond_i].add(1)
        # new_phonons = new_phonons.at[bond_i_r].add(-1)
        # new_phonons = jnp.where(new_phonons < 0, 0, new_phonons)
        # new_overlap = wave.calc_overlap([spins, new_phonons], parameters, lattice)
        # qp_weight += (phonons[bond_i]+1)**0.5 * (phonons[bond_i_r])**0.5 * new_overlap / overlap

        ## a c
        # new_phonons = phonons.at[bond_i].add(-1)
        # new_phonons = new_phonons.at[bond_i_r].add(1)
        # new_phonons = jnp.where(new_phonons < 0, 0, new_phonons)
        # new_overlap = wave.calc_overlap([spins, new_phonons], parameters, lattice)
        # qp_weight += (phonons[bond_i])**0.5 * (phonons[bond_i_r]+1)**0.5 * new_overlap / overlap

        ## a a
        # new_phonons = phonons.at[bond_i].add(-1)
        # new_phonons = new_phonons.at[bond_i_r].add(-1)
        # new_phonons = jnp.where(new_phonons < 0, 0, new_phonons)
        # new_overlap = wave.calc_overlap([spins, new_phonons], parameters, lattice)
        # qp_weight += (phonons[bond_i])**0.5 * (phonons[bond_i_r])**0.5 * new_overlap / overlap

        # diagonal
        energy = self.omega * jnp.sum(phonons) + 0.0j

        # spin flips
        # carry = energy
        def scanned_fun(carry, bond):
            neighbors = lattice.get_neighboring_sites(bond)
            i1 = neighbors[0]
            i2 = neighbors[1]
            nph = phonons[(*bond,)]
            sp1 = spins[(*i1,)]
            sp2 = spins[(*i2,)]
            new_phonons_c = phonons.at[(*bond,)].set(nph + 1)
            new_phonons_a = phonons.at[(*bond,)].set(nph - 1)
            new_phonons_a = jnp.where(new_phonons_a < 0, 0, new_phonons_a)

            new_walker_c = walker.at[1:].set(new_phonons_c)
            new_walker_a = walker.at[1:].set(new_phonons_a)

            # s1z s2z
            # bare
            carry += self.j * sp1 * sp2

            # create phonon
            new_walker_data = wave.build_walker_data(new_walker_c, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_0 = (
                (jnp.sum(phonons) < self.max_n_phonons)
                * jnp.exp(new_overlap - overlap)
                / (nph + 1) ** 0.5
            )
            carry += self.j * self.g * (nph + 1) ** 0.5 * ratio_0 * sp1 * sp2

            # annihilate phonon
            new_walker_data = wave.build_walker_data(new_walker_a, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_1 = (nph > 0) * (nph) ** 0.5 * jnp.exp(new_overlap - overlap)
            carry += self.j * self.g * (nph) ** 0.5 * ratio_1 * sp1 * sp2

            # s1+ s2-
            # bare
            new_walker = walker.at[(0, *i1)].set(0.5)
            new_walker = new_walker.at[(0, *i2)].set(-0.5)
            new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_2 = (sp1 == -0.5) * (sp2 == 0.5) * jnp.exp(new_overlap - overlap)
            carry += self.j * ratio_2 / 2

            # create phonon
            new_walker_c = new_walker_c.at[0].set(new_walker[0])
            new_walker_data = wave.build_walker_data(new_walker_c, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_3 = (
                (jnp.sum(phonons) < self.max_n_phonons)
                * (sp1 == -0.5)
                * (sp2 == 0.5)
                * jnp.exp(new_overlap - overlap)
                / (nph + 1) ** 0.5
            )
            carry += self.j * self.g * (nph + 1) ** 0.5 * ratio_3 / 2

            # annihilate phonon
            new_walker_a = new_walker_a.at[0].set(new_walker[0])
            new_walker_data = wave.build_walker_data(new_walker_a, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_4 = (
                (nph > 0)
                * (sp1 == -0.5)
                * (sp2 == 0.5)
                * nph**0.5
                * jnp.exp(new_overlap - overlap)
            )
            carry += self.j * self.g * nph**0.5 * ratio_4 / 2

            # s1- s2+
            # bare
            new_walker = walker.at[(0, *i1)].set(-0.5)
            new_walker = new_walker.at[(0, *i2)].set(0.5)
            new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_5 = (sp1 == 0.5) * (sp2 == -0.5) * jnp.exp(new_overlap - overlap)
            carry += self.j * ratio_5 / 2

            # create phonon
            new_walker_c = new_walker_c.at[0].set(new_walker[0])
            new_walker_data = wave.build_walker_data(new_walker_c, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_6 = (
                (jnp.sum(phonons) < self.max_n_phonons)
                * (sp1 == 0.5)
                * (sp2 == -0.5)
                * jnp.exp(new_overlap - overlap)
                / (nph + 1) ** 0.5
            )
            carry += self.j * self.g * (nph + 1) ** 0.5 * ratio_6 / 2

            # annihilate phonon
            new_walker_a = new_walker_a.at[0].set(new_walker[0])
            new_walker_data = wave.build_walker_data(new_walker_a, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_7 = (
                (nph > 0)
                * (sp1 == 0.5)
                * (sp2 == -0.5)
                * nph**0.5
                * jnp.exp(new_overlap - overlap)
            )
            carry += self.j * self.g * nph**0.5 * ratio_7 / 2

            return carry, (
                ratio_0,
                ratio_1,
                ratio_2,
                ratio_3,
                ratio_4,
                ratio_5,
                ratio_6,
                ratio_7,
            )

        energy, ratios = lax.scan(scanned_fun, energy, jnp.array(lattice.bonds))

        # update walker
        cumulative_ratios = jnp.cumsum(jnp.abs(jnp.concatenate(ratios)))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print("\nwalker: {}", walker)
        # jax.debug.print("overlap: {}", overlap)
        # # jax.debug.print('overlap_gradient: {}', overlap_gradient)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}\n", new_ind)

        bond = jnp.array(lattice.bonds)[new_ind % n_bonds]
        neighbors = lattice.get_neighboring_sites(bond)
        i1 = neighbors[0]
        i2 = neighbors[1]
        excitation_type = new_ind // n_bonds
        spin_change = jnp.array([0, 0, 1, 1, 1, -1, -1, -1])[excitation_type]
        phonon_change = jnp.array([1, -1, 0, 1, -1, 0, 1, -1])[excitation_type]
        walker = walker.at[(0, *i1)].set(walker[(0, *i1)] + spin_change)
        walker = walker.at[(0, *i2)].set(walker[(0, *i2)] - spin_change)
        phonons = phonons.at[(*bond,)].set(phonons[(*bond,)] + phonon_change)
        phonons = jnp.where(phonons < 0, 0, phonons)
        walker = walker.at[1:].set(phonons)

        # jax.debug.print("new_walker: {}", walker)

        return energy, qp_weight, overlap_gradient, weight, walker, jnp.exp(overlap)

    @partial(jit, static_argnums=(0, 3, 4))
    def sf_q(
        self, walker, parameters: Any, wave: wave_function, lattice: Any
    ) -> jnp.array:
        sz_i = walker[0]
        sz_q = jnp.abs(jnp.fft.fftn(sz_i, norm="ortho")) ** 2
        return jnp.array([sz_q, sz_q])

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update_lr(
        self,
        walker: jax.Array,
        parameters: Any,
        wave: t_projected_state,
        lattice,
        random_number: float,
        parameters_copy: Any,
    ):
        """Calculate the local energy of a walker and update it using lr sampling

        Parameters
        ----------
        walker : Sequence
            walker spin phonon configuration
        parameters : Any
            Parameters of the wavefunction
        wave : Any
            Wave function
        lattice : Any
            Lattice
        random_number : float
            Random number used for MC update
        parameters_copy : Any
            Workaround for nans in local energy gradients

        Returns
        -------
        Tuple
            Tuple of local energy, scalar property, overlap_gradient_ratio, weight, updated_walker, overlap
        """

        spins = walker[0]
        phonons = walker[1:]
        n_bonds = len(lattice.bonds)

        walker_data = wave.build_walker_data(walker, parameters, lattice)
        overlap = walker_data["log_overlap"]
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)
        prob = (
            jnp.abs(jnp.exp(overlap)) ** 2
            + (jnp.abs(overlap_gradient * jnp.exp(overlap)) ** 2).sum()
        )
        qp_weight = 0.0

        # assuming this will always be used with t_projected_state
        # calculating overlap with k=0 state
        overlaps = jnp.exp(walker_data["log_overlaps"])
        overlap_0 = jnp.sum(overlaps)
        k_factors = jnp.exp(-jnp.array(wave.trans_factors))
        spin_i = walker[0].reshape(-1)
        spin_q = jnp.sum(k_factors * spin_i) / jnp.sqrt(lattice.n_sites)
        vector_property = (
            jnp.array([spin_q, spin_q, 1.0]) * overlap_0 / jnp.exp(overlap)
        )  # these are < w | S_q | psi_0 > / < w | psi_q >, < w | N_q | psi_0 > / < w | psi_q >, and < w | psi_0 > / < w | psi_q >

        # diagonal
        energy = self.omega * jnp.sum(phonons) + 0.0j

        def scanned_fun(carry, bond):
            neighbors = lattice.get_neighboring_sites(bond)
            i1 = neighbors[0]
            i2 = neighbors[1]
            nph = phonons[(*bond,)]
            sp1 = spins[(*i1,)]
            sp2 = spins[(*i2,)]
            new_phonons_c = phonons.at[(*bond,)].set(nph + 1)
            new_phonons_a = phonons.at[(*bond,)].set(nph - 1)
            new_phonons_a = jnp.where(new_phonons_a < 0, 0, new_phonons_a)

            new_walker_c = walker.at[1:].set(new_phonons_c)
            new_walker_a = walker.at[1:].set(new_phonons_a)

            # s1z s2z
            # bare
            carry += self.j * sp1 * sp2

            # create phonon
            new_walker_data = wave.build_walker_data(new_walker_c, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_0 = (
                (jnp.sum(phonons) < self.max_n_phonons)
                * jnp.exp(new_overlap - overlap)
                / (nph + 1) ** 0.5
            )
            carry += self.j * self.g * (nph + 1) ** 0.5 * ratio_0 * sp1 * sp2
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = jnp.exp(overlap) * ratio_0
            new_prob = (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_0 = new_prob / prob

            # annihilate phonon
            new_walker_data = wave.build_walker_data(new_walker_a, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_1 = (nph > 0) * (nph) ** 0.5 * jnp.exp(new_overlap - overlap)
            carry += self.j * self.g * (nph) ** 0.5 * ratio_1 * sp1 * sp2
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = jnp.exp(overlap) * ratio_1
            new_prob = (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_1 = new_prob / prob

            # s1+ s2-
            # bare
            new_walker = walker.at[(0, *i1)].set(0.5)
            new_walker = new_walker.at[(0, *i2)].set(-0.5)
            new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_2 = (sp1 == -0.5) * (sp2 == 0.5) * jnp.exp(new_overlap - overlap)
            carry += self.j * ratio_2 / 2
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = jnp.exp(overlap) * ratio_2
            new_prob = (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_2 = new_prob / prob

            # create phonon
            new_walker_c = new_walker_c.at[0].set(new_walker[0])
            new_walker_data = wave.build_walker_data(new_walker_c, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_3 = (
                (jnp.sum(phonons) < self.max_n_phonons)
                * (sp1 == -0.5)
                * (sp2 == 0.5)
                * jnp.exp(new_overlap - overlap)
                / (nph + 1) ** 0.5
            )
            carry += self.j * self.g * (nph + 1) ** 0.5 * ratio_3 / 2
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = jnp.exp(overlap) * ratio_3
            new_prob = (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_3 = new_prob / prob

            # annihilate phonon
            new_walker_a = new_walker_a.at[0].set(new_walker[0])
            new_walker_data = wave.build_walker_data(new_walker_a, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_4 = (
                (nph > 0)
                * (sp1 == -0.5)
                * (sp2 == 0.5)
                * nph**0.5
                * jnp.exp(new_overlap - overlap)
            )
            carry += self.j * self.g * nph**0.5 * ratio_4 / 2
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = jnp.exp(overlap) * ratio_4
            new_prob = (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_4 = new_prob / prob

            # s1- s2+
            # bare
            new_walker = walker.at[(0, *i1)].set(-0.5)
            new_walker = new_walker.at[(0, *i2)].set(0.5)
            new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_5 = (sp1 == 0.5) * (sp2 == -0.5) * jnp.exp(new_overlap - overlap)
            carry += self.j * ratio_5 / 2
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = jnp.exp(overlap) * ratio_5
            new_prob = (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_5 = new_prob / prob

            # create phonon
            new_walker_c = new_walker_c.at[0].set(new_walker[0])
            new_walker_data = wave.build_walker_data(new_walker_c, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_6 = (
                (jnp.sum(phonons) < self.max_n_phonons)
                * (sp1 == 0.5)
                * (sp2 == -0.5)
                * jnp.exp(new_overlap - overlap)
                / (nph + 1) ** 0.5
            )
            carry += self.j * self.g * (nph + 1) ** 0.5 * ratio_6 / 2
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = jnp.exp(overlap) * ratio_6
            new_prob = (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_6 = new_prob / prob

            # annihilate phonon
            new_walker_a = new_walker_a.at[0].set(new_walker[0])
            new_walker_data = wave.build_walker_data(new_walker_a, parameters, lattice)
            new_overlap = new_walker_data["log_overlap"]
            ratio_7 = (
                (nph > 0)
                * (sp1 == 0.5)
                * (sp2 == -0.5)
                * nph**0.5
                * jnp.exp(new_overlap - overlap)
            )
            carry += self.j * self.g * nph**0.5 * ratio_7 / 2
            new_overlap_gradient = wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = jnp.exp(overlap) * ratio_7
            new_prob = (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_7 = new_prob / prob

            return carry, (
                prob_ratio_0,
                prob_ratio_1,
                prob_ratio_2,
                prob_ratio_3,
                prob_ratio_4,
                prob_ratio_5,
                prob_ratio_6,
                prob_ratio_7,
            )

        energy, prob_ratios = lax.scan(scanned_fun, energy, jnp.array(lattice.bonds))
        ratios = jnp.concatenate(prob_ratios)

        # update walker
        cumulative_ratios = jnp.cumsum(jnp.abs(ratios) ** 0.5)
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print("\nwalker: {}", walker)
        # jax.debug.print("overlap: {}", overlap)
        # # jax.debug.print('overlap_gradient: {}', overlap_gradient)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}\n", new_ind)

        bond = jnp.array(lattice.bonds)[new_ind % n_bonds]
        neighbors = lattice.get_neighboring_sites(bond)
        i1 = neighbors[0]
        i2 = neighbors[1]
        excitation_type = new_ind // n_bonds
        spin_change = jnp.array([0, 0, 1, 1, 1, -1, -1, -1])[excitation_type]
        phonon_change = jnp.array([1, -1, 0, 1, -1, 0, 1, -1])[excitation_type]
        walker = walker.at[(0, *i1)].set(walker[(0, *i1)] + spin_change)
        walker = walker.at[(0, *i2)].set(walker[(0, *i2)] - spin_change)
        phonons = phonons.at[(*bond,)].set(phonons[(*bond,)] + phonon_change)
        phonons = jnp.where(phonons < 0, 0, phonons)
        walker = walker.at[1:].set(phonons)

        # jax.debug.print("new_walker: {}", walker)

        return (
            energy,
            vector_property,
            overlap_gradient,
            weight,
            walker,
            jnp.exp(overlap),
        )

    def __hash__(self):
        return hash((self.omega, self.g, self.j, self.max_n_phonons))
