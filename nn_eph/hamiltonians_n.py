from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence, Tuple

import jax
from jax import jit, lax
from jax import numpy as jnp

from nn_eph.wavefunctions_n import t_projected_state, wave_function


@dataclass
class hubbard:
    """Hubbard model Hamiltonian

    Attributes
    ----------
    u : float
        On-site Coulomb repulsion
    n_orbs : int
        Number of orbitals
    n_elec : Sequence
        Number of electrons
    antiperiodic: bool
        Antiperiodic boundary conditions
    """

    u: float
    n_orbs: int
    n_elec: Sequence
    antiperiodic: bool = False

    @partial(jit, static_argnums=(0,))
    def sz(
        self,
        walker: Sequence,
    ) -> float:
        """Calculate the spin at all sites"""
        return (walker[0] - walker[1]).reshape(-1)

    @partial(jit, static_argnums=(0,))
    def num(self, walker: Sequence) -> float:
        """Calculate the number of electrons at all sites"""
        return (walker[0] + walker[1]).reshape(-1)

    @partial(jit, static_argnums=(0,))
    def double_occupancy(self, walker: Sequence) -> jax.Array:
        """Calculate the double occupancy"""
        return jnp.sum(walker[0] * walker[1])

    @partial(jit, static_argnums=(0,))
    def diagonal_energy(self, walker: Sequence) -> jax.Array:
        """Calculate the diagonal energy of a walker"""
        return self.u * jnp.sum(walker[0] * walker[1])

    @partial(jit, static_argnums=(0, 3, 4))
    def sf_q(
        self, walker: Sequence, parameters: Any, wave: wave_function, lattice: Any
    ) -> jnp.array:
        sz_i = walker[0] - walker[1]
        sz_q = jnp.abs(jnp.fft.fftn(sz_i, norm="ortho")) ** 2
        c_i = walker[0] + walker[1]
        c_q = jnp.abs(jnp.fft.fftn(c_i, norm="ortho")) ** 2
        return jnp.array([sz_q, c_q])

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(
        self,
        walker: jax.Array,
        parameters: Any,
        wave: Any,
        lattice: Any,
        random_number: float,
    ) -> Tuple:
        """Calculate the local energy of a walker and update it

        Parameters
        ----------
        walker : Sequence
            walker : (elec_occ_up, elec_occ_dn)
        parameters : Any
            Parameters of the wavefunction
        wave : Any
            Wavefunction
        lattice : Any
            Lattice
        random_number : float
            Random number used for MC update

        Returns
        -------
        Tuple
            Tuple of local energy, scalar property, overlap_gradient_ratio, weight, updated_walker, overlap
        """
        walker_data = wave.build_walker_data(walker, parameters, lattice)
        elec_idx_up = jnp.nonzero(walker[0].reshape(-1), size=self.n_elec[0])[0]
        elec_pos_up = jnp.array(lattice.sites)[elec_idx_up]
        elec_idx_dn = jnp.nonzero(walker[1].reshape(-1), size=self.n_elec[1])[0]
        elec_pos_dn = jnp.array(lattice.sites)[elec_idx_dn]

        # diagonal
        energy = self.diagonal_energy(walker) + 0.0j

        overlap = wave.calc_overlap(walker_data, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)
        double_occ = jnp.sum(walker[0] * walker[1])

        # electron hops
        # NB: parity is not included here, it is included in the wave function overlap ratio
        # scan over neighbors of elec_pos
        # excitation["sm_idx"]: [ spin, i_rel, a_abs ]
        # excitation["idx"]: [ spin, i_rel, a_abs ]
        # carry: [ energy, spin, elec_pos, occ_idx ]
        def scanned_fun(carry, x):
            neighbor, neighbor_edge_bond = x
            neighbor_site_num = lattice.get_site_num(neighbor)
            excitation = {}
            excitation["sm_idx"] = jnp.array((carry[1], carry[3], neighbor_site_num))
            excitation["idx"] = jnp.array((carry[1], carry[2], neighbor_site_num))
            ratio = (walker[carry[1]][(*neighbor,)] == 0) * wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            carry[0] -= ratio * (1 - 2 * neighbor_edge_bond * self.antiperiodic)
            return carry, ratio

        # scan over electrons
        # carry: [ energy, spin, occ_idx ]
        def outer_scanned_fun(carry, x):
            [carry[0], _, _, _], hop_ratios = lax.scan(
                scanned_fun,
                [carry[0], carry[1], lattice.get_site_num(x), carry[2]],
                (
                    lattice.get_nearest_neighbors(x),
                    lattice.get_nearest_neighbors_edge_bond(x),
                ),
            )
            carry[2] += 1
            return carry, hop_ratios

        [energy, _, _], ratios_up = lax.scan(
            outer_scanned_fun, [energy, 0, 0], elec_pos_up
        )
        [energy, _, _], ratios_dn = lax.scan(
            outer_scanned_fun, [energy, 1, 0], elec_pos_dn
        )
        ratios = jnp.concatenate((ratios_up.reshape(-1), ratios_dn.reshape(-1)))

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print("\nwalker: {}", walker)
        # jax.debug.print("overlap: {}", overlap)
        # jax.debug.print("grad: {}", overlap_gradient)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}\n", new_ind)

        # update
        spin_ind = (new_ind >= elec_pos_up.shape[0] * lattice.coord_num) * 1
        pos = lax.cond(
            spin_ind == 0,
            lambda x: elec_pos_up[x // lattice.coord_num],
            lambda x: elec_pos_dn[
                (x - elec_pos_up.shape[0] * lattice.coord_num) // lattice.coord_num
            ],
            new_ind,
        )
        neighbor_ind = new_ind % lattice.coord_num
        neighbor_pos = lattice.get_nearest_neighbors(pos)[neighbor_ind]
        walker = walker.at[(spin_ind, *pos)].set(0)
        walker = walker.at[(spin_ind, *neighbor_pos)].set(1)

        # jax.debug.print("new_walker: {}", walker)

        energy = jnp.array(jnp.where(jnp.isnan(energy), 0.0, energy))
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        return (
            energy,
            double_occ,
            overlap_gradient,
            weight,
            walker,
            jnp.exp(overlap),
        )

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update_lr(
        self,
        walker: jax.Array,
        parameters: Any,
        wave: Any,
        lattice: Any,
        random_number: float,
        parameters_copy: Any,
    ) -> Tuple:
        """Calculate the local energy of a walker and update it using lr sampling

        Parameters
        ----------
        walker : Sequence
            walker : (elec_occ_up, elec_occ_dn)
        parameters : Any
            Parameters of the wavefunction
        wave : Any
            Wavefunction
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
        walker_data = wave.build_walker_data(walker, parameters, lattice)
        elec_idx_up = jnp.nonzero(walker[0].reshape(-1), size=self.n_elec[0])[0]
        elec_pos_up = jnp.array(lattice.sites)[elec_idx_up]
        elec_idx_dn = jnp.nonzero(walker[1].reshape(-1), size=self.n_elec[1])[0]
        elec_pos_dn = jnp.array(lattice.sites)[elec_idx_dn]

        # diagonal
        energy = self.diagonal_energy(walker) + 0.0j

        overlap = wave.calc_overlap(walker_data, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)
        prob = (
            jnp.abs(jnp.exp(overlap)) ** 2
            + (jnp.abs(overlap_gradient * jnp.exp(overlap)) ** 2).sum()
        )
        # double_occ = jnp.sum(walker[0] * walker[1])
        # assuming this will always be used with t_projected_state
        # calculating overlap with k=0 state
        overlaps = jnp.exp(walker_data["log_overlaps"])
        overlap_0 = jnp.sum(overlaps)
        k_factors = jnp.exp(-jnp.array(wave.symm_factors))
        spin_i = (walker[0] - walker[1]).reshape(-1)
        charge_i = (walker[0] + walker[1]).reshape(-1)
        spin_q = jnp.sum(k_factors * spin_i) / jnp.sqrt(lattice.n_sites)
        charge_q = jnp.sum(k_factors * charge_i) / jnp.sqrt(lattice.n_sites)
        vector_property = (
            jnp.array([spin_q, charge_q, 1.0]) * overlap_0 / jnp.exp(overlap)
        )  # these are < w | S_q | psi_0 > / < w | psi_q >, < w | N_q | psi_0 > / < w | psi_q >, and < w | psi_0 > / < w | psi_q >

        # electron hops
        # NB: parity is not included here, it is included in the wave function overlap ratio
        # scan over neighbors of elec_pos
        # excitation["sm_idx"]: [ spin, i_rel, a_abs ]
        # excitation["idx"]: [ spin, i_rel, a_abs ]
        # carry: [ energy, spin, elec_id, occ_idx, elec_pos ]
        def scanned_fun(carry, x):
            neighbor, neighbor_edge_bond = x
            neighbor_site_num = lattice.get_site_num(neighbor)
            excitation = {}
            excitation["sm_idx"] = jnp.array((carry[1], carry[3], neighbor_site_num))
            excitation["idx"] = jnp.array((carry[1], carry[2], neighbor_site_num))
            ratio = (walker[carry[1]][(*neighbor,)] == 0) * wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            carry[0] -= ratio * (1 - 2 * neighbor_edge_bond * self.antiperiodic)
            new_walker = walker.at[(carry[1], *carry[4])].set(0)
            new_walker = new_walker.at[(carry[1], *neighbor)].set(1)
            new_walker_data = wave.build_walker_data(
                new_walker, parameters_copy, lattice
            )
            new_overlap_gradient = (
                walker[carry[1]][(*neighbor,)] == 0
            ) * wave.calc_overlap_gradient(new_walker_data, parameters_copy, lattice)
            new_overlap = ratio * jnp.exp(overlap)
            new_prob = (walker[carry[1]][(*neighbor,)] == 0) * (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio = (
                new_prob / prob
            )  # taking square root here inexplicably leads to nans in local energy gradients, even though it has no bearing on local energy
            return carry, prob_ratio

        # scan over electrons
        # carry: [ energy, spin, occ_idx ]
        def outer_scanned_fun(carry, x):
            [carry[0], _, _, _, _], hop_ratios = lax.scan(
                scanned_fun,
                [carry[0], carry[1], lattice.get_site_num(x), carry[2], x],
                (
                    lattice.get_nearest_neighbors(x),
                    lattice.get_nearest_neighbors_edge_bond(x),
                ),
            )
            carry[2] += 1
            return carry, hop_ratios

        [energy, _, _], ratios_up = lax.scan(
            outer_scanned_fun, [energy, 0, 0], elec_pos_up
        )
        [energy, _, _], ratios_dn = lax.scan(
            outer_scanned_fun, [energy, 1, 0], elec_pos_dn
        )
        ratios = jnp.concatenate((ratios_up.reshape(-1), ratios_dn.reshape(-1)))

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios) ** 0.5)  # moved square root here
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print("\nwalker: {}", walker)
        # jax.debug.print("overlap: {}", overlap)
        # jax.debug.print("overlap_gradient: {}", overlap_gradient)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}\n", new_ind)

        # update
        spin_ind = (new_ind >= elec_pos_up.shape[0] * lattice.coord_num) * 1
        pos = lax.cond(
            spin_ind == 0,
            lambda x: elec_pos_up[x // lattice.coord_num],
            lambda x: elec_pos_dn[
                (x - elec_pos_up.shape[0] * lattice.coord_num) // lattice.coord_num
            ],
            new_ind,
        )
        neighbor_ind = new_ind % lattice.coord_num
        neighbor_pos = lattice.get_nearest_neighbors(pos)[neighbor_ind]
        walker = walker.at[(spin_ind, *pos)].set(0)
        walker = walker.at[(spin_ind, *neighbor_pos)].set(1)

        # jax.debug.print("new_walker: {}", walker)

        energy = jnp.array(jnp.where(jnp.isnan(energy), 0.0, energy))
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        return (
            energy,
            vector_property,
            overlap_gradient,
            weight,
            walker,
            jnp.exp(overlap),
        )

    def __hash__(self):
        return hash(
            (
                self.u,
                self.n_orbs,
                self.n_elec,
            )
        )


@dataclass
class hubbard_holstein:
    """Hubbard-Holstein model Hamiltonian

    Attributes
    ----------
    omega : float
        Phonon frequency
    g : float
        Electron-phonon coupling
    u : float
        On-site Coulomb repulsion
    n_orbs : int
        Number of orbitals
    n_elec : Sequence
        Number of electrons
    max_n_phonons : any, optional
        Maximum number of phonons, by default jnp.inf
    """

    omega: float
    g: float
    u: float
    n_orbs: int
    n_elec: Sequence
    max_n_phonons: any = jnp.inf
    antiperiodic: bool = False

    @partial(jit, static_argnums=(0, 3, 4))
    def sf_q(
        self, walker: Sequence, parameters: Any, wave: wave_function, lattice: Any
    ) -> jnp.array:
        sz_i = walker[0] - walker[1]
        sz_q = jnp.abs(jnp.fft.fftn(sz_i, norm="ortho")) ** 2
        c_i = walker[0] + walker[1]
        c_q = jnp.abs(jnp.fft.fftn(c_i, norm="ortho")) ** 2
        return jnp.array([sz_q, c_q])

    @partial(jit, static_argnums=(0,))
    def diagonal_energy(self, walker: Sequence) -> jax.Array:
        """Calculate the diagonal energy of a walker

        Parameters
        ----------
        walker: Sequence
            Walker

        Returns
        -------
        float
            Diagonal energy
        """
        coulomb_energy = self.u * jnp.sum(walker[0] * walker[1])
        phonon_energy = self.omega * jnp.sum(walker[2])
        energy = coulomb_energy + phonon_energy
        return energy

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(
        self,
        walker: jax.Array,
        parameters: Any,
        wave: Any,
        lattice: Any,
        random_number: float,
    ) -> Tuple:
        """Calculate the local energy of a walker and update it

        Parameters
        ----------
        walker : Sequence
            walker : [ elec_occ_up, elec_occ_dn, phonon_occ ]
        parameters : Any
            Parameters of the wavefunction
        wave : Any
            Wavefunction
        lattice : Any
            Lattice
        random_number : float
            Random number used for MC update

        Returns
        -------
        Tuple
            Tuple of local energy, qp_weight (0. here), overlap_gradient_ratio, weight, updated_walker, overlap
        """
        walker_data = wave.build_walker_data(walker, parameters, lattice)
        elec_pos_up = jnp.array(lattice.sites)[
            jnp.nonzero(walker[0].reshape(-1), size=self.n_elec[0])[0]
        ]
        elec_pos_dn = jnp.array(lattice.sites)[
            jnp.nonzero(walker[1].reshape(-1), size=self.n_elec[1])[0]
        ]
        phonon_occ = walker[2]

        # diagonal
        energy = self.diagonal_energy(walker) + 0.0j

        overlap = wave.calc_overlap(walker_data, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)
        # qp_weight = (jnp.sum(phonon_occ) == 0) * 1.0
        qp_weight = jnp.sum(walker[0] * walker[1])

        # electron hops
        # scan over neighbors of elec_pos
        # carry: [ energy, spin, elec_pos, idx ]
        def scanned_fun(carry, x):
            neighbor, neighbor_edge_bond = x
            neighbor_site_num = lattice.get_site_num(neighbor)
            excitation_ee = {}
            excitation_ee["sm_idx"] = jnp.array((carry[1], carry[3], neighbor_site_num))
            excitation_ee["idx"] = jnp.array((carry[1], carry[2], neighbor_site_num))
            excitation_ph = jnp.array((0, 0))
            excitation = {"ee": excitation_ee, "ph": excitation_ph}
            ratio = (walker[carry[1]][(*neighbor,)] == 0) * wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            carry[0] -= ratio * (1 - 2 * neighbor_edge_bond * self.antiperiodic)
            return carry, ratio

        # scan over electrons
        # carry: [ energy, spin, occ_idx ]
        def outer_scanned_fun(carry, x):
            [carry[0], _, _, _], hop_ratios = lax.scan(
                scanned_fun,
                [carry[0], carry[1], lattice.get_site_num(x), carry[2]],
                (
                    lattice.get_nearest_neighbors(x),
                    lattice.get_nearest_neighbors_edge_bond(x),
                ),
            )
            carry[2] += 1
            return carry, hop_ratios

        [energy, _, _], ratios_up = lax.scan(
            outer_scanned_fun, [energy, 0, 0], elec_pos_up
        )
        [energy, _, _], ratios_dn = lax.scan(
            outer_scanned_fun, [energy, 1, 0], elec_pos_dn
        )
        hop_ratios = jnp.concatenate((ratios_up.reshape(-1), ratios_dn.reshape(-1)))

        # eph
        excitation_ee = {}
        excitation_ee["sm_idx"] = jnp.array((2, 0, 0))
        excitation_ee["idx"] = jnp.array((2, 0, 0))
        excitation = {}
        excitation["ee"] = excitation_ee

        # scan over lattice sites
        # carry: [ energy, site_idx ]
        def scanned_fun_ph(carry, x):
            # add phonon
            excitation["ph"] = jnp.array((carry[1], 1))
            overlap_ratio = wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            ratio_0 = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * ((walker[0][(*x,)] + walker[1][(*x,)]) > 0)
                * overlap_ratio
                / (phonon_occ[(*x,)] + 1) ** 0.5
            )
            carry[0] -= (
                self.g
                * (phonon_occ[(*x,)] + 1) ** 0.5
                * ratio_0
                * (walker[0][(*x,)] + walker[1][(*x,)])
            )

            # remove phonon
            excitation["ph"] = jnp.array((carry[1], -1))
            overlap_ratio = wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            ratio_1 = (
                (phonon_occ[(*x,)] > 0)
                * ((walker[0][(*x,)] + walker[1][(*x,)]) > 0)
                * overlap_ratio
                * phonon_occ[(*x,)] ** 0.5
            )
            carry[0] -= (
                self.g
                * phonon_occ[(*x,)] ** 0.5
                * ratio_1
                * (walker[0][(*x,)] + walker[1][(*x,)])
            )
            carry[1] += 1
            return carry, (ratio_0, ratio_1)

        [energy, _], (ratios_p, ratios_m) = lax.scan(
            scanned_fun_ph, [energy, 0], jnp.array(lattice.sites)
        )

        ratios = jnp.concatenate((hop_ratios, ratios_p, ratios_m))

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print("\nwalker: {}", walker_data)
        # jax.debug.print("overlap: {}", overlap)
        # jax.debug.print("grad: {}", overlap_gradient)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}\n", new_ind)

        # update
        # NB: some of these operations rely on jax not complaining about out of bounds array access
        ee = new_ind < (self.n_elec[0] + self.n_elec[1]) * lattice.coord_num
        spin_ind = (new_ind >= self.n_elec[0] * lattice.coord_num) * 1
        pos = lax.cond(
            spin_ind == 0,
            lambda x: elec_pos_up[x // lattice.coord_num],
            lambda x: elec_pos_dn[
                (x - elec_pos_up.shape[0] * lattice.coord_num) // lattice.coord_num
            ],
            new_ind,
        )
        neighbor_ind = new_ind % lattice.coord_num
        neighbor_pos = lattice.get_nearest_neighbors(pos)[neighbor_ind]
        walker = lax.cond(
            ee, lambda x: walker.at[(spin_ind, *pos)].set(0), lambda x: walker, 0
        )
        walker = lax.cond(
            ee,
            lambda x: walker.at[(spin_ind, *neighbor_pos)].set(1),
            lambda x: walker,
            0,
        )

        eph = 1 - ee
        ph_idx = new_ind - (self.n_elec[0] + self.n_elec[1]) * lattice.coord_num
        pos = jnp.array(lattice.sites)[ph_idx % lattice.n_sites]
        ph_change = (ph_idx // lattice.n_sites == 0) * 1 - (
            ph_idx // lattice.n_sites == 1
        ) * 1
        walker = lax.cond(
            eph,
            lambda x: walker.at[(2, *pos)].add(ph_change),
            lambda x: walker,
            0,
        )

        walker = jnp.where(walker < 0, 0, walker)

        # jax.debug.print("new_walker: {}", walker)

        energy = jnp.array(jnp.where(jnp.isnan(energy), 0.0, energy))
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        return (
            energy,
            qp_weight,
            overlap_gradient,
            weight,
            walker,
            jnp.exp(overlap),
        )

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update_lr(
        self,
        walker: jax.Array,
        parameters: Any,
        wave: t_projected_state,
        lattice: Any,
        random_number: float,
        parameters_copy: Any,
    ) -> Tuple:
        walker_data = wave.build_walker_data(walker, parameters, lattice)
        elec_pos_up = jnp.array(lattice.sites)[
            jnp.nonzero(walker[0].reshape(-1), size=self.n_elec[0])[0]
        ]
        elec_pos_dn = jnp.array(lattice.sites)[
            jnp.nonzero(walker[1].reshape(-1), size=self.n_elec[1])[0]
        ]
        phonon_occ = walker[2]

        # diagonal
        energy = self.diagonal_energy(walker) + 0.0j

        overlap = wave.calc_overlap(walker_data, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)
        prob = (
            jnp.abs(jnp.exp(overlap)) ** 2
            + (jnp.abs(overlap_gradient * jnp.exp(overlap)) ** 2).sum()
        )
        overlaps = jnp.exp(walker_data["log_overlaps"])
        overlap_0 = jnp.sum(overlaps)
        k_factors = jnp.exp(-jnp.array(wave.symm_factors))
        spin_i = (walker[0] - walker[1]).reshape(-1)
        charge_i = (walker[0] + walker[1]).reshape(-1)
        spin_q = jnp.sum(k_factors * spin_i) / jnp.sqrt(lattice.n_sites)
        charge_q = jnp.sum(k_factors * charge_i) / jnp.sqrt(lattice.n_sites)
        vector_property = (
            jnp.array([spin_q, charge_q, 1.0]) * overlap_0 / jnp.exp(overlap)
        )  # these are < w | S_q | psi_0 > / < w | psi_q >, < w | N_q | psi_0 > / < w | psi_q >, and < w | psi_0 > / < w | psi_q >

        # electron hops
        # scan over neighbors of elec_pos
        # carry: [ energy, spin, elec_pos, occ_idx, elec_pos ]
        def scanned_fun(carry, x):
            neighbor, neighbor_edge_bond = x
            neighbor_site_num = lattice.get_site_num(neighbor)
            excitation_ee = {}
            excitation_ee["sm_idx"] = jnp.array((carry[1], carry[3], neighbor_site_num))
            excitation_ee["idx"] = jnp.array((carry[1], carry[2], neighbor_site_num))
            excitation_ph = jnp.array((0, 0))
            excitation = {"ee": excitation_ee, "ph": excitation_ph}
            ratio = (walker[carry[1]][(*neighbor,)] == 0) * wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            carry[0] -= ratio * (1 - 2 * neighbor_edge_bond * self.antiperiodic)
            new_walker = walker.at[(carry[1], *carry[4])].set(0)
            new_walker = new_walker.at[(carry[1], *neighbor)].set(1)
            new_walker_data = wave.build_walker_data(
                new_walker, parameters_copy, lattice
            )
            new_overlap_gradient = (
                walker[carry[1]][(*neighbor,)] == 0
            ) * wave.calc_overlap_gradient(new_walker_data, parameters_copy, lattice)
            new_overlap = ratio * jnp.exp(overlap)
            new_prob = (walker[carry[1]][(*neighbor,)] == 0) * (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio = (
                new_prob / prob
            )  # taking square root here inexplicably leads to nans in local energy gradients, even though it has no bearing on local energy
            return carry, prob_ratio

        # scan over electrons
        # carry: [ energy, spin, occ_idx ]
        def outer_scanned_fun(carry, x):
            [carry[0], _, _, _, _], hop_ratios = lax.scan(
                scanned_fun,
                [carry[0], carry[1], lattice.get_site_num(x), carry[2], x],
                (
                    lattice.get_nearest_neighbors(x),
                    lattice.get_nearest_neighbors_edge_bond(x),
                ),
            )
            carry[2] += 1
            return carry, hop_ratios

        [energy, _, _], ratios_up = lax.scan(
            outer_scanned_fun, [energy, 0, 0], elec_pos_up
        )
        [energy, _, _], ratios_dn = lax.scan(
            outer_scanned_fun, [energy, 1, 0], elec_pos_dn
        )
        hop_ratios = jnp.concatenate((ratios_up.reshape(-1), ratios_dn.reshape(-1)))

        # eph
        excitation_ee = {}
        excitation_ee["sm_idx"] = jnp.array((2, 0, 0))
        excitation_ee["idx"] = jnp.array((2, 0, 0))
        excitation = {}
        excitation["ee"] = excitation_ee

        # scan over lattice sites
        # carry: [ energy, site_idx ]
        def scanned_fun_ph(carry, x):
            # add phonon
            excitation["ph"] = jnp.array((carry[1], 1))
            overlap_ratio = wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            ratio_0 = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * ((walker[0][(*x,)] + walker[1][(*x,)]) > 0)
                * overlap_ratio
                / (phonon_occ[(*x,)] + 1) ** 0.5
            )
            carry[0] -= (
                self.g
                * (phonon_occ[(*x,)] + 1) ** 0.5
                * ratio_0
                * (walker[0][(*x,)] + walker[1][(*x,)])
            )
            new_walker = walker.at[(2, *x)].add(1)
            new_walker_data = wave.build_walker_data(
                new_walker, parameters_copy, lattice
            )
            new_overlap_gradient = (
                jnp.sum(phonon_occ) < self.max_n_phonons
            ) * wave.calc_overlap_gradient(new_walker_data, parameters_copy, lattice)
            new_overlap = ratio_0 * jnp.exp(overlap)
            new_prob = (jnp.sum(phonon_occ) < self.max_n_phonons) * (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_0 = new_prob / prob / (phonon_occ[(*x,)] + 1)

            # remove phonon
            excitation["ph"] = jnp.array((carry[1], -1))
            overlap_ratio = wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            ratio_1 = (
                (phonon_occ[(*x,)] > 0)
                * ((walker[0][(*x,)] + walker[1][(*x,)]) > 0)
                * overlap_ratio
                * phonon_occ[(*x,)] ** 0.5
            )
            carry[0] -= (
                self.g
                * phonon_occ[(*x,)] ** 0.5
                * ratio_1
                * (walker[0][(*x,)] + walker[1][(*x,)])
            )
            new_walker = walker.at[(2, *x)].add(-1)
            new_walker_data = wave.build_walker_data(
                new_walker, parameters_copy, lattice
            )
            new_overlap_gradient = (phonon_occ[(*x,)] > 0) * wave.calc_overlap_gradient(
                new_walker_data, parameters_copy, lattice
            )
            new_overlap = ratio_1 * jnp.exp(overlap)
            new_prob = (phonon_occ[(*x,)] > 0) * (
                jnp.abs(new_overlap) ** 2.0
                + (jnp.abs(new_overlap_gradient * new_overlap) ** 2.0).sum()
            )
            prob_ratio_1 = new_prob / prob * phonon_occ[(*x,)]
            carry[1] += 1
            return carry, (prob_ratio_0, prob_ratio_1)

        [energy, _], (ratios_p, ratios_m) = lax.scan(
            scanned_fun_ph, [energy, 0], jnp.array(lattice.sites)
        )

        ratios = jnp.concatenate((hop_ratios, ratios_p, ratios_m))

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios) ** 0.5)  # moved square root here
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print("\nwalker: {}", walker_data)
        # jax.debug.print("overlap: {}", overlap)
        # jax.debug.print("grad: {}", overlap_gradient)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}\n", new_ind)

        # update
        # NB: some of these operations rely on jax not complaining about out of bounds array access
        ee = new_ind < (self.n_elec[0] + self.n_elec[1]) * lattice.coord_num
        spin_ind = (new_ind >= self.n_elec[0] * lattice.coord_num) * 1
        pos = lax.cond(
            spin_ind == 0,
            lambda x: elec_pos_up[x // lattice.coord_num],
            lambda x: elec_pos_dn[
                (x - elec_pos_up.shape[0] * lattice.coord_num) // lattice.coord_num
            ],
            new_ind,
        )
        neighbor_ind = new_ind % lattice.coord_num
        neighbor_pos = lattice.get_nearest_neighbors(pos)[neighbor_ind]
        walker = lax.cond(
            ee, lambda x: walker.at[(spin_ind, *pos)].set(0), lambda x: walker, 0
        )
        walker = lax.cond(
            ee,
            lambda x: walker.at[(spin_ind, *neighbor_pos)].set(1),
            lambda x: walker,
            0,
        )

        eph = 1 - ee
        ph_idx = new_ind - (self.n_elec[0] + self.n_elec[1]) * lattice.coord_num
        pos = jnp.array(lattice.sites)[ph_idx % lattice.n_sites]
        ph_change = (ph_idx // lattice.n_sites == 0) * 1 - (
            ph_idx // lattice.n_sites == 1
        ) * 1
        walker = lax.cond(
            eph,
            lambda x: walker.at[(2, *pos)].add(ph_change),
            lambda x: walker,
            0,
        )

        walker = jnp.where(walker < 0, 0, walker)

        # jax.debug.print("new_walker: {}", walker)

        energy = jnp.array(jnp.where(jnp.isnan(energy), 0.0, energy))
        energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        return (
            energy,
            vector_property,
            overlap_gradient,
            weight,
            walker,
            jnp.exp(overlap),
        )

    def __hash__(self):
        return hash(
            (self.omega, self.g, self.u, self.n_orbs, self.n_elec, self.max_n_phonons)
        )


@dataclass
class holstein_spinless:
    """Hubbard-Holstein model Hamiltonian

    Attributes
    ----------
    omega : float
        Phonon frequency
    g : float
        Electron-phonon coupling
    n_orbs : int
        Number of orbitals
    n_elec : int
        Number of electrons
    max_n_phonons : any, optional
        Maximum number of phonons, by default jnp.inf
    antiperiodic : bool, optional
        Antiperiodic boundary conditions, by default False
    """

    omega: float
    g: float
    n_orbs: int
    n_elec: int
    max_n_phonons: any = jnp.inf
    antiperiodic: bool = False

    @partial(jit, static_argnums=(0,))
    def diagonal_energy(self, walker: Sequence) -> jax.Array:
        """Calculate the diagonal energy of a walker

        Parameters
        ----------
        walker: Sequence
            Walker

        Returns
        -------
        float
            Diagonal energy
        """
        phonon_energy = self.omega * jnp.sum(walker[1])
        return phonon_energy

    @partial(jit, static_argnums=(0, 3, 4))
    def local_energy_and_update(
        self,
        walker: jax.Array,
        parameters: Any,
        wave: Any,
        lattice: Any,
        random_number: float,
    ) -> Tuple:
        """Calculate the local energy of a walker and update it

        Parameters
        ----------
        walker : Sequence
            walker : [ elec_occ_up, elec_occ_dn, phonon_occ ]
        parameters : Any
            Parameters of the wavefunction
        wave : Any
            Wavefunction
        lattice : Any
            Lattice
        random_number : float
            Random number used for MC update

        Returns
        -------
        Tuple
            Tuple of local energy, qp_weight (0. here), overlap_gradient_ratio, weight, updated_walker, overlap
        """
        walker_data = wave.build_walker_data(walker, parameters, lattice)
        elec_pos = jnp.array(lattice.sites)[
            jnp.nonzero(walker[0].reshape(-1), size=self.n_elec)[0]
        ]
        phonon_occ = walker[1]

        # diagonal
        energy = self.diagonal_energy(walker) + 0.0j

        overlap = wave.calc_overlap(walker_data, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)
        # qp_weight = (jnp.sum(phonon_occ) == 0) * 1.0
        qp_weight = jnp.sum(walker[1])

        # electron hops
        # scan over neighbors of elec_pos
        # carry: [ energy, elec_pos, idx ]
        def scanned_fun(carry, x):
            neighbor, neighbor_edge_bond = x
            neighbor_site_num = lattice.get_site_num(neighbor)
            excitation_ee = {}
            excitation_ee["sm_idx"] = jnp.array((1, carry[2], neighbor_site_num))
            excitation_ee["idx"] = jnp.array((1, carry[1], neighbor_site_num))
            excitation_ph = jnp.array((0, 0))
            excitation = {"ee": excitation_ee, "ph": excitation_ph}
            ratio = (walker[0][(*neighbor,)] == 0) * wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            carry[0] -= ratio * (1 - 2 * neighbor_edge_bond * self.antiperiodic)
            return carry, ratio

        # scan over electrons
        # carry: [ energy, occ_idx ]
        def outer_scanned_fun(carry, x):

            [carry[0], _, _], hop_ratios = lax.scan(
                scanned_fun,
                [carry[0], lattice.get_site_num(x), carry[1]],
                (
                    lattice.get_nearest_neighbors(x),
                    lattice.get_nearest_neighbors_edge_bond(x),
                ),
            )
            carry[1] += 1
            return carry, hop_ratios

        [energy, _], ratios = lax.scan(outer_scanned_fun, [energy, 0], elec_pos)
        hop_ratios = ratios.reshape(-1)

        # eph
        # this is g (n - 1/2) (b + b^dagger)
        excitation_ee = {}
        excitation_ee["sm_idx"] = jnp.array((-1, 0, 0))
        excitation_ee["idx"] = jnp.array((-1, 0, 0))  # to turn off ee
        excitation = {}
        excitation["ee"] = excitation_ee

        # scan over lattice sites
        # carry: [ energy, site_idx ]
        def scanned_fun_ph(carry, x):
            # add phonon
            excitation["ph"] = jnp.array((carry[1], 1))
            overlap_ratio = wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            ratio_0 = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * (walker[0][(*x,)] > -1)
                * overlap_ratio
                / (phonon_occ[(*x,)] + 1) ** 0.5
            )
            carry[0] -= (
                self.g
                * (phonon_occ[(*x,)] + 1) ** 0.5
                * ratio_0
                * (walker[0][(*x,)] - 0.5)
            )

            # remove phonon
            excitation["ph"] = jnp.array((carry[1], -1))
            overlap_ratio = wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            ratio_1 = (
                (phonon_occ[(*x,)] > 0)
                * (walker[0][(*x,)] > -1)
                * overlap_ratio
                * phonon_occ[(*x,)] ** 0.5
            )
            carry[0] -= (
                self.g * phonon_occ[(*x,)] ** 0.5 * ratio_1 * (walker[0][(*x,)] - 0.5)
            )
            carry[1] += 1
            return carry, (ratio_0, ratio_1)

        [energy, _], (ratios_p, ratios_m) = lax.scan(
            scanned_fun_ph, [energy, 0], jnp.array(lattice.sites)
        )

        ratios = jnp.concatenate((hop_ratios, ratios_p, ratios_m))

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        # jax.debug.print("\nwalker: {}", walker_data)
        # jax.debug.print("overlap: {}", overlap)
        # # jax.debug.print("grad: {}", overlap_gradient)
        # jax.debug.print("ratios: {}", ratios)
        # jax.debug.print("energy: {}", energy)
        # jax.debug.print("weight: {}", weight)
        # jax.debug.print("random_number: {}", random_number)
        # jax.debug.print("new_ind: {}\n", new_ind)

        # update
        # NB: some of these operations rely on jax not complaining about out of bounds array access
        ee = new_ind < (self.n_elec * lattice.coord_num)
        pos = elec_pos[new_ind // lattice.coord_num]
        neighbor_ind = new_ind % lattice.coord_num
        neighbor_pos = lattice.get_nearest_neighbors(pos)[neighbor_ind]
        walker = lax.cond(
            ee, lambda x: walker.at[(0, *pos)].set(0), lambda x: walker, 0
        )
        walker = lax.cond(
            ee,
            lambda x: walker.at[(0, *neighbor_pos)].set(1),
            lambda x: walker,
            0,
        )

        eph = 1 - ee
        ph_idx = new_ind - (self.n_elec * lattice.coord_num)
        pos = jnp.array(lattice.sites)[ph_idx % lattice.n_sites]
        ph_change = (ph_idx // lattice.n_sites == 0) * 1 - (
            ph_idx // lattice.n_sites == 1
        ) * 1
        walker = lax.cond(
            eph,
            lambda x: walker.at[(1, *pos)].add(ph_change),
            lambda x: walker,
            0,
        )

        walker = jnp.where(walker < 0, 0, walker)

        # jax.debug.print("new_walker: {}", walker)

        # energy = jnp.array(jnp.where(jnp.isnan(energy), 0.0, energy))
        # energy = jnp.where(jnp.isinf(energy), 0.0, energy)
        # weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        # weight = jnp.where(jnp.isinf(weight), 0.0, weight)

        return (
            energy,
            qp_weight,
            overlap_gradient,
            weight,
            walker,
            jnp.exp(overlap),
        )

    def __hash__(self):
        return hash((self.omega, self.g, self.n_orbs, self.n_elec, self.max_n_phonons))
