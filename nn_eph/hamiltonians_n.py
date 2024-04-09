import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence, Tuple

# os.environ['JAX_ENABLE_X64'] = 'True'
import jax
from jax import jit, lax
from jax import numpy as jnp

from nn_eph.wavefunctions_n import wave_function


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
    """

    u: float
    n_orbs: int
    n_elec: Sequence
    diag_scalar_property: str = "double_occ"
    diag_vector_property: str = "spin"
    scalar_fun: Callable = None
    vector_fun: Callable = None

    def __post_init__(self):
        if self.diag_vector_property == "spin":
            self.vector_fun = self.sz
        elif self.diag_vector_property == "number":
            self.vector_fun = self.num

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
    def double_occupancy(self, walker: Sequence) -> float:
        """Calculate the double occupancy"""
        return jnp.sum(walker[0] * walker[1])

    @partial(jit, static_argnums=(0,))
    def diagonal_energy(self, walker: Sequence) -> float:
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
        walker: Sequence,
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
            neighbor_site_num = lattice.get_site_num(x)
            excitation = {}
            excitation["sm_idx"] = jnp.array((carry[1], carry[3], neighbor_site_num))
            excitation["idx"] = jnp.array((carry[1], carry[2], neighbor_site_num))
            ratio = (walker[carry[1]][(*x,)] == 0) * wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            carry[0] -= ratio
            return carry, ratio

        # scan over electrons
        # carry: [ energy, spin, occ_idx ]
        def outer_scanned_fun(carry, x):
            [carry[0], _, _, _], hop_ratios = lax.scan(
                scanned_fun,
                [carry[0], carry[1], lattice.get_site_num(x), carry[2]],
                lattice.get_nearest_neighbors(x),
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

        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
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
        walker: Sequence,
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
            neighbor_site_num = lattice.get_site_num(x)
            excitation = {}
            excitation["sm_idx"] = jnp.array((carry[1], carry[3], neighbor_site_num))
            excitation["idx"] = jnp.array((carry[1], carry[2], neighbor_site_num))
            ratio = (walker[carry[1]][(*x,)] == 0) * wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            carry[0] -= ratio
            new_walker = walker.at[(carry[1], *carry[4])].set(0)
            new_walker = new_walker.at[(carry[1], *x)].set(1)
            new_walker_data = wave.build_walker_data(
                new_walker, parameters_copy, lattice
            )
            new_overlap_gradient = (
                walker[carry[1]][(*x,)] == 0
            ) * wave.calc_overlap_gradient(new_walker_data, parameters_copy, lattice)
            new_overlap = ratio * jnp.exp(overlap)
            new_prob = (walker[carry[1]][(*x,)] == 0) * (
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
                lattice.get_nearest_neighbors(x),
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

        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
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

    @partial(jit, static_argnums=(0,))
    def diagonal_energy(self, walker: Sequence) -> float:
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
        walker: Sequence,
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
            neighbor_site_num = lattice.get_site_num(x)
            excitation_ee = {}
            excitation_ee["sm_idx"] = jnp.array((carry[1], carry[3], neighbor_site_num))
            excitation_ee["idx"] = jnp.array((carry[1], carry[2], neighbor_site_num))
            excitation_ph = jnp.array((0, 0))
            excitation = {"ee": excitation_ee, "ph": excitation_ph}
            ratio = (walker[carry[1]][(*x,)] == 0) * wave.calc_overlap_ratio(
                walker_data, excitation, parameters, lattice
            )
            carry[0] -= ratio
            return carry, ratio

        # scan over electrons
        # carry: [ energy, spin, occ_idx ]
        def outer_scanned_fun(carry, x):
            [carry[0], _, _, _], hop_ratios = lax.scan(
                scanned_fun,
                [carry[0], carry[1], lattice.get_site_num(x), carry[2]],
                lattice.get_nearest_neighbors(x),
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
        excitation = {"ee": excitation_ee}

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

        energy = jnp.where(jnp.isnan(energy), 0.0, energy)
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

    def __hash__(self):
        return hash(
            (self.omega, self.g, self.u, self.n_orbs, self.n_elec, self.max_n_phonons)
        )


if __name__ == "__main__":
    import lattices
    import models
    import numpy as np
    import wavefunctions

    # l_x, l_y, l_z = 3, 3, 3
    # n_sites = l_x * l_y * l_z
    ##n_bands = 2
    # lattice = lattices.three_dimensional_grid(l_x, l_y, l_z)

    l_x, l_y = 4, 4
    n_sites = l_x * l_y
    lattice = lattices.two_dimensional_grid(l_x, l_y)
    ham = hubbard_holstein(1.0, 1.0, 1.0, 16, (8, 8))
    np.random.seed(0)
    elec_occ_up = jnp.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 1]])
    elec_occ_dn = jnp.array([[0, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 1, 1]])
    elec_occ_dn = jnp.array(
        [[np.random.randint(2) for _ in range(l_x)] for _ in range(l_y)]
    )
    phonon_occ = jnp.array(
        [[np.random.randint(2) for _ in range(l_x)] for _ in range(l_y)]
    )
    walker = [jnp.array([elec_occ_up, elec_occ_dn]), phonon_occ]
    energy = ham.diagonal_energy(walker)
    wave = wavefunctions.ghf(16, (8, 8))
    parameters = jnp.array(np.random.rand(32, 16))
    overlap = wave.calc_overlap(walker[0], parameters, lattice)
    ham.local_energy_and_update(walker, parameters, wave, lattice, 0.9)
    # print(overlap)

    # parameters = jnp.array(np.random.rand(len(lattice.shell_distances)))
    # wave = wavefunctions.merrifield(parameters.size)
    # phonon_occ = jnp.array([[[ np.random.randint(2) for _ in range(l_x) ] for _ in range(l_y) ] for _ in range#(l_z)])
    # walker = [ (0, 0, 0), phonon_occ ]
    # random_number = 0.9
    # ham = hubbard_holstein(1., 1.)
    # energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker, #parameters, wave, lattice, random_number)
    # print(energy)

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
