import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence, Tuple

# os.environ['JAX_ENABLE_X64'] = 'True'
import jax
from flax import linen as nn
from jax import grad, jit, lax
from jax import numpy as jnp
from jax import random, tree_util, value_and_grad, vmap


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
        return self.u * jnp.sum(walker[0] * walker[1])

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
            Tuple of local energy, qp_weight (0. here), overlap_gradient_ratio, weight, updated_walker, overlap
        """
        walker_data = wave.build_walker_data(walker, parameters, lattice)
        elec_pos_up = jnp.array(lattice.sites)[
            jnp.nonzero(walker[0].reshape(-1), size=self.n_elec[0])[0]
        ]
        elec_pos_dn = jnp.array(lattice.sites)[
            jnp.nonzero(walker[1].reshape(-1), size=self.n_elec[1])[0]
        ]

        # diagonal
        energy = self.diagonal_energy(walker) + 0.0j

        overlap = wave.calc_overlap(walker_data, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)
        qp_weight = 0.0

        # electron hops
        # scan over neighbors of elec_pos
        # carry: [ energy, spin, elec_pos, idx ]
        def scanned_fun(carry, x):
            neighbor_site_num = lattice.get_site_num(x)
            excitation = jnp.array((carry[1], carry[3], neighbor_site_num))
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

        # jax.debug.print("\nwalker: {}", walker_data)
        # jax.debug.print("overlap: {}", overlap)
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
        walker = walker.at[spin_ind, (*pos,)].set(0)
        walker = walker.at[spin_ind, (*neighbor_pos,)].set(1)

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
        coulomb_energy = self.u * jnp.sum(walker[0][0] * walker[0][1])
        phonon_energy = self.omega * jnp.sum(walker[1])
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
            walker : [ (elec_occ_up, elec_occ_dn), phonon_occ ]
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
            jnp.nonzero(walker[0][0].reshape(-1), size=self.n_elec[0])[0]
        ]
        elec_pos_dn = jnp.array(lattice.sites)[
            jnp.nonzero(walker[0][1].reshape(-1), size=self.n_elec[1])[0]
        ]
        phonon_occ = walker[1]

        # diagonal
        energy = self.diagonal_energy(walker) + 0.0j

        overlap = wave.calc_overlap(walker_data, parameters, lattice)
        overlap_gradient = wave.calc_overlap_gradient(walker_data, parameters, lattice)
        qp_weight = (jnp.sum(phonon_occ) == 0) * 1.0

        # electron hops
        # scan over neighbors of elec_pos
        # carry: [ energy, spin, elec_pos, idx ]
        def scanned_fun(carry, x):
            neighbor_site_num = lattice.get_site_num(x)
            excitation = jnp.array((carry[1], carry[3], neighbor_site_num))
            ratio = (walker[0][carry[1]][(*x,)] == 0) * wave.calc_overlap_ratio(
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
        walker_data_copy = walker_data.copy()
        elec_occ_up = walker[0][0]
        elec_occ_dn = walker[0][1]

        # scan over lattice sites
        # carry: energy
        def scanned_fun_ph(carry, x):
            # add phonon
            new_phonon_occ = phonon_occ.at[(*x,)].add(1)
            walker_data_copy["phonon_occ"] = new_phonon_occ
            new_overlap = wave.calc_overlap(walker_data_copy, parameters, lattice)
            ratio_0 = (
                (jnp.sum(phonon_occ) < self.max_n_phonons)
                * jnp.exp(new_overlap - overlap)
                / (phonon_occ[(*x,)] + 1) ** 0.5
            )
            carry -= (
                self.g
                * (phonon_occ[(*x,)] + 1) ** 0.5
                * ratio_0
                * (elec_occ_up[(*x,)] + elec_occ_dn[(*x,)])
            )

            # remove phonon
            new_phonon_occ = phonon_occ.at[(*x,)].add(-1)
            new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
            walker_data_copy["phonon_occ"] = new_phonon_occ
            new_overlap = wave.calc_overlap(walker_data_copy, parameters, lattice)
            ratio_1 = (
                (phonon_occ[(*x,)] > 0)
                * jnp.exp(new_overlap - overlap)
                * phonon_occ[(*x,)] ** 0.5
            )
            carry -= (
                self.g
                * phonon_occ[(*x,)] ** 0.5
                * ratio_1
                * (elec_occ_up[(*x,)] + elec_occ_dn[(*x,)])
            )
            return carry, (ratio_0, ratio_1)

        energy, (ratios_p, ratios_m) = lax.scan(
            scanned_fun_ph, energy, jnp.array(lattice.sites)
        )

        ratios = jnp.concatenate((hop_ratios, ratios_p, ratios_m))

        cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
        weight = 1 / cumulative_ratios[-1]
        new_ind = jnp.searchsorted(
            cumulative_ratios, random_number * cumulative_ratios[-1]
        )

        jax.debug.print("\nwalker: {}", walker_data)
        jax.debug.print("overlap: {}", overlap)
        jax.debug.print("ratios: {}", ratios)
        jax.debug.print("energy: {}", energy)
        jax.debug.print("weight: {}", weight)
        jax.debug.print("random_number: {}", random_number)
        jax.debug.print("new_ind: {}\n", new_ind)

        return 1

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
        walker = walker.at[spin_ind, (*pos,)].set(0)
        walker = walker.at[spin_ind, (*neighbor_pos,)].set(1)

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
