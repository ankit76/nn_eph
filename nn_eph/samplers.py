import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

# os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, lax
from jax import numpy as jnp
from jax import random, tree_util


@dataclass
class continuous_time_1:
    n_eql: int
    n_samples: int

    @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(self, walker, ham, parameters, wave, lattice, random_numbers):
        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight, dev_thresh, median_energy ]
        def scanned_fun(carry, x):
            (
                energy,
                qp_weight,
                gradient,
                weight,
                carry[0],
                _,
            ) = ham.local_energy_and_update(
                carry[0], parameters, wave, lattice, random_numbers[x]
            )
            weight = (jnp.abs(energy - carry[7]) < carry[6]) * weight + 1.0e-8
            energy = jnp.where(weight > 1.0e-8, energy, carry[7])
            carry[1] += weight
            carry[2] += weight * (jnp.real(energy) - carry[2]) / carry[1]
            carry[3] = carry[3] + weight * (jnp.real(gradient) - carry[3]) / carry[1]
            carry[4] = (
                carry[4]
                + weight
                * (jnp.real(jnp.conjugate(energy) * gradient) - carry[4])
                / carry[1]
            )
            carry[5] += weight * (jnp.real(qp_weight) - carry[5]) / carry[1]
            return carry, (jnp.real(energy), qp_weight, weight, carry[0])

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        [walker, _, _, _, _, _, _, _], (energies_eq, _, _, _) = lax.scan(
            scanned_fun,
            [walker, weight, energy, gradient, lene_gradient, qp_weight, jnp.inf, 0.0],
            jnp.arange(self.n_eql),
        )

        median_energy = jnp.median(energies_eq)
        d = jnp.abs(energies_eq - median_energy)
        mdev = jnp.median(d)
        mdev = jnp.where(mdev == 0.0, 1.0, mdev)

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        [_, weight, energy, gradient, lene_gradient, qp_weight, _, _], (
            energies,
            qp_weights,
            weights,
            walkers,
        ) = lax.scan(
            scanned_fun,
            [
                walker,
                weight,
                energy,
                gradient,
                lene_gradient,
                qp_weight,
                1000000.0 * mdev,
                median_energy,
            ],
            jnp.arange(self.n_samples),
        )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            energies,
            qp_weights,
            weights,
            walkers,
        )

    def __hash__(self):
        return hash((self.n_eql, self.n_samples))


@dataclass
class continuous_time:
    n_eql: int
    n_samples: int

    @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(
        self,
        walker,
        ham,
        parameters,
        wave,
        lattice,
        random_numbers,
        dev_thresh_fac=100.0,
    ):
        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight, dev_thresh, median_energy ]
        def scanned_fun(carry, x):
            (
                energy,
                qp_weight,
                gradient,
                weight,
                carry[0],
                _,
            ) = ham.local_energy_and_update(
                carry[0], parameters, wave, lattice, random_numbers[x]
            )
            weight = (jnp.abs(energy - carry[7]) < carry[6]) * weight + 1.0e-8
            energy = jnp.where(weight > 1.0e-8, energy, carry[7])
            carry[1] += weight
            carry[2] += weight * (jnp.real(energy) - carry[2]) / carry[1]
            carry[3] = carry[3] + weight * (jnp.real(gradient) - carry[3]) / carry[1]
            carry[4] = (
                carry[4]
                + weight
                * (jnp.real(jnp.conjugate(energy) * gradient) - carry[4])
                / carry[1]
            )
            carry[5] += weight * (jnp.real(qp_weight) - carry[5]) / carry[1]
            return carry, (jnp.real(energy), qp_weight, weight)

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        [walker, _, _, _, _, _, _, _], (energies_eq, _, _) = lax.scan(
            scanned_fun,
            [walker, weight, energy, gradient, lene_gradient, qp_weight, jnp.inf, 0.0],
            jnp.arange(self.n_eql),
        )

        median_energy = jnp.median(energies_eq)
        d = jnp.abs(energies_eq - median_energy)
        mdev = jnp.median(d)
        mdev = jnp.where(mdev == 0.0, 1.0, mdev)

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        [_, weight, energy, gradient, lene_gradient, qp_weight, _, _], (
            energies,
            qp_weights,
            weights,
        ) = lax.scan(
            scanned_fun,
            [
                walker,
                weight,
                energy,
                gradient,
                lene_gradient,
                qp_weight,
                dev_thresh_fac * mdev,
                median_energy,
            ],
            jnp.arange(self.n_samples),
        )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            energies,
            qp_weights,
            weights,
        )

    def __hash__(self):
        return hash((self.n_eql, self.n_samples))


@dataclass
class continuous_time_sr:
    n_eql: int
    n_samples: int

    @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(self, walker, ham, parameters, wave, lattice, random_numbers):
        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight, metric ]
        def scanned_fun(carry, x):
            (
                energy,
                qp_weight,
                gradient,
                weight,
                carry[0],
                _,
            ) = ham.local_energy_and_update(
                carry[0], parameters, wave, lattice, random_numbers[x]
            )
            carry[1] += weight
            carry[2] += weight * (jnp.real(energy) - carry[2]) / carry[1]
            carry[3] = carry[3] + weight * (jnp.real(gradient) - carry[3]) / carry[1]
            carry[4] = (
                carry[4]
                + weight
                * (jnp.real(jnp.conjugate(energy) * gradient) - carry[4])
                / carry[1]
            )
            carry[5] += weight * (jnp.real(qp_weight) - carry[5]) / carry[1]
            carry[6] += (
                weight
                * (
                    jnp.real(jnp.einsum("i,j->ij", jnp.conj(gradient), gradient))
                    - carry[6]
                )
                / carry[1]
            )
            return carry, (jnp.real(energy), qp_weight, weight)

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        metric = jnp.zeros((wave.n_parameters, wave.n_parameters))
        [walker, _, _, _, _, _, _], (_, _, _) = lax.scan(
            scanned_fun,
            [walker, weight, energy, gradient, lene_gradient, qp_weight, metric],
            jnp.arange(self.n_eql),
        )

        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        metric = jnp.zeros((wave.n_parameters, wave.n_parameters))
        [_, weight, energy, gradient, lene_gradient, qp_weight, metric], (
            energies,
            qp_weights,
            weights,
        ) = lax.scan(
            scanned_fun,
            [walker, weight, energy, gradient, lene_gradient, qp_weight, metric],
            jnp.arange(self.n_samples),
        )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            metric,
            energies,
            qp_weights,
            weights,
        )

    def __hash__(self):
        return hash((self.n_eql, self.n_samples))


@dataclass
class deterministic:
    max_n_phonon: int
    basis: Sequence = Any
    n_samples: int = 1

    def __init__(self, max_n_phonon, lattice):
        self.max_n_phonon = max_n_phonon
        self.basis = lattice.make_polaron_basis(max_n_phonon)

    # @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(self, walker, ham, parameters, wave, lattice, random_numbers):
        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight ]
        # def scanned_fun(carry, x):
        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        energies = jnp.zeros(len(self.basis))
        qp_weights = jnp.zeros(len(self.basis))
        weights = jnp.zeros(len(self.basis))
        for i, x in enumerate(self.basis):
            (
                energy_x,
                qp_weight_x,
                gradient_x,
                _,
                _,
                overlap_x,
            ) = ham.local_energy_and_update(x, parameters, wave, lattice, 0.0)
            weight_x = jnp.real(overlap_x.conj() * overlap_x)
            weights = weights.at[i].set(weight_x)
            weight += weight_x
            energy += weight_x * (jnp.real(energy_x) - energy) / weight
            energies = energies.at[i].set(jnp.real(energy_x))
            gradient = gradient + weight_x * (jnp.real(gradient_x) - gradient) / weight
            lene_gradient = (
                lene_gradient
                + weight_x
                * (jnp.real(jnp.conjugate(energy_x) * gradient_x) - lene_gradient)
                / weight
            )
            qp_weight += weight_x * (qp_weight_x - qp_weight) / weight
            qp_weights = qp_weights.at[i].set(qp_weight_x)

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            energies,
            qp_weights,
            weights,
        )

    def __hash__(self):
        return 0


@dataclass
class deterministic_sr:
    max_n_phonon: int
    basis: Sequence = Any
    n_samples: int = 1

    def __init__(self, max_n_phonon, lattice):
        self.max_n_phonon = max_n_phonon
        self.basis = lattice.make_polaron_basis(max_n_phonon)

    # @partial(jit, static_argnums=(0, 2, 4, 5))
    def sampling(self, walker, ham, parameters, wave, lattice, random_numbers):
        # carry : [ walker, weight, energy, grad, lene_grad, qp_weight ]
        # def scanned_fun(carry, x):
        weight = 0.0
        energy = 0.0
        gradient = jnp.zeros(wave.n_parameters)
        lene_gradient = jnp.zeros(wave.n_parameters)
        qp_weight = 0.0
        energies = jnp.zeros(len(self.basis))
        qp_weights = jnp.zeros(len(self.basis))
        weights = jnp.zeros(len(self.basis))
        metric = jnp.zeros((wave.n_parameters, wave.n_parameters))
        for i, x in enumerate(self.basis):
            (
                energy_x,
                qp_weight_x,
                gradient_x,
                _,
                _,
                overlap_x,
            ) = ham.local_energy_and_update(x, parameters, wave, lattice, 0.0)
            weight_x = jnp.real(overlap_x.conj() * overlap_x)
            weights = weights.at[i].set(weight_x)
            weight += weight_x
            energy += weight_x * (jnp.real(energy_x) - energy) / weight
            energies = energies.at[i].set(jnp.real(energy_x))
            gradient = gradient + weight_x * (jnp.real(gradient_x) - gradient) / weight
            lene_gradient = (
                lene_gradient
                + weight_x
                * (jnp.real(jnp.conjugate(energy_x) * gradient_x) - lene_gradient)
                / weight
            )
            qp_weight += weight_x * (qp_weight_x - qp_weight) / weight
            qp_weights = qp_weights.at[i].set(qp_weight_x)
            metric = (
                metric
                + weight_x
                * (
                    jnp.real(jnp.einsum("i,j->ij", jnp.conj(gradient_x), gradient_x))
                    - metric
                )
                / weight
            )

        # energy, gradient, lene_gradient are weighted
        return (
            weight,
            energy,
            gradient,
            lene_gradient,
            qp_weight,
            metric,
            energies,
            qp_weights,
            weights,
        )

    def __hash__(self):
        return 0


if __name__ == "__main__":
    import hamiltonians
    import lattices
    import models
    import wavefunctions

    # l_x, l_y = 2, 2
    # n_sites = l_x * l_y
    # n_eql = 10
    # n_samples = 100
    # sampler = continuous_time(10, 100)
    # key = random.PRNGKey(0)
    # random_numbers = random.uniform(key, shape=(n_samples,))
    # ham = hamiltonians.holstein_2d(1., 1.)
    # lattice = lattices.two_dimensional_grid(l_x, l_y)
    # np.random.seed(3)
    # elec_pos = (1, 0)
    # phonon_occ = jnp.array(np.random.randint(3, size=(l_y, l_x)))
    # gamma = jnp.array(np.random.rand(len(lattice.shell_distances)))
    # model = models.MLP([5, 1])
    # model_input = jnp.zeros(2*n_sites)
    # nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
    # n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
    # parameters = [ gamma, nn_parameters ]
    # reference = wavefunctions.merrifield(gamma.size)
    # wave = wavefunctions.nn_jastrow(model.apply, reference, n_nn_parameters)
    # walker = [ elec_pos, phonon_occ ]
    # sampler.sampling(walker, ham, parameters, wave, lattice, random_numbers)

    n_sites = 2
    max_n_phonon = 2
    lattice = lattices.one_dimensional_chain(n_sites)
    sampler = deterministic(max_n_phonon, lattice)
    n_samples = 10
    key = random.PRNGKey(0)
    random_numbers = random.uniform(key, shape=(n_samples,))
    g = 1.0
    omega = 1.0
    ham = hamiltonians.holstein_1d(omega, g)
    np.random.seed(3)
    elec_pos = (0,)
    phonon_occ = jnp.array([0] * n_sites)
    gamma = jnp.array(
        [g / omega / n_sites for _ in range(n_sites // 2 + 1)]
        + [0.0 for _ in range(n_sites // 2 + 1)]
    )
    model = models.MLP([20, 1])
    model_input = jnp.zeros(2 * n_sites)
    nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
    parameters = [gamma, nn_parameters]
    reference = wavefunctions.merrifield(gamma.size)
    wave = wavefunctions.nn_jastrow(model.apply, reference, n_nn_parameters)
    walker = [elec_pos, phonon_occ]
    sampler.sampling(walker, ham, parameters, wave, lattice, random_numbers)
