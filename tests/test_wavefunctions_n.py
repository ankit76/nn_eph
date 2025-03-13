from nn_eph import config

config.setup_jax()

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random, tree_util

from nn_eph import hamiltonians_n, lattices, models, wavefunctions_n

l_x, l_y = 3, 3
lattice = lattices.two_dimensional_grid(l_x, l_y)
n_sites = lattice.n_sites

n_elec = (4, 6)
walker = jnp.array(
    [[[1, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 1, 1], [1, 1, 1], [0, 0, 1]]]
)
excitation = {}
excitation["sm_idx"] = jnp.array([0, 1, 5])
excitation["idx"] = jnp.array([0, 1, 5])
new_walker = wavefunctions_n.apply_excitation_ee(walker, excitation, lattice)


def test_uhf():
    np.random.seed(14)
    parameters = [
        jnp.array(np.random.randn(n_sites, n_elec[0]) + 0.0j),
        jnp.array(np.random.randn(n_sites, n_elec[1]) + 0.0j),
    ]
    n_parameters = np.sum([parameters_i.size for parameters_i in parameters])
    wave = wavefunctions_n.uhf(n_elec, n_parameters)
    walker_data = wave.build_walker_data(walker, parameters, lattice)
    overlap = wave.calc_overlap(walker_data, parameters, lattice)
    new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
    overlap_new = wave.calc_overlap(new_walker_data, parameters, lattice)
    overlap_ratio = wave.calc_overlap_ratio(
        walker_data, excitation, parameters, lattice
    )
    parity = wave.calc_parity(walker_data, excitation, lattice)
    assert np.allclose(parity * overlap_ratio, jnp.exp(overlap_new - overlap))


def test_kuhf():
    np.random.seed(14)
    parameters = [
        jnp.array(np.random.randn(n_sites, n_elec[0]) + 0.0j),
        jnp.array(np.random.randn(n_sites, n_elec[0]) + 0.0j),
        jnp.array(np.random.randn(n_sites, n_elec[1]) + 0.0j),
        jnp.array(np.random.randn(n_sites, n_elec[1]) + 0.0j),
    ]
    n_parameters = np.sum([parameters_i.size for parameters_i in parameters])
    wave = wavefunctions_n.kuhf(n_elec, n_parameters)
    walker_data = wave.build_walker_data(walker, parameters, lattice)
    overlap = wave.calc_overlap(walker_data, parameters, lattice)
    new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
    overlap_new = wave.calc_overlap(new_walker_data, parameters, lattice)
    overlap_ratio = wave.calc_overlap_ratio(
        walker_data, excitation, parameters, lattice
    )
    parity = wave.calc_parity(walker_data, excitation, lattice)
    assert np.allclose(parity * overlap_ratio, jnp.exp(overlap_new - overlap))


def test_ghf():
    np.random.seed(14)
    parameters = [jnp.array(np.random.randn(2 * n_sites, sum(n_elec)) + 0.0j)]
    n_parameters = np.sum([parameters_i.size for parameters_i in parameters])
    wave = wavefunctions_n.ghf(n_elec, n_parameters)
    walker_data = wave.build_walker_data(walker, parameters, lattice)
    overlap = wave.calc_overlap(walker_data, parameters, lattice)
    new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
    overlap_new = wave.calc_overlap(new_walker_data, parameters, lattice)
    overlap_ratio = wave.calc_overlap_ratio(
        walker_data, excitation, parameters, lattice
    )
    parity = wave.calc_parity(walker_data, excitation, lattice)
    assert np.allclose(parity * overlap_ratio, jnp.exp(overlap_new - overlap))


def test_kghf():
    np.random.seed(14)
    parameters = [
        jnp.array(np.random.randn(2 * n_sites, sum(n_elec)) + 0.0j),
        jnp.array(np.random.randn(2 * n_sites, sum(n_elec)) + 0.0j),
    ]
    n_parameters = np.sum([parameters_i.size for parameters_i in parameters])
    wave = wavefunctions_n.kghf(n_elec, n_parameters)
    walker_data = wave.build_walker_data(walker, parameters, lattice)
    overlap = wave.calc_overlap(walker_data, parameters, lattice)
    new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
    overlap_new = wave.calc_overlap(new_walker_data, parameters, lattice)
    overlap_ratio = wave.calc_overlap_ratio(
        walker_data, excitation, parameters, lattice
    )
    parity = wave.calc_parity(walker_data, excitation, lattice)
    assert np.allclose(parity * overlap_ratio, jnp.exp(overlap_new - overlap))


def test_product_state():
    np.random.seed(14)
    ref_parameters = [jnp.array(np.random.randn(2 * n_sites, sum(n_elec)) + 0.0j)]
    n_ref_parameters = np.sum([parameters_i.size for parameters_i in ref_parameters])
    ref_wave = wavefunctions_n.ghf(n_elec, n_ref_parameters)

    model_j = models.MLP(
        [10, 1],
        param_dtype=jnp.complex64,
        kernel_init=models.complex_kernel_init,
        activation=jax.nn.relu,
    )
    model_input = jnp.zeros((2 * n_sites))
    nn_parameters = model_j.init(random.PRNGKey(891), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
    nn_j = wavefunctions_n.nn(
        model_j.apply, n_nn_parameters, wavefunctions_n.apply_excitation_ee
    )

    parameters = [nn_parameters, ref_parameters]
    wave = wavefunctions_n.product_state((nn_j, ref_wave))

    walker_data = wave.build_walker_data(walker, parameters, lattice)
    overlap = wave.calc_overlap(walker_data, parameters, lattice)
    new_walker_data = wave.build_walker_data(new_walker, parameters, lattice)
    overlap_new = wave.calc_overlap(new_walker_data, parameters, lattice)
    overlap_ratio = wave.calc_overlap_ratio(
        walker_data, excitation, parameters, lattice
    )
    parity = wave.calc_parity(walker_data, excitation, lattice)
    assert np.allclose(parity * overlap_ratio, jnp.exp(overlap_new - overlap))


def test_t_projected_state():
    np.random.seed(14)
    n_sites = 6
    lattice = lattices.one_dimensional_chain(n_sites)

    n_elec = (3, 3)
    walker = jnp.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
    excitation = {}
    excitation["sm_idx"] = jnp.array([0, 1, 5])
    excitation["idx"] = jnp.array([0, 2, 5])

    ref_parameters = [jnp.array(np.random.randn(2 * n_sites, sum(n_elec)) + 0.0j)]
    n_ref_parameters = np.sum([parameters_i.size for parameters_i in ref_parameters])
    ref_wave = wavefunctions_n.ghf(n_elec, n_ref_parameters)

    model_j = models.MLP(
        [10, 1],
        param_dtype=jnp.complex64,
        kernel_init=models.complex_kernel_init,
        activation=jax.nn.relu,
    )
    model_input = jnp.zeros((2 * n_sites))
    nn_parameters = model_j.init(random.PRNGKey(891), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
    nn_j = wavefunctions_n.nn(
        model_j.apply, n_nn_parameters, wavefunctions_n.apply_excitation_ee
    )

    parameters = [nn_parameters, ref_parameters]
    wave = wavefunctions_n.product_state((nn_j, ref_wave))

    wave_t = wavefunctions_n.t_projected_state(
        wave,
        wavefunctions_n.walker_translator,
        wavefunctions_n.excitation_translator_ee,
    )

    walker_data = wave_t.build_walker_data(walker, parameters, lattice)
    _ = wave_t.calc_overlap(walker_data, parameters, lattice)
    _ = wave_t.calc_overlap_ratio(walker_data, excitation, parameters, lattice)


if __name__ == "__main__":
    test_uhf()
    test_kuhf()
    test_ghf()
    test_kghf()
    test_product_state()
    test_t_projected_state()
