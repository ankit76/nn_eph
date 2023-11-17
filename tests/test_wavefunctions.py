import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random, tree_util

from nn_eph import lattices, models, wavefunctions

seed = 3455


def test_merrifield_overlap():
    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)
    np.random.seed(seed)

    gamma = jnp.array(np.random.rand(n_sites // 2 + 1))
    wave = wavefunctions.merrifield(gamma.size)

    elec_pos = (3,)
    phonon_occ = jnp.array([1, 3, 0, 2])
    overlap = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, gamma, lattice))

    phonon_occ = jnp.array([1, 3, 0, 3])
    overlap_ex = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, gamma, lattice))
    assert np.allclose(overlap_ex, overlap * gamma[0])

    l_x, l_y = 3, 3
    n_sites = l_x * l_y
    lattice = lattices.two_dimensional_grid(l_x, l_y)
    gamma = jnp.array(np.random.rand(len(lattice.shell_distances)))
    wave = wavefunctions.merrifield(gamma.size)

    elec_pos = (1, 1)
    phonon_occ_0 = jnp.array(np.random.randint(3, size=(l_y, l_x)))
    overlap = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ_0, gamma, lattice))

    phonon_occ = phonon_occ_0.at[0, 0].add(1)
    overlap_ex = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, gamma, lattice))
    assert np.allclose(overlap_ex, overlap * gamma[2])


def test_merrifield_overlap_map():
    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)
    np.random.seed(seed)

    gamma = jnp.array(np.random.rand(n_sites // 2 + 1))
    wave = wavefunctions.merrifield(gamma.size)

    elec_pos = jnp.array([(0,), (1,)])
    phonon_occ = jnp.array([1, 3, 0, 2])
    overlap = jnp.exp(wave.calc_overlap_map(elec_pos, phonon_occ, gamma, lattice))

    phonon_occ = jnp.array([1, 3, 0, 3])
    overlap_ex = jnp.exp(wave.calc_overlap_map(elec_pos, phonon_occ, gamma, lattice))
    assert np.allclose(overlap_ex, overlap * (gamma[1] + gamma[2]))


def test_ssh_merrifield_overlap():
    n_sites = 2
    lattice = lattices.one_dimensional_chain(n_sites)
    np.random.seed(seed)

    gamma = jnp.array(np.random.rand(1))
    wave = wavefunctions.ssh_merrifield(gamma.size)

    elec_pos = (0,)
    phonon_occ = jnp.array([[1]])
    overlap = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, gamma, lattice))

    phonon_occ = jnp.array([[2]])
    overlap_ex = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, gamma, lattice))
    assert np.allclose(overlap_ex, overlap * gamma[0])

    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)

    gamma = jnp.array([1.0, 1.0, 0.0])
    wave = wavefunctions.ssh_merrifield(gamma.size)

    elec_pos = (0,)
    phonon_occ = jnp.array([[3, 0, 0, 0]])
    overlap = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, gamma, lattice))

    phonon_occ = jnp.array([[4, 0, 0, 0]])
    overlap_ex = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, gamma, lattice))
    assert np.allclose(overlap_ex, overlap * gamma[0])

    l_x, l_y = 3, 3
    lattice = lattices.two_dimensional_grid(l_x, l_y)

    gamma = jnp.array(np.random.rand(len(lattice.bond_shell_distances)))
    wave = wavefunctions.ssh_merrifield(gamma.size)

    elec_pos = (0, 0)
    phonon_occ = jnp.array(
        [
            [[np.random.randint(2) for _ in range(l_x)] for _ in range(l_y)]
            for _ in range(2)
        ]
    )
    overlap = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, gamma, lattice))
    assert np.allclose(overlap, 0.05733717065853551)


def test_ssh_merrifield_overlap_map():
    l_x, l_y = 3, 3
    lattice = lattices.two_dimensional_grid(l_x, l_y)
    np.random.seed(seed)

    gamma = jnp.array(np.random.rand(len(lattice.bond_shell_distances)))
    wave = wavefunctions.ssh_merrifield(gamma.size)

    elec_pos = [(0, 0), (0, 1)]
    phonon_occ = jnp.array(
        [
            [[np.random.randint(2) for _ in range(l_x)] for _ in range(l_y)]
            for _ in range(2)
        ]
    )
    overlap = jnp.exp(wave.calc_overlap_map(elec_pos, phonon_occ, gamma, lattice))
    assert np.allclose(overlap, 4.256158210332774)


def test_bm_ssh_merrifield_overlap():
    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)
    np.random.seed(seed)

    gamma = jnp.array(np.random.rand(n_sites // 2 + 1))
    wave = wavefunctions.bm_ssh_merrifield(gamma.size)

    elec_pos = (3,)
    phonon_occ = jnp.array([[0, 0, 0, 2]])
    overlap = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, gamma, lattice))
    assert np.allclose(overlap, -0.02974506 + 0.0392389j)


def test_nn_jastrow_overlap():
    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)
    np.random.seed(seed)

    gamma = jnp.array(np.random.rand(n_sites // 2 + 1))
    reference = wavefunctions.ssh_merrifield(gamma.size)
    model = models.MLP([5, 1])
    model_input = jnp.zeros(2 * n_sites)
    nn_parameters = model.init(random.PRNGKey(seed), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
    parameters = [gamma, nn_parameters]
    wave = wavefunctions.nn_jastrow(model.apply, reference, n_nn_parameters)

    elec_pos = (0,)
    phonon_occ = jnp.array([[2, 0, 1, 0]])
    overlap = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice))
    assert np.allclose(overlap, 0.004827891090786609)

    model = models.CNN([4, 1], [(2,), (2,)])
    model_input = jnp.zeros((1, n_sites, 2))
    nn_parameters = model.init(random.PRNGKey(seed), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
    parameters = [gamma, nn_parameters]
    wave = wavefunctions.nn_jastrow(model.apply, reference, n_nn_parameters)

    overlap = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice))
    # assert np.allclose(overlap, 0.030337517632646055)

    l_x, l_y = 4, 4
    n_sites = l_x * l_y
    lattice = lattices.two_dimensional_grid(l_x, l_y)

    gamma = jnp.array(np.random.rand(len(lattice.shell_distances)))
    reference = wavefunctions.merrifield(gamma.size)
    model = models.MLP([4, 1])
    model_input = jnp.zeros(2 * n_sites)
    nn_parameters = model.init(random.PRNGKey(seed), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
    parameters = [gamma, nn_parameters]
    wave = wavefunctions.nn_jastrow(model.apply, reference, n_nn_parameters)

    elec_pos = (0, 0)
    phonon_occ = jnp.array(
        [[np.random.randint(2) for _ in range(l_x)] for _ in range(l_y)]
    )
    overlap = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice))
    assert np.allclose(overlap, 0.0007504690111926278)

    model = models.CNN([4, 1], [(2, 2), (2, 2)])
    model_input = jnp.zeros((1, l_y, l_x, 2))
    nn_parameters = model.init(random.PRNGKey(seed), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
    parameters = [gamma, nn_parameters]
    wave = wavefunctions.nn_jastrow(model.apply, reference, n_nn_parameters)

    overlap = jnp.exp(wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice))
    # assert np.allclose(overlap, 0.04981814914850025)


def test_nn_jastrow_2_overlap():
    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)
    np.random.seed(seed)

    gamma = jnp.array(np.random.rand(n_sites // 2 + 1))
    reference = wavefunctions.merrifield(gamma.size)
    model = models.CNN([4, 1], [(2,), (2,)])
    model_input = jnp.zeros((1, 2, 2))
    nn_parameters = model.init(random.PRNGKey(seed), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
    gamma_ee = jnp.array(np.random.rand(n_sites // 2 + 1))
    ee_j = wavefunctions.ee_jastrow(gamma_ee.size)
    parameters = [gamma, nn_parameters, gamma_ee]
    wave = wavefunctions.nn_jastrow_2(model.apply, reference, ee_j, n_nn_parameters)

    elec_pos = jnp.array([(0,), (2,)])
    phonon_occ = jnp.array([2, 0, 1, 0])
    overlap = jnp.exp(wave.calc_overlap_map(elec_pos, phonon_occ, parameters, lattice))
    assert np.allclose(overlap, 0.37565744 - 0.4129256j)


def test_uhf():
    n_sites = 5
    lattice = lattices.one_dimensional_chain(n_sites)
    np.random.seed(seed)

    n_elec = (2, 3)
    wave = wavefunctions.uhf(n_sites * sum(n_elec), n_elec)
    parameters = [
        jnp.array(
            np.random.randn(n_sites, n_elec[0])
            + 1.0j * np.random.randn(n_sites, n_elec[0])
        ),
        jnp.array(
            np.random.randn(n_sites, n_elec[1])
            + 1.0j * np.random.randn(n_sites, n_elec[1])
        ),
    ]
    walker = (jnp.array([1, 1, 0, 0, 0]), jnp.array([1, 1, 1, 0, 0]))
    walker_data = wave.build_walker_data(walker, parameters, lattice)
    overlap = wave.calc_overlap(walker_data, parameters, lattice)
    excitation = jnp.array((0, 1, 2))
    overlap_ratio = wave.calc_overlap_ratio(
        walker_data, excitation, parameters, lattice
    )
    walker_1 = (jnp.array([0, 2]), jnp.array([0, 1, 2]))
    walker_data["walker"] = walker_1
    overlap_1 = wave.calc_overlap(walker_data, parameters, lattice)
    assert np.allclose(jnp.exp(overlap_1), jnp.exp(overlap) * overlap_ratio)

    grad = wave.calc_overlap_gradient(walker_data, parameters, lattice)
    assert grad.size == wave.n_parameters


if __name__ == "__main__":
    test_merrifield_overlap()
    test_merrifield_overlap_map()
    test_ssh_merrifield_overlap()
    test_ssh_merrifield_overlap_map()
    test_bm_ssh_merrifield_overlap()
    test_nn_jastrow_overlap()
    test_nn_jastrow_2_overlap()
    test_uhf()
