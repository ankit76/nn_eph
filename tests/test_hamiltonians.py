import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
import pytest
import numpy as np
from jax import random, tree_util, numpy as jnp
from nn_eph import wavefunctions, lattices, models, hamiltonians

seed = 3455

np.random.seed(seed)
n_sites_1d = 4
lattice_1d = lattices.one_dimensional_chain(n_sites_1d)
gamma_1d = jnp.array(np.random.rand(n_sites_1d//2 + 1))
reference_h_1d = wavefunctions.merrifield(gamma_1d.size)
reference_s_1d = wavefunctions.ssh_merrifield(gamma_1d.size)
model_1d = models.MLP([5, 1])
model_input_1d = jnp.zeros(2*n_sites_1d)
nn_parameters_1d = model_1d.init(random.PRNGKey(seed), model_input_1d, mutable=True)
n_nn_parameters_1d = sum(x.size for x in tree_util.tree_leaves(nn_parameters_1d))
parameters_1d = [ gamma_1d, nn_parameters_1d ]
wave_h_1d = wavefunctions.nn_jastrow(model_1d.apply, reference_h_1d, n_nn_parameters_1d)
wave_s_1d = wavefunctions.nn_jastrow(model_1d.apply, reference_s_1d, n_nn_parameters_1d)
walker_1d = jnp.array([ 0 ] + [ 2, 0, 1, 0 ])
walker_1d_2 = jnp.array([ 0, 1 ] + [ 2, 0, 1, 0 ])

np.random.seed(seed)
l_x, l_y = 4, 4
n_sites_2d = l_x * l_y
lattice_2d = lattices.two_dimensional_grid(l_x, l_y)
gamma_2d = jnp.array(np.random.rand(len(lattice_2d.shell_distances)))
reference_2d = wavefunctions.merrifield(gamma_2d.size)
model_2d = models.MLP([5, 1])
model_input_2d = jnp.zeros(2*n_sites_2d)
nn_parameters_2d = model_2d.init(random.PRNGKey(seed), model_input_2d, mutable=True)
n_nn_parameters_2d = sum(x.size for x in tree_util.tree_leaves(nn_parameters_2d))
parameters_2d = [ gamma_2d, nn_parameters_2d ]
wave_2d = wavefunctions.nn_jastrow(model_2d.apply, reference_2d, n_nn_parameters_2d)

phonon_occ = jnp.array([[ np.random.randint(2) for _ in range(l_x) ] for _ in range(l_y) ])
walker_2d = [ (0, 0), phonon_occ ]


def test_holstein_1d():
  ham = hamiltonians.holstein_1d(1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker = ham.local_energy_and_update(walker_1d, parameters_1d, wave_h_1d, lattice_1d, random_number)
  assert np.allclose(energy, -32.4191830535864)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 28.137563833940348)
  assert np.allclose(weight, 0.03917701580761051)
  assert sum(walker) == 2

def test_holstein_1d_2():
  ham = hamiltonians.holstein_1d_2(1., 1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker = ham.local_energy_and_update(walker_1d_2, gamma_1d, reference_h_1d, lattice_1d, random_number)
  assert np.allclose(energy, -18.943967797424914)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 21.620797603398334)
  assert np.allclose(weight, 0.052438402703666474)
  assert sum(walker) == 5

def test_long_range_1d():
  ham = hamiltonians.long_range_1d(1., 1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker = ham.local_energy_and_update(walker_1d, parameters_1d, wave_h_1d, lattice_1d, random_number)
  assert np.allclose(energy, -32.47901747694962)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 28.137563833940348)
  assert np.allclose(weight, 0.0360180205080746)
  assert sum(walker) == 2

def test_ssh_1d():
  ham = hamiltonians.ssh_1d(1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker = ham.local_energy_and_update(walker_1d, parameters_1d, wave_s_1d, lattice_1d, random_number)
  assert np.allclose(energy, -30.099586948140804)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 23.908205908185963)
  assert np.allclose(weight, 0.039807592428511035)
  assert sum(walker) == 3

def test_holstein_2d():
  ham = hamiltonians.holstein_2d(1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker = ham.local_energy_and_update(walker_2d, parameters_2d, wave_2d, lattice_2d, random_number)
  assert np.allclose(energy, -4.604745245749261)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 20.663823636174804)
  assert np.allclose(weight, 0.10411520289333714)

def test_long_range_2d():
  ham = hamiltonians.long_range_2d(1., 1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker = ham.local_energy_and_update(walker_2d, parameters_2d, wave_2d, lattice_2d, random_number)
  assert np.allclose(energy, -7.768236289696451)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 20.663823636174804)
  assert np.allclose(weight, 0.024041890740621793)

if __name__ == "__main__":
  test_holstein_1d()
  test_holstein_1d_2()
  test_long_range_1d()
  test_ssh_1d()
  test_holstein_2d()
  test_long_range_2d()

