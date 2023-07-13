import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
#os.environ['JAX_ENABLE_X64'] = 'True'
import pytest
import numpy as np
from jax import random, tree_util, numpy as jnp
from nn_eph import wavefunctions, lattices, models, hamiltonians, hamiltonians_2

seed = 3455

np.random.seed(seed)
n_sites_1d = 4
lattice_1d = lattices.one_dimensional_chain(n_sites_1d)
gamma_1d = jnp.array(np.random.rand(n_sites_1d//2 + 1))
reference_h_1d = wavefunctions.merrifield(gamma_1d.size)
reference_s_1d = wavefunctions.ssh_merrifield(gamma_1d.size)
reference_bm_s_1d = wavefunctions.bm_ssh_lf(gamma_1d.size)
model_1d = models.MLP([5, 1])
model_input_1d = jnp.zeros(2*n_sites_1d)
nn_parameters_1d = model_1d.init(random.PRNGKey(seed), model_input_1d, mutable=True)
n_nn_parameters_1d = sum(x.size for x in tree_util.tree_leaves(nn_parameters_1d))
parameters_1d = [ gamma_1d, nn_parameters_1d ]
wave_h_1d = wavefunctions.nn_jastrow(model_1d.apply, reference_h_1d, n_nn_parameters_1d)
wave_s_1d = wavefunctions.nn_jastrow(model_1d.apply, reference_s_1d, n_nn_parameters_1d)
walker_1d = [ (0,), jnp.array([ 2, 0, 1, 0 ]) ]
walker_1d_2 = [ [ (0,), (1,) ], jnp.array([2, 0, 1, 0]) ]

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
walker_2d_2 = [ [ (0, 0), (0, 1) ], phonon_occ ]

gamma_s_2d = jnp.array(np.random.rand(len(lattice_2d.bond_shell_distances)))
reference_s_2d = wavefunctions.ssh_merrifield(gamma_s_2d.size)
model_2d = models.MLP([5, 1])
model_input_2d = jnp.zeros(3*n_sites_2d)
nn_parameters_s_2d = model_2d.init(random.PRNGKey(seed), model_input_2d, mutable=True)
n_nn_parameters_2d = sum(x.size for x in tree_util.tree_leaves(nn_parameters_s_2d))
parameters_s_2d = [ gamma_s_2d, nn_parameters_s_2d ]
wave_s_2d = wavefunctions.nn_jastrow(model_2d.apply, reference_s_2d, n_nn_parameters_2d)

phonon_occ_s = jnp.array([[[ np.random.randint(2) for _ in range(l_x) ] for _ in range(l_y) ] for _  in range(2) ])
walker_s_2d = [ (0, 0), phonon_occ_s ]
walker_s_2d_2 = [ [ (0, 0), (0, 1) ], phonon_occ_s ]

np.random.seed(seed)
l_x, l_y, l_z = 3, 3, 3
n_sites_3d = l_x * l_y * l_z
lattice_3d = lattices.three_dimensional_grid(l_x, l_y, l_z)
gamma_3d = jnp.array(np.random.rand(len(lattice_3d.shell_distances)))
reference_3d = wavefunctions.merrifield(gamma_3d.size)
model_3d = models.MLP([5, 1])
model_input_3d = jnp.zeros(2*n_sites_3d)
nn_parameters_3d = model_3d.init(random.PRNGKey(seed), model_input_3d, mutable=True)
n_nn_parameters_3d = sum(x.size for x in tree_util.tree_leaves(nn_parameters_3d))
parameters_3d = [ gamma_3d, nn_parameters_3d ]
wave_3d = wavefunctions.nn_jastrow(model_3d.apply, reference_3d, n_nn_parameters_3d)

phonon_occ = jnp.array([[[ np.random.randint(2) for _ in range(l_x) ] for _ in range(l_y) ] for _ in range(l_z) ])
walker_3d = [ (0, 0, 0), phonon_occ ]

def test_holstein_1d():
  ham = hamiltonians.holstein(1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_1d, parameters_1d, wave_h_1d, lattice_1d, random_number)
  assert np.allclose(energy, -32.4191830535864)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 28.137563833940348)
  assert np.allclose(weight, 0.03917701580761051)
  assert walker[0][0] + sum(walker[1]) == 2

def test_holstein_1d_2():
  ham = hamiltonians_2.holstein_1d(1., 1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_1d_2, gamma_1d, reference_h_1d, lattice_1d, random_number)
  assert np.allclose(energy, -18.943967797424914)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 21.620797603398334)
  assert np.allclose(weight, 0.052438402703666474)
  assert walker[0][0][0] + walker[0][1][0] + sum(walker[1]) == 5

def test_long_range_1d():
  ham = hamiltonians.long_range(1., 1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_1d, parameters_1d, wave_h_1d, lattice_1d, random_number)
  assert np.allclose(energy, -32.47901747694962)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 28.137563833940348)
  assert np.allclose(weight, 0.0360180205080746)
  assert walker[0][0] + sum(walker[1]) == 2

def test_long_range_1d_2():
  ham = hamiltonians_2.long_range_1d(1., 1., 1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_1d_2, gamma_1d, reference_h_1d, lattice_1d, random_number)
  assert np.allclose(energy, -20.63895349095183)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 21.620797603398334)
  assert np.allclose(weight, 0.04611878427324339)
  assert walker[0][0][0] + walker[0][1][0] + sum(walker[1]) == 5

def test_ssh_1d():
  ham = hamiltonians.bond_ssh(1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_1d, parameters_1d, wave_s_1d, lattice_1d, random_number)
  assert np.allclose(energy, -30.099586948140804)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 23.908205908185963)
  assert np.allclose(weight, 0.039807592428511035)
  assert walker[0][0] + sum(walker[1]) == 3

def test_bm_ssh_1d():
  ham = hamiltonians.ssh(1., 1.)
  random_number = 1.
  walker_1d = [(0,), jnp.array([ 2, 0, 0, 0 ])]
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_1d, gamma_1d, reference_bm_s_1d, lattice_1d, random_number)
  assert np.allclose(energy, -27.78881273208553)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 28.64919274)
  assert np.allclose(weight, 0.046798616005667495)
  assert walker[0][0] + sum(walker[1]) == 4

def test_holstein_2d():
  ham = hamiltonians.holstein(1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_2d, parameters_2d, wave_2d, lattice_2d, random_number)
  assert np.allclose(energy, -4.604745245749261)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 20.663823636174804)
  assert np.allclose(weight, 0.10411520289333714)

def test_ssh_2d():
  ham = hamiltonians.bond_ssh(1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_s_2d, parameters_s_2d, wave_s_2d, lattice_2d, random_number)
  assert np.allclose(energy, -49.956671130407884)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 399.1125417937265)
  assert np.allclose(weight, 0.015636617084037848)

def test_ssh_2d_2():
  ham = hamiltonians_2.ssh_2d(1., 1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_s_2d_2, gamma_s_2d, reference_s_2d, lattice_2d, random_number)
  assert np.allclose(energy, -12.836294885966295)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 44.022228220244706)
  assert np.allclose(weight, 0.038376743628477976)

def test_long_range_2d():
  ham = hamiltonians.long_range(1., 1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_2d, parameters_2d, wave_2d, lattice_2d, random_number)
  assert np.allclose(energy, -7.768236289696451)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 20.663823636174804)
  assert np.allclose(weight, 0.024041890740621793)

def test_long_range_2d_2():
  ham = hamiltonians_2.long_range_2d(1., 1., 1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_2d_2, gamma_2d, reference_2d, lattice_2d, random_number)
  assert np.allclose(energy, -6.061611058904939)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 19.116589584055067)
  assert np.allclose(weight, 0.034140078029097465)

def test_holstein_3d():
  ham = hamiltonians.holstein(1., 1.)
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker_3d, parameters_3d, wave_3d, lattice_3d, random_number)
  assert np.allclose(energy, -4.785211306275675)
  assert np.allclose(qp_weight, 0.0)
  assert np.allclose(sum(overlap_gradient), 18.07855867695138)
  assert np.allclose(weight, 0.06343059987847004)

if __name__ == "__main__":
  #test_holstein_1d()
  #test_holstein_1d_2()
  #test_long_range_1d()
  #test_long_range_1d_2()
  #test_ssh_1d()
  test_bm_ssh_1d()
  #test_holstein_2d()
  #test_ssh_2d()
  #test_ssh_2d_2()
  #test_long_range_2d()
  #test_long_range_2d_2()
  #test_holstein_3d()

