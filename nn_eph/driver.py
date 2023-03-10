import os
import numpy as np
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
#os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax, jit, custom_jvp, vmap, random, vjp, checkpoint, value_and_grad, tree_util
import jax
from mpi4py import MPI
import stat_utils

from functools import partial
print = partial(print, flush=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def driver(walker, ham, parameters, wave, lattice, sampler, n_steps = 1000, step_size = 0.1, seed = 0):
  moment_1 = jnp.zeros(wave.n_parameters)
  moment_2 = jnp.zeros(wave.n_parameters)
  decay_1 = 0.1
  decay_2 = 0.01
  key = random.PRNGKey(seed + rank)

  if rank == 0:
    print(f'  iter       ene           qp_weight        grad_norm')

  for iteration in range(n_steps):
    key, subkey = random.split(key)
    random_numbers = random.uniform(subkey, shape=(sampler.n_samples,))
    #g_scan = g #* (1 - jnp.exp(-iteration / 10))
    #starter_iters = 100
    #if iteration < starter_iters:
    #  g_scan = 0.
    #else:
    #  g_scan = g * (1. - jnp.exp(-(iteration - starter_iters) / 500))
      #g_scan = 6. - (6. - g) * (1. - jnp.exp(-(iteration - starter_iters) / 500))
    weight, energy, gradient, lene_gradient, qp_weight, metric, energies, qp_weights, weights = sampler.sampling(walker, ham, parameters, wave, lattice, random_numbers)

    # average and print energies for the current step
    weight = np.array([ weight ])
    energy = np.array([ weight * energy ])
    qp_weight = np.array([ weight * qp_weight ])
    metric = np.array(weight * metric)
    #energy = np.array([ energy ])
    total_weight = 0. * weight
    total_energy = 0. * weight
    total_qp_weight = 0. * weight
    total_metric = 0. * metric
    gradient = np.array(weight * gradient)
    lene_gradient = np.array(weight * lene_gradient)
    #gradient = np.array(gradient)
    #lene_gradient = np.array(lene_gradient)
    total_gradient = 0. * gradient
    total_lene_gradient = 0. * lene_gradient

    comm.barrier()
    comm.Reduce([weight, MPI.DOUBLE], [total_weight, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([energy, MPI.DOUBLE], [total_energy, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([qp_weight, MPI.DOUBLE], [total_qp_weight, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([metric, MPI.DOUBLE], [total_metric, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([gradient, MPI.DOUBLE], [total_gradient, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([lene_gradient, MPI.DOUBLE], [total_lene_gradient, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.barrier()

    new_parameters = None
    if rank == 0:
      total_energy /= total_weight
      total_qp_weight /= total_weight
      total_metric /= total_weight
      total_gradient /= total_weight
      total_lene_gradient /= total_weight
      
      metric = np.zeros((wave.n_parameters + 1, wave.n_parameters + 1))
      metric[0,0] = 1.
      metric[1:,1:] = total_metric + np.diag(np.ones(wave.n_parameters) * 1.e-3)
      metric[0, 1:] = total_gradient
      metric[1:, 0] = total_gradient
      y_vec = np.zeros(wave.n_parameters + 1)
      y_vec[0] = 1 - step_size * total_energy
      y_vec[1:] = total_gradient - total_lene_gradient * step_size
      update = np.linalg.solve(metric, y_vec)
      update = update[1:] / update[0]
      new_parameters = wave.update_parameters(parameters, update)

      ene_gradient = 2 * total_lene_gradient - 2 * total_gradient * total_energy
      print(f'{iteration: 5d}   {total_energy[0]: .6e}    {total_qp_weight[0]: .6e}   {jnp.linalg.norm(ene_gradient): .6e}')
      #print(f'iter: {iteration: 5d}, ene: {total_energy[0]: .6e}, qp_weight: {total_qp_weight[0]: .6e}, grad: {jnp.linalg.norm(ene_gradient): .6e}')
      #print(f'parameters: {parameters}')
      #print(f'total_weight: {total_weight}')
      #print(f'total_energy: {total_energy}')
      #print(f'total_gradient: {total_gradient}')
      
      #if iteration == 0:
      #  update = step_size * ene_gradient
      #  moment_1 = decay_1 * ene_gradient + (1 - decay_1) * moment_1
      #  moment_2 = jnp.maximum(moment_2, decay_2 * ene_gradient * ene_gradient + (1 - decay_2) * moment_2)
      #else:
      #  moment_1 = decay_1 * ene_gradient + (1 - decay_1) * moment_1
      #  moment_2 = jnp.maximum(moment_2, decay_2 * ene_gradient * ene_gradient + (1 - decay_2) * moment_2)
      #  update = step_size * moment_1 / (moment_2**0.5 + 1.e-8)
      #  #update = momentum * update + step_size * ene_gradient
      #new_parameters = wave.update_parameters(parameters, -update)
      
      #new_parameters = [ update_parameters(parameters[0], -update), parameters[1] ]
      #new_parameters = parameters - update
      #new_parameters = new_parameters.at[0].set(0.)
      #print(f'new_parameters: {new_parameters}')

    comm.barrier()
    parameters = comm.bcast(new_parameters, root=0)
    comm.barrier()

  weights = np.array(weights)
  energies = np.array(weights * energies)
  total_weights = 0. * weights
  total_energies = 0. * weights

  comm.barrier()
  comm.Reduce([weights, MPI.DOUBLE], [total_weights, MPI.DOUBLE], op=MPI.SUM, root=0)
  comm.Reduce([energies, MPI.DOUBLE], [total_energies, MPI.DOUBLE], op=MPI.SUM, root=0)
  comm.barrier()
  if rank == 0:
    total_energies /= total_weights
    #np.savetxt('parameters.dat', parameters[0])
    np.savetxt('samples.dat', np.stack((total_weights, total_energies)).T)
    stat_utils.blocking_analysis(total_weights, total_energies, neql = 0, printQ = True, writeBlockedQ = False)

if __name__ == "__main__":
  import lattices, models, wavefunctions, hamiltonians, samplers
  l_x, l_y = 6, 6
  n_sites = l_x * l_y
  omega = 2.
  g = 2.
  n_samples = 10000
  seed = 789941
  n_eql = 100
  n_steps = 10000
  step_size = 0.02
  sampler = samplers.continuous_time(n_eql, n_samples)
  ham = hamiltonians.holstein_2d(omega, g)
  lattice = lattices.two_dimensional_grid(l_x, l_y)
  gamma = jnp.array([ g / omega / n_sites for _ in range(len(lattice.shell_distances)) ])
  model = models.MLP([100, 1])
  model_input = jnp.zeros(2*n_sites)
  nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
  n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
  parameters = [ gamma, nn_parameters ]
  reference = wavefunctions.merrifield(gamma.size)
  wave = wavefunctions.nn_jastrow(model.apply, reference, n_nn_parameters)
  walker = [ (0, 0), jnp.array([[ int((g//(omega * n_sites))**2) for _ in range(l_x) ] for _ in range(l_y) ]) ]

  if rank == 0:
    print(f'# omega: {omega}')
    print(f'# g: {g}')
    print(f'# l_x, l_y: {l_x, l_y}')
    print(f'# n_samples: {n_samples}')
    print(f'# n_eql: {n_eql}')
    print(f'# seed: {seed}')
    print(f'# number of parameters: {wave.n_parameters}\n#')

  driver(walker, ham, parameters, wave, lattice, sampler, n_steps, step_size = 0.01)


