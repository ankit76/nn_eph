import os
import numpy as np
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
#os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
from jax import lax, jit, custom_jvp, vmap, random, vjp, checkpoint, value_and_grad, tree_util
import jax
from mpi4py import MPI

from functools import partial
print = partial(print, flush=True)

from nn_eph import lattices, models, wavefunctions, hamiltonians, samplers, driver

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
  print(f'# Number of cores: {size}\n#')

n_sites = 10
lattice = lattices.one_dimensional_chain(n_sites)

omega = 1.
g = 1.
ham = hamiltonians.holstein_1d(omega, g)

gamma = jnp.array([ g / omega / n_sites for _ in range(n_sites // 2 + 1) ] + [ 0. for _ in range(n_sites // 2 + 1) ])

model = models.MLP([20, 1])
model_input = jnp.zeros(n_sites)
nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
parameters = [ gamma, nn_parameters ]
reference = wavefunctions.merrifield(gamma.size)
wave = wavefunctions.nn_jastrow(model.apply, reference, n_nn_parameters)

wave = reference
parameters = gamma

seed = 789941
n_eql = 100
n_samples = 100000
sampler = samplers.continuous_time(n_eql, n_samples)

walker = [ (0,), jnp.array([ int((g//(omega * n_sites))**2) ] * n_sites) ]
n_steps = 500
step_size = 0.02

if rank == 0:
  print(f'# omega: {omega}')
  print(f'# g: {g}')
  print(f'# n_sites: {n_sites}')
  print(f'# n_samples per process: {n_samples}')
  print(f'# n_eql: {n_eql}')
  print(f'# seed: {seed}')
  print(f'# number of parameters: {wave.n_parameters}\n#')

driver.driver(walker, ham, parameters, wave, lattice, sampler, n_steps=n_steps, step_size=step_size, seed=seed)

