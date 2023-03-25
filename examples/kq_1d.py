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

omega_k = tuple(0.5 for _ in range(n_sites))
e_k = tuple(-2. * np.cos(2. * np.pi * k / n_sites) for k in range(n_sites))
g = tuple(tuple(-0.5/n_sites**0.5 for _ in range(n_sites)) for _ in range(n_sites))
ham = hamiltonians.kq_1d(omega_k, e_k, g)

model_r = models.MLP([20, 1], param_dtype=jnp.complex64, kernel_init=models.complex_kernel_init)
model_phi = models.MLP([10, 1], param_dtype=jnp.complex64, kernel_init=models.complex_kernel_init)
model_input = jnp.zeros(2*n_sites)
nn_parameters_r = model_r.init(random.PRNGKey(0), model_input, mutable=True)
nn_parameters_phi = model_phi.init(random.PRNGKey(1), model_input, mutable=True)
n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(
    nn_parameters_r)) + sum(x.size for x in tree_util.tree_leaves(nn_parameters_phi))
parameters = [ nn_parameters_r, nn_parameters_phi ]
wave = wavefunctions.nn_complex(model_r.apply, model_phi.apply, n_nn_parameters, get_input=wavefunctions.get_input_k)

seed = 789941
n_eql = 100
n_samples = 10000
sampler = samplers.continuous_time_sr(n_eql, n_samples)

walker = [ (0,), jnp.array([ 0 ] + [ 0 ] * (n_sites - 1)) ]
n_steps = 1000
step_size = 0.2

if rank == 0:
  #print(f'# omega: {omega}')
  #print(f'# g: {g}')
  print(f'# n_sites: {n_sites}')
  print(f'# n_samples per process: {n_samples}')
  print(f'# n_eql: {n_eql}')
  print(f'# seed: {seed}')
  print(f'# number of parameters: {wave.n_parameters}\n#')

driver.driver_sr(walker, ham, parameters, wave, lattice, sampler, n_steps=n_steps, step_size=step_size, seed=seed)

