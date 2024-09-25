import os

# os.environ[
#     "XLA_FLAGS"
# ] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_ENABLE_X64"] = "True"
from functools import partial

from nn_eph import config

config.setup_jax()
MPI = config.setup_comm()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, tree_util

# from mpi4py import MPI

print = partial(print, flush=True)

from nn_eph import driver, hamiltonians, lattices, models, samplers, wavefunctions

if rank == 0:
    print(f"# Number of cores: {size}\n#")

n_sites = 10
lattice = lattices.one_dimensional_chain(n_sites)

omega = 1.0
g = 1.0
omega_q = tuple(omega for _ in range(n_sites))
e_k = tuple(-2.0 * np.cos(2.0 * np.pi * k / n_sites) for k in range(n_sites))
g_kq = tuple(tuple(g / n_sites**0.5 for _ in range(n_sites)) for _ in range(n_sites))
ham = hamiltonians.kq(omega_q, e_k, g_kq, max_n_phonons=jnp.inf)

model_r = models.MLP(
    [1],
    param_dtype=jnp.complex64,
    kernel_init=models.complex_kernel_init,
    activation=jax.nn.relu,
)
model_phi = models.MLP(
    [0], param_dtype=jnp.complex64, kernel_init=jax.nn.initializers.zeros
)
model_input = jnp.zeros((2 * n_sites))
nn_parameters_r = model_r.init(random.PRNGKey(891), model_input, mutable=True)
nn_parameters_phi = model_phi.init(random.PRNGKey(1), model_input, mutable=True)
n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters_r)) + sum(
    x.size for x in tree_util.tree_leaves(nn_parameters_phi)
)
parameters = [nn_parameters_r, nn_parameters_phi]
wave = wavefunctions.nn_complex(
    model_r.apply,
    model_phi.apply,
    n_nn_parameters,
    mask=(1.0, 0.0),
    get_input=wavefunctions.get_input_k,
)
if rank == 0:
    print(model_r.tabulate(jax.random.PRNGKey(0), model_input))
    print(model_phi.tabulate(jax.random.PRNGKey(1), model_input))

n_eql = 500
n_samples = 10000
sampler = samplers.continuous_time(n_eql, n_samples)
walker = [(0,), jnp.array([0 * int(g / omega) ** 2] + [0] * (n_sites - 1))]

n_steps_gd = 50
step_size_gd = 0.01
seed = 78994
if rank == 0:
    # print(f'# omega: {omega}')
    # print(f'# g: {g}')
    print(f"# n_sites: {n_sites}")
    print(f"# n_samples per process: {n_samples}")
    print(f"# n_eql: {n_eql}")
    print(f"# seed: {seed}")
    print(f"# number of parameters: {wave.n_parameters}\n#")
ene, parameters = driver.driver(
    walker,
    ham,
    parameters,
    wave,
    lattice,
    sampler,
    MPI,
    n_steps=n_steps_gd,
    step_size=step_size_gd,
    seed=seed,
    dev_thresh_fac=jnp.inf,
    n_walkers=4,
)
n_steps_gd = 100
step_size_gd = 0.01
ene, parameters = driver.driver(
    walker,
    ham,
    parameters,
    wave,
    lattice,
    sampler,
    MPI,
    n_steps=n_steps_gd,
    step_size=step_size_gd,
    seed=seed,
    dev_thresh_fac=jnp.inf,
    n_walkers=4,
)
n_steps_gd = 200
step_size_gd = 0.01
ene, parameters = driver.driver(
    walker,
    ham,
    parameters,
    wave,
    lattice,
    sampler,
    MPI,
    n_steps=n_steps_gd,
    step_size=step_size_gd,
    seed=seed,
    dev_thresh_fac=jnp.inf,
    n_walkers=4,
)
