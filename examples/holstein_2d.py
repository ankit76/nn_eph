import os

os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
from functools import partial

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
from jax import random, tree_util
from mpi4py import MPI

print = partial(print, flush=True)

from nn_eph import driver, hamiltonians, lattices, models, samplers, wavefunctions

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print(f"# Number of cores: {size}\n#")

l_x, l_y = 6, 6
n_sites = l_x * l_y
lattice = lattices.two_dimensional_grid(l_x, l_y)

omega = 2.0
g = 2.0
ham = hamiltonians.holstein_2d(omega, g)

gamma = jnp.array([g / omega / n_sites for _ in range(len(lattice.shell_distances))])
reference = wavefunctions.merrifield(gamma.size)
model = models.CNN([4, 2, 1], [(2, 2), (2, 2), (2, 2)])
model_input = jnp.zeros((1, l_y, l_x, 1))
nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
parameters = [gamma, nn_parameters]
wave = wavefunctions.nn_jastrow(model.apply, reference, n_nn_parameters)

seed = 789941
n_eql = 100
n_samples = 10000
sampler = samplers.continuous_time(n_eql, n_samples)

walker = [
    (0, 0),
    jnp.array(
        [[int((g // (omega * n_sites)) ** 2) for _ in range(l_x)] for _ in range(l_y)]
    ),
]
n_steps = 500
step_size = 0.02

if rank == 0:
    print(f"# omega: {omega}")
    print(f"# g: {g}")
    print(f"# n_sites: {n_sites}")
    print(f"# n_samples per process: {n_samples}")
    print(f"# n_eql: {n_eql}")
    print(f"# seed: {seed}")
    print(f"# number of parameters: {wave.n_parameters}\n#")

driver.driver(
    walker,
    ham,
    parameters,
    wave,
    lattice,
    sampler,
    n_steps=n_steps,
    step_size=step_size,
    seed=seed,
)
