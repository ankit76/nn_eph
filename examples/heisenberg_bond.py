import os

os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from functools import partial

import jax

# os.environ['JAX_ENABLE_X64'] = 'True'
# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
from jax import random, tree_util
from mpi4py import MPI

print = partial(print, flush=True)

from nn_eph import driver, heisenberg, lattices, models, samplers, wavefunctions

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print(f"# Number of cores: {size}\n#")

n_sites = 10
lattice = lattices.one_dimensional_chain(n_sites)
omega = 1.0

# for g in [ 1. ]:
for g in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
    if rank == 0:
        print(f"\ng: {g}\n")
    ham = heisenberg.heisenberg_bond(omega, g)

    model_r = models.MLP(
        [20, 1], param_dtype=jnp.complex64, kernel_init=models.complex_kernel_init
    )
    # model_phi = models.MLP([1], param_dtype=jnp.complex64, kernel_init=models.complex_kernel_init)
    model_phi = models.MLP(
        [1], param_dtype=jnp.complex64, kernel_init=jax.nn.initializers.zeros
    )
    model_input = jnp.zeros(2 * n_sites)
    nn_parameters_r = model_r.init(random.PRNGKey(0), model_input, mutable=True)
    nn_parameters_phi = model_phi.init(random.PRNGKey(1), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters_r)) + sum(
        x.size for x in tree_util.tree_leaves(nn_parameters_phi)
    )
    parameters = [nn_parameters_r, nn_parameters_phi]
    wave = wavefunctions.spin_nn_complex(
        model_r.apply,
        model_phi.apply,
        n_nn_parameters,
        mask=jnp.array([1.0, 0.0]),
        get_input=wavefunctions.get_input_spin_phonon,
    )

    seed = 789941
    n_eql = 100
    n_samples = 10000
    sampler = samplers.continuous_time(n_eql, n_samples)

    walker = [
        jnp.array([0.5 if i % 2 == 0 else -0.5 for i in range(n_sites)]),
        jnp.array([0 for _ in range(n_sites)]),
    ]
    n_steps = 1000
    step_size = 0.02

    if rank == 0:
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
