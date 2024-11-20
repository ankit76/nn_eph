import numpy as np

from nn_eph import config

config.setup_jax()
MPI = config.setup_comm()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


import pickle
from functools import partial
from typing import Any, Tuple

import jax
from flax import linen as nn
from jax import jit, lax
from jax import numpy as jnp
from jax import random, tree_util, vmap

from nn_eph import (
    driver,
    hamiltonians,
    hamiltonians_n,
    heisenberg,
    lattices,
    models,
    samplers,
    wavefunctions,
    wavefunctions_n,
)

print = partial(print, flush=True)

from pyscf import fci, gto, scf

n_sites = 4
t = -1.0
u = 0.0
n_elec = n_sites // 2
antiperiodic = n_elec % 2 == 0

# hf calculation
mf_mo_coeff = None

if rank == 0:
    eri = np.zeros((n_sites, n_sites, n_sites, n_sites))
    h1 = np.zeros((n_sites, n_sites))

    for i in range(n_sites - 1):
        h1[i, i + 1] = t
        h1[i + 1, i] = t
        eri[i, i, i, i] = u

    h1[0, n_sites - 1] = t * (1 - 2 * antiperiodic)
    h1[n_sites - 1, 0] = t * (1 - 2 * antiperiodic)
    eri[n_sites - 1, n_sites - 1, n_sites - 1, n_sites - 1] = u

    # make dummy molecule
    mol = gto.Mole()
    mol.nelectron = n_elec
    mol.incore_anyway = True
    mol.spin = n_elec
    mol.build()

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1  # type: ignore
    mf.get_ovlp = lambda *args: np.eye(n_sites)  # type: ignore
    mf._eri = eri
    mf.kernel()
    mf_mo_coeff = mf.mo_coeff

comm.barrier()
mf_mo_coeff = comm.bcast(mf_mo_coeff, root=0)
comm.barrier()

lattice = lattices.one_dimensional_chain(n_sites)
omega = 1.0
g = 1.5
ham = hamiltonians_n.holstein_spinless(
    omega, g, n_sites, n_elec, max_n_phonons=jnp.inf, antiperiodic=antiperiodic
)
walker = jnp.array(
    [
        [1 if i % 2 == 0 else 0 for i in range(n_sites)],
        [0 for _ in range(n_sites)],
    ]
)
n_eql = 100
n_samples = 10000
sampler = samplers.continuous_time(n_eql, n_samples)

ref_parameters = [
    jnp.array(mf_mo_coeff[:, :n_elec]),
    0.0 * jnp.array(np.random.randn(n_sites, n_elec)),
]
n_ref_parameters = np.sum([parameters_i.size for parameters_i in ref_parameters])
ref_wave = wavefunctions_n.spinless_hf(n_elec, n_ref_parameters)


@partial(jit, static_argnums=(2,))
def apply_excitation_ee(
    walker: jax.Array, excitation: jax.Array, lattice: Any
) -> jax.Array:
    walker_copy = walker.copy()
    i_pos = jnp.array(lattice.sites)[excitation["idx"][1:][0]]
    a_pos = jnp.array(lattice.sites)[excitation["idx"][1:][1]]
    walker_copy = walker_copy.at[(0, *i_pos)].add(-1)
    walker_copy = walker_copy.at[(0, *a_pos)].add(1)
    return walker_copy


@partial(jit, static_argnums=(2,))
def apply_excitation_eph(walker, excitation, lattice):
    walker_copy = walker.copy()
    excitation_ee = excitation["ee"]
    walker_ee = apply_excitation_ee(walker_copy[:1], excitation_ee, lattice)
    walker_copy = walker_copy.at[:1].set(walker_ee)
    excitation_ph = excitation["ph"]
    site = jnp.array(lattice.sites)[excitation_ph[0]]
    phonon_change = excitation_ph[1]
    walker_copy = walker_copy.at[(1, *site)].add(phonon_change)
    return walker_copy


model_j_r = models.MLP(
    [16, 8, 4, 1],
    param_dtype=jnp.complex64,
    kernel_init=models.complex_kernel_init,
    # kernel_init=jax.nn.initializers.zeros,
    activation=jax.nn.relu,
)
model_j_phi = models.MLP(
    [16, 8, 4, 1],
    param_dtype=jnp.complex64,
    kernel_init=models.complex_kernel_init,
    # kernel_init=jax.nn.initializers.zeros,
    activation=jax.nn.relu,
)
model_input = jnp.zeros((2 * n_sites))
nn_parameters_r = model_j_r.init(random.PRNGKey(891), model_input, mutable=True)
nn_parameters_phi = model_j_phi.init(random.PRNGKey(891), model_input, mutable=True)
n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters_r)) + sum(
    x.size for x in tree_util.tree_leaves(nn_parameters_phi)
)
if rank == 0:
    print(model_j_r.tabulate(jax.random.PRNGKey(0), model_input))
    print(model_j_phi.tabulate(jax.random.PRNGKey(0), model_input))
nn_j = wavefunctions_n.nn_complex(
    model_j_r.apply,
    model_j_phi.apply,
    n_nn_parameters,
    apply_excitation_eph,
    input_adapter=wavefunctions_n.input_adapter_t,
)

parameters = [[nn_parameters_r, nn_parameters_phi], ref_parameters]


@jit
def walker_adapter_ee(walker: jax.Array) -> jax.Array:
    return walker[0]


@jit
def excitation_adapter_ee(excitation: dict) -> Tuple:
    return {
        "sm_idx": excitation["ee"]["sm_idx"][1:],
        "idx": excitation["ee"]["idx"][1:],
    }, excitation["ee"]["idx"][0] > -1


wave = wavefunctions_n.product_state(
    (nn_j, ref_wave),
    walker_adapters=(lambda x: x, walker_adapter_ee),
    excitation_adapters=(lambda x: (x, 1.0), excitation_adapter_ee),
)

n_steps_gd = 100
step_size_gd = 0.01
seed = 76789
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
)

n_steps_gd = 200
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
)

n_steps_gd = 400
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
)

if rank == 0:
    print(f"Energy per site: {ene / n_sites}")
