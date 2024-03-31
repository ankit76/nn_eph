import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random, tree_util

from nn_eph import hamiltonians_n, lattices, models, wavefunctions_n

seed = 3455


def test_hubbard():
    np.random.seed(14)
    l_x, l_y = 3, 3
    n_sites = l_x * l_y
    lattice = lattices.two_dimensional_grid(l_x, l_y)
    n_elec = (4, 6)
    wave = wavefunctions_n.uhf(n_elec, n_sites * sum(n_elec))
    parameters = [
        jnp.array(np.random.randn(n_sites, n_elec[0]) + 0.0j),
        jnp.array(np.random.randn(n_sites, n_elec[1]) + 0.0j),
    ]
    walker = jnp.array(
        [[[1, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 1, 1], [1, 1, 1], [0, 0, 1]]]
    )
    ham = hamiltonians_n.hubbard(u=1.0, n_orbs=n_sites, n_elec=n_elec)
    (
        energy,
        qp_weight,
        overlap_gradient,
        weight,
        walker_new,
        overlap,
    ) = ham.local_energy_and_update(walker, parameters, wave, lattice, 0.9)
    assert walker.shape == (2, *lattice.shape)
    assert overlap_gradient.size == wave.n_parameters


if __name__ == "__main__":
    test_hubbard()
