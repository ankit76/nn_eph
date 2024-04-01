import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax

# os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random, tree_util

from nn_eph import hamiltonians_n, lattices, models, wavefunctions_n


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


def test_hubbard_holstein():
    np.random.seed(14)
    l_x, l_y = 3, 3
    n_sites = l_x * l_y
    lattice = lattices.two_dimensional_grid(l_x, l_y)
    n_elec = (4, 6)
    ref_wave = wavefunctions_n.uhf(n_elec, n_sites * sum(n_elec))
    ref_parameters = [
        jnp.array(np.random.randn(n_sites, n_elec[0]) + 0.0j),
        jnp.array(np.random.randn(n_sites, n_elec[1]) + 0.0j),
    ]
    model_j = models.MLP(
        [10, 1],
        param_dtype=jnp.complex64,
        kernel_init=models.complex_kernel_init,
        activation=jax.nn.relu,
    )
    model_input = jnp.zeros((3 * n_sites))
    nn_parameters = model_j.init(random.PRNGKey(891), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
    nn_j = wavefunctions_n.nn(
        model_j.apply,
        wavefunctions_n.apply_excitation_eph,
        n_nn_parameters,
        input_adapter=wavefunctions_n.input_adapter_t,
    )

    parameters = [nn_parameters, ref_parameters]
    wave = wavefunctions_n.product_state(
        (nn_j, ref_wave),
        walker_adapters=(lambda x: x, wavefunctions_n.walker_adapter_ee),
        excitation_adapters=(lambda x: (x, 1.0), wavefunctions_n.excitation_adapter_ee),
    )
    walker = jnp.array(
        [
            [[1, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 1, 1], [1, 1, 1], [0, 0, 1]],
            [[1, 0, 0], [0, 2, 0], [1, 1, 0]],
        ]
    )
    ham = hamiltonians_n.hubbard_holstein(
        omega=1.0, g=1.0, u=1.0, n_orbs=n_sites, n_elec=n_elec
    )
    (
        energy,
        qp_weight,
        overlap_gradient,
        weight,
        walker_new,
        overlap,
    ) = ham.local_energy_and_update(walker, parameters, wave, lattice, 0.9)
    assert walker.shape == (3, *lattice.shape)
    assert overlap_gradient.size == wave.n_parameters


if __name__ == "__main__":
    test_hubbard()
    test_hubbard_holstein()
