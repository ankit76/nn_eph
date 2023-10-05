import os

import numpy as np
import optax
from jax.tree_util import register_pytree_node_class

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from dataclasses import dataclass
from functools import partial
from typing import Callable

# os.environ['JAX_ENABLE_X64'] = 'True'
import jax
from jax import jit
from jax import numpy as jnp
from jax import random, tree_util, vjp, vmap


@partial(jit, static_argnums=(1,))
def get_input_r(walker, lattice_shape):
    elec_pos = walker[0]
    phonon_occ = walker[1]
    input_ar = phonon_occ.reshape(-1, *lattice_shape)
    for ax in range(len(lattice_shape)):
        for phonon_type in range(phonon_occ.shape[0]):
            input_ar = input_ar.at[phonon_type].set(
                input_ar[phonon_type].take(
                    elec_pos[ax] + jnp.arange(lattice_shape[ax]), axis=ax, mode="wrap"
                )
            )
    return jnp.stack([*input_ar], axis=-1)


@partial(jit, static_argnums=(1,))
def get_input_k(walker, lattice_shape):
    elec_k = walker[0]
    phonon_occ = walker[1]
    elec_k_ar = jnp.zeros(lattice_shape)
    elec_k_ar = elec_k_ar.at[elec_k].set(1)
    input_ar = jnp.stack([elec_k_ar, *phonon_occ.reshape(-1, *lattice_shape)], axis=-1)
    return input_ar


@dataclass
@register_pytree_node_class
class wavefunction:
    nn_apply_r: Callable = None
    nn_apply_phi: Callable = None

    @partial(jax.jit, static_argnums=(0,))
    def calc_overlap(self, parameters, input):
        nn_r = parameters[0]
        nn_phi = parameters[1]
        output_r = self.nn_apply_r(nn_r, input)
        output_phi = self.nn_apply_phi(nn_phi, input)
        # output_r = jnp.array(self.nn_apply_r(nn_r, input + 0.j), dtype='complex64')
        # output_phi = jnp.array(self.nn_apply_phi(nn_phi, input + 0.j), dtype='complex64')
        overlap = jnp.exp(output_r[0]) * jnp.exp(1.0j * jnp.sum(output_phi))
        # overlap = jnp.exp(output_r[0]) * jnp.sum(output_phi)
        # overlap = jnp.sum(output_phi)
        return overlap

    def __hash__(self):
        return hash((self.nn_apply_r, self.nn_apply_phi))

    def tree_flatten(self):
        return (), (self.nn_apply_r, self.nn_apply_phi)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@partial(jax.jit, static_argnums=(1))
def calc_fidelity(parameters, psi, psi_target, basis):
    psi_overlaps = vmap(psi.calc_overlap, in_axes=(None, 0))(parameters, basis)
    fidelity = 1 - jnp.abs(jnp.sum(psi_overlaps * jnp.conj(psi_target))) ** 2 / (
        jnp.sum(jnp.abs(psi_overlaps) ** 2) * jnp.sum(jnp.abs(psi_target) ** 2)
    )
    return fidelity


def optimize(optimizer, parameters, psi, psi_target, basis, max_iter=1000):
    opt_state = optimizer.init(parameters)

    def step(parameters, opt_state, basis):
        value, grad_fun = vjp(calc_fidelity, parameters, psi, psi_target, basis)
        gradient = grad_fun(jnp.array([1.0])[0])[0]
        # value, gradient = value_and_grad(calc_fidelity, argnums=0)(parameters, psi, psi_target, basis)
        updates, opt_state = optimizer.update(gradient, opt_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, opt_state, value

    for i in range(max_iter):
        parameters, opt_state, value = step(parameters, opt_state, basis)
        if (i + 1) % 50 == 0:
            print(f"step {i}, loss: {value}")

    return parameters, opt_state


if __name__ == "__main__":
    import lattices
    import models

    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)
    lattice_shape = lattice.shape
    max_n_phonons = 3
    basis = lattice.make_polaron_basis(max_n_phonons)
    basis_inputs = []
    for state in basis:
        basis_inputs.append(get_input_r(state, lattice.shape))
    basis = jnp.array(basis_inputs)

    # model_r = models.MLP([1])
    model_r = models.MLP(
        [1], param_dtype=jnp.complex64, kernel_init=models.complex_kernel_init
    )
    # model_r = models.CNN([4, 2, 1], [(3,), (3,), (3,)], param_dtype=jnp.complex64,
    #                     kernel_init=models.                   complex_kernel_init)
    # model_r = models.MLP([20, 1], param_dtype=jnp.complex64, kernel_init=jax.nn.initializers.zeros)
    # model_phi = models.MLP([10, 1], param_dtype=jnp.complex64, kernel_init=jax.nn.initializers.zeros)
    # model_phi = models.MLP([1])#, activation=lambda x: jnp.log(jnp.cosh(x)))
    model_phi = models.MLP(
        [1], param_dtype=jnp.complex64, kernel_init=models.complex_kernel_init
    )

    # model_phi = models.CNN([4, 2, 1], [(3,), (3,), (3,)], param_dtype=jnp.complex64,
    #                       kernel_init=models.                 complex_kernel_init)
    psi = wavefunction(model_r.apply, model_phi.apply)

    model_input = jnp.zeros(n_sites)
    nn_parameters_r = model_r.init(random.PRNGKey(0), model_input, mutable=True)
    nn_parameters_phi = model_phi.init(random.PRNGKey(1), model_input, mutable=True)
    parameters = [nn_parameters_r, nn_parameters_phi]

    optimizer = optax.adam(1e-2)
    psi_target = jnp.array(np.loadtxt("psi_target_3.txt"))

    print(f"parameters:\n{parameters}\n")
    # value, grad_fun = vjp(calc_fidelity, parameters, psi, psi_target, basis)
    value, grad_fun = vjp(psi.calc_overlap, parameters, basis[1])
    gradient = grad_fun(jnp.array([1.0 + 0.0j])[0])[0]
    print(f"gradient:\n{gradient}\n")
    # exit()

    eps = 0.01
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters_r)) + sum(
        x.size for x in tree_util.tree_leaves(nn_parameters_phi)
    )
    update = jnp.zeros(n_nn_parameters)
    update = update.at[0].set(eps)
    # print(parameters[0])
    flat_tree, tree = tree_util.tree_flatten(parameters[0])
    counter = 0
    for i in range(len(flat_tree)):
        flat_tree[i] += update[counter : counter + flat_tree[i].size].reshape(
            flat_tree[i].shape
        )
        counter += flat_tree[i].size
    parameters[0] = tree_util.tree_unflatten(tree, flat_tree)

    flat_tree, tree = tree_util.tree_flatten(parameters[1])
    for i in range(len(flat_tree)):
        flat_tree[i] += update[counter : counter + flat_tree[i].size].reshape(
            flat_tree[i].shape
        )
        counter += flat_tree[i].size
    parameters[1] = tree_util.tree_unflatten(tree, flat_tree)
    # print(f'parameters_new:\n{parameters}\n')

    value_1 = psi.calc_overlap(parameters, basis[1])
    # value_1 = calc_fidelity(parameters, psi, psi_target, basis)
    fd_gradient = (value_1 - value) / eps
    print(f"fd_gradient: {fd_gradient}\n")
    print(value)
    print(value_1)
    exit()

    parameters, opt_state = optimize(
        optimizer, parameters, psi, psi_target, basis, max_iter=5
    )

    # hess = hessian(calc_fidelity, argnums=0)(parameters, psi, psi_target, basis)
    # eval, evec = jnp.linalg.eigh(hess)
    # print(hess)
    # exit()

    print(parameters)
    psi_overlaps = vmap(psi.calc_overlap, in_axes=(None, 0))(parameters, basis)
    np.savetxt("psi_overlaps.txt", psi_overlaps)
    # np.savetxt('psi_overlaps.txt', np.exp(-1.j * np.angle(psi_overlaps[0])) *
    # psi_overlaps / jnp.linalg.norm(psi_overlaps))
    exit()

    import driver
    import hamiltonians
    import samplers
    import wavefunctions

    omega = 3.0
    lam = 0.5
    g = (lam * omega / 2) ** 0.5
    ham = hamiltonians.bm_ssh_1d(omega, g, max_n_phonons=max_n_phonons)

    # parameters = parameters[1]
    # n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters_phi))
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters_r)) + sum(
        x.size for x in tree_util.tree_leaves(nn_parameters_phi)
    )

    for k in [0]:
        wave = wavefunctions.nn_complex(
            model_r.apply, model_phi.apply, n_nn_parameters, k=(k,)
        )
        # wave = wavefunctions.nn(model_phi.apply, n_nn_parameters, k=(k,))

        seed = 789941
        n_eql = 100
        n_samples = 10000
        sampler = samplers.deterministic(max_n_phonons, lattice)
        # sampler = samplers.continuous_time_sr(n_eql, n_samples)

        walker = [(0,), jnp.array([int((g // (omega * n_sites)) ** 2)] * n_sites)]
        n_steps = 1
        step_size = 0.02

        parameters = driver.driver(
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
