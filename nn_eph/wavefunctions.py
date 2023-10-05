import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import pickle
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

# os.environ['JAX_ENABLE_X64'] = 'True'
# os.environ['JAX_DISABLE_JIT'] = 'True'
from jax import jit, lax
from jax import numpy as jnp
from jax import random, tree_util, value_and_grad, vjp


# TODO: needs to be changed for 2 sites
@partial(jit, static_argnums=(2,))
def get_input_r(elec_pos, phonon_occ, lattice_shape):
    input_ar = phonon_occ.reshape(-1, *lattice_shape)
    for ax in range(len(lattice_shape)):
        for phonon_type in range(phonon_occ.shape[0]):
            input_ar = input_ar.at[phonon_type].set(
                input_ar[phonon_type].take(
                    elec_pos[ax] + jnp.arange(lattice_shape[ax]), axis=ax, mode="wrap"
                )
            )
    return jnp.stack([*input_ar], axis=-1)


# TODO: needs to be changed for 2 sites
@partial(jit, static_argnums=(2,))
def get_input_r_1(elec_pos, phonon_occ, lattice_shape):
    elec_ar = jnp.zeros(lattice_shape)
    elec_ar = elec_ar.at[tuple(0 * jnp.array(elec_pos))].set(1)
    input_ar = phonon_occ.reshape(-1, *lattice_shape)
    for ax in range(len(lattice_shape)):
        for phonon_type in range(phonon_occ.shape[0]):
            input_ar = input_ar.at[phonon_type].set(
                input_ar[phonon_type].take(
                    elec_pos[ax] + jnp.arange(lattice_shape[ax]), axis=ax, mode="wrap"
                )
            )
    return jnp.stack([elec_ar, *input_ar], axis=-1)


# TODO: needs to be changed for 2 sites
@partial(jit, static_argnums=(2,))
def get_input_k(elec_k, phonon_occ, lattice_shape):
    elec_k_ar = jnp.zeros(lattice_shape)
    elec_k_ar = elec_k_ar.at[elec_k].set(1)
    input_ar = jnp.stack([elec_k_ar, *phonon_occ.reshape(-1, *lattice_shape)], axis=-1)
    return input_ar


@partial(jit, static_argnums=(2,))
def get_input_n_k(elec_n_k, phonon_occ, lattice_shape):
    elec_k_ar_0 = jnp.zeros(lattice_shape[1:])
    elec_k_ar_0 = elec_k_ar_0.at[elec_n_k[1]].set(1)
    elec_k_ar = jnp.zeros(lattice_shape)
    elec_k_ar = elec_k_ar.at[elec_n_k[0]].set(elec_k_ar_0)
    input_ar = jnp.stack(
        [
            *elec_k_ar.reshape(-1, *lattice_shape[1:]),
            *phonon_occ.reshape(-1, *lattice_shape[1:]),
        ],
        axis=-1,
    )
    return input_ar


@partial(jit, static_argnums=(1,))
def get_input_spins(walker, lattice_shape):
    return walker


@partial(jit, static_argnums=(1,))
def get_input_spin_phonon(walker, lattice_shape):
    spins = walker[0]
    phonons = walker[1]
    input_ar = jnp.stack([spins, *phonons.reshape(-1, *lattice_shape)], axis=-1)
    return input_ar


@dataclass
class merrifield:
    n_parameters: int

    def serialize(self, parameters):
        return parameters

    def update_parameters(self, parameters, update):
        parameters += update
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        gamma = parameters[: self.n_parameters]

        def scanned_fun(carry, x):
            dist = lattice.get_distance(elec_pos, x)
            carry *= (gamma[dist]) ** (phonon_occ[(*x,)])
            return carry, x

        overlap = 1.0
        overlap, _ = lax.scan(scanned_fun, overlap, jnp.array(lattice.sites))

        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        def scanned_fun(carry, x):
            dist_0 = lattice.get_distance(elec_pos[0], x)
            dist_1 = lattice.get_distance(elec_pos[1], x)
            carry *= (parameters[dist_0] + parameters[dist_1]) ** (phonon_occ[(*x,)])
            return carry, x

        overlap = 1.0
        overlap, _ = lax.scan(scanned_fun, overlap, jnp.array(lattice.sites))

        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient / value

    def __hash__(self):
        return hash(self.n_parameters)


@dataclass
class sc_k_n:
    n_parameters: int
    n_e_bands: int

    def serialize(self, parameters):
        return parameters

    def update_parameters(self, parameters, update):
        parameters += update
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        nk = len(lattice.sites)
        gamma = (parameters[:nk] + 1.0j * parameters[nk : 2 * nk]).reshape(
            *lattice.shape
        )
        t = (
            parameters[2 * nk : 2 * nk + self.n_e_bands * nk]
            + 1.0j * parameters[2 * nk + self.n_e_bands * nk :]
        ).reshape(self.n_e_bands, -1)
        overlap = t[elec_pos[0], lattice.get_site_num(elec_pos[1])] * jnp.prod(
            gamma**phonon_occ
        )
        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        def scanned_fun(carry, x):
            dist_0 = lattice.get_distance(elec_pos[0], x)
            dist_1 = lattice.get_distance(elec_pos[1], x)
            carry *= (parameters[dist_0] + parameters[dist_1]) ** (phonon_occ[(*x,)])
            return carry, x

        overlap = 1.0
        overlap, _ = lax.scan(scanned_fun, overlap, jnp.array(lattice.sites))

        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        parameters_c = parameters + 0.0j
        value, grad_fun = vjp(
            self.calc_overlap, elec_pos, phonon_occ, parameters_c, lattice
        )
        gradient = grad_fun(1.0 + 0.0j)[2]
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient)
        gradient = gradient / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient

    def __hash__(self):
        return hash(self.n_parameters)


@dataclass
class sc:
    n_parameters: int

    def serialize(self, parameters):
        return parameters

    def update_parameters(self, parameters, update):
        parameters += update
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        nk = len(lattice.sites)
        gamma = (parameters[:nk] + 1.0j * parameters[nk : 2 * nk]).reshape(
            *lattice.shape
        )
        t = parameters[2 * nk : 3 * nk] + 1.0j * parameters[3 * nk :]
        overlap = t[lattice.get_site_num(elec_pos)] * jnp.prod(gamma**phonon_occ)
        return overlap

    @partial(jit, static_argnums=(0, 6))
    def calc_overlap_ratio(
        self,
        elec_pos_old,
        elec_pos_new,
        phonon_pos,
        phonon_change,
        parameters,
        lattice,
        overlap_old,
        phonon_occ_new,
    ):
        nk = len(lattice.sites)
        gamma_ind_r = lattice.get_site_num(phonon_pos)
        gamma_ind_i = nk + lattice.get_site_num(phonon_pos)
        t_ind_old_r = 2 * nk + lattice.get_site_num(elec_pos_old)
        t_ind_old_i = 3 * nk + lattice.get_site_num(elec_pos_old)
        t_ind_new_r = 2 * nk + lattice.get_site_num(elec_pos_new)
        t_ind_new_i = 3 * nk + lattice.get_site_num(elec_pos_new)
        t_ratio = (parameters[t_ind_new_r] + 1.0j * parameters[t_ind_new_i]) / (
            parameters[t_ind_old_r] + 1.0j * parameters[t_ind_old_i]
        )
        gamma_ratio = (
            parameters[gamma_ind_r] + 1.0j * parameters[gamma_ind_i]
        ) ** phonon_change
        return t_ratio * gamma_ratio

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        def scanned_fun(carry, x):
            dist_0 = lattice.get_distance(elec_pos[0], x)
            dist_1 = lattice.get_distance(elec_pos[1], x)
            carry *= (parameters[dist_0] + parameters[dist_1]) ** (phonon_occ[(*x,)])
            return carry, x

        overlap = 1.0
        overlap, _ = lax.scan(scanned_fun, overlap, jnp.array(lattice.sites))

        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        parameters_c = parameters + 0.0j
        value, grad_fun = vjp(
            self.calc_overlap, elec_pos, phonon_occ, parameters_c, lattice
        )
        gradient = grad_fun(1.0 + 0.0j)[2]
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient / value

    def __hash__(self):
        return hash(self.n_parameters)


@dataclass
class sc_k_nep:
    n_parameters: int
    n_e_bands: int
    n_p_bands: int

    def serialize(self, parameters):
        return parameters

    def update_parameters(self, parameters, update):
        parameters += update
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        nk = len(lattice.sites)
        nk_np = nk * self.n_p_bands
        gamma = (parameters[:nk_np] + 1.0j * parameters[nk_np : 2 * nk_np]).reshape(
            self.n_p_bands, *lattice.shape
        )
        t = (
            parameters[2 * nk_np : 2 * nk_np + self.n_e_bands * nk]
            + 1.0j * parameters[2 * nk_np + self.n_e_bands * nk :]
        ).reshape(self.n_e_bands, -1)
        overlap = t[elec_pos[0], lattice.get_site_num(elec_pos[1])] * jnp.prod(
            gamma**phonon_occ
        )
        return overlap

    @partial(jit, static_argnums=(0, 6))
    def calc_overlap_ratio(
        self,
        elec_pos_old,
        elec_pos_new,
        phonon_pos,
        phonon_change,
        parameters,
        lattice,
        overlap_old,
        phonon_occ_new,
    ):
        nk = len(lattice.sites)
        nk_np = nk * self.n_p_bands
        nk_ne = nk * self.n_e_bands
        gamma_ind_r = phonon_pos[0] * nk + lattice.get_site_num(phonon_pos[1])
        gamma_ind_i = nk_np + phonon_pos[0] * nk + lattice.get_site_num(phonon_pos[1])
        t_ind_old_r = (
            2 * nk_np + elec_pos_old[0] * nk + lattice.get_site_num(elec_pos_old[1])
        )
        t_ind_old_i = (
            2 * nk_np
            + nk_ne
            + elec_pos_old[0] * nk
            + lattice.get_site_num(elec_pos_old[1])
        )
        t_ind_new_r = (
            2 * nk_np + elec_pos_new[0] * nk + lattice.get_site_num(elec_pos_new[1])
        )
        t_ind_new_i = (
            2 * nk_np
            + nk_ne
            + elec_pos_new[0] * nk
            + lattice.get_site_num(elec_pos_new[1])
        )
        t_ratio = (parameters[t_ind_new_r] + 1.0j * parameters[t_ind_new_i]) / (
            parameters[t_ind_old_r] + 1.0j * parameters[t_ind_old_i]
        )
        gamma_ratio = (
            parameters[gamma_ind_r] + 1.0j * parameters[gamma_ind_i]
        ) ** phonon_change
        return t_ratio * gamma_ratio

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        def scanned_fun(carry, x):
            dist_0 = lattice.get_distance(elec_pos[0], x)
            dist_1 = lattice.get_distance(elec_pos[1], x)
            carry *= (parameters[dist_0] + parameters[dist_1]) ** (phonon_occ[(*x,)])
            return carry, x

        overlap = 1.0
        overlap, _ = lax.scan(scanned_fun, overlap, jnp.array(lattice.sites))

        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        parameters_c = parameters + 0.0j
        value, grad_fun = vjp(
            self.calc_overlap, elec_pos, phonon_occ, parameters_c, lattice
        )
        gradient = grad_fun(1.0 + 0.0j)[2]
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient / value

    def __hash__(self):
        return hash(self.n_parameters, self.n_e_bands, self.n_p_bands)


@dataclass
class merrifield_complex:
    n_parameters: int
    k: Sequence = None

    def serialize(self, parameters):
        return parameters

    def update_parameters(self, parameters, update):
        parameters += update
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        gamma = (
            parameters[: self.n_parameters // 2]
            + 1.0j * parameters[self.n_parameters // 2 :]
        )
        symm_fac = lattice.get_symm_fac(elec_pos, self.k)

        def scanned_fun(carry, x):
            dist = lattice.get_distance(elec_pos, x)
            carry *= (gamma[dist]) ** (phonon_occ[(*x,)])
            return carry, x

        overlap = 1.0 + 0.0j
        overlap, _ = lax.scan(scanned_fun, overlap, jnp.array(lattice.sites))

        return symm_fac * overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        def scanned_fun(carry, x):
            dist_0 = lattice.get_distance(elec_pos[0], x)
            dist_1 = lattice.get_distance(elec_pos[1], x)
            carry *= (parameters[dist_0] + parameters[dist_1]) ** (phonon_occ[(*x,)])
            return carry, x

        overlap = 1.0
        overlap, _ = lax.scan(scanned_fun, overlap, jnp.array(lattice.sites))

        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        # value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
        parameters_c = parameters + 0.0j
        value, grad_fun = vjp(
            self.calc_overlap, elec_pos, phonon_occ, parameters_c, lattice
        )
        gradient = grad_fun(1.0 + 0.0j)[2]
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient / value

    def __hash__(self):
        return hash(self.n_parameters)


@dataclass
class ee_jastrow:
    n_parameters: int

    def serialize(self, parameters):
        return parameters

    def update_parameters(self, parameters, update):
        parameters += update
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        gamma = (
            parameters[: self.n_parameters // 2]
            + 1.0j * parameters[self.n_parameters // 2 :]
        )
        dist = lattice.get_distance(elec_pos[0], elec_pos[1])
        overlap = jnp.exp(-gamma[dist])
        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    def __hash__(self):
        return hash(self.n_parameters)


@dataclass
class ssh_merrifield:
    n_parameters: int

    def serialize(self, parameters):
        return parameters

    def update_parameters(self, parameters, update):
        parameters += update
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        # carry: [ overlap, bond_position ]
        def scanned_fun(carry, x):
            dist = lattice.get_bond_distance(carry[1], x)
            carry[0] *= (parameters[dist]) ** (phonon_occ[(*x,)])
            return carry, x

        # carry: [ overlap ]
        def outer_scanned_fun(carry, x):
            overlap = 1.0
            [overlap, _], _ = lax.scan(
                scanned_fun, [overlap, x], jnp.array(lattice.bonds)
            )
            carry += overlap
            return carry, x

        overlap = 0.0
        neighboring_bonds = lattice.get_neighboring_bonds(elec_pos)
        overlap, _ = lax.scan(outer_scanned_fun, overlap, neighboring_bonds)

        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        neighboring_bonds_0 = lattice.get_neighboring_bonds(elec_pos[0])
        neighboring_bonds_1 = lattice.get_neighboring_bonds(elec_pos[1])

        # carry: [ overlap, bond_position_0, bond_position_1 ]
        def scanned_fun(carry, x):
            dist_0 = lattice.get_bond_distance(carry[1], x)
            dist_1 = lattice.get_bond_distance(carry[2], x)
            carry[0] *= (parameters[dist_0] + parameters[dist_1]) ** (phonon_occ[(*x,)])
            return carry, x

        # carry: [ overlap, bond_position_0 ]
        def outer_scanned_fun(carry, x):
            overlap = 1.0
            [overlap, _, _], _ = lax.scan(
                scanned_fun, [overlap, carry[1], x], jnp.array(lattice.bonds)
            )
            carry[0] += overlap
            return carry, x

        # carry: [ overlap ]
        def outer_outer_scanned_fun(carry, x):
            overlap = 0.0
            [overlap, _], _ = lax.scan(
                outer_scanned_fun, [overlap, x], neighboring_bonds_1
            )
            carry += overlap
            return carry, x

        overlap = 0.0
        overlap, _ = lax.scan(outer_outer_scanned_fun, overlap, neighboring_bonds_0)

        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient / value

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient / value

    def __hash__(self):
        return hash(self.n_parameters)


@dataclass
class bm_ssh_lf:
    n_parameters: int

    def serialize(self, parameters):
        return parameters

    def update_parameters(self, parameters, update):
        parameters += update
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        # carry: overlap
        def scanned_fun(carry, x):
            phonon_sites = lattice.get_neighboring_modes(x)
            carry += (
                ((parameters[0]) ** (phonon_occ[(*phonon_sites[0],)]))
                * ((-parameters[0]) ** (phonon_occ[(*phonon_sites[1],)]))
                * (
                    jnp.sum(phonon_occ)
                    == (
                        phonon_occ[(*phonon_sites[0],)]
                        + phonon_occ[(*phonon_sites[1],)]
                    )
                )
            )
            return carry, x

        overlap = 0.0
        neighboring_bonds = lattice.get_neighboring_bonds(elec_pos)
        overlap, _ = lax.scan(scanned_fun, overlap, neighboring_bonds)

        return overlap

    # needs to be fixed
    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        neighboring_bonds_0 = lattice.get_neighboring_bonds(elec_pos[0])
        neighboring_bonds_1 = lattice.get_neighboring_bonds(elec_pos[1])

        # carry: [ overlap, bond_position_0 ]
        # x: bond_position_1
        def scanned_fun(carry, x):
            phonon_sites_0 = lattice.get_neighboring_modes(carry[1])
            phonon_sites_1 = lattice.get_neighboring_modes(x)
            shift_l_0 = lax.cond(
                phonon_occ[(*phonon_sites_0[0],)] == 0,
                lambda w: 1.0,
                lambda w: parameters[0],
                0,
            )
            shift_r_0 = lax.cond(
                phonon_occ[(*phonon_sites_0[1],)] == 0,
                lambda w: 1.0,
                lambda w: -parameters[0],
                0,
            )
            shift_l_1 = lax.cond(
                phonon_occ[(*phonon_sites_1[0],)] == 0,
                lambda w: 1.0,
                lambda w: parameters[0],
                0,
            )
            shift_r_1 = lax.cond(
                phonon_occ[(*phonon_sites_1[1],)] == 0,
                lambda w: 1.0,
                lambda w: -parameters[0],
                0,
            )
            term_0 = (
                jnp.product(phonon_sites_0[0] == phonon_sites_1[0])
                * jnp.product(phonon_sites_0[1] == phonon_sites_1[1])
                * (shift_l_0 + shift_l_1) ** (phonon_occ[(*phonon_sites_0[0],)])
                * (shift_r_0 + shift_r_1) ** (phonon_occ[(*phonon_sites_0[1],)])
            )
            constraint_0 = jnp.sum(phonon_occ) == (
                phonon_occ[(*phonon_sites_0[0],)] + phonon_occ[(*phonon_sites_0[1],)]
            )
            term_1 = (
                jnp.product(phonon_sites_0[0] == phonon_sites_1[1])
                * jnp.product(phonon_sites_0[1] == phonon_sites_1[0])
                * (shift_l_0 + shift_r_1) ** (phonon_occ[(*phonon_sites_0[0],)])
                * (shift_r_0 + shift_l_1) ** (phonon_occ[(*phonon_sites_0[1],)])
            )
            constraint_1 = jnp.sum(phonon_occ) == (
                phonon_occ[(*phonon_sites_0[0],)] + phonon_occ[(*phonon_sites_0[1],)]
            )
            term_2 = (
                jnp.product(phonon_sites_0[0] == phonon_sites_1[0])
                * jnp.product(phonon_sites_0[1] != phonon_sites_1[1])
                * (shift_l_0 + shift_l_1) ** (phonon_occ[(*phonon_sites_0[0],)])
                * shift_r_0 ** (phonon_occ[(*phonon_sites_0[1],)])
                * shift_r_1 ** (phonon_occ[(*phonon_sites_1[1],)])
            )
            constraint_2 = jnp.sum(phonon_occ) == (
                phonon_occ[(*phonon_sites_0[0],)]
                + phonon_occ[(*phonon_sites_0[1],)]
                + phonon_occ[(*phonon_sites_1[1],)]
            )
            term_3 = (
                jnp.product(phonon_sites_0[0] == phonon_sites_1[1])
                * jnp.product(phonon_sites_0[1] != phonon_sites_1[0])
                * (shift_l_0 + shift_r_1) ** (phonon_occ[(*phonon_sites_0[0],)])
                * shift_r_0 ** (phonon_occ[(*phonon_sites_0[1],)])
                * shift_l_1 ** (phonon_occ[(*phonon_sites_1[0],)])
            )
            constraint_3 = jnp.sum(phonon_occ) == (
                phonon_occ[(*phonon_sites_0[0],)]
                + phonon_occ[(*phonon_sites_0[1],)]
                + phonon_occ[(*phonon_sites_1[0],)]
            )
            term_4 = (
                jnp.product(phonon_sites_0[0] != phonon_sites_1[0])
                * jnp.product(phonon_sites_0[1] == phonon_sites_1[1])
                * shift_l_0 ** (phonon_occ[(*phonon_sites_0[0],)])
                * (shift_r_0 + shift_r_1) ** (phonon_occ[(*phonon_sites_0[1],)])
                * (shift_l_1 + shift_l_0) ** (phonon_occ[(*phonon_sites_1[0],)])
            )
            constraint_4 = jnp.sum(phonon_occ) == (
                phonon_occ[(*phonon_sites_0[0],)]
                + phonon_occ[(*phonon_sites_0[1],)]
                + phonon_occ[(*phonon_sites_1[0],)]
            )
            term_5 = (
                jnp.product(phonon_sites_0[0] != phonon_sites_1[1])
                * jnp.product(phonon_sites_0[1] == phonon_sites_1[0])
                * shift_l_0 ** (phonon_occ[(*phonon_sites_0[0],)])
                * (shift_r_0 + shift_l_1) ** (phonon_occ[(*phonon_sites_0[1],)])
                * (shift_l_1 + shift_r_0) ** (phonon_occ[(*phonon_sites_1[1],)])
            )
            constraint_5 = jnp.sum(phonon_occ) == (
                phonon_occ[(*phonon_sites_0[0],)]
                + phonon_occ[(*phonon_sites_0[1],)]
                + phonon_occ[(*phonon_sites_1[1],)]
            )
            term_6 = (
                jnp.product(phonon_sites_0[0] != phonon_sites_1[0])
                * jnp.product(phonon_sites_0[1] != phonon_sites_1[1])
                * shift_l_0 ** (phonon_occ[(*phonon_sites_0[0],)])
                * shift_r_0 ** (phonon_occ[(*phonon_sites_0[1],)])
                * shift_l_1 ** (phonon_occ[(*phonon_sites_1[0],)])
                * shift_r_1 ** (phonon_occ[(*phonon_sites_1[1],)])
            )
            constraint_6 = jnp.sum(phonon_occ) == (
                phonon_occ[(*phonon_sites_0[0],)]
                + phonon_occ[(*phonon_sites_0[1],)]
                + phonon_occ[(*phonon_sites_1[0],)]
                + phonon_occ[(*phonon_sites_1[1],)]
            )
            carry += (
                term_0 * constraint_0
                + term_1 * constraint_1
                + term_2 * constraint_2
                + term_3 * constraint_3
                + term_4 * constraint_4
                + term_5 * constraint_5
                + term_6 * constraint_6
            )

            return carry, x

        # carry: overlap
        def outer_scanned_fun(carry, x):
            overlap_0 = 0.0
            [overlap_0, _], _ = lax.scan(
                scanned_fun, [overlap_0, x], neighboring_bonds_1
            )
            carry += overlap_0
            return carry, x

        overlap = 0.0
        overlap, _ = lax.scan(outer_scanned_fun, overlap, neighboring_bonds_0)

        return overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    def __hash__(self):
        return hash(self.n_parameters)


@dataclass
class bm_ssh_merrifield:
    n_parameters: int
    k: Sequence = None

    def serialize(self, parameters):
        return parameters

    def update_parameters(self, parameters, update):
        parameters += update
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        gamma = (
            parameters[: self.n_parameters // 2]
            + 1.0j * parameters[self.n_parameters // 2 :]
        )
        symm_fac = lattice.get_symm_fac(elec_pos, self.k)

        # carry : [ overlap, bond ]
        def scanned_fun(carry, x):
            dist, lr = lattice.get_bond_mode_distance(carry[1], x)
            # this deals with the strange 0^0 grad nan issue
            shift = lax.cond(
                phonon_occ[(*x,)] == 0, lambda w: 1.0 + 0.0j, lambda w: gamma[dist], 0
            )
            carry[0] *= (lr * shift) ** (phonon_occ[(*x,)])
            return carry, x

        # carry : [ overlap ]
        def outer_scanned_fun(carry, x):
            overlap_bond = 1.0 + 0.0j
            [overlap_bond, _], _ = lax.scan(
                scanned_fun, [overlap_bond, x], jnp.array(lattice.bonds)
            )
            carry += overlap_bond
            return carry, x

        overlap = 0.0j
        neighboring_bonds = lattice.get_neighboring_bonds(elec_pos)
        overlap, _ = lax.scan(outer_scanned_fun, overlap, neighboring_bonds)

        return symm_fac * overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        gamma = (
            parameters[: self.n_parameters // 2]
            + 1.0j * parameters[self.n_parameters // 2 :]
        )
        symm_fac = lattice.get_symm_fac(elec_pos[0], self.k)
        neighboring_bonds_0 = lattice.get_neighboring_bonds(elec_pos[0])
        neighboring_bonds_1 = lattice.get_neighboring_bonds(elec_pos[1])

        # carry: [ overlap, bond_position_0, bond_position_1 ]
        def scanned_fun(carry, x):
            dist_0, lr_0 = lattice.get_bond_mode_distance(carry[1], x)
            dist_1, lr_1 = lattice.get_bond_mode_distance(carry[2], x)
            # this deals with the strange 0^0 grad nan issue
            shift = lax.cond(
                phonon_occ[(*x,)] == 0,
                lambda w: 1.0 + 0.0j,
                lambda w: lr_0 * gamma[dist_0] + lr_1 * gamma[dist_1],
                0,
            )
            carry[0] *= (shift) ** (phonon_occ[(*x,)])
            return carry, x

        # carry: [ overlap, bond_position_0 ]
        def outer_scanned_fun(carry, x):
            overlap = 1.0
            [overlap, _, _], _ = lax.scan(
                scanned_fun, [overlap, carry[1], x], jnp.array(lattice.bonds)
            )
            carry[0] += overlap
            return carry, x

        # carry: [ overlap ]
        def outer_outer_scanned_fun(carry, x):
            overlap = 0.0
            [overlap, _], _ = lax.scan(
                outer_scanned_fun, [overlap, x], neighboring_bonds_1
            )
            carry += overlap
            return carry, x

        overlap = 0.0
        overlap, _ = lax.scan(outer_outer_scanned_fun, overlap, neighboring_bonds_0)

        return symm_fac * overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        # value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
        value, grad_fun = vjp(
            self.calc_overlap, elec_pos, phonon_occ, parameters, lattice
        )
        gradient = grad_fun(jnp.array([1.0 + 0.0j], dtype="complex64")[0])[2]
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        # value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
        value, grad_fun = vjp(
            self.calc_overlap_map, elec_pos, phonon_occ, parameters, lattice
        )
        gradient = grad_fun(jnp.array([1.0 + 0.0j], dtype="complex64")[0])[2]
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    def save_params(self, parameters, filename="parameters.bin"):
        with open(filename, "wb") as f:
            pickle.dump(parameters, f)

    def load_params(self, filename="parameters.bin"):
        parameters = None
        with open(filename, "rb") as f:
            parameters = pickle.load(f)
        return parameters

    def __hash__(self):
        return hash((self.n_parameters, self.k))


@dataclass
class nn_jastrow:
    nn_apply: Callable
    reference: Any
    n_parameters: int
    get_input: Callable = get_input_k

    def __post_init__(self):
        self.n_parameters += self.reference.n_parameters

    def serialize(self, parameters):
        flat_tree = tree_util.tree_flatten(parameters[1])[0]
        serialized = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized = jnp.concatenate((serialized, jnp.reshape(params, -1)))
        serialized = jnp.concatenate(
            (self.reference.serialize(parameters[0]), serialized)
        )
        return serialized

    # update is serialized, parameters are not
    def update_parameters(self, parameters, update):
        parameters[0] = self.reference.update_parameters(
            parameters[0], update[: self.reference.n_parameters]
        )
        # parameters[1] = models.update_nn(parameters[1], update[parameters[0].size:])
        flat_tree, tree = tree_util.tree_flatten(parameters[1])
        counter = self.reference.n_parameters
        for i in range(len(flat_tree)):
            flat_tree[i] += update[counter : counter + flat_tree[i].size].reshape(
                flat_tree[i].shape
            )
            counter += flat_tree[i].size
        parameters[1] = tree_util.tree_unflatten(tree, flat_tree)
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        nn = parameters[1]
        inputs = self.get_input(elec_pos, phonon_occ, lattice.shape)
        outputs = jnp.array(self.nn_apply(nn, inputs), dtype="float32")
        jastrow = jnp.exp(outputs[0])

        ref_overlap = self.reference.calc_overlap(
            elec_pos, phonon_occ, parameters[0], lattice
        )

        return jastrow * ref_overlap

    @partial(jit, static_argnums=(0, 6))
    def calc_overlap_ratio(
        self,
        elec_pos_old,
        elec_pos_new,
        phonon_pos,
        phonon_change,
        parameters,
        lattice,
        overlap_old,
        phonon_occ_new,
    ):
        overlap_new = self.calc_overlap(
            elec_pos_new, phonon_occ_new, parameters, lattice
        )
        return overlap_new / overlap_old

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient / value

    def __hash__(self):
        return hash((self.nn_apply, self.reference, self.n_parameters))


@dataclass
class nn_jastrow_complex:
    nn_apply_r: Callable
    nn_apply_phi: Callable
    reference: Any
    n_parameters: int
    lattice_shape: Sequence
    mask: Sequence = None
    k: Sequence = None
    get_input: Callable = get_input_n_k

    def __post_init__(self):
        self.n_parameters += self.reference.n_parameters
        if self.mask is None:
            self.mask = jnp.array([1.0, 1.0])

    def serialize(self, parameters):
        flat_tree = tree_util.tree_flatten(parameters[1])[0]
        serialized_1 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_1 = jnp.concatenate((serialized_1, jnp.reshape(params, -1)))
        flat_tree = tree_util.tree_flatten(parameters[2])[0]
        serialized_2 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_2 = jnp.concatenate((serialized_2, jnp.reshape(params, -1)))
        serialized = jnp.concatenate(
            (self.reference.serialize(parameters[0]), serialized_1, serialized_2)
        )
        return serialized

    # update is serialized, parameters are not
    def update_parameters(self, parameters, update):
        parameters[0] = self.reference.update_parameters(
            parameters[0], update[: self.reference.n_parameters]
        )
        flat_tree, tree = tree_util.tree_flatten(parameters[1])
        counter = self.reference.n_parameters
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[0] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[1] = tree_util.tree_unflatten(tree, flat_tree)

        flat_tree, tree = tree_util.tree_flatten(parameters[2])
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[1] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[2] = tree_util.tree_unflatten(tree, flat_tree)
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        nn_r = parameters[1]
        nn_phi = parameters[2]
        inputs = self.get_input(elec_pos, phonon_occ, self.lattice_shape)
        outputs_r = jnp.array(self.nn_apply_r(nn_r, inputs + 0.0j), dtype="complex64")
        outputs_phi = jnp.array(
            self.nn_apply_phi(nn_phi, inputs + 0.0j), dtype="complex64"
        )
        overlap = jnp.exp(outputs_r[0]) * jnp.exp(1.0j * jnp.sum(outputs_phi))
        ref_overlap = self.reference.calc_overlap(
            elec_pos, phonon_occ, parameters[0], lattice
        )
        # symm_fac = lattice.get_symm_fac(elec_pos, self.k)
        return ref_overlap * overlap

    @partial(jit, static_argnums=(0, 6))
    def calc_overlap_ratio(
        self,
        elec_pos_old,
        elec_pos_new,
        phonon_pos,
        phonon_change,
        parameters,
        lattice,
        overlap_old,
        phonon_occ_new,
    ):
        overlap_new = self.calc_overlap(
            elec_pos_new, phonon_occ_new, parameters, lattice
        )
        return overlap_new / overlap_old

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, grad_fun = vjp(
            self.calc_overlap, elec_pos, phonon_occ, parameters, lattice
        )
        gradient = grad_fun(jnp.array([1.0 + 0.0j], dtype="complex64")[0])[2]
        # value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    def __hash__(self):
        return hash(
            (
                self.nn_apply_r,
                self.nn_apply_phi,
                self.n_parameters,
                self.lattice_shape,
                self.k,
            )
        )


@dataclass
class nn_complex:
    nn_apply_r: Callable
    nn_apply_phi: Callable
    n_parameters: int
    mask: Sequence = None
    k: Sequence = None
    get_input: Callable = get_input_r

    def __post_init__(self):
        if self.mask is None:
            self.mask = jnp.array([1.0, 1.0])

    def serialize(self, parameters):
        flat_tree = tree_util.tree_flatten(parameters[0])[0]
        serialized_1 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_1 = jnp.concatenate((serialized_1, jnp.reshape(params, -1)))
        flat_tree = tree_util.tree_flatten(parameters[1])[0]
        serialized_2 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_2 = jnp.concatenate((serialized_2, jnp.reshape(params, -1)))
        serialized = jnp.concatenate((serialized_1, serialized_2))
        return serialized

    # update is serialized, parameters are not
    def update_parameters(self, parameters, update):
        flat_tree, tree = tree_util.tree_flatten(parameters[0])
        counter = 0
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[0] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[0] = tree_util.tree_unflatten(tree, flat_tree)

        flat_tree, tree = tree_util.tree_flatten(parameters[1])
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[1] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[1] = tree_util.tree_unflatten(tree, flat_tree)
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        nn_r = parameters[0]
        nn_phi = parameters[1]
        inputs = self.get_input(elec_pos, phonon_occ, lattice.shape)
        outputs_r = jnp.array(self.nn_apply_r(nn_r, inputs + 0.0j), dtype="complex64")
        outputs_phi = jnp.array(
            self.nn_apply_phi(nn_phi, inputs + 0.0j), dtype="complex64"
        )
        overlap = jnp.exp(outputs_r[0]) * jnp.exp(1.0j * jnp.sum(outputs_phi))

        symm_fac = lattice.get_symm_fac(elec_pos, self.k)
        return overlap * symm_fac

    @partial(jit, static_argnums=(0, 6))
    def calc_overlap_ratio(
        self,
        elec_pos_old,
        elec_pos_new,
        phonon_pos,
        phonon_change,
        parameters,
        lattice,
        overlap_old,
        phonon_occ_new,
    ):
        overlap_new = self.calc_overlap(
            elec_pos_new, phonon_occ_new, parameters, lattice
        )
        return overlap_new / overlap_old

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, grad_fun = vjp(
            self.calc_overlap, elec_pos, phonon_occ, parameters, lattice
        )
        gradient = grad_fun(jnp.array([1.0 + 0.0j], dtype="complex64")[0])[2]
        # value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    def __hash__(self):
        return hash((self.nn_apply_r, self.nn_apply_phi, self.n_parameters, self.k))


@dataclass
class nn_complex_n:
    nn_apply_r: Callable
    nn_apply_phi: Callable
    n_parameters: int
    lattice_shape: Sequence
    mask: Sequence = None
    k: Sequence = None
    get_input: Callable = get_input_n_k

    def __post_init__(self):
        if self.mask is None:
            self.mask = jnp.array([1.0, 1.0])

    def serialize(self, parameters):
        flat_tree = tree_util.tree_flatten(parameters[0])[0]
        serialized_1 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_1 = jnp.concatenate((serialized_1, jnp.reshape(params, -1)))
        flat_tree = tree_util.tree_flatten(parameters[1])[0]
        serialized_2 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_2 = jnp.concatenate((serialized_2, jnp.reshape(params, -1)))
        serialized = jnp.concatenate((serialized_1, serialized_2))
        return serialized

    # update is serialized, parameters are not
    def update_parameters(self, parameters, update):
        flat_tree, tree = tree_util.tree_flatten(parameters[0])
        counter = 0
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[0] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[0] = tree_util.tree_unflatten(tree, flat_tree)

        flat_tree, tree = tree_util.tree_flatten(parameters[1])
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[1] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[1] = tree_util.tree_unflatten(tree, flat_tree)
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
        nn_r = parameters[0]
        nn_phi = parameters[1]
        inputs = self.get_input(elec_pos, phonon_occ, self.lattice_shape)
        outputs_r = jnp.array(self.nn_apply_r(nn_r, inputs + 0.0j), dtype="complex64")
        outputs_phi = jnp.array(
            self.nn_apply_phi(nn_phi, inputs + 0.0j), dtype="complex64"
        )
        overlap = jnp.exp(outputs_r[0]) * jnp.exp(1.0j * jnp.sum(outputs_phi))

        symm_fac = lattice.get_symm_fac(elec_pos, self.k)
        return overlap * symm_fac

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, grad_fun = vjp(
            self.calc_overlap, elec_pos, phonon_occ, parameters, lattice
        )
        gradient = grad_fun(jnp.array([1.0 + 0.0j], dtype="complex64")[0])[2]
        # value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient / value

    def __hash__(self):
        return hash(
            (
                self.nn_apply_r,
                self.nn_apply_phi,
                self.n_parameters,
                self.lattice_shape,
                self.k,
            )
        )


@dataclass
class nn_jastrow_2:
    nn_apply: Callable
    reference: Any
    ee_jastrow: Any
    n_parameters: int

    def __post_init__(self):
        self.n_parameters += self.reference.n_parameters + self.ee_jastrow.n_parameters

    # TODO: needs to be changed for ssh > 1d, and for 2 sites
    @partial(jit, static_argnums=(0, 3))
    def get_input_2(self, elec_pos, phonon_occ, lattice_shape):
        elec_pos_ar = jnp.zeros(lattice_shape)
        elec_pos_ar = elec_pos_ar.at[elec_pos[0]].add(1)
        elec_pos_ar = elec_pos_ar.at[elec_pos[1]].add(1)
        input_ar = jnp.stack(
            [elec_pos_ar, *phonon_occ.reshape(-1, *lattice_shape)], axis=-1
        )
        return input_ar

    def serialize(self, parameters):
        flat_tree = tree_util.tree_flatten(parameters[1])[0]
        serialized = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized = jnp.concatenate((serialized, jnp.reshape(params, -1)))
        serialized = jnp.concatenate(
            (
                self.reference.serialize(parameters[0]),
                serialized,
                self.ee_jastrow.serialize(parameters[2]),
            )
        )
        return serialized

    # update is serialized, parameters are not
    def update_parameters(self, parameters, update):
        parameters[0] = self.reference.update_parameters(
            parameters[0], update[: self.reference.n_parameters]
        )
        flat_tree, tree = tree_util.tree_flatten(parameters[1])
        counter = self.reference.n_parameters
        for i in range(len(flat_tree)):
            flat_tree[i] += update[counter : counter + flat_tree[i].size].reshape(
                flat_tree[i].shape
            )
            counter += flat_tree[i].size
        parameters[1] = tree_util.tree_unflatten(tree, flat_tree)
        parameters[2] = self.reference.update_parameters(
            parameters[2], update[counter:]
        )
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        nn = parameters[1]
        inputs = self.get_input_2(elec_pos, phonon_occ, lattice.shape)
        outputs = jnp.array(self.nn_apply(nn, inputs), dtype="float32")
        jastrow = jnp.exp(outputs[0])

        ref_overlap = self.reference.calc_overlap_map(
            elec_pos, phonon_occ, parameters[0], lattice
        )
        ee_jastrow_overlap = self.ee_jastrow.calc_overlap_map(
            elec_pos, phonon_occ, parameters[2], lattice
        )

        return ee_jastrow_overlap * jastrow * ref_overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(
            elec_pos, phonon_occ, parameters, lattice
        )
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient / value

    def __hash__(self):
        return hash((self.nn_apply, self.reference, self.ee_jastrow, self.n_parameters))


@dataclass
class nn_jastrow_2_complex:
    nn_apply_r: Callable
    nn_apply_phi: Callable
    ee_jastrow: Any
    reference: Any
    n_parameters: int
    mask: Sequence = None

    def __post_init__(self):
        self.n_parameters += self.reference.n_parameters + self.ee_jastrow.n_parameters
        if self.mask is None:
            self.mask = jnp.array([1.0, 1.0])

    @partial(jit, static_argnums=(0, 3))
    def get_input_2(self, elec_pos, phonon_occ, lattice_shape):
        elec_ar = jnp.zeros(lattice_shape)
        elec_ar = elec_ar.at[tuple(0 * jnp.array(elec_pos[0]))].set(1)
        elec_ar = elec_ar.at[
            tuple(
                (
                    jnp.array(elec_pos[1])
                    - jnp.array(elec_pos[0]) % jnp.array(lattice_shape)
                )
            )
        ].set(1)
        input_ar = phonon_occ.reshape(-1, *lattice_shape)
        for ax in range(len(lattice_shape)):
            for phonon_type in range(phonon_occ.shape[0]):
                input_ar = input_ar.at[phonon_type].set(
                    input_ar[phonon_type].take(
                        elec_pos[0][ax] + jnp.arange(lattice_shape[ax]),
                        axis=ax,
                        mode="wrap",
                    )
                )
        return jnp.stack([elec_ar, *input_ar], axis=-1)

    def serialize(self, parameters):
        flat_tree = tree_util.tree_flatten(parameters[2])[0]
        serialized_1 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_1 = jnp.concatenate((serialized_1, jnp.reshape(params, -1)))
        flat_tree = tree_util.tree_flatten(parameters[3])[0]
        serialized_2 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_2 = jnp.concatenate((serialized_2, jnp.reshape(params, -1)))
        serialized = jnp.concatenate(
            (
                self.reference.serialize(parameters[0]),
                self.ee_jastrow.serialize(parameters[1]),
                serialized_1,
                serialized_2,
            )
        )
        return serialized

    # update is serialized, parameters are not
    def update_parameters(self, parameters, update):
        parameters[0] = self.reference.update_parameters(
            parameters[0], update[: self.reference.n_parameters]
        )
        parameters[1] = self.ee_jastrow.update_parameters(
            parameters[1],
            update[
                self.reference.n_parameters : self.ee_jastrow.n_parameters
                + self.reference.n_parameters
            ],
        )
        flat_tree, tree = tree_util.tree_flatten(parameters[2])
        counter = self.reference.n_parameters + self.ee_jastrow.n_parameters
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[0] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[2] = tree_util.tree_unflatten(tree, flat_tree)

        flat_tree, tree = tree_util.tree_flatten(parameters[3])
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[1] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[3] = tree_util.tree_unflatten(tree, flat_tree)
        return parameters

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
        nn_r = parameters[2]
        nn_phi = parameters[3]
        inputs = self.get_input_2(elec_pos, phonon_occ, lattice.shape)
        outputs_r = jnp.array(self.nn_apply_r(nn_r, inputs + 0.0j), dtype="complex64")
        outputs_phi = jnp.array(
            self.nn_apply_phi(nn_phi, inputs + 0.0j), dtype="complex64"
        )
        overlap = jnp.exp(outputs_r[0]) * jnp.exp(1.0j * jnp.sum(outputs_phi))
        ref_overlap = self.reference.calc_overlap_map(
            elec_pos, phonon_occ, parameters[0], lattice
        )
        ee_overlap = self.ee_jastrow.calc_overlap_map(
            elec_pos, phonon_occ, parameters[1], lattice
        )
        return ee_overlap * ref_overlap * overlap

    @partial(jit, static_argnums=(0, 4))
    def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
        value, grad_fun = vjp(
            self.calc_overlap_map, elec_pos, phonon_occ, parameters, lattice
        )
        gradient = grad_fun(jnp.array([1.0 + 0.0j], dtype="complex64")[0])[2]
        gradient = self.serialize(gradient) / value
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        gradient = jnp.where(jnp.isinf(gradient), 0.0, gradient)
        return gradient

    def __hash__(self):
        return hash(
            (
                self.nn_apply_r,
                self.nn_apply_phi,
                self.ee_jastrow,
                self.reference,
                self.n_parameters,
            )
        )


@dataclass
class spin_nn_complex:
    nn_apply_r: Callable
    nn_apply_phi: Callable
    n_parameters: int
    mask: Sequence = None
    k: Sequence = None
    get_input: Callable = get_input_spins

    def __post_init__(self):
        if self.mask is None:
            self.mask = jnp.array([1.0, 1.0])

    def serialize(self, parameters):
        flat_tree = tree_util.tree_flatten(parameters[0])[0]
        serialized_1 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_1 = jnp.concatenate((serialized_1, jnp.reshape(params, -1)))
        flat_tree = tree_util.tree_flatten(parameters[1])[0]
        serialized_2 = jnp.reshape(flat_tree[0], (-1))
        for params in flat_tree[1:]:
            serialized_2 = jnp.concatenate((serialized_2, jnp.reshape(params, -1)))
        serialized = jnp.concatenate((serialized_1, serialized_2))
        return serialized

    # update is serialized, parameters are not
    def update_parameters(self, parameters, update):
        flat_tree, tree = tree_util.tree_flatten(parameters[0])
        counter = 0
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[0] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[0] = tree_util.tree_unflatten(tree, flat_tree)

        flat_tree, tree = tree_util.tree_flatten(parameters[1])
        for i in range(len(flat_tree)):
            flat_tree[i] += self.mask[1] * update[
                counter : counter + flat_tree[i].size
            ].reshape(flat_tree[i].shape)
            counter += flat_tree[i].size
        parameters[1] = tree_util.tree_unflatten(tree, flat_tree)
        return parameters

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap(self, walker, parameters, lattice):
        nn_r = parameters[0]
        nn_phi = parameters[1]
        inputs = self.get_input(walker, lattice.shape)
        outputs_r = jnp.array(self.nn_apply_r(nn_r, inputs + 0.0j), dtype="complex64")
        outputs_phi = jnp.array(
            self.nn_apply_phi(nn_phi, inputs + 0.0j), dtype="complex64"
        )
        # overlap = jnp.exp(outputs_r[0]) * jnp.exp(1.j * jnp.sum(outputs_phi))
        overlap = jnp.exp(outputs_r[0]) * lattice.get_marshall_sign(walker)
        return overlap

    @partial(jit, static_argnums=(0, 3))
    def calc_overlap_gradient(self, walker, parameters, lattice):
        value, grad_fun = vjp(self.calc_overlap, walker, parameters, lattice)
        gradient = grad_fun(jnp.array([1.0 + 0.0j], dtype="complex64")[0])[1]
        # value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
        gradient = self.serialize(gradient)
        gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return gradient / value

    def __hash__(self):
        return hash((self.nn_apply_r, self.nn_apply_phi, self.n_parameters, self.k))


if __name__ == "__main__":
    import lattices
    import models

    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)

    model_r = models.MLP(
        [2, 1], param_dtype=jnp.complex64, kernel_init=models.complex_kernel_init
    )
    model_phi = models.MLP(
        [2, 1],
        activation=lambda x: jnp.log(jnp.cosh(x)),
        param_dtype=jnp.complex64,
        kernel_init=models.complex_kernel_init,
    )
    model_input = jnp.zeros(n_sites)
    nn_parameters_r = model_r.init(random.PRNGKey(0), model_input, mutable=True)
    nn_parameters_phi = model_phi.init(random.PRNGKey(1), model_input, mutable=True)
    n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters_r)) + sum(
        x.size for x in tree_util.tree_leaves(nn_parameters_phi)
    )
    parameters = [nn_parameters_r, nn_parameters_phi]
    wave = nn_complex(model_r.apply, model_phi.apply, n_nn_parameters)

    elec_pos = (0,)
    phonon_occ = jnp.array([2, 0, 1, 0])
    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    gradient_ratio = wave.calc_overlap_gradient(
        elec_pos, phonon_occ, parameters, lattice
    )
    print(f"overlap: {overlap}")
    print(f"gradient: {gradient_ratio * overlap}")

    eps = 0.0001
    update = jnp.zeros(n_nn_parameters)
    update = update.at[-3].set(eps)
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

    overlap_1 = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    print(f"overlap_1: {overlap_1}")
    print(f"fd grad: {(overlap_1 - overlap) / eps}")
    # print(parameters[0])
    # print(overlap)
    # print(gradient_ratio)
