import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
from jax import random, lax, tree_util, grad, value_and_grad, jit, numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple, Callable, Any
from dataclasses import dataclass
from functools import partial

@dataclass
class merrifield():
  n_parameters: int

  def serialize(self, parameters):
    return parameters

  def update_parameters(self, parameters, update):
    parameters += update
    return parameters

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
    def scanned_fun(carry, x):
      dist = lattice.get_distance(elec_pos, x)
      carry *= (parameters[dist])**(phonon_occ[(*x,)])
      return carry, x

    overlap = 1.
    overlap, _ = lax.scan(scanned_fun, overlap, jnp.array(lattice.sites))

    return overlap

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
    def scanned_fun(carry, x):
      dist_0 = lattice.get_distance(elec_pos[0], x)
      dist_1 = lattice.get_distance(elec_pos[1], x)
      carry *= (parameters[dist_0] + parameters[dist_1])**(phonon_occ[(*x,)])
      return carry, x

    overlap = 1.
    overlap, _ = lax.scan(scanned_fun, overlap, jnp.array(lattice.sites))

    return overlap

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
    value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
    gradient = self.serialize(gradient)
    gradient = jnp.where(jnp.isnan(gradient), 0., gradient)
    return gradient / value

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
    value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
    gradient = self.serialize(gradient)
    gradient = jnp.where(jnp.isnan(gradient), 0., gradient)
    return gradient / value

  def __hash__(self):
    return hash(self.n_parameters)

@dataclass
class ee_jastrow():
  n_parameters: int

  def serialize(self, parameters):
    return parameters

  def update_parameters(self, parameters, update):
    parameters += update
    return parameters

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
    dist = lattice.get_distance(elec_pos[0], elec_pos[1])
    overlap = jnp.exp(-parameters[dist])# * (jnp.sum(phonon_occ) == 0)
    return overlap

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
    value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
    gradient = self.serialize(gradient)
    gradient = jnp.where(jnp.isnan(gradient), 0., gradient)
    return gradient / value

  def __hash__(self):
    return hash(self.n_parameters)


@dataclass
class ssh_merrifield():
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
      carry[0] *= (parameters[dist])**(phonon_occ[(*x,)])
      return carry, x

    # carry: [ overlap ]
    def outer_scanned_fun(carry, x):
      overlap = 1.
      [ overlap, _ ], _ = lax.scan(scanned_fun, [ overlap, x ], jnp.array(lattice.bonds))
      carry += overlap
      return carry, x

    overlap = 0.
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
      carry[0] *= (parameters[dist_0] + parameters[dist_1])**(phonon_occ[(*x,)])
      return carry, x

    # carry: [ overlap, bond_position_0 ]
    def outer_scanned_fun(carry, x):
      overlap = 1.
      [ overlap, _, _ ], _ = lax.scan(scanned_fun, [ overlap, carry[1], x ], jnp.array(lattice.bonds))
      carry[0] += overlap
      return carry, x

    # carry: [ overlap ]
    def outer_outer_scanned_fun(carry, x):
      overlap = 0.
      [ overlap, _ ], _ = lax.scan(outer_scanned_fun, [ overlap, x ], neighboring_bonds_1)
      carry += overlap
      return carry, x

    overlap = 0.
    overlap, _ = lax.scan(outer_outer_scanned_fun, overlap, neighboring_bonds_0)

    return overlap

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
    value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
    gradient = self.serialize(gradient)
    gradient = jnp.where(jnp.isnan(gradient), 0., gradient)
    return gradient / value

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
    value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
    gradient = self.serialize(gradient)
    gradient = jnp.where(jnp.isnan(gradient), 0., gradient)
    return gradient / value

  def __hash__(self):
    return hash(self.n_parameters)


@dataclass
class nn_jastrow():
  nn_apply: Callable
  reference: Any
  n_parameters: int

  def __post_init__(self):
    self.n_parameters += self.reference.n_parameters

  # TODO: needs to be changed for 2 sites
  @partial(jit, static_argnums=(0, 3))
  def get_input(self, elec_pos, phonon_occ, lattice_shape):
    elec_pos_ar = jnp.zeros(lattice_shape)
    elec_pos_ar = elec_pos_ar.at[elec_pos].set(1)
    input_ar = jnp.stack([ elec_pos_ar, *phonon_occ.reshape(-1, *lattice_shape) ], axis=-1)
    return input_ar

  def serialize(self, parameters):
    flat_tree = tree_util.tree_flatten(parameters[1])[0]
    serialized = jnp.reshape(flat_tree[0], (-1))
    for params in flat_tree[1:]:
      serialized = jnp.concatenate((serialized, jnp.reshape(params, -1)))
    serialized = jnp.concatenate((self.reference.serialize(parameters[0]), serialized))
    return serialized

  # update is serialized, parameters are not
  def update_parameters(self, parameters, update):
    parameters[0] = self.reference.update_parameters(parameters[0], update[:self.reference.n_parameters])
    #parameters[1] = models.update_nn(parameters[1], update[parameters[0].size:])
    flat_tree, tree = tree_util.tree_flatten(parameters[1])
    counter = self.reference.n_parameters
    for i in range(len(flat_tree)):
      flat_tree[i] += update[counter: counter + flat_tree[i].size].reshape(flat_tree[i].shape)
      counter += flat_tree[i].size
    parameters[1] = tree_util.tree_unflatten(tree, flat_tree)
    return parameters

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap(self, elec_pos, phonon_occ, parameters, lattice):
    nn = parameters[1]
    inputs = self.get_input(elec_pos, phonon_occ, lattice.shape)
    outputs = jnp.array(self.nn_apply(nn, inputs), dtype='float64')
    jastrow = jnp.exp(outputs[0])

    ref_overlap = self.reference.calc_overlap(elec_pos, phonon_occ, parameters[0], lattice)

    return jastrow * ref_overlap

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
    value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
    gradient = self.serialize(gradient)
    gradient = jnp.where(jnp.isnan(gradient), 0., gradient)
    return gradient / value

  def __hash__(self):
    return hash((self.nn_apply, self.reference, self.n_parameters))


@dataclass
class nn_jastrow_2():
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
    input_ar = jnp.stack([ elec_pos_ar, *phonon_occ.reshape(-1, *lattice_shape) ], axis=-1)
    return input_ar

  def serialize(self, parameters):
    flat_tree = tree_util.tree_flatten(parameters[1])[0]
    serialized = jnp.reshape(flat_tree[0], (-1))
    for params in flat_tree[1:]:
      serialized = jnp.concatenate((serialized, jnp.reshape(params, -1)))
    serialized = jnp.concatenate((self.reference.serialize(parameters[0]), serialized, self.ee_jastrow.serialize(parameters[2])))
    return serialized

  # update is serialized, parameters are not
  def update_parameters(self, parameters, update):
    parameters[0] = self.reference.update_parameters(parameters[0], update[:self.reference.n_parameters])
    flat_tree, tree = tree_util.tree_flatten(parameters[1])
    counter = self.reference.n_parameters
    for i in range(len(flat_tree)):
      flat_tree[i] += update[counter: counter + flat_tree[i].size].reshape(flat_tree[i].shape)
      counter += flat_tree[i].size
    parameters[1] = tree_util.tree_unflatten(tree, flat_tree)
    parameters[2] = self.reference.update_parameters(parameters[2], update[counter:])
    return parameters

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap_map(self, elec_pos, phonon_occ, parameters, lattice):
    nn = parameters[1]
    inputs = self.get_input_2(elec_pos, phonon_occ, lattice.shape)
    outputs = jnp.array(self.nn_apply(nn, inputs), dtype='float64')
    jastrow = jnp.exp(outputs[0])

    ref_overlap = self.reference.calc_overlap_map(elec_pos, phonon_occ, parameters[0], lattice)
    ee_jastrow_overlap = self.ee_jastrow.calc_overlap_map(elec_pos, phonon_occ, parameters[2], lattice)

    return ee_jastrow_overlap * jastrow * ref_overlap

  @partial(jit, static_argnums=(0, 4))
  def calc_overlap_map_gradient(self, elec_pos, phonon_occ, parameters, lattice):
    value, gradient = value_and_grad(self.calc_overlap_map, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
    gradient = self.serialize(gradient)
    gradient = jnp.where(jnp.isnan(gradient), 0., gradient)
    return gradient / value

  def __hash__(self):
    return hash((self.nn_apply, self.reference, self.ee_jastrow, self.n_parameters))
