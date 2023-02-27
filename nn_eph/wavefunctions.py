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
  def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
    value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
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
      # TODO: needs to be changed for > 1d
      dist = lattice.get_distance(carry[1], x)
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
  def calc_overlap_gradient(self, elec_pos, phonon_occ, parameters, lattice):
    value, gradient = value_and_grad(self.calc_overlap, argnums=2)(elec_pos, phonon_occ, parameters, lattice)
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

  # TODO: needs to be changed for ssh > 1d, and for 2 sites
  @partial(jit, static_argnums=(0, 3))
  def get_input(self, elec_pos, phonon_occ, lattice_shape):
    elec_pos_ar = jnp.zeros(lattice_shape)
    elec_pos_ar = elec_pos_ar.at[elec_pos].set(1)
    input_ar = jnp.array([ elec_pos_ar, phonon_occ ])
    return input_ar.reshape(1, *input_ar.shape)

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

if __name__ == "__main__":
  import lattices, models
  n_sites = 2
  lattice = lattices.one_dimensional_chain(n_sites)
  np.random.seed(3)
  elec_pos = 0
  phonon_occ = jnp.array(np.random.randint(3, size=(n_sites,)))
  gamma = jnp.array(np.random.rand(n_sites // 2 + 1))
  #parameters = gamma
  #wave = ssh_merrifield(gamma.size)
  model = models.MLP([5, 1])
  model_input = jnp.zeros(2*n_sites)
  nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
  n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
  parameters = [ gamma, nn_parameters ]
  reference = ssh_merrifield(gamma.size)
  wave = nn_jastrow(model.apply, reference, n_nn_parameters)
  print(wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice))

  exit()

  l_x, l_y = 2, 2
  n_sites = l_x * l_y
  lattice = lattices.two_dimensional_grid(l_x, l_y)
  np.random.seed(3)
  elec_pos = (1, 0)
  phonon_occ = jnp.array(np.random.randint(3, size=(l_y, l_x)))
  gamma = jnp.array(np.random.rand(len(lattice.shell_distances)))
  model = models.MLP([5, 1])
  model_input = jnp.zeros(2*n_sites)
  nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
  #parameters = [ gamma, nn_parameters ]
  #reference = merrifield()
  #wave = nn_jastrow(model.apply, reference)
  parameters = gamma
  wave = merrifield(gamma.size)
