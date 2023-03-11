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
class continuous_time():
  n_eql: int
  n_samples: int

  @partial(jit, static_argnums=(0, 2, 4, 5))
  def sampling(self, walker, ham, parameters, wave, lattice, random_numbers):
    # carry : [ walker, weight, energy, grad, lene_grad, qp_weight ]
    def scanned_fun(carry, x):
      energy, qp_weight, gradient, weight, carry[0] = ham.local_energy_and_update(carry[0], parameters, wave, lattice, random_numbers[x])
      carry[1] += weight
      carry[2] += weight * (jnp.real(energy) - carry[2]) / carry[1]
      carry[3] = carry[3] + weight * (jnp.real(gradient) - carry[3]) / carry[1]
      carry[4] = carry[4] + weight * (jnp.real(jnp.conjugate(energy) * gradient) - carry[4]) / carry[1]
      carry[5] += weight * (qp_weight - carry[5]) / carry[1]
      return carry, (jnp.real(energy), qp_weight, weight)

    weight = 0.
    energy = 0.
    gradient = jnp.zeros(wave.n_parameters)
    lene_gradient = jnp.zeros(wave.n_parameters)
    qp_weight = 0.
    [walker, _, _, _, _, _] , (_, _, _) = lax.scan(scanned_fun, [ walker, weight, energy, gradient, lene_gradient, qp_weight ], jnp.arange(self.n_eql))

    weight = 0.
    energy = 0.
    gradient = jnp.zeros(wave.n_parameters)
    lene_gradient = jnp.zeros(wave.n_parameters)
    qp_weight = 0.
    [_, weight, energy, gradient, lene_gradient, qp_weight] , (energies, qp_weights,  weights) = lax.scan(scanned_fun, [ walker, weight, energy, gradient, lene_gradient, qp_weight ], jnp.arange(self.n_samples))

    # energy, gradient, lene_gradient are weighted
    return weight, energy, gradient, lene_gradient, qp_weight, energies, qp_weights, weights

  def __hash__(self):
    return hash((self.n_eql, self.n_samples))

if __name__ == "__main__":
  import lattices, models, wavefunctions, hamiltonians
  l_x, l_y = 2, 2
  n_sites = l_x * l_y
  n_eql = 10
  n_samples = 100
  sampler = continuous_time(10, 100)
  key = random.PRNGKey(0)
  random_numbers = random.uniform(key, shape=(n_samples,))
  ham = hamiltonians.holstein_2d(1., 1.)
  lattice = lattices.two_dimensional_grid(l_x, l_y)
  np.random.seed(3)
  elec_pos = (1, 0)
  phonon_occ = jnp.array(np.random.randint(3, size=(l_y, l_x)))
  gamma = jnp.array(np.random.rand(len(lattice.shell_distances)))
  model = models.MLP([5, 1])
  model_input = jnp.zeros(2*n_sites)
  nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
  n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters))
  parameters = [ gamma, nn_parameters ]
  reference = wavefunctions.merrifield(gamma.size)
  wave = wavefunctions.nn_jastrow(model.apply, reference, n_nn_parameters)
  walker = [ elec_pos, phonon_occ ]
  sampler.sampling(walker, ham, parameters, wave, lattice, random_numbers)

