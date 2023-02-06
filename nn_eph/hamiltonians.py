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
class holstein_1d():
  omega: float
  g: float

  @partial(jit, static_argnums = (0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1:]
    n_sites = phonon_occ.shape[0]

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('lf_overlap: {}', lf_overlap)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    # electron hops
    new_elec_pos = (elec_pos + 1) % n_sites
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0 = lax.cond(n_sites > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0

    new_elec_pos = (elec_pos - 1) % n_sites
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1 = lax.cond(n_sites > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1

    # e_ph coupling
    new_phonon_occ = phonon_occ.at[elec_pos].add(1)
    ratio_2 = wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos] + 1)**0.5
    energy -= self.g * (phonon_occ[elec_pos] + 1)**0.5 * ratio_2

    new_phonon_occ = phonon_occ.at[elec_pos].add(-1)
    ratio_3 = (phonon_occ[elec_pos])**0.5 * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap
    energy -= self.g * (phonon_occ[elec_pos])**0.5 * ratio_3

    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3 ])))
    #jax.debug.print('ratios: {}', jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3 ]))
    #jax.debug.print('energy: {}', energy)
    weight = 1 / cumulative_ratios[-1]
    #jax.debug.print('weight: {}', weight)
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)

    walker = lax.cond(new_ind < 2, lambda x: x.at[0].set((elec_pos + 1 - 2*new_ind) % n_sites), lambda x: x.at[1 + elec_pos].add(-2*new_ind + 5), walker)
    walker = jnp.where(walker < 0, 0, walker)

    return energy, qp_weight, overlap_gradient, weight, walker

  def __hash__(self):
    return hash((self.omega, self.g))


@dataclass
class holstein_2d():
  omega: float
  g: float

  @partial(jit, static_argnums=(0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1]
    (l_y, l_x) = phonon_occ.shape

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('lf_ovlp: {}', lf_overlap)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    # electron hops
    new_elec_pos = ( (elec_pos[0] + 1) % l_y, elec_pos[1] )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0_y = lax.cond(l_y > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0_y

    new_elec_pos = ( (elec_pos[0] - 1) % l_y, elec_pos[1] )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1_y = lax.cond(l_y > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1_y

    new_elec_pos = ( elec_pos[0], (elec_pos[1] + 1) % l_x )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0_x = lax.cond(l_x > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0_x

    new_elec_pos = ( elec_pos[0], (elec_pos[1] - 1) % l_x )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1_x = lax.cond(l_x > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1_x

    # e_ph coupling
    new_phonon_occ = phonon_occ.at[elec_pos[0], elec_pos[1]].add(1)
    ratio_2 = wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos[0], elec_pos[1]] + 1)**0.5
    energy -= self.g * (phonon_occ[elec_pos[0], elec_pos[1]] + 1)**0.5 * ratio_2

    new_phonon_occ = phonon_occ.at[elec_pos[0], elec_pos[1]].add(-1)
    ratio_3 = (phonon_occ[elec_pos[0], elec_pos[1]])**0.5 * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap
    energy -= self.g * (phonon_occ[elec_pos[0], elec_pos[1]])**0.5 * ratio_3

    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array([ ratio_0_x, ratio_1_x, ratio_0_y, ratio_1_y, ratio_2, ratio_3 ])))
    #jax.debug.print('ratios: {}', jnp.array([ ratio_0_x, ratio_1_x, ratio_0_y, ratio_1_y, ratio_2, ratio_3  ]))
    #jax.debug.print('energy: {}', energy)
    weight = 1 / cumulative_ratios[-1]
    #jax.debug.print('weight: {}', weight)
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)

    walker[0] = jnp.array(walker[0])
    # x ->
    walker[0] = lax.cond(new_ind == 0, lambda w: w.at[1].set((elec_pos[1] + 1) % l_x), lambda w: w, walker[0])
    # x <-
    walker[0] = lax.cond(new_ind == 1, lambda w: w.at[1].set((elec_pos[1] - 1) % l_x), lambda w: w, walker[0])
    # y ->
    walker[0] = lax.cond(new_ind == 2, lambda w: w.at[0].set((elec_pos[0] + 1) % l_y), lambda w: w, walker[0])
     # y <-
    walker[0] = lax.cond(new_ind == 3, lambda w: w.at[0].set((elec_pos[0] - 1) % l_y), lambda w: w, walker[0])
    walker[0] = (walker[0][0], walker[0][1])

    # phonon
    walker[1] = lax.cond(new_ind == 4, lambda w: w.at[elec_pos[0], elec_pos[1]].add(1), lambda w: w, walker[1])
    walker[1] = lax.cond(new_ind == 5, lambda w: w.at[elec_pos[0], elec_pos[1]].add(-1), lambda w: w, walker[1])
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

    return energy, qp_weight, overlap_gradient, weight, walker

  def __hash__(self):
    return hash((self.omega, self.g))


if __name__ == "__main__":
  import lattices, models, wavefunctions
  #n_sites = 6
  #ham = holstein_1d(1., 1.)
  #lattice = lattices.one_dimensional_chain(n_sites)
  #np.random.seed(3)
  #elec_pos = 1
  #phonon_occ = jnp.array(np.random.randint(3, size=n_sites))
  #gamma = jnp.array(np.random.rand(n_sites // 2 + 1))
  #parameters = gamma
  #wave = wavefunctions.merrifield()
  #walker = jnp.array([ elec_pos ] + list(phonon_occ))

  l_x, l_y = 2, 2
  n_sites = l_x * l_y
  ham = holstein_2d(1., 1.)
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
  random_number = 0.5
  energy, qp_weight, overlap_gradient, weight, walker = ham.local_energy_and_update(walker, parameters, wave, lattice, random_number)
