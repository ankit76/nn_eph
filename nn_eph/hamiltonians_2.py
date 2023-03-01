import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
import jax
from jax import random, lax, tree_util, grad, value_and_grad, jit, numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple, Callable, Any
from dataclasses import dataclass
from functools import partial

@dataclass
class holstein_1d_2():
  omega: float
  g: float
  u: float
  triplet: bool = False

  @partial(jit, static_argnums = (0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[:2]
    phonon_occ = walker[2:]
    n_sites = phonon_occ.shape[0]

    overlap = wave.calc_overlap_map(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_map_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ) + self.u * (elec_pos[0] == elec_pos[1])

    # electron hops
    new_elec_pos = elec_pos.at[0].set((elec_pos[0] + 1) % n_sites)
    new_overlap = wave.calc_overlap_map(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0 = lax.cond(n_sites > 1, lambda x: x / overlap, lambda x: 0., new_overlap) * (1 - self.triplet * (new_elec_pos[0] == new_elec_pos[1]))
    energy -= ratio_0

    new_elec_pos = elec_pos.at[0].set((elec_pos[0] - 1) % n_sites)
    new_overlap = wave.calc_overlap_map(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1 = lax.cond(n_sites > 2, lambda x: x / overlap, lambda x: 0., new_overlap) * (1 - self.triplet * (new_elec_pos[0] == new_elec_pos[1]))
    energy -= ratio_1

    new_elec_pos = elec_pos.at[1].set((elec_pos[1] + 1) % n_sites)
    new_overlap = wave.calc_overlap_map(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_2 = lax.cond(n_sites > 1, lambda x: x / overlap, lambda x: 0., new_overlap) * (1 - self.triplet * (new_elec_pos[0] == new_elec_pos[1]))
    energy -= ratio_2

    new_elec_pos = elec_pos.at[1].set((elec_pos[1] - 1) % n_sites)
    new_overlap = wave.calc_overlap_map(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_3 = lax.cond(n_sites > 2, lambda x: x / overlap, lambda x: 0., new_overlap) * (1 - self.triplet * (new_elec_pos[0] == new_elec_pos[1]))
    energy -= ratio_3

    # e_ph coupling
    new_phonon_occ = phonon_occ.at[elec_pos[0]].add(1)
    ratio_4 = wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos[0]] + 1)**0.5
    energy -= self.g * (phonon_occ[elec_pos[0]] + 1)**0.5 * ratio_4

    new_phonon_occ = phonon_occ.at[elec_pos[0]].add(-1)
    ratio_5 = (phonon_occ[elec_pos[0]])**0.5 * wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice) / overlap
    energy -= self.g * (phonon_occ[elec_pos[0]])**0.5 * ratio_5

    new_phonon_occ = phonon_occ.at[elec_pos[1]].add(1)
    ratio_6 = wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos[1]] + 1)**0.5
    energy -= self.g * (phonon_occ[elec_pos[1]] + 1)**0.5 * ratio_6

    new_phonon_occ = phonon_occ.at[elec_pos[1]].add(-1)
    ratio_7 = (phonon_occ[elec_pos[1]])**0.5 * wave.calc_overlap_map(elec_pos, new_phonon_occ, parameters, lattice) / overlap
    energy -= self.g * (phonon_occ[elec_pos[1]])**0.5 * ratio_7

    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3, ratio_4, ratio_5, ratio_6, ratio_7 ])))
    #jax.debug.print('ratios: {}', jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3 ]))
    #jax.debug.print('energy: {}', energy)
    weight = 1 / cumulative_ratios[-1]
    #jax.debug.print('weight: {}', weight)
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)

    walker = lax.cond(new_ind < 2, lambda x: x.at[0].set((elec_pos[0] + 1 - 2*new_ind) % n_sites), lambda x: x, walker)
    walker = lax.cond((new_ind - 1) * (new_ind - 4) < 0, lambda x: x.at[1].set((elec_pos[1] + 5 - 2*new_ind) % n_sites), lambda x: x, walker)
    walker = lax.cond((new_ind - 3) * (new_ind - 6) < 0, lambda x: x.at[2 + elec_pos[0]].add(-2*new_ind + 9), lambda x: x, walker)
    walker = lax.cond(new_ind > 5, lambda x: x.at[2 + elec_pos[1]].add(-2*new_ind + 13), lambda x: x, walker)
    walker = jnp.where(walker < 0, 0, walker)

    return energy, qp_weight, overlap_gradient, weight, walker

  def __hash__(self):
    return hash((self.omega, self.g))
