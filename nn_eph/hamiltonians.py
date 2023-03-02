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
class ssh_1d():
  omega: float
  g: float

  @partial(jit, static_argnums = (0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1:]
    n_sites = lattice.n_sites

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)

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
    # electron hop ->
    new_elec_pos = (elec_pos + 1) % n_sites
    bond_pos = elec_pos * (1 - (n_sites==2))

    new_phonon_occ = phonon_occ.at[bond_pos].add(1)
    ratio_2 = wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos] + 1)**0.5
    energy -= self.g * (phonon_occ[bond_pos] + 1)**0.5 * ratio_2

    new_phonon_occ = phonon_occ.at[bond_pos].add(-1)
    ratio_3 = (phonon_occ[bond_pos])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap
    energy -= self.g * (phonon_occ[bond_pos])**0.5 * ratio_3

    # electron hop <-
    new_elec_pos = (elec_pos - 1) % n_sites
    bond_pos = new_elec_pos * (1 - (n_sites==2))

    new_phonon_occ = phonon_occ.at[bond_pos].add(1)
    ratio_4 = lax.cond(n_sites > 2, lambda x: wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos] + 1)**0.5, lambda x: 0., 0.)
    energy -= self.g * (phonon_occ[bond_pos] + 1)**0.5 * ratio_4

    new_phonon_occ = phonon_occ.at[bond_pos].add(-1)
    ratio_5 = lax.cond(n_sites > 2, lambda x: (phonon_occ[bond_pos])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap, lambda x: 0., 0.)
    energy -= self.g * (phonon_occ[bond_pos])**0.5 * ratio_5

    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3, ratio_4, ratio_5 ])))
    #jax.debug.print('ratios: {}', jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3 ]))
    #jax.debug.print('energy: {}', energy)
    weight = 1 / cumulative_ratios[-1]
    #jax.debug.print('weight: {}', weight)
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)

    # electron hop
    walker = lax.cond(new_ind < 2, lambda x: x.at[0].set((elec_pos + 1 - 2*new_ind) % n_sites), lambda x: x, walker)

    # eph
    walker = lax.cond((new_ind - 1) * (new_ind - 4) < 0, lambda x: x.at[1 + elec_pos * (1 - (n_sites==2))].add(5 - 2*new_ind), lambda x: x, walker)
    walker = lax.cond((new_ind - 1) * (new_ind - 4) < 0, lambda x: x.at[0].set((elec_pos + 1) % n_sites), lambda x: x, walker)
    walker = lax.cond((new_ind - 3) * (new_ind - 6) < 0, lambda x: x.at[1 + (elec_pos - 1) % n_sites].add(9 - 2*new_ind), lambda x: x, walker)
    walker = lax.cond((new_ind - 3) * (new_ind - 6) < 0, lambda x: x.at[0].set((elec_pos - 1) % n_sites), lambda x: x, walker)
    walker = jnp.where(walker < 0, 0, walker)
    #jax.debug.print('new_walker: {}', walker)

    return energy, qp_weight, overlap_gradient, weight, walker

  def __hash__(self):
    return hash((self.omega, self.g))


@dataclass
class long_range_1d():
  omega: float
  g: float
  zeta: float

  @partial(jit, static_argnums = (0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1:]
    n_sites = phonon_occ.shape[0]

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    # 2 e hop terms, n_sites phonon terms
    ratios = jnp.zeros((2 + 2 * n_sites,))

    # electron hops
    new_elec_pos = (elec_pos + 1) % n_sites
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0 = lax.cond(n_sites > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0
    ratios = ratios.at[0].set(ratio_0)

    new_elec_pos = (elec_pos - 1) % n_sites
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1 = lax.cond(n_sites > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1
    ratios = ratios.at[1].set(ratio_1)

    # e_ph coupling
    def scanned_fun(carry, x):
      pos = x
      dist = jnp.min(jnp.array([(x - elec_pos) % n_sites, (elec_pos - x) % n_sites]))

      new_phonon_occ = phonon_occ.at[pos].add(1)
      ratio = wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[pos] + 1)**0.5
      carry[0] -= self.g * jnp.exp(-dist / self.zeta) * (phonon_occ[pos] + 1)**0.5 * ratio / (1 + dist**2)**1.5
      carry[1] = carry[1].at[2*x + 2].set(ratio)

      new_phonon_occ = phonon_occ.at[pos].add(-1)
      ratio = (phonon_occ[pos])**0.5 * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap
      carry[0] -= self.g * jnp.exp(-dist / self.zeta) * (phonon_occ[pos])**0.5 * ratio / (1 + dist**2)**1.5
      carry[1] = carry[1].at[2*x + 3].set(ratio)

      return carry, x

    [ energy, ratios ], _ = lax.scan(scanned_fun, [ energy, ratios ], jnp.arange(n_sites))

    cumulative_ratios = jnp.cumsum(jnp.abs(ratios))

    #jax.debug.print('ratios: {}', jnp.array(ratios))
    #jax.debug.print('energy: {}', energy)
    weight = 1 / cumulative_ratios[-1]
    #jax.debug.print('weight: {}', weight)
    #jax.debug.print('random_number: {}', random_number)
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    #jax.debug.print('new_ind: {}', new_ind)

    walker = lax.cond(new_ind < 2, lambda x: x.at[0].set((elec_pos + 1 - 2 * new_ind) % n_sites), lambda x: x.at[1 + (new_ind - 2) // 2].add(1 - 2*((new_ind - 2) % 2)), walker)
    walker = jnp.where(walker < 0, 0, walker)

    return energy, qp_weight, overlap_gradient, weight, walker

  def __hash__(self):
    return hash((self.omega, self.g, self.zeta))

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
    #jax.debug.print('ovlp: {}', overlap)

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


@dataclass
class long_range_2d():
  omega: float
  g: float
  zeta: float

  @partial(jit, static_argnums = (0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1]
    (l_y, l_x) = phonon_occ.shape
    n_sites = l_y * l_x
    lattice_sites = jnp.array(lattice.sites)

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    # 4 e hop terms, n_sites phonon terms
    ratios = jnp.zeros((4 + 2 * n_sites,))

    # electron hops
    new_elec_pos = ( elec_pos[0], (elec_pos[1] + 1) % l_x )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0_x = lax.cond(l_x > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0_x
    ratios = ratios.at[0].set(ratio_0_x)

    new_elec_pos = ( elec_pos[0], (elec_pos[1] - 1) % l_x )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1_x = lax.cond(l_x > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1_x
    ratios = ratios.at[1].set(ratio_1_x)

    new_elec_pos = ( (elec_pos[0] + 1) % l_y, elec_pos[1] )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0_y = lax.cond(l_y > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0_y
    ratios = ratios.at[2].set(ratio_0_y)

    new_elec_pos = ( (elec_pos[0] - 1) % l_y, elec_pos[1] )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1_y = lax.cond(l_y > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1_y
    ratios = ratios.at[3].set(ratio_1_y)

    # e_ph coupling
    def scanned_fun(carry, x):
      pos = x
      counter = carry[2]
      dist = lattice.get_distance(elec_pos, x)

      new_phonon_occ = phonon_occ.at[pos[0], pos[1]].add(1)
      ratio = wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[pos[0], pos[1]] + 1)**0.5
      carry[0] -= self.g * jnp.exp(-dist / self.zeta) * (phonon_occ[pos[0], pos[1]] + 1)**0.5 * ratio / (1 + dist**2)**1.5
      carry[1] = carry[1].at[2*carry[2] + 4].set(ratio)

      new_phonon_occ = phonon_occ.at[pos[0], pos[1]].add(-1)
      ratio = (phonon_occ[pos[0], pos[1]])**0.5 * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap
      carry[0] -= self.g * jnp.exp(-dist / self.zeta) * (phonon_occ[pos[0], pos[1]])**0.5 * ratio / (1 + dist**2)**1.5
      carry[1] = carry[1].at[2*carry[2] + 5].set(ratio)

      carry[2] = carry[2] + 1

      return carry, x

    counter = 0
    [ energy, ratios, _ ], _ = lax.scan(scanned_fun, [ energy, ratios, counter ], lattice_sites)

    cumulative_ratios = jnp.cumsum(jnp.abs(ratios))

    #jax.debug.print('ratios: {}', jnp.array(ratios))
    #jax.debug.print('energy: {}', energy)
    weight = 1 / cumulative_ratios[-1]
    #jax.debug.print('weight: {}', weight)
    #jax.debug.print('random_number: {}', random_number)
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
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
    walker[1] = lax.cond(new_ind >= 4, lambda w: w.at[(*(lattice_sites[(new_ind - 4)//2]),)].add(1 - 2*((new_ind-4) % 2)), lambda w: w, walker[1])
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

    return energy, qp_weight, overlap_gradient, weight, walker

  def __hash__(self):
    return hash((self.omega, self.g, self.zeta))


