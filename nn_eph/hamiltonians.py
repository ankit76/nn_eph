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
    phonon_occ = walker[1]
    n_sites = lattice.shape[0]

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('lf_overlap: {}', lf_overlap)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    # electron hops
    new_elec_pos = ((elec_pos[0] + 1) % n_sites,)
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0 = lax.cond(n_sites > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0

    new_elec_pos = ((elec_pos[0] - 1) % n_sites,)
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
    
    elec_pos = jnp.array(elec_pos)
    walker[0] = (lax.cond(new_ind < 2, lambda x: (elec_pos[0] + 1 - 2*new_ind) % n_sites, lambda x: elec_pos[0], 0.),)
    walker[1] = lax.cond(new_ind > 1, lambda x: x.at[elec_pos[0]].add(-2*new_ind + 5), lambda x: x, walker[1])
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

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
    phonon_occ = walker[1]
    n_sites = lattice.n_sites

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    # electron hop ->
    # bare
    new_elec_pos = ((elec_pos[0] + 1) % n_sites,)
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0 = lax.cond(n_sites > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0
        
    bond_pos = (elec_pos[0] * (1 - (n_sites == 2)),)
    # create phonon
    new_phonon_occ = phonon_occ.at[bond_pos].add(1)
    ratio_0_cp = wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos] + 1)**0.5
    energy -= self.g * (phonon_occ[bond_pos] + 1)**0.5 * ratio_0_cp
    
    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos].add(-1)
    ratio_0_dp = (phonon_occ[bond_pos])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap
    energy -= self.g * (phonon_occ[bond_pos])**0.5 * ratio_0_dp

    # electron hop <-
    # bare
    new_elec_pos = ((elec_pos[0] - 1) % n_sites,)
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1 = lax.cond(n_sites > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1

    bond_pos = (new_elec_pos[0] * (1 - (n_sites==2)),)
    # create phonon
    new_phonon_occ = phonon_occ.at[bond_pos].add(1)
    ratio_1_cp = lax.cond(n_sites > 2, lambda x: wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos] + 1)**0.5, lambda x: 0., 0.)
    energy -= self.g * (phonon_occ[bond_pos] + 1)**0.5 * ratio_1_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos].add(-1)
    ratio_1_dp = lax.cond(n_sites > 2, lambda x: (phonon_occ[bond_pos])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap, lambda x: 0., 0.)
    energy -= self.g * (phonon_occ[bond_pos])**0.5 * ratio_1_dp

    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array([ ratio_0, ratio_0_cp, ratio_0_dp, ratio_1, ratio_1_cp, ratio_1_dp ])))
    #jax.debug.print('ratios: {}', jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3 ]))
    #jax.debug.print('energy: {}', energy)
    weight = 1 / cumulative_ratios[-1]
    #jax.debug.print('weight: {}', weight)
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)

    # electron hop
    new_elec_pos = jnp.array([(elec_pos[0] + 1) % n_sites, (elec_pos[0] - 1) % n_sites])[new_ind // 3]
    walker[0] = (new_elec_pos,)

    # eph
    ind_0 = jnp.array([elec_pos[0], new_elec_pos])[new_ind // 3] * (n_sites == 2)
    ind_1 = jnp.array([0, 1, -1])[new_ind % 3]
    walker[1] = walker[1].at[ind_0].add(ind_1)
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])
    #jax.debug.print('new_walker: {}', walker)

    return energy, qp_weight, overlap_gradient, weight, walker

  def __hash__(self):
    return hash((self.omega, self.g))

@dataclass
class bm_ssh_1d():
  omega: float
  g: float

  @partial(jit, static_argnums = (0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1]
    n_sites = lattice.shape[0]

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap) 

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    ratios = [ ]
    for site, new_elec_pos in enumerate([ ((elec_pos[0] + 1) % n_sites,), ((elec_pos[0] - 1) % n_sites,) ]):
      # bare
      ratio_0 = (1 - (n_sites == 2) * (site == 1)) * wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice) / overlap
      ratios.append(ratio_0)
      energy -= ratio_0

      hop_sgn = 1 - 2 * site + (2 * site - 1 - (elec_pos[0] == 1) + (elec_pos[0] == 0)) * (n_sites == 2)
      # create phonon on old site
      new_phonon_occ = phonon_occ.at[elec_pos].add(1)
      ratio_1_co = (1 - (n_sites == 2) * (site == 1)) * wave.calc_overlap(new_elec_pos, new_phonon_occ, 
      parameters, lattice) / overlap / (phonon_occ[elec_pos] + 1)**0.5
      ratios.append(ratio_1_co)
      energy -= hop_sgn * self.g * (phonon_occ[elec_pos] + 1)**0.5 * ratio_1_co

      # destroy phonon on old site
      new_phonon_occ = phonon_occ.at[elec_pos].add(-1)
      ratio_1_do = (1 - (n_sites == 2) * (site == 1)) * (phonon_occ[elec_pos])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap
      ratios.append(ratio_1_do)
      energy -= hop_sgn * self.g * (phonon_occ[elec_pos])**0.5 * ratio_1_do

      # create phonon on new site
      new_phonon_occ = phonon_occ.at[new_elec_pos].add(1)
      ratio_1_cn = (1 - (n_sites == 2) * (site == 1)) * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[new_elec_pos] + 1)**0.5
      ratios.append(ratio_1_cn)
      energy += hop_sgn * self.g * (phonon_occ[new_elec_pos] + 1)**0.5 * ratio_1_cn

      # destroy phonon on new site
      new_phonon_occ = phonon_occ.at[new_elec_pos].add(-1)
      ratio_1_dn = (1 - (n_sites == 2) * (site == 1)) * (phonon_occ[new_elec_pos])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap
      ratios.append(ratio_1_dn)
      energy += hop_sgn * self.g * (phonon_occ[new_elec_pos])**0.5 * ratio_1_dn

    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array(ratios)))
    #jax.debug.print('ratios: {}', jnp.array(ratios))
    #jax.debug.print('energy: {}', energy)
    weight = 1 / cumulative_ratios[-1]
    #jax.debug.print('weight: {}', weight)
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)

    # electron hop
    new_elec_pos = jnp.array([(elec_pos[0] + 1) % n_sites, (elec_pos[0] - 1) % n_sites])[new_ind // 5]
    walker[0] = (new_elec_pos,)

    # eph
    ind_0 = jnp.array([elec_pos[0], elec_pos[0], elec_pos[0], new_elec_pos, new_elec_pos])[new_ind % 5]
    ind_1 = jnp.array([0, 1, -1, 1, -1])[new_ind % 5]
    walker[1] = walker[1].at[ind_0].add(ind_1)
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
    phonon_occ = walker[1]
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
    new_elec_pos = ((elec_pos[0] + 1) % n_sites,)
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0 = lax.cond(n_sites > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0
    ratios = ratios.at[0].set(ratio_0)

    new_elec_pos =((elec_pos[0] - 1) % n_sites,)
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1 = lax.cond(n_sites > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1
    ratios = ratios.at[1].set(ratio_1)

    # e_ph coupling
    def scanned_fun(carry, x):
      pos = x
      dist = jnp.min(jnp.array([(x - elec_pos[0]) % n_sites, (elec_pos[0] - x) % n_sites]))

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
    
    elec_pos = jnp.array(elec_pos)
    walker[0] = (lax.cond(new_ind < 2, lambda x: x + 1 - 2 * new_ind, lambda x: x, elec_pos[0]) % n_sites,)
    walker[1] = lax.cond(new_ind > 1, lambda x: x.at[(new_ind - 2) // 2].add(1 - 2*((new_ind - 2) % 2)), lambda x: x, walker[1])
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

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
class ssh_2d():
  omega: float
  g: float

  @partial(jit, static_argnums = (0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1]
    l_y, l_x = lattice.shape

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker:\n{}\n', walker)
    #jax.debug.print('overlap: {}', overlap)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    # electron hops
    # right
    # bare
    new_elec_pos = ( elec_pos[0], (elec_pos[1] + 1) % l_x )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0_x = lax.cond(l_x > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0_x

    bond_pos_0_x = (1, elec_pos[0], elec_pos[1] * (1 - (l_x==2)))
    # create phonon
    new_phonon_occ = phonon_occ.at[bond_pos_0_x].add(1)
    ratio_0_x_cp = wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos_0_x] + 1)**0.5
    energy -= self.g * (phonon_occ[bond_pos_0_x] + 1)**0.5 * ratio_0_x_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos_0_x].add(-1)
    ratio_0_x_dp = (phonon_occ[bond_pos_0_x])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap
    energy -= self.g * (phonon_occ[bond_pos_0_x])**0.5 * ratio_0_x_dp

    # left
    # bare
    new_elec_pos = ( elec_pos[0], (elec_pos[1] - 1) % l_x )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1_x = lax.cond(l_x > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1_x

    bond_pos_1_x = (1, elec_pos[0], new_elec_pos[1] * (1 - (l_x==2)))
    # create phonon
    new_phonon_occ = phonon_occ.at[bond_pos_1_x].add(1)
    ratio_1_x_cp = lax.cond(l_x > 2, lambda x: wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos_1_x] + 1)**0.5, lambda x: 0., 0.)
    energy -= self.g * (phonon_occ[bond_pos_1_x] + 1)**0.5 * ratio_1_x_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos_1_x].add(-1)
    ratio_1_x_dp = lax.cond(l_x > 2, lambda x: (phonon_occ[bond_pos_1_x])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap, lambda x: 0., 0.)
    energy -= self.g * (phonon_occ[bond_pos_1_x])**0.5 * ratio_1_x_dp

    # up
    # bare
    new_elec_pos = ( (elec_pos[0] - 1) % l_y, elec_pos[1] )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0_y = lax.cond(l_y > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0_y

    bond_pos_0_y = (0, new_elec_pos[0] * (1 - (l_y==2)), elec_pos[1])
    # create phonon
    new_phonon_occ = phonon_occ.at[bond_pos_0_y].add(1)
    ratio_0_y_cp = wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos_0_y] + 1)**0.5
    energy -= self.g * (phonon_occ[bond_pos_0_y] + 1)**0.5 * ratio_0_y_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos_0_y].add(-1)
    ratio_0_y_dp = (phonon_occ[bond_pos_0_y])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap
    energy -= self.g * (phonon_occ[bond_pos_0_y])**0.5 * ratio_0_y_dp

    # down
    # bare
    new_elec_pos = ( (elec_pos[0] + 1) % l_y, elec_pos[1] )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1_y = lax.cond(l_y > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1_y

    bond_pos_1_y = (0, elec_pos[0] * (1 - (l_y==2)), elec_pos[1])
    # create phonon
    new_phonon_occ = phonon_occ.at[bond_pos_1_y].add(1)
    ratio_1_y_cp = lax.cond(l_y > 2, lambda x: wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos_1_y] + 1)**0.5, lambda x: 0., 0.)
    energy -= self.g * (phonon_occ[bond_pos_1_y] + 1)**0.5 * ratio_1_y_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos_1_y].add(-1)
    ratio_1_y_dp = lax.cond(l_y > 2, lambda x: (phonon_occ[bond_pos_1_y])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap, lambda x: 0., 0.)
    energy -= self.g * (phonon_occ[bond_pos_1_y])**0.5 * ratio_1_y_dp


    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array([ ratio_0_x, ratio_0_x_cp, ratio_0_x_dp, ratio_1_x, ratio_1_x_cp, ratio_1_x_dp, ratio_0_y, ratio_0_y_cp, ratio_0_y_dp, ratio_1_y, ratio_1_y_cp, ratio_1_y_dp ])))
    bond_pos = jnp.array([ bond_pos_0_x, bond_pos_1_x, bond_pos_0_y, bond_pos_1_y ])
    #jax.debug.print('bond_pos:\n{}\n', bond_pos)
    #jax.debug.print('ratios: {}', jnp.array([ ratio_0_x, ratio_0_x_cp, ratio_0_x_dp, ratio_1_x, ratio_1_x_cp, ratio_1_x_dp, ratio_0_y, ratio_0_y_cp, ratio_0_y_dp, ratio_1_y, ratio_1_y_cp, ratio_1_y_dp ]))
    #jax.debug.print('energy: {}', energy)
    weight = 1 / cumulative_ratios[-1]
    #jax.debug.print('weight: {}', weight)
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)

    # electron hop
    walker[0] = jnp.array(walker[0])
    elec_pos = jnp.array(elec_pos)
    ind_0 = 1 - new_ind // 6
    ind_1 = jnp.array([1, -1, -1, 1])[new_ind // 3]
    l_x_l_y = (new_ind // 3 < 2) * l_x + (new_ind // 3 > 1) * l_y
    walker[0] = walker[0].at[ind_0].set((elec_pos[ind_0] + ind_1) % l_x_l_y)
    walker[0] = (walker[0][0], walker[0][1])

    # eph
    ind_0 = bond_pos[new_ind // 3]
    ind_1 = jnp.array([0, 1, -1])[new_ind % 3]
    walker[1] = walker[1].at[(*ind_0,)].add(ind_1)
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])
    #jax.debug.print('new_walker:\n{}\n', walker)

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


@dataclass
class holstein_3d():
  omega: float
  g: float

  @partial(jit, static_argnums=(0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1]
    (l_x, l_y, l_z) = phonon_occ.shape

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('ovlp: {}', overlap)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    # electron hops
    new_elec_pos = (elec_pos[0], elec_pos[1], (elec_pos[2] + 1) % l_x )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0_x = lax.cond(l_x > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0_x

    new_elec_pos = (elec_pos[0], elec_pos[1], (elec_pos[2] - 1) % l_x )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1_x = lax.cond(l_x > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1_x

    new_elec_pos = ( elec_pos[0], (elec_pos[1] + 1) % l_y, elec_pos[2] )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0_y = lax.cond(l_y > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0_y

    new_elec_pos = ( elec_pos[0], (elec_pos[1] - 1) % l_y, elec_pos[2] )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1_y = lax.cond(l_y > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1_y

    new_elec_pos = ( (elec_pos[0] + 1) % l_z, elec_pos[1], elec_pos[2] )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0_z = lax.cond(l_z > 1, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_0_z

    new_elec_pos = ( (elec_pos[0] - 1) % l_z, elec_pos[1], elec_pos[2] )
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1_z = lax.cond(l_z > 2, lambda x: x / overlap, lambda x: 0., new_overlap)
    energy -= ratio_1_z

    # e_ph coupling
    new_phonon_occ = phonon_occ.at[elec_pos[0], elec_pos[1], elec_pos[2]].add(1)
    ratio_2 = wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos[0], elec_pos[1],elec_pos[2]] + 1)**0.5
    energy -= self.g * (phonon_occ[elec_pos[0], elec_pos[1], elec_pos[2]] + 1)**0.5 * ratio_2

    new_phonon_occ = phonon_occ.at[elec_pos[0], elec_pos[1], elec_pos[2]].add(-1)
    ratio_3 = (phonon_occ[elec_pos[0], elec_pos[1], elec_pos[2]])**0.5 * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap
    energy -= self.g * (phonon_occ[elec_pos[0], elec_pos[1], elec_pos[2]])**0.5 * ratio_3

    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array([ ratio_0_x, ratio_1_x, ratio_0_y, ratio_1_y, ratio_0_z, ratio_1_z, ratio_2, ratio_3 ])))
    #jax.debug.print('ratios: {}', jnp.array([ ratio_0_x, ratio_1_x, ratio_0_y, ratio_1_y, ratio_2, ratio_3  ]))
    #jax.debug.print('energy: {}', energy)
    weight = 1 / cumulative_ratios[-1]
    #jax.debug.print('weight: {}', weight)
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)

    walker[0] = jnp.array(walker[0])
    # x ->
    walker[0] = lax.cond(new_ind == 0, lambda w: w.at[2].set((elec_pos[2] + 1) % l_x), lambda w: w, walker[0])
    # x <-
    walker[0] = lax.cond(new_ind == 1, lambda w: w.at[2].set((elec_pos[2] - 1) % l_x), lambda w: w, walker[0])
    # y ->
    walker[0] = lax.cond(new_ind == 2, lambda w: w.at[1].set((elec_pos[1] + 1) % l_y), lambda w: w, walker[0])
     # y <-
    walker[0] = lax.cond(new_ind == 3, lambda w: w.at[1].set((elec_pos[1] - 1) % l_y), lambda w: w, walker[0])
    # z ->
    walker[0] = lax.cond(new_ind == 4, lambda w: w.at[0].set((elec_pos[0] + 1) % l_y), lambda w: w, walker[0])
    # z <-
    walker[0] = lax.cond(new_ind == 5, lambda w: w.at[0].set((elec_pos[0] - 1) % l_y), lambda w: w, walker[0])
    walker[0] = (walker[0][0], walker[0][1], walker[0][2])

    # phonon
    walker[1] = lax.cond(new_ind == 6, lambda w: w.at[elec_pos[0], elec_pos[1], elec_pos[2]].add(1), lambda w: w, walker[1])
    walker[1] = lax.cond(new_ind == 7, lambda w: w.at[elec_pos[0], elec_pos[1], elec_pos[2]].add(-1), lambda w: w, walker[1])
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

    return energy, qp_weight, overlap_gradient, weight, walker

  def __hash__(self):
    return hash((self.omega, self.g))
