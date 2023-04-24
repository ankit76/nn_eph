from functools import partial
from dataclasses import dataclass
from typing import Sequence, Tuple, Callable, Any
from flax import linen as nn
from jax import random, lax, tree_util, vmap, grad, value_and_grad, jit, numpy as jnp
import jax
import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
#os.environ['JAX_ENABLE_X64'] = 'True'
#os.environ['JAX_DISABLE_JIT'] = 'True'

@dataclass
class heisenberg():
  j: float = 1.

  @partial(jit, static_argnums=(0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    n_bonds = len(lattice.bonds)

    overlap = wave.calc_overlap(walker, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(walker, parameters, lattice)
    qp_weight = 0.

    def z_ene(bond):
      neighbors = lattice.get_neighboring_sites(bond)
      return self.j * walker[neighbors[0]] * walker[neighbors[1]]

    # diagonal
    energy = jnp.sum(vmap(z_ene)(jnp.array(lattice.bonds)))

    # spin flips 
    def flip(bond):
      neighbors = lattice.get_neighboring_sites(bond)
      i1 = neighbors[0]
      i2 = neighbors[1]

      # s1+ s2-
      new_walker = walker.at[i1].set(0.5)
      new_walker = new_walker.at[i2].set(-0.5)
      new_overlap = (walker[i1] == -0.5) * (walker[i2] == 0.5) * wave.calc_overlap(new_walker, parameters, lattice)
      ratio_1 = new_overlap / overlap
      
      # s1- s2+
      new_walker = walker.at[i1].set(-0.5)
      new_walker = new_walker.at[i2].set(0.5)
      new_overlap = (walker[i1] == 0.5) * (walker[i2] == -0.5) * wave.calc_overlap(new_walker, parameters, lattice)
      ratio_2 = new_overlap / overlap
      
      return ratio_1, ratio_2

    ratios = vmap(flip)(jnp.array(lattice.bonds))      
    energy += self.j * jnp.sum(jnp.concatenate(ratios)) / 2

    # update walker
    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.concatenate(ratios)))
    weight = 1 / cumulative_ratios[-1]
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)
    ##jax.debug.print('overlap_gradient: {}', overlap_gradient)
    #jax.debug.print('weight: {}', weight)
    #jax.debug.print('ratios: {}', ratios)
    #jax.debug.print('energy: {}', energy)
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}\n', new_ind)
    
    bond = new_ind % n_bonds
    neighbors = lattice.get_neighboring_sites(jnp.array(lattice.bonds)[bond])
    i1 = neighbors[0]
    i2 = neighbors[1]
    walker = walker.at[i1].set(-walker[i1])
    walker = walker.at[i2].set(-walker[i2])

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.j,))


@dataclass
class heisenberg_bond():
  omega: float
  g: float
  j: float = 1.
  max_n_phonons: Any = jnp.inf

  @partial(jit, static_argnums=(0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    spins = walker[0]
    phonons = walker[1]
    n_bonds = len(lattice.bonds)

    overlap = wave.calc_overlap(walker, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(walker, parameters, lattice)
    qp_weight = 0.

    # diagonal
    energy = self.omega * jnp.sum(phonons)

    # spin flips 
    # carry = energy
    def scanned_fun(carry, bond):
      neighbors = lattice.get_neighboring_sites(bond)
      i1 = neighbors[0]
      i2 = neighbors[1]
      new_phonons_c = phonons.at[bond].set(phonons[bond][0] + 1)
      new_phonons_a = phonons.at[bond].set(phonons[bond][0] - 1)
      new_phonons_a = jnp.where(new_phonons_a < 0, 0, new_phonons_a)
      
      # s1z s2z
      # bare 
      carry += self.j * spins[i1] * spins[i2]

      # create phonon
      new_overlap = (phonons[bond][0] < self.max_n_phonons) * wave.calc_overlap([spins, new_phonons_c], parameters, lattice)
      ratio_0 = new_overlap / overlap / (phonons[bond][0] + 1)**0.5
      carry += self.j * self.g * (phonons[bond][0] + 1)**0.5 * ratio_0 * spins[i1] * spins[i2]

      # annihilate phonon
      new_overlap = (phonons[bond][0] > 0) * wave.calc_overlap([spins, new_phonons_a], parameters, lattice)
      ratio_1 = (phonons[bond][0])**0.5 * new_overlap / overlap
      carry += self.j * self.g * (phonons[bond][0])**0.5 * ratio_1 * spins[i1] * spins[i2]
      
      # s1+ s2-
      # bare
      new_spins = spins.at[i1].set(0.5)
      new_spins = new_spins.at[i2].set(-0.5)
      new_overlap = wave.calc_overlap([new_spins, phonons], parameters, lattice)
      ratio_2 = (spins[i1] == -0.5) * (spins[i2] == 0.5) * new_overlap / overlap
      carry += self.j * ratio_2 / 2
    
      # create phonon
      new_overlap = (phonons[bond][0] < self.max_n_phonons) * wave.calc_overlap([new_spins, new_phonons_c], parameters, lattice)
      ratio_3 = (spins[i1] == -0.5) * (spins[i2] == 0.5) * new_overlap / overlap / (phonons[bond][0] + 1)**0.5
      carry += self.j * self.g * (phonons[bond][0] + 1)**0.5 * ratio_3 / 2

      # annihilate phonon
      new_overlap = (phonons[bond][0] > 0) * wave.calc_overlap([new_spins, new_phonons_a], parameters, lattice)
      ratio_4 = (spins[i1] == -0.5) * (spins[i2] == 0.5) * (phonons[bond][0])**0.5 * new_overlap / overlap
      carry += self.j * self.g * (phonons[bond][0])**0.5 * ratio_4 / 2

      # s1- s2+
      # bare
      new_spins = spins.at[i1].set(-0.5)
      new_spins = new_spins.at[i2].set(0.5)
      new_overlap = wave.calc_overlap([new_spins, phonons], parameters, lattice)
      ratio_5 = (spins[i1] == 0.5) * (spins[i2] == -0.5) * new_overlap / overlap
      carry += self.j * ratio_5 / 2

      # create phonon
      new_overlap = (phonons[bond][0] < self.max_n_phonons) * wave.calc_overlap([new_spins, new_phonons_c], parameters, lattice)
      ratio_6 = (spins[i1] == 0.5) * (spins[i2] == -0.5) * new_overlap / overlap / (phonons[bond][0] + 1)**0.5
      carry += self.j * self.g * (phonons[bond][0] + 1)**0.5 * ratio_6 / 2

      # annihilate phonon
      new_overlap = (phonons[bond][0] > 0) * wave.calc_overlap([new_spins, new_phonons_a], parameters, lattice)
      ratio_7 = (spins[i1] == 0.5) * (spins[i2] == -0.5) * (phonons[bond][0])**0.5 * new_overlap / overlap
      carry += self.j * self.g * (phonons[bond][0])**0.5 * ratio_7 / 2

      return carry, (ratio_0, ratio_1, ratio_2, ratio_3, ratio_4, ratio_5, ratio_6, ratio_7)

    energy, ratios = lax.scan(scanned_fun, energy, jnp.array(lattice.bonds))

    # update walker
    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.concatenate(ratios)))
    weight = 1 / cumulative_ratios[-1]
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])
    
    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)
    ##jax.debug.print('overlap_gradient: {}', overlap_gradient)
    #jax.debug.print('weight: {}', weight)
    #jax.debug.print('ratios: {}', ratios)
    #jax.debug.print('energy: {}', energy)
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}\n', new_ind)
    
    bond = jnp.array(lattice.bonds)[new_ind % n_bonds]
    neighbors = lattice.get_neighboring_sites(bond)
    i1 = neighbors[0]
    i2 = neighbors[1]
    excitation_type = new_ind // n_bonds
    spin_change = jnp.array([0, 0, 1, 1, 1, -1, -1, -1])[excitation_type]
    phonon_change = jnp.array([1, -1, 0, 1, -1, 0, 1, -1])[excitation_type]
    walker[0] = walker[0].at[i1].set(walker[0][i1] + spin_change)
    walker[0] = walker[0].at[i2].set(walker[0][i2] - spin_change)
    walker[1] = walker[1].at[bond].set(walker[1][bond] + phonon_change)

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega, self.g, self.j, self.max_n_phonons))
