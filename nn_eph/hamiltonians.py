import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
#os.environ['JAX_ENABLE_X64'] = 'True'
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
  dtype: Any = jnp.float32   # this should be handled in the wave function
  zero_dtype: Any = jnp.array([0.], dtype=jnp.float32)[0]
  max_n_phonons: Any = jnp.inf

  def __post_init__(self):
    self.zero_dtype = jnp.array([0.], self.dtype)[0]

  @partial(jit, static_argnums = (0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1]
    n_sites = lattice.shape[0]

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    # electron hops
    new_elec_pos = ((elec_pos[0] + 1) % n_sites,)
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0 = lax.cond(n_sites > 1, lambda x: x / overlap, lambda x: self.zero_dtype, new_overlap)
    energy -= ratio_0

    new_elec_pos = ((elec_pos[0] - 1) % n_sites,)
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1 = lax.cond(n_sites > 2, lambda x: x / overlap, lambda x: self.zero_dtype, new_overlap)
    energy -= ratio_1

    # e_ph coupling
    new_phonon_occ = phonon_occ.at[elec_pos].add(1)
    ratio_2 = (phonon_occ[elec_pos] < self.max_n_phonons) * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos] + 1)**0.5
    #ratio_2 = lax.cond(phonon_occ[elec_pos] < self.max_n_phonons, lambda x: wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos] + 1)**0.5, lambda x: self.zero_dtype, 0.)
    energy -= self.g * (phonon_occ[elec_pos] + 1)**0.5 * ratio_2

    new_phonon_occ = phonon_occ.at[elec_pos].add(-1)
    ratio_3 = lax.cond(phonon_occ[elec_pos] > 0, lambda x: (phonon_occ[elec_pos])**0.5 * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap, lambda x: self.zero_dtype, 0.)
    energy -= self.g * (phonon_occ[elec_pos])**0.5 * ratio_3

    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3 ])))
    weight = 1 / cumulative_ratios[-1]
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])

    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)
    ##jax.debug.print('overlap_gradient: {}', overlap_gradient)
    #jax.debug.print('weight: {}', weight)
    #jax.debug.print('ratios: {}', jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3 ]))
    #jax.debug.print('energy: {}', energy)
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}\n', new_ind)

    elec_pos = jnp.array(elec_pos)
    walker[0] = (lax.cond(new_ind < 2, lambda x: (elec_pos[0] + 1 - 2*new_ind) % n_sites, lambda x: elec_pos[0], 0.),)
    walker[1] = lax.cond(new_ind > 1, lambda x: x.at[elec_pos[0]].add(-2*new_ind + 5), lambda x: x, walker[1])
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega, self.g))

@dataclass
class kq():
  omega_q: Sequence
  e_k: Sequence
  g_kq: Any
  max_n_phonons: Any = jnp.inf

  @partial(jit, static_argnums=(0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_k = walker[0]
    phonon_occ = walker[1]
    n_sites = jnp.array(lattice.shape)

    overlap = wave.calc_overlap(elec_k, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_k, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)

    # diagonal
    energy = jnp.sum(jnp.array(self.omega_q) * phonon_occ) + jnp.array(self.e_k)[elec_k] + 0.j

    ratios = jnp.zeros((2 * len(lattice.sites),)) + 0.j

    # e_ph coupling
    # carry = (energy, ratios)
    def scanned_fun(carry, kp):
      qc = tuple((jnp.array(elec_k) - kp) % n_sites)
      qd = tuple((kp - jnp.array(elec_k)) % n_sites)

      kp_i = lattice.get_site_num(kp)
      qc_i = lattice.get_site_num(qc)
      qd_i = lattice.get_site_num(qd)

      new_elec_k = tuple(kp)

      new_phonon_occ = phonon_occ.at[qc].add(1)
      ratio = (jnp.sum(phonon_occ) < self.max_n_phonons) * wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[qc] + 1)**0.5
      carry[0] -= jnp.array(self.g_kq)[kp_i, qc_i] * (phonon_occ[qc] + 1)**0.5 * ratio
      carry[1] = carry[1].at[2*kp_i].set(ratio)

      new_phonon_occ = phonon_occ.at[qd].add(-1)
      new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
      ratio = (phonon_occ[qd])**0.5 * wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice) / overlap
      carry[0] -= jnp.array(self.g_kq)[kp_i, qc_i] * (phonon_occ[qd])**0.5 * ratio
      carry[1] = carry[1].at[2*kp_i + 1].set(ratio)

      return carry, (qc, qd)

    [energy, ratios], (qc, qd) = lax.scan(scanned_fun, [energy, ratios], jnp.array(lattice.sites))

    qc = jnp.stack(qc, axis=-1)
    qd = jnp.stack(qd, axis=-1)

    cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
    weight = 1 / cumulative_ratios[-1]
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])

    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)
    ##jax.debug.print('overlap_gradient: {}', overlap_gradient)
    #jax.debug.print('weight: {}', weight)
    #jax.debug.print('ratios: {}', ratios)
    #jax.debug.print('energy: {}', energy)
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)
    #jax.debug.print('qc: {}', qc)

    #walker[0] = (new_ind//2,)
    #walker[1] = lax.cond(new_ind % 2 == 0, lambda x: x.at[qc[new_ind//2]].add(1), lambda x: x.at[qd[new_ind//2]].add(-1), walker[1])
    walker[0] = tuple(jnp.array(lattice.sites)[new_ind//2])
    walker[1] = lax.cond(new_ind % 2 == 0, lambda x: x.at[tuple(qc[new_ind//2])].add(1), lambda x: x.at[tuple(qd[new_ind//2])].add(-1), walker[1])
    #walker[1] = lax.cond(new_ind % 2 == 0, lambda x: x.at[qc[0][new_ind//2]].add(1), lambda x: x.at[qd[0][new_ind//2]].add(-1), walker[1])
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega_q, self.e_k, self.g_kq, self.max_n_phonons))

@dataclass
class kq_ne():
  omega_q: Sequence
  e_n_k: Sequence
  g_mn_kq: Any
  max_n_phonons: Any = jnp.inf

  @partial(jit, static_argnums=(0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_n_k = walker[0]
    elec_n = walker[0][0]
    elec_k = walker[0][1]
    phonon_occ = walker[1]
    n_sites = jnp.array(lattice.shape)
    n_bands = len(self.e_n_k)

    overlap = wave.calc_overlap(elec_n_k, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(
        elec_n_k, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0,
                         lambda x: 1., lambda x: 0., 0.)

    # diagonal
    energy = jnp.sum(jnp.array(self.omega_q) * phonon_occ) + jnp.array(self.e_n_k)[elec_n][elec_k] + 0.j

    ratios = jnp.zeros((n_bands, 2 * len(lattice.sites),)) + 0.j
    qc = None
    qd = None

    for m in range(n_bands):
      # e_ph coupling
      # carry = (energy, ratios)
      def scanned_fun(carry, kp):
        qc = tuple((jnp.array(elec_k) - kp) % n_sites)
        qd = tuple((kp - jnp.array(elec_k)) % n_sites)

        kp_i = lattice.get_site_num(kp)
        qc_i = lattice.get_site_num(qc)
        qd_i = lattice.get_site_num(qd)

        new_elec_n_k = (m, tuple(kp))

        new_phonon_occ = phonon_occ.at[qc].add(1)
        ratio = (jnp.sum(phonon_occ) < self.max_n_phonons) * wave.calc_overlap(new_elec_n_k,
                                                                               new_phonon_occ, parameters, lattice) / overlap /   (phonon_occ[qc] + 1)**0.5
        carry[0] -= jnp.array(self.g_mn_kq)[elec_n, m, kp_i, qc_i] * \
            (phonon_occ[qc] + 1)**0.5 * ratio
        carry[1] = carry[1].at[m, 2*kp_i].set(ratio)

        new_phonon_occ = phonon_occ.at[qd].add(-1)
        new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
        ratio = (phonon_occ[qd])**0.5 * wave.calc_overlap(new_elec_n_k,
                                                          new_phonon_occ, parameters, lattice) / overlap
        carry[0] -= jnp.array(self.g_mn_kq)[elec_n, m, kp_i, qc_i] * \
            (phonon_occ[qd])**0.5 * ratio
        carry[1] = carry[1].at[m, 2*kp_i + 1].set(ratio)

        return carry, (qc, qd)

      [energy, ratios], (qc, qd) = lax.scan(
          scanned_fun, [energy, ratios], jnp.array(lattice.sites))

      qc = jnp.stack(qc, axis=-1)
      qd = jnp.stack(qd, axis=-1)

    cumulative_ratios = jnp.cumsum(jnp.abs(ratios.reshape(-1)))
    weight = 1 / cumulative_ratios[-1]
    new_ind = jnp.searchsorted(
        cumulative_ratios, random_number * cumulative_ratios[-1])

    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)
    ##jax.debug.print('overlap_gradient: {}', overlap_gradient)
    #jax.debug.print('weight: {}', weight)
    #jax.debug.print('ratios: {}', ratios)
    #jax.debug.print('energy: {}', energy)
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)
    #jax.debug.print('qc: {}', qc)

    #walker[0] = (new_ind//2,)
    #walker[1] = lax.cond(new_ind % 2 == 0, lambda x: x.at[qc[new_ind//2]].add(1), lambda x: x.at[qd[new_ind//2]].add(-1), walker[1])
    new_band = new_ind // (2 * len(lattice.sites))
    new_ind = new_ind % (2 * len(lattice.sites))
    walker[0] = (new_band, tuple(jnp.array(lattice.sites)[new_ind//2]))
    walker[1] = lax.cond(new_ind % 2 == 0, lambda x: x.at[tuple(
        qc[new_ind//2])].add(1), lambda x: x.at[tuple(qd[new_ind//2])].add(-1), walker[1])
    #walker[1] = lax.cond(new_ind % 2 == 0, lambda x: x.at[qc[0][new_ind//2]].add(1), lambda x: x.at[qd[0][new_ind//2]].add(-1), walker[1])
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega_q, self.e_n_k, self.g_mn_kq, self.max_n_phonons))

@dataclass
class kq_ne_np():
  omega_nu_q: Sequence
  e_n_k: Sequence
  g_mn_nu_kq: Any
  max_n_phonons: Any = jnp.inf

  @partial(jit, static_argnums=(0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_n_k = walker[0]
    elec_n = walker[0][0]
    elec_k = walker[0][1]
    phonon_occ = walker[1]
    n_sites = jnp.array(lattice.shape)
    n_e_bands = len(self.e_n_k)
    n_p_bands = len(self.omega_nu_q)

    overlap = wave.calc_overlap(elec_n_k, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(
        elec_n_k, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)

    # diagonal
    energy = jnp.sum(jnp.array(self.omega_nu_q) * phonon_occ) + jnp.array(self.e_n_k)[elec_n][elec_k] + 0.j
    
    # e_ph coupling
    # carry = (energy, ratios, m, nu)
    def scanned_fun(carry, kp):
      m = carry[2]
      nu = carry[3]
      qc = tuple((jnp.array(elec_k) - kp) % n_sites)
      qd = tuple((kp - jnp.array(elec_k)) % n_sites)

      kp_i = lattice.get_site_num(kp)
      qc_i = lattice.get_site_num(qc)
      qd_i = lattice.get_site_num(qd)

      new_elec_n_k = (m, tuple(kp))

      new_phonon_occ = phonon_occ.at[(nu, *qc)].add(1)
      phonon_pos = (nu, tuple(qc))
      phonon_change = 1
      #ratio = (jnp.sum(phonon_occ) < self.max_n_phonons) * wave.calc_overlap(new_elec_n_k,
      #                                                                       new_phonon_occparameters, lattice) / overlap /     (phonon_occ[(nu, *qc)] + 1)**0.5
      ratio = (jnp.sum(phonon_occ) < self.max_n_phonons) * wave.calc_overlap_ratio(elec_n_k, new_elec_n_k, phonon_pos, phonon_change, parameters, lattice, overlap, new_phonon_occ) / (phonon_occ[(nu, *qc)] + 1)**0.5
      carry[0] -= jnp.array(self.g_mn_nu_kq)[elec_n, m, nu, kp_i, qc_i] * \
          (phonon_occ[(nu, *qc)] + 1)**0.5 * ratio
      carry[1] = carry[1].at[m, nu, 2*kp_i].set(ratio)

      new_phonon_occ = phonon_occ.at[(nu, *qd)].add(-1)
      new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
      phonon_pos = (nu, tuple(qd))
      phonon_change = -1 * (phonon_occ[(nu, *qd)] > 0)
      ratio = (phonon_occ[(nu, *qd)])**0.5 * wave.calc_overlap_ratio(elec_n_k, new_elec_n_k, phonon_pos, phonon_change, parameters, lattice, overlap, new_phonon_occ)
      carry[0] -= jnp.array(self.g_mn_nu_kq)[elec_n, m, nu, kp_i, qc_i] * (phonon_occ[(nu, *qd)])**0.5 * ratio
      carry[1] = carry[1].at[m, nu, 2*kp_i + 1].set(ratio)

      return carry, (qc, qd)
    
    def outer_scanned_fun(carry, m_nu):
      m = m_nu // n_p_bands
      nu = m_nu % n_p_bands
      [carry[0], carry[1], _, _], (qc, qd) = lax.scan(scanned_fun, [carry[0], carry[1], m, nu], jnp.array(lattice.sites))
      return [carry[0], carry[1], qc, qd], m_nu 
    
    ratios = jnp.zeros((n_e_bands, n_p_bands, 2 * len(lattice.sites),)) + 0.j
    qc = tuple([jnp.zeros(len(lattice.sites), dtype=jnp.int32) for _ in range(3)])
    qd = tuple([jnp.zeros(len(lattice.sites), dtype=jnp.int32) for _ in range(3)]) 
    [energy, ratios, qc, qd], _ = lax.scan(outer_scanned_fun, [energy, ratios, qc, qd], jnp.arange(n_e_bands * n_p_bands))    
    qc = jnp.stack(qc, axis=-1)
    qd = jnp.stack(qd, axis=-1)

    cumulative_ratios = jnp.cumsum(jnp.abs(ratios.reshape(-1)))
    weight = 1 / cumulative_ratios[-1]
    new_ind = jnp.searchsorted(
        cumulative_ratios, random_number * cumulative_ratios[-1])

    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)
    ##jax.debug.print('overlap_gradient: {}', overlap_gradient)
    #jax.debug.print('weight: {}', weight)
    #jax.debug.print('ratios: {}', ratios)
    #jax.debug.print('energy: {}', energy)
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)
    #jax.debug.print('qc: {}', qc)

    #walker[0] = (new_ind//2,)
    #walker[1] = lax.cond(new_ind % 2 == 0, lambda x: x.at[qc[new_ind//2]].add(1), lambda x: x.at[qd[new_ind//2]].add(-1), walker[1])
    new_e_band = new_ind // (2 * len(lattice.sites) * n_p_bands)
    p_band = (new_ind % (2 * len(lattice.sites) * n_p_bands)) // (2 * len(lattice.sites))
    new_ind = new_ind % (2 * len(lattice.sites))
    walker[0] = (new_e_band, tuple(jnp.array(lattice.sites)[new_ind//2]))
    walker[1] = lax.cond(new_ind % 2 == 0, lambda x: x.at[(p_band, *tuple(
        qc[new_ind//2]))].add(1), lambda x: x.at[(p_band, *tuple(qd[new_ind//2]))].add(-1), walker[1])
    #walker[1] = lax.cond(new_ind % 2 == 0, lambda x: x.at[qc[0][new_ind//2]].add(1), lambda x: x.at[qd[0][new_ind//2]].add(-1), walker[1])
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

    energy = jnp.where(jnp.isnan(energy), 0., energy)
    energy = jnp.where(jnp.isinf(energy), 0., energy)
    weight = jnp.where(jnp.isnan(weight), 0., weight)
    weight = jnp.where(jnp.isinf(weight), 0., weight)

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega_nu_q, self.e_n_k, self.g_mn_nu_kq, self.max_n_phonons))


@dataclass
class kq_1d():
  omega_k: Sequence
  e_k: Sequence
  g: Any
  max_n_phonons: Any = jnp.inf

  @partial(jit, static_argnums=(0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_k = walker[0]
    phonon_occ = walker[1]
    n_sites = lattice.shape[0]

    overlap = wave.calc_overlap(elec_k, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_k, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)

    # diagonal
    energy = jnp.sum(jnp.array(self.omega_k) * phonon_occ) + jnp.array(self.e_k)[elec_k[0]] + 0.j

    ratios = jnp.zeros((2 * n_sites,)) + 0.j
    # e_ph coupling
    # carry = (energy, ratios)
    def scanned_fun(carry, kp):
      qc = (elec_k[0] - kp) % n_sites
      qd = (kp - elec_k[0]) % n_sites

      new_elec_k = (kp,)

      new_phonon_occ = phonon_occ.at[qc].add(1)
      ratio = (phonon_occ[qc] < self.max_n_phonons) * wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[qc] + 1)**0.5
      carry[0] += jnp.array(self.g)[kp, qc] * (phonon_occ[qc] + 1)**0.5 * ratio
      carry[1] = carry[1].at[2*kp].set(ratio)

      new_phonon_occ = phonon_occ.at[qd].add(-1)
      new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
      ratio = (phonon_occ[qd])**0.5 * wave.calc_overlap(new_elec_k, new_phonon_occ, parameters, lattice) / overlap
      carry[0] += jnp.array(self.g)[kp, qc] * (phonon_occ[qd])**0.5 * ratio
      carry[1] = carry[1].at[2*kp + 1].set(ratio)

      return carry, (qc, qd)

    [energy, ratios], (qc, qd) = lax.scan(scanned_fun, [energy, ratios], jnp.arange(n_sites))

    cumulative_ratios = jnp.cumsum(jnp.abs(ratios))
    weight = 1 / cumulative_ratios[-1]
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])

    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)
    ##jax.debug.print('overlap_gradient: {}', overlap_gradient)
    #jax.debug.print('weight: {}', weight)
    #jax.debug.print('ratios: {}', jnp.array([ ratio_0, ratio_1, ratio_2, ratio_3 ]))
    #jax.debug.print('energy: {}', energy)
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}', new_ind)

    walker[0] = (new_ind//2,)
    walker[1] = lax.cond(new_ind % 2 == 0, lambda x: x.at[qc[new_ind//2]].add(1), lambda x: x.at[qd[new_ind//2]].add(-1), walker[1])
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega_k, self.e_k, self.g))

@dataclass
class ssh_1d():
  omega: float
  g: float
  max_n_phonons: Any = jnp.inf

  @partial(jit, static_argnums = (0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1]
    n_sites = lattice.n_sites

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)

    # diagonal
    energy = self.omega * jnp.sum(phonon_occ)

    # electron hop ->
    # bare
    new_elec_pos = ((elec_pos[0] + 1) % n_sites,)
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_0 = lax.cond(n_sites > 1, lambda x: x / overlap, lambda x: 0.j, new_overlap)
    energy -= ratio_0

    bond_pos = (elec_pos[0] * (1 - (n_sites == 2)),)
    # create phonon
    new_phonon_occ = phonon_occ.at[bond_pos].add(1)
    ratio_0_cp = (phonon_occ[bond_pos] < self.max_n_phonons) * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos] + 1)**0.5
    energy -= self.g * (phonon_occ[bond_pos] + 1)**0.5 * ratio_0_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos].add(-1)
    new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
    ratio_0_dp = (phonon_occ[bond_pos])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap
    energy -= self.g * (phonon_occ[bond_pos])**0.5 * ratio_0_dp

    # electron hop <-
    # bare
    new_elec_pos = ((elec_pos[0] - 1) % n_sites,)
    new_overlap = wave.calc_overlap(new_elec_pos, phonon_occ, parameters, lattice)
    ratio_1 = lax.cond(n_sites > 2, lambda x: x / overlap, lambda x: 0.j, new_overlap)
    energy -= ratio_1

    bond_pos = (new_elec_pos[0] * (1 - (n_sites==2)),)
    # create phonon
    new_phonon_occ = phonon_occ.at[bond_pos].add(1)
    ratio_1_cp = (phonon_occ[bond_pos] < self.max_n_phonons) * lax.cond(n_sites > 2, lambda x: wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos] + 1)**0.5, lambda x: 0.j, 0.)
    energy -= self.g * (phonon_occ[bond_pos] + 1)**0.5 * ratio_1_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos].add(-1)
    new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
    ratio_1_dp = lax.cond(n_sites > 2, lambda x: (phonon_occ[bond_pos])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap, lambda x: 0.j, 0.)
    energy -= self.g * (phonon_occ[bond_pos])**0.5 * ratio_1_dp

    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array([ ratio_0, ratio_0_cp, ratio_0_dp, ratio_1, ratio_1_cp, ratio_1_dp ])))
    weight = 1 / cumulative_ratios[-1]
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])

    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)
    #jax.debug.print('ratios: {}', jnp.array([ ratio_0, ratio_0_cp, ratio_0_dp, ratio_1, ratio_1_cp, ratio_1_dp ]))
    #jax.debug.print('energy: {}', energy)
    #jax.debug.print('weight: {}', weight)
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

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega, self.g))

@dataclass
class bm_ssh_1d():
  omega: float
  g: float
  max_n_phonons: Any = jnp.inf

  @partial(jit, static_argnums = (0, 3, 4))
  def local_energy_and_update(self, walker, parameters, wave, lattice, random_number):
    elec_pos = walker[0]
    phonon_occ = walker[1]
    n_phonons = jnp.sum(phonon_occ)
    n_sites = lattice.shape[0]

    overlap = wave.calc_overlap(elec_pos, phonon_occ, parameters, lattice)
    overlap_gradient = wave.calc_overlap_gradient(elec_pos, phonon_occ, parameters, lattice)
    qp_weight = lax.cond(jnp.sum(phonon_occ) == 0, lambda x: 1., lambda x: 0., 0.)

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
      ratio_1_co = (n_phonons < self.max_n_phonons) * (1 - (n_sites == 2) * (site == 1)) * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos] + 1)**0.5
      ratios.append(ratio_1_co)
      energy -= hop_sgn * self.g * (phonon_occ[elec_pos] + 1)**0.5 * ratio_1_co

      # destroy phonon on old site
      new_phonon_occ = phonon_occ.at[elec_pos].add(-1)
      new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
      ratio_1_do = (phonon_occ[elec_pos] > 0) *  (1 - (n_sites == 2) * (site == 1)) * (phonon_occ[elec_pos])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap
      ratios.append(ratio_1_do)
      energy -= hop_sgn * self.g * (phonon_occ[elec_pos])**0.5 * ratio_1_do

      # create phonon on new site
      new_phonon_occ = phonon_occ.at[new_elec_pos].add(1)
      ratio_1_cn = (n_phonons < self.max_n_phonons) *  (1 - (n_sites == 2) * (site == 1)) * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[new_elec_pos] + 1)**0.5
      ratios.append(ratio_1_cn)
      energy += hop_sgn * self.g * (phonon_occ[new_elec_pos] + 1)**0.5 * ratio_1_cn

      # destroy phonon on new site
      new_phonon_occ = phonon_occ.at[new_elec_pos].add(-1)
      new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
      ratio_1_dn = (phonon_occ[new_elec_pos] > 0) * (1 - (n_sites == 2) * (site == 1)) * (phonon_occ[new_elec_pos])**0.5 * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap
      ratios.append(ratio_1_dn)
      energy += hop_sgn * self.g * (phonon_occ[new_elec_pos])**0.5 * ratio_1_dn

    cumulative_ratios = jnp.cumsum(jnp.abs(jnp.array(ratios)))
    weight = 1 / cumulative_ratios[-1]
    new_ind = jnp.searchsorted(cumulative_ratios, random_number * cumulative_ratios[-1])

    #jax.debug.print('\nwalker: {}', walker)
    #jax.debug.print('overlap: {}', overlap)
    #jax.debug.print('ratios: {}', jnp.array(ratios))
    #jax.debug.print('energy: {}', energy)
    #jax.debug.print('weight: {}', weight)
    #jax.debug.print('random_number: {}', random_number)
    #jax.debug.print('new_ind: {}\n', new_ind)

    # electron hop
    new_elec_pos = jnp.array([(elec_pos[0] + 1) % n_sites, (elec_pos[0] - 1) % n_sites])[new_ind // 5]
    walker[0] = (new_elec_pos,)

    # eph
    ind_0 = jnp.array([elec_pos[0], elec_pos[0], elec_pos[0], new_elec_pos, new_elec_pos])[new_ind % 5]
    ind_1 = jnp.array([0, 1, -1, 1, -1])[new_ind % 5]
    walker[1] = walker[1].at[ind_0].add(ind_1)
    #jax.debug.print('new_walker: {}', walker)
    walker[1] = jnp.where(walker[1] < 0, 0, walker[1])
    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega, self.g))

@dataclass
class long_range_1d():
  omega: float
  g: float
  zeta: float
  max_n_phonons: Any = jnp.inf

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
      ratio = (phonon_occ[pos] < self.max_n_phonons) * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[pos] + 1)**0.5
      carry[0] -= self.g * jnp.exp(-dist / self.zeta) * (phonon_occ[pos] + 1)**0.5 * ratio / (1 + dist**2)**1.5
      carry[1] = carry[1].at[2*x + 2].set(ratio)

      new_phonon_occ = phonon_occ.at[pos].add(-1)
      new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
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

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega, self.g, self.zeta))

@dataclass
class holstein_2d():
  omega: float
  g: float
  max_n_phonons: Any = jnp.inf

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
    ratio_2 = (phonon_occ[elec_pos[0], elec_pos[1]] < self.max_n_phonons) * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos[0], elec_pos[1]] + 1)**0.5
    energy -= self.g * (phonon_occ[elec_pos[0], elec_pos[1]] + 1)**0.5 * ratio_2

    new_phonon_occ = phonon_occ.at[elec_pos[0], elec_pos[1]].add(-1)
    new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
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

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega, self.g))

@dataclass
class ssh_2d():
  omega: float
  g: float
  max_n_phonons: Any = jnp.inf

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
    ratio_0_x_cp = (phonon_occ[bond_pos_0_x] < self.max_n_phonons) * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos_0_x] + 1)**0.5
    energy -= self.g * (phonon_occ[bond_pos_0_x] + 1)**0.5 * ratio_0_x_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos_0_x].add(-1)
    new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
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
    ratio_1_x_cp = (phonon_occ[bond_pos_1_x] < self.max_n_phonons) * lax.cond(l_x > 2, lambda x: wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos_1_x] + 1)**0.5, lambda x: 0., 0.)
    energy -= self.g * (phonon_occ[bond_pos_1_x] + 1)**0.5 * ratio_1_x_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos_1_x].add(-1)
    new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
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
    ratio_0_y_cp = (phonon_occ[bond_pos_0_y] < self.max_n_phonons) * wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos_0_y] + 1)**0.5
    energy -= self.g * (phonon_occ[bond_pos_0_y] + 1)**0.5 * ratio_0_y_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos_0_y].add(-1)
    new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
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
    ratio_1_y_cp = (phonon_occ[bond_pos_1_y] < self.max_n_phonons) * lax.cond(l_y > 2, lambda x: wave.calc_overlap(new_elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[bond_pos_1_y] + 1)**0.5, lambda x: 0., 0.)
    energy -= self.g * (phonon_occ[bond_pos_1_y] + 1)**0.5 * ratio_1_y_cp

    # destroy phonon
    new_phonon_occ = phonon_occ.at[bond_pos_1_y].add(-1)
    new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
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

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega, self.g))


@dataclass
class long_range_2d():
  omega: float
  g: float
  zeta: float
  max_n_phonons: Any = jnp.inf

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
      ratio = (phonon_occ[pos[0], pos[1]] < self.max_n_phonons) * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[pos[0], pos[1]] + 1)**0.5
      carry[0] -= self.g * jnp.exp(-dist / self.zeta) * (phonon_occ[pos[0], pos[1]] + 1)**0.5 * ratio / (1 + dist**2)**1.5
      carry[1] = carry[1].at[2*carry[2] + 4].set(ratio)

      new_phonon_occ = phonon_occ.at[pos[0], pos[1]].add(-1)
      new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
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

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega, self.g, self.zeta))


@dataclass
class holstein_3d():
  omega: float
  g: float
  max_n_phonons: any = jnp.inf

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
    ratio_2 = (phonon_occ[elec_pos[0], elec_pos[1], elec_pos[2]] < self.max_n_phonons) * wave.calc_overlap(elec_pos, new_phonon_occ, parameters, lattice) / overlap / (phonon_occ[elec_pos[0], elec_pos[1],elec_pos[2]] + 1)**0.5
    energy -= self.g * (phonon_occ[elec_pos[0], elec_pos[1], elec_pos[2]] + 1)**0.5 * ratio_2

    new_phonon_occ = phonon_occ.at[elec_pos[0], elec_pos[1], elec_pos[2]].add(-1)
    new_phonon_occ = jnp.where(new_phonon_occ < 0, 0, new_phonon_occ)
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

    return energy, qp_weight, overlap_gradient, weight, walker, overlap

  def __hash__(self):
    return hash((self.omega, self.g))

if __name__ == "__main__":
  import lattices, wavefunctions, models
  n_sites = 4
  n_bands = 2
  lattice = lattices.one_dimensional_chain(n_sites)

  model_r = models.MLP([5, 1])
  model_phi = models.MLP([5, 1])
  model_input = jnp.zeros((1 + n_bands)*n_sites)
  nn_parameters_r = model_r.init(random.PRNGKey(0), model_input, mutable=True)
  nn_parameters_phi = model_phi.init(random.PRNGKey(1), model_input, mutable=True)
  n_nn_parameters = sum(x.size for x in tree_util.tree_leaves(nn_parameters_r)) + sum(x.size for x in tree_util.tree_leaves(nn_parameters_phi))
  parameters = [ nn_parameters_r, nn_parameters_phi ]
  lattice_shape = (n_bands, *lattice.shape)
  wave = wavefunctions.nn_complex_n(model_r.apply, model_phi.apply, n_nn_parameters, lattice_shape)

  walker = [ (0, (0,)), jnp.zeros(lattice.shape) ]

  omega_q = tuple(1. for _ in range(n_sites))
  e_n_k = (tuple(-2. * np.cos(2. * np.pi * k / n_sites) for k in range(n_sites)),
           tuple(-2. * np.cos(2. * np.pi * k / n_sites) for k in range(n_sites)))
  g_mn_kq = ((tuple(tuple(1./n_sites**0.5 for _ in range(n_sites)) for _ in range(n_sites)),
              tuple(tuple(1./n_sites**0.5 for _ in range(n_sites)) for _ in range(n_sites))),
             (tuple(tuple(1./n_sites**0.5 for _ in range(n_sites)) for _ in range(n_sites)),
              tuple(tuple(1./n_sites**0.5 for _ in range(n_sites)) for _ in range(n_sites))))

  ham = kq_ne(omega_q, e_n_k, g_mn_kq)
  random_number = 0.9
  energy, qp_weight, overlap_gradient, weight, walker, overlap = ham.local_energy_and_update(walker, parameters, wave, lattice, random_number)
