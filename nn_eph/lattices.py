import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from jax import random, lax, tree_util, grad, value_and_grad, jit, numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple, Any
from dataclasses import dataclass
from functools import partial
from jax.tree_util import register_pytree_node_class

def make_phonon_basis(n_sites, max_phonons):
  basis = [ ]
  coefficients = [ 1 for _ in range(n_sites) ]
  for i in range(max_phonons+1):
    basis += frobenius(n_sites, coefficients, i)
  return basis

def frobenius(n, coefficients, target):
    """
    Enumerates solutions of the Frobenius equation with n integers.

    Args:
    - n: int, number of integers in the solution
    - coefficients: list of ints, coefficients for the Frobenius equation
    - target: int, target value for the Frobenius equation

    Returns:
    - list of tuples, each tuple represents a solution of the Frobenius equation
    """
    dp = [0] + [-1] * target  # Initialize the dynamic programming array

    for i in range(1, target + 1):
        for j in range(n):
            if coefficients[j] <= i and dp[i - coefficients[j]] != -1:
                dp[i] = j
                break

    if dp[target] == -1:
        return []  # No solution exists

    solutions = []
    current_solution = [0] * n

    def get_solution(i, remaining):
        if i == -1:
            if remaining == 0:
                solutions.append(jnp.array(current_solution))
            return

        for j in range(remaining // coefficients[i], -1, -1):
            current_solution[i] = j
            get_solution(i - 1, remaining - j * coefficients[i])

    get_solution(n - 1, target)
    return solutions

def frobenius_0(n, coefficients, target):
  def backtrack(index, current_solution):
    nonlocal solutions

    # Base case: when we have n integers in the solution
    if index == n:
      # Check if the current solution satisfies the Frobenius equation
      if sum([coefficients[i] * current_solution[i] for i in range(n)]) == target:
        solutions.append(jnp.array(current_solution))
      return

    # Recursive case: for each possible value of the current integer
    for i in range(target + 1):
      current_solution[index] = i
      backtrack(index + 1, current_solution)

  solutions = []
  current_solution = [0] * n
  backtrack(0, current_solution)
  return solutions

@dataclass
@register_pytree_node_class
class one_dimensional_chain():
  n_sites: int
  shape: tuple = None
  sites: Sequence = None
  bonds: Sequence = None

  def __post_init__(self):
    self.shape = (self.n_sites,)
    self.sites = tuple([ (i,) for i in range(self.n_sites) ])
    self.bonds = tuple([ (i,) for i in range(self.n_sites) ]) if self.n_sites > 2 else tuple([ (0,) ])

  def get_bond_site_distance(self, bond, site):
    neigboring_sites = self.get_neighboring_sites(bond)
    dist_1 = self.get_distance(neigboring_sites[0], site)
    dist_2 = self.get_distance(neigboring_sites[1], site)
    lr = (dist_1 < dist_2) * 1. - (dist_1 > dist_2) * 1.
    return jnp.min(jnp.array([dist_1, dist_2])), lr
  
  def get_site_num(self, pos):
    return pos[0]

  def make_polaron_basis(self, max_n_phonons):
    phonon_basis = make_phonon_basis(self.n_sites, max_n_phonons)
    polaron_basis = tuple([ (i,), phonon_state ] for i in range(self.n_sites) for phonon_state in phonon_basis)
    return polaron_basis

  def get_marshall_sign(self, walker):
    if isinstance(walker, list):
      # TODO: this is a bit hacky
      walker = walker[0]
    walker_a = walker[::2]
    return (-1)**jnp.sum(jnp.where(walker_a > 0, 1, 0))

  def get_symm_fac(self, pos, k):
    return jnp.exp(2 * jnp.pi * 1.j * k[0] * pos[0] / self.n_sites) if k is not None else 1.

  def get_neighboring_bonds(self, pos):
    return jnp.array([ ((pos[0] - 1) % self.n_sites,), (pos[0],) ]) if self.n_sites > 2 else jnp.array([ (0,) ])

  def get_neighboring_sites(self, bond):
    return [ (bond[0] % self.n_sites,), ((bond[0] + 1) % self.n_sites,) ]

  def get_distance(self, pos_1, pos_2):
    return jnp.min(jnp.array([jnp.abs(pos_1[0] - pos_2[0]), self.n_sites - jnp.abs(pos_1[0] - pos_2[0])]))

  def get_bond_distance(self, pos_1, pos_2):
    return jnp.min(jnp.array([jnp.abs(pos_1[0] - pos_2[0]), self.n_sites - jnp.abs(pos_1[0] - pos_2[0])]))

  def __hash__(self):
    return hash((self.n_sites, self.shape, self.sites, self.bonds))

  def tree_flatten(self):
    return (), (self.n_sites, self.shape, self.sites, self.bonds)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*aux_data)

@dataclass
@register_pytree_node_class
class two_dimensional_grid():
  l_x: int
  l_y: int
  shape: tuple = None
  shell_distances: Sequence = None
  bond_shell_distances: Sequence = None
  sites: Sequence = None
  bonds: Sequence = None

  def __post_init__(self):
    self.shape = (self.l_y, self.l_x)
    distances = [ ]
    for x in range(self.l_x//2+1):
      for y in range(self.l_y//2+1):
        dist = x**2 + y**2
        distances.append(dist)
    distances = [*set(distances)]
    distances.sort()
    self.shell_distances = tuple(distances)
    self.sites = tuple([ (i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y) ])

    bond_distances = [ ]
    for x in range(self.l_x + 1):
      for y in range(self.l_y + 1):
        if x %2 == y%2:
          dist = x**2 + y**2
          bond_distances.append(dist)
    bond_distances = [*set(bond_distances)]
    bond_distances.sort()
    self.bond_shell_distances = tuple(bond_distances)
    if (self.l_x == 2) & (self.l_y == 2):
      self.bonds = ( (0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0) )
    elif self.l_x == 2:
      self.bonds = tuple([ (0, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y) ] + [ (1, i, 0) for i in self.l_y ])
    elif self.l_y == 2:
      self.bonds = tuple([ (0, 0, i) for i in self.l_x ] + [ (1, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y) ])
    else:
      self.bonds = tuple([ (0, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y) ] + [ (1, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y) ])

  def get_site_num(self, pos):
    return pos[1] + self.l_x * pos[0]

  def make_polaron_basis(self, max_n_phonons):
    phonon_basis = make_phonon_basis(self.l_x * self.l_y, max_n_phonons)
    polaron_basis = tuple([ site, phonon_state.reshape((self.l_y, self.l_x)) ] for site in self.sites for phonon_state in phonon_basis)
    return polaron_basis

  def get_marshall_sign(self, walker):
    if isinstance(walker, list):
      # TODO: this is a bit hacky
      walker = walker[0]
    walker_a = walker[::2, ::2]
    return (-1)**jnp.sum(jnp.where(walker_a > 0, 1, 0))

  # does not work
  def get_symm_fac(self, pos, k):
    return 1.
    #return jnp.exp(2 * jnp.pi * 1.j * k[0] * pos[0] / self.n_sites) * jnp.exp(2 * jnp.pi * 1.j * k[1] * pos[1] / self.n_sites) if k is not None else 1.

  def get_distance(self, pos_1, pos_2):
    dist_y = jnp.min(jnp.array([jnp.abs(pos_1[0] - pos_2[0]), self.l_y - jnp.abs(pos_1[0] - pos_2[0])]))
    dist_x = jnp.min(jnp.array([jnp.abs(pos_1[1] - pos_2[1]), self.l_x - jnp.abs(pos_1[1] - pos_2[1])]))
    dist = dist_x**2 + dist_y**2
    shell_number = jnp.searchsorted(jnp.array(self.shell_distances), dist)
    return shell_number

  def get_bond_distance(self, pos_1, pos_2):
    shifted_pos_1 = 2 * jnp.array(pos_1[1:])
    shifted_pos_1 = shifted_pos_1.at[pos_1[0]].add(1)
    shifted_pos_2 = 2 * jnp.array(pos_2[1:])
    shifted_pos_2 = shifted_pos_2.at[pos_2[0]].add(1)
    dist_y = jnp.min(jnp.array([jnp.abs(shifted_pos_1[0] - shifted_pos_2[0]), 2 * self.l_y - jnp.abs(shifted_pos_1[0] - shifted_pos_2[0])]))
    dist_x = jnp.min(jnp.array([jnp.abs(shifted_pos_1[1] - shifted_pos_2[1]), 2 * self.l_x - jnp.abs(shifted_pos_1[1] - shifted_pos_2[1])]))
    dist = dist_x**2 + dist_y**2
    shell_number = jnp.searchsorted(jnp.array(self.bond_shell_distances), dist)
    return shell_number

  def get_neighboring_bonds(self, pos):
    down = (0, *pos)
    right = (1, *pos)
    up = (0, (pos[0] - 1) % self.l_y, pos[1])
    left = (1, pos[0], (pos[1] - 1) % self.l_x)
    neighbors = [  ]
    if (self.l_x == 2) & (self.l_y == 2):
      neighbors = [ (0, 0, pos[1]), (1, pos[0], 0) ]
    elif self.l_x == 2:
      neighbors = [ down, up, right  if pos[1] == 0 else left ]
    elif self.l_y == 2:
      neighbors = [ left, right, down  if pos[0] == 0 else up ]
    else:
      neighbors = [ down, right, up, left ]
    return jnp.array(neighbors)

  def get_neighboring_sites(self, bond):
    neighbors = [ ]
    if bond[0] == 0:
      neighbors = [ (bond[1], bond[2]), ((bond[1] + 1) % self.l_y, bond[2]) ]
    else:
      neighbors = [ (bond[1], bond[2]), (bond[1], (bond[2] + 1) % self.l_x) ]
    return jnp.array(neighbors)

  def __hash__(self):
    return hash((self.l_x, self.l_y, self.shape, self.shell_distances, self.bond_shell_distances, self.sites, self.bonds))

  def tree_flatten(self):
    return (), (self.l_x, self.l_y, self.shape, self.shell_distances, self.bond_shell_distances, self.sites, self.bonds)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*aux_data)


@dataclass
@register_pytree_node_class
class three_dimensional_grid():
  l_x: int
  l_y: int
  l_z: int
  shape: tuple = None
  shell_distances: Sequence = None
  sites: Sequence = None
  bonds: Sequence = None

  def __post_init__(self):
    self.shape = (self.l_z, self.l_y, self.l_x)
    distances = [ ]
    for x in range(self.l_x//2+1):
      for y in range(self.l_y//2+1):
        for z in range(self.l_z//2+1):
          dist = x**2 + y**2 + z**2
          distances.append(dist)
    distances = [*set(distances)]
    distances.sort()
    self.shell_distances = tuple(distances)
    self.sites = tuple([ (i // (self.l_x * self.l_y), (i % (self.l_x * self.l_y)) // self.l_x, (i % (self.l_x * self.l_y)) % self.l_x) for i in range(self.l_x * self.l_y * self.l_z) ])
    # TODO: fix bonds
    #self.bonds = tuple([ (i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y) ])

  def make_polaron_basis(self, max_n_phonons):
    phonon_basis = make_phonon_basis(self.l_x * self.l_y * self.l_z, max_n_phonons)
    polaron_basis = tuple([ site, phonon_state.reshape((self.l_x, self.l_y, self.l_z)) ] for site in self.sites for phonon_state in phonon_basis)
    return polaron_basis

  def make_polaron_basis_n(self, n_bands, max_n_phonons):
    phonon_basis = make_phonon_basis(self.l_x * self.l_y * self.l_z, max_n_phonons)
    electronic_basis = tuple([ (n, site) for n in range(n_bands) for site in self.sites ])
    polaron_basis = tuple([ site, phonon_state.reshape((self.l_x, self.l_y, self.l_z)) ] for site in electronic_basis for phonon_state in phonon_basis)
    return polaron_basis


  def get_site_num(self, pos):
    return pos[2] + self.l_x * pos[1] + (self.l_x * self.l_y) * pos[0]

  # does not work
  def get_symm_fac(self, pos, k):
    return 1.
    #return jnp.exp(2 * jnp.pi * 1.j * k[0] * pos[0] / self.n_sites) * jnp.exp(2 * jnp.pi * 1.j * k[1] * pos[1] / self.n_sites) if k is not None else 1.

  def get_distance(self, pos_1, pos_2):
    dist_z = jnp.min(jnp.array([jnp.abs(pos_1[0] - pos_2[0]), self.l_z - jnp.abs(pos_1[0] - pos_2[0])]))
    dist_y = jnp.min(jnp.array([jnp.abs(pos_1[1] - pos_2[1]), self.l_y - jnp.abs(pos_1[1] - pos_2[1])]))
    dist_x = jnp.min(jnp.array([jnp.abs(pos_1[2] - pos_2[2]), self.l_x - jnp.abs(pos_1[2] - pos_2[2])]))
    dist = dist_x**2 + dist_y**2 + dist_z**2
    shell_number = jnp.searchsorted(jnp.array(self.shell_distances), dist)
    return shell_number

  def get_bond_distance(self, pos_1, pos_2):
    #dist_y = jnp.min(jnp.array([jnp.abs(pos_1[0] - pos_2[0]), self.l_y - jnp.abs(pos_1[0] - pos_2[0])]))
    #dist_x = jnp.min(jnp.array([jnp.abs(pos_1[1] - pos_2[1]), self.l_x - jnp.abs(pos_1[1] - pos_2[1])]))
    #dist = dist_x**2 + dist_y**2
    #shell_number = jnp.searchsorted(jnp.array(self.shell_distances), dist)
    #return shell_number
    return 0

  # TODO: fix this
  def get_neighboring_bonds(self, pos):
    return jnp.array([ pos ])

  def __hash__(self):
    return hash((self.l_x, self.l_y, self.l_z, self.shape, self.shell_distances, self.sites, self.bonds))

  def tree_flatten(self):
    return (), (self.l_x, self.l_y, self.l_z, self.shape, self.shell_distances, self.sites, self.bonds)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*aux_data)

if __name__ == "__main__":
  lattice = one_dimensional_chain(4)
  bond = (1,)
  site = (0,)
  #print(lattice.get_bond_site_distance(bond, site))
  lattice = three_dimensional_grid(2, 2, 2)
  basis = lattice.make_polaron_basis_n(2, 1)
  #basis = make_phonon_basis(30, 1)
  print(basis)
