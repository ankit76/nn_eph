import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from jax import random, lax, tree_util, grad, value_and_grad, jit, numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple, Any
from dataclasses import dataclass
from functools import partial

@dataclass
class one_dimensional_chain():
  n_sites: int
  shape: tuple = None
  sites: Sequence = None
  bonds: Sequence = None

  def __post_init__(self):
    self.shape = (self.n_sites,)
    self.sites = tuple([ (i,) for i in range(self.n_sites) ])
    self.bonds = tuple([ (i,) for i in range(self.n_sites) ]) if self.n_sites > 2 else tuple([ (0,) ])

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


@dataclass
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

  def __hash__(self):
    return hash((self.l_x, self.l_y, self.shape, self.shell_distances, self.bond_shell_distances, self.sites, self.bonds))


@dataclass
class three_dimensional_grid():
  l_x: int
  l_y: int
  l_z: int
  shape: tuple = None
  shell_distances: Sequence = None
  sites: Sequence = None
  bonds: Sequence = None

  def __post_init__(self):
    self.shape = (self.l_x, self.l_y, self.l_z)
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
