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
  sites: Sequence = None

  def __post_init__(self):
    self.sites = tuple([ (i,) for i in range(self.n_sites) ])

  def get_distance(self, pos_1, pos_2):
    return jnp.min(jnp.array([jnp.abs(pos_1 - pos_2), self.n_sites - jnp.abs(pos_1 - pos_2)]))

  def __hash__(self):
    return hash((self.n_sites, self.sites))


@dataclass
class two_dimensional_grid():
  l_x: int
  l_y: int
  shell_distances: Sequence = None
  sites: Sequence = None

  def __post_init__(self):
    distances = [ ]
    for x in range(self.l_x//2+1):
      for y in range(self.l_y//2+1):
        dist = x**2 + y**2
        distances.append(dist)
    distances = [*set(distances)]
    distances.sort()
    self.shell_distances = tuple(distances)
    self.sites = tuple([ (i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y) ])

  def get_distance(self, pos_1, pos_2):
    dist_y = jnp.min(jnp.array([jnp.abs(pos_1[0] - pos_2[0]), self.l_y - jnp.abs(pos_1[0] - pos_2[0])]))
    dist_x = jnp.min(jnp.array([jnp.abs(pos_1[1] - pos_2[1]), self.l_x - jnp.abs(pos_1[1] - pos_2[1])]))
    dist = dist_x**2 + dist_y**2
    shell_number = jnp.searchsorted(jnp.array(self.shell_distances), dist)
    return shell_number

  def __hash__(self):
    return hash((self.l_x, self.l_y, self.shell_distances))


if __name__ == "__main__":
  n_sites = 6
  lattice = one_dimensional_chain(n_sites)
  print(lattice.get_distance(0, 8))

  l_x, l_y = 4, 4
  lattice = two_dimensional_grid(l_x, l_y)
  print(lattice.get_distance((0,0), (0,1)))

