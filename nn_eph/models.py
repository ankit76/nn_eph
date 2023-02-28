import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from jax import random, lax, tree_util, grad, value_and_grad, jit, numpy as jnp
from flax import linen as nn
from typing import Sequence, Any
from functools import partial


class MLP(nn.Module):
  n_neurons: Sequence[int]
  activation: Any = nn.relu

  @nn.compact
  def __call__(self, x):
    x = x.reshape(-1)
    for n in self.n_neurons[:-1]:
      x = self.activation(nn.Dense(n)(x))
    x = nn.Dense(self.n_neurons[-1])(x)
    return x


class CNN(nn.Module):
  channels: Sequence[int]
  kernel_sizes: Sequence
  activation: Any = nn.relu

  @nn.compact
  def __call__(self, x):
    x = x.reshape(1, *x.shape)
    for layer in range(len(self.channels)):
      x = nn.Conv(features=self.channels[layer], kernel_size=self.kernel_sizes[layer], padding='CIRCULAR')(x)
      x = self.activation(x)
    return jnp.array([jnp.sum(x)])

if __name__ == "__main__":
  #model = MLP([ 2, 1 ])
  model = CNN([ 1 ], [ (2,) ])
  model_input = jnp.ones((1, 4, 1))
  nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
  #n_sites = 10
  #model = CNN([ 5, 1, n_sites // 2 + 1 ], [ (4,), (4,) ])
  model_input = jnp.zeros((1, n_sites, 1))
  output = model.apply(nn_parameters, model_input)
  #fnn_input = jnp.ones((1,))

