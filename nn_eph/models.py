import os
import numpy as np
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from jax import random, lax, tree_util, grad, value_and_grad, jit, numpy as jnp
import jax
from flax import linen as nn
from typing import Sequence, Any
from functools import partial

def complex_kernel_init(rng, shape, dtype):
  fan_in = np.prod(shape) // shape[-1]
  x = random.normal(rng, shape) + 0.j
  return x * (fan_in * 2)**(-0.5)

def complex_kernel_init_1(rng, shape, dtype):
  fan_in = np.prod(shape) // shape[-1]
  x = random.normal(rng, shape) + 0.j
  return 3. * x

class MLP(nn.Module):
  n_neurons: Sequence[int]
  activation: Any = nn.relu
  kernel_init: Any = jax.nn.initializers.lecun_normal()
  bias_init: Any = jax.nn.initializers.zeros
  #bias_init: Any = jax.nn.initializers.ones
  param_dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = x.reshape(-1)
    for n in self.n_neurons[:-1]:
      x = self.activation(nn.Dense(n, param_dtype=self.param_dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(x))
    x = nn.Dense(self.n_neurons[-1], param_dtype=self.param_dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
    return x


class CNN(nn.Module):
  channels: Sequence[int]
  kernel_sizes: Sequence
  activation: Any = nn.relu
  kernel_init: Any = jax.nn.initializers.lecun_normal()
  param_dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = x.reshape(1, *x.shape)
    for layer in range(len(self.channels)):
      x = nn.Conv(features=self.channels[layer], kernel_size=self.kernel_sizes[layer], padding='CIRCULAR', param_dtype=self.param_dtype, kernel_init=self.kernel_init)(x)
      x = self.activation(x)
    return jnp.array([jnp.sum(x)])

if __name__=="__main__":
  model = MLP([1], param_dtype=jnp.complex64, kernel_init=complex_kernel_init)
  model_input = jnp.array([0., -2., 2., 1.])
  nn_parameters = model.init(random.PRNGKey(0), model_input)
  print(nn_parameters)
