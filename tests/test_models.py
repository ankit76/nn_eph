import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
#os.environ['JAX_ENABLE_X64'] = 'True'
import pytest
from jax import random, numpy as jnp
from nn_eph import models

def test_mlp():
  model = models.MLP([ 4, 1 ])
  model_input = jnp.ones((8))
  nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
  output = model.apply(nn_parameters, model_input)
  assert output.size == 1

  model = models.MLP([ 2, 3, 1 ])
  model_input = jnp.ones((2, 2))
  nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
  output = model.apply(nn_parameters, model_input)
  assert output.size == 1

def test_cnn():
  model = models.CNN([ 4, 2 ], [ (2,), (2,) ])
  model_input = jnp.ones((1, 4, 1))
  nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
  output = model.apply(nn_parameters, model_input)
  assert output.size == 1

  model = models.CNN([ 4, 2 ], [ (2,2), (2,2) ])
  model_input = jnp.ones((1, 4, 4, 2))
  nn_parameters = model.init(random.PRNGKey(0), model_input, mutable=True)
  output = model.apply(nn_parameters, model_input)
  assert output.size == 1


if __name__ == "__main__":
  test_mlp()
  test_cnn()
