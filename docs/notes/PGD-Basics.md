 ```bash

from jax import grad, jit
import jax.numpy as jnp

def objective_fn(x):
    return jnp.sum(jnp.square(x - 3.0))

@jit
def PGD_step(x, eta=0.1):
    g = grad(objective_fn)(x)
    return x - eta * g
