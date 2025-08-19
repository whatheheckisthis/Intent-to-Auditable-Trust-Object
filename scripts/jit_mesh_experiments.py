import jax
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
import jax.numpy as jnp

devices = jax.devices()
mesh = mesh_utils.create_device_mesh((len(devices),))

def pjit_forward(fn, x):
    sharded_fn = pjit(fn)
    return sharded_fn(x)
