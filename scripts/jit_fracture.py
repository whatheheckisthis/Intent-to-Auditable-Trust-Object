import jax
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

def setup_mesh_and_sharding(devices):
    mesh = mesh_utils.create_device_mesh((len(devices),))
    return mesh

def pjit_transform(fn, in_shardings, out_shardings):
    return pjit(fn, in_shardings=in_shardings, out_shardings=out_shardings)
