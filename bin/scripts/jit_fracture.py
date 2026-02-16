# pjit_fracture.py
from jax.experimental import pjit, mesh_utils
import jax.numpy as jnp

def fractured_kernel(kernel_fn, x, devices=None, mesh_axis="x"):
    """Fracture kernel execution across devices using pjit."""
    devices = devices or mesh_utils.create_device_mesh((jax.device_count(),))
    
    def wrapped(x):
        return kernel_fn(x)
    
    sharded_fn = pjit.pjit(
        wrapped,
        in_shardings=(pjit.PartitionSpec(mesh_axis),),
        out_shardings=pjit.PartitionSpec(mesh_axis)
    )
    
    with mesh_utils.Mesh(devices, (mesh_axis,)):
        return sharded_fn(x)
