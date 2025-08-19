import jax
import jax.numpy as jnp
from utils_math import laplacian_3d

grid_size = 8
mass = 1.0
lambda_phi4 = 0.1
dx = 0.01

def H_lattice(phi):
    lap = laplacian_3d(phi, dx)
    kinetic = 0.5 * jnp.sum(lap**2)
    potential = 0.5 * mass**2 * jnp.sum(phi**2) + lambda_phi4 * jnp.sum(phi**4)
    return kinetic + potential
