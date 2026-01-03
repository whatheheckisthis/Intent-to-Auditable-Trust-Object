import jax.numpy as jnp

def discretize_lattice_field(dimensions, spacing=1.0):
    """
    Create a lattice for QFT field discretization.
    """
    grids = [jnp.arange(0, L, spacing) for L in dimensions]
    return jnp.meshgrid(*grids, indexing='ij')
