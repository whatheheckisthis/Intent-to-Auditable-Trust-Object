import jax.numpy as jnp
from itertools import permutations, combinations

def laplacian_3d(f, dx):
    lap = jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) + \
          jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) + \
          jnp.roll(f, 1, axis=2) + jnp.roll(f, -1, axis=2) - 6*f
    return lap / dx**2

def hash_commit(x, prev_hash=""):
    import hashlib
    return hashlib.sha256((prev_hash + str(x)).encode()).hexdigest()
