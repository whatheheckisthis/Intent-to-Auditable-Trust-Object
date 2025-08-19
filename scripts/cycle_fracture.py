import jax
import jax.numpy as jnp

def cycle_fracture(x, n_cycles=5, fracture_factor=0.5):
    results = []
    for cycle in range(n_cycles):
        x = x * (1.0 - fracture_factor) + fracture_factor * jnp.roll(x, 1, axis=0)
        results.append(x)
    return jnp.array(results)
