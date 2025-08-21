def regularize_entropy(H_x, epsilon=1e-8):
    """
    Ensure entropy term is well-behaved for PGD updates.
    """
    import jax.numpy as jnp
    return H_x + epsilon
