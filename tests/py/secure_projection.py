def project(K, x):
    """
    Euclidean projection onto closed set K
    """
    # placeholder: assumes K = box constraints
    import jax.numpy as jnp
    lower, upper = K
    return jnp.clip(x, lower, upper)
