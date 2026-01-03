def compute_validation_agreement(results):
    """
    Compute fraction of validators agreeing on each inference.
    """
    import jax.numpy as jnp
    return jnp.mean(jnp.array(results), axis=1)
