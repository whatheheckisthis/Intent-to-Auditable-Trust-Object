def hydrogen_trial_radii(num_points=100, alpha=1.0):
    """
    Generate r-values for hydrogen atom trial wavefunction.
    """
    import jax.numpy as jnp
    return jnp.linspace(0.0, 10.0, num_points), alpha
