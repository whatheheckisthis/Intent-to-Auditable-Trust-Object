def inject_entropy(confidences, alpha=0.05):
    """
    Add Gaussian noise to per-inference confidence scores
    to model uncertainty or adversarial perturbation.
    """
    noise = jax.random.normal(jax.random.PRNGKey(0), confidences.shape) * alpha
    return jnp.clip(confidences + noise, 0, 1)
