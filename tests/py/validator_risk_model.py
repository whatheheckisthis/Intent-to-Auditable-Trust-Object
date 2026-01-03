def correlated_binomial_risk(n_validators, p_compromise, correlation=0.1):
    """
    Generates correlated compromise probabilities for validators.
    """
    base = jax.random.bernoulli(jax.random.PRNGKey(1), p_compromise, (n_validators,))
    correlated = jnp.clip(base + correlation*jnp.mean(base), 0, 1)
    return correlated
