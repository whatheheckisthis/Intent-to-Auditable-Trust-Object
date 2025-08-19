import jax.numpy as jnp

def bayesian_update(prior, likelihood, evidence):
    """
    Standard Bayesian update for discrete probability distributions.
    """
    posterior = prior * likelihood(evidence)
    posterior /= jnp.sum(posterior)
    return posterior
