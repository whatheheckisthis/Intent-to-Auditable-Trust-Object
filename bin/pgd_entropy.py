# pgd_entropy.py
import jax
import jax.numpy as jnp

def entropy_pgd_step(f, x, grad_fn, step_size, proj, entropy_kappa, key):
    """Single PGD step with entropy injection."""
    grad = grad_fn(x)
    noise = jax.random.normal(key, x.shape) * entropy_kappa
    x_new = proj(x - step_size * grad + noise)
    return x_new

def entropy_pgd(f, x0, grad_fn, proj, step_size=0.01, entropy_kappa=1e-3, max_iter=180, seed=42):
    """PGD with entropy injection, fractured per cycle (c/n=180)."""
    key = jax.random.PRNGKey(seed)
    x = x0
    for i in range(max_iter):
        key, subkey = jax.random.split(key)
        x = entropy_pgd_step(f, x, grad_fn, step_size, proj, entropy_kappa, subkey)
    return x
