import jax.numpy as jnp
import jax

def inject_entropy(x, sigma=0.01):
    noise = sigma * jax.random.normal(jax.random.PRNGKey(0), x.shape)
    return x + noise

def bounded_update(x, grad_fn, eta=0.01, sigma=0.01):
    grad = grad_fn(x)
    x_new = x - eta * grad
    return inject_entropy(x_new, sigma)
