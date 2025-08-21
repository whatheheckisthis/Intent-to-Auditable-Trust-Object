import jax

def grad(f):
    """
    Return JAX gradient of function f
    """
    return jax.grad(f)
