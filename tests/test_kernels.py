# tests/test_kernels.py
import jax.numpy as jnp
from pgd_entropy import entropy_pgd

def test_entropy_pgd_runs():
    f = lambda x: jnp.sum(x**2)
    grad_fn = lambda x: 2*x
    proj = lambda x: jnp.clip(x, -1, 1)
    x0 = jnp.array([0.5, -0.5])
    out = entropy_pgd(f, x0, grad_fn, proj)
    assert out.shape == x0.shape
