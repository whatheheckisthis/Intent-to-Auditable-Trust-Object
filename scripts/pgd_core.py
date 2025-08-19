import jax
import jax.numpy as jnp
import hashlib

eta = 0.01  # step size
T = 10      # iterations

def laplacian_3d(f, dx):
    lap = jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) + \
          jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) + \
          jnp.roll(f, 1, axis=2) + jnp.roll(f, -1, axis=2) - 6*f
    return lap / dx**2

def hash_commit(x, prev_hash):
    return hashlib.sha256((prev_hash + str(x)).encode()).hexdigest()

@jax.jit
def PGD_step(x, grad_fn):
    g = grad_fn(x)
    return x - eta * g

def PGD_iterate(x0, grad_fn, T=T):
    x_seq = [x0]
    audit_log = []
    prev_hash = ""
    x = x0
    for t in range(T):
        x = PGD_step(x, grad_fn)
        h = hash_commit(x, prev_hash)
        audit_log.append(h)
        prev_hash = h
        x_seq.append(x)
    return jnp.array(x_seq), audit_log
