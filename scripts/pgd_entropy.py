import jax

eta = 0.01

@jax.jit
def PGD_step(x, grad_fn):
    g = grad_fn(x)
    return x - eta * g

def PGD_iterate_lawful(x0, grad_fn, T=10):
    from utils_math import hash_commit
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
    return jax.numpy.array(x_seq), audit_log
