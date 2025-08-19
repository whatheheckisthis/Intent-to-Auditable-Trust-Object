import jax
import jax.numpy as jnp

def J(x): return f(x) - tau*H(x) + Phi(x)

def PGD_step(x, eta):
    return project_K(x - eta * jax.grad(J)(x))

def PGD_iterate(x0, eta, T_steps):
    x_seq = [x0]
    for t in range(T_steps):
        x_next = PGD_step(x_seq[-1], eta)
        x_seq.append(x_next)
    return x_seq
