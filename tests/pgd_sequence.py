def PGD_iterate(x0, eta, T_steps, J, project):
    """
    Generate PGD iterates for T_steps.
    """
    x = x0
    seq = [x]
    for t in range(T_steps):
        x = project(x - eta * jax.grad(J)(x))
        seq.append(x)
    return seq
