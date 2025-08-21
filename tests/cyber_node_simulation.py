def PGD_step_cyber(x_nodes, phi_nodes, S_nodes, eta):
    """
    Step function for cyber-kinetic nodes in PGD.
    """
    next_x = [x - eta * grad(H_total)(x, phi) for x, phi in zip(x_nodes, phi_nodes)]
    return next_x
