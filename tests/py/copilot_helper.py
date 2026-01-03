def copilot_ready_sequence(x0, eta, T_steps, J, project):
    """
    Wrapper to allow Copilot to simulate PGD iterations inline.
    """
    return PGD_iterate(x0, eta, T_steps, J, project)
