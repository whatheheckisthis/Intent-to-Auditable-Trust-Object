def run_variational_experiment(x0, eta, T_steps, J, project):
    """
    Run a small-scale variational experiment and return final PGD iterate.
    """
    seq = PGD_iterate(x0, eta, T_steps, J, project)
    final_x = seq[-1]
    return final_x, seq
