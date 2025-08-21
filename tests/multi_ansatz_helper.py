def combine_ansatz_states(ansatz_list):
    """
    Combine multiple electron ansätze into a single tensor for PGD updates.
    """
    return jnp.stack(ansatz_list)
