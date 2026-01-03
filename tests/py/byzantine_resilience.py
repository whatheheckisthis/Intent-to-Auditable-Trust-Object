def byzantine_quorum(results, threshold):
    """
    Accept only if > threshold proportion of nodes agree.
    """
    count_true = jnp.sum(jnp.array(results))
    return count_true / len(results) >= threshold
