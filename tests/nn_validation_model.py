import jax.numpy as jnp

def nn_validation(inputs, validators, quorum_threshold):
    """
    n–n validation chain.
    - inputs: list of proposed inference outputs
    - validators: list of validator functions
    - quorum_threshold: min number of agreeing validators
    """
    results = [[v(x) for v in validators] for x in inputs]
    accepted = []
    for res in results:
        count_agree = sum(res)
        accepted.append(count_agree >= quorum_threshold)
    return accepted
