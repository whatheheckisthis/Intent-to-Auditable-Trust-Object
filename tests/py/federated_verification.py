def federated_check(nodes_outputs, quorum):
    """
    Federated verification across nodes; each node validates peer outputs.
    """
    approvals = [nn_validation([out], validators=[lambda x: x==out], quorum_threshold=quorum)
                 for out in nodes_outputs]
    return approvals
