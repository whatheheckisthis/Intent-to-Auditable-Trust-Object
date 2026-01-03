federated_log = []

def log_node_commit(node_id, commit_hash):
    federated_log.append((node_id, commit_hash))

def verify_federated_sequence():
    return all(federated_log[i][1] != federated_log[i-1][1] for i in range(1,len(federated_log)))
