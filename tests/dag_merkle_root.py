# dag_merkle_root.py
# Computes a Merkle root hash from a list of inference events (DAG leaves)

import hashlib

def hash_leaf(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def compute_merkle_root(leaves: list) -> str:
    if not leaves:
        return None
    hashes = [hash_leaf(str(leaf)) for leaf in leaves]

    while len(hashes) > 1:
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1])  # duplicate last if odd
        new_level = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i+1]
            new_level.append(hash_leaf(combined))
        hashes = new_level
    return hashes[0]

# Example usage
if __name__ == "__main__":
    dag_events = [
        {"stage": "input", "data": "John Doe"},
        {"stage": "vector", "data": [0.12, 0.34]},
        {"stage": "inference", "result": "match"}
    ]
    root = compute_merkle_root(dag_events)
    print("Merkle Root:", root)