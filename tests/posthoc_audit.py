def verify_audit(log_hashes, initial_hash):
    """
    Check that each hash matches chain commitments.
    """
    prev = initial_hash
    for h in log_hashes:
        recomputed = hashlib.sha256((prev + str(h)).encode()).hexdigest()
        if recomputed != h:
            return False
        prev = h
    return True
