def chain_append(prev_hash, data):
    """
    Append a new entry to audit chain.
    """
    import hashlib
    m = hashlib.sha256()
    m.update(prev_hash.encode())
    m.update(str(data).encode())
    return m.hexdigest()
