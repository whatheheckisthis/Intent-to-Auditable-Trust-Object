import hashlib

def packet_commit(packet, previous_hash):
    """
    Append-only hash chain for packet verification.
    """
    m = hashlib.sha256()
    m.update(previous_hash.encode())
    m.update(str(packet).encode())
    return m.hexdigest()
