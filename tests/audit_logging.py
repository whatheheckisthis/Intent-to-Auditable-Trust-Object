audit_log = []

def log_commit(commit_hash):
    audit_log.append(commit_hash)

def verify_commit_sequence():
    return all(audit_log[i] != audit_log[i-1] for i in range(1,len(audit_log)))
