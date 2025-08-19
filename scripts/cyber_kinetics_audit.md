def PGD_iterate_cyber(x0_list, phi0_list, S0_list, eta, T_steps):
    x_seq = [x0_list]; audit_log = []; prev_hash=""
    for t in range(T_steps):
        x_next = PGD_step_cyber(x_seq[-1], phi0_list, S0_list, eta)
        h = hash_commit(flatten(x_next), prev_hash)
        audit_log.append(h)
        prev_hash = h
        x_seq.append(x_next)
    return x_seq, audit_log
