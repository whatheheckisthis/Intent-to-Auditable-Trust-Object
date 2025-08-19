# Example: plug PGD + validation + audit workflow into Copilot notebooks
x0 = jnp.zeros((n,))
eta = 0.01
x_seq = PGD_iterate(x0, eta, T_steps=100)
audit_hashes = [packet_commit(x, "") for x in x_seq]
