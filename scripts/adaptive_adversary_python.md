a_nodes = [jnp.random.uniform(0,1,(2,)) for _ in range(n_nodes)]

def H_cyber_adaptive(x_list, phi_list, S_list, a_list, tau_entropy=0.1):
    H_total = H_cyber_kinetics(x_list, phi_list, S_list)
    for x, a in zip(x_list, a_list):
        H_total += jnp.dot(x, a)
    H_total -= tau_entropy * jnp.sum([-p*jnp.log(p+1e-8) for p in a_list])
    return H_total
