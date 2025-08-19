def H_cyber_kinetics(x_list, phi_list, S_list):
    H_total = 0.0
    for x, phi, S in zip(x_list, phi_list, S_list):
        risk = x[0]*0.5 + 0.3*jnp.sum(S**2) - 0.4*jnp.sum(phi)
        H_total += risk
    for i in range(len(x_list)-1):
        H_total += 0.1 * jnp.sum((x_list[i]-x_list[i+1])**2)
    return H_total
