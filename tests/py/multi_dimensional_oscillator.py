def multi_d_oscillator_energy(x, omega_list):
    """
    Energy = 1/2 Σ m ω_i² x_i²
    """
    return 0.5 * jnp.sum(jnp.array(omega_list) * x**2)
