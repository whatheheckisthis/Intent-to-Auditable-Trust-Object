def hydrogen_ansatz_energy(r, alpha):
    """
    Simple trial wavefunction ψ(r) ~ e^{-α r}, compute <ψ|H|ψ>
    """
    import jax.numpy as jnp
    psi = jnp.exp(-alpha * r)
    kinetic = -0.5 * jnp.gradient(jnp.gradient(psi))
    potential = -1.0 / r * psi
    return jnp.vdot(psi, kinetic + potential)
