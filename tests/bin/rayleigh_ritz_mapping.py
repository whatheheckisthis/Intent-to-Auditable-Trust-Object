def variational_bound(wavefunction, hamiltonian):
    """
    Compute ⟨ψ|H|ψ⟩ for variational ansatz
    """
    psi_vec = wavefunction()
    H_psi = hamiltonian(psi_vec)
    return jnp.vdot(psi_vec, H_psi).real
