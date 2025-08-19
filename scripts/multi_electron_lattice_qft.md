# x = electron parameters, ψ_x = wavefunction
# Lattice field φ with discrete nodes
# PGD updates respect lawful constraints

for t in range(T):
    grad_electron = jax.grad(H_total_electron)(x)
    grad_lattice = jax.grad(H_total_lattice)(φ)
    x = project_K(x - η * grad_electron)
    φ = project_K(φ - η * grad_lattice)
