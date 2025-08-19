import jax
import jax.numpy as jnp
from utils_math import laplacian_3d, hash_commit
from itertools import permutations, combinations

# Multi-electron molecular system example
n_nuclei = 3
R_nuclei = jnp.array([[0.0,0.0,0.0],
                      [0.0,0.0,0.957],
                      [0.926,0.0,-0.239]])
n_electrons = 10
det_coeffs = jnp.array([1.0, 0.5])
beta = 0.5
dx = 0.01

def psi_molecule(positions):
    det_total = 0.0
    for coeff, perm in zip(det_coeffs, permutations(range(n_electrons))):
        phi = jnp.array([jnp.exp(-jnp.sum((positions[perm[i]]-R_nuclei[i%n_nuclei])**2))
                         for i in range(n_electrons)])
        corr = jnp.prod(jnp.exp(-beta * jnp.linalg.norm(positions[i]-positions[j]))
                        for i,j in combinations(range(n_electrons),2))
        det_total += coeff * jnp.prod(phi) * corr
    return det_total

def H_molecule(positions):
    psi_val = psi_molecule(positions)
    kinetic = -0.5 * jnp.sum(psi_val * laplacian_3d(psi_val, dx))
    V_nuc = -jnp.sum(psi_val**2 / (jnp.linalg.norm(positions-R_nuclei[:,None], axis=-1)+1e-12))
    V_ee = jnp.sum(psi_val**2 / (jnp.linalg.norm(positions-positions, axis=-1)+1e-12))
    return kinetic + V_nuc + V_ee
