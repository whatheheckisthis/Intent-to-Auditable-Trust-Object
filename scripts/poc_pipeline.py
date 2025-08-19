import jax
from molecular_pgd import H_molecule
from lattice_pgd import H_lattice
from cyber_pgd import H_cyber
from pgd_entropy import PGD_iterate_lawful

def run_full_pipeline():
    # Molecular PGD
    x0_mol = jax.numpy.zeros((10,3))
    grad_fn_mol = jax.grad(lambda x: H_molecule(x))
    mol_seq, mol_audit = PGD_iterate_lawful(x0_mol, grad_fn_mol)

    # Lattice PGD
    phi0 = jax.numpy.zeros((8,8,8))
    grad_fn_phi = jax.grad(H_lattice)
    phi_seq, phi_audit = PGD_iterate_lawful(phi0, grad_fn_phi)

    # Cyber-kinetics PGD
    nodes0 = jax.numpy.zeros((5,3))
    grad_fn_cyber = jax.grad(H_cyber)
    nodes_seq, nodes_audit = PGD_iterate_lawful(nodes0, grad_fn_cyber)

    return {
        "molecular": (mol_seq, mol_audit),
        "lattice": (phi_seq, phi_audit),
        "cyber": (nodes_seq, nodes_audit)
    }
