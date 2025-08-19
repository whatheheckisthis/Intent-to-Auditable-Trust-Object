import jax.numpy as jnp

n_nodes = 5
adversary_strength = jnp.ones(n_nodes) * 0.1

def H_cyber(nodes):
    interaction = jnp.sum(jnp.square(nodes)) - jnp.sum(nodes * adversary_strength[:,None])
    return interaction
