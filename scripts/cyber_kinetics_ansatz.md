n_nodes = 5
x_nodes = [jnp.random.uniform(0,1,(3,)) for _ in range(n_nodes)]
phi_nodes = [jnp.random.uniform(0,1,(2,)) for _ in range(n_nodes)]
S_nodes = [jnp.random.normal(0,0.1,(3,)) for _ in range(n_nodes)]
