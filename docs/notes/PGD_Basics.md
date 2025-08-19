```bash

x = jnp.array([10.0, -5.0, 2.0])
for i in range(5):
    x = PGD_step(x)
    print("Step", i+1, ":", x)
