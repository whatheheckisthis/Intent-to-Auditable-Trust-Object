def PGD_step_cyber(x_list, phi_list, S_list, eta):
    grads_x = [jax.grad(lambda xx: H_cyber_kinetics([xx]+x_list[1:], [phi]+phi_list[1:], [S]+S_list[1:]))(x)
               for x, phi, S in zip(x_list, phi_list, S_list)]
    x_next = [jnp.clip(x - eta*grad, 0,1) for x, grad in zip(x_list, grads_x)]
    return x_next
