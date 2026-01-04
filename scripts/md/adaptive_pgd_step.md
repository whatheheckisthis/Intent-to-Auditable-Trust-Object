def PGD_step_adaptive(x_list, phi_list, S_list, a_list, eta_def, eta_adv):
    x_next = PGD_step_cyber(x_list, phi_list, S_list, eta_def)
    grads_a = [jax.grad(lambda a: H_cyber_adaptive(x_next, phi_list, S_list, [a]+a_list[1:]))(a) for a in a_list]
    a_next = [jnp.clip(a + eta_adv*grad, 0,1) for a, grad in zip(a_list, grads_a)]
    return x_next, a_next
