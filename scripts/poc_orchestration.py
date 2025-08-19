from pgd_core import PGD_iterate
from explainability_proto import explain_cycle
from entropy_injection import bounded_update
import jax.numpy as jnp

def run_poc(x0, grad_fn, model=None, cycles=10):
    x_seq, audit = PGD_iterate(x0, grad_fn, T=cycles)
    if model is not None:
        explanations = []
        for x in x_seq:
            explanations.append(explain_cycle(model, x))
    else:
        explanations = None
    return x_seq, audit, explanations
