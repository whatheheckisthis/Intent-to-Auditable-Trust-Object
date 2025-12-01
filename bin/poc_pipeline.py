# poc_pipeline.py
import jax.numpy as jnp
from pgd_entropy import entropy_pgd
from pjit_fracture import fractured_kernel
from explainability import shap_explain

def quadratic_model(x):
    return jnp.sum(x**2)

def main():
    f = quadratic_model
    grad_fn = lambda x: 2*x
    proj = lambda x: jnp.clip(x, -1, 1)
    
    # Run entropy PGD
    x0 = jnp.array([0.5, -0.5, 0.25])
    result = entropy_pgd(f, x0, grad_fn, proj)
    
    # Fracture cycle
    fractured = fractured_kernel(lambda x: f(x), result)
    
    # SHAP explain (c/n=180 cap)
    import numpy as np
    data = np.random.rand(200, 3)
    shap_result = shap_explain(lambda z: np.sum(z**2, axis=1), data)
    
    print("PGD result:", result)
    print("Fractured result:", fractured)
    print("SHAP summary:", shap_result)

if __name__ == "__main__":
    main()
