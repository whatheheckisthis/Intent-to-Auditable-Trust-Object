import shap
import lime
import jax.numpy as jnp

def cycle_explainability(model_fn, X_cycle, max_features=10):
    """
    Apply SHAP and LIME explanations per oscillatory cycle.
    """
    shap_values = shap.Explainer(model_fn, X_cycle)(X_cycle)
    lime_exp = [lime.lime_tabular.LimeTabularExplainer(X_cycle, mode="regression").explain_instance(x, model_fn, num_features=max_features)
                for x in X_cycle]
    return shap_values, lime_exp
