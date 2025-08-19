import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

def explain_cycle(model, X, method='shap'):
    if method == 'shap':
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        return shap_values
    elif method == 'lime':
        explainer = LimeTabularExplainer(
            X, mode="regression", discretize_continuous=True
        )
        explanations = [explainer.explain_instance(x, model.predict) for x in X]
        return explanations
    else:
        raise ValueError("Method must be 'shap' or 'lime'")
