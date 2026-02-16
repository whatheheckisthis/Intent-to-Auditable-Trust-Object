# explainability.py
import shap
import lime.lime_tabular
import numpy as np

def shap_explain(model, data, max_samples=180):
    explainer = shap.Explainer(model, data[:max_samples])
    return explainer(data[:max_samples])

def lime_explain(model, data, max_samples=180):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        data[:max_samples], feature_names=[f"f{i}" for i in range(data.shape[1])],
        verbose=True, mode="regression"
    )
    return [explainer.explain_instance(d, model.predict, num_features=10)
            for d in data[:max_samples]]
