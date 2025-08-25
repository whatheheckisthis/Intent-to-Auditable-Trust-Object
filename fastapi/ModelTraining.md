# 18.X Kernel: Multi-Cycle PGD Training with Trust-Object Logging and Explainability

This document combines **environment setup** and **Python training code** into one script. It can be executed in a Jupyter Notebook or via CLI.

---

## 1. Environment Setup & Execution (Bash + Python)

```bash
# Step 1: Create directories
mkdir -p ./data ./checkpoints ./logs

# Step 2: Optional: virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 3: Install required packages
pip install jax jaxlib numpy pandas matplotlib shap lime pickle5

# Step 4: Run the Python training script
python - <<'END_PYTHON'
# =========================
# Python Imports & Initialization
# =========================
import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.experimental import mesh_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Callable
import shap
from lime.lime_tabular import LimeTabularExplainer
import pickle

# Directories
ROOT_DIR = Path("./")
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"

for d in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "training.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# =========================
# Synthetic Dataset
# =========================
def generate_synthetic_data(n_samples=1000, n_features=10, key=random.PRNGKey(0)):
    X = random.normal(key, (n_samples, n_features))
    weights = jnp.arange(1, n_features + 1)
    y = jnp.dot(X, weights) + random.normal(key, (n_samples,))
    return np.array(X), np.array(y)

X, y = generate_synthetic_data()
print(f"Dataset Shape: {X.shape}, {y.shape}")

# =========================
# PGD Optimizer
# =========================
def pgd_step(params: jnp.ndarray, grad_fn: Callable, lr: float = 0.01, clip_val: float = 1.0):
    grads = grad_fn(params)
    updated = params - lr * grads
    return jnp.clip(updated, -clip_val, clip_val)

def quadratic_loss(params, X_batch, y_batch):
    preds = X_batch @ params
    return jnp.mean((preds - y_batch) ** 2)

grad_fn = jit(grad(quadratic_loss))

# =========================
# Multi-Device Setup
# =========================
devices = jax.devices()
n_devices = len(devices)
mesh = mesh_utils.create_device_mesh((n_devices,))
print("Available devices:", devices)
print("Device mesh created:", mesh)

# =========================
# Kernel State Initialization
# =========================
n_features = X.shape[1]
key = random.PRNGKey(42)
params = random.normal(key, (n_features,))
trust_objects = []

num_epochs = 50
batch_size = 32
learning_rate = 0.05
entropy_scale = 0.02

def batch_indices(n_samples, batch_size):
    return [slice(i, i+batch_size) for i in range(0, n_samples, batch_size)]

# =========================
# Training Loop
# =========================
for epoch in range(num_epochs):
    for idx in batch_indices(X.shape[0], batch_size):
        X_batch, y_batch = X[idx], y[idx]
        noise = entropy_scale * random.normal(key, params.shape)
        params = pgd_step(params + noise, lambda p: grad_fn(p, X_batch, y_batch), lr=learning_rate)
    trust_objects.append(params.copy())
    logging.info(f"Epoch {epoch} completed | Params sample: {params[:3]}")

# =========================
# Convergence Visualization
# =========================
params_array = np.array(trust_objects)
plt.figure(figsize=(10,6))
for i in range(params_array.shape[1]):
    plt.plot(params_array[:, i], label=f"Param {i+1}")
plt.title("Parameter Convergence Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Parameter Value")
plt.legend()
plt.show()

# =========================
# Explainability: SHAP & LIME
# =========================
explainer_shap = shap.Explainer(lambda X_: X_ @ params, X)
shap_values = explainer_shap(X[:100])
shap.summary_plot(shap_values, X[:100])

explainer_lime = LimeTabularExplainer(X, mode='regression')
exp = explainer_lime.explain_instance(X[0], lambda x: x @ params)
exp.show_in_notebook(show_table=True)

# =========================
# Checkpoint & Audit
# =========================
checkpoint_file = CHECKPOINT_DIR / "params_checkpoint.pkl"
with open(checkpoint_file, "wb") as f:
    pickle.dump(params, f)
logging.info(f"Checkpoint saved at {checkpoint_file}")

audit_file = LOG_DIR / "trust_objects_audit.csv"
pd.DataFrame(params_array).to_csv(audit_file, index=False)
logging.info(f"Audit report saved at {audit_file}")

print("Training completed. Final parameters:", params)
print("Trust objects logged:", len(trust_objects))
END_PYTHON