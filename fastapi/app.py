# -*- coding: utf-8 -*-
"""
FastAPI wrapper for ops-utilities 18.X Kernel
"""

from fastapi import FastAPI, Query
from typing import Optional
import numpy as np
from pathlib import Path
import os
from src.kernel.entropy_pgd import pgd_optimize  # Your kernel function
from src.kernel.trust_objects import log_trust_object  # Trust-object logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=Path("config/.env"))

app = FastAPI(title="Ops-Utilities 18.X Kernel API", version="0.1.0")

@app.get("/")
def index():
    return {"message": "Ops-Utilities 18.X Kernel API is running"}

@app.get("/compute/quadratic")
def compute_quadratic(
    x0: float = Query(0.0, description="Initial guess for PGD optimization"),
    lr: float = Query(0.1, description="Learning rate"),
    steps: int = Query(25, description="Number of optimization steps")
):
    # Define simple quadratic function as an example
    def f(x):
        return (x - 2) ** 2

    # Run PGD optimization
    result = pgd_optimize(f, x0, {"learning_rate": lr, "num_steps": steps})

    # Log as a trust object
    log_trust_object({"input": x0, "lr": lr, "steps": steps, "result": result})

    return {"optimized_value": result}

@app.get("/welcome")
def welcome(name: Optional[str] = Query("whattheheckisthis", description="Your name")):
    return {"message": f"Welcome to the Ops-Utilities workspace, {name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)