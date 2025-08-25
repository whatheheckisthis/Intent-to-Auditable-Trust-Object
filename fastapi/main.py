# -*- coding: utf-8 -*-
"""
FastAPI wrapper for ops-utilities kernel
"""

import uvicorn
from fastapi import FastAPI, Query
from typing import Optional
from pathlib import Path
import subprocess
import json

# Import your kernel functions
from src.kernel.pgd_entropy import pgd_optimize  # Example kernel import
# from scripts.trust_objects import log_trust_object  # Example

app = FastAPI(title="Ops-Utilities Kernel API", version="0.1.0")

# Root endpoint
@app.get("/")
def index():
    return {"message": "Ops-Utilities Kernel API Running"}

# Example endpoint for PGD optimization
@app.get("/pgd")
def run_pgd(x0: float = Query(0.0), lr: float = Query(0.1), steps: int = Query(25)):
    # Run kernel function
    result = pgd_optimize(lambda x: (x - 2) ** 2, x0, {"learning_rate": lr, "num_steps": steps})
    
    # Optional: log as trust object
    # log_trust_object({"x0": x0, "lr": lr, "steps": steps, "result": result})
    
    return {"x0": x0, "learning_rate": lr, "num_steps": steps, "result": result}

# Example endpoint for shell/utility scripts
@app.get("/run-script")
def run_script(script_name: str):
    script_path = Path("scripts") / script_name
    if not script_path.exists():
        return {"error": "Script not found"}
    
    output = subprocess.getoutput(f"python {script_path}")
    return {"script": script_name, "output": output}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)