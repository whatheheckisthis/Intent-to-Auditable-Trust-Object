from flask import Flask
import subprocess

app = Flask(__name__)

steps = {
    1: "echo 'Provision VM (manual step)'",
    2: "sudo apt update && sudo apt install -y nvidia-driver-535 nvidia-utils-535 cuda",
    3: "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh -b -p $HOME/miniconda",
    4: "$HOME/miniconda/bin/conda create -n 18x_kernel python=3.11 -y",
    5: "$HOME/miniconda/bin/pip install jax[cuda11_cudnn86] -f https://storage.googleapis.com/jax-releases/jax_releases.html",
    6: "$HOME/miniconda/bin/pip install numpy scipy itertools more-itertools hashlib",
    7: "echo 'print(\"18.X Kernel Ready\")' > run_18x_kernel.py",
    8: "$HOME/miniconda/bin/python -c 'import jax; print(jax.devices())'",
    9: "$HOME/miniconda/bin/python run_18x_kernel.py"
}

@app.post("/run_step/<int:step>")
def run_step(step):
    cmd = steps.get(step)
    if cmd:
        subprocess.run(cmd, shell=True, check=False)
    return {"status": "ok"}
