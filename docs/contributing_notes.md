# 18.X Kernel Setup and Run Guide:

This guide explains how to set up and run the **18.X Kernel** using **Python, JAX, and NVIDIA GPU acceleration** on a cloud-based or on-premises virtual machine. The instructions are written for both newcomers and technically proficient users who want a reproducible environment for lawful, auditable inference experiments.

---

## ✅ Setup Checklist

* [ ] **Provision a VM with NVIDIA GPU** (A100 / V100 / similar)
* [ ] **Install NVIDIA drivers and CUDA**

  * [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  * [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
* [ ] **Install Miniconda** and create environment `18x_kernel`

  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  conda create -n 18x_kernel python=3.10
  conda activate 18x_kernel
  ```
* [ ] **Install JAX with GPU support**

  ```bash
  pip install --upgrade pip
  pip install --upgrade "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
  ```
* [ ] **Install supporting Python libraries**

  ```bash
  pip install numpy scipy hashlib
  ```
* [ ] **Clone or create `run_18x_kernel.py`**

  ```bash
  git clone https://github.com/your-repo/18x_kernel.git
  cd 18x_kernel
  ```
* [ ] **Verify GPU detection**

  ```python
  import jax
  print(jax.devices())
  ```
* [ ] **Run the 18.X Kernel pipeline**

  ```bash
  python run_18x_kernel.py
  ```

---

## 1. Choose Your Virtual Machine

* **Cloud Options:**

  * [AWS EC2](https://aws.amazon.com/ec2/) – `p4d`, `g5` instances
  * [Google Cloud](https://cloud.google.com/) – A100 / V100
  * [Microsoft Azure](https://azure.microsoft.com/) – NC/ND-series

* **Recommended Specs:**

  Here’s a revised **Recommended Specs** section with links for each item:

---

## Recommended Specs

* **RAM:** 16–32 GB minimum

  * [Check your system RAM](https://www.howtogeek.com/202796/how-to-check-how-much-ram-you-have-on-windows-mac-or-linux/)

* **GPU:** NVIDIA GPU with CUDA support (A100, V100, or similar)

  * [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
  * [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  * [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

* **Operating System:** Ubuntu 20.04 LTS or later

  * [Download Ubuntu 20.04 LTS](https://releases.ubuntu.com/20.04/)
  * [Ubuntu Installation Guide](https://ubuntu.com/tutorials/install-ubuntu-desktop)

---



---

## 2. Guidance and Best Practices

1. **Environment Isolation:** Use a dedicated Conda environment to prevent conflicts with system Python or other projects.
2. **GPU Verification:** Always confirm `jax.devices()` lists your GPU before running heavy workloads.
3. **Incremental Runs:** Test each module (e.g., PGD optimization, explainability pipeline) individually before running the full pipeline.
4. **Logging & Audit:** Enable logging inside `run_18x_kernel.py` to capture each inference step for reproducibility.
5. **Version Pinning:** Pin library versions in a `requirements.txt` or `environment.yml` to ensure deterministic execution across VMs.

---


---

## Usage Requirements

Before running the 18.X Kernel or following the workflows below, ensure you have the following installed:

* [Python 3.10+](https://www.python.org/downloads/)
* [Miniconda / Conda](https://docs.conda.io/en/latest/miniconda.html)
* [Obsidian](https://obsidian.md/) – for note-taking, organizing references, and linking workflow artifacts.

---

## Key Resources and Useful Links

### Command-Line and Development Tools

* [tldr-pages](https://github.com/tldr-pages/tldr) – Simplified, community-driven man pages for common commands
* [Python Command-Line Chat GPT](https://github.com/davidtkeane/python-command-line-chat-gpt) – Lightweight CLI interface for interacting with ChatGPT
* [Microsoft Presidio](https://github.com/microsoft/presidio) – Privacy-preserving data detection and anonymization toolkit

### Security, Learning, and Reference

* [The Book of Secret Knowledge](https://github.com/trimstray/the-book-of-secret-knowledge) – Comprehensive hacking and security resource
* [Security Certification Roadmap](https://pauljerimy.com/security-certification-roadmap/) – Guidance on security certifications and career progression

### Microsoft & Cloud Resources

* [Microsoft Certifications](https://learn.microsoft.com/en-us/training/) – Official learning paths and certifications
* [Microsoft Azure for Students + \$100 Azure Credits](https://azure.microsoft.com/en-us/free/students/) – Free cloud credits and student access
* [Install Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) – Command-line interface for Azure

---

Here’s an updated **Recommendations** section with links and structured as a reference file layout for clarity:

---

### ✅ Recommendations

**File/Folder Structure Reference** *(example for maintaining reproducibility and note-taking)*:

```
18x_kernel/
├── docs/                  # Notes, references, Obsidian vault
│   ├── setup_notes.md
│   ├── module_explanations.md
│   └── audit_trails.md
├── src/                   # Kernel source files
│   ├── pgd_entropy.py
│   ├── jit_fracture.py
│   ├── integrated_pjit_with_processor.py
│   ├── explainability.py
│   └── poc_pipeline.py
├── run_18x_kernel.py       # Main execution script
├── environment.yml         # Conda environment definition
└── README.md
```

**Recommendations with Links:**

1. **Obsidian for Structured Notes**

   * Use [Obsidian](https://obsidian.md/) to maintain structured notes, link references, and capture reproducibility insights. Store your vault under `docs/` to integrate notes directly with the kernel code.

2. **Combine Command-Line Tools with Kernel**

   * Use [tldr-pages](https://github.com/tldr-pages/tldr) for quick command references.
   * Use [Python CLI ChatGPT](https://github.com/davidtkeane/python-command-line-chat-gpt) to interact with the kernel or generate code prompts on the fly.
   * Keep logs and experiment notes in `docs/audit_trails.md` to track reproducibility and debugging.

3. **Microsoft & Azure Learning Paths**

   * Follow [Microsoft Certifications](https://learn.microsoft.com/en-us/training/) to gain structured learning for cloud deployment.
   * Use [Azure for Students + \$100 Credits](https://azure.microsoft.com/en-us/free/students/) for GPU-enabled workloads.
   * Install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) for managing cloud resources via terminal commands.

**Tip:** Keep your `docs/` folder tightly linked to kernel modules, logging, and audit outputs. This ensures that experiments are traceable, reproducible, and aligned with lawful inference principles.

---



---

## Maintainability

This repository has been structured for **ease of maintenance, reproducibility, and pre-class readiness**. All environments, dependencies, and tools are explicitly defined so that users can start working with the kernel **without manual configuration at runtime**. Key maintainability features include:

1. **Predefined Environment**

   * A `environment.yml` file ensures that **Python version, JAX, CUDA, and all supporting libraries** (`numpy`, `scipy`, `hashlib`) are installed consistently.
   * Users can create the environment with:

     ```bash
     conda env create -f environment.yml
     conda activate 18x_kernel
     ```

2. **Dependency Pinning**

   * Library versions are fixed to avoid incompatibility across machines or cloud instances.
   * GPU-enabled libraries like JAX and CUDA are matched to recommended versions to guarantee reproducibility.

3. **Pre-Tested VM/Cloud Configuration**

   * The repository assumes **NVIDIA GPU (A100 / V100)** with proper CUDA drivers and Ubuntu 20.04+.
   * Scripts and instructions have been tested to ensure that all **modules run immediately once the environment is activated**.

4. **Structured Folder Layout**

   * `src/` contains all kernel modules.
   * `docs/` stores Obsidian notes, references, and audit trails.
   * `run_18x_kernel.py` orchestrates execution using the preconfigured environment.

5. **Ready for Class or Workshop Use**

   * Instructors or participants can **launch the kernel, run experiments, and access notes without additional setup**, ensuring a smooth, reproducible workflow.
   * Prebuilt environments minimize troubleshooting during sessions and maintain **consistency across multiple users or machines**.

**Summary:** This repository is designed to be **self-contained, auditable, and immediately usable**, making it ideal for academic instruction, workshops, or collaborative experiments where reproducibility and reliability are critical.








