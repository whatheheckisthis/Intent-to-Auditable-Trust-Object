# 18.X Kernel Setup Guide  
**Run JAX with NVIDIA GPU Acceleration on a Virtual Machine**

This guide explains how to set up and run the **18.X Kernel** using **Python, JAX, and NVIDIA GPU acceleration** on a cloud-based or on-premises virtual machine.

---

## ✅ Setup Checklist
- [ ] Provision a VM with NVIDIA GPU (A100 / V100 / similar)  
- [ ] Install NVIDIA drivers and CUDA  
- [ ] Install Miniconda & create environment (`18x_kernel`)  
- [ ] Install JAX with GPU support  
- [ ] Install supporting libraries (`numpy`, `scipy`, `hashlib`)  
- [ ] Clone or create `run_18x_kernel.py`  
- [ ] Verify GPU detection with `jax.devices()`  
- [ ] Run the 18.X Kernel pipeline  

---

## 1. Choose Your Virtual Machine
- **Cloud Options:**  
  - AWS EC2: `p4d`, `g5`  
  - Google Cloud: A100 / V100  
  - Azure: NC/ND-series  

- **Recommended Specs:**  
  - 16–32 GB RAM minimum  
  - NVIDIA GPU with CUDA support (A100, V100, or similar)  
  - Ubuntu 20.04+  

