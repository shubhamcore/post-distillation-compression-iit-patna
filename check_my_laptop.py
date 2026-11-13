#!/usr/bin/env python3
"""ACTUAL HARDWARE CHECK - run on the STUDENT'S LAPTOP (not sandbox).
Usage:  python src/check_my_laptop.py
Outputs JSON on screen + hardware_report.json in current folder. Send both back.
Checks: OS, CPU, RAM, disk, GPU (nvidia-smi / torch), Python & PyTorch feasibility.
"""
import platform, sys, shutil, os, json, subprocess, importlib.util

r = {"OS": platform.platform(), "system": platform.system(), "machine": platform.machine(),
     "python_version": sys.version, "python_executable": sys.executable,
     "cpu_logical_cores": os.cpu_count(), "disk_free_GB": round(shutil.disk_usage(".").free/1e9, 2),
     "torch_installed": importlib.util.find_spec("torch") is not None,
     "sklearn_installed": importlib.util.find_spec("sklearn") is not None,
     "numpy_installed": importlib.util.find_spec("numpy") is not None}
# RAM
try:
    if platform.system() == "Linux":
        r["RAM"] = [l.strip() for l in open("/proc/meminfo") if "MemTotal" in l]
    elif platform.system() == "Darwin":
        r["RAM_bytes"] = subprocess.getoutput("sysctl -n hw.memsize")
    elif platform.system() == "Windows":
        r["RAM_bytes"] = subprocess.getoutput("wmic computersystem get totalphysicalmemory")
except Exception as e: r["RAM_error"] = str(e)
# CPU model
try:
    if platform.system() == "Linux":
        r["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if "model name" in l][:1]
    elif platform.system() == "Windows":
        r["cpu_model"] = subprocess.getoutput("wmic cpu get name")
except Exception: pass
# GPU
try:
    out = subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    r["nvidia_smi"] = out if "not found" not in out.lower() and "no devices" not in out.lower() else "not available"
except Exception as e: r["nvidia_smi"] = f"error: {e}"
if r["torch_installed"]:
    import torch
    r["torch_version"] = torch.__version__; r["cuda_available"] = torch.cuda.is_available()
    r["torch_threads"] = torch.get_num_threads()
r["pytorch_feasibility"] = ("PyTorch already installed." if r["torch_installed"] else
  "Not installed. Feasible if: Python 3.9-3.13, >=2GB free disk, >=4GB RAM. "
  "Install (CPU): pip install torch --index-url https://download.pytorch.org/whl/cpu  -- DO NOT install until mentor approves.")
print(json.dumps(r, indent=1))
json.dump(r, open("hardware_report.json", "w"), indent=1)
print("\nSaved: hardware_report.json  -> send this file + terminal output to mentor.")
