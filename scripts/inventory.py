#!/usr/bin/env python3
"""Read-only local inventory. No driver changes, model loads, or device selection."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def run(args):
    if not shutil.which(args[0]):
        return {"available": False}
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=45)
        return {"available": True, "returncode": r.returncode,
                "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"available": True, "timeout": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    packages = {}
    for name in ("torch", "transformers", "safetensors", "numpy", "pytest", "jsonschema",
                 "accelerate", "deepspeed", "mlx", "mlx-lm", "mlx-vlm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"platform": platform.platform(), "machine": platform.machine(),
              "python": sys.version, "packages": packages,
              "nvidia": run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
                             "--format=csv,noheader"]),
              "topology": run(["nvidia-smi", "topo", "-m"]),
              "training_performed": False}
    if Path("/proc/meminfo").exists():
        result["meminfo"] = {k.strip(): v.strip() for k, v in
                             (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    if sys.platform == "darwin":
        result["chip"] = run(["sysctl", "-n", "machdep.cpu.brand_string"])
        result["cpu_cores"] = run(["sysctl", "-n", "hw.ncpu"])
        result["unified_memory_bytes"] = run(["sysctl", "-n", "hw.memsize"])
        displays = run(["system_profiler", "SPDisplaysDataType", "-json"])
        # Retain accelerator properties without monitor serials or display IDs.
        if displays.get("returncode") == 0:
            result["apple_gpus"] = [
                {key: item.get(key) for key in ("sppci_model", "sppci_cores", "spdisplays_mtlgpufamilysupport")}
                for item in json.loads(displays["stdout"]).get("SPDisplaysDataType", [])
            ]
        else:
            result["apple_gpus"] = displays
        result["memory_pressure"] = run(["memory_pressure", "-Q"])
        result["swap"] = run(["sysctl", "vm.swapusage"])
    try:
        import torch
        result["torch_backends"] = {
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda,
            "cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        }
        if torch.cuda.is_available():
            result["cuda_devices"] = [
                {"index": i, "name": torch.cuda.get_device_name(i),
                 "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                 "compute_capability": list(torch.cuda.get_device_capability(i))}
                for i in range(torch.cuda.device_count())
            ]
            result["torch_backends"]["native_bfloat16_current_device"] = torch.cuda.is_bf16_supported(including_emulation=False)
    except ImportError:
        result["torch_backends"] = {"status": "torch_not_installed"}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
