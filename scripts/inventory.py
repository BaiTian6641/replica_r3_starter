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
    for name in ("torch", "transformers", "accelerate", "deepspeed", "mlx", "mlx-lm", "mlx-vlm"):
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
        result["unified_memory_bytes"] = run(["sysctl", "-n", "hw.memsize"])
        result["display_devices"] = run(["system_profiler", "SPDisplaysDataType", "-json"])
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
