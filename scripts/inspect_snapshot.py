#!/usr/bin/env python3
"""Inventory sealed MiniCPM5 tensor shapes without materializing model tensors."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--model-lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new report path")
    from safetensors import safe_open
    from replica_lab.model_adapter import validate_model_lock
    lock = json.loads(args.model_lock.read_text())
    validate_model_lock(args.model_path, lock)
    tensors = {}
    muon_parameters = adamw_parameters = scale_blocks = 0
    for path in sorted(args.model_path.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensors:
                    raise ValueError("Duplicate tensor across checkpoint shards")
                tensor = handle.get_slice(name)
                shape = tensor.get_shape()
                count = math.prod(shape)
                muon = len(shape) == 2 and name not in {"model.embed_tokens.weight", "lm_head.weight"}
                tensors[name] = {"shape": shape, "stored_dtype": str(tensor.get_dtype()),
                                 "parameters": count, "planned_optimizer": "muon8bit" if muon else "adamw"}
                if muon:
                    muon_parameters += count
                    scale_blocks += (count + 2047) // 2048
                else:
                    adamw_parameters += count
    total = muon_parameters + adamw_parameters
    if total != 2516756480:
        raise ValueError(f"Unexpected original parameter inventory: {total}")
    report = {"status": "passed", "inspected_at": datetime.now(timezone.utc).isoformat(),
              "revision": lock["revision"], "original_parameters": total,
              "planned_muon_parameters": muon_parameters, "planned_adamw_parameters": adamw_parameters,
              "block_size": 2048,
              "fp32_weight_gradient_optimizer_persistent_bytes": 9 * muon_parameters + 16 * adamw_parameters + 4 * scale_blocks,
              "memory_estimate_excludes": ["new modules", "activations", "temporary optimizer buffers", "allocator overhead"],
              "scope": "Sealed tensor-header inventory only; no model forward/backward or fit measurement",
              "tensors": tensors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tensors"}, indent=2))


if __name__ == "__main__":
    main()
