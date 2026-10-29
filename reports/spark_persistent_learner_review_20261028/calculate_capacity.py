"""Recompute design arithmetic; never allocate an expanded bank or load a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/spark_persistent_learner_tiered_design_v1.json"
OUTPUT = Path(__file__).with_name("capacity_arithmetic.json")

def calculate():
    config = json.loads(CONFIG.read_text())
    rows = []
    for profile in config["capacity_profiles"]:
        n, d, h = (profile[k] for k in ("experts", "width", "hidden_width"))
        count = 3 * n * d * h
        assert profile["full_expert_parameters"] == count
        assert profile["fp32_weight_bytes"] == count * 4
        assert profile["all_expert_fp32_weight_gradient_adam_bytes"] == count * 16
        rows.append({
            "id": profile["id"], "full_parameters": count,
            "online_D_only_parameters": count // 3,
            "fp32_weights_MiB": count * 4 / 2**20,
            "all_expert_full_adam_state_MiB": count * 16 / 2**20,
            "all_expert_D_only_adam_plus_fixed_UG_MiB": count * 8 / 2**20,
            "one_active_full_expert_adam_MiB": 3 * d * h * 16 / 2**20,
            "one_active_D_adam_plus_fixed_UG_MiB": d * h * 24 / 2**20,
        })
    readout = config["readout"]
    # Only Q/K/V/O at two sites, without biases, norms, gates, codecs or other branches.
    reader_parameters = 2 * (2 * 2560 * readout["width"] + 2 * readout["width"] ** 2)
    per_record = (readout["ordered_slots_per_record"] * readout["width"] + readout["key_width"]) * 2
    return {
        "format": "replica.spark_tiered_capacity_arithmetic.v1",
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "measurement": False, "allocation_or_training": False,
        "formula": "bias-free gated expert: 3*N*d*h; FP32 W/G/Adam m/v: 16 bytes per parameter",
        "profiles": rows,
        "reader_pair_only": {"parameters": reader_parameters, "fp32_weight_MiB": reader_parameters * 4 / 2**20},
        "record_payload_only": {
            "bf16_bytes_per_record": per_record,
            "warm_target_MiB": per_record * config["tiers"]["warm"]["compact_record_target"] / 2**20,
            "cold_target_GiB": per_record * config["tiers"]["cold"]["compact_record_target"] / 2**30,
            "excluded": "canonical text, metadata, indices, journals, generations, allocator and staging copies",
        },
        "P64_to_current_parameter_ratio": rows[5]["full_parameters"] / rows[0]["full_parameters"],
        "excluded_from_expert_sizes": "base, native KV, activations, all spine/codec/router modules, nonexpert state and I/O copies",
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = calculate()
    if args.write:
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    else:
        assert json.loads(OUTPUT.read_text()) == result, "capacity receipt drift"
    print(json.dumps({"profiles": len(result["profiles"]), "verified": True,
                      "native_execution": False, "P64_parameter_ratio": result["P64_to_current_parameter_ratio"]}))
