#!/usr/bin/env python3
"""Run the selected-record literal-value and MiniCPM readout experiment.

The runner verifies prepared04 and the read-only native02 checkpoint in this
fresh process, freezes the development decisions before any label is parsed,
applies the literal endpoint, and only then loads MiniCPM once.  Heldout is
produced only after the development model endpoint passes.  Train is
descriptive and contains literal metrics only.

Parent commands
---------------
CPU contract replay (no model, no native job):
    .venv/bin/python -m pytest tests/test_memory_bank_value.py -q -m "not integration"

Optional real-artifact CPU replay (checkpoint binding + prepared04):
    .venv/bin/python -m pytest tests/test_memory_bank_value.py -q -m integration

Native generation under the capacity guard:
    REPLICA_CAPACITY_GUARD=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \\
    PYTORCH_MPS_FAST_MATH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/run_capacity_preflight.py \\
      --entry-point run_memory_bank_value.py \\
      --report runs/memory_bank_value_20261026/guard.json -- \\
      --device mps --config configs/memory_bank_value_v1.json \\
      --prepared data/memory_causal_fresh_20261026/prepared04 \\
      --output runs/memory_bank_value_20261026/native01

The parent owns adding ``run_memory_bank_value.py`` to the
``scripts/run_capacity_preflight.py`` ``--entry-point`` choices; this script
does not modify that guard.  Use a fresh output directory for every run.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text())


def _require_environment():
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise RuntimeError("Use the capacity guard")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def execute(args):
    _require_environment()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    if config.get("format") != "replica.memory_bank_value.config.v1":
        raise ValueError("Bank-value configuration format differs")
    prepared = args.prepared.resolve()
    protocol_path = prepared / "protocol.json"
    if _sha(protocol_path) != config["baseline"]["protocol_sha256"]:
        raise ValueError("Prepared protocol bytes differ from the pin")
    numeric = _read_json(protocol_path).get("numeric_policy", {})
    if numeric.get("torch_use_deterministic_algorithms") is not True:
        raise ValueError("Deterministic algorithms must be declared")
    if numeric.get("mps_fast_math") != "disabled":
        raise ValueError("MPS fast math must be declared disabled")
    if os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
        raise RuntimeError("PYTORCH_MPS_FAST_MATH=0 is required by the protocol")

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from replica_lab.architecture.loading import exclusive_mps_worker
    from replica_lab.memory_bank_value import (
        REPORT_FORMAT, build_model_generator, run_prepared_bank_value,
    )
    from replica_lab.profiling import RunProfiler

    torch.use_deterministic_algorithms(True)
    profiler = RunProfiler("mps", output)
    profiler.start()
    try:
        with exclusive_mps_worker(ROOT):
            report = run_prepared_bank_value(
                config, prepared=prepared, output=output,
                config_path=config_path,
                generation_factory=partial(
                    build_model_generator, config,
                    profiler=profiler, device="mps"),
                native=True, profiler=profiler, verify_source=True)
        if report.get("format") != REPORT_FORMAT:
            raise RuntimeError("Bank-value report format differs")
        if report["raw_outputs"]["verified_unchanged"] is not True:
            raise RuntimeError("Raw outputs changed after the first write")
        if report["resource_usage"]["model_loads"] > 1:
            raise RuntimeError("More than one model load occurred")
        if report["resource_usage"]["checkpoints_written"] != 0:
            raise RuntimeError("A checkpoint was written")
        print(json.dumps({
            "outcome": report["outcome"],
            "literal_dev_passed": report["gating"]["literal_dev"]["passed"],
            "model_dev": report["gating"]["model_dev"]["status"],
            "model_dev_passed": (
                None if report["gating"]["model_dev"].get("gate") is None
                else report["gating"]["model_dev"]["gate"]["passed"]),
            "heldout": report["gating"]["heldout"]["status"],
            "train": report["gating"]["train"]["status"],
            "report": str(output / "report.json"),
            "report_sha256": _sha(output / "report.json"),
            "decision_counts": report["counts"]["by_split"]
            if isinstance(report["counts"].get("by_split"), dict) else None,
            "resource_usage": report["resource_usage"],
        }, indent=2, sort_keys=True))
        return report
    except BaseException as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "profile": profiler.stop(),
        }
        (output / "failure_report.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/memory_bank_value_v1.json")
    parser.add_argument(
        "--prepared", type=Path,
        default=ROOT / "data/memory_causal_fresh_20261026/prepared04")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.device != "mps":
        parser.error("Native execution requires MPS")
    execute(args)


if __name__ == "__main__":
    main()
