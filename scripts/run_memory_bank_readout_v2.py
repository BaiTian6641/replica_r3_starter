#!/usr/bin/env python3
"""Run the predeclared prompt-v2 selected-record readout jobs.

Two fresh-process native jobs are declared and must run in order:

1. ``--job first``: verify prepared04 and native02, freeze dev decisions,
   apply the literal gate, load MiniCPM once and evaluate the first-process
   phase.  A failing literal gate touches no model or tokenizer.
2. ``--job restart``: additionally verify the restart02 checkpoint evidence and
   the first-job receipt, reapply the literal gate, load MiniCPM once, evaluate
   the restart and revocation phases, and only then open the role-sealed heldout
   labels.  Missing or failing first-process evidence stops before the model.

Parent commands
---------------
Role sealing (preparation only, no model):
    .venv/bin/python scripts/seal_memory_bank_roles.py \\
      --prepared data/memory_causal_fresh_20261026/prepared04 \\
      --output data/memory_causal_fresh_20261026/prepared04_roles

CPU contract replay:
    .venv/bin/python -m pytest tests/test_memory_bank_readout_v2.py -q -m "not integration"

First process:
    REPLICA_CAPACITY_GUARD=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \\
    PYTORCH_MPS_FAST_MATH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/run_capacity_preflight.py \\
      --entry-point run_memory_bank_readout_v2.py \\
      --report runs/memory_bank_readout_20261026/guard_first.json -- \\
      --job first --device mps --config configs/memory_bank_readout_v2.json \\
      --prepared data/memory_causal_fresh_20261026/prepared04 \\
      --roles data/memory_causal_fresh_20261026/prepared04_roles \\
      --output runs/memory_bank_readout_20261026/first

Restart (only after the first-process gate passes):
    REPLICA_CAPACITY_GUARD=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \\
    PYTORCH_MPS_FAST_MATH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/run_capacity_preflight.py \\
      --entry-point run_memory_bank_readout_v2.py \\
      --report runs/memory_bank_readout_20261026/guard_restart.json -- \\
      --job restart --device mps --config configs/memory_bank_readout_v2.json \\
      --prepared data/memory_causal_fresh_20261026/prepared04 \\
      --roles data/memory_causal_fresh_20261026/prepared04_roles \\
      --first-output runs/memory_bank_readout_20261026/first \\
      --output runs/memory_bank_readout_20261026/restart

The parent owns adding ``run_memory_bank_readout_v2.py`` to the
``scripts/run_capacity_preflight.py`` ``--entry-point`` choices; this script
does not modify that guard.  Use a fresh output directory for every job.
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
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def execute(args):
    _require_environment()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    if config.get("format") != "replica.memory_bank_readout.config.v2":
        raise ValueError("Readout v2 configuration format differs")
    prepared = args.prepared.resolve()
    roles = args.roles.resolve()
    protocol_path = prepared / "protocol.json"
    if _sha(protocol_path) != config["baseline"]["first_process"][
            "protocol_sha256"]:
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
    if not (roles / "manifest.json").is_file():
        raise ValueError("Seal the role label files first")
    if args.job == "restart" and args.first_output is None:
        raise ValueError("--first-output is required for the restart job")
    if args.job == "first" and args.first_output is not None:
        raise ValueError("--first-output is only valid for the restart job")

    import torch
    from replica_lab.architecture.loading import exclusive_mps_worker
    from replica_lab.memory_bank_readout_v2 import (
        REPORT_FORMAT, build_model_generator_v2, run_readout_job,
    )
    from replica_lab.profiling import RunProfiler

    torch.use_deterministic_algorithms(True)
    profiler = RunProfiler("mps", output)
    profiler.start()
    try:
        with exclusive_mps_worker(ROOT):
            report = run_readout_job(
                config, job=args.job, output=output,
                prepared=prepared, roles=roles,
                first_output=(args.first_output.resolve()
                              if args.first_output is not None else None),
                config_path=config_path,
                generation_factory=partial(
                    build_model_generator_v2, config,
                    profiler=profiler, device="mps"),
                native=True, profiler=profiler, verify_source=True)
        if report.get("format") != REPORT_FORMAT:
            raise RuntimeError("Readout v2 report format differs")
        if report["raw_outputs"]["verified_unchanged"] is not True:
            raise RuntimeError("Raw outputs changed after the first write")
        if report["resource_usage"]["model_loads"] > 1:
            raise RuntimeError("More than one model load occurred")
        if report["resource_usage"]["checkpoints_written"] != 0:
            raise RuntimeError("A checkpoint was written")
        if report["classification"]["structured_decoding"] is not False:
            raise RuntimeError("Structured decoding must not be used")
        if report["role_label_access"]["heldout_opened"] and (
                report["gating"]["heldout"]["status"] != "evaluated"):
            raise RuntimeError("Heldout labels were opened without evidence")
        print(json.dumps({
            "job": args.job,
            "outcome": report["outcome"],
            "literal_dev_passed": report["gating"]["literal_dev"]["passed"],
            "first_process": (
                None if report["gating"]["first_process"].get("gate") is None
                else report["gating"]["first_process"]["gate"]["passed"]),
            "restart": (
                None if report["gating"]["restart"].get("gate") is None
                else report["gating"]["restart"]["gate"]["passed"]),
            "revocation": (
                None if report["gating"]["revocation"].get("gate") is None
                else report["gating"]["revocation"]["gate"]["passed"]),
            "heldout": report["gating"]["heldout"]["status"],
            "train": report["gating"]["train"]["status"],
            "role_labels_opened": report["role_label_access"]["opened"],
            "report": str(output / "report.json"),
            "report_sha256": _sha(output / "report.json"),
            "generation_counts": report["counts"].get("generations"),
            "resource_usage": report["resource_usage"],
        }, indent=2, sort_keys=True))
        return report
    except BaseException as error:
        failure = {
            "status": "failed",
            "job": args.job,
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
    parser.add_argument("--job", choices=("first", "restart"), required=True)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/memory_bank_readout_v2.json")
    parser.add_argument(
        "--prepared", type=Path,
        default=ROOT / "data/memory_causal_fresh_20261026/prepared04")
    parser.add_argument(
        "--roles", type=Path,
        default=ROOT / "data/memory_causal_fresh_20261026/prepared04_roles")
    parser.add_argument("--first-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.device != "mps":
        parser.error("Native execution requires MPS")
    execute(args)


if __name__ == "__main__":
    main()
