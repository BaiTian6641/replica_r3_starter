#!/usr/bin/env python3
"""Prepare or run the frozen vision information probe under the native guard.

`prepare` writes one hash-pinned protocol from the accepted V0 vision/model
sections and the named V1 data admission sections. `run` performs zero-update
frozen-encoder feature collection and closed-form readouts, then writes a report.
Neither trains any parameter nor makes a perception claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_diagnostic import (  # noqa: E402
    DEFAULT_DECISION, DIAGNOSTIC_FORMAT, DIAGNOSTIC_IMPLEMENTATION_FILES,
    diagnostic_workload, run_diagnostic, validate_diagnostic_protocol,
)
from replica_lab.vision_qualification import (  # noqa: E402
    load_verified_dataset, require, resolve_project_path, sha256_file,
    verify_implementation_pins,
)
from replica_lab.vision_overfit import (  # noqa: E402
    OVERFIT_FORMAT, decide_overfit, evaluate_cases, overfit_training_rows,
    summarize, validate_overfit_protocol,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args):
    template = json.loads(args.template.read_text())
    data_protocol = json.loads(args.data_protocol.read_text())
    protocol = {
        "format": DIAGNOSTIC_FORMAT, "root": ".", "stage": "v1",
        "vision": template["vision"], "vision_digest": template["vision_digest"],
        "vision_source": template["vision_source"], "model": template["model"],
        "data": data_protocol["data"],
        "limits": {"max_observations": args.max_observations,
                   "max_seconds": args.max_seconds,
                   "min_free_disk_bytes": args.min_free_disk_bytes,
                   "max_rows": args.max_rows},
        "seed": args.seed, "train_worlds": args.train_worlds,
        "dev_worlds": args.dev_worlds, "per_world_cases": args.per_world_cases,
        "decision": dict(DEFAULT_DECISION),
        "implementation_sha256": {relative: sha256_file(ROOT / relative)
                                  for relative in DIAGNOSTIC_IMPLEMENTATION_FILES},
    }
    if args.stage == 'frozen_readout':
        workload = validate_diagnostic_protocol(protocol)
    else:
        protocol.update({
            'format': OVERFIT_FORMAT,
            'train': {
                'updates': args.updates, 'grad_accum': args.grad_accum,
                'lr_new': args.lr, 'momentum': 0.95, 'ns_steps': 5,
                'block_size': 2048, 'weight_decay': 0.01, 'max_grad_norm': 1.0,
                'max_seconds': args.max_seconds, 'optimizer': 'muon8bit',
                'repeats': 1, 'seed': args.seed, 'repeat_policy': 'once',
                'schedule_version': 'seeded_epoch_shuffle.v1',
            },
            'eval': {'max_new_tokens': args.max_new_tokens},
            'initial_checkpoint': {
                'path': str(args.initial_checkpoint),
                'manifest_sha256': sha256_file(args.initial_checkpoint / 'manifest.json'),
                'weights_sha256': sha256_file(args.initial_checkpoint / 'extensions.safetensors'),
            },
        })
        protocol.pop('decision', None)
        workload = validate_overfit_protocol(protocol)
    protocol_path = args.protocol_out if args.protocol_out.is_absolute() \
        else ROOT / args.protocol_out
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(protocol_path, protocol)
    print(json.dumps({"protocol": str(protocol_path),
                      "protocol_sha256": sha256_file(protocol_path),
                      "workload": workload}, indent=2))
    return 0


def run(args):
    import torch
    from replica_lab.architecture.loading import exclusive_mps_worker, load_local_mps
    from replica_lab.architecture.vision import VisionFeatureService
    from replica_lab.profiling import RunProfiler
    from dataclasses import asdict
    from replica_lab.architecture.contracts import VisionSpec

    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text())
    if protocol['format'] == OVERFIT_FORMAT:
        return run_overfit(args)
    workload = validate_diagnostic_protocol(protocol)
    verify_implementation_pins(protocol, ROOT)
    require(torch.backends.mps.is_available(), "MPS is unavailable; no CPU fallback")
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise RuntimeError("Launch through scripts/run_capacity_preflight.py")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    require(shutil.disk_usage(ROOT).free >= protocol["limits"]["min_free_disk_bytes"],
            "Insufficient free disk headroom")
    require(sha256_file(args.model_lock) == protocol["model"]["lock_sha256"],
            "Model lock bytes differ from the protocol pin")
    lock = json.loads(args.model_lock.read_text())
    require(lock.get("revision") == protocol["model"]["revision"],
            "Model revision differs from the protocol pin")
    dataset = load_verified_dataset(protocol)
    require(dataset.stage == "v1", "Vision diagnostic requires the v1 data stage")

    source = resolve_project_path(protocol, protocol["vision_source"]["path"])
    manifest_path = resolve_project_path(protocol, protocol["vision_source"]["manifest"]["path"])
    require(sha256_file(manifest_path) == protocol["vision_source"]["manifest"]["sha256"],
            "Vision source manifest bytes changed")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Use a fresh diagnostic output path")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "replica.vision_diagnostic.run.v1",
        "stage": "frozen_readout",
        "protocol_sha256": sha256_file(protocol_path),
        "data_audit": dataset.pins,
        "model_revision": lock["revision"],
        "model_lock_sha256": sha256_file(args.model_lock),
        "vision_source_manifest_sha256": sha256_file(manifest_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "safetensors")},
        "device": "mps", "fallback_disabled": True, "downloads": 0,
        "workload_declared": workload,
    }
    profiler = RunProfiler("mps", output, sample_interval=0.25)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            adapter, tokenizer, _ = load_local_mps(
                args.model_path, args.model_lock, profiler=profiler,
                vision=VisionSpec(**protocol["vision"]))
            spec = adapter.manifest.vision
            require(spec is not None and spec.digest == protocol["vision_digest"],
                    "Loaded vision spec differs from the protocol")
            service = VisionFeatureService(
                source, manifest_path, device="mps",
                max_cache_bytes=protocol["limits"].get("max_cache_bytes", 8 * 1024**2))
            diagnostic = run_diagnostic(adapter, tokenizer, service, dataset,
                                        protocol, output=output / "diagnostic.json")
            report["diagnostic"] = diagnostic
            report["profile"] = profiler.stop()
            report["free_disk_after_bytes"] = shutil.disk_usage(ROOT).free
            report["status"] = "completed"
            write_json(output / "run_report.json", report)
            print(json.dumps({"status": "completed",
                              "decision": diagnostic["verdict"]["decision"],
                              "changed": diagnostic["verdict"]["justified_next_change"],
                              "report": str(output / "run_report.json")}, indent=2))
            return 0
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc(),
                      profile=profiler.stop())
        write_json(output / "failure_report.json", report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--protocol-out", type=Path,
                                default=ROOT / "configs/vision_frozen_readout_v1.json")
    prepare_parser.add_argument("--template", type=Path,
                                default=ROOT / "configs/minicpm5_vision_v0_qualification_v3.json")
    prepare_parser.add_argument("--data-protocol", type=Path,
                                default=ROOT / "configs/minicpm5_vision_v1_projector_alignment_v2.json")
    prepare_parser.add_argument("--seed", type=int, default=17)
    prepare_parser.add_argument("--train-worlds", type=int, default=24)
    prepare_parser.add_argument("--dev-worlds", type=int, default=24)
    prepare_parser.add_argument("--per-world-cases", type=int, default=2)
    prepare_parser.add_argument("--max-observations", type=int, default=128)
    prepare_parser.add_argument("--max-seconds", type=int, default=1800)
    prepare_parser.add_argument("--max-rows", type=int, default=2048)
    prepare_parser.add_argument("--min-free-disk-bytes", type=int, default=5368709120)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--device", choices=("mps",), default="mps")
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--model-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    if args.device != "mps":
        raise RuntimeError("Only the declared MPS device is supported")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
