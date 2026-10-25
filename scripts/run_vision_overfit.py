#!/usr/bin/env python3
"""Bounded tiny-set projector overfit probe and fresh-process reload.

One declared projector-only update cohort is fitted on train worlds, exported,
reloaded, then evaluated on those worlds and fresh disjoint dev worlds with
correct/wrong/no-image controls. This is a diagnostic, not a V1 promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab.vision_overfit import (  # noqa: E402
    OVERFIT_FORMAT, decide_overfit, evaluate_cases, overfit_training_rows,
    summarize, validate_overfit_protocol,
)
from replica_lab.vision_diagnostic import (  # noqa: E402
    DIAGNOSTIC_IMPLEMENTATION_FILES, select_cases,
)
from replica_lab.vision_qualification import (  # noqa: E402
    VISUAL_QUESTION_TEMPLATE, load_verified_dataset, require,
    resolve_project_path, sha256_file, verify_implementation_pins,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args):
    template = json.loads(args.template.read_text())
    data_protocol = json.loads(args.data_protocol.read_text())
    checkpoint = args.initial_checkpoint
    require((checkpoint / "manifest.json").is_file(),
            "Initial checkpoint extension manifest is missing")
    protocol = {
        "format": OVERFIT_FORMAT, "root": ".", "stage": "v1",
        "vision": template["vision"], "vision_digest": template["vision_digest"],
        "vision_source": template["vision_source"], "model": template["model"],
        "data": data_protocol["data"],
        "limits": {"max_rows": args.max_rows, "max_seconds": args.max_seconds,
                   "min_free_disk_bytes": args.min_free_disk_bytes,
                   "max_export_bytes": args.max_export_bytes,
                   "max_prompt_tokens": args.max_prompt_tokens},
        "seed": args.seed,
        "train_worlds": args.train_worlds, "dev_worlds": args.dev_worlds,
        "per_world_cases": args.per_world_cases,
        "train": {
            "updates": args.updates, "grad_accum": args.grad_accum,
            "lr_new": args.lr, "momentum": 0.95, "ns_steps": 5,
            "block_size": 2048, "weight_decay": 0.01, "max_grad_norm": 1.0,
            "max_seconds": args.max_seconds, "optimizer": args.optimizer,
            "repeats": args.repeats, "seed": args.seed,
            "repeat_policy": "declared_epochs",
            "schedule_version": "seeded_epoch_shuffle.v1",
        },
        "eval": {"max_new_tokens": args.max_new_tokens},
        "initial_checkpoint": {
            "path": str(checkpoint),
            "manifest_sha256": sha256_file(checkpoint / "manifest.json"),
            "weights_sha256": sha256_file(checkpoint / "extensions.safetensors"),
        },
        "implementation_sha256": {relative: sha256_file(ROOT / relative)
                                  for relative in DIAGNOSTIC_IMPLEMENTATION_FILES},
    }
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
    from replica_lab.architecture.contracts import VisionSpec
    from replica_lab.architecture.loading import exclusive_mps_worker, load_local_mps
    from replica_lab.architecture.vision import VisionFeatureService
    from replica_lab.profiling import RunProfiler
    import run_vision_qualification as v1_runner

    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text())
    workload = validate_overfit_protocol(protocol)
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
    require(dataset.stage == "v1", "Overfit probe requires the v1 data stage")
    source = resolve_project_path(protocol, protocol["vision_source"]["path"])
    manifest_path = resolve_project_path(protocol, protocol["vision_source"]["manifest"]["path"])
    require(sha256_file(manifest_path) == protocol["vision_source"]["manifest"]["sha256"],
            "Vision source manifest bytes changed")
    checkpoint = protocol.get('initial_checkpoint')
    checkpoint_path = None if checkpoint is None else Path(checkpoint['path'])
    if checkpoint_path is not None:
        require(sha256_file(checkpoint_path / 'manifest.json') == checkpoint['manifest_sha256'],
                'Initial checkpoint manifest bytes changed')
        require(sha256_file(checkpoint_path / 'extensions.safetensors') == checkpoint['weights_sha256'],
                'Initial checkpoint weights changed')
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Use a fresh overfit output path")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "replica.vision_overfit_probe.run.v1", "stage": args.stage,
        "protocol_sha256": sha256_file(protocol_path),
        "data_audit": dataset.pins, "model_revision": lock["revision"],
        "model_lock_sha256": sha256_file(args.model_lock),
        "vision_source_manifest_sha256": sha256_file(manifest_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload_declared": workload, "device": "mps",
        "fallback_disabled": True, "downloads": 0,
    }
    profiler = RunProfiler("mps", output, sample_interval=0.25)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            adapter, tokenizer, lock = load_local_mps(
                args.model_path, args.model_lock, profiler=profiler,
                vision=VisionSpec(**protocol["vision"]))
            if checkpoint_path is not None:
                adapter.load_extension(checkpoint_path)
            scope = v1_runner.protocol_scope(protocol)
            service = VisionFeatureService(source, manifest_path, device="mps",
                                           projector_hidden=protocol["vision"]["projector_hidden"])
            require(service.spec.digest == adapter.manifest.vision.digest,
                    "Feature-service and adapter vision producer digests differ")
            train_cases = select_cases(dataset, "train",
                                       limit=protocol["train_worlds"],
                                       per_world=protocol["per_world_cases"])
            dev_cases = select_cases(dataset, "dev",
                                     limit=protocol["dev_worlds"],
                                     per_world=protocol["per_world_cases"])
            for cases in (train_cases, dev_cases):
                for index, item in enumerate(cases):
                    item["wrong_case_id"] = cases[(index + 1) % len(cases)]["case_id"]
            require({item["group_id"] for item in train_cases}.isdisjoint(
                {item["group_id"] for item in dev_cases}),
                "Overfit train and dev worlds overlap")
            training = None
            if args.stage == "overfit":
                (output / "v1").mkdir(parents=True, exist_ok=True)
                train_protocol = dict(
                    protocol,
                    v1={"train": protocol["train"],
                        "question_template": VISUAL_QUESTION_TEMPLATE},
                    limits=dict(protocol["limits"], max_prompt_tokens=256))
                admission = {"_training_rows": overfit_training_rows(dataset, train_cases)}
                session = v1_runner.make_session(
                    adapter, tokenizer, v1_runner.empty_ledger(output))
                training = v1_runner.train_projector(
                    adapter, session, tokenizer, train_protocol, service, dataset,
                    admission, scope, output)
                final_extension = v1_runner.write_extension(
                    adapter, output / "final_extension", train_protocol,
                    label="overfit_extension")
                adapter.load_extension(output / "final_extension")
                report["final_extension"] = final_extension
                report["training"] = {
                    "completed_updates": training["completed_updates"],
                    "declared_updates": training["declared_updates"],
                    "wall_seconds": training["wall_seconds"],
                    "optimizer_route": training["optimizer_route"],
                    "trainable_receipt": training["trainable_receipt"],
                    "base_parameter_stamp_unchanged": training["base_parameter_stamp_unchanged"],
                    "nonprojector_tensors_unchanged": training["nonprojector_tensors_unchanged"],
                }
            generations = []
            train_rows = evaluate_cases(
                adapter, tokenizer, service, dataset, train_cases, scope=scope,
                output=output, max_new_tokens=protocol["eval"]["max_new_tokens"])
            dev_rows = evaluate_cases(
                adapter, tokenizer, service, dataset, dev_cases, scope=scope,
                output=output, max_new_tokens=protocol["eval"]["max_new_tokens"])
            generations.extend(train_rows + dev_rows)
            train_summary, dev_summary = summarize(train_rows), summarize(dev_rows)
            verdict = decide_overfit(train_summary, dev_summary)
            write_json(output / "overfit_generations.jsonl", generations)
            report.update({
                "train_fit": train_summary, "fresh_dev": dev_summary,
                "verdict": verdict,
                "optimizer_updates": training["completed_updates"] if training else 0,
                "parameters_updated": training["trainable_receipt"] if training else [],
                "profile": profiler.stop(), "status": "completed",
            })
            if args.stage == "overfit_reload" and args.prior_report is not None:
                prior = json.loads(args.prior_report.read_text())
                prior_rows = prior.get("overfit_generations", prior.get("generations", []))
                prior_path = args.prior_report.parent / "overfit_generations.jsonl"
                if prior_path.is_file():
                    prior_rows = json.loads(prior_path.read_text())
                else:
                    prior_rows = prior.get("overfit_generations", prior.get("generations", []))
                require(prior_rows == generations,
                        "Fresh-process reload changed generated answers")
                report["reload_bitwise_equal"] = True
            write_json(output / "run_report.json", report)
            print(json.dumps({"status": "completed", "decision": verdict["decision"],
                              "changed": verdict["justified_next_change"],
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
                                default=ROOT / "configs/vision_overfit_v1.json")
    prepare_parser.add_argument("--template", type=Path,
                                default=ROOT / "configs/minicpm5_vision_v0_qualification_v3.json")
    prepare_parser.add_argument("--data-protocol", type=Path,
                                default=ROOT / "configs/minicpm5_vision_v1_projector_alignment_v2.json")
    prepare_parser.add_argument("--seed", type=int, default=17)
    prepare_parser.add_argument("--train-worlds", type=int, default=8)
    prepare_parser.add_argument("--optimizer", choices=("muon8bit", "sgd"), default="muon8bit")
    prepare_parser.add_argument("--dev-worlds", type=int, default=8)
    prepare_parser.add_argument("--per-world-cases", type=int, default=1)
    prepare_parser.add_argument("--updates", type=int, default=16)
    prepare_parser.add_argument("--grad-accum", type=int, default=1)
    prepare_parser.add_argument("--repeats", type=int, default=2)
    prepare_parser.add_argument("--lr", type=float, default=1e-4)
    prepare_parser.add_argument("--max-new-tokens", type=int, default=8)
    prepare_parser.add_argument("--max-rows", type=int, default=2048)
    prepare_parser.add_argument("--max-seconds", type=int, default=1800)
    prepare_parser.add_argument("--max-export-bytes", type=int, default=134217728)
    prepare_parser.add_argument("--max-prompt-tokens", type=int, default=256)
    prepare_parser.add_argument("--min-free-disk-bytes", type=int, default=5368709120)
    prepare_parser.add_argument("--initial-checkpoint", type=Path,
                                default=ROOT / "runs/minicpm5_vision_v0_20261025/attempt02/extension")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--device", choices=("mps",), default="mps")
    run_parser.add_argument("--stage", choices=("overfit", "overfit_reload"),
                            default="overfit")
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--model-lock", type=Path, required=True)
    run_parser.add_argument("--prior-report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    if args.device != "mps":
        raise RuntimeError("Only the declared MPS device is supported")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
