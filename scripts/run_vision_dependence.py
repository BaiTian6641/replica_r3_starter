#!/usr/bin/env python3
"""Run the paired image-dependence diagnostic and its one-shot finalizer.

``run`` is the only native stage: it collects frozen SigLIP2 patch features and
MiniCPM5 question states for train/dev/final rows, scores dev only, and writes
raw features, dev predictions, a hash-pinned dev report and a run report. It
performs zero parameter updates and never scores or reports the final split.

``finalize`` is CPU-only: it verifies the exact supplied dev-report hash, the
protocol/data/feature pins, reserves a write-once receipt, scores the final
holdout exactly once, and writes the final report. A second finalize attempt is
refused even with a different output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_key] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_dependence import (  # noqa: E402
    PRIMARY_READOUT, assemble_dev_report, assemble_native_run_report,
    audit_dataset, collect_frozen_features, evaluate_records,
    finalize_evaluation as run_finalizer, load_feature_bundle,
    publish_native_artifacts, records_from_bundle, require_fresh_output,
    validate_protocol, verify_data_pins, verify_implementation_pins,
    verify_vision_binding, write_feature_bundle,
)
from replica_lab.vision_qualification import require, sha256_file  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_protocol(path: Path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    workload = validate_protocol(protocol)
    return protocol, workload


def _validate_native_environment(protocol, args) -> None:
    import torch
    from replica_lab.architecture.contracts import VisionSpec
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.architecture.vision import VisionFeatureService
    from replica_lab.profiling import RunProfiler

    require(torch.backends.mps.is_available(), "MPS is unavailable; no CPU fallback")
    require(os.environ.get("REPLICA_CAPACITY_GUARD") == "1",
            "Launch through scripts/run_capacity_preflight.py")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
            "PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        require(os.environ.get(key) == "1", f"{key}=1 is required")
    require(shutil.disk_usage(ROOT).free >= protocol["limits"]["min_free_disk_bytes"],
            "Insufficient free disk headroom")
    require(sha256_file(args.model_lock) == protocol["model"]["lock_sha256"],
            "Model lock bytes differ from the protocol pin")
    lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
    require(lock.get("revision") == protocol["model"]["revision"],
            "Model revision differs from the protocol pin")
    return {"torch": torch, "VisionSpec": VisionSpec,
            "load_local_mps": load_local_mps,
            "VisionFeatureService": VisionFeatureService,
            "RunProfiler": RunProfiler}


def run_native(args) -> int:
    os.chdir(ROOT)
    protocol_path = _resolve(args.protocol)
    protocol, workload = _load_protocol(protocol_path)
    verify_implementation_pins(protocol, ROOT)
    native = _validate_native_environment(protocol, args)
    manifest = verify_data_pins(protocol, root=ROOT)
    dataset = audit_dataset(ROOT / protocol["data"]["root"])
    require(dataset.manifest["counts"] == protocol["data"]["counts"],
            "Audited data counts differ from the protocol")
    source = ROOT / protocol["vision_source"]["path"]
    source_manifest = ROOT / protocol["vision_source"]["manifest"]["path"]
    require(sha256_file(source_manifest)
            == protocol["vision_source"]["manifest"]["sha256"],
            "Vision source manifest bytes changed")
    require(shutil.disk_usage(ROOT).free >= protocol["limits"]["min_free_disk_bytes"],
            "Insufficient free disk headroom")
    output = require_fresh_output(_resolve(args.output))
    lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
    started = _now()
    started_monotonic = time.monotonic()
    report = {
        "format": "replica.vision_dependence.run.v2",
        "version": protocol["version"],
        "stage": "native_dev",
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "data": {"manifest_sha256": protocol["data"]["manifest"]["sha256"],
                 "pins": protocol["data"], "counts": manifest["counts"]},
        "seed": protocol["seed"],
        "model": {"revision": lock["revision"],
                  "lock_sha256": sha256_file(args.model_lock),
                  "vision_digest": protocol["vision_digest"],
                  "model_path": str(args.model_path)},
        "workload_declared": workload,
        "device": "mps",
        "fallback_disabled": True,
        "downloads": 0,
        "zero_updates": True,
        "optimizer_updates": 0,
        "parameters_updated": [],
        "feature_collection_splits": ["train", "dev", "final"],
        "scored_splits": ["dev"],
        "final_scored": False,
        "scores_final": False,
        "started_at_utc": started,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "safetensors")},
    }
    profiler = native["RunProfiler"]("mps", output, sample_interval=0.25)
    try:
        from replica_lab.architecture.loading import exclusive_mps_worker
        with exclusive_mps_worker(ROOT):
            profiler.start()
            adapter, tokenizer, _ = native["load_local_mps"](
                args.model_path, args.model_lock, profiler=profiler,
                vision=native["VisionSpec"](**protocol["vision"]))
            spec = adapter.manifest.vision
            service = native["VisionFeatureService"](
                source, source_manifest, device="mps",
                max_cache_bytes=args.max_cache_bytes)
            verify_vision_binding(protocol, adapter_spec=spec,
                                  service_spec=service.spec)
            frozen_before = service.frozen_audit()
            require(frozen_before["trainable_names"] == []
                    and frozen_before["gradient_names"] == [],
                    "Frozen encoder exposes trainable parameters")
            features = collect_frozen_features(
                service=service, model=adapter, tokenizer=tokenizer,
                dataset=dataset, region_grid=protocol["readout"]["region_grid"])
            require(len(features["rows"]) == workload["observations"],
                    "Feature workload differs from the declaration")
            frozen_after = service.frozen_audit()
            require(frozen_after["parameter_stamp_unchanged"] is True,
                    "Frozen encoder parameters changed during collection")
            feature_pins = write_feature_bundle(
                output, rows=features["rows"],
                image_pooled=features["image_pooled"],
                image_regions=features["image_regions"],
                question=features["question"],
                protocol_sha256=sha256_file(protocol_path),
                data=protocol["data"])
            feature_bytes = (Path(feature_pins["features"]["path"]).stat().st_size
                             + Path(feature_pins["index"]["path"]).stat().st_size)
            require(feature_bytes <= protocol["limits"]["max_feature_bytes"],
                    "Feature bundle exceeds the preregistered byte limit")
            bundle = load_feature_bundle(
                output, expected_index_sha256=feature_pins["index"]["sha256"])
            records = records_from_bundle(dataset, bundle)
            require(len(records) == workload["observations"],
                    "Feature rows differ from the declaration")
            final_worlds = {record["world_id"] for record in records
                            if record["split"] == "final"}
            evaluation = evaluate_records(records, readout=protocol["readout"],
                                          split="dev", seed=protocol["seed"])
            require(set(evaluation["evaluation_worlds"]).isdisjoint(final_worlds),
                    "Dev scoring touched final worlds")
            require(evaluation["split"] == "dev",
                    "The native stage may only score dev")
            predictions_path = output / "dev_predictions.json"
            _write_json(predictions_path, {
                "format": "replica.vision_dependence.dev_predictions.v2",
                "version": protocol["version"],
                "stage": "dev",
                "split": "dev",
                "protocol_sha256": sha256_file(protocol_path),
                "primary_readout": PRIMARY_READOUT,
                "rows": evaluation["predictions"],
                "shuffle": evaluation["shuffle"],
                "scores_final": False,
                "final_scored": False,
            })
            feature_report_pins = {
                "features": feature_pins["features"],
                "index": feature_pins["index"],
                "predictions": {"path": str(predictions_path),
                                "sha256": sha256_file(predictions_path)},
            }
            dev_report = assemble_dev_report(
                protocol=protocol, protocol_path=protocol_path,
                feature_pins=feature_report_pins,
                predictions_sha256=sha256_file(predictions_path),
                evaluation=evaluation, model=report["model"],
                frozen_audit={"before": frozen_before, "after": frozen_after},
                created_at=_now())
            require(dev_report["evaluation"]["split"] == "dev",
                    "Dev report is not a dev report")
            profile = profiler.stop()
            elapsed = time.monotonic() - started_monotonic
            run_report = assemble_native_run_report(
                protocol=protocol, protocol_path=protocol_path,
                dev_report=dev_report, output=output,
                feature_pins=feature_report_pins,
                dev_evaluation=evaluation, model=report["model"],
                frozen_audit={"before": frozen_before, "after": frozen_after},
                workload_declared=workload,
                elapsed_seconds=elapsed, feature_bytes=feature_bytes,
                started_at_utc=started,
                packages=report["packages"],
                extra={
                    "feature_shapes": feature_pins["array_shapes"],
                    "dev_summary": {
                        "primary_accuracy":
                            evaluation["readouts"][PRIMARY_READOUT]["accuracy"],
                        "question_only_accuracy":
                            evaluation["readouts"]["question_only"]["accuracy"],
                        "primary_gain_over_question_only":
                            evaluation["gains"]["over_question_only"],
                        "primary_gain_over_shuffled_image":
                            evaluation["gains"]["over_shuffled_image"],
                        "readout_limits": {
                            "ridge_penalty":
                                protocol["readout"]["ridge_penalty"],
                            "bootstrap_resamples":
                                protocol["readout"]["bootstrap"]["resamples"],
                            "confidence":
                                protocol["readout"]["bootstrap"]["confidence"],
                            "primary": protocol["readout"]["primary"],
                        },
                    },
                    "free_disk_after_bytes": shutil.disk_usage(ROOT).free,
                    "profile": profile,
                })
            run_report = publish_native_artifacts(
                output, dev_report=dev_report, run_report=run_report,
                max_seconds=protocol["limits"]["max_seconds"],
                max_feature_bytes=protocol["limits"]["max_feature_bytes"])
            report = run_report
            dev_report_path = output / "dev_report.json"
            print(json.dumps({
                "status": "completed",
                "dev_report": str(dev_report_path),
                "dev_report_sha256": report["dev_report"]["sha256"],
                "dev_verdict": evaluation["decision"]["verdict"],
                "dev_primary_accuracy":
                    evaluation["readouts"][PRIMARY_READOUT]["accuracy"],
                "dev_question_only_accuracy":
                    evaluation["readouts"]["question_only"]["accuracy"],
                "final_scored": False,
            }, indent=2))
            return 0
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc(),
                      finished_at_utc=_now(), profile=profiler.stop())
        _write_json(output / "failure_report.json", report)
        raise


def finalize(args) -> int:
    os.chdir(ROOT)
    protocol_path = _resolve(args.protocol)
    features_path = _resolve(args.features)
    dev_report_path = _resolve(args.dev_report)
    output = _resolve(args.output)
    report = run_finalizer(
        protocol_path=protocol_path, features_path=features_path,
        dev_report_path=dev_report_path,
        dev_report_sha256=args.dev_report_sha256, output=output)
    print(json.dumps({
        "status": "completed",
        "final_report": str(output / "final_report.json"),
        "dev_report_sha256": report["dev_report_sha256"],
        "verdict": report["evaluation"]["decision"]["verdict"],
        "supported": report["evaluation"]["decision"]["supported"],
        "primary_accuracy":
            report["evaluation"]["readouts"][PRIMARY_READOUT]["accuracy"],
        "gain_over_question_only":
            report["evaluation"]["gains"]["over_question_only"],
        "gain_over_shuffled_image":
            report["evaluation"]["gains"]["over_shuffled_image"],
    }, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--device", choices=("mps",), default="mps")
    run_parser.add_argument("--protocol", type=Path,
                            default=ROOT / "data/vision_dependence_v2/protocol.json")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--model-lock", type=Path, required=True)
    run_parser.add_argument("--max-cache-bytes", type=int, default=8 * 1024**2)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--protocol", type=Path,
                                 default=ROOT / "data/vision_dependence_v2/protocol.json")
    finalize_parser.add_argument("--features", type=Path, required=True)
    finalize_parser.add_argument("--dev-report", type=Path, required=True)
    finalize_parser.add_argument("--dev-report-sha256", type=str, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.device != "mps":
            raise RuntimeError("Only the declared MPS device is supported")
        return run_native(args)
    return finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
