#!/usr/bin/env python3
"""Prepare, admit or run the frozen memory storage/addressing/readout probe.

`prepare` writes fresh seeded opaque worlds plus exact source, data, protocol
and implementation pins. `admit` re-verifies every pin and writes an admission
receipt. `run` is a guarded native MPS runner: it requires
REPLICA_CAPACITY_GUARD=1, an admitted preparation, the sealed MiniCPM5 source,
and the explicitly pinned writer extension when the protocol names one.

The runner never updates weights, never generates answers, never builds a
pointer/copy arm and never promotes the rejected v5 memory. The parent adds a
capacity-guard entry for this script; this file does not modify the guard.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.memory_diagnosis import (  # noqa: E402
    DEFAULT_CORRECTION_INDEX,
    DEFAULT_DISTRACTORS_PER_WORLD,
    DEFAULT_FACTS_PER_WORLD,
    DEFAULT_LAYER,
    DEFAULT_REVOCATION_INDEX,
    DEFAULT_SEED,
    DEFAULT_WORLD_COUNT,
    admit_probe,
    planned_workload,
    prepare_probe,
    run_diagnostic,
    verify_prepared,
)
from replica_lab.memory_one_write import (  # noqa: E402
    admit_one_write_probe,
    one_write_workload,
    prepare_one_write,
    run_one_write_diagnostic,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def fresh_output(path, parser, label):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        parser.error(f"Use a fresh {label} output path")
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_paths(protocol):
    source = protocol.get("source")
    if not isinstance(source, dict):
        raise ValueError("Protocol carries no exact source pins")
    model_path = Path(source["model_path"])
    model_lock = Path(source["model_lock"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_lock.is_absolute():
        model_lock = ROOT / model_lock
    return model_path.resolve(), model_lock.resolve()


def require_admission(prepared):
    path = prepared / "admission.json"
    if not path.is_file():
        raise ValueError("Preparation is not admitted; run the admit subcommand first")
    admission = json.loads(path.read_text())
    if admission.get("status") != "admitted":
        raise ValueError("Preparation admission is rejected; do not run the native probe")
    current = verify_prepared(prepared)
    for name in ("prepared_manifest_sha256", "protocol_sha256", "data_sha256",
                 "implementation_sha256", "planned_workload"):
        if admission.get(name) != current.get(name):
            raise ValueError(f"Admission {name} differs from current prepared bytes")
    return admission, sha(path)


def checked_extension(args, protocol):
    pin = protocol.get("writer_extension")
    if pin is None:
        if args.writer_extension is not None:
            raise ValueError("Protocol pins no writer extension; refusing an unpinned extension")
        return None
    if args.writer_extension is None:
        raise ValueError("Protocol pins a writer extension; pass --writer-extension")
    supplied = Path(args.writer_extension).resolve()
    pinned = Path(pin["path"])
    if not pinned.is_absolute():
        pinned = (ROOT / pinned).resolve()
    else:
        pinned = pinned.resolve()
    if supplied != pinned:
        raise ValueError("Supplied writer extension path differs from the pin")
    weights = supplied / "extensions.safetensors"
    manifest = supplied / "manifest.json"
    if sha(weights) != pin["weights_sha256"] or sha(manifest) != pin["manifest_sha256"]:
        raise ValueError("Writer extension bytes differ from the pin")
    return supplied


def run_native(args, parser):
    import torch
    from replica_lab.architecture.contracts import FastMemorySpec
    from replica_lab.architecture.loading import exclusive_mps_worker, load_local_mps
    from replica_lab.profiling import RunProfiler

    prepared = Path(args.prepared).resolve()
    if not prepared.is_dir():
        parser.error("Prepared input directory does not exist")
    output = fresh_output(args.output, parser, "run")
    protocol_path = prepared / "protocol.json"
    data_path = prepared / "data.json"
    manifest_path = prepared / "manifest.json"
    protocol = json.loads(protocol_path.read_text())
    data = json.loads(data_path.read_text())
    one_write = protocol.get("format") == "replica.memory_one_write_probe.protocol.v1"
    workload = protocol["planned_workload"] if one_write else planned_workload(protocol)
    admission, admission_sha256 = require_admission(prepared)
    if args.device != "mps":
        raise ValueError("Native storage probe requires --device mps")
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise RuntimeError(
            "Launch through scripts/run_capacity_preflight.py --entry-point "
            "run_memory_storage_probe.py; no unguarded native run is allowed")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no CPU native fallback is permitted")

    model_path, model_lock = source_paths(protocol)
    source = protocol["source"]
    if sha(model_lock) != source["model_lock_sha256"]:
        raise ValueError("Sealed model lock bytes changed")
    if source["source_revision"] != source["required_revision"]:
        raise ValueError("Protocol source revision is not the sealed revision")
    extension_path = checked_extension(args, protocol)
    if protocol["data"]["sha256"] != sha(data_path):
        raise ValueError("Data bytes differ from the protocol pin")
    if protocol["implementation_sha256"] != verify_prepared(prepared)["implementation_sha256"]:
        raise ValueError("Implementation pins differ from the admitted preparation")

    report = {
        "format": "replica.memory_storage_probe.run.v1",
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "native": True,
        "device": "mps",
        "fallback_disabled": True,
        "guard_required": "REPLICA_CAPACITY_GUARD=1",
        "optimizer_updates": 0,
        "checkpoints_saved": 0,
        "downloads": 0,
        "prepared": {
            "path": str(prepared),
            "protocol_sha256": sha(protocol_path),
            "data_sha256": sha(data_path),
            "manifest_sha256": sha(manifest_path),
            "admission_sha256": admission_sha256,
        },
        "workload": workload,
        "source": source,
    }
    profiler = RunProfiler("mps", output)
    try:
        with exclusive_mps_worker(ROOT):
            engine = protocol.get("engine") or {}
            spec = FastMemorySpec(
                heads=engine.get("heads", 4), key_size=engine.get("key_size", 64),
                value_size=engine.get("value_size", 64),
                update_rule=engine.get("update_rule", "sgdn2"),
                depth_mode=engine.get("depth_mode", "shared"),
                depth_writers=engine.get("depth_writers", True))
            profiler.start()
            model, tokenizer, lock = load_local_mps(
                model_path, model_lock, profiler=profiler, fast_memory=spec)
            extension_receipt = None
            if extension_path is not None:
                extension_receipt = model.load_extension(extension_path)
                pin = protocol["writer_extension"]
                if extension_receipt["weights_sha256"] != pin["weights_sha256"]:
                    raise ValueError("Loaded writer extension weights differ from the pin")
                if extension_receipt.get("manifest_digest") != pin.get("manifest_digest"):
                    raise ValueError("Loaded writer extension manifest differs from the pin")
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            report["native"] = {
                "device": "mps",
                "fallback_disabled": True,
                "sealed_revision": lock["revision"],
                "source_lock_sha256": sha(model_lock),
                "writer_extension": extension_receipt,
            }
            report["implementation_sha256"] = protocol["implementation_sha256"]
            if one_write:
                result = run_one_write_diagnostic(
                    model, tokenizer, protocol, output_dir=output,
                    ledger_dir=output / "ledger", worlds=data["worlds"],
                    device="mps", writer_extension_receipt=extension_receipt)
            else:
                result = run_diagnostic(
                    model, tokenizer, protocol, output_dir=output,
                    ledger_dir=output / "ledger", worlds=data["worlds"],
                    device="mps", writer_extension_receipt=extension_receipt)
            rows = result.pop("readouts")
            result["status"] = "completed"
            result["started_at_utc"] = report["started_at_utc"]
            result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            result["native_run"] = report["native"]
            result["pre_allocation_workload"] = workload
            result["prepared"] = report["prepared"]
            result["source"] = source
            result["implementation_sha256"] = protocol["implementation_sha256"]
            result["raw_readouts_path"] = "readouts.jsonl"
            result["codebooks_path"] = "codebooks.json"
            if result["readouts_sha256"] != sha(output / "readouts.jsonl"):
                raise ValueError("Raw readout artifact bytes changed")
            if len(rows) != result["counts"]["rows"]:
                raise ValueError("Raw readout row count differs")
            result["profile"] = profiler.stop()
            dump(output / "run_report.json", result)
            print(json.dumps({
                "status": result["status"],
                "decision": result["decision"],
                "failed_gates": result["failed_gates"],
                "evaluated_rows": result["counts"]["evaluated_rows"],
                "diagnostic_cases": result["counts"].get(
                    "diagnostic_cases", result["counts"].get("unique_cases")),
                "report": str(output / "run_report.json"),
                "readouts": str(output / "readouts.jsonl"),
            }, sort_keys=True))
            return 0
    except BaseException as error:
        failure = dict(report, status="failed",
                       finished_at_utc=datetime.now(timezone.utc).isoformat(),
                       error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(),
                       profile=profiler.stop())
        dump(output / "run_report.json", failure)
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "error": str(error),
                          "report": str(output / "run_report.json")}, sort_keys=True),
              file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write exact pins and opaque worlds")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--mode", choices=("storage", "onewrite"), default="storage",
                         help="onewrite = one fact/one write storage discriminator")
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--worlds", type=int, default=DEFAULT_WORLD_COUNT)
    prepare.add_argument("--layers", default="13,27",
                         help="comma-separated injection layers for --mode onewrite")
    prepare.add_argument("--facts", type=int, default=DEFAULT_FACTS_PER_WORLD)
    prepare.add_argument("--distractors", type=int, default=DEFAULT_DISTRACTORS_PER_WORLD)
    prepare.add_argument("--correction-index", type=int, default=DEFAULT_CORRECTION_INDEX)
    prepare.add_argument("--revocation-index", type=int, default=DEFAULT_REVOCATION_INDEX)
    prepare.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    prepare.add_argument("--model-path", type=Path,
                         default=ROOT / "checkpoints/minicpm5_source")
    prepare.add_argument("--model-lock", type=Path,
                         default=ROOT / "configs/minicpm5_sealed_lock.local.json")
    prepare.add_argument("--writer-extension", type=Path)

    admit = subparsers.add_parser("admit", help="verify exact pins and admit a preparation")
    admit.add_argument("--prepared", type=Path, required=True)

    run = subparsers.add_parser("run", help="guarded native MPS storage/readout probe")
    run.add_argument("--device", choices=("mps",), default="mps")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--writer-extension", type=Path)

    args = parser.parse_args()
    if args.command == "prepare":
        output = fresh_output(args.output, parser, "prepare")
        if args.mode == "onewrite":
            try:
                layers = tuple(int(part) for part in args.layers.split(",") if part.strip())
            except ValueError:
                parser.error("--layers must be comma-separated integers")
            manifest = prepare_one_write(
                output, seed=args.seed, world_count=args.worlds, layers=layers,
                facts_per_world=args.facts,
                model_path=args.model_path, model_lock=args.model_lock,
                writer_extension=args.writer_extension)
        else:
            manifest = prepare_probe(
                output, seed=args.seed, world_count=args.worlds,
                facts_per_world=args.facts, distractors_per_world=args.distractors,
                correction_index=args.correction_index,
                revocation_index=args.revocation_index, layer=args.layer,
                model_path=args.model_path, model_lock=args.model_lock,
                writer_extension=args.writer_extension)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "admit":
        prepared = args.prepared.resolve()
        protocol = json.loads((prepared / "protocol.json").read_text()) \
            if (prepared / "protocol.json").is_file() else {}
        admit = (admit_one_write_probe
                 if protocol.get("format") == "replica.memory_one_write_probe.protocol.v1"
                 else admit_probe)
        try:
            receipt = admit(prepared)
        except Exception as error:
            print(json.dumps({"status": "rejected", "error_type": type(error).__name__,
                              "error": str(error)}, sort_keys=True), file=sys.stderr)
            return 1
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    try:
        return run_native(args, parser)
    except Exception as error:
        output = Path(args.output)
        if output.is_dir():
            dump(output / "run_report.json", {
                "format": "replica.memory_storage_probe.run.v1",
                "status": "failed",
                "native": True,
                "device": "mps",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            })
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
