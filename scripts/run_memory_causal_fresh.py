#!/usr/bin/env python3
"""Run the fresh causal selection protocol and its restart on MPS.

Two fresh processes are required: the first writes ``policy_checkpoint.pt``;
the second loads it with ``--resume-checkpoint`` and ``--baseline-report``,
encodes only genuinely new post-restart queries, verifies byte-identical
same-query parity and evaluates the live-revocation control.
"""
from __future__ import annotations

import argparse
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


def execute(args):
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise RuntimeError("Use the capacity guard")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    import torch
    from replica_lab.architecture.contracts import FastMemorySpec, fingerprint
    from replica_lab.architecture.loading import exclusive_mps_worker, load_local_mps
    from replica_lab.memory_causal_fresh import (
        PROTOCOL_FORMAT, VARIANT_NAMES, run_memory_causal_fresh)
    from replica_lab.memory_diagnosis import (
        _display_path, _extension_pins, _resolve_path, _source_pins)
    from replica_lab.profiling import RunProfiler

    protocol_path = Path(args.protocol).resolve()
    protocol = _read_json(protocol_path)
    numeric = protocol.get("numeric_policy", {})
    if numeric.get("torch_use_deterministic_algorithms") is True:
        torch.use_deterministic_algorithms(True)
    if numeric.get("mps_fast_math") == "disabled":
        if os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
            raise RuntimeError("PYTORCH_MPS_FAST_MATH=0 is required by the protocol")
    protocol["protocol_sha256"] = _sha(protocol_path)
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("Fresh causal protocol format differs")
    if tuple(protocol["variant_names"]) != VARIANT_NAMES:
        raise ValueError("Fresh causal variant inventory differs")
    if protocol["budget"]["optimizer_updates"] != 0:
        raise ValueError("Fresh causal protocol must declare zero updates")
    for condition in ("abort_on_checkpoint_binding_mismatch",
                      "abort_on_duplicate_observe_commit",
                      "abort_on_revoked_acceptance"):
        if condition not in protocol["stop_conditions"]:
            raise ValueError(f"Fresh causal protocol lacks stop condition: {condition}")

    prepared = Path(args.prepared).resolve()
    if _sha(prepared / "config.json") != protocol["config_sha256"]:
        raise ValueError("Prepared configuration snapshot differs")
    pins = protocol["source_prepared"]
    fixture_hashes = {
        "episodes_sha256": _sha(prepared / "episodes.json"),
        "labels_sha256": _sha(prepared / "labels.json"),
        "revocations_sha256": _sha(prepared / "revocations.json"),
        "inventory_sha256": _sha(prepared / "inventory.json"),
    }
    for key, value in fixture_hashes.items():
        if pins[key] != value:
            raise ValueError(f"Prepared fixture {key} differs from the protocol")
    episodes = _read_json(prepared / "episodes.json")["episodes"]
    labels = _read_json(prepared / "labels.json")["episodes"]
    revocations = _read_json(prepared / "revocations.json")["revocations"]

    source_pins = _source_pins(
        _resolve_path(protocol["source"]["model_path"]),
        _resolve_path(protocol["source"]["model_lock"]))
    lock = _read_json(_resolve_path(protocol["source"]["model_lock"]))
    snapshot = lock["snapshot_files_sha256"]
    source_pins["tokenizer_sha256"] = fingerprint({
        name: digest for name, digest in snapshot.items()
        if "tokenizer" in name.lower() or name == "special_tokens_map.json"})
    if source_pins != protocol["source_pins"]:
        raise ValueError("Sealed source or tokenizer bytes differ from the protocol")
    writer_pins = _extension_pins(
        _resolve_path(protocol["writer_extension"]["path"]))
    if writer_pins != protocol["writer_pins"]:
        raise ValueError("Frozen writer bytes differ from the protocol")
    implementation = {name: _sha(ROOT / name)
                      for name in protocol["implementation_paths"]}
    if implementation != protocol["implementation_sha256"]:
        raise ValueError("Implementation bytes differ from the protocol")

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use fresh output")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = None
    if args.resume_checkpoint:
        if not args.baseline_report:
            raise ValueError("Resume requires --baseline-report")
        baseline = _read_json(args.baseline_report)
        if baseline["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("Baseline protocol differs")

    profiler = RunProfiler("mps", args.output)
    profiler.start()
    try:
        with exclusive_mps_worker(ROOT):
            engine = protocol["engine"]
            spec = FastMemorySpec(
                heads=engine["heads"], key_size=engine["key_size"],
                value_size=engine["value_size"],
                update_rule=engine["update_rule"],
                depth_mode=engine["depth_mode"],
                depth_writers=engine["depth_writers"])
            model, tokenizer, lock = load_local_mps(
                _resolve_path(protocol["source"]["model_path"]),
                _resolve_path(protocol["source"]["model_lock"]),
                profiler=profiler, fast_memory=spec)
            receipt = model.load_extension(
                _resolve_path(protocol["writer_extension"]["path"]))
            if (receipt["weights_sha256"]
                    != protocol["writer_extension"]["weights_sha256"]
                    or receipt.get("manifest_digest")
                    != protocol["writer_extension"]["manifest_digest"]):
                raise ValueError("Loaded writer identity differs")
            report = run_memory_causal_fresh(
                model, tokenizer, protocol, output_dir=args.output,
                episodes=episodes, labels_by_episode=labels,
                revocations=revocations, writer_extension_receipt=receipt,
                source_pins=source_pins, implementation_sha256=implementation,
                fixture_hashes=fixture_hashes, device="mps",
                resume_checkpoint=args.resume_checkpoint,
                baseline_report=baseline, profiler=profiler)
        report["implementation_sha256"] = implementation
        report["source_pins"] = source_pins
        report["writer_pins"] = writer_pins
        report["prepared"] = {
            "path": _display_path(prepared), **fixture_hashes,
            "config_sha256": _sha(prepared / "config.json"),
            "protocol_path": _display_path(protocol_path),
        }
        report["fresh_process_parity"] = (
            report["parity"]["same_query_parity"] if args.resume_checkpoint
            else None)
        if args.resume_checkpoint:
            if report["encoded"]["old_observations_reencoded"] != 0:
                raise ValueError("Fresh process re-encoded old observations")
            if not report["fresh_process_parity"]:
                raise ValueError("Fresh process same-query parity failed")
        report["profile"] = profiler.stop()
        report["resource_telemetry"] = {
            "source": "RunProfiler",
            "memory": report["profile"].get("memory"),
            "mps_recommended_max_memory_bytes": report["profile"].get(
                "mps_recommended_max_memory_bytes"),
            "host_start": report["profile"].get("host_start"),
            "host_end": report["profile"].get("host_end"),
            "errors": report["profile"].get("errors"),
        }
        (args.output / "run_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "decision": report["decision"],
            "phase": report["phase"],
            "resumed": report["resumed"],
            "fresh_process_parity": report["fresh_process_parity"],
            "exactly_once": report["exactly_once"]["passed"],
            "limits": report["limits"]["passed"],
            "report": str(args.output / "run_report.json"),
        }, sort_keys=True))
        return report
    except BaseException as error:
        failure = {"status": "failed", "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc(),
                   "profile": profiler.stop()}
        (args.output / "failure_report.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "data/memory_causal_fresh_20261026/prepared01/protocol.json")
    parser.add_argument("--prepared", type=Path,
                        default=ROOT / "data/memory_causal_fresh_20261026/prepared01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args()
    if args.device != "mps":
        parser.error("Native execution requires MPS")
    execute(args)


if __name__ == "__main__":
    main()
