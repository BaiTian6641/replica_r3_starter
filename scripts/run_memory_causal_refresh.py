#!/usr/bin/env python3
"""Run frozen-writer causal refresh and fresh-process continuation on MPS."""
from __future__ import annotations

import argparse
from collections import Counter
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


def execute(args):
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise RuntimeError("Use the capacity guard")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    import torch
    from replica_lab.architecture.contracts import FastMemorySpec
    from replica_lab.architecture.loading import exclusive_mps_worker, load_local_mps
    from replica_lab.memory_causal_refresh import (
        VARIANT_NAMES, run_causal_refresh)
    from replica_lab.profiling import RunProfiler

    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    numeric = protocol.get("numeric_policy", {})
    if numeric.get("torch_use_deterministic_algorithms") is True:
        torch.use_deterministic_algorithms(True)
    if numeric.get("mps_fast_math") == "disabled":
        if os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
            raise RuntimeError("PYTORCH_MPS_FAST_MATH=0 is required by the protocol")
    protocol["protocol_sha256"] = _sha(protocol_path)
    if protocol.get("format") != "replica.memory_causal_refresh.protocol.v1":
        raise ValueError("Causal refresh protocol format differs")
    if tuple(protocol["variants"]) != VARIANT_NAMES:
        raise ValueError("Causal refresh variant inventory differs")
    prepared = Path(args.prepared).resolve()
    episodes_path = prepared / "episodes.json"
    labels_path = prepared / "labels.json"
    if (_sha(episodes_path) != protocol["source_prepared"]["episodes_sha256"]
            or _sha(labels_path) != protocol["source_prepared"]["labels_sha256"]):
        raise ValueError("Prepared episodes or labels differ from the protocol")
    episodes = json.loads(episodes_path.read_text())["episodes"]
    labels = json.loads(labels_path.read_text())["episodes"]
    source = protocol["source"]
    model_path = ROOT / source["model_path"]
    lock_path = ROOT / source["model_lock"]
    if _sha(lock_path) != source["model_lock_sha256"]:
        raise ValueError("Sealed model lock bytes changed")
    extension_path = ROOT / protocol["writer_extension"]["path"]
    if (_sha(extension_path / "extensions.safetensors")
            != protocol["writer_extension"]["weights_sha256"]
            or _sha(extension_path / "manifest.json")
            != protocol["writer_extension"]["manifest_sha256"]):
        raise ValueError("Frozen writer extension bytes differ")
    implementation = {name: _sha(ROOT / name)
                      for name in protocol["implementation_paths"]}
    control = protocol["reused_generation_controls"]
    control_report_path = ROOT / control["run_report"]
    if (_sha(control_report_path) != control["run_report_sha256"]
            or _sha(ROOT / control["guard_report"]) != control["guard_report_sha256"]):
        raise ValueError("Reused generation control bytes differ")
    control_report = json.loads(control_report_path.read_text())
    counts = Counter((row["arm"], row["correct"])
                     for row in control_report["answers"])
    expected = {
        "exact_source": control["exact_source_correct"],
        "full_context": control["full_context_correct"],
        "memory": control["memory_correct"],
        "context_removed": control["context_removed_correct"],
    }
    for arm, correct in expected.items():
        if counts[(arm, True)] != correct:
            raise ValueError(f"Reused {arm} control count differs")
        if sum(counts[(arm, value)] for value in (True, False)) != control["cases_per_arm"]:
            raise ValueError(f"Reused {arm} control coverage differs")

    output = Path(args.output)
    profiler = RunProfiler("mps", output)
    profiler.start()
    try:
        with exclusive_mps_worker(ROOT):
            engine = protocol["engine"]
            spec = FastMemorySpec(
                heads=engine["heads"], key_size=engine["key_size"],
                value_size=engine["value_size"], update_rule=engine["update_rule"],
                depth_mode=engine["depth_mode"], depth_writers=engine["depth_writers"])
            model, tokenizer, lock = load_local_mps(
                model_path, lock_path, profiler=profiler, fast_memory=spec)
            receipt = model.load_extension(extension_path)
            if (receipt["weights_sha256"] != protocol["writer_extension"]["weights_sha256"]
                    or receipt.get("manifest_digest")
                    != protocol["writer_extension"]["manifest_digest"]):
                raise ValueError("Loaded writer identity differs")
            report = run_causal_refresh(
                model, tokenizer, protocol, output_dir=output,
                episodes=episodes, labels_by_episode=labels,
                writer_extension_receipt=receipt, device="mps",
                resume_checkpoint=args.resume_checkpoint)
        report["implementation_sha256"] = implementation
        report["source"] = source
        report["prepared"] = {
            "path": str(prepared), "episodes_sha256": _sha(episodes_path),
            "labels_sha256": _sha(labels_path),
        }
        report["reused_generation_controls"] = {
            **control, "verified_counts": expected,
            "new_generation_executed": False,
        }
        report["fresh_process_parity"] = None
        if args.resume_checkpoint:
            if not args.baseline_report:
                raise ValueError("Resume requires --baseline-report for parity verification")
            baseline = json.loads(Path(args.baseline_report).read_text())
            if (baseline["protocol_sha256"] != protocol["protocol_sha256"]
                    or baseline["checkpoint"]["sha256"] != report["checkpoint"]["sha256"]):
                raise ValueError("Resume baseline protocol or checkpoint differs")
            if report["raw"] != baseline["raw"] or report["metrics"] != baseline["metrics"]:
                raise ValueError("Fresh-process outputs differ from the first process")
            if report["contexts_encoded"] != 0:
                raise ValueError("Fresh process re-encoded old observations")
            report["fresh_process_parity"] = True
        report["profile"] = profiler.stop()
        (output / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"decision": report["decision"],
                          "resumed": report["resumed"],
                          "fresh_process_parity": report["fresh_process_parity"],
                          "report": str(output / "run_report.json")}))
        return report
    except BaseException as error:
        failure = {"status": "failed", "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc(),
                   "profile": profiler.stop()}
        (output / "failure_report.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "configs/memory_causal_refresh_v1.json")
    parser.add_argument("--prepared", type=Path,
                        default=ROOT / "data/memory_matched_20261025/prepared02")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use fresh output")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device != "mps":
        parser.error("Native execution requires MPS")
    execute(args)


if __name__ == "__main__":
    main()
