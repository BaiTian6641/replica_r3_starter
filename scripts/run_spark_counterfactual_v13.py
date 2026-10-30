#!/usr/bin/env python3
"""Review or execute one bound fitting-only counterfactual diagnostic on MPS."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab.capacity_guard import CapacityGuard, CapacityPolicy
from replica_lab.intrinsic_memory_budget_v12 import audit_io
from replica_lab.intrinsic_memory_budget_v13 import IncrementalWorkLedger
from replica_lab.intrinsic_memory_counterfactual_protocol_v13 import (
    validate_authorization, validate_bindings)
from replica_lab.intrinsic_memory_counterfactual_v13 import (
    DiagnosticEngineV13, run_diagnostic)
from replica_lab.intrinsic_memory_execution_v5 import bound_json, contained, hash_file
from replica_lab.intrinsic_memory_execution_v12 import charged_write, record_attempt
from replica_lab.intrinsic_memory_native_v5 import base_hash
from replica_lab.intrinsic_memory_native_v12 import close_stage, load_native, reserved_read
from replica_lab.wall_deadline_v1 import WallPolicy, run_wall_bounded
from launch_spark_intrinsic_readout_environment_v1 import validate_environment
from run_spark_intrinsic_coverage_v12 import artifact_bytes, exclusive_worker, host_reading

ENV_PACKET = "REPLICA_COUNTERFACTUAL_V13_PACKET"
ENV_DEADLINE = "REPLICA_COUNTERFACTUAL_V13_DEADLINE"


def environment_only(packet_path):
    packet = json.loads(Path(packet_path).read_text())
    protocol = bound_json(ROOT, packet["protocol"])
    return validate_environment(bound_json(ROOT, protocol["environment"]), root=ROOT)


def worker(args):
    import psutil
    import torch

    packet = json.loads(args.packet.read_text())
    output = contained(ROOT, packet["output_root"])
    if os.environ.get(ENV_PACKET) != hash_file(args.packet):
        raise ValueError("diagnostic supervision missing")
    validate_authorization(ROOT, args.packet, args.authorization, allow_existing=True)
    protocol = bound_json(ROOT, packet["protocol"])
    arm = protocol["arms"]["two"]
    deadline = float(os.environ[ENV_DEADLINE])
    ledger = IncrementalWorkLedger.create(
        output / "work_ledger.jsonl", identity=hash_file(args.packet),
        plan=arm["budget"], deadline=deadline)
    guard = CapacityGuard(CapacityPolicy(), host_reading())
    if guard.stop_reason:
        raise RuntimeError(guard.stop_reason)
    path = None
    try:
        ledger.enter("admission")
        with ledger.work("admission", "environment_source_checkpoint_bindings",
                         artifact_read_bytes=arm["budget"]["admission"]["artifact_read_bytes"]) as event:
            environment = environment_only(args.packet)
            _, protocol, receipt = validate_bindings(ROOT, args.packet, check_source=True)
            observed = (receipt["source_bytes_hashed"]
                        + receipt["environment_closure"]["bytes_verified"])
            checkpoint = protocol["diagnostic"]["checkpoint"]
            checkpoint_root = contained(ROOT, checkpoint["path"])
            observed += sum((checkpoint_root / "components" / name).stat().st_size
                            for name in ("extension.safetensors", "optimizer.pt"))
            observed += (ROOT / protocol["data"]["fit"]["path"]).stat().st_size
            event["actual"] = {"artifact_read_bytes": observed}
            event["source_bytes_hashed"] = receipt["source_bytes_hashed"]
            event["environment_closure"] = receipt["environment_closure"]
        charged_write(ledger, output, "preflight", {
            **receipt, "environment_validation": environment,
            "native_execution": False, "development_labels_parsed": False,
        }, stage="admission")
        ledger.leave("admission", status="completed")

        ledger.enter("load")
        path, backend, spec, initial_hash, stage_ref = load_native(
            ROOT, protocol, "two", ledger, "load")
        checkpoint = protocol["diagnostic"]["checkpoint"]
        checkpoint_root = contained(ROOT, checkpoint["path"])
        checkpoint_bytes = sum(p.stat().st_size for p in checkpoint_root.rglob("*") if p.is_file())
        with ledger.work("load", "trained_extension_optimizer_restore", restart_restores=1,
                         artifact_read_bytes=checkpoint_bytes) as event:
            path.load(checkpoint_root)
            event["actual"] = {"artifact_read_bytes": checkpoint_bytes}
            event["checkpoint_manifest_sha256"] = checkpoint["manifest_sha256"]
        path.freeze_readout()
        ledger.leave("load", status="completed")

        ledger.enter("diagnose")
        engine = DiagnosticEngineV13(path, backend, spec, ledger, max_new_tokens=8,
                                     output=output, stage_ref=stage_ref)
        engine.stage = "diagnose"
        fit_rows = json.loads(reserved_read(
            ledger, "diagnose", "fit_split_read", ROOT / protocol["data"]["fit"]["path"]))["rows"]
        last_driver = 0.0
        last_host = 0.0
        with (output / "driver_samples.jsonl").open("x") as driver_stream, \
                (output / "host_samples.jsonl").open("x") as host_stream:
            def observe():
                nonlocal last_driver, last_host
                now = time.monotonic()
                if now - last_driver >= 0.25:
                    sample = {"monotonic": now, "pid": os.getpid(),
                              "rss_bytes": psutil.Process().memory_info().rss,
                              "mps_driver_bytes": int(torch.mps.driver_allocated_memory()),
                              "mps_active_bytes": int(torch.mps.current_allocated_memory())}
                    guard.observe_profile(sample)
                    driver_stream.write(json.dumps(sample) + "\n")
                    driver_stream.flush()
                    last_driver = now
                if now - last_host >= 2:
                    reading = host_reading()
                    guard.observe_host(reading)
                    host_stream.write(json.dumps({"monotonic": now, **reading}) + "\n")
                    host_stream.flush()
                    last_host = now
                snapshot = ledger.snapshot(cached=True)
                deadlines = ledger.deadline_state(snapshot)
                if (deadlines["outer_remaining_seconds"] <= 0
                        or any(value <= 0 for value in deadlines["stage_remaining_seconds"].values())):
                    raise RuntimeError("stage or outer diagnostic deadline")
                if artifact_bytes(output) > snapshot["reserved"]["artifact_bytes"]:
                    raise RuntimeError("unreserved diagnostic artifact growth")
                if guard.stop_reason:
                    raise RuntimeError(guard.stop_reason)
            observe()
            summary = run_diagnostic(engine, fit_rows, output, observe=observe)
        close_stage(engine, "diagnose", "completed")

        ledger.enter("finalize")
        unchanged = base_hash(path.adapter.base) == initial_hash
        if not unchanged:
            raise ValueError("original parameter bytes changed during diagnostic")
        snapshot = ledger.snapshot()
        charged_write(ledger, output, "result", {
            "format": "replica.intrinsic_memory.counterfactual_result.v13",
            "packet_sha256": hash_file(args.packet),
            "protocol_sha256": packet["protocol"]["sha256"],
            "checkpoint_manifest_sha256": protocol["diagnostic"]["checkpoint"]["manifest_sha256"],
            "summary": summary, "original_weights_unchanged": unchanged,
            "capacity": guard.report(), "artifact_bytes_observed": artifact_bytes(output),
            "reservations": snapshot["reserved"], "audit_io": audit_io(ledger),
            "deadline_state": ledger.deadline_state(snapshot),
            "development_labels_used": False, "parameter_updates": 0,
            "native_capability_or_generalization_claim": False,
        }, stage="finalize")
        ledger.leave("finalize", status="completed")
    except BaseException as error:
        charged_write(ledger, output, "failure", {
            "type": type(error).__name__, "message": str(error),
            "reserved_work_not_refunded": True,
            "development_labels_used": False, "parameter_updates": 0,
        })
        raise
    finally:
        if path is not None:
            path.close()


def launch(args):
    packet = json.loads(args.packet.read_text())
    protocol = bound_json(ROOT, packet["protocol"])
    validate_authorization(ROOT, args.packet, args.authorization)
    arm = protocol["arms"]["two"]
    output = contained(ROOT, packet["output_root"])
    marker = contained(ROOT, packet["attempt_marker"])
    if marker.resolve().is_relative_to(output.resolve()):
        raise ValueError("attempt marker must be outside the output tree")
    with exclusive_worker():
        record_attempt(marker, packet_sha256=hash_file(args.packet),
                       protocol_sha256=packet["protocol"]["sha256"], arm="two",
                       output_root=packet["output_root"],
                       authorization_sha256=hash_file(args.authorization))
        output.mkdir(parents=True, exist_ok=False)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "PYTORCH_ENABLE_MPS_FALLBACK": "0",
               "HF_MODULES_CACHE": str(output / "hf_modules"),
               ENV_PACKET: hash_file(args.packet),
               ENV_DEADLINE: str(time.monotonic() + arm["wall_seconds"] - 11)}
        command = [sys.executable, str(Path(__file__).resolve()), "--packet", str(args.packet),
                   "--authorization", str(args.authorization), "--worker"]
        result = run_wall_bounded(command, cwd=ROOT, output=output / "supervision",
                                  policy=WallPolicy(arm["wall_seconds"]),
                                  launch_authorized=True,
                                  authorization_identity=hash_file(args.authorization), env=env)
        return result.exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--authorization", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--environment-only", action="store_true")
    modes.add_argument("--review", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.packet = args.packet.resolve()
    if args.environment_only:
        print(json.dumps(environment_only(args.packet), indent=2))
        return 0
    if args.review:
        environment = environment_only(args.packet)
        _, _, receipt = validate_bindings(ROOT, args.packet)
        print(json.dumps({"environment_validation": environment, **receipt,
                          "launch_authorized": False}, indent=2))
        return 0
    if args.authorization is None:
        parser.error("an exact diagnostic authorization receipt is required")
    args.authorization = args.authorization.resolve()
    if args.launch:
        return launch(args)
    worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
