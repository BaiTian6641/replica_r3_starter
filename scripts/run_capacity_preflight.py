#!/usr/bin/env python3
"""Run one native MPS trainer behind a declared capacity stop policy.

Pass normal train_text.py arguments after --. Thresholds are sampled safeguards,
not a guarantee against an OOM. No allocator/environment limits are modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replica_lab.capacity_guard import CapacityGuard, CapacityPolicy, parse_host_readings


def host_reading():
    def run(args):
        return subprocess.run(args, capture_output=True, text=True, check=True, timeout=2).stdout
    return parse_host_readings(run(["/usr/sbin/sysctl", "-n", "vm.swapusage"]),
                               run(["/usr/bin/memory_pressure", "-Q"]))


def training_arguments(arguments):
    arguments = list(arguments)
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    def value(flag):
        if arguments.count(flag) != 1:
            raise ValueError(f"Exactly one {flag} argument is required")
        position = arguments.index(flag)
        if position + 1 >= len(arguments):
            raise ValueError(f"Missing {flag} value")
        return arguments[position + 1]
    if value("--device") != "mps":
        raise ValueError("This Mac capacity guard requires --device mps")
    output = Path(value("--output"))
    return arguments, (output if output.is_absolute() else ROOT / output).resolve()


def continuation_requested(arguments):
    """True only for an explicit resume or development/final continuation.

    An initial fit must never reuse an existing output.  The continuation flag
    is accepted only when the trainer arguments themselves declare resume or a
    post-fit stage, so arbitrary existing output cannot be smuggled through.
    """
    arguments = [item for item in arguments if item != "--"]
    if "--resume" in arguments:
        return True
    if arguments.count("--stage") != 1:
        return False
    position = arguments.index("--stage")
    if position + 1 >= len(arguments):
        return False
    return arguments[position + 1] in ("development", "final", "sealed_final")


def stop_child(child):
    """Interrupt first so the trainer can preserve its failure profile."""
    for sig, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        if child.poll() is not None:
            return
        child.send_signal(sig)
        try:
            child.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-point", choices=("train_text.py", "qualify_architecture.py",
                                                 "profile_full_training.py",
                                                 "probe_fused_adamw_sustained.py",
                                                 "run_replica_session.py", "e3_speed_probe.py",
                                                 "run_e3_qualification.py", "run_e3_task.py", "run_e3_d2_action.py", "run_e3_dagger.py", "run_e3_candidate_rerank.py", "run_e2_candidate_answers.py", "run_e2_retrieval.py",
                                                 "run_e2_capacity.py", "run_e2_diagnostic.py", "run_e4_fast_read.py",
                                                 "run_m2_dynamic.py",
                                                 "run_memory_reliability.py",
                                                 "run_vision_qualification.py",
                                                 "run_memory_storage_probe.py",
                                                 "run_vision_diagnostic.py",
                                                 "run_vision_dependence.py",
                                                 "run_memory_matched.py",
                                                 "run_vision_curriculum.py",
                                                 "run_vision_stage_diagnostic.py",
                                                 "run_vision_balanced_alignment.py",
                                                 "run_memory_update_dynamics.py",
                                                 "run_memory_causal_refresh.py",
                                                 "run_memory_causal_fresh.py", "run_memory_bank_value.py",
                                                 "run_memory_bank_readout_v2.py", "run_nested_text.py", "diagnose_nested_response.py", "qualify_spark2_5.py", "run_vision_learnability_diagnostic.py", "run_vision_learnability_curve.py", "run_vision_collapse_diagnostic.py",
                                                 "run_m3_restart.py",
                                                 "eval_m3_retention.py",
                                                 "run_vision_overfit.py",
                                                 "analyze_overfit_fit.py",
                                                 "run_m4_pattern.py",
                                                 "run_m4_restart.py",
                                                 "run_e2_cross_rerank.py", "run_dc_p1_native_response_v2_20261027.py", "run_dc_r1_native_value_v3_20261028.py", "run_dc_p1_source_free_feature_alignment_v2.py"),
                        default="train_text.py")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--driver-limit-gib", type=float,
                        default=CapacityPolicy().driver_limit_bytes / 1024**3)
    parser.add_argument("--swap-growth-limit-mib", type=float, default=512)
    parser.add_argument("--minimum-free-percent", type=float, default=15)
    parser.add_argument("--sustained-host-samples", type=int, default=3)
    parser.add_argument("--telemetry-timeout-seconds", type=float, default=30,
                        help="missing/stale driver telemetry timeout; production default 30s")
    parser.add_argument("--continue-existing-output", action="store_true",
                        help="allow an existing output only for --resume or "
                             "--stage development|final")
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.telemetry_timeout_seconds <= 0:
        parser.error("telemetry timeout must be positive")
    if sys.platform != "darwin":
        parser.error("This watchdog uses macOS host telemetry")
    try:
        trainer_args, output = training_arguments(args.trainer_args)
        policy = CapacityPolicy(int(args.driver_limit_gib * 1024**3),
                                int(args.swap_growth_limit_mib * 1024**2),
                                args.minimum_free_percent, args.sustained_host_samples)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    continuation = continuation_requested(args.trainer_args)
    if args.continue_existing_output and not continuation:
        parser.error("--continue-existing-output requires trainer --resume or "
                     "--stage development|final")
    existing = output.exists() and any(output.iterdir())
    if existing and not args.continue_existing_output:
        parser.error("Use a fresh trainer output path; existing output is "
                     "allowed only with --continue-existing-output for an "
                     "explicit resume or post-fit stage")
    if args.report.exists():
        parser.error("Use a fresh report path")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.report.with_suffix(".child.log")
    host_path = args.report.with_suffix(".host.jsonl")
    if log_path.exists() or host_path.exists():
        parser.error("Guard log paths already exist")
    command = [sys.executable, str(ROOT / "scripts" / args.entry_point), *trainer_args]
    report = {"status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "output": str(output),
              "continue_existing_output": bool(args.continue_existing_output),
              "continuation_requested": continuation, "child_log": str(log_path),
              "host_samples": str(host_path), "host_poll_seconds": 2,
              "profile_poll_seconds": 0.25,
              "missing_driver_sample_timeout_seconds": args.telemetry_timeout_seconds,
              "implementation_sha256": {
                  name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in ("scripts/run_capacity_preflight.py", "src/replica_lab/capacity_guard.py")},
              "notes": ["Sampled limits can miss transients and cannot guarantee no OOM",
                        "Host swap/pressure are system-wide, not attributable to this process",
                        "MPS driver and RSS counters overlap; do not add them",
                        "CPU fallback and OS/allocator safeguards are unchanged"]}
    child = None
    offset = 0
    tail = ""
    started = time.monotonic()
    last_driver_at = started
    guard = None
    try:
        baseline = host_reading()
        guard = CapacityGuard(policy, baseline)
        if guard.stop_reason:
            raise RuntimeError(guard.stop_reason)
        def drain_profile():
            nonlocal offset, tail, last_driver_at
            profile = output / "profile_samples.jsonl"
            if not profile.exists():
                return
            with profile.open() as stream:
                stream.seek(offset)
                tail += stream.read()
                offset = stream.tell()
            lines = tail.split("\n")
            tail = lines.pop()
            for line in lines:
                event = json.loads(line)
                before = guard.driver_samples
                guard.observe_profile(event)
                if guard.driver_samples > before:
                    last_driver_at = time.monotonic()
        with log_path.open("x") as log, host_path.open("x") as host:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "REPLICA_CAPACITY_GUARD": "1"})
            next_host = started
            while True:
                drain_profile()
                now = time.monotonic()
                if now >= next_host:
                    reading = host_reading()
                    host.write(json.dumps({"elapsed_seconds": now - started, **reading}) + "\n")
                    host.flush()
                    guard.observe_host(reading)
                    next_host = now + 2
                if guard.stop_reason:
                    stop_child(child)
                    break
                if child.poll() is not None:
                    break
                if now - last_driver_at > args.telemetry_timeout_seconds:
                    guard.stop_reason = "mps_driver_telemetry_stale_or_missing"
                    stop_child(child)
                    break
                time.sleep(0.25)
            report["child_exit_code"] = child.wait()
            # Child exit is observed after the preceding read. Consume its
            # final writes before claiming that every recorded peak was below
            # the limit, including when the process itself returned success.
            drain_profile()
            if tail.strip() and guard.stop_reason is None:
                guard.stop_reason = "incomplete_final_profile_record"
        if guard.driver_samples == 0 and guard.stop_reason is None:
            guard.stop_reason = "mps_driver_telemetry_missing"
        report["status"] = ("completed_within_declared_sampled_limits"
                            if child.returncode == 0 and not guard.stop_reason else "stopped_or_failed")
    except BaseException as error:
        if child is not None:
            stop_child(child)
        report.update(status="guard_or_child_failed", error_type=type(error).__name__, error=str(error))
        if child is not None:
            report["child_exit_code"] = child.returncode
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if guard is not None:
            report["guard"] = guard.report()
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "status": report["status"]}), flush=True)
    return 0 if report["status"] == "completed_within_declared_sampled_limits" else 1


if __name__ == "__main__":
    raise SystemExit(main())
