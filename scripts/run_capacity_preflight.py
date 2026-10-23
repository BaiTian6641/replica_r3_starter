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
                                                 "run_replica_session.py", "e3_speed_probe.py",
                                                 "run_e3_qualification.py", "run_e3_task.py", "run_e2_retrieval.py",
                                                 "run_e2_capacity.py"),
                        default="train_text.py")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--driver-limit-gib", type=float, default=40)
    parser.add_argument("--swap-growth-limit-mib", type=float, default=512)
    parser.add_argument("--minimum-free-percent", type=float, default=15)
    parser.add_argument("--sustained-host-samples", type=int, default=3)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This watchdog uses macOS host telemetry")
    try:
        trainer_args, output = training_arguments(args.trainer_args)
        policy = CapacityPolicy(int(args.driver_limit_gib * 1024**3),
                                int(args.swap_growth_limit_mib * 1024**2),
                                args.minimum_free_percent, args.sustained_host_samples)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    if args.report.exists() or (output.exists() and any(output.iterdir())):
        parser.error("Use fresh report and trainer output paths")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.report.with_suffix(".child.log")
    host_path = args.report.with_suffix(".host.jsonl")
    if log_path.exists() or host_path.exists():
        parser.error("Guard log paths already exist")
    command = [sys.executable, str(ROOT / "scripts" / args.entry_point), *trainer_args]
    report = {"status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "output": str(output), "child_log": str(log_path),
              "host_samples": str(host_path), "host_poll_seconds": 2,
              "profile_poll_seconds": 0.25, "missing_driver_sample_timeout_seconds": 30,
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
                if now - last_driver_at > 30:
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
