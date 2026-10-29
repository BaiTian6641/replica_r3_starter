"""POSIX command-tree wall supervisor for separately bound future launches.

The explicit caller must already hold launch authorization. This module does not
validate or create experiment authorization. Start the *outermost launcher* here
so preflight, fitting and restart share one monotonic budget. Descendants must
inherit the process group; intentional setsid/setpgid escape is unsupported.
The existing placement guard and restart runner inherit their parent's group.

Shutdown grace and child reap reserve are deducted from useful execution time.
OS scheduling and uninterruptible kernel work cannot be made hard real-time;
actual elapsed time and any overrun or unconfirmed group exit are reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from collections.abc import Sequence

FORMAT = "replica.command_tree.wall_deadline.v1"


class WallDeadlineError(ValueError):
    pass


@dataclass(frozen=True)
class WallPolicy:
    wall_seconds: float
    shutdown_grace_seconds: float = 10.0
    reap_reserve_seconds: float = 1.0
    poll_seconds: float = 0.05

    def __post_init__(self):
        values = asdict(self)
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in values.values())
                or self.wall_seconds <= 0 or self.shutdown_grace_seconds < 0
                or self.reap_reserve_seconds <= 0
                or self.shutdown_grace_seconds + self.reap_reserve_seconds >= self.wall_seconds
                or not 0 < self.poll_seconds <= min(0.25, self.reap_reserve_seconds)):
            raise WallDeadlineError("invalid finite wall budget or reserved shutdown interval")


@dataclass(frozen=True)
class WallResult:
    receipt: dict
    exit_code: int


def _validate(command, cwd, output, launch_authorized, authorization_identity):
    if launch_authorized is not True or not isinstance(authorization_identity, str) or not authorization_identity.strip():
        raise WallDeadlineError("explicit prior launch authorization identity is required")
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        raise WallDeadlineError("POSIX main-thread process-group supervision is required")
    if (not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command
            or any(not isinstance(value, str) or not value or "\x00" in value for value in command)):
        raise WallDeadlineError("command must be a nonempty explicit argv sequence")
    executable = Path(command[0])
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise WallDeadlineError("command executable must be an existing absolute executable path")
    cwd, output = Path(cwd), Path(output)
    if not cwd.is_absolute() or not cwd.is_dir():
        raise WallDeadlineError("explicit existing absolute working directory required")
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.parent.is_dir() or output.parent.is_symlink()):
        raise WallDeadlineError("fresh absolute output directory with an existing real parent required")
    if output.resolve().is_relative_to(cwd.resolve() / ".git"):
        raise WallDeadlineError("output directory cannot be inside Git metadata")
    return list(command), cwd.resolve(), output


def run_wall_bounded(command, *, cwd, output, policy: WallPolicy,
                     launch_authorized=False, authorization_identity=None, env=None) -> WallResult:
    """Run one already-authorized argv, recording raw output and machine outcome.

    No shell is introduced. Environment overrides are accepted only as a complete
    explicit string mapping; absent overrides inherit the caller's environment.
    SIGINT/SIGTERM received by this supervisor cause group cleanup, not a retry.
    Existing receipt/output paths are never overwritten. No command runs by default.
    """
    if not isinstance(policy, WallPolicy):
        raise WallDeadlineError("explicit WallPolicy is required")
    command, cwd, output = _validate(command, cwd, output, launch_authorized, authorization_identity)
    if env is not None and (not isinstance(env, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or not key
            or "=" in key or "\x00" in key or "\x00" in value for key, value in env.items())):
        raise WallDeadlineError("environment must be a complete valid string mapping")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    deadline = started + policy.wall_seconds
    shutdown_start = deadline - policy.shutdown_grace_seconds - policy.reap_reserve_seconds
    output.mkdir(exist_ok=False)
    received = []
    previous = {}
    child = None
    group = None
    group_gone = False
    signals_sent = []
    signal_errors = []
    probe_errors = []
    reason = "supervisor_error"
    error = None

    def interrupted(signum, frame):
        received.append({"signal": signum, "elapsed_seconds": time.monotonic()-started})

    def alive():
        nonlocal group_gone
        if group is None or group_gone:
            return False
        try:
            os.killpg(group, 0)
            return True
        except ProcessLookupError:
            group_gone = True
            return False
        except PermissionError as exc:
            if not probe_errors:
                probe_errors.append({"type": type(exc).__name__, "message": str(exc)})
            return True  # permission loss is not evidence that descendants exited

    def send(signum):
        if not alive():
            return
        try:
            os.killpg(group, signum)
            signals_sent.append({"signal": signum, "name": signal.Signals(signum).name,
                                 "process_group": group, "elapsed_seconds": time.monotonic()-started})
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            signal_errors.append({"signal": signum, "message": str(exc)})

    def wait_until(until):
        while time.monotonic() < until:
            if child is not None:
                child.poll()  # reap the direct child; grandchildren remain owned by their parents
            if not alive():
                return
            time.sleep(min(policy.poll_seconds, max(0.0, until-time.monotonic())))

    def cleanup():
        now = time.monotonic()
        kill_at = min(now + policy.shutdown_grace_seconds, deadline-policy.reap_reserve_seconds)
        if now < kill_at:
            send(signal.SIGINT)
            wait_until(now + (kill_at-now)/2)
            if alive():
                send(signal.SIGTERM)
                wait_until(kill_at)
        if alive():
            send(signal.SIGKILL)
        # Early interruption must not spend the unused experiment budget
        # waiting on an unobservable orphan group. Reap observation has its
        # own reserve, still capped by the original total deadline.
        wait_until(min(deadline, max(time.monotonic(), kill_at) + policy.reap_reserve_seconds))
        if child is not None:
            child.poll()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        with (output / "child.stdout").open("xb") as stdout, (output / "child.stderr").open("xb") as stderr:
            if received:
                reason = "interrupted"
            elif time.monotonic() >= shutdown_start:
                reason = "deadline_before_spawn"
            else:
                child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, shell=False, start_new_session=True)
                group = child.pid  # setsid makes the child the session and process-group leader
                while True:
                    code = child.poll()
                    if received:
                        reason = "interrupted"
                        break
                    if time.monotonic() >= shutdown_start:
                        reason = "wall_deadline"
                        break
                    if code is not None:
                        reason = "descendants_after_child_exit" if alive() else ("completed" if code == 0 else "child_failed")
                        break
                    time.sleep(min(policy.poll_seconds, max(0.0, shutdown_start-time.monotonic())))
                if alive():
                    cleanup()
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        reason = "spawn_failed" if child is None else "supervisor_error"
        if child is not None:
            try:
                cleanup()
            except BaseException as cleanup_error:
                error["cleanup_error"] = {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    elapsed = time.monotonic()-started
    child_exit = child.poll() if child is not None else None
    try:
        group_alive = alive()
    except OSError:
        group_alive = True
    exit_code = {"completed": 0, "wall_deadline": 124, "deadline_before_spawn": 124,
                 "descendants_after_child_exit": 125, "spawn_failed": 126,
                 "supervisor_error": 125}.get(reason)
    if reason == "interrupted":
        exit_code = 128 + received[0]["signal"]
    elif reason == "child_failed":
        exit_code = child_exit if child_exit >= 0 else 128-child_exit
    trigger = reason
    if group_alive or signal_errors or (child is not None and child_exit is None):
        reason, exit_code = "cleanup_unconfirmed", 125
    receipt = {"format": FORMAT, "status": reason, "termination_trigger": trigger, "exit_code": min(255, exit_code),
        "started_at_utc": started_utc,
        "command": command, "command_sha256": hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest(),
        "cwd": str(cwd), "policy": asdict(policy), "scope": "entire_inherited_process_group_including_preflight_parent_restart",
        "authorization_identity": authorization_identity, "authorization_validated_by_this_supervisor": False,
        "child_pid": child.pid if child is not None else None, "owned_process_group": group,
        "child_exit_code": child_exit, "direct_child_reaped": child is None or child_exit is not None,
        "process_group_exit_confirmed": not group_alive, "signals_sent": signals_sent,
        "signal_errors": signal_errors, "group_probe_errors": probe_errors,
        "interruptions": received, "elapsed_seconds": elapsed,
        "useful_execution_seconds_max": policy.wall_seconds-policy.shutdown_grace_seconds-policy.reap_reserve_seconds,
        "deadline_overrun_seconds": max(0.0, elapsed-policy.wall_seconds), "error": error,
        "containment_limit": "descendants_that_change_session_or_process_group_are_unsupported",
        "hard_realtime_guarantee": False,
        "stdout": "child.stdout", "stderr": "child.stderr"}
    for stream in ("stdout", "stderr"):
        file = output / receipt[stream]
        receipt[stream + "_bytes"] = file.stat().st_size if file.exists() else 0
    # The receipt is emitted after shutdown, so it reports measured overrun and
    # final observed status without pretending JSON I/O is hard real-time.
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return WallResult(receipt=receipt, exit_code=receipt["exit_code"])
