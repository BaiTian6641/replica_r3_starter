"""Portable POSIX subprocess tests; no model, accelerator, experiment or source loads."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from replica_lab.wall_deadline_v1 import WallDeadlineError, WallPolicy, run_wall_bounded

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/run_wall_bounded_command_v1.py"


def run(tmp_path, code, *, wall=2.0, grace=0.2, reap=0.15, args=()):
    return run_wall_bounded([sys.executable, "-u", "-c", code, *map(str, args)], cwd=tmp_path,
        output=tmp_path / "output", policy=WallPolicy(wall, grace, reap, 0.01),
        launch_authorized=True, authorization_identity="synthetic_CPU_test_only")


def heartbeat_stopped(path):
    first = path.stat().st_mtime_ns
    time.sleep(.12)
    return path.stat().st_mtime_ns == first


def wait_file(path, timeout=2.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if path.is_file() and path.stat().st_size:
            return
        time.sleep(0.01)
    raise AssertionError(f"child did not create {path}")


def tree_code(pidfile):
    leaf = """
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(PIDFILE).write_text(str(os.getpid()))
print('restart-ready', flush=True)
while True:
    Path(PIDFILE + '.heartbeat').touch()
    time.sleep(.02)
""".replace("PIDFILE", repr(str(pidfile)))
    parent = f"""
import signal, subprocess, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('parent-ready', flush=True)
subprocess.Popen([sys.executable, '-u', '-c', {leaf!r}])
while True: time.sleep(.02)
"""
    return f"""
import signal, subprocess, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('preflight-ready', flush=True)
time.sleep(.05)
subprocess.Popen([sys.executable, '-u', '-c', {parent!r}])
while True: time.sleep(.02)
"""


def test_normal_exit_preserves_exact_raw_output_and_child_failure(tmp_path):
    result = run(tmp_path, "import os; os.write(1, b'raw\\xff\\n'); os.write(2,b'error\\x00'); raise SystemExit(7)")
    assert result.exit_code == 7
    assert result.receipt["status"] == "child_failed"
    assert result.receipt["child_exit_code"] == 7
    assert result.receipt["direct_child_reaped"]
    assert (tmp_path / "output/child.stdout").read_bytes() == b"raw\xff\n"
    assert (tmp_path / "output/child.stderr").read_bytes() == b"error\x00"
    assert json.loads((tmp_path / "output/receipt.json").read_text()) == result.receipt
    assert result.receipt["authorization_validated_by_this_supervisor"] is False
    assert result.receipt["signals_sent"] == []


def test_success_has_no_shutdown_and_never_interpolates_shell_arguments(tmp_path):
    literal = "$(touch should-not-exist); echo unsafe"
    result = run(tmp_path, "import sys; print(sys.argv[1])", args=(literal,))
    assert result.exit_code == 0 and result.receipt["status"] == "completed"
    assert (tmp_path / "output/child.stdout").read_text() == literal + "\n"
    assert not (tmp_path / "should-not-exist").exists()
    assert result.receipt["process_group_exit_confirmed"]


def test_monotonic_deadline_covers_preflight_parent_and_restart_with_grace_inside_ceiling(tmp_path, monkeypatch):
    pidfile = tmp_path / "restart.pid"
    # A civil clock jump must have no influence on the deadline.
    monkeypatch.setattr("replica_lab.wall_deadline_v1.time.time", lambda: 0.0)
    start = time.monotonic()
    result = run(tmp_path, tree_code(pidfile), wall=0.8, grace=0.2, reap=0.2)
    elapsed = time.monotonic()-start
    assert result.receipt["termination_trigger"] == "wall_deadline"
    assert result.receipt["status"] in ("wall_deadline", "cleanup_unconfirmed")
    assert result.exit_code == (124 if result.receipt["process_group_exit_confirmed"] else 125)
    assert result.receipt["useful_execution_seconds_max"] == pytest.approx(0.4)
    assert elapsed < 1.2  # OS scheduling tolerance, never an additional grace budget
    assert result.receipt["elapsed_seconds"] < 1.1
    assert result.receipt["child_exit_code"] == -signal.SIGKILL
    assert result.receipt["direct_child_reaped"]
    signals = result.receipt["signals_sent"]
    assert [item["signal"] for item in signals] == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert signals[0]["elapsed_seconds"] >= 0.39
    assert signals[-1]["elapsed_seconds"] <= 0.75
    assert all(item["process_group"] == result.receipt["child_pid"] for item in signals)
    assert heartbeat_stopped(Path(str(pidfile) + ".heartbeat"))
    output = (tmp_path / "output/child.stdout").read_text()
    assert all(phase in output for phase in ("preflight-ready", "parent-ready", "restart-ready"))


def test_interrupt_supervisor_terminates_actual_inherited_descendants(tmp_path):
    pidfile = tmp_path / "restart.pid"
    command = [sys.executable, str(CLI), "--launch", "--authorization-identity", "synthetic-interruption",
        "--wall-seconds", "3", "--shutdown-grace-seconds", ".2", "--reap-reserve-seconds", ".2",
        "--cwd", str(tmp_path), "--output", str(tmp_path / "output"), "--",
        sys.executable, "-u", "-c", tree_code(pidfile)]
    supervisor = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_file(pidfile)
        supervisor.send_signal(signal.SIGTERM)
        stdout, stderr = supervisor.communicate(timeout=4)
        receipt = json.loads((tmp_path / "output/receipt.json").read_text())
        assert receipt["termination_trigger"] == "interrupted"
        assert receipt["status"] in ("interrupted", "cleanup_unconfirmed")
        assert supervisor.returncode == (128 + signal.SIGTERM if receipt["process_group_exit_confirmed"] else 125), (stdout, stderr)
        assert receipt["interruptions"][0]["signal"] == signal.SIGTERM
        assert receipt["child_exit_code"] == -signal.SIGKILL
        assert receipt["elapsed_seconds"] < 3.2
        assert heartbeat_stopped(Path(str(pidfile) + ".heartbeat"))
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()


def test_successful_direct_parent_cannot_leave_live_descendants(tmp_path):
    pidfile = tmp_path / "orphan.pid"
    leaf = "import os,time; from pathlib import Path; Path(%r).write_text(str(os.getpid()));\nwhile True: Path(%r).touch(); time.sleep(.02)" % (str(pidfile), str(pidfile) + ".heartbeat")
    code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(.1)" % leaf
    result = run(tmp_path, code)
    assert result.receipt["termination_trigger"] == "descendants_after_child_exit"
    assert result.receipt["status"] in ("descendants_after_child_exit", "cleanup_unconfirmed")
    assert result.exit_code == 125 and result.receipt["child_exit_code"] == 0
    assert heartbeat_stopped(Path(str(pidfile) + ".heartbeat"))


@pytest.mark.parametrize("options", [
    {"wall_seconds": float("nan")}, {"wall_seconds": float("inf")}, {"wall_seconds": True},
    {"wall_seconds": 0}, {"wall_seconds": 1, "shutdown_grace_seconds": 1},
    {"wall_seconds": 1, "shutdown_grace_seconds": -1},
    {"wall_seconds": 1, "shutdown_grace_seconds": .2, "reap_reserve_seconds": 0},
    {"wall_seconds": 1, "shutdown_grace_seconds": .2, "poll_seconds": 1},
])
def test_invalid_budget_refused(options):
    with pytest.raises(WallDeadlineError):
        WallPolicy(**options)


@pytest.mark.parametrize("damage", ["authorization", "identity", "string", "relative_executable", "nul", "relative_cwd", "existing_output", "env"])
def test_unsafe_inputs_refused_before_process_creation(tmp_path, monkeypatch, damage):
    arguments = dict(command=[sys.executable, "-c", "pass"], cwd=tmp_path, output=tmp_path / "output",
                     policy=WallPolicy(2, .2, .1), launch_authorized=True, authorization_identity="synthetic")
    if damage == "authorization": arguments["launch_authorized"] = False
    if damage == "identity": arguments["authorization_identity"] = ""
    if damage == "string": arguments["command"] = "echo unsafe"
    if damage == "relative_executable": arguments["command"] = ["python", "-c", "pass"]
    if damage == "nul": arguments["command"][-1] = "bad\x00argument"
    if damage == "relative_cwd": arguments["cwd"] = "."
    if damage == "existing_output": arguments["output"].mkdir()
    if damage == "env": arguments["env"] = {"BAD=KEY": "value"}
    monkeypatch.setattr("replica_lab.wall_deadline_v1.subprocess.Popen", lambda *a, **kw: pytest.fail("must not spawn"))
    with pytest.raises(WallDeadlineError):
        run_wall_bounded(**arguments)


def test_cli_never_executes_without_explicit_launch(tmp_path):
    result = subprocess.run([sys.executable, str(CLI), "--authorization-identity", "synthetic",
        "--wall-seconds", "2", "--shutdown-grace-seconds", ".2", "--reap-reserve-seconds", ".2",
        "--cwd", str(tmp_path), "--output", str(tmp_path / "output"), "--",
        sys.executable, "-c", "raise SystemExit('must not execute')"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "explicit prior launch authorization" in result.stderr
    assert not (tmp_path / "output").exists()


def test_signal_handlers_restored_and_escaped_sessions_are_explicitly_unsupported(tmp_path):
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    result = run(tmp_path, "pass")
    assert {sig: signal.getsignal(sig) for sig in old} == old
    assert result.receipt["containment_limit"] == "descendants_that_change_session_or_process_group_are_unsupported"
    assert result.receipt["hard_realtime_guarantee"] is False


def test_zero_grace_kills_only_owned_group_and_preserves_unrelated_process(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
    try:
        result = run(tmp_path, "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); time.sleep(10)",
                     wall=.4, grace=0, reap=.15)
        assert result.receipt["termination_trigger"] == "wall_deadline"
        assert [item["signal"] for item in result.receipt["signals_sent"]] == [signal.SIGKILL]
        assert result.receipt["child_exit_code"] == -signal.SIGKILL
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=1)


def test_spawn_error_gets_machine_receipt_without_claiming_execution(tmp_path):
    invalid = tmp_path / "not-an-executable"
    invalid.write_bytes(b"invalid executable bytes")
    invalid.chmod(0o700)
    result = run_wall_bounded([str(invalid)], cwd=tmp_path, output=tmp_path / "output",
        policy=WallPolicy(2, .2, .1), launch_authorized=True, authorization_identity="synthetic")
    assert result.receipt["status"] == "spawn_failed" and result.exit_code == 126
    assert result.receipt["child_pid"] is None
    assert result.receipt["error"] is not None
    assert (tmp_path / "output/receipt.json").is_file()


def test_unconfirmed_early_cleanup_uses_local_reserve_not_remaining_long_budget(tmp_path, monkeypatch):
    clock = [100.0]
    class ReapedChild:
        pid = 777777
        def poll(self):
            return 0
    def denied_group(group, signum):
        assert group == ReapedChild.pid
        raise PermissionError("synthetic orphan probe denied")
    monkeypatch.setattr("replica_lab.wall_deadline_v1.subprocess.Popen", lambda *a, **kw: ReapedChild())
    monkeypatch.setattr("replica_lab.wall_deadline_v1.os.killpg", denied_group)
    monkeypatch.setattr("replica_lab.wall_deadline_v1.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("replica_lab.wall_deadline_v1.time.sleep", lambda duration: clock.__setitem__(0, clock[0]+duration))
    result = run_wall_bounded([sys.executable, "-c", "pass"], cwd=tmp_path, output=tmp_path / "output",
        policy=WallPolicy(7200, .2, .1, .05), launch_authorized=True, authorization_identity="synthetic")
    assert result.receipt["status"] == "cleanup_unconfirmed" and result.exit_code == 125
    assert result.receipt["termination_trigger"] == "descendants_after_child_exit"
    assert result.receipt["elapsed_seconds"] == pytest.approx(.3)
    assert len(result.receipt["signal_errors"]) == 3


def test_cancellation_recorded_before_spawn_does_not_start_work(tmp_path, monkeypatch):
    real_signal = signal.signal
    injected = [False]
    def install(signum, handler):
        previous = real_signal(signum, handler)
        if signum == signal.SIGINT and callable(handler) and not injected[0]:
            injected[0] = True
            handler(signal.SIGINT, None)
        return previous
    monkeypatch.setattr("replica_lab.wall_deadline_v1.signal.signal", install)
    monkeypatch.setattr("replica_lab.wall_deadline_v1.subprocess.Popen", lambda *a, **kw: pytest.fail("cancelled work must not spawn"))
    result = run(tmp_path, "pass")
    assert result.receipt["status"] == "interrupted" and result.exit_code == 130
    assert result.receipt["child_pid"] is None and result.receipt["signals_sent"] == []
