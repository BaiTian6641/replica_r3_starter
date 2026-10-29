"""External supervision of the already launched, single authorized attempt."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT_PID = 30475
START = 'Thu Oct 29 14:45:03 2026'
MARKER = 'scripts/launch_spark_intrinsic_readout_placement_v2.py'
# Count the launcher's admission time too; do not grant a fresh 7200 seconds.
DEADLINE = datetime.datetime(2026, 10, 29, 21, 45, 3, tzinfo=datetime.timezone.utc).timestamp()
OUT = Path('/Users/flare/replica_r3_starter/reports/intrinsic_memory_v3/native_readout03/external_wall_guard.json')
record = {'format': 'replica.external_attempt_wall_guard.v1', 'root_pid': ROOT_PID,
          'root_started_local': START, 'deadline_utc': '2026-10-29T21:45:03+00:00',
          'wall_seconds_max': 7200, 'poll_seconds': 1, 'shutdown_reserve_seconds': 5,
          'scope': 'launcher_admission_plus_parent_and_restart_child',
          'sealed_executable_changed': False,
          'attached_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'status': 'monitoring', 'deadline_triggered': False}

def save():
    temp = OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(OUT)

def processes():
    rows = subprocess.check_output(['ps', '-ax', '-o', 'pid=,ppid=,lstart=,command='], text=True)
    found = {}
    for line in rows.splitlines():
        parts = line.split(None, 7)
        if len(parts) == 8:
            found[int(parts[0])] = (int(parts[1]), ' '.join(parts[2:7]), parts[7])
    return found

save()
while True:
    rows = processes()
    root = rows.get(ROOT_PID)
    if root is None or root[1] != START or MARKER not in root[2]:
        record['status'] = 'root_exited_before_deadline'
        record['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        break
    if time.time() >= DEADLINE - 5:
        family = {ROOT_PID}
        while True:
            grown = family | {pid for pid, row in rows.items() if row[0] in family}
            if grown == family:
                break
            family = grown
        record.update(status='deadline_stop_requested', deadline_triggered=True,
                      terminated_pids=sorted(family))
        save()
        for pid in sorted(family, reverse=True):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(3)
        remaining = processes()
        for pid in sorted(family, reverse=True):
            # Recheck process start and command so no reused PID is signalled.
            if remaining.get(pid) == rows.get(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        record['status'] = 'deadline_enforced'
        record['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        break
    time.sleep(1)
print(json.dumps(record))
