"""Best-effort single-process memory and synchronized phase timing telemetry.

MPS counters are sampled, so their maxima can miss short-lived allocations.
On Apple unified memory, process RSS and MPS driver memory overlap: never add
them to estimate physical use. Telemetry failures are recorded without failing
the experiment. No memory limits, fallback settings, or allocator policy change.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _ru_maxrss_bytes(value, platform=None):
    """Darwin reports bytes; Linux and other supported Unix hosts report KiB."""
    return int(value) * (1 if (platform or sys.platform) == "darwin" else 1024)


def _parse_swapusage(text):
    result = {}
    for name, number, unit in re.findall(
            r"(total|used|free)\s*=\s*([\d.]+)([BKMGTP])", text, re.IGNORECASE):
        result[name.lower() + "_bytes"] = int(
            float(number) * 1024 ** "BKMGTP".index(unit.upper()))
    return result


class RunProfiler:
    """Sample memory in the background; synchronize only main-thread marks.

    ``start()`` returns self, ``mark(phase, **fields)`` returns a phase event,
    and idempotent ``stop()`` returns the persisted summary. The context manager
    always stops on exit and never suppresses an experiment exception. Output
    files are ``profile_samples.jsonl`` and ``profile_summary.json``.
    """

    def __init__(self, device, output_dir, sample_interval=0.25):
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        self.device = str(device)
        self.device_type = self.device.split(":", 1)[0]
        self.output_dir = Path(output_dir)
        self.sample_interval = float(sample_interval)
        self._lock = threading.RLock()
        self._sample_lock = threading.Lock()
        self._done = threading.Event()
        self._thread = None
        self._stream = None
        self._torch = None
        self._backend_available = False
        self._started = False
        self._closed = False
        self._summary = None
        self._errors = {}
        self._phases = []
        self._peaks = {}
        self._sample_count = 0
        self._last_rss_at = float("-inf")
        self._last_rss_bytes = None
        self._phase = "start"
        self._last_mark_elapsed = 0.0

    def _error(self, operation, exc):
        with self._lock:
            message = f"{type(exc).__name__}: {exc}"
            key = (operation, message)
            now = _timestamp()
            if key not in self._errors:
                self._errors[key] = {"operation": operation, "message": message,
                                     "count": 0, "first_at": now}
            self._errors[key]["count"] += 1
            self._errors[key]["last_at"] = now

    def _guard(self, operation, callback):
        try:
            return callback()
        except Exception as exc:
            self._error(operation, exc)
            return None

    def _write(self, event):
        with self._lock:
            if self._stream is not None and not self._closed:
                self._guard("write_jsonl", lambda: (
                    self._stream.write(json.dumps(event, default=str) + "\n"),
                    self._stream.flush()))

    def _command(self, args):
        return subprocess.run(args, capture_output=True, text=True, timeout=2,
                              check=True).stdout.strip()

    def _host_snapshot(self):
        result = {"timestamp": _timestamp(), "platform": sys.platform}
        if sys.platform == "darwin":
            swap = self._guard("sysctl_vm_swapusage", lambda: self._command(
                ["/usr/sbin/sysctl", "-n", "vm.swapusage"]))
            pressure = self._guard("memory_pressure_query", lambda: self._command(
                ["/usr/bin/memory_pressure", "-Q"]))
            result["swapusage_raw"] = swap
            result["swapusage"] = _parse_swapusage(swap or "")
            result["memory_pressure_raw"] = pressure
            free = re.search(r"memory free percentage:\s*([\d.]+)%", pressure or "",
                             re.IGNORECASE)
            result["memory_free_percent"] = float(free.group(1)) if free else None
        return result

    def _synchronize(self):
        if self._torch is None or not self._backend_available:
            return
        if self.device_type == "mps":
            self._guard("mps_synchronize", lambda: self._torch.mps.synchronize())
        elif self.device_type == "cuda":
            self._guard("cuda_synchronize", lambda: self._torch.cuda.synchronize(self.device))

    def _sample(self, *, force_rss=False):
        with self._sample_lock:
            event = {"type": "sample", "timestamp": _timestamp(),
                     "elapsed_seconds": time.monotonic() - self._started_mono,
                     "phase": self._phase}
            event["rss_peak_process_lifetime_bytes"] = self._guard(
                "resource_ru_maxrss", lambda: _ru_maxrss_bytes(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            now = time.monotonic()
            if force_rss or now - self._last_rss_at >= 2:
                self._last_rss_at = now
                self._last_rss_bytes = self._guard("ps_current_rss", lambda: int(
                    self._command(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())])) * 1024)
            event["rss_current_bytes"] = self._last_rss_bytes
            event["rss_measurement_age_seconds"] = max(0, now - self._last_rss_at)
            if self._torch is not None and self._backend_available:
                if self.device_type == "mps":
                    for label, method in (
                            ("mps_allocated_bytes", "current_allocated_memory"),
                            ("mps_driver_bytes", "driver_allocated_memory")):
                        event[label] = self._guard(label, lambda method=method: int(
                            getattr(self._torch.mps, method)()))
                elif self.device_type == "cuda":
                    for label, method in (("cuda_allocated_bytes", "memory_allocated"),
                                          ("cuda_reserved_bytes", "memory_reserved")):
                        event[label] = self._guard(label, lambda method=method: int(
                            getattr(self._torch.cuda, method)(self.device)))
            with self._lock:
                if self._closed:
                    return event
                self._sample_count += 1
                peak_names = {"rss_peak_process_lifetime_bytes": "rss_peak_process_lifetime_bytes",
                              "rss_current_bytes": "rss_peak_sampled_bytes",
                              "mps_allocated_bytes": "mps_allocated_peak_sampled_bytes",
                              "mps_driver_bytes": "mps_driver_peak_sampled_bytes",
                              "cuda_allocated_bytes": "cuda_allocated_peak_sampled_bytes",
                              "cuda_reserved_bytes": "cuda_reserved_peak_sampled_bytes"}
                for name, peak_name in peak_names.items():
                    if event.get(name) is not None:
                        self._peaks[peak_name] = max(self._peaks.get(peak_name, 0), event[name])
                self._write(event)
            return event

    def _sample_loop(self):
        while not self._done.wait(self.sample_interval):
            self._guard("background_sample", self._sample)

    def start(self):
        if self._started:
            return self
        self._started = True
        self._started_at = _timestamp()
        self._started_mono = time.monotonic()
        self._guard("create_output_directory", lambda: self.output_dir.mkdir(parents=True, exist_ok=True))
        self._stream = self._guard("open_jsonl", lambda: (
            self.output_dir / "profile_samples.jsonl").open("w", encoding="utf-8"))
        if self.device_type in {"mps", "cuda"} and self._torch is None:
            self._torch = self._guard("import_torch", lambda: importlib.import_module("torch"))
        if self._torch is not None and self.device_type in {"mps", "cuda"}:
            available = self._guard(self.device_type + "_is_available", lambda: (
                self._torch.backends.mps.is_available() if self.device_type == "mps"
                else self._torch.cuda.is_available()))
            self._backend_available = available is True
            if not self._backend_available:
                self._error(self.device_type + "_unavailable", RuntimeError(
                    "Accelerator unavailable; native memory and synchronization APIs were skipped"))
        self._recommended_max_bytes = None
        if self._backend_available and self.device_type == "mps":
            self._recommended_max_bytes = self._guard(
                "mps_recommended_max_memory", lambda: self._torch.mps.recommended_max_memory())
        if self._backend_available and self.device_type == "cuda":
            self._guard("cuda_reset_peak_memory_stats", lambda:
                        self._torch.cuda.reset_peak_memory_stats(self.device))
        self._host_start = self._host_snapshot()
        self._write({"type": "host_start", **self._host_start})
        self._guard("initial_sample", lambda: self._sample(force_rss=True))
        self._thread = threading.Thread(target=self._sample_loop, name="run-memory-profiler",
                                        daemon=True)
        self._guard("start_sampling_thread", self._thread.start)
        return self

    def mark(self, phase, **fields):
        self.start()
        before_sync = time.monotonic()
        self._synchronize()
        synchronize_seconds = time.monotonic() - before_sync
        elapsed = time.monotonic() - self._started_mono
        self._phase = str(phase)
        sample = self._guard("phase_sample", self._sample)
        event = {"type": "phase", "phase": str(phase), "timestamp": _timestamp(),
                 "elapsed_seconds": elapsed,
                 "seconds_since_previous_mark": elapsed - self._last_mark_elapsed,
                 "synchronize_seconds": synchronize_seconds,
                 "memory_sample": sample, "fields": fields}
        self._last_mark_elapsed = elapsed
        with self._lock:
            if not self._closed:
                self._phases.append(event)
                self._write(event)
        return event

    def stop(self):
        if self._summary is not None:
            return self._summary
        self.start()
        self._done.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                self._error("stop_sampling_thread", TimeoutError("sampler did not stop within 3 seconds"))
        self._synchronize()
        # Avoid waiting indefinitely on a telemetry thread/backend call.
        if self._thread is None or not self._thread.is_alive():
            self._guard("final_sample", lambda: self._sample(force_rss=True))
        host_end = self._host_snapshot()
        self._write({"type": "host_end", **host_end})
        if self._backend_available and self.device_type == "cuda":
            for key, method in (("cuda_allocated_peak_bytes", "max_memory_allocated"),
                                ("cuda_reserved_peak_bytes", "max_memory_reserved")):
                value = self._guard(key, lambda method=method:
                                    int(getattr(self._torch.cuda, method)(self.device)))
                if value is not None:
                    self._peaks[key] = value
        with self._lock:
            self._closed = True
            if self._stream is not None:
                self._guard("close_jsonl", self._stream.close)
            self._summary = {
                "schema_version": 1, "device": self.device,
                "started_at": self._started_at, "ended_at": _timestamp(),
                "elapsed_seconds": time.monotonic() - self._started_mono,
                "sample_interval_seconds": self.sample_interval,
                "rss_sample_interval_seconds": 2.0,
                "sample_count": self._sample_count, "memory": dict(self._peaks),
                "mps_recommended_max_memory_bytes": self._recommended_max_bytes,
                "host_start": self._host_start, "host_end": host_end,
                "phases": list(self._phases), "errors": list(self._errors.values()),
                "notes": [
                    "All memory values are bytes; GiB = bytes / 2**30.",
                    "MPS and RSS sampled peaks may miss transient allocations.",
                    "ru_maxrss is the process lifetime high-water mark, not a phase or run reset counter.",
                    "Apple unified-memory MPS driver memory and process RSS overlap; do not add them.",
                    "CUDA peak counters are reset at profiler start; timing marks synchronize the device.",
                ],
            }
            self._guard("write_summary", lambda: (
                self.output_dir / "profile_summary.json").write_text(
                    json.dumps(self._summary, indent=2, default=str) + "\n", encoding="utf-8"))
            self._summary["errors"] = list(self._errors.values())
            return self._summary

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
