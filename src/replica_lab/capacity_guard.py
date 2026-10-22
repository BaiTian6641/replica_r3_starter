"""Declared, sampled stop policy for one bounded Mac capacity experiment.

This policy does not change allocator limits or provide an OOM guarantee. Host
swap and memory-pressure readings describe the whole machine, not just training.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re


def parse_host_readings(swap_text: str, pressure_text: str) -> dict:
    swap = re.search(r"used\s*=\s*([\d.]+)([BKMGTP])", swap_text, re.IGNORECASE)
    free = re.search(r"memory free percentage:\s*([\d.]+)%", pressure_text, re.IGNORECASE)
    if swap is None or free is None:
        raise ValueError("Host swap/pressure readings unavailable or unrecognized")
    used = float(swap.group(1)) * 1024 ** "BKMGTP".index(swap.group(2).upper())
    percentage = float(free.group(1))
    if not math.isfinite(used) or not math.isfinite(percentage) or not 0 <= percentage <= 100:
        raise ValueError("Invalid host telemetry")
    return {"swap_used_bytes": int(used), "memory_free_percent": percentage}


@dataclass(frozen=True)
class CapacityPolicy:
    driver_limit_bytes: int = 40 * 1024**3
    swap_growth_limit_bytes: int = 512 * 1024**2
    minimum_free_percent: float = 15.0
    sustained_host_samples: int = 3

    def __post_init__(self):
        for value in (self.driver_limit_bytes, self.swap_growth_limit_bytes,
                      self.sustained_host_samples):
            if type(value) is not int or value <= 0:
                raise ValueError("Capacity thresholds must be positive integers")
        if (not math.isfinite(self.minimum_free_percent)
                or not 0 <= self.minimum_free_percent <= 100):
            raise ValueError("Invalid memory-free threshold")


class CapacityGuard:
    def __init__(self, policy: CapacityPolicy, baseline: dict):
        self.policy = policy
        self.baseline = self._host(baseline)
        self.swap_streak = self.pressure_streak = 0
        self.peak_driver_bytes = 0
        self.peak_swap_growth_bytes = 0
        self.minimum_observed_free_percent = self.baseline["memory_free_percent"]
        self.host_samples = 0
        self.driver_samples = 0
        self.stop_reason = None
        if self.baseline["memory_free_percent"] <= policy.minimum_free_percent:
            self.stop_reason = "initial_host_free_percentage_at_or_below_threshold"

    @staticmethod
    def _host(reading):
        swap, free = reading.get("swap_used_bytes"), reading.get("memory_free_percent")
        if (type(swap) is not int or swap < 0 or isinstance(free, bool)
                or not isinstance(free, (int, float)) or not math.isfinite(free)
                or not 0 <= free <= 100):
            raise ValueError("Missing or invalid host reading")
        return {"swap_used_bytes": swap, "memory_free_percent": free}

    def observe_host(self, reading):
        reading = self._host(reading)
        self.host_samples += 1
        growth = max(0, reading["swap_used_bytes"] - self.baseline["swap_used_bytes"])
        self.peak_swap_growth_bytes = max(self.peak_swap_growth_bytes, growth)
        self.minimum_observed_free_percent = min(
            self.minimum_observed_free_percent, reading["memory_free_percent"])
        self.swap_streak = self.swap_streak + 1 if growth >= self.policy.swap_growth_limit_bytes else 0
        self.pressure_streak = (self.pressure_streak + 1
                               if reading["memory_free_percent"] <= self.policy.minimum_free_percent else 0)
        if self.stop_reason is None:
            if self.swap_streak >= self.policy.sustained_host_samples:
                self.stop_reason = "sustained_host_swap_growth"
            elif self.pressure_streak >= self.policy.sustained_host_samples:
                self.stop_reason = "sustained_low_host_free_percentage"
        return self.stop_reason

    def observe_profile(self, event):
        if event.get("type") == "phase":
            event = event.get("memory_sample") or {}
        value = event.get("mps_driver_bytes")
        if value is None:
            return self.stop_reason
        if type(value) is not int or value < 0:
            raise ValueError("Invalid MPS driver sample")
        self.driver_samples += 1
        self.peak_driver_bytes = max(self.peak_driver_bytes, value)
        if self.stop_reason is None and value >= self.policy.driver_limit_bytes:
            self.stop_reason = "sampled_mps_driver_limit"
        return self.stop_reason

    def report(self):
        return {"policy": asdict(self.policy), "baseline": self.baseline,
                "host_samples": self.host_samples, "driver_samples": self.driver_samples,
                "peak_driver_bytes": self.peak_driver_bytes,
                "peak_swap_growth_bytes": self.peak_swap_growth_bytes,
                "minimum_observed_free_percent": self.minimum_observed_free_percent,
                "stop_reason": self.stop_reason}
