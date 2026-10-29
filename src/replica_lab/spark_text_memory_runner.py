"""Bounded Spark text-memory fit/development/restart runner.

The runner is intentionally split into an orchestration layer and a backend
layer.  Orchestration owns fixture timing, arm information, work accounting,
checkpoint retention and evidence publication.  Backends own the decoder and
tokenizer.  The only native backend admitted here is the sealed official FP8
source executed through the Spark loader/adapter; the active adapter does not
publish a native-FP8 admission contract, so native construction refuses before
any model or tokenizer is loaded.  There is no BF16, dequantized-derivative or
CPU-fallback substitution for the native arm.

``cpu_fixture`` is a deterministic tiny Spark-shaped model used to exercise the
orchestration and work-accounting contracts.  It is never native-admissible and
its results are not protocol evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from . import artifact_publication as publication
from .architecture.spark2_5 import (
    BackendAdmissionError, Spark2_5ContractError, OFFICIAL_NATIVE_FP8,
    admit_load, evaluate_admission, load_spark2_5_config, load_tokenizer_identity,
    official_bf16_precision, official_native_fp8_precision, resolve_thinking_mode,
    sha256_bytes,
)
from .architecture.spark_adapter import Spark2_5Adapter, SparkAdapterError
from .architecture.spine import (
    SPINE_SCHEMA, SpineError, SpineForwardContext,
)
from .architecture.spine_memory import EvidenceBank, EvidenceRecord, SpineMemoryExperiment
from .architecture.spine_text import (
    TEXT_CODEC_VERSION, TEXT_PRODUCER_REVISION, TEXT_REPRESENTATION_VERSION,
    SparkTextMemoryPath, VersionedTextProducer,
)
from .optimizers import NativeFusedAdamW
from .testing import DeterministicTinyTokenizer

PROTOCOL_FORMAT = "replica.spark_text_memory.protocol.v1"
PROTOCOL_ID = "spark_text_memory_v1"
FIXTURE_FORMAT = "replica.spark_text_memory.fixture.v1"
RUNNER_STATE_FORMAT = "replica.spark_text_memory.runner_state.v1"
FIT_RECEIPT_FORMAT = "replica.spark_text_memory.fit_receipt.v1"
DEVELOPMENT_FORMAT = "replica.spark_text_memory.development.v1"
RESTART_FORMAT = "replica.spark_text_memory.restart.v1"
REPORT_FORMAT = "replica.spark_text_memory.run.v1"
NATIVE_BACKEND_EVIDENCE_FORMAT = "replica.spark_native_fp8_backend_evidence.v1"
CPU_FIXTURE_LABEL = "cpu_fixture"
NATIVE_LABEL = "native"
STAGES = ("all", "fit", "development", "restart")

ARMS = (
    "full_eligible_context",
    "selective_exact_and_procedure",
    "spine_E_only",
    "source_free_P_only",
    "E_plus_P",
    "no_update",
    "reset_P",
    "shuffled_memory",
    "inapplicable_procedure",
)
ORACLE_ARM = "selective_exact_and_procedure"

RESPONSE_LR = 1e-4
BANK_EXPERTS = 2
BANK_HIDDEN_WIDTH = 64
READ_RECORDS = 8
PLASTIC_EXPERT = 0


class RunnerError(RuntimeError):
    """Orchestration or evidence-contract failure."""


class Refused(RunnerError):
    """A declared precondition failed before work was performed."""

    def __init__(self, check: str, detail: str) -> None:
        super().__init__(f"{check}: {detail}")
        self.check = check
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        return {"check": self.check, "detail": self.detail}


class CapExceeded(RunnerError):
    """A declared work or artifact cap would be exceeded."""

    def __init__(self, counter: str, value: int, cap: int) -> None:
        super().__init__(f"work cap exceeded: {counter}={value} > {cap}")
        self.counter = counter
        self.value = value
        self.cap = cap


class GenerationFailure(RunnerError):
    """A failed generation with the partial trajectory retained."""

    def __init__(self, message: str, *, partial: Mapping[str, Any] | None = None,
                 cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.partial = dict(partial or {})
        self.cause = cause
        self.traceback = traceback.format_exc()


class WorkLedger:
    """Measured work with corrected protocol ceilings and exact reconciliation."""

    def __init__(self, profile: "WorkProfile") -> None:
        self.profile = profile
        self.counters: dict[str, int] = {
            name: 0 for name in (
                "fit_events", "development_cases", "development_cases_per_arm",
                "parity_requests", "restart_parity_requests", "restarts",
                "model_loads", "tokenizer_loads", "training_forwards",
                "query_sensing_forwards", "backward_calls",
                "backward_processed_positions", "optimizer_steps",
                "generation_requests", "autoregressive_forwards",
                "forward_calls", "generated_tokens", "processed_text_positions",
                "retrieval_calls", "verification_calls", "teacher_passes",
                "replay_events", "consolidation_events", "checkpoint_writes",
                "retained_checkpoint_artifacts", "reloads", "cache_hits",
                "cache_misses", "native_retries", "selected_records",
                "omitted_records", "checkpoint_bytes", "artifact_bytes",
                "failure_rows", "no_op_steps", "zero_new_work_retries",
                "retry_events", "native_query_passes",
                "plastic_forward_calls", "plastic_backward_calls",
                "plastic_feature_vectors", "plastic_backward_feature_vectors",
                "plastic_fit_evaluation_forwards",
            )
        }
        self.caps = profile.corrected_caps()
        self.per_arm: dict[str, int] = {arm: 0 for arm in ARMS}
        self.per_phase: dict[str, dict[str, int]] = {}
        self.wall_seconds = 0.0
        self._phase = "cold_start"

    # -- accounting -------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        if phase not in ("cold_start", "steady_state", "amortized"):
            raise RunnerError("unknown cost phase")
        self._phase = phase

    def _phase_row(self) -> dict[str, int]:
        return self.per_phase.setdefault(self._phase, {})

    def can_charge(self, counter: str, value: int = 1) -> bool:
        if value < 0:
            raise RunnerError("work charges must be nonnegative")
        return self.counters[counter] + value <= self.caps.get(counter, 0)

    def charge(self, counter: str, value: int = 1) -> None:
        if value < 0:
            raise RunnerError("work charges must be nonnegative")
        if counter not in self.counters:
            raise RunnerError(f"undeclared work counter {counter}")
        cap = self.caps.get(counter)
        if cap is not None and self.counters[counter] + value > cap:
            raise CapExceeded(counter, self.counters[counter] + value, cap)
        self.counters[counter] += value
        row = self._phase_row()
        row[counter] = row.get(counter, 0) + value

    def arm_case(self, arm: str) -> None:
        if arm not in self.per_arm:
            raise RunnerError(f"unknown arm {arm}")
        self.per_arm[arm] += 1
        self.charge("development_cases")
        if self.per_arm[arm] > self.profile.development_cases_per_arm:
            raise CapExceeded("development_cases_per_arm",
                              self.per_arm[arm],
                              self.profile.development_cases_per_arm)

    def query_sensing(self, *, positions: int) -> None:
        self.charge("query_sensing_forwards")
        self.charge("native_query_passes")
        self.charge("forward_calls")
        self.charge("processed_text_positions", positions)

    def training(self, *, positions: int) -> None:
        self.charge("training_forwards")
        self.charge("forward_calls")
        self.charge("processed_text_positions", positions)

    def plastic_forward(self, *, feature_vectors: int) -> None:
        self.charge("plastic_forward_calls")
        self.charge("plastic_feature_vectors", feature_vectors)
        self.charge("forward_calls")

    def backward(self, *, positions: int) -> None:
        self.charge("backward_calls")
        self.charge("backward_processed_positions", positions)

    def plastic_backward(self, *, feature_vectors: int) -> None:
        self.charge("backward_calls")
        self.charge("plastic_backward_calls")
        self.charge("plastic_backward_feature_vectors", feature_vectors)

    def optimizer(self) -> None:
        self.charge("optimizer_steps")

    def generation_request(self, *, new_tokens: int, positions: int,
                           retrieval: bool = True) -> None:
        self.charge("generation_requests")
        self.charge("autoregressive_forwards", new_tokens)
        self.charge("forward_calls", new_tokens)
        self.charge("generated_tokens", new_tokens)
        self.charge("processed_text_positions", positions)
        if retrieval:
            self.retrieval()

    def retrieval(self) -> None:
        self.charge("retrieval_calls")
        self.charge("cache_misses")

    def verification(self) -> None:
        self.charge("verification_calls")

    def checkpoint_write(self, *, bytes_written: int = 0) -> None:
        self.charge("checkpoint_writes")
        if bytes_written:
            self.charge("checkpoint_bytes", bytes_written)

    def reload(self) -> None:
        self.charge("reloads")

    def load_backend(self) -> None:
        self.charge("model_loads")
        self.charge("tokenizer_loads")

    def retry_zero_new_work(self, event_id: str) -> dict[str, Any]:
        """Record a retry without charging any step, forward or backward."""
        self.charge("retry_events")
        self.charge("zero_new_work_retries")
        return {"event_id": event_id, "charged_new_work": False,
                "optimizer_steps": 0, "forward_calls": 0,
                "backward_calls": 0, "status": "retried_zero_new_work"}

    def fail(self) -> None:
        self.charge("failure_rows")

    def merge(self, other: Mapping[str, Any]) -> None:
        for name, value in other.get("counters", {}).items():
            if name in self.counters:
                self.counters[name] += int(value)
        for arm, value in other.get("per_arm", {}).items():
            if arm in self.per_arm:
                self.per_arm[arm] += int(value)
        for phase, row in other.get("per_phase", {}).items():
            target = self.per_phase.setdefault(phase, {})
            for name, value in row.items():
                target[name] = target.get(name, 0) + int(value)
        self.wall_seconds += float(other.get("wall_seconds", 0.0))

    def cap_violations(self) -> list[dict[str, int | str]]:
        return [
            {"counter": name, "value": value, "cap": self.caps[name]}
            for name, value in sorted(self.counters.items())
            if name in self.caps and value > self.caps[name]
        ]

    def reconcile(self) -> dict[str, Any]:
        expected = self.profile.expected_measured()
        mismatches = {
            name: {"expected": expected[name], "actual": self.counters[name]}
            for name in expected
            if self.counters[name] != expected[name]
        }
        if mismatches:
            raise RunnerError(f"work counters do not reconcile: {mismatches}")
        violations = self.cap_violations()
        if violations:
            raise RunnerError(f"work caps were exceeded: {violations}")
        for arm, count in self.per_arm.items():
            if count != self.profile.development_cases_per_arm:
                raise RunnerError(
                    f"arm {arm} executed {count} development cases, "
                    f"expected {self.profile.development_cases_per_arm}")
        self.counters["retained_checkpoint_artifacts"] = min(
            self.counters["retained_checkpoint_artifacts"],
            self.profile.retained_checkpoints)
        if self.counters["retained_checkpoint_artifacts"] > self.profile.retained_checkpoints:
            raise RunnerError("retained checkpoint bound exceeded")
        return {
            "expected": expected,
            "actual": dict(self.counters),
            "reconciled": True,
            "cap_violations": [],
            "per_arm": dict(self.per_arm),
            "schedule_declared": self.profile.declared_schedule(),
            "corrected_caps": dict(self.caps),
            "cost_phases": copy.deepcopy(self.per_phase),
            "wall_seconds": self.wall_seconds,
        }

    def report(self) -> dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "caps": dict(self.caps),
            "per_arm": dict(self.per_arm),
            "per_phase": copy.deepcopy(self.per_phase),
            "wall_seconds": self.wall_seconds,
            "cap_violations": self.cap_violations(),
        }


@dataclass(frozen=True)
class WorkProfile:
    """Protocol profile plus explicitly scaled CPU-fixture orchestration bounds."""

    label: str
    context_tokens: int
    max_new_tokens: int
    fit_events: int = 4
    development_cases_per_arm: int = 12
    arms: int = 9
    parity_requests: int = 2
    restart_parity_requests: int = 1
    restarts: int = 1
    checkpoint_writes: int = 5
    retained_checkpoints: int = 4
    native_admissible: bool = False
    artifact_bytes_max: int = 0

    @property
    def generation_requests(self) -> int:
        return (self.parity_requests + self.restart_parity_requests
                + self.arms * self.development_cases_per_arm)

    def positions_per_request(self) -> int:
        new = self.max_new_tokens
        return self.context_tokens * new + new * (new - 1) // 2

    def declared_schedule(self) -> dict[str, int]:
        return {
            "fit_events": self.fit_events,
            "development_cases_per_arm": self.development_cases_per_arm,
            "development_cases": self.arms * self.development_cases_per_arm,
            "parity_requests": self.parity_requests,
            "restart_parity_requests": self.restart_parity_requests,
            "restarts": self.restarts,
            "model_loads": 1 + self.restarts,
            "tokenizer_loads": 1 + self.restarts,
            "training_forwards": self.fit_events,
            "plastic_forward_calls": self.fit_events,
            "plastic_backward_calls": self.fit_events,
            "query_sensing_forwards": (2 * self.fit_events
                                       + self.generation_requests * self.max_new_tokens),
            "native_query_passes": (2 * self.fit_events
                                    + self.generation_requests * self.max_new_tokens),
            "backward_calls": 2 * self.fit_events,
            "backward_processed_positions": (self.fit_events
                                             * self.context_tokens),
            "plastic_feature_vectors": self.fit_events * self.context_tokens,
            "plastic_backward_feature_vectors": self.fit_events * self.context_tokens,
            "optimizer_steps": 2 * self.fit_events,
            "generation_requests": self.generation_requests,
            "autoregressive_forwards": self.generation_requests * self.max_new_tokens,
            "forward_calls": (2 * self.fit_events
                              + (2 * self.fit_events
                                 + self.generation_requests * self.max_new_tokens)
                              + self.generation_requests * self.max_new_tokens),
            "generated_tokens": self.generation_requests * self.max_new_tokens,
            "processed_text_positions": (
                3 * self.fit_events * self.context_tokens
                + 2 * self.generation_requests * self.positions_per_request()),
            "retrieval_calls": self.fit_events + self.generation_requests,
            "verification_calls": (
                self.fit_events + self.arms * self.development_cases_per_arm),
            "checkpoint_writes": self.checkpoint_writes,
            "retained_checkpoint_artifacts": self.retained_checkpoints,
            "reloads": self.restarts,
            "cache_hits": 0,
            "cache_misses": self.fit_events + self.generation_requests,
            "teacher_passes": 0,
            "replay_events": 0,
            "consolidation_events": 0,
            "native_retries": 0,
        }

    def corrected_caps(self) -> dict[str, int]:
        """Worst-case bounds from the corrected protocol accounting.

        A P read may sense the native hidden state once per current query.  The
        conservative schedule additionally allows one sensing forward for every
        autoregressive step, because a deployment may rebuild P query features
        from the growing native sequence.  These are upper bounds; measured
        forwards are charged separately and must not exceed them.
        """
        new = self.max_new_tokens
        requests = self.generation_requests
        gen_positions = requests * self.positions_per_request()
        query_sensing = 2 * self.fit_events + requests * new
        return {
            "fit_events": self.fit_events,
            "development_cases": self.arms * self.development_cases_per_arm,
            "development_cases_per_arm": self.development_cases_per_arm,
            "parity_requests": self.parity_requests,
            "restart_parity_requests": self.restart_parity_requests,
            "restarts": self.restarts,
            "model_loads": 1 + self.restarts,
            "tokenizer_loads": 1 + self.restarts,
            "training_forwards": self.fit_events,
            "plastic_forward_calls": self.fit_events,
            "plastic_backward_calls": self.fit_events,
            "query_sensing_forwards": query_sensing,
            "native_query_passes": query_sensing,
            "backward_calls": 2 * self.fit_events,
            "backward_processed_positions": (self.fit_events
                                             * self.context_tokens),
            "plastic_feature_vectors": self.fit_events * self.context_tokens,
            "plastic_backward_feature_vectors": self.fit_events * self.context_tokens,
            "optimizer_steps": 2 * self.fit_events,
            "generation_requests": requests,
            "autoregressive_forwards": requests * new,
            "forward_calls": (2 * self.fit_events + query_sensing
                              + requests * new),
            "generated_tokens": requests * new,
            "processed_text_positions": (
                3 * self.fit_events * self.context_tokens + 2 * gen_positions),
            "retrieval_calls": self.fit_events + requests,
            "verification_calls": (
                self.fit_events + self.arms * self.development_cases_per_arm),
            "teacher_passes": 0,
            "replay_events": 0,
            "consolidation_events": 0,
            "checkpoint_writes": self.checkpoint_writes,
            "retained_checkpoint_artifacts": self.retained_checkpoints,
            "reloads": self.restarts,
            "cache_hits": 0,
            "cache_misses": self.fit_events + requests,
            "native_retries": 0,
            "artifact_bytes": self.artifact_bytes_max,
        }

    def expected_measured(self) -> dict[str, int]:
        """Counts that are exact in the implemented all-stage schedule."""
        requests = self.generation_requests
        new = self.max_new_tokens
        return {
            "fit_events": self.fit_events,
            "development_cases": self.arms * self.development_cases_per_arm,
            "parity_requests": self.parity_requests,
            "restart_parity_requests": self.restart_parity_requests,
            "restarts": self.restarts,
            "model_loads": 1 + self.restarts,
            "tokenizer_loads": 1 + self.restarts,
            "training_forwards": self.fit_events,
            "plastic_forward_calls": self.fit_events,
            "plastic_backward_calls": self.fit_events,
            "backward_calls": 2 * self.fit_events,
            "backward_processed_positions": (self.fit_events
                                             * self.context_tokens),
            "plastic_feature_vectors": self.fit_events * self.context_tokens,
            "plastic_backward_feature_vectors": self.fit_events * self.context_tokens,
            "optimizer_steps": 2 * self.fit_events,
            "generation_requests": requests,
            "autoregressive_forwards": requests * new,
            "generated_tokens": requests * new,
            "retrieval_calls": self.fit_events + requests,
            "verification_calls": self.fit_events + self.arms * self.development_cases_per_arm,
            "checkpoint_writes": self.checkpoint_writes,
            "retained_checkpoint_artifacts": self.retained_checkpoints,
            "reloads": self.restarts,
            "cache_hits": 0,
            "cache_misses": self.fit_events + requests,
            "teacher_passes": 0,
            "replay_events": 0,
            "consolidation_events": 0,
            "native_retries": 0,
            "zero_new_work_retries": 0,
            "failure_rows": 0,
            "plastic_fit_evaluation_forwards": self.fit_events,
        }

    @classmethod
    def from_protocol(cls, config: Mapping[str, Any], *,
                      backend_kind: str = "native",
                      fixture_context_tokens: int | None = None,
                      fixture_max_new_tokens: int | None = None) -> "WorkProfile":
        runtime = config["runtime"]
        schedule = config["work_and_cost"]["native_schedule"]
        native_context = int(runtime["context_tokens_max"])
        native_new = int(runtime["generation"]["max_new_tokens"])
        fixture_requested = (
            backend_kind == CPU_FIXTURE_LABEL
            or fixture_context_tokens is not None
            or fixture_max_new_tokens is not None)
        if not fixture_requested:
            return cls(
                label="native", context_tokens=native_context,
                max_new_tokens=native_new,
                fit_events=int(schedule["fit_events"]),
                development_cases_per_arm=int(schedule["development_cases_per_arm"]),
                arms=len(config["controls"]["arms"]),
                parity_requests=int(schedule["parity_requests"]),
                restart_parity_requests=int(schedule["restart_parity_requests"]),
                restarts=int(schedule["restarts"]),
                checkpoint_writes=int(schedule["checkpoint_writes_max"]),
                retained_checkpoints=int(schedule["retained_checkpoint_artifacts_max"]),
                native_admissible=True,
                artifact_bytes_max=int(schedule["artifact_bytes_max"]),
            )
        context = int(fixture_context_tokens if fixture_context_tokens is not None else 512)
        new = int(fixture_max_new_tokens if fixture_max_new_tokens is not None else 4)
        if context <= 0 or new <= 0:
            raise RunnerError("fixture context/new-token lengths must be positive")
        return cls(
            label=CPU_FIXTURE_LABEL, context_tokens=context,
            max_new_tokens=new,
            fit_events=int(schedule["fit_events"]),
            development_cases_per_arm=int(schedule["development_cases_per_arm"]),
            arms=len(config["controls"]["arms"]),
            parity_requests=int(schedule["parity_requests"]),
            restart_parity_requests=int(schedule["restart_parity_requests"]),
            restarts=int(schedule["restarts"]),
            checkpoint_writes=int(schedule["checkpoint_writes_max"]),
            retained_checkpoints=int(schedule["retained_checkpoint_artifacts_max"]),
            native_admissible=False,
            artifact_bytes_max=128 * 1024 * 1024,
        )


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path, name: str = "json") -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(name, f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(name, f"{path} must contain an object")
    return value


def _read_jsonl(path: str | Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Refused(name, f"cannot read {path}: {error}") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Refused(name, f"{path}:{number} is not JSON: {error}") from error
        if not isinstance(row, dict):
            raise Refused(name, f"{path}:{number} must be an object")
        rows.append(row)
    return rows


def _require(condition: bool, check: str, detail: str) -> None:
    if not condition:
        raise Refused(check, detail)


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                text=True, capture_output=True, check=True,
                                timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


@dataclass(frozen=True)
class Protocol:
    path: Path
    sha256: str
    config: Mapping[str, Any]

    @property
    def protocol_id(self) -> str:
        return str(self.config["protocol_id"])

    @property
    def schedule(self) -> Mapping[str, Any]:
        return self.config["work_and_cost"]["native_schedule"]

    @property
    def dataset_root(self) -> Path:
        return self.path.parent.parent / str(self.config["dataset"]["root"])

    @property
    def arms(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.config["controls"]["arms"])

    @property
    def source_identity(self) -> Mapping[str, Any]:
        return self.config["source_identities"]["official_fp8"]

    def implementation_hashes(self, root: Path | None = None) -> dict[str, str]:
        base = root or self.path.parent.parent
        rows = {}
        for relative in self.config["launch"]["implementation_files"]:
            path = base / relative
            _require(path.is_file(), "implementation", f"missing {relative}")
            rows[relative] = sha256_file(path)
        return rows


def load_protocol(path: str | Path) -> Protocol:
    path = Path(path)
    config = _read_json(path, "protocol")
    _require(config.get("format") == PROTOCOL_FORMAT, "protocol", "format differs")
    _require(config.get("protocol_id") == PROTOCOL_ID, "protocol", "id differs")
    _require(config["active_direction"]["source_baseline"] == "official_fp8",
             "source", "active baseline must remain official_fp8")
    _require(config["active_direction"]["internal_spine"] == "early_permanent_required",
             "spine", "internal spine must be early and permanent")
    _require(config["trainable_groups"]["optimizer"] == "NativeFusedAdamW",
             "optimizer", "only NativeFusedAdamW is admitted")
    _require(config["trainable_groups"]["frozen_original_backbone"] is True,
             "optimizer", "the original backbone must stay frozen")
    _require(config["runtime"]["generation"]["greedy"] is True
             and config["runtime"]["generation"]["sampling"] is False,
             "generation", "greedy decoding is required")
    _require(config["runtime"]["generation"]["enable_thinking"] is False,
             "generation", "thinking must be disabled")
    _require(config["runtime"]["generation"][
                 "terminal_stop_tokens_excluded_before_decode"] is True,
             "generation", "terminal stops must be excluded before decode")
    _require(config["dataset"]["sealed_final_labels_closed"] is True,
             "access", "sealed-final labels must remain closed")
    _require([int(site) for site in config["runtime"]["spine_sites"]] == [3, 7],
             "spine", "pilot sites differ")
    arms = tuple(arm["id"] for arm in config["controls"]["arms"])
    _require(arms == ARMS, "controls", f"arm set/order differs: {arms}")
    schedule = config["work_and_cost"]["native_schedule"]
    _require(int(schedule["fit_events"]) == 4, "schedule", "fit events must be four")
    _require(int(schedule["development_cases_per_arm"]) == 12,
             "schedule", "development cases per arm must be twelve")
    _require(int(schedule["restarts"]) == 1, "schedule", "one restart is required")
    _require(int(config["runtime"]["context_tokens_max"]) > 0
             and int(config["runtime"]["generation"]["max_new_tokens"]) > 0,
             "runtime", "context and generation lengths must be positive")
    _require(int(config["memory_contract"]["active_read_slots_max"]) <= 8,
             "memory", "read budget cannot exceed eight slots")
    _require(config["source_identities"]["official_fp8"][
                 "substitution_allowed"] is False,
             "source", "precision substitution must stay disabled")
    return Protocol(path=path.resolve(), sha256=sha256_file(path), config=config)


@dataclass
class PreparedDataset:
    """Hash-verified owned fixture with development labels behind a fit barrier."""

    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    _development_open: bool = False
    _rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: str | Path, *, expected_sha256: str | None = None) -> "PreparedDataset":
        root = Path(root)
        manifest_path = root / "manifest.json"
        manifest = _read_json(manifest_path, "prepared")
        _require(manifest.get("format") == FIXTURE_FORMAT, "prepared", "format differs")
        _require(manifest.get("protocol_id") == PROTOCOL_ID, "prepared", "protocol id differs")
        digest = sha256_file(manifest_path)
        if expected_sha256 is not None:
            _require(digest == expected_sha256, "prepared", "manifest hash differs")
        _require(manifest["sealed_final"]["materialized"] is False
                 and manifest["sealed_final"]["opened"] is False,
                 "prepared", "sealed-final data must stay unmaterialized and closed")
        _require(not (root / "sealed_final.jsonl").exists(), "prepared",
                 "sealed-final file must not be materialized")
        files = manifest.get("files")
        _require(isinstance(files, Mapping) and files, "prepared", "file manifest is empty")
        for name, entry in files.items():
            path = root / str(name)
            _require(path.is_file(), "prepared", f"missing fixture file {name}")
            _require(path.stat().st_size == int(entry["bytes"]), "prepared",
                     f"{name} byte count differs")
            _require(sha256_file(path) == entry["sha256"], "prepared",
                     f"{name} hash differs")
        counts = manifest.get("counts", {})
        _require(int(counts.get("fit_events", -1)) == 4, "prepared",
                 "fit event count differs")
        _require(int(counts.get("development_cases", -1)) == 12, "prepared",
                 "development case count differs")
        _require(int(counts.get("arms", -1)) == len(ARMS), "prepared",
                 "arm count differs")
        return cls(root=root.resolve(), manifest=manifest,
                   manifest_sha256=digest)

    def _rows_for(self, name: str) -> list[dict[str, Any]]:
        if name not in self._rows:
            self._rows[name] = _read_jsonl(self.root / name, name)
        return [copy.deepcopy(row) for row in self._rows[name]]

    def open_development(self, fit_receipt: Mapping[str, Any]) -> None:
        _require(fit_receipt.get("format") == FIT_RECEIPT_FORMAT, "fit_receipt",
                 "format differs")
        _require(fit_receipt.get("protocol_sha256") is not None, "fit_receipt",
                 "protocol binding is missing")
        _require(fit_receipt.get("prepared_manifest_sha256") == self.manifest_sha256,
                 "fit_receipt", "prepared manifest binding differs")
        _require(fit_receipt.get("derived_caches_required") is False, "fit_receipt",
                 "fit receipt requires derived caches")
        _require(int(fit_receipt.get("checkpoint_writes", 0)) >= 1, "fit_receipt",
                 "fit receipt has no checkpoint")
        _require(fit_receipt.get("status") in (
            "fit_executed", "fit_executed_no_update", "fit_failed"), "fit_receipt",
            "unknown fit status")
        self._development_open = True

    def _require_development(self) -> None:
        if not self._development_open:
            raise Refused("development_labels_closed",
                          "development labels require an opened fit receipt")

    @property
    def development_open(self) -> bool:
        return self._development_open

    def observations(self) -> list[dict[str, Any]]:
        return self._rows_for("observations.jsonl")

    def fit_events(self) -> list[dict[str, Any]]:
        return self._rows_for("fit_events.jsonl")

    def procedures(self) -> list[dict[str, Any]]:
        value = _read_json(self.root / "procedures.json", "procedures")
        rows = value.get("procedures")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise Refused("procedures", "procedures must be a list of objects")
        return copy.deepcopy(rows)

    def tasks(self) -> list[dict[str, Any]]:
        value = _read_json(self.root / "tasks.json", "tasks")
        rows = value.get("tasks")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise Refused("tasks", "tasks must be a list of objects")
        return copy.deepcopy(rows)

    def development_cases(self) -> list[dict[str, Any]]:
        self._require_development()
        cases = self._rows_for("development.jsonl")
        labels = {row["case_id"]: row for row in self._rows_for(
            "development_labels.jsonl")}
        _require(len(cases) == 12, "development", "there must be twelve cases")
        for case in cases:
            _require(case["case_id"] in labels, "development",
                     f"missing label for {case['case_id']}")
            case["expected"] = labels[case["case_id"]]["expected"]
        return cases

    def parity_requests(self, *, restart: bool = False) -> list[dict[str, Any]]:
        self._require_development()
        rows = self._rows_for("parity_requests.jsonl")
        selected = [row for row in rows if row["request_id"].startswith("restart-") == restart]
        _require(len(selected) == (1 if restart else 2), "parity",
                 "parity request count differs")
        return selected

    def worlds(self) -> list[dict[str, Any]]:
        value = _read_json(self.root / "worlds.json", "worlds")
        return copy.deepcopy(value["worlds"])
# ---------------------------------------------------------------------------
# Backends


@dataclass
class Backend:
    """One loaded decoder/tokenizer pair; no implicit device or precision change."""

    label: str
    adapter: Spark2_5Adapter
    tokenizer: Any
    device: str
    native_admissible: bool
    loads_reported: bool = False
    chat_mode: Mapping[str, Any] | None = None
    stop_token_ids: tuple[int, ...] = ()
    fixture_seed: int | None = None
    authorization: Mapping[str, Any] | None = None

    @property
    def pad_token_id(self) -> int:
        value = getattr(self.tokenizer, "pad_token_id", None)
        if value is None:
            mapping = getattr(self.tokenizer, "_token_to_id", None)
            if isinstance(mapping, Mapping):
                value = mapping.get("<pad>")
        if value is None:
            raise Refused("tokenizer", "tokenizer has no pad token id")
        return int(value)

    def render(self, content: str) -> str:
        kwargs = dict(self.chat_mode or {})
        kwargs.update(add_generation_prompt=True)
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, **kwargs)
        if not isinstance(rendered, str) or not rendered:
            raise Refused("tokenizer", "chat template produced no text")
        return rendered

    def encode_text(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        values = [int(value) for value in ids]
        if not values:
            raise Refused("tokenizer", "text produced no tokens")
        return values

    def encode_prompt(self, content: str, context_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.encode_text(self.render(content))
        if len(ids) > context_tokens:
            raise Refused("context_overflow",
                          f"prompt has {len(ids)} tokens, bound is {context_tokens}")
        pad = context_tokens - len(ids)
        values = [self.pad_token_id] * pad + ids
        mask = [0] * pad + [1] * len(ids)
        return (torch.tensor([values], dtype=torch.long, device=self.device),
                torch.tensor([mask], dtype=torch.long, device=self.device))

    def encode_training(self, content: str, answer: str,
                        context_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = self.encode_text(self.render(content))
        answer_ids = self.encode_text(answer)
        sequence = prompt + answer_ids
        if len(sequence) > context_tokens:
            raise Refused("context_overflow",
                          f"training sequence has {len(sequence)} tokens, bound is {context_tokens}")
        pad = context_tokens - len(sequence)
        values = sequence + [self.pad_token_id] * pad
        mask = [1] * len(sequence) + [0] * pad
        labels = [-100] * context_tokens
        answer_start = len(prompt)
        # Logits at position i predict the token at i+1 in this adapter ABI.
        for offset in range(len(answer_ids)):
            label_position = answer_start + offset - 1
            if label_position >= 0:
                labels[label_position] = answer_ids[offset]
        return (torch.tensor([values], dtype=torch.long, device=self.device),
                torch.tensor([mask], dtype=torch.long, device=self.device),
                torch.tensor([labels], dtype=torch.long, device=self.device))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False))

    def close(self) -> None:
        # The owners are local to one process; explicit release keeps the restart
        # boundary visible without mutating the sealed source.
        return None


class _TinyBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
        self.linear = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, cache_position=None, position_ids=None,
                **kwargs):
        return hidden_states + 0.2 * self.linear(self.norm(hidden_states))


class _TinySparkModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        geometry = config.geometry
        self.embedding = nn.Embedding(geometry.vocab_size, geometry.hidden_size,
                                      padding_idx=0, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([_TinyBlock(geometry.hidden_size)
                                     for _ in range(geometry.num_hidden_layers)])
        self.norm = nn.LayerNorm(geometry.hidden_size, dtype=torch.bfloat16)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                use_cache=False, **kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask,
                           position_ids=position_ids, use_cache=use_cache)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _TinySparkCausalLM(nn.Module):
    """Tiny Spark-shaped stub.  Local fixture only; never a native source."""

    def __init__(self, config) -> None:
        super().__init__()
        geometry = config.geometry
        self.config = SimpleNamespace(
            model_type="spark2_5", use_cache=True, _attn_implementation="eager",
            hidden_size=geometry.hidden_size,
            num_hidden_layers=geometry.num_hidden_layers,
            vocab_size=geometry.vocab_size)
        self.model = _TinySparkModel(config)
        self.lm_head = nn.Linear(geometry.hidden_size, geometry.vocab_size,
                                 bias=False, dtype=torch.bfloat16)
        self.lm_head.weight = self.model.embedding.weight

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                use_cache=False, **kwargs):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, use_cache=use_cache)
        logits = F.linear(hidden.last_hidden_state, self.model.embedding.weight)
        return SimpleNamespace(logits=logits)

    def get_input_embeddings(self):
        return self.model.embedding

    def get_output_embeddings(self):
        return self.lm_head


def _tiny_config_raw(*, vocab_size: int) -> dict[str, Any]:
    layer_types = ["sliding_attention"] * 3 + ["full_attention"]
    layer_types = layer_types * 2
    return {
        "architectures": ["Spark2_5ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_spark.Spark2_5Config",
            "AutoModel": "modeling_spark.Spark2_5Model",
            "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
        },
        "model_type": "spark2_5", "dtype": "bfloat16",
        "num_hidden_layers": len(layer_types), "hidden_size": 8,
        "intermediate_size": 16, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "vocab_size": vocab_size,
        "sliding_window": 4, "max_position_embeddings": 4096,
        "tie_word_embeddings": True, "use_cache": True, "hidden_act": "gelu",
        "attention_bias": False, "mlp_bias": False,
        "headwise_attn_output_gate": True, "gate_attn_act_mode": "sigmoid",
        "layer_types": layer_types,
        "rope_parameters": {
            "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
            "full_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
        },
        "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
        "transformers_version": "4.57.1",
    }


def _align_text_codec(adapter: Spark2_5Adapter) -> None:
    """Bind the spine core to the versioned text producer/codec.

    ``SpineCore`` defaults to placeholder ``codec-1``/``repr-1`` labels while
    ``VersionedTextProducer`` uses explicit text versions.  Evidence admission
    rejects the mismatch, so the runner binds the owner once before any record
    is observed.  This is an API alignment, not a representation conversion.
    """
    core = adapter.extension.spine
    core.codec_version = TEXT_CODEC_VERSION
    core.representation_version = TEXT_REPRESENTATION_VERSION


def _fixture_texts(dataset: PreparedDataset) -> list[str]:
    texts: list[str] = []
    for row in dataset.observations():
        texts.append(row["text"])
    for row in dataset.fit_events():
        texts.extend((row["question"], row["answer"], row["expected"]))
    for row in dataset.procedures():
        texts.extend(row["applicability"])
        texts.extend(row["prerequisites"])
        texts.extend(row["exceptions"])
        texts.append(row["outcome"])
        texts.append(row["failure_lesson"])
    return texts


def build_cpu_fixture_backend(dataset: PreparedDataset, profile: WorkProfile, *,
                              seed: int = 20261028) -> Backend:
    """Deterministic tiny CPU backend, explicitly outside native admission."""
    if profile.label != CPU_FIXTURE_LABEL:
        raise Refused("backend", "cpu_fixture requires a cpu_fixture work profile")
    if profile.native_admissible:
        raise Refused("backend", "cpu_fixture can never be native-admissible")
    tokenizer = DeterministicTinyTokenizer(texts=_fixture_texts(dataset))
    torch.manual_seed(seed)
    raw = _tiny_config_raw(vocab_size=8192)
    config = load_spark2_5_config(raw, require_official_geometry=False)
    base = _TinySparkCausalLM(config)
    artifact = sha256_bytes(b"cpu-fixture-tiny-spark-not-native")
    precision = official_bf16_precision(config, artifact_sha256=artifact)
    adapter = Spark2_5Adapter(
        base, config, precision, device="cpu", require_official_geometry=False,
        projector_layer=3, memory_size=4, heads=2,
        spine_sites=(3, 7), spine_slots=8)
    adapter.eval()
    _align_text_codec(adapter)
    adapter.seal_original_identity()
    stop_ids = tuple(sorted({int(tokenizer._token_to_id["<eos>"]),
                             int(tokenizer._token_to_id["<pad>"])}))
    return Backend(
        label=CPU_FIXTURE_LABEL, adapter=adapter, tokenizer=tokenizer,
        device="cpu", native_admissible=False,
        chat_mode={"enable_thinking": False}, stop_token_ids=stop_ids,
        fixture_seed=seed)


def _validate_authorization_receipt(protocol: Protocol, dataset: PreparedDataset,
                                    receipt: Mapping[str, Any], output_root: Path,
                                    *, implementation_hashes: Mapping[str, str] | None,
                                    head: str | None,
                                    allow_existing_output: bool = False) -> None:
    launch = protocol.config["launch"]
    for key in launch["required_authorization_fields"]:
        _require(key in receipt and receipt[key] not in (None, "", []),
                 "authorization", f"missing {key}")
    _require(receipt["protocol_id"] == protocol.protocol_id,
             "authorization", "protocol id differs")
    _require(receipt["protocol_sha256"] == protocol.sha256,
             "authorization", "protocol hash differs")
    _require(receipt["prepared_manifest_sha256"] == dataset.manifest_sha256,
             "authorization", "prepared manifest hash differs")
    computed = dict(implementation_hashes or protocol.implementation_hashes())
    _require(dict(receipt["implementation_sha256"]) == computed,
             "authorization", "implementation hashes differ")
    if launch.get("implementation_sha256"):
        _require(dict(launch["implementation_sha256"]) == computed,
                 "authorization", "protocol implementation binding differs")
    if head is not None:
        _require(receipt["protocol_commit"] == head, "authorization",
                 "protocol commit does not match the current HEAD")
        if launch.get("protocol_commit"):
            _require(launch["protocol_commit"] == head, "authorization",
                     "protocol is not bound to current HEAD")
    binding = launch["required_execution_binding"]
    _require(receipt["source_identity"] == binding["source_identity"],
             "authorization", "source identity differs")
    _require(receipt["execution_format"] == binding["execution_format"],
             "authorization", "execution format differs")
    _require(receipt["execution_backend"] == binding["execution_backend"],
             "authorization", "execution backend differs")
    _require(receipt["output_root"] == str(output_root),
             "authorization", "output root differs")
    if not allow_existing_output and output_root.exists() and any(output_root.iterdir()):
        raise Refused("output", "native output must be fresh")
    try:
        expiry = datetime.fromisoformat(str(receipt["expires_at_utc"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise Refused("authorization", "expires_at_utc is invalid") from error
    _require(expiry.tzinfo is not None
             and expiry > datetime.now(timezone.utc), "authorization",
             "authorization receipt is expired")
    _require(isinstance(receipt["issued_by"], str) and receipt["issued_by"].strip(),
             "authorization", "issued_by is required")


def verify_native_authorization(protocol: Protocol, dataset: PreparedDataset, *,
                                authorization_path: str | Path,
                                backend_evidence_path: str | Path,
                                output_root: str | Path,
                                implementation_hashes: Mapping[str, str] | None = None,
                                head: str | None = None,
                                allow_existing_output: bool = False) -> dict[str, Any]:
    """Validate exact external authorization and backend evidence without loading."""
    authorization_path = Path(authorization_path)
    if not authorization_path.is_file():
        raise Refused("authorization", f"missing external receipt: {authorization_path}")
    receipt = _read_json(authorization_path, "authorization")
    head = head if head is not None else _git_head(protocol.path.parent.parent)
    _validate_authorization_receipt(
        protocol, dataset, receipt, Path(output_root),
        implementation_hashes=implementation_hashes, head=head,
        allow_existing_output=allow_existing_output)
    evidence_path = Path(backend_evidence_path)
    if not evidence_path.is_file():
        raise Refused("native_fp8_backend_evidence",
                      f"missing {evidence_path}")
    evidence = _read_json(evidence_path, "native_fp8_backend_evidence")
    _require(receipt["backend_evidence_sha256"] == sha256_file(evidence_path),
             "authorization", "backend evidence hash differs")
    required = protocol.config["launch"]["required_backend_evidence"]
    for key, expected in required.items():
        _require(evidence.get(key) == expected, "native_fp8_backend_evidence",
                 f"{key} differs")
    return {"receipt": receipt, "backend_evidence": evidence,
            "authorization_path": str(authorization_path),
            "backend_evidence_path": str(evidence_path),
            "authorization_sha256": sha256_file(authorization_path),
            "backend_evidence_sha256": sha256_file(evidence_path)}


def _native_adapter_capability() -> Mapping[str, Any] | None:
    """Return the adapter's declared native-FP8 admission contract, if any.

    The current adapter publishes none and explicitly refuses native FP8
    construction.  Absence is a capability fact, not a fallback opportunity.
    """
    candidate = getattr(Spark2_5Adapter, "native_fp8_admission", None)
    return candidate if isinstance(candidate, Mapping) else None


def build_native_backend(protocol: Protocol, dataset: PreparedDataset, *,
                         authorization_path: str | Path,
                         backend_evidence_path: str | Path,
                         output_root: str | Path, device: str,
                         allow_existing_output: bool = False) -> Backend:
    """Construct the sealed official-FP8 backend or refuse before loading."""
    if device != "mps":
        raise Refused("native_device", "official FP8 execution requires --device mps")
    receipt = verify_native_authorization(
        protocol, dataset, authorization_path=authorization_path,
        backend_evidence_path=backend_evidence_path, output_root=output_root)
    source = protocol.source_identity
    root = protocol.path.parent.parent
    source_root = root / str(source["local_path"])
    config_path = source_root / "config.json"
    if not config_path.is_file():
        raise Refused("source", f"sealed official FP8 config is missing: {config_path}")
    _require(sha256_file(config_path) == source["config_sha256"],
             "source", "official FP8 config hash differs")
    raw = _read_json(config_path, "source_config")
    config = load_spark2_5_config(raw, require_official_geometry=True)
    precision = official_native_fp8_precision(
        config, artifact_sha256=source["weight_sha256"])
    _require(precision.kind == OFFICIAL_NATIVE_FP8, "source",
             "active source is not official native FP8")
    admission = evaluate_admission(precision, device=device)
    if not admission.admitted:
        raise Refused("native_fp8_backend",
                      "; ".join(admission.reasons))
    contract = _native_adapter_capability()
    if contract is None:
        raise Refused(
            "native_fp8_adapter",
            "the active Spark2_5Adapter publishes no native-FP8 admission contract "
            "and refuses native FP8 construction; no BF16/derivative substitution is allowed")
    # The checks below are unreachable until the adapter publishes a contract.
    for filename, expected in (
        ("tokenizer.json", protocol.config["runtime"]["tokenizer"]["tokenizer_json_sha256"]),
        ("tokenizer_config.json", protocol.config["runtime"]["tokenizer"]["tokenizer_config_sha256"]),
        ("chat_template.jinja", protocol.config["runtime"]["tokenizer"]["chat_template_sha256"]),
    ):
        path = source_root / filename
        _require(path.is_file() and sha256_file(path) == expected,
                 "tokenizer", f"{filename} identity differs")
    weight_path = source_root / "model.safetensors"
    _require(weight_path.is_file() and sha256_file(weight_path) == source["weight_sha256"],
             "source", "official FP8 weight hash differs")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise Refused("native_runtime", "transformers is not available") from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(source_root), local_files_only=True, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise Refused("tokenizer", "the sealed tokenizer must be fast")
    tokenizer_config = _read_json(source_root / "tokenizer_config.json", "tokenizer_config")
    template = (source_root / "chat_template.jinja").read_text(encoding="utf-8")
    tokenizer_json = (source_root / "tokenizer.json").read_text(encoding="utf-8")
    identity = load_tokenizer_identity(
        config, tokenizer_config, template, tokenizer_json=tokenizer_json)
    if identity.vocab_size != config.geometry.vocab_size:
        raise Refused("tokenizer", "tokenizer vocabulary differs from the config")
    mode = resolve_thinking_mode(identity, "none")
    base = AutoModelForCausalLM.from_pretrained(
        str(source_root), local_files_only=True, trust_remote_code=True,
        torch_dtype="auto", attn_implementation="eager", use_safetensors=True)
    if getattr(base.config, "model_type", None) != "spark2_5":
        raise Refused("source", "loaded model is not spark2_5")
    adapter = Spark2_5Adapter(
        base, config, precision, device=device, require_official_geometry=True,
        projector_in_dim=int(config.geometry.hidden_size), zero_init_output=True,
        memory_size=256, heads=4, spine_sites=(3, 7), spine_slots=8)
    adapter.eval()
    _align_text_codec(adapter)
    adapter.seal_original_identity()
    stop_ids = tuple(sorted({
        int(getattr(tokenizer, "eos_token_id", 1)),
        int(getattr(tokenizer, "pad_token_id", 2)),
    }))
    return Backend(label=NATIVE_LABEL, adapter=adapter, tokenizer=tokenizer,
                   device=device, native_admissible=True, chat_mode=mode,
                   stop_token_ids=stop_ids, authorization=receipt)


def build_backend(kind: str, protocol: Protocol, dataset: PreparedDataset,
                  profile: WorkProfile, *, device: str, output_root: Path,
                  authorization_path: str | Path | None = None,
                  backend_evidence_path: str | Path | None = None,
                  allow_existing_output: bool = False) -> Backend:
    if kind == CPU_FIXTURE_LABEL:
        if device != "cpu":
            raise Refused("cpu_fixture", "cpu_fixture requires --device cpu")
        return build_cpu_fixture_backend(dataset, profile,
                                         seed=int(protocol.config["seed"]))
    if kind != NATIVE_LABEL:
        raise Refused("backend", f"unknown backend {kind!r}")
    if authorization_path is None:
        raise Refused("authorization", "native execution requires an authorization")
    evidence_path = backend_evidence_path or (
        protocol.path.parent.parent / str(protocol.config["launch"]["backend_evidence_path"]))
    return build_native_backend(
        protocol, dataset, authorization_path=authorization_path,
        backend_evidence_path=evidence_path, output_root=output_root,
        device=device, allow_existing_output=allow_existing_output)
# ---------------------------------------------------------------------------
# Memory construction, prompts, and arms

EXPERIMENT_SCOPE = "spark_text_memory_v1"


def _new_experiment(protocol: Protocol, adapter: Spark2_5Adapter, *,
                    plastic_payload: Mapping[str, Any] | None = None,
                    evidence_payload: Mapping[str, Any] | None = None) -> SpineMemoryExperiment:
    core = adapter.extension.spine
    experiment = SpineMemoryExperiment(
        scope=EXPERIMENT_SCOPE, spec=adapter.spine_spec, core=core,
        bank_experts=BANK_EXPERTS, bank_hidden_width=BANK_HIDDEN_WIDTH,
        seed=int(protocol.config["seed"]),
        read_records=min(READ_RECORDS, int(adapter.spine_spec.slots)),
        max_journal_records=int(protocol.config["memory_contract"]["journal_records_max"]),
        max_taught_evidence=int(protocol.config["memory_contract"]["retained_history_versions_max"]),
        max_cached_snapshots=int(protocol.config["memory_contract"]["snapshot_cache_entries_max"]),
        codec_version=core.codec_version,
        representation_version=core.representation_version)
    if plastic_payload is not None:
        experiment.controller.load_checkpoint(copy.deepcopy(dict(plastic_payload)))
    if evidence_payload is not None:
        experiment.evidence.load_checkpoint(copy.deepcopy(dict(evidence_payload)))
    return experiment


def _new_path(protocol: Protocol, adapter: Spark2_5Adapter, *,
              plastic_payload: Mapping[str, Any] | None = None,
              evidence_payload: Mapping[str, Any] | None = None,
              producer: VersionedTextProducer | None = None,
              response_lr: float = RESPONSE_LR) -> SparkTextMemoryPath:
    experiment = _new_experiment(
        protocol, adapter, plastic_payload=plastic_payload,
        evidence_payload=evidence_payload)
    return SparkTextMemoryPath(
        adapter, experiment, producer=producer or VersionedTextProducer(),
        response_lr=response_lr)


def _observe_rows(path: SparkTextMemoryPath, rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        path.observe_text(
            str(row["text"]), record_id=str(row["record_id"]),
            source_epoch=int(row["source_epoch"]),
            available_at=int(row["available_at"]),
            valid_from=int(row["valid_from"]),
            valid_until=None if row.get("valid_until") is None else int(row["valid_until"]),
            source_lineage=dict(row["provenance"]),
            durable_ref=dict(row["durable_ref"]),
            retain_canonical_text=True,
            payload_type="observation", verified=False,
            supersedes=None if row.get("supersedes") is None else str(row["supersedes"]))


def _add_procedures(path: SparkTextMemoryPath,
                    rows: Iterable[Mapping[str, Any]],
                    wanted: Iterable[str] | None = None) -> list[str]:
    wanted_set = None if wanted is None else set(wanted)
    added = []
    for row in rows:
        procedure_id = str(row["procedure_id"])
        if wanted_set is not None and procedure_id not in wanted_set:
            continue
        sources = tuple(str(value) for value in row["source_record_ids"])
        if not sources or any(source not in path.texts for source in sources):
            continue
        if procedure_id in path.procedures:
            continue
        path.add_procedure(
            procedure_id,
            applicability=row["applicability"],
            prerequisites=row["prerequisites"],
            exceptions=row["exceptions"],
            outcome=str(row["outcome"]),
            outcome_independently_checked=bool(row["outcome_independently_checked"]),
            outcome_checker=row.get("outcome_checker"),
            source_record_ids=sources)
        added.append(procedure_id)
    return added


def _apply_task_rows(path: SparkTextMemoryPath, rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in path.tasks:
            path.open_task(task_id, tuple(
                (str(goal["subgoal_id"]), str(goal["description"]))
                for goal in row["subgoals"]))
        completed = {item.subgoal_id for item in path.tasks[task_id].observed_completions}
        for completion in row["observed_completions"]:
            subgoal_id = str(completion["subgoal_id"])
            if subgoal_id in completed:
                continue
            sources = tuple(str(value) for value in completion["evidence_record_ids"])
            if not sources or any(source not in path.texts for source in sources):
                continue
            path.observe_subgoal_completion(
                task_id, subgoal_id,
                observation=str(completion["observation"]),
                observed_at=int(completion["observed_at"]),
                checker=str(completion["checker"]),
                evidence_record_ids=sources)


def prepare_development_memory(base_path: SparkTextMemoryPath,
                               protocol: Protocol,
                               dataset: PreparedDataset) -> dict[str, Any]:
    """Observe development text without teaching or touching retained P."""
    core = base_path.experiment.core
    source_bank = base_path.experiment.evidence
    bank = EvidenceBank(
        source_bank.scope,
        max_records=source_bank.max_records,
        max_payload_bytes=source_bank.max_payload_bytes,
        max_read_records=source_bank.max_read_records,
        max_history_records=source_bank.max_history_records,
        max_history_bytes=source_bank.max_history_bytes,
        max_cached_snapshots=source_bank.max_cached_snapshots,
        max_revocations=source_bank.max_revocations,
        max_record_metadata_bytes=source_bank.max_record_metadata_bytes,
        representation_version=core.representation_version,
        codec_version=core.codec_version)
    bank.load_checkpoint(base_path.experiment.evidence.checkpoint_payload())
    texts = dict(base_path.texts)
    observations = [row for row in dataset.observations()
                    if row.get("role") == "development"]
    for row in observations:
        produced = base_path.producer.produce(
            str(row["text"]), record_id=str(row["record_id"]),
            scope=bank.scope, source_epoch=int(row["source_epoch"]),
            available_at=int(row["available_at"]), valid_from=int(row["valid_from"]),
            valid_until=None if row.get("valid_until") is None else int(row["valid_until"]),
            source_lineage=dict(row["provenance"]),
            durable_ref=dict(row["durable_ref"]), retain_canonical_text=True,
            payload_type="observation", verified=False,
            supersedes=None if row.get("supersedes") is None else str(row["supersedes"]))
        if bank.admit(produced.record):
            if row.get("supersedes") is not None:
                texts.pop(str(row["supersedes"]), None)
            texts[produced.stored.record_id] = produced.stored
    procedures = dict(base_path.procedures)
    procedure_rows = dataset.procedures()
    for row in procedure_rows:
        sources = tuple(str(value) for value in row["source_record_ids"])
        if not sources or any(source not in texts for source in sources):
            continue
        from .architecture.spine_text import ProcedureMemory
        procedures[str(row["procedure_id"])] = ProcedureMemory(
            procedure_id=str(row["procedure_id"]),
            applicability=tuple(row["applicability"]),
            prerequisites=tuple(row["prerequisites"]),
            exceptions=tuple(row["exceptions"]), outcome=str(row["outcome"]),
            outcome_independently_checked=bool(row["outcome_independently_checked"]),
            outcome_checker=row.get("outcome_checker"),
            source_record_ids=sources)
    return {
        "evidence_payload": bank.checkpoint_payload(),
        "texts": texts,
        "procedures": procedures,
        "procedure_rows": procedure_rows,
        "tasks": copy.deepcopy(base_path.tasks),
        "observations": observations,
        "derived_caches_required": False,
    }


@dataclass
class ArmView:
    arm_id: str
    path: SparkTextMemoryPath
    mode: str
    source_prompt_removed: bool
    oracle: bool
    shuffled: bool
    inapplicable: bool
    procedure_ids: tuple[str, ...]
    inapplicable_procedure_id: str | None
    controller_source: str
    evidence_source: str
    information: dict[str, Any]


def _shuffled_payload(experiment: SpineMemoryExperiment) -> Mapping[str, Any]:
    records = list(experiment.evidence.active)
    if len(records) < 2:
        return experiment.evidence.checkpoint_payload()
    payloads = [record.payload.clone() for record in records]
    permutation = [(index + 1) % len(records) for index in range(len(records))]
    experiment.evidence.clear()
    for index, record in enumerate(records):
        lineage = copy.deepcopy(dict(record.source_lineage))
        lineage["shuffled_from"] = records[permutation[index]].record_id
        lineage["shuffle_policy"] = "deterministic_rotation_same_slots_bytes_without_gold_ids"
        shuffled = EvidenceRecord(
            record_id=record.record_id, scope=record.scope,
            payload=payloads[permutation[index]], source_epoch=record.source_epoch,
            available_at=record.available_at, valid_from=record.valid_from,
            producer_revision=record.producer_revision,
            representation_version=record.representation_version,
            codec_version=record.codec_version,
            payload_type=record.payload_type, source_lineage=lineage,
            valid_until=record.valid_until, supersedes=None)
        experiment.observe(shuffled)
    return experiment.evidence.checkpoint_payload()


def _inapplicable_procedure(procedure_rows: Sequence[Mapping[str, Any]],
                            case: Mapping[str, Any]) -> str | None:
    for row in procedure_rows:
        if row.get("world_id") != case.get("world_id"):
            return str(row["procedure_id"])
    return None


def build_arm_views(protocol: Protocol, adapter: Spark2_5Adapter,
                    base_path: SparkTextMemoryPath,
                    development_memory: Mapping[str, Any],
                    *, trained_plastic: Mapping[str, Any],
                    pre_plastic: Mapping[str, Any]) -> dict[str, ArmView]:
    core = adapter.extension.spine
    evidence_payload = development_memory["evidence_payload"]
    texts = development_memory["texts"]
    procedures = development_memory["procedures"]
    procedure_rows = development_memory["procedure_rows"]
    tasks = development_memory["tasks"]
    views: dict[str, ArmView] = {}
    for arm_id in ARMS:
        if arm_id == "full_eligible_context":
            mode, plastic, evidence, controller_source = "none", pre_plastic, evidence_payload, "pre_teaching"
        elif arm_id == ORACLE_ARM:
            mode, plastic, evidence, controller_source = "E", trained_plastic, evidence_payload, "trained"
        elif arm_id == "spine_E_only":
            mode, plastic, evidence, controller_source = "E", pre_plastic, evidence_payload, "pre_teaching_reset"
        elif arm_id == "source_free_P_only":
            mode, plastic, evidence, controller_source = "P", trained_plastic, None, "trained"
        elif arm_id == "E_plus_P":
            mode, plastic, evidence, controller_source = "E_plus_P", trained_plastic, evidence_payload, "trained"
        elif arm_id == "no_update":
            mode, plastic, evidence, controller_source = "P", pre_plastic, None, "pre_teaching"
        elif arm_id == "reset_P":
            mode, plastic, evidence, controller_source = "E_plus_P", pre_plastic, evidence_payload, "pre_teaching_reset"
        elif arm_id == "shuffled_memory":
            mode, plastic, evidence, controller_source = "E", pre_plastic, evidence_payload, "pre_teaching_reset"
        elif arm_id == "inapplicable_procedure":
            mode, plastic, evidence, controller_source = "none", pre_plastic, None, "pre_teaching_reset"
        else:  # pragma: no cover - ARMS is closed
            raise RunnerError(f"unknown arm {arm_id}")

        path = _new_path(
            protocol, adapter, plastic_payload=plastic,
            evidence_payload=evidence, producer=base_path.producer,
            response_lr=RESPONSE_LR)
        if mode in ("E", "E_plus_P", "none") and evidence is not None:
            path.texts = dict(texts)
            path.procedures = dict(procedures)
            path.tasks = copy.deepcopy(tasks)
        else:
            path.texts = {}
            path.procedures = {}
            path.tasks = {}
        if arm_id == "shuffled_memory":
            path.experiment.evidence.load_checkpoint(copy.deepcopy(dict(evidence_payload)))
            shuffled_payload = _shuffled_payload(path.experiment)
            # Recreate the path so the shuffled bank is the owned state.
            path = _new_path(
                protocol, adapter, plastic_payload=plastic,
                evidence_payload=shuffled_payload, producer=base_path.producer,
                response_lr=RESPONSE_LR)
            path.texts = dict(texts)
            path.procedures = dict(procedures)
            path.tasks = copy.deepcopy(tasks)
        if arm_id == "inapplicable_procedure":
            # Scope control: the verified procedure belongs to another world.
            wrong = {row["procedure_id"] for row in procedure_rows}
            path.procedures = {key: value for key, value in procedures.items()
                               if key in wrong}
        views[arm_id] = ArmView(
            arm_id=arm_id, path=path, mode=mode,
            source_prompt_removed=arm_id != "full_eligible_context",
            oracle=arm_id == ORACLE_ARM,
            shuffled=arm_id == "shuffled_memory",
            inapplicable=arm_id == "inapplicable_procedure",
            procedure_ids=tuple(sorted(path.procedures)),
            inapplicable_procedure_id=None,
            controller_source=controller_source,
            evidence_source=("source_free" if arm_id == "source_free_P_only"
                             else "shuffled" if arm_id == "shuffled_memory"
                             else "none" if arm_id in ("no_update", "inapplicable_procedure")
                             else "eligible"),
            information={
                "arm_id": arm_id,
                "oracle_selected": arm_id == ORACLE_ARM,
                "oracle_label": ("diagnostic_upper_bound_not_product_routing"
                                 if arm_id == ORACLE_ARM else None),
                "source_prompt_removed": arm_id != "full_eligible_context",
                "evidence_source": ("source_free" if arm_id == "source_free_P_only"
                                    else "shuffled" if arm_id == "shuffled_memory"
                                    else "none" if arm_id in ("no_update", "inapplicable_procedure")
                                    else "eligible"),
                "controller": controller_source,
                "answer_aware_record_ids_used": arm_id == ORACLE_ARM,
                "answer_cache_used": False,
                "source_kv_used": False,
                "derived_caches_required": False,
            })
    return views


def _procedure_hint(procedure: Any) -> str:
    return (
        f"Procedure {procedure.procedure_id}: applicability "
        f"{'; '.join(procedure.applicability)}. Prerequisites "
        f"{'; '.join(procedure.prerequisites) or 'none'}. Exceptions "
        f"{'; '.join(procedure.exceptions) or 'none'}. Verified outcome: "
        f"{procedure.outcome}.")


def _eligible_text(view: ArmView, case: Mapping[str, Any]) -> str:
    records = [
        record for record in view.path.experiment.evidence.active
        if record.valid_from <= int(case["query_time"])
        and record.available_at <= int(case["query_time"])
        and (record.valid_until is None
             or int(case["query_time"]) <= record.valid_until)
    ]
    records.sort(key=lambda record: (record.available_at, record.source_epoch,
                                    record.record_id))
    lines = []
    for record in records:
        stored = view.path.texts.get(record.record_id)
        if stored is not None and stored.canonical_text:
            lines.append(stored.canonical_text)
    return "\n".join(lines)


def _prompt_for(view: ArmView, case: Mapping[str, Any],
                dataset: PreparedDataset) -> tuple[str, dict[str, Any]]:
    question = str(case["question"])
    context = ""
    procedure_id = case.get("procedure_id")
    information = {
        "source_text_included": False,
        "procedure_hint": None,
        "removed_source_text": view.source_prompt_removed,
    }
    if view.arm_id == "full_eligible_context":
        context = _eligible_text(view, case)
        information["source_text_included"] = bool(context)
        information["eligible_source_text_sha256"] = hashlib.sha256(
            context.encode("utf-8")).hexdigest() if context else None
    elif view.oracle:
        if procedure_id and procedure_id in view.path.procedures:
            hint = _procedure_hint(view.path.procedures[procedure_id])
            context = hint
            information["procedure_hint"] = procedure_id
    elif view.inapplicable:
        wrong = None
        for row in dataset.procedures():
            if row.get("world_id") != case.get("world_id"):
                wrong = view.path.procedures.get(str(row["procedure_id"]))
                if wrong is not None:
                    break
        if wrong is not None:
            context = _procedure_hint(wrong)
            information["procedure_hint"] = wrong.procedure_id
            information["inapplicable_world"] = True
    content = ((context + "\n\n") if context else "") + (
        f"Question: {question}\nAnswer:")
    information["question_sha256"] = hashlib.sha256(question.encode()).hexdigest()
    return content, information


def _prepare_query(view: ArmView, backend: Backend, prompt: str, *,
                   query_time: int, profile: WorkProfile,
                   ledger: WorkLedger, case: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    ids, mask = backend.encode_prompt(prompt, profile.context_tokens)
    path = view.path
    path.experiment.invalidate(reason="no_cache_reuse")
    host_ids = None
    if view.oracle:
        host_ids = tuple(str(value) for value in case.get("support_record_ids", ()))
    if view.inapplicable:
        host_ids = ()
    token_times = tuple(int(query_time) for _ in range(int(ids.shape[1])))
    selected = ()
    eligible = ()
    if view.mode == "P":
        query = path.prepare_query(
            str(case["question"]), ids, mask, token_available_at=token_times,
            query_time=int(query_time), memory_source="P",
            plastic_expert_id=PLASTIC_EXPERT, enabled_branches=("memory",))
    elif view.mode == "E_plus_P":
        p_query = path.prepare_query(
            str(case["question"]), ids, mask, token_available_at=token_times,
            query_time=int(query_time), memory_source="P",
            plastic_expert_id=PLASTIC_EXPERT, enabled_branches=("memory",))
        e_query = path.queries.build(
            query_id=f"{p_query.query_id}-e", question=str(case["question"]),
            input_ids=ids, attention_mask=mask, token_available_at=token_times,
            query_time=int(query_time), native_hidden=p_query.native_hidden,
            memory_source="E", plastic_expert_id=None,
            enabled_branches=("memory",), host_eligible_record_ids=None,
            semantic_top_k=None)
        combined = SpineForwardContext(
            path.experiment.core, e_query.context.packet,
            e_query.context.snapshot, enabled_branches=("memory",),
            memory_slots=p_query.context.slots["memory"], memory_source="P",
            memory_availability=p_query.context.branch_availability["memory"])
        combined.charges["e_codec_calls"] = e_query.context.charges["codec_calls"]
        combined.charges["p_read_calls"] = 1
        query = replace(e_query, context=combined, memory_source="E_plus_P")
        selected = e_query.selected_record_ids
        eligible = e_query.eligible_record_ids
    elif view.mode == "E":
        query = path.prepare_query(
            str(case["question"]), ids, mask, token_available_at=token_times,
            query_time=int(query_time), memory_source="E",
            enabled_branches=("memory",),
            host_eligible_record_ids=host_ids, semantic_top_k=None)
        selected = query.selected_record_ids
        eligible = query.eligible_record_ids
    else:
        query = path.prepare_query(
            str(case["question"]), ids, mask, token_available_at=token_times,
            query_time=int(query_time), memory_source="E",
            enabled_branches=(), host_eligible_record_ids=host_ids,
            semantic_top_k=None)
        selected = query.selected_record_ids
        eligible = query.eligible_record_ids
    ledger.query_sensing(positions=int(ids.shape[1]))
    ledger.retrieval()
    ledger.charge("selected_records", len(selected))
    ledger.charge("omitted_records", max(0, len(eligible) - len(selected)))
    details = {
        "query_id": query.query_id,
        "memory_source": query.memory_source,
        "context_memory_source": query.context.memory_source,
        "eligible_record_ids": list(eligible),
        "selected_record_ids": list(selected),
        "semantic_selection": query.semantic_selection,
        "evidence_revision": query.evidence_revision,
        "plastic_revision": query.plastic_revision,
        "oracle_selected": bool(view.oracle),
        "host_eligible_record_ids": (None if host_ids is None else list(host_ids)),
        "source_prompt_removed": view.source_prompt_removed,
        "context_charges": dict(query.context.charges),
        "codec_calls_before_injection": query.codec_calls_before_injection,
        "native_query_passes": query.native_query_passes,
        "removed_source_text": view.source_prompt_removed,
        "prompt_tokens": int(ids.shape[1]),
    }
    return query, details


def _extend_context(context: SpineForwardContext, positions: int = 1) -> SpineForwardContext:
    """Extend an immutable prompt context for one generated token.

    The selected record identities and payload slots are not recomputed.  Only
    the query axis and its availability are extended, so generation preserves
    the pre-response selection barrier.
    """
    packet = context.packet
    last_availability = packet.availability[:, -1:, :]
    availability = torch.cat(
        [packet.availability,
         last_availability.expand(-1, positions, -1)], dim=1)
    extended_packet = replace(packet, availability=availability)
    memory_slots = None
    memory_availability = None
    if context.memory_source in ("P", "external") and "memory" in context.slots:
        slots = context.slots["memory"]
        last_slot = slots[:, -1:, :, :]
        memory_slots = torch.cat(
            [slots, last_slot.expand(-1, positions, -1, -1)], dim=1)
        branch_availability = context.branch_availability["memory"]
        last_branch = branch_availability[:, -1:, :]
        memory_availability = torch.cat(
            [branch_availability, last_branch.expand(-1, positions, -1)], dim=1)
    return SpineForwardContext(
        context.core, extended_packet, context.snapshot,
        enabled_branches=context.enabled_branches,
        memory_slots=memory_slots,
        memory_source=(None if memory_slots is None else context.memory_source),
        memory_availability=memory_availability)


def greedy_generate(backend: Backend, query: Any, *, max_new_tokens: int,
                    ledger: WorkLedger | None = None) -> dict[str, Any]:
    """Greedy decode for exactly max_new_tokens with stop ids masked out."""
    ids = query.input_ids
    mask = query.attention_mask
    context = query.context
    stop_ids = tuple(int(value) for value in backend.stop_token_ids)
    generated: list[int] = []
    steps: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                if step:
                    context = _extend_context(context)
                logits = backend.adapter.forward_text(
                    ids, mask, use_projector=False, use_ports=False,
                    spine=context, use_spine=True,
                    enabled_spine_branches=context.enabled_branches)
                position_logits = logits[0, -1]
                masked = position_logits.clone()
                if stop_ids:
                    masked[torch.tensor(stop_ids, dtype=torch.long,
                                        device=masked.device)] = float("-inf")
                if not bool(torch.isfinite(masked).any()):
                    raise GenerationFailure(
                        "all terminal-stop logits were excluded",
                        partial={"steps": steps, "token_ids": generated})
                next_id = int(torch.argmax(masked).item())
                top = torch.topk(position_logits, k=min(5, position_logits.numel()))
                steps.append({
                    "step": step,
                    "token_id": next_id,
                    "score": float(position_logits[next_id].item()),
                    "top_token_ids": [int(value) for value in top.indices.tolist()],
                    "top_scores": [float(value) for value in top.values.tolist()],
                    "positions": int(ids.shape[1]),
                    "terminal_stop_ids_excluded": list(stop_ids),
                })
                generated.append(next_id)
                if ledger is not None:
                    ledger.charge("autoregressive_forwards")
                    ledger.charge("forward_calls")
                    ledger.charge("generated_tokens")
                    ledger.charge("processed_text_positions", int(ids.shape[1]))
                ids = torch.cat((ids, torch.tensor(
                    [[next_id]], dtype=torch.long, device=ids.device)), dim=1)
                mask = torch.cat((mask, torch.ones(
                    (1, 1), dtype=mask.dtype, device=mask.device)), dim=1)
    except GenerationFailure:
        raise
    except Exception as error:  # preserve the partial trajectory
        decoded = backend.decode(generated) if generated else ""
        raise GenerationFailure(
            f"generation failed at step {len(generated)}: {error}",
            partial={"steps": steps, "token_ids": generated, "decoded": decoded},
            cause=error) from error
    text = backend.decode(generated)
    return {
        "token_ids": generated,
        "steps": steps,
        "decoded": text,
        "terminal_stop_ids_excluded": list(stop_ids),
        "terminal_stop_tokens_emitted": sorted(set(generated) & set(stop_ids)),
        "autoregressive_forwards": len(steps),
        "generated_tokens": len(generated),
        "processed_positions": sum(step["positions"] for step in steps),
    }


def normalize_answer(text: str) -> str:
    value = str(text).lower()
    value = re.sub(r"[^\w\s]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def score_answer(expected: str, generated: str) -> dict[str, Any]:
    target = normalize_answer(expected)
    actual = normalize_answer(generated)
    exact = actual == target
    contains = bool(target) and (
        target in actual.split() if len(target.split()) == 1 else target in actual)
    return {
        "expected_normalized": target,
        "generated_normalized": actual,
        "exact_match": exact,
        "contains_expected": contains,
        "matched": exact or contains,
        "scoring": "normalized_exact_or_contains",
    }
# ---------------------------------------------------------------------------
# Durable runner state and checkpoints


class JsonlAppender:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def append(self, row: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlAppender":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _run_contract(protocol: Protocol, dataset: PreparedDataset,
                  backend: Backend) -> dict[str, Any]:
    core = backend.adapter.extension.spine
    return {
        "schema": SPINE_SCHEMA,
        "spine_schema": SPINE_SCHEMA,
        "model_identity": backend.adapter.durable_identity(),
        "codec_version": core.codec_version,
        "representation_version": core.representation_version,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "prepared_manifest_sha256": dataset.manifest_sha256,
        "backend": backend.label,
        "native_admissible": bool(backend.native_admissible),
    }


def _sealed_source_reference(protocol: Protocol, dataset: PreparedDataset,
                             backend: Backend, *,
                             authorization: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if backend.label == CPU_FIXTURE_LABEL:
        return {
            "identity": CPU_FIXTURE_LABEL,
            "official_source": False,
            "native_admissible": False,
            "model_identity": backend.adapter.durable_identity(),
            "fixture_seed": backend.fixture_seed,
            "derived_caches_required": False,
            "note": "deterministic tiny CPU fixture; never native-admissible",
        }
    source = protocol.source_identity
    return {
        "identity": "official_fp8",
        "official_source": True,
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "local_path": source["local_path"],
        "config_sha256": source["config_sha256"],
        "weight_sha256": source["weight_sha256"],
        "execution_format": source["execution_format"],
        "required_backend": source["required_backend"],
        "native_admissible": True,
        "model_identity": backend.adapter.durable_identity(),
        "authorization": copy.deepcopy(dict(authorization or {})),
        "derived_caches_required": False,
    }


def _producer_reference(path: SparkTextMemoryPath) -> dict[str, Any]:
    return {
        "producer_revision": path.producer.producer_revision,
        "representation_version": path.producer.representation_version,
        "codec_version": path.producer.codec_version,
        "max_characters": path.producer.max_characters,
        "derived_caches_required": False,
    }


def _durable_text_references(path: SparkTextMemoryPath) -> list[dict[str, Any]]:
    rows = []
    for record_id in sorted(path.texts):
        stored = path.texts[record_id]
        rows.append({
            "record_id": record_id,
            "text_sha256": stored.text_sha256,
            "canonical_text": stored.canonical_text is not None,
            "durable_ref": copy.deepcopy(stored.durable_ref),
            "producer_revision": stored.producer_revision,
            "representation_version": stored.representation_version,
            "codec_version": stored.codec_version,
        })
    return rows


def _plastic_digest(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()

    def visit(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().to("cpu", torch.float32).contiguous()
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                digest.update(str(key).encode("utf-8"))
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode("utf-8"))

    visit(payload)
    return digest.hexdigest()


def _artifact_bytes(*rows: Mapping[str, Any]) -> int:
    total = 0
    for row in rows:
        total += int(row.get("bytes", 0))
    return total


def _prune_checkpoints(checkpoint_root: Path,
                       writes: Sequence[Mapping[str, Any]],
                       keep: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain the last ``keep`` checkpoint operation directories."""
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for write in writes:
        relative = str(write["relative"])
        prefix = str(Path(relative).parent)
        if prefix in seen:
            continue
        seen.add(prefix)
        groups.append({"prefix": prefix, "relative": relative,
                       "artifacts": list(write.get("artifacts", []))})
    evicted: list[dict[str, Any]] = []
    retained = groups[-keep:] if keep > 0 else []
    for group in groups[:-keep] if keep > 0 else groups:
        path = checkpoint_root / group["prefix"]
        if path.exists():
            shutil.rmtree(path)
        evicted.append(group)
    return retained, evicted


def _write_failure(output: Path, *, stage: str, error: BaseException,
                   ledger: WorkLedger, partial: Mapping[str, Any] | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": REPORT_FORMAT,
        "status": "failed",
        "stage": stage,
        "execution_complete": False,
        "error": {"type": type(error).__name__, "message": str(error),
                  "traceback": traceback.format_exc()},
        "partial": copy.deepcopy(dict(partial or {})),
        "work": ledger.report(),
        "preserved_at_utc": _utc(),
    }
    publication.write_json(output, "failure.json", payload)
    return payload


@dataclass
class FitState:
    path: SparkTextMemoryPath
    run_contract: dict[str, Any]
    receipt: dict[str, Any]
    pre_plastic: Mapping[str, Any]
    trained_plastic: Mapping[str, Any]
    checkpoints: list[dict[str, Any]]
    ledger: WorkLedger


def _training_content(observation_texts: Sequence[str], question: str) -> str:
    source = "\n".join(text for text in observation_texts if text)
    return (source + "\n\n" if source else "") + f"Question: {question}\nAnswer:"


def _teacher_forced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        supervised = labels != -100
        count = int(supervised.sum().item())
        if not count:
            return {"positions": 0, "correct": 0, "token_accuracy": None}
        predictions = logits[0].argmax(dim=-1)
        mask = supervised[0]
        correct = int((predictions[mask] == labels[0][mask]).sum().item())
    return {"positions": count, "correct": correct,
            "token_accuracy": correct / count}


def _verify_fixture_outcome(expected: str, declared: str) -> dict[str, Any]:
    score = score_answer(expected, declared)
    return {"checker": "fixture-rule-table", "independent": True,
            "expected": expected, "declared": declared,
            "verified": score["exact_match"], "scoring": score}


def _plastic_target(path: SparkTextMemoryPath, query: Any,
                    answer: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Verified-answer target for the selected P expert.

    The target is the core memory projection of the versioned producer vector
    for the independently verified answer text.  It is not a hidden-state,
    label-free, or randomly fabricated target.
    """
    with torch.no_grad():
        answer_vector = path.producer.vector(answer)
        weight = path.experiment.core.memory_projection.weight
        projected = path.experiment.core.memory_projection(
            answer_vector.to(device=weight.device, dtype=torch.float32))
        features = path.plastic_features(query)
        target = projected.view(1, 1, -1).expand_as(features).detach().clone()
    digest = hashlib.sha256()
    value = target.detach().cpu().contiguous().float().numpy()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return target, {
        "representation": "core.memory_projection(versioned_answer_vector)",
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "producer_revision": path.producer.producer_revision,
        "representation_version": path.producer.representation_version,
        "codec_version": path.producer.codec_version,
        "target_sha256": digest.hexdigest(),
        "target_shape": list(target.shape),
        "target_positions": int(target.shape[1]),
    }


def run_fit(protocol: Protocol, dataset: PreparedDataset, backend: Backend,
            output: Path, profile: WorkProfile, ledger: WorkLedger, *,
            response_lr: float = RESPONSE_LR) -> FitState:
    """Four chronological events, each with a response update and a real P update."""
    if response_lr <= 0:
        raise Refused("zero_learning_rate",
                      "a nonpositive response learning rate cannot claim an update")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output
    fit_dir = output / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    path = _new_path(protocol, backend.adapter, response_lr=response_lr,
                     producer=VersionedTextProducer())
    audit = path.response_optimizer.audit()
    if any(float(group["lr"]) <= 0 for group in audit["groups"]):
        raise Refused("zero_learning_rate", "declared response groups must have lr > 0")
    run_contract = _run_contract(protocol, dataset, backend)
    observations = {row["record_id"]: row for row in dataset.observations()}
    procedure_rows = dataset.procedures()
    task_rows = dataset.tasks()
    pre_plastic = copy.deepcopy(path.experiment.controller.checkpoint_payload())
    events: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    ledger.set_phase("cold_start")
    trajectory = JsonlAppender(fit_dir / "trajectories.jsonl")
    try:
        for event in dataset.fit_events():
            row_observations = [observations[record_id]
                                for record_id in event["observation_record_ids"]]
            _observe_rows(path, row_observations)
            _add_procedures(path, procedure_rows)
            _apply_task_rows(path, task_rows)
            content = _training_content(
                [row["text"] for row in row_observations], str(event["question"]))
            ids, mask, labels = backend.encode_training(
                content, str(event["answer"]), profile.context_tokens)
            token_times = tuple(int(event["sequence"])
                                for _ in range(int(ids.shape[1])))
            query = path.prepare_query(
                str(event["question"]), ids, mask,
                token_available_at=token_times,
                query_time=int(event["sequence"]), memory_source="P",
                plastic_expert_id=PLASTIC_EXPERT, enabled_branches=("memory",))
            ledger.query_sensing(positions=int(ids.shape[1]))
            ledger.retrieval()
            ledger.charge("selected_records", 0)
            ledger.charge("omitted_records", 0)
            ledger.training(positions=int(ids.shape[1]))
            try:
                result = path.response_step(query, labels=labels)
            except BaseException as error:
                ledger.fail()
                _write_failure(output, stage="fit", error=error, ledger=ledger,
                               partial={"event_id": event["event_id"],
                                        "completed_events": events})
                raise
            ledger.backward(positions=int(ids.shape[1]))
            ledger.optimizer()
            if not result["actual_parameter_update"]:
                ledger.counters["no_op_steps"] += 1
                error = RunnerError(
                    f"fit event {event['event_id']} performed a zero-change step")
                _write_failure(output, stage="fit", error=error, ledger=ledger,
                               partial={"event_id": event["event_id"],
                                        "result": {key: value for key, value in result.items()
                                                   if key not in ("response",)}})
                raise error
            teacher_forced = _teacher_forced_accuracy(
                result["response"].logits, labels)
            verification = _verify_fixture_outcome(
                str(event["expected"]), str(event["answer"]))
            ledger.verification()
            if not verification["verified"]:
                raise RunnerError(
                    f"fit event {event['event_id']} answer is not independently verified")
            # The response update invalidated the prepared query.  Recreate it
            # with one explicit native query-sensing forward before P learning.
            teaching_query = path.prepare_query(
                str(event["question"]), ids, mask,
                token_available_at=token_times,
                query_time=int(event["sequence"]), memory_source="P",
                plastic_expert_id=PLASTIC_EXPERT, enabled_branches=("memory",))
            ledger.query_sensing(positions=int(ids.shape[1]))
            target, target_evidence = _plastic_target(
                path, teaching_query, str(event["answer"]))
            plastic_features = path.plastic_features(teaching_query)
            feature_vectors = int(plastic_features.shape[1])
            ledger.plastic_forward(feature_vectors=feature_vectors)
            try:
                teaching = path.teach_query(
                    teaching_query, event_id=f"{event['event_id']}-plastic",
                    target=target, expert_id=PLASTIC_EXPERT, steps=1)
            except BaseException as error:
                ledger.fail()
                _write_failure(output, stage="fit", error=error, ledger=ledger,
                               partial={"event_id": event["event_id"],
                                        "completed_events": events,
                                        "plastic_target": target_evidence})
                raise
            plastic_evidence = teaching["learning_evidence"]
            if teaching["outcome"]["retried"]:
                ledger.retry_zero_new_work(f"{event['event_id']}-plastic")
                raise RunnerError(
                    f"fit event {event['event_id']} plastic teaching was retried")
            ledger.plastic_backward(feature_vectors=feature_vectors)
            ledger.optimizer()
            ledger.charge("plastic_fit_evaluation_forwards",
                          int(plastic_evidence.get("fit_evaluation_forward_calls", 0)))
            if not plastic_evidence.get("parameter_update") \
                    or not plastic_evidence.get("changed_trainable_names"):
                ledger.counters["no_op_steps"] += 1
                raise RunnerError(
                    f"fit event {event['event_id']} performed a zero-change P step")
            checkpoint_relative = f"fit/event-{int(event['sequence']):02d}/state"
            checkpoint = path.checkpoint(checkpoint_root, checkpoint_relative,
                                         run_contract=run_contract)
            artifacts = [checkpoint["experiment"]["artifact"],
                         checkpoint["text_artifact"]]
            ledger.checkpoint_write(bytes_written=_artifact_bytes(*artifacts))
            writes.append({"relative": checkpoint_relative,
                           "artifacts": artifacts,
                           "event_id": event["event_id"]})
            row = {
                "stage": "fit", "event_id": event["event_id"],
                "sequence": int(event["sequence"]), "world_id": event["world_id"],
                "expected": event["expected"], "answer": event["answer"],
                "verification": verification,
                "teacher_forced": teacher_forced,
                "loss": result["loss"],
                "execution_complete": result["execution_complete"],
                "physical_optimizer_steps": result["physical_optimizer_steps"],
                "actual_parameter_update": result["actual_parameter_update"],
                "changed_trainable_names": list(result["changed_trainable_names"]),
                "changed_expert_names": list(
                    plastic_evidence["changed_trainable_names"]),
                "plastic_optimizer_steps": list(path.experiment.bank.expert_steps),
                "plastic_clock_events": int(teaching["outcome"]["eligible_events"]),
                "plastic_revision": int(teaching["outcome"]["revision"]),
                "plastic_teaching": {
                    "target": target_evidence,
                    "outcome": copy.deepcopy(teaching["outcome"]),
                    "learning_evidence": copy.deepcopy(plastic_evidence),
                    "feature_vectors": feature_vectors,
                },
                "base_unchanged": result["base_unchanged"],
                "gradient_audit": result["gradient_audit"],
                "plastic_nonzero_gradient_names": list(
                    result["plastic_nonzero_gradient_names"]),
                "charges": dict(result["charges"]),
                "checkpoint": {"relative": checkpoint_relative,
                               "artifacts": artifacts},
                "learning_evidence": copy.deepcopy(result["learning_evidence"]),
            }
            events.append(row)
            trajectory.append(row)
            ledger.charge("fit_events")
        if len(events) != profile.fit_events:
            raise RunnerError(f"fit executed {len(events)} events")
        # Final consolidated checkpoint + extension-only runner state.
        final_relative = "fit/final/state"
        final = path.checkpoint(checkpoint_root, final_relative,
                                run_contract=run_contract)
        final_artifacts = [final["experiment"]["artifact"],
                           final["text_artifact"]]
        ledger.checkpoint_write(bytes_written=_artifact_bytes(*final_artifacts))
        writes.append({"relative": final_relative,
                       "artifacts": final_artifacts, "event_id": "final"})
        trained_plastic = copy.deepcopy(
            path.experiment.controller.checkpoint_payload())
        runner_state = {
            "format": RUNNER_STATE_FORMAT,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "prepared_manifest_sha256": dataset.manifest_sha256,
            "run_contract": run_contract,
            "profile": {
                "label": profile.label,
                "context_tokens": profile.context_tokens,
                "max_new_tokens": profile.max_new_tokens,
                "native_admissible": profile.native_admissible,
            },
            "sealed_source": _sealed_source_reference(
                protocol, dataset, backend,
                authorization=backend.authorization),
            "producer": _producer_reference(path),
            "pre_fit_plastic": pre_plastic,
            "plastic_digest": _plastic_digest(trained_plastic),
            "pre_fit_plastic_digest": _plastic_digest(pre_plastic),
            "fit_checkpoint": {"relative": final_relative,
                               "artifacts": final_artifacts},
            "durable_text_references": _durable_text_references(path),
            "derived_caches_required": False,
            "arm_policy": {
                "arms": list(ARMS),
                "oracle_arm": ORACLE_ARM,
                "answer_aware_record_ids_only_in_oracle": True,
                "full_context_and_oracle_are_controls": True,
                "source_prompt_removed_for_memory_arms": True,
                "shuffle_policy": "deterministic_rotation_same_slots_bytes",
                "reset_restores_pre_teaching_parameters_and_optimizer": True,
            },
            "query_schedule": {
                "known_text_observations": True,
                "query_completion_barrier": True,
                "immutable_selection_before_response": True,
                "per_token_requery": False,
            },
            "written_at_utc": _utc(),
        }
        state_artifact = publication.write_torch(
            checkpoint_root, "fit/final/runner_state.pt", runner_state)
        writes[-1]["runner_state"] = state_artifact.as_dict()
        retained, evicted = _prune_checkpoints(
            checkpoint_root, writes, profile.retained_checkpoints)
        ledger.counters["retained_checkpoint_artifacts"] = len(retained)
        changed = sorted({name for event in events
                          for name in event["changed_trainable_names"]})
        changed_expert = sorted({name for event in events
                                 for name in event["changed_expert_names"]})
        response_steps = sum(int(event["physical_optimizer_steps"]) for event in events)
        plastic_steps = sum(int(event["plastic_teaching"]["learning_evidence"].get(
            "optimizer_steps", 0)) for event in events)
        plastic_updates = sum(1 for event in events if
                              event["plastic_teaching"]["learning_evidence"].get(
                                  "parameter_update"))
        plastic_clock = (int(events[-1]["plastic_clock_events"]) if events else 0)
        plastic_revisions = [int(event["plastic_revision"]) for event in events]
        receipt = {
            "format": FIT_RECEIPT_FORMAT,
            "status": "fit_executed",
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "prepared_manifest_sha256": dataset.manifest_sha256,
            "backend": backend.label,
            "native_admissible": bool(backend.native_admissible),
            "execution_complete": True,
            "fit_events": len(events),
            "optimizer_steps": ledger.counters["optimizer_steps"],
            "optimizer_scopes": {
                "response": response_steps,
                "plastic": plastic_steps,
                "total": ledger.counters["optimizer_steps"],
            },
            "declared_response_updates": len(events),
            "actual_parameter_updates": sum(
                1 for event in events if event["actual_parameter_update"]),
            "declared_plastic_updates": len(events),
            "actual_plastic_updates": plastic_updates,
            "changed_trainable_names": changed,
            "changed_tensor_count": len(changed),
            "changed_expert_names": changed_expert,
            "plastic_changed_tensor_count": len(changed_expert),
            "train_fit": {
                "evaluated": True,
                "teacher_forced": [event["teacher_forced"] for event in events],
                "losses": [event["loss"] for event in events],
                "diagnostic_only": True,
            },
            "transfer": {"evaluated": False, "improved": None},
            "retention": {"evaluated": False, "retained": None},
            "useful_learning": "unevaluated",
            "learning_success": None,
            "run_contract": run_contract,
            "fit_checkpoint": {"relative": final_relative,
                               "artifacts": final_artifacts},
            "runner_state": state_artifact.as_dict(),
            "checkpoint_writes": ledger.counters["checkpoint_writes"],
            "retained_checkpoints": [row["prefix"] for row in retained],
            "evicted_checkpoints": [row["prefix"] for row in evicted],
            "derived_caches_required": False,
            "durable_text_references": _durable_text_references(path),
            "plastic_learning": {
                "evaluated": True,
                "declared_updates": len(events),
                "actual_updates": plastic_updates,
                "optimizer_steps": plastic_steps,
                "forward_calls": sum(int(event["plastic_teaching"][
                    "learning_evidence"].get("forward_calls", 0)) for event in events),
                "backward_calls": sum(int(event["plastic_teaching"][
                    "learning_evidence"].get("backward_calls", 0)) for event in events),
                "fit_evaluation_forward_calls": ledger.counters[
                    "plastic_fit_evaluation_forwards"],
                "changed_trainable_names": changed_expert,
                "changed_tensor_count": len(changed_expert),
                "clock_events": plastic_clock,
                "revisions": plastic_revisions,
                "target_representation": (
                    "core.memory_projection(versioned_answer_vector)"),
                "pre_fit_plastic_digest": _plastic_digest(pre_plastic),
                "retained_plastic_digest": _plastic_digest(trained_plastic),
                "plastic_changed": _plastic_digest(pre_plastic)
                != _plastic_digest(trained_plastic),
                "per_event": [{
                    "event_id": event["event_id"],
                    "target": event["plastic_teaching"]["target"],
                    "outcome": event["plastic_teaching"]["outcome"],
                    "learning_evidence": event["plastic_teaching"]["learning_evidence"],
                } for event in events],
                "learning_success": all(
                    bool(event["plastic_teaching"]["learning_evidence"].get(
                        "learning_success"))
                    for event in events),
            },
            "events": events,
            "written_at_utc": _utc(),
        }
        publication.write_json(checkpoint_root, "fit/fit_receipt.json", receipt)
        return FitState(path=path, run_contract=run_contract, receipt=receipt,
                        pre_plastic=pre_plastic, trained_plastic=trained_plastic,
                        checkpoints=writes, ledger=ledger)
    finally:
        trajectory.close()
# ---------------------------------------------------------------------------
# Development


def _execute_request(view: ArmView, backend: Backend, dataset: PreparedDataset,
                     case: Mapping[str, Any], profile: WorkProfile,
                     ledger: WorkLedger, *, arm_id: str | None = None,
                     task_progress: bool = True) -> dict[str, Any]:
    prompt, information = _prompt_for(view, case, dataset)
    query, query_details = _prepare_query(
        view, backend, prompt, query_time=int(case["query_time"]),
        profile=profile, ledger=ledger, case=case)
    ledger.charge("generation_requests")
    try:
        raw = greedy_generate(backend, query,
                              max_new_tokens=profile.max_new_tokens,
                              ledger=ledger)
    except GenerationFailure as failure:
        ledger.fail()
        row = {
            "stage": "development",
            "arm_id": arm_id or view.arm_id,
            "case_id": case.get("case_id") or case.get("request_id"),
            "world_id": case.get("world_id"),
            "expected": case.get("expected"),
            "execution_complete": False,
            "failure": {"type": type(failure).__name__, "message": str(failure),
                        "traceback": failure.traceback},
            "partial_trajectory": failure.partial,
            "information": information,
            "query": query_details,
        }
        raise GenerationFailure(str(failure), partial=row, cause=failure.cause) from failure
    score = score_answer(str(case["expected"]), raw["decoded"])
    ledger.verification()
    progress: dict[str, Any] = {"evaluated": False}
    if task_progress:
        if view.oracle:
            sources = tuple(str(value) for value in case.get("support_record_ids", ()))
        else:
            sources = tuple(str(value) for value in
                            query_details.get("selected_record_ids", ()))
        retained_sources = tuple(value for value in sources
                                 if value in view.path.texts)
        if retained_sources:
            task_id = f"{view.arm_id}:{case.get('case_id')}"
            if task_id not in view.path.tasks:
                view.path.open_task(task_id, (("answer", "generate the declared outcome"),))
            try:
                task = view.path.observe_subgoal_completion(
                    task_id, "answer",
                    observation=("generated answer independently checked against "
                                 "the fixture rule table"),
                    observed_at=int(case["query_time"]),
                    checker="fixture-rule-table",
                    evidence_record_ids=retained_sources)
                progress = {
                    "evaluated": True, "task_id": task_id,
                    "subgoals": len(task.subgoals),
                    "observed_completions": len(task.observed_completions),
                    "independent_checker": "fixture-rule-table",
                }
            except SpineError as error:
                progress = {"evaluated": False, "error": str(error)}
    return {
        "stage": "development",
        "arm_id": arm_id or view.arm_id,
        "case_id": case.get("case_id") or case.get("request_id"),
        "world_id": case.get("world_id"),
        "kind": case.get("kind"),
        "streams": list(case.get("streams", ())),
        "older_rule_retention": bool(case.get("older_rule_retention", False)),
        "expected": case.get("expected"),
        "execution_complete": True,
        "answer": raw["decoded"],
        "score": score,
        "raw": raw,
        "information": information,
        "query": query_details,
        "task_progress": progress,
        "removed_source_text": view.source_prompt_removed,
        "answer_cache_used": False,
        "source_kv_used": False,
    }


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0, "matched": 0, "exact": 0,
                "accuracy": None, "exact_accuracy": None}
    matched = sum(1 for row in rows if row["score"]["matched"])
    exact = sum(1 for row in rows if row["score"]["exact_match"])
    return {"cases": len(rows), "matched": matched, "exact": exact,
            "accuracy": matched / len(rows), "exact_accuracy": exact / len(rows)}


def _learning_evidence(rows: Sequence[Mapping[str, Any]],
                       fit_receipt: Mapping[str, Any]) -> dict[str, Any]:
    by_arm = {arm: [row for row in rows if row["arm_id"] == arm] for arm in ARMS}
    accuracy = {arm: _accuracy(value) for arm, value in by_arm.items()}
    older = [row for row in rows if row.get("older_rule_retention")]
    correction = [row for row in rows if row["case_id"] in ("dev-07", "dev-08")]
    unknown = [row for row in rows if row.get("expected") == "unknown"]
    unknown_false = [
        row for row in unknown
        if "unknown" not in normalize_answer(row["answer"])]
    nonlearning = ("no_update", "reset_P")
    strongest_control = max(
        (accuracy[arm]["accuracy"] or 0.0) for arm in nonlearning)
    memory_arms = ("spine_E_only", "source_free_P_only", "E_plus_P")
    best_memory = max(memory_arms, key=lambda arm: accuracy[arm]["accuracy"] or 0.0)
    best_gain = (accuracy[best_memory]["accuracy"] or 0.0) - strongest_control
    unknown_recall = (len(unknown_false) / len(unknown)) if unknown else 0.0
    older_rows = [row for arm in memory_arms for row in older
                  if row["arm_id"] == arm]
    older_accuracy = _accuracy(older_rows)
    reset_gain = (accuracy["E_plus_P"]["accuracy"] or 0.0) - (
        accuracy["reset_P"]["accuracy"] or 0.0)
    wrong_gain = (accuracy["E_plus_P"]["accuracy"] or 0.0) - (
        accuracy["shuffled_memory"]["accuracy"] or 0.0)
    correction_accuracy = _accuracy(correction)
    passed = (
        best_gain >= 0.10
        and correction_accuracy["accuracy"] is not None
        and unknown_recall <= 0.10
        and reset_gain >= 0.0
        and wrong_gain >= 0.0)
    return {
        "execution_complete": True,
        "optimizer_steps": int(fit_receipt.get("optimizer_steps", 0)),
        "changed_trainable_names": list(
            fit_receipt.get("changed_trainable_names", [])),
        "changed_expert_names": list(
            fit_receipt.get("changed_expert_names", [])),
        "plastic_learning": copy.deepcopy(fit_receipt.get("plastic_learning")),
        "train_fit": copy.deepcopy(fit_receipt.get("train_fit")),
        "transfer": {
            "evaluated": True,
            "per_arm_accuracy": {arm: accuracy[arm] for arm in ARMS},
            "best_memory_arm": best_memory,
            "strongest_nonlearning_control": strongest_control,
            "gain_over_strongest_nonlearning": best_gain,
            "paired_by_independent_world": True,
        },
        "retention": {
            "older_rule": older_accuracy,
            "drop_max_allowed": 0.02,
            "retained": (older_accuracy["accuracy"] is not None
                         and older_accuracy["accuracy"] >= 1.0 - 0.02),
        },
        "useful_learning": {
            "evaluated": True,
            "criterion": ("memory_arm_gain_ge_0.10_over_strongest_nonlearning_and_"
                          "correction_and_unknown_false_recall"),
            "passed": bool(passed),
            "gain": best_gain,
            "unknown_false_recall": unknown_recall,
            "unknown_false_recall_max": 0.10,
            "correction": correction_accuracy,
            "reset_or_wrong_p_reduces_gain": reset_gain >= 0.0 and wrong_gain >= 0.0,
        },
        "learning_success": bool(passed),
        "per_arm": accuracy,
        "correction_effectiveness": correction_accuracy,
        "unknown_false_recall": unknown_recall,
    }


def run_development(protocol: Protocol, dataset: PreparedDataset, backend: Backend,
                    output: Path, profile: WorkProfile, ledger: WorkLedger,
                    fit_state: FitState) -> dict[str, Any]:
    """Nine matched-information arms over twelve development worlds."""
    dataset.open_development(fit_state.receipt)
    development_memory = prepare_development_memory(
        fit_state.path, protocol, dataset)
    views = build_arm_views(
        protocol, backend.adapter, fit_state.path, development_memory,
        trained_plastic=fit_state.trained_plastic,
        pre_plastic=fit_state.pre_plastic)
    development_dir = output / "development"
    development_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    trajectory = JsonlAppender(development_dir / "trajectories.jsonl")
    ledger.set_phase("steady_state")
    try:
        for case in dataset.development_cases():
            for arm_id in ARMS:
                view = views[arm_id]
                try:
                    row = _execute_request(
                        view, backend, dataset, case, profile, ledger,
                        arm_id=arm_id)
                except GenerationFailure as failure:
                    row = dict(failure.partial)
                    row.setdefault("stage", "development")
                    row.setdefault("arm_id", arm_id)
                    row.setdefault("case_id", case["case_id"])
                    row["failure"] = {
                        "type": type(failure).__name__, "message": str(failure),
                        "traceback": failure.traceback,
                    }
                    trajectory.append(row)
                    _write_failure(output, stage="development", error=failure,
                                   ledger=ledger, partial=row)
                    raise
                rows.append(row)
                ledger.arm_case(arm_id)
                trajectory.append(row)
        # Two parity requests, separate from development scoring.
        base_path = fit_state.path
        parity_rows = []
        for request in dataset.parity_requests():
            ledger.charge("parity_requests")
            case = {
                "request_id": request["request_id"], "world_id": None,
                "kind": request["kind"], "question": request["question"],
                "query_time": 10, "expected": "parity_not_scored",
                "support_record_ids": [],
                "procedure_id": request.get("procedure_id"),
                "streams": ["parity"], "older_rule_retention": False,
                "parity": True,
            }
            if request["memory_source"] == "P":
                view = views["source_free_P_only"]
                prompt, information = _prompt_for(view, case, dataset)
                query, details = _prepare_query(
                    view, backend, prompt, query_time=10, profile=profile,
                    ledger=ledger, case=case)
            else:
                prompt = f"Question: {request['question']}\nAnswer:"
                ids, mask = backend.encode_prompt(prompt, profile.context_tokens)
                base_path.experiment.invalidate(reason="no_cache_reuse")
                query = base_path.prepare_query(
                    str(request["question"]), ids, mask,
                    token_available_at=tuple(10 for _ in range(int(ids.shape[1]))),
                    query_time=10, memory_source="E", enabled_branches=(),
                    host_eligible_record_ids=())
                ledger.query_sensing(positions=int(ids.shape[1]))
                ledger.retrieval()
                information = {"source_prompt_removed": True, "parity": True}
                details = {"query_id": query.query_id,
                           "memory_source": query.memory_source,
                           "context_memory_source": query.context.memory_source,
                           "eligible_record_ids": [],
                           "selected_record_ids": [],
                           "semantic_selection": query.semantic_selection,
                           "oracle_selected": False,
                           "host_eligible_record_ids": [],
                           "source_prompt_removed": True,
                           "context_charges": dict(query.context.charges)}
            ledger.charge("generation_requests")
            raw = greedy_generate(backend, query,
                                  max_new_tokens=profile.max_new_tokens,
                                  ledger=ledger)
            parity_rows.append({
                "stage": "development", "parity": True,
                "request_id": request["request_id"], "kind": request["kind"],
                "execution_complete": True, "answer": raw["decoded"],
                "raw": raw, "information": information, "query": details,
                "paired_request_id": request.get("paired_request_id"),
            })
        with (development_dir / "parity.jsonl").open("w", encoding="utf-8") as handle:
            for row in parity_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        report = {
            "format": DEVELOPMENT_FORMAT,
            "status": "complete",
            "execution_complete": True,
            "arms": list(ARMS),
            "development_cases_per_arm": profile.development_cases_per_arm,
            "development_cases_total": len(rows),
            "parity_requests": len(parity_rows),
            "learning_evidence": _learning_evidence(rows, fit_state.receipt),
            "durable_text_references": _durable_text_references(fit_state.path),
            "development_text_references": [
                {"record_id": key, "text_sha256": value.text_sha256}
                for key, value in sorted(development_memory["texts"].items())],
            "derived_caches_required": False,
            "rows_sha256": hashlib.sha256(
                (development_dir / "trajectories.jsonl").read_bytes()).hexdigest(),
            "parity_sha256": hashlib.sha256(
                (development_dir / "parity.jsonl").read_bytes()).hexdigest(),
            "written_at_utc": _utc(),
        }
        publication.write_json(development_dir, "development_report.json", report)
        return report
    finally:
        trajectory.close()
# ---------------------------------------------------------------------------
# Fresh-process restart and strict durable restore


def _verify_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    path = root / str(artifact["path"])
    _require(path.is_file(), "artifact", f"missing {artifact['path']}")
    _require(path.stat().st_size == int(artifact["bytes"]), "artifact",
             f"{artifact['path']} byte count differs")
    _require(sha256_file(path) == artifact["sha256"], "artifact",
             f"{artifact['path']} hash differs")


def load_runner_state(output: Path, receipt: Mapping[str, Any],
                      protocol: Protocol, dataset: PreparedDataset,
                      backend: Backend) -> dict[str, Any]:
    artifact = receipt.get("runner_state")
    _require(isinstance(artifact, Mapping), "runner_state", "artifact is missing")
    _verify_artifact(output, artifact)
    payload = publication.load_torch(output, str(artifact["path"]))
    required = {
        "format", "protocol_id", "protocol_sha256", "prepared_manifest_sha256",
        "run_contract", "profile", "sealed_source", "producer",
        "pre_fit_plastic", "plastic_digest", "pre_fit_plastic_digest",
        "fit_checkpoint", "durable_text_references", "derived_caches_required",
        "arm_policy", "query_schedule", "written_at_utc",
    }
    _require(isinstance(payload, Mapping) and set(payload) == required,
             "runner_state", "keys/format differ")
    _require(payload["format"] == RUNNER_STATE_FORMAT, "runner_state",
             "format differs")
    _require(payload["protocol_id"] == protocol.protocol_id, "runner_state",
             "protocol id differs")
    _require(payload["protocol_sha256"] == protocol.sha256, "runner_state",
             "protocol hash differs")
    _require(payload["prepared_manifest_sha256"] == dataset.manifest_sha256,
             "runner_state", "manifest hash differs")
    _require(payload["run_contract"] == receipt["run_contract"],
             "runner_state", "run contract differs from the fit receipt")
    _require(payload["derived_caches_required"] is False, "runner_state",
             "runner state requires derived caches")
    _require(payload["producer"]["producer_revision"] == TEXT_PRODUCER_REVISION
             and payload["producer"]["representation_version"] == TEXT_REPRESENTATION_VERSION
             and payload["producer"]["codec_version"] == TEXT_CODEC_VERSION,
             "runner_state", "producer/codec versions differ")
    profile_row = payload["profile"]
    _require(profile_row["label"] == backend.label, "runner_state",
             "backend label differs")
    _require(bool(profile_row["native_admissible"]) == bool(backend.native_admissible),
             "runner_state", "native admission flag differs")
    sealed = payload["sealed_source"]
    _require(sealed["model_identity"] == backend.adapter.durable_identity(),
             "runner_state", "model identity differs")
    if backend.label == CPU_FIXTURE_LABEL:
        _require(sealed["identity"] == CPU_FIXTURE_LABEL
                 and sealed["official_source"] is False
                 and sealed["native_admissible"] is False
                 and sealed.get("fixture_seed") == backend.fixture_seed,
                 "runner_state", "cpu fixture source reference differs")
    else:
        source = protocol.source_identity
        for key in ("config_sha256", "weight_sha256", "revision", "local_path"):
            _require(sealed[key] == source[key], "runner_state",
                     f"sealed source {key} differs")
        _require(sealed["identity"] == "official_fp8", "runner_state",
                 "sealed source identity differs")
    _require(_plastic_digest(payload["pre_fit_plastic"])
             == payload["pre_fit_plastic_digest"], "runner_state",
             "pre-fit plastic digest differs")
    fit_checkpoint = payload["fit_checkpoint"]
    _require(fit_checkpoint["relative"] == receipt["fit_checkpoint"]["relative"],
             "runner_state", "fit checkpoint relative path differs")
    for row in fit_checkpoint["artifacts"]:
        _verify_artifact(output, row)
    for row in payload["durable_text_references"]:
        _require(bool(row.get("canonical_text")) or row.get("durable_ref") is not None,
                 "runner_state", "a text record has no canonical or durable source")
    return payload


def run_restart(protocol: Protocol, dataset: PreparedDataset, backend: Backend,
                output: Path, profile: WorkProfile, ledger: WorkLedger, *,
                restart_process: str = "fresh_process") -> dict[str, Any]:
    """Restore one checkpoint in a fresh process and evaluate source-free P."""
    started = time.monotonic()
    fit_receipt = _read_json(output / "fit/fit_receipt.json", "fit_receipt")
    dataset.open_development(fit_receipt)
    state = load_runner_state(output, fit_receipt, protocol, dataset, backend)
    ledger.set_phase("amortized")
    path = _new_path(protocol, backend.adapter, response_lr=RESPONSE_LR,
                     producer=VersionedTextProducer())
    restored = path.restore(output, "fit/final/state",
                            run_contract=fit_receipt["run_contract"])
    _require(restored["derived_caches_restored"] is False, "restore",
             "derived caches were restored")
    ledger.reload()
    restored_digest = _plastic_digest(path.experiment.controller.checkpoint_payload())
    _require(restored_digest == state["plastic_digest"], "restore",
             "retained plastic digest differs after restore")
    _require(len(path.texts) == len(state["durable_text_references"]),
             "restore", "text owner count differs")
    plastic_before = _plastic_digest(path.experiment.controller.checkpoint_payload())
    inventory_before = path.state_inventory()
    path.clear_for_p_only()
    plastic_after = _plastic_digest(path.experiment.controller.checkpoint_payload())
    _require(plastic_after == plastic_before, "restore",
             "source removal changed retained plastic state")
    _require(path.experiment.evidence.pending == 0 and not path.texts
             and not path.procedures and not path.tasks and not path.explanations,
             "restore", "source evidence was not fully removed")
    request = dataset.parity_requests(restart=True)[0]
    question = str(request["question"])
    ids, mask = backend.encode_prompt(
        f"Question: {question}\nAnswer:", profile.context_tokens)
    path.experiment.invalidate(reason="no_cache_reuse")
    query = path.prepare_query(
        question, ids, mask,
        token_available_at=tuple(10 for _ in range(int(ids.shape[1]))),
        query_time=10, memory_source="P", plastic_expert_id=PLASTIC_EXPERT,
        enabled_branches=("memory",))
    ledger.query_sensing(positions=int(ids.shape[1]))
    ledger.retrieval()
    ledger.charge("restart_parity_requests")
    ledger.charge("generation_requests")
    raw = greedy_generate(backend, query,
                          max_new_tokens=profile.max_new_tokens, ledger=ledger)
    paired = None
    parity_path = output / "development/parity.jsonl"
    if parity_path.is_file():
        for row in _read_jsonl(parity_path, "parity"):
            if row.get("request_id") == request.get("paired_request_id"):
                paired = row
                break
    _require(paired is not None, "restart_parity",
             "the paired pre-restart parity row is missing")
    paired_tokens = paired["raw"]["token_ids"]
    parity_match = list(raw["token_ids"]) == list(paired_tokens)
    report = {
        "format": RESTART_FORMAT,
        "status": "complete",
        "execution_complete": True,
        "restart_process": restart_process,
        "request_id": request["request_id"],
        "paired_request_id": request.get("paired_request_id"),
        "parity_match_exact_tokens": parity_match,
        "restart_parity_requests": 1,
        "source_evidence_removed": True,
        "derived_caches_restored": False,
        "restored": {
            "experiment": restored["experiment"],
            "text_records": restored["text_records"],
            "procedures": restored["procedures"],
            "task_workspaces": restored["task_workspaces"],
        },
        "inventory_before_source_removal": inventory_before,
        "inventory_after_source_removal": path.state_inventory(),
        "plastic_digest_before": plastic_before,
        "plastic_digest_after": plastic_after,
        "answer": raw["decoded"],
        "paired_tokens": paired_tokens,
        "raw": raw,
        "work": ledger.report(),
        "written_at_utc": _utc(),
    }
    ledger.wall_seconds = time.monotonic() - started
    report["work"] = ledger.report()
    restart_dir = output / "restart"
    restart_dir.mkdir(parents=True, exist_ok=True)
    publication.write_json(restart_dir, "restart_report.json", report)
    with (restart_dir / "trajectory.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "request_id": request["request_id"],
            "answer": raw["decoded"], "raw": raw,
            "parity_match_exact_tokens": parity_match,
        }, sort_keys=True) + "\n")
    return report


def _restart_command(protocol: Protocol, dataset: PreparedDataset, output: Path,
                     *, backend_kind: str, device: str,
                     authorization_path: str | Path | None,
                     backend_evidence_path: str | Path | None,
                     profile: WorkProfile) -> list[str]:
    root = protocol.path.parent.parent
    script = root / "scripts/run_spark_text_memory_v1.py"
    command = [
        sys.executable, str(script), "--stage", "restart",
        "--backend", backend_kind, "--device", device,
        "--output", str(output), "--config", str(protocol.path),
        "--prepared", str(dataset.root),
    ]
    if backend_kind == CPU_FIXTURE_LABEL:
        command.extend(["--fixture-context-tokens", str(profile.context_tokens),
                        "--fixture-max-new-tokens", str(profile.max_new_tokens)])
    if authorization_path is not None:
        command.extend(["--authorization", str(authorization_path)])
    if backend_evidence_path is not None:
        command.extend(["--backend-evidence", str(backend_evidence_path)])
    return command


def _spawn_restart(protocol: Protocol, dataset: PreparedDataset, output: Path,
                   *, backend_kind: str, device: str,
                   authorization_path: str | Path | None,
                   backend_evidence_path: str | Path | None,
                   profile: WorkProfile) -> dict[str, Any]:
    command = _restart_command(
        protocol, dataset, output, backend_kind=backend_kind, device=device,
        authorization_path=authorization_path,
        backend_evidence_path=backend_evidence_path, profile=profile)
    restart_dir = output / "restart"
    restart_dir.mkdir(parents=True, exist_ok=True)
    (restart_dir / "child_command.json").write_text(
        json.dumps({"command": command, "written_at_utc": _utc()},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.monotonic()
    result = subprocess.run(command, cwd=protocol.path.parent.parent,
                            text=True, capture_output=True,
                            timeout=int(protocol.config["work_and_cost"][
                                "native_schedule"]["wall_seconds_max"]))
    (restart_dir / "child.stdout").write_text(result.stdout, encoding="utf-8")
    (restart_dir / "child.stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RunnerError(
            f"fresh-process restart failed with exit code {result.returncode}: "
            f"{result.stderr.strip()[-2000:]}")
    report_path = restart_dir / "restart_report.json"
    if not report_path.is_file():
        raise RunnerError("fresh-process restart wrote no restart report")
    report = _read_json(report_path, "restart")
    report["subprocess"] = {
        "returncode": result.returncode,
        "wall_seconds": time.monotonic() - started,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
    }
    report["restart_process"] = "fresh_process"
    return report


# ---------------------------------------------------------------------------
# Protocol execution and explicit mismatches


def _protocol_mismatches(protocol: Protocol, profile: WorkProfile,
                         ledger: WorkLedger) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    if profile.label == NATIVE_LABEL:
        schedule = protocol.schedule
        corrected = profile.corrected_caps()
        for key in ("query_sensing_forwards_max", "forward_calls_max",
                    "processed_text_positions_max"):
            corrected_key = {
                "query_sensing_forwards_max": "query_sensing_forwards",
                "forward_calls_max": "forward_calls",
                "processed_text_positions_max": "processed_text_positions",
            }[key]
            if int(schedule[key]) != int(corrected[corrected_key]):
                mismatches.append({
                    "field": f"work_and_cost.native_schedule.{key}",
                    "protocol": int(schedule[key]),
                    "corrected": int(corrected[corrected_key]),
                    "reason": ("P query features depend on native hidden state; "
                               "the conservative schedule senses once per "
                               "autoregressive step and counts both passes"),
                })
    return mismatches


def _checkpoint_summary(output: Path, fit_receipt: Mapping[str, Any],
                        ledger: WorkLedger) -> dict[str, Any]:
    total = publication.total_bytes(output)
    return {
        "writes": int(fit_receipt["checkpoint_writes"]),
        "retained": list(fit_receipt["retained_checkpoints"]),
        "evicted": list(fit_receipt["evicted_checkpoints"]),
        "retained_count": int(ledger.counters["retained_checkpoint_artifacts"]),
        "artifact_bytes": total,
        "artifact_cap": int(ledger.caps["artifact_bytes"]),
        "extension_only_with_sealed_source": True,
        "derived_caches_saved": False,
    }


def run_protocol(*, protocol_path: str | Path, prepared_root: str | Path,
                 output_root: str | Path, backend_kind: str = NATIVE_LABEL,
                 device: str = "mps",
                 authorization_path: str | Path | None = None,
                 backend_evidence_path: str | Path | None = None,
                 profile: WorkProfile | None = None,
                 fixture_context_tokens: int | None = None,
                 fixture_max_new_tokens: int | None = None,
                 response_lr: float = RESPONSE_LR,
                 restart_in_process: bool = False,
                 implementation_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run fit, development and one restart with real measured work."""
    started = time.monotonic()
    protocol = load_protocol(protocol_path)
    expected_manifest = protocol.config["dataset"].get("manifest_sha256")
    dataset = PreparedDataset.load(
        prepared_root, expected_sha256=expected_manifest)
    output = Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise Refused("output", "run output must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    if profile is None:
        profile = WorkProfile.from_protocol(
            protocol.config, backend_kind=backend_kind,
            fixture_context_tokens=(fixture_context_tokens
                                    if backend_kind == CPU_FIXTURE_LABEL else None),
            fixture_max_new_tokens=(fixture_max_new_tokens
                                    if backend_kind == CPU_FIXTURE_LABEL else None))
    if profile.label != (CPU_FIXTURE_LABEL if backend_kind == CPU_FIXTURE_LABEL
                         else NATIVE_LABEL):
        raise Refused("profile", "work profile and backend kind differ")
    ledger = WorkLedger(profile)
    ledger.set_phase("cold_start")
    stage = "backend"
    try:
        backend = build_backend(
            backend_kind, protocol, dataset, profile, device=device,
            output_root=output, authorization_path=authorization_path,
            backend_evidence_path=backend_evidence_path,
            allow_existing_output=False)
        ledger.load_backend()
        stage = "fit"
        fit_state = run_fit(protocol, dataset, backend, output, profile, ledger,
                            response_lr=response_lr)
        stage = "development"
        development = run_development(
            protocol, dataset, backend, output, profile, ledger, fit_state)
        stage = "restart"
        ledger.charge("restarts")
        if restart_in_process:
            restart_backend = build_backend(
                backend_kind, protocol, dataset, profile, device=device,
                output_root=output, authorization_path=authorization_path,
                backend_evidence_path=backend_evidence_path,
                allow_existing_output=True)
            ledger.load_backend()
            restart = run_restart(
                protocol, dataset, restart_backend, output, profile, ledger,
                restart_process="in_process_test_fixture")
        else:
            restart = _spawn_restart(
                protocol, dataset, output, backend_kind=backend_kind,
                device=device, authorization_path=authorization_path,
                backend_evidence_path=backend_evidence_path, profile=profile)
            ledger.merge(restart["work"])
        if not restart.get("execution_complete"):
            raise RunnerError("restart did not complete")
        stage = "reconcile"
        artifact_bytes = publication.total_bytes(output)
        if artifact_bytes > int(ledger.caps["artifact_bytes"]):
            raise CapExceeded("artifact_bytes", artifact_bytes,
                              int(ledger.caps["artifact_bytes"]))
        ledger.counters["artifact_bytes"] = artifact_bytes
        ledger.wall_seconds = time.monotonic() - started
        work = ledger.reconcile()
        mismatches = _protocol_mismatches(protocol, profile, ledger)
        checkpoints = _checkpoint_summary(output, fit_state.receipt, ledger)
        report = {
            "format": REPORT_FORMAT,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "prepared_manifest_sha256": dataset.manifest_sha256,
            "status": ("complete" if not mismatches
                       else "complete_with_protocol_mismatch"),
            "execution_complete": True,
            "admitted": bool(backend.native_admissible),
            "backend": {
                "label": backend.label,
                "native_admissible": bool(backend.native_admissible),
                "device": backend.device,
                "official_fp8_only": backend.label == NATIVE_LABEL,
                "precision_substitution": False,
                "evidence_class": ("cpu_fixture_not_protocol_evidence"
                                   if backend.label == CPU_FIXTURE_LABEL
                                   else "native_official_fp8"),
            },
            "work_profile": {
                "label": profile.label,
                "context_tokens": profile.context_tokens,
                "max_new_tokens": profile.max_new_tokens,
                "native_admissible": profile.native_admissible,
            },
            "native_admission": ({
                "attempted": True,
                "admitted": True,
                "authorization": str(authorization_path),
                "backend_evidence": str(backend_evidence_path),
            } if backend.label == NATIVE_LABEL else {
                "attempted": False,
                "admitted": False,
                "reason": "cpu_fixture selected; never native-admissible",
            }),
            "fit": {
                "receipt": "fit/fit_receipt.json",
                "status": fit_state.receipt["status"],
                "execution_complete": fit_state.receipt["execution_complete"],
                "optimizer_steps": fit_state.receipt["optimizer_steps"],
                "changed_trainable_names": fit_state.receipt[
                    "changed_trainable_names"],
                "train_fit": fit_state.receipt["train_fit"],
                "plastic_learning": fit_state.receipt["plastic_learning"],
                "checkpoint_writes": fit_state.receipt["checkpoint_writes"],
                "retained_checkpoints": fit_state.receipt["retained_checkpoints"],
            },
            "development": {
                "report": "development/development_report.json",
                "cases_total": development["development_cases_total"],
                "cases_per_arm": development["development_cases_per_arm"],
                "parity_requests": development["parity_requests"],
                "learning_evidence": development["learning_evidence"],
            },
            "restart": {
                "process": restart.get("restart_process"),
                "request_id": restart["request_id"],
                "parity_match_exact_tokens": restart["parity_match_exact_tokens"],
                "source_evidence_removed": restart["source_evidence_removed"],
                "derived_caches_restored": restart["derived_caches_restored"],
                "text_records": restart["restored"]["text_records"],
                "inventory_after_source_removal": restart[
                    "inventory_after_source_removal"],
            },
            "work": work,
            "checkpoints": checkpoints,
            "protocol_mismatches": mismatches,
            "cost": {
                "cold_start": work["cost_phases"].get("cold_start", {}),
                "steady_state": work["cost_phases"].get("steady_state", {}),
                "amortized_over_repeated_tasks": work["cost_phases"].get(
                    "amortized", {}),
                "cold_and_steady_are_not_added_into_amortized": True,
            },
            "non_claims": list(protocol.config["non_claims"]),
            "wall_seconds": time.monotonic() - started,
            "written_at_utc": _utc(),
        }
        publication.write_json(output, "run_report.json", report)
        return report
    except BaseException as error:
        _write_failure(output, stage=stage, error=error, ledger=ledger)
        raise


__all__ = [
    "ARMS", "ORACLE_ARM", "CPU_FIXTURE_LABEL", "NATIVE_LABEL", "STAGES",
    "PROTOCOL_FORMAT", "PROTOCOL_ID", "WorkProfile", "WorkLedger",
    "Protocol", "PreparedDataset", "Backend", "FitState", "ArmView",
    "RunnerError", "Refused", "CapExceeded", "GenerationFailure",
    "load_protocol", "PreparedDataset", "build_cpu_fixture_backend",
    "build_native_backend", "build_backend", "verify_native_authorization",
    "prepare_development_memory", "build_arm_views", "greedy_generate",
    "normalize_answer", "score_answer", "run_fit", "run_development",
    "run_restart", "load_runner_state", "run_protocol",
]
def load_fit_state(protocol: Protocol, dataset: PreparedDataset, backend: Backend,
                   output: Path, ledger: WorkLedger) -> FitState:
    """Restore one fit receipt and checkpoint for a standalone stage."""
    receipt = _read_json(output / "fit/fit_receipt.json", "fit_receipt")
    dataset.open_development(receipt)
    state = load_runner_state(output, receipt, protocol, dataset, backend)
    path = _new_path(protocol, backend.adapter, response_lr=RESPONSE_LR,
                     producer=VersionedTextProducer())
    path.restore(output, "fit/final/state",
                 run_contract=receipt["run_contract"])
    ledger.reload()
    trained = copy.deepcopy(path.experiment.controller.checkpoint_payload())
    _require(_plastic_digest(trained) == state["plastic_digest"], "restore",
             "retained plastic digest differs after restore")
    return FitState(path=path, run_contract=receipt["run_contract"],
                    receipt=receipt, pre_plastic=state["pre_fit_plastic"],
                    trained_plastic=trained,
                    checkpoints=[], ledger=ledger)


__all__.append("load_fit_state")
