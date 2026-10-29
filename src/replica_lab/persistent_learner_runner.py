"""Versioned P1 persistent-learner fit, controls and restart runner.

The native path admits only the sealed official FP8 source and its exact backend
receipt.  The contextual CPU path is an engineering fixture, never a native
capability result.  Fit uses direct response CE through selected P and a
separately owned port-output optimizer; no local-MSE or broad response-spine
update is hidden in this protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import torch

from . import artifact_publication as publication
from .architecture.spark2_5 import (
    OFFICIAL_NATIVE_FP8, evaluate_admission, load_spark2_5_config,
    load_tokenizer_identity, official_native_fp8_precision, resolve_thinking_mode,
)
from .architecture.spark_adapter import Spark2_5Adapter
from .architecture.loading import exclusive_mps_worker
from .architecture.spine import SPINE_SCHEMA
from .architecture.spine_memory import EvidenceRecord, SpineMemoryExperiment
from .architecture.spine_text import (
    TEXT_CODEC_VERSION, TEXT_REPRESENTATION_VERSION,
    SparkTextMemoryPath, VersionedTextProducer,
)
from .optimizers import NativeFusedAdamW
from .p1_response_learning import (
    audit_p_only_claim, capture_p_state, direct_response_ce_step, evaluate_response_ce,
    prepare_arm_query, restore_p_state,
)
from .spark_text_memory_runner import Backend, greedy_generate, score_answer
from .tiered_readout import ReadoutBudget, ScopedMemoryReadout, merge_readout_metrics, read_scoped_memory, readout_cache_state, resume_scoped_readout
from .tiered_memory import (
    EvidenceRecordDraft, MemoryScope, Outcome, TierCeilings, TieredMemoryStore,
    build_retrieval_key, detect_file_type,
)
from .testing import DeterministicTinyTokenizer

PROTOCOL_FORMAT = "replica.spark_persistent_learner.protocol.v2"
PROTOCOL_ID = "spark_persistent_learner_v2"
FIXTURE_FORMAT = "replica.spark_persistent_learner.fixture.v2"
FIT_RECEIPT_FORMAT = "replica.spark_persistent_learner.fit_receipt.v2"
DYNAMIC_FORMAT = "replica.spark_persistent_learner.dynamic_state.v3"
DEVELOPMENT_FORMAT = "replica.spark_persistent_learner.development.v2"
RESTART_FORMAT = "replica.spark_persistent_learner.restart.v2"
REPORT_FORMAT = "replica.spark_persistent_learner.run.v2"
CPU_LABEL = "cpu_contextual_fixture"
NATIVE_LABEL = "native"
ARMS = (
    "full_eligible_context", "selective_exact_and_procedure", "spine_E_only",
    "reader_trained_E_only", "source_free_P_only", "E_plus_P", "no_update", "reset_P",
    "shuffled_memory", "inapplicable_procedure", "matched_replay",
)


class RunnerError(RuntimeError):
    """Base runner failure; partial artifacts are preserved by the caller."""


class Refused(RunnerError):
    def __init__(self, check: str, detail: str) -> None:
        super().__init__(f"{check}: {detail}")
        self.check = check
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        return {"check": self.check, "detail": self.detail}


class CapExceeded(RunnerError):
    pass


def _require(condition: bool, check: str, detail: str) -> None:
    if not condition:
        raise Refused(check, detail)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return publication.sha256_path(Path(path))


def _read_json(path: Path, label: str = "json") -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(label, f"cannot read {path}: {error}") from error
    _require(isinstance(value, dict), label, f"{path} must contain an object")
    return value


def _read_jsonl(path: Path, label: str = "jsonl") -> list[dict[str, Any]]:
    rows = []
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                _require(isinstance(value, dict), label,
                         f"{path}:{line_number} must contain an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(label, f"cannot read {path}: {error}") from error
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return publication.tensor_digest(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        return repr(value)
    return value


def _write_json(root: Path, relative: str, value: Mapping[str, Any], *,
                meter: publication.LiveByteMeter | None = None) -> None:
    publication.write_json(root, relative, _json_safe(value), meter=meter)


def _write_jsonl(root: Path, relative: str, rows: Iterable[Mapping[str, Any]], *,
                 meter: publication.LiveByteMeter | None = None) -> None:
    publication.write_jsonl(root, relative, (_json_safe(row) for row in rows), meter=meter)


@dataclass(frozen=True)
class Protocol:
    path: Path
    sha256: str
    config: Mapping[str, Any]

    @property
    def protocol_id(self) -> str:
        return str(self.config["protocol_id"])

    @property
    def root(self) -> Path:
        return self.path.parent.parent

    @property
    def schedule(self) -> Mapping[str, Any]:
        return self.config["work_and_cost"]["native_schedule"]

    @property
    def source_identity(self) -> Mapping[str, Any]:
        return self.config["source_identities"]["official_fp8"]

    def implementation_hashes(self) -> dict[str, str]:
        result = {}
        for relative in self.config["launch"]["implementation_files"]:
            path = self.root / relative
            _require(path.is_file(), "implementation", f"missing {relative}")
            result[relative] = _sha256(path)
        return result

    @classmethod
    def load(cls, path: str | Path) -> "Protocol":
        path = Path(path).resolve()
        config = _read_json(path, "protocol")
        _require(config.get("format") == PROTOCOL_FORMAT, "protocol", "format differs")
        _require(config.get("protocol_id") == PROTOCOL_ID, "protocol", "id differs")
        return cls(path=path, sha256=_sha256(path), config=config)


@dataclass
class PreparedDataset:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    protocol_sha256: str
    _rows_cache: dict[str, list[dict[str, Any]]]
    _development_open: bool = False

    @classmethod
    def load(cls, root: str | Path, *, expected_sha256: str | None,
             protocol_sha256: str) -> "PreparedDataset":
        root = Path(root).resolve()
        manifest_path = root / "manifest.json"
        manifest = _read_json(manifest_path, "prepared")
        _require(manifest.get("format") == FIXTURE_FORMAT, "prepared", "format differs")
        _require(manifest.get("protocol_id") == PROTOCOL_ID, "prepared", "protocol id differs")
        digest = _sha256(manifest_path)
        if expected_sha256 is not None:
            _require(digest == expected_sha256, "prepared", "manifest hash differs")
        files = manifest.get("files")
        _require(isinstance(files, Mapping), "prepared", "manifest files must be an object")
        required = {"observations.jsonl", "fit_events.jsonl", "development.jsonl",
                    "development_labels.jsonl", "parity_requests.jsonl", "procedures.json",
                    "tasks.json", "worlds.json"}
        _require(required <= set(files), "prepared", "required fixture stream is missing")
        for name, metadata in files.items():
            path = root / name
            _require(path.is_file(), "prepared", f"missing fixture file {name}")
            _require(isinstance(metadata, Mapping), "prepared", f"bad file receipt {name}")
            _require(path.stat().st_size == int(metadata["bytes"]), "prepared",
                     f"byte count differs for {name}")
            _require(_sha256(path) == metadata["sha256"], "prepared",
                     f"content hash differs for {name}")
        counts = manifest.get("counts", {})
        for key, expected in (("fit_events", 4), ("development_cases", 12),
                              ("development_labels", 12), ("arms", len(ARMS)),
                              ("parity_requests", 2), ("restart_parity_requests", 1)):
            _require(int(counts.get(key, -1)) == expected, "prepared",
                     f"count {key} differs")
        sealed = manifest.get("sealed_final", {})
        _require(sealed.get("materialized") is False and sealed.get("opened") is False
                 and sealed.get("path") is None, "access", "sealed final must remain closed")
        return cls(root=root, manifest=manifest, manifest_sha256=digest,
                   protocol_sha256=protocol_sha256, _rows_cache={})

    def _rows(self, name: str) -> list[dict[str, Any]]:
        if name not in self._rows_cache:
            self._rows_cache[name] = _read_jsonl(self.root / name, "fixture_stream")
        return copy.deepcopy(self._rows_cache[name])

    def observations(self) -> list[dict[str, Any]]:
        rows = self._rows("observations.jsonl")
        ids = [str(row["record_id"]) for row in rows]
        _require(len(ids) == len(set(ids)), "prepared", "duplicate observation record ID")
        epochs = [int(row["source_epoch"]) for row in rows]
        _require(epochs == sorted(epochs), "prepared", "observations are not chronological")
        return rows

    def fit_events(self) -> list[dict[str, Any]]:
        rows = self._rows("fit_events.jsonl")
        _require(len(rows) == 4, "prepared", "four fit events required")
        _require([int(row["sequence"]) for row in rows] == [1, 2, 3, 4],
                 "prepared", "fit chronology differs")
        return rows

    def procedures(self) -> list[dict[str, Any]]:
        value = _read_json(self.root / "procedures.json", "procedures")
        rows = value.get("procedures")
        _require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
                 "procedures", "procedures must be a list of objects")
        return copy.deepcopy(rows)

    def tasks(self) -> list[dict[str, Any]]:
        value = _read_json(self.root / "tasks.json", "tasks")
        rows = value.get("tasks")
        _require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
                 "tasks", "tasks must be a list of objects")
        return copy.deepcopy(rows)

    def worlds(self) -> list[dict[str, Any]]:
        value = _read_json(self.root / "worlds.json", "worlds")
        rows = value.get("worlds")
        _require(isinstance(rows, list), "worlds", "worlds must be a list")
        return copy.deepcopy(rows)

    def open_development(self, receipt: Mapping[str, Any]) -> None:
        _require(receipt.get("format") == FIT_RECEIPT_FORMAT, "fit_receipt", "format differs")
        _require(receipt.get("protocol_id") == PROTOCOL_ID, "fit_receipt", "protocol id differs")
        _require(receipt.get("protocol_sha256") == self.protocol_sha256,
                 "fit_receipt", "protocol hash differs")
        _require(receipt.get("prepared_manifest_sha256") == self.manifest_sha256,
                 "fit_receipt", "manifest hash differs")
        _require(receipt.get("status") == "fit_executed"
                 and receipt.get("execution_complete") is True,
                 "fit_receipt", "fit is not complete")
        self._development_open = True

    def development_cases(self) -> list[dict[str, Any]]:
        _require(self._development_open, "development_labels_closed",
                 "development labels require a completed fit receipt")
        cases = self._rows("development.jsonl")
        labels = {str(row["case_id"]): str(row["expected"])
                  for row in self._rows("development_labels.jsonl")}
        _require(len(cases) == 12 and len(labels) == 12, "development", "twelve cases/labels required")
        result = []
        for case in cases:
            _require("expected" not in case, "access", "development inputs contain labels")
            _require(str(case["case_id"]) in labels, "development", "label is missing")
            case["expected"] = labels[str(case["case_id"])]
            result.append(case)
        return result

    def parity_requests(self, *, restart: bool = False) -> list[dict[str, Any]]:
        _require(self._development_open, "development_labels_closed",
                 "parity requires a completed fit receipt")
        rows = self._rows("parity_requests.jsonl")
        selected = [row for row in rows
                    if str(row["request_id"]).startswith("restart-") == restart]
        _require(len(selected) == (1 if restart else 2), "parity", "parity count differs")
        return selected


class WorkLedger:
    """Measured counters with conservative native upper bounds."""

    NAMES = (
        "fit_events", "development_cases", "parity_requests", "restart_parity_requests",
        "restarts", "model_loads", "tokenizer_loads", "training_forwards",
        "matched_nonlearning_fit_events", "matched_nonlearning_forward_calls",
        "matched_nonlearning_backward_calls", "matched_nonlearning_processed_positions",
        "direct_response_ce_objective_forwards", "direct_response_ce_backward_calls",
        "direct_response_ce_feature_vectors", "plastic_forward_calls", "plastic_backward_calls",
        "query_sensing_forwards", "native_query_passes", "backward_calls",
        "backward_processed_positions", "plastic_feature_vectors",
        "plastic_backward_feature_vectors", "plastic_fit_evaluation_forwards",
        "optimizer_steps", "response_spine_optimizer_steps", "response_port_optimizer_steps",
        "selected_P_optimizer_steps", "generation_requests", "autoregressive_forwards",
        "forward_calls", "generated_tokens", "processed_text_positions",
        "initialization_parity_forwards", "initialization_parity_processed_positions",
        "influence_check_query_sensing", "influence_check_forwards",
        "influence_check_processed_positions", "replay_events", "replay_forward_calls",
        "replay_processed_positions", "retrieval_calls", "verification_calls",
        "checkpoint_writes", "retained_checkpoint_artifacts", "reloads", "cache_hits",
        "cache_misses", "teacher_passes", "consolidation_events", "artifact_bytes", "native_retries",
        "readout_cache_bytes", "readout_result_bytes", "readout_selection_calls", "readout_file_reads", "readout_io_bytes", "readout_ns", "cold_recall_tickets",
        "cold_recall_resumes", "revalidation_failures", "payload_read_calls",
        "payload_file_reads", "payload_io_bytes", "revalidations",
    )

    def __init__(self, protocol: Protocol, *, context_tokens: int, max_new_tokens: int) -> None:
        self.protocol = protocol
        self.context_tokens = int(context_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.counters = {name: 0 for name in self.NAMES}
        self.caps = {}
        for name in self.NAMES:
            if name in protocol.schedule:
                self.caps[name] = int(protocol.schedule[name])
            elif f"{name}_max" in protocol.schedule:
                self.caps[name] = int(protocol.schedule[f"{name}_max"])
        self.per_arm = {arm: 0 for arm in ARMS}

    def charge(self, name: str, amount: int = 1) -> None:
        _require(name in self.counters, "counter", f"unknown counter {name}")
        _require(type(amount) is int and amount >= 0, "counter", "amount must be nonnegative")
        value = self.counters[name] + amount
        if name in self.caps and value > self.caps[name]:
            raise CapExceeded(f"{name}={value} exceeds cap {self.caps[name]}")
        self.counters[name] = value

    def set_measured(self, name: str, value: int) -> None:
        _require(name in self.counters, "counter", f"unknown counter {name}")
        if name in self.caps and int(value) > self.caps[name]:
            raise CapExceeded(f"{name}={value} exceeds cap {self.caps[name]}")
        self.counters[name] = int(value)

    def query_sensing(self, positions: int, *, cache_hits: int = 0,
                      cache_misses: int = 0) -> None:
        self.charge("query_sensing_forwards")
        self.charge("native_query_passes")
        self.charge("forward_calls")
        self.charge("processed_text_positions", int(positions))
        self.charge("cache_hits", int(cache_hits))
        self.charge("cache_misses", int(cache_misses))

    def retrieval(self) -> None:
        self.charge("retrieval_calls")

    def objective(self, positions: int, *, nonlearning: bool = False) -> None:
        self.charge("training_forwards")
        self.charge("direct_response_ce_objective_forwards")
        self.charge("forward_calls")
        self.charge("processed_text_positions", int(positions))
        self.charge("backward_calls")
        self.charge("direct_response_ce_backward_calls")
        self.charge("backward_processed_positions", int(positions))
        self.charge("direct_response_ce_feature_vectors", int(positions))
        if nonlearning:
            self.charge("matched_nonlearning_fit_events")
            self.charge("matched_nonlearning_forward_calls")
            self.charge("matched_nonlearning_backward_calls")
            self.charge("matched_nonlearning_processed_positions", int(positions))

    def replay(self, positions: int) -> None:
        self.charge("replay_events")
        self.charge("replay_forward_calls")
        self.charge("replay_processed_positions", int(positions))
        self.charge("forward_calls")
        self.charge("processed_text_positions", int(positions))

    def parity_forwards(self, positions: int) -> None:
        self.charge("initialization_parity_forwards", 2)
        self.charge("initialization_parity_processed_positions", 2 * int(positions))
        self.charge("forward_calls", 2)
        self.charge("processed_text_positions", 2 * int(positions))

    def influence_forwards(self, positions: int) -> None:
        self.charge("influence_check_query_sensing")
        self.charge("influence_check_forwards", 2)
        self.charge("influence_check_processed_positions", 2 * int(positions))
        self.charge("forward_calls", 2)
        self.charge("processed_text_positions", 2 * int(positions))

    def development_case(self, arm: str) -> None:
        _require(arm in self.per_arm, "controls", f"unknown arm {arm}")
        self.per_arm[arm] += 1
        self.charge("development_cases")

    def merge_restart(self, counters: Mapping[str, Any]) -> None:
        for name in (
            "restarts", "restart_parity_requests", "model_loads", "tokenizer_loads", "query_sensing_forwards",
            "native_query_passes", "forward_calls", "processed_text_positions",
            "generation_requests", "autoregressive_forwards", "generated_tokens",
            "retrieval_calls", "cache_misses", "verification_calls", "reloads",
        ):
            self.charge(name, int(counters.get(name, 0)))

    def finish(self) -> None:
        _require(self.counters["fit_events"] == 4, "completion", "fit event count differs")
        _require(all(value == 12 for value in self.per_arm.values()),
                 "completion", f"development arm counts differ: {self.per_arm}")
        _require(self.counters["parity_requests"] == 2
                 and self.counters["restart_parity_requests"] == 1
                 and self.counters["restarts"] == 1, "completion", "parity/restart count differs")
        _require(self.counters["revalidation_failures"] == 0,
                 "completion", "a scoped readout failed post-I/O revalidation")
        _require(self.counters["response_spine_optimizer_steps"] == 0
                 and self.counters["plastic_forward_calls"] == 0
                 and self.counters["plastic_backward_calls"] == 0,
                 "completion", "undeclared local-MSE or broad spine work was charged")

    def report(self) -> dict[str, Any]:
        return {"counters": dict(self.counters), "caps": dict(self.caps),
                "per_arm": dict(self.per_arm), "context_tokens": self.context_tokens,
                "max_new_tokens": self.max_new_tokens,
                "upper_bounds_are_not_measurements": True}


@dataclass
class FitState:
    path: SparkTextMemoryPath
    nonlearning_path: SparkTextMemoryPath
    replay_path: SparkTextMemoryPath
    pre_plastic: Mapping[str, Any]
    trained_plastic: Mapping[str, Any]
    receipt: Mapping[str, Any]
    run_contract: Mapping[str, Any]
    pre_port_state: Mapping[str, torch.Tensor]
    trained_port_state: Mapping[str, torch.Tensor]


@dataclass
class ArmView:
    arm_id: str
    path: SparkTextMemoryPath
    mode: str
    source_prompt_removed: bool
    oracle: bool = False
    inapplicable: bool = False
    controller_source: str = ""
    evidence_source: str = ""
    wrong_p_transform: str | None = None
    port_state: Mapping[str, torch.Tensor] | None = None
    untouched_baseline: bool = False
    frozen_trained_reader: bool = False
    reader_role: str = "pre_reader"
    tiered_store: TieredMemoryStore | None = None
    full_context_records_max: int = 16
    full_context_value_slots_max: int = 64
    readout_results: dict[str, ScopedMemoryReadout] = field(default_factory=dict)


def _port_groups(adapter: Spark2_5Adapter, lr: float) -> list[dict[str, Any]]:
    pairs = [(name, parameter) for name, parameter in adapter.new_named_parameters()
             if name.startswith("extension.spine.") and ".ports." in name
             and "output_projection" in name and parameter.requires_grad]
    _require(bool(pairs), "optimizer", "no final port output projections were found")
    return [{
        "params": [parameter for _, parameter in pairs],
        "param_names": [name for name, _ in pairs],
        "group_name": "port_output", "scope": "text_response/port_output_only",
        "algorithm": "adamw_fused", "fused": True, "lr": float(lr),
        "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0,
    }]


def _new_path(protocol: Protocol, adapter: Spark2_5Adapter, *,
              plastic_payload: Mapping[str, Any] | None = None,
              evidence_payload: Mapping[str, Any] | None = None,
              producer: VersionedTextProducer | None = None) -> SparkTextMemoryPath:
    core = adapter.extension.spine
    memory = protocol.config["memory_contract"]
    experiment = SpineMemoryExperiment(
        scope=PROTOCOL_ID, spec=adapter.spine_spec, core=core,
        bank_experts=1, bank_hidden_width=512, seed=int(protocol.config["seed"]),
        read_records=2, max_journal_records=int(memory["journal_records_max"]),
        max_taught_evidence=int(memory["retained_history_versions_max"]),
        max_cached_snapshots=int(memory["snapshot_cache_entries_max"]),
        codec_version=core.codec_version, representation_version=core.representation_version)
    experiment.bank.lr = float(protocol.config["trainable_groups"]["learning_rate"])
    if plastic_payload is not None:
        experiment.controller.load_checkpoint(copy.deepcopy(dict(plastic_payload)))
    if evidence_payload is not None:
        experiment.evidence.load_checkpoint(copy.deepcopy(dict(evidence_payload)))
    path = SparkTextMemoryPath(
        adapter, experiment,
        producer=producer or VersionedTextProducer(
            representation_version=core.representation_version,
            codec_version=core.codec_version),
        response_lr=float(protocol.config["trainable_groups"]["response_port_learning_rate"]))
    # V2 deliberately reuses the persisted response-optimizer slot for only
    # final port outputs.  No broad response-spine group is constructed here.
    path.response_optimizer = NativeFusedAdamW(
        _port_groups(adapter, float(protocol.config["trainable_groups"]["response_port_learning_rate"])),
        optimizer_name="adamw_fused")
    return path


def _validate_port_state(adapter: Spark2_5Adapter,
                         port_state: Mapping[str, torch.Tensor]) -> None:
    expected = {name for name, _ in adapter.new_named_parameters()
                if name.startswith("extension.spine.") and ".ports." in name
                and "output_projection" in name}
    _require(set(expected) == set(port_state), "port_state",
             "port output identities differ; refusing partial or stale restoration")


def _restore_port_output(adapter: Spark2_5Adapter,
                         port_state: Mapping[str, torch.Tensor]) -> None:
    _validate_port_state(adapter, port_state)
    with torch.no_grad():
        for name, parameter in adapter.new_named_parameters():
            if name in port_state:
                parameter.copy_(port_state[name])


def _dynamic_payload(path: SparkTextMemoryPath) -> dict[str, Any]:
    return {"format": DYNAMIC_FORMAT,
            "plastic": path.experiment.controller.checkpoint_payload(),
            "evidence": path.experiment.evidence.checkpoint_payload(),
            "port_state": _port_output_snapshot(path),
            "port_state_schema": "extension.spine.ports.*.output_projection.v1",
            "text": path._text_checkpoint_payload()}


def _apply_dynamic(path: SparkTextMemoryPath, payload: Mapping[str, Any]) -> None:
    _require(payload.get("format") == DYNAMIC_FORMAT, "dynamic_state", "format differs")
    _validate_port_state(path.adapter, payload["port_state"])
    path.experiment.controller.load_checkpoint(copy.deepcopy(dict(payload["plastic"])))
    path.experiment.evidence.load_checkpoint(copy.deepcopy(dict(payload["evidence"])))
    decoded = path._decode_text_checkpoint(payload["text"])
    path.texts = decoded["texts"]
    path.procedures = decoded["procedures"]
    path.tasks = decoded["tasks"]
    path.explanations = decoded["explanations"]
    path.response_steps = decoded["response_steps"]
    path.response_parameter_updates = decoded["response_parameter_updates"]
    path._query_sequence = decoded["next_query_sequence"]
    path.response_optimizer.load_state_dict(decoded["response_optimizer_state"])
    _restore_port_output(path.adapter, payload["port_state"])
    path.queries.invalidate()


def _clone_path(protocol: Protocol, source: SparkTextMemoryPath, *,
                dynamic: Mapping[str, Any] | None = None) -> SparkTextMemoryPath:
    path = _new_path(protocol, source.adapter, producer=source.producer)
    _apply_dynamic(path, dynamic or _dynamic_payload(source))
    return path


def _run_contract(protocol: Protocol, dataset: PreparedDataset, backend: Backend) -> dict[str, Any]:
    core = backend.adapter.extension.spine
    return {
        "schema": SPINE_SCHEMA, "spine_schema": SPINE_SCHEMA,
        "model_identity": backend.adapter.durable_identity(),
        "codec_version": core.codec_version,
        "representation_version": core.representation_version,
        "protocol_id": protocol.protocol_id, "protocol_sha256": protocol.sha256,
        "prepared_manifest_sha256": dataset.manifest_sha256,
        "backend": backend.label, "native_admissible": bool(backend.native_admissible),
    }


def _source_reference(protocol: Protocol, backend: Backend) -> dict[str, Any]:
    if backend.label == CPU_LABEL:
        return {"identity": CPU_LABEL, "official_source": False,
                "native_admissible": False, "model_identity": backend.adapter.durable_identity(),
                "fixture_seed": backend.fixture_seed, "derived_caches_required": False,
                "note": "contextual CPU engineering fixture; not Spark native evidence"}
    source = protocol.source_identity
    return {"identity": "official_fp8", "official_source": True,
            "repo_id": source["repo_id"], "revision": source["revision"],
            "local_path": source["local_path"], "config_sha256": source["config_sha256"],
            "weight_sha256": source["weight_sha256"], "execution_format": source["execution_format"],
            "required_backend": source["required_backend"], "native_admissible": True,
            "model_identity": backend.adapter.durable_identity(), "derived_caches_required": False}


def _fixture_texts(dataset: PreparedDataset) -> list[str]:
    texts = []
    for row in dataset.observations():
        texts.append(str(row["text"]))
    for row in dataset.fit_events():
        texts.extend((str(row["question"]), str(row["answer"]), str(row["expected"])))
    for row in dataset.procedures():
        for key in ("applicability", "prerequisites", "exceptions"):
            texts.extend(str(item) for item in row[key])
        texts.extend((str(row["outcome"]), str(row["failure_lesson"])))
    for row in dataset._rows("development.jsonl"):
        texts.append(str(row["question"]))
    for row in dataset._rows("parity_requests.jsonl"):
        texts.append(str(row["question"]))
    return texts


def build_cpu_backend(protocol: Protocol, dataset: PreparedDataset, *,
                      spine_sites: Sequence[int]) -> Backend:
    from replica_lab.contextual_spark_fixture import build_contextual_adapter
    tokenizer = DeterministicTinyTokenizer(texts=_fixture_texts(dataset))
    adapter, _, _, _ = build_contextual_adapter(
        seed=int(protocol.config["seed"]), layers=32,
        spine_sites=tuple(int(site) for site in spine_sites), spine_slots=8,
        memory_size=256, heads=2, device="cpu", vocab_size=8192,
        max_position_embeddings=4096)
    stop_ids = tuple(sorted({int(tokenizer._token_to_id["<eos>"]),
                             int(tokenizer._token_to_id["<pad>"])}))
    return Backend(label=CPU_LABEL, adapter=adapter, tokenizer=tokenizer, device="cpu",
                   native_admissible=False, chat_mode={"enable_thinking": False},
                   stop_token_ids=stop_ids, fixture_seed=int(protocol.config["seed"]))


def verify_native_authorization(protocol: Protocol, dataset: PreparedDataset, *,
                                authorization_path: str | Path, output_root: Path,
                                allow_existing_output: bool = False) -> dict[str, Any]:
    from preflight_spark_persistent_learner_v2 import _validate_authorization
    receipt_path = Path(authorization_path)
    _require(receipt_path.is_file(), "authorization", f"missing {receipt_path}")
    receipt = _validate_authorization(
        protocol.config, receipt_path, output_root, protocol.root,
        protocol.implementation_hashes(), config_path=protocol.path,
        allow_existing_output=allow_existing_output)
    evidence_path = protocol.root / str(protocol.config["launch"]["backend_evidence_path"])
    evidence = _read_json(evidence_path, "native_fp8_backend_evidence")
    return {"receipt": receipt, "backend_evidence": evidence,
            "authorization_path": str(receipt_path), "backend_evidence_path": str(evidence_path),
            "authorization_sha256": _sha256(receipt_path),
            "backend_evidence_sha256": _sha256(evidence_path)}


def build_native_backend(protocol: Protocol, dataset: PreparedDataset, *,
                         authorization_path: str | Path, output_root: Path,
                         device: str, spine_sites: Sequence[int],
                         allow_existing_output: bool = False) -> Backend:
    _require(device == "mps", "native_device", "official FP8 execution requires --device mps")
    authorization = verify_native_authorization(
        protocol, dataset, authorization_path=authorization_path, output_root=output_root,
        allow_existing_output=allow_existing_output)
    source = protocol.source_identity
    source_root = protocol.root / str(source["local_path"])
    raw = _read_json(source_root / "config.json", "source_config")
    _require(_sha256(source_root / "config.json") == source["config_sha256"],
             "source", "official FP8 config hash differs")
    config = load_spark2_5_config(raw, require_official_geometry=True)
    precision = official_native_fp8_precision(config, artifact_sha256=source["weight_sha256"])
    _require(precision.kind == OFFICIAL_NATIVE_FP8, "source", "active source is not official FP8")
    admission = evaluate_admission(precision, device=device)
    _require(admission.admitted, "native_fp8_backend", "; ".join(admission.reasons))
    contract = getattr(Spark2_5Adapter, "native_fp8_admission", None)
    _require(isinstance(contract, Mapping), "native_fp8_adapter",
             "the active adapter publishes no native-FP8 admission contract; no BF16/derivative substitution is allowed")
    for filename, expected in (
        ("tokenizer.json", protocol.config["runtime"]["tokenizer"]["tokenizer_json_sha256"]),
        ("tokenizer_config.json", protocol.config["runtime"]["tokenizer"]["tokenizer_config_sha256"]),
        ("chat_template.jinja", protocol.config["runtime"]["tokenizer"]["chat_template_sha256"]),
    ):
        path = source_root / filename
        _require(path.is_file() and _sha256(path) == expected, "tokenizer",
                 f"{filename} identity differs")
    weight_path = source_root / "model.safetensors"
    _require(weight_path.is_file() and _sha256(weight_path) == source["weight_sha256"],
             "source", "official FP8 weight hash differs")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise Refused("native_runtime", "transformers is unavailable") from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(source_root), local_files_only=True, trust_remote_code=True)
    _require(getattr(tokenizer, "is_fast", False), "tokenizer", "fast tokenizer required")
    tokenizer_config = _read_json(source_root / "tokenizer_config.json", "tokenizer_config")
    template = (source_root / "chat_template.jinja").read_text(encoding="utf-8")
    tokenizer_json = (source_root / "tokenizer.json").read_text(encoding="utf-8")
    identity = load_tokenizer_identity(config, tokenizer_config, template, tokenizer_json=tokenizer_json)
    _require(identity.vocab_size == config.geometry.vocab_size, "tokenizer", "vocabulary differs")
    mode = resolve_thinking_mode(identity, "none")
    base = AutoModelForCausalLM.from_pretrained(
        str(source_root), local_files_only=True, trust_remote_code=True,
        torch_dtype="auto", attn_implementation="eager", use_safetensors=True)
    _require(getattr(base.config, "model_type", None) == "spark2_5", "source",
             "loaded model is not spark2_5")
    adapter = Spark2_5Adapter(
        base, config, precision, device=device, require_official_geometry=True,
        projector_in_dim=int(config.geometry.hidden_size), zero_init_output=True,
        memory_size=256, heads=4, spine_sites=tuple(int(site) for site in spine_sites),
        spine_slots=8)
    adapter.extension.spine.codec_version = TEXT_CODEC_VERSION
    adapter.extension.spine.representation_version = TEXT_REPRESENTATION_VERSION
    adapter.eval()
    adapter.seal_original_identity()
    stop_ids = tuple(sorted({int(getattr(tokenizer, "eos_token_id", 1)),
                             int(getattr(tokenizer, "pad_token_id", 2))}))
    return Backend(label=NATIVE_LABEL, adapter=adapter, tokenizer=tokenizer, device=device,
                   native_admissible=True, chat_mode=mode, stop_token_ids=stop_ids,
                   authorization=authorization["receipt"])


def build_backend(kind: str, protocol: Protocol, dataset: PreparedDataset, *,
                  device: str, output_root: Path, spine_sites: Sequence[int],
                  authorization_path: str | Path | None = None,
                  allow_existing_output: bool = False) -> Backend:
    if kind == CPU_LABEL:
        _require(device == "cpu", "cpu_fixture", "CPU fixture requires --device cpu")
        return build_cpu_backend(protocol, dataset, spine_sites=spine_sites)
    _require(kind == NATIVE_LABEL, "backend", f"unknown backend {kind}")
    _require(authorization_path is not None, "authorization", "native execution requires a receipt")
    return build_native_backend(
        protocol, dataset, authorization_path=authorization_path, output_root=output_root,
        device=device, spine_sites=spine_sites, allow_existing_output=allow_existing_output)


def _observe_rows(path: SparkTextMemoryPath, rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        path.observe_text(
            str(row["text"]), record_id=str(row["record_id"]),
            source_epoch=int(row["source_epoch"]), available_at=int(row["available_at"]),
            valid_from=int(row["valid_from"]),
            valid_until=None if row.get("valid_until") is None else int(row["valid_until"]),
            source_lineage=dict(row["provenance"]), durable_ref=dict(row["durable_ref"]),
            retain_canonical_text=True, payload_type="observation", verified=False,
            supersedes=None if row.get("supersedes") is None else str(row["supersedes"]))


def _add_procedures(path: SparkTextMemoryPath, rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        procedure_id = str(row["procedure_id"])
        if procedure_id in path.procedures:
            continue
        sources = tuple(str(value) for value in row["source_record_ids"])
        if not sources or any(source not in path.texts for source in sources):
            continue
        path.add_procedure(
            procedure_id, applicability=row["applicability"], prerequisites=row["prerequisites"],
            exceptions=row["exceptions"], outcome=str(row["outcome"]),
            outcome_independently_checked=bool(row["outcome_independently_checked"]),
            outcome_checker=row.get("outcome_checker"), source_record_ids=sources)


def _apply_task_rows(path: SparkTextMemoryPath, rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in path.tasks:
            path.open_task(task_id, tuple((str(goal["subgoal_id"]), str(goal["description"]))
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
                task_id, subgoal_id, observation=str(completion["observation"]),
                observed_at=int(completion["observed_at"]), checker=str(completion["checker"]),
                evidence_record_ids=sources)


def _invalidate_path_dependencies(path: SparkTextMemoryPath,
                                  superseded_ids: set[str]) -> None:
    if not superseded_ids:
        return
    for procedure_id, procedure in list(path.procedures.items()):
        if any(record_id in procedure.source_record_ids for record_id in superseded_ids):
            del path.procedures[procedure_id]
    for task_id, workspace in list(path.tasks.items()):
        remaining = tuple(item for item in workspace.observed_completions
                          if not any(record_id in item.evidence_record_ids
                                     for record_id in superseded_ids))
        if len(remaining) != len(workspace.observed_completions):
            path.tasks[task_id] = replace(workspace, observed_completions=remaining)
    path.experiment.invalidate(reason="source_correction_dependents")
def _restore_readout_records(path: SparkTextMemoryPath,
                             readout: ScopedMemoryReadout) -> tuple[str, ...]:
    restored = []
    for record in readout.records:
        if record.status not in {"ok", "degraded"}:
            raise Refused("tiered_readout_restoration",
                          f"{record.record_id}: {record.reason}")
        required = (
            record.text, record.source_epoch, record.available_at, record.valid_from,
            record.event_at, record.durable_ref, record.source_provenance,
            record.producer_revision, record.representation_version,
            record.codec_version, record.payload_type,
        )
        if (any(value is None for value in required)
                or not record.available_at_mapped
                or not record.producer_revision_available
                or not record.representation_version_available
                or not record.codec_version_available
                or not record.payload_type_available):
            raise Refused("tiered_readout_restoration",
                          f"{record.record_id}: incomplete restoration authority")
        if record.kind == "correction" and not record.supersedes:
            raise Refused("tiered_readout_restoration",
                          f"{record.record_id}: correction predecessor is missing")
        authority = dict(record.source_provenance)
        if (authority.get("record_id") != record.record_id
                or authority.get("text_sha256") != record.text_sha256):
            raise Refused("tiered_readout_restoration",
                          f"{record.record_id}: current provenance identity differs")
        exact_fields = {
            "source_epoch": record.source_epoch,
            "available_at": record.available_at,
            "event_at": record.event_at,
            "valid_from": record.valid_from,
            "valid_until": record.valid_until,
            "producer_revision": record.producer_revision,
            "representation_version": record.representation_version,
            "codec_version": record.codec_version,
            "payload_type": record.payload_type,
        }
        for name, value in exact_fields.items():
            if name in authority and authority[name] != value:
                raise Refused("tiered_readout_restoration",
                              f"{record.record_id}: provenance {name} differs")
            authority[name] = value
        if authority.get("source", record.source_id) != record.source_id:
            raise Refused("tiered_readout_restoration",
                          f"{record.record_id}: source identity differs")
        if (record.producer_revision != path.producer.producer_revision
                or record.representation_version != path.producer.representation_version
                or record.codec_version != path.producer.codec_version):
            raise Refused("tiered_readout_restoration",
                          f"{record.record_id}: neural producer version differs")
        authority["record_lineage"] = list(record.source_lineage)
        if record.supersedes is not None:
            if "supersedes" in authority and authority["supersedes"] != record.supersedes:
                raise Refused("tiered_readout_restoration",
                              f"{record.record_id}: supersedes differs")
            authority["supersedes"] = record.supersedes
        existing = path.texts.get(record.record_id)
        if existing is not None:
            evidence = next((item for item in path.experiment.evidence.active
                             if item.record_id == record.record_id), None)
            if evidence is None:
                raise Refused("tiered_readout_restoration",
                              f"{record.record_id}: active evidence mirror is missing")
            if (existing.canonical_text != record.text
                    or existing.text_sha256 != record.text_sha256
                    or existing.provenance.get("text_sha256") != record.text_sha256
                    or existing.producer_revision != record.producer_revision
                    or existing.representation_version != record.representation_version
                    or existing.codec_version != record.codec_version
                    or evidence.source_epoch != record.source_epoch
                    or evidence.available_at != record.available_at
                    or evidence.valid_from != record.valid_from
                    or evidence.valid_until != record.valid_until
                    or evidence.payload_type != record.payload_type
                    or evidence.supersedes != record.supersedes):
                raise Refused("tiered_readout_restoration",
                              f"{record.record_id}: mirrored canonical authority differs")
            continue
        path.observe_text(
            record.text, record_id=record.record_id,
            source_epoch=record.source_epoch, available_at=record.available_at,
            valid_from=record.valid_from, valid_until=record.valid_until,
            source_lineage=authority, durable_ref=record.durable_ref,
            retain_canonical_text=True, payload_type=record.payload_type,
            verified=record.verification_state in {
                "verified_source", "observed_tool", "verified_human"},
            supersedes=record.supersedes,
            reconcile=(record.supersedes is not None and not any(
                item.record_id == record.supersedes
                for item in path.experiment.evidence.active)))
        restored.append(record.record_id)
    return tuple(restored)

def _training_content(observation_texts: Sequence[str], question: str) -> str:
    source = "\n".join(text for text in observation_texts if text)
    return (source + "\n\n" if source else "") + f"Question: {question}\nAnswer:"


def _readout_scope(query_time: int) -> MemoryScope:
    return MemoryScope(subject_id="replica", branch_id="main", purpose="assistant",
                       as_of=int(query_time))


def _tiered_readout(store: TieredMemoryStore, question: str,
                    query_time: int, *, full_context: bool = False,
                    full_context_records_max: int = 16,
                    full_context_value_slots_max: int = 64,
                    allow_empty: bool = False) -> ScopedMemoryReadout:
    scope = _readout_scope(query_time)
    selection_mode = "all_eligible" if full_context else "semantic"
    budget = ReadoutBudget(
        max_records=full_context_records_max if full_context else 2,
        max_value_slots=full_context_value_slots_max if full_context else 8,
        cold_wait_seconds=0, allow_expanded_context=full_context)
    attempts = [read_scoped_memory(
        store, question, scope, budget=budget, selection_mode=selection_mode)]
    for ticket in attempts[0].pending_cold:
        attempts.append(resume_scoped_readout(
            store, ticket.ticket_id, scope=scope, budget=budget))
    stop = any(attempt.new_decision_boundary or not attempt.ok
               for attempt in attempts[1:])
    if not attempts[0].pending_cold:
        final = attempts[0]
    elif stop:
        final = attempts[-1]
    else:
        final = read_scoped_memory(
            store, question, scope, budget=budget, selection_mode=selection_mode)
        attempts.append(final)
    boundary = any(attempt.new_decision_boundary for attempt in attempts)
    diagnostics = dict(final.diagnostics)
    diagnostics["selection_mode"] = selection_mode
    diagnostics["readout_attempts"] = [{
        "selection_id": attempt.selection_id,
        "resumed_ticket_id": attempt.resumed_ticket_id,
        "status": attempt.status,
        "reason": attempt.outcome.reason,
        "record_ids": list(attempt.record_ids),
        "new_decision_boundary": attempt.new_decision_boundary,
        "metrics": attempt.metrics.document(),
    } for attempt in attempts]
    final = replace(
        final, metrics=merge_readout_metrics([attempt.metrics for attempt in attempts]),
        new_decision_boundary=boundary, diagnostics=diagnostics)
    if boundary:
        final = replace(
            final, status="invalidated",
            outcome=Outcome(
                status="invalidated", reason="decision_boundary_during_readout",
                detail={"attempt_count": len(attempts)}, retryable=True))
    if (final.pending_cold or final.new_decision_boundary or final.representation_stale
            or final.incomplete_record_ids or final.invalidated_record_ids
            or final.denied_record_ids):
        raise Refused("tiered_readout", final.outcome.reason)
    empty = (allow_empty and final.candidate_count == 0 and not final.records
             and final.outcome.reason == "no_semantic_candidate")
    if not final.ok and not empty:
        raise Refused("tiered_readout", final.outcome.reason)
    if not final.records and not empty:
        raise Refused("tiered_readout", "empty_scoped_readout")
    if full_context and (final.truncated or final.candidate_count != len(final.records)):
        raise Refused(
            "full_context_incomplete",
            f"eligible={final.candidate_count} restored={len(final.records)}")
    return final


def _charge_readout(ledger: WorkLedger, result: ScopedMemoryReadout) -> None:
    metrics = result.metrics
    ledger.set_measured("readout_cache_bytes", metrics.cache_bytes)
    ledger.set_measured("readout_result_bytes", metrics.result_payload_bytes)
    ledger.charge("readout_selection_calls", metrics.selection_calls)
    ledger.charge("readout_file_reads", metrics.total_file_reads)
    ledger.charge("readout_io_bytes", metrics.total_io_bytes)
    ledger.charge("readout_ns", metrics.total_ns)
    ledger.charge("cold_recall_tickets", metrics.cold_tickets)
    ledger.charge("cold_recall_resumes", metrics.cold_resumes)
    ledger.charge("revalidation_failures", metrics.revalidation_failures)
    ledger.charge("payload_read_calls", metrics.payload_read_calls)
    ledger.charge("payload_file_reads", metrics.payload_file_reads)
    ledger.charge("payload_io_bytes", metrics.payload_io_bytes)
    ledger.charge("revalidations", metrics.revalidations)


def _fit_query(path: SparkTextMemoryPath, backend: Backend, event: Mapping[str, Any],
               observations: Mapping[str, Any], ledger: WorkLedger,
               tiered: TieredMemoryStore | None = None):
    rows = [observations[str(record_id)] for record_id in event["observation_record_ids"]]
    content = _training_content([str(row["text"]) for row in rows], str(event["question"]))
    ids, mask, labels = backend.encode_training(content, str(event["answer"]), ledger.context_tokens)
    path.experiment.invalidate(reason="v2_fit_no_cache_reuse")
    readout = None if tiered is None else _tiered_readout(
        tiered, str(event["question"]), int(event["sequence"]))
    if readout is not None:
        _restore_readout_records(path, readout)
    query = prepare_arm_query(
        path, arm="e_plus_p", question=str(event["question"]), input_ids=ids,
        attention_mask=mask, query_time=int(event["sequence"]),
        token_available_at=tuple(int(event["sequence"]) for _ in range(ids.shape[1])),
        selected_record_ids=None if readout is None else readout.record_ids)
    if readout is not None:
        _charge_readout(ledger, readout)
        ledger.query_sensing(int(ids.shape[1]), cache_hits=readout.metrics.cache_hits,
                             cache_misses=readout.metrics.cache_misses)
    else:
        ledger.query_sensing(int(ids.shape[1]))
    ledger.retrieval()
    return query, labels, rows


def _null_comparison(path: SparkTextMemoryPath, query) -> dict[str, Any]:
    with torch.no_grad():
        response = path.respond(query, extra_passes=("null",))
    null = response.null_logits["null"]
    difference = (response.logits - null).abs()
    return {"exact_null_parity": bool(torch.equal(response.logits, null)),
            "max_absolute_difference": float(difference.max()),
            "nonzero_influence": bool(torch.count_nonzero(difference)),
            "decoder_forwards": 2, "charges": dict(response.charges)}


def _checkpoint(path: SparkTextMemoryPath, output: Path, relative: str,
                run_contract: Mapping[str, Any], ledger: WorkLedger,
                meter: publication.LiveByteMeter) -> dict[str, Any]:
    receipt = path.checkpoint(output, relative, run_contract=run_contract, meter=meter)
    ledger.charge("checkpoint_writes")
    return {"relative": relative, **receipt}


def _retire_oldest(output: Path, records: list[dict[str, Any]],
                   evicted: list[dict[str, Any]], ledger: WorkLedger) -> None:
    if len(records) <= 4:
        ledger.set_measured("retained_checkpoint_artifacts", len(records))
        return
    old = records.pop(0)
    artifacts = [old["experiment"]["artifact"], old["text_artifact"]]
    for artifact in artifacts:
        target = output / str(artifact["path"])
        if target.is_file():
            target.unlink()
    evicted.append(old)
    ledger.set_measured("retained_checkpoint_artifacts", len(records))


def _open_tiered(protocol: Protocol, output: Path) -> TieredMemoryStore:
    tiers = protocol.config["tiered_storage"]
    memory = protocol.config["memory_contract"]
    ceilings = TierCeilings(
        hot_bytes=int(tiers["hot_bytes_max"]), warm_bytes=int(tiers["warm_bytes_max"]),
        cold_bytes=int(tiers["cold_bytes_max"]),
        identity_reserved_bytes=int(tiers["identity_bytes_max"]),
        replay_reserved_bytes=int(tiers["replay_bytes_max"]),
        workspace_quota_bytes=int(memory["metadata_bytes_max"]),
        artifact_budget_bytes=min(int(tiers["cold_bytes_max"]),
                                  int(protocol.schedule["artifact_bytes_max"])))
    return TieredMemoryStore(output / "tiered", profile_id="P1", ceilings=ceilings)


def _scope_for_row(row: Mapping[str, Any]) -> MemoryScope:
    return MemoryScope(subject_id="replica", branch_id="main", purpose="assistant",
                       as_of=int(row["available_at"]))


def _readable_values(text: str) -> tuple[tuple[float, ...], ...]:
    # Deterministic readable engineering values, separate from the retrieval key.
    payload = hashlib.shake_256(text.encode("utf-8")).digest(4 * 256)
    return tuple(tuple((byte - 127.5) / 127.5 for byte in payload[slot * 256:(slot + 1) * 256])
                 for slot in range(4))



def _row_provenance(row: Mapping[str, Any], producer: VersionedTextProducer) -> dict[str, Any]:
    text = str(row["text"])
    record_id = str(row["record_id"])
    provenance = dict(row["provenance"])
    if provenance.get("record_id") != record_id:
        raise Refused("tiered_provenance", f"{record_id}: record identity differs")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if provenance.get("text_sha256") != digest:
        raise Refused("tiered_provenance", f"{record_id}: text digest differs")
    expected = {
        "producer_revision": producer.producer_revision,
        "representation_version": producer.representation_version,
        "codec_version": producer.codec_version,
        "payload_type": "observation",
    }
    for name, value in expected.items():
        if provenance.get(name) not in (None, value):
            raise Refused("tiered_provenance", f"{record_id}: {name} differs")
        provenance[name] = value
    return provenance
def _draft_for_row(row: Mapping[str, Any], producer: VersionedTextProducer) -> EvidenceRecordDraft:
    text = str(row["text"])
    provenance = _row_provenance(row, producer)
    return EvidenceRecordDraft(
        record_id=str(row["record_id"]), text=text,
        source_id=str(provenance.get("source", row["world_id"])),
        source_epoch=int(row["source_epoch"]), observed_at=int(row["available_at"]),
        event_at=int(row["valid_from"]), valid_from=int(row["valid_from"]),
        valid_to=None if row.get("valid_until") is None else int(row["valid_until"]),
        authenticity="source_report", verification_state="verified_source",
        kind="source", values=_readable_values(text), key=build_retrieval_key(text),
        value_producer_id="v2_deterministic_readable_values",
        value_producer_version="v1",
        source_lineage=(json.dumps(provenance, sort_keys=True),))


def _tiered_admit_rows(store: TieredMemoryStore,
                       rows: Iterable[Mapping[str, Any]], producer: VersionedTextProducer) -> list[dict[str, Any]]:
    receipts = []
    for row in rows:
        scope = _scope_for_row(row)
        provenance = _row_provenance(row, producer)
        if row.get("supersedes") is not None:
            result = store.correct_record(
                str(row["supersedes"]), correction_id=str(row["record_id"]),
                text=str(row["text"]), source_id=str(provenance.get("source", row["world_id"])),
                source_epoch=int(row["source_epoch"]), observed_at=int(row["available_at"]),
                event_at=int(row["valid_from"]), valid_from=int(row["valid_from"]),
                valid_to=None if row.get("valid_until") is None else int(row["valid_until"]),
                values=_readable_values(str(row["text"])),
                value_producer_id="v2_deterministic_readable_values",
                value_producer_version="v1", scope=scope,
                authenticity="source_report", verification_state="verified_source",
                source_provenance=provenance)
        else:
            result = store.admit_cluster(
                str(row["world_id"]), scope=scope, records=[_draft_for_row(row, producer)], placement="warm")
        receipts.append(result.document())
    return receipts


def _tiered_sync_procedures(store: TieredMemoryStore,
                            rows: Iterable[Mapping[str, Any]]) -> None:
    existing = {(str(row["skill_id"]), int(row["version"])) for row in store.procedures()}
    for row in rows:
        key = (str(row["procedure_id"]), 1)
        if key in existing:
            continue
        checked = bool(row["outcome_independently_checked"])
        outcome = store.record_procedure(
            key[0], 1, title=key[0],
            applicability={"world_id": row["world_id"],
                           "expressions": list(row["applicability"]),
                           "exceptions": list(row["exceptions"])},
            prerequisites=list(row["prerequisites"]),
            steps=["read exact evidence", "apply independently checked outcome"],
            outcome="verified" if checked else "failed",
            error=None if checked else "outcome was not independently checked",
            evidence_record_ids=[str(value) for value in row["source_record_ids"]],
            verified_by=str(row["outcome_checker"]) if checked else None)
        _require(outcome.ok, "tiered_procedure", outcome.reason)
        existing.add(key)


def _fp32_bytes(tensors: Iterable[torch.Tensor]) -> bytes:
    parts = []
    for tensor in tensors:
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        parts.append(value.numpy().tobytes())
    return b"".join(parts)


def _expert_bytes(parameters: Sequence[tuple[str, torch.Tensor]],
                  moment_state: Mapping[str, Any]) -> tuple[bytes, bytes]:
    moment_tensors = []
    for name, parameter in parameters:
        state = moment_state.get(f"0.{name}")
        if state is None:
            moment_tensors.extend((torch.zeros_like(parameter, dtype=torch.float32, device="cpu"),
                                   torch.zeros_like(parameter, dtype=torch.float32, device="cpu")))
        else:
            moment_tensors.extend((state["exp_avg"], state["exp_avg_sq"]))
    return (_fp32_bytes(parameter for _, parameter in parameters),
            _fp32_bytes(moment_tensors))


def _tiered_write_expert(store: TieredMemoryStore, path: SparkTextMemoryPath,
                         *, publish: bool = False,
                         lineage: Sequence[str] | None = None) -> dict[str, Any]:
    parameters = list(path.experiment.bank.expert(0).named_parameters())
    weights, moments = _expert_bytes(parameters, path.experiment.bank.optimizer_state_for_expert(0))
    outcome = store.write_expert_state(
        "expert-0", weights=weights, moments=moments,
        clock=int(path.experiment.controller.clock.eligible_events),
        lineage=(tuple(lineage) if lineage is not None else
                 tuple(sorted(str(key) for key in path.experiment.taught_evidence))),
        encoder_id=path.experiment.core.representation_version)
    _require(outcome.ok, "tiered_expert", outcome.reason)
    if publish:
        published = store.publish_expert("expert-0")
        _require(published.ok, "tiered_expert_publication", published.reason)
    loaded = store.load_expert("expert-0")
    _require(loaded.outcome.ok and loaded.state is not None,
             "tiered_expert_readback", "published P state cannot be read back")
    return loaded.state.document()


def _tiered_reset_expert(store: TieredMemoryStore, snapshot) -> dict[str, Any]:
    parameters = list(snapshot.parameters.items())
    weights, moments = _expert_bytes(parameters, snapshot.optimizer_state)
    outcome = store.reset_expert(
        "expert-0", weights=weights, moments=moments, clock=int(snapshot.clock_events),
        lineage=(), authorization="authenticated_reset",
        reason="correction invalidated dependent P and port state")
    _require(outcome.ok, "tiered_expert_reset", outcome.reason)
    return outcome.document()


def _port_output_snapshot(path: SparkTextMemoryPath) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone()
            for name, parameter in path.adapter.new_named_parameters()
            if name.startswith("extension.spine.") and ".ports." in name
            and "output_projection" in name}


def _reset_path_for_correction(path: SparkTextMemoryPath, snapshot,
                               port_parameters: Mapping[str, torch.Tensor],
                               port_optimizer_state: Mapping[str, Any]) -> None:
    restore_p_state(path, snapshot)
    for name, parameter in path.adapter.new_named_parameters():
        if name in port_parameters:
            parameter.data.copy_(port_parameters[name])
    path.response_optimizer.load_state_dict(copy.deepcopy(dict(port_optimizer_state)))
    path.response_steps = 0
    path.response_parameter_updates = 0
    path.queries.invalidate()


def _tiered_sync_tasks(store: TieredMemoryStore, path: SparkTextMemoryPath,
                       dataset: PreparedDataset) -> None:
    rows = {str(row["task_id"]): row for row in dataset.tasks()}
    for task_id, workspace in path.tasks.items():
        source = rows.get(task_id, {})
        completions = list(workspace.observed_completions)
        completed_ids = {item.subgoal_id for item in completions}
        unresolved = [item.subgoal_id for item in workspace.subgoals
                      if item.subgoal_id not in completed_ids]
        evidence_ids = sorted({str(record_id) for item in completions
                               for record_id in item.evidence_record_ids})
        payload = {
            "task_id": task_id, "subject": "replica", "branch": "main", "purpose": "assistant",
            "authorized_goal": "inspect the authorized fixture state and route only by current evidence",
            "goal_authority": "fixture_protocol_not_model_proposal",
            "success_conditions": ["every required subgoal has an independent observed completion"],
            "constraints": ["generated claims are not receipts", "no cross-scope evidence"],
            "subgoals": [{"subgoal_id": item.subgoal_id, "description": item.description}
                         for item in workspace.subgoals],
            "claims": {"known": evidence_ids, "inferred": [], "unknown": [], "hypothetical": []},
            "evidence_references": evidence_ids, "unresolved_obligations": unresolved,
            "actions": {"proposed": [], "attempted": [],
                        "completed": [{"subgoal_id": item.subgoal_id, "checker": item.checker}
                                      for item in completions], "denied": [], "failed": []},
            "actual_outcomes": [{"subgoal_id": item.subgoal_id, "observation": item.observation,
                                 "observed_at": item.observed_at, "checker": item.checker}
                                for item in completions],
            "task_revisions": [{"source_epoch": int(item.observed_at), "subgoal_id": item.subgoal_id}
                               for item in completions],
            "remaining_budget": {"unresolved_subgoals": len(unresolved), "native_retries": 0},
            "source_task": source,
            "model_component_versions": {
                "codec": path.experiment.core.codec_version,
                "representation": path.experiment.core.representation_version,
                "path": "SparkTextMemoryPath"},
        }
        receipt = store.save_workspace_snapshot(task_id, payload, resumable=bool(unresolved))
        _require(receipt.outcome.ok, "tiered_workspace", receipt.outcome.reason)


def _tiered_summary(store: TieredMemoryStore) -> dict[str, Any]:
    return store.state_summary()


def run_fit(protocol: Protocol, dataset: PreparedDataset, backend: Backend,
            output: Path, ledger: WorkLedger, meter: publication.LiveByteMeter) -> FitState:
    output.mkdir(parents=True, exist_ok=True)
    (output / "fit").mkdir(parents=True, exist_ok=True)
    run_contract = _run_contract(protocol, dataset, backend)
    path = _new_path(protocol, backend.adapter)
    nonlearning_path = _new_path(protocol, backend.adapter)
    replay_path = _new_path(protocol, backend.adapter)
    pre_plastic = copy.deepcopy(path.experiment.controller.checkpoint_payload())
    initial_p_snapshot = capture_p_state(path)
    initial_port_parameters = _port_output_snapshot(path)
    learning_port_state = initial_port_parameters
    initial_port_optimizer_state = copy.deepcopy(path.response_optimizer.state_dict())
    fit_lineage: list[str] = []
    correction_resets: list[dict[str, Any]] = []
    observations = {str(row["record_id"]): row for row in dataset.observations()}
    procedures = dataset.procedures()
    tasks = dataset.tasks()
    tiered = _open_tiered(protocol, output)
    tiered.register_expert("expert-0", profile_id="P1", lineage=(),
                           encoder_id=path.experiment.core.representation_version)
    tiered_events: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    evicted: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    initial_parity = None
    last_event = None
    last_rows = None
    for event in dataset.fit_events():
        row_observations = [observations[str(record_id)] for record_id in event["observation_record_ids"]]
        for current in (path, nonlearning_path, replay_path):
            _observe_rows(current, row_observations)
            _invalidate_path_dependencies(
                current, {str(row["supersedes"]) for row in row_observations
                          if row.get("supersedes") is not None})
            _add_procedures(current, procedures)
            _apply_task_rows(current, tasks)
        tiered_admission = _tiered_admit_rows(tiered, row_observations, path.producer)
        correction_reset = None
        if any(row.get("supersedes") is not None for row in row_observations):
            for current in (path, nonlearning_path, replay_path):
                _reset_path_for_correction(
                    current, initial_p_snapshot, initial_port_parameters,
                    initial_port_optimizer_state)
            correction_reset = _tiered_reset_expert(tiered, initial_p_snapshot)
            correction_resets.append({
                "event_id": event["event_id"], "admission": tiered_admission,
                "reset": correction_reset, "exact_unlearning_claimed": False})
            fit_lineage = []
            learning_port_state = initial_port_parameters
        fit_lineage.extend(str(row["record_id"]) for row in row_observations)
        _tiered_sync_procedures(tiered, procedures)
        _tiered_sync_tasks(tiered, path, dataset)
        _restore_port_output(path.adapter, learning_port_state)
        query, labels, _ = _fit_query(path, backend, event, observations, ledger, tiered=tiered)
        if initial_parity is None:
            initial_parity = _null_comparison(path, query)
            ledger.parity_forwards(int(query.input_ids.shape[1]))
        learned = direct_response_ce_step(
            path, query, labels, event_id=str(event["event_id"]), expert_id=0,
            port_optimizer=path.response_optimizer, arm="learned_P")
        ledger.objective(int(query.input_ids.shape[1]))
        port_step = bool(learned["optimizer_ownership"].get("port_optimizer_stepped", False))
        if learned["optimizer_ownership"]["physical_optimizer_steps"]:
            ledger.charge("optimizer_steps")
            ledger.charge("selected_P_optimizer_steps")
        if port_step:
            ledger.charge("optimizer_steps")
            ledger.charge("response_port_optimizer_steps")
            path.response_steps += 1
            path.response_parameter_updates += int(bool(
                learned.get("port_changed_parameter_names")))
            learned["optimizer_ownership"]["response_steps_after"] = path.response_steps
        learning_port_state = _port_output_snapshot(path)
        _restore_port_output(nonlearning_path.adapter, initial_port_parameters)
        nl_query, nl_labels, _ = _fit_query(nonlearning_path, backend, event, observations, ledger, tiered=tiered)
        nonlearning = direct_response_ce_step(
            nonlearning_path, nl_query, nl_labels,
            event_id=str(event["event_id"]) + ":matched_nonlearning", expert_id=0,
            apply=False, port_optimizer=nonlearning_path.response_optimizer, arm="no_update")
        ledger.objective(int(nl_query.input_ids.shape[1]), nonlearning=True)
        _restore_port_output(replay_path.adapter, initial_port_parameters)
        replay_query, replay_labels, _ = _fit_query(replay_path, backend, event, observations, ledger, tiered=tiered)
        replay = evaluate_response_ce(replay_path, replay_query, labels=replay_labels)
        ledger.replay(int(replay_query.input_ids.shape[1]))
        tiered_expert = _tiered_write_expert(tiered, path, lineage=fit_lineage)
        for _ in range(3):
            ledger.charge("verification_calls")
        _restore_port_output(path.adapter, learning_port_state)
        relative = f"fit/event_{int(event['sequence']):02d}/state"
        receipt = _checkpoint(path, output, relative, run_contract, ledger, meter)
        checkpoints.append(receipt)
        _retire_oldest(output, checkpoints, evicted, ledger)
        ledger.charge("fit_events")
        row = {"event_id": event["event_id"], "sequence": event["sequence"],
               "expected_contract_match": event["answer"] == event["expected"],
               "learned": learned, "matched_nonlearning": nonlearning,
               "replay": {"loss": replay["loss"], "charges": replay["charges"]},
               "checkpoint": receipt, "tiered_expert": tiered_expert,
               "correction_reset": correction_reset}
        tiered_events.append({"event_id": event["event_id"],
                              "admission": tiered_admission,
                              "expert": tiered_expert, "correction_reset": correction_reset})
        rows.append(row)
        _write_jsonl(output, "fit/trajectories.jsonl", rows, meter=meter)
        last_event, last_rows = event, row_observations
    _require(last_event is not None and last_rows is not None, "fit", "no fit events executed")
    _restore_port_output(path.adapter, learning_port_state)
    influence_query, _, _ = _fit_query(path, backend, last_event, observations, ledger, tiered=tiered)
    influence = _null_comparison(path, influence_query)
    ledger.influence_forwards(int(influence_query.input_ids.shape[1]))
    trained_port_state = learning_port_state
    tiered_expert = _tiered_write_expert(tiered, path, publish=True, lineage=fit_lineage)
    tiered_state = _tiered_summary(tiered)
    tiered.close()
    final_receipt = _checkpoint(path, output, "fit/final/state", run_contract, ledger, meter)
    checkpoints.append(final_receipt)
    _retire_oldest(output, checkpoints, evicted, ledger)
    trained_plastic = copy.deepcopy(path.experiment.controller.checkpoint_payload())
    publication.write_torch(output, "fit/pre_fit_plastic.pt", pre_plastic,
                            meter=meter, reuse_identical=True)
    publication.write_torch(output, "fit/trained_plastic.pt", trained_plastic,
                            meter=meter, reuse_identical=True)
    _restore_port_output(nonlearning_path.adapter, initial_port_parameters)
    publication.write_torch(output, "fit/nonlearning_dynamic.pt", _dynamic_payload(nonlearning_path),
                            meter=meter, reuse_identical=True)
    _restore_port_output(replay_path.adapter, initial_port_parameters)
    publication.write_torch(output, "fit/replay_dynamic.pt", _dynamic_payload(replay_path),
                            meter=meter, reuse_identical=True)
    _write_json(output, "fit/checkpoint_index.json", {
        "writes": checkpoints + evicted, "retained": checkpoints, "evicted": evicted,
        "retention": "final plus three most recent event states; evicted receipts preserved",
    }, meter=meter)
    _write_json(output, "fit/tiered_receipts.json", {
        "events": tiered_events, "expert": tiered_expert, "state": tiered_state,
        "storage_outside_neural_boundaries": True}, meter=meter)
    fit_receipt = {
        "format": FIT_RECEIPT_FORMAT, "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256, "prepared_manifest_sha256": dataset.manifest_sha256,
        "status": "fit_executed", "execution_complete": True, "backend": backend.label,
        "native_execution": bool(backend.native_admissible), "fit_events": len(rows),
        "optimizer_steps": ledger.counters["optimizer_steps"],
        "selected_P_optimizer_steps": ledger.counters["selected_P_optimizer_steps"],
        "response_port_optimizer_steps": ledger.counters["response_port_optimizer_steps"],
        "response_spine_optimizer_steps": 0, "local_mse_optimizer_steps": 0,
        "pre_fit_plastic": "fit/pre_fit_plastic.pt", "trained_plastic": "fit/trained_plastic.pt",
        "nonlearning_dynamic": "fit/nonlearning_dynamic.pt", "replay_dynamic": "fit/replay_dynamic.pt",
        "pre_fit_plastic_digest": _sha256(output / "fit/pre_fit_plastic.pt"),
        "trained_plastic_digest": _sha256(output / "fit/trained_plastic.pt"),
        "fit_checkpoint": "fit/final/state", "checkpoint_writes": ledger.counters["checkpoint_writes"],
        "retained_checkpoints": [row["relative"] for row in checkpoints],
        "evicted_checkpoints": [row["relative"] for row in evicted],
        "initial_null_parity": initial_parity, "post_fit_influence": influence,
        "pre_port_state": initial_port_parameters,
        "trained_port_state": trained_port_state,
        "correction_resets": correction_resets, "exact_unlearning_claimed": False,
        "tiered_root": "tiered", "tiered_expert": tiered_expert,
        "tiered_state": tiered_state,
        "tiered_receipts": "fit/tiered_receipts.json",
        "run_contract": run_contract, "written_at_utc": _utc(),
    }
    _write_json(output, "fit/fit_receipt.json", fit_receipt, meter=meter)
    return FitState(path=path, nonlearning_path=nonlearning_path, replay_path=replay_path,
                    pre_plastic=pre_plastic, trained_plastic=trained_plastic,
                    pre_port_state=initial_port_parameters,
                    trained_port_state=trained_port_state,
                    receipt=fit_receipt, run_contract=run_contract)


def _load_learning_path(protocol: Protocol, dataset: PreparedDataset, backend: Backend,
                        output: Path) -> tuple[SparkTextMemoryPath, Mapping[str, Any], Mapping[str, Any]]:
    receipt = _read_json(output / "fit/fit_receipt.json", "fit_receipt")
    dataset.open_development(receipt)
    path = _new_path(protocol, backend.adapter)
    run_contract = _run_contract(protocol, dataset, backend)
    path.restore(output, str(receipt["fit_checkpoint"]), run_contract=run_contract)
    pre = publication.load_torch(output, str(receipt["pre_fit_plastic"]))
    trained = publication.load_torch(output, str(receipt["trained_plastic"]))
    return path, pre, trained


def load_fit_state(protocol: Protocol, dataset: PreparedDataset, backend: Backend,
                   output: Path) -> FitState:
    path, pre, trained = _load_learning_path(protocol, dataset, backend, output)
    trained_port_state = _port_output_snapshot(path)
    receipt = _read_json(output / "fit/fit_receipt.json", "fit_receipt")
    nonlearning = _new_path(protocol, backend.adapter)
    replay = _new_path(protocol, backend.adapter)
    _apply_dynamic(nonlearning, publication.load_torch(output, str(receipt["nonlearning_dynamic"])))
    pre_port_state = _port_output_snapshot(nonlearning)
    _apply_dynamic(replay, publication.load_torch(output, str(receipt["replay_dynamic"])))
    return FitState(path=path, nonlearning_path=nonlearning, replay_path=replay,
                    pre_plastic=pre, trained_plastic=trained, receipt=receipt,
                    pre_port_state=pre_port_state,
                    trained_port_state=trained_port_state,
                    run_contract=_run_contract(protocol, dataset, backend))


def _development_base(protocol: Protocol, source: SparkTextMemoryPath,
                      dataset: PreparedDataset) -> SparkTextMemoryPath:
    path = _clone_path(protocol, source)
    _observe_rows(path, [row for row in dataset.observations() if row.get("role") == "development"])
    _add_procedures(path, dataset.procedures())
    _apply_task_rows(path, dataset.tasks())
    return path


def _shuffle_evidence(path: SparkTextMemoryPath) -> None:
    records = list(path.experiment.evidence.active)
    if len(records) < 2:
        return
    payloads = [record.payload.clone() for record in records]
    path.experiment.evidence.clear()
    for index, record in enumerate(records):
        source_index = (index + 1) % len(records)
        lineage = copy.deepcopy(dict(record.source_lineage))
        lineage["shuffled_from"] = records[source_index].record_id
        lineage["shuffle_policy"] = "v2_deterministic_rotation_same_slots_bytes_without_gold_ids"
        shuffled = EvidenceRecord(
            record_id=record.record_id, scope=record.scope, payload=payloads[source_index],
            source_epoch=record.source_epoch, available_at=record.available_at,
            valid_from=record.valid_from, producer_revision=record.producer_revision,
            representation_version=record.representation_version, codec_version=record.codec_version,
            payload_type=record.payload_type, source_lineage=lineage,
            valid_until=record.valid_until, supersedes=record.supersedes)
        path.experiment.observe(shuffled, reconcile=record.supersedes is not None)
    path.queries.invalidate()


def build_arm_views(protocol: Protocol, fit: FitState,
                    dataset: PreparedDataset,
                    tiered: TieredMemoryStore | None = None) -> dict[str, ArmView]:
    _restore_port_output(fit.path.adapter, fit.trained_port_state)
    trained_base = _development_base(protocol, fit.path, dataset)
    _restore_port_output(fit.nonlearning_path.adapter, fit.pre_port_state)
    pre_base = _development_base(protocol, fit.nonlearning_path, dataset)
    _restore_port_output(fit.replay_path.adapter, fit.pre_port_state)
    replay_base = _development_base(protocol, fit.replay_path, dataset)
    views: dict[str, ArmView] = {}
    memory_contract = protocol.config["memory_contract"]
    for arm in ARMS:
        if arm in {"E_plus_P", "source_free_P_only", "shuffled_memory"}:
            path = _clone_path(protocol, trained_base)
            controller = "trained"
        elif arm == "matched_replay":
            path = _clone_path(protocol, replay_base)
            controller = "replay_only_no_new_update"
        else:
            path = _clone_path(protocol, pre_base)
            controller = "pre_teaching"
        mode = {
            "full_eligible_context": "E", "selective_exact_and_procedure": "E",
            "spine_E_only": "E", "reader_trained_E_only": "E", "source_free_P_only": "P", "no_update": "P", "reset_P": "E_plus_P", "E_plus_P": "E_plus_P",
            "shuffled_memory": "E_plus_P", "inapplicable_procedure": "E", "matched_replay": "E_plus_P",
        }[arm]
        port_state = fit.pre_port_state
        reader_role = "pre_reader"
        frozen_trained_reader = False
        untouched_baseline = arm == "full_eligible_context"
        if arm in {"reader_trained_E_only", "source_free_P_only", "E_plus_P",
                   "no_update", "reset_P", "shuffled_memory"}:
            port_state = fit.trained_port_state
            reader_role = "trained_reader"
            frozen_trained_reader = arm in {"reader_trained_E_only", "no_update", "reset_P"}
            if frozen_trained_reader:
                reader_role = "trained_reader_frozen_comparison"
        if arm == "no_update" or arm == "reset_P":
            controller = "pre_teaching_P_frozen_trained_reader"
        elif arm == "reader_trained_E_only":
            controller = "trained_reader_frozen_comparison"
        elif arm == "full_eligible_context":
            controller = "untouched_original_backbone"
        if arm in {"source_free_P_only", "no_update"}:
            path.clear_for_p_only()
        wrong = None
        if arm == "shuffled_memory":
            _shuffle_evidence(path)
            with torch.no_grad():
                for parameter in path.experiment.bank.expert(0).parameters():
                    parameter.mul_(-1)
            path.experiment.invalidate(reason="v2_wrong_memory_control")
            path.queries.invalidate()
            wrong = "sign_flip_of_selected_P_plus_deterministic_E_payload_rotation_not_an_update"
        views[arm] = ArmView(
            arm_id=arm, path=path, mode=mode, source_prompt_removed=arm != "full_eligible_context",
            oracle=arm == "selective_exact_and_procedure", inapplicable=arm == "inapplicable_procedure",
            controller_source=controller,
            evidence_source=("source_free" if arm in {"source_free_P_only", "no_update"}
                             else "shuffled" if arm == "shuffled_memory" else "eligible"),
            wrong_p_transform=wrong,
            port_state=port_state, untouched_baseline=untouched_baseline,
            frozen_trained_reader=frozen_trained_reader, reader_role=reader_role,
            full_context_records_max=int(memory_contract.get("full_context_records_max", 16)),
            full_context_value_slots_max=int(memory_contract.get("full_context_value_slots_max", 64)),
            tiered_store=tiered)
    return views


def _readout_for_view(view: ArmView, case: Mapping[str, Any]) -> ScopedMemoryReadout | None:
    if view.inapplicable:
        return None
    key = str(case.get("case_id") or case.get("request_id"))
    if key not in view.readout_results:
        _require(view.tiered_store is not None, "tiered_readout",
                 "evidence-bearing arm has no authoritative tiered store")
        view.readout_results[key] = _tiered_readout(
            view.tiered_store, str(case["question"]), int(case.get("query_time", 10)),
            full_context=view.arm_id == "full_eligible_context",
            full_context_records_max=view.full_context_records_max,
            full_context_value_slots_max=view.full_context_value_slots_max,
            allow_empty=view.oracle or view.untouched_baseline)
    return view.readout_results[key]


def _procedure_hint(procedure: Any) -> str:
    return (f"Procedure {procedure.procedure_id}: applicability "
            f"{'; '.join(procedure.applicability)}. Prerequisites "
            f"{'; '.join(procedure.prerequisites) or 'none'}. Exceptions "
            f"{'; '.join(procedure.exceptions) or 'none'}. Verified outcome: {procedure.outcome}.")


def _prompt_for(view: ArmView, case: Mapping[str, Any], dataset: PreparedDataset) -> tuple[str, dict[str, Any]]:
    question = str(case["question"])
    readout = None if view.mode == "P" else _readout_for_view(view, case)
    context = ""
    information: dict[str, Any] = {
        "source_text_included": False, "procedure_hint": None,
        "removed_source_text": view.source_prompt_removed,
        "oracle_selected": view.oracle, "answer_aware_record_ids_used": view.oracle,
        "wrong_p_transform": view.wrong_p_transform,
    }
    if view.arm_id == "full_eligible_context":
        _require(readout is not None, "tiered_readout",
                 "full eligible context requires authoritative scoped readout")
        context = "\n".join(text for text in readout.texts if text)
        information["source_text_included"] = bool(context)
        information["eligible_source_text_sha256"] = (
            hashlib.sha256(context.encode()).hexdigest() if context else None)
        information["tiered_readout_selected_record_ids"] = list(readout.record_ids)
        information["canonical_text_reference"] = True
        information["fallback_mode"] = readout.fallback_mode
        information["synthetic_value_controls"] = True
    elif view.oracle:
        procedure_id = str(case.get("procedure_id"))
        if procedure_id in view.path.procedures:
            context = _procedure_hint(view.path.procedures[procedure_id])
            information["procedure_hint"] = procedure_id
            information["oracle_label"] = "diagnostic_upper_bound_not_product_routing"
    elif view.inapplicable:
        for row in dataset.procedures():
            if row.get("world_id") != case.get("world_id") and str(row["procedure_id"]) in view.path.procedures:
                procedure = view.path.procedures[str(row["procedure_id"])]
                context = _procedure_hint(procedure)
                information["procedure_hint"] = procedure.procedure_id
                information["inapplicable_world"] = True
                break
    content = ((context + "\n\n") if context else "") + f"Question: {question}\nAnswer:"
    information["question_sha256"] = hashlib.sha256(question.encode()).hexdigest()
    return content, information


def _prepare_query(view: ArmView, backend: Backend, prompt: str, case: Mapping[str, Any],
                   ledger: WorkLedger):
    ids, mask = backend.encode_prompt(prompt, ledger.context_tokens)
    path = view.path
    _require(view.port_state is not None, "port_state", "arm reader state is missing")
    _restore_port_output(path.adapter, view.port_state)
    path.experiment.invalidate(reason="v2_no_cache_reuse")
    query_time = int(case.get("query_time", 10))
    times = tuple(query_time for _ in range(ids.shape[1]))
    readout = (None if view.mode == "P" or
               (view.untouched_baseline and view.tiered_store is None)
               else _readout_for_view(view, case))
    if readout is not None:
        _restore_readout_records(path, readout)
    selected_ids = () if readout is None else readout.record_ids
    if view.untouched_baseline:
        query = path.prepare_query(
            str(case["question"]), ids, mask, token_available_at=times,
            query_time=query_time, memory_source="E", enabled_branches=(),
            host_eligible_record_ids=(), semantic_top_k=1,
            selected_record_ids=())
    else:
        if view.mode == "P":
            arm = "p_only"
        elif view.mode == "E_plus_P":
            arm = "e_plus_p"
        elif view.arm_id == "selective_exact_and_procedure":
            arm = "selected_exact"
        elif view.inapplicable:
            arm = "absent_memory"
        else:
            arm = "full_eligible"
        target = None
        if arm == "selected_exact":
            support = [str(value) for value in case.get("support_record_ids", ())]
            target = support[0] if support else None
            selected_ids = (target,)
        query = prepare_arm_query(
            path, arm=arm, question=str(case["question"]), input_ids=ids,
            attention_mask=mask, target_record_id=target, query_time=query_time,
            token_available_at=times, selected_record_ids=selected_ids)
    if readout is not None:
        _charge_readout(ledger, readout)
        ledger.query_sensing(int(ids.shape[1]), cache_hits=readout.metrics.cache_hits,
                             cache_misses=readout.metrics.cache_misses)
    else:
        ledger.query_sensing(int(ids.shape[1]))
    ledger.retrieval()
    details = {"query_id": query.query_id, "memory_source": query.memory_source,
               "tiered_readout": None if readout is None else readout.document(),
               "readout_source": "authoritative_tiered_store",
               "selected_record_ids": list(query.selected_record_ids),
               "eligible_record_ids": list(query.eligible_record_ids),
               "semantic_selection": query.semantic_selection,
               "evidence_revision": query.evidence_revision,
               "plastic_revision": query.plastic_revision,
               "context_charges": dict(query.context.charges),
               "source_prompt_removed": view.source_prompt_removed,
               "enabled_spine_branches": list(query.context.enabled_branches or ()),
               "untouched_baseline": view.untouched_baseline,
               "frozen_trained_reader": view.frozen_trained_reader,
               "reader_role": view.reader_role,
    }
    return query, details


def _generate(view: ArmView, backend: Backend, prompt: str, case: Mapping[str, Any],
              ledger: WorkLedger) -> tuple[dict[str, Any], dict[str, Any], Any]:
    query, details = _prepare_query(view, backend, prompt, case, ledger)
    ledger.charge("generation_requests")
    raw = greedy_generate(backend, query, max_new_tokens=ledger.max_new_tokens, ledger=ledger)
    return raw, details, query


def _learning_metrics(rows: Sequence[Mapping[str, Any]], fit_receipt: Mapping[str, Any],
                      backend: Backend) -> dict[str, Any]:
    by_arm = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm_id"] == arm]
        by_arm[arm] = {
            "cases": len(selected),
            "accuracy": (sum(bool(row["score"]["matched"]) for row in selected) / len(selected)
                         if selected else None),
            "unknown_abstention": (
                sum(bool(row["score"]["matched"]) for row in selected
                    if row["expected"] == "unknown") /
                max(1, sum(row["expected"] == "unknown" for row in selected))),
        }
    def accuracy(arm: str, predicate) -> float | None:
        selected = [row for row in rows if row["arm_id"] == arm and predicate(row)]
        return (sum(bool(row["score"]["matched"]) for row in selected) / len(selected)
                if selected else None)
    new_transfer = accuracy("E_plus_P", lambda row: "new_information" in row["streams"]
                            or "new_procedure_instances" in row["streams"])
    nonlearning_transfer = accuracy("no_update", lambda row: "new_information" in row["streams"]
                                    or "new_procedure_instances" in row["streams"])
    retention_trained = accuracy("E_plus_P", lambda row: row["older_rule_retention"])
    retention_exact = accuracy("selective_exact_and_procedure", lambda row: row["older_rule_retention"])
    new_transfer_reset = accuracy("reset_P", lambda row: "new_information" in row["streams"]
                                  or "new_procedure_instances" in row["streams"])
    retention_reset = accuracy("reset_P", lambda row: row["older_rule_retention"])
    def delta(left, right):
        return None if left is None or right is None else left - right
    return {
        "by_arm": by_arm,
        "full_context_competence": by_arm["full_eligible_context"]["accuracy"],
        "new_skill_transfer_E_plus_P": new_transfer,
        "new_skill_transfer_no_update": nonlearning_transfer,
        "new_skill_absolute_delta": (None if new_transfer is None or nonlearning_transfer is None
                                     else new_transfer - nonlearning_transfer),
        "new_skill_transfer_reset_P": new_transfer_reset,
        "P_learning_new_skill_delta": delta(new_transfer, new_transfer_reset),
        "reader_learning": {
            "baseline_arm": "spine_E_only", "comparison_arm": "reader_trained_E_only",
            "accuracy_delta": delta(by_arm["reader_trained_E_only"]["accuracy"],
                                    by_arm["spine_E_only"]["accuracy"]),
            "comparison": "same E access; trained port reader versus pre-reader",
        },
        "P_learning": {
            "source_free_delta": delta(by_arm["source_free_P_only"]["accuracy"],
                                       by_arm["no_update"]["accuracy"]),
            "with_evidence_delta": delta(by_arm["E_plus_P"]["accuracy"],
                                         by_arm["reset_P"]["accuracy"]),
            "comparison": "same frozen trained reader; learned P versus reset/no-update P",
        },
        "retrieval_benefit": {
            "accuracy_delta": delta(by_arm["E_plus_P"]["accuracy"],
                                    by_arm["source_free_P_only"]["accuracy"]),
            "comparison": "same trained reader and P; eligible evidence added",
        },
        "older_rule_retention_reset_P": retention_reset,
        "P_learning_retention_drop": delta(retention_reset, retention_trained),
        "matched_replay": {
            "accuracy": by_arm["matched_replay"]["accuracy"],
            "replay_training": False,
            "label": "replay_compute_without_new_update",
        },
        "control_semantics": {
            "untouched_baseline": "full_eligible_context",
            "frozen_trained_reader_comparisons": ["reader_trained_E_only", "no_update", "reset_P"],
            "new_skill_absolute_delta_is_not_a_causal_P_claim": True,
        },
        "older_rule_retention_E_plus_P": retention_trained,
        "older_rule_retention_exact_baseline": retention_exact,
        "older_rule_retention_drop": (None if retention_trained is None or retention_exact is None
                                      else retention_exact - retention_trained),
        "fit_optimizer_steps": fit_receipt["optimizer_steps"],
        "fit_selected_P_steps": fit_receipt["selected_P_optimizer_steps"],
        "fit_response_port_steps": fit_receipt["response_port_optimizer_steps"],
        "useful_learning": "inconclusive_engineering_fixture" if backend.label == CPU_LABEL else "requires_full_context_and_policy_review",
        "sealed_evaluation": "closed",
    }


def _latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    samples = sorted(
        int(row["query"]["end_to_end_ns"]) for row in rows
        if row.get("execution_complete") and isinstance(row.get("query"), Mapping)
        and "end_to_end_ns" in row["query"])
    if not samples:
        return {"count": 0, "wall_clock": True, "p50_ns": None, "p95_ns": None,
                "max_ns": None, "scope": "prompt_readout_generation_and_independent_scoring"}
    return {
        "count": len(samples), "wall_clock": True,
        "p50_ns": samples[(50 * len(samples) + 99) // 100 - 1],
        "p95_ns": samples[(95 * len(samples) + 99) // 100 - 1],
        "max_ns": samples[-1],
        "scope": "prompt_readout_generation_and_independent_scoring",
    }

def run_development(protocol: Protocol, dataset: PreparedDataset, backend: Backend,
                    output: Path, ledger: WorkLedger, meter: publication.LiveByteMeter,
                    fit: FitState) -> dict[str, Any]:
    dataset.open_development(fit.receipt)
    tiered = _open_tiered(protocol, output)
    views = build_arm_views(protocol, fit, dataset, tiered=tiered)
    development_dir = output / "development"
    development_dir.mkdir(parents=True, exist_ok=True)
    tiered_admission = _tiered_admit_rows(
        tiered, [row for row in dataset.observations() if row.get("role") == "development"], fit.path.producer)
    _tiered_sync_procedures(tiered, dataset.procedures())
    tiered_state = _tiered_summary(tiered)
    _write_json(output, "development/tiered_receipts.json", {
        "admission": tiered_admission, "state": tiered_state,
        "storage_outside_neural_boundaries": True}, meter=meter)
    rows: list[dict[str, Any]] = []
    for case in dataset.development_cases():
        for arm in ARMS:
            view = views[arm]
            ledger.development_case(arm)
            case_started = time.perf_counter_ns()
            prompt, information = _prompt_for(view, case, dataset)
            try:
                raw, query_details, query = _generate(view, backend, prompt, case, ledger)
                score = score_answer(str(case["expected"]), raw["decoded"])
                ledger.charge("verification_calls")
                row = {"stage": "development", "execution_complete": True, "arm_id": arm,
                       "case_id": case["case_id"], "world_id": case["world_id"], "kind": case["kind"],
                       "streams": case["streams"], "expected": case["expected"],
                       "older_rule_retention": case["older_rule_retention"], "answer": raw["decoded"],
                       "raw": raw, "information": information, "query": query_details, "score": score}
                if arm == "source_free_P_only":
                    row["p_only_audit"] = audit_p_only_claim(
                        view.path, query,
                        answer_text=str(case["expected"]),
                        answer_token_ids=backend.encode_text(str(case["expected"])), labels=None)
                query_details["end_to_end_ns"] = time.perf_counter_ns() - case_started
                rows.append(row)
                _write_jsonl(output, "development/trajectories.jsonl", rows, meter=meter)
            except Exception as error:
                partial = getattr(error, "partial", {})
                failure = {"stage": "development", "execution_complete": False, "arm_id": arm,
                           "case_id": case["case_id"], "error": str(error), "partial": partial,
                           "traceback": traceback.format_exc()}
                rows.append(failure)
                _write_jsonl(output, "development/trajectories.jsonl", rows, meter=meter)
                _write_json(output, "failure.json", failure, meter=meter)
                raise
    parity_rows = []
    null_view = _clone_path(protocol, fit.nonlearning_path)
    null_view.clear_for_p_only()
    null_view = ArmView("parity_null", null_view, "none", True,
                        controller_source="untouched_original_backbone",
                        port_state=fit.pre_port_state, untouched_baseline=True,
                        reader_role="untouched_original_backbone")
    trained_view = views["source_free_P_only"]
    for request in dataset.parity_requests():
        view = null_view if request["memory_source"] == "none" else trained_view
        request_started = time.perf_counter_ns()
        prompt = f"Question: {request['question']}\nAnswer:"
        ledger.charge("parity_requests")
        raw, details, _query = _generate(view, backend, prompt, request, ledger)
        ledger.charge("verification_calls")
        details["end_to_end_ns"] = time.perf_counter_ns() - request_started
        parity_rows.append({"stage": "development", "parity": True, "execution_complete": True,
                            "request_id": request["request_id"], "kind": request["kind"],
                            "answer": raw["decoded"], "raw": raw, "query": details,
                            "memory_source": request["memory_source"]})
    _write_jsonl(output, "development/parity.jsonl", parity_rows, meter=meter)
    report = {
        "format": DEVELOPMENT_FORMAT, "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256, "prepared_manifest_sha256": dataset.manifest_sha256,
        "backend": backend.label, "native_execution": bool(backend.native_admissible),
        "tiered_state": tiered_state, "tiered_receipts": "development/tiered_receipts.json",
        "readout_cache": readout_cache_state(tiered),
        "arms": list(ARMS), "development_cases_per_arm": 12,
        "development_cases_total": len(rows), "parity_requests": len(parity_rows),
        "learning_evidence": _learning_metrics(rows, fit.receipt, backend),
        "latency": _latency_summary(rows),
        "raw_trajectories": "development/trajectories.jsonl", "raw_parity": "development/parity.jsonl",
        "derived_caches_required": False, "sealed_final_opened": False,
        "rows_sha256": _sha256(development_dir / "trajectories.jsonl"),
        "parity_sha256": _sha256(development_dir / "parity.jsonl"), "written_at_utc": _utc(),
    }
    _write_json(output, "development/development_report.json", report, meter=meter)
    _write_json(output, "development/readout_receipts.json", {
        "cache": readout_cache_state(tiered),
        "storage_outside_neural_boundaries": True}, meter=meter)
    tiered.close()
    return report


def run_restart(protocol: Protocol, dataset: PreparedDataset, backend: Backend,
                output: Path, ledger: WorkLedger, meter: publication.LiveByteMeter) -> dict[str, Any]:
    path, _, _ = _load_learning_path(protocol, dataset, backend, output)
    ledger.charge("reloads")
    ledger.charge("restarts")
    receipt = _read_json(output / "fit/fit_receipt.json", "fit_receipt")
    tiered = _open_tiered(protocol, output)
    loaded_expert = tiered.load_expert("expert-0")
    _require(loaded_expert.outcome.ok and loaded_expert.state is not None,
             "tiered_restart", "published expert state is unavailable")
    tiered_expert = loaded_expert.state.document()
    expected_expert = receipt["tiered_expert"]
    for key in ("weights_sha256", "moments_sha256", "clock"):
        _require(tiered_expert[key] == expected_expert[key],
                 "tiered_restart", f"published expert {key} differs after restart")
    tiered_state = _tiered_summary(tiered)
    tiered.close()
    inventory_before = path.state_inventory()
    p_before = (list(path.experiment.bank.expert_steps),
                path.experiment.controller.clock.eligible_events,
                path.experiment.controller.revision)
    path.clear_for_p_only()
    inventory_after = path.state_inventory()
    p_after = (list(path.experiment.bank.expert_steps),
               path.experiment.controller.clock.eligible_events,
               path.experiment.controller.revision)
    _require(p_before == p_after, "restart", "P state changed during source removal")
    _require(not path.texts and not path.procedures and not path.tasks and not path.explanations
             and path.experiment.evidence.pending == 0, "restart", "source-bearing state remains")
    request = dataset.parity_requests(restart=True)[0]
    paired = None
    parity_path = output / "development/parity.jsonl"
    if parity_path.is_file():
        for row in _read_jsonl(parity_path, "parity"):
            if row.get("request_id") == request.get("paired_request_id"):
                paired = row
                break
    _require(paired is not None, "restart_parity", "development parity output is required before restart")
    view = ArmView("restart_p_only", path, "P", True,
                   controller_source="trained_restored",
                   port_state=_port_output_snapshot(path), reader_role="trained_restored")
    prompt = f"Question: {request['question']}\nAnswer:"
    ledger.charge("restart_parity_requests")
    raw, details, query = _generate(view, backend, prompt, request, ledger)
    ledger.charge("verification_calls")
    exact = list(raw["token_ids"]) == list(paired["raw"]["token_ids"])
    audit = audit_p_only_claim(path, query, labels=None)
    report = {
        "format": RESTART_FORMAT, "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256, "restart_process": "fresh_process",
        "execution_complete": True, "backend": backend.label,
        "native_execution": bool(backend.native_admissible), "request_id": request["request_id"],
        "source_removed": True, "P_state_preserved": p_before == p_after,
        "tiered_reloaded": True, "tiered_expert": tiered_expert,
        "tiered_state": tiered_state,
        "inventory_before_source_removal": inventory_before, "inventory_after_source_removal": inventory_after,
        "parity": {"exact_token_ids": exact, "decoded_equal": raw["decoded"] == paired["answer"],
                   "paired_request_id": request["paired_request_id"], "before": paired["raw"],
                   "after": raw},
        "p_only_audit": audit, "query": details, "derived_caches_restored": False,
        "work": ledger.report(), "written_at_utc": _utc(),
    }
    _write_json(output, "restart/restart_report.json", report, meter=meter)
    return report


def _restart_command(protocol: Protocol, output: Path, backend: Backend,
                     ledger: WorkLedger, prepared_root: Path,
                     authorization_path: str | Path | None) -> list[str]:
    command = [sys.executable, str(protocol.root / "scripts/run_spark_persistent_learner_v2.py"),
               "--stage", "restart", "--backend", backend.label, "--device", backend.device,
               "--output", str(output), "--config", str(protocol.path),
               "--prepared", str(prepared_root)]
    if backend.label == CPU_LABEL:
        command.extend(["--fixture-context-tokens", str(ledger.context_tokens),
                        "--fixture-max-new-tokens", str(ledger.max_new_tokens)])
    if authorization_path is not None:
        command.extend(["--authorization", str(authorization_path)])
    return command


def _merge_restart_report(ledger: WorkLedger, report: Mapping[str, Any]) -> None:
    ledger.merge_restart(report["work"]["counters"])


def _write_storage_receipts(output: Path,
                            meter: publication.LiveByteMeter) -> dict[str, Any]:
    rows = []
    for candidate in sorted(output.rglob("*")):
        if not candidate.is_file() or candidate.name in {"run_report.json", "storage_artifacts.json"}:
            continue
        kind, magic, size = detect_file_type(candidate)
        relative = candidate.relative_to(output).as_posix()
        rows.append({
            "path": relative, "bytes": int(size), "sha256": _sha256(candidate),
            "detected_type": kind, "magic_prefix": magic,
            "extensionless": candidate.suffix == "",
            "deletion_authority": "explicit_retention_policy_required",
            "extensionless_archive_safe_to_delete": False,
        })
    report = {
        "format": "replica.spark_persistent_learner.artifact_receipts.v2",
        "files": rows, "file_count": len(rows), "bytes": sum(row["bytes"] for row in rows),
        "excluded_self_referential_files": ["run_report.json", "storage_artifacts.json"],
        "extensionless_archives_safe_to_delete": False,
        "base_model_weights_included": False,
    }
    _write_json(output, "storage_artifacts.json", report, meter=meter)
    return report


def _run_protocol_impl(protocol_path: str | Path, prepared_root: str | Path, output_root: str | Path, *,
                 backend_kind: str = NATIVE_LABEL, device: str = "mps", stage: str = "all",
                 authorization_path: str | Path | None = None,
                 fixture_context_tokens: int = 256, fixture_max_new_tokens: int = 4,
                 spine_sites: Sequence[int] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    protocol = Protocol.load(protocol_path)
    dataset = PreparedDataset.load(
        prepared_root, expected_sha256=protocol.config["dataset"].get("manifest_sha256"),
        protocol_sha256=protocol.sha256)
    output = Path(output_root).resolve()
    _require(stage in {"all", "fit", "development", "restart"}, "stage", "unknown stage")
    fresh = stage in {"all", "fit"}
    if fresh:
        _require(not output.exists(), "output", "fit output must be fresh")
    else:
        _require(output.is_dir(), "output", "resume output is missing")
    native = backend_kind == NATIVE_LABEL
    context = int(protocol.config["runtime"]["context_tokens_max"] if native else fixture_context_tokens)
    new_tokens = int(protocol.config["runtime"]["generation"]["max_new_tokens"] if native
                     else fixture_max_new_tokens)
    _require(context > 0 and new_tokens > 0, "budget", "context and generation bounds must be positive")
    sites = tuple(spine_sites or protocol.config["runtime"]["spine_sites"])
    ledger = WorkLedger(protocol, context_tokens=context, max_new_tokens=new_tokens)
    meter = publication.LiveByteMeter(
        output, label="spark_persistent_learner_v2",
        cap_bytes=int(protocol.schedule["artifact_bytes_max"]))
    try:
        backend = build_backend(
            backend_kind, protocol, dataset, device=device, output_root=output,
            spine_sites=sites, authorization_path=authorization_path,
            allow_existing_output=not fresh)
        ledger.charge("model_loads")
        ledger.charge("tokenizer_loads")
        if stage == "fit":
            fit = run_fit(protocol, dataset, backend, output, ledger, meter)
            report = {"format": REPORT_FORMAT, "stage": "fit", "status": "fit_executed",
                      "fit": fit.receipt, "work": ledger.report(),
                      "source": _source_reference(protocol, backend)}
        elif stage == "development":
            fit = load_fit_state(protocol, dataset, backend, output)
            report = run_development(protocol, dataset, backend, output, ledger, meter, fit)
        elif stage == "restart":
            report = run_restart(protocol, dataset, backend, output, ledger, meter)
        else:
            fit = run_fit(protocol, dataset, backend, output, ledger, meter)
            development = run_development(protocol, dataset, backend, output, ledger, meter, fit)
            command = _restart_command(protocol, output, backend, ledger, dataset.root, authorization_path)
            fit_receipt = fit.receipt
            source_reference = _source_reference(protocol, backend)
            backend_label = backend.label
            backend_native = bool(backend.native_admissible)
            # Only the restart child may hold a model after this boundary.
            fit = None
            backend.close()
            backend = None
            gc.collect()
            restart_dir = output / "restart"
            restart_dir.mkdir(parents=True, exist_ok=True)
            _write_json(output, "restart/child_command.json", {"command": command, "written_at_utc": _utc()}, meter=meter)
            child = subprocess.run(command, cwd=protocol.root, text=True, capture_output=True,
                                   timeout=int(protocol.schedule["wall_seconds_max"]),
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                        "PYTORCH_ENABLE_MPS_FALLBACK": "0", "HF_HUB_OFFLINE": "1",
                                       "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                                       "REPLICA_NATIVE_SUPERVISED_CHILD": "1"})
            (restart_dir / "child.stdout").write_text(child.stdout, encoding="utf-8")
            (restart_dir / "child.stderr").write_text(child.stderr, encoding="utf-8")
            _require(child.returncode == 0, "restart", f"fresh-process restart failed: {child.stderr[-2000:]}")
            restart = _read_json(restart_dir / "restart_report.json", "restart")
            _merge_restart_report(ledger, restart)
            ledger.finish()
            report = {
                "format": REPORT_FORMAT, "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol.sha256, "prepared_manifest_sha256": dataset.manifest_sha256,
                "status": "complete", "stage": "all", "backend": backend_label,
                "native_execution": backend_native,
                "evidence_class": "native_official_fp8" if native else "cpu_engineering_fixture_not_native",
                "source": source_reference, "fit": fit_receipt,
                "development": development, "restart": restart,
                "engineering_acceptance": {
                    "initial_null_parity": fit_receipt["initial_null_parity"]["exact_null_parity"],
                    "post_fit_nonzero_influence": fit_receipt["post_fit_influence"]["nonzero_influence"],
                    "fresh_process_restart": restart["parity"]["exact_token_ids"],
                    "source_free_restart": restart["source_removed"] and restart["P_state_preserved"],
                    "tiered_p_reload": restart["tiered_reloaded"],
                    "p_only_audit": restart["p_only_audit"]["p_only_claim"],
                },
                "learning_claim": development["learning_evidence"]["useful_learning"],
                "sealed_final_opened": False, "work": ledger.report(),
                "artifact": meter.report(), "wall_seconds": time.monotonic() - started,
                "raw_paths": ["fit/trajectories.jsonl", "development/trajectories.jsonl",
                              "development/parity.jsonl", "restart/restart_report.json"],
            }
        report["storage_artifacts"] = _write_storage_receipts(output, meter)
        meter.observe()
        if stage == "all":
            for _ in range(3):
                ledger.set_measured("artifact_bytes", meter.peak_bytes)
                report["work"] = ledger.report()
                report["artifact"] = meter.report()
                _write_json(output, "run_report.json", report, meter=meter)
                meter.observe()
        else:
            ledger.set_measured("artifact_bytes", meter.peak_bytes)
            report["work"] = ledger.report()
            _write_json(output, "run_report.json", report, meter=meter)
            meter.observe()
        return report
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        failure = {"format": "replica.spark_persistent_learner.failure.v2",
                   "stage": stage, "backend": backend_kind, "error": str(error),
                   "error_type": type(error).__name__, "traceback": traceback.format_exc(),
                   "work": ledger.report(), "native_execution": False,
                   "preserved_partial_artifacts": True, "written_at_utc": _utc()}
        try:
            _write_json(output, "failure.json", failure)
        except Exception:
            pass
        raise


def run_protocol(protocol_path: str | Path, prepared_root: str | Path, output_root: str | Path, *,
                 backend_kind: str = NATIVE_LABEL, device: str = "mps", stage: str = "all",
                 authorization_path: str | Path | None = None,
                 fixture_context_tokens: int = 256, fixture_max_new_tokens: int = 4,
                 spine_sites: Sequence[int] | None = None) -> dict[str, Any]:
    if backend_kind != NATIVE_LABEL:
        return _run_protocol_impl(
            protocol_path, prepared_root, output_root, backend_kind=backend_kind,
            device=device, stage=stage, authorization_path=authorization_path,
            fixture_context_tokens=fixture_context_tokens,
            fixture_max_new_tokens=fixture_max_new_tokens, spine_sites=spine_sites)
    protocol = Protocol.load(protocol_path)
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise Refused("capacity_guard",
                      "native execution requires scripts/run_capacity_preflight.py")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise Refused("cpu_fallback", "PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise Refused("offline", f"{name}=1 is required")
    if os.environ.get("REPLICA_NATIVE_SUPERVISED_CHILD") == "1":
        return _run_protocol_impl(
            protocol_path, prepared_root, output_root, backend_kind=backend_kind,
            device=device, stage=stage, authorization_path=authorization_path,
            fixture_context_tokens=fixture_context_tokens,
            fixture_max_new_tokens=fixture_max_new_tokens, spine_sites=spine_sites)
    with exclusive_mps_worker(protocol.root):
        return _run_protocol_impl(
            protocol_path, prepared_root, output_root, backend_kind=backend_kind,
            device=device, stage=stage, authorization_path=authorization_path,
            fixture_context_tokens=fixture_context_tokens,
            fixture_max_new_tokens=fixture_max_new_tokens, spine_sites=spine_sites)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/spark_persistent_learner_protocol_v2.json"))
    parser.add_argument("--prepared", type=Path, default=Path("data/spark_persistent_learner_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=[NATIVE_LABEL, CPU_LABEL], default=NATIVE_LABEL)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--stage", choices=["all", "fit", "development", "restart"], default="all")
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--fixture-context-tokens", type=int, default=256)
    parser.add_argument("--fixture-max-new-tokens", type=int, default=4)
    parser.add_argument("--spine-sites", type=str, default=None,
                        help="comma-separated zero-based engineering/native reader sites")
    args = parser.parse_args(argv)
    sites = None if args.spine_sites is None else tuple(int(value) for value in args.spine_sites.split(","))
    try:
        report = run_protocol(
            args.config, args.prepared, args.output, backend_kind=args.backend,
            device=args.device, stage=args.stage, authorization_path=args.authorization,
            fixture_context_tokens=args.fixture_context_tokens,
            fixture_max_new_tokens=args.fixture_max_new_tokens, spine_sites=sites)
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        detail = error.as_dict() if isinstance(error, Refused) else {"check": "runner", "detail": str(error)}
        print(json.dumps({"status": "refused", **detail}, indent=2, sort_keys=True))
        return 2


__all__ = [
    "ARMS", "CPU_LABEL", "NATIVE_LABEL", "Protocol", "PreparedDataset", "WorkLedger",
    "FitState", "ArmView", "build_backend", "build_cpu_backend", "build_native_backend",
    "verify_native_authorization", "run_fit", "run_development", "run_restart",
    "load_fit_state", "run_protocol", "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
