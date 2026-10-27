"""One-clock selected-expert plastic baseline for MiniCPM5 text.

This module *composes* the reviewed NL0 owners instead of reimplementing them:

* ``ExpertBank`` / ``PlasticLearningController`` own the selected-expert
  transaction, optimizer moments, one teaching clock and event dedup.
* ``ObservationCoordinator`` owns the fast associative transition; exactly one
  sealed chunk performs at most one transition and every chunk here is one
  teaching event.
* ``RouteControl`` is fixed (or the declared shuffled control).  The route key
  is derived from prompt tokens only; world/split labels never enter routing.
* ``PersonalityIdentity`` is stable and never written by learning.

The native reader sites 13/27 are detached, masked-mean pooled and mapped by a
frozen projection ``R``.  The same ``R`` builds verified-response targets and
the fixed read decoder.  A teaching transaction is half-sum squared error in
that projected space.  Initial plastic reads are exactly null because every
expert's final ``down`` matrix starts at zero; a committed transaction can then change the
read.  No meta-gradient, replay, external-gradient transaction or second
controller exists here.

The module is deliberately free of MiniCPM imports.  ``p_only_export`` carries
weights plus immutable binding metadata only: no moments, dedup ledger, clock,
coordinator, observations or source text.  A fresh process can therefore load
``P`` and answer from the fixed read path while proving that no source-bearing
state was constructed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .architecture.contracts import (DifferentiablePlasticRead, PlasticRead,
                                     PlasticSpec, TaskGraph, fingerprint)
from .architecture.plastic import fixed_projection, projection_digest
from .nested_learning import (ChunkOutcome, DuplicateObservation, ExpertBank,
                              NestedLearningError, Observation,
                              ObservationCoordinator,
                              PersonalityIdentity, PlasticLearningController,
                              RouteControl, TransactionOutcome, _float_tensor)

__all__ = [
    "NestedTextError",
    "NestedTextLearner",
    "P_ONLY_FORMAT",
    "P_ONLY_KEYS",
    "PlasticReadObserver",
    "TextAccessOrder",
    "TextEventOutcome",
    "TextProtocol",
    "TextTeachingEvent",
    "evaluate_dev_gate",
    "TEXT_CHECKPOINT_FORMAT",
    "initialize_null_plastic",
    "plastic_slot",
    "prompt_route_key",
    "validate_p_only_payload",
]

TEXT_CHECKPOINT_FORMAT = "replica.nested_text.checkpoint.v1"
P_ONLY_FORMAT = "replica.nested_learning.text_p_only.v1"
P_ONLY_KEYS = frozenset({
    "format", "scope", "theta", "protocol_sha256", "bank", "route",
    "feature_producer", "read_path", "expert_steps", "parameters"})
P_ONLY_FORBIDDEN = frozenset({
    "optimizer", "dedup", "clock", "revision", "coordinator", "observations",
    "moments", "sources", "kv", "source_text"})
_ALLOWED_ROLES = frozenset({"teaching", "development", "sealed_final"})
_ALLOWED_QUERY_KINDS = frozenset({"teaching", "future_query"})
_ALLOWED_SOURCES = frozenset({"observation", "feedback", "correction"})
_ALLOWED_FEEDBACK = frozenset({"answer", "correction", "observed_outcome"})
_PARAMETER_NAMES = ("gate", "up", "down")
_ACCESS_ORDER = ("teaching", "development", "sealed_final")


class NestedTextError(NestedLearningError):
    """A nested-text protocol, binding or lifecycle contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NestedTextError(message)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value in ("", None):
        return ""
    _require(isinstance(value, str) and len(value) == 64
             and all(char in "0123456789abcdef" for char in value),
             f"{name} must be a 64-character lowercase SHA-256")
    return value


def _tensor_sha(tensor: Tensor) -> str:
    values = tensor.detach().to("cpu", torch.float32).contiguous()
    return _sha256_bytes(values.numpy().tobytes())


def _finite_number(value: Any, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and torch.isfinite(torch.tensor(float(value))), f"{name} must be finite")
    return float(value)


# ---------------------------------------------------------------------------
# Protocol and events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextProtocol:
    """Immutable binding of world, model release, bank shape and read path.

    Empty binding digests are permitted only for isolated CPU tests; a native
    checkpoint or P-only state with an empty binding is rejected by the
    comparison helpers that require the full release identity.
    """
    world_uid: str
    scope: str
    expert_count: int = 16
    width: int = 256
    hidden_width: int = 1024
    hidden_size: int = 2048
    feature_sites: tuple[int, ...] = (13, 27)
    target_projection_seed: int = 0
    target_projection_sha256: str = ""
    read_gain: float = 0.1
    read_path_sha256: str = ""
    route_mode: str = "fixed"
    route_seed: int = 0
    theta_revision: str = ""
    theta_lock_sha256: str = ""
    parameter_stamp_sha256: str = ""
    tokenizer_sha256: str = ""
    p_init: str = "zero_read_down"
    steps_per_event: int = 1
    total_step_cap: int = 4
    clock_period: int = 16
    gradient_norm_max: float = 1.0
    protocol_sha256: str = ""
    _canonical: str = field(init=False, repr=False, compare=False, default="")

    def __post_init__(self) -> None:
        _require(isinstance(self.world_uid, str) and self.world_uid,
                 "world_uid must be nonempty")
        _require(isinstance(self.scope, str) and self.scope, "scope must be nonempty")
        object.__setattr__(self, "feature_sites", tuple(self.feature_sites))
        for name in ("expert_count", "width", "hidden_width", "hidden_size"):
            value = getattr(self, name)
            _require(type(value) is int and value > 0,
                     f"{name} must be a positive integer")
        sites = self.feature_sites
        _require(sites and len(set(sites)) == len(sites)
                 and all(type(site) is int and site >= 0 for site in sites),
                 "feature_sites must be distinct nonnegative integers")
        _require(type(self.target_projection_seed) is int,
                 "target_projection_seed must be an integer")
        _digest(self.target_projection_sha256, "target_projection_sha256")
        _digest(self.read_path_sha256, "read_path_sha256")
        _finite_number(self.read_gain, "read_gain")
        _require(self.read_gain >= 0, "read_gain must be nonnegative")
        _require(self.route_mode in {"fixed", "shuffled"},
                 "route_mode must be 'fixed' or 'shuffled'")
        _require(type(self.route_seed) is int, "route_seed must be an integer")
        for name in ("theta_revision", "theta_lock_sha256", "parameter_stamp_sha256"):
            if name == "theta_revision":
                _require(isinstance(self.theta_revision, str),
                         "theta_revision must be a string")
            else:
                _digest(getattr(self, name), name, allow_empty=True)
        _digest(self.tokenizer_sha256, "tokenizer_sha256", allow_empty=True)
        _require(self.p_init == "zero_read_down",
                 "only the declared zero-read initialization is supported")
        _require(self.steps_per_event == 1,
                 "the one-clock baseline applies exactly one step per event")
        _require(type(self.total_step_cap) is int and 1 <= self.total_step_cap <= 4,
                 "total_step_cap must lie in [1, 4]")
        _require(type(self.clock_period) is int and self.clock_period > 0,
                 "clock_period must be positive")
        _finite_number(self.gradient_norm_max, "gradient_norm_max")
        _require(self.gradient_norm_max > 0, "gradient_norm_max must be positive")
        projection = self.target_projection()
        _require(projection_digest(projection) == self.target_projection_sha256,
                 "target projection digest differs from the declared seed/width")
        _require(self.read_path_sha256 == self.target_projection_sha256,
                 "the fixed read decoder must equal the target projection")
        canonical = self._canonical_payload()
        computed = _sha256_json(canonical)
        if self.protocol_sha256:
            _require(self.protocol_sha256 == computed,
                     "protocol_sha256 differs from the protocol body")
        else:
            object.__setattr__(self, "protocol_sha256", computed)

    def target_projection(self) -> Tensor:
        return fixed_projection(self.hidden_size, self.width,
                                self.target_projection_seed)

    def _canonical_payload(self) -> dict:
        return {
            "world_uid": self.world_uid, "scope": self.scope,
            "expert_count": self.expert_count, "width": self.width,
            "hidden_width": self.hidden_width, "hidden_size": self.hidden_size,
            "feature_sites": list(self.feature_sites),
            "target_projection_seed": self.target_projection_seed,
            "target_projection_sha256": self.target_projection_sha256,
            "read_gain": self.read_gain, "read_path_sha256": self.read_path_sha256,
            "route_mode": self.route_mode, "route_seed": self.route_seed,
            "theta_revision": self.theta_revision,
            "theta_lock_sha256": self.theta_lock_sha256,
            "parameter_stamp_sha256": self.parameter_stamp_sha256,
            "tokenizer_sha256": self.tokenizer_sha256, "p_init": self.p_init,
            "steps_per_event": self.steps_per_event,
            "total_step_cap": self.total_step_cap,
            "clock_period": self.clock_period,
            "gradient_norm_max": self.gradient_norm_max,
        }

    def to_dict(self) -> dict:
        value = self._canonical_payload()
        value["protocol_sha256"] = self.protocol_sha256
        return value

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TextProtocol":
        _require(isinstance(payload, Mapping), "protocol payload must be a mapping")
        allowed = set(cls.__dataclass_fields__) - {"_canonical"}
        unknown = set(payload) - allowed
        _require(not unknown, f"unknown protocol fields: {sorted(unknown)}")
        values = dict(payload)
        values["feature_sites"] = tuple(values.get("feature_sites", (13, 27)))
        return cls(**values)

    @property
    def plastic_spec(self) -> PlasticSpec:
        return PlasticSpec(experts=self.expert_count, width=self.width,
                           hidden_width=self.hidden_width,
                           sites=self.feature_sites, route_mode=self.route_mode,
                           route_seed=self.route_seed,
                           projection_seed=self.target_projection_seed,
                           read_gain=self.read_gain,
                           feature_version="sites_masked_mean_projection.v1",
                           p_init=self.p_init)


@dataclass(frozen=True)
class TextTeachingEvent:
    """One role-separated text event. Routing sees prompt tokens only."""
    event_id: str
    world_uid: str
    scope: str
    split_role: str
    teaching_or_query: str
    prompt_sha256: str
    response_sha256: str
    target_available_at: float
    causal_cutoff: float
    eligible: bool
    source: str
    verified: bool
    route_key: tuple[float, ...]
    concept_family: str = ""
    contamination_group: str = ""
    feedback_type: str = "answer"
    supersedes: str | None = None

    def __post_init__(self) -> None:
        for name in ("event_id", "world_uid", "scope"):
            _require(isinstance(getattr(self, name), str) and getattr(self, name),
                     f"{name} must be nonempty")
        _require(self.split_role in _ALLOWED_ROLES,
                 f"split_role must be one of {sorted(_ALLOWED_ROLES)}")
        _require(self.teaching_or_query in _ALLOWED_QUERY_KINDS,
                 "teaching_or_query must be teaching or future_query")
        _digest(self.prompt_sha256, "prompt_sha256")
        _digest(self.response_sha256, "response_sha256")
        _finite_number(self.target_available_at, "target_available_at")
        _finite_number(self.causal_cutoff, "causal_cutoff")
        _require(self.target_available_at <= self.causal_cutoff,
                 "target availability cannot follow the causal cutoff")
        _require(isinstance(self.eligible, bool) and isinstance(self.verified, bool),
                 "eligible and verified must be boolean")
        _require(self.source in _ALLOWED_SOURCES,
                 "source must be observation, feedback or correction")
        _require(self.feedback_type in _ALLOWED_FEEDBACK,
                 "feedback_type must be answer, correction or observed_outcome")
        object.__setattr__(self, "route_key",
                           tuple(_finite_number(item, "route_key")
                                 for item in self.route_key))
        _require(self.route_key, "route_key must be nonempty")
        if self.eligible:
            _require(self.verified, "an eligible event requires a verified target")
            _require(self.split_role == "teaching",
                     "only teaching-role events are eligible for updates")
            _require(self.teaching_or_query == "teaching",
                     "an eligible event must be a teaching event")
        if self.supersedes is not None:
            _require(isinstance(self.supersedes, str) and self.supersedes,
                     "supersedes must be a nonempty event id")

    def identity(self) -> dict:
        return {
            "event_id": self.event_id, "world_uid": self.world_uid,
            "scope": self.scope, "split_role": self.split_role,
            "teaching_or_query": self.teaching_or_query,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "target_available_at": self.target_available_at,
            "causal_cutoff": self.causal_cutoff, "eligible": self.eligible,
            "source": self.source, "verified": self.verified,
            "route_key": list(self.route_key),
            "concept_family": self.concept_family,
            "contamination_group": self.contamination_group,
            "feedback_type": self.feedback_type, "supersedes": self.supersedes,
        }

    def fingerprint(self) -> str:
        return _sha256_json(self.identity())


@dataclass(frozen=True)
class TextEventOutcome:
    """Durable outcome of one event: one transaction at most, one chunk at most."""
    status: str
    event_id: str
    expert_id: int
    route_mode: str
    feature_sha256: str
    target_sha256: str
    transaction: TransactionOutcome | None
    chunk: ChunkOutcome
    retried: bool = False

    def to_dict(self) -> dict:
        return {
            "status": self.status, "event_id": self.event_id,
            "expert_id": self.expert_id, "route_mode": self.route_mode,
            "feature_sha256": self.feature_sha256,
            "target_sha256": self.target_sha256,
            "retried": self.retried,
            "transaction": (None if self.transaction is None
                            else self.transaction.to_dict()),
            "chunk": self.chunk.to_dict(),
        }


# ---------------------------------------------------------------------------
# Access order / heldout discipline
# ---------------------------------------------------------------------------


@dataclass
class TextAccessOrder:
    """Monotonic teaching -> development -> sealed_final access gate.

    A failed development gate permanently blocks ``sealed_final`` in this
    process, and a heldout failure cannot be retried: the phase is recorded as
    failed and every later heldout access raises.
    """
    phase: str = _ACCESS_ORDER[0]
    dev_gate: bool | None = None
    heldout_opened: bool = False
    failed: str | None = None
    events: list[dict] = field(default_factory=list)

    def access(self, role: str) -> None:
        _require(role in _ACCESS_ORDER, f"unknown access role {role!r}")
        _require(not (role == "sealed_final" and self.failed),
                 f"heldout is sealed after failure: {self.failed}")
        _require(not (role == "sealed_final" and self.heldout_opened),
                 "the sealed heldout split is already open; no retry")
        _require(_ACCESS_ORDER.index(role) <= _ACCESS_ORDER.index(self.phase),
                 f"access {role!r} precedes phase {self.phase!r}")
        self.events.append({"access": role, "phase": self.phase})

    def advance(self, role: str) -> None:
        _require(role in _ACCESS_ORDER, f"unknown phase {role!r}")
        current = _ACCESS_ORDER.index(self.phase)
        _require(current + 1 < len(_ACCESS_ORDER)
                 and _ACCESS_ORDER[current + 1] == role,
                 f"cannot advance phase {self.phase!r} to {role!r}")
        self.phase = role
        self.events.append({"advanced_to": role})

    def record_dev_gate(self, report: Mapping[str, Any]) -> None:
        _require(self.phase in {"teaching", "development"},
                 "development gate must be recorded before heldout")
        if self.phase == "teaching":
            self.advance("development")
        self.dev_gate = bool(report.get("all_declared_gates_passed"))
        self.events.append({"role": "development", "dev_gate": self.dev_gate,
                            "report_sha256": _sha256_json(dict(report))})
        if not self.dev_gate:
            self.fail("development gate failed; sealed heldout remains closed")

    def open_heldout(self) -> None:
        _require(not self.heldout_opened,
                 "the sealed heldout split is already open; no retry")
        _require(self.phase == "development",
                 "heldout may only open after development")
        _require(self.dev_gate is True,
                 "heldout is blocked until the development gate passes")
        _require(not self.failed, "heldout is sealed after a recorded failure")
        self.advance("sealed_final")
        self.access("sealed_final")
        self.heldout_opened = True

    def fail(self, reason: str) -> None:
        _require(isinstance(reason, str) and reason, "failure reason required")
        if self.failed is None:
            self.failed = reason
            self.events.append({"failed": reason, "phase": self.phase})

    def snapshot(self) -> dict:
        return {"phase": self.phase, "dev_gate": self.dev_gate,
                "heldout_opened": self.heldout_opened, "failed": self.failed,
                "order": list(_ACCESS_ORDER),
                "events": [dict(event) for event in self.events]}


# ---------------------------------------------------------------------------
# Bank initialization helpers
# ---------------------------------------------------------------------------


def initialize_null_plastic(bank: ExpertBank) -> None:
    """Declare the initial P read as exactly null by zeroing each ``down``.

    Only an untaught bank may be initialized.  This keeps the fixed decoder
    unchanged (unlike a zero decoder, which could never produce a read) and
    gives the pre-teach / reset / frozen controls an exact null read path.
    """
    _require(isinstance(bank, ExpertBank), "bank must be an ExpertBank")
    _require(all(step == 0 for step in bank.expert_steps),
             "null initialization requires zero expert steps")
    if bank._optimizer is not None:
        _require(not any(bank._optimizer.state.values()),
                 "null initialization requires an empty optimizer state")
    for expert in bank.experts:
        with torch.no_grad():
            expert.down.zero_()
        _require(not bool(expert.down.any()), "down matrix was not zeroed")


def plastic_slot(bank: ExpertBank, expert_id: int, features: Tensor) -> Tensor:
    """Detached selected-expert output in the declared plastic width."""
    expert = bank.expert(expert_id)
    value = _float_tensor(features, "features")
    _require(value.shape[-1] == bank.width, "feature width differs from the bank")
    with torch.no_grad():
        slot = expert(value)
    return slot.detach().to("cpu", torch.float32)


def prompt_route_key(prompt_ids: Tensor) -> Tensor:
    """Deterministic prompt-token-only routing vector.

    No response, label, world, family or split identity is accepted.  The key
    is stable across model weights and processes and has only eight values.
    """
    ids = torch.as_tensor(prompt_ids).detach().to("cpu").reshape(-1).tolist()
    _require(ids and all(type(int(token)) is int for token in ids),
             "prompt ids must be a nonempty integer sequence")
    digest = hashlib.sha256(json.dumps([int(token) for token in ids]).encode()).digest()
    values = [digest[index] / 255.0 for index in range(8)]
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


class PlasticReadObserver:
    """Read-only tap over immutable plastic read values.

    It snapshots metadata and verifies tensor immutability; it cannot write
    weights, moments, observations or routes.  P-only parity means identical
    outputs with and without this tap attached.
    """

    read_only = True

    def __init__(self) -> None:
        self._reports: list[dict] = []

    @property
    def reports(self) -> tuple[dict, ...]:
        return tuple(self._reports)

    def observe(self, read: PlasticRead) -> dict:
        _require(isinstance(read, PlasticRead),
                 "plastic observer requires a PlasticRead")
        read.assert_immutable()
        report = {"expert_id": read.expert_id, "revision": read.revision,
                  "route_mode": read.route_mode,
                  "read_path_sha256": read.read_path_sha256,
                  "slot_sha256": _tensor_sha(read.slot),
                  "readonly": True}
        self._reports.append(report)
        return report


class NestedTextLearner:
    """Compose NL0 owners for one text scope; never duplicate their contracts."""

    def __init__(self, adapter, *, bank: ExpertBank, protocol: TextProtocol,
                 controller: PlasticLearningController | None = None,
                 coordinator: ObservationCoordinator | None = None,
                 route: RouteControl | None = None,
                 personality: PersonalityIdentity | None = None,
                 tokenizer_sha256: str | None = None) -> None:
        _require(isinstance(bank, ExpertBank), "bank must be an ExpertBank")
        _require(isinstance(protocol, TextProtocol),
                 "protocol must be a TextProtocol")
        if controller is not None:
            _require(isinstance(controller, PlasticLearningController),
                     "controller must be a PlasticLearningController")
            _require(controller.bank is bank,
                     "controller and learner must share one bank")
            _require(controller.scope == protocol.scope,
                     "controller scope differs from the protocol")
        if coordinator is not None:
            _require(isinstance(coordinator, ObservationCoordinator),
                     "coordinator must be an ObservationCoordinator")
            _require(coordinator.scope == protocol.scope,
                     "coordinator scope differs from the protocol")
            _require(coordinator.width == protocol.width,
                     "coordinator width differs from the protocol")
            if controller is not None and controller.coordinator is not None:
                _require(controller.coordinator is coordinator,
                         "controller and learner must share one coordinator")
        _require(bank.width == protocol.width
                 and bank.hidden_width == protocol.hidden_width
                 and bank.expert_count == protocol.expert_count,
                 "bank shape differs from the protocol")
        self.adapter = adapter
        self.bank = bank
        self.protocol = protocol
        self.controller = controller
        self.coordinator = coordinator
        self.personality = personality or (
            coordinator.personality if coordinator is not None
            else (controller.personality if controller is not None
                  else PersonalityIdentity.default()))
        for owner in (controller, coordinator):
            if owner is not None:
                _require(owner.personality == self.personality,
                         "personality identity differs across owners")
        expected_hidden = getattr(adapter, "hidden_size", None)
        if expected_hidden is None:
            base = getattr(adapter, "base", None)
            config = getattr(base, "config", None)
            expected_hidden = getattr(config, "hidden_size", None)
        if expected_hidden is not None:
            _require(int(expected_hidden) == protocol.hidden_size,
                     "adapter hidden size differs from the protocol")
        manifest = getattr(adapter, "manifest", None)
        if manifest is not None and getattr(manifest, "plastic", None) is not None:
            _require(manifest.plastic == protocol.plastic_spec,
                     "adapter plastic spec differs from the protocol")
            _require(adapter.plastic_read_digest() == protocol.read_path_sha256,
                     "adapter decoder differs from the protocol read path")
        self.route = route or RouteControl(protocol.route_mode,
                                           protocol.expert_count,
                                           protocol.route_seed)
        _require(self.route.expert_count == protocol.expert_count,
                 "route expert count differs from the protocol")
        if self.controller is None and self.coordinator is None:
            _require(self.route.mode == protocol.route_mode,
                     "read-only route mode differs from the protocol")
        if tokenizer_sha256 is not None:
            _digest(tokenizer_sha256, "tokenizer_sha256", allow_empty=True)
            if protocol.tokenizer_sha256:
                _require(tokenizer_sha256 == protocol.tokenizer_sha256,
                         "tokenizer differs from the protocol")
        self.tokenizer_sha256 = tokenizer_sha256 or protocol.tokenizer_sha256
        if all(step == 0 for step in bank.expert_steps) and (
                controller is None or controller.revision == 0):
            initialize_null_plastic(bank)
        self._projection = protocol.target_projection()
        self.accounting: dict[str, int] = {
            "events_observed": 0, "eligible_events": 0,
            "ineligible_events": 0, "feature_vectors": 0,
            "response_tokens": 0, "optimizer_steps": 0,
            "chunks_sealed": 0, "transitions": 0, "generated_tokens": 0,
            "teacher_requests": 0,
        }
        self._last_event: TextEventOutcome | None = None
        self._event_fingerprints: dict[str, str] = {}
        self.read_observer: PlasticReadObserver | None = None

    def attach_read_observer(self, observer: PlasticReadObserver | None) -> None:
        if observer is not None:
            _require(isinstance(observer, PlasticReadObserver),
                     "read observer must be read-only")
        self.read_observer = observer

    # -- state access ------------------------------------------------------

    @property
    def projection(self) -> Tensor:
        return self._projection.detach().clone()

    @property
    def revision(self) -> int:
        if self.controller is not None:
            return self.controller.revision
        return 0

    @property
    def last_event(self) -> TextEventOutcome | None:
        return self._last_event

    def _adapter_device(self) -> torch.device:
        parameters = getattr(self.adapter, "parameters", None)
        if parameters is not None:
            first = next(iter(parameters()), None)
            if first is not None:
                return first.device
        return torch.device("cpu")

    def _producer(self) -> str:
        key = getattr(self.adapter, "plastic_producer_key", None)
        if key is not None:
            return key()
        return fingerprint({"protocol": self.protocol.protocol_sha256,
                            "read_path": self.protocol.read_path_sha256,
                            "sites": list(self.protocol.feature_sites)})

    # -- feature/target extraction ----------------------------------------

    def prompt_features(self, prompt_ids: Tensor, prompt_mask: Tensor) -> dict[int, Tensor]:
        ids, mask = self._ids_mask(prompt_ids, prompt_mask, "prompt")
        _require(bool(mask.any()), "prompt mask must select at least one position")
        device = self._adapter_device()
        ids, mask = ids.to(device), mask.to(device)
        mask_cpu = mask.to("cpu")
        with torch.inference_mode():
            hidden = self.adapter.encode_sites(ids, mask,
                                               self.protocol.feature_sites)
            features = {}
            for site in self.protocol.feature_sites:
                value = hidden[site].to("cpu", torch.float32)
                _require(value.shape[0] == 1, "one prompt per extraction")
                count = mask_cpu.sum(dim=1, keepdim=True).clamp_min(1)
                pooled = (value * mask_cpu.unsqueeze(-1)).sum(dim=1) / count
                features[site] = (pooled @ self._projection.T)[0].detach().to(
                    "cpu", torch.float32)
        # A clone outside inference_mode produces ordinary tensors that a
        # local controller graph may consume.
        features = {site: value.clone() for site, value in features.items()}
        self.accounting["feature_vectors"] += len(features)
        return features

    def response_targets(self, input_ids: Tensor, attention_mask: Tensor,
                         response_mask: Tensor) -> dict[int, Tensor]:
        ids, mask = self._ids_mask(input_ids, attention_mask, "input")
        response = torch.as_tensor(response_mask).detach().to("cpu", torch.bool)
        _require(response.shape == mask.shape,
                 "response mask must match the input mask")
        _require(bool(response.any()), "response mask must select target tokens")
        _require(not bool((response & ~mask).any()),
                 "response tokens cannot be masked out of the input")
        device = self._adapter_device()
        ids, mask = ids.to(device), mask.to(device)
        with torch.inference_mode():
            hidden = self.adapter.encode_sites(ids, mask,
                                               self.protocol.feature_sites)
            targets = {}
            count = response.sum().clamp_min(1)
            for site in self.protocol.feature_sites:
                value = hidden[site].to("cpu", torch.float32)
                pooled = (value * response.unsqueeze(-1)).sum(dim=1) / count
                targets[site] = (pooled @ self._projection.T)[0].detach().to(
                    "cpu", torch.float32)
        targets = {site: value.clone() for site, value in targets.items()}
        self.accounting["response_tokens"] += int(response.sum())
        return targets

    @staticmethod
    def _ids_mask(ids: Tensor, mask: Tensor, name: str) -> tuple[Tensor, Tensor]:
        ids = torch.as_tensor(ids)
        mask = torch.as_tensor(mask)
        _require(ids.ndim == 2 and ids.dtype == torch.long and ids.shape[0] == 1
                 and ids.shape[1] > 0, f"{name} ids must be nonempty [1, tokens] int64")
        _require(tuple(mask.shape) == tuple(ids.shape),
                 f"{name} mask must match the ids")
        return ids, mask.bool()

    def _stacked(self, values: Mapping[int, Tensor]) -> Tensor:
        return torch.stack([values[site]
                            for site in self.protocol.feature_sites])

    # -- teaching ----------------------------------------------------------

    def learn_event(self, event: TextTeachingEvent, *, prompt_ids: Tensor,
                    prompt_mask: Tensor, input_ids: Tensor | None = None,
                    attention_mask: Tensor | None = None,
                    response_mask: Tensor | None = None,
                    reader_ids: Sequence[str] | None = None,
                    sealed_at: float | None = None,
                    loss_gradients=None) -> TextEventOutcome:
        """Apply exactly one transaction and seal exactly one observation chunk."""
        _require(isinstance(event, TextTeachingEvent), "event contract required")
        _require(event.scope == self.protocol.scope,
                 "event scope differs from the learner")
        _require(event.world_uid == self.protocol.world_uid,
                 "event world differs from the protocol")
        event_fingerprint = event.fingerprint()
        previous_fingerprint = self._event_fingerprints.get(event.event_id)
        if previous_fingerprint is not None and previous_fingerprint != event_fingerprint:
            raise DuplicateObservation(
                f"event {event.event_id!r} was already observed with different content")
        if event.supersedes is not None:
            _require(event.supersedes in self._event_fingerprints,
                     f"correction {event.event_id!r} supersedes an unknown/absent event")
        readers = tuple(reader_ids or tuple(str(site)
                                            for site in self.protocol.feature_sites))
        if not event.eligible:
            _require(self.coordinator is not None,
                     "sealing an ineligible chunk requires a coordinator")
            chunk = self.coordinator.seal_observation_chunk(
                chunk_id=event.event_id, observations=(),
                reader_ids=readers,
                sealed_at=event.target_available_at if sealed_at is None else sealed_at)
            _require(chunk.transition_count == 0,
                     "an ineligible event performed a fast transition")
            self.accounting["events_observed"] += 1
            self.accounting["ineligible_events"] += 1
            self.accounting["chunks_sealed"] += 1
            self._event_fingerprints[event.event_id] = event_fingerprint
            outcome = TextEventOutcome(
                status="ineligible", event_id=event.event_id, expert_id=-1,
                route_mode=self.route.mode, feature_sha256="",
                target_sha256="", transaction=None, chunk=chunk,
                retried=chunk.retried)
            self._last_event = outcome
            return outcome
        _require(self.controller is not None and self.coordinator is not None,
                 "teaching requires a controller and a coordinator")
        _require(event.verified and event.source in _ALLOWED_SOURCES,
                 "teaching requires a verified eligible observation")
        _require(input_ids is not None and attention_mask is not None
                 and response_mask is not None,
                 "teaching requires prompt and verified-response token tensors")
        features = self.prompt_features(prompt_ids, prompt_mask)
        targets = self.response_targets(input_ids, attention_mask, response_mask)
        feature_tensor = self._stacked(features)
        target_tensor = self._stacked(targets)
        feature_sha = _tensor_sha(feature_tensor)
        target_sha = _tensor_sha(target_tensor)
        route_key = prompt_route_key(prompt_ids)
        expert_id = self.route.route(route_key)
        transaction = self.controller.apply_teaching_transaction(
            event_id=event.event_id, expert_id=expert_id,
            features=feature_tensor, target=target_tensor,
            steps=self.protocol.steps_per_event, scope=self.protocol.scope,
            loss_gradients=loss_gradients)
        _require(transaction.expert_id == expert_id,
                 "transaction route differs from the selected expert")
        mean_feature = feature_tensor.mean(dim=0)
        mean_target = target_tensor.mean(dim=0)
        key = torch.nn.functional.normalize(mean_feature, dim=0)
        value = mean_target
        norm = value.norm()
        if float(norm) > 1.0:
            value = value / norm
        available_at = (event.target_available_at if sealed_at is None
                        else float(sealed_at))
        observation = Observation(
            event_id=event.event_id, scope=event.scope,
            key=tuple(float(item) for item in key), 
            value=tuple(float(item) for item in value),
            available_at=available_at, eligible=True, source=event.source)
        chunk = self.coordinator.seal_observation_chunk(
            chunk_id=event.event_id, observations=(observation,),
            reader_ids=readers,
            sealed_at=available_at if sealed_at is None else sealed_at)
        _require(chunk.transition_count == 1,
                 "an eligible chunk must perform exactly one transition")
        _require(transaction.revision >= 1 and chunk.post_revision >= 1,
                 "committed event did not advance transaction and chunk revisions")
        self.accounting["events_observed"] += 1
        self.accounting["eligible_events"] += 1
        self.accounting["optimizer_steps"] += transaction.steps
        self.accounting["chunks_sealed"] += 1
        self.accounting["transitions"] += chunk.transition_count
        self._event_fingerprints[event.event_id] = event_fingerprint
        outcome = TextEventOutcome(
            status="committed", event_id=event.event_id,
            expert_id=expert_id, route_mode=self.route.mode,
            feature_sha256=feature_sha, target_sha256=target_sha,
            transaction=transaction, chunk=chunk,
            retried=bool(transaction.retried or chunk.retried))
        self._last_event = outcome
        return outcome

    def learn_event_response_ce(self, session, event: TextTeachingEvent, *,
                                scope, prompt_ids: Tensor, prompt_mask: Tensor,
                                input_ids: Tensor, attention_mask: Tensor,
                                response_mask: Tensor,
                                reader_ids: Sequence[str] | None = None,
                                sealed_at: float | None = None) -> TextEventOutcome:
        """Commit one event using verified response CE through the frozen decoder.

        Detached ``prompt_features``/``response_targets`` remain observation
        evidence only.  This callback constructs a fresh autograd read and never
        reuses those detached inference helpers as its training graph.
        """
        def response_gradients(expert):
            device = self._adapter_device()
            train_prompt_ids = prompt_ids.to(device)
            train_prompt_mask = prompt_mask.to(device)
            train_input_ids = input_ids.to(device)
            train_attention_mask = attention_mask.to(device)
            train_response_mask = response_mask.to(device)
            read = self.differentiable_plastic_read(
                train_prompt_ids, train_prompt_mask)
            task = TaskGraph(task_id=f"response-ce:{event.event_id}",
                             goal=event.event_id, revision=0)
            labels = train_input_ids.clone()
            labels[~train_response_mask] = -100
            with session.training_request(
                    prompt_ids=train_prompt_ids, prompt_mask=train_prompt_mask,
                    query=task.goal, scope=scope, task=task,
                    enabled_ports=(), core_impl="none",
                    enabled_branches=("plastic",), plastic=read,
                    identity=None, identity_enabled=False) as request:
                output = session.forward(
                    request, train_input_ids, labels=labels,
                    attention_mask=train_attention_mask)
                _require(output.loss is not None and output.loss.requires_grad,
                         "response CE must retain an autograd path")
                gradients = torch.autograd.grad(output.loss, tuple(expert.parameters()))
                return output.loss.detach(), {
                    name: gradient.detach()
                    for (name, _), gradient in zip(expert.named_parameters(), gradients)}

        return self.learn_event(
            event, prompt_ids=prompt_ids, prompt_mask=prompt_mask,
            input_ids=input_ids, attention_mask=attention_mask,
            response_mask=response_mask, reader_ids=reader_ids,
            sealed_at=sealed_at, loss_gradients=response_gradients)


    def extract_site_features(self, prompt_ids: Tensor,

                              prompt_mask: Tensor) -> dict[int, Tensor]:
        """Declared compatibility name for detached site features."""
        return self.prompt_features(prompt_ids, prompt_mask)

    def extract_site_targets(self, input_ids: Tensor, attention_mask: Tensor,
                             response_mask: Tensor) -> dict[int, Tensor]:
        """Declared compatibility name for verified-response targets."""
        return self.response_targets(input_ids, attention_mask, response_mask)

    def observe_teaching_event(self, event: TextTeachingEvent, *,
                               prompt_ids: Tensor, prompt_mask: Tensor,
                               full_ids: Tensor,
                               response_mask: Tensor,
                               attention_mask: Tensor | None = None,
                               reader_ids: Sequence[str] | None = None,
                               sealed_at: float | None = None
                               ) -> TextEventOutcome:
        attention = (torch.ones_like(full_ids, dtype=torch.bool)
                     if attention_mask is None else attention_mask)
        return self.learn_event(
            event, prompt_ids=prompt_ids, prompt_mask=prompt_mask,
            input_ids=full_ids, attention_mask=attention,
            response_mask=response_mask, reader_ids=reader_ids,
            sealed_at=sealed_at)

    def seal_text_chunk(self, chunk_id: str,
                        observations: Sequence[Observation], *,
                        reader_ids: Sequence[str] | None = None,
                        sealed_at: float | None = None) -> ChunkOutcome:
        _require(self.coordinator is not None, "chunk sealing requires a coordinator")
        readers = tuple(reader_ids or tuple(str(site)
                                            for site in self.protocol.feature_sites))
        return self.coordinator.seal_observation_chunk(
            chunk_id=chunk_id, observations=observations,
            reader_ids=readers, sealed_at=sealed_at)

    # -- P-only read path --------------------------------------------------

    def plastic_read(self, prompt_ids: Tensor, prompt_mask: Tensor, *,
                     route_mode: str | None = None) -> PlasticRead:
        """Selected-expert read value without any source or optimizer state."""
        _require(route_mode in (None, "fixed", "shuffled"),
                 "unknown read route mode")
        control = (self.route if route_mode is None or route_mode == self.route.mode
                   else RouteControl(route_mode, self.protocol.expert_count,
                                     self.protocol.route_seed))
        features = self.prompt_features(prompt_ids, prompt_mask)
        mean = self._stacked(features).mean(dim=0, keepdim=True)
        expert_id = control.route(prompt_route_key(prompt_ids))
        slot = plastic_slot(self.bank, expert_id, mean)
        slot = slot.to(self._adapter_device())
        _require(slot.requires_grad is False, "plastic slot must be detached")
        read = PlasticRead(
            slot=slot, expert_id=expert_id, producer=self._producer(),
            revision=self.revision, route_mode=control.mode,
            route_permutation=control.permutation,
            read_path_sha256=self.protocol.read_path_sha256)
        if self.read_observer is not None:
            self.read_observer.observe(read)
        return read

    # -- Differentiable training read path --------------------------------

    def prompt_features_differentiable(self, prompt_ids: Tensor,
                                       prompt_mask: Tensor) -> dict[int, Tensor]:
        """Encode prompt sites without inference-mode or detached tensors."""
        ids, mask = self._ids_mask(prompt_ids, prompt_mask, "prompt")
        _require(bool(mask.any()), "prompt mask must select at least one position")
        device = self._adapter_device()
        ids, mask = ids.to(device), mask.to(device)
        mask_cpu = mask.to("cpu")
        hidden = self.adapter.encode_sites(ids, mask, self.protocol.feature_sites)
        features = {}
        for site in self.protocol.feature_sites:
            value = hidden[site].to("cpu", torch.float32)
            _require(value.shape[0] == 1, "one prompt per extraction")
            count = mask_cpu.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (value * mask_cpu.unsqueeze(-1)).sum(dim=1) / count
            features[site] = (pooled @ self._projection.T)[0]
        self.accounting["feature_vectors"] += len(features)
        return features

    def differentiable_plastic_read(self, prompt_ids: Tensor, prompt_mask: Tensor, *,
                                    route_mode: str | None = None) -> DifferentiablePlasticRead:
        """Build an autograd-preserving read for a training request only."""
        _require(route_mode in (None, "fixed", "shuffled"),
                 "unknown training read route mode")
        control = (self.route if route_mode is None or route_mode == self.route.mode
                   else RouteControl(route_mode, self.protocol.expert_count,
                                     self.protocol.route_seed))
        features = self.prompt_features_differentiable(prompt_ids, prompt_mask)
        mean = self._stacked(features).mean(dim=0, keepdim=True)
        expert_id = control.route(prompt_route_key(prompt_ids))
        slot = self.bank.expert(expert_id)(mean)
        slot = slot.to(self._adapter_device())
        _require(slot.requires_grad, "training plastic slot must preserve autograd")
        read = DifferentiablePlasticRead(
            slot=slot, expert_id=expert_id, producer=self._producer(),
            revision=self.revision, route_mode=control.mode,
            route_permutation=control.permutation,
            read_path_sha256=self.protocol.read_path_sha256)
        if self.read_observer is not None:
            self.read_observer.observe(read)
        return read

    # -- persistence -------------------------------------------------------

    def _theta_binding(self) -> dict:
        return {"revision": self.protocol.theta_revision,
                "lock_sha256": self.protocol.theta_lock_sha256,
                "stamp_sha256": self.protocol.parameter_stamp_sha256}

    def _route_binding(self) -> dict:
        return {"mode": self.route.mode, "expert_count": self.route.expert_count,
                "seed": self.protocol.route_seed,
                "permutation": list(self.route.permutation)}

    def _feature_binding(self) -> dict:
        return {"sites": list(self.protocol.feature_sites),
                "width": self.protocol.width,
                "hidden_size": self.protocol.hidden_size,
                "target_projection_seed": self.protocol.target_projection_seed,
                "target_projection_sha256": self.protocol.target_projection_sha256}

    def _read_binding(self) -> dict:
        return {"decoder_sha256": self.protocol.read_path_sha256,
                "gain": self.protocol.read_gain,
                "version": "fixed_projection_sites_masked_mean.v1"}

    def p_only_export(self) -> dict:
        """Weights plus immutable bindings; intentionally no learning state."""
        parameters = [[name, parameter.detach().to("cpu", torch.float32).clone()]
                      for expert in self.bank.experts
                      for name, parameter in expert.named_parameters()]
        return {
            "format": P_ONLY_FORMAT,
            "scope": self.protocol.scope,
            "theta": self._theta_binding(),
            "protocol_sha256": self.protocol.protocol_sha256,
            "bank": {"experts": self.bank.expert_count,
                     "width": self.bank.width,
                     "hidden_width": self.bank.hidden_width,
                     "seed": self.bank.seed},
            "route": self._route_binding(),
            "feature_producer": self._feature_binding(),
            "read_path": self._read_binding(),
            "expert_steps": list(self.bank.expert_steps),
            "parameters": parameters,
        }

    def save_p_only(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        torch.save(self.p_only_export(), temporary)
        os.replace(temporary, destination)
        return destination

    def load_p_only(self, payload: Mapping[str, Any]) -> None:
        """Load weights only into this read-only learner; fail closed on mismatch."""
        _require(self.controller is None and self.coordinator is None,
                 "P-only load is forbidden on a learning learner")
        validate_p_only_payload(payload, protocol=self.protocol,
                                route_permutation=self.route.permutation)
        parameters = [(name, parameter)
                      for expert in self.bank.experts
                      for name, parameter in expert.named_parameters()]
        saved = payload["parameters"]
        _require(len(saved) == len(parameters),
                 "P-only parameter count differs from the bank")
        for (expected_name, parameter), item in zip(parameters, saved):
            _require(item[0] == expected_name,
                     "P-only parameter order differs from the bank")
            value = item[1]
            _require(isinstance(value, Tensor) and value.dtype == torch.float32
                     and tuple(value.shape) == tuple(parameter.shape)
                     and bool(torch.isfinite(value).all()),
                     "P-only weights must be finite FP32 of the declared shape")
            with torch.no_grad():
                parameter.copy_(value)
        self.bank.expert_steps[:] = [int(item)
                                     for item in payload["expert_steps"]]

    def checkpoint_payload(self) -> dict:
        _require(self.controller is not None,
                 "a full checkpoint requires the learning controller")
        return {
            "format": TEXT_CHECKPOINT_FORMAT,
            "kind": "nested_text_full_checkpoint",
            "scope": self.protocol.scope,
            "controller": self.controller.checkpoint_payload(),
            "text": {
                "protocol": self.protocol.to_dict(),
                "theta": self._theta_binding(),
                "route": self._route_binding(),
                "feature_producer": self._feature_binding(),
                "read_path": self._read_binding(),
                "personality": self.personality.to_dict(),
                "world_uid": self.protocol.world_uid,
                "p_init": self.protocol.p_init,
                "tokenizer_sha256": self.tokenizer_sha256,
                "event_fingerprints": dict(sorted(
                    self._event_fingerprints.items())),
                "p_only_policy": {"moments": "absent",
                                  "updates": "resume_enabled"},
            },
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_text = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        temporary = Path(temporary_text)
        os.close(descriptor)
        try:
            torch.save(self.checkpoint_payload(), temporary)
            with open(temporary, "rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        return destination

    def load_checkpoint_file(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.load_checkpoint(payload)

    def load_checkpoint(self, payload: Mapping[str, Any] | str | Path) -> None:
        if isinstance(payload, (str, Path)):
            self.load_checkpoint_file(payload)
            return
        _require(self.controller is not None,
                 "a full checkpoint requires the learning controller")
        _require(isinstance(payload, Mapping)
                 and payload.get("format") == TEXT_CHECKPOINT_FORMAT
                 and payload.get("kind") == "nested_text_full_checkpoint",
                 "invalid nested-text checkpoint")
        text = payload.get("text")
        _require(isinstance(text, Mapping), "checkpoint text metadata required")
        stored_protocol = TextProtocol.from_dict(text["protocol"])
        _require(stored_protocol == self.protocol,
                 "checkpoint protocol differs from the live protocol")
        _require(text["theta"] == self._theta_binding(),
                 "checkpoint theta binding is stale")
        _require(text["route"] == self._route_binding(),
                 "checkpoint route binding differs")
        _require(text["feature_producer"] == self._feature_binding(),
                 "checkpoint feature producer differs")
        _require(text["read_path"] == self._read_binding(),
                 "checkpoint read path differs")
        _require(text["world_uid"] == self.protocol.world_uid
                 and payload.get("scope") == self.protocol.scope,
                 "checkpoint world/scope differs")
        _require(PersonalityIdentity.from_dict(text["personality"])
                 == self.personality,
                 "checkpoint personality differs from the stable identity")
        _require(text["tokenizer_sha256"] == self.tokenizer_sha256,
                 "checkpoint tokenizer differs")
        fingerprints = text.get("event_fingerprints", {})
        _require(isinstance(fingerprints, Mapping)
                 and all(isinstance(key, str) and isinstance(value, str)
                         and len(value) == 64 for key, value in fingerprints.items()),
                 "checkpoint event identity ledger is invalid")
        self.controller.load_checkpoint(payload["controller"])
        self._event_fingerprints = dict(fingerprints)

    # -- accounting --------------------------------------------------------

    def record_cost(self, **values: int) -> None:
        for name, value in values.items():
            _require(type(value) is int and value >= 0,
                     f"cost counter {name} must be a nonnegative integer")
            self.accounting[name] = self.accounting.get(name, 0) + value


def evaluate_dev_gate(*, metrics: Mapping[str, Any], cost: Mapping[str, Any],
                      gates: Mapping[str, Any],
                      cost_limits: Mapping[str, Any]) -> list[str]:
    """Return declared development-gate failures without opening heldout.

    This is deliberately pure so the stop-before-heldout decision is covered on
    CPU.  A missing wrong-world control or an unmeasured duty cycle fails
    closed rather than becoming a silent pass.
    """
    failures: list[str] = []
    new_accuracy = float(metrics.get("new_instance_accuracy", 0.0))
    preteach_accuracy = float(metrics.get("preteach_accuracy", 0.0))
    gain = new_accuracy - preteach_accuracy
    if new_accuracy < float(gates["min_new_instance_accuracy"]):
        failures.append("min_new_instance_accuracy")
    if gain < float(gates["min_gain_over_preteach"]):
        failures.append("min_gain_over_preteach")
    if float(metrics.get("unaffected_accuracy", 0.0)) < float(
            gates["min_unaffected_accuracy"]):
        failures.append("min_unaffected_accuracy")
    if float(metrics.get("correction_accuracy", 0.0)) < float(
            gates["min_correction_accuracy"]):
        failures.append("min_correction_accuracy")
    if float(metrics.get("unknown_false_recall", 1.0)) > float(
            gates["max_unknown_false_recall"]):
        failures.append("max_unknown_false_recall")
    if float(metrics.get("revoked_false_recall", 1.0)) > float(
            gates["max_revoked_false_recall"]):
        failures.append("max_revoked_false_recall")
    if float(metrics.get("retention_decline", 1.0)) > float(
            gates["max_retention_decline"]):
        failures.append("max_retention_decline")
    if gates.get("require_shuffled_not_better") and (
            float(metrics.get("shuffled_accuracy", 1.0))
            > new_accuracy + 1e-9):
        failures.append("shuffled_route_not_better")
    if gates.get("require_controls_not_reproduce_gain"):
        threshold = float(gates["min_gain_over_preteach"])
        if (float(metrics.get("frozen_accuracy", 1.0)) - preteach_accuracy
                >= threshold
                or float(metrics.get("reset_P_accuracy", 1.0))
                - preteach_accuracy >= threshold):
            failures.append("controls_reproduce_gain")
    if metrics.get("spine_absent_parity") is not True:
        failures.append("spine_absent_parity")
    if metrics.get("wrong_world_available") is not True:
        failures.append("wrong_world_control_unavailable")
    elif (float(metrics.get("wrong_world_on_primary_accuracy", 1.0))
          - preteach_accuracy >= float(gates["min_gain_over_preteach"])):
        failures.append("wrong_world_reproduces_gain")
    if int(cost.get("peak_driver_bytes", 1 << 62)) > int(
            cost_limits["max_peak_driver_bytes"]):
        failures.append("max_peak_driver_bytes")
    if float(cost.get("warm_p95_overhead_ms", 1e30)) > float(
            cost_limits["max_warm_p95_overhead_ms"]):
        failures.append("max_warm_p95_overhead_ms")
    duty = cost.get("duty_cycle")
    if duty is None:
        failures.append("duty_cycle_unavailable")
    elif float(duty) > float(cost_limits["max_duty_cycle"]):
        failures.append("max_duty_cycle")
    checkpoint_bytes = cost.get("checkpoint_bytes")
    if checkpoint_bytes is None or int(checkpoint_bytes) > int(
            cost_limits["max_checkpoint_bytes"]):
        failures.append("max_checkpoint_bytes")
    if int(cost.get("p_only_bytes", 1 << 62)) > int(
            cost_limits["max_p_only_bytes"]):
        failures.append("max_p_only_bytes")
    return failures


# ---------------------------------------------------------------------------
# P-only validation
# ---------------------------------------------------------------------------


def _expected_parameter_shapes(protocol: TextProtocol) -> list[tuple[str, tuple]]:
    shapes = [("gate", (protocol.hidden_width, protocol.width)),
              ("up", (protocol.hidden_width, protocol.width)),
              ("down", (protocol.width, protocol.hidden_width))]
    return [(name, shape) for _ in range(protocol.expert_count)
            for name, shape in shapes]


def validate_p_only_payload(payload: Mapping[str, Any], *, protocol: TextProtocol,
                            route_permutation: Sequence[int] | None = None) -> None:
    """Strict weights-only validation; no learning state may be present."""
    _require(isinstance(payload, Mapping), "P-only payload must be a mapping")
    keys = set(payload)
    forbidden = keys & P_ONLY_FORBIDDEN
    _require(not forbidden, f"P-only payload contains forbidden state: {sorted(forbidden)}")
    missing = P_ONLY_KEYS - keys
    _require(not missing, f"P-only payload is missing: {sorted(missing)}")
    unknown = keys - P_ONLY_KEYS
    _require(not unknown, f"P-only payload has unknown fields: {sorted(unknown)}")
    _require(payload["format"] == P_ONLY_FORMAT, "unsupported P-only format")
    _require(payload["scope"] == protocol.scope, "P-only scope differs from the protocol")
    _require(payload["protocol_sha256"] == protocol.protocol_sha256,
             "P-only protocol binding is stale")
    theta = payload["theta"]
    _require(isinstance(theta, Mapping), "P-only theta binding must be a mapping")
    _require(theta == {"revision": protocol.theta_revision,
                       "lock_sha256": protocol.theta_lock_sha256,
                       "stamp_sha256": protocol.parameter_stamp_sha256},
             "P-only theta binding differs from the sealed release")
    bank = payload["bank"]
    _require(isinstance(bank, Mapping)
             and bank.get("experts") == protocol.expert_count
             and bank.get("width") == protocol.width
             and bank.get("hidden_width") == protocol.hidden_width
             and type(bank.get("seed")) is int,
             "P-only bank shape differs from the protocol")
    route = payload["route"]
    _require(isinstance(route, Mapping)
             and route.get("mode") == protocol.route_mode
             and route.get("expert_count") == protocol.expert_count
             and route.get("seed") == protocol.route_seed,
             "P-only route binding differs from the protocol")
    permutation = tuple(route.get("permutation", ()))
    expected_permutation = (RouteControl(protocol.route_mode, protocol.expert_count,
                                        protocol.route_seed).permutation)
    _require(permutation == expected_permutation,
             "P-only route permutation differs from the declared control")
    if route_permutation is not None and tuple(route_permutation) != permutation:
        raise NestedTextError("P-only route permutation differs from the live route")
    feature = payload["feature_producer"]
    _require(isinstance(feature, Mapping)
             and tuple(feature.get("sites", ())) == tuple(protocol.feature_sites)
             and feature.get("width") == protocol.width
             and feature.get("hidden_size") == protocol.hidden_size
             and feature.get("target_projection_seed") == protocol.target_projection_seed
             and feature.get("target_projection_sha256")
             == protocol.target_projection_sha256,
             "P-only feature producer differs from the protocol")
    read = payload["read_path"]
    _require(isinstance(read, Mapping)
             and read.get("decoder_sha256") == protocol.read_path_sha256
             and float(read.get("gain", -1)) == protocol.read_gain,
             "P-only read path differs from the protocol")
    steps = payload["expert_steps"]
    _require(isinstance(steps, (list, tuple)) and len(steps) == protocol.expert_count
             and all(type(item) is int and item >= 0 for item in steps),
             "P-only expert counters are invalid")
    parameters = payload["parameters"]
    expected_shapes = _expected_parameter_shapes(protocol)
    _require(len(parameters) == len(expected_shapes),
             "P-only parameter inventory differs")
    for (expected_name, expected_shape), item in zip(expected_shapes, parameters):
        _require(isinstance(item, (list, tuple)) and len(item) == 2
                 and item[0] == expected_name,
                 "P-only parameter names/order differ")
        value = item[1]
        _require(isinstance(value, Tensor) and value.dtype == torch.float32
                 and tuple(value.shape) == expected_shape
                 and bool(torch.isfinite(value).all()),
                 "P-only parameters must be finite FP32 of the declared shape")
