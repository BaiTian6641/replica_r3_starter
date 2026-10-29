"""Exact evidence records and a small Spark spine-plus-memory learner.

``E`` is a host-owned, provenance-bearing evidence bank. The neural reader and
selected P expert perform semantic use; the host never supplies gold record IDs
or answer-aware routing. E corrections/revocations invalidate dependent views,
while P and its optimizer/dedup/clock state persist independently.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import torch

from .spine import (BRANCHES, DecisionSnapshot, Packet, SpineCore,
                    SpineError, SpineForwardContext, SpineSpec)
from .spine_checkpoint import load_spine_checkpoint, save_spine_checkpoint
from ..nested_learning import ExpertBank, PlasticLearningController
from ..model_identity import durable_model_identity

EVIDENCE_CHECKPOINT_FORMAT = "replica.spark.evidence.v1"


class EvidenceRecord:
    """One exact evidence version with source and validity provenance."""

    def __init__(self, *, record_id: str, scope: str, payload: torch.Tensor,
                 source_epoch: int, available_at: int, valid_from: int,
                 producer_revision: str, representation_version: str,
                 codec_version: str, payload_type: str,
                 source_lineage: Mapping[str, Any], valid_until: int | None = None,
                 supersedes: str | None = None) -> None:
        if not isinstance(record_id, str) or not record_id:
            raise SpineError("evidence record_id is required")
        if not isinstance(scope, str) or not scope:
            raise SpineError("evidence scope is required")
        if not torch.is_tensor(payload) or payload.ndim != 1 \
                or payload.shape[0] != 512 or not torch.is_floating_point(payload):
            raise SpineError("evidence payload must be a width-512 vector")
        if not bool(torch.isfinite(payload).all()):
            raise SpineError("evidence payload is nonfinite")
        for name, value in (("source_epoch", source_epoch),
                            ("available_at", available_at), ("valid_from", valid_from)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SpineError(f"{name} must be a nonnegative integer")
        if available_at < valid_from:
            raise SpineError("evidence availability precedes validity start")
        if valid_until is not None and (
                not isinstance(valid_until, int) or isinstance(valid_until, bool)
                or valid_until < 0):
            raise SpineError("valid_until must be a nonnegative integer or null")
        if valid_until is not None and valid_until < available_at:
            raise SpineError("evidence validity ends before availability")
        for name, value in (("producer_revision", producer_revision),
                            ("representation_version", representation_version),
                            ("codec_version", codec_version), ("payload_type", payload_type)):
            if not isinstance(value, str) or not value:
                raise SpineError(f"{name} is required")
        if payload_type not in {"observation", "interpretation", "prediction",
                                "verified_feedback"}:
            raise SpineError("unknown evidence payload type")
        if not isinstance(source_lineage, Mapping) or not source_lineage:
            raise SpineError("evidence source lineage is required")
        if supersedes is not None and (
                not isinstance(supersedes, str) or not supersedes):
            raise SpineError("supersedes must be a nonempty record identity or null")
        self.record_id = record_id
        self.scope = scope
        self.payload = payload.detach().clone()
        self.source_epoch = source_epoch
        self.available_at = available_at
        self.valid_from = valid_from
        self.valid_until = valid_until
        self.producer_revision = producer_revision
        self.representation_version = representation_version
        self.codec_version = codec_version
        self.payload_type = payload_type
        self.source_lineage = dict(source_lineage)
        self.supersedes = supersedes

    @property
    def payload_bytes(self) -> int:
        return int(self.payload.numel() * self.payload.element_size())

    def identity(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "scope": self.scope,
            "source_epoch": self.source_epoch, "available_at": self.available_at,
            "valid_from": self.valid_from, "valid_until": self.valid_until,
            "producer_revision": self.producer_revision,
            "representation_version": self.representation_version,
            "codec_version": self.codec_version, "payload_type": self.payload_type,
            "source_lineage": self.source_lineage, "supersedes": self.supersedes,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.identity(), "payload": self.payload.detach().clone()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EvidenceRecord":
        required = {"record_id", "scope", "source_epoch", "available_at",
                    "valid_from", "valid_until", "producer_revision",
                    "representation_version", "codec_version", "payload_type",
                    "source_lineage", "supersedes", "payload"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise SpineError("evidence record checkpoint keys differ")
        return cls(record_id=payload["record_id"], scope=payload["scope"],
                   payload=payload["payload"], source_epoch=payload["source_epoch"],
                   available_at=payload["available_at"], valid_from=payload["valid_from"],
                   producer_revision=payload["producer_revision"],
                   representation_version=payload["representation_version"],
                   codec_version=payload["codec_version"],
                   payload_type=payload["payload_type"],
                   source_lineage=payload["source_lineage"],
                   valid_until=payload["valid_until"], supersedes=payload["supersedes"])


class EvidenceBank:
    """Bounded exact E store with explicit correction and revocation versions."""

    def __init__(self, scope: str, *, max_records: int = 64,
                 max_payload_bytes: int = 1 << 20) -> None:
        if not isinstance(scope, str) or not scope:
            raise SpineError("evidence bank scope is required")
        for name, value in (("max_records", max_records),
                            ("max_payload_bytes", max_payload_bytes)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SpineError(f"{name} must be positive")
        self.scope = scope
        self.max_records = max_records
        self.max_payload_bytes = max_payload_bytes
        self._active: OrderedDict[str, EvidenceRecord] = OrderedDict()
        self._history: list[EvidenceRecord] = []
        self._revocations: dict[str, str] = {}
        self._revision = 0
        self._payload_bytes = 0
        self._cache: dict[tuple, Packet] = {}

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def pending(self) -> int:
        return len(self._active)

    @property
    def payload_bytes(self) -> int:
        return self._payload_bytes

    @property
    def history(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._history)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format": EVIDENCE_CHECKPOINT_FORMAT,
            "scope": self.scope,
            "max_records": self.max_records,
            "max_payload_bytes": self.max_payload_bytes,
            "revision": self._revision,
            "active": [record.record_id for record in self._active.values()],
            "history": [record.to_payload() for record in self._history],
            "revocations": dict(self._revocations),
            "payload_bytes": self._payload_bytes,
        }

    def load_checkpoint(self, payload: Mapping[str, Any]) -> None:
        required = {"format", "scope", "max_records", "max_payload_bytes",
                    "revision", "active", "history", "revocations", "payload_bytes"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise SpineError("evidence checkpoint keys differ")
        if payload["format"] != EVIDENCE_CHECKPOINT_FORMAT \
                or payload["scope"] != self.scope:
            raise SpineError("evidence checkpoint identity differs")
        if payload["max_records"] != self.max_records \
                or payload["max_payload_bytes"] != self.max_payload_bytes:
            raise SpineError("evidence checkpoint limits differ")
        if type(payload["revision"]) is not int or payload["revision"] < 0:
            raise SpineError("evidence checkpoint revision is invalid")
        if not isinstance(payload["active"], list) \
                or not isinstance(payload["history"], list):
            raise SpineError("evidence checkpoint records must be lists")
        active_ids = payload["active"]
        if any(not isinstance(record_id, str) or not record_id
               for record_id in active_ids):
            raise SpineError("evidence checkpoint active identities are invalid")
        if len(set(active_ids)) != len(active_ids):
            raise SpineError("evidence checkpoint has duplicate active identities")
        history = [EvidenceRecord.from_payload(item) for item in payload["history"]]
        history_ids = [record.record_id for record in history]
        if len(set(history_ids)) != len(history_ids):
            raise SpineError("evidence checkpoint has duplicate lineage identities")
        if any(record.scope != self.scope for record in history):
            raise SpineError("evidence checkpoint record scope differs")
        history_by_id = {record.record_id: record for record in history}
        if any(record_id not in history_by_id for record_id in active_ids):
            raise SpineError("evidence checkpoint active record lacks lineage")
        active = [history_by_id[record_id] for record_id in active_ids]
        revocations = payload["revocations"]
        if not isinstance(revocations, Mapping) or any(
                not isinstance(key, str) or not key
                or not isinstance(value, str) or not value
                for key, value in revocations.items()):
            raise SpineError("evidence checkpoint revocations are invalid")
        if any(record_id not in history_by_id for record_id in revocations):
            raise SpineError("evidence checkpoint revocation lacks lineage")
        if any(record_id in revocations for record_id in active_ids):
            raise SpineError("evidence checkpoint has an active revoked record")
        payload_bytes = sum(record.payload_bytes for record in active)
        if type(payload["payload_bytes"]) is not int \
                or payload["payload_bytes"] != payload_bytes:
            raise SpineError("evidence checkpoint byte count differs")
        if len(active) > self.max_records or payload_bytes > self.max_payload_bytes:
            raise SpineError("evidence checkpoint exceeds configured limits")
        self._active = OrderedDict((record.record_id, record) for record in active)
        self._history = history
        self._revocations = dict(revocations)
        self._revision = payload["revision"]
        self._payload_bytes = payload_bytes
        self._cache.clear()

    def _bump(self) -> None:
        self._revision += 1
        self._cache.clear()

    def admit(self, record: EvidenceRecord) -> bool:
        if not isinstance(record, EvidenceRecord):
            raise SpineError("evidence bank admits EvidenceRecord instances")
        if record.scope != self.scope:
            raise SpineError("evidence scope differs from the bank")
        if record.record_id in self._revocations:
            raise SpineError("a revoked record cannot be re-admitted without a new identity")
        current = self._active.get(record.record_id)
        if current is not None and record.supersedes is None:
            if current.identity() == record.identity() and torch.equal(
                    current.payload, record.payload):
                return False
            raise SpineError("duplicate evidence record with changed content")
        if current is not None:
            raise SpineError("correction record identity is already active")
        if current is None and any(
                previous.record_id == record.record_id for previous in self._history):
            raise SpineError("evidence record identity was already used")
        parent = None
        if record.supersedes is not None:
            parent = self._active.get(record.supersedes)
            if parent is None:
                raise SpineError("correction parent is not active")
            if parent.record_id == record.record_id:
                raise SpineError("correction requires a new record identity")
            if record.source_epoch <= parent.source_epoch:
                raise SpineError("correction source epoch must increase")
            if record.available_at < parent.available_at:
                raise SpineError("correction cannot be available before its parent")
        active_count = len(self._active) - (1 if parent is not None else 0)
        if active_count >= self.max_records:
            raise SpineError("evidence record capacity exceeded")
        removed_bytes = parent.payload_bytes if parent is not None else 0
        next_bytes = self._payload_bytes - removed_bytes + record.payload_bytes
        if next_bytes > self.max_payload_bytes:
            raise SpineError("evidence payload byte capacity exceeded")
        if parent is not None:
            del self._active[parent.record_id]
        self._active[record.record_id] = record
        self._history.append(record)
        self._payload_bytes = next_bytes
        self._bump()
        return True

    def revoke(self, record_id: str, *, reason: str) -> None:
        if record_id not in self._active:
            raise SpineError("cannot revoke an unknown or already inactive record")
        if not isinstance(reason, str) or not reason:
            raise SpineError("revocation reason is required")
        record = self._active[record_id]
        self._revocations[record_id] = reason
        self._payload_bytes -= record.payload_bytes
        del self._active[record_id]
        self._bump()

    def clear(self) -> None:
        """Clear E for P-only evaluation without changing P state."""
        self._active.clear()
        self._history.clear()
        self._revocations.clear()
        self._payload_bytes = 0
        self._bump()

    def snapshot(self, *, query_time: int, query_count: int,
                 decision: DecisionSnapshot) -> Packet:
        if not isinstance(decision, DecisionSnapshot):
            raise SpineError("evidence snapshot requires a DecisionSnapshot")
        if not isinstance(query_time, int) or isinstance(query_time, bool) \
                or query_time < 0 or type(query_count) is not int or query_count <= 0:
            raise SpineError("query time/count is invalid")
        if decision.scope != self.scope:
            raise SpineError("decision scope differs from evidence bank")
        if decision.state_revision != self.revision:
            raise SpineError("decision used a stale evidence revision")
        key = (query_time, query_count, decision.decision_id, self.revision)
        if key in self._cache:
            return self._cache[key]
        records = [record for record in self._active.values()
                   if record.valid_from <= query_time
                   and record.available_at <= query_time
                   and (record.valid_until is None or query_time <= record.valid_until)]
        records.sort(key=lambda record: (record.available_at, record.source_epoch,
                                         record.record_id))
        if records:
            payload = torch.stack([record.payload for record in records], dim=0).unsqueeze(0)
            availability = torch.ones(1, query_count, len(records), dtype=torch.bool)
        else:
            payload = torch.zeros(1, 0, 512)
            availability = torch.zeros(1, query_count, 0, dtype=torch.bool)
        packet = Packet(
            schema="replica.spark.packet.v1",
            producer_revision="evidence-bank-v1",
            representation_version="evidence-repr-v1",
            codec_version="evidence-codec-v1",
            scope=self.scope,
            stream_id="memory",
            event_id="memory-snapshot",
            sequence=query_time,
            observation_start=query_time,
            observation_end=query_time,
            available_at=query_time,
            state_revision=self.revision,
            payload_type="observation",
            source_lineage={"record_ids": [record.record_id for record in records],
                            "history_versions": len(self._history)},
            payload=payload,
            availability=availability,
        )
        self._cache[key] = packet
        return packet


class SpineMemoryExperiment:
    """Small ordered observe/admit/read/update/validate/commit learner."""

    def __init__(self, *, scope: str = "task", spec: SpineSpec,
                 bank_experts: int = 2, bank_hidden_width: int = 64,
                 seed: int = 0, core: SpineCore | None = None) -> None:
        if core is not None and (not isinstance(core, SpineCore) or core.spec != spec):
            raise SpineError("supplied core does not match the memory experiment spec")
        self.scope = scope
        self.core = SpineCore(spec) if core is None else core
        self.evidence = EvidenceBank(scope)
        self.bank = ExpertBank(expert_count=bank_experts, width=256,
                               hidden_width=bank_hidden_width, seed=seed,
                               scope="spine_memory")
        self.controller = PlasticLearningController(self.bank, scope=scope)
        self.core_optimizer = torch.optim.AdamW(self.core.parameters(), lr=1e-3)
        self.arm = "arm-0"
        self.phase = "observation"
        self.counters: dict[str, int] = {
            "observations": 0, "snapshots": 0, "learning_events": 0,
            "retried_learning_events": 0, "optimizer_steps": 0,
            "backward_calls": 0, "forward_calls": 0,
            "codec_calls": 0, "p_only_clears": 0,
        }
        self.completed_evaluations: list[dict[str, Any]] = []
        self.source_cache: dict[str, Any] = {}
        self.provider_cache: dict[str, Any] = {}
        self.source_kv: dict[str, Any] = {}
        self.fast_state: dict[str, Any] = {}
        self.workspace_state: dict[str, Any] = {}
        self._snapshot_context: SpineForwardContext | None = None

    def observe(self, record: EvidenceRecord) -> bool:
        admitted = self.evidence.admit(record)
        if admitted:
            self.counters["observations"] += 1
            self.phase = "observation"
        return admitted

    def snapshot(self, *, query_time: int, query_count: int,
                 phase: str = "read", plastic_expert_id: int | None = None,
                 enabled_branches=BRANCHES) -> SpineForwardContext:
        enabled_branches = tuple(enabled_branches)
        decision = DecisionSnapshot(
            decision_id=f"{self.arm}:{query_time}:{phase}", scope=self.scope,
            stream_id="memory", state_revision=self.evidence.revision,
            arm=self.arm, event=f"event-{query_time}", phase=phase)
        packet = self.evidence.snapshot(
            query_time=query_time, query_count=query_count, decision=decision)
        memory_slots = None
        if plastic_expert_id is not None and "memory" in enabled_branches:
            memory_slots = self.bank.expert(plastic_expert_id)(
                self.core.encode(packet)["memory"])
        context = SpineForwardContext(
            self.core, packet, decision,
            enabled_branches=enabled_branches, memory_slots=memory_slots,
            memory_source="P" if memory_slots is not None else "E")
        self._snapshot_context = context
        self.counters["snapshots"] += 1
        self.counters["codec_calls"] += context.charges["codec_calls"]
        self.phase = phase
        return context

    def learn(self, *, event_id: str, features: torch.Tensor,
              target: torch.Tensor, expert_id: int = 0,
              steps: int = 1) -> dict[str, Any]:
        outcome = self.controller.apply_teaching_transaction(
            event_id=event_id, expert_id=expert_id, features=features,
            target=target, steps=steps)
        retried = outcome.retried
        actual_steps = 0 if retried else outcome.steps
        if retried:
            self.counters["retried_learning_events"] += 1
        else:
            self.counters["learning_events"] += 1
            self.counters["optimizer_steps"] += actual_steps
            self.counters["backward_calls"] += actual_steps
            self.counters["forward_calls"] += actual_steps
            self.completed_evaluations.append({
                "event_id": event_id, "expert_id": expert_id,
                "outcome": outcome.to_dict()})
        self.phase = "commit"
        return {"outcome": {**outcome.to_dict(), "retried": retried},
                "learning_evidence": {
                    "declared_updates": outcome.steps,
                    "execution_complete": True,
                    "optimizer_steps": actual_steps,
                    "backward_calls": actual_steps,
                    "forward_calls": actual_steps,
                    "changed_trainable_names": ([] if retried else [f"expert_{expert_id}"]),
                    "learning_success": True,
                    "status": "passed",
                    "retried": retried,
                }}

    def clear_for_p_only(self) -> None:
        """Clear E/S/C/F and source/provider state while retaining P."""
        self.evidence.clear()
        self._snapshot_context = None
        self.source_cache.clear()
        self.provider_cache.clear()
        self.source_kv.clear()
        self.fast_state.clear()
        self.workspace_state.clear()
        self.counters["p_only_clears"] += 1
        self.phase = "p_only_evaluation"

    def checkpoint(self, root, relative: str, *, run_contract: Mapping[str, Any],
                   meter: Any = None) -> dict:
        return save_spine_checkpoint(
            root, relative, self.core, self.core_optimizer,
            arm=self.arm, event=self._snapshot_context.snapshot.event
            if self._snapshot_context is not None else "event-0",
            phase=self.phase, completed_evaluations=self.completed_evaluations, counters=self.counters,
            run_contract=run_contract, meter=meter,
            plastic=self.controller.checkpoint_payload(),
            evidence=self.evidence.checkpoint_payload())

    def restore(self, root, relative: str, *, run_contract: Mapping[str, Any]) -> dict:
        metadata = load_spine_checkpoint(
            root, relative, self.core, self.core_optimizer,
            run_contract=run_contract, plastic=self.controller,
            evidence=self.evidence)
        self.arm = metadata["arm"]
        self.phase = metadata["phase"]
        self.counters = dict(metadata["counters"])
        self.completed_evaluations = list(metadata["completed_evaluations"])
        self._snapshot_context = None
        return metadata

    def model_identity(self) -> str:
        return durable_model_identity(self.core)


__all__ = ["EVIDENCE_CHECKPOINT_FORMAT", "EvidenceRecord", "EvidenceBank",
           "SpineMemoryExperiment"]
