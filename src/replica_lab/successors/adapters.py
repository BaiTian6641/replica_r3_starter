"""CPU adapters from successor contracts to existing Replica/Core/Task interfaces.

Adapters are deliberately thin wrappers.  They do not edit the existing
CognitiveCore, TaskRuntime, WorkspaceStore or identity modules and they add no
authority of their own: the host still owns task success, permissions and the
formation transaction.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from ..architecture.contracts import CoreSpec, CoreState
from ..architecture.core import CognitiveCore
from ..replica.identity import IdentityContext, IdentityStore, render_identity_context
from ..replica.reader import IdentityReader
from ..runtime.workspace_store import WorkspaceStore, detach_to_cpu
from .chunks import tensor_digest
from .formation import (FormationRecord, FormationStore, IneligibleRecord,
                        InteractionProposal, MemoryPin, Observation, OutcomeReceipt,
                        ProvisionalNote, ScopeViolation)

CORE_ADAPTER_VERSION = "replica.successors.core_adapter.v1"
TASK_ADAPTER_VERSION = "replica.successors.task_adapter.v1"
IDENTITY_ADAPTER_VERSION = "replica.successors.identity_adapter.v1"


def _now_ns() -> int:
    return time.time_ns()


class CoreWorkspaceAdapter:
    """Build/save/reset one task-owned workspace through the existing core."""

    def __init__(self, spec: CoreSpec | None = None, *,
                 identity_reader: IdentityReader | None = None,
                 producer: str = "core-adapter-v1") -> None:
        self.spec = spec or CoreSpec()
        self.core = CognitiveCore(self.spec)
        self.identity_reader = identity_reader
        if not isinstance(producer, str) or not producer:
            raise ValueError("producer identity required")
        self.producer = producer
        self.previous: CoreState | None = None

    def build(self, sense: Tensor, sense_valid: Tensor, memory: Tensor,
              memory_valid: Tensor, task_features: Tensor, *, task_digest: str,
              revision: int, previous: CoreState | None = None,
              enabled_inputs: tuple[str, ...] = ("sense", "memory", "task"),
              training: bool = False, steps: int | None = None,
              identity_slots: Tensor | None = None,
              identity_valid: Tensor | None = None,
              identity_enabled: bool = True,
              implementation: str = "recurrent") -> CoreState:
        if not isinstance(task_digest, str) or not task_digest:
            raise ValueError("exact task digest required")
        if type(revision) is not int or revision < 0:
            raise ValueError("revision must be nonnegative")
        if implementation not in ("recurrent", "feedforward"):
            raise ValueError("unknown core implementation")
        prior = previous
        if prior is None:
            prior = self.previous
        prior_tensor = prior.workspace.detach() if prior is not None else None
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            workspace, valid = self.core(
                sense, sense_valid, memory, memory_valid, task_features,
                previous=prior_tensor, enabled_inputs=enabled_inputs, steps=steps)
            identity_digest = None
            if (self.identity_reader is not None and identity_slots is not None
                    and identity_enabled and identity_slots.shape[1] > 0):
                delta = self.identity_reader(
                    workspace, identity_slots, identity_valid, enabled=True)
                workspace = workspace + delta
                identity_digest = tensor_digest(identity_slots)
            if prior is not None:
                prior.assert_immutable()
        state = CoreState(
            workspace if training else workspace.detach(),
            valid if training else valid.detach(),
            task_features if training else task_features.detach(),
            self.producer, task_digest, revision,
            self.spec.steps if steps is None else steps, implementation,
            identity_digest)
        self.previous = state.detached()
        return state

    def build_condition(self, condition: str, sense: Tensor, sense_valid: Tensor,
                        memory: Tensor, memory_valid: Tensor, task_features: Tensor,
                        *, task_digest: str, revision: int,
                        previous: CoreState | None = None,
                        identity_slots: Tensor | None = None,
                        identity_valid: Tensor | None = None,
                        training: bool = False) -> CoreState:
        """Build one declared equal-information control condition.

        The same tensors, task digest and previous workspace are supplied to
        every condition; only the declared mask, workspace zeroing or the extra
        known-input inference pass differs.  No condition receives future
        observations, expected values or evaluator text.
        """
        conditions = {item["name"]: item
                      for item in self.equal_information_conditions()}
        if condition not in conditions:
            raise ValueError("unknown equal-information condition")
        config = conditions[condition]
        saved_previous = self.previous
        self.previous = None
        try:
            state = self.build(
                sense, sense_valid, memory, memory_valid, task_features,
                task_digest=task_digest, revision=revision, previous=previous,
                enabled_inputs=tuple(config["enabled_inputs"]), training=training,
                identity_slots=identity_slots, identity_valid=identity_valid,
                identity_enabled=bool(config["identity_enabled"]))
        finally:
            self.previous = saved_previous
        if condition == "workspace_zero":
            state = CoreState(
                state.workspace * 0, state.valid, state.task_features,
                state.producer, state.task_digest, state.revision, state.steps,
                state.implementation, state.identity_digest)
        if config.get("extra_inference"):
            with torch.no_grad():
                self.core(sense, sense_valid, memory, memory_valid,
                          task_features, enabled_inputs=config["enabled_inputs"])
        return state

    def save(self, store: WorkspaceStore, core: CoreState) -> None:
        if not isinstance(store, WorkspaceStore):
            raise TypeError("WorkspaceStore required")
        if not isinstance(core, CoreState):
            raise TypeError("CoreState required")
        store.save(detach_to_cpu(core))

    def load(self, store: WorkspaceStore, *, expected_task_digest: str,
             expected_revision: int,
             expected_identity_digest: str | None = None) -> CoreState | None:
        core = store.load(expected_task_digest=expected_task_digest,
                          expected_revision=expected_revision,
                          expected_identity_digest=expected_identity_digest)
        if core is not None:
            self.previous = core
        return core

    def reset_task(self, store: WorkspaceStore | None = None) -> None:
        """C-only reset: drop the task-owned latent workspace, retain I."""
        self.previous = None
        if store is not None:
            store.clear()

    @staticmethod
    def equal_information_conditions() -> tuple[dict, ...]:
        """Masks that change only the mechanism under test, never evidence."""
        return (
            {"name": "full", "enabled_inputs": ("sense", "memory", "task"),
             "identity_enabled": True},
            {"name": "memory_only", "enabled_inputs": ("memory",),
             "identity_enabled": False},
            {"name": "workspace_zero", "enabled_inputs": ("sense", "memory", "task"),
             "identity_enabled": True},
            {"name": "identity_null", "enabled_inputs": ("sense", "memory", "task"),
             "identity_enabled": False},
            {"name": "extra_inference", "enabled_inputs": ("sense", "memory", "task"),
             "identity_enabled": True, "extra_inference": True},
        )


class IdentityAdapter:
    """Resolve and render an approved identity context without granting authority."""

    def __init__(self, store: IdentityStore) -> None:
        if not isinstance(store, IdentityStore):
            raise TypeError("IdentityStore required")
        self.store = store
        self.context: IdentityContext | None = None

    def resolve(self, snapshot_id: str, *, mode: str, branch_id: str, purpose: str,
                cutoff: int, branch_state=None) -> IdentityContext:
        self.context = self.store.resolve(
            snapshot_id, mode=mode, branch_id=branch_id, purpose=purpose,
            cutoff=cutoff, branch_state=branch_state)
        return self.context

    def null(self) -> IdentityContext:
        self.context = IdentityContext.null()
        return self.context

    def render(self) -> str:
        return render_identity_context(self.context)

    @property
    def is_null(self) -> bool:
        return self.context is None or self.context.is_null

    def reset(self) -> None:
        """I-only reset: the null path cannot change task success authority."""
        self.null()

    @staticmethod
    def reader_delta(reader: IdentityReader, hidden: Tensor, slots: Tensor,
                     valid: Tensor | None = None, *, enabled: bool = True) -> Tensor:
        return reader(hidden, slots, valid, enabled=enabled)


@dataclass(frozen=True, slots=True)
class InteractionCommit:
    writer_enabled: bool
    report: object | None
    proposal: InteractionProposal

    def document(self) -> dict:
        return {"writer_enabled": self.writer_enabled,
                "status": None if self.report is None else getattr(self.report, "status", None),
                "receipt": (None if self.report is None else self.report.document()),
                "proposal_digest": self.proposal.digest}


class TaskRuntimeFormationAdapter:
    """Bind completed-interaction formation to an exact TaskRuntime instance.

    The adapter never reads future observations or expected values.  Only host
    outcome receipts become observations; generated raw text is a provisional
    branch note.  Task success authority remains in the runtime's verifier.
    """

    def __init__(self, runtime, formation: FormationStore, *, branch_id: str,
                 purpose: str, mode: str = "assistant_for_source",
                 writer_enabled: bool = True,
                 permitted_uses: Sequence[str] = ("retrieval",),
                 as_of_skew_ns: int = 3600 * 10 ** 9) -> None:
        if not isinstance(formation, FormationStore):
            raise TypeError("FormationStore required")
        for attribute in ("state", "journal", "workspace_store"):
            if not hasattr(runtime, attribute):
                raise TypeError("runtime does not expose the exact task interface")
        self.runtime = runtime
        self.formation = formation
        self.branch_id = branch_id
        self.purpose = purpose
        self.mode = mode
        self.writer_enabled = bool(writer_enabled)
        self.permitted_uses = tuple(permitted_uses)
        self.as_of_skew_ns = int(as_of_skew_ns)
        self._pin: MemoryPin | None = None
        self._baseline_outcomes: frozenset[str] = frozenset()
        self._baseline_revision = 0
        self._source_epoch = 0

    def begin_interaction(self) -> MemoryPin:
        if self._pin is not None:
            raise RuntimeError("one pin per interaction; finish before beginning another")
        state = self.runtime.state
        if state is None:
            raise RuntimeError("runtime has no initialized exact task state")
        self._pin = self.formation.pin(
            branch_id=self.branch_id, purpose=self.purpose, mode=self.mode,
            as_of=_now_ns() + self.as_of_skew_ns, task_id=state.task_id,
            task_revision=state.revision,
            permitted_uses=self.permitted_uses,
            include_verification_states=("verified", "unverified"))
        self._baseline_outcomes = frozenset(item.outcome_id for item in state.outcomes)
        self._baseline_revision = state.revision
        self._source_epoch = int(getattr(state, "source_epoch", 0))
        if self._source_epoch not in self._pin.source_epochs:
            self._clear_pin()
            raise ScopeViolation("runtime source epoch is not admitted by the formation pin")
        return self._pin

    def _clear_pin(self) -> None:
        self._pin = None
        self._baseline_outcomes = frozenset()
        self._baseline_revision = 0

    def finish_interaction(self, *, traces: Sequence[Mapping[str, object]] = ()
                           ) -> InteractionCommit:
        if self._pin is None:
            raise RuntimeError("finish requires an active pin")
        try:
            return self._finish_locked(traces=traces)
        finally:
            self._clear_pin()

    def _finish_locked(self, *, traces: Sequence[Mapping[str, object]] = ()
                       ) -> InteractionCommit:
        pin = self._pin
        state = self.runtime.state
        if state is None:
            raise RuntimeError("runtime state disappeared before completion")
        observed_at = _now_ns()
        observations: list[Observation] = []
        outcomes: list[OutcomeReceipt] = []
        for item in state.outcomes:
            if item.outcome_id in self._baseline_outcomes:
                continue
            receipt = dict(item.receipt or {})
            receipt_id = str(receipt.get("receipt_id") or item.outcome_id)
            summary = f"{item.action_id} {item.status}"
            if receipt:
                summary += f" receipt={receipt_id}"
            outcomes.append(OutcomeReceipt(
                outcome_id=item.outcome_id, action_id=item.action_id,
                status=item.status, receipt_id=receipt_id,
                observed_at=observed_at, summary=summary, revision=item.revision,
                source_epoch=self._source_epoch))
            observations.append(Observation(
                observation_id="obs_" + item.outcome_id,
                text=summary, source_id="task-runtime",
                source_epoch=self._source_epoch, observed_at=observed_at,
                receipt_id=receipt_id,
                verification_state="unverified", authenticity="host_observed",
                permitted_uses=tuple(self.permitted_uses)))
        notes: list[ProvisionalNote] = []
        for index, trace in enumerate(traces):
            raw = trace.get("raw_text") if isinstance(trace, Mapping) else None
            if isinstance(raw, str) and raw.strip():
                notes.append(ProvisionalNote(
                    note_id=f"note-{index}-" + str(trace.get("sequence", index)),
                    text=raw[:8192], origin="model_generated",
                    interaction_id=f"task-{pin.task_id}-{pin.authority_sequence}"))
        proposal = InteractionProposal(
            interaction_id=f"task-{pin.task_id}-{pin.authority_sequence}",
            subject_id=pin.subject_id, branch_id=pin.branch_id,
            purpose=pin.purpose, mode=pin.mode, task_id=pin.task_id,
            task_revision=pin.task_revision,
            decision_revision=pin.decision_revision + 1,
            source_epoch=self._source_epoch,
            producer_versions=pin.producer_versions,
            representation_version=pin.representation_version,
            codec_id=pin.codec_id, permitted_uses=pin.permitted_uses,
            observations=tuple(observations), outcomes=tuple(outcomes),
            notes=tuple(notes), created_at_ns=observed_at)
        report = None
        if self.writer_enabled and (observations or notes or outcomes):
            report = self.formation.commit_interaction(proposal, pin=pin)
        return InteractionCommit(self.writer_enabled, report, proposal)

    def run(self, proposer, max_decisions: int) -> dict:
        self.begin_interaction()
        try:
            result = self.runtime.run(proposer, max_decisions)
            completion = self.finish_interaction(
                traces=tuple(result.get("trace", ())))
            result["formation"] = completion.document()
            return result
        finally:
            self._clear_pin()

    # -- interventions -----------------------------------------------------

    def reset_task_workspace(self) -> None:
        """Reset C (latent workspace) while retaining identity and evidence."""
        if self.runtime.workspace_store is not None:
            self.runtime.workspace_store.clear()
        self.runtime.previous_workspace = None

    def reset_identity(self) -> None:
        """Null the identity context while retaining the exact task and evidence."""
        self.runtime.identity = IdentityContext.null()

    @staticmethod
    def intervention_conditions() -> tuple[str, ...]:
        return ("none", "task_reset", "identity_null", "task_then_identity",
                "identity_then_task", "both_reset")

    def apply_intervention(self, name: str) -> dict:
        if name not in self.intervention_conditions():
            raise ValueError("unknown intervention")
        order = {
            "none": (),
            "task_reset": ("task",),
            "identity_null": ("identity",),
            "task_then_identity": ("task", "identity"),
            "identity_then_task": ("identity", "task"),
            "both_reset": ("task", "identity"),
        }[name]
        operations: list[str] = []
        for step in order:
            if step == "task":
                self.reset_task_workspace()
                operations.append("task")
            else:
                self.reset_identity()
                operations.append("identity")
        return {"intervention": name, "order": list(operations),
                "identity_null": bool(self.runtime.identity is not None
                                      and self.runtime.identity.is_null),
                "task_workspace_cleared": "task" in operations}


def formation_authority(formation: FormationStore, pin: MemoryPin, *,
                        producer: str = "formation",
                        representation: str | None = None,
                        scope_digest: str | None = None,
                        cutoff: int | None = None):
    """Authority callback that also pins tier manifest identity and cutoff.

    The returned mapping carries expected producer/representation/scope/cutoff;
    tier reads validate them against the actual manifest after I/O and on cache
    hits, so authority-only reads cannot silently accept a different record
    identity or expired payload.
    """
    expected_representation = (representation
                               if representation is not None
                               else pin.representation_version)
    expected_scope = scope_digest if scope_digest is not None else pin.digest
    expected_cutoff = pin.cutoff if cutoff is None else cutoff

    def authority(record_id: str) -> Mapping[str, object] | None:
        try:
            record = formation.read_record(record_id, pin)
        except IneligibleRecord as error:
            return {"eligible": False, "reason": str(error)}
        return {"eligible": True, "generation": record.generation,
                "eligibility_revision": 0, "producer": producer,
                "representation": expected_representation,
                "scope_digest": expected_scope, "cutoff": expected_cutoff}
    return authority


def records_as_tier_payloads(records: Iterable[FormationRecord]) -> dict[str, bytes]:
    """Canonical payload bytes for tier publication of eligible records."""
    output = {}
    for record in records:
        output[record.record_id] = (record.text.encode("utf-8"))
    return output
