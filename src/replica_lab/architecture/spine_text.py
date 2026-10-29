"""Offline text observations, causal spine reads, and Spark response supervision.

The frozen Spark decoder is used only through :class:`Spark2_5Adapter`.  Exact
text/provenance (E), generated candidates, procedures, task workspaces, and
plastic parameters/optimizer/clock (P) have different owners.  In particular,
a generated explanation never becomes evidence merely by being generated.

``SpineTextQueryAdapter`` is the deliberately small seam around the public
``SpineMemoryExperiment.read`` query API.  It retains a local construction for
versions of the experiment which predate that API and for per-token evidence
availability, which the scalar-``query_time`` API does not yet express.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import weakref

import torch
from torch import nn

from .spark_adapter import Spark2_5Adapter
from .spine import (
    PACKET_SCHEMA, DecisionSnapshot, Packet, SpineError,
    SpineForwardContext,
)
from .spine_memory import EvidenceRecord, SpineMemoryExperiment
from ..optimizers import NativeFusedAdamW
from .. import artifact_publication as publication

TEXT_MEMORY_SCHEMA = "replica.spark.spine_text.v1"
TEXT_MEMORY_CHECKPOINT_FORMAT = "replica.spark.spine_text.checkpoint.v1"
TEXT_PRODUCER_REVISION = "spark-text-producer-v1"
TEXT_REPRESENTATION_VERSION = "unicode-token-hash-v1"
TEXT_CODEC_VERSION = "spine-ingestion-codec-v1"
_TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpineError(f"{name} must be a nonempty string")
    return value


def _time(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SpineError(f"{name} must be a nonnegative integer")
    return value


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strings(values: Iterable[str], name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    result = tuple(values)
    if not allow_empty and not result:
        raise SpineError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise SpineError(f"{name} entries must be nonempty strings")
    if len(set(result)) != len(result):
        raise SpineError(f"{name} entries must be unique")
    return result


@dataclass(frozen=True)
class TextSpan:
    """One exact Unicode token and its Python string offsets."""

    token: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class StoredText:
    """Canonical source text, or a durable locator capable of rebuilding it."""

    record_id: str
    text_sha256: str
    canonical_text: str | None
    durable_ref: Mapping[str, Any] | None
    provenance: Mapping[str, Any]
    spans: tuple[TextSpan, ...]
    producer_revision: str
    representation_version: str
    codec_version: str
    def __post_init__(self) -> None:
        _nonempty(self.record_id, "record_id")
        if len(self.text_sha256) != 64:
            raise SpineError("text_sha256 must be a SHA-256 digest")
        if self.canonical_text is None:
            if not isinstance(self.durable_ref, Mapping) \
                    or not self.durable_ref.get("locator") \
                    or self.durable_ref.get("text_sha256") != self.text_sha256:
                raise SpineError("source-free text requires a matching durable reference")
        elif _text_sha256(self.canonical_text) != self.text_sha256:
            raise SpineError("canonical text digest differs")
        if not isinstance(self.provenance, Mapping) \
                or self.provenance.get("text_sha256") != self.text_sha256:
            raise SpineError("text provenance digest differs")
        for name in ("producer_revision", "representation_version", "codec_version"):
            _nonempty(getattr(self, name), name)
        if self.canonical_text is not None and any(
                self.canonical_text[span.char_start:span.char_end] != span.token
                for span in self.spans):
            raise SpineError("canonical text span offsets differ")


@dataclass(frozen=True)
class ProducedText:
    record: EvidenceRecord
    stored: StoredText


class VersionedTextProducer:
    """Deterministic, offline Unicode text-to-width-512 observation producer.

    Hashing is SHA-256/BLAKE2 based rather than Python's randomized ``hash``.
    Token unigrams and adjacent bigrams provide a small semantic query key;
    the original text and spans, not this lossy vector, remain exact evidence.
    """

    def __init__(self, *, producer_revision: str = TEXT_PRODUCER_REVISION,
                 representation_version: str = TEXT_REPRESENTATION_VERSION,
                 codec_version: str = TEXT_CODEC_VERSION,
                 max_characters: int = 16384) -> None:
        self.producer_revision = _nonempty(producer_revision, "producer_revision")
        self.representation_version = _nonempty(
            representation_version, "representation_version")
        self.codec_version = _nonempty(codec_version, "codec_version")
        if not isinstance(max_characters, int) or isinstance(max_characters, bool) \
                or max_characters <= 0:
            raise SpineError("max_characters must be positive")
        self.max_characters = max_characters

    def spans(self, text: str) -> tuple[TextSpan, ...]:
        _nonempty(text, "text")
        if len(text) > self.max_characters:
            raise SpineError("text exceeds the producer character bound")
        return tuple(TextSpan(match.group(0), match.start(), match.end())
                     for match in _TOKEN.finditer(text))

    def vector(self, text: str) -> torch.Tensor:
        spans = self.spans(text)
        if not spans:
            raise SpineError("text must contain at least one Unicode token")
        terms = [(span.token, 1.0) for span in spans]
        terms.extend((f"{left.token}\u0001{right.token}", 0.5)
                     for left, right in zip(spans, spans[1:]))
        vector = torch.zeros(512, dtype=torch.float32)
        version = self.representation_version.encode("utf-8")
        for term, weight in terms:
            digest = hashlib.blake2b(
                term.encode("utf-8"), key=version[:64], digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % 512
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * weight
        return vector / vector.norm().clamp_min(1e-8)

    def produce(self, text: str, *, record_id: str, scope: str,
                source_epoch: int, available_at: int, valid_from: int,
                source_lineage: Mapping[str, Any],
                valid_until: int | None = None,
                durable_ref: Mapping[str, Any] | None = None,
                retain_canonical_text: bool = True,
                payload_type: str = "observation", verified: bool = False,
                supersedes: str | None = None) -> ProducedText:
        _nonempty(record_id, "record_id")
        _nonempty(scope, "scope")
        if not isinstance(source_lineage, Mapping) or not source_lineage:
            raise SpineError("source_lineage must be a nonempty mapping")
        if not retain_canonical_text:
            if not isinstance(durable_ref, Mapping) \
                    or not isinstance(durable_ref.get("locator"), str) \
                    or not durable_ref["locator"]:
                raise SpineError(
                    "dropping canonical text requires a durable_ref locator")
            if durable_ref.get("text_sha256") != _text_sha256(text):
                raise SpineError("durable_ref text digest differs from the observation")
        spans = self.spans(text)
        digest = _text_sha256(text)
        provenance = dict(source_lineage)
        provenance.update({
            "text_sha256": digest,
            "producer_revision": self.producer_revision,
            "representation_version": self.representation_version,
            "canonical_text_retained": bool(retain_canonical_text),
        })
        if durable_ref is not None:
            provenance["durable_ref"] = dict(durable_ref)
        vector = self.vector(text)
        record = EvidenceRecord(
            record_id=record_id, scope=scope, payload=vector,
            source_epoch=_time(source_epoch, "source_epoch"),
            available_at=_time(available_at, "available_at"),
            valid_from=_time(valid_from, "valid_from"), valid_until=valid_until,
            producer_revision=self.producer_revision,
            representation_version=self.representation_version,
            codec_version=self.codec_version, payload_type=payload_type,
            source_lineage=provenance, supersedes=supersedes)
        stored = StoredText(
            record_id=record_id, text_sha256=digest,
            canonical_text=text if retain_canonical_text else None,
            durable_ref=None if durable_ref is None else dict(durable_ref),
            provenance=provenance, spans=spans,
            producer_revision=self.producer_revision,
            representation_version=self.representation_version,
            codec_version=self.codec_version)
        return ProducedText(record, stored)


@dataclass(frozen=True)
class ProcedureMemory:
    procedure_id: str
    applicability: tuple[str, ...]
    prerequisites: tuple[str, ...]
    exceptions: tuple[str, ...]
    outcome: str
    outcome_independently_checked: bool
    outcome_checker: str | None
    source_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class TaskSubgoal:
    subgoal_id: str
    description: str


@dataclass(frozen=True)
class ObservedCompletion:
    subgoal_id: str
    observation: str
    observed_at: int
    checker: str
    evidence_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class TaskWorkspace:
    task_id: str
    subgoals: tuple[TaskSubgoal, ...]
    observed_completions: tuple[ObservedCompletion, ...] = ()


@dataclass(frozen=True)
class GeneratedExplanation:
    explanation_id: str
    text: str
    source_record_ids: tuple[str, ...]
    status: str = "candidate"
    verified: bool = False


@dataclass(frozen=True)
class TextQuery:
    """A completely prepared query; constructing it performs no injection."""

    query_id: str
    question: str
    question_sha256: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_available_at: tuple[int, ...]
    query_time: int
    query_features: torch.Tensor
    native_hidden: torch.Tensor
    context: SpineForwardContext
    memory_source: str
    eligible_record_ids: tuple[str, ...]
    selected_record_ids: tuple[str, ...]
    semantic_selection: str
    evidence_revision: int
    plastic_revision: int
    codec_calls_before_injection: int
    e_codec_calls_before_injection: int = 0
    p_read_calls_before_injection: int = 0
    p_codec_calls_before_injection: int = 0
    native_query_passes: int = 1


@dataclass(frozen=True)
class TextResponse:
    query: TextQuery
    logits: torch.Tensor
    loss: torch.Tensor | None
    injection_audit: Mapping[str, Any]
    null_logits: Mapping[str, torch.Tensor]
    charges: Mapping[str, int]


class SpineTextQueryAdapter:
    """Adapt text queries to public spine memory reads without gold routing.

    ``read`` is used for query-driven P when available.  E uses the public
    ``EvidenceBank.snapshot`` plus a local packet so host eligibility, semantic
    selection, and per-position late-arrival masks remain separate and visible.
    """

    def __init__(self, experiment: SpineMemoryExperiment,
                 producer: VersionedTextProducer) -> None:
        self.experiment = experiment
        self.producer = producer
        self._contexts: weakref.WeakSet[SpineForwardContext] = weakref.WeakSet()

    def invalidate(self) -> None:
        for context in tuple(self._contexts):
            context.invalidate()
        self._contexts.clear()

    def _register(self, context: SpineForwardContext) -> SpineForwardContext:
        self._contexts.add(context)
        return context

    def _decision(self, query_id: str, query_time: int) -> DecisionSnapshot:
        return DecisionSnapshot(
            decision_id=f"text:{query_id}", scope=self.experiment.scope,
            stream_id="memory", state_revision=self.experiment.evidence.revision,
            arm=self.experiment.arm, event=query_id, phase="text_query")


    def _read_p(self, *, query_id: str, features: torch.Tensor,
                native_hidden: torch.Tensor, query_time: int,
                decision: DecisionSnapshot, plastic_expert_id: int,
                branches: tuple[str, ...]) -> tuple[SpineForwardContext, int]:
        read = getattr(self.experiment, "read", None)
        used_public_read = callable(read)
        if used_public_read:
            context = read(
                query_features=features, native_hidden=native_hidden,
                query_time=query_time, plastic_expert_id=plastic_expert_id,
                enabled_branches=branches, phase="text_query")
        else:
            projected = self.experiment.core.memory_projection(features)
            port = self.experiment.core.ports["memory"][
                str(self.experiment.core.spec.sites[0])]
            projected = projected + port.query_projection(port.query_norm(native_hidden))
            memory_slots = self.experiment.bank.expert(plastic_expert_id)(
                projected).unsqueeze(2)
            batch, query_count = features.shape[:2]
            query_packet = Packet(
                schema=PACKET_SCHEMA,
                producer_revision=self.producer.producer_revision,
                representation_version=self.producer.representation_version,
                codec_version=self.producer.codec_version,
                scope=decision.scope, stream_id="memory", event_id=query_id,
                sequence=query_time, observation_start=0,
                observation_end=query_time, available_at=query_time,
                state_revision=decision.state_revision, payload_type="observation",
                source_lineage={"query_id": query_id,
                                "query_features": "current_question_only"},
                payload=torch.zeros(batch, 1, 512),
                availability=torch.ones(batch, query_count, 1, dtype=torch.bool))
            context = SpineForwardContext(
                self.experiment.core, query_packet, decision,
                enabled_branches=branches, memory_slots=memory_slots,
                memory_source="P")
        extra_codec = 0 if used_public_read else int(features.shape[1])
        codec_calls = int(context.charges["codec_calls"]) + extra_codec
        return context, codec_calls

    def build(self, *, query_id: str, question: str, input_ids: torch.Tensor,
              attention_mask: torch.Tensor, token_available_at: Sequence[int],
              query_time: int, native_hidden: torch.Tensor,
              memory_source: str, plastic_expert_id: int | None,
              enabled_branches: Iterable[str],
              host_eligible_record_ids: Iterable[str] | None,
              semantic_top_k: int | None,
              selected_record_ids: Iterable[str] | None = None) -> TextQuery:
        branches = tuple(enabled_branches)
        if memory_source not in {"E", "P", "E_plus_P"}:
            raise SpineError("memory_source must be E, P, or E_plus_P")
        if memory_source in {"P", "E_plus_P"} and plastic_expert_id is None:
            raise SpineError("P-bearing reads require an explicit plastic_expert_id")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise SpineError("input_ids must be a nonempty [batch, tokens] tensor")
        if attention_mask.shape != input_ids.shape:
            raise SpineError("attention_mask must match input_ids")
        batch, query_count = input_ids.shape
        times = tuple(_time(value, "token_available_at") for value in token_available_at)
        if len(times) != query_count or max(times) != query_time:
            raise SpineError("every question token needs a time ending at query_time")
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            raise SpineError("question token availability must be nondecreasing")
        if tuple(native_hidden.shape[:2]) != (batch, query_count):
            raise SpineError("native_hidden batch/query dimensions differ from input_ids")
        native_hidden = native_hidden.detach().to(
            device=self.experiment.core.memory_projection.weight.device,
            dtype=torch.float32)
        query_vector = self.producer.vector(question)
        features = query_vector.view(1, 1, 512).expand(batch, query_count, -1)
        decision = self._decision(query_id, query_time)

        if memory_source == "P":
            context, p_codec_calls = self._read_p(
                query_id=query_id, features=features, native_hidden=native_hidden,
                query_time=query_time, decision=decision,
                plastic_expert_id=plastic_expert_id, branches=branches)
            selected = eligible = ()
            semantic = "query_driven_P_no_evidence_selection"
            e_codec_calls = 0
            p_read_calls = 1
            codec_calls = p_codec_calls
        else:
            base = self.experiment.evidence.snapshot(
                query_time=query_time, query_count=1, decision=decision)
            by_id = {record.record_id: record
                     for record in self.experiment.evidence.history}
            active = getattr(self.experiment.evidence, "active", None)
            if active is None:
                record_ids = tuple(base.source_lineage["record_ids"])
            else:
                eligible_now = [record for record in active
                                if record.valid_from <= query_time
                                and record.available_at <= query_time
                                and (record.valid_until is None
                                     or query_time <= record.valid_until)]
                eligible_now.sort(key=lambda record: (
                    record.available_at, record.source_epoch, record.record_id))
                record_ids = tuple(record.record_id for record in eligible_now)
                by_id = {record.record_id: record for record in eligible_now}
            host_set = (None if host_eligible_record_ids is None
                        else frozenset(host_eligible_record_ids))
            host = record_ids if host_set is None else tuple(
                record_id for record_id in record_ids if record_id in host_set)
            read_cap = self.experiment.core.spec.slots
            declared_cap = getattr(self.experiment, "read_records", read_cap)
            read_cap = min(read_cap, declared_cap)
            if semantic_top_k is not None and (
                    not isinstance(semantic_top_k, int)
                    or isinstance(semantic_top_k, bool)
                    or not 1 <= semantic_top_k <= read_cap):
                raise SpineError("semantic_top_k must be within the declared read cap")
            effective_top_k = read_cap if semantic_top_k is None else semantic_top_k
            if selected_record_ids is not None:
                host = tuple(selected_record_ids)
                if (len(host) > read_cap or len(set(host)) != len(host)
                        or any(record_id not in record_ids for record_id in host)
                        or (host_set is not None
                            and any(record_id not in host_set for record_id in host))):
                    raise SpineError("ordered scoped selection exceeds eligible read authority")
                selected = host
                semantic = "tiered_scoped_ordered_selection"
            else:
                scored = []
                for record_id in host:
                    payload = by_id[record_id].payload
                    score = float(torch.dot(payload, query_vector) /
                                  (payload.norm() * query_vector.norm()).clamp_min(1e-8))
                    scored.append((-score, record_id))
                ranked = {record_id for _, record_id in
                          sorted(scored)[:effective_top_k]}
                selected = tuple(record_id for record_id in host if record_id in ranked)
                semantic = f"cosine_top_{effective_top_k}_within_host_eligible"
            eligible = tuple(host)
            if selected:
                payload = torch.stack([by_id[record_id].payload
                                       for record_id in selected]).unsqueeze(0)
                payload = payload.expand(batch, -1, -1)
            else:
                payload = torch.zeros(batch, 0, 512)
            availability = torch.zeros(
                batch, query_count, len(selected), dtype=torch.bool)
            for slot, record_id in enumerate(selected):
                record = by_id[record_id]
                for position, available_at in enumerate(times):
                    availability[:, position, slot] = bool(
                        record.valid_from <= available_at
                        and record.available_at <= available_at
                        and (record.valid_until is None
                             or available_at <= record.valid_until))
            packet = replace(
                base, payload=payload, availability=availability,
                event_id=query_id,
                source_lineage={
                    "record_ids": list(selected),
                    "selected_records": [{
                        "record_id": record_id,
                        "producer_revision": by_id[record_id].producer_revision,
                        "representation_version": by_id[record_id].representation_version,
                        "codec_version": by_id[record_id].codec_version,
                        "source_epoch": by_id[record_id].source_epoch,
                        "available_at": by_id[record_id].available_at,
                        "valid_from": by_id[record_id].valid_from,
                        "valid_until": by_id[record_id].valid_until,
                        "supersedes": by_id[record_id].supersedes,
                        "source_lineage": dict(by_id[record_id].source_lineage),
                    } for record_id in selected],
                    "history_versions": base.source_lineage.get("history_versions", 0),
                    "eligible_record_count": len(record_ids),
                    "read_record_count": len(selected),
                    "omitted_record_count": len(record_ids) - len(selected),
                    "query_id": query_id,
                    "question_sha256": _text_sha256(question),
                    "host_eligible_record_ids": list(eligible),
                    "selected_record_ids": list(selected),
                    "semantic_selection": semantic})
            if memory_source == "E_plus_P":
                p_context, p_codec_calls = self._read_p(
                    query_id=query_id, features=features, native_hidden=native_hidden,
                    query_time=query_time, decision=decision,
                    plastic_expert_id=plastic_expert_id, branches=branches)
                context = SpineForwardContext(
                    self.experiment.core, packet, decision,
                    enabled_branches=branches,
                    memory_slots=p_context.slots["memory"],
                    memory_source="E_plus_P",
                    memory_availability=p_context.branch_availability["memory"])
                semantic = f"{semantic}+query_driven_P"
                p_read_calls = 1
            else:
                context = SpineForwardContext(
                    self.experiment.core, packet, decision,
                    enabled_branches=branches, memory_source="E")
                p_codec_calls = 0
                p_read_calls = 0
            e_codec_calls = int(context.charges["codec_calls"])
            codec_calls = e_codec_calls + p_codec_calls
        self._register(context)
        return TextQuery(
            query_id=query_id, question=question,
            question_sha256=_text_sha256(question), input_ids=input_ids,
            attention_mask=attention_mask, token_available_at=times,
            query_time=query_time, query_features=features,
            native_hidden=native_hidden, context=context,
            memory_source=memory_source, eligible_record_ids=eligible,
            selected_record_ids=selected, semantic_selection=semantic,
            evidence_revision=self.experiment.evidence.revision,
            plastic_revision=self.experiment.controller.revision,
            e_codec_calls_before_injection=e_codec_calls,
            p_read_calls_before_injection=p_read_calls,
            p_codec_calls_before_injection=p_codec_calls,
            codec_calls_before_injection=codec_calls)


class SparkTextMemoryPath:
    """Usable CPU text-to-memory-to-frozen-Spark-response composition."""

    def __init__(self, adapter: Spark2_5Adapter,
                 experiment: SpineMemoryExperiment, *,
                 producer: VersionedTextProducer | None = None,
                 response_lr: float = 1e-3) -> None:
        if not isinstance(adapter, Spark2_5Adapter) \
                or not isinstance(experiment, SpineMemoryExperiment):
            raise SpineError("adapter and experiment must be Spark spine owners")
        if adapter.extension.spine is not experiment.core:
            raise SpineError("experiment must own the adapter's exact SpineCore")
        self.adapter = adapter
        self.experiment = experiment
        self.producer = producer or VersionedTextProducer(
            representation_version=self.experiment.evidence.representation_version,
            codec_version=self.experiment.evidence.codec_version)
        self.queries = SpineTextQueryAdapter(experiment, self.producer)
        self.texts: dict[str, StoredText] = {}
        self.procedures: dict[str, ProcedureMemory] = {}
        self.tasks: dict[str, TaskWorkspace] = {}
        self.explanations: dict[str, GeneratedExplanation] = {}
        self.response_steps = 0
        self.response_parameter_updates = 0
        self._query_sequence = 0
        self.response_optimizer = NativeFusedAdamW(
            self._response_groups(response_lr), optimizer_name="adamw_fused")

    def _response_groups(self, lr: float) -> list[dict[str, Any]]:
        groups: dict[str, list[tuple[str, nn.Parameter]]] = {}
        for name, parameter in self.adapter.new_named_parameters():
            if not name.startswith("extension.spine.") or not parameter.requires_grad:
                continue
            if ".codec." in name:
                component = "codec"
            elif ".ports." in name:
                component = "ports"
            elif ".memory_projection." in name:
                component = "memory"
            else:
                component = "query"
            groups.setdefault(component, []).append((name, parameter))
        if not groups:
            raise SpineError("no trainable spine response parameters")
        return [{
            "params": [parameter for _, parameter in pairs],
            "param_names": [name for name, _ in pairs],
            "group_name": component, "scope": f"text_response/{component}",
            "algorithm": "adamw_fused", "fused": True,
            "lr": lr, "betas": (0.9, 0.999), "eps": 1e-8,
            "weight_decay": 0.0,
        } for component, pairs in sorted(groups.items())]

    def observe_text(self, text: str, *, record_id: str,
                     source_epoch: int = 0, available_at: int = 0,
                     valid_from: int = 0, valid_until: int | None = None,
                     source_lineage: Mapping[str, Any] | None = None,
                     durable_ref: Mapping[str, Any] | None = None,
                     retain_canonical_text: bool = True,
                     payload_type: str = "observation", verified: bool = False,
                     supersedes: str | None = None, reconcile: bool = False) -> StoredText:
        produced = self.producer.produce(
            text, record_id=record_id, scope=self.experiment.scope,
            source_epoch=source_epoch, available_at=available_at,
            valid_from=valid_from, valid_until=valid_until,
            source_lineage=source_lineage or {"source": "direct_text_observation"},
            durable_ref=durable_ref, retain_canonical_text=retain_canonical_text,
            payload_type=payload_type, verified=verified, supersedes=supersedes)
        admitted = self.experiment.observe(produced.record, reconcile=reconcile)
        if admitted:
            if supersedes is not None:
                self.texts.pop(supersedes, None)
            self.texts[record_id] = produced.stored
            self.queries.invalidate()
        return produced.stored

    def add_procedure(self, procedure_id: str, *, applicability: Iterable[str],
                      prerequisites: Iterable[str], exceptions: Iterable[str],
                      outcome: str, outcome_independently_checked: bool,
                      outcome_checker: str | None,
                      source_record_ids: Iterable[str]) -> ProcedureMemory:
        _nonempty(procedure_id, "procedure_id")
        sources = _strings(source_record_ids, "source_record_ids", allow_empty=False)
        if any(record_id not in self.texts for record_id in sources):
            raise SpineError("procedure sources must name retained exact evidence")
        checked = outcome_independently_checked
        if not isinstance(checked, bool):
            raise SpineError("outcome_independently_checked must be boolean")
        if checked and (not outcome_checker or not sources):
            raise SpineError("checked procedure outcomes require an independent checker")
        procedure = ProcedureMemory(
            procedure_id=procedure_id,
            applicability=_strings(applicability, "applicability", allow_empty=False),
            prerequisites=_strings(prerequisites, "prerequisites"),
            exceptions=_strings(exceptions, "exceptions"),
            outcome=_nonempty(outcome, "outcome"),
            outcome_independently_checked=checked,
            outcome_checker=outcome_checker, source_record_ids=sources)
        self.procedures[procedure_id] = procedure
        return procedure

    def open_task(self, task_id: str,
                  subgoals: Iterable[tuple[str, str]]) -> TaskWorkspace:
        _nonempty(task_id, "task_id")
        items = tuple(subgoals)
        if not items:
            raise SpineError("a task requires at least one subgoal")
        ids = _strings((item[0] for item in items), "subgoal_ids", allow_empty=False)
        workspace = TaskWorkspace(task_id, tuple(
            TaskSubgoal(subgoal_id, _nonempty(description, "subgoal description"))
            for subgoal_id, description in zip(ids, (item[1] for item in items))))
        self.tasks[task_id] = workspace
        return workspace

    def observe_subgoal_completion(self, task_id: str, subgoal_id: str, *,
                                   observation: str, observed_at: int,
                                   checker: str,
                                   evidence_record_ids: Iterable[str]) -> TaskWorkspace:
        workspace = self.tasks[task_id]
        if subgoal_id not in {item.subgoal_id for item in workspace.subgoals}:
            raise SpineError("completion names an unknown subgoal")
        if subgoal_id in {item.subgoal_id for item in workspace.observed_completions}:
            raise SpineError("subgoal completion is already observed")
        sources = _strings(evidence_record_ids, "evidence_record_ids", allow_empty=False)
        if any(record_id not in self.texts for record_id in sources):
            raise SpineError("completion requires retained independent evidence")
        completion = ObservedCompletion(
            subgoal_id, _nonempty(observation, "observation"),
            _time(observed_at, "observed_at"), _nonempty(checker, "checker"), sources)
        updated = replace(workspace, observed_completions=(
            *workspace.observed_completions, completion))
        self.tasks[task_id] = updated
        return updated

    def add_generated_explanation(self, explanation_id: str, text: str,
                                  source_record_ids: Iterable[str]) -> GeneratedExplanation:
        explanation = GeneratedExplanation(
            _nonempty(explanation_id, "explanation_id"), _nonempty(text, "text"),
            _strings(source_record_ids, "source_record_ids"))
        self.explanations[explanation_id] = explanation
        return explanation

    def prepare_query(self, question: str, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor | None = None, *,
                      token_available_at: Sequence[int] | None = None,
                      query_time: int = 0, memory_source: str = "E",
                      plastic_expert_id: int | None = None,
                      enabled_branches: Iterable[str] = ("memory",),
                      host_eligible_record_ids: Iterable[str] | None = None,
                      semantic_top_k: int | None = None,
                      selected_record_ids: Iterable[str] | None = None) -> TextQuery:
        """Finish native query features and memory reads before any injection."""
        _nonempty(question, "question")
        query_time = _time(query_time, "query_time")
        attention_mask = (torch.ones_like(input_ids) if attention_mask is None
                          else attention_mask)
        times = (tuple(0 for _ in range(input_ids.shape[1]))
                 if token_available_at is None else token_available_at)
        if token_available_at is None:
            times = tuple(query_time for _ in times)
        with torch.no_grad():
            native_hidden = self.adapter.forward_hidden(
                input_ids, attention_mask, use_projector=False, use_ports=False,
                use_spine=False)
        query_id = f"query-{self._query_sequence}"
        self._query_sequence += 1
        return self.queries.build(
            query_id=query_id, question=question, input_ids=input_ids,
            attention_mask=attention_mask, token_available_at=times,
            query_time=query_time, native_hidden=native_hidden.detach(),
            memory_source=memory_source, plastic_expert_id=plastic_expert_id,
            enabled_branches=enabled_branches,
            host_eligible_record_ids=host_eligible_record_ids,
            semantic_top_k=semantic_top_k,
            selected_record_ids=selected_record_ids)

    def _assert_fresh(self, query: TextQuery) -> None:
        if query.context.invalidated \
                or query.evidence_revision != self.experiment.evidence.revision:
            raise SpineError("text query is stale")
        if query.memory_source in {"P", "E_plus_P"} \
                and query.plastic_revision != self.experiment.controller.revision:
            raise SpineError("text query used stale plastic state")

    def respond(self, query: TextQuery, *, labels: torch.Tensor | None = None,
                extra_passes: Iterable[str] = ()) -> TextResponse:
        """Run the response and explicitly requested, separately charged passes."""
        self._assert_fresh(query)
        extras = tuple(extra_passes)
        if len(set(extras)) != len(extras) or any(item != "null" for item in extras):
            raise SpineError("extra_passes may contain only unique 'null' passes")
        logits, audit = self.adapter.forward_text(
            query.input_ids, query.attention_mask, use_projector=False,
            use_ports=False, spine=query.context, use_spine=True,
            enabled_spine_branches=query.context.enabled_branches,
            return_audit=True)
        null_logits = {}
        for extra in extras:
            null_logits[extra] = self.adapter.forward_text(
                query.input_ids, query.attention_mask, use_projector=False,
                use_ports=False, spine=query.context, use_spine=True,
                enabled_spine_branches=())
        loss = (self.adapter.causal_loss(logits, query.input_ids, labels)
                if labels is not None else None)
        charges = {
            "native_query_passes": query.native_query_passes,
            "decoder_passes": 1 + len(extras),
            "charged_extra_passes": len(extras),
            "codec_calls": query.codec_calls_before_injection,
            "e_codec_calls": query.e_codec_calls_before_injection,
            "p_read_calls": query.p_read_calls_before_injection,
            "p_codec_calls": query.p_codec_calls_before_injection,
            "memory_read_slots": query.context.charges.get("memory_read_slots", 0),
            "e_memory_read_slots": query.context.charges.get("e_memory_read_slots", 0),
            "p_memory_read_slots": query.context.charges.get("p_memory_read_slots", 0),
            "port_calls": query.context.charges["port_calls"],
            "packet_count": query.context.charges["packet_count"],
            "packet_bytes": query.context.charges["packet_bytes"],
        }
        return TextResponse(query, logits, loss, audit.to_dict(), null_logits, charges)

    def response_step(self, query: TextQuery, *,
                      labels: torch.Tensor,
                      extra_passes: Iterable[str] = ()) -> dict[str, Any]:
        """Backpropagate frozen-decoder response loss; update only spine modules.

        P receives ordinary autograd gradients when used, but its selected-expert
        NativeFusedAdamW moments and teaching clock remain exclusively owned by
        ``PlasticLearningController`` and are not advanced here.
        """
        self.adapter.zero_grad(set_to_none=True)
        self.experiment.bank.zero_gradients()
        before = {name: parameter.detach().clone()
                  for name, parameter in self.adapter.new_named_parameters()
                  if name.startswith("extension.spine.")}
        response = self.respond(query, labels=labels, extra_passes=extra_passes)
        assert response.loss is not None
        response.loss.backward()
        gradient_audit = self.adapter.gradient_audit()
        p_gradient_names = [name for name, parameter in self.experiment.bank.named_parameters()
                            if parameter.grad is not None
                            and bool(torch.count_nonzero(parameter.grad))]
        self.response_optimizer.step()
        changed = [name for name, parameter in self.adapter.new_named_parameters()
                   if name in before and not torch.equal(before[name], parameter.detach())]
        actual_update = bool(changed)
        self.response_steps += 1
        self.response_parameter_updates += int(actual_update)
        if actual_update:
            self.queries.invalidate()
        learning_evidence = {
            "execution_complete": True,
            "physical_optimizer_steps": 1,
            "physical_optimizer_steps_total": self.response_steps,
            "actual_parameter_update": actual_update,
            "actual_parameter_updates_total": self.response_parameter_updates,
            "changed_trainable_names": changed,
            "changed_expert_names": [],
            "actual_expert_update": False,
            "train_fit": "unevaluated",
            "transfer": "unevaluated",
            "retention": "unevaluated",
            "useful_learning": "unevaluated",
            "learning_success": None,
        }
        return {
            "response": response, "loss": float(response.loss.detach()),
            "execution_complete": True,
            "physical_optimizer_steps": 1,
            "actual_parameter_update": actual_update,
            "changed_trainable_names": changed,
            "changed_expert_names": [],
            "train_fit": "unevaluated", "transfer": "unevaluated",
            "retention": "unevaluated", "useful_learning": "unevaluated",
            "learning_evidence": learning_evidence,
            "gradient_audit": gradient_audit,
            "plastic_nonzero_gradient_names": p_gradient_names,
            "plastic_optimizer_steps": list(self.experiment.bank.expert_steps),
            "plastic_clock_events": self.experiment.controller.clock.eligible_events,
            "base_unchanged": self.adapter.base_unchanged(),
            "charges": response.charges,
        }

    def plastic_features(self, query: TextQuery) -> torch.Tensor:
        """Detached query-driven P inputs, never E payloads or gold record IDs."""
        self._assert_fresh(query)
        if query.memory_source not in {"P", "E_plus_P"}:
            raise SpineError("plastic_features requires a P-bearing query")
        with torch.no_grad():
            projected = self.experiment.core.memory_projection(query.query_features)
            port = self.experiment.core.ports["memory"][
                str(self.experiment.core.spec.sites[0])]
            projected = projected + port.query_projection(
                port.query_norm(query.native_hidden))
            return projected.detach()

    def teach_query(self, query: TextQuery, *, event_id: str,
                    target: torch.Tensor, expert_id: int = 0,
                    steps: int = 1) -> dict[str, Any]:
        features = self.plastic_features(query)
        if target.shape != features.shape:
            raise SpineError("P target shape must match query-driven plastic features")
        outcome = self.experiment.learn(
            event_id=event_id, features=features, target=target,
            expert_id=expert_id, steps=steps)
        if not outcome["outcome"]["retried"]:
            self.queries.invalidate()
        return outcome

    def clear_for_p_only(self) -> None:
        """Remove every source-bearing owner while preserving P and its clock."""
        self.experiment.clear_for_p_only()
        self.texts.clear()
        self.procedures.clear()
        self.tasks.clear()
        self.explanations.clear()
        self.queries.invalidate()

    def _text_checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format": TEXT_MEMORY_CHECKPOINT_FORMAT,
            "producer": {
                "producer_revision": self.producer.producer_revision,
                "representation_version": self.producer.representation_version,
                "codec_version": self.producer.codec_version,
                "max_characters": self.producer.max_characters,
            },
            "texts": [asdict(self.texts[key]) for key in sorted(self.texts)],
            "procedures": [asdict(self.procedures[key])
                           for key in sorted(self.procedures)],
            "tasks": [asdict(self.tasks[key]) for key in sorted(self.tasks)],
            "explanations": [asdict(self.explanations[key])
                             for key in sorted(self.explanations)],
            "response_steps": self.response_steps,
            "response_parameter_updates": self.response_parameter_updates,
            "next_query_sequence": self._query_sequence,
            "response_optimizer_state": self.response_optimizer.state_dict(),
            "derived_caches_required": False,
        }

    def _decode_text_checkpoint(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "format", "producer", "texts", "procedures", "tasks", "explanations",
            "response_steps", "response_parameter_updates", "next_query_sequence",
            "response_optimizer_state", "derived_caches_required",
        }
        if not isinstance(payload, Mapping) or set(payload) != required \
                or payload["format"] != TEXT_MEMORY_CHECKPOINT_FORMAT:
            raise SpineError("text checkpoint keys/format differ")
        producer = {
            "producer_revision": self.producer.producer_revision,
            "representation_version": self.producer.representation_version,
            "codec_version": self.producer.codec_version,
            "max_characters": self.producer.max_characters,
        }
        if payload["producer"] != producer \
                or payload["derived_caches_required"] is not False:
            raise SpineError("text checkpoint producer/rebuild contract differs")
        if any(not isinstance(payload[name], list) for name in
               ("texts", "procedures", "tasks", "explanations")):
            raise SpineError("text checkpoint stores must be lists")
        steps = _time(payload["response_steps"], "response_steps")
        updates = _time(payload["response_parameter_updates"],
                        "response_parameter_updates")
        next_query = _time(payload["next_query_sequence"], "next_query_sequence")
        if updates > steps or not isinstance(payload["response_optimizer_state"], Mapping):
            raise SpineError("text checkpoint response state is invalid")

        texts = {}
        for row in payload["texts"]:
            stored = StoredText(
                record_id=row["record_id"], text_sha256=row["text_sha256"],
                canonical_text=row["canonical_text"], durable_ref=row["durable_ref"],
                provenance=row["provenance"],
                spans=tuple(TextSpan(**span) for span in row["spans"]),
                producer_revision=row["producer_revision"],
                representation_version=row["representation_version"],
                codec_version=row["codec_version"])
            if stored.record_id in texts:
                raise SpineError("duplicate text checkpoint record")
            texts[stored.record_id] = stored
        procedures = {}
        for row in payload["procedures"]:
            procedure = ProcedureMemory(
                procedure_id=row["procedure_id"],
                applicability=tuple(row["applicability"]),
                prerequisites=tuple(row["prerequisites"]),
                exceptions=tuple(row["exceptions"]), outcome=row["outcome"],
                outcome_independently_checked=row["outcome_independently_checked"],
                outcome_checker=row["outcome_checker"],
                source_record_ids=tuple(row["source_record_ids"]))
            if procedure.procedure_id in procedures \
                    or any(source not in texts for source in procedure.source_record_ids):
                raise SpineError("procedure checkpoint identity/source differs")
            if procedure.outcome_independently_checked and not procedure.outcome_checker:
                raise SpineError("checked procedure checkpoint lacks its checker")
            procedures[procedure.procedure_id] = procedure
        tasks = {}
        for row in payload["tasks"]:
            workspace = TaskWorkspace(
                task_id=row["task_id"],
                subgoals=tuple(TaskSubgoal(**item) for item in row["subgoals"]),
                observed_completions=tuple(ObservedCompletion(
                    subgoal_id=item["subgoal_id"], observation=item["observation"],
                    observed_at=item["observed_at"], checker=item["checker"],
                    evidence_record_ids=tuple(item["evidence_record_ids"]))
                    for item in row["observed_completions"]))
            subgoal_ids = {item.subgoal_id for item in workspace.subgoals}
            completions = workspace.observed_completions
            if workspace.task_id in tasks \
                    or len(subgoal_ids) != len(workspace.subgoals) \
                    or len({item.subgoal_id for item in completions}) != len(completions) \
                    or any(item.subgoal_id not in subgoal_ids for item in completions) \
                    or any(source not in texts for item in completions
                           for source in item.evidence_record_ids):
                raise SpineError("task checkpoint identity/completion differs")
            tasks[workspace.task_id] = workspace
        explanations = {}
        for row in payload["explanations"]:
            explanation = GeneratedExplanation(
                explanation_id=row["explanation_id"], text=row["text"],
                source_record_ids=tuple(row["source_record_ids"]),
                status=row["status"], verified=row["verified"])
            if explanation.explanation_id in explanations \
                    or explanation.status != "candidate" or explanation.verified:
                raise SpineError("generated explanation checkpoint is not a candidate")
            explanations[explanation.explanation_id] = explanation
        return {
            "texts": texts, "procedures": procedures, "tasks": tasks,
            "explanations": explanations, "response_steps": steps,
            "response_parameter_updates": updates, "next_query_sequence": next_query,
            "response_optimizer_state": payload["response_optimizer_state"],
        }

    def checkpoint(self, root, relative: str, *, run_contract: Mapping[str, Any],
                   meter: Any = None) -> dict[str, Any]:
        """Save experiment/P state plus a text/response sidecar, without caches."""
        experiment_receipt = self.experiment.checkpoint(
            root, relative, run_contract=run_contract, meter=meter)
        artifact = publication.write_torch(
            Path(root), f"{relative}.text.pt", self._text_checkpoint_payload(),
            meter=meter, reuse_identical=True)
        return {
            "experiment": experiment_receipt,
            "text_artifact": artifact.as_dict(),
            "derived_caches_saved": False,
        }

    def restore(self, root, relative: str, *,
                run_contract: Mapping[str, Any]) -> dict[str, Any]:
        """Restore a fresh path using durable records, never derived query caches."""
        payload = publication.load_torch(Path(root), f"{relative}.text.pt")
        decoded = self._decode_text_checkpoint(payload)
        metadata = self.experiment.restore(root, relative, run_contract=run_contract)
        self.response_optimizer.load_state_dict(decoded["response_optimizer_state"])
        active = {record.record_id: record for record in self.experiment.evidence.active}
        if set(active) != set(decoded["texts"]):
            raise SpineError("text and experiment evidence identities differ")
        for record_id, stored in decoded["texts"].items():
            record = active[record_id]
            if (stored.producer_revision, stored.representation_version,
                    stored.codec_version) != (
                        record.producer_revision, record.representation_version,
                        record.codec_version):
                raise SpineError("text and evidence producer versions differ")
        self.texts = decoded["texts"]
        self.procedures = decoded["procedures"]
        self.tasks = decoded["tasks"]
        self.explanations = decoded["explanations"]
        self.response_steps = decoded["response_steps"]
        self.response_parameter_updates = decoded["response_parameter_updates"]
        self._query_sequence = decoded["next_query_sequence"]
        self.queries.invalidate()
        return {
            "experiment": metadata,
            "text_records": len(self.texts), "procedures": len(self.procedures),
            "task_workspaces": len(self.tasks),
            "candidate_explanations": len(self.explanations),
            "derived_caches_restored": False,
        }

    def state_inventory(self) -> dict[str, Any]:
        """Owner counts and clocks; E/procedure/task/P states are not merged."""
        return {
            "schema": TEXT_MEMORY_SCHEMA,
            "exact_text_records": len(self.texts),
            "evidence_records": self.experiment.evidence.pending,
            "procedures": len(self.procedures),
            "task_workspaces": len(self.tasks),
            "candidate_explanations": len(self.explanations),
            "plastic_expert_steps": list(self.experiment.bank.expert_steps),
            "plastic_clock_events": self.experiment.controller.clock.eligible_events,
            "plastic_revision": self.experiment.controller.revision,
            "response_steps": self.response_steps,
            "response_parameter_updates": self.response_parameter_updates,
            "response_optimizer": self.response_optimizer.audit()["implementation"],
            "p_optimizer_is_distinct": self.response_optimizer
                is not self.experiment.bank.optimizer,
        }


__all__ = [
    "TEXT_MEMORY_SCHEMA", "TEXT_MEMORY_CHECKPOINT_FORMAT", "TEXT_PRODUCER_REVISION",
    "TEXT_REPRESENTATION_VERSION", "TEXT_CODEC_VERSION",
    "GeneratedExplanation", "ObservedCompletion", "ProcedureMemory",
    "ProducedText", "SparkTextMemoryPath", "SpineTextQueryAdapter",
    "StoredText", "TaskSubgoal", "TaskWorkspace", "TextQuery", "TextResponse",
    "TextSpan", "VersionedTextProducer",
]
