"""S2a completed-interaction formation: isolated CPU contract, version 1.

The host owns the transaction.  An interaction pins one immutable authority
generation before generation, accumulates actual observations and outcome
receipts after it, and commits once at the host-controlled completion boundary.
Model-generated explanations are provisional branch notes and never evidence.

Durability model (``replica.successors.s2a.v1``):

- payloads are content-addressed canonical JSON files written atomically and
  fsynced *before* authority publication;
- a single SQLite transaction inserts the event receipt, binds the predecessor
  receipt digest, advances the authority generation and sequence exactly once,
  and records correction/revocation tombstones;
- duplicate delivery of the same logical event with identical bytes returns the
  existing receipt; identical identity with different bytes is a conflict that
  fails closed and preserves the conflicting trajectory;
- the payload is authority; the record index is a derived cache that a rebuild
  may restore but can never use to invent a committed event.

The module deliberately does not import the native runtime, torch or the sealed
v3-v11 implementations.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Iterable, Iterator, Mapping, Sequence

FORMAT = "replica.successors.s2a.v1"
EVENT_FORMAT = "replica.successors.s2a.event.v1"
RECORD_FORMAT = "replica.successors.s2a.record.v1"
RECEIPT_FORMAT = "replica.successors.s2a.receipt.v1"
DB_FORMAT = "replica.successors.s2a.authority.v1"

MODES = ("source_snapshot", "source_current", "parallel_replica", "assistant_for_source")
VERIFICATION_STATES = ("verified", "unverified")
AUTHENTICITY_STATES = ("host_observed", "source_attested", "verified")
NOTE_ORIGINS = ("model_generated", "unverified_claim", "provisional_workspace")
MAX_TEXT_BYTES = 8192
MAX_ITEMS = 64

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127}$")


class FormationError(RuntimeError):
    """Base class for formation contract failures."""


class ScopeViolation(FormationError):
    """The proposal does not match the pinned scope, branch or purpose."""


class FormationConflict(FormationError):
    """Same logical event identity arrived with different bytes."""


class CorruptAuthority(FormationError):
    """The durable authority chain or payload bytes fail verification."""


class StalePin(FormationError):
    """The pin predates available authority in a way that cannot be committed."""


class IneligibleRecord(FormationError):
    """A selected record is not eligible under the current pin or tombstone."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_text(_canonical(value))


def _identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded identifier")


def _timestamp(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer nanosecond timestamp")


def _now_ns() -> int:
    return time.time_ns()


def _ordered_uses(values: Iterable[str]) -> tuple[str, ...]:
    items = tuple(values)
    if not items:
        raise ValueError("permitted uses cannot be empty")
    if len(set(items)) != len(items):
        raise ValueError("permitted uses must be distinct")
    for item in items:
        _identifier("permitted use", item)
    return tuple(sorted(items))


def _producer_versions(values: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("producer versions must be a nonempty mapping")
    items = tuple(sorted((str(key), str(value)) for key, value in values.items()))
    if len(dict(items)) != len(items):
        raise ValueError("producer versions must be distinct")
    for key, value in items:
        _identifier("producer component", key)
        _identifier("producer version", value)
    return items


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory)
        except OSError:
            pass
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class MemoryPin:
    """One read snapshot created before an interaction reads memory.

    Corrections are generation/cutoff gated for this snapshot. Privacy
    revocation is an immediate external barrier and invalidates it.
    """

    generation: int
    authority_sequence: int
    head_digest: str
    subject_id: str
    branch_id: str
    purpose: str
    mode: str
    as_of: int
    snapshot_at: int | None
    source_epochs: tuple[int, ...]
    producer_versions: tuple[tuple[str, str], ...]
    representation_version: str
    codec_id: str
    task_id: str
    task_revision: int
    decision_revision: int
    permitted_uses: tuple[str, ...]
    include_verification_states: tuple[str, ...]
    created_at_ns: int

    def __post_init__(self) -> None:
        _nonnegative = ("generation", "authority_sequence", "as_of", "task_revision",
                        "decision_revision", "created_at_ns")
        for name in _nonnegative:
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"pin {name} must be a nonnegative integer")
        for name in ("subject_id", "branch_id", "purpose", "task_id", "codec_id"):
            _identifier(name, getattr(self, name))
        if self.mode not in MODES:
            raise ValueError("unknown identity mode")
        if self.snapshot_at is not None:
            _timestamp("snapshot_at", self.snapshot_at)
            if self.mode != "source_snapshot":
                raise ValueError("snapshot_at is only valid for source_snapshot")
        elif self.mode == "source_snapshot":
            raise ValueError("source_snapshot requires snapshot_at")
        if not self.source_epochs:
            raise ValueError("a pin names at least one admitted source epoch")
        if any(type(item) is not int or item < 0 for item in self.source_epochs):
            raise ValueError("source epochs must be nonnegative integers")
        if not isinstance(self.producer_versions, tuple) or not self.producer_versions:
            raise ValueError("pin producer versions required")
        _identifier("representation_version", self.representation_version)
        if not isinstance(self.head_digest, str) or len(self.head_digest) != 64:
            raise ValueError("pin head digest must be a SHA256")
        if not self.permitted_uses:
            raise ValueError("pin permitted uses required")
        if not self.include_verification_states or any(
                item not in VERIFICATION_STATES for item in self.include_verification_states):
            raise ValueError("pin verification filter is invalid")

    @property
    def cutoff(self) -> int:
        return (min(self.as_of, self.snapshot_at)
                if self.snapshot_at is not None else self.as_of)

    def document(self) -> dict:
        value = {
            "format": FORMAT,
            "generation": self.generation,
            "authority_sequence": self.authority_sequence,
            "head_digest": self.head_digest,
            "subject_id": self.subject_id,
            "branch_id": self.branch_id,
            "purpose": self.purpose,
            "mode": self.mode,
            "as_of": self.as_of,
            "snapshot_at": self.snapshot_at,
            "source_epochs": list(self.source_epochs),
            "producer_versions": {key: value for key, value in self.producer_versions},
            "representation_version": self.representation_version,
            "codec_id": self.codec_id,
            "task_id": self.task_id,
            "task_revision": self.task_revision,
            "decision_revision": self.decision_revision,
            "permitted_uses": list(self.permitted_uses),
            "include_verification_states": list(self.include_verification_states),
            "created_at_ns": self.created_at_ns,
        }
        return value

    @property
    def digest(self) -> str:
        return _sha256_json(self.document())


@dataclass(frozen=True, slots=True)
class Observation:
    """An actual host-attributed observation; a receipt is mandatory.

    Model text belongs in :class:`ProvisionalNote`, not here.  ``verified`` adds
    an independent verifier receipt; ``unverified`` is still an actual observed
    event, published with its verification state preserved.
    """

    observation_id: str
    text: str
    source_id: str
    source_epoch: int
    observed_at: int
    receipt_id: str
    verification_state: str = "unverified"
    authenticity: str = "host_observed"
    verifier_receipt_id: str | None = None
    permitted_uses: tuple[str, ...] = ("retrieval",)
    event_at: int | None = None
    valid_from: int | None = None
    valid_to: int | None = None
    importance: int = 0
    critical: bool = False
    source_lineage: tuple[str, ...] = ()
    supersedes: str | None = None

    def __post_init__(self) -> None:
        _identifier("observation_id", self.observation_id)
        _identifier("source_id", self.source_id)
        _identifier("receipt_id", self.receipt_id)
        _timestamp("observed_at", self.observed_at)
        if self.event_at is None:
            object.__setattr__(self, "event_at", self.observed_at)
        else:
            _timestamp("event_at", self.event_at)
        if self.valid_from is None:
            object.__setattr__(self, "valid_from", self.observed_at)
        else:
            _timestamp("valid_from", self.valid_from)
        if self.valid_to is not None:
            _timestamp("valid_to", self.valid_to)
            if self.valid_to <= (self.valid_from or 0):
                raise ValueError("valid_to must follow valid_from")
        if type(self.source_epoch) is not int or self.source_epoch < 0:
            raise ValueError("source_epoch must be a nonnegative integer")
        if self.verification_state not in VERIFICATION_STATES:
            raise ValueError("unknown verification state")
        if self.authenticity not in AUTHENTICITY_STATES:
            raise ValueError("unknown authenticity state")
        if self.verification_state == "verified":
            if not self.verifier_receipt_id:
                raise ValueError("a verified observation requires a verifier receipt")
            _identifier("verifier_receipt_id", self.verifier_receipt_id)
        elif self.verifier_receipt_id is not None:
            raise ValueError("an unverified observation cannot name a verifier receipt")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("observation text must be a nonempty string")
        if len(self.text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("observation text exceeds the bounded span")
        object.__setattr__(self, "permitted_uses", _ordered_uses(self.permitted_uses))
        if type(self.importance) is not int or not 0 <= self.importance <= 100:
            raise ValueError("importance must be an integer in [0, 100]")
        if not isinstance(self.critical, bool):
            raise ValueError("critical must be boolean")
        for item in self.source_lineage:
            _identifier("source lineage item", item)
        if len(self.source_lineage) > MAX_ITEMS:
            raise ValueError("source lineage is bounded")
        object.__setattr__(self, "source_lineage", tuple(sorted(set(self.source_lineage))))
        if self.supersedes is not None:
            _identifier("supersedes", self.supersedes)
            if self.supersedes == self.observation_id:
                raise ValueError("an observation cannot supersede itself")

    @property
    def text_sha256(self) -> str:
        return _sha256_text(self.text)

    def document(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "source_id": self.source_id,
            "source_epoch": self.source_epoch,
            "observed_at": self.observed_at,
            "event_at": self.event_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "receipt_id": self.receipt_id,
            "verification_state": self.verification_state,
            "authenticity": self.authenticity,
            "verifier_receipt_id": self.verifier_receipt_id,
            "permitted_uses": list(self.permitted_uses),
            "importance": self.importance,
            "critical": self.critical,
            "source_lineage": list(self.source_lineage),
            "supersedes": self.supersedes,
        }


@dataclass(frozen=True, slots=True)
class ProvisionalNote:
    """Generated or unverified text owned by the branch, never E or S."""

    note_id: str
    text: str
    origin: str = "model_generated"
    interaction_id: str = "interaction"

    def __post_init__(self) -> None:
        _identifier("note_id", self.note_id)
        _identifier("interaction_id", self.interaction_id)
        if self.origin not in NOTE_ORIGINS:
            raise ValueError("unknown provisional note origin")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("provisional note text must be nonempty")
        if len(self.text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("provisional note exceeds the bounded span")

    def document(self) -> dict:
        return {"note_id": self.note_id, "text": self.text,
                "text_sha256": _sha256_text(self.text), "origin": self.origin,
                "interaction_id": self.interaction_id}


@dataclass(frozen=True, slots=True)
class OutcomeReceipt:
    """An actual outcome observation, separate from generated explanation."""

    outcome_id: str
    action_id: str
    status: str
    receipt_id: str
    observed_at: int
    summary: str
    revision: int = 0
    source_epoch: int = 0

    def __post_init__(self) -> None:
        _identifier("outcome_id", self.outcome_id)
        _identifier("action_id", self.action_id)
        _identifier("receipt_id", self.receipt_id)
        _timestamp("observed_at", self.observed_at)
        if self.status not in ("completed", "failed", "denied", "pending_input"):
            raise ValueError("unknown outcome status")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("outcome summary required")
        if len(self.summary.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("outcome summary exceeds the bounded span")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("outcome revision must be nonnegative")
        if type(self.source_epoch) is not int or self.source_epoch < 0:
            raise ValueError("outcome source_epoch must be nonnegative")

    def document(self) -> dict:
        return {"outcome_id": self.outcome_id, "action_id": self.action_id,
                "status": self.status, "receipt_id": self.receipt_id,
                "observed_at": self.observed_at, "summary": self.summary,
                "summary_sha256": _sha256_text(self.summary),
                "revision": self.revision, "source_epoch": self.source_epoch,
                "actual": True}


@dataclass(frozen=True, slots=True)
class InteractionProposal:
    """Bounded formation proposal for one completed interaction."""

    interaction_id: str
    subject_id: str
    branch_id: str
    purpose: str
    mode: str
    task_id: str
    task_revision: int
    decision_revision: int
    source_epoch: int
    producer_versions: tuple[tuple[str, str], ...]
    representation_version: str
    codec_id: str
    permitted_uses: tuple[str, ...]
    observations: tuple[Observation, ...] = ()
    outcomes: tuple[OutcomeReceipt, ...] = ()
    notes: tuple[ProvisionalNote, ...] = ()
    created_at_ns: int = field(default_factory=_now_ns)

    def __post_init__(self) -> None:
        _identifier("interaction_id", self.interaction_id)
        for name in ("subject_id", "branch_id", "purpose", "task_id", "codec_id"):
            _identifier(name, getattr(self, name))
        _identifier("representation_version", self.representation_version)
        if self.mode not in MODES:
            raise ValueError("unknown identity mode")
        for name in ("task_revision", "decision_revision", "created_at_ns"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.source_epoch) is not int or self.source_epoch < 0:
            raise ValueError("source_epoch must be a nonnegative integer")
        if not isinstance(self.producer_versions, tuple) or not self.producer_versions:
            raise ValueError("producer versions required")
        object.__setattr__(self, "permitted_uses", _ordered_uses(self.permitted_uses))
        ids = [item.observation_id for item in self.observations]
        if len(set(ids)) != len(ids):
            raise ValueError("observation IDs must be distinct")
        outcome_ids = [item.outcome_id for item in self.outcomes]
        if len(set(outcome_ids)) != len(outcome_ids):
            raise ValueError("outcome IDs must be distinct")
        note_ids = [item.note_id for item in self.notes]
        if len(set(note_ids)) != len(note_ids):
            raise ValueError("note IDs must be distinct")
        if any(not isinstance(item, Observation) for item in self.observations):
            raise TypeError("observations must be Observation instances")
        if any(not isinstance(item, OutcomeReceipt) for item in self.outcomes):
            raise TypeError("outcomes must be OutcomeReceipt instances")
        if any(not isinstance(item, ProvisionalNote) for item in self.notes):
            raise TypeError("notes must be ProvisionalNote instances")
        if len(self.observations) + len(self.outcomes) + len(self.notes) > MAX_ITEMS:
            raise ValueError("proposal item count exceeds the bounded maximum")

    def document(self) -> dict:
        return {
            "format": FORMAT,
            "interaction_id": self.interaction_id,
            "subject_id": self.subject_id,
            "branch_id": self.branch_id,
            "purpose": self.purpose,
            "mode": self.mode,
            "task_id": self.task_id,
            "task_revision": self.task_revision,
            "decision_revision": self.decision_revision,
            "source_epoch": self.source_epoch,
            "producer_versions": {key: value for key, value in self.producer_versions},
            "representation_version": self.representation_version,
            "codec_id": self.codec_id,
            "permitted_uses": list(self.permitted_uses),
            "observations": [item.document() for item in self.observations],
            "outcomes": [item.document() for item in self.outcomes],
            "notes": [item.document() for item in self.notes],
            "created_at_ns": self.created_at_ns,
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.document())

    @property
    def logical_key(self) -> str:
        """Stable event identity; retries reuse it, changed bytes conflict."""
        return "ev_" + _sha256_json({
            "format": FORMAT,
            "subject_id": self.subject_id,
            "branch_id": self.branch_id,
            "purpose": self.purpose,
            "mode": self.mode,
            "task_id": self.task_id,
            "task_revision": self.task_revision,
            "decision_revision": self.decision_revision,
            "interaction_id": self.interaction_id,
            "source_epoch": self.source_epoch,
            "codec_id": self.codec_id,
            "representation_version": self.representation_version,
        })[:48]


@dataclass(frozen=True, slots=True)
class FormationRecord:
    record_id: str
    kind: str
    text: str
    text_sha256: str
    subject_id: str
    branch_id: str
    purpose: str
    mode: str
    source_id: str
    source_epoch: int
    observed_at: int
    event_at: int
    valid_from: int
    valid_to: int | None
    verification_state: str
    authenticity: str
    receipt_id: str
    verifier_receipt_id: str | None
    permitted_uses: tuple[str, ...]
    importance: int
    critical: bool
    source_lineage: tuple[str, ...]
    supersedes: str | None
    interaction_id: str
    record_digest: str
    event_key: str
    generation: int
    sequence: int

    def document(self) -> dict:
        return {
            "record_id": self.record_id,
            "kind": self.kind,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "subject_id": self.subject_id,
            "branch_id": self.branch_id,
            "purpose": self.purpose,
            "mode": self.mode,
            "source_id": self.source_id,
            "source_epoch": self.source_epoch,
            "observed_at": self.observed_at,
            "event_at": self.event_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "verification_state": self.verification_state,
            "authenticity": self.authenticity,
            "receipt_id": self.receipt_id,
            "verifier_receipt_id": self.verifier_receipt_id,
            "permitted_uses": list(self.permitted_uses),
            "importance": self.importance,
            "critical": self.critical,
            "source_lineage": list(self.source_lineage),
            "supersedes": self.supersedes,
            "interaction_id": self.interaction_id,
            "record_digest": self.record_digest,
            "event_key": self.event_key,
            "generation": self.generation,
            "sequence": self.sequence,
        }


@dataclass(frozen=True, slots=True)
class FormationReceipt:
    event_key: str
    kind: str
    sequence: int
    generation: int
    predecessor_sequence: int
    predecessor_digest: str
    payload_digest: str
    record_ids: tuple[str, ...]
    record_digests: tuple[str, ...]
    note_ids: tuple[str, ...]
    outcome_ids: tuple[str, ...]
    pin_digest: str
    committed_at_ns: int
    affected_record_ids: tuple[str, ...] = ()
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        if not self.receipt_digest:
            document = {key: value for key, value in self.document().items()
                        if key != "receipt_digest"}
            object.__setattr__(self, "receipt_digest", _sha256_json(document))

    def document(self) -> dict:
        return {
            "format": RECEIPT_FORMAT,
            "event_key": self.event_key,
            "kind": self.kind,
            "sequence": self.sequence,
            "generation": self.generation,
            "predecessor_sequence": self.predecessor_sequence,
            "predecessor_digest": self.predecessor_digest,
            "payload_digest": self.payload_digest,
            "record_ids": list(self.record_ids),
            "record_digests": list(self.record_digests),
            "note_ids": list(self.note_ids),
            "outcome_ids": list(self.outcome_ids),
            "pin_digest": self.pin_digest,
            "committed_at_ns": self.committed_at_ns,
            "affected_record_ids": list(self.affected_record_ids),
            "receipt_digest": self.receipt_digest,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, object]) -> "FormationReceipt":
        fields = dict(value)
        fields.pop("format", None)
        for name in ("record_ids", "record_digests", "note_ids", "outcome_ids",
                     "affected_record_ids"):
            fields[name] = tuple(fields[name])
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class CommitReport:
    status: str  # committed | duplicate
    receipt: FormationReceipt
    prepared_at_ns: int

    def document(self) -> dict:
        return {"status": self.status, "receipt": self.receipt.document()}


@dataclass(frozen=True, slots=True)
class PreparedFormation:
    proposal: InteractionProposal
    pin: MemoryPin
    event_key: str
    payload_digest: str
    payload_path: Path
    record_ids: tuple[str, ...]
    record_digests: tuple[str, ...]
    prepared_at_ns: int


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    status: str
    events: int
    generation: int
    sequence: int
    head_digest: str
    orphan_payloads: int
    quarantined: tuple[str, ...]
    index_rebuilt: bool
    derived_errors: tuple[str, ...] = ()
    derived_rebuilt: bool = False

    def document(self) -> dict:
        return {"status": self.status, "events": self.events,
                "generation": self.generation, "sequence": self.sequence,
                "head_digest": self.head_digest,
                "orphan_payloads": self.orphan_payloads,
                "quarantined": list(self.quarantined),
                "index_rebuilt": self.index_rebuilt,
                "derived_errors": list(self.derived_errors),
                "derived_rebuilt": self.derived_rebuilt}


class FormationStore:
    """Durable staged authority for completed-interaction formation."""

    def __init__(self, root: Path | str, *, subject_id: str, source_epochs: Sequence[int],
                 producer_versions: Mapping[str, str], representation_version: str,
                 codec_id: str) -> None:
        _identifier("subject_id", subject_id)
        epochs = tuple(int(item) for item in source_epochs)
        if not epochs or any(item < 0 for item in epochs):
            raise ValueError("source_epochs must be nonempty nonnegative integers")
        producers = _producer_versions(producer_versions)
        _identifier("representation_version", representation_version)
        _identifier("codec_id", codec_id)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "payloads").mkdir(exist_ok=True)
        (self.root / "conflicts").mkdir(exist_ok=True)
        self._connection = sqlite3.connect(
            self.root / "authority.sqlite3", isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        identity = {
            "format": DB_FORMAT,
            "subject_id": subject_id,
            "source_epochs": list(epochs),
            "producer_versions": {key: value for key, value in producers},
            "representation_version": representation_version,
            "codec_id": codec_id,
        }
        digest = _sha256_json(identity)
        existing = {row["key"]: row["value"]
                    for row in self._connection.execute("SELECT key, value FROM meta")}
        if existing:
            if existing.get("identity") != _canonical(identity) or existing.get("identity_digest") != digest:
                raise CorruptAuthority("store identity differs from its durable authority")
        else:
            with self._transaction():
                for key, value in (("identity", _canonical(identity)),
                                   ("identity_digest", digest),
                                   ("format", DB_FORMAT)):
                    self._connection.execute("INSERT INTO meta(key, value) VALUES (?, ?)",
                                             (key, value))
                self._connection.execute(
                    "INSERT INTO authority(id, sequence, generation, head_digest) "
                    "VALUES (1, 0, 0, ?)", (digest,))
        self.subject_id = subject_id
        self.source_epochs = epochs
        self.producer_versions = producers
        self.representation_version = representation_version
        self.codec_id = codec_id
        self._closed = False
        self._metrics = {
            "commits": 0, "duplicates": 0, "conflicts": 0, "reads": 0,
            "bytes_written": 0, "bytes_read": 0, "failed_reads": 0,
            "commit_latency_ns": [], "read_latency_ns": [],
        }
        self._pending_recovery: tuple[bool, bool] = (False, False)
        self.recovery_report = self.recover()
        self._pending_recovery = (self.recovery_report.derived_rebuilt,
                                  self.recovery_report.index_rebuilt)

    # -- lifecycle ---------------------------------------------------------

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS authority(
                id INTEGER PRIMARY KEY CHECK(id=1), sequence INTEGER NOT NULL,
                generation INTEGER NOT NULL, head_digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
                event_key TEXT PRIMARY KEY, sequence INTEGER NOT NULL UNIQUE,
                generation INTEGER NOT NULL, kind TEXT NOT NULL,
                payload_digest TEXT NOT NULL, receipt_json TEXT NOT NULL,
                subject_id TEXT NOT NULL, branch_id TEXT NOT NULL,
                purpose TEXT NOT NULL, mode TEXT NOT NULL,
                source_epoch INTEGER NOT NULL, observed_at INTEGER NOT NULL,
                committed_at_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS record_index(
                record_id TEXT PRIMARY KEY, event_key TEXT NOT NULL,
                record_digest TEXT NOT NULL, kind TEXT NOT NULL,
                ordinal INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(event_key) REFERENCES events(event_key));
            CREATE TABLE IF NOT EXISTS record_lineage(
                record_id TEXT NOT NULL, ancestor_id TEXT NOT NULL,
                PRIMARY KEY(record_id, ancestor_id));
            CREATE TABLE IF NOT EXISTS tombstones(
                record_id TEXT PRIMARY KEY, kind TEXT NOT NULL, reason TEXT NOT NULL,
                event_key TEXT, at_ns INTEGER NOT NULL,
                observed_at INTEGER NOT NULL,
                generation INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS notes(
                note_id TEXT PRIMARY KEY, event_key TEXT NOT NULL, branch_id TEXT NOT NULL,
                purpose TEXT NOT NULL, interaction_id TEXT NOT NULL, origin TEXT NOT NULL,
                text TEXT NOT NULL, text_digest TEXT NOT NULL, committed_at_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS revocations(
                record_id TEXT PRIMARY KEY, event_key TEXT NOT NULL,
                payload_digest TEXT NOT NULL, receipt_json TEXT NOT NULL,
                at_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS conflicts(
                event_key TEXT NOT NULL, payload_digest TEXT NOT NULL,
                path TEXT NOT NULL, stored_at_ns INTEGER NOT NULL);
            """
        )
        self._ensure_column("record_index", "ordinal", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("tombstones", "generation", "INTEGER NOT NULL DEFAULT 0")

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        names = {row["name"] for row in self._connection.execute(
            f"PRAGMA table_info({table})")}
        if column not in names:
            self._connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def __enter__(self) -> "FormationStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    # -- authority ---------------------------------------------------------

    def _authority(self) -> sqlite3.Row:
        row = self._connection.execute("SELECT * FROM authority WHERE id=1").fetchone()
        if row is None:
            raise CorruptAuthority("authority row is missing")
        return row

    def pin(self, *, branch_id: str, purpose: str, mode: str, as_of: int,
            task_id: str, task_revision: int, decision_revision: int = 0,
            snapshot_at: int | None = None,
            permitted_uses: Sequence[str] = ("retrieval",),
            include_verification_states: Sequence[str] = ("verified",),
            ) -> MemoryPin:
        row = self._authority()
        return MemoryPin(
            generation=int(row["generation"]),
            authority_sequence=int(row["sequence"]),
            head_digest=str(row["head_digest"]),
            subject_id=self.subject_id,
            branch_id=branch_id,
            purpose=purpose,
            mode=mode,
            as_of=as_of,
            snapshot_at=snapshot_at,
            source_epochs=self.source_epochs,
            producer_versions=self.producer_versions,
            representation_version=self.representation_version,
            codec_id=self.codec_id,
            task_id=task_id,
            task_revision=task_revision,
            decision_revision=decision_revision,
            permitted_uses=_ordered_uses(permitted_uses),
            include_verification_states=tuple(include_verification_states),
            created_at_ns=_now_ns(),
        )

    def _check_pin(self, pin: MemoryPin) -> None:
        if not isinstance(pin, MemoryPin):
            raise TypeError("pin must be a MemoryPin")
        if (pin.subject_id != self.subject_id
                or tuple(pin.producer_versions) != self.producer_versions
                or pin.representation_version != self.representation_version
                or pin.codec_id != self.codec_id
                or tuple(pin.source_epochs) != self.source_epochs):
            raise ScopeViolation("pin does not belong to this store identity")
        authority = self._authority()
        if pin.generation > int(authority["generation"]):
            raise StalePin("pin generation is ahead of durable authority")
        if pin.generation == int(authority["generation"]) and \
                pin.head_digest != str(authority["head_digest"]):
            raise CorruptAuthority("pin head digest differs from the authority")

    def _validate_proposal(self, proposal: InteractionProposal, pin: MemoryPin) -> None:
        if (proposal.subject_id != pin.subject_id
                or proposal.branch_id != pin.branch_id
                or proposal.purpose != pin.purpose
                or proposal.mode != pin.mode):
            raise ScopeViolation("proposal scope/branch/purpose differs from the pin")
        if proposal.task_id != pin.task_id or proposal.task_revision != pin.task_revision:
            raise ScopeViolation("proposal task binding differs from the pin")
        if proposal.decision_revision < pin.decision_revision:
            raise ScopeViolation("proposal decision revision regressed")
        if proposal.source_epoch not in pin.source_epochs:
            raise ScopeViolation("proposal source epoch was not admitted by the pin")
        if proposal.producer_versions != pin.producer_versions:
            raise ScopeViolation("producer versions differ from the pin")
        if proposal.representation_version != pin.representation_version:
            raise ScopeViolation("representation version differs from the pin")
        if proposal.codec_id != pin.codec_id:
            raise ScopeViolation("codec identity differs from the pin")
        if not set(proposal.permitted_uses) <= set(pin.permitted_uses):
            raise ScopeViolation("proposal requests a use outside the pin")
        now = _now_ns()
        for observation in proposal.observations:
            if observation.observed_at < pin.created_at_ns:
                raise ScopeViolation("observation predates the immutable read pin")
            if observation.observed_at > now:
                raise ScopeViolation("observation time is in the future")
            if observation.source_epoch != proposal.source_epoch:
                raise ScopeViolation("observation source epoch differs from the proposal")
            if not set(observation.permitted_uses) <= set(pin.permitted_uses):
                raise ScopeViolation("observation permitted use exceeds the pin")
        for outcome in proposal.outcomes:
            if outcome.observed_at < pin.created_at_ns:
                raise ScopeViolation("outcome predates the immutable read pin")
            if outcome.observed_at > now:
                raise ScopeViolation("outcome time is in the future")
            if outcome.source_epoch != proposal.source_epoch:
                raise ScopeViolation("outcome source epoch differs from the proposal")

    def _build_payload(self, proposal: InteractionProposal, pin: MemoryPin,
                       ) -> tuple[dict, tuple[str, ...], tuple[str, ...]]:
        event_key = proposal.logical_key
        records: list[dict] = []
        record_ids: list[str] = []
        record_digests: list[str] = []
        for index, observation in enumerate(proposal.observations):
            kind = "correction" if observation.supersedes is not None else "observation"
            record_id = "r_" + _sha256_json({"event_key": event_key, "index": index,
                                             "kind": kind})[:40]
            record = {
                "format": RECORD_FORMAT,
                "record_id": record_id,
                "kind": kind,
                **observation.document(),
                "subject_id": proposal.subject_id,
                "branch_id": proposal.branch_id,
                "purpose": proposal.purpose,
                "mode": proposal.mode,
                "interaction_id": proposal.interaction_id,
            }
            digest = _sha256_json({key: record[key] for key in record if key != "record_id"})
            record["record_digest"] = digest
            records.append(record)
            record_ids.append(record_id)
            record_digests.append(digest)
        payload = {
            "format": EVENT_FORMAT,
            "kind": "correction" if any(item["kind"] == "correction" for item in records)
                    else "interaction",
            "event_key": event_key,
            "proposal": proposal.document(),
            "proposal_digest": proposal.digest,
            "pin": pin.document(),
            "pin_digest": pin.digest,
            "records": records,
            "notes": [item.document() for item in proposal.notes],
            "outcomes": [{**item.document(), "ordinal": index}
                         for index, item in enumerate(proposal.outcomes)],
            "scope": {
                "subject_id": proposal.subject_id,
                "branch_id": proposal.branch_id,
                "purpose": proposal.purpose,
                "mode": proposal.mode,
                "source_epoch": proposal.source_epoch,
                "task_id": proposal.task_id,
                "task_revision": proposal.task_revision,
                "decision_revision": proposal.decision_revision,
            },
            "interaction_id": proposal.interaction_id,
            "created_at_ns": proposal.created_at_ns,
        }
        return payload, tuple(record_ids), tuple(record_digests)

    def _stage_payload(self, payload: Mapping[str, object]) -> tuple[str, Path]:
        text = _canonical(payload) + "\n"
        digest = _sha256_text(text)
        path = self.root / "payloads" / f"{digest}.json"
        data = text.encode("utf-8")
        if path.exists():
            stored = path.read_bytes()
            if _sha256_bytes(stored) != digest:
                raise CorruptAuthority("staged payload path holds different bytes")
            return digest, path
        _atomic_write(path, data)
        self._metrics["bytes_written"] += len(data)
        return digest, path

    def prepare(self, proposal: InteractionProposal, *, pin: MemoryPin) -> PreparedFormation:
        if not isinstance(proposal, InteractionProposal):
            raise TypeError("proposal must be an InteractionProposal")
        self._check_pin(pin)
        self._validate_proposal(proposal, pin)
        payload, record_ids, record_digests = self._build_payload(proposal, pin)
        digest, path = self._stage_payload(payload)
        return PreparedFormation(
            proposal=proposal, pin=pin, event_key=proposal.logical_key,
            payload_digest=digest, payload_path=path, record_ids=record_ids,
            record_digests=record_digests, prepared_at_ns=_now_ns())

    def _existing_receipt(self, event_key: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM events WHERE event_key=?", (event_key,)).fetchone()

    def _write_conflict_payload(self, event_key: str, payload_digest: str) -> Path:
        path = self.root / "conflicts" / f"{event_key}-{payload_digest}.json"
        if not path.exists():
            source = self.root / "payloads" / f"{payload_digest}.json"
            data = source.read_bytes() if source.exists() else b""
            _atomic_write(path, data)
        return path

    def _note_conflict(self, event_key: str, payload_digest: str) -> None:
        path = self._write_conflict_payload(event_key, payload_digest)
        self._connection.execute(
            "INSERT INTO conflicts(event_key, payload_digest, path, stored_at_ns) "
            "VALUES (?, ?, ?, ?)",
            (event_key, payload_digest, str(path.relative_to(self.root)), _now_ns()))
        self._metrics["conflicts"] += 1

    @staticmethod
    def _scope_matches(record: FormationRecord, scope: Mapping[str, object]) -> bool:
        return (record.subject_id == scope["subject_id"]
                and record.branch_id == scope["branch_id"]
                and record.purpose == scope["purpose"]
                and record.mode == scope["mode"])

    def _validate_corrections(self, payload: Mapping[str, object]) -> list[str]:
        scope = payload["scope"]
        affected: list[str] = []
        for record in payload["records"]:
            parent = record.get("supersedes")
            if parent is None:
                continue
            row = self._connection.execute(
                "SELECT record_id FROM record_index WHERE record_id=?", (parent,)).fetchone()
            if row is None:
                raise IneligibleRecord("correction names an unknown record")
            tombstone = self._connection.execute(
                "SELECT kind FROM tombstones WHERE record_id=?", (parent,)).fetchone()
            if tombstone is not None:
                raise IneligibleRecord(
                    f"correction parent is already {tombstone['kind']}")
            parent_payload = self._load_record(parent)
            if not self._scope_matches(parent_payload, scope):
                raise ScopeViolation("correction parent is outside the proposal scope")
            rank = {"unverified": 0, "verified": 1}
            if rank[record["verification_state"]] < rank[parent_payload.verification_state]:
                raise ScopeViolation(
                    "an unverified successor cannot supersede verified evidence")
            if not set(parent_payload.permitted_uses) <= set(record["permitted_uses"]):
                raise ScopeViolation(
                    "correction successor has narrower permitted uses than its parent")
            if parent_payload.observed_at >= record["observed_at"]:
                raise ScopeViolation("correction must be observed after its parent")
            affected.append(parent)
            for descendant in self._descendants(parent):
                descendant_record = self._load_record(descendant)
                if not self._scope_matches(descendant_record, scope):
                    raise ScopeViolation("correction descendant is outside the proposal scope")
                affected.append(descendant)
        return sorted(set(affected))

    def _load_payload(self, payload_digest: str) -> tuple[dict, bytes]:
        path = self.root / "payloads" / f"{payload_digest}.json"
        if not path.is_file():
            raise CorruptAuthority(f"payload is missing: {payload_digest}")
        data = path.read_bytes()
        if _sha256_bytes(data) != payload_digest:
            raise CorruptAuthority(f"payload bytes differ from their digest: {payload_digest}")
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CorruptAuthority("payload is not canonical JSON") from error
        if payload.get("format") != EVENT_FORMAT:
            raise CorruptAuthority("payload format is unknown")
        return payload, data

    def _payload_for_event(self, event_key: str) -> tuple[dict, bytes, sqlite3.Row]:
        row = self._existing_receipt(event_key)
        if row is None:
            raise CorruptAuthority("index refers to an absent committed event")
        payload, data = self._load_payload(str(row["payload_digest"]))
        if payload.get("event_key") != event_key:
            raise CorruptAuthority("payload identity differs from the committed event")
        return payload, data, row

    def _record_payloads(self, payload: Mapping[str, object]) -> dict[str, dict]:
        return {record["record_id"]: dict(record) for record in payload["records"]}

    def _load_record(self, record_id: str) -> FormationRecord:
        row = self._connection.execute(
            "SELECT * FROM record_index WHERE record_id=?", (record_id,)).fetchone()
        if row is None:
            raise IneligibleRecord("record is not committed")
        payload, data, event = self._payload_for_event(str(row["event_key"]))
        self._metrics["bytes_read"] += len(data)
        self._metrics["reads"] += 1
        records = self._record_payloads(payload)
        record = records.get(record_id)
        if record is None:
            raise CorruptAuthority("record index references an absent record")
        if record.get("record_digest") != row["record_digest"]:
            raise CorruptAuthority("record digest differs from the index")
        return FormationRecord(
            record_id=record_id, kind=record["kind"], text=record["text"],
            text_sha256=record["text_sha256"], subject_id=record["subject_id"],
            branch_id=record["branch_id"], purpose=record["purpose"],
            mode=record["mode"], source_id=record["source_id"],
            source_epoch=record["source_epoch"], observed_at=record["observed_at"],
            event_at=record["event_at"], valid_from=record["valid_from"],
            valid_to=record["valid_to"], verification_state=record["verification_state"],
            authenticity=record["authenticity"], receipt_id=record["receipt_id"],
            verifier_receipt_id=record["verifier_receipt_id"],
            permitted_uses=tuple(record["permitted_uses"]),
            importance=record["importance"], critical=record["critical"],
            source_lineage=tuple(record["source_lineage"]),
            supersedes=record["supersedes"], interaction_id=record["interaction_id"],
            record_digest=record["record_digest"], event_key=record["event_key"] if "event_key" in record else str(event["event_key"]),
            generation=int(event["generation"]), sequence=int(event["sequence"]))

    def _descendants(self, record_id: str) -> list[str]:
        rows = self._connection.execute(
            "WITH RECURSIVE d(record_id) AS ("
            " SELECT record_id FROM record_lineage WHERE ancestor_id=?"
            " UNION SELECT record_lineage.record_id FROM record_lineage"
            " JOIN d ON record_lineage.ancestor_id=d.record_id)"
            " SELECT record_id FROM d", (record_id,)).fetchall()
        return [str(item["record_id"]) for item in rows]

    def commit(self, prepared: PreparedFormation, *,
               expected_predecessor: int | None = None) -> CommitReport:
        if not isinstance(prepared, PreparedFormation):
            raise TypeError("prepared formation required")
        start = time.perf_counter_ns()
        payload, _ = self._load_payload(prepared.payload_digest)
        if payload.get("event_key") != prepared.event_key:
            raise CorruptAuthority("prepared payload identity changed")
        try:
          try:
            with self._transaction():
                existing = self._existing_receipt(prepared.event_key)
                if existing is not None:
                    if existing["payload_digest"] == prepared.payload_digest:
                        self._metrics["duplicates"] += 1
                        return CommitReport(
                            "duplicate", FormationReceipt.from_document(
                                json.loads(existing["receipt_json"])),
                            prepared.prepared_at_ns)
                    raise FormationConflict(
                        "identical event identity arrived with different bytes")
                authority = self._authority()
                if expected_predecessor is not None and \
                        expected_predecessor != int(authority["sequence"]):
                    raise StalePin("expected predecessor sequence is no longer current")
                if prepared.pin.generation > int(authority["generation"]):
                    raise StalePin("pin generation is ahead of durable authority")
                affected = self._validate_corrections(payload)
                now = _now_ns()
                sequence = int(authority["sequence"]) + 1
                generation = int(authority["generation"]) + 1
                scope = payload["scope"]
                observed = min([record["observed_at"] for record in payload["records"]]
                               or [now])
                receipt = FormationReceipt(
                    event_key=prepared.event_key, kind=str(payload["kind"]),
                    sequence=sequence, generation=generation,
                    predecessor_sequence=int(authority["sequence"]),
                    predecessor_digest=str(authority["head_digest"]),
                    payload_digest=prepared.payload_digest,
                    record_ids=prepared.record_ids,
                    record_digests=prepared.record_digests,
                    note_ids=tuple(note["note_id"] for note in payload["notes"]),
                    outcome_ids=tuple(item["outcome_id"] for item in payload["outcomes"]),
                    pin_digest=prepared.pin.digest, committed_at_ns=now,
                    affected_record_ids=tuple(affected))
                self._connection.execute(
                    "INSERT INTO events(event_key, sequence, generation, kind, "
                    "payload_digest, receipt_json, subject_id, branch_id, purpose, mode, "
                    "source_epoch, observed_at, committed_at_ns) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (receipt.event_key, sequence, generation, receipt.kind,
                     receipt.payload_digest, _canonical(receipt.document()),
                     scope["subject_id"], scope["branch_id"], scope["purpose"],
                     scope["mode"], scope["source_epoch"], observed, now))
                for ordinal, record in enumerate(payload["records"]):
                    self._connection.execute(
                        "INSERT INTO record_index(record_id, event_key, record_digest, "
                        "kind, ordinal) VALUES (?, ?, ?, ?, ?)",
                        (record["record_id"], receipt.event_key,
                         record["record_digest"], record["kind"], ordinal))
                    for ancestor in record["source_lineage"]:
                        if self._connection.execute(
                                "SELECT 1 FROM record_index WHERE record_id=?",
                                (ancestor,)).fetchone() is None:
                            raise IneligibleRecord("source lineage names an unknown record")
                        ancestor_record = self._load_record(ancestor)
                        if not self._scope_matches(ancestor_record, scope):
                            raise ScopeViolation(
                                "source lineage ancestor is outside the proposal scope")
                        if self._connection.execute(
                                "SELECT 1 FROM tombstones WHERE record_id=?",
                                (ancestor,)).fetchone() is not None:
                            raise IneligibleRecord("source lineage contains invalidated evidence")
                        self._connection.execute(
                            "INSERT OR IGNORE INTO record_lineage(record_id, ancestor_id) "
                            "VALUES (?, ?)", (record["record_id"], ancestor))
                    if record["supersedes"] is not None:
                        self._connection.execute(
                            "INSERT INTO tombstones(record_id, kind, reason, event_key, "
                            "at_ns, observed_at, generation) VALUES (?, 'superseded', "
                            "?, ?, ?, ?, ?)",
                            (record["supersedes"], "correction successor",
                             receipt.event_key, now, record["observed_at"], generation))
                correction_observed = max(
                    [record["observed_at"] for record in payload["records"]
                     if record["supersedes"] is not None] or [now])
                for affected_id in affected:
                    self._connection.execute(
                        "INSERT OR IGNORE INTO tombstones(record_id, kind, reason, "
                        "event_key, at_ns, observed_at, generation) VALUES "
                        "(?, 'ancestor_superseded', ?, ?, ?, ?, ?)",
                        (affected_id, "ancestor correction", receipt.event_key, now,
                         correction_observed, generation))
                for note in payload["notes"]:
                    self._connection.execute(
                        "INSERT INTO notes(note_id, event_key, branch_id, purpose, "
                        "interaction_id, origin, text, text_digest, committed_at_ns) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (note["note_id"], receipt.event_key, scope["branch_id"],
                         scope["purpose"], note["interaction_id"], note["origin"],
                         note["text"], note["text_sha256"], now))
                self._connection.execute(
                    "UPDATE authority SET sequence=?, generation=?, head_digest=? WHERE id=1",
                    (sequence, generation, receipt.receipt_digest))
                self._metrics["commits"] += 1
                return CommitReport("committed", receipt, prepared.prepared_at_ns)
          except FormationConflict:
            self._note_conflict(prepared.event_key, prepared.payload_digest)
            raise
        finally:
            self._metrics["commit_latency_ns"].append(time.perf_counter_ns() - start)

    def commit_interaction(self, proposal: InteractionProposal, *, pin: MemoryPin,
                           expected_predecessor: int | None = None) -> CommitReport:
        prepared = self.prepare(proposal, pin=pin)
        return self.commit(prepared, expected_predecessor=expected_predecessor)

    # -- eligibility and reads --------------------------------------------

    def _eligible(self, record: FormationRecord, pin: MemoryPin) -> str | None:
        cutoff = pin.cutoff
        if record.subject_id != pin.subject_id or record.branch_id != pin.branch_id \
                or record.purpose != pin.purpose or record.mode != pin.mode:
            return "scope"
        if record.source_epoch not in pin.source_epochs:
            return "source_epoch"
        if record.generation > pin.generation:
            return "not_visible_to_pin"
        if record.observed_at > cutoff:
            return "future_observation"
        if record.valid_from > cutoff:
            return "not_yet_valid"
        if record.valid_to is not None and cutoff >= record.valid_to:
            return "expired"
        if record.verification_state not in pin.include_verification_states:
            return "verification_filter"
        if not set(record.permitted_uses) & set(pin.permitted_uses):
            return "permitted_use"
        tombstone = self._connection.execute(
            "SELECT kind, observed_at, generation FROM tombstones WHERE record_id=?",
            (record.record_id,)).fetchone()
        if tombstone is not None:
            if tombstone["kind"] == "revoked":
                # Privacy revocation is an immediate barrier: a pin never
                # overrides later revocation. The host must abort/rebuild the
                # affected interaction before resuming.
                return "tombstoned:revoked"
            if int(tombstone["generation"]) <= pin.generation \
                    and int(tombstone["observed_at"]) <= pin.cutoff:
                # Corrections remain immutable-snapshot semantics.
                return f"tombstoned:{tombstone['kind']}"
        return None

    def visible(self, pin: MemoryPin) -> tuple[FormationRecord, ...]:
        """All records eligible under one immutable pin, in commit order."""
        self._check_pin(pin)
        start = time.perf_counter_ns()
        rows = self._connection.execute(
            "SELECT record_id FROM record_index ri JOIN events e "
            "ON e.event_key=ri.event_key WHERE e.generation<=? "
            "AND e.subject_id=? AND e.branch_id=? AND e.purpose=? AND e.mode=? "
            "ORDER BY e.sequence, ri.ordinal, ri.record_id",
            (pin.generation, pin.subject_id, pin.branch_id, pin.purpose, pin.mode)
        ).fetchall()
        output: list[FormationRecord] = []
        try:
            for row in rows:
                record = self._load_record(str(row["record_id"]))
                if record.kind != "observation" and record.kind != "correction":
                    continue
                if self._eligible(record, pin) is None:
                    output.append(record)
        except BaseException:
            self._metrics["failed_reads"] += 1
            raise
        finally:
            self._metrics["read_latency_ns"].append(time.perf_counter_ns() - start)
        return tuple(output)

    def read_record(self, record_id: str, pin: MemoryPin) -> FormationRecord:
        self._check_pin(pin)
        start = time.perf_counter_ns()
        try:
            record = self._load_record(record_id)
            reason = self._eligible(record, pin)
            if reason is not None:
                raise IneligibleRecord(f"record is not eligible: {reason}")
            return record
        except BaseException:
            self._metrics["failed_reads"] += 1
            raise
        finally:
            self._metrics["read_latency_ns"].append(time.perf_counter_ns() - start)

    def outcomes(self, pin: MemoryPin) -> tuple[dict, ...]:
        """Committed outcome receipts eligible under one pin (never E records).

        Revocation is an immediate privacy barrier: outcomes belonging to a
        revoked record's event are excluded even for an older pin. Correction
        filtering remains generation/cutoff bound.
        """
        self._check_pin(pin)
        rows = self._connection.execute(
            "SELECT * FROM events WHERE generation<=? AND subject_id=? AND branch_id=? "
            "AND purpose=? AND mode=? ORDER BY sequence",
            (pin.generation, pin.subject_id, pin.branch_id, pin.purpose, pin.mode)
        ).fetchall()
        revoked_events = {
            str(row["event_key"]) for row in self._connection.execute(
                "SELECT DISTINCT ri.event_key FROM record_index ri "
                "JOIN tombstones t ON t.record_id=ri.record_id "
                "WHERE t.kind='revoked'")}
        entries: list[tuple[int, int, dict]] = []
        for row in rows:
            if str(row["event_key"]) in revoked_events:
                continue
            payload, data = self._load_payload(str(row["payload_digest"]))
            self._metrics["bytes_read"] += len(data)
            for outcome in payload.get("outcomes", ()):
                if int(outcome.get("observed_at", 0)) > pin.cutoff:
                    continue
                if int(outcome.get("source_epoch", 0)) not in pin.source_epochs:
                    continue
                entries.append((int(row["sequence"]), int(outcome.get("ordinal", 0)),
                                dict(outcome)))
        entries.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in entries)

    def read_provisional(self, *, branch_id: str, purpose: str) -> tuple[dict, ...]:
        """Branch-owned provisional notes; never evidence and never reader input."""
        _identifier("branch_id", branch_id)
        _identifier("purpose", purpose)
        rows = self._connection.execute(
            "SELECT * FROM notes WHERE branch_id=? AND purpose=? ORDER BY committed_at_ns",
            (branch_id, purpose)).fetchall()
        return tuple(dict(row) for row in rows)

    # -- correction and revocation ---------------------------------------

    def correct(self, record_id: str, *, text: str, reason: str, pin: MemoryPin,
                source_id: str | None = None, source_epoch: int | None = None,
                verifier_receipt_id: str | None = None) -> CommitReport:
        parent = self._load_record(record_id)
        if parent.branch_id != pin.branch_id or parent.purpose != pin.purpose:
            raise ScopeViolation("correction pin does not own the record")
        observation = Observation(
            observation_id="corr_" + _sha256_json(
                {"record_id": record_id, "text": text, "reason": reason})[:40],
            text=text, source_id=source_id or parent.source_id,
            source_epoch=parent.source_epoch if source_epoch is None else source_epoch,
            observed_at=_now_ns(), receipt_id="host-correction-" + _sha256_text(reason)[:24],
            verification_state="verified" if verifier_receipt_id else "unverified",
            verifier_receipt_id=verifier_receipt_id,
            permitted_uses=parent.permitted_uses, importance=parent.importance,
            critical=parent.critical, source_lineage=(record_id,),
            supersedes=record_id)
        proposal = InteractionProposal(
            interaction_id="correction-" + observation.observation_id,
            subject_id=pin.subject_id, branch_id=pin.branch_id, purpose=pin.purpose,
            mode=pin.mode, task_id=pin.task_id, task_revision=pin.task_revision,
            decision_revision=pin.decision_revision + 1, source_epoch=parent.source_epoch,
            producer_versions=pin.producer_versions,
            representation_version=pin.representation_version, codec_id=pin.codec_id,
            permitted_uses=pin.permitted_uses, observations=(observation,))
        return self.commit_interaction(proposal, pin=pin)

    def revoke(self, record_id: str, *, reason: str,
               scope: str = "record_scope",
               authorized_by: str | None = None) -> CommitReport:
        """Revoke one record and its in-scope descendants.

        ``scope='record_scope'`` (default) cannot cross subject/branch/purpose/
        mode. ``scope='host_wide'`` is an explicit host authority operation and
        requires ``authorized_by='host'``. Privacy revocation is an immediate
        barrier: it invalidates every pin, including pins created earlier.
        Correction tombstones, by contrast, remain immutable-snapshot gated.
        """
        _identifier("record_id", record_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("revocation reason required")
        if scope not in ("record_scope", "host_wide"):
            raise ValueError("unknown revocation scope")
        if scope == "host_wide" and authorized_by != "host":
            raise PermissionError("host-wide revocation requires host authority")
        record = self._load_record(record_id)
        descendants = self._descendants(record_id)
        if scope == "record_scope":
            descendants = [item for item in descendants
                           if self._scope_matches(self._load_record(item),
                                                  {"subject_id": record.subject_id,
                                                   "branch_id": record.branch_id,
                                                   "purpose": record.purpose,
                                                   "mode": record.mode})]
        payload_body = {
            "format": EVENT_FORMAT, "kind": "revocation", "record_id": record_id,
            "reason": reason, "descendants": sorted(descendants),
            "revocation_scope": scope, "authorized_by": authorized_by,
        }
        payload = {**payload_body, "event_key": "", "records": [], "notes": [],
                   "outcomes": []}
        event_key = "rv_" + _sha256_json({"format": FORMAT, "record_id": record_id})[:48]
        payload["event_key"] = event_key
        payload["scope"] = {"subject_id": self.subject_id}
        start = time.perf_counter_ns()
        try:
          try:
            payload_digest = _sha256_text(_canonical(payload) + "\n")
            path = self.root / "payloads" / f"{payload_digest}.json"
            if path.exists():
                if _sha256_bytes(path.read_bytes()) != payload_digest:
                    raise CorruptAuthority("revocation payload collision")
            else:
                _atomic_write(path, (_canonical(payload) + "\n").encode("utf-8"))
                self._metrics["bytes_written"] += len(_canonical(payload)) + 1
            with self._transaction():
                existing = self._connection.execute(
                    "SELECT * FROM revocations WHERE record_id=?", (record_id,)).fetchone()
                if existing is not None:
                    if existing["payload_digest"] == payload_digest:
                        self._metrics["duplicates"] += 1
                        return CommitReport("duplicate", FormationReceipt.from_document(
                            json.loads(existing["receipt_json"])), _now_ns())
                    raise FormationConflict("revocation identity reused with different bytes")
                if self._connection.execute(
                        "SELECT 1 FROM record_index WHERE record_id=?", (record_id,)
                ).fetchone() is None:
                    raise IneligibleRecord("revocation names an unknown record")
                authority = self._authority()
                sequence = int(authority["sequence"]) + 1
                generation = int(authority["generation"]) + 1
                now = _now_ns()
                affected = [record_id] + sorted(descendants)
                receipt = FormationReceipt(
                    event_key=event_key, kind="revocation", sequence=sequence,
                    generation=generation,
                    predecessor_sequence=int(authority["sequence"]),
                    predecessor_digest=str(authority["head_digest"]),
                    payload_digest=payload_digest, record_ids=(record_id,),
                    record_digests=(), note_ids=(), outcome_ids=(),
                    pin_digest=_sha256_json({"record_id": record_id, "reason": reason,
                                             "scope": scope}),
                    committed_at_ns=now, affected_record_ids=tuple(affected))
                self._connection.execute(
                    "INSERT INTO events(event_key, sequence, generation, kind, payload_digest, "
                    "receipt_json, subject_id, branch_id, purpose, mode, source_epoch, "
                    "observed_at, committed_at_ns) VALUES (?, ?, ?, 'revocation', ?, ?, ?, "
                    "'host', 'revocation', 'assistant_for_source', 0, ?, ?)",
                    (event_key, sequence, generation, payload_digest,
                     _canonical(receipt.document()), self.subject_id, now, now))
                for affected_id in affected:
                    self._connection.execute(
                        "INSERT OR REPLACE INTO tombstones(record_id, kind, reason, "
                        "event_key, at_ns, observed_at, generation) VALUES "
                        "(?, 'revoked', ?, ?, ?, ?, ?)",
                        (affected_id, reason if affected_id == record_id
                         else "ancestor revoked", event_key, now, now, generation))
                self._connection.execute(
                    "INSERT INTO revocations(record_id, event_key, payload_digest, "
                    "receipt_json, at_ns) VALUES (?, ?, ?, ?, ?)",
                    (record_id, event_key, payload_digest,
                     _canonical(receipt.document()), now))
                self._connection.execute(
                    "UPDATE authority SET sequence=?, generation=?, head_digest=? WHERE id=1",
                    (sequence, generation, receipt.receipt_digest))
                self._metrics["commits"] += 1
                return CommitReport("committed", receipt, now)
          except FormationConflict:
            self._note_conflict(event_key, _sha256_text(_canonical(payload) + "\n"))
            raise
        finally:
            self._metrics["commit_latency_ns"].append(time.perf_counter_ns() - start)

    # -- recovery ----------------------------------------------------------

    def _expected_derived(self) -> tuple[dict, set, dict, dict, dict]:
        events = self._connection.execute(
            "SELECT * FROM events ORDER BY sequence").fetchall()
        index: dict[str, tuple[str, str, str, int]] = {}
        lineage: set[tuple[str, str]] = set()
        notes: dict[str, tuple] = {}
        tombstones: dict[str, tuple] = {}
        revocations: dict[str, tuple] = {}
        payloads: dict[str, dict] = {}
        scopes: dict[str, dict] = {}
        for row in events:
            event_key = str(row["event_key"])
            payload, _ = self._load_payload(str(row["payload_digest"]))
            if payload.get("kind") != row["kind"]:
                raise CorruptAuthority("event kind differs from its payload")
            payloads[event_key] = payload
            if payload.get("format") != EVENT_FORMAT:
                raise CorruptAuthority("event payload format is unknown")
            if payload.get("kind") == "revocation":
                if str(row["subject_id"]) != self.subject_id:
                    raise CorruptAuthority("revocation event subject mismatch")
                scopes[event_key] = {"subject_id": self.subject_id}
            else:
                scope = payload.get("scope", {})
                expected = (self.subject_id, scope.get("branch_id"),
                            scope.get("purpose"), scope.get("mode"),
                            scope.get("source_epoch"))
                actual = (str(row["subject_id"]), str(row["branch_id"]),
                          str(row["purpose"]), str(row["mode"]),
                          int(row["source_epoch"]))
                if actual != expected:
                    raise CorruptAuthority("event scope columns differ from the payload")
                scopes[event_key] = scope
            for note in payload.get("notes", ()):
                notes[note["note_id"]] = (
                    event_key, scopes[event_key]["branch_id"],
                    scopes[event_key]["purpose"], note["interaction_id"],
                    note["origin"], note["text"], note["text_sha256"],
                    int(row["committed_at_ns"]))
            for ordinal, record in enumerate(payload.get("records", ())):
                index[record["record_id"]] = (
                    event_key, record["record_digest"], record["kind"], ordinal)
                for ancestor in record["source_lineage"]:
                    lineage.add((record["record_id"], ancestor))
        def descendants(parent: str) -> set[str]:
            found: set[str] = set()
            frontier = {parent}
            while frontier:
                next_frontier = {child for child, ancestor in lineage
                                 if ancestor in frontier and child not in found}
                found |= next_frontier
                frontier = next_frontier
            return found
        for row in events:
            event_key = str(row["event_key"])
            payload = payloads[event_key]
            committed = int(row["committed_at_ns"])
            generation = int(row["generation"])
            if payload.get("kind") == "revocation":
                record_id = payload["record_id"]
                tombstones[record_id] = (
                    "revoked", payload["reason"], event_key, committed,
                    committed, generation)
                for descendant_id in payload.get("descendants", ()):
                    tombstones[descendant_id] = (
                        "revoked", "ancestor revoked", event_key, committed,
                        committed, generation)
                revocations[record_id] = (
                    event_key, str(row["payload_digest"]), str(row["receipt_json"]),
                    committed)
            else:
                for record in payload.get("records", ()):
                    parent = record.get("supersedes")
                    if parent is None:
                        continue
                    tombstones[parent] = (
                        "superseded", "correction successor", event_key, committed,
                        record["observed_at"], generation)
                    for descendant_id in descendants(parent):
                        tombstones.setdefault(descendant_id, (
                            "ancestor_superseded", "ancestor correction", event_key,
                            committed, record["observed_at"], generation))
        return index, lineage, notes, tombstones, revocations

    def _rebuild_derived(self) -> tuple[bool, bool]:
        expected = self._expected_derived()
        current_index = {
            str(row["record_id"]): (str(row["event_key"]), str(row["record_digest"]),
                                    str(row["kind"]), int(row["ordinal"]))
            for row in self._connection.execute("SELECT * FROM record_index")}
        current_lineage = {(str(row["record_id"]), str(row["ancestor_id"]))
                           for row in self._connection.execute(
                               "SELECT * FROM record_lineage")}
        current_notes = {
            str(row["note_id"]): (str(row["event_key"]), str(row["branch_id"]),
                                  str(row["purpose"]), str(row["interaction_id"]),
                                  str(row["origin"]), str(row["text"]),
                                  str(row["text_digest"]), int(row["committed_at_ns"]))
            for row in self._connection.execute("SELECT * FROM notes")}
        current_tombstones = {
            str(row["record_id"]): (str(row["kind"]), str(row["reason"]),
                                    str(row["event_key"]), int(row["at_ns"]),
                                    int(row["observed_at"]), int(row["generation"]))
            for row in self._connection.execute("SELECT * FROM tombstones")}
        current_revocations = {
            str(row["record_id"]): (str(row["event_key"]),
                                    str(row["payload_digest"]),
                                    str(row["receipt_json"]), int(row["at_ns"]))
            for row in self._connection.execute("SELECT * FROM revocations")}
        expected_index, expected_lineage, expected_notes, expected_tombstones, \
            expected_revocations = expected
        index_changed = current_index != expected_index
        changed = (index_changed or current_lineage != expected_lineage
                   or current_notes != expected_notes
                   or current_tombstones != expected_tombstones
                   or current_revocations != expected_revocations)
        if not changed:
            return False, False
        with self._transaction():
            self._connection.execute("DELETE FROM record_index")
            self._connection.execute("DELETE FROM record_lineage")
            self._connection.execute("DELETE FROM notes")
            self._connection.execute("DELETE FROM tombstones")
            self._connection.execute("DELETE FROM revocations")
            for record_id, (event_key, digest, kind, ordinal) in expected_index.items():
                self._connection.execute(
                    "INSERT INTO record_index(record_id, event_key, record_digest, "
                    "kind, ordinal) VALUES (?, ?, ?, ?, ?)",
                    (record_id, event_key, digest, kind, ordinal))
            for record_id, ancestor_id in expected_lineage:
                self._connection.execute(
                    "INSERT INTO record_lineage(record_id, ancestor_id) VALUES (?, ?)",
                    (record_id, ancestor_id))
            for note_id, values in expected_notes.items():
                self._connection.execute(
                    "INSERT INTO notes(note_id, event_key, branch_id, purpose, "
                    "interaction_id, origin, text, text_digest, committed_at_ns) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (note_id, *values))
            for record_id, values in expected_tombstones.items():
                self._connection.execute(
                    "INSERT INTO tombstones(record_id, kind, reason, event_key, at_ns, "
                    "observed_at, generation) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (record_id, *values))
            for record_id, values in expected_revocations.items():
                self._connection.execute(
                    "INSERT INTO revocations(record_id, event_key, payload_digest, "
                    "receipt_json, at_ns) VALUES (?, ?, ?, ?, ?)",
                    (record_id, *values))
        return True, index_changed

    def _derived_errors(self) -> tuple[str, ...]:
        """Authenticate derived tombstone/note/lineage/revocation references."""
        errors: list[str] = []
        events = {str(row["event_key"]) for row in self._connection.execute(
            "SELECT event_key FROM events")}
        records = {str(row["record_id"]) for row in self._connection.execute(
            "SELECT record_id FROM record_index")}
        for row in self._connection.execute("SELECT * FROM tombstones"):
            if row["event_key"] is not None and str(row["event_key"]) not in events:
                errors.append(f"tombstone {row['record_id']} cites absent event")
            if str(row["record_id"]) not in records:
                errors.append(f"tombstone names absent record {row['record_id']}")
            if int(row["generation"]) < 0 or int(row["observed_at"]) < 0:
                errors.append(f"tombstone {row['record_id']} has invalid time/generation")
        for row in self._connection.execute("SELECT * FROM notes"):
            if str(row["event_key"]) not in events:
                errors.append(f"note {row['note_id']} cites absent event")
        for row in self._connection.execute("SELECT * FROM record_lineage"):
            if str(row["record_id"]) not in records or str(row["ancestor_id"]) not in records:
                errors.append("lineage references an absent record")
        for row in self._connection.execute("SELECT * FROM revocations"):
            if str(row["event_key"]) not in events:
                errors.append(f"revocation {row['record_id']} cites absent event")
        return tuple(errors)

    def recover(self) -> RecoveryReport:
        """Verify the authority chain and rebuild every derived table.

        Corrections are generation/cutoff gated; privacy revocations apply
        immediately. Derived mismatches are repaired and reported as degraded
        rather than silently accepted.
        """
        authority = self._authority()
        rows = self._connection.execute(
            "SELECT * FROM events ORDER BY sequence").fetchall()
        sequence = 0
        previous = _sha256_json({
            "format": DB_FORMAT,
            "subject_id": self.subject_id,
            "source_epochs": list(self.source_epochs),
            "producer_versions": {key: value for key, value in self.producer_versions},
            "representation_version": self.representation_version,
            "codec_id": self.codec_id,
        })
        quarantined: list[str] = []
        for row in rows:
            sequence += 1
            if int(row["sequence"]) != sequence:
                raise CorruptAuthority("committed sequence has a gap")
            if int(row["generation"]) != sequence:
                raise CorruptAuthority("generation did not advance exactly once per event")
            receipt = json.loads(row["receipt_json"])
            if receipt.get("predecessor_digest") != previous:
                raise CorruptAuthority("receipt chain is broken")
            if receipt.get("receipt_digest") != _sha256_json(
                    {key: value for key, value in receipt.items()
                     if key != "receipt_digest"}):
                raise CorruptAuthority("receipt digest is invalid")
            try:
                self._load_payload(str(row["payload_digest"]))
            except CorruptAuthority:
                quarantined.append(str(row["event_key"]))
            previous = str(receipt["receipt_digest"])
        if sequence != int(authority["sequence"]):
            raise CorruptAuthority("authority sequence differs from committed events")
        if int(authority["generation"]) != sequence:
            raise CorruptAuthority("authority generation differs from committed events")
        if str(authority["head_digest"]) != previous:
            raise CorruptAuthority("authority head differs from the receipt chain")
        derived_rebuilt = False
        index_rebuilt = False
        if not quarantined:
            derived_rebuilt, index_rebuilt = self._rebuild_derived()
        derived_errors = self._derived_errors()
        referenced = {str(row["payload_digest"]) for row in rows}
        orphans = [path for path in (self.root / "payloads").glob("*.json")
                   if path.stem not in referenced]
        pending_derived, pending_index = self._pending_recovery
        self._pending_recovery = (False, False)
        if not derived_rebuilt and not index_rebuilt and (pending_derived or pending_index):
            # Surface a repair performed by construction recovery exactly once
            # on the caller's next explicit recovery call.
            return RecoveryReport(
                status="degraded" if pending_derived else "ok",
                events=sequence, generation=sequence, sequence=sequence,
                head_digest=previous, orphan_payloads=len(orphans),
                quarantined=tuple(quarantined), index_rebuilt=pending_index,
                derived_errors=derived_errors, derived_rebuilt=pending_derived)
        return RecoveryReport(
            status="degraded" if (quarantined or derived_rebuilt
                                   or derived_errors) else "ok",
            events=sequence, generation=sequence, sequence=sequence,
            head_digest=previous, orphan_payloads=len(orphans),
            quarantined=tuple(quarantined), index_rebuilt=index_rebuilt,
            derived_errors=derived_errors, derived_rebuilt=derived_rebuilt)

    # -- metrics -----------------------------------------------------------

    def metrics_document(self) -> dict:
        self.recovery_report = self.recover()
        commit = sorted(self._metrics["commit_latency_ns"])
        reads = sorted(self._metrics["read_latency_ns"])

        def percentile(values: Sequence[int], fraction: float) -> float | None:
            if not values:
                return None
            index = min(len(values) - 1, max(0, int(math.ceil(fraction * len(values))) - 1))
            return values[index] / 1e6

        return {
            "format": FORMAT,
            "commits": self._metrics["commits"],
            "duplicates": self._metrics["duplicates"],
            "conflicts": self._metrics["conflicts"],
            "reads": self._metrics["reads"],
            "failed_reads": self._metrics["failed_reads"],
            "bytes_written": self._metrics["bytes_written"],
            "bytes_read": self._metrics["bytes_read"],
            "commit_latency_ms": {
                "p50": percentile(commit, 0.5), "p95": percentile(commit, 0.95),
                "samples": len(commit)},
            "read_latency_ms": {
                "p50": percentile(reads, 0.5), "p95": percentile(reads, 0.95),
                "samples": len(reads)},
            "recovery": (self.recovery_report.document()
                         if self.recovery_report is not None else None),
        }
