"""S2b chunk-start/chunk-barrier formation: isolated CPU contract, version 1.

One immutable snapshot is read at chunk start.  Pure tensor computation may
accumulate bounded proposals, but nothing is read, written, counted or optimized
during forward/backward recomputation.  Exactly one commit occurs at the chunk
barrier and the next revision becomes visible only to later chunks.

Logical identity (``replica.successors.s2b.v1``) binds schema, owner/branch,
sequence, decision revision, chunk index, absolute token start/end, token prefix
digest, input state revision, writer version and effect kind.  Attempt IDs,
transport retries and checkpoint-recomputation IDs are audit fields and never
change the logical key.  A committed absolute-position watermark plus prefix
digest prevents a growing-prefix evaluation from writing old tokens again.

The tensor fast state is a session-owned linear delta rule.  It is a CPU
engineering contract, not a qualified native within-reasoning memory and not
persistent P.  An imperative optimizer is not used anywhere here; a live
differentiable unroll is available and its detach/serialization boundary is
explicit.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Mapping, Sequence

import torch
from torch import Tensor

FORMAT = "replica.successors.s2b.v1"
DB_FORMAT = "replica.successors.s2b.authority.v1"
RECEIPT_FORMAT = "replica.successors.s2b.receipt.v1"
EFFECT_KINDS = ("associate", "provisional_fact", "procedure_candidate")
MAX_UNROLL = 8
MAX_TOKENS = 8192

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ChunkError(RuntimeError):
    """Base class for chunk formation contract failures."""


class StaleChunk(ChunkError):
    """The snapshot revision or decision revision has moved."""


class PrefixOverlap(ChunkError):
    """The proposal overlaps the committed absolute-position watermark."""


class PrefixChanged(ChunkError):
    """The committed prefix digest does not match the supplied prefix."""


class PositionGap(ChunkError):
    """The proposal does not begin at the committed watermark."""


class ChunkConflict(ChunkError):
    """The same logical key arrived with a different proposal digest."""


class CorruptChunkAuthority(ChunkError):
    """The durable chunk authority or receipt chain is invalid."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_text(_canonical(value))


def _identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded identifier")


def _digest(name: str, value: object) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")


def _timestamp(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer nanosecond timestamp")


def _now_ns() -> int:
    return time.time_ns()


def prefix_digest(tokens: Sequence[int]) -> str:
    """Canonical digest of a committed token prefix; positions are absolute."""
    values = tuple(int(item) for item in tokens)
    if any(item < 0 for item in values):
        raise ValueError("token positions must be nonnegative")
    return _sha256_json({"format": FORMAT, "tokens": list(values)})


def tensor_digest(tensor: Tensor) -> str:
    """Detached serialization-boundary digest; never used inside a forward pass."""
    if not isinstance(tensor, Tensor):
        raise TypeError("tensor digest requires a Tensor")
    detached = tensor.detach().to("cpu").contiguous()
    raw = detached.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(
        _canonical({"shape": list(detached.shape),
                    "dtype": str(detached.dtype).split(".")[-1]}).encode("utf-8")
        + b"\0" + raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ChunkKey:
    """Logical identity for exactly one chunk effect."""

    schema: str
    subject_id: str
    owner: str
    branch_id: str
    sequence_id: str
    decision_revision: int
    chunk_index: int
    token_start: int
    token_end: int
    token_prefix_digest: str
    input_state_revision: int
    writer_version: str
    effect_kind: str

    def __post_init__(self) -> None:
        if self.schema != FORMAT:
            raise ValueError("chunk key schema mismatch")
        for name in ("subject_id", "owner", "branch_id", "sequence_id", "writer_version"):
            _identifier(name, getattr(self, name))
        for name in ("decision_revision", "chunk_index", "token_start", "token_end",
                     "input_state_revision"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.token_end <= self.token_start:
            raise ValueError("token_end must follow token_start")
        if self.token_end > MAX_TOKENS:
            raise ValueError("chunk token range exceeds the bounded maximum")
        _digest("token_prefix_digest", self.token_prefix_digest)
        if self.effect_kind not in EFFECT_KINDS:
            raise ValueError("unknown effect kind")

    def document(self) -> dict:
        return {name: getattr(self, name) for name in (
            "schema", "subject_id", "owner", "branch_id", "sequence_id",
            "decision_revision", "chunk_index", "token_start", "token_end",
            "token_prefix_digest", "input_state_revision", "writer_version",
            "effect_kind")}

    @property
    def logical_key(self) -> str:
        return "ck_" + _sha256_json(self.document())[:48]


@dataclass(frozen=True, slots=True)
class AttemptAudit:
    """Audit-only identity: never part of the logical commit key."""

    attempt_id: str = "attempt-1"
    transport_retries: int = 0
    checkpoint_recomputation_id: str = "recompute-0"

    def __post_init__(self) -> None:
        _identifier("attempt_id", self.attempt_id)
        _identifier("checkpoint_recomputation_id", self.checkpoint_recomputation_id)
        if type(self.transport_retries) is not int or self.transport_retries < 0:
            raise ValueError("transport_retries must be a nonnegative integer")

    def document(self) -> dict:
        return {"attempt_id": self.attempt_id,
                "transport_retries": self.transport_retries,
                "checkpoint_recomputation_id": self.checkpoint_recomputation_id}


@dataclass(frozen=True, slots=True)
class ChunkSnapshot:
    subject_id: str
    owner: str
    branch_id: str
    sequence_id: str
    revision: int
    decision_revision: int
    committed_position: int
    prefix_digest: str
    status: str = "active"

    def document(self) -> dict:
        return {name: getattr(self, name) for name in (
            "subject_id", "owner", "branch_id", "sequence_id", "revision",
            "decision_revision", "committed_position", "prefix_digest", "status")}


@dataclass(frozen=True, slots=True)
class ChunkReceipt:
    logical_key: str
    subject_id: str
    owner: str
    branch_id: str
    sequence_id: str
    chunk_index: int
    revision_before: int
    revision_after: int
    decision_revision: int
    token_start: int
    token_end: int
    proposal_digest: str
    prefix_digest: str
    committed_at_ns: int
    audit: tuple[tuple[str, object], ...] = ()
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        if not self.receipt_digest:
            document = {key: value for key, value in self.document().items()
                        if key != "receipt_digest"}
            object.__setattr__(self, "receipt_digest", _sha256_json(document))

    def document(self) -> dict:
        return {
            "format": RECEIPT_FORMAT,
            "logical_key": self.logical_key,
            "subject_id": self.subject_id,
            "owner": self.owner,
            "branch_id": self.branch_id,
            "sequence_id": self.sequence_id,
            "chunk_index": self.chunk_index,
            "revision_before": self.revision_before,
            "revision_after": self.revision_after,
            "decision_revision": self.decision_revision,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "proposal_digest": self.proposal_digest,
            "prefix_digest": self.prefix_digest,
            "committed_at_ns": self.committed_at_ns,
            "audit": {key: value for key, value in self.audit},
            "receipt_digest": self.receipt_digest,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, object]) -> "ChunkReceipt":
        fields = dict(value)
        fields.pop("format", None)
        audit = fields.pop("audit", {})
        fields["audit"] = tuple(sorted(audit.items()))
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class ChunkCommit:
    status: str  # committed | duplicate
    receipt: ChunkReceipt

    def document(self) -> dict:
        return {"status": self.status, "receipt": self.receipt.document()}


def functional_inner_update(keys: Tensor, values: Tensor, state: Tensor, *,
                            decay: float | Tensor = 0.95,
                            rate: float | Tensor = 0.5) -> tuple[Tensor, Tensor]:
    """One pure gated delta update; live tensors, no optimizer or side effect.

    ``keys`` and ``values`` are ``[B, K, D]``; ``state`` is ``[B, D, D]`` and is
    read as a linear map (read = state @ k).  The returned state and readout
    keep autograd history.  No I/O, counter, context mutation or hash occurs
    here; the caller may unroll up to ``MAX_UNROLL`` live updates.
    """
    if not isinstance(keys, Tensor) or not isinstance(values, Tensor) or \
            not isinstance(state, Tensor):
        raise TypeError("chunk update requires tensors")
    if keys.ndim != 3 or values.ndim != 3 or state.ndim != 3:
        raise ValueError("keys/values must be [B,K,D] and state [B,D,D]")
    if tuple(keys.shape) != tuple(values.shape):
        raise ValueError("keys and values must have identical shapes")
    if keys.shape[0] != state.shape[0] or keys.shape[-1] != state.shape[-1] \
            or state.shape[1] != state.shape[2]:
        raise ValueError("state geometry does not match the chunk features")
    if not (keys.is_floating_point() and values.is_floating_point()
            and state.is_floating_point()):
        raise ValueError("chunk update requires floating tensors")
    if not bool(torch.isfinite(keys).all()) or not bool(torch.isfinite(values).all()) \
            or not bool(torch.isfinite(state).all()):
        raise ValueError("chunk update inputs must be finite")
    width = keys.shape[-1]
    norm = keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = keys / norm
    read = torch.bmm(state, normalized.transpose(1, 2)).transpose(1, 2)
    error = values - read
    outer = torch.einsum("bkd,bke->bde", error, normalized)
    decay_tensor = torch.as_tensor(decay, dtype=state.dtype, device=state.device)
    rate_tensor = torch.as_tensor(rate, dtype=state.dtype, device=state.device)
    next_state = decay_tensor * state + rate_tensor * outer
    if tuple(next_state.shape) != (keys.shape[0], width, width):
        raise RuntimeError("chunk update produced an unexpected shape")
    return next_state, read


class ChunkStore:
    """Durable exactly-once chunk authority with a committed prefix watermark."""

    def __init__(self, root: Path | str, *, subject_id: str) -> None:
        _identifier("subject_id", subject_id)
        self.subject_id = subject_id
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.root / "chunks.sqlite3", isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sequences(
                subject_id TEXT NOT NULL, owner TEXT NOT NULL, branch_id TEXT NOT NULL,
                sequence_id TEXT NOT NULL, revision INTEGER NOT NULL,
                decision_revision INTEGER NOT NULL, committed_position INTEGER NOT NULL,
                prefix_digest TEXT NOT NULL, status TEXT NOT NULL, updated_ns INTEGER NOT NULL,
                PRIMARY KEY(subject_id, owner, branch_id, sequence_id));
            CREATE TABLE IF NOT EXISTS commits(
                logical_key TEXT PRIMARY KEY, subject_id TEXT NOT NULL, owner TEXT NOT NULL,
                branch_id TEXT NOT NULL, sequence_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL, revision_before INTEGER NOT NULL,
                revision_after INTEGER NOT NULL, decision_revision INTEGER NOT NULL,
                token_start INTEGER NOT NULL, token_end INTEGER NOT NULL,
                proposal_digest TEXT NOT NULL, prefix_digest TEXT NOT NULL,
                audit_json TEXT NOT NULL, receipt_json TEXT NOT NULL,
                committed_at_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS conflicts(
                logical_key TEXT NOT NULL, proposal_digest TEXT NOT NULL,
                stored_at_ns INTEGER NOT NULL);
            """
        )
        existing = {row["key"]: row["value"]
                    for row in self._connection.execute("SELECT key, value FROM meta")}
        identity = {"format": DB_FORMAT, "subject_id": subject_id}
        if existing:
            if existing.get("identity") != _canonical(identity):
                raise CorruptChunkAuthority("chunk store identity mismatch")
        else:
            with self._transaction():
                self._connection.execute("INSERT INTO meta(key, value) VALUES (?, ?)",
                                         ("identity", _canonical(identity)))
                self._connection.execute("INSERT INTO meta(key, value) VALUES (?, ?)",
                                         ("format", DB_FORMAT))
        self._metrics = {
            "commits": 0, "duplicates": 0, "conflicts": 0, "stale": 0,
            "prefix_refusals": 0, "commit_latency_ns": [],
        }

    @contextmanager
    def _transaction(self):
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ChunkStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _row(self, owner: str, branch_id: str, sequence_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM sequences WHERE subject_id=? AND owner=? AND branch_id=? "
            "AND sequence_id=?", (self.subject_id, owner, branch_id, sequence_id)
        ).fetchone()
        if row is None:
            raise StaleChunk("sequence is not open")
        return row

    def _snapshot(self, row: sqlite3.Row) -> ChunkSnapshot:
        return ChunkSnapshot(
            subject_id=str(row["subject_id"]), owner=str(row["owner"]),
            branch_id=str(row["branch_id"]), sequence_id=str(row["sequence_id"]),
            revision=int(row["revision"]), decision_revision=int(row["decision_revision"]),
            committed_position=int(row["committed_position"]),
            prefix_digest=str(row["prefix_digest"]), status=str(row["status"]))

    def open_sequence(self, owner: str, branch_id: str, sequence_id: str, *,
                      decision_revision: int = 0,
                      initial_prefix: Sequence[int] = ()) -> ChunkSnapshot:
        _identifier("owner", owner)
        _identifier("branch_id", branch_id)
        _identifier("sequence_id", sequence_id)
        if type(decision_revision) is not int or decision_revision < 0:
            raise ValueError("decision_revision must be a nonnegative integer")
        digest = prefix_digest(initial_prefix)
        now = _now_ns()
        with self._transaction():
            row = self._connection.execute(
                "SELECT * FROM sequences WHERE subject_id=? AND owner=? AND branch_id=? "
                "AND sequence_id=?",
                (self.subject_id, owner, branch_id, sequence_id)).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO sequences(subject_id, owner, branch_id, sequence_id, "
                    "revision, decision_revision, committed_position, prefix_digest, "
                    "status, updated_ns) VALUES (?, ?, ?, ?, 0, ?, ?, ?, 'active', ?)",
                    (self.subject_id, owner, branch_id, sequence_id,
                     decision_revision, len(initial_prefix), digest, now))
            else:
                if int(row["decision_revision"]) != decision_revision:
                    raise StaleChunk("sequence already exists at another decision revision")
                if row["prefix_digest"] != digest:
                    raise PrefixChanged("initial prefix differs from the stored prefix")
                return self._snapshot(row)
        return self.snapshot(owner, branch_id, sequence_id)

    def snapshot(self, owner: str, branch_id: str, sequence_id: str) -> ChunkSnapshot:
        return self._snapshot(self._row(owner, branch_id, sequence_id))

    def _existing(self, logical_key: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM commits WHERE logical_key=?", (logical_key,)).fetchone()

    def barrier(self, snapshot: ChunkSnapshot, *, key: ChunkKey,
                proposal_digest: str, next_prefix_digest: str,
                audit: AttemptAudit | None = None) -> ChunkCommit:
        """Commit exactly one chunk effect; retries return the prior receipt."""
        if not isinstance(snapshot, ChunkSnapshot):
            raise TypeError("snapshot must be a ChunkSnapshot")
        if not isinstance(key, ChunkKey):
            raise TypeError("key must be a ChunkKey")
        if (key.subject_id != self.subject_id or key.owner != snapshot.owner
                or key.branch_id != snapshot.branch_id
                or key.sequence_id != snapshot.sequence_id):
            raise ChunkError("logical key identity differs from the snapshot")
        _digest("proposal_digest", proposal_digest)
        _digest("next_prefix_digest", next_prefix_digest)
        if audit is not None and not isinstance(audit, AttemptAudit):
            raise TypeError("audit must be an AttemptAudit or None")
        start = time.perf_counter_ns()
        try:
          try:
            with self._transaction():
                row = self._row(key.owner, key.branch_id, key.sequence_id)
                existing = self._existing(key.logical_key)
                if existing is not None:
                    if existing["proposal_digest"] == proposal_digest:
                        self._metrics["duplicates"] += 1
                        return ChunkCommit("duplicate", ChunkReceipt.from_document(
                            json.loads(existing["receipt_json"])))
                    raise ChunkConflict(
                        "logical chunk identity reused with a different proposal")
                # Only an uncommitted logical key needs a live snapshot binding;
                # a retry of a committed key returns its receipt above even when
                # the caller presents the now-later sequence snapshot.
                if key.input_state_revision != snapshot.revision \
                        or key.decision_revision != snapshot.decision_revision:
                    self._metrics["stale"] += 1
                    raise StaleChunk("logical key was built from a different snapshot")
                if row["status"] != "active":
                    self._metrics["stale"] += 1
                    raise StaleChunk("sequence is corrected; start a new decision revision")
                if (int(row["revision"]) != snapshot.revision
                        or int(row["decision_revision"]) != snapshot.decision_revision):
                    self._metrics["stale"] += 1
                    raise StaleChunk("authority advanced during the outstanding chunk")
                committed = int(row["committed_position"])
                if key.token_start < committed:
                    self._metrics["prefix_refusals"] += 1
                    raise PrefixOverlap(
                        "proposal overlaps the committed absolute-position watermark")
                if key.token_start > committed:
                    self._metrics["prefix_refusals"] += 1
                    raise PositionGap("proposal does not begin at the watermark")
                if key.token_prefix_digest != row["prefix_digest"]:
                    self._metrics["prefix_refusals"] += 1
                    raise PrefixChanged("committed prefix digest differs")
                now = _now_ns()
                revision_after = int(row["revision"]) + 1
                audit_value = (audit or AttemptAudit()).document()
                receipt = ChunkReceipt(
                    logical_key=key.logical_key, subject_id=self.subject_id,
                    owner=key.owner, branch_id=key.branch_id,
                    sequence_id=key.sequence_id, chunk_index=key.chunk_index,
                    revision_before=int(row["revision"]),
                    revision_after=revision_after,
                    decision_revision=key.decision_revision,
                    token_start=key.token_start, token_end=key.token_end,
                    proposal_digest=proposal_digest,
                    prefix_digest=next_prefix_digest, committed_at_ns=now,
                    audit=tuple(sorted(audit_value.items())))
                self._connection.execute(
                    "INSERT INTO commits(logical_key, subject_id, owner, branch_id, "
                    "sequence_id, chunk_index, revision_before, revision_after, "
                    "decision_revision, token_start, token_end, proposal_digest, "
                    "prefix_digest, audit_json, receipt_json, committed_at_ns) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (receipt.logical_key, self.subject_id, key.owner, key.branch_id,
                     key.sequence_id, key.chunk_index, receipt.revision_before,
                     revision_after, key.decision_revision, key.token_start,
                     key.token_end, proposal_digest, next_prefix_digest,
                     _canonical(audit_value), _canonical(receipt.document()), now))
                self._connection.execute(
                    "UPDATE sequences SET revision=?, committed_position=?, "
                    "prefix_digest=?, updated_ns=? WHERE subject_id=? AND owner=? "
                    "AND branch_id=? AND sequence_id=?",
                    (revision_after, key.token_end, next_prefix_digest, now,
                     self.subject_id, key.owner, key.branch_id, key.sequence_id))
                self._metrics["commits"] += 1
                return ChunkCommit("committed", receipt)
          except ChunkConflict:
            self._metrics["conflicts"] += 1
            self._connection.execute(
                "INSERT INTO conflicts(logical_key, proposal_digest, stored_at_ns) "
                "VALUES (?, ?, ?)", (key.logical_key, proposal_digest, _now_ns()))
            raise
        finally:
            self._metrics["commit_latency_ns"].append(time.perf_counter_ns() - start)

    def correct(self, owner: str, branch_id: str, sequence_id: str, *,
                reason: str = "correction",
                next_decision_revision: int | None = None) -> ChunkSnapshot:
        """Invalidate outstanding chunks and keep the committed prefix watermark."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("correction reason required")
        with self._transaction():
            row = self._row(owner, branch_id, sequence_id)
            decision = (int(row["decision_revision"]) + 1
                        if next_decision_revision is None else next_decision_revision)
            if decision <= int(row["decision_revision"]):
                raise ValueError("correction must advance the decision revision")
            self._connection.execute(
                "UPDATE sequences SET revision=?, decision_revision=?, status='active', "
                "updated_ns=? WHERE subject_id=? AND owner=? AND branch_id=? AND sequence_id=?",
                (int(row["revision"]) + 1, decision, _now_ns(), self.subject_id,
                 owner, branch_id, sequence_id))
            row = self._row(owner, branch_id, sequence_id)
            return self._snapshot(row)

    def history(self, owner: str, branch_id: str, sequence_id: str) -> tuple[dict, ...]:
        rows = self._connection.execute(
            "SELECT * FROM commits WHERE subject_id=? AND owner=? AND branch_id=? "
            "AND sequence_id=? ORDER BY revision_after", (self.subject_id, owner,
                                                          branch_id, sequence_id)).fetchall()
        return tuple(dict(row) for row in rows)

    def recover(self) -> dict:
        """Verify receipts, revision progress and the committed watermark chain."""
        rows = self._connection.execute(
            "SELECT * FROM commits ORDER BY subject_id, owner, branch_id, sequence_id, "
            "revision_after").fetchall()
        checked = 0
        sequences: dict[tuple, dict] = {}
        for row in rows:
            receipt = json.loads(row["receipt_json"])
            if receipt["receipt_digest"] != _sha256_json(
                    {key: value for key, value in receipt.items()
                     if key != "receipt_digest"}):
                raise CorruptChunkAuthority("chunk receipt digest is invalid")
            if receipt["proposal_digest"] != row["proposal_digest"]:
                raise CorruptChunkAuthority("chunk proposal digest differs")
            key = (row["subject_id"], row["owner"], row["branch_id"],
                   row["sequence_id"])
            state = sequences.get(key)
            if state is None:
                if int(row["revision_before"]) != 0 or int(row["revision_after"]) < 1:
                    raise CorruptChunkAuthority("chunk revision chain does not start at zero")
                state = {"revision": int(row["revision_after"]),
                         "position": int(row["token_end"]),
                         "prefix": row["prefix_digest"]}
                sequences[key] = state
            else:
                if int(row["revision_before"]) < state["revision"] or \
                        int(row["revision_after"]) != int(row["revision_before"]) + 1 or \
                        int(row["token_start"]) != state["position"]:
                    raise CorruptChunkAuthority("chunk revision or watermark chain is broken")
                state["revision"] = int(row["revision_after"])
                state["position"] = int(row["token_end"])
                state["prefix"] = row["prefix_digest"]
            checked += 1
        for (subject, owner, branch, sequence), state in sequences.items():
            row = self._connection.execute(
                "SELECT * FROM sequences WHERE subject_id=? AND owner=? AND branch_id=? "
                "AND sequence_id=?", (subject, owner, branch, sequence)).fetchone()
            if row is None:
                raise CorruptChunkAuthority("committed sequence row is missing")
            if int(row["revision"]) < state["revision"]:
                raise CorruptChunkAuthority("sequence revision is behind its commits")
            if int(row["revision"]) == state["revision"] and (
                    int(row["committed_position"]) != state["position"]
                    or row["prefix_digest"] != state["prefix"]):
                raise CorruptChunkAuthority("sequence watermark differs from its commits")
        return {"status": "ok", "commits": checked, "sequences": len(sequences)}

    def metrics_document(self) -> dict:
        values = sorted(self._metrics["commit_latency_ns"])

        def percentile(fraction: float) -> float | None:
            if not values:
                return None
            index = min(len(values) - 1,
                        max(0, int(math.ceil(fraction * len(values))) - 1))
            return values[index] / 1e6

        return {"format": FORMAT, "commits": self._metrics["commits"],
                "duplicates": self._metrics["duplicates"],
                "conflicts": self._metrics["conflicts"],
                "stale_rejections": self._metrics["stale"],
                "prefix_refusals": self._metrics["prefix_refusals"],
                "commit_latency_ms": {"p50": percentile(0.5),
                                      "p95": percentile(0.95),
                                      "samples": len(values)}}


class ChunkSession:
    """Session-owned live fast state with an explicit chunk commit barrier.

    ``propose``/``functional_inner_update`` are pure tensor operations.  The
    durable commit receives a detached digest; in inference mode the live state
    is replaced by the detached copy.  In a declared training arm the live graph
    may continue up to ``max_unroll`` chunks; durable checkpoints never carry a
    live graph.
    """

    def __init__(self, store: ChunkStore, snapshot: ChunkSnapshot, state: Tensor, *,
                 committed_tokens: Sequence[int],
                 writer_version: str = "chunk-writer-v1", effect_kind: str = "associate",
                 max_unroll: int = 2, training: bool = False,
                 detach_after_commit: bool = True) -> None:
        if not isinstance(store, ChunkStore):
            raise TypeError("store must be a ChunkStore")
        if not isinstance(snapshot, ChunkSnapshot):
            raise TypeError("snapshot must be a ChunkSnapshot")
        if not isinstance(state, Tensor) or state.ndim != 3 \
                or state.shape[0] != 1 or state.shape[1] != state.shape[2]:
            raise ValueError("state must be [1,D,D]")
        if type(max_unroll) is not int or not 1 <= max_unroll <= MAX_UNROLL:
            raise ValueError("max_unroll is bounded to [1, 8]")
        _identifier("writer_version", writer_version)
        if effect_kind not in EFFECT_KINDS:
            raise ValueError("unknown effect kind")
        self.store = store
        self.snapshot = snapshot
        self.state = state
        self.writer_version = writer_version
        self.effect_kind = effect_kind
        self.max_unroll = max_unroll
        self.training = bool(training)
        self.detach_after_commit = bool(detach_after_commit)
        self._unrolled = 0
        tokens = tuple(int(item) for item in committed_tokens)
        if len(tokens) != snapshot.committed_position:
            raise ValueError("committed token count differs from the watermark")
        if prefix_digest(tokens) != snapshot.prefix_digest:
            raise PrefixChanged("committed token prefix digest differs from the snapshot")
        self._committed_tokens = tokens
        self._chunk_index = 0
        self._pending: Tensor | None = None
        self._pending_tokens: tuple[int, ...] = ()
        self._pending_key: ChunkKey | None = None
        self._last_receipt: ChunkReceipt | None = None

    def begin_chunk(self, tokens: Sequence[int]) -> ChunkKey:
        """Read one immutable snapshot and return the logical key for this chunk."""
        positions = tuple(int(item) for item in tokens)
        if not positions or len(positions) > MAX_TOKENS:
            raise ValueError("a chunk requires a bounded nonempty token range")
        if any(item < 0 for item in positions):
            raise ValueError("token values must be nonnegative integers")
        start = self.snapshot.committed_position
        end = start + len(positions)
        if end > MAX_TOKENS:
            raise ValueError("chunk token range exceeds the bounded maximum")
        key = ChunkKey(
            schema=FORMAT, subject_id=self.snapshot.subject_id, owner=self.snapshot.owner,
            branch_id=self.snapshot.branch_id, sequence_id=self.snapshot.sequence_id,
            decision_revision=self.snapshot.decision_revision,
            chunk_index=self._chunk_index, token_start=start, token_end=end,
            token_prefix_digest=self.snapshot.prefix_digest,
            input_state_revision=self.snapshot.revision,
            writer_version=self.writer_version, effect_kind=self.effect_kind)
        self._pending_tokens = positions
        self._pending_key = key
        return key

    def propose(self, keys: Tensor, values: Tensor, *, decay: float | Tensor = 0.95,
                rate: float | Tensor = 0.5) -> Tensor:
        """Pure live update; no I/O, hash, counter or commit occurs here."""
        if self._unrolled >= self.max_unroll:
            raise ChunkError("differentiable unroll bound reached; commit or detach")
        next_state, _ = functional_inner_update(
            keys, values, self.state, decay=decay, rate=rate)
        self.state = next_state
        self._pending = next_state
        self._unrolled += 1
        return next_state

    def commit(self, *, audit: AttemptAudit | None = None) -> ChunkCommit:
        if self._pending is None or self._pending_key is None:
            raise ChunkError("no proposed chunk state to commit")
        key = self._pending_key
        durable = self._pending.detach().to("cpu").contiguous().clone()
        proposal = tensor_digest(durable)
        next_prefix = prefix_digest(self._committed_tokens + self._pending_tokens)
        result = self.store.barrier(self.snapshot, key=key, proposal_digest=proposal,
                                   next_prefix_digest=next_prefix, audit=audit)
        self._committed_tokens = self._committed_tokens + self._pending_tokens
        self.snapshot = self.store.snapshot(self.snapshot.owner, self.snapshot.branch_id,
                                            self.snapshot.sequence_id)
        self._pending = None
        self._pending_key = None
        self._chunk_index += 1
        self._unrolled = 0
        self._last_receipt = result.receipt
        if self.detach_after_commit or not self.training:
            self.state = durable
        return result

    def recompute(self, keys: Tensor, values: Tensor, *,
                  decay: float | Tensor = 0.95,
                  rate: float | Tensor = 0.5) -> tuple[Tensor, str]:
        """Pure checkpoint recomputation; no durable effect or session mutation."""
        base = self._pending if self._pending is not None else self.state
        next_state, _ = functional_inner_update(
            keys, values, base, decay=decay, rate=rate)
        return next_state, tensor_digest(next_state.detach())

    def invalidate(self) -> None:
        """Discard an uncommitted proposal after a correction or revision change."""
        self._pending = None
        self._pending_key = None
        self._unrolled = 0
