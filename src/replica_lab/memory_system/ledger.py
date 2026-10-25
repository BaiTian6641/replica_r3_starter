"""Exact, temporal source records for the E2 memory path.

The host supplies authenticated scope. Record text never grants permissions.
These checks are logical isolation within a trusted process, not an operating
system security boundary. Revocation redacts live payloads and invalidates
selections; it does not erase backups, retained tensors, or trained weights.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Iterator
import uuid


MODES = frozenset({
    "assistant_for_source", "parallel_replica", "source_current", "source_snapshot",
})
AUTHENTICITY_STATUSES = frozenset({
    "unverified", "source_report", "verified_source", "observed_tool", "agent_report",
})


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string without NUL")


def _timestamp(name: str, value: object) -> None:
    # bool is not a logical timestamp despite being an int subclass.
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be a nonnegative SQLite integer timestamp")


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One immutable, attributed record; times use the caller's declared clock.

    `observed_at` is the causal cutoff. `event_at` describes the event and may
    differ, including a known future appointment. Authenticity is a host-supplied
    evidence label, never an automatic assertion that a report is true.
    """

    record_id: str
    subject_id: str
    branch_id: str
    text: str
    observed_at: int
    event_at: int
    purpose: str
    kind: str = "source"
    supersedes: str | None = None
    source_id: str | None = None
    authenticity: str = "unverified"

    def __post_init__(self) -> None:
        for name in ("record_id", "subject_id", "branch_id", "purpose", "kind"):
            _identifier(name, getattr(self, name))
        if not isinstance(self.text, str):
            raise ValueError("text must be an exact string")
        # Fail before hashing or SQL on invalid surrogate code points.
        self.text.encode("utf-8")
        for name in ("observed_at", "event_at"):
            _timestamp(name, getattr(self, name))
        for name in ("supersedes", "source_id"):
            if getattr(self, name) is not None:
                _identifier(name, getattr(self, name))
        if self.supersedes == self.record_id:
            raise ValueError("a record cannot supersede itself")
        if self.authenticity not in AUTHENTICITY_STATUSES:
            raise ValueError("unknown authenticity status")

    @property
    def record_hash(self) -> str:
        """Bind content, scope, chronology, provenance and correction metadata."""
        return _hash(_canonical(asdict(self)))

    @property
    def source_hash(self) -> str:
        """Hash exact UTF-8 text without whitespace or Unicode normalization."""
        return _hash(self.text)


@dataclass(frozen=True, slots=True)
class EligibilityScope:
    subject_id: str
    branch_id: str
    purpose: str
    as_of: int
    mode: str = "assistant_for_source"
    snapshot_at: int | None = None

    def __post_init__(self) -> None:
        for name in ("subject_id", "branch_id", "purpose"):
            _identifier(name, getattr(self, name))
        _timestamp("as_of", self.as_of)
        if self.mode not in MODES:
            raise ValueError("unknown identity mode")
        if self.snapshot_at is not None:
            _timestamp("snapshot_at", self.snapshot_at)
        if self.mode == "source_snapshot" and self.snapshot_at is None:
            raise ValueError("source_snapshot requires snapshot_at")
        if self.mode != "source_snapshot" and self.snapshot_at is not None:
            raise ValueError("snapshot_at is only valid for source_snapshot")

    @property
    def cutoff(self) -> int:
        if self.mode == "source_snapshot":
            assert self.snapshot_at is not None
            return min(self.as_of, self.snapshot_at)
        return self.as_of


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    records: tuple[MemoryRecord, ...]
    epoch: int
    ledger_id: str
    scope: EligibilityScope


@dataclass(frozen=True, slots=True)
class ExactSpan:
    """An exact Unicode-code-point span; `end` is exclusive."""

    record_id: str
    text: str
    start: int
    end: int
    record_hash: str
    source_hash: str
    span_hash: str
    epoch: int


class RecordLedger:
    """Durable records, correction metadata and a conservative invalidation epoch.

    Each material admission, correction or revocation increments the global
    epoch. Even an unrelated addition expires old selections. This deliberately
    favors a simple safe reference contract over narrowly scoped cache reuse.
    One instance owns one SQLite connection; use a distinct instance per thread.
    """

    SCHEMA_VERSION = "1"

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._db = sqlite3.connect(self.path, isolation_level=None, timeout=10.0)
        self._db.execute("PRAGMA foreign_keys=ON")
        try:
            with self._transaction(write=True):
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS memory_metadata "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                self._db.execute("""
                    CREATE TABLE IF NOT EXISTS memory_records (
                        record_id TEXT PRIMARY KEY,
                        subject_id TEXT NOT NULL,
                        branch_id TEXT NOT NULL,
                        purpose TEXT NOT NULL,
                        observed_at INTEGER NOT NULL,
                        event_at INTEGER NOT NULL,
                        supersedes TEXT REFERENCES memory_records(record_id),
                        record_hash TEXT NOT NULL,
                        source_hash TEXT NOT NULL,
                        payload TEXT,
                        admitted_at_ns INTEGER NOT NULL,
                        revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1)),
                        CHECK((revoked=0 AND payload IS NOT NULL) OR
                              (revoked=1 AND payload IS NULL)))
                """)
                self._db.execute(
                    "CREATE INDEX IF NOT EXISTS memory_scope_time ON memory_records "
                    "(subject_id,purpose,branch_id,observed_at)")
                self._db.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS memory_one_correction "
                    "ON memory_records(supersedes) WHERE supersedes IS NOT NULL")
                for key, value in (("schema_version", self.SCHEMA_VERSION),
                                   ("ledger_id", uuid.uuid4().hex), ("epoch", "0")):
                    self._db.execute("INSERT OR IGNORE INTO memory_metadata VALUES(?,?)",
                                     (key, value))
                if self._meta("schema_version") != self.SCHEMA_VERSION:
                    raise ValueError("unsupported memory ledger schema")
        except BaseException:
            self._db.close()
            raise

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[None]:
        self._db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield
        except BaseException:
            self._db.rollback()
            raise
        else:
            self._db.commit()

    def _meta(self, key: str) -> str:
        row = self._db.execute("SELECT value FROM memory_metadata WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError(f"missing ledger metadata: {key}")
        return row[0]

    def record_status(self, record_id: str) -> dict | None:
        """Read-only immutable binding of one row, including revoked/superseded rows."""
        if not isinstance(record_id, str):
            raise TypeError("record_id must be text")
        row = self._db.execute(
            "SELECT record_id,supersedes,record_hash,source_hash,revoked "
            "FROM memory_records WHERE record_id=?", (record_id,)).fetchone()
        if row is None:
            return None
        return {"record_id": row[0], "supersedes": row[1], "record_hash": row[2],
                "source_hash": row[3], "revoked": bool(row[4])}

    def revoked_source_hashes(self) -> tuple[str, ...]:
        """Exact source-text hashes removed from every eligible view."""
        rows = self._db.execute(
            "SELECT source_hash FROM memory_records WHERE revoked=1").fetchall()
        return tuple(row[0] for row in rows)

    @property
    def epoch(self) -> int:
        return int(self._meta("epoch"))

    @property
    def ledger_id(self) -> str:
        return self._meta("ledger_id")

    def _advance_epoch(self) -> None:
        self._db.execute("UPDATE memory_metadata SET value=? WHERE key='epoch'",
                         (str(self.epoch + 1),))

    @staticmethod
    def _authorize(record_subject: str, record_branch: str, *,
                   authorized_subject: str, authorized_branch: str,
                   source_authorized: bool) -> None:
        if (record_subject, record_branch) != (authorized_subject, authorized_branch):
            raise PermissionError("record scope does not match host authority")
        if record_branch == "source" and source_authorized is not True:
            raise PermissionError("source archive requires explicit source authority")

    def admit(self, record: MemoryRecord, *, authorized_subject: str,
              authorized_branch: str, source_authorized: bool = False) -> bool:
        if not isinstance(record, MemoryRecord):
            raise TypeError("admit requires a MemoryRecord")
        self._authorize(record.subject_id, record.branch_id,
                        authorized_subject=authorized_subject,
                        authorized_branch=authorized_branch,
                        source_authorized=source_authorized)
        with self._transaction(write=True):
            old = self._db.execute(
                "SELECT record_hash,revoked FROM memory_records WHERE record_id=?",
                (record.record_id,)).fetchone()
            if old:
                if old[1]:
                    raise ValueError("revoked identifiers cannot be reused")
                if old[0] != record.record_hash:
                    raise ValueError("immutable record identifier collision")
                return False
            if record.supersedes is not None:
                parent = self._db.execute(
                    "SELECT subject_id,branch_id,purpose,observed_at,revoked "
                    "FROM memory_records WHERE record_id=?", (record.supersedes,)).fetchone()
                if not parent or parent[4] or parent[:3] != (
                        record.subject_id, record.branch_id, record.purpose):
                    raise PermissionError("correction must stay in a live subject, branch and purpose")
                if record.observed_at <= parent[3]:
                    raise ValueError("correction must be observed after its parent")
                if self._db.execute("SELECT 1 FROM memory_records WHERE supersedes=?",
                                    (record.supersedes,)).fetchone():
                    raise ValueError("correct the current descendant; correction forks are forbidden")
            self._db.execute(
                "INSERT INTO memory_records VALUES(?,?,?,?,?,?,?,?,?,?,?,0)",
                (record.record_id, record.subject_id, record.branch_id, record.purpose,
                 record.observed_at, record.event_at, record.supersedes,
                 record.record_hash, record.source_hash, _canonical(asdict(record)), time.time_ns()))
            self._advance_epoch()
        return True

    def _eligible(self, scope: EligibilityScope) -> tuple[MemoryRecord, ...]:
        branches = ("source",) if scope.mode in {"source_current", "source_snapshot"} else (
            tuple(dict.fromkeys(("source", scope.branch_id))))
        placeholders = ",".join("?" for _ in branches)
        rows = self._db.execute(
            "SELECT record_id,subject_id,branch_id,purpose,observed_at,event_at,"
            "supersedes,record_hash,source_hash,payload,revoked FROM memory_records "
            "WHERE subject_id=? AND purpose=? AND observed_at<=? "
            f"AND branch_id IN ({placeholders}) ORDER BY observed_at,event_at,record_id",
            (scope.subject_id, scope.purpose, scope.cutoff, *branches)).fetchall()
        # Include revoked correction metadata. Deleting a correction must never
        # make its earlier assertion current again. A genuinely earlier causal
        # view still contains that earlier assertion unless it was itself revoked.
        superseded = {row[6] for row in rows if row[6] is not None}
        selected: list[MemoryRecord] = []
        for row in rows:
            if row[10] or row[0] in superseded:
                continue
            record = MemoryRecord(**json.loads(row[9]))
            if (record.record_id, record.subject_id, record.branch_id, record.purpose,
                    record.observed_at, record.event_at, record.supersedes,
                    record.record_hash, record.source_hash) != row[:9]:
                raise ValueError("ledger payload no longer matches its immutable source binding")
            selected.append(record)
        return tuple(selected)

    def snapshot(self, scope: EligibilityScope) -> LedgerSnapshot:
        if not isinstance(scope, EligibilityScope):
            raise TypeError("snapshot requires an EligibilityScope")
        with self._transaction():
            return LedgerSnapshot(self._eligible(scope), self.epoch, self.ledger_id, scope)

    def eligible(self, *, subject_id: str, branch_id: str, purpose: str, as_of: int,
                 mode: str = "assistant_for_source",
                 snapshot_at: int | None = None) -> tuple[MemoryRecord, ...]:
        return self.snapshot(EligibilityScope(subject_id, branch_id, purpose, as_of,
                                              mode, snapshot_at)).records

    select = eligible

    def fetch_exact(self, record_id: str, *, subject_id: str, branch_id: str,
                    purpose: str, as_of: int, mode: str = "assistant_for_source",
                    snapshot_at: int | None = None, start: int = 0,
                    end: int | None = None, expected_hash: str | None = None,
                    expected_epoch: int | None = None) -> ExactSpan:
        snap = self.snapshot(EligibilityScope(subject_id, branch_id, purpose, as_of,
                                              mode, snapshot_at))
        if expected_epoch is not None and expected_epoch != snap.epoch:
            raise ValueError("exact fetch epoch is stale")
        record = next((r for r in snap.records if r.record_id == record_id), None)
        if record is None:
            raise PermissionError("exact fetch record is absent or ineligible")
        if expected_hash is not None and expected_hash != record.record_hash:
            raise ValueError("exact fetch source hash mismatch")
        if end is None:
            end = len(record.text)
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(record.text):
            raise ValueError("exact span must use valid Unicode-code-point offsets")
        text = record.text[start:end]
        return ExactSpan(record.record_id, text, start, end, record.record_hash,
                         record.source_hash, _hash(text), snap.epoch)

    def revoke(self, record_id: str, *, authorized_subject: str,
               authorized_branch: str, source_authorized: bool = False) -> tuple[str, ...]:
        """Redact this source and all its correction descendants, atomically."""
        with self._transaction(write=True):
            parent = self._db.execute(
                "SELECT subject_id,branch_id FROM memory_records WHERE record_id=?",
                (record_id,)).fetchone()
            if parent is None:
                raise PermissionError("record is absent or unauthorized")
            self._authorize(*parent, authorized_subject=authorized_subject,
                            authorized_branch=authorized_branch,
                            source_authorized=source_authorized)
            rows = self._db.execute("""
                WITH RECURSIVE descendants(record_id) AS (
                    SELECT record_id FROM memory_records WHERE record_id=?
                    UNION ALL SELECT r.record_id FROM memory_records r
                    JOIN descendants d ON r.supersedes=d.record_id)
                SELECT r.record_id FROM memory_records r JOIN descendants d
                ON r.record_id=d.record_id WHERE r.revoked=0 ORDER BY r.record_id
            """, (record_id,)).fetchall()
            removed = tuple(row[0] for row in rows)
            self._db.executemany(
                "UPDATE memory_records SET payload=NULL,revoked=1 WHERE record_id=?",
                ((rid,) for rid in removed))
            if removed:
                self._advance_epoch()
            return removed

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> RecordLedger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
