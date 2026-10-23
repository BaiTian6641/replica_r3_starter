"""Deterministic candidate retrieval after exact host eligibility filtering.

Dense features are supplied by an explicit external producer, not invented by
this module. A learned reranker may reorder these permitted candidates; it must
preserve the source/epoch scope pin before a model consumes the selection.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
from typing import Mapping, Sequence

from .ledger import EligibilityScope, MemoryRecord, RecordLedger, _canonical, _hash, _identifier


def _tokens(text: str) -> Counter[str]:
    return Counter(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dense vectors have incompatible dimensions")
    # Normalize scales before multiplication so large/small finite features do
    # not overflow/underflow into a spurious rank or a nonfinite score.
    left_scale, right_scale = max(map(abs, left)), max(map(abs, right))
    if not left_scale or not right_scale:
        return 0.0
    lhs, rhs = tuple(v / left_scale for v in left), tuple(v / right_scale for v in right)
    numerator = math.fsum(a * b for a, b in zip(lhs, rhs))
    left_norm = math.sqrt(math.fsum(a * a for a in lhs))
    right_norm = math.sqrt(math.fsum(b * b for b in rhs))
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _vector(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if not result or not all(math.isfinite(v) for v in result):
        raise ValueError("dense vector must be nonempty and finite")
    return result


@dataclass(frozen=True, slots=True)
class DenseRecordVector:
    record_hash: str
    values: tuple[float, ...]
    producer_id: str

    def __post_init__(self) -> None:
        _identifier("producer_id", self.producer_id)
        if not re.fullmatch(r"[0-9a-f]{64}", self.record_hash):
            raise ValueError("dense record vector requires an exact record SHA256")
        object.__setattr__(self, "values", _vector(self.values))


@dataclass(frozen=True, slots=True)
class RetrievalSelection:
    records: tuple[MemoryRecord, ...]
    scores: tuple[float, ...]
    epoch: int
    ledger_id: str
    scope: EligibilityScope
    query: str
    dense_producer_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "scores", tuple(self.scores))
        if not isinstance(self.scope, EligibilityScope):
            raise TypeError("selection scope must be an EligibilityScope")
        if len(self.records) != len(self.scores):
            raise ValueError("one score is required per selected record")
        if not all(isinstance(r, MemoryRecord) for r in self.records):
            raise TypeError("selection records must be immutable MemoryRecords")
        if len({r.record_id for r in self.records}) != len(self.records):
            raise ValueError("selection cannot repeat a record")
        if not all(math.isfinite(score) for score in self.scores):
            raise ValueError("selection scores must be finite")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("invalid selection epoch")
        _identifier("ledger_id", self.ledger_id)
        if not isinstance(self.query, str):
            raise ValueError("query must be a string")

    @property
    def record_hashes(self) -> tuple[str, ...]:
        return tuple(record.record_hash for record in self.records)

    @property
    def selection_hash(self) -> str:
        return _hash(_canonical({
            "record_hashes": self.record_hashes, "scores": self.scores,
            "epoch": self.epoch, "ledger_id": self.ledger_id,
            "scope": asdict(self.scope), "query": self.query,
            "dense_producer_id": self.dense_producer_id,
        }))

    @property
    def subject_id(self) -> str:
        return self.scope.subject_id

    @property
    def branch_id(self) -> str:
        return self.scope.branch_id

    @property
    def purpose(self) -> str:
        return self.scope.purpose

    @property
    def as_of(self) -> int:
        return self.scope.as_of

    @property
    def mode(self) -> str:
        return self.scope.mode

    @property
    def snapshot_at(self) -> int | None:
        return self.scope.snapshot_at


class RetrievalIndex:
    def __init__(self, ledger: RecordLedger):
        self.ledger = ledger

    def retrieve(self, query: str, *, subject_id: str, branch_id: str, purpose: str,
                 as_of: int, mode: str = "assistant_for_source", snapshot_at: int | None = None,
                 limit: int = 16, dense_query: Sequence[float] | None = None,
                 dense_vectors: Mapping[str, DenseRecordVector] | None = None,
                 dense_producer_id: str | None = None, lexical_weight: float = 1.0,
                 dense_weight: float = 1.0, min_score: float = 0.0) -> RetrievalSelection:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        if (not all(math.isfinite(v) for v in (lexical_weight, dense_weight, min_score))
                or lexical_weight < 0 or dense_weight < 0):
            raise ValueError("retrieval weights must be finite and nonnegative")
        if dense_query is None:
            if dense_vectors is not None or dense_producer_id is not None:
                raise ValueError("dense records/producer require a dense query")
            dense = None
        else:
            if dense_vectors is None or dense_producer_id is None:
                raise ValueError("dense retrieval requires source-bound vectors and producer identity")
            _identifier("dense_producer_id", dense_producer_id)
            dense = _vector(dense_query)
        scope = EligibilityScope(subject_id, branch_id, purpose, as_of, mode, snapshot_at)
        snap = self.ledger.snapshot(scope)
        terms = _tokens(query)
        query_norm = math.sqrt(sum(value * value for value in terms.values()))
        ranked: list[tuple[float, MemoryRecord]] = []
        # Only eligible records are inspected or scored. Unrelated dense-vector
        # entries cannot affect ranking or validation of this owner's query.
        for record in snap.records:
            counts = _tokens(record.text)
            norm = math.sqrt(sum(value * value for value in counts.values()))
            lexical = (sum(value * counts.get(term, 0) for term, value in terms.items())
                       / (query_norm * norm)) if query_norm and norm else 0.0
            score = lexical_weight * lexical
            if dense is not None and record.record_id in dense_vectors:
                feature = dense_vectors[record.record_id]
                if not isinstance(feature, DenseRecordVector):
                    raise TypeError("dense record features must be source-bound DenseRecordVectors")
                if feature.record_hash != record.record_hash:
                    raise ValueError("dense feature source record hash mismatch")
                if feature.producer_id != dense_producer_id:
                    raise ValueError("dense feature producer mismatch")
                score += dense_weight * _cosine(dense, feature.values)
            if not math.isfinite(score):
                raise ValueError("retrieval score is nonfinite")
            if score > min_score:
                ranked.append((score, record))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].observed_at, pair[1].record_id))
        selected = ranked[:limit]
        return RetrievalSelection(tuple(record for _, record in selected),
                                  tuple(score for score, _ in selected), snap.epoch,
                                  snap.ledger_id, scope, query, dense_producer_id)

    def verify_selection(self, selection: RetrievalSelection, *,
                         expected_scope: EligibilityScope | None = None) -> None:
        if not isinstance(selection, RetrievalSelection):
            raise TypeError("verification requires a RetrievalSelection")
        if expected_scope is not None and selection.scope != expected_scope:
            raise PermissionError("selection scope does not match the request")
        snap = self.ledger.snapshot(selection.scope)
        if (selection.ledger_id, selection.epoch) != (snap.ledger_id, snap.epoch):
            raise ValueError("retrieval selection is stale or belongs to another ledger")
        eligible = {record.record_id: record for record in snap.records}
        for record in selection.records:
            if eligible.get(record.record_id) != record:
                raise PermissionError("selected source is absent, altered, or no longer eligible")

