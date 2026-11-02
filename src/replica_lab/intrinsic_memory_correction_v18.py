"""Bounded authoritative current view backed by separate immutable history."""
from dataclasses import replace
import hashlib
import json

from .contextual_memory import ContextualMemoryError, canonical_spans
from .intrinsic_memory_coverage_v12 import COMPOUND_FAMILIES, _render_compound
from .intrinsic_memory_budget_v12 import atomic
from .tiered_memory import RetrievalBudget


def current_view(row):
    # Match chronological_correction's schema dispatch: legacy direct facts
    # are lists, including synthetic lifecycle fixtures with family labels.
    if isinstance(row.get("facts"), dict) and row.get("rule_family") in COMPOUND_FAMILIES:
        history = row["revision_history"]
        facts = {**row["facts"], **{k: history[-1][k] for k in ("answer", "anchor_colour")},
                 "unrecorded_target": False}
        return _render_compound(facts, row["rule_family"])
    latest = {}
    for fact in row["facts"]:
        if fact["entity"] not in latest or fact["revision"] > latest[fact["entity"]]["revision"]:
            latest[fact["entity"]] = fact
    return " ".join(f"The recorded color of {name} is {fact['color']}."
                    for name, fact in sorted(latest.items())) + " No color is recorded for any other entity."


def _history_document(row, span):
    return {"format": "replica.correction_history.v18", "row": row,
            "predecessor": span.supersedes, "current_record": span.record_id,
            "current_view": span.text,
            "current_view_sha256": hashlib.sha256(span.text.encode()).hexdigest()}


def _history_bytes(history):
    return (json.dumps(history, sort_keys=True, indent=2) + "\n").encode()


def correction_evaluation_view(row, span, publication):
    """Bind both controls to published evidence, leaving the history row intact.

    The minimal evaluation row deliberately contains no chronological source.
    Its provenance points to the complete, separately stored history. Recompute
    that binding in memory; no unreserved history read or source truncation.
    """
    history = _history_document(row, span)
    identity = hashlib.sha256(_history_bytes(history)).hexdigest()
    if (span.text != publication["current_text"] or span.text != current_view(row)
            or span.record_id != publication["current_record"]
            or span.source_id != row["world_id"] or span.source_epoch != row["chronology"]
            or history["current_view_sha256"] != publication["current_text_sha256"]
            or identity != publication["history_sha256"]
            or publication["history_path"] != f"correction_history/{identity}.json"):
        raise ContextualMemoryError("correction evaluation publication/history binding mismatch")
    return {"format": "replica.correction_evaluation_view.v18", "world_id": row["world_id"],
            "question": row["question"], "answer": row["answer"], "text": span.text,
            "chronology": row["chronology"],
            "provenance": {"current_record": span.record_id, "predecessor": span.supersedes,
                           "current_text_sha256": history["current_view_sha256"],
                           "history_path": publication["history_path"],
                           "history_sha256": identity}}


def publish_correction(engine, store, old, row, scope):
    """One atomic store correction; history is durable before the authority flip.

    This deliberately supports a one-span current view, within the existing
    two-span/eight-slot ceiling. Oversize views fail *before* publication. No
    partial view or first-span truncation is ever admitted.
    """
    if len(old.spans) != 1:
        raise ContextualMemoryError("bounded-current-view correction requires one predecessor")
    view = current_view(row)
    spans = canonical_spans(view, record_id=row["world_id"] + f".revision{row['chronology']}",
                            source_id=row["world_id"], source_epoch=row["chronology"],
                            available_at=row["chronology"])
    if len(spans) != 1:
        raise ContextualMemoryError("complete correction current view exceeds admitted single-span budget")
    span = replace(spans[0], supersedes=old.spans[0].record_id)
    history = _history_document(row, span)
    payload = _history_bytes(history)
    identity = hashlib.sha256(payload).hexdigest()
    destination = engine.output / "correction_history" / (identity + ".json")
    with engine.measured("immutable_correction_history", writes=1, artifact_bytes=len(payload)):
        destination.parent.mkdir(exist_ok=True, parents=True)
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ContextualMemoryError("immutable correction history conflict")
        else:
            atomic(destination, history)
    with engine.measured("bounded_current_view_correction", writes=1):
        result = engine.path.correct(store, old.spans[0].record_id, span, scope=scope)
        if not result.outcome.ok:
            raise ContextualMemoryError("correction publication failed: " + str(result.outcome))
    return span, {"history_path": str(destination.relative_to(engine.output)),
                  "history_sha256": identity, "current_text": view,
                  "current_text_sha256": history["current_view_sha256"],
                  "current_record": span.record_id,
                  "current_span_count": len(spans), "current_slots": 4}


def fresh_revocation_check(store, question, scope, revoked_ids):
    """Exercise fresh selection separately from invalidating a cached snapshot."""
    selected = store.select(question, scope=scope,
                            budget=RetrievalBudget(max_records=2, max_value_slots=8),
                            include_text=False, admit_payload=False)
    empty_eligible = (selected.outcome.reason == "no_semantic_candidate"
                      and selected.candidate_count == 0 and not selected.records
                      and not selected.representation_stale)
    if (not selected.outcome.ok and not empty_eligible) or selected.truncated:
        raise ContextualMemoryError("fresh post-revocation retrieval failed operationally")
    for record_id in revoked_ids:
        row = store.connection.execute("SELECT status FROM records WHERE record_id=?", (record_id,)).fetchone()
        if row is None or row["status"] != "revoked":
            raise ContextualMemoryError("fresh retrieval lacks explicit revocation authority")
    ids = [record.record_id for record in selected.records]
    if set(ids) & set(revoked_ids):
        raise ContextualMemoryError("revoked evidence remained freshly retrievable")
    return {"selected_record_ids": ids, "revoked_record_ids": list(revoked_ids),
            "revoked_ids_excluded": True, "io_bytes": selected.io_bytes,
            "file_reads": selected.file_reads, "outcome": str(selected.outcome)}
