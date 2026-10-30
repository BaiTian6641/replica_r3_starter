"""S2a completed-interaction formation CPU contracts (isolated, no native work)."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time

import pytest

from replica_lab.successors.formation import (CorruptAuthority, FormationConflict,
                                              FormationStore, IneligibleRecord,
                                              InteractionProposal, MemoryPin,
                                              Observation, OutcomeReceipt,
                                              ProvisionalNote, ScopeViolation)

MODE = "parallel_replica"
BRANCH = "branch-a"
PURPOSE = "formation-test"
AS_OF = time.time_ns() + 3600 * 10 ** 9


def make_store(root: Path) -> FormationStore:
    return FormationStore(
        root, subject_id="subject-1", source_epochs=(0,),
        producer_versions={"reader": "r1", "writer": "w1"},
        representation_version="rep-v1", codec_id="codec-v1")


def pin(store: FormationStore, *, branch: str = BRANCH, revision: int = 0,
        as_of: int = AS_OF) -> MemoryPin:
    return store.pin(branch_id=branch, purpose=PURPOSE, mode=MODE, as_of=as_of,
                     task_id="task-1", task_revision=revision,
                     include_verification_states=("verified", "unverified"))


def observation(name: str, text: str, *, lineage=(), supersedes=None) -> Observation:
    return Observation(
        observation_id=name, text=text, source_id="host", source_epoch=0,
        observed_at=time.time_ns(), receipt_id="receipt-" + name,
        verification_state="unverified", authenticity="host_observed",
        source_lineage=tuple(lineage), supersedes=supersedes)


def proposal(pin_value: MemoryPin, observations=(), notes=(), outcomes=(),
             *, interaction: str = "interaction-1",
             branch: str | None = None, source_epoch: int = 0) -> InteractionProposal:
    return InteractionProposal(
        interaction_id=interaction, subject_id=pin_value.subject_id,
        branch_id=branch or pin_value.branch_id, purpose=pin_value.purpose,
        mode=pin_value.mode, task_id=pin_value.task_id,
        task_revision=pin_value.task_revision,
        decision_revision=pin_value.decision_revision + 1,
        source_epoch=source_epoch, producer_versions=pin_value.producer_versions,
        representation_version=pin_value.representation_version,
        codec_id=pin_value.codec_id, permitted_uses=pin_value.permitted_uses,
        observations=tuple(observations), notes=tuple(notes),
        outcomes=tuple(outcomes))


def commit(store, pin_value, observations=(), notes=(), outcomes=(), **kwargs):
    return store.commit_interaction(
        proposal(pin_value, observations, notes, outcomes, **kwargs), pin=pin_value)


def test_commit_is_visible_only_to_later_pin(tmp_path):
    with make_store(tmp_path) as store:
        before_pin = pin(store)
        first = commit(store, before_pin, [observation("o1", "observed value 41")])
        assert first.status == "committed"
        assert store.visible(before_pin) == ()
        later_pin = pin(store)
        visible = store.visible(later_pin)
        assert len(visible) == 1
        assert visible[0].text == "observed value 41"
        assert visible[0].record_id == first.receipt.record_ids[0]
        assert visible[0].generation == first.receipt.generation


def test_duplicate_retry_returns_receipt_and_conflict_fails_closed(tmp_path):
    with make_store(tmp_path) as store:
        current = pin(store)
        value = proposal(current, [observation("o1", "bounded fact")])
        first = store.commit_interaction(value, pin=current)
        duplicate = store.commit_interaction(value, pin=current)
        assert duplicate.status == "duplicate"
        assert duplicate.receipt.receipt_digest == first.receipt.receipt_digest
        changed = proposal(current, [observation("o1", "different bytes")])
        with pytest.raises(FormationConflict):
            store.commit_interaction(changed, pin=current)
        assert store.metrics_document()["conflicts"] == 1
        conflicts = list((tmp_path / "conflicts").glob("*.json"))
        assert len(conflicts) == 1
        assert conflicts[0].read_bytes()


def test_crash_before_and_after_publication(tmp_path):
    prepared = None
    pin_value = None
    with make_store(tmp_path) as store:
        pin_value = pin(store)
        prepared = store.prepare(proposal(pin_value, [observation("o1", "durable")]),
                                 pin=pin_value)
    assert (tmp_path / "payloads" / f"{prepared.payload_digest}.json").is_file()
    with make_store(tmp_path) as store:
        report = store.recover()
        assert report.events == 0
        assert report.orphan_payloads == 1
        assert store.visible(pin_value) == ()
        first = store.commit(prepared)
        assert first.status == "committed"
    with make_store(tmp_path) as store:
        later = pin(store)
        assert len(store.visible(later)) == 1
        recovered = store.recover()
        assert recovered.status == "ok" and recovered.events == 1
        assert recovered.orphan_payloads == 0
        duplicate = store.commit(prepared)
        assert duplicate.status == "duplicate"
        assert duplicate.receipt.receipt_digest == first.receipt.receipt_digest


def test_corrupt_payload_is_detected_but_unrelated_state_survives(tmp_path):
    with make_store(tmp_path) as store:
        pin_value = pin(store)
        first = commit(store, pin_value, [observation("o1", "healthy record")],
                       interaction="interaction-1")
        second = commit(store, pin(store), [observation("o2", "corrupt record")],
                        interaction="interaction-2")
        later = pin(store)
        payload_digest = second.receipt.payload_digest
    (tmp_path / "payloads" / f"{payload_digest}.json").write_bytes(b"{broken")
    with make_store(tmp_path) as store:
        report = store.recover()
        assert report.status == "degraded"
        assert second.receipt.event_key in report.quarantined
        with pytest.raises(CorruptAuthority):
            store.visible(later)
        healthy = store.read_record(first.receipt.record_ids[0], later)
        assert healthy.text == "healthy record"


def test_missing_or_tampered_authority_fails_closed(tmp_path):
    with make_store(tmp_path) as store:
        pin_value = pin(store)
        commit(store, pin_value, [observation("o1", "one")])
    connection = sqlite3.connect(tmp_path / "authority.sqlite3")
    connection.execute("UPDATE authority SET head_digest=?", ("0" * 64,))
    connection.commit()
    connection.close()
    with pytest.raises(CorruptAuthority):
        make_store(tmp_path)

    with make_store(tmp_path / "missing") as store:
        pin_value = pin(store)
        commit(store, pin_value, [observation("o1", "one")])
    connection = sqlite3.connect(tmp_path / "missing" / "authority.sqlite3")
    connection.execute("DELETE FROM authority")
    connection.commit()
    connection.close()
    with pytest.raises(CorruptAuthority):
        make_store(tmp_path / "missing")


def test_scope_branch_and_source_epoch_are_refused(tmp_path):
    with make_store(tmp_path) as store:
        current = pin(store)
        wrong_branch = proposal(current, [observation("o1", "x")], branch="branch-b")
        with pytest.raises(ScopeViolation):
            store.prepare(wrong_branch, pin=current)
        wrong_epoch = proposal(current, [observation("o1", "x")], source_epoch=9)
        with pytest.raises(ScopeViolation):
            store.prepare(wrong_epoch, pin=current)
        wrong_task = InteractionProposal(
            interaction_id="i", subject_id=current.subject_id,
            branch_id=current.branch_id, purpose=current.purpose,
            mode=current.mode, task_id="other-task",
            task_revision=current.task_revision, decision_revision=1,
            source_epoch=0, producer_versions=current.producer_versions,
            representation_version=current.representation_version,
            codec_id=current.codec_id, permitted_uses=current.permitted_uses,
            observations=(observation("o1", "x"),))
        with pytest.raises(ScopeViolation):
            store.prepare(wrong_task, pin=current)
        prepared = store.prepare(proposal(current, [observation("o1", "x")]),
                                 pin=current)
        with pytest.raises(Exception):
            store.commit(prepared, expected_predecessor=5)


def test_correction_is_temporal_and_does_not_leak_backward(tmp_path):
    with make_store(tmp_path) as store:
        base = commit(store, pin(store), [observation("base", "dated fact")],
                      interaction="interaction-base")
        record_id = base.receipt.record_ids[0]
        historical_cutoff = time.time_ns()
        historical = pin(store, as_of=historical_cutoff)
        store.correct(record_id, text="corrected later", reason="late evidence",
                      pin=pin(store), verifier_receipt_id="verifier-late")
        now_view = pin(store)
        assert record_id in {item.record_id for item in store.visible(historical)}
        assert record_id not in {item.record_id for item in store.visible(now_view)}
        assert record_id not in {item.record_id
                                 for item in store.visible(pin(store, as_of=AS_OF))}


def test_correction_invalidates_parent_and_descendants(tmp_path):
    with make_store(tmp_path) as store:
        base = commit(store, pin(store), [observation("base", "base fact")],
                      interaction="interaction-base")
        parent_id = base.receipt.record_ids[0]
        child = commit(store, pin(store),
                       [observation("child", "derived fact", lineage=(parent_id,))],
                       interaction="interaction-child")
        child_id = child.receipt.record_ids[0]
        correction_pin = pin(store)
        corrected = store.correct(
            parent_id, text="corrected fact", reason="host review",
            pin=correction_pin, verifier_receipt_id="verifier-correction")
        assert corrected.status == "committed"
        later = pin(store)
        visible = {record.record_id for record in store.visible(later)}
        assert parent_id not in visible
        assert child_id not in visible
        assert corrected.receipt.record_ids[0] in visible
        with pytest.raises(IneligibleRecord):
            store.correct(parent_id, text="again", reason="repeat",
                          pin=pin(store))
        revoked = store.revoke(corrected.receipt.record_ids[0], reason="source revoked")
        assert revoked.status == "committed"
        after = store.visible(pin(store))
        assert {record.record_id for record in after} == set()


def test_revocation_is_idempotent_and_conflicting_reason_conflicts(tmp_path):
    with make_store(tmp_path) as store:
        record = commit(store, pin(store), [observation("o1", "revocable")])
        record_id = record.receipt.record_ids[0]
        first = store.revoke(record_id, reason="revoked")
        duplicate = store.revoke(record_id, reason="revoked")
        assert duplicate.status == "duplicate"
        assert duplicate.receipt.receipt_digest == first.receipt.receipt_digest
        with pytest.raises(FormationConflict):
            store.revoke(record_id, reason="another reason")
        with pytest.raises(IneligibleRecord):
            store.read_record(record_id, pin(store))


def test_provisional_explanation_is_not_evidence(tmp_path):
    with make_store(tmp_path) as store:
        current = pin(store)
        note = ProvisionalNote(note_id="n1", text="the model explained the answer",
                               origin="model_generated", interaction_id="interaction-1")
        outcome = OutcomeReceipt(outcome_id="u1", action_id="a1", status="completed",
                                 receipt_id="receipt-u1", observed_at=time.time_ns(),
                                 summary="a1 completed")
        report = commit(store, current, notes=[note], outcomes=[outcome])
        assert report.status == "committed"
        assert store.visible(pin(store)) == ()
        assert len(store.outcomes(pin(store))) == 1
        provisional = store.read_provisional(branch_id=BRANCH, purpose=PURPOSE)
        assert len(provisional) == 1
        assert report.receipt.note_ids == ("n1",)
        assert report.receipt.outcome_ids == ("u1",)


def test_index_rebuild_cannot_invent_events(tmp_path):
    with make_store(tmp_path) as store:
        current = pin(store)
        first = commit(store, current, [observation("o1", "indexed")])
        later = pin(store)
        before = tuple(record.record_id for record in store.visible(later))
        authority = store._authority()
        assert authority["sequence"] == 1
    with make_store(tmp_path) as store:
        assert store.recovery_report.index_rebuilt is False
        connection = sqlite3.connect(tmp_path / "authority.sqlite3")
        connection.execute("DELETE FROM record_index")
        connection.execute("DELETE FROM record_lineage")
        connection.commit()
        connection.close()
        report = store.recover()
        assert report.index_rebuilt is True
        assert report.events == 1 and report.sequence == 1
        assert tuple(record.record_id for record in store.visible(later)) == before
        assert store._authority()["sequence"] == 1


def test_receipts_bind_predecessor_and_exactly_once_generation(tmp_path):
    with make_store(tmp_path) as store:
        first = commit(store, pin(store), [observation("o1", "one")],
                       interaction="interaction-1")
        second = commit(store, pin(store), [observation("o2", "two")],
                        interaction="interaction-2")
        assert second.receipt.predecessor_sequence == first.receipt.sequence
        assert second.receipt.predecessor_digest == first.receipt.receipt_digest
        assert second.receipt.generation == first.receipt.generation + 1
        assert store.recover().events == 2
