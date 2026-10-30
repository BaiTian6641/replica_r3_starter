"""Isolated CPU later-answer comparison for S2a formation.

This is engineering behavior on fictional fixtures.  It shows whether one
completed-interaction write becomes visible to a later interaction under the
same reader, query and evidence opportunity, and whether removing the write
removes the apparent improvement.  It is not native recall, personal fidelity or
formation utility evidence.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import time
from typing import Iterable, Sequence

from .formation import (FormationConflict, FormationRecord, FormationStore,
                        InteractionProposal, MemoryPin, Observation,
                        OutcomeReceipt, ProvisionalNote)

QUERY = "vault code"
BASE_TEXT = "vault policy: approved operators rotate the key at the main desk"
NEW_TEXT = "vault code 4127 was observed for shelf seven"
PROCEDURE_TEXT = "procedure: verify the receipt before writing any answer"
UNKNOWN_ANSWER = "unknown"
PURPOSE = "successor-study"
MODE = "parallel_replica"


def _now_ns() -> int:
    return time.time_ns()


def _pin(store: FormationStore, branch_id: str, *, revision: int = 0) -> MemoryPin:
    return store.pin(branch_id=branch_id, purpose=PURPOSE, mode=MODE,
                     as_of=_now_ns() + 3600 * 10 ** 9, task_id="study-task",
                     task_revision=revision,
                     include_verification_states=("verified", "unverified"))


def answer_query(query: str, records: Iterable[FormationRecord]) -> dict:
    """Deterministic extractive reader; identical across every condition."""
    tokens = [token for token in query.casefold().split() if token]
    matches = [record for record in records
               if all(token in record.text.casefold() for token in tokens)]
    if not matches:
        return {"answer": UNKNOWN_ANSWER, "record_id": None,
                "evidence": tuple(record.record_id for record in records)}
    chosen = matches[-1]
    return {"answer": chosen.text, "record_id": chosen.record_id,
            "evidence": tuple(record.record_id for record in records)}


def _observation(text: str, *, index: int, verification: str = "verified") -> Observation:
    return Observation(
        observation_id=f"study-observation-{index}",
        text=text, source_id="study-source", source_epoch=0,
        observed_at=_now_ns(), receipt_id=f"study-receipt-{index}",
        verification_state=verification,
        verifier_receipt_id=(f"study-verifier-{index}"
                             if verification == "verified" else None),
        permitted_uses=("retrieval",), importance=10)


def _commit(store: FormationStore, pin: MemoryPin, observations: Sequence[Observation],
            *, interaction: str, notes: Sequence[ProvisionalNote] = (),
            outcomes: Sequence[OutcomeReceipt] = ()):
    proposal = InteractionProposal(
        interaction_id=interaction, subject_id=pin.subject_id,
        branch_id=pin.branch_id, purpose=pin.purpose, mode=pin.mode,
        task_id=pin.task_id, task_revision=pin.task_revision,
        decision_revision=pin.decision_revision + 1, source_epoch=0,
        producer_versions=pin.producer_versions,
        representation_version=pin.representation_version,
        codec_id=pin.codec_id, permitted_uses=pin.permitted_uses,
        observations=tuple(observations), outcomes=tuple(outcomes),
        notes=tuple(notes))
    return store.commit_interaction(proposal, pin=pin)


def _store(root: Path) -> FormationStore:
    return FormationStore(
        root, subject_id="fictional-subject", source_epochs=(0,),
        producer_versions={"reader": "successor-study-v1", "writer": "s2a-v1"},
        representation_version="rep-successor-study-v1",
        codec_id="codec-successor-study-v1")


def run_later_answer_comparison(root: Path | str) -> dict:
    """Run enabled/disabled, delayed, branch, correction and revocation cases."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"format": "replica.successors.study.v1",
                                 "query": QUERY, "conditions": {}}

    # Enabled: base evidence plus one completed-interaction write.
    enabled_root = root / "enabled"
    with _store(enabled_root) as store:
        branch = "replica-branch-a"
        base_pin = _pin(store, branch)
        base_commit = _commit(store, base_pin, [
            _observation(BASE_TEXT + " [base]", index=1, verification="unverified"),
            _observation(PROCEDURE_TEXT + " [base]", index=2, verification="unverified"),
        ], interaction="seed-base")
        interaction_pin = _pin(store, branch)
        interaction_observation = _observation(NEW_TEXT, index=3, verification="unverified")
        interaction_outcome = OutcomeReceipt(
            outcome_id="outcome-1", action_id="action-1", status="completed",
            receipt_id="receipt-1", observed_at=_now_ns(),
            summary="action-1 completed receipt=receipt-1")
        interaction_note = ProvisionalNote(
            note_id="explanation-1",
            text="the assistant claimed the vault code was 4127",
            origin="model_generated", interaction_id="interaction-1")
        interaction_proposal = InteractionProposal(
            interaction_id="interaction-1", subject_id=interaction_pin.subject_id,
            branch_id=interaction_pin.branch_id, purpose=interaction_pin.purpose,
            mode=interaction_pin.mode, task_id=interaction_pin.task_id,
            task_revision=interaction_pin.task_revision,
            decision_revision=interaction_pin.decision_revision + 1,
            source_epoch=0, producer_versions=interaction_pin.producer_versions,
            representation_version=interaction_pin.representation_version,
            codec_id=interaction_pin.codec_id,
            permitted_uses=interaction_pin.permitted_uses,
            observations=(interaction_observation,),
            outcomes=(interaction_outcome,), notes=(interaction_note,))
        before = answer_query(QUERY, store.visible(interaction_pin))
        new_commit = store.commit_interaction(interaction_proposal, pin=interaction_pin)
        stale_pin_answer = answer_query(QUERY, store.visible(interaction_pin))
        later_pin = _pin(store, branch)
        after = answer_query(QUERY, store.visible(later_pin))
        duplicate = store.commit_interaction(interaction_proposal, pin=interaction_pin)
        # Conflicting retry: same logical event, different observation bytes.
        conflict = None
        try:
            store.commit_interaction(
                InteractionProposal(
                    interaction_id="interaction-1", subject_id=interaction_pin.subject_id,
                    branch_id=interaction_pin.branch_id, purpose=interaction_pin.purpose,
                    mode=interaction_pin.mode, task_id=interaction_pin.task_id,
                    task_revision=interaction_pin.task_revision,
                    decision_revision=interaction_pin.decision_revision + 1,
                    source_epoch=0, producer_versions=interaction_pin.producer_versions,
                    representation_version=interaction_pin.representation_version,
                    codec_id=interaction_pin.codec_id,
                    permitted_uses=interaction_pin.permitted_uses,
                    observations=(_observation(NEW_TEXT + " [changed]", index=3,
                                               verification="unverified"),),
                    outcomes=(interaction_outcome,), notes=(interaction_note,)),
                pin=interaction_pin)
        except FormationConflict as error:
            conflict = str(error)
        provisional = store.read_provisional(branch_id=branch, purpose=PURPOSE)
        visible_ids = tuple(record.record_id for record in store.visible(later_pin))
        new_id = new_commit.receipt.record_ids[0]
        provisional_ids = tuple(note["note_id"] for note in store.read_provisional(
            branch_id=branch, purpose=PURPOSE))
        report["conditions"]["enabled"] = {
            "new_record_id": new_id,
            "provisional_note_ids": provisional_ids,
            "before": before, "after": after, "stale_pin_answer": stale_pin_answer,
            "new_record_visible_later": new_id in visible_ids,
            "new_record_visible_to_stale_pin": new_id in tuple(
                record.record_id for record in store.visible(interaction_pin)),
            "duplicate_status": duplicate.status,
            "duplicate_same_receipt": duplicate.receipt.receipt_digest
            == new_commit.receipt.receipt_digest,
            "conflict_rejected": conflict is not None,
            "provisional_notes": len(provisional),
            "provisional_note_ids_all": provisional_ids,
            "base_record_count": len(base_commit.receipt.record_ids),
        }
        # Correction: the successor is eligible, the parent is not.
        correction_pin = _pin(store, branch)
        corrected = store.correct(
            new_id, text="vault code 9001 was observed after review",
            reason="verified correction", pin=correction_pin,
            verifier_receipt_id="study-correction-verifier")
        correction_pin2 = _pin(store, branch)
        corrected_answer = answer_query(QUERY, store.visible(correction_pin2))
        report["conditions"]["corrected"] = {
            "correction_status": corrected.status,
            "answer": corrected_answer,
            "old_record_excluded": new_id not in tuple(
                record.record_id for record in store.visible(correction_pin2)),
        }
        # Revocation: no answer from the revoked record or its descendants.
        revoke_pin = _pin(store, branch)
        revocation = store.revoke(corrected.receipt.record_ids[0],
                                  reason="source revoked")
        revoked_pin = _pin(store, branch)
        revoked_answer = answer_query(QUERY, store.visible(revoked_pin))
        report["conditions"]["revoked"] = {
            "revocation_status": revocation.status,
            "answer": revoked_answer,
            "revoked_record_absent": corrected.receipt.record_ids[0] not in tuple(
                record.record_id for record in store.visible(revoked_pin)),
            "healthy_unrelated_visible": all(
                record_id in tuple(item.record_id for item in store.visible(revoked_pin))
                for record_id in base_commit.receipt.record_ids),
        }
        report["enabled_metrics"] = store.metrics_document()
        report["recovery_after_lifecycle"] = store.recover().document()

    # Disabled: identical base evidence, no completed-interaction write.
    disabled_root = root / "disabled"
    with _store(disabled_root) as store:
        branch = "replica-branch-a"
        base_pin = _pin(store, branch)
        _commit(store, base_pin, [
            _observation(BASE_TEXT + " [base]", index=1, verification="unverified"),
            _observation(PROCEDURE_TEXT + " [base]", index=2, verification="unverified"),
        ], interaction="seed-base")
        interaction_pin = _pin(store, branch)
        before = answer_query(QUERY, store.visible(interaction_pin))
        later_pin = _pin(store, branch)
        after = answer_query(QUERY, store.visible(later_pin))
        report["conditions"]["disabled"] = {
            "before": before, "after": after,
            "new_record_visible_later": False,
            "base_record_count": 2,
        }

    # Wrong branch: a write on branch B is invisible to branch A.
    branch_root = root / "wrong_branch"
    with _store(branch_root) as store:
        branch_a = "replica-branch-a"
        branch_b = "replica-branch-b"
        pin_a = _pin(store, branch_a)
        _commit(store, pin_a, [
            _observation(BASE_TEXT + " [base]", index=1, verification="unverified")],
            interaction="seed-a")
        pin_b = _pin(store, branch_b)
        branch_b_commit = _commit(store, pin_b, [
            _observation(NEW_TEXT, index=2, verification="unverified")],
            interaction="other-branch")
        later_a = _pin(store, branch_a)
        branch_a_visible = store.visible(later_a)
        branch_b_visible = store.visible(_pin(store, branch_b))
        report["conditions"]["wrong_branch"] = {
            "answer": answer_query(QUERY, branch_a_visible),
            "branch_a_records": len(branch_a_visible),
            "branch_b_records": len(branch_b_visible),
            "branch_a_record_ids": tuple(item.record_id for item in branch_a_visible),
            "branch_b_record_ids": tuple(item.record_id for item in branch_b_visible),
            "branch_b_record_id": branch_b_commit.receipt.record_ids[0],
        }

    enabled = report["conditions"]["enabled"]
    disabled = report["conditions"]["disabled"]
    wrong = report["conditions"]["wrong_branch"]
    new_id = enabled["new_record_id"]
    enabled_after_id = enabled["after"]["record_id"]
    disabled_after_id = disabled["after"]["record_id"]
    note_ids = set(enabled["provisional_note_ids"])
    enabled_used = enabled_after_id == new_id
    disabled_used = disabled_after_id == new_id if disabled_after_id else False
    before_absent = enabled["before"]["record_id"] is None
    provisional_excluded = (
        new_id not in note_ids
        and all(note_id not in enabled["after"]["evidence"] for note_id in note_ids)
        and enabled_after_id not in note_ids)
    report["measurement"] = {
        "enabled_used_new_observation": enabled_used,
        "disabled_used_new_observation": disabled_used,
        "delayed_visibility_holds": (
            enabled["stale_pin_answer"]["answer"] == UNKNOWN_ANSWER
            and enabled["new_record_visible_later"]
            and enabled_used),
        "removing_write_removes_improvement": (
            enabled_used and not disabled_used
            and disabled_after_id is None),
        "wrong_branch_isolated": (
            wrong["answer"]["answer"] == UNKNOWN_ANSWER
            and wrong["branch_b_record_id"] not in wrong["branch_a_record_ids"]
            and wrong["branch_b_record_id"] in wrong["branch_b_record_ids"]),
        "provisional_not_evidence": provisional_excluded,
        "improvement_attributable_to_write": (
            before_absent and enabled_used and not disabled_used
            and enabled["new_record_visible_later"]
            and not enabled["new_record_visible_to_stale_pin"]),
    }
    return report
