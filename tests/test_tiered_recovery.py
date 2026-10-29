"""Regression coverage for durable authority across index loss.

The reported baseline failure (reproduced by the parent at
``reports/spark_persistent_learner_v2/recovery_baseline.json``) deleted
``<tmp>/store/memory.sqlite3`` and then observed:

* S, I, resumable C, K and phi counts of zero after a recovery that still
  reported ``ok``;
* dependent-P quarantine dropping from 1 to 0 after revoked teaching evidence;
* generation 2 expert weights/clock recovering with generation 1 lineage and
  encoder metadata.

These tests use the same nested ``store/memory.sqlite3`` layout and are
synthetic engineering fixtures only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from replica_lab.tiered_memory import (
    ScopeViolation,
    get_profile,
)
from test_tiered_memory import make_draft, open_store, payload, scope


def drop_index(store) -> list[Path]:
    """Delete the real SQLite index exactly as the baseline fixture did."""
    removed: list[Path] = []
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(store.sqlite_path) + suffix)
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def expert_bytes(seed: bytes) -> tuple[bytes, bytes]:
    profile = get_profile("P1")
    return (payload(profile.full_parameters * 4, seed + b"-weights"),
            payload(profile.full_parameters * 8, seed + b"-moments"))


def test_index_loss_restores_every_persistent_class_and_dependent_quarantine(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    user_scope = scope()
    admission = store.admit_cluster(
        "evidence", scope=user_scope,
        records=[make_draft("r1", "old exact claim", seed=1)])
    weights_one, moments_one = expert_bytes(b"gen-one")
    weights_two, moments_two = expert_bytes(b"gen-two")
    assert store.register_expert(
        "e1", lineage=("r1",), encoder_id="encoder-v1").location == "cold"
    assert store.write_expert_state(
        "e1", weights=weights_one, moments=moments_one, clock=10,
        lineage=("r1",), encoder_id="encoder-v1").ok
    first = store.publish_expert("e1")
    assert first.ok is True and first.detail["generation"] == 1
    assert store.write_expert_state(
        "e1", weights=weights_two, moments=moments_two, clock=55,
        lineage=("r1", "r2"), encoder_id="encoder-v2").ok
    second = store.publish_expert("e1")
    assert second.ok is True and second.detail["generation"] == 2
    assert store.record_procedure(
        "proc.alpha", 1, title="verified path",
        applicability={"task": "alpha"}, prerequisites=["source"],
        steps=["step one"], outcome="verified",
        evidence_record_ids=("r1",), verified_by="verifier").ok
    assert store.approve_identity(
        "identity.main", 1, core_text="core one",
        authority="user_approved").outcome.ok
    assert store.approve_identity(
        "identity.main", 2, core_text="core two",
        authority="user_approved", supersedes=1).outcome.ok
    workspace = store.save_workspace_snapshot(
        "task-resume", {"goal": "resume later"}, resumable=True)
    assert workspace.outcome.ok and workspace.resumable is True
    assert store.register_phi_release(
        "codec.phi", 1, producer_version="producer-v1",
        artifact_sha256="a" * 64).ok
    assert store.record_capability(
        "cap.answer.v1", model_id="synthetic-engineering", mode="cpu_fixture",
        task_family="answer", budget={"records": 8}, measured_value=0.5,
        uncertainty={"samples": 0}, measured=True).ok
    correction = store.correct_record(
        "r1", correction_id="r1-corrected", text="new corrected claim",
        source_id="user-correction", source_epoch=2, observed_at=300,
        importance=90, scope=user_scope)
    assert correction.quarantined_experts == ("e1",)
    assert set(correction.invalidated_procedures) == {("proc.alpha", 1)}
    assert correction.exact_unlearning_claimed is False
    assert admission.cluster_id == correction.cluster_id
    store.close()

    removed = drop_index(store)
    assert store.sqlite_path.parent.name == "store"
    assert str(store.sqlite_path) in {str(path) for path in removed}

    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "ok"
        assert report.incomplete_authority == ()
        assert report.authority_records_recovered >= 7
        assert report.journal_events_recovered > 0
        summary = rebuilt.state_summary()["counts"]
        assert summary["S_procedure_versions"] == 1
        assert summary["I_identity_versions"] == 2
        assert summary["I_approved_current"] == 1
        assert summary["C_workspace_snapshots"] == 1
        assert summary["K_capability_records"] == 1
        assert summary["K_measured"] == 1
        assert summary["phi_releases"] == 1
        assert summary["P_experts"] == 1
        assert summary["P_quarantined"] == 1
        assert summary["E_records"] == 2

        identity = rebuilt.identity_core()
        assert identity.outcome.ok is True
        assert identity.identity_id == "identity.main"
        assert identity.version == 2 and identity.text == "core two"
        procedures = rebuilt.procedures()
        assert len(procedures) == 1
        assert procedures[0]["outcome"] == "verified"
        assert procedures[0]["usable"] is False
        assert procedures[0]["invalidated_by_evidence"] is True
        snapshots = rebuilt.workspace_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["task_id"] == "task-resume"
        assert snapshots[0]["resumable"] is True
        assert (rebuilt.root / snapshots[0]["path"]).is_file()
        assert rebuilt.phi_releases()[0]["producer_version"] == "producer-v1"
        capabilities = rebuilt.capabilities(task_family="answer")
        assert capabilities[0]["measured"] is True
        assert capabilities[0]["measured_value"] == pytest.approx(0.5)

        experts = rebuilt.expert_records()
        assert len(experts) == 1
        assert experts[0].publication_generation == 2
        assert experts[0].encoder_id == "encoder-v2"
        assert experts[0].quarantine_status == "quarantined"
        loaded = rebuilt.load_expert("e1")
        assert loaded.outcome.status == "degraded"
        assert loaded.state is not None
        assert loaded.state.clock == 55
        assert loaded.state.lineage == ("r1", "r2")
        assert loaded.state.encoder_id == "encoder-v2"
        assert loaded.state.quarantine_status == "quarantined"
        assert loaded.state.source == "cold_publication"

        kinds = {event["kind"] for event in rebuilt.journal_events(limit=200)}
        assert {"record_corrected", "expert_quarantined",
                "procedure_invalidated", "identity_approved",
                "workspace_snapshot", "expert_published"} <= kinds
    finally:
        rebuilt.close()


def test_newer_expert_generation_keeps_its_own_clock_lineage_and_encoder(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    weights_one, moments_one = expert_bytes(b"older")
    weights_two, moments_two = expert_bytes(b"newer")
    store.register_expert("e1", lineage=("r1",), encoder_id="encoder-v1")
    assert store.write_expert_state(
        "e1", weights=weights_one, moments=moments_one, clock=11,
        lineage=("r1",), encoder_id="encoder-v1").ok
    assert store.publish_expert("e1").detail["generation"] == 1
    assert store.evict_expert("e1").ok
    assert store.write_expert_state(
        "e1", weights=weights_two, moments=moments_two, clock=99,
        lineage=("r1", "r2"), encoder_id="encoder-v2").ok
    assert store.publish_expert("e1").detail["generation"] == 2
    assert store.evict_expert("e1").ok
    store.close()

    drop_index(store)
    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "ok"
        records = rebuilt.expert_records()
        assert len(records) == 1
        assert records[0].publication_generation == 2
        assert records[0].encoder_id == "encoder-v2"
        loaded = rebuilt.load_expert("e1")
        assert loaded.outcome.ok is True
        assert loaded.state is not None
        assert loaded.state.clock == 99
        assert loaded.state.lineage == ("r1", "r2")
        assert loaded.state.encoder_id == "encoder-v2"
        assert loaded.state.source == "cold_publication"
    finally:
        rebuilt.close()


def test_corrupt_authority_is_explicitly_incomplete_never_silent_ok(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    store.record_procedure(
        "proc.alpha", 1, title="verified path",
        applicability={"task": "alpha"}, prerequisites=["source"],
        steps=["step one"], outcome="verified",
        evidence_record_ids=("r1",), verified_by="verifier")
    store.approve_identity(
        "identity.main", 1, core_text="core one", authority="user_approved")
    store.save_workspace_snapshot("task-resume", {"goal": "resume"}, resumable=True)
    store.register_phi_release(
        "codec.phi", 1, producer_version="producer-v1",
        artifact_sha256="b" * 64)
    store.record_capability(
        "cap.answer.v1", model_id="synthetic-engineering", mode="cpu_fixture",
        task_family="answer", budget={"records": 8}, measured_value=0.5,
        uncertainty={"samples": 0}, measured=True)
    store.close()

    drop_index(store)
    identity_files = sorted((store.root / "authority/identity").rglob("*.json"))
    assert identity_files
    identity_files[0].write_bytes(b"{not a valid authority record")

    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "incomplete"
        assert report.incomplete_authority
        assert any("authority/identity" in item
                   for item in report.incomplete_authority)
        assert any("authority_preserved" in note for note in report.notes)
        assert rebuilt.identity_core().outcome.status == "incomplete"
        # Unrelated authority is still recovered, while the damaged class is
        # refused rather than reported as successfully recovered.
        assert len(rebuilt.procedures()) == 1
        assert len(rebuilt.workspace_snapshots()) == 1
        assert len(rebuilt.phi_releases()) == 1
        assert len(rebuilt.capabilities()) == 1
    finally:
        rebuilt.close()


def test_missing_expert_payload_is_explicitly_incomplete(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    weights, moments = expert_bytes(b"lost")
    store.register_expert("e1", lineage=("r1",))
    assert store.write_expert_state(
        "e1", weights=weights, moments=moments, clock=7).ok
    assert store.publish_expert("e1").ok
    assert store.evict_expert("e1").ok
    store.close()

    drop_index(store)
    cold_weights = (store.root
                    / "storage/cold/experts/e1/gen-00000001/weights.bin")
    assert cold_weights.is_file()
    cold_weights.unlink()

    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "incomplete"
        assert any("expert_payload_missing" in note for note in report.notes)
        assert report.incomplete_authority
        records = rebuilt.expert_records()
        assert len(records) == 1
        loaded = rebuilt.load_expert("e1")
        assert loaded.outcome.status == "failed"
        assert loaded.outcome.reason == "expert_health_not_ok"
        assert loaded.state is None
        assert rebuilt.state_summary()["counts"]["P_experts"] == 1
    finally:
        rebuilt.close()


def test_missing_record_payload_is_incomplete_not_silently_ok(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    store.admit_cluster(
        "evidence", scope=scope(),
        records=[make_draft("r1", "interrupted exact claim", seed=5)])
    store.close()

    drop_index(store)
    text_path = store.root / "storage/warm/records/r1/text.bin"
    assert text_path.is_file()
    text_path.unlink()

    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "incomplete"
        assert any("record_payload_invalid" in note for note in report.notes)
        assert rebuilt.state_summary()["counts"]["E_records"] == 0
    finally:
        rebuilt.close()


def test_rebuild_preserves_cross_scope_isolation_and_revocation(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    scope_a = scope("user_a")
    scope_b = scope("user_b")
    admission_a = store.admit_cluster(
        "evidence", scope=scope_a,
        records=[make_draft("ra", "alpha shared claim", seed=6)])
    admission_b = store.admit_cluster(
        "evidence", scope=scope_b,
        records=[make_draft("rb", "beta shared claim", seed=7)])
    revoked = store.revoke_record(
        "ra", reason="scope-a revocation", scope=scope_a)
    assert revoked.exact_unlearning_claimed is False
    store.close()

    drop_index(store)
    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "ok"
        assert rebuilt.cluster(admission_a.cluster_id)["scope_key"] != (
            rebuilt.cluster(admission_b.cluster_id)["scope_key"])
        selected_a = rebuilt.select("shared claim", scope=scope_a)
        selected_b = rebuilt.select("shared claim", scope=scope_b)
        try:
            assert "ra" not in selected_a.record_ids
            assert "rb" not in selected_a.record_ids
            assert "rb" in selected_b.record_ids
            assert "ra" not in selected_b.record_ids
        finally:
            rebuilt.release_selection(selected_a.selection_id)
            rebuilt.release_selection(selected_b.selection_id)
        rows = {row["record_id"]: row for row in rebuilt.connection.execute(
            "SELECT record_id, status, scope_key FROM records")}
        assert rows["ra"]["status"] == "revoked"
        assert rows["ra"]["scope_key"] == scope_a.scope_key
        assert rows["rb"]["status"] == "current"
        assert rows["rb"]["scope_key"] == scope_b.scope_key
        with pytest.raises(ScopeViolation):
            rebuilt.revoke_record("rb", reason="wrong scope", scope=scope_a)
    finally:
        rebuilt.close()


def test_quarantine_reset_durability_keeps_exact_unlearning_distinction(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    user_scope = scope()
    store.admit_cluster(
        "evidence", scope=user_scope,
        records=[make_draft("r1", "old rule", seed=8)])
    weights, moments = expert_bytes(b"reset")
    store.register_expert("e1", lineage=("r1",))
    assert store.write_expert_state(
        "e1", weights=weights, moments=moments, clock=12).ok
    assert store.publish_expert("e1").ok
    correction = store.correct_record(
        "r1", correction_id="r1-corrected", text="new rule",
        source_id="user-correction", source_epoch=2, observed_at=400,
        scope=user_scope)
    assert correction.quarantined_experts == ("e1",)
    assert store.load_expert("e1").state.quarantine_status == "quarantined"
    denied = store.reset_expert(
        "e1", weights=weights, moments=moments, clock=0,
        authorization="unauthenticated", reason="test")
    assert denied.status == "denied"
    reset = store.reset_expert(
        "e1", weights=weights, moments=moments, clock=0,
        authorization="authenticated_reset", reason="verified correction")
    assert reset.ok is True
    assert reset.detail["exact_unlearning_claimed"] is False
    store.close()

    drop_index(store)
    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "ok"
        loaded = rebuilt.load_expert("e1")
        assert loaded.outcome.ok is True
        assert loaded.state is not None
        assert loaded.state.quarantine_status == "none"
        assert loaded.state.clock == 0
        assert loaded.state.lineage == ()
        assert loaded.state.source == "hot_dirty"
        assert rebuilt.state_summary()["counts"]["P_quarantined"] == 0
        reset_events = [
            event for event in rebuilt.journal_events(limit=200)
            if event["kind"] == "expert_authenticated_reset"]
        assert reset_events
        assert reset_events[-1]["detail"]["exact_unlearning_claimed"] is False
        correction_events = [
            event for event in rebuilt.journal_events(limit=200)
            if event["kind"] == "record_corrected"]
        assert correction_events
        assert correction_events[-1]["detail"]["exact_unlearning_claimed"] is False
    finally:
        rebuilt.close()
