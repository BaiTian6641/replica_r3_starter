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


# ---------------------------------------------------------------------------
# Latest-expected-state regressions: newest/first/intermediate authority loss,
# interrupted publication, explicit migration and restart
# ---------------------------------------------------------------------------

def authority_record(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))["authority"]


def component_authority_files(store, authority_class: str,
                              entity_id: str) -> list[Path]:
    """Return one component's numbered authority files, oldest first."""
    root = store.root / "authority" / authority_class
    files = [path for path in sorted(root.rglob("*.json"))
             if authority_record(path).get(
                 "expert_id" if authority_class == "experts"
                 else "identity_id") == entity_id]
    return files


def test_missing_newest_quarantine_authority_preserves_quarantine_and_health(
        tmp_path: Path) -> None:
    """Deleting the newest quarantined-authority file must not clear it."""
    store = open_store(tmp_path)
    user_scope = scope()
    store.admit_cluster(
        "evidence", scope=user_scope,
        records=[make_draft("r1", "revoked rule", seed=31),
                 make_draft("r2", "healthy rule", seed=32)])
    weights_one, moments_one = expert_bytes(b"revoked-lineage")
    weights_two, moments_two = expert_bytes(b"healthy-lineage")
    store.register_expert("e1", lineage=("r1",), encoder_id="encoder-v1")
    assert store.write_expert_state(
        "e1", weights=weights_one, moments=moments_one, clock=9,
        lineage=("r1",), encoder_id="encoder-v1").ok
    assert store.publish_expert("e1").ok
    assert store.evict_expert("e1").ok
    store.register_expert("e2", lineage=("r2",), encoder_id="encoder-v1")
    assert store.write_expert_state(
        "e2", weights=weights_two, moments=moments_two, clock=12,
        lineage=("r2",), encoder_id="encoder-v1").ok
    assert store.publish_expert("e2").ok
    assert store.evict_expert("e2").ok
    revoked = store.revoke_record(
        "r1", reason="revoked teaching evidence", scope=user_scope)
    assert revoked.quarantined_experts == ("e1",)
    e1_files = component_authority_files(store, "experts", "e1")
    assert authority_record(e1_files[-1])["quarantine_status"] == "quarantined"
    store.close()

    e1_files[-1].unlink()
    drop_index(store)

    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "incomplete"
        assert any("expert_authority_uncertain" in note
                   and "quarantine" in note for note in report.notes)
        records = {record.expert_id: record
                   for record in rebuilt.expert_records()}
        assert records["e1"].quarantine_status == "quarantined"
        loaded_e1 = rebuilt.load_expert("e1")
        assert loaded_e1.outcome.status == "failed"
        assert loaded_e1.outcome.reason == "expert_health_not_ok"
        assert loaded_e1.state is None
        # Unrelated healthy owner remains fully available.
        loaded_e2 = rebuilt.load_expert("e2")
        assert loaded_e2.outcome.ok is True
        assert loaded_e2.state is not None
        assert loaded_e2.state.clock == 12
        assert rebuilt.state_summary()["counts"]["P_experts"] == 2
        assert rebuilt.state_summary()["counts"]["P_quarantined"] == 1
    finally:
        rebuilt.close()


def test_missing_newest_identity_authority_keeps_current_unset(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    assert store.approve_identity(
        "identity.main", 1, core_text="core one",
        authority="user_approved").outcome.ok
    assert store.approve_identity(
        "identity.main", 2, core_text="core two",
        authority="user_approved", supersedes=1).outcome.ok
    files = component_authority_files(store, "identity", "identity.main")
    assert len(files) == 2
    store.close()

    files[-1].unlink()
    drop_index(store)

    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "incomplete"
        assert any("identity_authority_missing_newest" in note
                   for note in report.notes)
        identity = rebuilt.identity_core("identity.main")
        assert identity.outcome.status == "incomplete"
        assert identity.version is None and identity.text is None
        rows = {int(row["version"]): row["status"] for row in rebuilt.connection.execute(
            "SELECT version, status FROM identity_versions "
            "WHERE identity_id='identity.main'")}
        assert rows == {1: "superseded"}
        summary = rebuilt.state_summary()["counts"]
        assert summary["I_identity_versions"] == 1
        assert summary["I_approved_current"] == 0
    finally:
        rebuilt.close()


def test_corrupt_newest_authority_never_clears_quarantine(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    user_scope = scope()
    store.admit_cluster(
        "evidence", scope=user_scope,
        records=[make_draft("r1", "old exact claim", seed=33)])
    weights, moments = expert_bytes(b"corrupt-newest")
    store.register_expert("e1", lineage=("r1",))
    assert store.write_expert_state(
        "e1", weights=weights, moments=moments, clock=4).ok
    assert store.publish_expert("e1").ok
    correction = store.correct_record(
        "r1", correction_id="r1-corrected", text="new claim",
        source_id="user-correction", source_epoch=2, observed_at=500,
        scope=user_scope)
    assert correction.quarantined_experts == ("e1",)
    newest = component_authority_files(store, "experts", "e1")[-1]
    store.close()

    newest.write_bytes(b"{not a valid authority envelope")
    drop_index(store)

    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "incomplete"
        assert any("authority_preserved" in note for note in report.notes)
        assert newest.relative_to(store.root).as_posix() in (
            report.incomplete_authority)
        records = rebuilt.expert_records()
        assert len(records) == 1
        assert records[0].quarantine_status == "quarantined"
        assert rebuilt.load_expert("e1").outcome.status == "failed"
        assert rebuilt.state_summary()["counts"]["P_quarantined"] == 1
    finally:
        rebuilt.close()


def test_first_and_intermediate_authority_loss_is_reported_not_hidden(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    weights_one, moments_one = expert_bytes(b"first-gen")
    weights_two, moments_two = expert_bytes(b"second-gen")
    store.register_expert("e1", lineage=("r1",), encoder_id="encoder-v1")
    assert store.write_expert_state(
        "e1", weights=weights_one, moments=moments_one, clock=10,
        lineage=("r1",), encoder_id="encoder-v1").ok
    assert store.publish_expert("e1").ok
    assert store.evict_expert("e1").ok
    assert store.write_expert_state(
        "e1", weights=weights_two, moments=moments_two, clock=77,
        lineage=("r1", "r2"), encoder_id="encoder-v2").ok
    assert store.publish_expert("e1").ok
    assert store.evict_expert("e1").ok
    files = component_authority_files(store, "experts", "e1")
    assert len(files) >= 6
    deleted = [files[0], files[2]]
    store.close()

    for path in deleted:
        path.unlink()
    drop_index(store)

    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "incomplete"
        for path in deleted:
            assert path.relative_to(store.root).as_posix() in (
                report.incomplete_authority)
        assert any("authority_sequence_gap" in note for note in report.notes)
        # The newest surviving record is a complete snapshot; its payload is
        # still coherent, while the lost chain is explicit.
        loaded = rebuilt.load_expert("e1")
        assert loaded.outcome.ok is True
        assert loaded.state is not None
        assert loaded.state.generation == 2
        assert loaded.state.clock == 77
        assert loaded.state.lineage == ("r1", "r2")
        assert loaded.state.encoder_id == "encoder-v2"
    finally:
        rebuilt.close()


def test_interrupted_publication_is_not_promoted_without_authority(
        tmp_path: Path) -> None:
    import json

    store = open_store(tmp_path)
    weights_one, moments_one = expert_bytes(b"acked-gen")
    weights_two, moments_two = expert_bytes(b"unacked-gen")
    store.register_expert("e1", lineage=("r1",), encoder_id="encoder-v1")
    assert store.write_expert_state(
        "e1", weights=weights_one, moments=moments_one, clock=10,
        lineage=("r1",), encoder_id="encoder-v1").ok
    assert store.publish_expert("e1").ok
    assert store.evict_expert("e1").ok
    assert store.write_expert_state(
        "e1", weights=weights_two, moments=moments_two, clock=66,
        lineage=("r1",), encoder_id="encoder-v2").ok
    assert store.publish_expert("e1").ok
    assert store.evict_expert("e1").ok
    files = component_authority_files(store, "experts", "e1")
    acknowledged = [path for path in files
                    if int(authority_record(path)["publication_generation"]) <= 1]
    assert acknowledged and files[-1] not in acknowledged
    journal_path = store.root / "journal/events.jsonl"
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    last_acked_event = max(
        index for index, line in enumerate(lines)
        if json.loads(line)["kind"] == "expert_evicted")
    store.close()

    # Authority never acknowledged generation 2; its journal events also never
    # committed.  Only the cold publication files/manifest exist.
    for path in files:
        if path not in acknowledged:
            path.unlink()
    journal_path.write_text(
        "\n".join(lines[:last_acked_event + 1]) + "\n", encoding="utf-8")
    drop_index(store)

    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "incomplete"
        assert any("publication_ahead_of_authority" in note
                   for note in report.notes)
        records = rebuilt.expert_records()
        assert len(records) == 1
        assert records[0].publication_generation == 1
        loaded = rebuilt.load_expert("e1")
        assert loaded.outcome.status == "failed"
        assert loaded.state is None
    finally:
        rebuilt.close()


def test_legacy_manifest_import_requires_explicit_migration(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    weights, moments = expert_bytes(b"legacy-manifest")
    store.register_expert("e1", lineage=("r1",))
    assert store.write_expert_state(
        "e1", weights=weights, moments=moments, clock=21).ok
    assert store.publish_expert("e1").ok
    assert store.evict_expert("e1").ok
    files = component_authority_files(store, "experts", "e1")
    store.close()

    for path in files:
        path.unlink()

    # The authority class directory still exists, so durable authority was
    # expected: a manifest must not be imported silently.
    reopened = open_store(tmp_path)
    try:
        assert (reopened.root / "authority/experts").is_dir()
        refused = reopened.recover()
        assert refused.outcome.status == "incomplete"
        assert any("refusing_manifest_downgrade" in note
                   for note in refused.notes)
        assert reopened.load_expert("e1").outcome.status == "failed"
        migrated = reopened.recover(
            allow_legacy_expert_manifest_import=True)
        assert migrated.outcome.status == "ok"
        assert any("legacy_expert_manifest_migration" in note
                   for note in migrated.notes)
        loaded = reopened.load_expert("e1")
        assert loaded.outcome.ok is True
        assert loaded.state is not None
        assert loaded.state.weights == weights
        assert loaded.state.clock == 21
    finally:
        reopened.close()

    # A store with no authority/experts directory at all is structurally a
    # pre-authority store; manifest import is then the documented migration.
    other = tmp_path / "legacy-absent"
    legacy = open_store(tmp_path, name="legacy-absent")
    legacy.register_expert("e2", lineage=("r2",))
    assert legacy.write_expert_state(
        "e2", weights=weights, moments=moments, clock=5).ok
    assert legacy.publish_expert("e2").ok
    assert legacy.evict_expert("e2").ok
    legacy.close()
    authority_root = other / "authority" / "experts"
    for child in sorted(authority_root.rglob("*.json")):
        child.unlink()
    for directory in sorted(
            (path for path in authority_root.rglob("*") if path.is_dir()),
            reverse=True):
        directory.rmdir()
    authority_root.rmdir()
    drop_index(legacy)
    rebuilt = open_store(tmp_path, name="legacy-absent")
    try:
        report = rebuilt.recovery_report
        assert report is not None
        # No structural guess is made even when the whole class directory is
        # absent: only the explicit migration flag may import manifests.
        assert report.outcome.status == "incomplete"
        assert any("refusing_manifest_downgrade" in note
                   for note in report.notes)
        assert rebuilt.load_expert("e2").outcome.status == "failed"
        migrated = rebuilt.recover(
            allow_legacy_expert_manifest_import=True)
        assert migrated.outcome.status == "ok"
        assert any("legacy_expert_manifest_migration" in note
                   for note in migrated.notes)
        assert rebuilt.load_expert("e2").outcome.ok is True
    finally:
        rebuilt.close()


def test_restart_and_index_loss_with_complete_authority_keep_latest_state(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    weights_one, moments_one = expert_bytes(b"restart-one")
    weights_two, moments_two = expert_bytes(b"restart-two")
    store.register_expert("e1", lineage=("r1",), encoder_id="encoder-v1")
    assert store.write_expert_state(
        "e1", weights=weights_one, moments=moments_one, clock=1,
        lineage=("r1",), encoder_id="encoder-v1").ok
    assert store.publish_expert("e1").ok
    assert store.evict_expert("e1").ok
    assert store.write_expert_state(
        "e1", weights=weights_two, moments=moments_two, clock=2,
        lineage=("r1", "r2"), encoder_id="encoder-v2").ok
    assert store.publish_expert("e1").ok
    assert store.evict_expert("e1").ok
    assert store.approve_identity(
        "identity.main", 1, core_text="core one").outcome.ok
    assert store.approve_identity(
        "identity.main", 2, core_text="core two", supersedes=1).outcome.ok
    store.close()

    restarted = open_store(tmp_path)
    try:
        assert restarted.recovery_report is None
        assert restarted.identity_core().version == 2
        loaded = restarted.load_expert("e1")
        assert loaded.outcome.ok is True and loaded.state is not None
        assert loaded.state.generation == 2
        assert loaded.state.encoder_id == "encoder-v2"
        report = restarted.recover()
        assert report.outcome.status == "ok"
        assert report.incomplete_authority == ()
    finally:
        restarted.close()

    drop_index(store)
    rebuilt = open_store(tmp_path)
    try:
        assert rebuilt.identity_core().version == 2
        loaded = rebuilt.load_expert("e1")
        assert loaded.outcome.ok is True and loaded.state is not None
        assert loaded.state.generation == 2
        assert loaded.state.clock == 2
        assert rebuilt.recovery_report.outcome.status == "ok"
    finally:
        rebuilt.close()


def test_identity_versions_need_not_be_contiguous(tmp_path: Path) -> None:
    """Caller-supplied identity versions are not an internally numbered chain."""
    store = open_store(tmp_path)
    assert store.approve_identity(
        "identity.main", 5, core_text="core five").outcome.ok
    assert store.approve_identity(
        "identity.main", 7, core_text="core seven", supersedes=5).outcome.ok
    store.close()

    drop_index(store)
    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "ok"
        assert report.incomplete_authority == ()
        identity = rebuilt.identity_core()
        assert identity.outcome.ok is True
        assert identity.version == 7
        assert identity.text == "core seven"
        rows = {int(row["version"]): row["status"]
                for row in rebuilt.connection.execute(
                    "SELECT version, status FROM identity_versions "
                    "WHERE identity_id='identity.main'")}
        assert rows == {5: "superseded", 7: "current"}
    finally:
        rebuilt.close()


def test_missing_identity_authority_class_is_never_revived(
        tmp_path: Path) -> None:
    store = open_store(tmp_path)
    assert store.approve_identity(
        "identity.main", 1, core_text="core one").outcome.ok
    store.close()

    identity_root = store.root / "authority" / "identity"
    for path in sorted(identity_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    drop_index(store)

    rebuilt = open_store(tmp_path)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.outcome.status == "incomplete"
        assert any("identity_authority_missing_class" in note
                   for note in report.notes)
        assert rebuilt.identity_core().outcome.status == "incomplete"
        assert rebuilt.state_summary()["counts"]["I_approved_current"] == 0
    finally:
        rebuilt.close()
