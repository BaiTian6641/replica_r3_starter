"""Engineering tests for the durable logical tier runtime.

These tests use small real SQLite databases and files under pytest's temporary
directory.  They do not load Spark, allocate an expanded expert bank, train a
model, or qualify native capability.  Values and expert payloads are synthetic
engineering bytes; the module labels them as such.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from replica_lab.tiered_memory import (
    ACTIVE_EXPERTS_MAX,
    ARTIFACT_BUDGET_BYTES,
    COLD_CEILING_BYTES,
    EVIDENCE_RECORDS_MAX,
    EVIDENCE_VALUE_SLOTS_MAX,
    HOT_CEILING_BYTES,
    IDENTITY_RESERVED_BYTES,
    KEY_WIDTH,
    MINIMUM_VOLUME_FREE_BYTES,
    NATIVE_GUARDS,
    PROFILES,
    REPLAY_RESERVED_BYTES,
    STATE_CLASSES,
    VALUE_SLOTS_PER_RECORD,
    VALUE_WIDTH,
    WARM_CEILING_BYTES,
    BudgetError,
    CapacityExceeded,
    EvidenceRecordDraft,
    MemoryScope,
    RetrievalBudget,
    TierCeilings,
    TieredMemoryStore,
    UnsupportedOperation,
    build_retrieval_key,
    check_native_guards,
    design_contract,
    detect_file_type,
    engineering_limits,
    exact_unlearning_supported,
    get_profile,
    native_execution_qualification,
    pack_values,
    plan_profile,
    unpack_values,
)


# ---------------------------------------------------------------------------
# Synthetic durable fixtures
# ---------------------------------------------------------------------------

def sample_values(seed: int, *, slots: int = VALUE_SLOTS_PER_RECORD,
                  width: int = VALUE_WIDTH) -> tuple[tuple[float, ...], ...]:
    rows = []
    for slot in range(slots):
        row = []
        for index in range(width):
            numerator = ((seed * 29 + slot * 31 + index * 17) % 101) - 50
            row.append(numerator / 50.0)
        rows.append(tuple(row))
    return tuple(rows)


def payload(byte_count: int, seed: bytes) -> bytes:
    return hashlib.shake_128(seed).digest(byte_count)


def scope(subject: str = "user_a", *, purpose: str = "assistant",
          as_of: int = 1000) -> MemoryScope:
    return MemoryScope(subject_id=subject, branch_id="main",
                       purpose=purpose, as_of=as_of)


def make_draft(record_id: str, text: str, *, seed: int = 1,
               include_values: bool = True, critical: bool = False,
               source_id: str = "source-1", observed_at: int = 100,
               key_producer_version: str = "v1",
               value_producer_version: str = "test-values-v1",
               **kwargs) -> EvidenceRecordDraft:
    values = sample_values(seed) if include_values else None
    return EvidenceRecordDraft(
        record_id=record_id, text=text, source_id=source_id,
        source_epoch=1, observed_at=observed_at, critical=critical,
        values=values,
        value_producer_id="test-values" if values is not None else None,
        value_producer_version=value_producer_version if values is not None else None,
        key_producer_version=key_producer_version,
        **kwargs,
    )


def policy(**overrides) -> TierCeilings:
    base = {
        "hot_bytes": 32 * 1024**2,
        "warm_bytes": 32 * 1024**2,
        "cold_bytes": 32 * 1024**2,
        "identity_reserved_bytes": 64 * 1024,
        "replay_reserved_bytes": 256 * 1024,
        "minimum_volume_free_bytes": 0,
        "workspace_quota_bytes": 1 * 1024**2,
        "artifact_budget_bytes": 1 * 1024**2,
        "frequent_reserve_fraction": 0.5,
        "frequent_min_accesses": 2,
    }
    base.update(overrides)
    return TierCeilings(**base)


def open_store(tmp_path: Path, *, name: str = "store",
               ceilings: TierCeilings | None = None,
               profile_id: str = "P1",
               two_active_ablation: bool = False) -> TieredMemoryStore:
    return TieredMemoryStore(
        tmp_path / name,
        profile_id=profile_id,
        ceilings=ceilings or policy(),
        two_active_ablation=two_active_ablation,
        enforce_volume_free=False,
    )


# ---------------------------------------------------------------------------
# Design contract, profiles and unsupported outcomes
# ---------------------------------------------------------------------------

def test_design_contract_profiles_and_native_limits(tmp_path: Path) -> None:
    assert HOT_CEILING_BYTES == 2 * 1024**3
    assert WARM_CEILING_BYTES == 8 * 1024**3
    assert COLD_CEILING_BYTES == 64 * 1024**3
    assert IDENTITY_RESERVED_BYTES == 8 * 1024**2
    assert REPLAY_RESERVED_BYTES == 256 * 1024**2
    assert MINIMUM_VOLUME_FREE_BYTES == 25 * 1024**3
    assert EVIDENCE_RECORDS_MAX == 8
    assert EVIDENCE_VALUE_SLOTS_MAX == 32
    assert VALUE_SLOTS_PER_RECORD == 4
    assert VALUE_WIDTH == 256 and KEY_WIDTH == 512
    assert len(STATE_CLASSES) == 7

    contract = design_contract()
    assert contract["allocation_or_training"] is False
    assert contract["native_capability"] is False
    assert "P128_stretch" in contract["profiles_verified"]
    assert PROFILES["P64"].full_parameters == 50_331_648
    assert PROFILES["P128_stretch"].full_parameters == 100_663_296
    assert PROFILES["P1"].one_active_full_training_bytes == 6 * 1024**2

    plan = plan_profile("P64")
    assert plan.active_experts == 1
    assert plan.inactive_experts == 63
    assert plan.gpu_active_bytes == 12 * 1024**2
    assert plan.gpu_inactive_bytes == 0
    assert plan.inactive_full_training_bytes > 0
    assert plan.all_banks_allocated is False
    assert plan.ablation is None
    assert plan.document()["optimizer"] == "NativeFusedAdamW"
    assert plan.document()["muon"] == "not_active"

    with pytest.raises(UnsupportedOperation):
        plan_profile("P64", active_experts=2)
    ablation = plan_profile("P64", active_experts=2, two_active_ablation=True)
    assert ablation.gpu_active_bytes == 2 * 12 * 1024**2
    assert ablation.ablation == "two_active_ablation_v1"
    assert ablation.status == "proposed_unexecuted_ablation"
    with pytest.raises(UnsupportedOperation):
        plan_profile("P64", active_experts=ACTIVE_EXPERTS_MAX + 1)
    with pytest.raises(ValueError):
        plan_profile("P1", active_experts=2, two_active_ablation=True)

    assert check_native_guards(driver_bytes=0, swap_growth_bytes=0,
                               host_free_fraction=0.5).status == "ok"
    stopped = check_native_guards(
        driver_bytes=NATIVE_GUARDS.sampled_driver_bytes + 1,
        swap_growth_bytes=NATIVE_GUARDS.sustained_swap_growth_bytes + 1,
        host_free_fraction=0.1)
    assert stopped.status == "failed"
    assert "sampled_driver_limit_exceeded" in stopped.detail["failures"]
    assert "sustained_swap_growth_exceeded" in stopped.detail["failures"]
    assert "minimum_host_free_fraction_violated" in stopped.detail["failures"]

    assert native_execution_qualification().status == "unsupported"
    assert exact_unlearning_supported().status == "unsupported"
    limits = engineering_limits()
    assert limits["native_capability"] is False
    assert limits["exact_unlearning_claimed"] is False
    assert limits["tier_ceilings_are_logical_not_physical_vram"] is True

    # Test-only tiny ceilings remain valid only when the reservation fits.
    with pytest.raises(ValueError):
        TierCeilings(hot_bytes=1024, identity_reserved_bytes=2048,
                     minimum_volume_free_bytes=0, artifact_budget_bytes=1024,
                     cold_bytes=1024, warm_bytes=1024)
    assert str(tmp_path)  # keep the fixture explicit
    del tmp_path


def test_budget_contract_rejects_overwide_evidence() -> None:
    with pytest.raises(BudgetError):
        RetrievalBudget(max_records=EVIDENCE_RECORDS_MAX + 1)
    with pytest.raises(BudgetError):
        RetrievalBudget(max_value_slots=EVIDENCE_VALUE_SLOTS_MAX + 1)
    with pytest.raises(BudgetError):
        RetrievalBudget(p_query_slots=33)
    with pytest.raises(BudgetError):
        RetrievalBudget(identity_bytes=IDENTITY_RESERVED_BYTES + 1)
    plan = RetrievalBudget(max_records=2, max_value_slots=8).quota_plan()
    assert plan["cross_class_borrowing"] is False
    assert plan["record_values_hold_identity_workspace_or_p"] is False


def test_pack_unpack_and_retrieval_key_separation() -> None:
    values = sample_values(3)
    packed = pack_values(values)
    assert len(packed) == 4
    assert all(len(item) == VALUE_WIDTH * 4 for item in packed)
    decoded = unpack_values(packed)
    assert decoded[0][0] == pytest.approx(values[0][0], rel=1e-6, abs=1e-7)
    with pytest.raises(ValueError):
        pack_values((values[0],))
    with pytest.raises(ValueError):
        pack_values(tuple(tuple(item[:-1]) for item in values))
    first = build_retrieval_key("exact needle text")
    second = build_retrieval_key("exact needle text")
    assert first == second and len(first) == KEY_WIDTH
    assert first != build_retrieval_key("different text")


# ---------------------------------------------------------------------------
# Clusters, manifests and scope-before-semantics
# ---------------------------------------------------------------------------

def test_stable_cluster_id_and_immutable_generations(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user_scope = scope()
        first = store.admit_cluster(
            "facts", scope=user_scope,
            records=[make_draft("r1", "alpha stable cluster")])
        assert first.cluster_id.startswith("cl_")
        document_one = store.generation_manifest(first.cluster_id, 1)
        manifest_path = store.root / document_one.path
        original_bytes = manifest_path.read_bytes()
        assert document_one.document["generation"] == 1
        assert document_one.document["record_count"] == 1

        second = store.admit_cluster(
            "facts", scope=user_scope,
            records=[make_draft("r2", "beta stable cluster")])
        assert second.cluster_id == first.cluster_id
        assert first.generation == 1 and second.generation == 2
        assert manifest_path.read_bytes() == original_bytes
        document_two = store.generation_manifest(first.cluster_id)
        assert document_two.document["generation"] == 2
        assert document_two.document["parent_generation"] == 1
        assert document_two.document["record_count"] == 2
        assert tuple(entry["record_id"] for entry in
                     document_two.document["records"]) == ("r1", "r2")
        assert document_two.document["native_capability"] is False
        assert document_two.document["exact_unlearning_claimed"] is False

        # Stable IDs survive a different observation/as-of time in the same
        # scope identity.
        later_scope = MemoryScope("user_a", "main", "assistant", 2000)
        third = store.admit_cluster(
            "facts", scope=later_scope,
            records=[make_draft("r3", "gamma stable cluster")])
        assert third.cluster_id == first.cluster_id
        assert store.cluster(first.cluster_id)["current_generation"] == 3

        external = sqlite3.connect(store.sqlite_path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                external.execute(
                    "UPDATE cluster_generations SET record_count=99 "
                    "WHERE cluster_id=? AND generation=1", (first.cluster_id,))
                external.commit()
        finally:
            external.close()
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "DELETE FROM cluster_generations WHERE cluster_id=? AND generation=1",
                (first.cluster_id,))


def test_scope_before_semantics_and_bounded_selection(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        subject_a = scope("subject_a")
        subject_b = scope("subject_b")
        a = store.admit_cluster(
            "a", scope=subject_a,
            records=[make_draft("a1", "zebra launch protocol alpha")])
        b = store.admit_cluster(
            "b", scope=subject_b,
            records=[make_draft("b1", "zebra launch protocol alpha")])
        many = store.admit_cluster(
            "many", scope=subject_a,
            records=[make_draft(f"m{index}", f"zebra launch protocol item {index}")
                     for index in range(10)])
        assert a.cluster_id != b.cluster_id != many.cluster_id

        selected_a = store.select("zebra launch protocol", scope=subject_a)
        assert selected_a.record_ids
        assert all(record_id.startswith(("a1", "m")) for record_id in selected_a.record_ids)
        assert "b1" not in selected_a.record_ids
        assert all(record.cluster_id != b.cluster_id for record in selected_a.records)
        store.release_selection(selected_a.selection_id)

        selected_b = store.select("zebra launch protocol", scope=subject_b)
        assert selected_b.record_ids == ("b1",)
        assert selected_b.outcome.status == "ok"
        store.release_selection(selected_b.selection_id)

        narrowed = store.select(
            "zebra launch protocol", scope=subject_a,
            budget=RetrievalBudget(max_records=2, max_value_slots=8))
        assert len(narrowed.records) == 2
        assert narrowed.truncated is True
        assert narrowed.used_value_slots <= 8
        assert len({record.cluster_id for record in narrowed.records}) <= 2
        assert narrowed.budget_plan["evidence_records"] == EVIDENCE_RECORDS_MAX
        store.release_selection(narrowed.selection_id)


# ---------------------------------------------------------------------------
# Keys, ordered values, canonical text and exact spans
# ---------------------------------------------------------------------------

def test_retrieval_keys_are_separate_from_ordered_values(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        text = "The exact needle phrase is ALPHA-7 in the source."
        admission = store.admit_cluster(
            "values", scope=scope(),
            records=[make_draft("v1", text, seed=7)])
        key_row = store.connection.execute(
            "SELECT * FROM record_keys WHERE record_id='v1'").fetchone()
        value_rows = store.connection.execute(
            "SELECT * FROM record_values WHERE record_id='v1' ORDER BY slot_index"
        ).fetchall()
        record_row = store.connection.execute(
            "SELECT * FROM records WHERE record_id='v1'").fetchone()
        assert len(value_rows) == 4
        assert key_row["key_width"] == KEY_WIDTH
        assert key_row["key_path"].startswith("storage/warm/keys/")
        assert all(row["value_width"] == VALUE_WIDTH for row in value_rows)
        assert all(row["value_path"].startswith("storage/warm/records/v1/slot-")
                   for row in value_rows)
        assert key_row["key_path"] != value_rows[0]["value_path"]
        assert (store.root / key_row["key_path"]).stat().st_size == KEY_WIDTH
        assert (store.root / value_rows[0]["value_path"]).stat().st_size == (
            VALUE_WIDTH * 4)
        assert (store.root / record_row["text_path"]).read_text() == text

        result = store.select(
            "exact needle phrase", scope=scope(),
            budget=RetrievalBudget(max_records=1, max_value_slots=4),
            exact_phrases=("needle phrase", "ALPHA-7"))
        assert result.record_ids == ("v1",)
        record = result.records[0]
        assert record.values_available is True
        assert len(record.values) == 4
        assert len(record.values[0]) == VALUE_WIDTH
        expected = sample_values(7)
        assert record.values[0][0] == pytest.approx(expected[0][0], rel=1e-6, abs=1e-7)
        assert record.value_slots == 4
        assert record.key_sha256 == key_row["key_sha256"]
        assert record.text == text
        assert result.used_value_slots == 4
        assert result.exact_spans
        span = result.exact_spans[0]
        assert span.text == "needle phrase"
        assert (span.start, span.end) == (
            text.index("needle phrase"),
            text.index("needle phrase") + len("needle phrase"))
        assert result.used_span_bytes > 0
        assert admission.generation == 1
        store.release_selection(result.selection_id)

    # Literal file receipts survive close/reopen without a model.
    reopened = open_store(tmp_path)
    try:
        assert reopened.identity_core().identity_id is None
        assert reopened.usage()["tiers"]["warm"]["by_category"]["keys"] == KEY_WIDTH
    finally:
        reopened.close()


def test_validity_interval_and_observation_cutoff(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        admission = store.admit_cluster(
            "validity", scope=scope("validity"),
            records=[make_draft(
                "v1", "temporary launch password", observed_at=100,
                valid_to=150, source_lineage=("original-source",))])
        assert admission.outcome.status == "ok"
        inside = store.select(
            "temporary launch password", scope=scope("validity", as_of=120),
            budget=RetrievalBudget(max_records=1, max_value_slots=4))
        assert inside.record_ids == ("v1",)
        assert inside.outcome.status == "ok"
        store.release_selection(inside.selection_id)
        after = store.select(
            "temporary launch password", scope=scope("validity", as_of=200))
        assert after.record_ids == ()
        assert after.outcome.status == "incomplete"
        assert after.outcome.reason == "no_semantic_candidate"
        before = store.select(
            "temporary launch password", scope=scope("validity", as_of=90))
        assert before.record_ids == ()
        store.release_selection(after.selection_id)
        store.release_selection(before.selection_id)


# ---------------------------------------------------------------------------
# Cold recall: wait/resume, revalidation, correction and stale versions
# ---------------------------------------------------------------------------

def test_cold_recall_wait_resume_revalidation_and_cross_scope(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user_scope = scope()
        other_scope = scope("someone_else")
        admission = store.admit_cluster(
            "cold", scope=user_scope, placement="cold",
            records=[make_draft("c1", "alphaunique cold payload", seed=2),
                     make_draft("c2", "betaunique cold payload", seed=3)])
        selected = store.select(
            "alphaunique cold payload", scope=user_scope,
            budget=RetrievalBudget(max_records=1, max_value_slots=4))
        assert selected.requires_cold_recall is True
        assert selected.outcome.status == "incomplete"
        assert selected.records[0].values_available is False
        assert selected.used_value_slots == 0
        assert "c1" in selected.cold_record_ids

        ticket = store.begin_recall(selected.selection_id)
        assert ticket.status == "pending"
        assert ticket.required_bytes > 0
        waiting = store.wait_recall(ticket.ticket_id, 0)
        assert waiting.status == "in_progress"
        ready = store.resume_recall(ticket.ticket_id)
        assert ready.status == "ready"
        assert ready.io_bytes >= ticket.required_bytes
        recalled = {record.record_id: record for record in ready.records}
        assert set(recalled) == {"c1"}
        assert recalled["c1"].text == "alphaunique cold payload"
        assert recalled["c1"].value_slots == 4
        assert len(recalled["c1"].values[0]) == VALUE_WIDTH

        denied = store.read_record("c1", scope=other_scope)
        assert denied.status == "denied"
        assert denied.outcome.reason == "cross_scope_access"

        # A correction between ticket creation and post-I/O revalidation must
        # discard both the selected payload and the old decision boundary.
        beta_selection = store.select(
            "betaunique cold payload", scope=user_scope,
            budget=RetrievalBudget(max_records=1, max_value_slots=4))
        beta_ticket = store.begin_recall(beta_selection.selection_id)
        correction = store.correct_record(
            "c2", correction_id="c2-corrected", text="beta corrected payload",
            source_id="correction-source", source_epoch=2, observed_at=200,
            values=sample_values(9), value_producer_id="test-values",
            value_producer_version="test-values-v1", scope=user_scope)
        invalid = store.resume_recall(beta_ticket.ticket_id)
        assert invalid.status == "invalidated"
        assert invalid.records == ()
        assert invalid.new_decision_boundary is True
        assert invalid.outcome.reason == "record_revoked_or_superseded_during_recall"
        assert correction.exact_unlearning_claimed is False

        old_status = store.connection.execute(
            "SELECT status FROM records WHERE record_id='c2'").fetchone()["status"]
        assert old_status == "superseded"
        assert store.read_record("c2", scope=user_scope).status == "invalidated"
        corrected = store.read_record(correction.new_record_id, scope=user_scope)
        assert corrected.status == "ready"
        assert corrected.records[0].text == "beta corrected payload"
        assert admission.placement == "cold"
        store.release_selection(selected.selection_id)
        store.release_selection(beta_selection.selection_id)


def test_stale_representation_and_exact_span_fallback(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user_scope = scope()
        text = "The launch code is ALPHA-7; keep the exact source text."
        admission = store.admit_cluster(
            "stale", scope=user_scope,
            records=[make_draft("s1", text, seed=4)])
        assert admission.generation == 1
        invalidation = store.invalidate_representations(
            "v1", reason="encoder version superseded")
        assert invalidation.reencode_required is True
        assert invalidation.stale_keys >= 1
        assert invalidation.outcome.status == "degraded"

        fallback = store.select(
            "launch code", scope=user_scope,
            exact_phrases=("ALPHA-7",))
        assert fallback.outcome.status == "degraded"
        assert fallback.representation_stale is True
        assert fallback.records == ()
        assert fallback.fallback_mode == "exact_text"
        assert len(fallback.exact_spans) == 1
        span = fallback.exact_spans[0]
        assert span.text == "ALPHA-7"
        assert span.start == text.index("ALPHA-7")
        assert span.end == span.start + len("ALPHA-7")
        assert fallback.used_span_bytes == len("ALPHA-7")
        assert fallback.used_value_slots == 0
        key_status = store.connection.execute(
            "SELECT status FROM record_keys WHERE record_id='s1'").fetchone()["status"]
        assert key_status == "stale"

        # A cold ticket opened before invalidation must not return a value
        # produced by the stale encoder after the I/O.
        cold = store.admit_cluster(
            "cold-stale", scope=user_scope, placement="cold",
            records=[make_draft("s2", "cold stale token", seed=5)])
        selected = store.select("cold stale token", scope=user_scope)
        ticket = store.begin_recall(selected.selection_id)
        store.invalidate_representations("v1", reason="second encoder update")
        stale = store.resume_recall(ticket.ticket_id)
        assert stale.status in ("stale", "invalidated")
        assert stale.records == ()
        assert stale.new_decision_boundary is True
        assert cold.generation == 1
        store.release_selection(selected.selection_id)


# ---------------------------------------------------------------------------
# Corrections, procedures, P quarantine and preserved failed evidence
# ---------------------------------------------------------------------------

def test_correction_quarantines_p_and_preserves_failed_procedures(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        user_scope = scope()
        store.admit_cluster(
            "evidence", scope=user_scope,
            records=[make_draft("r1", "old exact claim", seed=11)])
        profile = get_profile("P1")
        assert store.register_expert("e1", lineage=("r1",)).location == "cold"
        write = store.write_expert_state(
            "e1", weights=payload(profile.full_parameters * 4, b"weights"),
            moments=payload(profile.full_parameters * 8, b"moments"),
            clock=41, lineage=("r1",))
        assert write.ok is True
        published = store.publish_expert("e1")
        assert published.ok is True
        assert published.detail["generation"] == 1

        verified = store.record_procedure(
            "procedure.alpha", 1, title="verified path",
            applicability={"task": "alpha"}, prerequisites=["source"],
            steps=["step one"], outcome="verified",
            evidence_record_ids=("r1",), verified_by="verifier")
        failed = store.record_procedure(
            "procedure.alpha", 2, title="failed path",
            applicability={"task": "alpha"}, prerequisites=["source"],
            steps=["step one"], outcome="failed", error="verification failed",
            evidence_record_ids=("r1",))
        assert verified.ok and failed.ok

        correction = store.correct_record(
            "r1", correction_id="r1-corrected", text="corrected exact claim",
            source_id="user-correction", source_epoch=2, observed_at=300,
            importance=90, values=sample_values(12),
            value_producer_id="test-values",
            value_producer_version="correction-values-v1", scope=user_scope)
        assert correction.exact_unlearning_claimed is False
        assert correction.quarantined_experts == ("e1",)
        assert set(correction.invalidated_procedures) == {
            ("procedure.alpha", 1), ("procedure.alpha", 2)}
        assert correction.stale_representations >= 1
        assert correction.outcome.status == "ok"

        old = store.connection.execute(
            "SELECT status, text_path FROM records WHERE record_id='r1'").fetchone()
        assert old["status"] == "superseded"
        assert (store.root / old["text_path"]).exists()
        old_values = store.connection.execute(
            "SELECT status FROM record_values WHERE record_id='r1'").fetchall()
        assert old_values and all(row["status"] == "stale" for row in old_values)

        experts = store.expert_records()
        assert experts[0].quarantine_status == "quarantined"
        loaded = store.load_expert("e1")
        assert loaded.state is not None
        assert loaded.outcome.status == "degraded"
        assert loaded.state.quarantine_status == "quarantined"
        assert loaded.state.clock == 41

        procedures = store.procedures(include_failed=True)
        failed_rows = [row for row in procedures if row["outcome"] == "failed"]
        assert len(failed_rows) == 1
        assert failed_rows[0]["error"] == "verification failed"
        assert all(row["usable"] is False for row in procedures)
        assert all(row["invalidated_by_evidence"] is True for row in procedures)

        selected = store.select("corrected exact claim", scope=user_scope)
        assert correction.new_record_id in selected.record_ids
        assert "r1" not in selected.record_ids
        store.release_selection(selected.selection_id)


# ---------------------------------------------------------------------------
# Dirty P persistence before eviction, active limit and cold capacity
# ---------------------------------------------------------------------------

def test_dirty_expert_is_published_before_eviction(tmp_path: Path) -> None:
    ceilings = policy(hot_bytes=64 * 1024**2, cold_bytes=64 * 1024**2)
    with open_store(tmp_path, ceilings=ceilings) as store:
        profile = get_profile("P1")
        weights = payload(profile.full_parameters * 4, b"weights-one")
        moments = payload(profile.full_parameters * 8, b"moments-one")
        store.register_expert("e1", lineage=("lineage-record",))
        store.register_expert("e2", lineage=())
        assert store.write_expert_state(
            "e1", weights=weights, moments=moments, clock=7).ok
        limited = store.write_expert_state(
            "e2", weights=weights, moments=moments, clock=8)
        assert limited.status == "unsupported"
        assert store.expert_records()[0].location == "hot"

        evicted = store.evict_expert("e1")
        assert evicted.ok is True
        records = {record.expert_id: record for record in store.expert_records()}
        assert records["e1"].location == "cold"
        assert records["e1"].dirty is False
        assert records["e1"].publication_generation == 1
        assert not (store.root / "storage/hot/experts/e1/weights.bin").exists()
        cold_weights = store.root / (
            "storage/cold/experts/e1/gen-00000001/weights.bin")
        assert cold_weights.exists()
        loaded = store.load_expert("e1")
        assert loaded.outcome.ok is True
        assert loaded.state is not None
        assert loaded.state.weights == weights
        assert loaded.state.moments == moments
        assert loaded.state.clock == 7
        assert loaded.state.source == "cold_publication"

        # The eviction released one active slot; the second owner can now be
        # materialized under the same one-active policy.
        assert store.write_expert_state(
            "e2", weights=weights, moments=moments, clock=9).ok
        assert sum(record.location == "hot"
                   for record in store.expert_records()) == 1  # inspection only


def test_dirty_expert_eviction_refuses_when_publication_cannot_fit(
        tmp_path: Path) -> None:
    ceilings = policy(
        hot_bytes=64 * 1024**2,
        cold_bytes=4 * 1024**2,
        artifact_budget_bytes=512 * 1024,
        workspace_quota_bytes=256 * 1024,
    )
    with open_store(tmp_path, ceilings=ceilings) as store:
        profile = get_profile("P1")
        store.register_expert("e1", lineage=("r1",))
        assert store.write_expert_state(
            "e1",
            weights=payload(profile.full_parameters * 4, b"weights-fail"),
            moments=payload(profile.full_parameters * 8, b"moments-fail"),
            clock=13).ok
        refused = store.evict_expert("e1")
        assert refused.status == "incomplete"
        assert refused.detail["dirty_retained"] is True
        assert refused.detail["hot_retained"] is True
        record = store.expert_records()[0]
        assert record.location == "hot" and record.dirty is True
        assert (store.root / "storage/hot/experts/e1/weights.bin").exists()
        assert not (store.root / "storage/cold/experts/e1/gen-00000001").exists()
        loaded = store.load_expert("e1")
        assert loaded.outcome.ok is True and loaded.state is not None
        assert loaded.state.source == "hot_dirty"
        assert loaded.state.clock == 13


def test_two_active_ablation_is_explicit(tmp_path: Path) -> None:
    ceilings = policy(hot_bytes=96 * 1024**2, cold_bytes=64 * 1024**2)
    with open_store(tmp_path, ceilings=ceilings, profile_id="P4",
                    two_active_ablation=True) as store:
        allocation = store.allocation(active_experts=2)
        assert allocation.active_experts == 2
        assert allocation.ablation == "two_active_ablation_v1"
        assert allocation.gpu_inactive_bytes == 0
        assert allocation.all_banks_allocated is False
        profile = get_profile("P4")
        weights = payload(profile.per_expert_fp32_weight_bytes, b"ablation-weights")
        moments = payload(profile.per_expert_moments_bytes, b"ablation-moments")
        store.register_expert("a1")
        store.register_expert("a2")
        assert store.write_expert_state(
            "a1", weights=weights, moments=moments, clock=1).ok
        assert store.write_expert_state(
            "a2", weights=weights, moments=moments, clock=2).ok
        assert sum(record.location == "hot"
                   for record in store.expert_records()) == 2


# ---------------------------------------------------------------------------
# Recovery from manifests after index loss; interrupted writes preserved
# ---------------------------------------------------------------------------

def test_crash_rebuild_from_manifests_and_corrupt_manifest_preserved(
        tmp_path: Path) -> None:
    ceilings = policy(hot_bytes=64 * 1024**2, cold_bytes=64 * 1024**2)
    store = open_store(tmp_path, ceilings=ceilings)
    user_scope = scope()
    admission = store.admit_cluster(
        "rebuild", scope=user_scope,
        records=[make_draft("r1", "first rebuild record", seed=21),
                 make_draft("r2", "second rebuild record", seed=22)])
    profile = get_profile("P1")
    weights = payload(profile.full_parameters * 4, b"rebuild-weights")
    moments = payload(profile.full_parameters * 8, b"rebuild-moments")
    store.register_expert("e1", lineage=("r1",))
    assert store.write_expert_state(
        "e1", weights=weights, moments=moments, clock=31).ok
    assert store.publish_expert("e1").ok
    assert store.evict_expert("e1").ok
    cluster_id = admission.cluster_id
    store.close()

    for suffix in ("", "-wal", "-shm"):
        path = Path(str(store.sqlite_path) + suffix)
        path.unlink(missing_ok=True)
    (store.root / "staging/.tmp-interrupted").write_bytes(b"interrupted")
    bad_manifest = store.root / f"manifests/clusters/{cluster_id}/00000099.json"
    bad_manifest.parent.mkdir(parents=True, exist_ok=True)
    bad_manifest.write_bytes(b"not a manifest")

    rebuilt = open_store(tmp_path, ceilings=ceilings)
    try:
        report = rebuilt.recovery_report
        assert report is not None
        assert report.accounting_rebuilt is True
        assert cluster_id in report.clusters_recovered
        assert "e1" in report.experts_recovered
        assert report.generations_recovered >= 1
        assert report.records_recovered >= 2
        assert f"manifests/clusters/{cluster_id}/00000099.json" in (
            report.corrupt_manifests)
        assert "staging/.tmp-interrupted" in report.ignored_temporaries
        assert report.outcome.status == "incomplete"
        assert not (store.root / "staging/.tmp-interrupted").exists()
        assert bad_manifest.exists()
        assert rebuilt.generation_manifest(cluster_id).document["record_count"] == 2
        loaded = rebuilt.load_expert("e1")
        assert loaded.outcome.ok is True and loaded.state is not None
        assert loaded.state.weights == weights
        assert loaded.state.moments == moments
        assert loaded.state.clock == 31
        assert rebuilt.audit_disk()["accounting"]["tiers"]["cold"]["used_bytes"] > 0
        summary = rebuilt.state_summary()
        assert summary["counts"]["E_records"] >= 2
        assert summary["counts"]["P_experts"] == 1
        assert summary["native_capability"] is False
    finally:
        rebuilt.close()


# ---------------------------------------------------------------------------
# Accounting, migration, replay, publication overlap and checkpoints
# ---------------------------------------------------------------------------

def test_lazy_tier_accounting_migration_overlap_and_checkpoints(
        tmp_path: Path) -> None:
    ceilings = policy(
        hot_bytes=8 * 1024**2, warm_bytes=8 * 1024**2,
        cold_bytes=8 * 1024**2, identity_reserved_bytes=64 * 1024,
        replay_reserved_bytes=256 * 1024, workspace_quota_bytes=256 * 1024,
        artifact_budget_bytes=512 * 1024)
    with open_store(tmp_path, ceilings=ceilings) as store:
        usage = store.usage()
        assert usage["tiers"]["hot"]["limit_bytes"] == ceilings.hot_bytes
        assert usage["tiers"]["hot"]["identity_used_bytes"] == 0
        assert usage["tiers"]["hot"]["non_identity_limit_bytes"] == (
            ceilings.hot_bytes - ceilings.identity_reserved_bytes)
        assert usage["tiers"]["cold"]["by_category"].get("database", 0) > 0

        assert store.write_replay("episode-1", b"replay" * 16).ok
        rejected = store.write_replay("episode-2", b"r" * (300 * 1024))
        assert rejected.status == "incomplete"
        replay_usage = store.usage()
        assert replay_usage["tiers"]["warm"]["replay_used_bytes"] == 96
        # The exact replay bytes are the written payload, not the attempted one.
        assert replay_usage["tiers"]["warm"]["by_category"]["replay"] == 96

        observed = store.observe_page_cache_pressure(
            4096, source="test host observation")
        assert observed["page_cache_pressure_bytes"] == 4096
        assert observed["page_cache_is_distinct_from_logical"] is False

        store.admit_cluster(
            "migrate", scope=scope(),
            records=[make_draft("m1", "migration record", seed=31)])
        warm_before = store.usage()["tiers"]["warm"]["by_category"]
        moved = store.migrate_records(["m1"], to_tier="cold")
        assert moved.ok is True
        warm_after = store.usage()["tiers"]["warm"]["by_category"]
        cold_after = store.usage()["tiers"]["cold"]["by_category"]
        assert warm_after.get("canonical_text", 0) < warm_before["canonical_text"]
        assert warm_after.get("values", 0) == 0
        assert warm_after.get("keys", 0) == KEY_WIDTH
        assert cold_after.get("canonical_text", 0) >= len("migration record")
        assert cold_after.get("values", 0) >= 4 * VALUE_WIDTH * 4
        assert (store.root / "storage/cold/records/m1/text.bin").exists()
        assert not (store.root / "storage/warm/records/m1/text.bin").exists()

        staged = store.stage_publication("publication-1", b"p" * 1000)
        assert staged.ok is True
        assert (store.root / "staging/publication-1.bin").exists()
        cold_overlap = store.usage()["tiers"]["cold"]["by_category"]
        assert cold_overlap["publication_overlap"] == 1000
        released = store.release_publication("publication-1")
        assert released.ok is True
        assert not (store.root / "staging/publication-1.bin").exists()
        assert "publication_overlap" not in (
            store.usage()["tiers"]["cold"]["by_category"])

        first_checkpoint = store.publish_checkpoint("ckpt", b"c" * 128)
        second_checkpoint = store.publish_checkpoint("ckpt", b"d" * 128)
        assert first_checkpoint.ok and second_checkpoint.ok
        assert [row["generation"] for row in store.checkpoints()] == [1, 2]
        for row in store.checkpoints():
            assert (store.root / row["payload_path"]).exists()

        audit = store.audit_disk()
        assert audit["bytes_by_area"]["cold"] > 0
        assert audit["journal_lines"] > 0
        assert audit["interrupted_temporaries"] == ()
        assert audit["page_cache_pressure_is_not_added_to_logical_usage"] is True
        assert audit["volume_headroom_satisfied"] is True
        assert audit["accounting"]["tiers"]["cold"]["used_bytes"] <= (
            ceilings.cold_bytes)


def test_bounded_disk_growth_rejects_new_generation_without_deleting_old(
        tmp_path: Path) -> None:
    ceilings = policy(
        hot_bytes=8 * 1024**2, warm_bytes=8 * 1024**2,
        cold_bytes=4 * 1024**2, identity_reserved_bytes=32 * 1024,
        replay_reserved_bytes=128 * 1024, workspace_quota_bytes=64 * 1024,
        artifact_budget_bytes=512 * 1024)
    with open_store(tmp_path, ceilings=ceilings) as store:
        successes = 0
        failure = None
        for index in range(12):
            outcome = store.publish_checkpoint("bounded", b"z" * 700_000)
            if outcome.ok:
                successes += 1
            else:
                failure = outcome
                break
        assert successes >= 1
        assert failure is not None and failure.status == "incomplete"
        checkpoints = store.checkpoints()
        assert len(checkpoints) == successes
        for row in checkpoints:
            assert (store.root / row["payload_path"]).exists()
        assert store.audit_disk()["interrupted_temporaries"] == ()
        assert store.usage()["tiers"]["cold"]["within_limit"] is True


# ---------------------------------------------------------------------------
# Frequent + critical retention and hot/warm eviction
# ---------------------------------------------------------------------------

def test_hot_cache_protects_critical_and_frequent_entries(tmp_path: Path) -> None:
    ceilings = policy(
        hot_bytes=256 * 1024, identity_reserved_bytes=8 * 1024,
        warm_bytes=1 * 1024**2, cold_bytes=8 * 1024**2,
        replay_reserved_bytes=128 * 1024, workspace_quota_bytes=64 * 1024,
        artifact_budget_bytes=512 * 1024,
        frequent_reserve_fraction=0.5, frequent_min_accesses=2)
    with open_store(tmp_path, ceilings=ceilings) as store:
        user_scope = scope()
        store.admit_cluster(
            "cache", scope=user_scope,
            records=[
                make_draft("r1", "ordinary one", seed=41, critical=True),
                make_draft("r2", "quokka", seed=42),
                make_draft("r3", "ordinary three", seed=43),
                make_draft("r4", "ordinary four", seed=44),
            ])
        for _ in range(2):
            selected = store.select(
                "quokka", scope=user_scope,
                budget=RetrievalBudget(max_records=1, max_value_slots=4))
            assert selected.record_ids == ("r2",)
            store.release_selection(selected.selection_id)
        assert store.retrieval_count("r2") == 2
        for record_id in ("r1", "r2", "r3", "r4"):
            cached = store.cache_record(record_id)
            assert cached.ok is True, cached.document()
        assert len(store.hot_cache_snapshot()) == 4

        trimmed = store.trim_hot_cache(9 * 1024)
        assert trimmed.ok is True
        assert trimmed.status == "ok"
        remaining = {entry["record_id"] for entry in store.hot_cache_snapshot()}
        assert remaining == {"r1", "r2"}
        assert "r3" not in remaining and "r4" not in remaining

        protected_only = store.trim_hot_cache(1)
        assert protected_only.status == "incomplete"
        assert protected_only.reason == "hot_cache_reserved_only"
        assert {entry["record_id"] for entry in store.hot_cache_snapshot()} == {
            "r1", "r2"}

        # Cold payloads never enter hot cache through a hidden read.
        cold_admission = store.admit_cluster(
            "cold-cache", scope=user_scope, placement="cold",
            records=[make_draft("cold1", "cold cache token", seed=45)])
        cold_cache = store.cache_record("cold1")
        assert cold_cache.status == "incomplete"
        assert cold_cache.reason == "cold_payload_requires_wait_resume"
        assert cold_admission.placement == "cold"


# ---------------------------------------------------------------------------
# Artifact storage: extensionless type receipts and hard bound
# ---------------------------------------------------------------------------

def test_extensionless_artifacts_have_size_type_receipts_and_bound(
        tmp_path: Path) -> None:
    ceilings = policy(
        hot_bytes=8 * 1024**2, warm_bytes=8 * 1024**2,
        cold_bytes=8 * 1024**2, identity_reserved_bytes=32 * 1024,
        replay_reserved_bytes=128 * 1024, workspace_quota_bytes=64 * 1024,
        artifact_budget_bytes=8 * 1024)
    source = tmp_path / "state"
    source.write_bytes(b"PK\x03\x04" + b"payload" * 20)
    big_source = tmp_path / "state-big"
    big_source.write_bytes(b"7z\xbc\xaf\x27\x1c" + b"x" * (16 * 1024))
    with open_store(tmp_path, ceilings=ceilings) as store:
        receipt = store.store_artifact(source, pinned=True, label="extensionless state")
        assert receipt.status == "stored"
        assert receipt.outcome.status == "ok"
        assert receipt.detected_type == "zip_archive"
        assert receipt.bytes == source.stat().st_size
        assert receipt.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
        assert receipt.magic_hex.startswith("504b0304")
        assert (store.root / receipt.path).exists()
        assert receipt.document()["extension_used_for_detection"] is False

        rejected = store.store_artifact(big_source)
        assert rejected.status == "rejected"
        assert rejected.outcome.reason == "artifact_budget_exceeded"
        assert rejected.path == ""
        assert "7z" in rejected.detected_type
        assert (store.root / receipt.path).exists()

        historical = store.root / "artifacts/sha256/historical/state"
        historical.parent.mkdir(parents=True, exist_ok=True)
        historical.write_bytes(b"SQLite format 3\x00" + b"x" * 48)
        scan = store.scan_artifacts()
        assert scan.outcome.status == "ok"
        found = [item for item in scan.receipts
                 if item.path.endswith("historical/state")]
        assert len(found) == 1
        assert found[0].detected_type == "sqlite_database"
        assert found[0].bytes == historical.stat().st_size
        assert found[0].outcome.status == "ok"
        assert historical.exists()
        assert any(row["detected_type"] == "sqlite_database"
                   for row in store.artifact_receipts())
        detected, magic, size = detect_file_type(historical)
        assert detected == "sqlite_database" and size > 0 and magic
        assert store.usage()["tiers"]["cold"]["by_category"]["artifacts"] <= (
            ceilings.artifact_budget_bytes)


# ---------------------------------------------------------------------------
# Identity, workspace, capability and class separation
# ---------------------------------------------------------------------------

def test_identity_workspace_and_capability_remain_distinct(
        tmp_path: Path) -> None:
    ceilings = policy(
        hot_bytes=1 * 1024**2, warm_bytes=1 * 1024**2,
        cold_bytes=8 * 1024**2, identity_reserved_bytes=32,
        replay_reserved_bytes=128 * 1024, workspace_quota_bytes=64,
        artifact_budget_bytes=512 * 1024)
    store = open_store(tmp_path, ceilings=ceilings)
    user_scope = scope()
    try:
        approved = store.approve_identity(
            "identity.main", 1, core_text="abc", authority="user_approved")
        assert approved.outcome.status == "ok"
        assert approved.bytes == 3
        denied = store.approve_identity(
            "identity.main", 2, core_text="self", authority="self_generated")
        assert denied.outcome.status == "denied"
        over = store.approve_identity(
            "identity.main", 2, core_text="x" * 40, authority="user_approved")
        assert over.outcome.status == "incomplete"
        current = store.identity_core()
        assert current.identity_id == "identity.main"
        assert current.version == 1 and current.text == "abc"
        assert current.bytes == 3

        ephemeral = store.save_workspace_snapshot(
            "task-ephemeral", {"goal": "temporary"}, resumable=False)
        resumable = store.save_workspace_snapshot(
            "task-resumable", {"goal": "resume later"}, resumable=True)
        assert ephemeral.outcome.ok and ephemeral.tier == "hot"
        assert resumable.outcome.ok and resumable.tier == "cold"
        too_big = store.save_workspace_snapshot(
            "task-too-big", b"x" * 100, resumable=False)
        assert too_big.outcome.status == "incomplete"
        assert too_big.outcome.reason == "workspace_quota_exceeded"
        ephemeral_path = store.root / ephemeral.path
        resumable_path = store.root / resumable.path
        assert ephemeral_path.exists() and resumable_path.exists()

        capability = store.record_capability(
            "cap.answer.v1", model_id="synthetic-engineering",
            mode="cpu_fixture", task_family="answer",
            budget={"records": 8}, measured_value=0.5,
            uncertainty={"samples": 0}, measured=True)
        assert capability.ok is True
        assert store.capabilities(task_family="answer")[0]["measured"] is True
        summary = store.state_summary()
        assert summary["counts"]["I_approved_current"] == 1
        assert summary["counts"]["C_workspace_snapshots"] == 2
        assert summary["counts"]["K_measured"] == 1
        assert summary["counts"]["K_capability_records"] == 1
        assert summary["state_classes"] == STATE_CLASSES
        assert summary["values_hold_identity_workspace_or_p"] is False
        assert summary["native_capability"] is False

        # A P-query has a separate slot budget and never returns values.
        store.register_expert("idle-expert")
        p_read = store.read_p_query("idle-expert", slots=8)
        assert p_read["outcome"]["status"] == "ok"
        assert p_read["evidence_value_slots_borrowed"] == 0
        assert p_read["parameter_values_returned"] is False
        with pytest.raises(BudgetError):
            store.read_p_query("idle-expert", slots=33)
    finally:
        store.close()
    assert not ephemeral_path.exists()
    assert resumable_path.exists()

    reopened = open_store(tmp_path, ceilings=ceilings)
    try:
        assert reopened.identity_core().text == "abc"
        assert reopened.workspace_snapshots() == tuple([
            {"task_id": "task-resumable",
             "path": resumable.path,
             "sha256": resumable.sha256,
             "bytes": resumable.bytes,
             "resumable": True}])
        assert reopened.capabilities()[0]["capability_id"] == "cap.answer.v1"
    finally:
        reopened.close()


def test_procedure_failed_evidence_survives_without_reactivation(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        verified = store.record_procedure(
            "skill.one", 1, title="works", applicability={},
            prerequisites=[], steps=["a"], outcome="verified",
            evidence_record_ids=("never-corrected",), verified_by="host")
        failed = store.record_procedure(
            "skill.one", 2, title="fails", applicability={},
            prerequisites=[], steps=["a"], outcome="failed",
            error="known negative", evidence_record_ids=())
        assert verified.ok and failed.ok
        rows = store.procedures(include_failed=True)
        assert len(rows) == 2
        assert [row["outcome"] for row in rows] == ["verified", "failed"]
        assert rows[0]["usable"] is True
        assert rows[1]["usable"] is False
        assert rows[1]["error"] == "known negative"
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "UPDATE procedures SET outcome='verified' "
                "WHERE skill_id='skill.one' AND version=2")


def test_tier_store_enforces_volume_free_policy(tmp_path: Path) -> None:
    ceilings = policy(minimum_volume_free_bytes=10**18)
    store = TieredMemoryStore(
        tmp_path / "volume", profile_id="P1", ceilings=ceilings,
        enforce_volume_free=True)
    try:
        from replica_lab.tiered_memory import InsufficientVolumeFree
        with pytest.raises(InsufficientVolumeFree):
            store.admit_cluster(
                "blocked", scope=scope(),
                records=[make_draft("r1", "space check")])
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Explicit stale/incomplete paths and private-file safety
# ---------------------------------------------------------------------------

def test_cold_ticket_unknown_and_selection_unknown_are_explicit(
        tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        unknown = store.begin_recall("sel_missing")
        assert unknown.status == "unsupported"
        assert unknown.outcome.reason == "selection_unknown_after_restart"
        missing = store.resume_recall("tkt_missing")
        assert missing.status == "unsupported"
        assert missing.outcome.reason == "recall_ticket_unknown"
        no_scope = store.identity_core()
        assert no_scope.outcome.status == "incomplete"
        assert store.record_capability(
            "cap.unmeasured", model_id="synthetic", mode="fixture",
            task_family="answer", budget={}, measured_value=None,
            uncertainty={"samples": 0}, measured=False).ok is True
        assert store.capabilities()[0]["measured"] is False
        assert store.usage()["physical_vram_pool_invented"] is False
        assert store.audit_disk()["native_capability"] is False
        assert store.state_summary()["usage"]["native_capability"] is False
        assert "error" not in store.audit_disk()


def test_capacity_exception_does_not_leak_files(tmp_path: Path) -> None:
    ceilings = policy(
        hot_bytes=512 * 1024, cold_bytes=256 * 1024,
        identity_reserved_bytes=16 * 1024, warm_bytes=512 * 1024,
        replay_reserved_bytes=64 * 1024, workspace_quota_bytes=32 * 1024,
        artifact_budget_bytes=64 * 1024)
    with open_store(tmp_path, ceilings=ceilings) as store:
        with pytest.raises(CapacityExceeded):
            store.admit_cluster(
                "too-large", scope=scope(),
                records=[make_draft("big", "x" * 600_000, include_values=False)])
        assert store.connection.execute(
            "SELECT COUNT(*) AS value FROM records").fetchone()["value"] == 0
        assert not any(store.root.rglob("storage/*/records/big/*"))
        assert store.audit_disk()["interrupted_temporaries"] == ()
