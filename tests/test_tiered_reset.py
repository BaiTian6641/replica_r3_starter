"""Regression coverage for authenticated correction-dependent P reset."""
from __future__ import annotations

from pathlib import Path

from test_tiered_memory import make_draft, open_store, payload, scope
from replica_lab.tiered_memory import get_profile


def test_authenticated_reset_replaces_quarantined_state_without_unlearning_claim(tmp_path: Path):
    with open_store(tmp_path) as store:
        store.admit_cluster("reset", scope=scope(), records=[make_draft("r1", "old rule", seed=101)])
        profile = get_profile("P1")
        weights = payload(profile.full_parameters * 4, b"before-reset")
        moments = payload(profile.full_parameters * 8, b"moments-reset")
        store.register_expert("e1", lineage=("r1",))
        assert store.write_expert_state("e1", weights=weights, moments=moments, clock=7).ok
        assert store.publish_expert("e1").ok
        correction = store.correct_record(
            "r1", correction_id="r1-corrected", text="new rule", source_id="fixture",
            source_epoch=2, observed_at=2, scope=scope(),
            authenticity="source_report", verification_state="verified_source")
        assert correction.quarantined_experts == ("e1",)
        denied = store.reset_expert(
            "e1", weights=weights, moments=moments, clock=0,
            authorization="unauthenticated", reason="test")
        assert denied.status == "denied"
        assert store.expert_records()[0].quarantine_status == "quarantined"
        reset = store.reset_expert(
            "e1", weights=weights, moments=moments, clock=0,
            authorization="authenticated_reset", reason="test correction")
        assert reset.ok is True
        assert reset.detail["exact_unlearning_claimed"] is False
        loaded = store.load_expert("e1")
        assert loaded.outcome.ok is True
        assert loaded.state is not None
        assert loaded.state.quarantine_status == "none"
        assert loaded.state.clock == 0
        assert loaded.state.lineage == ()
        assert store.expert_records()[0].publication_generation == 1
        assert (store.root / "storage/cold/experts/e1/gen-00000001").is_dir()
