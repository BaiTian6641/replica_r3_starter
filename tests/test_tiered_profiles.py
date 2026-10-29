"""Geometry regressions: a bank total is never one expert's shape.

P1 has one expert, so its bank and per-expert shapes coincide.  P4 stores four
experts in a 1,572,864-parameter bank; each owner still has 393,216 parameters
and is validated, charged and reloaded at that per-expert shape.  These tests
never materialize P16/P32/P64 or any native model.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from replica_lab.tiered_memory import (
    get_profile,
    plan_profile,
)
from test_tiered_memory import open_store, payload


def expert_payload(profile_id: str, seed: bytes) -> tuple[bytes, bytes]:
    profile = get_profile(profile_id)
    return (payload(profile.per_expert_fp32_weight_bytes, seed + b"-weights"),
            payload(profile.per_expert_moments_bytes, seed + b"-moments"))


def test_p4_geometry_separates_bank_totals_from_per_expert_shapes() -> None:
    p1 = get_profile("P1")
    p4 = get_profile("P4")
    assert p1.experts == 1
    assert p1.full_parameters == 393_216
    assert p1.per_expert_parameters == 393_216
    assert p4.experts == 4
    assert p4.full_parameters == 1_572_864
    assert p4.per_expert_parameters == 393_216
    assert p4.per_expert_fp32_weight_bytes == 393_216 * 4
    assert p4.per_expert_moments_bytes == 393_216 * 8
    assert p4.full_parameters == p4.experts * p4.per_expert_parameters

    receipt = p4.arithmetic_receipt()
    assert receipt["bank_parameters"] == 1_572_864
    assert receipt["per_expert_parameters"] == 393_216
    assert receipt["per_expert_fp32_weight_bytes"] == 393_216 * 4
    assert receipt["allocation"] is False

    # Active-one planning is per expert; the stored bank total stays distinct
    # and the inactive bank is never materialized or counted as active.
    plan = plan_profile("P4")
    assert plan.active_experts == 1
    assert plan.stored_experts == 4
    assert plan.inactive_experts == 3
    assert plan.active_full_training_bytes == 393_216 * 16
    assert plan.inactive_full_training_bytes == 3 * 393_216 * 16
    assert plan.gpu_active_bytes == plan.active_full_training_bytes
    assert plan.gpu_inactive_bytes == 0
    assert plan.all_banks_allocated is False
    document = plan.document()
    assert document["stored_experts"] == 4
    assert document["active_experts"] == 1
    assert document["all_banks_allocated"] is False


def test_p4_registration_write_reload_and_budget_use_per_expert_shape(
        tmp_path: Path) -> None:
    profile = get_profile("P4")
    weights, moments = expert_payload("P4", b"p4-owner-one")
    with open_store(tmp_path, profile_id="P4") as store:
        record = store.register_expert("e1", lineage=("r1",),
                                       encoder_id="encoder-v1")
        assert record.profile_id == "P4"
        assert record.parameter_count == 393_216
        assert record.parameter_count != profile.full_parameters
        assert record.parameter_count <= profile.full_parameters

        # A whole-bank payload for one owner is rejected, not reinterpreted.
        with pytest.raises(ValueError, match="one expert's parameter_count"):
            store.write_expert_state(
                "e1",
                weights=payload(profile.full_parameters * 4, b"wrong"),
                moments=payload(profile.full_parameters * 8, b"wrong"),
                clock=1)

        written = store.write_expert_state(
            "e1", weights=weights, moments=moments, clock=7,
            lineage=("r1",), encoder_id="encoder-v1")
        assert written.ok is True
        assert written.detail["bytes"] == 393_216 * 12
        hot = store.usage()["tiers"]["hot"]
        assert hot["by_category"]["expert_active"] == 393_216 * 12

        loaded = store.load_expert("e1")
        assert loaded.outcome.ok is True
        assert loaded.state is not None
        assert len(loaded.state.weights) == 393_216 * 4
        assert len(loaded.state.moments) == 393_216 * 8
        assert loaded.state.weights == weights
        assert loaded.state.moments == moments
        assert loaded.state.generation == 0

        published = store.publish_expert("e1")
        assert published.ok is True
        # Weights + moments are per-expert; the manifest envelope is separate.
        assert published.detail["bytes"] > 393_216 * 12
        assert store.evict_expert("e1").ok
        cold = store.load_expert("e1")
        assert cold.outcome.ok is True and cold.state is not None
        assert cold.state.source == "cold_publication"
        assert cold.state.clock == 7
        assert len(cold.state.weights) == 393_216 * 4

        # A second owner is cold metadata only until it, not the bank, is
        # explicitly activated.  The evicted first owner no longer counts.
        second = store.register_expert("e2", lineage=("r2",))
        assert second.parameter_count == 393_216
        assert store.write_expert_state(
            "e2", weights=weights, moments=moments, clock=9).ok
        assert store.usage()["tiers"]["hot"]["by_category"][
            "expert_active"] == 393_216 * 12
        assert all(record.parameter_count == 393_216
                   for record in store.expert_records())
        plan = store.allocation()
        assert plan.active_experts == 1
        assert plan.stored_experts == 4


def test_p4_active_limit_is_still_one_without_ablation(tmp_path: Path) -> None:
    weights, moments = expert_payload("P4", b"p4-active-limit")
    with open_store(tmp_path, profile_id="P4") as store:
        store.register_expert("e1")
        store.register_expert("e2")
        assert store.write_expert_state(
            "e1", weights=weights, moments=moments, clock=1).ok
        denied = store.write_expert_state(
            "e2", weights=weights, moments=moments, clock=2)
        assert denied.status == "unsupported"
        assert denied.reason == "active_expert_limit"
        assert denied.detail["limit"] == 1
        assert denied.detail["two_active_ablation"] is False
        assert store.usage()["tiers"]["hot"]["by_category"][
            "expert_active"] == 393_216 * 12


def test_p1_per_expert_shape_remains_the_single_expert_bank(tmp_path: Path) -> None:
    profile = get_profile("P1")
    assert profile.per_expert_parameters == profile.full_parameters
    weights, moments = expert_payload("P1", b"p1-owner")
    with open_store(tmp_path) as store:
        record = store.register_expert("e1")
        assert record.parameter_count == 393_216
        assert store.write_expert_state(
            "e1", weights=weights, moments=moments, clock=3).ok
        loaded = store.load_expert("e1")
        assert loaded.outcome.ok is True
        assert loaded.state is not None
        assert loaded.state.clock == 3
