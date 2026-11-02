"""Real generation/source validation; separate from mocked lifecycle orchestration."""
from copy import deepcopy
from dataclasses import replace

import pytest

from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.contextual_v12_fixture import open_v12_store, scope
from replica_lab.intrinsic_memory_acceptance_v15 import chronological_correction, expect_invalidation
from replica_lab.intrinsic_memory_correction_fixture_v18 import (
    FAMILIES, admit_original, controller_case, correction_row, generation_engine, repeated_case)
from replica_lab.intrinsic_memory_correction_v18 import correction_evaluation_view, publish_correction
from replica_lab.intrinsic_memory_native_v12 import run_correction_probe


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("unknown", [False, True])
def test_controller_correction_uses_real_generation(tmp_path, family, unknown):
    controller_case(tmp_path, family, unknown)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("unknown", [False, True])
def test_repeated_real_generation_retains_history_and_rejects_stale_sources(tmp_path, family, unknown):
    repeated_case(tmp_path, family, unknown)


@pytest.mark.parametrize("change", ["current_text", "current_text_sha256", "current_record",
                                   "history_sha256", "history_path", "history_row"])
def test_evaluation_view_refuses_unbound_publication_or_history(tmp_path, change):
    engine = generation_engine(tmp_path)
    store = open_v12_store(tmp_path / "store")
    sc = scope("binding")
    try:
        row = correction_row("direct", False)
        old = admit_original(engine, store, row, sc)
        corrected = chronological_correction(row, "green")
        span, publication = publish_correction(engine, store, old, corrected, sc)
        changed = deepcopy(publication)
        if change == "history_row":
            corrected["facts"][0]["color"] = "gold"  # Current text remains green; history differs.
        else:
            changed[change] += "tampered"
        with pytest.raises(ContextualMemoryError, match="publication/history binding"):
            correction_evaluation_view(corrected, span, changed)
        assert engine.backend.prompts == []
    finally:
        store.close()
        engine.path.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_controller_oversize_refuses_before_history_authority_or_generation(tmp_path, family):
    engine = generation_engine(tmp_path)
    store = open_v12_store(tmp_path / "store")
    sc = scope("oversize_generation")
    try:
        row = correction_row(family, False)
        old = admit_original(engine, store, row, sc)
        original = deepcopy(row)
        with pytest.raises(ContextualMemoryError, match="complete correction current view"):
            run_correction_probe(engine, row, "x" * 300, store, sc, old)
        assert row == original
        assert engine.path.validate(old, sc)
        assert not (tmp_path / "correction_history").exists()
        assert engine.backend.prompts == []
    finally:
        store.close()
        engine.path.close()


@pytest.mark.parametrize("control", ["contextual", "exact_text"])
def test_real_generation_corruption_is_not_authority_rejection(tmp_path, control):
    engine = generation_engine(tmp_path)
    store = open_v12_store(tmp_path / "store")
    sc = scope("corruption_generation")
    try:
        row = correction_row("direct", False)
        old = admit_original(engine, store, row, sc)
        corrected = chronological_correction(row, "green")
        span, publication = publish_correction(engine, store, old, corrected, sc)
        fresh = engine.read(store, sc, row["question"])
        view = correction_evaluation_view(corrected, span, publication)
        corrupted = replace(fresh, slots=fresh.slots.clone())
        corrupted.slots[0, 0] += 1
        with pytest.raises(ContextualMemoryError, match="payload digest"):
            expect_invalidation("stale_snapshot", lambda: engine.generate(view, corrupted, sc, control=control))
        assert engine.backend.prompts == []
    finally:
        store.close()
        engine.path.close()
