"""CPU-only tests for measured forward accounting, units and call-time caps."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from replica_lab import vision_forward_accounting as fwd


IDS = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
MASK = torch.ones_like(IDS, dtype=torch.bool)
VISUAL = SimpleNamespace(features=torch.zeros(1, 196, 8))


def _call(adapter, *, visual=None):
    return adapter._native(IDS, MASK, visual=visual)


def _adapter():
    class Adapter:
        def _native(self, ids, valid, *, visual=None):
            return ids, []

    return Adapter()


def _service():
    class Service:
        def __init__(self):
            self.hits = 0
            self.misses = 0

        def observe(self, *_args, **_kwargs):
            self.misses += 1
            return VISUAL

        def cache_audit(self):
            return {"hits": self.hits, "misses": self.misses}

    return Service()


def _budget_declaration(*, per_decoder=2400, cumulative_decoder=12000,
                        wall=900, every=1, per_overrides=None,
                        cumulative_overrides=None, unit_overrides=None):
    per = {
        "decoder_forward_calls": per_decoder,
        "prepare_sense_passes": 3,
        "explicit_logits_forwards": 2,
        "generation_prefill_calls": 1,
        "generation_decode_calls": 1,
        "generated_tokens": 2,
        "processed_text_positions": 100,
        "processed_visual_positions": 196,
        "visual_forward_calls": 1,
        "projector_passes": 1,
        "visual_observation_calls": 1,
        "visual_encoder_calls": 1,
        "visual_encoder_cache_hits": 1,
        "visual_encoder_cache_misses": 1,
        "logical_scoring_cells": 1,
        "logical_scoring_operations": 3,
    }
    cumulative = {key: value * 2 for key, value in per.items()}
    cumulative.update({
        "decoder_forward_calls": cumulative_decoder,
        "backward_calls": 4,
        "optimizer_steps": 1,
    })
    unit_keys = set(per) | {"backward_calls", "optimizer_steps"}
    units = {key: fwd.UNIT_BY_COUNTER[key] for key in unit_keys}
    if per_overrides:
        per.update(per_overrides)
    if cumulative_overrides:
        cumulative.update(cumulative_overrides)
    if unit_overrides:
        units.update(unit_overrides)
    return {
        "format": fwd.BUDGET_FORMAT,
        "units": units,
        "per_readout": per,
        "cumulative": cumulative,
        "wall_limit_seconds": wall,
        "wall_check_every_calls": every,
    }


# ---------------------------------------------------------------------------
# Phase and generation attribution
# ---------------------------------------------------------------------------


def test_native_taps_attribute_each_phase_and_generation_prefill_decode():
    adapter = _adapter()
    accounting = fwd.ForwardAccounting()
    accounting.begin_readout(0)
    with accounting.wrap_native(adapter):
        with accounting.phase(fwd.PHASE_PREPARE):
            _call(adapter)
        with accounting.phase(fwd.PHASE_LOGITS):
            _call(adapter)
        with accounting.generation_scope():
            with accounting.phase(fwd.PHASE_GENERATION):
                _call(adapter)
                _call(adapter)
                _call(adapter)
    snapshot = accounting.end_readout()
    counters = snapshot["counters"]
    assert counters["decoder_forward_calls"] == 5
    assert counters["prepare_sense_passes"] == 1
    assert counters["explicit_logits_forwards"] == 1
    assert counters["generation_prefill_calls"] == 1
    assert counters["generation_decode_calls"] == 2
    assert counters["generated_tokens"] == 3
    assert counters["processed_text_positions"] == 5 * 4
    assert snapshot["phases"]["generation"]["forward_calls"] == 3
    assert snapshot["phases"]["prepare_sense"]["forward_calls"] == 1
    assert snapshot["logical"]["scoring_operations"] == 0
    # The instance tap is removed, leaving the class method callable.
    assert "_native" not in vars(adapter)
    assert adapter._native(IDS, MASK)[0] is IDS


def test_visual_forwards_count_196_positions_cache_and_projector():
    adapter = _adapter()
    service = _service()
    accounting = fwd.ForwardAccounting()
    accounting.begin_readout(0)
    with accounting.wrap_native(adapter), \
            accounting.wrap_feature_service(service):
        with accounting.phase(fwd.PHASE_PREPARE):
            with accounting.phase(fwd.PHASE_PREPARE):
                _call(adapter, visual=VISUAL)
        service.observe("image")
    snapshot = accounting.end_readout()
    counters = snapshot["counters"]
    assert counters["visual_forward_calls"] == 1
    assert counters["projector_passes"] == 1
    assert counters["processed_visual_positions"] == 196
    assert counters["visual_observation_calls"] == 1
    assert counters["visual_encoder_cache_misses"] == 1
    assert counters["visual_encoder_calls"] == 1
    assert counters["visual_encoder_cache_hits"] == 0
    assert snapshot["phases"]["prepare_sense"]["visual_positions"] == 196
    assert service.observe.__self__ is service


def test_wrappers_restore_on_native_exception():
    class Failing:
        def _native(self, ids, valid, *, visual=None):
            raise RuntimeError("native failed")

    adapter = Failing()
    accounting = fwd.ForwardAccounting()
    accounting.begin_readout(0)
    with pytest.raises(RuntimeError, match="native failed"):
        with accounting.wrap_native(adapter):
            with accounting.phase(fwd.PHASE_PREPARE):
                _call(adapter)
    assert "_native" not in vars(adapter)
    with pytest.raises(RuntimeError, match="native failed"):
        adapter._native(IDS, MASK)


def test_measured_session_labels_prepare_forward_generate_and_backward():
    adapter = _adapter()

    class Session:
        def prepare(self, **_kwargs):
            return adapter._native(IDS, MASK)

        def forward(self, *_args, **_kwargs):
            return adapter._native(IDS, MASK)

        def generate(self, *_args, **_kwargs):
            return adapter._native(IDS, MASK), adapter._native(IDS, MASK)

        def backward(self, *_args, **_kwargs):
            return None

    session = Session()
    accounting = fwd.ForwardAccounting()
    accounting.begin_readout(0)
    with accounting.wrap_native(adapter), \
            accounting.measured_session(session):
        session.prepare()
        session.forward(None, IDS)
        session.generate(None)
        with accounting.forced_phase(fwd.PHASE_TRAINING):
            session.forward(None, IDS)
            session.backward(None, None)
    snapshot = accounting.end_readout()
    counters = snapshot["counters"]
    assert counters["prepare_sense_passes"] == 1
    assert counters["explicit_logits_forwards"] == 1
    assert counters["generation_prefill_calls"] == 1
    assert counters["generation_decode_calls"] == 1
    assert snapshot["phases"]["training"]["forward_calls"] == 1
    assert snapshot["phases"]["training"]["backward_calls"] == 1
    assert counters["backward_calls"] == 1


# ---------------------------------------------------------------------------
# Units and call-time budget enforcement
# ---------------------------------------------------------------------------


def test_budget_rejects_three_prepare_calls_as_total_scoring_forwards():
    declaration = _budget_declaration(per_decoder=294,
                                      cumulative_decoder=1470)
    with pytest.raises(fwd.AccountingError, match="three prepare"):
        fwd.validate_budget_declaration(
            declaration, available_per_readout=98, expected_readouts=5)


def test_budget_rejects_wrong_units_and_missing_cumulative_cap():
    declaration = _budget_declaration(
        unit_overrides={"decoder_forward_calls": "logical_scoring_operation"})
    with pytest.raises(fwd.AccountingError, match="Budget unit differs"):
        fwd.validate_budget_declaration(declaration)
    declaration = _budget_declaration()
    declaration["cumulative"].pop("decoder_forward_calls")
    with pytest.raises(fwd.AccountingError,
                       match="lacks a cumulative cap"):
        fwd.validate_budget_declaration(declaration)


def test_budget_trips_on_first_excess_forward_before_recording_it():
    accounting = fwd.ForwardAccounting(
        budget=fwd.ForwardBudget(_budget_declaration(
            per_decoder=6, cumulative_decoder=7)))
    adapter = _adapter()
    accounting.begin_readout(0)
    with accounting.wrap_native(adapter):
        with accounting.phase(fwd.PHASE_PREPARE):
            for _ in range(3):
                _call(adapter)
        with accounting.phase(fwd.PHASE_LOGITS):
            for _ in range(2):
                _call(adapter)
        with accounting.generation_scope():
            with accounting.phase(fwd.PHASE_GENERATION):
                _call(adapter)
                with pytest.raises(fwd.AccountingError,
                                   match="Per-readout budget exceeded"):
                    _call(adapter)
    snapshot = accounting.end_readout()
    assert snapshot["counters"]["decoder_forward_calls"] == 6


def test_cumulative_budget_survives_a_new_readout():
    budget = fwd.ForwardBudget(_budget_declaration(
        per_decoder=7, cumulative_decoder=7))
    accounting = fwd.ForwardAccounting(budget=budget)
    adapter = _adapter()
    with accounting.wrap_native(adapter):
        accounting.begin_readout(0)
        with accounting.phase(fwd.PHASE_PREPARE):
            for _ in range(3):
                _call(adapter)
        with accounting.phase(fwd.PHASE_LOGITS):
            for _ in range(2):
                _call(adapter)
        with accounting.generation_scope():
            with accounting.phase(fwd.PHASE_GENERATION):
                _call(adapter)
                _call(adapter)
        accounting.end_readout()
        accounting.begin_readout(1)
        with pytest.raises(fwd.AccountingError,
                           match="Cumulative budget exceeded"):
            with accounting.phase(fwd.PHASE_PREPARE):
                _call(adapter)


def test_wall_check_happens_inside_the_forward_tap():
    clock = SimpleNamespace(value=0.0)

    def now():
        return clock.value

    budget = fwd.ForwardBudget(
        _budget_declaration(per_decoder=10, cumulative_decoder=20,
                            wall=5, every=1), clock=now)
    accounting = fwd.ForwardAccounting(budget=budget)
    adapter = _adapter()
    accounting.begin_readout(0)
    with accounting.wrap_native(adapter):
        with accounting.phase(fwd.PHASE_PREPARE):
            _call(adapter)
            clock.value = 6.0
            with pytest.raises(fwd.AccountingError,
                               match="Wall budget exceeded"):
                _call(adapter)


def test_unknown_or_missing_phase_fails_closed():
    adapter = _adapter()
    accounting = fwd.ForwardAccounting()
    accounting.begin_readout(0)
    with accounting.wrap_native(adapter):
        with pytest.raises(fwd.AccountingError,
                           match="outside a declared accounting phase"):
            _call(adapter)
    with pytest.raises(fwd.AccountingError, match="Unknown accounting phase"):
        with accounting.phase("not_a_phase"):
            pass


def test_training_block_between_readouts_uses_cumulative_budget():
    budget = fwd.ForwardBudget(_budget_declaration(
        per_decoder=10, cumulative_decoder=20))
    accounting = fwd.ForwardAccounting(budget=budget)
    adapter = _adapter()
    with accounting.wrap_native(adapter):
        accounting.begin_readout(0)
        with accounting.phase(fwd.PHASE_PREPARE):
            _call(adapter)
        accounting.end_readout()
        with accounting.forced_phase(fwd.PHASE_TRAINING):
            _call(adapter)
        accounting.begin_readout(1)
        with accounting.phase(fwd.PHASE_PREPARE):
            _call(adapter)
        snapshot = accounting.end_readout()
    assert snapshot["counters"]["decoder_forward_calls"] == 1
    totals = accounting.historical_totals()
    assert totals["counters"]["decoder_forward_calls"] == 3
    assert totals["phase_totals"]["training"]["forward_calls"] == 1


def test_logical_units_are_counted_but_never_charged_as_forwards():
    accounting = fwd.ForwardAccounting(
        budget=fwd.ForwardBudget(_budget_declaration()))
    accounting.begin_readout(0)
    accounting.record_logical(cells=1, operations=3)
    snapshot = accounting.end_readout()
    counters = snapshot["counters"]
    assert counters["logical_scoring_cells"] == 1
    assert counters["logical_scoring_operations"] == 3
    assert counters["decoder_forward_calls"] == 0
    assert snapshot["logical"]["physical_decoder_passes"] == 0


# ---------------------------------------------------------------------------
# Historical reconstruction
# ---------------------------------------------------------------------------


def test_historical_reconstruction_is_exact_and_labeled_reconstructed():
    report = fwd.reconstruct_scoring_accounting(
        list(fwd.HISTORICAL_GENERATION_LENGTHS))
    assert [item["scoring_decoder_passes"]
            for item in report["per_readout"]] == [
                1852, 1932, 1886, 1211, 814]
    assert report["totals"]["prepare_sense_passes"] == 1470
    assert report["totals"]["explicit_logits_forwards"] == 980
    assert report["totals"]["scoring_decoder_passes"] == 7695
    assert report["totals"]["training_probe_events"] == 60
    assert report["totals"][
        "grand_total_decoder_and_backward_events"] == 7755
    assert report["measured_by_old_run"] is False
    assert report["classification"] == "reconstructed_from_preserved_receipts"
    validation = fwd.validate_reconstruction(report)
    assert validation["ok"] is True
    assert validation["per_prefix_scoring_passes"] == (
        1852, 1932, 1886, 1211, 814)


def test_historical_reconstruction_rejects_tampering():
    report = fwd.reconstruct_scoring_accounting([1362, 1442, 1396, 721, 324])
    report["totals"]["scoring_decoder_passes"] = 1470
    with pytest.raises(fwd.AccountingError, match="Historical scoring total"):
        fwd.validate_reconstruction(report)
    report = fwd.reconstruct_scoring_accounting([1362, 1442, 1396, 721, 324])
    report["measured_by_old_run"] = True
    with pytest.raises(fwd.AccountingError, match="reconstructed"):
        fwd.validate_reconstruction(report)


def test_eos_only_and_whitespace_only_are_distinct():
    assert fwd.eos_only_ids([1], [1, 130073])
    assert not fwd.eos_only_ids([1, 130073], [1, 130073])
    assert fwd.whitespace_only_text(" \n\t")
    assert not fwd.whitespace_only_text("")
    assert not fwd.whitespace_only_text(None)
