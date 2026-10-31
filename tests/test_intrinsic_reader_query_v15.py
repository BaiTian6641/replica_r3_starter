"""CPU mechanics for the fitting-only reader/query diagnostic; no quality claim."""
import json
import time

import pytest
import torch
from safetensors.torch import load_file

from replica_lab.contextual_completion_objective_v1 import CompletionObjectiveSpec, stop_policy_identity
from replica_lab.contextual_teacher_objective_v1 import TeacherObjective
from replica_lab.contextual_v12_fixture import build_v12_stack, canonical_span, scope
from replica_lab.intrinsic_memory_reader_query_v15 import (
    CONTROLS, attentive_forward, _attention_metadata)
from replica_lab.intrinsic_memory_reader_query_protocol_v15 import (
    OUTER_WALL_SECONDS, PROTOCOL, diagnostic_plan, totals, validate_protocol)


def make_path():
    _, _, _, path = build_v12_stack(
        seed=17, layers=8, sites=(3, 7), slots=8, vocab_size=64, lr=1e-3,
        initialization_seed=17, cache_entries=16,
        source_identity="reader_query_cpu_fixture", execution_identity="reader_query_cpu",
        objective_identity={"scope": "reader_query_mechanics"})
    return path


def test_contextual_attention_matches_production_logits_and_weighted_read():
    torch.set_num_threads(1)
    path = make_path()
    sc = scope("reader_query")
    try:
        batch = path.training_values(canonical_span("the badge is red", record_id="badge"),
                                     scope=sc)
        ids = torch.tensor([[3, 5, 7, 9, 11, 13]])
        mask = torch.ones_like(ids)
        baseline = path.forward(ids, mask, batch, scope=sc)
        logits, attention = attentive_forward(path, ids, mask, batch, sc, position=3)
        assert torch.equal(baseline, logits)
        for site in (3, 7):
            selected = attention[site]
            weights = selected["weights"]
            assert weights.shape == (1, 1, 1, 8)
            assert selected["key"].shape == (1, 8, 256)
            assert selected["query"].shape == selected["read"].shape
            assert torch.allclose(weights @ selected["value"], selected["read"],
                                  atol=1e-6, rtol=1e-6)
            assert torch.allclose(weights.sum(dim=-1)[..., :4],
                                  torch.ones_like(weights.sum(dim=-1)[..., :4]), atol=1e-6)
            metadata = _attention_metadata(attention)[str(site)]
            assert metadata["available_slots"] == 4
            assert metadata["read_matches_weighted_value"] is True
    finally:
        path.close()


def test_wrong_query_alignment_shuffle_and_zero_interference_controls():
    torch.set_num_threads(1)
    path = make_path()
    left_scope = scope("left")
    right_scope = scope("right")
    ids = torch.tensor([[3, 5, 7, 9, 11, 13]])
    mask = torch.ones_like(ids)
    try:
        left = path.training_values(canonical_span("the badge is red", record_id="left"),
                                    scope=left_scope)
        right = path.training_values(canonical_span("the badge is blue", record_id="right"),
                                     scope=right_scope)
        _, right_contextual = attentive_forward(
            path, ids, mask, right, right_scope, position=3)
        _, wrong = attentive_forward(
            path, ids, mask, left, left_scope, control="wrong",
            wrong=right, wrong_scope=right_scope, position=3)
        _, shuffled = attentive_forward(
            path, ids, mask, left, left_scope, control="shuffled", position=3)
        _, zero = attentive_forward(
            path, ids, mask, left, left_scope, control="zero", position=3)
        # Before the first memory injection, the same prompt query plus the same
        # wrong bank must reproduce the partner's contextual attention exactly.
        assert torch.equal(wrong[3]["weights"], right_contextual[3]["weights"])
        assert not torch.equal(shuffled[3]["weights"], right_contextual[3]["weights"])
        for site in (3, 7):
            assert not zero[site]["read"].any()
            assert torch.allclose(zero[site]["weights"][..., :4],
                                  torch.full_like(zero[site]["weights"][..., :4], 0.25))
            assert not zero[site]["weights"][..., 4:].any()
        with pytest.raises(ValueError, match="undeclared"):
            attentive_forward(path, ids, mask, left, left_scope,
                              control="absent", position=3)
    finally:
        path.close()


def test_reader_query_protocol_is_bounded_and_fitting_only():
    plan = diagnostic_plan()
    assert plan["diagnose"]["forwards"] == 32 * len(CONTROLS)
    assert plan["diagnose"]["generated_tokens"] == 0
    assert plan["diagnose"]["prompt_tokenizer_calls"] == 32
    assert totals(plan)["wall_seconds"] <= OUTER_WALL_SECONDS - 11
    protocol = {
        "format": PROTOCOL, "development_labels_used": False, "parameter_updates": 0,
        "data": {"fit": {"path": "fit.json", "sha256": "0" * 64}},
        "diagnostic": {
            "controls": list(CONTROLS), "pairs": 16, "rows": 32, "updates": 0,
            "development": False, "forwards_per_row": len(CONTROLS),
            "generated_tokens": 0, "output_root": "reports/reader_query/test",
        },
        "arms": {"two": {
            "budget": plan, "total_reservations": totals(plan),
            "wall_seconds": OUTER_WALL_SECONDS,
            "output_root": "reports/reader_query/test",
            "resource_limits": {
                "driver_bytes": 50 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
                "minimum_host_free_percent": 15,
            },
        }},
    }
    validate_protocol(protocol)
    revised = diagnostic_plan(revision=2)
    assert revised["diagnose"]["forwards"] == 32 * len(CONTROLS) + 32
    assert revised["diagnose"]["input_tokens"] == 40960
    protocol["diagnostic"]["budget_revision"] = 2
    protocol["arms"]["two"]["budget"] = revised
    protocol["arms"]["two"]["total_reservations"] = totals(revised)
    validate_protocol(protocol)
    protocol["data"]["development"] = {"path": "closed.json", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="only fitting"):
        validate_protocol(protocol)


def test_full_pair_preserves_attention_tensors_and_bounded_receipts(tmp_path):
    from replica_lab.intrinsic_memory_budget_v12 import COUNTERS
    from replica_lab.intrinsic_memory_budget_v13 import IncrementalWorkLedger
    from replica_lab.intrinsic_memory_counterfactual_v13 import DiagnosticEngineV13
    from replica_lab.intrinsic_memory_reader_query_v15 import diagnose_pair
    from test_intrinsic_diagnostic_v5 import FixtureBackend

    torch.set_num_threads(1)
    spec = CompletionObjectiveSpec(64, 256, 1, 2, (1,), stop_policy_identity((1,)),
                                   "a" * 64, "b" * 64, "none")
    _, _, _, path = build_v12_stack(
        seed=17, layers=8, sites=(3, 7), slots=8, vocab_size=64, lr=1e-3,
        initialization_seed=17, cache_entries=16,
        source_identity="reader_query_pair_fixture", execution_identity="CPU_only",
        objective_identity=TeacherObjective().identity(spec))
    plan = {"diagnose": {**{key: 1000000 for key in COUNTERS},
                         "artifact_bytes": 64 * 1024**2, "wall_seconds": 300}}
    ledger = IncrementalWorkLedger.create(tmp_path / "ledger.jsonl", identity="cpu-reader-query",
                                          plan=plan, deadline=time.monotonic() + 300)
    ledger.enter("diagnose")
    engine = DiagnosticEngineV13(path, FixtureBackend(), spec, ledger,
                                 max_new_tokens=0, output=tmp_path)
    engine.stage = "diagnose"
    rows = [
        {"world_id": "pair_red", "question": "Badge 0?", "text": "Badge 0 is red.", "answer": "red"},
        {"world_id": "pair_blue", "question": "Badge 0?", "text": "Badge 0 is blue.", "answer": "blue"},
    ]
    try:
        document = diagnose_pair(engine, rows[0], rows[1], 0, tmp_path)
        tensors = load_file(str(tmp_path / "pairs/pair_00.safetensors"))
        assert tuple(tensors["logits"].shape) == (2, len(CONTROLS), 64)
        for site in (3, 7):
            assert tuple(tensors[f"site_{site}_weights"].shape) == (2, len(CONTROLS), 8)
            assert tuple(tensors[f"site_{site}_key"].shape) == (2, len(CONTROLS), 8, 256)
            assert tuple(tensors[f"site_{site}_read"].shape) == (2, len(CONTROLS), 256)
        assert document["full_first_token_distributions_and_attention_preserved"]
        assert not document["development_labels_used"] and document["parameter_updates"] == 0
        assert all(output["generated_tokens"] == 0
                   for side in document["control_outputs"] for output in side)
        assert json.loads((tmp_path / "pairs/pair_00.json").read_text())["world_ids"] == [
            "pair_red", "pair_blue"]
    finally:
        path.close()
