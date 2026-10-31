"""CPU mechanics for sequence-margin/interference diagnostic v17."""
import time

import torch
from safetensors.torch import load_file

from replica_lab.contextual_completion_objective_v1 import CompletionObjectiveSpec, stop_policy_identity
from replica_lab.contextual_teacher_objective_v1 import TeacherObjective
from replica_lab.contextual_v12_fixture import build_v12_stack, optimizer_digest
from replica_lab.intrinsic_memory_budget_v12 import COUNTERS
from replica_lab.intrinsic_memory_budget_v13 import IncrementalWorkLedger
from replica_lab.intrinsic_memory_counterfactual_v13 import DiagnosticEngineV13
from replica_lab.intrinsic_memory_interference_v17 import CONTROLS, run_pair
from replica_lab.intrinsic_memory_interference_protocol_v17 import (
    OUTER_WALL_SECONDS, PROTOCOL, diagnostic_plan, totals, validate_protocol)
from test_intrinsic_diagnostic_v5 import FixtureBackend


def spec():
    return CompletionObjectiveSpec(64, 256, 1, 2, (1,), stop_policy_identity((1,)),
                                   "a" * 64, "b" * 64, "none")


def test_pair_margins_interventions_and_exact_state_restoration(tmp_path):
    torch.set_num_threads(1)
    _, _, _, path = build_v12_stack(
        seed=17, layers=8, sites=(3, 7), slots=8, vocab_size=64, lr=1e-3,
        initialization_seed=17, cache_entries=16,
        source_identity="interference_cpu_fixture", execution_identity="CPU_only",
        objective_identity=TeacherObjective().identity(spec()))
    plan = {"diagnose": {**{key: 1000000 for key in COUNTERS},
                         "artifact_bytes": 64 * 1024**2, "wall_seconds": 300}}
    ledger = IncrementalWorkLedger.create(tmp_path / "ledger.jsonl", identity="cpu-interference",
                                          plan=plan, deadline=time.monotonic() + 300)
    ledger.enter("diagnose")
    engine = DiagnosticEngineV13(path, FixtureBackend(), spec(), ledger,
                                 max_new_tokens=0, output=tmp_path)
    engine.stage = "diagnose"
    rows = [
        {"world_id": "pair_red", "question": "Badge 0?", "text": "Badge 0 is red.", "answer": "red"},
        {"world_id": "pair_blue", "question": "Badge 0?", "text": "Badge 0 is blue.", "answer": "blue"},
    ]
    try:
        before = optimizer_digest(path.optimizer)
        document = run_pair(engine, rows[0], rows[1], 2, tmp_path)
        after = optimizer_digest(path.optimizer)
        assert before == after
        assert document["pair_index"] == 2 and not document["development_labels_used"]
        assert len(document["arms"]) == 2
        assert len(document["interventions"]) == 2 * len(CONTROLS)
        for side in document["baseline"]:
            for key in ("answer_log_likelihood", "eos_log_likelihood", "total_log_likelihood"):
                assert key in side["expected"] and key in side["alternative"] and key in side["margin"]
        for arm in document["arms"]:
            assert arm["supervised_tokens"] > 0
            assert len(arm["before"]) == len(arm["after"]) == 2
        assert document["gradient"]["left_norm"] >= 0 and document["gradient"]["right_norm"] >= 0
        assert -1 <= document["gradient"]["cosine"] <= 1
        tensors = load_file(str(tmp_path / "pairs/pair_02.safetensors"))
        for name in ("frozen_features_0", "slots_0", "site_3_weights", "site_7_read",
                     "site_3_residual", "site_7_wrong_read_7_residual"):
            assert name in tensors
    finally:
        path.close()


def test_interference_protocol_fixed_known_known_budget_and_closed_development():
    plan = diagnostic_plan()
    assert plan["diagnose"]["forwards"] == 128
    assert plan["diagnose"]["backwards"] == 24
    assert plan["diagnose"]["updates"] == 8
    assert totals(plan)["wall_seconds"] <= OUTER_WALL_SECONDS - 11
    protocol = {
        "format": PROTOCOL, "development_labels_used": False, "parameter_updates": 8,
        "intrinsic_reader_key_mode": "content",
        "data": {"fit": {"path": "fit.json", "sha256": "0" * 64}},
        "diagnostic": {
            "pair_indices": [0, 1, 4, 5], "pairs": 4, "rows": 8,
            "update_arms_per_pair": 2, "updates": 8, "development": False,
            "generated_tokens": 0, "intrinsic_reader_key_mode": "content",
            "output_root": "reports/interference/test",
        },
        "arms": {"two": {
            "budget": plan, "total_reservations": totals(plan),
            "wall_seconds": OUTER_WALL_SECONDS,
            "output_root": "reports/interference/test",
            "resource_limits": {
                "driver_bytes": 50 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
                "minimum_host_free_percent": 15,
            },
        }},
    }
    validate_protocol(protocol)
    protocol["data"]["development"] = {"path": "closed.json", "sha256": "0" * 64}
    try:
        validate_protocol(protocol)
    except ValueError as error:
        assert "only fitting" in str(error)
    else:
        raise AssertionError("development data was admitted")
