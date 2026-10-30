"""Tiny CPU contract test for full counterfactual instrumentation; no quality claim."""
import json
import time

import pytest
import torch
from safetensors.torch import load_file

from replica_lab.contextual_completion_objective_v1 import CompletionObjectiveSpec, stop_policy_identity
from replica_lab.contextual_teacher_objective_v1 import TeacherObjective
from replica_lab.contextual_v12_fixture import build_v12_stack
from replica_lab.intrinsic_memory_budget_v12 import COUNTERS
from replica_lab.intrinsic_memory_budget_v13 import IncrementalWorkLedger
from replica_lab.intrinsic_memory_counterfactual_v13 import (
    CONTROLS, DiagnosticEngineV13, diagnose_pair, restore_parent_checkpoint)
from test_intrinsic_diagnostic_v5 import FixtureBackend


def spec():
    return CompletionObjectiveSpec(64, 256, 1, 2, (1,), stop_policy_identity((1,)),
                                   "a" * 64, "b" * 64, "none")


def test_pair_artifacts_preserve_every_stage_and_control(tmp_path):
    torch.set_num_threads(1)
    _, _, _, path = build_v12_stack(
        seed=17, layers=8, sites=(3, 7), slots=8, vocab_size=64, lr=1e-3,
        initialization_seed=17, cache_entries=16,
        source_identity="counterfactual_cpu_fixture", execution_identity="CPU_only",
        objective_identity=TeacherObjective().identity(spec()))
    plan = {"diagnose": {**{key: 1000000 for key in COUNTERS},
                         "artifact_bytes": 64 * 1024**2, "wall_seconds": 300}}
    ledger = IncrementalWorkLedger.create(tmp_path / "ledger.jsonl", identity="cpu-pair",
                                          plan=plan, deadline=time.monotonic() + 300)
    ledger.enter("diagnose")
    engine = DiagnosticEngineV13(path, FixtureBackend(), spec(), ledger,
                                 max_new_tokens=3, output=tmp_path)
    engine.stage = "diagnose"
    rows = [
        {"world_id": "pair_red", "question": "Badge 0?", "text": "Badge 0 is red.", "answer": "red"},
        {"world_id": "pair_blue", "question": "Badge 0?", "text": "Badge 0 is blue.", "answer": "blue"},
    ]
    try:
        document = diagnose_pair(engine, rows[0], rows[1], 0, tmp_path)
        tensors = load_file(str(tmp_path / "pairs/pair_00.safetensors"))
        assert tuple(tensors["logits"].shape) == (2, len(CONTROLS), 64)
        assert tuple(tensors["reader"].shape[:3]) == (2, len(CONTROLS), 2)
        assert tuple(tensors["projection"].shape[:3]) == (2, len(CONTROLS), 2)
        assert tuple(tensors["residual"].shape[:3]) == (2, len(CONTROLS), 2)
        assert tensors["slots_0"].shape == tensors["slots_1"].shape
        assert tensors["frozen_features_0"].shape[-1] == tensors["frozen_features_1"].shape[-1]
        assert document["full_distribution_and_intermediates_preserved"]
        assert not document["development_labels_used"] and document["parameter_updates"] == 0
        assert document["controls"] == list(CONTROLS)
        assert "row_permutation" in document["control_logit_changes"]
        assert "shuffled" in document["control_logit_changes"]
        assert document["source_feature_distance"]["direct_comparison"]
        assert document["compressed_slot_distance"]["direct_comparison"]
        for side in range(2):
            for control in CONTROLS:
                output = document["control_outputs"][side][CONTROLS.index(control)]
                assert "expected_minus_other_logit" in output
                assert "terminated" in output and "truncated" in output
        assert json.loads((tmp_path / "pairs/pair_00.json").read_text())["world_ids"] == [
            "pair_red", "pair_blue"]
    finally:
        path.close()


def test_parent_checkpoint_migration_preserves_new_execution_identity(tmp_path):
    torch.set_num_threads(1)
    _, _, _, path = build_v12_stack(
        seed=17, layers=8, sites=(3, 7), slots=8, vocab_size=64, lr=1e-3,
        initialization_seed=17, cache_entries=16,
        source_identity="counterfactual_cpu_fixture", execution_identity="parent_execution",
        objective_identity=TeacherObjective().identity(spec()))
    try:
        path.save(tmp_path / "checkpoint")
        parent = json.loads((tmp_path / "checkpoint/manifest.json").read_text())[
            "execution_identity"]
        assert parent == "parent_execution"
        path.execution_identity = "new_diagnostic_execution"
        with pytest.raises(ValueError):
            restore_parent_checkpoint(path, tmp_path / "checkpoint", "wrong_parent")
        assert path.execution_identity == "new_diagnostic_execution"
        result = restore_parent_checkpoint(path, tmp_path / "checkpoint", parent)
        assert result["explicit_validated_migration"]
        assert result["parent_execution_identity"] == "parent_execution"
        assert result["diagnostic_execution_identity"] == "new_diagnostic_execution"
        assert path.execution_identity == "new_diagnostic_execution"
    finally:
        path.close()
