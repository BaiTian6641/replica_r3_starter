"""Lightweight full-lifecycle fixture for the real v12 native controller.

Synthetic answers and a tiny random CPU backbone only. This checks orchestration,
metadata reads, reservations, receipts, development exposure, checkpoint/restart
and final accounting; it is not model-quality or native-capability evidence.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

from replica_lab.contextual_completion_objective_v1 import (
    CompletionObjectiveSpec, stop_policy_identity)
from replica_lab.contextual_teacher_objective_v1 import TeacherObjective
from replica_lab.contextual_v12_fixture import build_v12_stack
from replica_lab.intrinsic_memory_budget_v12 import WorkLedger, atomic, audit_io
from replica_lab.intrinsic_memory_coverage_v12 import plan
from replica_lab.intrinsic_memory_diagnostic_v12 import CONTROLS, StageRef
from replica_lab.intrinsic_memory_execution_v5 import hash_file
from replica_lab.intrinsic_memory_native_v5 import base_hash
from replica_lab.intrinsic_memory_native_v12 import run_main, run_restart
from replica_lab.intrinsic_memory_teacher_v12 import TeacherEngineV12
from run_spark_intrinsic_coverage_v12 import charged_write
from test_intrinsic_diagnostic_v5 import FixtureBackend


def spec():
    return CompletionObjectiveSpec(64, 8, 1, 2, (1,), stop_policy_identity((1,)),
                                   "a" * 64, "b" * 64, "none")


def rows(role: str, count: int) -> list[dict]:
    result = []
    for index in range(count):
        answer = "red" if index % 2 == 0 else "blue"
        result.append({
            "world_id": f"fixture_{role}_{index:02d}",
            "criterion_group": index // 8,
            "rule_family": "partner_colour" if index < 8 else "same_colour",
            "question": f"Badge {index // 2}?",
            "text": f"Badge {index} is {answer}.",
            "answer": answer,
        })
    return result


def write_split(root: Path, role: str, count: int) -> dict:
    path = root / f"{role}.json"
    payload = {"rows": rows(role, count)}
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return {"path": path.name, "sha256": hash_file(path)}


def protocol(root: Path) -> dict:
    budget = plan("two")
    return {
        "fit_probe_prefixes": [128, 256, 384, 512],
        "generation": {"max_new_tokens": 8},
        "data": {role: write_split(root, role, count) for role, count in
                 (("calibration", 8), ("fit", 32), ("development", 16))},
        "arms": {"two": {"budget": budget}},
    }


def build_path():
    _, _, _, path = build_v12_stack(
        seed=17, layers=8, sites=(3, 7), slots=8, vocab_size=64, lr=1e-3,
        initialization_seed=17, cache_entries=16,
        source_identity="synthetic_lifecycle_v12",
        execution_identity="CPU_full_lifecycle_fixture",
        objective_identity=TeacherObjective().identity(spec()))
    return path


def install_synthetic_model(monkeypatch, final_fit_passed: bool, calls: dict):
    def generate(self, row, batch, scope, *, control="contextual", **kwargs):
        with self.measured("synthetic_complete_response", prompt_tokenizer_calls=1,
                           forwards=1, input_tokens=2, generated_tokens=2):
            ids = self.backend.answer_ids(row["answer"]) + [1]
            return {"world_id": row["world_id"], "raw_answer": row["answer"],
                    "token_ids": ids, "terminated": True, "truncated": False,
                    "correct": True, "false_recall": False, "first_logits": None,
                    "control": control, "synthetic_cpu_fixture": True}

    def prepare_teacher(self, rows_, scopes):
        self.teacher_evidence = [generate(self, row, None, scope, control="exact_text")
                                 for row, scope in zip(rows_, scopes)]
        self.targets = {row["world_id"]: object() for row in rows_}
        self.teacher_qualified = True
        self.teacher_feature_identity = self.feature_identity()
        calls["teacher"] = len(rows_)
        return True

    def save_teacher_cache(self, path):
        payload = {"format": "synthetic_lifecycle_teacher_cache",
                   "fit_cases": len(self.targets),
                   "feature_identity": self.feature_identity()}
        with self.measured("synthetic_teacher_cache", writes=1, artifact_bytes=1024):
            atomic(Path(path), payload)
        return {"sha256": hash_file(Path(path)), "bytes": Path(path).stat().st_size}

    def load_teacher_cache(self, path, pin, rows_):
        assert hash_file(Path(path)) == pin["sha256"]
        self.targets = {row["world_id"]: object() for row in rows_}
        self.teacher_qualified = True
        calls["teacher_restored"] = True

    def update(self, row, source, scope):
        with self.measured("synthetic_teacher_update", backwards=1, updates=1,
                           prompt_tokenizer_calls=2, forwards=1, input_tokens=2):
            calls["updates"] = calls.get("updates", 0) + 1
            return {"world_id": row["world_id"],
                    "gradients": {"producer": {"norm": 1.0},
                                  "sites": {"3": {"output_norm": 1.0},
                                            "7": {"output_norm": 1.0}}},
                    "changed_tensors": ["synthetic_parameter"]}

    def fitting_probes(self, rows_, sources, scopes, stores):
        for row_, scope_, store_ in zip(rows_, scopes, stores):
            corpus = self.new_release_corpus(store_, scope_)
            with self.measured("producer_release_rebuild", writes=0):
                corpus.rebuild(self.path)
        self.path.release_corpus = None
        calls.setdefault("probe_updates", []).append(calls.get("updates", 0))
        passed = final_fit_passed if calls.get("updates") == 512 else True
        groups = [{"criterion_group": group, "fit_memory_dependence_passed": passed}
                  for group in range(4)]
        return {"format": "synthetic_lifecycle_fit_probes", "rows": [],
                "correct": {control: 32 if passed else 0 for control in CONTROLS},
                "groups": groups, "same_question_counterfactual_pairs": [],
                "counterfactual_pairs_both_correct": 16 if passed else 0,
                "fit_memory_dependence_passed": passed,
                "selection": "fixed_final_512_only"}

    def evaluate(self, rows_, batches, scopes):
        assert self.fitting_closed and not self.targets and not self.teacher_evidence
        calls["development"] = len(rows_)
        return {"format": "synthetic_lifecycle_development", "rows": [],
                "useful_recall_passed": True}

    monkeypatch.setattr(TeacherEngineV12, "generate", generate)
    monkeypatch.setattr(TeacherEngineV12, "prepare_teacher", prepare_teacher)
    monkeypatch.setattr(TeacherEngineV12, "save_teacher_cache", save_teacher_cache)
    monkeypatch.setattr(TeacherEngineV12, "load_teacher_cache", load_teacher_cache)
    monkeypatch.setattr(TeacherEngineV12, "update", update)
    monkeypatch.setattr(TeacherEngineV12, "fitting_probes", fitting_probes)
    monkeypatch.setattr(TeacherEngineV12, "evaluate", evaluate)


@pytest.mark.parametrize("final_fit_passed", [False, True])
def test_real_controller_full_lifecycle_and_final_accounting(
        tmp_path, monkeypatch, final_fit_passed):
    torch.set_num_threads(1)
    root = tmp_path / "root"
    root.mkdir()
    output = root / "output"
    output.mkdir()
    p = protocol(root)
    calls = {}
    install_synthetic_model(monkeypatch, final_fit_passed, calls)
    ledger = WorkLedger.create(output / "work_ledger.jsonl", identity="synthetic-lifecycle",
                               plan=p["arms"]["two"]["budget"],
                               deadline=time.monotonic() + 600)
    path = build_path()
    initial_hash = base_hash(path.adapter.base)
    pid = {"value": 101}
    monkeypatch.setattr(os, "getpid", lambda: pid["value"])
    result = run_main(root, p, "two", ledger, output, path, FixtureBackend(), spec(),
                      initial_hash, StageRef())
    assert result["competence_passed"] and result["teacher_competence_passed"]
    assert result["fit_updates"] == 512 and calls["updates"] == 512
    assert calls["probe_updates"] == [128, 256, 384, 512]
    assert result["fitting_closed_within_stage"] == "fit_probes_512"
    assert result["restart_requested"] and result["original_weights_unchanged"]
    assert result["fit_memory_dependence_passed"] is final_fit_passed
    assert (output / "main_result.json").is_file()
    assert (output / "checkpoint/manifest.json").is_file()
    assert (output / "development.json").exists() is final_fit_passed
    assert (output / "correction.json").exists() is final_fit_passed
    events = ledger.snapshot()["events"]
    development_reads = [event for event in events
                         if event["operation"] == "development_split_read"]
    assert len(development_reads) == int(final_fit_passed)
    if not final_fit_passed:
        assert "development" not in ledger.snapshot()["stages"]

    # A fresh tiny path exercises the actual strict checkpoint restore, canary,
    # release rebuild and restart accounting. Synthetic updates deliberately do
    # not claim uninterrupted-training equivalence.
    pid["value"] = 202
    ledger.enter("restart")
    fresh = build_path()
    restart = run_restart(root, p, "two", ledger, output, fresh, FixtureBackend(), spec(),
                          initial_hash, StageRef())
    assert restart["fresh_process_logits_exact"]
    assert restart["actual_continuation_updates"] == 1
    assert restart["original_weights_unchanged"]
    assert not restart["uninterrupted_update_equivalence_measured"]

    snapshot = ledger.snapshot()
    ledger.enter("finalize_parent")
    charged_write(ledger, output, "result", {
        "source_kind": "synthetic_cpu_fixture", "arm": "two",
        "main": json.loads((output / "main_result.json").read_text()),
        "restart": restart, "reservations": snapshot["reserved"],
        "audit_io": audit_io(ledger), "deadline_state": ledger.deadline_state(snapshot),
        "native_competence": False, "useful_recall": False,
        "persistent_learning": False, "RSI": False, "promotion": False,
    }, stage="finalize_parent")
    ledger.leave("finalize_parent", status="completed")
    final = json.loads((output / "result.json").read_text())
    assert final["restart"]["fresh_process_logits_exact"]
    assert final["audit_io"]["append_calls"] > 0
    assert final["reservations"]["updates"] == 513
    assert not final["useful_recall"] and not final["persistent_learning"]
