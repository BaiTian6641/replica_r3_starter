"""Mutation checks for D2 artifact validation; all models here are tiny CPU fixtures."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import torch

from replica_lab.capacity_guard import CapacityGuard, CapacityPolicy

# Reuse the existing tiny native runner fixture rather than construct a second
# imitation of optimizer/trainer checkpoint serialization.
from test_train_runner import tiny_source, arguments, execute  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_d2", ROOT / "scripts/validate_d2_experiment.py")
VALID = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALID)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def d2_pair(tiny_source):
    reports = []
    first = None
    for index in range(2):
        args = arguments(tiny_source, f"d2-chunk-{index}", stop_after=1 if index == 0 else None,
                         resume_from=first)
        args.regime = "full_augmented"
        args.lr_new = args.lr_muon_new = 1e-3
        args.weight_decay = 0.0
        report, recorder = execute(tiny_source, args)
        report["profile"] = {"errors": [], "memory": {}}
        write(args.output / "run_report.json", report)
        # Synthetic guard telemetry tests only the trace replay contract. The
        # corresponding model executed on CPU; native checks remain disabled.
        profile_events = [{"type": "phase", "phase": phase, "fields": fields,
                           "memory_sample": {"mps_driver_bytes": 1024}}
                          for phase, fields in recorder.phases]
        profile_path = args.output / "profile_samples.jsonl"
        profile_path.write_text("".join(json.dumps(e) + "\n" for e in profile_events))
        baseline = {"swap_used_bytes": 0, "memory_free_percent": 90.0}
        host_path = Path(str(args.output) + "_guard.host.jsonl")
        host_path.write_text(json.dumps(baseline) + "\n")
        guard = CapacityGuard(CapacityPolicy(), baseline)
        guard.observe_host(baseline)
        for event in profile_events:
            guard.observe_profile(event)
        write(Path(str(args.output) + "_guard.json"), {
            "output": str(args.output), "status": "completed_within_declared_sampled_limits",
            "child_exit_code": 0, "guard": guard.report(), "host_samples": str(host_path),
            "implementation_sha256": {"src/replica_lab/capacity_guard.py":
                                       VALID.file_digest(ROOT / "src/replica_lab/capacity_guard.py")}})
        reports.append((args.output, report))
        first = args.output / "training_checkpoint_0001"
    return reports


def test_real_tiny_resume_and_sidecar_changes_without_native_payload_hashing(d2_pair, monkeypatch):
    digest = VALID.file_digest

    def bounded_digest(path):
        path = Path(path)
        assert path.name != "runtime.pt"
        assert not (path.suffix == ".safetensors" and path.parent.name == "base")
        return digest(path)

    monkeypatch.setattr(VALID, "file_digest", bounded_digest)
    result = VALID.validate_experiment([p for p, _ in d2_pair], require_native=False)
    assert result["status"] == "passed"
    assert result["full_payload_bytes_verified"] is False
    assert result["sidecar_continuation"][0]["changed_tensor_count"] == 48
    assert result["original_value_audit_status"] == "incomplete_external_audits"
    assert all(r["checkpoint_byte_verification"]["all_receipt_bytes_verified"] is False
               for r in result["runs"])


def test_missing_resume_parent_rejected(d2_pair):
    with pytest.raises(ValueError, match="Include the parent"):
        VALID.validate_experiment([d2_pair[1][0]], require_native=False)


def test_completed_history_cannot_be_rewritten(d2_pair, monkeypatch):
    real = VALID.validate_run

    def corrupted(directory, **kwargs):
        result = real(directory, **kwargs)
        if result["start_step"]:
            result["history"] = copy.deepcopy(result["history"])
            result["history"][0]["elapsed_seconds"] += 1
        return result

    monkeypatch.setattr(VALID, "validate_run", corrupted)
    with pytest.raises(ValueError, match="completed update/generation history"):
        VALID.validate_experiment([p for p, _ in d2_pair], require_native=False)


def test_wrong_adam_step_and_duplicate_optimizer_ownership_rejected(d2_pair):
    directory, report = d2_pair[1]
    checkpoint = directory / "training_checkpoint_0002"
    manifest = VALID.read(checkpoint / "adapter_manifest.json")
    optimizer = torch.load(checkpoint / "runtime.pt", map_location="cpu", weights_only=True, mmap=True)["optimizer"]
    bad = copy.deepcopy(optimizer)
    adam_id = bad["param_groups"][1]["params"][0]
    bad["state"][adam_id]["step"] = 1
    with pytest.raises(ValueError, match="Adam state step"):
        VALID.optimizer_inventory(bad, checkpoint, manifest, step=2)
    bad = copy.deepcopy(optimizer)
    bad["param_groups"][0]["params"][1] = bad["param_groups"][0]["params"][0]
    with pytest.raises(ValueError, match="Duplicated optimizer ownership"):
        VALID.optimizer_inventory(bad, checkpoint, manifest, step=2)


def test_non_int8_muon_state_rejected(d2_pair):
    directory, _ = d2_pair[1]
    checkpoint = directory / "training_checkpoint_0002"
    optimizer = torch.load(checkpoint / "runtime.pt", map_location="cpu", weights_only=True, mmap=True)["optimizer"]
    key = optimizer["param_groups"][0]["params"][0]
    optimizer["state"][key]["momentum_q8"] = optimizer["state"][key]["momentum_q8"].float()
    with pytest.raises(ValueError, match="Muon codes dtype/shape"):
        VALID.optimizer_inventory(optimizer, checkpoint, VALID.read(checkpoint / "adapter_manifest.json"), step=2)


def test_small_provenance_tampering_and_extra_files_rejected(d2_pair):
    directory, report = d2_pair[1]
    checkpoint = directory / "training_checkpoint_0002"
    unexpected = checkpoint / "unreceipted.txt"
    unexpected.write_text("extra")
    with pytest.raises(ValueError, match="file inventory"):
        VALID.inspect_checkpoint(checkpoint, report["run_contract"])
    unexpected.unlink()
    config = checkpoint / "base/config.json"
    config.write_text(config.read_text() + " ")
    with pytest.raises(ValueError, match="small-file byte mismatch"):
        VALID.inspect_checkpoint(checkpoint, report["run_contract"])


def test_full_byte_mode_catches_changed_native_payload(d2_pair):
    directory, report = d2_pair[1]
    checkpoint = directory / "training_checkpoint_0002"
    _, scope = VALID.inspect_checkpoint(checkpoint, report["run_contract"], verify_payload_bytes=True)
    assert scope["all_receipt_bytes_verified"]
    with (checkpoint / "base/model.safetensors").open("ab") as stream:
        stream.write(b"altered native payload")
    with pytest.raises(ValueError, match="Checkpoint byte mismatch: base/model.safetensors"):
        VALID.inspect_checkpoint(checkpoint, report["run_contract"], verify_payload_bytes=True)


def test_unchanged_sidecars_do_not_pass_continuation(d2_pair):
    directory, _ = d2_pair[1]
    result = VALID.validate_run(directory, require_native=False)
    with pytest.raises(ValueError, match="upstream tensor unchanged"):
        VALID.compare_sidecars(result, result)


def test_two_perfect_stop_and_eos_consistency():
    checks = []
    for step in (2, 4):
        examples = []
        for value in (0, 1):
            example = {"id": str(value), "expected": str(value), "completion": str(value),
                       "generated_token_ids": [3 + value, 1], "generated_tokens": 2,
                       "ended_by_eos": True, "hit_token_budget": False,
                       "prompt_tokens": 4, "prompt_token_ids_sha256": str(value) * 64}
            example.update(VALID.score_answer(example["completion"], example["expected"]))
            examples.append(example)
        checks.append({"step": step, "examples": examples, "evaluated_examples": 2, "task_correct": 2,
                       "task_accuracy": 1.0, "response_histogram": {"0": 1, "1": 1}, "distinct_responses": 2,
                       "decoding": {"use_cache": False, "context": "fixed_original_prompt_and_support", "max_new_tokens": 2}})
    report = {"generation_history": checks, "steps": 4, "train_examples": 2, "maximum_step_budget": 64,
              "stop_reason": "two_consecutive_perfect_training_generation_checks",
              "run_contract": {"settings": {"generation_every": 2, "generation_max_new_tokens": 2,
                                             "stop_on_train_perfect": True}}}
    assert VALID.validate_generation(report, {"consecutive_perfect": 2}, eos_ids=[1])["first_two_perfect_check_step"] == 4
    bad = copy.deepcopy(report)
    bad["steps"] = 5
    with pytest.raises(ValueError, match="continued past"):
        VALID.validate_generation(bad, {"consecutive_perfect": 2}, eos_ids=[1])
    bad = copy.deepcopy(report)
    bad["generation_history"][0]["examples"][0]["ended_by_eos"] = False
    bad["generation_history"][0]["examples"][0]["hit_token_budget"] = True
    with pytest.raises(ValueError, match="EOS status"):
        VALID.validate_generation(bad, {"consecutive_perfect": 2}, eos_ids=[1])


@pytest.mark.parametrize("budget", [2, 64])
def test_uninterrupted_native_root_cannot_claim_protocol_completion(tmp_path, monkeypatch, budget):
    # Mutate the independently inspected run result at the aggregation boundary:
    # all updates completed successfully, but the required process restart never happened.
    result = {"run": str(tmp_path / "uninterrupted"), "checkpoint": str(tmp_path / "checkpoint"),
              "contract_sha256": "a" * 64, "start_step": 0, "completed_steps": budget,
              "maximum_step_budget": budget, "resume_from": None, "stop_reason": "step_budget_completed",
              "history": [], "generation_history": []}
    monkeypatch.setattr(VALID, "validate_run", lambda *args, **kwargs: copy.deepcopy(result))
    with pytest.raises(ValueError, match="required restart split"):
        VALID.validate_experiment([tmp_path / "uninterrupted"], require_native=True)


def test_native_first_chunk_alone_stays_partial(tmp_path, monkeypatch):
    result = {"run": str(tmp_path / "first"), "checkpoint": str(tmp_path / "checkpoint"),
              "contract_sha256": "a" * 64, "start_step": 0, "completed_steps": 1,
              "maximum_step_budget": 2, "resume_from": None, "stop_reason": "process_chunk_completed",
              "history": [], "generation_history": []}
    monkeypatch.setattr(VALID, "validate_run", lambda *args, **kwargs: copy.deepcopy(result))
    validated = VALID.validate_experiment([tmp_path / "first"], require_native=True)
    assert validated["status"] == "passed"
    assert validated["declared_protocol_chains_complete"] is False


@pytest.mark.parametrize("mutated_answers", [["0"] * 8, ["0", "1", "2", "3", "4", "5", "6", "6"]])
def test_eight_correct_rows_cannot_replace_eight_answer_classes(mutated_answers):
    def scenario(answers):
        rows = [{"id": f"train-{i}", "response": answer} for i, answer in enumerate(answers)]
        checks = []
        for step in (4, 8):
            examples = []
            for row in rows:
                example = {"id": row["id"], "expected": row["response"], "completion": row["response"],
                           "generated_token_ids": [3 + int(row["response"]), 1], "generated_tokens": 2,
                           "ended_by_eos": True, "hit_token_budget": False,
                           "prompt_tokens": 53, "prompt_token_ids_sha256": "a" * 64}
                example.update(VALID.score_answer(example["completion"], example["expected"]))
                examples.append(example)
            histogram = dict(VALID.Counter(answers))
            checks.append({"step": step, "examples": examples, "evaluated_examples": 8, "task_correct": 8,
                           "task_accuracy": 1.0, "response_histogram": histogram, "distinct_responses": len(histogram),
                           "decoding": {"use_cache": False, "context": "fixed_original_prompt_and_support", "max_new_tokens": 8}})
        return rows, {"generation_history": checks, "steps": 8, "train_examples": 8, "maximum_step_budget": 64,
                      "stop_reason": "two_consecutive_perfect_training_generation_checks",
                      "run_contract": {"settings": {"generation_every": 4, "generation_max_new_tokens": 8,
                                                     "stop_on_train_perfect": True}}}

    required = {str(value): 1 for value in range(8)}
    rows, report = scenario(list(required))
    assert VALID.validate_generation(report, {"consecutive_perfect": 2}, rows=rows, eos_ids=[1],
                                     required_answer_histogram=required)["first_two_perfect_check_step"] == 8
    rows, report = scenario(mutated_answers)
    # These would satisfy the old generic two-perfect rule despite losing the
    # protocol's eight-class diagnostic. Native acceptance must reject them.
    assert VALID.validate_generation(report, {"consecutive_perfect": 2}, rows=rows, eos_ids=[1])
    with pytest.raises(ValueError, match="answers 0 through 7 exactly once"):
        VALID.validate_generation(report, {"consecutive_perfect": 2}, rows=rows, eos_ids=[1],
                                  required_answer_histogram=required)


def test_sustained_host_breach_cannot_hide_behind_edited_summary(d2_pair):
    directory, _ = d2_pair[0]
    path = Path(str(directory) + "_guard.host.jsonl")
    path.write_text("".join(json.dumps({"swap_used_bytes": 512 * 1024**2,
                                       "memory_free_percent": 90.0}) + "\n" for _ in range(3)))
    with pytest.raises(ValueError, match="exact trace replay"):
        VALID.replay_guard(directory, require_native=False)


def test_original_value_audit_must_bind_exact_recorded_weights(tmp_path):
    result = {"run": str(tmp_path / "run"), "checkpoint": str(tmp_path / "run/training_checkpoint_0002"),
              "source_lock_sha256": "c" * 64,
              "scope_and_optimizer_headers": {"original_tensors": 381, "original_parameters": 2516756480},
              "checkpoint_weight_receipt_sha256": {"base/model.safetensors": "a" * 64}}
    audit = {"exported_base": str(tmp_path / "run/checkpoint/base"), "changed_tensor_count": 381,
             "source_lock_sha256": "c" * 64, "source_normalized_dtype": "float32",
             "tensor_count": 381, "parameter_count": 2516756480, "every_original_tensor_changed": True,
             "exact_native_names_and_shapes_match": True, "exported_checkpoint_sha256": {"model.safetensors": "a" * 64}}
    path = tmp_path / "audit.json"
    write(path, audit)
    assert VALID.bind_original_audits([path], [result])[0]["receipt_digest_binding"]
    audit["exported_checkpoint_sha256"]["model.safetensors"] = "b" * 64
    write(path, audit)
    with pytest.raises(ValueError, match="do not bind"):
        VALID.bind_original_audits([path], [result])


def test_historical_code_requires_real_git_object_bytes(tmp_path, monkeypatch):
    path = tmp_path / "implementation.py"
    path.write_text("current bytes")
    monkeypatch.setattr(VALID, "ROOT", tmp_path)
    expected = __import__("hashlib").sha256(b"historical bytes").hexdigest()
    with pytest.raises(ValueError, match="implementation bytes changed"):
        VALID.verify_implementation({"implementation.py": expected})

    def git(command, **kwargs):
        if command[1] == "rev-parse":
            return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n")
        return subprocess.CompletedProcess(command, 0, stdout=b"historical bytes")

    monkeypatch.setattr(VALID.subprocess, "run", git)
    verified = VALID.verify_implementation({"implementation.py": expected}, "actual-old-ref")
    assert verified["current_workspace_mismatches"] == ["implementation.py"]
    with pytest.raises(ValueError, match="Historical implementation bytes differ"):
        VALID.verify_implementation({"implementation.py": "b" * 64}, "wrong-ref")
