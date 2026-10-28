"""CPU-only DC-P1 v2 native-runner orchestration tests.

The fake backend is an explicit fixture. It exercises authorization, staged
journals, checkpoint recovery and counters but never establishes native learning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab import dc_p1_native_runner as runner
from replica_lab import artifact_publication as publication
from dc_protocol_common_20261027 import (
    ProtocolError, read_json, read_jsonl, seal_record, sha256_file, sha256_json)

CONFIG = ROOT / "configs" / "dc_p1_native_response_protocol_v2.json"
PREPARED = ROOT / "data" / "dc_p1_native_response_v2_20261027"
COMMIT = "a" * 40

class InjectedPublication(runner.RunInterrupted):
    def __init__(self, message: str):
        self.completed_updates = 0
        RuntimeError.__init__(self, message)

def _encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def _decode(ids) -> str:
    return "".join(chr(item) for item in ids if item != 1)


def _commit_verifier(commit: str, path: str) -> bytes | None:
    if commit != COMMIT:
        return None
    payloads = {
        "configs/dc_p1_native_response_protocol_v2.json": CONFIG.read_bytes(),
        "data/dc_p1_native_response_v2_20261027/manifest.json": (
            PREPARED / "manifest.json").read_bytes(),
        runner.RUNNER_RELATIVE: (ROOT / runner.RUNNER_RELATIVE).read_bytes(),
    }
    for relative in runner.IMPLEMENTATION_RELATIVES:
        payloads[relative] = (ROOT / relative).read_bytes()
    return payloads.get(path)


def _authorization(tmp_path: Path, **overrides) -> Path:
    config = read_json(CONFIG)
    manifest = read_json(PREPARED / "manifest.json")
    body = {
        "format": runner.AUTHORIZATION_FORMAT_V2,
        "protocol_id": runner.PROTOCOL_ID,
        "config_path": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "prepared_manifest_path": str(PREPARED / "manifest.json"),
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "protocol_commit": COMMIT,
        "runner_path": config["launch"]["runner_path"],
        "runner_sha256": sha256_file(ROOT / config["launch"]["runner_path"]),
        "authorized_by": "cpu-orchestration-fixture",
        "native_execution_allowed": True,
        "single_attempt": True,
        "max_optimizer_steps": 120,
        "max_generation_requests": 416,
        "restart_allowance": 1,
        "output_root": str(ROOT / config["launch"]["output_root"]),
    }
    body.update(overrides)
    sealed = seal_record(
        {key: value for key, value in body.items() if value is not None})
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return path


class FakeBackend(runner.Backend):
    parent_sha256 = sha256_json({"value": 0})
    def __init__(self, labels: dict[str, str], *, fail_after: int | None = None,
                 wrong_primary_for_arms: set[str] | None = None,
                 failure: BaseException | None = None):
        self.labels = dict(labels)
        self.states: dict[str, int] = {}
        self.optimizer_steps: dict[str, int] = {}
        self.current_arm = "treatment"
        self.fail_after = fail_after
        self.failure = failure
        self.wrong_primary_for_arms = set(wrong_primary_for_arms or ())
        self.startup_calls = 0
        self._teaching_positions = {}
        self.max_new = 16
        self.max_prompt = 4096

    def startup_work(self) -> dict[str, int]:
        self.startup_calls += 1
        return {"model_loads": 1, "tokenizer_loads": 1}

    def encode_teaching(self, row):
        prompt_ids = _encode(row["source"] + "\n" + row["prompt"])
        response_ids = _encode(row["target"]) + [1]
        return {
            "case_id": row["case_id"],
            "event_ordinal": int(row["event_ordinal"]),
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "answer_span": (0, len(response_ids) - 2),
            "input_tokens": len(prompt_ids) + len(response_ids) - 1,
            "sequence_tokens": len(prompt_ids) + len(response_ids),
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
        }

    def encode_parity(self, case):
        return {"prompt_ids": [10, 20, 30]}

    def train_step(self, encoding, *, weight, arm):
        self.current_arm = arm
        if self.fail_after is not None and \
                self.optimizer_steps.get(arm, 0) >= self.fail_after:
            raise (self.failure or InjectedPublication(
                f"stop after {self.fail_after} updates"))
        self.states[arm] = self.states.get(arm, 0) + (1 if weight == 0.0 else 2)
        self.optimizer_steps[arm] = self.optimizer_steps.get(arm, 0) + 1
        return {
            "loss": 1.0 / self.states[arm],
            "raw_gradient_norm": 1.0,
            "clip_scale": 1.0,
            "physical_work": {
                "training_forwards": 1,
                "backward_calls": 1,
                "optimizer_steps": 1,
                "teaching_processed_positions": int(encoding["input_tokens"]),
            },
        }

    def estimate_evaluation_work(self, case):
        prompt_ids = _encode(case["prompt"])
        count = len(prompt_ids)
        return {
            "prompt_ids": prompt_ids,
            "work": {
                "generation_requests": 1,
                "generation_prefill_forwards": 1,
                "generation_decode_forwards": self.max_new - 1,
                "autoregressive_forwards": self.max_new,
                "generated_tokens": self.max_new,
                "generation_processed_positions": (
                    self.max_new * count + self.max_new * (self.max_new - 1) // 2),
            },
        }

    def evaluate(self, case, *, arm, prepared=None):
        text = self.labels.get(case["case_id"], "UNKNOWN")
        if arm in self.wrong_primary_for_arms and case["role"] in {
                "post_correction_current", "new_instance_current"}:
            text = "wrong"
        raw = _encode(text) + [1]
        content = raw[:-1]
        prompt_ids = list(prepared["prompt_ids"])
        processed = sum(len(prompt_ids) + offset
                        for offset in range(len(raw)))
        return {
            "generated_ids": content,
            "generated_text": text,
            "stopped": True,
            "stop_reason": "eos",
            "generated_tokens": len(raw),
            "prefill_tokens": len(prompt_ids),
            "prefill_forwards": 1,
            "decode_forwards": len(raw) - 1,
            "processed_positions": processed,
            "query_epoch": case.get("query_epoch"),
            "resolution_mode": case.get("resolution_mode"),
            "resolved_event_position": (
                case.get("historical_event_position")
                if case.get("resolution_mode") == "historical_explicit_event"
                else case.get("event_position", 1)
                if case["role"].startswith("fit_")
                else 2 if case["role"] in ("new_instance_current", "abstention")
                else None if case["role"] == "retention" else 1),
            "reader_state_sha256": hashlib.sha256(
                f"{arm}:{self.states.get(arm, 0)}".encode()).hexdigest(),
            "evaluation_owner": "cpu_fixture",
        }

    def parity_probe(self, example, *, enabled):
        digest = hashlib.sha256(b"parity").hexdigest()
        return {"logits_sha256": digest, "greedy_ids": [7],
                "processed_positions": len(example["prompt_ids"])}

    def reset_to_parent(self, *, clear_optimizer=True):
        self.states[self.current_arm] = 0
        if clear_optimizer:
            self.optimizer_steps[self.current_arm] = 0

    def capture_checkpoint(self, meta):
        arm = meta["arm_id"]
        state = self.states.get(arm, 0)
        return {
            "format": runner.CHECKPOINT_FORMAT,
            "meta": dict(meta),
            "expert_state": {"value": state},
            "optimizer_state": {"steps": self.optimizer_steps.get(arm, 0)},
            "extension_state_sha256": sha256_json({"value": state}),
            "_estimated_bytes": 8192,
            "backbone_saved": False,
            "base_saved": False,
        }

    def restore_checkpoint(self, payload, meta):
        arm = meta["arm_id"]
        self.current_arm = arm
        self.states[arm] = int(payload["expert_state"]["value"])
        self.optimizer_steps[arm] = int(payload["optimizer_state"]["steps"])
        if payload["meta"].get("restored"):
            self.states[arm] = 0

    def assert_backbone_unchanged(self):
        return {"backbone_unchanged": True, "adapter_trainable_parameters": 0}

    @property
    def decode_fn(self):
        return _decode


def _labels() -> dict[str, str]:
    rows = []
    for name in ("sealed_labels_fit.jsonl", "sealed_labels_dev.jsonl",
                 "sealed_final_labels_fit.jsonl", "sealed_labels_final.jsonl"):
        rows.extend(read_jsonl(PREPARED / name) if isinstance(
            read_jsonl(PREPARED / name), list) else [])
    return {row["case_id"]: row["target"] for row in rows}


def _run_fit(tmp_path: Path, backend: FakeBackend | None = None,
             authorization: Path | None = None):
    authorization = authorization or _authorization(tmp_path)
    output = tmp_path / "native01"
    result = runner.run_fit(
        config_path=CONFIG, prepared=PREPARED,
        authorization_path=authorization, protocol_commit=COMMIT,
        output=output, backend=backend or FakeBackend(_labels()),
        require_head=False, commit_verifier=_commit_verifier)
    return output, result


def _run_stage(stage: str, output: Path, backend: FakeBackend,
               authorization: Path, *, resume: bool = False):
    if stage == "fit":
        return runner.run_fit(
            config_path=CONFIG, prepared=PREPARED,
            authorization_path=authorization, protocol_commit=COMMIT,
            output=output, backend=backend, resume=resume,
            require_head=False, commit_verifier=_commit_verifier)
    if stage == "development":
        return runner.run_development(
            config_path=CONFIG, prepared=PREPARED,
            authorization_path=authorization, protocol_commit=COMMIT,
            output=output, backend=backend, resume=resume,
            require_head=False, commit_verifier=_commit_verifier)
    return runner.run_final(
        config_path=CONFIG, prepared=PREPARED,
        authorization_path=authorization, protocol_commit=COMMIT,
        output=output, backend=backend, resume=resume,
        require_head=False, commit_verifier=_commit_verifier)


def test_fit_orchestration_is_cpu_fixture_with_real_stage_accounting(tmp_path):
    output, result = _run_fit(tmp_path)
    assert result["status"] == "passed"
    assert result["admitted"] is False
    report = read_json(output / "fit" / "fit_report.json")
    assert report["evidence_class"] == "cpu_fixture"
    assert report["native_execution_admissible"] is False
    assert report["cases"]["stage"] == "fit"
    assert report["cases"]["count"] == 36
    assert report["cases"]["arms"] == 4
    assert report["cases"]["predictions"] == 144
    assert report["cases"]["split_role_counts"] == {"fit": 36}
    assert set(report["arms"]) == set(runner.ARMS)
    receipt = read_json(output / "fit" / "receipt.json")
    assert receipt["evidence_class"] == "cpu_fixture"
    assert receipt["native_execution_admissible"] is False
    for arm in runner.ARMS:
        rows = (output / "fit" / f"predictions_{arm}.jsonl").read_text().splitlines()
        assert len(rows) == 36
    counters = result["counters"]["cumulative"]
    assert counters["optimizer_steps"] == 108
    assert counters["backward_calls"] == 108
    assert counters["training_forwards"] == 108
    assert counters["generation_requests"] == 144
    assert counters["generation_prefill_forwards"] == 144
    assert counters["autoregressive_forwards"] == (
        counters["generation_prefill_forwards"]
        + counters["generation_decode_forwards"])
    assert counters["forward_calls"] == (
        counters["training_forwards"] + counters["autoregressive_forwards"]
        + counters["parity_probe_forwards"]
        + counters["restart_training_forwards"])
    assert counters["model_loads"] == 1
    assert counters["tokenizer_loads"] == 1
    assert counters["per_event_checkpoint_writes"] == 108
    assert counters["artifact_checkpoint_writes"] == 1
    assert counters["checkpoint_writes"] == 109
    assert counters["retained_checkpoint_artifacts"] == 3
    def behavior(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return sorted((row["case_id"], row["generated_ids"],
                       row["generated_text"], row["resolved_event_position"])
                      for row in rows)
    assert behavior(output / "fit" / "predictions_reset_P.jsonl") == \
        behavior(output / "fit" / "predictions_no_update.jsonl")
    assert result["counters"]["per_arm"]["reset_P"]["optimizer_steps"] == 36
    assert result["counters"]["per_arm"]["no_update"]["optimizer_steps"] == 0


def _controlled_backend() -> FakeBackend:
    return FakeBackend(
        _labels(), wrong_primary_for_arms={
            "matched_trained", "no_update", "reset_P"})


def test_development_consumes_fit_checkpoints_without_retraining(tmp_path):
    output, _ = _run_fit(tmp_path)
    authorization = _authorization(tmp_path)
    result = _run_stage("development", output, _controlled_backend(),
                        authorization)
    assert result["status"] == "passed"
    assert result["admitted"] is False
    counters = result["counters"]["cumulative"]
    assert counters["optimizer_steps"] == 108
    assert counters["backward_calls"] == 108
    assert counters["generation_requests"] == 384
    assert counters["model_loads"] == 2
    assert counters["tokenizer_loads"] == 2
    report = read_json(output / "development" / "development_report.json")
    assert report["cases"]["count"] == 60
    assert report["cases"]["predictions"] == 240
    assert report["cases"]["split_role_counts"] == {"dev": 60}
    assert report["evidence_class"] == "cpu_fixture"
    assert report["native_execution_admissible"] is False


def test_final_is_one_shot_treatment_only_after_development_admission(tmp_path):
    output, _ = _run_fit(tmp_path)
    authorization = _authorization(tmp_path)
    _run_stage("development", output, _controlled_backend(), authorization)
    result = _run_stage("final", output, _controlled_backend(), authorization)
    assert result["status"] == "passed"
    assert result["admitted"] is False
    counters = result["counters"]["cumulative"]
    assert counters["optimizer_steps"] == 120
    assert counters["backward_calls"] == 120
    assert counters["generation_requests"] == 416
    assert counters["model_loads"] == 3
    assert counters["tokenizer_loads"] == 3
    assert counters["per_event_checkpoint_writes"] == 120
    assert counters["artifact_checkpoint_writes"] == 1
    assert counters["checkpoint_writes"] == 121
    assert counters["retained_checkpoint_artifacts"] == 4
    report = read_json(output / "final" / "final_report.json")
    assert report["confirmation_only"] is True
    assert report["final_comparison"] is None
    assert report["cases"]["count"] == 32
    assert report["cases"]["predictions"] == 32
    assert report["cases"]["split_role_counts"] == {
        "final_fit": 12, "sealed_final": 20}
    assert report["native_execution_admissible"] is False
    journal = read_json(output / "final" / "journal.json")
    assert journal["final_started"] is True


def test_failed_fit_receipt_blocks_development_backend_and_labels(
        tmp_path, monkeypatch):
    output, _ = _run_fit(tmp_path)
    authorization = _authorization(tmp_path)
    receipt_path = output / "fit" / "receipt.json"
    receipt = read_json(receipt_path)
    receipt["status"] = "failed"
    receipt_path.write_text(json.dumps(seal_record(receipt), indent=2) + "\n")
    calls = []
    original = runner.read_jsonl

    def spy(path):
        calls.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(runner, "read_jsonl", spy)
    backend = _controlled_backend()
    with pytest.raises(Exception):
        _run_stage("development", output, backend, authorization)
    assert backend.startup_calls == 0
    assert "sealed_labels_dev.jsonl" not in calls
    assert "sealed_final_labels_fit.jsonl" not in calls
    assert "sealed_labels_final.jsonl" not in calls


def test_failed_development_receipt_blocks_final_backend_and_labels(
        tmp_path, monkeypatch):
    output, _ = _run_fit(tmp_path)
    authorization = _authorization(tmp_path)
    _run_stage("development", output, _controlled_backend(), authorization)
    receipt_path = output / "development" / "receipt.json"
    receipt = read_json(receipt_path)
    receipt["status"] = "failed"
    receipt_path.write_text(json.dumps(seal_record(receipt), indent=2) + "\n")
    calls = []
    original = runner.read_jsonl

    def spy(path):
        calls.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(runner, "read_jsonl", spy)
    backend = _controlled_backend()
    with pytest.raises(Exception):
        _run_stage("final", output, backend, authorization)
    assert backend.startup_calls == 0
    assert "sealed_final_teaching.jsonl" not in calls
    assert "sealed_final_labels_fit.jsonl" not in calls
    assert "sealed_labels_final.jsonl" not in calls


def test_injected_backend_cannot_claim_native_evidence(tmp_path):
    class NativeClaimingBackend(FakeBackend):
        @property
        def evidence_class(self):
            return "native_mps"

    output, result = _run_fit(
        tmp_path, backend=NativeClaimingBackend(_labels()))
    assert result["admitted"] is False
    report = read_json(output / "fit" / "fit_report.json")
    assert report["evidence_class"] == "cpu_fixture"
    assert report["native_execution_admissible"] is False
    receipt = read_json(output / "fit" / "receipt.json")
    assert receipt["evidence_class"] == "cpu_fixture"


def test_resume_after_interrupted_training_does_not_repeat_optimizer_work(
        tmp_path):
    output = tmp_path / "resume"
    authorization = _authorization(tmp_path)
    with pytest.raises(InjectedPublication):
        _run_stage("fit", output, FakeBackend(_labels(), fail_after=2),
                   authorization)
    result = _run_stage("fit", output, _controlled_backend(), authorization,
                        resume=True)
    assert result["status"] == "passed"
    counters = result["counters"]["cumulative"]
    assert counters["optimizer_steps"] == 108
    assert counters["backward_calls"] == 108
    assert counters["restart_backward_calls"] == 1
    assert counters["recovery_reloads"] == 1
    assert counters["restart_model_loads"] == 1
    assert counters["restart_tokenizer_loads"] == 1
    assert counters["generation_requests"] == 144
    journal = read_json(output / "fit" / "journal.json")
    assert journal["resumes_used"] == 1
    assert journal["ambiguous_work"] == []


def test_failed_fit_report_write_recovers_without_replaying_optimizer_work(
        tmp_path, monkeypatch):
    output = tmp_path / "publication"
    authorization = _authorization(tmp_path)
    original = publication.write_json

    def fail(root, relative, value, **kwargs):
        if relative == "fit/fit_report.json":
            raise InjectedPublication(relative)
        return original(root, relative, value, **kwargs)

    monkeypatch.setattr(publication, "write_json", fail)
    with pytest.raises(InjectedPublication):
        _run_stage("fit", output, _controlled_backend(), authorization)
    monkeypatch.setattr(publication, "write_json", original)
    assert (output / "fit" / "journal.json").is_file()
    result = _run_stage("fit", output, _controlled_backend(), authorization,
                        resume=True)
    assert result["status"] == "passed"
    assert result["counters"]["cumulative"]["optimizer_steps"] == 108
    assert (output / "fit" / "fit_report.json").is_file()


def test_interrupted_final_publication_recovers_without_replaying_updates(
        tmp_path, monkeypatch):
    output = tmp_path / "final_publication"
    authorization = _authorization(tmp_path)
    original = publication.Journal.record_peak

    def fail(self, report):
        raise InjectedPublication("fit live-peak publication")

    monkeypatch.setattr(publication.Journal, "record_peak", fail)
    with pytest.raises(InjectedPublication):
        _run_stage("fit", output, _controlled_backend(), authorization)
    journal = read_json(output / "fit" / "journal.json")
    assert journal["stage_complete"] is False
    assert journal["publication_stage_done"] is False
    monkeypatch.setattr(publication.Journal, "record_peak", original)
    result = _run_stage("fit", output, _controlled_backend(), authorization,
                        resume=True)
    assert result["status"] == "passed"
    assert result["counters"]["cumulative"]["optimizer_steps"] == 108
    journal = read_json(output / "fit" / "journal.json")
    assert journal["stage_complete"] is True
    assert journal["publication_stage_done"] is True
    assert result["backbone"]["backbone_unchanged"] is True


def test_ambiguous_native_failure_blocks_resume(tmp_path):
    output = tmp_path / "ambiguous"
    authorization = _authorization(tmp_path)
    with pytest.raises(RuntimeError, match="crash"):
        _run_stage("fit", output,
                   FakeBackend(_labels(), fail_after=2,
                               failure=RuntimeError("crash")), authorization)
    journal = read_json(output / "fit" / "journal.json")
    assert journal["ambiguous_work"]
    backend = _controlled_backend()
    with pytest.raises(Exception, match="ambiguous in-flight native work"):
        _run_stage("fit", output, backend, authorization, resume=True)
    assert backend.startup_calls == 0


def test_git_head_accepts_a_full_commit(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"))
    assert runner.git_head() == "a" * 40


def test_authorization_refuses_tampered_native_implementation(tmp_path):
    authorization = _authorization(tmp_path)

    def verifier(commit, path):
        if path == "src/replica_lab/dc_p1_native_runner.py":
            return b"# tampered"
        return _commit_verifier(commit, path)

    with pytest.raises(ProtocolError, match="bytes at"):
        runner.verify_authorization(
            config_path=CONFIG, prepared=PREPARED,
            authorization_path=authorization, protocol_commit=COMMIT,
            require_head=False, stage="fit", commit_verifier=verifier)


def test_work_meter_reservation_reconcile_and_abort(tmp_path):
    budget = read_json(CONFIG)["budget"]
    budget["physical_limits"]["max_training_forwards"] = 1
    meter = runner.WorkMeter(budget)
    reservation = meter.reserve({"training_forwards": 1}, arm="treatment")
    meter.reconcile(reservation, {"training_forwards": 1})
    assert meter.cumulative["training_forwards"] == 1
    with pytest.raises(Exception):
        meter.reserve({"training_forwards": 1}, arm="treatment")
    meter.abort(reservation, RuntimeError("in-flight")
    )
    assert meter.cumulative["training_forwards"] == 1
    assert meter.take_ambiguous_work()
    assert meter.take_ambiguous_work() == []
