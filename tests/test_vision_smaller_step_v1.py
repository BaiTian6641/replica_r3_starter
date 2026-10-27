"""CPU-only contracts for the predeclared vision smaller-step protocol."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vision_smaller_step_lr3e5_v1.json"
SCRIPT = ROOT / "scripts/vision_smaller_step_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "vision_smaller_step_validator_for_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def protocol_module():
    return _module()


@pytest.fixture()
def config(protocol_module):
    return protocol_module.load_protocol(CONFIG)


def _write_config(tmp_path, config):
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(config))
    return path


def _receipt(config, digest, commit="a" * 40):
    return {
        "format": "replica.vision_smaller_step.authorization.v1",
        "status": "authorized",
        "protocol_name": config["name"],
        "protocol_path": Path(CONFIG).as_posix(),
        "protocol_commit": commit,
        "config_sha256": digest,
        "native_execution_authorized": True,
        "dev_authorized": False,
        "heldout_authorized": False,
        "v0_authorized": False,
        "semantic_audit_authorized": False,
        "promotion_authorized": False,
        "retry_authorized": False,
        "issued_at_utc": "2026-10-27T00:00:00+00:00",
    }


def test_protocol_is_valid_deterministic_and_preparation_only(
        protocol_module, config):
    assert config["mode"] == "smaller_step"
    assert config["optimizer"]["lr"] == 3e-05
    assert config["hypothesis"]["completed_lr"] == 3e-04
    assert config["gradient_scope"]["trainable_tensors"] == [
        "vision_projector.0.weight",
        "vision_projector.0.bias",
        "vision_projector.1.weight",
        "vision_projector.1.bias",
        "vision_projector.3.weight",
        "vision_projector.3.bias",
    ]
    assert config["gradient_scope"]["backward_calls"] == 4
    assert config["gradient_scope"]["optimizer_steps"] == 1
    assert config["update_batch"]["rows"] == 4
    assert config["readouts"]["sequence"] == [
        "prefix_0", "update_batch_pre", "update_1", "update_batch_post",
        "prefix_1",
    ]
    assert config["readouts"]["update_batch"]["separate_evaluations"] == 2
    assert config["accounting"]["campaign_cumulative"]["backward_calls"] == 4
    assert config["accounting"]["campaign_cumulative"]["optimizer_steps"] == 1
    assert config["gates"]["optimization"][
        "max_update_batch_mean_nll_delta"] == 0.25
    assert config["gates"]["functional_fit"]["answer_correct_min"] == 1
    assert config["gates"]["decision"][
        "development_authorized_here"] is False
    assert config["claims"]["native_execution_authorized"] is False
    first = protocol_module.protocol_sha256(CONFIG)
    second = protocol_module.protocol_sha256(CONFIG)
    assert first == second
    assert len(first) == 64
    int(first, 16)
    summary = protocol_module.protocol_summary(config, config_path=CONFIG)
    assert summary["protocol_sha256"] == first
    assert summary["native_execution"] is False
    assert summary["model_loads"] == 0


@pytest.mark.parametrize("lr", [0.0003, 0.0001, 0.000003])
def test_wrong_lr_is_rejected(protocol_module, config, lr):
    drifted = copy.deepcopy(config)
    drifted["optimizer"]["lr"] = lr
    with pytest.raises(protocol_module.ProtocolError, match="lr"):
        protocol_module.validate_protocol(drifted, verify_files=False)


@pytest.mark.parametrize("mutate, message", [
    (lambda value: value["accounting"]["campaign_cumulative"].update(
        {"backward_calls": 5}), "backward_calls"),
    (lambda value: value["accounting"]["wider_readout_template"][
        "per_readout"].update({"prepare_sense_passes": 147}), "prepare"),
    (lambda value: value["accounting"]["update"].update(
        {"decoder_forward_calls": 9}), "decoder_forward_calls"),
    (lambda value: value["limits"].update(
        {"max_visual_encoder_calls": 169}), "max_visual_encoder_calls"),
    (lambda value: value["accounting"].update(
        {"artifact_bytes": 1024}), "artifact"),
    (lambda value: value["readouts"]["expected_measured_work"].update(
        {"decoder_forward_calls": 1}), "decoder_forward_calls"),
])
def test_wrong_budget_is_rejected(protocol_module, config, mutate, message):
    drifted = copy.deepcopy(config)
    mutate(drifted)
    with pytest.raises(protocol_module.ProtocolError, match=message):
        protocol_module.validate_protocol(drifted, verify_files=False)


@pytest.mark.parametrize("mutate, message", [
    (lambda value: value["gradient_scope"]["trainable_tensors"].append(
        "model.embed_tokens.weight"), "Gradient scope differs"),
    (lambda value: value["gradient_scope"].update(
        {"optimizer_owned_parameters": 7}), "exactly six"),
    (lambda value: value["optimizer"].update(
        {"wrapper": "torch.optim.AdamW"}), "NativeFusedAdamW"),
    (lambda value: value["update_batch"].update({"rows": 5}),
     "four rows"),
    (lambda value: value["gradient_scope"].update(
        {"backbone_gradient_tensors_allowed": 1}), "Backbone gradients"),
])
def test_wrong_scope_is_rejected(protocol_module, config, mutate, message):
    drifted = copy.deepcopy(config)
    mutate(drifted)
    with pytest.raises(protocol_module.ProtocolError, match=message):
        protocol_module.validate_protocol(drifted, verify_files=False)


@pytest.mark.parametrize("mutate, message", [
    (lambda value: value["access"].update(
        {"development": "allowed"}), "development"),
    (lambda value: value["access"]["training_paths"].append(
        "evaluation/dev.jsonl"), "Role access differs"),
    (lambda value: value["access"].update(
        {"sealed_evaluation_access": 1}), "Sealed evaluation"),
    (lambda value: value["authorization"].update(
        {"native_execution_authorized": True}), "predeclared protocol"),
])
def test_wrong_role_access_is_rejected(protocol_module, config, mutate, message):
    drifted = copy.deepcopy(config)
    mutate(drifted)
    with pytest.raises(protocol_module.ProtocolError, match=message):
        protocol_module.validate_protocol(drifted, verify_files=False)


def test_review_launch_refuses_without_authorization_receipt(
        protocol_module):
    with pytest.raises(protocol_module.AuthorizationRefused,
                       match="receipt missing"):
        protocol_module.review_launch(CONFIG)


def test_review_launch_refuses_receipt_for_wrong_config(
        protocol_module, config, tmp_path):
    receipt = _receipt(config, "0" * 64)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(protocol_module.ProtocolError,
                       match="does not bind the exact config"):
        protocol_module.review_launch(
            CONFIG, authorization_path=path, verify_commit=False)


def test_review_launch_refuses_malformed_receipt(
        protocol_module, config, tmp_path):
    receipt = _receipt(config, protocol_module.protocol_sha256(CONFIG))
    receipt.pop("protocol_commit")
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(protocol_module.ProtocolError, match="fields differ"):
        protocol_module.review_launch(
            CONFIG, authorization_path=path, verify_commit=False)


def test_review_launch_refuses_nonhex_or_uncommitted_commit(
        protocol_module, config, tmp_path):
    digest = protocol_module.protocol_sha256(CONFIG)
    nonhex = _receipt(config, digest, commit="not-a-commit")
    path = tmp_path / "nonhex.json"
    path.write_text(json.dumps(nonhex))
    with pytest.raises(protocol_module.ProtocolError, match="40-character"):
        protocol_module.review_launch(
            CONFIG, authorization_path=path, verify_commit=False)
    uncommitted = _receipt(config, digest, commit="0" * 40)
    path = tmp_path / "uncommitted.json"
    path.write_text(json.dumps(uncommitted))
    with pytest.raises(protocol_module.AuthorizationRefused,
                       match="not present"):
        protocol_module.review_launch(CONFIG, authorization_path=path)


def test_review_only_success_never_executes_native(
        protocol_module, config, tmp_path, monkeypatch):
    digest = protocol_module.protocol_sha256(CONFIG)
    receipt = _receipt(config, digest)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(protocol_module, "verify_commit_binding",
                        lambda *args, **kwargs: None)
    review = protocol_module.review_launch(
        CONFIG, authorization_path=path)
    assert review["status"] == "review_only_no_native_execution"
    assert review["native_execution"] is False
    assert review["model_loads"] == 0
    assert review["native_runner_implemented"] is True
    assert review["native_execution_performed_here"] is False
    assert review["launch_command"] == review["run_command"]
    assert review["launch_command"][2] == "run"
    assert review["authorization"]["protocol_commit"] == "a" * 40
    assert "review-launch" in review["review_command"]
    assert review["native_environment"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    assert review["native_environment_enforced_on_run"] is True


def test_run_refuses_before_native_load_without_receipt(
        protocol_module, config, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(protocol_module, "_execute_native_schedule",
                        lambda *args, **kwargs: calls.append(1))
    with pytest.raises(protocol_module.AuthorizationRefused,
                       match="receipt missing"):
        protocol_module.run_protocol(
            CONFIG, output=tmp_path / "run",
            model_path=ROOT / config["source"]["model_path"],
            model_lock=ROOT / config["source"]["model_lock"])
    assert calls == []


def test_run_refuses_malformed_authorization_before_native_load(
        protocol_module, config, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(protocol_module, "_execute_native_schedule",
                        lambda *args, **kwargs: calls.append(1))
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_receipt(config, "0" * 64)))
    with pytest.raises(protocol_module.ProtocolError,
                       match="does not bind the exact config"):
        protocol_module.run_protocol(
            CONFIG, output=tmp_path / "run",
            model_path=ROOT / config["source"]["model_path"],
            model_lock=ROOT / config["source"]["model_lock"],
            authorization_path=path)
    assert calls == []


def test_run_refuses_uncommitted_receipt_before_native_load(
        protocol_module, config, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(protocol_module, "_execute_native_schedule",
                        lambda *args, **kwargs: calls.append(1))
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_receipt(
        config, protocol_module.protocol_sha256(CONFIG), commit="0" * 40)))
    with pytest.raises(protocol_module.AuthorizationRefused,
                       match="not present"):
        protocol_module.run_protocol(
            CONFIG, output=tmp_path / "run",
            model_path=ROOT / config["source"]["model_path"],
            model_lock=ROOT / config["source"]["model_lock"],
            authorization_path=path)
    assert calls == []


def test_run_authorizes_once_before_mocked_native_dispatch(
        protocol_module, config, tmp_path, monkeypatch):
    for key in protocol_module.NATIVE_ENVIRONMENT:
        monkeypatch.setenv(key, "0")
    digest = protocol_module.protocol_sha256(CONFIG)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_receipt(config, digest)))
    monkeypatch.setattr(protocol_module, "verify_commit_binding",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(protocol_module, "_verify_run_inputs",
                        lambda *args, **kwargs: None)
    calls = []

    def fake_execute(config_value, config_path, receipt, **kwargs):
        calls.append((config_value, config_path, receipt, kwargs))
        return {"status": "mocked_native"}

    monkeypatch.setattr(protocol_module, "_execute_native_schedule",
                        fake_execute)
    report = protocol_module.run_protocol(
        CONFIG, output=tmp_path / "run",
        model_path=ROOT / config["source"]["model_path"],
        model_lock=ROOT / config["source"]["model_lock"],
        authorization_path=path)
    assert report == {"status": "mocked_native"}
    assert len(calls) == 1
    assert calls[0][2]["config_sha256"] == digest
    assert calls[0][3]["root"] == protocol_module.ROOT


def test_native_environment_overrides_caller_values(protocol_module):
    environment = {
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "HF_DATASETS_OFFLINE": "0",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    }
    forced = protocol_module._enforce_native_environment(environment)
    assert forced == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    assert environment == forced


def test_run_forces_native_environment_before_dispatch(
        protocol_module, config, tmp_path, monkeypatch):
    for key in protocol_module.NATIVE_ENVIRONMENT:
        monkeypatch.setenv(key, "0" if key != "PYTORCH_ENABLE_MPS_FALLBACK"
                           else "1")
    digest = protocol_module.protocol_sha256(CONFIG)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_receipt(config, digest)))
    monkeypatch.setattr(protocol_module, "verify_commit_binding",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(protocol_module, "_verify_run_inputs",
                        lambda *args, **kwargs: None)
    seen = {}

    def fake_execute(*args, **kwargs):
        seen.update({key: os.environ.get(key)
                     for key in protocol_module.NATIVE_ENVIRONMENT})
        return {"status": "mocked"}

    monkeypatch.setattr(protocol_module, "_execute_native_schedule",
                        fake_execute)
    assert protocol_module.run_protocol(
        CONFIG, output=tmp_path / "run",
        model_path=ROOT / config["source"]["model_path"],
        model_lock=ROOT / config["source"]["model_lock"],
        authorization_path=path) == {"status": "mocked"}
    assert seen == protocol_module.NATIVE_ENVIRONMENT


def test_artifact_budget_includes_the_final_report(
        protocol_module, tmp_path):
    output = tmp_path / "artifacts"
    output.mkdir()
    payload = output / "payload.bin"
    payload.write_bytes(b"x" * 2000)
    report = {"format": "test", "status": "ok", "payload": "x" * 1000}
    limit = 2100
    result = protocol_module._write_report_with_artifact_budget(
        report, output, limit)
    assert result["artifact_budget_ok"] is False
    assert result["artifact_bytes"] > limit
    assert result["artifact_bytes"] == protocol_module._artifact_bytes(output)
    assert (output / "report.json").is_file()
    assert payload.is_file()
    written = json.loads((output / "report.json").read_text())
    assert written["artifact_bytes"] == result["artifact_bytes"]
    assert written["artifact_budget_ok"] is False
    assert written["artifact_limit_bytes"] == limit

    permissive = tmp_path / "permissive"
    permissive.mkdir()
    (permissive / "payload.bin").write_bytes(b"x" * 1000)
    small_report = {"format": "test", "status": "ok", "payload": "x" * 100}
    allowed = protocol_module._write_report_with_artifact_budget(
        small_report, permissive, 100000)
    assert allowed["artifact_budget_ok"] is True
    assert allowed["artifact_bytes"] == protocol_module._artifact_bytes(
        permissive)
    assert (permissive / "report.json").is_file()


def test_artifact_overflow_preserves_report_and_evidence(
        protocol_module, config, tmp_path):
    output = tmp_path / "overflow"
    output.mkdir()
    (output / "payload.bin").write_bytes(b"x" * 200)
    limited = copy.deepcopy(config)
    limited["limits"]["artifact_bytes"] = 100

    class FakeProfiler:
        def mark(self, *args, **kwargs):
            return None

        def stop(self):
            return {"ok": True}

    class FakeGuard:
        def report(self):
            return {"ok": True}

    class FakeLoads:
        def counts(self):
            return {"model_loads": 1, "extension_loads": 0}

    with pytest.raises(protocol_module.ArtifactBudgetError):
        protocol_module._finish_run(
            limited, CONFIG, {
                "protocol_commit": "a" * 40,
                "native_execution_authorized": True,
            }, output=output,
            started=protocol_module.time.monotonic(),
            guard=FakeGuard(), profiler=FakeProfiler(),
            accounting_loads=FakeLoads(),
            accounting_totals={"counters": {}},
            measured_work={"ok": True},
            readouts=[{"label": "prefix_0"}, {"label": "prefix_1"}],
            update={"backward_calls": 4, "optimizer_steps": 1},
            pre={"mean_response_nll": 10.0},
            post={"mean_response_nll": 10.1},
            optimization={"ok": True,
                          "status_on_fail": "optimization_gate_failed"},
            functional={"ok": True,
                        "status_on_fail":
                            "smaller_step_functional_fit_failed"},
            init_hash="0" * 64, stop_source="test")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "stopped_engineering_failure"
    assert report["artifact_budget_ok"] is False
    assert report["artifact_budget_exceeded"] is True
    assert report["artifact_bytes"] > 100
    assert (output / "artifact_budget_failure.json").is_file()
    assert (output / "payload.bin").is_file()
    assert (output / "report.json").is_file()


def test_schedule_is_exact_and_training_only(protocol_module, config):
    steps = protocol_module.schedule_steps(config)
    assert [step["label"] for step in steps] == [
        "prefix_0", "update_batch_pre", "update_1", "update_batch_post",
        "prefix_1",
    ]
    assert [step["kind"] for step in steps] == [
        "wider_readout", "update_batch_loss", "optimizer_update",
        "update_batch_loss", "wider_readout",
    ]
    assert {step["role"] for step in steps} <= {
        "train_fit", "train_update_batch"}
    protocol_module.assert_training_only_schedule(config)
    drifted = copy.deepcopy(config)
    drifted["access"]["development"] = "allowed"
    with pytest.raises(protocol_module.ProtocolError, match="development"):
        protocol_module.assert_training_only_schedule(drifted)


def _forward(budget, *, prepare):
    increments = {
        "decoder_forward_calls": 1,
        "processed_text_positions": 13,
        "processed_visual_positions": 196,
        "visual_forward_calls": 1,
        "projector_passes": 1,
    }
    if prepare:
        increments["prepare_sense_passes"] = 1
        phase = "prepare_sense"
    else:
        increments["explicit_logits_forwards"] = 1
        phase = "explicit_logits_loss"
    budget.check_forward(phase=phase, text_positions=13, visual_tokens=196)
    budget.commit_forward(increments)


def _evaluation_readout(budget):
    for _ in range(4):
        _forward(budget, prepare=True)
        _forward(budget, prepare=False)
        budget.check_event("visual_observation_calls", 1)
        budget.commit_event("visual_observation_calls", 1)


def test_pre_and_post_update_batch_accounting_is_stage_exact(
        protocol_module, config):
    budget = protocol_module._ScheduleBudget(config, clock=lambda: 0.0)
    budget.select("update_batch")
    budget.begin_readout(1)
    _evaluation_readout(budget)
    pre = budget.end_readout()
    budget.finish_readout()
    assert pre["decoder_forward_calls"] == 8
    assert pre["prepare_sense_passes"] == 4
    assert pre["explicit_logits_forwards"] == 4
    assert pre["visual_observation_calls"] == 4
    assert pre["generated_tokens"] == 0
    assert pre["backward_calls"] == 0
    assert pre["optimizer_steps"] == 0

    budget.begin_readout(2)
    _evaluation_readout(budget)
    post = budget.end_readout()
    budget.finish_readout()
    assert post["decoder_forward_calls"] == 8
    assert post["prepare_sense_passes"] == 4
    assert post["explicit_logits_forwards"] == 4
    assert post["visual_observation_calls"] == 4

    budget.begin_readout(3)
    with pytest.raises(protocol_module.fwd.AccountingError):
        _forward(budget, prepare=True)


def _update_record(protocol_module):
    return {
        "backward_calls": 4,
        "optimizer_steps": 1,
        "changed_projector_tensors": list(protocol_module.PROJECTOR_TENSORS),
        "nonprojector_gradient_names": [],
        "base_stamp_unchanged": True,
        "response_only_labels": True,
        "losses": [1.0, 1.0, 1.0, 1.0],
        "loss": 1.0,
        "grad_norm": 1.0,
    }


def _prefix(answer, caption):
    return {"teacher_forced": {"answer": answer, "caption": caption}}


def test_optimization_gate_rejects_completed_overshoot(
        protocol_module, config):
    decision = protocol_module.evaluate_optimization_gate(
        config, update=_update_record(protocol_module),
        pre={"mean_response_nll": 10.964820146560669},
        post={"mean_response_nll": 12.560267925262451},
        prefix_0=_prefix(10.0, 10.0), prefix_1=_prefix(10.1, 10.1))
    assert decision["ok"] is False
    assert "optimization:update_batch_nll_delta" in decision["violations"]
    passing = protocol_module.evaluate_optimization_gate(
        config, update=_update_record(protocol_module),
        pre={"mean_response_nll": 10.964820146560669},
        post={"mean_response_nll": 11.0},
        prefix_0=_prefix(10.0, 10.0), prefix_1=_prefix(10.1, 10.1))
    assert passing["ok"] is True


def _functional_readout(*, answer_accuracy=0.0, prefill=0.0,
                        token_f1=0.0, semantic_f1=0.0, margin=0.0,
                        eos_only=0, whitespace_only=0):
    return {
        "scores": {
            "answer": {
                "answer_correct_cases": 14,
                "answer_correct_accuracy": answer_accuracy,
                "answer_prefill_gold_accuracy": prefill,
                "answer_correct_margin_sibling": margin,
                "answer_correct_margin_shuffled": margin,
                "answer_correct_margin_noimage": margin,
            },
            "caption": {
                "caption_token_f1": token_f1,
                "caption_semantic_f1": semantic_f1,
            },
        },
        "generation_summary": {
            "generation_cells": 98,
            "eos_only": eos_only,
            "whitespace_only": whitespace_only,
        },
    }


def test_functional_fit_gate_rejects_empty_outputs_and_rank_only(
        protocol_module, config):
    baseline = _functional_readout()
    rejected = protocol_module.evaluate_functional_fit_gate(
        config, prefix_0=baseline, prefix_1=_functional_readout())
    assert rejected["ok"] is False
    assert "functional:answer_correct" in rejected["violations"]
    assert "functional:caption_token_f1_gain" in rejected["violations"]
    eos_rejected = protocol_module.evaluate_functional_fit_gate(
        config, prefix_0=baseline,
        prefix_1=_functional_readout(
            answer_accuracy=1 / 14, prefill=1 / 14, token_f1=0.2,
            semantic_f1=0.1, margin=0.1, eos_only=98))
    assert eos_rejected["ok"] is False
    assert "functional:eos_only_rate" in eos_rejected["violations"]
    passing = protocol_module.evaluate_functional_fit_gate(
        config, prefix_0=baseline,
        prefix_1=_functional_readout(
            answer_accuracy=1 / 14, prefill=1 / 14, token_f1=0.2,
            semantic_f1=0.1, margin=0.1))
    assert passing["ok"] is True


def test_cli_run_refusal_is_nonzero_and_loads_no_model(
        protocol_module, config, tmp_path, capsys):
    code = protocol_module.main([
        "run", "--config", str(CONFIG),
        "--output", str(tmp_path / "run"),
        "--model-path", str(ROOT / config["source"]["model_path"]),
        "--model-lock", str(ROOT / config["source"]["model_lock"]),
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "refused" in captured.err
    assert "model_loads" not in captured.out


def test_protocol_hash_cli_matches_file_digest(protocol_module, capsys):
    code = protocol_module.main(["hash", "--config", str(CONFIG)])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
