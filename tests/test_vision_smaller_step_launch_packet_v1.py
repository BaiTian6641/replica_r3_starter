"""CPU-only contracts for the smaller-step launch/evidence packet.

These tests build the packet from the repository, prove it is deterministic and
complete, and refuse stale hashes, wrong limits/gates and any launch attempt
before native loading.  No test loads a model, imports a native runtime or runs
a native command.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vision_smaller_step_lr3e5_v1.json"
PACKET = (ROOT / "reports/vision_smaller_step_launch_packet_v1_20261027/"
                 "launch_packet.json")
BUILDER = ROOT / "scripts/build_vision_smaller_step_launch_packet_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "vision_smaller_step_launch_packet_for_tests", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _module()


@pytest.fixture(scope="module")
def protocol(builder):
    return builder._protocol_module().load_protocol(CONFIG)


@pytest.fixture(scope="module")
def packet(builder):
    return builder.build_packet(CONFIG)


def _tamper(builder, packet, mutate):
    clone = copy.deepcopy(packet)
    mutate(clone)
    clone["packet_sha256"] = builder._packet_sha256(clone)
    return clone


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_packet_is_deterministic_complete_and_matches_the_repository(
        builder, packet):
    rebuilt = builder.build_packet(CONFIG)
    relative = builder.build_packet(
        "configs/vision_smaller_step_lr3e5_v1.json")
    assert builder.packet_bytes(packet) == builder.packet_bytes(rebuilt)
    assert builder.packet_bytes(packet) == builder.packet_bytes(relative)
    assert set(packet) == builder.PACKET_KEYS
    assert packet["format"] == builder.FORMAT
    assert packet["kind"] == "preparation_only_launch_evidence"
    assert packet["deterministic"] is True
    assert packet["native_execution"] is False
    assert packet["model_loads"] == 0
    assert packet["authorization_created"] is False
    assert packet["protocol"]["config_sha256"] == _sha256(CONFIG)
    assert packet["implementation"]["config"]["sha256"] == _sha256(CONFIG)
    assert packet["implementation"]["runner"]["sha256"] == _sha256(
        ROOT / builder.RUNNER_PATH)
    assert packet["implementation"]["packet_builder"]["sha256"] == _sha256(
        BUILDER)
    assert packet["implementation"]["packet_test"]["sha256"] == _sha256(
        Path(__file__))
    for pin in packet["implementation"]["pins"].values():
        assert pin["verified"] is True
        assert pin["sha256"] == _sha256(ROOT / pin["path"])
        assert pin["sha256"] == pin["declared_sha256"]
    assert len(packet["packet_sha256"]) == 64
    int(packet["packet_sha256"], 16)
    assert packet["packet_sha256"] == builder._packet_sha256(packet)
    assert builder.DEFAULT_PACKET == (
        "reports/vision_smaller_step_launch_packet_v1_20261027/"
        "launch_packet.json")
    assert PACKET.is_file()
    on_disk = json.loads(PACKET.read_text())
    assert on_disk == packet
    assert PACKET.read_bytes() == builder.packet_bytes(packet)


def test_limits_are_exact_and_include_the_final_report(builder, packet,
                                                       protocol):
    limits = packet["limits"]
    assert limits == {
        "decoder_forward_calls": 4824,
        "visual_observation_calls": 168,
        "backward_calls": 4,
        "optimizer_steps": 1,
        "wall_seconds": 1200,
        "artifact_bytes": 33554432,
        "artifact_includes_final_report": True,
        "model_loads": 1,
        "downloads": 0,
        "retries": 0,
        "workers": 1,
        "cpu_fallback": False,
    }
    assert limits["decoder_forward_calls"] == protocol["limits"][
        "max_decoder_forwards"]
    assert limits["visual_observation_calls"] == protocol["accounting"][
        "campaign_cumulative"]["visual_observation_calls"]
    assert limits["backward_calls"] == protocol["limits"][
        "max_total_backward_calls"]
    assert limits["optimizer_steps"] == protocol["limits"][
        "max_optimizer_steps"]
    assert limits["wall_seconds"] == protocol["limits"]["max_seconds"]
    assert limits["artifact_bytes"] == protocol["limits"]["artifact_bytes"]
    assert limits["artifact_includes_final_report"] is True


def test_single_variable_parent_seed_batch_and_scope_are_preserved(
        builder, packet, protocol):
    assert packet["protocol"]["lr"] == 3e-05
    assert packet["protocol"]["completed_lr"] == 3e-04
    assert packet["scope"]["single_variable"] == {
        "changed": "optimizer.lr",
        "completed": 0.0003,
        "arm": 3e-05,
        "statement": protocol["hypothesis"]["single_discriminating_change"],
    }
    assert packet["scope"]["preserved"]["parent"] == (
        "checkpoints/minicpm5_source")
    assert packet["scope"]["preserved"]["seed"] == 29
    assert packet["scope"]["preserved"]["update_batch_example_ids"] == [
        "caption:vba2-w013-p0-a",
        "caption:vba2-w025-p0-a",
        "answer:vba2-w000-p0-a",
        "answer:vba2-w036-p0-b",
    ]
    assert packet["scope"]["preserved"]["projector_tensors"] == [
        "vision_projector.0.weight",
        "vision_projector.0.bias",
        "vision_projector.1.weight",
        "vision_projector.1.bias",
        "vision_projector.3.weight",
        "vision_projector.3.bias",
    ]
    assert packet["scope"]["preserved"]["optimizer_wrapper"] == (
        "NativeFusedAdamW")
    assert packet["scope"]["preserved"]["response_only_labels"] is True
    assert packet["scope"]["preserved"][
        "completed_reproduction_repeated"] is False
    assert packet["lineage"]["initialization"]["init_seed"] == 29
    assert packet["lineage"]["initialization"]["expected_init_sha256"] == (
        "511ba5c7648851424649962467972f0e926303d64873dfce08ecd3de299314b9")
    assert packet["lineage"]["update_batch"]["completed_batch_reused"] is True
    assert packet["lineage"]["update_batch"]["rows"] == 4
    assert packet["lineage"]["update_batch"]["roles"] == {
        "answer": 2, "caption": 2}
    assert packet["gates"]["optimization"][
        "max_update_batch_mean_nll_delta"] == 0.25
    assert packet["gates"]["functional_fit"]["answer_correct_min"] == 1
    assert packet["gates"]["functional_fit"][
        "answer_correct_denominator"] == 14
    assert packet["gates"]["decision"]["development_authorized_here"] is False
    assert packet["lineage"]["evidence"]["completed_arm"]["files"]["report"][
        "verified"] is True
    assert packet["lineage"]["precondition"]["satisfied"] is True
    assert packet["lineage"]["precondition"]["observed_decision"] == (
        "spatial_contribution_observed")


def test_image_controls_and_raw_generation_evidence_are_required(packet):
    controls = ["correct", "sibling", "shuffled", "noimage"]
    assert packet["gates"]["image_controls"] == controls
    evidence = packet["gates"]["raw_generation_evidence"]
    assert evidence["required"] is True
    assert evidence["raw_generation_required"] is True
    assert evidence["controls_required"] == controls
    assert evidence["readouts"] == ["prefix_0", "prefix_1"]
    assert evidence["generation_roles"] == ["answer", "caption"]
    assert evidence["update_batch_generation"] is False
    assert evidence["expected_generated_tokens"] == 3800
    assert evidence["eos_only_not_counted_correct"] is True
    assert evidence["whitespace_only_not_counted_correct"] is True
    assert evidence["max_new_tokens_by_kind"] == {
        "answer": 8, "caption": 32}
    for entry in evidence["per_readout"]:
        assert entry["controls"] == controls
        assert entry["answer_generations"].endswith(
            "/train_answer_generations.jsonl")
        assert entry["caption_generations"].endswith(
            "/train_caption_generations.jsonl")
        assert entry["records"].endswith("/records.jsonl")
        assert entry["records_expected"] == 104
        assert entry["available"] == 98
        assert entry["placeholders"] == 6
    assert len(evidence["per_readout"]) == 2
    assert evidence["update_batch_records"] == [
        "runs/vision_smaller_step_lr3e5_v1_20261027/readouts/"
        "update_batch_pre/records.jsonl",
        "runs/vision_smaller_step_lr3e5_v1_20261027/readouts/"
        "update_batch_post/records.jsonl",
    ]


def test_launch_command_offline_environment_and_fresh_output(packet):
    launch = packet["launch"]
    assert launch["cwd"] == "."
    assert launch["argv"] == [
        "python", "scripts/vision_smaller_step_v1.py", "run",
        "--config", "configs/vision_smaller_step_lr3e5_v1.json",
        "--output", "runs/vision_smaller_step_lr3e5_v1_20261027",
        "--model-path", "checkpoints/minicpm5_source",
        "--model-lock", "configs/minicpm5_sealed_lock.local.json",
        "--authorization",
        "reports/vision_smaller_step_lr3e5_v1_20261027/authorization.json",
    ]
    assert launch["environment"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    assert launch["shell"].startswith(
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
        "PYTORCH_ENABLE_MPS_FALLBACK=0 python "
        "scripts/vision_smaller_step_v1.py run")
    assert launch["offline_required"] is True
    assert launch["mps_cpu_fallback_disabled"] is True
    assert launch["authorization_before_native_imports"] is True
    assert launch["native_imports_before_authorization"] == 0
    assert launch["model_loads_before_authorization"] == 0
    assert launch["fresh_output"] == {
        "path": "runs/vision_smaller_step_lr3e5_v1_20261027",
        "must_not_exist": True,
        "checked_before_model_load": True,
    }
    assert launch["preflight_command"][2] == "review-launch"
    assert not (ROOT / launch["fresh_output"]["path"]).exists()


def test_authorization_is_separate_declared_and_never_minted(builder, packet,
                                                             protocol):
    authorization = packet["authorization"]
    assert authorization["required"] is True
    assert authorization["created"] is False
    assert authorization["present_at_build"] is False
    assert authorization["minted_by_packet"] is False
    assert authorization["separate_from_packet"] is True
    assert authorization["path"] == (
        "reports/vision_smaller_step_lr3e5_v1_20261027/authorization.json")
    assert authorization["path"] != builder.DEFAULT_PACKET
    assert authorization["path"] != packet["protocol"]["config_path"]
    assert authorization["format"] == (
        "replica.vision_smaller_step.authorization.v1")
    assert authorization["status"] == "absent"
    assert authorization["protocol_commit_required"] is True
    assert authorization["protocol_commit"] is None
    assert authorization["commit_bound"] is False
    assert authorization["commit_binding_required"] is True
    assert authorization["config_sha256"] == packet["protocol"][
        "config_sha256"]
    assert authorization["native_execution_authorized"] is False
    schema = authorization["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == protocol["authorization"][
        "required_receipt_fields"]
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["properties"]["format"]["const"] == (
        "replica.vision_smaller_step.authorization.v1")
    assert schema["properties"]["status"]["const"] == "authorized"
    assert schema["properties"]["config_sha256"]["const"] == (
        packet["protocol"]["config_sha256"])
    assert schema["properties"]["protocol_commit"]["pattern"] == (
        "^[0-9a-f]{40}$")
    assert schema["properties"]["native_execution_authorized"]["const"] is True
    for role in ("dev", "heldout", "v0", "semantic_audit", "promotion",
                 "retry"):
        assert schema["properties"][f"{role}_authorized"]["const"] is False
    assert not (ROOT / authorization["path"]).exists()


def test_preflight_lists_every_check_and_native_gap(packet):
    statuses = {check["id"]: check["status"] for check in packet["preflight"]}
    assert statuses["protocol_contract"] == "passed"
    assert statuses["config_sha256"] == "passed"
    assert statuses["implementation_pins"] == "passed"
    assert statuses["evidence_lineage"] == "passed"
    assert statuses["prepared_inputs"] == "passed"
    assert statuses["precondition_decision"] == "passed"
    assert statuses["single_variable_scope"] == "passed"
    assert statuses["limits_exact"] == "passed"
    assert statuses["gates_declared"] == "passed"
    assert statuses["protocol_commit"] == "pending"
    assert statuses["authorization_receipt"] == "pending"
    assert statuses["fresh_output"] == "pending"
    assert statuses["native_execution"] == "not_performed"
    assert statuses["native_environment"] == "declared"
    assert all(check["requirement"] for check in packet["preflight"])
    assert len(packet["unverified_native_steps"]) >= 8
    joined = " ".join(packet["unverified_native_steps"])
    assert "4 backward calls" in joined
    assert "1 NativeFusedAdamW optimizer step" in joined
    assert "authorization receipt" in joined
    assert "raw answer/caption generations" in joined
    limitations = " ".join(packet["limitations"])
    assert "never an authorization receipt" in limitations
    assert "unobserved" in limitations
    assert "Development, held-out, semantic-audit, V0, promotion and retry" \
        in limitations


def test_packet_hash_tamper_is_refused(builder, packet):
    broken = copy.deepcopy(packet)
    broken["limits"]["decoder_forward_calls"] = 4825
    with pytest.raises(builder.StalePacketError, match="packet sha256"):
        builder.verify_packet(broken, config_path=CONFIG)


@pytest.mark.parametrize("mutate, section", [
    (lambda value: value["implementation"]["runner"].update(
        {"sha256": "0" * 64}), "implementation"),
    (lambda value: value["implementation"]["pins"][
        "src/replica_lab/vision_forward_accounting.py"].update(
            {"sha256": "0" * 64}), "implementation"),
    (lambda value: value["lineage"]["evidence"]["completed_arm"]["files"][
        "report"].update({"sha256": "0" * 64}), "lineage"),
    (lambda value: value["lineage"]["prepared"]["files"]["manifest"].update(
        {"sha256": "0" * 64}), "lineage"),
    (lambda value: value["protocol"].update(
        {"config_sha256": "0" * 64}), "protocol"),
    (lambda value: value["protocol"].update(
        {"lr": 0.0003}), "protocol"),
])
def test_stale_hashes_and_config_bindings_are_refused(
        builder, packet, mutate, section):
    tampered = _tamper(builder, packet, mutate)
    with pytest.raises(builder.StalePacketError, match=section):
        builder.verify_packet(tampered, config_path=CONFIG)


@pytest.mark.parametrize("key, value", [
    ("decoder_forward_calls", 4825),
    ("visual_observation_calls", 169),
    ("backward_calls", 5),
    ("optimizer_steps", 2),
    ("wall_seconds", 1201),
    ("artifact_bytes", 33554433),
    ("artifact_includes_final_report", False),
    ("model_loads", 2),
    ("downloads", 1),
    ("retries", 1),
    ("workers", 2),
    ("cpu_fallback", True),
])
def test_wrong_limits_are_refused(builder, packet, key, value):
    tampered = _tamper(
        builder, packet, lambda item: item["limits"].update({key: value}))
    with pytest.raises(builder.StalePacketError, match="limits"):
        builder.verify_packet(tampered, config_path=CONFIG)


@pytest.mark.parametrize("mutate, section", [
    (lambda value: value["gates"]["optimization"].update(
        {"max_update_batch_mean_nll_delta": 0.3}), "gates"),
    (lambda value: value["gates"]["optimization"].update(
        {"max_wider_answer_mean_nll_delta": 0.3}), "gates"),
    (lambda value: value["gates"]["functional_fit"].update(
        {"answer_correct_min": 0}), "gates"),
    (lambda value: value["gates"]["functional_fit"].update(
        {"caption_semantic_f1_min": 0.0}), "gates"),
    (lambda value: value["gates"]["functional_fit"].update(
        {"eos_only_rate_max": 0.9}), "gates"),
    (lambda value: value["gates"]["decision"].update(
        {"development_authorized_here": True}), "gates"),
    (lambda value: value["gates"]["decision"].update(
        {"heldout_authorized": True}), "gates"),
    (lambda value: value["gates"]["image_controls"].remove("noimage"),
     "gates"),
    (lambda value: value["gates"]["raw_generation_evidence"].update(
        {"raw_generation_required": False}), "gates"),
    (lambda value: value["gates"]["raw_generation_evidence"].update(
        {"expected_generated_tokens": 3799}), "gates"),
    (lambda value: value["scope"]["optimizer"].update(
        {"lr": 0.0003}), "scope"),
    (lambda value: value["scope"]["optimizer"].update(
        {"wrapper": "torch.optim.AdamW"}), "scope"),
    (lambda value: value["scope"]["gradient_scope"][
        "trainable_tensors"].append("model.embed_tokens.weight"), "scope"),
    (lambda value: value["launch"]["environment"].update(
        {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}), "launch"),
    (lambda value: value["launch"].update({"offline_required": False}),
     "launch"),
    (lambda value: value["authorization"].update(
        {"native_execution_authorized": True}), "authorization"),
    (lambda value: value["authorization"].update(
        {"protocol_commit": "0" * 40, "commit_bound": True}),
     "authorization"),
    (lambda value: value["authorization"].update(
        {"created": True}), "authorization"),
])
def test_wrong_gates_scope_environment_and_authorization_are_refused(
        builder, packet, mutate, section):
    tampered = _tamper(builder, packet, mutate)
    with pytest.raises(builder.StalePacketError, match=section):
        builder.verify_packet(tampered, config_path=CONFIG)


def test_launch_is_refused_before_any_native_loading(
        builder, packet, monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("native execution or subprocess attempted")

    protocol_module = builder._protocol_module()
    monkeypatch.setattr(protocol_module, "run_protocol", boom)
    monkeypatch.setattr(protocol_module, "_execute_native_schedule", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    rebuilt = builder.build_packet(CONFIG)
    builder.verify_packet(rebuilt, config_path=CONFIG)
    disposition = builder.launch_preflight(rebuilt, config_path=CONFIG)
    assert calls == []
    assert disposition["format"] == (
        "replica.vision_smaller_step.launch_preflight.v1")
    assert disposition["status"] == "review_only_launch_refused"
    assert disposition["launch_allowed"] is False
    assert disposition["native_execution"] is False
    assert disposition["model_loads"] == 0
    assert "packet_is_not_an_authorization_receipt" in disposition["refusals"]
    assert "packet_is_preparation_only" in disposition["refusals"]
    assert "authorization_receipt_missing" in disposition["refusals"]
    assert "protocol_commit_unbound" in disposition["refusals"]
    assert "commit_binding_unverified" in disposition["refusals"]
    assert "native_execution_not_authorized" in disposition["refusals"]
    assert disposition["protocol_commit"] is None
    assert disposition["required_review_command"] == rebuilt["launch"][
        "preflight_command"]
    assert disposition["authorized_launch_command"] == rebuilt["launch"][
        "argv"]
    assert disposition["authorization_path"] == rebuilt["authorization"][
        "path"]


def test_builder_and_preflight_create_no_authorization_or_run_artifacts(
        builder, packet):
    assert not (ROOT / packet["launch"]["fresh_output"]["path"]).exists()
    assert not (ROOT / packet["authorization"]["path"]).exists()
    assert not (ROOT / "runs/vision_smaller_step_lr3e5_v1_20261027/"
                       "report.json").exists()
    assert packet["authorization"]["minted_by_packet"] is False
    assert packet["authorization_created"] is False
    assert packet["native_execution"] is False
    assert packet["model_loads"] == 0
    disposition = builder.launch_preflight(packet, config_path=CONFIG)
    assert disposition["launch_allowed"] is False
    assert not (ROOT / packet["authorization"]["path"]).exists()
