"""CPU-only contracts for the separately scoped co-adaptation control.

The control may only be considered after a preserved smaller-step failure and a
preserved frozen-feature diagnosis.  These tests prove it keeps the parent,
data, batch, initialization, lr, loss, readouts and budget identical, changes
only gradient scope under grounded encoder tensor names and drift bounds, is
not eligible yet, is never authorized by preparation, and refuses any expansion
or tampering.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/vision_coadaptation_control_v1.py"
CONFIG = "configs/vision_coadaptation_control_v1.json"
SMALLER = "configs/vision_smaller_step_lr3e5_v1.json"
REPORT = ("reports/vision_coadaptation_control_v1_20261027/"
          "readiness.json")
REVIEWED_COMMIT = "4c5a84851009898bc876c0b8aa3ccb596e04037d"
CONFIG_SHA256 = "931851ecc954cea91d97aa7a087c1a2bef528172379b06d453e6bec93ef4abad"
SMALLER_REPORT = "reports/vision_smaller_step_lr3e5_v1_20261027/report.json"
DIAGNOSIS = ("reports/vision_smaller_step_frozen_feature_diagnosis_v1_"
             "20261027/diagnosis.json")
OUTPUT = "runs/vision_coadaptation_control_v1_20261027"
RUNNER = "scripts/vision_coadaptation_control_runner_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "vision_coadaptation_control_for_tests", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def control():
    return _module()


@pytest.fixture(scope="module")
def config():
    return json.loads((ROOT / CONFIG).read_text())


@pytest.fixture(scope="module")
def smaller():
    return json.loads((ROOT / SMALLER).read_text())


@pytest.fixture(scope="module")
def assessment(control):
    return control.assess(root=ROOT)


def test_assessment_is_deterministic_current_and_preparation_only(
        control, assessment):
    rebuilt = control.assess(root=ROOT)
    assert control.assessment_bytes(assessment) == (
        control.assessment_bytes(rebuilt))
    assert assessment["format"] == control.FORMAT
    assert assessment["deterministic"] is True
    assert assessment["native_execution"] is False
    assert assessment["model_loads"] == 0
    assert assessment["authorization_created"] is False
    assert assessment["decision"]["preparation_complete"] is True
    assert assessment["decision"]["launch_allowed"] is False
    assert assessment["decision"]["native_execution"] is False
    assert assessment["decision"]["model_loads"] == 0
    assert assessment["decision"]["authorization_created"] is False
    assert assessment["assessment_sha256"] == (
        control._assessment_sha256(assessment))
    checked_in = ROOT / REPORT
    assert checked_in.is_file()
    assert checked_in.read_bytes() == control.assessment_bytes(assessment)
    assert control.verify_assessment(report=assessment, root=ROOT) == assessment


def test_scope_is_grounded_in_the_preserved_encoder_manifest(
        control, config, assessment):
    scope = config["gradient_scope"]
    structure = scope["encoder_structure"]
    manifest = json.loads(
        (ROOT / config["parent"]["vision_source_manifest"]).read_text())
    model = manifest["model"]
    assert structure["architecture"] == model["architecture"] == (
        "SiglipVisionModel")
    assert structure["base_model_prefix"] == model["base_model_prefix"]
    assert structure["feature_version"] == model["feature_version"]
    assert structure["num_hidden_layers"] == model["num_hidden_layers"] == 12
    assert structure["hidden_size"] == model["hidden_size"] == 768
    assert structure["intermediate_size"] == model["intermediate_size"] == 3072
    assert structure["adapted_layer_index"] == (
        model["num_hidden_layers"] - 1) == 11
    weights = next(entry["sha256"] for entry in manifest["files"]
                   if entry["role"] == "vision_encoder_weights")
    assert config["parent"]["vision_weights_sha256"] == weights
    names = control._encoder_tensor_names(structure)
    assert names == scope["encoder_tensors"]
    assert len(names) == 18
    for name in names:
        assert name.startswith(
            f"vision_model.encoder.layers.{structure['adapted_layer_index']}."
        ) or name.startswith("vision_model.post_layernorm.")
    assert scope["encoder_parameter_count"] == 7089408
    assert scope["projector_parameter_count"] == 11542016
    assert scope["trainable_parameter_count"] == 18631424
    assert len(scope["trainable_tensors"]) == 24
    assert scope["optimizer_owned_parameters"] == 24
    assert scope["optimizer_param_groups"] == 1
    assert scope["encoder_update_bounds"] == {
        "max_abs_update": 1e-4, "max_delta_l2": 0.25}
    assert assessment["scope"]["trainable_parameter_count"] == 18631424
    assert assessment["scope"]["decoder_gradient_tensors"] == 0


def test_one_axis_changes_and_every_other_axis_is_identical(
        config, smaller):
    for block in ("prepared", "schedule", "initialization", "optimizer",
                  "update_batch", "readouts", "accounting", "limits",
                  "policy", "access"):
        assert config[block] == smaller[block], block
    assert config["optimizer"]["lr"] == smaller["optimizer"]["lr"] == 3e-05
    assert config["optimizer"]["wrapper"] == "NativeFusedAdamW"
    assert config["schedule"]["updates_planned"] == 1
    assert config["update_batch"]["example_ids"] == (
        smaller["update_batch"]["example_ids"])
    assert config["update_batch"]["loss_metric"] == "mean_response_nll"
    assert config["scope_axis"]["changed"] == "gradient_scope"
    assert config["scope_axis"]["unchanged"] == [
        "parent", "encoder_weights", "data", "update_batch", "seed",
        "initialization", "lr", "optimizer_hyperparameters", "loss",
        "decoding", "readout_protocol", "generation_policy", "budget",
        "closed_roles",
    ]
    assert config["distinctness"]["single_changed_axis"] == (
        "gradient_scope_encoder_last_block_and_post_layernorm")
    assert config["parent"]["model_path"] == smaller["source"]["model_path"]
    assert config["parent"]["model_revision"] == (
        smaller["source"]["model_revision"])
    assert config["parent"]["vision_source_manifest_sha256"] == (
        smaller["source"]["vision_source_manifest_sha256"])


def test_budget_is_identical_and_never_larger(config, smaller, assessment):
    budget = config["budget"]
    assert budget["relation_to_smaller_step"] == "identical_not_larger"
    assert budget["limits"] == smaller["limits"] == config["limits"]
    assert budget["campaign_cumulative"] == (
        smaller["accounting"]["campaign_cumulative"])
    assert budget["expected_measured_work"] == (
        smaller["readouts"]["expected_measured_work"])
    for key in ("additional_forward_calls", "additional_backward_calls",
                "additional_optimizer_steps", "additional_wall_seconds",
                "additional_artifact_bytes"):
        assert budget[key] == 0
    assert budget["wall_limit_seconds"] == 1200
    assert budget["artifact_bytes"] == 33554432
    assert budget["encoder_backward_included_in_the_same_four_backward_calls"]
    assert config["limits"]["max_decoder_forwards"] == 4824
    assert config["limits"]["max_visual_encoder_calls"] == 168
    assert config["limits"]["max_total_backward_calls"] == 4
    assert config["limits"]["max_optimizer_steps"] == 1
    assert config["limits"]["artifact_bytes"] == 33554432
    assert assessment["budget"]["additional_forward_calls"] == 0
    assert assessment["budget"]["limits"] == smaller["limits"]
    assert config["accounting"]["campaign_cumulative"] == (
        smaller["accounting"]["campaign_cumulative"])


def test_loss_gates_and_decision_remain_unchanged(config, smaller):
    loss = config["loss_contract"]
    assert loss["name"] == "response_only_mean_nll"
    assert loss["metric"] == smaller["update_batch"]["loss_metric"]
    assert loss["auxiliary_terms"] == []
    assert loss["auxiliary_loss_weight"] == 0.0
    assert loss["new_loss_introduced"] is False
    optimization = config["gates"]["optimization"]
    smaller_optimization = smaller["gates"]["optimization"]
    for key in ("update_count", "backward_calls", "optimizer_steps",
                "backbone_gradient_tensors", "finite_losses",
                "finite_grad_norm", "max_update_batch_mean_nll_delta",
                "max_wider_answer_mean_nll_delta",
                "max_wider_caption_mean_nll_delta", "delta_definition",
                "loss_movement_is_not_conversion"):
        assert optimization[key] == smaller_optimization[key]
    assert optimization["max_encoder_abs_update"] == 1e-4
    assert optimization["max_encoder_delta_l2"] == 0.25
    assert optimization["optimizer_owned_tensors"] == 24
    assert optimization["encoder_tensors"] == 18
    functional = config["gates"]["functional_fit"]
    smaller_functional = smaller["gates"]["functional_fit"]
    for key, value in smaller_functional.items():
        if key == "status_on_fail":
            assert functional[key] == "coadaptation_functional_fit_failed"
        else:
            assert functional[key] == value
    decision = config["gates"]["decision"]
    for key, value in smaller["gates"]["decision"].items():
        assert decision[key] == value
    assert decision["more_coadaptation_authorized"] is False


def test_claims_and_distinctness_forbid_expansion(config):
    claims = config["claims"]
    for flag in ("smaller_step_retry", "encoder_swapped", "new_loss_introduced",
                 "budget_increased", "generated_answer_conversion_claim",
                 "rank_improvement_is_conversion", "development_authorized",
                 "heldout_authorized", "v0_authorized",
                 "semantic_audit_authorized", "promotion", "retry_authorized",
                 "native_execution_authorized"):
        assert claims[flag] is False, flag
    distinctness = config["distinctness"]
    assert distinctness["not_a_retry_of"] == "vision_smaller_step_lr3e5_v1"
    assert distinctness["retry_of_completed_or_failed_arm"] is False
    assert distinctness["same_generated_tokens"] == 3800
    assert distinctness["same_update_count"] == 1
    assert distinctness["predictions"].keys() == {
        "frozen_features_sufficient", "frozen_features_insufficient"}
    assert "not a disguised" in config["purpose"].lower() or (
        "not a retry" in config["purpose"].lower())
    assert len(distinctness["why_distinct_hypothesis"]) > 200
    assert len(distinctness["why_not_an_expansion"]) > 200
    assert "no further co-adaptation retry" in (
        distinctness["closure_on_failure"].lower())


def test_control_is_not_eligible_and_unauthorized(control, assessment):
    precondition = assessment["precondition"]
    assert precondition["required"] is True
    assert precondition["smaller_step_report"]["present"] is False
    assert precondition["smaller_step_report"]["status"] == "absent"
    assert precondition["diagnosis"]["present"] is False
    assert precondition["diagnosis"]["violations"] == [
        "diagnosis_report_missing"]
    assert precondition["eligible"] is False
    assert assessment["decision"]["eligible"] is False
    unsatisfied = [item["id"] for item in assessment["remaining_requirements"]
                   if not item["satisfied_by_preparation"]]
    assert unsatisfied == [
        "smaller_step_failure_preserved",
        "frozen_feature_diagnosis_preserved",
        "coadaptation_runner_implemented",
        "separate_authorization_receipt_issued",
        "protocol_commit_recorded",
        "review_launch_gate_passed",
        "exclusive_worker_and_capacity_guard",
        "native_execution_within_identical_budget",
    ]
    fresh = next(item for item in assessment["remaining_requirements"]
                 if item["id"] == "fresh_output_available")
    assert fresh["satisfied_by_preparation"] is True
    assert not (ROOT / OUTPUT).exists()
    authorization = assessment["authorization"]
    assert authorization["present"] is False
    assert authorization["status"] == "absent"
    assert authorization["protocol_commit"] is None
    assert authorization["commit_bound"] is False
    assert not (ROOT / config_receipt(control)).exists()
    assert not (ROOT / RUNNER).exists()
    assert assessment["decision"]["reason"].startswith("preparation_complete")


def config_receipt(control):
    config = json.loads((ROOT / CONFIG).read_text())
    return config["authorization"]["receipt_path"]


def test_receipts_cannot_upgrade_an_ineligible_control(
        control, config, smaller, tmp_path):
    receipt = {
        "format": config["authorization"]["receipt_format"],
        "status": "authorized",
        "protocol_name": config["name"],
        "protocol_path": CONFIG,
        "protocol_commit": REVIEWED_COMMIT,
        "config_sha256": control._sha256_file(ROOT / CONFIG),
        "native_execution_authorized": True,
        "dev_authorized": False,
        "heldout_authorized": False,
        "v0_authorized": False,
        "semantic_audit_authorized": False,
        "promotion_authorized": False,
        "retry_authorized": False,
        "issued_at_utc": "2026-10-27T00:00:00+00:00",
    }
    target = tmp_path / config["authorization"]["receipt_path"]
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(receipt) + "\n")
    ineligible = {"eligible": False}
    state = control._authorization_state(
        config, tmp_path, ineligible, smaller,
        control._sha256_file(ROOT / CONFIG), git_root=ROOT)
    assert state["present"] is True
    assert state["status"] == "present_invalid"
    assert "receipt_issued_before_diagnosis_precondition" in state["violations"]
    assert "receipt_commit_does_not_contain_protocol_bytes" in (
        state["violations"])
    eligible = {"eligible": True}
    state = control._authorization_state(
        config, tmp_path, eligible, smaller,
        control._sha256_file(ROOT / CONFIG), git_root=ROOT)
    assert "receipt_issued_before_diagnosis_precondition" not in (
        state["violations"])
    assert state["commit_bound"] is False
    assert state["status"] == "present_invalid"
    for mutate, expected in (
            (lambda item: item.update(dev_authorized=True),
             "receipt_opens_closed_field:dev_authorized"),
            (lambda item: item.update(native_execution_authorized=False),
             "receipt_does_not_authorize_native_execution"),
            (lambda item: item.update(status="proposed"),
             "receipt_status_not_authorized"),
            (lambda item: item.update(config_sha256="0" * 64),
             "receipt_config_sha256_differs")):
        tampered = copy.deepcopy(receipt)
        mutate(tampered)
        target.write_text(json.dumps(tampered) + "\n")
        state = control._authorization_state(
            config, tmp_path, eligible, smaller,
            control._sha256_file(ROOT / CONFIG), git_root=ROOT)
        assert state["status"] == "present_invalid"
        assert expected in state["violations"]


@pytest.mark.parametrize("section, mutate", [
    ("hypothesis", lambda item: item.update(lr=1e-04)),
    ("optimizer", lambda item: item.update(lr=1e-04)),
    ("optimizer", lambda item: item.update(weight_decay=0.02)),
    ("schedule", lambda item: item.update(updates_planned=2)),
    ("update_batch",
     lambda item: item["example_ids"].append("answer:vba2-w001-p0-a")),
    ("readouts",
     lambda item: item["wider"].update(records_per_readout=105)),
    ("budget", lambda item: item.update(additional_forward_calls=1)),
    ("budget", lambda item: item.update(additional_backward_calls=1)),
    ("budget", lambda item: item.update(additional_artifact_bytes=1)),
    ("budget", lambda item: item["limits"].update(decoder_forward_calls=4825)),
    ("limits", lambda item: item.update(max_decoder_forwards=4825)),
    ("limits", lambda item: item.update(artifact_bytes=33554433)),
    ("gradient_scope",
     lambda item: item["trainable_tensors"].append(
         "vision_model.encoder.layers.10.mlp.fc1.weight")),
    ("gradient_scope",
     lambda item: item["encoder_update_bounds"].update(max_abs_update=1e-3)),
    ("gradient_scope",
     lambda item: item["encoder_structure"].update(adapted_layer_index=10)),
    ("gradient_scope",
     lambda item: item.update(backbone_gradient_tensors_allowed=1)),
    ("loss_contract",
     lambda item: item.update(auxiliary_terms=["kl_divergence"])),
    ("loss_contract", lambda item: item.update(auxiliary_loss_weight=0.1)),
    ("loss_contract", lambda item: item.update(new_loss_introduced=True)),
    ("gates",
     lambda item: item["optimization"].update(
         max_update_batch_mean_nll_delta=0.5)),
    ("gates",
     lambda item: item["functional_fit"].update(answer_correct_min=0)),
    ("gates",
     lambda item: item["functional_fit"].update(status_on_fail="passed")),
    ("gates",
     lambda item: item["decision"].update(retry_authorized=True)),
    ("gates",
     lambda item: item["decision"].update(
         more_coadaptation_authorized=True)),
    ("parent", lambda item: item.update(encoder_swap=True)),
    ("parent", lambda item: item.update(model_revision="0" * 40)),
    ("precondition",
     lambda item: item["accepted_smaller_step_statuses"].append(
         "stopped_engineering_failure")),
    ("precondition",
     lambda item: item["required_excluded_causes"].remove("step_size")),
    ("authorization", lambda item: item.update(native_execution_authorized=True)),
    ("authorization", lambda item: item.update(protocol_commit="0" * 40)),
    ("implementation", lambda item: item.update(runner_implemented=True)),
    ("claims", lambda item: item.update(new_loss_introduced=True)),
    ("claims", lambda item: item.update(budget_increased=True)),
    ("distinctness", lambda item: item.update(
         retry_of_completed_or_failed_arm=True)),
    ("distinctness",
     lambda item: item["predictions"].pop("frozen_features_sufficient")),
    ("scope_axis", lambda item: item.update(budget_increase=True)),
    ("scope_axis", lambda item: item.update(new_loss=True)),
    ("scope_axis", lambda item: item.update(encoder_swap=True)),
    ("budget", lambda item: item.update(relation_to_smaller_step="larger")),
])
def test_expansion_or_tampering_is_refused(
        control, config, tmp_path, section, mutate):
    tampered = copy.deepcopy(config)
    mutate(tampered[section])
    target = tmp_path / "coadaptation.json"
    target.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(control.CoadaptationError):
        control.assess(target, root=ROOT)


def test_stale_and_malformed_assessments_are_refused(
        control, assessment, tmp_path):
    clone = copy.deepcopy(assessment)
    clone["budget"]["additional_forward_calls"] = 1
    clone["assessment_sha256"] = control._assessment_sha256(clone)
    with pytest.raises(control.StaleCoadaptationError):
        control.verify_assessment(report=clone, root=ROOT)
    bad_hash = copy.deepcopy(assessment)
    bad_hash["assessment_sha256"] = "0" * 64
    with pytest.raises(control.CoadaptationError):
        control.verify_assessment(report=bad_hash, root=ROOT)
    with pytest.raises(control.CoadaptationError):
        control.verify_assessment(report_path=tmp_path / "missing.json",
                                 root=ROOT)
