"""Contracts for the chronological value-readout abstention preparation.

The tests use only the generated CPU fixture and standard library.  They load
no model, tokenizer or native runner and write no historical receipt.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "value_readout_abstention_protocol_v1.json"
PREPARED = ROOT / "data" / "value_readout_abstention_20261027" / "fixture01"

_SPEC = importlib.util.spec_from_file_location(
    "value_readout_abstention_contract",
    ROOT / "scripts" / "value_readout_abstention_contract_v1.py")
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "fixture"
    shutil.copytree(PREPARED, target)
    return target


def _reseal(prepared: Path) -> None:
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {
        name: contract.sha256_file(prepared / name)
        for name in manifest["files"]}
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = contract.sha256_json(manifest)
    contract.write_json(manifest_path, manifest)


def _write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_jsonl(path, rows)


def _fit_metrics(config: dict, **overrides) -> dict:
    thresholds = config["evaluation"]["fit"]["thresholds"]
    metrics = {
        "future_answer_accuracy": 1.0,
        "correction_accuracy": 1.0,
        "unaffected_retention_accuracy": 1.0,
        "unknown_abstention_accuracy": 1.0,
        "wrong_world_abstention_accuracy": 1.0,
        "stale_value_rate": 0.0,
    }
    metrics.update(overrides)
    assert set(metrics) == set(contract._metric_names_from_thresholds(thresholds))
    return metrics


def _fit_receipt() -> dict:
    config = _config()
    report = contract.validate_prepared(PREPARED, CONFIG_PATH)
    body = {
        "format": contract.FIT_RECEIPT_FORMAT,
        "protocol_id": contract.PROTOCOL_ID,
        "status": "passed",
        "mode": "fit",
        "created_by": "future_authorized_runner",
        "config_sha256": report["config_sha256"],
        "prepared_manifest_sha256": report["prepared_manifest_sha256"],
        "seed": config["transaction"]["seed"],
        "optimizer": "NativeFusedAdamW",
        "train_case_count": 15,
        "replay_case_count": 12,
        "optimizer_steps": 24,
        "losses": list(contract.LOSS_NAMES),
        "trainable_parameters": list(config["architecture"]["trainable_parameters"]),
        "earlier_write_gradient": False,
        "heldout_opened": False,
        "dev_labels_read": False,
        "tuning_performed": False,
        "artifacts": {"checkpoint_sha256": "a" * 64,
                      "fit_report_sha256": "b" * 64},
        "fit_gates": {
            "all_declared_gates_passed": True,
            "local_associative_loss_gate": False,
            "metrics": _fit_metrics(config),
            "thresholds": dict(config["evaluation"]["fit"]["thresholds"]),
        },
    }
    return contract.seal_receipt(body)


def _dev_receipt(fit: dict) -> dict:
    config = _config()
    body = {
        "format": contract.DEVELOPMENT_RECEIPT_FORMAT,
        "protocol_id": contract.PROTOCOL_ID,
        "status": "passed",
        "mode": "dev-eval",
        "created_by": "future_authorized_runner",
        "attempt": 1,
        "fit_receipt_sha256": fit["receipt_sha256"],
        "config_sha256": fit["config_sha256"],
        "prepared_manifest_sha256": fit["prepared_manifest_sha256"],
        "heldout_opened": False,
        "tuning_performed": False,
        "all_declared_gates_passed": True,
        "dev_metrics": _fit_metrics(config, stale_value_rate=0.0),
    }
    return contract.seal_receipt(body)


def _heldout_receipt(fit: dict, dev: dict) -> dict:
    config = _config()
    report = contract.validate_prepared(PREPARED, CONFIG_PATH)
    body = {
        "format": contract.HELDOUT_RECEIPT_FORMAT,
        "protocol_id": contract.PROTOCOL_ID,
        "status": "passed",
        "mode": "heldout-eval",
        "created_by": "future_authorized_runner",
        "attempt": 1,
        "dev_receipt_sha256": dev["receipt_sha256"],
        "config_sha256": report["config_sha256"],
        "prepared_manifest_sha256": report["prepared_manifest_sha256"],
        "tuning_performed": False,
        "heldout_metrics": _fit_metrics(config),
    }
    return contract.seal_receipt(body)


def _authorization() -> dict:
    config = _config()
    report = contract.validate_prepared(PREPARED, CONFIG_PATH)
    body = {
        "format": contract.AUTHORIZATION_FORMAT,
        "protocol_id": contract.PROTOCOL_ID,
        "config_sha256": report["config_sha256"],
        "prepared_manifest_sha256": report["prepared_manifest_sha256"],
        "authorized_by": "reviewer",
        "native_execution_allowed": True,
        "single_attempt": True,
    }
    assert set(body) | {"receipt_sha256"} == set(
        config["launch"]["authorization_fields"])
    return contract.seal_receipt(body)


def test_fixture_validates_deterministically_and_claims_no_native_work():
    first = contract.validate_prepared(PREPARED, CONFIG_PATH)
    second = contract.validate_prepared(PREPARED, CONFIG_PATH)
    assert first["status"] == "valid"
    assert first["validation_sha256"] == second["validation_sha256"]
    assert first["native_computation"] is False
    assert first["model_loaded"] is False
    assert first["tokenizer_loaded"] is False
    assert first["optimizer_updates"] == 0
    assert first["gradients"] == 0
    assert first["checkpoints_written"] == 0
    assert first["downloads"] == 0
    assert first["heldout_opened"] is False
    assert first["counts"]["teaching_cases"] == 24
    assert first["counts"]["optimizer_steps"] == 24


def test_chronology_labels_and_source_epochs_are_authoritative():
    manifest = _manifest()
    assert manifest["counts"]["events"] == 68
    assert manifest["counts"]["queries"] == 47
    assert manifest["splits"]["train"]["events"] == 36
    assert manifest["splits"]["dev"]["events"] == 16
    assert manifest["splits"]["heldout"]["events"] == 16
    labels = {row["case_id"]: row for row in _rows(PREPARED / "labels_dev.jsonl")}
    assert labels["case_dev_alpha_future"]["label_source_epoch"] == 0
    assert labels["case_dev_alpha_correction"]["label_source_epoch"] == 3
    assert labels["case_dev_alpha_correction"]["stale_value"] == \
        "value_dev_alpha_amber"
    assert labels["case_dev_alpha_retention"]["label_source_epoch"] == 2
    assert labels["case_dev_alpha_unknown"]["label_source_epoch"] == 6
    assert labels["case_dev_alpha_wrong_world"]["label_source_epoch"] == 7
    wrong = labels["case_dev_alpha_wrong_world"]
    assert wrong["target"]["token"] == "UNKNOWN"
    assert wrong["rejected_world_id"] == "world_dev_beta"
    assert wrong["rejected_world_id"] != "world_dev_alpha"


def test_training_replay_is_disjoint_and_one_step_per_event():
    manifest = _manifest()
    assert manifest["disjointness"]["train_cases_disjoint_from_dev"]
    assert manifest["disjointness"]["replay_cases_disjoint_from_dev"]
    assert manifest["disjointness"]["training_semantics_disjoint_from_evaluation"]
    episodes = _rows(PREPARED / "episodes.jsonl")
    teaching = [row for row in episodes
                if row["kind"] == "query" and row["optimizer_steps"] == 1]
    assert len(teaching) == 24
    assert all(row["gradient"] is True for row in teaching)
    retention = [row for row in episodes
                 if row["kind"] == "query"
                 and row["case_type"] == "unaffected_retention"]
    assert len(retention) == 7
    assert all(row["optimizer_steps"] == 0 and row["gradient"] is False
               for row in retention)
    evaluation = [row for row in episodes
                  if row["split"] in ("dev", "heldout") and row["kind"] == "query"]
    assert len(evaluation) == 20
    assert all(row["optimizer_steps"] == 0 and row["gradient"] is False
               for row in evaluation)
    for row in episodes:
        assert row["clock_step"] == row["epoch"]
        assert row["optimizer_steps"] in (0, 1)


def test_workload_derivation_names_three_trained_arms_and_fit_scope():
    config = _config()
    ceiling = config["budget"]["authorized_run_ceiling"]
    derivation = ceiling["workload_derivation"]
    assert [arm["arm_id"] for arm in derivation["arms"]] == [
        "learned_value_readout", "shuffled_values", "wrong_world_values"]
    assert [arm["phase"] for arm in derivation["arms"]] == [
        "fit_treatment", "control_fit", "control_fit"]
    assert [arm["optimizer_steps"] for arm in derivation["arms"]] == [24, 24, 24]
    assert sum(arm["optimizer_steps"] for arm in derivation["arms"]) == 72
    assert derivation["fit_receipt_optimizer_steps"] == 24
    assert derivation["control_fit_optimizer_steps"] == 48
    assert derivation["total_optimizer_steps"] == 72
    assert derivation["independent_arm_clocks"] == 3
    assert derivation["shared_clock_across_arms"] is False
    assert derivation["extra_steps_per_event"] == 0
    assert derivation["readout_forward_calls_derivation"] == {
        "training_forwards": 72,
        "evaluation_forwards": 140,
        "retention_diagnostic_forwards": 9,
        "total": 221,
    }
    assert ceiling["optimizer_updates_total"] == 72
    assert ceiling["backward_calls_total"] == 72
    assert ceiling["readout_forward_calls_total"] == 221
    assert ceiling["write_events_total"] == 105
    assert ceiling["checkpoint_writes"] == 3
    scope = config["admission"]["fit_receipt"]["scope"]
    assert scope["arm_id"] == "learned_value_readout"
    assert scope["optimizer_steps"] == 24
    assert scope["excluded_trained_controls"] == [
        "shuffled_values", "wrong_world_values"]
    manifest = _manifest()
    assert manifest["workload_derivation"] == derivation
    assert manifest["planned_workload"]["optimizer_updates_total"] == 72
    assert manifest["fit_receipt_requirements"]["exact_values"][
        "optimizer_steps"] == 24
    assert manifest["fit_receipt_requirements"]["scope"]["arm_id"] == \
        "learned_value_readout"


def test_protocol_config_rejects_writer_only_and_expanded_orders():
    config = _config()
    assert contract.validate_config(config, CONFIG_PATH) is None
    mutations = [
        lambda value: value["losses"]["local_associative_loss"].update(
            {"enabled": True}),
        lambda value: value["losses"]["gradient_contract"].update(
            {"backprop_through_earlier_writes": True}),
        lambda value: value["evaluation"]["exact_context_reference"].update(
            {"treatment_required_to_beat": True}),
        lambda value: value["controls"]["arms"][4].update(
            {"state": value["controls"]["arms"][3]["state"]}),
        lambda value: value["evaluation"]["heldout"].update({"attempts": 2}),
        lambda value: value["launch"].update({"native_execution": True}),
        lambda value: value["transaction"].update({"optimizer": "Adam"}),
        lambda value: value["architecture"].update(
            {"no_architecture_expansion": False}),
        lambda value: value["budget"]["authorized_run_ceiling"][
            "workload_derivation"]["arms"].pop(),
        lambda value: value["budget"]["authorized_run_ceiling"][
            "workload_derivation"].update({"total_optimizer_steps": 24}),
        lambda value: value["admission"]["fit_receipt"]["scope"].update(
            {"arm_id": "shuffled_values"}),
    ]
    for mutate in mutations:
        broken = deepcopy(config)
        mutate(broken)
        with pytest.raises(contract.ProtocolError):
            contract.validate_config(broken, CONFIG_PATH)


def test_evaluation_label_or_case_leak_mutations_fail_closed(tmp_path):
    config = _config()
    manifest = _manifest()
    files = config["prepared"]["files"]

    prepared = _copy_fixture(tmp_path / "label")
    labels = _rows(prepared / files["labels_dev"])
    labels[0]["target"]["token"] = "UNKNOWN"
    labels[0]["target"]["kind"] = "status"
    labels[0]["target"]["index"] = 0
    _write_rows(prepared / files["labels_dev"], labels)
    _reseal(prepared)
    with pytest.raises(contract.ProtocolError):
        contract.validate_prepared(prepared, CONFIG_PATH)

    prepared = _copy_fixture(tmp_path / "leak")
    cases = _rows(prepared / files["cases_dev"])
    cases[0]["target"] = {"kind": "value", "token": "value_dev_alpha_amber",
                          "index": 1}
    _write_rows(prepared / files["cases_dev"], cases)
    _reseal(prepared)
    with pytest.raises(contract.ProtocolError):
        contract.validate_prepared(prepared, CONFIG_PATH)

    prepared = _copy_fixture(tmp_path / "hash")
    cases = _rows(prepared / files["cases_dev"])
    cases[0]["query_text"] = "tampered"
    _write_rows(prepared / files["cases_dev"], cases)
    with pytest.raises(contract.ProtocolError):
        contract.validate_prepared(prepared, CONFIG_PATH)


def test_duplicate_epoch_and_step_mutations_fail_closed(tmp_path):
    config = _config()
    files = config["prepared"]["files"]

    prepared = _copy_fixture(tmp_path / "epoch")
    events = _rows(prepared / files["episodes"])
    target = next(row for row in events
                  if row["split"] == "dev" and row["epoch"] == 1)
    target["epoch"] = 2
    target["clock_step"] = 2
    _write_rows(prepared / files["episodes"], events)
    _reseal(prepared)
    with pytest.raises(contract.ProtocolError):
        contract.validate_prepared(prepared, CONFIG_PATH)

    prepared = _copy_fixture(tmp_path / "steps")
    train_cases = _rows(prepared / files["cases_train"])
    teaching = next(row for row in train_cases if row["optimizer_steps"] == 1)
    teaching["optimizer_steps"] = 2
    _write_rows(prepared / files["cases_train"], train_cases)
    _reseal(prepared)
    with pytest.raises(contract.ProtocolError):
        contract.validate_prepared(prepared, CONFIG_PATH)


def test_opened_heldout_manifest_mutation_fails_closed(tmp_path):
    prepared = _copy_fixture(tmp_path / "heldout")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    manifest["heldout_opened"] = True
    contract.write_json(prepared / "manifest.json", manifest)
    _reseal(prepared)
    with pytest.raises(contract.ProtocolError):
        contract.validate_prepared(prepared, CONFIG_PATH)


def test_fit_receipt_admission_is_ordered_and_fail_closed():
    fit = _fit_receipt()
    admitted = contract.admit_fit_receipt(fit, PREPARED, CONFIG_PATH)
    assert admitted["admitted"] is True
    assert admitted["heldout_opened"] is False
    dev = _dev_receipt(fit)
    development = contract.admit_development(
        dev, fit_receipt=fit, prepared_dir=PREPARED, config_path=CONFIG_PATH)
    assert development["admitted"] is True
    heldout = _heldout_receipt(fit, dev)
    final = contract.admit_heldout(
        heldout, development_receipt=dev, fit_receipt=fit,
        prepared_dir=PREPARED, config_path=CONFIG_PATH)
    assert final["admitted"] is True

    broken = deepcopy(fit)
    broken["heldout_opened"] = True
    broken = contract.seal_receipt(broken)
    with pytest.raises(contract.ProtocolError):
        contract.admit_fit_receipt(broken, PREPARED, CONFIG_PATH)

    broken = deepcopy(fit)
    broken["fit_gates"]["metrics"]["future_answer_accuracy"] = 0.0
    broken = contract.seal_receipt(broken)
    with pytest.raises(contract.ProtocolError):
        contract.admit_fit_receipt(broken, PREPARED, CONFIG_PATH)

    broken = deepcopy(dev)
    broken["heldout_opened"] = True
    broken = contract.seal_receipt(broken)
    with pytest.raises(contract.ProtocolError):
        contract.admit_development(
            broken, fit_receipt=fit, prepared_dir=PREPARED,
            config_path=CONFIG_PATH)

    broken = deepcopy(heldout)
    broken["attempt"] = 2
    broken = contract.seal_receipt(broken)
    with pytest.raises(contract.ProtocolError):
        contract.admit_heldout(
            broken, development_receipt=dev, fit_receipt=fit,
            prepared_dir=PREPARED, config_path=CONFIG_PATH)


def test_preflight_is_review_only_and_native_launch_refuses(tmp_path):
    review = contract.preflight_report(
        config_path=CONFIG_PATH, prepared_dir=PREPARED,
        receipt_path=tmp_path / "preflight_review.json")
    assert review["exit_code"] == 0
    assert review["native_execution"] is False
    assert review["launch_status"] == "not_requested"
    assert (tmp_path / "preflight_review.json").is_file()

    refused = contract.preflight_report(
        config_path=CONFIG_PATH, prepared_dir=PREPARED,
        receipt_path=tmp_path / "preflight_launch.json", launch=True)
    assert refused["exit_code"] == 3
    assert refused["native_execution"] is False
    assert refused["launch_status"] == "refused_missing_authorization_receipt"

    authorization = _authorization()
    auth_path = tmp_path / "authorization.json"
    contract.write_json(auth_path, authorization)
    still_refused = contract.preflight_report(
        config_path=CONFIG_PATH, prepared_dir=PREPARED,
        receipt_path=tmp_path / "preflight_authorized.json",
        authorization_path=auth_path, launch=True)
    assert still_refused["exit_code"] == 3
    assert still_refused["native_execution"] is False
    assert still_refused["launch_status"] == "refused_preparation_only"
    assert still_refused["authorization_present"] is True

    broken = deepcopy(authorization)
    broken["native_execution_allowed"] = False
    broken = contract.seal_receipt(broken)
    contract.write_json(tmp_path / "authorization_bad.json", broken)
    with pytest.raises(contract.ProtocolError):
        contract.preflight_report(
            config_path=CONFIG_PATH, prepared_dir=PREPARED,
            receipt_path=tmp_path / "preflight_bad.json",
            authorization_path=tmp_path / "authorization_bad.json",
            launch=True)
