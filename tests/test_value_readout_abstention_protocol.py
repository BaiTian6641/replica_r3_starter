"""Contracts for the chronological value-readout abstention preparation.

The tests use the generated CPU fixture and the local software-integration
runner.  They load no model, tokenizer, native runtime or download and write no
historical receipt.  Positive evidence is produced by the runner; admission is
then exercised independently, including missing, tampered and cross-state cases.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "value_readout_abstention_protocol_v1.json"
PREPARED = ROOT / "data" / "value_readout_abstention_20261027" / "fixture02"

_SPEC = importlib.util.spec_from_file_location(
    "value_readout_abstention_contract",
    ROOT / "scripts" / "value_readout_abstention_contract_v1.py")
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)

from replica_lab.value_readout_abstention import (  # noqa: E402
    enforce_artifact_budget, run_controls, run_development, run_fit,
    run_heldout,
)


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


def _copy_evidence(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / "evidence"
    shutil.copytree(source, target)
    return target


def _refresh_budget(root: Path) -> dict:
    return enforce_artifact_budget(root, _config(), context="test_refresh")


def _read_receipt(path: Path, check: str = "receipt") -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise contract.ProtocolError(
            check, f"receipt is missing or unreadable: {path}") from error


def _receipt(root: Path) -> dict:
    return _read_receipt(root / "fit_receipt.json", "fit_receipt")


def _dev_receipt(root: Path) -> dict:
    return _read_receipt(root / "dev_receipt.json", "development_receipt")


def _heldout_receipt(root: Path) -> dict:
    path = Path(root) / "heldout_receipt.json"
    if not path.is_file():
        # Let the contract reject the missing receipt after development
        # admission, so no held-out label is parsed.
        return {}
    return _read_receipt(path, "heldout_receipt")


def _write_receipt(root: Path, receipt: dict) -> dict:
    sealed = contract.seal_receipt(receipt)
    contract.write_json(root / "fit_receipt.json", sealed)
    return sealed


def _fit_report(root: Path) -> dict:
    return json.loads((root / "fit" / "fit_report.json").read_text(encoding="utf-8"))


def _write_fit_report(root: Path, report: dict) -> None:
    contract.write_json(root / "fit" / "fit_report.json", report)


def _refresh_fit_artifacts(root: Path) -> dict:
    """Rebind changed evidence bytes so a negative case fails admission later."""
    predictions_path = root / "fit" / "fit_predictions.jsonl"
    predictions_sha = contract.sha256_file(predictions_path)
    report = _fit_report(root)
    report["predictions_sha256"] = predictions_sha
    _write_fit_report(root, report)
    checkpoint_path = root / "fit" / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=True)
    checkpoint["predictions_sha256"] = predictions_sha
    torch.save(checkpoint, checkpoint_path)
    receipt = _receipt(root)
    receipt["artifacts"] = {
        "checkpoint_sha256": contract.sha256_file(checkpoint_path),
        "fit_report_sha256": contract.sha256_file(
            root / "fit" / "fit_report.json"),
        "fit_predictions_sha256": predictions_sha,
    }
    return _write_receipt(root, receipt)


def _fit_admission(root: Path, *, spy=None) -> dict:
    return contract.admit_fit_receipt(
        _receipt(root), PREPARED, CONFIG_PATH,
        evidence_dir=root / "fit", spy=spy)


def _vault(spy=None) -> contract.LabelVault:
    return contract.LabelVault(PREPARED, config_path=CONFIG_PATH, spy=spy)


# ---------------------------------------------------------------------------
# Fixture and config contracts
# ---------------------------------------------------------------------------


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
    assert config["admission"]["fit_receipt"]["created_by"] == \
        "scripts/run_value_readout_abstention_v1.py"
    assert config["admission"]["fit_receipt"]["exact_values"]["created_by"] == \
        "scripts/run_value_readout_abstention_v1.py"
    assert manifest["fit_receipt_requirements"]["exact_values"]["created_by"] == \
        "scripts/run_value_readout_abstention_v1.py"


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


def test_protocol_rejects_the_underprovisioned_old_cap():
    config = _config()
    assert config["budget"]["authorized_run_ceiling"][
        "artifact_bytes_max"] == 33554432
    config["budget"]["authorized_run_ceiling"]["artifact_bytes_max"] = 16777216
    with pytest.raises(contract.ProtocolError):
        contract.validate_config(config, CONFIG_PATH)
    config = _config()
    config["budget"]["authorized_run_ceiling"]["artifact_budget"][
        "headroom_bytes"] += 1
    with pytest.raises(contract.ProtocolError):
        contract.validate_config(config, CONFIG_PATH)


def test_evaluation_label_or_case_leak_mutations_fail_closed(tmp_path):
    config = _config()
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


# ---------------------------------------------------------------------------
# Construction audit, runtime precheck and label access
# ---------------------------------------------------------------------------


def test_runtime_precheck_parses_no_jsonl_and_audit_is_explicit(monkeypatch):
    calls: list[Path] = []
    original = contract.read_jsonl

    def spy(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(contract, "read_jsonl", spy)
    runtime = contract.runtime_precheck(PREPARED, CONFIG_PATH)
    assert calls == []
    assert runtime["dataset_audit"]["mode"] == "runtime_hash_precheck"
    assert runtime["dataset_audit"]["jsonl_rows_parsed"] == 0
    assert runtime["dataset_audit"]["evaluation_labels_parsed"] is False
    assert runtime["label_access"]["opened_splits"] == []
    assert runtime["label_access"]["dev_parsed"] is False
    assert runtime["label_access"]["heldout_parsed"] is False
    assert runtime["model_evaluation"]["performed"] is False

    audit = contract.validate_prepared(PREPARED, CONFIG_PATH)
    assert audit["dataset_audit"]["mode"] == "trusted_construction_audit"
    assert audit["dataset_audit"]["splits_parsed"] == [
        "train", "dev", "heldout"]
    assert audit["dataset_audit"]["evaluation_labels_parsed"] is True
    assert audit["dataset_audit"]["runtime_pre_admission_allowed"] is False
    assert audit["label_access"]["construction_audit_only"] is True


def test_label_vault_denies_before_admission_and_is_ordered(tmp_path):
    spy = contract.AccessSpy()
    vault = _vault(spy)
    with pytest.raises(contract.ProtocolError):
        vault.open_evaluation("dev", admission=None)
    with pytest.raises(contract.ProtocolError):
        vault.open_evaluation("dev", admission={"admitted": False})
    with pytest.raises(contract.ProtocolError):
        vault.open_evaluation("heldout", admission={"admitted": True,
                                                    "stage": "fit"})
    assert vault.opened == []
    assert spy.of_kind("label_parse") == []

    fit_root = tmp_path / "fit_run"
    run_fit(PREPARED, CONFIG_PATH, fit_root)
    fit = _fit_admission(fit_root)
    labels = vault.open_evaluation("dev", admission=fit)
    assert len(labels) == 10
    assert [row["split"] for row in vault.opened] == ["dev"]
    assert spy.of_kind("label_parse")[0]["admission_stage"] == "fit"
    cached = vault.open_evaluation("dev", admission=fit)
    assert cached == labels
    assert [row["split"] for row in vault.opened] == ["dev"]
    assert len(spy.of_kind("label_parse")) == 1
    with pytest.raises(contract.ProtocolError):
        vault.open_evaluation("heldout", admission=fit)
    assert [row["split"] for row in vault.opened] == ["dev"]


def test_native_entry_is_blocked_without_admission():
    spy = contract.AccessSpy()
    called: list[str] = []

    def loader():
        called.append("loaded")
        return "native"

    with pytest.raises(contract.ProtocolError):
        contract.require_admission_before_native(
            {"admitted": False}, loader=loader, spy=spy)
    with pytest.raises(contract.ProtocolError):
        contract.require_admission_before_native(
            {"admitted": True, "native_execution_allowed": False},
            loader=loader, spy=spy)
    assert called == []
    assert spy.of_kind("native_load") == []


# ---------------------------------------------------------------------------
# Positive evidence and independent fit admission
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fit_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("fit_evidence") / "fit_run"
    result = run_fit(PREPARED, CONFIG_PATH, root)
    assert result["status"] == "passed"
    return root


def test_positive_fit_admission_recomputes_every_case(fit_run):
    receipt = _receipt(fit_run)
    assert receipt["created_by"] == "scripts/run_value_readout_abstention_v1.py"
    spy = contract.AccessSpy()
    admitted = contract.admit_fit_receipt(
        receipt, PREPARED, CONFIG_PATH, evidence_dir=fit_run / "fit", spy=spy)
    assert admitted["admitted"] is True
    assert admitted["stage"] == "fit"
    assert admitted["optimizer_steps"] == 24
    assert admitted["completed_steps"] == 24
    assert admitted["event_clock_steps"] == 36
    assert admitted["per_case"]["count"] == 27
    assert admitted["per_case"]["covered"] == 27
    assert admitted["per_case"]["unique_ids"] is True
    assert admitted["per_case"]["counts"]["future_answer"]["correct"] == 6
    assert admitted["per_case"]["counts"]["unaffected_retention"]["correct"] == 3
    assert admitted["metrics"]["unknown_abstention_accuracy"] >= 0.9
    assert admitted["metrics"]["stale_value_rate"] <= 0.05
    assert admitted["heldout_opened"] is False
    assert admitted["dev_labels_read"] is False
    assert admitted["native_execution_allowed"] is False
    assert admitted["label_access"]["opened_splits"] == []
    assert admitted["label_access"]["training_metadata_parsed"] is True
    assert admitted["dataset_audit"]["mode"] == "runtime_hash_precheck"
    assert admitted["model_evaluation"]["kind"] == "preserved_fit_predictions"
    assert spy.of_kind("label_parse") == []
    assert spy.of_kind("native_load") == []
    assert admitted["artifact_budget"]["status"] == "within_budget"
    assert admitted["artifact_budget"]["cap_bytes"] == 33554432
    assert admitted["artifact_budget"]["measured_bytes"] <= 33554432
    assert {row["arm_id"] for row in admitted["control_fits"]} == {
        "shuffled_values", "wrong_world_values"}
    assert all(row["fit_receipt_binding"] is False
               for row in admitted["control_fits"])
    assert all(row["optimizer_steps"] == 24
               for row in admitted["control_fits"])


def test_fit_admission_refuses_missing_and_placeholder_artifacts(fit_run, tmp_path):
    receipt = _receipt(fit_run)
    with pytest.raises(contract.ProtocolError):
        contract.admit_fit_receipt(receipt, PREPARED, CONFIG_PATH)
    with pytest.raises(contract.ProtocolError):
        contract.admit_fit_receipt(
            receipt, PREPARED, CONFIG_PATH, evidence_dir=tmp_path / "missing")

    broken = deepcopy(receipt)
    broken["artifacts"] = {
        "checkpoint_sha256": "a" * 64,
        "fit_report_sha256": "b" * 64,
        "fit_predictions_sha256": "c" * 64,
    }
    broken = contract.seal_receipt(broken)
    with pytest.raises(contract.ProtocolError):
        contract.admit_fit_receipt(
            broken, PREPARED, CONFIG_PATH, evidence_dir=fit_run / "fit")


def test_fit_admission_refuses_tampered_report_metrics(fit_run, tmp_path):
    root = _copy_evidence(fit_run, tmp_path)
    report = _fit_report(root)
    report["metrics"]["future_answer_accuracy"] = 0.0
    _write_fit_report(root, report)
    receipt = _receipt(root)
    receipt["artifacts"]["fit_report_sha256"] = contract.sha256_file(
        root / "fit" / "fit_report.json")
    _write_receipt(root, receipt)
    with pytest.raises(contract.ProtocolError):
        _fit_admission(root)


def test_fit_admission_refuses_missing_and_duplicate_prediction_cases(
        fit_run, tmp_path):
    root = _copy_evidence(fit_run, tmp_path)
    rows = _rows(root / "fit" / "fit_predictions.jsonl")
    _write_rows(root / "fit" / "fit_predictions.jsonl", rows[:-1])
    _refresh_fit_artifacts(root)
    with pytest.raises(contract.ProtocolError):
        _fit_admission(root)

    root = _copy_evidence(fit_run, tmp_path / "duplicate")
    rows = _rows(root / "fit" / "fit_predictions.jsonl")
    rows.append(deepcopy(rows[0]))
    _write_rows(root / "fit" / "fit_predictions.jsonl", rows)
    _refresh_fit_artifacts(root)
    with pytest.raises(contract.ProtocolError):
        _fit_admission(root)


def test_fit_admission_refuses_cross_state_predictions(fit_run, tmp_path):
    root = _copy_evidence(fit_run, tmp_path)
    control = _rows(root / "fit" / "control_shuffled_values_predictions.jsonl")
    _write_rows(root / "fit" / "fit_predictions.jsonl", control)
    _refresh_fit_artifacts(root)
    with pytest.raises(contract.ProtocolError):
        _fit_admission(root)


def test_fit_admission_refuses_control_or_aggregate_receipt(fit_run, tmp_path):
    root = _copy_evidence(fit_run, tmp_path)
    checkpoint_path = root / "fit" / "checkpoint.pt"
    shutil.copyfile(root / "fit" / "control_shuffled_values.pt",
                    checkpoint_path)
    receipt = _receipt(root)
    receipt["artifacts"]["checkpoint_sha256"] = contract.sha256_file(
        checkpoint_path)
    _write_receipt(root, receipt)
    with pytest.raises(contract.ProtocolError):
        _fit_admission(root)

    root = _copy_evidence(fit_run, tmp_path / "arm")
    receipt = _receipt(root)
    receipt["optimizer_steps"] = 72
    _write_receipt(root, receipt)
    with pytest.raises(contract.ProtocolError):
        _fit_admission(root)

    root = _copy_evidence(fit_run, tmp_path / "report")
    report = _fit_report(root)
    report["arm_id"] = "shuffled_values"
    _write_fit_report(root, report)
    receipt = _receipt(root)
    receipt["artifacts"]["fit_report_sha256"] = contract.sha256_file(
        root / "fit" / "fit_report.json")
    _write_receipt(root, receipt)
    with pytest.raises(contract.ProtocolError):
        _fit_admission(root)


def test_fit_admission_refuses_bad_update_records(fit_run, tmp_path):
    root = _copy_evidence(fit_run, tmp_path)
    report = _fit_report(root)
    report["update_records"] = report["update_records"][:-1]
    _write_fit_report(root, report)
    receipt = _receipt(root)
    receipt["artifacts"]["fit_report_sha256"] = contract.sha256_file(
        root / "fit" / "fit_report.json")
    _write_receipt(root, receipt)
    with pytest.raises(contract.ProtocolError):
        _fit_admission(root)


# ---------------------------------------------------------------------------
# Development and held-out evidence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluation_runs(fit_run, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("evaluation_evidence")
    development = run_development(
        PREPARED, CONFIG_PATH, fit_run, root / "dev_run")
    heldout = run_heldout(
        PREPARED, CONFIG_PATH, fit_run, root / "dev_run", root / "heldout_run")
    controls = run_controls(
        PREPARED, CONFIG_PATH, fit_run, root / "controls_run", split="dev")
    assert development["status"] == "passed"
    assert heldout["status"] == "passed"
    assert controls["status"] == "passed"
    return {"fit": fit_run, "dev": root / "dev_run",
            "heldout": root / "heldout_run", "controls": root / "controls_run"}


def _admit_dev(evidence: dict, *, spy=None, vault=None) -> dict:
    return contract.admit_development(
        _dev_receipt(evidence["dev"]), fit_receipt=_receipt(evidence["fit"]),
        prepared_dir=PREPARED, config_path=CONFIG_PATH,
        evidence_dir=evidence["dev"] / "dev",
        fit_evidence_dir=evidence["fit"] / "fit",
        label_vault=vault or _vault(spy), spy=spy)


def _admit_heldout(evidence: dict, *, spy=None, vault=None) -> dict:
    return contract.admit_heldout(
        _heldout_receipt(evidence["heldout"]),
        development_receipt=_dev_receipt(evidence["dev"]),
        fit_receipt=_receipt(evidence["fit"]), prepared_dir=PREPARED,
        config_path=CONFIG_PATH,
        evidence_dir=evidence["heldout"] / "heldout",
        dev_evidence_dir=evidence["dev"] / "dev",
        fit_evidence_dir=evidence["fit"] / "fit",
        label_vault=vault or _vault(spy), spy=spy)


def test_development_and_heldout_admission_recompute_per_case(evaluation_runs):
    assert _dev_receipt(evaluation_runs["dev"])["created_by"] == \
        "scripts/run_value_readout_abstention_v1.py"
    assert _heldout_receipt(evaluation_runs["heldout"])["created_by"] == \
        "scripts/run_value_readout_abstention_v1.py"
    spy = contract.AccessSpy()
    vault = _vault(spy)
    development = _admit_dev(evaluation_runs, spy=spy, vault=vault)
    assert development["stage"] == "development"
    assert development["per_case"]["count"] == 10
    assert development["metrics"]["future_answer_accuracy"] == 1.0
    assert development["label_access"]["dev_parsed"] is True
    assert development["label_access"]["heldout_parsed"] is False
    assert [row["split"] for row in vault.opened] == ["dev"]
    heldout = _admit_heldout(evaluation_runs, spy=spy, vault=vault)
    assert heldout["stage"] == "heldout"
    assert heldout["per_case"]["count"] == 10
    assert heldout["heldout_opened"] is True
    assert heldout["label_access"]["opened_splits"] == ["dev", "heldout"]
    assert [row["split"] for row in vault.opened] == ["dev", "heldout"]
    assert all(event["admission_stage"] in ("fit", "development")
               for event in spy.of_kind("label_parse"))


def test_development_refuses_claimed_metrics_after_label_recompute(evaluation_runs,
                                                                  tmp_path):
    evidence = {
        "fit": evaluation_runs["fit"],
        "dev": _copy_evidence(evaluation_runs["dev"], tmp_path),
    }
    receipt = _dev_receipt(evidence["dev"])
    receipt["dev_metrics"]["unknown_abstention_accuracy"] = 0.0
    contract.write_json(evidence["dev"] / "dev_receipt.json",
                        contract.seal_receipt(receipt))
    _refresh_budget(evidence["dev"])
    spy = contract.AccessSpy()
    with pytest.raises(contract.ProtocolError):
        _admit_dev(evidence, spy=spy)
    # Development labels are opened only after fit admission; the claimed
    # aggregate is rejected after independent per-case recomputation.
    assert [event["split"] for event in spy.of_kind("label_parse")] == ["dev"]


def test_failed_fit_receipt_reads_no_evaluation_labels(evaluation_runs, tmp_path):
    evidence = {
        "fit": _copy_evidence(evaluation_runs["fit"], tmp_path / "fit"),
        "dev": evaluation_runs["dev"],
    }
    checkpoint_path = evidence["fit"] / "fit" / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=True)
    checkpoint["model_state"]["selected_expert.gate"] = (
        checkpoint["model_state"]["selected_expert.gate"] + 1.0)
    torch.save(checkpoint, checkpoint_path)
    spy = contract.AccessSpy()
    with pytest.raises(contract.ProtocolError):
        _admit_dev(evidence, spy=spy)
    assert spy.of_kind("label_parse") == []
    assert spy.of_kind("native_load") == []


def test_heldout_is_not_opened_when_development_fails(evaluation_runs, tmp_path):
    evidence = {
        "fit": evaluation_runs["fit"],
        "dev": _copy_evidence(evaluation_runs["dev"], tmp_path),
        "heldout": evaluation_runs["heldout"],
    }
    receipt = _dev_receipt(evidence["dev"])
    receipt["dev_metrics"]["unknown_abstention_accuracy"] = 0.0
    contract.write_json(evidence["dev"] / "dev_receipt.json",
                        contract.seal_receipt(receipt))
    _refresh_budget(evidence["dev"])
    spy = contract.AccessSpy()
    vault = _vault(spy)
    with pytest.raises(contract.ProtocolError):
        _admit_heldout(evidence, spy=spy, vault=vault)
    assert [row["split"] for row in vault.opened] == ["dev"]
    assert all(event["split"] != "heldout" for event in spy.of_kind("label_parse"))


def test_heldout_missing_evidence_refuses_before_heldout_labels(evaluation_runs,
                                                               tmp_path):
    evidence = {
        "fit": evaluation_runs["fit"],
        "dev": evaluation_runs["dev"],
        "heldout": tmp_path / "missing",
    }
    spy = contract.AccessSpy()
    vault = _vault(spy)
    with pytest.raises(contract.ProtocolError):
        _admit_heldout(evidence, spy=spy, vault=vault)
    assert [row["split"] for row in vault.opened] == ["dev"]
    assert all(event["split"] != "heldout" for event in spy.of_kind("label_parse"))


# ---------------------------------------------------------------------------
# Review-only preflight
# ---------------------------------------------------------------------------


def _authorization() -> dict:
    config = _config()
    runtime = contract.runtime_precheck(PREPARED, CONFIG_PATH)
    body = {
        "format": contract.AUTHORIZATION_FORMAT,
        "protocol_id": contract.PROTOCOL_ID,
        "config_sha256": runtime["config_sha256"],
        "prepared_manifest_sha256": runtime["prepared_manifest_sha256"],
        "authorized_by": "reviewer",
        "native_execution_allowed": True,
        "single_attempt": True,
    }
    assert set(body) | {"receipt_sha256"} == set(
        config["launch"]["authorization_fields"])
    return contract.seal_receipt(body)


def test_preflight_is_review_only_and_native_launch_refuses(tmp_path):
    review = contract.preflight_report(
        config_path=CONFIG_PATH, prepared_dir=PREPARED,
        receipt_path=tmp_path / "preflight_review.json")
    assert review["exit_code"] == 0
    assert review["native_execution"] is False
    assert review["launch_status"] == "not_requested"
    assert review["dataset_audit"]["mode"] == "runtime_hash_precheck"
    assert review["label_access"]["opened_splits"] == []
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
