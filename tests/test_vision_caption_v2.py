"""Deterministic CPU contracts for the corrected v2 target/scoring protocol."""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest

import replica_lab.vision_balanced_alignment as vba
from conftest import require_local_artifacts

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/vision_balanced_alignment_v2.json"
FIXTURE_FORMAT = "replica.vision_balanced_alignment.evidence_fixture.v1"
EVIDENCE_FIXTURE_PATH = (
    ROOT / "reports/vision_balanced_alignment_20261026/"
    "evidence_fixture_v1.json")
RAW_RUN_REPORT_PATH = (
    ROOT / "runs/vision_balanced_alignment_20261026/train01/run_report.json")
RAW_RELOAD_REPORT_PATH = (
    ROOT / "runs/vision_balanced_alignment_20261026/reload01/"
    "reload_report.json")
RAW_REPORT_REQUIREMENTS = (
    "runs/vision_balanced_alignment_20261026/train01/run_report.json",
    "runs/vision_balanced_alignment_20261026/reload01/reload_report.json")
EXPECTED_COMPARISONS = (
    "heldout_answer_generations.jsonl",
    "heldout_caption_generations.jsonl",
    "retention_after.jsonl")


@lru_cache(maxsize=1)
def _config():
    return vba.load_config(CONFIG_PATH)


@lru_cache(maxsize=1)
def _evidence_fixture():
    return json.loads(EVIDENCE_FIXTURE_PATH.read_text())


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _validate_fixture_shape(fixture):
    """The committed fixture is self-describing; reject structural drift."""
    assert fixture["format"] == FIXTURE_FORMAT
    assert fixture["schema_version"] == 1
    assert set(fixture) == {
        "format", "schema_version", "description", "source", "protocol",
        "data", "run", "reload", "extension", "heldout_gates", "provenance"}
    source = fixture["source"]
    assert set(source) == {"run_report", "reload_report"}
    for name in ("run_report", "reload_report"):
        entry = source[name]
        assert set(entry) == {"path", "format", "bytes", "sha256"}
        assert entry["path"].startswith("runs/vision_balanced_alignment_")
        assert entry["path"].endswith(".json")
        assert entry["format"].startswith(
            "replica.vision_balanced_alignment.")
        assert isinstance(entry["bytes"], int) and entry["bytes"] > 0
        assert _is_sha256(entry["sha256"])
    assert _is_sha256(fixture["protocol"]["sha256"])
    assert _is_sha256(fixture["data"]["manifest_sha256"])
    assert _is_sha256(fixture["data"]["heldout_caption_case_ids_sha256"])
    assert _is_sha256(fixture["data"]["heldout_answer_case_ids_sha256"])

    run = fixture["run"]
    assert run["status"] == "completed_pending_reload"
    assert run["disposition"] == "diagnostic_only_no_promotion"
    assert run["base_saved"] is False
    assert run["promotion"] is False
    assert run["trainable_scope"].endswith(".")
    assert run["trainable_tensors"] > 0
    assert run["updates"] == 10
    assert run["total_examples"] >= run["unique_examples"] > 0

    reload_ = fixture["reload"]
    assert reload_["status"] == "completed"
    assert reload_["bitwise_equal"] is True
    assert reload_["zero_updates"] is True
    assert reload_["run_report_sha256"] == source["run_report"]["sha256"]
    assert reload_["protocol_sha256"] == fixture["protocol"]["sha256"]
    assert set(reload_["comparisons"]) == set(EXPECTED_COMPARISONS)
    assert all(reload_["comparisons"][name] is True
               for name in EXPECTED_COMPARISONS)

    extension = fixture["extension"]
    assert extension["format"] == "replica.minicpm5.extensions.v4"
    assert extension["label"] == "alignment"
    assert _is_sha256(extension["weights_sha256"])
    assert extension["bytes"] > 0
    assert extension["base_saved"] is False

    gates = fixture["heldout_gates"]
    assert gates["label"] == "heldout"
    assert gates["diagnostic_only"] is True
    assert gates["promotion"] is False
    assert gates["all_declared_gates_passed"] is False
    assert gates["gate_failures"]
    assert all(isinstance(name, str) and name
               for name in gates["gate_failures"])

    provenance = fixture["provenance"]
    assert provenance["raw_reports_integration_only"] is True
    assert provenance["raw_available_in_clean_checkout"] is False
    assert tuple(provenance["requirements"]) == RAW_REPORT_REQUIREMENTS
    assert provenance["requirements"] == [
        source["run_report"]["path"], source["reload_report"]["path"]]


@lru_cache(maxsize=1)
def _built():
    return vba.build_balanced_dataset(_config(), ROOT)


def _caption_inputs(role):
    built = _built()
    cases = built[f"{role}_caption_cases"]
    labels = built[f"{role}_caption_labels"]
    controls = {row["case_id"]: row
                for row in built[f"{role}_caption_controls"]}
    return cases, labels, controls


def _answer_inputs(role):
    built = _built()
    cases = built[f"{role}_answer_cases"]
    labels = built[f"{role}_answer_labels"]
    controls = {row["case_id"]: row
                for row in built[f"{role}_answer_controls"]}
    scored = [{**case, "label": {row["case_id"]: row
                                 for row in labels}[case["case_id"]]["answer"]}
              for case in cases]
    return scored, labels, controls


def test_v2_config_is_fresh_and_declares_corrected_contracts():
    config = _config()
    assert config["format"] == vba.CONFIG_FORMAT_V2
    assert config["version"] == vba.VERSION_V2
    assert config["seed"] != vba.load_config(
        ROOT / "configs/vision_balanced_alignment_v1.json")["seed"]
    assert config["procedural_overrides"]["id_prefix"] == "vba2"
    assert len(config["prior_consumed"]) == 2
    assert config["train"]["optimizer"] == "adamw_fused"
    assert config["train"]["fused"] is True
    assert config["scorer_contract"]["fixed_original_target"] is True
    assert config["scorer_contract"]["following_target"] is True
    assert config["preaudit"]["order"] == ["train_fit", "dev", "heldout"]
    assert config["preaudit"]["on_failure"] == "stop_before_heldout_access"
    assert config["schedule"]["alignment"]["updates"] != 10
    assert config["schedule"]["alignment"]["batch"] != 6
    assert config["train"]["lr"] != 0.0001
    built = _built()
    assert built["manifest"]["format"] == vba.MANIFEST_FORMAT_V2
    assert built["manifest"]["scorer_contract"] == config["scorer_contract"]
    assert built["manifest"]["fit_contract"] == config["fit_contract"]
    assert built["manifest"]["counts"]["train_examples"] < 46


def test_v1_semantics_and_evidence_are_preserved():
    v1 = vba.load_config(
        ROOT / "configs/vision_balanced_alignment_v1.json")
    assert v1["format"] == vba.CONFIG_FORMAT
    assert v1["version"] == vba.VERSION
    assert v1["schedule"]["alignment"]["updates"] == 10
    fixture = _evidence_fixture()
    _validate_fixture_shape(fixture)
    assert fixture["run"]["status"] == "completed_pending_reload"
    assert fixture["reload"]["status"] == "completed"
    assert fixture["reload"]["bitwise_equal"] is True
    assert fixture["heldout_gates"]["gate_failures"]


@require_local_artifacts(*RAW_REPORT_REQUIREMENTS)
@pytest.mark.integration
def test_raw_reports_match_committed_evidence_fixture():
    """Integration-only: the ignored raw reports agree with the fixture."""
    fixture = _evidence_fixture()
    _validate_fixture_shape(fixture)
    raw = {}
    for name, path in (("run_report", RAW_RUN_REPORT_PATH),
                       ("reload_report", RAW_RELOAD_REPORT_PATH)):
        payload = path.read_bytes()
        declared = fixture["source"][name]
        assert hashlib.sha256(payload).hexdigest() == declared["sha256"]
        assert len(payload) == declared["bytes"]
        report = json.loads(payload)
        assert report["format"] == declared["format"]
        raw[name] = report
    run_report = raw["run_report"]
    reload_report = raw["reload_report"]

    assert run_report["status"] == fixture["run"]["status"]
    assert run_report["disposition"] == fixture["run"]["disposition"]
    assert run_report["protocol_sha256"] == fixture["protocol"]["sha256"]
    assert run_report["data_manifest_sha256"] == \
        fixture["data"]["manifest_sha256"]
    seals = run_report["data_seals"]
    assert seals["heldout_caption_case_ids"] == \
        fixture["data"]["heldout_caption_case_ids_sha256"]
    assert seals["heldout_answer_case_ids"] == \
        fixture["data"]["heldout_answer_case_ids_sha256"]
    assert run_report["trainable_scope"] == fixture["run"]["trainable_scope"]
    assert run_report["trainable_tensors"] == \
        fixture["run"]["trainable_tensors"]
    assert run_report["training"]["total_updates"] == fixture["run"]["updates"]
    assert run_report["training"]["total_examples"] == \
        fixture["run"]["total_examples"]
    assert run_report["training"]["schedule"]["unique_examples"] == \
        fixture["run"]["unique_examples"]
    assert run_report["base_saved"] == fixture["run"]["base_saved"]
    assert run_report["promotion"] == fixture["run"]["promotion"]

    extension = run_report["extension"]
    for key in ("format", "label", "weights_sha256", "bytes", "base_saved"):
        assert extension[key] == fixture["extension"][key]
    gates = run_report["heldout"]["gates"]
    for key in ("label", "diagnostic_only", "promotion",
                "all_declared_gates_passed"):
        assert gates[key] == fixture["heldout_gates"][key]
    assert gates["gate_failures"] == fixture["heldout_gates"]["gate_failures"]

    assert reload_report["status"] == fixture["reload"]["status"]
    assert reload_report["bitwise_equal"] == fixture["reload"][
        "bitwise_equal"]
    assert reload_report["zero_updates"] == fixture["reload"]["zero_updates"]
    assert reload_report["run_report_sha256"] == \
        fixture["reload"]["run_report_sha256"]
    assert reload_report["protocol_sha256"] == \
        fixture["reload"]["protocol_sha256"]
    assert reload_report["extension_weights_sha256"] == \
        fixture["extension"]["weights_sha256"]
    assert reload_report["comparisons"] == fixture["reload"]["comparisons"]


def test_caption_target_assignment_is_derived_and_image_bound():
    _built()
    cases, labels, controls = _caption_inputs("dev")
    labels_by_id = {row["case_id"]: row for row in labels}
    specs = vba.caption_target_specs_v2(
        cases, labels, controls, controls=list(vba._CAPTION_CONTROLS))
    case = next(case for case in cases
                if controls[case["case_id"]]["counterfactual_same_image"]
                is False)
    label = labels_by_id[case["case_id"]]
    row = controls[case["case_id"]]
    assert row["assignment_sha256"] == vba.control_assignment_sha256(row)
    for control in vba._CAPTION_CONTROLS:
        spec = specs[(case["case_id"], control)]
        assert spec["original_text"] == label["answer"]
        assert spec["original_objects"] == [list(item) for item in
                                            label["visible_objects"]]
        assert spec["assignment_sha256"] == row["assignment_sha256"]
    counterfactual = specs[(case["case_id"], "counterfactual")]
    assert counterfactual["expected_image_case_id"] \
        == case["counterfactual_case_id"]
    assert counterfactual["expected_image_sha256"] \
        == case["counterfactual_image_sha256"]
    assert counterfactual["following_objects"] == [
        list(item) for item in label["counterfactual_visible_objects"]]
    assert counterfactual["image_changed"] is True
    shuffled = specs[(case["case_id"], "shuffled")]
    assert shuffled["expected_image_sha256"] == row["shuffled"][
        "image_sha256"]
    assert shuffled["following_objects"] == row["shuffled"][
        "visible_objects"]
    noimage = specs[(case["case_id"], "noimage")]
    assert noimage["expected_image_sha256"] is None
    assert noimage["following_available"] is False


def test_answer_target_assignment_keeps_changed_answer_separate():
    cases, labels, controls = _answer_inputs("heldout")
    specs = vba.answer_target_specs_v2(
        cases, labels, controls,
        controls=_config()["evaluation"]["heldout_answer_controls"])
    labels_by_id = {row["case_id"]: row for row in labels}
    case = next(case for case in cases
                if labels_by_id[case["case_id"]][
                    "answer_changes_under_counterfactual"] is True)
    label = labels_by_id[case["case_id"]]
    row = controls[case["case_id"]]
    counterfactual = specs[(case["case_id"], "counterfactual")]
    assert counterfactual["original_answer"] == label["answer"]
    assert counterfactual["following_answer"] \
        == label["counterfactual_answer"]
    assert counterfactual["original_answer"] != counterfactual[
        "following_answer"]
    assert row["counterfactual"]["sealed_answer"] \
        == label["counterfactual_answer"]
    wrong = specs[(case["case_id"], "wrong")]
    assert wrong["following_answer"] == row["wrong"]["sealed_answer"]
    assert wrong["following_answer"] != label["answer"]
    shuffled = specs[(case["case_id"], "shuffled")]
    assert shuffled["following_answer"] == row["shuffled"]["sealed_answer"]
    noimage = specs[(case["case_id"], "noimage")]
    assert noimage["following_applicable"] is False


def _positive_caption_rows(role="dev"):
    cases, labels, controls = _caption_inputs(role)
    return cases, labels, controls, vba._caption_probe_rows_v2(
        cases, labels, controls, list(vba._CAPTION_CONTROLS),
        "perfect_image_following")


def test_caption_fixed_target_and_tamper_rejection():
    cases, labels, controls, rows = _positive_caption_rows()
    config = _config()
    score = vba.score_caption_controls_v2(
        cases, labels, rows, control_map=controls,
        controls=list(config["evaluation"]["caption_controls"]),
        seed=1, bootstrap={"cluster": "world", "confidence": 0.95,
                           "resamples": 100})
    assert score["format"] == vba.CAPTION_SCORE_FORMAT
    assert score["controls"]["correct"]["original_semantic_f1"] == \
        pytest.approx(1.0)
    assert score["dependence"]["counterfactual"]["following_accuracy"] == \
        pytest.approx(1.0)
    assert score["dependence"]["counterfactual"][
        "original_semantic_f1_drop"] > 0.0
    assert score["noimage"]["original_semantic_f1"] == 0.0
    original_by_case = {label["case_id"]: label["answer"] for label in labels}
    assert all(row["target_text"] == original_by_case[row["case_id"]]
               for row in score["raw_generations"])

    def reject(mutator):
        broken = [dict(row) for row in rows]
        mutator(broken[0])
        with pytest.raises(ValueError):
            vba.score_caption_controls_v2(
                cases, labels, broken, control_map=controls,
                controls=list(config["evaluation"]["caption_controls"]),
                seed=1, bootstrap={"cluster": "world", "confidence": 0.95,
                                   "resamples": 100})

    reject(lambda row: row.pop("response_loss"))
    reject(lambda row: row.update(image_sha256="0" * 64))
    reject(lambda row: row.update(target_text="a purple hexagon"))
    reject(lambda row: row.update(target_visible_objects=[
        ["purple", "hexagon"]]))
    reject(lambda row: row.update(control_assignment_sha256="0" * 64))
    reject(lambda row: row.update(target_case_id="not-the-target"))


def test_caption_positive_and_negative_controls():
    cases, labels, controls = _caption_inputs("dev")
    receipt = vba.verify_caption_control_contracts_v2(
        cases, labels, controls,
        controls=list(_config()["evaluation"]["caption_controls"]),
        seed=5, bootstrap={"cluster": "world", "confidence": 0.95,
                           "resamples": 100},
        evaluation=_config()["evaluation"])
    assert receipt["all_contracts_passed"] is True
    assert receipt["positive"]["all_declared_gates_passed"] is True
    assert receipt["negative"]["gate_failures"]["image_ignoring"]
    assert receipt["negative"]["gate_failures"]["constant_caption"]
    assert receipt["negative"]["gate_failures"]["shuffled"]
    assert receipt["negative"]["malformed_rejected"]
    assert receipt["negative"]["target_tampering_rejected"]


def test_caption_gate_semantics_reward_following_not_original_recall():
    cases, labels, controls = _caption_inputs("heldout")
    evaluation = copy.deepcopy(_config()["evaluation"])
    rows = vba._caption_probe_rows_v2(
        cases, labels, controls, list(vba._CAPTION_CONTROLS),
        "perfect_image_following")
    positive = vba.score_caption_controls_v2(
        cases, labels, rows, control_map=controls,
        controls=list(evaluation["caption_controls"]), seed=7,
        bootstrap=evaluation["bootstrap"])
    report = vba.diagnostic_gate_report_v2(
        label="positive", answer=None, caption=positive,
        evaluation=evaluation)
    assert report["all_declared_gates_passed"] is True
    ignoring_rows = vba._caption_probe_rows_v2(
        cases, labels, controls, list(vba._CAPTION_CONTROLS),
        "image_ignoring")
    ignoring = vba.score_caption_controls_v2(
        cases, labels, ignoring_rows, control_map=controls,
        controls=list(evaluation["caption_controls"]), seed=7,
        bootstrap=evaluation["bootstrap"])
    ignored = vba.diagnostic_gate_report_v2(
        label="ignoring", answer=None, caption=ignoring,
        evaluation=evaluation)
    assert "min_caption_following_accuracy_counterfactual" in \
        ignored["gate_failures"]
    assert "min_caption_original_agreement_drop_counterfactual" in \
        ignored["gate_failures"]
    noimage_rows = [dict(row) for row in rows]
    for row in noimage_rows:
        if row["control"] == "noimage":
            row["generated_text"] = labels[0]["answer"]
    noimage_score = vba.score_caption_controls_v2(
        cases, labels, noimage_rows, control_map=controls,
        controls=list(evaluation["caption_controls"]), seed=7,
        bootstrap=evaluation["bootstrap"])
    noimage_report = vba.diagnostic_gate_report_v2(
        label="noimage", answer=None, caption=noimage_score,
        evaluation=evaluation)
    assert "max_noimage_original_semantic_f1" in \
        noimage_report["gate_failures"]


def test_answer_following_and_gate_separation():
    cases, labels, controls = _answer_inputs("heldout")
    evaluation = copy.deepcopy(_config()["evaluation"])
    controls_list = list(evaluation["heldout_answer_controls"])
    positive_rows = vba._answer_probe_rows_v2(
        cases, labels, controls, controls_list,
        "perfect_image_following")
    positive = vba.score_answer_controls_v2(
        cases, labels, positive_rows, control_map=controls,
        controls=controls_list, seed=11,
        bootstrap=evaluation["bootstrap"])
    assert positive["format"] == vba.ANSWER_SCORE_FORMAT
    assert positive["counterfactual_answerable"]["following_accuracy"] \
        == pytest.approx(1.0)
    assert positive["counterfactual_answerable"]["original_label_margin"] \
        > 0.0
    positive_report = vba.diagnostic_gate_report_v2(
        label="positive_answer", answer=positive, caption=None,
        evaluation=evaluation)
    assert positive_report["all_declared_gates_passed"] is True
    ignoring_rows = vba._answer_probe_rows_v2(
        cases, labels, controls, controls_list, "image_ignoring")
    ignoring = vba.score_answer_controls_v2(
        cases, labels, ignoring_rows, control_map=controls,
        controls=controls_list, seed=11,
        bootstrap=evaluation["bootstrap"])
    ignoring_report = vba.diagnostic_gate_report_v2(
        label="ignoring_answer", answer=ignoring, caption=None,
        evaluation=evaluation)
    assert "min_counterfactual_following_accuracy" in \
        ignoring_report["gate_failures"]
    assert "min_counterfactual_margin_answerable" in \
        ignoring_report["gate_failures"]
    malformed = [dict(row) for row in positive_rows]
    malformed[0].pop("control_assignment_sha256")
    with pytest.raises(ValueError):
        vba.score_answer_controls_v2(
            cases, labels, malformed, control_map=controls,
            controls=controls_list, seed=11,
            bootstrap=evaluation["bootstrap"])


def test_answer_positive_and_negative_contract_receipts():
    cases, labels, controls = _answer_inputs("dev")
    receipt = vba.verify_answer_control_contracts_v2(
        cases, labels, controls,
        controls=list(_config()["evaluation"]["dev_answer_controls"]),
        seed=13, bootstrap={"cluster": "world", "confidence": 0.95,
                            "resamples": 100},
        evaluation=_config()["evaluation"])
    assert receipt["all_contracts_passed"] is True
    assert receipt["positive"]["all_declared_gates_passed"] is True
    for mode in ("constant_answer", "image_ignoring", "shuffled"):
        assert receipt["negative"]["gate_failures"][mode]
    assert receipt["negative"]["malformed_rejected"]
    assert receipt["negative"]["assignment_tampering_rejected"]


def test_synthetic_scorer_preflight_reads_no_prepared_data():
    preflight = vba.scorer_contract_preflight_v2(_config()["evaluation"])
    assert preflight["all_contracts_passed"] is True
    assert preflight["reads_prepared_data"] is False
    assert preflight["caption"]["positive"]["all_declared_gates_passed"]
    assert preflight["answer"]["positive"]["all_declared_gates_passed"]


def test_access_order_blocks_heldout_before_dev():
    order = vba.AccessOrder()
    order.access("train")
    with pytest.raises(ValueError):
        order.access("dev")
    order.advance("train_fit")
    order.access("train_fit")
    order.advance("dev")
    order.access("dev")
    with pytest.raises(ValueError):
        order.access("audit")
    order.advance("audit")
    order.access("audit")
    assert [event for event in order.snapshot()["events"]
            if "role" in event] == [
        {"role": "train", "phase": "train"},
        {"role": "train_fit", "phase": "train_fit"},
        {"role": "dev", "phase": "dev"},
        {"role": "audit", "phase": "audit"},
    ]


def test_preaudit_and_reload_receipts_cover_dev_and_scores():
    protocol = {
        "preaudit": {"require_train_fit_gates": True,
                     "require_dev_gates": True},
    }
    passing = {"all_declared_gates_passed": True, "gate_failures": []}
    failing = {"all_declared_gates_passed": False,
               "gate_failures": ["min_correct_accuracy"]}
    assert vba.preaudit_allows_heldout(passing, passing, protocol) is True
    assert vba.preaudit_allows_heldout(failing, passing, protocol) is False
    assert vba.preaudit_allows_heldout(passing, failing, protocol) is False
    names = set(vba.reload_comparison_names_v2())
    for required in (
            "dev_answer_generations.jsonl", "dev_caption_generations.jsonl",
            "train_fit_answer_generations.jsonl",
            "scores/dev_answer.json", "scores/dev_caption.json",
            "scores/train_fit_answer.json", "gates/dev.json",
            "gates/train_fit.json", "heldout_caption_generations.jsonl"):
        assert required in names


def test_v2_runner_imports_and_uses_corrected_target_contract():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vision_balanced_runner_v2",
        ROOT / "scripts/run_vision_balanced_alignment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module._execute_train_v2)
    assert callable(module._execute_reload_v2)
    assert "PROTOCOL_FORMAT_V2" in module.__dict__


def test_prepared_v2_roles_load_separately_with_bound_assignments():
    import importlib.util
    protocol_path = (ROOT / "data/vision_balanced_alignment_v2_20261026/prepared04/"
                     "protocol.json")
    with pytest.raises(ValueError, match="Implementation bytes changed"):
        vba.load_prepared(protocol_path, ROOT)
    diagnostic_spec = importlib.util.spec_from_file_location(
        "vision_collapse_runner_for_roles",
        ROOT / "scripts/run_vision_collapse_diagnostic.py")
    diagnostic_module = importlib.util.module_from_spec(diagnostic_spec)
    diagnostic_spec.loader.exec_module(diagnostic_module)
    diagnostic_config = diagnostic_module.load_config(
        ROOT / "configs/vision_collapse_diagnostic_v1.json")
    verified = diagnostic_module.verify_pins(diagnostic_config)
    protocol, manifest = verified["protocol"], verified["manifest"]
    assert protocol["format"] == vba.PROTOCOL_FORMAT_V2
    spec = importlib.util.spec_from_file_location(
        "vision_balanced_runner_v2_roles",
        ROOT / "scripts/run_vision_balanced_alignment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    training = module._load_training_v2(protocol)
    dev = module._load_dev_v2(protocol)
    heldout = module._load_heldout(protocol)
    train_ids = {row["case_id"] for row in training["training"]}
    dev_ids = {row["case_id"] for row in dev["dev_answer_cases"]}
    dev_ids |= {row["case_id"] for row in dev["dev_caption_cases"]}
    heldout_ids = {row["case_id"] for row in heldout["heldout_answer_cases"]}
    heldout_ids |= {row["case_id"] for row in heldout["heldout_caption_cases"]}
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(heldout_ids)
    assert dev_ids.isdisjoint(heldout_ids)
    for mapping in (dev["dev_caption_controls"],
                    heldout["heldout_caption_controls"],
                    dev["dev_answer_controls"],
                    heldout["heldout_answer_controls"]):
        for row in mapping.values():
            assert row["assignment_sha256"] == \
                vba.control_assignment_sha256(row)
    for row in heldout["heldout_answer_controls"].values():
        assert "sealed_answer" in row["counterfactual"]
        assert "sealed_answer" in row["wrong"]
        assert "sealed_answer" in row["shuffled"]
    assert manifest["format"] == vba.MANIFEST_FORMAT_V2


def test_fit_tensor_contract_allows_shared_eos_pad_but_rejects_interior_pad():
    import importlib.util
    from types import SimpleNamespace
    import torch

    spec = importlib.util.spec_from_file_location(
        "vision_balanced_runner_fit",
        ROOT / "scripts/run_vision_balanced_alignment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=2)
    protocol = {
        "fit_contract": {
            "caption_template": "Describe the image briefly.",
            "loss": "mean_response_nll",
            "pad_tokens_forbidden": True,
        },
        "limits": {"max_sequence_tokens": 8},
    }
    row = {"kind": "answer", "question": "q", "response": "ok"}
    prompt = torch.tensor([[7]])
    input_ids = torch.tensor([[7, 11, 2]])
    labels = torch.tensor([[-100, 11, 2]])
    receipt = module._verify_fit_tensor_contract(
        tokenizer, row, prompt, input_ids, labels, protocol)
    assert receipt["pad_free_response"] is True
    with pytest.raises(RuntimeError, match="before the final EOS"):
        module._verify_fit_tensor_contract(
            tokenizer, row, prompt, torch.tensor([[7, 2, 2]]),
            labels, protocol)


def test_runner_binds_target_case_to_original_case_for_every_control(tmp_path):
    import importlib.util
    import time
    from types import SimpleNamespace
    import torch

    spec = importlib.util.spec_from_file_location(
        "vision_balanced_runner_target_case",
        ROOT / "scripts/run_vision_balanced_alignment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import run_vision_curriculum as shared
    shared.curriculum_prompt = lambda text: text
    shared.observe = lambda service, **kwargs: {"image": kwargs["image_sha256"]}
    shared.generate_row = lambda session, tokenizer, **kwargs: {
        "control": kwargs["control"], "case_id": kwargs["case"]["case_id"],
        "generated_text": "ok", "image_case_id": kwargs.get("image_case_id"),
        "image_sha256": kwargs.get("image_sha256"),
    }
    module._teacher_forced_loss = lambda *args, **kwargs: 0.0
    case = {"case_id": "original", "question": "q",
            "image_path": "orig.png", "image_sha256": "a" * 64,
            "counterfactual_case_id": "donor",
            "counterfactual_image_sha256": "b" * 64}
    labels = [{"case_id": "original", "answer": "original caption",
               "visible_objects": [["red", "circle"]],
               "counterfactual_answer": "changed caption",
               "counterfactual_visible_objects": [["blue", "square"]]}]
    control_map = {"original": {
        "counterfactual": {"case_id": "donor", "image_path": "donor.png",
                           "image_sha256": "b" * 64},
        "shuffled": {"case_id": "shuffle", "image_path": "shuffle.png",
                     "image_sha256": "c" * 64, "caption": "shuffled",
                     "visible_objects": [["green", "triangle"]]},
    }}
    adapter = SimpleNamespace(eval=lambda: None)
    session = SimpleNamespace(device=torch.device("cpu"),
                              ledger=SimpleNamespace(epoch=1))
    service = SimpleNamespace(spec=object())
    protocol = {"format": module.PROTOCOL_FORMAT_V2,
                "evaluation": {"caption_max_new_tokens": 8},
                "limits": {"max_seconds": 1000, "max_prompt_tokens": 256},
                "vision_source": {"source_id": "test-vision-source"}}
    rows = module.evaluate_caption_controls(
        adapter, session, object(), service, protocol, object(), [case], labels,
        ["correct", "counterfactual", "shuffled", "noimage"], control_map,
        tmp_path / "rows.jsonl", guard=None, start_time=time.monotonic())
    assert [row["control"] for row in rows] == [
        "correct", "counterfactual", "shuffled", "noimage"]
    assert {row["target_case_id"] for row in rows} == {"original"}
    assert {row["target_text"] for row in rows} == {"original caption"}
