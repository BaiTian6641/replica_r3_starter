"""Preparation and scoring contracts for DC-P1 native response learning.

These tests use only the deterministic text fixture and a synthetic decoder.
They load no MiniCPM5 model, no tokenizer, no optimizer and no native runtime.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dc_p1_native_response_20261027 as p1  # noqa: E402
from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, check_work_budget, read_json, read_jsonl, require_fresh_path,
    sha256_file, sha256_json, validate_guards, write_jsonl,
)

PREPARED = p1.PREPARED_DIR
CONFIG = p1.CONFIG_PATH


def _encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def _decode(ids) -> str:
    return "".join(chr(item) for item in ids)


def _config() -> dict:
    return read_json(CONFIG)


def _binding(config: dict) -> dict:
    return p1.validate_tokenizer_binding(config["tokenizer_binding"])


def _labels() -> dict[str, dict]:
    rows = {}
    for name in ("sealed_labels_fit.jsonl", "sealed_labels_dev.jsonl",
                 "sealed_labels_final.jsonl"):
        for row in read_jsonl(PREPARED / name):
            rows[row["case_id"]] = row
    return rows


def _cases() -> dict[str, dict]:
    rows = {}
    for name in ("fit_cases.jsonl", "dev_cases.jsonl"):
        for row in read_jsonl(PREPARED / name):
            rows[row["case_id"]] = row
    return rows


def _prediction(case: dict, arm: str, text: str, config: dict) -> dict:
    binding = _binding(config)
    return {
        "format": p1.PREDICTION_FORMAT,
        "protocol_id": p1.PROTOCOL_ID,
        "arm_id": arm,
        "case_id": case["case_id"],
        "prompt": case["prompt"],
        "prompt_sha256": case["prompt_sha256"],
        "generated_ids": _encode(text),
        "generated_text": text,
        "empty_generation": text == "",
        "tokenizer_binding_sha256": p1.tokenizer_binding_sha(binding),
        "decode_policy": binding["decode_policy"],
        "source_context_present": False,
        "source_kv_loaded": False,
        "source_prefill_tokens": 0,
        "source_kv_reused": False,
        "prompt_source_spans": 0,
        "routing_metadata": {"mode": "fixed", "expert_id": 0},
        "stopped": True,
        "stop_reason": "eos",
    }


def _rows(*, controls_perfect_primary: bool = False,
          treatment_empty_paraphrase: bool = False) -> list[dict]:
    config = _config()
    labels = _labels()
    cases = _cases()
    rows = []
    for case_id, case in cases.items():
        label = labels[case_id]
        target = label["target"]
        role = label["role"]
        for arm in p1.ARMS:
            if arm == "treatment":
                text = target
                if (treatment_empty_paraphrase and role == "paraphrase"
                        and case_id.endswith("_paraphrase")):
                    text = ""
            elif controls_perfect_primary and role in p1.PRIMARY_ROLES:
                text = target
            elif role in ("retention", "abstention"):
                text = target
            else:
                text = "wrong"
            rows.append(_prediction(case, arm, text, config))
    return rows


def _score(tmp_path: Path, rows: list[dict]) -> dict:
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, rows)
    return p1.score_p1(
        path,
        fit_cases_path=PREPARED / "fit_cases.jsonl",
        fit_labels_path=PREPARED / "sealed_labels_fit.jsonl",
        dev_cases_path=PREPARED / "dev_cases.jsonl",
        dev_labels_path=PREPARED / "sealed_labels_dev.jsonl",
        config_path=CONFIG, decode_fn=_decode)


# ---------------------------------------------------------------------------
# Fixture, config and leakage


def test_fixture_is_deterministic_role_sealed_and_source_free():
    first = p1.validate_prepared()
    second = p1.validate_prepared()
    assert first["status"] == "valid"
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["counts"]["teaching"] == 36
    assert first["counts"]["dev_cases"] == 60
    assert first["counts"]["sealed_final_teaching"] == 12
    assert first["counts"]["sealed_final_fit_cases"] == 12
    assert first["counts"]["sealed_final_cases"] == 20
    assert (first["counts"]["teaching"]
            + first["counts"]["sealed_final_teaching"] == 48)
    assert first["native_computation"] is False
    assert first["heldout_opened"] is False
    assert first["labels_read_for_construction_audit"] is True


def test_evaluation_rows_do_not_carry_targets_or_source_context():
    labels = _labels()
    for name in ("dev_cases.jsonl", "sealed_final_cases.jsonl"):
        for row in read_jsonl(PREPARED / name):
            for forbidden in ("target", "expected", "answer", "gold",
                              "answer_index", "source", "source_text"):
                assert forbidden not in row
            assert row["source_context_present"] is False
            assert row["source_kv_loaded"] is False
            assert row["inference_inputs"] == ["prompt"]
            target = labels[row["case_id"]]["target"]
            assert target.casefold() not in row["prompt"].casefold()
            assert row["prompt_sha256"] == sha256_json(
                {"prompt": row["prompt"]})


def test_fit_and_dev_prompts_are_distinct_and_group_disjoint():
    teaching = read_jsonl(PREPARED / "teaching.jsonl")
    dev = read_jsonl(PREPARED / "dev_cases.jsonl")
    final = read_jsonl(PREPARED / "sealed_final_cases.jsonl")
    prompts = [row["prompt"] for row in teaching + dev]
    assert len(prompts) == len(set(prompts))
    train_groups = {row["source_group"] for row in teaching}
    dev_groups = {row["source_group"] for row in dev}
    final_groups = {row["source_group"] for row in final}
    assert train_groups == dev_groups
    assert not (train_groups & final_groups)
    assert len(final_groups) == 4


def test_config_has_exactly_one_recipe_change_and_rejects_budget02():
    config = _config()
    p1.validate_config(config)
    changed = deepcopy(config)
    changed["recipe"]["treatment"]["changes"].append({
        "field": "learning_rate", "from": 0.0001, "to": 0.0003})
    with pytest.raises(ProtocolError, match="exactly one primary recipe change"):
        p1.validate_config(changed)
    budget = deepcopy(config)
    budget["recipe"]["treatment"]["steps_per_event"] = 2
    with pytest.raises(ProtocolError, match="budget02 revival"):
        p1.validate_config(budget)
    hidden = deepcopy(config)
    hidden["anti_shortcut"]["answer_index_bypass_allowed"] = True
    with pytest.raises(ProtocolError, match="answer-index bypass"):
        p1.validate_config(hidden)
    guard = deepcopy(config)
    guard["resource_guard"]["peak_driver_bytes_max"] = 1
    with pytest.raises(ProtocolError, match="resource_guard"):
        p1.validate_config(guard)
    tokenizer = deepcopy(config)
    tokenizer["tokenizer_binding"]["decode_policy"][
        "answer_selection_allowed"] = True
    with pytest.raises(ProtocolError, match="decode_policy"):
        p1.validate_config(tokenizer)


def test_22_class_fixture_is_explicitly_integration_only():
    config = _config()
    fixture = config["integration_only_fixture"]
    assert fixture["status"] == "integration_only"
    assert fixture["used_as_parent"] is False
    assert fixture["used_as_scoring_oracle"] is False
    assert fixture["claims_allowed"] == []
    assert fixture["protocol_id"] == "value_readout_abstention_v1"
    assert config["prepared"]["root"] != fixture["prepared_root"]


def test_baseline_parity_and_guards_are_frozen():
    config = _config()
    validate_guards(config)
    assert config["optimizer_policy"]["primary"] == "NativeFusedAdamW"
    assert config["optimizer_policy"]["default_name"] == "adamw_fused"
    assert config["optimizer_policy"]["changed"] is False
    assert config["resource_guard"]["peak_driver_bytes_max"] == 53687091200
    assert config["resource_guard"]["sustained_swap_growth_bytes_max"] == 536870912
    assert config["resource_guard"]["min_host_free_fraction"] == 0.15
    assert config["resource_guard"]["cpu_fallback"] is False
    assert config["resource_guard"]["mps_fallback"] is False
    parity = config["baseline_parity"]
    assert parity["no_extension_is_default"] is True
    assert parity["disabled_extension_preserves_native_logits"] is True
    assert parity["disabled_extension_preserves_native_greedy_tokens"] is True
    assert parity["logit_rtol"] == 0.0
    assert parity["logit_atol"] == 0.0


def test_physical_budget_fails_on_an_extra_optimizer_step():
    budget = _config()["budget"]
    check_work_budget({"optimizer_steps": 48}, budget)
    with pytest.raises(ProtocolError, match="exceeds"):
        check_work_budget({"optimizer_steps": 49}, budget)


# ---------------------------------------------------------------------------
# Generated-answer scoring and corrupted controls


def test_perfect_treatment_passes_and_reports_source_free_scoring(tmp_path):
    report = _score(tmp_path, _rows())
    assert report["status"] == "passed"
    assert report["source_free"] is True
    assert report["answer_index_used"] is False
    assert report["arms"]["treatment"]["fit"]["exact_accuracy"] == 1.0
    assert report["arms"]["treatment"]["dev"]["paraphrase"]["exact_accuracy"] == 1.0
    assert report["arms"]["treatment"]["dev"]["new_instance"]["exact_accuracy"] == 1.0
    assert report["arms"]["treatment"]["dev"]["correction"]["exact_accuracy"] == 1.0
    assert report["arms"]["treatment"]["dev"]["abstention"]["exact_accuracy"] == 1.0
    assert report["arms"]["treatment"]["dev"]["retention"]["exact_accuracy"] == 1.0
    assert report["gains"]["paraphrase_vs_no_update"]["mean_gain"] == 1.0
    assert report["gates"]["retention_decline_vs_reset_P"]["value"] == 0.0


def test_controls_that_reproduce_the_gain_fail(tmp_path):
    report = _score(tmp_path, _rows(controls_perfect_primary=True))
    assert report["status"] == "failed"
    assert report["gates"]["paraphrase_gain_vs_no_update"]["passed"] is False
    assert report["gates"]["no_update_paraphrase_control_ceiling"]["passed"] is False
    assert report["gates"]["reset_P_new_instance_control_ceiling"]["passed"] is False


def test_source_retained_or_source_kv_retained_prediction_is_rejected(tmp_path):
    rows = _rows()
    rows[0]["source_context_present"] = True
    with pytest.raises(ProtocolError, match="source_free_evaluation"):
        _score(tmp_path, rows)
    rows = _rows()
    rows[1]["source_kv_loaded"] = True
    with pytest.raises(ProtocolError, match="source_free_evaluation"):
        _score(tmp_path, rows)


def test_answer_index_and_answer_bearing_routing_are_rejected(tmp_path):
    rows = _rows()
    rows[0]["answer_index"] = 3
    with pytest.raises(ProtocolError, match="answer-bearing fields"):
        _score(tmp_path, rows)
    rows = _rows()
    rows[0]["routing_metadata"]["target"] = "leak"
    with pytest.raises(ProtocolError, match="answer-bearing fields"):
        _score(tmp_path, rows)


def test_tokenizer_binding_and_decode_mismatch_are_rejected(tmp_path):
    rows = _rows()
    rows[0]["generated_text"] = "fabricated"
    with pytest.raises(ProtocolError, match="decoded ids"):
        _score(tmp_path, rows)
    rows = _rows()
    rows[0]["tokenizer_binding_sha256"] = "0" * 64
    with pytest.raises(ProtocolError, match="tokenizer binding"):
        _score(tmp_path, rows)


def test_empty_generation_receives_no_credit(tmp_path):
    report = _score(tmp_path, _rows(treatment_empty_paraphrase=True))
    assert report["status"] == "failed"
    assert report["gates"]["treatment_paraphrase_accuracy"]["passed"] is False
    assert report["arms"]["treatment"]["dev"]["paraphrase"]["exact_accuracy"] < 1.0


def test_unlabelled_target_in_evaluation_prompt_is_a_protocol_error():
    labels = _labels()
    case = read_jsonl(PREPARED / "dev_cases.jsonl")[0]
    target = labels[case["case_id"]]["target"]
    with pytest.raises(ProtocolError, match="target"):
        p1.assert_target_absent(case["prompt"] + " " + target, target)


# ---------------------------------------------------------------------------
# Review-only preflight


def test_preflight_review_is_fresh_and_launch_fails_closed(tmp_path):
    review = p1.preflight()
    assert review["status"] == "review_ready"
    assert review["exit_code"] == 0
    assert review["native_execution"] is False
    with pytest.raises(ProtocolError, match="authorization"):
        p1.preflight(launch=True)
    config_sha = sha256_file(CONFIG)
    manifest = read_json(PREPARED / "manifest.json")
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({
        "format": "replica.dc_protocol.authorization.v1",
        "protocol_id": p1.PROTOCOL_ID,
        "config_sha256": config_sha,
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "authorized_by": "test",
        "native_execution_allowed": True,
        "single_attempt": True,
        "receipt_sha256": "1" * 64,
    }), encoding="utf-8")
    launch = p1.preflight(launch=True, authorization=authorization)
    assert launch["status"] == "refused_native_runner_absent"
    assert launch["exit_code"] == 2
    assert launch["native_execution"] is False


def test_review_receipt_path_must_be_fresh(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolError, match="not fresh"):
        require_fresh_path(path, "preflight_receipt")
