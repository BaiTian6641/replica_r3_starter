"""Preparation and scoring contracts for DC-R1 native record readout.

These tests use only the deterministic text fixture, the declared 8-bit codec
and a synthetic decoder.  They load no MiniCPM5 model, no tokenizer, no
optimizer and no native runtime.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dc_r1_native_value_20261027 as r1  # noqa: E402
from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, check_work_budget, read_json, read_jsonl,
    require_fresh_path, seal_manifest, sha256_file, sha256_json,
    validate_guards, write_jsonl,
)

PREPARED = r1.PREPARED_DIR
CONFIG = r1.CONFIG_PATH


def _encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def _decode(ids) -> str:
    return "".join(chr(item) for item in ids)


def _config() -> dict:
    return read_json(CONFIG)


def _binding(config: dict) -> dict:
    return r1.validate_tokenizer_binding(config["tokenizer_binding"])


def _labels() -> dict[str, dict]:
    return {row["case_id"]: row
            for row in read_jsonl(PREPARED / "sealed_labels_dev.jsonl")}


def _cases() -> dict[str, dict]:
    return {row["case_id"]: row
            for row in read_jsonl(PREPARED / "dev_cases.jsonl")}


def _conditions() -> dict[tuple[str, str], dict]:
    rows = {}
    for condition in r1.CONDITIONS:
        for row in read_jsonl(PREPARED / f"dev_conditions_{condition}.jsonl"):
            rows[(condition, row["case_id"])] = row
    return rows


def _row(condition: str, condition_row: dict, case: dict,
         texts: list[str], config: dict) -> dict:
    binding = _binding(config)
    outputs = []
    for layer, text in zip(case["layer_group"], texts):
        outputs.append({
            "layer": layer,
            "generated_ids": _encode(text),
            "generated_text": text,
            "abstain": text == "UNKNOWN",
            "empty_generation": text == "",
        })
    return {
        "format": r1.PREDICTION_FORMAT,
        "protocol_id": r1.PROTOCOL_ID,
        "condition": condition,
        "case_id": case["case_id"],
        "prompt": condition_row["prompt"],
        "prompt_sha256": condition_row["prompt_sha256"],
        "layer_outputs": outputs,
        "source_context_present": condition_row["source_context_present"],
        "source_kv_loaded": False,
        "prompt_source_spans": 0,
        "tokenizer_binding_sha256": r1.tokenizer_binding_sha(binding),
        "decode_policy": binding["decode_policy"],
        "routing_metadata": {"mode": "fixed", "expert_id": 0},
    }


def _rows(*, no_memory_leaks: bool = False,
          selected_only_one_layer: bool = False) -> list[dict]:
    config = _config()
    codec = read_json(PREPARED / "codec.json")
    labels = _labels()
    cases = _cases()
    conditions = _conditions()
    rows = []
    for (condition, case_id), condition_row in conditions.items():
        target = labels[case_id]["target"]
        case = cases[case_id]
        if condition in ("exact_context", "selected_literal"):
            texts = [target] * 3
        elif condition == "compressed_record":
            text = r1.decode_compressed_literal(
                codec, labels[case_id]["compressed_code"])
            texts = [text] * 3
        elif condition == "wrong_value":
            texts = [condition_row["evidence"]["literal"]] * 3
        else:
            texts = [target] * 3 if no_memory_leaks else ["UNKNOWN"] * 3
        if selected_only_one_layer and condition == "selected_literal":
            texts = [target, "UNKNOWN", "UNKNOWN"]
        rows.append(_row(condition, condition_row, case, texts, config))
    return rows


def _score(tmp_path: Path, rows: list[dict]) -> dict:
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, rows)
    return r1.score_r1(
        path,
        dev_cases_path=PREPARED / "dev_cases.jsonl",
        dev_labels_path=PREPARED / "sealed_labels_dev.jsonl",
        dev_conditions={
            condition: PREPARED / f"dev_conditions_{condition}.jsonl"
            for condition in r1.CONDITIONS},
        config_path=CONFIG, decode_fn=_decode)


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "fixture"
    shutil.copytree(PREPARED, target)
    return target


def _reseal(prepared: Path) -> None:
    manifest = read_json(prepared / "manifest.json")
    manifest["files"] = {
        name: sha256_file(prepared / name) for name in manifest["files"]}
    seal_manifest(prepared, manifest)


# ---------------------------------------------------------------------------
# Fixture, codec and config


def test_fixture_is_deterministic_role_sealed_and_reports_reconstruction():
    first = r1.validate_prepared()
    second = r1.validate_prepared()
    assert first["status"] == "valid"
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["counts"]["dev_cases"] == 12
    assert first["counts"]["sealed_final_cases"] == 4
    assert first["counts"]["dev_condition_rows"] == 60
    assert first["compressed_reconstruction"]["dev"] == 0.75
    assert first["compressed_reconstruction"]["sealed_final"] == 0.75
    assert first["native_computation"] is False
    assert first["heldout_opened"] is False


def test_queries_do_not_contain_targets_or_answer_indexes():
    labels = _labels()
    for case in read_jsonl(PREPARED / "dev_cases.jsonl"):
        for forbidden in ("target", "expected", "answer", "gold",
                          "answer_index", "literal", "wrong_value"):
            assert forbidden not in case
        target = labels[case["case_id"]]["target"]
        assert target.casefold() not in case["query"].casefold()
        assert case["layer_group"] == [13, 27, 41]


def test_condition_evidence_policies_are_explicit():
    labels = _labels()
    conditions = _conditions()
    for (condition, case_id), row in conditions.items():
        target = labels[case_id]["target"]
        assert row["prompt_sha256"] == sha256_json({"prompt": row["prompt"]})
        assert row["source_kv_loaded"] is False
        assert row["source_context_present"] == (condition == "exact_context")
        if condition == "no_memory":
            assert row["evidence"] == {"kind": "none"}
            assert target.casefold() not in row["prompt"].casefold()
        if condition == "wrong_value":
            assert row["evidence"]["literal"] != target
            assert target.casefold() not in row["prompt"].casefold()
        if condition == "compressed_record":
            assert row["evidence"]["kind"] == "compressed_code"
            assert target.casefold() not in row["prompt"].casefold()
        if condition == "selected_literal":
            assert row["evidence"]["literal"] == target


def test_config_blocks_learned_retrieval_and_storage_expansion():
    config = _config()
    r1.validate_config(config)
    assert config["readout_gate"]["learned_retrieval_interpreted"] is False
    assert config["readout_gate"]["storage_expansion_admitted"] is False
    assert config["classification"]["learned_retrieval"] is False
    assert config["classification"]["storage_expansion"] is False
    assert config["optimizer_policy"]["primary"] == "NativeFusedAdamW"
    validate_guards(config)
    parity = config["baseline_parity"]
    assert parity["no_extension_is_default"] is True
    assert parity["disabled_extension_preserves_native_logits"] is True
    assert parity["disabled_extension_preserves_native_greedy_tokens"] is True
    assert parity["logit_rtol"] == 0.0
    assert parity["logit_atol"] == 0.0
    assert parity["no_memory_condition_abstains_instead_of_inventing"] is True
    broken = deepcopy(config)
    broken["readout_gate"]["storage_expansion_admitted"] = True
    with pytest.raises(ProtocolError, match="storage expansion"):
        r1.validate_config(broken)
    broken = deepcopy(config)
    broken["conditions"][-1]["source_kv_loaded"] = True
    with pytest.raises(ProtocolError, match="source KV"):
        r1.validate_config(broken)


def test_22_class_fixture_is_explicitly_integration_only():
    fixture = _config()["integration_only_fixture"]
    assert fixture["status"] == "integration_only"
    assert fixture["used_as_parent"] is False
    assert fixture["used_as_scoring_oracle"] is False
    assert fixture["claims_allowed"] == []
    assert fixture["protocol_id"] == "value_readout_abstention_v1"


def test_physical_budget_has_zero_optimizer_and_bounded_forwards():
    budget = _config()["budget"]
    check_work_budget({
        "optimizer_steps": 0,
        "backward_calls": 0,
        "forward_calls": 240,
    }, budget)
    with pytest.raises(ProtocolError, match="exceeds"):
        check_work_budget({"forward_calls": 241}, budget)


# ---------------------------------------------------------------------------
# Corrupted controls


def test_wrong_value_equal_to_target_is_invalid(tmp_path):
    prepared = _copy_fixture(tmp_path)
    labels = {row["case_id"]: row for row in
              read_jsonl(prepared / "sealed_labels_dev.jsonl")}
    rows = read_jsonl(prepared / "dev_conditions_wrong_value.jsonl")
    rows[0]["evidence"]["literal"] = labels[rows[0]["case_id"]]["target"]
    write_jsonl(prepared / "dev_conditions_wrong_value.jsonl", rows)
    _reseal(prepared)
    with pytest.raises(ProtocolError, match="equals target"):
        r1._validate_fixture(prepared, _config(), validate_labels=True,
                             config_binding=False)


def test_no_memory_condition_must_have_empty_evidence(tmp_path):
    prepared = _copy_fixture(tmp_path)
    rows = read_jsonl(prepared / "dev_conditions_no_memory.jsonl")
    rows[0]["evidence"] = {"kind": "none", "literal": "leak"}
    write_jsonl(prepared / "dev_conditions_no_memory.jsonl", rows)
    _reseal(prepared)
    with pytest.raises(ProtocolError, match="no-memory evidence"):
        r1._validate_fixture(prepared, _config(), validate_labels=True,
                             config_binding=False)


# ---------------------------------------------------------------------------
# Readout scoring


def test_useful_readout_passes_and_keeps_retrieval_blocked(tmp_path):
    report = _score(tmp_path, _rows())
    assert report["status"] == "readout_useful"
    assert report["readout_useful"] is True
    assert report["learned_retrieval_interpretable"] is False
    assert report["storage_expansion_allowed"] is False
    assert report["reconstruction"]["compressed_reconstruction_accuracy"] == 0.75
    assert report["conditions"]["selected_literal"]["accepted_layer_accuracy"] == 1.0
    assert report["conditions"]["compressed_record"]["accepted_layer_accuracy"] == 0.75
    assert report["gates"]["wrong_value_correct_target"]["passed"] is True
    assert report["gates"]["no_memory_abstain"]["passed"] is True


def test_no_memory_answering_the_target_fails_the_abstention_gate(tmp_path):
    report = _score(tmp_path, _rows(no_memory_leaks=True))
    assert report["status"] == "readout_not_useful"
    assert report["gates"]["no_memory_abstain"]["passed"] is False
    assert report["gates"]["no_memory_non_abstain"]["passed"] is False


def test_accepted_layer_is_reported_separately_from_complete_group(tmp_path):
    report = _score(tmp_path, _rows(selected_only_one_layer=True))
    selected = report["accepted_layer_vs_complete_group"]["selected_literal"]
    assert selected["accepted_layer_accuracy"] == 1.0
    assert selected["complete_group_accuracy"] == 0.0
    assert selected["gap"] == 1.0
    assert report["status"] == "readout_not_useful"


def test_wrong_value_control_cannot_pass_readout(tmp_path):
    report = _score(tmp_path, _rows())
    wrong = report["conditions"]["wrong_value"]
    assert wrong["accepted_layer_accuracy"] == 0.0
    assert wrong["complete_group_accuracy"] == 0.0
    assert report["gates"]["wrong_value_correct_target"]["value"] == 0.0


def test_answer_index_and_target_routing_are_rejected(tmp_path):
    rows = _rows()
    rows[0]["answer_index"] = 2
    with pytest.raises(ProtocolError, match="answer-bearing fields"):
        _score(tmp_path, rows)
    rows = _rows()
    rows[0]["routing_metadata"]["gold"] = "leak"
    with pytest.raises(ProtocolError, match="answer-bearing fields"):
        _score(tmp_path, rows)


def test_layer_tokenizer_or_prompt_mismatch_is_rejected(tmp_path):
    rows = _rows()
    rows[0]["layer_outputs"][0]["generated_text"] = "fabricated"
    with pytest.raises(ProtocolError, match="decoded ids"):
        _score(tmp_path, rows)
    rows = _rows()
    rows[0]["prompt"] = "different"
    with pytest.raises(ProtocolError, match="prompt differs"):
        _score(tmp_path, rows)


def test_missing_complete_layer_group_is_rejected(tmp_path):
    rows = _rows()
    rows[0]["layer_outputs"] = rows[0]["layer_outputs"][:2]
    with pytest.raises(ProtocolError, match="layer group"):
        _score(tmp_path, rows)


# ---------------------------------------------------------------------------
# Review-only preflight


def test_preflight_review_is_fresh_and_launch_fails_closed(tmp_path):
    review = r1.preflight()
    assert review["status"] == "review_ready"
    assert review["exit_code"] == 0
    assert review["native_execution"] is False
    with pytest.raises(ProtocolError, match="authorization"):
        r1.preflight(launch=True)
    config_sha = sha256_file(CONFIG)
    manifest = read_json(PREPARED / "manifest.json")
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({
        "format": "replica.dc_protocol.authorization.v1",
        "protocol_id": r1.PROTOCOL_ID,
        "config_sha256": config_sha,
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "authorized_by": "test",
        "native_execution_allowed": True,
        "single_attempt": True,
        "receipt_sha256": "2" * 64,
    }), encoding="utf-8")
    launch = r1.preflight(launch=True, authorization=authorization)
    assert launch["status"] == "refused_native_runner_absent"
    assert launch["exit_code"] == 2
    assert launch["native_execution"] is False


def test_review_receipt_path_must_be_fresh(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolError, match="not fresh"):
        require_fresh_path(path, "preflight_receipt")
