"""Versioned DC-R1 v2 preparation, abstention, compressed and budget contracts.

These tests are CPU-only: they build synthetic prediction rows and use an
explicit character codec as a tiny decode stand-in.  They do not load a model or
tokenizer, do not execute generation and make no learning claim.
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

import dc_r1_native_value_v2_20261027 as r  # noqa: E402
from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, read_json, read_jsonl, require_fresh_path, sha256_file,
)

PREPARED = r.PREPARED_DIR
CONFIG = r.CONFIG_PATH


def _encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def _decode(ids) -> str:
    return "".join(chr(item) for item in ids)


def _config() -> dict:
    return read_json(CONFIG)


def _binding(config: dict) -> dict:
    return r.validate_tokenizer_binding(config["tokenizer_binding"])


def _labels(prefix: str = "dev") -> dict[str, dict]:
    name = "sealed_labels_dev.jsonl" if prefix == "dev" else \
        "sealed_labels_final.jsonl"
    return {row["case_id"]: row for row in read_jsonl(PREPARED / name)}


def _cases(prefix: str = "dev") -> dict[str, dict]:
    return {row["case_id"]: row
            for row in read_jsonl(PREPARED / f"{prefix}_cases.jsonl")}


def _conditions(prefix: str = "dev") -> dict[tuple[str, str], dict]:
    rows = {}
    for condition in r.CONDITIONS:
        for row in read_jsonl(
                PREPARED / f"{prefix}_conditions_{condition}.jsonl"):
            rows[(condition, row["case_id"])] = row
    return rows


def _invented(case_id: str, layer: int) -> str:
    return f"invented_{case_id}_{layer}"


def _row(condition: str, condition_row: dict, case: dict,
         texts: list[str], config: dict) -> dict:
    binding = _binding(config)
    outputs = []
    for layer, text in zip(case["layer_group"], texts):
        ids = _encode(text)
        prompt_tokens = len(condition_row["prompt"])
        count = len(ids)
        outputs.append({
            "layer": layer,
            "physical_stream_id": f"{condition}/{case['case_id']}/layer:{layer}",
            "generated_ids": ids,
            "generated_text": text,
            "empty_generation": text == "",
            "stopped": True,
            "stop_reason": "eos",
            "work": {
                "generation_requests": 1,
                "generation_prefill_forwards": 1,
                "generation_decode_forwards": count - 1,
                "autoregressive_forwards": count,
                "forward_calls": count,
                "generated_tokens": count,
                "prefill_tokens": prompt_tokens,
                "processed_positions": (
                    count * prompt_tokens + count * (count - 1) // 2),
                "layer_readout_streams": 1,
            },
        })
    return {
        "format": r.PREDICTION_FORMAT,
        "protocol_id": r.PROTOCOL_ID,
        "condition": condition,
        "case_id": case["case_id"],
        "prompt": condition_row["prompt"],
        "prompt_sha256": condition_row["prompt_sha256"],
        "layer_outputs": outputs,
        "source_context_present": condition_row["source_context_present"],
        "source_kv_loaded": False,
        "prompt_source_spans": 0,
        "tokenizer_binding_sha256": r.tokenizer_binding_sha(binding),
        "decode_policy": binding["decode_policy"],
        "routing_metadata": {"mode": "fixed", "expert_id": 0},
    }


def _rows(*, no_memory_leaks: bool = False,
          selected_only_one_layer: bool = False,
          no_memory_one_unknown_two_invented: bool = False,
          no_memory_accepted_shortcut: bool = False) -> list[dict]:
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
            text = r.decode_compressed_literal(
                codec, labels[case_id]["compressed_code"])
            texts = [text] * 3
        elif condition == "wrong_value":
            texts = [condition_row["evidence"]["literal"]] * 3
        else:
            texts = ["UNKNOWN"] * 3
            if no_memory_leaks:
                texts = [target] * 3
            elif no_memory_one_unknown_two_invented:
                texts = ["UNKNOWN",
                         _invented(case_id, 27),
                         _invented(case_id, 41)]
            elif no_memory_accepted_shortcut:
                texts = [target,
                         _invented(case_id, 27),
                         _invented(case_id, 41)]
        if selected_only_one_layer and condition == "selected_literal":
            texts = [target, "UNKNOWN", "UNKNOWN"]
        rows.append(_row(condition, condition_row, case, texts, config))
    return rows


def _score_dev(tmp_path: Path, rows: list[dict], *,
               prepared: Path = PREPARED) -> dict:
    path = tmp_path / "predictions.jsonl"
    from dc_protocol_common_20261027 import write_jsonl
    write_jsonl(path, rows)
    return r.score_dev(
        path, prepared=prepared, config_path=CONFIG, decode_fn=_decode)


def _dev_report(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    report = _score_dev(tmp_path, _rows() if rows is None else rows)
    path = tmp_path / "dev_score_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "fixture"
    shutil.copytree(PREPARED, target)
    return target


# ---------------------------------------------------------------------------
# Versioned fixture, lineage and compressed channel


def test_fixture_is_deterministic_versioned_and_preserves_v1_evidence():
    first = r.validate_prepared()
    second = r.validate_prepared()
    assert first["status"] == "valid"
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["counts"]["dev_cases"] == 12
    assert first["counts"]["sealed_final_cases"] == 4
    assert first["counts"]["dev_condition_rows"] == 60
    assert first["compressed_reconstruction"]["dev"] == 0.75
    assert first["compressed_reconstruction"]["sealed_final"] == 0.75
    assert first["lineage"]["amends_protocol_id"] == r.V1_PROTOCOL_ID
    assert first["lineage"]["v1_preserved"] is True
    assert first["native_computation"] is False
    assert first["heldout_opened"] is False
    # v1 files and evidence are untouched and still hash to the declared bytes.
    assert sha256_file(r.V1_CONFIG_PATH) == r.V1_CONFIG_SHA256
    assert sha256_file(r.V1_PREPARED_DIR / "manifest.json") \
        == r.V1_MANIFEST_SHA256
    v1_manifest = read_json(r.V1_PREPARED_DIR / "manifest.json")
    assert v1_manifest["protocol_id"] == r.V1_PROTOCOL_ID
    v2_manifest = read_json(PREPARED / "manifest.json")
    assert v2_manifest["protocol_id"] == r.PROTOCOL_ID
    assert v2_manifest["format"] == r.MANIFEST_FORMAT
    assert v2_manifest["lineage"]["amends_manifest_sha256"] \
        == r.V1_MANIFEST_SHA256


def test_compressed_prompt_is_codec_id_plus_integer_only():
    codec = read_json(PREPARED / "codec.json")
    entry_literals = {literal
                      for entry in codec["entries"]
                      for literal in entry["members"]}
    for prefix in ("dev", "sealed_final"):
        labels = _labels(prefix)
        for (condition, case_id), row in _conditions(prefix).items():
            if condition != "compressed_record":
                continue
            evidence = row["evidence"]
            assert evidence["kind"] == "compressed_code"
            assert evidence["codec_id"] == codec["codec_id"]
            assert evidence["native_input_fields"] == ["codec_id", "code"]
            assert evidence["host_codec_table_visible_to_model"] is False
            assert set(evidence) == {
                "kind", "codec_id", "code", "native_input_fields",
                "host_codec_table_visible_to_model"}
            assert "members" not in row["prompt"]
            assert "entries" not in row["prompt"]
            assert "canonical_literal" not in row["prompt"]
            for literal in entry_literals:
                assert literal.casefold() not in row["prompt"].casefold()
            assert labels[case_id]["target"].casefold() \
                not in row["prompt"].casefold()
            assert f"Code: {evidence['code']}" in row["prompt"]
    # The host table exists only for the separate reconstruction audit.
    canonical = r.decode_compressed_literal(codec, 0)
    assert isinstance(canonical, str) and canonical


def test_host_reconstruction_is_not_a_native_compressed_gate(tmp_path):
    report = _score_dev(tmp_path, _rows())
    control = report["gates"]["compressed_reconstruction_control"]
    assert control["passed"] is True
    assert control["evidence_role"] == (
        "host_reconstruction_audit_control_only")
    assert control["native_readout_evidence"] is False
    for name in ("compressed_accepted_native",
                 "compressed_complete_native"):
        gate = report["gates"][name]
        assert gate["blocked"] is True
        assert gate["evaluable"] is False
        assert gate["passed"] is False
        assert gate["evidence_status"] == (
            "blocked_no_native_compressed_channel_receipt")
        assert gate["not_a_host_reconstruction_gate"] is True
        assert gate["reported_actual_output_metric"] is True
    channel = report["compressed_channel"]
    assert channel["native_input_fields"] == ["codec_id", "code"]
    assert channel["host_codec_table_visible_to_model"] is False
    assert channel["host_decode_is_native_readout_evidence"] is False
    assert channel["native_compressed_readout_status"] == (
        "blocked_no_native_compressed_channel_receipt")
    assert set(channel["native_compressed_gates"]) == {
        "compressed_accepted_native", "compressed_complete_native"}
    assert channel["actual_generated_compressed_metrics"][
        "accepted_layer_accuracy"] == 0.75
    assert report["status"] == "native_compressed_channel_blocked"
    assert report["readout_useful"] is False


# ---------------------------------------------------------------------------
# All-layer group abstention and actual-output metrics


def test_one_unknown_and_two_invented_answers_are_not_abstention(tmp_path):
    report = _score_dev(
        tmp_path, _rows(no_memory_one_unknown_two_invented=True))
    no_memory = report["conditions"]["no_memory"]
    assert no_memory["group_abstention_rate"] == 0.0
    assert no_memory["group_non_abstention_rate"] == 1.0
    assert no_memory["any_layer_abstention_rate_informational_only"] == 1.0
    assert no_memory["per_layer_non_abstention_rate"] == {
        "13": 0.0, "27": 1.0, "41": 1.0}
    assert no_memory["per_layer_hallucination_rate"] == {
        "13": 0.0, "27": 1.0, "41": 1.0}
    assert no_memory["hallucination_rate"] == 1.0
    assert report["gates"]["no_memory_group_abstain"]["passed"] is False
    assert report["gates"]["no_memory_group_non_abstain"]["passed"] is False
    assert report["gates"]["no_memory_per_layer_non_abstain"]["passed"] is False
    assert report["gates"]["no_memory_hallucination"]["passed"] is False
    assert report["status"] == "readout_not_useful"
    assert report["development_evidence_admissible"] is False
    # The actual layer outputs, not a host flag, drive the outcome.
    first = report["accepted_layer_vs_complete_group"]["no_memory"]
    assert first["group_abstention_rate"] == 0.0
    assert first["accepted_layer_accuracy"] == 0.0


def test_accepted_layer_cannot_hide_group_non_abstention(tmp_path):
    report = _score_dev(
        tmp_path, _rows(no_memory_accepted_shortcut=True))
    no_memory = report["conditions"]["no_memory"]
    assert no_memory["accepted_layer_accuracy"] == 1.0
    assert no_memory["complete_group_accuracy"] == 0.0
    assert no_memory["group_abstention_rate"] == 0.0
    assert no_memory["group_non_abstention_rate"] == 1.0
    assert no_memory["hallucination_rate"] == 1.0
    assert report["gates"]["no_memory_group_abstain"]["passed"] is False
    assert report["gates"]["no_memory_hallucination"]["passed"] is False
    assert report["status"] == "readout_not_useful"


def test_host_abstention_flags_are_rejected(tmp_path):
    rows = _rows()
    rows[0]["layer_outputs"][0]["abstain"] = True
    with pytest.raises(ProtocolError, match="host-computed outcome"):
        _score_dev(tmp_path, rows)
    rows = _rows()
    rows[0]["group_abstained"] = True
    with pytest.raises(ProtocolError, match="host-computed outcome"):
        _score_dev(tmp_path, rows)


def test_accepted_layer_is_reported_separately_from_complete_group(tmp_path):
    report = _score_dev(tmp_path, _rows(selected_only_one_layer=True))
    selected = report["accepted_layer_vs_complete_group"]["selected_literal"]
    assert selected["accepted_layer_accuracy"] == 1.0
    assert selected["complete_group_accuracy"] == 0.0
    assert selected["gap"] == 1.0
    assert report["gates"]["selected_literal_complete"]["passed"] is False
    assert report["status"] == "readout_not_useful"


def test_no_memory_target_leak_fails_group_abstention(tmp_path):
    report = _score_dev(tmp_path, _rows(no_memory_leaks=True))
    no_memory = report["conditions"]["no_memory"]
    assert no_memory["group_abstention_rate"] == 0.0
    assert no_memory["group_non_abstention_rate"] == 1.0
    assert report["gates"]["no_memory_group_abstain"]["passed"] is False
    assert report["status"] == "readout_not_useful"


def test_wrong_value_control_cannot_pass_readout(tmp_path):
    report = _score_dev(tmp_path, _rows())
    wrong = report["conditions"]["wrong_value"]
    assert wrong["accepted_layer_accuracy"] == 0.0
    assert wrong["complete_group_accuracy"] == 0.0
    assert wrong["wrong_value_follow_rate"] == 1.0
    assert report["gates"]["wrong_value_correct_target"]["value"] == 0.0
    assert report["readout_useful"] is False


def test_answer_index_and_target_routing_are_rejected(tmp_path):
    rows = _rows()
    rows[0]["answer_index"] = 2
    with pytest.raises(ProtocolError, match="answer-bearing fields"):
        _score_dev(tmp_path, rows)
    rows = _rows()
    rows[0]["routing_metadata"]["gold"] = "leak"
    with pytest.raises(ProtocolError, match="answer-bearing fields"):
        _score_dev(tmp_path, rows)


def test_generated_ids_and_layer_group_are_verified(tmp_path):
    rows = _rows()
    rows[0]["layer_outputs"][0]["generated_text"] = "fabricated"
    with pytest.raises(ProtocolError, match="decoded ids"):
        _score_dev(tmp_path, rows)
    rows = _rows()
    rows[0]["layer_outputs"] = rows[0]["layer_outputs"][:2]
    with pytest.raises(ProtocolError, match="layer group"):
        _score_dev(tmp_path, rows)


# ---------------------------------------------------------------------------
# Corrected physical-work budget


def test_physical_budget_separates_units_and_preserves_240():
    config = _config()
    budget = config["budget"]
    assert budget["physical_limits"] == r.expected_physical_limits()
    assert budget["physical_limits"]["max_generation_requests"] == 240
    assert budget["physical_limits"]["max_prefill_forwards"] == 240
    assert budget["physical_limits"]["max_decode_forwards"] == 1680
    assert budget["physical_limits"]["max_forward_calls"] == 1920
    assert budget["physical_limits"]["max_generated_tokens"] == 1920
    assert budget["physical_limits"]["max_layer_readout_streams"] == 240
    assert budget["physical_limits"]["max_prefill_tokens"] == 30720
    assert budget["physical_limits"]["max_processed_positions"] == 252480
    schedule = budget["schedule"]
    assert schedule["mode"] == "sequential_per_layer_no_shared_prefill"
    assert schedule["prefill_token_ceiling_per_request"] == 128
    assert schedule["max_generated_tokens_per_request"] == 8
    assert schedule["processed_positions_ceiling_per_request"] == 1052
    assert schedule["decode_forwards_formula"] == (
        "generated_tokens - active_requests")
    assert schedule["batched_or_shared_prefill"] is False
    historical = budget["historical_v1_limits"]
    assert historical["max_forward_calls"] == 240
    assert historical["max_generation_calls"] == 240
    assert historical["max_generated_tokens"] == 1920
    assert historical["max_prefill_tokens"] == 30720
    accounting = budget["historical_request_accounting"]
    assert accounting["value"] == 240
    assert accounting["status"] == (
        "incorrect_historical_request_accounting_preserved_not_a_cap")
    assert accounting["corrected_generation_requests"] == 240
    assert accounting["corrected_layer_readout_streams"] == 240
    assert accounting["corrected_prefill_forwards"] == 240
    assert accounting["corrected_decode_forwards"] == 1680
    assert accounting["corrected_forward_calls"] == 1920
    assert accounting["corrected_prefill_tokens"] == 30720
    assert accounting["corrected_processed_positions"] == 252480
    total = budget["workload_derivation"]["total"]
    assert total["generation_requests"] == 240
    assert total["prefill_forwards"] == 240
    assert total["decode_forwards_max"] == 1680
    assert total["generated_tokens_max"] == 1920
    assert total["layer_readout_streams"] == 240
    assert total["forward_calls_max"] == 1920
    assert total["prefill_tokens_max"] == 30720
    assert total["processed_positions_max"] == 252480
    assert total["conditions"] == 5
    assert total["layers"] == 3


def test_sequential_counter_identities_and_early_stop_are_upper_bounds():
    requests = 240
    full = r.sequential_request_work(requests)
    assert full["generation_requests"] == 240
    assert full["layer_readout_streams"] == 240
    assert full["prefill_forwards"] == 240
    assert full["generated_tokens"] == 1920
    assert full["decode_forwards"] == 1680
    assert full["forward_calls"] == 1920
    assert full["prefill_tokens"] == 30720
    assert full["processed_positions_ceiling"] == 252480
    assert full["forward_calls"] == (
        full["prefill_forwards"] + full["decode_forwards"])
    assert full["decode_forwards"] == (
        full["generated_tokens"] - full["generation_requests"])
    # Early stop: fewer generated tokens, and some requests can emit zero tokens.
    early = r.sequential_request_work(
        requests, generated_tokens=1000, active_requests=200)
    assert early["decode_forwards"] == 800
    assert early["forward_calls"] == 1040
    assert early["forward_calls"] <= full["forward_calls"]
    assert early["generated_tokens"] <= full["generated_tokens"]
    stopped = r.sequential_request_work(
        requests, generated_tokens=0, active_requests=0)
    assert stopped["decode_forwards"] == 0
    assert stopped["forward_calls"] == 240
    assert stopped["forward_calls"] <= full["forward_calls"]
    # No-cache growing-prefix positions: T*L + T*(T-1)/2.
    assert r.processed_positions_for_request(128, 0) == 0
    assert r.processed_positions_for_request(128, 1) == 128
    assert r.processed_positions_for_request(128, 8) == 1052
    assert r.processed_positions_for_request(100, 8) == 828
    for prompt in (1, 64, 128):
        for generated in range(9):
            assert r.processed_positions_for_request(
                prompt, generated) <= 1052


def test_condition_and_layer_breakdowns_cover_all_five_and_three():
    budget = _config()["budget"]
    for split, cases in (("dev", 12), ("sealed_final", 4)):
        requests = cases * 5 * 3
        per_condition = budget["condition_breakdown"][
            "per_split_condition"][split]
        assert set(per_condition) == set(r.CONDITIONS)
        assert sum(row["generation_requests"]
                   for row in per_condition.values()) == requests
        assert sum(row["prefill_forwards"]
                   for row in per_condition.values()) == requests
        assert sum(row["layer_readout_streams"]
                   for row in per_condition.values()) == requests
        assert sum(row["generated_tokens"]
                   for row in per_condition.values()) == requests * 8
        assert sum(row["decode_forwards"]
                   for row in per_condition.values()) == requests * 8 - requests
        assert sum(row["forward_calls"]
                   for row in per_condition.values()) == requests * 8
        assert sum(row["prefill_tokens"]
                   for row in per_condition.values()) == requests * 128
        assert sum(row["processed_positions"]
                   for row in per_condition.values()) == requests * 1052
        # Dev per-condition row exactly matches Main's sequential schedule.
        if split == "dev":
            row = per_condition["exact_context"]
            assert row == {
                "generation_requests": 36,
                "prefill_forwards": 36,
                "prefill_tokens": 4608,
                "layer_readout_streams": 36,
                "generated_tokens": 288,
                "decode_forwards": 252,
                "forward_calls": 288,
                "processed_positions": 37872,
            }
        else:
            row = per_condition["exact_context"]
            assert row == {
                "generation_requests": 12,
                "prefill_forwards": 12,
                "prefill_tokens": 1536,
                "layer_readout_streams": 12,
                "generated_tokens": 96,
                "decode_forwards": 84,
                "forward_calls": 96,
                "processed_positions": 12624,
            }
        per_layer = budget["layer_breakdown"]["per_split_layer"][split]
        assert set(per_layer) == {"13", "27", "41"}
        layer_requests = cases * 5
        for row in per_layer.values():
            assert row["generation_requests"] == layer_requests
            assert row["layer_readout_streams"] == layer_requests
            assert row["prefill_forwards"] == layer_requests
            assert row["generated_tokens"] == layer_requests * 8
            assert row["decode_forwards"] == layer_requests * 8 - layer_requests
            assert row["forward_calls"] == layer_requests * 8
            assert row["prefill_tokens"] == layer_requests * 128
            assert row["processed_positions"] == layer_requests * 1052
        assert sum(row["generation_requests"]
                   for row in per_layer.values()) == requests
        assert sum(row["generated_tokens"]
                   for row in per_layer.values()) == requests * 8
        assert sum(row["processed_positions"]
                   for row in per_layer.values()) == requests * 1052


def test_shared_v2_checker_enforces_corrected_caps():
    config = _config()
    exact = {
        "generation_requests": 240,
        "prefill_forwards": 240,
        "decode_forwards": 1680,
        "layer_readout_streams": 240,
        "forward_calls": 1920,
        "generated_tokens": 1920,
        "prefill_tokens": 30720,
        "processed_positions": 252480,
        "model_loads": 1,
        "tokenizer_loads": 1,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_writes": 0,
        "downloads": 0,
        "retries": 0,
        "workers": 1,
    }
    result = r.check_physical_work(exact, config)
    assert result["format"] == "replica.dc_protocol.work_budget.v2"
    assert result["within_budget"] is True
    with pytest.raises(ProtocolError, match="exceeds"):
        r.check_physical_work({"forward_calls": 1921}, config)
    with pytest.raises(ProtocolError, match="exceeds"):
        r.check_physical_work({"generation_requests": 241}, config)
    with pytest.raises(ProtocolError, match="exceeds"):
        r.check_physical_work({"processed_positions": 252481}, config)
    with pytest.raises(ProtocolError, match="unrecognized"):
        r.check_physical_work({"not_a_counter": 1}, config)
    # Early stop is a <= ceiling, not an exact-only constant check.
    early = {
        "generation_requests": 240,
        "prefill_forwards": 240,
        "decode_forwards": 800,
        "forward_calls": 1040,
        "generated_tokens": 1000,
        "processed_positions": 100000,
    }
    assert r.check_physical_work(early, config)["within_budget"] is True


def test_amendments_cannot_overwrite_v1_limits_or_reintroduce_any_layer():
    config = _config()
    broken = deepcopy(config)
    broken["budget"]["historical_v1_limits"]["max_forward_calls"] = 241
    with pytest.raises(ProtocolError, match="preserved exactly"):
        r.validate_config(broken)
    broken = deepcopy(config)
    broken["abstention_policy"]["any_layer_abstention_acceptance"] = True
    with pytest.raises(ProtocolError, match="any-layer abstention"):
        r.validate_config(broken)
    broken = deepcopy(config)
    broken["budget"]["schedule"]["batched_or_shared_prefill"] = True
    with pytest.raises(ProtocolError, match="schedule differs"):
        r.validate_config(broken)
    broken = deepcopy(config)
    broken["budget"]["physical_limits"]["max_decode_forwards"] = 1920
    with pytest.raises(ProtocolError):
        r.validate_config(broken)
    broken = deepcopy(config)
    broken["evaluation"]["gates"]["exact_context_accepted_min"] = 0.79
    with pytest.raises(ProtocolError):
        r.validate_config(broken)


# ---------------------------------------------------------------------------
# Final-split admission and review-only preflight


def test_final_scoring_requires_admitted_dev_report(tmp_path):
    report_path = _dev_report(tmp_path)
    dev_report = read_json(report_path)
    assert dev_report["split"] == "dev"
    assert dev_report["final_labels_opened"] is False
    assert dev_report["development_evidence_admissible"] is True
    manifest_sha = dev_report["prepared_manifest_sha256"]
    with pytest.raises(ProtocolError, match="requires --dev-report"):
        r.admit_dev_report(None, config_path=CONFIG,
                           prepared_manifest_sha256=manifest_sha,
                           dev_predictions_path=None, decode_fn=_decode)
    admission = r.admit_dev_report(
        report_path, config_path=CONFIG,
        prepared_manifest_sha256=manifest_sha,
        dev_predictions_path=tmp_path / "predictions.jsonl",
        decode_fn=_decode)
    assert admission["admitted"] is True

    broken = deepcopy(dev_report)
    broken["development_evidence_admissible"] = False
    path = tmp_path / "broken_dev_report.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ProtocolError, match="not admissible"):
        r.admit_dev_report(path, config_path=CONFIG,
                           prepared_manifest_sha256=manifest_sha,
                           dev_predictions_path=tmp_path / "predictions.jsonl",
                           decode_fn=_decode)
    broken = deepcopy(dev_report)
    broken["final_labels_opened"] = True
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ProtocolError, match="must not have opened"):
        r.admit_dev_report(path, config_path=CONFIG,
                           prepared_manifest_sha256=manifest_sha,
                           dev_predictions_path=tmp_path / "predictions.jsonl",
                           decode_fn=_decode)



def test_final_admission_rejects_a_tampered_development_report(tmp_path):
    report_path = _dev_report(tmp_path)
    manifest_sha = read_json(report_path)["prepared_manifest_sha256"]
    broken = read_json(report_path)
    broken["conditions"]["no_memory"]["group_abstention_rate"] = 0.0
    path = tmp_path / "tampered_dev_report.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ProtocolError, match="independently recomputed"):
        r.admit_dev_report(
            path, config_path=CONFIG, prepared_manifest_sha256=manifest_sha,
            dev_predictions_path=tmp_path / "predictions.jsonl",
            decode_fn=_decode)

def test_final_scoring_does_not_open_labels_before_admission(tmp_path):
    prepared = _copy_fixture(tmp_path)
    # Remove a final label file.  If the implementation validated or parsed
    # final labels before admitting the dev report, the error would be a
    # missing/ corrupt fixture file instead of the admission refusal.
    (prepared / "sealed_labels_final.jsonl").unlink()
    with pytest.raises(ProtocolError, match="requires --dev-report"):
        r.score_sealed_final(
            tmp_path / "predictions.jsonl", dev_report_path=None,
            dev_predictions_path=None,
            prepared=prepared, config_path=CONFIG, decode_fn=_decode)


def test_preflight_review_is_fresh_and_launch_fails_closed(tmp_path):
    review = r.preflight()
    assert review["status"] == "review_ready"
    assert review["exit_code"] == 0
    assert review["native_execution"] is False
    readiness = review["readiness"]
    assert readiness["final_labels_openable"] is False
    assert readiness["final_scoring_requires_dev_admission"] is True
    assert readiness["native_compressed_gate_status"] == (
        "blocked_no_native_compressed_channel_receipt")
    assert readiness["v1_config_sha256"] == r.V1_CONFIG_SHA256
    assert readiness["v1_manifest_sha256"] == r.V1_MANIFEST_SHA256
    assert review["budget_readiness"]["historical_request_accounting"][
        "value"] == 240
    derived = review["budget_readiness"]["derived_workload"]
    assert derived["generation_requests"] == 240
    assert derived["prefill_forwards"] == 240
    assert derived["decode_forwards_max"] == 1680
    assert derived["forward_calls_max"] == 1920
    assert derived["processed_positions_max"] == 252480
    with pytest.raises(ProtocolError, match="authorization"):
        r.preflight(launch=True)
    manifest = read_json(PREPARED / "manifest.json")
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({
        "format": "replica.dc_protocol.authorization.v1",
        "protocol_id": r.PROTOCOL_ID,
        "config_sha256": sha256_file(CONFIG),
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "authorized_by": "test",
        "native_execution_allowed": True,
        "single_attempt": True,
        "receipt_sha256": "2" * 64,
    }), encoding="utf-8")
    launch = r.preflight(launch=True, authorization=authorization)
    assert launch["status"] == "refused_native_runner_absent"
    assert launch["exit_code"] == 2
    assert launch["native_execution"] is False


def test_review_receipt_path_must_be_fresh(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolError, match="not fresh"):
        require_fresh_path(path, "preflight_receipt")


def test_per_case_work_evidence_is_complete_and_stream_specific(tmp_path):
    report = _score_dev(tmp_path, _rows())
    physical = report["physical_work"]
    assert physical["complete"] is True
    assert physical["budget_check"] == "not_applicable_non_native_preparation"
    totals = physical["totals"]
    assert totals["generation_requests"] == 180
    assert totals["generation_prefill_forwards"] == 180
    assert totals["layer_readout_streams"] == 180
    assert totals["autoregressive_forwards"] == (
        totals["generation_prefill_forwards"]
        + totals["generation_decode_forwards"])
    assert totals["forward_calls"] == totals["generated_tokens"]
    assert len(report["per_case_outputs"]) == 60
    streams = [item["physical_stream_id"]
               for row in report["per_case_outputs"] for item in row["layers"]]
    assert len(streams) == len(set(streams)) == 180
    for row in report["per_case_outputs"]:
        for item in row["layers"]:
            count = len(item["generated_ids"])
            work = item["work"]
            assert work["generation_decode_forwards"] == count - 1
            assert work["autoregressive_forwards"] == count
            assert work["generated_tokens"] == count
            assert work["processed_positions"] == (
                count * work["prefill_tokens"] + count * (count - 1) // 2)
            assert item["completion_valid"] is True
    with pytest.raises(ProtocolError, match="processed_positions"):
        r._work_evidence(
            report["per_case_outputs"], config=_config(), enforce_budget=True)


def test_truncated_layer_output_is_not_correct_or_abstaining(tmp_path):
    rows = _rows()
    rows[0]["layer_outputs"][0]["stop_reason"] = "max_new_tokens"
    report = _score_dev(tmp_path, rows)
    item = report["per_case_outputs"][0]["layers"][0]
    assert item["completion_valid"] is False
    assert item["correct"] is False
    assert item["abstained"] is False
    assert item["invented_non_target"] is True


def test_duplicate_physical_stream_identity_is_rejected(tmp_path):
    report = _score_dev(tmp_path, _rows())
    evidence = deepcopy(report["per_case_outputs"])
    evidence[0]["layers"][1]["physical_stream_id"] = \
        evidence[0]["layers"][0]["physical_stream_id"]
    with pytest.raises(ProtocolError, match="duplicate physical layer stream"):
        r._work_evidence(evidence, config=_config())
