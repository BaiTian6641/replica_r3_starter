"""DC-P1 v2 protocol, fixture, work-unit and preflight contracts.

These tests use the deterministic text fixture, a synthetic map-based commit
verifier and a synthetic token-length artifact.  They load no MiniCPM5 model, no
tokenizer, no optimizer and no native runtime, and they create no authorization.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import dc_p1_native_response_v2_20261027 as p1v2  # noqa: E402
from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, check_work_budget_v2, find_forbidden_fields, read_json,
    read_jsonl, seal_record, sha256_file, sha256_json,
)

PREPARED = p1v2.PREPARED_DIR
CONFIG = p1v2.CONFIG_PATH


def _config() -> dict:
    return read_json(CONFIG)


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _authorization(tmp_path: Path, **overrides) -> Path:
    config = _config()
    manifest = read_json(PREPARED / "manifest.json")
    body = {
        "format": p1v2.AUTHORIZATION_FORMAT_V2,
        "protocol_id": p1v2.PROTOCOL_ID,
        "config_path": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "prepared_manifest_path": str(PREPARED / "manifest.json"),
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "protocol_commit": "a" * 40,
        "runner_path": config["launch"]["runner_path"],
        "runner_sha256": sha256_file(ROOT / config["launch"]["runner_path"]),
        "authorized_by": "external-review-board",
        "native_execution_allowed": True,
        "single_attempt": True,
        "max_optimizer_steps": 120,
        "max_generation_requests": 416,
        "restart_allowance": 1,
    }
    body.update(overrides)
    sealed = seal_record(
        {key: value for key, value in body.items() if value is not None})
    return _write_json(tmp_path / "authorization.json", sealed)


def _commit_verifier(payloads: dict[str, bytes]):
    def read(commit: str, path: str):
        if commit != "a" * 40:
            return None
        for name, payload in payloads.items():
            if str(path).endswith(name):
                return payload
        return None
    return read


def _real_commit_payloads() -> dict[str, bytes]:
    from replica_lab import dc_p1_native_runner as runner

    config = _config()
    payloads = {
        "dc_p1_native_response_protocol_v2.json": CONFIG.read_bytes(),
        "manifest.json": (PREPARED / "manifest.json").read_bytes(),
        "teaching.jsonl": (PREPARED / "teaching.jsonl").read_bytes(),
    }
    for relative in runner.IMPLEMENTATION_RELATIVES:
        payloads[relative] = (ROOT / relative).read_bytes()
    return payloads


# ---------------------------------------------------------------------------
# Config, amendments and lineage


def test_config_is_valid_and_declares_the_versioned_amendments():
    report = p1v2.validate_config(_config())
    assert report["status"] == "valid"
    assert report["protocol_id"] == "dc_p1_native_response_v2"
    assert report["declared_total_updates"] == 120
    assert report["declared_claim"] == "global_ce_rescale"
    assert set(report["amendments"]) == {
        "dc_p1_v2_a1_matched_control", "dc_p1_v2_a2_work_unit_correction",
        "dc_p1_v2_a3_staged_admission", "dc_p1_v2_a4_temporal_semantics",
        "dc_p1_v2_a5_objective_audit", "dc_p1_v2_a6_final_scope"}


def test_v1_limits_and_the_incorrect_368_are_preserved_not_reused():
    config = _config()
    lineage = p1v2.validate_lineage(config)
    assert lineage["v1_forward_calls"] == 368
    assert lineage["v1_optimizer_steps"] == 48
    historical = config["budget"]["historical_v1"]
    assert historical["accounting_status"] == (
        "incorrect_request_equals_forward_accounting")
    assert historical["reused_as_a_ceiling"] is False
    physical = config["budget"]["physical_limits"]
    assert physical["max_forward_calls"] == 6780
    assert physical["max_forward_calls"] != 368
    # The v1 values are still readable as data even though v2 does not use them.
    assert historical["physical_limits"]["max_checkpoint_writes"] == 4
    assert physical["max_checkpoint_writes"] == 121
    assert config["prepared"]["root"] != "data/dc_p1_native_response_20261027"


def test_amendments_may_not_be_silent_or_drop_the_old_limit():
    config = deepcopy(_config())
    config["amendments"][0]["silent"] = True
    with pytest.raises(ProtocolError, match="may not be silent"):
        p1v2.validate_config(config)
    config = deepcopy(_config())
    config["amendments"][1]["old_limit_preserved"] = False
    with pytest.raises(ProtocolError, match="preserve the old limit"):
        p1v2.validate_config(config)
    config = deepcopy(_config())
    del config["amendments"][0]
    with pytest.raises(ProtocolError, match="at least the six"):
        p1v2.validate_config(config)


def test_matched_control_is_declared_and_final_is_confirmation_only():
    config = _config()
    arms = config["controls"]["arms"]
    assert [arm["arm_id"] for arm in arms] == [
        "treatment", "matched_trained", "no_update", "reset_P"]
    assert [arm["answer_span_ce_weight"] for arm in arms] == [
        0.5, 0.0, 0.5, 0.5]
    assert [arm["updates"] for arm in arms] == [36, 36, 0, 36]
    assert arms[3]["restore_parent_before_evaluation"] is True
    assert config["controls"]["matched_control_explicit"] is True
    assert config["controls"]["budget_not_silently_raised"] is True
    final = config["controls"]["final_stage"]
    assert final["arms"] == ["treatment"]
    assert final["updates"] == 12
    assert final["cases"] == 20
    assert final["role"] == "confirmation_only"
    assert final["comparison"] is None
    declaration = config["budget"]["amendment_work_declaration"]
    assert declaration["matched_control_updates"] == 36
    assert declaration["matched_control_generation_requests"] == 96


def test_span_specific_claim_cannot_be_declared_for_coincident_masks():
    config = deepcopy(_config())
    config["objective"]["span_specific_claim_allowed"] = True
    with pytest.raises(ProtocolError, match="span-specific claim"):
        p1v2.validate_config(config)
    config = deepcopy(_config())
    config["objective"]["declared_claim"] = "span_specific_weighting"
    with pytest.raises(ProtocolError, match="global CE rescale"):
        p1v2.validate_config(config)
    config = deepcopy(_config())
    config["objective"]["narrowed_claim"]["statement"] = (
        "a span_specific_weighting claim")
    with pytest.raises(ProtocolError, match="span-specific statement"):
        p1v2.validate_config(config)


def test_budget02_and_a_stale_final_comparison_are_refused():
    config = deepcopy(_config())
    config["recipe"]["treatment"]["steps_per_event"] = 2
    with pytest.raises(ProtocolError, match="budget02 revival"):
        p1v2.validate_config(config)
    config = deepcopy(_config())
    config["controls"]["final_stage"]["comparison"] = "answer_span_ce_weight"
    with pytest.raises(ProtocolError, match="0.5-vs-0.0"):
        p1v2.validate_config(config)
    config = deepcopy(_config())
    config["controls"]["arms"][1]["answer_span_ce_weight"] = 0.5
    with pytest.raises(ProtocolError, match="answer-span weights"):
        p1v2.validate_config(config)


def test_guards_optimizer_and_tokenizer_binding_stay_frozen():
    config = _config()
    assert config["resource_guard"]["peak_driver_bytes_max"] == 53687091200
    assert config["resource_guard"]["sustained_swap_growth_bytes_max"] == 536870912
    assert config["resource_guard"]["min_host_free_fraction"] == 0.15
    assert config["optimizer_policy"]["primary"] == "NativeFusedAdamW"
    assert config["optimizer_policy"]["default_name"] == "adamw_fused"
    assert config["optimizer_policy"]["changed"] is False
    assert config["plastic_spec"]["features"]["sites"] == [13, 27]
    assert config["plastic_spec"]["features"]["detached"] is True
    broken = deepcopy(config)
    broken["plastic_spec"]["read_path"]["gain"] = 0.5
    with pytest.raises(ProtocolError, match="read gain"):
        p1v2.validate_config(broken)
    broken = deepcopy(config)
    broken["tokenizer_binding"]["decode_policy"]["answer_selection_allowed"] = True
    with pytest.raises(ProtocolError, match="decode_policy"):
        p1v2.validate_config(broken)


# ---------------------------------------------------------------------------
# Fixture: events, checkpoints and temporal semantics


def test_fixture_is_deterministic_and_binds_events_to_checkpoints():
    first = p1v2.validate_prepared()
    second = p1v2.validate_prepared()
    assert first["status"] == "valid"
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["counts"]["teaching"] == 36
    assert first["counts"]["dev_cases"] == 60
    assert first["counts"]["sealed_final_teaching"] == 12
    assert first["counts"]["sealed_final_cases"] == 20
    assert first["prompt_overlap_with_v1"] == 0
    assert first["heldout_opened"] is False
    teaching = read_jsonl(PREPARED / "teaching.jsonl")
    for row in teaching:
        assert row["checkpoint_id_after"] == (
            f"dcp1v2:{row['source_group']}:ev{row['event_position']:02d}")
        assert row["query_epoch"] == row["event_position"]
        assert row["resolution_mode"] == "explicit_event_reference"
        assert row["response_layout"] == "answer_tokens_then_terminal_eos"
        assert row["eos_id"] == 1
        assert row["state_after_sha256"] == sha256_json(row["state_after"])
    chained = [row for row in teaching
               if row["source_group"] == "p1v2src_dev_00"]
    chained.sort(key=lambda row: row["event_position"])
    assert chained[0]["checkpoint_id_before"] == p1v2.PARENT_CHECKPOINT_ID
    assert chained[1]["checkpoint_id_before"] == chained[0]["checkpoint_id_after"]
    assert chained[2]["checkpoint_id_before"] == chained[1]["checkpoint_id_after"]


def test_ordinary_queries_resolve_current_and_historical_name_the_event():
    labels = {row["case_id"]: row for row in read_jsonl(
        PREPARED / "sealed_labels_dev.jsonl")}
    cases = {row["case_id"]: row for row in read_jsonl(PREPARED / "dev_cases.jsonl")}
    group = "p1v2src_dev_00"
    current = cases[f"{group}_post_correction_current"]
    historical = cases[f"{group}_historical_state"]
    assert current["resolution_mode"] == "current_at_query_epoch"
    assert current["query_epoch"] == 2
    assert historical["resolution_mode"] == "historical_explicit_event"
    assert historical["historical_event_position"] == 0
    assert "event 0" in historical["prompt"]
    current_label = labels[current["case_id"]]
    historical_label = labels[historical["case_id"]]
    assert current_label["resolved_event_position"] == 1
    assert historical_label["resolved_event_position"] == 0
    assert current_label["target"] != historical_label["target"]
    assert current_label["superseded_value"] != current_label["target"]
    assert current_label["state_sha256"] != historical_label["state_sha256"]
    assert historical_label["target"] not in historical["prompt"]
    retention = cases[f"{group}_retention"]
    assert retention["resolution_mode"] == "stateless_rule"
    assert labels[retention["case_id"]]["target"] == (
        labels[retention["case_id"]]["retention_input"][::-1])


def test_evaluation_rows_carry_no_target_source_or_state_digest():
    for name in ("dev_cases.jsonl", "sealed_final_cases.jsonl",
                 "fit_cases.jsonl", "sealed_final_fit_cases.jsonl"):
        for row in read_jsonl(PREPARED / name):
            assert not find_forbidden_fields(row, {"target", "answer", "gold",
                                                   "value", "label"})
            assert "source" not in row
            assert "state_sha256" not in row
            assert "state_after" not in row
            assert "event_checkpoint_id" not in row
            assert row["inference_inputs"] == ["prompt"]
            assert row["source_context_present"] is False
            assert row["source_kv_loaded"] is False


def test_prompts_are_unique_and_sealed_groups_stay_closed():
    teaching = read_jsonl(PREPARED / "teaching.jsonl")
    dev = read_jsonl(PREPARED / "dev_cases.jsonl")
    final = read_jsonl(PREPARED / "sealed_final_cases.jsonl")
    final_teaching = read_jsonl(PREPARED / "sealed_final_teaching.jsonl")
    prompts = [row["prompt"] for row in
               teaching + dev + final + final_teaching]
    assert len(prompts) == len(set(prompts))
    assert ({row["source_group"] for row in teaching}
            == {row["source_group"] for row in dev})
    assert ({row["source_group"] for row in final_teaching}
            == {row["source_group"] for row in final})
    assert not ({row["source_group"] for row in teaching}
                & {row["source_group"] for row in final})


def test_fit_rows_bind_their_teaching_event_and_prompt():
    teaching = {(row["source_group"], row["event_position"]): row
                for row in read_jsonl(PREPARED / "teaching.jsonl")}
    labels = {row["case_id"]: row for row in read_jsonl(
        PREPARED / "sealed_labels_fit.jsonl")}
    for case in read_jsonl(PREPARED / "fit_cases.jsonl"):
        source = teaching[(case["source_group"], case["event_position"])]
        assert case["prompt"] == source["prompt"]
        assert case["prompt_sha256"] == source["prompt_sha256"]
        label = labels[case["case_id"]]
        assert label["target"] == source["target"]
        assert label["resolved_event_position"] == case["event_position"]


# ---------------------------------------------------------------------------
# Work units


def test_schedule_counts_real_execution_units():
    config = _config()
    schedule = p1v2.derive_work_schedule(config)
    counters = schedule["counters"]
    assert counters["generation_requests"] == 416
    assert counters["generation_prefill_forwards"] == 416
    assert counters["generation_decode_forwards"] == 6240
    assert counters["autoregressive_forwards"] == 6656
    assert counters["generated_tokens"] == 6656
    assert counters["training_forwards"] == 120
    assert counters["backward_calls"] == 120
    assert counters["optimizer_steps"] == 120
    assert counters["processed_text_positions"] == 385040
    assert schedule["inputs"]["per_call_positions"] == 888
    assert schedule["development_updates"] == 108
    assert schedule["final_updates"] == 12
    assert counters["forward_calls"] == 6776 + 3 + 1
    assert counters["checkpoint_writes"] == 121
    assert counters["per_event_checkpoint_writes"] == 120
    assert counters["artifact_checkpoint_writes"] == 1
    assert counters["retained_checkpoint_artifacts"] == 4
    assert counters["model_loads"] == 4
    assert counters["tokenizer_loads"] == 7
    assert counters["restart_optimizer_steps"] == 0
    assert counters["temporary_storage_bytes"] == 1615233024


def test_per_arm_work_is_declared_and_matches_the_derivation():
    schedule = p1v2.validate_work_units(_config())
    assert schedule["per_arm"]["treatment"]["generation_requests"] == 96
    assert schedule["per_arm"]["no_update"]["optimizer_steps"] == 0
    assert schedule["per_arm"]["reset_P"]["processed_text_positions"] == 89856
    assert schedule["per_arm"]["treatment_final"]["updates"] == 12
    assert sum(block["updates"] for name, block in schedule["per_arm"].items()
               if name != "treatment_final") == 108


def test_one_more_unit_is_refused_at_the_run_and_arm_level():
    budget = _config()["budget"]
    check_work_budget_v2({"optimizer_steps": 120}, budget)
    with pytest.raises(ProtocolError, match="exceeds"):
        check_work_budget_v2({"optimizer_steps": 121}, budget)
    with pytest.raises(ProtocolError, match="exceeds"):
        check_work_budget_v2({"generation_requests": 417}, budget)
    with pytest.raises(ProtocolError, match="exceeds"):
        check_work_budget_v2({"forward_calls": 6781}, budget)
    with pytest.raises(ProtocolError, match="exceeds"):
        check_work_budget_v2({"checkpoint_writes": 122}, budget)
    with pytest.raises(ProtocolError, match="exceeds"):
        check_work_budget_v2({"restart_optimizer_steps": 1}, budget)
    check_work_budget_v2({"optimizer_steps": 36}, budget, arm="treatment")
    with pytest.raises(ProtocolError, match="exceeds"):
        check_work_budget_v2({"optimizer_steps": 37}, budget, arm="treatment")
    with pytest.raises(ProtocolError, match="no declared per-arm budget"):
        check_work_budget_v2({"optimizer_steps": 1}, budget, arm="unknown_arm")


def test_unknown_counter_and_missing_arm_block_are_refused():
    budget = _config()["budget"]
    with pytest.raises(ProtocolError, match="unrecognized accounting counter"):
        check_work_budget_v2({"mystery_work": 1}, budget)
    stripped = deepcopy(budget)
    del stripped["per_arm_limits"]["matched_trained"]
    with pytest.raises(ProtocolError, match="per-arm budget"):
        check_work_budget_v2({"optimizer_steps": 1}, stripped,
                             arm="matched_trained")


def test_inconsistent_declared_ceiling_is_refused():
    config = deepcopy(_config())
    config["budget"]["physical_limits"]["max_generation_requests"] = 417
    with pytest.raises(ProtocolError, match="differs from the derived"):
        p1v2.validate_work_units(config)


def test_measured_token_lengths_may_only_reduce_positions():
    config = _config()
    prompts = {}
    for name in p1v2.FILES:
        for row in read_jsonl(PREPARED / name):
            if "prompt_sha256" in row:
                prompts[row["prompt_sha256"]] = 12
    artifact = seal_record({
        "format": p1v2.TOKEN_LENGTHS_FORMAT,
        "protocol_id": p1v2.PROTOCOL_ID,
        "tokenizer_binding_sha256": p1v2.tokenizer_binding_sha(
            config["tokenizer_binding"]),
        "maxima": {
            "evaluation_prompt_tokens": 12,
            "teaching_prompt_tokens": 20,
            "teaching_sequence_tokens": 40,
            "retention_prompt_tokens": 12,
        },
        "prompts": prompts,
        "counts": {"prompts": len(prompts)},
    })
    p1v2.validate_token_lengths(config, artifact, prepared=PREPARED)
    measured = p1v2.validate_work_units(config, token_lengths=artifact)
    assert measured["inputs"]["token_lengths_source"] == "measured_sealed_tokenizer"
    assert measured["counters"]["generation_processed_positions"] == (
        416 * (16 * 12 + 120))
    assert measured["counters"]["teaching_processed_positions"] == 120 * 40
    assert measured["counters"]["processed_text_positions"] < (
        config["budget"]["physical_limits"]["max_processed_text_positions"])
    over = seal_record({**artifact, "maxima": {
        **artifact["maxima"], "evaluation_prompt_tokens": 4096}})
    with pytest.raises(ProtocolError, match="exceeds the declared ceiling"):
        p1v2.validate_token_lengths(config, over, prepared=PREPARED)


# ---------------------------------------------------------------------------
# Preflight


def test_preflight_review_records_readiness_without_native_work():
    review = p1v2.preflight()
    assert review["status"] == "review_ready"
    assert review["exit_code"] == 0
    assert review["native_execution"] is False
    assert review["model_loaded"] is False
    assert review["tokenizer_loaded"] is False
    assert review["heldout_opened"] is False
    assert review["authorization_receipt_present"] is False
    assert review["prepared_validation"]["labels_read"] is False
    assert review["token_lengths_status"] in ("pending_tokenizer_step",
                                              "sealed")
    assert review["work_units"]["ceilings"]["max_forward_calls"] == 6780
    assert review["runner_path"] == (
        "scripts/run_dc_p1_native_response_v2_20261027.py")


def test_preflight_launch_requires_an_external_receipt(tmp_path):
    with pytest.raises(ProtocolError, match="authorization"):
        p1v2.preflight(launch=True)
    # Even a present receipt is refused unless it is sealed and commit-bound.
    receipt = _write_json(tmp_path / "authorization.json",
                          {"format": p1v2.AUTHORIZATION_FORMAT_V2})
    with pytest.raises(ProtocolError, match="receipt_sha256"):
        p1v2.preflight(launch=True, authorization=receipt)


def test_preflight_launch_admits_only_a_fully_bound_receipt(tmp_path):
    authorization = _authorization(tmp_path)
    verifier = _commit_verifier(_real_commit_payloads())
    admitted = p1v2.preflight(launch=True, authorization=authorization,
                              commit_verifier=verifier)
    assert admitted["status"] == "launch_admitted"
    assert admitted["exit_code"] == 0
    assert admitted["native_execution"] is False
    assert admitted["protocol_commit"] == "a" * 40
    command = shlex.split(admitted["next_command"])
    assert command[:3] == [
        "PYTHONDONTWRITEBYTECODE=1", "python3",
        "scripts/run_capacity_preflight.py"]
    assert command[command.index("--stage") + 1] == "fit"
    assert command[command.index("--device") + 1] == "mps"
    assert command[command.index("--output") + 1] == (
        _config()["launch"]["output_root"])
    assert command[command.index("--authorization") + 1] == str(authorization)
    assert command[command.index("--protocol-commit") + 1] == "a" * 40
    assert "--receipt" not in command
    output = Path(_config()["launch"]["output_root"])
    reports = [Path(command[index + 1]) for index, token in enumerate(command)
               if token == "--report"]
    assert all(output not in path.parents for path in reports)


@pytest.mark.parametrize("override, match", [
    ({"config_sha256": "f" * 64}, "exact config bytes"),
    ({"prepared_manifest_sha256": "f" * 64}, "exact prepared fixture"),
    ({"protocol_commit": "abc"}, "full 40-hex protocol commit"),
    ({"native_execution_allowed": False}, "native_execution_allowed"),
    ({"single_attempt": False}, "exactly one attempt"),
    ({"authorized_by": " "}, "authorized_by"),
    ({"max_optimizer_steps": 121}, "derived optimizer steps"),
    ({"max_generation_requests": 417}, "derived generation count"),
    ({"restart_allowance": 0}, "restart allowance"),
    ({"runner_sha256": "f" * 64}, "exact runner bytes"),
    ({"runner_path": "scripts/other.py"}, "runner path"),
])
def test_preflight_launch_refuses_each_unbound_field(tmp_path, override, match):
    authorization = _authorization(tmp_path, **override)
    verifier = _commit_verifier(_real_commit_payloads())
    with pytest.raises(ProtocolError, match=match):
        p1v2.preflight(launch=True, authorization=authorization,
                       commit_verifier=verifier)


def test_preflight_launch_refuses_a_stale_or_absent_commit(tmp_path):
    authorization = _authorization(tmp_path)
    with pytest.raises(ProtocolError, match="absent from protocol commit"):
        p1v2.preflight(launch=True, authorization=authorization,
                       commit_verifier=lambda commit, path: None)
    payloads = _real_commit_payloads()
    payloads["manifest.json"] = b"{}"
    with pytest.raises(ProtocolError, match="bytes at"):
        p1v2.preflight(launch=True, authorization=authorization,
                       commit_verifier=_commit_verifier(payloads))


def test_preflight_launch_refuses_tampered_native_implementation(tmp_path):
    authorization = _authorization(tmp_path)
    payloads = _real_commit_payloads()
    payloads["src/replica_lab/dc_p1_native_runner.py"] = b"# tampered"
    with pytest.raises(ProtocolError, match="bytes at"):
        p1v2.preflight(launch=True, authorization=authorization,
                       commit_verifier=_commit_verifier(payloads))


def test_preflight_never_creates_authorization_evidence():
    assert not p1v2.AUTHORIZATION_PATH.exists()
    config = _config()
    assert config["launch"]["requires_authorization_receipt"] is True
    assert config["launch"]["authorization_receipt_present_at_declaration"] is False
    assert config["classification"]["native_execution_allowed"] is False
    assert config["launch"]["native_execution"] is False


def test_review_receipt_path_must_be_fresh(tmp_path):
    from dc_protocol_common_20261027 import require_fresh_path

    path = tmp_path / "receipt.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolError, match="not fresh"):
        require_fresh_path(path, "preflight_receipt")
