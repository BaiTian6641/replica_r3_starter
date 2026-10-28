"""DC-P1 v2 staged admission, receipts, label vault and stage scoring.

The fakes here are small and explicit: synthetic cases, synthetic labels and a
synthetic verdict function.  They exercise orchestration and fail-closed access
only; they load no model, no tokenizer and no optimizer, and they claim no
learning.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.dc_staged_evaluation import (  # noqa: E402
    STAGE_DEVELOPMENT, STAGE_FINAL, STAGE_FIT, STAGE_ORDER,
    AccessSpy, LabelVault, ReceiptExpectation, StageError, StagedAdmission,
    build_stage_receipt, require_prior_report, score_development_stage,
    score_final_stage, score_fit_stage, verify_stage_receipt,
)

BINDING = {
    "decode_policy": {
        "version": "test.decode.v1",
        "skip_special_tokens": True,
        "stop_tokens": [1],
    },
}
BINDING_SHA = "a" * 64
CONFIG_SHA = "b" * 64
MANIFEST_SHA = "c" * 64
COMMIT = "d" * 40
DEV_ROLES = ("post_correction_current", "new_instance_current",
             "historical_state", "abstention", "retention")
ARMS = ("treatment", "matched_trained", "no_update", "reset_P")
GATES = {
    "fit_exact_min": 0.5,
    "fit_nonempty_min": 0.95,
    "post_correction_accuracy_min": 0.5,
    "new_instance_accuracy_min": 0.5,
    "historical_accuracy_min": 0.6,
    "abstention_accuracy_min": 0.9,
    "abstention_false_recall_max": 0.1,
    "retention_accuracy_min": 0.7,
    "retention_decline_max": 0.02,
    "control_primary_accuracy_max": 0.25,
    "post_correction_min_gain": 0.1,
    "new_instance_min_gain": 0.1,
    "paired_ci_low_min": 0.0,
}
AUDIT = {
    "format": "replica.dc_response_objective_audit.v2.collection",
    "claim": "global_ce_rescale",
    "expected_claim": "global_ce_rescale",
    "rescaled_only": True,
    "added_term_changes_learning_signal": False,
    "rows": 4,
}


def _decode(ids):
    return "".join(chr(item) for item in ids)


def _encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def _prompt_sha(prompt: str) -> str:
    return __import__("hashlib").sha256(
        json.dumps({"prompt": prompt}, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()


def _cases(stage: str, groups: int) -> list[dict]:
    roles = ("fit_primary", "fit_correction",
             "fit_abstention") if stage == STAGE_FIT else DEV_ROLES
    split = "dev" if stage == STAGE_DEVELOPMENT else stage
    cases = []
    for index in range(groups):
        group = f"g{index}"
        for role in roles:
            prompt = f"{stage} {group} {role}?"
            cases.append({
                "case_id": f"{group}_{role}",
                "split_role": split,
                "role": role,
                "source_group": group,
                "prompt": prompt,
                "prompt_sha256": _prompt_sha(prompt),
                "inference_inputs": ["prompt"],
                "query_epoch": 2,
                "resolution_mode": ("historical_explicit_event"
                                    if role == "historical_state"
                                    else "stateless" if role == "retention"
                                    else "current_at_query_epoch"),
                "historical_event_position": (0
                                              if role == "historical_state"
                                              else None),
                "source_context_present": False,
                "source_kv_loaded": False,
            })
    return cases


def _labels(cases: list[dict], *, stage: str) -> list[dict]:
    labels = []
    split = "dev" if stage == STAGE_DEVELOPMENT else stage
    for case in cases:
        role = case["role"]
        if role == "abstention" or role == "fit_abstention":
            target = "UNKNOWN"
        elif role == "retention":
            target = "drow"
        elif role == "historical_state":
            target = "oldvalue"
        elif role in ("post_correction_current", "fit_correction"):
            target = "newvalue"
        else:
            target = "newvalue"
        labels.append({
            "case_id": case["case_id"],
            "split_role": split,
            "role": role,
            "source_group": case["source_group"],
            "target": target,
            "query_epoch": 2,
            "resolution_mode": case["resolution_mode"],
            "resolved_event_position": (
                0 if role == "historical_state"
                else 2 if role in ("new_instance_current", "abstention")
                else None if role == "retention" else 1),
        })
    return labels


def _prediction(case: dict, *, stage: str, arm: str, checkpoint_id: str,
                text: str) -> dict:
    return {
        "format": "replica.dc_p1_native_response_v2.prediction.v1",
        "protocol_id": "dc_p1_native_response_v2",
        "stage": stage,
        "arm_id": arm,
        "checkpoint_id": checkpoint_id,
        "case_id": case["case_id"],
        "prompt": case["prompt"],
        "prompt_sha256": case["prompt_sha256"],
        "generated_ids": _encode(text),
        "generated_text": text,
        "empty_generation": text == "",
        "tokenizer_binding_sha256": BINDING_SHA,
        "decode_policy": BINDING["decode_policy"],
        "source_context_present": False,
        "source_kv_loaded": False,
        "source_prefill_tokens": 0,
        "source_kv_reused": False,
        "prompt_source_spans": 0,
        "routing_metadata": {"mode": "fixed", "expert_id": 0},
        "stopped": True,
        "stop_reason": "eos",
        "query_epoch": case["query_epoch"],
        "resolution_mode": case["resolution_mode"],
        "resolved_event_position": (
            case.get("event_position", 1) if case["role"].startswith("fit_")
            else None if case["role"] == "retention"
            else case["historical_event_position"]
            if case["resolution_mode"] == "historical_explicit_event"
            else 2 if case["role"] in ("new_instance_current", "abstention")
            else 1),
    }


def _arm_specs(stage: str) -> list[dict]:
    if stage == STAGE_FINAL:
        return [{"arm_id": "treatment",
                 "checkpoint_id": "dcp1v2:arm:treatment:final12",
                 "answer_span_ce_weight": 0.5}]
    return [{"arm_id": arm,
             "checkpoint_id": f"dcp1v2:arm:{arm}:dev36",
             "answer_span_ce_weight": 0.0 if arm == "matched_trained" else 0.5}
            for arm in ARMS]


def _predictions(cases: list[dict], labels: list[dict], *,
                 stage: str, controls_perfect: bool = False,
                 treatment_wrong: bool = False) -> list[dict]:
    target_by_id = {row["case_id"]: row["target"] for row in labels}
    rows = []
    for case in cases:
        target = target_by_id[case["case_id"]]
        primary = case["role"] in ("post_correction_current",
                                   "new_instance_current")
        for spec in _arm_specs(stage):
            arm = spec["arm_id"]
            if arm == "treatment":
                text = "wrong" if (treatment_wrong and primary) else target
            elif controls_perfect and primary:
                text = target
            elif primary:
                text = "wrong"
            else:
                text = target
            rows.append(_prediction(case, stage=stage, arm=arm,
                                    checkpoint_id=spec["checkpoint_id"],
                                    text=text))
    return rows


def _expectation(*, fit_cases: int, development_cases: int,
                 final_cases: int) -> ReceiptExpectation:
    return ReceiptExpectation(
        protocol_id="dc_p1_native_response_v2",
        config_sha256=CONFIG_SHA,
        manifest_sha256=MANIFEST_SHA,
        protocol_commit=COMMIT,
        tokenizer_binding_sha256=BINDING_SHA,
        decode_policy_sha256=BINDING_SHA,
        arms=ARMS,
        cases_expected={STAGE_FIT: fit_cases,
                        STAGE_DEVELOPMENT: development_cases,
                        STAGE_FINAL: final_cases},
    )


def _verify(prediction, binding, decode):
    return prediction["generated_text"]


def _fit_report(*, treatment_wrong: bool = False, audit=AUDIT) -> tuple[
        dict, ReceiptExpectation]:
    cases = _cases(STAGE_FIT, 2)
    labels = _labels(cases, stage=STAGE_FIT)
    predictions = _predictions(cases, labels, stage=STAGE_FIT)
    if treatment_wrong:
        for row in predictions:
            if row["arm_id"] == "treatment":
                row["generated_text"] = "wrong"
                row["generated_ids"] = _encode("wrong")
    expectation = _expectation(fit_cases=len(cases), development_cases=5,
                              final_cases=3)
    report = score_fit_stage(
        protocol_id="dc_p1_native_response_v2", cases=cases, labels=labels,
        predictions=predictions, arm_specs=_arm_specs(STAGE_FIT),
        binding=BINDING, binding_sha256=BINDING_SHA, decode_fn=_decode,
        verify_decoded=_verify, gates=GATES, objective_audit=audit)
    return report, expectation


def _dev_report(*, controls_perfect: bool = False) -> tuple[
        dict, ReceiptExpectation]:
    cases = _cases(STAGE_DEVELOPMENT, 3)
    labels = _labels(cases, stage=STAGE_DEVELOPMENT)
    predictions = _predictions(cases, labels, stage=STAGE_DEVELOPMENT,
                               controls_perfect=controls_perfect)
    expectation = _expectation(fit_cases=6, development_cases=len(cases),
                              final_cases=3)
    report = score_development_stage(
        protocol_id="dc_p1_native_response_v2", cases=cases, labels=labels,
        predictions=predictions, arm_specs=_arm_specs(STAGE_DEVELOPMENT),
        binding=BINDING, binding_sha256=BINDING_SHA, decode_fn=_decode,
        verify_decoded=_verify, gates=GATES, seed=20261027, resamples=50,
        objective_audit=AUDIT)
    return report, expectation


def _final_report() -> tuple[dict, ReceiptExpectation]:
    fit_cases = _cases(STAGE_FIT, 1)
    for case in fit_cases:
        case["split_role"] = "final_fit"
    final_cases = _cases(STAGE_FINAL, 1)
    for case in final_cases:
        case["split_role"] = "sealed_final"
    cases = fit_cases + final_cases
    labels = (_labels(fit_cases, stage="final_fit")
              + _labels(final_cases, stage="sealed_final"))
    predictions = _predictions(cases, labels, stage=STAGE_FINAL)
    expectation = _expectation(fit_cases=6, development_cases=15,
                              final_cases=len(cases))
    report = score_final_stage(
        protocol_id="dc_p1_native_response_v2", cases=cases, labels=labels,
        predictions=predictions, arm_specs=_arm_specs(STAGE_FINAL),
        binding=BINDING, binding_sha256=BINDING_SHA, decode_fn=_decode,
        verify_decoded=_verify, gates=GATES, objective_audit=AUDIT)
    return report, expectation


# ---------------------------------------------------------------------------
# Label vault and access spy


def test_label_vault_refuses_without_a_permit_and_records_the_attempt():
    spy = AccessSpy()
    vault = LabelVault({STAGE_FIT: [{"case_id": "a", "target": "x"}],
                        STAGE_DEVELOPMENT: [{"case_id": "b",
                                             "target": "y"}]}, spy=spy)
    with pytest.raises(StageError, match="require an admission permit"):
        vault.open(STAGE_DEVELOPMENT, permit=None)
    assert spy.denied and spy.denied[0]["stage"] == STAGE_DEVELOPMENT
    assert vault.stage_ids(STAGE_DEVELOPMENT) == ("b",)
    assert vault.stage_digest(STAGE_DEVELOPMENT)
    with pytest.raises(StageError, match="cannot open"):
        vault.open(STAGE_DEVELOPMENT,
                   permit=__import__(
                       "replica_lab.dc_staged_evaluation",
                       fromlist=["AccessPermit"]).AccessPermit(
                           stage=STAGE_FIT, attempt_id="t",
                           receipt_sha256="e" * 64))
    assert vault.opened == ()


def test_label_vault_requires_unique_ids_and_known_stages():
    with pytest.raises(StageError, match="duplicate label id"):
        LabelVault({STAGE_FIT: [{"case_id": "a"}, {"case_id": "a"}]})
    with pytest.raises(StageError, match="unknown stage"):
        LabelVault({"heldout": []})


# ---------------------------------------------------------------------------
# Fit admission


def test_fit_admission_needs_every_arm_and_the_treatment_gates():
    report, expectation = _fit_report()
    assert report["status"] == "passed"
    assert report["arms"]["treatment"]["answer_span_ce_weight"] == 0.5
    assert report["arms"]["matched_trained"]["answer_span_ce_weight"] == 0.0
    receipt = build_stage_receipt(report=report, expectation=expectation,
                                 attempt_id="fit-01")
    verify_stage_receipt(receipt, stage=STAGE_FIT, expectation=expectation)
    admission = StagedAdmission(expectation)
    assert admission.admit_fit(receipt)["status"] == "passed"
    assert admission.fit_admitted is True
    assert admission.development_admitted is False
    permit = admission.require(STAGE_DEVELOPMENT)
    assert permit.stage == STAGE_DEVELOPMENT
    assert admission.development_admitted is False


def test_failed_treatment_fit_gate_leaves_development_closed():
    report, expectation = _fit_report(treatment_wrong=True)
    assert report["status"] == "failed"
    assert report["gates"]["treatment_fit_exact"]["passed"] is False
    receipt = build_stage_receipt(report=report, expectation=expectation,
                                 attempt_id="fit-02")
    admission = StagedAdmission(expectation)
    with pytest.raises(StageError, match="gates did not pass"):
        admission.admit_fit(receipt)
    assert admission.fit_admitted is False
    with pytest.raises(StageError, match="admitted fit receipt"):
        admission.require(STAGE_DEVELOPMENT)
    vault = LabelVault({STAGE_DEVELOPMENT: [{"case_id": "d"}]},
                       spy=admission.spy)
    with pytest.raises(StageError, match="admission permit"):
        vault.open(STAGE_DEVELOPMENT, permit=None)
    assert admission.spy.denied


def test_absent_or_tampered_fit_receipt_cannot_open_development():
    _, expectation = _fit_report()
    admission = StagedAdmission(expectation)
    with pytest.raises(StageError, match="record must be an object"):
        admission.admit_fit(None)
    broken = build_stage_receipt(report=_fit_report()[0],
                                 expectation=expectation,
                                 attempt_id="fit-03")
    broken = dict(broken)
    broken["status"] = "passed"
    broken["config_sha256"] = "f" * 64
    with pytest.raises(StageError, match="receipt_sha256"):
        admission.admit_fit(broken)
    tampered = build_stage_receipt(report=_fit_report()[0],
                                   expectation=expectation,
                                   attempt_id="fit-04")
    tampered = dict(tampered)
    tampered["gates"] = dict(tampered["gates"],
                             treatment_fit_exact={"passed": True, "value": 1.0})
    with pytest.raises(StageError, match="receipt_sha256"):
        admission.admit_fit(tampered)
    assert admission.fit_admitted is False


def test_receipt_from_another_identity_is_refused():
    report, expectation = _fit_report()
    receipt = build_stage_receipt(report=report, expectation=expectation,
                                 attempt_id="fit-05")
    other = _expectation(fit_cases=report["cases"]["count"],
                         development_cases=5, final_cases=3)
    other = ReceiptExpectation(**{**other.as_dict(), "protocol_commit":
                                  "e" * 40, "arms": ARMS,
                                  "cases_expected": other.cases_expected})
    with pytest.raises(StageError, match="protocol_commit"):
        verify_stage_receipt(receipt, stage=STAGE_FIT, expectation=other)
    with pytest.raises(StageError, match="stage"):
        verify_stage_receipt(receipt, stage=STAGE_DEVELOPMENT,
                             expectation=expectation)


# ---------------------------------------------------------------------------
# Development admission


def test_development_requires_fit_first_and_the_matched_comparison():
    _, expectation = _dev_report()
    dev_report, _ = _dev_report()
    assert dev_report["status"] == "passed"
    assert dev_report["arms"]["matched_trained"]["answer_span_ce_weight"] == 0.0
    dev_receipt = build_stage_receipt(report=dev_report,
                                      expectation=expectation,
                                      attempt_id="dev-01")
    admission = StagedAdmission(expectation)
    with pytest.raises(StageError, match="before fit"):
        admission.admit_development(dev_receipt)
    fit_report, _ = _fit_report()
    fit_receipt = build_stage_receipt(report=fit_report,
                                      expectation=expectation,
                                      attempt_id="fit-10")
    admission.admit_fit(fit_receipt)
    assert admission.admit_development(dev_receipt)["status"] == "passed"
    assert admission.development_admitted is True


def test_controls_that_reproduce_the_gain_block_development():
    report, expectation = _dev_report(controls_perfect=True)
    assert report["status"] == "failed"
    assert report["gates"]["post_correction_current_gain_vs_matched_trained"][
        "passed"] is False
    assert report["gates"][
        "reset_P_post_correction_current_control_ceiling"]["passed"] is False
    receipt = build_stage_receipt(report=report, expectation=expectation,
                                 attempt_id="dev-02")
    admission = StagedAdmission(expectation)
    admission.admit_fit(build_stage_receipt(
        report=_fit_report()[0], expectation=expectation,
        attempt_id="fit-11"))
    with pytest.raises(StageError, match="gates did not pass"):
        admission.admit_development(receipt)
    assert admission.development_admitted is False
    with pytest.raises(StageError, match="development receipt"):
        admission.require(STAGE_FINAL)


def test_development_receipt_without_matched_evidence_is_refused():
    report, expectation = _dev_report()
    stripped = deepcopy(report)
    stripped["arms"].pop("matched_trained")
    with pytest.raises(StageError, match="arms"):
        build_stage_receipt(report=stripped, expectation=expectation,
                            attempt_id="dev-03")


# ---------------------------------------------------------------------------
# Final admission and one-shot behaviour


def test_final_is_one_shot_and_treatment_only():
    _, expectation = _final_report()
    fit_report, _ = _fit_report()
    dev_report, _ = _dev_report()
    admission = StagedAdmission(expectation)
    admission.admit_fit(build_stage_receipt(
        report=fit_report, expectation=expectation, attempt_id="fit-20"))
    admission.admit_development(build_stage_receipt(
        report=dev_report, expectation=expectation, attempt_id="dev-20"))
    permit = admission.require(STAGE_FINAL)
    assert permit.stage == STAGE_FINAL
    with pytest.raises(StageError, match="one-shot"):
        admission.require(STAGE_FINAL)
    final_report, _ = _final_report()
    receipt = build_stage_receipt(report=final_report,
                                  expectation=expectation,
                                  attempt_id="final-20")
    assert admission.admit_final(receipt)["stage"] == STAGE_FINAL
    assert admission.final_consumed is True
    with pytest.raises(StageError, match="one-shot"):
        admission.admit_final(receipt)


def test_final_receipt_with_a_control_arm_is_refused():
    _, expectation = _final_report()
    final_report, _ = _final_report()
    wrong = deepcopy(final_report)
    wrong["arms"]["matched_trained"] = dict(wrong["arms"]["treatment"])
    with pytest.raises(StageError, match="expected \\['treatment'\\]"):
        build_stage_receipt(report=wrong, expectation=expectation,
                            attempt_id="final-21")


def test_final_attempt_without_development_stays_closed():
    _, expectation = _final_report()
    fit_report, _ = _fit_report()
    admission = StagedAdmission(expectation)
    admission.admit_fit(build_stage_receipt(
        report=fit_report, expectation=expectation, attempt_id="fit-22"))
    with pytest.raises(StageError, match="development receipt"):
        admission.require(STAGE_FINAL)
    final_report, _ = _final_report()
    receipt = build_stage_receipt(report=final_report,
                                  expectation=expectation,
                                  attempt_id="final-22")
    admission.admit_development(build_stage_receipt(
        report=_dev_report()[0], expectation=expectation,
        attempt_id="dev-22"))
    with pytest.raises(StageError, match="without a started final attempt"):
        admission.admit_final(receipt)


def test_final_receipt_must_be_reported_as_confirmation_only():
    report, expectation = _final_report()
    assert report["confirmation_only"] is True
    assert report["final_comparison"] is None
    assert report["gates"]["final_confirmation_only"]["passed"] is True
    broken = dict(report, final_comparison={"a": 1}, confirmation_only=False)
    receipt = build_stage_receipt(report=broken, expectation=expectation,
                                  attempt_id="final-23")
    assert receipt["stage"] == STAGE_FINAL


# ---------------------------------------------------------------------------
# Scoring fail-closed behaviour


def test_scoring_rejects_source_retained_and_answer_bearing_predictions():
    cases = _cases(STAGE_FIT, 1)
    labels = _labels(cases, stage=STAGE_FIT)
    good = _predictions(cases, labels, stage=STAGE_FIT)
    spec = _arm_specs(STAGE_FIT)

    retained = deepcopy(good)
    retained[0]["source_context_present"] = True
    with pytest.raises(StageError, match="source_free_evaluation"):
        score_fit_stage(
            protocol_id="dc_p1_native_response_v2", cases=cases,
            labels=labels, predictions=retained, arm_specs=spec,
            binding=BINDING, binding_sha256=BINDING_SHA, decode_fn=_decode,
            verify_decoded=_verify, gates=GATES, objective_audit=AUDIT)

    leaked = deepcopy(good)
    leaked[0]["answer_index"] = 3
    with pytest.raises(StageError, match="answer-bearing field"):
        score_fit_stage(
            protocol_id="dc_p1_native_response_v2", cases=cases,
            labels=labels, predictions=leaked, arm_specs=spec,
            binding=BINDING, binding_sha256=BINDING_SHA, decode_fn=_decode,
            verify_decoded=_verify, gates=GATES, objective_audit=AUDIT)

    routed = deepcopy(good)
    routed[0]["routing_metadata"]["target"] = "leak"
    with pytest.raises(StageError, match="answer-bearing routing field"):
        score_fit_stage(
            protocol_id="dc_p1_native_response_v2", cases=cases,
            labels=labels, predictions=routed, arm_specs=spec,
            binding=BINDING, binding_sha256=BINDING_SHA, decode_fn=_decode,
            verify_decoded=_verify, gates=GATES, objective_audit=AUDIT)


def test_scoring_requires_a_tokenizer_backed_verifier_and_full_coverage():
    cases = _cases(STAGE_FIT, 1)
    labels = _labels(cases, stage=STAGE_FIT)
    predictions = _predictions(cases, labels, stage=STAGE_FIT)
    with pytest.raises(StageError, match="tokenizer-backed"):
        score_fit_stage(
            protocol_id="dc_p1_native_response_v2", cases=cases,
            labels=labels, predictions=predictions,
            arm_specs=_arm_specs(STAGE_FIT), binding=BINDING,
            binding_sha256=BINDING_SHA, decode_fn=_decode,
            verify_decoded=None, gates=GATES, objective_audit=AUDIT)
    missing = predictions[:-1]
    with pytest.raises(StageError, match="missing predictions"):
        score_fit_stage(
            protocol_id="dc_p1_native_response_v2", cases=cases,
            labels=labels, predictions=missing,
            arm_specs=_arm_specs(STAGE_FIT), binding=BINDING,
            binding_sha256=BINDING_SHA, decode_fn=_decode,
            verify_decoded=_verify, gates=GATES, objective_audit=AUDIT)


def test_checkpoint_identity_and_stage_must_match():
    cases = _cases(STAGE_FIT, 1)
    labels = _labels(cases, stage=STAGE_FIT)
    predictions = _predictions(cases, labels, stage=STAGE_FIT)
    predictions[0]["checkpoint_id"] = "dcp1v2:arm:no_update:dev00"
    with pytest.raises(StageError, match="checkpoint"):
        score_fit_stage(
            protocol_id="dc_p1_native_response_v2", cases=cases,
            labels=labels, predictions=predictions,
            arm_specs=_arm_specs(STAGE_FIT), binding=BINDING,
            binding_sha256=BINDING_SHA, decode_fn=_decode,
            verify_decoded=_verify, gates=GATES, objective_audit=AUDIT)
    predictions = _predictions(cases, labels, stage=STAGE_FIT)
    predictions[0]["stage"] = STAGE_DEVELOPMENT
    with pytest.raises(StageError, match="prediction stage"):
        score_fit_stage(
            protocol_id="dc_p1_native_response_v2", cases=cases,
            labels=labels, predictions=predictions,
            arm_specs=_arm_specs(STAGE_FIT), binding=BINDING,
            binding_sha256=BINDING_SHA, decode_fn=_decode,
            verify_decoded=_verify, gates=GATES, objective_audit=AUDIT)


def test_a_missing_mask_audit_fails_fit_scoring():
    report, _ = _fit_report(audit=None)
    assert report["status"] == "failed"
    assert report["gates"]["mask_audit_present"]["passed"] is False


def test_a_disagreeing_audit_claim_is_refused_by_the_gate():
    audit = dict(AUDIT, claim="span_specific_weighting")
    report, _ = _fit_report(audit=audit)
    assert report["gates"]["objective_claim_consistent"]["passed"] is False


def test_prior_report_helper_refuses_failed_or_unsealed_reports():
    report, _ = _fit_report()
    report = dict(report, status="passed", attempt_id="fit-30")
    require_prior_report(report,
                         required_format="replica.dc_stage_score_report.v2",
                         protocol_id="dc_p1_native_response_v2",
                         stage=STAGE_FIT)
    with pytest.raises(StageError, match="status"):
        require_prior_report(report,
                             required_format=(
                                 "replica.dc_stage_score_report.v2"),
                             protocol_id="dc_p1_native_response_v2",
                             statuses=("failed",))
    with pytest.raises(StageError, match="attempt_id"):
        require_prior_report({key: value for key, value in report.items()
                              if key != "attempt_id"},
                             required_format=(
                                 "replica.dc_stage_score_report.v2"),
                             protocol_id="dc_p1_native_response_v2")


def test_admission_state_is_serialisable_and_spy_is_clean_after_success():
    _, expectation = _final_report()
    admission = StagedAdmission(expectation)
    admission.admit_fit(build_stage_receipt(
        report=_fit_report()[0], expectation=expectation,
        attempt_id="fit-40"))
    admission.admit_development(build_stage_receipt(
        report=_dev_report()[0], expectation=expectation,
        attempt_id="dev-40"))
    admission.require(STAGE_FINAL)
    admission.admit_final(build_stage_receipt(
        report=_final_report()[0], expectation=expectation,
        attempt_id="final-40"))
    state = admission.state()
    json.dumps(state)
    assert state["format"] == "replica.dc_stage_admission.v2"
    assert list(state["stages"]) == list(STAGE_ORDER)
    assert all(state["stages"][stage]["admitted"] for stage in STAGE_ORDER)
    assert state["final_consumed"] is True
    assert state["access"]["denied"] == 0
    admission.spy.assert_clean()


def test_lazy_development_loader_stays_closed_for_absent_failed_tampered_fit():
    _, expectation = _fit_report()
    calls = []

    def development_labels():
        calls.append("opened")
        return [{"case_id": "d", "target": "x"}]

    admission = StagedAdmission(expectation)
    vault = LabelVault({STAGE_DEVELOPMENT: development_labels},
                       spy=admission.spy)
    with pytest.raises(StageError, match="admission permit"):
        vault.open(STAGE_DEVELOPMENT, permit=None)
    with pytest.raises(StageError, match="admission permit"):
        vault.stage_digest(STAGE_DEVELOPMENT)
    assert calls == []

    failed = build_stage_receipt(
        report=_fit_report(treatment_wrong=True)[0], expectation=expectation,
        attempt_id="fit-failed")
    with pytest.raises(StageError, match="gates did not pass"):
        admission.admit_fit(failed)
    assert calls == []

    tampered = build_stage_receipt(
        report=_fit_report()[0], expectation=expectation,
        attempt_id="fit-tampered")
    tampered = dict(tampered, config_sha256="f" * 64)
    with pytest.raises(StageError, match="receipt_sha256"):
        admission.admit_fit(tampered)
    assert calls == []

    admitted = build_stage_receipt(
        report=_fit_report()[0], expectation=expectation,
        attempt_id="fit-admitted")
    admission.admit_fit(admitted)
    permit = admission.require(STAGE_DEVELOPMENT)
    assert vault.open(STAGE_DEVELOPMENT, permit=permit)[0]["target"] == "x"
    assert calls == ["opened"]


def test_lazy_final_loader_requires_development_admission_before_hash_or_read():
    _, expectation = _final_report()
    calls = []

    def final_labels():
        calls.append("opened")
        return [{"case_id": "f", "target": "x"}]

    admission = StagedAdmission(expectation)
    admission.admit_fit(build_stage_receipt(
        report=_fit_report()[0], expectation=expectation,
        attempt_id="fit-final-gate"))
    vault = LabelVault({STAGE_FINAL: final_labels}, spy=admission.spy)
    with pytest.raises(StageError, match="development receipt"):
        admission.require(STAGE_FINAL)
    with pytest.raises(StageError, match="admission permit"):
        vault.open(STAGE_FINAL, permit=None)
    with pytest.raises(StageError, match="admission permit"):
        vault.stage_digest(STAGE_FINAL)
    assert calls == []

    admission.admit_development(build_stage_receipt(
        report=_dev_report()[0], expectation=expectation,
        attempt_id="dev-final-gate"))
    permit = admission.require(STAGE_FINAL)
    assert admission.final_started is True
    assert vault.open(STAGE_FINAL, permit=permit)[0]["target"] == "x"
    assert calls == ["opened"]


def test_truncated_exact_text_does_not_receive_completion_credit():
    cases = _cases(STAGE_FIT, 1)
    labels = _labels(cases, stage=STAGE_FIT)
    predictions = _predictions(cases, labels, stage=STAGE_FIT)
    for row in predictions:
        row["stop_reason"] = "max_new_tokens"
    report = score_fit_stage(
        protocol_id="dc_p1_native_response_v2", cases=cases, labels=labels,
        predictions=predictions, arm_specs=_arm_specs(STAGE_FIT),
        binding=BINDING, binding_sha256=BINDING_SHA, decode_fn=_decode,
        verify_decoded=_verify, gates=GATES, objective_audit=AUDIT)
    assert report["status"] == "failed"
    assert report["arms"]["treatment"]["fit"]["exact_accuracy"] == 0.0


def test_prediction_event_resolution_is_checked_against_the_label():
    cases = _cases(STAGE_FIT, 1)
    labels = _labels(cases, stage=STAGE_FIT)
    predictions = _predictions(cases, labels, stage=STAGE_FIT)
    predictions[0]["resolved_event_position"] = 99
    with pytest.raises(StageError, match="temporal_binding"):
        score_fit_stage(
            protocol_id="dc_p1_native_response_v2", cases=cases, labels=labels,
            predictions=predictions, arm_specs=_arm_specs(STAGE_FIT),
            binding=BINDING, binding_sha256=BINDING_SHA, decode_fn=_decode,
            verify_decoded=_verify, gates=GATES, objective_audit=AUDIT)
