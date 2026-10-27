"""CPU-only contracts for measured v2 readouts, audits and step-1 protocol."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from replica_lab import vision_forward_accounting as fwd
from replica_lab import vision_learnability_curve as curve_v1
from replica_lab import vision_learnability_curve_v2 as curve_v2
from replica_lab import vision_learnability_diagnostic as learn
from conftest import require_local_artifacts


ROOT = Path(__file__).resolve().parents[1]
CURVE_V2_CONFIG = ROOT / "configs/vision_learnability_curve_v2.json"
STEP1_CONFIG = ROOT / "configs/vision_learnability_step1_reproduction_v1.json"
V1_CONFIG = ROOT / "configs/vision_learnability_curve_v1.json"

V1_PRESERVED_HASHES = {
    "configs/vision_learnability_curve_v1.json":
        "c97902da2862c8141ab22d98cbb6ad3c01739916854da8e7fd1a15639ec81c40",
    "src/replica_lab/vision_learnability_curve.py":
        "124676cead4b14c4c055dc4befa40f19b3712a151066262826262bb2aa367751",
    "src/replica_lab/vision_learnability_diagnostic.py":
        "459ed73379f5b45d16f75f9d471838d71c67a3b0a431a329851e8b69342bf6f1",
}


def _sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Preserved v1 files, v1 config and v2 declarations
# ---------------------------------------------------------------------------


def test_v1_files_and_config_remain_byte_identical():
    for relative, digest in V1_PRESERVED_HASHES.items():
        assert _sha(relative) == digest, relative
    v1 = curve_v1.load_config(V1_CONFIG)
    assert v1["mode"] == "curve"
    assert v1["limits"]["max_scoring_forwards"] == 1470
    assert v1["expectations"]["total_scoring_forwards"] == 1470
    assert v1["implementation_pins"]["new_files"][
        "src/replica_lab/vision_learnability_curve.py"] == (
            V1_PRESERVED_HASHES[
                "src/replica_lab/vision_learnability_curve.py"])


def test_v2_curve_config_declares_measured_units_and_preserves_verdict():
    config = curve_v2.load_config(CURVE_V2_CONFIG)
    assert config["mode"] == curve_v2.CURVE_MODE
    assert config["accounting"]["format"] == fwd.BUDGET_FORMAT
    assert config["accounting"]["per_readout"][
        "decoder_forward_calls"] == 2400
    assert config["accounting"]["cumulative"][
        "decoder_forward_calls"] == 12000
    assert config["accounting"]["per_readout"][
        "prepare_sense_passes"] == 294
    assert config["expectations"]["historical_total_scoring_decoder_passes"] == 7695
    assert config["expectations"]["measured_by_old_run"] is False
    assert config["authorization"]["native_execution_authorized"] is False
    assert config["authorization"]["protocol_commit"] is None
    assert config["claims"]["verdict"] == curve_v2.V1_VERDICT
    assert config["claims"]["verdict_preserved"] is True
    assert config["claims"]["dev_authorized"] is False
    assert config["claims"]["heldout_authorized"] is False
    for relative in curve_v2.NEW_FILE_PINS:
        assert config["implementation_pins"]["new_files"][relative] == _sha(
            relative)


@pytest.mark.parametrize("mutation, message", [
    (lambda config: config["accounting"]["per_readout"].update(
        {"decoder_forward_calls": 294}), "three prepare"),
    (lambda config: config["accounting"]["units"].update(
        {"decoder_forward_calls": "logical_scoring_operation"}),
     "Budget unit differs"),
    (lambda config: config["accounting"]["cumulative"].pop(
        "decoder_forward_calls"), "cumulative cap"),
    (lambda config: config["authorization"].update(
        {"native_execution_authorized": True}), "predeclared protocol"),
    (lambda config: config["authorization"].update(
        {"protocol_commit": "a" * 40}), "committed identity"),
    (lambda config: config["expectations"].update(
        {"measured_by_old_run": True}), "reconstruction"),
    (lambda config: config["implementation_pins"]["new_files"].pop(
        "src/replica_lab/vision_forward_accounting.py"), "New-file"),
])
def test_v2_curve_config_rejects_accounting_or_authorization_drift(
        tmp_path, mutation, message):
    config = json.loads(CURVE_V2_CONFIG.read_text())
    mutation(config)
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(config))
    with pytest.raises(learn.LearnabilityDiagnosticError, match=message):
        curve_v2.load_config(path)


def test_step1_protocol_is_predeclared_and_fail_closed():
    config = curve_v2.load_config(STEP1_CONFIG)
    assert config["mode"] == curve_v2.STEP1_MODE
    assert config["experiment"]["init_seed"] == 29
    assert config["experiment"]["lr"] == 0.0003
    assert config["experiment"]["rows_per_update"] == 4
    assert config["experiment"]["backward_calls"] == 4
    assert config["experiment"]["optimizer_steps"] == 1
    assert config["experiment"]["probes"] == 0
    assert config["experiment"]["forbidden_prefixes"] == [2, 4, 7]
    assert config["readouts"]["labels"] == [
        "prefix_0", "prefix_0_null", "prefix_1"]
    assert config["expected_measured_work"] == curve_v2.STEP1_EXPECTED_WORK
    assert config["expected_measured_work"]["decoder_forward_calls"] == 5644
    assert config["expected_measured_work"]["generated_tokens"] == 4166
    assert config["expected_measured_work"]["prepare_sense_passes"] == 882
    assert config["expected_measured_work"]["explicit_logits_forwards"] == 588
    assert config["expected_measured_work"]["update_forwards"] == 8
    assert config["historical_anchors_as_hypotheses"] == (
        curve_v2.STEP1_ANCHORS)
    assert config["nondeterminism_stop"]["on_mismatch"] == (
        curve_v2.STEP1_STATUS_NONDETERMINISTIC)
    assert config["authorization"]["native_execution_authorized"] is False
    assert config["claims"]["curve_v1_verdict"] == curve_v2.V1_VERDICT
    for relative in curve_v2.NEW_FILE_PINS:
        assert config["implementation_pins"]["new_files"][relative] == _sha(
            relative)


@pytest.mark.parametrize("mutation, message", [
    (lambda config: config["experiment"].update({"rows_per_update": 8}),
     "rows_per_update"),
    (lambda config: config["experiment"].update({"probes": 1}), "probes"),
    (lambda config: config["expected_measured_work"].update(
        {"decoder_forward_calls": 1470}), "measured-work"),
    (lambda config: config["authorization"].update(
        {"native_execution_authorized": True}), "predeclared protocol"),
    (lambda config: config["historical_anchors_as_hypotheses"].update(
        {"repeat_abs_tolerance": 0.1}), "historical anchors"),
    (lambda config: config["claims"].update(
        {"curve_verdict_preserved": False}), "claims"),
])
def test_step1_config_rejects_scope_or_anchor_drift(tmp_path, mutation,
                                                     message):
    config = json.loads(STEP1_CONFIG.read_text())
    mutation(config)
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(config))
    with pytest.raises(learn.LearnabilityDiagnosticError, match=message):
        curve_v2.load_config(path)


# ---------------------------------------------------------------------------
# Measured readout validation
# ---------------------------------------------------------------------------


def _measured_counters(*, generated=1362, prepares=294, logits=196,
                       prefill=98, logical_cells=98, encoder_misses=72):
    decode = generated - prefill
    visual_forward = 200
    counters = {key: 0 for key in fwd.COUNTER_KEYS}
    counters.update({
        "decoder_forward_calls": prepares + logits + prefill + decode,
        "prepare_sense_passes": prepares,
        "explicit_logits_forwards": logits,
        "generation_prefill_calls": prefill,
        "generation_decode_calls": decode,
        "generated_tokens": generated,
        "processed_text_positions": counters.get("processed_text_positions", 0)
        + 100000,
        "processed_visual_positions": 196 * visual_forward,
        "visual_forward_calls": visual_forward,
        "projector_passes": visual_forward,
        "visual_observation_calls": 72,
        "visual_encoder_calls": encoder_misses,
        "visual_encoder_cache_hits": 72 - encoder_misses,
        "visual_encoder_cache_misses": encoder_misses,
        "logical_scoring_cells": logical_cells,
        "logical_scoring_operations": 3 * logical_cells,
    })
    return counters, {
        "format": fwd.ACCOUNTING_FORMAT,
        "version": fwd.VERSION,
        "counters": counters,
        "units": {key: fwd.UNIT_BY_COUNTER[key] for key in fwd.COUNTER_KEYS},
        "phases": {},
        "logical": {
            "scoring_cells": logical_cells,
            "scoring_operations": 3 * logical_cells,
            "physical_decoder_passes": counters["decoder_forward_calls"],
        },
    }


def _records(prefix, generated):
    records = []
    remaining = generated - 98
    assert remaining >= 0
    lengths = [1] * 98
    kinds = ["answer"] * 52 + ["caption"] * 46
    caption_indexes = [index for index, kind in enumerate(kinds)
                       if kind == "caption"]
    answer_indexes = [index for index, kind in enumerate(kinds)
                      if kind == "answer"]
    filler = 0
    while remaining:
        index = caption_indexes[filler % len(caption_indexes)]
        if lengths[index] < 32:
            lengths[index] += 1
            remaining -= 1
        filler += 1
        if filler > 100000:
            raise AssertionError("cannot place synthetic generation lengths")
    assert all(lengths[index] <= 8 for index in answer_indexes)
    for index, length in enumerate(lengths):
        ids = [10 + (index % 5)] * length
        records.append({
            "prefix": prefix,
            "case_id": f"case-{index}",
            "control": "correct" if index % 2 else "noimage",
            "kind": kinds[index],
            "available": True,
            "generated_ids": ids,
            "generated_text": "x",
            "prefill": {"argmax_token": ids[0], "gold_token": ids[0]},
            "prefill_argmax_matches_first_token": True,
        })
    assert sum(len(record["generated_ids"]) for record in records) == generated
    return records


def _readout(prefix=0, generated=1362, *, gates=None, records=None):
    counters, measured = _measured_counters(generated=generated)
    return {
        "format": curve_v2.READOUT_FORMAT,
        "prefix": prefix,
        "counts": {
            "records": 104,
            "available": 98,
            "placeholders": 6,
            "generations": 98,
            "prefill_forwards": 98,
            "teacher_forced_cells": 98,
            "generated_tokens": generated,
            "logical_scoring_cells": 98,
            "logical_scoring_operations": 294,
        },
        "measured": measured,
        "generation_summary": {
            "generated_tokens": generated,
            "lengths_by_kind": {},
            "generation_cells": 98,
            "prefill_calls": 98,
            "decode_calls": generated - 98,
            "eos_only": 0,
            "whitespace_only": 0,
            "generation_digest": "a" * 64,
        },
        "gates": gates,
    }


def test_measured_readout_accepts_physical_and_logical_units():
    readout = _readout()
    report = curve_v2.validate_measured_readout(
        readout, records=_records(0, 1362))
    assert report["ok"] is True, report["violations"]
    assert report["decoder_forward_calls"] == 1852
    assert report["logical_scoring_operations"] == 294


def test_measured_readout_rejects_missing_counters_and_bad_arithmetic():
    readout = _readout()
    del readout["measured"]["counters"]["generation_decode_calls"]
    report = curve_v2.validate_measured_readout(readout)
    assert report["ok"] is False
    assert "missing" in " ".join(report["violations"])
    readout = _readout()
    readout["measured"]["counters"]["decoder_forward_calls"] += 1
    report = curve_v2.validate_measured_readout(readout)
    assert "decoder_phase_sum" in " ".join(report["violations"])
    readout = _readout()
    readout["counts"]["prefill_forwards"] = 97
    report = curve_v2.validate_measured_readout(readout)
    assert any("counts_prefill" in item for item in report["violations"])


def test_measured_readout_rejects_generation_length_and_record_sum():
    readout = _readout(generated=1362)
    report = curve_v2.validate_measured_readout(
        readout, records=_records(0, 1369))
    assert any("record_token_sum" in item for item in report["violations"])
    records = _records(0, 1362)
    records[0]["generated_ids"] = [1] * 200
    report = curve_v2.validate_measured_readout(
        readout, records=records,
        max_new_tokens_by_kind={"answer": 8, "caption": 32})
    assert any("generation_length" in item for item in report["violations"])


def test_endpoint_gates_must_be_context_only():
    gates = {"all_declared_gates_passed": False}
    report = curve_v2.validate_measured_readout(
        _readout(7, 325, gates=gates),  # input prefix is 7
        prefixes=(0, 1, 2, 4, 7), prefix_position=4, endpoint=True)
    assert "endpoint_gate_scope" in " ".join(report["violations"])
    report = curve_v2.validate_measured_readout(
        _readout(7, 325, gates={
            "all_declared_gates_passed": False,
            "endpoint_context_only": True,
            "curve_decision_input": False}),
        prefixes=(0, 1, 2, 4, 7), prefix_position=4, endpoint=True)
    assert report["ok"] is True


# ---------------------------------------------------------------------------
# EOS, masks, targets, first tokens
# ---------------------------------------------------------------------------


def _response_records():
    rows = {
        "answer-1": {"response": "Blue"},
        "caption-1": {"answer": "A blue square."},
    }
    labels = {"caption-1": {"answer": "A blue square."}}
    records = [{
        "prefix": 0, "case_id": "answer-1", "control": "correct",
        "kind": "answer", "available": True,
        "generated_ids": [1],
        "generated_text": "",
        "stop_tokens": [1, 130073],
        "target_text": "Blue",
        "prefill": {"argmax_token": 1, "gold_token": 12},
        "prefill_argmax_matches_first_token": True,
    }, {
        "prefix": 0, "case_id": "caption-1", "control": "noimage",
        "kind": "caption", "available": True,
        "generated_ids": [5],
        "generated_text": " \n",
        "stop_tokens": [1, 130073],
        "target_text": "A blue square.",
        "prefill": {"argmax_token": 5, "gold_token": 5},
        "prefill_argmax_matches_first_token": True,
    }]
    return records, rows, labels


def test_response_audit_separates_eos_only_from_whitespace_only():
    records, rows, labels = _response_records()
    audit = curve_v2.response_audit(
        records, rows_by_case=rows, caption_labels=labels)
    assert audit["ok"] is True, audit["violations"]
    assert audit["eos_only"] == 1
    assert audit["whitespace_only"] == 1
    assert audit["by_prefix"]["0"]["eos_only"] == 1
    assert audit["by_prefix"]["0"]["whitespace_only"] == 1
    assert audit["target_equality"]["answer_checked"] == 1
    assert audit["target_equality"]["caption_checked"] == 1


def test_response_audit_rejects_target_inequality_and_self_inconsistency():
    records, rows, labels = _response_records()
    records[0]["target_text"] = "Red"
    records[1]["prefill_argmax_matches_first_token"] = False
    audit = curve_v2.response_audit(
        records, rows_by_case=rows, caption_labels=labels)
    assert audit["ok"] is False
    assert any("answer_target" in item for item in audit["violations"])
    assert any("prefill_self_consistency" in item
               for item in audit["violations"])


def _fit_row(**overrides):
    row = {
        "example_id": "caption:one",
        "kind": "caption",
        "prompt_masked": True,
        "response_only_labels": True,
        "label_shift": 1,
        "contract_valid": True,
        "prompt_masked_tokens": 27,
        "prompt_tokens": 27,
        "supervised_label_tokens": 16,
        "response_tokens": 16,
        "loss_tokens": 16,
        "response_eos": True,
        "eos_in_stop_set": True,
        "pad_free_response": True,
    }
    row.update(overrides)
    return row


def test_response_mask_audit_accepts_exact_masks_and_rejects_leaks():
    audit = curve_v2.validate_response_masks([_fit_row(), _fit_row()])
    assert audit["ok"] is True
    assert audit["rows_checked"] == 2
    assert audit["unique_rows"] == 1
    bad = _fit_row(prompt_masked=False, supervised_label_tokens=17,
                   response_eos=False, pad_free_response=False)
    audit = curve_v2.validate_response_masks([bad])
    assert audit["ok"] is False
    joined = " ".join(audit["violations"])
    assert "prompt_not_masked" in joined
    assert "supervised_width" in joined
    assert "eos_supervision" in joined
    assert "pad_in_response" in joined


# ---------------------------------------------------------------------------
# Projector scale and feature distinguishability
# ---------------------------------------------------------------------------


def test_projector_scale_and_control_features_are_measured():
    torch.manual_seed(0)
    projected = torch.randn(1, 196, 8)
    scale = curve_v2.projector_scale_report(
        projected, text_embedding_rms=0.0159)
    assert scale["shape"] == [1, 196, 8]
    assert scale["rms"] > 0
    assert scale["projected_to_text_rms_ratio"] > 1
    controls = {
        "case-a": {
            "correct": torch.ones(1, 196, 8),
            "sibling": torch.cat(
                (torch.ones(1, 98, 8), -torch.ones(1, 98, 8)), dim=1),
            "shuffled": torch.zeros(1, 196, 8),
        },
        "case-b": {
            "correct": torch.ones(1, 196, 8) * 2,
            "sibling": torch.cat(
                (torch.ones(1, 98, 8), -torch.ones(1, 98, 8)), dim=1),
            "shuffled": torch.zeros(1, 196, 8) + 0.25,
        },
    }
    report = curve_v2.feature_distinguishability_report(
        controls, text_embedding_rms=0.0159)
    assert report["feature_distinguishable"] is True
    assert report["mean_correct_sibling_cosine"] < 0.999
    assert report["mean_correct_shuffled_cosine"] < 0.999
    assert all(case["noimage_present"] is False for case in report["cases"])


def test_feature_report_requires_sibling_and_shuffled_controls():
    controls = {"case": {"correct": torch.ones(1, 4, 8)}}
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="required"):
        curve_v2.feature_distinguishability_report(controls)


# ---------------------------------------------------------------------------
# Step-1 accounting and decisions
# ---------------------------------------------------------------------------


def _step1_readouts():
    return [
        _readout(0, 1362),
        _readout(0, 1362),
        _readout(1, 1442),
    ]


def _training_totals():
    counters = {key: 0 for key in fwd.COUNTER_KEYS}
    counters.update({"backward_calls": 4, "optimizer_steps": 1,
                     "visual_observation_calls": 4,
                     "visual_encoder_calls": 4,
                     "visual_encoder_cache_misses": 4,
                     "projector_passes": 4})
    return {
        "format": fwd.ACCOUNTING_FORMAT,
        "counters": counters,
        "phase_totals": {
            "training": {
                "forward_calls": 8,
                "text_positions": 800,
                "visual_positions": 4 * 196,
                "backward_calls": 4,
            }
        },
    }


def test_step1_accounting_reconciles_5644_decoder_passes():
    report = curve_v2.validate_step1_accounting(
        _step1_readouts(), _training_totals())
    assert report["ok"] is True, report["violations"]
    assert report["totals"]["generated_tokens"] == 4166
    assert report["totals"]["prepare_sense_passes"] == 882
    assert report["totals"]["explicit_logits_forwards"] == 588
    assert report["totals"]["update_forwards"] == 8
    assert report["totals"]["decoder_forward_calls"] == 5644
    assert report["totals"]["backward_calls"] == 4
    assert report["totals"]["optimizer_steps"] == 1
    assert report["visual_observation_calls"] == 220
    assert report["visual_encoder_calls"] == 220
    assert report["visual_encoder_cache_misses"] == 220


def test_step1_accounting_fails_closed_on_length_or_update_drift():
    readouts = _step1_readouts()
    readouts[1]["generation_summary"]["generated_tokens"] = 1361
    report = curve_v2.validate_step1_accounting(
        readouts, _training_totals())
    assert report["ok"] is False
    assert any("generation_lengths" in item for item in report["violations"])
    totals = _training_totals()
    totals["phase_totals"]["training"]["forward_calls"] = 4
    report = curve_v2.validate_step1_accounting(
        _step1_readouts(), totals)
    assert report["ok"] is False
    assert any("decoder_forward_calls" in item for item in report["violations"])


def _anchor_readout(prefix, answer, caption, digest):
    return {
        "prefix": prefix,
        "teacher_forced": {"answer": answer, "caption": caption},
        "projector_state_sha256": digest,
        "generation_digest": f"gen-{prefix}",
    }


def test_step1_statuses_are_explicit_and_preserve_the_v1_verdict():
    prefix0 = _anchor_readout(
        0, curve_v2.STEP1_ANCHORS["prefix0_teacher_forced_loss"]["answer"],
        curve_v2.STEP1_ANCHORS["prefix0_teacher_forced_loss"]["caption"],
        curve_v2.STEP1_INIT_SHA256)
    prefix0_null = copy.deepcopy(prefix0)
    prefix1 = _anchor_readout(
        1, curve_v2.STEP1_ANCHORS["prefix1_teacher_forced_loss"]["answer"],
        curve_v2.STEP1_ANCHORS["prefix1_teacher_forced_loss"]["caption"],
        curve_v2.STEP1_PREFIX1_SHA256)
    accounting = {"ok": True}
    update = {"optimizer_steps": 1, "backward_calls": 4}
    supported = curve_v2.step1_decision(
        prefix0=prefix0, prefix0_null=prefix0_null, prefix1=prefix1,
        update=update, accounting=accounting)
    assert supported["status"] == curve_v2.STEP1_STATUS_SUPPORTED
    assert supported["curve_v1_verdict"] == curve_v2.V1_VERDICT
    assert supported["curve_verdict_preserved"] is True
    assert supported["dev_authorized"] is False
    assert supported["promotion"] is False
    lower = copy.deepcopy(prefix1)
    lower["teacher_forced"] = {"answer": 9.0, "caption": 8.0}
    refuted = curve_v2.step1_decision(
        prefix0=prefix0, prefix0_null=prefix0_null, prefix1=lower,
        update=update, accounting=accounting)
    assert refuted["status"] == curve_v2.STEP1_STATUS_REFUTED
    nondeterministic = copy.deepcopy(prefix0_null)
    nondeterministic["teacher_forced"]["answer"] += 1e-3
    report = curve_v2.step1_decision(
        prefix0=prefix0, prefix0_null=nondeterministic, prefix1=prefix1,
        update=update, accounting=accounting)
    assert report["status"] == curve_v2.STEP1_STATUS_NONDETERMINISTIC
    engineering = curve_v2.step1_decision(
        prefix0=prefix0, prefix0_null=prefix0_null, prefix1=prefix1,
        update=update, accounting={"ok": False})
    assert engineering["status"] == curve_v2.STEP1_STATUS_ENGINEERING


def test_authorization_receipt_binds_config_and_commit():
    config = curve_v2.load_config(STEP1_CONFIG)
    config_sha = _sha("configs/vision_learnability_step1_reproduction_v1.json")
    receipt = {
        "format": curve_v2.AUTHORIZATION_FORMAT,
        "status": "authorized",
        "config_sha256": config_sha,
        "protocol_name": config["name"],
        "protocol_commit": "abcdef1234567890",
        "native_execution_authorized": True,
        "dev_access": False,
        "heldout_access": False,
    }
    assert curve_v2.validate_authorization_receipt(
        receipt, config_sha256=config_sha, config=config)["status"] == (
            "authorized")
    bad = dict(receipt, config_sha256="0" * 64)
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="exact config"):
        curve_v2.validate_authorization_receipt(
            bad, config_sha256=config_sha, config=config)
    bad = dict(receipt, protocol_commit=None)
    with pytest.raises(learn.LearnabilityDiagnosticError,
                       match="protocol commit"):
        curve_v2.validate_authorization_receipt(
            bad, config_sha256=config_sha, config=config)


# ---------------------------------------------------------------------------
# Preserved receipts
# ---------------------------------------------------------------------------


@require_local_artifacts(
    "runs/vision_learnability_curve_v1_20261026_attempt04",
    "reports/vision_learnability_curve_v1_20261026",
    "data/vision_balanced_alignment_v2_20261026/prepared04",
)
def test_historical_correction_reconstructs_7695_and_audits_receipts():
    evidence = curve_v2.build_historical_correction(
        root=ROOT,
        run="runs/vision_learnability_curve_v1_20261026_attempt04",
        summary="reports/vision_learnability_curve_v1_20261026/summary.json",
        step1_config="configs/vision_learnability_step1_reproduction_v1.json")
    validation = curve_v2.validate_historical_correction(evidence)
    assert validation["ok"] is True
    correction = evidence["correction"]
    assert correction["measured_by_old_run"] is False
    assert correction["declared_readout_decoder_passes"] == {
        "0": 1852, "1": 1932, "2": 1886, "4": 1211, "7": 814}
    assert correction["reconstruction"]["totals"][
        "scoring_decoder_passes"] == 7695
    assert correction["grand_total_events"] == 7755
    assert evidence["audits"]["response"]["eos_only"] == 103
    assert evidence["audits"]["response"]["whitespace_only"] == 47
    assert evidence["audits"]["response"]["by_prefix"]["7"] == {
        "available": 98, "eos_only": 52, "whitespace_only": 0}
    assert evidence["audits"]["response"]["target_equality"][
        "checked"] == 490
    assert evidence["audits"]["response_masks"]["ok"] is True
    assert evidence["audits"]["response_masks"]["unique_rows"] == 26
    assert evidence["preserved_source_receipts"][
        "reports/vision_learnability_curve_v1_20261026/summary.json"] == _sha(
            "reports/vision_learnability_curve_v1_20261026/summary.json")
    assert evidence["v1_verdict"] == curve_v2.V1_VERDICT
