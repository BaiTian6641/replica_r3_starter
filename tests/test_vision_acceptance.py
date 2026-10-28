"""Behavioral tests for the vision diagnostic acceptance contract."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab.vision_acceptance import (  # noqa: E402
    VisionAcceptanceError, evaluate_vision_acceptance,
)
import validate_vision_acceptance as cli  # noqa: E402


THRESHOLDS = {
    "min_answer_accuracy_gain": 0.05,
    "min_correct_margin_wrong": 0.05,
    "min_correct_margin_shuffled": 0.05,
    "min_correct_margin_noimage": 0.05,
    "max_retention_decline": 0.02,
}
PROVENANCE = {
    "correct": "correct_image",
    "wrong": "changed_sibling_image",
    "shuffled": "cross_pair_shuffled_image",
    "noimage": "no_image",
}
RELOAD_COMPARISONS = ("answers.jsonl", "retention.json", "gates.json")
DIGEST = "d" * 64


def _score(accuracy: float, margin: float = 0.1) -> dict:
    return {
        "case_ids": ["case-a", "case-b"],
        "controls": {
            name: {"available": True, "accuracy": accuracy}
            for name in ("correct", "wrong", "shuffled", "noimage")
        },
        "paired": {
            name: {"correct_margin": margin}
            for name in ("wrong", "shuffled", "noimage")
        },
        "control_provenance": dict(PROVENANCE),
    }


def _retention(before: float = 0.8, after: float = 0.79) -> dict:
    def score(accuracy: float) -> dict:
        return {
            "cases": 2,
            "accuracy": accuracy,
            "rows": [{"case_id": "case-a"}, {"case_id": "case-b"}],
        }
    return {"text_only": True, "before": score(before), "after": score(after)}


def _reload() -> dict:
    return {
        "fresh_process": True,
        "zero_updates": True,
        "extension_only": True,
        "base_unchanged": True,
        "base_saved": False,
        "bitwise_equal": True,
        "run_report_sha256": DIGEST,
        "comparisons": {name: True for name in RELOAD_COMPARISONS},
    }


def _assess(**changes) -> dict:
    values = {
        "baseline_answer": _score(0.2),
        "candidate_answer": _score(0.5),
        "retention": _retention(),
        "reload_receipt": _reload(),
        "thresholds": THRESHOLDS,
        "candidate_report_sha256": DIGEST,
        "required_reload_comparisons": RELOAD_COMPARISONS,
    }
    values.update(changes)
    return evaluate_vision_acceptance(**values)


def test_acceptance_requires_all_evidence_and_separates_engineering():
    report = _assess()
    assert report["status"] == "accepted"
    assert report["ok"] is True
    assert report["learned_evidence_ok"] is True
    assert report["engineering_evidence_ok"] is True
    assert report["metrics"]["answer_accuracy_gain"] == pytest.approx(0.3)
    assert report["promotion"] is False
    assert report["development_authorized"] is False


def test_acceptance_rejects_no_generated_answer_improvement():
    report = _assess(candidate_answer=_score(0.2))
    assert report["status"] == "rejected"
    assert "generated_answer_improvement" in report["failures"]
    assert report["engineering_evidence_ok"] is True


def test_acceptance_rejects_failed_image_control_and_case_mismatch():
    candidate = _score(0.5, margin=0.0)
    candidate["case_ids"] = ["different-case"]
    report = _assess(candidate_answer=candidate)
    assert "correct_vs_wrong_margin" in report["failures"]
    assert "correct_vs_shuffled_margin" in report["failures"]
    assert "correct_vs_noimage_margin" in report["failures"]
    assert "matched_answer_case_identity" in report["failures"]


def test_acceptance_missing_blank_provenance_is_inconclusive_not_pass():
    candidate = _score(0.5)
    del candidate["control_provenance"]["noimage"]
    report = _assess(candidate_answer=candidate)
    assert report["status"] == "inconclusive"
    assert report["ok"] is False
    assert "controls:noimage_provenance" in report["missing_evidence"]


def test_acceptance_rejects_text_retention_decline():
    report = _assess(retention=_retention(before=0.8, after=0.7))
    assert report["status"] == "rejected"
    assert "text_retention_decline" in report["failures"]
    assert report["learned_evidence_ok"] is True
    assert report["engineering_evidence_ok"] is False


def test_acceptance_rejects_unbound_or_nonfresh_reload():
    receipt = _reload()
    receipt["fresh_process"] = False
    receipt["run_report_sha256"] = "e" * 64
    report = _assess(reload_receipt=receipt)
    assert report["status"] == "rejected"
    assert "fresh_process_reload:fresh_process" in report["failures"]
    assert "fresh_process_reload:candidate_report_binding" in report["failures"]


def test_acceptance_empty_evidence_fails_closed():
    report = _assess(baseline_answer={}, candidate_answer={}, retention={},
                     reload_receipt={}, candidate_report_sha256=None)
    assert report["status"] == "inconclusive"
    assert report["ok"] is False
    assert report["promotion"] is False
    assert report["missing_evidence"]


def test_acceptance_rejects_nonpositive_improvement_threshold():
    with pytest.raises(VisionAcceptanceError, match="positive"):
        _assess(thresholds={**THRESHOLDS, "min_answer_accuracy_gain": 0.0})


def test_acceptance_cli_reads_real_json_reports_and_binds_candidate(tmp_path):
    baseline = {"dev": {"answer": _score(0.2)}}
    candidate = {"dev": {"answer": _score(0.5)}, "retention": _retention()}
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    reload_path = tmp_path / "reload.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    reload_receipt = _reload()
    reload_receipt["run_report_sha256"] = digest
    reload_receipt["comparisons"] = {
        name: True for name in (
            "train_fit_answer_generations.jsonl",
            "train_fit_caption_generations.jsonl",
            "retention_after.jsonl",
            "scores/retention.json",
            "gates/train_fit.json",
        )
    }
    reload_path.write_text(json.dumps(reload_receipt), encoding="utf-8")
    output = tmp_path / "assessment.json"
    assert cli.main([
        "--baseline-report", str(baseline_path),
        "--candidate-report", str(candidate_path),
        "--reload-report", str(reload_path),
        "--output", str(output),
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "accepted"
    assert report["inputs"]["candidate"]["sha256"] == digest
    assert report["native_execution"] is False
    assert report["model_loads"] == 0
