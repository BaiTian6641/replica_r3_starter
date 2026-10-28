"""Tests for positive-diagnostic-gated vision expansion readiness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import check_vision_expansion_readiness as expansion  # noqa: E402
import validate_vision_acceptance as acceptance  # noqa: E402


CONFIG = ROOT / "configs/vision_expansion_readiness_v1.json"
ACCEPTANCE_CONFIG = ROOT / "configs/vision_diagnostic_acceptance_v1.json"
PROVENANCE = {
    "correct": "correct_image",
    "wrong": "changed_sibling_image",
    "shuffled": "cross_pair_shuffled_image",
    "noimage": "no_image",
}


def _score(accuracy: float) -> dict:
    return {
        "case_ids": ["case-a", "case-b"],
        "controls": {
            name: {"available": True, "accuracy": accuracy}
            for name in ("correct", "wrong", "shuffled", "noimage")
        },
        "paired": {
            name: {"correct_margin": 0.1}
            for name in ("wrong", "shuffled", "noimage")
        },
        "control_provenance": dict(PROVENANCE),
    }


def _retention() -> dict:
    def side(accuracy: float) -> dict:
        return {"cases": 2, "accuracy": accuracy,
                "rows": [{"case_id": "case-a"}, {"case_id": "case-b"}]}
    return {"text_only": True, "before": side(0.8), "after": side(0.79)}


def _link_inputs(root: Path) -> None:
    for relative in (
            "reports/caption_admission_preparation_20261027/partition_identity.json",
            "configs/caption_admission_receipt_20261027.json",
            "configs/vision_diagnostic_acceptance_v1.json",
            "data/synthetic_scene_eval_20261027/scene_specs.json"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / relative)


def _write_positive(root: Path, *, device: str = "mps") -> None:
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    baseline_path = evidence / "baseline.json"
    candidate_path = evidence / "candidate.json"
    reload_path = evidence / "reload.json"
    baseline_path.write_text(json.dumps({"dev": {"answer": _score(0.2)}}),
                             encoding="utf-8")
    candidate_path.write_text(json.dumps({
        "device": device,
        "cpu_fallback_enabled": False,
        "dev": {"answer": _score(0.5)},
        "retention": _retention(),
    }), encoding="utf-8")
    digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    comparisons = (
        "train_fit_answer_generations.jsonl",
        "train_fit_caption_generations.jsonl",
        "retention_after.jsonl",
        "scores/retention.json",
        "gates/train_fit.json",
    )
    reload_path.write_text(json.dumps({
        "fresh_process": True,
        "zero_updates": True,
        "extension_only": True,
        "base_unchanged": True,
        "base_saved": False,
        "bitwise_equal": True,
        "run_report_sha256": digest,
        "comparisons": {name: True for name in comparisons},
    }), encoding="utf-8")
    report = acceptance.assess(
        baseline_path=baseline_path, candidate_path=candidate_path,
        reload_path=reload_path, config_path=ACCEPTANCE_CONFIG)
    destination = root / "reports/vision_diagnostic_acceptance_v1_20261028/assessment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report), encoding="utf-8")


def test_current_partitions_pass_but_expansion_waits_for_positive_diagnostic():
    report = expansion.assess(config_path=CONFIG)
    assert report["partitions"]["ok"] is True
    assert report["partitions"]["counts"] == {
        "admitted": 9, "exclusions": 21, "exposed": 8,
        "diagnostic_eval_scenes": 24,
    }
    assert report["partitions"]["source_group_counts"] == {
        "admitted": 9, "exclusions": 21, "exposed": 8,
    }
    assert report["positive_diagnostic"]["violations"] == [
        "positive_diagnostic_missing"]
    assert report["expansion_preparation_allowed"] is False
    assert report["native_execution"] is False
    assert report["model_loads"] == 0


def test_positive_mps_diagnostic_allows_only_preparation(tmp_path):
    _link_inputs(tmp_path)
    _write_positive(tmp_path)
    report = expansion.assess(root=tmp_path, config_path=CONFIG)
    assert report["positive_diagnostic"]["ok"] is True
    assert report["expansion_preparation_allowed"] is True
    assert report["status"] == "review_ready_expansion_preparation_allowed"
    assert report["training_authorized"] is False
    assert report["native_execution"] is False


def test_cpu_diagnostic_cannot_open_expansion(tmp_path):
    _link_inputs(tmp_path)
    _write_positive(tmp_path, device="cpu")
    report = expansion.assess(root=tmp_path, config_path=CONFIG)
    assert report["expansion_preparation_allowed"] is False
    assert "positive_diagnostic_not_mps" in report["refusals"]


def test_partition_tampering_fails_closed(tmp_path):
    _link_inputs(tmp_path)
    identity_path = tmp_path / "reports/caption_admission_preparation_20261027/partition_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["counts"]["admitted"] = 8
    identity_path.unlink()
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    report = expansion.assess(root=tmp_path, config_path=CONFIG)
    assert report["partitions"]["ok"] is False
    assert "declared_count:admitted" in report["refusals"]
    assert report["expansion_preparation_allowed"] is False
