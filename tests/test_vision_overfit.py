"""Overfit diagnostic helpers: supervision template, controls and decision mapping."""
import json
from pathlib import Path

import pytest

from replica_lab import vision_diagnostic as vd
from replica_lab import vision_overfit as vo
from replica_lab import vision_qualification as vq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "vision_preparation_v1"


def dataset():
    protocol = json.loads((ROOT / "configs/minicpm5_vision_v1_projector_alignment_v2.json").read_text())
    protocol["data"]["root"] = str(DATA)
    return vq.load_verified_dataset(protocol)


def test_overfit_training_rows_use_pinned_template_and_train_split():
    value = dataset()
    cases = vd.select_cases(value, "train", limit=2, per_world=1)
    rows = vo.overfit_training_rows(value, cases)
    assert rows
    for row in rows:
        assert row["split"] == "train"
        assert row["response"]
        assert row["prompt"].startswith("Question: ")
        assert row["prompt"].endswith("Reply briefly with the answer alone.")
        assert row["image_sha256"]
        assert len(row["case_id"]) > 4


def test_summary_counts_by_control():
    rows = [
        {"control": "correct", "correct": True},
        {"control": "correct", "correct": False},
        {"control": "wrong", "correct": True},
        {"control": "noimage", "correct": False},
    ]
    summary = vo.summarize(rows)
    assert summary["correct"]["accuracy"] == 0.5
    assert summary["wrong"]["accuracy"] == 1.0
    assert summary["noimage"]["accuracy"] == 0.0


def test_decision_separates_capacity_failure_overfit_and_grounding():
    failure = vo.decide_overfit({"correct": {"accuracy": 0.4}},
                                {"correct": {"accuracy": 0.4}, "wrong": {"accuracy": 0.3},
                                 "noimage": {"accuracy": 0.2}})
    assert failure["decision"] == "pipeline_capacity_or_training_failure"
    overfit = vo.decide_overfit({"correct": {"accuracy": 1.0}},
                                {"correct": {"accuracy": 0.25}, "wrong": {"accuracy": 0.3},
                                 "noimage": {"accuracy": 0.3}})
    assert overfit["decision"] == "overfit_without_image_grounded_generalization"
    grounded = vo.decide_overfit({"correct": {"accuracy": 1.0}},
                                 {"correct": {"accuracy": 0.7}, "wrong": {"accuracy": 0.3},
                                  "noimage": {"accuracy": 0.3}})
    assert grounded["decision"] == "overfit_cohort_shows_image_grounding"


def test_overfit_protocol_bounds(tmp_path):
    template = json.loads((ROOT / "configs/vision_overfit_v1.json").read_text())
    assert vo.validate_overfit_protocol(template)["training_updates"] == 16
    broken = json.loads(json.dumps(template))
    broken["train"]["updates"] = 129
    with pytest.raises(vq.QualificationError, match="bounded"):
        vo.validate_overfit_protocol(broken)
    broken = json.loads(json.dumps(template))
    broken["train_worlds"] = 32
    with pytest.raises(vq.QualificationError, match="tiny"):
        vo.validate_overfit_protocol(broken)
    broken = json.loads(json.dumps(template))
    broken["implementation_sha256"] = {}
    with pytest.raises(vq.QualificationError, match="pins"):
        vo.validate_overfit_protocol(broken)


def test_overfit_supports_declared_sgd_and_wider_projector_options(tmp_path):
    template = json.loads((ROOT / "configs/vision_overfit_sgd_v1.json").read_text())
    assert template["train"]["optimizer"] == "sgd"
    vo.validate_overfit_protocol(template)
    wide = json.loads((ROOT / "configs/vision_overfit_wide_v1.json").read_text())
    assert wide["vision"]["projector_hidden"] == 4096
    vo.validate_overfit_protocol(wide)
    from replica_lab.architecture.vision import VisionFeatureService
    import inspect
    assert "projector_hidden" in inspect.signature(VisionFeatureService.__init__).parameters
