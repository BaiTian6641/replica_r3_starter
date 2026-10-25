"""CPU contracts for the frozen vision information probe (no native model)."""
import json
from pathlib import Path

import numpy as np
import pytest

from replica_lab import vision_diagnostic as vd
from replica_lab import vision_qualification as vq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "vision_preparation_v1"


def protocol():
    template = json.loads((ROOT / "configs/minicpm5_vision_v0_qualification_v3.json").read_text())
    data_protocol = json.loads((ROOT / "configs/minicpm5_vision_v1_projector_alignment_v2.json").read_text())
    return {
        "format": vd.DIAGNOSTIC_FORMAT, "root": ".", "stage": "v1",
        "vision": template["vision"], "vision_digest": template["vision_digest"],
        "vision_source": template["vision_source"], "model": template["model"],
        "data": data_protocol["data"],
        "implementation_sha256": {name: vq.sha256_file(ROOT / name)
                                  for name in vd.DIAGNOSTIC_IMPLEMENTATION_FILES},
        "limits": {"max_observations": 128, "max_seconds": 1800,
                   "min_free_disk_bytes": 5368709120, "max_rows": 2048},
        "seed": 17, "train_worlds": 24, "dev_worlds": 24, "per_world_cases": 2,
        "decision": dict(vd.DEFAULT_DECISION),
    }


def dataset():
    value = protocol()
    value["data"]["root"] = str(DATA)
    value["data"]["stage"] = "v1"
    return vq.load_verified_dataset(value), value


def test_protocol_rejects_unbounded_or_unpinned_geometry(tmp_path):
    value = protocol()
    vd.validate_diagnostic_protocol(value)
    broken = json.loads(json.dumps(value))
    broken["limits"]["max_observations"] = 8
    with pytest.raises(vq.QualificationError, match="max_observations"):
        vd.validate_diagnostic_protocol(broken)
    broken = json.loads(json.dumps(value))
    broken["implementation_sha256"].pop("src/replica_lab/vision_diagnostic.py")
    with pytest.raises(vq.QualificationError, match="implementation pins"):
        vd.validate_diagnostic_protocol(broken)
    broken = json.loads(json.dumps(value))
    broken["train_worlds"] = 1
    with pytest.raises(vq.QualificationError, match="eight"):
        vd.validate_diagnostic_protocol(broken)


def test_world_selection_is_whole_deterministic_and_split_disjoint():
    dataset_value, _ = dataset()
    train = vd.select_cases(dataset_value, "train", limit=6, per_world=2)
    again = vd.select_cases(dataset_value, "train", limit=6, per_world=2)
    dev = vd.select_cases(dataset_value, "dev", limit=6, per_world=2)
    assert [item["case_id"] for item in train] == [item["case_id"] for item in again]
    assert {item["group_id"] for item in train}.isdisjoint(
        {item["group_id"] for item in dev})
    for group in {item["group_id"] for item in train}:
        tasks = [item["case"]["task"] for item in train if item["group_id"] == group]
        assert len(tasks) == len(set(tasks))


def test_ridge_recovers_separable_synthetic_signal_and_ignores_noise():
    rng = np.random.default_rng(3)
    centres = {"a": [2.0, 0.0], "b": [0.0, 2.0]}
    labels = ["a", "b"] * 6
    features = np.stack([np.array(centres[label]) + rng.normal(scale=0.1, size=2)
                         for label in labels])
    classes = ["a", "b"]
    model = vd._ridge_fit(features, vd._one_hot(labels, classes))
    predictions = vd._ridge_predict(model, features, classes)
    assert predictions == labels
    noise = rng.normal(size=(4, 2)) * 0.05
    assert vd._ridge_predict(model, noise, classes) != ["a"] * 4


def _results(combined, image, question, chance=0.125, shuffled=None):
    baseline = {"dev_majority_accuracy": chance, "dev_class_count": 8,
                "dev_random_chance": chance, "train_label_count": 8}
    return {
        "combined": {"dev_accuracy": combined,
                     "dev_accuracy_shuffled_images": combined if shuffled is None else shuffled},
        "image_only": {"dev_accuracy": image},
        "question_only": {"dev_accuracy": question},
        "baselines": baseline,
    }


def test_decision_maps_evidence_to_one_justified_change():
    supervision = vd.decide(_results(0.15, 0.15, 0.15), {})
    assert supervision["decision"] == "fail_supervision_or_pairing"
    encoder = vd.decide(_results(0.30, 0.15, 0.30), {})
    assert encoder["decision"] == "fail_encoder_features_carry_no_linear_signal"
    insufficient = vd.decide(_results(0.15, 0.30, 0.25), {})
    assert insufficient["decision"] == "fail_features_insufficient"
    supported = vd.decide(_results(0.75, 0.45, 0.30, shuffled=0.40), {})
    assert supported["decision"] == "frozen_features_support_readout"
    assert supported["shuffled_image_drop"] == pytest.approx(0.35)
    assert supported["justified_next_change"].startswith("A linear readout")
    inconclusive = vd.decide(_results(0.4167, 0.1875, 0.4375, chance=1 / 12,
                                      shuffled=0.375), {})
    assert inconclusive["decision"] == "inconclusive_visual_contribution"
    no_drop = vd.decide(_results(0.75, 0.45, 0.30), {})
    assert no_drop["decision"] == "inconclusive_visual_contribution"


def test_feature_statistics_report_scale_mask_and_pair_shape():
    rng = np.random.default_rng(5)
    records = [{"image": rng.normal(size=8), "patches": rng.normal(size=(4, 8)),
                "task": "color", "label": "blue", "split": "train", "world": "w",
                "case_id": "c"} for _ in range(3)]
    stats = vd.feature_statistics(records)
    assert stats["cases"] == 3 and stats["patch_count"] == 4
    assert stats["image_dim"] == 8
    assert stats["labels_by_task"] == {"color": ["blue"]}
    assert -1.0 <= stats["patch_adjacent_case_cosine_mean"] <= 1.0
