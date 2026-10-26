"""Fresh stage-wise vision data and protocol contracts (CPU only)."""
import json
from pathlib import Path

import replica_lab.vision_stage_data as stage
from replica_lab.vision_curriculum import load_config
from test_vision_curriculum import synthetic_external_source

ROOT = Path(__file__).resolve().parents[1]


def test_fresh_stage_data_protocol_and_seals(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "load_external_source",
                        lambda project_root, config: synthetic_external_source())
    config = stage.load_stage_config(ROOT / "configs/vision_stage_diagnostic_v1.json")
    data_config = load_config(ROOT / config["parent_data_config"]["path"])
    data_config["seed"] = config["seed"]
    data_config["data_root"] = "data/vision_stage_test"
    data_config["procedural"]["id_prefix"] = config["procedural_overrides"]["id_prefix"]
    built = stage.build_stage_dataset(
        data_config, ROOT, cases_per_split=config["evaluation"]["cases_per_split"])
    config["data_root"] = "data/vision_stage_test"
    config["project_root"] = str(ROOT)
    assert built["manifest"]["fresh_evaluation"] is True
    assert built["manifest"]["consumed_audit_or_final_examples_reused"] is False
    assert built["manifest"]["case_namespace"] == "vc2"
    assert all(row["case_id"].startswith("vc2-")
               for row in built["dev"] + built["audit"])
    assert len(built["dev"]) == len(built["audit"]) == 24
    assert len(built["caption_dev"]) == len(built["caption_audit"]) == 24
    assert any(row["label"] == "unknown" for row in built["dev"])
    assert {row["task"] for row in built["dev"]} == {
        "spatial", "count", "presence", "color", "abstain"}
    assert all(row["response"].casefold() not in row["question"].casefold()
               for row in built["stage_a"] + built["stage_b"])
    for rows in (built["dev_labels"], built["audit_labels"],
                 built["caption_dev_labels"], built["caption_audit_labels"]):
        assert all(row["label_visibility"] == "verifier_only" for row in rows)
    stage.write_stage_dataset(tmp_path / "prepared", built)
    parent = json.loads((ROOT / "data/vision_curriculum_20261025/protocol.json").read_text())
    protocol = stage.build_stage_protocol(config, built, parent)
    protocol_sha = stage.write_stage_protocol(tmp_path / "prepared/protocol.json", protocol)
    assert len(protocol_sha) == 64
    assert protocol["train"]["optimizer"] == "adamw_fused"
    assert protocol["train"]["fused"] is True
    assert protocol["schedule"]["total_updates"] == 32
    assert protocol["schedule"]["total_examples"] == 128
    assert protocol["resource_policy"]["driver_limit_bytes"] == 50 * 1024**3
    assert protocol["data"]["seals"] == built["manifest"]["seals"]
    assert protocol["data"]["fresh_evaluation"] is True


def test_caption_excludes_occluded_objects():
    scene = {"objects": [
        {"role": "target", "color": "red", "shape": "circle"},
        {"role": "occluded", "color": "blue", "shape": "square"},
        {"role": "distractor", "color": "green", "shape": "triangle"},
    ]}
    caption = stage.scene_caption(scene)
    assert caption == "a red circle and a green triangle"
    assert "blue" not in caption and "square" not in caption
