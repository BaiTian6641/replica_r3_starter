#!/usr/bin/env python3
"""Prepare balanced caption/answer data and its sealed alignment protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_balanced_alignment import (  # noqa: E402
    build_balanced_dataset, build_protocol, load_prepared,
    load_prior_evaluation, sha256_file, validate_config, write_dataset,
    write_protocol,
)


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/vision_balanced_alignment_v1.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports/vision_balanced_alignment_20261026/preparation.json")
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = validate_config(json.loads(config_path.read_text()))
    output = (args.output or project_root / config["data_root"]).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh prepared-data output directory")
    for name in ("parent_data_config", "parent_protocol"):
        pin = config[name]
        if sha256_file(project_root / pin["path"]) != pin["sha256"]:
            parser.error(f"{name} bytes differ from the declared pin")
    prior = load_prior_evaluation(project_root, config)
    parent_protocol = json.loads(
        (project_root / config["parent_protocol"]["path"]).read_text())

    built = build_balanced_dataset(config, project_root)
    output.mkdir(parents=True, exist_ok=True)
    write_dataset(output, built)
    protocol_config = dict(config)
    resolved_data_root = (
        output.relative_to(project_root).as_posix()
        if output.is_relative_to(project_root) else str(output))
    protocol_config["data_root"] = resolved_data_root
    protocol_config = validate_config(protocol_config)
    protocol = build_protocol(protocol_config, built, parent_protocol,
                              project_root=project_root)
    protocol_sha256 = write_protocol(output / "protocol.json", protocol)
    verified_protocol, manifest, implementation, _ = load_prepared(
        output / "protocol.json", project_root)
    evaluation_cases = (
        built["dev_answer_cases"] + built["heldout_answer_cases"]
        + built["dev_caption_cases"] + built["heldout_caption_cases"])
    label_rows = (
        built["dev_answer_labels"] + built["heldout_answer_labels"]
        + built["dev_caption_labels"] + built["heldout_caption_labels"])
    control_rows = (
        built["dev_answer_controls"] + built["heldout_answer_controls"]
        + built["dev_caption_controls"] + built["heldout_caption_controls"])
    labels_outside = all(
        "answer" not in row and "label" not in row and "response" not in row
        for row in evaluation_cases) and all(
        row["label_visibility"] == "verifier_only" for row in label_rows)
    controls_prepared = all(
        {"shuffled", "counterfactual"} <= set(row) for row in control_rows)
    prior_preserved = all(
        sha256_file(project_root / config["prior_consumed"][name]["path"])
        == config["prior_consumed"][name]["sha256"]
        for name in ("protocol", "manifest"))
    checks = {
        "fresh_namespace": (manifest["case_namespace"]
                            not in prior["namespaces"]),
        "heldout_disjoint_from_prior_evaluation": all(
            value == 0 for metric in ("case_overlap", "image_overlap",
                                      "scene_overlap")
            for value in (manifest["freshness"]["heldout"][metric],)),
        "dev_disjoint_from_prior_evaluation": all(
            value == 0 for metric in ("case_overlap", "image_overlap",
                                      "scene_overlap")
            for value in (manifest["freshness"]["dev"][metric],)),
        "train_disjoint_from_prior_evaluation": all(
            value == 0 for metric in ("case_overlap", "image_overlap",
                                      "scene_overlap")
            for value in (manifest["freshness"]["train"][metric],)),
        "all_used_images_disjoint_from_prior_assets": (
            manifest["freshness"]["all_used_prior_assets"]["image_overlap"]
            == 0),
        "semantic_image_caption_checks": (
            manifest["semantic_audit"]["all_passed"] is True
            and manifest["semantic_audit"]["cases_checked"] > 0
            and manifest["semantic_audit"]["visible_pixels_verified"] > 0),
        "balanced_answer_and_caption_classes": all(
            len(set(counts.values())) == 1
            for role in manifest["balance"].values()
            for counts in (role["answer_histogram_selected"],
                           role["caption_histogram_selected"])),
        "labels_outside_model_cases": labels_outside,
        "counterfactual_controls_prepared": controls_prepared,
        "prior_stage_result_preserved": prior_preserved,
        "native_execution": False,
        "protocol_verified_after_write": (
            verified_protocol["data"]["manifest"]["sha256"]
            == sha256_file(output / "manifest.json")),
    }
    if not all(value for key, value in checks.items()
               if key not in ("native_execution",)):
        raise RuntimeError(f"Preparation checks failed: {checks}")
    admission = {
        "format": "replica.vision_balanced_alignment.admission.v1",
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": protocol["data"]["manifest"]["sha256"],
        "config_sha256": protocol["config_sha256"],
        "implementation_sha256": implementation,
        "counts": protocol["data"]["counts"],
        "seals": protocol["data"]["seals"],
        "balance": protocol["data"]["balance"],
        "freshness": protocol["data"]["freshness"],
        "semantic_audit": protocol["data"]["semantic_audit"],
        "checks": checks,
        "native_command": (
            "python scripts/run_vision_balanced_alignment.py --mode train "
            "--protocol <prepared>/protocol.json --device mps "
            "--output <fresh-report-dir> --model-path <MiniCPM5> "
            "--model-lock <lock>"),
        "reload_command": (
            "python scripts/run_vision_balanced_alignment.py --mode reload "
            "--protocol <prepared>/protocol.json --device mps "
            "--run <train-report-dir> --output <fresh-reload-dir> "
            "--model-path <MiniCPM5> --model-lock <lock>"),
        "promotion": False,
        "diagnostic_only": True,
    }
    _json(output / "admission.json", admission)
    report = {
        "format": "replica.vision_balanced_alignment.preparation_report.v1",
        "status": "prepared",
        "config": str(config_path.relative_to(project_root)
                      if config_path.is_relative_to(project_root)
                      else config_path),
        "output": resolved_data_root,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": admission["manifest_sha256"],
        "counts": admission["counts"],
        "balance": admission["balance"],
        "freshness": admission["freshness"],
        "semantic_audit": admission["semantic_audit"],
        "checks": checks,
        "native_command": admission["native_command"],
        "reload_command": admission["reload_command"],
        "note": ("CPU preparation only. Native training is run by the parent "
                 "after protocol review. The rejected stage result is "
                 "preserved and is not reused as a target."),
    }
    _json(args.report, report)
    print(json.dumps({
        "status": "prepared",
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": admission["manifest_sha256"],
        "counts": admission["counts"],
        "output": str(output),
        "report": str(args.report),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
