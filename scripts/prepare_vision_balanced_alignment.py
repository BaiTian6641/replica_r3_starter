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
    CONFIG_FORMAT_V2, SCORER_CONTRACT_FORMAT, build_balanced_dataset,
    build_protocol, load_prepared, load_prior_evaluation, sha256_file,
    validate_config, verify_answer_control_contracts_v2,
    verify_caption_control_contracts_v2, write_dataset, write_protocol,
)


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def _answer_scored(cases, labels):
    by_id = {row["case_id"]: row for row in labels}
    return [{**case, "label": by_id[case["case_id"]]["answer"]}
            for case in cases]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/vision_balanced_alignment_v1.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = validate_config(json.loads(config_path.read_text()))
    if args.report is None:
        report_root = config.get("report_root",
                                 "reports/vision_balanced_alignment_20261026")
        args.report = project_root / report_root / "preparation.json"
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
    scorer_contracts = None
    if config["format"] == CONFIG_FORMAT_V2:
        # The sealed payload is not written until the corrected target,
        # identity, tamper and positive/negative control contracts pass.
        dev_caption_controls = {
            row["case_id"]: row
            for row in built["dev_caption_controls"]}
        heldout_caption_controls = {
            row["case_id"]: row
            for row in built["heldout_caption_controls"]}
        dev_answer_controls = {
            row["case_id"]: row for row in built["dev_answer_controls"]}
        heldout_answer_controls = {
            row["case_id"]: row
            for row in built["heldout_answer_controls"]}
        scorer_contracts = {
            "format": SCORER_CONTRACT_FORMAT,
            "dev_caption": verify_caption_control_contracts_v2(
                built["dev_caption_cases"], built["dev_caption_labels"],
                dev_caption_controls, controls=config["evaluation"]
                ["caption_controls"], seed=int(config["seed"]),
                bootstrap=config["evaluation"]["bootstrap"],
                evaluation=config["evaluation"]),
            "heldout_caption": verify_caption_control_contracts_v2(
                built["heldout_caption_cases"],
                built["heldout_caption_labels"],
                heldout_caption_controls, controls=config["evaluation"]
                ["caption_controls"], seed=int(config["seed"]) + 1,
                bootstrap=config["evaluation"]["bootstrap"],
                evaluation=config["evaluation"]),
            "dev_answer": verify_answer_control_contracts_v2(
                built["dev_answer_scored"] if "dev_answer_scored" in built
                else _answer_scored(built["dev_answer_cases"],
                                    built["dev_answer_labels"]),
                built["dev_answer_labels"], dev_answer_controls,
                controls=["correct", "wrong", "shuffled", "noimage"],
                seed=int(config["seed"]) + 2,
                bootstrap=config["evaluation"]["bootstrap"],
                evaluation=config["evaluation"]),
            "heldout_answer": verify_answer_control_contracts_v2(
                _answer_scored(built["heldout_answer_cases"],
                               built["heldout_answer_labels"]),
                built["heldout_answer_labels"], heldout_answer_controls,
                controls=config["evaluation"]["heldout_answer_controls"],
                seed=int(config["seed"]) + 3,
                bootstrap=config["evaluation"]["bootstrap"],
                evaluation=config["evaluation"]),
        }
        report_root = project_root / config["report_root"]
        report_root.mkdir(parents=True, exist_ok=True)
        _json(report_root / "scorer_contracts.json", scorer_contracts)
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
    if config["format"] == CONFIG_FORMAT_V2:
        controls_prepared = controls_prepared and all(
            row["assignment_sha256"]
            for row in control_rows)
    if isinstance(config["prior_consumed"], list):
        prior_preserved = all(
            sha256_file(project_root / entry[name]["path"])
            == entry[name]["sha256"]
            for entry in config["prior_consumed"]
            for name in ("protocol", "manifest"))
    else:
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
    if scorer_contracts is not None:
        checks["scorer_contracts_passed"] = all(
            value.get("all_contracts_passed") is True
            and value["positive"]["all_declared_gates_passed"] is True
            for value in scorer_contracts.values()
            if isinstance(value, dict) and "all_contracts_passed" in value)
    if not all(value for key, value in checks.items()
               if key not in ("native_execution",)):
        raise RuntimeError(f"Preparation checks failed: {checks}")
    admission = {
        "format": ("replica.vision_balanced_alignment.admission.v2"
                   if scorer_contracts is not None else
                   "replica.vision_balanced_alignment.admission.v1"),
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
        **({"scorer_contracts": scorer_contracts} if scorer_contracts else {}),
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
        "format": ("replica.vision_balanced_alignment.preparation_report.v2"
                   if scorer_contracts is not None else
                   "replica.vision_balanced_alignment.preparation_report.v1"),
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
        **({"scorer_contracts": scorer_contracts} if scorer_contracts else {}),
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
