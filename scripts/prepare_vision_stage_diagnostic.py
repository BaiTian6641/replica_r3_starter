#!/usr/bin/env python3
"""Prepare fresh stage-wise vision data and its sealed native protocol."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_curriculum import load_config, sha256_file  # noqa: E402
from replica_lab.vision_stage_data import (  # noqa: E402
    build_stage_dataset, build_stage_protocol, load_stage_config,
    write_stage_dataset, write_stage_protocol,
)


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/vision_stage_diagnostic_v1.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_stage_config(config_path)
    project_root = args.project_root.resolve()
    output = (args.output or project_root / config["data_root"]).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh prepared-data output directory")
    for name in ("parent_data_config", "parent_protocol"):
        pin = config[name]
        if sha256_file(project_root / pin["path"]) != pin["sha256"]:
            parser.error(f"{name} bytes differ from the declared pin")
    parent_protocol = json.loads(
        (project_root / config["parent_protocol"]["path"]).read_text())
    parent_data_config = load_config(project_root / config["parent_data_config"]["path"])
    data_config = copy.deepcopy(parent_data_config)
    data_config["seed"] = config["seed"]
    data_config["data_root"] = config["data_root"]
    overrides = config["procedural_overrides"]
    data_config["procedural"]["id_prefix"] = overrides["id_prefix"]
    for split, layouts in overrides.get("split_layouts", {}).items():
        data_config["procedural"]["split_layouts"][split] = list(layouts)
    built = build_stage_dataset(data_config, project_root,
                                cases_per_split=config["evaluation"]["cases_per_split"])
    output.mkdir(parents=True, exist_ok=True)
    write_stage_dataset(output, built)
    protocol_config = copy.deepcopy(config)
    protocol_config["project_root"] = str(project_root)
    protocol = build_stage_protocol(protocol_config, built, parent_protocol)
    protocol_sha256 = write_stage_protocol(output / "protocol.json", protocol)
    admission = {
        "format": "replica.vision_stage_diagnostic.admission.v1",
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": protocol["data"]["manifest"]["sha256"],
        "config_sha256": protocol["config_sha256"],
        "implementation_sha256": protocol["implementation_sha256"],
        "counts": protocol["data"]["counts"],
        "seals": protocol["data"]["seals"],
        "checks": {
            "fresh_case_namespace": True,
            "consumed_audit_or_final_examples_reused": False,
            "train_label_coverage_preserved": True,
            "external_final_not_used": True,
            "audit_sealed_until_after_stage_b": True,
            "native_execution": False,
        },
    }
    _json(output / "admission.json", admission)
    print(json.dumps({"protocol_sha256": protocol_sha256,
                      "manifest_sha256": admission["manifest_sha256"],
                      "counts": admission["counts"],
                      "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
