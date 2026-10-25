#!/usr/bin/env python3
"""Generate and audit owned procedural visual data; no model or training work.

The command is idempotent for unchanged inputs: existing artifacts must match
byte-for-byte and are never silently overwritten. It writes a small inspectable
V0 asset/case manifest, a capped V1 candidate manifest, temporal-memory
candidates, the external component allowlist and a preparation report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.visual_data import (  # noqa: E402
    REPORT_FORMAT,
    VisualDataError,
    audit_visual_dataset,
    generate_visual_dataset,
    load_config,
    write_exact,
)


GENERATED_ON = "2026-10-25"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_summary(output: Path, stage: str) -> dict:
    manifest = _load(output / stage / "vision_manifest.json")
    counts = manifest["counts"]
    return {
        "status": manifest["status"],
        "assets": counts["assets"],
        "root_assets": counts["root_assets"],
        "crop_assets": counts["crop_assets"],
        "groups": counts["groups"],
        "model_rows": counts["model_rows"],
        "verifier_cases": counts["verifier_cases"],
        "asset_bytes": counts["asset_bytes"],
        "task_histogram": counts["task_histogram"],
        "splits": {
            split: {"assets": counts["split_assets"].get(split, 0),
                    "groups": counts["split_groups"].get(split, 0),
                    "model_rows": counts["split_model_rows"].get(split, 0)}
            for split in sorted(set(counts["split_assets"]) | set(counts["split_groups"]))
        },
        "train_candidate_assets": (counts["split_assets"].get("train", 0)
                                   if stage == "v1" else None),
        "dev_candidate_assets": (counts["split_assets"].get("dev", 0)
                                 if stage == "v1" else None),
        "generation_fingerprint": manifest["generation_fingerprint"],
        "manifest_sha256": _sha256(output / stage / "vision_manifest.json"),
        "admitted_rows": 0,
    }


def _temporal_summary(output: Path) -> dict:
    manifest = _load(output / "temporal" / "temporal_manifest.json")
    counts = manifest["counts"]
    return {
        "status": manifest["status"],
        "assets": counts["assets"],
        "groups": counts["groups"],
        "observations": counts["observations"],
        "queries_verifier_only": counts["queries"],
        "asset_bytes": counts["asset_bytes"],
        "model_rows_exposed": 0,
        "admitted_rows": 0,
        "generation_fingerprint": manifest["generation_fingerprint"],
        "manifest_sha256": _sha256(output / "temporal" / "temporal_manifest.json"),
    }


def build_report(config_path: Path, config: dict, output: Path,
                 audit: dict) -> dict:
    allowlist = _load(output / "external_component_allowlist.json")
    intake = _load(output / "intake_summary.json")
    stages = {"v0": _stage_summary(output, "v0"),
              "v1": _stage_summary(output, "v1")}
    temporal = _temporal_summary(output)
    files = [path for path in output.rglob("*")
             if path.is_file() and not path.name.startswith(".")]
    total_generated_bytes = sum(path.stat().st_size for path in files)
    candidate_rows = stages["v0"]["model_rows"] + stages["v1"]["model_rows"]
    stage_status = {
        name: sum(entry["stages"][name]["status"] for entry in allowlist["entries"])
        for name in ("located", "pinned", "acquired",
                     "inspected", "normalized", "admitted")
    }
    catalog = allowlist["catalog"]
    manifest_hashes = {name: stages[name]["manifest_sha256"] for name in stages}
    manifest_hashes["temporal"] = temporal["manifest_sha256"]
    manifest_hashes["external_component_allowlist"] = _sha256(
        output / "external_component_allowlist.json")
    samples = {
        "v0": stages["v0"]["assets"] and _load(output / "v0/vision_manifest.json")["assets"][0]["path"],
        "v1": stages["v1"]["assets"] and _load(output / "v1/vision_manifest.json")["assets"][0]["path"],
        "temporal": temporal["assets"] and _load(
            output / "temporal/temporal_manifest.json")["assets"][0]["path"],
    }
    return {
        "format": REPORT_FORMAT,
        "generated_on": GENERATED_ON,
        "status": "audit_passed" if audit["status"] == "passed" else "audit_failed",
        "config": {"path": str(config_path),
                   "fingerprint": intake["config_fingerprint"]},
        "inputs": {
            "source_catalog": {"path": catalog["path"], "sha256": catalog["sha256"],
                               "external_sources": allowlist["counts"]["catalog_external_sources"],
                               "metadata_pinned": allowlist["counts"]["metadata_pinned"],
                               "awaiting_immutable_pin": allowlist["counts"]["awaiting_immutable_pin"]},
            "runtime_registry": allowlist["runtime_registry"],
        },
        "external_components": {
            "catalog_sources": allowlist["counts"]["catalog_external_sources"],
            "stage_status_counts": stage_status,
            "eligible_for_this_campaign": allowlist["counts"]["eligible_for_this_campaign"],
            "assets_acquired_here": 0,
            "bytes_acquired_here": 0,
            "admitted_here": 0,
            "policy": allowlist["acquisition_policy"]["reason"],
            "note": ("No external asset was eligible/needed for this owned procedural "
                     "preparation tranche. Generated image bytes are internally distinct "
                     "from any catalog asset and are not a substitute for real-image "
                     "evaluation."),
        },
        "generated": {
            "stages": stages,
            "temporal": temporal,
            "internal_assets": stages["v0"]["assets"] + stages["v1"]["assets"]
            + temporal["assets"],
            "internal_asset_bytes": (stages["v0"]["asset_bytes"]
                                     + stages["v1"]["asset_bytes"]
                                     + temporal["asset_bytes"]),
            "candidate_model_rows": candidate_rows,
            "total_generated_files": len(files),
            "total_generated_bytes": total_generated_bytes,
        },
        "source_catalog_vs_generated": {
            "catalog_external_sources": allowlist["counts"]["catalog_external_sources"],
            "catalog_pinned_metadata_sources": allowlist["counts"]["metadata_pinned"],
            "catalog_assets_acquired": 0,
            "catalog_asset_bytes_acquired": 0,
            "internal_generated_assets": stages["v0"]["assets"]
            + stages["v1"]["assets"] + temporal["assets"],
            "internal_generated_asset_bytes": stages["v0"]["asset_bytes"]
            + stages["v1"]["asset_bytes"] + temporal["asset_bytes"],
            "internal_generated_candidate_rows": candidate_rows,
            "admitted_training_rows": 0,
        },
        "training_admission": {
            "admitted": False,
            "admitted_training_rows": 0,
            "v0": "Interface-qualification candidate only; not a training run.",
            "v1": ("Capped candidate manifest only. Do not treat tiny procedural "
                   "scenes as broad V1 alignment or admit them without a named protocol."),
            "temporal": ("Fresh storage-probe candidates; observation captions are "
                         "inputs while expected answers/statuses and the event ledger "
                         "remain verifier-only."),
        },
        "campaign_budget": {
            "asset_ceiling_bytes": config["campaign"]["asset_ceiling_bytes"],
            "internal_asset_bytes": audit["asset_bytes"],
            "headroom_bytes": config["campaign"]["asset_ceiling_bytes"] - audit["asset_bytes"],
            "within_asset_ceiling": audit["within_asset_ceiling"],
            "checkpoint_bytes_accounted_separately": True,
            "ram_guard_accounted_separately": True,
            "note": "Checkpoint storage and the 50 GiB native RAM stop guard are separate budgets.",
        },
        "audit": audit,
        "manifest_hashes": manifest_hashes,
        "sample_images": {stage: (str(output / path) if path else None)
                          for stage, path in samples.items()},
        "limitations": [
            "Generated scenes are synthetic controls, not evidence of broad real-image competence.",
            "V0 is interface qualification; V1 rows are candidates pending a named admission protocol.",
            "Rendering metadata, scene graphs and answers are verifier-only; model rows expose image and question.",
            "No native model computation, MPS worker, training or bulk download occurred.",
            "A programmatic label is not infallible; the executable checks reduce but do not remove generator risk.",
        ],
    }


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/vision_data_preparation_v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/vision_preparation_v1")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports/vision_data_preparation_20261025/report.json")
    parser.add_argument("--audit-only", action="store_true",
                        help="Audit existing artifacts without generating.")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.audit_only:
            audit = audit_visual_dataset(config, root=ROOT, output=args.output)
            invocation = "audit_only"
        else:
            result = generate_visual_dataset(config, root=ROOT, output=args.output)
            audit = result["audit"]
            invocation = "generated_and_audited"
        report = build_report(args.config, config, args.output, audit)
        write_exact(args.report, _json_bytes(report))
    except VisualDataError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1
    print(json.dumps({
        "status": "completed",
        "audit": audit["status"],
        "mode": invocation,
        "v0_assets": report["generated"]["stages"]["v0"]["assets"],
        "v0_rows": report["generated"]["stages"]["v0"]["model_rows"],
        "v1_assets": report["generated"]["stages"]["v1"]["assets"],
        "v1_rows": report["generated"]["stages"]["v1"]["model_rows"],
        "temporal_assets": report["generated"]["temporal"]["assets"],
        "temporal_queries": report["generated"]["temporal"]["queries_verifier_only"],
        "internal_asset_bytes": report["generated"]["internal_asset_bytes"],
        "total_generated_bytes": report["generated"]["total_generated_bytes"],
        "admitted_training_rows": 0,
        "external_assets_acquired": 0,
        "source_catalog_sources": report["external_components"]["catalog_sources"],
        "report": str(args.report),
        "samples": report["sample_images"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
