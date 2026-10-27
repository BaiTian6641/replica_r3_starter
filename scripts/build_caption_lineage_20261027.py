#!/usr/bin/env python3
"""Build a new corrected caption-training lineage from the rejected external QA.

The rejected ``external_caption_asset_qa_20261027`` decision remains published
as ``reject``.  This builder creates a *new* child lineage for exactly the nine
rows that the recorded review marked ``match`` inside the train role, with the
original row/target/image hashes, inherited terms and split, duplicate records
and an explicit exclusion ledger for the other twenty-one rows.

No model is loaded, no image is copied and no external source is contacted.  The
builder is deterministic: the same pins always produce the same bytes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.contracts import canonical_json  # noqa: E402
from replica_lab.external_intake import normalize_caption  # noqa: E402

CONFIG_FORMAT = "replica.caption_lineage_correction_config.v1"
MANIFEST_FORMAT = "replica.caption_lineage_manifest.v1"
ROW_FORMAT = "replica.caption_lineage_row.v1"
EXCLUSION_FORMAT = "replica.caption_lineage_exclusion.v1"
SEALED_EXTERNAL_ROLE = "final_eval_untouched"
DEVELOPMENT_ROLE = "development"
TRAIN_ROLE = "train"


class BuildError(ValueError):
    """Pinned evidence does not match the declared lineage contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            _require(isinstance(value, dict), f"{path} contains a non-object row")
            rows.append(value)
    return rows


def verify_pin(pin: Mapping[str, Any], *, project_root: Path) -> Path:
    path = project_root / str(pin["path"])
    _require(path.is_file(), f"pinned file is missing: {pin['path']}")
    _require(sha256_file(path) == pin["sha256"],
             f"sha256 mismatch for {pin['path']}")
    if "bytes" in pin:
        _require(path.stat().st_size == pin["bytes"],
                 f"byte-size mismatch for {pin['path']}")
    return path


def image_aggregate(directory: Path) -> tuple[str, int]:
    files = sorted(path for path in directory.iterdir() if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest(), len(files)


def _image_aggregate_pin(config: Mapping[str, Any], project_root: Path) -> None:
    pin = config["parents"]["image_root"]
    directory = project_root / pin["path"]
    _require(directory.is_dir(), "pinned image directory is missing")
    aggregate, count = image_aggregate(directory)
    _require(aggregate == pin["aggregate_sha256"],
             "pinned image aggregate sha256 mismatch")
    _require(count == pin["files"], "pinned image file count mismatch")


def _row_hash(row: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(row))


def _normalized_caption_sha(caption: str) -> str:
    return sha256_text(normalize_caption(caption))


def build(config: Mapping[str, Any], *, project_root: Path = ROOT) -> dict[str, Any]:
    _require(config.get("format") == CONFIG_FORMAT, "unexpected config format")
    _require(config.get("status") == "corrected_selection_derived_from_rejected_qa",
             "unexpected lineage status")

    pins = config["parents"]
    paths = {name: verify_pin(pin, project_root=project_root)
             for name, pin in pins.items() if isinstance(pin, dict) and "sha256" in pin}
    _image_aggregate_pin(config, project_root)

    tranche = load_json(paths["tranche_manifest"])
    corpus = load_rows(paths["corpus"])
    train_manifest = load_json(paths["train_manifest"])
    development_manifest = load_json(paths["development_manifest"])
    sealed_manifest = load_json(paths["sealed_eval_manifest"])
    decision = load_json(paths["qa_decision"])
    exposure = load_json(paths["exposure_incident"])

    _require(tranche.get("format") == "replica.external_caption_tranche.v1",
             "unexpected tranche manifest format")
    _require(decision.get("decision") == "reject",
             "the parent QA decision must remain 'reject'")
    _require(decision.get("training_admitted_row_ids") is None,
             "the parent QA decision must not have admitted training rows")
    _require(decision.get("qa_id") == config["qa_id"],
             "QA decision id mismatch")
    _require(exposure.get("status") == "quarantined_after_semantic_review_exposure",
             "unexpected exposure incident status")
    exposed_ids = tuple(exposure["exposed_row_ids"])
    _require(tuple(pins["exposure_incident"]["exposed_row_ids"]) == exposed_ids,
             "exposure ids disagree between config and incident receipt")
    _require(len(exposed_ids) == config["selection"]["excluded_counts"]["sealed_exposed"],
             "exposure row count differs from the declared exclusion count")

    verdicts = decision["semantic_reviews"]
    validation_only = set(decision["validation_only_row_ids"])
    train_rows = {row["row_id"]: row for row in train_manifest["rows"]}
    development_rows = {row["row_id"]: row for row in development_manifest["rows"]}
    sealed_rows = {row["row_id"]: row for row in sealed_manifest["rows"]}
    corpus_by_id = {row["row_id"]: row for row in corpus}
    _require(len(corpus_by_id) == len(corpus), "corpus contains duplicate row ids")
    _require(set(corpus_by_id) == set(train_rows) | set(development_rows) | set(sealed_rows),
             "role manifests and corpus do not cover the same rows")
    _require(len(corpus_by_id) == config["selection"]["total_parent_rows_accounted"],
             "corpus row count differs from the declared total")

    admitted: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row_id in sorted(corpus_by_id):
        row = corpus_by_id[row_id]
        role = row["split"]["role"]
        if role == TRAIN_ROLE:
            review = verdicts.get(row_id)
            verdict = review.get("verdict") if review else None
        else:
            verdict = None
        if row_id in exposed_ids:
            reason = "sealed_exposed_quarantine"
        elif role == SEALED_EXTERNAL_ROLE:
            reason = "sealed_role_never_training"
        elif role == DEVELOPMENT_ROLE:
            reason = "validation_only_role"
        elif row_id in validation_only:
            reason = "validation_only_verdict"
        elif verdict == "mismatch":
            reason = "semantic_mismatch"
        elif verdict == "unclear":
            reason = "semantic_unclear"
        elif verdict == "match":
            reason = "admitted"
        else:
            reason = "missing_semantic_verdict"
        if reason == "admitted":
            admitted.append(row)
        else:
            exclusions.append({
                "format": EXCLUSION_FORMAT,
                "lineage_id": config["lineage_id"],
                "row_id": row_id,
                "role": role,
                "recorded_verdict": verdict,
                "reason_code": reason,
                "excluded": True,
                "training_eligible": False,
                "parent_decision": "reject",
                "parent_decision_relabeled": False,
            })

    expected_admitted = tuple(sorted(config["selection"]["admitted_row_ids"]))
    _require(tuple(row["row_id"] for row in admitted) == expected_admitted,
             "selected training rows differ from the config's admitted set")
    _require(len(admitted) == config["selection"]["admitted_count"],
             "admitted row count differs from the config")

    expected_exclusion_ids = (
        set(config["selection"]["excluded_mismatch_row_ids"])
        | set(config["selection"]["excluded_unclear_row_ids"])
        | set(config["selection"]["validation_only_row_ids"])
        | set(exposed_ids)
    )
    _require({item["row_id"] for item in exclusions} == expected_exclusion_ids,
             "exclusion ledger differs from the config's excluded sets")
    _require(set(config["selection"]["validation_only_row_ids"]) == validation_only,
             "validation-only ids differ from the QA decision")
    _require(set(config["selection"]["excluded_mismatch_row_ids"])
             == {row_id for row_id, review in verdicts.items()
                 if review.get("verdict") == "mismatch"},
             "mismatch ids differ from the QA decision")
    _require(set(config["selection"]["excluded_unclear_row_ids"])
             == {row_id for row_id, review in verdicts.items()
                 if review.get("verdict") == "unclear"},
             "unclear ids differ from the QA decision")
    _require(not (set(expected_admitted) & set(exposed_ids)),
             "an exposed row appears in the admitted set")
    _require(not (set(expected_admitted) & validation_only),
             "a validation-only row appears in the admitted set")
    _require(not (set(expected_admitted) & set(sealed_rows)),
             "a sealed row appears in the admitted set")

    rights_policy = config["lineage_policy"]["rights_policy"]
    required_terms = rights_policy["required_terms"]
    duplicate_keys: dict[str, dict[str, list[str]]] = {
        key: defaultdict(list) for key in config["lineage_policy"]["duplicate_checks"]["keys"]
    }
    child_rows: list[dict[str, Any]] = []
    for parent in admitted:
        row_id = parent["row_id"]
        binding = train_rows[row_id]
        upstream_id = parent["provenance"]["upstream_id"]
        image_path = project_root / pins["image_root"]["path"] / \
            f"{upstream_id}.jpg"
        _require(image_path.is_file(), f"image missing for {row_id}")
        image_bytes = image_path.read_bytes()
        image_sha = sha256_bytes(image_bytes)
        _require(image_sha == binding["image_sha256"],
                 f"image sha256 mismatch for {row_id}")
        _require(image_sha == parent["root_asset"]["image_sha256"],
                 f"row image sha256 mismatch for {row_id}")
        _require(len(image_bytes) == parent["root_asset"]["image_bytes"],
                 f"image byte size mismatch for {row_id}")

        caption = parent["target"]["caption"]
        target_sha = sha256_text(caption)
        _require(target_sha == binding["target_sha256"],
                 f"target sha256 mismatch for {row_id}")
        _require(_row_hash(parent) == binding["row_sha256"],
                 f"parent row sha256 mismatch for {row_id}")
        _require(parent["terms"] == required_terms,
                 f"parent terms differ from the required rights for {row_id}")
        review = verdicts[row_id]
        _require(review.get("verdict") == "match",
                 f"missing recorded match verdict for {row_id}")

        child_id = f"{config['lineage_policy']['child_namespace']}:" \
                   f"{upstream_id}#raw_caption"
        child = {
            "format": ROW_FORMAT,
            "lineage_id": config["lineage_id"],
            "row_id": child_id,
            "group_key": f"{config['lineage_policy']['child_namespace']}:{upstream_id}",
            "split": TRAIN_ROLE,
            "role": TRAIN_ROLE,
            "parent": {
                "tranche_row_id": row_id,
                "tranche_row_sha256": binding["row_sha256"],
                "tranche_manifest_sha256": pins["tranche_manifest"]["sha256"],
                "qa_decision_id": decision["qa_id"],
                "qa_decision_sha256": pins["qa_decision"]["sha256"],
                "qa_decision": decision["decision"],
                "qa_decision_relabeled": False,
                "recorded_verdict": review["verdict"],
                "reviewer": review["reviewer"],
                "reviewed_at": review["reviewed_at"],
                "exposure_incident_sha256": pins["exposure_incident"]["sha256"],
                "exposed_row_used": False,
            },
            "source": {
                "id": parent["source"]["id"],
                "revision": parent["source"]["revision"],
                "split": parent["source"]["split"],
                "family": parent["source"]["family"],
                "upstream_id": upstream_id,
                "upstream_image_path": parent["root_asset"]["upstream_image_path"],
                "annotation_row_index": parent["provenance"]["annotation_row_index"],
                "rank_sha256": parent["provenance"]["rank_sha256"],
                "sample_seed": parent["provenance"]["sample_seed"],
            },
            "observation": {
                "image_ref": parent["observation"]["image_ref"],
                "prompt": parent["observation"]["prompt"],
                "task": parent["observation"]["task"],
            },
            "target": {
                "caption": caption,
                "caption_sha256": target_sha,
            },
            "asset": {
                "image_sha256": image_sha,
                "image_bytes": len(image_bytes),
                "image_format": parent["root_asset"]["image_format"],
                "width": parent["root_asset"]["width"],
                "height": parent["root_asset"]["height"],
                "local_path": str(image_path.relative_to(project_root)),
                "copied": False,
            },
            "provenance": {
                "preprocessing_version": parent["provenance"]["preprocessing_version"],
                "caption_source": parent["provenance"]["caption_source"],
                "blip_caption_equals_target": parent["provenance"]["blip_caption_equals_target"],
                "blip_caption_sha256": parent["provenance"]["blip_caption_sha256"],
                "caption_provenance_note": parent["provenance"]["caption_provenance_note"],
                "image_inspection": parent["provenance"]["image_inspection"],
                "normalized_caption_sha256": _normalized_caption_sha(caption),
            },
            "rights": {
                "permitted_use": parent["terms"]["permitted_use"],
                "research_use_only": parent["terms"]["research_use_only"],
                "redistribution": parent["terms"]["redistribution"],
                "commercial_use": False,
                "acknowledgements": list(parent["terms"]["acknowledgements"]),
                "inherited_from_parent": True,
            },
            "inheritance": {
                "parent_role": TRAIN_ROLE,
                "child_role": TRAIN_ROLE,
                "policy": config["lineage_policy"]["split_inheritance"]["policy"],
                "reassigned": False,
            },
        }
        child["row_sha256"] = _row_hash(child)
        child_rows.append(child)

        duplicate_keys["row_sha256"][child["row_sha256"]].append(child_id)
        duplicate_keys["target_sha256"][target_sha].append(child_id)
        duplicate_keys["image_sha256"][image_sha].append(child_id)
        duplicate_keys["normalized_caption_sha256"][
            child["provenance"]["normalized_caption_sha256"]].append(child_id)
        duplicate_keys["root_group_key"][row_id.split("#")[0]].append(child_id)

    collisions = []
    for key, groups in duplicate_keys.items():
        for value, members in groups.items():
            if len(members) > 1:
                collisions.append({"key": key, "value": value, "members": sorted(members)})
    _require(not collisions,
             f"admitted lineage contains duplicate groups: {collisions}")
    _require(len({row["row_sha256"] for row in child_rows}) == len(child_rows),
             "child row hashes are not unique")
    _require(len({row["row_id"] for row in child_rows}) == len(child_rows),
             "child row ids are not unique")

    rows_sha = sha256_text("".join(canonical_json(row) + "\n" for row in child_rows))
    exclusions_sha = sha256_text("".join(canonical_json(row) + "\n" for row in exclusions))
    output_paths = config["outputs"]
    manifest = {
        "format": MANIFEST_FORMAT,
        "lineage_id": config["lineage_id"],
        "created_on": config["created_on"],
        "status": config["status"],
        "generator": "scripts/build_caption_lineage_20261027.py",
        "generator_version": "1.0.0",
        "parents": {
            name: {"path": pin["path"], "sha256": pin["sha256"]}
            for name, pin in pins.items() if "sha256" in pin
        },
        "selection": {
            "rule": config["selection"]["rule"],
            "admitted_row_ids": [row["row_id"] for row in child_rows],
            "excluded_counts": config["selection"]["excluded_counts"],
            "total_parent_rows_accounted": config["selection"]["total_parent_rows_accounted"],
        },
        "policy": config["lineage_policy"],
        "counts": {
            "rows": len(child_rows),
            "root_groups": len({row["group_key"] for row in child_rows}),
            "exclusions": len(exclusions),
            "train": len(child_rows),
        },
        "rows": [
            {
                "row_id": row["row_id"],
                "group_key": row["group_key"],
                "parent_row_id": row["parent"]["tranche_row_id"],
                "row_sha256": row["row_sha256"],
                "image_sha256": row["asset"]["image_sha256"],
                "target_sha256": row["target"]["caption_sha256"],
                "normalized_caption_sha256": row["provenance"]["normalized_caption_sha256"],
                "local_path": row["asset"]["local_path"],
                "role": TRAIN_ROLE,
            }
            for row in child_rows
        ],
        "rows_sha256": rows_sha,
        "exclusions": exclusions,
        "exclusions_sha256": exclusions_sha,
        "duplicate_groups": {
            "rule": "all key groups are unique inside the admitted lineage",
            "admitted_collisions": [],
            "keys_checked": config["lineage_policy"]["duplicate_checks"]["keys"],
        },
        "rights": {
            "required_terms": config["lineage_policy"]["rights_policy"]["required_terms"],
            "required_acknowledgements":
                config["lineage_policy"]["rights_policy"]["required_acknowledgements"],
            "redistribution": "prohibited",
        },
        "eval_partition": config["eval_partition"],
        "admission": config["admission"],
        "non_claims": config["non_claims"],
        "outputs": {
            "lineage_manifest": output_paths["lineage_manifest"],
            "rows": output_paths["rows"],
            "exclusions": output_paths["exclusions"],
        },
    }
    _require(manifest["counts"]["rows"] == 9, "exactly nine rows are required")
    manifest["manifest_sha256"] = _row_hash({k: v for k, v in manifest.items()
                                             if k != "manifest_sha256"})
    return {
        "manifest": manifest,
        "rows": child_rows,
        "exclusions": exclusions,
        "rows_text": "".join(canonical_json(row) + "\n" for row in child_rows),
        "exclusions_text": "".join(canonical_json(row) + "\n" for row in exclusions),
        "manifest_text": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    }


def write_outputs(built: Mapping[str, Any], config: Mapping[str, Any],
                  *, project_root: Path = ROOT, force: bool = False) -> dict[str, str]:
    outputs = config["outputs"]
    root = project_root / outputs["root"]
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        outputs["lineage_manifest"]: built["manifest_text"],
        outputs["rows"]: built["rows_text"],
        outputs["exclusions"]: built["exclusions_text"],
    }
    for relative, text in plan.items():
        target = project_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding="utf-8") != text and not force:
            raise BuildError(
                f"refusing to overwrite different existing output {relative}; "
                "re-run with --force only for an intentional rebuild")
        target.write_text(text, encoding="utf-8")
    return {
        "lineage_manifest": str(root / "lineage_manifest.json"),
        "rows": str(root / "rows.jsonl"),
        "exclusions": str(root / "exclusions.jsonl"),
        "rows_sha256": built["manifest"]["rows_sha256"],
        "manifest_sha256": built["manifest"]["manifest_sha256"],
        "exclusions_sha256": built["manifest"]["exclusions_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(
        ROOT / "configs/caption_lineage_correction_20261027.json"))
    parser.add_argument("--force", action="store_true",
                        help="replace a differing existing output (intentional rebuild)")
    args = parser.parse_args(argv)
    config = load_json(args.config)
    _require(config.get("format") == CONFIG_FORMAT, "unexpected config format")
    try:
        built = build(config)
        summary = write_outputs(built, config, force=args.force)
    except BuildError as error:
        print(json.dumps({"status": "refused", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "built", **summary,
                      "rows": built["manifest"]["counts"]["rows"],
                      "exclusions": built["manifest"]["counts"]["exclusions"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
