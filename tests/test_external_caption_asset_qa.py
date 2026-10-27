"""Deterministic CPU tests for the external caption asset QA entry point.

The tests build a tiny synthetic tranche (PNG bytes, four rows) and never touch
the network or the real acquisition.  Pillow is not required: the QA script's
decoder is injected, so decode/geometry failure paths always run.  One optional
integration test re-runs the real pinned QA when Pillow and the ignored raw
images are present, and skips with an explicit provisioning reason otherwise.
"""
from __future__ import annotations

import copy
from collections import Counter
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zlib

import pytest

from conftest import require_local_artifacts
from replica_lab.external_intake import (
    _role_manifest,
    _row_document,
    inspect_image_bytes,
    leakage_report,
    normalize_caption,
    normalize_prompt,
)
from replica_lab.vision_qualification import canonical_json, sha256_text

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qa_external_caption_tranche.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qa = _load(SCRIPT, "qa_external_caption_tranche_test")

ISO = "2026-10-27T00:00:00Z"
ROLE_ORDER = {"train": 0, "development": 1, "final_eval_untouched": 2}
ROLES = ("train", "development", "final_eval_untouched")


def _sha(data: bytes) -> str:
    return qa.sha256_bytes(data)


def _json_text(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def make_png(index: int = 0, width: int = 16, height: int = 16) -> bytes:
    """A valid RGB PNG whose bytes differ per index; no Pillow dependency."""
    color = bytes(
        (
            (index * 37 + 11) % 256,
            (index * 59 + 7) % 256,
            (index * 83 + 3) % 256,
        )
    )
    scanlines = b"".join(b"\x00" + color * width for _ in range(height))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def fake_decoder(data: bytes) -> dict:
    return {
        "verified": True,
        "decoder": "fake-png-decoder",
        "format": "PNG",
        "mode": "RGB",
        "width": 16,
        "height": 16,
        "channels": 3,
    }


def wrong_geometry_decoder(data: bytes) -> dict:
    return {
        "verified": True,
        "decoder": "fake-png-decoder",
        "format": "PNG",
        "mode": "RGB",
        "width": 15,
        "height": 16,
        "channels": 3,
    }


def failing_decoder(data: bytes) -> dict:
    raise qa.DecodeError("image.decode", "fixture image is truncated")


def unavailable_decoder(data: bytes) -> dict:
    raise qa.DecodeUnavailable(
        "image.decode", "pillow_unavailable: fixture", blocked=True
    )


def build_fixture(tmp_path: Path) -> dict:
    root = tmp_path
    data_root = root / "data" / "fixture_tranche"
    raw = data_root / "raw"
    images_dir = raw / "images"
    manifests_dir = data_root / "manifests"
    config_dir = root / "configs"
    for path in (images_dir, manifests_dir, config_dir):
        path.mkdir(parents=True, exist_ok=True)

    dataset_id = "fixture/dataset"
    seed = 7
    roles = ["train", "train", "development", "final_eval_untouched"]
    prompts = [
        "Describe the image.\n<image>",
        "<image>\nWhat does this show?",
        "Write a short caption for this picture.\n<image>",
        "Describe the image.\n<image>",
    ]
    captions = [f"fixture caption {index} on a plain canvas" for index in range(4)]

    image_bytes_by_id: dict[str, bytes] = {}
    annotations = []
    metadata = []
    for index in range(4):
        identifier = f"{index + 1:09d}"
        image_path = f"{index + 1:05d}/{identifier}.jpg"
        png = make_png(index)
        image_bytes_by_id[identifier] = png
        (images_dir / f"{identifier}.jpg").write_bytes(png)
        annotations.append(
            {
                "id": identifier,
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": prompts[index]},
                    {"from": "gpt", "value": captions[index]},
                ],
            }
        )
        metadata.append(
            {
                "id": identifier,
                "image": image_path,
                "blip_caption": captions[index],
                "url": f"https://images.example.test/{identifier}.jpg",
            }
        )

    annotation_bytes = json.dumps(annotations, ensure_ascii=False).encode("utf-8")
    metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    card_bytes = b"fixture dataset card\n"
    central_directory = b"fixture-central-directory"
    (raw / "annotation_prefix.json").write_bytes(annotation_bytes)
    (raw / "metadata_prefix.json").write_bytes(metadata_bytes)
    (raw / "card.md").write_bytes(card_bytes)
    (raw / "images_central_directory.bin").write_bytes(central_directory)

    image_constraints = {
        "formats": ["PNG"],
        "min_dimension": 4,
        "max_dimension": 64,
        "max_bytes": 65536,
        "require_extension_match": False,
    }
    terms = {
        "license_expression": "fixture research-only terms",
        "permitted_use": "internal_research",
        "research_use_only": True,
        "redistribution": "prohibited",
        "commercial_use": False,
        "acknowledgements": ["Fixture Source"],
        "components": [
            {"name": "fixture image", "terms": "research only", "redistributable": False,
             "used": "image"},
            {"name": "fixture caption", "terms": "fixture text", "redistributable": True,
             "used": "caption"},
        ],
        "reserved_family_avoidance": {
            "reserved": ["fixture_reserved_family"],
            "selected_family": "fixture_family",
            "note": "fixture only",
        },
    }
    intake_config = {
        "format": "replica.external_caption_intake_config.v1",
        "intake_id": "fixture_tranche",
        "created_on": "2026-10-27",
        "dataset": {
            "id": dataset_id,
            "family": "fixture_family",
            "source_url": "https://example.test/dataset",
            "revision": "a" * 40,
            "split": "train",
            "shard": "fixture-window",
            "card": {
                "path": "README.md",
                "size": len(card_bytes),
                "sha256": _sha(card_bytes),
            },
            "files": {
                "annotation": {
                    "path": "annotations.json",
                    "size": len(annotation_bytes),
                    "sha256": _sha(annotation_bytes),
                },
                "metadata": {
                    "path": "metadata.json",
                    "size": len(metadata_bytes),
                    "sha256": _sha(metadata_bytes),
                },
                "images": {
                    "path": "images.zip",
                    "size": len(central_directory),
                    "sha256": _sha(central_directory),
                },
            },
            "composition_note": "fixture composition",
        },
        "terms": terms,
        "sampling": {
            "seed": seed,
            "order": "ascending fixture rank over the synthetic window",
            "window_rows": 4,
            "window_bytes": 1024,
            "max_window_bytes": 4096,
            "target_roots": 4,
            "split_quotas": {"train": 2, "development": 1, "final_eval_untouched": 1},
        },
        "preprocessing": {
            "version": "external-caption-normalization-v1",
            "prompt_normalization": "collapse_whitespace_remove_image_token",
            "caption_normalization": "collapse_whitespace_strip_image_token",
            "min_caption_chars": 3,
            "max_caption_chars": 300,
            "image": image_constraints,
        },
        "admission": {
            "forbidden_url_patterns": ["(?<![a-z])coco(?![a-z])", "mscoco"],
            "max_central_directory_bytes": 67108864,
            "max_acquired_bytes": 1048576,
            "campaign_disk_ceiling_bytes": 1073741824,
        },
    }
    intake_text = _json_text(intake_config)
    intake_path = config_dir / "external_caption_tranche_v1.json"
    intake_path.write_text(intake_text, encoding="utf-8")

    rows = []
    for index, role in enumerate(roles):
        identifier = f"{index + 1:09d}"
        image_path = f"{index + 1:05d}/{identifier}.jpg"
        inspection = inspect_image_bytes(
            image_bytes_by_id[identifier], image_constraints, member_name=image_path
        )
        inspection["crc32"] = (
            f"{zlib.crc32(image_bytes_by_id[identifier]) & 0xFFFFFFFF:08x}"
        )
        candidate = {
            "annotation_index": index,
            "root_image_id": f"{dataset_id}:{identifier}",
            "upstream_id": identifier,
            "upstream_image_path": image_path,
            "upstream_url": f"https://images.example.test/{identifier}.jpg",
            "prompt": normalize_prompt(prompts[index]),
            "caption": normalize_caption(captions[index]),
            "blip_caption_sha256": sha256_text(captions[index]),
            "blip_caption_equals_target": True,
            "rank": sha256_text(f"{seed}\x00{identifier}"),
        }
        rows.append(_row_document(candidate, inspection, role=role, config=intake_config))

    rows.sort(
        key=lambda row: (
            ROLE_ORDER[row["split"]["role"]],
            row["split"]["assignment_rank"],
            row["row_id"],
        )
    )

    corpus_text = "".join(canonical_json(row) + "\n" for row in rows)
    (data_root / "corpus.jsonl").write_text(corpus_text, encoding="utf-8")
    (data_root / "rejections.jsonl").write_bytes(b"")

    role_manifests = {}
    role_manifest_paths = {}
    role_manifest_texts = {}
    for role in ROLES:
        role_manifest = _role_manifest(rows, role, intake_config)
        text = _json_text(role_manifest)
        path = manifests_dir / f"{role}.json"
        path.write_text(text, encoding="utf-8")
        role_manifests[role] = role_manifest
        role_manifest_paths[role] = path
        role_manifest_texts[role] = text

    members = [
        {
            "zip_member": row["root_asset"]["upstream_image_path"],
            "local_path": f"raw/images/{row['provenance']['upstream_id']}.jpg",
            "sha256": row["root_asset"]["image_sha256"],
            "crc32": row["root_asset"]["zip_member_crc32"],
            "bytes": row["root_asset"]["image_bytes"],
        }
        for row in rows
    ]
    assets = [
        {
            "root_image_id": row["root_asset"]["root_image_id"],
            "role": row["split"]["role"],
            "group_key": row["split"]["group_key"],
            "upstream_image_path": row["root_asset"]["upstream_image_path"],
            "upstream_url": row["root_asset"]["upstream_url"],
            "zip_member": row["root_asset"]["zip_member"],
            "zip_member_crc32": row["root_asset"]["zip_member_crc32"],
            "image_sha256": row["root_asset"]["image_sha256"],
            "image_bytes": row["root_asset"]["image_bytes"],
            "format": row["root_asset"]["image_format"],
            "width": row["root_asset"]["width"],
            "height": row["root_asset"]["height"],
            "row_id": row["row_id"],
            "row_sha256": sha256_text(canonical_json(row)),
        }
        for row in rows
    ]
    role_counts = {
        role: {
            "rows": sum(1 for row in rows if row["split"]["role"] == role),
            "root_groups": len(
                {
                    row["split"]["group_key"]
                    for row in rows
                    if row["split"]["role"] == role
                }
            ),
        }
        for role in ROLES
    }
    counts = {
        "window_rows": 4,
        "metadata_rows": 4,
        "metadata_candidates": 4,
        "inspected_assets": 4,
        "admitted_rows": 4,
        "admitted_root_assets": 4,
        "normalized_rows": 4,
        "rejected_rows": 0,
        "roles": role_counts,
        "rejection_reasons": {},
    }
    inspection_summary = {
        "assets": 4,
        "decode_verified": sum(
            1
            for row in rows
            if row["provenance"]["image_inspection"]["decode_verified"] is True
        ),
        "structural_verified": sum(
            1
            for row in rows
            if row["provenance"]["image_inspection"]["structural_verified"] is True
        ),
        "decoders": dict(
            sorted(
                Counter(
                    row["provenance"]["image_inspection"]["decoder"] for row in rows
                ).items()
            )
        ),
        "formats": dict(
            sorted(Counter(row["root_asset"]["image_format"] for row in rows).items())
        ),
        "image_bytes": sum(row["root_asset"]["image_bytes"] for row in rows),
    }
    rejection_sha = _sha(b"")
    artifacts = {
        "corpus": {
            "path": "corpus.jsonl",
            "sha256": _sha(corpus_text.encode("utf-8")),
            "rows": 4,
        },
        "rejections": {
            "path": "rejections.jsonl",
            "sha256": rejection_sha,
            "rows": 0,
        },
        "manifests": {
            role: {
                "path": f"manifests/{role}.json",
                "sha256": _sha(role_manifest_texts[role].encode("utf-8")),
                "rows": len(role_manifests[role]["rows"]),
                "sealed": role_manifests[role]["sealed"],
                "seal_sha256": role_manifests[role].get("seal_sha256"),
            }
            for role in ROLES
        },
    }
    acquisition = {
        "acquired_bytes": (
            len(card_bytes)
            + len(annotation_bytes)
            + len(metadata_bytes)
            + len(central_directory)
            + sum(row["root_asset"]["image_bytes"] for row in rows)
        ),
        "budget_bytes": intake_config["admission"]["max_acquired_bytes"],
        "campaign_disk_ceiling_bytes": intake_config["admission"][
            "campaign_disk_ceiling_bytes"
        ],
        "within_budget": True,
        "read_count": 4,
        "window_alignment": {
            "annotation_rows": 4,
            "metadata_rows": 4,
            "shared_ids": 4,
            "annotation_only_ids": 0,
            "metadata_only_ids": 0,
            "method": "fixture rows joined by id",
        },
        "sources": {
            "card": {
                "bytes": len(card_bytes),
                "sha256": _sha(card_bytes),
                "full_sha256_verified": True,
            },
            "annotation": {
                "acquired_prefix_bytes": len(annotation_bytes),
                "acquired_prefix_sha256": _sha(annotation_bytes),
                "complete_rows_in_window": 4,
                "bytes_in_window_sha256": sha256_text(canonical_json(annotations)),
                "declared_full_sha256": _sha(annotation_bytes),
                "full_sha256_verified": False,
            },
            "metadata": {
                "acquired_prefix_bytes": len(metadata_bytes),
                "acquired_prefix_sha256": _sha(metadata_bytes),
                "complete_rows_in_window": 4,
                "bytes_in_window_sha256": sha256_text(canonical_json(metadata)),
                "declared_full_sha256": _sha(metadata_bytes),
                "full_sha256_verified": False,
            },
            "images_zip": {
                "central_directory_bytes": len(central_directory),
                "central_directory_offset": 0,
                "central_directory_sha256": _sha(central_directory),
                "central_directory_records": 4,
                "file_entries": 4,
                "declared_full_sha256": _sha(central_directory),
                "declared_size": len(central_directory),
                "full_sha256_verified": False,
                "members_verified": 4,
                "members": members,
            },
        },
    }
    manifest = {
        "format": "replica.external_caption_tranche.v1",
        "intake_id": intake_config["intake_id"],
        "status": "admitted",
        "created_on": "2026-10-27",
        "config": {
            "path": "configs/external_caption_tranche_v1.json",
            "sha256": _sha(intake_text.encode("utf-8")),
        },
        "dataset": intake_config["dataset"],
        "terms": terms,
        "sampling": {
            key: intake_config["sampling"][key]
            for key in (
                "seed",
                "order",
                "window_bytes",
                "max_window_bytes",
                "window_rows",
                "target_roots",
                "split_quotas",
            )
        },
        "preprocessing": intake_config["preprocessing"],
        "counts": counts,
        "exclusions": {
            "window": {
                "rows": 4,
                "bytes": len(annotation_bytes),
                "scope": intake_config["dataset"]["shard"],
            },
            "rows_outside_window": {"inspected": False, "reason": "fixture"},
            "metadata_candidates": 4,
            "downloaded_assets": 4,
            "admitted_assets": 4,
            "candidates_beyond_quota": 0,
            "rejected": {
                "rows": 0,
                "metadata": 0,
                "content": 0,
                "reasons": {},
                "ledger": artifacts["rejections"],
            },
        },
        "inspection": inspection_summary,
        "reserved_family_check": {
            "patterns": intake_config["admission"]["forbidden_url_patterns"],
            "admitted_asset_matches": 0,
            "reserved_families": terms["reserved_family_avoidance"]["reserved"],
        },
        "acquisition": acquisition,
        "leakage": leakage_report(rows),
        "artifacts": artifacts,
        "assets": assets,
    }
    manifest_text = _json_text(manifest)
    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    image_files, image_total, image_aggregate = qa.image_aggregate(images_dir)
    per_image = {
        path.name: {"sha256": _sha(path.read_bytes()), "bytes": path.stat().st_size}
        for path in image_files
    }
    expected = {
        "rows": 4,
        "root_groups": 4,
        "roles": {"train": 2, "development": 1, "final_eval_untouched": 1},
        "training_role": "train",
        "validation_roles": ["development"],
        "sealed_role": "final_eval_untouched",
        "annotation_rows": 4,
        "metadata_rows": 4,
        "metadata_candidates": 4,
        "rejections_rows": 0,
        "preprocessing_version": intake_config["preprocessing"]["version"],
        "sampling_seed": seed,
        "sampling_order": intake_config["sampling"]["order"],
        "split_policy": "complete root asset and all derivatives share one split",
        "image_constraints": image_constraints,
        "row_terms": {
            "acknowledgements": terms["acknowledgements"],
            "permitted_use": terms["permitted_use"],
            "redistribution": terms["redistribution"],
            "research_use_only": terms["research_use_only"],
        },
        "terms": {
            "permitted_use": terms["permitted_use"],
            "research_use_only": terms["research_use_only"],
            "redistribution": terms["redistribution"],
            "commercial_use": terms["commercial_use"],
            "license_expression": terms["license_expression"],
            "acknowledgements": terms["acknowledgements"],
        },
        "reserved_patterns": intake_config["admission"]["forbidden_url_patterns"],
        "reserved_families": terms["reserved_family_avoidance"]["reserved"],
        "leakage": leakage_report(rows),
        "blip_caption_equals_target": True,
        "caption_min_chars": 3,
        "caption_max_chars": 300,
        "annotation_source_name": "annotations.json",
        "metadata_source_name": "metadata.json",
    }
    qa_config = {
        "format": qa.QA_CONFIG_FORMAT,
        "qa_id": "fixture_asset_qa",
        "created_on": "2026-10-27",
        "data_root": _rel(root, data_root),
        "report_dir": _rel(root, root / "reports" / "qa"),
        "source": {
            "id": intake_config["dataset"]["id"],
            "family": intake_config["dataset"]["family"],
            "revision": intake_config["dataset"]["revision"],
            "split": intake_config["dataset"]["split"],
            "shard": intake_config["dataset"]["shard"],
            "source_url": intake_config["dataset"]["source_url"],
        },
        "pins": {
            "intake_config": {
                "path": _rel(root, intake_path),
                "sha256": _sha(intake_text.encode("utf-8")),
                "bytes": len(intake_text.encode("utf-8")),
            },
            "tranche_manifest": {
                "path": _rel(root, manifest_path),
                "sha256": _sha(manifest_text.encode("utf-8")),
                "bytes": len(manifest_text.encode("utf-8")),
            },
            "card": {
                "path": _rel(root, raw / "card.md"),
                "sha256": _sha(card_bytes),
                "bytes": len(card_bytes),
            },
            "annotation": {
                "path": _rel(root, raw / "annotation_prefix.json"),
                "sha256": _sha(annotation_bytes),
                "bytes": len(annotation_bytes),
                "complete_rows": 4,
                "window_sha256": sha256_text(canonical_json(annotations)),
                "declared_full_sha256": _sha(annotation_bytes),
            },
            "metadata": {
                "path": _rel(root, raw / "metadata_prefix.json"),
                "sha256": _sha(metadata_bytes),
                "bytes": len(metadata_bytes),
                "complete_rows": 4,
                "window_sha256": sha256_text(canonical_json(metadata)),
                "declared_full_sha256": _sha(metadata_bytes),
            },
            "media": {
                "central_directory": {
                    "path": _rel(root, raw / "images_central_directory.bin"),
                    "sha256": _sha(central_directory),
                    "bytes": len(central_directory),
                    "records": 4,
                    "file_entries": 4,
                    "offset": 0,
                    "declared_full_sha256": _sha(central_directory),
                    "declared_size": len(central_directory),
                },
                "images": {
                    "dir": _rel(root, images_dir),
                    "count": 4,
                    "total_bytes": image_total,
                    "aggregate_sha256": image_aggregate,
                    "aggregate_method": "sha256(concat(sorted filename + NUL + bytes))",
                    "per_image": per_image,
                },
            },
            "corpus": {
                "path": _rel(root, data_root / "corpus.jsonl"),
                "sha256": _sha(corpus_text.encode("utf-8")),
                "bytes": len(corpus_text.encode("utf-8")),
                "rows": 4,
            },
            "rejections": {
                "path": _rel(root, data_root / "rejections.jsonl"),
                "sha256": rejection_sha,
                "bytes": 0,
                "rows": 0,
            },
            "role_manifests": {
                role: {
                    "path": _rel(root, role_manifest_paths[role]),
                    "sha256": artifacts["manifests"][role]["sha256"],
                    "bytes": len(role_manifest_texts[role].encode("utf-8")),
                    "rows": len(role_manifests[role]["rows"]),
                    "sealed": role_manifests[role]["sealed"],
                    "seal_sha256": role_manifests[role].get("seal_sha256"),
                }
                for role in ROLES
            },
        },
        "expected": expected,
        "semantic_review": {
            "required": True,
            "reviewable_sections": ["training_candidates", "validation_only"],
            "reviewable_queue_entries": 3,
            "sealed_queue_entries": 1,
            "sealed_semantic_review": "prohibited",
            "required_verdicts": ["match", "mismatch", "unclear"],
            "training_candidate_verdict": "match",
            "decision_file": "admission_decision.json",
            "template_file": "admission_decision.template.json",
            "allowed_decisions": [
                qa.PENDING,
                qa.REJECT,
                qa.ADMIT,
            ],
            "require_all_nonsealed_queue_entries_reviewed": True,
        },
        "non_claims": {
            "semantic_caption_image_agreement": False,
            "training_admission_from_machine_checks_alone": False,
        },
    }
    qa_config_text = _json_text(qa_config)
    qa_config_path = config_dir / "external_caption_asset_qa_v1.json"
    qa_config_path.write_text(qa_config_text, encoding="utf-8")

    return {
        "root": root,
        "data_root": data_root,
        "report_dir": root / "reports" / "qa",
        "config_path": qa_config_path,
        "qa_config": qa_config,
        "intake_config": intake_config,
        "rows": rows,
        "manifest": manifest,
        "role_manifests": role_manifests,
        "role_manifest_paths": role_manifest_paths,
        "images_dir": images_dir,
    }


def _write_qa_config(state: dict, mutate=None) -> None:
    config = copy.deepcopy(state["qa_config"])
    if mutate is not None:
        mutate(config)
    state["config_path"].write_text(_json_text(config), encoding="utf-8")


def _run(state: dict, *, decoder=fake_decoder, write: bool = False, decision_path=None):
    return qa.run_qa(
        project_root=state["root"],
        config_path=state["config_path"],
        report_dir=state["report_dir"],
        decision_path=decision_path,
        decoder=decoder,
        write=write,
    )


def _failed_ids(report: dict) -> set[str]:
    return {check["id"] for check in report["machine_checks"]["failed_checks"]}


def _prepare_reports(state: dict):
    report = _run(state, write=True)
    template = json.loads(
        (state["report_dir"] / "admission_decision.template.json").read_text(
            encoding="utf-8"
        )
    )
    return report, template


def _complete_reviews(template: dict, verdict: str = "match") -> dict:
    return {
        row_id: {
            "verdict": verdict,
            "reviewer": "main",
            "reviewed_at": ISO,
            "notes": None,
        }
        for row_id in template["semantic_reviews"]
    }


def _write_decision(state: dict, decision: dict) -> Path:
    path = state["report_dir"] / "admission_decision.json"
    path.write_text(_json_text(decision), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# happy path and queue/seal separation
# ---------------------------------------------------------------------------


def test_fixture_machine_checks_pass_and_queue_is_separated(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    report, template = _prepare_reports(state)

    assert report["status"] == "machine_checks_passed_semantic_review_pending"
    assert report["machine_checks"]["passed"] is True
    assert report["machine_checks"]["failed"] == 0
    assert report["machine_checks"]["blocked"] == 0
    assert report["admission_checks"]["passed"] is True
    assert report["admission"]["decision"] == qa.PENDING
    assert report["admission"]["training_admitted"] is False
    assert report["admission"]["training_admitted_row_ids"] == []
    assert report["admission"]["refusal_reasons"] == []
    assert report["decode"]["verified"] == 4
    assert report["decode"]["failed"] == 0
    assert all(not values for values in report["duplicates"].values())
    assert report["leakage"]["label_separation_verified"] is True
    assert report["reserved_family"]["admitted_asset_matches"] == 0
    assert report["seals"]["final_eval_untouched"]["sealed"] is True
    assert report["seals"]["final_eval_untouched"]["untouched"] is True
    assert (
        report["seals"]["final_eval_untouched"]["seal_sha256"]
        == report["seals"]["final_eval_untouched"]["recomputed_seal_sha256"]
    )
    assert report["scope_separation"]["training_output_contains_only_training_role"] is None
    assert report["claims"]["semantic_caption_image_agreement_asserted_by_qa"] is False
    assert report["claims"]["training_admission_claimed"] is False

    queue = json.loads(
        (state["report_dir"] / "review_queue.json").read_text(encoding="utf-8")
    )
    assert queue["counts"] == {
        "total": 4,
        "reviewable": 3,
        "sealed": 1,
        "sections": {
            "training_candidates": 2,
            "validation_only": 1,
            "sealed_final_eval": 1,
        },
    }
    assert queue["sealed_semantic_review"] == "prohibited"
    assert queue["sealed_target_text_redacted"] is True
    assert all(
        entry["training_admission_eligible"] is False
        for entry in queue["sections"]["sealed_final_eval"]
    )
    assert all(
        entry["training_admission_eligible"] is True
        for entry in queue["sections"]["training_candidates"]
    )
    assert all(
        entry["admission_scope"] == "validation_only"
        for entry in queue["sections"]["validation_only"]
    )
    assert all(
        entry["machine_checks"]["passed"] is True
        for entry in queue["sections"]["training_candidates"]
    )
    assert all(
        entry["image"]["decode"]["verified"] is True
        for entry in queue["sections"]["training_candidates"]
    )
    for section_name in ("training_candidates", "validation_only"):
        for entry in queue["sections"][section_name]:
            assert entry["semantic_review"]["required"] is True
            assert entry["semantic_review"]["status"] == "pending"
            assert entry["semantic_review"]["verdict"] is None
            assert entry["caption"] is not None
            assert entry["target_text_redacted"] is False

    sealed_entry = queue["sections"]["sealed_final_eval"][0]
    assert sealed_entry["semantic_review"]["required"] is False
    assert sealed_entry["semantic_review"]["status"] == "prohibited_sealed"
    assert sealed_entry["semantic_review"]["verdict"] is None
    assert sealed_entry["caption"] is None
    assert sealed_entry["prompt"] is None
    assert sealed_entry["provenance"]["upstream_prompt"] is None
    assert sealed_entry["target_text_redacted"] is True
    assert sealed_entry["redaction"]["target_sha256_retained"] is True
    assert sealed_entry["target_sha256"]
    assert sealed_entry["image"]["decode"]["verified"] is True
    assert sealed_entry["machine_checks"]["passed"] is True
    assert sealed_entry["sealed"] is True

    sealed_row = next(
        row for row in state["rows"] if row["row_id"] == sealed_entry["row_id"]
    )
    queue_text = (state["report_dir"] / "review_queue.json").read_text(
        encoding="utf-8"
    )
    assert sealed_row["target"]["caption"] not in queue_text

    assert len(template["semantic_reviews"]) == 3
    assert set(template["semantic_reviews"]) == set(template["reviewable_row_ids"])
    assert not (
        set(template["sealed_row_ids"]) & set(template["semantic_reviews"])
    )
    assert template["sealed_semantic_review"] == "prohibited"
    assert all(
        review["verdict"] is None for review in template["semantic_reviews"].values()
    )


# ---------------------------------------------------------------------------
# drift, decode and geometry failures
# ---------------------------------------------------------------------------


def test_config_pin_drift_is_detected(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)

    def mutate(config: dict) -> None:
        config["pins"]["corpus"]["sha256"] = "0" * 64

    _write_qa_config(state, mutate)
    report = _run(state)

    assert report["machine_checks"]["passed"] is False
    assert "pin.corpus.sha256" in _failed_ids(report)
    assert report["admission"]["training_admitted"] is False


def test_image_byte_drift_is_detected(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    row = state["rows"][0]
    image_path = state["data_root"] / "raw" / row["observation"]["image_ref"]
    image_path.write_bytes(make_png(99))

    report = _run(state)
    failed = _failed_ids(report)

    assert f"pin.media.image.{image_path.name}.sha256" in failed
    assert "row.image.sha256" in failed
    assert "row.image.pinned_sha256" in failed
    assert report["decode"]["verified"] == 4  # replacement is still a valid PNG


def test_annotation_and_metadata_provenance_drift_is_detected(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    raw = state["data_root"] / "raw"
    annotations = json.loads((raw / "annotation_prefix.json").read_text(encoding="utf-8"))
    metadata = json.loads((raw / "metadata_prefix.json").read_text(encoding="utf-8"))
    annotations[0]["conversations"][0]["value"] = "Describe a different image.\n<image>"
    annotations[0]["conversations"][1]["value"] = "a different caption"
    metadata[0]["blip_caption"] = "a different caption"
    (raw / "annotation_prefix.json").write_text(
        json.dumps(annotations, ensure_ascii=False), encoding="utf-8"
    )
    (raw / "metadata_prefix.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )

    report = _run(state)
    failed = _failed_ids(report)

    assert "pin.annotation.sha256" in failed
    assert "pin.metadata.sha256" in failed
    assert "row.provenance.prompt" in failed
    assert "row.provenance.caption" in failed
    assert "row.provenance.blip_sha256" in failed
    assert "row.provenance.blip_target" in failed
    assert report["admission"]["training_admitted"] is False


def test_decode_failure_is_detected(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    report = _run(state, decoder=failing_decoder)

    assert report["machine_checks"]["passed"] is False
    assert "row.image.decode" in _failed_ids(report)
    assert report["decode"]["verified"] == 0
    assert report["decode"]["failed"] == 4
    assert report["admission"]["training_admitted"] is False


def test_geometry_mismatch_is_detected(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    report = _run(state, decoder=wrong_geometry_decoder)

    assert report["machine_checks"]["passed"] is False
    assert "row.image.decode.dimensions" in _failed_ids(report)
    assert report["admission"]["training_admitted"] is False


def test_missing_pillow_is_blocked_not_silent(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    report = _run(state, decoder=unavailable_decoder)

    assert report["machine_checks"]["passed"] is False
    blocked = [
        check
        for check in report["machine_checks"]["failed_checks"]
        if check["id"] == "row.image.decode"
    ]
    assert len(blocked) == 4
    assert all(check["blocked"] is True for check in blocked)
    assert report["machine_checks"]["blocked"] == 4
    assert report["admission"]["training_admitted"] is False


# ---------------------------------------------------------------------------
# duplicate, leakage, reserved-family and seal violations
# ---------------------------------------------------------------------------


def test_duplicate_image_and_caption_groups_are_rejected(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    rows = copy.deepcopy(state["rows"])
    rows[1]["target"]["caption"] = rows[0]["target"]["caption"]
    rows[1]["root_asset"]["image_sha256"] = rows[0]["root_asset"]["image_sha256"]
    rows[1]["split"]["group_key"] = rows[0]["split"]["group_key"]
    rows[1]["root_asset"]["upstream_url"] = rows[0]["root_asset"]["upstream_url"]

    duplicates = qa.find_duplicates(rows)
    assert duplicates["caption"]
    assert duplicates["image_sha256"]
    assert duplicates["root_group"]
    assert duplicates["upstream_url"]
    with pytest.raises(qa.QAError):
        qa.assert_no_duplicates(rows)


def test_cross_split_and_label_leakage_is_rejected(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    rows = copy.deepcopy(state["rows"])
    train_row = next(row for row in rows if row["split"]["role"] == "train")
    development_row = next(
        row for row in rows if row["split"]["role"] == "development"
    )
    development_row["target"]["caption"] = train_row["target"]["caption"]
    development_row["root_asset"]["image_sha256"] = train_row["root_asset"][
        "image_sha256"
    ]
    with pytest.raises(qa.QAError):
        qa.assert_leakage(rows)

    rows = copy.deepcopy(state["rows"])
    rows[0]["observation"]["prompt"] = rows[0]["target"]["caption"]
    with pytest.raises(qa.QAError):
        qa.assert_leakage(rows)


def test_reserved_family_is_rejected(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    patterns = state["qa_config"]["expected"]["reserved_patterns"]
    reserved_families = state["qa_config"]["expected"]["reserved_families"]

    rows = copy.deepcopy(state["rows"])
    rows[0]["root_asset"]["upstream_url"] = (
        "https://images.example.test/mscoco/train2017/000000000001.jpg"
    )
    matches = qa.find_reserved_family_matches(rows, patterns, reserved_families)
    assert any(match["pattern"] == "mscoco" for match in matches)

    rows = copy.deepcopy(state["rows"])
    rows[0]["source"]["family"] = reserved_families[0]
    matches = qa.find_reserved_family_matches(rows, patterns, reserved_families)
    assert any(match["pattern"] == "<reserved family>" for match in matches)


def test_sealed_eval_tamper_is_detected(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    path = state["role_manifest_paths"]["final_eval_untouched"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["sealed"] = False
    manifest["untouched"] = False
    manifest["seal_sha256"] = "0" * 64
    path.write_text(_json_text(manifest), encoding="utf-8")

    report = _run(state)
    failed = _failed_ids(report)

    assert "pin.role_manifest.final_eval_untouched.sha256" in failed
    assert "tranche.seal.final_eval_untouched.sealed" in failed
    assert "tranche.seal.final_eval_untouched.sha256" in failed
    assert report["seals"]["final_eval_untouched"]["sealed"] is False
    assert report["admission"]["training_admitted"] is False


# ---------------------------------------------------------------------------
# admission gating
# ---------------------------------------------------------------------------


def test_admission_refused_when_semantic_review_incomplete(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    report, template = _prepare_reports(state)
    reviews = _complete_reviews(template)
    first_row = next(iter(reviews))
    decision = copy.deepcopy(template)
    decision.update(
        {
            "decision": qa.ADMIT,
            "decided_by": "main",
            "decided_at": ISO,
            "semantic_reviews": {first_row: reviews[first_row]},
            "training_admitted_row_ids": None,
        }
    )
    decision_path = _write_decision(state, decision)

    refused = _run(state, write=True, decision_path=decision_path)

    assert refused["machine_checks"]["passed"] is True
    assert refused["status"] == "machine_checks_passed_admission_refused"
    assert refused["admission_checks"]["passed"] is False
    assert refused["admission"]["training_admitted"] is False
    assert "semantic_review_incomplete" in refused["admission"]["refusal_reasons"]
    assert set(template["semantic_reviews"]).isdisjoint(template["sealed_row_ids"])
    assert not (
        set(refused["admission"]["semantic_review"]["incomplete_row_ids"])
        & set(template["sealed_row_ids"])
    )
    assert refused["admission"]["semantic_review"]["required_row_ids"] == template[
        "reviewable_row_ids"
    ]
    assert not (state["report_dir"] / qa.TRAINING_OUTPUT_NAME).exists()


def test_admission_refuses_nontraining_rows(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    report, template = _prepare_reports(state)
    sealed_row = template["sealed_row_ids"][0]
    decision = copy.deepcopy(template)
    decision.update(
        {
            "decision": qa.ADMIT,
            "decided_by": "main",
            "decided_at": ISO,
            "semantic_reviews": _complete_reviews(template),
            "training_admitted_row_ids": report["admission"][
                "training_candidate_row_ids"
            ]
            + [sealed_row],
        }
    )
    decision_path = _write_decision(state, decision)

    refused = _run(state, write=True, decision_path=decision_path)

    assert refused["machine_checks"]["passed"] is True
    assert refused["admission"]["training_admitted"] is False
    assert "training_admitted_row_ids_mismatch" in refused["admission"][
        "refusal_reasons"
    ]
    assert "non_training_rows_requested_for_admission" in refused["admission"][
        "refusal_reasons"
    ]
    assert not (state["report_dir"] / qa.TRAINING_OUTPUT_NAME).exists()


@pytest.mark.parametrize(
    "sealed_review",
    [
        {
            "verdict": "match",
            "reviewer": "main",
            "reviewed_at": ISO,
            "notes": "attempted sealed review",
        },
        {
            "verdict": None,
            "reviewer": None,
            "reviewed_at": None,
            "notes": None,
        },
    ],
)
def test_sealed_semantic_review_is_refused(
    tmp_path: Path, sealed_review: dict
) -> None:
    state = build_fixture(tmp_path)
    _, template = _prepare_reports(state)
    sealed_row = template["sealed_row_ids"][0]
    reviews = _complete_reviews(template)
    reviews[sealed_row] = sealed_review
    decision = copy.deepcopy(template)
    decision.update(
        {
            "decision": qa.ADMIT,
            "decided_by": "main",
            "decided_at": ISO,
            "semantic_reviews": reviews,
            "training_admitted_row_ids": None,
        }
    )
    decision_path = _write_decision(state, decision)

    refused = _run(state, write=True, decision_path=decision_path)

    assert refused["machine_checks"]["passed"] is True
    assert refused["admission_checks"]["passed"] is False
    assert refused["admission"]["training_admitted"] is False
    assert "sealed_semantic_review_forbidden" in refused["admission"][
        "refusal_reasons"
    ]
    assert sealed_row in refused["admission"]["semantic_review"][
        "sealed_review_forbidden_row_ids"
    ]
    assert refused["admission"]["semantic_review"]["sealed_semantic_review"] == "prohibited"
    assert not (state["report_dir"] / qa.TRAINING_OUTPUT_NAME).exists()


def test_complete_semantic_review_admits_only_training_rows(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    report, template = _prepare_reports(state)
    training_ids = report["admission"]["training_candidate_row_ids"]
    decision = copy.deepcopy(template)
    decision.update(
        {
            "decision": qa.ADMIT,
            "decided_by": "main",
            "decided_at": ISO,
            "semantic_reviews": _complete_reviews(template),
            "training_admitted_row_ids": training_ids,
        }
    )
    decision_path = _write_decision(state, decision)

    admitted = _run(state, write=True, decision_path=decision_path)

    assert set(template["semantic_reviews"]) == set(template["reviewable_row_ids"])
    assert not (
        set(template["sealed_row_ids"]) & set(template["semantic_reviews"])
    )
    assert set(admitted["admission"]["semantic_review"]["required_row_ids"]) == set(
        template["reviewable_row_ids"]
    )
    assert admitted["status"] == "training_admitted"
    assert admitted["machine_checks"]["passed"] is True
    assert admitted["admission_checks"]["passed"] is True
    assert admitted["admission"]["training_admitted"] is True
    assert admitted["admission"]["training_admitted_row_ids"] == training_ids
    assert admitted["admission"]["refusal_reasons"] == []
    assert admitted["scope_separation"]["training_output_contains_only_training_role"] is True
    assert admitted["scope_separation"]["validation_only_excluded_from_training"] is True
    assert admitted["scope_separation"]["sealed_excluded_from_training"] is True

    output_path = state["report_dir"] / qa.TRAINING_OUTPUT_NAME
    assert output_path.exists()
    output_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["row_id"] for row in output_rows] == training_ids
    assert all(row["split"]["role"] == "train" for row in output_rows)
    assert not (
        set(admitted["admission"]["validation_only_row_ids"])
        & {row["row_id"] for row in output_rows}
    )
    assert not (
        set(admitted["admission"]["sealed_row_ids"])
        & {row["row_id"] for row in output_rows}
    )


def test_machine_failure_blocks_complete_semantic_admission(tmp_path: Path) -> None:
    state = build_fixture(tmp_path)
    _, template = _prepare_reports(state)
    decision = copy.deepcopy(template)
    decision.update(
        {
            "decision": qa.ADMIT,
            "decided_by": "main",
            "decided_at": ISO,
            "semantic_reviews": _complete_reviews(template),
            "training_admitted_row_ids": None,
        }
    )
    decision_path = _write_decision(state, decision)

    def mutate(config: dict) -> None:
        config["pins"]["corpus"]["sha256"] = "0" * 64

    _write_qa_config(state, mutate)
    refused = _run(state, write=True, decision_path=decision_path)

    assert refused["machine_checks"]["passed"] is False
    assert refused["machine_checks"]["failed"] > 0
    assert refused["admission"]["training_admitted"] is False
    assert "machine_checks_failed" in refused["admission"]["refusal_reasons"]
    assert refused["status"] == "machine_checks_failed"
    assert not (state["report_dir"] / qa.TRAINING_OUTPUT_NAME).exists()


# ---------------------------------------------------------------------------
# optional real-data integration (skipped with an explicit provisioning reason)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@require_local_artifacts("data/external_caption_tranche_20261025_v2/raw/images", "PIL")
def test_real_tranche_qa_is_clean_and_pending(tmp_path: Path) -> None:
    report = qa.run_qa(
        project_root=ROOT,
        config_path=ROOT / "configs" / "external_caption_asset_qa_v1.json",
        report_dir=tmp_path / "real_qa_report",
        write=False,
    )
    assert report["status"] == "machine_checks_passed_semantic_review_pending"
    assert report["machine_checks"]["passed"] is True
    assert report["review_queue"]["total_entries"] == 30
    assert report["decode"]["verified"] == 30
    assert report["admission"]["training_admitted"] is False
    assert len(report["admission"]["sealed_row_ids"]) == 8
    assert report["claims"]["semantic_caption_image_agreement_asserted_by_qa"] is False
