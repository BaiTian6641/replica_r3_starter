#!/usr/bin/env python3
"""Exact authenticated visual-source memory baseline on owned temporal assets (CPU).

This runner observes original temporal PNG bytes, clears the input material,
persists and reopens the ledger/blob store without decoding or re-encoding old
observations, retrieves original sources for delayed independently derived
requests, appends one metadata/crop/owner correction, verifies an unaffected
reference, and revokes the corrected source to invalidate its correction
descendant. Corrections are source-lineage metadata only, never corrected visual
factual memory.

It is an exact source-retrieval engineering baseline. It loads no model, trains
no encoder, predicts no captions/colors/OCR, and makes no perception or V3
claim. Learned/factual memory-only recall is not implemented or evaluated; the
``provenance_only`` operation returns metadata and no bytes. Manifest asset paths
are resolved under the corpus root before both reads. The public/verifier caption
files are not read; only the temporal asset manifest and original PNG bytes enter
this path. Pixel rehash and PNG IHDR checks below are explicitly host-only oracle
checks with no scene labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.memory_system import EligibilityScope  # noqa: E402


REPORT_FORMAT = "replica.exact_visual_memory.report.v1"
SUBJECT = "visual-source-memory"
BRANCH = "source"
PURPOSE = "exact_visual_source"
OWNER = "owner:temporal-owned-corpus"
INHERIT = object()
_VARIANT = re.compile(r"t(\d+)\Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scope(as_of: int) -> EligibilityScope:
    return EligibilityScope(SUBJECT, BRANCH, PURPOSE, as_of, "source_current", None)


def _variant_time(asset: dict) -> int:
    variant = asset["lineage"]["variant"]
    match = _VARIANT.fullmatch(variant)
    if match is None:
        raise ValueError(f"unsupported temporal variant: {variant}")
    return int(match.group(1))


def _layout(asset: dict):
    from replica_lab.runtime.visual_source_memory import SourceLayout
    return SourceLayout(asset["lineage"]["layout_template"])


def _asset_crop(asset: dict):
    crop = asset["lineage"]["crop_box"]
    return tuple(crop) if crop is not None else None


def _asset_path(corpus_root: Path, asset: dict) -> Path:
    from replica_lab.runtime.visual_source_memory import resolve_corpus_asset
    return resolve_corpus_asset(corpus_root, asset["path"])


def _build_request(asset: dict, *, source_family: str, query_prefix: str,
                   owner_id: str = OWNER, crop_box: object = INHERIT,
                   layout=None, event_id: str | None = None,
                   observation_id: str | None = None):
    from replica_lab.runtime.visual_source_memory import (
        PNG_REPRESENTATION, VisualSourceIdentity, VisualSourceRequest,
    )
    crop = _asset_crop(asset) if crop_box is INHERIT else crop_box
    identity = VisualSourceIdentity(
        source_id=f"source:{source_family}", family_id=source_family,
        independent_world_id=asset["group_id"], asset_id=asset["asset_id"],
        owner_id=owner_id, event_id=event_id or f"event:{asset['asset_id']}",
        representation=PNG_REPRESENTATION, content_sha256=asset["sha256"],
        width=asset["width"], height=asset["height"],
        byte_length=asset["bytes"], crop_box=crop,
        layout=layout if layout is not None else _layout(asset),
    )
    return VisualSourceRequest(
        f"{query_prefix}:{asset['asset_id']}",
        observation_id or asset["asset_id"], identity)


def _load_corpus(corpus_root: Path, max_worlds: int) -> tuple[dict, list[dict], dict]:
    manifest_path = corpus_root / "temporal" / "temporal_manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format") != "replica.temporal_vision_manifest.v1":
        raise ValueError("unexpected temporal manifest format")
    # Captions live under `groups`; drop them before any asset reaches memory.
    manifest.pop("groups", None)
    worlds = sorted({asset["group_id"] for asset in manifest["assets"]})
    if max_worlds < 1 or max_worlds > len(worlds):
        raise ValueError(f"--max-worlds must be in 1..{len(worlds)}")
    selected_worlds = worlds[:max_worlds]
    assets = [asset for asset in manifest["assets"]
              if asset["group_id"] in set(selected_worlds)]
    for asset in assets:
        path = _asset_path(corpus_root, asset)
        data = path.read_bytes()
        if len(data) != asset["bytes"] or _sha256(data) != asset["sha256"]:
            raise ValueError(f"temporal asset differs from manifest: {asset['asset_id']}")
        del data
    owned_source_family = manifest["provenance"]["source_id"]
    corpus = {
        "root": str(corpus_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(raw),
        "config_fingerprint": manifest["config_fingerprint"],
        "generation_fingerprint": manifest["generation_fingerprint"],
        "provenance_source_id": owned_source_family,
        "owned_source_family": owned_source_family,
        "stage": manifest["stage"],
        "selected_independent_worlds": selected_worlds,
        "max_worlds": max_worlds,
        "captions_passed_to_memory": False,
        "verifier_labels_loaded": False,
        "public_caption_file_read": False,
        "external_assets": 0,
        "asset_path_policy": (
            "resolve_relative containment under the corpus root before both reads"),
    }
    derivation_groups = sorted({asset["group_id"] for asset in assets})
    manifest_world_families = sorted({asset["family"] for asset in assets})
    counts = {
        # The owned renderer/source family is singular; world/derivation groups
        # are a separate lineage field and must not be counted as source families.
        "owned_source_families": [owned_source_family],
        "owned_source_family_count": 1,
        "world_derivation_groups": derivation_groups,
        "world_derivation_group_count": len(derivation_groups),
        "manifest_world_family_ids": manifest_world_families,
        "manifest_world_family_count": len(manifest_world_families),
        "independent_worlds": derivation_groups,
        "independent_world_count": len(derivation_groups),
        "assets": len(assets),
        "asset_bytes": sum(asset["bytes"] for asset in assets),
    }
    return corpus, assets, counts


def _expect(call, error_types) -> str:
    try:
        call()
    except error_types as error:
        return type(error).__name__
    raise RuntimeError("an expected exact-source rejection did not occur")


def run(args) -> dict:
    from replica_lab.runtime.visual_source_memory import (
        PNG_REPRESENTATION, SourceLayout, SourceUnavailableError, StaleCropError,
        StaleEpochError, StaleOwnerError, VisualSourceMemory, decode_rgb_png,
    )

    corpus, assets, corpus_counts = _load_corpus(args.corpus, args.max_worlds)
    output = args.output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    store = VisualSourceMemory(output / "visual_source.sqlite", output / "blobs")
    receipts = []
    asset_bytes_read = 0
    try:
        # 1. Observe original bytes, then clear the input material.
        for asset in assets:
            data = _asset_path(args.corpus, asset).read_bytes()
            if len(data) != asset["bytes"] or _sha256(data) != asset["sha256"]:
                raise ValueError(f"asset changed during capture: {asset['asset_id']}")
            observed_at = _variant_time(asset)
            receipt = store.observe(
                data, observation_id=asset["asset_id"], subject_id=SUBJECT,
                branch_id=BRANCH, purpose=PURPOSE,
                source_id=f"source:{corpus['owned_source_family']}",
                family_id=corpus["owned_source_family"],
                independent_world_id=asset["group_id"], asset_id=asset["asset_id"],
                owner_id=OWNER, event_id=f"event:{asset['asset_id']}",
                observed_at=observed_at, event_at=observed_at,
                producer_id=corpus["provenance_source_id"],
                crop_box=_asset_crop(asset), layout=_layout(asset),
                representation=PNG_REPRESENTATION,
                authorized_subject=SUBJECT, authorized_branch=BRANCH,
                source_authorized=True)
            receipts.append(receipt.summary())
            asset_bytes_read += len(data)
            del data
        capture_epoch = store.epoch
        ledger_id = store.ledger_id
        input_blob_stats = store.blob_statistics()
        store.close()

        # 2. Reopen without touching input asset files or re-encoding observations.
        store = VisualSourceMemory(output / "visual_source.sqlite", output / "blobs")
        reload_epoch = store.epoch
        reload_ledger_id = store.ledger_id
        reload_inventory = store.inventory(_scope(20), expected_epoch=capture_epoch)

        # 3. Delayed, independently derived exact-source requests.
        retrieval = {
            "requests": 0, "source_evidence": 0, "provenance_only": 0,
            "source_bytes_returned": 0,
            "dispositions": Counter(), "evidence": [],
            "provenance_metadata_records": 0,
            "learned_factual_memory_recall": {
                "status": "not_evaluated",
                "qualified": False,
                "reason": ("this exact-source baseline has no learned or factual "
                           "memory-only recall implementation"),
            },
            "host_only_oracle": {
                "label": "host_only_pixel_rehash_and_rgb8_png_decode",
                "scope": "returned original bytes only; no scene labels",
                "perception": False, "v3_claim": False,
                "scene_labels_verified": False,
                "all_sha256_match": True, "all_dimensions_match": True,
            },
        }
        for asset in assets:
            request = _build_request(
                asset, source_family=corpus["owned_source_family"],
                query_prefix="delayed")
            evidence = store.retrieve(request, _scope(20), expected_epoch=capture_epoch)
            retrieval["requests"] += 1
            retrieval["source_evidence"] += 1
            retrieval["source_bytes_returned"] += len(evidence.source_bytes)
            retrieval["dispositions"][evidence.disposition] += 1
            oracle_hash = _sha256(evidence.source_bytes)
            width, height, _rows = decode_rgb_png(evidence.source_bytes)
            hash_match = oracle_hash == asset["sha256"]
            dimensions_match = (width, height) == (asset["width"], asset["height"])
            retrieval["host_only_oracle"]["all_sha256_match"] &= hash_match
            retrieval["host_only_oracle"]["all_dimensions_match"] &= dimensions_match
            entry = evidence.summary()
            entry["host_only_pixel_sha256_match"] = hash_match
            entry["host_only_dimensions_match"] = dimensions_match
            retrieval["evidence"].append(entry)

            provenance = store.provenance_only(
                _scope(20), query_id=f"provenance-only:{asset['asset_id']}",
                observation_id=asset["asset_id"], expected_epoch=capture_epoch)
            retrieval["provenance_only"] += 1
            retrieval["dispositions"][provenance.disposition] += 1
            retrieval["provenance_metadata_records"] += 1
            if (provenance.source_bytes_returned != 0
                    or provenance.learned_factual_memory_recall != "not_evaluated"):
                raise AssertionError("provenance-only metadata crossed its contract")

        # 4. Append one correction; the parent bytes are reused from the blob store.
        target = assets[0]
        unaffected = next((asset for asset in assets[1:]
                           if asset["group_id"] != target["group_id"]), None)
        if unaffected is None:
            unaffected = assets[1] if len(assets) > 1 else None
        if unaffected is None:
            raise ValueError("the baseline needs at least two selected assets")
        target_request = _build_request(
            target, source_family=corpus["owned_source_family"],
            query_prefix="target")
        unaffected_request = _build_request(
            unaffected, source_family=corpus["owned_source_family"],
            query_prefix="unaffected")
        target_before = store.retrieve(target_request, _scope(20))
        unaffected_before = store.retrieve(unaffected_request, _scope(20))
        target_reference = target_before.reference()
        unaffected_reference = unaffected_before.reference()

        corrected_owner = "owner:temporal-owned-corpus-corrected"
        corrected_crop = (8, 8, 120, 88)
        corrected_layout = SourceLayout(f"{target['lineage']['layout_template']}:corrected")
        correction_id = f"{target['asset_id']}:correction"
        correction_event = f"event:{target['asset_id']}:correction"
        correction_time = max(_variant_time(target), 3) + 1
        correction_epoch = store.epoch
        correction_receipt = store.correct(
            target["asset_id"], correction_id=correction_id, scope=_scope(20),
            owner_id=corrected_owner, crop_box=corrected_crop, layout=corrected_layout,
            event_id=correction_event, observed_at=correction_time,
            event_at=correction_time, authorized_subject=SUBJECT,
            authorized_branch=BRANCH, source_authorized=True)
        corrected_request = _build_request(
            target, source_family=corpus["owned_source_family"],
            query_prefix="corrected", owner_id=corrected_owner,
            crop_box=corrected_crop, layout=corrected_layout,
            event_id=correction_event, observation_id=correction_id)
        corrected_evidence = store.retrieve(corrected_request, _scope(20))
        old_owner_request = _build_request(
            target, source_family=corpus["owned_source_family"],
            query_prefix="old-owner", owner_id=OWNER,
            crop_box=corrected_crop, layout=corrected_layout,
            event_id=correction_event, observation_id=correction_id)
        old_crop_request = _build_request(
            target, source_family=corpus["owned_source_family"],
            query_prefix="old-crop", owner_id=corrected_owner,
            crop_box=_asset_crop(target), layout=corrected_layout,
            event_id=correction_event, observation_id=correction_id)
        old_epoch_error = _expect(
            lambda: store.retrieve(corrected_request, _scope(20),
                                   expected_epoch=capture_epoch), StaleEpochError)
        superseded_error = _expect(
            lambda: store.retrieve(target_request, _scope(20)), SourceUnavailableError)
        old_owner_error = _expect(
            lambda: store.retrieve(old_owner_request, _scope(20)), StaleOwnerError)
        old_crop_error = _expect(
            lambda: store.retrieve(old_crop_request, _scope(20)), StaleCropError)
        superseded_reference_error = _expect(
            lambda: store.verify_reference(target_reference, scope=_scope(20)),
            SourceUnavailableError)
        unaffected_verification = store.verify_reference(
            unaffected_reference, scope=_scope(20))
        if not unaffected_verification.valid:
            raise AssertionError("unaffected reference did not survive the correction")
        unaffected_after_correction = store.retrieve(unaffected_request, _scope(20))

        correction_report = {
            "parent_observation_id": target["asset_id"],
            "correction_observation_id": correction_id,
            "kind": correction_receipt.kind,
            "supersedes": correction_receipt.supersedes,
            "admitted": correction_receipt.admitted,
            "parent_record_hash_before": target_before.record_hash,
            "correction_record_hash": correction_receipt.record_hash,
            "content_sha256": correction_receipt.content_sha256,
            "content_sha256_unchanged": (
                correction_receipt.content_sha256 == target_before.identity.content_sha256),
            "owner_before": target_before.identity.owner_id,
            "owner_after": corrected_evidence.identity.owner_id,
            "crop_before": target_before.identity.crop_box,
            "crop_after": corrected_evidence.identity.crop_box,
            "layout_before": target_before.identity.layout.as_dict(),
            "layout_after": corrected_evidence.identity.layout.as_dict(),
            "blob_created": correction_receipt.blob_created,
            "epoch_before": correction_epoch,
            "epoch_after": store.epoch,
            "corrected_retrieval": corrected_evidence.summary(),
            "old_owner_rejection": old_owner_error,
            "old_crop_rejection": old_crop_error,
            "stale_epoch_rejection": old_epoch_error,
            "superseded_parent_rejection": superseded_error,
            "superseded_parent_reference_rejection": superseded_reference_error,
            "correction_scope": (
                "metadata/crop/owner source-lineage only; not a corrected "
                "visual factual memory"),
            "corrects_visual_fact": False,
        }
        unaffected_report = {
            "observation_id": unaffected["asset_id"],
            "reference": unaffected_verification.summary(),
            "retrieved_after_correction": (
                len(unaffected_after_correction.source_bytes) > 0),
            "source_sha256": _sha256(unaffected_after_correction.source_bytes),
            "provenance_only_disposition": store.provenance_only(
                _scope(20), query_id="unaffected-provenance-only",
                observation_id=unaffected["asset_id"]).disposition,
        }

        # 5. Revoke the corrected source; the ledger invalidates its descendant.
        corrected_reference = corrected_evidence.reference()
        revocation = store.revoke(
            target["asset_id"], scope=_scope(20), authorized_subject=SUBJECT,
            authorized_branch=BRANCH, source_authorized=True)
        if correction_id not in revocation.revoked_observation_ids:
            raise AssertionError("the correction descendant was not invalidated")
        parent_after_error = _expect(
            lambda: store.retrieve(target_request, _scope(20)), SourceUnavailableError)
        correction_after_error = _expect(
            lambda: store.retrieve(corrected_request, _scope(20)),
            SourceUnavailableError)
        provenance_after_error = _expect(
            lambda: store.provenance_only(
                _scope(20), query_id="revoked-provenance-only",
                observation_id=correction_id), SourceUnavailableError)
        target_reference_after_error = _expect(
            lambda: store.verify_reference(target_reference, scope=_scope(20)),
            SourceUnavailableError)
        correction_reference_after_error = _expect(
            lambda: store.verify_reference(corrected_reference, scope=_scope(20)),
            SourceUnavailableError)
        unaffected_after_revocation = store.retrieve(unaffected_request, _scope(20))
        unaffected_revocation_verification = store.verify_reference(
            unaffected_reference, scope=_scope(20))
        final_inventory = store.inventory(_scope(20))
        final_blobs = store.blob_statistics()
        revocation_report = {
            "target_observation_id": target["asset_id"],
            "revoked_observation_ids": list(revocation.revoked_observation_ids),
            "revoked_count": len(revocation.revoked_observation_ids),
            "descendant_invalidated": correction_id in revocation.revoked_observation_ids,
            "epoch_before": revocation.epoch_before,
            "epoch_after": revocation.epoch_after,
            "retained_blob_sha256": list(revocation.retained_blob_sha256),
            "parent_retrieval_rejection": parent_after_error,
            "correction_retrieval_rejection": correction_after_error,
            "provenance_only_rejection": provenance_after_error,
            "target_reference_rejection": target_reference_after_error,
            "correction_reference_rejection": correction_reference_after_error,
            "unaffected_retrieval": {
                "source_sha256": _sha256(unaffected_after_revocation.source_bytes),
                "disposition": unaffected_after_revocation.disposition,
            },
            "unaffected_reference": unaffected_revocation_verification.summary(),
        }
        if not unaffected_revocation_verification.valid:
            raise AssertionError("unaffected source did not survive the revocation")

        report = {
            "format": REPORT_FORMAT,
            "kind": "exact_visual_source_memory_cpu_baseline",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": "cpu",
            "model_loaded": False,
            "model_learning": False,
            "encoder_training": False,
            "neural_memory_replacement": False,
            "perception_claim": False,
            "v3_claim": False,
            "scope": {
                "subject_id": SUBJECT, "branch_id": BRANCH, "purpose": PURPOSE,
                "as_of": 20, "mode": "source_current",
            },
            "representation": PNG_REPRESENTATION,
            "corpus": {**corpus, "counts": corpus_counts},
            "capture": {
                "observed": len(receipts),
                "admitted": sum(1 for receipt in receipts if receipt["admitted"]),
                "asset_bytes_read": asset_bytes_read,
                "input_context_cleared": True,
                "reencode_operations": 0,
                "epoch_after_capture": capture_epoch,
                "ledger_id": ledger_id,
                "blob_count": input_blob_stats["blob_count"],
                "blob_bytes": input_blob_stats["blob_bytes"],
                "receipts": receipts,
            },
            "persistence": {
                "reopened": True,
                "ledger_id": reload_ledger_id,
                "epoch": reload_epoch,
                "epoch_matches_capture": reload_epoch == capture_epoch,
                "inventory_count": len(reload_inventory.observations),
                "blob_count": input_blob_stats["blob_count"],
                "blob_bytes": input_blob_stats["blob_bytes"],
                "asset_files_read_during_retrieval": 0,
                "observations_reencoded": 0,
            },
            "retrieval": {
                **{key: value for key, value in retrieval.items()
                   if key != "dispositions"},
                "dispositions": dict(sorted(retrieval["dispositions"].items())),
                "source_evidence_disposition": "source_evidence",
                "provenance_only_disposition": "provenance_only",
                "learned_factual_memory_recall_status": "not_evaluated",
            },
            "correction": correction_report,
            "unaffected_reference": unaffected_report,
            "revocation": revocation_report,
            "final_inventory": final_inventory.summary(),
            "final_store": {
                "ledger_id": store.ledger_id,
                "epoch": store.epoch,
                "blob_count": final_blobs["blob_count"],
                "blob_bytes": final_blobs["blob_bytes"],
            },
            "counts_contract": {
                "note": ("the owned renderer/source family is singular and is "
                         "counted separately from temporal world/derivation groups"),
                "owned_source_family": corpus_counts["owned_source_families"][0],
                "owned_source_family_count": (
                    corpus_counts["owned_source_family_count"]),
                "world_derivation_group_count": (
                    corpus_counts["world_derivation_group_count"]),
                "world_derivation_groups": (
                    corpus_counts["world_derivation_groups"]),
                "manifest_world_family_count": (
                    corpus_counts["manifest_world_family_count"]),
                "independent_world_count": corpus_counts["independent_world_count"],
                "final_owned_source_family_count": (
                    final_inventory.source_family_count),
                "final_independent_world_count": (
                    final_inventory.independent_world_count),
            },
            "host_only_oracle": retrieval["host_only_oracle"],
            "unrun_scope": [
                "no model, encoder, neural memory or optimizer was loaded",
                "no perception, captioning, OCR, color or scene answer was predicted",
                "no verifier scene label or public caption entered memory",
                "learned/factual memory-only recall was not implemented or evaluated",
                "no learned visual memory or neural RSI capability is claimed",
                "no V3 perception or generalization evaluation was run",
                "no external or real-image asset was used",
                "corrections are metadata/crop/owner lineage only, not corrected visual facts",
            ],
        }
        if not (report["retrieval"]["host_only_oracle"]["all_sha256_match"]
                and report["retrieval"]["host_only_oracle"]["all_dimensions_match"]):
            raise AssertionError("host-only pixel oracle disagreed with the manifest")
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh run directory for ledger, blobs and report.json")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data/vision_preparation_v1",
                        help="prepared vision corpus root containing temporal/")
    parser.add_argument("--max-worlds", type=int, default=12,
                        help="bounded number of independent temporal worlds to observe")
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir()
                                 or any(args.output.iterdir())):
        parser.error("use a fresh output directory")
    report = run(args)
    summary = {
        "format": report["format"],
        "kind": report["kind"],
        "observed": report["capture"]["observed"],
        "retrieval": report["retrieval"]["dispositions"],
        "correction": {
            "correction_observation_id": report["correction"]["correction_observation_id"],
            "superseded_parent_rejection": report["correction"]["superseded_parent_rejection"],
            "old_owner_rejection": report["correction"]["old_owner_rejection"],
            "old_crop_rejection": report["correction"]["old_crop_rejection"],
            "stale_epoch_rejection": report["correction"]["stale_epoch_rejection"],
            "scope": report["correction"]["correction_scope"],
            "corrects_visual_fact": report["correction"]["corrects_visual_fact"],
        },
        "unaffected_reference_valid": report["unaffected_reference"]["reference"]["valid"],
        "revocation": {
            "revoked_count": report["revocation"]["revoked_count"],
            "descendant_invalidated": report["revocation"]["descendant_invalidated"],
            "epoch_before": report["revocation"]["epoch_before"],
            "epoch_after": report["revocation"]["epoch_after"],
        },
        "final_epoch": report["final_store"]["epoch"],
        "owned_source_family_count": (
            report["counts_contract"]["owned_source_family_count"]),
        "world_derivation_group_count": (
            report["counts_contract"]["world_derivation_group_count"]),
        "independent_world_count": report["counts_contract"]["independent_world_count"],
        "host_only_oracle": report["host_only_oracle"],
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"report: {(args.output / 'report.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
