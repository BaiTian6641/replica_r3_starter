#!/usr/bin/env python3
"""Deterministic CPU validation for the corrected caption lineage and sealed eval.

The validator never loads a model and never writes to the data roots.  It checks:

* pinned parent evidence and the still-published ``reject`` QA decision;
* the new nine-row training lineage (hashes, provenance, rights, split
  inheritance, duplicate groups and the complete exclusion ledger);
* the sealed synthetic evaluation partition (seal digest, artifact hashes,
  strict PNG decode, analytic geometry/color predicates, caption derivation,
  probes and label provenance);
* separation between training, validation-only, sealed/exposed and synthetic
  evaluation rows;
* an explicitly diagnostic synthetic evaluation contract, never natural-image
  or broad visual validation;
* a deterministic unsigned admission evidence packet carrying the exact receipt
  fields and bindings, without a decision, reviewer, timestamp or self-digest;
* a separate trusted local reviewed decision receipt.

Training admission is refused by default.  A receipt is honoured only when its
self-digest verifies, its decision names the exact lineage, configs, manifests,
rows, exclusions and evaluation seal, and it lists exactly the nine eligible
child rows.  The receipt is a trusted local review decision, not a cryptographic
signature: it records what a named reviewer decided and which artifacts were
reviewed, and it uses no environment secret or cryptographic dependency.  Missing,
stale, tampered or wrong-row receipts never admit training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import zlib
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.contracts import canonical_json  # noqa: E402
from replica_lab.external_intake import normalize_caption  # noqa: E402

VALIDATOR_VERSION = "1.1.0"
LINEAGE_CONFIG_FORMAT = "replica.caption_lineage_correction_config.v1"
EVAL_CONFIG_FORMAT = "replica.synthetic_scene_eval_contract.v1"
LINEAGE_MANIFEST_FORMAT = "replica.caption_lineage_manifest.v1"
LINEAGE_ROW_FORMAT = "replica.caption_lineage_row.v1"
EXCLUSION_FORMAT = "replica.caption_lineage_exclusion.v1"
SPECS_FORMAT = "replica.synthetic_scene_specs.v1"
SPEC_FORMAT = "replica.synthetic_scene_spec.v1"
EVAL_ROW_FORMAT = "replica.synthetic_scene_eval_row.v1"
EVAL_MANIFEST_FORMAT = "replica.synthetic_scene_eval_manifest.v1"
SEAL_FORMAT = "replica.synthetic_scene_eval_seal.v1"
RECEIPT_FORMAT = "replica.caption_training_admission_receipt.v1"
RECEIPT_SCOPE = "training_admission"
PACKET_FORMAT = "replica.caption_training_admission_packet.v1"
PACKET_STATUS = "prepared_unsigned_awaiting_review"
PACKET_PATH = "reports/caption_admission_preparation_20261027/admission_packet.json"
PACKET_PENDING_FIELDS = ("decision", "decided_by", "decided_at",
                         "conditions", "receipt_sha256")
PACKET_FIELDS = {
    "format", "created_on", "status", "not_a_receipt", "unsigned",
    "signature_present", "lineage_id", "receipt_format", "receipt_path",
    "receipt_fields", "pending_reviewer_fields", "evidence",
    "reviewer_instructions", "non_claims",
}
PACKET_NON_CLAIMS = {
    "training_admitted": False,
    "model_execution": False,
    "native_execution": False,
    "relabels_rejected_decision": False,
    "clears_exposure_incident": False,
    "benchmark_quality": "none",
}
EVAL_DIAGNOSTIC_FLAGS = {
    "kind": "diagnostic_synthetic_scene_evaluation",
    "diagnostic_only": True,
    "natural_image_validation": False,
    "broad_visual_validation": False,
    "benchmark_quality": "none",
}
VALIDATION_FORMAT = "replica.caption_lineage_eval_validation.v1"
SEALED_EXTERNAL_ROLE = "final_eval_untouched"
DEVELOPMENT_ROLE = "development"
TRAIN_ROLE = "train"
EVAL_ROLE = "final_eval_untouched_synthetic"
EVAL_SPLIT = "synthetic_scene_eval_sealed"


class GuardError(ValueError):
    """A fail-closed validation condition was violated."""

    def __init__(self, check: str, message: str):
        super().__init__(f"{check}: {message}")
        self.check = check
        self.message = message


def _require(check: str, condition: bool, message: str) -> None:
    if not condition:
        raise GuardError(check, message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require("load", isinstance(value, dict), f"{path} is not a JSON object")
    return value


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    _require("load", all(isinstance(row, dict) for row in rows),
             f"{path} contains a non-object row")
    return rows


def _strip(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _hash_of(value: Mapping[str, Any], field: str) -> str:
    return sha256_text(canonical_json(_strip(value, field)))


def _normalized_caption_sha(caption: str) -> str:
    return sha256_text(normalize_caption(caption))


DUPLICATE_KEYS = ("row_sha256", "target_sha256", "image_sha256",
                  "normalized_caption_sha256", "root_group_key")


def find_duplicate_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Return non-empty duplicate groups across the declared lineage keys."""
    groups: dict[str, dict[str, list[str]]] = {key: defaultdict(list)
                                               for key in DUPLICATE_KEYS}
    for row in rows:
        row_id = row["row_id"]
        groups["row_sha256"][row["row_sha256"]].append(row_id)
        groups["target_sha256"][row["target"]["caption_sha256"]].append(row_id)
        groups["image_sha256"][row["asset"]["image_sha256"]].append(row_id)
        groups["normalized_caption_sha256"][
            row["provenance"]["normalized_caption_sha256"]].append(row_id)
        groups["root_group_key"][row["parent"]["tranche_row_id"].split("#")[0]].append(row_id)
    return {key: {value: members for value, members in values.items()
                  if len(members) > 1}
            for key, values in groups.items()
            if any(len(members) > 1 for members in values.values())}


# ---------------------------------------------------------------------------
# Strict PNG decode and independent geometric expectations


def decode_png(data: bytes, *, name: str = "image") -> dict[str, Any]:
    check = f"eval.decode.{name}"
    _require(check, data.startswith(b"\x89PNG\r\n\x1a\n"), "missing PNG signature")
    offset = 8
    width = height = None
    idat = bytearray()
    saw_end = False
    chunks = []
    while offset < len(data):
        _require(check, offset + 12 <= len(data), "truncated chunk header")
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        _require(check, len(payload) == length, "truncated chunk payload")
        crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
        _require(check, crc == (zlib.crc32(kind + payload) & 0xFFFFFFFF),
                 f"CRC mismatch in {kind!r}")
        chunks.append(kind)
        offset += 12 + length
        if kind == b"IHDR":
            _require(check, length == 13, "bad IHDR length")
            (width, height, depth, color_type, compression, filtering,
             interlace) = struct.unpack(">IIBBBBB", payload)
            _require(check, (depth, color_type, compression, filtering, interlace)
                     == (8, 2, 0, 0, 0), "only 8-bit RGB non-interlaced PNG")
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            saw_end = True
            break
        else:
            _require(check, kind in (b"PLTE", b"tRNS"), "unexpected chunk type")
    _require(check, saw_end and width and height is not None, "incomplete PNG")
    _require(check, chunks[:1] == [b"IHDR"], "IHDR must be first")
    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    _require(check, len(raw) == (stride + 1) * height,
             "decompressed scanline size mismatch")
    pixels = []
    for y in range(height):
        start = y * (stride + 1)
        _require(check, raw[start] == 0, "unsupported PNG filter")
        line = raw[start + 1:start + 1 + stride]
        pixels.append([[line[x * 3], line[x * 3 + 1], line[x * 3 + 2]]
                       for x in range(width)])
    return {"width": width, "height": height, "pixels": pixels, "chunks": chunks}


def _covered(item: Mapping[str, Any], px: float, py: float) -> bool:
    shape = item["shape"]
    cx, cy = item["center"]
    if shape == "circle":
        radius = item["radius"]
        return (px - cx) ** 2 + (py - cy) ** 2 <= radius * radius
    if shape == "square":
        half = item["half"]
        return abs(px - cx) <= half and abs(py - cy) <= half
    if shape == "triangle":
        (x1, y1), (x2, y2), (x3, y3) = item["vertices"]

        def edge(ax: float, ay: float, bx: float, by: float) -> float:
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

        e1, e2, e3 = edge(x1, y1, x2, y2), edge(x2, y2, x3, y3), edge(x3, y3, x1, y1)
        return (e1 >= 0 and e2 >= 0 and e3 >= 0) or \
               (e1 <= 0 and e2 <= 0 and e3 <= 0)
    raise GuardError("eval.geometry", f"unsupported shape {shape!r}")


def _expected_pixel(scene: Mapping[str, Any], x: int, y: int) -> list[int]:
    px, py = x + 0.5, y + 0.5
    current = list(scene["background"]["rgb"])
    for item in scene["objects"]:
        if _covered(item, px, py):
            current = list(item["rgb"])
    return current


def derive_caption(scene: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    template = config["generation"]["templates"][scene["template_id"]]
    params = dict(scene["caption_params"])
    if "count" in params:
        params["plural"] = "" if int(params["count"]) == 1 else "s"
    if "shape" in params:
        params["plural"] = params.get("plural", "")
    return template.format(**params)


def derive_probes(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    probes = []
    for point in ([2, 2], [125, 2], [2, 125]):
        probes.append({"kind": "background", "point": point,
                       "expected_rgb": list(scene["background"]["rgb"])})
    for index, item in enumerate(scene["objects"]):
        probes.append({"kind": "object", "object_index": index,
                       "point": list(item["probe"]),
                       "expected_rgb": list(item["rgb"]),
                       "shape": item["shape"],
                       "color_name": item["color_name"]})
    return probes


def _bbox(item: Mapping[str, Any]) -> tuple[int, int, int, int]:
    cx, cy = item["center"]
    if item["shape"] == "circle":
        radius = item["radius"]
        return cx - radius, cy - radius, cx + radius, cy + radius
    if item["shape"] == "square":
        half = item["half"]
        return cx - half, cy - half, cx + half, cy + half
    xs = [point[0] for point in item["vertices"]]
    ys = [point[1] for point in item["vertices"]]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Lineage checks


def _check_pins(config: Mapping[str, Any], project_root: Path,
                checks: list[dict[str, str]]) -> dict[str, Any]:
    parents = config["parents"]
    loaded = {}
    for name, pin in parents.items():
        if not isinstance(pin, dict) or "sha256" not in pin:
            continue
        path = project_root / pin["path"]
        _require(f"lineage.pin.{name}", path.is_file(), f"missing {pin['path']}")
        _require(f"lineage.pin.{name}", sha256_file(path) == pin["sha256"],
                 f"sha256 mismatch for {pin['path']}")
        if "bytes" in pin:
            _require(f"lineage.pin.{name}", path.stat().st_size == pin["bytes"],
                     f"byte-size mismatch for {pin['path']}")
        loaded[name] = path
        checks.append({"id": f"lineage.pin.{name}", "status": "pass",
                       "detail": pin["path"]})

    image_pin = parents["image_root"]
    image_root = project_root / image_pin["path"]
    files = sorted(path for path in image_root.iterdir() if path.is_file())
    aggregate = hashlib.sha256()
    for path in files:
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(path.read_bytes())
    _require("lineage.pin.images", aggregate.hexdigest() == image_pin["aggregate_sha256"],
             "image aggregate sha256 mismatch")
    _require("lineage.pin.images", len(files) == image_pin["files"],
             "image file count mismatch")
    checks.append({"id": "lineage.pin.images", "status": "pass",
                   "detail": f"{len(files)} files"})
    return {"paths": loaded, "image_root": image_root}


def validate_lineage(config: Mapping[str, Any], *, project_root: Path,
                     checks: list[dict[str, str]]) -> dict[str, Any]:
    _require("lineage.config_format", config.get("format") == LINEAGE_CONFIG_FORMAT,
             "unexpected lineage config format")
    _require("lineage.config_status",
             config.get("status") == "corrected_selection_derived_from_rejected_qa",
             "unexpected lineage status")

    pins = _check_pins(config, project_root, checks)
    decision = load_json(pins["paths"]["qa_decision"])
    exposure = load_json(pins["paths"]["exposure_incident"])
    corpus = load_rows(pins["paths"]["corpus"])
    train_manifest = load_json(pins["paths"]["train_manifest"])
    development_manifest = load_json(pins["paths"]["development_manifest"])
    sealed_manifest = load_json(pins["paths"]["sealed_eval_manifest"])

    _require("lineage.qa_decision", decision.get("decision") == "reject",
             "the parent QA decision is no longer 'reject'")
    _require("lineage.qa_decision",
             decision.get("training_admitted_row_ids") is None,
             "the parent QA decision illegally admitted training rows")
    _require("lineage.qa_decision",
             decision.get("qa_id") == config["qa_id"], "QA id mismatch")
    checks.append({"id": "lineage.qa_decision", "status": "pass",
                   "detail": "parent decision remains reject with no admitted rows"})

    exposed = tuple(exposure["exposed_row_ids"])
    _require("lineage.exposure",
             exposure.get("status") == "quarantined_after_semantic_review_exposure",
             "exposure incident is not quarantined")
    _require("lineage.exposure", tuple(
        config["parents"]["exposure_incident"]["exposed_row_ids"]) == exposed,
        "exposure ids differ between config and incident receipt")
    _require("lineage.exposure", len(exposed) == 8, "expected eight exposed rows")
    _require("lineage.exposure",
             tuple(decision.get("sealed_row_ids", ())) == exposed,
             "the QA decision's sealed row ids differ from the exposure receipt")
    _require("lineage.exposure",
             set(row["row_id"] for row in sealed_manifest["rows"]) == set(exposed)
             and sealed_manifest.get("sealed") is True
             and sealed_manifest.get("untouched") is True,
             "the original sealed manifest no longer matches the exposed rows")
    checks.append({"id": "lineage.exposure", "status": "pass",
                   "detail": "eight exposed rows remain sealed and quarantined"})

    outputs = config["outputs"]
    manifest = load_json(project_root / outputs["lineage_manifest"])
    rows = load_rows(project_root / outputs["rows"])
    exclusions = load_rows(project_root / outputs["exclusions"])
    _require("lineage.manifest_format",
             manifest.get("format") == LINEAGE_MANIFEST_FORMAT, "bad manifest format")
    _require("lineage.manifest_self_hash",
             manifest.get("manifest_sha256") == _hash_of(manifest, "manifest_sha256"),
             "lineage manifest self-hash mismatch")
    _require("lineage.rows_hash",
             manifest.get("rows_sha256")
             == sha256_text("".join(canonical_json(row) + "\n" for row in rows)),
             "rows file does not match the manifest rows hash")
    _require("lineage.exclusions_hash",
             manifest.get("exclusions_sha256")
             == sha256_text("".join(canonical_json(row) + "\n" for row in exclusions)),
             "exclusions file does not match the manifest hash")
    _require("lineage.counts", manifest["counts"]["rows"] == len(rows) == 9,
             "lineage must contain exactly nine rows")
    _require("lineage.counts", manifest["counts"]["exclusions"] == len(exclusions),
             "manifest exclusion count differs from the exclusions file")
    _require("lineage.counts", len({row["row_id"] for row in rows}) == 9,
             "row ids are not unique")
    checks.append({"id": "lineage.counts", "status": "pass",
                   "detail": "9 rows, 21 exclusions"})

    admitted_parent_ids = tuple(sorted(config["selection"]["admitted_row_ids"]))
    namespace = config["lineage_policy"]["child_namespace"]
    expected_child_ids = tuple(sorted(
        f"{namespace}:{row_id.split(':', 1)[1]}" for row_id in admitted_parent_ids))
    _require("lineage.selection", tuple(sorted(row["row_id"] for row in rows))
             == expected_child_ids, "lineage rows differ from the admitted set")
    _require("lineage.selection", len(admitted_parent_ids) == 9,
             "nine admitted ids required")
    checks.append({"id": "lineage.selection", "status": "pass",
                   "detail": "admitted set matches the config"})

    corpus_by_id = {row["row_id"]: row for row in corpus}
    train_rows = {row["row_id"]: row for row in train_manifest["rows"]}
    development_rows = {row["row_id"]: row for row in development_manifest["rows"]}
    sealed_rows = {row["row_id"]: row for row in sealed_manifest["rows"]}
    _require("lineage.corpus_coverage",
             set(corpus_by_id) == set(train_rows) | set(development_rows) | set(sealed_rows),
             "role manifests and corpus disagree")
    _require("lineage.corpus_coverage", len(corpus_by_id) == 30,
             "expected thirty parent rows")
    verdicts = decision["semantic_reviews"]
    validation_only = set(decision["validation_only_row_ids"])
    _require("lineage.validation_only", len(validation_only) == 7,
             "expected seven validation-only rows")

    required_terms = config["lineage_policy"]["rights_policy"]["required_terms"]
    required_acks = config["lineage_policy"]["rights_policy"]["required_acknowledgements"]
    seen_ids: set[str] = set()
    for row in rows:
        row_id = row["row_id"]
        _require("lineage.row_format", row.get("format") == LINEAGE_ROW_FORMAT,
                 f"bad row format for {row_id}")
        _require("lineage.row_id", row_id.startswith(
            config["lineage_policy"]["child_namespace"] + ":"),
            f"row is outside the new namespace: {row_id}")
        _require("lineage.row_id", row_id not in seen_ids, f"duplicate row id {row_id}")
        seen_ids.add(row_id)
        _require("lineage.row_sha256", row.get("row_sha256") == _hash_of(row, "row_sha256"),
                 f"row self-hash mismatch for {row_id}")
        _require("lineage.split", row["split"] == TRAIN_ROLE
                 and row["role"] == TRAIN_ROLE, f"bad child split for {row_id}")

        parent = row["parent"]
        parent_id = parent["tranche_row_id"]
        _require("lineage.parent_binding", parent_id in corpus_by_id,
                 f"missing parent row {parent_id}")
        parent_row = corpus_by_id[parent_id]
        binding = train_rows.get(parent_id)
        _require("lineage.parent_binding", binding is not None,
                 f"parent {parent_id} is not in the train manifest")
        _require("lineage.parent_role", parent_row["split"]["role"] == TRAIN_ROLE,
                 f"parent {parent_id} is not a train row")
        _require("lineage.parent_not_exposed", parent_id not in exposed,
                 f"exposed row {parent_id} appears in the lineage")
        _require("lineage.parent_not_validation", parent_id not in validation_only,
                 f"validation-only row {parent_id} appears in the lineage")
        _require("lineage.parent_not_sealed", parent_id not in sealed_rows,
                 f"sealed row {parent_id} appears in the lineage")
        _require("lineage.parent_row_sha",
                 sha256_text(canonical_json(parent_row)) == parent["tranche_row_sha256"]
                 == binding["row_sha256"], f"parent row hash mismatch for {row_id}")
        _require("lineage.parent_decision", parent["qa_decision"] == "reject",
                 "parent decision must remain reject")
        _require("lineage.parent_decision", parent["qa_decision_relabeled"] is False,
                 "lineage must not relabel the rejected decision")
        _require("lineage.parent_decision",
                 parent["qa_decision_sha256"] == config["parents"]["qa_decision"]["sha256"],
                 "parent decision hash mismatch")
        _require("lineage.parent_exposure",
                 parent["exposure_incident_sha256"]
                 == config["parents"]["exposure_incident"]["sha256"]
                 and parent["exposed_row_used"] is False,
                 "exposure binding mismatch")
        review = verdicts.get(parent_id)
        _require("lineage.match_verdict", review is not None
                 and review.get("verdict") == "match"
                 and parent["recorded_verdict"] == "match",
                 f"missing recorded match verdict for {parent_id}")
        _require("lineage.match_verdict",
                 parent["reviewer"] == review.get("reviewer")
                 and parent["reviewed_at"] == review.get("reviewed_at"),
                 f"review provenance mismatch for {parent_id}")

        caption = row["target"]["caption"]
        _require("lineage.target", row["target"]["caption_sha256"]
                 == sha256_text(caption) == binding["target_sha256"],
                 f"target hash mismatch for {row_id}")
        _require("lineage.target", caption == parent_row["target"]["caption"],
                 f"caption is not inherited verbatim for {row_id}")

        image_path = project_root / row["asset"]["local_path"]
        _require("lineage.image", image_path.is_file(),
                 f"image missing for {row_id}")
        image_bytes = image_path.read_bytes()
        _require("lineage.image", sha256_bytes(image_bytes)
                 == row["asset"]["image_sha256"] == binding["image_sha256"]
                 == parent_row["root_asset"]["image_sha256"],
                 f"image hash mismatch for {row_id}")
        _require("lineage.image", len(image_bytes) == row["asset"]["image_bytes"],
                 f"image byte size mismatch for {row_id}")

        _require("lineage.provenance", row["provenance"]["preprocessing_version"]
                 == parent_row["provenance"]["preprocessing_version"],
                 f"preprocessing provenance changed for {row_id}")
        _require("lineage.provenance", row["provenance"]["blip_caption_sha256"]
                 == parent_row["provenance"]["blip_caption_sha256"],
                 f"caption provenance changed for {row_id}")
        _require("lineage.provenance", row["source"]["id"] == parent_row["source"]["id"]
                 and row["source"]["revision"] == parent_row["source"]["revision"],
                 f"source identity changed for {row_id}")

        _require("lineage.rights", row["rights"]["permitted_use"]
                 == required_terms["permitted_use"]
                 and row["rights"]["research_use_only"] is True
                 and row["rights"]["redistribution"] == "prohibited"
                 and row["rights"]["commercial_use"] is False,
                 f"rights are not inherited for {row_id}")
        _require("lineage.rights", list(row["rights"]["acknowledgements"])
                 == list(required_acks), f"acknowledgements changed for {row_id}")
        _require("lineage.rights", row["rights"]["inherited_from_parent"] is True,
                 f"rights not marked inherited for {row_id}")

        inheritance = row["inheritance"]
        _require("lineage.split_inheritance",
                 inheritance["parent_role"] == TRAIN_ROLE
                 and inheritance["child_role"] == TRAIN_ROLE
                 and inheritance["reassigned"] is False,
                 f"split was reassigned for {row_id}")

    duplicates = find_duplicate_groups(rows)
    _require("lineage.duplicates", not duplicates,
             f"duplicate lineage groups: {duplicates}")
    checks.append({"id": "lineage.duplicates", "status": "pass",
                   "detail": "all duplicate keys unique"})

    excluded_expected = (
        set(config["selection"]["excluded_mismatch_row_ids"])
        | set(config["selection"]["excluded_unclear_row_ids"])
        | set(config["selection"]["validation_only_row_ids"])
        | set(exposed)
    )
    _require("lineage.exclusions", {item["row_id"] for item in exclusions}
             == excluded_expected, "exclusion ledger does not cover the expected rows")
    _require("lineage.exclusions", len(exclusions) == 21,
             "lineage must account for exactly twenty-one exclusions")
    _require("lineage.exclusions",
             len(exclusions) + len(rows) == len(corpus_by_id) == 30,
             "lineage and exclusions do not account for all thirty rows")
    for item in exclusions:
        row_id = item["row_id"]
        _require("lineage.exclusions", item.get("format") == EXCLUSION_FORMAT,
                 f"bad exclusion format for {row_id}")
        _require("lineage.exclusions", item["excluded"] is True
                 and item["training_eligible"] is False,
                 f"excluded row is marked eligible: {row_id}")
        _require("lineage.exclusions", row_id not in seen_ids,
                 f"excluded row also admitted: {row_id}")
        _require("lineage.exclusions", item["parent_decision"] == "reject"
                 and item["parent_decision_relabeled"] is False,
                 f"exclusion relabels the decision: {row_id}")
        role = corpus_by_id[row_id]["split"]["role"]
        verdict = verdicts.get(row_id, {}).get("verdict")
        if row_id in exposed:
            expected_reason = "sealed_exposed_quarantine"
        elif role == SEALED_EXTERNAL_ROLE:
            expected_reason = "sealed_role_never_training"
        elif role == DEVELOPMENT_ROLE:
            expected_reason = "validation_only_role"
        elif row_id in validation_only:
            expected_reason = "validation_only_verdict"
        elif verdict == "mismatch":
            expected_reason = "semantic_mismatch"
        elif verdict == "unclear":
            expected_reason = "semantic_unclear"
        else:
            expected_reason = "missing_semantic_verdict"
        _require("lineage.exclusions", item["reason_code"] == expected_reason,
                 f"wrong exclusion reason for {row_id}: {item['reason_code']}")
    checks.append({"id": "lineage.exclusions", "status": "pass",
                   "detail": "21 exclusions with reasons; no exposed row admitted"})

    manifest_ids = {item["row_id"] for item in manifest["rows"]}
    _require("lineage.manifest_rows", manifest_ids == seen_ids,
             "manifest row bindings differ from the rows file")
    for item in manifest["rows"]:
        _require("lineage.manifest_rows", item["role"] == TRAIN_ROLE,
                 f"manifest role mismatch for {item['row_id']}")
    checks.append({"id": "lineage.manifest_rows", "status": "pass",
                   "detail": "manifest bindings match"})

    return {
        "manifest": manifest,
        "rows": rows,
        "exclusions": exclusions,
        "corpus": corpus_by_id,
        "train_rows": train_rows,
        "development_rows": development_rows,
        "sealed_rows": sealed_rows,
        "decision": decision,
        "exposure": exposure,
        "exposed": exposed,
        "image_root": pins["image_root"],
    }


# ---------------------------------------------------------------------------
# Synthetic evaluation checks


def _check_eval_config_pins(config: Mapping[str, Any], project_root: Path,
                            checks: list[dict[str, str]]) -> dict[str, Path]:
    joins = config["disjointness"]
    lineage_config = project_root / joins["caption_lineage_config"]["path"]
    _require("eval.pin.lineage_config",
             sha256_file(lineage_config) == joins["caption_lineage_config"]["sha256"],
             "lineage config pin mismatch")
    exposure_path = project_root / joins["exposure_incident"]["path"]
    _require("eval.pin.exposure",
             sha256_file(exposure_path) == joins["exposure_incident"]["sha256"],
             "exposure pin mismatch")
    corpus_path = project_root / joins["external_corpus"]["path"]
    _require("eval.pin.corpus",
             sha256_file(corpus_path) == joins["external_corpus"]["sha256"],
             "external corpus pin mismatch")
    checks.append({"id": "eval.pin.sources", "status": "pass",
                   "detail": "lineage config, exposure and corpus pins verified"})
    return {"exposure": exposure_path, "corpus": corpus_path,
            "lineage_config": lineage_config}


def _validate_scene(scene: Mapping[str, Any], config: Mapping[str, Any],
                    palette: Mapping[str, Sequence[int]],
                    pixels: Sequence[Sequence[Sequence[int]]],
                    checks_by_family: dict[str, int]) -> dict[str, Any]:
    scene_id = scene["scene_id"]
    check = f"eval.scene.{scene_id}"
    _require(check, scene.get("format") == SPEC_FORMAT, "bad scene format")
    _require(check, scene["width"] == config["generation"]["width"]
             and scene["height"] == config["generation"]["height"],
             "scene dimensions differ from the contract")
    _require(check, scene["renderer_version"] == config["generation"]["renderer_version"],
             "renderer version mismatch")
    _require(check, scene.get("scene_spec_sha256") == _hash_of(scene, "scene_spec_sha256"),
             "scene spec self-hash mismatch")
    _require(check, len(pixels) == scene["height"]
             and all(len(row) == scene["width"] for row in pixels),
             "decoded image matrix shape mismatch")

    family = next((item for item in config["generation"]["families"]
                   if item["id"] == scene["family"]), None)
    _require(check, family is not None, "unknown family")
    _require(check, scene["template_id"] == family["template"], "template mismatch")
    background = scene["background"]
    _require(check, background["color_name"] in palette
             and list(background["rgb"]) == list(palette[background["color_name"]]),
             "background palette mismatch")

    objects = scene["objects"]
    _require(check, objects, "scene has no objects")
    for item in objects:
        _require(check, item["shape"] in config["generation"]["shape_order"],
                 "unsupported shape")
        _require(check, item["color_name"] in palette
                 and list(item["rgb"]) == list(palette[item["color_name"]]),
                 "object palette mismatch")
        cx, cy = item["center"]
        if item["shape"] in ("circle", "triangle"):
            _require(check, isinstance(item["radius"], int) and item["radius"] > 0,
                     "bad radius")
        else:
            _require(check, isinstance(item["half"], int) and item["half"] > 0,
                     "bad half size")
        x0, y0, x1, y1 = _bbox(item)
        _require(check, 0 <= x0 and 0 <= y0 and x1 < scene["width"]
                 and y1 < scene["height"], "object leaves the canvas")
        _require(check, _covered(item, item["probe"][0] + 0.5, item["probe"][1] + 0.5),
                 "object probe is outside its object")
        _require(check, all(0 <= value < scene["width"] for value in item["probe"]),
                 "probe point out of bounds")
    boxes = [_bbox(item) for item in objects]
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            a, b = boxes[first], boxes[second]
            _require(check, a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1],
                     "objects overlap")

    params = scene["caption_params"]
    shapes = [item["shape"] for item in objects]
    colors = [item["color_name"] for item in objects]
    if scene["template_id"] in ("counted_plural", "row_count"):
        _require(check, len(objects) == int(params["count"]),
                 "caption count differs from the rendered object count")
        _require(check, all(shape == params["shape"] for shape in shapes)
                 and all(color == params["color"] for color in colors),
                 "caption shape/color differ from the rendered objects")
        _require(check, params["background"] == background["color_name"],
                 "caption background mismatch")
    elif scene["template_id"] == "positioned_square":
        _require(check, len(objects) == 1, "positioned scene must render one object")
        _require(check, objects[0]["shape"] == params["shape"]
                 and objects[0]["color_name"] == params["color"],
                 "positioned object differs from the caption")
        position = next((item for item in family["positions"]
                         if item["phrase"] == params["position"]), None)
        _require(check, position is not None, "unknown position phrase")
        _require(check, list(objects[0]["center"]) == list(position["center"]),
                 "object center differs from the named position")
        _require(check, params["background"] == background["color_name"],
                 "caption background mismatch")
    elif scene["template_id"] == "pair_left_right":
        _require(check, len(objects) == 2, "pair scene must render two objects")
        left, right = objects
        _require(check, left["center"][0] < right["center"][0],
                 "pair objects are not left-to-right")
        _require(check, left["shape"] == params["left_shape"]
                 and left["color_name"] == params["left_color"]
                 and right["shape"] == params["right_shape"]
                 and right["color_name"] == params["right_color"],
                 "pair shapes/colors differ from the caption")
        _require(check, params["background"] == background["color_name"],
                 "caption background mismatch")
    else:
        raise GuardError(check, f"unknown template {scene['template_id']!r}")

    caption = derive_caption(scene, config)
    _require(check, scene["caption"] == caption, "caption template derivation mismatch")
    _require(check, derive_probes(scene) == scene["probes"],
             "stored probes differ from derived probes")
    for probe in scene["probes"]:
        x, y = probe["point"]
        _require(check, pixels[y][x] == probe["expected_rgb"],
                 f"probe {probe['point']} pixel mismatch")
        if probe["kind"] == "background":
            _require(check, not any(_covered(item, x + 0.5, y + 0.5)
                                    for item in objects),
                     "background probe is covered by an object")
            _require(check, probe["expected_rgb"] == scene["background"]["rgb"],
                     "background probe color mismatch")
        else:
            item = objects[probe["object_index"]]
            _require(check, item["shape"] == probe["shape"]
                     and item["color_name"] == probe["color_name"],
                     "object probe metadata mismatch")
            _require(check, _covered(item, x + 0.5, y + 0.5),
                     "object probe is not inside its object")

    for y in range(scene["height"]):
        for x in range(scene["width"]):
            if pixels[y][x] != _expected_pixel(scene, x, y):
                raise GuardError(check, f"pixel mismatch at ({x},{y})")
    checks_by_family[scene["family"]] = checks_by_family.get(scene["family"], 0) + 1
    return scene


def validate_eval(config: Mapping[str, Any], *, project_root: Path,
                  lineage: Mapping[str, Any],
                  checks: list[dict[str, str]]) -> dict[str, Any]:
    _require("eval.config_format", config.get("format") == EVAL_CONFIG_FORMAT,
             "unexpected eval config format")
    _require("eval.config_status",
             config.get("status") == "sealed_preparation_only_no_model_use",
             "unexpected eval status")
    limits = " ".join(config["benchmark_limits"]).casefold()
    _require("eval.diagnostic_scope",
             config["non_claims"]["external_data"] is False
             and config["non_claims"]["benchmark_quality"] == "none"
             and config["non_claims"]["training_use"] is False
             and config["non_claims"]["model_use"] == "none"
             and "no natural images" in limits
             and "cannot support claims about natural-image" in limits
             and "not a quality benchmark" in limits,
             "evaluation contract must be explicitly diagnostic, not "
             "natural-image validation")
    checks.append({"id": "eval.diagnostic_scope", "status": "pass",
                   "detail": "24 scenes are declared diagnostic synthetic "
                             "evaluation only"})
    pins = _check_eval_config_pins(config, project_root, checks)
    artifacts = config["artifacts"]
    specs_path = project_root / artifacts["scene_specs"]
    rows_path = project_root / artifacts["rows"]
    manifest_path = project_root / artifacts["manifest"]
    seal_path = project_root / artifacts["seal"]
    image_dir = project_root / artifacts["images"]
    for path in (specs_path, rows_path, manifest_path, seal_path):
        _require("eval.artifacts", path.is_file(), f"missing {path}")
    _require("eval.artifacts", image_dir.is_dir(), "missing image directory")

    specs = load_json(specs_path)
    rows = load_rows(rows_path)
    manifest = load_json(manifest_path)
    seal = load_json(seal_path)

    _require("eval.seal_format", seal.get("format") == SEAL_FORMAT, "bad seal format")
    _require("eval.seal_digest", seal.get("seal_sha256") == _hash_of(seal, "seal_sha256"),
             "seal digest mismatch")
    _require("eval.seal_identity", seal["eval_id"] == config["eval_id"]
             and seal["role"] == EVAL_ROLE and seal["split"] == EVAL_SPLIT,
             "seal identity mismatch")
    attestations = seal["attestations"]
    _require("eval.seal_attestation",
             attestations.get("sealed_before_any_model_use") is True
             and attestations.get("model_executions_at_seal") == 0
             and attestations.get("human_semantic_review_used") is False
             and attestations.get("labels_from_renderer_scene_spec") is True
             and attestations.get("no_external_images") is True
             and attestations.get("no_third_party_content") is True,
             "seal attestations do not match the contract")
    canonical_config_sha = sha256_text(canonical_json(config))
    _require("eval.contract_binding",
             seal.get("contract_canonical_sha256") == canonical_config_sha,
             "seal does not bind the canonical contract")
    checks.append({"id": "eval.seal", "status": "pass",
                   "detail": seal["seal_sha256"]})

    _require("eval.specs_format", specs.get("format") == SPECS_FORMAT
             and specs.get("eval_id") == config["eval_id"],
             "bad scene specs format")
    palette = specs.get("palette")
    _require("eval.palette", palette == config["generation"]["palette"],
             "palette differs from the contract")
    scenes = specs["scenes"]
    expected = config["expected"]
    _require("eval.counts", len(scenes) == expected["rows"] == len(rows) == 24,
             "expected twenty-four scenes and rows")
    _require("eval.counts", len({scene["scene_id"] for scene in scenes}) == len(scenes),
             "scene ids are not unique")
    _require("eval.counts", len({scene["caption"] for scene in scenes}) == len(scenes),
             "captions are not unique")

    for artifact, expected_sha in (
        ("scene_specs_sha256", sha256_file(specs_path)),
        ("rows_sha256", sha256_text("".join(canonical_json(row) + "\n" for row in rows))),
        ("manifest_sha256", sha256_file(manifest_path)),
    ):
        _require(f"eval.seal_binding.{artifact}",
                 seal["sealed_artifacts"][artifact] == expected_sha,
                 f"seal binding mismatch for {artifact}")
    checks.append({"id": "eval.seal_bindings", "status": "pass",
                   "detail": "specs, rows and manifest bindings verified"})

    image_files = sorted(path for path in image_dir.iterdir() if path.is_file())
    _require("eval.images_count", len(image_files) == expected["images"] == 24,
             "expected twenty-four images")
    aggregate = hashlib.sha256()
    image_by_name = {}
    for path in image_files:
        data = path.read_bytes()
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(data)
        image_by_name[path.name] = data
    _require("eval.images_aggregate",
             aggregate.hexdigest() == seal["sealed_artifacts"]["images_aggregate_sha256"],
             "image aggregate mismatch")
    checks.append({"id": "eval.images", "status": "pass",
                   "detail": f"{len(image_files)} PNG files, aggregate verified"})

    _require("eval.manifest_format",
             manifest.get("format") == EVAL_MANIFEST_FORMAT
             and manifest.get("eval_id") == config["eval_id"],
             "bad eval manifest format")
    _require("eval.manifest_counts", manifest["counts"]["rows"] == 24
             and manifest["counts"]["images"] == 24
             and manifest["counts"]["families"] == 4,
             "manifest counts mismatch")
    _require("eval.manifest_role", manifest["role"] == EVAL_ROLE
             and manifest["split"] == EVAL_SPLIT, "manifest role/split mismatch")
    _require("eval.manifest_spec_binding",
             manifest["scene_specs_sha256"] == seal["sealed_artifacts"]["scene_specs_sha256"]
             and manifest["rows_sha256"] == seal["sealed_artifacts"]["rows_sha256"],
             "manifest bindings differ from the seal")
    _require("eval.manifest_limits", manifest["benchmark_limits"]
             == config["benchmark_limits"], "benchmark limits not preserved")
    _require("eval.manifest_provenance",
             manifest["provenance"] == config["provenance"], "provenance mismatch")

    images_by_path = {item["path"]: item for item in manifest["images"]}
    _require("eval.manifest_images", len(images_by_path) == 24,
             "manifest image list is incomplete")
    for item in manifest["images"]:
        name = item["path"].rsplit("/", 1)[-1]
        _require("eval.manifest_images", name in image_by_name,
                 f"manifest references missing image {item['path']}")
        data = image_by_name[name]
        _require("eval.manifest_images", sha256_bytes(data) == item["sha256"]
                 and len(data) == item["bytes"], f"image binding mismatch for {name}")
        decoded = decode_png(data, name=name)
        _require("eval.image_geometry", decoded["width"] == item["width"]
                 and decoded["height"] == item["height"]
                 and item["width"] == expected["width"]
                 and item["height"] == expected["height"],
                 f"image dimensions mismatch for {name}")
        _require("eval.image_format", item["format"] == "PNG",
                 f"image format mismatch for {name}")
    checks.append({"id": "eval.image_decode", "status": "pass",
                   "detail": "all images decode as strict 8-bit RGB PNG"})

    manifest_rows = {item["row_id"]: item for item in manifest["rows"]}
    _require("eval.row_bindings", set(manifest_rows) == {row["row_id"] for row in rows},
             "manifest rows differ from the rows file")
    spec_by_id = {scene["scene_id"]: scene for scene in scenes}
    checks_by_family: dict[str, int] = {}
    for row in rows:
        row_id = row["row_id"]
        _require("eval.row_format", row.get("format") == EVAL_ROW_FORMAT,
                 f"bad row format for {row_id}")
        _require("eval.row_split", row["split"] == EVAL_SPLIT and row["role"] == EVAL_ROLE,
                 f"bad row role for {row_id}")
        _require("eval.row_namespace", row_id.startswith(config["eval_id"] + ":"),
                 f"row outside the eval namespace: {row_id}")
        _require("eval.row_sha256", row["row_sha256"] == _hash_of(row, "row_sha256"),
                 f"row self-hash mismatch for {row_id}")
        _require("eval.row_target", row["target"]["caption_sha256"]
                 == sha256_text(row["target"]["caption"]),
                 f"target hash mismatch for {row_id}")
        _require("eval.row_target", row["provenance"]["normalized_caption_sha256"]
                 == _normalized_caption_sha(row["target"]["caption"]),
                 f"normalized caption hash mismatch for {row_id}")
        _require("eval.row_rights", row["rights"] == config["rights"],
                 f"rights mismatch for {row_id}")
        _require("eval.row_provenance", row["provenance"]["kind"]
                 == config["provenance"]["kind"]
                 and row["provenance"]["external_images"] is False
                 and row["provenance"]["upstream_dataset"] is None
                 and row["provenance"]["third_party_content"] is False,
                 f"provenance mismatch for {row_id}")
        _require("eval.row_label", row["label_authority"]["human_semantic_review"] is False
                 and row["label_authority"]["model_inference"] is False
                 and row["label_authority"]["kind"] == config["label_authority"]["kind"],
                 f"label authority mismatch for {row_id}")
        scene = spec_by_id.get(row["scene_id"])
        _require("eval.row_scene", scene is not None
                 and scene["row_id"] == row_id, f"missing scene for {row_id}")
        _require("eval.row_scene", row["label_authority"]["scene_spec_sha256"]
                 == scene["scene_spec_sha256"], f"scene hash mismatch for {row_id}")
        _require("eval.agreement", row["target"]["caption"] == scene["caption"],
                 f"row caption differs from the scene caption for {row_id}")
        _require("eval.agreement", row["probes"] == scene["probes"],
                 f"row probes differ from the scene for {row_id}")
        _require("eval.row_scene", row["asset"]["image_sha256"]
                 == manifest_rows[row_id]["image_sha256"], "manifest image mismatch")
        binding = manifest_rows[row_id]
        _require("eval.row_bindings", binding["row_sha256"] == row["row_sha256"]
                 and binding["target_sha256"] == row["target"]["caption_sha256"]
                 and binding["image_sha256"] == row["asset"]["image_sha256"]
                 and binding["scene_spec_sha256"] == scene["scene_spec_sha256"],
                 f"manifest binding mismatch for {row_id}")
        name = row["observation"]["image_ref"].rsplit("/", 1)[-1]
        _require("eval.row_image", name in image_by_name,
                 f"missing image for {row_id}")
        decoded = decode_png(image_by_name[name], name=name)
        _validate_scene(scene, config, palette, decoded["pixels"], checks_by_family)
        _require("eval.agreement", row["scene_id"] in row["row_id"]
                 and row["group_key"] == row_id, f"row identity mismatch for {row_id}")

    _require("eval.families", checks_by_family == {
        family["id"]: expected["scenes_per_family"]
        for family in config["generation"]["families"]},
        f"family coverage mismatch: {checks_by_family}")
    checks.append({"id": "eval.scene_checks", "status": "pass",
                   "detail": "24 scenes pass per-pixel geometry, probes and caption checks"})
    return {"specs": specs, "rows": rows, "manifest": manifest, "seal": seal,
            "scenes": scenes, "images": image_by_name}


# ---------------------------------------------------------------------------
# Separation and admission


def validate_separation(lineage: Mapping[str, Any], evals: Mapping[str, Any],
                        config: Mapping[str, Any], *,
                        checks: list[dict[str, str]]) -> dict[str, Any]:
    known: dict[str, str] = {}

    def register(value: Any, label: str) -> None:
        if value:
            known.setdefault(str(value), label)

    for row in lineage["rows"]:
        parent = lineage["corpus"][row["parent"]["tranche_row_id"]]
        register(row["row_id"], "training_row")
        register(row["parent"]["tranche_row_id"], "training_parent")
        register(row["asset"]["image_sha256"], "training_image")
        register(row["target"]["caption_sha256"], "training_target")
        register(row["provenance"]["normalized_caption_sha256"], "training_caption")
        register(parent["split"]["group_key"], "training_group")
    for row in evals["rows"]:
        register(row["row_id"], "eval_row")
        register(row["group_key"], "eval_group")
        register(row["asset"]["image_sha256"], "eval_image")
        register(row["target"]["caption_sha256"], "eval_target")
        register(row["provenance"]["normalized_caption_sha256"], "eval_caption")
        register(row["target"]["caption"], "eval_caption_text")
    for row_id in lineage["exposed"]:
        register(row_id, "exposed_row")
    for item in lineage["exclusions"]:
        register(item["row_id"], "excluded_row")
    for corpus_row in lineage["corpus"].values():
        register(corpus_row["row_id"], "external_row")
        register(corpus_row["root_asset"]["image_sha256"], "external_image")
        register(sha256_text(corpus_row["target"]["caption"]), "external_target")

    _require("separation.exposed",
             not (set(row["row_id"] for row in evals["rows"])
                  & set(lineage["exposed"])),
             "an exposed row appears in the evaluation partition")
    collisions = []
    for row in evals["rows"]:
        for value, kind in (
            (row["row_id"], "row_id"),
            (row["group_key"], "group_key"),
            (row["asset"]["image_sha256"], "image_sha256"),
            (row["target"]["caption_sha256"], "target_sha256"),
            (row["provenance"]["normalized_caption_sha256"], "normalized_caption_sha256"),
        ):
            if value in known and not known[value].startswith("eval_"):
                collisions.append({"key": kind, "value": value,
                                   "conflict": known[value]})
    _require("separation.keys", not collisions,
             f"evaluation partition collides with training/external rows: {collisions}")
    _require("separation.splits",
             set(row["split"] for row in evals["rows"]) == {EVAL_SPLIT}
             and set(row["split"] for row in lineage["rows"]) == {TRAIN_ROLE}
             and EVAL_SPLIT != TRAIN_ROLE,
             "training and evaluation splits are not disjoint")
    exposure_bound = evals["seal"]["disjointness"]
    _require("separation.seal_disjointness",
             exposure_bound["training_lineage_manifest_sha256"]
             == lineage["manifest"]["manifest_sha256"]
             and exposure_bound["training_rows_sha256"]
             == lineage["manifest"]["rows_sha256"]
             and exposure_bound["exposure_incident_sha256"]
             == config["parents"]["exposure_incident"]["sha256"]
             and exposure_bound["collisions"] == []
             and tuple(exposure_bound["exposed_row_ids"]) == tuple(lineage["exposed"]),
             "seal disjointness receipt does not match the validated data")
    checks.append({"id": "separation", "status": "pass",
                   "detail": "training, external, exposed and eval keys are disjoint"})
    return {"known_keys": len(known), "collisions": [], "eval_rows": len(evals["rows"])}


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and \
        all(character in "0123456789abcdef" for character in value)


def verify_admission_receipt(receipt: Mapping[str, Any], *,
                             admission: Mapping[str, Any],
                             lineage_id: str,
                             lineage_config_sha256: str,
                             eval_config_sha256: str,
                             lineage_manifest_sha256: str,
                             rows_sha256: str,
                             exclusions_sha256: str,
                             eval_manifest_sha256: str,
                             eval_rows_sha256: str,
                             eval_seal_sha256: str,
                             eligible_row_ids: Sequence[str]) -> dict[str, Any]:
    """Verify a trusted local reviewed decision receipt (not a signature)."""
    digest_field = admission["self_digest_field"]
    _require("admission.fields",
             set(receipt) == set(admission["required_fields"]),
             "receipt fields differ from the declared schema")
    _require(f"admission.{digest_field}", _is_hex64(receipt.get(digest_field)),
             "receipt self-digest is missing")
    expected_digest = sha256_text(canonical_json(_strip(receipt, digest_field)))
    _require(f"admission.{digest_field}", expected_digest == receipt[digest_field],
             "receipt self-digest differs from its content")

    _require("admission.format", receipt.get("format") == RECEIPT_FORMAT,
             "unexpected receipt format")
    _require("admission.scope", receipt.get("scope") == RECEIPT_SCOPE,
             "receipt scope is not training_admission")
    _require("admission.trust",
             receipt.get("trust") == admission["trust_model"],
             "receipt does not declare a trusted local reviewed decision")
    _require("admission.lineage", receipt.get("lineage_id") == lineage_id,
             "receipt lineage mismatch")
    decision = receipt.get("decision")
    _require("admission.decision", decision in admission["decision_values"],
             f"unknown receipt decision {decision!r}")
    decided_by = receipt.get("decided_by")
    _require("admission.decided_by", isinstance(decided_by, str)
             and decided_by.strip(), "reviewer is required")
    decided_at = receipt.get("decided_at")
    _require("admission.decided_at", isinstance(decided_at, str)
             and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                              decided_at) is not None,
             "decided_at must be an ISO 8601 UTC timestamp")
    try:
        date.fromisoformat(decided_at[:10])
    except ValueError as error:
        raise GuardError("admission.decided_at", "invalid decided_at date") from error
    conditions = receipt.get("conditions")
    _require("admission.conditions", isinstance(conditions, list)
             and all(isinstance(item, str) and item.strip()
                     for item in conditions),
             "conditions must be a list of non-empty strings")

    bindings = receipt.get("bindings")
    _require("admission.bindings", isinstance(bindings, dict)
             and set(bindings) == set(admission["must_bind"]),
             "receipt bindings differ from the declared binding set")
    expected_bindings = {
        "lineage_config_sha256": lineage_config_sha256,
        "eval_config_sha256": eval_config_sha256,
        "lineage_manifest_sha256": lineage_manifest_sha256,
        "lineage_rows_sha256": rows_sha256,
        "lineage_exclusions_sha256": exclusions_sha256,
        "eval_manifest_sha256": eval_manifest_sha256,
        "eval_rows_sha256": eval_rows_sha256,
        "eval_seal_sha256": eval_seal_sha256,
        "validator_version": VALIDATOR_VERSION,
    }
    for key in admission["must_bind"]:
        _require("admission.bindings", bindings.get(key) == expected_bindings[key],
                 f"binding mismatch for {key}")

    listed = receipt.get("admitted_row_ids")
    _require("admission.rows", isinstance(listed, list)
             and tuple(sorted(listed)) == tuple(sorted(eligible_row_ids)),
             "receipt does not list exactly the nine eligible rows")
    _require("admission.rows", len(set(listed)) == len(listed),
             "receipt row ids are not unique")
    _require("admission.rows", set(listed) <= set(eligible_row_ids),
             "receipt lists a row that is not eligible")

    base = {
        "scope": RECEIPT_SCOPE,
        "trust": admission["trust_model"],
        "receipt_path": admission["receipt_path"],
        "receipt_sha256": receipt[digest_field],
        "decision": decision,
        "decided_by": decided_by,
        "decided_at": decided_at,
        "eligible_row_ids": list(eligible_row_ids),
    }
    if decision != admission["required_decision"]:
        return {"status": "not_admitted", "reason": "receipt_decision_reject",
                **base, "admitted_row_ids": []}
    return {"status": "admitted", **base, "admitted_row_ids": list(listed)}


# ---------------------------------------------------------------------------
# Unsigned admission evidence packet
#
# The packet is the deterministic, unsigned preparation for the trusted local
# reviewed receipt.  It carries every immutable receipt field plus the evidence
# bindings, and leaves exactly the reviewer-owned fields pending.  It is not a
# receipt: it has no decision, reviewer, timestamp or self-digest.


def _packet_bindings(*, lineage_config_sha256: str, eval_config_sha256: str,
                     lineage: Mapping[str, Any],
                     evals: Mapping[str, Any],
                     admission: Mapping[str, Any]) -> dict[str, str]:
    bindings = {
        "lineage_config_sha256": lineage_config_sha256,
        "eval_config_sha256": eval_config_sha256,
        "lineage_manifest_sha256": lineage["manifest"]["manifest_sha256"],
        "lineage_rows_sha256": lineage["manifest"]["rows_sha256"],
        "lineage_exclusions_sha256": lineage["manifest"]["exclusions_sha256"],
        "eval_manifest_sha256":
            evals["seal"]["sealed_artifacts"]["manifest_sha256"],
        "eval_rows_sha256": evals["seal"]["sealed_artifacts"]["rows_sha256"],
        "eval_seal_sha256": evals["seal"]["seal_sha256"],
        "validator_version": VALIDATOR_VERSION,
    }
    _require("admission.packet.bindings",
             set(bindings) == set(admission["must_bind"]),
             "packet bindings differ from the declared receipt bindings")
    return bindings


def _packet_evidence(*, config: Mapping[str, Any],
                     eval_config: Mapping[str, Any],
                     lineage: Mapping[str, Any],
                     evals: Mapping[str, Any]) -> dict[str, Any]:
    eligible = sorted(
        (
            {
                "row_id": row["row_id"],
                "group_key": row["group_key"],
                "parent_row_id": row["parent"]["tranche_row_id"],
                "row_sha256": row["row_sha256"],
                "image_sha256": row["asset"]["image_sha256"],
                "target_sha256": row["target"]["caption_sha256"],
            }
            for row in lineage["rows"]
        ),
        key=lambda item: item["row_id"],
    )
    exclusions = sorted(
        (
            {
                "row_id": row["row_id"],
                "role": row["role"],
                "recorded_verdict": row["recorded_verdict"],
                "reason_code": row["reason_code"],
                "training_eligible": row["training_eligible"],
            }
            for row in lineage["exclusions"]
        ),
        key=lambda item: item["row_id"],
    )
    reason_counts: dict[str, int] = defaultdict(int)
    for item in exclusions:
        reason_counts[item["reason_code"]] += 1
    exposed = sorted(lineage["exposed"])
    decision = lineage["decision"]
    exposure = lineage["exposure"]
    seal = evals["seal"]
    counts = {
        "rows": len(eligible),
        "exclusions": len(exclusions),
        "exposed": len(exposed),
        "parent_rows_accounted":
            config["selection"]["total_parent_rows_accounted"],
        "eval_rows": len(evals["rows"]),
        "eval_images": len(evals["images"]),
    }
    return {
        "lineage": {
            "lineage_id": config["lineage_id"],
            "status": config["status"],
            "manifest_sha256": lineage["manifest"]["manifest_sha256"],
            "rows_sha256": lineage["manifest"]["rows_sha256"],
            "exclusions_sha256": lineage["manifest"]["exclusions_sha256"],
            "counts": counts,
            "eligible_rows": eligible,
            "exclusions": exclusions,
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "exposure": {
                "incident_sha256":
                    config["parents"]["exposure_incident"]["sha256"],
                "status": exposure["status"],
                "exposed_row_ids": exposed,
                "excluded_from_training": True,
                "history_preserved": True,
                "relabeled": False,
                "untouched_eval_status_restored": False,
            },
            "parent_decision": {
                "decision": decision["decision"],
                "relabeled": False,
                "training_admitted_row_ids": None,
                "queue_sha256": decision["queue_sha256"],
            },
        },
        "evaluation": {
            "eval_id": eval_config["eval_id"],
            "role": seal["role"],
            "split": seal["split"],
            "manifest_sha256":
                seal["sealed_artifacts"]["manifest_sha256"],
            "rows_sha256": seal["sealed_artifacts"]["rows_sha256"],
            "seal_sha256": seal["seal_sha256"],
            "scene_specs_sha256":
                seal["sealed_artifacts"]["scene_specs_sha256"],
            "images_aggregate_sha256":
                seal["sealed_artifacts"]["images_aggregate_sha256"],
            "counts": {
                "rows": len(evals["rows"]),
                "images": len(evals["images"]),
                "families": len({row["family"] for row in evals["rows"]}),
            },
            "diagnostic_scope": {
                **EVAL_DIAGNOSTIC_FLAGS,
                "rows": len(evals["rows"]),
                "images": len(evals["images"]),
                "permitted_claim":
                    "fixed synthetic scene families only: image-conditioned "
                    "grounding and caption agreement under programmatic labels",
                "forbidden_claims": [
                    "natural-image caption quality",
                    "VLM generality, robustness or fairness",
                    "zero-shot external transfer",
                    "benchmark superiority",
                ],
            },
            "sealed_before_any_model_use":
                seal["attestations"]["sealed_before_any_model_use"],
            "model_executions_at_seal":
                seal["attestations"]["model_executions_at_seal"],
            "disjoint_from_training_and_exposed":
                seal["attestations"]["disjoint_from_training_and_exposed_rows"],
        },
        "admission": {
            "default_status": config["admission"]["default_status"],
            "requires_separate_reviewed_receipt":
                config["admission"]["requires_separate_reviewed_receipt"],
            "receipt_path": config["admission"]["receipt_path"],
            "receipt_created": False,
            "training_admitted_row_ids": [],
        },
    }


def _packet_payload(*, admission: Mapping[str, Any], config: Mapping[str, Any],
                    eval_config: Mapping[str, Any],
                    lineage: Mapping[str, Any], evals: Mapping[str, Any],
                    bindings: Mapping[str, str]) -> dict[str, Any]:
    receipt_fields = {
        "format": RECEIPT_FORMAT,
        "scope": RECEIPT_SCOPE,
        "trust": admission["trust_model"],
        "lineage_id": config["lineage_id"],
        "decision": None,
        "decided_by": None,
        "decided_at": None,
        "bindings": dict(bindings),
        "admitted_row_ids": sorted(row["row_id"] for row in lineage["rows"]),
        "conditions": [],
        "receipt_sha256": None,
    }
    _require("admission.packet.receipt_fields",
             set(receipt_fields) == set(admission["required_fields"]),
             "packet receipt fields differ from the declared receipt schema")
    return {
        "format": PACKET_FORMAT,
        "created_on": config["created_on"],
        "status": PACKET_STATUS,
        "not_a_receipt": True,
        "unsigned": True,
        "signature_present": False,
        "lineage_id": config["lineage_id"],
        "receipt_format": RECEIPT_FORMAT,
        "receipt_path": admission["receipt_path"],
        "receipt_fields": receipt_fields,
        "pending_reviewer_fields": list(PACKET_PENDING_FIELDS),
        "evidence": _packet_evidence(config=config, eval_config=eval_config,
                                     lineage=lineage, evals=evals),
        "reviewer_instructions": [
            "Review the lineage manifest, rows/exclusions hashes, the preserved "
            "exposure incident and the sealed diagnostic evaluation.",
            "Only a named reviewer may copy receipt_fields to the receipt path, "
            "fill decision, decided_by, decided_at and conditions, then compute "
            "receipt_sha256 over canonical JSON without that field.",
            "Never edit the nine eligible row ids, any binding, or the "
            "exposure/exclusion evidence.",
            "The receipt is a trusted local review decision, not a "
            "cryptographic signature, and grants no benchmark claim.",
        ],
        "non_claims": dict(PACKET_NON_CLAIMS),
    }


def build_admission_packet(lineage_config_path: str | Path,
                           eval_config_path: str | Path, *,
                           project_root: str | Path = ROOT) -> dict[str, Any]:
    """Prepare, but never sign, the reviewed-receipt evidence packet."""
    project_root = Path(project_root)
    lineage_config_path = Path(lineage_config_path)
    eval_config_path = Path(eval_config_path)
    if not lineage_config_path.is_absolute():
        lineage_config_path = project_root / lineage_config_path
    if not eval_config_path.is_absolute():
        eval_config_path = project_root / eval_config_path
    checks: list[dict[str, str]] = []
    config = load_json(lineage_config_path)
    eval_config = load_json(eval_config_path)
    lineage = validate_lineage(config, project_root=project_root, checks=checks)
    evals = validate_eval(eval_config, project_root=project_root,
                          lineage=lineage, checks=checks)
    validate_separation(lineage, evals, config, checks=checks)
    admission = config["admission"]
    bindings = _packet_bindings(
        lineage_config_sha256=sha256_file(lineage_config_path),
        eval_config_sha256=sha256_file(eval_config_path),
        lineage=lineage, evals=evals, admission=admission)
    return _packet_payload(admission=admission, config=config,
                           eval_config=eval_config, lineage=lineage,
                           evals=evals, bindings=bindings)


def _first_difference(left: Any, right: Any, path: str = "") -> str | None:
    if type(left) is not type(right):
        return path or "<root>"
    if isinstance(left, dict):
        if set(left) != set(right):
            missing = sorted(set(right) - set(left))
            extra = sorted(set(left) - set(right))
            return (f"{path or '<root>'}: missing={missing} extra={extra}")
        for key in sorted(left):
            found = _first_difference(left[key], right[key],
                                      f"{path}.{key}" if path else key)
            if found is not None:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path or '<root>'}: length {len(left)} != {len(right)}"
        for index, (first, second) in enumerate(zip(left, right)):
            found = _first_difference(first, second, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    return None if left == right else (path or "<root>")


def verify_admission_packet(packet: Mapping[str, Any], *,
                            admission: Mapping[str, Any],
                            config: Mapping[str, Any],
                            eval_config: Mapping[str, Any],
                            lineage: Mapping[str, Any],
                            evals: Mapping[str, Any],
                            lineage_config_sha256: str,
                            eval_config_sha256: str) -> dict[str, Any]:
    """Fail closed unless the packet is the current unsigned preparation."""
    _require("admission.packet.fields", isinstance(packet, dict)
             and set(packet) == PACKET_FIELDS,
             "packet fields differ from the declared unsigned schema")
    _require("admission.packet.format", packet.get("format") == PACKET_FORMAT,
             "unexpected admission packet format")
    _require("admission.packet.status", packet.get("status") == PACKET_STATUS,
             "packet is not in prepared-unsigned state")
    _require("admission.packet.unsigned",
             packet.get("not_a_receipt") is True
             and packet.get("unsigned") is True
             and packet.get("signature_present") is False,
             "packet is signed or claims to be a receipt")
    _require("admission.packet.lineage",
             packet.get("lineage_id") == config["lineage_id"],
             "packet lineage mismatch")
    _require("admission.packet.receipt",
             packet.get("receipt_format") == RECEIPT_FORMAT
             and packet.get("receipt_path") == admission["receipt_path"],
             "packet receipt target mismatch")
    _require("admission.packet.pending",
             tuple(packet.get("pending_reviewer_fields", ()))
             == PACKET_PENDING_FIELDS,
             "pending reviewer fields differ from the declared set")
    receipt_fields = packet.get("receipt_fields")
    _require("admission.packet.receipt_fields", isinstance(receipt_fields, dict)
             and set(receipt_fields) == set(admission["required_fields"]),
             "packet does not carry exactly the receipt fields")
    for field in PACKET_PENDING_FIELDS:
        value = receipt_fields.get(field)
        if field == "conditions":
            _require("admission.packet.unsigned",
                     value == [], "packet must not pre-fill conditions")
        else:
            _require("admission.packet.unsigned", value is None,
                     f"packet must leave {field} unsigned")
    _require("admission.packet.unsigned",
             receipt_fields.get("receipt_sha256") is None
             and packet.get("non_claims") == PACKET_NON_CLAIMS,
             "packet carries a self-digest or claims admission")

    bindings = _packet_bindings(
        lineage_config_sha256=lineage_config_sha256,
        eval_config_sha256=eval_config_sha256,
        lineage=lineage, evals=evals, admission=admission)
    expected = _packet_payload(admission=admission, config=config,
                               eval_config=eval_config, lineage=lineage,
                               evals=evals, bindings=bindings)
    difference = _first_difference(packet, expected)
    _require("admission.packet.content", difference is None,
             f"packet differs from the current artifacts at {difference}")
    return {
        "status": "prepared_unsigned",
        "path": PACKET_PATH,
        "format": PACKET_FORMAT,
        "unsigned": True,
        "receipt_path": admission["receipt_path"],
        "eligible_row_ids": list(receipt_fields["admitted_row_ids"]),
        "exclusions": len(packet["evidence"]["lineage"]["exclusions"]),
        "exposed_rows": len(packet["evidence"]["lineage"]["exposure"][
            "exposed_row_ids"]),
    }


def resolve_packet(lineage: Mapping[str, Any], evals: Mapping[str, Any],
                   config: Mapping[str, Any], eval_config: Mapping[str, Any], *,
                   lineage_config_path: Path, eval_config_path: Path,
                   project_root: Path,
                   checks: list[dict[str, str]]) -> dict[str, Any]:
    path = project_root / PACKET_PATH
    if not path.is_file():
        checks.append({"id": "admission.packet", "status": "pass",
                       "detail": "no unsigned packet present"})
        return {"status": "absent", "path": PACKET_PATH}
    packet = load_json(path)
    info = verify_admission_packet(
        packet, admission=config["admission"], config=config,
        eval_config=eval_config, lineage=lineage, evals=evals,
        lineage_config_sha256=sha256_file(lineage_config_path),
        eval_config_sha256=sha256_file(eval_config_path))
    info["packet_sha256"] = sha256_file(path)
    checks.append({"id": "admission.packet", "status": "pass",
                   "detail": "unsigned packet matches the validated artifacts"})
    return info


def resolve_admission(lineage: Mapping[str, Any], evals: Mapping[str, Any],
                      config: Mapping[str, Any], *,
                      lineage_config_path: Path, eval_config_path: Path,
                      project_root: Path,
                      checks: list[dict[str, str]]) -> dict[str, Any]:
    admission = config["admission"]
    _require("admission.default",
             admission["default_status"] == "not_admitted"
             and config["non_claims"]["training_admitted"] is False,
             "config does not default to not_admitted")
    _require("admission.trust_model",
             admission["trust_model"]
             == "trusted_local_reviewed_receipt_not_a_signature",
             "admission must be a reviewed local receipt, not a signature")
    receipt_path = project_root / admission["receipt_path"]
    data_roots = [project_root / config["outputs"]["root"],
                  project_root / config["eval_partition"]["data_root"]]
    _require("admission.location",
             not any(root == receipt_path.parent or root in receipt_path.parents
                     for root in data_roots),
             "an admission receipt must live outside generated data roots")
    generators = [
        project_root / "scripts/build_caption_lineage_20261027.py",
        project_root / "scripts/build_synthetic_scene_eval_20261027.py",
    ]
    receipt_name = receipt_path.name
    for generator in generators:
        _require("admission.generator", generator.is_file(),
                 f"missing generator {generator.name}")
        _require("admission.generator", receipt_name not in
                 generator.read_text(encoding="utf-8"),
                 "a generator references the admission receipt path")
    checks.append({"id": "admission.generator", "status": "pass",
                   "detail": "generators never create admission receipts"})

    eligible = [row["row_id"] for row in lineage["rows"]]
    if not receipt_path.is_file():
        result = {"status": "not_admitted", "reason": "receipt_absent",
                  "receipt_path": admission["receipt_path"],
                  "eligible_row_ids": eligible, "admitted_row_ids": []}
        checks.append({"id": "admission.status", "status": "pass",
                       "detail": "no receipt; training remains not admitted"})
        return result
    try:
        receipt = load_json(receipt_path)
        result = verify_admission_receipt(
            receipt,
            admission=admission,
            lineage_id=config["lineage_id"],
            lineage_config_sha256=sha256_file(lineage_config_path),
            eval_config_sha256=sha256_file(eval_config_path),
            lineage_manifest_sha256=lineage["manifest"]["manifest_sha256"],
            rows_sha256=lineage["manifest"]["rows_sha256"],
            exclusions_sha256=lineage["manifest"]["exclusions_sha256"],
            eval_manifest_sha256=evals["seal"]["sealed_artifacts"]["manifest_sha256"],
            eval_rows_sha256=evals["seal"]["sealed_artifacts"]["rows_sha256"],
            eval_seal_sha256=evals["seal"]["seal_sha256"],
            eligible_row_ids=eligible,
        )
    except GuardError as error:
        result = {"status": "blocked", "reason": f"{error.check}: {error.message}",
                  "receipt_path": admission["receipt_path"],
                  "eligible_row_ids": eligible, "admitted_row_ids": []}
        checks.append({"id": "admission.status", "status": "pass",
                       "detail": "invalid or stale receipt refused; training not admitted"})
        return result
    detail = ("reviewed receipt rejects training admission"
              if result["status"] != "admitted"
              else "valid reviewed local receipt; training admission recorded")
    checks.append({"id": "admission.status", "status": "pass", "detail": detail})
    return result


# ---------------------------------------------------------------------------
# Orchestration


def validate(lineage_config_path: str | Path, eval_config_path: str | Path, *,
             project_root: str | Path = ROOT,
             as_of: str | None = None,
             require_admission: bool = False) -> dict[str, Any]:
    project_root = Path(project_root)
    lineage_config_path = Path(lineage_config_path)
    eval_config_path = Path(eval_config_path)
    if not lineage_config_path.is_absolute():
        lineage_config_path = project_root / lineage_config_path
    if not eval_config_path.is_absolute():
        eval_config_path = project_root / eval_config_path
    as_of = as_of or date.today().isoformat()
    checks: list[dict[str, str]] = []
    try:
        lineage_config = load_json(lineage_config_path)
        eval_config = load_json(eval_config_path)
        lineage = validate_lineage(lineage_config, project_root=project_root,
                                   checks=checks)
        evals = validate_eval(eval_config, project_root=project_root,
                              lineage=lineage, checks=checks)
        separation = validate_separation(lineage, evals, lineage_config,
                                         checks=checks)
        admission = resolve_admission(
            lineage, evals, lineage_config,
            lineage_config_path=lineage_config_path,
            eval_config_path=eval_config_path,
            project_root=project_root, checks=checks)
        packet = resolve_packet(
            lineage, evals, lineage_config, eval_config,
            lineage_config_path=lineage_config_path,
            eval_config_path=eval_config_path,
            project_root=project_root, checks=checks)
    except GuardError as error:
        report = {
            "format": VALIDATION_FORMAT,
            "validator_version": VALIDATOR_VERSION,
            "status": "invalid",
            "failed_check": error.check,
            "error": error.message,
            "checks": checks,
            "as_of": as_of,
            "non_claims": {
                "model_execution": False,
                "training_admitted": False,
                "native_execution": False,
            },
        }
        if require_admission:
            raise
        return report
    report = {
        "format": VALIDATION_FORMAT,
        "validator_version": VALIDATOR_VERSION,
        "status": "valid",
        "checks": checks,
        "lineage": {
            "lineage_id": lineage_config["lineage_id"],
            "manifest_sha256": lineage["manifest"]["manifest_sha256"],
            "rows_sha256": lineage["manifest"]["rows_sha256"],
            "exclusions_sha256": lineage["manifest"]["exclusions_sha256"],
            "rows": len(lineage["rows"]),
            "exclusions": len(lineage["exclusions"]),
            "exposed_rows": len(lineage["exposed"]),
        },
        "eval": {
            "eval_id": eval_config["eval_id"],
            "seal_sha256": evals["seal"]["seal_sha256"],
            "manifest_sha256": evals["seal"]["sealed_artifacts"]["manifest_sha256"],
            "rows_sha256": evals["seal"]["sealed_artifacts"]["rows_sha256"],
            "scene_specs_sha256": evals["seal"]["sealed_artifacts"]["scene_specs_sha256"],
            "images_aggregate_sha256":
                evals["seal"]["sealed_artifacts"]["images_aggregate_sha256"],
            "rows": len(evals["rows"]),
            "images": len(evals["images"]),
        },
        "separation": separation,
        "admission": admission,
        "admission_packet": packet,
        "as_of": as_of,
        "non_claims": {
            "model_execution": False,
            "training_admitted": admission["status"] == "admitted",
            "native_execution": False,
            "benchmark_quality": "none",
        },
    }
    if require_admission and admission["status"] != "admitted":
        raise GuardError("admission.required",
                         "a valid reviewed admission receipt is required")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-config", default=str(
        ROOT / "configs/caption_lineage_correction_20261027.json"))
    parser.add_argument("--eval-config", default=str(
        ROOT / "configs/synthetic_scene_eval_20261027.json"))
    parser.add_argument("--require-admission", action="store_true",
                        help="exit non-zero unless a valid reviewed receipt admits training")
    parser.add_argument("--as-of", default=None,
                        help="ISO date recorded in the validation report (default: today)")
    parser.add_argument("--write-packet", nargs="?", const=PACKET_PATH,
                        default=None,
                        help="write the deterministic unsigned admission packet "
                             "to this path (default: the prepared packet path) "
                             "after validation passes")
    args = parser.parse_args(argv)
    try:
        if args.write_packet:
            target = Path(args.write_packet)
            if not target.is_absolute():
                target = ROOT / target
            packet = build_admission_packet(args.lineage_config,
                                            args.eval_config)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
        report = validate(args.lineage_config, args.eval_config,
                          as_of=args.as_of, require_admission=args.require_admission)
        status = 0 if report["status"] == "valid" else 1
    except GuardError as error:
        report = {
            "format": VALIDATION_FORMAT,
            "validator_version": VALIDATOR_VERSION,
            "status": "invalid",
            "failed_check": error.check,
            "error": error.message,
            "as_of": args.as_of,
            "admission": {"status": "not_admitted", "reason": "validation_failed"},
        }
        status = 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
