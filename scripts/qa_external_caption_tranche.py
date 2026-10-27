#!/usr/bin/env python3
"""Deterministic CPU-only asset QA for the pinned external caption tranche.

The tool is deliberately offline.  It re-reads the frozen intake configuration,
tranche manifest, dataset card, annotation/metadata prefixes, central directory
and every local image, then verifies the pins byte-for-byte.  It re-runs the
structural checks (format, geometry, per-image SHA-256, annotation/row/target
hashes, provenance, terms, duplicate groups, split/label separation, reserved
family exclusion and sealed final-evaluation rows) and decodes every image.

Two claims are kept strictly apart:

* machine checks prove byte-level/structural consistency only;
* a human must view every queued image and record a semantic verdict.

The QA report and review queue never assert caption/image agreement.  Training
admission remains ``pending_human_semantic_review`` until a human-authored
``admission_decision.json`` covers every queued row, every training candidate
is marked ``match``, and all machine checks pass.  The eight sealed
``final_eval_untouched`` rows and the development validation rows are never
eligible for the training-admitted output.

Usage::

    .venv-vision/bin/python scripts/qa_external_caption_tranche.py \
        --config configs/external_caption_asset_qa_v1.json

    # Review reports/review_queue.json, copy admission_decision.template.json
    # to admission_decision.json, fill every semantic verdict, then:
    .venv-vision/bin/python scripts/qa_external_caption_tranche.py \
        --config configs/external_caption_asset_qa_v1.json

Pillow is required for real decode verification.  If it is unavailable the
decode checks are recorded as blocked and admission is refused; the script
never silently degrades to structural-only evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from replica_lab.external_intake import (  # noqa: E402
    EVAL_ROLE,
    MANIFEST_FORMAT,
    ROLES,
    ROW_FORMAT,
    TRANCHE_FORMAT,
    ExternalIntakeError,
    assert_row_label_separation,
    inspect_image_bytes,
    leakage_report,
    normalize_caption,
    normalize_prompt,
    normalize_text,
    parse_json_array_prefix,
)
from replica_lab.vision_qualification import canonical_json, sha256_text  # noqa: E402

QA_CONFIG_FORMAT = "replica.external_caption_asset_qa_config.v1"
QA_REPORT_FORMAT = "replica.external_caption_asset_qa_report.v1"
REVIEW_QUEUE_FORMAT = "replica.external_caption_asset_review_queue.v1"
DECISION_FORMAT = "replica.external_caption_asset_admission_decision.v1"

PENDING = "pending_human_semantic_review"
REJECT = "reject"
ADMIT = "admit_machine_checked_train_rows"
VERDICTS = ("match", "mismatch", "unclear")
SPLIT_POLICY = "complete root asset and all derivatives share one split"

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ISO_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)

SECTION_BY_ROLE = {
    "train": "training_candidates",
    "development": "validation_only",
    "final_eval_untouched": "sealed_final_eval",
}
SCOPE_BY_ROLE = {
    "train": "training_candidate",
    "development": "validation_only",
    "final_eval_untouched": "sealed_evaluation_only",
}
TRAINING_OUTPUT_NAME = "training_admitted.jsonl"


class QAError(RuntimeError):
    """A single QA check failed or was blocked."""

    def __init__(self, check_id: str, detail: str, *, blocked: bool = False):
        super().__init__(f"{check_id}: {detail}")
        self.check_id = check_id
        self.detail = detail
        self.blocked = blocked


class DecodeUnavailable(QAError):
    """The required image decoder is not installed."""


class DecodeError(QAError):
    """The image decoder rejected the bytes."""


@dataclass
class Check:
    check_id: str
    scope: str
    passed: bool
    detail: str = ""
    blocked: bool = False


class Collector:
    """Ordered check ledger; every check is emitted, not only failures."""

    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        check_id: str,
        scope: str,
        passed: bool,
        detail: str = "",
        *,
        blocked: bool = False,
    ) -> bool:
        self.checks.append(
            Check(
                check_id,
                scope,
                bool(passed),
                "" if passed else str(detail),
                bool(blocked and not passed),
            )
        )
        return bool(passed)

    def require(
        self,
        check_id: str,
        scope: str,
        condition: bool,
        detail: str,
        *,
        blocked: bool = False,
    ) -> bool:
        return self.add(check_id, scope, condition, detail, blocked=blocked)

    def run(
        self,
        check_id: str,
        scope: str,
        function: Callable[[], Any],
        *,
        blocked_on_error: bool = False,
    ) -> Any:
        try:
            value = function()
        except QAError as error:
            self.add(check_id, scope, False, error.detail, blocked=error.blocked)
            return None
        except ExternalIntakeError as error:
            self.add(check_id, scope, False, str(error), blocked=blocked_on_error)
            return None
        except Exception as error:  # noqa: BLE001 - every failure must be evidence
            self.add(
                check_id,
                scope,
                False,
                f"{type(error).__name__}: {error}",
                blocked=blocked_on_error,
            )
            return None
        detail = value if isinstance(value, str) else ""
        self.add(check_id, scope, True, detail)
        return value

    def merge(self, other: "Collector") -> None:
        self.checks.extend(other.checks)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: Any) -> str:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def check_record(check: Check) -> dict[str, Any]:
    return {
        "id": check.check_id,
        "scope": check.scope,
        "passed": check.passed,
        "detail": check.detail,
        "blocked": check.blocked,
    }


def require(condition: bool, check_id: str, detail: str, *, blocked: bool = False) -> None:
    if not condition:
        raise QAError(check_id, detail, blocked=blocked)


def resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def display_path(project_root: Path, path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_json_file(path: Path, check_id: str) -> Any:
    if not path.is_file():
        raise QAError(check_id, f"missing {path}", blocked=True)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QAError(check_id, f"{path} is not valid JSON: {error}") from error


def load_jsonl_file(path: Path, check_id: str) -> list[Any]:
    if not path.is_file():
        raise QAError(check_id, f"missing {path}", blocked=True)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise QAError(check_id, f"{path} is not UTF-8: {error}") from error
    rows: list[Any] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise QAError(check_id, f"{path}:{number} is not valid JSON: {error}") from error
    return rows


def conversation_pair(raw: Mapping[str, Any]) -> tuple[str, str]:
    conversations = raw.get("conversations")
    require(
        isinstance(conversations, list) and len(conversations) >= 2,
        "annotation.conversations",
        "annotation row needs at least two conversation turns",
    )
    human, gpt = conversations[0], conversations[1]
    require(
        isinstance(human, Mapping) and human.get("from") == "human",
        "annotation.conversations",
        "first conversation turn is not a human prompt",
    )
    require(
        isinstance(gpt, Mapping) and gpt.get("from") == "gpt",
        "annotation.conversations",
        "second conversation turn is not a gpt answer",
    )
    prompt, answer = human.get("value"), gpt.get("value")
    require(
        isinstance(prompt, str) and isinstance(answer, str),
        "annotation.conversations",
        "conversation values must be strings",
    )
    return prompt, answer


def image_aggregate(images_dir: Path) -> tuple[list[Path], int, str]:
    files = sorted(path for path in images_dir.iterdir() if path.is_file())
    digest = hashlib.sha256()
    total = 0
    for path in files:
        data = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0" + data)
        total += len(data)
    return files, total, digest.hexdigest()


def decode_with_pillow(data: bytes) -> dict[str, Any]:
    try:
        import PIL  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as error:  # pragma: no cover - exercised when Pillow absent
        raise DecodeUnavailable(
            "image.decode",
            "pillow_unavailable: use .venv-vision/bin/python or install the vision overlay",
            blocked=True,
        ) from error
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return {
                "verified": True,
                "decoder": f"pillow-{getattr(PIL, '__version__', 'unknown')}",
                "format": image.format,
                "mode": image.mode,
                "width": image.size[0],
                "height": image.size[1],
                "channels": len(image.getbands()),
            }
    except Exception as error:  # noqa: BLE001 - Pillow raises many classes
        raise DecodeError(
            "image.decode", f"{type(error).__name__}: {error}"
        ) from error


def validate_qa_config(config: Mapping[str, Any]) -> None:
    require(
        isinstance(config, Mapping) and config.get("format") == QA_CONFIG_FORMAT,
        "qa_config.format",
        f"expected {QA_CONFIG_FORMAT}",
    )
    for key in (
        "qa_id",
        "created_on",
        "data_root",
        "report_dir",
        "source",
        "pins",
        "expected",
        "semantic_review",
    ):
        require(key in config, f"qa_config.{key}", f"missing config field {key!r}")
    expected = config["expected"]
    for key in (
        "rows",
        "root_groups",
        "roles",
        "training_role",
        "validation_roles",
        "sealed_role",
        "annotation_rows",
        "metadata_rows",
        "metadata_candidates",
        "rejections_rows",
        "preprocessing_version",
        "sampling_seed",
        "sampling_order",
        "split_policy",
        "image_constraints",
        "row_terms",
        "terms",
        "reserved_patterns",
        "reserved_families",
        "leakage",
        "blip_caption_equals_target",
        "caption_min_chars",
        "caption_max_chars",
        "annotation_source_name",
        "metadata_source_name",
    ):
        require(key in expected, f"qa_config.expected.{key}", f"missing expected field {key!r}")
    require(
        set(expected["roles"]) == set(ROLES),
        "qa_config.expected.roles",
        f"roles must be exactly {list(ROLES)}",
    )
    require(
        expected["sealed_role"] == EVAL_ROLE,
        "qa_config.expected.sealed_role",
        f"sealed role must be {EVAL_ROLE}",
    )
    require(
        expected["training_role"] in ROLES
        and expected["training_role"] != expected["sealed_role"]
        and expected["training_role"] not in expected["validation_roles"]
        and expected["sealed_role"] not in expected["validation_roles"],
        "qa_config.expected.scope",
        "training, validation and sealed roles must be disjoint",
    )
    review = config["semantic_review"]
    require(
        review.get("required") is True,
        "qa_config.semantic_review.required",
        "semantic review must be required",
    )
    require(
        set(review.get("allowed_decisions", [])) == {PENDING, REJECT, ADMIT},
        "qa_config.semantic_review.allowed_decisions",
        f"allowed decisions must be {[PENDING, REJECT, ADMIT]}",
    )
    require(
        tuple(review.get("required_verdicts", [])) == VERDICTS,
        "qa_config.semantic_review.required_verdicts",
        f"required verdicts must be {list(VERDICTS)}",
    )
    require(
        review.get("sealed_semantic_review") == "prohibited",
        "qa_config.semantic_review.sealed_semantic_review",
        "sealed final-eval semantic review must be prohibited",
    )
    require(
        set(review.get("reviewable_sections", []))
        == {"training_candidates", "validation_only"},
        "qa_config.semantic_review.reviewable_sections",
        "only training and validation sections may be reviewable",
    )
    require(
        review.get("require_all_nonsealed_queue_entries_reviewed") is True,
        "qa_config.semantic_review.require_all_nonsealed_queue_entries_reviewed",
        "all non-sealed queue entries must be reviewed before admission",
    )
    reviewable_expected = expected["roles"][expected["training_role"]] + sum(
        expected["roles"][role] for role in expected["validation_roles"]
    )
    require(
        review.get("reviewable_queue_entries") == reviewable_expected,
        "qa_config.semantic_review.reviewable_queue_entries",
        f"expected {reviewable_expected} reviewable entries",
    )
    require(
        review.get("sealed_queue_entries")
        == expected["roles"][expected["sealed_role"]],
        "qa_config.semantic_review.sealed_queue_entries",
        "sealed queue entry count differs from the sealed role count",
    )


def load_qa_config(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), "qa_config.file", f"missing {path}", blocked=True)
    raw = path.read_bytes()
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QAError("qa_config.json", f"{path} is not valid JSON: {error}") from error
    validate_qa_config(config)
    return config, sha256_bytes(raw)


# ---------------------------------------------------------------------------
# Pins and tranche loading
# ---------------------------------------------------------------------------


def read_pinned_file(
    project_root: Path,
    pin: Mapping[str, Any],
    collector: Collector,
    check_id: str,
    observed: dict[str, Any],
    *,
    scope: str = "pins",
) -> bytes | None:
    path = resolve(project_root, pin["path"])
    if not collector.require(
        f"{check_id}.exists", scope, path.is_file(), f"missing {path}", blocked=True
    ):
        return None
    data = path.read_bytes()
    actual_sha = sha256_bytes(data)
    observed[check_id] = {
        "path": display_path(project_root, path),
        "sha256": actual_sha,
        "bytes": len(data),
    }
    if "bytes" in pin:
        collector.require(
            f"{check_id}.bytes",
            scope,
            len(data) == pin["bytes"],
            f"{len(data)} != pinned {pin['bytes']}",
        )
    collector.require(
        f"{check_id}.sha256",
        scope,
        actual_sha == pin["sha256"],
        f"{actual_sha} != pinned {pin['sha256']}",
    )
    return data


def read_prefix(
    project_root: Path,
    pin: Mapping[str, Any],
    collector: Collector,
    check_id: str,
    observed: dict[str, Any],
) -> tuple[list[dict[str, Any]], bytes | None]:
    data = read_pinned_file(project_root, pin, collector, check_id, observed)
    if data is None:
        return [], None
    try:
        rows = parse_json_array_prefix(data, pin["complete_rows"])
    except Exception as error:  # noqa: BLE001 - parser errors must be evidence
        collector.add(f"{check_id}.parse", "pins", False, f"{type(error).__name__}: {error}")
        return [], data
    collector.require(
        f"{check_id}.complete_rows",
        "pins",
        len(rows) == pin["complete_rows"],
        f"{len(rows)} != pinned {pin['complete_rows']}",
    )
    window_sha = sha256_text(canonical_json(rows[: pin["complete_rows"]]))
    collector.require(
        f"{check_id}.window_sha256",
        "pins",
        window_sha == pin["window_sha256"],
        f"{window_sha} != pinned {pin['window_sha256']}",
    )
    return rows, data


def find_duplicates(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, dict[str, list[str]]] = {
        "image_sha256": defaultdict(list),
        "caption": defaultdict(list),
        "root_group": defaultdict(list),
        "row_id": defaultdict(list),
        "upstream_url": defaultdict(list),
    }
    for row in rows:
        row_id = str(row.get("row_id", "<missing>"))
        try:
            groups["image_sha256"][row["root_asset"]["image_sha256"]].append(row_id)
            groups["caption"][row["target"]["caption"].casefold()].append(row_id)
            groups["root_group"][row["split"]["group_key"]].append(row_id)
            groups["row_id"][row_id].append(row_id)
            groups["upstream_url"][row["root_asset"]["upstream_url"]].append(row_id)
        except (KeyError, TypeError):
            groups["row_id"][row_id].append(row_id)
    return {
        kind: [
            {"key": key, "row_ids": sorted(ids)}
            for key, ids in sorted(entries.items())
            if len(ids) > 1
        ]
        for kind, entries in groups.items()
    }


def assert_no_duplicates(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    duplicates = find_duplicates(rows)
    collisions = {kind: values for kind, values in duplicates.items() if values}
    if collisions:
        raise QAError("tranche.duplicates", json.dumps(collisions, sort_keys=True))
    return duplicates


def find_reserved_family_matches(
    rows: Sequence[Mapping[str, Any]],
    patterns: Sequence[str],
    reserved_families: Sequence[str],
) -> list[dict[str, str]]:
    compiled = [(pattern, re.compile(pattern, re.IGNORECASE)) for pattern in patterns]
    matches: list[dict[str, str]] = []
    for row in rows:
        try:
            root = row["root_asset"]
            source = row["source"]
        except (KeyError, TypeError):
            continue
        fields = {
            "upstream_url": str(root.get("upstream_url", "")),
            "upstream_image_path": str(root.get("upstream_image_path", "")),
            "root_image_id": str(root.get("root_image_id", "")),
            "source_id": str(source.get("id", "")),
            "source_family": str(source.get("family", "")),
        }
        for field, value in fields.items():
            for pattern, regex in compiled:
                if regex.search(value):
                    matches.append(
                        {
                            "row_id": str(row.get("row_id", "<missing>")),
                            "field": field,
                            "value": value,
                            "pattern": pattern,
                        }
                    )
        if str(source.get("family", "")) in reserved_families:
            matches.append(
                {
                    "row_id": str(row.get("row_id", "<missing>")),
                    "field": "source_family",
                    "value": str(source.get("family", "")),
                    "pattern": "<reserved family>",
                }
            )
    return matches


def assert_leakage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        leakage = leakage_report(rows)
    except ExternalIntakeError as error:
        raise QAError("tranche.leakage", str(error)) from error
    expected = {
        "root_groups_disjoint": True,
        "image_hashes_disjoint": True,
        "caption_hashes_disjoint": True,
        "observation_caption_leakage_cross_split": 0,
        "row_ids_unique": True,
        "label_separation_verified": True,
        "eval_vs_non_eval_image_overlap": 0,
        "eval_vs_non_eval_caption_overlap": 0,
    }
    if leakage != expected:
        raise QAError("tranche.leakage", f"unexpected leakage report {leakage!r}")
    return leakage


def role_manifest_seal(manifest: Mapping[str, Any]) -> str:
    rows = manifest["rows"]
    return sha256_text(
        canonical_json(
            {
                "row_ids": [row["row_id"] for row in rows],
                "image_sha256": [row["image_sha256"] for row in rows],
                "target_sha256": [row["target_sha256"] for row in rows],
            }
        )
    )


# ---------------------------------------------------------------------------
# Per-row verification
# ---------------------------------------------------------------------------


def verify_row(
    row: Mapping[str, Any],
    *,
    project_root: Path,
    data_root: Path,
    config: Mapping[str, Any],
    annotation_rows: Sequence[Mapping[str, Any]],
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    assets_by_row: Mapping[str, Mapping[str, Any]],
    members_by_zip: Mapping[str, Mapping[str, Any]],
    role_rows_by_id: Mapping[str, Mapping[str, Any]],
    decode_fn: Callable[[bytes], dict[str, Any]],
    decode_summary: dict[str, dict[str, Any]],
) -> Collector:
    collector = Collector()
    row_id = str(row.get("row_id", "<missing>"))
    scope = f"row:{row_id}"
    expected = config["expected"]
    pins = config["pins"]
    source = config["source"]

    collector.require(
        "row.format", scope, row.get("format") == ROW_FORMAT, f"expected {ROW_FORMAT}"
    )
    for key in ("source", "root_asset", "split", "provenance", "observation", "target", "terms"):
        collector.require(f"row.field.{key}", scope, isinstance(row.get(key), Mapping), "missing object")

    source_block = row.get("source") if isinstance(row.get("source"), Mapping) else {}
    collector.require("row.source.id", scope, source_block.get("id") == source["id"], "source id differs")
    collector.require(
        "row.source.revision",
        scope,
        source_block.get("revision") == source["revision"],
        "source revision differs",
    )
    collector.require(
        "row.source.family", scope, source_block.get("family") == source["family"], "source family differs"
    )
    collector.require(
        "row.source.split", scope, source_block.get("split") == source["split"], "source split differs"
    )
    collector.require(
        "row.source.shard", scope, source_block.get("shard") == source["shard"], "source shard differs"
    )

    root = row.get("root_asset") if isinstance(row.get("root_asset"), Mapping) else {}
    split = row.get("split") if isinstance(row.get("split"), Mapping) else {}
    observation = row.get("observation") if isinstance(row.get("observation"), Mapping) else {}
    target = row.get("target") if isinstance(row.get("target"), Mapping) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
    terms = row.get("terms") if isinstance(row.get("terms"), Mapping) else {}

    role = split.get("role")
    collector.require("row.split.role", scope, role in expected["roles"], f"unknown role {role!r}")
    collector.require(
        "row.split.group_key",
        scope,
        split.get("group_key") == root.get("root_image_id"),
        "group key must equal root image id",
    )
    collector.require(
        "row.split.policy", scope, split.get("policy") == expected["split_policy"], "split policy differs"
    )

    upstream_id = provenance.get("upstream_id")
    upstream_path = root.get("upstream_image_path")
    caption = target.get("caption")
    prompt = observation.get("prompt")
    collector.require("row.observation.task", scope, observation.get("task") == "image_captioning", "task differs")
    collector.require("row.observation.prompt", scope, isinstance(prompt, str) and bool(prompt), "prompt is empty")
    collector.require("row.target.caption", scope, isinstance(caption, str) and bool(caption), "caption is empty")
    collector.require(
        "row.target.normalized",
        scope,
        isinstance(caption, str) and normalize_caption(caption) == caption,
        "caption is not normalized",
    )
    if isinstance(caption, str):
        collector.require(
            "row.target.length",
            scope,
            expected["caption_min_chars"] <= len(caption) <= expected["caption_max_chars"],
            f"caption length {len(caption)} outside configured bounds",
        )
    collector.require(
        "row.upstream_id",
        scope,
        isinstance(upstream_id, str) and upstream_id == root.get("upstream_id"),
        "provenance upstream id differs from root asset",
    )
    collector.require(
        "row.root.image_id",
        scope,
        root.get("root_image_id") == f"{source['id']}:{upstream_id}",
        "root image id is not derived from the source id and upstream id",
    )
    collector.require(
        "row.root.image_path",
        scope,
        isinstance(upstream_path, str) and re.fullmatch(r"\d{5}/\d{9}\.jpg", upstream_path) is not None,
        "upstream image path has an unexpected form",
    )
    collector.require(
        "row.root.zip_member",
        scope,
        root.get("zip_member") == upstream_path,
        "zip member differs from the upstream image path",
    )
    collector.require(
        "row.observation.image_ref",
        scope,
        observation.get("image_ref") == f"images/{upstream_id}.jpg",
        "image_ref is not derived from the upstream id",
    )
    collector.require(
        "row.row_id",
        scope,
        row_id == f"{source['id']}:{upstream_id}#raw_caption",
        "row id is not derived from the source id and upstream id",
    )

    # Terms are identical on every row and are pinned in the QA configuration.
    collector.require(
        "row.terms", scope, dict(terms) == expected["row_terms"], "row terms differ from the pinned terms"
    )

    # Image byte, geometry and decode checks.
    image_ref = observation.get("image_ref") if isinstance(observation.get("image_ref"), str) else ""
    image_path = data_root / "raw" / image_ref
    decode: dict[str, Any] = {
        "verified": False,
        "decoder": None,
        "error": "not attempted",
    }
    decode_summary[row_id] = decode
    collector.require("row.image.file", scope, image_path.is_file(), f"missing {image_path}")
    if image_path.is_file():
        data = image_path.read_bytes()
        collector.require(
            "row.image.bytes",
            scope,
            len(data) == root.get("image_bytes"),
            f"{len(data)} != {root.get('image_bytes')}",
        )
        collector.require(
            "row.image.sha256",
            scope,
            sha256_bytes(data) == root.get("image_sha256"),
            "image digest differs from the corpus",
        )
        per_image = pins["media"]["images"]["per_image"].get(image_path.name)
        if per_image is not None:
            collector.require(
                "row.image.pinned_sha256",
                scope,
                sha256_bytes(data) == per_image["sha256"],
                "image digest differs from the QA configuration pin",
            )
            collector.require(
                "row.image.pinned_bytes",
                scope,
                len(data) == per_image["bytes"],
                "image byte length differs from the QA configuration pin",
            )
        try:
            info = inspect_image_bytes(
                data,
                expected["image_constraints"],
                member_name=str(upstream_path or ""),
            )
            collector.require(
                "row.image.structural_format",
                scope,
                info["format"] == root.get("image_format"),
                f"{info['format']} != {root.get('image_format')}",
            )
            collector.require(
                "row.image.structural_dimensions",
                scope,
                (info["width"], info["height"]) == (root.get("width"), root.get("height")),
                f"{(info['width'], info['height'])} != {(root.get('width'), root.get('height'))}",
            )
            if root.get("channels") is not None and info.get("channels") is not None:
                collector.require(
                    "row.image.structural_channels",
                    scope,
                    info["channels"] == root.get("channels"),
                    f"{info['channels']} != {root.get('channels')}",
                )
        except ExternalIntakeError as error:
            collector.add("row.image.structural", scope, False, str(error))
        try:
            decoded = decode_fn(data)
            decode.update(decoded)
            decode.pop("error", None)
            collector.require(
                "row.image.decode",
                scope,
                decoded.get("verified") is True,
                "decoder did not report a verified full decode",
            )
            collector.require(
                "row.image.decode.format",
                scope,
                decoded.get("format") == root.get("image_format"),
                f"{decoded.get('format')} != {root.get('image_format')}",
            )
            collector.require(
                "row.image.decode.dimensions",
                scope,
                (decoded.get("width"), decoded.get("height"))
                == (root.get("width"), root.get("height")),
                f"{(decoded.get('width'), decoded.get('height'))} != "
                f"{(root.get('width'), root.get('height'))}",
            )
            if root.get("channels") is not None and decoded.get("channels") is not None:
                collector.require(
                    "row.image.decode.channels",
                    scope,
                    decoded.get("channels") == root.get("channels"),
                    f"{decoded.get('channels')} != {root.get('channels')}",
                )
        except QAError as error:
            decode.update({"verified": False, "decoder": None, "error": error.detail})
            collector.add("row.image.decode", scope, False, error.detail, blocked=error.blocked)
        except Exception as error:  # noqa: BLE001
            detail = f"{type(error).__name__}: {error}"
            decode.update({"verified": False, "decoder": None, "error": detail})
            collector.add("row.image.decode", scope, False, detail)

    # Annotation provenance.
    annotation_index = provenance.get("annotation_row_index")
    collector.require(
        "row.provenance.annotation_index",
        scope,
        isinstance(annotation_index, int) and 0 <= annotation_index < len(annotation_rows),
        f"annotation row index {annotation_index!r} is out of range",
    )
    if isinstance(annotation_index, int) and 0 <= annotation_index < len(annotation_rows):
        annotation = annotation_rows[annotation_index]
        collector.require(
            "row.provenance.annotation_id",
            scope,
            annotation.get("id") == upstream_id,
            "annotation id differs from the upstream id",
        )
        collector.require(
            "row.provenance.annotation_image",
            scope,
            annotation.get("image") == upstream_path,
            "annotation image differs from the upstream image path",
        )
        try:
            annotation_prompt, annotation_answer = conversation_pair(annotation)
            collector.require(
                "row.provenance.prompt",
                scope,
                normalize_prompt(annotation_prompt) == prompt
                and provenance.get("upstream_prompt") == prompt,
                "annotation prompt differs from the normalized observation prompt",
            )
            collector.require(
                "row.provenance.caption",
                scope,
                normalize_caption(annotation_answer) == caption,
                "annotation answer differs from the normalized target caption",
            )
        except QAError as error:
            collector.add("row.provenance.annotation_shape", scope, False, error.detail)

    # Metadata provenance.
    metadata = metadata_by_id.get(str(upstream_id))
    collector.require("row.provenance.metadata", scope, metadata is not None, "no metadata row for id")
    if metadata is not None:
        collector.require(
            "row.provenance.metadata_image",
            scope,
            metadata.get("image") == upstream_path,
            "metadata image differs from the upstream image path",
        )
        collector.require(
            "row.provenance.metadata_url",
            scope,
            metadata.get("url") == root.get("upstream_url"),
            "metadata URL differs from the root asset URL",
        )
        blip = metadata.get("blip_caption")
        collector.require(
            "row.provenance.blip_sha256",
            scope,
            isinstance(blip, str) and sha256_text(blip) == provenance.get("blip_caption_sha256"),
            "metadata blip_caption digest differs from provenance",
        )
        collector.require(
            "row.provenance.blip_target",
            scope,
            isinstance(blip, str) and normalize_text(blip) == caption,
            "metadata blip_caption does not normalize to the target caption",
        )
        collector.require(
            "row.provenance.blip_equals_target",
            scope,
            provenance.get("blip_caption_equals_target") is expected["blip_caption_equals_target"],
            "blip_caption_equals_target differs from the pinned expectation",
        )

    # Ranking, sampling and preprocessing provenance.
    collector.require(
        "row.provenance.sample_seed",
        scope,
        provenance.get("sample_seed") == expected["sampling_seed"],
        "sample seed differs",
    )
    collector.require(
        "row.provenance.sample_order",
        scope,
        provenance.get("sample_order") == expected["sampling_order"],
        "sample order differs",
    )
    if isinstance(upstream_id, str):
        rank = sha256_text(f"{expected['sampling_seed']}\x00{upstream_id}")
        collector.require(
            "row.provenance.rank",
            scope,
            split.get("assignment_rank") == rank and provenance.get("rank_sha256") == rank,
            "rank digest differs from the deterministic sample rank",
        )
    collector.require(
        "row.provenance.preprocessing_version",
        scope,
        provenance.get("preprocessing_version") == expected["preprocessing_version"],
        "preprocessing version differs",
    )
    collector.require(
        "row.provenance.caption_source",
        scope,
        provenance.get("caption_source") == "llava_pretrain_answer_text",
        "caption source differs",
    )
    collector.require(
        "row.provenance.caption_note",
        scope,
        isinstance(provenance.get("caption_provenance_note"), str)
        and bool(provenance.get("caption_provenance_note")),
        "caption provenance note is missing",
    )

    # Derivative ledger.
    derivatives = root.get("derivatives")
    collector.require(
        "row.root.derivatives",
        scope,
        isinstance(derivatives, list) and len(derivatives) >= 3,
        "derivative ledger is missing",
    )
    if isinstance(derivatives, list):
        by_kind = {
            item.get("kind"): item
            for item in derivatives
            if isinstance(item, Mapping) and isinstance(item.get("kind"), str)
        }
        collector.require(
            "row.root.derivative.upstream",
            scope,
            by_kind.get("upstream_rehost", {}).get("value") == "images.zip",
            "upstream rehost derivative differs",
        )
        annotation_file = expected["annotation_source_name"]
        collector.require(
            "row.root.derivative.annotation",
            scope,
            by_kind.get("annotation_row", {}).get("value")
            == f"{annotation_file}#row[{annotation_index}]",
            "annotation derivative differs",
        )
        collector.require(
            "row.root.derivative.blip",
            scope,
            by_kind.get("blip_caption_metadata", {}).get("value")
            == provenance.get("blip_caption_sha256"),
            "blip metadata derivative differs",
        )

    # Hash bindings across corpus, asset ledger and role manifest.
    row_sha = sha256_text(canonical_json(row))
    target_sha = sha256_text(caption) if isinstance(caption, str) else ""
    asset_entry = assets_by_row.get(row_id)
    collector.require("row.hash.asset_entry", scope, asset_entry is not None, "no manifest asset entry")
    if asset_entry is not None:
        collector.require(
            "row.hash.asset_row", scope, asset_entry.get("row_sha256") == row_sha, "manifest row digest differs"
        )
        collector.require(
            "row.hash.asset_image",
            scope,
            asset_entry.get("image_sha256") == root.get("image_sha256"),
            "manifest asset image digest differs",
        )
        collector.require(
            "row.hash.asset_role", scope, asset_entry.get("role") == role, "manifest asset role differs"
        )
        collector.require(
            "row.hash.asset_group",
            scope,
            asset_entry.get("group_key") == split.get("group_key"),
            "manifest asset group differs",
        )
        collector.require(
            "row.hash.asset_bytes",
            scope,
            asset_entry.get("image_bytes") == root.get("image_bytes"),
            "manifest asset byte length differs",
        )
        collector.require(
            "row.hash.asset_format",
            scope,
            asset_entry.get("format") == root.get("image_format"),
            "manifest asset format differs",
        )
        collector.require(
            "row.hash.asset_dimensions",
            scope,
            (asset_entry.get("width"), asset_entry.get("height"))
            == (root.get("width"), root.get("height")),
            "manifest asset dimensions differ",
        )
        collector.require(
            "row.hash.asset_zip",
            scope,
            asset_entry.get("zip_member") == root.get("zip_member")
            and asset_entry.get("zip_member_crc32") == root.get("zip_member_crc32"),
            "manifest asset zip evidence differs",
        )
    role_entry = role_rows_by_id.get(row_id)
    collector.require("row.hash.role_entry", scope, role_entry is not None, "no role manifest entry")
    if role_entry is not None:
        collector.require(
            "row.hash.role_row",
            scope,
            role_entry.get("row_sha256") == row_sha,
            "role manifest row digest differs",
        )
        collector.require(
            "row.hash.role_target",
            scope,
            role_entry.get("target_sha256") == target_sha,
            "role manifest target digest differs",
        )
        collector.require(
            "row.hash.role_image",
            scope,
            role_entry.get("image_sha256") == root.get("image_sha256"),
            "role manifest image digest differs",
        )
    member = members_by_zip.get(str(upstream_path))
    collector.require("row.hash.acquisition_member", scope, member is not None, "no acquisition member")
    if member is not None:
        collector.require(
            "row.hash.member_bytes",
            scope,
            member.get("bytes") == root.get("image_bytes"),
            "acquisition member byte length differs",
        )
        collector.require(
            "row.hash.member_sha256",
            scope,
            member.get("sha256") == root.get("image_sha256"),
            "acquisition member digest differs",
        )
        collector.require(
            "row.hash.member_crc32",
            scope,
            member.get("crc32") == root.get("zip_member_crc32"),
            "acquisition member CRC32 differs",
        )
        collector.require(
            "row.hash.member_local_path",
            scope,
            member.get("local_path") == f"raw/images/{upstream_id}.jpg",
            "acquisition member local path differs",
        )

    # Source image inspection evidence (structural only in the intake).
    inspection = provenance.get("image_inspection")
    collector.require(
        "row.provenance.image_inspection",
        scope,
        isinstance(inspection, Mapping)
        and inspection.get("structural_verified") is True
        and isinstance(inspection.get("decode_verified"), bool)
        and isinstance(inspection.get("decoder"), str)
        and bool(inspection.get("decoder")),
        "source image inspection evidence is incomplete",
    )

    try:
        assert_row_label_separation(row)
        collector.add("row.label_separation", scope, True)
    except ExternalIntakeError as error:
        collector.add("row.label_separation", scope, False, str(error))

    return collector


# ---------------------------------------------------------------------------
# Admission decision
# ---------------------------------------------------------------------------


def _valid_review(review: Any) -> bool:
    return (
        isinstance(review, Mapping)
        and review.get("verdict") in VERDICTS
        and isinstance(review.get("reviewer"), str)
        and bool(review.get("reviewer", "").strip())
        and isinstance(review.get("reviewed_at"), str)
        and bool(ISO_PATTERN.match(review.get("reviewed_at", "")))
    )


def evaluate_admission(
    *,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    queue_sha256: str,
    data_checks_passed: bool,
    decision_path: Path | None,
    decision_explicit: bool,
    report_dir: Path,
    project_root: Path,
    write: bool,
    collector: Collector,
) -> dict[str, Any]:
    expected = config["expected"]
    training_role = expected["training_role"]
    training_ids = [row["row_id"] for row in rows if row["split"]["role"] == training_role]
    validation_ids = [
        row["row_id"]
        for row in rows
        if row["split"]["role"] in expected["validation_roles"]
    ]
    sealed_ids = [row["row_id"] for row in rows if row["split"]["role"] == expected["sealed_role"]]
    sealed_set = set(sealed_ids)
    all_ids = [row["row_id"] for row in rows]
    reviewable_ids = [row_id for row_id in all_ids if row_id not in sealed_set]

    admission: dict[str, Any] = {
        "decision": PENDING,
        "decision_present": False,
        "decision_file": (
            display_path(project_root, decision_path) if decision_path is not None else None
        ),
        "template_file": display_path(
            project_root, report_dir / config["semantic_review"]["template_file"]
        ),
        "decided_by": None,
        "decided_at": None,
        "allowed_decisions": list(config["semantic_review"]["allowed_decisions"]),
        "training_candidate_row_ids": training_ids,
        "validation_only_row_ids": validation_ids,
        "sealed_row_ids": sealed_ids,
        "training_admitted": False,
        "training_admitted_row_ids": [],
        "validation_only_included_in_training": False,
        "sealed_included_in_training": False,
        "semantic_review": {
            "required": True,
            "complete": False,
            "required_row_ids": reviewable_ids,
            "sealed_row_ids": sealed_ids,
            "sealed_semantic_review": "prohibited",
            "sealed_review_forbidden_row_ids": [],
            "verdict_counts": {verdict: 0 for verdict in VERDICTS},
            "incomplete_row_ids": sorted(reviewable_ids),
            "training_not_match_row_ids": [],
            "reviewed_by": None,
        },
        "refusal_reasons": [],
        "admission_output": None,
    }

    decision: Mapping[str, Any] | None = None
    if decision_path is not None and decision_path.is_file():
        try:
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            admission["refusal_reasons"].append(f"decision_malformed: {error}")
            collector.add("admission.decision", "admission", False, "decision file is not valid JSON")
            return admission
        admission["decision_present"] = True
    elif decision_explicit:
        admission["refusal_reasons"].append("decision_file_missing")
        collector.add("admission.decision", "admission", False, "explicit decision file is missing")
        return admission

    if decision is None:
        collector.add(
            "admission.decision",
            "admission",
            True,
            "no human decision file; training admission remains pending",
        )
        collector.add(
            "admission.scope",
            "admission",
            True,
            "training candidates, validation-only and sealed evaluation rows are disjoint",
        )
        collector.add(
            "admission.sealed_semantic_review",
            "admission",
            True,
            "no semantic reviews were supplied for sealed final-eval rows",
        )
        return admission

    if not isinstance(decision, Mapping):
        admission["refusal_reasons"].append("decision_malformed: root is not an object")
        collector.add("admission.decision", "admission", False, "decision root is not an object")
        return admission

    admission["decided_by"] = decision.get("decided_by")
    admission["decided_at"] = decision.get("decided_at")
    requested = decision.get("decision")
    admission["decision"] = requested
    if requested not in (PENDING, REJECT, ADMIT):
        admission["refusal_reasons"].append(f"decision_unknown: {requested!r}")
        collector.add("admission.decision", "admission", False, "decision value is not allowed")
        return admission

    reviews = decision.get("semantic_reviews")
    review_map = reviews if isinstance(reviews, Mapping) else {}
    sealed_review_forbidden = sorted(set(review_map) & sealed_set)
    admission["semantic_review"]["sealed_review_forbidden_row_ids"] = (
        sealed_review_forbidden
    )
    collector.add(
        "admission.sealed_semantic_review",
        "admission",
        not sealed_review_forbidden,
        (
            "sealed final-eval rows must not appear in semantic_reviews: "
            + ", ".join(sealed_review_forbidden)
            if sealed_review_forbidden
            else "no semantic reviews were supplied for sealed final-eval rows"
        ),
    )
    if sealed_review_forbidden:
        admission["refusal_reasons"].append("sealed_semantic_review_forbidden")
    verdict_counts = {verdict: 0 for verdict in VERDICTS}
    incomplete: list[str] = []
    require_all = config["semantic_review"].get(
        "require_all_nonsealed_queue_entries_reviewed", True
    )
    required_review_ids = reviewable_ids if require_all else training_ids
    for row_id in reviewable_ids:
        review = review_map.get(row_id)
        if _valid_review(review):
            verdict_counts[review["verdict"]] += 1
        elif row_id in required_review_ids:
            incomplete.append(row_id)
    admission["semantic_review"].update(
        {
            "complete": not incomplete,
            "verdict_counts": verdict_counts,
            "incomplete_row_ids": incomplete,
            "reviewed_by": decision.get("reviewed_by"),
        }
    )
    training_not_match = [
        row_id
        for row_id in training_ids
        if not _valid_review(review_map.get(row_id))
        or review_map[row_id]["verdict"] != "match"
    ]
    admission["semantic_review"]["training_not_match_row_ids"] = training_not_match

    if requested == PENDING:
        collector.add("admission.decision", "admission", True, "human decision is still pending")
        collector.add(
            "admission.scope",
            "admission",
            True,
            "training candidates, validation-only and sealed evaluation rows are disjoint",
        )
        return admission

    if requested == REJECT:
        collector.add("admission.decision", "admission", True, "human rejected training admission")
        return admission

    # request == ADMIT
    reasons = admission["refusal_reasons"]
    if not data_checks_passed:
        reasons.append("machine_checks_failed")
    if decision.get("queue_sha256") != queue_sha256:
        reasons.append("review_queue_sha256_mismatch")
    if incomplete:
        reasons.append("semantic_review_incomplete")
    if training_not_match:
        reasons.append("training_candidate_not_match")
    if not isinstance(decision.get("decided_by"), str) or not decision.get("decided_by", "").strip():
        reasons.append("decision_not_signed")
    if not isinstance(decision.get("decided_at"), str) or not ISO_PATTERN.match(
        decision.get("decided_at", "")
    ):
        reasons.append("decision_timestamp_invalid")
    requested_ids = decision.get("training_admitted_row_ids")
    if requested_ids is not None:
        if not isinstance(requested_ids, list) or not all(
            isinstance(value, str) for value in requested_ids
        ):
            reasons.append("training_admitted_row_ids_malformed")
        elif sorted(requested_ids) != sorted(training_ids):
            reasons.append("training_admitted_row_ids_mismatch")
            if set(requested_ids) - set(training_ids):
                reasons.append("non_training_rows_requested_for_admission")

    collector.add(
        "admission.decision",
        "admission",
        not reasons,
        "; ".join(reasons) if reasons else "human decision covers every row and every training candidate matches",
    )
    collector.add(
        "admission.scope",
        "admission",
        not (set(training_ids) & (set(validation_ids) | set(sealed_ids))),
        "training candidates, validation-only and sealed evaluation rows are disjoint",
    )
    if reasons:
        return admission

    admitted_rows = [row for row in rows if row["row_id"] in set(training_ids)]
    if any(row["split"]["role"] != training_role for row in admitted_rows):
        reasons.append("admission_scope_violation")
        collector.add("admission.output_scope", "admission", False, "admitted output contains a non-training row")
        return admission
    output_text = jsonl_text(admitted_rows)
    output_path = report_dir / TRAINING_OUTPUT_NAME
    admission["training_admitted"] = True
    admission["training_admitted_row_ids"] = training_ids
    admission["admission_output"] = {
        "path": display_path(project_root, output_path),
        "sha256": sha256_bytes(output_text.encode("utf-8")),
        "rows": len(admitted_rows),
        "row_ids": training_ids,
        "written": bool(write),
    }
    if write:
        output_path.write_text(output_text, encoding="utf-8")
    collector.add(
        "admission.output_scope",
        "admission",
        not (set(training_ids) & (set(validation_ids) | set(sealed_ids))),
        "training output contains only the training role",
    )
    return admission


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _check_counts(checks: Sequence[Check]) -> dict[str, int]:
    return {
        "total": len(checks),
        "passed": sum(1 for check in checks if check.passed),
        "failed": sum(1 for check in checks if not check.passed),
        "blocked": sum(1 for check in checks if check.blocked),
    }


def _count_prefix(checks: Sequence[Check], prefix: str) -> int:
    return sum(1 for check in checks if check.check_id.startswith(prefix) and check.passed)


def run_qa(
    *,
    project_root: Path,
    config_path: Path,
    data_root: Path | None = None,
    report_dir: Path | None = None,
    decision_path: Path | None = None,
    decoder: Callable[[bytes], dict[str, Any]] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config_path = resolve(project_root, config_path)
    config, config_sha256 = load_qa_config(config_path)
    data_root = resolve(project_root, data_root or config["data_root"]).resolve()
    report_dir = resolve(project_root, report_dir or config["report_dir"]).resolve()
    decision_explicit = decision_path is not None
    if decision_path is None:
        decision_path = report_dir / config["semantic_review"]["decision_file"]
    else:
        decision_path = resolve(project_root, decision_path)
    decode_fn = decoder or decode_with_pillow

    if write:
        report_dir.mkdir(parents=True, exist_ok=True)

    collector = Collector()
    observed: dict[str, Any] = {}
    pins = config["pins"]
    expected = config["expected"]
    source_pin = config["source"]

    collector.add("qa_config.format", "config", config["format"] == QA_CONFIG_FORMAT)
    collector.add("qa_config.cpu_only", "config", True, "no model, GPU or network path is used")
    collector.add(
        "qa_config.revision",
        "config",
        bool(REVISION_PATTERN.fullmatch(str(source_pin.get("revision", "")))),
        "source revision must be a 40-character lowercase SHA-1",
    )

    # ---- static pins -----------------------------------------------------
    intake_config_bytes = read_pinned_file(
        project_root, pins["intake_config"], collector, "pin.intake_config", observed
    )
    intake_config: Mapping[str, Any] | None = None
    if intake_config_bytes is not None:
        try:
            intake_config = json.loads(intake_config_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            collector.add("pin.intake_config.json", "pins", False, f"invalid JSON: {error}")

    manifest_bytes = read_pinned_file(
        project_root, pins["tranche_manifest"], collector, "pin.tranche_manifest", observed
    )
    manifest: Mapping[str, Any] | None = None
    if manifest_bytes is not None:
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            collector.add("pin.tranche_manifest.json", "pins", False, f"invalid JSON: {error}")

    card_bytes = read_pinned_file(project_root, pins["card"], collector, "pin.card", observed)
    annotation_rows, annotation_bytes = read_prefix(
        project_root, pins["annotation"], collector, "pin.annotation", observed
    )
    metadata_rows, metadata_bytes = read_prefix(
        project_root, pins["metadata"], collector, "pin.metadata", observed
    )
    central_directory_bytes = read_pinned_file(
        project_root,
        pins["media"]["central_directory"],
        collector,
        "pin.media.central_directory",
        observed,
    )

    images_pin = pins["media"]["images"]
    images_dir = resolve(project_root, pins["media"]["images"]["dir"])
    if not collector.require(
        "pin.media.images.dir", "pins", images_dir.is_dir(), f"missing {images_dir}", blocked=True
    ):
        image_files: list[Path] = []
        image_total = 0
        image_aggregate_sha = ""
    else:
        image_files, image_total, image_aggregate_sha = image_aggregate(images_dir)
        observed["pin.media.images"] = {
            "path": display_path(project_root, images_dir),
            "count": len(image_files),
            "bytes": image_total,
            "sha256": image_aggregate_sha,
        }
        collector.require(
            "pin.media.images.count",
            "pins",
            len(image_files) == images_pin["count"],
            f"{len(image_files)} != pinned {images_pin['count']}",
        )
        collector.require(
            "pin.media.images.bytes",
            "pins",
            image_total == images_pin["total_bytes"],
            f"{image_total} != pinned {images_pin['total_bytes']}",
        )
        collector.require(
            "pin.media.images.sha256",
            "pins",
            image_aggregate_sha == images_pin["aggregate_sha256"],
            f"{image_aggregate_sha} != pinned {images_pin['aggregate_sha256']}",
        )
        observed_names = {path.name for path in image_files}
        pinned_names = set(images_pin["per_image"])
        collector.require(
            "pin.media.images.set",
            "pins",
            observed_names == pinned_names,
            f"unexpected files: {sorted(observed_names - pinned_names)}; "
            f"missing files: {sorted(pinned_names - observed_names)}",
        )
        for name, per_image in sorted(images_pin["per_image"].items()):
            path = images_dir / name
            if not collector.require(
                f"pin.media.image.{name}.exists",
                "pins",
                path.is_file(),
                f"missing {path}",
                blocked=True,
            ):
                continue
            data = path.read_bytes()
            actual = sha256_bytes(data)
            collector.require(
                f"pin.media.image.{name}.bytes",
                "pins",
                len(data) == per_image["bytes"],
                f"{len(data)} != pinned {per_image['bytes']}",
            )
            collector.require(
                f"pin.media.image.{name}.sha256",
                "pins",
                actual == per_image["sha256"],
                f"{actual} != pinned {per_image['sha256']}",
            )

    corpus_bytes = read_pinned_file(
        project_root, pins["corpus"], collector, "pin.corpus", observed
    )
    rows: list[dict[str, Any]] = []
    if corpus_bytes is not None:
        try:
            rows = [
                json.loads(line)
                for line in corpus_bytes.decode("utf-8").splitlines()
                if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            collector.add("pin.corpus.jsonl", "pins", False, f"invalid JSONL: {error}")
        collector.require(
            "pin.corpus.rows",
            "pins",
            len(rows) == pins["corpus"]["rows"],
            f"{len(rows)} != pinned {pins['corpus']['rows']}",
        )

    rejections_bytes = read_pinned_file(
        project_root, pins["rejections"], collector, "pin.rejections", observed
    )
    rejection_rows: list[Any] = []
    if rejections_bytes is not None:
        try:
            rejection_rows = [
                json.loads(line)
                for line in rejections_bytes.decode("utf-8").splitlines()
                if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            collector.add("pin.rejections.jsonl", "pins", False, f"invalid JSONL: {error}")
        collector.require(
            "pin.rejections.rows",
            "pins",
            len(rejection_rows) == pins["rejections"]["rows"],
            f"{len(rejection_rows)} != pinned {pins['rejections']['rows']}",
        )

    role_manifests: dict[str, Mapping[str, Any]] = {}
    role_manifest_bytes: dict[str, bytes] = {}
    for role in ROLES:
        pin = pins["role_manifests"][role]
        data = read_pinned_file(
            project_root, pin, collector, f"pin.role_manifest.{role}", observed
        )
        if data is None:
            continue
        role_manifest_bytes[role] = data
        try:
            role_manifest = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            collector.add(
                f"pin.role_manifest.{role}.json", "pins", False, f"invalid JSON: {error}"
            )
            continue
        role_manifests[role] = role_manifest
        collector.require(
            f"pin.role_manifest.{role}.rows",
            "pins",
            len(role_manifest.get("rows", [])) == pin["rows"],
            f"{len(role_manifest.get('rows', []))} != pinned {pin['rows']}",
        )
        collector.require(
            f"pin.role_manifest.{role}.sealed",
            "pins",
            role_manifest.get("sealed") is pin["sealed"],
            "sealed flag differs from the QA pin",
        )
        collector.require(
            f"pin.role_manifest.{role}.seal_sha256",
            "pins",
            role_manifest.get("seal_sha256") == pin["seal_sha256"],
            "seal digest differs from the QA pin",
        )

    # ---- intake config binding ------------------------------------------
    if intake_config is not None:
        intake_dataset = intake_config.get("dataset", {})
        collector.require(
            "intake.source.id",
            "intake",
            intake_dataset.get("id") == source_pin["id"],
            "intake dataset id differs from the QA source pin",
        )
        collector.require(
            "intake.source.revision",
            "intake",
            intake_dataset.get("revision") == source_pin["revision"],
            "intake revision differs from the QA source pin",
        )
        collector.require(
            "intake.source.family",
            "intake",
            intake_dataset.get("family") == source_pin["family"],
            "intake family differs from the QA source pin",
        )
        collector.require(
            "intake.source.split",
            "intake",
            intake_dataset.get("split") == source_pin["split"],
            "intake split differs from the QA source pin",
        )
        collector.require(
            "intake.source.shard",
            "intake",
            intake_dataset.get("shard") == source_pin["shard"],
            "intake shard differs from the QA source pin",
        )
        collector.require(
            "intake.card",
            "intake",
            intake_dataset.get("card", {}).get("sha256") == pins["card"]["sha256"],
            "intake card digest differs from the QA pin",
        )
        intake_files = intake_dataset.get("files", {})
        collector.require(
            "intake.annotation",
            "intake",
            intake_files.get("annotation", {}).get("sha256")
            == pins["annotation"]["declared_full_sha256"],
            "intake annotation digest differs from the QA pin",
        )
        collector.require(
            "intake.metadata",
            "intake",
            intake_files.get("metadata", {}).get("sha256")
            == pins["metadata"]["declared_full_sha256"],
            "intake metadata digest differs from the QA pin",
        )
        collector.require(
            "intake.media",
            "intake",
            intake_files.get("images", {}).get("sha256")
            == pins["media"]["central_directory"].get("declared_full_sha256"),
            "intake image archive digest differs from the QA pin",
        )
        collector.require(
            "intake.terms",
            "intake",
            intake_config.get("terms", {}).get("permitted_use")
            == expected["terms"]["permitted_use"]
            and intake_config.get("terms", {}).get("research_use_only")
            is expected["terms"]["research_use_only"]
            and intake_config.get("terms", {}).get("redistribution")
            == expected["terms"]["redistribution"]
            and intake_config.get("terms", {}).get("commercial_use")
            is expected["terms"]["commercial_use"],
            "intake terms differ from the QA expectation",
        )

    # ---- manifest binding -----------------------------------------------
    if manifest is not None:
        collector.require(
            "tranche.manifest.format",
            "tranche",
            manifest.get("format") == TRANCHE_FORMAT,
            f"expected {TRANCHE_FORMAT}",
        )
        collector.require(
            "tranche.manifest.status", "tranche", manifest.get("status") == "admitted", "manifest is not admitted"
        )
        dataset = manifest.get("dataset", {})
        collector.require(
            "tranche.manifest.source",
            "tranche",
            dataset.get("id") == source_pin["id"]
            and dataset.get("revision") == source_pin["revision"]
            and dataset.get("family") == source_pin["family"]
            and dataset.get("split") == source_pin["split"]
            and dataset.get("shard") == source_pin["shard"],
            "manifest dataset pin differs from the QA source pin",
        )
        if intake_config is not None:
            collector.require(
                "tranche.manifest.dataset_binding",
                "tranche",
                dataset.get("card") == intake_config.get("dataset", {}).get("card")
                and dataset.get("files") == intake_config.get("dataset", {}).get("files"),
                "manifest card/files differ from the pinned intake configuration",
            )
            collector.require(
                "tranche.manifest.terms_binding",
                "tranche",
                manifest.get("terms") == intake_config.get("terms"),
                "manifest terms differ from the pinned intake configuration",
            )
            collector.require(
                "tranche.manifest.config_binding",
                "tranche",
                manifest.get("config", {}).get("sha256")
                == pins["intake_config"]["sha256"]
                and manifest.get("config", {}).get("path")
                == pins["intake_config"]["path"],
                "manifest config binding differs from the QA pin",
            )
            collector.require(
                "tranche.manifest.preprocessing_binding",
                "tranche",
                manifest.get("preprocessing") == intake_config.get("preprocessing"),
                "manifest preprocessing differs from the pinned intake configuration",
            )
            collector.require(
                "tranche.manifest.sampling_binding",
                "tranche",
                manifest.get("sampling", {}).get("seed")
                == intake_config.get("sampling", {}).get("seed")
                and manifest.get("sampling", {}).get("order")
                == intake_config.get("sampling", {}).get("order")
                and manifest.get("sampling", {}).get("split_quotas")
                == intake_config.get("sampling", {}).get("split_quotas"),
                "manifest sampling differs from the pinned intake configuration",
            )

        artifacts = manifest.get("artifacts", {})
        collector.require(
            "tranche.manifest.artifact.corpus",
            "tranche",
            artifacts.get("corpus", {}).get("sha256") == pins["corpus"]["sha256"]
            and artifacts.get("corpus", {}).get("rows") == pins["corpus"]["rows"],
            "manifest corpus artifact differs from the QA pin",
        )
        collector.require(
            "tranche.manifest.artifact.rejections",
            "tranche",
            artifacts.get("rejections", {}).get("sha256") == pins["rejections"]["sha256"]
            and artifacts.get("rejections", {}).get("rows") == pins["rejections"]["rows"],
            "manifest rejection artifact differs from the QA pin",
        )
        for role in ROLES:
            artifact = artifacts.get("manifests", {}).get(role, {})
            pin = pins["role_manifests"][role]
            collector.require(
                f"tranche.manifest.artifact.{role}",
                "tranche",
                artifact.get("sha256") == pin["sha256"]
                and artifact.get("rows") == pin["rows"]
                and artifact.get("sealed") is pin["sealed"]
                and artifact.get("seal_sha256") == pin["seal_sha256"],
                "role manifest artifact differs from the QA pin",
            )
            if role in role_manifest_bytes:
                collector.require(
                    f"tranche.manifest.artifact.{role}.actual",
                    "tranche",
                    sha256_bytes(role_manifest_bytes[role]) == artifact.get("sha256"),
                    "role manifest bytes differ from the manifest artifact",
                )

        counts = manifest.get("counts", {})
        collector.require(
            "tranche.manifest.counts.rows",
            "tranche",
            counts.get("admitted_rows") == expected["rows"]
            and counts.get("normalized_rows") == expected["rows"]
            and counts.get("admitted_root_assets") == expected["root_groups"],
            "manifest row/root counts differ from the QA expectation",
        )
        for role in ROLES:
            role_counts = counts.get("roles", {}).get(role, {})
            collector.require(
                f"tranche.manifest.counts.{role}",
                "tranche",
                role_counts.get("rows") == expected["roles"][role]
                and role_counts.get("root_groups") == expected["roles"][role],
                "manifest role counts differ from the QA expectation",
            )
        collector.require(
            "tranche.manifest.counts.window",
            "tranche",
            counts.get("window_rows") == expected["annotation_rows"]
            and counts.get("metadata_rows") == expected["metadata_rows"]
            and counts.get("metadata_candidates") == expected["metadata_candidates"]
            and counts.get("rejected_rows") == expected["rejections_rows"],
            "manifest window/metadata counts differ from the QA expectation",
        )

        inspection = manifest.get("inspection", {})
        recomputed_formats = Counter(row.get("root_asset", {}).get("image_format") for row in rows)
        recomputed_bytes = sum(
            int(row.get("root_asset", {}).get("image_bytes", 0)) for row in rows
        )
        recomputed_structural = sum(
            1
            for row in rows
            if row.get("provenance", {}).get("image_inspection", {}).get("structural_verified")
            is True
        )
        recomputed_decode = sum(
            1
            for row in rows
            if row.get("provenance", {}).get("image_inspection", {}).get("decode_verified")
            is True
        )
        recomputed_decoders = Counter(
            row.get("provenance", {}).get("image_inspection", {}).get("decoder")
            for row in rows
        )
        collector.require(
            "tranche.manifest.inspection",
            "tranche",
            inspection.get("assets") == len(rows)
            and inspection.get("image_bytes") == recomputed_bytes
            and inspection.get("formats") == dict(sorted(recomputed_formats.items()))
            and inspection.get("structural_verified") == recomputed_structural
            and inspection.get("decode_verified") == recomputed_decode
            and inspection.get("decoders") == dict(sorted(recomputed_decoders.items())),
            "manifest inspection ledger differs from the corpus",
        )

        acquisition = manifest.get("acquisition", {})
        collector.require(
            "tranche.manifest.acquisition.budget",
            "tranche",
            acquisition.get("within_budget") is True
            and isinstance(acquisition.get("acquired_bytes"), int)
            and isinstance(acquisition.get("budget_bytes"), int)
            and acquisition["acquired_bytes"] <= acquisition["budget_bytes"],
            "acquisition ledger is not within budget",
        )
        if intake_config is not None:
            collector.require(
                "tranche.manifest.acquisition.budget_binding",
                "tranche",
                acquisition.get("budget_bytes")
                == intake_config.get("admission", {}).get("max_acquired_bytes")
                and acquisition.get("campaign_disk_ceiling_bytes")
                == intake_config.get("admission", {}).get("campaign_disk_ceiling_bytes"),
                "acquisition budget differs from the pinned intake configuration",
            )
        sources = acquisition.get("sources", {})
        collector.require(
            "tranche.manifest.acquisition.card",
            "tranche",
            sources.get("card", {}).get("sha256") == pins["card"]["sha256"]
            and sources.get("card", {}).get("bytes") == pins["card"]["bytes"]
            and sources.get("card", {}).get("full_sha256_verified") is True,
            "card acquisition evidence differs",
        )
        for kind, pin, expected_rows in (
            ("annotation", pins["annotation"], expected["annotation_rows"]),
            ("metadata", pins["metadata"], expected["metadata_rows"]),
        ):
            source = sources.get(kind, {})
            collector.require(
                f"tranche.manifest.acquisition.{kind}",
                "tranche",
                source.get("acquired_prefix_sha256") == pin["sha256"]
                and source.get("acquired_prefix_bytes") == pin["bytes"]
                and source.get("complete_rows_in_window") == expected_rows
                and source.get("bytes_in_window_sha256") == pin["window_sha256"]
                and source.get("full_sha256_verified") is False,
                f"{kind} acquisition evidence differs",
            )
        zip_source = sources.get("images_zip", {})
        collector.require(
            "tranche.manifest.acquisition.images_zip",
            "tranche",
            zip_source.get("central_directory_sha256")
            == pins["media"]["central_directory"]["sha256"]
            and zip_source.get("central_directory_bytes")
            == pins["media"]["central_directory"]["bytes"]
            and zip_source.get("central_directory_records")
            == pins["media"]["central_directory"]["records"]
            and zip_source.get("central_directory_offset")
            == pins["media"]["central_directory"].get("offset")
            and zip_source.get("file_entries") == pins["media"]["central_directory"]["file_entries"]
            and zip_source.get("full_sha256_verified") is False,
            "image archive acquisition evidence differs",
        )
        window_alignment = acquisition.get("window_alignment", {})
        annotation_ids = {
            row.get("id") for row in annotation_rows if isinstance(row, Mapping)
        }
        metadata_ids = {row.get("id") for row in metadata_rows if isinstance(row, Mapping)}
        collector.require(
            "tranche.manifest.acquisition.window_alignment",
            "tranche",
            window_alignment.get("annotation_rows") == len(annotation_rows)
            and window_alignment.get("metadata_rows") == len(metadata_rows)
            and window_alignment.get("shared_ids") == len(annotation_ids & metadata_ids)
            and window_alignment.get("annotation_only_ids")
            == len(annotation_ids - metadata_ids)
            and window_alignment.get("metadata_only_ids")
            == len(metadata_ids - annotation_ids),
            "window alignment evidence differs",
        )
        if intake_config is not None:
            for kind, pin in (("annotation", pins["annotation"]), ("metadata", pins["metadata"])):
                collector.require(
                    f"tranche.manifest.acquisition.{kind}.declared",
                    "tranche",
                    sources.get(kind, {}).get("declared_full_sha256")
                    == intake_config.get("dataset", {}).get("files", {}).get(kind, {}).get("sha256"),
                    f"{kind} declared full digest differs from the intake config",
                )
            collector.require(
                "tranche.manifest.acquisition.images_zip.declared",
                "tranche",
                zip_source.get("declared_full_sha256")
                == intake_config.get("dataset", {}).get("files", {}).get("images", {}).get("sha256")
                and zip_source.get("declared_size")
                == intake_config.get("dataset", {}).get("files", {}).get("images", {}).get("size"),
                "declared image archive digest/size differs from the intake config",
            )

        members = zip_source.get("members", [])
        collector.require(
            "tranche.manifest.acquisition.members",
            "tranche",
            isinstance(members, list)
            and len(members) == expected["rows"]
            and zip_source.get("members_verified") == expected["rows"],
            "acquisition member ledger does not cover every row",
        )

        assets = manifest.get("assets", [])
        collector.require(
            "tranche.manifest.assets",
            "tranche",
            isinstance(assets, list) and len(assets) == expected["rows"],
            "asset ledger does not cover every row",
        )

        exclusions = manifest.get("exclusions", {})
        rejected = exclusions.get("rejected", {})
        collector.require(
            "tranche.manifest.exclusions",
            "tranche",
            rejected.get("rows") == expected["rejections_rows"]
            and rejected.get("metadata") == 0
            and rejected.get("content") == 0
            and rejected.get("reasons") == {}
            and rejected.get("ledger", {}).get("sha256") == pins["rejections"]["sha256"]
            and exclusions.get("downloaded_assets") == expected["rows"]
            and exclusions.get("admitted_assets") == expected["rows"],
            "manifest exclusion/rejection ledger differs",
        )

    # ---- corpus-level checks --------------------------------------------
    collector.require(
        "tranche.rows.count", "tranche", len(rows) == expected["rows"], f"{len(rows)} != {expected['rows']}"
    )
    role_counts = Counter(row.get("split", {}).get("role") for row in rows)
    collector.require(
        "tranche.rows.roles",
        "tranche",
        dict(sorted(role_counts.items())) == dict(sorted(expected["roles"].items())),
        f"{dict(role_counts)} != {expected['roles']}",
    )
    root_groups = {
        row.get("split", {}).get("group_key")
        for row in rows
        if isinstance(row.get("split"), Mapping)
    }
    collector.require(
        "tranche.rows.root_groups",
        "tranche",
        len(root_groups) == expected["root_groups"],
        f"{len(root_groups)} != {expected['root_groups']}",
    )
    role_order = {role: index for index, role in enumerate(ROLES)}
    expected_order = [
        f"{source_pin['id']}:{row.get('provenance', {}).get('upstream_id')}#raw_caption"
        for row in sorted(
            rows,
            key=lambda item: (
                role_order.get(item.get("split", {}).get("role"), 99),
                item.get("split", {}).get("assignment_rank", ""),
                item.get("row_id", ""),
            ),
        )
    ]
    collector.require(
        "tranche.rows.order",
        "tranche",
        [row.get("row_id") for row in rows] == expected_order,
        "corpus row order differs from the deterministic role/rank order",
    )

    try:
        duplicates = assert_no_duplicates(rows)
        collector.add("tranche.duplicates", "tranche", True)
    except QAError as error:
        duplicates = find_duplicates(rows)
        collector.add("tranche.duplicates", "tranche", False, error.detail)

    try:
        leakage = assert_leakage(rows)
        collector.add("tranche.leakage", "tranche", True)
    except QAError as error:
        leakage = {}
        collector.add("tranche.leakage", "tranche", False, error.detail)
    if manifest is not None:
        collector.require(
            "tranche.manifest.leakage",
            "tranche",
            manifest.get("leakage") == expected["leakage"],
            "manifest leakage report differs from the QA expectation",
        )

    try:
        reserved_matches = find_reserved_family_matches(
            rows, expected["reserved_patterns"], expected["reserved_families"]
        )
    except re.error as error:
        reserved_matches = []
        collector.add("tranche.reserved_family", "tranche", False, f"invalid pattern: {error}")
    else:
        collector.require(
            "tranche.reserved_family",
            "tranche",
            not reserved_matches,
            json.dumps(reserved_matches, sort_keys=True),
        )
    collector.require(
        "tranche.reserved_family.source",
        "tranche",
        source_pin["family"] not in expected["reserved_families"],
        f"selected source family {source_pin['family']!r} is reserved",
    )
    if manifest is not None:
        manifest_reserved = manifest.get("reserved_family_check", {})
        collector.require(
            "tranche.manifest.reserved_family",
            "tranche",
            manifest_reserved.get("patterns") == expected["reserved_patterns"]
            and manifest_reserved.get("reserved_families") == expected["reserved_families"]
            and manifest_reserved.get("admitted_asset_matches") == 0,
            "manifest reserved-family evidence differs",
        )
        collector.require(
            "tranche.manifest.source_family",
            "tranche",
            manifest.get("dataset", {}).get("family") not in expected["reserved_families"],
            "manifest source family is reserved",
        )

    # ---- role manifests and seals ---------------------------------------
    assets_by_row: dict[str, Mapping[str, Any]] = {}
    members_by_zip: dict[str, Mapping[str, Any]] = {}
    if manifest is not None:
        for asset in manifest.get("assets", []):
            if isinstance(asset, Mapping) and isinstance(asset.get("row_id"), str):
                assets_by_row[asset["row_id"]] = asset
        members = manifest.get("acquisition", {}).get("sources", {}).get("images_zip", {}).get("members", [])
        if isinstance(members, list):
            for member in members:
                if isinstance(member, Mapping) and isinstance(member.get("zip_member"), str):
                    members_by_zip[member["zip_member"]] = member

    role_rows_by_id: dict[str, Mapping[str, Any]] = {}
    for role in ROLES:
        role_manifest = role_manifests.get(role)
        if role_manifest is None:
            continue
        collector.require(
            f"tranche.role_manifest.{role}.format",
            "tranche",
            role_manifest.get("format") == MANIFEST_FORMAT,
            f"expected {MANIFEST_FORMAT}",
        )
        collector.require(
            f"tranche.role_manifest.{role}.role",
            "tranche",
            role_manifest.get("role") == role,
            "role manifest role differs",
        )
        collector.require(
            f"tranche.role_manifest.{role}.source",
            "tranche",
            role_manifest.get("source_id") == source_pin["id"]
            and role_manifest.get("source_revision") == source_pin["revision"],
            "role manifest source differs",
        )
        manifest_rows = role_manifest.get("rows", [])
        collector.require(
            f"tranche.role_manifest.{role}.counts",
            "tranche",
            role_manifest.get("counts", {}).get("rows") == expected["roles"][role]
            and role_manifest.get("counts", {}).get("root_groups") == expected["roles"][role],
            "role manifest counts differ",
        )
        corpus_role_ids = [
            row["row_id"] for row in rows if row["split"]["role"] == role
        ]
        collector.require(
            f"tranche.role_manifest.{role}.rows",
            "tranche",
            [entry.get("row_id") for entry in manifest_rows] == corpus_role_ids,
            "role manifest rows differ from the corpus",
        )
        for entry in manifest_rows:
            if isinstance(entry, Mapping) and isinstance(entry.get("row_id"), str):
                role_rows_by_id[entry["row_id"]] = entry
        if role == EVAL_ROLE:
            collector.require(
                f"tranche.seal.{role}.sealed",
                "tranche",
                role_manifest.get("sealed") is True and role_manifest.get("untouched") is True,
                "final-evaluation manifest is not sealed/untouched",
            )
            recomputed = role_manifest_seal(role_manifest)
            collector.require(
                f"tranche.seal.{role}.sha256",
                "tranche",
                role_manifest.get("seal_sha256") == recomputed,
                f"{role_manifest.get('seal_sha256')} != recomputed {recomputed}",
            )
            if manifest is not None:
                collector.require(
                    f"tranche.seal.{role}.artifact",
                    "tranche",
                    manifest.get("artifacts", {})
                    .get("manifests", {})
                    .get(role, {})
                    .get("seal_sha256")
                    == recomputed,
                    "manifest final-evaluation seal differs from the recomputed seal",
                )
        else:
            collector.require(
                f"tranche.seal.{role}.open",
                "tranche",
                role_manifest.get("sealed") is False and role_manifest.get("seal_sha256") is None,
                "non-evaluation manifest must not be sealed",
            )

    role_ids = {
        role: [row["row_id"] for row in rows if row["split"]["role"] == role]
        for role in ROLES
    }
    sealed_ids = set(role_ids[EVAL_ROLE])
    non_sealed_ids = set(role_ids["train"]) | set(role_ids["development"])
    collector.require(
        "tranche.seal.exclusion",
        "tranche",
        len(sealed_ids) == expected["roles"][EVAL_ROLE] and not (sealed_ids & non_sealed_ids),
        "sealed final-evaluation rows overlap or do not match the expected count",
    )
    validation_ids = set(role_ids["development"])
    collector.require(
        "tranche.validation_only.separate",
        "tranche",
        not (validation_ids & (set(role_ids["train"]) | sealed_ids)),
        "development validation rows overlap training or sealed evaluation rows",
    )
    if manifest is not None:
        collector.require(
            "tranche.manifest.asset_set",
            "tranche",
            set(assets_by_row) == set(row["row_id"] for row in rows),
            "manifest asset ledger row ids differ from the corpus",
        )
        collector.require(
            "tranche.manifest.member_set",
            "tranche",
            set(members_by_zip)
            == set(row["root_asset"]["upstream_image_path"] for row in rows),
            "acquisition member ledger paths differ from the corpus",
        )

    # ---- per-row checks -------------------------------------------------
    metadata_by_id = {
        row["id"]: row
        for row in metadata_rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    decode_summary: dict[str, dict[str, Any]] = {}
    row_collectors: dict[str, Collector] = {}
    for row in rows:
        row_id = str(row.get("row_id", "<missing>"))
        row_collector = verify_row(
            row,
            project_root=project_root,
            data_root=data_root,
            config=config,
            annotation_rows=annotation_rows,
            metadata_by_id=metadata_by_id,
            assets_by_row=assets_by_row,
            members_by_zip=members_by_zip,
            role_rows_by_id=role_rows_by_id,
            decode_fn=decode_fn,
            decode_summary=decode_summary,
        )
        row_collectors[row_id] = row_collector
        collector.merge(row_collector)

    # ---- review queue ----------------------------------------------------
    sections: dict[str, list[dict[str, Any]]] = {
        "training_candidates": [],
        "validation_only": [],
        "sealed_final_eval": [],
    }
    for review_index, row in enumerate(rows, 1):
        row_id = row["row_id"]
        role = row["split"]["role"]
        sealed = role == expected["sealed_role"]
        row_collector = row_collectors.get(row_id, Collector())
        image_path = data_root / "raw" / row["observation"]["image_ref"]
        section = SECTION_BY_ROLE.get(role, "unknown")
        if sealed:
            semantic_review = {
                "required": False,
                "performed": False,
                "status": "prohibited_sealed",
                "verdict": None,
                "reviewer": None,
                "reviewed_at": None,
                "notes": None,
                "scope_note": (
                    "Sealed final-eval rows are structural-only. Their target text is "
                    "withheld and semantic review is prohibited."
                ),
            }
        else:
            semantic_review = {
                "required": True,
                "performed": False,
                "status": "pending",
                "verdict": None,
                "reviewer": None,
                "reviewed_at": None,
                "notes": None,
                "scope_note": (
                    "Machine checks never verify that the caption describes the image. "
                    "A human must open the image and record a verdict."
                ),
            }
        entry = {
            "review_index": review_index,
            "row_id": row_id,
            "role": role,
            "section": section,
            "admission_scope": SCOPE_BY_ROLE.get(role, "unknown"),
            "training_admission_eligible": role == expected["training_role"] and not sealed,
            "sealed": sealed,
            "target_text_redacted": sealed,
            "redaction": (
                {
                    "caption": "withheld_sealed_final_eval",
                    "prompt": "withheld_sealed_final_eval",
                    "target_sha256_retained": True,
                }
                if sealed
                else None
            ),
            "group_key": row["split"]["group_key"],
            "source": dict(row["source"]),
            "image": {
                "image_ref": row["observation"]["image_ref"],
                "image_path": display_path(project_root, image_path),
                "sha256": row["root_asset"]["image_sha256"],
                "bytes": row["root_asset"]["image_bytes"],
                "format": row["root_asset"]["image_format"],
                "width": row["root_asset"]["width"],
                "height": row["root_asset"]["height"],
                "channels": row["root_asset"].get("channels"),
                "zip_member": row["root_asset"]["zip_member"],
                "zip_member_crc32": row["root_asset"]["zip_member_crc32"],
                "upstream_url": row["root_asset"]["upstream_url"],
                "decode": dict(decode_summary.get(row_id, {})),
            },
            "prompt": None if sealed else row["observation"]["prompt"],
            "caption": None if sealed else row["target"]["caption"],
            "target_sha256": sha256_text(row["target"]["caption"]),
            "row_sha256": sha256_text(canonical_json(row)),
            "provenance": {
                "annotation_row_index": row["provenance"].get("annotation_row_index"),
                "upstream_id": row["provenance"].get("upstream_id"),
                "upstream_prompt": (
                    None if sealed else row["provenance"].get("upstream_prompt")
                ),
                "upstream_url": row["root_asset"].get("upstream_url"),
                "blip_caption_sha256": row["provenance"].get("blip_caption_sha256"),
                "blip_caption_equals_target": row["provenance"].get("blip_caption_equals_target"),
                "caption_source": row["provenance"].get("caption_source"),
                "caption_provenance_note": row["provenance"].get("caption_provenance_note"),
                "rank_sha256": row["provenance"].get("rank_sha256"),
                "sample_seed": row["provenance"].get("sample_seed"),
                "sample_order": row["provenance"].get("sample_order"),
                "preprocessing_version": row["provenance"].get("preprocessing_version"),
            },
            "terms": dict(row["terms"]),
            "machine_checks": {
                "passed": row_collector.passed,
                "total": len(row_collector.checks),
                "failed": [check.check_id for check in row_collector.checks if not check.passed],
                "detail_path": "qa_report.json#machine_checks.row_checks",
            },
            "semantic_review": semantic_review,
        }
        sections.setdefault(section, []).append(entry)

    reviewable_sections = ("training_candidates", "validation_only")
    reviewable_count = sum(len(sections[name]) for name in reviewable_sections)
    sealed_count = len(sections["sealed_final_eval"])
    queue = {
        "format": REVIEW_QUEUE_FORMAT,
        "qa_id": config["qa_id"],
        "created_on": config["created_on"],
        "data_root": display_path(project_root, data_root),
        "source": dict(source_pin),
        "instructions": {
            "machine_checks": (
                "Structural/byte-level checks have already run; see machine_checks on each entry. "
                "Sealed final-eval entries retain only structural hashes and decode evidence."
            ),
            "semantic_review": (
                "Only training_candidates and validation_only entries are reviewable. Open "
                "image.image_path and compare it with caption; record match/mismatch/unclear in "
                "admission_decision.json. The QA script never sets this. Sealed final-eval rows "
                "have no caption text and cannot receive a semantic verdict."
            ),
            "admission": (
                "Copy admission_decision.template.json to admission_decision.json only after every "
                "non-sealed queue entry has been inspected. Sealed final-eval and validation-only "
                "rows are never eligible for the training-admitted output, and sealed rows must "
                "never appear in semantic_reviews."
            ),
        },
        "counts": {
            "total": len(rows),
            "reviewable": reviewable_count,
            "sealed": sealed_count,
            "sections": {name: len(entries) for name, entries in sections.items()},
        },
        "sections": sections,
        "sealed_semantic_review": "prohibited",
        "sealed_target_text_redacted": True,
        "non_claims": {
            "semantic_caption_image_agreement_verified": False,
            "training_admission_claimed": False,
            "validation_only_assets_included_in_training": False,
            "sealed_final_eval_included_in_training": False,
            "sealed_semantic_review_performed": False,
        },
    }
    queue_text = json_text(queue)
    queue_sha256 = sha256_bytes(queue_text.encode("utf-8"))

    # ---- admission -------------------------------------------------------
    data_checks_passed = collector.passed
    data_check_count = len(collector.checks)
    admission = evaluate_admission(
        config=config,
        rows=rows,
        queue_sha256=queue_sha256,
        data_checks_passed=data_checks_passed,
        decision_path=decision_path,
        decision_explicit=decision_explicit,
        report_dir=report_dir,
        project_root=project_root,
        write=write,
        collector=collector,
    )

    # A stale admitted output is dangerous when no valid decision authorizes it.
    output_path = report_dir / TRAINING_OUTPUT_NAME
    if output_path.exists() and not admission["training_admitted"]:
        collector.add(
            "admission.stale_output",
            "admission",
            False,
            f"{display_path(project_root, output_path)} exists without a valid admission decision",
        )
        admission["refusal_reasons"].append("stale_training_admitted_output_present")

    data_checks = collector.checks[:data_check_count]
    admission_checks = collector.checks[data_check_count:]
    global_checks = [check for check in data_checks if not check.scope.startswith("row:")]
    row_checks = [check for check in data_checks if check.scope.startswith("row:")]
    check_stats = _check_counts(data_checks)
    machine_passed = data_checks_passed and check_stats["failed"] == 0
    if not machine_passed and not admission["training_admitted"]:
        if admission["decision"] == ADMIT:
            admission["refusal_reasons"].append("machine_checks_failed")
        admission["training_admitted"] = False
        admission["training_admitted_row_ids"] = []
        admission["admission_output"] = None
    admission["refusal_reasons"] = list(dict.fromkeys(admission["refusal_reasons"]))
    admission["machine_checks_passed"] = machine_passed
    admission["admission_checks_passed"] = all(check.passed for check in admission_checks)

    if not machine_passed:
        report_status = "machine_checks_failed"
    elif admission["training_admitted"]:
        report_status = "training_admitted"
    elif admission["decision"] == REJECT:
        report_status = "rejected_by_human"
    elif admission["refusal_reasons"]:
        report_status = "machine_checks_passed_admission_refused"
    else:
        report_status = "machine_checks_passed_semantic_review_pending"

    # ---- template --------------------------------------------------------
    template = {
        "format": DECISION_FORMAT,
        "qa_id": config["qa_id"],
        "queue_sha256": queue_sha256,
        "decision": PENDING,
        "decided_by": None,
        "decided_at": None,
        "reviewed_by": None,
        "training_admitted_row_ids": None,
        "reviewable_row_ids": [
            entry["row_id"]
            for name in ("training_candidates", "validation_only")
            for entry in sections[name]
        ],
        "validation_only_row_ids": admission["validation_only_row_ids"],
        "sealed_row_ids": admission["sealed_row_ids"],
        "sealed_semantic_review": "prohibited",
        "semantic_reviews": {
            entry["row_id"]: {
                "verdict": None,
                "reviewer": None,
                "reviewed_at": None,
                "notes": None,
            }
            for name in ("training_candidates", "validation_only")
            for entry in sections[name]
        },
        "instructions": (
            "Copy this file to admission_decision.json, fill every non-sealed semantic verdict "
            "(match/mismatch/unclear) with reviewer and ISO reviewed_at, then set decision to "
            "admit_machine_checked_train_rows or reject. Sealed final-eval rows must never appear "
            "in semantic_reviews; they are listed separately in sealed_row_ids. The QA script "
            "validates the decision but never writes it."
        ),
    }
    template_text = json_text(template)

    # ---- report ----------------------------------------------------------
    report = {
        "format": QA_REPORT_FORMAT,
        "qa_id": config["qa_id"],
        "created_on": config["created_on"],
        "status": report_status,
        "cpu_only": True,
        "network_access_performed": False,
        "native_execution_performed": False,
        "data_root": display_path(project_root, data_root),
        "qa_config": {
            "path": display_path(project_root, config_path),
            "sha256": config_sha256,
        },
        "source": dict(source_pin),
        "pins": {
            "expected": {
                key: pins[key]
                for key in (
                    "intake_config",
                    "tranche_manifest",
                    "card",
                    "annotation",
                    "metadata",
                    "corpus",
                    "rejections",
                )
            }
            | {
                "media": pins["media"],
                "role_manifests": pins["role_manifests"],
            },
            "observed": observed,
        },
        "tranche": {
            "data_root": display_path(project_root, data_root),
            "manifest": {
                "path": display_path(project_root, resolve(project_root, pins["tranche_manifest"]["path"])),
                "sha256": pins["tranche_manifest"]["sha256"],
                "format": manifest.get("format") if manifest else None,
                "status": manifest.get("status") if manifest else None,
            },
            "corpus": {
                "path": display_path(project_root, resolve(project_root, pins["corpus"]["path"])),
                "sha256": pins["corpus"]["sha256"],
                "rows": len(rows),
            },
            "rejections": {
                "path": display_path(project_root, resolve(project_root, pins["rejections"]["path"])),
                "sha256": pins["rejections"]["sha256"],
                "rows": len(rejection_rows),
            },
            "counts": {
                "rows": len(rows),
                "root_groups": len(root_groups),
                "roles": dict(sorted(role_counts.items())),
            },
            "roles": {
                role: {
                    "rows": expected["roles"][role],
                    "root_groups": expected["roles"][role],
                    "manifest_path": pins["role_manifests"][role]["path"],
                    "manifest_sha256": pins["role_manifests"][role]["sha256"],
                    "sealed": pins["role_manifests"][role]["sealed"],
                    "seal_sha256": pins["role_manifests"][role]["seal_sha256"],
                }
                for role in ROLES
            },
        },
        "machine_checks": {
            "performed": True,
            "passed": machine_passed,
            "total": check_stats["total"],
            "passed_checks": check_stats["passed"],
            "failed": check_stats["failed"],
            "blocked": check_stats["blocked"],
            "failed_checks": [
                check_record(check) for check in data_checks if not check.passed
            ],
            "global_checks": [check_record(check) for check in global_checks],
            "row_checks": [check_record(check) for check in row_checks],
        },
        "admission_checks": {
            "performed": True,
            "passed": all(check.passed for check in admission_checks),
            "checks": [check_record(check) for check in admission_checks],
            "failed_checks": [
                check_record(check) for check in admission_checks if not check.passed
            ],
        },
        "provenance": {
            "source_revision": source_pin["revision"],
            "annotation_prefix_sha256": pins["annotation"]["sha256"],
            "annotation_window_sha256": pins["annotation"]["window_sha256"],
            "annotation_rows_verified": len(annotation_rows),
            "metadata_prefix_sha256": pins["metadata"]["sha256"],
            "metadata_window_sha256": pins["metadata"]["window_sha256"],
            "metadata_rows_verified": len(metadata_rows),
            "admitted_rows_verified": len(rows),
            "annotation_checks_passed": _count_prefix(row_checks, "row.provenance.annotation"),
            "metadata_checks_passed": _count_prefix(row_checks, "row.provenance.metadata"),
            "row_hash_checks_passed": _count_prefix(row_checks, "row.hash."),
            "per_image_sha256_checks_passed": _count_prefix(row_checks, "row.image.sha256"),
        },
        "decode": {
            "verified": sum(1 for value in decode_summary.values() if value.get("verified") is True),
            "failed": sum(1 for value in decode_summary.values() if value.get("verified") is not True),
            "decoders": dict(
                sorted(
                    Counter(
                        value.get("decoder")
                        for value in decode_summary.values()
                        if value.get("verified") is True and value.get("decoder")
                    ).items()
                )
            ),
            "source_declared_decode_verified": sum(
                1
                for row in rows
                if row.get("provenance", {}).get("image_inspection", {}).get("decode_verified")
                is True
            ),
            "gap": (
                "the intake recorded structural-only inspection; this QA decodes every image "
                "and does not rewrite the source artifacts"
            ),
        },
        "duplicates": duplicates,
        "leakage": leakage,
        "reserved_family": {
            "patterns": expected["reserved_patterns"],
            "reserved_families": expected["reserved_families"],
            "matches": reserved_matches,
            "admitted_asset_matches": len(reserved_matches),
        },
        "seals": {
            role: {
                "path": pins["role_manifests"][role]["path"],
                "observed_sha256": observed.get(
                    f"pin.role_manifest.{role}", {}
                ).get("sha256"),
                "pinned_sha256": pins["role_manifests"][role]["sha256"],
                "rows": len(role_manifests.get(role, {}).get("rows", [])),
                "pinned_rows": pins["role_manifests"][role]["rows"],
                "sealed": role_manifests.get(role, {}).get("sealed"),
                "pinned_sealed": pins["role_manifests"][role]["sealed"],
                "untouched": role_manifests.get(role, {}).get("untouched"),
                "seal_sha256": role_manifests.get(role, {}).get("seal_sha256"),
                "pinned_seal_sha256": pins["role_manifests"][role]["seal_sha256"],
                "recomputed_seal_sha256": (
                    role_manifest_seal(role_manifests[role]) if role in role_manifests else None
                ),
            }
            for role in ROLES
        },
        "terms": {
            **expected["terms"],
            "verified_rows": len(rows),
            "manifest_terms_match_intake_config": (
                intake_config is not None
                and manifest is not None
                and manifest.get("terms") == intake_config.get("terms")
            ),
        },
        "scope_separation": {
            "training_candidate_row_ids": admission["training_candidate_row_ids"],
            "validation_only_row_ids": admission["validation_only_row_ids"],
            "sealed_final_eval_row_ids": admission["sealed_row_ids"],
            "training_output_contains_only_training_role": (
                None
                if not admission["training_admitted"]
                else set(admission["training_admitted_row_ids"])
                == set(admission["training_candidate_row_ids"])
            ),
            "validation_only_excluded_from_training": not (
                set(admission["training_admitted_row_ids"])
                & set(admission["validation_only_row_ids"])
            ),
            "sealed_excluded_from_training": not (
                set(admission["training_admitted_row_ids"]) & set(admission["sealed_row_ids"])
            ),
        },
        "review_queue": {
            "path": display_path(project_root, report_dir / "review_queue.json"),
            "sha256": queue_sha256,
            "format": REVIEW_QUEUE_FORMAT,
            "total_entries": len(rows),
            "reviewable_entries": reviewable_count,
            "sealed_entries": sealed_count,
            "sections": {
                name: {
                    "count": len(entries),
                    "roles": sorted({entry["role"] for entry in entries}),
                }
                for name, entries in sections.items()
            },
            "semantic_review_required": True,
            "semantic_review_required_row_ids": admission["semantic_review"][
                "required_row_ids"
            ],
            "sealed_semantic_review": "prohibited",
            "sealed_target_text_redacted": True,
            "semantic_review_complete": admission["semantic_review"]["complete"],
        },
        "admission": admission,
        "claims": {
            "machine_checks_performed": True,
            "machine_checks_passed": machine_passed,
            "semantic_caption_image_agreement_asserted_by_qa": False,
            "semantic_review_recorded": admission["semantic_review"]["complete"],
            "training_admission_claimed": admission["training_admitted"],
            "validation_only_assets_separated": True,
            "sealed_final_eval_rows_excluded_from_training": not (
                set(admission["training_admitted_row_ids"]) & set(admission["sealed_row_ids"])
            ),
            "sealed_final_eval_target_text_redacted_in_queue": True,
            "sealed_semantic_review_permitted": False,
            "sealed_semantic_review_attempted": bool(
                admission["semantic_review"]["sealed_review_forbidden_row_ids"]
            ),
            "sealed_semantic_review_accepted": False,
            "perceptual_or_near_duplicate_detection_claimed": False,
            "model_quality_evaluated": False,
        },
        "non_claims": {
            "caption_image_semantic_agreement": False,
            "training_admission": admission["training_admitted"],
            "validation_only_rows_are_training_rows": False,
            "sealed_rows_are_training_rows": False,
            "sealed_semantic_review": False,
            "sealed_target_text_in_review_queue": False,
        },
    }

    if write:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "review_queue.json").write_text(queue_text, encoding="utf-8")
        template_path = report_dir / config["semantic_review"]["template_file"]
        template_path.write_text(template_text, encoding="utf-8")
        (report_dir / "qa_report.json").write_text(json_text(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/external_caption_asset_qa_v1.json",
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--decision",
        type=Path,
        default=None,
        help="human admission decision; defaults to <report-dir>/admission_decision.json",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="run every check and print the summary without writing report artifacts",
    )
    args = parser.parse_args(argv)
    try:
        report = run_qa(
            project_root=args.project_root,
            config_path=args.config,
            data_root=args.data_root,
            report_dir=args.report_dir,
            decision_path=args.decision,
            write=not args.no_write,
        )
    except QAError as error:
        print(f"external caption asset QA refused: {error}", file=sys.stderr)
        return 2
    summary = {
        "status": report["status"],
        "machine_checks": {
            "passed": report["machine_checks"]["passed"],
            "total": report["machine_checks"]["total"],
            "passed_checks": report["machine_checks"]["passed_checks"],
            "failed": report["machine_checks"]["failed"],
            "blocked": report["machine_checks"]["blocked"],
        },
        "admission_checks": {
            "passed": report["admission_checks"]["passed"],
            "failed": len(report["admission_checks"]["failed_checks"]),
        },
        "admission": {
            "decision": report["admission"]["decision"],
            "training_admitted": report["admission"]["training_admitted"],
            "training_admitted_row_ids": report["admission"]["training_admitted_row_ids"],
            "validation_only_row_ids": report["admission"]["validation_only_row_ids"],
            "sealed_row_ids": report["admission"]["sealed_row_ids"],
            "refusal_reasons": report["admission"]["refusal_reasons"],
        },
        "report": report["review_queue"]["path"].replace("review_queue.json", "qa_report.json"),
        "review_queue": report["review_queue"]["path"],
        "semantic_review_required": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not report["machine_checks"]["passed"]:
        return 1
    if report["admission"]["refusal_reasons"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
