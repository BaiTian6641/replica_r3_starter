#!/usr/bin/env python3
"""Verify the SparkX 2.6 dataset research package without network or media access.

The verifier reads the committed research catalog, its evidence manifest,
Hugging Face metadata snapshot, Clotho provenance audit, repository snapshot,
the research report/plan documents, and the runtime dataset registry.  It
recomputes inventory counts, evidence hash coverage, source/revision pins,
rights/split/provenance requirements, and the documented no-asset and
no-training claims.  Contradictory inputs produce a refusal report and a
non-zero exit status instead of a passing verdict.

It performs documentation and metadata validation only.  It does not inspect
media bytes, download assets, execute dataset loaders, audit dataset quality,
resolve rights, or admit anything to the runtime registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCHEMA_VERSION = 1
REVIEW_DATE = "2026-10-26"
OUTPUT_DIR = "reports/sparkx_dataset_validation_20261026"
VALIDATION_NAME = "validation.json"
MANIFEST_NAME = "intake_manifest.json"
PROPOSED_ASSET_CEILING_GIB = 25
NATIVE_RAM_GUARD_GIB = 50
MAX_TEXT_EVIDENCE_BYTES = 16 * 1024 * 1024

CATALOG = "docs/datasets/20261026_SparkX_Dataset_Catalog.json"
PLAN = "docs/datasets/20261026_SparkX_Dataset_Plan.md"
MINICPM_VALIDATION = "docs/datasets/20261026_MiniCPM_Dataset_Validation.md"
EARLIER_CATALOG = "docs/datasets/20261024_Source_Catalog.json"
REPORT_DIR = "reports/sparkx_dataset_hunt_20261026"
REPORT_README = f"{REPORT_DIR}/README.md"
EVIDENCE = f"{REPORT_DIR}/evidence.json"
HF_METADATA = f"{REPORT_DIR}/hf_metadata.json"
CLOTHO_AUDIT = f"{REPORT_DIR}/clotho_metadata_audit.json"
REPOSITORY_SNAPSHOT = f"{REPORT_DIR}/repository_snapshot.json"
PRIOR_DOCUMENTATION_VALIDATION = f"{REPORT_DIR}/documentation_validation.json"
FROZEN_ARCHITECTURE = f"{REPORT_DIR}/architecture_v4_3_snapshot.txt"
REGISTRY = "configs/dataset_registry.json"
ACQUISITION_GUIDE = "docs/08_Dataset_Acquisition_and_Curriculum.md"
SPINE_TEACHING = "docs/research/sparkx_intrinsic_learning_spine_20261026/Teaching_and_Validation.md"
SPINE_ARCHITECTURE = "docs/research/sparkx_intrinsic_learning_spine_20261026/Architecture_and_Spine.md"

INPUT_ROLES = (
    (CATALOG, "dataset_catalog"),
    (PLAN, "dataset_plan"),
    (MINICPM_VALIDATION, "validation_companion"),
    (EARLIER_CATALOG, "earlier_catalog"),
    (REPORT_README, "research_report"),
    (EVIDENCE, "evidence_manifest"),
    (HF_METADATA, "hf_metadata_snapshot"),
    (CLOTHO_AUDIT, "clotho_provenance_audit"),
    (REPOSITORY_SNAPSHOT, "repository_snapshot"),
    (PRIOR_DOCUMENTATION_VALIDATION, "prior_documentation_validation"),
    (FROZEN_ARCHITECTURE, "frozen_architecture"),
    (REGISTRY, "runtime_dataset_registry"),
    (ACQUISITION_GUIDE, "acquisition_guide"),
    (SPINE_TEACHING, "spine_document"),
    (SPINE_ARCHITECTURE, "spine_document"),
)

EXPECTED_UNRESOLVED = frozenset(
    {
        "dataset_api_nvidia__MF-Think",
        "dataset_card_confit__clotho",
        "dataset_card_UnFaZeD07__Music-AVQA",
        "dataset_card_DreamyWanderer__MusicNet",
    }
)
EXPECTED_UNRESOLVED_CARD = frozenset(
    {
        "dataset_card_UnFaZeD07__Music-AVQA",
        "dataset_card_DreamyWanderer__MusicNet",
        "dataset_card_confit__clotho",
    }
)
EXPECTED_FAMILY_CARD_UNRESOLVED = frozenset(
    {
        "dataset_card_UnFaZeD07__Music-AVQA",
        "dataset_card_DreamyWanderer__MusicNet",
    }
)
EXPECTED_LOOKUP_REPOS = frozenset({"apple/mmau", "confit/clotho"})
EXPECTED_LOOKUP_FINDING_REPOS = frozenset(
    {"nvidia/MF-Think", "confit/clotho", "apple/mmau", "VoxEmo"}
)
ALLOWED_PRIORITIES = frozenset(
    {
        "P1_conditional",
        "P1_component_review",
        "P2_conditional",
        "P2_component_review",
        "P2_experimental",
        "P2_restricted_research",
        "P2_mirror_unqualified",
        "EVAL",
    }
)
RESTRICTED_PRIORITIES = frozenset(
    {"P2_restricted_research", "P2_mirror_unqualified"}
)
RESTRICTED_LICENSE_PRIORITIES = frozenset(
    {"P2_restricted_research", "P2_mirror_unqualified", "EVAL"}
)
FORBIDDEN_MEDIA_SUFFIXES = frozenset(
    {
        ".wav",
        ".flac",
        ".mp3",
        ".ogg",
        ".m4a",
        ".aac",
        ".opus",
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".zip",
        ".tar",
        ".tgz",
        ".gz",
        ".7z",
        ".rar",
        ".parquet",
        ".arrow",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".safetensors",
        ".bin",
        ".onnx",
    }
)
RIGHTS_VOCABULARY = (
    "licen",
    "right",
    "term",
    "restrict",
    "permission",
    "clearance",
    "noncommercial",
    "academic",
    "research",
    "attribution",
    "consent",
    "obligation",
    "gate",
    "evaluation",
    "training",
    "protocol",
    "provenance",
    "conditions",
    "intended-use",
    "odc-",
    "cc-by",
    "card",
)
REQUIRED_FAMILY_TEXT_FIELDS = (
    "id",
    "title",
    "role",
    "priority",
    "reported_scale",
    "annotation_provenance",
    "asset_access",
    "rights_review",
)
REQUIRED_FAMILY_LIST_FIELDS = (
    "hugging_face",
    "modalities",
    "split_group_keys",
    "required_checks",
)
class InputError(RuntimeError):
    """Input is absent or structurally unusable; no verification was performed."""


class CheckLog:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, check_id: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"id": check_id, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def failures(self) -> list[dict]:
        return [row for row in self.rows if not row["ok"]]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _license_values(license_metadata: object) -> tuple[str, ...]:
    if license_metadata is None:
        return ()
    if isinstance(license_metadata, str):
        return (license_metadata.strip().lower(),)
    if isinstance(license_metadata, list):
        return tuple(sorted(str(item).strip().lower() for item in license_metadata))
    return (str(license_metadata).strip().lower(),)


def _has_rights_vocabulary(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in RIGHTS_VOCABULARY)


def _has_restriction_vocabulary(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "restrict",
            "noncommercial",
            "non-commercial",
            "nc-",
            "-nc",
            "academic",
            "oneway",
            "one-way",
            "no product",
            "license",
            "rights",
            "terms",
            "permission",
            "clearance",
            "gate",
        )
    )


def _suffix(url: str) -> str:
    path = url.split("?", 1)[0].split("#", 1)[0]
    filename = path.rsplit("/", 1)[-1]
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read JSON input {path}: {exc}") from exc


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read text input {path}: {exc}") from exc


def _load_inputs(root: Path) -> tuple[list[dict], dict[str, object], dict[str, str]]:
    inputs: list[dict] = []
    payloads: dict[str, object] = {}
    texts: dict[str, str] = {}
    for relative, role in INPUT_ROLES:
        path = root / relative
        if not path.is_file():
            raise InputError(f"required input is missing: {relative}")
        data = path.read_bytes()
        inputs.append(
            {
                "path": relative,
                "role": role,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
        )
        if relative.endswith(".json"):
            payloads[relative] = _load_json(path)
        else:
            texts[relative] = _load_text(path)

    catalog = payloads[CATALOG]
    evidence = payloads[EVIDENCE]
    metadata = payloads[HF_METADATA]
    snapshot = payloads[REPOSITORY_SNAPSHOT]
    if not isinstance(catalog, dict) or not isinstance(catalog.get("source_families"), list):
        raise InputError(f"{CATALOG} lacks a source_families list")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("sources"), list):
        raise InputError(f"{EVIDENCE} lacks a sources list")
    if not isinstance(metadata, list):
        raise InputError(f"{HF_METADATA} is not a list")
    if not isinstance(snapshot, dict):
        raise InputError(f"{REPOSITORY_SNAPSHOT} is not an object")
    for relative in (CLOTHO_AUDIT, REGISTRY, EARLIER_CATALOG):
        if not isinstance(payloads[relative], dict):
            raise InputError(f"{relative} is not an object")
    return inputs, payloads, texts


def _family_repository_entries(families: list[dict]) -> list[tuple[dict, dict]]:
    entries: list[tuple[dict, dict]] = []
    for family in families:
        repositories = family.get("hugging_face")
        if not isinstance(repositories, list):
            continue
        for repository in repositories:
            if isinstance(repository, dict):
                entries.append((family, repository))
    return entries


def _iter_reported_configs(reported_splits: object):
    if isinstance(reported_splits, dict):
        yield reported_splits
    elif isinstance(reported_splits, list):
        for item in reported_splits:
            if isinstance(item, dict):
                yield item


def _is_positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def _split_summary(entry: tuple[dict, dict]) -> tuple[int, int, list[str]]:
    family, repository = entry
    configs = 0
    splits = 0
    problems: list[str] = []
    reported = repository.get("reported_splits")
    if reported is None:
        # Some cards expose no dataset-config metadata.  That is recorded as an
        # explicit null, not silently treated as a validated split layout.
        return 0, 0, []
    if not isinstance(reported, (dict, list)):
        return 0, 0, [
            f"{family.get('id')}/{repository.get('repo_id')}: reported_splits type"
        ]
    for config in _iter_reported_configs(reported):
        configs += 1
        split_rows = config.get("splits")
        if not isinstance(split_rows, list) or not split_rows:
            problems.append(f"{family.get('id')}/{repository.get('repo_id')}: no splits")
            continue
        for split in split_rows:
            splits += 1
            if not isinstance(split, dict) or not _nonempty_str(split.get("name")):
                problems.append(
                    f"{family.get('id')}/{repository.get('repo_id')}: unnamed split"
                )
                continue
            examples = split.get("num_examples")
            if not isinstance(examples, int) or isinstance(examples, bool) or examples <= 0:
                problems.append(
                    f"{family.get('id')}/{repository.get('repo_id')}/{split['name']}: "
                    "num_examples is not a positive integer"
                )
            num_bytes = split.get("num_bytes")
            if num_bytes is not None and not _is_positive_number(num_bytes):
                problems.append(
                    f"{family.get('id')}/{repository.get('repo_id')}/{split['name']}: "
                    "num_bytes is not a positive number"
                )
        features = config.get("features")
        if not isinstance(features, list) or not features:
            problems.append(
                f"{family.get('id')}/{repository.get('repo_id')}: no feature schema"
            )
        for size_key in ("dataset_size", "download_size"):
            size = config.get(size_key)
            if size is not None and not _is_positive_number(size):
                problems.append(
                    f"{family.get('id')}/{repository.get('repo_id')}: "
                    f"{size_key} is not a positive number"
                )
    return configs, splits, problems


def _local_markdown_links(text: str) -> list[str]:
    links: list[str] = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        links.append(target.split("#", 1)[0])
    return links


def _evidence_kind(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "dataset":
        return parts[1]
    return "unknown"


def verify(root: Path) -> tuple[dict, dict | None]:
    """Return (validation_report, intake_manifest_or_None)."""
    root = root.resolve()
    inputs, payloads, texts = _load_inputs(root)

    catalog: dict = payloads[CATALOG]  # type: ignore[assignment]
    evidence: dict = payloads[EVIDENCE]  # type: ignore[assignment]
    metadata: list[dict] = payloads[HF_METADATA]  # type: ignore[assignment]
    clotho: dict = payloads[CLOTHO_AUDIT]  # type: ignore[assignment]
    snapshot: dict = payloads[REPOSITORY_SNAPSHOT]  # type: ignore[assignment]
    registry: dict = payloads[REGISTRY]  # type: ignore[assignment]
    earlier: dict = payloads[EARLIER_CATALOG]  # type: ignore[assignment]
    prior_validation: dict = payloads[PRIOR_DOCUMENTATION_VALIDATION]  # type: ignore[assignment]
    plan_text = _norm_text(texts[PLAN])
    readme_text = _norm_text(texts[REPORT_README])

    families: list[dict] = [
        family for family in catalog["source_families"] if isinstance(family, dict)
    ]
    entries = _family_repository_entries(families)
    evidence_sources: list[dict] = [
        source for source in evidence["sources"] if isinstance(source, dict)
    ]
    evidence_by_name = {
        source.get("name"): source
        for source in evidence_sources
        if _nonempty_str(source.get("name"))
    }
    metadata_by_repo = {
        record.get("repo_id"): record
        for record in metadata
        if isinstance(record, dict) and _nonempty_str(record.get("repo_id"))
    }
    repo_ids = {
        repository.get("repo_id")
        for _, repository in entries
        if _nonempty_str(repository.get("repo_id"))
    }
    lookup_repos = set(metadata_by_repo) - repo_ids

    resolved = [
        source for source in evidence_sources if _is_sha256(source.get("sha256"))
    ]
    unresolved = [
        source for source in evidence_sources if not _is_sha256(source.get("sha256"))
    ]
    unresolved_by_name = {
        source.get("name"): source
        for source in unresolved
        if _nonempty_str(source.get("name"))
    }

    log = CheckLog()
    warnings: list[str] = []

    # ---- Catalog inventory and source pins ---------------------------------
    log.check(
        "catalog_schema",
        catalog.get("schema_version") == 1 and catalog.get("review_date") == REVIEW_DATE,
        "schema_version=1 and review_date=2026-10-26 required",
    )
    log.check(
        "catalog_family_count",
        catalog.get("source_family_count") == len(families) and len(families) == 36,
        f"declared={catalog.get('source_family_count')!r} actual={len(families)} expected=36",
    )
    family_ids = [family.get("id") for family in families]
    unique_family_ids = {family_id for family_id in family_ids if _nonempty_str(family_id)}
    log.check(
        "catalog_family_ids_unique",
        len(unique_family_ids) == len(families) and len(families) == 36,
        f"unique_ids={len(unique_family_ids)} families={len(families)}",
    )
    missing_required: list[str] = []
    for family in families:
        family_id = family.get("id", "?")
        for field in REQUIRED_FAMILY_TEXT_FIELDS:
            if not _nonempty_str(family.get(field)):
                missing_required.append(f"{family_id}:{field}")
        for field in REQUIRED_FAMILY_LIST_FIELDS:
            value = family.get(field)
            if not isinstance(value, list) or not value:
                missing_required.append(f"{family_id}:{field}")
        for field in ("media_audited", "training_admitted", "auto_admit"):
            if not isinstance(family.get(field), bool):
                missing_required.append(f"{family_id}:{field}")
        if not isinstance(family.get("notes"), str):
            missing_required.append(f"{family_id}:notes")
        if not isinstance(family.get("primary_evidence"), list):
            missing_required.append(f"{family_id}:primary_evidence")
        checks_rows = family.get("required_checks")
        if isinstance(checks_rows, list) and len(checks_rows) < 2:
            missing_required.append(f"{family_id}:required_checks<2")
    log.check(
        "catalog_family_required_fields",
        not missing_required,
        "missing_or_invalid=" + ",".join(sorted(missing_required))[:600],
    )
    invalid_priorities = sorted(
        str(family.get("priority"))
        for family in families
        if family.get("priority") not in ALLOWED_PRIORITIES
    )
    log.check(
        "catalog_priority_vocabulary",
        not invalid_priorities,
        "invalid=" + ",".join(invalid_priorities),
    )
    log.check(
        "repository_entries_unique",
        len(entries) == 40 and len(repo_ids) == 40,
        f"entries={len(entries)} unique_repo_ids={len(repo_ids)} expected=40",
    )
    log.check(
        "repository_count_declaration",
        catalog.get("hugging_face_repositories_inspected") == 42
        and len(metadata_by_repo) == 42
        and len(metadata_by_repo) == len(repo_ids) + len(lookup_repos),
        (
            f"declared={catalog.get('hugging_face_repositories_inspected')!r} "
            f"metadata={len(metadata_by_repo)} catalog={len(repo_ids)} "
            f"lookups={len(lookup_repos)}"
        ),
    )
    catalog_missing_metadata = sorted(repo_ids - set(metadata_by_repo))
    metadata_missing_catalog = sorted(set(metadata_by_repo) - repo_ids - lookup_repos)
    log.check(
        "repository_metadata_coverage",
        not catalog_missing_metadata and not metadata_missing_catalog,
        (
            f"catalog_missing_metadata={catalog_missing_metadata} "
            f"metadata_missing_catalog={metadata_missing_catalog}"
        ),
    )
    lookup_findings = catalog.get("lookup_findings")
    lookup_finding_repos = {
        finding.get("repo")
        for finding in lookup_findings
        if isinstance(finding, dict) and _nonempty_str(finding.get("repo"))
    } if isinstance(lookup_findings, list) else set()
    lookup_findings_complete = (
        isinstance(lookup_findings, list)
        and len(lookup_findings) == 4
        and lookup_finding_repos == EXPECTED_LOOKUP_FINDING_REPOS
        and all(
            _nonempty_str(finding.get("result"))
            for finding in lookup_findings
            if isinstance(finding, dict)
        )
    )
    log.check(
        "lookup_repositories_documented",
        lookup_repos == EXPECTED_LOOKUP_REPOS and lookup_findings_complete,
        (
            f"metadata_lookups={sorted(lookup_repos)} "
            f"finding_repos={sorted(repo for repo in lookup_finding_repos if repo)}"
        ),
    )
    invalid_revisions = sorted(
        repository.get("repo_id", "?")
        for _, repository in entries
        if not re.fullmatch(r"[0-9a-f]{40}", str(repository.get("revision", "")))
    )
    log.check(
        "revision_pins_format",
        not invalid_revisions,
        "non_40_hex_revisions=" + ",".join(invalid_revisions),
    )
    pin_problems: list[str] = []
    for _, repository in entries:
        repo_id = repository.get("repo_id")
        revision = repository.get("revision")
        expected = f"https://huggingface.co/datasets/{repo_id}/blob/{revision}/README.md"
        if repository.get("pinned_card_url") != expected:
            pin_problems.append(f"{repo_id}: pinned_card_url")
        api_source = evidence_by_name.get(repository.get("api_evidence"))
        if api_source is None or api_source.get("url") != (
            f"https://huggingface.co/api/datasets/{repo_id}"
        ):
            pin_problems.append(f"{repo_id}: api_evidence_url")
    log.check(
        "pinned_card_urls",
        not pin_problems,
        "problems=" + ",".join(sorted(pin_problems))[:600],
    )
    file_count_problems: list[str] = []
    for _, repository in entries:
        record = metadata_by_repo.get(repository.get("repo_id"))
        if record is None:
            file_count_problems.append(f"{repository.get('repo_id')}: missing metadata")
            continue
        if repository.get("observed_repo_file_count") != record.get("file_count"):
            file_count_problems.append(f"{repository.get('repo_id')}: file_count")
        if repository.get("gated_metadata") != record.get("gated"):
            file_count_problems.append(f"{repository.get('repo_id')}: gated_metadata")
        if repository.get("revision") != record.get("revision"):
            file_count_problems.append(f"{repository.get('repo_id')}: revision")
    log.check(
        "observed_file_counts",
        not file_count_problems,
        "problems=" + ",".join(sorted(file_count_problems))[:600],
    )
    split_problems: list[str] = []
    config_count = 0
    split_count = 0
    repos_without_split_metadata = 0
    for entry in entries:
        configs, splits, problems = _split_summary(entry)
        config_count += configs
        split_count += splits
        if entry[1].get("reported_splits") is None:
            repos_without_split_metadata += 1
        split_problems.extend(problems)
    log.check(
        "reported_splits_structure",
        not split_problems
        and config_count >= 16
        and split_count > config_count,
        (
            f"configs={config_count} splits={split_count} "
            f"repos_without_split_metadata={repos_without_split_metadata} "
            f"problems={'; '.join(split_problems)[:600]}"
        ),
    )
    missing_warnings = sorted(
        str(repository.get("repo_id"))
        for _, repository in entries
        if not _nonempty_str(repository.get("access_warning"))
    )
    log.check(
        "access_warnings",
        not missing_warnings,
        "missing=" + ",".join(missing_warnings),
    )
    license_problems: list[str] = []
    for _, repository in entries:
        if "publisher_license_metadata" not in repository:
            license_problems.append(f"{repository.get('repo_id')}: missing key")
            continue
        record = metadata_by_repo.get(repository.get("repo_id"))
        if record is None:
            continue
        card_license = (record.get("card_data") or {}).get("license")
        if _license_values(repository.get("publisher_license_metadata")) != _license_values(
            card_license
        ):
            license_problems.append(f"{repository.get('repo_id')}: license mismatch")
    log.check(
        "license_metadata_present",
        not license_problems,
        "problems=" + ",".join(sorted(license_problems))[:600],
    )

    # ---- Evidence inventory and hash coverage ------------------------------
    names = [source.get("name") for source in evidence_sources]
    evidence_by_kind: dict[str, int] = {}
    for name in names:
        if _nonempty_str(name):
            kind = _evidence_kind(name)
            evidence_by_kind[kind] = evidence_by_kind.get(kind, 0) + 1
    log.check(
        "evidence_declared_inventory",
        len(evidence_sources) == 125
        and len(set(names)) == 125
        and _nonempty_str(evidence.get("scope"))
        and _nonempty_str(evidence.get("licensing"))
        and isinstance(evidence.get("reused_prior_papers"), list)
        and len(evidence.get("reused_prior_papers")) == 2,
        f"records={len(evidence_sources)} unique={len(set(names))} expected=125",
    )
    invalid_hashes = sorted(
        str(source.get("name"))
        for source in resolved
        if not _is_sha256(source.get("sha256"))
        or not isinstance(source.get("bytes"), int)
        or source.get("bytes") <= 0
        or source.get("error") is not None
    )
    log.check(
        "evidence_hash_coverage",
        len(resolved) == 121
        and len(unresolved) == 4
        and not invalid_hashes
        and not any(
            "sha256" in source or source.get("bytes") is not None
            for source in unresolved
        )
        and len(resolved) + len(unresolved) == 125,
        (
            f"hashed={len(resolved)} unresolved={len(unresolved)} "
            f"invalid={invalid_hashes}"
        ),
    )
    log.check(
        "evidence_unresolved_set",
        set(unresolved_by_name) == EXPECTED_UNRESOLVED,
        "unresolved=" + ",".join(sorted(unresolved_by_name)),
    )
    card_unresolved_names = {
        repository.get("card_evidence")
        for _, repository in entries
        if repository.get("card_read") is False
    }
    lookup_card_unresolved = {
        name for name in unresolved_by_name if name == "dataset_card_confit__clotho"
    }
    unresolved_error_problems: list[str] = []
    for name, source in unresolved_by_name.items():
        error = source.get("error")
        if not _nonempty_str(error):
            unresolved_error_problems.append(f"{name}:no_error")
            continue
        expected_code = "401" if name == "dataset_api_nvidia__MF-Think" else "404"
        if expected_code not in str(error):
            unresolved_error_problems.append(f"{name}:unexpected_error")
    log.check(
        "unresolved_mapping",
        card_unresolved_names == EXPECTED_FAMILY_CARD_UNRESOLVED
        and lookup_card_unresolved == {"dataset_card_confit__clotho"}
        and not unresolved_error_problems
        and "dataset_api_nvidia__MF-Think" in unresolved_by_name,
        (
            f"catalog_card_unresolved={sorted(card_unresolved_names)} "
            f"errors={unresolved_error_problems}"
        ),
    )
    missing_references: list[str] = []
    referenced_hashes = 0
    for _, repository in entries:
        for key in ("api_evidence", "card_evidence"):
            name = repository.get(key)
            if name not in evidence_by_name:
                missing_references.append(f"{repository.get('repo_id')}:{key}")
        for name in repository.get("primary_evidence") or []:
            if name not in evidence_by_name:
                missing_references.append(f"{repository.get('repo_id')}:{name}")
            elif not _is_sha256(evidence_by_name[name].get("sha256")):
                missing_references.append(f"{repository.get('repo_id')}:{name}:unhashed")
            elif name in evidence_by_name:
                referenced_hashes += 1
        for key in ("api_evidence", "card_evidence"):
            name = repository.get(key)
            if name in evidence_by_name and _is_sha256(evidence_by_name[name].get("sha256")):
                referenced_hashes += 1
    for family in families:
        for name in family.get("primary_evidence") or []:
            if not _nonempty_str(name):
                missing_references.append(f"{family.get('id')}:empty_primary_evidence")
            elif name not in evidence_by_name:
                missing_references.append(f"{family.get('id')}:{name}")
            elif not _is_sha256(evidence_by_name[name].get("sha256")):
                missing_references.append(f"{family.get('id')}:{name}:unhashed")
            else:
                referenced_hashes += 1
    log.check(
        "evidence_references_resolve",
        not missing_references,
        f"problems={missing_references} hashed_references={referenced_hashes}",
    )
    api_hash_problems: list[str] = []
    for repo_id, record in metadata_by_repo.items():
        source = evidence_by_name.get(record.get("api_evidence"))
        if source is None:
            api_hash_problems.append(f"{repo_id}:missing_api_evidence")
        elif source.get("sha256") != record.get("metadata_sha256"):
            api_hash_problems.append(f"{repo_id}:hash_mismatch")
    log.check(
        "api_evidence_hash_matches_metadata",
        not api_hash_problems,
        "problems=" + ",".join(sorted(api_hash_problems))[:600],
    )
    card_problems: list[str] = []
    resolved_card_count = 0
    for _, repository in entries:
        name = repository.get("card_evidence")
        source = evidence_by_name.get(name)
        if source is None:
            card_problems.append(f"{repository.get('repo_id')}:missing_card_evidence")
            continue
        expected_url = (
            f"https://huggingface.co/datasets/{repository.get('repo_id')}/resolve/"
            f"{repository.get('revision')}/README.md"
        )
        if repository.get("card_read") is True:
            resolved_card_count += 1
            if not _is_sha256(source.get("sha256")):
                card_problems.append(f"{repository.get('repo_id')}:card_not_hashed")
            if source.get("url") != expected_url:
                card_problems.append(f"{repository.get('repo_id')}:card_url")
        elif repository.get("card_read") is False:
            if source.get("name") not in unresolved_by_name:
                card_problems.append(f"{repository.get('repo_id')}:card_not_marked_unresolved")
        else:
            card_problems.append(f"{repository.get('repo_id')}:card_read_not_boolean")
    log.check(
        "card_evidence_consistency",
        not card_problems and resolved_card_count == 38,
        f"resolved_cards={resolved_card_count} problems={card_problems[:8]}",
    )
    textual_problems: list[str] = []
    csv_records: list[str] = []
    max_bytes = 0
    for source in resolved:
        url = str(source.get("url", ""))
        suffix = _suffix(url)
        if suffix in FORBIDDEN_MEDIA_SUFFIXES:
            textual_problems.append(f"{source.get('name')}:{suffix}")
        if suffix == ".csv":
            csv_records.append(str(source.get("name")))
        size = source.get("bytes")
        if isinstance(size, int):
            max_bytes = max(max_bytes, size)
    if max_bytes > MAX_TEXT_EVIDENCE_BYTES:
        textual_problems.append(f"oversized_text_evidence:{max_bytes}")
    if sorted(csv_records) != [
        "dataset_clotho_metadata_development",
        "dataset_clotho_metadata_evaluation",
        "dataset_clotho_metadata_validation",
    ]:
        textual_problems.append("unexpected_csv_set")
    log.check(
        "evidence_urls_textual_only",
        not textual_problems,
        (
            f"problems={textual_problems} csv_records={len(csv_records)} "
            f"max_bytes={max_bytes}"
        ),
    )
    provenance_problems: list[str] = []
    for source in resolved:
        name = source.get("name")
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00",
            str(source.get("retrieved_at_utc", "")),
        ):
            provenance_problems.append(f"{name}:retrieved_at")
        if not str(source.get("url", "")).startswith(("http://", "https://")):
            provenance_problems.append(f"{name}:url")
        if not str(source.get("final_url", "")).startswith(("http://", "https://")):
            provenance_problems.append(f"{name}:final_url")
        for key in ("scratch_path", "text_path"):
            raw = source.get(key)
            if not _nonempty_str(raw):
                provenance_problems.append(f"{name}:{key}")
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                provenance_problems.append(f"{name}:{key}:relative")
                continue
            try:
                if os.path.commonpath([str(root), str(candidate)]) == str(root):
                    provenance_problems.append(f"{name}:{key}:inside_repository")
            except ValueError:
                pass
    log.check(
        "evidence_provenance_fields",
        not provenance_problems,
        (
            f"problems={provenance_problems[:8]} "
            f"resolved_records={len(resolved)}"
        ),
    )

    # ---- No-asset / no-training claims -------------------------------------
    claim_problems: list[str] = []
    if catalog.get("training_admitted") is not False:
        claim_problems.append("catalog.training_admitted")
    if catalog.get("auto_admit") is not False:
        claim_problems.append("catalog.auto_admit")
    if catalog.get("media_assets_downloaded") != 0:
        claim_problems.append("catalog.media_assets_downloaded")
    if catalog.get("model_runs") != 0:
        claim_problems.append("catalog.model_runs")
    if evidence.get("media_assets_downloaded") != 0:
        claim_problems.append("evidence.media_assets_downloaded")
    if evidence.get("bulk_training_annotations_downloaded") != 0:
        claim_problems.append("evidence.bulk_training_annotations_downloaded")
    if evidence.get("small_provenance_csvs_downloaded") != 3:
        claim_problems.append("evidence.small_provenance_csvs_downloaded")
    if len(csv_records) != 3:
        claim_problems.append("evidence.csv_record_count")
    log.check(
        "no_asset_or_training_claims",
        not claim_problems,
        "problems=" + ",".join(claim_problems),
    )
    admitted_families = sorted(
        str(family.get("id")) for family in families if family.get("training_admitted") is not False
    )
    auto_admit_families = sorted(
        str(family.get("id")) for family in families if family.get("auto_admit") is not False
    )
    audited_families = sorted(
        str(family.get("id")) for family in families if family.get("media_audited") is not False
    )
    log.check(
        "no_family_admission",
        not admitted_families and not auto_admit_families and not audited_families,
        (
            f"training_admitted={admitted_families} auto_admit={auto_admit_families} "
            f"media_audited={audited_families}"
        ),
    )

    # ---- Rights, provenance and split requirements -------------------------
    rights_problems: list[str] = []
    for family in families:
        family_id = family.get("id", "?")
        text = str(family.get("rights_review") or "")
        if len(text.strip()) < 20 or not _has_rights_vocabulary(text):
            rights_problems.append(f"{family_id}:rights_review")
        provenance = str(family.get("annotation_provenance") or "")
        if len(provenance.strip()) < 20:
            rights_problems.append(f"{family_id}:annotation_provenance")
        keys = family.get("split_group_keys")
        if (
            not isinstance(keys, list)
            or not keys
            or not all(_nonempty_str(key) for key in keys)
            or len(set(keys)) != len(keys)
        ):
            rights_problems.append(f"{family_id}:split_group_keys")
        checks_rows = family.get("required_checks")
        if (
            not isinstance(checks_rows, list)
            or len(checks_rows) < 2
            or not all(_nonempty_str(row) for row in checks_rows)
        ):
            rights_problems.append(f"{family_id}:required_checks")
    log.check(
        "rights_split_provenance_requirements",
        not rights_problems,
        "problems=" + ",".join(sorted(rights_problems))[:600],
    )
    restricted_family_problems: list[str] = []
    for family in families:
        if family.get("priority") not in RESTRICTED_PRIORITIES:
            continue
        text = str(family.get("rights_review") or "")
        if not _has_restriction_vocabulary(text):
            restricted_family_problems.append(f"{family.get('id')}:rights_text")
        if family.get("training_admitted") is not False:
            restricted_family_problems.append(f"{family.get('id')}:admitted")
    log.check(
        "restricted_family_rights",
        not restricted_family_problems,
        "problems=" + ",".join(restricted_family_problems),
    )
    restricted_license_problems: list[str] = []
    unknown_license_problems: list[str] = []
    for _, repository in entries:
        record = metadata_by_repo.get(repository.get("repo_id"))
        if record is None:
            continue
        license_values = _license_values((record.get("card_data") or {}).get("license"))
        license_text = " ".join(license_values)
        family = next(
            family
            for family, candidate in entries
            if candidate is repository
        )
        priority = family.get("priority")
        restricted_license = (
            "noncommercial" in license_text
            or "other" in license_values
            or "nc-" in license_text
            or "-nc" in license_text
        )
        if restricted_license and priority not in RESTRICTED_LICENSE_PRIORITIES:
            restricted_license_problems.append(
                f"{repository.get('repo_id')}:{priority}"
            )
        if not license_values and str(priority).startswith("P1"):
            unknown_license_problems.append(f"{repository.get('repo_id')}:{priority}")
    log.check(
        "restricted_license_priority",
        not restricted_license_problems,
        "problems=" + ",".join(restricted_license_problems),
    )
    log.check(
        "unknown_license_priority",
        not unknown_license_problems,
        "problems=" + ",".join(unknown_license_problems),
    )
    gated_problems = sorted(
        f"{repository.get('repo_id')}:{family.get('priority')}"
        for family, repository in entries
        if repository.get("gated_metadata")
        and family.get("priority") not in RESTRICTED_LICENSE_PRIORITIES
    )
    log.check(
        "gated_repository_scope",
        not gated_problems,
        "problems=" + ",".join(gated_problems),
    )
    eval_family_problems: list[str] = []
    for family in families:
        if family.get("priority") == "EVAL" and family.get("role") != "evaluation_only":
            eval_family_problems.append(f"{family.get('id')}:role")
        if family.get("role") == "evaluation_only" and family.get("priority") != "EVAL":
            eval_family_problems.append(f"{family.get('id')}:priority")
    log.check(
        "eval_family_scope",
        not eval_family_problems,
        "problems=" + ",".join(eval_family_problems),
    )

    # ---- Clotho provenance audit -------------------------------------------
    clotho_problems: list[str] = []
    expected_splits = {
        "development": (3839, 1577, 1572),
        "validation": (1045, None, None),
        "evaluation": (1045, None, None),
    }
    clotho_total = 0
    license_totals = 0
    for split_name, (rows_expected, cc0_expected, by_expected) in expected_splits.items():
        block = clotho.get(split_name)
        if not isinstance(block, dict):
            clotho_problems.append(f"{split_name}:missing")
            continue
        rows = block.get("rows")
        unique = block.get("unique_filenames")
        licenses = block.get("license_counts")
        if rows != rows_expected:
            clotho_problems.append(f"{split_name}:rows={rows!r}")
        if unique != rows:
            clotho_problems.append(f"{split_name}:unique_filenames={unique!r}")
        if not isinstance(licenses, dict) or not licenses:
            clotho_problems.append(f"{split_name}:license_counts")
            continue
        if not all(
            isinstance(count, int) and not isinstance(count, bool) and count > 0
            for count in licenses.values()
        ):
            clotho_problems.append(f"{split_name}:license_values")
            continue
        if rows not in (None, sum(licenses.values())):
            clotho_problems.append(f"{split_name}:license_sum")
        clotho_total += rows if isinstance(rows, int) else 0
        license_totals += sum(licenses.values())
        if cc0_expected is not None:
            cc0 = licenses.get("http://creativecommons.org/publicdomain/zero/1.0/")
            by = licenses.get("http://creativecommons.org/licenses/by/3.0/")
            if cc0 != cc0_expected:
                clotho_problems.append(f"{split_name}:cc0={cc0!r}")
            if by != by_expected:
                clotho_problems.append(f"{split_name}:cc_by={by!r}")
    if clotho_total != 5929:
        clotho_problems.append(f"total={clotho_total}")
    if license_totals != 5929:
        clotho_problems.append(f"license_total={license_totals}")
    log.check(
        "clotho_audit_counts",
        not clotho_problems,
        "problems=" + ",".join(clotho_problems) + f" total={clotho_total}",
    )
    log.check(
        "clotho_document_numbers",
        "3,839/1,045/1,045" in readme_text
        and "5,929" in readme_text
        and "1,577 CC0 and 1,572 CC-BY" in readme_text
        and "3,839 development, 1,045 validation and 1,045 evaluation" in plan_text,
        "README/plan Clotho counts agree with the provenance audit",
    )

    # ---- Document claims and local links -----------------------------------
    readme_phrases = (
        "36-family machine-readable catalog",
        "inspected 42 Hugging Face repositories",
        "125 retrieval records, 121 successful hashed documents/metadata responses "
        "and four unresolved card/API requests",
        "No audio, image or video assets, model weights, or bulk training annotation "
        "collections were downloaded.",
    )
    missing_readme = [phrase for phrase in readme_phrases if phrase not in readme_text]
    log.check(
        "readme_claims",
        not missing_readme,
        "missing=" + " | ".join(missing_readme),
    )
    plan_phrases = (
        "36 source families and 42 Hugging Face repositories",
        "No image, audio or video assets were acquired or audited; no training was run.",
        "25 GiB asset ceiling",
        "a bounded asset audit",
        "Research shortlist, not training admission.",
    )
    missing_plan = [phrase for phrase in plan_phrases if phrase not in plan_text]
    log.check(
        "plan_claims",
        not missing_plan,
        "missing=" + " | ".join(missing_plan),
    )
    link_problems: list[str] = []
    for document, text in ((REPORT_README, texts[REPORT_README]), (PLAN, texts[PLAN])):
        for target in _local_markdown_links(text):
            resolved_target = (root / document).parent / target
            if not resolved_target.exists():
                link_problems.append(f"{document} -> {target}")
    log.check(
        "local_links_resolve",
        not link_problems,
        "problems=" + ",".join(link_problems)[:600],
    )
    prior_checks = prior_validation.get("checks")
    prior_rows = [
        row for row in prior_checks if isinstance(row, dict)
    ] if isinstance(prior_checks, list) else []
    prior_passed = sum(1 for row in prior_rows if row.get("passed") is True)
    prior_failed = sum(1 for row in prior_rows if row.get("passed") is False)
    log.check(
        "prior_documentation_report_consistent",
        isinstance(prior_checks, list)
        and len(prior_rows) == len(prior_checks) == 426
        and prior_validation.get("passed") == prior_passed == 426
        and prior_validation.get("failed") == prior_failed == 0
        and prior_validation.get("model_runs") == 0
        and prior_validation.get("media_assets_downloaded") == 0,
        (
            "The retained 426-check report is arithmetically consistent but its "
            "generator/verifier was not retained; it is not treated as replay evidence."
        ),
    )

    # ---- Repository snapshot and runtime registry --------------------------
    snapshot_files = snapshot.get("files")
    snapshot_problems: list[str] = []
    if not re.fullmatch(r"[0-9a-f]{40}", str(snapshot.get("head", ""))):
        snapshot_problems.append("head")
    if snapshot.get("protected_runtime_file_accessed") is not False:
        snapshot_problems.append("protected_runtime_file_accessed")
    if not isinstance(snapshot_files, list) or len(snapshot_files) != 5:
        snapshot_problems.append("files")
    pinned_hashes: dict[str, str] = {}
    for row in snapshot_files if isinstance(snapshot_files, list) else []:
        if not isinstance(row, dict):
            snapshot_problems.append("file_row")
            continue
        relative = row.get("path")
        pinned = row.get("sha256")
        if not _nonempty_str(relative) or not _is_sha256(pinned):
            snapshot_problems.append("file_entry")
            continue
        pinned_hashes[str(relative)] = str(pinned)
        if not (root / str(relative)).is_file():
            snapshot_problems.append(f"missing:{relative}")
    log.check(
        "snapshot_metadata",
        not snapshot_problems,
        "problems=" + ",".join(snapshot_problems),
    )
    registry_hash = next(
        row["sha256"] for row in inputs if row["path"] == REGISTRY
    )
    registry_pin = pinned_hashes.get(REGISTRY)
    log.check(
        "registry_pin_matches",
        registry_pin == registry_hash,
        (
            f"pinned={registry_pin} actual={registry_hash} "
            "(the runtime registry must remain unchanged)"
        ),
    )
    registry_entries = registry.get("entries")
    registry_problems: list[str] = []
    if registry.get("schema_version") != "replica.dataset_registry.v1":
        registry_problems.append("schema_version")
    if registry.get("status") != "research_catalog_not_an_ingestion_manifest":
        registry_problems.append("status")
    policy = registry.get("global_policy")
    required_policy = {
        "all_entries_default_quarantined": True,
        "require_immutable_revision": True,
        "require_file_hashes_on_ingestion": True,
        "require_license_review": True,
        "require_group_split_audit": True,
        "benchmark_targets_training_allowed": False,
        "public_availability_implies_training_permission": False,
        "external_teacher_personal_data_allowed_by_default": False,
        "copy_dataset_into_starter_archive": False,
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) is not value for key, value in required_policy.items()
    ):
        registry_problems.append("global_policy")
    if not isinstance(registry_entries, list) or len(registry_entries) != 13:
        registry_problems.append(f"entries={len(registry_entries) if isinstance(registry_entries, list) else '?'}")
        registry_entries = registry_entries if isinstance(registry_entries, list) else []
    for entry in registry_entries:
        if not isinstance(entry, dict):
            registry_problems.append("entry_type")
            continue
        if entry.get("auto_admit") is not False:
            registry_problems.append(f"{entry.get('source_key')}:auto_admit")
        if entry.get("ingestion_status") != "not_downloaded":
            registry_problems.append(f"{entry.get('source_key')}:ingestion_status")
        if not str(entry.get("default_eligibility", "")).startswith("quarantined"):
            registry_problems.append(f"{entry.get('source_key')}:eligibility")
    log.check(
        "registry_quarantine_policy",
        not registry_problems,
        (
            f"problems={registry_problems[:8]} entries={len(registry_entries)} "
            "admitted=0"
        ),
    )
    frozen_hash = next(
        row["sha256"] for row in inputs if row["path"] == FROZEN_ARCHITECTURE
    )
    log.check(
        "frozen_architecture_pin",
        frozen_hash == snapshot.get("frozen_architecture_sha256"),
        (
            f"pinned={snapshot.get('frozen_architecture_sha256')} "
            f"actual={frozen_hash}"
        ),
    )
    earlier_sources = earlier.get("sources")
    log.check(
        "earlier_catalog_inventory",
        earlier.get("schema_version") == "replica.research_dataset_catalog.v1"
        and earlier.get("external_sources") == 34
        and earlier.get("pinned_metadata_sources") == 22
        and isinstance(earlier_sources, list)
        and len(earlier_sources) == 34
        and earlier.get("dataset_assets_downloaded_in_this_update") == 0
        and earlier.get("training_admitted_in_this_update") == 0
        and earlier.get("existing_runtime_registry") == REGISTRY,
        (
            f"external={earlier.get('external_sources')!r} "
            f"pins={earlier.get('pinned_metadata_sources')!r} "
            f"sources={len(earlier_sources) if isinstance(earlier_sources, list) else '?'}"
        ),
    )
    log.check(
        "input_hashes_recorded",
        len(inputs) == len(INPUT_ROLES)
        and all(_is_sha256(row["sha256"]) and row["bytes"] > 0 for row in inputs),
        f"inputs={len(inputs)} expected={len(INPUT_ROLES)}",
    )

    # Repository snapshot pins are historical: record drift, but only the
    # runtime registry is a hard contradiction.  The snapshot itself notes that
    # concurrent source edits were not reviewed or altered.
    drift: list[dict] = []
    matched: list[str] = []
    for relative, pinned in sorted(pinned_hashes.items()):
        actual = _sha256_file(root / relative)
        if actual == pinned:
            matched.append(relative)
        else:
            drift.append({"path": relative, "pinned_sha256": pinned, "actual_sha256": actual})
            warnings.append(
                f"pinned snapshot drift: {relative} differs from the repository "
                f"snapshot (pinned {pinned}, actual {actual}); the snapshot records "
                "a historical capture and concurrent edits were not reviewed."
            )

    priority_counts: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    families_with_multiple: dict[str, int] = {}
    for family in families:
        priority = str(family.get("priority"))
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        for modality in family.get("modalities") or []:
            key = str(modality)
            modality_counts[key] = modality_counts.get(key, 0) + 1
        repositories = family.get("hugging_face") or []
        if len(repositories) > 1:
            families_with_multiple[str(family.get("id"))] = len(repositories)

    unresolved_records = [
        {
            "name": source.get("name"),
            "url": source.get("url"),
            "error": source.get("error"),
        }
        for source in sorted(unresolved, key=lambda row: str(row.get("name")))
    ]

    failures = log.failures
    status = "contradiction_refused" if failures else "pass_metadata_only"
    summary = (
        "Metadata-only validation passed: "
        f"{len(families)} source families, {len(metadata_by_repo)} inspected Hugging Face "
        f"repositories ({len(entries)} catalog pins plus {len(lookup_repos)} documented "
        f"lookups), {len(evidence_sources)} evidence records "
        f"({len(resolved)} hashed, {len(unresolved)} unresolved), "
        "0 media assets, 0 asset QA passes, 0 training admissions. "
        "Rights clearance, duplicate-graph execution, split enforcement and registry "
        "admission are not claimed."
    )
    if failures:
        summary = (
            "Inputs contradict the documented metadata-only research claims; "
            f"{len(failures)} of {len(log.rows)} checks failed. No pass is asserted."
        )

    validation = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sparkx_dataset_metadata_validation",
        "verifier": "scripts/verify_sparkx_dataset_research.py",
        "review_date": REVIEW_DATE,
        "scope": (
            "Offline documentation, inventory, evidence-hash and cross-reference "
            "validation of the SparkX dataset research package. No network access, "
            "no media bytes, no dataset loaders, no model runs."
        ),
        "verdict": {
            "status": status,
            "metadata_only": True,
            "training_admitted": False,
            "media_assets_downloaded": 0,
            "asset_qa_run": False,
            "registry_unchanged": registry_pin == registry_hash,
            "failed_check_count": len(failures),
            "total_check_count": len(log.rows),
            "summary": summary,
        },
        "inputs": sorted(inputs, key=lambda row: row["path"]),
        "inventory": {
            "source_families": len(families),
            "source_family_ids": sorted(unique_family_ids),
            "hugging_face_repositories_inspected": len(metadata_by_repo),
            "catalog_repository_entries": len(entries),
            "catalog_unique_repositories": len(repo_ids),
            "lookup_repositories": sorted(lookup_repos),
            "evidence_records": len(evidence_sources),
            "evidence_hashed": len(resolved),
            "evidence_unresolved": len(unresolved),
            "evidence_by_kind": evidence_by_kind,
            "evidence_max_document_bytes": max_bytes,
            "priority_counts": priority_counts,
            "modality_counts": modality_counts,
            "families_with_multiple_repositories": families_with_multiple,
            "reported_config_count": config_count,
            "reported_split_count": split_count,
            "repositories_without_reported_split_metadata": repos_without_split_metadata,
            "training_admitted_families": len(admitted_families),
            "media_audited_families": len(audited_families),
            "media_assets_downloaded": 0,
            "model_runs": 0,
            "registry_entries": len(registry_entries),
            "prior_documentation_validation_checks": len(prior_rows),
            "prior_documentation_validation_replayable": False,
            "proposed_asset_ceiling_gib": PROPOSED_ASSET_CEILING_GIB,
            "proposed_asset_ceiling_status": "prospective_not_authorization",
        },
        "unresolved_records": unresolved_records,
        "source_pins": {
            "registry_unchanged": registry_pin == registry_hash,
            "registry_sha256": registry_hash,
            "frozen_architecture_sha256": frozen_hash,
            "snapshot_matched_files": matched,
            "snapshot_drifted_files": drift,
            "note": (
                "The runtime registry hash is a hard requirement. Other snapshot "
                "entries are historical captures; drift is recorded as a warning."
            ),
        },
        "checks": log.rows,
        "failures": failures,
        "warnings": sorted(warnings),
        "limitations": [
            "Metadata, card, paper and provenance-audit text only; no image, audio or "
            "video bytes, decoded waveforms, labels or model behavior were inspected.",
            "No asset QA, rights clearance, license compatibility review, redistribution "
            "review or intended-use determination was performed.",
            "Source-overlap/duplicate graphs are documented as requirements, not executed; "
            "no perceptual or exact-media deduplication was run.",
            "Split keys are recorded requirements; no split was materialized or enforced.",
            "Hugging Face revisions pin repository files, not bytes served by external "
            "media URLs; publisher row counts are not counts of unique or licensed assets.",
            "Evidence SHA-256 pins are checked for format, coverage, uniqueness and "
            "cross-consistency with the metadata snapshot; the originally retrieved "
            "bytes live in a temporary cache outside this repository and were not "
            "re-hashed by this clean-export verifier.",
            "The earlier 426-check documentation report is retained as an input and "
            "checked only for internal arithmetic consistency; its generator/verifier "
            "was not retained, so it is not clean-export replay evidence.",
            "Four evidence requests are unresolved and recorded; missing cards or API "
            "responses are not treated as permission.",
            "The runtime registry was read only to verify its pinned hash and quarantine "
            "policy; it was not modified and no entry was admitted.",
            "The repository snapshot head was not re-verified against git history; "
            "snapshot-pin drift for mutable documents is reported, not repaired.",
            "The 25 GiB asset ceiling and the 50 GiB native RAM guard are prospective "
            "planning constraints, not authorization to download or execute anything.",
            "A passing verdict is not dataset quality, rights, split-safety or "
            "training-admission validation.",
        ],
        "next_steps": {
            "status": "not_started",
            "authorized": False,
            "requirement": (
                "A separately authorized bounded asset audit under an explicit manifest; "
                "see intake_manifest.json. This report does not authorize downloads, "
                "native execution, asset QA or registry admission."
            ),
        },
    }

    for row in validation["checks"]:
        if row["ok"]:
            row["detail"] = row["detail"] or "ok"

    manifest: dict | None = None
    if not failures:
        manifest = _build_intake_manifest(validation, catalog, registry)

    return validation, manifest


def _build_intake_manifest(
    validation: dict, catalog: dict, registry: dict
) -> dict:
    """Build the deterministic prospective intake record (not an ingestion manifest)."""
    families = [
        family for family in catalog.get("source_families", []) if isinstance(family, dict)
    ]
    p1_families = [
        {
            "id": family.get("id"),
            "title": family.get("title"),
            "priority": family.get("priority"),
            "repositories": sorted(
                repository.get("repo_id")
                for repository in family.get("hugging_face", [])
                if isinstance(repository, dict)
            ),
            "training_admitted": False,
            "media_audited": False,
            "rights_status": "unresolved_at_metadata_stage",
        }
        for family in families
        if str(family.get("priority", "")).startswith("P1")
    ]
    restricted_ids = sorted(
        str(family.get("id"))
        for family in families
        if family.get("priority") in RESTRICTED_PRIORITIES
    )
    registry_input = next(
        row for row in validation["inputs"] if row["path"] == REGISTRY
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "prospective_intake_manifest",
        "status": "metadata_only_prospective_not_an_ingestion_manifest",
        "review_date": REVIEW_DATE,
        "metadata_only_not_an_ingestion_manifest": True,
        "training_admitted": False,
        "media_assets_downloaded": 0,
        "asset_qa_not_run": True,
        "registry_unchanged": True,
        "native_execution_not_run": True,
        "authorization": {
            "asset_download_authorized": False,
            "asset_qa_authorized": False,
            "training_authorized": False,
            "registry_admission_authorized": False,
            "note": (
                "Prospective planning record only. It is not an ingestion manifest, "
                "download approval, asset-QA pass or training admission."
            ),
        },
        "source_validation": {
            "validation_report": f"{OUTPUT_DIR}/{VALIDATION_NAME}",
            "validation_verdict": validation["verdict"]["status"],
            "source_family_count": validation["inventory"]["source_families"],
            "repositories_inspected": validation["inventory"][
                "hugging_face_repositories_inspected"
            ],
            "evidence_records": validation["inventory"]["evidence_records"],
            "evidence_unresolved": validation["inventory"]["evidence_unresolved"],
            "unresolved_records": validation["unresolved_records"],
        },
        "proposed_asset_ceiling_gib": PROPOSED_ASSET_CEILING_GIB,
        "proposed_ceiling_status": "prospective_not_authorization",
        "free_space_check": "not_run",
        "native_ram_guard_gib": NATIVE_RAM_GUARD_GIB,
        "native_ram_guard_status": "separate_constraint_not_asset_budget",
        "asset_qa": {
            "status": "not_run",
            "samples_reviewed": 0,
            "authorized": False,
        },
        "rights_resolution": {
            "status": "not_started",
            "cleared_family_count": 0,
            "unresolved_family_count": len(families),
            "restricted_family_ids": restricted_ids,
        },
        "duplicate_graph": {
            "status": "not_executed",
            "documented_edges_only": True,
            "exact_media_matches_measured": 0,
        },
        "split_validation": {
            "status": "requirements_only",
            "enforced_splits": 0,
        },
        "registry": {
            "path": REGISTRY,
            "sha256": registry_input["sha256"],
            "status": registry.get("status"),
            "entries": validation["inventory"]["registry_entries"],
            "admitted_entries": 0,
            "changed": False,
        },
        "proposed_review_scope": {
            "status": "proposed_not_selected_or_authorized",
            "note": (
                "P1 families are listed only as candidate review order. Listing is "
                "not selection, clearance, download approval or training admission."
            ),
            "families": p1_families,
        },
        "next_bounded_asset_audit": {
            "status": "not_started",
            "authorized": False,
            "distinct_source_groups_per_selected_source": 100,
            "scope_note": (
                "The ~100-source-group sample is the plan's suggested first scope, "
                "stratified by task and difficulty; it is not a run authorization."
            ),
            "prerequisites": [
                {
                    "id": "explicit_authorization",
                    "requirement": (
                        "Obtain explicit authorization for a named manifest and byte "
                        "ceiling; this prospective record is not that authorization."
                    ),
                },
                {
                    "id": "unresolved_evidence",
                    "requirement": (
                        "Re-resolve or permanently record the four unresolved evidence "
                        "requests without substituting an unverified mirror."
                    ),
                },
                {
                    "id": "rights_source_map",
                    "requirement": (
                        "Resolve per-source annotation, media, redistribution and "
                        "intended-use terms; no gate may be accepted implicitly."
                    ),
                },
                {
                    "id": "free_space_and_accounting",
                    "requirement": (
                        "Run a free-space check against the proposed 25 GiB ceiling and "
                        "account separately for checkpoints, resampling, feature caches "
                        "and backups; the 50 GiB native RAM guard is a distinct limit."
                    ),
                },
                {
                    "id": "immutable_media_manifest",
                    "requirement": (
                        "Choose explicit immutable revisions and record independent "
                        "media-byte hashes; URL-string hashes do not establish media "
                        "integrity."
                    ),
                },
                {
                    "id": "duplicate_graph_execution",
                    "requirement": (
                        "Execute exact-media and provenance overlap checks across the "
                        "documented source lineages before sampling any asset."
                    ),
                },
                {
                    "id": "bounded_asset_qa",
                    "requirement": (
                        "Sample approximately 100 distinct source groups per selected "
                        "source, stratified by task and difficulty, and review label "
                        "correctness, asset availability and modality necessity."
                    ),
                },
                {
                    "id": "asset_qa_report",
                    "requirement": (
                        "Record QA outcomes in a new asset-QA report; do not retroactively "
                        "convert this prospective manifest into an ingestion manifest."
                    ),
                },
                {
                    "id": "registry_admission",
                    "requirement": (
                        "Seek runtime registry admission separately; the registry remains "
                        "quarantined, unchanged and hash-pinned."
                    ),
                },
            ],
        },
        "blocking_unknowns": [
            "rights clearance not performed for any of the 36 source families",
            "four evidence requests unresolved (see source_validation.unresolved_records)",
            "no asset-level inventory or media-byte hashes exist",
            "duplicate/source-overlap graph documented but not executed",
            "split keys recorded but not materialized or enforced",
            "runtime registry admission not requested",
            "no native execution or dataset-loader run occurred",
        ],
        "explicitly_not_claimed": [
            "this is not an ingestion manifest",
            "no dataset, subset or media asset has been admitted or downloaded",
            "no asset QA, rights clearance, deduplication or split-safety pass occurred",
            "no model training, fine-tuning, native learning or dataset-loader execution occurred",
            "the 25 GiB ceiling is prospective and confers no authorization",
            "publisher counts are not unique, licensed or training-ready asset counts",
        ],
        "limitations": validation["limitations"],
    }


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _dump(payload: dict) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_outputs(
    root: Path, out_dir: Path, validation: dict, manifest: dict | None
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_path = out_dir / VALIDATION_NAME
    validation_bytes = _dump(validation)
    validation_path.write_bytes(validation_bytes)
    manifest_path: Path | None = None
    stale_manifest = out_dir / MANIFEST_NAME
    if manifest is None and stale_manifest.exists():
        # A previous passing run must not leave a prospective intake record that
        # contradicts the current refusal report.
        stale_manifest.unlink()
    if manifest is not None:
        manifest = dict(manifest)
        manifest["source_validation"] = dict(manifest["source_validation"])
        manifest["source_validation"]["validation_report_sha256"] = _sha256_bytes(
            validation_bytes
        )
        manifest_path = out_dir / MANIFEST_NAME
        manifest_path.write_bytes(_dump(manifest))
    return validation_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the SparkX dataset research package offline and emit a "
            "metadata-only report."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing the catalog, evidence and registry",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"output directory (default: <root>/{OUTPUT_DIR})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run checks without writing validation.json or intake_manifest.json",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out_dir = (args.out_dir or (root / OUTPUT_DIR)).resolve()
    try:
        validation, manifest = verify(root)
    except InputError as exc:
        print(f"verification aborted: {exc}", file=sys.stderr)
        return 2

    verdict = validation["verdict"]
    inventory = validation["inventory"]
    print(
        f"{verdict['status']}: {inventory['source_families']} families, "
        f"{inventory['hugging_face_repositories_inspected']} repositories, "
        f"{inventory['evidence_records']} evidence records "
        f"({inventory['evidence_hashed']} hashed, "
        f"{inventory['evidence_unresolved']} unresolved), "
        f"{inventory['media_assets_downloaded']} media assets, "
        f"{inventory['training_admitted_families']} training admissions"
    )
    if validation["failures"]:
        for failure in validation["failures"]:
            print(f"  - {failure['id']}: {failure['detail']}", file=sys.stderr)
    if not args.check_only:
        validation_path, manifest_path = write_outputs(root, out_dir, validation, manifest)
        print(f"wrote {_display_path(root, validation_path)}")
        if manifest_path is not None:
            print(f"wrote {_display_path(root, manifest_path)}")
        elif manifest is None:
            print("intake_manifest.json not written: inputs contradict the claims")
    return 0 if not validation["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
