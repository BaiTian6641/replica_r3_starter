#!/usr/bin/env python3
"""Bounded acquisition and offline verification for the real-image tranche.

This tool implements the protocol frozen in
``configs/real_image_tranche_v1.json``.  It is deliberately narrow:

* it validates the complete protocol before making any network request;
* with ``--download`` it fetches exactly one declared 960 px thumbnail per
  asset, never an original URL and never more than the declared request,
  per-asset byte or total byte budget;
* it decodes every JPEG with Pillow and checks format, geometry, exact byte
  length, SHA-256, exact-hash uniqueness, split leakage and rights fields;
* it stages all bytes outside the final asset directory and only renames the
  completed seven-file directory into place after every check passes;
* without ``--download`` (the default) it recomputes every hash and QA check
  from the local assets and rejects drift against ``manifest.json``.

Evidence is split by phase: download writes a stable ``acquisition.json`` plus
the current ``report.json``; offline verification writes a stable
``verification.json`` plus the current ``report.json`` and never rewrites the
acquisition receipt.  Verification includes validated prior acquisition
evidence separately, so the verify operation's own network/accounting remains
zero while the download phase's request history is preserved.  A missing,
malformed, protocol/manifest-incompatible or tampered acquisition receipt
fails verification closed; a structurally valid refused attempt is excluded
with an explicit warning.

The generated JSON is protocol evidence, not a training authorization.  No
perceptual or near-duplicate detection is implemented or claimed.

Usage:
  python3 scripts/prepare_real_image_tranche.py \
      --config configs/real_image_tranche_v1.json \
      --assets-dir data/external/real_image_tranche_20261026/assets \
      --report-dir reports/real_image_tranche_20261026 --download

  python3 scripts/prepare_real_image_tranche.py \
      --config configs/real_image_tranche_v1.json \
      --assets-dir data/external/real_image_tranche_20261026/assets \
      --report-dir reports/real_image_tranche_20261026

Exit codes: 0 admitted/verified, 1 refused or verification failed, 2 invalid
protocol or missing verification manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "real_image_tranche_v1.json"
DEFAULT_ASSETS_DIR = ROOT / "data" / "external" / "real_image_tranche_20261026" / "assets"
DEFAULT_REPORT_DIR = ROOT / "reports" / "real_image_tranche_20261026"

PROTOCOL_FORMAT = "replica.real_image_tranche.v1"
MANIFEST_FORMAT = "replica.real_image_tranche_manifest.v1"
REPORT_FORMAT = "replica.real_image_tranche_report.v1"
ACQUISITION_FORMAT = "replica.real_image_tranche_acquisition.v1"
VERIFICATION_FORMAT = "replica.real_image_tranche_verification.v1"
PROTOCOL_STATUS = "predeclared_candidate_not_admitted"
MANIFEST_STATUS = "admitted_derivative_set"
MANIFEST_NAME = "manifest.json"
REPORT_NAME = "report.json"
ACQUISITION_NAME = "acquisition.json"
VERIFICATION_NAME = "verification.json"
SPLITS = ("train", "development", "audit")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ASSET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
HOST_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$")
CHUNK_SIZE = 64 * 1024
MAX_DECODED_PIXELS = 50_000_000
ASPECT_RELATIVE_TOLERANCE = 0.001
ASPECT_ABSOLUTE_TOLERANCE = 1.0

SOURCE_BLOCK_FIELDS = (
    "title",
    "pageid",
    "revision_id",
    "revision_timestamp",
    "original_sha1",
    "original_bytes",
    "original_width",
    "original_height",
    "mime",
    "thumb_url",
    "source_page",
    "institutional_source",
    "provenance_evidence",
    "author",
    "credit",
    "license",
    "license_short_name",
    "license_url",
    "attribution_required",
    "copyrighted",
    "restrictions",
    "rights_basis",
)

MANDATORY_ASSET_FIELDS = (
    "asset_id",
    *SOURCE_BLOCK_FIELDS,
)

# These fields are mandatory keys but may legitimately be null/empty in the
# frozen protocol: no license URL exists for public-domain U.S. works, and a
# cleared work may carry no additional restriction text.
NULLABLE_OR_EMPTY_ASSET_FIELDS = frozenset({"license_url", "restrictions"})

ASSET_CHECK_NAMES = (
    "jpeg_format",
    "full_decode",
    "dimensions_positive",
    "derivative_width_exact",
    "aspect_ratio_consistent",
    "pixel_limit",
    "byte_length",
    "sha256",
    "exact_hash_unique",
    "split_matches_protocol",
    "source_group_match",
    "rights_fields",
    "license_allowed",
    "source_not_excluded",
)

NON_CLAIMS = {
    "training_admitted": False,
    "training_authorization": False,
    "perceptual_dedup_claimed": False,
    "perceptual_dedup_performed": False,
    "near_duplicate_detection_claimed": False,
    "model_quality_evaluated": False,
    "native_execution_performed": False,
    "exhaustive_semantic_leakage_detection_claimed": False,
    "source_bytes_are_not_original_media": True,
}


class ProtocolError(ValueError):
    """The frozen protocol is structurally unusable or contradictory."""


class PillowUnavailable(RuntimeError):
    """Pillow is required for the JPEG decode QA and is not installed."""


# Reuse the one frozen split convention instead of creating a second one.
def _load_first_tranche_module():
    path = Path(__file__).resolve().with_name("prepare_sparkx_first_tranche.py")
    name = "_replica_prepare_sparkx_first_tranche"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load split helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_SPLIT_HELPER = _load_first_tranche_module()
assign_split = _SPLIT_HELPER.assign_split


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_json(payload: Any) -> str:
    """Stable JSON serialization for tracked manifest/report evidence."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _display_path(path: Path) -> str:
    candidate = Path(path)
    try:
        resolved = candidate.resolve()
    except OSError:
        return str(candidate)
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{field} must be a non-negative integer")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def _timestamp(value: object, field: str) -> str:
    text = _nonempty_string(value, field)
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ProtocolError(f"{field} must be an RFC3339 UTC timestamp") from error
    return text


def _https_url(value: object, field: str) -> str:
    text = _nonempty_string(value, field)
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ProtocolError(f"{field} must be an absolute HTTPS URL")
    if parsed.username or parsed.password:
        raise ProtocolError(f"{field} must not contain credentials")
    try:
        port = parsed.port
    except ValueError as error:
        raise ProtocolError(f"{field} has an invalid port") from error
    if port not in (None, 443):
        raise ProtocolError(f"{field} must use HTTPS port 443")
    return text


def _host_allowed(host: str | None, allowed_hosts: tuple[str, ...]) -> bool:
    return bool(host) and host.lower() in allowed_hosts


def _validate_derivative_url(
    value: object,
    field: str,
    derivative_width: int,
    allowed_hosts: tuple[str, ...],
) -> str:
    text = _https_url(value, field)
    parsed = urllib.parse.urlsplit(text)
    if not _host_allowed(parsed.hostname, allowed_hosts):
        raise ProtocolError(
            f"{field} host {parsed.hostname!r} is not in allowed_hosts {list(allowed_hosts)}"
        )
    if parsed.query or parsed.fragment:
        raise ProtocolError(f"{field} must not carry a query or fragment")
    path = urllib.parse.unquote(parsed.path)
    if any(segment in {".", ".."} for segment in path.split("/")):
        raise ProtocolError(f"{field} must not contain dot segments")
    if "/thumb/" not in path:
        raise ProtocolError(f"{field} is not a derivative thumbnail URL")
    if f"/{derivative_width}px-" not in path:
        raise ProtocolError(
            f"{field} is not a declared {derivative_width}px derivative"
        )
    return text


def _validate_remote_url(
    value: str,
    derivative_width: int,
    allowed_hosts: tuple[str, ...],
    context: str,
) -> str:
    """Validate HTTPS, host and derivative-path after a redirect.

    Redirect targets may legitimately carry a signed CDN query string even
    though the declared protocol URL does not, so the query is not rejected
    here.  Scheme, host, credentials, port and the derivative path remain
    mandatory.
    """
    text = _https_url(value, context)
    parsed = urllib.parse.urlsplit(text)
    if not _host_allowed(parsed.hostname, allowed_hosts):
        raise ProtocolError(
            f"{context} host {parsed.hostname!r} is not in allowed_hosts "
            f"{list(allowed_hosts)}"
        )
    path = urllib.parse.unquote(parsed.path)
    if any(segment in {".", ".."} for segment in path.split("/")):
        raise ProtocolError(f"{context} must not contain dot segments")
    if "/thumb/" not in path:
        raise ProtocolError(f"{context} is not a derivative thumbnail URL")
    if f"/{derivative_width}px-" not in path:
        raise ProtocolError(
            f"{context} is not a declared {derivative_width}px derivative"
        )
    return text


def _rights_fields_present(record: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    if not set(policy["required_fields"]) <= set(record):
        return False
    for field in MANDATORY_ASSET_FIELDS:
        if field not in record:
            return False
        if field in NULLABLE_OR_EMPTY_ASSET_FIELDS:
            continue
        if record[field] in (None, "", [], {}):
            return False
    return True


def _license_allowed(license_value: object, policy: Mapping[str, Any]) -> bool:
    if not isinstance(license_value, str) or not license_value:
        return False
    lowered = license_value.strip().lower()
    if lowered in {str(item).lower() for item in policy["deny_exact"]}:
        return False
    if any(lowered.startswith(str(prefix).lower()) for prefix in policy["deny_prefixes"]):
        return False
    return lowered in {str(item).lower() for item in policy["allow"]}


def _benchmark_excluded(
    source_id: str,
    group_key: str,
    asset_id: str,
    leakage: Mapping[str, Any],
) -> bool:
    identifiers = (source_id.lower(), group_key.lower(), asset_id.lower())
    excluded = {str(item).lower() for item in leakage["excluded_source_ids"]}
    fragments = [str(item).lower() for item in leakage["excluded_source_id_fragments"]]
    for identifier in identifiers:
        if identifier in excluded:
            return True
        if any(fragment and fragment in identifier for fragment in fragments):
            return True
    return False


# ---------------------------------------------------------------------------
# protocol validation
# ---------------------------------------------------------------------------


@dataclass
class ProtocolPlan:
    protocol: dict
    path: str
    sha256: str
    derivative_width: int
    assets: tuple[dict, ...]
    group_splits: dict
    asset_counts: dict
    expected_group_splits: dict
    expected_asset_counts: dict
    exact_assets: int
    exact_sources: int
    max_requests: int
    max_asset_bytes: int
    max_total_bytes: int
    timeout_seconds: float
    allowed_hosts: tuple[str, ...]
    user_agent: str


def validate_protocol(protocol: object) -> dict:
    """Validate every protocol invariant before any network request.

    Returns the derived plan values.  Raises :class:`ProtocolError` for any
    contradiction, unsupported value or missing/duplicated identity.
    """
    if not isinstance(protocol, dict):
        raise ProtocolError("protocol root must be a JSON object")
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ProtocolError(f"unsupported protocol format: {protocol.get('format')!r}")
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ProtocolError(
            f"protocol status must be {PROTOCOL_STATUS!r}, got {protocol.get('status')!r}"
        )
    _nonempty_string(protocol.get("purpose"), "purpose")
    _timestamp(protocol.get("declared_at_utc"), "declared_at_utc")

    jurisdiction = protocol.get("jurisdiction")
    if not isinstance(jurisdiction, dict):
        raise ProtocolError("jurisdiction must be an object")
    if jurisdiction.get("intended") != "US":
        raise ProtocolError("jurisdiction.intended must be 'US'")
    _nonempty_string(jurisdiction.get("basis"), "jurisdiction.basis")

    acquisition = protocol.get("acquisition")
    if not isinstance(acquisition, dict):
        raise ProtocolError("acquisition must be an object")
    if acquisition.get("download_authorized") is not True:
        raise ProtocolError("acquisition.download_authorized must be true")
    if acquisition.get("bulk_download") is not False:
        raise ProtocolError("acquisition.bulk_download must be false")
    if acquisition.get("originals_downloaded") is not False:
        raise ProtocolError("acquisition.originals_downloaded must be false")
    derivative_width = _positive_int(
        acquisition.get("derivative_width"), "acquisition.derivative_width"
    )
    max_asset_bytes = _positive_int(
        acquisition.get("max_asset_bytes"), "acquisition.max_asset_bytes"
    )
    max_total_bytes = _positive_int(
        acquisition.get("max_total_bytes"), "acquisition.max_total_bytes"
    )
    if max_total_bytes < max_asset_bytes:
        raise ProtocolError("max_total_bytes must be at least max_asset_bytes")
    timeout = acquisition.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ProtocolError("acquisition.timeout_seconds must be positive")
    user_agent = _nonempty_string(
        acquisition.get("user_agent"), "acquisition.user_agent"
    )
    hosts_value = acquisition.get("allowed_hosts")
    if not isinstance(hosts_value, list) or not hosts_value:
        raise ProtocolError("acquisition.allowed_hosts must be a non-empty list")
    allowed_hosts: list[str] = []
    for raw_host in hosts_value:
        host = _nonempty_string(raw_host, "acquisition.allowed_hosts[]").lower()
        if "://" in host or "/" in host or ":" in host or not HOST_PATTERN.fullmatch(host):
            raise ProtocolError(f"invalid allowed host: {host!r}")
        if host in allowed_hosts:
            raise ProtocolError(f"duplicate allowed host: {host}")
        allowed_hosts.append(host)

    caps = protocol.get("caps")
    if not isinstance(caps, dict):
        raise ProtocolError("caps must be an object")
    exact_assets = _positive_int(caps.get("exact_assets"), "caps.exact_assets")
    exact_sources = _positive_int(caps.get("exact_sources"), "caps.exact_sources")
    duplicate_rate = caps.get("max_duplicate_rate")
    if isinstance(duplicate_rate, bool) or duplicate_rate != 0.0:
        raise ProtocolError("caps.max_duplicate_rate must be exactly 0.0")
    basis_points = []
    for split in SPLITS:
        field = f"{split}_basis_points"
        basis_points.append(_positive_int(caps.get(field), f"caps.{field}"))
    if sum(basis_points) != 10000:
        raise ProtocolError("split basis points must sum to 10000")
    required_nonempty = caps.get("require_nonempty_splits")
    if not isinstance(required_nonempty, list) or not required_nonempty:
        raise ProtocolError("caps.require_nonempty_splits must be a non-empty list")
    if len(set(required_nonempty)) != len(required_nonempty):
        raise ProtocolError("caps.require_nonempty_splits contains duplicates")
    for split in required_nonempty:
        if split not in SPLITS:
            raise ProtocolError(f"unknown required split: {split!r}")
    if set(required_nonempty) != set(SPLITS):
        raise ProtocolError("all three splits must be required to be non-empty")

    split_assignment = protocol.get("split_assignment")
    if not isinstance(split_assignment, dict):
        raise ProtocolError("split_assignment must be an object")
    if split_assignment.get("method") != "sha256_first_64_bits_mod_10000":
        raise ProtocolError("unsupported split_assignment.method")
    salt = _nonempty_string(split_assignment.get("salt"), "split_assignment.salt")
    if split_assignment.get("group_key_required") is not True:
        raise ProtocolError("split_assignment.group_key_required must be true")
    if split_assignment.get("group_key_scope") != "global":
        raise ProtocolError("split_assignment.group_key_scope must be 'global'")
    expected_group_splits = split_assignment.get("expected_group_splits")
    if not isinstance(expected_group_splits, dict) or not expected_group_splits:
        raise ProtocolError("split_assignment.expected_group_splits must be non-empty")
    expected_asset_counts = split_assignment.get("expected_asset_counts")
    if not isinstance(expected_asset_counts, dict):
        raise ProtocolError("split_assignment.expected_asset_counts must be an object")

    license_policy = protocol.get("license_policy")
    if not isinstance(license_policy, dict):
        raise ProtocolError("license_policy must be an object")
    for key in ("allow", "deny_prefixes", "deny_exact", "required_fields"):
        values = license_policy.get(key)
        if not isinstance(values, list) or not values:
            raise ProtocolError(f"license_policy.{key} must be a non-empty list")
        if not all(isinstance(item, str) and item for item in values):
            raise ProtocolError(f"license_policy.{key} entries must be strings")
    allow_lower = {item.lower() for item in license_policy["allow"]}
    deny_lower = {item.lower() for item in license_policy["deny_exact"]}
    if allow_lower & deny_lower:
        raise ProtocolError("license_policy allow/deny_exact overlap")
    for prefix in license_policy["deny_prefixes"]:
        if any(allowed.startswith(prefix.lower()) for allowed in allow_lower):
            raise ProtocolError("license_policy allow value matches a deny prefix")
    if not set(license_policy["required_fields"]) <= set(MANDATORY_ASSET_FIELDS):
        missing = sorted(set(license_policy["required_fields"]) - set(MANDATORY_ASSET_FIELDS))
        raise ProtocolError(f"required_fields contains unsupported fields: {missing}")

    dedup = protocol.get("dedup")
    if not isinstance(dedup, dict):
        raise ProtocolError("dedup must be an object")
    if dedup.get("exact_sha256") is not True:
        raise ProtocolError("dedup.exact_sha256 must be true")
    for key in ("duplicate_asset_id", "duplicate_source_identity"):
        if dedup.get(key) != "reject":
            raise ProtocolError(f"dedup.{key} must be 'reject'")

    leakage = protocol.get("leakage")
    if not isinstance(leakage, dict):
        raise ProtocolError("leakage must be an object")
    for key in ("excluded_source_ids", "excluded_source_id_fragments", "checks"):
        values = leakage.get(key)
        if not isinstance(values, list) or not values:
            raise ProtocolError(f"leakage.{key} must be a non-empty list")
        if not all(isinstance(item, str) and item for item in values):
            raise ProtocolError(f"leakage.{key} entries must be strings")

    sources = protocol.get("sources")
    if not isinstance(sources, list):
        raise ProtocolError("sources must be a list")
    if len(sources) != exact_sources:
        raise ProtocolError(
            f"exact source count mismatch: declared {exact_sources}, found {len(sources)}"
        )

    source_ids: set[str] = set()
    group_keys: set[str] = set()
    asset_ids: set[str] = set()
    source_pages: set[str] = set()
    page_ids: set[int] = set()
    thumb_urls: set[str] = set()
    group_splits: dict[str, str] = {}
    group_sources: dict[str, str] = {}
    asset_counts = {split: 0 for split in SPLITS}
    assets: list[dict] = []
    required_fields = set(license_policy["required_fields"])

    for source_index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ProtocolError(f"sources[{source_index}] must be an object")
        source_id = _nonempty_string(source.get("source_id"), "source_id")
        if not SOURCE_ID_PATTERN.fullmatch(source_id):
            raise ProtocolError(f"invalid source_id: {source_id!r}")
        if source_id in source_ids:
            raise ProtocolError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        group_key = _nonempty_string(source.get("group_key"), f"{source_id}.group_key")
        if not SOURCE_ID_PATTERN.fullmatch(group_key):
            raise ProtocolError(f"invalid group_key: {group_key!r}")
        if group_key in group_keys:
            raise ProtocolError(f"duplicate group_key: {group_key}")
        group_keys.add(group_key)
        if source.get("jurisdiction") != "US":
            raise ProtocolError(f"source {source_id} jurisdiction must be 'US'")
        raw_assets = source.get("assets")
        if not isinstance(raw_assets, list) or not raw_assets:
            raise ProtocolError(f"source {source_id} must contain assets")

        computed_split = assign_split(group_key, salt, caps)
        if computed_split not in SPLITS:
            raise ProtocolError(f"split helper returned an unknown split for {group_key}")
        group_splits[group_key] = computed_split
        group_sources[group_key] = source_id

        for asset_index, asset in enumerate(raw_assets):
            prefix = f"{source_id}.assets[{asset_index}]"
            if not isinstance(asset, dict):
                raise ProtocolError(f"{prefix} must be an object")
            missing = [
                field
                for field in MANDATORY_ASSET_FIELDS
                if field not in asset
                or (
                    field not in NULLABLE_OR_EMPTY_ASSET_FIELDS
                    and asset[field] in (None, "", [], {})
                )
            ]
            if missing:
                raise ProtocolError(f"{prefix} missing required fields: {missing}")
            policy_missing = sorted(required_fields - set(asset))
            if policy_missing:
                raise ProtocolError(
                    f"{prefix} missing license_policy.required_fields: {policy_missing}"
                )

            asset_id = _nonempty_string(asset.get("asset_id"), f"{prefix}.asset_id")
            if not ASSET_ID_PATTERN.fullmatch(asset_id):
                raise ProtocolError(f"invalid asset_id: {asset_id!r}")
            if asset_id in asset_ids:
                raise ProtocolError(f"duplicate asset_id: {asset_id}")
            asset_ids.add(asset_id)

            _nonempty_string(asset.get("title"), f"{prefix}.title")
            pageid = _positive_int(asset.get("pageid"), f"{prefix}.pageid")
            if pageid in page_ids:
                raise ProtocolError(f"duplicate source identity pageid: {pageid}")
            page_ids.add(pageid)
            revision_id = _positive_int(asset.get("revision_id"), f"{prefix}.revision_id")
            revision_timestamp = _timestamp(
                asset.get("revision_timestamp"), f"{prefix}.revision_timestamp"
            )
            original_sha1 = _nonempty_string(
                asset.get("original_sha1"), f"{prefix}.original_sha1"
            ).lower()
            if not SHA1_PATTERN.fullmatch(original_sha1):
                raise ProtocolError(f"{prefix}.original_sha1 must be 40 hex characters")
            original_bytes = _positive_int(
                asset.get("original_bytes"), f"{prefix}.original_bytes"
            )
            original_width = _positive_int(
                asset.get("original_width"), f"{prefix}.original_width"
            )
            original_height = _positive_int(
                asset.get("original_height"), f"{prefix}.original_height"
            )
            if asset.get("mime") != "image/jpeg":
                raise ProtocolError(f"{prefix}.mime must be image/jpeg")
            thumb_url = _validate_derivative_url(
                asset.get("thumb_url"),
                f"{prefix}.thumb_url",
                derivative_width,
                tuple(allowed_hosts),
            )
            if thumb_url in thumb_urls:
                raise ProtocolError(f"duplicate derivative source identity: {thumb_url}")
            thumb_urls.add(thumb_url)
            source_page = _https_url(asset.get("source_page"), f"{prefix}.source_page")
            if source_page in source_pages:
                raise ProtocolError(f"duplicate source identity: {source_page}")
            source_pages.add(source_page)
            institutional_source = _https_url(
                asset.get("institutional_source"), f"{prefix}.institutional_source"
            )
            evidence = asset.get("provenance_evidence")
            if not isinstance(evidence, list) or not evidence:
                raise ProtocolError(f"{prefix}.provenance_evidence must be non-empty")
            for evidence_index, value in enumerate(evidence):
                _https_url(value, f"{prefix}.provenance_evidence[{evidence_index}]")
            _nonempty_string(asset.get("author"), f"{prefix}.author")
            _nonempty_string(asset.get("credit"), f"{prefix}.credit")
            _nonempty_string(asset.get("rights_basis"), f"{prefix}.rights_basis")
            _nonempty_string(asset.get("license_short_name"), f"{prefix}.license_short_name")
            license_url = asset.get("license_url")
            if license_url is not None:
                _https_url(license_url, f"{prefix}.license_url")
            for boolean_field in ("attribution_required", "copyrighted"):
                if not isinstance(asset.get(boolean_field), bool):
                    raise ProtocolError(f"{prefix}.{boolean_field} must be a boolean")
            if not isinstance(asset.get("restrictions"), str):
                raise ProtocolError(f"{prefix}.restrictions must be a string")
            if not _license_allowed(asset.get("license"), license_policy):
                raise ProtocolError(
                    f"{prefix}.license is not allowed: {asset.get('license')!r}"
                )
            if _benchmark_excluded(source_id, group_key, asset_id, leakage):
                raise ProtocolError(
                    f"{prefix} matches an excluded benchmark/source identity"
                )

            record = dict(asset)
            record.update(
                {
                    "source_id": source_id,
                    "group_key": group_key,
                    "jurisdiction": "US",
                    "split": computed_split,
                    "revision_id": revision_id,
                    "revision_timestamp": revision_timestamp,
                    "original_sha1": original_sha1,
                    "original_bytes": original_bytes,
                    "original_width": original_width,
                    "original_height": original_height,
                    "source_page": source_page,
                    "institutional_source": institutional_source,
                    "thumb_url": thumb_url,
                }
            )
            assets.append(record)
            asset_counts[computed_split] += 1

    if len(assets) != exact_assets:
        raise ProtocolError(
            f"exact asset count mismatch: declared {exact_assets}, found {len(assets)}"
        )
    if set(expected_group_splits) != set(group_splits):
        raise ProtocolError("expected_group_splits does not cover the declared groups exactly")
    for group_key, expected in expected_group_splits.items():
        if expected not in SPLITS:
            raise ProtocolError(f"expected split for {group_key} is invalid")
        if group_splits[group_key] != expected:
            raise ProtocolError(
                f"expected split mismatch for {group_key}: "
                f"computed {group_splits[group_key]}, declared {expected}"
            )
    if set(expected_asset_counts) != set(SPLITS):
        raise ProtocolError("expected_asset_counts must name train, development and audit")
    for split in SPLITS:
        expected = _non_negative_int(
            expected_asset_counts[split], f"expected_asset_counts.{split}"
        )
        if expected != asset_counts[split]:
            raise ProtocolError(
                f"expected asset count mismatch for {split}: "
                f"computed {asset_counts[split]}, declared {expected}"
            )
    for split in required_nonempty:
        if asset_counts[split] <= 0:
            raise ProtocolError(f"required split is empty: {split}")

    max_requests = _positive_int(
        acquisition.get("max_requests"), "acquisition.max_requests"
    )
    if max_requests < exact_assets:
        raise ProtocolError("max_requests is smaller than the exact asset count")

    return {
        "derivative_width": derivative_width,
        "assets": tuple(sorted(assets, key=lambda item: item["asset_id"])),
        "group_splits": dict(sorted(group_splits.items())),
        "asset_counts": dict(asset_counts),
        "expected_group_splits": dict(expected_group_splits),
        "expected_asset_counts": dict(expected_asset_counts),
        "exact_assets": exact_assets,
        "exact_sources": exact_sources,
        "max_requests": max_requests,
        "max_asset_bytes": max_asset_bytes,
        "max_total_bytes": max_total_bytes,
        "timeout_seconds": float(timeout),
        "allowed_hosts": tuple(sorted(allowed_hosts)),
        "user_agent": user_agent,
    }


def load_plan(config_path: str | Path) -> ProtocolPlan:
    path = Path(config_path)
    if not path.is_file():
        raise ProtocolError(f"protocol configuration missing: {path}")
    raw = path.read_bytes()
    try:
        protocol = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"protocol configuration is not valid JSON: {error}") from error
    derived = validate_protocol(protocol)
    return ProtocolPlan(
        protocol=protocol,
        path=_display_path(path),
        sha256=sha256_bytes(raw),
        **derived,
    )


# ---------------------------------------------------------------------------
# JPEG QA
# ---------------------------------------------------------------------------


def _pillow_image():
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - depends on environment
        raise PillowUnavailable(
            "Pillow is required for JPEG decode QA; install Pillow before downloading "
            "or verifying derivative images"
        ) from error
    return Image


def _qa_jpeg(
    data: bytes,
    original_width: int,
    original_height: int,
    derivative_width: int,
) -> tuple[dict | None, dict[str, bool], str | None, str]:
    """Decode and geometrically verify one JPEG.

    Returns ``(derivative, checks, failure_reason, detail)``.  The checks map
    always contains every JPEG-related check so consumers see the failed
    boundary rather than an omitted result.
    """
    checks = {
        "jpeg_format": False,
        "full_decode": False,
        "dimensions_positive": False,
        "derivative_width_exact": False,
        "aspect_ratio_consistent": False,
        "pixel_limit": False,
    }
    try:
        Image = _pillow_image()
    except PillowUnavailable as error:
        return None, checks, "pillow_unavailable", str(error)

    try:
        image = Image.open(io.BytesIO(data))
    except Exception as error:  # Pillow raises several format-specific types
        return None, checks, "wrong_format", f"{type(error).__name__}: {error}"

    with image:
        image_format = str(image.format or "")
        if image_format != "JPEG":
            return None, checks, "wrong_format", f"decoded format is {image_format!r}"
        checks["jpeg_format"] = True
        width, height = image.size
        checks["dimensions_positive"] = width > 0 and height > 0
        if not checks["dimensions_positive"]:
            return None, checks, "wrong_geometry", f"non-positive size {width}x{height}"
        if width * height > MAX_DECODED_PIXELS:
            return (
                None,
                checks,
                "pixel_limit_exceeded",
                f"{width}x{height} exceeds {MAX_DECODED_PIXELS} decoded pixels",
            )
        checks["pixel_limit"] = True
        checks["derivative_width_exact"] = width == derivative_width
        expected_height = original_height * derivative_width / original_width
        tolerance = max(
            ASPECT_ABSOLUTE_TOLERANCE,
            expected_height * ASPECT_RELATIVE_TOLERANCE,
        )
        checks["aspect_ratio_consistent"] = abs(height - expected_height) <= tolerance

        try:
            from PIL import ImageFile

            previous_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            try:
                image.load()
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated
        except Exception as error:
            return (
                None,
                checks,
                "truncated_image",
                f"full decode failed: {type(error).__name__}: {error}",
            )
        if image.size != (width, height):
            return (
                None,
                checks,
                "wrong_geometry",
                f"size changed during decode: {image.size}",
            )
        checks["full_decode"] = True

    if not checks["derivative_width_exact"] or not checks["aspect_ratio_consistent"]:
        return (
            None,
            checks,
            "wrong_geometry",
            f"decoded {width}x{height}, expected width {derivative_width} and "
            f"height {expected_height:.2f}+/-{tolerance:.2f} from original "
            f"{original_width}x{original_height}",
        )

    derivative = {
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "mime": "image/jpeg",
        "format": "JPEG",
        "width": width,
        "height": height,
        "max_dimension": max(width, height),
    }
    return derivative, checks, None, ""


def _source_block(record: Mapping[str, Any]) -> dict:
    return {field: record[field] for field in SOURCE_BLOCK_FIELDS}


def _report_asset(
    record: Mapping[str, Any],
    derivative: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> dict:
    return {
        "asset_id": record["asset_id"],
        "source_id": record["source_id"],
        "group_key": record["group_key"],
        "split": record["split"],
        "filename": f"{record['asset_id']}.jpg",
        "source": _source_block(record),
        "derivative": dict(derivative),
        "checks": dict(sorted(checks.items())),
    }


# ---------------------------------------------------------------------------
# HTTP fetching with hard bounds
# ---------------------------------------------------------------------------


@dataclass
class FetchOutcome:
    requested: bool = False
    ok: bool = False
    data: bytes = b""
    received: int = 0
    final_url: str = ""
    content_type: str | None = None
    content_length: int | None = None
    reason: str = ""
    detail: str = ""


class _RedirectAllowlist(urllib.request.HTTPRedirectHandler):
    """Reject redirects before following them to a disallowed origin."""

    def __init__(self, derivative_width: int, allowed_hosts: tuple[str, ...]):
        super().__init__()
        self._derivative_width = derivative_width
        self._allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            _validate_remote_url(
                newurl,
                self._derivative_width,
                self._allowed_hosts,
                "redirect target",
            )
        except ProtocolError as error:
            raise urllib.error.HTTPError(
                req.full_url, code, f"redirect rejected: {error}", headers, fp
            ) from error
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_opener_factory(plan: ProtocolPlan):
    redirect_handler = _RedirectAllowlist(plan.derivative_width, plan.allowed_hosts)
    opener = urllib.request.build_opener(redirect_handler)

    def open_url(url: str, timeout: float):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": plan.user_agent,
                "Accept": "image/jpeg",
                "Accept-Encoding": "identity",
            },
            method="GET",
        )
        return opener.open(request, timeout=timeout)

    return open_url


def _header_value(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    value = getter(name)
    if value is None:
        value = getter(name.lower())
    return None if value is None else str(value)


def _content_length(headers: Any) -> tuple[int | None, str]:
    raw = _header_value(headers, "Content-Length")
    if raw is None or not raw.strip():
        return None, ""
    try:
        value = int(raw.strip())
    except ValueError:
        return None, f"invalid Content-Length: {raw!r}"
    if value < 0:
        return None, f"negative Content-Length: {value}"
    return value, ""


def _fetch_one(
    opener: Callable[[str, float], Any],
    url: str,
    *,
    plan: ProtocolPlan,
    prior_total: int,
    max_total: int,
) -> FetchOutcome:
    """Perform exactly one bounded request.  Never retries and never falls back."""
    outcome = FetchOutcome()
    try:
        _validate_derivative_url(
            url, "request URL", plan.derivative_width, plan.allowed_hosts
        )
    except ProtocolError as error:
        outcome.reason = "request_url_rejected"
        outcome.detail = str(error)
        return outcome

    outcome.requested = True
    response = None
    try:
        response = opener(url, plan.timeout_seconds)
        outcome.final_url = str(getattr(response, "geturl", lambda: url)() or url)
        try:
            _validate_remote_url(
                outcome.final_url,
                plan.derivative_width,
                plan.allowed_hosts,
                "final response URL",
            )
        except ProtocolError as error:
            outcome.reason = "redirect_rejected"
            outcome.detail = str(error)
            return outcome

        headers = getattr(response, "headers", None)
        content_type = _header_value(headers, "Content-Type")
        outcome.content_type = content_type
        if content_type is not None:
            base_type = content_type.split(";", 1)[0].strip().lower()
            if base_type not in {"image/jpeg", "image/jpg"}:
                outcome.reason = "wrong_content_type"
                outcome.detail = f"Content-Type is {content_type!r}"
                return outcome

        content_length, length_error = _content_length(headers)
        if length_error:
            outcome.reason = "invalid_content_length"
            outcome.detail = length_error
            return outcome
        outcome.content_length = content_length
        if content_length is not None:
            if content_length == 0:
                outcome.reason = "empty_response"
                outcome.detail = "Content-Length is 0"
                return outcome
            if content_length > plan.max_asset_bytes:
                outcome.reason = "asset_byte_cap_exceeded"
                outcome.detail = (
                    f"Content-Length {content_length} > max_asset_bytes "
                    f"{plan.max_asset_bytes}"
                )
                return outcome
            if prior_total + content_length > max_total:
                outcome.reason = "total_byte_cap_exceeded"
                outcome.detail = (
                    f"Content-Length {content_length} plus {prior_total} received "
                    f"> max_total_bytes {max_total}"
                )
                return outcome

        chunks: list[bytes] = []
        received = 0
        while True:
            try:
                chunk = response.read(CHUNK_SIZE)
            except Exception as error:
                outcome.reason = "read_error"
                outcome.detail = f"{type(error).__name__}: {error}"
                outcome.received = received
                return outcome
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                outcome.reason = "invalid_response"
                outcome.detail = "response.read returned non-bytes"
                outcome.received = received
                return outcome
            received += len(chunk)
            if received > plan.max_asset_bytes:
                outcome.reason = "asset_byte_cap_exceeded"
                outcome.detail = (
                    f"received {received} > max_asset_bytes {plan.max_asset_bytes}"
                )
                outcome.received = received
                return outcome
            if prior_total + received > max_total:
                outcome.reason = "total_byte_cap_exceeded"
                outcome.detail = (
                    f"received {prior_total + received} > max_total_bytes {max_total}"
                )
                outcome.received = received
                return outcome
            chunks.append(chunk)

        outcome.received = received
        if received == 0:
            outcome.reason = "empty_response"
            outcome.detail = "response body is empty"
            return outcome
        if content_length is not None and received != content_length:
            outcome.reason = "content_length_mismatch"
            outcome.detail = (
                f"Content-Length {content_length}, received {received}"
            )
            return outcome
        outcome.data = b"".join(chunks)
        outcome.ok = True
        return outcome
    except urllib.error.HTTPError as error:
        if "redirect rejected" in str(error.reason):
            outcome.reason = "redirect_rejected"
        else:
            outcome.reason = "fetch_error"
        outcome.detail = f"HTTP {error.code}: {error.reason}"
        return outcome
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        outcome.reason = "fetch_error"
        outcome.detail = f"{type(error).__name__}: {error}"
        return outcome
    except Exception as error:  # defensive: a fetcher must not crash the runner
        outcome.reason = "fetch_error"
        outcome.detail = f"{type(error).__name__}: {error}"
        return outcome
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# evidence construction
# ---------------------------------------------------------------------------


def _implementation_info() -> dict:
    path = Path(__file__).resolve()
    return {"path": _display_path(path), "sha256": sha256_file(path)}


def _protocol_info(plan: ProtocolPlan) -> dict:
    return {
        "path": plan.path,
        "sha256": plan.sha256,
        "format": plan.protocol["format"],
        "status": plan.protocol["status"],
        "declared_at_utc": plan.protocol.get("declared_at_utc"),
        "derivative_width": plan.derivative_width,
    }


def _geometry_policy() -> dict:
    return {
        "rule": "decoded width equals acquisition.derivative_width and height matches "
        "the declared original aspect ratio",
        "aspect_ratio_relative_tolerance": ASPECT_RELATIVE_TOLERANCE,
        "aspect_ratio_absolute_tolerance_px": ASPECT_ABSOLUTE_TOLERANCE,
        "protocol_max_dimension_claim": False,
        "note": "Wikimedia Npx thumbnail URLs constrain width, not maximum dimension. "
        "Portrait derivatives therefore have a maximum dimension larger than the "
        "declared derivative width; the runner verifies the declared width and the "
        "original aspect ratio instead of making a false max-dimension=960 claim.",
    }


def _base_report(
    *,
    mode: str,
    protocol_info: dict,
    implementation: dict,
    started_at_utc: str,
    started_monotonic: float,
) -> dict:
    return {
        "format": REPORT_FORMAT,
        "kind": "bounded_real_image_derivative_report_not_training_authorization",
        "mode": mode,
        "status": "refused",
        "started_at_utc": started_at_utc,
        "completed_at_utc": None,
        "wall_time_seconds": None,
        "declared_nondeterministic_fields": [
            "started_at_utc",
            "completed_at_utc",
            "wall_time_seconds",
        ],
        "protocol": protocol_info,
        "implementation": implementation,
        "training_admitted": False,
        "admitted_count": 0,
        "asset_count": 0,
        "verified": False,
        "verification": {"performed": False},
        "assets": [],
        "asset_results": [],
        "splits": {},
        "source_counts": {},
        "checks": {},
        "requests": {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "max_requests": None,
        },
        "bytes": {
            "received": 0,
            "staged": 0,
            "admitted": 0,
            "max_asset_bytes": None,
            "max_total_bytes": None,
        },
        "storage": {
            "assets_dir": None,
            "report_dir": None,
            "asset_bytes": 0,
            "manifest_bytes": 0,
            "total_stored_bytes": 0,
        },
        "failures": [],
        "warnings": [],
        "non_claims": dict(NON_CLAIMS),
        "geometry_policy": _geometry_policy(),
        "network_used": False,
        "native_execution": False,
        "perceptual_dedup_claimed": False,
        "model_quality_evaluated": False,
        "_started_monotonic": started_monotonic,
    }


def _finalize_report(
    report: dict,
    report_dir: Path,
    *,
    sidecar_name: str | None = None,
    sidecar_factory: Callable[[dict], dict] | None = None,
) -> dict:
    report["completed_at_utc"] = _now_utc()
    report["wall_time_seconds"] = round(
        time.monotonic() - float(report.pop("_started_monotonic")), 6
    )
    report["failures"] = sorted(
        report.get("failures", []),
        key=lambda item: (
            str(item.get("asset_id") or ""),
            str(item.get("stage") or ""),
            str(item.get("reason") or ""),
            str(item.get("detail") or ""),
        ),
    )
    report["asset_results"] = sorted(
        report.get("asset_results", []), key=lambda item: str(item.get("asset_id") or "")
    )
    report["warnings"] = sorted(str(item) for item in report.get("warnings", []))
    report_dir = Path(report_dir)
    report["storage"]["report_dir"] = _display_path(report_dir)
    report["storage"]["report_path"] = _display_path(report_dir / REPORT_NAME)

    if sidecar_name is None and sidecar_factory is None:
        if report.get("mode") == "download":
            sidecar_name = ACQUISITION_NAME
            sidecar_factory = _build_acquisition_receipt
        elif report.get("mode") == "verify":
            sidecar_name = VERIFICATION_NAME
            sidecar_factory = _build_verification_evidence

    evidence: dict[str, dict] = {}
    if sidecar_name is not None and sidecar_factory is not None:
        sidecar_path = report_dir / sidecar_name
        try:
            payload = sidecar_factory(report)
            _write_json_atomic(sidecar_path, payload)
            evidence[sidecar_name] = {
                "path": _display_path(sidecar_path),
                "sha256": sha256_bytes(serialize_json(payload).encode("utf-8")),
                "written": True,
            }
        except OSError as error:
            evidence[sidecar_name] = {
                "path": _display_path(sidecar_path),
                "written": False,
                "error": f"{type(error).__name__}: {error}",
            }
        except Exception as error:  # evidence must never suppress the report
            evidence[sidecar_name] = {
                "path": _display_path(sidecar_path),
                "written": False,
                "error": f"receipt builder failed: {type(error).__name__}: {error}",
            }
    report["evidence"] = evidence

    try:
        _write_json_atomic(report_dir / REPORT_NAME, report)
        report["report_written"] = True
    except OSError as error:
        report["report_written"] = False
        report["report_write_error"] = f"{type(error).__name__}: {error}"
    return report


def _write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = serialize_json(payload)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _write_json_temp(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = serialize_json(payload)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise
    return temporary_path


def _result_checks(
    results: list[Mapping[str, Any]],
    failures: list[Mapping[str, Any]],
    *,
    include_manifest: bool = False,
) -> dict[str, bool]:
    checks = {
        "protocol_valid": True,
        "exact_asset_count": True,
        "exact_source_count": True,
        "unique_asset_ids": True,
        "unique_source_identities": True,
        "rights_fields_present": True,
        "licenses_allowed": True,
        "us_jurisdiction": True,
        "https_thumb_hosts": True,
        "caps_respected": not any(
            failure.get("reason") in {
                "asset_byte_cap_exceeded",
                "total_byte_cap_exceeded",
                "request_cap_exceeded",
            }
            for failure in failures
        ),
        "deterministic_splits": True,
        "expected_asset_counts": True,
        "nonempty_splits": True,
        "no_benchmark_or_excluded_sources": True,
        "all_assets_jpeg": bool(results)
        and all(result.get("checks", {}).get("jpeg_format", False) for result in results),
        "all_assets_fully_decoded": bool(results)
        and all(result.get("checks", {}).get("full_decode", False) for result in results),
        "all_geometry_valid": bool(results)
        and all(
            result.get("checks", {}).get("derivative_width_exact", False)
            and result.get("checks", {}).get("aspect_ratio_consistent", False)
            for result in results
        ),
        "all_hashes_and_lengths_valid": bool(results)
        and all(
            result.get("checks", {}).get("byte_length", False)
            and result.get("checks", {}).get("sha256", False)
            for result in results
        ),
        "exact_sha256_unique": bool(results)
        and all(
            result.get("checks", {}).get("exact_hash_unique", False)
            for result in results
        ),
        "group_splits_disjoint": bool(results)
        and all(
            result.get("checks", {}).get("split_matches_protocol", False)
            and result.get("checks", {}).get("source_group_match", False)
            for result in results
        ),
        "no_partial_admission": not any(
            failure.get("reason")
            in {"extra_asset_file", "asset_file_not_regular", "asset_symlink"}
            for failure in failures
        ),
        "all_checks_passed": False,
    }
    for name in ASSET_CHECK_NAMES:
        checks[f"all_{name}"] = bool(results) and all(
            bool(result.get("checks", {}).get(name)) for result in results
        )
    if include_manifest:
        failure_reasons = {str(failure.get("reason")) for failure in failures}
        checks.update(
            {
                "manifest_matches_protocol": not bool(
                    failure_reasons
                    & {
                        "manifest_invalid",
                        "manifest_asset_count",
                        "manifest_asset_id_drift",
                        "manifest_source_count",
                        "manifest_split_drift",
                        "manifest_source_drift",
                        "manifest_filename_drift",
                        "manifest_total_bytes_drift",
                        "manifest_missing_asset",
                        "manifest_duplicate_asset_id",
                        "protocol_drift",
                    }
                ),
                "no_extra_asset_files": "extra_asset_file" not in failure_reasons,
                "no_symlinks": not bool(
                    failure_reasons & {"asset_symlink", "asset_file_not_regular"}
                ),
                "storage_bytes_match": not bool(
                    failure_reasons
                    & {
                        "manifest_total_bytes_drift",
                        "byte_length_mismatch",
                        "sha256_mismatch",
                    }
                ),
            }
        )
    checks["all_checks_passed"] = (
        not failures
        and len(results) > 0
        and all(
            value
            for key, value in checks.items()
            if key != "all_checks_passed" and not key.startswith("all_")
        )
    )
    return checks


def _manifest_bytes(manifest: dict) -> int:
    return len(serialize_json(manifest).encode("utf-8"))


def _result_stats(results: list[Mapping[str, Any]]) -> tuple[dict, dict]:
    splits = {split: {"assets": 0, "bytes": 0} for split in SPLITS}
    sources: dict[str, dict] = {}
    for entry in results:
        derivative = entry.get("derivative") or {}
        byte_count = int(derivative.get("bytes") or 0)
        split = entry.get("split")
        if split in splits:
            splits[split]["assets"] += 1
            splits[split]["bytes"] += byte_count
        source_id = str(entry.get("source_id") or "")
        stats = sources.setdefault(source_id, {"assets": 0, "bytes": 0})
        stats["assets"] += 1
        stats["bytes"] += byte_count
    return splits, dict(sorted(sources.items()))


DERIVATIVE_EVIDENCE_KEYS = (
    "bytes",
    "sha256",
    "mime",
    "format",
    "width",
    "height",
    "max_dimension",
)


def _acquisition_manifest_binding(report: Mapping[str, Any]) -> dict | None:
    if report.get("status") != "admitted":
        return None
    verification = report.get("verification") or {}
    assets = report.get("assets") or []
    asset_hashes = {
        entry["asset_id"]: entry["derivative"]["sha256"]
        for entry in assets
        if entry.get("derivative", {}).get("sha256")
    }
    return {
        "sha256": verification.get("manifest_sha256"),
        "asset_count": report.get("asset_count"),
        "source_count": len(report.get("source_counts") or {}),
        "total_bytes": (report.get("storage") or {}).get("asset_bytes"),
        "asset_hashes": dict(sorted(asset_hashes.items())),
    }


def _acquisition_asset_evidence(report: Mapping[str, Any]) -> list[dict]:
    evidence = []
    for entry in report.get("assets") or []:
        derivative = entry.get("derivative") or {}
        evidence.append(
            {
                "asset_id": entry.get("asset_id"),
                "source_id": entry.get("source_id"),
                "split": entry.get("split"),
                "filename": entry.get("filename"),
                "derivative": {
                    key: derivative.get(key) for key in DERIVATIVE_EVIDENCE_KEYS
                },
            }
        )
    return evidence


def _build_acquisition_receipt(report: Mapping[str, Any]) -> dict:
    """Stable download-phase receipt; written on success and refusal alike."""
    storage = {
        key: value
        for key, value in (report.get("storage") or {}).items()
        if key not in {"assets_dir", "report_dir", "report_path"}
    }
    return {
        "format": ACQUISITION_FORMAT,
        "kind": "bounded_real_image_derivative_acquisition_receipt",
        "mode": "download",
        "status": report["status"],
        "started_at_utc": report.get("started_at_utc"),
        "completed_at_utc": report.get("completed_at_utc"),
        "wall_time_seconds": report.get("wall_time_seconds"),
        "declared_nondeterministic_fields": [
            "started_at_utc",
            "completed_at_utc",
            "wall_time_seconds",
        ],
        "protocol": report.get("protocol"),
        "implementation": report.get("implementation"),
        "network_used": report.get("network_used"),
        "requests": report.get("requests"),
        "bytes": report.get("bytes"),
        "admitted_count": report.get("admitted_count"),
        "asset_count": report.get("asset_count"),
        "splits": report.get("splits"),
        "source_counts": report.get("source_counts"),
        "storage": storage,
        "manifest": _acquisition_manifest_binding(report),
        "assets": _acquisition_asset_evidence(report),
        "failures": report.get("failures", []),
        "warnings": report.get("warnings", []),
        "non_claims": report.get("non_claims"),
        "training_admitted": False,
        "perceptual_dedup_claimed": False,
        "model_quality_evaluated": False,
    }


def _build_verification_evidence(report: Mapping[str, Any]) -> dict:
    """Stable offline-verification receipt; includes separate acquisition evidence."""
    verification = report.get("verification") or {}
    storage = {
        key: value
        for key, value in (report.get("storage") or {}).items()
        if key != "report_path"
    }
    manifest = {
        "path": verification.get("manifest_path"),
        "sha256": verification.get("manifest_sha256"),
        "asset_count": report.get("asset_count"),
        "total_bytes": (report.get("storage") or {}).get("asset_bytes"),
    }
    return {
        "format": VERIFICATION_FORMAT,
        "kind": "bounded_real_image_derivative_verification_receipt",
        "mode": "verify",
        "status": report["status"],
        "started_at_utc": report.get("started_at_utc"),
        "completed_at_utc": report.get("completed_at_utc"),
        "wall_time_seconds": report.get("wall_time_seconds"),
        "declared_nondeterministic_fields": [
            "started_at_utc",
            "completed_at_utc",
            "wall_time_seconds",
        ],
        "protocol": report.get("protocol"),
        "implementation": report.get("implementation"),
        "manifest": manifest,
        "verification": verification,
        "implementation_matches_manifest": verification.get(
            "implementation_matches_manifest"
        ),
        "acquisition": report.get("acquisition"),
        "admitted_count": report.get("admitted_count"),
        "asset_count": report.get("asset_count"),
        "verified": report.get("verified"),
        "geometry_policy": report.get("geometry_policy"),
        "network_used": False,
        "requests": report.get("requests"),
        "bytes": report.get("bytes"),
        "splits": report.get("splits"),
        "source_counts": report.get("source_counts"),
        "checks": report.get("checks"),
        "storage": storage,
        "assets": report.get("assets", []),
        "asset_results": report.get("asset_results", []),
        "failures": report.get("failures", []),
        "warnings": report.get("warnings", []),
        "non_claims": report.get("non_claims"),
        "training_admitted": False,
        "perceptual_dedup_claimed": False,
        "model_quality_evaluated": False,
    }


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _acquisition_structure_problem(
    receipt: object,
    plan: ProtocolPlan,
) -> tuple[str | None, str]:
    """Return (reason, detail) for structural/protocol problems."""
    if not isinstance(receipt, dict):
        return "acquisition_invalid", "receipt root is not an object"
    if receipt.get("format") != ACQUISITION_FORMAT:
        return "acquisition_invalid", f"format is {receipt.get('format')!r}"
    if receipt.get("mode") != "download":
        return "acquisition_invalid", f"mode is {receipt.get('mode')!r}"
    status = receipt.get("status")
    if status not in {"admitted", "refused"}:
        return "acquisition_invalid", f"status is {status!r}"
    protocol = receipt.get("protocol")
    if not isinstance(protocol, dict):
        return "acquisition_invalid", "protocol block is missing"
    if protocol.get("sha256") != plan.sha256:
        return (
            "acquisition_incompatible",
            "receipt protocol sha256 differs from the frozen protocol",
        )
    if protocol.get("format") != plan.protocol["format"]:
        return "acquisition_incompatible", "receipt protocol format differs"
    if protocol.get("derivative_width") != plan.derivative_width:
        return "acquisition_incompatible", "receipt derivative_width differs"
    implementation = receipt.get("implementation")
    if not isinstance(implementation, dict):
        return "acquisition_invalid", "implementation block is missing"
    if not SHA256_PATTERN.fullmatch(str(implementation.get("sha256"))):
        return "acquisition_invalid", "implementation sha256 is invalid"
    for field in ("started_at_utc", "completed_at_utc"):
        value = receipt.get(field)
        if not isinstance(value, str) or not value:
            return "acquisition_invalid", f"{field} is missing"
        try:
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return "acquisition_invalid", f"{field} is not an RFC3339 UTC timestamp"
    if not _is_number(receipt.get("wall_time_seconds")):
        return "acquisition_invalid", "wall_time_seconds is invalid"
    requests = receipt.get("requests")
    if not isinstance(requests, dict):
        return "acquisition_invalid", "requests block is missing"
    for field in ("attempted", "succeeded", "failed"):
        if not _is_int(requests.get(field)) or requests.get(field) < 0:
            return "acquisition_invalid", f"requests.{field} is invalid"
    max_requests = requests.get("max_requests")
    if max_requests is not None and (
        not _is_int(max_requests) or max_requests < 0
    ):
        return "acquisition_invalid", "requests.max_requests is invalid"
    if (
        max_requests is not None
        and requests["attempted"] > max_requests
    ):
        return (
            "acquisition_incompatible",
            "receipt request count exceeds its declared maximum",
        )
    if not isinstance(receipt.get("network_used"), bool):
        return "acquisition_invalid", "network_used is not boolean"
    for field in ("bytes", "storage", "splits", "source_counts", "non_claims"):
        if not isinstance(receipt.get(field), dict):
            return "acquisition_invalid", f"{field} block is missing"
    for field in ("admitted_count", "asset_count"):
        if not _is_int(receipt.get(field)) or receipt.get(field) < 0:
            return "acquisition_invalid", f"{field} is invalid"
    if not isinstance(receipt.get("failures"), list):
        return "acquisition_invalid", "failures is not a list"
    if not isinstance(receipt.get("warnings"), list):
        return "acquisition_invalid", "warnings is not a list"
    if receipt.get("training_admitted") is not False:
        return "acquisition_invalid", "training_admitted must be false"
    if receipt.get("perceptual_dedup_claimed") is not False:
        return "acquisition_invalid", "perceptual_dedup_claimed must be false"
    if receipt.get("model_quality_evaluated") is not False:
        return "acquisition_invalid", "model_quality_evaluated must be false"
    assets = receipt.get("assets")
    if not isinstance(assets, list):
        return "acquisition_invalid", "assets evidence is not a list"
    if status == "admitted":
        if not isinstance(receipt.get("manifest"), dict):
            return "acquisition_incompatible", "admitted receipt lacks a manifest binding"
    else:
        if receipt.get("manifest") is not None:
            return "acquisition_incompatible", "refused receipt carries a manifest binding"
        if receipt.get("admitted_count") != 0 or receipt.get("asset_count") != 0:
            return "acquisition_incompatible", "refused receipt claims admitted assets"
        if assets:
            return "acquisition_incompatible", "refused receipt carries admitted asset evidence"
        receipt_bytes = receipt.get("bytes")
        storage = receipt.get("storage")
        if not isinstance(receipt_bytes, dict) or receipt_bytes.get("admitted") != 0:
            return "acquisition_incompatible", "refused receipt claims admitted bytes"
        if not isinstance(storage, dict) or storage.get("asset_bytes") != 0:
            return "acquisition_incompatible", "refused receipt claims stored asset bytes"
    return None, ""


def _acquisition_binding_problem(
    receipt: Mapping[str, Any],
    plan: ProtocolPlan,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    manifest_bytes: int,
    manifest_assets: list,
) -> str:
    block = receipt.get("manifest")
    if not isinstance(block, dict):
        return "manifest binding is missing"
    if block.get("sha256") != manifest_sha256 or not SHA256_PATTERN.fullmatch(
        str(manifest_sha256)
    ):
        return "manifest sha256 differs"
    if not _is_int(block.get("asset_count")):
        return "manifest binding asset_count is invalid"
    if not _is_int(block.get("source_count")):
        return "manifest binding source_count is invalid"
    if not _is_int(block.get("total_bytes")):
        return "manifest binding total_bytes is invalid"
    manifest_count = manifest.get("asset_count")
    manifest_source_count = manifest.get("source_count")
    manifest_total = manifest.get("total_bytes")
    if not _is_int(manifest_count) or manifest_count != plan.exact_assets:
        return "manifest asset_count is not the frozen asset count"
    if manifest_count != len(manifest_assets):
        return "manifest asset_count does not match the asset list"
    if block["asset_count"] != manifest_count:
        return "receipt manifest asset_count differs"
    if block["source_count"] != manifest_source_count:
        return "receipt manifest source_count differs"
    if block["total_bytes"] != manifest_total:
        return "receipt manifest total_bytes differs"
    if receipt.get("admitted_count") != manifest_count:
        return "receipt admitted_count differs from the manifest"
    if receipt.get("asset_count") != manifest_count:
        return "receipt asset_count differs from the manifest"
    if receipt.get("splits") != manifest.get("splits"):
        return "receipt splits differ from the manifest"
    if receipt.get("source_counts") != manifest.get("source_counts"):
        return "receipt source_counts differ from the manifest"
    manifest_implementation = manifest.get("implementation")
    receipt_implementation = receipt.get("implementation")
    if (
        not isinstance(manifest_implementation, dict)
        or not isinstance(receipt_implementation, dict)
        or receipt_implementation.get("sha256")
        != manifest_implementation.get("sha256")
    ):
        return "receipt implementation sha256 differs from the manifest"
    if receipt.get("non_claims") != manifest.get("non_claims"):
        return "receipt non_claims differ from the manifest"
    receipt_bytes = receipt.get("bytes")
    if not isinstance(receipt_bytes, dict):
        return "receipt bytes block is invalid"
    if receipt_bytes.get("admitted") != manifest_total:
        return "receipt admitted bytes differ from the manifest"
    if not _is_int(receipt_bytes.get("received")) or receipt_bytes["received"] < manifest_total:
        return "receipt received bytes are smaller than the admitted bytes"
    if not _is_int(receipt_bytes.get("staged")) or receipt_bytes["staged"] < manifest_total:
        return "receipt staged bytes are smaller than the admitted bytes"
    storage = receipt.get("storage")
    if not isinstance(storage, dict):
        return "receipt storage block is invalid"
    if storage.get("asset_bytes") != manifest_total:
        return "receipt storage asset_bytes differs from the manifest"
    if storage.get("manifest_bytes") != manifest_bytes:
        return "receipt storage manifest_bytes differs from the manifest file"
    if storage.get("total_stored_bytes") != manifest_total + manifest_bytes:
        return "receipt storage total_stored_bytes differs"
    if storage.get("asset_count") != manifest_count:
        return "receipt storage asset_count differs"
    if storage.get("source_count") != manifest_source_count:
        return "receipt storage source_count differs"
    requests = receipt.get("requests")
    if not isinstance(requests, dict):
        return "receipt requests block is invalid"
    if requests.get("attempted", 0) < manifest_count:
        return "receipt attempted fewer requests than admitted assets"
    if requests.get("succeeded") != manifest_count:
        return "receipt succeeded request count differs from the asset count"
    if requests.get("failed") != 0:
        return "admitted receipt reports failed requests"
    if receipt.get("network_used") is not True:
        return "admitted receipt does not declare network use"
    if receipt.get("failures"):
        return "admitted receipt carries failures"

    expected_hashes: dict[str, str] = {}
    expected_bytes: dict[str, int] = {}
    for entry in manifest_assets:
        if not isinstance(entry, dict):
            return "manifest asset entry is not an object"
        asset_id = entry.get("asset_id")
        derivative = entry.get("derivative")
        if not isinstance(asset_id, str) or not isinstance(derivative, dict):
            return "manifest asset entry is malformed"
        expected_hashes[asset_id] = derivative.get("sha256")
        expected_bytes[asset_id] = derivative.get("bytes")
    if block.get("asset_hashes") != expected_hashes:
        return "receipt manifest asset hashes differ from the manifest"

    evidence_hashes: dict[str, str] = {}
    evidence_bytes: dict[str, int] = {}
    assets = receipt.get("assets")
    if not isinstance(assets, list) or len(assets) != manifest_count:
        return "receipt asset evidence does not match the manifest asset count"
    for entry in assets:
        if not isinstance(entry, dict):
            return "receipt asset evidence entry is not an object"
        asset_id = entry.get("asset_id")
        derivative = entry.get("derivative")
        if not isinstance(asset_id, str) or not isinstance(derivative, dict):
            return "receipt asset evidence entry is malformed"
        if asset_id in evidence_hashes:
            return "receipt asset evidence repeats an asset_id"
        evidence_hashes[asset_id] = derivative.get("sha256")
        evidence_bytes[asset_id] = derivative.get("bytes")
    if evidence_hashes != expected_hashes:
        return "receipt asset hashes differ from the manifest"
    if evidence_bytes != expected_bytes:
        return "receipt asset bytes differ from the manifest"
    return ""


def _acquisition_evidence(
    receipt: Mapping[str, Any],
    receipt_path: Path,
    receipt_sha256: str,
) -> dict:
    return {
        "receipt_path": _display_path(receipt_path),
        "receipt_sha256": receipt_sha256,
        "mode": receipt.get("mode"),
        "status": receipt.get("status"),
        "started_at_utc": receipt.get("started_at_utc"),
        "completed_at_utc": receipt.get("completed_at_utc"),
        "wall_time_seconds": receipt.get("wall_time_seconds"),
        "declared_nondeterministic_fields": receipt.get(
            "declared_nondeterministic_fields", []
        ),
        "requests": receipt.get("requests"),
        "network_used": receipt.get("network_used"),
        "failures": receipt.get("failures", []),
        "warnings": receipt.get("warnings", []),
        "admitted_count": receipt.get("admitted_count"),
        "asset_count": receipt.get("asset_count"),
        "bytes": receipt.get("bytes"),
        "storage": receipt.get("storage"),
        "splits": receipt.get("splits"),
        "source_counts": receipt.get("source_counts"),
        "manifest": receipt.get("manifest"),
        "protocol": receipt.get("protocol"),
        "implementation": receipt.get("implementation"),
        "assets": receipt.get("assets", []),
        "non_claims": receipt.get("non_claims"),
        "training_admitted": receipt.get("training_admitted"),
        "perceptual_dedup_claimed": receipt.get("perceptual_dedup_claimed"),
        "model_quality_evaluated": receipt.get("model_quality_evaluated"),
    }


def _build_manifest(
    plan: ProtocolPlan,
    admitted: list[dict],
    implementation: dict,
) -> dict:
    assets = []
    split_stats = {split: {"assets": 0, "bytes": 0} for split in SPLITS}
    source_stats: dict[str, dict] = {}
    for entry in sorted(admitted, key=lambda item: item["record"]["asset_id"]):
        record = entry["record"]
        derivative = entry["derivative"]
        assets.append(
            {
                "asset_id": record["asset_id"],
                "source_id": record["source_id"],
                "group_key": record["group_key"],
                "split": record["split"],
                "filename": f"{record['asset_id']}.jpg",
                "source": _source_block(record),
                "derivative": dict(derivative),
                "checks": dict(sorted(entry["checks"].items())),
            }
        )
        split_stats[record["split"]]["assets"] += 1
        split_stats[record["split"]]["bytes"] += int(derivative["bytes"])
        stats = source_stats.setdefault(
            record["source_id"], {"assets": 0, "bytes": 0}
        )
        stats["assets"] += 1
        stats["bytes"] += int(derivative["bytes"])
    total_bytes = sum(int(entry["derivative"]["bytes"]) for entry in admitted)
    return {
        "format": MANIFEST_FORMAT,
        "status": MANIFEST_STATUS,
        "protocol": _protocol_info(plan),
        "implementation": implementation,
        "asset_count": len(assets),
        "source_count": len(source_stats),
        "total_bytes": total_bytes,
        "splits": split_stats,
        "source_counts": dict(sorted(source_stats.items())),
        "assets": assets,
        "training_admitted": False,
        "perceptual_dedup_claimed": False,
        "non_claims": dict(NON_CLAIMS),
        "geometry_policy": _geometry_policy(),
    }


# ---------------------------------------------------------------------------
# download / admission
# ---------------------------------------------------------------------------


def _prepare_destination(assets_dir: Path, report_dir: Path) -> tuple[bool, str]:
    try:
        resolved_assets = assets_dir.resolve()
        resolved_report = report_dir.resolve()
    except OSError as error:
        return False, f"cannot resolve destination paths: {error}"
    if resolved_report == resolved_assets or resolved_assets in resolved_report.parents:
        return False, "report_dir must not be the assets_dir or a child of it"
    if assets_dir.is_symlink():
        return False, "assets_dir must not be a symlink"
    if assets_dir.exists():
        if not assets_dir.is_dir():
            return False, "assets_dir exists and is not a directory"
        try:
            entries = list(assets_dir.iterdir())
        except OSError as error:
            return False, f"cannot inspect assets_dir: {error}"
        if entries:
            return False, "assets_dir is not empty; refusing to overwrite an admitted set"
    return True, ""


def _cleanup_stage(stage: Path | None) -> None:
    if stage is not None:
        shutil.rmtree(stage, ignore_errors=True)


def _write_staged(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_move_assets(stage: Path, assets_dir: Path) -> None:
    if assets_dir.is_symlink():
        raise OSError("assets_dir must not be a symlink")
    if assets_dir.exists():
        if not assets_dir.is_dir():
            raise OSError("assets_dir exists and is not a directory")
        if any(assets_dir.iterdir()):
            raise OSError("assets_dir is not empty")
        try:
            os.replace(stage, assets_dir)
            return
        except OSError:
            assets_dir.rmdir()
    os.replace(stage, assets_dir)


def _remove_asset_dir_if_expected(assets_dir: Path, expected: set[str]) -> None:
    try:
        if assets_dir.is_symlink() or not assets_dir.is_dir():
            return
        names = {entry.name for entry in assets_dir.iterdir()}
        if names == expected:
            shutil.rmtree(assets_dir)
    except OSError:
        return


def download(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    assets_dir: str | Path = DEFAULT_ASSETS_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    *,
    opener: Callable[[str, float], Any] | None = None,
) -> dict:
    """Acquire and admit the declared seven derivatives, or refuse everything.

    ``opener`` exists for deterministic tests: it must return an object with
    ``read``, ``geturl``, ``headers`` and ``close``.  Production uses a
    redirect-allowlisted urllib opener.
    """
    started_at = _now_utc()
    started_monotonic = time.monotonic()
    assets_dir = Path(assets_dir)
    report_dir = Path(report_dir)
    implementation = _implementation_info()
    try:
        plan = load_plan(config_path)
    except ProtocolError as error:
        protocol_info = {
            "path": _display_path(Path(config_path)),
            "sha256": (
                sha256_file(Path(config_path))
                if Path(config_path).is_file()
                else None
            ),
            "format": None,
            "status": None,
            "declared_at_utc": None,
            "derivative_width": None,
        }
        report = _base_report(
            mode="download",
            protocol_info=protocol_info,
            implementation=implementation,
            started_at_utc=started_at,
            started_monotonic=started_monotonic,
        )
        report["failures"] = [
            {
                "stage": "protocol",
                "asset_id": None,
                "source_id": None,
                "reason": "protocol_invalid",
                "detail": str(error),
            }
        ]
        report["checks"] = _result_checks([], report["failures"])
        report["checks"]["protocol_valid"] = False
        report["storage"].update(
            {"assets_dir": _display_path(assets_dir), "report_dir": _display_path(report_dir)}
        )
        return _finalize_report(report, report_dir)

    protocol_info = _protocol_info(plan)
    report = _base_report(
        mode="download",
        protocol_info=protocol_info,
        implementation=implementation,
        started_at_utc=started_at,
        started_monotonic=started_monotonic,
    )
    report["requests"]["max_requests"] = plan.max_requests
    report["bytes"].update(
        {
            "max_asset_bytes": plan.max_asset_bytes,
            "max_total_bytes": plan.max_total_bytes,
        }
    )
    report["storage"].update(
        {
            "assets_dir": _display_path(assets_dir),
            "report_dir": _display_path(report_dir),
        }
    )
    report["network_used"] = False
    report["splits"], report["source_counts"] = _result_stats([])

    failures: list[dict] = []
    results: list[dict] = []
    admitted: list[dict] = []
    seen_sha: dict[str, str] = {}
    total_received = 0
    staged_bytes = 0
    requests_attempted = 0
    requests_succeeded = 0
    requests_failed = 0

    def fail(stage: str, reason: str, detail: str, record: dict | None = None) -> None:
        failures.append(
            {
                "stage": stage,
                "asset_id": record.get("asset_id") if record else None,
                "source_id": record.get("source_id") if record else None,
                "reason": reason,
                "detail": detail,
            }
        )

    destination_ok, destination_detail = _prepare_destination(assets_dir, report_dir)
    if not destination_ok:
        fail("destination", "assets_dir_not_usable", destination_detail)
        report["failures"] = failures
        report["checks"] = _result_checks(results, failures)
        return _finalize_report(report, report_dir)

    try:
        assets_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{assets_dir.name}.stage-", dir=str(assets_dir.parent)
            )
        )
    except OSError as error:
        fail("destination", "staging_failed", f"{type(error).__name__}: {error}")
        report["failures"] = failures
        report["checks"] = _result_checks(results, failures)
        return _finalize_report(report, report_dir)

    if opener is None:
        opener = _default_opener_factory(plan)

    stopped_for_total_cap = False
    try:
        for record in plan.assets:
            if stopped_for_total_cap:
                fail(
                    "fetch",
                    "not_attempted_after_total_cap",
                    "total byte budget was exhausted before this asset",
                    record,
                )
                continue
            if requests_attempted >= plan.max_requests:
                fail(
                    "fetch",
                    "request_cap_exceeded",
                    f"max_requests {plan.max_requests} reached",
                    record,
                )
                stopped_for_total_cap = True
                continue
            if total_received >= plan.max_total_bytes:
                fail(
                    "fetch",
                    "total_byte_cap_exceeded",
                    f"{total_received} bytes received >= max_total_bytes "
                    f"{plan.max_total_bytes}",
                    record,
                )
                stopped_for_total_cap = True
                continue

            outcome = _fetch_one(
                opener,
                record["thumb_url"],
                plan=plan,
                prior_total=total_received,
                max_total=plan.max_total_bytes,
            )
            if outcome.requested:
                requests_attempted += 1
            total_received += outcome.received
            if outcome.ok:
                requests_succeeded += 1
            else:
                requests_failed += 1
                fail("fetch", outcome.reason, outcome.detail, record)
                if outcome.reason in {
                    "total_byte_cap_exceeded",
                    "request_cap_exceeded",
                }:
                    stopped_for_total_cap = True
                continue

            staged_path = stage / f"{record['asset_id']}.jpg"
            try:
                _write_staged(staged_path, outcome.data)
                staged_bytes += len(outcome.data)
            except OSError as error:
                fail(
                    "stage",
                    "staging_write_failed",
                    f"{type(error).__name__}: {error}",
                    record,
                )
                continue

            try:
                actual_bytes = staged_path.stat().st_size
                actual_sha = sha256_file(staged_path)
                staged_data = staged_path.read_bytes()
            except OSError as error:
                fail(
                    "qa",
                    "asset_read_failed",
                    f"{type(error).__name__}: {error}",
                    record,
                )
                continue
            checks = {name: False for name in ASSET_CHECK_NAMES}
            checks["byte_length"] = actual_bytes == len(outcome.data)
            checks["sha256"] = bool(SHA256_PATTERN.fullmatch(actual_sha))
            checks["split_matches_protocol"] = (
                record["split"] == plan.group_splits[record["group_key"]]
            )
            checks["source_group_match"] = (
                record["group_key"] in plan.group_splits
                and plan.group_splits[record["group_key"]] == record["split"]
            )
            checks["rights_fields"] = _rights_fields_present(
                record, plan.protocol["license_policy"]
            )
            checks["license_allowed"] = _license_allowed(
                record.get("license"), plan.protocol["license_policy"]
            )
            checks["source_not_excluded"] = not _benchmark_excluded(
                record["source_id"],
                record["group_key"],
                record["asset_id"],
                plan.protocol["leakage"],
            )
            checks["exact_hash_unique"] = actual_sha not in seen_sha

            derivative, jpeg_checks, qa_reason, qa_detail = _qa_jpeg(
                staged_data,
                record["original_width"],
                record["original_height"],
                plan.derivative_width,
            )
            checks.update(jpeg_checks)
            if derivative is not None:
                derivative["bytes"] = actual_bytes
                derivative["sha256"] = actual_sha

            result = {
                "asset_id": record["asset_id"],
                "source_id": record["source_id"],
                "split": record["split"],
                "filename": f"{record['asset_id']}.jpg",
                "source": _source_block(record),
                "derivative": derivative or {},
                "checks": dict(sorted(checks.items())),
            }
            results.append(result)

            if derivative is None:
                fail("qa", qa_reason or "jpeg_qa_failed", qa_detail, record)
                continue
            if not all(checks.values()):
                failed = sorted(name for name, value in checks.items() if not value)
                reason = (
                    "duplicate_exact_sha256"
                    if not checks["exact_hash_unique"]
                    else "check_failed"
                )
                fail("qa", reason, f"failed checks: {failed}", record)
                continue
            seen_sha[actual_sha] = record["asset_id"]
            admitted.append(
                {
                    "record": record,
                    "derivative": derivative,
                    "checks": checks,
                }
            )

        report["requests"].update(
            {
                "attempted": requests_attempted,
                "succeeded": requests_succeeded,
                "failed": requests_failed,
            }
        )
        report["network_used"] = requests_attempted > 0
        report["bytes"].update({"received": total_received, "staged": staged_bytes})
        report["splits"], report["source_counts"] = _result_stats(results)

        if failures:
            fail(
                "admission",
                "all_or_nothing_refused",
                f"{len(failures)} failure(s); no asset was admitted",
            )
            report["failures"] = failures
            report["asset_results"] = results
            report["checks"] = _result_checks(results, failures)
            return _finalize_report(report, report_dir)

        expected_names = {f"{record['asset_id']}.jpg" for record in plan.assets}
        staged_names = {entry.name for entry in stage.iterdir()}
        if staged_names != expected_names:
            fail(
                "admission",
                "staged_set_mismatch",
                f"staged names differ from the declared set: {sorted(staged_names)}",
            )
            report["failures"] = failures
            report["asset_results"] = results
            report["checks"] = _result_checks(results, failures)
            return _finalize_report(report, report_dir)

        manifest = _build_manifest(plan, admitted, implementation)
        manifest_bytes = _manifest_bytes(manifest)
        manifest_path = report_dir / MANIFEST_NAME
        try:
            manifest_temp = _write_json_temp(manifest_path, manifest)
        except OSError as error:
            fail(
                "admission",
                "manifest_write_failed",
                f"{type(error).__name__}: {error}",
            )
            report["failures"] = failures
            report["asset_results"] = results
            report["checks"] = _result_checks(results, failures)
            return _finalize_report(report, report_dir)

        try:
            _atomic_move_assets(stage, assets_dir)
            stage = None
            os.replace(manifest_temp, manifest_path)
        except OSError as error:
            try:
                manifest_temp.unlink()
            except OSError:
                pass
            _cleanup_stage(stage)
            stage = None
            _remove_asset_dir_if_expected(assets_dir, expected_names)
            fail(
                "admission",
                "atomic_admission_failed",
                f"{type(error).__name__}: {error}",
            )
            report["failures"] = failures
            report["asset_results"] = results
            report["checks"] = _result_checks(results, failures)
            return _finalize_report(report, report_dir)

        report["status"] = "admitted"
        report["admitted_count"] = len(admitted)
        report["asset_count"] = len(admitted)
        report["assets"] = [
            _report_asset(entry["record"], entry["derivative"], entry["checks"])
            for entry in sorted(admitted, key=lambda item: item["record"]["asset_id"])
        ]
        report["asset_results"] = results
        report["splits"] = manifest["splits"]
        report["source_counts"] = manifest["source_counts"]
        report["checks"] = _result_checks(results, failures)
        report["bytes"]["admitted"] = manifest["total_bytes"]
        report["storage"].update(
            {
                "asset_bytes": manifest["total_bytes"],
                "manifest_bytes": manifest_bytes,
                "total_stored_bytes": manifest["total_bytes"] + manifest_bytes,
                "asset_count": manifest["asset_count"],
                "source_count": manifest["source_count"],
            }
        )
        report["verification"] = {
            "performed": False,
            "manifest_path": _display_path(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
        return _finalize_report(report, report_dir)
    finally:
        _cleanup_stage(stage)


# ---------------------------------------------------------------------------
# offline verification
# ---------------------------------------------------------------------------


def _manifest_failure_report(
    *,
    mode: str,
    protocol_info: dict,
    implementation: dict,
    started_at: str,
    started_monotonic: float,
    assets_dir: Path,
    report_dir: Path,
    failures: list[dict],
    results: list[dict] | None = None,
    storage: dict | None = None,
    include_manifest: bool = False,
    protocol_valid: bool = True,
    finalize: bool = True,
    acquisition: dict | None = None,
    acquisition_receipt: str = "not_evaluated",
    acquisition_path: Path | None = None,
    acquisition_sha256: str | None = None,
) -> dict:
    report = _base_report(
        mode=mode,
        protocol_info=protocol_info,
        implementation=implementation,
        started_at_utc=started_at,
        started_monotonic=started_monotonic,
    )
    report["status"] = "verification_failed"
    report["failures"] = failures
    report["asset_results"] = results or []
    report["checks"] = _result_checks(
        results or [], failures, include_manifest=include_manifest
    )
    report["checks"]["protocol_valid"] = protocol_valid
    report["storage"].update(
        {
            "assets_dir": _display_path(assets_dir),
            "report_dir": _display_path(report_dir),
        }
    )
    if storage:
        report["storage"].update(storage)
    report["acquisition"] = acquisition
    report["verification"] = {
        "performed": True,
        "recomputed": True,
        "acquisition_receipt": acquisition_receipt,
        "acquisition_receipt_path": (
            _display_path(acquisition_path) if acquisition_path is not None else None
        ),
        "acquisition_receipt_sha256": acquisition_sha256,
    }
    if finalize:
        return _finalize_report(report, report_dir)
    return report


def verify(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    assets_dir: str | Path = DEFAULT_ASSETS_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> dict:
    """Offline verification of the local asset set against ``manifest.json``."""
    started_at = _now_utc()
    started_monotonic = time.monotonic()
    assets_dir = Path(assets_dir)
    report_dir = Path(report_dir)
    implementation = _implementation_info()
    try:
        plan = load_plan(config_path)
    except ProtocolError as error:
        protocol_info = {
            "path": _display_path(Path(config_path)),
            "sha256": (
                sha256_file(Path(config_path))
                if Path(config_path).is_file()
                else None
            ),
            "format": None,
            "status": None,
            "declared_at_utc": None,
            "derivative_width": None,
        }
        failures = [
            {
                "stage": "protocol",
                "asset_id": None,
                "source_id": None,
                "reason": "protocol_invalid",
                "detail": str(error),
            }
        ]
        return _manifest_failure_report(
            mode="verify",
            protocol_info=protocol_info,
            implementation=implementation,
            started_at=started_at,
            started_monotonic=started_monotonic,
            assets_dir=assets_dir,
            report_dir=report_dir,
            failures=failures,
            protocol_valid=False,
        )

    protocol_info = _protocol_info(plan)
    failures: list[dict] = []
    warnings: list[str] = []
    results: list[dict] = []
    manifest_path = report_dir / MANIFEST_NAME

    def fail(stage: str, reason: str, detail: str, record: dict | None = None) -> None:
        failures.append(
            {
                "stage": stage,
                "asset_id": record.get("asset_id") if record else None,
                "source_id": record.get("source_id") if record else None,
                "reason": reason,
                "detail": detail,
            }
        )

    if not manifest_path.is_file():
        fail("manifest", "manifest_missing", f"missing {manifest_path}")
        return _manifest_failure_report(
            mode="verify",
            protocol_info=protocol_info,
            implementation=implementation,
            started_at=started_at,
            started_monotonic=started_monotonic,
            assets_dir=assets_dir,
            report_dir=report_dir,
            failures=failures,
        )

    try:
        manifest_raw = manifest_path.read_bytes()
    except OSError as error:
        fail(
            "manifest",
            "manifest_read_failed",
            f"{type(error).__name__}: {error}",
        )
        return _manifest_failure_report(
            mode="verify",
            protocol_info=protocol_info,
            implementation=implementation,
            started_at=started_at,
            started_monotonic=started_monotonic,
            assets_dir=assets_dir,
            report_dir=report_dir,
            failures=failures,
        )
    manifest_sha = sha256_bytes(manifest_raw)
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail("manifest", "manifest_invalid", f"invalid JSON: {error}")
        return _manifest_failure_report(
            mode="verify",
            protocol_info=protocol_info,
            implementation=implementation,
            started_at=started_at,
            started_monotonic=started_monotonic,
            assets_dir=assets_dir,
            report_dir=report_dir,
            failures=failures,
            storage={"manifest_bytes": len(manifest_raw)},
        )

    if not isinstance(manifest, dict):
        fail("manifest", "manifest_invalid", "manifest root is not an object")
        return _manifest_failure_report(
            mode="verify",
            protocol_info=protocol_info,
            implementation=implementation,
            started_at=started_at,
            started_monotonic=started_monotonic,
            assets_dir=assets_dir,
            report_dir=report_dir,
            failures=failures,
            storage={"manifest_bytes": len(manifest_raw)},
        )

    if manifest.get("format") != MANIFEST_FORMAT:
        fail("manifest", "manifest_invalid", f"format is {manifest.get('format')!r}")
    if manifest.get("status") != MANIFEST_STATUS:
        fail("manifest", "manifest_invalid", f"status is {manifest.get('status')!r}")
    if manifest.get("training_admitted") is not False:
        fail("manifest", "manifest_nonclaim_drift", "training_admitted must be false")
    if manifest.get("perceptual_dedup_claimed") is not False:
        fail(
            "manifest",
            "manifest_nonclaim_drift",
            "perceptual_dedup_claimed must be false",
        )
    if manifest.get("non_claims") != NON_CLAIMS:
        fail("manifest", "manifest_nonclaim_drift", "non_claims differ from the runner")
    if manifest.get("geometry_policy") != _geometry_policy():
        fail("manifest", "manifest_geometry_policy_drift", "geometry policy differs")

    manifest_protocol = manifest.get("protocol")
    if not isinstance(manifest_protocol, dict):
        fail("manifest", "manifest_invalid", "protocol block is missing")
    else:
        if manifest_protocol.get("sha256") != plan.sha256:
            fail(
                "manifest",
                "protocol_drift",
                f"manifest protocol sha256 {manifest_protocol.get('sha256')!r} "
                f"!= current {plan.sha256}",
            )
        if manifest_protocol.get("format") != plan.protocol["format"]:
            fail("manifest", "protocol_drift", "manifest protocol format differs")
        if manifest_protocol.get("declared_at_utc") != plan.protocol.get("declared_at_utc"):
            fail("manifest", "protocol_drift", "manifest declared_at_utc differs")
        if manifest_protocol.get("derivative_width") != plan.derivative_width:
            fail("manifest", "protocol_drift", "manifest derivative_width differs")
    manifest_implementation = manifest.get("implementation")
    implementation_matches_manifest = False
    if not isinstance(manifest_implementation, dict):
        fail("manifest", "manifest_invalid", "implementation block is missing")
    elif manifest_implementation.get("sha256") != implementation["sha256"]:
        # Implementation drift is surfaced and reported. It is not treated as
        # asset drift: the immutable acquired bytes and QA are recomputed from
        # the protocol, while protocol hash drift remains fatal.
        warnings.append(
            f"implementation_sha256_drift:{manifest_implementation.get('sha256')}"
        )
    else:
        implementation_matches_manifest = True

    manifest_assets = manifest.get("assets")
    if not isinstance(manifest_assets, list):
        fail("manifest", "manifest_invalid", "assets is not a list")
        manifest_assets = []
    if len(manifest_assets) != plan.exact_assets:
        fail(
            "manifest",
            "manifest_asset_count",
            f"manifest lists {len(manifest_assets)} assets, expected {plan.exact_assets}",
        )
    if manifest.get("asset_count") != plan.exact_assets:
        fail("manifest", "manifest_asset_count", "asset_count differs")
    if manifest.get("source_count") != plan.exact_sources:
        fail("manifest", "manifest_source_count", "source_count differs")

    expected_filenames = {f"{record['asset_id']}.jpg" for record in plan.assets}
    manifest_by_id: dict[str, dict] = {}
    for index, entry in enumerate(manifest_assets):
        if not isinstance(entry, dict):
            fail("manifest", "manifest_invalid", f"assets[{index}] is not an object")
            continue
        asset_id = entry.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            fail("manifest", "manifest_invalid", f"assets[{index}] lacks asset_id")
            continue
        if asset_id in manifest_by_id:
            fail("manifest", "manifest_duplicate_asset_id", asset_id)
        manifest_by_id[asset_id] = entry
        if entry.get("filename") not in expected_filenames:
            fail(
                "manifest",
                "manifest_filename_drift",
                f"{asset_id}: unexpected filename {entry.get('filename')!r}",
            )
    expected_asset_ids = {record["asset_id"] for record in plan.assets}
    for extra_id in sorted(set(manifest_by_id) - expected_asset_ids):
        fail("manifest", "manifest_asset_id_drift", f"unexpected asset_id {extra_id!r}")

    acquisition_path = report_dir / ACQUISITION_NAME
    acquisition_evidence: dict | None = None
    acquisition_state = "missing"
    acquisition_sha256: str | None = None
    if acquisition_path.is_file():
        try:
            acquisition_raw = acquisition_path.read_bytes()
        except OSError as error:
            acquisition_state = "invalid"
            fail(
                "acquisition",
                "acquisition_invalid",
                f"cannot read acquisition receipt: {type(error).__name__}: {error}",
            )
        else:
            acquisition_sha256 = sha256_bytes(acquisition_raw)
            try:
                acquisition_receipt = json.loads(acquisition_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                acquisition_state = "invalid"
                fail(
                    "acquisition",
                    "acquisition_invalid",
                    f"invalid acquisition JSON: {error}",
                )
            else:
                reason, detail = _acquisition_structure_problem(
                    acquisition_receipt, plan
                )
                if reason is not None:
                    acquisition_state = (
                        "incompatible"
                        if reason == "acquisition_incompatible"
                        else "invalid"
                    )
                    fail("acquisition", reason, detail)
                elif acquisition_receipt["status"] == "admitted":
                    binding_problem = _acquisition_binding_problem(
                        acquisition_receipt,
                        plan,
                        manifest=manifest,
                        manifest_sha256=manifest_sha,
                        manifest_bytes=len(manifest_raw),
                        manifest_assets=manifest_assets,
                    )
                    if binding_problem:
                        acquisition_state = "incompatible"
                        fail(
                            "acquisition",
                            "acquisition_incompatible",
                            binding_problem,
                        )
                    else:
                        acquisition_state = "included"
                        acquisition_evidence = _acquisition_evidence(
                            acquisition_receipt,
                            acquisition_path,
                            acquisition_sha256,
                        )
                else:
                    acquisition_state = "not_admitted"
                    warnings.append("acquisition_receipt_not_admitted")
    elif acquisition_path.exists():
        acquisition_state = "invalid"
        fail(
            "acquisition",
            "acquisition_invalid",
            f"acquisition receipt is not a regular file: {acquisition_path}",
        )
    else:
        acquisition_state = "missing"
        fail(
            "acquisition",
            "acquisition_missing",
            f"required acquisition receipt is missing: {acquisition_path}",
        )

    if not assets_dir.is_dir() or assets_dir.is_symlink():
        fail(
            "storage",
            "assets_dir_not_usable",
            f"missing or unusable assets directory: {assets_dir}",
        )
        actual_names: set[str] = set()
    else:
        try:
            entries = list(assets_dir.iterdir())
        except OSError as error:
            fail("storage", "assets_dir_not_usable", f"{type(error).__name__}: {error}")
            entries = []
        actual_names = {entry.name for entry in entries}
        for entry in entries:
            if entry.is_symlink():
                fail("storage", "asset_symlink", entry.name)
            elif not entry.is_file():
                fail("storage", "asset_file_not_regular", entry.name)
        for missing in sorted(expected_filenames - actual_names):
            fail("storage", "missing_asset_file", missing)
        for extra in sorted(actual_names - expected_filenames):
            fail("storage", "extra_asset_file", extra)

    seen_sha: dict[str, str] = {}
    total_asset_bytes = 0
    for record in plan.assets:
        entry = manifest_by_id.get(record["asset_id"])
        if entry is None:
            fail("manifest", "manifest_missing_asset", record["asset_id"], record)
            continue
        filename = f"{record['asset_id']}.jpg"
        path = assets_dir / filename
        checks = {name: False for name in ASSET_CHECK_NAMES}
        checks.update(
            {
                "manifest_source_match": False,
                "manifest_derivative_match": False,
            }
        )
        derivative: dict | None = None
        manifest_checks = entry.get("checks")
        manifest_checks_ok = (
            isinstance(manifest_checks, dict)
            and set(ASSET_CHECK_NAMES) <= set(manifest_checks)
            and all(manifest_checks[name] is True for name in ASSET_CHECK_NAMES)
        )

        if entry.get("source_id") != record["source_id"]:
            fail("manifest", "manifest_source_drift", "source_id differs", record)
        if entry.get("group_key") != record["group_key"]:
            fail("manifest", "manifest_source_drift", "group_key differs", record)
        if entry.get("split") != record["split"]:
            fail("manifest", "manifest_split_drift", "split differs", record)
        if entry.get("filename") != filename:
            fail("manifest", "manifest_filename_drift", "filename differs", record)
        if entry.get("source") != _source_block(record):
            fail(
                "manifest",
                "manifest_source_drift",
                "source rights/revision/provenance block differs from the protocol",
                record,
            )
        else:
            checks["manifest_source_match"] = True

        if path.is_symlink() or not path.is_file():
            fail("storage", "missing_asset_file", filename, record)
        else:
            try:
                data = path.read_bytes()
            except OSError as error:
                data = b""
                fail(
                    "storage",
                    "asset_read_failed",
                    f"{type(error).__name__}: {error}",
                    record,
                )
            actual_bytes = len(data)
            actual_sha = sha256_bytes(data)
            total_asset_bytes += actual_bytes
            checks["byte_length"] = True
            checks["sha256"] = bool(SHA256_PATTERN.fullmatch(actual_sha))
            checks["exact_hash_unique"] = actual_sha not in seen_sha
            checks["split_matches_protocol"] = (
                record["split"] == plan.group_splits[record["group_key"]]
            )
            checks["source_group_match"] = (
                plan.group_splits.get(record["group_key"]) == record["split"]
            )
            checks["rights_fields"] = _rights_fields_present(
                record, plan.protocol["license_policy"]
            )
            checks["license_allowed"] = _license_allowed(
                record["license"], plan.protocol["license_policy"]
            )
            checks["source_not_excluded"] = not _benchmark_excluded(
                record["source_id"],
                record["group_key"],
                record["asset_id"],
                plan.protocol["leakage"],
            )

            qa_derivative, jpeg_checks, qa_reason, qa_detail = _qa_jpeg(
                data,
                record["original_width"],
                record["original_height"],
                plan.derivative_width,
            )
            checks.update(jpeg_checks)
            derivative = qa_derivative
            if derivative is not None:
                derivative["bytes"] = actual_bytes
                derivative["sha256"] = actual_sha
            elif qa_reason:
                fail("qa", qa_reason, qa_detail, record)

            manifest_derivative = entry.get("derivative")
            if not isinstance(manifest_derivative, dict):
                fail("manifest", "manifest_invalid", "derivative block missing", record)
            elif derivative is not None:
                expected_derivative = {
                    key: derivative[key]
                    for key in (
                        "bytes",
                        "sha256",
                        "mime",
                        "format",
                        "width",
                        "height",
                        "max_dimension",
                    )
                }
                if manifest_derivative != expected_derivative:
                    fail(
                        "manifest",
                        "derivative_metadata_drift",
                        f"manifest {manifest_derivative!r} != recomputed "
                        f"{expected_derivative!r}",
                        record,
                    )
                else:
                    checks["manifest_derivative_match"] = True
                if manifest_derivative.get("bytes") != actual_bytes:
                    fail("manifest", "byte_length_mismatch", "derivative bytes differ", record)
                if manifest_derivative.get("sha256") != actual_sha:
                    fail("manifest", "sha256_mismatch", "derivative sha256 differs", record)
            if not manifest_checks_ok:
                fail(
                    "manifest",
                    "manifest_check_failed",
                    "manifest per-asset checks are missing or not all true",
                    record,
                )
            if not all(checks.values()):
                failed = sorted(name for name, value in checks.items() if not value)
                reason = (
                    "duplicate_exact_sha256"
                    if not checks["exact_hash_unique"]
                    else "check_failed"
                )
                fail("qa", reason, f"failed checks: {failed}", record)
            seen_sha.setdefault(actual_sha, record["asset_id"])

        results.append(
            {
                "asset_id": record["asset_id"],
                "source_id": record["source_id"],
                "split": record["split"],
                "filename": filename,
                "source": _source_block(record),
                "derivative": derivative or {},
                "checks": dict(sorted(checks.items())),
            }
        )

    if len(seen_sha) != len(plan.assets) and not any(
        failure["reason"] == "duplicate_exact_sha256" for failure in failures
    ):
        # A missing/unreadable file also reduces the unique set; dedup is only
        # meaningful for successfully read assets.
        pass

    manifest_splits = manifest.get("splits")
    recomputed_splits = {split: {"assets": 0, "bytes": 0} for split in SPLITS}
    recomputed_sources: dict[str, dict] = {}
    for entry in results:
        derivative = entry.get("derivative") or {}
        byte_count = int(derivative.get("bytes") or 0)
        split = entry.get("split")
        if split in recomputed_splits:
            recomputed_splits[split]["assets"] += 1
            recomputed_splits[split]["bytes"] += byte_count
        stats = recomputed_sources.setdefault(
            entry["source_id"], {"assets": 0, "bytes": 0}
        )
        stats["assets"] += 1
        stats["bytes"] += byte_count
    if manifest_splits != recomputed_splits:
        fail(
            "manifest",
            "manifest_split_drift",
            f"manifest splits {manifest_splits!r} != recomputed {recomputed_splits!r}",
        )
    expected_source_counts = {
        record["source_id"]: {"assets": 0, "bytes": 0} for record in plan.assets
    }
    for record in plan.assets:
        expected_source_counts[record["source_id"]]["assets"] += 1
    recomputed_sources_with_bytes = {
        source_id: stats for source_id, stats in sorted(recomputed_sources.items())
    }
    manifest_sources = manifest.get("source_counts")
    if not isinstance(manifest_sources, dict) or set(manifest_sources) != set(
        expected_source_counts
    ):
        fail("manifest", "manifest_source_count", "source_counts keys differ")
    else:
        expected_sources = {
            source_id: {
                "assets": sum(
                    1 for record in plan.assets if record["source_id"] == source_id
                ),
                "bytes": recomputed_sources_with_bytes.get(source_id, {}).get("bytes", 0),
            }
            for source_id in expected_source_counts
        }
        if manifest_sources != expected_sources:
            fail("manifest", "manifest_source_count", "source_counts values differ")
    if manifest.get("total_bytes") != total_asset_bytes:
        fail(
            "manifest",
            "manifest_total_bytes_drift",
            f"manifest total_bytes {manifest.get('total_bytes')!r} != {total_asset_bytes}",
        )
    for split in SPLITS:
        if recomputed_splits[split]["assets"] != plan.expected_asset_counts[split]:
            fail(
                "split",
                "expected_asset_count_mismatch",
                f"{split}: {recomputed_splits[split]['assets']} != "
                f"{plan.expected_asset_counts[split]}",
            )
    if failures:
        report = _manifest_failure_report(
            mode="verify",
            protocol_info=protocol_info,
            implementation=implementation,
            started_at=started_at,
            started_monotonic=started_monotonic,
            assets_dir=assets_dir,
            report_dir=report_dir,
            failures=failures,
            results=results,
            storage={
                "asset_bytes": total_asset_bytes,
                "manifest_bytes": len(manifest_raw),
                "total_stored_bytes": total_asset_bytes + len(manifest_raw),
                "asset_count": len(results),
                "source_count": len(recomputed_sources_with_bytes),
            },
            include_manifest=True,
            finalize=False,
            acquisition=acquisition_evidence,
            acquisition_receipt=acquisition_state,
            acquisition_path=acquisition_path,
            acquisition_sha256=acquisition_sha256,
        )
        report["splits"] = recomputed_splits
        report["source_counts"] = recomputed_sources_with_bytes
        report["requests"]["max_requests"] = plan.max_requests
        report["bytes"].update(
            {
                "max_asset_bytes": plan.max_asset_bytes,
                "max_total_bytes": plan.max_total_bytes,
            }
        )
        report["verification"] = {
            "performed": True,
            "recomputed": True,
            "manifest_path": _display_path(manifest_path),
            "manifest_sha256": manifest_sha,
            "implementation_matches_manifest": implementation_matches_manifest,
            "acquisition_receipt": acquisition_state,
            "acquisition_receipt_path": _display_path(acquisition_path),
            "acquisition_receipt_sha256": acquisition_sha256,
        }
        report["acquisition"] = acquisition_evidence
        report["warnings"] = warnings
        # Re-finalize so the modified report (splits/verification) is written.
        return _finalize_report(report, report_dir)

    report = _base_report(
        mode="verify",
        protocol_info=protocol_info,
        implementation=implementation,
        started_at_utc=started_at,
        started_monotonic=started_monotonic,
    )
    report["status"] = "verified"
    report["verified"] = True
    report["admitted_count"] = len(results)
    report["asset_count"] = len(results)
    report["assets"] = [
        _report_asset(record, entry["derivative"], entry["checks"])
        for record, entry in zip(plan.assets, results)
    ]
    report["asset_results"] = results
    report["splits"] = recomputed_splits
    report["source_counts"] = recomputed_sources_with_bytes
    report["checks"] = _result_checks(results, failures, include_manifest=True)
    report["checks"]["all_or_nothing_admission"] = True
    report["checks"]["all_checks_passed"] = all(report["checks"].values())
    report["requests"]["max_requests"] = plan.max_requests
    report["bytes"].update(
        {
            "received": 0,
            "staged": 0,
            "admitted": total_asset_bytes,
            "max_asset_bytes": plan.max_asset_bytes,
            "max_total_bytes": plan.max_total_bytes,
        }
    )
    report["storage"].update(
        {
            "assets_dir": _display_path(assets_dir),
            "asset_bytes": total_asset_bytes,
            "manifest_bytes": len(manifest_raw),
            "total_stored_bytes": total_asset_bytes + len(manifest_raw),
            "asset_count": len(results),
            "source_count": len(recomputed_sources_with_bytes),
        }
    )
    report["verification"] = {
        "performed": True,
        "recomputed": True,
        "manifest_path": _display_path(manifest_path),
        "manifest_sha256": manifest_sha,
        "implementation_matches_manifest": implementation_matches_manifest,
        "acquisition_receipt": acquisition_state,
        "acquisition_receipt_path": _display_path(acquisition_path),
        "acquisition_receipt_sha256": acquisition_sha256,
    }
    report["acquisition"] = acquisition_evidence
    report["network_used"] = False
    report["warnings"] = warnings
    return _finalize_report(report, report_dir)


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    assets_dir: str | Path = DEFAULT_ASSETS_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    *,
    mode: str = "verify",
    opener: Callable[[str, float], Any] | None = None,
) -> dict:
    if mode == "download":
        return download(config_path, assets_dir, report_dir, opener=opener)
    if mode == "verify":
        return verify(config_path, assets_dir, report_dir)
    raise ValueError("mode must be 'download' or 'verify'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded real-image derivative acquisition and offline verification. "
            "Default mode verifies existing assets without network."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="frozen real-image tranche protocol JSON",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help="destination/verification directory for the seven image bytes",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="directory for tracked manifest.json/report.json",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--download",
        action="store_true",
        help="perform the single bounded seven-thumbnail acquisition",
    )
    group.add_argument(
        "--verify",
        action="store_true",
        help="offline verification of existing assets against manifest.json (default)",
    )
    args = parser.parse_args(argv)

    if args.download:
        report = download(args.config, args.assets_dir, args.report_dir)
    else:
        report = verify(args.config, args.assets_dir, args.report_dir)

    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": report["mode"],
                "admitted_count": report["admitted_count"],
                "verified": report["verified"],
                "failures": len(report["failures"]),
                "report": report["storage"].get("report_path"),
            },
            sort_keys=True,
        )
    )
    if not report.get("report_written", True):
        return 1
    if any(
        not item.get("written", False)
        for item in (report.get("evidence") or {}).values()
    ):
        return 1
    if report["status"] in {"admitted", "verified"}:
        return 0
    reasons = {failure["reason"] for failure in report["failures"]}
    if reasons & {"protocol_invalid", "manifest_missing"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
