#!/usr/bin/env python3
"""Acquire the two user-authorised official Spark-X2.6-4B snapshots, bounded.

This is a byte-level acquisition tool. It never imports torch, transformers,
safetensors or the remote model code, and it never runs a model. All revisions,
sizes and SHA256 values are derived from the reviewed source-preparation plan
(``configs/spark_source_preparation_v2.json``) and cross-checked against the
retained preparation manifest; this plan embeds no independent pins.

Every snapshot is staged on the same filesystem, resumed only from a verified
partial prefix, verified with an exact full-file SHA256, and renamed into place
only after every file passes. A completed snapshot is verified in place and
reused; it is never copied or downloaded again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "replica.spark_weight_acquisition_plan.v1"
MANIFEST_SCHEMA = "replica.spark_weight_acquisition_manifest.v1"
RECEIPT_SCHEMA = "replica.spark_weight_acquisition_receipt.v1"
FAILURE_SCHEMA = "replica.spark_weight_acquisition_failure.v1"
STAGING_SCHEMA = "replica.spark_weight_acquisition_staging.v1"
DEFAULT_PLAN = Path("configs/spark_weight_acquisition_v1.json")
USER_AGENT = "replica-lab-spark-acquisition/1.0"

# Hard ceilings are independent of the plan so a malformed plan cannot
# authorise an unbounded transfer.
HARD_MAX_FILE_BYTES = 8 * 1024 ** 3
HARD_MAX_ASSET_BYTES = 64 * 1024 ** 2
HARD_MAX_SNAPSHOT_BYTES = 16 * 1024 ** 3
HARD_MAX_TOTAL_BYTES = 64 * 1024 ** 3
MAX_CHUNK_BYTES = 64 * 1024 ** 2

SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9._/-]+$")
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")

REMOTE_CODE_FILES = frozenset({"configuration_spark.py", "modeling_spark.py"})
TOKENIZER_FILES = frozenset({
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "tokenizer.model",
})
ASSET_KINDS = {
    "config.json": "config",
    "generation_config.json": "generation_config",
    "chat_template.jinja": "chat_template",
    "LICENSE": "license",
    "README.md": "readme",
}


class AcquisitionError(RuntimeError):
    """Base error for a refused or failed acquisition step."""

    reason = "acquisition_error"

    def evidence(self) -> dict:
        return {"reason": self.reason, "detail": str(self)}


class PlanError(AcquisitionError):
    reason = "invalid_plan"


class PinMismatch(AcquisitionError):
    reason = "pin_mismatch"


class FetchRefused(AcquisitionError):
    reason = "fetch_refused"

    def __init__(self, message: str, *, url: str | None = None,
                 status: int | None = None, detail: str | None = None):
        super().__init__(f"{message}: {detail}" if detail else message)
        self.reason = message
        self.url = url
        self.status = status
        self.detail = detail

    def evidence(self) -> dict:
        return {"reason": self.reason, "url": self.url,
                "http_status": self.status, "detail": self.detail}


class BudgetExceeded(AcquisitionError):
    reason = "budget_exceeded"


class PartialCorrupt(AcquisitionError):
    reason = "partial_corrupt"

    def __init__(self, message: str, *, filename: str | None = None,
                 detail: dict | None = None):
        super().__init__(message)
        self.filename = filename
        self.detail = detail or {}

    def evidence(self) -> dict:
        return {"reason": self.reason, "filename": self.filename,
                "detail": self.detail}


class HashMismatch(AcquisitionError):
    reason = "hash_mismatch"

    def __init__(self, message: str, *, filename: str | None = None,
                 expected: str | None = None, actual: str | None = None):
        super().__init__(message)
        self.filename = filename
        self.expected = expected
        self.actual = actual
        self.detail: dict = {}

    def evidence(self) -> dict:
        return {"reason": self.reason, "filename": self.filename,
                "expected_sha256": self.expected, "actual_sha256": self.actual,
                "detail": self.detail}


class SnapshotConflict(AcquisitionError):
    reason = "snapshot_conflict"


# ---------------------------------------------------------------------------
# Small utilities


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(value)).encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _fresh_receipt_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}.{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise SnapshotConflict(f"cannot allocate a fresh failure receipt path: {path}")


def _safe_relative(name: Any, field: str) -> str:
    if not isinstance(name, str) or not name or not SAFE_RELATIVE.fullmatch(name):
        raise PlanError(f"{field} must be a safe relative path")
    parsed = PurePosixPath(name)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != name:
        raise PlanError(f"{field} must be a normalised relative path")
    return name


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise PlanError(f"path escapes the acquisition root: {relative}")
    return path


def _require_int(value: Any, field: str, *, minimum: int = 1,
                 maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum:
        raise PlanError(f"{field} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise PlanError(f"{field} exceeds the hard ceiling {maximum}")
    return value


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise PlanError(f"{field} must be a lowercase SHA256")
    return value


def _classify(filename: str) -> tuple[str, str, bool]:
    """Return (kind, payload_class, weight_payload)."""
    if filename.endswith(".safetensors"):
        return "weights", "weight_shard", True
    if filename.endswith(".index.json"):
        return "weights_index", "weight_index", False
    if filename in REMOTE_CODE_FILES:
        return "remote_code", "remote_code", False
    if filename in TOKENIZER_FILES:
        return "tokenizer", "tokenizer", False
    if filename in ASSET_KINDS:
        return ASSET_KINDS[filename], "asset", False
    if filename.endswith(".json"):
        return "config", "asset", False
    return "asset", "asset", False


# ---------------------------------------------------------------------------
# Plan loading: pins are derived, never authored here


def _load_pinned_source(pins: Mapping[str, Any], manifest: Mapping[str, Any],
                        declared: Mapping[str, Any]) -> dict:
    role = declared.get("role")
    if not isinstance(role, str) or not role:
        raise PlanError("sources[].role must be a non-empty string")
    pinned_by_role = {source.get("role"): source for source in pins.get("sources", [])}
    pin = pinned_by_role.get(role)
    if pin is None:
        raise PinMismatch(f"role {role} is absent from the pinned source plan")
    if pin.get("repo_id") != declared.get("repo_id"):
        raise PinMismatch(f"{role}: repo_id differs from the pinned source plan")
    if pin.get("revision") != declared.get("revision"):
        raise PinMismatch(f"{role}: revision differs from the pinned source plan")
    if not REVISION.fullmatch(str(pin.get("revision"))):
        raise PinMismatch(f"{role}: revision is not a full 40-character commit")
    if pin.get("storage") != declared.get("storage"):
        raise PinMismatch(f"{role}: storage differs from the pinned source plan")

    files: list[dict] = []
    for weight in pin.get("weight_files") or []:
        filename = _safe_relative(weight.get("filename"), f"{role}.weight_files.filename")
        kind, payload_class, weight_payload = _classify(filename)
        if not weight_payload:
            raise PinMismatch(f"{role}: pinned weight {filename} is not a safetensors shard")
        files.append({
            "filename": filename,
            "kind": kind,
            "payload_class": payload_class,
            "weight_payload": True,
            "bytes": _require_int(weight.get("bytes"), f"{role}.{filename}.bytes"),
            "sha256": _require_sha(weight.get("hub_lfs_sha256"),
                                   f"{role}.{filename}.hub_lfs_sha256"),
        })
    if not files:
        raise PlanError(f"{role}: weight_files must not be empty")

    assets = list(pin.get("assets") or [])
    asset_names = {asset.get("filename") for asset in assets}
    index_name = pin.get("weight_index")
    if index_name is not None:
        _safe_relative(index_name, f"{role}.weight_index")
        if index_name not in asset_names:
            raise PlanError(f"{role}: weight_index must also be a pinned asset")
    for asset in assets:
        filename = _safe_relative(asset.get("filename"), f"{role}.assets.filename")
        kind, payload_class, weight_payload = _classify(filename)
        if filename == index_name:
            kind, payload_class = "weights_index", "weight_index"
        files.append({
            "filename": filename,
            "kind": kind,
            "payload_class": payload_class,
            "weight_payload": weight_payload,
            "bytes": _require_int(asset.get("expected_bytes"),
                                  f"{role}.{filename}.expected_bytes"),
            "sha256": _require_sha(asset.get("expected_sha256"),
                                   f"{role}.{filename}.expected_sha256"),
        })
    names = [record["filename"] for record in files]
    if len(names) != len(set(names)):
        raise PlanError(f"{role}: duplicate filenames in the pinned file list")

    config_record = next((record for record in files
                          if record["filename"] == "config.json"), None)
    if config_record is None:
        raise PlanError(f"{role}: a pinned config.json is required")
    if not any(record["kind"] == "remote_code" for record in files):
        raise PlanError(f"{role}: pinned remote-code sources are required")
    if not any(record["kind"] == "tokenizer" for record in files):
        raise PlanError(f"{role}: pinned tokenizer files are required")
    if not any(record["filename"] == "generation_config.json" for record in files):
        raise PlanError(f"{role}: generation_config.json is required")

    # Preparation-manifest cross-check. The manifest must agree with every pin
    # and must remain header-only evidence.
    manifest_sources = {source.get("role"): source
                        for source in manifest.get("sources", [])}
    evidence = manifest_sources.get(role)
    if evidence is None:
        raise PinMismatch(f"{role}: missing from the preparation manifest")
    if evidence.get("repo_id") != pin.get("repo_id"):
        raise PinMismatch(f"{role}: preparation manifest repo_id differs")
    if evidence.get("revision") != pin.get("revision"):
        raise PinMismatch(f"{role}: preparation manifest revision differs")
    manifest_weights = {item.get("filename"): item
                        for item in evidence.get("weight_files") or []}
    for record in (item for item in files if item["weight_payload"]):
        observed = manifest_weights.get(record["filename"])
        if observed is None:
            raise PinMismatch(f"{role}: {record['filename']} missing from preparation manifest")
        if observed.get("bytes") != record["bytes"]:
            raise PinMismatch(f"{role}: {record['filename']} byte count differs from preparation manifest")
        if observed.get("hub_lfs_sha256") != record["sha256"]:
            raise PinMismatch(f"{role}: {record['filename']} SHA256 differs from preparation manifest")
        if observed.get("locally_verified") is not False:
            raise PinMismatch(f"{role}: preparation manifest must not claim local weight verification")
    manifest_assets = {item.get("filename"): item
                       for item in evidence.get("assets") or []}
    for record in (item for item in files if not item["weight_payload"]):
        observed = manifest_assets.get(record["filename"])
        if observed is None:
            raise PinMismatch(f"{role}: {record['filename']} missing from preparation manifest assets")
        if observed.get("bytes") != record["bytes"]:
            raise PinMismatch(f"{role}: {record['filename']} byte count differs from preparation manifest")
        if observed.get("sha256") != record["sha256"]:
            raise PinMismatch(f"{role}: {record['filename']} SHA256 differs from preparation manifest")

    headers = evidence.get("safetensors_headers") or []
    header_by_name = {item.get("filename"): item for item in headers}
    header_records = []
    for record in (item for item in files if item["weight_payload"]):
        observed = header_by_name.get(record["filename"])
        if observed is None:
            raise PinMismatch(f"{role}: {record['filename']} has no header evidence")
        if observed.get("total_bytes") != record["bytes"]:
            raise PinMismatch(f"{role}: {record['filename']} header total_bytes differs")
        header_bytes = _require_int(observed.get("header_bytes"),
                                    f"{role}.{record['filename']}.header_bytes")
        payload_bytes = _require_int(observed.get("payload_bytes"),
                                     f"{role}.{record['filename']}.payload_bytes")
        if header_bytes + payload_bytes + 8 != record["bytes"]:
            raise PinMismatch(
                f"{role}: {record['filename']} header/payload split plus the "
                "8-byte length prefix does not sum to the pin")
        header_records.append({
            "filename": record["filename"],
            "header_bytes": header_bytes,
            "header_prefix_bytes": 8,
            "header_sha256": _require_sha(
                observed.get("header_sha256"),
                f"{role}.{record['filename']}.header_sha256"),
            "payload_bytes": payload_bytes,
            "total_bytes": record["bytes"],
        })

    inventory = evidence.get("tensor_inventory") or {}
    weight_bytes = sum(item["bytes"] for item in files if item["weight_payload"])
    asset_bytes = sum(item["bytes"] for item in files if not item["weight_payload"])
    snapshot_bytes = weight_bytes + asset_bytes
    if declared.get("expected_snapshot_bytes") != snapshot_bytes:
        raise PinMismatch(f"{role}: expected_snapshot_bytes differs from the derived pins")
    if declared.get("weight_file_count") != sum(
            1 for item in files if item["weight_payload"]):
        raise PinMismatch(f"{role}: weight_file_count differs from the derived pins")
    if declared.get("asset_count") != sum(
            1 for item in files if not item["weight_payload"]):
        raise PinMismatch(f"{role}: asset_count differs from the derived pins")
    if declared.get("weight_index") != index_name:
        raise PinMismatch(f"{role}: weight_index differs from the derived pins")

    return {
        "role": role,
        "repo_id": pin["repo_id"],
        "revision": pin["revision"],
        "storage": pin["storage"],
        "output_dir": _safe_relative(declared.get("output_dir"), f"{role}.output_dir"),
        "files": files,
        "weight_bytes": weight_bytes,
        "asset_bytes": asset_bytes,
        "snapshot_bytes": snapshot_bytes,
        "weight_index": index_name,
        "quantization": pin.get("quantization"),
        "parameter_elements_excluding_scales": inventory.get(
            "parameter_elements_excluding_scales"),
        "header_evidence": {
            "path": None,  # filled by load_acquisition_plan
            "sha256": None,
            "header_only": True,
            "payload_downloaded_during_preparation": False,
            "shards": header_records,
        },
    }


def load_acquisition_plan(path: str | os.PathLike,
                          *, root: str | os.PathLike | None = None) -> dict:
    root_path = Path(root).resolve() if root is not None else ROOT
    plan_path = Path(path)
    raw = _read_json(plan_path)
    if not isinstance(raw, Mapping):
        raise PlanError("plan must be a JSON object")
    if raw.get("schema") != SCHEMA:
        raise PlanError(f"plan schema must be {SCHEMA}")

    authorization = raw.get("authorization") or {}
    if authorization.get("download_authorized") is not True:
        raise PlanError("download_authorized must be true for this plan")
    if authorization.get("model_execution") is not False:
        raise PlanError("model_execution must be false: acquisition is byte-level only")
    if authorization.get("remote_code_execution") is not False:
        raise PlanError("remote_code_execution must be false")
    authorized_repos = set(authorization.get("authorized_repo_ids") or [])
    if not authorized_repos:
        raise PlanError("authorized_repo_ids must not be empty")

    pins_ref = raw.get("pins") or {}
    source_ref = pins_ref.get("source_plan") or {}
    manifest_ref = pins_ref.get("preparation_manifest") or {}
    source_path = _resolve(root_path, _safe_relative(source_ref.get("path"), "pins.source_plan.path"))
    manifest_path = _resolve(root_path, _safe_relative(
        manifest_ref.get("path"), "pins.preparation_manifest.path"))
    if not source_path.is_file() or not manifest_path.is_file():
        raise PinMismatch("pinned source plan or preparation manifest is missing")
    if sha256_file(source_path) != _require_sha(source_ref.get("sha256"), "pins.source_plan.sha256"):
        raise PinMismatch("source-preparation plan bytes differ from the declared SHA256")
    if sha256_file(manifest_path) != _require_sha(
            manifest_ref.get("sha256"), "pins.preparation_manifest.sha256"):
        raise PinMismatch("preparation manifest bytes differ from the declared SHA256")

    pins = _read_json(source_path)
    manifest = _read_json(manifest_path)
    if pins.get("schema") != source_ref.get("schema"):
        raise PinMismatch("source-preparation plan schema differs from the declared schema")
    if manifest.get("schema") != manifest_ref.get("schema"):
        raise PinMismatch("preparation manifest schema differs from the declared schema")
    if manifest.get("schema") == "replica.spark_source_preparation.v1":
        fetch_policy = manifest.get("fetch_policy") or {}
        if fetch_policy.get("weights_downloaded") is not False:
            raise PinMismatch("preparation manifest must record weights_downloaded=false")

    limits = raw.get("limits") or {}
    max_file_bytes = _require_int(limits.get("max_file_bytes"), "limits.max_file_bytes",
                                  maximum=HARD_MAX_FILE_BYTES)
    max_asset_bytes = _require_int(limits.get("max_asset_bytes"), "limits.max_asset_bytes",
                                   maximum=HARD_MAX_ASSET_BYTES)
    max_snapshot_bytes = _require_int(limits.get("max_snapshot_bytes"),
                                      "limits.max_snapshot_bytes",
                                      maximum=HARD_MAX_SNAPSHOT_BYTES)
    max_total_bytes = _require_int(limits.get("max_total_acquisition_bytes"),
                                   "limits.max_total_acquisition_bytes",
                                   maximum=HARD_MAX_TOTAL_BYTES)
    min_free_bytes = _require_int(limits.get("min_free_bytes"), "limits.min_free_bytes")
    chunk_bytes = _require_int(limits.get("chunk_bytes"), "limits.chunk_bytes",
                               maximum=MAX_CHUNK_BYTES)
    checkpoint_bytes = _require_int(limits.get("checkpoint_bytes"), "limits.checkpoint_bytes")
    timeout_seconds = _require_int(limits.get("timeout_seconds"), "limits.timeout_seconds",
                                   maximum=3600)
    if checkpoint_bytes < chunk_bytes:
        raise PlanError("limits.checkpoint_bytes must be >= limits.chunk_bytes")

    budget = raw.get("budget") or {}
    total_budget = _require_int(budget.get("total_budget_bytes"), "budget.total_budget_bytes",
                                maximum=HARD_MAX_TOTAL_BYTES)
    if total_budget != max_total_bytes:
        raise PlanError("budget.total_budget_bytes must equal limits.max_total_acquisition_bytes")
    for field in ("planned_weights_bytes", "planned_assets_bytes",
                  "planned_payload_bytes", "environment_reserve_bytes",
                  "evidence_reserve_bytes", "unallocated_bytes"):
        _require_int(budget.get(field), f"budget.{field}", minimum=0)

    declared_sources = raw.get("sources") or []
    if not declared_sources:
        raise PlanError("sources must not be empty")
    sources = []
    for declared in declared_sources:
        source = _load_pinned_source(pins, manifest, declared)
        if source["repo_id"] not in authorized_repos:
            raise PlanError(f"{source['role']}: repo_id is not user-authorised")
        source["header_evidence"]["path"] = str(manifest_path.relative_to(root_path)) \
            if manifest_path.is_relative_to(root_path) else str(manifest_path)
        source["header_evidence"]["sha256"] = manifest_ref["sha256"]
        sources.append(source)

    roles = [source["role"] for source in sources]
    if roles != list(raw.get("access_order") or []):
        raise PlanError("access_order must list every source role exactly once, in order")
    if len({source["output_dir"] for source in sources}) != len(sources):
        raise PlanError("output_dir values must be unique")

    planned_weights = sum(source["weight_bytes"] for source in sources)
    planned_assets = sum(source["asset_bytes"] for source in sources)
    planned_payload = planned_weights + planned_assets
    if budget["planned_weights_bytes"] != planned_weights:
        raise PlanError("budget.planned_weights_bytes differs from the pinned files")
    if budget["planned_assets_bytes"] != planned_assets:
        raise PlanError("budget.planned_assets_bytes differs from the pinned files")
    if budget["planned_payload_bytes"] != planned_payload:
        raise PlanError("budget.planned_payload_bytes differs from the pinned files")
    if planned_payload + budget["environment_reserve_bytes"] + \
            budget["evidence_reserve_bytes"] + budget["unallocated_bytes"] != total_budget:
        raise PlanError("budget components do not sum to total_budget_bytes")
    if planned_payload > max_total_bytes:
        raise PlanError("planned payload exceeds limits.max_total_acquisition_bytes")
    for source in sources:
        if source["snapshot_bytes"] > max_snapshot_bytes:
            raise PlanError(f"{source['role']}: snapshot exceeds limits.max_snapshot_bytes")
        for record in source["files"]:
            ceiling = max_file_bytes if record["weight_payload"] else max_asset_bytes
            if record["bytes"] > ceiling:
                raise PlanError(
                    f"{source['role']}: {record['filename']} exceeds the per-file byte cap")

    return {
        "schema": SCHEMA,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "pins": {
            "source_plan": {"path": str(source_path), "sha256": source_ref["sha256"]},
            "preparation_manifest": {"path": str(manifest_path),
                                     "sha256": manifest_ref["sha256"]},
        },
        "root": str(root_path),
        "sources": sources,
        "limits": {
            "max_file_bytes": max_file_bytes,
            "max_asset_bytes": max_asset_bytes,
            "max_snapshot_bytes": max_snapshot_bytes,
            "max_total_acquisition_bytes": max_total_bytes,
            "min_free_bytes": min_free_bytes,
            "chunk_bytes": chunk_bytes,
            "checkpoint_bytes": checkpoint_bytes,
            "timeout_seconds": timeout_seconds,
        },
        "budget": {
            "total_budget_bytes": total_budget,
            "planned_weights_bytes": planned_weights,
            "planned_assets_bytes": planned_assets,
            "planned_payload_bytes": planned_payload,
            "environment_reserve_bytes": budget["environment_reserve_bytes"],
            "evidence_reserve_bytes": budget["evidence_reserve_bytes"],
            "unallocated_bytes": budget["unallocated_bytes"],
        },
        "environment": raw.get("environment") or {},
        "data_access": raw.get("data_access") or {},
        "protocol": raw.get("protocol") or {},
        "non_claims": raw.get("non_claims") or [],
    }


# ---------------------------------------------------------------------------
# HTTP transport


class _UrllibResponse:
    __slots__ = ("_response",)

    def __init__(self, response: Any):
        self._response = response

    @property
    def status(self) -> int:
        return int(getattr(self._response, "status", None)
                   or self._response.getcode())

    @property
    def headers(self) -> Mapping[str, str]:
        return self._response.headers

    @property
    def url(self) -> str:
        return self._response.geturl()

    def read(self, size: int = -1) -> bytes:
        return self._response.read(size)

    def close(self) -> None:
        self._response.close()


class UrllibTransport:
    """Pinned HTTPS download transport. Tests inject a bounded fake instead."""

    def __init__(self, *, timeout: int = 120,
                 opener: Callable[..., Any] | None = None,
                 user_agent: str = USER_AGENT):
        self.timeout = timeout
        self.opener = opener
        self.user_agent = user_agent
        self.requests = 0

    def open(self, url: str, *, start: int = 0) -> _UrllibResponse:
        import urllib.error
        import urllib.request

        headers = {"Accept-Encoding": "identity", "User-Agent": self.user_agent}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(url, headers=headers)
        opener = self.opener or urllib.request.urlopen
        self.requests += 1
        try:
            response = opener(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            error.close()
            raise FetchRefused("http_error", url=url, status=error.code,
                               detail=str(error)) from error
        except Exception as error:
            raise FetchRefused("transport_error", url=url,
                               detail=str(error)) from error
        return _UrllibResponse(response)


def _header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    if hasattr(headers, "items"):
        for key, value in headers.items():
            if str(key).lower() == name.lower():
                return value
        return None
    return None


def _validate_response(response: Any, *, start: int, expected_bytes: int,
                       url: str) -> None:
    status = int(getattr(response, "status", 0) or 0)
    headers = getattr(response, "headers", None)
    encoding = (_header(headers, "Content-Encoding") or "identity").lower()
    if encoding not in ("", "identity"):
        raise FetchRefused("encoded_body", url=url, status=status,
                           detail=f"Content-Encoding={encoding}")
    content_length = _header(headers, "Content-Length")
    content_range = _header(headers, "Content-Range")
    if start == 0:
        if status != 200:
            raise FetchRefused("not_ok", url=url, status=status,
                               detail=f"unexpected HTTP {status}")
        if content_range:
            raise FetchRefused("unexpected_partial_content", url=url, status=status,
                               detail=content_range)
        if content_length is not None:
            if not str(content_length).isdigit() or int(content_length) != expected_bytes:
                raise FetchRefused("content_length_mismatch", url=url, status=status,
                                   detail=f"{content_length} != {expected_bytes}")
        return
    if start >= expected_bytes:
        raise FetchRefused("range_beyond_file", url=url, status=status)
    if status != 206:
        raise FetchRefused("not_partial_content", url=url, status=status,
                           detail=f"unexpected HTTP {status}")
    match = CONTENT_RANGE.fullmatch(str(content_range or ""))
    if not match:
        raise FetchRefused("invalid_content_range", url=url, status=status,
                           detail=str(content_range))
    got_start, got_end, got_total = (int(value) for value in match.groups())
    if got_start != start or got_total != expected_bytes or got_end != expected_bytes - 1:
        raise FetchRefused("content_range_mismatch", url=url, status=status,
                           detail=f"{content_range} for start={start} total={expected_bytes}")
    if content_length is not None:
        if not str(content_length).isdigit() or int(content_length) != expected_bytes - start:
            raise FetchRefused("content_length_mismatch", url=url, status=status,
                               detail=f"{content_length} != {expected_bytes - start}")


# ---------------------------------------------------------------------------
# Partial state and streaming download


def _resume_offset(part: Path, state_path: Path, *, filename: str,
                   expected_bytes: int, pinned_sha256: str) -> tuple[int, Any, int]:
    """Return (verified_offset, hasher, discarded_tail_bytes)."""
    if not part.exists():
        if state_path.exists():
            raise PartialCorrupt("partial state exists without partial bytes",
                                 filename=filename,
                                 detail={"state": str(state_path)})
        return 0, hashlib.sha256(), 0
    if part.is_symlink() or not part.is_file():
        raise PartialCorrupt("partial path is not a regular file", filename=filename)
    size = part.stat().st_size
    if size == 0:
        if state_path.exists():
            state = _read_json(state_path)
            if state.get("bytes") not in (0, None):
                raise PartialCorrupt("empty partial contradicts its state",
                                     filename=filename)
        return 0, hashlib.sha256(), 0
    if not state_path.exists():
        raise PartialCorrupt("partial bytes exist without a verification state",
                             filename=filename,
                             detail={"bytes": size})
    state = _read_json(state_path)
    if not isinstance(state, Mapping):
        raise PartialCorrupt("partial state is not an object", filename=filename)
    if state.get("filename") != filename:
        raise PartialCorrupt("partial state names a different file", filename=filename)
    if state.get("expected_bytes") != expected_bytes:
        raise PartialCorrupt("partial state byte count differs from the pin",
                             filename=filename)
    if state.get("pinned_sha256") != pinned_sha256:
        raise PartialCorrupt("partial state SHA256 pin differs", filename=filename)
    recorded = state.get("bytes")
    if type(recorded) is not int or not 0 <= recorded <= expected_bytes:
        raise PartialCorrupt("partial state offset is invalid", filename=filename)
    if size < recorded:
        raise PartialCorrupt("partial file is shorter than its recorded offset",
                             filename=filename,
                             detail={"bytes": size, "recorded": recorded})
    discarded = max(0, size - recorded)
    if discarded:
        with part.open("r+b") as stream:
            stream.truncate(recorded)
            stream.flush()
            os.fsync(stream.fileno())
    hasher = hashlib.sha256()
    remaining = recorded
    with part.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise PartialCorrupt("partial file is shorter than its recorded offset",
                                     filename=filename)
            hasher.update(chunk)
            remaining -= len(chunk)
    observed = hasher.hexdigest()
    if observed != state.get("sha256"):
        raise PartialCorrupt("partial prefix SHA256 does not match its state",
                             filename=filename,
                             detail={"recorded_sha256": state.get("sha256"),
                                     "observed_sha256": observed,
                                     "bytes": recorded})
    return recorded, hasher, discarded


def _write_partial_state(path: Path, *, filename: str, bytes_written: int,
                         digest: str, expected_bytes: int,
                         pinned_sha256: str) -> None:
    _write_json_atomic(path, {
        "schema": "replica.spark_weight_acquisition_partial.v1",
        "filename": filename,
        "bytes": bytes_written,
        "sha256": digest,
        "expected_bytes": expected_bytes,
        "pinned_sha256": pinned_sha256,
        "updated_at": utc_now(),
    })


class RunBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.downloaded = 0

    def consume(self, amount: int) -> None:
        if amount < 0 or self.downloaded + amount > self.limit:
            raise BudgetExceeded(
                f"download budget exceeded: {self.downloaded + amount} > {self.limit}")
        self.downloaded += amount


def _download_file(transport: Any, *, url: str, part: Path, state_path: Path,
                   filename: str, expected_bytes: int, pinned_sha256: str,
                   chunk_bytes: int, checkpoint_bytes: int,
                   budget: RunBudget) -> dict:
    offset, hasher, discarded = _resume_offset(
        part, state_path, filename=filename, expected_bytes=expected_bytes,
        pinned_sha256=pinned_sha256)
    resumed = offset > 0
    discarded_tail = discarded
    if offset == expected_bytes:
        # Crash after writing every byte, before the atomic rename. Verify the
        # complete prefix and finalise without any network request.
        actual = hasher.hexdigest()
        if actual != pinned_sha256:
            raise HashMismatch("completed partial fails the pinned SHA256",
                               filename=filename, expected=pinned_sha256,
                               actual=actual)
        return {"bytes": expected_bytes, "sha256": actual,
                "resumed": True, "resumed_bytes": offset,
                "discarded_tail_bytes": discarded_tail,
                "requests": 0, "bytes_downloaded": 0}

    response = transport.open(url, start=offset)
    requests = 1
    downloaded = 0
    last_checkpoint = offset
    try:
        _validate_response(response, start=offset, expected_bytes=expected_bytes, url=url)
        with part.open("ab") as stream:
            written = offset
            try:
                while written < expected_bytes:
                    wanted = min(chunk_bytes, expected_bytes - written)
                    chunk = response.read(wanted)
                    if not chunk:
                        raise FetchRefused(
                            "short_body", url=url,
                            detail=f"EOF at {written} of {expected_bytes}")
                    if len(chunk) > wanted:
                        raise FetchRefused(
                            "chunk_exceeds_bound", url=url,
                            detail=f"{len(chunk)} > {wanted}")
                    budget.consume(len(chunk))
                    stream.write(chunk)
                    hasher.update(chunk)
                    written += len(chunk)
                    downloaded += len(chunk)
                    if written - last_checkpoint >= checkpoint_bytes or \
                            written == expected_bytes:
                        stream.flush()
                        os.fsync(stream.fileno())
                        _write_partial_state(
                            state_path, filename=filename, bytes_written=written,
                            digest=hasher.hexdigest(), expected_bytes=expected_bytes,
                            pinned_sha256=pinned_sha256)
                        last_checkpoint = written
                extra = response.read(1)
                if extra:
                    raise FetchRefused(
                        "body_exceeds_pinned_size", url=url,
                        detail=f"more than {expected_bytes} bytes")
            finally:
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()

    actual = hasher.hexdigest()
    if part.stat().st_size != expected_bytes:
        raise HashMismatch("partial size differs from the pin", filename=filename,
                           expected=str(expected_bytes),
                           actual=str(part.stat().st_size))
    if actual != pinned_sha256:
        raise HashMismatch("full-file SHA256 differs from the pin", filename=filename,
                           expected=pinned_sha256, actual=actual)
    return {"bytes": expected_bytes, "sha256": actual, "resumed": resumed,
            "resumed_bytes": offset if resumed else 0,
            "discarded_tail_bytes": discarded_tail,
            "requests": requests, "bytes_downloaded": downloaded}


def _quarantine_partial(part: Path, state_path: Path, *, stamp: str) -> dict:
    """Preserve mismatched bytes under a distinct name; never delete them."""
    result = {}
    if part.exists():
        target = part.with_name(f"{part.name}.corrupt-{stamp}")
        os.replace(part, target)
        result["partial"] = str(target)
    if state_path.exists():
        target = state_path.with_name(f"{state_path.name}.corrupt-{stamp}")
        os.replace(state_path, target)
        result["state"] = str(target)
    return result


def _staging_dir(target: Path, revision: str) -> Path:
    return target.parent / f"{target.name}.{revision[:12]}.staging"


def _payload_bytes(path: Path) -> int:
    """Bytes of payload files, excluding staging state and the manifest."""
    total = 0
    if not path.exists():
        return 0
    for child in path.rglob("*"):
        if child.is_symlink() or not child.is_file():
            continue
        if child.name in ("staging.json", "acquisition_manifest.json"):
            continue
        if child.name.endswith(".part.json"):
            continue
        total += child.stat().st_size
    return total


def _default_free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


# ---------------------------------------------------------------------------
# Completed-snapshot verification and idempotence


def _verify_completed(target: Path, source: Mapping[str, Any]) -> dict:
    manifest_path = target / "acquisition_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise SnapshotConflict(f"target exists without an acquisition manifest: {target}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise SnapshotConflict("acquisition manifest is not an object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SnapshotConflict("acquisition manifest schema differs")
    for field in ("role", "repo_id", "revision", "storage", "output_dir"):
        if manifest.get(field) != source[field]:
            raise SnapshotConflict(f"completed snapshot {field} differs from the plan")
    stored_fingerprint = manifest.get("manifest_fingerprint")
    body = {key: value for key, value in manifest.items()
            if key != "manifest_fingerprint"}
    if stored_fingerprint != fingerprint(body):
        raise SnapshotConflict("acquisition manifest fingerprint is inconsistent")
    if manifest.get("status") != "complete":
        raise SnapshotConflict("acquisition manifest is not complete")
    verification = manifest.get("verification") or {}
    if verification.get("all_files_verified") is not True:
        raise SnapshotConflict("acquisition manifest does not attest full verification")

    expected = {record["filename"]: record for record in source["files"]}
    observed: dict[str, Mapping[str, Any]] = {}
    for record in manifest.get("files") or []:
        filename = record.get("filename")
        if filename in observed:
            raise SnapshotConflict(f"duplicate manifest record: {filename}")
        observed[filename] = record
    if set(observed) != set(expected):
        raise SnapshotConflict("completed snapshot file list differs from the pins")
    total = 0
    for filename, pin in expected.items():
        record = observed[filename]
        if record.get("bytes") != pin["bytes"] or record.get("sha256") != pin["sha256"]:
            raise SnapshotConflict(f"{filename}: manifest record differs from the pin")
        path = target / filename
        if path.is_symlink() or not path.is_file():
            raise SnapshotConflict(f"{filename}: missing or not a regular file")
        size = path.stat().st_size
        if size != pin["bytes"]:
            raise SnapshotConflict(f"{filename}: size {size} differs from the pin")
        if sha256_file(path) != pin["sha256"]:
            raise SnapshotConflict(f"{filename}: SHA256 differs from the pin")
        total += size
    if total != source["snapshot_bytes"]:
        raise SnapshotConflict("completed snapshot byte total differs from the pins")
    if _payload_bytes(target) != total:
        raise SnapshotConflict("completed snapshot contains unexpected extra files")
    return manifest


def verify_completed_snapshot(plan: Mapping[str, Any], source: Mapping[str, Any]) -> dict:
    root = Path(plan["root"])
    target = _resolve(root, source["output_dir"])
    manifest = _verify_completed(target, source)
    return {
        "status": "already_complete",
        "role": source["role"],
        "target": str(target),
        "manifest_path": str(target / "acquisition_manifest.json"),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "verified": True,
        "reused_completed_snapshot": True,
        "network_requests": 0,
        "bytes_downloaded": 0,
    }


# ---------------------------------------------------------------------------
# Snapshot acquisition


def _write_staging_identity(staging: Path, source: Mapping[str, Any]) -> None:
    path = staging / "staging.json"
    payload = {
        "schema": STAGING_SCHEMA,
        "role": source["role"],
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "storage": source["storage"],
        "output_dir": source["output_dir"],
        "created_at": utc_now(),
    }
    if path.exists():
        existing = _read_json(path)
        for field in ("role", "repo_id", "revision", "storage", "output_dir"):
            if existing.get(field) != payload[field]:
                raise SnapshotConflict(
                    f"staging directory belongs to a different snapshot: {staging}")
        return
    _write_json_atomic(path, payload)


def _file_url(source: Mapping[str, Any], filename: str) -> str:
    return (f"https://huggingface.co/{source['repo_id']}/resolve/"
            f"{source['revision']}/{filename}")


def acquire_snapshot(plan: Mapping[str, Any], source: Mapping[str, Any], *,
                     transport: Any, free_bytes_fn: Callable[[Path], int] | None = None,
                     now: Callable[[], str] = utc_now,
                     prior_completed_bytes: int = 0,
                     budget: RunBudget | None = None) -> dict:
    root = Path(plan["root"])
    limits = plan["limits"]
    target = _resolve(root, source["output_dir"])
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        # Idempotent reuse: verify in place, never copy or re-download.
        summary = verify_completed_snapshot(plan, source)
        summary["free_bytes_before"] = _free(free_bytes_fn, target.parent)
        summary["free_bytes_after"] = summary["free_bytes_before"]
        return summary

    staging = _staging_dir(target, source["revision"])
    staging.mkdir(parents=True, exist_ok=True)
    _write_staging_identity(staging, source)

    free_before = _free(free_bytes_fn, target.parent)
    staged_before = _payload_bytes(staging)
    required = source["snapshot_bytes"] - staged_before + limits["min_free_bytes"]
    if free_before < required:
        raise BudgetExceeded(
            f"insufficient free space: {free_before} < {required} bytes required")
    peak_staged = staged_before
    budget = budget or RunBudget(limits["max_total_acquisition_bytes"])
    started_at = now()
    resumed_any = False
    resumed_bytes = 0
    discarded_tail_bytes = 0
    network_requests = 0
    bytes_downloaded = 0
    records: list[dict] = []

    # Small assets and indexes first, then weight payloads. A partial failure
    # therefore never leaves a full weight payload in an unverified target.
    ordered = sorted(source["files"],
                     key=lambda record: (record["weight_payload"],
                                         record["filename"]))
    for record in ordered:
        filename = record["filename"]
        final = staging / filename
        part = staging / f"{filename}.part"
        state_path = staging / f"{filename}.part.json"
        if final.exists():
            if final.is_symlink() or not final.is_file() or \
                    final.stat().st_size != record["bytes"] or \
                    sha256_file(final) != record["sha256"]:
                raise SnapshotConflict(
                    f"staged file is corrupt and will not be overwritten: {final}")
            # The rename completed before an interruption: no duplicate bytes.
            if part.exists():
                part.unlink()
            if state_path.exists():
                state_path.unlink()
            records.append(_file_record(record, source, outcome="staged_verified",
                                        resumed=0, downloaded=0))
            continue

        url = _file_url(source, filename)
        try:
            result = _download_file(
                transport, url=url, part=part, state_path=state_path,
                filename=filename, expected_bytes=record["bytes"],
                pinned_sha256=record["sha256"],
                chunk_bytes=limits["chunk_bytes"],
                checkpoint_bytes=limits["checkpoint_bytes"], budget=budget)
        except HashMismatch as error:
            stamp = now().replace(":", "").replace("-", "")
            quarantined = _quarantine_partial(part, state_path, stamp=stamp)
            error.detail = {"quarantined": quarantined, **getattr(error, "detail", {})}
            raise
        network_requests += result["requests"]
        bytes_downloaded += result["bytes_downloaded"]
        resumed_any = resumed_any or result["resumed"]
        resumed_bytes += result["resumed_bytes"]
        discarded_tail_bytes += result["discarded_tail_bytes"]
        if state_path.exists():
            state_path.unlink()
        os.replace(part, final)
        staged_bytes = _payload_bytes(staging)
        peak_staged = max(peak_staged, staged_bytes)
        if staged_bytes > limits["max_snapshot_bytes"]:
            raise BudgetExceeded(
                f"staging exceeds the snapshot cap: {staged_bytes}")
        if prior_completed_bytes + staged_bytes > limits["max_total_acquisition_bytes"]:
            raise BudgetExceeded("staging plus completed snapshots exceeds the total cap")
        records.append(_file_record(record, source, outcome="downloaded",
                                    resumed=result["resumed_bytes"],
                                    downloaded=result["bytes_downloaded"]))

    peak_staged = max(peak_staged, _payload_bytes(staging))
    if _payload_bytes(staging) != source["snapshot_bytes"]:
        raise SnapshotConflict("staging byte total differs from the pins")
    for record in source["files"]:
        path = staging / record["filename"]
        if path.stat().st_size != record["bytes"] or \
                sha256_file(path) != record["sha256"]:
            raise SnapshotConflict(f"final staging verification failed: {record['filename']}")

    free_before_finalize = _free(free_bytes_fn, target.parent)
    if free_before_finalize < limits["min_free_bytes"]:
        raise BudgetExceeded("free-space reserve would be violated by finalisation")

    # The staging identity is transient state, not payload evidence; the
    # manifest records it. Removing it keeps the completed snapshot to the
    # pinned repository files plus the manifest.
    identity = staging / "staging.json"
    if identity.exists():
        identity.unlink()

    manifest = _build_manifest(
        plan, source, records, started_at=started_at, finished_at=now(),
        network_requests=network_requests, bytes_downloaded=bytes_downloaded,
        resumed=resumed_any, resumed_bytes=resumed_bytes,
        discarded_tail_bytes=discarded_tail_bytes,
        staging=str(staging), staged_bytes=peak_staged,
        free_bytes_before=free_before, free_bytes_after=free_before_finalize,
        prior_completed_bytes=prior_completed_bytes)
    manifest["acquisition"]["high_water_estimate_bytes"] = \
        prior_completed_bytes + peak_staged
    body = {key: value for key, value in manifest.items()
            if key != "manifest_fingerprint"}
    manifest["manifest_fingerprint"] = fingerprint(body)
    _write_json_atomic(staging / "acquisition_manifest.json", manifest)

    if target.exists():
        raise SnapshotConflict(f"target appeared during acquisition: {target}")
    # Same-filesystem rename: exactly one copy of the payload exists afterwards.
    os.rename(staging, target)
    free_after = _free(free_bytes_fn, target.parent)
    manifest["acquisition"]["free_bytes_after"] = free_after
    body = {key: value for key, value in manifest.items()
            if key != "manifest_fingerprint"}
    manifest["manifest_fingerprint"] = fingerprint(body)
    _write_json_atomic(target / "acquisition_manifest.json", manifest)
    return {
        "status": "complete",
        "role": source["role"],
        "target": str(target),
        "manifest_path": str(target / "acquisition_manifest.json"),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "verified": True,
        "reused_completed_snapshot": False,
        "network_requests": network_requests,
        "bytes_downloaded": bytes_downloaded,
        "resumed_from_partial": resumed_any,
        "resumed_bytes": resumed_bytes,
        "staging_path": str(staging),
        "staging_removed": not staging.exists(),
        "staged_bytes": peak_staged,
        "final_bytes": source["snapshot_bytes"],
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "high_water_estimate_bytes": prior_completed_bytes + peak_staged,
    }


def _free(free_bytes_fn: Callable[[Path], int] | None, path: Path) -> int:
    if free_bytes_fn is None:
        return _default_free_bytes(path)
    value = free_bytes_fn(path)
    if type(value) is not int or value < 0:
        raise AcquisitionError("free_bytes_fn must return a non-negative integer")
    return value


def _file_record(record: Mapping[str, Any], source: Mapping[str, Any], *,
                 outcome: str, resumed: int, downloaded: int) -> dict:
    return {
        "filename": record["filename"],
        "kind": record["kind"],
        "payload_class": record["payload_class"],
        "weight_payload": record["weight_payload"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "pinned_bytes": record["bytes"],
        "pinned_sha256": record["sha256"],
        "size_verified": True,
        "sha256_verified": True,
        "outcome": outcome,
        "resumed_bytes": resumed,
        "bytes_downloaded": downloaded,
        "url": _file_url(source, record["filename"]),
        "revision": source["revision"],
    }


def _build_manifest(plan: Mapping[str, Any], source: Mapping[str, Any],
                    records: list[dict], *, started_at: str, finished_at: str,
                    network_requests: int, bytes_downloaded: int,
                    resumed: bool, resumed_bytes: int,
                    discarded_tail_bytes: int, staging: str,
                    staged_bytes: int, free_bytes_before: int,
                    free_bytes_after: int, prior_completed_bytes: int) -> dict:
    limits = plan["limits"]
    weight_bytes = sum(record["bytes"] for record in records
                       if record["weight_payload"])
    asset_bytes = sum(record["bytes"] for record in records
                      if not record["weight_payload"])
    remote_code = [{
        "filename": record["filename"],
        "sha256": record["sha256"],
        "bytes": record["bytes"],
        "executed": False,
    } for record in records if record["kind"] == "remote_code"]
    kinds: dict[str, int] = {}
    for record in records:
        kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "role": source["role"],
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "storage": source["storage"],
        "output_dir": source["output_dir"],
        "generated_at": finished_at,
        "weight_payload_present": True,
        "header_only": False,
        "snapshot": {
            "weight_payload_present": True,
            "header_only": False,
            "weight_payload_files": sum(1 for record in records
                                        if record["weight_payload"]),
            "weight_bytes": weight_bytes,
            "asset_bytes": asset_bytes,
            "total_bytes": weight_bytes + asset_bytes,
            "files_by_kind": kinds,
            "parameter_elements_excluding_scales":
                source.get("parameter_elements_excluding_scales"),
        },
        "header_evidence": {
            **source["header_evidence"],
            "payload_downloaded_during_preparation": False,
        },
        "files": records,
        "remote_code": {
            "files": remote_code,
            "revision": source["revision"],
            "pins_sha256": {record["filename"]: record["sha256"]
                            for record in remote_code},
            "executed": False,
            "execution": "none; byte acquisition only",
        },
        "quantization": source.get("quantization"),
        "verification": {
            "final_hash_verification": "exact_sha256_full_file",
            "all_files_verified": True,
            "size_matches_pin": True,
            "sha256_matches_pin": True,
            "per_file_cap_bytes": limits["max_file_bytes"],
            "per_asset_cap_bytes": limits["max_asset_bytes"],
            "snapshot_cap_bytes": limits["max_snapshot_bytes"],
            "total_cap_bytes": limits["max_total_acquisition_bytes"],
            "headers_distinguished_from_weights": True,
        },
        "acquisition": {
            "transport": "https_huggingface_resolve",
            "started_at": started_at,
            "finished_at": finished_at,
            "network_requests": network_requests,
            "bytes_downloaded": bytes_downloaded,
            "resumed_from_partial": resumed,
            "resumed_bytes": resumed_bytes,
            "discarded_unverified_tail_bytes": discarded_tail_bytes,
            "atomic_staging": True,
            "staging_path": staging,
            "staging_removed": True,
            "staged_bytes": staged_bytes,
            "final_bytes": weight_bytes + asset_bytes,
            "free_bytes_before": free_bytes_before,
            "free_bytes_after": free_bytes_after,
            "budget_limit_bytes": limits["max_total_acquisition_bytes"],
            "budget_respected": True,
            "prior_completed_bytes": prior_completed_bytes,
            "high_water_estimate_bytes": prior_completed_bytes + staged_bytes,
            "duplicate_completed_payload": False,
        },
        "non_claims": list(plan.get("non_claims") or []),
    }


# ---------------------------------------------------------------------------
# Run orchestration and CLI


def _failure_receipt(target: Path, source: Mapping[str, Any], error: BaseException,
                     *, staging: Path, free_before: int, free_after: int,
                     stamp: str, network_requests: int) -> Path:
    partials = []
    if staging.exists():
        for part in sorted(staging.glob("*.part")):
            state_path = Path(str(part) + ".json")
            state = None
            if state_path.is_file():
                try:
                    state = _read_json(state_path)
                except (OSError, ValueError):
                    state = {"unreadable": True}
            partials.append({
                "filename": part.name.removesuffix(".part"),
                "bytes": part.stat().st_size,
                "state": state,
                "retained": True,
            })
    path = _fresh_receipt_path(
        target.parent / f"{target.name}.failure-{stamp}.json")
    payload = {
        "schema": FAILURE_SCHEMA,
        "status": "failed",
        "role": source["role"],
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "storage": source["storage"],
        "output_dir": source["output_dir"],
        "generated_at": utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "evidence": error.evidence() if hasattr(error, "evidence") else {},
        "staging_path": str(staging),
        "staging_preserved": staging.exists(),
        "partials": partials,
        "staged_bytes": _payload_bytes(staging),
        "network_requests": network_requests,
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "target_created": target.exists(),
        "target_path": str(target),
        "auto_deleted_user_evidence": False,
        "resume_requires_verified_prefix": True,
    }
    _write_json_atomic(path, payload)
    return path


def run_acquisition(plan: Mapping[str, Any], *,
                    transport: Any | None = None,
                    free_bytes_fn: Callable[[Path], int] | None = None,
                    receipt: str | os.PathLike | None = None,
                    roles: list[str] | None = None,
                    now: Callable[[], str] = utc_now) -> dict:
    root = Path(plan["root"])
    limits = plan["limits"]
    selected = [source for source in plan["sources"]
                if roles is None or source["role"] in roles]
    if not selected:
        raise PlanError("no selected source role")
    if transport is None:
        raise PlanError("a download transport is required; use --offline to validate")
    unknown = set(roles or []) - {source["role"] for source in plan["sources"]}
    if unknown:
        raise PlanError(f"unknown roles: {sorted(unknown)}")
    if receipt is not None and Path(receipt).exists():
        raise SnapshotConflict(f"receipt path must be fresh: {receipt}")

    budget = RunBudget(limits["max_total_acquisition_bytes"])
    started = now()
    free_before = _free(free_bytes_fn, root)
    results: list[dict] = []
    failures: list[dict] = []
    prior_completed = 0
    for source in selected:
        target = _resolve(root, source["output_dir"])
        staging = _staging_dir(target, source["revision"])
        try:
            result = acquire_snapshot(
                plan, source, transport=transport,
                free_bytes_fn=free_bytes_fn, now=now,
                prior_completed_bytes=prior_completed, budget=budget)
            results.append(result)
            prior_completed += source["snapshot_bytes"]
        except AcquisitionError as error:
            stamp = now().replace(":", "").replace("-", "")
            path = _failure_receipt(
                target, source, error, staging=staging,
                free_before=free_before,
                free_after=_free(free_bytes_fn, target.parent),
                stamp=stamp,
                network_requests=getattr(transport, "requests", 0))
            failures.append({
                "role": source["role"],
                "error_type": type(error).__name__,
                "error": str(error),
                "evidence": error.evidence(),
                "failure_receipt": str(path),
            })
    free_after = _free(free_bytes_fn, root)
    status = "complete" if not failures else "failed"
    summary = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "plan_path": plan["plan_path"],
        "plan_sha256": plan["plan_sha256"],
        "started_at": started,
        "finished_at": now(),
        "sources": results,
        "failures": failures,
        "weights_downloaded_this_run": any(
            result.get("bytes_downloaded", 0) > 0 for result in results),
        "model_executed": False,
        "remote_code_executed": False,
        "budget": {
            **plan["budget"],
            "bytes_downloaded": budget.downloaded,
            "budget_respected": budget.downloaded <= budget.limit,
            "staging_bytes": sum(result.get("staged_bytes", 0) for result in results),
            "final_bytes": sum(result.get("final_bytes", 0) for result in results),
            "high_water_estimate_bytes": max(
                (result.get("high_water_estimate_bytes", 0) for result in results),
                default=0),
        },
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "data_access": plan.get("data_access") or {},
    }
    if receipt is not None:
        receipt_path = Path(receipt)
        if receipt_path.exists():
            raise SnapshotConflict(f"receipt path must be fresh: {receipt_path}")
        _write_json_atomic(receipt_path, summary)
        summary["receipt_path"] = str(receipt_path)
    return summary


def validate_completed(plan: Mapping[str, Any], *,
                       roles: list[str] | None = None) -> dict:
    selected = [source for source in plan["sources"]
                if roles is None or source["role"] in roles]
    results = []
    failures = []
    for source in selected:
        try:
            results.append(verify_completed_snapshot(plan, source))
        except SnapshotConflict as error:
            failures.append({"role": source["role"], "error": str(error)})
    return {
        "status": "complete" if not failures else "failed",
        "offline": True,
        "network_requests": 0,
        "sources": results,
        "failures": failures,
        "model_executed": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=ROOT / DEFAULT_PLAN)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="root for both the pins and the checkpoint outputs")
    parser.add_argument("--role", action="append", default=None,
                        help="acquire only this role (repeatable)")
    parser.add_argument("--receipt", type=Path,
                        help="fresh combined receipt path")
    parser.add_argument("--offline", action="store_true",
                        help="verify completed snapshots with no network requests")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved plan, caps and budget only")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _plan_summary(plan: Mapping[str, Any]) -> dict:
    return {
        "schema": plan["schema"],
        "status": "planned",
        "plan_sha256": plan["plan_sha256"],
        "pins": plan["pins"],
        "limits": plan["limits"],
        "budget": plan["budget"],
        "sources": [{
            "role": source["role"],
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "storage": source["storage"],
            "output_dir": source["output_dir"],
            "snapshot_bytes": source["snapshot_bytes"],
            "weight_bytes": source["weight_bytes"],
            "asset_bytes": source["asset_bytes"],
            "weight_files": sum(1 for record in source["files"]
                                if record["weight_payload"]),
            "assets": sum(1 for record in source["files"]
                          if not record["weight_payload"]),
            "files": [{key: record[key] for key in
                       ("filename", "kind", "bytes", "sha256")}
                      for record in source["files"]],
        } for source in plan["sources"]],
        "network_requests": 0,
        "model_executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = load_acquisition_plan(args.plan, root=args.root)
        if args.dry_run:
            if not args.quiet:
                print(json.dumps(_plan_summary(plan), indent=2, sort_keys=True))
            return 0
        if args.offline:
            summary = validate_completed(plan, roles=args.role)
            if not args.quiet:
                print(json.dumps(summary, indent=2, sort_keys=True))
            return 0 if summary["status"] == "complete" else 2
        transport = UrllibTransport(timeout=plan["limits"]["timeout_seconds"])
        summary = run_acquisition(
            plan, transport=transport, receipt=args.receipt, roles=args.role)
        if not args.quiet:
            print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "complete" else 2
    except AcquisitionError as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "error": str(error), "evidence": error.evidence()},
                         indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
