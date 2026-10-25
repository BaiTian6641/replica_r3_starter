"""Bounded, metadata-only preparation of the official Spark-X2.6 sources.

This module never imports torch/transformers/safetensors, never executes remote
Python and never downloads a weight payload. It pins two independent official
Hugging Face repositories, hashes their small text assets, and inspects the
safetensors header of every shard with HTTP range requests. A response that
ignores ``Range`` (HTTP 200), exceeds a byte bound or disagrees with the pinned
file size is refused before its body is used.

The generated manifest is an acquisition record, not evidence of native support.
The static capability matrix, qualification protocol and adapter ABI in it are
prospective designs; no FP8 or BF16 forward/backward has been run.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
from typing import Any, Callable, Iterable, Mapping

SCHEMA = "replica.spark_source_preparation.v1"
PLAN_SCHEMA = "replica.spark_source_preparation_plan.v1"
HEADER_BUNDLE_SCHEMA = "replica.spark_safetensors_header_bundle.v1"
DEFAULT_PLAN = Path("configs/spark_source_preparation_v1.json")
USER_AGENT = "replica-lab-spark-preparation/1.0"

# Hard ceilings are deliberately independent of the plan so a malformed plan
# cannot authorise an unbounded fetch.
HARD_MAX_HEADER_BYTES = 32 * 1024 * 1024
HARD_MAX_ASSET_BYTES = 64 * 1024 * 1024
HARD_MAX_TOTAL_FETCH_BYTES = 512 * 1024 * 1024
HARD_MAX_API_BYTES = 8 * 1024 * 1024
HARD_MAX_PER_FILE_RETAIN_BYTES = 8 * 1024 * 1024

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"})
WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")
SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9._/-]+$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")

EXPECTED_ROLES = {
    "official_fp8_adaptation_base": "fp8_compressed_tensors",
    "official_bf16_control": "bf16",
}


class PreparationError(RuntimeError):
    """Base error for a refused or invalid preparation step."""


class PlanError(PreparationError):
    """The plan is malformed, over-broad or internally inconsistent."""


class FetchRefused(PreparationError):
    """A network response was refused; no unbounded body was consumed."""

    def __init__(self, reason: str, *, url: str | None = None,
                 status: int | None = None, detail: str | None = None,
                 bytes_read: int = 0, content_range: str | None = None):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.url = url
        self.status = status
        self.detail = detail
        self.bytes_read = bytes_read
        self.content_range = content_range

    def evidence(self) -> dict:
        return {"reason": self.reason, "url": self.url, "http_status": self.status,
                "detail": self.detail, "bytes_read": self.bytes_read,
                "content_range": self.content_range}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise PlanError(f"{name} must be a lowercase SHA256")
    return value


def _require_bytes(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise PlanError(f"{name} must be a positive integer byte count")
    return value


def _safe_relative(name: Any, name_field: str = "filename") -> str:
    if not isinstance(name, str) or not name or not SAFE_RELATIVE.fullmatch(name):
        raise PlanError(f"{name_field} must be a safe relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise PlanError(f"{name_field} must be a normalised relative path")
    return name


# ---------------------------------------------------------------------------
# Plan validation


def _validate_limits(limits: Any) -> dict:
    if not isinstance(limits, Mapping):
        raise PlanError("limits must be an object")
    required = ("max_header_bytes", "max_asset_bytes", "max_total_fetch_bytes",
                "max_api_bytes", "min_free_bytes")
    result = {}
    for key in required:
        result[key] = _require_bytes(limits.get(key), f"limits.{key}")
    if result["max_header_bytes"] > HARD_MAX_HEADER_BYTES:
        raise PlanError("max_header_bytes exceeds the hard ceiling")
    if result["max_asset_bytes"] > HARD_MAX_ASSET_BYTES:
        raise PlanError("max_asset_bytes exceeds the hard ceiling")
    if result["max_total_fetch_bytes"] > HARD_MAX_TOTAL_FETCH_BYTES:
        raise PlanError("max_total_fetch_bytes exceeds the hard ceiling")
    if result["max_api_bytes"] > HARD_MAX_API_BYTES:
        raise PlanError("max_api_bytes exceeds the hard ceiling")
    for key in ("timeout_seconds",):
        value = limits.get(key)
        if type(value) is not int or not 1 <= value <= 600:
            raise PlanError(f"limits.{key} must be an integer in [1, 600]")
        result[key] = value
    return result


def _validate_topology(topology: Any) -> dict:
    if not isinstance(topology, Mapping):
        raise PlanError("topology must be an object")
    expected = {"num_hidden_layers": 36, "hidden_size": 2560,
                "intermediate_size": 10240, "num_attention_heads": 16,
                "num_key_value_heads": 4, "head_dim": 256,
                "vocab_size": 131072, "sliding_window": 512,
                "layer_pattern": ["sliding_attention", "sliding_attention",
                                  "sliding_attention", "full_attention"] * 9,
                "tie_word_embeddings": True}
    result = {}
    for key, wanted in expected.items():
        if topology.get(key) != wanted:
            raise PlanError(f"topology.{key} is not the supported Spark2_5 value")
        result[key] = wanted
    if len(result["layer_pattern"]) != result["num_hidden_layers"]:
        raise PlanError("topology.layer_pattern must describe every layer")
    if result["num_attention_heads"] % result["num_key_value_heads"]:
        raise PlanError("query heads must be divisible by KV heads")
    return result


def _validate_source(source: Any, topology: Mapping) -> dict:
    if not isinstance(source, Mapping):
        raise PlanError("each source must be an object")
    role = source.get("role")
    if role not in EXPECTED_ROLES:
        raise PlanError(f"unknown source role: {role!r}")
    storage = source.get("storage")
    if storage != EXPECTED_ROLES[role]:
        raise PlanError(f"role {role} must declare storage {EXPECTED_ROLES[role]}")
    repo_id = source.get("repo_id")
    revision = source.get("revision")
    if not isinstance(repo_id, str) or not REPO_ID.fullmatch(repo_id):
        raise PlanError("repo_id must be an explicit owner/repository")
    if not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise PlanError("revision must be a 40-character commit, never a branch")
    if source.get("license_claim") != "apache-2.0":
        raise PlanError("both official sources must retain their Apache-2.0 claim")

    assets = source.get("assets")
    if not isinstance(assets, list) or not assets:
        raise PlanError("assets must be a non-empty list")
    seen: set[str] = set()
    normalised_assets = []
    for asset in assets:
        if not isinstance(asset, Mapping):
            raise PlanError("asset entries must be objects")
        filename = _safe_relative(asset.get("filename"))
        if filename in seen:
            raise PlanError(f"duplicate asset: {filename}")
        seen.add(filename)
        if filename.lower().endswith(WEIGHT_FILE_SUFFIXES):
            raise PlanError("weight payloads cannot be ordinary assets; only bounded headers are read")
        retain = asset.get("retain", False)
        required = asset.get("required", True)
        if type(retain) is not bool or type(required) is not bool:
            raise PlanError("asset retain/required must be booleans")
        entry = {"filename": filename, "retain": retain, "required": required}
        if "expected_sha256" in asset:
            entry["expected_sha256"] = _require_sha(asset["expected_sha256"], filename)
        if "expected_bytes" in asset:
            entry["expected_bytes"] = _require_bytes(asset["expected_bytes"], filename)
            if retain and entry["expected_bytes"] > HARD_MAX_PER_FILE_RETAIN_BYTES:
                raise PlanError(f"{filename}: retained asset exceeds the per-file ceiling")
        normalised_assets.append(entry)
    for mandatory in ("config.json", "LICENSE", "chat_template.jinja",
                      "configuration_spark.py", "modeling_spark.py"):
        if mandatory not in seen:
            raise PlanError(f"source is missing mandatory asset: {mandatory}")

    weights = source.get("weight_files")
    if not isinstance(weights, list) or not weights:
        raise PlanError("weight_files must be a non-empty list")
    normalised_weights = []
    weight_names: set[str] = set()
    for item in weights:
        if not isinstance(item, Mapping):
            raise PlanError("weight_files entries must be objects")
        filename = _safe_relative(item.get("filename"))
        if not filename.endswith(".safetensors"):
            raise PlanError("only safetensors shards can be header-inspected")
        if filename in weight_names:
            raise PlanError(f"duplicate weight file: {filename}")
        weight_names.add(filename)
        entry = {"filename": filename,
                 "bytes": _require_bytes(item.get("bytes"), f"{filename}.bytes")}
        lfs = item.get("hub_lfs_sha256")
        if lfs is not None:
            entry["hub_lfs_sha256"] = _require_sha(lfs, f"{filename}.hub_lfs_sha256")
        normalised_weights.append(entry)

    index = source.get("weight_index")
    if index is not None:
        index = _safe_relative(index, "weight_index")
        if not index.endswith(".json"):
            raise PlanError("weight_index must be a JSON filename")
        if index not in seen:
            raise PlanError("weight_index must also be declared as an asset")
    if storage == "bf16" and len(normalised_weights) > 1 and index is None:
        raise PlanError("a multi-shard checkpoint requires a weight index")
    if storage == "fp8_compressed_tensors" and len(normalised_weights) != 1:
        raise PlanError("the official FP8 source is a single safely-inspected shard")

    quant = source.get("quantization")
    if storage == "fp8_compressed_tensors":
        if not isinstance(quant, Mapping):
            raise PlanError("the FP8 source must declare quantization semantics")
        if quant.get("format") != "float-quantized" or quant.get("method") != "compressed-tensors":
            raise PlanError("only compressed-tensors float-quantized FP8 is supported")
        if quant.get("weight_strategy") != "channel" or quant.get("weight_symmetric") is not True:
            raise PlanError("FP8 weights must be static symmetric channel-wise")
        if quant.get("weight_bits") != 8 or quant.get("weight_dtype") != "F8_E4M3":
            raise PlanError("FP8 weights must be 8-bit F8_E4M3")
        if not isinstance(quant.get("version"), str) or not quant["version"]:
            raise PlanError("FP8 source must declare the compressed-tensors version")
        if quant.get("activation_strategy") != "token":
            raise PlanError("FP8 activations must be dynamic per-token")
        if quant.get("activation_dynamic") is not True or quant.get("activation_symmetric") is not True:
            raise PlanError("FP8 activations must be dynamic and symmetric")
        if quant.get("activation_bits") != 8 or quant.get("activation_dtype") != "F8_E4M3":
            raise PlanError("FP8 activations must be 8-bit F8_E4M3")
        if quant.get("zero_points") is not False:
            raise PlanError("this FP8 release has no zero-point tensors")
        if quant.get("kv_cache_quantized") is not False:
            raise PlanError("this FP8 release has an unquantized KV cache")
        if quant.get("scale_dtype") != "BF16":
            raise PlanError("stored channel scales must be BF16")
        targets = quant.get("quantized_suffixes")
        if not isinstance(targets, list) or sorted(targets) != sorted(
                ["self_attn.q_k_v_proj.weight", "self_attn.out_proj.weight",
                 "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]):
            raise PlanError("unexpected FP8 quantized module set")
        if quant.get("unquantized_suffixes_required") != ["self_attn.g_proj.weight"]:
            raise PlanError("head gates must remain unquantized")
        if quant.get("dequantization_equivalent_to_bf16_release") is not False:
            raise PlanError("a dequantized FP8 derivative is never the BF16 release")
    elif quant is not None:
        raise PlanError("BF16 storage must not declare a quantization config")

    return {"role": role, "repo_id": repo_id, "revision": revision,
            "storage": storage, "license_claim": "apache-2.0",
            "assets": normalised_assets, "weight_files": normalised_weights,
            "weight_index": index,
            "quantization": dict(quant) if isinstance(quant, Mapping) else None,
            "topology": dict(topology)}


def validate_plan(plan: Any) -> dict:
    """Validate and normalise a preparation plan without touching the network."""
    if not isinstance(plan, Mapping):
        raise PlanError("plan must be an object")
    if plan.get("schema") != PLAN_SCHEMA:
        raise PlanError(f"plan schema must be {PLAN_SCHEMA}")
    topology = _validate_topology(plan.get("topology"))
    limits = _validate_limits(plan.get("limits"))
    sources = plan.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise PlanError("exactly two official sources are required")
    normalised = [_validate_source(source, topology) for source in sources]
    roles = {source["role"] for source in normalised}
    if roles != set(EXPECTED_ROLES):
        raise PlanError("plan must contain the FP8 adaptation base and the BF16 control")
    if normalised[0]["repo_id"] == normalised[1]["repo_id"]:
        raise PlanError("the BF16 control must be an independent repository, not an alias")
    if normalised[0]["revision"] == normalised[1]["revision"]:
        raise PlanError("the two sources must use distinct immutable revisions")
    lfs = {source["role"]: {item.get("hub_lfs_sha256") for item in source["weight_files"]}
           for source in normalised}
    if lfs[normalised[0]["role"]] & lfs[normalised[1]["role"]]:
        raise PlanError("the two sources must not share a weight shard digest")
    if plan.get("weights_downloaded") is not False:
        raise PlanError("the plan must declare weights_downloaded=false")
    if plan.get("remote_code_execution") is not False:
        raise PlanError("remote Python execution must be declared false")
    return {"schema": PLAN_SCHEMA, "reviewed_on": str(plan.get("reviewed_on", "")),
            "scope": str(plan.get("scope", "")), "topology": topology,
            "limits": limits, "sources": normalised,
            "environment": plan.get("environment", {}),
            "weights_downloaded": False, "remote_code_execution": False}


def load_plan(path: str | os.PathLike) -> dict:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot read plan {path}: {error}") from error
    return validate_plan(payload)


# ---------------------------------------------------------------------------
# Bounded HTTP


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    return int(status)


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return value if value is None else str(value).strip()


def _close(response: Any) -> None:
    close = getattr(response, "close", None)
    if close is not None:
        close()


def _read_at_most(response: Any, limit: int) -> bytes:
    """Read at most ``limit`` bytes, refusing to silently truncate."""
    chunks: list[bytes] = []
    total = 0
    while total < limit:
        chunk = response.read(min(65536, limit - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    data = b"".join(chunks)
    extra = b""
    if total == limit:
        extra = response.read(1)
    if extra:
        raise FetchRefused("body_exceeds_bound", bytes_read=total + len(extra),
                           detail=f"response exceeded {limit} bytes")
    return data


class BoundedHttpTransport:
    """Range-first HTTP client with a hard cumulative byte budget."""

    def __init__(self, *, total_limit: int, timeout: int = 60,
                 opener: Callable[..., Any] | None = None,
                 user_agent: str = USER_AGENT):
        if type(total_limit) is not int or total_limit <= 0:
            raise ValueError("total_limit must be positive")
        self.total_limit = total_limit
        self.timeout = timeout
        self.opener = opener
        self.user_agent = user_agent
        self.spent = 0
        self.requests = 0

    def _open(self, url: str, headers: Mapping[str, str]):
        import urllib.request
        request = urllib.request.Request(url, headers=dict(headers))
        opener = self.opener or urllib.request.urlopen
        self.requests += 1
        try:
            return opener(request, timeout=self.timeout)
        except Exception as error:  # urllib.error.HTTPError is also a response
            status = getattr(error, "status", None) or getattr(error, "code", None)
            _close(error)
            raise FetchRefused("http_error", url=url,
                               status=int(status) if status is not None else None,
                               detail=str(error)) from error

    def _reserve(self, amount: int, url: str | None = None) -> None:
        if amount < 0 or self.spent + amount > self.total_limit:
            raise FetchRefused("total_fetch_budget", url=url,
                               detail=f"{self.spent + amount} > {self.total_limit}")

    def range(self, url: str, start: int, end: int, *,
              expected_total: int, max_bytes: int) -> dict:
        """Fetch exactly [start, end]; require HTTP 206 and matching total."""
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError("invalid range")
        wanted = end - start + 1
        if wanted > max_bytes:
            raise FetchRefused("range_exceeds_bound", url=url, bytes_read=0,
                               detail=f"{wanted} > {max_bytes}")
        self._reserve(wanted, url)
        response = self._open(url, {"Range": f"bytes={start}-{end}",
                                    "Accept-Encoding": "identity",
                                    "User-Agent": self.user_agent})
        try:
            status = _response_status(response)
            content_range = _header(response, "Content-Range")
            content_length = _header(response, "Content-Length")
            etag = _header(response, "ETag")
            encoding = (_header(response, "Content-Encoding") or "identity").lower()
            if status == 200:
                # The server ignored Range. Refuse before reading the payload.
                raise FetchRefused("server_ignored_range", url=url, status=status,
                                   content_range=content_range, bytes_read=0,
                                   detail="HTTP 200 would return the full body")
            if status != 206:
                raise FetchRefused("not_partial_content", url=url, status=status,
                                   content_range=content_range, bytes_read=0,
                                   detail=f"unexpected HTTP {status}")
            if encoding not in ("", "identity"):
                raise FetchRefused("encoded_range", url=url, status=status,
                                   content_range=content_range, bytes_read=0,
                                   detail=f"Content-Encoding={encoding}")
            if content_length is None or not content_length.isdigit():
                raise FetchRefused("missing_content_length", url=url, status=status,
                                   content_range=content_range, bytes_read=0)
            if int(content_length) != wanted:
                raise FetchRefused("content_length_mismatch", url=url, status=status,
                                   content_range=content_range, bytes_read=0,
                                   detail=f"{content_length} != {wanted}")
            match = CONTENT_RANGE.fullmatch(content_range or "")
            if not match:
                raise FetchRefused("invalid_content_range", url=url, status=status,
                                   content_range=content_range, bytes_read=0)
            got_start, got_end, got_total = (int(value) for value in match.groups())
            if (got_start, got_end) != (start, end):
                raise FetchRefused("content_range_mismatch", url=url, status=status,
                                   content_range=content_range, bytes_read=0,
                                   detail=f"requested {start}-{end}")
            if got_total != expected_total:
                raise FetchRefused("total_size_mismatch", url=url, status=status,
                                   content_range=content_range, bytes_read=0,
                                   detail=f"{got_total} != {expected_total}")
            data = _read_at_most(response, wanted)
            if len(data) != wanted:
                raise FetchRefused("short_range_body", url=url, status=status,
                                   content_range=content_range, bytes_read=len(data),
                                   detail=f"{len(data)} != {wanted}")
            self.spent += len(data)
            return {"data": data, "status": status, "url": url,
                    "final_url": getattr(response, "url", None) or getattr(
                        response, "geturl", lambda: url)(),
                    "content_range": content_range,
                    "content_length": int(content_length), "etag": etag,
                    "bytes_read": len(data), "requested_range": [start, end],
                    "total_bytes": got_total}
        finally:
            _close(response)

    def get(self, url: str, *, max_bytes: int,
            expected_bytes: int | None = None) -> dict:
        """Fetch a small asset in full, bounded by content length and max_bytes."""
        self._reserve(max_bytes, url)
        response = self._open(url, {"Accept-Encoding": "identity",
                                    "User-Agent": self.user_agent})
        try:
            status = _response_status(response)
            content_range = _header(response, "Content-Range")
            content_length = _header(response, "Content-Length")
            encoding = (_header(response, "Content-Encoding") or "identity").lower()
            etag = _header(response, "ETag")
            if status == 206:
                raise FetchRefused("unexpected_partial_content", url=url, status=status,
                                   content_range=content_range, bytes_read=0)
            if status != 200:
                raise FetchRefused("not_ok", url=url, status=status,
                                   content_range=content_range, bytes_read=0,
                                   detail=f"unexpected HTTP {status}")
            if encoding not in ("", "identity"):
                raise FetchRefused("encoded_body", url=url, status=status,
                                   content_range=content_range, bytes_read=0,
                                   detail=f"Content-Encoding={encoding}")
            declared = None
            if content_length is not None:
                if not content_length.isdigit():
                    raise FetchRefused("invalid_content_length", url=url, status=status,
                                       content_range=content_range, bytes_read=0)
                declared = int(content_length)
                if declared > max_bytes:
                    raise FetchRefused("asset_exceeds_bound", url=url, status=status,
                                       content_range=content_range, bytes_read=0,
                                       detail=f"{declared} > {max_bytes}")
                if expected_bytes is not None and declared != expected_bytes:
                    raise FetchRefused("asset_size_mismatch", url=url, status=status,
                                       content_range=content_range, bytes_read=0,
                                       detail=f"{declared} != {expected_bytes}")
            data = _read_at_most(response, max_bytes)
            self._reserve(len(data), url)
            if declared is not None and len(data) != declared:
                raise FetchRefused("short_body", url=url, status=status,
                                   content_range=content_range, bytes_read=len(data),
                                   detail=f"{len(data)} != {declared}")
            if expected_bytes is not None and len(data) != expected_bytes:
                raise FetchRefused("asset_size_mismatch", url=url, status=status,
                                   content_range=content_range, bytes_read=len(data),
                                   detail=f"{len(data)} != {expected_bytes}")
            self.spent += len(data)
            return {"data": data, "status": status, "url": url,
                    "final_url": getattr(response, "url", None) or getattr(
                        response, "geturl", lambda: url)(),
                    "content_range": content_range,
                    "content_length": declared, "etag": etag,
                    "bytes_read": len(data)}
        finally:
            _close(response)


# ---------------------------------------------------------------------------
# Safetensors header parsing


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_safetensors_header(raw: bytes, *, filename: str, total_bytes: int,
                             max_header_bytes: int) -> dict:
    """Parse one header and prove its tensor table covers the shard payload."""
    if not isinstance(raw, (bytes, bytearray)):
        raise PreparationError("header bytes must be bytes")
    raw = bytes(raw)
    if not 2 <= len(raw) <= max_header_bytes:
        raise PreparationError(f"{filename}: header length {len(raw)} is out of bounds")
    _require_bytes(total_bytes, f"{filename}.total_bytes")
    try:
        text = raw.decode("utf-8")
        payload = json.loads(text, object_pairs_hook=_unique_object)
    except (_DuplicateKey, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"{filename}: invalid safetensors header JSON: {error}") from error
    if not isinstance(payload, dict):
        raise PreparationError(f"{filename}: header must be a JSON object")
    metadata = payload.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise PreparationError(f"{filename}: __metadata__ must be an object")
    if metadata is not None and any(not isinstance(k, str) or not isinstance(v, str)
                                    for k, v in metadata.items()):
        raise PreparationError(f"{filename}: __metadata__ must map strings to strings")
    if not payload:
        raise PreparationError(f"{filename}: header contains no tensors")

    data_bytes = total_bytes - 8 - len(raw)
    if data_bytes < 0:
        raise PreparationError(f"{filename}: header is larger than the shard")
    intervals: list[tuple[int, int, str]] = []
    tensors: dict[str, dict] = {}
    dtype_counts: dict[str, dict] = {}
    total_elements = 0
    total_payload = 0
    for name, value in payload.items():
        if not isinstance(name, str) or not name:
            raise PreparationError(f"{filename}: tensor names must be nonempty strings")
        if not isinstance(value, dict):
            raise PreparationError(f"{filename}: tensor {name!r} must be an object")
        extra = set(value) - {"dtype", "shape", "data_offsets"}
        if extra:
            raise PreparationError(f"{filename}: tensor {name!r} has unknown keys {sorted(extra)}")
        dtype = value.get("dtype")
        if dtype not in DTYPE_BYTES:
            raise PreparationError(f"{filename}: tensor {name!r} has unsupported dtype {dtype!r}")
        shape = value.get("shape")
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise PreparationError(f"{filename}: tensor {name!r} has an invalid shape")
        offsets = value.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)):
            raise PreparationError(f"{filename}: tensor {name!r} has invalid data_offsets")
        begin, end = offsets
        if begin < 0 or end < begin or end > data_bytes:
            raise PreparationError(f"{filename}: tensor {name!r} offsets leave the shard")
        elements = math.prod(shape) if shape else 1
        expected = elements * DTYPE_BYTES[dtype]
        if end - begin != expected:
            raise PreparationError(
                f"{filename}: tensor {name!r} has {end - begin} bytes, expected {expected}")
        intervals.append((begin, end, name))
        tensors[name] = {"dtype": dtype, "shape": list(shape),
                         "data_offsets": [begin, end], "elements": elements,
                         "bytes": end - begin}
        bucket = dtype_counts.setdefault(dtype, {"tensors": 0, "elements": 0, "bytes": 0})
        bucket["tensors"] += 1
        bucket["elements"] += elements
        bucket["bytes"] += end - begin
        total_elements += elements
        total_payload += end - begin

    intervals.sort()
    previous_end = 0
    for begin, end, name in intervals:
        if begin < previous_end:
            raise PreparationError(f"{filename}: tensor {name!r} overlaps another tensor")
        if begin > previous_end:
            raise PreparationError(f"{filename}: unclaimed byte gap before {name!r}")
        previous_end = end
    if previous_end != data_bytes:
        raise PreparationError(
            f"{filename}: tensor payload covers {previous_end} of {data_bytes} bytes")

    scale_names = [name for name in tensors if name.endswith("_scale")]
    zero_names = [name for name in tensors if "zero_point" in name]
    if any(tensors[name]["dtype"] not in FLOAT_DTYPES for name in scale_names):
        raise PreparationError(f"{filename}: scale tensors must use a float dtype")
    return {
        "filename": filename,
        "total_bytes": total_bytes,
        "header_bytes": len(raw),
        "header_sha256": sha256_bytes(raw),
        "tensor_count": len(tensors),
        "dtype_totals": dtype_counts,
        "payload_bytes": total_payload,
        "tensor_elements": total_elements,
        "metadata": metadata or {},
        "tensors": tensors,
        "scale_tensors": sorted(scale_names),
        "zero_point_tensors": sorted(zero_names),
    }


def _expected_layer_suffixes(storage: str) -> list[str]:
    base = ["input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_k_v_proj.weight", "self_attn.g_proj.weight",
            "self_attn.out_proj.weight", "mlp.gate_proj.weight",
            "mlp.up_proj.weight", "mlp.down_proj.weight"]
    if storage == "fp8_compressed_tensors":
        base += [f"{name}_scale" for name in base if name in {
            "self_attn.q_k_v_proj.weight", "self_attn.out_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}]
    return base


def validate_spark_inventory(headers: list[dict], *, topology: Mapping,
                             storage: str) -> dict:
    """Prove the header inventory matches Spark2_5 and the declared storage."""
    if not headers:
        raise PreparationError("at least one shard header is required")
    names: dict[str, str] = {}
    for header in headers:
        for name in header["tensors"]:
            if name in names:
                raise PreparationError(f"tensor {name!r} appears in multiple shards")
            names[name] = header["filename"]
    layers = topology["num_hidden_layers"]
    suffixes = _expected_layer_suffixes(storage)
    expected = {f"model.layers.{index}.{suffix}" for index in range(layers)
                for suffix in suffixes}
    expected |= {"model.embedding.weight", "model.norm.weight"}
    if storage == "fp8_compressed_tensors":
        quantized = set(topology.get("quantized_suffixes") or []) or {
            "self_attn.q_k_v_proj.weight", "self_attn.out_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
        expected |= {f"model.layers.{index}.{suffix}_scale"
                     for index in range(layers) for suffix in quantized}
    missing = sorted(expected - set(names))
    extra = sorted(set(names) - expected)
    if missing:
        raise PreparationError(f"missing expected tensors: {missing[:8]}")
    if extra:
        raise PreparationError(f"unexpected tensors: {extra[:8]}")
    if "model.lm_head.weight" in names:
        raise PreparationError("lm_head.weight is serialized; tied storage assumption is false")

    tensors = {}
    layer_map = {}
    for header in headers:
        for name, record in header["tensors"].items():
            tensors[name] = record
            if name.startswith("model.layers."):
                parts = name.split(".")
                layer_map.setdefault(int(parts[2]), []).append(name)

    hidden = topology["hidden_size"]
    if tensors["model.embedding.weight"]["shape"] != [topology["vocab_size"], hidden]:
        raise PreparationError("embedding shape differs from topology")
    if tensors["model.norm.weight"]["shape"] != [hidden]:
        raise PreparationError("final norm shape differs from topology")
    if len(layer_map) != layers:
        raise PreparationError(f"observed {len(layer_map)} layers, expected {layers}")

    quantized_weight_elements = 0
    nonquantized_elements = 0
    scale_elements = 0
    if storage == "fp8_compressed_tensors":
        for index in range(layers):
            prefix = f"model.layers.{index}."
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                         "self_attn.g_proj.weight"):
                record = tensors[prefix + name]
                if record["dtype"] != "BF16":
                    raise PreparationError(f"{prefix + name} must remain BF16")
                nonquantized_elements += record["elements"]
            for suffix in ("self_attn.q_k_v_proj.weight", "self_attn.out_proj.weight",
                           "mlp.gate_proj.weight", "mlp.up_proj.weight",
                           "mlp.down_proj.weight"):
                weight = tensors[prefix + suffix]
                scale = tensors[prefix + suffix + "_scale"]
                if weight["dtype"] != "F8_E4M3":
                    raise PreparationError(f"{prefix + suffix} must be F8_E4M3")
                if scale["dtype"] != "BF16" or scale["shape"] != [weight["shape"][0], 1]:
                    raise PreparationError(f"{prefix + suffix}_scale must be BF16 [out, 1]")
                if scale["elements"] != weight["shape"][0]:
                    raise PreparationError(f"{prefix + suffix}_scale cardinality differs")
                quantized_weight_elements += weight["elements"]
                scale_elements += scale["elements"]
        # Embedding and final norm remain BF16 and are not part of a layer block.
        nonquantized_elements += (tensors["model.embedding.weight"]["elements"]
                                  + tensors["model.norm.weight"]["elements"])
        if any(header["zero_point_tensors"] for header in headers):
            raise PreparationError("zero-point tensors contradict symmetric FP8 semantics")
    else:
        for name, record in tensors.items():
            if record["dtype"] != "BF16":
                raise PreparationError(f"BF16 control contains non-BF16 tensor {name}")
        projection_modules = {"q_k_v_proj", "out_proj", "gate_proj",
                              "up_proj", "down_proj"}

        def is_projection(name):
            return (name.endswith(".weight")
                    and name.rsplit(".", 2)[-2] in projection_modules)
        quantized_weight_elements = sum(
            record["elements"] for name, record in tensors.items()
            if is_projection(name))
        nonquantized_elements = sum(record["elements"] for name, record in tensors.items()
                                    if not is_projection(name))
        if any(header["scale_tensors"] for header in headers):
            raise PreparationError("BF16 control unexpectedly contains scale tensors")

    intermediate = topology["intermediate_size"]
    qkv_out = ((topology["num_attention_heads"] + 2 * topology["num_key_value_heads"])
               * topology["head_dim"])
    out_dim = topology["num_attention_heads"] * topology["head_dim"]
    layer_quantized = (qkv_out * hidden + out_dim * hidden
                       + 2 * intermediate * hidden + hidden * intermediate)
    # out_proj is [hidden, q_dim]; its channel scale is over hidden outputs.
    layer_scale = qkv_out + hidden + 2 * intermediate + hidden
    expected_quantized = layers * layer_quantized
    expected_nonquantized = (layers * (2 * hidden + topology["num_attention_heads"] * hidden)
                             + topology["vocab_size"] * hidden + hidden)
    expected_scale = layers * layer_scale if storage == "fp8_compressed_tensors" else 0
    if (quantized_weight_elements, nonquantized_elements, scale_elements) != (
            expected_quantized, expected_nonquantized, expected_scale):
        raise PreparationError(
            "header parameter accounting differs from the declared topology: "
            f"observed={(quantized_weight_elements, nonquantized_elements, scale_elements)} "
            f"expected={(expected_quantized, expected_nonquantized, expected_scale)}")
    total_excluding_scales = quantized_weight_elements + nonquantized_elements
    dtype_totals: dict[str, dict] = {}
    for header in headers:
        for dtype, totals in header["dtype_totals"].items():
            bucket = dtype_totals.setdefault(dtype, {"tensors": 0, "elements": 0, "bytes": 0})
            for key in bucket:
                bucket[key] += totals[key]
    return {
        "layer_count": layers,
        "tensor_count": len(tensors),
        "shard_count": len(headers),
        "dtype_totals": dtype_totals,
        "quantized_weight_elements": quantized_weight_elements,
        "unquantized_elements": nonquantized_elements,
        "scale_elements": scale_elements,
        "parameter_elements_excluding_scales": total_excluding_scales,
        "parameter_elements_including_scales": total_excluding_scales + scale_elements,
        "payload_bytes": sum(header["payload_bytes"] for header in headers),
        "file_bytes": sum(header["total_bytes"] for header in headers),
        "scale_tensor_count": sum(len(header["scale_tensors"]) for header in headers),
        "zero_point_tensor_count": sum(len(header["zero_point_tensors"]) for header in headers),
        "tied_storage": {
            "lm_head_serialized": False,
            "lm_head_alias": "model.embedding.weight",
            "basis": ["config.tie_word_embeddings=true",
                      "modeling_spark.py _tied_weights_keys=['lm_head.weight']",
                      "F.linear(hidden_states, self.model.embedding.weight)"],
        },
        "quantized_suffixes": sorted({name.split(".", 3)[3] for name in tensors
                                      if ".layers." in name and name.endswith(".weight")
                                      and tensors[name]["dtype"] == "F8_E4M3"}),
        "shards": [{"filename": header["filename"],
                    "total_bytes": header["total_bytes"],
                    "header_bytes": header["header_bytes"],
                    "header_sha256": header["header_sha256"],
                    "tensor_count": header["tensor_count"],
                    "payload_bytes": header["payload_bytes"]} for header in headers],
        "tensor_shard_map": names,
    }


# ---------------------------------------------------------------------------
# Static source review


def audit_python_source(text: str, filename: str) -> dict:
    """Statically inspect Python without importing or executing it."""
    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError as error:
        raise PreparationError(f"{filename}: remote Python does not parse: {error}") from error
    imports: set[str] = set()
    top_level_calls: list[str] = []
    risky_calls: set[str] = set()
    classes: list[dict] = []
    functions: list[str] = []
    risky_modules = {"subprocess", "socket", "requests", "urllib", "http", "ctypes",
                     "multiprocessing", "pickle", "importlib"}
    risky_names = {"eval", "exec", "compile", "__import__", "open", "system",
                   "popen", "run", "check_output", "urlopen", "load"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Name):
            if node.id in risky_names:
                risky_calls.add(node.id)
        elif isinstance(node, ast.Attribute):
            if node.attr in risky_names:
                risky_calls.add(node.attr)
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            top_level_calls.append(ast.unparse(node.value.func))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Call):
                top_level_calls.append(ast.unparse(value.func))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(node, ast.ClassDef):
                methods = [child.name for child in node.body
                           if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append({"name": node.name,
                                "bases": [ast.unparse(base) for base in node.bases],
                                "methods": methods})
            else:
                functions.append(node.name)
    root_modules = {name.split(".")[0] for name in imports if name}
    return {
        "filename": filename,
        "executed": False,
        "imports": sorted(imports),
        "root_modules": sorted(root_modules),
        "top_level_calls": top_level_calls,
        "classes": classes,
        "module_functions": sorted(functions),
        "risky_symbols": sorted(risky_calls),
        "imports_process_or_network": sorted(root_modules & risky_modules),
        "module_executes_code_on_import": bool(top_level_calls),
    }


def architecture_review(code_audits: Mapping[str, list[dict]],
                        topology: Mapping) -> dict:
    """Record the inspected spark2_5 semantics, not a new model claim."""
    fp8_audit = code_audits["official_fp8_adaptation_base"]
    bf16_audit = code_audits["official_bf16_control"]
    same_code_bytes = (fp8_audit == bf16_audit)
    return {
        "code_identical_across_sources": same_code_bytes,
        "remote_code_executed": False,
        "imports": sorted({name for audit in fp8_audit for name in audit["imports"]}),
        "process_or_network_imports": sorted({name for audit in fp8_audit
                                              for name in audit["imports_process_or_network"]}),
        "import_time_calls": sorted({call for audit in fp8_audit
                                     for call in audit["top_level_calls"]}),
        "import_time_side_effects": (
            "registration/logging only (logging.get_logger, ALL_LAYERNORM_LAYERS.append)"
            if all(set(audit["top_level_calls"]) <= {"logging.get_logger",
                                                     "ALL_LAYERNORM_LAYERS.append"}
                   for audit in fp8_audit)
            else "inspect: unexpected import-time calls"),
        "classes": sorted({(cls["name"], tuple(cls["bases"]))
                           for audit in fp8_audit for cls in audit["classes"]}),
        "decoder": {
            "layers": topology["num_hidden_layers"],
            "hidden_size": topology["hidden_size"],
            "intermediate_size": topology["intermediate_size"],
            "attention": "custom Spark2_5Attention, fused q_k_v_proj, eager matmul attention",
            "head_dim": topology["head_dim"],
            "query_heads": topology["num_attention_heads"],
            "kv_heads": topology["num_key_value_heads"],
            "kv_repeat": topology["num_attention_heads"] // topology["num_key_value_heads"],
            "attention_schedule": "three sliding_attention blocks then one full_attention block, repeated nine times",
            "sliding_window": topology["sliding_window"],
            "headwise_output_gate": {
                "enabled": True,
                "gate_projection": "self_attn.g_proj",
                "shape_per_layer": [topology["num_attention_heads"], topology["hidden_size"]],
                "activation": "sigmoid (config gate_attn_act_mode)",
                "location": "after attention-weighted values, before out_proj",
                "quantized_in_fp8": False,
            },
            "rope": {
                "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
                "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
                "positions": "cache_position (absolute sequence positions)",
            },
            "cache": {
                "implementation": "transformers.cache_utils.DynamicCache",
                "per_layer_indexed": True,
                "sliding_eviction": False,
                "note": "sliding layers receive a sliding causal mask, but DynamicCache does not evict old keys/values; do not claim constant-memory sliding cache",
            },
            "norm": "Spark2_5RMSNorm computes in float32, multiplies float32 weight, casts back",
            "mlp": "gelu(gate_proj(x)) * up_proj(x) -> down_proj",
            "residual_stream": "inputs_embeds.float(); each layer casts to its MLP gate_proj weight dtype",
            "logits": "F.linear with model.embedding.weight when tie_word_embeddings",
        },
        "generation": {
            "mixin": "transformers GenerationMixin",
            "custom_generate_override": False,
            "thinking_modes": {
                "none": {"apply_chat_template": {"enable_thinking": False}},
                "low": {"apply_chat_template": {"enable_thinking": True,
                                                "reasoning_effort": "low"}},
                "high": {"apply_chat_template": {"enable_thinking": True,
                                                 "reasoning_effort": "high"}},
            },
            "template_requires_messages": True,
            "template_defaults_thinking": True,
            "reasoning_delimiters": ["<think>", "</think>"],
            "token_budget": "caller-supplied; generation_config max_tokens is not a qualification",
        },
        "backend_dependencies": {
            "python": ">=3.12 per repository; exact upstream interpreter not qualified",
            "torch": "required; exact wheel/version not pinned by the checkpoint",
            "transformers": "4.57.x semantics required (FP8 config serialized 4.57.6, BF16 4.57.1); isolated from the project's locked MiniCPM5 environment",
            "compressed_tensors": "0.16.0 for float-quantized loading; required for the FP8 source",
            "safetensors": "required to read checkpoint storage",
            "attention_kernels": "none imported; the custom code uses eager matmul (no flash_attn/causal_conv1d/triton)",
            "apple_mps": "not mentioned by upstream; MPS FP8 kernel support is unproven",
        },
        "adapter_hazards": [
            "spark2_5 is a custom auto_map architecture; a Llama or MiniCPM5 adapter cannot load it",
            "the model hard-codes model.embedding, model.layers[i], model.norm; inject through a wrapper, not by editing upstream files",
            "DynamicCache sharing and absent sliding eviction change memory-growth assumptions",
            "no lm_head tensor is serialized; tying must be verified after load",
            "the residual stream is float32 while layer compute follows each linear dtype; mixed-precision bookkeeping is required",
            "gradient checkpointing re-enters a layer without past_key_values; cache use must be disabled during training",
        ],
    }


# ---------------------------------------------------------------------------
# Asset content observation


def observe_config(source: Mapping, config_bytes: bytes) -> dict:
    """Check the downloaded config against the plan and observed code semantics."""
    try:
        config = json.loads(config_bytes)
    except json.JSONDecodeError as error:
        raise PreparationError(f"config.json is not JSON: {error}") from error
    if not isinstance(config, dict):
        raise PreparationError("config.json must be an object")
    topology = source["topology"]
    expected = {
        "architectures": ["Spark2_5ForCausalLM"],
        "model_type": "spark2_5",
        "num_hidden_layers": topology["num_hidden_layers"],
        "hidden_size": topology["hidden_size"],
        "intermediate_size": topology["intermediate_size"],
        "num_attention_heads": topology["num_attention_heads"],
        "num_key_value_heads": topology["num_key_value_heads"],
        "head_dim": topology["head_dim"],
        "vocab_size": topology["vocab_size"],
        "sliding_window": topology["sliding_window"],
        "tie_word_embeddings": topology["tie_word_embeddings"],
        "use_cache": True,
        "hidden_act": "gelu",
        "attention_bias": False,
        "mlp_bias": False,
        "headwise_attn_output_gate": True,
        "gate_attn_act_mode": "sigmoid",
        "layer_types": topology["layer_pattern"],
        "rope_parameters": {
            "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
        },
        "auto_map": {
            "AutoConfig": "configuration_spark.Spark2_5Config",
            "AutoModel": "modeling_spark.Spark2_5Model",
            "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
        },
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise PreparationError(f"config.json {key} differs from the supported Spark2_5 value")
    if source["storage"] == "fp8_compressed_tensors":
        quant = config.get("quantization_config")
        expected_quant = source["quantization"]
        if not isinstance(quant, dict):
            raise PreparationError("FP8 config.json has no quantization_config")
        checks = {
            "quant_method": expected_quant["method"],
            "format": expected_quant["format"],
            "version": expected_quant["version"],
            "kv_cache_scheme": None,
        }
        for key, wanted in checks.items():
            if quant.get(key) != wanted:
                raise PreparationError(f"quantization_config.{key} differs from the plan")
        groups = quant.get("config_groups")
        if not isinstance(groups, dict) or len(groups) != 1:
            raise PreparationError("expected exactly one compressed-tensors group")
        group = next(iter(groups.values()))
        if group.get("format") != "float-quantized":
            raise PreparationError("quantization group is not float-quantized")
        weights = group.get("weights") or {}
        activations = group.get("input_activations") or {}
        if (weights.get("strategy"), weights.get("num_bits"), weights.get("dynamic"),
                weights.get("symmetric")) != ("channel", 8, False, True):
            raise PreparationError("static channel-wise symmetric weight quantization differs")
        if (activations.get("strategy"), activations.get("num_bits"),
                activations.get("dynamic"), activations.get("symmetric")) != (
                "token", 8, True, True):
            raise PreparationError("dynamic per-token symmetric activation quantization differs")
        ignored = set(quant.get("ignore") or [])
        required_ignored = {f"model.layers.{index}.self_attn.g_proj"
                            for index in range(topology["num_hidden_layers"])}
        required_ignored.add("lm_head")
        if ignored != required_ignored:
            raise PreparationError("FP8 ignore list is not exactly all head gates plus lm_head")
        targets = group.get("targets")
        wanted_targets = [
            "re:.*self_attn.q_k_v_proj$", "re:.*self_attn.out_proj$",
            "re:.*mlp.(gate_proj|up_proj|gate_up_proj|down_proj)$",
        ]
        if sorted(targets or []) != sorted(wanted_targets):
            raise PreparationError("FP8 target regexes differ from the observed release")
        quantization = {
            "quant_method": quant.get("quant_method"), "format": quant.get("format"),
            "version": quant.get("version"), "kv_cache_scheme": quant.get("kv_cache_scheme"),
            "group_count": len(groups), "targets": targets, "ignore": sorted(ignored),
            "weights": {key: weights.get(key) for key in (
                "strategy", "num_bits", "dynamic", "symmetric", "observer")},
            "input_activations": {key: activations.get(key) for key in (
                "strategy", "num_bits", "dynamic", "symmetric")},
        }
    elif config.get("quantization_config") is not None:
        raise PreparationError("BF16 config.json unexpectedly declares quantization")
    else:
        quantization = None
    return {
        "architectures": config["architectures"], "model_type": config["model_type"],
        "num_hidden_layers": config["num_hidden_layers"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "attention_heads": config["num_attention_heads"],
        "kv_heads": config["num_key_value_heads"],
        "head_dim": config["head_dim"],
        "vocab_size": config["vocab_size"],
        "sliding_window": config["sliding_window"],
        "layer_types": config["layer_types"],
        "rope_parameters": config["rope_parameters"],
        "tie_word_embeddings": config["tie_word_embeddings"],
        "headwise_attn_output_gate": config["headwise_attn_output_gate"],
        "gate_attn_act_mode": config["gate_attn_act_mode"],
        "max_position_embeddings": config.get("max_position_embeddings"),
        "bos_token_id": config.get("bos_token_id"),
        "eos_token_id": config.get("eos_token_id"),
        "pad_token_id": config.get("pad_token_id"),
        "rms_norm_eps": config.get("rms_norm_eps"),
        "initializer_range": config.get("initializer_range"),
        "transformers_version": config.get("transformers_version"),
        "quantization": quantization,
        "sha256": sha256_bytes(config_bytes),
    }


def observe_tokenizer(source: Mapping, assets: Mapping[str, bytes]) -> dict:
    observed = {}
    tokenizer_bytes = assets.get("tokenizer_config.json")
    if tokenizer_bytes is not None:
        try:
            tokenizer = json.loads(tokenizer_bytes)
        except json.JSONDecodeError as error:
            raise PreparationError(f"tokenizer_config.json is not JSON: {error}") from error
        if tokenizer.get("tokenizer_class") != "PreTrainedTokenizerFast":
            raise PreparationError("tokenizer must be a fast tokenizer for offset provenance")
        if tokenizer.get("model_max_length") != source["topology"]["vocab_size"]:
            raise PreparationError("tokenizer model_max_length differs from vocab_size")
        if tokenizer.get("add_bos_token") is not False or tokenizer.get("add_eos_token") is not False:
            raise PreparationError("tokenizer must not add special tokens implicitly")
        template = tokenizer.get("chat_template")
        if not isinstance(template, str) or not template:
            raise PreparationError("tokenizer_config.json has no embedded chat template")
        observed["tokenizer_class"] = tokenizer["tokenizer_class"]
        observed["model_max_length"] = tokenizer["model_max_length"]
        observed["add_bos_token"] = tokenizer["add_bos_token"]
        observed["add_eos_token"] = tokenizer["add_eos_token"]
        observed["embedded_chat_template_sha256"] = sha256_bytes(template.encode())
    template_bytes = assets.get("chat_template.jinja")
    if template_bytes is not None:
        template = template_bytes.decode("utf-8")
        markers = ["enable_thinking", "reasoning_effort", "'high'", "'low'",
                   "<think>", "</think>", "add_generation_prompt",
                   "No messages provided"]
        missing = [marker for marker in markers if marker not in template]
        if missing:
            raise PreparationError(f"chat template lacks expected thinking markers: {missing}")
        observed["chat_template"] = {
            "sha256": sha256_bytes(template_bytes),
            "supports_enable_thinking": True,
            "supports_reasoning_effort_high_low": True,
            "thinking_delimiters": ["<think>", "</think>"],
            "default_thinking": "enabled (enable_thinking defaults true)",
        }
    generation_bytes = assets.get("generation_config.json")
    if generation_bytes is not None:
        try:
            generation = json.loads(generation_bytes)
        except json.JSONDecodeError as error:
            raise PreparationError(f"generation_config.json is not JSON: {error}") from error
        observed["generation_defaults"] = {
            key: generation.get(key) for key in (
                "bos_token_id", "eos_token_id", "pad_token_id", "max_tokens",
                "temperature", "top_p", "top_k", "repetition_penalty",
                "do_sample", "transformers_version")
        }
    return observed


def cross_check_hub_metadata(api_metadata: Mapping | None, inventory: Mapping,
                             storage: str) -> dict:
    """Relate metadata-only Hub counts to header-derived tensor counts."""
    if not api_metadata:
        return {"available": False}
    reported = (api_metadata.get("safetensors") or {}).get("parameters")
    total = (api_metadata.get("safetensors") or {}).get("total")
    if not isinstance(reported, dict) or type(total) is not int:
        raise PreparationError("Hub metadata has no usable safetensors parameter counts")
    if storage == "fp8_compressed_tensors":
        wanted = {"BF16": inventory["unquantized_elements"],
                  "F8_E4M3": inventory["quantized_weight_elements"]}
    else:
        wanted = {"BF16": inventory["parameter_elements_excluding_scales"]}
    if reported != wanted:
        raise PreparationError(
            f"Hub metadata counts {reported} differ from header-derived {wanted}")
    if total != inventory["parameter_elements_excluding_scales"]:
        raise PreparationError("Hub total parameter count differs from the header inventory")
    return {
        "available": True,
        "reported_parameters": reported,
        "reported_total": total,
        "header_parameters_excluding_scales": inventory["parameter_elements_excluding_scales"],
        "header_scale_elements": inventory["scale_elements"],
        "agreement": True,
        "note": "Counts agree, but only the stored header proves dtype/shape/scale coverage.",
    }


# ---------------------------------------------------------------------------
# Prospective precision and qualification design

PRECISION_SEMANTICS = {
    "fp8_checkpoint_storage": {
        "method": "compressed-tensors 0.16.0 float-quantized",
        "weight": "static symmetric channel-wise F8_E4M3 with BF16 scale [out_channels, 1]",
        "activation": "dynamic symmetric per-token F8_E4M3, observer-free at runtime",
        "quantized_modules": ["self_attn.q_k_v_proj", "self_attn.out_proj",
                              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        "unquantized_modules": ["self_attn.g_proj", "lm_head (tied, not serialized)",
                                "embedding", "RMSNorm weights"],
        "kv_cache": "not quantized",
        "zero_points": "none",
        "top_level_dtype_note": "config dtype=bfloat16 describes remaining tensors, not the F8_E4M3 weights",
    },
    "bf16_checkpoint_storage": {
        "method": "plain safetensors",
        "weight": "BF16",
        "activation": "not quantized by the checkpoint",
        "zero_points": "not present",
        "kv_cache": "not quantized",
    },
    "separation_of_precision_concepts": {
        "checkpoint_storage": "what bytes are on disk",
        "forward_backward_compute": "what kernels actually execute; not established by storage format",
        "optimizer_state": "separate from both; Muon int8 momentum does not make weights or gradients FP8",
    },
    "no_silent_dequantization": "A dequantized FP8 derivative must be a new artifact with its own hash and identifier; it must never be loaded under the official FP8 or BF16 identity, and it does not restore precision lost during quantization.",
}


def capability_matrix() -> list[dict]:
    """Prospective matrix; no cell is measured on this host."""
    rows = [
        ("forward_inference", "Unqualified. Requires an isolated transformers 4.57.x + compressed-tensors environment and a supported FP8 kernel path. Upstream provides no MPS evidence.",
         "Blocked pending a backend that supports compressed-tensors storage on the selected device. No local BF16 execution either."),
        ("input_gradients", "Unknown. Compressed-tensors linear layers may propagate input gradients, but MPS FP8 activation-quantization autograd is unproven.",
         "Proposed. BF16 eager attention should support input gradients; must be demonstrated, not inferred."),
        ("parameter_gradients_frozen_base", "Not applicable while the quantized base is frozen. The trainable projector must still receive input gradients; no_grad around the decoder contradicts that path.",
         "Proposed for the BF16 control under the same frozen-base contract."),
        ("native_parameter_updates", "Deferred. FP8 training kernels, master-copy policy, scaling and optimizer support are absent locally.",
         "Proposed experiment only after a measured update/reload qualification on the control."),
        ("optimizer_membership", "Frozen base has no optimizer state. New projector/LoRA parameters use a declared optimizer; Muon int8 momentum is limited to audited eligible groups.",
         "Same new-module policy; base membership differs by experiment."),
        ("export_reload", "Proposed. Persist the original FP8 storage identity plus new trainable tensors; reload in a fresh process without changing format.",
         "Proposed. Persist BF16 base identity plus the same new-module schema; reject a cross-precision reload."),
        ("kv_cache_long_context", "Unqualified. DynamicCache does not evict sliding-window keys; the 512-token window is a mask, not a bounded cache. Configured 1,048,576 positions are not a measured capacity.",
         "Same structural caveat with BF16 storage and a larger byte footprint."),
        ("mlx_inference_reference", "Not admitted. Spark-MLX-LLM is a separate implementation and must be pinned and qualified; it is not PyTorch/MPS training evidence.",
         "Not applicable to BF16 training evidence."),
    ]
    return [{
        "capability": name,
        "official_fp8": {"status": fp8, "evidence": "static source/config/header inspection only"},
        "official_bf16": {"status": bf16, "evidence": "static source/config/header inspection only"},
        "measured": False,
    } for name, fp8, bf16 in rows]


def qualification_protocol() -> dict:
    return {
        "status": "prospective_not_executed",
        "preconditions": [
            "isolated environment; never upgrade the locked MiniCPM5 environment",
            "revision/digest validation against this manifest before load",
            "one exclusive native worker; no CPU fallback and no silent dequantization",
            "no claim of MPS FP8 training until every stage below has measured evidence",
        ],
        "matched_inference": {
            "sources": {
                "fp8": "XHToken/Spark-X2.6-4B-FP8@0318d6f6ab070c7b5b43052db571489b2508006c",
                "bf16": "XHToken/Spark-X2.6-4B@66e3ae7bee54e0a523d692cb8ef841f35409807b",
            },
            "prompts": "source-independent frozen prompt set shared with MiniCPM5 preparation",
            "thinking_modes": ["none", "low", "high"],
            "budgets": {"max_new_tokens": [512, 2048], "greedy": True},
            "report": ["exact token IDs", "decoded text", "reasoning-token count",
                       "stop reason", "peak device bytes", "latency"],
            "comparison": "paired per prompt; preregister task-accuracy tolerance, abstention and length budgets. Never demand bitwise equality across precision schemes.",
            "long_context": "measure a small prefix budget first; do not extrapolate to 1,048,576 configured positions",
        },
        "input_gradient_probe": {
            "setup": "freeze all original parameters; attach one new projector to hidden width 2560 with zero-initialized output projection",
            "checks": [
                "decoder is not wrapped in no_grad and a gradient reaches the projector input",
                "all new-module gradients are finite; at least one is nonzero after the first step",
                "original weight bytes and parameter versions are unchanged",
                "activation quantization remains enabled in the FP8 path; a dequantized fallback aborts the probe",
                "sliding and full blocks both appear in the traced path",
            ],
            "failure_action": "record the unsupported kernel/backend and retain preparation status",
        },
        "single_update": {
            "scope": "one declared new module group only",
            "optimizer": "AdamW float32 master for new parameters; no Muon on the frozen FP8 base",
            "seed": "fixed and recorded",
            "acceptance": "finite loss and gradients, named parameter delta, base digest unchanged, no NaN/Inf",
            "non_claim": "a successful update is not evidence of FP8 weight training",
        },
        "export_reload": {
            "artifact": "new trainable tensors plus base identity (repo, revision, storage, shard digests, quantization config hash)",
            "fresh_process": True,
            "checks": [
                "reload with the same precision semantics and no automatic format conversion",
                "logits/tokens compared under a preregistered tolerance",
                "mismatched base digest or storage format is rejected",
                "optimizer state reload is separately checked when persisted",
            ],
        },
        "dequantized_derivative_branch": {
            "status": "not created",
            "requirements": ["new artifact identifier", "new SHA256 per shard",
                             "explicit dequantization script and scale provenance",
                             "separate activation semantics disclosure",
                             "independent comparison against the official BF16 control"],
            "prohibited": "never substitute this branch under the official FP8 identity",
        },
    }


def adapter_abi_design() -> dict:
    return {
        "status": "design_only_not_implemented",
        "name": "Spark2_5Adapter",
        "purpose": "A dedicated adapter for the custom spark2_5 architecture, including frozen-base input-gradient paths and later new modules. It is not a Llama or MiniCPM5 adapter.",
        "construction": {
            "input": "sealed local snapshot + preparation manifest digest",
            "validation": ["repo/revision/config SHA256", "storage mode and quantization config SHA256",
                           "shard digests", "36-layer Spark2_5 inventory and tied lm_head"],
            "loader": "transformers AutoConfig/AutoModelForCausalLM with trust_remote_code from the sealed local source only; remote code is audited by hash and never fetched at load",
            "device_policy": "explicit device/dtype; refuse MPS if the precision path is unavailable",
        },
        "api": {
            "forward_text": "forward_text(input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False) -> logits",
            "forward_hidden": "forward_hidden(input_ids, ...) -> final hidden states for a new projector",
            "generate": "generate_text(messages, thinking='none'|'low'|'high', max_new_tokens, generation_kwargs) -> text, token_ids, reasoning_token_count",
            "reset_cache": "reset_cache() -> None",
            "snapshot_cache": "snapshot_cache() -> opaque session-owned state (must be functional and bounded)",
            "new_projector": "attach_projector(in_dim, out_dim=2560, zero_init_output=True) -> module handle",
            "memory_ports": "enable_memory_ports([11, 23]) -> candidate only; 11/23 are hypotheses to test, not accepted positions",
            "export": "export_candidate(path) -> base identity + trainable tensor index; reload_candidate(path) verifies both",
        },
        "invariants": [
            "original checkpoint bytes are never rewritten by candidate export",
            "no_grad is never placed around the frozen decoder when new modules require input gradients",
            "every forward records storage mode and rejects an implicit dequantized fallback",
            "sliding/full layer types come from config, not from an assumed alternating Llama pattern",
            "ports are disabled for native parity; enabling them is an experimental identity change",
            "cache behavior is measured; the upstream DynamicCache does not implement sliding eviction",
            "generation uses the pinned chat template and reports actual reasoning tokens",
        ],
        "non_goals": ["native execution in this preparation task", "an adapter that pretends to load Spark before backend qualification",
                      "reusing MiniCPM5 width-2048 connectors or serialized fast states"],
    }


def environment_plan(plan_source_environment: Mapping | None) -> dict:
    declared = dict(plan_source_environment or {})
    return {
        "isolation_required": True,
        "reason": "the pinned source serializes transformers 4.57.x and compressed-tensors 0.16.0; the locked MiniCPM5 environment must not be upgraded to satisfy a quickstart",
        "required": {
            "python": "3.12-compatible isolated environment",
            "torch": "exact wheel version to be pinned during S1; no local FP8 kernel claim",
            "transformers": "4.57.x (FP8 4.57.6 / BF16 4.57.1 serialized; compatibility not guaranteed)",
            "compressed_tensors": "0.16.0",
            "safetensors": "required",
            "attention_backend": "eager matmul (upstream code imports none)",
        },
        "forbidden": ["installing Spark packages into the active MiniCPM5 environment",
                      "CPU fallback while labelling a run native",
                      "auto-dequantization used to make an FP8 run appear supported"],
        "declared": declared,
        "mps_fp8_support": "unknown; no native MPS FP8 training claim is made",
    }


def unresolved_terms() -> list[str]:
    return [
        "No native forward/backward, optimizer step or reload has been executed for either source.",
        "MPS support for compressed-tensors F8_E4M3 storage and dynamic token activation quantization is unmeasured.",
        "Exact torch/transformers wheels that load both sources without silent conversions are not pinned.",
        "The upstream GitHub implementation and Spark-MLX-LLM revisions are not pinned in this task.",
        "Configured maximum positions (1,048,576) are not a measured context-capacity result.",
        "License review records the declared Apache-2.0 claim; it is not a legal opinion about weights or training use.",
        "Training-data and evaluation-corpus licenses for Spark adaptation are not resolved by this source manifest.",
        "The FP8 card names the BF16 release as parent; weight-level equivalence and quality deltas are not measured here.",
    ]


# ---------------------------------------------------------------------------
# Local header intake


def load_header_bundle(path: str | os.PathLike, plan: Mapping,
                       *, max_header_bytes: int) -> dict[str, list[dict]]:
    """Load previously captured raw headers, verifying every digest and size."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read header bundle: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != HEADER_BUNDLE_SCHEMA:
        raise PreparationError(f"header bundle schema must be {HEADER_BUNDLE_SCHEMA}")
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise PreparationError("header bundle requires a sources list")
    expected = {source["role"]: source for source in plan["sources"]}
    result: dict[str, list[dict]] = {}
    for bundle_source in sources:
        if not isinstance(bundle_source, dict):
            raise PreparationError("header bundle sources must be objects")
        role = bundle_source.get("role")
        source = expected.get(role)
        if source is None:
            raise PreparationError(f"header bundle has unknown role: {role!r}")
        if role in result:
            raise PreparationError(f"duplicate header bundle role: {role}")
        if (bundle_source.get("repo_id") != source["repo_id"]
                or bundle_source.get("revision") != source["revision"]):
            raise PreparationError(f"header bundle identity differs for {role}")
        shards = bundle_source.get("shards")
        if not isinstance(shards, list) or not shards:
            raise PreparationError(f"header bundle has no shards for {role}")
        by_name = {item["filename"]: item for item in source["weight_files"]}
        if {item.get("filename") for item in shards} != set(by_name):
            raise PreparationError(f"header bundle shard inventory differs for {role}")
        headers = []
        for shard in shards:
            filename = _safe_relative(shard.get("filename"))
            if filename not in by_name:
                raise PreparationError(f"unexpected header shard: {filename}")
            total_bytes = _require_bytes(shard.get("total_bytes"),
                                         f"{filename}.total_bytes")
            if total_bytes != by_name[filename]["bytes"]:
                raise PreparationError(f"{filename}: header bundle total_bytes differs from plan")
            if shard.get("source_url") not in (
                    None, f"https://huggingface.co/{source['repo_id']}/resolve/"
                          f"{source['revision']}/{filename}"):
                raise PreparationError(f"{filename}: header bundle URL differs from the pinned source")
            relative = _safe_relative(shard.get("header_file"), "header_file")
            header_path = (path.parent / relative).resolve()
            try:
                header_path.relative_to(path.parent.resolve())
            except ValueError as error:
                raise PreparationError(f"{filename}: header_file escapes the bundle") from error
            try:
                raw = header_path.read_bytes()
            except OSError as error:
                raise PreparationError(f"cannot read header file {header_path}: {error}") from error
            declared_sha = _require_sha(shard.get("header_sha256"), "header_sha256")
            if sha256_bytes(raw) != declared_sha:
                raise PreparationError(f"{filename}: local header SHA256 differs")
            if shard.get("header_bytes") not in (None, len(raw)):
                raise PreparationError(f"{filename}: local header length differs")
            header = parse_safetensors_header(raw, filename=filename,
                                              total_bytes=total_bytes,
                                              max_header_bytes=max_header_bytes)
            header["provenance"] = {
                "outcome": "local_header",
                "bundle": str(path),
                "bundle_sha256": sha256_bytes(path.read_bytes()),
                "header_file": relative,
                "declared_sha256": declared_sha,
            }
            headers.append(header)
        result[role] = headers
    if set(result) != set(expected):
        raise PreparationError("header bundle must cover both official sources")
    return result


# ---------------------------------------------------------------------------
# Run orchestration


def _source_url(source: Mapping, filename: str) -> str:
    return (f"https://huggingface.co/{source['repo_id']}/resolve/"
            f"{source['revision']}/{filename}")


def _api_url(source: Mapping) -> str:
    return (f"https://huggingface.co/api/models/{source['repo_id']}/revision/"
            f"{source['revision']}?blobs=true")


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _disk_check(path: Path, minimum_free: int, just_fetched: int = 0) -> None:
    usage = shutil.disk_usage(path)
    if usage.free - just_fetched < minimum_free:
        raise PreparationError("insufficient disk space under the output root")


def _fetch_asset(source: Mapping, asset: Mapping, transport: BoundedHttpTransport,
                 *, limits: Mapping, assets_from: Path | None) -> tuple[dict, bytes | None]:
    filename = asset["filename"]
    expected = asset.get("expected_bytes")
    url = _source_url(source, filename)
    if assets_from is not None:
        local = assets_from / source["role"] / filename
        if not local.is_file() or local.is_symlink():
            raise PreparationError(f"local asset is missing or a symlink: {local}")
        data = local.read_bytes()
        if len(data) > limits["max_asset_bytes"]:
            raise FetchRefused("local_asset_exceeds_bound", url=url, bytes_read=0)
        record = {"filename": filename, "url": url, "revision": source["revision"],
                  "outcome": "local_asset", "http_status": None, "bytes": len(data),
                  "sha256": sha256_bytes(data), "content_length": None,
                  "content_range": None, "etag": None, "retained": bool(asset["retain"]),
                  "network": False, "local_path": str(local)}
    else:
        response = transport.get(url, max_bytes=limits["max_asset_bytes"],
                                 expected_bytes=expected)
        data = response["data"]
        record = {"filename": filename, "url": url, "revision": source["revision"],
                  "outcome": "fetched", "http_status": response["status"],
                  "bytes": response["bytes_read"],
                  "sha256": sha256_bytes(data),
                  "content_length": response["content_length"],
                  "content_range": response["content_range"],
                  "etag": response["etag"], "final_url": response["final_url"],
                  "retained": bool(asset["retain"]), "network": True}
    if expected is not None and record["bytes"] != expected:
        raise FetchRefused("asset_size_mismatch", url=url, status=record["http_status"],
                           bytes_read=record["bytes"],
                           detail=f"{record['bytes']} != {expected}")
    if "expected_sha256" in asset and record["sha256"] != asset["expected_sha256"]:
        raise FetchRefused("asset_sha256_mismatch", url=url, status=record["http_status"],
                           bytes_read=record["bytes"],
                           detail="downloaded bytes differ from the pinned digest")
    return record, (data if asset["retain"] else None)


def _fetch_api(source: Mapping, transport: BoundedHttpTransport, *,
               limits: Mapping, output: Path) -> tuple[dict, dict]:
    url = _api_url(source)
    response = transport.get(url, max_bytes=limits["max_api_bytes"])
    data = response["data"]
    try:
        metadata = json.loads(data)
    except json.JSONDecodeError as error:
        raise PreparationError(f"Hub metadata is not JSON: {error}") from error
    if metadata.get("sha") != source["revision"]:
        raise PreparationError(
            f"Hub resolved revision {metadata.get('sha')!r}, expected {source['revision']}")
    if (metadata.get("cardData") or {}).get("license") != source["license_claim"]:
        raise PreparationError("Hub card license differs from the pinned Apache-2.0 claim")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise PreparationError("Hub metadata has no siblings inventory")
    listing = {item.get("rfilename"): item for item in siblings
               if isinstance(item, dict) and isinstance(item.get("rfilename"), str)}
    for asset in source["assets"]:
        item = listing.get(asset["filename"])
        if item is None:
            if asset["required"]:
                raise PreparationError(f"Hub listing is missing {asset['filename']}")
            continue
        if "expected_bytes" in asset and item.get("size") != asset["expected_bytes"]:
            raise PreparationError(
                f"{asset['filename']}: Hub size {item.get('size')} != pinned size")
        lfs = (item.get("lfs") or {}).get("sha256")
        if "expected_sha256" in asset and lfs is not None and lfs != asset["expected_sha256"]:
            raise PreparationError(f"{asset['filename']}: Hub LFS digest differs from pin")
    expected_weights = {item["filename"] for item in source["weight_files"]}
    listed_weights = {name for name in listing if name.endswith(".safetensors")}
    if listed_weights != expected_weights:
        raise PreparationError(
            f"Hub weight inventory differs: listed={sorted(listed_weights)}")
    for shard in source["weight_files"]:
        item = listing[shard["filename"]]
        if item.get("size") != shard["bytes"]:
            raise PreparationError(f"{shard['filename']}: Hub shard size differs")
        lfs = (item.get("lfs") or {}).get("sha256")
        if "hub_lfs_sha256" in shard and lfs != shard["hub_lfs_sha256"]:
            raise PreparationError(f"{shard['filename']}: Hub LFS digest differs")
        if lfs is None:
            raise PreparationError(f"{shard['filename']}: Hub exposes no LFS digest")
    relative = Path("metadata") / f"{source['role']}_hub.json"
    _write_new(output / relative, data)
    record = {"filename": str(relative), "url": url, "revision": source["revision"],
              "outcome": "fetched", "http_status": response["status"],
              "bytes": len(data), "sha256": sha256_bytes(data),
              "content_length": response["content_length"], "content_range": None,
              "etag": response["etag"], "retained": True, "network": True,
              "safetensors_metadata": metadata.get("safetensors"),
              "card_license": (metadata.get("cardData") or {}).get("license"),
              "last_modified": metadata.get("lastModified")}
    return metadata, record


def _fetch_headers(source: Mapping, transport: BoundedHttpTransport, *,
                   limits: Mapping, output: Path,
                   records: list[dict]) -> list[dict]:
    headers = []
    for shard in source["weight_files"]:
        filename = shard["filename"]
        url = _source_url(source, filename)
        try:
            first = transport.range(url, 0, 7, expected_total=shard["bytes"],
                                    max_bytes=8)
            if len(first["data"]) != 8:
                raise FetchRefused("short_length_prefix", url=url,
                                   bytes_read=len(first["data"]))
            length = struct.unpack("<Q", first["data"])[0]
            if not 2 <= length <= limits["max_header_bytes"]:
                raise FetchRefused("header_size_out_of_bounds", url=url,
                                   bytes_read=8,
                                   detail=f"{length} not in [2, {limits['max_header_bytes']}]")
            second = transport.range(url, 8, 8 + length - 1,
                                     expected_total=shard["bytes"],
                                     max_bytes=limits["max_header_bytes"])
            raw = second["data"]
            header = parse_safetensors_header(
                raw, filename=filename, total_bytes=shard["bytes"],
                max_header_bytes=limits["max_header_bytes"])
            header["provenance"] = {
                "outcome": "fetched_range",
                "url": url, "revision": source["revision"],
                "ranges": [
                    {"range": [0, 7], "http_status": first["status"],
                     "bytes": first["bytes_read"], "content_range": first["content_range"],
                     "etag": first["etag"], "final_url": first["final_url"]},
                    {"range": [8, 8 + length - 1], "http_status": second["status"],
                     "bytes": second["bytes_read"], "content_range": second["content_range"],
                     "etag": second["etag"], "final_url": second["final_url"]},
                ],
                "header_sha256": header["header_sha256"],
                "full_body_refused": True,
            }
            relative = Path("headers") / source["role"] / f"{filename}.header"
            _write_new(output / relative, raw)
            header["retained_path"] = str(relative)
            headers.append(header)
            records.append({"filename": filename, "url": url,
                            "revision": source["revision"], "outcome": "fetched_range",
                            "http_status": 206, "bytes": first["bytes_read"] + second["bytes_read"],
                            "sha256": header["header_sha256"],
                            "retained": True, "network": True,
                            "content_range": second["content_range"],
                            "etag": second["etag"], "kind": "safetensors_header"})
        except FetchRefused as error:
            records.append({"filename": filename, "url": url,
                            "revision": source["revision"], "outcome": "refused",
                            "http_status": error.status, "bytes": 0,
                            "sha256": None, "retained": False, "network": True,
                            "kind": "safetensors_header",
                            "refusal": error.evidence()})
            raise
    return headers


def _validate_index(index_bytes: bytes, *, filename: str,
                    headers: list[dict], inventory: Mapping) -> dict:
    try:
        index = json.loads(index_bytes)
    except json.JSONDecodeError as error:
        raise PreparationError(f"{filename}: invalid index JSON: {error}") from error
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise PreparationError(f"{filename}: index requires a weight_map")
    metadata = index.get("metadata") or {}
    observed = inventory["tensor_shard_map"]
    mapping = index["weight_map"]
    if set(mapping) != set(observed):
        raise PreparationError(f"{filename}: index tensor names differ from headers")
    for name, shard in mapping.items():
        if shard != observed[name]:
            raise PreparationError(f"{filename}: index places {name} in {shard}, header says {observed[name]}")
    if metadata.get("total_parameters") != inventory["parameter_elements_excluding_scales"]:
        raise PreparationError(f"{filename}: index total_parameters differs from headers")
    if metadata.get("total_size") != inventory["payload_bytes"]:
        raise PreparationError(f"{filename}: index total_size differs from header payload bytes")
    return {"filename": filename, "sha256": sha256_bytes(index_bytes),
            "tensor_count": len(mapping),
            "metadata": metadata,
            "shard_fingerprints": {header["filename"]: header["header_sha256"]
                                   for header in headers}}


def run_preparation(plan: Mapping, output: str | os.PathLike, *,
                    header_bundle: str | os.PathLike | None = None,
                    assets_from: str | os.PathLike | None = None,
                    offline: bool = False,
                    opener: Callable[..., Any] | None = None,
                    free_bytes: int | None = None,
                    now: str | None = None) -> Path:
    """Execute one fresh acquisition into ``output``; return the manifest path."""
    plan = validate_plan(plan)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists; choose a fresh path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    limits = plan["limits"]
    free = shutil.disk_usage(output.parent).free if free_bytes is None else free_bytes
    upper = (limits["max_total_fetch_bytes"] + 4 * limits["max_header_bytes"]
             + 16 * 1024 * 1024)
    if free - upper < limits["min_free_bytes"]:
        raise PreparationError("insufficient free disk for the bounded acquisition")
    if offline and (header_bundle is None or assets_from is None):
        raise PreparationError("offline preparation requires --header-bundle and --assets-from")
    header_map = None
    if header_bundle is not None:
        header_map = load_header_bundle(header_bundle, plan,
                                        max_header_bytes=limits["max_header_bytes"])
    transport = BoundedHttpTransport(total_limit=limits["max_total_fetch_bytes"],
                                     timeout=limits["timeout_seconds"], opener=opener)
    output.mkdir(exist_ok=False)
    sources_out = []
    fetch_records: list[dict] = []
    tensor_indices: dict[str, list[dict]] = {}
    inventories: dict[str, dict] = {}
    code_audits: dict[str, list[dict]] = {}
    try:
        for source in plan["sources"]:
            role = source["role"]
            api_metadata = None
            if not offline:
                try:
                    api_metadata, api_record = _fetch_api(
                        source, transport, limits=limits, output=output)
                except FetchRefused as error:
                    fetch_records.append({
                        "filename": f"metadata/{role}_hub.json",
                        "url": _api_url(source), "revision": source["revision"],
                        "outcome": "refused", "http_status": error.status,
                        "bytes": 0, "sha256": None, "retained": False,
                        "network": True, "kind": "hub_metadata",
                        "refusal": error.evidence(),
                    })
                    raise
                fetch_records.append(api_record)
            asset_records = []
            retained: dict[str, bytes] = {}
            for asset in source["assets"]:
                try:
                    record, data = _fetch_asset(source, asset, transport, limits=limits,
                                                assets_from=Path(assets_from)
                                                if assets_from is not None else None)
                except FetchRefused as error:
                    fetch_records.append({
                        "filename": asset["filename"],
                        "url": _source_url(source, asset["filename"]),
                        "revision": source["revision"], "outcome": "refused",
                        "http_status": error.status, "bytes": 0, "sha256": None,
                        "retained": False,
                        "network": assets_from is None and not offline,
                        "kind": "asset", "refusal": error.evidence(),
                    })
                    raise
                asset_records.append(record)
                fetch_records.append(record)
                if data is not None:
                    _disk_check(output, limits["min_free_bytes"])
                    relative = Path("assets") / role / asset["filename"]
                    _write_new(output / relative, data)
                    retained[asset["filename"]] = data
                    record["retained_path"] = str(relative)
            if header_map is not None:
                headers = header_map[role]
                for header in headers:
                    relative = Path("headers") / role / f"{header['filename']}.header"
                    # The bundle already proved the raw header digest; copy the
                    # exact sidecar into this fresh run for symmetry.
                    bundle_path = Path(header["provenance"]["bundle"])
                    sidecar = (bundle_path.parent / header["provenance"]["header_file"]).resolve()
                    raw = sidecar.read_bytes()
                    _write_new(output / relative, raw)
                    fetch_records.append({
                        "filename": header["filename"],
                        "url": _source_url(source, header["filename"]),
                        "revision": source["revision"], "outcome": "local_header",
                        "http_status": None, "bytes": len(raw),
                        "sha256": header["header_sha256"], "retained": True,
                        "network": False, "kind": "safetensors_header",
                        "bundle_sha256": header["provenance"]["bundle_sha256"],
                    })
            else:
                headers = _fetch_headers(source, transport, limits=limits,
                                         output=output, records=fetch_records)
            inventory = validate_spark_inventory(headers, topology=source["topology"],
                                                 storage=source["storage"])
            inventories[role] = inventory
            index_summary = None
            if source["weight_index"] is not None:
                index_name = source["weight_index"]
                if index_name not in retained:
                    raise PreparationError(f"retained index is missing: {index_name}")
                index_summary = _validate_index(
                    retained[index_name], filename=index_name,
                    headers=headers, inventory=inventory)
                _write_new(output / "metadata" / f"{role}_index_inventory.json",
                           (json.dumps(index_summary, indent=2, sort_keys=True) + "\n").encode())
                index_summary["retained_path"] = f"metadata/{role}_index_inventory.json"
            index_payload = [
                {"name": name, "dtype": tensor["dtype"], "shape": tensor["shape"],
                 "data_offsets": tensor["data_offsets"], "shard": inventory["tensor_shard_map"][name]}
                for header in headers for name, tensor in header["tensors"].items()
            ]
            tensor_index = {"schema": "replica.spark_tensor_index.v1",
                            "role": role, "repo_id": source["repo_id"],
                            "revision": source["revision"], "storage": source["storage"],
                            "shard_headers": [{k: h[k] for k in (
                                "filename", "total_bytes", "header_bytes", "header_sha256",
                                "tensor_count", "payload_bytes")} for h in headers],
                            "tensors": index_payload}
            index_bytes = (json.dumps(tensor_index, indent=2, sort_keys=True) + "\n").encode()
            tensor_relative = Path("headers") / role / "tensor_index.json"
            _write_new(output / tensor_relative, index_bytes)
            tensor_indices[role] = {"path": str(tensor_relative),
                                    "sha256": sha256_bytes(index_bytes),
                                    "tensor_count": len(index_payload)}

            audits = []
            for asset in source["assets"]:
                if asset["filename"].endswith(".py"):
                    text = retained[asset["filename"]].decode("utf-8")
                    audit = audit_python_source(text, asset["filename"])
                    audit["sha256"] = sha256_bytes(retained[asset["filename"]])
                    audits.append(audit)
            code_audits[role] = audits
            config_observed = observe_config(source, retained["config.json"])
            tokenizer_observed = observe_tokenizer(source, retained)
            hub_cross_check = cross_check_hub_metadata(
                api_metadata, inventories[role], source["storage"])
            source_out = {
                "role": role,
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "source_url": f"https://huggingface.co/{source['repo_id']}",
                "storage": source["storage"],
                "license_claim": source["license_claim"],
                "license_asset": "LICENSE" if "LICENSE" in retained else None,
                "assets": asset_records,
                "hub_metadata": {
                    "reported_safetensors": (api_metadata or {}).get("safetensors"),
                    "card_license": (api_metadata or {}).get("cardData", {}).get("license"),
                    "last_modified": (api_metadata or {}).get("lastModified"),
                    "cross_check": hub_cross_check,
                    "note": "Hub parameter counts are metadata; agreement with headers is checked explicitly.",
                },
                "config_observed": config_observed,
                "tokenizer_observed": tokenizer_observed,
                "weight_files": [dict(item, locally_verified=False,
                                      evidence="Hub LFS metadata only; payload never downloaded")
                                 for item in source["weight_files"]],
                "weight_index": index_summary,
                "safetensors_headers": inventories[role]["shards"],
                "tensor_inventory": {key: value for key, value in inventories[role].items()
                                     if key != "tensor_shard_map"},
                "tensor_index": tensor_indices[role],
                "code_audit": {
                    "executed": False,
                    "files": audits,
                    "shared_with_other_source": None,
                },
            }
            sources_out.append(source_out)
    except Exception:
        failure = {"schema": SCHEMA, "status": "failed",
                   "generated_at": now or utc_now(),
                   "fetch_records": fetch_records,
                   "error": "see stdout/stderr"}
        _write_new(output / "failure.json",
                   (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode())
        raise

    if not offline and transport.spent > limits["max_total_fetch_bytes"]:
        raise PreparationError("fetch budget was exceeded")
    for source_out in sources_out:
        other = [s for s in sources_out if s["role"] != source_out["role"]][0]
        source_out["code_audit"]["shared_with_other_source"] = [
            {"filename": a["filename"], "sha256": a["sha256"]}
            for a in source_out["code_audit"]["files"]
            if any(b["sha256"] == a["sha256"] and b["filename"] == a["filename"]
                   for b in other["code_audit"]["files"])
        ]
    review = architecture_review(code_audits, plan["topology"])
    review["code_identical_across_sources"] = all(
        [{"filename": a["filename"], "sha256": a["sha256"]}
         for a in source["code_audit"]["files"]]
        == [{"filename": a["filename"], "sha256": a["sha256"]}
            for a in sources_out[0]["code_audit"]["files"]]
        for source in sources_out)
    manifest = {
        "schema": SCHEMA,
        "status": "preparation_only_unqualified",
        "generated_at": now or utc_now(),
        "reviewed_on": plan["reviewed_on"],
        "scope": plan["scope"],
        "fetch_policy": {
            "weights_downloaded": False,
            "remote_code_executed": False,
            "native_execution": False,
            "network_requests": transport.requests,
            "bytes_fetched": transport.spent,
            "total_fetch_limit_bytes": limits["max_total_fetch_bytes"],
            "max_header_bytes": limits["max_header_bytes"],
            "max_asset_bytes": limits["max_asset_bytes"],
            "min_free_bytes": limits["min_free_bytes"],
            "refuses_full_body_on_range_request": True,
            "offline": offline,
        },
        "sources": sources_out,
        "provenance": {
            "schema": "replica.spark_fetch_provenance.v1",
            "records": fetch_records,
            "note": "Header records hash the raw safetensors header; weight payloads were never downloaded.",
        },
        "identity": {
            "independent_repositories": sources_out[0]["repo_id"] != sources_out[1]["repo_id"],
            "distinct_revisions": sources_out[0]["revision"] != sources_out[1]["revision"],
            "distinct_weight_digests": not bool(
                {item.get("hub_lfs_sha256") for item in sources_out[0]["weight_files"]}
                & {item.get("hub_lfs_sha256") for item in sources_out[1]["weight_files"]}),
            "shared_asset_bytes": _shared_assets(sources_out),
            "statement": "The FP8 and BF16 artifacts are separate official Hub repositories at separate immutable revisions with different weight storage. The FP8 card names the BF16 release as parent, so they are related releases, not bitwise-equivalent controls; neither is a dequantized substitute for the other.",
        },
        "architecture_review": review,
        "precision_semantics": PRECISION_SEMANTICS,
        "capability_matrix": capability_matrix(),
        "qualification_protocol": qualification_protocol(),
        "adapter_abi": adapter_abi_design(),
        "environment": environment_plan(plan["environment"]),
        "unresolved": unresolved_terms(),
    }
    manifest["manifest_fingerprint"] = fingerprint(manifest)
    manifest_path = output / "preparation_manifest.json"
    _write_new(manifest_path,
               (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())
    return manifest_path


def _shared_assets(sources: list[dict]) -> list[dict]:
    left, right = sources
    shared = []
    right_assets = {asset["filename"]: asset for asset in right["assets"]}
    for asset in left["assets"]:
        other = right_assets.get(asset["filename"])
        if other and other["sha256"] == asset["sha256"]:
            shared.append({"filename": asset["filename"], "sha256": asset["sha256"]})
    return shared


def load_manifest(path: str | os.PathLike) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise PreparationError(f"not a preparation manifest: {path}")
    if payload.get("status") != "preparation_only_unqualified":
        raise PreparationError("manifest does not declare preparation-only status")
    if payload.get("fetch_policy", {}).get("weights_downloaded") is not False:
        raise PreparationError("manifest must record that weights were not downloaded")
    claimed = payload.get("manifest_fingerprint")
    without = {key: value for key, value in payload.items() if key != "manifest_fingerprint"}
    if not isinstance(claimed, str) or claimed != fingerprint(without):
        raise PreparationError("manifest fingerprint does not match its content")
    return payload
