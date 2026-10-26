#!/usr/bin/env python3
"""Verify research artifacts; never import model code or execute ML workloads.

The verifier has two explicitly separated result groups:

* document/config/link/cost checks, which need only committed text and JSON;
* source-payload checks, which read the declared content-addressed cache and
  never the historical absolute ``scratch_path``/``text_path`` provenance in
  ``evidence.json``.

Strict mode requires all third-party payloads in the cache and fails clearly
when any are absent or mismatched. ``--metadata-only`` inspects no payload and
therefore reports ``source_payload_verified=false`` with status
``unprovisioned``; it never labels a source verified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
DOCS = ROOT / "docs/research/sparkx_nested_learning_20261026"
CFG = ROOT / "configs/sparkx_2_6_4b_nested_learning_v1.json"
SOURCES_CFG = ROOT / "configs/sparkx_nested_learning_sources_v1.json"
EVIDENCE = HERE / "evidence.json"
CANON = ROOT / "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md"
PROVISION = ROOT / "scripts/provision_sparkx_nested_sources.py"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
TEXT_KEY_POLICY = "declared_text_sha256_else_source_sha256"
SOURCES_SCHEMA = "replica.sparkx_nested_learning_sources.v1"


class SourcePayloadError(RuntimeError):
    """Evidence or cache metadata is unsafe, inconsistent or outside the cache."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourcePayloadError(f"{label} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise SourcePayloadError(f"{label} must be a single safe path component: {value!r}")
    return value


def validate_sources_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema") != SOURCES_SCHEMA:
        raise SourcePayloadError(f"sources config must declare schema {SOURCES_SCHEMA}")
    count = config.get("required_source_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise SourcePayloadError("sources config required_source_count must be a positive integer")
    network = config.get("network") or {}
    if network.get("downloads_allowed") is not False:
        raise SourcePayloadError("sources config must disable network downloads")
    statuses = config.get("status") or {}
    if statuses.get("missing") != "unprovisioned" or statuses.get("mismatched") != "mismatched":
        raise SourcePayloadError("sources config must declare explicit missing/mismatched statuses")
    layout = config.get("layout") or {}
    for key in ("sources_dir", "text_dir", "staging_dir"):
        _safe_component(layout.get(key), f"layout.{key}")
    if layout.get("text_key_policy") != TEXT_KEY_POLICY:
        raise SourcePayloadError(f"layout.text_key_policy must be {TEXT_KEY_POLICY}")
    if "{sha256}" not in str(layout.get("source_filename_template", "")):
        raise SourcePayloadError("layout.source_filename_template must contain {sha256}")
    if "{text_key}" not in str(layout.get("text_filename_template", "")):
        raise SourcePayloadError("layout.text_filename_template must contain {text_key}")
    suffixes = layout.get("allowed_source_suffixes")
    if not isinstance(suffixes, list) or not suffixes or any(
        not isinstance(suffix, str) or not re.fullmatch(r"\.[a-z0-9]+", suffix) for suffix in suffixes
    ):
        raise SourcePayloadError("layout.allowed_source_suffixes must be lowercase dotted suffixes")
    provisioning = config.get("provisioning") or {}
    if provisioning.get("network") is not False:
        raise SourcePayloadError("provisioning must be offline")
    if provisioning.get("overwrite_existing") is not False:
        raise SourcePayloadError("provisioning must not overwrite existing payloads")
    for key in ("max_file_bytes", "max_total_write_bytes"):
        value = provisioning.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise SourcePayloadError(f"provisioning.{key} must be a positive integer")


def load_sources_config(path: Path = SOURCES_CFG) -> dict:
    config = load_json(path)
    validate_sources_config(config)
    return config


def resolve_cache_root(cli_value=None, config: dict | None = None, env=None, root: Path = ROOT) -> Path:
    """Resolve CLI over environment over declared repo-relative default."""
    config = config if config is not None else load_sources_config()
    root = Path(root)
    if cli_value:
        candidate = Path(cli_value).expanduser()
        return candidate if candidate.is_absolute() else (root / candidate)
    env = os.environ if env is None else env
    env_name = config.get("cache_root_env")
    if not isinstance(env_name, str) or not env_name:
        raise SourcePayloadError("sources config must declare cache_root_env")
    env_value = env.get(env_name)
    if env_value:
        candidate = Path(env_value).expanduser()
        return candidate if candidate.is_absolute() else (root / candidate)
    default = config.get("default_cache_root")
    if not isinstance(default, str) or not default:
        raise SourcePayloadError("sources config must declare default_cache_root")
    candidate = Path(default)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise SourcePayloadError("default_cache_root must be repository-relative and gitignored")
    return root / candidate


def _suffix(record: dict, config: dict) -> str:
    raw = record.get("scratch_path")
    if not isinstance(raw, str) or not raw.strip():
        raise SourcePayloadError("record must retain a historical scratch_path for provenance")
    suffix = Path(raw).suffix.lower()
    allowed = config["layout"]["allowed_source_suffixes"]
    if suffix not in allowed:
        raise SourcePayloadError(f"unsupported source suffix {suffix!r}; expected one of {allowed}")
    return suffix


def build_source_plan(evidence: dict, config: dict) -> list[dict]:
    """Validate evidence metadata and derive cache-relative payload names.

    Absolute historical paths are retained as provenance strings only. No
    filesystem access happens here, and no cache path depends on them.
    """
    validate_sources_config(config)
    records = evidence.get("records") if isinstance(evidence, dict) else None
    if not isinstance(records, list):
        raise SourcePayloadError("evidence manifest must contain a records list")
    required = config["required_source_count"]
    if len(records) != required:
        raise SourcePayloadError(f"evidence declares {len(records)} sources; config requires {required}")
    layout = config["layout"]
    entries: list[dict] = []
    names: set[str] = set()
    source_keys: set[str] = set()
    text_keys: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SourcePayloadError(f"record {index} is not an object")
        name = record.get("name")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise SourcePayloadError(f"record {index} has unsafe name {name!r}")
        if name in names:
            raise SourcePayloadError(f"duplicate source name: {name}")
        names.add(name)
        sha256 = record.get("sha256")
        if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256.lower()):
            raise SourcePayloadError(f"{name}: sha256 must be 64 lowercase hex characters")
        sha256 = sha256.lower()
        if sha256 in source_keys:
            raise SourcePayloadError(f"{name}: duplicate source sha256 {sha256}")
        source_keys.add(sha256)
        size = record.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise SourcePayloadError(f"{name}: bytes must be a positive integer")
        suffix = _suffix(record, config)
        declared_text = record.get("text_sha256")
        if declared_text is not None:
            if not isinstance(declared_text, str) or not SHA256_RE.fullmatch(declared_text.lower()):
                raise SourcePayloadError(f"{name}: text_sha256 must be 64 lowercase hex characters")
            declared_text = declared_text.lower()
        text_key = declared_text if declared_text is not None else sha256
        if text_key in text_keys:
            raise SourcePayloadError(f"{name}: duplicate text cache key {text_key}")
        text_keys.add(text_key)
        source_filename = layout["source_filename_template"].format(sha256=sha256, source_suffix=suffix)
        text_filename = layout["text_filename_template"].format(text_key=text_key)
        if not re.fullmatch(r"[0-9a-f]{64}\.[a-z0-9]+", source_filename):
            raise SourcePayloadError(f"{name}: derived source filename is unsafe: {source_filename}")
        if not re.fullmatch(r"[0-9a-f]{64}\.txt", text_filename):
            raise SourcePayloadError(f"{name}: derived text filename is unsafe: {text_filename}")
        historical_source = str(record.get("scratch_path"))
        historical_text = str(record.get("text_path") or "")
        entries.append(
            {
                "name": name,
                "sha256": sha256,
                "bytes": size,
                "source_suffix": suffix,
                "declared_text_sha256": declared_text,
                "text_key": text_key,
                "source_rel": f"{layout['sources_dir']}/{source_filename}",
                "text_rel": f"{layout['text_dir']}/{text_filename}",
                "historical_scratch_path": historical_source,
                "historical_text_path": historical_text or None,
                "historical_source_basename": Path(historical_source).name,
                "historical_text_basename": Path(historical_text).name if historical_text else None,
                "provenance_only": True,
            }
        )
    return entries


def confined_cache_path(cache_root: Path, relative: str) -> Path:
    """Return a cache-relative path, rejecting traversal, absolutes and escapes."""
    rel = Path(relative)
    if rel.is_absolute():
        raise SourcePayloadError(f"cache path must be relative: {relative}")
    if any(part in {"", ".", ".."} for part in rel.parts):
        raise SourcePayloadError(f"cache path contains an unsafe component: {relative}")
    cache_root = Path(cache_root)
    root = cache_root.resolve()
    candidate = cache_root.joinpath(*rel.parts)
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise SourcePayloadError(f"cache path escapes the declared root: {relative}")
    return candidate


def _inspect_source(entry: dict, cache_root: Path) -> dict:
    try:
        path = confined_cache_path(cache_root, entry["source_rel"])
    except SourcePayloadError as exc:
        return {"status": "mismatched", "path": None, "error": str(exc), "confinement_ok": False}
    if path.is_symlink():
        return {"status": "mismatched", "path": path, "error": "source payload is a symlink", "confinement_ok": False}
    if not path.exists():
        return {"status": "missing", "path": path, "error": None, "confinement_ok": True}
    if not path.is_file():
        return {"status": "mismatched", "path": path, "error": "source payload is not a regular file", "confinement_ok": False}
    observed_bytes = path.stat().st_size
    observed_sha = sha256_file(path)
    if observed_bytes != entry["bytes"]:
        return {
            "status": "mismatched",
            "path": path,
            "error": f"size mismatch: expected {entry['bytes']}, observed {observed_bytes}",
            "observed_bytes": observed_bytes,
            "observed_sha256": observed_sha,
            "confinement_ok": True,
        }
    if observed_sha != entry["sha256"]:
        return {
            "status": "mismatched",
            "path": path,
            "error": f"sha256 mismatch: expected {entry['sha256']}, observed {observed_sha}",
            "observed_bytes": observed_bytes,
            "observed_sha256": observed_sha,
            "confinement_ok": True,
        }
    return {
        "status": "verified",
        "path": path,
        "error": None,
        "observed_bytes": observed_bytes,
        "observed_sha256": observed_sha,
        "confinement_ok": True,
    }


def _inspect_text(entry: dict, cache_root: Path) -> dict:
    try:
        path = confined_cache_path(cache_root, entry["text_rel"])
    except SourcePayloadError as exc:
        return {"status": "mismatched", "path": None, "error": str(exc), "confinement_ok": False}
    if path.is_symlink():
        return {"status": "mismatched", "path": path, "error": "extracted text is a symlink", "confinement_ok": False}
    if not path.exists():
        return {"status": "missing", "path": path, "error": None, "confinement_ok": True}
    if not path.is_file():
        return {"status": "mismatched", "path": path, "error": "extracted text is not a regular file", "confinement_ok": False}
    observed_bytes = path.stat().st_size
    observed_sha = sha256_file(path)
    if observed_bytes <= 0:
        return {
            "status": "mismatched",
            "path": path,
            "error": "extracted text is empty",
            "observed_bytes": 0,
            "observed_sha256": observed_sha,
            "confinement_ok": True,
        }
    if entry["declared_text_sha256"] is not None and observed_sha != entry["declared_text_sha256"]:
        return {
            "status": "mismatched",
            "path": path,
            "error": (
                f"declared text sha256 mismatch: expected {entry['declared_text_sha256']}, "
                f"observed {observed_sha}"
            ),
            "observed_bytes": observed_bytes,
            "observed_sha256": observed_sha,
            "confinement_ok": True,
        }
    return {
        "status": "verified",
        "path": path,
        "error": None,
        "observed_bytes": observed_bytes,
        "observed_sha256": observed_sha,
        "confinement_ok": True,
    }


def inspect_source_cache(entries: list[dict], cache_root: Path, config: dict) -> dict:
    rows = []
    for entry in entries:
        source = _inspect_source(entry, cache_root)
        text = _inspect_text(entry, cache_root)
        if source["status"] == "mismatched" or text["status"] == "mismatched":
            status = "mismatched"
        elif source["status"] == "missing" or text["status"] == "missing":
            status = "unprovisioned"
        else:
            status = "verified"
        errors = [error for error in (source["error"], text["error"]) if error]
        rows.append(
            {
                "name": entry["name"],
                "source_rel": entry["source_rel"],
                "text_rel": entry["text_rel"],
                "source_path": str(source["path"]) if source["path"] else None,
                "text_path": str(text["path"]) if text["path"] else None,
                "expected_sha256": entry["sha256"],
                "expected_bytes": entry["bytes"],
                "expected_text_sha256": entry["declared_text_sha256"],
                "observed_sha256": source.get("observed_sha256"),
                "observed_bytes": source.get("observed_bytes"),
                "observed_text_sha256": text.get("observed_sha256"),
                "source_status": source["status"],
                "text_status": text["status"],
                "status": status,
                "confinement_ok": source["confinement_ok"] and text["confinement_ok"],
                "error": "; ".join(errors) if errors else None,
            }
        )
    mismatched = [row["name"] for row in rows if row["status"] == "mismatched"]
    unprovisioned = [row["name"] for row in rows if row["status"] == "unprovisioned"]
    verified = [row["name"] for row in rows if row["status"] == "verified"]
    if mismatched:
        status = "mismatched"
    elif unprovisioned:
        status = "unprovisioned"
    elif rows and len(verified) == len(rows):
        status = "verified"
    else:
        status = config["status"]["missing"]
    errors: list[str] = []
    for row in rows:
        if row["source_status"] == "missing":
            errors.append(f"{row['name']}: missing source payload {row['source_rel']}")
        if row["text_status"] == "missing":
            errors.append(f"{row['name']}: missing extracted text {row['text_rel']}")
        if row["error"]:
            errors.append(f"{row['name']}: {row['error']}")
    if rows and all(row["source_status"] == "missing" and row["text_status"] == "missing" for row in rows):
        errors.insert(0, f"no source payloads found under cache root: {cache_root}")
    return {
        "mode": "strict",
        "inspected": True,
        "cache_root": str(cache_root),
        "required_source_count": config["required_source_count"],
        "declared_source_count": len(rows),
        "verified_source_count": len(verified),
        "verified_sources": verified,
        "unprovisioned_sources": unprovisioned,
        "mismatched_sources": mismatched,
        "source_payload_verified": status == "verified",
        "source_payload_status": status,
        "network_used": False,
        "entries": rows,
        "errors": errors,
    }


def _payload_checks(inspection: dict, cache_root: Path) -> list[dict]:
    checks = []
    for row in inspection["entries"]:
        checks.append(
            {
                "name": "source_hash:" + row["name"],
                "pass": row["source_status"] == "verified",
                "detail": {
                    "status": row["source_status"],
                    "expected_sha256": row["expected_sha256"],
                    "observed_sha256": row["observed_sha256"],
                    "expected_bytes": row["expected_bytes"],
                    "observed_bytes": row["observed_bytes"],
                    "error": row["error"],
                },
            }
        )
        checks.append(
            {
                "name": "source_text:" + row["name"],
                "pass": row["text_status"] == "verified",
                "detail": {
                    "status": row["text_status"],
                    "expected_text_sha256": row["expected_text_sha256"],
                    "observed_text_sha256": row["observed_text_sha256"],
                    "error": row["error"],
                },
            }
        )
        checks.append(
            {
                "name": "source_confinement:" + row["name"],
                "pass": row["confinement_ok"],
                "detail": {"source_rel": row["source_rel"], "text_rel": row["text_rel"]},
            }
        )
    checks.append(
        {
            "name": "all_source_payloads_verified",
            "pass": inspection["source_payload_status"] == "verified",
            "detail": {
                "status": inspection["source_payload_status"],
                "verified": inspection["verified_source_count"],
                "required": inspection["required_source_count"],
            },
        }
    )
    pinned = next((row for row in inspection["entries"] if row["name"] == "nested_adamw_code_pinned"), None)
    if pinned is not None and pinned["status"] == "verified":
        try:
            text = confined_cache_path(cache_root, pinned["text_rel"]).read_text(encoding="utf-8", errors="replace")
        except (OSError, SourcePayloadError) as exc:
            checks.append({"name": "pinned_optimizer_rejection", "pass": False, "detail": str(exc)})
        else:
            checks.append(
                {
                    "name": "pinned_optimizer_rejection",
                    "pass": "`fused` does not support `differentiable`" in text,
                    "detail": pinned["text_rel"],
                }
            )
    else:
        checks.append(
            {
                "name": "pinned_optimizer_rejection",
                "pass": False,
                "detail": "extracted optimizer source is not verified in the declared cache",
            }
        )
    return checks


def _metadata_only_payload(entries: list[dict], cache_root: Path, config: dict) -> dict:
    names = [entry["name"] for entry in entries]
    return {
        "mode": "metadata_only",
        "inspected": False,
        "cache_root": str(cache_root),
        "required_source_count": config["required_source_count"],
        "declared_source_count": len(entries),
        "verified_source_count": 0,
        "verified_sources": [],
        "unverified_sources": names,
        "unprovisioned_sources": [],
        "mismatched_sources": [],
        "source_payload_verified": False,
        "source_payload_status": "unprovisioned",
        "network_used": False,
        "checks": [],
        "checks_total": 0,
        "checks_failed": 0,
        "errors": [],
        "note": (
            "metadata-only mode does not inspect the cache; it never labels "
            "historical or cached sources as verified"
        ),
    }


def _check_sources_config(config: dict, entries: list[dict], plan_error: str | None) -> list[dict]:
    checks = []

    def add(name, condition, detail=None):
        checks.append({"name": name, "pass": bool(condition), "detail": detail})

    add("sources_config_required_count", config.get("required_source_count") == 18,
        config.get("required_source_count"))
    add("sources_config_network_disabled", (config.get("network") or {}).get("downloads_allowed") is False)
    add("sources_config_missing_status",
        (config.get("status") or {}).get("missing") == "unprovisioned",
        (config.get("status") or {}).get("missing"))
    add("sources_config_default_cache",
        config.get("default_cache_root") == "data/cache/sparkx_nested_learning_20261026",
        config.get("default_cache_root"))
    add("sources_config_env_name", bool(config.get("cache_root_env")), config.get("cache_root_env"))
    gitignore = (ROOT / ".gitignore").read_text()
    add("sources_config_default_cache_ignored", "/data/cache/" in gitignore)
    add("source_plan_valid", plan_error is None, plan_error)
    for entry in entries:
        name = entry["name"]
        add("source_name:" + name, bool(NAME_RE.fullmatch(name)))
        add("declared_source_hash:" + name, bool(SHA256_RE.fullmatch(entry["sha256"])))
        add("declared_source_size:" + name,
            isinstance(entry["bytes"], int) and not isinstance(entry["bytes"], bool) and entry["bytes"] > 0,
            entry["bytes"])
        if entry["declared_text_sha256"] is not None:
            add("declared_text_hash:" + name, bool(SHA256_RE.fullmatch(entry["declared_text_sha256"])))
        detail = None
        try:
            for relative in (entry["source_rel"], entry["text_rel"]):
                rel = Path(relative)
                if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
                    raise SourcePayloadError(f"unsafe cache-relative path: {relative}")
            confined = True
        except SourcePayloadError as exc:
            confined = False
            detail = str(exc)
        else:
            detail = {"source": entry["source_rel"], "text": entry["text_rel"]}
        add("cache_target_confined:" + name, confined, detail)
    return checks


def _document_checks(config: dict, evidence: dict, snapshot: dict, entries: list[dict],
                     plan_error: str | None, sources_config: dict) -> list[dict]:
    checks = []

    def check(name, condition, detail=None):
        checks.append({"name": name, "pass": bool(condition), "detail": detail})

    check("design_only", config["status"] == "design_only_unrun" and not config["training_admitted"]
          and not config["native_execution"])
    check("architecture_revision",
          config["architecture_revision"] == "4.5" and "**Revision 4.5" in CANON.read_text())
    check("fixed_original_backbone", config["invariants"]["deployment_original_backbone_frozen"])
    check("full_multimodal_before_online_learning",
          config["invariants"]["multimodal_acquisition"]
          == "full_offline_finetuning_before_online_multimodal_learning")
    check("spine_separate", config["invariants"]["memory_intrinsic_to_decoder"]
          and config["invariants"]["spine_optional_and_separate"])
    bank = config["plastic_bank"]
    check("expert_partition", bank["fast_experts"] + bank["slow_experts"] == bank["experts"] == 16)
    check("two_active_experts", bank["active_read_experts_max"] == bank["active_write_experts_max"] == 2)
    opt = config["optimizer"]
    check("optimizer_policy", opt["implementation"] == "NativeFusedAdamW" and opt["dtype"] == "FP32")
    check("four_total_steps",
          opt["fast_steps_max"] + opt["slow_steps_max"] == opt["steps_total_per_transaction_max"] == 4)
    check("slow_clock_exercised",
          config["clocks"]["reference_episode_eligible_teaching_events_min"]
          >= 2 * config["clocks"]["Ps_every_eligible_teaching_events"])
    check("no_per_site_write", not config["clocks"]["one_write_per_reader_site"])
    check("gradient_limitation_explicit",
          not config["meta_training"]["fused_optimizer_exact_higher_order_supported"]
          and config["meta_training"]["P_gradient_estimator"]
          == "first_order_identity_adaptation_Jacobian")
    check("final_labels_excluded",
          not config["meta_training"]["final_test_labels_used_in_updates_or_acceptance"])
    check("moment_state_atomic", opt["weights_moments_and_counters_committed_atomically"])
    check("slot_not_identity", not config["vocab"]["optimizer_state_keyed_only_by_slot"])
    check("no_hypothetical_learning_claim", not config["validation"]["claimed_lifelong_learning"])
    for path in config["contracts"]:
        check("contract_exists:" + path, (ROOT / path).is_file())
    for row in snapshot["inputs"]:
        stored = ROOT / row["snapshot"]
        check("input_snapshot_hash:" + row["path"],
              stored.is_file() and sha256_file(stored) == row["sha256"])
    check("pre_edit_architecture_4_4",
          "**Revision 4.4" in (HERE / "architecture_v4_4_snapshot.txt").read_text())
    check("evidence_count", len(evidence["records"]) == 18)
    check("evidence_unique", len({r["name"] for r in evidence["records"]}) == 18)
    papers = [r for r in evidence["records"] if not r["name"].startswith("nested_adamw")]
    check("sixteen_papers", len(papers) == 16)
    check("nine_new_papers", sum(not r["reused_from_earlier_review"] for r in papers) == 9)
    for row in evidence["records"]:
        check("source_sections:" + row["name"], bool(row["sections_reviewed"]))
    by = {r["name"]: r for r in evidence["records"]}
    check("local_torch_pin", "torch==2.14.1" in (ROOT / "pyproject.toml").read_text())
    engram = by.get("density_engram") or {}
    check("engram_version",
          engram.get("version_in_extracted_text") == "2601.07372v2")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("research_costs", HERE / "calculate_costs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calculated = module.calculate()
    saved = load_json(HERE / "analytical_costs.json")
    check("cost_reproducible", calculated == saved)
    for name, profile in saved["profiles"].items():
        check("plastic_parameters:" + name, profile["plastic_parameters"] == 12582912)
        check("moments_96_MiB:" + name, profile["live_bytes"]["P_two_Adam_moments"] == 96 * 2**20)
        check("128_MiB_rejected:" + name, not profile["128_MiB_envelope_fits_live_Adam_state"])
        check("payload_below_candidate:" + name, profile["payload_below_256_MiB_envelope"])
    check("scalar_exact_gradient", saved["scalar_math_only"]["exact_matches_finite_difference"])
    check("scalar_approximation_differs", saved["scalar_math_only"]["first_order_is_not_exact"])
    check("cost_not_native", not saved["native_execution"] and not saved["training_admitted"])
    validation = (DOCS / "Validation_and_Open_Issues.md").read_text()
    for n in range(22):
        check("issue:N%02d" % (n + 1), "| N%02d |" % (n + 1) in validation)
    for n in range(8):
        check("stage:NL%d" % n, "### NL%d" % n in validation)
    documents = list(DOCS.glob("*.md")) + [
        HERE / "README.md", CANON,
        ROOT / "docs/architecture/22_State_Mechanism_and_RSI_Contracts.md",
        ROOT / "docs/architecture/README.md",
        ROOT / "docs/datasets/20261026_MiniCPM_Dataset_Validation.md",
        *list((ROOT / "docs/research/sparkx_intrinsic_learning_spine_20261026").glob("*.md")),
    ]
    for document in documents:
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link) or link.startswith("#"):
                continue
            target = unquote(link.strip("<>").split("#")[0])
            candidate = (document.parent / target).resolve()
            if candidate.is_relative_to((ROOT / "runs").resolve()):
                # Native run payloads are optional, ignored evidence, not clean-
                # checkout source inputs. Document/config/source checks remain strict.
                continue
            exists = target == "verification.json" and document == HERE / "README.md"
            exists = exists or candidate.exists()
            check("link:" + str(document.relative_to(ROOT)) + ":" + target, exists)
    checks.extend(_check_sources_config(sources_config, entries, plan_error))
    return checks


def _observed_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify(
    mode: str = "strict",
    cache_root=None,
    evidence_path: Path = EVIDENCE,
    sources_config_path: Path = SOURCES_CFG,
    config_path: Path = CFG,
    evidence: dict | None = None,
    sources_config: dict | None = None,
    root: Path = ROOT,
    now: str | None = None,
) -> dict:
    if mode not in {"strict", "metadata_only"}:
        raise ValueError(f"unknown verification mode: {mode}")
    sources_config = sources_config if sources_config is not None else load_sources_config(sources_config_path)
    validate_sources_config(sources_config)
    evidence = evidence if evidence is not None else load_json(evidence_path)
    config = load_json(config_path)
    snapshot = load_json(HERE / "repository_snapshot.json")
    cache_root = Path(cache_root) if cache_root is not None else resolve_cache_root(
        config=sources_config, root=root
    )
    plan_error = None
    try:
        entries = build_source_plan(evidence, sources_config)
    except SourcePayloadError as exc:
        entries = []
        plan_error = str(exc)

    if mode == "strict" and plan_error is None:
        inspection = inspect_source_cache(entries, cache_root, sources_config)
        payload_checks = _payload_checks(inspection, cache_root)
        inspection["checks"] = payload_checks
        inspection["checks_total"] = len(payload_checks)
        inspection["checks_failed"] = sum(not check["pass"] for check in payload_checks)
        payload = inspection
    elif mode == "strict":
        payload = {
            "mode": "strict",
            "inspected": False,
            "cache_root": str(cache_root),
            "required_source_count": sources_config["required_source_count"],
            "declared_source_count": len(entries),
            "verified_source_count": 0,
            "verified_sources": [],
            "unprovisioned_sources": [],
            "mismatched_sources": [],
            "source_payload_verified": False,
            "source_payload_status": "invalid",
            "network_used": False,
            "entries": [],
            "checks": [],
            "checks_total": 0,
            "checks_failed": 0,
            "errors": [plan_error],
        }
    else:
        payload = _metadata_only_payload(entries, cache_root, sources_config)
        if plan_error is not None:
            payload["errors"].append(plan_error)

    document_checks = _document_checks(
        config, evidence, snapshot, entries, plan_error, sources_config
    )
    document_failed = sum(not check["pass"] for check in document_checks)
    document_verdict = "pass" if document_failed == 0 else "fail"
    payload_failed = payload.get("checks_failed", 0)
    if mode == "strict":
        passed = document_verdict == "pass" and payload["source_payload_verified"] \
            and payload_failed == 0 and payload["source_payload_status"] == "verified"
        verdict = "pass_document_source_arithmetic_checks_only" if passed else "fail"
    else:
        verdict = (
            "pass_metadata_only_document_config_arithmetic_checks_only"
            if document_verdict == "pass" else "fail"
        )
    failed_checks = [
        check for check in document_checks if not check["pass"]
    ] + [check for check in payload.get("checks", []) if not check["pass"]]
    result = {
        "kind": "research_artifact_consistency_receipt_not_neural_qualification",
        "mode": mode,
        "verified_at_utc": now or datetime.now(timezone.utc).isoformat(),
        "head_observed_at_verification": _observed_head(root),
        "network_used": False,
        "native_execution": False,
        "training_admitted": False,
        "source_payload_verified": payload["source_payload_verified"],
        "source_payload_status": payload["source_payload_status"],
        "results": {
            "document_config_link_cost": {
                "checks": document_checks,
                "checks_total": len(document_checks),
                "checks_failed": document_failed,
                "verdict": document_verdict,
            },
            "source_payload": payload,
        },
        "checks_total": len(document_checks) + payload.get("checks_total", 0),
        "checks_failed": document_failed + payload_failed,
        "failed_checks": failed_checks,
        "artifact_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted(
                set(
                    list(DOCS.glob("*.md"))
                    + [
                        HERE / "README.md", CANON, CFG, SOURCES_CFG, EVIDENCE,
                        HERE / "analytical_costs.json", HERE / "calculate_costs.py",
                        PROVISION, Path(__file__),
                    ]
                )
            )
        },
        "verdict": verdict,
    }
    return result


def _summary(result: dict) -> dict:
    payload = result["results"]["source_payload"]
    return {
        "mode": result["mode"],
        "verdict": result["verdict"],
        "source_payload_verified": result["source_payload_verified"],
        "source_payload_status": result["source_payload_status"],
        "checks_total": result["checks_total"],
        "checks_failed": result["checks_failed"],
        "document_checks_failed": result["results"]["document_config_link_cost"]["checks_failed"],
        "payload_checks_failed": payload.get("checks_failed", 0),
        "verified_source_count": payload.get("verified_source_count", 0),
        "required_source_count": payload.get("required_source_count", 0),
        "failed_checks": [check["name"] for check in result["failed_checks"]],
        "payload_errors": payload.get("errors", []),
        "network_used": result["network_used"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true",
                        help="check committed metadata only; never verify or label source payloads")
    parser.add_argument("--cache-root", help="cache root override (default: config/env)")
    parser.add_argument("--sources-config", default=str(SOURCES_CFG))
    parser.add_argument("--evidence", default=str(EVIDENCE))
    parser.add_argument("--output", help="optional path for the full JSON result")
    args = parser.parse_args(argv)
    sources_config = load_sources_config(Path(args.sources_config))
    resolved_cache_root = resolve_cache_root(args.cache_root, config=sources_config)
    result = verify(
        mode="metadata_only" if args.metadata_only else "strict",
        cache_root=resolved_cache_root,
        evidence_path=Path(args.evidence),
        sources_config=sources_config,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(_summary(result), indent=2))
    return 0 if result["verdict"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
