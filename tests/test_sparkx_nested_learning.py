"""Self-contained tests for nested-learning source cache verification.

The tests synthesize all payload bytes locally. They never use third-party
full text, the network, a model, a native job, or the historical
``/private/tmp`` scratch directory. Optional real-cache integration is left to
the artifact-owning integration suite.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "reports" / "sparkx_nested_learning_20261026" / "verify.py"
PROVISION_PATH = ROOT / "scripts" / "provision_sparkx_nested_sources.py"
SOURCES_CONFIG = ROOT / "configs" / "sparkx_nested_learning_sources_v1.json"
EVIDENCE = ROOT / "reports" / "sparkx_nested_learning_20261026" / "evidence.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify = _load("sparkx_nested_verify_tests", VERIFY_PATH)
provision = _load("sparkx_nested_provision_tests", PROVISION_PATH)


def _synthetic_bundle(tmp_path: Path) -> tuple[dict, dict, Path]:
    """Build 18 synthetic sources while preserving the real manifest names."""
    real = json.loads(EVIDENCE.read_text())
    history = tmp_path / "history"
    history.mkdir()
    records = []
    for index, record in enumerate(real["records"]):
        name = record["name"]
        suffix = Path(record["scratch_path"]).suffix
        source = f"synthetic source payload for {name} ({index})\n".encode()
        text = f"synthetic extracted text for {name}\n".encode()
        if name == "nested_adamw_code_pinned":
            text += b"`fused` does not support `differentiable`\n"
        source_path = history / f"{name}{suffix}"
        text_path = history / f"{name}.txt"
        source_path.write_bytes(source)
        text_path.write_bytes(text)
        synthetic = dict(record)
        synthetic["sha256"] = hashlib.sha256(source).hexdigest()
        synthetic["bytes"] = len(source)
        synthetic["scratch_path"] = str(source_path)
        synthetic["text_path"] = str(text_path)
        if index % 3 == 0:
            synthetic["text_sha256"] = hashlib.sha256(text).hexdigest()
        else:
            synthetic.pop("text_sha256", None)
        records.append(synthetic)
    evidence = {
        "scope": real.get("scope"),
        "records": records,
        "failed_request_notes": real.get("failed_request_notes"),
        "third_party_full_text_committed": False,
    }
    config = json.loads(SOURCES_CONFIG.read_text())
    assert len(records) == config["required_source_count"] == 18
    return evidence, config, history


def _entry(evidence: dict, config: dict, name: str) -> dict:
    return next(entry for entry in verify.build_source_plan(evidence, config) if entry["name"] == name)


def _target(cache_root: Path, relative: str) -> Path:
    return verify.confined_cache_path(cache_root, relative)


def test_sources_config_declares_offline_content_addressed_cache() -> None:
    config = verify.load_sources_config(SOURCES_CONFIG)
    assert config["schema"] == "replica.sparkx_nested_learning_sources.v1"
    assert config["required_source_count"] == 18
    assert config["default_cache_root"] == "data/cache/sparkx_nested_learning_20261026"
    assert config["cache_root_env"] == "SPARKX_NESTED_LEARNING_SOURCE_CACHE"
    assert config["network"]["downloads_allowed"] is False
    assert config["status"]["missing"] == "unprovisioned"
    assert config["layout"]["text_key_policy"] == "declared_text_sha256_else_source_sha256"
    assert config["provisioning"]["network"] is False
    assert config["provisioning"]["overwrite_existing"] is False
    assert "/data/cache/" in (ROOT / ".gitignore").read_text()
    entries = verify.build_source_plan(json.loads(EVIDENCE.read_text()), config)
    assert len(entries) == 18
    assert {entry["source_suffix"] for entry in entries} <= set(
        config["layout"]["allowed_source_suffixes"]
    )
    assert all(entry["provenance_only"] for entry in entries)


def test_cache_root_resolution_prefers_cli_then_env_then_default(tmp_path: Path) -> None:
    config = verify.load_sources_config(SOURCES_CONFIG)
    env_root = tmp_path / "env-cache"
    cli_root = tmp_path / "cli-cache"
    env = {config["cache_root_env"]: str(env_root)}
    assert verify.resolve_cache_root(None, config=config, env=env) == env_root
    assert verify.resolve_cache_root(cli_root, config=config, env=env) == cli_root
    assert verify.resolve_cache_root(None, config=config, env={}) == (
        ROOT / "data/cache/sparkx_nested_learning_20261026"
    )


def test_strict_verification_fails_unprovisioned_without_historical_fallback(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    cache = tmp_path / "empty-cache"
    assert all(Path(record["scratch_path"]).is_file() for record in evidence["records"])
    result = verify.verify(
        mode="strict", cache_root=cache, evidence=evidence, sources_config=config
    )
    assert result["verdict"] == "fail"
    assert result["source_payload_verified"] is False
    assert result["source_payload_status"] == "unprovisioned"
    payload = result["results"]["source_payload"]
    assert payload["inspected"] is True
    assert payload["verified_source_count"] == 0
    assert len(payload["unprovisioned_sources"]) == 18
    assert any("no source payloads found" in error for error in payload["errors"])
    assert result["results"]["document_config_link_cost"]["verdict"] == "pass"
    assert str(cache) not in {str(record["scratch_path"]) for record in evidence["records"]}
    assert history.is_dir()
    metadata = verify.verify(
        mode="metadata_only", cache_root=cache, evidence=evidence, sources_config=config
    )
    assert metadata["source_payload_verified"] is False
    assert metadata["source_payload_status"] == "unprovisioned"
    assert metadata["results"]["source_payload"]["verified_sources"] == []


def test_valid_cache_passes_strict_verification(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    cache = tmp_path / "cache"
    summary = provision.provision(
        cache,
        evidence=evidence,
        config=config,
        historical_scratch_root=history,
    )
    assert summary["network_used"] is False
    assert summary["verified_sources"] == 18
    assert summary["written_files"] == 36
    assert summary["write_bytes"] > 0
    entries = verify.build_source_plan(evidence, config)
    assert any(entry["declared_text_sha256"] for entry in entries)
    assert any(
        entry["declared_text_sha256"] is None and entry["text_key"] == entry["sha256"]
        for entry in entries
    )
    result = verify.verify(
        mode="strict", cache_root=cache, evidence=evidence, sources_config=config
    )
    assert result["source_payload_verified"] is True
    assert result["source_payload_status"] == "verified"
    assert result["verdict"] == "pass_document_source_arithmetic_checks_only"
    payload = result["results"]["source_payload"]
    assert payload["verified_source_count"] == 18
    assert payload["checks_failed"] == 0
    assert result["results"]["document_config_link_cost"]["verdict"] == "pass"
    assert all(check["pass"] for check in payload["checks"])


def test_metadata_only_never_labels_sources_verified(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    cache = tmp_path / "cache"
    provision.provision(cache, evidence=evidence, config=config, historical_scratch_root=history)
    result = verify.verify(
        mode="metadata_only", cache_root=cache, evidence=evidence, sources_config=config
    )
    assert result["source_payload_verified"] is False
    assert result["source_payload_status"] == "unprovisioned"
    payload = result["results"]["source_payload"]
    assert payload["inspected"] is False
    assert payload["verified_source_count"] == 0
    assert payload["verified_sources"] == []
    assert payload["checks"] == []
    assert all(not check["name"].startswith("source_hash:") for check in payload["checks"])
    assert result["results"]["document_config_link_cost"]["verdict"] == "pass"
    assert result["verdict"] == "pass_metadata_only_document_config_arithmetic_checks_only"


def test_sha256_mismatch_fails_strict_verification(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    cache = tmp_path / "cache"
    provision.provision(cache, evidence=evidence, config=config, historical_scratch_root=history)
    entry = _entry(evidence, config, "density_hope")
    target = _target(cache, entry["source_rel"])
    data = bytearray(target.read_bytes())
    data[0] ^= 0x01
    target.write_bytes(bytes(data))
    result = verify.verify(
        mode="strict", cache_root=cache, evidence=evidence, sources_config=config
    )
    assert result["verdict"] == "fail"
    assert result["source_payload_verified"] is False
    assert result["source_payload_status"] == "mismatched"
    assert "density_hope" in result["results"]["source_payload"]["mismatched_sources"]
    assert "source_hash:density_hope" in {
        check["name"] for check in result["failed_checks"]
    }


def test_declared_size_mismatch_fails_strict_verification(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    cache = tmp_path / "cache"
    provision.provision(cache, evidence=evidence, config=config, historical_scratch_root=history)
    altered = copy.deepcopy(evidence)
    for record in altered["records"]:
        if record["name"] == "density_titans":
            record["bytes"] += 1
    result = verify.verify(
        mode="strict", cache_root=cache, evidence=altered, sources_config=config
    )
    assert result["source_payload_status"] == "mismatched"
    payload = result["results"]["source_payload"]
    row = next(row for row in payload["entries"] if row["name"] == "density_titans")
    assert "size mismatch" in (row["error"] or "")
    assert "density_titans" in payload["mismatched_sources"]


def test_cache_path_confinement_rejects_traversal_and_symlink_escape(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    cache = tmp_path / "cache"
    provision.provision(cache, evidence=evidence, config=config, historical_scratch_root=history)
    with pytest.raises(verify.SourcePayloadError):
        verify.confined_cache_path(cache, "../outside.html")
    entry = _entry(evidence, config, "bridge_miras_arxiv")
    target = _target(cache, entry["source_rel"])
    outside = tmp_path / "outside.html"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    result = verify.verify(
        mode="strict", cache_root=cache, evidence=evidence, sources_config=config
    )
    assert result["source_payload_status"] == "mismatched"
    payload = result["results"]["source_payload"]
    row = next(row for row in payload["entries"] if row["name"] == "bridge_miras_arxiv")
    assert row["confinement_ok"] is False
    assert "escapes" in (row["error"] or "")
    assert f"source_confinement:{entry['name']}" in {
        check["name"] for check in result["failed_checks"]
    }


def test_provision_requires_explicit_source_root_even_when_history_exists(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    cache = tmp_path / "cache"
    assert all(Path(record["scratch_path"]).is_file() for record in evidence["records"])
    with pytest.raises(provision.ProvisionError, match="explicit local source root"):
        provision.provision(cache, evidence=evidence, config=config)
    assert not cache.exists()


def test_provision_from_declared_directory_is_offline_and_idempotent(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    cache = tmp_path / "clean-cache"
    first = provision.provision(cache, evidence=evidence, config=config, from_dir=history)
    assert first["written_files"] == 36
    assert first["network_used"] is False
    second = provision.provision(cache, evidence=evidence, config=config, from_dir=history)
    assert second["written_files"] == 0
    assert second["skipped_files"] == 36
    assert second["write_bytes"] == 0


def test_provision_never_overwrites_mismatched_cache_file(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    cache = tmp_path / "cache"
    provision.provision(cache, evidence=evidence, config=config, from_dir=history)
    entry = _entry(evidence, config, "nested_uttt")
    target = _target(cache, entry["source_rel"])
    tampered = b"tampered cache payload"
    target.write_bytes(tampered)
    with pytest.raises(provision.ProvisionError, match="refusing to overwrite"):
        provision.provision(cache, evidence=evidence, config=config, from_dir=history)
    assert target.read_bytes() == tampered


def test_provision_verifies_all_inputs_before_publication(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    bad_inputs = tmp_path / "bad-inputs"
    shutil.copytree(history, bad_inputs)
    broken = bad_inputs / "density_hope.html"
    broken.write_bytes(b"not the declared source")
    cache = tmp_path / "cache"
    with pytest.raises(provision.ProvisionError, match="mismatch"):
        provision.provision(cache, evidence=evidence, config=config, from_dir=bad_inputs)
    assert not cache.exists()


def test_provision_declares_and_enforces_write_budget(tmp_path: Path) -> None:
    evidence, config, history = _synthetic_bundle(tmp_path)
    tight = copy.deepcopy(config)
    tight["provisioning"]["max_total_write_bytes"] = 1
    cache = tmp_path / "cache"
    with pytest.raises(provision.ProvisionError, match="write budget exceeded"):
        provision.provision(cache, evidence=evidence, config=tight, from_dir=history)
    assert not cache.exists()


def test_metadata_only_cli_entrypoint_writes_explicit_unprovisioned_result(
        tmp_path: Path) -> None:
    output = tmp_path / "verification.json"
    result = verify.main([
        "--metadata-only",
        "--cache-root", str(tmp_path / "absent-cache"),
        "--output", str(output),
    ])
    assert result == 0
    receipt = json.loads(output.read_text())
    assert receipt["source_payload_verified"] is False
    assert receipt["source_payload_status"] == "unprovisioned"
    assert receipt["results"]["source_payload"]["inspected"] is False
