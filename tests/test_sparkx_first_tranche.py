"""Deterministic tests for the research bundle and bounded first tranche.

Self-contained tests build temporary trees and owned synthetic bytes. One
artifact-dependent test verifies the committed manifest/receipt when the
provisioned bundle exists; it skips with the named provisioning command
otherwise. No network, model, dataset loader or native job is used.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import require_local_artifacts

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_SCRIPT = ROOT / "scripts" / "prepare_sparkx_research_bundle.py"
TRANCHE_SCRIPT = ROOT / "scripts" / "prepare_sparkx_first_tranche.py"
CONFIG = ROOT / "configs" / "sparkx_first_tranche_v1.json"
BUNDLE_MANIFEST = ROOT / "reports" / "sparkx_first_tranche_20261026" / "research_bundle_manifest.json"
TRANCHE_RECEIPT = ROOT / "reports" / "sparkx_first_tranche_20261026" / "tranche_receipt.json"
SYNTHETIC_ASSETS = ROOT / "reports" / "sparkx_first_tranche_20261026" / "synthetic_assets" / "assets.jsonl"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bundle = _load(BUNDLE_SCRIPT, "prepare_sparkx_research_bundle")
tranche = _load(TRANCHE_SCRIPT, "prepare_sparkx_first_tranche")


# ---------------------------------------------------------------------------
# bundle provisioning
# ---------------------------------------------------------------------------


def _bundle_tree(root: Path, *, git: bool = True) -> Path:
    for relative, _role in bundle.INPUT_ROLES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic bundle input: {relative}\n", encoding="utf-8")
    if git:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def test_bundle_manifest_is_small_hashed_and_relative(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path / "repo")
    manifest = bundle.build_manifest(root)
    assert manifest["format"] == "replica.sparkx_research_bundle.v1"
    assert manifest["input_count"] == len(bundle.INPUT_ROLES)
    assert manifest["clean_checkout_ready"] is True
    assert manifest["untracked_inputs"] == []
    assert manifest["total_input_bytes"] == sum(entry["bytes"] for entry in manifest["inputs"])
    assert len(manifest["bundle_digest"]) == 64
    assert all(not entry["path"].startswith("/") for entry in manifest["inputs"])
    assert all(entry["tracked_in_git"] for entry in manifest["inputs"])
    serialized = json.dumps(manifest)
    assert len(serialized) < 64 * 1024


def test_bundle_manifest_is_deterministic(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path / "repo")
    assert bundle.build_manifest(root) == bundle.build_manifest(root)


def test_bundle_cli_write_succeeds_for_historical_profile(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path / "repo")
    result = subprocess.run(
        [sys.executable, str(BUNDLE_SCRIPT), "--root", str(root), "--write"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "profile=historical" in result.stdout
    written_path = root / bundle.DEFAULT_MANIFEST
    assert written_path.is_file()
    written = json.loads(written_path.read_text(encoding="utf-8"))
    assert len(written["inputs"]) == len(bundle.INPUT_ROLES)
    report = bundle.verify_manifest(root, written)
    assert report["verified"] is True, report["failures"]


def test_bundle_rejects_unstaged_change_but_allows_staged_update(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path / "repo")
    manifest = bundle.build_manifest(root)
    relative = bundle.INPUT_ROLES[0][0]
    target = root / relative
    target.write_text("unstaged user change\n", encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="unstaged required input change"):
        bundle.build_manifest(root)
    report = bundle.verify_manifest(root, manifest)
    assert report["verified"] is False
    assert any(
        failure["reason"] == "unstaged_modification"
        for failure in report["failures"]
    )
    subprocess.run(["git", "add", "--", relative], cwd=root, check=True)
    updated = bundle.build_manifest(root)
    assert updated["clean_checkout_ready"] is True
    assert updated["unstaged_modified_inputs"] == []
    assert updated["bundle_digest"] != manifest["bundle_digest"]


def test_bundle_verify_detects_hash_tamper(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path / "repo")
    manifest = bundle.build_manifest(root)
    (root / bundle.INPUT_ROLES[0][0]).write_text("tampered\n", encoding="utf-8")
    report = bundle.verify_manifest(root, manifest)
    assert report["verified"] is False
    assert any(
        failure["reason"] in {"sha256_mismatch", "size_mismatch"}
        for failure in report["failures"]
    )


def test_bundle_verify_detects_missing_input(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path / "repo")
    manifest = bundle.build_manifest(root)
    (root / bundle.INPUT_ROLES[0][0]).unlink()
    report = bundle.verify_manifest(root, manifest)
    assert report["verified"] is False
    assert any(failure["reason"] == "missing" for failure in report["failures"])


def test_bundle_refuses_untracked_inputs_by_default(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path / "repo", git=False)
    with pytest.raises(bundle.BundleError, match="untracked"):
        bundle.build_manifest(root)
    manifest = bundle.build_manifest(root, allow_untracked=True)
    assert manifest["clean_checkout_ready"] is False
    assert manifest["untracked_inputs"]
    report = bundle.verify_manifest(root, manifest, allow_untracked=True)
    assert report["verified"] is True
    strict = bundle.verify_manifest(root, manifest)
    assert strict["verified"] is False
    assert any(failure["reason"] == "untracked_in_git" for failure in strict["failures"])


def test_bundle_digest_tamper_and_missing_entry_are_rejected(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path / "repo")
    manifest = bundle.build_manifest(root)

    digest_tampered = copy.deepcopy(manifest)
    digest_tampered["bundle_digest"] = "0" * 64
    report = bundle.verify_manifest(root, digest_tampered)
    assert any(failure["reason"] == "bundle_digest_mismatch" for failure in report["failures"])

    missing_entry = copy.deepcopy(manifest)
    missing_entry["inputs"] = missing_entry["inputs"][1:]
    report = bundle.verify_manifest(root, missing_entry)
    assert any(failure["reason"] == "missing_manifest_inputs" for failure in report["failures"])

    role_tampered = copy.deepcopy(manifest)
    role_tampered["inputs"][0]["role"] = "not-the-declared-role"
    report = bundle.verify_manifest(root, role_tampered)
    assert any(failure["reason"] == "role_mismatch" for failure in report["failures"])

    stale_clean_flag = copy.deepcopy(manifest)
    stale_clean_flag["clean_checkout_ready"] = False
    report = bundle.verify_manifest(root, stale_clean_flag)
    assert any(
        failure["reason"] == "manifest_not_clean_checkout_ready"
        for failure in report["failures"]
    )


def test_bundle_rejects_absolute_and_traversal_paths(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path / "repo")
    manifest = bundle.build_manifest(root)
    for bad_path in ("/etc/passwd", "../outside"):
        tampered = copy.deepcopy(manifest)
        tampered["inputs"][0]["path"] = bad_path
        report = bundle.verify_manifest(root, tampered)
        assert any(
            failure["reason"] == "non_relative_path" for failure in report["failures"]
        )


@require_local_artifacts(
    "reports/sparkx_first_tranche_20261026/research_bundle_manifest.json"
)
def test_committed_bundle_manifest_matches_local_inputs() -> None:
    manifest = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    report = bundle.verify_manifest(ROOT, manifest)
    assert report["verified"] is True, report["failures"]
    assert report["tracking_enforced"] is True
    assert manifest["clean_checkout_ready"] is True
    assert report["input_count"] == len(bundle.INPUT_ROLES)
    assert report["network_used"] is False


# ---------------------------------------------------------------------------
# first tranche protocol
# ---------------------------------------------------------------------------


def _base_config() -> dict:
    return copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))


def _asset_bytes(asset_id: str, payload: str | None = None) -> bytes:
    return (
        json.dumps(
            {
                "asset_id": asset_id,
                "synthetic": True,
                "note": "owned synthetic fixture; not external media",
                "payload": payload or asset_id,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _case(
    root: Path,
    specs: list[dict],
    *,
    min_admitted: int = 3,
    max_assets: int = 24,
    max_assets_per_source: int = 6,
    max_duplicate_rate: float = 0.2,
) -> Path:
    config = _base_config()
    config["caps"].update(
        {
            "min_admitted_assets": min_admitted,
            "max_assets": max_assets,
            "max_assets_per_source": max_assets_per_source,
            "max_duplicate_rate_per_source": max_duplicate_rate,
        }
    )
    config["synthetic_fixtures"]["asset_root"] = "fixtures/assets"
    config["synthetic_fixtures"]["asset_manifest"] = "fixtures/assets/assets.jsonl"
    config_path = root / "configs" / "sparkx_first_tranche_v1.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    asset_dir = root / "fixtures" / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        asset_id = spec["asset_id"]
        raw = spec.get("raw") or _asset_bytes(asset_id, spec.get("payload"))
        if not spec.get("missing_file"):
            (asset_dir / f"{asset_id}.json").write_bytes(raw)
        row = {
            "asset_id": asset_id,
            "source_id": spec.get("source_id", "owned_synthetic_v1"),
            "source_revision": spec.get("source_revision", "synthetic-20261026-v1"),
            "group_key": spec.get("group_key", f"group_{asset_id}"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "media_kind": spec.get("media_kind", "synthetic_scene_json"),
            "license": spec.get("license", "repository_owned_synthetic"),
            "path": f"fixtures/assets/{asset_id}.json",
            "provenance": spec.get("provenance", "owned synthetic fixture; no external media"),
            "synthetic": spec.get("synthetic", True),
        }
        if "declared_split" in spec:
            row["declared_split"] = spec["declared_split"]
        rows.append(row)
    (asset_dir / "assets.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


def _run(root: Path):
    return tranche.run(
        root=root,
        config_path="configs/sparkx_first_tranche_v1.json",
    )


def test_real_config_is_bounded_and_pinned() -> None:
    config = tranche.validate_config(json.loads(CONFIG.read_text(encoding="utf-8")))
    assert config["download_authorized"] is False
    assert config["bulk_download"] is False
    assert config["native_execution"] is False
    assert len(config["source_pins"]) == 4
    assert all(tranche.is_pinned_revision(pin["revision"]) for pin in config["source_pins"])
    assert config["synthetic_fixtures"]["label"] == "owned_synthetic_not_external_media"
    assert config["caps"]["train_basis_points"] + config["caps"]["development_basis_points"] + config["caps"]["audit_basis_points"] == 10000


def test_split_assignment_is_pure_and_covers_all_splits() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    caps = config["caps"]
    salt = config["split_assignment"]["salt"]
    groups = ["vision_group_01", "audio_group_01", "audit_av_group_03"]
    splits = [tranche.assign_split(group, salt, caps) for group in groups]
    assert splits == ["train", "development", "audit"]
    for group in groups:
        assert tranche.assign_split(group, salt, caps) == tranche.assign_split(group, salt, caps)
        digest = hashlib.sha256(f"{salt}\x00{group}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 10000
        expected = "train" if bucket < 7000 else "development" if bucket < 8500 else "audit"
        assert tranche.assign_split(group, salt, caps) == expected


def test_clean_synthetic_fixture_is_admitted_with_balanced_roles(tmp_path: Path) -> None:
    specs = [
        {"asset_id": f"scene_{index}", "group_key": "vision_group_01"}
        for index in range(2)
    ] + [
        {"asset_id": f"audio_{index}", "group_key": "audio_group_01", "media_kind": "synthetic_audio_manifest"}
        for index in range(2)
    ] + [
        {"asset_id": f"av_{index}", "group_key": "audit_av_group_03", "media_kind": "synthetic_av_manifest"}
        for index in range(2)
    ]
    root = _case(tmp_path / "repo", specs)
    admitted, rejected, receipt = _run(root)
    assert admitted and not rejected
    assert receipt["admitted_count"] == 6
    assert receipt["splits"] == {
        "train": {"assets": 2, "bytes": sum(row["bytes"] for row in admitted if row["split"] == "train")},
        "development": {"assets": 2, "bytes": sum(row["bytes"] for row in admitted if row["split"] == "development")},
        "audit": {"assets": 2, "bytes": sum(row["bytes"] for row in admitted if row["split"] == "audit")},
    }
    assert receipt["synthetic_only"] is True
    assert receipt["external_media_acquired"] is False
    assert receipt["training_admitted"] is False
    assert receipt["status"] == "admitted_synthetic_protocol_fixtures"
    assert receipt["stopped"] is False
    assert all(row["synthetic"] is True for row in admitted)
    assert all(row["license"] == "repository_owned_synthetic" for row in admitted)


def test_exact_duplicate_media_is_deduplicated(tmp_path: Path) -> None:
    raw = _asset_bytes("duplicate", "same bytes")
    specs = [
        {"asset_id": "a_original", "raw": raw, "group_key": "vision_group_01"},
        {"asset_id": "b_duplicate", "raw": raw, "group_key": "audio_group_01"},
        {"asset_id": "c_unique", "group_key": "audit_av_group_03", "media_kind": "synthetic_av_manifest"},
    ]
    root = _case(
        tmp_path / "repo", specs, min_admitted=1, max_duplicate_rate=1.0
    )
    admitted, rejected, receipt = _run(root)
    assert [row["asset_id"] for row in admitted] == ["a_original", "c_unique"]
    assert any(row["reason"] == "duplicate_exact_media" for row in rejected)
    assert receipt["dedup"]["duplicates_rejected"] == 1


def test_duplicate_rate_stops_the_source(tmp_path: Path) -> None:
    raw = _asset_bytes("duplicate", "same bytes")
    specs = [
        {"asset_id": "a_original", "raw": raw, "group_key": "vision_group_01"},
        {"asset_id": "b_duplicate", "raw": raw, "group_key": "audio_group_01"},
        {"asset_id": "c_after_stop", "group_key": "audit_av_group_03"},
    ]
    root = _case(tmp_path / "repo", specs, min_admitted=1, max_duplicate_rate=0.2)
    admitted, rejected, receipt = _run(root)
    assert len(admitted) == 1
    assert any(row["reason"] == "source_stopped" for row in rejected)
    codes = {reason["code"] for reason in receipt["stop_reasons"]}
    assert "duplicate_rate_exceeded" in codes
    assert "owned_synthetic_v1" in receipt["source_stopped"]


def test_declared_split_conflict_refuses_the_whole_tranche(tmp_path: Path) -> None:
    specs = [
        {"asset_id": "a_leak", "group_key": "audio_group_01", "declared_split": "train"},
        {"asset_id": "b_leak", "group_key": "audio_group_01", "declared_split": "train"},
    ]
    root = _case(tmp_path / "repo", specs, min_admitted=1)
    admitted, rejected, receipt = _run(root)
    assert admitted == []
    assert receipt["status"] == "refused_no_admission"
    assert receipt["admitted_count"] == 0
    assert any(reason["code"] == "global_leakage" for reason in receipt["stop_reasons"])
    assert any(row["reason"] == "declared_split_conflict" for row in rejected)


def test_missing_file_stops_source_and_prevents_admission(tmp_path: Path) -> None:
    specs = [
        {"asset_id": "a_missing", "group_key": "vision_group_01", "missing_file": True},
        {"asset_id": "b_after", "group_key": "audio_group_01"},
    ]
    root = _case(tmp_path / "repo", specs, min_admitted=1)
    admitted, rejected, receipt = _run(root)
    assert admitted == []
    assert any(row["reason"] == "missing_asset_file" for row in rejected)
    assert any(row["reason"] == "source_stopped" for row in rejected)
    assert "source_integrity_failure" in {reason["code"] for reason in receipt["stop_reasons"]}


def test_hash_mismatch_is_rejected_and_stops_source(tmp_path: Path) -> None:
    root = _case(
        tmp_path / "repo",
        [{"asset_id": "a_bad", "group_key": "vision_group_01"}],
        min_admitted=1,
    )
    manifest_path = root / "fixtures" / "assets" / "assets.jsonl"
    row = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    row["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    admitted, rejected, receipt = _run(root)
    assert admitted == []
    assert any(row["reason"] == "hash_or_size_mismatch" for row in rejected)
    assert "source_integrity_failure" in {reason["code"] for reason in receipt["stop_reasons"]}


def test_tranche_and_source_caps_stop_admission(tmp_path: Path) -> None:
    specs = [
        {"asset_id": "a", "group_key": "vision_group_01"},
        {"asset_id": "b", "group_key": "audio_group_01"},
        {"asset_id": "c", "group_key": "audit_av_group_03"},
    ]
    root = _case(tmp_path / "repo", specs, min_admitted=1, max_assets=2)
    admitted, rejected, receipt = _run(root)
    assert len(admitted) == 2
    assert any(row["reason"] == "tranche_cap_exceeded" for row in rejected)
    assert "global_cap_exceeded" in {reason["code"] for reason in receipt["stop_reasons"]}

    root2 = _case(
        tmp_path / "repo2", specs, min_admitted=1, max_assets=3, max_assets_per_source=2
    )
    admitted2, rejected2, receipt2 = _run(root2)
    assert len(admitted2) == 2
    assert any(row["reason"] == "source_cap_exceeded" for row in rejected2)


def test_uncleared_external_source_is_refused(tmp_path: Path) -> None:
    specs = [
        {
            "asset_id": "external_asset",
            "source_id": "pixmo_points",
            "source_revision": "2b5c6931e790e00ae00d4a2857e5f95d88f09a66",
            "license": "odc-by",
            "group_key": "vision_group_01",
            "synthetic": False,
        }
    ]
    root = _case(tmp_path / "repo", specs, min_admitted=1)
    admitted, rejected, receipt = _run(root)
    assert admitted == []
    assert any(row["reason"] == "source_not_acquired_or_cleared" for row in rejected)
    assert "external_media_not_acquired" in {reason["code"] for reason in receipt["stop_reasons"]}
    assert receipt["external_media_acquired"] is False


def test_configuration_rejects_bulk_or_unpinned_inputs(tmp_path: Path) -> None:
    config = _base_config()
    bulk = copy.deepcopy(config)
    bulk["bulk_download"] = True
    with pytest.raises(tranche.TrancheError):
        tranche.validate_config(bulk)

    revision = copy.deepcopy(config)
    revision["source_pins"][0]["revision"] = "main"
    with pytest.raises(tranche.TrancheError):
        tranche.validate_config(revision)

    basis = copy.deepcopy(config)
    basis["caps"]["train_basis_points"] = 8000
    with pytest.raises(tranche.TrancheError):
        tranche.validate_config(basis)

    label = copy.deepcopy(config)
    label["synthetic_fixtures"]["label"] = "external_media"
    with pytest.raises(tranche.TrancheError):
        tranche.validate_config(label)


@require_local_artifacts(
    "reports/sparkx_first_tranche_20261026/research_bundle_manifest.json"
)
def test_committed_tranche_receipt_reproduces() -> None:
    committed = json.loads(TRANCHE_RECEIPT.read_text(encoding="utf-8"))
    admitted, rejected, receipt = tranche.run(root=ROOT)
    assert receipt == committed
    assert receipt["network_used"] is False
    assert receipt["native_execution"] is False
    assert receipt["external_media_acquired"] is False
    assert len(admitted) == receipt["admitted_count"]
    assert len(rejected) == receipt["rejected_count"]
    assert all(row["synthetic"] is True for row in admitted)
    assert all(not row["path"].startswith("/") for row in admitted)
