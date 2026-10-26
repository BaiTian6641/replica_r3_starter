"""Offline behavior tests for the SparkX dataset research verifier.

The verifier reads only local research evidence.  Every test copies the real
input package into a temporary tree, mutates one input through JSON, and runs
the verifier CLI.  No network, media asset, dataset loader or model is used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "scripts" / "verify_sparkx_dataset_research.py"
CATALOG = "docs/datasets/20261026_SparkX_Dataset_Catalog.json"
EVIDENCE = "reports/sparkx_dataset_hunt_20261026/evidence.json"
HF_METADATA = "reports/sparkx_dataset_hunt_20261026/hf_metadata.json"
REGISTRY = "configs/dataset_registry.json"
PLAN = "docs/datasets/20261026_SparkX_Dataset_Plan.md"

_spec = importlib.util.spec_from_file_location(
    "verify_sparkx_dataset_research", VERIFIER
)
assert _spec is not None and _spec.loader is not None
verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verifier)

MISSING_RESEARCH_INPUTS = [
    relative for relative, _role in verifier.INPUT_ROLES
    if not (REPO_ROOT / relative).is_file()
]
pytestmark = pytest.mark.skipif(
    bool(MISSING_RESEARCH_INPUTS),
    reason="SparkX research inputs are not provisioned in this export",
)


@pytest.fixture()
def research_tree(tmp_path: Path) -> Path:
    root = tmp_path / "research"
    for relative, _role in verifier.INPUT_ROLES:
        source = REPO_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return root


def _read_json(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _write_json(root: Path, relative: str, payload: object) -> None:
    (root / relative).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _mutate_json(root: Path, relative: str, mutate) -> None:
    payload = _read_json(root, relative)
    mutate(payload)
    _write_json(root, relative, payload)


def _run(root: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--root",
            str(root),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )


def _out(root: Path) -> Path:
    return root / "validation_output"


def test_valid_replay_is_deterministic_and_metadata_only(research_tree: Path) -> None:
    out_dir = _out(research_tree)

    first = _run(research_tree, out_dir)
    assert first.returncode == 0, first.stderr
    validation_bytes = (out_dir / "validation.json").read_bytes()
    manifest_bytes = (out_dir / "intake_manifest.json").read_bytes()

    second = _run(research_tree, out_dir)
    assert second.returncode == 0, second.stderr
    assert (out_dir / "validation.json").read_bytes() == validation_bytes
    assert (out_dir / "intake_manifest.json").read_bytes() == manifest_bytes
    assert first.stdout == second.stdout
    assert validation_bytes == validation_bytes.replace(b"\r\n", b"\n")

    validation = json.loads(validation_bytes)
    verdict = validation["verdict"]
    assert verdict["status"] == "pass_metadata_only"
    assert verdict["metadata_only"] is True
    assert verdict["training_admitted"] is False
    assert verdict["media_assets_downloaded"] == 0
    assert verdict["asset_qa_run"] is False
    assert verdict["registry_unchanged"] is True
    assert validation["failures"] == []
    assert all(check["ok"] for check in validation["checks"])
    assert len(validation["checks"]) >= 40

    inventory = validation["inventory"]
    assert inventory["source_families"] == 36
    assert inventory["hugging_face_repositories_inspected"] == 42
    assert inventory["catalog_repository_entries"] == 40
    assert inventory["lookup_repositories"] == ["apple/mmau", "confit/clotho"]
    assert inventory["evidence_records"] == 125
    assert inventory["evidence_hashed"] == 121
    assert inventory["evidence_unresolved"] == 4
    assert inventory["media_assets_downloaded"] == 0
    assert inventory["model_runs"] == 0
    assert inventory["training_admitted_families"] == 0
    assert inventory["media_audited_families"] == 0
    assert inventory["registry_entries"] == 13
    assert inventory["proposed_asset_ceiling_gib"] == 25
    assert inventory["proposed_asset_ceiling_status"] == "prospective_not_authorization"

    unresolved = {row["name"] for row in validation["unresolved_records"]}
    assert unresolved == {
        "dataset_api_nvidia__MF-Think",
        "dataset_card_confit__clotho",
        "dataset_card_UnFaZeD07__Music-AVQA",
        "dataset_card_DreamyWanderer__MusicNet",
    }
    assert all(
        row["error"].startswith("HTTP Error ") for row in validation["unresolved_records"]
    )

    inputs = {row["path"]: row for row in validation["inputs"]}
    assert len(inputs) == len(verifier.INPUT_ROLES)
    for row in inputs.values():
        assert len(row["sha256"]) == 64
        assert row["bytes"] > 0
    registry_hash = hashlib.sha256((research_tree / REGISTRY).read_bytes()).hexdigest()
    assert validation["source_pins"]["registry_sha256"] == registry_hash
    assert inputs[REGISTRY]["sha256"] == registry_hash

    manifest = json.loads(manifest_bytes)
    assert manifest["metadata_only_not_an_ingestion_manifest"] is True
    assert manifest["training_admitted"] is False
    assert manifest["media_assets_downloaded"] == 0
    assert manifest["asset_qa_not_run"] is True
    assert manifest["registry_unchanged"] is True
    assert manifest["status"] == (
        "metadata_only_prospective_not_an_ingestion_manifest"
    )
    assert manifest["authorization"] == {
        "asset_download_authorized": False,
        "asset_qa_authorized": False,
        "training_authorized": False,
        "registry_admission_authorized": False,
        "note": (
            "Prospective planning record only. It is not an ingestion manifest, "
            "download approval, asset-QA pass or training admission."
        ),
    }
    assert manifest["proposed_asset_ceiling_gib"] == 25
    assert manifest["proposed_ceiling_status"] == "prospective_not_authorization"
    assert manifest["asset_qa"] == {
        "status": "not_run",
        "samples_reviewed": 0,
        "authorized": False,
    }
    assert manifest["registry"]["changed"] is False
    assert manifest["registry"]["admitted_entries"] == 0
    assert manifest["source_validation"]["validation_report_sha256"] == hashlib.sha256(
        validation_bytes
    ).hexdigest()
    prerequisite_ids = {
        row["id"] for row in manifest["next_bounded_asset_audit"]["prerequisites"]
    }
    assert {
        "explicit_authorization",
        "unresolved_evidence",
        "rights_source_map",
        "free_space_and_accounting",
        "immutable_media_manifest",
        "duplicate_graph_execution",
        "bounded_asset_qa",
        "asset_qa_report",
        "registry_admission",
    } <= prerequisite_ids
    assert manifest["duplicate_graph"]["status"] == "not_executed"
    assert manifest["split_validation"]["status"] == "requirements_only"
    assert manifest["rights_resolution"]["cleared_family_count"] == 0
    assert manifest["native_execution_not_run"] is True


@pytest.mark.parametrize(
    ("relative", "mutate", "expected_failure"),
    [
        (
            CATALOG,
            lambda catalog: catalog.__setitem__("source_family_count", 35),
            "catalog_family_count",
        ),
        (
            CATALOG,
            lambda catalog: catalog.__setitem__(
                "hugging_face_repositories_inspected", 41
            ),
            "repository_count_declaration",
        ),
        (
            HF_METADATA,
            lambda metadata: metadata.pop(),
            "repository_count_declaration",
        ),
    ],
)
def test_count_mismatch_is_refused_without_a_pass(
    research_tree: Path,
    relative: str,
    mutate,
    expected_failure: str,
) -> None:
    _mutate_json(research_tree, relative, mutate)
    out_dir = _out(research_tree)
    result = _run(research_tree, out_dir)

    assert result.returncode == 1
    validation = _read_json(research_tree, str(out_dir / "validation.json"))
    assert validation["verdict"]["status"] == "contradiction_refused"
    assert validation["verdict"]["metadata_only"] is True
    assert validation["verdict"]["training_admitted"] is False
    assert validation["failures"]
    assert expected_failure in {row["id"] for row in validation["failures"]}
    assert "pass_metadata_only" not in json.dumps(validation["verdict"])
    assert not (out_dir / "intake_manifest.json").exists()


@pytest.mark.parametrize(
    ("relative", "mutate", "expected_failure"),
    [
        (
            EVIDENCE,
            lambda evidence: evidence["sources"][0].pop("sha256"),
            "evidence_hash_coverage",
        ),
        (
            EVIDENCE,
            lambda evidence: next(
                row
                for row in evidence["sources"]
                if row["name"] == "dataset_api_nvidia__MF-Think"
            ).__setitem__("error", "HTTP Error 500: Server Error"),
            "unresolved_mapping",
        ),
        (
            EVIDENCE,
            lambda evidence: next(
                row for row in evidence["sources"] if row["name"] == "dataset_paper_pixmo"
            ).pop("sha256"),
            "evidence_references_resolve",
        ),
        (
            CATALOG,
            lambda catalog: catalog.__setitem__("training_admitted", True),
            "no_asset_or_training_claims",
        ),
        (
            CATALOG,
            lambda catalog: catalog["source_families"][0].__setitem__(
                "training_admitted", True
            ),
            "no_family_admission",
        ),
        (
            CATALOG,
            lambda catalog: catalog["source_families"][0].__setitem__(
                "media_audited", True
            ),
            "no_family_admission",
        ),
    ],
)
def test_unresolved_or_contradictory_evidence_is_refused(
    research_tree: Path,
    relative: str,
    mutate,
    expected_failure: str,
) -> None:
    _mutate_json(research_tree, relative, mutate)
    out_dir = _out(research_tree)
    result = _run(research_tree, out_dir)

    assert result.returncode == 1
    validation = _read_json(research_tree, str(out_dir / "validation.json"))
    failure_ids = {row["id"] for row in validation["failures"]}
    assert expected_failure in failure_ids, failure_ids
    assert validation["verdict"]["status"] == "contradiction_refused"
    assert validation["verdict"]["failed_check_count"] == len(validation["failures"])
    assert not (out_dir / "intake_manifest.json").exists()


def test_tampered_registry_and_hash_pins_are_refused(research_tree: Path) -> None:
    registry = _read_json(research_tree, REGISTRY)
    registry["entries"][0]["auto_admit"] = True
    registry["entries"][0]["ingestion_status"] = "downloaded"
    _write_json(research_tree, REGISTRY, registry)

    out_dir = _out(research_tree)
    result = _run(research_tree, out_dir)
    assert result.returncode == 1
    validation = _read_json(research_tree, str(out_dir / "validation.json"))
    failure_ids = {row["id"] for row in validation["failures"]}
    assert "registry_pin_matches" in failure_ids
    assert "registry_quarantine_policy" in failure_ids
    assert validation["verdict"]["registry_unchanged"] is False
    assert validation["source_pins"]["registry_unchanged"] is False
    assert not (out_dir / "intake_manifest.json").exists()


def test_hash_metadata_tamper_is_refused(research_tree: Path) -> None:
    metadata = _read_json(research_tree, HF_METADATA)
    metadata[0]["metadata_sha256"] = "f" * 64
    _write_json(research_tree, HF_METADATA, metadata)

    out_dir = _out(research_tree)
    result = _run(research_tree, out_dir)
    assert result.returncode == 1
    validation = _read_json(research_tree, str(out_dir / "validation.json"))
    assert "api_evidence_hash_matches_metadata" in {
        row["id"] for row in validation["failures"]
    }


@pytest.mark.parametrize(
    ("relative", "mutate"),
    [
        (
            CATALOG,
            lambda catalog: catalog["source_families"][0].__setitem__(
                "required_checks", []
            ),
        ),
        (
            CATALOG,
            lambda catalog: catalog["source_families"][0].__setitem__(
                "rights_review", ""
            ),
        ),
        (
            CATALOG,
            lambda catalog: catalog["source_families"][0].__setitem__(
                "split_group_keys", []
            ),
        ),
        (
            EVIDENCE,
            lambda evidence: evidence["sources"][0].__setitem__(
                "scratch_path", "local/scratch/source"
            ),
        ),
    ],
)
def test_requirement_tamper_is_refused(
    research_tree: Path, relative: str, mutate
) -> None:
    _mutate_json(research_tree, relative, mutate)
    out_dir = _out(research_tree)
    result = _run(research_tree, out_dir)
    assert result.returncode == 1
    validation = _read_json(research_tree, str(out_dir / "validation.json"))
    assert validation["verdict"]["status"] == "contradiction_refused"
    assert validation["failures"]
    assert not (out_dir / "intake_manifest.json").exists()


def test_missing_input_aborts_without_output(research_tree: Path) -> None:
    (research_tree / PLAN).unlink()
    out_dir = _out(research_tree)
    result = _run(research_tree, out_dir)
    assert result.returncode == 2
    assert "verification aborted" in result.stderr
    assert not (out_dir / "validation.json").exists()
    assert not (out_dir / "intake_manifest.json").exists()


def test_refusal_removes_stale_prospective_manifest(research_tree: Path) -> None:
    out_dir = _out(research_tree)
    assert _run(research_tree, out_dir).returncode == 0
    assert (out_dir / "intake_manifest.json").is_file()

    _mutate_json(
        research_tree,
        CATALOG,
        lambda catalog: catalog.__setitem__("media_assets_downloaded", 1),
    )
    result = _run(research_tree, out_dir)
    assert result.returncode == 1
    validation = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["verdict"]["status"] == "contradiction_refused"
    assert not (out_dir / "intake_manifest.json").exists()
