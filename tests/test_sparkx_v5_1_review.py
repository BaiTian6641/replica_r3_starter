"""Clean-checkout and tamper tests for the SparkX v5.1 documentation bundle.

Everything is CPU-only and offline.  The fixture copies only the small files
the portable verifier reads; it never copies the 273 GiB workspace and never
imports a model.  Native gates must remain unresolved design records here.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/sparkx_v5_1_review_20261027"
VERIFY_PATH = ROOT / REPORT / "verify.py"
OUTPUT = "verification.json"


def _load_verifier(path: Path):
    spec = importlib.util.spec_from_file_location("sparkx_v5_1_review_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = _load_verifier(VERIFY_PATH)


def _required_paths(root: Path) -> set[str]:
    deliverables = json.loads((root / REPORT / "deliverables.json").read_text(encoding="utf-8"))
    migration = json.loads((root / REPORT / "migration_manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / REPORT / "repository_snapshot.json").read_text(encoding="utf-8"))
    config = json.loads((root / "configs/sparkx_2_6_v5_1_direction_design.json").read_text(encoding="utf-8"))
    local = {row["path"] for row in migration.get("local_artifact_references", [])}
    paths = {entry["path"] for entry in deliverables["files"]}
    paths |= {row["snapshot"] for row in snapshot["inputs"]}
    paths.add(snapshot["attachment"]["snapshot"])
    paths |= set(config["references"].values())
    for entry in deliverables["files"]:
        source = root / entry["path"]
        if source.suffix != ".md" or not source.is_file():
            continue
        for target in re.findall(r"\]\(([^\n)]+)\)", source.read_text(encoding="utf-8")):
            if urlparse(target).scheme:
                continue
            destination = (source.parent / unquote(target.split("#", 1)[0])).resolve()
            if destination.is_file() and destination.is_relative_to(root):
                relative = destination.relative_to(root).as_posix()
                if relative not in local:
                    paths.add(relative)
    return paths


def _copy_fixture(destination: Path) -> Path:
    for relative in sorted(_required_paths(ROOT)):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, target)
        elif source.is_dir():
            shutil.copytree(source, target)
        else:
            raise AssertionError(f"required fixture path missing: {relative}")
    return destination


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory) -> Path:
    return _copy_fixture(tmp_path_factory.mktemp("sparkx_v5_1_review_base") / "repo")


@pytest.fixture()
def repository(base_tree: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "repo"
    shutil.copytree(base_tree, destination)
    return destination


def _run(root: Path, tracked: str = "skip") -> dict:
    return verify.verify(root=root, out=root / REPORT, write=True, tracked=tracked)


def _failed(result: dict) -> set[str]:
    return {row["name"] for row in result["failed"]}


def _mutate_json(path: Path, mutate) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_clean_fixture_passes_and_is_deterministic(repository: Path) -> None:
    first = _run(repository)
    assert not first["failed"], first["failed"]
    first_bytes = (repository / REPORT / OUTPUT).read_bytes()
    second = _run(repository)
    assert not second["failed"]
    assert (repository / REPORT / OUTPUT).read_bytes() == first_bytes
    assert isinstance(first["pending_commit_additions"], list)
    assert first["native_gates"]["executed"] == 0
    assert set(first["native_gates"]["unresolved"])


def test_document_byte_tamper_fails_pin(repository: Path) -> None:
    path = repository / "docs/architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nTampered.\n", encoding="utf-8")
    failed = _failed(_run(repository))
    assert "deliverable:docs/architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md" in failed


def test_configarithmetic_tamper_fails(repository: Path) -> None:
    path = repository / "configs/sparkx_2_6_v5_1_direction_design.json"

    def mutate(data):
        data["plastic"]["profiles"]["mini_P32"]["records"] = 2048

    _mutate_json(path, mutate)
    failed = _failed(_run(repository))
    assert "deliverable:configs/sparkx_2_6_v5_1_direction_design.json" in failed
    assert "profile_arithmetic:mini_P32" in failed
    assert "ledger_reproduces" in failed


def test_migration_old_identity_tamper_fails(repository: Path) -> None:
    path = repository / REPORT / "migration_manifest.json"

    def mutate(data):
        for entry in data["entries"]:
            if entry["kind"] == "changed_existing":
                entry["old"]["sha256"] = "0" * 64
                break

    _mutate_json(path, mutate)
    failed = _failed(_run(repository))
    assert any(name.startswith("migration_old_matches_snapshot:") for name in failed)
    assert "no_unexplained_snapshot_drift" in failed


def test_migration_new_identity_tamper_fails(repository: Path) -> None:
    path = repository / REPORT / "migration_manifest.json"

    def mutate(data):
        for entry in data["entries"]:
            if entry["kind"] == "added":
                entry["new"]["sha256"] = "0" * 64
                break

    _mutate_json(path, mutate)
    failed = _failed(_run(repository))
    assert any(name.startswith("added_identity_present:") for name in failed)


def test_migration_digest_tamper_fails(repository: Path) -> None:
    path = repository / REPORT / "migration_manifest.json"

    def mutate(data):
        data["identity"]["digest_sha256"] = "0" * 64

    _mutate_json(path, mutate)
    failed = _failed(_run(repository))
    assert "migration_identity_digest" in failed


def test_missing_added_identity_fails(repository: Path) -> None:
    (repository / REPORT / "inputs/attachment_v5_1_draft.txt").unlink()
    failed = _failed(_run(repository))
    assert "added_identity_present:reports/sparkx_v5_1_review_20261027/inputs/attachment_v5_1_draft.txt" in failed
    assert "preserved_input:reports/sparkx_v5_1_review_20261027/inputs/attachment_v5_1_draft.txt" in failed


def test_broken_local_link_fails(repository: Path) -> None:
    path = repository / "docs/architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n[broken](missing_target.md)\n", encoding="utf-8")
    failed = _failed(_run(repository))
    assert any(name.startswith("local_link:") and "missing_target.md" in name for name in failed)


def test_preserved_input_tamper_fails(repository: Path) -> None:
    path = repository / REPORT / "inputs/00_23_SparkX_2_6_4B_Final_Product_Blueprint.txt"
    path.write_text(path.read_text(encoding="utf-8") + "\nTampered snapshot.\n", encoding="utf-8")
    failed = _failed(_run(repository))
    assert "preserved_input:reports/sparkx_v5_1_review_20261027/inputs/00_23_SparkX_2_6_4B_Final_Product_Blueprint.txt" in failed


def test_historical_control_tamper_fails(repository: Path) -> None:
    path = repository / "configs/sparkx_2_6_4b_final_product_v5.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    failed = _failed(_run(repository))
    assert "historical_v5_configuration_unchanged" in failed
    assert "retained_identity_present:configs/sparkx_2_6_4b_final_product_v5.json" in failed


def test_native_gate_promotion_fails(repository: Path) -> None:
    path = repository / REPORT / "native_gates.json"

    def mutate(data):
        data["gates"][0]["executed"] = True
        data["gates"][0]["status"] = "passed"
        data["gates"][0]["measured"] = "invented"

    _mutate_json(path, mutate)
    failed = _failed(_run(repository))
    assert "native_gates_not_executed" in failed
    assert "native_gates_summary" in failed


def test_unexplained_untracked_identity_fails(repository: Path) -> None:
    path = repository / REPORT / "migration_manifest.json"

    def mutate(data):
        data["entries"] = [entry for entry in data["entries"]
                           if entry["path"] != "docs/architecture/25_MiniCPM5_Direction_Correction_and_Learning_Capacity_Plan.md"]

    _mutate_json(path, mutate)
    failed = _failed(_run(repository))
    assert "deliverables_declared_in_migration" in failed


def test_local_artifact_absence_is_allowed_only_when_declared(repository: Path) -> None:
    # The fixture never copies `runs/`; the declared references are absent and pass.
    result = _run(repository)
    assert not result["failed"]
    assert all(not row["present"] for row in result["local_artifacts"].values())
    # An undeclared missing link is still a failure.
    path = repository / "docs/architecture/25_MiniCPM5_Direction_Correction_and_Learning_Capacity_Plan.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n[missing](missing_run.json)\n", encoding="utf-8")
    assert any(name.startswith("local_link:") for name in _failed(_run(repository)))
