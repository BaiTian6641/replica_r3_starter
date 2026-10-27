"""Portable final-product verifier, evidence and bundle tests.

Every test builds its own tracked fixture from repository-relative files.  The
tests prove that required checks are self-contained in tracked evidence, that a
missing or tampered v4/calibration/continuation/curve/clean-clone receipt fails,
that untracked or missing document links fail, and that only explicitly
reviewed drift is accepted.  No model, dataset loader, native job or network is
used.
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

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/sparkx_final_product_20261026"
VERIFY_SCRIPT = ROOT / REPORT / "verify.py"
BUNDLE_SCRIPT = ROOT / "scripts/prepare_sparkx_research_bundle.py"
SNAPSHOT = ROOT / REPORT / "repository_snapshot.json"
FINAL_MANIFEST = ROOT / REPORT / "final_product_bundle_manifest.json"
OUTPUT = "verification_v2.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify = _load(VERIFY_SCRIPT, "sparkx_final_product_verify")
bundle = _load(BUNDLE_SCRIPT, "sparkx_research_bundle_final_product")


def _fixture_paths() -> set[str]:
    paths = {relative for relative, _role in bundle.INPUT_ROLES}
    paths |= {relative for relative, _role in bundle.FINAL_PRODUCT_INPUT_ROLES}
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    paths |= {entry["path"] for entry in snapshot["input_files"]}
    for tree in ("docs", "configs"):
        for path in (ROOT / tree).rglob("*"):
            if path.is_file():
                paths.add(path.relative_to(ROOT).as_posix())
    paths.add("tests/test_sparkx_final_product.py")
    return paths


def _copy_fixture(destination: Path) -> Path:
    for relative in sorted(_fixture_paths()):
        source = ROOT / relative
        target = destination / relative
        assert source.is_file(), f"fixture source missing: {relative}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    subprocess.run(["git", "init", "-q"], cwd=destination, check=True)
    subprocess.run(["git", "add", "-A"], cwd=destination, check=True)
    manifest = bundle.build_manifest(destination, profile="final-product")
    manifest_path = destination / bundle.DEFAULT_FINAL_PRODUCT_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=destination, check=True)
    return destination


@pytest.fixture(scope="module")
def base_repository(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sparkx_final_product_base") / "repo"
    return _copy_fixture(root)


@pytest.fixture()
def repository(base_repository: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "repo"
    shutil.copytree(base_repository, destination)
    return destination


def _run(root: Path) -> dict:
    return verify.verify(root=root, out=root / REPORT, write=True)


def _failed(result: dict) -> set[str]:
    return {entry["name"] for entry in result["failed"]}


def _mutate_json(path: Path, mutate) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_clean_checkout_verification_passes_without_source_cache(repository: Path) -> None:
    result = _run(repository)
    assert result["status"] == "passed", result["failed"][:10]
    assert result["failed"] == []
    assert result["unexplained_input_drift"] == []
    assert result["historical_manifest"]["input_count"] == 117
    assert result["historical_manifest"]["unreviewed_drift"] == []
    portable = json.loads((repository / REPORT / "portable_evidence.json").read_text())
    assert all(
        not record["path"].startswith("/")
        for group in ("required_receipts", "required_configs")
        for record in portable[group]
    )
    assert (repository / REPORT / OUTPUT).is_file()


def test_v51_reviewed_migration_is_applied_and_pinned(repository: Path) -> None:
    result = _run(repository)
    assert result["status"] == "passed", result["failed"][:10]
    applied = {entry["path"] for entry in result["reviewed_input_changes_applied"]}
    assert {
        "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md",
        "docs/architecture/22_State_Mechanism_and_RSI_Contracts.md",
        "docs/architecture/README.md",
    } <= applied
    manifest = json.loads((repository / bundle.DEFAULT_FINAL_PRODUCT_MANIFEST).read_text())
    inputs = {entry["path"]: entry for entry in manifest["inputs"]}
    required = {
        "docs/architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md",
        "docs/architecture/25_MiniCPM5_Direction_Correction_and_Learning_Capacity_Plan.md",
        "configs/sparkx_2_6_v5_1_direction_design.json",
        "reports/sparkx_v5_1_review_20261027/README.md",
        "reports/sparkx_v5_1_review_20261027/migration_manifest.json",
        "reports/sparkx_v5_1_review_20261027/native_gates.json",
        "reports/sparkx_v5_1_review_20261027/deliverables.json",
    }
    assert required <= set(inputs)
    for relative in required:
        assert inputs[relative]["sha256"] == hashlib.sha256((repository / relative).read_bytes()).hexdigest()


def test_v4_tamper_fails_denominators(repository: Path) -> None:
    path = repository / "reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v4.json"

    def mutate(data: dict) -> None:
        candidate = data["candidate_variant"]
        data["metrics"]["variants"][candidate]["quality"]["accepted_correct"] = 22

    _mutate_json(path, mutate)
    failed = _failed(_run(repository))
    assert "current_acceptance_v4_denominators" in failed
    assert any(name.startswith("receipt_hash:") for name in failed)


def test_v4_missing_fails(repository: Path) -> None:
    (repository / "reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v4.json").unlink()
    failed = _failed(_run(repository))
    assert "current_acceptance_v4_present" in failed
    assert any(name.startswith("receipt_present:") for name in failed)


def test_calibration_tamper_and_missing_fail(repository: Path) -> None:
    path = repository / "reports/memory_acceptance_20261026/calibration_dev01.json"

    def mutate(data: dict) -> None:
        disposition = data["gate_classification"]["disposition"]
        disposition["selected_rule"] = "promote"
        disposition["writer_training_permitted"] = True

    _mutate_json(path, mutate)
    failed = _failed(_run(repository))
    assert "calibration_rejected" in failed
    assert "calibration_no_promotion_or_native_execution" in failed

    path.unlink()
    missing = _failed(_run(repository))
    assert "calibration_present" in missing


def test_continuation_tamper_and_scope_limit_fail(repository: Path) -> None:
    comparison = repository / "reports/nested_text_continuation_comparison_20261026/comparison.json"
    protocol = repository / "configs/nested_text_continuation_comparison_v1.json"

    def mutate_comparison(data: dict) -> None:
        data["status"] = "mismatch"

    _mutate_json(comparison, mutate_comparison)
    failed = _failed(_run(repository))
    assert "continuation_exact_match" in failed

    def mutate_protocol(data: dict) -> None:
        data["bank"]["experts"] = 8

    _mutate_json(protocol, mutate_protocol)
    scope_failed = _failed(_run(repository))
    assert "continuation_16_expert_scope" in scope_failed
    assert "continuation_protocol_binding" in scope_failed


def test_continuation_missing_guard_fails(repository: Path) -> None:
    (repository / "reports/nested_text_continuation_comparison_20261026/resume.guard.json").unlink()
    failed = _failed(_run(repository))
    assert "continuation_guard_present:guard-resume" in failed


def test_vision_curve_tamper_fails(repository: Path) -> None:
    path = repository / "reports/vision_learnability_curve_v1_20261026/summary.json"

    def mutate(data: dict) -> None:
        roles = data["decision"]["t2"]["roles"]
        roles["answer"]["series"][1]["loss"] = roles["answer"]["series"][0]["loss"]
        roles["answer"]["steps"][0]["non_increasing"] = True
        criteria = data["decision"]["t3"]["criteria"]
        criteria["answer_correct_accuracy"]["prefix7"] = 0.2
        criteria["answer_prefill_gold_accuracy_from_zero"]["prefix7"] = 0.2

    _mutate_json(path, mutate)
    failed = _failed(_run(repository))
    assert "vision_curve_first_prefix_overshoot:answer" in failed
    assert "vision_curve_monotonicity_fails" in failed
    assert "vision_curve_zero_answer_accuracy" in failed


def test_vision_curve_missing_fails(repository: Path) -> None:
    (repository / "reports/vision_learnability_curve_v1_20261026/summary.json").unlink()
    failed = _failed(_run(repository))
    assert "vision_curve_present" in failed


def test_clean_clone_receipt_tamper_and_missing_fail(repository: Path) -> None:
    path = repository / "reports/final_verification_20261027/summary.json"

    def mutate(data: dict) -> None:
        data["clean_export"]["failed"] = 1

    _mutate_json(path, mutate)
    failed = _failed(_run(repository))
    assert "final_verification_clean_clone_passed" in failed

    path.unlink()
    missing = _failed(_run(repository))
    assert "final_verification_present" in missing


def test_untracked_and_missing_links_fail(repository: Path) -> None:
    untracked = repository / "reports/untracked_link_target.json"
    untracked.write_text("{}\n", encoding="utf-8")
    readme = repository / REPORT / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n[untracked evidence](../../reports/untracked_link_target.json)\n",
        encoding="utf-8",
    )
    failed = _failed(_run(repository))
    assert any(name.startswith("local_link_tracked:") for name in failed)

    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n[missing evidence](../../reports/does_not_exist.json)\n",
        encoding="utf-8",
    )
    missing = _failed(_run(repository))
    assert any(name.startswith("local_link:") for name in missing)


def test_unexplained_snapshot_drift_fails(repository: Path) -> None:
    path = repository / "docs/experiments/20261026_Nested_Learning_NL0.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nunreviewed drift\n", encoding="utf-8")
    failed = _failed(_run(repository))
    assert "unexplained_input_drift:docs/experiments/20261026_Nested_Learning_NL0.md" in failed
    assert "no_unexplained_snapshot_drift" in failed
    assert "historical_manifest_unreviewed_drift:docs/experiments/20261026_Nested_Learning_NL0.md" in failed


def test_reviewed_change_requires_exact_current_hash(repository: Path) -> None:
    path = repository / "docs/architecture/README.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nunreviewed pointer drift\n", encoding="utf-8")
    failed = _failed(_run(repository))
    assert "unexplained_input_drift:docs/architecture/README.md" in failed
    assert "no_unexplained_snapshot_drift" in failed


def test_final_manifest_tamper_and_missing_fail(repository: Path) -> None:
    manifest_path = repository / bundle.DEFAULT_FINAL_PRODUCT_MANIFEST
    original = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original)
    manifest["clean_export_links"] = manifest["clean_export_links"][:-1]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failed = _failed(_run(repository))
    assert "final_manifest_clean_export_links" in failed

    manifest_path.unlink()
    missing = _failed(_run(repository))
    assert "final_product_bundle_manifest_present" in missing


def test_final_manifest_input_tamper_fails(repository: Path) -> None:
    manifest_path = repository / bundle.DEFAULT_FINAL_PRODUCT_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["inputs"]:
        if entry["path"] == "reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v4.json":
            entry["sha256"] = "0" * 64
            break
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failed = _failed(_run(repository))
    assert "final_manifest_inputs_match_files" in failed
    assert "final_manifest_digest" in failed


def test_final_profile_is_additive_and_does_not_rewrite_historical(repository: Path) -> None:
    historical = repository / "reports/sparkx_first_tranche_20261026/research_bundle_manifest.json"
    before = hashlib.sha256(historical.read_bytes()).hexdigest()
    manifest = bundle.build_manifest(repository, profile="final-product")
    after = hashlib.sha256(historical.read_bytes()).hexdigest()
    assert before == after
    assert manifest["profile"] == "final-product"
    assert manifest["format"] == bundle.FINAL_PRODUCT_FORMAT
    assert manifest["clean_checkout_ready"] is True
    paths = {entry["path"] for entry in manifest["inputs"]}
    assert bundle.DEFAULT_FINAL_PRODUCT_MANIFEST not in paths
    assert bundle.FINAL_PRODUCT_GENERATED_OUTPUT not in paths
    for link in manifest["clean_export_links"]:
        assert link["path"] in paths
        assert len(link["sha256"]) == 64
    report = bundle.verify_manifest(repository, manifest)
    assert report["verified"] is True, report["failures"]
    assert report["profile"] == "final-product"
