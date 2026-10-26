"""Shared test conventions: artifact-dependent integration tests never skip silently.

Self-contained unit tests build synthetic fixtures and always run. Tests that
need ignored local acquisitions (sealed model source, local lock, vision
overlay, external caches) declare them through :func:`require_local_artifacts`
so a missing input produces one explicit skip whose reason names the exact
provisioning command, never a silent pass or an obscure failure.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROVISIONING = {
    "checkpoints/minicpm5_source": "scripts/pin_model.py (sealed MiniCPM5-2B acquisition; ignored local payload)",
    "configs/minicpm5_sealed_lock.local.json": "scripts/seal_snapshot.py over the sealed local source snapshot",
    "PIL": "python -m venv .venv-vision && .venv-vision/bin/pip install -r requirements-vision-overlay.txt",
    "checkpoints/siglip2_base_patch16_224_source": "scripts/prepare_vision_encoder.py --offline",
    "data/external_caption_tranche_20261025_v2/raw/images": ".venv/bin/python scripts/prepare_external_tranche.py --config configs/external_caption_tranche_v1.json (pinned acquisition; or copy the verified ignored raw/images directory from the provisioned workspace)",
    "data/cache/spark_sources/run_02": "scripts/prepare_spark_sources.py --plan configs/spark_source_preparation_v2.json "
                                       "--output data/cache/spark_sources/run_02 --export-header-bundle",
    "runs/minicpm5_vision_v0_20261025/score_restart_replay01/score_report.json": "copy the immutable V0 score replay from the provisioned native-evidence workspace; historical integration only",
    "runs/vision_balanced_alignment_20261026/train01/run_report.json": "copy the immutable v1 train01 run report from the provisioned native-evidence workspace; optional integration only",
    "runs/vision_balanced_alignment_20261026/reload01/reload_report.json": "copy the immutable v1 reload01 report from the provisioned native-evidence workspace; optional integration only",
    "data/cache/sparkx_nested_learning_20261026": "python3 scripts/provision_sparkx_nested_sources.py --source-dir <verified-local-source-directory> (offline, hash-checked; optional source-payload integration)",
    "reports/sparkx_first_tranche_20261026/research_bundle_manifest.json": "python3 scripts/prepare_sparkx_research_bundle.py --write (commit every listed input, then verify with --check without --allow-untracked)",
}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires ignored local artifacts; skipped with provisioning "
        "instructions when they are absent (see docs/maintenance/20261025_Test_Provisioning.md)",
    )


def _absent(name: str) -> bool:
    if "/" in name:
        return not (ROOT / name).exists()
    return importlib.util.find_spec(name) is None


def provisioning_reason(*requirements: str) -> tuple[bool, str]:
    """Return (missing, reason) naming every absent artifact and its provisioning."""
    missing = [name for name in requirements if _absent(name)]
    if not missing:
        return False, "all required local artifacts present"
    how = "; ".join(f"{name}: {PROVISIONING.get(name, 'not documented — add it to tests/conftest.py')}"
                    for name in missing)
    return True, f"missing local artifact(s) — provision first: {how}"


def require_local_artifacts(*requirements: str):
    """Explicit skip-with-instructions for artifact-dependent integration tests."""
    missing, reason = provisioning_reason(*requirements)
    return pytest.mark.skipif(missing, reason=reason)
