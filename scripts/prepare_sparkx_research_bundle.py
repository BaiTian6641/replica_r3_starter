#!/usr/bin/env python3
"""Hash-pin the existing SparkX research inputs and verify a clean checkout.

The bundle is a *manifest*, not a copy. It never duplicates model weights,
media assets or other large payloads, and it never rewrites an input file.
Every required input is identified by repository-relative path, role, byte
size and SHA-256 digest.

A clean checkout is one where every required input is tracked by git and its
worktree bytes match the manifest. Untracked inputs are refused by default:
they are a hidden user-file dependency even when they happen to exist locally.
``--allow-untracked`` is an explicit development-only override; a manifest
written that way records ``clean_checkout_ready: false``.

Commands:
  python3 scripts/prepare_sparkx_research_bundle.py --write
  python3 scripts/prepare_sparkx_research_bundle.py --check
  python3 scripts/prepare_sparkx_research_bundle.py --check --allow-untracked

Exit codes: 0 verified, 1 verification failure, 2 missing/unreadable input
or manifest. No network access, no model or dataset loader, no native job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

FORMAT = "replica.sparkx_research_bundle.v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "reports/sparkx_first_tranche_20261026/research_bundle_manifest.json"

# Ordered (path, role) pairs. Paths are repository-relative; the manifest is
# deterministic because entries are sorted by path before serialization.
INPUT_ROLES: list[tuple[str, str]] = [
    (
        "reports/spark_preparation_20261025/run_01/headers/"
        "official_bf16_control/tensor_index.json",
        "sealed BF16 source tensor index used for the backbone parameter inventory",
    ),
    (
        "reports/spark_preparation_20261025/run_01/headers/"
        "official_fp8_adaptation_base/tensor_index.json",
        "sealed mixed-dtype FP8 adaptation tensor index (source storage only)",
    ),
    (
        "reports/fused_adamw_20261025/sustained01.guard.json",
        "repository driver-guard policy values",
    ),
    (
        "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md",
        "declared encoder counts, revisions, projectors and memory-module scope",
    ),
    (
        "reports/sparkx_residency_20261026/calculate_residency.py",
        "deterministic residency ledger generator",
    ),
    (
        "scripts/prepare_sparkx_research_bundle.py",
        "bundle manifest generator/verifier (this file's hash is pinned separately)",
    ),
    (
        "scripts/prepare_sparkx_first_tranche.py",
        "bounded first-tranche admission implementation",
    ),
    (
        "reports/sparkx_residency_20261026/residency_v2.json",
        "current analytical residency receipt (v2)",
    ),
    (
        "docs/experiments/20261026_Spark_Residency_Reconciliation.md",
        "residency reconciliation note: 18-byte formula, dtype inventory and gates",
    ),
    (
        "docs/datasets/20261026_SparkX_Dataset_Catalog.json",
        "source pins, licenses, split metadata and priorities",
    ),
    (
        "docs/datasets/20261026_SparkX_Dataset_Plan.md",
        "dataset strategy and intake constraints",
    ),
    (
        "docs/datasets/20261026_MiniCPM_Dataset_Validation.md",
        "MiniCPM validation plan referenced by the dataset package",
    ),
    (
        "docs/datasets/20261024_Source_Catalog.json",
        "earlier source catalog retained as a reference",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/evidence.json",
        "per-source evidence records and hashed provenance",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/hf_metadata.json",
        "pinned Hugging Face metadata snapshot",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/README.md",
        "dataset research report and limitations",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/documentation_validation.json",
        "prior documentation validation record retained as an input",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/repository_snapshot.json",
        "repository snapshot pins for the research package",
    ),
    (
        "reports/sparkx_dataset_hunt_20261026/clotho_metadata_audit.json",
        "Clotho provenance CSV count audit",
    ),
    (
        "reports/sparkx_dataset_validation_20261026/validation.json",
        "metadata-only validation report",
    ),
    (
        "reports/sparkx_dataset_validation_20261026/intake_manifest.json",
        "prospective metadata-only intake record",
    ),
    (
        "configs/dataset_registry.json",
        "runtime dataset registry (quarantined, read-only for this bundle)",
    ),
    (
        "configs/sparkx_first_tranche_v1.json",
        "bounded first-tranche protocol configuration",
    ),
    (
        "docs/08_Dataset_Acquisition_and_Curriculum.md",
        "acquisition guide retained with its known snapshot drift",
    ),
    (
        "docs/experiments/20261026_Spark_Dataset_Preparation.md",
        "dataset preparation, provisioning and tranche documentation",
    ),
    (
        "reports/sparkx_first_tranche_20261026/synthetic_assets/assets.jsonl",
        "owned synthetic asset manifest for first-tranche protocol replay",
    ),
    *[
        (
            f"reports/sparkx_first_tranche_20261026/synthetic_assets/{name}",
            "owned synthetic fixture bytes; not external media",
        )
        for name in (
            "synthetic_scene_0001.json",
            "synthetic_scene_0002.json",
            "synthetic_audio_0001.json",
            "synthetic_audio_0002.json",
            "synthetic_av_0001.json",
            "synthetic_av_0002.json",
        )
    ],
]

# Large payloads that must never be copied into this bundle. They are either
# absent or separately provisioned; their hashes are not required here.
NOT_COPIED = [
    "checkpoints/** (model weights)",
    "data/** (bulk media, annotations and caches)",
    "runs/** (ignored native run evidence)",
    "reports/**/*.safetensors",
]


class BundleError(RuntimeError):
    """A required input is missing, unreadable, untracked or hash-mismatched."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_tracked(root: Path, relative: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _entry(root: Path, relative: str, role: str, allow_untracked: bool) -> dict:
    path = root / relative
    if not path.is_file():
        raise BundleError(f"missing required input: {relative}")
    try:
        digest = sha256_file(path)
        size = path.stat().st_size
    except OSError as error:  # pragma: no cover - permission/IO failure
        raise BundleError(f"unreadable required input: {relative}: {error}") from error
    tracked = _git_tracked(root, relative)
    if not tracked and not allow_untracked:
        raise BundleError(
            f"untracked required input: {relative} (commit it, or use "
            "--allow-untracked for a development-only manifest)"
        )
    return {
        "path": relative,
        "role": role,
        "bytes": size,
        "sha256": digest,
        "tracked_in_git": tracked,
    }


def _canonical_digest(entries: list[dict]) -> str:
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_manifest(root: Path, allow_untracked: bool = False) -> dict:
    root = Path(root)
    entries = sorted(
        (_entry(root, relative, role, allow_untracked) for relative, role in INPUT_ROLES),
        key=lambda entry: entry["path"],
    )
    untracked = [entry["path"] for entry in entries if not entry["tracked_in_git"]]
    return {
        "format": FORMAT,
        "kind": "hashed_manifest_not_an_asset_copy",
        "generator": "scripts/prepare_sparkx_research_bundle.py",
        "root_policy": (
            "all paths are repository-relative; no absolute path, home directory "
            "or temporary directory is recorded"
        ),
        "input_count": len(entries),
        "total_input_bytes": sum(entry["bytes"] for entry in entries),
        "bundle_digest": _canonical_digest(entries),
        "clean_checkout_ready": not untracked,
        "untracked_inputs": untracked,
        "tracking_enforced": not allow_untracked,
        "inputs": entries,
        "not_copied": NOT_COPIED,
        "provisioning": {
            "write_command": "python3 scripts/prepare_sparkx_research_bundle.py --write",
            "check_command": "python3 scripts/prepare_sparkx_research_bundle.py --check",
            "development_override": (
                "python3 scripts/prepare_sparkx_research_bundle.py --check "
                "--allow-untracked"
            ),
            "named_skip": (
                "tests that require this bundle should call "
                "pytest.importorskip or skip with the --check command in the reason; "
                "do not claim clean replay from an unprovisioned checkout"
            ),
            "network_required": False,
            "large_payload_copied": False,
        },
    }


def verify_manifest(
    root: Path,
    manifest: dict,
    *,
    allow_untracked: bool = False,
    require_tracking: bool = True,
) -> dict:
    root = Path(root)
    if manifest.get("format") != FORMAT:
        raise BundleError(f"unsupported manifest format: {manifest.get('format')!r}")
    declared = manifest.get("inputs")
    if not isinstance(declared, list) or not declared:
        raise BundleError("manifest has no inputs")

    failures: list[dict] = []
    entries: list[dict] = []
    seen_paths = set()
    expected_roles = dict(INPUT_ROLES)
    for record in declared:
        relative = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not relative:
            failures.append({"path": relative, "reason": "invalid_manifest_entry"})
            continue
        if relative.startswith("/") or ".." in Path(relative).parts:
            failures.append({"path": relative, "reason": "non_relative_path"})
            continue
        if relative in seen_paths:
            failures.append({"path": relative, "reason": "duplicate_manifest_path"})
            continue
        if relative not in expected_roles:
            failures.append({"path": relative, "reason": "unexpected_manifest_input"})
            continue
        if record.get("role") != expected_roles[relative]:
            failures.append({"path": relative, "reason": "role_mismatch"})
            continue
        seen_paths.add(relative)
        path = root / relative
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        tracked = _git_tracked(root, relative)
        if size != record.get("bytes"):
            failures.append(
                {"path": relative, "reason": "size_mismatch", "expected": record.get("bytes"), "actual": size}
            )
        if digest != record.get("sha256"):
            failures.append(
                {"path": relative, "reason": "sha256_mismatch", "expected": record.get("sha256"), "actual": digest}
            )
        if require_tracking and not allow_untracked and not tracked:
            failures.append({"path": relative, "reason": "untracked_in_git"})
        entries.append(
            {
                "path": relative,
                "role": record.get("role"),
                "bytes": size,
                "sha256": digest,
                "tracked_in_git": tracked,
            }
        )

    if require_tracking and not allow_untracked and not manifest.get("clean_checkout_ready"):
        failures.append(
            {
                "path": "<bundle>",
                "reason": "manifest_not_clean_checkout_ready",
                "detail": "regenerate the manifest after committing every listed input",
            }
        )

    entries.sort(key=lambda entry: entry["path"])
    recomputed = _canonical_digest(entries)
    if recomputed != manifest.get("bundle_digest"):
        failures.append(
            {
                "path": "<bundle>",
                "reason": "bundle_digest_mismatch",
                "expected": manifest.get("bundle_digest"),
                "actual": recomputed,
            }
        )

    missing = sorted(
        relative for relative, _role in INPUT_ROLES if relative not in seen_paths
    )
    if missing:
        failures.append({"path": "<bundle>", "reason": "missing_manifest_inputs", "paths": missing})

    return {
        "format": "replica.sparkx_research_bundle_verification.v1",
        "verified": not failures,
        "clean_checkout_ready": not failures and bool(manifest.get("clean_checkout_ready")),
        "tracking_enforced": require_tracking and not allow_untracked,
        "input_count": len(entries),
        "total_input_bytes": sum(entry["bytes"] for entry in entries),
        "bundle_digest": recomputed,
        "failures": failures,
        "network_used": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true", help="write the manifest")
    action.add_argument("--check", action="store_true", help="verify the manifest")
    parser.add_argument(
        "--allow-untracked",
        action="store_true",
        help=(
            "development override: accept inputs that are not tracked by git. "
            "The manifest records clean_checkout_ready=false. Never use this to "
            "claim a clean-checkout replay."
        ),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest or (root / DEFAULT_MANIFEST)

    try:
        if args.write:
            manifest = build_manifest(root, allow_untracked=args.allow_untracked)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(
                f"Wrote {manifest_path}: {manifest['input_count']} inputs, "
                f"digest {manifest['bundle_digest']}, "
                f"clean_checkout_ready={manifest['clean_checkout_ready']}"
            )
            return 0

        if not manifest_path.is_file():
            raise BundleError(f"manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = verify_manifest(
            root,
            manifest,
            allow_untracked=args.allow_untracked,
            require_tracking=True,
        )
    except (BundleError, json.JSONDecodeError, OSError) as error:
        print(f"bundle error: {error}", file=sys.stderr)
        return 2

    if not report["verified"]:
        print(json.dumps(report, indent=2))
        print(
            "bundle verification failed; fix the named inputs or, for development "
            "only, re-run with --allow-untracked",
            file=sys.stderr,
        )
        return 1

    print(
        f"Bundle verified: {report['input_count']} inputs, "
        f"{report['total_input_bytes']} bytes, "
        f"digest {report['bundle_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
