#!/usr/bin/env python3
"""Measure preserved releases and prove resumable migration on a disposable copy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from replica_lab.intrinsic_memory_archive_migration_v16 import (
    MigrationInterrupted, inventory_archive, migrate_generation, sha256)
from replica_lab.intrinsic_memory_budget_v12 import atomic
from replica_lab.retention import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--release-costs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("archive migration preparation receipt must be fresh")
    run = args.run.resolve()
    inventory = inventory_archive(run)
    current = measure(run / "stores")
    row = next((row for row in inventory["rows"] if row["world"] == "v11_fit_00"),
               inventory["rows"][0])
    source = Path(row["release_root"]) / f"{row['final']['generation']:08d}"
    with tempfile.TemporaryDirectory(prefix="replica-archive-migration-v16-") as temporary:
        destination = Path(temporary) / "migration"
        interrupted = False
        try:
            migrate_generation(source, destination, interrupt_after="manifest")
        except MigrationInterrupted:
            interrupted = True
        pointer_before_recovery = (destination / "current.json").exists()
        recovery = migrate_generation(source, destination)
        pointer = json.loads((destination / "current.json").read_text())
        dry_run = {
            "source_generation": str(source),
            "interruption_injected_after_manifest_before_pointer": interrupted,
            "pointer_existed_before_recovery": pointer_before_recovery,
            "recovery": recovery,
            "pointer_manifest_matches_source": pointer["manifest_sha256"] == sha256(
                source / "manifest.json"),
            "second_completion_idempotent": migrate_generation(
                source, destination)["status"] == "already_complete",
            "temporary_destination_removed": True,
        }
    costs = json.loads(args.release_costs.read_text())
    result = {
        "format": "replica.intrinsic_memory.archive_migration_preparation.v16",
        "inventory": inventory,
        "current_archive": {key: value for key, value in current.items() if key != "inventory"},
        "existing_release_costs": {
            "path": str(args.release_costs), "sha256": sha256(args.release_costs),
            "rebuilds": costs["rebuilds"], "totals": costs["totals"],
            "current_archive": costs["current_archive"],
        },
        "dry_run": dry_run,
        "migration_policy": {
            "source_mode": "read_only_never_mutated_or_deleted",
            "transaction_granularity": "one_verified_release_generation",
            "copy_order": ["payloads", "manifest", "independent_destination_verification", "pointer"],
            "pointer_commit": "atomic_after_all_payload_and_manifest_hashes_match",
            "resume": "verify existing exact files, skip duplicates, refuse mismatches, retain charged work",
            "large_scale_migration_executed": False,
        },
        "retention_policy": {
            "raw_source_and_unsuccessful_trajectories": "preserve",
            "historical_manifests_with_overwritten_payloads": "evidence_only_never_rollback_or_delete_as_part_of_migration",
            "rollback_eligibility": "only independently restorable generations with valid pointer, manifest and payload hashes",
            "required_online_retention": "current release plus previous independently restorable working release when available",
            "deletion": "separate explicit approval after independent verification; no deletion candidate authorized here",
            "deletion_performed": False,
        },
        "scope": {
            "native_model_executed": False,
            "initial_archive_baseline_is_earliest_retained_not_original_pretraining_snapshot": True,
            "large_scale_or_incremental_production_migration_measured": False,
            "formation_persistent_learning_density_and_RSI_unchanged_and_gated": True,
        },
    }
    if (not interrupted or pointer_before_recovery or recovery["status"] != "completed"
            or recovery["bytes_copied"] != 0 or recovery["files_copied"] != 0
            or not dry_run["pointer_manifest_matches_source"]
            or not dry_run["second_completion_idempotent"]):
        raise ValueError("migration dry run did not prove interrupted zero-recopy recovery")
    atomic(args.output.resolve(), result)
    print(json.dumps({
        "initial_logical": inventory["initial_logical"],
        "final_logical": inventory["final_logical"],
        "historical_invalid": len(inventory["historical_generations_not_independently_restorable"]),
        "current_archive_bytes": current["bytes"],
        "dry_run_recovery": recovery["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
