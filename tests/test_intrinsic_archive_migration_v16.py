"""Interruption and retention contracts for archive migration v16."""
import json
from pathlib import Path

import pytest

from replica_lab.intrinsic_memory_archive_migration_v16 import (
    MigrationInterrupted, inventory_archive, migrate_generation, sha256)


def make_release(root, generation, payload, *, value_hash=None):
    root.mkdir(parents=True)
    payload_path = root / "values.slots"
    payload_path.write_bytes(payload)
    manifest = {
        "format": "replica.contextual_producer_release.v12",
        "generation": generation,
        "records": {
            "record": {
                "value_file": "values.slots", "value_bytes": len(payload),
                "value_sha256": value_hash or sha256(payload_path),
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def test_interrupted_migration_resumes_without_pointer_or_duplicate_copy(tmp_path):
    source = tmp_path / "source/00000001"
    make_release(source, 1, b"payload")
    destination = tmp_path / "destination"
    with pytest.raises(MigrationInterrupted, match="before pointer"):
        migrate_generation(source, destination, interrupt_after="manifest")
    assert not (destination / "current.json").exists()
    assert (destination / "00000001/values.slots").read_bytes() == b"payload"
    result = migrate_generation(source, destination)
    assert result["status"] == "completed"
    assert result["bytes_copied"] == 0  # exact files survived the interruption
    pointer = json.loads((destination / "current.json").read_text())
    assert pointer["manifest_sha256"] == sha256(source / "manifest.json")
    assert migrate_generation(source, destination)["status"] == "already_complete"


def test_invalid_source_or_destination_is_refused_before_new_authority(tmp_path):
    source = tmp_path / "source/00000001"
    make_release(source, 1, b"payload", value_hash="0" * 64)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="payload identity"):
        migrate_generation(source, destination)
    assert not destination.exists()
    valid = tmp_path / "valid/00000001"
    make_release(valid, 1, b"payload")
    migrate_generation(valid, destination)
    (destination / "00000001/values.slots").write_bytes(b"damage")
    with pytest.raises(ValueError, match="committed destination"):
        migrate_generation(valid, destination)


def test_inventory_distinguishes_earliest_manifest_from_restorable_current(tmp_path):
    run = tmp_path / "run"
    root = run / "stores/world/artifacts/contextual_v12/releases/scope"
    first = root / "00000001"
    second = root / "00000002"
    make_release(first, 1, b"old")
    make_release(second, 2, b"new")
    # The first manifest is retained evidence, but its same-named payload has
    # been replaced by generation two and is not independently restorable.
    (first / "values.slots").write_bytes(b"new")
    current = {
        "generation": 2, "manifest_sha256": sha256(second / "manifest.json"),
    }
    (root / "current.json").write_text(json.dumps(current))
    result = inventory_archive(run)
    assert result["initial_logical"]["restorable_generations"] == 0
    assert result["final_logical"]["restorable_generations"] == 1
    assert result["current_pointer_valid_generations"] == 1
    assert result["historical_generations_not_independently_restorable"] == [
        {"world": "world", "generation": 1}]
    assert result["final_logical"]["bytes"] == len((second / "manifest.json").read_bytes()) + 3
