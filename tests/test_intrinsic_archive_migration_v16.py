"""Interruption, consumer and security contracts for archive migration v16."""
import json
from pathlib import Path

import pytest

from replica_lab.contextual_memory_v12 import digest
from replica_lab.contextual_release_v12 import FORMAT as RELEASE_FORMAT
from replica_lab.intrinsic_memory_archive_migration_v16 import (
    MigrationInterrupted, inventory_archive, migrate_generation, sha256)

PAYLOAD = "a" * 64 + ".slots"
SCOPE = {
    "subject_id": "world", "branch_id": "main", "purpose": "engineering",
    "as_of": 1000, "mode": "assistant_for_source", "snapshot_at": None,
}


def write_pointer(root, manifest_path, generation):
    pointer = {
        "format": RELEASE_FORMAT + ".pointer", "scope": SCOPE,
        "generation": generation, "manifest_sha256": sha256(manifest_path), "e_epoch": 0,
    }
    (root / "current.json").write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")


def make_release(root, generation, payload, *, value_hash=None):
    root.mkdir(parents=True)
    payload_path = root / PAYLOAD
    payload_path.write_bytes(payload)
    records = {
        "record": {
            "producer_release": "v12-test", "value_file": PAYLOAD,
            "value_bytes": len(payload), "value_sha256": value_hash or sha256(payload_path),
        },
    }
    manifest = {
        "format": RELEASE_FORMAT, "generation": generation, "scope": SCOPE,
        "producer_release": "v12-test", "record_digest": digest(records), "records": records,
        "representation": "four_ordered_frozen_source_contextual_256_v12",
        "codec": "contextual_source_features_fp32_slots_v12", "source_identity": "source",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_pointer(root.parent, manifest_path, generation)
    return manifest


def test_interrupted_migration_resumes_without_pointer_or_duplicate_copy(tmp_path):
    source = tmp_path / "source/00000001"
    make_release(source, 1, b"payload")
    destination = tmp_path / "destination"
    with pytest.raises(MigrationInterrupted, match="before pointer"):
        migrate_generation(source, destination, interrupt_after="manifest")
    assert not (destination / "current.json").exists()
    assert (destination / "00000001" / PAYLOAD).read_bytes() == b"payload"
    result = migrate_generation(source, destination)
    assert result["status"] == "completed"
    assert result["bytes_copied"] == result["files_copied"] == 0
    pointer = json.loads((destination / "current.json").read_text())
    assert pointer["format"] == RELEASE_FORMAT + ".pointer"
    assert pointer["scope"] == SCOPE and pointer["e_epoch"] == 0
    assert pointer["manifest_sha256"] == sha256(source / "manifest.json")
    assert migrate_generation(source, destination)["status"] == "already_complete"


def test_real_release_consumer_opens_migrated_authority(tmp_path):
    from replica_lab.contextual_release_v12 import ProducerReleaseCorpus
    from replica_lab.contextual_v12_fixture import (
        build_v12_stack, canonical_span, open_v12_store, scope)

    _, _, _, path = build_v12_stack(seed=17, layers=8, sites=(3, 7), slots=8,
                                    vocab_size=64, objective_identity={"scope": "migration"})
    store = open_v12_store(tmp_path)
    sc = scope("world")
    try:
        path.admit(store, canonical_span("the badge is red", record_id="record0")[0], scope=sc)
        corpus = ProducerReleaseCorpus(store, sc)
        corpus.rebuild(path)
        destination = tmp_path / "migrated"
        migrate_generation(corpus.root / "00000001", destination)
        consumer = ProducerReleaseCorpus(store, sc)
        consumer.root = destination
        pointer, manifest = consumer._current()
        assert manifest["generation"] == pointer["generation"] == 1
        record = manifest["records"]["record0.span0"]
        assert consumer._payload_path(manifest, record).is_file()
    finally:
        path.close()
        store.close()


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
    (destination / "00000001" / PAYLOAD).write_bytes(b"damage")
    with pytest.raises(ValueError, match="payload identity"):
        migrate_generation(valid, destination)


def test_payload_path_traversal_is_refused_before_any_copy(tmp_path):
    source = tmp_path / "source/00000001"
    make_release(source, 1, b"payload")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"]["record"]["value_file"] = "../escape.slots"
    manifest["record_digest"] = digest(manifest["records"])
    manifest_path.write_text(json.dumps(manifest))
    write_pointer(source.parent, manifest_path, 1)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="payload identity"):
        migrate_generation(source, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape.slots").exists()


def test_source_mutation_during_copy_cannot_commit(tmp_path, monkeypatch):
    import replica_lab.intrinsic_memory_archive_migration_v16 as migration

    source = tmp_path / "source/00000001"
    make_release(source, 1, b"payload")
    destination = tmp_path / "destination"
    original = migration._atomic_copy

    def mutate(path, target, expected):
        Path(path).write_bytes(b"changed")
        return original(path, target, expected)

    monkeypatch.setattr(migration, "_atomic_copy", mutate)
    with pytest.raises(ValueError, match="changed during migration copy"):
        migrate_generation(source, destination)
    assert not (destination / "current.json").exists()


def test_symlinked_committed_payload_is_not_independent(tmp_path):
    source = tmp_path / "source/00000001"
    make_release(source, 1, b"payload")
    destination = tmp_path / "destination"
    migrate_generation(source, destination)
    copied = destination / "00000001" / PAYLOAD
    copied.unlink()
    copied.symlink_to(source / PAYLOAD)
    with pytest.raises(ValueError, match="missing or substituted"):
        migrate_generation(source, destination)


def test_inventory_supports_rollback_and_rejects_missing_authority(tmp_path):
    run = tmp_path / "run"
    root = run / "stores/world/artifacts/contextual_v12/releases/scope"
    first = root / "00000001"
    second = root / "00000002"
    make_release(first, 1, b"old")
    make_release(second, 2, b"new")
    # A valid rollback points to the older generation, not the numerical latest.
    write_pointer(root, first / "manifest.json", 1)
    result = inventory_archive(run)
    assert result["final_logical"]["restorable_generations"] == 1
    assert result["rows"][0]["final"]["generation"] == 1
    pointer = json.loads((root / "current.json").read_text())
    del pointer["e_epoch"]
    (root / "current.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="current release authority"):
        inventory_archive(run)




def test_malformed_scope_is_rejected_by_migration_and_inventory(tmp_path):
    source = tmp_path / "source/00000001"
    make_release(source, 1, b"payload")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scope"] = {}
    manifest_path.write_text(json.dumps(manifest))
    pointer = json.loads((source.parent / "current.json").read_text())
    pointer["scope"] = {}
    (source.parent / "current.json").write_text(json.dumps(pointer))
    with pytest.raises((TypeError, ValueError)):
        migrate_generation(source, tmp_path / "destination")
    run = tmp_path / "run"
    root = run / "stores/world/artifacts/contextual_v12/releases/scope"
    make_release(root / "00000001", 1, b"payload")
    pointer = json.loads((root / "current.json").read_text())
    pointer["scope"] = {}
    (root / "current.json").write_text(json.dumps(pointer))
    with pytest.raises((TypeError, ValueError)):
        inventory_archive(run)


def test_boolean_zero_and_noncanonical_generations_are_rejected(tmp_path):
    source = tmp_path / "source/00000001"
    make_release(source, 1, b"payload")
    pointer = json.loads((source.parent / "current.json").read_text())
    pointer["generation"] = True
    (source.parent / "current.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="scope/format/authority"):
        migrate_generation(source, tmp_path / "destination")
    zero = tmp_path / "zero/00000000"
    make_release(zero, 0, b"payload")
    with pytest.raises(ValueError, match="generation invalid"):
        migrate_generation(zero, tmp_path / "zero-destination")
    noncanonical = tmp_path / "noncanonical/1"
    make_release(noncanonical, 1, b"payload")
    with pytest.raises(ValueError, match="generation invalid"):
        migrate_generation(noncanonical, tmp_path / "noncanonical-destination")
    run = tmp_path / "run"
    root = run / "stores/world/artifacts/contextual_v12/releases/scope"
    make_release(root / "00000001", 1, b"payload")
    pointer = json.loads((root / "current.json").read_text())
    pointer["generation"] = True
    (root / "current.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="current release authority"):
        inventory_archive(run)
def test_inventory_marks_overwritten_historical_payload_unrestorable(tmp_path):
    run = tmp_path / "run"
    root = run / "stores/world/artifacts/contextual_v12/releases/scope"
    first = root / "00000001"
    second = root / "00000002"
    make_release(first, 1, b"old")
    make_release(second, 2, b"new")
    (first / PAYLOAD).write_bytes(b"new")
    result = inventory_archive(run)
    assert result["initial_logical"]["restorable_generations"] == 0
    assert result["final_logical"]["restorable_generations"] == 1
    assert result["historical_generations_not_independently_restorable"] == [
        {"world": "world", "generation": 1}]
    assert result["final_logical"]["bytes"] == len((second / "manifest.json").read_bytes()) + 3
