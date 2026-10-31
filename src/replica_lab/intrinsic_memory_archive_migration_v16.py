"""Read-only release inventory and interruption-safe migration preparation v16.

Migration writes only a fresh destination. Payloads and manifests are verified
before an atomic pointer commit; source archives are never mutated or deleted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .intrinsic_memory_budget_v12 import atomic

FORMAT = "replica.intrinsic_memory.archive_migration.v16"


class MigrationInterrupted(RuntimeError):
    """Declared crash-injection boundary used only by tests/dry runs."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_copy(source, destination):
    source, destination = Path(source), Path(destination)
    expected = sha256(source)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink() or sha256(destination) != expected:
            raise ValueError("migration destination differs from source")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _release_files(source):
    manifest_path = Path(source) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "replica.contextual_producer_release.v12":
        raise ValueError("unsupported release format")
    records = manifest.get("records", {})
    if not records:
        raise ValueError("release records missing")
    payloads = sorted({record["value_file"] for record in records.values()})
    for record in records.values():
        path = source / record["value_file"]
        if not path.is_file() or path.is_symlink():
            raise ValueError("release payload missing or substituted")
        if path.stat().st_size != record["value_bytes"] or sha256(path) != record["value_sha256"]:
            raise ValueError("release payload identity differs")
    return manifest, manifest_path, payloads


def migrate_generation(source, destination, *, interrupt_after=None):
    """Copy one verified generation and commit its pointer last; safe to resume."""
    source, destination = Path(source), Path(destination)
    started = time.monotonic()
    manifest, manifest_path, payloads = _release_files(source)
    manifest_hash = sha256(manifest_path)
    pointer_path = destination / "current.json"
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text())
        if pointer.get("manifest_sha256") != manifest_hash:
            raise ValueError("destination already points to a different release")
        for name in ["manifest.json", *payloads]:
            if sha256(destination / source.name / name) != sha256(source / name):
                raise ValueError("committed destination payload differs")
        return {"status": "already_complete", "bytes_copied": 0, "files_copied": 0,
                "manifest_sha256": manifest_hash}
    destination.mkdir(parents=True, exist_ok=True)
    journal_path = destination / "migration.json"
    identity = {"source": str(source), "manifest_sha256": manifest_hash}
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal.get("source") != identity["source"] or journal.get("manifest_sha256") != manifest_hash:
            raise ValueError("migration journal identity differs")
    else:
        atomic(journal_path, {"format": FORMAT + ".journal", "state": "prepared", **identity})
    copied_bytes = copied_files = 0
    generation = destination / source.name
    for name in payloads:
        source_file = source / name
        if _atomic_copy(source_file, generation / name):
            copied_files += 1
            copied_bytes += source_file.stat().st_size
    atomic(journal_path, {"format": FORMAT + ".journal", "state": "payloads_copied", **identity})
    if interrupt_after == "payloads":
        raise MigrationInterrupted("after payloads before manifest")
    if _atomic_copy(manifest_path, generation / "manifest.json"):
        copied_files += 1
        copied_bytes += manifest_path.stat().st_size
    atomic(journal_path, {"format": FORMAT + ".journal", "state": "manifest_copied", **identity})
    if interrupt_after == "manifest":
        raise MigrationInterrupted("after manifest before pointer")
    for name in ["manifest.json", *payloads]:
        if sha256(generation / name) != sha256(source / name):
            raise ValueError("destination verification failed before pointer")
    pointer = {
        "format": FORMAT + ".pointer", "generation": manifest["generation"],
        "manifest_sha256": manifest_hash,
        "source_format": manifest["format"],
    }
    atomic(pointer_path, pointer)
    atomic(journal_path, {"format": FORMAT + ".journal", "state": "completed", **identity})
    return {"status": "completed", "bytes_copied": copied_bytes, "files_copied": copied_files,
            "manifest_sha256": manifest_hash, "wall_seconds": time.monotonic() - started}


def _logical_generation(source):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records", {})
    names = sorted({record.get("value_file") for record in records.values()})
    files = [manifest_path] + [source / name for name in names if name]
    valid = True
    for record in records.values():
        path = source / record["value_file"]
        valid = valid and path.is_file() and not path.is_symlink() \
            and path.stat().st_size == record["value_bytes"] \
            and sha256(path) == record["value_sha256"]
    return {
        "generation": manifest.get("generation"),
        "manifest_sha256": sha256(manifest_path),
        "bytes": sum(path.stat().st_size for path in files if path.is_file()),
        "files": sum(path.is_file() for path in files),
        "restorable": valid,
    }


def inventory_archive(run):
    """Measure initial/final logical generations without modifying the archive."""
    run = Path(run)
    roots = sorted(run.glob("stores/*/artifacts/contextual_v12/releases/*"))
    if not roots:
        raise ValueError("no contextual release archives found")
    rows = []
    for root in roots:
        generations = sorted((path for path in root.iterdir()
                              if path.is_dir() and len(path.name) == 8 and path.name.isdigit()),
                             key=lambda path: path.name)
        if not generations:
            continue
        measured = [_logical_generation(path) for path in generations]
        pointer_path = root / "current.json"
        pointer = json.loads(pointer_path.read_text()) if pointer_path.is_file() else None
        final = measured[-1]
        pointer_valid = bool(pointer and pointer.get("generation") == final["generation"]
                             and pointer.get("manifest_sha256") == final["manifest_sha256"]
                             and final["restorable"])
        rows.append({
            "world": root.parts[-5], "release_root": str(root),
            "initial": measured[0], "final": final,
            "generations": measured, "current_pointer_valid": pointer_valid,
        })
    if not rows or any(not row["current_pointer_valid"] for row in rows):
        raise ValueError("current release authority or payload is not independently restorable")
    historical_invalid = [
        {"world": row["world"], "generation": generation["generation"]}
        for row in rows for generation in row["generations"][:-1] if not generation["restorable"]
    ]
    return {
        "format": FORMAT + ".inventory", "run_root": str(run),
        "stores": len(rows),
        "initial_logical": {
            "bytes": sum(row["initial"]["bytes"] for row in rows),
            "files": sum(row["initial"]["files"] for row in rows),
            "restorable_generations": sum(row["initial"]["restorable"] for row in rows),
        },
        "final_logical": {
            "bytes": sum(row["final"]["bytes"] for row in rows),
            "files": sum(row["final"]["files"] for row in rows),
            "restorable_generations": sum(row["final"]["restorable"] for row in rows),
        },
        "historical_generations_not_independently_restorable": historical_invalid,
        "current_pointer_valid_generations": sum(row["current_pointer_valid"] for row in rows),
        "scope": {
            "initial_is_earliest_retained_not_historical_archive_baseline": True,
            "final_is_current_pointer_release": True,
            "physical_disk_blocks_or_write_amplification_measured": False,
        },
        "rows": rows,
    }
