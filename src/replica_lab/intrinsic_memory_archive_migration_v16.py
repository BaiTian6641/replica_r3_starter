"""Read-only release inventory and interruption-safe migration preparation v16.

Migration writes only a fresh destination. Payloads and manifests are verified
before an atomic v12 authority pointer is committed; sources are never mutated.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

from .contextual_feature_identity_v12 import CODEC, REPRESENTATION
from .contextual_memory_v12 import digest
from .contextual_release_v12 import FORMAT as RELEASE_FORMAT
from .tiered_memory import MemoryScope
from .intrinsic_memory_budget_v12 import atomic

FORMAT = "replica.intrinsic_memory.archive_migration.v16"
PAYLOAD_NAME = re.compile(r"[0-9a-f]{64}\.slots\Z")
HEX_256 = re.compile(r"[0-9a-f]{64}\Z")


class MigrationInterrupted(RuntimeError):
    """Declared crash-injection boundary used only by tests/dry runs."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _regular_bytes(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or substituted release file")
    return path.read_bytes()


def _payload_name(value):
    if not isinstance(value, str) or not PAYLOAD_NAME.fullmatch(value):
        raise ValueError("release payload identity differs")
    return value


def _release_files(source, *, require_pointer):
    source = Path(source)
    manifest_bytes = _regular_bytes(source / "manifest.json")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    generation = manifest.get("generation")
    if type(generation) is not int or generation < 1 or source.name != f"{generation:08d}":
        raise ValueError("release generation invalid")
    scope = MemoryScope(**manifest.get("scope", {}))
    if (manifest.get("format") != RELEASE_FORMAT
            or not isinstance(manifest.get("scope"), dict)
            or manifest.get("representation") != REPRESENTATION
            or manifest.get("codec") != CODEC
            or not isinstance(manifest.get("producer_release"), str)
            or not manifest["producer_release"]
            or manifest.get("record_digest") != digest(manifest.get("records"))):
        raise ValueError("release manifest identity differs")
    records = manifest["records"]
    if not records:
        raise ValueError("release records missing")
    payloads = {}
    for record in records.values():
        if record.get("producer_release") != manifest["producer_release"]:
            raise ValueError("mixed producer releases refused")
        name = _payload_name(record.get("value_file"))
        expected_hash = record.get("value_sha256")
        if not isinstance(expected_hash, str) or not HEX_256.fullmatch(expected_hash):
            raise ValueError("release payload digest identity differs")
        payload = _regular_bytes(source / name)
        if len(payload) != record.get("value_bytes") or hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError("release payload identity differs")
        payloads[name] = expected_hash
    pointer = pointer_hash = None
    if require_pointer:
        pointer_bytes = _regular_bytes(source.parent / "current.json")
        pointer_hash = hashlib.sha256(pointer_bytes).hexdigest()
        pointer = json.loads(pointer_bytes)
        if (pointer.get("format") != RELEASE_FORMAT + ".pointer"
                or type(pointer.get("generation")) is not int
                or pointer["generation"] != generation
                or MemoryScope(**pointer.get("scope", {})).scope_key != scope.scope_key
                or type(pointer.get("e_epoch")) is not int or pointer["e_epoch"] < 0
                or pointer.get("manifest_sha256") != manifest_hash):
            raise ValueError("release pointer scope/format/authority differs")
    return {
        "manifest": manifest, "manifest_hash": manifest_hash,
        "manifest_path": source / "manifest.json", "payloads": payloads,
        "pointer": pointer, "pointer_hash": pointer_hash,
    }


def _atomic_copy(source, destination, expected_hash):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink() or sha256(destination) != expected_hash:
            raise ValueError("migration destination differs from source")
        return False
    _regular_bytes(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    digest_builder = hashlib.sha256()
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                digest_builder.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if digest_builder.hexdigest() != expected_hash:
            raise ValueError("source content changed during migration copy")
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    if not destination.is_file() or destination.is_symlink() or sha256(destination) != expected_hash:
        raise ValueError("destination verification failed after copy")
    return True


def _same_release(left, right):
    return (left["manifest_hash"] == right["manifest_hash"]
            and left["pointer_hash"] == right["pointer_hash"]
            and left["payloads"] == right["payloads"])


def migrate_generation(source, destination, *, interrupt_after=None):
    """Copy one verified generation and commit its v12 pointer last; safe to resume."""
    source, destination = Path(source), Path(destination)
    started = time.monotonic()
    original = _release_files(source, require_pointer=True)
    pointer_path = destination / "current.json"
    if pointer_path.exists():
        committed = _release_files(destination / source.name, require_pointer=True)
        if not _same_release(original, committed):
            raise ValueError("committed destination release differs")
        return {"status": "already_complete", "bytes_copied": 0, "files_copied": 0,
                "manifest_sha256": original["manifest_hash"]}
    destination.mkdir(parents=True, exist_ok=True)
    journal_path = destination / "migration.json"
    identity = {"source": str(source), "manifest_sha256": original["manifest_hash"],
                "pointer_sha256": original["pointer_hash"]}
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if any(journal.get(key) != value for key, value in identity.items()):
            raise ValueError("migration journal identity differs")
    else:
        atomic(journal_path, {"format": FORMAT + ".journal", "state": "prepared", **identity})
    copied_bytes = copied_files = 0
    generation = destination / source.name
    for name, expected_hash in original["payloads"].items():
        source_file = source / name
        if _atomic_copy(source_file, generation / name, expected_hash):
            copied_files += 1
            copied_bytes += source_file.stat().st_size
    atomic(journal_path, {"format": FORMAT + ".journal", "state": "payloads_copied", **identity})
    if interrupt_after == "payloads":
        raise MigrationInterrupted("after payloads before manifest")
    if _atomic_copy(original["manifest_path"], generation / "manifest.json",
                    original["manifest_hash"]):
        copied_files += 1
        copied_bytes += original["manifest_path"].stat().st_size
    atomic(journal_path, {"format": FORMAT + ".journal", "state": "manifest_copied", **identity})
    if interrupt_after == "manifest":
        raise MigrationInterrupted("after manifest before pointer")
    # Revalidate immutable identities before authority. A source mutation or a
    # destination substitution fails closed and cannot create a valid pointer.
    if not _same_release(original, _release_files(source, require_pointer=True)):
        raise ValueError("source release changed during migration")
    _release_files(generation, require_pointer=False)
    _atomic_copy(source.parent / "current.json", pointer_path, original["pointer_hash"])
    _release_files(generation, require_pointer=True)
    atomic(journal_path, {"format": FORMAT + ".journal", "state": "completed", **identity})
    return {"status": "completed", "bytes_copied": copied_bytes, "files_copied": copied_files,
            "manifest_sha256": original["manifest_hash"], "wall_seconds": time.monotonic() - started}


def _logical_generation(source):
    manifest_path = source / "manifest.json"
    manifest_hash = sha256(manifest_path) if manifest_path.is_file() and not manifest_path.is_symlink() else None
    generation = int(source.name) if source.name.isdigit() else None
    try:
        release = _release_files(source, require_pointer=False)
        restorable = True
        manifest = release["manifest"]
        names = list(release["payloads"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        restorable = False
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            manifest = {}
        names = []
        for record in manifest.get("records", {}).values():
            try:
                names.append(_payload_name(record.get("value_file")))
            except ValueError:
                continue
    files = [manifest_path] + [source / name for name in dict.fromkeys(names)]
    return {
        "generation": generation, "scope": manifest.get("scope"),
        "manifest_sha256": manifest_hash,
        "bytes": sum(path.stat().st_size for path in files
                     if path.is_file() and not path.is_symlink()),
        "files": sum(path.is_file() and not path.is_symlink() for path in files),
        "restorable": restorable,
    }


def inventory_archive(run):
    """Measure initial/current logical generations without modifying the archive."""
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
        pointer = json.loads(_regular_bytes(pointer_path)) if pointer_path.exists() else None
        pointer_generation = (pointer.get("generation")
                              if pointer and type(pointer.get("generation")) is int else None)
        final = next((row for row in measured if row["generation"] == pointer_generation),
                     measured[-1])
        try:
            pointer_scope_key = (MemoryScope(**pointer.get("scope", {})).scope_key
                                 if pointer else None)
            final_scope_key = MemoryScope(**final["scope"]).scope_key
        except (TypeError, ValueError):
            pointer_scope_key = final_scope_key = None
        pointer_valid = bool(pointer and pointer.get("format") == RELEASE_FORMAT + ".pointer"
                             and type(pointer.get("generation")) is int
                             and pointer_generation == final["generation"]
                             and pointer_scope_key is not None
                             and pointer_scope_key == final_scope_key
                             and type(pointer.get("e_epoch")) is int and pointer["e_epoch"] >= 0
                             and pointer.get("manifest_sha256") == final["manifest_sha256"]
                             and final["restorable"])
        rows.append({
            "world": root.parts[-5], "release_root": str(root),
            "initial": measured[0], "final": final,
            "generations": measured, "current_pointer_valid": pointer_valid,
        })
    if not rows or not any(row["current_pointer_valid"] for row in rows):
        raise ValueError("no independently restorable committed release generation")
    quarantined = [
        {"world": row["world"], "generation": row["final"]["generation"],
         "reason": "invalid_current_authority_or_payload"}
        for row in rows if not row["current_pointer_valid"]
    ]
    historical_invalid = [
        {"world": row["world"], "generation": generation["generation"]}
        for row in rows for generation in row["generations"]
        if generation is not row["final"] and not generation["restorable"]
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
        "quarantined_generations": quarantined,
        "current_pointer_valid_generations": sum(row["current_pointer_valid"] for row in rows),
        "scope": {
            "initial_is_earliest_retained_not_historical_archive_baseline": True,
            "final_is_current_pointer_release_including_rollback": True,
            "physical_disk_blocks_or_write_amplification_measured": False,
            "source_archive_modified": False,
        },
        "rows": rows,
    }
