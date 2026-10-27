"""Immutable historical source snapshots and reviewed current-source receipts.

Historical diagnostics were authored against a moving source tree.  Their
configs record the implementation bytes that existed when the receipt was
produced, but those bytes are not the live bytes forever.  This module keeps the
two checks explicit and fail-closed:

* ``historical`` replay verifies every pinned implementation path against a
  self-contained snapshot manifest whose blobs are content-addressed and whose
  bytes are re-hashed on every load.  Live source drift cannot change the
  verdict.
* ``current`` qualification requires a separate reviewed receipt.  The receipt
  is self-digested, binds the prepared protocol/manifest bytes and the exact
  config revision, must cover exactly the live implementation drift, and must
  pin every declared new file.  A missing, stale or tampered receipt fails.

The module is torch-free so CPU-only replay tests do not import native code.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

SNAPSHOT_FORMAT = "replica.implementation_source_snapshot.v1"
CURRENT_RECEIPT_FORMAT = "replica.implementation_current_receipt.v1"
CURRENT_RECEIPT_MODE = "reviewed_current_source"
MODES = ("historical", "current")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_FIELDS = {"format", "created_at_utc", "commit", "reason", "entries"}
_SNAPSHOT_ENTRY_FIELDS = {"path", "sha256", "bytes", "blob"}
_CURRENT_FIELDS = {"format", "mode", "declared_at_utc", "reason", "prepared",
                   "configs", "migrations", "new_files", "receipt_digest"}
_MIGRATION_FIELDS = {"parent_sha256", "current_sha256", "reason"}
_NEW_FILE_FIELDS = {"pinned_sha256", "reason"}


class SourceProvenanceError(RuntimeError):
    """A historical snapshot or current receipt cannot authenticate bytes."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceProvenanceError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_digest(value) -> str:
    return sha256_bytes(canonical_json(value))


def receipt_digest(receipt) -> str:
    """Digest the receipt body excluding its own ``receipt_digest`` field."""
    require(isinstance(receipt, dict), "Current receipt must be an object")
    body = {key: value for key, value in receipt.items()
            if key != "receipt_digest"}
    return canonical_digest(body)


def is_sha256(value) -> bool:
    return isinstance(value, str) and bool(_SHA256.match(value))


def require_sha256(value, label: str) -> None:
    require(is_sha256(value), f"{label} is not a lowercase SHA-256 digest")


def safe_relative(value, label: str) -> Path:
    """Resolve one relative POSIX path without permitting escape or absolutes."""
    require(isinstance(value, str) and bool(value),
            f"{label} is not a nonempty relative path")
    require("\\" not in value, f"{label} contains a backslash: {value!r}")
    require(not PurePosixPath(value).is_absolute(),
            f"{label} is absolute: {value!r}")
    parts = value.split("/")
    require(all(part not in ("", ".", "..") for part in parts),
            f"{label} is unsafe: {value!r}")
    return Path(*parts)


def load_json_object(path: str | Path, label: str) -> dict:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SourceProvenanceError(f"{label} cannot be read: {error}") from error
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def load_source_snapshot(path: str | Path) -> dict:
    """Load and fully re-hash one immutable source snapshot manifest."""
    path = Path(path)
    if path.is_dir():
        path = path / "source_snapshot.json"
    manifest = load_json_object(path, "Historical source snapshot")
    require(set(manifest) == _SNAPSHOT_FIELDS,
            "Historical source snapshot fields differ: "
            f"{sorted(set(manifest) ^ _SNAPSHOT_FIELDS)}")
    require(manifest["format"] == SNAPSHOT_FORMAT,
            "Historical source snapshot format differs")
    require(isinstance(manifest["reason"], str) and manifest["reason"].strip(),
            "Historical source snapshot lacks a reason")
    require(isinstance(manifest["created_at_utc"], str)
            and bool(manifest["created_at_utc"]),
            "Historical source snapshot lacks a creation time")
    require(isinstance(manifest["commit"], str) and bool(manifest["commit"]),
            "Historical source snapshot lacks a source commit")
    entries = manifest["entries"]
    require(isinstance(entries, list) and bool(entries),
            "Historical source snapshot has no entries")
    root = path.parent
    by_path: dict[str, dict[str, dict]] = {}
    for entry in entries:
        require(isinstance(entry, dict)
                and set(entry) == _SNAPSHOT_ENTRY_FIELDS,
                "Historical source snapshot entry is malformed")
        relative = safe_relative(entry["path"], "Snapshot path")
        require_sha256(entry["sha256"], f"Snapshot pin {entry['path']}")
        require(isinstance(entry["bytes"], int)
                and not isinstance(entry["bytes"], bool)
                and entry["bytes"] > 0,
                f"Snapshot byte size is invalid: {entry['path']}")
        blob_relative = safe_relative(entry["blob"], "Snapshot blob")
        blob = root / blob_relative
        require(blob.is_file(),
                f"Historical source snapshot blob is missing: {entry['blob']}")
        value = blob.read_bytes()
        require(len(value) == entry["bytes"],
                f"Historical source snapshot byte size differs: {entry['path']}")
        require(sha256_bytes(value) == entry["sha256"],
                f"Historical source snapshot bytes differ: {entry['path']}")
        key = relative.as_posix()
        pin = entry["sha256"]
        require(pin not in by_path.get(key, {}),
                f"Historical source snapshot duplicates a pin: {key}")
        by_path.setdefault(key, {})[pin] = {
            "sha256": pin, "bytes": entry["bytes"],
            "blob": blob_relative.as_posix(),
        }
    return {
        "path": path,
        "sha256": sha256_file(path),
        "manifest": manifest,
        "by_path": by_path,
    }


def historical_pin(snapshot: dict, relative: str, expected_sha256: str) -> str:
    """Return the authenticated snapshot hash for one expected pinned byte set."""
    require(is_sha256(expected_sha256), f"Pinned hash is invalid: {relative}")
    entries = snapshot["by_path"].get(relative) or {}
    require(expected_sha256 in entries,
            f"Historical source snapshot lacks pinned bytes: {relative}")
    return expected_sha256


def historical_map(snapshot: dict, paths) -> dict[str, str]:
    """Return the unique snapshot hash for each path, rejecting ambiguity."""
    result = {}
    for relative in paths:
        entries = snapshot["by_path"].get(relative) or {}
        require(len(entries) == 1,
                f"Historical source snapshot does not pin exactly one version: "
                f"{relative}")
        result[relative] = next(iter(entries))
    return result


def compute_implementation_map(root: str | Path, paths) -> dict[str, str]:
    root = Path(root)
    result = {}
    for relative in paths:
        safe = safe_relative(relative, "Implementation path")
        path = root / safe
        require(path.is_file(), f"Implementation file is missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def load_current_receipt(path: str | Path) -> dict:
    """Load one reviewed current-source receipt and re-check its self-digest."""
    path = Path(path)
    receipt = load_json_object(path, "Current implementation receipt")
    require(set(receipt) == _CURRENT_FIELDS,
            "Current implementation receipt fields differ: "
            f"{sorted(set(receipt) ^ _CURRENT_FIELDS)}")
    require(receipt["format"] == CURRENT_RECEIPT_FORMAT,
            "Current implementation receipt format differs")
    require(receipt["mode"] == CURRENT_RECEIPT_MODE,
            "Current implementation receipt must be explicitly reviewed")
    require(isinstance(receipt["reason"], str) and receipt["reason"].strip(),
            "Current implementation receipt lacks a reason")
    require(receipt_digest(receipt) == receipt["receipt_digest"],
            "Current implementation receipt digest differs")
    prepared = receipt["prepared"]
    require(isinstance(prepared, dict)
            and set(prepared) == {"protocol_sha256", "manifest_sha256"},
            "Current implementation receipt prepared pins are malformed")
    require_sha256(prepared["protocol_sha256"],
                   "Current receipt protocol_sha256")
    require_sha256(prepared["manifest_sha256"],
                   "Current receipt manifest_sha256")
    configs = receipt["configs"]
    require(isinstance(configs, dict) and bool(configs),
            "Current implementation receipt pins no configs")
    for name, value in configs.items():
        require(isinstance(name, str) and name,
                "Current implementation receipt config name is invalid")
        require_sha256(value, f"Current receipt config {name}")
    migrations = receipt["migrations"]
    require(isinstance(migrations, dict),
            "Current implementation receipt migrations are malformed")
    for relative, migration in migrations.items():
        safe_relative(relative, "Current receipt migration path")
        require(isinstance(migration, dict)
                and set(migration) == _MIGRATION_FIELDS,
                f"Current implementation migration is malformed: {relative}")
        require_sha256(migration["parent_sha256"],
                       f"Current migration parent {relative}")
        require_sha256(migration["current_sha256"],
                       f"Current migration current {relative}")
        require(isinstance(migration["reason"], str)
                and migration["reason"].strip(),
                f"Current implementation migration lacks a reason: {relative}")
    new_files = receipt["new_files"]
    require(isinstance(new_files, dict) and bool(new_files),
            "Current implementation receipt pins no new files")
    for relative, entry in new_files.items():
        safe_relative(relative, "Current receipt new-file path")
        require(isinstance(entry, dict)
                and set(entry) == _NEW_FILE_FIELDS,
                f"Current implementation new-file pin is malformed: {relative}")
        require_sha256(entry["pinned_sha256"],
                       f"Current new-file pin {relative}")
        require(isinstance(entry["reason"], str) and entry["reason"].strip(),
                f"Current implementation new-file pin lacks a reason: {relative}")
    return receipt


def verify_current_source(receipt_path: str | Path, *, root: str | Path,
                          prepared_pins: dict, protocol_sha256: str,
                          manifest_sha256: str, config_name: str,
                          config_body_sha256: str,
                          config_new_files) -> dict:
    """Authenticate live bytes against one reviewed current-source receipt."""
    receipt_file = Path(receipt_path)
    receipt = load_current_receipt(receipt_file)
    prepared = receipt["prepared"]
    require(prepared["protocol_sha256"] == protocol_sha256,
            "Current implementation receipt pins different prepared protocol bytes")
    require(prepared["manifest_sha256"] == manifest_sha256,
            "Current implementation receipt pins different prepared manifest bytes")
    configs = receipt["configs"]
    require(config_name in configs,
            f"Current implementation receipt is not reviewed for config: "
            f"{config_name}")
    require(configs[config_name] == config_body_sha256,
            "Current implementation receipt does not pin this config revision: "
            f"{config_name}")
    live = compute_implementation_map(root, prepared_pins)
    drift = {relative for relative in prepared_pins
             if live[relative] != prepared_pins[relative]}
    migrations = receipt["migrations"]
    require(set(migrations) == drift,
            "Current implementation receipt does not cover exactly the live "
            f"implementation drift: {sorted(set(migrations) ^ drift)}")
    for relative in sorted(drift):
        migration = migrations[relative]
        require(migration["parent_sha256"] == prepared_pins[relative],
                f"Current implementation receipt parent is stale: {relative}")
        require(migration["current_sha256"] == live[relative],
                f"Current implementation receipt current pin is stale: "
                f"{relative}")
    new_files = receipt["new_files"]
    resolved_new = {}
    for relative, entry in sorted(new_files.items()):
        candidate = (Path(root) / safe_relative(
            relative, "Current receipt new-file path"))
        require(candidate.is_file(),
                f"Current implementation file is missing: {relative}")
        actual = sha256_file(candidate)
        require(actual == entry["pinned_sha256"],
                f"Current implementation bytes differ: {relative}")
        resolved_new[relative] = {
            "pinned_sha256": entry["pinned_sha256"],
            "actual_sha256": actual,
            "reason": entry["reason"],
        }
    missing = set(config_new_files) - set(new_files)
    require(not missing,
            "Current implementation receipt lacks reviewed new-file pins: "
            f"{sorted(missing)}")
    return {
        "source_mode": "current_receipt",
        "receipt_path": str(receipt_file),
        "receipt_sha256": sha256_file(receipt_file),
        "live": live,
        "drift": sorted(drift),
        "migrations": {relative: dict(migrations[relative])
                       for relative in sorted(drift)},
        "new_files": resolved_new,
    }


def verify_declared_live(root: str | Path, *, prepared_pins: dict,
                         migrations: dict, new_files: dict) -> dict:
    """Preserve the pre-existing live check for callers without an explicit mode.

    New callers must select ``historical`` (immutable snapshot) or ``current``
    (reviewed receipt).  This route exists only so additive consumers such as
    the v2 accounting runner keep their previous declared-pin semantics until
    they migrate; it never claims a historical or reviewed-current verdict.
    """
    expected = dict(prepared_pins)
    for relative, migration in migrations.items():
        require(relative in prepared_pins,
                f"Implementation migration names an unpinned file: {relative}")
        expected[relative] = migration["current_sha256"]
    live = compute_implementation_map(root, expected)
    for relative in sorted(expected):
        if relative in migrations:
            require(live[relative] == expected[relative],
                    f"Migrated implementation bytes differ: {relative}")
        else:
            require(live[relative] == expected[relative],
                    f"Implementation bytes changed: {relative}")
    resolved_new = {}
    for relative, pinned in sorted(new_files.items()):
        path = Path(root) / safe_relative(relative, "Declared new-file path")
        require(path.is_file(),
                f"New implementation file is missing: {relative}")
        actual = sha256_file(path)
        require(actual == pinned,
                f"New implementation bytes differ: {relative}")
        resolved_new[relative] = actual
    return {
        "source_mode": "declared_live",
        "snapshot_path": None,
        "snapshot_sha256": None,
        "receipt_sha256": None,
        "actual": live,
        "new_files": resolved_new,
    }


def verify_historical_source(snapshot_path: str | Path, *,
                             prepared_pins: dict, migrations: dict,
                             new_files: dict) -> dict:
    """Authenticate declared historical bytes against the immutable snapshot."""
    snapshot = load_source_snapshot(snapshot_path)
    expected = dict(prepared_pins)
    for relative, migration in migrations.items():
        require(relative in prepared_pins,
                f"Implementation migration names an unpinned file: {relative}")
        expected[relative] = migration["current_sha256"]
    actual = {}
    for relative in sorted(expected):
        try:
            actual[relative] = historical_pin(
                snapshot, relative, expected[relative])
        except SourceProvenanceError as error:
            if relative in migrations:
                raise SourceProvenanceError(
                    f"Migrated implementation bytes differ: {relative}") from error
            raise
    resolved_new = {}
    for relative, pinned in sorted(new_files.items()):
        try:
            resolved_new[relative] = historical_pin(snapshot, relative, pinned)
        except SourceProvenanceError as error:
            raise SourceProvenanceError(
                f"New implementation bytes differ: {relative}") from error
    return {
        "source_mode": "historical_snapshot",
        "snapshot_path": str(snapshot["path"]),
        "snapshot_sha256": snapshot["sha256"],
        "actual": actual,
        "new_files": resolved_new,
    }
