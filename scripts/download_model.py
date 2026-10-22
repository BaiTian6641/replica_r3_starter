#!/usr/bin/env python3
"""Reuse a complete sealed local MiniCPM5 snapshot offline, without model loads.

Initial acquisition or explicit repair requires --allow-network. A valid local
snapshot is always reused without importing Hugging Face or rewriting its
acquisition receipt. Other source identities never share an output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from glob import escape as literal_pattern
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SELECTED_SUFFIXES = frozenset({".json", ".safetensors", ".model", ".jinja"})
CHUNK_BYTES = 8 * 1024 * 1024
IDENTITY_FIELDS = ("repo_id", "revision", "config_sha256")


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular JSON file: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_metadata(lock: dict) -> None:
    if (lock.get("repo_id") != "openbmb/MiniCPM5-2B"
            or not re.fullmatch(r"[0-9a-f]{40}", str(lock.get("revision", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(lock.get("config_sha256", "")))):
        raise ValueError("Requires an immutable MiniCPM5-2B metadata lock and config SHA256")


def _same_identity(left: dict, right: dict, *, description: str) -> None:
    if any(left.get(field) != right.get(field) for field in IDENTITY_FIELDS):
        raise ValueError(f"{description} source identity differs; choose a new output directory/lock")


def _regular_snapshot_file(root: Path, path: Path) -> None:
    if root.is_symlink() or not root.is_dir() or not path.is_relative_to(root):
        raise ValueError(f"Expected an existing nonsymlink local snapshot: {root}")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"Snapshot symlinks are not eligible for byte-sealed reuse: {item}")
        if item == root:
            break
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Expected a regular snapshot file: {path}")


def digest(path: Path) -> str:
    """Bounded streaming hash; never import tensor/model libraries."""
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            value.update(chunk)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f"Snapshot file changed while hashing: {path}")
    return value.hexdigest()


def _validate_config(snapshot: Path, lock: dict) -> dict:
    path = snapshot / "config.json"
    _regular_snapshot_file(snapshot, path)
    if digest(path) != lock["config_sha256"]:
        raise ValueError("Local configuration differs from the pinned metadata lock")
    config = _read_json(path)
    expected = {"model_type": "llama", "hidden_size": 2048,
                "num_hidden_layers": 42, "vocab_size": 130560,
                "tie_word_embeddings": False, "architectures": ["LlamaForCausalLM"]}
    if any(config.get(key) != value for key, value in expected.items()) or config.get("auto_map"):
        raise ValueError("Local configuration is not the supported native MiniCPM5 Llama model")
    if not any(path.is_file() for path in snapshot.glob("*.safetensors")):
        raise ValueError("A complete local safetensors checkpoint is required")
    return config


def _validate_seal_claim(metadata: dict, seal: dict) -> dict:
    """Invalid authority/manifest data cannot be repaired by downloading bytes."""
    _validate_metadata(seal)
    _same_identity(metadata, seal, description="Sealed lock")
    if seal.get("local_bytes_sealed") is not True or seal.get("weights_downloaded") is not True:
        raise ValueError("A completed local byte-sealed lock is required")
    sealed = seal.get("snapshot_files_sha256")
    if not isinstance(sealed, dict) or not sealed:
        raise ValueError("Seal local snapshot bytes with scripts/seal_snapshot.py first")
    if sealed.get("config.json") != metadata["config_sha256"]:
        raise ValueError("Sealed configuration hash differs from the metadata identity")
    for name, expected in sealed.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or PurePosixPath(name).as_posix() != name
                or not re.fullmatch(r"[0-9a-f]{64}", str(expected))):
            raise ValueError("Snapshot manifest contains an invalid path or SHA256")
    return sealed


def validate_local_seal(snapshot: Path, metadata: dict, seal: dict) -> dict:
    """Mirror the trainer's selected suffix inventory, identity and byte checks.

    The seal is a local reproducibility record, not a publisher signature. Extra
    files with unselected suffixes (such as HF lock files) are not seal members.
    """
    sealed = _validate_seal_claim(metadata, seal)
    _validate_config(snapshot, metadata)
    relevant = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*")
                if path.is_file() and path.suffix in SELECTED_SUFFIXES}
    if set(sealed) != relevant or "config.json" not in sealed:
        raise ValueError("Snapshot file inventory differs from its local byte seal")
    for relative, expected in sealed.items():
        path = snapshot / relative
        _regular_snapshot_file(snapshot, path)
        if digest(path) != expected:
            raise ValueError(f"Snapshot byte mismatch: {relative}")
    after = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*")
             if path.is_file() and path.suffix in SELECTED_SUFFIXES}
    if after != relevant:
        raise ValueError("Snapshot inventory changed during validation")
    return {"verified_files": len(sealed),
            "selected_inventory_suffixes": sorted(SELECTED_SUFFIXES)}


def damaged_sealed_members(snapshot: Path, metadata: dict, seal: dict) -> list[str]:
    """Identify only missing/corrupt sealed repository files for explicit repair."""
    sealed = _validate_seal_claim(metadata, seal)
    relevant = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*")
                if path.is_file() and path.suffix in SELECTED_SUFFIXES}
    extra = sorted(relevant - set(sealed))
    if extra:
        raise ValueError(f"Explicit repair refused: extra selected inventory files: {extra}")
    damaged = []
    generated_missing = []
    for name, expected in sealed.items():
        path = snapshot / name
        for item in (path, *path.parents):
            if item.is_symlink():
                raise ValueError(f"Explicit repair refused: snapshot symlink: {item}")
            if item == snapshot:
                break
        missing = not path.exists()
        if not missing:
            _regular_snapshot_file(snapshot, path)
        if missing or digest(path) != expected:
            if PurePosixPath(name).parts[0] == ".cache":
                # HF creates its tree cache during acquisition; it is not a remote repo file.
                if not missing:
                    raise ValueError(f"Explicit repair refused: corrupt local-generated cache metadata: {name}")
                generated_missing.append(name)
            else:
                damaged.append(name)
    if not damaged:
        detail = (f"missing local-generated cache metadata: {generated_missing}" if generated_missing else
                  "sealed bytes already match; receipt/configuration inconsistency needs investigation")
        raise ValueError(f"Explicit repair refused: {detail}")
    return sorted(damaged)


def _write_receipt(path: Path, receipt: dict) -> None:
    """Replace only after a complete JSON write, preserving failure provenance."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(receipt, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def ensure_snapshot(model_lock: Path, output: Path, *, sealed_lock: Path | None = None,
                    allow_network: bool = False) -> dict:
    metadata = _read_json(model_lock)
    _validate_metadata(metadata)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Output must be a nonsymlink local snapshot directory")
    receipt_path = output.parent / (output.name + ".download.json")
    previous = _read_json(receipt_path) if receipt_path.exists() or receipt_path.is_symlink() else None
    if previous is not None:
        if any(previous.get(key) != metadata[key] for key in ("repo_id", "revision")):
            raise ValueError("Output belongs to another revision; choose a new directory")
        if previous.get("config_sha256", metadata["config_sha256"]) != metadata["config_sha256"]:
            raise ValueError("Acquisition receipt configuration identity differs")
        if previous.get("status") not in {"downloading", "downloaded_not_loaded"}:
            raise ValueError("Acquisition receipt has an unknown status; preserve it and investigate")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Existing nonempty output has no acquisition receipt; choose a new directory")

    seal = None
    if sealed_lock is not None and (sealed_lock.exists() or sealed_lock.is_symlink()):
        seal = _read_json(sealed_lock)
    elif sealed_lock is None and "snapshot_files_sha256" in metadata:
        seal = metadata
    if seal is not None:
        # Invalid identities/manifests are never repaired by fetching bytes.
        _validate_seal_claim(metadata, seal)

    try:
        if previous is None or previous.get("status") != "downloaded_not_loaded" or not previous.get("completed_at"):
            raise ValueError("No completed acquisition receipt for offline source reuse")
        if seal is None:
            raise ValueError("No local byte seal is available for offline source reuse")
        verification = validate_local_seal(output, metadata, seal)
    except (OSError, ValueError) as error:
        if not allow_network:
            raise ValueError(f"Offline source reuse refused: {error}; no files or receipts changed. "
                             "Acquisition/repair requires an explicit --allow-network choice.") from error
    else:
        return {"status": "reused_local_sealed_snapshot", "repo_id": metadata["repo_id"],
                "revision": metadata["revision"], "snapshot": str(output),
                "receipt_path": str(receipt_path), "receipt_preserved": True,
                "network_used": False, **verification}

    repair_members = None
    if previous is not None and seal is not None:
        repair_members = damaged_sealed_members(output, metadata, seal)
    elif previous is not None and previous["status"] == "downloaded_not_loaded":
        raise ValueError("Explicit repair requires a valid local seal; no original files were reacquired")

    # The sole network/import branch. Default and successful reuse never enter it.
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    from huggingface_hub import snapshot_download

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(previous) if previous is not None else {
        "repo_id": metadata["repo_id"], "revision": metadata["revision"],
        "config_sha256": metadata["config_sha256"],
        "requested_at": datetime.now(timezone.utc).isoformat(), "status": "downloading"}
    if previous is None:
        # Existing receipts are preserved if an explicitly requested repair fails.
        _write_receipt(receipt_path, receipt)
    repair_options = ({"force_download": True,
                       "allow_patterns": [literal_pattern(name) for name in repair_members]}
                      if repair_members is not None else {})
    snapshot_download(repo_id=metadata["repo_id"], revision=metadata["revision"],
                      local_dir=output, cache_dir=ROOT / ".cache" / "huggingface" / "hub",
                      max_workers=2, **repair_options)
    _validate_config(output, metadata)
    if seal is not None:
        validate_local_seal(output, metadata, seal)
    completed_at = datetime.now(timezone.utc).isoformat()
    receipt.update(status="downloaded_not_loaded")
    if not receipt.get("completed_at"):
        receipt["completed_at"] = completed_at
    if previous is not None:
        receipt.update(last_network_completed_at=completed_at,
                       last_network_action="explicit_repair_or_resume")
        if repair_members is not None:
            receipt["last_repaired_sealed_members"] = repair_members
    _write_receipt(receipt_path, receipt)
    return {**receipt, "network_used": True, "snapshot": str(output),
            "receipt_path": str(receipt_path), "local_bytes_sealed": seal is not None,
            "forced_repair_members": repair_members}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model-lock", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    default_seal = ROOT / "configs" / "minicpm5_sealed_lock.local.json"
    value.add_argument("--sealed-lock", type=Path, default=default_seal if default_seal.exists() else None,
                       help="Local byte seal; defaults to the project's sealed lock when present")
    value.add_argument("--allow-network", action="store_true",
                       help="Explicitly allow initial acquisition or same-identity repair/resume")
    return value


def main(argv: list[str] | None = None) -> None:
    value = parser()
    args = value.parse_args(argv)
    try:
        result = ensure_snapshot(args.model_lock, args.output, sealed_lock=args.sealed_lock,
                                 allow_network=args.allow_network)
    except (OSError, ValueError) as error:
        value.error(str(error))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
