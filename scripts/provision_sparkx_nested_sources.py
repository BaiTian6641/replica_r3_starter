#!/usr/bin/env python3
"""Provision the SparkX nested-learning source cache strictly offline.

Payloads are copied only from an explicitly supplied local directory. The
historical ``scratch_path``/``text_path`` values in ``evidence.json`` are not
read unless the caller passes ``--historical-scratch-root`` naming that root.
Nothing is downloaded.

Every input is hashed and size-checked, every staged copy is checked again, and
publication is atomic per file with no overwrite. A clean cache root is
published by one atomic directory rename. Existing matching files are skipped;
an existing mismatched file aborts the run rather than being replaced.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "reports/sparkx_nested_learning_20261026/verify.py"
DEFAULT_CONFIG = ROOT / "configs/sparkx_nested_learning_sources_v1.json"
DEFAULT_EVIDENCE = ROOT / "reports/sparkx_nested_learning_20261026/evidence.json"


class ProvisionError(RuntimeError):
    """An input is absent, unsafe, mismatched, oversized or would be overwritten."""


def _load_verifier():
    spec = importlib.util.spec_from_file_location("sparkx_nested_verify", VERIFY_PATH)
    if spec is None or spec.loader is None:
        raise ProvisionError(f"cannot load verifier from {VERIFY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sparkx_nested_verify", module)
    spec.loader.exec_module(module)
    return module


verify = _load_verifier()


def _resolved_root(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _checked_input(root: Path, candidate: Path, label: str, entry_name: str) -> Path:
    root = _resolved_root(root)
    candidate = Path(candidate)
    if candidate.is_symlink():
        raise ProvisionError(f"{entry_name}: refusing symlink {label} input {candidate}")
    if not candidate.exists():
        raise ProvisionError(f"{entry_name}: {label} input disappeared: {candidate}")
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ProvisionError(f"{entry_name}: {label} input escapes declared root: {candidate}")
    if not resolved.is_file():
        raise ProvisionError(f"{entry_name}: {label} input is not a regular file: {candidate}")
    return resolved


def _find_input(root: Path, directories: list[Path], names: list[str | None],
                label: str, entry_name: str) -> Path:
    seen_paths: set[str] = set()
    looked_names: list[str] = []
    for directory in directories:
        for name in names:
            if not name:
                continue
            candidate = Path(directory) / name
            key = str(candidate)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            if name not in looked_names:
                looked_names.append(name)
            if candidate.exists() or candidate.is_symlink():
                return _checked_input(root, candidate, label, entry_name)
    locations = ", ".join(str(directory) for directory in directories)
    raise ProvisionError(
        f"{entry_name}: no {label} payload under {locations}; looked for: {', '.join(looked_names)}"
    )


def _source_names(entry: dict) -> list[str]:
    return [
        entry["historical_source_basename"],
        f"{entry['sha256']}{entry['source_suffix']}",
        f"{entry['name']}{entry['source_suffix']}",
    ]


def _text_names(entry: dict) -> list[str]:
    return [
        entry["historical_text_basename"],
        f"{entry['text_key']}.txt",
        f"{entry['name']}.txt",
    ]


def _locate(root: Path, entry: dict, kind: str, config: dict) -> Path:
    layout = config["layout"]
    if kind == "source":
        names = _source_names(entry)
        directories = [Path(root), Path(root) / layout["sources_dir"]]
        label = "source"
    else:
        names = _text_names(entry)
        directories = [Path(root), Path(root) / layout["text_dir"]]
        label = "text"
    return _find_input(root, directories, names, label, entry["name"])


def _verify_source(path: Path, entry: dict, config: dict, enforce_limit: bool = True) -> None:
    size = path.stat().st_size
    if enforce_limit:
        limit = config["provisioning"]["max_file_bytes"]
        if size > limit:
            raise ProvisionError(f"{entry['name']}: source payload {size} bytes exceeds per-file budget {limit}")
    if size != entry["bytes"]:
        raise ProvisionError(
            f"{entry['name']}: source size mismatch: expected {entry['bytes']}, observed {size}"
        )
    digest = verify.sha256_file(path)
    if digest != entry["sha256"]:
        raise ProvisionError(
            f"{entry['name']}: source sha256 mismatch: expected {entry['sha256']}, observed {digest}"
        )


def _verify_text(path: Path, entry: dict, config: dict, enforce_limit: bool = True) -> None:
    size = path.stat().st_size
    if enforce_limit:
        limit = config["provisioning"]["max_file_bytes"]
        if size > limit:
            raise ProvisionError(f"{entry['name']}: text payload {size} bytes exceeds per-file budget {limit}")
    if size <= 0:
        raise ProvisionError(f"{entry['name']}: text payload is empty: {path}")
    declared = entry["declared_text_sha256"]
    if declared is not None:
        digest = verify.sha256_file(path)
        if digest != declared:
            raise ProvisionError(
                f"{entry['name']}: declared text sha256 mismatch: expected {declared}, observed {digest}"
            )


def _target_matches(path: Path, entry: dict, kind: str, config: dict,
                    expected_text_sha256: str | None = None) -> bool:
    if path.is_symlink():
        raise ProvisionError(f"refusing existing symlink at cache target: {path}")
    if not path.exists():
        return False
    if not path.is_file():
        raise ProvisionError(f"refusing non-file cache target: {path}")
    try:
        if kind == "source":
            _verify_source(path, entry, config, enforce_limit=False)
        else:
            _verify_text(path, entry, config, enforce_limit=False)
            if entry["declared_text_sha256"] is None and expected_text_sha256 is not None:
                if verify.sha256_file(path) != expected_text_sha256:
                    raise ProvisionError(
                        "text content differs from the selected input and has no declared hash"
                    )
    except ProvisionError as exc:
        raise ProvisionError(
            f"{entry['name']}: refusing to overwrite mismatched cache file {path}: {exc}"
        ) from exc
    return True


def _input_roots(from_dir: Path | None, historical_scratch_root: Path | None):
    roots: list[tuple[str, Path]] = []
    if from_dir is not None:
        roots.append(("user_declared_directory", Path(from_dir).expanduser()))
    if historical_scratch_root is not None:
        roots.append((
            "explicit_historical_scratch_root", Path(historical_scratch_root).expanduser()
        ))
    if not roots:
        raise ProvisionError(
            "an explicit local source root is required; pass --from-dir and/or "
            "--historical-scratch-root (historical evidence paths are provenance only)"
        )
    return roots


def _locate_in_roots(roots, entry: dict, kind: str, config: dict) -> Path:
    errors = []
    for kind_name, root in roots:
        if not root.is_dir():
            errors.append(f"{kind_name} {root} is not a directory")
            continue
        try:
            return _locate(root, entry, kind, config)
        except ProvisionError as exc:
            errors.append(str(exc))
    raise ProvisionError("; ".join(errors))


def _action_paths(entry: dict, cache_root: Path, config: dict) -> tuple[Path, Path, str, str]:
    layout = config["layout"]
    source_rel = (
        f"{layout['sources_dir']}/"
        f"{layout['source_filename_template'].format(sha256=entry['sha256'], source_suffix=entry['source_suffix'])}"
    )
    text_rel = (
        f"{layout['text_dir']}/"
        f"{layout['text_filename_template'].format(text_key=entry['text_key'])}"
    )
    source_target = verify.confined_cache_path(cache_root, source_rel)
    text_target = verify.confined_cache_path(cache_root, text_rel)
    return source_target, text_target, source_rel, text_rel


def _copy_verified(staged: Path, entry: dict, kind: str, config: dict) -> None:
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(entry["_input"], staged)
    if kind == "source":
        _verify_source(staged, entry, config)
    else:
        _verify_text(staged, entry, config)
        if entry["declared_text_sha256"] is None:
            if verify.sha256_file(staged) != verify.sha256_file(entry["_input"]):
                raise ProvisionError(f"{entry['name']}: staged text differs from the verified input")


def _publish_clean(stage: Path, cache_root: Path) -> None:
    parent = cache_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    if cache_root.exists():
        raise ProvisionError(f"cache root appeared during provisioning: {cache_root}")
    try:
        os.replace(stage, cache_root)
    except OSError as exc:
        raise ProvisionError(f"atomic cache publication failed for {cache_root}: {exc}") from exc


def _publish_existing(
    cache_root: Path, actions: list[tuple[dict, str, Path, str]]
) -> None:
    for entry, kind, staged, relative in actions:
        target = verify.confined_cache_path(cache_root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(staged, target)
        except FileExistsError as exc:
            raise ProvisionError(f"refusing to overwrite cache target that appeared: {target}") from exc
        except OSError as exc:
            raise ProvisionError(f"atomic per-file publication failed for {target}: {exc}") from exc


def provision(
    cache_root: Path,
    *,
    evidence: dict,
    config: dict,
    from_dir: Path | None = None,
    historical_scratch_root: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Copy, verify and publish all source/text payloads offline.

    ``cache_root`` is the only write location. Inputs are accepted only from
    ``from_dir`` or an explicitly supplied historical scratch root.
    """
    verify.validate_sources_config(config)
    entries = verify.build_source_plan(evidence, config)
    roots = _input_roots(from_dir, historical_scratch_root)
    cache_root = Path(cache_root).expanduser()

    located: list[dict] = []
    for entry in entries:
        source = _locate_in_roots(roots, entry, "source", config)
        text = _locate_in_roots(roots, entry, "text", config)
        _verify_source(source, entry, config)
        _verify_text(text, entry, config)
        located.append({**entry, "_source_input": source, "_text_input": text})

    actions: list[tuple[dict, str, Path, str]] = []
    skipped = 0
    write_bytes = 0
    max_file = config["provisioning"]["max_file_bytes"]
    budget = config["provisioning"]["max_total_write_bytes"]
    for entry in located:
        source_target, text_target, source_rel, text_rel = _action_paths(entry, cache_root, config)
        text_input_sha256 = verify.sha256_file(entry["_text_input"])
        if _target_matches(source_target, entry, "source", config):
            skipped += 1
        else:
            size = entry["_source_input"].stat().st_size
            if size > max_file:
                raise ProvisionError(f"{entry['name']}: source exceeds per-file budget")
            write_bytes += size
            actions.append(({**entry, "_input": entry["_source_input"]}, "source",
                            source_target, source_rel))
        if _target_matches(text_target, entry, "text", config,
                           expected_text_sha256=text_input_sha256):
            skipped += 1
        else:
            size = entry["_text_input"].stat().st_size
            if size > max_file:
                raise ProvisionError(f"{entry['name']}: text exceeds per-file budget")
            write_bytes += size
            actions.append(({**entry, "_input": entry["_text_input"]}, "text",
                            text_target, text_rel))

    if write_bytes > budget:
        raise ProvisionError(
            f"declared write budget exceeded: planned {write_bytes} bytes > {budget} bytes"
        )
    summary = {
        "cache_root": str(cache_root),
        "network_used": False,
        "overwrite_existing": False,
        "atomic_publication": True,
        "dry_run": bool(dry_run),
        "verified_sources": len(entries),
        "written_files": len(actions),
        "skipped_files": skipped,
        "write_bytes": write_bytes,
        "published": [relative for _, _, _, relative in actions],
    }
    if dry_run or not actions:
        return summary

    if not cache_root.exists():
        parent = cache_root.parent
        parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{cache_root.name}.staging-", dir=parent))
        try:
            for entry, kind, target, relative in actions:
                staged = stage / relative
                _copy_verified(staged, entry, kind, config)
            _publish_clean(stage, cache_root)
            stage = None
        finally:
            if stage is not None and stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
    else:
        stage_root = cache_root / config["layout"]["staging_dir"]
        stage_root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f"{os.getpid()}-{uuid.uuid4().hex[:8]}-", dir=stage_root))
        try:
            for entry, kind, target, relative in actions:
                staged = stage / relative
                _copy_verified(staged, entry, kind, config)
            _publish_existing(cache_root, actions)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            try:
                stage_root.rmdir()
            except OSError:
                pass
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", help="cache root override (default: config/env)")
    parser.add_argument("--from-dir", help="explicit user-declared local payload directory")
    parser.add_argument("--historical-scratch-root",
                        help="explicit historical scratch root chosen by the caller; never implied")
    parser.add_argument("--sources-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = verify.load_sources_config(Path(args.sources_config))
        evidence = verify.load_json(Path(args.evidence))
        cache_root = verify.resolve_cache_root(args.cache_root, config=config)
        summary = provision(
            cache_root,
            evidence=evidence,
            config=config,
            from_dir=Path(args.from_dir) if args.from_dir else None,
            historical_scratch_root=Path(args.historical_scratch_root)
            if args.historical_scratch_root else None,
            dry_run=args.dry_run,
        )
    except (ProvisionError, verify.SourcePayloadError, OSError, ValueError) as exc:
        print(f"provision error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
