#!/usr/bin/env python3
"""Plan byte-identical native checkpoint compaction; --apply uses APFS clones.

No model is loaded. Only explicit, quiescent run directories are eligible. The
reference is never replaced. Sidecars, runtime state and sealed sources are
outside the allowed path grammar. Atomicity/rollback is per file, not per batch.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 8 * 1024 * 1024
MODEL_FILE = re.compile(r"model(?:-\d+-of-\d+)?\.safetensors\Z")
CHECKPOINT_DIR = re.compile(r"(?:checkpoint|adapter_export|training_checkpoint_\d{4,})\Z")
SNAPSHOT_KEYS = ("device", "inode", "size", "mtime_ns", "ctime_ns", "links")
ALLOWED_RUN_NAMES = frozenset(
    f"minicpm5_D1_mask1_B8_lr{rate}_chunk{chunk}_20261022"
    for rate in ("1e3", "1e2") for chunk in ("01", "02")
) | {"minicpm5_augmented_generation_qualification256_20261022",
     "minicpm5_native_adapter_qualification256_20261022"}


class RollbackError(RuntimeError):
    """Keep the original rollback clone for manual recovery after an OS error."""


def _absolute(path: Path, workspace: Path) -> Path:
    if ".." in path.parts:
        raise ValueError(f"Parent traversal is forbidden: {path}")
    return path if path.is_absolute() else workspace / path


def _no_symlinks(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        if component.is_symlink():
            raise ValueError(f"Symlink is forbidden: {component}")


def _snapshot(path: Path) -> dict:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError(f"Expected a regular file with one hard-link name: {path}")
    return {"device": value.st_dev, "inode": value.st_ino,
            "size": value.st_size, "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns, "links": value.st_nlink,
            "allocated_bytes": value.st_blocks * 512}


def _unchanged(path: Path, expected: dict) -> None:
    actual = _snapshot(path)
    if any(actual[key] != expected[key] for key in SNAPSHOT_KEYS):
        raise ValueError(f"File changed during planning or compaction: {path}")


def sha256_file(path: Path) -> str:
    """Stream a stable, nonsymlink file and verify its identity after reading."""
    _no_symlinks(path)
    before = _snapshot(path)
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before["device"], before["inode"]):
            raise ValueError(f"File was replaced while opening: {path}")
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    _unchanged(path, before)
    return digest.hexdigest()


def _native_fp32_header(path: Path) -> None:
    """Header-only validation, with no tensor library or model allocation."""
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Missing safetensors header: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= length <= min(16 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header object: {path}")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors or any(
        not key.startswith(("model.", "lm_head.")) or not isinstance(value, dict)
        or value.get("dtype") != "F32" for key, value in tensors.items()
    ):
        raise ValueError(f"Expected native FP32 base tensors, not sidecars/state: {path}")


def validate_run_roots(run_roots: list[Path], workspace: Path) -> list[Path]:
    """Require named direct children of runs, never runs itself or sources."""
    workspace = workspace.resolve(strict=True)
    roots = []
    for item in run_roots:
        root = _absolute(Path(item), workspace)
        _no_symlinks(root)
        if (root.parent != workspace / "runs" or not root.is_dir()
                or root.name not in ALLOWED_RUN_NAMES):
            raise ValueError(f"Run root is not a known D1/qualification directory directly in runs/: {root}")
        report_name = "run_report.json" if "_D1_" in root.name else "qualification_report.json"
        report = root / report_name
        _no_symlinks(report)
        _snapshot(report)
        expected_status = ("completed_local_training_run" if "_D1_" in root.name else
                           "passed_native_augmented_generation_qualification" if "augmented" in root.name else
                           "passed_native_adapter_forward_qualification")
        if json.loads(report.read_text()).get("status") != expected_status:
            raise ValueError(f"Run does not have its expected completed report: {root}")
        if root not in roots:
            roots.append(root)
    if not roots:
        raise ValueError("At least one explicit run root is required")
    return roots


def assert_quiescent(roots: list[Path], *, process_listing: str | None = None) -> None:
    """Reject visible processes naming a selected run; this is not a writer lock."""
    if process_listing is None:
        process_listing = subprocess.check_output(
            ["/bin/ps", "-ww", "-axo", "pid=,ppid=,command="], text=True)
    processes = {}
    for line in process_listing.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            processes[int(fields[0])] = (int(fields[1]), fields[2])
    ancestors, current = set(), os.getpid()
    while current and current not in ancestors:
        ancestors.add(current)
        current = processes.get(current, (0, ""))[0]
    for pid, (_, command) in processes.items():
        if pid not in ancestors and any(root.name in command for root in roots):
            raise ValueError(f"Selected run is named by active PID {pid}; wait until it exits")


def validate_weight(path: Path, roots: list[Path]) -> Path:
    _no_symlinks(path)
    relative = next((path.relative_to(root) for root in roots if path.is_relative_to(root)), None)
    if (relative is None or len(relative.parts) != 3 or relative.parts[1] != "base"
            or not CHECKPOINT_DIR.fullmatch(relative.parts[0])
            or not MODEL_FILE.fullmatch(relative.name)):
        raise ValueError(f"Only checkpoint/base/model*.safetensors in explicit runs are eligible: {path}")
    _snapshot(path)
    _native_fp32_header(path)
    return path


def build_plan(run_roots: list[Path], reference: Path, *, workspace: Path = ROOT,
               expected_sha256: str | None = None) -> dict:
    workspace = workspace.resolve(strict=True)
    roots = validate_run_roots(run_roots, workspace)
    assert_quiescent(roots)
    reference = validate_weight(_absolute(reference, workspace), roots)
    reference_before = _snapshot(reference)
    reference_sha = sha256_file(reference)
    if expected_sha256 is not None and reference_sha != expected_sha256:
        raise ValueError("Reference SHA256 does not match the declared screen")
    files = []
    for root in roots:
        # No recursive selection outside these three known checkpoint forms.
        for directory in sorted(root.iterdir()):
            if not CHECKPOINT_DIR.fullmatch(directory.name):
                continue
            _no_symlinks(directory)
            base = directory / "base"
            _no_symlinks(base)
            if not base.exists():
                continue
            for path in sorted(base.glob("*.safetensors")):
                validate_weight(path, roots)
                before = _snapshot(path)
                if before["device"] != reference_before["device"]:
                    raise ValueError(f"Cross-filesystem clone is forbidden: {path}")
                item = {"path": str(path), "before": before}
                if path == reference:
                    item.update(sha256=reference_sha, action="keep_reference")
                elif before["size"] != reference_before["size"]:
                    item.update(sha256=None, action="skip_different_size")
                else:
                    digest = sha256_file(path)
                    item.update(sha256=digest, action=("clone_and_replace" if digest == reference_sha
                                                       else "skip_different_bytes"))
                _unchanged(path, before)
                files.append(item)
    _unchanged(reference, reference_before)
    if not any(item["path"] == str(reference) for item in files):
        raise ValueError("Reference was not discovered inside the selected roots")
    targets = [item for item in files if item["action"] == "clone_and_replace"]
    return {
        "schema_version": 1, "status": "plan_only_no_checkpoint_mutation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace), "run_roots": [str(root) for root in roots],
        "reference": str(reference), "reference_sha256": reference_sha,
        "reference_before": reference_before, "files": files,
        "matching_target_count": len(targets),
        "maximum_reclaimable_duplicate_bytes": sum(item["before"]["size"] for item in targets),
        "disk_free_before_bytes": shutil.disk_usage(workspace).free,
        "limitations": ["Requires quiescent checkpoints; not a lock against concurrent writers",
                        "Atomicity and rollback are per file; a batch may be partially completed",
                        "Duplicate logical bytes are an upper bound, not measured physical reclaim",
                        "Existing APFS clones, snapshots and open old files can reduce or delay reclaim",
                        "Inode and ctime change; content, target mode and mtime are preserved"],
    }


def clonefile(source: Path, destination: Path) -> None:
    """Public macOS clonefile(2), fail closed; no cp fallback or hard links."""
    if platform.system() != "Darwin":
        raise OSError("Apply requires macOS clonefile(2)")
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    function = library.clonefile
    function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    function.restype = ctypes.c_int
    # CLONE_NOFOLLOW; all ancestor components are checked separately as well.
    if function(os.fsencode(source), os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def compact_file(item: dict, plan: dict, *, clone=clonefile) -> dict:
    roots = validate_run_roots([Path(root) for root in plan["run_roots"]], Path(plan["workspace"]))
    assert_quiescent(roots)
    reference = validate_weight(Path(plan["reference"]), roots)
    target = validate_weight(Path(item["path"]), roots)
    if target == reference or item["action"] != "clone_and_replace":
        raise ValueError("Only planned nonreference identical targets can be replaced")
    expected_sha = plan["reference_sha256"]
    if item["sha256"] != expected_sha:
        raise ValueError("Plan does not declare cryptographic byte equality")
    _unchanged(reference, plan["reference_before"])
    _unchanged(target, item["before"])
    if sha256_file(target) != expected_sha:
        raise ValueError("Byte equality no longer holds before cloning")
    target_stat = target.stat()
    if target_stat.st_dev != reference.stat().st_dev:
        raise ValueError("Cross-filesystem clone is forbidden")
    if (target_stat.st_uid, target_stat.st_gid) != (reference.stat().st_uid, reference.stat().st_gid):
        raise ValueError("Reference and target ownership differ")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-compaction-", dir=target.parent))
    replacement, backup = temporary / "replacement", temporary / "original"
    replaced = False
    preserve_recovery_directory = False
    try:
        clone(reference, replacement)
        replacement_stat = _snapshot(replacement)
        if replacement_stat["inode"] in (plan["reference_before"]["inode"], item["before"]["inode"]):
            raise ValueError("Clone must have its own inode, never a hard link")
        if sha256_file(replacement) != expected_sha:
            raise ValueError("Cloned bytes failed verification")
        # Keep a COW rollback copy until post-replacement verification succeeds.
        clone(target, backup)
        if sha256_file(backup) != expected_sha:
            raise ValueError("Rollback clone failed verification")
        os.chmod(replacement, stat.S_IMODE(target_stat.st_mode))
        os.utime(replacement, ns=(target_stat.st_atime_ns, target_stat.st_mtime_ns))
        with replacement.open("rb") as stream:
            os.fsync(stream.fileno())
        _no_symlinks(target)
        _unchanged(reference, plan["reference_before"])
        _unchanged(target, item["before"])
        assert_quiescent(roots)
        os.replace(replacement, target)
        replaced = True
        if sha256_file(target) != expected_sha:
            raise ValueError("Post-replacement SHA256 differs")
        after = _snapshot(target)
        if after["inode"] == plan["reference_before"]["inode"]:
            raise ValueError("Post-replacement inode aliases reference")
        return {"path": str(target), "status": "cloned_replaced_verified",
                "sha256_before": expected_sha, "sha256_after": expected_sha,
                "before": item["before"], "after": after}
    except BaseException as error:
        if replaced:
            try:
                os.replace(backup, target)
            except Exception as rollback_error:
                preserve_recovery_directory = True
                raise RollbackError(
                    f"Rollback failed ({rollback_error}); original bytes retained at {backup}"
                ) from error
        raise
    finally:
        if not preserve_recovery_directory:
            shutil.rmtree(temporary)


def execute(plan: dict, *, apply: bool = False, clone=clonefile) -> dict:
    report = {**plan, "apply_requested": apply, "results": [],
              "files": [{**item, "replaced": False} for item in plan["files"]]}
    if apply:
        report["status"] = "completed_clone_compaction"
        for item in report["files"]:
            if item["action"] != "clone_and_replace":
                continue
            try:
                report["results"].append(compact_file(item, plan, clone=clone))
                item["replaced"] = True
            except Exception as error:
                report["status"] = ("failed_compaction_manual_recovery_required" if isinstance(error, RollbackError)
                                    else "failed_compaction_original_preserved_for_failed_file")
                report["error"] = {"path": item["path"], "type": type(error).__name__, "message": str(error)}
                break
    report["disk_free_after_bytes"] = shutil.disk_usage(plan["workspace"]).free
    report["observed_disk_free_change_bytes"] = report["disk_free_after_bytes"] - plan["disk_free_before_bytes"]
    report["observed_disk_free_change_note"] = "System-wide observation; not proof all change was caused by this operation"
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-root", type=Path, action="append", required=True,
                       help="Explicit direct child of this workspace's runs/; repeat to add roots")
    value.add_argument("--reference", type=Path, required=True,
                       help="Untouched canonical native weight file inside a selected run")
    value.add_argument("--expected-sha256", help="Optional known audit hash; full bytes are still verified")
    value.add_argument("--apply", action="store_true", help="Explicitly replace verified duplicates using macOS COW clones")
    value.add_argument("--output", type=Path, help="Fresh JSON report path; otherwise write JSON to stdout")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.output is not None and args.output.exists():
        raise SystemExit("Use a fresh output report path")
    plan = build_plan(args.run_root, args.reference, expected_sha256=args.expected_sha256)
    report = execute(plan, apply=args.apply)
    report["implementation_sha256"] = sha256_file(Path(__file__))
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(json.dumps({"report": str(args.output), "status": report["status"],
                          "matching_target_count": report["matching_target_count"],
                          "completed_replacements": len(report["results"])}))
    if report["status"].startswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
