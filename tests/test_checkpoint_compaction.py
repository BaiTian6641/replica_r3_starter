"""Small file-only safety tests; mock copies do not measure APFS space savings."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct

import pytest

spec = importlib.util.spec_from_file_location(
    "checkpoint_compaction", Path(__file__).parents[1] / "scripts/compact_identical_checkpoints.py")
compaction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compaction)


def native_bytes(value=1.0):
    header = json.dumps({"model.embed_tokens.weight": {
        "dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", value)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    # Process-list safety itself is tested separately; fixtures are not running jobs.
    monkeypatch.setattr(compaction, "assert_quiescent", lambda roots: None)
    roots = [tmp_path / "runs" / f"minicpm5_D1_mask1_B8_lr1e3_chunk{chunk}_20261022"
             for chunk in ("01", "02")]
    for root in roots:
        root.mkdir(parents=True)
        (root / "run_report.json").write_text(json.dumps({"status": "completed_local_training_run"}))
    reference = roots[0] / "checkpoint/base/model.safetensors"
    target = roots[1] / "training_checkpoint_0016/base/model.safetensors"
    for path in (reference, target):
        path.parent.mkdir(parents=True)
        path.write_bytes(native_bytes())
    sidecar = target.parent.parent / "sidecars.safetensors"
    sidecar.write_bytes(b"unique learned sidecar")
    runtime = target.parent.parent / "runtime.pt"
    runtime.write_bytes(b"unique optimizer and RNG state")
    receipt = target.parent.parent / "receipt.json"
    receipt.write_text(json.dumps({"model_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
    return tmp_path, roots, reference, target, (sidecar, runtime, receipt)


def plan_for(tree):
    workspace, roots, reference, _, _ = tree
    return compaction.build_plan(roots, reference, workspace=workspace)


def mock_clone(source, destination):
    assert not destination.exists()
    shutil.copy2(source, destination)


def test_default_plan_changes_no_checkpoint_and_excludes_state(tree):
    _, _, reference, target, excluded = tree
    before = {p: (p.read_bytes(), p.stat().st_ino) for p in (reference, target, *excluded)}
    plan = plan_for(tree)
    report = compaction.execute(plan)
    assert report["status"] == "plan_only_no_checkpoint_mutation"
    assert not report["apply_requested"] and report["results"] == []
    assert report["matching_target_count"] == 1
    assert report["maximum_reclaimable_duplicate_bytes"] == target.stat().st_size
    assert {Path(item["path"]) for item in plan["files"]} == {reference, target}
    assert {p: (p.read_bytes(), p.stat().st_ino) for p in before} == before


def test_success_preserves_bytes_receipt_reference_mode_and_mtime(tree):
    _, _, reference, target, excluded = tree
    os.chmod(target, 0o640)
    os.utime(target, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    before = {p: p.read_bytes() for p in (reference, target, *excluded)}
    old_reference_inode, old_target_inode = reference.stat().st_ino, target.stat().st_ino
    report = compaction.execute(plan_for(tree), apply=True, clone=mock_clone)
    assert report["status"] == "completed_clone_compaction"
    assert len(report["results"]) == 1
    assert reference.stat().st_ino == old_reference_inode
    assert target.stat().st_ino not in (old_reference_inode, old_target_inode)
    assert reference.stat().st_nlink == target.stat().st_nlink == 1
    assert target.stat().st_mode & 0o777 == 0o640
    assert target.stat().st_mtime_ns == 1_600_000_000_000_000_000
    assert {p: p.read_bytes() for p in before} == before
    result = report["results"][0]
    assert result["sha256_before"] == result["sha256_after"]
    assert "disk_free_before_bytes" in report and "disk_free_after_bytes" in report
    assert not list(target.parent.glob(".checkpoint-compaction-*"))


@pytest.mark.parametrize("failure_call", [1, 2])
def test_clone_failure_preserves_original_and_cleans_temporary(tree, failure_call):
    _, _, _, target, _ = tree
    original, inode = target.read_bytes(), target.stat().st_ino
    calls = 0
    def fail(source, destination):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            destination.write_bytes(b"partial failed clone")
            raise OSError("cloning unsupported or failed")
        mock_clone(source, destination)
    report = compaction.execute(plan_for(tree), apply=True, clone=fail)
    assert report["status"] == "failed_compaction_original_preserved_for_failed_file"
    assert report["results"] == []
    assert target.read_bytes() == original and target.stat().st_ino == inode
    assert not list(target.parent.glob(".checkpoint-compaction-*"))


def test_bad_clone_bytes_are_never_installed(tree):
    _, _, _, target, _ = tree
    original, inode = target.read_bytes(), target.stat().st_ino
    def corrupt(source, destination):
        destination.write_bytes(native_bytes(2.0))
    report = compaction.execute(plan_for(tree), apply=True, clone=corrupt)
    assert "Cloned bytes failed verification" in report["error"]["message"]
    assert target.read_bytes() == original and target.stat().st_ino == inode


def test_hardlink_is_never_accepted_as_clone(tree):
    _, _, reference, target, _ = tree
    original = target.read_bytes()
    report = compaction.execute(plan_for(tree), apply=True, clone=os.link)
    assert report["status"].startswith("failed")
    assert target.read_bytes() == original
    assert reference.stat().st_nlink == target.stat().st_nlink == 1


def test_post_replace_verification_failure_rolls_back(tree, monkeypatch):
    _, _, _, target, _ = tree
    plan = plan_for(tree)
    original, inode = target.read_bytes(), target.stat().st_ino
    real_hash = compaction.sha256_file
    def fail_installed(path):
        digest = real_hash(path)
        return "0" * 64 if path == target and path.stat().st_ino != inode else digest
    monkeypatch.setattr(compaction, "sha256_file", fail_installed)
    report = compaction.execute(plan, apply=True, clone=mock_clone)
    assert report["status"] == "failed_compaction_original_preserved_for_failed_file"
    assert target.read_bytes() == original
    assert not list(target.parent.glob(".checkpoint-compaction-*"))


def test_failed_rollback_retains_original_recovery_clone(tree, monkeypatch):
    _, _, _, target, _ = tree
    plan = plan_for(tree)
    original, inode = target.read_bytes(), target.stat().st_ino
    real_hash, real_replace = compaction.sha256_file, compaction.os.replace
    def fail_installed(path):
        digest = real_hash(path)
        return "0" * 64 if path == target and path.stat().st_ino != inode else digest
    def fail_rollback(source, destination):
        if Path(source).name == "original":
            raise OSError("simulated OS rollback failure")
        return real_replace(source, destination)
    monkeypatch.setattr(compaction, "sha256_file", fail_installed)
    monkeypatch.setattr(compaction.os, "replace", fail_rollback)
    report = compaction.execute(plan, apply=True, clone=mock_clone)
    assert report["status"] == "failed_compaction_manual_recovery_required"
    recovery = list(target.parent.glob(".checkpoint-compaction-*/original"))
    assert len(recovery) == 1 and recovery[0].read_bytes() == original


@pytest.mark.parametrize("which", ["reference", "target"])
def test_changed_file_after_planning_is_not_overwritten(tree, which):
    _, _, reference, target, _ = tree
    plan = plan_for(tree)
    changed = reference if which == "reference" else target
    changed.write_bytes(native_bytes(3.0))
    report = compaction.execute(plan, apply=True, clone=mock_clone)
    assert report["status"].startswith("failed")
    assert changed.read_bytes() == native_bytes(3.0)


def test_same_size_different_bytes_are_reported_and_skipped(tree):
    _, _, _, target, _ = tree
    target.write_bytes(native_bytes(4.0))
    plan = plan_for(tree)
    assert plan["matching_target_count"] == 0
    assert next(item for item in plan["files"] if item["path"] == str(target))["action"] == "skip_different_bytes"
    report = compaction.execute(plan, apply=True, clone=lambda *args: pytest.fail("must not clone"))
    assert report["status"] == "completed_clone_compaction" and not report["results"]


def test_wrong_expected_audit_hash_fails_closed(tree):
    workspace, roots, reference, _, _ = tree
    with pytest.raises(ValueError, match="declared screen"):
        compaction.build_plan(roots, reference, workspace=workspace, expected_sha256="0" * 64)


@pytest.mark.parametrize("reference_kind", ["sidecar", "runtime", "sealed_source", "unselected_root", "parent_escape"])
def test_only_explicit_native_base_files_can_be_selected(tree, reference_kind):
    workspace, roots, reference, target, excluded = tree
    paths = {"sidecar": excluded[0], "runtime": excluded[1],
             "sealed_source": workspace / "checkpoints/source/model.safetensors",
             "unselected_root": target, "parent_escape": reference.parent / "../model.safetensors"}
    with pytest.raises(ValueError):
        compaction.build_plan(roots[:1], paths[reference_kind], workspace=workspace)


@pytest.mark.parametrize("root_kind", ["runs", "unknown", "incomplete"])
def test_broad_unknown_or_unfinished_run_roots_are_rejected(tree, root_kind):
    workspace, roots, reference, _, _ = tree
    root = workspace / "runs" if root_kind == "runs" else workspace / "runs/unknown"
    if root_kind == "unknown":
        root.mkdir()
    if root_kind == "incomplete":
        root = roots[0]
        (root / "run_report.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError):
        compaction.build_plan([root], reference, workspace=workspace)


@pytest.mark.parametrize("symlink_kind", ["file", "base", "run", "dangling_base"])
def test_symlinks_are_rejected_without_following(tree, symlink_kind):
    workspace, roots, reference, target, _ = tree
    if symlink_kind == "file":
        target.unlink()
        target.symlink_to(reference)
    elif symlink_kind == "base":
        target.unlink()
        target.parent.rmdir()
        target.parent.symlink_to(reference.parent, target_is_directory=True)
    elif symlink_kind == "dangling_base":
        target.unlink()
        target.parent.rmdir()
        target.parent.symlink_to(workspace / "missing", target_is_directory=True)
    else:
        saved = workspace / "saved_run"
        roots[1].rename(saved)
        roots[1].symlink_to(saved, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        compaction.build_plan(roots, reference, workspace=workspace)


def test_existing_hardlinks_are_rejected(tree):
    workspace, _, _, target, _ = tree
    os.link(target, workspace / "alias")
    with pytest.raises(ValueError, match="one hard-link"):
        plan_for(tree)


def test_native_header_rejects_sidecar_payload_even_at_native_path(tree):
    _, _, _, target, _ = tree
    header = json.dumps({"workspace.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0,4]}}).encode()
    target.write_bytes(struct.pack("<Q",len(header)) + header + b"1234")
    with pytest.raises(ValueError, match="native FP32"):
        plan_for(tree)


def test_visible_active_run_process_is_rejected(tmp_path):
    root = tmp_path / next(iter(compaction.ALLOWED_RUN_NAMES))
    listing = f"{os.getpid()} 0 python compact\n987654 0 python scripts/train_text.py --output {root}\n"
    with pytest.raises(ValueError, match="active PID 987654"):
        compaction.assert_quiescent([root], process_listing=listing)


def test_compactor_and_ancestors_are_not_misidentified_as_training(tmp_path):
    root = tmp_path / next(iter(compaction.ALLOWED_RUN_NAMES))
    listing = f"{os.getpid()} 222 python compact --run-root {root}\n222 0 shell compact {root}\n"
    compaction.assert_quiescent([root], process_listing=listing)


def test_clone_backend_has_no_non_macos_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(compaction.platform, "system", lambda: "Linux")
    with pytest.raises(OSError, match="macOS clonefile"):
        compaction.clonefile(tmp_path / "source", tmp_path / "destination")
    assert not (tmp_path / "destination").exists()


def test_cli_is_plan_only_unless_explicit_apply():
    parser = compaction.parser()
    args = parser.parse_args(["--run-root", "runs/example", "--reference", "runs/example/checkpoint/base/model.safetensors"])
    assert args.apply is False
    assert parser.parse_args(["--run-root", "runs/example", "--reference", "file", "--apply"]).apply is True
