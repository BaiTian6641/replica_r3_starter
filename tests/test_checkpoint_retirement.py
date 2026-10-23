"""Tiny-file checks for the reviewed post-D2 retirement and orphan boundaries."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "checkpoint_retirement", SCRIPTS / "retire_completed_checkpoints.py"
)
retirement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retirement)
recovery_spec = importlib.util.spec_from_file_location(
    "checkpoint_retirement_recovery", SCRIPTS / "finish_checkpoint_retirement_pool.py"
)
recovery = importlib.util.module_from_spec(recovery_spec)
recovery_spec.loader.exec_module(recovery)


def pool_weight(pool, contents, *, digest=None):
    digest = digest or hashlib.sha256(contents).hexdigest()
    path = pool / "sha256" / digest[:2] / (digest + ".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def identity(path):
    return retirement.safe.cow._snapshot(path)


def test_exact_reviewed_scope_is_accepted():
    retirement.validate_scope(retirement.CANDIDATES, retirement.PROTECTED)


@pytest.mark.parametrize("change", ["omitted", "added", "outside", "absolute", "protected"])
def test_scope_changes_are_rejected(change):
    deletions = list(retirement.CANDIDATES)
    protected = list(retirement.PROTECTED)
    if change == "omitted":
        deletions.pop()
    elif change == "added":
        deletions.append("runs/another_run/checkpoint")
    elif change == "outside":
        deletions[0] = "runs/../../outside"
    elif change == "absolute":
        deletions[0] = "/tmp/checkpoint"
    else:
        protected.pop()
    with pytest.raises(ValueError, match="fixed reviewed"):
        retirement.validate_scope(deletions, protected)


@pytest.mark.parametrize("protected", ["runs/run/checkpoint", "runs/run", "runs/run/checkpoint/base"])
def test_even_fixed_scope_cannot_overlap_protected(monkeypatch, protected):
    candidate = "runs/run/checkpoint"
    monkeypatch.setattr(retirement, "CANDIDATES", {candidate: "fixture"})
    monkeypatch.setattr(retirement, "PROTECTED", (protected,))
    with pytest.raises(ValueError, match="overlaps protected"):
        retirement.validate_scope([candidate], [protected])


def test_live_pool_hash_is_never_removed(tmp_path):
    pool = tmp_path / ".checkpoint_storage"
    path = pool_weight(pool, b"retained model bytes")
    before = identity(path)
    with pytest.raises(ValueError, match="exact orphan"):
        retirement.retire_pool_file(path, before, {path.stem}, pool=pool)
    assert identity(path) == before


def test_changed_orphan_identity_refuses_unlink(tmp_path):
    pool = tmp_path / ".checkpoint_storage"
    path = pool_weight(pool, b"obsolete model bytes")
    before = identity(path)
    path.write_bytes(b"different model bytes")
    with pytest.raises(ValueError, match="File changed"):
        retirement.retire_pool_file(path, before, set(), pool=pool)
    assert path.read_bytes() == b"different model bytes"


def test_replaced_orphan_inode_refuses_unlink(tmp_path):
    pool = tmp_path / ".checkpoint_storage"
    path = pool_weight(pool, b"obsolete model bytes")
    before = identity(path)
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(ValueError, match="File changed"):
        retirement.retire_pool_file(path, before, set(), pool=pool)
    assert path.read_bytes() == b"obsolete model bytes"


def test_symlink_orphan_refuses_unlink_and_preserves_target(tmp_path):
    pool = tmp_path / ".checkpoint_storage"
    target = tmp_path / "protected-model"
    target.write_bytes(b"retained")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    path = pool / "sha256" / digest[:2] / (digest + ".safetensors")
    path.parent.mkdir(parents=True)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        retirement.retire_pool_file(path, identity(target), set(), pool=pool)
    assert path.is_symlink() and target.read_bytes() == b"retained"


def test_corrupt_content_address_is_not_removed(tmp_path):
    pool = tmp_path / ".checkpoint_storage"
    path = pool_weight(pool, b"wrong contents", digest="0" * 64)
    before = identity(path)
    with pytest.raises(ValueError, match="content address"):
        retirement.retire_pool_file(path, before, set(), pool=pool)
    assert identity(path) == before


@pytest.mark.parametrize("form", ["outside_pool", "wrong_prefix", "unexpected_filename"])
def test_noncanonical_pool_path_is_rejected(tmp_path, form):
    pool = tmp_path / ".checkpoint_storage"
    data = b"obsolete model bytes"
    digest = hashlib.sha256(data).hexdigest()
    if form == "outside_pool":
        path = tmp_path / "other" / "sha256" / digest[:2] / (digest + ".safetensors")
    elif form == "wrong_prefix":
        path = pool / "sha256" / "zz" / (digest + ".safetensors")
    else:
        path = pool / "sha256" / digest[:2] / "model.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="exact orphan"):
        retirement.retire_pool_file(path, identity(path), set(), pool=pool)
    assert path.read_bytes() == data


def same_prefix_payloads():
    seen = {}
    for number in range(1024):
        payload = f"tiny native fixture {number}".encode()
        digest = hashlib.sha256(payload).hexdigest()
        prefix = digest[:2]
        if prefix in seen:
            return seen[prefix], payload
        seen[prefix] = payload
    raise AssertionError("Expected a two-hex-digit SHA prefix collision")


def test_one_orphan_unlink_keeps_same_prefix_entry_and_self_contained_copy(tmp_path):
    pool = tmp_path / ".checkpoint_storage"
    obsolete, retained = same_prefix_payloads()
    orphan = pool_weight(pool, obsolete)
    live = pool_weight(pool, retained)
    assert orphan.parent == live.parent and orphan != live
    saved = tmp_path / "runs/selected/checkpoint/base/model.safetensors"
    saved.parent.mkdir(parents=True)
    shutil.copy2(orphan, saved)
    saved_before, live_before = identity(saved), identity(live)

    retirement.retire_pool_file(orphan, identity(orphan), {live.stem}, pool=pool)

    assert not orphan.exists()
    assert live.read_bytes() == retained and identity(live) == live_before
    assert saved.read_bytes() == obsolete and identity(saved) == saved_before
    assert (pool / "locks" / (orphan.stem + ".lock")).is_file()


@pytest.fixture
def preservation_plan():
    return {
        "deletions": [{"path": "runs/retired/checkpoint"}],
        "pool": [
            {"path": "runs/.checkpoint_storage/sha256/aa/orphan.safetensors", "action": "retire_orphan"},
            {"path": "runs/.checkpoint_storage/sha256/bb/live.safetensors", "action": "keep_live"},
        ],
        "preserved_file_state": {
            "runs/selected/run_report.json": {"inode": 1, "size": 32},
            "checkpoints/minicpm5_source/model.safetensors": {"inode": 2, "size": 64},
            "runs/.checkpoint_storage/sha256/bb/live.safetensors": {"inode": 3, "size": 64},
        },
    }


@pytest.mark.parametrize("orphan_present", [True, False])
def test_preservation_accepts_only_planned_orphan_presence_or_absence(monkeypatch, preservation_plan, orphan_present):
    current = dict(preservation_plan["preserved_file_state"])
    if orphan_present:
        current[preservation_plan["pool"][0]["path"]] = {"inode": 4, "size": 64}

    def snapshot(selection):
        assert selection == {"deletions": preservation_plan["deletions"], "copies": []}
        return dict(current)

    monkeypatch.setattr(retirement.safe, "preserved_state", snapshot)
    retirement.check_preserved(preservation_plan)
    assert preservation_plan["pool"][1]["path"] in current


@pytest.mark.parametrize("path", [
    "runs/selected/run_report.json",
    "checkpoints/minicpm5_source/model.safetensors",
    "runs/.checkpoint_storage/sha256/bb/live.safetensors",
])
@pytest.mark.parametrize("change", ["changed", "missing"])
def test_preservation_refuses_retained_raw_source_or_live_pool_mutation(monkeypatch, preservation_plan, path, change):
    current = dict(preservation_plan["preserved_file_state"])
    if change == "changed":
        current[path] = {**current[path], "size": 999}
    else:
        current.pop(path)
    monkeypatch.setattr(retirement.safe, "preserved_state", lambda selection: dict(current))
    with pytest.raises(ValueError, match="Protected/source/raw file identities changed"):
        retirement.check_preserved(preservation_plan)


def test_preservation_refuses_unplanned_new_payload(monkeypatch, preservation_plan):
    current = dict(preservation_plan["preserved_file_state"])
    current["runs/new/checkpoint/base/model.safetensors"] = {"inode": 5, "size": 64}
    monkeypatch.setattr(retirement.safe, "preserved_state", lambda selection: dict(current))
    with pytest.raises(ValueError, match="Protected/source/raw file identities changed"):
        retirement.check_preserved(preservation_plan)


@pytest.fixture
def recovery_fixture(tmp_path):
    rows = []
    for number in range(14):
        digest = f"{number:064x}"
        name = f"runs/.checkpoint_storage/sha256/{digest[:2]}/{digest}.safetensors"
        row = {"path": name, "sha256": digest,
               "action": "retire_orphan" if number < 10 else "keep_live"}
        rows.append(row)
        if number:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"small pool fixture {number}".encode())
    deleted = [{"path": name, "logical_bytes": 64} for name in recovery.cleanup.CANDIDATES]
    plan = {"deletions": deleted, "protected": list(recovery.cleanup.PROTECTED), "pool": rows}
    initial = {
        "status": "partial_or_failed_inspect_journal",
        "deleted_payloads": deleted,
        "deleted_pool_files": [{"path": rows[0]["path"], "sha256": rows[0]["sha256"]}],
        "protected_byte_claims_verified": 35,
        "original_source": {"status": "verified_local_source"},
    }
    return tmp_path, plan, initial


def test_recovery_accepts_exact_19_payloads_one_orphan_and_returns_nine(recovery_fixture):
    root, plan, initial = recovery_fixture
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*.safetensors")}
    remaining = recovery.recovery_scope(plan, initial, root=root)
    assert len(remaining) == 9
    assert {r["path"] for r in remaining} == {r["path"] for r in plan["pool"][1:10]}
    assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*.safetensors")} == before


@pytest.mark.parametrize("change", ["status", "payload_count", "payload_scope", "no_orphan", "duplicate_orphan", "live_orphan"])
def test_recovery_rejects_wrong_recorded_19_plus_one_result(recovery_fixture, change):
    root, plan, initial = recovery_fixture
    if change == "status":
        initial["status"] = "completed_reviewed_retirement_and_orphan_pool_gc"
    elif change == "payload_count":
        initial["deleted_payloads"] = initial["deleted_payloads"][:-1]
    elif change == "payload_scope":
        initial["deleted_payloads"] = [*initial["deleted_payloads"][:-1], {"path": "runs/other/checkpoint"}]
    elif change == "no_orphan":
        initial["deleted_pool_files"] = []
    elif change == "duplicate_orphan":
        initial["deleted_pool_files"] *= 2
    else:
        initial["deleted_pool_files"] = [{"path": plan["pool"][-1]["path"]}]
    with pytest.raises(ValueError, match="recorded19\\+1|outside reviewed orphans"):
        recovery.recovery_scope(plan, initial, root=root)


@pytest.mark.parametrize("change", ["claim_count", "source_missing"])
def test_recovery_requires_prior_protected_and_source_verification(recovery_fixture, change):
    root, plan, initial = recovery_fixture
    if change == "claim_count":
        initial["protected_byte_claims_verified"] = 34
    else:
        initial["original_source"] = None
    with pytest.raises(ValueError, match="Missing completed protected-byte verification"):
        recovery.recovery_scope(plan, initial, root=root)


@pytest.mark.parametrize("form", ["payload", "pool"])
def test_recovery_rejects_reappeared_deleted_objects(recovery_fixture, form):
    root, plan, initial = recovery_fixture
    if form == "payload":
        path = root / initial["deleted_payloads"][0]["path"]
        path.mkdir(parents=True)
    else:
        path = root / initial["deleted_pool_files"][0]["path"]
        path.write_bytes(b"reappeared orphan")
    with pytest.raises(ValueError, match="reappeared"):
        recovery.recovery_scope(plan, initial, root=root)


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_recovery_rejects_unreviewed_pool_inventory(recovery_fixture, change):
    root, plan, initial = recovery_fixture
    if change == "extra":
        extra = root / "runs/.checkpoint_storage/sha256/ff/unreviewed.safetensors"
        extra.parent.mkdir()
        extra.write_bytes(b"unreviewed")
    else:
        (root / plan["pool"][1]["path"]).unlink()
    with pytest.raises(ValueError, match="Pool inventory changed"):
        recovery.recovery_scope(plan, initial, root=root)


def test_recovery_rejects_even_dangling_pool_symlink(recovery_fixture):
    root, plan, initial = recovery_fixture
    link = root / "runs/.checkpoint_storage/sha256/unreviewed-link"
    link.symlink_to(root / "missing-target")
    with pytest.raises(ValueError, match="symlink|Symlink"):
        recovery.recovery_scope(plan, initial, root=root)
    assert link.is_symlink()
