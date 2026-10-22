"""File-only COW storage tests; no pretrained models, downloads or GPU work."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import shutil
import struct
from types import SimpleNamespace

import pytest

from replica_lab import checkpoint_storage as storage


def weights(value=1.0):
    header = json.dumps({"model.embed_tokens.weight": {
        "dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", value)


def saved(tmp_path, name, value=1.0):
    base = tmp_path / "runs" / name / "base"
    base.mkdir(parents=True)
    (base / "model.safetensors").write_bytes(weights(value))
    return base


def pool(tmp_path):
    return tmp_path / "runs/.checkpoint_storage"


def mock_clone(source, destination):
    assert not destination.exists()
    shutil.copy2(source, destination)


def share(base, tmp_path, clone=mock_clone):
    return storage.deduplicate_native_weights(base, pool_directory=pool(tmp_path), clone=clone)


def canonicals(tmp_path):
    return sorted(pool(tmp_path).glob("sha256/*/*.safetensors"))


def test_identical_saves_share_one_pool_entry_without_paths_or_byte_changes(tmp_path):
    first, second = saved(tmp_path, "first"), saved(tmp_path, "second")
    os.chmod(second / "model.safetensors", 0o640)
    original_time = (second / "model.safetensors").stat().st_mtime_ns
    a, b = share(first, tmp_path), share(second, tmp_path)
    assert a["files"][0]["action"] == "published_pool_clone_of_saved_file"
    assert b["files"][0]["action"] == "reused_verified_pool_clone"
    assert len(canonicals(tmp_path)) == 1
    paths = [first / "model.safetensors", second / "model.safetensors", canonicals(tmp_path)[0]]
    assert len({path.stat().st_ino for path in paths}) == 3
    assert all(path.stat().st_nlink == 1 and path.read_bytes() == weights() for path in paths)
    assert paths[1].stat().st_mode & 0o777 == 0o640
    assert paths[1].stat().st_mtime_ns == original_time
    shutil.rmtree(pool(tmp_path))
    assert paths[0].read_bytes() == paths[1].read_bytes() == weights()


def test_changed_bytes_on_same_inode_publish_a_distinct_version(tmp_path):
    base = saved(tmp_path, "versions")
    path = base / "model.safetensors"
    share(base, tmp_path)
    initial_inode = path.stat().st_ino
    path.write_bytes(weights(2.0))
    assert path.stat().st_ino == initial_inode
    result = share(base, tmp_path)
    assert result["files"][0]["action"] == "published_pool_clone_of_saved_file"
    assert len(canonicals(tmp_path)) == 2
    assert {file.read_bytes() for file in canonicals(tmp_path)} == {weights(), weights(2.0)}


def test_all_native_shards_are_included_and_sidecars_runtime_are_independent(tmp_path):
    base = saved(tmp_path, "sharded")
    (base / "model.safetensors").rename(base / "model-00001-of-00002.safetensors")
    (base / "model-00002-of-00002.safetensors").write_bytes(weights(2.0))
    index = {"weight_map": {"left": "model-00001-of-00002.safetensors",
                            "right": "model-00002-of-00002.safetensors"}}
    (base / "model.safetensors.index.json").write_text(json.dumps(index))
    excluded = [base.parent / "sidecars.safetensors", base.parent / "runtime.pt", base / "config.json"]
    for path in excluded:
        path.write_bytes(b"independent metadata/state")
    before = [(path.read_bytes(), path.stat().st_ino) for path in excluded]
    result = share(base, tmp_path)
    assert len(result["files"]) == len(canonicals(tmp_path)) == 2
    assert [(path.read_bytes(), path.stat().st_ino) for path in excluded] == before
    assert json.loads((base / "model.safetensors.index.json").read_text()) == index


@pytest.mark.parametrize("failure", ["unsupported", "replacement_clone", "backup_clone", "replace"])
def test_ordinary_saved_bytes_survive_clone_or_replace_failure(tmp_path, monkeypatch, failure):
    first, second = saved(tmp_path, "first"), saved(tmp_path, "second")
    if failure != "unsupported":
        share(first, tmp_path)
    path = second / "model.safetensors"
    original_inode = path.stat().st_ino
    calls = 0
    def failing_clone(source, destination):
        nonlocal calls
        calls += 1
        if failure == "unsupported" or (failure == "replacement_clone" and calls == 1) or (failure == "backup_clone" and calls == 2):
            raise OSError("clone unavailable")
        mock_clone(source, destination)
    real_replace = storage.os.replace
    def failing_replace(source, destination):
        if failure == "replace" and destination == path:
            raise OSError("atomic replacement failed")
        return real_replace(source, destination)
    monkeypatch.setattr(storage.os, "replace", failing_replace)
    result = share(second, tmp_path, clone=failing_clone)
    assert result["files"][0]["action"] == "ordinary_save_retained"
    assert path.read_bytes() == weights() and path.stat().st_ino == original_inode
    assert not list(second.glob(".native-cow-*"))


def test_corrupt_pool_entry_is_never_installed(tmp_path):
    first, second = saved(tmp_path, "first"), saved(tmp_path, "second")
    share(first, tmp_path)
    canonical = canonicals(tmp_path)[0]
    canonical.chmod(0o644)
    canonical.write_bytes(weights(9.0))
    with pytest.raises(storage.StorageIntegrityError, match="pool entry is corrupt"):
        share(second, tmp_path)
    assert (second / "model.safetensors").read_bytes() == weights()


def test_corrupt_clone_is_never_installed(tmp_path):
    first, second = saved(tmp_path, "first"), saved(tmp_path, "second")
    share(first, tmp_path)
    def corrupt(source, destination):
        destination.write_bytes(weights(7.0))
    with pytest.raises(storage.StorageIntegrityError, match="clone bytes"):
        share(second, tmp_path, clone=corrupt)
    assert (second / "model.safetensors").read_bytes() == weights()


def test_post_replacement_verification_rolls_back_to_saved_bytes(tmp_path, monkeypatch):
    first, second = saved(tmp_path, "first"), saved(tmp_path, "second")
    share(first, tmp_path)
    path = second / "model.safetensors"
    original_inode = path.stat().st_ino
    real_hash = storage.file_sha256
    def fail_installed(file):
        result = real_hash(file)
        return "0" * 64 if file == path and file.stat().st_ino != original_inode else result
    monkeypatch.setattr(storage, "file_sha256", fail_installed)
    with pytest.raises(storage.StorageIntegrityError, match="Installed native weight"):
        share(second, tmp_path)
    assert path.read_bytes() == weights() and not list(second.glob(".native-cow-*"))


def test_temporary_cleanup_failure_prevents_publication(tmp_path, monkeypatch):
    first, second = saved(tmp_path, "first"), saved(tmp_path, "second")
    share(first, tmp_path)
    real_remove = storage.shutil.rmtree
    def failed_remove(path):
        if path.parent == second:
            raise OSError("simulated cleanup failure")
        return real_remove(path)
    monkeypatch.setattr(storage.shutil, "rmtree", failed_remove)
    with pytest.raises(storage.StorageIntegrityError, match="do not publish"):
        share(second, tmp_path)
    assert (second / "model.safetensors").read_bytes() == weights()


def test_hardlink_backend_is_rejected_and_ordinary_file_remains_independent(tmp_path):
    base = saved(tmp_path, "bad_backend")
    with pytest.raises(storage.StorageIntegrityError, match="independent regular"):
        share(base, tmp_path, clone=os.link)
    assert (base / "model.safetensors").read_bytes() == weights()
    assert (base / "model.safetensors").stat().st_nlink == 1


def test_pool_symlink_and_nested_pool_are_rejected(tmp_path):
    base = saved(tmp_path, "first")
    outside = tmp_path / "outside"
    outside.mkdir()
    pool(tmp_path).symlink_to(outside, target_is_directory=True)
    with pytest.raises(storage.StorageIntegrityError, match="symlink"):
        share(base, tmp_path)
    with pytest.raises(storage.StorageIntegrityError, match="outside"):
        storage.deduplicate_native_weights(base, pool_directory=base.parent / ".checkpoint_storage", clone=mock_clone)


def test_parent_traversal_cannot_bypass_default_runs_or_pool_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    outside = tmp_path / "checkpoints/export/base"
    outside.mkdir(parents=True)
    (outside / "model.safetensors").write_bytes(weights())
    alias = tmp_path / "runs/../checkpoints/export/base"
    with pytest.raises(storage.StorageIntegrityError, match="parent path traversal"):
        storage.deduplicate_native_weights(alias, clone=mock_clone)
    base = saved(tmp_path, "traversal")
    with pytest.raises(storage.StorageIntegrityError, match="parent path traversal"):
        storage.deduplicate_native_weights(base, pool_directory=base / "../.checkpoint_storage", clone=mock_clone)
    assert not pool(tmp_path).exists()


def test_concurrent_same_hash_publication_creates_one_complete_entry(tmp_path):
    bases = [saved(tmp_path, name) for name in ("first", "second")]
    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = list(executor.map(lambda base: share(base, tmp_path), bases))
    actions = [report["files"][0]["action"] for report in reports]
    assert sorted(actions) == ["published_pool_clone_of_saved_file", "reused_verified_pool_clone"]
    assert len(canonicals(tmp_path)) == 1 and canonicals(tmp_path)[0].read_bytes() == weights()


def test_nonmacos_default_backend_keeps_normal_save_without_a_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setattr(storage.platform, "system", lambda: "Linux")
    base = saved(tmp_path, "normal")
    report = storage.deduplicate_native_weights(base)
    assert report["status"] == "ordinary_save_no_macos_clone_backend"
    assert not pool(tmp_path).exists() and (base / "model.safetensors").read_bytes() == weights()


def test_default_never_touches_source_checkpoints_or_exports_outside_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    source = tmp_path / "checkpoints/source"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"original BF16 source fixture")
    result = storage.deduplicate_native_weights(source)
    assert result["status"] == "ordinary_save_outside_runs"
    assert (source / "model.safetensors").read_bytes() == b"original BF16 source fixture"
    assert not pool(tmp_path).exists()


@pytest.mark.skipif(platform.system() != "Darwin", reason="Actual public APFS clonefile smoke requires macOS")
def test_actual_apfs_clones_have_private_writes_and_survive_pool_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    first, second = saved(tmp_path, "actual_first"), saved(tmp_path, "actual_second")
    reports = [storage.deduplicate_native_weights(base) for base in (first, second)]
    assert all(report["backend"] == "macos_clonefile" for report in reports)
    assert reports[0]["files"][0]["action"] == "published_pool_clone_of_saved_file"
    assert reports[1]["files"][0]["action"] == "reused_verified_pool_clone"
    canonical = canonicals(tmp_path)[0]
    paths = [first / "model.safetensors", second / "model.safetensors", canonical]
    assert len({path.stat().st_ino for path in paths}) == 3
    paths[0].write_bytes(weights(2.0))
    assert paths[1].read_bytes() == canonical.read_bytes() == weights()
    paths[1].write_bytes(weights(3.0))
    assert paths[0].read_bytes() == weights(2.0) and canonical.read_bytes() == weights()
    shutil.rmtree(pool(tmp_path))
    assert paths[0].read_bytes() == weights(2.0) and paths[1].read_bytes() == weights(3.0)


def test_adapter_save_uses_storage_before_manifest_without_changing_layout(tmp_path, monkeypatch):
    # Fake native serializer, no model forward/backward or pretrained model load.
    pytest.importorskip("torch")
    from replica_lab.model_adapter import LlamaMemoryAdapter
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    real_share = storage.deduplicate_native_weights
    monkeypatch.setattr(storage, "deduplicate_native_weights",
                        lambda base: real_share(base, clone=mock_clone))
    class FakeBase:
        def save_pretrained(self, path, **kwargs):
            path.mkdir(parents=True)
            (path / "model.safetensors").write_bytes(weights())
            (path / "config.json").write_text('{"model_type":"llama"}')
    adapter = SimpleNamespace(_context=None, augmented=False, base=FakeBase(), regime="full",
                              settings={}, parameter_audit=lambda: {"fixture": True})
    destinations = [tmp_path / "runs" / name for name in ("runtime", "final_export")]
    for destination in destinations:
        LlamaMemoryAdapter.save_pretrained(adapter, destination)
    assert len(canonicals(tmp_path)) == 1
    for destination in destinations:
        manifest = json.loads((destination / "adapter_manifest.json").read_text())
        assert manifest["checkpoint_storage"]["status"] == "cow_pool_processed"
        assert (destination / "base/model.safetensors").read_bytes() == weights()
        assert set(path.name for path in destination.iterdir()) == {"base", "adapter_manifest.json"}


def test_training_receipts_validate_with_shared_base_and_independent_sidecars_runtime(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from replica_lab.model_adapter import LlamaMemoryAdapter
    from replica_lab.checkpointing import save_training_checkpoint, validate_training_checkpoint
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    real_share = storage.deduplicate_native_weights
    monkeypatch.setattr(storage, "deduplicate_native_weights",
                        lambda base: real_share(base, clone=mock_clone))
    class FakeBase:
        def save_pretrained(self, path, **kwargs):
            path.mkdir(parents=True)
            (path / "model.safetensors").write_bytes(weights())
            (path / "config.json").write_text('{"model_type":"llama"}')
    adapter = SimpleNamespace(_context=None, augmented=False, base=FakeBase(), regime="full",
                              settings={}, parameter_audit=lambda: {"fixture": True})
    class Model:
        version = 1
        def save_pretrained(self, directory, *, manifest):
            LlamaMemoryAdapter.save_pretrained(adapter, directory, manifest=manifest)
            (directory / "sidecars.safetensors").write_bytes(f"independent-sidecar-{self.version}".encode())
    class Tokenizer:
        def save_pretrained(self, directory):
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (directory / name).write_text("{}")
    class Optimizer:
        version = 1
        def state_dict(self):
            return {"fixture_version": self.version}
    model, optimizer, contract = Model(), Optimizer(), {"purpose": "byte-storage-fixture"}
    destinations = [tmp_path / "runs/example" / f"training_checkpoint_{step:04}" for step in (1, 2)]
    for step, destination in enumerate(destinations, 1):
        model.version = optimizer.version = step
        save_training_checkpoint(model, Tokenizer(), optimizer, destination, step=step,
                                 data_state={"step": step}, schedule_state={"step": step},
                                 run_contract=contract, manifest={"step": step})
        receipt = validate_training_checkpoint(destination, contract)
        assert all(".native-cow-" not in name for name in receipt["files_sha256"])
    assert len(canonicals(tmp_path)) == 1
    assert (destinations[0] / "sidecars.safetensors").read_bytes() != (destinations[1] / "sidecars.safetensors").read_bytes()
    assert torch.load(destinations[0] / "runtime.pt", weights_only=True)["optimizer"] != torch.load(destinations[1] / "runtime.pt", weights_only=True)["optimizer"]
    assert (destinations[0] / "base/model.safetensors").stat().st_ino != (destinations[1] / "base/model.safetensors").stat().st_ino
    shutil.rmtree(pool(tmp_path))
    for destination in destinations:
        validate_training_checkpoint(destination, contract)
