"""Tiny stdlib-only snapshot tests: no real download, tensor library or model."""
import builtins
import fnmatch
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/download_model.py"
spec = importlib.util.spec_from_file_location("download_model_reuse", SCRIPT)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "ROOT", tmp_path)
    monkeypatch.delenv("HF_HOME", raising=False)
    output = tmp_path / "checkpoints/source"
    output.mkdir(parents=True)
    config = {"model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 42,
              "vocab_size": 130560, "tie_word_embeddings": False,
              "architectures": ["LlamaForCausalLM"]}
    files = {"config.json": json.dumps(config).encode(),
             "model.safetensors": b"tiny fake sealed weights, never executed",
             "tokenizer.json": b'{"tiny":"fixture"}',
             "tokenizer_config.json": b"{}", "tokenizer.model": b"tiny tokenizer model",
             "chat_template.jinja": b"tiny fixed template",
             ".cache/huggingface/trees/fixture.json": b'{"revision":"fixture"}'}
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    metadata = {"repo_id": "openbmb/MiniCPM5-2B", "revision": "a" * 40,
                "config_sha256": hashlib.sha256(files["config.json"]).hexdigest(),
                "weights_downloaded": False}
    metadata_path = tmp_path / "configs/metadata.json"
    write_json(metadata_path, metadata)
    seal = {**metadata, "weights_downloaded": True, "local_bytes_sealed": True,
            "snapshot_files_sha256": {name: hashlib.sha256(content).hexdigest()
                                      for name, content in files.items()}}
    seal_path = tmp_path / "configs/minicpm5_sealed_lock.local.json"
    write_json(seal_path, seal)
    receipt = {"repo_id": metadata["repo_id"], "revision": metadata["revision"],
               "requested_at": "2026-10-20T01:02:03+00:00", "status": "downloaded_not_loaded",
               "completed_at": "2026-10-20T01:04:05+00:00", "retained_extra": "provenance"}
    receipt_path = output.parent / (output.name + ".download.json")
    write_json(receipt_path, receipt)
    return SimpleNamespace(root=tmp_path, output=output, metadata=metadata, metadata_path=metadata_path,
                           seal=seal, seal_path=seal_path, receipt=receipt,
                           receipt_path=receipt_path, files=files)


def fingerprint(root):
    values = {}
    for path in root.rglob("*"):
        info = path.lstat()
        values[path.relative_to(root).as_posix()] = (
            path.read_bytes() if path.is_file() else None,
            info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)
    return values


def forbid_model_or_network(monkeypatch):
    real_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"huggingface_hub", "torch", "transformers", "safetensors"}:
            raise AssertionError(f"Forbidden model/network import: {name}")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    def forbidden_socket(*args, **kwargs):
        raise AssertionError("No network sockets may be created")
    monkeypatch.setattr(socket, "socket", forbidden_socket)


@pytest.mark.parametrize("allow_network", [False, True])
def test_valid_sealed_snapshot_reuses_offline_without_any_writes(sealed, monkeypatch, allow_network):
    before = fingerprint(sealed.root)
    forbid_model_or_network(monkeypatch)
    result = download.ensure_snapshot(sealed.metadata_path, sealed.output,
                                      sealed_lock=sealed.seal_path, allow_network=allow_network)
    assert result["status"] == "reused_local_sealed_snapshot"
    assert result["network_used"] is False and result["receipt_preserved"] is True
    assert result["verified_files"] == len(sealed.files)
    assert fingerprint(sealed.root) == before
    assert "HF_HOME" not in os.environ


def test_cli_uses_existing_project_seal_and_prints_reuse_result(sealed, monkeypatch, capsys):
    before = fingerprint(sealed.root)
    forbid_model_or_network(monkeypatch)
    download.main(["--model-lock", str(sealed.metadata_path), "--output", str(sealed.output)])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "reused_local_sealed_snapshot"
    assert fingerprint(sealed.root) == before


def test_fresh_interpreter_cli_does_not_import_model_libraries(sealed):
    code = """
import builtins, runpy, socket, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'huggingface_hub','torch','transformers','safetensors'}:
        raise AssertionError('forbidden import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
def no_network(*args, **kwargs): raise AssertionError('network forbidden')
socket.socket = no_network
sys.argv = [sys.argv[1], '--model-lock', sys.argv[2], '--output', sys.argv[3], '--sealed-lock', sys.argv[4]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    before = fingerprint(sealed.root)
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPT), str(sealed.metadata_path),
                             str(sealed.output), str(sealed.seal_path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "reused_local_sealed_snapshot"
    assert fingerprint(sealed.root) == before


@pytest.mark.parametrize("failure", ["corrupt_weight", "corrupt_config", "missing_weight", "missing_snapshot",
                                     "missing_receipt", "incomplete_receipt", "missing_completion_time",
                                     "missing_seal", "unsealed", "not_downloaded", "extra_json",
                                     "missing_tokenizer", "changed_cached_json", "missing_manifest"])
def test_offline_failure_never_imports_fetches_or_rewrites(sealed, monkeypatch, failure):
    if failure == "corrupt_weight":
        (sealed.output / "model.safetensors").write_bytes(b"corrupt")
    elif failure == "corrupt_config":
        (sealed.output / "config.json").write_text("{}")
    elif failure == "missing_weight":
        (sealed.output / "model.safetensors").unlink()
    elif failure == "missing_snapshot":
        shutil.rmtree(sealed.output)
    elif failure == "missing_receipt":
        sealed.receipt_path.unlink()
    elif failure in {"incomplete_receipt", "missing_completion_time"}:
        value = dict(sealed.receipt)
        if failure == "incomplete_receipt": value["status"] = "downloading"
        else: value.pop("completed_at")
        write_json(sealed.receipt_path, value)
    elif failure == "missing_seal":
        sealed.seal_path.unlink()
    elif failure in {"unsealed", "not_downloaded", "missing_manifest"}:
        value = dict(sealed.seal)
        if failure == "missing_manifest": value.pop("snapshot_files_sha256")
        else: value["local_bytes_sealed" if failure == "unsealed" else "weights_downloaded"] = False
        write_json(sealed.seal_path, value)
    elif failure == "extra_json":
        (sealed.output / "unsealed_added_file.json").write_text("{}")
    elif failure == "missing_tokenizer":
        (sealed.output / "tokenizer.json").unlink()
    else:
        (sealed.output / ".cache/huggingface/trees/fixture.json").write_text('{"changed":true}')
    before = fingerprint(sealed.root)
    forbid_model_or_network(monkeypatch)
    with pytest.raises(ValueError):
        download.ensure_snapshot(sealed.metadata_path, sealed.output, sealed_lock=sealed.seal_path)
    assert fingerprint(sealed.root) == before
    assert "HF_HOME" not in os.environ


@pytest.mark.parametrize("location,field", [(location, field) for location in ("receipt", "seal")
                                          for field in ("repo_id", "revision", "config_sha256")])
@pytest.mark.parametrize("allow_network", [False, True])
def test_source_identity_conflicts_always_fail_before_network(sealed, monkeypatch, location, field, allow_network):
    value = dict(sealed.receipt if location == "receipt" else sealed.seal)
    value[field] = {"repo_id": "other/model", "revision": "b" * 40, "config_sha256": "f" * 64}[field]
    write_json(sealed.receipt_path if location == "receipt" else sealed.seal_path, value)
    before = fingerprint(sealed.root)
    forbid_model_or_network(monkeypatch)
    with pytest.raises(ValueError):
        download.ensure_snapshot(sealed.metadata_path, sealed.output, sealed_lock=sealed.seal_path,
                                 allow_network=allow_network)
    assert fingerprint(sealed.root) == before


@pytest.mark.parametrize("path", ["../outside.json", "/outside.json", "folder/../outside.json",
                                  "folder//outside.json", "folder\\outside.json"])
@pytest.mark.parametrize("allow_network", [False, True])
def test_manifest_path_escape_or_alias_is_rejected(sealed, monkeypatch, path, allow_network):
    sealed.seal["snapshot_files_sha256"][path] = "a" * 64
    write_json(sealed.seal_path, sealed.seal)
    before = fingerprint(sealed.root)
    forbid_model_or_network(monkeypatch)
    with pytest.raises(ValueError, match="invalid path"):
        download.ensure_snapshot(sealed.metadata_path, sealed.output, sealed_lock=sealed.seal_path,
                                 allow_network=allow_network)
    assert fingerprint(sealed.root) == before


@pytest.mark.parametrize("invalid", ["unsealed", "not_downloaded", "missing_manifest"])
def test_invalid_seal_authority_cannot_trigger_a_download_even_with_opt_in(sealed, monkeypatch, invalid):
    value = dict(sealed.seal)
    if invalid == "missing_manifest": value.pop("snapshot_files_sha256")
    else: value["local_bytes_sealed" if invalid == "unsealed" else "weights_downloaded"] = False
    write_json(sealed.seal_path, value)
    before = fingerprint(sealed.root)
    forbid_model_or_network(monkeypatch)
    with pytest.raises(ValueError):
        download.ensure_snapshot(sealed.metadata_path, sealed.output,
                                 sealed_lock=sealed.seal_path, allow_network=True)
    assert fingerprint(sealed.root) == before


def test_unselected_hf_lock_files_do_not_invalidate_selected_inventory(sealed, monkeypatch):
    (sealed.output / ".cache/huggingface/local.lock").write_bytes(b"not a sealed suffix")
    forbid_model_or_network(monkeypatch)
    assert download.ensure_snapshot(sealed.metadata_path, sealed.output,
                                    sealed_lock=sealed.seal_path)["status"] == "reused_local_sealed_snapshot"


def test_sealed_model_lock_can_itself_supply_the_seal(sealed, monkeypatch):
    forbid_model_or_network(monkeypatch)
    result = download.ensure_snapshot(sealed.seal_path, sealed.output)
    assert result["status"] == "reused_local_sealed_snapshot"


def install_fake_network(monkeypatch, calls, implementation):
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return implementation(**kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))


def test_explicit_initial_acquisition_uses_pinned_identity_and_keeps_metadata_immutable(sealed, monkeypatch):
    calls = []
    output = sealed.root / "new/source"
    metadata_before, seal_before = sealed.metadata_path.read_bytes(), sealed.seal_path.read_bytes()
    def acquire(**kwargs):
        for name, content in sealed.files.items():
            path = kwargs["local_dir"] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    install_fake_network(monkeypatch, calls, acquire)
    result = download.ensure_snapshot(sealed.metadata_path, output, allow_network=True)
    assert len(calls) == 1
    assert calls[0]["repo_id"] == sealed.metadata["repo_id"]
    assert calls[0]["revision"] == "a" * 40 and calls[0]["max_workers"] == 2
    assert result["network_used"] is True and result["local_bytes_sealed"] is False
    receipt = json.loads((output.parent / "source.download.json").read_text())
    assert receipt["status"] == "downloaded_not_loaded" and receipt["completed_at"]
    assert receipt["config_sha256"] == sealed.metadata["config_sha256"]
    assert sealed.metadata_path.read_bytes() == metadata_before and sealed.seal_path.read_bytes() == seal_before


def test_missing_initial_source_defaults_to_no_network_and_no_directory_creation(sealed, monkeypatch):
    output = sealed.root / "not_created/source"
    forbid_model_or_network(monkeypatch)
    with pytest.raises(ValueError, match="explicit --allow-network"):
        download.ensure_snapshot(sealed.metadata_path, output)
    assert not output.parent.exists() and "HF_HOME" not in os.environ


@pytest.mark.parametrize("previous_status", ["downloading", "downloaded_not_loaded"])
def test_explicit_same_identity_resume_or_repair_preserves_acquisition_identity(sealed, monkeypatch, previous_status):
    previous = {**sealed.receipt, "status": previous_status}
    if previous_status == "downloading": previous.pop("completed_at")
    write_json(sealed.receipt_path, previous)
    (sealed.output / "model.safetensors").write_bytes(b"incomplete or corrupt")
    calls = []
    def repair(**kwargs):
        (kwargs["local_dir"] / "model.safetensors").write_bytes(sealed.files["model.safetensors"])
    install_fake_network(monkeypatch, calls, repair)
    result = download.ensure_snapshot(sealed.metadata_path, sealed.output,
                                      sealed_lock=sealed.seal_path, allow_network=True)
    assert len(calls) == 1 and result["network_used"] is True
    assert calls[0]["force_download"] is True
    assert calls[0]["allow_patterns"] == ["model.safetensors"]
    assert result["forced_repair_members"] == ["model.safetensors"]
    receipt = json.loads(sealed.receipt_path.read_text())
    for key in ("repo_id", "revision", "requested_at", "retained_extra"):
        assert receipt[key] == previous[key]
    if previous_status == "downloaded_not_loaded":
        assert receipt["completed_at"] == previous["completed_at"]
    assert receipt["last_network_action"] == "explicit_repair_or_resume"
    assert receipt["last_repaired_sealed_members"] == ["model.safetensors"]


def test_failed_explicit_repair_preserves_existing_receipt(sealed, monkeypatch):
    (sealed.output / "model.safetensors").write_bytes(b"corrupt")
    before = sealed.receipt_path.read_bytes()
    calls = []
    def fail(**kwargs):
        raise OSError("simulated download failure")
    install_fake_network(monkeypatch, calls, fail)
    with pytest.raises(OSError, match="simulated"):
        download.ensure_snapshot(sealed.metadata_path, sealed.output,
                                 sealed_lock=sealed.seal_path, allow_network=True)
    assert len(calls) == 1 and sealed.receipt_path.read_bytes() == before


def test_no_receipt_nonempty_directory_is_never_adopted_by_network(sealed, monkeypatch):
    sealed.receipt_path.unlink()
    forbid_model_or_network(monkeypatch)
    with pytest.raises(ValueError, match="no acquisition receipt"):
        download.ensure_snapshot(sealed.metadata_path, sealed.output, allow_network=True)


def test_cli_network_opt_in_defaults_false_and_default_seal_is_optional(sealed):
    args = download.parser().parse_args(["--model-lock", "metadata.json", "--output", "source"])
    assert args.allow_network is False and args.sealed_lock == sealed.seal_path
    sealed.seal_path.unlink()
    args = download.parser().parse_args(["--model-lock", "metadata.json", "--output", "source"])
    assert args.sealed_lock is None


def test_explicit_repair_defeats_cached_revision_and_mtime_reuse_without_touching_valid_files(sealed, monkeypatch):
    target = sealed.output / "model.safetensors"
    previous_stat = target.stat()
    target.write_bytes(b"X" * len(sealed.files["model.safetensors"]))
    os.utime(target, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    preserved = {name: ((sealed.output / name).read_bytes(), (sealed.output / name).stat().st_ino,
                        (sealed.output / name).stat().st_mtime_ns)
                 for name in sealed.files if name != "model.safetensors"}
    calls, writes = [], []
    def cached_snapshot(**kwargs):
        # Emulate HF's trusted same-revision metadata: unchanged mtime skips bytes
        # unless force_download is requested, even though our seal detects corruption.
        if not kwargs.get("force_download"):
            return
        for name, content in sealed.files.items():
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in kwargs["allow_patterns"]):
                (kwargs["local_dir"] / name).write_bytes(content)
                writes.append(name)
    install_fake_network(monkeypatch, calls, cached_snapshot)
    result = download.ensure_snapshot(sealed.metadata_path, sealed.output,
                                      sealed_lock=sealed.seal_path, allow_network=True)
    assert writes == ["model.safetensors"] and result["forced_repair_members"] == writes
    assert target.read_bytes() == sealed.files["model.safetensors"]
    assert preserved == {name: ((sealed.output / name).read_bytes(), (sealed.output / name).stat().st_ino,
                                (sealed.output / name).stat().st_mtime_ns) for name in preserved}


@pytest.mark.parametrize("missing", ["model.safetensors", "tokenizer.json", "config.json"])
def test_missing_sealed_member_is_targeted_without_redownloading_valid_originals(sealed, monkeypatch, missing):
    (sealed.output / missing).unlink()
    preserved = {name: ((sealed.output / name).read_bytes(), (sealed.output / name).stat().st_ino,
                        (sealed.output / name).stat().st_mtime_ns) for name in sealed.files if name != missing}
    calls = []
    def acquire_missing(**kwargs):
        assert kwargs["force_download"] is True and kwargs["allow_patterns"] == [missing]
        (kwargs["local_dir"] / missing).write_bytes(sealed.files[missing])
    install_fake_network(monkeypatch, calls, acquire_missing)
    result = download.ensure_snapshot(sealed.metadata_path, sealed.output,
                                      sealed_lock=sealed.seal_path, allow_network=True)
    assert result["forced_repair_members"] == [missing]
    assert preserved == {name: ((sealed.output / name).read_bytes(), (sealed.output / name).stat().st_ino,
                                (sealed.output / name).stat().st_mtime_ns) for name in preserved}


def test_multiple_corrupt_members_are_the_only_force_download_patterns(sealed, monkeypatch):
    damaged = ["config.json", "tokenizer.json"]
    for name in damaged:
        (sealed.output / name).write_bytes(b"corrupt")
    calls = []
    def repair(**kwargs):
        assert kwargs["allow_patterns"] == damaged and kwargs["force_download"] is True
        for name in damaged:
            (kwargs["local_dir"] / name).write_bytes(sealed.files[name])
    install_fake_network(monkeypatch, calls, repair)
    result = download.ensure_snapshot(sealed.metadata_path, sealed.output,
                                      sealed_lock=sealed.seal_path, allow_network=True)
    assert result["forced_repair_members"] == damaged


@pytest.mark.parametrize("conflict", ["extra_inventory", "config_identity", "missing_seal", "receipt_only",
                                      "corrupt_generated_cache", "only_missing_generated_cache"])
def test_nonrepairable_conflicts_refuse_before_network_and_preserve_all_existing_files(sealed, monkeypatch, conflict):
    if conflict == "extra_inventory":
        (sealed.output / "extra.json").write_text("{}")
    elif conflict == "config_identity":
        sealed.seal["snapshot_files_sha256"]["config.json"] = "f" * 64
        write_json(sealed.seal_path, sealed.seal)
    elif conflict == "missing_seal":
        sealed.seal_path.unlink()
    elif conflict == "receipt_only":
        receipt = {**sealed.receipt, "status": "downloading"}
        receipt.pop("completed_at")
        write_json(sealed.receipt_path, receipt)
    else:
        path = sealed.output / ".cache/huggingface/trees/fixture.json"
        if conflict == "corrupt_generated_cache": path.write_bytes(b"corrupt")
        else: path.unlink()
    before = fingerprint(sealed.root)
    forbid_model_or_network(monkeypatch)
    with pytest.raises(ValueError):
        download.ensure_snapshot(sealed.metadata_path, sealed.output,
                                 sealed_lock=sealed.seal_path, allow_network=True)
    assert fingerprint(sealed.root) == before
    assert "HF_HOME" not in os.environ


@pytest.mark.parametrize("resume", [False, True])
def test_initial_acquisition_or_resume_works_with_project_seal(sealed, monkeypatch, capsys, resume):
    output = sealed.root / "initial/source"
    if resume:
        write_json(output.parent / "source.download.json",
                   {**sealed.receipt, "status": "downloading", "completed_at": None})
    calls = []
    def acquire(**kwargs):
        for name, content in sealed.files.items():
            # Real HF generates its tree JSON as acquisition metadata rather than
            # fetching that local-only path from the remote repository.
            selected = kwargs.get("allow_patterns")
            if name.startswith(".cache/") or selected is None or name in selected:
                path = kwargs["local_dir"] / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    install_fake_network(monkeypatch, calls, acquire)
    download.main(["--model-lock", str(sealed.metadata_path), "--output", str(output), "--allow-network"])
    result = json.loads(capsys.readouterr().out)
    assert result["network_used"] is True and result["local_bytes_sealed"] is True
    assert result["completed_at"]
    assert len(calls) == 1
    if resume:
        assert calls[0]["force_download"] is True
        assert all(not name.startswith(".cache/") for name in calls[0]["allow_patterns"])
    else:
        assert "force_download" not in calls[0] and "allow_patterns" not in calls[0]
    assert download.ensure_snapshot(sealed.metadata_path, output,
                                    sealed_lock=sealed.seal_path)["status"] == "reused_local_sealed_snapshot"
    assert len(calls) == 1


def test_repair_allow_patterns_treat_sealed_filenames_literally(sealed, monkeypatch):
    for name in ("tokenizer[1].json", "tokenizer1.json"):
        content = name.encode()
        sealed.files[name] = content
        (sealed.output / name).write_bytes(content)
        sealed.seal["snapshot_files_sha256"][name] = hashlib.sha256(content).hexdigest()
    write_json(sealed.seal_path, sealed.seal)
    (sealed.output / "tokenizer[1].json").write_bytes(b"corrupt")
    untouched = (sealed.output / "tokenizer1.json").stat().st_mtime_ns
    calls, writes = [], []
    def repair(**kwargs):
        for name, content in sealed.files.items():
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in kwargs["allow_patterns"]):
                (kwargs["local_dir"] / name).write_bytes(content)
                writes.append(name)
    install_fake_network(monkeypatch, calls, repair)
    result = download.ensure_snapshot(sealed.metadata_path, sealed.output,
                                      sealed_lock=sealed.seal_path, allow_network=True)
    assert result["forced_repair_members"] == ["tokenizer[1].json"]
    assert writes == ["tokenizer[1].json"]
    assert (sealed.output / "tokenizer1.json").stat().st_mtime_ns == untouched
