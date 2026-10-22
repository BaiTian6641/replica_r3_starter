"""Exercise the guard against real staged blobs, including force-added files."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_git_artifacts", ROOT / "scripts/check_git_artifacts.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    return tmp_path


def stage(repo, name, data, force=False):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    subprocess.run(["git", "add", *( ["-f"] if force else []), "--", name], check=True)


def test_source_and_reports_remain_shareable(repo):
    stage(repo, "src/example.py", b"print('source')\n")
    stage(repo, "reports/result.json", b'{"complete":true}\n')
    report = GUARD.audit_index()
    assert report["status"] == "passed" and report["checked_files"] == 2


@pytest.mark.parametrize("name", ["checkpoints/source/config.json", "runs/old/result.txt",
    "models/model.json", "outside/weight.safetensors", "outside/model.pth",
    "outside/pytorch_model-00001.bin", "outside/example.mlpackage/Data/file.bin",
    "data/raw/input.jsonl", ".venv-extra/lib/file.py",
    "CHECKPOINTS/source/config.json", "Runs/run/runtime.json",
    "DATA/RAW/sample.jsonl", ".VENV/lib/cache.json"])
def test_force_added_local_artifacts_are_rejected_without_deletion(repo, name):
    stage(repo, name, b"local fixture", force=True)
    report = GUARD.audit_index()
    assert report["status"] == "failed"
    assert report["violations"][0]["path"] == name
    assert (repo / name).read_bytes() == b"local fixture"


def test_size_check_uses_staged_blob_instead_of_changed_working_file(repo):
    stage(repo, "large.dat", b"x" * 33)
    (repo / "large.dat").write_bytes(b"small now")
    report = GUARD.audit_index(max_bytes=32)
    assert report["status"] == "failed" and report["violations"][0]["bytes"] == 33


def test_tracked_scope_catches_existing_blob_even_when_nothing_is_staged(repo):
    stage(repo, "too-large.txt", b"x" * 33)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    assert GUARD.audit_index(max_bytes=32)["checked_files"] == 0
    assert GUARD.audit_index("tracked", max_bytes=32)["status"] == "failed"


@pytest.mark.parametrize("name", ["checkpoints/source/config.json", "runs/run/checkpoint/weights.bin",
    "models/source/tokenizer.json", "elsewhere/model.safetensors", "elsewhere/optimizer.bin",
    "elsewhere/model.gguf", "elsewhere/model.onnx", "elsewhere/example.mlpackage/metadata.json",
    ".cache/huggingface/file.json", "data/processed/train.jsonl"])
def test_ignore_rules_cover_local_paths(repo, name):
    result = subprocess.run(["git", "check-ignore", "--no-index", "--", name], capture_output=True)
    assert result.returncode == 0
