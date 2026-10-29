"""Byte-bound local trust_remote_code inventory, independent of conversion history.

Building an inventory is preparation, never authorization. Verification runs
before either tokenizer or model loading, including when weight rehashing is
explicitly skipped for a metadata review. All local Python/native executable
files are bound so a sibling import cannot silently substitute a dependency.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Mapping

FORMAT = "replica.spark.execution_manifest.v1"
EXECUTABLE_SUFFIXES = {".py", ".pyw", ".so", ".dylib", ".pyd", ".dll"}


class ExecutionManifestError(ValueError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        if path.suffix in EXECUTABLE_SUFFIXES:
            if path.is_symlink() or not path.is_file() or any(
                    parent.is_symlink() for parent in path.parents if parent != root.parent):
                raise ExecutionManifestError(f"non-regular execution code: {path}")
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def _entrypoints(config: Mapping) -> dict[str, str]:
    mapping = config.get("auto_map", {})
    result = {}
    for key in ("AutoConfig", "AutoModelForCausalLM"):
        value = mapping.get(key)
        if not isinstance(value, str) or "--" in value:
            raise ExecutionManifestError(f"local auto_map {key} is required")
        module, _, symbol = value.rpartition(".")
        if not symbol.isidentifier() or not module or any(
                not part.isidentifier() for part in module.split(".")):
            raise ExecutionManifestError(f"invalid auto_map {key}")
        result[key] = module.replace(".", "/") + ".py"
    if set(result.values()) != {"configuration_spark.py", "modeling_spark.py"}:
        raise ExecutionManifestError("Spark model/configuration code entrypoints differ")
    return result


def build_execution_manifest(source_root: Path, *, artifact_sha256: str) -> dict:
    root = Path(source_root).absolute()
    config = json.loads((root / "config.json").read_text())
    entries = _entrypoints(config)
    files = _files(root)
    if not set(entries.values()) <= files.keys():
        raise ExecutionManifestError("missing model/configuration execution code")
    dependencies = {}
    for name in files:
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse((root / name).read_text(), filename=name)
        except (SyntaxError, UnicodeError) as error:
            raise ExecutionManifestError(f"invalid execution code: {name}") from error
        local = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = Path(name).parent
                for _ in range(node.level - 1):
                    base = base.parent
                modules = [node.module] if node.module else [a.name for a in node.names]
                for module in modules:
                    candidate = base / module.replace(".", "/")
                    choices = [candidate.with_suffix(".py"), candidate / "__init__.py"]
                    found = next((p.as_posix() for p in choices if p.as_posix() in files), None)
                    if found is None:
                        raise ExecutionManifestError(f"missing local dependency: {name}: {module}")
                    local.add(found)
        dependencies[name] = sorted(local)
    return {"format": FORMAT, "artifact_sha256": artifact_sha256,
            "config_sha256": digest(root / "config.json"),
            "auto_map": config["auto_map"], "entrypoints": entries,
            "files": files, "relative_dependencies": dependencies,
            "policy": "exact_local_executable_inventory_no_symlinks"}


def verify_execution_manifest(source_root: Path, binding: Mapping, *,
                              repository_root: Path, artifact_sha256: str) -> dict:
    try:
        path = Path(binding["path"])
        if not path.is_absolute():
            path = repository_root / path
        if not path.is_file() or digest(path) != binding["sha256"]:
            raise ExecutionManifestError("execution manifest missing or hash differs")
        expected = json.loads(path.read_text())
        observed = build_execution_manifest(source_root, artifact_sha256=artifact_sha256)
        if expected != observed:
            raise ExecutionManifestError("execution code inventory, contents or dependencies differ")
        return {"format": FORMAT, "sha256": binding["sha256"],
                "files": observed["files"], "verified_before_model_load": True}
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ExecutionManifestError(f"execution manifest unavailable: {error}") from error
