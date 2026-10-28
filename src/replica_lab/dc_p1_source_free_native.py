"""Authorization and native execution for the source-free v2 diagnostic."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

from dc_protocol_common_20261027 import (
    read_json, read_jsonl, require, require_committed_bytes, seal_record,
    sha256_file, sha256_json, tokenizer_binding_sha, validate_tokenizer_binding,
)
from replica_lab.dc_p1_native_runner import NativeP1Backend
from replica_lab.dc_p1_source_free_runner import run_fit_diagnostic

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION_FORMAT = "replica.dc_protocol.authorization.v2"
PROTOCOL_ID = "dc_p1_source_free_feature_alignment_v2"
IMPLEMENTATION_RELATIVES = (
    "configs/dc_p1_source_free_feature_alignment_v2.json",
    "data/dc_p1_native_response_v2_20261027/manifest.json",
    "scripts/preflight_dc_p1_source_free_feature_alignment_v2.py",
    "scripts/run_dc_p1_source_free_feature_alignment_v2.py",
    "src/replica_lab/dc_p1_source_free_runner.py",
    "src/replica_lab/dc_p1_tiny_learning.py",
    "src/replica_lab/dc_p1_learning_audit.py",
    "src/replica_lab/testing.py",
    "src/replica_lab/dc_p1_p_only_reader.py",
    "src/replica_lab/dc_response_objective.py",
    "src/replica_lab/artifact_publication.py",
    "src/replica_lab/dc_p1_source_free_native.py",
    "src/replica_lab/dc_p1_native_runner.py",
    "src/replica_lab/nested_learning.py",
    "src/replica_lab/architecture/plastic.py",
    "src/replica_lab/architecture/loading.py",
    "src/replica_lab/capacity_guard.py",
    "scripts/run_capacity_preflight.py",
    "scripts/dc_protocol_common_20261027.py",
)


@dataclass(frozen=True)
class SourceFreeContext:
    root: Path
    config_path: Path
    config: dict
    config_sha256: str
    prepared: Path
    manifest: dict
    manifest_sha256: str
    authorization: dict
    authorization_sha256: str
    output: Path
    attempt_id: str
    protocol_commit: str
    binding: dict


class SourceFreeAttemptRegistry:
    def __init__(self, authorization_path: Path) -> None:
        self.path = authorization_path.with_suffix(authorization_path.suffix + ".uses.json")

    def register(self, context: SourceFreeContext) -> None:
        payload = {
            "format": "replica.dc_p1_source_free.authorization_use.v1",
            "authorization_receipt_sha256": context.authorization_sha256,
            "attempt_id": context.attempt_id,
            "output_root": str(context.output),
            "protocol_commit": context.protocol_commit,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if self.path.exists():
            previous = read_json(self.path)
            for key in ("authorization_receipt_sha256", "attempt_id", "output_root",
                        "protocol_commit"):
                require(previous.get(key) == payload[key], "single_attempt",
                        f"authorization was already used with different {key}")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                self.path.unlink()
            except OSError:
                pass
            raise


def verify_source_free_authorization(*, config_path: str | Path,
                                    prepared: str | Path,
                                    authorization_path: str | Path,
                                    output: str | Path,
                                    require_head: bool = True,
                                    commit_verifier: Callable | None = None,
                                    register: bool = True) -> tuple[SourceFreeContext, dict]:
    from preflight_dc_p1_source_free_feature_alignment_v2 import validate_config

    config_path = Path(config_path)
    prepared = Path(prepared)
    authorization_path = Path(authorization_path)
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    config = read_json(config_path)
    derived = validate_config(config)
    from preflight_dc_p1_source_free_feature_alignment_v1 import validate_prepared
    validate_prepared(prepared, config)
    manifest = read_json(prepared / "manifest.json")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(manifest["manifest_sha256"] == sha256_json(body), "prepared",
            "manifest self-integrity digest differs")
    for name, digest in manifest["files"].items():
        path = prepared / name
        require(path.is_file(), "consumed_artifact", f"missing consumed artifact {path}")
        require(sha256_file(path) == digest, "consumed_artifact",
                f"consumed artifact hash differs: {name}")
    config_sha = sha256_file(config_path)
    authorization = read_json(authorization_path)
    require(authorization.get("format") == AUTHORIZATION_FORMAT,
            "authorization", "authorization format differs")
    require(authorization.get("protocol_id") == PROTOCOL_ID,
            "authorization", "authorization protocol differs")
    require(authorization.get("config_sha256") == config_sha,
            "authorization", "config bytes differ")
    require(authorization.get("prepared_manifest_sha256")
            == manifest["manifest_sha256"], "authorization", "fixture bytes differ")
    commit = authorization.get("protocol_commit")
    require(isinstance(commit, str) and len(commit) == 40
            and all(character in "0123456789abcdef" for character in commit),
            "authorization", "protocol_commit must be full 40-hex")
    if require_head:
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True).stdout.strip()
        require(actual == commit, "authorization", "protocol commit is not HEAD")
        dirty = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        require(not dirty, "authorization", "tracked worktree changes are not allowed")
    for relative in IMPLEMENTATION_RELATIVES:
        require_committed_bytes(
            commit, relative, sha256_file(ROOT / relative), verifier=commit_verifier)
    attempt_id = authorization.get("attempt_id")
    require(isinstance(attempt_id, str) and attempt_id.strip(),
            "authorization", "attempt_id is required")
    require(authorization.get("output_root") == str(output),
            "authorization", "authorization output root differs")
    require(authorization.get("native_execution_allowed") is True,
            "authorization", "native execution is not allowed")
    require(authorization.get("single_attempt") is True,
            "authorization", "single_attempt must be true")
    require(isinstance(authorization.get("authorized_by"), str)
            and authorization["authorized_by"].strip(), "authorization",
            "authorized_by is required")
    require(authorization.get("max_optimizer_steps") == derived["optimizer_steps"],
            "authorization", "optimizer ceiling differs from executable schedule")
    require(authorization.get("max_generation_requests") == derived["generation_requests"],
            "authorization", "generation ceiling differs from executable schedule")
    restart_allowance = authorization.get("restart_allowance", 0)
    require(isinstance(restart_allowance, int) and not isinstance(restart_allowance, bool)
            and 0 <= restart_allowance <= 1, "authorization",
            "restart allowance must be zero or one")
    require(sha256_json({key: value for key, value in authorization.items()
                         if key != "receipt_sha256"})
            == authorization.get("receipt_sha256"), "authorization",
            "authorization self digest differs")
    context = SourceFreeContext(
        root=ROOT, config_path=config_path, config=config, config_sha256=config_sha,
        prepared=prepared, manifest=manifest, manifest_sha256=manifest["manifest_sha256"],
        authorization=authorization, authorization_sha256=sha256_file(authorization_path),
        output=output, attempt_id=attempt_id, protocol_commit=commit,
        binding=validate_tokenizer_binding(config["tokenizer_binding"]))
    if register:
        SourceFreeAttemptRegistry(authorization_path).register(context)
    return context, derived


def run_source_free_fit(*, config_path: str | Path, prepared: str | Path,
                        authorization_path: str | Path, output: str | Path,
                        resume: bool = False, require_head: bool = True,
                        commit_verifier: Callable | None = None) -> dict:
    context, _ = verify_source_free_authorization(
        config_path=config_path, prepared=prepared,
        authorization_path=authorization_path, output=output,
        require_head=require_head, commit_verifier=commit_verifier)
    backend = NativeP1Backend(context, "fit")
    try:
        groups = set(context.config["prepared"]["source_groups"])
        teaching = [row for row in read_jsonl(prepared / "teaching.jsonl")
                    if row["source_group"] in groups]
        labels = {row["case_id"]: row for row in read_jsonl(prepared / "sealed_labels_fit.jsonl")}
        cases = [{**row, "target": labels[row["case_id"]]["target"]}
                 for row in read_jsonl(prepared / "fit_cases.jsonl")
                 if row["source_group"] in groups]
        state = run_fit_diagnostic(
            backend, teaching, cases, output, prepared=prepared, resume=resume)
        state["backbone"] = backend.assert_backbone_unchanged()
        state["evidence_class"] = "native_mps"
        state["native_execution_admissible"] = True
        state["fit_is_learnability_only"] = True
        state["development_or_final_access"] = False
        return state
    finally:
        backend.close()
