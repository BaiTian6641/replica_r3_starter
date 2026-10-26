"""Publish complete local model/runtime checkpoints with a byte receipt."""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .training_state import save_runtime_state


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root):
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*")
            if p.is_file() and p.relative_to(root).as_posix() != "checkpoint_receipt.json"}


def _optimizer_state_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


@contextmanager
def _cpu_serialization_residency(model, optimizer):
    """Avoid simultaneous device residency and serializer-owned CPU copies."""
    parameters = getattr(model, "parameters", None)
    device = (next(model.parameters()).device if callable(parameters)
              else optimizer.param_groups[0]["params"][0].device)
    if device.type == "cpu":
        yield
        return
    model.to("cpu")
    _optimizer_state_to(optimizer, "cpu")
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        yield
    finally:
        model.to(device)
        _optimizer_state_to(optimizer, device)


def save_training_checkpoint(model, tokenizer, optimizer, path, *, step,
                             data_state, schedule_state, run_contract, manifest):
    """Rename a complete model + optimizer/RNG/cursor directory into place.

    Call between updates with gradients cleared. A failed partial directory is
    retained under its temporary name for diagnosis and is never resumable.
    This trusted local receipt is not a publisher signature or power-loss proof.
    """
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Use a new checkpoint directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    with _cpu_serialization_residency(model, optimizer):
        model.save_pretrained(temporary, manifest=manifest)
        tokenizer.save_pretrained(temporary / "base")
        metadata = save_runtime_state(temporary / "runtime.pt", optimizer, step=step,
                                      data_state=data_state, schedule_state=schedule_state,
                                      run_contract=run_contract)
        receipt = {"format_version": 1, "completed_steps": step,
                   "run_contract": run_contract, "contract_sha256": metadata["contract_sha256"],
                   "files_sha256": {name: file_digest(p) for name, p in sorted(_files(temporary).items())}}
        (temporary / "checkpoint_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temporary, destination)
    return {"path": str(destination), "step": step,
            "receipt_sha256": file_digest(destination / "checkpoint_receipt.json")}


def validate_training_checkpoint(path, run_contract):
    root = Path(path)
    receipt = json.loads((root / "checkpoint_receipt.json").read_text())
    if receipt.get("format_version") != 1 or receipt.get("run_contract") != run_contract:
        raise ValueError("Checkpoint training contract does not match")
    files = _files(root)
    if set(files) != set(receipt.get("files_sha256", {})):
        raise ValueError("Checkpoint file inventory differs from its receipt")
    expected_digest = hashlib.sha256(json.dumps(run_contract, sort_keys=True, separators=(",", ":"),
                                               ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if receipt.get("contract_sha256") != expected_digest:
        raise ValueError("Checkpoint contract digest differs")
    if "runtime.pt" not in files or "adapter_manifest.json" not in files:
        raise ValueError("Checkpoint is missing model/runtime state")
    if "base/config.json" not in files:
        raise ValueError("Checkpoint is missing native model config")
    if not {"base/tokenizer.json", "base/tokenizer_config.json"}.issubset(files):
        raise ValueError("Checkpoint is missing native tokenizer artifacts")
    weights = {name for name in files if name.startswith("base/")
               and name.count("/") == 1 and name.endswith(".safetensors")}
    if not weights:
        raise ValueError("Checkpoint is missing native model weights")
    if "base/model.safetensors.index.json" in files:
        index = json.loads(files["base/model.safetensors.index.json"].read_text())
        referenced = {"base/" + name for name in index["weight_map"].values()}
        if referenced != weights:
            raise ValueError("Checkpoint native weight shard index differs")
    for name, digest in receipt["files_sha256"].items():
        if file_digest(files[name]) != digest:
            raise ValueError(f"Checkpoint byte mismatch: {name}")
    # Memory-map tensor storage: metadata inspection must not allocate a full
    # multi-gigabyte optimizer copy before the model is loaded.
    runtime = torch.load(root / "runtime.pt", map_location="cpu", weights_only=True, mmap=True)
    metadata = runtime.get("metadata", {})
    if metadata.get("step") != receipt.get("completed_steps"):
        raise ValueError("Checkpoint receipt step differs from runtime state")
    if (metadata.get("run_contract") != run_contract
            or metadata.get("contract_sha256") != expected_digest):
        raise ValueError("Checkpoint runtime contract differs from receipt")
    return receipt
