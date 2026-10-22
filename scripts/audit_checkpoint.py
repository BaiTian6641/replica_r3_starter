#!/usr/bin/env python3
"""CPU-only, bounded-chunk comparison of sealed MiniCPM5 source and D0 export.

Run after the measured MPS process exits. This script never instantiates a model
or transfers a tensor to a GPU. Source BF16 values are promoted to FP32 before
comparison, so the training runner's initial dtype conversion is not an update.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHUNK_BYTES = 8 * 1024 * 1024
EXPECTED_TENSORS = 381
EXPECTED_PARAMETERS = 2_516_756_480


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_inventory(directory: Path) -> tuple[dict, list[Path]]:
    """Read tensor headers and check any native shard index against the files."""
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors checkpoint in {directory}")
    inventory = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in inventory:
                    raise ValueError(f"Duplicate tensor name: {name}")
                view = handle.get_slice(name)
                inventory[name] = {"file": path.name, "shape": view.get_shape(),
                                   "dtype": view.get_dtype()}
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        if weight_map != {name: item["file"] for name, item in inventory.items()}:
            raise ValueError(f"Safetensors index does not match files in {directory}")
    return inventory, files


def chunk_indices(shape: list[int], max_elements: int):
    """Yield slices with at most max_elements, including wide and 1D tensors."""
    if max_elements < 1:
        raise ValueError("Chunk budget must hold at least one FP32 element")
    if not shape:
        yield ()
        return
    if any(size == 0 for size in shape):
        return
    axis = 0
    while math.prod(shape[axis + 1:]) > max_elements:
        axis += 1
    trailing = math.prod(shape[axis + 1:])
    extent = max(1, max_elements // trailing)
    for prefix in itertools.product(*(range(size) for size in shape[:axis])):
        for start in range(0, shape[axis], extent):
            yield (*prefix, slice(start, min(start + extent, shape[axis])),
                   *(slice(None) for _ in shape[axis + 1:]))


def compare_checkpoints(source_path: Path, exported_base: Path, *,
                        expected_tensors: int | None = None,
                        expected_parameters: int | None = None,
                        chunk_bytes: int = CHUNK_BYTES) -> dict:
    """Compare exact names/shapes with BF16 -> FP32 normalization on CPU.

    Each normalized source, exported, or delta chunk is at most chunk_bytes.
    Several chunks and reduction temporaries coexist; this is a per-tensor
    chunk-size bound, not a whole-process RSS bound.
    """
    if not 4 <= chunk_bytes <= CHUNK_BYTES:
        raise ValueError("Chunk budget must be between 4 bytes and 8 MiB")
    source, source_files = tensor_inventory(source_path)
    exported, exported_files = tensor_inventory(exported_base)
    if source.keys() != exported.keys():
        missing = sorted(source.keys() - exported.keys())
        extra = sorted(exported.keys() - source.keys())
        raise ValueError(f"Tensor name mismatch: missing={missing}, extra={extra}")
    count = len(source)
    parameters = sum(math.prod(item["shape"]) for item in source.values())
    if expected_tensors is not None and count != expected_tensors:
        raise ValueError(f"Expected {expected_tensors} native tensors, found {count}")
    if expected_parameters is not None and parameters != expected_parameters:
        raise ValueError(f"Expected {expected_parameters} parameters, found {parameters}")
    for name, original in source.items():
        if original["shape"] != exported[name]["shape"]:
            raise ValueError(f"Tensor shape mismatch: {name}")
        if original["dtype"] != "BF16" or exported[name]["dtype"] != "F32":
            raise ValueError(f"Expected source BF16/export FP32 tensor: {name}")

    comparisons = {}
    with ExitStack() as stack:
        original_handles = {path.name: stack.enter_context(
            safe_open(str(path), framework="pt", device="cpu")) for path in source_files}
        exported_handles = {path.name: stack.enter_context(
            safe_open(str(path), framework="pt", device="cpu")) for path in exported_files}
        for name in sorted(source):
            original = source[name]
            shape = original["shape"]
            original_handle = original_handles[original["file"]]
            exported_handle = exported_handles[exported[name]["file"]]
            left_view = original_handle.get_slice(name)
            right_view = exported_handle.get_slice(name)
            changed = 0
            max_delta = 0.0
            for index in chunk_indices(shape, chunk_bytes // 4):
                # Scalar get_slice indexing varies between safetensors releases;
                # get_tensor is bounded to a single element in this branch.
                left = (original_handle.get_tensor(name) if not shape else left_view[index]).float()
                right = (exported_handle.get_tensor(name) if not shape else right_view[index])
                if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                    raise ValueError(f"Nonfinite checkpoint tensor: {name}")
                changed += int(torch.count_nonzero(left != right))
                max_delta = max(max_delta, float((right - left).abs().max()))
                del left, right
            comparisons[name] = {"shape": shape, "total": math.prod(shape),
                                 "changed_elements": changed,
                                 "max_abs_delta": max_delta,
                                 "source_dtype": original["dtype"],
                                 "exported_dtype": exported[name]["dtype"]}

    changed_tensors = sum(item["changed_elements"] > 0 for item in comparisons.values())
    def changed_flag(name):
        return (comparisons[name]["changed_elements"] > 0 if name in comparisons else None)
    return {
        "tensor_count": count, "parameter_count": parameters,
        "changed_tensor_count": changed_tensors,
        "unchanged_tensor_count": count - changed_tensors,
        "changed_elements": sum(item["changed_elements"] for item in comparisons.values()),
        "every_original_tensor_changed": changed_tensors == count,
        "input_embedding_changed": changed_flag("model.embed_tokens.weight"),
        "output_head_changed": changed_flag("lm_head.weight"),
        "exact_native_names_and_shapes_match": True,
        "source_normalized_dtype": "float32", "comparison_device": "cpu",
        "max_fp32_chunk_bytes": chunk_bytes,
        "tensors": comparisons,
        "exported_checkpoint_sha256": {path.name: sha256_file(path) for path in exported_files},
        "scope_note": (
            "Observed saved-weight changes are paired with the run's trainable-parameter, "
            "optimizer-ownership and gradient audits to establish full D0 scope. "
            "A tensor changing does not imply that every element received a nonzero gradient."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--exported-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output file to preserve earlier audits")
    from replica_lab.model_adapter import validate_model_lock
    lock = json.loads(args.source_lock.read_text())
    validate_model_lock(args.source_path, lock)
    result = compare_checkpoints(args.source_path, args.exported_base,
                                 expected_tensors=EXPECTED_TENSORS,
                                 expected_parameters=EXPECTED_PARAMETERS)
    report = {
        "status": "completed_cpu_checkpoint_comparison",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(args.source_path.resolve()),
        "exported_base": str(args.exported_base.resolve()),
        "source_repo_id": lock["repo_id"], "source_revision": lock["revision"],
        "source_lock_sha256": sha256_file(args.source_lock),
        "source_snapshot_files_sha256": lock["snapshot_files_sha256"],
        "audit_script_sha256": sha256_file(Path(__file__)),
        "expected_native_tensor_count": EXPECTED_TENSORS,
        "expected_native_parameter_count": EXPECTED_PARAMETERS,
        "eligibility_note": "No model instantiation or GPU allocations; run after measured training exits.",
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "changed_tensor_count": result["changed_tensor_count"],
                      "tensor_count": result["tensor_count"],
                      "input_embedding_changed": result["input_embedding_changed"],
                      "output_head_changed": result["output_head_changed"]}), flush=True)


if __name__ == "__main__":
    main()
