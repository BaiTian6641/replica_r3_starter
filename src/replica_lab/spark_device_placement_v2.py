"""Device-only placement admission for the FP8-derived BF16 Spark runner.

The sealed source is never patched or cast. Hashing uses bounded CPU chunks,
including after placement; it retains no full-backbone copy and exports nothing.
"""
from __future__ import annotations

import hashlib
import json
import time

import torch


FORMAT = "replica.spark_device_placement.v2"
HASH_CHUNK_BYTES = 4 * 1024 * 1024


class DevicePlacementError(ValueError):
    pass


def _named_tensors(base):
    return [("parameter/" + name, tensor) for name, tensor in
            base.named_parameters(remove_duplicate=False)] + [
                ("buffer/" + name, tensor) for name, tensor in
                base.named_buffers(remove_duplicate=False)]


def _inventory(base):
    named = _named_tensors(base)
    if not named:
        raise DevicePlacementError("empty model has no placement to qualify")
    aliases = {}
    rows = []
    for name, tensor in named:
        if tensor.device.type == "meta" or tensor.layout != torch.strided:
            raise DevicePlacementError(f"unmaterialized or non-strided tensor: {name}")
        # Canonical Spark tensors are contiguous. Refuse rather than allocate a
        # potentially full-sized contiguous copy for an unexpected tensor view.
        if not tensor.is_contiguous():
            raise DevicePlacementError(f"non-contiguous source tensor: {name}")
        aliases.setdefault(id(tensor), []).append(name)
        rows.append((name, str(tensor.dtype), list(tensor.shape)))
    return named, rows, sorted(sorted(names) for names in aliases.values())


def _content_digest(named, rows):
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode())
    seen = set()
    byte_count = 0
    for name, tensor in named:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        digest.update(name.encode())
        flat = tensor.detach().reshape(-1)
        count = max(1, HASH_CHUNK_BYTES // tensor.element_size())
        for start in range(0, flat.numel(), count):
            chunk = flat[start:start + count].to(device="cpu").view(torch.uint8)
            digest.update(memoryview(chunk.numpy()))
        byte_count += tensor.numel() * tensor.element_size()
    return digest.hexdigest(), byte_count


def _require_tied_output(base):
    try:
        incoming = base.get_input_embeddings().weight
        outgoing = base.get_output_embeddings().weight
    except (AttributeError, TypeError) as error:
        raise DevicePlacementError("missing input/output embedding path") from error
    if incoming is not outgoing:
        raise DevicePlacementError("input/output embedding weights are not tied")


def place_frozen_spark_base(base, *, device):
    """Move all original tensors once; return an independently checked receipt.

    CPU is available solely for engineering fixtures. The native caller admits
    only MPS. Device placement does not request a dtype and cannot silently repair
    mixed/meta/offloaded models. The caller must discard a refused model.
    """
    started = time.monotonic()
    target = torch.device(device)
    if target.type not in {"cpu", "mps"} or target.index not in {None, 0}:
        raise DevicePlacementError("only explicit CPU-fixture or MPS placement is admitted")
    # CPU and MPS each have one local device for this contract.
    target = torch.device(target.type)
    before, rows, aliases = _inventory(base)
    devices = {str(tensor.device) for _, tensor in before}
    if len(devices) != 1:
        raise DevicePlacementError("heterogeneous source devices are not admitted")
    source_device = next(iter(devices))
    if torch.device(source_device).type not in {"cpu", target.type}:
        raise DevicePlacementError("source must be on CPU or the declared device")
    _require_tied_output(base)
    before_digest, byte_count = _content_digest(before, rows)
    # Do not retain original tensor references across device placement.
    del before
    base.requires_grad_(False)
    moved = base.to(device=target)
    if moved is not base:
        raise DevicePlacementError("model.to replaced the model owner")
    after, after_rows, after_aliases = _inventory(base)
    if any(tensor.device.type != target.type for _, tensor in after):
        raise DevicePlacementError("not every parameter and buffer reached the declared device")
    if rows != after_rows or aliases != after_aliases:
        raise DevicePlacementError("placement changed tensor inventory, dtype or aliases")
    _require_tied_output(base)
    if any(parameter.requires_grad for parameter in base.parameters()):
        raise DevicePlacementError("original model was not frozen")
    after_digest, after_bytes = _content_digest(after, after_rows)
    if before_digest != after_digest or byte_count != after_bytes:
        raise DevicePlacementError("placement changed original tensor contents")
    return {
        "format": FORMAT, "status": "verified", "source_device": source_device,
        "target_device": target.type, "parameter_names": len(list(base.named_parameters())),
        "buffer_names": len(list(base.named_buffers())), "unique_tensor_bytes": byte_count,
        "content_sha256_before": before_digest, "content_sha256_after": after_digest,
        "hash_chunk_bytes_max": HASH_CHUNK_BYTES, "bytes_hashed": byte_count * 2,
        "dtype_unchanged": True, "tied_output_preserved": True,
        "all_original_parameters_frozen": True, "wall_seconds": time.monotonic() - started,
        "full_backbone_copy_or_export": False,
    }


__all__ = ["DevicePlacementError", "place_frozen_spark_base"]
