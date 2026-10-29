"""Durable model-content identity, separate from process-local mutation stamps.

A verified source-artifact SHA-256 is preferred for sealed large checkpoints so
identity computation does not copy the full backbone. Tiny/custom models can be
hashed tensor-by-tensor with bounded temporary storage. No ``id()``, ``data_ptr``
or autograd version is included in either identity.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def _tensor_identity(name: str, tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    if value.numel():
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"name": name, "dtype": str(value.dtype),
            "shape": list(value.shape), "sha256": digest.hexdigest()}


def durable_model_identity(module: Any, *, source_identity: str | None = None,
                           config: Mapping[str, Any] | None = None) -> str:
    """Return content-based SHA-256 for a model or a verified source identity.

    ``source_identity`` must be an already verified artifact/content digest,
    not a filename or process-local stamp. When supplied, it is incorporated
    with the optional config rather than copying model tensors. Otherwise all
    parameters and buffers are hashed individually in name order.
    """
    if source_identity is not None:
        if not isinstance(source_identity, str) or len(source_identity) != 64 \
                or any(character not in "0123456789abcdef" for character in source_identity):
            raise ValueError("source_identity must be a lowercase SHA-256")
        payload = {"verified_source_sha256": source_identity,
                   "config": dict(config or {})}
    else:
        payload = {
            "parameters": [_tensor_identity(name, parameter)
                           for name, parameter in module.named_parameters()],
            "buffers": [_tensor_identity(name, buffer)
                        for name, buffer in module.named_buffers()],
            "config": dict(config or {}),
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["durable_model_identity"]
