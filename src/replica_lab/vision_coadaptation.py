"""Conditional co-adaptation scope, mutable-encoder service and gates.

The co-adaptation arm keeps every axis of the lr 3e-05 smaller-step arm except
gradient scope: the six projector tensors gain the last SigLIP2 encoder block
(16 tensors) and ``vision_model.post_layernorm`` (2 tensors), for 24 tensors and
18,631,424 parameters.  The sealed MiniCPM5 decoder, the frozen encoder layers
0-10, embeddings and head never receive gradients.

This module is deliberately separate from the frozen
``VisionFeatureService``: that service keeps its ``no_grad``/detached contract
untouched.  Here the frozen prefix (embeddings and layers 0-10) is evaluated
under ``no_grad`` and cached under an explicit producer version, while the
adapted block 11 and post-layernorm are evaluated with autograd so gradients
reach exactly the eighteen encoder tensors.  Final features are never served
from a stale producer: a producer-version change invalidates the cache.

All native imports are local to the functions that need them.  Importing this
module is CPU-only and never loads a model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .architecture.contracts import VisualInput, tensor_stamp

PROTOCOL_FORMAT = "replica.vision_coadaptation_control.protocol.v2"
PROTOCOL_VERSION = "replica.vision_coadaptation_control.v2"
NAME = "vision_coadaptation_control_v2"
ASSESSMENT_FORMAT = "replica.vision_coadaptation_control.assessment.v2"
AUTHORIZATION_FORMAT = "replica.vision_coadaptation.authorization.v2"
REVIEW_FORMAT = "replica.vision_coadaptation.review.v2"
DIAGNOSIS_FORMAT = "replica.vision_smaller_step.frozen_feature_diagnosis.v1"

ADAPTED_LAYER_INDEX = 11
PROJECTOR_TENSORS = (
    "vision_projector.0.weight",
    "vision_projector.0.bias",
    "vision_projector.1.weight",
    "vision_projector.1.bias",
    "vision_projector.3.weight",
    "vision_projector.3.bias",
)
ENCODER_TENSORS = (
    "vision_model.encoder.layers.11.self_attn.q_proj.weight",
    "vision_model.encoder.layers.11.self_attn.q_proj.bias",
    "vision_model.encoder.layers.11.self_attn.k_proj.weight",
    "vision_model.encoder.layers.11.self_attn.k_proj.bias",
    "vision_model.encoder.layers.11.self_attn.v_proj.weight",
    "vision_model.encoder.layers.11.self_attn.v_proj.bias",
    "vision_model.encoder.layers.11.self_attn.out_proj.weight",
    "vision_model.encoder.layers.11.self_attn.out_proj.bias",
    "vision_model.encoder.layers.11.layer_norm1.weight",
    "vision_model.encoder.layers.11.layer_norm1.bias",
    "vision_model.encoder.layers.11.layer_norm2.weight",
    "vision_model.encoder.layers.11.layer_norm2.bias",
    "vision_model.encoder.layers.11.mlp.fc1.weight",
    "vision_model.encoder.layers.11.mlp.fc1.bias",
    "vision_model.encoder.layers.11.mlp.fc2.weight",
    "vision_model.encoder.layers.11.mlp.fc2.bias",
    "vision_model.post_layernorm.weight",
    "vision_model.post_layernorm.bias",
)
SCOPE_TENSORS = PROJECTOR_TENSORS + ENCODER_TENSORS
PROJECTOR_PARAMETERS = 11542016
ENCODER_PARAMETERS = 7089408
TRAINABLE_PARAMETERS = PROJECTOR_PARAMETERS + ENCODER_PARAMETERS
ENCODER_DRIFT = {"max_abs_update": 1e-4, "max_delta_l2": 0.25}
FROZEN_ENCODER_LAYERS = "embeddings plus layers 0-10 plus head"
FROZEN_COMPONENTS = (
    "sealed MiniCPM5 parent",
    "frozen SigLIP2 vision encoder layers 0-10, embeddings and head",
)


class CoadaptationError(ValueError):
    """The co-adaptation protocol or native scope is invalid."""


class ScopeError(CoadaptationError):
    """Trainable/gradient ownership differs from the declared 24 tensors."""


class AuthorizationRefused(CoadaptationError):
    """Native execution is refused before any model load."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise CoadaptationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def expected_shapes() -> dict[str, tuple[int, ...]]:
    """Exact shapes for the 24 owned tensors of the SigLIP2 base model."""
    hidden, inter = 768, 3072
    shapes: dict[str, tuple[int, ...]] = {
        "vision_projector.0.weight": (hidden,),
        "vision_projector.0.bias": (hidden,),
        "vision_projector.1.weight": (4096, hidden),
        "vision_projector.1.bias": (4096,),
        "vision_projector.3.weight": (2048, 4096),
        "vision_projector.3.bias": (2048,),
    }
    layer = "vision_model.encoder.layers.11"
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        shapes[f"{layer}.self_attn.{name}.weight"] = (hidden, hidden)
        shapes[f"{layer}.self_attn.{name}.bias"] = (hidden,)
    for name in ("layer_norm1", "layer_norm2"):
        shapes[f"{layer}.{name}.weight"] = (hidden,)
        shapes[f"{layer}.{name}.bias"] = (hidden,)
    shapes[f"{layer}.mlp.fc1.weight"] = (inter, hidden)
    shapes[f"{layer}.mlp.fc1.bias"] = (inter,)
    shapes[f"{layer}.mlp.fc2.weight"] = (hidden, inter)
    shapes[f"{layer}.mlp.fc2.bias"] = (hidden,)
    shapes["vision_model.post_layernorm.weight"] = (hidden,)
    shapes["vision_model.post_layernorm.bias"] = (hidden,)
    _require(set(shapes) == set(SCOPE_TENSORS),
             "Declared co-adaptation tensor shapes differ")
    _require(sum(_elements(shape) for shape in shapes.values())
             == TRAINABLE_PARAMETERS,
             "Declared co-adaptation parameter total differs")
    return shapes


def _elements(shape: Sequence[int]) -> int:
    total = 1
    for value in shape:
        total *= int(value)
    return total


def projector_elements() -> int:
    shapes = expected_shapes()
    return sum(_elements(shapes[name]) for name in PROJECTOR_TENSORS)


def encoder_elements() -> int:
    shapes = expected_shapes()
    return sum(_elements(shapes[name]) for name in ENCODER_TENSORS)


def scope_report(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that ``parameters`` is exactly the declared 24-tensor scope.

    A parameter may be a ``torch.nn.Parameter`` or any object exposing
    ``numel()`` and ``dtype``.  Names must be the unprefixed native names.
    """
    names = set(parameters)
    _require(names == set(SCOPE_TENSORS),
             f"Trainable scope differs: {sorted(names ^ set(SCOPE_TENSORS))}")
    projector = [parameters[name] for name in PROJECTOR_TENSORS]
    encoder = [parameters[name] for name in ENCODER_TENSORS]
    projector_count = sum(int(value.numel()) for value in projector)
    encoder_count = sum(int(value.numel()) for value in encoder)
    _require(projector_count == PROJECTOR_PARAMETERS,
             f"Projector parameter count differs: {projector_count}")
    _require(encoder_count == ENCODER_PARAMETERS,
             f"Encoder parameter count differs: {encoder_count}")
    _require(projector_count + encoder_count == TRAINABLE_PARAMETERS,
             "Co-adaptation parameter total differs")
    shapes = expected_shapes()
    for name, value in parameters.items():
        shape = tuple(int(item) for item in value.shape)
        _require(shape == shapes[name],
                 f"Trainable tensor shape differs: {name} {shape}")
    return {
        "format": "replica.vision_coadaptation.scope.v1",
        "trainable_tensors": list(SCOPE_TENSORS),
        "projector_tensors": len(PROJECTOR_TENSORS),
        "encoder_tensors": len(ENCODER_TENSORS),
        "projector_parameter_count": projector_count,
        "encoder_parameter_count": encoder_count,
        "trainable_parameter_count": projector_count + encoder_count,
        "adapted_layer_index": ADAPTED_LAYER_INDEX,
        "frozen_encoder_layers": FROZEN_ENCODER_LAYERS,
        "decoder_trainable_tensors": 0,
        "out_of_scope_trainable_tensors": 0,
    }


def enable_scope(parameters: Mapping[str, Any], *,
                 allow_extra: bool = False) -> dict[str, Any]:
    """Set ``requires_grad`` to exactly the declared 24 tensors.

    ``parameters`` is a name->parameter mapping covering at least the declared
    scope.  Every other entry is frozen.  With ``allow_extra`` the mapping may
    contain any number of named tensors (decoder, frozen encoder layers, ...).
    """
    report = scope_report({name: parameters[name] for name in SCOPE_TENSORS})
    for name, parameter in parameters.items():
        if name in SCOPE_TENSORS:
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(False)
    trainable = [name for name, value in parameters.items()
                 if getattr(value, "requires_grad", False)]
    _require(sorted(trainable) == sorted(SCOPE_TENSORS),
             f"Trainable parameter ownership differs: {trainable}")
    report["trainable_names"] = sorted(trainable)
    report["frozen_names"] = sorted(set(parameters) - set(SCOPE_TENSORS))
    return report


def scope_gradient_names(named_parameters: Iterable[tuple[str, Any]],
                         allowed: Sequence[str] = SCOPE_TENSORS
                         ) -> dict[str, list[str]]:
    """Return gradient ownership without importing torch."""
    present, nonzero, nonfinite = [], [], []
    for name, parameter in named_parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        present.append(name)
        if bool(gradient.ne(0).any()):
            nonzero.append(name)
        if not bool(gradient.isfinite().all()):
            nonfinite.append(name)
    unexpected = sorted(set(present) - set(allowed))
    return {
        "present": sorted(present),
        "nonzero": sorted(nonzero),
        "nonfinite": sorted(nonfinite),
        "unexpected": unexpected,
        "within_scope": not unexpected and set(present) <= set(allowed),
    }


def assert_gradient_ownership(report: Mapping[str, Any], *,
                              expect_nonzero: bool = True) -> None:
    _require(not report["unexpected"],
             f"Out-of-scope gradients: {report['unexpected']}")
    _require(not report["nonfinite"],
             f"Nonfinite gradients: {report['nonfinite']}")
    _require(set(report["present"]) <= set(SCOPE_TENSORS),
             "Gradient scope exceeds the declared tensors")
    if expect_nonzero:
        _require(set(report["present"]) == set(SCOPE_TENSORS),
                 "Not every declared scope tensor produced a gradient")
        _require(sorted(report["nonzero"]) == sorted(SCOPE_TENSORS),
                 "A declared scope tensor has an all-zero gradient")


def snapshot_tensor_state(named_parameters: Iterable[tuple[str, Any]],
                          names: Sequence[str] = SCOPE_TENSORS
                          ) -> dict[str, Any]:
    """Clone FP32 tensors by name for a later drift check."""
    wanted = set(names)
    state = {}
    for name, parameter in named_parameters:
        if name not in wanted:
            continue
        state[name] = parameter.detach().to("cpu", copy=True).clone()
    _require(set(state) == wanted,
             f"Missing tensors for drift check: {sorted(wanted - set(state))}")
    return state


def drift_report(before: Mapping[str, Any], after: Mapping[str, Any],
                 *, scope: Sequence[str] = SCOPE_TENSORS) -> dict[str, Any]:
    """Compute max absolute and L2 parameter drift for the declared scope."""
    _require(set(before) == set(after) == set(scope),
             "Drift inputs must cover exactly the declared scope")
    per_tensor = {}
    max_abs = 0.0
    delta_l2_sq = 0.0
    for name in scope:
        delta = after[name].float() - before[name].float()
        value_abs = float(delta.abs().max().item())
        value_l2 = float(delta.norm().item())
        per_tensor[name] = {
            "max_abs_update": value_abs,
            "delta_l2": value_l2,
            "changed": value_abs > 0.0,
        }
        max_abs = max(max_abs, value_abs)
        delta_l2_sq += value_l2 * value_l2
    return {
        "format": "replica.vision_coadaptation.drift.v1",
        "scope_tensors": len(scope),
        "max_abs_update": max_abs,
        "delta_l2": math_sqrt(delta_l2_sq),
        "per_tensor": per_tensor,
        "changed_tensors": sorted(name for name, value in per_tensor.items()
                                  if value["changed"]),
    }


def math_sqrt(value: float) -> float:
    return float(value) ** 0.5


def validate_drift(report: Mapping[str, Any], *,
                   limits: Mapping[str, Any] = ENCODER_DRIFT) -> dict[str, Any]:
    _require(report["max_abs_update"] <= float(limits["max_abs_update"]),
             f"Encoder max-abs drift exceeded: {report['max_abs_update']} > "
             f"{limits['max_abs_update']}")
    _require(report["delta_l2"] <= float(limits["max_delta_l2"]),
             f"Encoder L2 drift exceeded: {report['delta_l2']} > "
             f"{limits['max_delta_l2']}")
    _require(set(report["changed_tensors"]) <= set(SCOPE_TENSORS),
             "Drift changed an undeclared tensor")
    return dict(report)


def producer_version(named_parameters: Iterable[tuple[str, Any]],
                     *, extra: Mapping[str, Any] | None = None) -> str:
    """Hash every producer-relevant tensor identity and version."""
    entries = []
    for name, parameter in named_parameters:
        gradient = getattr(parameter, "grad", None)
        entries.append({
            "name": str(name),
            "shape": [int(item) for item in parameter.shape],
            "dtype": str(parameter.dtype),
            "requires_grad": bool(getattr(parameter, "requires_grad", False)),
            "version": int(getattr(parameter, "_version", 0)),
            "data_ptr": int(getattr(parameter, "data_ptr", lambda: 0)()),
            "has_grad": gradient is not None,
        })
    payload = {"tensors": sorted(entries, key=lambda item: item["name"]),
               "extra": dict(extra or {})}
    return _sha256_bytes(_canonical(payload).encode())


class VersionedFeatureCache:
    """A tiny cache that never serves features across a producer change."""

    def __init__(self, *, max_entries: int = 64):
        _require(type(max_entries) is int and max_entries > 0,
                 "max_entries must be a positive integer")
        self.max_entries = int(max_entries)
        self._version: str | None = None
        self._entries: dict[str, tuple[Any, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._invalidations: list[str] = []

    @property
    def version(self) -> str | None:
        return self._version

    def invalidate(self, reason: str) -> dict[str, Any]:
        _require(isinstance(reason, str) and reason.strip(),
                 "Invalidation reason is required")
        self._entries.clear()
        self._invalidations.append(reason)
        return self.audit()

    def _ensure_version(self, version: str) -> None:
        _require(isinstance(version, str) and len(version) == 64,
                 "Cache version must be a SHA256")
        if self._version is None:
            self._version = version
        elif self._version != version:
            self._entries.clear()
            self._invalidations.append("producer_version_changed")
            self._version = version

    def lookup(self, key: str, *, version: str) -> tuple[Any, Any] | None:
        self._ensure_version(version)
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        # Preserve LRU ordering.
        self._entries.pop(key)
        self._entries[key] = entry
        return entry

    def put(self, key: str, features: Any, valid: Any, *,
            version: str) -> None:
        self._ensure_version(version)
        self._entries.pop(key, None)
        self._entries[key] = (features, valid)
        while len(self._entries) > self.max_entries:
            self._entries.pop(next(iter(self._entries)))

    def note_miss(self, *, version: str) -> None:
        """Record a forced miss (for training observations that bypass reuse)."""
        self._ensure_version(version)
        self._misses += 1

    def clear(self) -> None:
        self._entries.clear()

    def audit(self) -> dict[str, Any]:
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "version": self._version,
            "hits": self._hits,
            "misses": self._misses,
            "invalidations": list(self._invalidations),
        }


class MutableVisualInput(VisualInput):
    """A VisualInput whose features deliberately carry an autograd graph.

    The frozen path keeps ``VisualInput`` unchanged and still rejects
    grad-enabled features.  This subclass is only constructed by the mutable
    encoder service in a training request.
    """

    def __post_init__(self):
        # Re-run the base metadata contract while allowing requires_grad.
        for name in ("original_size", "crop", "grid"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (self.features.ndim != 3 or self.features.shape[0] != 1
                or self.features.shape[1] < 1
                or not self.features.is_floating_point()
                or not self.features.requires_grad
                or not bool(self.features.isfinite().all())):
            raise ValueError(
                "Mutable visual features must be finite grad-enabled "
                "[1, patches, width] tensors")
        import torch as _torch
        if (self.valid.shape != self.features.shape[:2]
                or self.valid.dtype != _torch.bool
                or self.valid.device != self.features.device
                or not bool(self.valid.any())):
            raise ValueError(
                "Mutable patch mask must be matching nonempty boolean state")
        for name in ("producer", "encoder_producer", "image_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError(f"MutableVisualInput.{name} must be a SHA256")
        for name in ("source_id", "task_id", "subject_id", "branch_id",
                     "purpose"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"MutableVisualInput.{name} is required")
        for name in ("observed_at", "source_epoch"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"MutableVisualInput.{name} must be nonnegative")
        if (len(self.original_size) != 2
                or any(type(value) is not int or value < 1
                       for value in self.original_size)
                or len(self.crop) != 4
                or any(type(value) is not int for value in self.crop)
                or not 0 <= self.crop[0] < self.crop[2] <= self.original_size[0]
                or not 0 <= self.crop[1] < self.crop[3] <= self.original_size[1]
                or len(self.grid) != 2
                or any(type(value) is not int or value < 1
                       for value in self.grid)
                or self.grid[0] * self.grid[1] != self.features.shape[1]):
            raise ValueError(
                "Mutable visual crop, dimensions or patch layout differs")
        object.__setattr__(
            self, "_tensor_stamps",
            tuple(tensor_stamp(value) for value in self.tensors()))

    def assert_immutable(self):
        if not self.features.requires_grad:
            raise RuntimeError("Mutable visual graph was detached")
        if self._tensor_stamps != tuple(
                tensor_stamp(value) for value in self.tensors()):
            raise RuntimeError("Mutable visual input changed after construction")


def _detached(value: Any) -> Any:
    return value.detach().to("cpu", dtype=__import__("torch").float32
                             ).contiguous()


class MutableEncoderFeatureService:
    """Grad-enabled last-block/post-norm features with a frozen-prefix cache.

    The encoder is expected to expose ``vision_model.embeddings``,
    ``vision_model.encoder.layers[0..11]`` and
    ``vision_model.post_layernorm`` (the standard ``SiglipVisionModel``
    layout).  Layers 0-10 are evaluated under ``no_grad`` and cached by image
    and producer version.  Layer 11 and post-layernorm run per observation; in
    training mode their output carries gradients to exactly the eighteen
    declared encoder tensors.
    """

    def __init__(self, *, encoder: Any, spec: Any, device: Any,
                 source: Any = None, processor: Any = None,
                 processor_stamp: str | None = None,
                 max_entries: int = 64):
        self._encoder = encoder
        self._spec = spec
        self._device = device
        self._source = source
        self._processor = processor
        self._processor_stamp = processor_stamp
        self._cache = VersionedFeatureCache(max_entries=max_entries)
        self._observations = 0
        self._encoder_calls = 0

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, source_path: str | Path, manifest_path: str | Path, *,
             device: str = "mps", max_entries: int = 64) -> "MutableEncoderFeatureService":
        """Load the sealed SigLIP2 encoder for mutable scope.

        Mirrors the frozen service safeguards: offline local load, float32,
        eval, exact source manifest and no CPU encoder fallback.
        """
        from .architecture import vision as vision_module
        source = vision_module.load_vision_source(manifest_path)
        if Path(source_path).resolve() != source.root.resolve():
            raise CoadaptationError(
                "source_path and source_manifest_path disagree")
        _require(device == "mps",
                 "The mutable encoder runs only on the declared MPS device")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        _require(torch.backends.mps.is_available(),
                 "Actual MPS is required; no CPU encoder fallback")
        processor = vision_module.load_pil_image_processor(source)
        processor_stamp = vision_module.processor_configuration_digest(processor)
        from transformers import SiglipVisionModel
        model = SiglipVisionModel.from_pretrained(
            source.root, local_files_only=True, force_download=False,
            trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, attn_implementation="eager")
        _require(type(model).__name__ == "SiglipVisionModel",
                 "Refusing a non-SiglipVisionModel encoder")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(torch.device("mps"))
        service = cls(encoder=model, spec=vision_module.vision_spec_from_source(
            source), device=torch.device("mps"), source=source,
            processor=processor, processor_stamp=processor_stamp,
            max_entries=max_entries)
        return service

    @classmethod
    def for_testing(cls, encoder: Any, *, spec: Any,
                    processor: Any = None) -> "MutableEncoderFeatureService":
        """Build around an injected tiny encoder; never native."""
        return cls(encoder=encoder, spec=spec, device=None,
                   processor=processor)

    # -- ownership ---------------------------------------------------------
    def named_parameters(self):
        return list(self._encoder.named_parameters())

    def producer_version(self, *, extra: Mapping[str, Any] | None = None) -> str:
        payload = {
            "source_id": None if self._source is None
            else getattr(self._source, "source_id", None),
            "processor_stamp": self._processor_stamp,
            "spec_digest": getattr(self._spec, "digest", None),
        }
        payload.update(dict(extra or {}))
        return producer_version(self.named_parameters(), extra=payload)

    def enable_scope(self, *, allow_extra: bool = True) -> dict[str, Any]:
        report = enable_scope(dict(self.named_parameters()),
                              allow_extra=allow_extra)
        return report

    # -- encoding ----------------------------------------------------------
    def _prefix(self, pixels: Any) -> Any:
        import torch
        transformer = self._encoder.vision_model
        with torch.no_grad():
            hidden = transformer.embeddings(pixel_values=pixels)
            for layer in transformer.encoder.layers[:ADAPTED_LAYER_INDEX]:
                hidden = layer(hidden)
        return hidden.detach()

    def _adapted(self, prefix: Any, *, mutable: bool):
        import torch
        transformer = self._encoder.vision_model
        context = torch.enable_grad() if mutable else torch.no_grad()
        with context:
            hidden = prefix
            if mutable:
                hidden.requires_grad_(True)
            hidden = transformer.encoder.layers[ADAPTED_LAYER_INDEX](hidden)
            features = transformer.post_layernorm(hidden)
        return features

    def _validate_features(self, features: Any) -> None:
        import torch
        _require(getattr(features, "ndim", 0) == 3,
                 "Encoder did not return [1, patches, width] features")
        expected_width = self._spec.encoder_width
        expected_tokens = self._spec.max_visual_tokens
        _require(tuple(features.shape)
                 == (1, expected_tokens, expected_width),
                 f"Encoder patch shape {tuple(features.shape)} differs from "
                 f"(1, {expected_tokens}, {expected_width})")
        _require(bool(torch.isfinite(features).all()),
                 "Encoder returned non-finite visual features")

    def _pixels(self, image_path: str | Path):
        import io
        from PIL import Image
        path = Path(image_path)
        size = path.stat().st_size
        _require(0 < size <= 64 * 1024 * 1024,
                 "Visual input image byte length is outside its bound")
        data = path.read_bytes()
        _require(len(data) == size, "Visual input image changed while reading")
        image_sha = _sha256_bytes(data)
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            rgb = opened.convert("RGB")
        if self._processor is None:
            raise CoadaptationError(
                "A native mutable service requires the official processor")
        pixels = self._processor(images=rgb, return_tensors="pt")
        values = pixels["pixel_values"] if isinstance(pixels, Mapping) else pixels
        import torch
        _require(tuple(values.shape) == (1, 3, 224, 224)
                 and values.dtype == torch.float32,
                 "Processor did not return float32 [1, 3, 224, 224] pixels")
        return values, image_sha

    def _cache_key(self, *, image_sha: str, source_id: str, task_id: str,
                   scope: Any, observed_at: int, source_epoch: int) -> str:
        return _sha256_bytes(_canonical({
            "image_sha256": image_sha,
            "source_id": source_id,
            "task_id": task_id,
            "subject_id": scope.subject_id,
            "branch_id": scope.branch_id,
            "purpose": scope.purpose,
            "observed_at": int(observed_at),
            "source_epoch": int(source_epoch),
            "version": self.producer_version(),
        }).encode())

    def _visual(self, features: Any, *, image_sha: str, source_id: str,
                task_id: str, scope: Any, observed_at: int,
                source_epoch: int, mutable: bool):
        import torch
        grid = (self._spec.image_size // self._spec.patch_size,) * 2
        device = (self._device if self._device is not None
                  else features.device)
        common = dict(
            valid=torch.ones((1, self._spec.max_visual_tokens),
                             dtype=torch.bool, device=device),
            producer=self._spec.digest,
            encoder_producer=self.producer_version(),
            image_sha256=image_sha,
            source_id=source_id,
            task_id=task_id,
            subject_id=scope.subject_id,
            branch_id=scope.branch_id,
            purpose=scope.purpose,
            observed_at=int(observed_at),
            source_epoch=int(source_epoch),
            original_size=(self._spec.image_size, self._spec.image_size),
            crop=(0, 0, self._spec.image_size, self._spec.image_size),
            grid=grid,
        )
        if mutable:
            features = features.to(device=device, copy=True).contiguous()
            _require(features.requires_grad,
                     "Mutable visual features lost their graph")
            return MutableVisualInput(features=features, **common)
        with torch.no_grad():
            detached = features.detach().to(device=device, copy=True
                                            ).contiguous()
        return VisualInput(features=detached, **common)

    def observe(self, image_path: str | Path, *, source_id: str,
                task_id: str, scope: Any, observed_at: int,
                source_epoch: int, training: bool = False):
        """Observe one image.

        ``training=True`` returns a ``MutableVisualInput`` whose features carry
        gradients and always executes the adapted block.  Evaluation returns a
        detached ``VisualInput`` and may serve a same-producer cached feature.
        """
        import torch
        pixels, image_sha = self._pixels(image_path)
        if self._device is not None:
            pixels = pixels.to(device=self._device, dtype=torch.float32)
        version = self.producer_version()
        key = self._cache_key(
            image_sha=image_sha, source_id=source_id, task_id=task_id,
            scope=scope, observed_at=observed_at,
            source_epoch=source_epoch)
        self._observations += 1
        if not training:
            cached = self._cache.lookup(key, version=version)
            if cached is not None:
                features, _ = cached
                return self._visual(
                    features, image_sha=image_sha, source_id=source_id,
                    task_id=task_id, scope=scope, observed_at=observed_at,
                    source_epoch=source_epoch, mutable=False)
        else:
            # A training observation must execute the adapted block; it can
            # never be served from a detached cache entry.
            self._cache.note_miss(version=version)
        prefix = self._prefix(pixels)
        features = self._adapted(prefix, mutable=training)
        self._validate_features(features)
        self._encoder_calls += 1
        if not training:
            self._cache.put(key, _detached(features),
                            torch.ones((1, self._spec.max_visual_tokens),
                                       dtype=torch.bool), version=version)
        return self._visual(
            features, image_sha=image_sha, source_id=source_id,
            task_id=task_id, scope=scope, observed_at=observed_at,
            source_epoch=source_epoch, mutable=training)

    # -- evidence ----------------------------------------------------------
    def invalidate(self, reason: str) -> dict[str, Any]:
        return self._cache.invalidate(reason)

    def cache_audit(self) -> dict[str, Any]:
        audit = self._cache.audit()
        audit.update({
            "producer_version": self.producer_version(),
            "observations": self._observations,
            "encoder_calls": self._encoder_calls,
            "observation_identity": (
                audit["hits"] + audit["misses"] == self._observations),
        })
        return audit

    def encoder_audit(self) -> dict[str, Any]:
        names = [name for name, _ in self.named_parameters()]
        trainable = [name for name, value in self.named_parameters()
                     if value.requires_grad]
        gradients = [name for name, value in self.named_parameters()
                     if value.grad is not None]
        return {
            "tensor_count": len(names),
            "trainable_names": sorted(trainable),
            "gradient_names": sorted(gradients),
            "frozen_layer_names": FROZEN_ENCODER_LAYERS,
        }


def migrate_checkpoint(parent: Mapping[str, Any],
                       extension: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate a projector-only manifest to an encoder-inclusive manifest.

    Both inputs are JSON-like mappings with ``tensors`` (name -> SHA256),
    ``parameter_elements`` and ``scope``.  The parent manifest's exact digest
    is recorded, the six projector tensors must be byte-identical, and the
    eighteen encoder tensors must be complete with the declared element count.
    A mismatch refuses migration instead of silently widening scope.
    """
    _require(isinstance(parent, Mapping) and isinstance(extension, Mapping),
             "Checkpoint manifests must be mappings")
    parent_tensors = dict(parent.get("tensors") or {})
    extension_tensors = dict(extension.get("tensors") or {})
    _require(set(parent_tensors) == set(PROJECTOR_TENSORS),
             "Parent checkpoint must contain exactly the six projector tensors")
    missing_encoder = set(ENCODER_TENSORS) - set(extension_tensors)
    extra = set(extension_tensors) - set(SCOPE_TENSORS)
    _require(not missing_encoder,
             f"Extension checkpoint is missing encoder tensors: "
             f"{sorted(missing_encoder)}")
    _require(not extra,
             f"Extension checkpoint has out-of-scope tensors: {sorted(extra)}")
    for name in PROJECTOR_TENSORS:
        _require(extension_tensors[name] == parent_tensors[name],
                 f"Extension checkpoint changed a projector tensor: {name}")
    parent_digest = _sha256_bytes(_canonical(parent).encode())
    declared_parent = extension.get("parent_manifest_sha256")
    _require(declared_parent == parent_digest,
             "Extension parent manifest digest differs")
    elements = extension.get("parameter_elements")
    _require(elements == TRAINABLE_PARAMETERS,
             f"Extension parameter count differs: {elements}")
    _require(len(extension_tensors) == len(SCOPE_TENSORS),
             "Extension checkpoint tensor count differs")
    merged = {
        "format": "replica.vision_coadaptation.checkpoint_manifest.v2",
        "parent_manifest_sha256": parent_digest,
        "parent_scope": list(PROJECTOR_TENSORS),
        "extension_scope": list(ENCODER_TENSORS),
        "scope": list(SCOPE_TENSORS),
        "tensor_count": len(SCOPE_TENSORS),
        "projector_parameter_count": PROJECTOR_PARAMETERS,
        "encoder_parameter_count": ENCODER_PARAMETERS,
        "parameter_elements": TRAINABLE_PARAMETERS,
        "tensors": {name: extension_tensors[name]
                    for name in SCOPE_TENSORS},
        "migration": {
            "from": "six_projector_tensors",
            "to": "six_projector_plus_last_block_and_post_layernorm",
            "parent_tensors_verified": list(PROJECTOR_TENSORS),
            "encoder_tensors_verified": list(ENCODER_TENSORS),
            "frozen_encoder_layers": FROZEN_ENCODER_LAYERS,
        },
    }
    _require(merged["tensor_count"] == 24
             and merged["parameter_elements"] == 18631424,
             "Migrated checkpoint declaration differs")
    return merged


def verify_checkpoint_state(manifest: Mapping[str, Any],
                            tensor_shas: Mapping[str, str]) -> dict[str, Any]:
    """Verify an encoder-inclusive checkpoint against an exact tensor map."""
    expected = dict(manifest.get("tensors") or {})
    actual = dict(tensor_shas)
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    moved = sorted(name for name in set(expected) & set(actual)
                   if expected[name] != actual[name])
    ok = not missing and not extra and not moved
    _require(ok, f"Checkpoint tensor verification failed: missing="
                 f"{sorted(missing)} extra={sorted(extra)} changed={moved}")
    return {
        "ok": True,
        "tensor_count": len(actual),
        "parameter_elements": manifest["parameter_elements"],
        "verified_tensors": sorted(actual),
    }


# ---------------------------------------------------------------------------
# Protocol validation, eligibility and authorization (CPU only)
# ---------------------------------------------------------------------------

PROTOCOL_KEYS = {
    "format", "version", "mode", "name", "declared_at_utc", "purpose",
    "hypothesis", "parent", "prepared", "schedule", "initialization",
    "optimizer", "loss_contract", "gradient_scope", "scope_axis",
    "update_batch", "readouts", "gates", "accounting", "budget", "limits", "policy",
    "access", "precondition", "implementation", "authorization", "statuses",
    "claims", "distinctness", "limitations", "output", "report_root",
}


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    _require(set(value) == keys,
             f"{label} fields differ: {sorted(set(value) ^ keys)}")


def validate_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the v2 co-adaptation protocol without native imports."""
    _exact_keys(config, PROTOCOL_KEYS, "co-adaptation protocol")
    _require(config["format"] == PROTOCOL_FORMAT
             and config["version"] == PROTOCOL_VERSION
             and config["name"] == NAME,
             "Co-adaptation protocol identity differs")
    _require(config["mode"] == "coadaptation_control",
             "Co-adaptation mode differs")
    scope = config["gradient_scope"]
    _require(scope["trainable_tensors"] == list(SCOPE_TENSORS),
             "Declared trainable tensors differ")
    _require(scope["projector_tensors"] == list(PROJECTOR_TENSORS),
             "Declared projector tensors differ")
    _require(scope["encoder_tensors"] == list(ENCODER_TENSORS),
             "Declared encoder tensors differ")
    _require(scope["projector_parameter_count"] == PROJECTOR_PARAMETERS
             and scope["encoder_parameter_count"] == ENCODER_PARAMETERS
             and scope["trainable_parameter_count"] == TRAINABLE_PARAMETERS,
             "Declared parameter counts differ")
    _require(scope["optimizer_owned_parameters"] == len(SCOPE_TENSORS)
             and scope["optimizer_param_groups"] == 1,
             "Optimizer ownership declaration differs")
    _require(scope["decoder_gradient_tensors_allowed"] == 0
             and scope["backbone_gradient_tensors_allowed"] == 0
             and scope["non_scope_gradient_names_allowed"] == [],
             "Decoder/backbone gradients must remain forbidden")
    _require(scope["encoder_update_bounds"] == ENCODER_DRIFT,
             "Encoder drift bounds differ")
    structure = scope["encoder_structure"]
    _require(structure["adapted_layer_index"] == ADAPTED_LAYER_INDEX
             and structure["num_hidden_layers"] == 12,
             "Adapted layer declaration differs")
    _require(structure["post_layernorm"] is True,
             "Post-layernorm adaptation is required")
    _require(config["optimizer"]["name"] == "adamw_fused"
             and config["optimizer"]["wrapper"] == "NativeFusedAdamW"
             and config["optimizer"]["lr"] == 3e-05,
             "Optimizer must remain NativeFusedAdamW at lr 3e-05")
    _require(config["loss_contract"]["same_loss_as_smaller_step"] is True
             and config["loss_contract"]["new_loss_introduced"] is False,
             "Loss contract must be unchanged")
    _require(config["accounting"]["format"] == "replica.vision_smaller_step.accounting.v2",
             "Co-adaptation must reuse the v2 derived accounting")
    from . import vision_smaller_step_accounting_v2 as acc_v2
    acc_v2.validate_accounting(config["accounting"])
    campaign = config["accounting"]["campaign_cumulative"]
    _require(campaign["decoder_forward_calls"] == 4804
             and campaign["visual_observation_calls"] == 156
             and campaign["visual_forward_calls"] == 4544,
             "Co-adaptation budget differs from the matched smaller-step arm")
    for key in ("decoder_forward_calls", "backward_calls",
                "optimizer_steps", "generated_tokens", "visual_forward_calls"):
        _require(config["accounting"]["campaign_cumulative"][key]
                 in (4804, 4, 1, 3800, 4544),
                 f"Co-adaptation campaign {key} differs")
    gates = config["gates"]
    _require(gates["optimization"]["optimizer_owned_tensors"]
             == len(SCOPE_TENSORS)
             and gates["optimization"]["encoder_tensors"]
             == len(ENCODER_TENSORS)
             and gates["optimization"]["projector_tensors"]
             == len(PROJECTOR_TENSORS),
             "Optimization gate scope differs")
    _require(gates["optimization"]["decoder_gradient_tensors"] == 0
             and gates["optimization"]["backbone_gradient_tensors"] == 0,
             "Gate must forbid decoder/backbone gradients")
    _require(gates["optimization"]["max_encoder_abs_update"]
             == ENCODER_DRIFT["max_abs_update"]
             and gates["optimization"]["max_encoder_delta_l2"]
             == ENCODER_DRIFT["max_delta_l2"],
             "Encoder drift gate differs")
    _require(gates["decision"]["more_coadaptation_authorized"] is False,
             "Further co-adaptation must remain closed")
    _require(config["policy"]["no_checkpoint_writes"] is True
             and config["limits"]["checkpoint_writes"] == 0,
             "Protocol must not write checkpoints")
    _require(config["precondition"]["required"] is True
             and config["precondition"]["eligible"] is False,
             "Eligibility must remain unsatisfied at declaration")
    return dict(config)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _valid_commit(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 40
            and all(char in "0123456789abcdef" for char in value))


def eligibility_state(config: Mapping[str, Any], *,
                      root: str | Path = ".") -> dict[str, Any]:
    """Which preserved prerequisites exist for the co-adaptation control."""
    root_path = Path(root)
    precondition = config["precondition"]
    report_path = root_path / precondition["smaller_step_report"]["path"]
    diagnosis_path = root_path / precondition["diagnosis"]["path"]
    violations: list[str] = []
    report_ok = False
    if not report_path.is_file():
        violations.append("smaller_step_report_missing")
    else:
        report = _read_json(report_path)
        status = str(report.get("status") or "")
        acceptable = {"optimization_gate_failed",
                      "smaller_step_functional_fit_failed"}
        report_ok = status in acceptable
        if not report_ok:
            violations.append("smaller_step_report_not_failed")
        if _sha256_file(report_path) != precondition["smaller_step_report"].get(
                "sha256"):
            violations.append("smaller_step_report_sha256_differs")
        if report.get("dev_authorized") is not False:
            violations.append("smaller_step_report_authorized_dev")
    diagnosis_ok = False
    diagnosis = None
    if not diagnosis_path.is_file():
        violations.append("frozen_feature_diagnosis_missing")
    else:
        diagnosis = _read_json(diagnosis_path)
        diagnosis_ok = (
            diagnosis.get("format") == DIAGNOSIS_FORMAT
            and diagnosis.get("status") == "diagnosed"
            and diagnosis.get("conclusion")
            == "frozen_backbone_projection_insufficient"
            and bool(diagnosis.get("smaller_step_report_sha256")))
        if not diagnosis_ok:
            violations.append("frozen_feature_diagnosis_invalid")
        else:
            excluded = set(diagnosis.get("excluded_causes") or [])
            required_excluded = {
                "step_size", "labels_or_masking", "accounting",
                "precondition", "seed_or_initialization"}
            if not required_excluded <= excluded:
                violations.append("diagnosis_does_not_exclude_competing_causes")
                diagnosis_ok = False
    eligible = report_ok and diagnosis_ok and not violations
    return {
        "eligible": eligible,
        "violations": violations,
        "smaller_step_report": {
            "path": str(report_path),
            "present": report_path.is_file(),
            "failed": report_ok,
            "sha256": (_sha256_file(report_path)
                       if report_path.is_file() else None),
        },
        "diagnosis": {
            "path": str(diagnosis_path),
            "present": diagnosis_path.is_file(),
            "valid": diagnosis_ok,
            "conclusion": (None if diagnosis is None
                           else diagnosis.get("conclusion")),
        },
        "note": ("Eligibility requires a preserved failed smaller-step report "
                 "and a preserved frozen-feature diagnosis; an engineering "
                 "stop, a passing arm or a missing diagnosis is not "
                 "eligibility."),
    }


def _git_blob_sha256(root: Path, commit: str, relative: str) -> str | None:
    import subprocess
    try:
        blob = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob",
             f"{commit}:{relative}"], check=False, capture_output=True)
    except OSError:
        return None
    if blob.returncode != 0:
        return None
    return _sha256_bytes(blob.stdout)


def validate_receipt(receipt: Mapping[str, Any], *, config: Mapping[str, Any],
                     config_sha256: str, config_path: str | Path,
                     root: str | Path = ".", verify_commit: bool = True
                     ) -> dict[str, Any]:
    """Validate an external authorization receipt; never mint or infer one."""
    required = {
        "format", "status", "protocol_name", "protocol_path",
        "protocol_commit", "config_sha256", "native_execution_authorized",
        "dev_authorized", "heldout_authorized", "v0_authorized",
        "semantic_audit_authorized", "promotion_authorized",
        "retry_authorized", "issued_at_utc",
    }
    _require(set(receipt) == required,
             f"Receipt fields differ: {sorted(set(receipt) ^ required)}")
    _require(receipt["format"] == AUTHORIZATION_FORMAT,
             "Receipt format differs")
    _require(receipt["status"] == "authorized",
             "Native execution is not authorized")
    _require(receipt["protocol_name"] == config["name"],
             "Receipt names a different protocol")
    _require(receipt["protocol_path"] == Path(config_path).as_posix(),
             "Receipt binds a different protocol path")
    _require(receipt["config_sha256"] == config_sha256,
             "Receipt does not bind the exact config")
    _require(_valid_commit(receipt["protocol_commit"]),
             "Receipt protocol_commit is not a full commit")
    _require(receipt["native_execution_authorized"] is True,
             "Receipt does not authorize native execution")
    for key in ("dev_authorized", "heldout_authorized", "v0_authorized",
                "semantic_audit_authorized", "promotion_authorized",
                "retry_authorized"):
        _require(receipt[key] is False, f"Receipt opens a closed role: {key}")
    _require(isinstance(receipt["issued_at_utc"], str)
             and receipt["issued_at_utc"].strip(),
             "Receipt lacks an issue time")
    if verify_commit:
        root_path = Path(root).resolve()
        path = Path(config_path)
        if not path.is_absolute():
            path = root_path / path
        relative = path.resolve().relative_to(root_path).as_posix()
        actual = _git_blob_sha256(root_path, receipt["protocol_commit"],
                                  relative)
        _require(actual == config_sha256,
                 "Committed protocol bytes do not match the receipt")
    return dict(receipt)


def review_launch(config_path: str | Path = "configs/vision_coadaptation_control_v2.json",
                  authorization_path: str | Path | None = None, *,
                  root: str | Path = ".", verify_commit: bool = True
                  ) -> dict[str, Any]:
    """Refuse or review the co-adaptation launch without native imports.

    Refusal is the default: a failed smaller-step report, a frozen-feature
    diagnosis, a separate receipt and a fresh output are all required.
    """
    root_path = Path(root)
    config = validate_protocol(_read_json(root_path / config_path))
    config_sha = _sha256_file(root_path / config_path)
    state = eligibility_state(config, root=root_path)
    receipt_path = Path(authorization_path or (
        root_path / config["authorization"]["receipt_path"]))
    if not receipt_path.is_absolute():
        receipt_path = root_path / receipt_path
    refusal: list[str] = list(state["violations"])
    receipt = None
    if not receipt_path.is_file():
        refusal.append("authorization_receipt_missing")
    else:
        receipt = validate_receipt(
            _read_json(receipt_path), config=config,
            config_sha256=config_sha, config_path=config_path,
            root=root_path, verify_commit=verify_commit)
    output = Path(root_path / config["output"])
    if output.exists():
        refusal.append("fresh_output_required")
    allowed = not refusal
    command = [
        "python", "scripts/vision_coadaptation_control_runner_v2.py", "run",
        "--config", str(config_path), "--output", config["output"],
        "--model-path", config["parent"]["model_path"],
        "--model-lock", config["parent"]["model_lock"],
        "--vision-source", config["parent"]["vision_source_manifest"],
        "--authorization", str(receipt_path),
    ]
    return {
        "format": REVIEW_FORMAT,
        "status": ("review_only_launch_refused" if not allowed
                   else "review_only_launch_ready"),
        "native_execution": False,
        "model_loads": 0,
        "config": str(config_path),
        "config_sha256": config_sha,
        "protocol_name": config["name"],
        "eligible": state["eligible"],
        "eligibility": state,
        "authorization": {
            "present": receipt is not None,
            "path": str(receipt_path),
            "status": (receipt or {}).get("status", "absent"),
            "protocol_commit": (receipt or {}).get("protocol_commit"),
        },
        "refusals": refusal,
        "launch_allowed": allowed,
        "launch_command": command,
        "review_command": [
            "python", "scripts/vision_coadaptation_control_runner_v2.py",
            "review-launch", "--config", str(config_path),
            "--authorization", str(receipt_path)],
        "handoff": {
            "requires_preserved_smaller_step_failure": True,
            "requires_frozen_feature_diagnosis": True,
            "requires_separate_authorization": True,
            "requires_full_commit_binding": True,
            "requires_fresh_output": True,
            "development_heldout_v0_semantic_promotion_retry_closed": True,
        },
    }
