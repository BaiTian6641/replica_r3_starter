"""Frozen-base Spark2_5 adapter with a zero-initialized residual projector.

The adapter targets the pinned custom ``spark2_5`` ABI audited in
``reports/spark_preparation_20261025/run_01/assets``: ``model.embedding``,
``model.layers``, ``model.norm`` and a tied output projection implemented as
``F.linear(hidden_states, model.embedding.weight)``. It is not a Llama or
MiniCPM5 adapter and it does not implement a second decoder.

Original parameters are frozen and stamped before training. New extension
tensors are FP32: one projector with a zero-initialized final projection and,
optionally, ``MemoryAccessPort`` hypotheses at layers 11 and 23. The zero
initialization makes the first forward exactly equal to the untouched base;
the final projection receives a nonzero first-step gradient, after which the
branch carries signal. ``no_grad`` is never placed around the decoder.

Extension export is *extension-only*: it writes the new tensors and base
identity, never the checkpoint bytes. Reloading rejects a stale or different
base identity. The qualification runner is responsible for the fresh-process
logits/token comparison and for refusing non-MPS or fallback environments.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
import torch.nn.functional as F

from .ports import MemoryAccessPort
from .spine import BRANCHES, SpineCore, SpineForwardContext, SpineSpec
from replica_lab.model_identity import durable_model_identity
from .spark2_5 import (
    OFFICIAL_NATIVE_FP8, PrecisionIdentity, Spark2_5Config, Spark2_5ContractError,
    admit_load, build_extension_export, build_extensions, build_memory_ports,
    derivative_precision_binding, fingerprint, sha256_bytes,
    validate_extension_export,
)

ADAPTER_EXTENSION_SCHEMA = "replica.spark2_5_adapter_extension.v1"
REGIMES = ("modules_only",)
DEFAULT_MEMORY_SIZE = 256
DEFAULT_PORT_HEADS = 4
DEFAULT_PROJECTOR_HIDDEN = 1024


class SparkAdapterError(RuntimeError):
    """The adapter contract, base identity or extension payload was violated."""


class _VisionProjector(nn.Module):
    """Names match ``spark2_5.build_projector`` exactly: norm, fc1, fc2."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(self.norm(features))))


@dataclass
class InjectionAudit:
    """Evidence for one decoder traversal; tensors are intentionally absent."""

    traced_layers: tuple[int, ...]
    sliding_layers: tuple[int, ...]
    full_layers: tuple[int, ...]
    projector_layer: int
    port_layers: tuple[int, ...]
    injected_layers: tuple[int, ...]
    delta_max_abs: float
    projector_input_requires_grad: bool
    projector_input_connected: bool
    projector_input_requires_grad: bool
    memory_attention: dict | None = None

    def to_dict(self) -> dict:
        return {
            "traced_layers": list(self.traced_layers),
            "sliding_layers": list(self.sliding_layers),
            "full_layers": list(self.full_layers),
            "projector_layer": self.projector_layer,
            "port_layers": list(self.port_layers),
            "injected_layers": list(self.injected_layers),
            "delta_max_abs": self.delta_max_abs,
            "projector_input_connected": self.projector_input_connected,
            "projector_input_requires_grad": self.projector_input_requires_grad,
            "memory_attention": self.memory_attention,
        }


@dataclass
class _ForwardContext:
    use_projector: bool
    use_ports: bool
    projector_input: torch.Tensor | None
    memory: torch.Tensor | None
    memory_mask: torch.Tensor | None
    spine_context: SpineForwardContext | None = None
    enabled_spine_branches: tuple[str, ...] = ()
    intrinsic_memory: object | None = None
    intrinsic_placement: str = "block_input_query_to_post_block_residual"
    traced: list[tuple[int, str]] | None = None
    injected: list[int] | None = None
    deltas: list[float] | None = None
    projector_source: torch.Tensor | None = None

    pre_mlp_deltas: dict[int, torch.Tensor] | None = None
    pre_mlp_injected: set[int] | None = None
    capture_attention: bool = False
    attention_position: int | None = None
    intrinsic_audit: dict[int, dict] | None = None

    def __post_init__(self) -> None:
        self.traced = []
        self.injected = []
        self.deltas = []
        self.pre_mlp_deltas = {}
        self.pre_mlp_injected = set()
        self.intrinsic_audit = {}


def _config_fingerprint(base: nn.Module, config: Spark2_5Config) -> str:
    return fingerprint({
        "spark_config_digest": config.digest,
        "attention": "eager",
        "use_cache": False,
        "base_config_model_type": getattr(getattr(base, "config", None), "model_type", None),
    })


@dataclass(frozen=True)
class ExtensionFile:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def to_dict(self) -> dict:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


class Spark2_5Adapter(nn.Module):
    """One frozen Spark2_5 base plus explicitly owned FP32 extension tensors."""

    def __init__(
        self,
        base: nn.Module,
        config: Spark2_5Config,
        precision: PrecisionIdentity,
        *,
        device: str = "mps",
        capabilities=None,
        require_official_geometry: bool = True,
        projector_in_dim: int | None = None,
        projector_hidden_dim: int = DEFAULT_PROJECTOR_HIDDEN,
        projector_layer: int | None = None,
        zero_init_output: bool = True,
        memory_size: int = DEFAULT_MEMORY_SIZE,
        heads: int = DEFAULT_PORT_HEADS,
        spine_sites: tuple[int, ...] = (11, 23),
        spine_slots: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(config, Spark2_5Config):
            raise SparkAdapterError("config must be a validated Spark2_5Config")
        if not isinstance(precision, PrecisionIdentity):
            raise SparkAdapterError("precision must be a PrecisionIdentity")
        try:
            derivative_precision_binding(config, precision)
        except Spark2_5ContractError as error:
            raise SparkAdapterError(str(error)) from error
        if require_official_geometry:
            config.geometry.assert_official()
        admission = admit_load(precision, device=device, capabilities=capabilities)
        if getattr(precision, "kind", None) == OFFICIAL_NATIVE_FP8:
            raise SparkAdapterError(
                "native FP8 adapter construction is not admitted; use an FP8-capable "
                "backend and qualify activation quantization separately")
        self.admission = admission
        self.config = config
        self.precision = precision
        self.regime = "modules_only"
        self._validate_base(base)
        self.base = base
        attention = getattr(base.config, "_attn_implementation", "eager") or "eager"
        if attention != "eager":
            raise SparkAdapterError(
                f"training requires eager attention; loaded implementation is {attention!r}")
        self.attention_implementation = "eager"
        base.config.use_cache = False
        self._original_parameters = dict(base.named_parameters())
        for parameter in self._original_parameters.values():
            parameter.requires_grad_(False)

        hidden = config.geometry.hidden_size
        in_dim = hidden if projector_in_dim is None else int(projector_in_dim)
        if in_dim <= 0:
            raise SparkAdapterError("projector_in_dim must be positive")
        layer = self._default_projector_layer() if projector_layer is None else int(projector_layer)
        if not 0 <= layer < config.geometry.num_hidden_layers:
            raise SparkAdapterError("projector_layer is outside the decoder")
        if not isinstance(zero_init_output, bool) or not zero_init_output:
            raise SparkAdapterError(
                "the qualification projector requires zero_init_output=True for initial parity")
        if not 1 <= hidden:
            raise SparkAdapterError("invalid Spark hidden width")

        self.extension = nn.Module()
        self.extension.vision_projector = _VisionProjector(in_dim, int(projector_hidden_dim), hidden)
        nn.init.zeros_(self.extension.vision_projector.fc2.weight)
        if self.extension.vision_projector.fc2.bias is not None:
            nn.init.zeros_(self.extension.vision_projector.fc2.bias)
        self.extension.memory_ports = nn.ModuleDict()
        self.spine_spec = SpineSpec(hidden_size=hidden, slots=int(spine_slots),
                                    sites=tuple(int(site) for site in spine_sites))
        self.extension.spine = SpineCore(self.spine_spec)

        self.projector_in_dim = in_dim
        self.projector_hidden_dim = int(projector_hidden_dim)
        self.projector_layer = layer
        self.zero_init_output = True
        self.memory_size = int(memory_size)
        self.heads = int(heads)
        self._port_layers: tuple[int, ...] = ()
        if any(site >= config.geometry.num_hidden_layers for site in self.spine_spec.sites):
            raise SparkAdapterError("spine site is outside the native decoder")
        self._original_identity: tuple | None = None
        self.extension.to(device)
        self.assert_scope()

    # ------------------------------------------------------------------
    # Structure and ownership

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        # A device/dtype move can replace Parameter objects. Refresh ownership
        # before an optimizer or export is built, and keep extension FP32.
        self._original_parameters = dict(self.base.named_parameters())
        for parameter in self._original_parameters.values():
            parameter.requires_grad_(False)
        for name, parameter in self.extension.named_parameters():
            if parameter.dtype != torch.float32:
                raise SparkAdapterError(
                    f"device/dtype conversion moved extension tensor {name} off FP32")
        self.assert_scope()
        return result

    def _validate_base(self, base: nn.Module) -> None:
        model = getattr(base, "model", None)
        if model is None:
            raise SparkAdapterError("base must expose .model (Spark2_5Model)")
        embedding = getattr(model, "embedding", None)
        if not isinstance(embedding, nn.Embedding):
            raise SparkAdapterError("base.model.embedding must be the Spark input embedding")
        expected = (self.config.geometry.vocab_size, self.config.geometry.hidden_size)
        if tuple(embedding.weight.shape) != expected:
            raise SparkAdapterError(f"embedding shape {tuple(embedding.weight.shape)} != {expected}")
        layers = getattr(model, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise SparkAdapterError("base.model.layers must be a ModuleList")
        if len(layers) != self.config.geometry.num_hidden_layers:
            raise SparkAdapterError("decoder layer count differs from the validated config")
        if not hasattr(model, "norm"):
            raise SparkAdapterError("base.model.norm is required")
        model_type = getattr(getattr(base, "config", None), "model_type", None)
        if model_type is not None and model_type != "spark2_5":
            raise SparkAdapterError(f"adapter supports spark2_5, not {model_type!r}")
        output_embedding = None
        getter = getattr(base, "get_output_embeddings", None)
        if callable(getter):
            output_embedding = getter()
        if output_embedding is not None:
            if tuple(output_embedding.weight.shape) != expected:
                raise SparkAdapterError("output projection shape differs from the tied table")
            if output_embedding.weight.data_ptr() != embedding.weight.data_ptr():
                raise SparkAdapterError(
                    "Spark2_5 output embeddings must be tied to the input embedding")

    def _default_projector_layer(self) -> int:
        schedule = [index for index, kind in enumerate(self.config.geometry.layer_types)
                    if kind == "full_attention"]
        if schedule:
            return schedule[0]
        return self.config.geometry.num_hidden_layers - 1

    def set_regime(self, regime: str) -> None:
        if regime not in REGIMES:
            raise SparkAdapterError(
                "local Spark qualification is frozen-base adaptation only; "
                "full fine-tuning requires separately provisioned hardware")
        self.regime = regime

    def new_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, parameter in self.extension.named_parameters():
            yield f"extension.{name}", parameter

    def new_named_tensors(self) -> tuple[ExtensionFile, ...]:
        return tuple(ExtensionFile(name, tuple(parameter.shape), "FLOAT32")
                     for name, parameter in self.new_named_parameters())

    def assert_scope(self) -> None:
        if any(parameter.requires_grad for parameter in self._original_parameters.values()):
            raise SparkAdapterError("an original Spark parameter is not frozen")
        for name, parameter in self.new_named_parameters():
            if not parameter.requires_grad:
                raise SparkAdapterError(f"new extension tensor {name} is frozen")
            if parameter.dtype != torch.float32:
                raise SparkAdapterError(f"new extension tensor {name} is not FP32")
        enabled = {f"extension.memory_ports.{layer}" for layer in self._port_layers}
        actual = {".".join(name.split(".")[:3]) for name, _ in self.new_named_parameters()
                  if name.startswith("extension.memory_ports.")}
        if actual != enabled:
            raise SparkAdapterError("memory-port module ownership differs from the declared layers")

    def enable_memory_ports(self, layers: tuple[int, ...] | list[int] = (11, 23),
                            *, memory_size: int | None = None,
                            heads: int | None = None) -> dict:
        """Attach optional full-attention port hypotheses; zero-effect at init."""
        if self._port_layers:
            raise SparkAdapterError("memory ports are already enabled")
        requested = tuple(int(layer) for layer in layers)
        if len(set(requested)) != len(requested):
            raise SparkAdapterError("memory port layers must be distinct")
        specs = build_memory_ports(self.config, layers=requested)
        size = self.memory_size if memory_size is None else int(memory_size)
        head_count = self.heads if heads is None else int(heads)
        if size <= 0 or head_count <= 0 or size % head_count:
            raise SparkAdapterError("positive head-divisible memory_size required")
        for spec in specs:
            port = MemoryAccessPort(self.config.geometry.hidden_size, size, head_count)
            port.to(next(self.extension.vision_projector.parameters()).device)
            port.train(self.training)
            self.extension.memory_ports[str(spec.layer)] = port
        self._port_layers = tuple(spec.layer for spec in specs)
        self.memory_size = size
        self.heads = head_count
        self.assert_scope()
        return {"layers": list(self._port_layers), "memory_size": size,
                "heads": head_count, "zero_init_output": True,
                "status": "experimental_identity_change"}

    @property
    def port_layers(self) -> tuple[int, ...]:
        return self._port_layers

    # ------------------------------------------------------------------
    # Identity stamping

    def parameter_stamp(self, original_only: bool = False) -> tuple:
        parameters = self.base.named_parameters() if original_only else self.named_parameters()
        entries = tuple(
            (name, id(parameter), parameter._version, tuple(parameter.shape),
             str(parameter.dtype), str(parameter.device), parameter.data_ptr(),
             parameter.requires_grad)
            for name, parameter in parameters)
        buffers = tuple(
            (f"buffer:{name}", id(buffer), buffer._version, tuple(buffer.shape),
             str(buffer.dtype), str(buffer.device), buffer.data_ptr())
            for name, buffer in self.base.named_buffers())
        return (*entries, *buffers,
                ("base_config", _config_fingerprint(self.base, self.config),
                self.attention_implementation, bool(self.base.config.use_cache)))

    def durable_identity(self) -> str:
        """Content identity for reports/reload, distinct from mutation stamps."""
        return durable_model_identity(
            self.base,
            source_identity=getattr(self.precision, "artifact_sha256", None),
            config={
                "model_type": getattr(self.base.config, "model_type", None),
                "hidden_size": getattr(self.base.config, "hidden_size", None),
                "layers": len(self.base.model.layers),
            })



    def seal_original_identity(self) -> None:
        if self._original_identity is not None:
            raise SparkAdapterError("original identity is already sealed")
        self._original_identity = self.parameter_stamp(original_only=True)

    def base_unchanged(self) -> bool:
        return (self._original_identity is not None
                and self._original_identity == self.parameter_stamp(original_only=True))

    def parameter_audit(self) -> dict:
        def summarize(items) -> dict:
            pairs = list(items)
            return {
                "tensor_count": len(pairs),
                "parameters": sum(parameter.numel() for _, parameter in pairs),
                "trainable_parameters": sum(parameter.numel()
                                            for _, parameter in pairs
                                            if parameter.requires_grad),
                "dtypes": sorted({str(parameter.dtype) for _, parameter in pairs}),
            }

        return {"regime": self.regime,
                "original": summarize(self._original_parameters.items()),
                "new": summarize(self.new_named_parameters()),
                "new_names": [name for name, _ in self.new_named_parameters()]}

    def optimizer_groups(self, base_lr: float, new_lr: float) -> list[dict]:
        self.assert_scope()
        if base_lr < 0 or new_lr <= 0:
            raise SparkAdapterError("base_lr must be nonnegative and new_lr positive")
        parameters = [parameter for _, parameter in self.new_named_parameters()
                      if parameter.requires_grad]
        groups = []
        if parameters:
            groups.append({"params": parameters, "lr": new_lr, "group_name": "new"})
        owned = [id(parameter) for group in groups for parameter in group["params"]]
        expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if len(owned) != len(set(owned)) or set(owned) != expected:
            raise SparkAdapterError("optimizer parameter ownership is incomplete or duplicated")
        return groups

    def gradient_audit(self) -> dict:
        def dense(gradient: torch.Tensor) -> torch.Tensor:
            return gradient.to_dense() if gradient.is_sparse else gradient

        result = {}
        for group, pairs in (("original", self._original_parameters.items()),
                             ("new", self.new_named_parameters())):
            pairs = [(name, parameter) for name, parameter in pairs if parameter.requires_grad]
            present = [name for name, parameter in pairs if parameter.grad is not None]
            finite = [name for name, parameter in pairs
                      if parameter.grad is not None and bool(torch.isfinite(dense(parameter.grad)).all())]
            nonzero = [name for name, parameter in pairs
                       if parameter.grad is not None and bool(torch.count_nonzero(dense(parameter.grad)))]
            result[group] = {
                "eligible_tensors": len(pairs),
                "gradient_tensors": len(present),
                "finite_gradient_tensors": len(finite),
                "nonzero_gradient_tensors": len(nonzero),
                "missing_gradient_names": [name for name, parameter in pairs
                                           if parameter.grad is None],
                "nonfinite_names": [name for name, parameter in pairs
                                    if parameter.grad is not None
                                    and not bool(torch.isfinite(dense(parameter.grad)).all())],
            }
        result["new_all_finite"] = result["new"]["finite_gradient_tensors"] == \
            result["new"]["gradient_tensors"]
        result["new_any_nonzero"] = result["new"]["nonzero_gradient_tensors"] > 0
        return result

    # ------------------------------------------------------------------
    # Forward path

    def _validate_inputs(self, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if (input_ids.ndim != 2 or input_ids.dtype != torch.long
                or not input_ids.shape[1]):
            raise SparkAdapterError("nonempty [batch, tokens] int64 IDs required")
        if bool(((input_ids < 0) | (input_ids >= self.config.geometry.vocab_size)).any()):
            raise SparkAdapterError("token outside the validated Spark vocabulary")
        if attention_mask is None:
            return None
        if attention_mask.shape != input_ids.shape:
            raise SparkAdapterError("attention_mask must match input_ids")
        if attention_mask.dtype not in (torch.long, torch.bool, torch.int32):
            raise SparkAdapterError("attention_mask must be integer or boolean")
        if not bool(attention_mask.any(dim=1).all()):
            raise SparkAdapterError("every row needs at least one known token")
        return attention_mask

    def _intrinsic_delta(self, index, hidden, context):
        device = next(self.extension.vision_projector.parameters()).device
        hidden = hidden.to(device=device, dtype=torch.float32)
        if context.capture_attention:
            result = context.intrinsic_memory.delta(
                index, hidden, return_attention=True)
        else:
            return context.intrinsic_memory.delta(index, hidden)
        delta, audit = result
        position = context.attention_position
        if position is not None:
            selected = {}
            for key, value in audit.items():
                if key in ("query", "weights", "read"):
                    selected[key] = (value[:, :, position:position + 1]
                                     if value.ndim >= 4
                                     else value[:, position:position + 1])
                elif key in ("key", "value") and value.ndim == 4 \
                        and value.shape[1] == hidden.shape[1]:
                    selected[key] = value[:, position]
                else:
                    selected[key] = value
            audit = selected
        context.intrinsic_audit[index] = audit
        return delta

    def _make_hook(self, index: int, layer_type: str, context: _ForwardContext):
        def hook(module: nn.Module, args: tuple, output):
            if not args or not torch.is_tensor(args[0]):
                raise SparkAdapterError("decoder layer did not receive a hidden-state tensor")
            hidden = args[0]
            context.traced.append((index, layer_type))
            device = next(self.extension.vision_projector.parameters()).device
            delta = None
            if context.use_projector and index == self.projector_layer:
                source = context.projector_input
                if self.projector_in_dim == self.config.geometry.hidden_size:
                    source = hidden
                elif source is None:
                    raise SparkAdapterError(
                        "projector_in_dim differs from hidden width; projector_input is required")
                if tuple(source.shape[:2]) != tuple(hidden.shape[:2]) \
                        or source.shape[-1] != self.projector_in_dim:
                    raise SparkAdapterError("projector_input shape differs from the declared width")
                if not source.requires_grad:
                    source = source.detach().requires_grad_(True)
                context.projector_source = source
                delta = self.extension.vision_projector(
                    source.to(device=device, dtype=torch.float32))
            if context.use_ports and index in self._port_layers:
                if context.memory is None or context.memory_mask is None:
                    raise SparkAdapterError("memory and memory_mask are required for enabled ports")
                port = self.extension.memory_ports[str(index)]
                port_delta = port(
                    hidden.to(device=device, dtype=torch.float32),
                    context.memory.to(device=device, dtype=torch.float32),
                    context.memory_mask.to(device=device))
                delta = port_delta if delta is None else delta + port_delta
            if context.spine_context is not None and index in self.spine_spec.sites:
                spine_hidden = hidden.to(device=device, dtype=torch.float32)
                for branch in context.enabled_spine_branches:
                    spine_delta = context.spine_context.delta(
                        branch, index, spine_hidden)
                    delta = spine_delta if delta is None else delta + spine_delta
            if (context.intrinsic_memory is not None
                    and context.intrinsic_placement == "block_input_query_to_post_block_residual"
                    and index in context.intrinsic_memory.spec.sites):
                memory_delta = self._intrinsic_delta(index, hidden, context)
                delta = memory_delta if delta is None else delta + memory_delta
            if delta is None:
                return None
            if tuple(delta.shape) != tuple(hidden.shape):
                raise SparkAdapterError("extension delta shape differs from the decoder hidden state")
            delta = delta.to(output.dtype if torch.is_tensor(output) else hidden.dtype)
            context.injected.append(index)
            context.deltas.append(float(delta.detach().abs().max().item()))
            if isinstance(output, tuple):
                return (output[0] + delta, *output[1:])
            return output + delta
        return hook

    def _make_layer_pre_hook(self, index: int, context: _ForwardContext):
        def hook(module: nn.Module, args: tuple):
            if not args or not torch.is_tensor(args[0]):
                raise SparkAdapterError("decoder layer did not receive a hidden-state tensor")
            if context.intrinsic_memory is not None and index in context.intrinsic_memory.spec.sites:
                device = next(self.extension.vision_projector.parameters()).device
                context.pre_mlp_deltas[index] = self._intrinsic_delta(index, args[0], context)
            return None
        return hook

    def _make_mlp_pre_hook(self, index: int, context: _ForwardContext):
        def hook(module: nn.Module, args: tuple):
            delta = context.pre_mlp_deltas.get(index)
            if delta is None or not args or not torch.is_tensor(args[0]):
                raise SparkAdapterError("pre-MLP memory delta or input is missing")
            hidden = args[0]
            if tuple(delta.shape) != tuple(hidden.shape):
                raise SparkAdapterError("pre-MLP memory delta shape differs")
            delta = delta.to(hidden.dtype)
            if index not in context.pre_mlp_injected:
                context.injected.append(index)
                context.deltas.append(float(delta.detach().abs().max().item()))
                context.pre_mlp_injected.add(index)
            return (hidden + delta, *args[1:])
        return hook

    def _run(self, *, hidden_only: bool, input_ids: torch.Tensor,
             attention_mask: torch.Tensor | None, position_ids: torch.Tensor | None,
             projector_input: torch.Tensor | None, memory: torch.Tensor | None,
             memory_mask: torch.Tensor | None, use_projector: bool, use_ports: bool,
             spine: SpineForwardContext | None = None, use_spine: bool = False,
            intrinsic_memory=None,
            intrinsic_placement="block_input_query_to_post_block_residual",
            capture_attention=False, attention_position=None,
            enabled_spine_branches: tuple[str, ...] = BRANCHES):
        input_ids = input_ids
        mask = self._validate_inputs(input_ids, attention_mask)
        if not use_projector and projector_input is not None:
            raise SparkAdapterError("projector_input was supplied while the projector is disabled")
        if not use_spine and spine is not None:
            raise SparkAdapterError("spine context was supplied while spine is disabled")
        if use_spine and spine is None:
            raise SparkAdapterError("spine context is required when spine is enabled")
        if intrinsic_placement not in (
                "block_input_query_to_post_block_residual", "block_input_query_to_pre_mlp_residual"):
            raise SparkAdapterError("undeclared intrinsic-memory placement")
        context = _ForwardContext(
            use_projector=use_projector, use_ports=use_ports,
            projector_input=projector_input, memory=memory,
            memory_mask=memory_mask,
            spine_context=spine if use_spine else None,
            intrinsic_memory=intrinsic_memory, intrinsic_placement=intrinsic_placement,
            capture_attention=capture_attention, attention_position=attention_position,
            enabled_spine_branches=(tuple(enabled_spine_branches)
                                    if use_spine else ()))
        handles = []
        try:
            for index, layer in enumerate(self.base.model.layers):
                handles.append(layer.register_forward_hook(
                    self._make_hook(index, self.config.geometry.layer_types[index], context)))
            if (intrinsic_placement == "block_input_query_to_pre_mlp_residual"
                    and intrinsic_memory is not None):
                for index, layer in enumerate(self.base.model.layers):
                    if index in intrinsic_memory.spec.sites:
                        handles.append(layer.register_forward_pre_hook(
                            self._make_layer_pre_hook(index, context)))
                        if hasattr(layer, "mlp"):
                            handles.append(layer.mlp.register_forward_pre_hook(
                                self._make_mlp_pre_hook(index, context)))
                        else:
                            for name in ("gate_proj", "up_proj"):
                                handles.append(getattr(layer, name).register_forward_pre_hook(
                                    self._make_mlp_pre_hook(index, context)))
            common = {"input_ids": input_ids, "attention_mask": mask,
                      "position_ids": position_ids, "use_cache": False}
            if hidden_only:
                outputs = self.base.model(**common)
                hidden = (outputs.last_hidden_state if hasattr(outputs, "last_hidden_state")
                          else outputs[0])
                result = hidden
            else:
                outputs = self.base(**common)
                result = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        finally:
            for handle in handles:
                handle.remove()
        traced_indices = [index for index, _ in context.traced]
        expected = list(range(self.config.geometry.num_hidden_layers))
        if traced_indices != expected:
            raise SparkAdapterError(
                "decoder traversal was incomplete or out of order; hooks cannot be skipped")
        if use_projector and self.projector_layer not in context.injected:
            raise SparkAdapterError("the projector layer did not execute")
        sliding = tuple(index for index, kind in context.traced if kind == "sliding_attention")
        full = tuple(index for index, kind in context.traced if kind == "full_attention")
        if self.config.geometry.layer_types and not sliding and not full:
            raise SparkAdapterError("decoder layer schedule was not traced")
        audit = InjectionAudit(
            traced_layers=tuple(traced_indices), sliding_layers=sliding,
            full_layers=full, projector_layer=self.projector_layer,
            port_layers=self._port_layers, injected_layers=tuple(context.injected),
            delta_max_abs=max(context.deltas, default=0.0),
            projector_input_connected=context.projector_source is not None,
            projector_input_requires_grad=bool(
                context.projector_source is not None
                and context.projector_source.requires_grad),
            memory_attention=context.intrinsic_audit or None)
        return result, audit, context

    def forward_text(self, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor | None = None,
                     position_ids: torch.Tensor | None = None, *,
                     projector_input: torch.Tensor | None = None,
                     memory: torch.Tensor | None = None,
                     memory_mask: torch.Tensor | None = None,
                     use_projector: bool = True, use_ports: bool = True,
                     spine: SpineForwardContext | None = None,
                     use_spine: bool = False,
                     intrinsic_memory=None,
                     intrinsic_placement="block_input_query_to_post_block_residual",
                     capture_attention=False, attention_position=None,
                     enabled_spine_branches: tuple[str, ...] = BRANCHES,
                     return_audit: bool = False):
        logits, audit, _ = self._run(
            hidden_only=False, input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, projector_input=projector_input, memory=memory,
            memory_mask=memory_mask, use_projector=use_projector, use_ports=use_ports,
            spine=spine, use_spine=use_spine, intrinsic_memory=intrinsic_memory,
            intrinsic_placement=intrinsic_placement,
            capture_attention=capture_attention, attention_position=attention_position,
            enabled_spine_branches=enabled_spine_branches)
        return (logits, audit) if return_audit else logits

    def forward_hidden(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor | None = None,
                       position_ids: torch.Tensor | None = None, *,
                       projector_input: torch.Tensor | None = None,
                       memory: torch.Tensor | None = None,
                       memory_mask: torch.Tensor | None = None,
                     use_projector: bool = True, use_ports: bool = True,
                     spine: SpineForwardContext | None = None,
                     use_spine: bool = False,
                     intrinsic_memory=None,
                     intrinsic_placement="block_input_query_to_post_block_residual",
                     capture_attention=False, attention_position=None,
                     enabled_spine_branches: tuple[str, ...] = BRANCHES,
                     return_audit: bool = False):
        hidden, audit, _ = self._run(
            hidden_only=True, input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, projector_input=projector_input, memory=memory,
            memory_mask=memory_mask, use_projector=use_projector, use_ports=use_ports,
            spine=spine, use_spine=use_spine, intrinsic_memory=intrinsic_memory,
            intrinsic_placement=intrinsic_placement,
            capture_attention=capture_attention, attention_position=attention_position,
            enabled_spine_branches=enabled_spine_branches)
        return (hidden, audit) if return_audit else hidden

    def reset_cache(self) -> None:
        """The qualification path is uncached; no cache object is retained."""
        self.base.config.use_cache = False

    def snapshot_cache(self) -> None:
        """Explicitly no cache: ``use_cache=False`` is required for training."""
        return None

    def causal_loss(self, logits: torch.Tensor, input_ids: torch.Tensor,
                    labels: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None:
            shifted_logits = logits[:, :-1]
            shifted_labels = input_ids[:, 1:]
        else:
            if labels.shape != logits.shape[:2]:
                raise SparkAdapterError("labels must match sequence positions")
            shifted_logits = logits
            shifted_labels = labels
        if shifted_labels.numel() == 0 or not bool((shifted_labels != -100).any()):
            raise SparkAdapterError("no supervised target positions")
        return F.cross_entropy(shifted_logits.float().reshape(-1, shifted_logits.shape[-1]),
                               shifted_labels.reshape(-1).to(shifted_logits.device),
                               ignore_index=-100)

    def backward_probe(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor | None = None, *,
                       projector_input: torch.Tensor | None = None,
                       memory: torch.Tensor | None = None,
                       memory_mask: torch.Tensor | None = None,
                       labels: torch.Tensor | None = None,
                       use_projector: bool = True,
                       spine: SpineForwardContext | None = None,
                       use_spine: bool = False,
                       enabled_spine_branches: tuple[str, ...] = BRANCHES) -> dict:
        """One loss backward plus the gradient that reaches the projector input.

        The decoder is never wrapped in ``no_grad``. The projector input is a
        leaf so its own gradient can be read even though the base is frozen and
        its upstream embedding output does not require grad.
        """
        if not torch.is_grad_enabled():
            raise SparkAdapterError("backward probe requires autograd to be enabled")
        self.zero_grad(set_to_none=True)
        logits, audit, context = self._run(
            hidden_only=False, input_ids=input_ids, attention_mask=attention_mask,
            position_ids=None, projector_input=projector_input, memory=memory,
            memory_mask=memory_mask, use_projector=use_projector, use_ports=True,
            spine=spine, use_spine=use_spine,
            enabled_spine_branches=enabled_spine_branches)
        loss = self.causal_loss(logits, input_ids, labels)
        if not bool(torch.isfinite(loss)):
            raise SparkAdapterError("loss is not finite")
        source = context.projector_source
        if source is None:
            raise SparkAdapterError("projector input was not connected to the decoder")
        gradient = torch.autograd.grad(loss, source, retain_graph=True,
                                       allow_unused=False)[0]
        input_gradient = {
            "present": gradient is not None,
            "norm": float(gradient.detach().float().norm()),
            "nonzero": bool(torch.count_nonzero(gradient)),
        }
        loss.backward()
        audit_result = self.gradient_audit()
        return {
            "loss": float(loss.detach()),
            "loss_finite": True,
            "grad_enabled": torch.is_grad_enabled(),
            "input_gradient": input_gradient,
            "gradient_audit": audit_result,
            "injection": audit.to_dict(),
            "new_gradient_nonzero": audit_result["new_any_nonzero"],
            "new_gradient_finite": audit_result["new_all_finite"],
        }

    # ------------------------------------------------------------------
    # Extension export / reload

    def adapter_settings(self) -> dict:
        return {
            "regime": self.regime,
            "projector_in_dim": self.projector_in_dim,
            "projector_hidden_dim": self.projector_hidden_dim,
            "projector_out_dim": self.config.geometry.hidden_size,
            "projector_layer": self.projector_layer,
            "zero_init_output": self.zero_init_output,
            "memory_port_layers": list(self._port_layers),
            "memory_size": self.memory_size,
            "port_heads": self.heads,
            "spine": self.spine_spec.to_dict(),
            "spine_sites": list(self.spine_spec.sites),
            "spine_slots": self.spine_spec.slots,
            "attention_implementation": self.attention_implementation,
            "use_cache": False,
        }

    def base_snapshot(self) -> dict:
        return {
            "precision_kind": self.precision.kind,
            "storage": self.precision.storage,
            "weight_dtype": self.precision.weight_dtype,
            "artifact_sha256": self.precision.artifact_sha256,
            "quantization_fingerprint": self.precision.quantization_fingerprint,
            "derivative_of": self.precision.derivative_of,
            "precision_digest": self.precision.digest,
            "config_digest": self.config.digest,
            "geometry_fingerprint": fingerprint(self.config.geometry.to_dict()),
        }

    def extension_metadata(self) -> dict:
        extensions = build_extensions(
            self.config, projector_in_dim=self.projector_in_dim,
            projector_hidden_dim=self.projector_hidden_dim,
            memory_ports=list(self._port_layers))
        payload = build_extension_export(
            self.config, self.precision, extensions, trainable_dtype="float32",
            spine=self.extension.spine.inventory())
        payload["adapter_tensors"] = [tensor.to_dict() for tensor in self.new_named_tensors()]
        payload["adapter_settings"] = self.adapter_settings()
        payload["base_snapshot"] = self.base_snapshot()
        payload["content_model_identity"] = self.durable_identity()
        payload["fingerprint"] = fingerprint(
            {key: value for key, value in payload.items() if key != "fingerprint"})
        return payload

    def save_extension(self, directory: str | Path) -> dict:
        from safetensors.torch import save_file

        if self._original_identity is None:
            raise SparkAdapterError("seal the unchanged original identity before export")
        if not self.base_unchanged():
            raise SparkAdapterError("original checkpoint identity changed before export")
        self.assert_scope()
        payload = self.extension_metadata()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        tensors = {name: parameter.detach().cpu().contiguous()
                   for name, parameter in self.new_named_parameters()}
        expected = [entry["name"] for entry in payload["adapter_tensors"]]
        if list(tensors) != expected:
            raise SparkAdapterError("extension tensor order or inventory changed during export")
        save_file(tensors, str(path / "extensions.safetensors"))
        weights_sha256 = sha256_bytes((path / "extensions.safetensors").read_bytes())
        manifest = {
            "schema": ADAPTER_EXTENSION_SCHEMA,
            "export": payload,
            "weights_file": "extensions.safetensors",
            "weights_sha256": weights_sha256,
            "contains_base_weights": False,
            "base_saved": False,
            "optimizer_saved": False,
            "fresh_process_reload_required": True,
            "notes": ("Extension-only weights plus the sealed base identity; no checkpoint "
                      "bytes are copied or rewritten. Compare logits/tokens in a fresh process."),
        }
        manifest["fingerprint"] = fingerprint(manifest)
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest

    def load_extension(self, directory: str | Path) -> dict:
        from safetensors.torch import load_file

        if self._original_identity is None:
            raise SparkAdapterError("seal the unchanged original identity before reload")
        if not self.base_unchanged():
            raise SparkAdapterError("original checkpoint identity changed before reload")
        self.assert_scope()
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("schema") != ADAPTER_EXTENSION_SCHEMA:
            raise SparkAdapterError("unexpected extension manifest schema")
        body = {key: value for key, value in manifest.items() if key != "fingerprint"}
        if manifest.get("fingerprint") != fingerprint(body):
            raise SparkAdapterError("extension manifest fingerprint mismatch")
        if manifest.get("contains_base_weights") is not False:
            raise SparkAdapterError("extension manifest records copied base weights")
        if manifest.get("weights_file") != "extensions.safetensors":
            raise SparkAdapterError("extension manifest names an unexpected weights file")
        payload = manifest.get("export")
        receipt = validate_extension_export(payload, self.config, self.precision)
        expected_payload = self.extension_metadata()
        for key in ("adapter_tensors", "adapter_settings", "base_snapshot",
                    "content_model_identity"):
            if payload.get(key) != expected_payload.get(key):
                raise SparkAdapterError(f"extension {key} differs from the current adapter")
        weights_sha256 = sha256_bytes((path / manifest["weights_file"]).read_bytes())
        if weights_sha256 != manifest.get("weights_sha256"):
            raise SparkAdapterError("extension weight bytes changed")
        weights = load_file(str(path / manifest["weights_file"]))
        expected = {name: parameter for name, parameter in self.new_named_parameters()}
        if set(weights) != set(expected):
            raise SparkAdapterError("extension tensor names differ from the current adapter")
        for name, parameter in expected.items():
            tensor = weights[name]
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise SparkAdapterError(f"{name} shape differs")
            if tensor.dtype != parameter.dtype:
                raise SparkAdapterError(f"{name} dtype differs")
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(weights[name].to(parameter.device))
        return {
            **receipt,
            "loaded_tensors": len(expected),
            "loaded_parameter_elements": sum(parameter.numel()
                                             for parameter in expected.values()),
            "weights_sha256": weights_sha256,
            "extension_only": True,
            "fresh_process_comparison_required": True,
        }

    def describe(self) -> dict:
        return {
            "adapter": "Spark2_5Adapter",
            "regime": self.regime,
            "precision": self.precision.to_dict(),
            "admission": self.admission.to_dict(),
            "attention_implementation": self.attention_implementation,
            "use_cache": False,
            "official_geometry": self.config.geometry.is_official_geometry,
            "geometry": self.config.geometry.to_dict(),
            "parameters": self.parameter_audit(),
            "settings": self.adapter_settings(),
            "ports": list(self._port_layers),
            "spine": self.extension.spine.inventory(),
            "content_model_identity": self.durable_identity(),
            "sealed_original": self._original_identity is not None,
            "base_unchanged": self.base_unchanged(),
        }


__all__ = [
    "ADAPTER_EXTENSION_SCHEMA", "REGIMES", "SparkAdapterError",
    "InjectionAudit", "ExtensionFile", "Spark2_5Adapter",
]
