"""Dedicated Spark2_5 preparation contract; no weights, kernels or native code.

This module inspects the *declared* geometry, layer schedule, tied embedding
inventory, tokenizer/template identity, precision identity, backend admission
and extension dimensions for the pinned custom ``spark2_5`` architecture. It
does not import torch/transformers, execute upstream code, download weights or
run a Spark model.

Three precision identities are deliberately distinct:

* ``official_bf16`` -- the independent BF16 release, plain BF16 storage;
* ``official_native_fp8`` -- compressed-tensors F8_E4M3 storage with dynamic
  per-token activation quantization declared in the pinned config;
* ``dequantized_fp8_derivative`` -- a *new* artifact reconstructed from the
  stored FP8 values and scales. It is not the official BF16 release, does not
  restore lost precision and is never loaded under the official FP8 identity.

Admission is explicit and conservative. On this host MPS cannot allocate
``float8_e4m3fn`` (see ``reports/spark_qualification_20261025/backend_audit.json``),
so native FP8 on MPS is refused. Admission never changes the requested device,
never dequantizes implicitly and never falls back to CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "replica.spark2_5_preparation.v1"
EXTENSION_EXPORT_SCHEMA = "replica.spark2_5_extension_export.v1"
SPINE_EXTENSION_EXPORT_SCHEMA = "replica.spark2_5_extension_export.v2"
SPINE_SCHEMA = "replica.spark.spine.v1"

OFFICIAL_BF16 = "official_bf16"
OFFICIAL_NATIVE_FP8 = "official_native_fp8"
DEQUANTIZED_FP8_DERIVATIVE = "dequantized_fp8_derivative"
PRECISION_KINDS = (OFFICIAL_BF16, OFFICIAL_NATIVE_FP8, DEQUANTIZED_FP8_DERIVATIVE)

DERIVATIVE_MANIFEST_SCHEMA = "replica.spark2_5_dequantized_derivative.v1"
DERIVATIVE_EXECUTION_FORMAT = "dequantized_fp8_derivative_bf16_storage"
DERIVATIVE_ACTIVATION_SEMANTICS = (
    "BF16/FP32 activations with no per-token F8 quantization; this is not the "
    "official FP8 forward semantics")
DERIVATIVE_CLAIMS_FALSE = (
    "activation_quantization", "native_execution", "official_bf16_release",
    "official_native_fp8", "restores_quantization_loss")
P1_SPINE_SITES = (3, 7)
LATE_SPINE_SITES = (23, 31)
P1_SPINE_SLOTS = 8
P1_MEMORY_SIZE = 256
P1_PORT_HEADS = 4
CURRENT_SOURCE_LAYERS = 36

OFFICIAL_MODEL_TYPE = "spark2_5"
OFFICIAL_ARCHITECTURE = "Spark2_5ForCausalLM"
OFFICIAL_AUTO_MAP = {
    "AutoConfig": "configuration_spark.Spark2_5Config",
    "AutoModel": "modeling_spark.Spark2_5Model",
    "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
}

OFFICIAL_LAYER_TYPES = tuple(
    ["sliding_attention"] * 3 + ["full_attention"] +
    (["sliding_attention"] * 3 + ["full_attention"]) * 8
)
assert len(OFFICIAL_LAYER_TYPES) == 36

OFFICIAL_GEOMETRY = {
    "num_hidden_layers": 36,
    "hidden_size": 2560,
    "intermediate_size": 10240,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 131072,
    "sliding_window": 512,
    "max_position_embeddings": 1048576,
    "tie_word_embeddings": True,
    "headwise_attn_output_gate": True,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "gelu",
    "gate_attn_act_mode": "sigmoid",
    "layer_types": OFFICIAL_LAYER_TYPES,
}
OFFICIAL_PARAMETER_ELEMENTS = 4_112_079_360
MINICPM5_REUSED_DIMENSIONS = {
    "hidden_size": 2048,
    "num_hidden_layers": 42,
    "vocab_size": 130560,
    "tie_word_embeddings": False,
    "projector_out_dim": 2048,
    "memory_width": 2048,
    "accepted_ports": (13, 27),
}

THINKING_MODES = {
    "none": {"enable_thinking": False},
    "low": {"enable_thinking": True, "reasoning_effort": "low"},
    "high": {"enable_thinking": True, "reasoning_effort": "high"},
}
DEFAULT_THINKING_MODE = "high"
REASONING_DELIMITERS = ("<think>", "</think>")

LAYER_TYPES = frozenset({"sliding_attention", "full_attention"})
QUANTIZED_SUFFIXES = (
    "self_attn.q_k_v_proj",
    "self_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
UNQUANTIZED_SUFFIXES = ("self_attn.g_proj",)

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Spark2_5ContractError(ValueError):
    """Base class for a refused Spark2_5 preparation fact."""


class GeometryError(Spark2_5ContractError):
    """The declared config is not a supported Spark2_5 geometry."""


class PrecisionError(Spark2_5ContractError):
    """Storage and declared compute precision are inconsistent or unlabelled."""


class BackendAdmissionError(Spark2_5ContractError):
    """A load was refused for the requested device/precision pair."""

    def __init__(self, reason: str, *, device: str, detail: str | None = None,
                 identity: "PrecisionIdentity | None" = None):
        self.reason = reason
        self.device = device
        self.detail = detail
        self.precision = None if identity is None else identity.kind
        super().__init__(f"{reason} on {device}: {detail}" if detail else f"{reason} on {device}")

    def evidence(self) -> dict:
        return {"reason": self.reason, "requested_device": self.device,
                "precision_kind": self.precision, "detail": self.detail,
                "fallback_used": False, "dequantization_performed": False}


class TokenizerIdentityError(Spark2_5ContractError):
    """Tokenizer or chat-template identity is incomplete or inconsistent."""


class ThinkingModeError(Spark2_5ContractError):
    """A generation request names an unsupported thinking mode."""


class TiedEmbeddingError(Spark2_5ContractError):
    """The tied input/output embedding inventory is violated."""


class InventoryError(Spark2_5ContractError):
    """The tensor inventory does not match the declared Spark2_5 contract."""


class ExtensionStateError(Spark2_5ContractError):
    """A dimension-specific extension export/reload is invalid."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derivative_artifact_digest(files: Sequence[Mapping]) -> str:
    """Deterministic identity over a derivative's declared filename/hash pairs."""
    entries = sorted(({"filename": str(item["filename"]),
                       "sha256": str(item["sha256"])} for item in files),
                     key=lambda item: item["filename"])
    if not entries:
        raise PrecisionError("a derivative artifact must contain at least one shard")
    for item in entries:
        _require_sha(item["sha256"], "shard sha256", PrecisionError)
    return sha256_bytes(canonical_json(entries).encode())


def _require_sha(value: Any, name: str, Error=Spark2_5ContractError) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise Error(f"{name} must be a lowercase SHA256")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 1,
                 Error=GeometryError) -> int:
    if type(value) is not int or value < minimum:
        raise Error(f"{name} must be an integer >= {minimum}")
    return value


def _require_bool(value: Any, name: str, Error=GeometryError) -> bool:
    if type(value) is not bool:
        raise Error(f"{name} must be a boolean")
    return value


def _mapping(value: Any, name: str, Error=GeometryError) -> Mapping:
    if not isinstance(value, Mapping):
        raise Error(f"{name} must be an object")
    return value


def _text(value: Any, name: str, Error=Spark2_5ContractError) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Error(f"{name} must be a nonempty string")
    return value


# ---------------------------------------------------------------------------
# Geometry and config


@dataclass(frozen=True)
class Spark2_5Geometry:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    headwise_attn_output_gate: bool
    attention_bias: bool
    mlp_bias: bool
    hidden_act: str
    gate_attn_act_mode: str
    layer_types: tuple[str, ...]

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def qkv_dim(self) -> int:
        return self.q_dim + 2 * self.kv_dim

    @property
    def kv_repeat(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def parameter_elements(self) -> int:
        """Unique elements with the tied input/output table counted once."""
        total = self.vocab_size * self.hidden_size
        per_layer = 2 * self.hidden_size
        per_layer += self.qkv_dim * self.hidden_size
        per_layer += self.hidden_size * self.q_dim
        per_layer += self.hidden_size * self.intermediate_size * 3
        if self.headwise_attn_output_gate:
            per_layer += self.num_attention_heads * self.hidden_size
        total += self.num_hidden_layers * per_layer
        total += self.hidden_size
        return total

    @property
    def is_official_geometry(self) -> bool:
        return all(getattr(self, key) == value for key, value in OFFICIAL_GEOMETRY.items())

    def assert_official(self) -> None:
        if not self.is_official_geometry:
            raise GeometryError("config is structurally valid but not the pinned 36x2560 Spark2_5 geometry")

    def to_dict(self) -> dict:
        data = {
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "sliding_window": self.sliding_window,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tie_word_embeddings,
            "headwise_attn_output_gate": self.headwise_attn_output_gate,
            "attention_bias": self.attention_bias,
            "mlp_bias": self.mlp_bias,
            "hidden_act": self.hidden_act,
            "gate_attn_act_mode": self.gate_attn_act_mode,
            "layer_types": list(self.layer_types),
            "q_dim": self.q_dim,
            "kv_dim": self.kv_dim,
            "qkv_dim": self.qkv_dim,
            "kv_repeat": self.kv_repeat,
            "parameter_elements": self.parameter_elements,
            "official_geometry": self.is_official_geometry,
        }
        return data


@dataclass(frozen=True)
class Spark2_5Config:
    geometry: Spark2_5Geometry
    rope_parameters: dict
    raw: dict
    digest: str

    def to_dict(self) -> dict:
        return {"geometry": self.geometry.to_dict(),
                "rope_parameters": self.rope_parameters,
                "digest": self.digest}


def _reject_minicpm5_dimensions(geometry: Spark2_5Geometry) -> None:
    if geometry.hidden_size == MINICPM5_REUSED_DIMENSIONS["hidden_size"]:
        raise GeometryError("hidden_size 2048 belongs to MiniCPM5; Spark2_5 uses width 2560")
    if geometry.num_hidden_layers == MINICPM5_REUSED_DIMENSIONS["num_hidden_layers"]:
        raise GeometryError("42 layers belong to MiniCPM5; Spark2_5 uses 36")
    if geometry.vocab_size == MINICPM5_REUSED_DIMENSIONS["vocab_size"]:
        raise GeometryError("vocab 130560 belongs to MiniCPM5; Spark2_5 uses 131072")
    if not geometry.tie_word_embeddings:
        raise GeometryError("Spark2_5 requires tied input/output embeddings")
    if geometry.head_dim % 2:
        raise GeometryError("head_dim must be even for rotary attention")


def load_spark2_5_config(raw: Mapping, *, require_official_geometry: bool = False) -> Spark2_5Config:
    """Validate one pinned Spark2_5 config without loading any model code."""
    config = _mapping(raw, "config")
    if config.get("model_type") != OFFICIAL_MODEL_TYPE:
        raise GeometryError(f"model_type must be {OFFICIAL_MODEL_TYPE!r}")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or OFFICIAL_ARCHITECTURE not in architectures:
        raise GeometryError(f"architectures must contain {OFFICIAL_ARCHITECTURE!r}")
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, Mapping):
        raise GeometryError("auto_map is required for the custom architecture")
    for key, wanted in OFFICIAL_AUTO_MAP.items():
        if auto_map.get(key) != wanted:
            raise GeometryError(f"auto_map.{key} must be the pinned Spark2_5 implementation")

    layers = _require_int(config.get("num_hidden_layers"), "num_hidden_layers")
    hidden = _require_int(config.get("hidden_size"), "hidden_size")
    intermediate = _require_int(config.get("intermediate_size"), "intermediate_size")
    heads = _require_int(config.get("num_attention_heads"), "num_attention_heads")
    kv_heads = _require_int(config.get("num_key_value_heads"), "num_key_value_heads")
    if heads % kv_heads:
        raise GeometryError("num_attention_heads must be divisible by num_key_value_heads")
    if config.get("head_dim") is None:
        raise GeometryError("head_dim must be explicit; never assume hidden_size // heads")
    head_dim = _require_int(config.get("head_dim"), "head_dim")
    vocab = _require_int(config.get("vocab_size"), "vocab_size")
    window = _require_int(config.get("sliding_window"), "sliding_window")
    max_positions = _require_int(config.get("max_position_embeddings"), "max_position_embeddings")

    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != layers:
        raise GeometryError("layer_types must list every layer exactly once")
    if any(not isinstance(item, str) or item not in LAYER_TYPES for item in layer_types):
        raise GeometryError(f"layer_types entries must be one of {sorted(LAYER_TYPES)}")
    if "full_attention" not in layer_types or "sliding_attention" not in layer_types:
        raise GeometryError("Spark2_5 declares both sliding_attention and full_attention layers")

    tie = _require_bool(config.get("tie_word_embeddings"), "tie_word_embeddings")
    headwise_gate = _require_bool(config.get("headwise_attn_output_gate"),
                                  "headwise_attn_output_gate")
    if not headwise_gate:
        raise GeometryError("Spark2_5 requires the headwise attention output gate")
    if config.get("gate_attn_act_mode") != "sigmoid":
        raise GeometryError("gate_attn_act_mode must be 'sigmoid'")
    if config.get("hidden_act") != "gelu":
        raise GeometryError("hidden_act must be 'gelu'")
    if config.get("attention_bias") is not False or config.get("mlp_bias") is not False:
        raise GeometryError("Spark2_5 has no attention or MLP bias tensors")
    if config.get("dtype") not in ("bfloat16", "float32"):
        raise GeometryError("config dtype must declare bfloat16 or float32 storage")

    rope = _mapping(config.get("rope_parameters"), "rope_parameters")
    for layer_type in sorted(set(layer_types)):
        entry = _mapping(rope.get(layer_type), f"rope_parameters.{layer_type}")
        theta = entry.get("rope_theta")
        factor = entry.get("partial_rotary_factor")
        if type(theta) not in (int, float) or theta <= 0:
            raise GeometryError(f"rope_parameters.{layer_type}.rope_theta must be positive")
        if type(factor) not in (int, float) or not 0 < factor <= 1:
            raise GeometryError(f"rope_parameters.{layer_type}.partial_rotary_factor must be in (0, 1]")

    geometry = Spark2_5Geometry(
        num_hidden_layers=layers, hidden_size=hidden, intermediate_size=intermediate,
        num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim,
        vocab_size=vocab, sliding_window=window, max_position_embeddings=max_positions,
        tie_word_embeddings=tie, headwise_attn_output_gate=headwise_gate,
        attention_bias=False, mlp_bias=False, hidden_act="gelu",
        gate_attn_act_mode="sigmoid", layer_types=tuple(layer_types))
    _reject_minicpm5_dimensions(geometry)
    if require_official_geometry:
        geometry.assert_official()
    return Spark2_5Config(geometry=geometry, rope_parameters=dict(rope), raw=dict(config),
                          digest=fingerprint(config))


# ---------------------------------------------------------------------------
# Layer schedule


@dataclass(frozen=True)
class LayerSchedule:
    layer_types: tuple[str, ...]
    sliding_indices: tuple[int, ...]
    full_indices: tuple[int, ...]
    runs: tuple[tuple[str, int], ...]
    period_types: tuple[str, ...]
    period_runs: tuple[tuple[str, int], ...]
    period_repeats: int

    def to_dict(self) -> dict:
        return {
            "layer_types": list(self.layer_types),
            "sliding_indices": list(self.sliding_indices),
            "full_indices": list(self.full_indices),
            "runs": [[kind, count] for kind, count in self.runs],
            "period_types": list(self.period_types),
            "period_runs": [[kind, count] for kind, count in self.period_runs],
            "period_repeats": self.period_repeats,
            "source": "config.layer_types",
        }


def _runs(types: Sequence[str]) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for layer_type in types:
        if result and result[-1][0] == layer_type:
            result[-1] = (layer_type, result[-1][1] + 1)
        else:
            result.append((layer_type, 1))
    return tuple(result)


def derive_layer_schedule(config: Spark2_5Config) -> LayerSchedule:
    """Read the sliding/full schedule from config; never assume a Llama pattern."""
    types = config.geometry.layer_types
    period_types: tuple[str, ...] = types
    repeats = 1
    for period in range(1, len(types) // 2 + 1):
        if len(types) % period == 0 and types == types[:period] * (len(types) // period):
            period_types = types[:period]
            repeats = len(types) // period
            break
    return LayerSchedule(
        layer_types=types,
        sliding_indices=tuple(i for i, kind in enumerate(types) if kind == "sliding_attention"),
        full_indices=tuple(i for i, kind in enumerate(types) if kind == "full_attention"),
        runs=_runs(types),
        period_types=period_types,
        period_runs=_runs(period_types),
        period_repeats=repeats,
    )


# ---------------------------------------------------------------------------
# Precision identities


@dataclass(frozen=True)
class PrecisionIdentity:
    kind: str
    storage: str
    weight_dtype: str
    activation: str
    quantization_method: str | None
    quantization_version: str | None
    quantization_fingerprint: str | None
    artifact_sha256: str | None
    dequantized: bool
    activation_quantized: bool
    quantized_modules: tuple[str, ...]
    unquantized_modules: tuple[str, ...]
    derivative_of: str | None = None
    dequantization_sha256: str | None = None
    equivalent_to_bf16_release: bool = False
    note: str = ""

    def __post_init__(self):
        if self.kind not in PRECISION_KINDS:
            raise PrecisionError(f"unknown precision kind: {self.kind!r}")
        if self.kind == DEQUANTIZED_FP8_DERIVATIVE:
            if not self.dequantized or self.derivative_of is None:
                raise PrecisionError("a dequantized derivative needs an explicit parent identity")
            if self.activation_quantized:
                raise PrecisionError("a dequantized derivative has no F8 activation quantization")
            if self.equivalent_to_bf16_release:
                raise PrecisionError("dequantization never restores the official BF16 release identity")
        else:
            if self.dequantized:
                raise PrecisionError("only a derivative may set dequantized=True")
            if self.kind == OFFICIAL_BF16 and "fp8" in self.storage:
                raise PrecisionError("official_bf16 cannot carry FP8 storage")
        if self.kind == OFFICIAL_NATIVE_FP8:
            if self.storage != "fp8_compressed_tensors" or self.weight_dtype != "F8_E4M3":
                raise PrecisionError("official native FP8 requires F8_E4M3 compressed-tensors storage")
            if not self.activation_quantized:
                raise PrecisionError("official FP8 declares dynamic per-token activation quantization")
        if self.artifact_sha256 is not None:
            _require_sha(self.artifact_sha256, "artifact_sha256", PrecisionError)
        if self.dequantization_sha256 is not None:
            _require_sha(self.dequantization_sha256, "dequantization_sha256", PrecisionError)

    @property
    def digest(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "storage": self.storage,
            "weight_dtype": self.weight_dtype,
            "activation": self.activation,
            "quantization_method": self.quantization_method,
            "quantization_version": self.quantization_version,
            "quantization_fingerprint": self.quantization_fingerprint,
            "artifact_sha256": self.artifact_sha256,
            "dequantized": self.dequantized,
            "activation_quantized": self.activation_quantized,
            "quantized_modules": list(self.quantized_modules),
            "unquantized_modules": list(self.unquantized_modules),
            "derivative_of": self.derivative_of,
            "dequantization_sha256": self.dequantization_sha256,
            "equivalent_to_bf16_release": self.equivalent_to_bf16_release,
            "note": self.note,
        }


def official_bf16_precision(config: Spark2_5Config, *, artifact_sha256: str | None = None) -> PrecisionIdentity:
    if "quantization_config" in config.raw:
        raise PrecisionError("a BF16 identity cannot silently strip a quantization_config")
    if "replica_precision" in config.raw:
        raise PrecisionError(
            "a dequantized FP8 derivative config cannot be labelled official_bf16; "
            "use its derivative manifest instead")
    if config.raw.get("dtype") != "bfloat16":
        raise PrecisionError("official BF16 storage must declare dtype=bfloat16")
    return PrecisionIdentity(
        kind=OFFICIAL_BF16, storage="bf16", weight_dtype="BF16",
        activation="bfloat16; unquantized by checkpoint", quantization_method=None,
        quantization_version=None, quantization_fingerprint=None,
        artifact_sha256=artifact_sha256, dequantized=False, activation_quantized=False,
        quantized_modules=(), unquantized_modules=(),
        note="independent official BF16 release; not a dequantized FP8 derivative",
    )


def _quantization_group(config: Spark2_5Config) -> Mapping:
    quant = _mapping(config.raw.get("quantization_config"), "quantization_config", PrecisionError)
    if quant.get("quant_method") != "compressed-tensors":
        raise PrecisionError("quantization_config.quant_method must be compressed-tensors")
    if quant.get("format") != "float-quantized":
        raise PrecisionError("quantization_config.format must be float-quantized")
    if quant.get("version") != "0.16.0":
        raise PrecisionError("quantization_config.version must be the pinned 0.16.0")
    if quant.get("kv_cache_scheme") is not None:
        raise PrecisionError("the pinned Spark2_5 checkpoint does not quantize the KV cache")
    groups = _mapping(quant.get("config_groups"), "quantization_config.config_groups", PrecisionError)
    if set(groups) != {"group_0"}:
        raise PrecisionError("exactly one compressed-tensors group is supported")
    group = _mapping(groups["group_0"], "config_groups.group_0", PrecisionError)
    weights = _mapping(group.get("weights"), "config_groups.group_0.weights", PrecisionError)
    if (weights.get("strategy"), weights.get("num_bits"), weights.get("dynamic"),
            weights.get("symmetric")) != ("channel", 8, False, True):
        raise PrecisionError("FP8 weights must be static symmetric channel-wise 8-bit")
    activations = _mapping(group.get("input_activations"),
                           "config_groups.group_0.input_activations", PrecisionError)
    if (activations.get("strategy"), activations.get("num_bits"), activations.get("dynamic"),
            activations.get("symmetric")) != ("token", 8, True, True):
        raise PrecisionError("FP8 activations must be dynamic symmetric per-token 8-bit")
    expected_targets = {
        "re:.*self_attn.q_k_v_proj$",
        "re:.*self_attn.out_proj$",
        "re:.*mlp.(gate_proj|up_proj|gate_up_proj|down_proj)$",
    }
    targets = group.get("targets")
    if not isinstance(targets, list) or set(targets) != expected_targets:
        raise PrecisionError("quantization targets differ from the pinned Spark2_5 modules")
    ignore = quant.get("ignore")
    if not isinstance(ignore, list):
        raise PrecisionError("quantization_config.ignore must be a list")
    required_ignored = {"lm_head"} | {
        f"model.layers.{index}.self_attn.g_proj"
        for index in range(config.geometry.num_hidden_layers)}
    if not required_ignored.issubset(ignore):
        raise PrecisionError("attention gates and the tied lm_head must stay unquantized")
    return quant


def official_native_fp8_precision(config: Spark2_5Config, *,
                                  artifact_sha256: str | None = None) -> PrecisionIdentity:
    quant = _quantization_group(config)
    return PrecisionIdentity(
        kind=OFFICIAL_NATIVE_FP8, storage="fp8_compressed_tensors", weight_dtype="F8_E4M3",
        activation="dynamic symmetric per-token F8_E4M3 (observer-free at runtime)",
        quantization_method="compressed-tensors", quantization_version="0.16.0",
        quantization_fingerprint=fingerprint(quant), artifact_sha256=artifact_sha256,
        dequantized=False, activation_quantized=True,
        quantized_modules=QUANTIZED_SUFFIXES, unquantized_modules=UNQUANTIZED_SUFFIXES,
        note="official FP8 storage; activation quantization is part of forward semantics",
    )


def dequantized_fp8_derivative(
    config: Spark2_5Config, *, parent: PrecisionIdentity, artifact_sha256: str,
    dequantization_sha256: str, activation_semantics: str, storage: str = "bf16",
) -> PrecisionIdentity:
    """Label a *new* artifact rebuilt from FP8 values/scales; never the parent."""
    if not isinstance(parent, PrecisionIdentity) or parent.kind != OFFICIAL_NATIVE_FP8:
        raise PrecisionError("a dequantized derivative must name an official native FP8 parent")
    _require_sha(artifact_sha256, "artifact_sha256", PrecisionError)
    _require_sha(dequantization_sha256, "dequantization_sha256", PrecisionError)
    if parent.artifact_sha256 is not None and artifact_sha256 == parent.artifact_sha256:
        raise PrecisionError("the derivative cannot reuse the parent artifact digest")
    if "quantization_config" in config.raw:
        raise PrecisionError("a dequantized derivative must not keep float-quantized storage")
    if storage not in ("bf16", "fp32"):
        raise PrecisionError("dequantized derivative storage must be explicitly bf16 or fp32")
    expected_dtype = "bfloat16" if storage == "bf16" else "float32"
    if config.raw.get("dtype") != expected_dtype:
        raise PrecisionError(f"{storage} derivative requires config dtype={expected_dtype}")
    _text(activation_semantics, "activation_semantics", PrecisionError)
    declared = config.raw.get("replica_precision")
    if declared is not None:
        declared = _mapping(declared, "replica_precision", PrecisionError)
        expected = {
            "kind": DEQUANTIZED_FP8_DERIVATIVE,
            "parent_kind": OFFICIAL_NATIVE_FP8,
            "parent_artifact_sha256": parent.artifact_sha256,
            "dequantization_sha256": dequantization_sha256,
            "activation_semantics": activation_semantics,
            "equivalent_to_bf16_release": False,
        }
        for key, value in expected.items():
            if declared.get(key) != value:
                raise PrecisionError(
                    f"config.replica_precision.{key} differs from the derivative manifest")
    return PrecisionIdentity(
        kind=DEQUANTIZED_FP8_DERIVATIVE, storage=storage,
        weight_dtype="BF16" if storage == "bf16" else "F32",
        activation=activation_semantics, quantization_method=None, quantization_version=None,
        quantization_fingerprint=None, artifact_sha256=artifact_sha256,
        dequantized=True, activation_quantized=False, quantized_modules=(),
        unquantized_modules=(), derivative_of=parent.artifact_sha256 or parent.digest,
        dequantization_sha256=dequantization_sha256,
        note=("reconstructed from the official FP8 values and scales; it does not restore "
              "quantization loss, change activation semantics, or equal the official BF16 release"),
    )


def classify_precision(config: Spark2_5Config, *, artifact_sha256: str | None = None,
                       derivative_manifest: Mapping | None = None) -> PrecisionIdentity:
    declared = config.raw.get("replica_precision")
    if derivative_manifest is not None:
        manifest = _mapping(derivative_manifest, "derivative_manifest", PrecisionError)
        missing = {"parent", "artifact_sha256", "dequantization_sha256", "activation_semantics"} - set(manifest)
        if missing:
            raise PrecisionError(f"derivative manifest is missing {sorted(missing)}")
        parent = manifest["parent"]
        if not isinstance(parent, PrecisionIdentity):
            raise PrecisionError(
                "derivative_manifest.parent must be a PrecisionIdentity built from the "
                "official FP8 config, not an unlabelled mapping")
        return dequantized_fp8_derivative(
            config, parent=parent, artifact_sha256=manifest["artifact_sha256"],
            dequantization_sha256=manifest["dequantization_sha256"],
            activation_semantics=manifest["activation_semantics"],
            storage=manifest.get("storage", "bf16"))
    if declared is not None:
        raise PrecisionError(
            "a dequantized FP8 derivative config requires its explicit derivative "
            "manifest; official BF16 and official FP8 substitution are refused")
    if "quantization_config" in config.raw:
        return official_native_fp8_precision(config, artifact_sha256=artifact_sha256)
    return official_bf16_precision(config, artifact_sha256=artifact_sha256)


# ---------------------------------------------------------------------------
# Backend admission


@dataclass(frozen=True)
class BackendCapabilities:
    device: str
    bf16: bool
    fp8_storage: bool
    fp8_compute: bool
    fp8_activation_quantization: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {"device": self.device, "bf16": self.bf16,
                "fp8_storage": self.fp8_storage, "fp8_compute": self.fp8_compute,
                "fp8_activation_quantization": self.fp8_activation_quantization,
                "note": self.note}


DEFAULT_BACKENDS = {
    "mps": BackendCapabilities(
        "mps", True, False, False, False,
        "MPS cannot allocate torch.float8_e4m3fn; native FP8 is unsupported (backend audit 2026-10-25)"),
    "cpu": BackendCapabilities(
        "cpu", True, False, False, False,
        "no FP8 admission claim; CPU fallback may never be used silently"),
    "cuda": BackendCapabilities(
        "cuda", True, False, False, False,
        "unmeasured; declare qualified native FP8 support explicitly"),
}


def normalize_device(device: Any) -> tuple[str, str]:
    if not isinstance(device, str) or not device.strip():
        raise BackendAdmissionError("invalid_device", device=str(device),
                                    detail="device must be a nonempty string")
    raw = device.strip().lower()
    base, separator, index = raw.partition(":")
    if base not in DEFAULT_BACKENDS:
        raise BackendAdmissionError("unknown_device", device=raw,
                                    detail=f"supported devices are {sorted(DEFAULT_BACKENDS)}")
    if separator and (not index.isdigit() or (base == "cpu")):
        raise BackendAdmissionError("invalid_device", device=raw,
                                    detail="device index must be numeric and cpu takes no index")
    return raw, base


@dataclass(frozen=True)
class LoadAdmission:
    requested_device: str
    effective_device: str
    precision_kind: str
    storage: str
    admitted: bool
    reasons: tuple[str, ...]
    fallback_allowed: bool = False
    dequantization_allowed: bool = False

    def to_dict(self) -> dict:
        return {"requested_device": self.requested_device,
                "effective_device": self.effective_device,
                "precision_kind": self.precision_kind, "storage": self.storage,
                "admitted": self.admitted, "reasons": list(self.reasons),
                "fallback_allowed": self.fallback_allowed,
                "dequantization_allowed": self.dequantization_allowed}


def _resolved_backend(device: str, capabilities: BackendCapabilities | None) -> tuple[str, str, BackendCapabilities]:
    requested, base = normalize_device(device)
    caps = DEFAULT_BACKENDS[base] if capabilities is None else capabilities
    if not isinstance(caps, BackendCapabilities):
        raise BackendAdmissionError("invalid_capabilities", device=requested,
                                    detail="capabilities must be BackendCapabilities")
    if caps.device != base:
        raise BackendAdmissionError("capability_device_mismatch", device=requested,
                                    detail=f"capabilities describe {caps.device!r}, not {base!r}")
    return requested, base, caps


def evaluate_admission(identity: PrecisionIdentity, *, device: str,
                       capabilities: BackendCapabilities | None = None) -> LoadAdmission:
    """Evaluate one explicit device/precision request without changing it."""
    requested, base, caps = _resolved_backend(device, capabilities)
    reasons: list[str] = []
    if identity.kind == OFFICIAL_NATIVE_FP8:
        if base == "mps":
            reasons.append("mps_fp8_unsupported: MPS cannot allocate float8_e4m3fn")
        elif base == "cpu":
            reasons.append("cpu_fp8_unsupported: native FP8 is not admitted on CPU")
        if not caps.fp8_storage:
            reasons.append("backend_lacks_fp8_storage")
        if not caps.fp8_compute:
            reasons.append("backend_lacks_fp8_compute")
        if identity.activation_quantized and not caps.fp8_activation_quantization:
            reasons.append("backend_lacks_fp8_activation_quantization")
    elif identity.kind == DEQUANTIZED_FP8_DERIVATIVE:
        if not identity.dequantized:
            reasons.append("derivative_not_labelled_dequantized")
        if not caps.bf16:
            reasons.append("backend_lacks_derivative_storage_dtype")
        if identity.storage not in ("bf16", "fp32"):
            reasons.append("derivative_storage_not_admitted")
    else:
        if not caps.bf16:
            reasons.append("backend_lacks_bf16")
    reasons = list(dict.fromkeys(reasons))
    return LoadAdmission(
        requested_device=requested, effective_device=requested,
        precision_kind=identity.kind, storage=identity.storage,
        admitted=not reasons, reasons=tuple(reasons),
        fallback_allowed=False, dequantization_allowed=False)


def admit_load(identity: PrecisionIdentity, *, device: str,
               capabilities: BackendCapabilities | None = None) -> LoadAdmission:
    """Return an explicit admission or refuse; never switch device or precision."""
    admission = evaluate_admission(identity, device=device, capabilities=capabilities)
    if not admission.admitted:
        raise BackendAdmissionError(admission.reasons[0].split(":", 1)[0],
                                    device=admission.requested_device,
                                    detail="; ".join(admission.reasons), identity=identity)
    return admission


# ---------------------------------------------------------------------------
# Derivative P1/spine admission


def derivative_precision_binding(config: Spark2_5Config,
                                 precision: PrecisionIdentity) -> dict | None:
    """Refuse precision labels that do not match ``config.replica_precision``.

    Returns the declared provenance for a derivative config and ``None`` for an
    ordinary official config. An official BF16/FP8 identity can never load a
    derivative config, and a derivative identity can never load an official
    config or an artifact without declared provenance.
    """
    if not isinstance(config, Spark2_5Config):
        raise PrecisionError("config must be a validated Spark2_5Config")
    if not isinstance(precision, PrecisionIdentity):
        raise PrecisionError("precision must be a PrecisionIdentity")
    declared = config.raw.get("replica_precision")
    if declared is None:
        if precision.kind == DEQUANTIZED_FP8_DERIVATIVE:
            raise PrecisionError(
                "a derivative precision identity requires config.replica_precision provenance")
        return None
    declared = _mapping(declared, "replica_precision", PrecisionError)
    if declared.get("kind") != DEQUANTIZED_FP8_DERIVATIVE:
        raise PrecisionError("config.replica_precision.kind is not a dequantized derivative")
    if precision.kind != DEQUANTIZED_FP8_DERIVATIVE:
        raise PrecisionError(
            "an official FP8/BF16 identity cannot load this dequantized derivative config")
    if "quantization_config" in config.raw:
        raise PrecisionError("derivative storage must not retain quantization_config")
    if config.raw.get("dtype") != "bfloat16":
        raise PrecisionError("the admitted derivative config must declare dtype=bfloat16")
    expected = {
        "parent_kind": OFFICIAL_NATIVE_FP8,
        "parent_artifact_sha256": precision.derivative_of,
        "dequantization_sha256": precision.dequantization_sha256,
        "activation_semantics": precision.activation,
        "equivalent_to_bf16_release": False,
    }
    for key, value in expected.items():
        if declared.get(key) != value:
            raise PrecisionError(
                f"config.replica_precision.{key} differs from the derivative identity")
    if precision.storage != "bf16" or precision.weight_dtype != "BF16":
        raise PrecisionError("the derivative identity is not BF16 storage")
    if precision.equivalent_to_bf16_release is not False:
        raise PrecisionError("a derivative cannot equal the official BF16 release")
    return dict(declared)


def evaluate_derivative_p1_spine(
    precision: PrecisionIdentity, config: Spark2_5Config, *, device: str = "mps",
    capabilities: BackendCapabilities | None = None,
    spine_sites: Sequence[int] = P1_SPINE_SITES, spine_slots: int = P1_SPINE_SLOTS,
    memory_size: int = P1_MEMORY_SIZE, heads: int = P1_PORT_HEADS,
) -> dict:
    """Admit exactly the current P1/spine experiment on a derivative artifact.

    It never substitutes official FP8 or official BF16, never moves the
    requested device and never dequantizes. The late 23/31 sites are refused
    here; they need the separately declared matched 36-layer comparison.
    """
    reasons: list[str] = []
    if not isinstance(config, Spark2_5Config):
        raise PrecisionError("config must be a validated Spark2_5Config")
    try:
        derivative_precision_binding(config, precision)
    except PrecisionError as error:
        reasons.append(f"precision_substitution_refused: {error}")
    if not isinstance(precision, PrecisionIdentity):
        raise PrecisionError("precision must be a PrecisionIdentity")
    if precision.kind != DEQUANTIZED_FP8_DERIVATIVE:
        reasons.append(
            "precision_substitution_refused: official FP8 and official BF16 are "
            "separate identities and are not this experiment")
    if "quantization_config" in config.raw:
        reasons.append("derivative_storage_retains_quantization_config")
    if config.raw.get("dtype") != "bfloat16":
        reasons.append("derivative_config_not_bfloat16")
    layers = config.geometry.num_hidden_layers
    if layers != CURRENT_SOURCE_LAYERS:
        reasons.append(
            f"current_p1_requires_{CURRENT_SOURCE_LAYERS}_layer_source: got {layers}")
    requested: list[int] = []
    if not isinstance(spine_sites, (list, tuple)):
        reasons.append("spine_sites_must_be_a_sequence")
    else:
        for site in spine_sites:
            if type(site) is not int:
                reasons.append("spine_sites_must_be_integers")
                break
            requested.append(site)
    if requested:
        if len(set(requested)) != len(requested):
            reasons.append("spine_sites_must_be_distinct")
        if any(site < 0 or site >= layers for site in requested):
            reasons.append("spine_site_outside_the_decoder")
        if tuple(requested) != P1_SPINE_SITES:
            reasons.append("spine_sites_differ_from_current_p1_3_7")
        if set(requested) & set(LATE_SPINE_SITES):
            reasons.append(
                "late_sites_require_the_matched_36_layer_comparison_not_the_early_p1_arm")
    if spine_slots != P1_SPINE_SLOTS:
        reasons.append("spine_slot_read_budget_differs_from_current_p1")
    if memory_size != P1_MEMORY_SIZE or heads != P1_PORT_HEADS:
        reasons.append("memory_port_geometry_differs_from_current_p1")
    backend = evaluate_admission(precision, device=device, capabilities=capabilities)
    reasons.extend(backend.reasons)
    normalized = tuple(requested)
    return {
        "admitted": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "device": backend.effective_device,
        "precision_kind": precision.kind,
        "storage": precision.storage,
        "execution_format": DERIVATIVE_EXECUTION_FORMAT,
        "activation_semantics": precision.activation,
        "fallback_allowed": False,
        "dequantization_allowed": False,
        "substitution_allowed": False,
        "requires_external_authorization": True,
        "backend_admission": backend.to_dict(),
        "p1": {
            "profile": "P1",
            "active_experts": 1,
            "width": 256,
            "hidden_width": 512,
            "selected_parameter_owner": "expert_0.gate/up/down",
            "optimizer": "NativeFusedAdamW",
            "optimizer_state_dtype": "FP32",
            "frozen_original_backbone": True,
            "single_parameter_owner": True,
            "zero_init_output_projections": True,
        },
        "spine": {
            "sites": list(normalized),
            "candidate_late_sites": list(LATE_SPINE_SITES),
            "slots": int(spine_slots),
            "memory_size": int(memory_size),
            "heads": int(heads),
            "current_p1_spine": normalized == P1_SPINE_SITES,
            "late_sites_admitted": False,
            "late_site_comparison": (
                "matched_candidate_23_31_on_the_same_36_layer_source_only"),
        },
        "adapter_kwargs": {
            "device": backend.effective_device,
            "require_official_geometry": True,
            "projector_in_dim": int(config.geometry.hidden_size),
            "zero_init_output": True,
            "memory_size": int(memory_size),
            "heads": int(heads),
            "spine_sites": list(normalized),
            "spine_slots": int(spine_slots),
        },
        "native_execution": False,
    }


# ---------------------------------------------------------------------------
# Tokenizer and chat-template identity


@dataclass(frozen=True)
class TokenizerIdentity:
    tokenizer_class: str
    model_max_length: int
    vocab_size: int
    special_tokens: dict
    tokenizer_config_sha256: str
    chat_template_sha256: str
    inline_chat_template_sha256: str | None
    chat_template_sources_agree: bool | None
    tokenizer_json_sha256: str | None
    thinking_modes: dict
    default_thinking_mode: str
    reasoning_delimiters: tuple[str, ...]

    @property
    def digest(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "tokenizer_class": self.tokenizer_class,
            "model_max_length": self.model_max_length,
            "vocab_size": self.vocab_size,
            "special_tokens": dict(self.special_tokens),
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "inline_chat_template_sha256": self.inline_chat_template_sha256,
            "chat_template_sources_agree": self.chat_template_sources_agree,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "thinking_modes": {name: dict(values) for name, values in self.thinking_modes.items()},
            "default_thinking_mode": self.default_thinking_mode,
            "reasoning_delimiters": list(self.reasoning_delimiters),
        }


def _as_text(value: Any, name: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise TokenizerIdentityError(f"{name} is not UTF-8") from error
    if not isinstance(value, str):
        raise TokenizerIdentityError(f"{name} must be text")
    return value


def _special_token(value: Any, name: str) -> str:
    if isinstance(value, Mapping):
        value = value.get("content")
    return _text(value, name, TokenizerIdentityError)


def _validate_chat_template(template: str, label: str) -> None:
    for token in ("enable_thinking", "reasoning_effort", "'high'", "'low'",
                  *REASONING_DELIMITERS):
        if token not in template:
            raise TokenizerIdentityError(f"{label} is missing {token!r}")
    if "not messages" not in template and "No messages provided" not in template:
        raise TokenizerIdentityError(f"{label} must refuse an empty message list")
    if "default(true)" not in template.replace(" ", ""):
        raise TokenizerIdentityError(f"{label} must default thinking on")
    if "'low'" not in template or "'high'" not in template:
        raise TokenizerIdentityError(f"{label} must name the low and high thinking efforts")
    if "medium" in template or "auto" in template:
        raise TokenizerIdentityError(f"{label} declares an unsupported thinking mode")


def load_tokenizer_identity(config: Spark2_5Config, tokenizer_config: Mapping,
                            chat_template: str | bytes, *,
                            tokenizer_json: str | bytes | None = None) -> TokenizerIdentity:
    config_map = _mapping(tokenizer_config, "tokenizer_config", TokenizerIdentityError)
    if config_map.get("tokenizer_class") != "PreTrainedTokenizerFast":
        raise TokenizerIdentityError("tokenizer_class must be PreTrainedTokenizerFast")
    model_max_length = _require_int(config_map.get("model_max_length"), "model_max_length",
                                    Error=TokenizerIdentityError)
    if model_max_length <= 0:
        raise TokenizerIdentityError("model_max_length must be positive")
    special_tokens = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        special_tokens[key] = _special_token(config_map.get(key), key)
    if len(set(special_tokens.values())) != len(special_tokens):
        raise TokenizerIdentityError("bos/eos/pad/unk tokens must be distinct")
    expected_ids = {"bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2}
    for key, value in expected_ids.items():
        if config.raw.get(key) != value:
            raise TokenizerIdentityError(f"config.{key} must be {value} for the pinned tokenizer")

    template = _as_text(chat_template, "chat_template")
    _validate_chat_template(template, "chat_template")
    inline_sha = None
    agree = None
    inline = config_map.get("chat_template")
    if inline is not None:
        inline_text = _as_text(inline, "tokenizer_config.chat_template")
        _validate_chat_template(inline_text, "tokenizer_config.chat_template")
        inline_sha = sha256_bytes(inline_text.encode())
        # The pinned bundle serializes the inline copy with escaped newlines
        # resolved differently from chat_template.jinja. Both bytes are pinned;
        # the standalone file is authoritative and the inline copy must still
        # declare the same thinking contract. Do not silently pick one.
        agree = inline_text == template

    tokenizer_json_sha = None
    if tokenizer_json is not None:
        tokenizer_json_sha = sha256_bytes(_as_text(tokenizer_json, "tokenizer_json").encode())
    return TokenizerIdentity(
        tokenizer_class="PreTrainedTokenizerFast",
        model_max_length=model_max_length, vocab_size=config.geometry.vocab_size,
        special_tokens=special_tokens,
        tokenizer_config_sha256=sha256_bytes(canonical_json(config_map).encode()),
        chat_template_sha256=sha256_bytes(template.encode()),
        inline_chat_template_sha256=inline_sha,
        chat_template_sources_agree=agree,
        tokenizer_json_sha256=tokenizer_json_sha,
        thinking_modes={name: dict(values) for name, values in THINKING_MODES.items()},
        default_thinking_mode=DEFAULT_THINKING_MODE,
        reasoning_delimiters=REASONING_DELIMITERS,
    )


def resolve_thinking_mode(identity: TokenizerIdentity, mode: Any) -> dict:
    """Map an explicit generation mode to template kwargs; reject anything else."""
    if not isinstance(mode, str) or mode not in THINKING_MODES:
        raise ThinkingModeError(
            f"thinking mode must be one of {sorted(THINKING_MODES)}; got {mode!r}")
    resolved = dict(identity.thinking_modes[mode])
    if mode == "none":
        if "reasoning_effort" in resolved:
            raise ThinkingModeError("thinking mode none must not request a reasoning effort")
    else:
        effort = resolved.get("reasoning_effort")
        if resolved.get("enable_thinking") is not True or effort not in ("low", "high"):
            raise ThinkingModeError(f"thinking mode {mode!r} is not pinned by the template")
    return resolved


# ---------------------------------------------------------------------------
# Tensor inventory and tied embeddings


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        total = 1
        for value in self.shape:
            total *= value
        return total


def _parse_tensor(entry: Mapping, index: int) -> TensorRecord:
    entry = _mapping(entry, f"tensors[{index}]", InventoryError)
    name = entry.get("name")
    if not isinstance(name, str) or not name or name.startswith(("__", ".")):
        raise InventoryError(f"tensors[{index}].name must be a normalised tensor name")
    dtype = entry.get("dtype")
    if not isinstance(dtype, str) or dtype.upper() not in DTYPE_BYTES:
        raise InventoryError(f"{name}: unknown dtype {dtype!r}")
    shape = entry.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise InventoryError(f"{name}: shape must be a nonempty sequence")
    values = tuple(_require_int(value, f"{name}.shape", Error=InventoryError) for value in shape)
    return TensorRecord(name=name, dtype=dtype.upper(), shape=values)


def _base_dtypes(precision: PrecisionIdentity) -> tuple[str, ...]:
    if precision.kind == OFFICIAL_NATIVE_FP8:
        return ("BF16",)
    if precision.kind == OFFICIAL_BF16:
        return ("BF16",)
    return ("BF16", "F32")


def _expected_tensors(config: Spark2_5Config, precision: PrecisionIdentity,
                      base_dtype: str) -> dict[str, tuple[str, ...]]:
    geometry = config.geometry
    expected: dict[str, tuple[str, ...]] = {
        "model.embedding.weight": (base_dtype, geometry.vocab_size, geometry.hidden_size),
    }
    quantized = precision.kind == OFFICIAL_NATIVE_FP8
    for index in range(geometry.num_hidden_layers):
        prefix = f"model.layers.{index}."
        expected[prefix + "input_layernorm.weight"] = (base_dtype, geometry.hidden_size)
        expected[prefix + "post_attention_layernorm.weight"] = (base_dtype, geometry.hidden_size)
        for suffix, shape in (
            ("self_attn.q_k_v_proj.weight", (geometry.qkv_dim, geometry.hidden_size)),
            ("self_attn.out_proj.weight", (geometry.hidden_size, geometry.q_dim)),
            ("self_attn.g_proj.weight", (geometry.num_attention_heads, geometry.hidden_size)),
            ("mlp.gate_proj.weight", (geometry.intermediate_size, geometry.hidden_size)),
            ("mlp.up_proj.weight", (geometry.intermediate_size, geometry.hidden_size)),
            ("mlp.down_proj.weight", (geometry.hidden_size, geometry.intermediate_size)),
        ):
            weight_dtype = "F8_E4M3" if quantized and suffix.rsplit(".", 1)[0] in QUANTIZED_SUFFIXES else base_dtype
            expected[prefix + suffix] = (weight_dtype, *shape)
            if quantized and suffix.rsplit(".", 1)[0] in QUANTIZED_SUFFIXES:
                expected[prefix + suffix + "_scale"] = (base_dtype, shape[0], 1)
    expected["model.norm.weight"] = (base_dtype, geometry.hidden_size)
    # The output table is tied and never serialized.
    return expected


@dataclass(frozen=True)
class InventoryReport:
    storage: str
    precision_kind: str
    tensor_count: int
    parameter_elements: int
    scale_tensor_count: int
    quantized_tensor_count: int
    dtypes: tuple[str, ...]
    payload_bytes: int
    digest: str

    def to_dict(self) -> dict:
        return {"storage": self.storage, "precision_kind": self.precision_kind,
                "tensor_count": self.tensor_count,
                "parameter_elements": self.parameter_elements,
                "scale_tensor_count": self.scale_tensor_count,
                "quantized_tensor_count": self.quantized_tensor_count,
                "dtypes": list(self.dtypes), "payload_bytes": self.payload_bytes,
                "digest": self.digest}


def verify_tensor_inventory(tensors: Sequence[Mapping], config: Spark2_5Config,
                            precision: PrecisionIdentity) -> InventoryReport:
    """Match a header/index inventory to Spark2_5 geometry and storage identity."""
    if not isinstance(tensors, (list, tuple)) or not tensors:
        raise InventoryError("tensor inventory must be a nonempty sequence")
    records = [_parse_tensor(entry, index) for index, entry in enumerate(tensors)]
    by_name: dict[str, TensorRecord] = {}
    for record in records:
        if record.name in by_name:
            raise InventoryError(f"duplicate tensor: {record.name}")
        by_name[record.name] = record
    if "lm_head.weight" in by_name or "model.lm_head.weight" in by_name:
        raise TiedEmbeddingError("tied lm_head.weight must not be serialized")
    if "model.embedding.weight" not in by_name:
        raise InventoryError("missing model.embedding.weight")

    base_dtypes = _base_dtypes(precision)
    embedding_dtype = by_name["model.embedding.weight"].dtype
    if embedding_dtype not in base_dtypes:
        raise InventoryError(f"embedding dtype {embedding_dtype} is not {base_dtypes}")
    expected = _expected_tensors(config, precision, embedding_dtype)
    missing = sorted(set(expected) - set(by_name))
    extra = sorted(set(by_name) - set(expected))
    if missing:
        raise InventoryError(f"missing expected tensors: {missing[:5]}")
    if extra:
        raise InventoryError(f"unexpected tensors for {precision.kind}: {extra[:5]}")
    for name, (dtype, *shape) in expected.items():
        record = by_name[name]
        if record.dtype != dtype:
            raise InventoryError(f"{name}: dtype {record.dtype} must be {dtype}")
        if record.shape != tuple(shape):
            raise InventoryError(f"{name}: shape {record.shape} must be {tuple(shape)}")

    parameter_elements = sum(record.elements for record in records
                             if not record.name.endswith(("weight_scale", "weight_zero_point")))
    if parameter_elements != config.geometry.parameter_elements:
        raise InventoryError(
            f"unique parameter elements {parameter_elements} != declared "
            f"{config.geometry.parameter_elements}")
    scale_count = sum(record.name.endswith(("weight_scale", "weight_zero_point"))
                      for record in records)
    quantized_count = sum(record.dtype.startswith("F8_") for record in records)
    payload_bytes = sum(record.elements * DTYPE_BYTES[record.dtype] for record in records)
    summary = {
        "storage": precision.storage,
        "precision_kind": precision.kind,
        "names": sorted(by_name),
        "parameter_elements": parameter_elements,
        "scale_tensor_count": scale_count,
        "quantized_tensor_count": quantized_count,
        "payload_bytes": payload_bytes,
    }
    return InventoryReport(
        storage=precision.storage, precision_kind=precision.kind,
        tensor_count=len(records), parameter_elements=parameter_elements,
        scale_tensor_count=scale_count, quantized_tensor_count=quantized_count,
        dtypes=tuple(sorted({record.dtype for record in records})),
        payload_bytes=payload_bytes, digest=fingerprint(summary))


@dataclass(frozen=True)
class TiedEmbeddingInventory:
    declared_tied: bool
    inspected: bool
    serialized_lm_head: bool
    embedding_shape: tuple[int, ...] | None
    unique_parameter_elements: int
    logits_path: str

    def to_dict(self) -> dict:
        return {"declared_tied": self.declared_tied, "inspected": self.inspected,
                "serialized_lm_head": self.serialized_lm_head,
                "embedding_shape": None if self.embedding_shape is None else list(self.embedding_shape),
                "unique_parameter_elements": self.unique_parameter_elements,
                "logits_path": self.logits_path}


def verify_tied_embeddings(config: Spark2_5Config,
                           tensors: Sequence[Mapping] | None) -> TiedEmbeddingInventory:
    geometry = config.geometry
    if not geometry.tie_word_embeddings:
        raise TiedEmbeddingError("Spark2_5 output embeddings must be tied")
    if tensors is None:
        return TiedEmbeddingInventory(
            declared_tied=True, inspected=False, serialized_lm_head=False,
            embedding_shape=None, unique_parameter_elements=geometry.parameter_elements,
            logits_path="F.linear(hidden_states, model.embedding.weight)")
    records = [_parse_tensor(entry, index) for index, entry in enumerate(tensors)]
    names = {record.name for record in records}
    if "lm_head.weight" in names or "model.lm_head.weight" in names:
        raise TiedEmbeddingError("tied lm_head.weight must not be serialized")
    embedding = next((record for record in records
                      if record.name == "model.embedding.weight"), None)
    if embedding is None:
        raise InventoryError("missing model.embedding.weight")
    if embedding.shape != (geometry.vocab_size, geometry.hidden_size):
        raise InventoryError("embedding shape differs from the declared vocabulary and width")
    return TiedEmbeddingInventory(
        declared_tied=True, inspected=True, serialized_lm_head=False,
        embedding_shape=embedding.shape,
        unique_parameter_elements=geometry.parameter_elements,
        logits_path="F.linear(hidden_states, model.embedding.weight)")


# ---------------------------------------------------------------------------
# Dimension-specific projector and extension inventory


@dataclass(frozen=True)
class ProjectorInventory:
    in_dim: int
    hidden_dim: int
    out_dim: int
    normalization: str
    activation: str
    zero_init_output: bool
    tensors: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def parameter_elements(self) -> int:
        return sum(_shape_elements(shape) for _, shape in self.tensors)

    def to_dict(self) -> dict:
        return {"in_dim": self.in_dim, "hidden_dim": self.hidden_dim,
                "out_dim": self.out_dim, "normalization": self.normalization,
                "activation": self.activation,
                "zero_init_output": self.zero_init_output,
                "parameter_elements": self.parameter_elements,
                "tensors": [{"name": name, "shape": list(shape)}
                            for name, shape in self.tensors]}


@dataclass(frozen=True)
class MemoryPortInventory:
    layer: int
    layer_type: str
    width: int
    zero_init_output: bool
    status: str = "hypothesis"

    def to_dict(self) -> dict:
        return {"layer": self.layer, "layer_type": self.layer_type,
                "width": self.width, "zero_init_output": self.zero_init_output,
                "status": self.status}


@dataclass(frozen=True)
class ExtensionInventory:
    projector: ProjectorInventory | None
    memory_ports: tuple[MemoryPortInventory, ...]
    enabled_for_native_parity: bool = False

    @property
    def trainable_tensors(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return () if self.projector is None else self.projector.tensors

    def to_dict(self) -> dict:
        return {"projector": None if self.projector is None else self.projector.to_dict(),
                "memory_ports": [port.to_dict() for port in self.memory_ports],
                "enabled_for_native_parity": self.enabled_for_native_parity}


def _shape_elements(shape: Sequence[int]) -> int:
    total = 1
    for value in shape:
        total *= value
    return total


def _reject_extension_width(width: int, name: str) -> None:
    if width == MINICPM5_REUSED_DIMENSIONS["projector_out_dim"]:
        raise ExtensionStateError(f"{name} 2048 is the MiniCPM5 connector width; use the Spark width")


def build_projector(config: Spark2_5Config, *, in_dim: int,
                    hidden_dim: int | None = None,
                    zero_init_output: bool = True) -> ProjectorInventory:
    geometry = config.geometry
    in_dim = _require_int(in_dim, "projector.in_dim", Error=ExtensionStateError)
    hidden_dim = (geometry.hidden_size if hidden_dim is None
                  else _require_int(hidden_dim, "projector.hidden_dim", Error=ExtensionStateError))
    out_dim = geometry.hidden_size
    _reject_extension_width(out_dim, "projector out_dim")
    _reject_extension_width(hidden_dim, "projector hidden_dim")
    _require_bool(zero_init_output, "projector.zero_init_output", ExtensionStateError)
    tensors = (
        ("extension.vision_projector.norm.weight", (in_dim,)),
        ("extension.vision_projector.norm.bias", (in_dim,)),
        ("extension.vision_projector.fc1.weight", (hidden_dim, in_dim)),
        ("extension.vision_projector.fc1.bias", (hidden_dim,)),
        ("extension.vision_projector.fc2.weight", (out_dim, hidden_dim)),
        ("extension.vision_projector.fc2.bias", (out_dim,)),
    )
    return ProjectorInventory(
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        normalization="layernorm", activation="gelu",
        zero_init_output=zero_init_output, tensors=tensors)


def build_memory_ports(config: Spark2_5Config, *,
                       layers: Sequence[int] | None = None) -> tuple[MemoryPortInventory, ...]:
    geometry = config.geometry
    if layers is None:
        layers = tuple(index for index in (11, 23) if index < geometry.num_hidden_layers)
        if geometry.num_hidden_layers < 24:
            layers = ()
    layer_list = tuple(layers)
    if len(set(layer_list)) != len(layer_list):
        raise ExtensionStateError("memory port layers must be distinct")
    if layer_list == MINICPM5_REUSED_DIMENSIONS["accepted_ports"]:
        raise ExtensionStateError("ports 13/27 are the MiniCPM5 convention; choose Spark hypotheses")
    ports = []
    for layer in layer_list:
        layer = _require_int(layer, "memory_port.layer", minimum=0, Error=ExtensionStateError)
        if layer >= geometry.num_hidden_layers:
            raise ExtensionStateError(f"memory port layer {layer} is outside the decoder")
        if geometry.layer_types[layer] != "full_attention":
            raise ExtensionStateError(f"memory port layer {layer} is not a full-attention block")
        ports.append(MemoryPortInventory(
            layer=layer, layer_type=geometry.layer_types[layer],
            width=geometry.hidden_size, zero_init_output=True))
    return tuple(ports)


def build_extensions(config: Spark2_5Config, *, projector_in_dim: int | None = None,
                     projector_hidden_dim: int | None = None,
                     memory_ports: Sequence[int] | None = None) -> ExtensionInventory:
    projector = (None if projector_in_dim is None else
                 build_projector(config, in_dim=projector_in_dim,
                                 hidden_dim=projector_hidden_dim,
                                 zero_init_output=True))
    return ExtensionInventory(projector=projector,
                              memory_ports=build_memory_ports(config, layers=memory_ports))


# ---------------------------------------------------------------------------
# Extension-only export/reload validation


def _spine_export_tensors(spine: Mapping, trainable_dtype: str) -> list[dict]:
    """Validate one additive extension.spine inventory without relabeling."""
    if not isinstance(trainable_dtype, str) \
            or trainable_dtype.upper() not in ("FLOAT32", "BF16"):
        raise ExtensionStateError("spine tensor dtype differs")
    if spine.get("schema") != SPINE_SCHEMA:
        raise ExtensionStateError("spine schema differs")
    spec = _mapping(spine.get("spec"), "spine.spec", ExtensionStateError)
    for key in ("hidden_size", "slots", "heads"):
        _require_int(spec.get(key), f"spine.spec.{key}", Error=ExtensionStateError)
    if spec.get("packet_width") != 512 or spec.get("workspace_width") != 256:
        raise ExtensionStateError("spine dimensions must be 512 packet / 256 workspace")
    sites = spec.get("sites")
    if not isinstance(sites, list) or not sites or len(set(sites)) != len(sites) \
            or any(not isinstance(site, int) or isinstance(site, bool) or site < 0
                   for site in sites):
        raise ExtensionStateError("spine sites are invalid")
    if spine.get("trainable") is not True \
            or spine.get("zero_effect_final_projections") is not True \
            or spine.get("live_gate_path") is not True:
        raise ExtensionStateError("spine initialization/gradient contract differs")
    tensors = spine.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ExtensionStateError("spine tensor inventory is empty")
    expected_dtype = trainable_dtype.upper()
    for tensor in tensors:
        tensor = _mapping(tensor, "spine.tensor", ExtensionStateError)
        name = tensor.get("name")
        if not isinstance(name, str) or not name.startswith("extension.spine."):
            raise ExtensionStateError("spine tensor is outside extension.spine")
        if tensor.get("dtype") != expected_dtype:
            raise ExtensionStateError("spine tensor dtype differs")
        shape = tensor.get("shape")
        if not isinstance(shape, list) or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in shape):
            raise ExtensionStateError("spine tensor shape is invalid")
    if spine.get("parameter_elements") != sum(
            _shape_elements(tuple(tensor["shape"])) for tensor in tensors):
        raise ExtensionStateError("spine parameter count differs")
    if spine.get("parameter_inventory_sha256") != fingerprint(tensors):
        raise ExtensionStateError("spine inventory digest differs")
    return [dict(tensor) for tensor in tensors]


def build_extension_export(config: Spark2_5Config, precision: PrecisionIdentity,
                           extensions: ExtensionInventory, *,
                           trainable_dtype: str = "float32",
                           spine: Mapping | None = None) -> dict:
    """Build a metadata/state-index export containing no base checkpoint bytes."""
    if not isinstance(extensions, ExtensionInventory):
        raise ExtensionStateError("extensions must be an ExtensionInventory")
    trainable_dtype = str(trainable_dtype).lower()
    if trainable_dtype not in ("float32", "bfloat16"):
        raise ExtensionStateError("new trainable tensors must stay float32 or bfloat16")
    if precision.kind == OFFICIAL_NATIVE_FP8 and trainable_dtype != "float32":
        raise ExtensionStateError("new tensors on a frozen native FP8 base must stay float32")
    tensors = [{"name": name, "shape": list(shape), "dtype": trainable_dtype.upper()}
               for name, shape in extensions.trainable_tensors]
    for tensor in tensors:
        if not tensor["name"].startswith("extension."):
            raise ExtensionStateError("extension export may contain only extension.* tensors")
    payload = {
        "schema": EXTENSION_EXPORT_SCHEMA,
        "base": {
            "precision_kind": precision.kind, "storage": precision.storage,
            "weight_dtype": precision.weight_dtype,
            "artifact_sha256": precision.artifact_sha256,
            "quantization_fingerprint": precision.quantization_fingerprint,
            "derivative_of": precision.derivative_of,
            "config_digest": config.digest,
            "geometry_fingerprint": fingerprint(config.geometry.to_dict()),
        },
        "extension": extensions.to_dict(),
        "trainable_dtype": trainable_dtype.upper(),
        "tensors": tensors,
        "contains_base_weights": False,
        "original_checkpoint_bytes_rewritten": False,
        "native_execution": False,
    }
    if spine is not None:
        spine = _mapping(spine, "spine", ExtensionStateError)
        spine_tensors = _spine_export_tensors(spine, trainable_dtype)
        payload["schema"] = SPINE_EXTENSION_EXPORT_SCHEMA
        payload["extension"]["spine"] = dict(spine)
        payload["spine_migration"] = {
            "from": EXTENSION_EXPORT_SCHEMA,
            "to": SPINE_EXTENSION_EXPORT_SCHEMA,
            "new_namespace": "extension.spine",
            "vision_projector_relabelled": False,
        }
        tensors.extend(spine_tensors)
    payload["fingerprint"] = fingerprint(payload)
    return payload


def validate_extension_export(payload: Mapping, config: Spark2_5Config,
                              precision: PrecisionIdentity) -> dict:
    """Validate an extension-only export/reload against current base identity."""
    payload = _mapping(payload, "extension_export", ExtensionStateError)
    if payload.get("schema") not in (EXTENSION_EXPORT_SCHEMA, SPINE_EXTENSION_EXPORT_SCHEMA):
        raise ExtensionStateError("unsupported extension export schema")
    recorded = _mapping(payload.get("fingerprint"), "fingerprint", ExtensionStateError) \
        if isinstance(payload.get("fingerprint"), Mapping) else payload.get("fingerprint")
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if recorded != fingerprint(body):
        raise ExtensionStateError("extension export fingerprint mismatch")
    if payload.get("contains_base_weights") is not False:
        raise ExtensionStateError("extension export must not contain base weights")
    if payload.get("original_checkpoint_bytes_rewritten") is not False:
        raise ExtensionStateError("extension export must not rewrite original checkpoint bytes")
    base = _mapping(payload.get("base"), "base", ExtensionStateError)
    expected_base = {
        "precision_kind": precision.kind, "storage": precision.storage,
        "weight_dtype": precision.weight_dtype,
        "artifact_sha256": precision.artifact_sha256,
        "quantization_fingerprint": precision.quantization_fingerprint,
        "derivative_of": precision.derivative_of,
        "config_digest": config.digest,
        "geometry_fingerprint": fingerprint(config.geometry.to_dict()),
    }
    for key, value in expected_base.items():
        if base.get(key) != value:
            raise ExtensionStateError(f"extension export base.{key} differs from the current base")
    extension = _mapping(payload.get("extension"), "extension", ExtensionStateError)
    projector = extension.get("projector")
    projector_spec = None
    if projector is not None:
        projector = _mapping(projector, "extension.projector", ExtensionStateError)
        projector_spec = build_projector(
            config, in_dim=_require_int(projector.get("in_dim"), "projector.in_dim",
                                        Error=ExtensionStateError),
            hidden_dim=_require_int(projector.get("hidden_dim"), "projector.hidden_dim",
                                    Error=ExtensionStateError),
            zero_init_output=projector.get("zero_init_output"))
    ports = extension.get("memory_ports")
    if not isinstance(ports, list):
        raise ExtensionStateError("extension.memory_ports must be a list")
    port_layers = []
    for index, port in enumerate(ports):
        port = _mapping(port, f"extension.memory_ports[{index}]", ExtensionStateError)
        port_layers.append(_require_int(port.get("layer"), "memory_port.layer",
                                        minimum=0, Error=ExtensionStateError))
    expected_extensions = build_extensions(
        config, projector_in_dim=None if projector_spec is None else projector_spec.in_dim,
        projector_hidden_dim=None if projector_spec is None else projector_spec.hidden_dim,
        memory_ports=port_layers)
    extension_core = {key: value for key, value in extension.items() if key != "spine"}
    if expected_extensions.to_dict() != extension_core:
        raise ExtensionStateError("extension inventory does not match the current geometry")
    if "spine" in extension and payload["schema"] != SPINE_EXTENSION_EXPORT_SCHEMA:
        raise ExtensionStateError("spine tensors require the v2 extension schema")
    if "spine" not in extension and payload["schema"] == SPINE_EXTENSION_EXPORT_SCHEMA:
        raise ExtensionStateError("v2 extension requires an explicit spine inventory")
    expected_spine_tensors = []
    trainable_dtype = payload.get("trainable_dtype")
    spine = extension.get("spine")
    if spine is not None:
        spine = _mapping(spine, "extension.spine", ExtensionStateError)
        expected_spine_tensors = _spine_export_tensors(spine, trainable_dtype)
        migration = payload.get("spine_migration")
        if migration != {
                "from": EXTENSION_EXPORT_SCHEMA,
                "to": SPINE_EXTENSION_EXPORT_SCHEMA,
                "new_namespace": "extension.spine",
                "vision_projector_relabelled": False}:
            raise ExtensionStateError("spine migration declaration differs")
    elif "spine_migration" in payload:
        raise ExtensionStateError("spine migration declared without spine tensors")
    trainable_dtype = payload.get("trainable_dtype")
    if trainable_dtype not in ("FLOAT32", "BF16"):
        raise ExtensionStateError("trainable_dtype must be FLOAT32 or BF16")
    expected_tensors = [{"name": name, "shape": list(shape), "dtype": trainable_dtype}
                        for name, shape in expected_extensions.trainable_tensors]
    expected_tensors.extend(expected_spine_tensors)
    if payload.get("tensors") != expected_tensors:
        raise ExtensionStateError("extension tensor index does not match the declared modules")
    return {
        "schema": payload["schema"],
        "extension_only": True,
        "base_unchanged": True,
        "precision_kind": precision.kind,
        "trainable_dtype": trainable_dtype,
        "tensor_count": len(expected_tensors),
        "parameter_elements": sum(_shape_elements(tuple(item["shape"]))
                                  for item in expected_tensors),
        "memory_port_layers": [port.layer for port in expected_extensions.memory_ports],
    }


# ---------------------------------------------------------------------------
# Combined preparation report


STAGES = ("load", "inference", "input_gradients", "parameter_gradients",
          "single_update", "export_reload")


def build_stage_status(admission: LoadAdmission) -> dict:
    if not admission.admitted:
        load = f"blocked_precision_backend: {'; '.join(admission.reasons)}"
        upstream = "blocked_by_load"
    else:
        load = ("blocked_unexecuted: preparation contract only; no isolated Spark "
                "runtime, weights or remote code")
        upstream = "blocked_by_load"
    return {
        "load": load,
        "inference": upstream,
        "input_gradients": upstream,
        "parameter_gradients": upstream,
        "single_update": upstream,
        "export_reload": upstream,
    }


def prepare_spark2_5(
    config_raw: Mapping, tokenizer_config: Mapping, chat_template: str | bytes, *,
    tensors: Sequence[Mapping] | None = None, device: str = "mps",
    capabilities: BackendCapabilities | None = None,
    derivative_manifest: Mapping | None = None,
    artifact_sha256: str | None = None,
    require_official_geometry: bool = False,
    projector_in_dim: int | None = None,
    projector_hidden_dim: int | None = None,
    memory_ports: Sequence[int] | None = None,
    tokenizer_json: str | bytes | None = None,
) -> dict:
    """Assemble a metadata-only preparation report with explicit blockers."""
    config = load_spark2_5_config(config_raw,
                                  require_official_geometry=require_official_geometry)
    precision = classify_precision(config, artifact_sha256=artifact_sha256,
                                   derivative_manifest=derivative_manifest)
    tokenizer = load_tokenizer_identity(config, tokenizer_config, chat_template,
                                        tokenizer_json=tokenizer_json)
    schedule = derive_layer_schedule(config)
    extensions = build_extensions(
        config, projector_in_dim=projector_in_dim,
        projector_hidden_dim=projector_hidden_dim, memory_ports=memory_ports)
    inventory = None
    tied = verify_tied_embeddings(config, tensors)
    if tensors is not None:
        inventory = verify_tensor_inventory(tensors, config, precision)
    admission = evaluate_admission(precision, device=device, capabilities=capabilities)
    stages = build_stage_status(admission)
    blockers = ["load: no Spark runtime or weight payload is used by this preparation report",
                "inference: not run; no native Spark execution"]
    blockers.extend(f"{stage}: {value}" for stage, value in stages.items()
                    if value.startswith("blocked_precision_backend"))
    blockers.extend([
        "input_gradients: not measured",
        "parameter_gradients: not measured",
        "single_update: not executed",
        "export_reload: extension metadata validated only; no base checkpoint reload",
    ])
    if inventory is None:
        blockers.append("tensor_inventory: not inspected")
    if precision.kind == OFFICIAL_NATIVE_FP8:
        blockers.append("native_fp8: requires an isolated transformers 4.57.x + "
                        "compressed-tensors 0.16.0 environment and an FP8-capable backend")
    report = {
        "schema": SCHEMA,
        "status": "preparation_only_unqualified",
        "config": {"model_type": OFFICIAL_MODEL_TYPE,
                   "architectures": [OFFICIAL_ARCHITECTURE],
                   "digest": config.digest,
                   "official_geometry": config.geometry.is_official_geometry},
        "geometry": config.geometry.to_dict(),
        "layer_schedule": schedule.to_dict(),
        "precision": precision.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "tied_embeddings": tied.to_dict(),
        "inventory": None if inventory is None else inventory.to_dict(),
        "extensions": extensions.to_dict(),
        "admission": admission.to_dict(),
        "stage_status": stages,
        "blockers": blockers,
        "native_execution": False,
        "weights_downloaded": False,
        "remote_code_executed": False,
        "capability_claim": "none",
        "fallback_allowed": False,
        "dequantization_performed": False,
    }
    report["fingerprint"] = fingerprint(report)
    return report


__all__ = [
    "SCHEMA", "EXTENSION_EXPORT_SCHEMA",
    "OFFICIAL_BF16", "OFFICIAL_NATIVE_FP8", "DEQUANTIZED_FP8_DERIVATIVE",
    "PRECISION_KINDS", "OFFICIAL_GEOMETRY", "OFFICIAL_PARAMETER_ELEMENTS",
    "DERIVATIVE_MANIFEST_SCHEMA", "DERIVATIVE_EXECUTION_FORMAT",
    "DERIVATIVE_ACTIVATION_SEMANTICS", "DERIVATIVE_CLAIMS_FALSE",
    "P1_SPINE_SITES", "LATE_SPINE_SITES", "P1_SPINE_SLOTS", "P1_MEMORY_SIZE",
    "P1_PORT_HEADS", "CURRENT_SOURCE_LAYERS",
    "OFFICIAL_LAYER_TYPES", "MINICPM5_REUSED_DIMENSIONS", "THINKING_MODES",
    "DEFAULT_THINKING_MODE", "LAYER_TYPES", "QUANTIZED_SUFFIXES",
    "UNQUANTIZED_SUFFIXES", "STAGES",
    "Spark2_5ContractError", "GeometryError", "PrecisionError",
    "BackendAdmissionError", "TokenizerIdentityError", "ThinkingModeError",
    "TiedEmbeddingError", "InventoryError", "ExtensionStateError",
    "Spark2_5Geometry", "Spark2_5Config", "LayerSchedule",
    "PrecisionIdentity", "BackendCapabilities", "LoadAdmission",
    "TokenizerIdentity", "TensorRecord", "InventoryReport",
    "TiedEmbeddingInventory", "ProjectorInventory", "MemoryPortInventory",
    "ExtensionInventory",
    "canonical_json", "fingerprint", "sha256_bytes", "derivative_artifact_digest",
    "load_spark2_5_config", "derive_layer_schedule",
    "official_bf16_precision", "official_native_fp8_precision",
    "dequantized_fp8_derivative", "classify_precision",
    "derivative_precision_binding", "evaluate_derivative_p1_spine",
    "normalize_device", "DEFAULT_BACKENDS", "evaluate_admission", "admit_load",
    "load_tokenizer_identity", "resolve_thinking_mode",
    "verify_tensor_inventory", "verify_tied_embeddings",
    "build_projector", "build_memory_ports", "build_extensions",
    "build_extension_export", "validate_extension_export",
    "build_stage_status", "prepare_spark2_5",
]
