"""Contextual tiny Spark-shaped fixture for CPU engineering evidence.

The older ``test_spark2_5_adapter.py`` fixture is intentionally left untouched:
its blocks are pointwise and therefore cannot diagnose Spark token mixing,
full-context behavior or memory readout.  This module supplies a *separate*
fixture with a real causal self-attention path, a GELU MLP and Spark's tied
input/output embedding.  It remains a software fixture only; it is not the
native Spark checkpoint, does not establish native-FP8 behavior and cannot be
used to claim useful learning.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from replica_lab.architecture.spark2_5 import (
    load_spark2_5_config, official_bf16_precision, sha256_bytes,
)
from replica_lab.architecture.spark_adapter import Spark2_5Adapter
from replica_lab.architecture.spine_memory import SpineMemoryExperiment
from replica_lab.architecture.spine_text import SparkTextMemoryPath

ENGINEERING_LABEL = "engineering_cpu_contextual_fixture_only"
# The active SparkX2.6-4B source has 36 layers.  An 8-layer spark2_5 model is a
# separate tiny engineering fixture, never the active source geometry.
NATIVE_SEALED_LAYERS = 36
LATE_SITE_MIN_LAYERS = 32
DEFAULT_LAYERS = 32
DEFAULT_VOCAB = 32
DEFAULT_HIDDEN = 8
DEFAULT_HEADS = 2
DEFAULT_KV_HEADS = 1
DEFAULT_HEAD_DIM = 4
DEFAULT_INTERMEDIATE = 16
DEFAULT_SLIDING_WINDOW = 4


def contextual_config_raw(*, layers: int = DEFAULT_LAYERS,
                          vocab_size: int = DEFAULT_VOCAB,
                          max_position_embeddings: int = 64) -> dict:
    """Minimal Spark2_5-shaped config with a valid sliding/full schedule."""
    if layers < 4:
        raise ValueError("the contextual fixture requires at least four layers")
    if vocab_size < 4 or max_position_embeddings < 1:
        raise ValueError("fixture vocabulary and position bounds must be positive")
    layer_types = (["sliding_attention"] * 3 + ["full_attention"]) * (
        layers // 4) + ["sliding_attention"] * (layers % 4)
    layer_types = layer_types[:layers]
    if "full_attention" not in layer_types:
        layer_types[-1] = "full_attention"
    return {
        "architectures": ["Spark2_5ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_spark.Spark2_5Config",
            "AutoModel": "modeling_spark.Spark2_5Model",
            "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
        },
        "model_type": "spark2_5",
        "dtype": "bfloat16",
        "num_hidden_layers": layers,
        "hidden_size": DEFAULT_HIDDEN,
        "intermediate_size": DEFAULT_INTERMEDIATE,
        "num_attention_heads": DEFAULT_HEADS,
        "num_key_value_heads": DEFAULT_KV_HEADS,
        "head_dim": DEFAULT_HEAD_DIM,
        "vocab_size": int(vocab_size),
        "sliding_window": DEFAULT_SLIDING_WINDOW,
        "max_position_embeddings": int(max_position_embeddings),
        "tie_word_embeddings": True,
        "use_cache": False,
        "hidden_act": "gelu",
        "attention_bias": False,
        "mlp_bias": False,
        "headwise_attn_output_gate": True,
        "gate_attn_act_mode": "sigmoid",
        "layer_types": layer_types,
        "rope_parameters": {
            "sliding_attention": {
                "partial_rotary_factor": 1.0, "rope_theta": 10000},
            "full_attention": {
                "partial_rotary_factor": 1.0, "rope_theta": 10000},
        },
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "transformers_version": "4.57.1",
    }


class ContextualCausalAttention(nn.Module):
    """Bias-free Q/K/V/O attention with an explicit causal mask.

    Every position mixes only earlier positions within the same row.  The
    sliding-attention variant additionally caps the visible distance.  This is
    the property the old pointwise fixture lacks.
    """

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, *,
                 layer_type: str, sliding_window: int) -> None:
        super().__init__()
        if heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        self.hidden = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.kv_repeat = heads // kv_heads
        self.layer_type = layer_type
        self.sliding_window = int(sliding_window)
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False,
                                dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False,
                                dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False,
                                dtype=torch.bfloat16)
        self.out_proj = nn.Linear(heads * head_dim, hidden, bias=False,
                                  dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.kv_repeat != 1:
            key = key.repeat_interleave(self.kv_repeat, dim=1)
            value = value.repeat_interleave(self.kv_repeat, dim=1)
        scores = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal = torch.ones(length, length, dtype=torch.bool,
                            device=hidden_states.device).tril()
        if self.layer_type == "sliding_attention":
            index = torch.arange(length, device=hidden_states.device)
            distance = index[:, None] - index[None, :]
            causal = causal & (distance < self.sliding_window)
        if attention_mask is not None:
            causal = causal & attention_mask.to(torch.bool)[:, None, None, :]
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        mixed = (weights @ value).transpose(1, 2).reshape(
            batch, length, self.heads * self.head_dim)
        return self.out_proj(mixed)


class ContextualBlock(nn.Module):
    """Pre-norm Spark-shaped block: causal attention then GELU MLP."""

    def __init__(self, hidden: int, intermediate: int, heads: int,
                 kv_heads: int, head_dim: int, *, layer_type: str,
                 sliding_window: int) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.attention_norm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
        self.self_attn = ContextualCausalAttention(
            hidden, heads, kv_heads, head_dim, layer_type=layer_type,
            sliding_window=sliding_window)
        self.mlp_norm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False,
                                   dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False,
                                 dtype=torch.bfloat16)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False,
                                   dtype=torch.bfloat16)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                position_embeddings=None, past_key_values=None,
                cache_position=None, position_ids=None, **kwargs):
        hidden_states = hidden_states + self.self_attn(
            self.attention_norm(hidden_states), attention_mask)
        normed = self.mlp_norm(hidden_states)
        mlp = self.activation(self.gate_proj(normed)) * self.up_proj(normed)
        return hidden_states + self.down_proj(mlp)


class ContextualSparkModel(nn.Module):
    """Tiny Spark2_5 module layout: embedding, layers, final norm."""

    def __init__(self, config) -> None:
        super().__init__()
        geometry = config.geometry
        self.embedding = nn.Embedding(
            geometry.vocab_size, geometry.hidden_size, padding_idx=2,
            dtype=torch.bfloat16)
        self.layers = nn.ModuleList([
            ContextualBlock(
                geometry.hidden_size, geometry.intermediate_size,
                geometry.num_attention_heads, geometry.num_key_value_heads,
                geometry.head_dim,
                layer_type=geometry.layer_types[index],
                sliding_window=geometry.sliding_window)
            for index in range(geometry.num_hidden_layers)])
        self.norm = nn.LayerNorm(geometry.hidden_size, dtype=torch.bfloat16)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                use_cache=False, **kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask,
                           position_ids=position_ids, use_cache=False)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class ContextualSparkCausalLM(nn.Module):
    """Causal LM with Spark's tied input/output embedding table."""

    def __init__(self, config) -> None:
        super().__init__()
        geometry = config.geometry
        self.config = SimpleNamespace(
            model_type="spark2_5", use_cache=False,
            _attn_implementation="eager",
            hidden_size=geometry.hidden_size,
            num_hidden_layers=geometry.num_hidden_layers,
            vocab_size=geometry.vocab_size)
        self.model = ContextualSparkModel(config)
        self.lm_head = nn.Linear(
            geometry.hidden_size, geometry.vocab_size, bias=False,
            dtype=torch.bfloat16)
        self.lm_head.weight = self.model.embedding.weight

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                use_cache=False, **kwargs):
        hidden = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False)
        logits = F.linear(hidden.last_hidden_state, self.model.embedding.weight)
        return SimpleNamespace(logits=logits)

    def get_input_embeddings(self):
        return self.model.embedding

    def get_output_embeddings(self):
        return self.lm_head


def build_contextual_adapter(
        seed: int = 7, *, layers: int = DEFAULT_LAYERS,
        spine_sites: tuple[int, ...] = (3, 7), spine_slots: int = 4,
        memory_size: int = 8, heads: int = 2, device: str = "cpu",
        vocab_size: int = DEFAULT_VOCAB, max_position_embeddings: int = 64):
    """Build the contextual transparent fixture and its frozen-base adapter."""
    if max(spine_sites) >= layers:
        raise ValueError("spine site is outside the contextual decoder")
    torch.manual_seed(seed)
    config = load_spark2_5_config(contextual_config_raw(
        layers=layers, vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings),
                                  require_official_geometry=False)
    base = ContextualSparkCausalLM(config)
    precision = official_bf16_precision(
        config, artifact_sha256=sha256_bytes(b"contextual-spark-fixture"))
    adapter = Spark2_5Adapter(
        base, config, precision, device=device, require_official_geometry=False,
        projector_layer=3, memory_size=memory_size, heads=heads,
        spine_sites=tuple(int(site) for site in spine_sites),
        spine_slots=int(spine_slots))
    adapter.seal_original_identity()
    return adapter, base, config, precision


def build_contextual_path(
        seed: int = 7, *, layers: int = DEFAULT_LAYERS,
        spine_sites: tuple[int, ...] = (3, 7), spine_slots: int = 4,
        read_records: int = 3, bank_experts: int = 1,
        bank_hidden_width: int = 8, bank_lr: float = 1e-3,
        response_lr: float = 1e-3, device: str = "cpu",
        vocab_size: int = DEFAULT_VOCAB, max_position_embeddings: int = 64):
    """One CPU SparkTextMemoryPath over the contextual fixture."""
    adapter, _, _, _ = build_contextual_adapter(
        seed=seed, layers=layers, spine_sites=spine_sites,
        spine_slots=spine_slots, device=device, vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings)
    experiment = SpineMemoryExperiment(
        spec=adapter.spine_spec, core=adapter.extension.spine,
        bank_experts=bank_experts, bank_hidden_width=bank_hidden_width,
        seed=seed, read_records=read_records,
        scope="contextual-spark-fixture")
    # The bank is created with its historical default learning rate; assign a
    # fixture-local rate before the lazily-created optimizer is first used.
    experiment.bank.lr = float(bank_lr)
    return SparkTextMemoryPath(adapter, experiment, response_lr=response_lr)


def open_memory_ports(path: SparkTextMemoryPath, *, seed: int = 0,
                      scale: float = 0.02) -> None:
    """Open zero-initialized memory-port outputs without touching the base."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for port in path.experiment.core.ports["memory"].values():
            port.output_projection.weight.normal_(0.0, scale, generator=generator)


def ids_and_mask(sequence: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([[3, 5, 7, 9, 11, 13][:sequence]], dtype=torch.long)
    return ids, torch.ones_like(ids)


def prefix_ids() -> tuple[torch.Tensor, torch.Tensor]:
    """Two prompts with different prefixes but the same final token."""
    left = torch.tensor([[3, 5, 7, 9, 11, 13]], dtype=torch.long)
    right = torch.tensor([[17, 19, 21, 23, 11, 13]], dtype=torch.long)
    return left, right


def labels_from_ids(ids: torch.Tensor, ignore_first: int = 2) -> torch.Tensor:
    labels = ids.clone()
    labels[:, :ignore_first] = -100
    return labels


def populate_contextual_evidence(
        path: SparkTextMemoryPath,
        records: list[tuple[str, str]] | tuple[tuple[str, str], ...],
        *, order: str = "chronological") -> dict[str, int]:
    """Admit bounded exact text records with a declared slot order.

    ``order='reversed'`` assigns source epochs backwards so the same record set
    is selected in a different evidence-slot order.  This is the fixture-level
    shuffled-memory control.
    """
    if order not in {"chronological", "reversed"}:
        raise ValueError("order must be chronological or reversed")
    count = len(records)
    epochs = {}
    for index, (record_id, text) in enumerate(records):
        source_epoch = index if order == "chronological" else count - 1 - index
        epochs[record_id] = source_epoch
        path.observe_text(
            text, record_id=record_id, source_epoch=source_epoch,
            available_at=0, valid_from=0,
            source_lineage={"source": "contextual-fixture",
                            "fixture_order": order})
    return epochs


def token_mixing_probe(adapter: Spark2_5Adapter) -> dict:
    """Return same-final-token prefix logits for the causality assertion."""
    left, right = prefix_ids()
    mask = torch.ones_like(left)
    with torch.no_grad():
        left_logits = adapter.forward_text(
            left, mask, use_projector=False, use_ports=False, use_spine=False)
        right_logits = adapter.forward_text(
            right, mask, use_projector=False, use_ports=False, use_spine=False)
    delta = (left_logits[:, -1] - right_logits[:, -1]).abs()
    return {
        "same_last_token": bool(torch.equal(left[:, -1], right[:, -1])),
        "different_prefix": bool(not torch.equal(left[:, :-1], right[:, :-1])),
        "last_position_max_abs_delta": float(delta.max()),
        "last_position_equal": bool(torch.equal(
            left_logits[:, -1], right_logits[:, -1])),
        "fixture": ENGINEERING_LABEL,
        "attention": "causal_explicit",
        "mlp_activation": "gelu",
        "tied_output": True,
    }


__all__ = [
    "ENGINEERING_LABEL", "NATIVE_SEALED_LAYERS", "LATE_SITE_MIN_LAYERS",
    "DEFAULT_LAYERS", "ContextualCausalAttention", "ContextualBlock",
    "ContextualSparkModel", "ContextualSparkCausalLM",
    "build_contextual_adapter", "build_contextual_path", "contextual_config_raw",
    "ids_and_mask", "labels_from_ids", "open_memory_ports", "populate_contextual_evidence",
    "prefix_ids", "token_mixing_probe",
]
