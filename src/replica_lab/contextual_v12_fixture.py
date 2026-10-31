"""Tiny random 36-block BF16 contextual-v12 fixture; CPU engineering only.

Nothing here is a native checkpoint, a competence claim or evidence of useful
recall. The fixture supplies explicit synthetic tokenization, an explicit
in-process source-work accounting provider and TieredMemoryStore construction
so the real v12 producer/store/reader path can be exercised without a download.
"""
from __future__ import annotations

import hashlib
import json

import torch

from .architecture.intrinsic_memory_layout_v4 import IntrinsicMemoryLayout
from .contextual_memory import CanonicalSpan, ContextualMemoryError, canonical_spans
from .contextual_memory_v12 import ContextualMemoryV12
from .contextual_source_features_v12 import (
    FEATURE_FORMAT, SOURCE_FORWARD, SOURCE_TOKENIZE, SourceFeatureSpec,
    SourceWorkRequest, FrozenSourceFeatures, adapter_content_identity,
    adapter_contract_digest, identity_digest, source_work_amounts,
    tokenizer_contract_digest)
from .contextual_spark_fixture import build_contextual_adapter
from .tiered_memory import MemoryScope, TierCeilings, TieredMemoryStore

ENGINEERING_LABEL = "engineering_cpu_v12_contextual_fixture_only"
DEFAULT_VOCAB = 128
DEFAULT_MAX_TOKENS = 32


class SyntheticTokenizer:
    """Whitespace/alphanumeric synthetic tokenizer with an immutable identity.

    ``add_special_tokens`` and ``truncation`` must stay disabled: the canonical
    source pass is template-, question-, answer- and special-token-free.
    """

    def __init__(self, vocab_size=DEFAULT_VOCAB, *, mode="alnum"):
        if type(vocab_size) is not int or vocab_size < 8:
            raise ContextualMemoryError("synthetic tokenizer requires a bounded vocabulary")
        if mode not in ("alnum", "words"):
            raise ContextualMemoryError("unknown synthetic tokenizer mode")
        self.vocab_size, self.mode = vocab_size, mode
        self.calls = 0

    def _token_id(self, word):
        raw = hashlib.blake2b(word.encode(), digest_size=8).digest()
        return 1 + int.from_bytes(raw, "little") % (self.vocab_size - 1)

    def __call__(self, text, add_special_tokens=False, truncation=False):
        if add_special_tokens is not False:
            raise ContextualMemoryError("source passes forbid special tokens")
        if truncation is not False:
            raise ContextualMemoryError("source passes forbid silent truncation")
        if not isinstance(text, str):
            raise ContextualMemoryError("tokenizer input must be canonical text")
        self.calls += 1
        tokens = text.split()
        if self.mode == "alnum":
            tokens = [token for token in tokens if any(character.isalnum() for character in token)]
        return {"input_ids": [self._token_id(token.casefold()) for token in tokens]}

    def identity(self):
        return {"format": FEATURE_FORMAT + ".synthetic_tokenizer", "version": 1,
                "vocab_size": self.vocab_size, "mode": self.mode,
                "policy": "whitespace_tokens_no_special_no_truncation"}


class SyntheticWorkAccounting:
    """In-process source work admission with declared caps, charged pre-dispatch."""

    UNITS = ("source_encodes", "forwards", "input_tokens")

    def __init__(self, caps=None):
        self.caps = {unit: 10**7 for unit in self.UNITS}
        for unit, value in dict(caps or {}).items():
            if unit not in self.UNITS or type(value) is not int or value < 0:
                raise ContextualMemoryError("invalid synthetic work cap")
            self.caps[unit] = value
        self.reserved = {unit: 0 for unit in self.UNITS}
        self.events = []

    def reserve(self, request):
        if not isinstance(request, SourceWorkRequest):
            raise ContextualMemoryError("SourceWorkRequest required")
        amounts = source_work_amounts(request)
        for unit, value in amounts.items():
            if self.reserved[unit] + value > self.caps[unit]:
                raise ContextualMemoryError("source work budget exceeded before dispatch")
        for unit, value in amounts.items():
            self.reserved[unit] += value
        token = len(self.events)
        self.events.append({"operation": request.operation, "reserved": dict(amounts),
                            "status": "reserved"})
        return token

    def complete(self, token, *, status="completed", actual=None, detail=None):
        if status not in ("completed", "failed"):
            raise ContextualMemoryError("invalid source work status")
        event = self.events[token]
        if event["status"] != "reserved":
            raise ContextualMemoryError("source work reservation already terminal")
        actual = dict(actual or {})
        for unit, value in actual.items():
            if unit not in event["reserved"] or type(value) is not int or value < 0:
                raise ContextualMemoryError("invalid actual source work")
            if value > event["reserved"][unit]:
                raise ContextualMemoryError("actual source work exceeds its reservation")
        event.update(status=status, actual=actual, detail=dict(detail or {}))
        return True


def parameter_digest(module):
    """Deterministic content digest for a tiny fixture's declared originals."""
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def logits_digest(logits):
    return hashlib.sha256(
        logits.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def optimizer_digest(optimizer):
    """Canonical digest of owned parameters' optimizer moments, by declared name."""
    digest = hashlib.sha256()
    for group in optimizer.param_groups:
        for name, parameter in zip(group["param_names"], group["params"]):
            digest.update(name.encode())
            state = optimizer.state.get(parameter)
            if not state:
                digest.update(b"nostate")
                continue
            for key in sorted(state):
                value = state[key]
                digest.update(key.encode())
                if torch.is_tensor(value):
                    digest.update(value.detach().cpu().contiguous().reshape(-1)
                                  .view(torch.uint8).numpy().tobytes())
                else:
                    digest.update(str(value).encode())
    return digest.hexdigest()


def model_state_digest(path):
    digest = hashlib.sha256()
    for name, value in path._tensors().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1)
                      .view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_feature_spec(adapter, tokenizer, *, original_parameters_sha256=None,
                        max_tokens=DEFAULT_MAX_TOKENS, cache_bytes=4 * 1024**2,
                        cache_entries=16):
    if original_parameters_sha256 is None:
        # Declared fixture original identity. A native run must replace this
        # with the verified full-original digest; the provider only binds and
        # propagates it, it does not hash a 4B backbone implicitly.
        original_parameters_sha256 = parameter_digest(adapter.base)
    content = adapter_content_identity(adapter)
    # Explicit operator attestation: avoids re-hashing the adapter on repeated
    # provider construction and makes a substitute adapter fail the contract.
    adapter.v12_adapter_content_sha256 = content
    return SourceFeatureSpec(
        original_parameters_sha256=original_parameters_sha256,
        source_execution_sha256=adapter_contract_digest(
            adapter, content_identity=content),
        tokenizer_sha256=tokenizer_contract_digest(tokenizer),
        hidden_size=adapter.config.geometry.hidden_size,
        max_tokens=max_tokens, cache_bytes=cache_bytes, cache_entries=cache_entries)


def build_v12_stack(*, seed=17, layers=36, sites=(3, 7), slots=8, vocab_size=DEFAULT_VOCAB,
                    lr=1e-3, initialization_seed=17, max_tokens=DEFAULT_MAX_TOKENS,
                    cache_bytes=4 * 1024**2, cache_entries=16, accounting=None,
                    tokenizer=None, tokenizer_mode="alnum", pre_buffers=(),
                    source_identity="cpu_v12_tiny_bf16_fixture",
                    execution_identity="CPU_v12_fixture",
                    placement="block_input_query_to_post_block_residual",
                    objective_identity=None, key_mode="ordered"):
    """Build the real v12 producer/store path over a tiny random BF16 backbone."""
    sites = tuple(int(site) for site in sites)
    adapter, _, config, _ = build_contextual_adapter(
        seed=seed, layers=layers, spine_sites=sites, spine_slots=slots,
        memory_size=256, heads=2, device="cpu", vocab_size=vocab_size,
        max_position_embeddings=64)
    adapter.eval()
    for parameter in adapter.base.parameters():
        parameter.requires_grad_(False)
    existing_buffers = {name for name, _ in adapter.base.named_buffers()}
    for name, value in pre_buffers:
        if not isinstance(name, str) or not name or name in existing_buffers:
            raise ContextualMemoryError("invalid or duplicate pre-existing fixture buffer")
        adapter.base.register_buffer(name, torch.as_tensor(value).detach().clone())
        existing_buffers.add(name)
    tokenizer = tokenizer or SyntheticTokenizer(config.geometry.vocab_size,
                                                mode=tokenizer_mode)
    if getattr(tokenizer, "vocab_size", None) is not None \
            and tokenizer.vocab_size != config.geometry.vocab_size:
        raise ContextualMemoryError("tokenizer vocabulary differs from the fixture backbone")
    spec = source_feature_spec(adapter, tokenizer, max_tokens=max_tokens,
                               cache_bytes=cache_bytes, cache_entries=cache_entries)
    accounting = accounting or SyntheticWorkAccounting()
    provider = FrozenSourceFeatures(adapter, tokenizer, spec, accounting=accounting)
    layout = IntrinsicMemoryLayout(hidden_size=config.geometry.hidden_size,
                                   decoder_layers=layers, sites=sites,
                                   width=256, slots=slots, placement=placement)
    path = ContextualMemoryV12(
        adapter, layout, source_identity=source_identity, feature_provider=provider,
        lr=lr, initialization_seed=initialization_seed, key_mode=key_mode,
        objective_identity=objective_identity or {"scope": "CPU_v12_correctness_only"},
        execution_identity=execution_identity)
    return adapter, layout, provider, path


def open_v12_store(root, *, name="store", ceilings=None):
    return TieredMemoryStore(
        root / name, profile_id="P1", enforce_volume_free=False,
        ceilings=ceilings or TierCeilings(
            hot_bytes=32 * 1024**2, warm_bytes=32 * 1024**2, cold_bytes=32 * 1024**2,
            identity_reserved_bytes=65536, replay_reserved_bytes=262144,
            workspace_quota_bytes=1048576, artifact_budget_bytes=4 * 1048576,
            minimum_volume_free_bytes=0))


def scope(subject_id="user_a", *, purpose="engineering", as_of=1000):
    return MemoryScope(subject_id=subject_id, branch_id="main", purpose=purpose, as_of=as_of)


def canonical_span(text, *, record_id, source_id="fixture_source", source_epoch=1,
                   available_at=1, valid_until=None):
    if not text:
        raise ContextualMemoryError("canonical source text must be nonempty")
    return canonical_spans(text, record_id=record_id, source_id=source_id,
                           source_epoch=source_epoch, available_at=available_at,
                           valid_until=valid_until)


def spans_for(texts, *, source_epoch=1, available_at=1):
    result = []
    for index, text in enumerate(texts):
        result.append(canonical_span(text, record_id=f"record{index}",
                                     source_epoch=source_epoch,
                                     available_at=available_at)[0])
    return tuple(result)


__all__ = [
    "CanonicalSpan", "ContextualMemoryError", "ENGINEERING_LABEL",
    "FrozenSourceFeatures", "SyntheticTokenizer", "SyntheticWorkAccounting",
    "build_v12_stack", "canonical_span", "canonical_spans", "identity_digest",
    "logits_digest", "model_state_digest", "open_v12_store", "optimizer_digest",
    "parameter_digest", "scope", "source_feature_spec", "spans_for",
    "tokenizer_contract_digest", "SOURCE_FORWARD", "SOURCE_TOKENIZE",
]
