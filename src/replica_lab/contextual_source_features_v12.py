"""Frozen canonical-source features with a live trainable four-slot compressor.

The source backbone is never a child module or optimizer parameter of the
producer. Its no-grad feature boundary is intentional; the compressor, ordered
pooling and reader remain differentiable. The bounded RAM feature cache is C,
not persistent learning, and is never serialized with an extension checkpoint.

Every source pass is admitted through an explicit ``SourceWorkAccounting``
provider *before* tokenizer work and *before* the frozen forward. There is no
implicit in-process counter-only fallback: a provider that cannot reserve work
is refused. ``WorkLedgerSourceAccounting`` maps the interface onto the durable
``replica_lab.intrinsic_memory_budget_v5.WorkLedger`` used by native packets.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import time
from typing import Mapping, Protocol, runtime_checkable

import torch
from torch import nn

from .contextual_memory import CanonicalSpan, ContextualMemoryError, SLOTS, WIDTH, tensor_digest
from .model_identity import durable_model_identity

PRODUCER = "frozen_spark_features_contextual_compressor_v12"
REPRESENTATION = "four_ordered_frozen_source_contextual_256_v12"
CODEC = "contextual_source_features_fp32_slots_v12"
FEATURE_FORMAT = "replica.canonical_source_features.v12"

SOURCE_TOKENIZE = "source_tokenize"
SOURCE_FORWARD = "source_forward"
SOURCE_OPERATIONS = (SOURCE_TOKENIZE, SOURCE_FORWARD)


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _digest64(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ContextualMemoryError(f"complete {name} identity required")
    return value


def tokenizer_contract_digest(tokenizer):
    """Digest of the tokenizer's declared immutable identity, never of its state.

    A native tokenizer wrapper supplies vocab/config file hashes through
    ``identity()``. A mutable tokenizer without an explicit identity is refused
    because a vocabulary substitution cannot then be detected.
    """
    identity = getattr(tokenizer, "identity", None)
    if not callable(identity):
        raise ContextualMemoryError("tokenizer must expose an immutable identity() mapping")
    try:
        value = identity()
    except Exception as error:  # noqa: BLE001 - surfaced as a contract refusal
        raise ContextualMemoryError("tokenizer identity() failed") from error
    if not isinstance(value, Mapping) or not value:
        raise ContextualMemoryError("tokenizer identity must be a nonempty mapping")
    try:
        payload = json.loads(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as error:
        raise ContextualMemoryError("tokenizer identity is not canonical JSON") from error
    return identity_digest({"format": FEATURE_FORMAT + ".tokenizer", "identity": payload})


def adapter_content_identity(adapter, *, content_identity=None):
    """Exact frozen-adapter content digest used to bind substitutions.

    A native operator that already verified the source artifact may attach
    ``adapter.v12_adapter_content_sha256`` (or pass ``content_identity``) so the
    provider does not hash the backbone again. Otherwise the tiny/custom
    backbone is hashed once, tensor by tensor, with bounded temporary storage.
    """
    supplied = content_identity or getattr(adapter, "v12_adapter_content_sha256", None)
    if supplied is not None:
        return _digest64(supplied, "adapter content")
    return durable_model_identity(adapter.base, config={
        "format": FEATURE_FORMAT + ".adapter_content",
        "geometry": adapter.config.geometry.to_dict(),
    })


def adapter_contract_digest(adapter, *, content_identity=None):
    """Structural identity of the exact frozen adapter the provider is bound to.

    Weight content is bound through an exact content digest; a different
    adapter with identical geometry is refused. Full-original hashes remain a
    native admission requirement and may be supplied as the content identity.
    """
    try:
        payload = {
            "format": FEATURE_FORMAT + ".adapter",
            "base_snapshot": adapter.base_snapshot(),
            "geometry": adapter.config.geometry.to_dict(),
            "attention_implementation": getattr(adapter, "attention_implementation", None),
            "use_cache": bool(getattr(adapter.base.config, "use_cache", False)),
            "regime": getattr(adapter, "regime", None),
            "base_content_sha256": adapter_content_identity(
                adapter, content_identity=content_identity),
        }
    except AttributeError as error:
        raise ContextualMemoryError("adapter does not expose a complete source contract") from error
    return identity_digest(payload)


@dataclass(frozen=True)
class SourceFeatureSpec:
    """Immutable feature contract; operational cache bounds are not part of it."""

    original_parameters_sha256: str
    source_execution_sha256: str
    tokenizer_sha256: str
    hidden_size: int
    max_tokens: int = 256
    cache_bytes: int = 32 * 1024**2
    cache_entries: int = 64
    input_policy: str = "canonical_span_only_no_template_no_special_tokens"
    output_policy: str = "last_decoder_hidden_after_final_norm"

    def __post_init__(self):
        _digest64(self.original_parameters_sha256, "original parameters")
        _digest64(self.source_execution_sha256, "source execution")
        _digest64(self.tokenizer_sha256, "tokenizer")
        if (type(self.hidden_size) is not int or self.hidden_size < 1
                or type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 256
                or type(self.cache_bytes) is not int or not 0 <= self.cache_bytes <= 32 * 1024**2
                or type(self.cache_entries) is not int or not 0 <= self.cache_entries <= 64
                or self.input_policy != "canonical_span_only_no_template_no_special_tokens"
                or self.output_policy != "last_decoder_hidden_after_final_norm"):
            raise ContextualMemoryError("invalid source-feature contract or bounds")

    def identity(self):
        return {
            "format": FEATURE_FORMAT,
            "original_parameters_sha256": self.original_parameters_sha256,
            "source_execution_sha256": self.source_execution_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "hidden_size": self.hidden_size,
            "max_tokens": self.max_tokens,
            "input_policy": self.input_policy,
            "output_policy": self.output_policy,
            "original_updates": 0,
            "detach_boundary": "frozen_original_hidden_before_trainable_compressor",
            "source_position_ids": "0_to_source_token_count_minus_one",
            "thinking_mode": "not_applicable_unframed_source",
        }


@dataclass(frozen=True)
class SourceWorkRequest:
    """One pre-dispatch reservation request for source-feature work."""

    operation: str
    source_bytes: int
    max_tokens: int
    tokens: int | None = None

    def __post_init__(self):
        if self.operation not in SOURCE_OPERATIONS:
            raise ContextualMemoryError("unknown source work operation")
        if type(self.source_bytes) is not int or self.source_bytes < 0:
            raise ContextualMemoryError("invalid source byte reservation")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ContextualMemoryError("invalid source token reservation bound")
        if self.tokens is not None and (type(self.tokens) is not int or self.tokens < 0):
            raise ContextualMemoryError("invalid source token count")
        if self.operation == SOURCE_FORWARD and self.tokens is None:
            raise ContextualMemoryError("a source forward requires the exact token count")


@runtime_checkable
class SourceWorkAccounting(Protocol):
    """Explicit admission provider; reservations happen before the work."""

    def reserve(self, request: SourceWorkRequest) -> object:
        ...

    def complete(self, token, *, status: str = "completed",
                 actual: Mapping[str, int] | None = None,
                 detail: Mapping[str, object] | None = None) -> None:
        ...


def source_work_amounts(request):
    """Ledger counters for one source-work reservation (public mapping)."""
    if not isinstance(request, SourceWorkRequest):
        raise ContextualMemoryError("SourceWorkRequest required")
    if request.operation == SOURCE_TOKENIZE:
        if request.tokens is not None:
            raise ContextualMemoryError("tokenizer reservation must precede tokenization")
        return {"source_encodes": 1, "input_tokens": request.source_bytes}
    return {"forwards": 1, "input_tokens": request.tokens}


class WorkLedgerSourceAccounting:
    """Map source work onto the durable v5 work ledger, charged before dispatch.

    ``ledger`` is duck-typed (``reserve(stage, operation, **amounts)`` and
    ``finish(token, status=..., actual=..., detail=...)``). One tokenizer pass is
    charged as ``source_encodes=1`` with an upper-bound input-token reservation;
    a cache-missing frozen forward adds ``forwards=1`` and its exact tokens.
    """

    def __init__(self, ledger, stage):
        if not callable(getattr(ledger, "reserve", None)) or not callable(getattr(ledger, "finish", None)):
            raise ContextualMemoryError("explicit work ledger with reserve/finish required")
        if not isinstance(stage, str) or not stage:
            raise ContextualMemoryError("work-ledger stage required")
        self.ledger, self.stage = ledger, stage
        self.reserved = {operation: 0 for operation in SOURCE_OPERATIONS}

    def reserve(self, request):
        amounts = source_work_amounts(request)
        self.reserved[request.operation] += 1
        return self.ledger.reserve(self.stage, request.operation, **amounts)

    def complete(self, token, *, status="completed", actual=None, detail=None):
        if status not in ("completed", "failed"):
            raise ContextualMemoryError("invalid source work status")
        return self.ledger.finish(token, status=status, actual=dict(actual or {}),
                                   detail=dict(detail or {}))


class FrozenSourceFeatures:
    """One bounded source pass, with no memory, external spine or generation.

    A native caller must supply the already verified original/code/tokenizer
    hashes. Runtime mutation stamps additionally reject normal in-place changes;
    they do not replace the experiment's full original hashes before/after work.
    Cache hits never grant source authority: the contextual path validates that
    authority before every training/read boundary and decoder injection.
    """

    def __init__(self, adapter, tokenizer, spec, *, accounting):
        if not isinstance(spec, SourceFeatureSpec):
            raise ContextualMemoryError("declared source-feature contract required")
        if spec.hidden_size != adapter.config.geometry.hidden_size:
            raise ContextualMemoryError("source-feature geometry differs")
        if accounting is None:
            raise ContextualMemoryError(
                "explicit source work accounting is required before any source forward")
        if not isinstance(accounting, SourceWorkAccounting):
            raise ContextualMemoryError("source work accounting must implement reserve/complete")
        self.adapter, self.tokenizer, self.spec, self.accounting = adapter, tokenizer, spec, accounting
        if getattr(adapter, "attention_implementation", None) != "eager":
            raise ContextualMemoryError("source passes require the audited eager attention path")
        if bool(getattr(adapter.base.config, "use_cache", True)):
            raise ContextualMemoryError("source passes require an uncached frozen backbone")
        if adapter.base.training or any(p.requires_grad for p in adapter.base.parameters()):
            raise ContextualMemoryError("source backbone must be frozen and in evaluation mode")
        self._adapter_content = adapter_content_identity(adapter)
        self._adapter_contract = adapter_contract_digest(
            adapter, content_identity=self._adapter_content)
        if self._adapter_contract != spec.source_execution_sha256:
            raise ContextualMemoryError("source execution contract differs from the bound adapter")
        self._tokenizer_contract = tokenizer_contract_digest(tokenizer)
        if self._tokenizer_contract != spec.tokenizer_sha256:
            raise ContextualMemoryError("tokenizer contract differs from the declared identity")
        self._base_snapshot = adapter.base_snapshot()
        self._stamp = self._original_stamp()
        self._cache = OrderedDict()
        self._tokens = OrderedDict()
        self.cost = {key: 0 for key in (
            "requests", "tokenizer_calls", "tokenizer_cache_hits", "source_bytes", "forwards",
            "input_tokens", "cache_hits", "cache_misses", "cache_evictions", "cache_bytes",
            "peak_cache_bytes", "feature_bytes_produced", "source_traced_layers", "wall_ns")}

    # ------------------------------------------------------------------
    # Identity and authority

    @property
    def adapter_contract(self):
        return self._adapter_contract

    @property
    def tokenizer_contract(self):
        return self._tokenizer_contract

    def identity(self):
        return {"feature_contract": self.spec.identity(),
                "adapter_contract": self._adapter_contract,
                "tokenizer_contract": self._tokenizer_contract,
                "original_updates": 0}

    def contract_digest(self):
        """Digest of the *current* feature/adapter/tokenizer contract.

        Recomputed from the bound objects so a swapped tokenizer or adapter
        cannot be hidden by a producer parameter-mutation stamp.
        """
        return identity_digest({
            "feature_contract": self.spec.identity(),
            "adapter_contract": adapter_contract_digest(
                self.adapter, content_identity=self._adapter_content),
            "tokenizer_contract": tokenizer_contract_digest(self.tokenizer),
        })

    def _original_stamp(self):
        return tuple((name, p._version, p.data_ptr(), str(p.dtype), str(p.device), tuple(p.shape))
                     for name, p in self.adapter.base.named_parameters())

    def validate_original(self):
        if (self.adapter.base.training or any(p.requires_grad for p in self.adapter.base.parameters())):
            self.clear()
            raise ContextualMemoryError("frozen source backbone changed; cached features invalidated")
        if (self._original_stamp() != self._stamp
                or self.adapter.base_snapshot() != self._base_snapshot
                or adapter_contract_digest(self.adapter,
                                           content_identity=self._adapter_content)
                    != self._adapter_contract):
            self.clear()
            raise ContextualMemoryError("frozen source backbone or adapter contract changed")
        if tokenizer_contract_digest(self.tokenizer) != self._tokenizer_contract:
            self.clear()
            raise ContextualMemoryError("tokenizer contract changed; cached features invalidated")
        return True

    # ------------------------------------------------------------------
    # Bounded caches (C, never a serialized learning artifact)

    def clear(self):
        self._cache.clear()
        self._tokens.clear()
        self.cost["cache_bytes"] = 0

    def evict_record(self, record_id):
        removed_count = 0
        removed = 0
        for key in list(self._cache):
            name, value = self._cache[key]
            if name == record_id:
                removed += value.numel() * value.element_size()
                del self._cache[key]
                removed_count += 1
        if removed:
            self.cost["cache_bytes"] -= removed
            self.cost["cache_evictions"] += 1
        for key in list(self._tokens):
            if key[0] == record_id:
                del self._tokens[key]
        return removed_count

    def _store_tokens(self, key, ids):
        if not self.spec.cache_entries:
            return
        self._tokens[key] = tuple(ids)
        self._tokens.move_to_end(key)
        while len(self._tokens) > self.spec.cache_entries:
            self._tokens.popitem(last=False)

    # ------------------------------------------------------------------
    # Source tokenization and frozen features

    def token_ids(self, span):
        """Token sequence for one canonical span, charged before tokenizer work."""
        if not isinstance(span, CanonicalSpan):
            raise ContextualMemoryError("canonical span is the only source-feature input")
        span.validate()
        key = (span.record_id, span.canonical_sha256, span.start, span.end,
               span.source_epoch, span.source_order)
        if key in self._tokens:
            self.cost["tokenizer_cache_hits"] += 1
            return list(self._tokens[key])
        raw = span.text.encode()
        token = self.accounting.reserve(SourceWorkRequest(
            SOURCE_TOKENIZE, source_bytes=len(raw), max_tokens=self.spec.max_tokens))
        try:
            encoded = self.tokenizer(span.text, add_special_tokens=False, truncation=False)
        except BaseException as error:
            self._complete(token, status="failed", detail={"stage": "tokenizer",
                                                          "error": type(error).__name__})
            raise
        self.cost["tokenizer_calls"] += 1
        self.cost["source_bytes"] += len(raw)
        ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        vocab = self.adapter.config.geometry.vocab_size
        if (not isinstance(ids, list) or not 0 < len(ids) <= self.spec.max_tokens
                or any(type(index) is not int or not 0 <= index < vocab for index in ids)):
            self._complete(token, status="failed", detail={"stage": "tokenizer",
                                                          "reason": "invalid_sequence"})
            raise ContextualMemoryError("empty, truncated, oversized or invalid canonical token sequence")
        self._complete(token, actual={"source_encodes": 1, "input_tokens": len(ids)},
                       detail={"stage": "tokenizer", "tokens": len(ids)})
        self._store_tokens(key, ids)
        return ids

    def _complete(self, token, *, status="completed", actual=None, detail=None):
        try:
            self.accounting.complete(token, status=status, actual=actual, detail=detail)
        except ContextualMemoryError:
            raise
        except Exception as error:  # noqa: BLE001 - accounting failures are contract refusals
            raise ContextualMemoryError("source work accounting rejected the completion") from error

    def provenance(self, span):
        ids = self.token_ids(span)
        count = len(ids)
        regions = []
        for slot in range(SLOTS):
            begin = min(count - 1, slot * count // SLOTS)
            end = max(begin + 1, (slot + 1) * count // SLOTS)
            regions.append([begin, min(end, count)])
        # The feature-contract digest transitively binds the declared original
        # parameter digest, source execution/adapter contract and tokenizer
        # contract. Keeping one digest here fits the store's bounded lineage
        # field without weakening the durable binding.
        return {"feature_contract_sha256": identity_digest(self.spec.identity()),
                "original_parameters_sha256": self.spec.original_parameters_sha256,
                "source_token_ids_sha256": identity_digest(ids),
                "source_token_count": count,
                "pool_token_regions": regions}

    def features(self, span):
        """Frozen FP32 hidden states for one canonical span, detached by contract."""
        self.validate_original()
        ids = self.token_ids(span)
        key = identity_digest({"contract": self.spec.identity(), "span": asdict(span), "tokens": ids})
        self.cost["requests"] += 1
        self.cost["source_bytes"] += len(span.text.encode())
        if key in self._cache:
            self.cost["cache_hits"] += 1
            self._cache.move_to_end(key)
            # A caller cannot mutate the one cached authority-independent feature tensor.
            return self._cache[key][1].clone()
        self.cost["cache_misses"] += 1
        device = next(self.adapter.base.parameters()).device
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        position = torch.arange(len(ids), device=device)[None]
        token = self.accounting.reserve(SourceWorkRequest(
            SOURCE_FORWARD, source_bytes=len(span.text.encode()),
            max_tokens=self.spec.max_tokens, tokens=len(ids)))
        start = time.monotonic_ns()
        try:
            # One intentional no_grad boundary. Extension branches are disabled
            # by arguments and the adapter's own injection audit is checked below.
            with torch.no_grad():
                hidden, audit = self.adapter.forward_hidden(
                    tokens, torch.ones_like(tokens), position_ids=position,
                    use_projector=False, use_ports=False, use_spine=False,
                    intrinsic_memory=None, return_audit=True)
            values = hidden[0].detach().float()
        except BaseException as error:
            self._complete(token, status="failed", detail={"stage": "source_forward",
                                                          "error": type(error).__name__})
            raise
        self.cost["forwards"] += 1
        self.cost["input_tokens"] += len(ids)
        self.cost["wall_ns"] += time.monotonic_ns() - start
        if (tuple(audit.injected_layers) or audit.projector_input_connected
                or tuple(audit.traced_layers) != tuple(range(len(self.adapter.base.model.layers)))):
            self._complete(token, status="failed", detail={"stage": "source_forward",
                                                          "reason": "extension_branch_executed"})
            raise ContextualMemoryError("source pass executed a disabled extension branch")
        self.cost["source_traced_layers"] = len(audit.traced_layers)
        if (tuple(values.shape) != (len(ids), self.spec.hidden_size)
                or not torch.isfinite(values).all()):
            self._complete(token, status="failed", detail={"stage": "source_forward",
                                                          "reason": "invalid_features"})
            raise ContextualMemoryError("invalid frozen canonical features")
        self._complete(token, actual={"forwards": 1, "input_tokens": len(ids)},
                       detail={"stage": "source_forward", "tokens": len(ids)})
        size = values.numel() * values.element_size()
        self.cost["feature_bytes_produced"] += size
        if size <= self.spec.cache_bytes and self.spec.cache_entries:
            while self._cache and (self.cost["cache_bytes"] + size > self.spec.cache_bytes
                                   or len(self._cache) >= self.spec.cache_entries):
                _, (_, removed) = self._cache.popitem(last=False)
                self.cost["cache_bytes"] -= removed.numel() * removed.element_size()
                self.cost["cache_evictions"] += 1
            self._cache[key] = (span.record_id, values.clone())
            self.cost["cache_bytes"] += size
            self.cost["peak_cache_bytes"] = max(self.cost["peak_cache_bytes"], self.cost["cache_bytes"])
        return values


class SourceFeatureValueEncoder(nn.Module):
    """Live trainable compression of detached frozen source hidden states."""

    def __init__(self, provider):
        super().__init__()
        if not isinstance(provider, FrozenSourceFeatures):
            raise ContextualMemoryError("explicit frozen source-feature provider required")
        # Plain object: the original model is not registered here and cannot
        # enter the optimizer or the extension checkpoint through an alias.
        self.provider = provider
        self.input_norm = nn.LayerNorm(provider.spec.hidden_size)
        self.input_projection = nn.Linear(provider.spec.hidden_size, WIDTH, bias=False)
        self.position = nn.Embedding(provider.spec.max_tokens, WIDTH)
        layer = nn.TransformerEncoderLayer(WIDTH, 4, 512, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH, bias=False))
        self.slot_queries = nn.Parameter(torch.empty(SLOTS, WIDTH))
        nn.init.normal_(self.slot_queries, std=0.02)

    def forward(self, spans):
        if not spans:
            raise ContextualMemoryError("source encoder requires canonical spans")
        values = []
        for span in spans:
            frozen = self.provider.features(span).to(self.input_projection.weight.device)
            # Only the frozen feature boundary detaches. Every operation below
            # stays live through response loss; durable slots are never a graph.
            hidden = self.input_projection(self.input_norm(frozen))
            if len(hidden) > self.position.num_embeddings:
                raise ContextualMemoryError("source sequence exceeds the compressor position bound")
            hidden = hidden + self.position(torch.arange(len(hidden), device=hidden.device))
            hidden = self.transformer(hidden[None])[0]
            slots = []
            for slot in range(SLOTS):
                begin = min(len(hidden) - 1, slot * len(hidden) // SLOTS)
                end = max(begin + 1, (slot + 1) * len(hidden) // SLOTS)
                region = hidden[begin:end]
                weights = torch.softmax(region @ self.slot_queries[slot] / WIDTH**0.5, dim=0)
                slots.append(self.output((weights[:, None] * region).sum(0)))
            values.append(torch.stack(slots))
        return torch.stack(values)

    def release_from_state(self, state):
        digest = hashlib.sha256(self.provider.contract_digest().encode())
        for name in self.state_dict():
            if name not in state:
                raise ContextualMemoryError("producer state inventory differs")
            digest.update(name.encode())
            digest.update(tensor_digest(state[name]).encode())
        return "v12-" + digest.hexdigest()

    def release(self):
        return self.release_from_state(self.state_dict())
