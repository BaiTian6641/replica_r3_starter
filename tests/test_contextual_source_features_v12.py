"""v12 frozen-source producer contracts: canonical text, binding, accounting, cache."""
from dataclasses import replace
import json

import pytest
import torch

from replica_lab.contextual_source_features_v12 import (
    FrozenSourceFeatures, SourceFeatureSpec, SourceWorkRequest, WorkLedgerSourceAccounting,
    adapter_contract_digest, identity_digest, source_work_amounts,
    tokenizer_contract_digest)
from replica_lab.contextual_spark_fixture import build_contextual_adapter
from replica_lab.contextual_v12_fixture import (
    ContextualMemoryError, SyntheticTokenizer, SyntheticWorkAccounting, build_v12_stack,
    canonical_span, scope)


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class RecordingTokenizer(SyntheticTokenizer):
    def __init__(self, vocab_size, *, mode="alnum"):
        super().__init__(vocab_size, mode=mode)
        self.observed = []

    def __call__(self, text, add_special_tokens=False, truncation=False):
        self.observed.append((text, add_special_tokens, truncation))
        return super().__call__(text, add_special_tokens=add_special_tokens,
                                truncation=truncation)


class RecordingLedger:
    def __init__(self):
        self.calls = []
        self.events = {}

    def reserve(self, stage, operation, **amounts):
        token = len(self.calls)
        self.calls.append(("reserve", stage, operation, dict(amounts)))
        self.events[token] = {"reserved": dict(amounts), "status": "reserved"}
        return token

    def finish(self, token, *, status="completed", actual=None, detail=None):
        event = self.events[token]
        assert event["status"] == "reserved"
        for unit, value in dict(actual or {}).items():
            assert unit in event["reserved"] and value <= event["reserved"][unit]
        event.update(status=status, actual=dict(actual or {}))
        self.calls.append(("finish", status))
        return True


def stack(**kwargs):
    return build_v12_stack(layers=36, sites=(3, 7), **kwargs)


def test_spec_and_provider_refuse_adapter_tokenizer_and_accounting_substitution():
    adapter, _, _, path = stack()
    provider = path.feature_provider
    spec, tokenizer = provider.spec, provider.tokenizer
    with pytest.raises(ContextualMemoryError):
        FrozenSourceFeatures(adapter, tokenizer, spec, accounting=None)
    other_tokenizer = SyntheticTokenizer(adapter.config.geometry.vocab_size, mode="words")
    with pytest.raises(ContextualMemoryError, match="tokenizer contract"):
        FrozenSourceFeatures(adapter, other_tokenizer, spec,
                             accounting=SyntheticWorkAccounting())
    with pytest.raises(ContextualMemoryError, match="source execution contract"):
        FrozenSourceFeatures(adapter, tokenizer, replace(spec, source_execution_sha256="0" * 64),
                             accounting=SyntheticWorkAccounting())
    with pytest.raises(ContextualMemoryError, match="source-feature geometry"):
        FrozenSourceFeatures(adapter, tokenizer, replace(spec, hidden_size=spec.hidden_size + 1),
                             accounting=SyntheticWorkAccounting())
    adapter2, _, _, _ = build_contextual_adapter(
        seed=18, layers=36, spine_sites=(3, 7), spine_slots=8, memory_size=256,
        vocab_size=adapter.config.geometry.vocab_size)
    adapter2.eval()
    with pytest.raises(ContextualMemoryError, match="source execution contract"):
        FrozenSourceFeatures(adapter2, tokenizer, spec, accounting=SyntheticWorkAccounting())
    assert adapter_contract_digest(adapter) == spec.source_execution_sha256
    assert tokenizer_contract_digest(tokenizer) == spec.tokenizer_sha256


@pytest.mark.parametrize("mutation", ["training", "requires_grad", "use_cache"])
def test_provider_refuses_a_mutable_or_cached_source_backbone(mutation):
    adapter, _, _, path = stack()
    if mutation == "training":
        adapter.base.train()
    elif mutation == "requires_grad":
        next(adapter.base.parameters()).requires_grad_(True)
    else:
        adapter.base.config.use_cache = True
    with pytest.raises(ContextualMemoryError):
        FrozenSourceFeatures(adapter, path.feature_provider.tokenizer,
                             path.feature_provider.spec, accounting=SyntheticWorkAccounting())


def test_source_text_is_canonical_only_without_special_tokens_or_truncation():
    adapter, _, _, path = stack()
    tokenizer = RecordingTokenizer(adapter.config.geometry.vocab_size)
    adapter, _, _, path = build_v12_stack(layers=36, sites=(3, 7), tokenizer=tokenizer)
    span = canonical_span("the badge is red", record_id="record0")[0]
    path.training_values((span,), scope=scope())
    assert tokenizer.observed
    assert all(text == span.text for text, _, _ in tokenizer.observed)
    assert all(special is False and truncate is False
               for _, special, truncate in tokenizer.observed)
    with pytest.raises(ContextualMemoryError):
        tokenizer(span.text, add_special_tokens=True)
    with pytest.raises(ContextualMemoryError):
        tokenizer(span.text, truncation=True)


def test_empty_and_oversize_token_sequences_are_refused():
    adapter, _, _, path = stack()
    provider = path.feature_provider
    with pytest.raises(ContextualMemoryError):
        provider.token_ids(canonical_span("...", record_id="punctuation")[0])
    _, _, _, small = stack(max_tokens=4)
    words = " ".join(["word"] * 12)
    span = canonical_span(words, record_id="long")[0]
    with pytest.raises(ContextualMemoryError):
        small.feature_provider.token_ids(span)
    with pytest.raises(ContextualMemoryError):
        SourceWorkRequest("source_forward", source_bytes=1, max_tokens=4, tokens=None)


def test_work_ledger_adapter_reserves_before_dispatch_and_reports_actual():
    adapter, _, _, path = stack()
    provider = path.feature_provider
    ledger = RecordingLedger()
    provider.accounting = WorkLedgerSourceAccounting(ledger, "mechanics")
    span = canonical_span("the badge is red", record_id="record0")[0]
    observed = []
    original = adapter.forward_hidden

    def wrapped(*args, **kwargs):
        observed.append([call[2] for call in ledger.calls if call[0] == "reserve"])
        return original(*args, **kwargs)

    adapter.forward_hidden = wrapped
    values = provider.features(span)
    adapter.forward_hidden = original
    assert values.shape[1] == provider.spec.hidden_size
    assert observed and observed[0][-1] == "source_forward"
    operations = [call[2] for call in ledger.calls if call[0] == "reserve"]
    assert operations == ["source_tokenize", "source_forward"]
    for token, event in ledger.events.items():
        assert event["status"] == "completed"
    amounts = [call[3] for call in ledger.calls if call[0] == "reserve"]
    assert amounts[0]["source_encodes"] == 1
    assert amounts[0]["input_tokens"] == len(span.text.encode())
    assert amounts[1]["forwards"] == 1
    assert amounts[1]["input_tokens"] == provider.cost["input_tokens"]


def test_budget_admission_fails_before_tokenizer_or_source_forward():
    adapter, _, _, path = stack()
    provider = path.feature_provider
    provider.accounting = SyntheticWorkAccounting(caps={"source_encodes": 0})
    tokenizer = RecordingTokenizer(adapter.config.geometry.vocab_size)
    provider.tokenizer = tokenizer
    span = canonical_span("the badge is red", record_id="record0")[0]
    with pytest.raises(ContextualMemoryError, match="budget"):
        provider.features(span)
    assert tokenizer.observed == []
    assert provider.cost["forwards"] == 0

    _, _, _, second = stack()
    second.feature_provider.accounting = SyntheticWorkAccounting(caps={"forwards": 0})
    with pytest.raises(ContextualMemoryError, match="budget"):
        second.feature_provider.features(span)
    assert second.feature_provider.cost["forwards"] == 0


def test_source_work_amounts_mapping_is_explicit():
    tokenize = SourceWorkRequest("source_tokenize", source_bytes=17, max_tokens=32)
    forward = SourceWorkRequest("source_forward", source_bytes=17, max_tokens=32, tokens=4)
    assert source_work_amounts(tokenize) == {"source_encodes": 1, "input_tokens": 17}
    assert source_work_amounts(forward) == {"forwards": 1, "input_tokens": 4}


def test_bounded_feature_cache_hits_clone_and_evict_without_source_authority():
    adapter, _, _, path = stack(cache_entries=1)
    provider = path.feature_provider
    left = canonical_span("the badge is red", record_id="left")[0]
    right = canonical_span("the badge is blue", record_id="right")[0]
    first = provider.features(left)
    first.add_(1000.0)  # caller mutation must not poison the one cached tensor
    forwards = provider.cost["forwards"]
    second = provider.features(left)
    assert provider.cost["forwards"] == forwards
    assert provider.cost["cache_hits"] == 1
    assert torch.allclose(second, provider.features(left))
    provider.features(right)
    assert provider.cost["cache_evictions"] >= 1
    assert len(provider._cache) <= 1
    assert provider.cost["cache_bytes"] <= provider.spec.cache_bytes
    assert provider.evict_record("right.span0") == 1
    assert provider.cost["cache_bytes"] == 0
    assert provider.evict_record("unknown.span0") == 0
    provider.clear()
    assert provider.cost["cache_bytes"] == 0 and provider._cache == {} and provider._tokens == {}


def test_zero_entry_cache_is_not_retained():
    adapter, _, _, path = stack(cache_entries=0)
    provider = path.feature_provider
    span = canonical_span("the badge is red", record_id="record0")[0]
    provider.features(span)
    assert provider._cache == {} and provider.cost["cache_bytes"] == 0
    provider.features(span)
    assert provider.cost["forwards"] == 2


def test_source_extraction_is_extension_free_and_detached():
    adapter, _, _, path = stack()
    provider = path.feature_provider
    for name, port in list(path.reader.ports.items()):
        path.reader.ports[name].forward = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reader executed during a source pass"))
    span = canonical_span("the badge is red", record_id="record0")[0]
    features = provider.features(span)
    assert not features.requires_grad
    assert provider.cost["source_traced_layers"] == 36
    values = path.encoder((span,))
    assert values.requires_grad and values.shape == (1, 4, 256)


def test_provenance_binds_source_positions_contracts_and_fits_lineage_bound():
    adapter, _, _, path = stack()
    provider = path.feature_provider
    span = canonical_span("the badge is red and blue", record_id="record0",
                          source_epoch=3, available_at=2, valid_until=50)[0]
    provenance = provider.provenance(span)
    assert provenance["feature_contract_sha256"] == identity_digest(provider.spec.identity())
    assert provenance["original_parameters_sha256"] == provider.spec.original_parameters_sha256
    assert provenance["source_token_count"] == 6
    regions = provenance["pool_token_regions"]
    assert regions[0][0] == 0 and regions[-1][1] == 6
    assert all(left < right for left, right in regions)
    assert len(json.dumps(provenance, sort_keys=True)) <= 1024
    assert provenance["source_token_ids_sha256"] == identity_digest(provider.token_ids(span))
    full = path.provenance(span, "v12-" + "a" * 64)
    assert len(json.dumps(full, sort_keys=True)) <= 1024
    assert full["feature_contract_sha256"] == provenance["feature_contract_sha256"]


def test_validate_original_invalidates_cache_on_mutation_and_tokenizer_swap():
    adapter, _, _, path = stack()
    provider = path.feature_provider
    span = canonical_span("the badge is red", record_id="record0")[0]
    provider.features(span)
    assert provider.cost["cache_bytes"] > 0
    with torch.no_grad():
        next(adapter.base.parameters()).add_(0.5)
    with pytest.raises(ContextualMemoryError):
        provider.validate_original()
    assert provider.cost["cache_bytes"] == 0

    adapter, _, _, path = stack()
    provider = path.feature_provider
    provider.features(span)
    provider.tokenizer = SyntheticTokenizer(adapter.config.geometry.vocab_size, mode="words")
    with pytest.raises(ContextualMemoryError):
        provider.validate_original()
    assert provider.cost["cache_bytes"] == 0


def test_spec_bounds_and_identity_exclude_operational_cache_sizes():
    adapter, _, _, path = stack()
    spec = path.feature_provider.spec
    baseline = spec.identity()
    for field, value in (("max_tokens", 0), ("max_tokens", 257), ("cache_bytes", -1),
                         ("cache_entries", 65), ("input_policy", "template"),
                         ("output_policy", "logits")):
        with pytest.raises(ContextualMemoryError):
            replace(spec, **{field: value})
    assert replace(spec, cache_bytes=1).identity() == baseline
    with pytest.raises(ContextualMemoryError):
        replace(spec, original_parameters_sha256="xy")
