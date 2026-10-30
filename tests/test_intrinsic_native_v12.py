"""v12 producer-interface tests: source accounting, order-once, declared bounds.

No weights, no native engine, no sealed packet. These tests exercise the pieces
that must be right *before* an expensive attempt: reservation amounts and
refusals, the single order addition, and agreement between the prepared
declaration and the producer's own contract.
"""
import types
from contextlib import contextmanager

import pytest
import torch

from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.contextual_source_features_v12 import (
    SOURCE_FORWARD, SOURCE_TOKENIZE, FrozenSourceFeatures, SourceFeatureSpec,
    SourceWorkAccounting as SourceWorkAccountingProtocol, SourceWorkRequest,
    tokenizer_contract_digest)
from replica_lab.intrinsic_memory_coverage_v12 import (
    FEATURE_CACHE_BYTES, FEATURE_CACHE_ENTRIES, SOURCE_MAX_BYTES, SOURCE_MAX_TOKENS,
    declared_tokenizer_identity, tokenizer_identity_digest)
from replica_lab.intrinsic_memory_diagnostic_v12 import (
    DiagnosticEngineV12, SourceWorkAccounting, StageRef, right_padded_prefix)


class FakeLedger:
    def __init__(self):
        self.reserved, self.finished = [], []

    def reserve(self, stage, operation, **amounts):
        self.reserved.append((stage, operation, amounts))
        return len(self.reserved)-1

    def finish(self, token, *, status='completed', actual=None, detail=None):
        self.finished.append((token, status, actual, detail))


def accounting(stage='fit_128'):
    ledger = FakeLedger(); ref = StageRef(); ref.name = stage
    return SourceWorkAccounting(ledger, ref), ledger, ref


def test_accounting_satisfies_the_producer_protocol():
    adapter, _, _ = accounting()
    assert isinstance(adapter, SourceWorkAccountingProtocol)


def test_tokenizer_reservation_precedes_work_and_has_no_token_count():
    adapter, ledger, _ = accounting()
    token = adapter.reserve(SourceWorkRequest(SOURCE_TOKENIZE, source_bytes=120,
                                              max_tokens=SOURCE_MAX_TOKENS))
    assert token == 0
    stage, operation, amounts = ledger.reserved[0]
    assert stage == 'fit_128' and operation == 'source_tokenize'
    assert amounts == {'source_encodes': 1, 'tokenizer_calls': 1,
                       'source_token_bytes': 120, 'input_tokens': 120}
    assert 'forwards' not in amounts and 'cache_misses' not in amounts
    # The producer's request type accepts a token count, but a tokenizer
    # reservation that already knows it could only be made after tokenization,
    # so the adapter refuses it before any ledger write.
    late = SourceWorkRequest(SOURCE_TOKENIZE, source_bytes=1,
                             max_tokens=SOURCE_MAX_TOKENS, tokens=4)
    with pytest.raises(ContextualMemoryError, match='must precede tokenization'):
        adapter.reserve(late)
    assert len(ledger.reserved) == 1


def test_forward_reservation_carries_the_full_v12_vocabulary():
    adapter, ledger, _ = accounting()
    adapter.reserve(SourceWorkRequest(SOURCE_FORWARD, source_bytes=120,
                                      max_tokens=SOURCE_MAX_TOKENS, tokens=37))
    _, operation, amounts = ledger.reserved[0]
    assert operation == 'source_forward'
    assert amounts == {'forwards': 1, 'source_feature_forwards': 1, 'input_tokens': 37,
                       'cache_misses': 1, 'cache_evictions': 1, 'cache_recomputations': 1}


def test_budget_refusal_happens_before_any_dispatch():
    adapter, ledger, ref = accounting()
    with pytest.raises(ContextualMemoryError, match='outside a declared stage'):
        ref.name = None
        adapter.reserve(SourceWorkRequest(SOURCE_FORWARD, source_bytes=1,
                                          max_tokens=SOURCE_MAX_TOKENS, tokens=1))
    assert ledger.reserved == []
    ref.name = 'fit_128'
    with pytest.raises(ContextualMemoryError, match='source byte bound'):
        adapter.reserve(SourceWorkRequest(SOURCE_TOKENIZE, source_bytes=SOURCE_MAX_BYTES+1,
                                          max_tokens=SOURCE_MAX_TOKENS))
    with pytest.raises(ContextualMemoryError, match='token bound'):
        adapter.reserve(SourceWorkRequest(SOURCE_TOKENIZE, source_bytes=1, max_tokens=1024))
    with pytest.raises(ContextualMemoryError, match='exact bounded token count'):
        adapter.reserve(SourceWorkRequest(SOURCE_FORWARD, source_bytes=1,
                                          max_tokens=SOURCE_MAX_TOKENS, tokens=300))
    assert ledger.reserved == []


def test_completion_records_observations_without_refund():
    adapter, ledger, _ = accounting()
    token = adapter.reserve(SourceWorkRequest(SOURCE_FORWARD, source_bytes=1,
                                              max_tokens=SOURCE_MAX_TOKENS, tokens=9))
    adapter.complete(token, status='failed', actual={'forwards': 0},
                     detail={'stage': 'source_forward'})
    assert ledger.finished == [(token, 'failed', {'forwards': 0},
                                {'stage': 'source_forward'})]
    with pytest.raises(ContextualMemoryError, match='invalid source work status'):
        adapter.complete(token, status='retried')


class FakeContext:
    values = torch.zeros(1, 1, 8, 256)


class FakeReader:
    def __init__(self, log): self.log = log

    def snapshot(self, values, availability, *, validate):
        self.log.append('reader_snapshot')
        validate()
        return FakeContext()


class FakePath:
    """Records the call shape of the two forward routes without a real model."""

    def __init__(self):
        self.log = []
        self.spec = types.SimpleNamespace(slots=8)
        self.reader = FakeReader(self.log)
        self.cost = {'snapshot_copies': 0, 'snapshot_bytes': 0}
        self.release_corpus = None
        self.adapter = types.SimpleNamespace(
            forward_text=self._forward_text,
            base=types.SimpleNamespace(parameters=lambda: iter([torch.zeros(1)])))

    def order(self, index):
        self.log.append('order')
        return torch.zeros(len(index), 256)

    def validate(self, batch, scope): self.log.append('validate')

    def forward(self, ids, mask, batch, *, scope, enabled=True):
        self.log.append('path_forward')
        assert enabled is (batch is not None)
        return torch.zeros(1, 4, 4)

    def _forward_text(self, ids, mask, **kwargs):
        self.log.append('forward_text')
        return torch.zeros(1, 4, 4)


def engine(tmp_path):
    path = FakePath()
    built = DiagnosticEngineV12(path, None, types.SimpleNamespace(), None,
                                max_new_tokens=8, output=tmp_path)
    batch = types.SimpleNamespace(slots=torch.zeros(4, 256), spans=tuple(), store=None)
    ids = torch.zeros(1, 4, dtype=torch.long); mask = torch.ones_like(ids)
    return built, path, batch, ids, mask, 'scope'


def test_contextual_route_uses_path_forward_and_never_adds_order(tmp_path):
    built, path, batch, ids, mask, scope = engine(tmp_path)
    built._forward(ids, mask, batch, scope, control='contextual')
    assert path.log == ['path_forward']


def test_intervention_routes_add_order_exactly_once(tmp_path):
    for control in ('shuffled', 'zero'):
        built, path, batch, ids, mask, scope = engine(tmp_path)
        built._forward(ids, mask, batch, scope, control=control)
        assert path.log.count('order') == 1, control
        assert path.log.count('reader_snapshot') == 1
        assert 'path_forward' not in path.log
        assert path.cost['snapshot_copies'] == 1


def test_disabled_routes_never_build_a_snapshot_or_add_order(tmp_path):
    built, path, batch, ids, mask, scope = engine(tmp_path)
    built._forward(ids, mask, batch, scope, control='absent')
    built._forward(ids, mask, None, scope, control='contextual')
    assert path.log == ['path_forward', 'path_forward']
    with pytest.raises(ValueError, match='undeclared control'):
        built._forward(ids, mask, batch, scope, control='oracle')


def test_wrong_control_requires_its_own_validated_source(tmp_path):
    built, path, batch, ids, mask, scope = engine(tmp_path)
    with pytest.raises(ValueError, match='wrong control source missing'):
        built._forward(ids, mask, batch, scope, control='wrong')
    # The refused call validated the recorded batch before discovering that the
    # substituted source is missing; that authority check is required, so it
    # counts once even though no intervention ran.
    assert path.log.count('validate') == 1
    assert path.log.count('order') == 0
    wrong = types.SimpleNamespace(slots=torch.ones(4, 256), spans=tuple(), store=None)
    built._forward(ids, mask, batch, scope, control='wrong', wrong=wrong,
                   wrong_scope='other')
    assert path.log.count('order') == 1
    # Four more: the recorded batch and the substituted batch at the intervention
    # boundary, then both again at the decoder injection boundary. No check is
    # skipped or duplicated by the engine.
    assert path.log.count('validate') == 5


def test_declared_feature_bounds_are_accepted_by_the_producer_contract():
    base = dict(original_parameters_sha256='a'*64, source_execution_sha256='b'*64,
                tokenizer_sha256='c'*64, hidden_size=2560,
                max_tokens=SOURCE_MAX_TOKENS, cache_bytes=FEATURE_CACHE_BYTES,
                cache_entries=FEATURE_CACHE_ENTRIES)
    identity = SourceFeatureSpec(**base).identity()
    assert identity['format'] == 'replica.canonical_source_features.v12'
    assert identity['input_policy'] == 'canonical_span_only_no_template_no_special_tokens'
    assert identity['output_policy'] == 'last_decoder_hidden_after_final_norm'
    assert identity['original_updates'] == 0
    assert identity['source_position_ids'] == '0_to_source_token_count_minus_one'
    for damage in ({'max_tokens': 512}, {'max_tokens': 0},
                   {'cache_bytes': 64*1024**2}, {'cache_entries': 65},
                   {'input_policy': 'chat_template'},
                   {'output_policy': 'pooled_mean'}):
        with pytest.raises(ContextualMemoryError):
            SourceFeatureSpec(**{**base, **damage})


def test_provider_requires_explicit_accounting():
    adapter = types.SimpleNamespace(
        config=types.SimpleNamespace(geometry=types.SimpleNamespace(hidden_size=2560)))
    spec = SourceFeatureSpec(original_parameters_sha256='a'*64,
                             source_execution_sha256='b'*64,
                             tokenizer_sha256='c'*64, hidden_size=2560)
    with pytest.raises(ContextualMemoryError, match='explicit source work accounting'):
        FrozenSourceFeatures(adapter, None, spec, accounting=None)


def test_adapter_contract_binds_exact_content_without_rehashing():
    from replica_lab.contextual_source_features_v12 import (
        adapter_content_identity, adapter_contract_digest)

    def fake(content):
        base = types.SimpleNamespace(
            training=False,
            config=types.SimpleNamespace(use_cache=False),
            parameters=lambda: iter(()))
        return types.SimpleNamespace(
            base=base, attention_implementation='eager', regime='modules_only',
            config=types.SimpleNamespace(
                geometry=types.SimpleNamespace(to_dict=lambda: {'hidden_size': 2560})),
            base_snapshot=lambda: {'config_digest': 'x', 'geometry_fingerprint': 'y'},
            v12_adapter_content_sha256=content)

    first, second = fake('a'*64), fake('b'*64)
    assert adapter_content_identity(first) == 'a'*64
    assert adapter_contract_digest(first) != adapter_contract_digest(second)
    # A supplied digest is reused verbatim, so the backbone is not hashed again;
    # an explicit override wins and a malformed digest is refused.
    assert adapter_contract_digest(first) == adapter_contract_digest(
        first, content_identity='a'*64)
    with pytest.raises(ContextualMemoryError):
        adapter_contract_digest(fake('not-a-digest'))


class RecordingLedger:
    """Records reservation amounts through the engine's measured() context."""

    def __init__(self):
        self.reserved = []

    @contextmanager
    def work(self, stage, operation, **amounts):
        record = {'stage': stage, 'operation': operation, 'reserved': dict(amounts),
                  'detail': {}}
        self.reserved.append(record)
        yield record['detail']

    def reserve(self, stage, operation, **amounts):
        self.reserved.append({'stage': stage, 'operation': operation,
                              'reserved': dict(amounts), 'detail': {}})
        return len(self.reserved)-1

    def finish(self, token, **kwargs):
        return None


class ReadPath(FakePath):
    def __init__(self, batches):
        super().__init__()
        self.batches, self.reads, self.resumes = list(batches), 0, 0

    def read(self, store, question, *, scope, wait_seconds=0.0):
        self.log.append('read')
        self.reads += 1
        return self.batches.pop(0)

    def resume(self, store, pending, *, scope, wait_seconds=0.0):
        self.log.append('resume')
        self.resumes += 1
        return self.batches.pop(0)


def read_engine(tmp_path, batches):
    from replica_lab.intrinsic_memory_diagnostic_v12 import StageRef as Ref
    ledger = RecordingLedger(); path = ReadPath(batches); ref = Ref(); ref.name = 'prepare'
    built = DiagnosticEngineV12(path, None, types.SimpleNamespace(context_tokens=256),
                                ledger, max_new_tokens=8, output=tmp_path, stage_ref=ref)
    return built, ledger, path


def test_successful_read_reserves_the_exact_descriptor_bound(tmp_path):
    batch = types.SimpleNamespace(read_costs={'cold_wait_ns': 0})
    built, ledger, path = read_engine(tmp_path, [batch])
    assert built.read(None, 'scope', 'question') is batch
    retrieval = [r for r in ledger.reserved if r['operation'] == 'retrieval']
    assert len(retrieval) == 1
    # Three persists is the exact enumerated maximum for one selection (success 3,
    # timeout 2, failed dispatch 3); one selection, no resume.
    assert retrieval[0]['reserved'] == {'retrievals': 1, 'writes': 3}
    assert [r for r in ledger.reserved if r['operation'] == 'contextual_resume'] == []
    assert path.resumes == 0


def test_each_explicit_resume_is_charged_before_it_dispatches(tmp_path):
    from replica_lab.contextual_memory_v5 import PendingContextualRead
    pending = PendingContextualRead('a'*32, '/tmp/store', {'scope': 1}, 'b'*64, {})
    batch = types.SimpleNamespace(read_costs={'cold_wait_ns': 1})
    built, ledger, path = read_engine(tmp_path, [pending, batch])
    assert built.read(None, 'scope', 'question') is batch
    operations = [r['operation'] for r in ledger.reserved]
    assert operations == ['retrieval', 'contextual_resume']
    assert ledger.reserved[1]['reserved'] == {'writes': 2}
    assert path.resumes == 1


def test_unresolved_pending_read_refuses_after_five_waits(tmp_path):
    from replica_lab.contextual_memory_v5 import PendingContextualRead
    pending = PendingContextualRead('a'*32, '/tmp/store', {'scope': 1}, 'b'*64, {})
    built, ledger, path = read_engine(tmp_path, [pending]*5)
    with pytest.raises(ContextualMemoryError, match='five bounded waits'):
        built.read(None, 'scope', 'question')
    # One selection and four bounded resumes, each charged before dispatch.
    assert path.reads == 1 and path.resumes == 4
    assert [r['operation'] for r in ledger.reserved].count('contextual_resume') == 4


def test_tokenizer_wrapper_reproduces_the_declared_digest():
    from replica_lab.intrinsic_memory_native_v12 import BoundSourceTokenizer
    identity = declared_tokenizer_identity(
        [{'path': f'x/{name}', 'sha256': str(index)*64}
         for index, name in enumerate(
             ('chat_template.jinja', 'merges.txt', 'special_tokens_map.json',
              'tokenizer.json', 'tokenizer_config.json', 'vocab.json'))])

    class Real:
        def __call__(self, text, **kwargs): return {'input_ids': [1, 2, 3]}
        def decode(self, ids): return 'decoded'

    wrapped = BoundSourceTokenizer(Real(), identity)
    assert wrapped.identity() == identity
    assert wrapped('anything') == {'input_ids': [1, 2, 3]}
    assert wrapped.decode([1]) == 'decoded'
    assert tokenizer_contract_digest(wrapped) == tokenizer_identity_digest(identity)
    with pytest.raises(ValueError, match='identity'):
        BoundSourceTokenizer(Real(), {})
    # A changed declared mapping can never reproduce the previous digest.
    changed = dict(identity); changed['vocab_size'] = 1
    assert tokenizer_identity_digest(changed) != tokenizer_identity_digest(identity)


def test_context_width_and_prefix_bounds_match_the_envelope():
    spec = types.SimpleNamespace(context_tokens=256, pad_token_id=2)
    ids, mask = right_padded_prefix([5, 6, 7], spec, torch.device('cpu'))
    assert tuple(ids.shape) == (1, 256) and int(mask.sum()) == 3
    with pytest.raises(ValueError, match='truncate'):
        right_padded_prefix(list(range(257)), spec, torch.device('cpu'))
