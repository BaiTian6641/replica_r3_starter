"""v12 producer-interface tests: source accounting, order-once, declared bounds.

No weights, no native engine, no sealed packet. These tests exercise the pieces
that must be right *before* an expensive attempt: reservation amounts and
refusals, the single order addition, and agreement between the prepared
declaration and the producer's own contract.
"""
import hashlib
import json
import threading
import time
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
    ENVIRONMENT_CLOSURE_BYTES, FEATURE_CACHE_BYTES, FEATURE_CACHE_ENTRIES, FIT_CASES,
    READ_WRITE_BOUND, RELEASE_READ_BYTES_BOUND, RESTART_PROMPT_PASSES, SOURCE_FILE_BYTES,
    SOURCE_MAX_BYTES, release_direct_read_bound,
    SOURCE_MAX_TOKENS, TOKENIZER_IDENTITY_FILES, declared_tokenizer_identity,
    plan as coverage_plan, tokenizer_identity_digest)
from replica_lab.intrinsic_memory_budget_v12 import BudgetError, WorkLedger
from replica_lab.intrinsic_memory_diagnostic_v12 import (
    DiagnosticEngineV12, SourceWorkAccounting, StageRef, right_padded_prefix)
from replica_lab.intrinsic_memory_native_v12 import begin_stage, close_stage


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
        self.feature_provider = types.SimpleNamespace(cost={'requests': 0, 'forwards': 0,
                                                            'cache_hits': 0})
        self.adapter = types.SimpleNamespace(
            forward_text=self._forward_text,
            base=types.SimpleNamespace(parameters=lambda: iter([torch.zeros(1)])))

    def order(self, index):
        self.log.append('order')
        return torch.zeros(len(index), 256)

    def freeze_readout(self): self.log.append('freeze_readout')
    def resume_training(self): self.log.append('resume_training')

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
    assert retrieval[0]['reserved'] == {'retrievals': 1, 'writes': READ_WRITE_BOUND,
                                        'artifact_read_bytes': RELEASE_READ_BYTES_BOUND}
    assert retrieval[0]['detail']['actual']['writes'] == 3   # one read, no resume
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
    assert ledger.reserved[1]['reserved'] == {'writes': 0}   # timing only; bound in the read
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
    assert ledger.reserved[0]['reserved']['writes'] == READ_WRITE_BOUND


def test_tokenizer_identity_is_derived_from_the_loaded_tokenizer(tmp_path):
    from replica_lab.intrinsic_memory_native_v12 import (
        BoundSourceTokenizer, observed_tokenizer_identity)

    directory = tmp_path/'source'; directory.mkdir()
    for name in TOKENIZER_IDENTITY_FILES:
        (directory/name).write_bytes(('content:'+name).encode())
    declared = {
        'files': {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                  for name in TOKENIZER_IDENTITY_FILES},
        'vocab_size': 8, 'eos_token_id': 1, 'pad_token_id': 2, 'is_fast': True,
        'add_special_tokens_for_source': False, 'truncation_for_source': False,
        'local_files_only': True}

    class Real:
        def __init__(self, *, origin=None, vocab_size=8, fast=True):
            self.name_or_path = str(origin or directory)
            self.vocab_size, self.eos_token_id, self.pad_token_id = vocab_size, 1, 2
            self.is_fast = fast
        def get_vocab(self): return {str(i): i for i in range(self.vocab_size)}
        def __call__(self, text, **kwargs): return {'input_ids': [1, 2, 3]}
        def decode(self, ids): return 'decoded'

    loaded = Real()
    assert observed_tokenizer_identity(loaded, directory) == declared
    wrapped = BoundSourceTokenizer(loaded, directory, declared)
    assert wrapped.identity() == declared
    assert tokenizer_contract_digest(wrapped) == tokenizer_identity_digest(declared)
    assert wrapped('anything') == {'input_ids': [1, 2, 3]}
    assert wrapped.decode([1]) == 'decoded'

    # A substituted file byte, vocabulary size, fast-tokenizer property or
    # directory of origin is refused before any source work.
    (directory/'vocab.json').write_bytes(b'substituted')
    with pytest.raises(ValueError, match='differs from the loaded tokenizer'):
        BoundSourceTokenizer(loaded, directory, declared)
    (directory/'vocab.json').write_bytes(('content:vocab.json').encode())
    with pytest.raises(ValueError, match='differs from the loaded tokenizer'):
        BoundSourceTokenizer(Real(vocab_size=9), directory, declared)
    with pytest.raises(ValueError, match='differs from the loaded tokenizer'):
        BoundSourceTokenizer(Real(fast=False), directory, declared)
    with pytest.raises(ValueError, match='verified source directory'):
        BoundSourceTokenizer(Real(origin=tmp_path), directory, declared)
    with pytest.raises(ValueError, match='declared tokenizer identity'):
        BoundSourceTokenizer(loaded, directory, {})
    changed = json.loads(json.dumps(declared)); changed['vocab_size'] = 1
    assert tokenizer_identity_digest(changed) != tokenizer_identity_digest(declared)


def test_restart_plan_reserves_the_derived_prompt_tokenization():
    # 2 encodings for each of 32 restored targets, plus canary, update student
    # and the exact-text binding that alignment re-derives.
    assert RESTART_PROMPT_PASSES == 2*FIT_CASES + 3
    stage = coverage_plan('two')['restart']
    assert stage['prompt_tokenizer_calls'] == RESTART_PROMPT_PASSES
    assert stage['prompt_tokenizer_calls'] >= 2*FIT_CASES + 3


def test_restart_stage_accepts_exactly_the_derived_prompt_passes(tmp_path):
    """Real ledger consumption: the derived cap is usable and refuses the next."""
    from replica_lab.intrinsic_memory_coverage_v12 import plan as stage_plan
    ledger = WorkLedger.create(tmp_path/'restart.jsonl', identity='packet-sha',
                               plan={'restart': stage_plan('two')['restart']},
                               deadline=__import__('time').monotonic()+600)
    ledger.enter('restart')
    path = FakePath()
    backend = types.SimpleNamespace(encode=lambda content: [1, 2, 3],
                                    answer_ids=lambda text: [1], decode=lambda ids: 'x')
    ref = StageRef(); ref.name = 'restart'
    built = DiagnosticEngineV12(path, backend, types.SimpleNamespace(), ledger,
                                max_new_tokens=8, output=tmp_path, stage_ref=ref)
    row = {'text': 'badge red', 'question': 'color', 'answer': 'red'}
    for _ in range(RESTART_PROMPT_PASSES):
        built.tokens(row)
    snapshot = ledger.snapshot()
    assert snapshot['reserved']['prompt_tokenizer_calls'] == RESTART_PROMPT_PASSES
    assert snapshot['stages']['restart']['reserved']['prompt_tokenizer_calls'] \
        == RESTART_PROMPT_PASSES
    # The next reservation is refused before dispatch, which is the failure the
    # review reproduced at the fifth pass of 67.
    with pytest.raises(BudgetError, match='exceeded before dispatch'):
        built.tokens(row)


def test_nonterminating_generation_is_marked_truncated(tmp_path):
    from replica_lab.intrinsic_memory_diagnostic_v12 import StageRef as Ref
    path = types.SimpleNamespace(validate=lambda batch, scope: None,
        adapter=types.SimpleNamespace(base=types.SimpleNamespace(
            parameters=lambda: iter([torch.zeros(1)]))))
    ref = Ref(); ref.name = 'baseline'
    built = DiagnosticEngineV12(path, None, types.SimpleNamespace(
        context_tokens=16, pad_token_id=0, allowed_stop_token_ids=(1,), vocab_size=8),
        None, max_new_tokens=2, output=tmp_path, stage_ref=ref)
    built.backend = types.SimpleNamespace(decode=lambda ids: 'red')
    built.tokens = lambda row, exact=False: [2]

    @contextmanager
    def measured(operation, **amounts):
        yield {}

    built.measured = measured

    def forward(ids, mask, batch, scope, **kwargs):
        logits = torch.zeros(1, ids.shape[1], 8)
        logits[:, :, 3] = 1        # never the bound stop token
        return logits

    built.forward = forward
    result = built.generate({'text': 'badge red', 'question': 'color', 'answer': 'red'},
                            None, 'scope')
    assert result['terminated'] is False
    assert result['truncated'] is True
    assert result['generated_tokens'] == 2
    # A terminating answer reports the opposite pair.
    def stopping(ids, mask, batch, scope, **kwargs):
        logits = torch.zeros(1, ids.shape[1], 8)
        logits[:, :, 1] = 1        # bound stop token first
        return logits

    built.forward = stopping
    stopped = built.generate({'text': 'badge red', 'question': 'color', 'answer': 'red'},
                             None, 'scope')
    assert stopped['terminated'] is True and stopped['truncated'] is False


def test_snapshot_reads_under_a_shared_lock(tmp_path):
    """A reader must not observe a mid-append journal."""
    import fcntl, time as _time
    log = WorkLedger.create(tmp_path/'ledger.jsonl', identity='packet-sha',
                            plan={'restart': coverage_plan('two')['restart']},
                            deadline=_time.monotonic()+600)
    log.enter('restart')
    lock_path = log.path.with_suffix('.lock')
    holder = lock_path.open('a')
    fcntl.flock(holder, fcntl.LOCK_EX)
    finished = threading.Event()

    def reader():
        log.snapshot(cached=True)
        finished.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    assert not finished.wait(0.2)      # blocked while the writer is mid-append
    fcntl.flock(holder, fcntl.LOCK_UN); holder.close()
    assert finished.wait(5)
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_fit_stage_receipts_and_source_cost_consume_exactly_the_write_cap(tmp_path):
    """X3: every receipt write is reserved first, and the cap is exactly consumed."""
    caps = coverage_plan('two')['fit_128']
    ledger = WorkLedger.create(tmp_path/'fit.jsonl', identity='packet-sha',
                               plan={'fit_128': caps}, deadline=time.monotonic()+600)
    ledger.enter('fit_128')
    path = FakePath(); ref = StageRef(); ref.name = 'fit_128'
    built = DiagnosticEngineV12(path, None, types.SimpleNamespace(), ledger,
                                max_new_tokens=8, output=tmp_path, stage_ref=ref)
    for index in range(128):
        built.write_receipt('fit', {'updates': index, 'planned_updates': 512})
    built.write_source_costs('fit_128')
    reserved = ledger.snapshot()['stages']['fit_128']['reserved']
    assert reserved['writes'] == 129 == caps['writes']      # 128 receipts + 1 source cost
    with pytest.raises(BudgetError, match='exceeded before dispatch'):
        built.write_receipt('fit', {'updates': 129})


def test_restart_stage_reservations_fit_the_declared_caps(tmp_path):
    """X3 whole-stage consumption for the stage the review found under-reserved."""
    caps = coverage_plan('two')['restart']
    ledger = WorkLedger.create(tmp_path/'restart.jsonl', identity='packet-sha',
                               plan={'restart': caps}, deadline=time.monotonic()+600)
    ledger.enter('restart')
    batch = types.SimpleNamespace(read_costs={'cold_wait_ns': 0})
    path = ReadPath([batch]); ref = StageRef(); ref.name = 'restart'
    backend = types.SimpleNamespace(encode=lambda content: [1, 2, 3],
                                    answer_ids=lambda text: [1], decode=lambda ids: 'x')
    built = DiagnosticEngineV12(path, backend, types.SimpleNamespace(), ledger,
                                max_new_tokens=8, output=tmp_path, stage_ref=ref)
    checkpoint = caps['artifact_bytes']
    teacher = coverage_plan('two')['teacher']['artifact_bytes']
    with ledger.work('restart', 'official_bf16_model_load', model_loads=1,
                     artifact_read_bytes=SOURCE_FILE_BYTES) as event:
        event['actual'] = {'model_loads': 1, 'artifact_read_bytes': SOURCE_FILE_BYTES}
    with ledger.work('restart', 'child_source_revalidation',
                     artifact_read_bytes=SOURCE_FILE_BYTES + ENVIRONMENT_CLOSURE_BYTES) as event:
        event['actual'] = {'artifact_read_bytes':
                           SOURCE_FILE_BYTES + ENVIRONMENT_CLOSURE_BYTES}
    for operation, restore_bytes in (('extension_optimizer_restore', checkpoint),
                                     ('teacher_cache_restore', teacher)):
        with ledger.work('restart', operation, restart_restores=1,
                         artifact_read_bytes=restore_bytes) as event:
            event['actual'] = {'artifact_read_bytes': restore_bytes}
    row = {'text': 'badge red', 'question': 'color', 'answer': 'red'}
    for _ in range(RESTART_PROMPT_PASSES):
        built.tokens(row)
    built.read(None, 'scope', 'question')
    built.write_receipt('restart_result', {'ok': True})
    built.write_source_costs('restart')
    reserved = ledger.snapshot()['stages']['restart']['reserved']
    assert all(reserved[key] <= caps[key] for key in reserved)
    assert reserved['model_loads'] == 1 == caps['model_loads']
    assert reserved['restart_restores'] == 2 == caps['restart_restores']
    assert reserved['prompt_tokenizer_calls'] == 67
    assert reserved['writes'] == READ_WRITE_BOUND + 2   # bounded read + result + source cost
    assert reserved['artifact_read_bytes'] <= caps['artifact_read_bytes']


def test_admission_reserves_source_and_closure_hashing(tmp_path):
    """X3: the preflight hash passes are charged before they happen."""
    caps = coverage_plan('two')['admission']
    ledger = WorkLedger.create(tmp_path/'admission.jsonl', identity='packet-sha',
                               plan={'admission': caps}, deadline=time.monotonic()+600)
    ledger.enter('admission')
    with ledger.work('admission', 'bounded_audit_and_store_envelope',
                     artifact_bytes=caps['artifact_bytes'], writes=1):
        pass
    amount = SOURCE_FILE_BYTES + ENVIRONMENT_CLOSURE_BYTES
    with ledger.work('admission', 'environment_source_code_data_preflight',
                     artifact_read_bytes=amount) as event:
        event['actual'] = {'artifact_read_bytes': amount}
    reserved = ledger.snapshot()['stages']['admission']['reserved']
    assert reserved['artifact_read_bytes'] == amount <= caps['artifact_read_bytes']
    assert reserved['artifact_bytes'] == caps['artifact_bytes']
    assert reserved['writes'] == 1
    # One more byte of hashing would exceed the declared envelope before dispatch.
    with pytest.raises(BudgetError, match='exceeded before dispatch'):
        ledger.reserve('admission', 'extra_hash', artifact_read_bytes=1)


def test_attempt_marker_is_durable_append_only_and_survives_output_deletion(tmp_path):
    """X9: attempt consumption no longer depends on a deletable output directory."""
    from replica_lab.intrinsic_memory_execution_v12 import (
        attempt_consumed, read_attempt_marker, record_attempt)
    marker = tmp_path/'runs'/'attempts.jsonl'
    output = tmp_path/'reports'/'attempt01'
    output.mkdir(parents=True)
    record = record_attempt(marker, packet_sha256='a'*64, protocol_sha256='b'*64,
                           arm='two', output_root='reports/attempt01',
                           authorization_sha256='c'*64)
    assert read_attempt_marker(marker) == [record]
    with pytest.raises(ValueError, match='already consumed for this packet'):
        record_attempt(marker, packet_sha256='a'*64, protocol_sha256='b'*64, arm='two',
                       output_root='reports/other', authorization_sha256='c'*64)
    with pytest.raises(ValueError, match='already consumed for this output root'):
        record_attempt(marker, packet_sha256='d'*64, protocol_sha256='b'*64, arm='two',
                       output_root='reports/attempt01', authorization_sha256='c'*64)
    output.rmdir()                       # deleting the output tree restores nothing
    assert attempt_consumed(marker, packet_sha256='a'*64) is True
    assert attempt_consumed(marker, output_root='reports/attempt01') is True
    assert marker.read_bytes().count(b'\n') == 1     # append-only, one record
    marker.write_bytes(marker.read_bytes() + b'{"format":"replica')
    with pytest.raises(ValueError, match='torn attempt marker'):
        read_attempt_marker(marker)


def test_environment_closure_detects_substitution_and_added_libraries(tmp_path):
    """X9: shared libraries, wheels and the interpreter are bound by actual hashes."""
    from replica_lab.intrinsic_memory_execution_v12 import (
        ENVIRONMENT_CLOSURE_FORMAT, verify_environment_closure)
    prefix = tmp_path/'env'
    (prefix/'bin').mkdir(parents=True)
    site = prefix/'lib/python3.12/site-packages'
    libdir = site/'torch'/'lib'
    libdir.mkdir(parents=True)
    interp = prefix/'bin/python3.12'; interp.write_bytes(b'interpreter')
    origin = site/'torch/__init__.py'; origin.write_bytes(b'init')
    ext = site/'torch/_C.so'; ext.write_bytes(b'ext')
    lib = libdir/'libtorch.dylib'; lib.write_bytes(b'lib')
    record = site/'x-1.dist-info/RECORD'
    record.parent.mkdir(); record.write_bytes(b'record')
    crit = prefix/'crit.py'; crit.write_bytes(b'crit')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    def entry(path, relative):
        return {'path_relative': relative, 'sha256': digest(path),
                'bytes': path.stat().st_size}
    closure = {
        'format': ENVIRONMENT_CLOSURE_FORMAT, 'prefix': str(prefix.resolve()),
        'interpreter': {'path': str(interp), 'path_relative': 'bin/python3.12',
                        'resolved_path': str(interp),
                        'sha256': digest(interp), 'bytes': interp.stat().st_size},
        'module_origins': {'torch': entry(origin,
            'lib/python3.12/site-packages/torch/__init__.py')},
        'native_extensions': {'torch._C': entry(
            ext, 'lib/python3.12/site-packages/torch/_C.so')},
        'shared_library_root': 'lib/python3.12/site-packages/torch/lib',
        'shared_libraries': {'lib/python3.12/site-packages/torch/lib/libtorch.dylib':
                             {'sha256': digest(lib), 'bytes': lib.stat().st_size}},
        'site_packages': 'lib/python3.12/site-packages',
        'wheel_records': {'lib/python3.12/site-packages/x-1.dist-info/RECORD':
                          {'sha256': digest(record), 'bytes': record.stat().st_size}},
        'critical_files': {'crit.py': digest(crit)},
    }
    receipt = verify_environment_closure(closure, prefix=prefix, interpreter=interp)
    assert receipt['files_verified'] == 6
    assert receipt['bytes_verified'] == sum(
        (interp, origin, ext, lib, record, crit)[i].stat().st_size for i in range(6))
    lib.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='content differs'):
        verify_environment_closure(closure, prefix=prefix, interpreter=interp)
    lib.write_bytes(b'lib')
    (libdir/'added.dylib').write_bytes(b'added')
    with pytest.raises(ValueError, match='shared library set differs'):
        verify_environment_closure(closure, prefix=prefix, interpreter=interp)
    (libdir/'added.dylib').unlink()
    with pytest.raises(ValueError, match='interpreter differs'):
        verify_environment_closure(closure, prefix=prefix, interpreter=crit)


def test_fitting_close_and_checkpoint_are_charged_in_declared_stages(tmp_path):
    """X14: the terminal fitting receipt must be written inside an active stage."""
    stages = coverage_plan('two')
    ledger = WorkLedger.create(tmp_path/'lifecycle.jsonl', identity='packet-sha',
                               plan={'fit_probes_512': stages['fit_probes_512'],
                                     'checkpoint': stages['checkpoint'],
                                     'finalize_worker': stages['finalize_worker']},
                               deadline=time.monotonic()+600)
    path = FakePath(); ref = StageRef(); ref.name = 'fit_probes_512'
    built = DiagnosticEngineV12(path, None, types.SimpleNamespace(), ledger,
                                max_new_tokens=8, output=tmp_path, stage_ref=ref)
    begin_stage(built, 'fit_probes_512')
    # The review's reproduction: writing the receipt after the stage closes is
    # refused by the real ledger, which is why the write now happens first.
    built.write_receipt('probe', {'ok': True})
    built.freeze()
    ledger.leave('fit_probes_512', status='completed')
    with pytest.raises(BudgetError, match='not active'):
        built.write_receipt('fitting_closed', {'closed': True})
    # The repaired order: freeze and the fitting-closed receipt happen while the
    # final probe stage is still open.
    ledger2 = WorkLedger.create(tmp_path/'lifecycle2.jsonl', identity='packet-sha',
                                plan={'fit_probes_512': stages['fit_probes_512'],
                                      'checkpoint': stages['checkpoint']},
                                deadline=time.monotonic()+600)
    ref2 = StageRef(); ref2.name = 'fit_probes_512'
    path2 = FakePath()
    built2 = DiagnosticEngineV12(path2, None, types.SimpleNamespace(), ledger2,
                                 max_new_tokens=8, output=tmp_path, stage_ref=ref2)
    begin_stage(built2, 'fit_probes_512')
    built2.write_receipt('probe', {'ok': True})
    built2.write_receipt('fit_curve', {'prefixes': []})
    built2.freeze()
    built2.write_receipt('fitting_closed', {'closed': True, 'updates': 512})
    close_stage(built2, 'fit_probes_512', 'completed')
    begin_stage(built2, 'checkpoint')
    checkpoint = stages['checkpoint']
    with ledger2.work('checkpoint', 'extension_optimizer_checkpoint',
                      artifact_bytes=checkpoint['artifact_bytes'], writes=4) as event:
        (tmp_path/'checkpoint').mkdir()
        for name in ('weights', 'optimizer', 'components', 'manifest'):
            (tmp_path/'checkpoint'/name).write_bytes(b'x')
        event['actual'] = {'writes': 4}
    built2.write_receipt('restart_request', {'updates_allowed': 1})
    close_stage(built2, 'checkpoint', 'completed')
    snapshot = ledger2.snapshot()
    assert snapshot['stages']['fit_probes_512']['status'] == 'completed'
    assert snapshot['stages']['checkpoint']['status'] == 'completed'
    probe_reserved = snapshot['stages']['fit_probes_512']['reserved']
    checkpoint_reserved = snapshot['stages']['checkpoint']['reserved']
    assert probe_reserved['writes'] <= stages['fit_probes_512']['writes']
    assert checkpoint_reserved['writes'] <= checkpoint['writes']
    assert checkpoint_reserved['artifact_bytes'] == checkpoint['artifact_bytes']
    assert 'fitting_closed_receipt' in [
        event['operation'] for event in snapshot['events']
        if event['stage'] == 'fit_probes_512']


def test_finalize_receipt_charges_the_terminal_result_in_its_own_stage(tmp_path):
    """X3: the terminal result is never written outside a declared stage."""
    from replica_lab.intrinsic_memory_native_v12 import finalize_receipt
    stages = coverage_plan('two')
    ledger = WorkLedger.create(tmp_path/'finalize.jsonl', identity='packet-sha',
                               plan={'finalize_worker': stages['finalize_worker'],
                                     'fit_128': stages['fit_128']},
                               deadline=time.monotonic()+600)
    ref = StageRef(); ref.name = 'fit_128'
    path = FakePath()
    built = DiagnosticEngineV12(path, None, types.SimpleNamespace(), ledger,
                                max_new_tokens=8, output=tmp_path, stage_ref=ref)
    ledger.enter('fit_128')
    # A failure leaves the working stage open; finalize closes it as failed and
    # writes the result inside the declared finalize stage.
    ledger.reserve('fit_128', 'simulated_pending_work', writes=1)
    assert finalize_receipt(built, 'main_result', {'status': 'failed'}) is False
    uncharged = json.loads((tmp_path/'main_result.json').read_text())
    assert uncharged['final_receipt_charging']['charged'] is False
    # With no uncertain reservation the result is charged in its own stage.
    ledger2 = WorkLedger.create(tmp_path/'finalize2.jsonl', identity='packet-sha',
                                plan={'finalize_worker': stages['finalize_worker']},
                                deadline=time.monotonic()+600)
    ref2 = StageRef(); ref2.name = None
    built2 = DiagnosticEngineV12(FakePath(), None, types.SimpleNamespace(), ledger2,
                                 max_new_tokens=8, output=tmp_path/'second', stage_ref=ref2)
    assert finalize_receipt(built2, 'main_result', {'status': 'completed'}) is True
    snapshot = ledger2.snapshot()
    assert snapshot['stages']['finalize_worker']['status'] == 'completed'
    assert snapshot['stages']['finalize_worker']['reserved']['writes'] == 2
    receipt = json.loads((tmp_path/'second'/'main_result.json').read_text())
    assert 'final_receipt_charging' not in receipt


def test_terminal_receipts_use_distinct_stages_and_paths(tmp_path):
    """X3: capacity/terminal receipts never share a path or a consumed stage."""
    from replica_lab.intrinsic_memory_execution_v12 import charged_write
    stages = coverage_plan('two')
    plan = {name: stages[name] for name in
            ('capacity_main', 'capacity_restart', 'finalize_parent')}
    ledger = WorkLedger.create(tmp_path/'parent.jsonl', identity='packet-sha',
                               plan=plan, deadline=time.monotonic()+600)
    for stage, receipt, window in (('capacity_main', 'capacity_guard_main', 'main'),
                                   ('capacity_restart', 'capacity_guard_restart', 'restart')):
        ledger.enter(stage)
        assert charged_write(ledger, tmp_path, receipt,
                             {'supervision_window': window}, stage=stage) is True
        ledger.leave(stage, status='completed')
    ledger.enter('finalize_parent')
    assert charged_write(ledger, tmp_path, 'result', {'ok': True},
                         stage='finalize_parent') is True
    ledger.leave('finalize_parent', status='completed')
    snapshot = ledger.snapshot()
    for stage in plan:
        assert snapshot['stages'][stage]['status'] == 'completed'
        assert snapshot['stages'][stage]['reserved']['writes'] == 1
    main = json.loads((tmp_path/'capacity_guard_main.json').read_text())
    restart = json.loads((tmp_path/'capacity_guard_restart.json').read_text())
    assert main['supervision_window'] == 'main'
    assert restart['supervision_window'] == 'restart'
    assert 'receipt_charging' not in main and 'receipt_charging' not in restart
    result = json.loads((tmp_path/'result.json').read_text())
    assert result['ok'] is True and 'receipt_charging' not in result


def test_charged_write_never_consumes_a_terminal_stage(tmp_path):
    from replica_lab.intrinsic_memory_execution_v12 import charged_write
    stages = coverage_plan('two')
    ledger = WorkLedger.create(tmp_path/'no-stage.jsonl', identity='packet-sha',
                               plan={name: stages[name] for name in
                                     ('finalize_worker', 'finalize_parent')},
                               deadline=time.monotonic()+600)
    for name in ('capacity_guard_main', 'capacity_guard_restart', 'result'):
        assert charged_write(ledger, tmp_path, name, {'ok': True}) is False
        payload = json.loads((tmp_path/(name+'.json')).read_text())
        assert payload['receipt_charging']['charged'] is False
    snapshot = ledger.snapshot()
    assert 'finalize_worker' not in snapshot['stages']
    assert 'finalize_parent' not in snapshot['stages']
    assert snapshot['reserved']['writes'] == 0


def test_charged_write_does_not_mask_the_original_error(tmp_path):
    from replica_lab.intrinsic_memory_execution_v12 import charged_write

    class BrokenLedger:
        def snapshot(self):
            raise RuntimeError('ledger unavailable')

        def work(self, *args, **kwargs):
            raise RuntimeError('ledger unavailable')

    def failing():
        raise ValueError('original failure')

    with pytest.raises(ValueError, match='original failure'):
        try:
            failing()
        except ValueError:
            charged_write(BrokenLedger(), tmp_path, 'failure', {'ok': False})
            raise
    payload = json.loads((tmp_path/'failure.json').read_text())
    assert payload['receipt_charging']['reason'] == 'RuntimeError'


def test_release_reads_are_charged_to_native_read_bytes(tmp_path):
    """X3: release authority/manifest/payload reads enter artifact_read_bytes."""
    from replica_lab.intrinsic_memory_coverage_v12 import RELEASE_READ_BYTES_BOUND
    caps = coverage_plan('two')['prepare']
    ledger = WorkLedger.create(tmp_path/'reads.jsonl', identity='packet-sha',
                               plan={'prepare': caps}, deadline=time.monotonic()+600)
    ledger.enter('prepare')
    path = ReadPath([types.SimpleNamespace(read_costs={})])
    path.feature_provider = types.SimpleNamespace(cost={})
    path.cost = {'read_file_bytes': 0, 'ticket_descriptor_read_bytes': 0}
    path.release_corpus = types.SimpleNamespace(
        cost={'source_read_bytes': 0, 'authority_read_bytes': 0, 'read_bytes': 0})
    ref = StageRef(); ref.name = 'prepare'
    built = DiagnosticEngineV12(path, None, types.SimpleNamespace(), ledger,
                                max_new_tokens=8, output=tmp_path, stage_ref=ref)
    original_read = path.read

    def read(store, question, *, scope, wait_seconds=0.0):
        # A release read touches the authority, the manifest and a payload.
        path.release_corpus.cost['authority_read_bytes'] += 512
        path.release_corpus.cost['read_bytes'] += 4096
        path.release_corpus.cost['source_read_bytes'] += 128
        path.cost['read_file_bytes'] += 2048
        return original_read(store, question, scope=scope, wait_seconds=wait_seconds)

    path.read = read
    built.read(None, 'scope', 'question')
    reserved = ledger.snapshot()['stages']['prepare']['reserved']
    assert reserved['artifact_read_bytes'] == RELEASE_READ_BYTES_BOUND
    reservation = [event for event in ledger.snapshot()['events']
                   if event['operation'] == 'retrieval'][0]
    assert reservation['reserved']['artifact_read_bytes'] == RELEASE_READ_BYTES_BOUND
    # Direct release reads are reserved separately by the corpus provider, so
    # this event's actual counts only the staged path bytes.
    assert reservation['actual']['artifact_read_bytes'] == 2048
    assert reservation['actual']['artifact_read_bytes'] < RELEASE_READ_BYTES_BOUND


def test_release_direct_reads_are_reserved_on_the_native_ledger(tmp_path):
    """Direct release reads enter artifact_read_bytes, staged reads do not repeat."""
    from replica_lab.contextual_release_v12 import WorkLedgerReleaseReadAccounting
    from replica_lab.intrinsic_memory_coverage_v12 import release_direct_read_bound
    caps = coverage_plan('two')['fit_probes_128']
    ledger = WorkLedger.create(tmp_path/'release.jsonl', identity='packet-sha',
                               plan={'fit_probes_128': caps}, deadline=time.monotonic()+600)
    ledger.enter('fit_probes_128')
    accounting = WorkLedgerReleaseReadAccounting(ledger, 'fit_probes_128')
    token = accounting.reserve_read('release_authority_read', 4096)
    accounting.complete_read(token, actual_bytes=3000)
    token = accounting.reserve_read('release_source_read', 256)
    accounting.complete_read(token, actual_bytes=256)
    reserved = ledger.snapshot()['stages']['fit_probes_128']['reserved']
    assert reserved['artifact_read_bytes'] == 4096 + 256
    events = [event for event in ledger.snapshot()['events']
              if event['operation'].startswith('release_')]
    assert [event['actual']['artifact_read_bytes'] for event in events] == [3000, 256]
    # The probe stage reserves the staged read path and the corpus' direct reads
    # for every world, so both kinds have room.
    assert caps['artifact_read_bytes'] >= 32*(RELEASE_READ_BYTES_BOUND
                                              + release_direct_read_bound(2))


def test_engine_builds_release_corpora_with_the_read_provider(monkeypatch, tmp_path):
    """The constructor's origin/lock work is reserved before it runs."""
    from replica_lab.contextual_release_v12 import WorkLedgerReleaseReadAccounting
    import replica_lab.intrinsic_memory_diagnostic_v12 as diagnostic
    stages = coverage_plan('two')
    ledger = WorkLedger.create(tmp_path/'origin.jsonl', identity='packet-sha',
                               plan={'fit_probes_128': stages['fit_probes_128']},
                               deadline=time.monotonic()+600)
    ledger.enter('fit_probes_128')
    captured = {}

    class Recorder:
        cost = {'artifact_read_bytes': 512}

        def __init__(self, store, scope, *, read_accounting=None):
            captured['accounting'] = read_accounting

    monkeypatch.setattr(diagnostic, 'ProducerReleaseCorpus', Recorder)
    ref = StageRef(); ref.name = 'fit_probes_128'
    built = DiagnosticEngineV12(FakePath(), None, types.SimpleNamespace(), ledger,
                                max_new_tokens=8, output=tmp_path, stage_ref=ref)
    built.new_release_corpus('store', 'scope')
    accounting = captured['accounting']
    assert isinstance(accounting, WorkLedgerReleaseReadAccounting)
    assert accounting.ledger is ledger and accounting.stage == 'fit_probes_128'
    # The corpus provider reserves its own origin write and direct reads, so the
    # engine must not add a second origin reservation.
    operations = [event['operation'] for event in ledger.snapshot()['events']]
    assert 'release_origin_and_lock' not in operations
    assert ledger.snapshot()['reserved']['writes'] == 0


def test_release_direct_read_bound_scales_with_active_sites():
    """Release-batch forwards revalidate per site, so the bound must grow with S."""
    # current_calls = 2 (rebuild) + (2 + R) (read) + (S + 1) (forward) = R + S + 5
    assert release_direct_read_bound(2) == (2 + 2 + 2 + 3) * 2 * 4*1024**2 + 16*1024**2
    assert release_direct_read_bound(2, records=1) == (2 + 2 + 1 + 3) * 2 * 4*1024**2 + 16*1024**2
    assert release_direct_read_bound(2, records=2) - release_direct_read_bound(2, records=1) \
        == 2 * 4*1024**2
    assert release_direct_read_bound(9) > release_direct_read_bound(2)
    assert release_direct_read_bound(36) > release_direct_read_bound(9)
    from replica_lab.intrinsic_memory_coverage_v12 import ARMS
    for arm in ARMS:
        sites = len(ARMS[arm])
        stage = coverage_plan(arm)['fit_probes_128']
        assert stage['artifact_read_bytes'] >= 32*(RELEASE_READ_BYTES_BOUND
                                                  + release_direct_read_bound(sites))


def test_context_width_and_prefix_bounds_match_the_envelope():
    spec = types.SimpleNamespace(context_tokens=256, pad_token_id=2)
    ids, mask = right_padded_prefix([5, 6, 7], spec, torch.device('cpu'))
    assert tuple(ids.shape) == (1, 256) and int(mask.sum()) == 3
    with pytest.raises(ValueError, match='truncate'):
        right_padded_prefix(list(range(257)), spec, torch.device('cpu'))
