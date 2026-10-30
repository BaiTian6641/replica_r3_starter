"""v12 diagnostic engine: live frozen-source compression with explicit work accounting.

Differences from the sealed v5 engine, all deliberate:

* The producer is a :class:`FrozenSourceFeatures` provider that reserves source
  work through an explicit accounting adapter *before* tokenizer work and
  *before* the frozen forward. A counter-only fallback does not exist.
* ``ContextualMemoryLayoutV12.snapshot`` adds the learned slot order itself, so
  the contextual path calls ``path.forward`` once and never adds order again.
  Intervention controls (wrong/shuffled/zero) build their own snapshot and add
  order exactly once, because they intentionally bypass ``path.snapshot``.
* Dispatch wall and synchronized wall are recorded separately, and stage
  receipts carry observed cache behaviour next to (never inside) the durable
  reservations.
* Chat-template prompt tokenization is charged to ``prompt_tokenizer_calls``,
  which is distinct from the canonical-span ``tokenizer_calls`` vocabulary.

No P update, promotion, teacher update, retrieval or write happens inside a
decoder forward. Controls are explicit probes, not admitted evidence.
"""
from contextlib import contextmanager
import hashlib
import time

import torch

from .baseline_competence import score
from .contextual_completion_objective_v1 import encode_complete_response, complete_response_loss
from .contextual_memory import ContextualMemoryError, WIDTH, tensor_digest
from .contextual_memory_v5 import PendingContextualRead
from .contextual_release_v12 import (ProducerReleaseCorpus,
    WorkLedgerReleaseReadAccounting)
from .contextual_source_features_v12 import (
    SOURCE_FORWARD, SOURCE_TOKENIZE, SourceWorkRequest)
from .intrinsic_memory_budget_v12 import atomic
from .intrinsic_memory_coverage_v12 import (
    RELEASE_READ_BYTES_BOUND, SOURCE_MAX_BYTES, SOURCE_MAX_TOKENS)

FORMAT = 'replica.intrinsic_memory.diagnostic.v12'
CONTROLS = ('contextual', 'exact_text', 'wrong', 'shuffled', 'zero', 'absent')
PROMPT = ('Reply with only the answer word. If unsupported reply unknown.\n'
          '{evidence}Question: {question}\nAnswer:')
PROVIDER_COUNTERS = (
    'requests', 'tokenizer_calls', 'tokenizer_cache_hits', 'source_bytes', 'forwards',
    'input_tokens', 'cache_hits', 'cache_misses', 'cache_evictions', 'cache_bytes',
    'peak_cache_bytes', 'feature_bytes_produced', 'source_traced_layers', 'wall_ns')


def prompt(row, *, exact=False):
    return PROMPT.format(evidence=('Evidence: '+row['text']+'\n') if exact else '',
                         question=row['question'])


def right_padded_prefix(tokens, spec, device):
    if not tokens or len(tokens) > spec.context_tokens:
        raise ValueError('prefix is empty or would truncate evidence')
    ids = torch.full((1, spec.context_tokens), spec.pad_token_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    ids[0, :len(tokens)] = torch.tensor(tokens, device=device); mask[0, :len(tokens)] = 1
    return ids, mask


def logits_digest(logits):
    """Bounded CPU chunks; retain no full logit or original-weight export."""
    h = hashlib.sha256(); flat = logits.detach().reshape(-1)
    for start in range(0, flat.numel(), 1024*1024):
        part = flat[start:start+1024*1024].float().cpu().contiguous()
        h.update(memoryview(part.numpy()))
    return h.hexdigest()


class StageRef:
    """Shared stage name so durable source accounting always sees the live stage."""

    def __init__(self):
        self.name = None


class SourceWorkAccounting:
    """Reserve canonical-source work on the v12 ledger before it is dispatched.

    Implements ``contextual_source_features_v12.SourceWorkAccounting`` exactly:
    the provider calls ``reserve`` before tokenizer work and before the frozen
    forward, and ``complete`` with the observed counts. Amounts are computed
    here so the v12 vocabulary is charged in the same durable record as the
    inherited counters; no second event is invented, and a refusal happens
    before any model or tokenizer work.
    """

    def __init__(self, ledger, stage_ref):
        if not callable(getattr(ledger, 'reserve', None)) or not callable(getattr(ledger, 'finish', None)):
            raise ContextualMemoryError('explicit work ledger with reserve/finish required')
        if not isinstance(stage_ref, StageRef):
            raise ContextualMemoryError('live stage reference required for source accounting')
        self.ledger, self.stage_ref = ledger, stage_ref
        self.reserved = {SOURCE_TOKENIZE: 0, SOURCE_FORWARD: 0}

    def _amounts(self, request):
        if not isinstance(request, SourceWorkRequest):
            raise ContextualMemoryError('SourceWorkRequest required')
        if request.source_bytes > SOURCE_MAX_BYTES:
            raise ContextualMemoryError('canonical span exceeds the declared source byte bound')
        if request.max_tokens != SOURCE_MAX_TOKENS:
            raise ContextualMemoryError('source token bound differs from the declared envelope')
        if request.operation == SOURCE_TOKENIZE:
            if request.tokens is not None:
                raise ContextualMemoryError('tokenizer reservation must precede tokenization')
            return {'source_encodes': 1, 'tokenizer_calls': 1,
                    'source_token_bytes': request.source_bytes,
                    'input_tokens': request.source_bytes}
        if request.tokens is None or request.tokens > SOURCE_MAX_TOKENS:
            raise ContextualMemoryError('a source forward requires its exact bounded token count')
        return {'forwards': 1, 'source_feature_forwards': 1, 'input_tokens': request.tokens,
                'cache_misses': 1, 'cache_evictions': 1, 'cache_recomputations': 1}

    def reserve(self, request):
        stage = self.stage_ref.name
        if not isinstance(stage, str) or not stage:
            # Fail closed before any tokenizer or model work.
            raise ContextualMemoryError('source feature work outside a declared stage')
        amounts = self._amounts(request)
        self.reserved[request.operation] += 1
        # request.operation is already the declared vocabulary term
        # ('source_tokenize' / 'source_forward'); do not prefix it again.
        return self.ledger.reserve(stage, request.operation, **amounts)

    def complete(self, token, *, status='completed', actual=None, detail=None):
        if status not in ('completed', 'failed'):
            raise ContextualMemoryError('invalid source work status')
        return self.ledger.finish(token, status=status, actual=dict(actual or {}),
                                  detail=dict(detail or {}))


class DiagnosticEngineV12:
    def __init__(self, path, backend, spec, ledger, *, max_new_tokens, output,
                 stage_ref=None):
        self.path, self.backend, self.spec, self.ledger = path, backend, spec, ledger
        self.max_new_tokens, self.output = max_new_tokens, output
        self.output.mkdir(exist_ok=True, parents=True)
        self._stage_ref = stage_ref if stage_ref is not None else StageRef()
        self.events = []
        self.fitting_closed = False
        self.source_observations = []

    @property
    def stage(self):
        return self._stage_ref.name

    @stage.setter
    def stage(self, value):
        self._stage_ref.name = value

    @property
    def device(self):
        return next(self.path.adapter.base.parameters()).device

    def sync(self):
        if self.device.type == 'mps': torch.mps.synchronize()

    @contextmanager
    def measured(self, operation, **amounts):
        with self.ledger.work(self.stage, operation, **amounts) as event:
            self.sync(); start = time.monotonic_ns()
            try:
                yield event
            finally:
                dispatched = time.monotonic_ns()
                self.sync()
                event['dispatch_ns'] = dispatched-start
                event['synchronized_wall_ns'] = time.monotonic_ns()-start
                event['physical_io_bytes_measured'] = False

    # ------------------------------------------------------------------
    # Prompt tokens and encodings

    def tokens(self, row, *, exact=False):
        with self.measured('prompt_tokenization', prompt_tokenizer_calls=1):
            tokens = self.backend.encode(prompt(row, exact=exact))
        if exact:
            self.path.cost['fallback_calls'] += 1
            self.path.cost['fallback_bytes'] += len(row['text'].encode())
            self.path.cost['fallback_tokens'] += len(self.backend.answer_ids(row['text']))
        return tokens

    def encoding(self, row, *, exact=False):
        return encode_complete_response(self.tokens(row, exact=exact),
                                        self.backend.answer_ids(row['answer']),
                                        spec=self.spec, device=self.device)

    # ------------------------------------------------------------------
    # Source features and durable reads

    def provider_cost(self):
        return dict(self.path.feature_provider.cost)

    def live(self, source, scope):
        """Re-encode admitted canonical spans; reservations precede dispatch."""
        before = self.provider_cost()
        with self.measured('source_encoding') as event:
            values = self.path.training_from_stored(source, scope=scope)
        after = self.provider_cost()
        delta = {key: after.get(key, 0)-before.get(key, 0) for key in PROVIDER_COUNTERS}
        event['source_observation'] = delta
        self.source_observations.append({'stage': self.stage, **delta})
        return values

    def read(self, store, scope, question, *, writes=3, resume_writes=2,
             read_bytes=RELEASE_READ_BYTES_BOUND):
        # Descriptor writes, per the producer owner's final enumeration: a
        # successful wait-based read persists three (initial pending, dispatch
        # reservation, terminal), a waiting read that times out persists two, and
        # a read whose own dispatch fails is also three. Three is therefore the
        # exact unconditional bound. Each explicit bounded resume persists at most
        # two (one when it attaches to an in-flight ticket, none when refused
        # before dispatch). There is no automatic retry: one selection, at most
        # four caller waits, then a typed refusal.
        #
        # The durable read also touches release authority, manifest, pointer and
        # payload files; that work is reserved here and the observed bytes are
        # recorded afterwards so it lands on native artifact_read_bytes.
        with self.measured('retrieval', retrievals=1, writes=writes,
                           artifact_read_bytes=read_bytes) as event:
            before = self.observed_read_bytes()
            batch = self.path.read(store, question, scope=scope, wait_seconds=1)
            waits = 0
            while isinstance(batch, PendingContextualRead) and waits < 4:
                waits += 1
                with self.measured('contextual_resume', writes=resume_writes):
                    batch = self.path.resume(store, batch, scope=scope, wait_seconds=1)
            if isinstance(batch, PendingContextualRead):
                event['pending'] = batch.__dict__
                raise ContextualMemoryError('cold recall incomplete after five bounded waits')
            observed = self.observed_read_bytes() - before
            event['costs'] = getattr(batch, 'read_costs', {})
            event['observed_read_bytes'] = observed
            event['bounded_resumes'] = waits
            event['actual'] = {'writes': writes, 'artifact_read_bytes': observed}
            if self.path.release_corpus is not None:
                event['release_authority_costs'] = dict(self.path.release_corpus.cost)
            return batch

    def observed_read_bytes(self):
        """Read bytes the path and any release corpus have actually charged."""
        cost = self.path.cost
        total = cost.get('read_file_bytes', 0) + cost.get('ticket_descriptor_read_bytes', 0)
        corpus = getattr(self.path, 'release_corpus', None)
        if corpus is not None:
            source = corpus.cost
            total += (source.get('source_read_bytes', 0)
                      + source.get('authority_read_bytes', 0)
                      + source.get('read_bytes', 0))
        return total

    def new_release_corpus(self, store, scope):
        """Corpus bound to the native read ledger, charged before each direct read.

        Direct release reads (pointer, manifest authority, canonical source and
        payload values) are reserved as ``artifact_read_bytes`` before they
        happen and completed with the observed bytes. Staged cold/release payload
        reads stay on the path retrieval accounting and are not double-reserved.
        """
        return ProducerReleaseCorpus(
            store, scope,
            read_accounting=WorkLedgerReleaseReadAccounting(self.ledger, self.stage))

    def write_receipt(self, name, payload, *, operation=None):
        """Charge one declared write before it happens, then record the actual."""
        path = self.output/(name+'.json')
        with self.measured(operation or (name+'_receipt'), writes=1) as event:
            atomic(path, payload)
            event['actual'] = {'writes': 1}
            event['serialized_write_bytes'] = path.stat().st_size
        return payload

    def write_source_costs(self, name):
        """Observed provider work next to, never inside, the ledger reservations."""
        path = self.output/('source_cost_'+name+'.json')
        payload = {
            'format': FORMAT+'.source_cost',
            'stage': name,
            'provider_observed': self.provider_cost(),
            'observations_this_call': [o for o in self.source_observations if o['stage'] == name],
            'reservation_source': 'durable_work_ledger_v12',
            'note': ('observations are measured after dispatch; reservations are the charged '
                     'upper bounds and are never reduced by an observation'),
        }
        with self.measured('source_cost_receipt', writes=1) as event:
            atomic(path, payload)
            event['actual'] = {'writes': 1}
            event['serialized_write_bytes'] = path.stat().st_size

    # ------------------------------------------------------------------
    # Forward paths

    def _forward(self, ids, mask, batch, scope, *, control='contextual', wrong=None,
                 wrong_scope=None):
        if control not in CONTROLS: raise ValueError('undeclared control')
        if control in ('absent', 'exact_text') or batch is None:
            return self.path.forward(ids, mask, None, scope=scope, enabled=False)
        if control == 'contextual':
            # snapshot() adds the learned slot order once; do not add it here.
            return self.path.forward(ids, mask, batch, scope=scope)
        # A measured intervention after authentic-source validation. It cannot
        # bypass admission, manufacture a durable record or change its authority.
        self.path.validate(batch, scope)
        slots = batch.slots
        if control == 'wrong':
            if wrong is None or wrong_scope is None: raise ValueError('wrong control source missing')
            self.path.validate(wrong, wrong_scope)
            slots = wrong.slots
        elif control == 'shuffled':
            slots = slots.flip(0)
        # Exactly one order addition on this path: path.snapshot is bypassed
        # because it would add order a second time.
        order = self.path.order(torch.arange(len(slots), device=ids.device))
        slots = slots + order
        if control == 'zero': slots = torch.zeros_like(slots)
        padding = torch.zeros((self.path.spec.slots-len(slots), WIDTH),
                              device=slots.device, dtype=slots.dtype)
        values = torch.cat((slots, padding))[None, None].expand(ids.shape[0], ids.shape[1], -1, -1)
        mask_slots = (torch.arange(self.path.spec.slots, device=ids.device) < len(slots))
        mask_slots = mask_slots[None, None].expand(values.shape[:3])

        def validate():
            self.path.validate(batch, scope)
            if control == 'wrong': self.path.validate(wrong, wrong_scope)

        context = self.path.reader.snapshot(values, mask_slots, validate=validate)
        self.path.cost['snapshot_copies'] += 1
        self.path.cost['snapshot_bytes'] += context.values.numel()*context.values.element_size()
        return self.path.adapter.forward_text(ids, mask, use_projector=False, use_ports=False,
                                              use_spine=False, intrinsic_memory=context)

    def forward(self, ids, mask, batch, scope, **kwargs):
        with self.measured('backbone_forward', forwards=1, input_tokens=ids.numel()) as event:
            event['nonpadding_input_tokens'] = int(mask.sum())
            event['phase'] = 'uncached_prefix_with_full_prefill_each_step'
            event['source_feature_provider_forward'] = False
            result = self._forward(ids, mask, batch, scope, **kwargs)
            if not bool(torch.isfinite(result).all()): raise ValueError('nonfinite logits')
            return result

    def generate(self, row, batch, scope, *, control='contextual', wrong=None,
                 wrong_scope=None):
        self.sync(); started = time.monotonic_ns()
        if batch is not None:
            self.path.validate(batch, scope)
            if ''.join(s.text for s in batch.spans) != row['text']:
                raise ValueError('exact-text control must match the selected canonical spans')
        tokens = self.tokens(row, exact=control == 'exact_text'); prefix = len(tokens)
        if prefix+self.max_new_tokens > self.spec.context_tokens:
            raise ValueError('generation budget would truncate context')
        answer, stopped, first = [], False, None
        for _ in range(self.max_new_tokens):
            # Reserve a possible emitted token before the forward, including a
            # stop token. No refund if execution fails or EOS ends the run early.
            with self.measured('decode_token', generated_tokens=1):
                ids, mask = right_padded_prefix(tokens, self.spec, self.device)
                with torch.no_grad():
                    logits = self.forward(ids, mask, batch, scope, control=control,
                                          wrong=wrong, wrong_scope=wrong_scope)
                selected = logits[0, len(tokens)-1]
                if first is None: first = selected.detach().clone()
                token = int(selected.argmax()); answer.append(token)
                if token in self.spec.allowed_stop_token_ids:
                    stopped = True; break
                tokens.append(token)
        content = answer[:-1] if stopped else answer
        return {'raw_answer': self.backend.decode(content), 'token_ids': answer,
                'terminated': stopped, 'stop_token': answer[-1] if stopped else None,
                'generated_tokens': len(answer), 'prompt_tokens': prefix,
                # A run that exhausts the token budget without the bound stop
                # token is truncated, never a complete answer.
                'truncated': not stopped,
                'total_latency_ns': time.monotonic_ns()-started,
                'correct': score(row['answer'], self.backend.decode(content)),
                'false_recall': (row['answer'] == 'unknown'
                                 and not score('unknown', self.backend.decode(content))),
                'first_logits': first}

    # ------------------------------------------------------------------
    # Training, probes and evaluation

    def gradients(self):
        result = {}
        for name, module in self.path.modules().items():
            missing, finite, squared = [], True, 0.0
            for key, p in module.named_parameters():
                if p.grad is None: missing.append(key); continue
                finite = finite and bool(torch.isfinite(p.grad).all())
                squared += float(p.grad.detach().float().square().sum())
            result[name] = {'norm': squared**0.5, 'finite': finite, 'missing': missing}
        result['sites'] = {site: {'norm': sum(float(p.grad.float().square().sum())
                                              for p in port.parameters() if p.grad is not None)**0.5,
                                 'output_norm': float(port.output_projection.weight.grad.float().norm())
                                 if port.output_projection.weight.grad is not None else 0}
                           for site, port in self.path.reader.ports.items()}
        if any(p.grad is not None for p in self.path.adapter.base.parameters()):
            raise ValueError('original weight gradient')
        if any(v['missing'] or not v['finite'] for k, v in result.items() if k != 'sites'):
            raise ValueError('missing or nonfinite intended gradients')
        return result

    def update(self, row, source, scope):
        if self.fitting_closed: raise ValueError('fitting is closed')
        # Reserve backward/update before building a training graph. A failed
        # forward still consumes this attempted update; retry is not implicit.
        with self.measured('response_update', backwards=1, updates=1) as event:
            self.path.optimizer.zero_grad(set_to_none=True)
            batch = self.live(source, scope); encoding = self.encoding(row)
            logits = self.forward(encoding.input_ids, encoding.attention_mask, batch, scope)
            losses = complete_response_loss(logits, encoding, spec=self.spec)
            losses.loss.backward(); gradients = self.gradients()
            before = {k: tensor_digest(v) for k, v in self.path._tensors().items()}
            self.path.optimizer.step(); self.path.assert_optimizer_ownership()
            if any(not torch.isfinite(v).all() for v in self.path._tensors().values()):
                raise ValueError('nonfinite updated parameter')
            if any(torch.is_tensor(v) and not torch.isfinite(v).all()
                   for state in self.path.optimizer.state.values() for v in state.values()):
                raise ValueError('nonfinite optimizer moment')
            changed = [k for k, v in self.path._tensors().items() if tensor_digest(v) != before[k]]
            if not changed: raise ValueError('no actual parameter update')
            event.update(loss=float(losses.loss.detach()), word_loss=float(losses.word_loss.detach()),
                         eos_loss=float(losses.terminal_loss.detach()), gradients=gradients,
                         changed_tensors=changed)
            return dict(event)

    @contextmanager
    def layer_probe(self):
        rows, hooks = {}, []
        for site, port in self.path.reader.ports.items():
            def observe(module, args, output, site=site):
                hidden = args[0].detach().float(); delta = output.detach().float()
                rows[site] = {'gate': float(module.gate_logit.sigmoid()),
                              'hidden_rms': float(hidden.square().mean().sqrt()),
                              'residual_rms': float(delta.square().mean().sqrt()),
                              'projection_calls': 1, 'projection_cache': 'none'}
            hooks.append(port.register_forward_hook(observe))
        try:
            yield rows
        finally:
            for hook in hooks: hook.remove()

    def fitting_probes(self, rows, sources, scopes, stores):
        result = []
        # Explicit publication once per world, not per reader/forward/control.
        durable = []
        for row, scope, store in zip(rows, scopes, stores):
            corpus = self.new_release_corpus(store, scope)
            with self.measured('producer_release_rebuild', writes=4) as event:
                corpus.rebuild(self.path); event['corpus_cost'] = dict(corpus.cost)
            self.path.release_corpus = corpus
            durable.append(self.read(store, scope, row['question']))
        self.path.release_corpus = None  # batches below are validated with their corpus explicitly
        for i, (row, source, scope, store) in enumerate(zip(rows, sources, scopes, stores)):
            corpus = self.new_release_corpus(store, scope); self.path.release_corpus = corpus
            with torch.no_grad(): live = self.live(source, scope)
            encoding = self.encoding(row)
            with torch.no_grad(), self.layer_probe() as layers:
                live_logits = self.forward(encoding.input_ids, encoding.attention_mask, live, scope)
                stored_logits = self.forward(encoding.input_ids, encoding.attention_mask,
                                             durable[i], scope)
            parity = torch.equal(live.slots, durable[i].slots) and torch.equal(live_logits, stored_logits)
            if not parity: raise ValueError('live/durable round-trip differs')
            # Each control revalidates authentic E. For the deliberately wrong
            # world, use its live re-encoding (no cross-world release pointer).
            self.path.release_corpus = None
            with torch.no_grad(): wrong = self.live(sources[i ^ 1], scopes[i ^ 1])
            one = {'world_id': row['world_id'], 'round_trip_exact': parity, 'layers': layers,
                   'corpus_validation_cost': dict(corpus.cost), 'controls': {}}
            for control in CONTROLS:
                encoded = self.encoding(row, exact=control == 'exact_text')
                with torch.no_grad():
                    logits = self.forward(encoded.input_ids, encoded.attention_mask, live, scope,
                                          control=control, wrong=wrong, wrong_scope=scopes[i ^ 1])
                    losses = complete_response_loss(logits, encoded, spec=self.spec)
                position = len(self.tokens(row, exact=control == 'exact_text'))-1
                greedy = self.generate(row, live, scope, control=control, wrong=wrong,
                                       wrong_scope=scopes[i ^ 1])
                first = greedy.pop('first_logits')
                greedy.update(
                    word_loss=float(losses.word_loss), eos_loss=float(losses.terminal_loss),
                    first_token_correct=int(logits[0, position].argmax())
                    == self.backend.answer_ids(row['answer'])[0],
                    fit_generation_prefix_logits_exact=torch.equal(first, logits[0, position]),
                    logit_max_delta_from_correct=float(
                        (logits[0, position]-live_logits[0, len(self.tokens(row))-1]).abs().max()))
                one['controls'][control] = greedy
            result.append(one)
        self.path.release_corpus = None
        counts = {c: sum(r['controls'][c]['correct'] for r in result) for c in CONTROLS}
        # Shuffled values diagnose order dependence; they are not required to
        # destroy a correctly encoded fact. The stricter utility test reports it.
        passed = (len(result) == 8 and counts['contextual'] == 8
                  and all(r['controls']['contextual']['first_token_correct']
                          and r['controls']['contextual']['terminated']
                          and r['controls']['contextual']['fit_generation_prefix_logits_exact']
                          for r in result)
                  and all(counts['contextual'] > counts[c] for c in ('wrong', 'zero', 'absent')))
        return {'format': FORMAT+'.fit_probes', 'rows': result, 'correct': counts,
                'same_question_counterfactual_pairs': [[rows[i]['world_id'], rows[i+1]['world_id']]
                                                       for i in range(0, len(rows), 2)],
                'counterfactual_pairs_both_correct': sum(
                    result[i]['controls']['contextual']['correct']
                    and result[i+1]['controls']['contextual']['correct']
                    for i in range(0, len(result), 2)),
                'fit_memory_dependence_passed': passed, 'generalization_measured': False}

    def freeze(self):
        self.fitting_closed = True; self.path.freeze_readout()

    def evaluate(self, rows, batches, scopes):
        if not self.fitting_closed: raise ValueError('development labels require frozen fitting')
        decisions = []
        for i, (row, batch, scope) in enumerate(zip(rows, batches, scopes)):
            one = {'world_id': row['world_id'], 'controls': {}}
            for control in CONTROLS:
                result = self.generate(row, batch, scope, control=control,
                                       wrong=batches[i ^ 1], wrong_scope=scopes[i ^ 1])
                result.pop('first_logits'); one['controls'][control] = result
            decisions.append(one)
        counts = {c: sum(r['controls'][c]['correct'] for r in decisions) for c in CONTROLS}
        false = {c: sum(r['controls'][c]['false_recall'] for r in decisions) for c in CONTROLS}
        passed = (len(rows) == 8 and counts['contextual'] >= 6 and false['contextual'] == 0
                  and counts['contextual'] >= counts['exact_text']-1
                  and all(counts['contextual'] > counts[c]
                          for c in ('wrong', 'shuffled', 'zero', 'absent')))
        return {'rows': decisions, 'correct': counts, 'false_recall': false,
                'useful_recall_passed': passed,
                'generalization_scope': 'one_fixed_synthetic_family_and_seed_feasibility_only'}
