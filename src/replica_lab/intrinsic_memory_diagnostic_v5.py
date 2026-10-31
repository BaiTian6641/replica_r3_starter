"""Fit-only semantic/stopping diagnosis, shared by separate 2/9/36 packets.

The original model is frozen. Ordinary response CE includes EOS. There is no P
update, teacher distillation, automatic tuning, promotion or write inside forward.
Controls are explicit probe interventions, never admitted as authentic evidence.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch

from .baseline_competence import score
from .contextual_completion_objective_v1 import encode_complete_response, complete_response_loss
from .contextual_memory import canonical_spans, tensor_digest
from .contextual_memory_v5 import PendingContextualRead, ContextualMemoryError
from .contextual_release_v5 import ProducerReleaseCorpus
from .intrinsic_memory_budget_v5 import atomic
from .intrinsic_memory_acceptance_v13 import assess_response

FORMAT = 'replica.intrinsic_memory.diagnostic.v5'
CONTROLS = ('contextual', 'exact_text', 'wrong', 'shuffled', 'zero', 'absent')
PROMPT = 'Reply with only the answer word. If unsupported reply unknown.\n{evidence}Question: {question}\nAnswer:'


def prompt(row, *, exact=False):
    return PROMPT.format(evidence=('Evidence: '+row['text']+'\n') if exact else '', question=row['question'])


def right_padded_prefix(tokens, spec, device):
    if not tokens or len(tokens) > spec.context_tokens:
        raise ValueError('prefix is empty or would truncate evidence')
    ids = torch.full((1,spec.context_tokens), spec.pad_token_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    ids[0,:len(tokens)] = torch.tensor(tokens, device=device); mask[0,:len(tokens)] = 1
    return ids, mask


def logits_digest(logits):
    """Bounded CPU chunks; retain no full logit or original-weight export."""
    h = hashlib.sha256(); flat = logits.detach().reshape(-1)
    for start in range(0,flat.numel(),1024*1024):
        part = flat[start:start+1024*1024].float().cpu().contiguous()
        h.update(memoryview(part.numpy()))
    return h.hexdigest()


class DiagnosticEngine:
    def __init__(self, path, backend, spec, ledger, *, max_new_tokens, output):
        self.path, self.backend, self.spec, self.ledger = path, backend, spec, ledger
        self.max_new_tokens, self.output = max_new_tokens, Path(output)
        self.output.mkdir(exist_ok=True, parents=True)
        self.stage = None
        self.events = []
        self.fitting_closed = False

    @property
    def device(self): return next(self.path.adapter.base.parameters()).device

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

    def tokens(self, row, *, exact=False):
        tokens=self.backend.encode(prompt(row, exact=exact))
        if exact:
            self.path.cost['fallback_calls']+=1
            self.path.cost['fallback_bytes']+=len(row['text'].encode())
            self.path.cost['fallback_tokens']+=len(self.backend.answer_ids(row['text']))
        return tokens

    def encoding(self, row, *, exact=False):
        return encode_complete_response(self.tokens(row,exact=exact), self.backend.answer_ids(row['answer']),
                                       spec=self.spec, device=self.device)

    def live(self, source, scope):
        with self.measured('source_encoding', source_encodes=len(source.spans)):
            return self.path.training_from_stored(source, scope=scope)

    def read(self, store, scope, question):
        with self.measured('retrieval', retrievals=1, writes=1) as event:
            batch = self.path.read(store, question, scope=scope, wait_seconds=1)
            # A caller wait is bounded. At most five waits for this declared
            # attempt; the outer supervisor also bounds uninterruptible I/O.
            for _ in range(4):
                if not isinstance(batch, PendingContextualRead): break
                batch = self.path.resume(store,batch,scope=scope,wait_seconds=1)
            if isinstance(batch, PendingContextualRead):
                event['pending'] = batch.__dict__
                raise ContextualMemoryError('cold recall incomplete after five bounded waits')
            event['costs'] = getattr(batch,'read_costs',{})
            if self.path.release_corpus is not None:
                event['release_authority_costs']=dict(self.path.release_corpus.cost)
            return batch

    def _forward(self, ids, mask, batch, scope, *, control='contextual', wrong=None, wrong_scope=None):
        if control not in CONTROLS: raise ValueError('undeclared control')
        if control in ('absent','exact_text') or batch is None:
            return self.path.forward(ids,mask,None,scope=scope,enabled=False)
        if control == 'contextual': return self.path.forward(ids,mask,batch,scope=scope)
        # A measured intervention after authentic-source validation. It cannot
        # bypass admission, manufacture a durable record or change its authority.
        self.path.validate(batch,scope)
        slots = batch.slots
        if control == 'wrong':
            if wrong is None or wrong_scope is None: raise ValueError('wrong control source missing')
            self.path.validate(wrong,wrong_scope)
            slots = wrong.slots
        elif control == 'shuffled': slots = slots.flip(0)
        order = self.path.order(torch.arange(len(slots),device=ids.device))
        slots = slots+order
        if control == 'zero': slots = torch.zeros_like(slots)
        padding = torch.zeros((self.path.spec.slots-len(slots),256),device=slots.device)
        values = torch.cat((slots,padding))[None,None].expand(ids.shape[0],ids.shape[1],-1,-1)
        mask_slots = (torch.arange(self.path.spec.slots,device=ids.device)<len(slots))
        mask_slots = mask_slots[None,None].expand(values.shape[:3])
        def validate():
            self.path.validate(batch,scope)
            if control == 'wrong': self.path.validate(wrong,wrong_scope)
        context = self.path.reader.snapshot(values,mask_slots,validate=validate)
        self.path.cost['snapshot_copies'] += 1
        self.path.cost['snapshot_bytes'] += context.values.numel()*context.values.element_size()
        return self.path.adapter.forward_text(ids,mask,use_projector=False,use_ports=False,
                                              use_spine=False,intrinsic_memory=context)

    def forward(self, ids, mask, batch, scope, **kwargs):
        with self.measured('backbone_forward', forwards=1, input_tokens=ids.numel()) as event:
            result = self._forward(ids,mask,batch,scope,**kwargs)
            event['nonpadding_input_tokens'] = int(mask.sum())
            event['phase'] = 'uncached_prefix_with_full_prefill_each_step'
            if not bool(torch.isfinite(result).all()): raise ValueError('nonfinite logits')
            return result

    def generate(self, row, batch, scope, *, control='contextual', wrong=None, wrong_scope=None):
        self.sync();started=time.monotonic_ns()
        if batch is not None:
            self.path.validate(batch,scope)
            if ''.join(s.text for s in batch.spans)!=row['text']:
                raise ValueError('exact-text control must match the selected canonical spans')
        tokens = self.tokens(row,exact=control=='exact_text'); prefix = len(tokens)
        if prefix+self.max_new_tokens > self.spec.context_tokens:
            raise ValueError('generation budget would truncate context')
        answer, stopped, first = [], False, None
        for _ in range(self.max_new_tokens):
            # Reserve a possible emitted token before the forward, including a
            # stop token. No refund if execution fails or EOS ends early.
            with self.measured('decode_token', generated_tokens=1):
                ids, mask = right_padded_prefix(tokens,self.spec,self.device)
                with torch.no_grad():
                    logits = self.forward(ids,mask,batch,scope,control=control,wrong=wrong,wrong_scope=wrong_scope)
                selected = logits[0,len(tokens)-1]
                if first is None: first = selected.detach().clone()
                token = int(selected.argmax()); answer.append(token)
                if token in self.spec.allowed_stop_token_ids:
                    stopped = True; break
                tokens.append(token)
        content = answer[:-1] if stopped else answer
        return {'raw_answer': self.backend.decode(content), 'token_ids': answer,
                'terminated': stopped, 'stop_token': answer[-1] if stopped else None,
                'generated_tokens': len(answer), 'prompt_tokens': prefix, 'truncated': False,
                'total_latency_ns':time.monotonic_ns()-started,
                'correct': score(row['answer'],self.backend.decode(content)),
                'false_recall': row['answer']=='unknown' and not score('unknown',self.backend.decode(content)),
                'first_logits': first}

    def gradients(self):
        result = {}
        for name,module in self.path.modules().items():
            missing, finite, squared = [], True, 0.0
            for key,p in module.named_parameters():
                if p.grad is None: missing.append(key); continue
                finite = finite and bool(torch.isfinite(p.grad).all())
                squared += float(p.grad.detach().float().square().sum())
            result[name] = {'norm': squared**0.5,'finite':finite,'missing':missing}
        result['sites'] = {site:{'norm':sum(float(p.grad.float().square().sum()) for p in port.parameters() if p.grad is not None)**0.5,
                                'output_norm':float(port.output_projection.weight.grad.float().norm()) if port.output_projection.weight.grad is not None else 0}
                           for site,port in self.path.reader.ports.items()}
        if any(p.grad is not None for p in self.path.adapter.base.parameters()): raise ValueError('original weight gradient')
        if any(v['missing'] or not v['finite'] for k,v in result.items() if k!='sites'):
            raise ValueError('missing or nonfinite intended gradients')
        return result

    def update(self, row, source, scope):
        if self.fitting_closed: raise ValueError('fitting is closed')
        # Reserve backward/update before building a training graph. A failed
        # forward still consumes this attempted update; retry is not implicit.
        with self.measured('response_update', backwards=1, updates=1) as event:
            self.path.optimizer.zero_grad(set_to_none=True)
            batch = self.live(source,scope); encoding = self.encoding(row)
            logits = self.forward(encoding.input_ids,encoding.attention_mask,batch,scope)
            losses = complete_response_loss(logits,encoding,spec=self.spec)
            losses.loss.backward(); gradients = self.gradients()
            before = {k: tensor_digest(v) for k,v in self.path._tensors().items()}
            self.path.optimizer.step(); self.path.assert_optimizer_ownership()
            if any(not torch.isfinite(v).all() for v in self.path._tensors().values()):
                raise ValueError('nonfinite updated parameter')
            if any(torch.is_tensor(v) and not torch.isfinite(v).all()
                   for state in self.path.optimizer.state.values() for v in state.values()):
                raise ValueError('nonfinite optimizer moment')
            changed = [k for k,v in self.path._tensors().items() if tensor_digest(v)!=before[k]]
            if not changed: raise ValueError('no actual parameter update')
            event.update(loss=float(losses.loss.detach()),word_loss=float(losses.word_loss.detach()),
                         eos_loss=float(losses.terminal_loss.detach()),gradients=gradients,changed_tensors=changed)
            return dict(event)

    @contextmanager
    def layer_probe(self):
        rows, hooks = {}, []
        for site,port in self.path.reader.ports.items():
            def observe(module,args,output,site=site):
                hidden = args[0].detach().float(); delta = output.detach().float()
                rows[site] = {'gate':float(module.gate_logit.sigmoid()),
                    'hidden_rms':float(hidden.square().mean().sqrt()),
                    'residual_rms':float(delta.square().mean().sqrt()),
                    'projection_calls':1,'projection_cache':'none'}
            hooks.append(port.register_forward_hook(observe))
        try: yield rows
        finally:
            for hook in hooks: hook.remove()

    def fitting_probes(self, rows, sources, scopes, stores):
        result = []
        # Explicit publication once per world, not per reader/forward/control.
        durable = []
        for row,scope,store in zip(rows,scopes,stores):
            corpus = ProducerReleaseCorpus(store,scope)
            with self.measured('producer_release_rebuild',source_encodes=1,writes=1) as event:
                corpus.rebuild(self.path); event['corpus_cost']=dict(corpus.cost)
            self.path.release_corpus = corpus
            durable.append(self.read(store,scope,row['question']))
        self.path.release_corpus = None  # batches below are validated with their corpus explicitly
        for i,(row,source,scope,store) in enumerate(zip(rows,sources,scopes,stores)):
            corpus = ProducerReleaseCorpus(store,scope); self.path.release_corpus=corpus
            with torch.no_grad(): live = self.live(source,scope)
            encoding = self.encoding(row)
            with torch.no_grad(), self.layer_probe() as layers:
                live_logits = self.forward(encoding.input_ids,encoding.attention_mask,live,scope)
                stored_logits = self.forward(encoding.input_ids,encoding.attention_mask,durable[i],scope)
            parity = torch.equal(live.slots,durable[i].slots) and torch.equal(live_logits,stored_logits)
            if not parity: raise ValueError('live/durable round-trip differs')
            # Each control revalidates authentic E. For the deliberately wrong
            # world, use its live re-encoding (no cross-world release pointer).
            self.path.release_corpus=None
            with torch.no_grad(): wrong=self.live(sources[i^1],scopes[i^1])
            one={'world_id':row['world_id'],'round_trip_exact':parity,'layers':layers,
                 'corpus_validation_cost':dict(corpus.cost),'controls':{}}
            for control in CONTROLS:
                encoded=self.encoding(row,exact=control=='exact_text')
                with torch.no_grad():
                    logits=self.forward(encoded.input_ids,encoded.attention_mask,live,scope,
                        control=control,wrong=wrong,wrong_scope=scopes[i^1])
                    losses=complete_response_loss(logits,encoded,spec=self.spec)
                position=len(self.tokens(row,exact=control=='exact_text'))-1
                greedy=self.generate(row,live,scope,control=control,wrong=wrong,wrong_scope=scopes[i^1])
                first=greedy.pop('first_logits')
                greedy.update(word_loss=float(losses.word_loss),eos_loss=float(losses.terminal_loss),
                    first_token_correct=int(logits[0,position].argmax())==self.backend.answer_ids(row['answer'])[0],
                    fit_generation_prefix_logits_exact=torch.equal(first,logits[0,position]),
                    logit_max_delta_from_correct=float((logits[0,position]-live_logits[0,len(self.tokens(row))-1]).abs().max()))
                one['controls'][control]=greedy
            result.append(one)
        self.path.release_corpus=None
        counts={c:sum(r['controls'][c]['correct'] for r in result) for c in CONTROLS}
        # Shuffled values diagnose order dependence; they are not required to
        # destroy a correctly encoded fact. The stricter utility test reports it.
        passed=(len(result)==8 and counts['contextual']==8
                and all(r['controls']['contextual']['first_token_correct'] and r['controls']['contextual']['terminated']
                        and r['controls']['contextual']['fit_generation_prefix_logits_exact'] for r in result)
                and all(counts['contextual']>counts[c] for c in ('wrong','zero','absent')))
        return {'format':FORMAT+'.fit_probes','rows':result,'correct':counts,
                'same_question_counterfactual_pairs':[[rows[i]['world_id'],rows[i+1]['world_id']] for i in range(0,len(rows),2)],
                'counterfactual_pairs_both_correct':sum(result[i]['controls']['contextual']['correct']
                    and result[i+1]['controls']['contextual']['correct'] for i in range(0,len(result),2)),
                'fit_memory_dependence_passed':passed,'generalization_measured':False}

    def freeze(self):
        self.fitting_closed=True; self.path.freeze_readout()

    def evaluate(self, rows, batches, scopes):
        if not self.fitting_closed: raise ValueError('development labels require frozen fitting')
        decisions=[]
        for i,(row,batch,scope) in enumerate(zip(rows,batches,scopes)):
            one={'world_id':row['world_id'],'controls':{}}
            for control in CONTROLS:
                result=self.generate(row,batch,scope,control=control,wrong=batches[i^1],wrong_scope=scopes[i^1])
                result.pop('first_logits')
                assessment = assess_response(
                    row['answer'], result['raw_answer'], control=control,
                    terminated=result['terminated'], truncated=result['truncated'])
                result.update(assessment.as_dict())
                one['controls'][control] = result
            decisions.append(one)
        counts = {c: sum(r['controls'][c]['correct'] for r in decisions) for c in CONTROLS}
        false = {c: sum(r['controls'][c]['false_recall'] for r in decisions) for c in CONTROLS}
        complete = {c: sum(r['controls'][c]['complete'] for r in decisions) for c in CONTROLS}
        complete_correct = {c: sum(r['controls'][c]['complete_correct'] for r in decisions)
                            for c in CONTROLS}
        completion_passed = all(
            row['controls'][control]['complete']
            for row in decisions for control in CONTROLS)
        passed = (completion_passed and len(rows) == 8 and counts['contextual'] >= 6
                  and false['contextual'] == 0
                  and counts['contextual'] >= counts['exact_text']-1
                  and all(counts['contextual'] > counts[c]
                          for c in ('wrong', 'shuffled', 'zero', 'absent')))
        return {'rows': decisions, 'correct': counts, 'false_recall': false,
                'complete': complete, 'complete_correct': complete_correct,
                'completion_passed': completion_passed,
                'useful_recall_passed': passed,
                'generalization_scope': 'one_fixed_synthetic_family_and_seed_feasibility_only'}
