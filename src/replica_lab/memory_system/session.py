"""Connected E1/E2 source-to-port path with host checks outside neural forwards."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import uuid
from contextlib import contextmanager, nullcontext

import torch
from safetensors.torch import load_file, save_file

from .ledger import EligibilityScope, RecordLedger
from .index import RetrievalIndex, DenseRecordVector
from ..architecture.contracts import (EvidencePin, MemoryRead, PendingWrite,
                                     RequestContext, SourceSpan, TaskGraph, fingerprint)


def pin_selection(selection):
    return EvidencePin(selection.ledger_id, selection.epoch, selection.subject_id,
                       selection.branch_id, selection.purpose, selection.as_of,
                       selection.mode, selection.snapshot_at,
                       tuple((r.record_id, r.record_hash) for r in selection.records))


@contextmanager
def frozen_parameters(*modules):
    """Temporarily freeze optimizer-visible parameters, restoring flags exactly.

    Used when a sensor graph must pass through the core while ``train_core`` is
    false: gradients may traverse core computations to reach sensors, but the
    core/task-encoder parameters must not accumulate gradients or stay frozen
    after this request.
    """
    flags = [(p, p.requires_grad) for module in modules for p in module.parameters()
             if p.requires_grad]
    try:
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


class MemorySession:
    """One trusted host's source admission/retrieval and request-owned neural state.

    Frozen native features can be cached in D1; trainable writer slots never are.
    Persistent inference latents are content/version scoped and limited to 128
    entries. Existing historical experiment machinery is not imported here.
    """
    def __init__(self, adapter, tokenizer, ledger: RecordLedger, *, cache_dir=None,
                 max_candidate_units=64):
        self.adapter, self.tokenizer, self.ledger = adapter, tokenizer, ledger
        self.index = RetrievalIndex(ledger)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if type(max_candidate_units) is not int or max_candidate_units < adapter.ports.max_units:
            raise ValueError('Candidate budget must cover the memory unit budget')
        self.max_candidate_units = max_candidate_units
        self._features = {}
        self._feature_producer = None
        self._training_leases = set()
        self.cache_stats = {'feature_hits':0, 'feature_misses':0, 'latent_hits':0, 'latent_misses':0}

    @property
    def device(self):
        return self.adapter.base.get_input_embeddings().weight.device

    def spans(self, records):
        result = []
        for record in records:
            encoded = self.tokenizer(record.text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = encoded['input_ids'], encoded['offset_mapping']
            for start in range(0, len(ids), self.adapter.ports.source_tokens):
                stop = min(len(ids), start + self.adapter.ports.source_tokens)
                first, last = offsets[start][0], offsets[stop-1][1]
                if not 0 <= first <= last <= len(record.text):
                    raise ValueError('Tokenizer source offsets are invalid')
                result.append(SourceSpan(record.record_id, record.record_hash, record.source_hash,
                    first, last, start, stop, tuple(ids[start:stop]),
                    hashlib.sha256(record.text[first:last].encode()).hexdigest()))
        if len(result) > self.max_candidate_units:
            raise ValueError('Candidate chunk budget exceeded; request narrower sources, never silently truncate')
        return tuple(result)

    def _feature(self, span):
        producer = self.adapter.producer_key()
        if self._feature_producer != producer:
            self._features.clear()
            self._feature_producer = producer
        frozen = not any(p.requires_grad for p in self.adapter.base.parameters())
        key = span.key
        if frozen and key in self._features:
            self.cache_stats['feature_hits'] += 1
            return self._features[key]
        self.cache_stats['feature_misses'] += 1
        ids = torch.tensor([span.token_ids], dtype=torch.long, device=self.device)
        if frozen:
            with torch.no_grad():
                result = self.adapter.encode_observations(ids, torch.ones_like(ids, dtype=torch.bool)).detach()
            if len(self._features) >= self.max_candidate_units:
                self._features.pop(next(iter(self._features)))
            self._features[key] = result
        else:
            result = self.adapter.encode_observations(ids, torch.ones_like(ids, dtype=torch.bool))
        return result

    def _writer_hash(self):
        h = hashlib.sha256()
        for name, p in self.adapter.record_writer.named_parameters():
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def _latent(self, span, pin, training, writer_hash):
        cache_meta = {'format':'replica.record_latent.v1', 'span':asdict(span), 'pin':asdict(pin),
                      'manifest':self.adapter.manifest.digest,'writer':writer_hash}
        # Stable across process restart; in-process feature cache additionally binds tensor versions.
        key = fingerprint(cache_meta)
        can_cache = (not training and self.adapter.regime == 'modules_only'
                     and self.adapter._original_identity == self.adapter.parameter_stamp(original_only=True)
                     and self.cache_dir is not None)
        path = self.cache_dir / f'{key}.safetensors' if can_cache else None
        if path and path.exists():
            meta_path = path.with_suffix('.json')
            metadata = json.loads(meta_path.read_text())
            if (metadata['key'] != key or metadata['payload_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()
                    or metadata['meta'] != json.loads(json.dumps(cache_meta))):
                raise ValueError('Latent cache identity/content mismatch')
            tensors = load_file(str(path))
            slots, valid = tensors['slots'].to(self.device), tensors['valid'].to(self.device)
            if (slots.shape != (1,1,self.adapter.ports.slots_per_unit,self.adapter.ports.memory_width)
                    or valid.shape != slots.shape[:3] or valid.dtype != torch.bool
                    or slots.dtype != self.adapter.base.get_input_embeddings().weight.dtype
                    or not bool(torch.isfinite(slots).all())):
                raise ValueError('Latent cache tensor contract mismatch')
            self.cache_stats['latent_hits'] += 1
            return slots, valid
        self.cache_stats['latent_misses'] += 1
        features = self._feature(span)
        valid = torch.ones(features.shape[:2], dtype=torch.bool, device=self.device)
        slots, mask = self.adapter.record_writer(features[:,None], valid[:,None])
        if path:
            save_file({'slots':slots.detach().cpu().contiguous(), 'valid':mask.cpu().contiguous()}, str(path))
            path.with_suffix('.json').write_text(json.dumps({'key':key,'meta':cache_meta,
                'payload_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')
            self._prune_cache(pin)
        return slots, mask

    def _prune_cache(self, pin):
        if not self.cache_dir: return
        keep = []
        for meta_path in self.cache_dir.glob('*.json'):
            try: meta = json.loads(meta_path.read_text())
            except (ValueError, OSError): continue
            if meta.get('meta',{}).get('format') != 'replica.record_latent.v1': continue
            previous = meta['meta']['pin']
            if previous['ledger_id'] != pin.ledger_id: continue
            payload = meta_path.with_suffix('.safetensors')
            if previous['epoch'] != pin.epoch:
                payload.unlink(missing_ok=True); meta_path.unlink()
            else: keep.append(meta_path)
        for meta_path in sorted(keep, key=lambda p:p.stat().st_mtime_ns)[:-128]:
            meta_path.with_suffix('.safetensors').unlink(missing_ok=True);meta_path.unlink()

    def verify(self, request):
        try:
            request.assert_ready()
            expected = EligibilityScope(request.frame.evidence.subject_id, request.frame.evidence.branch_id,
                request.frame.evidence.purpose, request.frame.evidence.as_of, request.frame.evidence.mode,
                request.frame.evidence.snapshot_at)
            self.index.verify_selection(request.selection, expected_scope=expected)
            if pin_selection(request.selection) != request.frame.evidence:
                raise ValueError('Selection differs from frame evidence')
            self.adapter._check_frame(request.frame)
        except Exception:
            request.invalidate()
            self._features.clear()
            self._prune_cache(EvidencePin(self.ledger.ledger_id,self.ledger.epoch,'','','',0,'',None,()))
            raise

    def prepare(self, *, prompt_ids, prompt_mask, query, scope: EligibilityScope,
                task: TaskGraph, training=False, selection=None, rerank=False,
                enabled_ports=None, train_sense=False, task_features=None,
                previous_core=None, core_impl='recurrent',
                enabled_inputs=('sense','memory','task'), enabled_branches=None,
                train_core=False, task_digest=None, revision=None, fast_state=None):
        if core_impl not in {'recurrent', 'feedforward', 'none'}:
            raise ValueError("core_impl must be 'recurrent', 'feedforward' or 'none'")
        if core_impl == 'none':
            if task_features is not None:
                raise ValueError("core_impl='none' does not accept task_features")
            if previous_core is not None:
                raise ValueError("core_impl='none' does not accept a previous core")
            branches = ('memory',) if enabled_branches is None else tuple(enabled_branches)
            if 'workspace' in branches:
                raise ValueError("core_impl='none' cannot enable the workspace branch")
            enabled_branches = branches
        selection = selection or self.index.retrieve(query, **asdict(scope), limit=self.adapter.ports.max_units)
        self.index.verify_selection(selection, expected_scope=scope)
        pin = pin_selection(selection)
        self._prune_cache(pin)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            # The sense pass keeps autograd when the sensors, the core or the
            # workspace readers must receive the action loss; otherwise it is
            # built under no_grad to avoid a second native graph.
            with (torch.enable_grad() if training and (train_sense or train_core) else torch.no_grad()):
                sense = self.adapter.sense(prompt_ids, prompt_mask)
            spans = self.spans(selection.records)
            writer_hash = self._writer_hash() if not training and self.cache_dir else 'training-no-slot-cache'
            encoded = [self._latent(s, pin, training, writer_hash) for s in spans]
            if encoded:
                slots, valid = torch.cat([v[0] for v in encoded],dim=1), torch.cat([v[1] for v in encoded],dim=1)
                if rerank:
                    query_features = dict(sense.summaries)[self.adapter.ports.injection_layers[0]][:,0]
                    scores = self.adapter.reranker(query_features, slots.mean(dim=2), valid.any(dim=2))
                    if not bool(torch.isfinite(scores[valid.any(dim=2)]).all()):
                        raise ValueError('Nonfinite candidate score')
                    order = torch.argsort(scores[0], descending=True, stable=True)[:self.adapter.ports.max_units]
                else:
                    order = torch.arange(min(len(spans),self.adapter.ports.max_units),device=self.device)
                # Overflow is explicit selection, not discarded during source encoding.
                spans = tuple(spans[i] for i in order.tolist())
                slots, valid = slots[:,order], valid[:,order]
            else:
                slots = self.adapter.base.get_input_embeddings().weight.new_zeros(
                    1,0,self.adapter.ports.slots_per_unit,self.adapter.ports.memory_width)
                valid = torch.zeros(slots.shape[:3],dtype=torch.bool,device=self.device)
            memory = MemoryRead(spans,slots,valid,self.adapter.memory_producer_key())
            core = None
            if core_impl != 'none' and self.adapter.core is not None:
                if task_features is None:
                    raise ValueError('The core requires host-computed exact task_features; none were supplied')
                if not isinstance(task_digest, str) or not task_digest or type(revision) is not int or revision < 0:
                    raise ValueError('The core requires the exact-journal task_digest and revision')
                # A live previous graph would span decisions; the prior workspace
                # is always taken as a task-owned value.
                previous = previous_core.detached() if previous_core is not None else None
                sensor_only = training and train_sense and not train_core
                freeze = frozen_parameters(self.adapter.core, self.adapter.ff_core) \
                    if sensor_only else nullcontext()
                with (torch.enable_grad() if training and (train_core or sensor_only)
                      else torch.no_grad()), freeze:
                    # In the sensor-only arm the action loss traverses a frozen
                    # core to the sensors; core/task-encoder parameters never
                    # accumulate gradients and their flags are restored exactly.
                    core = self.adapter.build_core_state(
                        sense, memory, task_features, task_digest=task_digest,
                        revision=revision, previous=previous,
                        enabled_inputs=tuple(enabled_inputs), implementation=core_impl)
            frame = self.adapter.make_frame(uuid.uuid4().hex,pin,memory,sense,enabled_ports,
                                            core=core,enabled_branches=enabled_branches,
                                            fast=fast_state)
        request = RequestContext(uuid.uuid4().hex,task,frame,selection)
        self.verify(request)
        return request

    def forward(self, request, input_ids, *, labels=None, attention_mask=None):
        self.verify(request)
        return self.adapter.forward_with_frame(input_ids,request.frame,labels=labels,attention_mask=attention_mask)

    def ranking_loss(self, request, preferred_record_id):
        """Explicit record-ranking supervision; hard top-k does not train from answer loss.

        Use train_sense=True to train the query sensor as well as the reranker/writer.
        The preferred ID is a loss target, never a source/writer input. All units of
        the target record share its target mass; the bank must contain a distractor.
        """
        self.verify(request)
        memory, sense = request.frame.memory, request.frame.sense
        target = torch.tensor([[s.record_id == preferred_record_id for s in memory.spans]],
                              dtype=torch.bool, device=self.device)
        valid = memory.valid.any(dim=2)
        if not bool((target & valid).any()) or not bool((~target & valid).any()):
            raise ValueError('Ranking requires an eligible preferred record and a distractor')
        query = dict(sense.summaries)[self.adapter.ports.injection_layers[0]][:,0]
        scores = self.adapter.reranker(query, memory.slots.mean(dim=2), valid)
        loss = -(scores.log_softmax(dim=-1)[target & valid]).mean()
        if loss.requires_grad:
            frame = request.frame
            def check_backward(gradient):
                # MPS autograd hooks may run on a worker thread. Keep storage
                # checks on the host before/after backward; hooks are tensor-only.
                self.adapter._check_frame(frame)
                return gradient
            loss.register_hook(check_backward)
        return loss

    def backward(self, request, loss):
        self.verify(request)
        loss.backward()
        self.verify(request)

    def generate(self, request, **kwargs):
        return self.adapter.generate_action(request,verify=self.verify,**kwargs)

    @contextmanager
    def training_request(self, **kwargs):
        request = self.prepare(training=True, **kwargs)
        self._training_leases.add(request.request_id);request.state='training'
        try:
            yield request
            self.verify(request)  # before caller performs the optimizer step
        finally:
            self._training_leases.discard(request.request_id);request.release()

    def exact_text(self, request, span):
        self.verify(request)
        if span not in request.frame.memory.spans:
            raise PermissionError('Exact fetch must refer to a selected memory unit')
        return self.ledger.fetch_exact(span.record_id, **asdict(request.selection.scope),
            start=span.char_start,end=span.char_end,expected_hash=span.record_hash,
            expected_epoch=request.selection.epoch)

    def commit_observation(self, pending: PendingWrite, *, authorized_subject, authorized_branch,
                           source_authorized=False):
        if self._training_leases:
            raise RuntimeError('Finish backward and release owned frames before source commits')
        if pending.expected_epoch != self.ledger.epoch:
            raise ValueError('Observation proposal has stale epoch')
        if pending.event_id != pending.record.record_id:
            raise ValueError('Event ID must be the immutable ledger transaction ID')
        changed = self.ledger.admit(pending.record,authorized_subject=authorized_subject,
            authorized_branch=authorized_branch,source_authorized=source_authorized)
        if changed:self._features.clear()
        return changed

    def dense_candidates(self, query, scope, *, limit=16, lexical_weight=1.0):
        """Native source-prefix mean features, only for eligible records.

        This is a deterministic dense baseline, not a learned semantic retriever.
        ``lexical_weight=0`` gives the pure-dense comparison; the default keeps
        the historical hybrid behavior. A separately trainable reranker is
        applied in prepare(rerank=True).
        """
        snapshot = self.ledger.snapshot(scope)
        all_units = self.spans(snapshot.records)  # bound total work before encoding
        by_record = {r.record_id: tuple(s for s in all_units if s.record_id == r.record_id) for r in snapshot.records}
        vectors={}
        with torch.no_grad():
            query_ids=self.tokenizer(query,add_special_tokens=False)['input_ids']
            ids=torch.tensor([query_ids],dtype=torch.long,device=self.device)
            q=self.adapter.encode_observations(ids,torch.ones_like(ids,dtype=torch.bool)).mean(dim=1)[0]
            for record in snapshot.records:
                units=by_record[record.record_id]
                if units:
                    value=torch.cat([self._feature(s) for s in units],dim=1).mean(dim=1)[0]
                    vectors[record.record_id]=DenseRecordVector(record.record_hash,tuple(value.tolist()),self.adapter.producer_key())
        if snapshot.epoch != self.ledger.epoch:
            raise ValueError('Sources changed during dense encoding')
        return self.index.retrieve(query,**asdict(scope),limit=limit,dense_query=q.tolist(),
            dense_vectors=vectors,dense_producer_id=self.adapter.producer_key(),
            lexical_weight=lexical_weight,dense_weight=1.0)
