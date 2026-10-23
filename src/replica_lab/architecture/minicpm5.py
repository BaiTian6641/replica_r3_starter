"""Explicit MiniCPM5 native depth ports without hooks or model-global context.

Native blocks, attention/masking/positions, final norm and LM head stay intact.
This is a pinned Llama ABI integration, not an arbitrary Transformers adapter.
The reference re-prefills with no KV cache. Every pass has local ownership.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama import modeling_llama

from .contracts import (ComponentManifest, CopyRead, CoreState, DecisionFrame,
                        FastMemoryRead, IdentityRead, InjectFrame, MemoryRead,
                        PortSpec, RequestContext, SenseFrame, fingerprint)
from .core import CognitiveCore, FeedForwardCore, SOURCES
from .heads import TaskSuccessHead
from .ports import (FastMemoryAccessPort, FastWriteEncoder, MemoryAccessPort,
                    RecordCopyHead, RecordPointerHead, SensePort,
                    WorkspaceAccessPort)
from ..memory_system.writer import ContextualRecordWriter, TrainableRecordReranker
from ..replica.reader import IdentityReader


def native_abi_digest():
    paths = [Path(inspect.getfile(modeling_llama)), Path(inspect.getfile(create_causal_mask))]
    from transformers.modeling_layers import GradientCheckpointingLayer
    paths.append(Path(inspect.getfile(GradientCheckpointingLayer)))
    return fingerprint({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


class MiniCPM5Adapter(nn.Module):
    def __init__(self, base, manifest: ComponentManifest, regime='modules_only'):
        super().__init__()
        if base.config.model_type != 'llama':
            raise ValueError('E1/E2 supports only the pinned native Llama text ABI')
        if getattr(base.config, 'attention_dropout', 0) != 0:
            raise ValueError('Pinned E1/E2 feature caching requires zero native attention dropout')
        manifest.ports.validate(len(base.model.layers))
        if manifest.native_abi_digest != native_abi_digest():
            raise ValueError('Native implementation differs from the declared ABI')
        if manifest.cached_generation_supported:
            raise ValueError('Cached generation is not implemented by this reference')
        self.base, self.manifest, self.ports = base, manifest, manifest.ports
        h, d = base.config.hidden_size, self.ports.memory_width
        self.record_writer = ContextualRecordWriter(h, d, self.ports.slots_per_unit, self.ports.heads)
        self.memory_ports = nn.ModuleDict({str(i): MemoryAccessPort(h, d, self.ports.heads)
                                          for i in self.ports.injection_layers})
        self.sense_ports = nn.ModuleDict({str(i): SensePort(h, d) for i in
                                         sorted(set((*self.ports.injection_layers, self.ports.observer_layer)))})
        self.reranker = TrainableRecordReranker(d)
        self.core = self.ff_core = self.workspace_ports = self.task_head = None
        self.identity_reader = None
        self.fast_ports = self.fast_write_encoder = None
        self.copy_head = None
        if manifest.copy is not None:
            self.copy_head = (RecordPointerHead(h) if manifest.copy.mode == 'pointer'
                              else RecordCopyHead(h))
        if manifest.fast_memory is not None:
            spec = manifest.fast_memory
            self.fast_write_encoder = FastWriteEncoder(h, spec.key_size, spec.value_size)
            self.fast_ports = nn.ModuleDict({
                str(i): FastMemoryAccessPort(h, spec.heads, spec.key_size,
                                             spec.value_size)
                for i in self.ports.injection_layers})
        if manifest.core is not None:
            manifest.core.validate(self.ports)
            if d != manifest.core.width:
                raise ValueError('Core width must equal the record-memory width')
            self.core = CognitiveCore(manifest.core)
            self.ff_core = FeedForwardCore(manifest.core)
            self.workspace_ports = nn.ModuleDict({str(i): WorkspaceAccessPort(h, d, self.ports.heads)
                                                  for i in self.ports.injection_layers})
            self.task_head = TaskSuccessHead(manifest.core)
        if manifest.identity is not None:
            manifest.identity.validate(self.ports, manifest.core)
            self.identity_reader = IdentityReader(manifest.identity.width,
                                                  manifest.identity.heads)
        self.set_regime(regime)
        self._original_identity = None  # recorded after final device placement

    @property
    def _original_parameters(self):
        return dict(self.base.named_parameters())

    def set_regime(self, regime):
        if regime not in {'modules_only', 'full_augmented'}:
            raise ValueError('Declare modules_only or full_augmented')
        self.regime = regime
        for p in self.base.parameters():
            p.requires_grad_(regime == 'full_augmented')

    def seal_original_identity(self):
        """After native loading/device conversion, bind an unchanged-base export."""
        if self._original_identity is not None:
            raise RuntimeError('Original identity is already sealed')
        self._original_identity = self.parameter_stamp(original_only=True)

    def parameter_stamp(self, original_only=False):
        items = self.base.named_parameters() if original_only else self.named_parameters()
        parameters = tuple((n, id(p), p._version, tuple(p.shape), str(p.dtype), str(p.device),
                            p.data_ptr(), p.requires_grad) for n, p in items)
        buffers = tuple(('buffer:'+n, id(b), b._version, tuple(b.shape), str(b.dtype), str(b.device),
                         b.data_ptr()) for n,b in self.base.named_buffers())
        config = ('native_config', fingerprint(self.base.config.to_dict()),
                  self.base.config._attn_implementation,
                  str(getattr(self.base.model.rotary_emb, 'attention_scaling', None)))
        return (*parameters, *buffers, config)

    def producer_key(self):
        return fingerprint({'manifest': self.manifest.digest,
                            'base_parameters': self.parameter_stamp(original_only=True)})

    def memory_producer_key(self):
        return fingerprint({'features':self.producer_key(),'writer':[(n,id(p),p._version,str(p.dtype),str(p.device),p.data_ptr()) for n,p in self.record_writer.named_parameters()]})

    def sense_producer_key(self):
        return fingerprint({'features':self.producer_key(),'sense':[(n,id(p),p._version,str(p.dtype),str(p.device),p.data_ptr()) for n,p in self.sense_ports.named_parameters()]})

    def _core_component_stamp(self):
        stamp = {'features':self.producer_key()}
        prefixes = ['core','ff_core','workspace_ports','task_head','sense_ports','record_writer']
        if self.identity_reader is not None:
            prefixes.append('identity_reader')
        for prefix in prefixes:
            stamp[prefix] = [(n,id(p),p._version,str(p.dtype),str(p.device),p.data_ptr())
                             for n,p in getattr(self,prefix).named_parameters()]
        return stamp

    def core_producer_key(self):
        if self.core is None:
            raise ValueError('Manifest declares no core; core producer identity is undefined')
        return fingerprint(self._core_component_stamp())

    def copy_producer_key(self):
        if self.copy_head is None:
            raise ValueError('Manifest declares no copy branch')
        return fingerprint({'features': self.producer_key(),
                            'copy_head': [(n, id(p), p._version, str(p.dtype),
                                          str(p.device), p.data_ptr())
                                         for n, p in self.copy_head.named_parameters()]})

    def fast_producer_key(self):
        if self.fast_ports is None or self.fast_write_encoder is None:
            raise ValueError('Manifest declares no fast memory')
        return fingerprint({'features': self.producer_key(),
                            'fast_write_encoder': [(n, id(p), p._version, str(p.dtype),
                                                    str(p.device), p.data_ptr())
                                                   for n, p in self.fast_write_encoder.named_parameters()],
                            'fast_ports': [(n, id(p), p._version, str(p.dtype),
                                            str(p.device), p.data_ptr())
                                           for n, p in self.fast_ports.named_parameters()]})

    def encode_fast_observation(self, text, tokenizer, *, max_tokens: int = 256):
        """Encode one actual source page into fast-memory keys/values.

        The host must call this only for observed source/tool text, never for a
        generated proposal or a supervision target. The base embedding is a
        frozen feature producer; only the declared write projection is learned.
        No model forward, answer suffix or future observation is accepted.
        """
        if self.fast_write_encoder is None:
            raise ValueError('Manifest declares no fast write encoder')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('A nonempty observed source page is required')
        if type(max_tokens) is not int or not 0 < max_tokens <= 256:
            raise ValueError('max_tokens must be in 1..256')
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        if not ids:
            raise ValueError('Observed source page produced no tokens')
        token_ids = torch.tensor([ids[:max_tokens]], dtype=torch.long,
                                 device=self.base.get_input_embeddings().weight.device)
        embeddings = self.base.get_input_embeddings()(token_ids)
        mask = torch.ones(1, token_ids.shape[1], dtype=torch.bool,
                          device=token_ids.device)
        return self.fast_write_encoder(embeddings, mask)

    def build_core_state(self, sense: SenseFrame, memory: MemoryRead, task_features,
                         *, task_digest: str, revision: int, previous: CoreState | None = None,
                         enabled_inputs=SOURCES, implementation='recurrent',
                         identity: IdentityRead | None = None,
                         identity_enabled: bool = True) -> CoreState:
        """Run the declared core over known-input features; no labels are accepted.

        ``task_digest`` and ``revision`` bind the state to one exact journal
        revision; they are host-supplied, never derived from tensors.
        """
        if self.core is None:
            raise ValueError('Core state requires a core-bearing manifest')
        spec = self.manifest.core
        if implementation not in {'recurrent','feedforward'}:
            raise ValueError('Unknown core implementation')
        inputs = tuple(enabled_inputs)
        if inputs and (set(inputs) - set(SOURCES) or len(set(inputs)) != len(inputs)):
            raise ValueError('enabled_inputs must be a distinct subset of sense/memory/task')
        if not isinstance(task_digest, str) or not task_digest:
            raise ValueError('Core state requires a nonempty host task_digest')
        if type(revision) is not int or revision < 0:
            raise ValueError('Core state requires a nonnegative integer journal revision')
        features = torch.as_tensor(task_features,
                                   dtype=self.base.get_input_embeddings().weight.dtype
                                   if not torch.is_tensor(task_features) else task_features.dtype,
                                   device=self.base.get_input_embeddings().weight.device)
        if features.ndim == 1 and features.shape[0] == spec.task_features:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape != (1, spec.task_features) or not features.is_floating_point() \
                or not bool(torch.isfinite(features).all()):
            raise ValueError(f'task_features must be finite floating point [1, {spec.task_features}]')
        if previous is not None and (not isinstance(previous, CoreState)
                                     or previous.implementation != 'recurrent'):
            raise ValueError('Previous core must be a recurrent CoreState owned by this task')
        if previous is not None and previous.producer != self.core_producer_key():
            raise ValueError('Previous core producer differs from current core features')
        if (memory.slots.ndim != 4 or memory.slots.shape[0] != 1
                or memory.slots.shape[2] != self.ports.slots_per_unit
                or memory.slots.shape[3] != spec.width):
            raise ValueError('Memory must hold [1, records, slots, width]')
        if (memory.valid.dtype != torch.bool
                or tuple(memory.valid.shape) != tuple(memory.slots.shape[:3])
                or memory.valid.device != memory.slots.device):
            raise ValueError('Memory validity must be a matching boolean mask')
        summaries = [summary for _, summary in sense.summaries]
        if not summaries:
            raise ValueError('Sense frame carries no known-input summaries')
        sense_tokens = torch.cat(summaries, dim=1)
        sense_valid = torch.ones(1, sense_tokens.shape[1], dtype=torch.bool, device=sense_tokens.device)
        memory_slots = memory.slots.flatten(1, 2)
        memory_valid = memory.valid.flatten(1, 2)
        if identity is not None:
            if not isinstance(identity, IdentityRead):
                raise ValueError('identity must be an IdentityRead or None')
            if self.identity_reader is None:
                raise ValueError('Adapter manifest declares no identity reader')
            if (identity.slots.shape[2] != spec.width
                    or identity.slots.device != features.device
                    or not identity.slots.is_floating_point()):
                raise ValueError('Identity slots must match workspace width and device')
        if implementation == 'recurrent':
            workspace, valid = self.core(sense_tokens, sense_valid, memory_slots, memory_valid,
                                         features, previous=None if previous is None else previous.workspace,
                                         enabled_inputs=inputs)
            steps = spec.steps
        else:
            # Mask before normalization so an all-invalid pool contributes
            # exactly zero instead of inheriting padded/garbage bytes.
            pooled_sense = sense_tokens.masked_fill(~sense_valid.unsqueeze(-1), 0).sum(dim=1) / \
                sense_valid.sum(dim=1, keepdim=True).clamp_min(1)
            pooled_memory = memory_slots.masked_fill(~memory_valid.unsqueeze(-1), 0).sum(dim=1) / \
                memory_valid.sum(dim=1, keepdim=True).clamp_min(1)
            workspace, valid = self.ff_core(pooled_sense, pooled_memory, features)
            steps = 0
        identity_digest = None
        if identity is not None:
            workspace = workspace + self.identity_reader(
                workspace, identity.slots, identity.valid, enabled=bool(identity_enabled))
            identity_digest = identity.context_digest
        return CoreState(workspace, valid, features, self.core_producer_key(), task_digest,
                         revision, steps, implementation, identity_digest)

    def predict_success(self, core_state: CoreState):
        """Monitor logit for a core state; detached, never a task fact or authority.

        The head's inputs are detached inside the head, and this boundary also
        detaches the returned logit so a monitor read can never create a graph
        that outlives the decision. A separate training driver may call
        ``task_head`` directly for head-only supervision.
        """
        if self.task_head is None:
            raise ValueError('Success head requires a core-bearing manifest')
        if not isinstance(core_state, CoreState):
            raise ValueError('predict_success requires an owned CoreState')
        core_state.assert_immutable()
        if core_state.producer != self.core_producer_key():
            raise ValueError('Core producer differs from current native/core features')
        return self.task_head(core_state.task_features, core_state.workspace,
                              core_state.valid).detach()

    def new_named_parameters(self):
        prefixes = ['record_writer', 'memory_ports', 'sense_ports', 'reranker']
        if self.core is not None:
            prefixes += ['core', 'ff_core', 'workspace_ports', 'task_head']
        if self.identity_reader is not None:
            prefixes += ['identity_reader']
        if self.fast_ports is not None:
            prefixes += ['fast_write_encoder', 'fast_ports']
        if self.copy_head is not None:
            prefixes += ['copy_head']
        for prefix in prefixes:
            yield from ((f'{prefix}.{n}', p) for n, p in getattr(self, prefix).named_parameters())

    def _parent_named_parameters(self):
        """Exact E1/E2 new-tensor inventory, without any E3 additions."""
        for prefix in ('record_writer', 'memory_ports', 'sense_ports', 'reranker'):
            yield from ((f'{prefix}.{n}', p) for n, p in getattr(self, prefix).named_parameters())

    def _component_inventory(self):
        prefixes = ['record_writer', 'memory_ports', 'sense_ports', 'reranker']
        if self.core is not None:
            prefixes += ['core', 'ff_core', 'workspace_ports', 'task_head']
        if self.identity_reader is not None:
            prefixes += ['identity_reader']
        if self.fast_ports is not None:
            prefixes += ['fast_write_encoder', 'fast_ports']
        if self.copy_head is not None:
            prefixes += ['copy_head']
        return {prefix: [f'{prefix}.{n}' for n, _ in getattr(self, prefix).named_parameters()]
                for prefix in prefixes}

    def optimizer_groups(self, base_lr, new_lr):
        groups = []
        for name, pairs, lr in (('original', self.base.named_parameters(), base_lr),
                                ('new', self.new_named_parameters(), new_lr)):
            params = [p for _, p in pairs if p.requires_grad]
            if params:
                groups.append({'params': params, 'lr': lr, 'group_name': name})
        owned = [id(p) for g in groups for p in g['params']]
        if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in self.parameters() if p.requires_grad}:
            raise RuntimeError('Incomplete optimizer ownership')
        return groups

    def parameter_audit(self):
        result = {'regime': self.regime}
        for group, pairs in (('original', self.base.named_parameters()), ('new', self.new_named_parameters())):
            pairs = list(pairs)
            result[group] = {'tensors': len(pairs), 'parameters': sum(p.numel() for _, p in pairs),
                             'trainable': sum(p.numel() for _, p in pairs if p.requires_grad),
                             'names': [n for n, p in pairs if p.requires_grad]}
        return result

    def gradient_audit(self):
        result = {}
        for group, pairs in (('original', self.base.named_parameters()), ('new', self.new_named_parameters())):
            pairs = list(pairs)
            result[group] = {'eligible': sum(p.requires_grad for _, p in pairs),
                'present': [n for n, p in pairs if p.grad is not None],
                'nonzero': [n for n, p in pairs if p.grad is not None and bool(p.grad.ne(0).any())],
                'nonfinite': [n for n, p in pairs if p.grad is not None and not bool(torch.isfinite(p.grad).all())]}
        return result

    def _inputs(self, input_ids, valid):
        if (input_ids.ndim != 2 or input_ids.dtype != torch.long or not input_ids.shape[1]
                or valid.shape != input_ids.shape or valid.device != input_ids.device):
            raise ValueError('Nonempty [batch,tokens] int64 IDs and matching masks required')
        if input_ids.device != self.base.get_input_embeddings().weight.device:
            raise ValueError('Model computation and input must use the same device')
        if bool(((input_ids < 0) | (input_ids >= self.base.config.vocab_size)).any()):
            raise ValueError('Token outside native vocabulary')
        if bool(((valid != 0) & (valid != 1)).any()) or not bool(valid.any(dim=1).all()):
            raise ValueError('Binary masks with a known token in every row required')
        return valid.bool()

    def _native(self, ids, valid, *, frame=None, stop_layer=None, capture=False):
        valid = self._inputs(ids, valid)
        hidden = self.base.model.embed_tokens(ids)
        # Match native generation's left-padding convention; no cache is reused.
        positions = valid.long().cumsum(-1) - 1
        positions = positions.masked_fill(~valid, 1)
        causal = create_causal_mask(config=self.base.config, inputs_embeds=hidden,
                                   attention_mask=valid.long(), past_key_values=None,
                                   position_ids=positions)
        rotary = self.base.model.rotary_emb(hidden, position_ids=positions)
        summaries = []
        for i, layer in enumerate(self.base.model.layers):
            # __call__ retains native nonreentrant checkpoint behavior.
            hidden = layer(hidden, attention_mask=causal, position_ids=positions,
                           position_embeddings=rotary, past_key_values=None, use_cache=False)
            if not isinstance(hidden, torch.Tensor):
                raise TypeError('Unexpected native decoder ABI')
            if capture and str(i) in self.sense_ports:
                summary, _ = self.sense_ports[str(i)](hidden, valid)
                summaries.append((i, summary))
            if frame is not None and str(i) in self.memory_ports and frame.branch_enabled('memory'):
                view = InjectFrame(frame, i)
                memory = frame.memory
                hidden = hidden + self.memory_ports[str(i)](
                    hidden, memory.slots.flatten(1, 2), memory.valid.flatten(1, 2), enabled=view.enabled)
            if (frame is not None and frame.core is not None and frame.branch_enabled('workspace')
                    and str(i) in self.workspace_ports):
                view = InjectFrame(frame, i)
                core = frame.core
                hidden = hidden + self.workspace_ports[str(i)](
                    hidden, core.workspace, core.valid, enabled=view.enabled)
            if (frame is not None and frame.fast is not None and frame.branch_enabled('fast')
                    and str(i) in self.fast_ports):
                view = InjectFrame(frame, i)
                hidden = hidden + self.fast_ports[str(i)](
                    hidden, frame.fast.weights, enabled=view.enabled)
            if i == stop_layer:
                return hidden, summaries
        return self.base.model.norm(hidden), summaries

    def encode_observations(self, input_ids, valid):
        """Source-only prefix features. This method never accepts target labels."""
        if input_ids.shape[1] > self.ports.source_tokens:
            raise ValueError('Chunk source records before encoding; no implicit truncation')
        return self._native(input_ids, valid, stop_layer=self.ports.source_layer)[0]

    def sense(self, prompt_ids, prompt_mask):
        # E1 exposes these summaries; the learned recurrent core is E3.
        _, summaries = self._native(prompt_ids, prompt_mask, capture=True)
        return SenseFrame(tuple(summaries), prompt_ids.clone(), prompt_mask.bool().clone(), self.sense_producer_key())

    def make_frame(self, decision_id, evidence, memory: MemoryRead, sense: SenseFrame,
                   enabled_ports=None, core: CoreState | None = None,
                   enabled_branches=None, fast: FastMemoryRead | None = None,
                   copy: CopyRead | None = None):
        if memory.slots.ndim != 4 or memory.slots.shape[0] != 1:
            raise ValueError('One owned request per bank; batch requests explicitly later')
        if memory.slots.shape[1:] != (len(memory.spans), self.ports.slots_per_unit, self.ports.memory_width):
            raise ValueError('Record identity/slot layout mismatch')
        if memory.valid.shape != memory.slots.shape[:3] or len(memory.spans) > self.ports.max_units:
            raise ValueError('Memory unit budget/mask mismatch')
        if (memory.valid.dtype != torch.bool or memory.slots.device != sense.prompt_ids.device
                or memory.valid.device != memory.slots.device
                or memory.slots.dtype != self.base.get_input_embeddings().weight.dtype
                or not bool(torch.isfinite(memory.slots).all())):
            raise ValueError('Memory tensor dtype/device/finite contract mismatch')
        if any((s.record_id, s.record_hash) not in evidence.records for s in memory.spans):
            raise ValueError('Memory source not in evidence pin')
        if memory.producer != self.memory_producer_key():
            raise ValueError('Memory producer differs from current native features/writer')
        if sense.producer != self.sense_producer_key():
            raise ValueError('Sense producer differs from current parameters')
        enabled = self.ports.injection_layers if enabled_ports is None else tuple(enabled_ports)
        if not set(enabled) <= set(self.ports.injection_layers):
            raise ValueError('Undeclared port')
        if core is not None:
            if self.core is None:
                raise ValueError('Frame core requires a core-bearing manifest')
            spec = self.manifest.core
            if tuple(core.task_features.shape) != (1, spec.task_features):
                raise ValueError(f'Core task features must have shape [1, {spec.task_features}]')
            if not isinstance(core.task_digest, str) or not core.task_digest:
                raise ValueError('Frame core requires a nonempty exact task digest')
            if core.producer != self.core_producer_key():
                raise ValueError('Core producer differs from current native/core features')
        if fast is not None:
            if self.fast_ports is None or self.manifest.fast_memory is None:
                raise ValueError('Fast frame requires a fast-memory manifest')
            spec = self.manifest.fast_memory
            if tuple(fast.weights.shape) != (1, spec.heads, spec.key_size, spec.value_size):
                raise ValueError('Fast weights shape differs from the declared spec')
            if fast.producer != self.fast_producer_key():
                raise ValueError('Fast producer differs from current native/port features')
        if copy is not None:
            if self.copy_head is None or self.manifest.copy is None:
                raise ValueError('Copy frame requires a copy-bearing manifest')
            if copy.token_ids.shape[1] > self.manifest.copy.max_tokens:
                raise ValueError('Copy candidates exceed the declared bound')
            if copy.producer != self.copy_producer_key():
                raise ValueError('Copy producer differs from current native features/head')
        default_branches = ['memory']
        if core is not None:
            default_branches.append('workspace')
        if fast is not None:
            default_branches.append('fast')
        if copy is not None:
            default_branches.append('copy')
        branches = tuple(default_branches) if enabled_branches is None \
            else tuple(enabled_branches)
        if 'workspace' in branches and core is None:
            raise ValueError('Workspace branch requires an owned CoreState')
        if 'fast' in branches and fast is None:
            raise ValueError('Fast branch requires an owned FastMemoryRead')
        if 'copy' in branches and copy is None:
            raise ValueError('Copy branch requires an owned CopyRead')
        return DecisionFrame(decision_id, evidence, memory, sense, self.manifest.digest,
                             self.parameter_stamp(), enabled, core, fast, copy, branches)

    def pointer_scores(self, frame, candidate_ids, token_mask, candidate_mask):
        """Score exact source-token spans from the prompt's final hidden state."""
        if self.manifest.copy is None or self.manifest.copy.mode != 'pointer':
            raise ValueError('Pointer scoring requires a pointer copy manifest')
        self._check_frame(frame)
        if self.copy_head is None or frame.copy is None:
            raise ValueError('Pointer scoring requires an owned copy frame')
        ids, mask = frame.sense.prompt_ids, frame.sense.prompt_mask
        hidden, _ = self._native(ids, mask, frame=frame)
        query = hidden[:, -1]
        if token_mask.shape != candidate_ids.shape or token_mask.dtype != torch.bool:
            raise ValueError('pointer token mask must be boolean matching candidates')
        embeddings = self.base.get_input_embeddings()(candidate_ids)
        return self.copy_head(query, embeddings, token_mask, candidate_mask)

    def _check_frame(self, frame):
        frame.assert_immutable()
        if frame.manifest_digest != self.manifest.digest or frame.parameter_stamp != self.parameter_stamp():
            raise RuntimeError('Model/extension changed; rebuild the DecisionFrame')

    def forward_with_frame(self, input_ids, frame, *, attention_mask=None, labels=None,
                           logits_to_keep=0, past_key_values=None):
        self._check_frame(frame)
        if past_key_values is not None:
            raise ValueError('External/native cache reuse is not qualified')
        mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask
        known = frame.sense.prompt_ids.shape[1]
        if (input_ids.shape[0] != 1 or input_ids.shape[1] < known
                or not torch.equal(input_ids[:, :known], frame.sense.prompt_ids)
                or not torch.equal(mask[:, :known].bool(), frame.sense.prompt_mask)):
            raise ValueError('Action input must retain the exact known prefix and mask')
        if labels is not None:
            if labels.shape != input_ids.shape or bool((labels[:, :known] != -100).any()):
                raise ValueError('Sense prefix may not contain supervised targets')
            if logits_to_keep != 0:
                raise ValueError('Training requires all response logits')
        hidden, _ = self._native(input_ids, mask, frame=frame)
        logits = self.base.lm_head(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)
        if (frame.copy is not None and frame.branch_enabled('copy')
                and self.copy_head is not None and frame.copy.token_ids.shape[1] > 0
                and self.manifest.copy is not None and self.manifest.copy.mode == 'logit_bias'):
            copy_hidden = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
            embeddings = self.base.get_input_embeddings()(frame.copy.token_ids)
            bias = self.copy_head(copy_hidden, embeddings, frame.copy.token_ids,
                                  frame.copy.mask)
            if bias.shape[-1] > logits.shape[-1]:
                raise ValueError('Copy candidate token exceeds the LM vocabulary')
            if bias.shape[-1] < logits.shape[-1]:
                bias = torch.nn.functional.pad(bias, (0, logits.shape[-1] - bias.shape[-1]))
            logits = logits + bias
        loss = (self.base.loss_function(logits=logits, labels=labels, vocab_size=self.base.config.vocab_size)
                if labels is not None else None)
        if loss is not None and loss.requires_grad:
            def check_backward(gradient):
                self._check_frame(frame)
                return gradient
            loss.register_hook(check_backward)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def generate_action(self, request: RequestContext, *, verify, max_new_tokens=16,
                        eos_token_id=None, banned_token_ids=None):
        """Raw greedy proposal. Host verification runs before each outward segment/token.

        ``banned_token_ids`` is an interface constraint used only for a declared
        action-space experiment; it masks tokens, never supplies a repair value.
        """
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('Positive generation budget required')
        banned = () if banned_token_ids is None else tuple(banned_token_ids)
        if any(type(token) is not int or token < 0
               or token >= self.base.config.vocab_size for token in banned):
            raise ValueError('Banned token IDs must be within the LM vocabulary')
        banned = tuple(sorted(set(banned)))
        if banned and len(banned) >= self.base.config.vocab_size:
            raise ValueError('Cannot ban the entire LM vocabulary')
        request.assert_ready()
        stops = self.base.config.eos_token_id if eos_token_id is None else eos_token_id
        stops = set(stops if isinstance(stops, (tuple, list)) else ([] if stops is None else [stops]))
        modes = [(m, m.training) for m in self.modules()]
        try:
            self.eval()
            with torch.no_grad():
                ids = request.frame.sense.prompt_ids.clone()
                mask = request.frame.sense.prompt_mask.clone()
                for _ in range(max_new_tokens):
                    verify(request)
                    output = self.forward_with_frame(ids, request.frame, attention_mask=mask, logits_to_keep=1)
                    logits = output.logits[:, -1]
                    if banned:
                        logits = logits.clone()
                        logits[:, list(banned)] = -1e30
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError('Nonfinite generation logits')
                    token = logits.argmax(-1, keepdim=True)
                    ids = torch.cat((ids, token), dim=1)
                    mask = torch.cat((mask, torch.ones_like(token, dtype=torch.bool)), dim=1)
                    if token.item() in stops:
                        break
                verify(request)
                request.generated_ids = tuple(ids[0, request.frame.sense.prompt_ids.shape[1]:].tolist())
                return ids
        finally:
            for module, training in modes:
                module.training = training

    def save_extension(self, directory):
        """No native save_pretrained call and no base copy, even temporarily."""
        from safetensors.torch import save_file
        if (self.regime != 'modules_only' or self._original_identity is None
                or self._original_identity != self.parameter_stamp(original_only=True)):
            raise RuntimeError('Extension export requires the unchanged sealed original; full-base export is separate')
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        weights = {n: p.detach().cpu().contiguous() for n, p in self.new_named_parameters()}
        save_file(weights, str(path / 'extensions.safetensors'))
        if self.identity_reader is not None:
            fmt = 'replica.minicpm5.extensions.v3'
        elif self.core is not None or self.fast_ports is not None or self.copy_head is not None:
            fmt = 'replica.minicpm5.extensions.v2'
        else:
            fmt = 'replica.minicpm5.extensions.v1'
        receipt = {'format':fmt,'manifest':self.manifest.to_dict(),
                   'manifest_digest':self.manifest.digest,'regime':self.regime,
                   'weights_sha256':hashlib.sha256((path/'extensions.safetensors').read_bytes()).hexdigest(),
                   'parameters':self.parameter_audit(),'base_saved':False,
                   'notes':'Weights only; no optimizer/RNG training-resume state. Reload against validated local sealed source.'}
        if fmt != 'replica.minicpm5.extensions.v1':
            receipt['components'] = self._component_inventory()
        (path/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt

    def load_extension(self, directory, accept_parent_v1=False,
                       port_capacity_migration=False):
        from safetensors.torch import load_file
        if (self.regime != 'modules_only' or self._original_identity is None
                or self._original_identity != self.parameter_stamp(original_only=True)):
            raise RuntimeError('Extension reload requires the unchanged sealed original')
        path = Path(directory)
        receipt = json.loads((path/'manifest.json').read_text())
        fmt = receipt.get('format')
        if fmt in ('replica.minicpm5.extensions.v2', 'replica.minicpm5.extensions.v3'):
            if set(receipt) != {'format','manifest','manifest_digest','regime','weights_sha256',
                                'parameters','base_saved','notes','components'}:
                raise ValueError('Incompatible or changed extension manifest/bytes')
            if (receipt.get('manifest_digest') != self.manifest.digest
                    or receipt.get('manifest') != json.loads(json.dumps(self.manifest.to_dict()))
                    or hashlib.sha256((path/'extensions.safetensors').read_bytes()).hexdigest()
                    != receipt.get('weights_sha256')):
                raise ValueError('Incompatible or changed extension manifest/bytes')
            if receipt.get('components') != json.loads(json.dumps(self._component_inventory())):
                raise ValueError('Extension component inventory differs')
            expected = dict(self.new_named_parameters())
        elif fmt == 'replica.minicpm5.extensions.v1':
            shared = json.loads(json.dumps(self.manifest.shared_dict()))
            receipt_manifest = receipt.get('manifest')
            capacity_migration = False
            if receipt_manifest != shared:
                if not port_capacity_migration:
                    raise ValueError('Incompatible or changed extension manifest/bytes')
                if receipt.get('manifest_digest') != fingerprint(receipt_manifest):
                    raise ValueError('Parent manifest digest is inconsistent')
                if not isinstance(receipt_manifest, dict):
                    raise ValueError('Parent manifest is malformed')
                parent_ports = dict(receipt_manifest.get('ports') or {})
                target_ports = dict(shared.get('ports') or {})
                if (set(parent_ports) != set(target_ports)
                        or {k: v for k, v in parent_ports.items() if k != 'max_units'}
                        != {k: v for k, v in target_ports.items() if k != 'max_units'}
                        or type(parent_ports.get('max_units')) is not int
                        or type(target_ports.get('max_units')) is not int
                        or target_ports['max_units'] < parent_ports['max_units']):
                    raise ValueError('Port-capacity migration changes more than max_units')
                for key in set(shared) - {'ports'}:
                    if receipt_manifest.get(key) != shared[key]:
                        raise ValueError('Port-capacity migration changes another manifest field')
                capacity_migration = True
            if (not capacity_migration and receipt.get('manifest') != shared) or \
                    hashlib.sha256((path/'extensions.safetensors').read_bytes()).hexdigest() \
                    != receipt.get('weights_sha256'):
                raise ValueError('Incompatible or changed extension manifest/bytes')
            expected = dict(self._parent_named_parameters())
            if (self.manifest.core is not None or self.fast_ports is not None
                    or self.copy_head is not None or self.identity_reader is not None):
                if not accept_parent_v1:
                    raise ValueError('Loading the E1/E2 parent v1 extension requires accept_parent_v1=True')
        else:
            raise ValueError('Incompatible or changed extension manifest/bytes')
        weights = load_file(str(path/'extensions.safetensors'))
        if set(weights) != set(expected) or any(weights[n].shape != p.shape or weights[n].dtype != p.dtype
                                              for n,p in expected.items()):
            raise ValueError('Extension tensor inventory differs')
        with torch.no_grad():
            for n,p in expected.items(): p.copy_(weights[n].to(p.device))
        if fmt == 'replica.minicpm5.extensions.v1' and (
                self.manifest.core is not None or self.fast_ports is not None
                or self.identity_reader is not None or port_capacity_migration):
            parent = set(expected)
            migration = {
                'accepted_parent_v1': True, 'copied': sorted(parent),
                'fresh_init': sorted(n for n,_ in self.new_named_parameters() if n not in parent)}
            if port_capacity_migration:
                migration.update({
                    'port_capacity_only': True,
                    'parent_max_units': int(receipt_manifest['ports']['max_units']),
                    'target_max_units': int(shared['ports']['max_units'])})
            return {**receipt, 'migration': migration}
        return receipt
