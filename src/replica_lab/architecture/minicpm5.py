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

from .contracts import (ComponentManifest, DecisionFrame, InjectFrame, MemoryRead,
                        PortSpec, RequestContext, SenseFrame, fingerprint)
from .ports import MemoryAccessPort, SensePort
from ..memory_system.writer import ContextualRecordWriter, TrainableRecordReranker


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

    def new_named_parameters(self):
        for prefix in ('record_writer', 'memory_ports', 'sense_ports', 'reranker'):
            yield from ((f'{prefix}.{n}', p) for n, p in getattr(self, prefix).named_parameters())

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
            if frame is not None and str(i) in self.memory_ports:
                view = InjectFrame(frame, i)
                memory = frame.memory
                hidden = hidden + self.memory_ports[str(i)](
                    hidden, memory.slots.flatten(1, 2), memory.valid.flatten(1, 2), enabled=view.enabled)
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
                   enabled_ports=None):
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
        return DecisionFrame(decision_id, evidence, memory, sense, self.manifest.digest,
                             self.parameter_stamp(), enabled)

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
        loss = (self.base.loss_function(logits=logits, labels=labels, vocab_size=self.base.config.vocab_size)
                if labels is not None else None)
        if loss is not None and loss.requires_grad:
            def check_backward(gradient):
                self._check_frame(frame)
                return gradient
            loss.register_hook(check_backward)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def generate_action(self, request: RequestContext, *, verify, max_new_tokens=16, eos_token_id=None):
        """Raw greedy proposal. Host verification runs before each outward segment/token."""
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('Positive generation budget required')
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
                    if not bool(torch.isfinite(output.logits).all()):
                        raise ValueError('Nonfinite generation logits')
                    token = output.logits[:, -1].argmax(-1, keepdim=True)
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
        receipt = {'format':'replica.minicpm5.extensions.v1','manifest':self.manifest.to_dict(),
                   'manifest_digest':self.manifest.digest,'regime':self.regime,
                   'weights_sha256':hashlib.sha256((path/'extensions.safetensors').read_bytes()).hexdigest(),
                   'parameters':self.parameter_audit(),'base_saved':False,
                   'notes':'Weights only; no optimizer/RNG training-resume state. Reload against validated local sealed source.'}
        (path/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt

    def load_extension(self, directory):
        from safetensors.torch import load_file
        if (self.regime != 'modules_only' or self._original_identity is None
                or self._original_identity != self.parameter_stamp(original_only=True)):
            raise RuntimeError('Extension reload requires the unchanged sealed original')
        path = Path(directory)
        receipt = json.loads((path/'manifest.json').read_text())
        if (receipt.get('format') != 'replica.minicpm5.extensions.v1'
                or receipt.get('manifest_digest') != self.manifest.digest
                or receipt.get('manifest') != json.loads(json.dumps(self.manifest.to_dict()))
                or hashlib.sha256((path/'extensions.safetensors').read_bytes()).hexdigest() != receipt.get('weights_sha256')):
            raise ValueError('Incompatible or changed extension manifest/bytes')
        weights = load_file(str(path/'extensions.safetensors'))
        expected = dict(self.new_named_parameters())
        if set(weights) != set(expected) or any(weights[n].shape != p.shape or weights[n].dtype != p.dtype
                                              for n,p in expected.items()):
            raise ValueError('Extension tensor inventory differs')
        with torch.no_grad():
            for n,p in expected.items(): p.copy_(weights[n].to(p.device))
        return receipt
