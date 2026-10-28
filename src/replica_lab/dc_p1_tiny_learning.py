"""Nondegenerate tiny-Llama/selected-expert fixture for software learning audits."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from replica_lab.architecture.plastic import PlasticKnowledgePort, fixed_projection
from replica_lab.dc_p1_native_runner import NativeP1Backend
from replica_lab.nested_learning import ExpertBank
from replica_lab.nested_text import initialize_null_plastic
from replica_lab.testing import tiny_tokenizer


class TinyAdapter(nn.Module):
    def __init__(self, base: LlamaForCausalLM, width: int, sites: tuple[int, ...]):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.plastic_ports = nn.ModuleDict({
            str(site): PlasticKnowledgePort(
                base.config.hidden_size, width,
                decoder=fixed_projection(base.config.hidden_size, width, 20261027),
                gain=0.1)
            for site in sites})

    def parameter_stamp(self, original_only=False):
        return tuple((name, parameter.data_ptr(), parameter._version)
                     for name, parameter in self.base.named_parameters())


def make_tiny_learning_view(*, phrases=(), texts=(), lr: float = 0.01,
                            seed: int = 7, max_new: int = 4,
                            epsilon: float = 1e-8):
    torch.manual_seed(seed)
    width = 4
    sites = (13, 27)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=8, intermediate_size=16,
        num_hidden_layers=28, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0))
    adapter = TinyAdapter(base, width, sites)
    bank = ExpertBank(
        expert_count=1, width=width, hidden_width=8, seed=0, lr=lr,
        eps=epsilon)
    initialize_null_plastic(bank)
    view = SimpleNamespace(
        torch=torch, adapter=adapter, bank=bank,
        projection=adapter.plastic_ports["13"].decoder.detach(),
        sites=sites, device=torch.device("cpu"),
        tokenizer=tiny_tokenizer(phrases=phrases, texts=texts),
        stops=(1, 130073), eos_id=1, max_new=max_new, max_prompt=32,
        max_parity=32, max_teaching=64,
        _teaching_positions={"p1v2src_dev_00": (0, 1, 2),
                             "p1v2src_dev_01": (0, 1, 2)},
        _reader=None)
    view.optimizer = bank.optimizer
    view._state_sha = NativeP1Backend._state_sha.__get__(view, NativeP1Backend)
    view._render = NativeP1Backend._render.__get__(view, NativeP1Backend)
    view._features = NativeP1Backend._features.__get__(view, NativeP1Backend)
    view._native_forward = NativeP1Backend._native_forward.__get__(view, NativeP1Backend)
    view._logits = NativeP1Backend._logits.__get__(view, NativeP1Backend)
    view.parent_state = {
        name: parameter.detach().clone()
        for name, parameter in bank.expert(0).named_parameters()}
    view.parent_sha256 = view._state_sha(view.parent_state)
    view.decoder_digest = adapter.plastic_ports["13"].decoder_digest
    return view
