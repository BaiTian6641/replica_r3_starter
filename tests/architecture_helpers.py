"""Small offline native fixtures shared by explicit-architecture contract tests."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from replica_lab.architecture.contracts import (
    ComponentManifest, EvidencePin, MemoryRead, PortSpec, SourceSpan, fingerprint,
)
from replica_lab.architecture.minicpm5 import MiniCPM5Adapter, native_abi_digest


def tiny_adapter(regime="modules_only"):
    """Random local tensors only; never resolves a Hugging Face repository."""
    torch.manual_seed(103)
    torch.set_num_threads(1)
    config = LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
        tie_word_embeddings=False, pad_token_id=0, eos_token_id=63,
    )
    config._attn_implementation = "eager"
    ports = PortSpec(injection_layers=(0, 2), observer_layer=3, source_layer=0,
                     memory_width=16, slots_per_unit=4, max_units=4,
                     source_tokens=16, heads=4)
    manifest = ComponentManifest(
        source_revision="offline-tiny-native-test",
        source_seal_digest=fingerprint("tiny-original"),
        tokenizer_digest=fingerprint("tiny-tokenizer"),
        native_config_digest=fingerprint(config.to_dict()),
        native_abi_digest=native_abi_digest(), ports=ports,
    )
    return MiniCPM5Adapter(LlamaForCausalLM(config), manifest, regime)


def synthetic_frame(adapter, *, enabled_ports=None, prompt_ids=None,
                    prompt_mask=None, empty=False, all_masked=False):
    """Construct explicitly synthetic provenance; no ledger or hidden answer."""
    if prompt_ids is None:
        prompt_ids = torch.tensor([[1, 2, 3]])
    if prompt_mask is None:
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    source_ids = torch.tensor([[7, 8, 9], [10, 11, 12]])
    source_mask = torch.ones_like(source_ids, dtype=torch.bool)
    spans = tuple(SourceSpan(
        record_id=f"record-{i}", record_hash=fingerprint(["record", i]),
        source_hash=fingerprint(["source", i]), char_start=0, char_end=3,
        token_start=0, token_end=3, token_ids=tuple(ids.tolist()),
        span_hash=fingerprint(ids.tolist()),
    ) for i, ids in enumerate(source_ids))
    if empty:
        spans = ()
        slots = adapter.base.get_input_embeddings().weight.new_zeros(
            1, 0, adapter.ports.slots_per_unit, adapter.ports.memory_width)
        valid = torch.zeros(1, 0, adapter.ports.slots_per_unit, dtype=torch.bool)
    else:
        features = adapter.encode_observations(source_ids, source_mask)
        slots, valid = adapter.record_writer(features.unsqueeze(0), source_mask.unsqueeze(0))
        if all_masked:
            valid = torch.zeros_like(valid)
    evidence = EvidencePin(
        ledger_id="offline-test-ledger", epoch=0, subject_id="test-subject",
        branch_id="test-branch", purpose="architecture-test", as_of=10,
        mode="current", snapshot_at=None,
        records=tuple((s.record_id, s.record_hash) for s in spans),
    )
    memory = MemoryRead(spans, slots, valid, adapter.memory_producer_key())
    sense = adapter.sense(prompt_ids, prompt_mask)
    return adapter.make_frame("test-decision", evidence, memory, sense,
                              enabled_ports=enabled_ports)


def action_batch(frame):
    """Known prefix followed by two teacher targets, with response-only loss."""
    suffix = torch.tensor([[4, 5]], dtype=torch.long)
    ids = torch.cat((frame.sense.prompt_ids, suffix), dim=1)
    mask = torch.cat((frame.sense.prompt_mask, torch.ones_like(suffix, dtype=torch.bool)), dim=1)
    labels = ids.clone()
    labels[:, :frame.sense.prompt_ids.shape[1]] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}
