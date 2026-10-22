"""Correctness-first greedy decoding with immutable decision-boundary context.

One request owns one batch-1 context. Each token recomputes the entire causal
prefix with caching disabled. Generated answers never enter the source writer
or prompt workspace, which are constructed once from original inputs. This is
a reference inference route, not an optimized serving/cache implementation.
"""
from __future__ import annotations

import torch


def _ids(value, name: str, *, device=None, vocabulary: int) -> None:
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1
            or value.dtype != torch.long):
        raise ValueError(f"{name} must be an int64 tensor with shape [1, positions]")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on the prompt device")
    if bool(((value < 0) | (value >= vocabulary)).any()):
        raise ValueError(f"{name} contains IDs outside the native vocabulary")


def _mask(value, ids, name: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.shape != ids.shape
            or value.device != ids.device
            or value.dtype not in {torch.bool, torch.uint8, torch.int8,
                                   torch.int16, torch.int32, torch.int64}
            or bool(((value != 0) & (value != 1)).any())):
        raise ValueError(f"{name} must be a binary mask matching its IDs and device")
    return value.bool()


def _eos_ids(value, vocabulary: int) -> set[int]:
    if value is None:
        return set()
    values = value if isinstance(value, list) else [value]
    if any(type(token) is not int or not 0 <= token < vocabulary for token in values):
        raise ValueError("eos_token_id must be an in-vocabulary integer or list of integers")
    return set(values)


def generate_fixed_context(model, *, prompt_ids, prompt_mask, memory_ids=None,
                           memory_mask=None, max_new_tokens=8,
                           eos_token_id=None) -> torch.Tensor:
    """Return prompt plus greedy answer tokens, stopping at EOS or the budget.

    Supports one unpadded or left-padded prompt. ``None`` EOS inherits the native
    model config; an explicit empty list disables EOS stopping. An omitted
    memory becomes an empty permitted-memory context. No labels or teacher-
    forced answer inputs are accepted. Caller-owned IDs/masks and existing
    gradients/weights are unchanged. Per-module training modes are restored.

    A context already owned by training or another request is rejected without
    releasing it. New owned context is released after success or any exception.
    Native baseline/full arms use the same uncached decoding loop.
    """
    if getattr(model, "_context", None) is not None:
        raise RuntimeError("Previous batch is still in flight; generation cannot acquire its context")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if getattr(model, "regime", None) not in {"baseline", "full", "modules_only", "full_augmented"}:
        raise ValueError("Generation requires a declared native Llama adapter regime")
    base = model.base
    vocabulary = base.config.vocab_size
    _ids(prompt_ids, "prompt_ids", vocabulary=vocabulary)
    if not prompt_ids.shape[1]:
        raise ValueError("Generation requires a nonempty prompt")
    valid_prompt = _mask(prompt_mask, prompt_ids, "prompt_mask")
    if (not bool(valid_prompt[:, -1].all())
            or bool((valid_prompt[:, :-1] & ~valid_prompt[:, 1:]).any())):
        raise ValueError("Prompt masks must describe unpadded or left-padded prompts")
    if base.get_input_embeddings().weight.device != prompt_ids.device:
        raise ValueError("Prompt IDs must be on the native model device")
    if memory_ids is None:
        if memory_mask is not None:
            raise ValueError("memory_mask requires memory_ids")
        memory_ids = prompt_ids.new_empty((1, 0))
    _ids(memory_ids, "memory_ids", device=prompt_ids.device, vocabulary=vocabulary)
    valid_memory = (torch.ones_like(memory_ids, dtype=torch.bool) if memory_mask is None
                    else _mask(memory_mask, memory_ids, "memory_mask"))
    stops = _eos_ids(base.config.eos_token_id if eos_token_id is None else eos_token_id,
                     vocabulary)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            model.prepare_context(prompt_ids=prompt_ids, prompt_mask=valid_prompt,
                                  memory_ids=memory_ids, memory_mask=valid_memory)
            sequence = prompt_ids.clone()
            attention = valid_prompt.long()
            for _ in range(max_new_tokens):
                # Match native generation's position convention for left padding.
                positions = attention.cumsum(-1) - 1
                positions = positions.masked_fill(attention == 0, 1)
                output = base(input_ids=sequence, attention_mask=attention,
                              position_ids=positions, use_cache=False, return_dict=True)
                last_logits = output.logits[:, -1, :]
                if not bool(torch.isfinite(last_logits).all()):
                    raise ValueError("Generation produced nonfinite next-token logits")
                next_token = last_logits.argmax(-1, keepdim=True)
                del last_logits, output
                sequence = torch.cat((sequence, next_token), dim=1)
                if next_token.item() in stops:
                    break
                attention = torch.cat((attention, attention.new_ones((1, 1))), dim=1)
            return sequence
    finally:
        model.release_context()
        for module, training in modes:
            module.training = training
