"""Small measured-training utilities shared by local fine-tuning experiments."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import torch
from torch.nn import functional as F
from transformers import GenerationConfig

from replica_lab.evaluation import score_answer


def response_statistics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Separate causal answer NLL from each row's final supervised EOS token.

    The encoder contract puts EOS at the final supervised position. Positions
    with label -100 and the unpredicted first token are excluded. Only selected,
    detached logits enter cross entropy; this never builds a gradient graph or
    a full sequence-sized FP32 copy.
    """
    if (logits.ndim != 3 or labels.ndim != 2
            or tuple(logits.shape[:2]) != tuple(labels.shape)):
        raise ValueError("logits must be [batch, sequence, vocabulary] and labels [batch, sequence]")
    if logits.device != labels.device or labels.dtype != torch.long:
        raise ValueError("labels must be int64 on the logits device")
    targets = labels[:, 1:]
    mask = targets != -100
    if not bool(mask.any()):
        return {"answer_nll_sum": 0.0, "answer_tokens": 0,
                "eos_nll_sum": 0.0, "eos_tokens": 0}
    # Reverse cumulative counts identify exactly the final supervised token,
    # including examples with ignored positions after their terminal EOS.
    final = mask & (mask.flip(-1).cumsum(-1).flip(-1) == 1)
    losses = F.cross_entropy(logits.detach()[:, :-1][mask].float(), targets[mask],
                             reduction="none")
    eos_selected = final[mask]
    return {"answer_nll_sum": losses[~eos_selected].sum().item(),
            "answer_tokens": int((~eos_selected).sum().item()),
            "eos_nll_sum": losses[eos_selected].sum().item(),
            "eos_tokens": int(eos_selected.sum().item())}


def apply_learning_rates(optimizer, base_rates: list, completed_steps: int,
                         total_steps: int, warmup_steps: int,
                         schedule: str = "constant", min_lr_ratio: float = 0.2) -> float:
    """Set rates for the next update and return their common schedule factor.

    With two warmup updates the factors start at 0.5, 1.0. Constant schedules
    remain at one afterward. Cosine schedules finish at ``min_lr_ratio``; with
    no warmup their first update uses the base learning rate. The immutable
    ``base_rates`` belong in checkpoint schedule state so resume cannot compound
    decay by treating an already-scaled current rate as its base.
    """
    if (any(not isinstance(value, int) or isinstance(value, bool)
            for value in (completed_steps, total_steps, warmup_steps))
            or total_steps < 1 or not 0 <= completed_steps < total_steps
            or not 0 <= warmup_steps <= total_steps):
        raise ValueError("Invalid completed, total or warmup step count")
    if schedule not in {"constant", "cosine"}:
        raise ValueError("schedule must be constant or cosine")
    if not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be finite and in [0, 1]")
    if (len(base_rates) != len(optimizer.param_groups)
            or any(not math.isfinite(rate) or rate < 0 for rate in base_rates)):
        raise ValueError("base_rates must provide a finite nonnegative rate for each group")
    next_step = completed_steps + 1
    if warmup_steps and next_step <= warmup_steps:
        factor = next_step / warmup_steps
    elif schedule == "cosine" and total_steps > max(warmup_steps, 1):
        decay_start = max(warmup_steps, 1)
        progress = (next_step - decay_start) / (total_steps - decay_start)
        factor = min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    else:
        factor = 1.0
    for group, rate in zip(optimizer.param_groups, base_rates):
        group["lr"] = rate * factor
    return factor


def evaluate_generations(model, encoded: list, batch_rows: list, device: str,
                         max_new_tokens: int = 8, *, tokenizer) -> dict:
    """Greedily generate and strictly score native or fixed-context answers.

    Only past-only ``prompt_ids`` and their mask reach generation. Reference
    labels and response-bearing training input IDs never reach the model.
    Derive generation configuration from the model config, bypassing inherited
    stochastic generation settings. Preserve training modes and model config.
    """
    if getattr(model, "regime", None) not in {"baseline", "full", "modules_only", "full_augmented"}:
        raise ValueError("Unknown generation regime")
    augmented = model.regime in {"modules_only", "full_augmented"}
    if len(encoded) != len(batch_rows) or not encoded:
        raise ValueError("Nonempty encoded batches and rows must have matching lengths")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    base = model.base
    generation_config = GenerationConfig.from_model_config(base.config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.use_cache = True
    generation_config.max_new_tokens = max_new_tokens
    eos = generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
        generation_config.eos_token_id = eos
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    modes = [(module, module.training) for module in model.modules()]
    examples = []
    try:
        model.eval()
        with torch.inference_mode():
            for row, batch in zip(batch_rows, encoded):
                prompt = batch["prompt_ids"]
                mask = batch["prompt_mask"]
                if prompt.ndim != 2 or prompt.shape[0] != 1 or mask.shape != prompt.shape:
                    raise ValueError("Generation expects one past-only prompt and matching mask per row")
                input_ids = prompt.to(device)
                if augmented:
                    from replica_lab.generation import generate_fixed_context
                    output = generate_fixed_context(
                        model, prompt_ids=input_ids, prompt_mask=mask.to(device),
                        memory_ids=batch["memory_ids"].to(device),
                        memory_mask=batch["memory_mask"].to(device),
                        max_new_tokens=max_new_tokens, eos_token_id=eos)
                else:
                    output = base.generate(
                        input_ids=input_ids, attention_mask=mask.to(device),
                        generation_config=generation_config, do_sample=False,
                        use_cache=True, max_new_tokens=max_new_tokens)
                generated = output[0, input_ids.shape[1]:].cpu().tolist()
                ended_by_eos = bool(generated and generated[-1] in eos_ids)
                answer_ids = generated[:-1] if ended_by_eos else generated
                completion = tokenizer.decode(answer_ids, skip_special_tokens=False)
                examples.append({
                    "id": row["id"], "expected": row["response"],
                    "prompt_tokens": input_ids.shape[1],
                    "prompt_token_ids_sha256": hashlib.sha256(
                        json.dumps(prompt.cpu().tolist()).encode()).hexdigest(),
                    "generated_token_ids": generated, "generated_tokens": len(generated),
                    "ended_by_eos": ended_by_eos,
                    "hit_token_budget": not ended_by_eos and len(generated) >= max_new_tokens,
                    "raw_completion": tokenizer.decode(generated, skip_special_tokens=False),
                    "completion": completion, **score_answer(completion, row["response"]),
                })
    finally:
        model.train(modes[0][1])
        # Preserve intentional per-module eval modes as well as the root mode.
        for module, training in modes:
            module.training = training
    correct = sum(item["task_correct"] for item in examples)
    histogram = dict(Counter(item["completion"] for item in examples))
    return {"examples": examples, "evaluated_examples": len(examples),
            "task_correct": correct, "task_accuracy": correct / len(examples),
            "fixture_task_accuracy": correct / len(examples),
            "exact_text_accuracy": sum(item["exact_text_match"] for item in examples) / len(examples),
            "response_histogram": histogram, "distinct_responses": len(histogram),
            "decoding": {"do_sample": False, "num_beams": 1, "use_cache": not augmented,
                         "context": "fixed_original_prompt_and_support" if augmented else "native_prompt",
                         "max_new_tokens": max_new_tokens}}
