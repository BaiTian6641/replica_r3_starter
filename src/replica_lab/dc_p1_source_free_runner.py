"""Source-free query-CE learning diagnostic and bounded alternative runner.

The old coincident-mask treatment remains a global CE rescale.  This module uses
an additional source-free query CE term instead, with real frozen-backbone
features, the fixed plastic bridge, selected expert gradients and P-only greedy
readout.  Tiny fixtures are software diagnostics, not native learning evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from dc_protocol_common_20261027 import read_json, require, sha256_json
from replica_lab import artifact_publication as publication
from replica_lab.dc_p1_p_only_reader import POnlyReader
from replica_lab.dc_response_objective import answer_span_objective, shift_labels


def source_free_train_step(backend: Any, row: Mapping[str, Any], *,
                           source_free_weight: float,
                           gradient_amplification: float = 1.0) -> dict:
    """Execute both objective forwards, then one clipped optimizer update."""
    require(isinstance(source_free_weight, (int, float))
            and not isinstance(source_free_weight, bool)
            and 0.0 <= source_free_weight <= 1.0, "objective",
            "source-free weight must be in [0, 1]")
    require(isinstance(gradient_amplification, (int, float))
            and not isinstance(gradient_amplification, bool)
            and gradient_amplification > 0, "objective",
            "gradient amplification must be positive")
    source_prompt = backend._render(f"{row['source']}\n{row['prompt']}")
    query_prompt = backend._render(str(row["prompt"]))
    response = backend.tokenizer(
        str(row["target"]), add_special_tokens=False)["input_ids"] + [backend.eos_id]
    losses = {}
    positions = {}
    for name, prompt_ids in (("source_conditioned", source_prompt),
                             ("source_free", query_prompt)):
        shifted, masks = shift_labels(
            prompt_ids=prompt_ids, response_ids=response,
            answer_span=(0, len(response) - 2), eos_id=backend.eos_id)
        input_ids = prompt_ids + response[:-1]
        final, hidden_states = backend._native_forward(input_ids)
        features = backend._features(hidden_states, len(prompt_ids))
        slot = backend.bank.expert(0)(features)
        port = backend.adapter.plastic_ports[str(backend.sites[0])]
        final = final + port(final, slot, enabled=True)
        logits = backend.adapter.base.lm_head(final)
        start = len(prompt_ids) - 1
        response_logits = logits[0, start:start + len(response)]
        labels = torch.tensor(shifted, dtype=torch.long, device=backend.device)
        objective = answer_span_objective(
            response_logits, labels, masks, answer_span_ce_weight=0.0)
        losses[name] = objective["loss"]
        positions[name] = len(input_ids)
    total = losses["source_conditioned"] + source_free_weight * losses["source_free"]
    require(bool(torch.isfinite(total)), "training", "source-free loss is nonfinite")
    backend.bank.zero_gradients()
    (total * gradient_amplification).backward()
    parameters = [parameter for _, parameter in backend.bank.expert(0).named_parameters()]
    for name, parameter in backend.bank.expert(0).named_parameters():
        require(parameter.grad is not None, "gradient_scope",
                f"{name} received no gradient")
        require(bool(torch.isfinite(parameter.grad).all()), "gradient_scope",
                f"nonfinite gradient for {name}")
    for name, parameter in backend.adapter.named_parameters():
        require(parameter.grad is None, "gradient_scope",
                f"frozen backbone parameter {name} acquired a gradient")
    raw_norm = float(torch.sqrt(sum(
        (parameter.grad.detach() ** 2).sum() for parameter in parameters)).cpu())
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    clipped_norm = float(torch.sqrt(sum(
        (parameter.grad.detach() ** 2).sum() for parameter in parameters)).cpu())
    backend.optimizer.step()
    backend.bank.zero_gradients()
    return {
        "loss": float(total.detach()),
        "source_conditioned_loss": float(losses["source_conditioned"].detach()),
        "gradient_amplification": gradient_amplification,
        "source_free_loss": float(losses["source_free"].detach()),
        "source_free_weight": float(source_free_weight),
        "raw_gradient_norm": raw_norm,
        "optimizer_class": "NativeFusedAdamW",
        "epsilon": backend.optimizer.param_groups[0]["eps"],
        "clip_scale": min(1.0, 1.0 / max(raw_norm, 1e-12)),
        "clipped_gradient_norm": clipped_norm,
        "gradient_ownership": {"gate_up_down": True, "frozen_backbone": True,
                               "finite": True},
        "physical_work": {
            "training_forwards": 2,
            "source_free_query_forwards": 1,
            "backward_calls": 1,
            "optimizer_steps": 1,
            "teaching_processed_positions": sum(positions.values()),
            "forward_calls": 2,
            "processed_text_positions": sum(positions.values()),
            "source_conditioned_processed_positions": positions["source_conditioned"],
            "source_free_query_processed_positions": positions["source_free"],
        },
    }


def _merge_case(case: Mapping[str, Any], label: Mapping[str, Any]) -> dict:
    merged = dict(case)
    merged["target"] = label["target"]
    return merged


def measure_behavior(backend: Any, cases: Sequence[Mapping[str, Any]], *,
                     arm: str) -> dict:
    """Measure actual P-only generated text, probabilities, margins and cost."""
    reader = POnlyReader(backend)
    rows = []
    for case in cases:
        prompt_ids = reader._render(str(case["prompt"]))
        result = reader.evaluate(case, arm=arm, prepared={"prompt_ids": prompt_ids})
        logits = reader._logits(prompt_ids, len(prompt_ids))
        target_id = reader.tokenizer(
            str(case["target"]), add_special_tokens=False)["input_ids"][0]
        probabilities = torch.softmax(logits[0, -1].float(), dim=-1)
        target_logit = logits[0, -1, target_id].float()
        other = logits[0, -1].clone()
        other[target_id] = -torch.inf
        margin = target_logit - other.max()
        rows.append({
            "case_id": case["case_id"], "arm": arm,
            "generated_ids": result["generated_ids"],
            "generated_text": result["generated_text"],
            "stopped": result["stopped"], "stop_reason": result["stop_reason"],
            "target_id": int(target_id),
            "target_probability": float(probabilities[target_id].detach().cpu()),
            "target_margin": float(margin.detach().cpu()),
            "generated_tokens": result["generated_tokens"],
            "processed_positions": result["processed_positions"],
        })
    return {
        "rows": rows,
        "generated_tokens": sum(row["generated_tokens"] for row in rows),
        "processed_positions": sum(row["processed_positions"] for row in rows),
        "mean_target_probability": sum(row["target_probability"] for row in rows) / len(rows),
        "mean_target_margin": sum(row["target_margin"] for row in rows) / len(rows),
    }


def behavior_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict:
    changes = []
    for old, new in zip(before["rows"], after["rows"]):
        changes.append({
            "case_id": old["case_id"],
            "before_text": old["generated_text"], "after_text": new["generated_text"],
            "text_changed": old["generated_text"] != new["generated_text"],
            "target_probability_delta": new["target_probability"] - old["target_probability"],
            "target_margin_delta": new["target_margin"] - old["target_margin"],
        })
    return {
        "rows": changes,
        "changed_answer_count": sum(row["text_changed"] for row in changes),
        "mean_target_probability_delta": (
            after["mean_target_probability"] - before["mean_target_probability"]),
        "mean_target_margin_delta": after["mean_target_margin"] - before["mean_target_margin"],
        "generated_tokens": after["generated_tokens"],
        "processed_positions": after["processed_positions"],
    }


def reset_to_parent(backend: Any) -> None:
    with torch.no_grad():
        for name, parameter in backend.bank.expert(0).named_parameters():
            parameter.copy_(backend.parent_state[name])
    backend.bank.zero_gradients()
    backend.optimizer.state.clear()
    backend.bank.expert_steps = [0]


def _state_payload(backend: Any, arm: str, event_index: int,
                   counters: Mapping[str, Any]) -> dict:
    return {
        "format": "replica.dc_p1_source_free_checkpoint.v1",
        "meta": {"arm_id": arm, "event_index": event_index,
                 "counters": dict(counters)},
        "expert_state": {
            name: parameter.detach().to("cpu").clone()
            for name, parameter in backend.bank.expert(0).named_parameters()},
        "optimizer_state": backend.optimizer.state_dict(),
        "parent_state_sha256": backend.parent_sha256,
        "_estimated_bytes": 4096,
    }


def _restore_payload(backend: Any, payload: Mapping[str, Any]) -> None:
    with torch.no_grad():
        for name, parameter in backend.bank.expert(0).named_parameters():
            parameter.copy_(payload["expert_state"][name])
    backend.optimizer.load_state_dict(payload["optimizer_state"])
    backend.bank.zero_gradients()


def _sum_work(counters: dict, work: Mapping[str, int]) -> None:
    for name, value in work.items():
        counters[name] = counters.get(name, 0) + int(value)

def _behavior_work(measure: Mapping[str, Any], prefix: str) -> dict[str, int]:
    rows = len(measure["rows"])
    work = {
        f"{prefix}generation_requests": rows,
        f"{prefix}generation_prefill_forwards": rows,
        f"{prefix}generation_decode_forwards": measure["generated_tokens"] - rows,
        f"{prefix}autoregressive_forwards": measure["generated_tokens"],
        f"{prefix}generated_tokens": measure["generated_tokens"],
        f"{prefix}generation_processed_positions": measure["processed_positions"],
        "forward_calls": measure["generated_tokens"],
        "processed_text_positions": measure["processed_positions"],
    }
    work.update({
        "generation_requests": rows,
        "generation_prefill_forwards": rows,
        "generation_decode_forwards": measure["generated_tokens"] - rows,
        "autoregressive_forwards": measure["generated_tokens"],
        "generated_tokens": measure["generated_tokens"],
        "generation_processed_positions": measure["processed_positions"],
    })
    if prefix:
        work[f"{prefix}forward_calls"] = measure["generated_tokens"]
        work[f"{prefix}processed_text_positions"] = measure["processed_positions"]
    return work


def run_fit_diagnostic(backend: Any, teaching: Sequence[Mapping[str, Any]],
                       cases: Sequence[Mapping[str, Any]], output: Path, *,
                       interrupt_after: int | None = None,
                       resume: bool = False,
                       prepared: str | Path | None = None) -> dict:
    """Run the declared six-event/four-arm schedule with optional recovery."""
    if prepared is not None:
        from preflight_dc_p1_source_free_feature_alignment_v1 import (
            CONFIG_PATH, validate_prepared)
        validate_prepared(Path(prepared), read_json(CONFIG_PATH))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meter = publication.LiveByteMeter(output, cap_bytes=33_554_432,
                                      label="dc_p1_source_free")
    state_path = output / "fit_state.json"
    if resume:
        require(state_path.is_file(), "resume", "fit state is missing")
        state = json.loads(state_path.read_text())
        counters = dict(state["counters"])
        recovery_eval = measure_behavior(backend, cases[:1], arm="recovery")
        _sum_work(counters, {
            "restarts": 1, "model_loads": 1, "tokenizer_loads": 1,
            "recovery_model_loads": 1, "recovery_tokenizer_loads": 1,
            "recovery_generation_requests": 1,
            "recovery_generation_prefill_forwards": 1,
            "recovery_generation_decode_forwards": recovery_eval["generated_tokens"] - 1,
            "recovery_forward_calls": recovery_eval["generated_tokens"],
            "recovery_generated_tokens": recovery_eval["generated_tokens"],
            "recovery_processed_positions": recovery_eval["processed_positions"],
            "recovery_training_forwards": 0,
            "recovery_backward_calls": 0,
            "recovery_optimizer_steps": 0,
            "recovery_processed_text_positions": recovery_eval["processed_positions"],
        })
        for arm, arm_state in state["arms"].items():
            completed = arm_state["events_completed"]
            if completed:
                payload = publication.load_torch(
                    output, f"checkpoints/{arm}_{completed[-1]}.pt")
                _restore_payload(backend, payload)
                break
    else:
        require(not state_path.exists(), "fresh_output", "fit state already exists")
        counters = {"model_loads": 1, "tokenizer_loads": 1}
        state = {"format": "replica.dc_p1_source_free_fit_state.v1",
                 "status": "in_progress", "counters": counters,
                 "arms": {arm: {"events_completed": []}
                          for arm in ("treatment", "matched_trained", "no_update", "reset_P")}
                 }
    arm_weights = {"treatment": 0.25, "matched_trained": 0.0,
                   "no_update": 0.25, "reset_P": 0.25}
    before = dict(state.get("baseline_behavior", {}))
    after = {}
    updates = 0
    for arm, weight in arm_weights.items():
        completed = set(state["arms"][arm]["events_completed"])
        if not completed:
            reset_to_parent(backend)
        if arm not in before:
            before[arm] = measure_behavior(backend, cases, arm=arm)
            state.setdefault("baseline_behavior", {})[arm] = before[arm]
            publication.write_json(output, "fit_state.json", state, meter=meter)
            _sum_work(counters, _behavior_work(before[arm], "baseline_"))
        if arm == "no_update":
            after[arm] = measure_behavior(backend, cases, arm=arm)
            _sum_work(counters, _behavior_work(after[arm], ""))
            continue
        for index, row in enumerate(teaching):
            if index in completed:
                continue
            if interrupt_after is not None and updates >= interrupt_after and not resume:
                state["status"] = "interrupted"
                state["counters"] = counters
                publication.write_json(output, "fit_state.json", state, meter=meter)
                raise RuntimeError("injected fit interruption")
            result = source_free_train_step(backend, row, source_free_weight=weight)
            _sum_work(counters, result["physical_work"])
            updates += 1
            payload = _state_payload(backend, arm, index, counters)
            publication.write_torch(output, f"checkpoints/{arm}_{index}.pt",
                                    payload, meter=meter)
            _sum_work(counters, {"per_event_checkpoint_writes": 1,
                                 "checkpoint_writes": 1})
            state["arms"][arm]["events_completed"].append(index)
            state["counters"] = counters
            publication.write_json(output, "fit_state.json", state, meter=meter)
        if arm == "reset_P":
            reset_to_parent(backend)
            payload = _state_payload(backend, arm, len(teaching) - 1, counters)
            publication.write_torch(output, "checkpoints/reset_P_final.pt",
                                    payload, meter=meter)
            _sum_work(counters, {"artifact_checkpoint_writes": 1,
                                 "checkpoint_writes": 1})
        after[arm] = measure_behavior(backend, cases, arm=arm)
        _sum_work(counters, _behavior_work(after[arm], ""))
    behavior = {arm: behavior_delta(before[arm], after[arm])
                for arm in arm_weights}
    state.update({
        "status": "passed_fit_learnability_diagnostic",
        "counters": counters,
        "behavior": behavior,
        "artifact": meter.report(),
        "development_or_final_access": False,
        "generalization_qualified": False,
    })
    publication.write_json(output, "fit_state.json", state, meter=meter)
    return state


def state_digest(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(sha256_json(state).encode()).hexdigest()
