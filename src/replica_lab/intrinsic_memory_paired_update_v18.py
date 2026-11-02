"""V18.2 paired-joint versus sequential CE, with live slots and full trajectories.

No native loader, evaluation split or promotion is present in this module.
"""
from pathlib import Path
import json

import torch

from .contextual_completion_objective_v1 import encode_complete_response
from .contextual_memory import canonical_spans
from .intrinsic_memory_acceptance_v15 import sequence_likelihood_parts, sequence_margin
from .intrinsic_memory_budget_v12 import atomic
from .intrinsic_memory_interference_protocol_v17 import PAIR_INDICES
from .intrinsic_memory_measurement_v18 import (
    capture_state, restore_state, module_statistics, live_values, prediction_evidence, digest, named_parameters)
from .intrinsic_memory_native_v5 import world_scope
from .optimizers import NativeFusedAdamW

FORMAT = "replica.intrinsic_memory.paired_update.v18.2"
ARMS = ("sequential", "paired_joint")


def validate_pairs(rows):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("fitting rows must be objects")
    by_id = {row["world_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate world identities")
    selected = []
    for pair in PAIR_INDICES:
        ids = [f"v11_fit_{2*pair+side:02d}" for side in (0, 1)]
        if any(name not in by_id for name in ids):
            raise ValueError("required original fitting pair missing")
        left, right = [by_id[name] for name in ids]
        if (left["question"] != right["question"] or left["family"] != right["family"]
                or left["answer"] == right["answer"]
                or any(row["answer"] not in ("red", "blue", "green", "gold") for row in (left, right))):
            raise ValueError("known-known original pair membership differs")
        selected.extend((left, right))
    return selected


def _plain(parts):
    return {key: float(value.detach()) if torch.is_tensor(value) else value
            for key, value in parts.items()}


def _write(engine, name, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with engine.measured("paired_update_artifact", writes=1, artifact_bytes=len(payload)):
        atomic(Path(engine.output) / name, value)


def run_comparison(engine, rows, *, observe=None):
    rows = validate_pairs(rows)  # before source computation or any optimizer work
    path = engine.path
    if path.key_mode != "content" or not isinstance(path.optimizer, NativeFusedAdamW):
        raise ValueError("content keys and NativeFusedAdamW required")
    scopes = [world_scope(row) for row in rows]
    spans = [canonical_spans(row["text"], record_id=row["world_id"], source_id=row["world_id"],
                             source_epoch=1, available_at=1) for row in rows]
    if any(len(items) != 1 for items in spans):
        raise ValueError("one complete fitting source span required")
    encodings = []
    prompts = [engine.tokens(row) for row in rows]
    answers = []
    for row in rows:
        with engine.measured("answer_tokenization", prompt_tokenizer_calls=1):
            answers.append(engine.backend.answer_ids(row["answer"]))
    for index in range(len(rows)):
        encodings.append([encode_complete_response(prompts[index], answers[other],
                          spec=engine.spec, device=engine.device) for other in (index, index ^ 1)])
    with engine.measured("initial_state_capture"):
        initial = capture_state(path)
    arms = {}

    def measure(step):
        measurements = []
        with torch.no_grad(), engine.measured("all_world_measurement"):
            for index, row in enumerate(rows):
                live = live_values(engine, spans[index], scopes[index])
                candidates, predictions = [], []
                for encoding in encodings[index]:
                    logits = engine.forward(encoding.input_ids, encoding.attention_mask, live, scopes[index])
                    candidates.append(_plain(sequence_likelihood_parts(logits, encoding, spec=engine.spec)))
                    predictions.append(prediction_evidence(logits, encoding))
                measurements.append({"world_id": row["world_id"], "pair_index": PAIR_INDICES[index//2],
                    "expected": candidates[0], "alternative": candidates[1],
                    "margin": sequence_margin(candidates[0], candidates[1]),
                    "prediction": predictions[0], "alternative_prediction": predictions[1],
                    "slot_digest": digest(live.slots)})
                if observe:
                    observe()
        if observe:
            observe()
        return {"step": step, "worlds": measurements}

    try:
        for arm in ARMS:
            with engine.measured("complete_state_restore"):
                restoration = restore_state(path, initial)
            trajectory, updates = [measure(0)], []
            for offset, pair in enumerate(PAIR_INDICES):
                pair_rows = [2*offset, 2*offset+1]
                if pair in (1, 5):
                    pair_rows.reverse()
                groups = [[index] for index in pair_rows] if arm == "sequential" else [pair_rows]
                for group in groups:
                    path.optimizer.zero_grad(set_to_none=True)
                    with engine.measured("pre_update_parameter_capture"):
                        before = {name: value.detach().clone() for name, value in named_parameters(path)}
                    losses, tokens = [], []
                    for index in group:
                        live = live_values(engine, spans[index], scopes[index])
                        encoding = encodings[index][0]
                        logits = engine.forward(encoding.input_ids, encoding.attention_mask, live, scopes[index])
                        parts = sequence_likelihood_parts(logits, encoding, spec=engine.spec)
                        loss = -parts["total_log_likelihood"] / parts["supervised_tokens"]
                        with engine.measured("per_world_backward", backwards=1):
                            (loss / len(group)).backward()
                        losses.append(float(loss.detach())); tokens.append(parts["supervised_tokens"])
                    with engine.measured("gradient_measurement"):
                        gradients = module_statistics(path)
                    if not all(item["finite"] for item in gradients.values()):
                        raise ValueError("nonfinite gradients")
                    with engine.measured("native_fused_adamw_step", updates=1):
                        path.optimizer.step()
                    with engine.measured("actual_displacement_measurement"):
                        displacement = module_statistics(path, before=before)
                    updates.append({"step": len(updates)+1, "pair_index": pair,
                        "world_ids": [rows[index]["world_id"] for index in group],
                        "per_world_mean_CE": losses, "supervised_tokens": tokens,
                        "loss_coefficients": [1/len(group)] * len(group),
                        "optimized_loss": sum(losses)/len(group),
                        "per_module_gradients": gradients, "actual_adamw_displacement": displacement})
                    trajectory.append(measure(len(updates)))
            arms[arm] = {"restoration": restoration, "trajectory": trajectory, "updates": updates,
                         "optimizer_steps": len(updates), "world_exposures": 8}
            _write(engine, arm + ".json", arms[arm])
        if arms["sequential"]["trajectory"][0] != arms["paired_joint"]["trajectory"][0]:
            raise ValueError("common baseline changed after complete restoration")
        endpoints = {}
        for arm, result in arms.items():
            baseline = result["trajectory"][0]["worlds"]
            final = result["trajectory"][-1]["worlds"]
            margins = [row["margin"]["total_log_likelihood"] for row in final]
            minima = [min(margins[i:i+2]) for i in range(0, 8, 2)]
            endpoints[arm] = {"mean_final_pair_minimum_margin": sum(minima)/4,
                "pair_minimum_margins": minima, "absolute_final_margins": margins,
                "guard_count": sum(row["margin"]["total_log_likelihood"] < old["margin"]["total_log_likelihood"]
                                   for row, old in zip(final, baseline)),
                "positive_pair_margin_worlds": sum(margin > 0 for margin in margins),
                "full_vocabulary_first_token_correct": sum(row["prediction"]["first_token_correct"] for row in final),
                "teacher_forced_complete_correct": sum(row["prediction"]["complete_teacher_forced_correct"] for row in final)}
        contrast = endpoints["paired_joint"]["mean_final_pair_minimum_margin"] - endpoints["sequential"]["mean_final_pair_minimum_margin"]
        summary = {"format": FORMAT, "endpoints": endpoints, "joint_minus_sequential": contrast,
            "relative_conditions_passed": contrast > 0 and endpoints["paired_joint"]["guard_count"] <= endpoints["sequential"]["guard_count"],
            "optimizer_steps": {arm: result["optimizer_steps"] for arm, result in arms.items()},
            "per_world_exposures_per_arm": 1, "parameter_updates": 12,
            "objective": "answer_plus_EOS_mean_CE_with_inherited_V16_teacher_KL_moments",
            "interpretation": "relative improvement may leave both worlds incorrect; no generalization or promotion",
            "development_labels_used": False, "generated_tokens": 0}
    finally:
        with engine.measured("final_complete_state_restore"):
            final_restoration = restore_state(path, initial)
    summary["final_restoration"] = final_restoration
    _write(engine, "comparison.json", summary)
    return summary
