"""Fitting-only sequence-margin and gradient-interference diagnostic v17.

Each pair gets identical parent state, two isolated one-update arms and complete
answer/EOS evidence. No development rows, best-prefix selection or promotion.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch
from safetensors.torch import save_file

from .contextual_completion_objective_v1 import encode_complete_response
from .contextual_memory import WIDTH
from .intrinsic_memory_acceptance_v15 import sequence_likelihood_parts, sequence_margin
from .intrinsic_memory_budget_v12 import atomic
from .intrinsic_memory_counterfactual_v13 import _capture, _source
from .intrinsic_memory_diagnostic_v12 import right_padded_prefix
from .intrinsic_memory_reader_query_v15 import attentive_forward

CONTROLS = ("wrong_value", "wrong_read_3", "wrong_read_7")
PAIR_ARTIFACT_BYTES = 8 * 1024**2
FORMAT = "replica.intrinsic_memory.interference_diagnostic.v17"


def _atomic_tensors(path, tensors, ledger, stage):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with ledger.work(stage, "interference_tensor_artifact", writes=1,
                     artifact_bytes=PAIR_ARTIFACT_BYTES) as event:
        try:
            save_file({key: value.detach().cpu().contiguous()
                       for key, value in tensors.items()}, str(temporary))
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            event["actual"] = {"artifact_bytes": path.stat().st_size}
            event["tensor_names"] = sorted(tensors)
        finally:
            if temporary.exists():
                temporary.unlink()


def _float_parts(parts):
    return {key: float(parts[key]) if key.endswith("log_likelihood") else parts[key]
            for key in parts}


def _candidate(engine, row, batch, scope, answer, *, prompt, control="contextual",
               wrong=None, wrong_scope=None, capture=False, override_site=None,
               override_read=None):
    encoding = encode_complete_response(prompt, engine.backend.answer_ids(answer),
                                        spec=engine.spec, device=engine.device)
    position = len(prompt) - 1
    hooks = []
    if override_site is not None:
        port = engine.path.reader.ports[str(override_site)]
        replacement = override_read.detach().to(
            device=engine.device, dtype=port.output_projection.weight.dtype)

        def replace_read(module, args):
            return (replacement.view(1, 1, -1).expand(args[0].shape),)
        hooks.append(port.output_projection.register_forward_pre_hook(replace_read))
    layer_capture, layer_hooks = _capture(engine.path, position)
    try:
        if capture:
            with torch.no_grad(), engine.measured(
                    "interference_attention_forward", forwards=1,
                    input_tokens=encoding.input_ids.numel()):
                logits, attention = attentive_forward(
                    engine.path, encoding.input_ids, encoding.attention_mask, batch, scope,
                    control=control, wrong=wrong, wrong_scope=wrong_scope,
                    position=position)
        else:
            with torch.no_grad(), engine.measured(
                    "interference_forward", forwards=1,
                    input_tokens=encoding.input_ids.numel()):
                logits = engine.forward(
                    encoding.input_ids, encoding.attention_mask, batch, scope,
                    control=control, wrong=wrong, wrong_scope=wrong_scope)
    finally:
        for hook in hooks + layer_hooks:
            hook.remove()
    return encoding, logits, sequence_likelihood_parts(logits, encoding, spec=engine.spec), \
        attention if capture else None, layer_capture


def _baseline(engine, row, batch, scope, alternative, *, prompt):
    expected = _candidate(engine, row, batch, scope, row["answer"], prompt=prompt,
                          capture=True)
    other = _candidate(engine, row, batch, scope, alternative, prompt=prompt)
    return {
        "expected": _float_parts(expected[2]), "alternative": _float_parts(other[2]),
        "margin": _float_parts(sequence_margin(expected[2], other[2])),
        "expected_encoding": expected[0], "logits": expected[1],
        "attention": expected[3], "layers": expected[4],
    }


def _named_parameters(path):
    for module in path.modules().values():
        yield from module.named_parameters()


def _parameter_state(path):
    return {
        "parameters": {name: parameter.detach().clone()
                       for name, parameter in _named_parameters(path)},
        "optimizer": copy.deepcopy(path.optimizer.state_dict()),
        "cpu_rng": torch.get_rng_state().clone(),
        "mps_rng": (torch.mps.get_rng_state().clone()
                    if torch.backends.mps.is_available() else None),
    }


def _restore_parameter_state(path, state):
    with torch.no_grad():
        for name, parameter in _named_parameters(path):
            parameter.copy_(state["parameters"][name].to(
                device=parameter.device, dtype=parameter.dtype))
    path.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    torch.set_rng_state(state["cpu_rng"])
    if state["mps_rng"] is not None:
        torch.mps.set_rng_state(state["mps_rng"])


def _gradient(engine, row, batch, scope, *, prompt):
    encoding = encode_complete_response(prompt, engine.backend.answer_ids(row["answer"]),
                                        spec=engine.spec, device=engine.device)
    with engine.measured("interference_gradient_forward", forwards=1,
                         input_tokens=encoding.input_ids.numel()):
        live_batch = engine.path.training_values(batch.spans, scope=scope)
        logits = engine.forward(encoding.input_ids, encoding.attention_mask, live_batch, scope)
    parts = sequence_likelihood_parts(logits, encoding, spec=engine.spec)
    loss = -parts["total_log_likelihood"] / parts["supervised_tokens"]
    with engine.measured("interference_backward", backwards=1):
        loss.backward()
    gradients = {name: parameter.grad.detach().clone()
                 for name, parameter in _named_parameters(engine.path)
                 if parameter.grad is not None}
    return gradients, float(loss.detach())
def _norm_cosine(left, right):
    left_norm = math.sqrt(sum(float(value.square().sum()) for value in left.values()))
    right_norm = math.sqrt(sum(float(value.square().sum()) for value in right.values()))
    dot = sum(float((left[name] - right[name]).sum()) for name in left.keys() & right.keys())
    denominator = left_norm * right_norm
    return {"left_norm": left_norm, "right_norm": right_norm,
            "cosine": dot / denominator if denominator > 1e-12 else 0.0,
            "dot": dot}


def _intervention(engine, row, batch, scope, alternative, partner_batch, partner_scope,
                  partner_baseline, *, prompt, control):
    if control == "wrong_value":
        expected = _candidate(engine, row, batch, scope, row["answer"], prompt=prompt,
                              control="wrong", wrong=partner_batch, wrong_scope=partner_scope)
        other = _candidate(engine, row, batch, scope, alternative, prompt=prompt,
                           control="wrong", wrong=partner_batch, wrong_scope=partner_scope)
        return expected, other
    site = int(control.rsplit("_", 1)[1])
    partner_read = partner_baseline["layers"][str(site)]["reader"]
    expected = _candidate(engine, row, batch, scope, row["answer"], prompt=prompt,
                          override_site=site, override_read=partner_read)
    other = _candidate(engine, row, batch, scope, alternative, prompt=prompt,
                       override_site=site, override_read=partner_read)
    return expected, other


def run_pair(engine, left, right, pair_index, output, *, observe=None):
    if left["question"] != right["question"] or left["answer"] == right["answer"]:
        raise ValueError("same-question known-known pair required")
    if "unknown" in (left["answer"], right["answer"]):
        raise ValueError("interference diagnostic requires known-known pairs")
    rows = (left, right)
    batches, features, scopes, prompts, baselines = [], [], [], [], []
    for row in rows:
        batch, frozen, scope = _source(engine.path, engine, row)
        prompt = engine.tokens(row)
        batches.append(batch)
        features.append(frozen)
        scopes.append(scope)
        prompts.append(prompt)
        baselines.append(_baseline(engine, row, batch, scope, rows[1 - len(baselines)]["answer"],
                                   prompt=prompt))
        if observe is not None:
            observe()
    initial = _parameter_state(engine.path)
    gradients, gradient_losses = [], []
    for row, batch, scope, prompt in zip(rows, batches, scopes, prompts):
        engine.path.optimizer.zero_grad(set_to_none=True)
        gradient, loss = _gradient(engine, row, batch, scope, prompt=prompt)
        gradients.append(gradient)
        gradient_losses.append(loss)
    pair_gradient = _norm_cosine(gradients[0], gradients[1])
    engine.path.optimizer.zero_grad(set_to_none=True)
    arms = []
    for update_side in (0, 1):
        _restore_parameter_state(engine.path, initial)
        engine.path.optimizer.zero_grad(set_to_none=True)
        row = rows[update_side]
        with engine.measured("interference_update_forward", forwards=1,
                             input_tokens=len(prompts[update_side])):
            encoding = encode_complete_response(
                prompts[update_side], engine.backend.answer_ids(row["answer"]),
                spec=engine.spec, device=engine.device)
            live_batch = engine.path.training_values(
                batches[update_side].spans, scope=scopes[update_side])
            logits = engine.path.forward(encoding.input_ids, encoding.attention_mask,
                                         live_batch, scope=scopes[update_side])
        parts = sequence_likelihood_parts(logits, encoding, spec=engine.spec)
        loss = -parts["total_log_likelihood"] / parts["supervised_tokens"]
        with engine.measured("interference_update_backward", backwards=1):
            loss.backward()
        with engine.measured("interference_optimizer_step", updates=1):
            engine.path.optimizer.step()
        after = []
        for row_, batch_, scope_, prompt_ in zip(rows, batches, scopes, prompts):
            baseline = _baseline(engine, row_, batch_, scope_, rows[1 - rows.index(row_)]["answer"],
                                 prompt=prompt_)
            after.append({"expected": baseline["expected"], "alternative": baseline["alternative"],
                          "margin": baseline["margin"]})
        arms.append({"update_side": update_side, "update_world_id": row["world_id"],
                     "loss": float(loss), "supervised_tokens": parts["supervised_tokens"],
                     "before": [{"expected": baseline["expected"],
                                 "alternative": baseline["alternative"],
                                 "margin": baseline["margin"]} for baseline in baselines],
                     "after": after})
        if observe is not None:
            observe()
    _restore_parameter_state(engine.path, initial)
    engine.path.optimizer.zero_grad(set_to_none=True)
    interventions = []
    for control in CONTROLS:
        for side, row in enumerate(rows):
            partner = side ^ 1
            result, other = _intervention(
                engine, row, batches[side], scopes[side], rows[partner]["answer"],
                batches[partner], scopes[partner], baselines[partner], prompt=prompts[side],
                control=control)
            interventions.append({
                "control": control, "side": side, "world_id": row["world_id"],
                "expected": _float_parts(result[2]), "alternative": _float_parts(other[2]),
                "margin": _float_parts(sequence_margin(result[2], other[2])),
                "layers": result[4],
            })
            if observe is not None:
                observe()
    receipt_interventions = [
        {**row, "layers": {site: {"gate": row["layers"][site]["gate"]}
                          for site in row["layers"]}}
        for row in interventions
    ]
    tensor_document = {
        "frozen_features_0": features[0], "frozen_features_1": features[1],
        "slots_0": batches[0].slots.detach().float().cpu(),
        "slots_1": batches[1].slots.detach().float().cpu(),
    }
    for site in sorted(engine.path.reader.ports):
        for name in ("query", "key", "value", "weights", "read"):
            tensor_document[f"site_{site}_{name}"] = torch.stack([
                baselines[side]["attention"][int(site)][name][0, 0, 0].detach().float().cpu()
                if name not in ("key", "value") else
                baselines[side]["attention"][int(site)][name][0].detach().float().cpu()
                for side in (0, 1)])
        for name in ("projection", "residual"):
            tensor_document[f"site_{site}_{name}"] = torch.stack([
                baselines[side]["layers"][str(site)][name].detach().float().cpu()
                for side in (0, 1)])
        for control in CONTROLS:
            selected = [row for row in interventions if row["control"] == control]
            tensor_document[f"site_{site}_{control}_projection"] = torch.stack([
                row["layers"][str(site)]["projection"].detach().float().cpu()
                for row in selected])
            tensor_document[f"site_{site}_{control}_residual"] = torch.stack([
                row["layers"][str(site)]["residual"].detach().float().cpu()
                for row in selected])
    pair_dir = Path(output) / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    artifact = pair_dir / f"pair_{pair_index:02d}.safetensors"
    _atomic_tensors(artifact, tensor_document, engine.ledger, engine.stage)
    document = {
        "format": FORMAT + ".pair", "pair_index": pair_index,
        "world_ids": [row["world_id"] for row in rows],
        "question": left["question"], "expected_answers": [row["answer"] for row in rows],
        "baseline": [{"expected": baseline["expected"], "alternative": baseline["alternative"],
                      "margin": baseline["margin"]} for baseline in baselines],
        "gradient": {**pair_gradient, "losses": gradient_losses},
        "arms": arms, "interventions": receipt_interventions,
        "state_restoration": {
            "parameters_optimizer_rng_restored_between_arms": True,
            "updates_isolated_per_arm": True,
            "native_fused_adamw": True,
        },
        "tensor_artifact": str(artifact.relative_to(output)),
        "tensor_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "development_labels_used": False,
    }
    with engine.measured("interference_pair_receipt", writes=1):
        atomic(pair_dir / f"pair_{pair_index:02d}.json", document)
    return document


def run_diagnostic(engine, rows, output, *, pair_indices=(0, 2, 4, 6), observe=None):
    if len(rows) != 32:
        raise ValueError("fixed fitting corpus required")
    documents = [run_pair(engine, rows[index], rows[index + 1], pair_index, output,
                          observe=observe)
                 for pair_index in pair_indices for index in (pair_index * 2,)]
    summary = {
        "format": FORMAT, "pairs": [document["pair_index"] for document in documents],
        "rows_per_pair": 2, "update_arms_per_pair": 2,
        "known_known_pairs": len(documents), "development_labels_used": False,
        "parameter_updates": sum(len(document["arms"]) for document in documents),
        "pair_receipts": [f"pairs/pair_{index:02d}.json" for index in pair_indices],
    }
    with engine.measured("interference_summary", writes=1):
        atomic(Path(output) / "diagnostic.json", summary)
    return summary
