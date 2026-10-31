"""Fitting-only native reader/query and interference diagnostic v15.

One uncached prefill per fitting row/control captures the actual query, key,
value, normalized slot weights and read at the final prompt token. No generation,
parameter update, development row or capability claim is permitted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch
from safetensors.torch import save_file

from .contextual_memory import WIDTH
from .intrinsic_memory_budget_v12 import atomic
from .intrinsic_memory_counterfactual_v13 import _digest, _source
from .intrinsic_memory_diagnostic_v12 import right_padded_prefix

CONTROLS = ("contextual", "wrong", "shuffled", "zero")
PAIR_ARTIFACT_BYTES = 4 * 1024**2
FORMAT = "replica.intrinsic_memory.reader_query_diagnostic.v15"


def _atomic_tensors(path, tensors, ledger, stage):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with ledger.work(stage, "reader_query_tensor_artifact", writes=1,
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


def attentive_forward(path, ids, mask, batch, scope, *, control="contextual",
                      wrong=None, wrong_scope=None, position):
    """Run the established memory intervention and expose its actual attention."""
    if control not in CONTROLS:
        raise ValueError("undeclared reader/query control")
    path.validate(batch, scope)
    key_values = None
    if control == "contextual":
        # This is the same snapshot owner as the production contextual path: the
        # learned order is added exactly once and validation stays live.
        context = path.snapshot(ids, batch, scope=scope)
    else:
        slots = batch.slots
        if control == "wrong":
            if wrong is None or wrong_scope is None:
                raise ValueError("wrong control source missing")
            path.validate(wrong, wrong_scope)
            slots = wrong.slots
        elif control == "shuffled":
            slots = slots.flip(0)
        content_slots = slots
        order = path.order(torch.arange(len(slots), device=ids.device))
        slots = slots + order
        if control == "zero":
            slots = torch.zeros_like(slots)
            content_slots = torch.zeros_like(content_slots)
        padding = torch.zeros((path.spec.slots - len(slots), WIDTH),
                              device=slots.device, dtype=slots.dtype)
        values = torch.cat((slots, padding))[None, None].expand(
            ids.shape[0], ids.shape[1], -1, -1)
        available = (torch.arange(path.spec.slots, device=ids.device) < len(slots))
        available = available[None, None].expand(values.shape[:3])
        key_values = None
        if getattr(path, "key_mode", "ordered") == "content":
            key_values = torch.cat((content_slots, padding))[None, None].expand(
                ids.shape[0], ids.shape[1], -1, -1)

        def validate():
            path.validate(batch, scope)
            if control == "wrong":
                path.validate(wrong, wrong_scope)

        if key_values is None:
            context = path.reader.snapshot(values, available, validate=validate)
        else:
            context = path.reader.snapshot(values, available, validate=validate,
                                           key_values=key_values)
    path.cost["snapshot_copies"] += 1
    path.cost["snapshot_bytes"] += context.values.numel() * context.values.element_size()
    if key_values is not None:
        path.cost["snapshot_bytes"] += key_values.numel() * key_values.element_size()
    logits, audit = path.adapter.forward_text(
        ids, mask, use_projector=False, use_ports=False, use_spine=False,
        intrinsic_memory=context,
        intrinsic_placement=getattr(path.spec, "placement",
                                    "block_input_query_to_post_block_residual"),
        return_audit=True, capture_attention=True, attention_position=position)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite reader/query logits")
    if set(audit.memory_attention or {}) != {int(site) for site in path.reader.ports}:
        raise ValueError("active reader/query control did not expose every site")
    return logits, audit.memory_attention


def _attention_metadata(attention):
    result = {}
    for site, tensors in attention.items():
        weights = tensors["weights"].detach().float()
        available = int((weights > 0).sum())
        entropy = float(-(weights * weights.clamp_min(1e-12).log()).sum())
        query = tensors["query"].detach().float().reshape(-1)
        key = tensors["key"].detach().float()
        scores = (key @ query) / math.sqrt(query.numel())
        result[str(site)] = {
            "available_slots": available,
            "entropy": entropy,
            "effective_slots": math.exp(entropy),
            "max_weight": float(weights.max()),
            "top_slot": int(weights.argmax()),
            "weight_sum": float(weights.sum()),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "read_matches_weighted_value": bool(torch.allclose(
                weights @ tensors["value"].float(), tensors["read"].float(),
                atol=1e-5, rtol=1e-5)),
        }
    return result


def _tensor(tensors, name):
    value = tensors[name]
    if name in ("key", "value"):
        return value[0].detach().float().cpu()
    return value[0, 0, 0].detach().float().cpu()


def diagnose_pair(engine, left, right, pair_index, output, *, observe=None):
    if left["question"] != right["question"] or left["answer"] == right["answer"]:
        raise ValueError("same-question counterfactual pair required")
    rows = (left, right)
    batches, scopes = [], []
    for row in rows:
        batch, _, scope = _source(engine.path, engine, row)
        batches.append(batch)
        scopes.append(scope)
    logits = [[None for _ in CONTROLS] for _ in rows]
    captured = {site: {name: [[None for _ in CONTROLS] for _ in rows]
                       for name in ("query", "key", "value", "weights", "read")}
                for site in engine.path.reader.ports}
    metadata = [[None for _ in CONTROLS] for _ in rows]
    for side, row in enumerate(rows):
        tokens = engine.tokens(row)
        ids, mask = right_padded_prefix(tokens, engine.spec, engine.device)
        position = len(tokens) - 1
        for control_index, control in enumerate(CONTROLS):
            with torch.no_grad(), engine.measured(
                    "reader_query_backbone_forward", forwards=1,
                    input_tokens=ids.numel()) as event:
                event["nonpadding_input_tokens"] = int(mask.sum())
                event["phase"] = "single_uncached_prefill_no_generation"
                event["source_feature_provider_forward"] = False
                result, attention = attentive_forward(
                    engine.path, ids, mask, batches[side], scopes[side],
                    control=control, wrong=batches[side ^ 1],
                    wrong_scope=scopes[side ^ 1], position=position)
            first = result[0, position].detach().float().cpu()
            logits[side][control_index] = first
            for site in engine.path.reader.ports:
                selected = attention[int(site)]
                for name in captured[site]:
                    captured[site][name][side][control_index] = _tensor(selected, name)
            expected_id = engine.backend.answer_ids(row["answer"])[0]
            other_id = engine.backend.answer_ids(rows[side ^ 1]["answer"])[0]
            metadata[side][control_index] = {
                "control": control,
                "first_token_correct": int(first.argmax()) == expected_id,
                "expected_token_id": expected_id,
                "other_token_id": other_id,
                "expected_logit": float(first[expected_id]),
                "other_logit": float(first[other_id]),
                "expected_minus_other_logit": float(first[expected_id] - first[other_id]),
                "first_logits_sha256": _digest(first),
                "reader_gates": {site: float(
                    engine.path.reader.ports[site].gate_logit.detach().sigmoid())
                                 for site in engine.path.reader.ports},
                "attention": _attention_metadata(attention),
                "generated_tokens": 0,
            }
            if observe is not None:
                observe()
    tensor_document = {
        "logits": torch.stack([torch.stack(side) for side in logits]).bfloat16(),
        "slots_0": batches[0].slots.detach().float().cpu(),
        "slots_1": batches[1].slots.detach().float().cpu(),
    }
    for site, values in captured.items():
        for name, rows_values in values.items():
            tensor_document[f"site_{site}_{name}"] = torch.stack(
                [torch.stack(side) for side in rows_values])
    pair_dir = Path(output) / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    artifact = pair_dir / f"pair_{pair_index:02d}.safetensors"
    _atomic_tensors(artifact, tensor_document, engine.ledger, engine.stage)
    document = {
        "format": FORMAT + ".pair", "pair_index": pair_index,
        "world_ids": [row["world_id"] for row in rows],
        "question": left["question"], "expected_answers": [row["answer"] for row in rows],
        "controls": list(CONTROLS), "control_outputs": metadata,
        "both_contextual_first_token_correct": all(
            metadata[side][0]["first_token_correct"] for side in range(2)),
        "tensor_artifact": str(artifact.relative_to(output)),
        "tensor_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "full_first_token_distributions_and_attention_preserved": True,
        "development_labels_used": False, "parameter_updates": 0,
    }
    with engine.measured("reader_query_pair_receipt", writes=1):
        atomic(pair_dir / f"pair_{pair_index:02d}.json", document)
    return document


def run_diagnostic(engine, rows, output, *, observe=None):
    if len(rows) != 32:
        raise ValueError("the fixed fitting-only reader/query diagnostic requires 32 rows")
    documents = [diagnose_pair(engine, rows[index], rows[index + 1], index // 2, output,
                               observe=observe)
                 for index in range(0, 32, 2)]
    correct = {control: sum(document["control_outputs"][side][index]["first_token_correct"]
                            for document in documents for side in range(2))
               for index, control in enumerate(CONTROLS)}
    summary = {
        "format": FORMAT, "pairs": len(documents), "rows": 32,
        "controls": list(CONTROLS), "first_token_correct": correct,
        "counterfactual_pairs_both_contextual_first_token_correct": sum(
            document["both_contextual_first_token_correct"] for document in documents),
        "forwards_per_row": len(CONTROLS), "generated_tokens": 0,
        "development_labels_used": False, "parameter_updates": 0,
        "native_capability_or_generalization_claim": False,
        "pair_receipts": [f"pairs/pair_{index:02d}.json" for index in range(16)],
    }
    with engine.measured("reader_query_summary", writes=1):
        atomic(Path(output) / "diagnostic.json", summary)
    return summary
