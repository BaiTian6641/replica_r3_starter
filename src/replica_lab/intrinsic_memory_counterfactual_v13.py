"""Fitting-only counterfactual instrumentation for the trained v12 checkpoint.

No development rows, parameter updates, or capability gates. Full distributions
and intermediate tensors are evidence, not calibrated confidence or quality scores.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import torch
from safetensors.torch import save_file

from .contextual_memory import WIDTH, canonical_spans
from .intrinsic_memory_budget_v12 import atomic
from .intrinsic_memory_diagnostic_v12 import DiagnosticEngineV12
from .intrinsic_memory_native_v5 import world_scope

CONTROLS = ("contextual", "row_permutation", "shuffled", "wrong", "zero", "absent", "exact_text")
PAIR_ARTIFACT_BYTES = 16 * 1024**2
FORMAT = "replica.intrinsic_memory.counterfactual_diagnostic.v13"


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _distance(left, right):
    if left.shape != right.shape:
        return {"shape_left": list(left.shape), "shape_right": list(right.shape),
                "direct_comparison": False}
    delta = (left.detach().float() - right.detach().float()).abs()
    return {"shape": list(left.shape), "direct_comparison": True,
            "mean_absolute": float(delta.mean()), "max_absolute": float(delta.max()),
            "left_sha256": _digest(left), "right_sha256": _digest(right)}


def _distribution_metrics(left, right):
    left_log = torch.log_softmax(left.float(), dim=-1)
    right_log = torch.log_softmax(right.float(), dim=-1)
    left_p, right_p = left_log.exp(), right_log.exp()
    kl = float((left_p * (left_log - right_log)).sum())
    return {"probability_l1": float((left_p - right_p).abs().sum()),
            "kl_left_to_right": kl,
            "mean_absolute_logits": float((left - right).abs().mean()),
            "max_absolute_logits": float((left - right).abs().max())}


def _atomic_tensors(path, tensors, ledger, stage):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with ledger.work(stage, "counterfactual_tensor_artifact", writes=1,
                     artifact_bytes=PAIR_ARTIFACT_BYTES) as event:
        try:
            save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()},
                      str(temporary))
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


def restore_parent_checkpoint(path, checkpoint_root, parent_execution_identity):
    """Validate the parent identity, load, then restore the diagnostic identity."""
    checkpoint_root = Path(checkpoint_root)
    manifest = json.loads((checkpoint_root / "manifest.json").read_text())
    if manifest.get("execution_identity") != parent_execution_identity:
        raise ValueError("parent checkpoint execution identity differs")
    diagnostic_identity = path.execution_identity
    path.execution_identity = parent_execution_identity
    try:
        result = path.load(checkpoint_root)
    finally:
        path.execution_identity = diagnostic_identity
    return {"checkpoint": result, "parent_execution_identity": parent_execution_identity,
            "diagnostic_execution_identity": diagnostic_identity,
            "explicit_validated_migration": True}


class DiagnosticEngineV13(DiagnosticEngineV12):
    """Adds complete-row permutation without changing sealed v12 shuffle semantics."""

    def _forward(self, ids, mask, batch, scope, *, control="contextual", wrong=None,
                 wrong_scope=None):
        if control != "row_permutation":
            return super()._forward(ids, mask, batch, scope, control=control,
                                    wrong=wrong, wrong_scope=wrong_scope)
        self.path.validate(batch, scope)
        # Permute complete rows: slot content and its learned address/order vector
        # move together. Sealed `shuffled` instead reassigns content to old addresses.
        slots = batch.slots.flip(0)
        order = self.path.order(torch.arange(len(slots), device=ids.device)).flip(0)
        slots = slots + order
        padding = torch.zeros((self.path.spec.slots - len(slots), WIDTH),
                              device=slots.device, dtype=slots.dtype)
        values = torch.cat((slots, padding))[None, None].expand(ids.shape[0], ids.shape[1], -1, -1)
        available = (torch.arange(self.path.spec.slots, device=ids.device) < len(slots))
        available = available[None, None].expand(values.shape[:3])
        context = self.path.reader.snapshot(values, available,
                                            validate=lambda: self.path.validate(batch, scope))
        self.path.cost["snapshot_copies"] += 1
        self.path.cost["snapshot_bytes"] += context.values.numel() * context.values.element_size()
        return self.path.adapter.forward_text(ids, mask, use_projector=False, use_ports=False,
                                              use_spine=False, intrinsic_memory=context)


def _source(path, engine, row):
    scope = world_scope(row)
    spans = canonical_spans(row["text"], record_id=row["world_id"], source_id=row["world_id"],
                            source_epoch=1, available_at=1)
    if len(spans) != 1:
        raise ValueError("diagnostic requires one canonical span per fitting world")
    with engine.measured("counterfactual_source_encoding"):
        batch = path.training_values(spans, scope=scope)
    # This is a cache hit after training_values; no second source forward is hidden.
    features = path.feature_provider.features(spans[0]).detach().float().cpu()
    return batch, features, scope


def _capture(path, position):
    captured = {site: {} for site in path.reader.ports}
    calls = {site: 0 for site in path.reader.ports}
    hooks = []
    for site, port in path.reader.ports.items():
        def projection(module, args, output, site=site):
            if calls[site]:
                return
            captured[site]["reader"] = args[0][0, position].detach().float().cpu()
            captured[site]["projection"] = output[0, position].detach().float().cpu()
        def residual(module, args, output, site=site):
            if calls[site]:
                return
            captured[site]["residual"] = output[0, position].detach().float().cpu()
            captured[site]["gate"] = float(module.gate_logit.detach().sigmoid())
            calls[site] = 1
        hooks.append(port.output_projection.register_forward_hook(projection))
        hooks.append(port.register_forward_hook(residual))
    return captured, hooks


def diagnose_pair(engine, left, right, pair_index, output, *, observe=None):
    if left["question"] != right["question"] or left["answer"] == right["answer"]:
        raise ValueError("same-question counterfactual pair required")
    rows = (left, right)
    batches, features, scopes = [], [], []
    for row in rows:
        batch, frozen, scope = _source(engine.path, engine, row)
        batches.append(batch)
        features.append(frozen)
        scopes.append(scope)
    logits = [[None for _ in CONTROLS] for _ in rows]
    reader = [[None for _ in CONTROLS] for _ in rows]
    projection = [[None for _ in CONTROLS] for _ in rows]
    residual = [[None for _ in CONTROLS] for _ in rows]
    metadata = [[None for _ in CONTROLS] for _ in rows]
    for side, row in enumerate(rows):
        for control_index, control in enumerate(CONTROLS):
            position = len(engine.tokens(row, exact=control == "exact_text")) - 1
            captured, hooks = _capture(engine.path, position)
            try:
                answer = engine.generate(row, batches[side], scopes[side], control=control,
                                         wrong=batches[side ^ 1], wrong_scope=scopes[side ^ 1])
            finally:
                for hook in hooks:
                    hook.remove()
            first = answer.pop("first_logits").detach().float().cpu()
            if control in ("absent", "exact_text"):
                if any(captured[site] for site in captured):
                    raise ValueError("disabled control executed a memory reader")
                for site in captured:
                    port = engine.path.reader.ports[site]
                    captured[site] = {
                        "reader": torch.zeros(port.query_projection.out_features),
                        "projection": torch.zeros(port.output_projection.out_features),
                        "residual": torch.zeros(port.output_projection.out_features),
                        "gate": float(port.gate_logit.detach().sigmoid()),
                    }
            if any(set(captured[site]) != {"reader", "projection", "residual", "gate"}
                   for site in captured):
                raise ValueError("active control did not expose every reader stage")
            logits[side][control_index] = first
            reader[side][control_index] = torch.stack(
                [captured[site]["reader"] for site in sorted(captured)])
            projection[side][control_index] = torch.stack(
                [captured[site]["projection"] for site in sorted(captured)])
            residual[side][control_index] = torch.stack(
                [captured[site]["residual"] for site in sorted(captured)])
            expected_id = engine.backend.answer_ids(row["answer"])[0]
            other_id = engine.backend.answer_ids(rows[side ^ 1]["answer"])[0]
            metadata[side][control_index] = {
                **answer, "control": control,
                "first_token_correct": int(first.argmax()) == expected_id,
                "expected_token_id": expected_id,
                "other_token_id": other_id,
                "expected_logit": float(first[expected_id]),
                "other_logit": float(first[other_id]),
                "expected_minus_other_logit": float(first[expected_id] - first[other_id]),
                "first_logits_sha256": _digest(first),
                "reader_gates": {str(site): captured[site]["gate"] for site in captured},
            }
            if observe is not None:
                observe()
    tensor_document = {
        "logits": torch.stack([torch.stack(side) for side in logits]),
        "reader": torch.stack([torch.stack(side) for side in reader]),
        "projection": torch.stack([torch.stack(side) for side in projection]),
        "residual": torch.stack([torch.stack(side) for side in residual]),
        "frozen_features_0": features[0], "frozen_features_1": features[1],
        "slots_0": batches[0].slots.detach().float().cpu(),
        "slots_1": batches[1].slots.detach().float().cpu(),
    }
    pair_dir = Path(output) / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    artifact = pair_dir / f"pair_{pair_index:02d}.safetensors"
    _atomic_tensors(artifact, tensor_document, engine.ledger, engine.stage)
    contextual = CONTROLS.index("contextual")
    control_comparisons = {}
    for index, control in enumerate(CONTROLS[1:], start=1):
        control_comparisons[control] = [
            _distribution_metrics(logits[side][contextual], logits[side][index])
            for side in range(2)]
    document = {
        "format": FORMAT + ".pair", "pair_index": pair_index,
        "world_ids": [row["world_id"] for row in rows],
        "question": left["question"], "expected_answers": [row["answer"] for row in rows],
        "controls": list(CONTROLS), "control_outputs": metadata,
        "both_contextual_correct": all(metadata[side][contextual]["correct"] for side in range(2)),
        "source_feature_distance": _distance(features[0], features[1]),
        "compressed_slot_distance": _distance(batches[0].slots, batches[1].slots),
        "reader_distance": [_distance(reader[0][contextual][site], reader[1][contextual][site])
                            for site in range(len(engine.path.reader.ports))],
        "residual_distance": [_distance(residual[0][contextual][site], residual[1][contextual][site])
                              for site in range(len(engine.path.reader.ports))],
        "contextual_distribution_distance": _distribution_metrics(
            logits[0][contextual], logits[1][contextual]),
        "control_logit_changes": control_comparisons,
        "tensor_artifact": str(artifact.relative_to(output)),
        "tensor_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "full_distribution_and_intermediates_preserved": True,
        "development_labels_used": False, "parameter_updates": 0,
    }
    with engine.measured("counterfactual_pair_receipt", writes=1):
        atomic(pair_dir / f"pair_{pair_index:02d}.json", document)
    return document


def run_diagnostic(engine, rows, output, *, observe=None):
    if len(rows) != 32:
        raise ValueError("the fixed fitting-only diagnostic requires 32 rows")
    documents = [diagnose_pair(engine, rows[index], rows[index + 1], index // 2, output,
                               observe=observe)
                 for index in range(0, 32, 2)]
    correct = {control: sum(document["control_outputs"][side][index]["correct"]
                            for document in documents for side in range(2))
               for index, control in enumerate(CONTROLS)}
    summary = {
        "format": FORMAT, "pairs": len(documents), "rows": 32,
        "controls": list(CONTROLS), "correct": correct,
        "counterfactual_pairs_both_correct": sum(
            document["both_contextual_correct"] for document in documents),
        "complete_responses": {control: {
            "terminated": sum(document["control_outputs"][side][index]["terminated"]
                              for document in documents for side in range(2)),
            "truncated": sum(document["control_outputs"][side][index]["truncated"]
                             for document in documents for side in range(2)),
        } for index, control in enumerate(CONTROLS)},
        "development_labels_used": False, "parameter_updates": 0,
        "native_capability_or_generalization_claim": False,
        "pair_receipts": [f"pairs/pair_{index:02d}.json" for index in range(16)],
    }
    with engine.measured("counterfactual_summary", writes=1):
        atomic(Path(output) / "diagnostic.json", summary)
    return summary
