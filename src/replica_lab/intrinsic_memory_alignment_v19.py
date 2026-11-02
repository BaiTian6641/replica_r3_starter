"""Matched direct-CE vs top-64 teacher-KL producer-reader alignment (V19).

Implements the bounded CPU-verifiable alignment comparison and diagnostic
separation between slot compression and frozen-decoder interface alignment,
as well as the fresh bounded paired-joint fitting successor protocol described in:
- docs/experiments/20261031_Spark_Research_Crosscheck_and_Alignment.md
- MemT-1.0 (arXiv:2610.26825v2, Appendix E.2.4: CE vs top-K KD tradeoffs)
- ICAE (arXiv:2307.06945v4: frozen-decoder memory slot interface)

This module is outside the frozen V18.2 execution closure and never loads a
native model or opens sealed evaluation labels.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
import math
import types
from typing import Any

import torch
import torch.nn.functional as F

from .contextual_completion_objective_v1 import encode_complete_response
from .contextual_memory import SLOTS, WIDTH, ContextualMemoryError, canonical_spans
from .intrinsic_memory_acceptance_v15 import (
    sequence_likelihood_parts,
    sequence_margin,
)
from .intrinsic_memory_interference_protocol_v17 import PAIR_INDICES
from .intrinsic_memory_measurement_v18 import (
    capture_state,
    digest,
    live_values,
    module_statistics,
    named_parameters,
    prediction_evidence,
    restore_state,
    verify_state,
)
from .intrinsic_memory_native_v5 import world_scope
from .intrinsic_memory_paired_update_v18 import validate_pairs
from .intrinsic_memory_probe_v18 import probe

ALIGNMENT_PROTOCOL = "spark_producer_reader_alignment_v19_protocol_v1"
DEFAULT_TOP_K = 64
VALID_ANSWERS = {"red", "blue", "green", "gold"}


def alignment_protocol_contract(
    *,
    top_k: int = DEFAULT_TOP_K,
    steps: int = 4,
    paired_joint: bool = True,
) -> dict[str, Any]:
    """Return the machine-checkable contract for matched CE vs top-K KL alignment."""
    if top_k < 2:
        raise ValueError("top_k must be at least 2")
    if steps < 1:
        raise ValueError("steps must be positive")
    return {
        "format": ALIGNMENT_PROTOCOL,
        "arms": ["direct_answer_plus_eos_ce", f"top_{top_k}_teacher_kl"],
        "sites": [3, 7],
        "intrinsic_reader_key_mode": "content",
        "slot_count": 4,
        "slot_dim": 256,
        "trainable_modules": ["producer", "reader"],
        "frozen_backbone": True,
        "frozen_source_feature_extractor": True,
        "optimizer": "NativeFusedAdamW",
        "top_k": int(top_k),
        "steps_per_arm": int(steps),
        "paired_joint_updates": bool(paired_joint),
        "exact_text_competence_qualification_required": True,
        "shuffled_probe_controls": True,
        "disjoint_diagnostic_split_supported": True,
        "terminal_state_only": True,
        "prefix_selection_forbidden": True,
        "common_behavioral_metrics": [
            "mean_final_pair_minimum_margin",
            "guard_count",
            "positive_pair_margin_worlds",
            "known_known_pair_both_correct_count",
            "full_vocabulary_first_token_correct",
            "teacher_forced_complete_correct",
            "mean_contextual_minus_wrong_read_margin",
            "mean_contextual_minus_absent_margin",
            "source_feature_probe_accuracy",
            "slot_probe_accuracy",
            "read_probe_accuracy",
            "source_feature_shuffled_probe_accuracy",
            "slot_shuffled_probe_accuracy",
            "read_shuffled_probe_accuracy",
            "slot_probe_minus_shuffled_gap",
        ],
        "loss_comparison_rule": (
            "Never declare a winner from raw CE vs raw KL scalar loss; "
            "compare both arms on identical post-update behavioral margins, "
            "counterfactual read gaps, and full-vocabulary correctness."
        ),
    }


def validate_alignment_pairs_v19(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate 4 known-known pairs (8 worlds) for V19 alignment or V18 compatibility."""
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("alignment protocol requires 8 worlds (4 pairs)")
    if all(str(r.get("world_id", "")).startswith("v11_fit_") for r in rows):
        return validate_pairs(rows)
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for pair_idx in range(4):
        left = dict(rows[2 * pair_idx])
        right = dict(rows[2 * pair_idx + 1])
        for side_row in (left, right):
            wid = str(side_row.get("world_id", "")).strip()
            if not wid or wid in seen_ids:
                raise ValueError(f"duplicate or missing world_id: {wid!r}")
            seen_ids.add(wid)
            ans = str(side_row.get("answer", "")).strip()
            if ans not in VALID_ANSWERS:
                raise ValueError(f"unsupported answer {ans!r} in {wid}")
            spans = canonical_spans(
                side_row["text"],
                record_id=wid,
                source_id=wid,
                source_epoch=1,
                available_at=1,
            )
            if len(spans) != 1:
                raise ValueError(f"world {wid} must produce exactly one canonical span")
        if left.get("question") != right.get("question"):
            raise ValueError(f"pair {pair_idx} worlds must share an identical question")
        if left.get("family") != right.get("family"):
            raise ValueError(f"pair {pair_idx} worlds must share an identical family")
        if left.get("answer") == right.get("answer"):
            raise ValueError(f"pair {pair_idx} worlds must have distinct answers")
        validated.extend([left, right])
    return validated


def _pair_index_for(rows: list[dict[str, Any]], pair_offset: int) -> int:
    row = rows[2 * pair_offset]
    if str(row.get("world_id", "")).startswith("v11_fit_"):
        return int(PAIR_INDICES[pair_offset])
    return int(row.get("group_id", pair_offset))


def _answer_slice(encoding) -> slice:
    begin = int(encoding.metadata["prompt_tokens"]) - 1
    end = int(encoding.metadata["terminal_prediction_position"]) + 1
    if end <= begin:
        raise ValueError("invalid answer-plus-EOS supervision slice")
    return slice(begin, end)


def _plain(parts: dict[str, Any]) -> dict[str, Any]:
    return {
        key: float(value.detach().cpu()) if torch.is_tensor(value) else value
        for key, value in parts.items()
    }


def _measured(engine, name: str, **counts):
    if hasattr(engine, "measured"):
        return engine.measured(name, **counts)
    return nullcontext()


def _pool_causal_feature_innovation(feat_tensor: torch.Tensor) -> torch.Tensor:
    """Pool causal source features using prefix-demeaned answer-tail innovation."""
    ft = feat_tensor.detach().to(device="cpu", dtype=torch.float32)
    if ft.ndim == 2 and ft.shape[0] >= 3:
        return ft[-2:].mean(dim=0) - ft[:-2].mean(dim=0)
    return ft.mean(dim=0)


def _tail_innovation_forward(self, spans):
    """Entity-invariant causal tail-innovation compressor forward for SourceFeatureValueEncoder."""
    if not spans:
        raise ContextualMemoryError("source encoder requires canonical spans")
    values = []
    for span in spans:
        frozen = self.provider.features(span).to(
            self.input_projection.weight.device
        )
        h = self.input_projection(self.input_norm(frozen))
        if len(h) > self.position.num_embeddings:
            raise ContextualMemoryError(
                "source sequence exceeds the compressor position bound"
            )
        if len(h) >= 4:
            prefix_mean = h[:-2].mean(dim=0)
            u0 = h[-2] - prefix_mean
            u1 = h[-1] - prefix_mean
            u2 = h[-2] - h[-3]
            u3 = h[-1] - 0.5 * (h[-3] + h[-4])
            u = torch.stack((u0, u1, u2, u3), dim=0)
            target_rms = h[-2:].pow(2).mean().add(1e-6).sqrt()
            u_rms = u.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            u = u * (target_rms / u_rms)
        elif len(h) == 3:
            prefix_mean = h[:-2].mean(dim=0)
            u = torch.stack(
                (
                    h[-2] - prefix_mean,
                    h[-1] - prefix_mean,
                    h[-2] - h[-3],
                    h[-1] - h[-3],
                ),
                dim=0,
            )
        else:
            u = h
        hidden = u + self.position(torch.arange(len(u), device=u.device))
        hidden = self.transformer(hidden[None])[0]
        slots = []
        for slot in range(SLOTS):
            weights = torch.softmax(
                hidden @ self.slot_queries[slot] / (WIDTH**0.5), dim=0
            )
            anchor = hidden[min(slot, len(hidden) - 1)]
            pooled = (weights[:, None] * hidden).sum(dim=0) + 0.5 * anchor
            slots.append(self.output(pooled))
        values.append(torch.stack(slots))
    return torch.stack(values)


def install_tail_innovation_encoder(path) -> None:
    """Bind the entity-invariant causal tail-innovation forward onto path.encoder."""
    encoder = getattr(path, "encoder", None)
    if encoder is not None and not getattr(
        encoder, "_tail_innovation_installed", False
    ):
        encoder._original_forward = encoder.forward
        encoder.forward = types.MethodType(_tail_innovation_forward, encoder)
        encoder._tail_innovation_installed = True


def _candidate_color_token_ids(engine) -> dict[str, int]:
    """Return the first token ID for each of the 4 canonical answer colors."""
    out: dict[str, int] = {}
    for color_name in ("red", "blue", "green", "gold"):
        ids = engine.backend.answer_ids(color_name)
        if ids:
            out[color_name] = int(ids[0])
    return out


def evaluate_exact_text_competence(
    engine,
    rows: list[dict[str, Any]],
    *,
    observe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Evaluate frozen-backbone exact-text competence with full source in prompt and no intrinsic memory."""
    rows = validate_alignment_pairs_v19(rows)
    full_rows = [
        {
            **row,
            "question": f"{row['text'].strip()}\n{row['question']}",
        }
        for row in rows
    ]
    prompts = [engine.tokens(r) for r in full_rows]
    answers = [engine.backend.answer_ids(r["answer"]) for r in rows]
    encodings = [
        [
            encode_complete_response(
                prompts[index],
                answers[other],
                spec=engine.spec,
                device=engine.device,
            )
            for other in (index, index ^ 1)
        ]
        for index in range(len(rows))
    ]
    worlds = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            sc = world_scope(row)
            candidates, predictions = [], []
            for encoding in encodings[index]:
                logits = engine.forward(
                    encoding.input_ids,
                    encoding.attention_mask,
                    None,
                    sc,
                )
                candidates.append(
                    _plain(
                        sequence_likelihood_parts(logits, encoding, spec=engine.spec)
                    )
                )
                predictions.append(prediction_evidence(logits, encoding))
            if observe is not None:
                observe()
            margin = sequence_margin(candidates[0], candidates[1])
            worlds.append(
                {
                    "world_id": row["world_id"],
                    "pair_index": _pair_index_for(rows, index // 2),
                    "expected": candidates[0],
                    "alternative": candidates[1],
                    "margin": margin,
                    "prediction": predictions[0],
                    "alternative_prediction": predictions[1],
                }
            )
    summary = _summarize_worlds(worlds)
    summary["competent"] = bool(
        summary["teacher_forced_complete_correct"] == len(rows)
        and summary["positive_pair_margin_worlds"] == len(rows)
    )
    return {
        "worlds": worlds,
        "summary": summary,
    }


def teacher_topk_targets(
    engine,
    row: dict[str, Any],
    *,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Extract frozen-backbone full-source teacher top-K log-probabilities on answer+EOS tokens."""
    if top_k < 2:
        raise ValueError("top_k must be at least 2")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    full_row = {
        **row,
        "question": f"{row['text'].strip()}\n{row['question']}",
    }
    prompt_ids = engine.tokens(full_row)
    answer_ids = engine.backend.answer_ids(row["answer"])
    encoding = encode_complete_response(
        prompt_ids, answer_ids, spec=engine.spec, device=engine.device
    )
    window = _answer_slice(encoding)
    sc = world_scope(row)
    with torch.no_grad():
        logits = engine.forward(
            encoding.input_ids,
            encoding.attention_mask,
            None,
            sc,
        )
        step_logits = logits[0, window].to(dtype=torch.float32) / float(temperature)
        k = min(int(top_k), int(step_logits.shape[-1]))
        top_logits, top_indices = torch.topk(step_logits, k=k, dim=-1)
        target_ids = encoding.labels[0, window].to(device=top_indices.device)
        color_ids = list(_candidate_color_token_ids(engine).values())
        eos_id = int(engine.spec.eos_token_id)
        for pos in range(target_ids.shape[0]):
            gold = int(target_ids[pos].item())
            must_ids = [gold]
            if k >= 6:
                for cid in [*color_ids, eos_id]:
                    if cid not in must_ids and 0 <= cid < step_logits.shape[-1]:
                        must_ids.append(cid)
            replace_slot = k - 1
            for req_id in must_ids:
                if not bool((top_indices[pos] == req_id).any()) and replace_slot >= 0:
                    while (
                        replace_slot > 0
                        and int(top_indices[pos, replace_slot].item()) in must_ids
                    ):
                        replace_slot -= 1
                    top_indices[pos, replace_slot] = req_id
                    top_logits[pos, replace_slot] = step_logits[pos, req_id]
                    replace_slot -= 1
        teacher_log_probs = F.log_softmax(top_logits, dim=-1)
    return {
        "top_indices": top_indices.detach(),
        "teacher_log_probs": teacher_log_probs.detach(),
        "supervised_tokens": int(target_ids.numel()),
        "temperature": float(temperature),
        "top_k": int(k),
    }


def _summarize_worlds(
    worlds: list[dict[str, Any]],
    baseline_margins: dict[str, float] | None = None,
) -> dict[str, Any]:
    margins = [w["margin"]["total_log_likelihood"] for w in worlds]
    minima = [min(margins[i : i + 2]) for i in range(0, len(margins), 2)]
    guard_count = (
        0
        if baseline_margins is None
        else sum(
            w["margin"]["total_log_likelihood"] < baseline_margins[w["world_id"]]
            for w in worlds
        )
    )
    known_known_both = 0
    for i in range(0, len(worlds), 2):
        w0, w1 = worlds[i], worlds[i + 1]
        if (
            w0["prediction"]["complete_teacher_forced_correct"]
            and w1["prediction"]["complete_teacher_forced_correct"]
            and w0["margin"]["total_log_likelihood"] > 0.0
            and w1["margin"]["total_log_likelihood"] > 0.0
        ):
            known_known_both += 1
    return {
        "mean_final_pair_minimum_margin": sum(minima) / len(minima),
        "pair_minimum_margins": minima,
        "absolute_final_margins": margins,
        "guard_count": guard_count,
        "positive_pair_margin_worlds": sum(m > 0.0 for m in margins),
        "known_known_pair_both_correct_count": known_known_both,
        "full_vocabulary_first_token_correct": sum(
            int(w["prediction"]["first_token_correct"]) for w in worlds
        ),
        "teacher_forced_complete_correct": sum(
            int(w["prediction"]["complete_teacher_forced_correct"]) for w in worlds
        ),
    }


def _prepare_split_bindings(engine, split_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scopes = [world_scope(row) for row in split_rows]
    spans = [
        canonical_spans(
            row["text"],
            record_id=row["world_id"],
            source_id=row["world_id"],
            source_epoch=1,
            available_at=1,
        )
        for row in split_rows
    ]
    prompts = [engine.tokens(row) for row in split_rows]
    answers = [engine.backend.answer_ids(row["answer"]) for row in split_rows]
    encodings = [
        [
            encode_complete_response(
                prompts[index],
                answers[other],
                spec=engine.spec,
                device=engine.device,
            )
            for other in (index, index ^ 1)
        ]
        for index in range(len(split_rows))
    ]
    return {
        "rows": split_rows,
        "scopes": scopes,
        "spans": spans,
        "prompts": prompts,
        "answers": answers,
        "encodings": encodings,
    }


def _evaluate_split(
    engine,
    bindings: dict[str, Any],
    baseline_margins: dict[str, float] | None = None,
    *,
    observe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    path = engine.path
    split_rows = bindings["rows"]
    scopes = bindings["scopes"]
    spans = bindings["spans"]
    encodings = bindings["encodings"]

    worlds = []
    feature_vectors = []
    slot_vectors = []
    read_vectors = []
    gold_answers = []
    identities = []
    absent_gaps = []
    wrong_gaps = []

    with torch.no_grad():
        live_batches = [
            live_values(engine, spans[idx], scopes[idx])
            for idx in range(len(split_rows))
        ]
        for index, row in enumerate(split_rows):
            live = live_batches[index]
            other_live = live_batches[index ^ 1]
            other_scope = scopes[index ^ 1]
            candidates, predictions = [], []
            absent_candidates, wrong_candidates = [], []
            for encoding in encodings[index]:
                logits = engine.forward(
                    encoding.input_ids,
                    encoding.attention_mask,
                    live,
                    scopes[index],
                )
                candidates.append(
                    _plain(
                        sequence_likelihood_parts(logits, encoding, spec=engine.spec)
                    )
                )
                predictions.append(prediction_evidence(logits, encoding))

                absent_logits = engine.forward(
                    encoding.input_ids,
                    encoding.attention_mask,
                    None,
                    scopes[index],
                )
                absent_candidates.append(
                    _plain(
                        sequence_likelihood_parts(
                            absent_logits, encoding, spec=engine.spec
                        )
                    )
                )

                wrong_logits = engine.forward(
                    encoding.input_ids,
                    encoding.attention_mask,
                    other_live,
                    other_scope,
                )
                wrong_candidates.append(
                    _plain(
                        sequence_likelihood_parts(
                            wrong_logits, encoding, spec=engine.spec
                        )
                    )
                )
            if observe is not None:
                observe()

            ctx_margin = sequence_margin(candidates[0], candidates[1])[
                "total_log_likelihood"
            ]
            abs_margin = sequence_margin(
                absent_candidates[0], absent_candidates[1]
            )["total_log_likelihood"]
            wrg_margin = sequence_margin(
                wrong_candidates[0], wrong_candidates[1]
            )["total_log_likelihood"]
            absent_gaps.append(ctx_margin - abs_margin)
            wrong_gaps.append(ctx_margin - wrg_margin)

            p_idx = _pair_index_for(split_rows, index // 2)
            worlds.append(
                {
                    "world_id": row["world_id"],
                    "pair_index": p_idx,
                    "expected": candidates[0],
                    "alternative": candidates[1],
                    "margin": sequence_margin(candidates[0], candidates[1]),
                    "prediction": predictions[0],
                    "alternative_prediction": predictions[1],
                    "counterfactual": {
                        "absent_margin": abs_margin,
                        "wrong_read_margin": wrg_margin,
                        "contextual_minus_absent_margin": ctx_margin - abs_margin,
                        "contextual_minus_wrong_read_margin": ctx_margin - wrg_margin,
                    },
                    "slot_digest": digest(live.slots),
                }
            )

            feat_tensor = path.feature_provider.features(spans[index][0])
            pooled_feat = _pool_causal_feature_innovation(feat_tensor)
            pooled_slot = (
                live.slots.detach()
                .to(device="cpu", dtype=torch.float32)
                .reshape(-1)
            )
            q_tensor = feat_tensor.detach().unsqueeze(0)
            ctx = path.snapshot(q_tensor[:, :, 0].long(), live, scope=scopes[index])
            reads = []
            for site in path.spec.sites:
                r = ctx.delta(site, q_tensor)
                reads.append(_pool_causal_feature_innovation(r[0]))
            feature_vectors.append(pooled_feat)
            slot_vectors.append(pooled_slot)
            read_vectors.append(torch.cat(reads, dim=-1))
            gold_answers.append(row["answer"])
            identities.append(
                {
                    "world_id": str(row["world_id"]),
                    "pair_index": int(p_idx),
                }
            )

    summary = _summarize_worlds(worlds, baseline_margins)
    feat_stack = torch.stack(feature_vectors, dim=0)
    slot_stack = torch.stack(slot_vectors, dim=0)
    read_stack = torch.stack(read_vectors, dim=0)
    shuffled_perm = [(i + 2) % len(split_rows) for i in range(len(split_rows))]

    feature_probe = probe(feat_stack, gold_answers, identities=identities)
    slot_probe = probe(slot_stack, gold_answers, identities=identities)
    read_probe = probe(read_stack, gold_answers, identities=identities)

    feature_probe_shuffled = probe(
        feat_stack, gold_answers, identities=identities, shuffled=shuffled_perm
    )
    slot_probe_shuffled = probe(
        slot_stack, gold_answers, identities=identities, shuffled=shuffled_perm
    )
    read_probe_shuffled = probe(
        read_stack, gold_answers, identities=identities, shuffled=shuffled_perm
    )

    feat_acc = sum(int(r["correct"]) for r in feature_probe) / len(feature_probe)
    slot_acc = sum(int(r["correct"]) for r in slot_probe) / len(slot_probe)
    read_acc = sum(int(r["correct"]) for r in read_probe) / len(read_probe)

    feat_shuf_acc = sum(int(r["correct"]) for r in feature_probe_shuffled) / len(
        feature_probe_shuffled
    )
    slot_shuf_acc = sum(int(r["correct"]) for r in slot_probe_shuffled) / len(
        slot_probe_shuffled
    )
    read_shuf_acc = sum(int(r["correct"]) for r in read_probe_shuffled) / len(
        read_probe_shuffled
    )

    pos_rate = summary["positive_pair_margin_worlds"] / len(worlds)
    mean_wrong_gap = sum(wrong_gaps) / len(wrong_gaps)
    mean_absent_gap = sum(absent_gaps) / len(absent_gaps)

    if slot_acc < 0.5:
        regime = "compression_bottleneck"
    elif pos_rate < 0.75 or mean_wrong_gap <= 0.0:
        regime = "decoder_interface_bottleneck"
    else:
        regime = "aligned_contextual_recall"

    summary.update(
        {
            "mean_contextual_minus_absent_margin": mean_absent_gap,
            "mean_contextual_minus_wrong_read_margin": mean_wrong_gap,
            "source_feature_probe_accuracy": feat_acc,
            "slot_probe_accuracy": slot_acc,
            "read_probe_accuracy": read_acc,
            "source_feature_shuffled_probe_accuracy": feat_shuf_acc,
            "slot_shuffled_probe_accuracy": slot_shuf_acc,
            "read_shuffled_probe_accuracy": read_shuf_acc,
            "slot_probe_minus_shuffled_gap": slot_acc - slot_shuf_acc,
            "diagnostic_regime": regime,
        }
    )
    return {
        "worlds": worlds,
        "summary": summary,
        "probes": {
            "source_features": feature_probe,
            "slots": slot_probe,
            "reads": read_probe,
            "source_features_shuffled": feature_probe_shuffled,
            "slots_shuffled": slot_probe_shuffled,
            "reads_shuffled": read_probe_shuffled,
            "shuffled_permutation": shuffled_perm,
        },
    }


def run_alignment_comparison(
    engine,
    rows: list[dict[str, Any]],
    *,
    diagnostic_rows: list[dict[str, Any]] | None = None,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = 1.0,
    steps: int = 4,
    paired_joint: bool = True,
    require_competence: bool = False,
    observe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run matched direct-CE vs top-K teacher-KL alignment from identical initial state."""
    rows = validate_alignment_pairs_v19(rows)
    if diagnostic_rows is not None:
        diagnostic_rows = validate_alignment_pairs_v19(diagnostic_rows)
        fit_ids = {r["world_id"] for r in rows}
        diag_ids = {r["world_id"] for r in diagnostic_rows}
        if fit_ids & diag_ids:
            raise ValueError("diagnostic_rows must be disjoint from fitting rows")

    path = engine.path
    install_tail_innovation_encoder(path)
    color_ids_list = list(_candidate_color_token_ids(engine).values())
    eos_id = int(engine.spec.eos_token_id)
    contract = alignment_protocol_contract(
        top_k=top_k, steps=steps, paired_joint=paired_joint
    )
    fit_bindings = _prepare_split_bindings(engine, rows)
    diag_bindings = (
        _prepare_split_bindings(engine, diagnostic_rows)
        if diagnostic_rows is not None
        else None
    )

    initial = capture_state(path)
    exact_text_fit = evaluate_exact_text_competence(
        engine, rows, observe=observe
    )
    exact_text_diag = (
        evaluate_exact_text_competence(
            engine, diagnostic_rows, observe=observe
        )
        if diagnostic_rows is not None
        else None
    )
    if require_competence:
        if not exact_text_fit["summary"]["competent"]:
            raise ValueError(
                "exact-text competence qualification failed on fitting split; stopping both arms"
            )
        if exact_text_diag is not None and not exact_text_diag["summary"]["competent"]:
            raise ValueError(
                "exact-text competence qualification failed on diagnostic split; stopping both arms"
            )

    teacher_cache = {
        row["world_id"]: teacher_topk_targets(
            engine, row, top_k=top_k, temperature=temperature
        )
        for row in rows
    }
    if observe is not None:
        observe()

    arm_specs = [
        ("direct_answer_plus_eos_ce", "direct_answer_plus_eos_ce"),
        (f"top_{top_k}_teacher_kl", "top_k_teacher_kl"),
    ]
    arms_out: dict[str, Any] = {}
    restoration_checks = []
    spans = fit_bindings["spans"]
    scopes = fit_bindings["scopes"]
    encodings = fit_bindings["encodings"]

    try:
        for arm_name, objective in arm_specs:
            restoration_checks.append(restore_state(path, initial))
            baseline_eval = _evaluate_split(
                engine, fit_bindings, observe=observe
            )
            baseline_margins = {
                w["world_id"]: w["margin"]["total_log_likelihood"]
                for w in baseline_eval["worlds"]
            }
            diag_baseline_eval = (
                _evaluate_split(engine, diag_bindings, observe=observe)
                if diag_bindings is not None
                else None
            )
            diag_baseline_margins = (
                {
                    w["world_id"]: w["margin"]["total_log_likelihood"]
                    for w in diag_baseline_eval["worlds"]
                }
                if diag_baseline_eval is not None
                else None
            )
            updates = []
            color_slot_prototypes: dict[str, torch.Tensor] = {}
            groups = (
                [[2 * offset, 2 * offset + 1] for offset in range(4)]
                if paired_joint
                else [[idx] for idx in range(len(rows))]
            )
            for step_idx in range(steps):
                group = groups[step_idx % len(groups)]
                path.optimizer.zero_grad(set_to_none=True)
                before = {
                    name: value.detach().clone()
                    for name, value in named_parameters(path)
                }
                world_meta = []
                step_losses = []
                for index in group:
                    live = live_values(engine, spans[index], scopes[index])
                    encoding = encodings[index][0]
                    alt_encoding = encodings[index][1]
                    logits = engine.forward(
                        encoding.input_ids,
                        encoding.attention_mask,
                        live,
                        scopes[index],
                    )
                    parts = sequence_likelihood_parts(
                        logits, encoding, spec=engine.spec
                    )
                    ce_loss = (
                        -parts["total_log_likelihood"] / parts["supervised_tokens"]
                    )
                    first_pos = int(encoding.metadata["prompt_tokens"]) - 1
                    eos_pos = int(
                        encoding.metadata["terminal_prediction_position"]
                    )
                    first_logits = logits[0, first_pos].to(dtype=torch.float32)
                    gold_id = int(encoding.labels[0, first_pos].item())
                    alt_id = int(alt_encoding.labels[0, first_pos].item())
                    gold_logit = first_logits[gold_id]
                    alt_logit = first_logits[alt_id]
                    pair_tok_margin = gold_logit - alt_logit
                    pair_margin_loss = F.softplus(
                        5.0 - pair_tok_margin
                    ) + 0.08 * F.relu(pair_tok_margin - 5.5).pow(2)

                    rival_ids = [cid for cid in color_ids_list if cid != gold_id]
                    if rival_ids:
                        rival_logits = first_logits[rival_ids]
                        color_4way_loss = F.softplus(
                            5.0 - (gold_logit - rival_logits)
                        ).mean()
                    else:
                        color_4way_loss = torch.zeros_like(ce_loss)
                    if gold_id in color_ids_list and len(color_ids_list) >= 2:
                        c_idx = color_ids_list.index(gold_id)
                        color_4way_loss = (
                            color_4way_loss
                            - 0.5
                            * F.log_softmax(first_logits[color_ids_list], dim=-1)[
                                c_idx
                            ]
                        )

                    if eos_pos > first_pos and color_ids_list:
                        eos_logits = logits[0, eos_pos].to(dtype=torch.float32)
                        eos_guard_loss = F.softplus(
                            5.0 - (eos_logits[eos_id] - eos_logits[color_ids_list])
                        ).mean()
                    else:
                        eos_guard_loss = torch.zeros_like(ce_loss)

                    ans_color = str(rows[index]["answer"])
                    s_float = live.slots.to(dtype=torch.float32)
                    s_norm = F.normalize(s_float, dim=-1)
                    slot_proto_loss = (
                        0.02 * (s_float.norm(dim=-1) - 4.0).pow(2).mean()
                    )
                    if ans_color in color_slot_prototypes:
                        proto = color_slot_prototypes[ans_color]
                        slot_proto_loss = (
                            slot_proto_loss
                            + (1.0 - (s_norm * proto).sum(dim=-1)).mean()
                        )
                    rival_protos = [
                        p
                        for c, p in color_slot_prototypes.items()
                        if c != ans_color
                    ]
                    if rival_protos:
                        slot_proto_loss = slot_proto_loss + sum(
                            F.relu((s_norm * rp).sum(dim=-1) + 0.25).mean()
                            for rp in rival_protos
                        ) / float(len(rival_protos))
                    color_slot_prototypes[ans_color] = (
                        s_norm.detach()
                        if ans_color not in color_slot_prototypes
                        else F.normalize(
                            0.8 * color_slot_prototypes[ans_color]
                            + 0.2 * s_norm.detach(),
                            dim=-1,
                        )
                    )

                    if objective == "direct_answer_plus_eos_ce":
                        loss = (
                            ce_loss
                            + 0.25 * color_4way_loss
                            + 0.20 * pair_margin_loss
                            + 0.15 * eos_guard_loss
                            + 0.35 * slot_proto_loss
                        )
                        kl_val = None
                    else:
                        window = _answer_slice(encoding)
                        step_logits = logits[0, window].to(dtype=torch.float32)
                        teacher = teacher_cache[rows[index]["world_id"]]
                        indices = teacher["top_indices"].to(device=step_logits.device)
                        t_log_probs = teacher["teacher_log_probs"].to(
                            device=step_logits.device, dtype=torch.float32
                        )
                        full_s_log_probs = F.log_softmax(
                            step_logits / float(temperature), dim=-1
                        )
                        s_log_probs = torch.gather(
                            full_s_log_probs,
                            dim=-1,
                            index=indices,
                        )
                        t_probs = t_log_probs.exp()
                        kl_loss = (
                            (t_probs * (t_log_probs - s_log_probs))
                            .sum(dim=-1)
                            .mean()
                            * (float(temperature) ** 2)
                        )
                        kl_val = float(kl_loss.detach().cpu())
                        loss = (
                            kl_loss
                            + 0.25 * color_4way_loss
                            + 0.20 * pair_margin_loss
                            + 0.20 * eos_guard_loss
                            + 0.35 * slot_proto_loss
                        )
                    with _measured(engine, "per_world_backward", backwards=1):
                        (loss / len(group)).backward()
                    step_losses.append(float(loss.detach().cpu()))
                    world_meta.append(
                        {
                            "world_id": rows[index]["world_id"],
                            "supervised_tokens": int(parts["supervised_tokens"]),
                            "per_world_mean_CE": float(ce_loss.detach().cpu()),
                            "per_world_mean_top_k_KL": kl_val,
                            "slot_digest": digest(live.slots),
                        }
                    )
                grads = module_statistics(path)
                if not all(item["finite"] for item in grads.values()):
                    raise ValueError(f"nonfinite gradients in {arm_name}")
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for _, p in named_parameters(path)
                        if p.grad is not None
                    ],
                    max_norm=5.0,
                )
                with _measured(engine, "native_fused_adamw_step", updates=1):
                    path.optimizer.step()
                    path.optimizer.zero_grad(set_to_none=True)
                displacement = module_statistics(path, before=before)
                if not all(item["finite"] for item in displacement.values()):
                    raise ValueError(f"nonfinite displacement in {arm_name}")
                if observe is not None:
                    observe()
                updates.append(
                    {
                        "step": step_idx + 1,
                        "objective": objective,
                        "world_ids": [m["world_id"] for m in world_meta],
                        "optimized_loss": sum(step_losses) / len(step_losses),
                        "worlds": world_meta,
                        "per_module_gradients": grads,
                        "actual_adamw_displacement": displacement,
                    }
                )
            final_eval = _evaluate_split(
                engine,
                fit_bindings,
                baseline_margins=baseline_margins,
                observe=observe,
            )
            arm_payload: dict[str, Any] = {
                "baseline": baseline_eval["summary"],
                "final": final_eval["summary"],
                "updates": updates,
                "final_worlds": final_eval["worlds"],
                "probes": final_eval["probes"],
            }
            if diag_bindings is not None and diag_baseline_eval is not None:
                diag_final_eval = _evaluate_split(
                    engine,
                    diag_bindings,
                    baseline_margins=diag_baseline_margins,
                    observe=observe,
                )
                arm_payload["disjoint_diagnostic"] = {
                    "baseline": diag_baseline_eval["summary"],
                    "final": diag_final_eval["summary"],
                    "final_worlds": diag_final_eval["worlds"],
                    "probes": diag_final_eval["probes"],
                }
            arms_out[arm_name] = arm_payload
    finally:
        restoration_checks.append(restore_state(path, initial))

    verify_state(path, initial)
    ce_final = arms_out["direct_answer_plus_eos_ce"]["final"]
    kl_name = f"top_{top_k}_teacher_kl"
    kl_final = arms_out[kl_name]["final"]
    ce_minus_kl = (
        ce_final["mean_final_pair_minimum_margin"]
        - kl_final["mean_final_pair_minimum_margin"]
    )
    ce_guard_no_worse = ce_final["guard_count"] <= kl_final["guard_count"]
    ce_complete_no_worse = (
        ce_final["teacher_forced_complete_correct"]
        >= kl_final["teacher_forced_complete_correct"]
        and ce_final["known_known_pair_both_correct_count"]
        >= kl_final["known_known_pair_both_correct_count"]
    )
    return {
        "protocol": contract,
        "exact_text_competence": {
            "fit": exact_text_fit,
            "disjoint_diagnostic": exact_text_diag,
        },
        "arms": arms_out,
        "state_restoration": restoration_checks,
        "comparison": {
            "ce_minus_kl_pair_minimum_margin": ce_minus_kl,
            "ce_guard_no_worse": ce_guard_no_worse,
            "ce_complete_correct_no_worse": ce_complete_no_worse,
            "ce_primary_support_passed": bool(
                ce_minus_kl > 0.0 and ce_guard_no_worse and ce_complete_no_worse
            ),
            "kl_minus_ce_pair_minimum_margin": -ce_minus_kl,
            "kl_guard_no_worse": kl_final["guard_count"] <= ce_final["guard_count"],
            "kl_wrong_read_gap_delta": (
                kl_final["mean_contextual_minus_wrong_read_margin"]
                - ce_final["mean_contextual_minus_wrong_read_margin"]
            ),
        },
    }
