"""Training-only MiniCPM5 vision-collapse evidence contracts.

This module owns the reusable, model-free half of the collapse diagnostic:

* the declared generation stop set (``</s>`` *and* ``<|im_end|>``), stop-reason
  evidence and the prefill/greedy agreement check;
* per-position shifted-NLL, first-token, top-k, rank, gap and entropy math;
* answer/caption pair layout and matched control selection that never consults
  an expected target text;
* text- and feature-ceiling summaries (frozen encoder features only);
* the fit-contract per-row report and coverage audit used by the training
  runners.

Everything here is deterministic, CPU-only and importable without torch,
``transformers`` or a model. Native loading, forwards and generation live in
``scripts/run_vision_collapse_diagnostic.py``.

No function in this module promotes learned quality, assigns a verdict or claims
perception/semantic competence. It reports first-order evidence only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


DIAGNOSTIC_FORMAT = "replica.vision_collapse_diagnostic.v1"
CONFIG_FORMAT = "replica.vision_collapse_diagnostic_config.v1"
REPORT_FORMAT = "replica.vision_collapse_diagnostic.report.v1"
CELL_FORMAT = "replica.vision_collapse_diagnostic.cell.v1"
ROW_FORMAT = "replica.vision_collapse_diagnostic.row.v1"

CONTROL_NAMES = ("correct", "sibling", "shuffled", "noimage")

# The native snapshot declares ``eos_token_id = [1, 130073]``: ``</s>`` and
# ``<|im_end|>``.  Generation stops on either.  Reporting only the tokenizer's
# ``</s>`` loses every ``<|im_end|>`` stop, which is an evidence defect.
EXPECTED_STOP_TOKENS = (1, 130073)
LABEL_IGNORE_INDEX = -100
LABEL_SHIFT = 1
LOSS_NAME = "mean_response_nll"


class DiagnosticError(RuntimeError):
    """Raised when an evidence contract or declared pin is violated."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


# ---------------------------------------------------------------------------
# Stop set and generation evidence
# ---------------------------------------------------------------------------


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple)) else (value,)
    if not values:
        raise DiagnosticError(f"{name} must not be empty")
    if any(type(item) is not int or item < 0 for item in values):
        raise DiagnosticError(f"{name} must be nonnegative integers")
    return tuple(sorted(set(values)))


def resolve_stop_tokens(config_eos: Any, tokenizer_eos: Any = None) -> tuple[tuple[int, ...], str]:
    """Return the declared stop set and where it came from.

    The model config is authoritative (``[1, 130073]``).  The tokenizer EOS is
    a declared fallback for CPU doubles that do not carry a model config; the
    fallback is reported so a reader can see the stop set was not the native
    one.
    """
    configured = _int_tuple(config_eos, "model eos_token_id")
    if configured:
        return configured, "model_config"
    fallback = _int_tuple(tokenizer_eos, "tokenizer eos_token_id")
    if fallback:
        return fallback, "tokenizer_fallback"
    raise DiagnosticError("No generation stop token is declared")


def stop_evidence(generated_ids: Sequence[int], stops: Sequence[int]) -> dict[str, Any]:
    """Classify how one generation sequence ended.

    ``ended_by_eos`` means the final emitted token belongs to the declared stop
    set.  A stop token before the final position is reported separately; it can
    only arise from a caller that kept decoding past a stop, which is evidence
    of a broken stop contract rather than a normal finish.
    """
    ids = [int(value) for value in generated_ids]
    require(ids, "Generation produced no tokens")
    stop_set = set(_int_tuple(list(stops), "stops"))
    require(stop_set, "Generation requires a declared stop set")
    last = ids[-1]
    earlier = [index for index, token in enumerate(ids[:-1]) if token in stop_set]
    ended = last in stop_set
    return {
        "generated_tokens": len(ids),
        "ended_by_eos": ended,
        "stop_token": last if ended else None,
        "stop_reason": "stop_token" if ended else "token_budget",
        "stop_tokens": sorted(stop_set),
        "stop_token_before_final": earlier[0] if earlier else None,
    }


def generation_evidence(generated_ids: Sequence[int], generated_text: str, *,
                        stops: Sequence[int], prefill_argmax: int,
                        max_new_tokens: int) -> dict[str, Any]:
    """Combine stop evidence, the token budget and greedy/prefill agreement."""
    require(type(max_new_tokens) is int and max_new_tokens > 0,
            "max_new_tokens must be a positive integer")
    require(type(prefill_argmax) is int and prefill_argmax >= 0,
            "prefill argmax must be a nonnegative token id")
    evidence = stop_evidence(generated_ids, stops)
    ids = [int(value) for value in generated_ids]
    ended = evidence["ended_by_eos"]
    evidence.update({
        "generated_ids": ids,
        "generated_text": str(generated_text),
        "first_generated_token": ids[0],
        "prefill_argmax": int(prefill_argmax),
        "prefill_argmax_matches_first_token": ids[0] == int(prefill_argmax),
        "max_new_tokens": int(max_new_tokens),
        "hit_token_budget": (not ended) and len(ids) >= int(max_new_tokens),
    })
    return evidence


# ---------------------------------------------------------------------------
# Logit math
# ---------------------------------------------------------------------------


def _torch_like(row: Any) -> bool:
    return hasattr(row, "logsumexp") and hasattr(row, "argmax") and hasattr(row, "numel")


def _row_values(row: Any) -> list[float]:
    values = row.tolist() if hasattr(row, "tolist") else list(row)
    return [float(value) for value in values]


def _log_softmax_values(row: Any) -> list[float]:
    values = _row_values(row)
    require(values, "A logit row must not be empty")
    maximum = max(values)
    terms = [value - maximum for value in values]
    total = math.log(sum(math.exp(term) for term in terms))
    return [term - total for term in terms]


def _log_softmax(row: Any) -> Any:
    """Return a log-softmax over one vocabulary row (torch or list)."""
    if not _torch_like(row):
        return _log_softmax_values(row)
    import torch
    with torch.no_grad():
        return torch.log_softmax(row.detach().float(), dim=-1)


def entropy(row: Any) -> float:
    """Shannon entropy in nats of one logit row."""
    if not _torch_like(row):
        logp = _log_softmax_values(row)
        return float(-sum(value * math.exp(value) for value in logp))
    import torch
    with torch.no_grad():
        logp = torch.log_softmax(row.detach().float(), dim=-1)
        return float(-(logp.exp() * logp).sum().item())


def rank_of(row: Any, token: int) -> int:
    """One-based descending rank, ties broken against the token (strict >)."""
    require(type(token) is int and token >= 0, "Rank token must be nonnegative")
    if _torch_like(row):
        with_no_grad = row.detach()
        require(int(token) < int(with_no_grad.numel()),
                "Rank token is outside the vocabulary")
        return int((with_no_grad > with_no_grad[int(token)]).sum().item()) + 1
    values = _row_values(row)
    require(token_inside(values, token), "Rank token is outside the vocabulary")
    target = values[token]
    return sum(1 for value in values if value > target) + 1


def token_inside(values: Sequence[float], token: int) -> bool:
    return 0 <= int(token) < len(values)


def logit_of(row: Any, token: int) -> float:
    if _torch_like(row):
        return float(row.detach()[int(token)].item())
    values = _row_values(row)
    require(token_inside(values, token), "Token is outside the vocabulary")
    return float(values[int(token)])


def top_k(row: Any, k: int = 5) -> list[dict[str, Any]]:
    """Deterministic descending top-k; ties break toward the smaller token id."""
    require(type(k) is int and k > 0, "top_k requires a positive k")
    if _torch_like(row):
        import torch
        detached = row.detach().float()
        k = min(int(k), int(detached.numel()))
        values, indices = torch.topk(detached, k)
        pairs = sorted(((int(token), float(value))
                        for token, value in zip(indices.tolist(), values.tolist())),
                       key=lambda item: (-item[1], item[0]))
        return [{"token": token, "logit": value} for token, value in pairs]
    values = _row_values(row)
    require(values, "A logit row must not be empty")
    k = min(k, len(values))
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return [{"token": index, "logit": values[index]} for index in order[:k]]


def first_token_analysis(logits_row: Any, gold_token: int, *,
                         opposite_token: int | None = None,
                         top_k_size: int = 5) -> dict[str, Any]:
    """Teacher-forced first-answer evidence at the prefill decision boundary."""
    gold = int(gold_token)
    row = logits_row
    result = {
        "gold_token": gold,
        "gold_logit": logit_of(row, gold),
        "gold_rank": rank_of(row, gold),
        "argmax_token": _argmax(row),
        "argmax_is_gold": _argmax(row) == gold,
        "top_k": top_k(row, top_k_size),
        "entropy": entropy(row),
        "vocabulary": _vocabulary(row),
    }
    if opposite_token is not None:
        opposite = int(opposite_token)
        result.update({
            "opposite_token": opposite,
            "opposite_logit": logit_of(row, opposite),
            "opposite_rank": rank_of(row, opposite),
            "gap": result["gold_logit"] - logit_of(row, opposite),
            "opposite_is_gold": opposite == gold,
        })
    else:
        result.update({"opposite_token": None, "opposite_logit": None,
                       "opposite_rank": None, "gap": None,
                       "opposite_is_gold": None})
    return result


def _argmax(row: Any) -> int:
    if _torch_like(row):
        return int(row.detach().argmax().item())
    values = _row_values(row)
    require(values, "A logit row must not be empty")
    return min(range(len(values)), key=lambda index: (-values[index], index))


def _vocabulary(row: Any) -> int:
    if hasattr(row, "numel"):
        return int(row.numel())
    return len(list(row))


def eos_pair_analysis(logits_row: Any, stops: Sequence[int]) -> dict[str, Any]:
    """Compare the declared stop tokens at the position that predicts EOS."""
    stop_set = _int_tuple(list(stops), "stops")
    require(stop_set, "EOS analysis requires the declared stop set")
    result = {
        "stop_tokens": list(stop_set),
        "argmax_token": _argmax(logits_row),
        "argmax_is_stop": _argmax(logits_row) in set(stop_set),
        "entropy": entropy(logits_row),
    }
    for token in stop_set:
        result[f"stop_{token}_logit"] = logit_of(logits_row, token)
        result[f"stop_{token}_rank"] = rank_of(logits_row, token)
    if len(stop_set) >= 2:
        first, second = stop_set[0], stop_set[1]
        result["stop_gap"] = result[f"stop_{first}_logit"] - result[f"stop_{second}_logit"]
        result["preferred_stop_token"] = (first if result["stop_gap"] > 0
                                          else (second if result["stop_gap"] < 0
                                                else None))
    return result


def position_nll(logits: Sequence[Any], labels: Sequence[int], prompt_tokens: int,
                 *, label_shift: int = LABEL_SHIFT) -> dict[str, Any]:
    """Per-position shifted next-token NLL for the supervised response.

    ``logits[q - 1]`` predicts ``labels[q]`` (``label_shift == 1``).  Only
    positions with ``labels[q] != -100`` are scored.  The first scored position
    is the first response token, predicted by the final prompt row; the rest are
    continuation tokens.  Count fields name the shifted convention explicitly.
    """
    require(label_shift == LABEL_SHIFT,
            "Only the native one-position label shift is supported")
    require(len(logits) == len(labels),
            "Text-aligned logits and labels differ in length")
    require(type(prompt_tokens) is int and prompt_tokens >= 1,
            "prompt_tokens must be a positive integer")
    require(prompt_tokens <= len(labels),
            "The prompt prefix exceeds the supervised sequence")
    label_values = [int(value) for value in labels]
    supervised = sum(value != LABEL_IGNORE_INDEX for value in label_values)
    shifted = sum(value != LABEL_IGNORE_INDEX for value in label_values[1:])
    prompt_masked = sum(value == LABEL_IGNORE_INDEX
                        for value in label_values[:prompt_tokens])
    positions: list[dict[str, Any]] = []
    for target_position in range(prompt_tokens, len(label_values)):
        target = label_values[target_position]
        if target == LABEL_IGNORE_INDEX:
            continue
        row = logits[target_position - 1]
        logp = _log_softmax(row)
        if _torch_like(row):
            nll = float(-logp[int(target)].item())
        else:
            nll = float(-logp[int(target)])
        positions.append({
            "target_position": target_position,
            "predicted_from": target_position - 1,
            "target_token": target,
            "nll": nll,
            "rank": rank_of(row, int(target)),
            "argmax_token": _argmax(row),
            "top1_matches_target": _argmax(row) == int(target),
        })
    require(positions, "No supervised target position exists")
    first = positions[0]
    continuation = positions[1:]
    return {
        "label_shift": LABEL_SHIFT,
        "prompt_tokens": prompt_tokens,
        "total_tokens": len(label_values),
        "label_positions": [first["target_position"],
                            positions[-1]["target_position"]],
        "predictor_positions": [first["predicted_from"],
                                positions[-1]["predicted_from"]],
        "supervised_label_tokens": supervised,
        "shifted_supervision_tokens": shifted,
        "loss_tokens": len(positions),
        "prompt_masked_tokens": prompt_masked,
        "prompt_fully_masked": prompt_masked == prompt_tokens,
        "mean_nll": sum(item["nll"] for item in positions) / len(positions),
        "first_token_nll": first["nll"],
        "first_token_rank": first["rank"],
        "continuation_tokens": len(continuation),
        "continuation_mean_nll": (sum(item["nll"] for item in continuation)
                                  / len(continuation) if continuation else None),
        "top1_matches": sum(item["top1_matches_target"] for item in positions),
        "positions": positions,
    }


def first_token_ceiling(rows: Sequence[Mapping[str, Any]],
                        *, expected_cases: int | None = None) -> dict[str, Any]:
    """Text ceiling: first-token gold rank/accuracy over image-free rows."""
    require(rows, "The text ceiling requires at least one row")
    analyses = []
    for row in rows:
        require("case_id" in row and "gold_token" in row and "logits" in row,
                "Text-ceiling rows require case_id, gold_token and logits")
        analysis = first_token_analysis(row["logits"], int(row["gold_token"]))
        analyses.append({"case_id": row["case_id"],
                         "gold_token": analysis["gold_token"],
                         "gold_rank": analysis["gold_rank"],
                         "argmax_token": analysis["argmax_token"]})
    return first_token_ceiling_summary(analyses, expected_cases=expected_cases)


def first_token_ceiling_summary(
        analyses: Sequence[Mapping[str, Any]],
        *, expected_cases: int | None = None) -> dict[str, Any]:
    """Summarize precomputed first-token gold ranks into the text ceiling."""
    require(analyses, "The text ceiling requires at least one analysis")
    records = []
    for row in analyses:
        for field in ("case_id", "gold_token", "gold_rank", "argmax_token"):
            require(field in row, f"Text-ceiling analysis lacks {field}")
        records.append({"case_id": row["case_id"],
                        "gold_token": int(row["gold_token"]),
                        "gold_rank": int(row["gold_rank"]),
                        "argmax_token": int(row["argmax_token"])})
    if expected_cases is not None:
        require(len(records) == int(expected_cases),
                f"Text ceiling expected {expected_cases} cases, got {len(records)}")
    ranks = sorted(record["gold_rank"] for record in records)
    correct = sum(rank == 1 for rank in ranks)
    return {
        "cases": len(records),
        "first_token_accuracy": correct / len(records),
        "first_token_correct": correct,
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": ranks[len(ranks) // 2] if len(ranks) % 2
        else (ranks[len(ranks) // 2 - 1] + ranks[len(ranks) // 2]) / 2,
        "min_rank": ranks[0],
        "max_rank": ranks[-1],
        "rows": records,
    }


# ---------------------------------------------------------------------------
# Pair layout and matched controls
# ---------------------------------------------------------------------------


def _kind_of(row: Mapping[str, Any]) -> str:
    kind = row.get("kind")
    require(kind in {"answer", "caption"},
            f"Unknown training kind: {kind!r}")
    return str(kind)


def _case_suffix(case_id: str) -> str:
    suffix = str(case_id).rsplit("-", 1)[-1]
    require(suffix in {"a", "b"}, f"Pair case id lacks an a/b suffix: {case_id}")
    return suffix


def pair_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group training rows into answer and caption pair ids.

    A complete pair carries both ``-a`` and ``-b`` siblings.  ``prepared04``
    schedules five complete answer pairs and seven caption pair ids, two of
    which (``-b`` only / ``-a`` only) have no sibling row in the training
    payload.  Incomplete groups are returned with ``complete = False`` rather
    than inventing a sibling image.
    """
    by_kind: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "answer": defaultdict(list), "caption": defaultdict(list)}
    seen: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        require(case_id not in seen, f"Duplicate training case id: {case_id}")
        seen.add(case_id)
        by_kind[_kind_of(row)][str(row["pair_id"])].append(row)
    groups: dict[str, list[dict[str, Any]]] = {"answer": [], "caption": []}
    for kind in ("answer", "caption"):
        for pair_id in sorted(by_kind[kind]):
            members = sorted(by_kind[kind][pair_id],
                             key=lambda item: str(item["case_id"]))
            require(1 <= len(members) <= 2,
                    f"Pair {pair_id} has more than two training rows")
            suffixes = [_case_suffix(str(item["case_id"])) for item in members]
            require(len(set(suffixes)) == len(suffixes),
                    f"Pair {pair_id} repeats a sibling suffix")
            groups[kind].append({
                "kind": kind,
                "pair_id": pair_id,
                "rows": members,
                "case_ids": [str(item["case_id"]) for item in members],
                "complete": len(members) == 2,
                "suffixes": suffixes,
            })
    return groups


def pair_layout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stable pair inventory used to pin the control plan.

    The training payload has nine answer pair ids: five complete opposite pairs
    plus four singleton answer rows.  Only the five complete pairs form the
    opposite-answer comparison; the singletons are reported, never upgraded.
    Seven caption pair ids appear, five of them complete.
    """
    groups = pair_groups(rows)
    complete_answers = [group for group in groups["answer"] if group["complete"]]
    singleton_answers = [group for group in groups["answer"]
                         if not group["complete"]]
    complete_captions = [group for group in groups["caption"]
                         if group["complete"]]
    require(len(complete_answers) == 5,
            "The diagnostic pins exactly five complete opposite answer pairs")
    require(len(groups["caption"]) == 7,
            "The diagnostic pins exactly seven caption pair ids")
    return {
        "answer_pairs": [_group_ref(group) for group in groups["answer"]],
        "caption_pairs": [_group_ref(group) for group in groups["caption"]],
        "answer_pair_count": len(groups["answer"]),
        "caption_pair_count": len(groups["caption"]),
        "complete_answer_pairs": len(complete_answers),
        "complete_caption_pairs": len(complete_captions),
        "singleton_answer_pair_count": len(singleton_answers),
        "singleton_answer_pairs": [group["pair_id"]
                                   for group in singleton_answers],
        "incomplete_caption_pairs": [group["pair_id"]
                                     for group in groups["caption"]
                                     if not group["complete"]],
        "opposite_answer_pairs": len(complete_answers),
        "caption_sibling_pairs": len(groups["caption"]),
    }


def _group_ref(group: Mapping[str, Any]) -> dict[str, Any]:
    return {"kind": group["kind"], "pair_id": group["pair_id"],
            "case_ids": list(group["case_ids"]),
            "complete": bool(group["complete"])}


def control_plan(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Deterministic matched controls for every training row.

    Selection uses only case ids, pair ids and sorted order.  No expected target
    text, answer token or label participates, so a control cannot be chosen to
    make a target easier.  ``shuffled`` is the first row of the next pair id in
    the sorted pair rotation; it therefore always belongs to a different pair.
    """
    groups = pair_groups(rows)
    ordered: list[dict[str, Any]] = [*groups["answer"], *groups["caption"]]
    ordered.sort(key=lambda group: (group["kind"], group["pair_id"]))
    require(len(ordered) >= 2, "Shuffling requires at least two pairs")
    require(len({group["pair_id"] for group in ordered}) == len(ordered),
            "Pair ids repeat across training kinds")
    index_of = {group["pair_id"]: index for index, group in enumerate(ordered)}
    row_by_case = {str(row["case_id"]): row for row in rows}
    plan: dict[str, dict[str, Any]] = {}
    for group in ordered:
        shuffled_group = ordered[(index_of[group["pair_id"]] + 1) % len(ordered)]
        shuffled_case = str(shuffled_group["rows"][0]["case_id"])
        for row in group["rows"]:
            case_id = str(row["case_id"])
            partner = [str(item["case_id"]) for item in group["rows"]
                       if str(item["case_id"]) != case_id]
            sibling = partner[0] if partner else None
            require(shuffled_case != case_id and shuffled_case != sibling,
                    "Shuffled control collides with another matched control")
            images = [row["image_sha256"]]
            if sibling is not None:
                images.append(row_by_case[sibling]["image_sha256"])
            images.append(row_by_case[shuffled_case]["image_sha256"])
            require(len(set(images)) == len(images),
                    "Matched controls must use distinct image identities")
            plan[case_id] = {
                "case_id": case_id,
                "example_id": str(row["example_id"]),
                "kind": _kind_of(row),
                "pair_id": str(row["pair_id"]),
                "correct_case_id": case_id,
                "sibling_case_id": sibling,
                "sibling_available": sibling is not None,
                "shuffled_case_id": shuffled_case,
                "shuffled_pair_id": str(shuffled_group["pair_id"]),
                "noimage": None,
                "controls": ["correct"] + (["sibling"] if sibling else [])
                            + ["shuffled", "noimage"],
            }
    require(set(plan) == {str(row["case_id"]) for row in rows},
            "Control plan does not cover every training row")
    return plan


def comparison_cells(group: Mapping[str, Any],
                     plan: Mapping[str, Mapping[str, Any]]
                     ) -> list[dict[str, Any]]:
    """Image/target comparison cells for one pair.

    Complete pairs give the 2x2 own/sibling image by own/opposite target plus
    the cross-pair shuffled image against both targets.  Incomplete pairs give
    their single image against the own target plus the shuffled image, with the
    missing cells reported as unavailable instead of fabricated.
    """
    members = group["rows"]
    own = members[0]
    own_case = str(own["case_id"])
    partner = members[1] if len(members) == 2 else None
    plan_row = plan[own_case]
    images = [("own", own_case)]
    if partner is not None:
        images.append(("changed_sibling", str(partner["case_id"])))
    images.append(("cross_pair_shuffled", str(plan_row["shuffled_case_id"])))
    cells = []
    for image_role, image_case in images:
        for target_role, target in (("own", own), ("opposite", partner)):
            if target_role == "opposite" and target is None:
                continue
            cells.append({
                "pair_id": group["pair_id"],
                "kind": group["kind"],
                "pair_complete": len(members) == 2,
                "image_role": image_role,
                "target_role": target_role,
                "image_case_id": image_case,
                "target_case_id": str(target["case_id"]),
                "own_case_id": own_case,
            })
    return cells


# ---------------------------------------------------------------------------
# Feature diagnostics and the feature ceiling
# ---------------------------------------------------------------------------


def variance_across(vectors: Sequence[Sequence[float]]) -> float:
    """Mean across dimensions of the variance across patches."""
    require(vectors, "Spatial variance requires at least one patch")
    width = len(vectors[0])
    require(width > 0 and all(len(vector) == width for vector in vectors),
            "Patch rows differ in width")
    total = 0.0
    count = len(vectors)
    for index in range(width):
        column = [float(vector[index]) for vector in vectors]
        centre = sum(column) / count
        total += sum((value - centre) ** 2 for value in column) / count
    return total / width


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    require(len(left) == len(right), "Cosine vectors differ in width")
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    require(left_norm > 0 and right_norm > 0, "Cosine requires nonzero vectors")
    dot = sum(a * b for a, b in zip(left_values, right_values))
    return 1.0 - dot / (left_norm * right_norm)


def pca_content_ratio(vectors: Sequence[Sequence[float]], *,
                      iterations: int = 48) -> dict[str, Any]:
    """Top-component share of total centred variance via deterministic power iteration.

    ``ratio = sigma_1**2 / sum_i sigma_i**2``.  The deterministic start vector
    is a fixed sine pattern; an all-ones start can be exactly orthogonal to the
    leading feature direction, so it is not used.  No optimizer, random seed or
    fitted model is involved.
    """
    require(len(vectors) >= 2, "PCA requires at least two vectors")
    require(type(iterations) is int and iterations >= 1,
            "PCA requires at least one power iteration")
    width = len(vectors[0])
    require(width > 1 and all(len(vector) == width for vector in vectors),
            "PCA vectors differ in width")
    rows = [[float(value) for value in vector] for vector in vectors]
    centre = [sum(row[index] for row in rows) / len(rows) for index in range(width)]
    centred = [[row[index] - centre[index] for index in range(width)] for row in rows]
    total = sum(value * value for row in centred for value in row)
    if total <= 0:
        return {"ratio": None, "top_singular_squared": 0.0, "total_singular_squared": 0.0,
                "iterations": 0, "converged": False}
    pattern = [math.sin((index + 1) * 1.618033988749895)
               for index in range(width)]
    pattern_norm = math.sqrt(sum(value * value for value in pattern))
    direction = [value / pattern_norm for value in pattern]
    top = 0.0
    for step in range(int(iterations)):
        projected = [sum(row[index] * direction[index] for index in range(width))
                     for row in centred]
        combined = [0.0] * width
        for row, weight in zip(centred, projected):
            for index in range(width):
                combined[index] += row[index] * weight
        norm = math.sqrt(sum(value * value for value in combined))
        if norm <= 0:
            return {"ratio": 0.0, "top_singular_squared": 0.0,
                    "total_singular_squared": total,
                    "iterations": step + 1, "converged": False}
        direction = [value / norm for value in combined]
        next_top = sum(value * value for value in projected)
        if step and abs(next_top - top) <= 1e-12 * max(1.0, total):
            top = next_top
            break
        top = next_top
    return {"ratio": min(1.0, max(0.0, top / total)),
            "top_singular_squared": top,
            "total_singular_squared": total,
            "iterations": step + 1,
            "converged": True}


def _cosine_or_none(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Cosine distance that stays defined-or-None for collapsed zero vectors."""
    if len(left) != len(right):
        raise DiagnosticError("Cosine vectors differ in width")
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm <= 0 or right_norm <= 0:
        return None
    dot = sum(a * b for a, b in zip(left_values, right_values))
    return 1.0 - dot / (left_norm * right_norm)


def _distance_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    items = sorted(float(value) for value in values)
    return {"count": len(items), "mean": sum(items) / len(items),
            "min": items[0], "max": items[-1]}


def feature_summary(stage: str,
                    image_stats: Mapping[str, Mapping[str, Any]],
                    encoder_means: Mapping[str, Sequence[float]],
                    plan: Mapping[str, Mapping[str, Any]],
                    text_embedding_rms: Sequence[float]
                    ) -> dict[str, Any]:
    """First-order projector/feature evidence for one stage.

    ``image_stats`` carries per-case projected statistics computed natively
    (patch RMS/max, mean-patch vector, spatial variance).  Distances, PCA and
    ratios are computed here; no classifier is fitted.
    """
    require(image_stats, "Feature diagnostics require image statistics")
    require(len(encoder_means) == len(image_stats),
            "Encoder and projected image inventories differ")
    cases = sorted(image_stats)
    for case_id in cases:
        stats = image_stats[case_id]
        for field in ("projected_rms", "projected_max", "mean_vector_rms",
                      "spatial_variance", "mean_patch"):
            require(field in stats, f"Image stats lack {field}")
    means = [image_stats[case_id]["mean_patch"] for case_id in cases]
    sibling_distances: list[float] = []
    cross_distances: list[float] = []
    skipped_distances = 0
    for case_id in cases:
        plan_row = plan.get(case_id)
        require(plan_row is not None, "Feature diagnostics require a control plan")
        own = image_stats[case_id]["mean_patch"]
        if plan_row.get("sibling_available"):
            sibling = str(plan_row["sibling_case_id"])
            distance = _cosine_or_none(
                own, image_stats[sibling]["mean_patch"])
            if distance is None:
                skipped_distances += 1
            else:
                sibling_distances.append(distance)
        shuffled = str(plan_row["shuffled_case_id"])
        if shuffled != case_id:
            distance = _cosine_or_none(
                own, image_stats[shuffled]["mean_patch"])
            if distance is None:
                skipped_distances += 1
            else:
                cross_distances.append(distance)
    rms_all = [image_stats[case_id]["projected_rms"] for case_id in cases]
    rms_mean = [image_stats[case_id]["mean_vector_rms"] for case_id in cases]
    ratios = []
    undefined_ratios = 0
    for projected, mean in zip(rms_all, rms_mean):
        if projected > 0:
            ratios.append(mean / projected)
        else:
            undefined_ratios += 1
    maximums = [image_stats[case_id]["projected_max"] for case_id in cases]
    spatial = [image_stats[case_id]["spatial_variance"] for case_id in cases]
    text = [float(value) for value in text_embedding_rms]
    require(text, "Text embedding RMS is required")
    text_mean = sum(text) / len(text)
    mean_rms_mean = sum(rms_mean) / len(rms_mean)
    return {
        "stage": stage,
        "images": len(cases),
        "projected_rms_mean": sum(rms_all) / len(rms_all),
        "projected_max_mean": sum(maximums) / len(maximums),
        "projected_mean_vector_rms_mean": mean_rms_mean,
        "rms_projected_mean_over_projected": (
            sum(ratios) / len(ratios) if ratios else None),
        "rms_projected_mean_over_projected_min": (min(ratios) if ratios
                                                  else None),
        "rms_projected_mean_over_projected_max": (max(ratios) if ratios
                                                  else None),
        "undefined_projected_ratios": undefined_ratios,
        "within_image_spatial_variance_mean": sum(spatial) / len(spatial),
        "pca_content": pca_content_ratio(means),
        "sibling_cosine_distance": _distance_summary(sibling_distances),
        "cross_pair_cosine_distance": _distance_summary(cross_distances),
        "skipped_zero_norm_distances": skipped_distances,
        "text_embedding_rms": {
            "count": len(text),
            "mean": text_mean,
            "min": min(text),
            "max": max(text),
        },
        "projected_mean_over_text_rms": (mean_rms_mean / text_mean
                                         if text_mean > 0 else None),
        "encoder_feature_matrix_sha256": feature_matrix_sha256(
            [encoder_means[case_id] for case_id in cases]),
        "projected_feature_matrix_sha256": feature_matrix_sha256(means),
        "scope": "first-order scale/separation evidence only; no learned-quality claim",
    }


def feature_matrix_sha256(vectors: Sequence[Sequence[float]]) -> str:
    payload = json.dumps([[round(float(value), 6) for value in vector]
                          for vector in vectors],
                         separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _normalise(vector: Sequence[float]) -> list[float] | None:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm <= 0:
        return None
    return [float(value) / norm for value in vector]


def feature_ceiling(pairs: Sequence[Mapping[str, Any]],
                    means: Mapping[str, Sequence[float]],
                    *, expected_pairs: int = 5) -> dict[str, Any]:
    """Leave-one-pair-out readout over frozen encoder mean-patch features.

    For each held-out pair, the direction is the sum of the other pairs'
    ``mean(b) - mean(a)`` differences in the frozen 768-d encoder space.  The
    held-out pair is correct when its sibling ordering follows that direction.
    Nothing is fitted or tuned; there is no candidate model.
    """
    require(len(pairs) == int(expected_pairs),
            f"The feature ceiling pins {expected_pairs} opposite pairs")
    pair_ids = [str(pair["pair_id"]) for pair in pairs]
    require(len(set(pair_ids)) == len(pair_ids), "Feature-ceiling pairs repeat")
    folds = []
    for held_out in pairs:
        training = [pair for pair in pairs if pair["pair_id"] != held_out["pair_id"]]
        require(training, "Leave-one-pair-out needs at least one training pair")
        direction = [0.0] * len(means[str(held_out["a_case_id"])])
        for pair in training:
            first = means[str(pair["a_case_id"])]
            second = means[str(pair["b_case_id"])]
            require(len(first) == len(direction) and len(second) == len(direction),
                    "Feature-ceiling vectors differ in width")
            delta = [float(b) - float(a) for a, b in zip(first, second)]
            direction = [value + change for value, change in zip(direction, delta)]
        unit = _normalise(direction)
        if unit is None:
            folds.append({"held_out_pair": held_out["pair_id"],
                          "training_pairs": [pair["pair_id"] for pair in training],
                          "projection_a": 0.0, "projection_b": 0.0,
                          "gap": 0.0, "correct": False,
                          "direction_norm": 0.0})
            continue
        projection_a = _dot(means[str(held_out["a_case_id"])], unit)
        projection_b = _dot(means[str(held_out["b_case_id"])], unit)
        folds.append({
            "held_out_pair": held_out["pair_id"],
            "training_pairs": [pair["pair_id"] for pair in training],
            "projection_a": projection_a,
            "projection_b": projection_b,
            "gap": projection_b - projection_a,
            "correct": projection_b > projection_a,
            "direction_norm": math.sqrt(sum(value * value for value in direction)),
        })
    correct = sum(fold["correct"] for fold in folds)
    gaps = [fold["gap"] for fold in folds]
    return {
        "pairs": len(folds),
        "accuracy": correct / len(folds),
        "correct": correct,
        "mean_gap": sum(gaps) / len(gaps),
        "folds": folds,
        "feature_width": len(means[str(pairs[0]["a_case_id"])]),
        "scope": "deterministic leave-one-pair-out readout; no fitted model, no tuning",
    }


# ---------------------------------------------------------------------------
# Fit contract: per-row report and coverage
# ---------------------------------------------------------------------------


def fit_contract_row_report(*, example_id: str, case_id: str, kind: str,
                            prompt_tokens: int,
                            input_ids: Sequence[int],
                            labels: Sequence[int],
                            eos_token: int,
                            stop_tokens: Sequence[int],
                            pad_token: int | None = None,
                            question: str | None = None,
                            caption_template: str | None = None,
                            loss: str = LOSS_NAME,
                            max_sequence_tokens: int | None = None
                            ) -> dict[str, Any]:
    """Check one training row's prompt/label/EOS contract and return its record.

    The returned fields name the shifted convention without abbreviation:
    ``supervised_label_tokens`` counts ``labels != -100``;
    ``shifted_supervision_tokens`` counts ``labels[1:] != -100``; both must equal
    ``response_tokens``.  The first supervised target lives at
    ``label_positions[0]`` and is predicted from ``predictor_positions[0]``.
    """
    inputs = [int(value) for value in input_ids]
    targets = [int(value) for value in labels]
    prompt = int(prompt_tokens)
    require(prompt >= 1, "Fit rows require a nonempty prompt")
    require(len(inputs) == len(targets), "Fit inputs and labels differ in length")
    require(prompt <= len(inputs), "Fit prompt exceeds the sequence")
    require(all(value == LABEL_IGNORE_INDEX for value in targets[:prompt]),
            "Prompt labels are not fully masked")
    response_targets = targets[prompt:]
    require(response_targets and all(value != LABEL_IGNORE_INDEX
                                     for value in response_targets),
            "Response labels are masked or empty")
    response_ids = inputs[prompt:]
    require(len(response_ids) == len(response_targets),
            "Response inputs and labels differ in length")
    eos = int(eos_token)
    stops = _int_tuple(list(stop_tokens), "stops")
    require(eos in set(stops), "The target EOS token is outside the declared stop set")
    require(response_ids[-1] == eos, "Input response does not end with the target EOS")
    require(response_targets[-1] == eos, "Label response does not end with the target EOS")
    pad_positions = ([index for index, token in enumerate(response_ids[:-1])
                     if pad_token is not None and token == int(pad_token)])
    require(not pad_positions, "A pad token appears before the final EOS")
    if max_sequence_tokens is not None:
        require(len(inputs) <= int(max_sequence_tokens),
                "Fit row exceeds the declared sequence budget")
    template_checked = False
    if str(kind) == "caption":
        require(caption_template is not None,
                "Caption rows require the declared caption template")
        require(question == caption_template,
                "Caption prompt differs from the fit contract")
        template_checked = True
    require(isinstance(loss, str) and loss == LOSS_NAME,
            "Fit loss differs from the declared mean response NLL")
    label_positions = [prompt, len(targets) - 1]
    predictor_positions = [prompt - 1, len(targets) - 2]
    return {
        "example_id": str(example_id),
        "case_id": str(case_id),
        "kind": str(kind),
        "prompt_tokens": prompt,
        "response_tokens": len(response_ids),
        "total_tokens": len(inputs),
        "supervised_label_tokens": sum(value != LABEL_IGNORE_INDEX
                                       for value in targets),
        "shifted_supervision_tokens": sum(value != LABEL_IGNORE_INDEX
                                          for value in targets[LABEL_SHIFT:]),
        "loss_tokens": len(response_targets),
        "label_shift": LABEL_SHIFT,
        "label_positions": label_positions,
        "predictor_positions": predictor_positions,
        "prompt_masked_tokens": prompt,
        "prompt_masked": True,
        "response_token_ids": response_ids,
        "response_eos_token": eos,
        "response_eos": True,
        "eos_token_id": eos,
        "stop_tokens": list(stops),
        "eos_in_stop_set": True,
        "pad_token": None if pad_token is None else int(pad_token),
        "pad_positions_before_eos": pad_positions,
        "pad_free_response": not pad_positions,
        "template_checked": template_checked,
        "response_only_labels": True,
        "loss": LOSS_NAME,
        "contract_valid": True,
    }


def fit_coverage(rows: Sequence[Mapping[str, Any]],
                 expected_example_ids: Sequence[str]) -> dict[str, Any]:
    """Audit that every expected fit row was checked at least once.

    ``rows`` are the retained per-row reports from every update (they may
    repeat).  Coverage is by unique ``example_id`` and by kind, so a caption row
    cannot satisfy an answer row's obligation.
    """
    expected = [str(value) for value in expected_example_ids]
    require(len(set(expected)) == len(expected),
            "Expected fit example ids repeat")
    seen: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    invalid: list[str] = []
    for row in rows:
        example_id = str(row.get("example_id"))
        require(example_id and example_id != "None",
                "A retained fit row lacks example_id")
        if example_id in seen:
            duplicates.append(example_id)
        seen[example_id] = row
        if row.get("contract_valid") is not True:
            invalid.append(example_id)
    checked = sorted(seen)
    missing = sorted(set(expected) - set(checked))
    unexpected = sorted(set(checked) - set(expected))
    kinds = Counter(str(row.get("kind")) for row in seen.values())
    rows_checked = len(checked)
    return {
        "expected_rows": len(expected),
        "checked_rows": rows_checked,
        "retained_row_reports": len(rows),
        "missing": missing,
        "unexpected": unexpected,
        "invalid": sorted(set(invalid)),
        "repeated_example_ids": sorted(set(duplicates)),
        "per_kind": dict(sorted(kinds.items())),
        "all_rows_checked": not missing and not unexpected and not invalid,
        "rows": [{"example_id": example_id,
                  "case_id": seen[example_id].get("case_id"),
                  "kind": seen[example_id].get("kind"),
                  "prompt_tokens": seen[example_id].get("prompt_tokens"),
                  "response_tokens": seen[example_id].get("response_tokens"),
                  "supervised_label_tokens": seen[example_id].get(
                      "supervised_label_tokens"),
                  "loss_tokens": seen[example_id].get("loss_tokens"),
                  "response_eos_token": seen[example_id].get("response_eos_token")}
                 for example_id in checked],
    }


def require_fit_coverage(report: Mapping[str, Any],
                         *, expected_kinds: Mapping[str, int] | None = None) -> None:
    """Raise unless every expected row was checked with the pinned kind counts."""
    require(report.get("all_rows_checked") is True,
            "Fit contract coverage is incomplete: "
            f"missing={report.get('missing')} "
            f"unexpected={report.get('unexpected')} "
            f"invalid={report.get('invalid')}")
    if expected_kinds is not None:
        for kind, count in expected_kinds.items():
            require(report["per_kind"].get(kind, 0) == int(count),
                    f"Fit contract coverage for {kind} is "
                    f"{report['per_kind'].get(kind, 0)}, expected {count}")


# ---------------------------------------------------------------------------
# Stage summaries and stop conditions
# ---------------------------------------------------------------------------


def control_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """First-order per-control counts; no gate, verdict or promotion."""
    require(rows, "A control summary requires rows")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["control"])].append(row)
    summary: dict[str, Any] = {}
    for control in sorted(grouped):
        items = grouped[control]
        first_ranks = [float(item["first_token"]["gold_rank"])
                       for item in items if item.get("first_token")]
        nlls = [float(item["positions"]["mean_nll"]) for item in items
                if item.get("positions")]
        ended = sum(bool(item["generation"]["ended_by_eos"]) for item in items
                    if item.get("generation"))
        summary[control] = {
            "rows": len(items),
            "first_token_gold_accuracy": (
                sum(rank == 1 for rank in first_ranks) / len(first_ranks)
                if first_ranks else None),
            "first_token_mean_gold_rank": (
                sum(first_ranks) / len(first_ranks) if first_ranks else None),
            "response_mean_nll": (sum(nlls) / len(nlls) if nlls else None),
            "generation_ended_by_stop": ended,
            "generation_rows": sum(1 for item in items if item.get("generation")),
        }
    return summary


def stop_conditions_report(*, wall_seconds: float, wall_limit_seconds: float,
                           artifact_bytes: int, artifact_limit_bytes: int,
                           updates: int, optimizer_steps: int,
                           gradients_computed: int, checkpoint_writes: int,
                           extension_writes: int, model_loads: int,
                           extension_loads: int, downloads: int,
                           cpu_fallback: bool, sealed_evaluation_access: int,
                           guard_stop_reason: str | None) -> dict[str, Any]:
    """Predeclared resource/access accounting for one diagnostic run."""
    report = {
        "wall_seconds": float(wall_seconds),
        "wall_limit_seconds": float(wall_limit_seconds),
        "wall_seconds_exceeded": float(wall_seconds) > float(wall_limit_seconds),
        "artifact_bytes": int(artifact_bytes),
        "artifact_limit_bytes": int(artifact_limit_bytes),
        "artifact_bytes_exceeded": int(artifact_bytes) > int(artifact_limit_bytes),
        "updates_performed": int(updates),
        "optimizer_steps": int(optimizer_steps),
        "gradients_computed": int(gradients_computed),
        "checkpoint_writes": int(checkpoint_writes),
        "extension_writes": int(extension_writes),
        "model_loads": int(model_loads),
        "extension_loads": int(extension_loads),
        "downloads": int(downloads),
        "cpu_fallback": bool(cpu_fallback),
        "sealed_evaluation_access": int(sealed_evaluation_access),
        "guard_stop_reason": guard_stop_reason,
        "no_training_updates": int(updates) == 0,
        "no_optimizer_steps": int(optimizer_steps) == 0,
        "no_gradients": int(gradients_computed) == 0,
        "no_checkpoint_writes": int(checkpoint_writes) == 0,
        "no_extension_writes": int(extension_writes) == 0,
        "single_model_load": int(model_loads) == 1,
        "no_downloads": int(downloads) == 0,
        "no_cpu_fallback": bool(cpu_fallback) is False,
        "no_sealed_evaluation_access": int(sealed_evaluation_access) == 0,
        "within_declared_limits": (
            float(wall_seconds) <= float(wall_limit_seconds)
            and int(artifact_bytes) <= int(artifact_limit_bytes)),
    }
    report["violations"] = [name for name in
                            ("wall_seconds_exceeded", "artifact_bytes_exceeded")
                            if report[name]]
    return report


def require_stop_conditions(report: Mapping[str, Any]) -> None:
    require(not report.get("violations"), "Declared stop condition tripped: "
            f"{report.get('violations')}")
    for flag in ("no_training_updates", "no_optimizer_steps", "no_gradients",
                 "no_checkpoint_writes", "no_extension_writes",
                 "single_model_load", "no_downloads", "no_cpu_fallback",
                 "no_sealed_evaluation_access"):
        require(report.get(flag) is True, f"Declared policy violated: {flag}")


def claims_block() -> dict[str, Any]:
    """The diagnostic never promotes learned quality or semantic competence."""
    return {
        "first_order_evidence_only": True,
        "promotion": False,
        "perception_quality": False,
        "semantic_quality": False,
        "overall_baseline_eligible": False,
        "learned_quality": False,
        "scope": ("training-only vision-collapse evidence over prepared04 "
                  "training rows and the pinned train04 extension; no "
                  "evaluation split is read and no update is performed"),
    }
