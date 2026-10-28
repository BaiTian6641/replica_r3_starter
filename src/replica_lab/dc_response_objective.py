"""DC-P1 v2 response objective: full-response CE plus answer-span CE.

The frozen objective is

    loss = CE(response_mask) + answer_span_ce_weight * CE(answer_span_mask)

where ``response_mask`` covers every supervised response token (the generated
answer plus the terminal EOS) and ``answer_span_mask`` covers the answer content
expressed by ``answer_span`` plus the terminal EOS token.

Two properties are load bearing and are asserted, not assumed:

* with ``answer_span_ce_weight == 0.0`` the callable is exactly the matched
  control recipe: the identical masks and the identical full-response term with
  the span term multiplied by zero.  No aliasing, no rescaling, no second code
  path.
* when the two masks coincide the added term merely rescales the same token CE
  by ``(1 + weight)``; that is a *global CE rescale*, not a span-specific
  learning signal, and :func:`audit_objective_signal` says so.  The v2 protocol
  requires the narrower claim in that case.

Everything here is deterministic.  The reference path uses only the standard
library so that mask construction, the reference loss and the gradient audit can
be exercised on CPU without torch.  :func:`answer_span_objective` is the real
autograd entry point and imports torch lazily; the native runner calls it, the
CPU tests call :func:`reference_objective`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

OBJECTIVE_FORMAT = "replica.dc_response_objective.v2"
MASK_FORMAT = "replica.dc_response_masks.v2"
AUDIT_FORMAT = "replica.dc_response_objective_audit.v2"
TERMINAL_EOS_ID = 1
LABEL_SHIFT = 1

CLAIM_SPAN_SPECIFIC = "span_specific_weighting"
CLAIM_GLOBAL_RESCALE = "global_ce_rescale"


class ObjectiveError(ValueError):
    """One fail-closed objective/mask contract violation."""

    def __init__(self, check: str, detail: str):
        super().__init__(f"{check}: {detail}")
        self.check = check
        self.detail = detail


def _require(condition: bool, check: str, detail: str) -> None:
    if not condition:
        raise ObjectiveError(check, detail)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class ResponseMasks:
    """Token masks aligned with the supervised response positions."""

    response_mask: tuple[bool, ...]
    answer_span_mask: tuple[bool, ...]
    answer_span: tuple[int, int]
    eos_id: int

    @property
    def response_tokens(self) -> int:
        return sum(1 for flag in self.response_mask if flag)

    @property
    def answer_span_tokens(self) -> int:
        return sum(1 for flag in self.answer_span_mask if flag)

    @property
    def terminal_eos_index(self) -> int:
        return len(self.response_mask) - 1

    @property
    def masks_coincide(self) -> bool:
        return self.response_mask == self.answer_span_mask

    @property
    def answer_span_superset(self) -> bool:
        return all(not answer or response
                   for answer, response in zip(self.answer_span_mask,
                                               self.response_mask))

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": MASK_FORMAT,
            "response_mask": list(self.response_mask),
            "answer_span_mask": list(self.answer_span_mask),
            "answer_span": list(self.answer_span),
            "eos_id": self.eos_id,
            "response_tokens": self.response_tokens,
            "answer_span_tokens": self.answer_span_tokens,
            "terminal_eos_index": self.terminal_eos_index,
            "masks_coincide": self.masks_coincide,
        }


def build_response_masks(
        *, input_ids: Sequence[int], response_ids: Sequence[int],
        answer_span: tuple[int, int],
        eos_id: int = TERMINAL_EOS_ID) -> ResponseMasks:
    """Build the full-response and answer-span+EOS masks.

    ``input_ids`` is the full teaching sequence (prompt plus response) and
    ``response_ids`` its supervised suffix.  ``answer_span`` is an inclusive
    ``(start, end)`` offset range into ``response_ids`` covering the answer
    content only; the terminal EOS is added to the span explicitly, so the span
    term always trains the stop decision as well.
    """
    _require(bool(input_ids), "masks", "input_ids must not be empty")
    _require(bool(response_ids), "masks", "response_ids must not be empty")
    _require(all(_is_int(item) and item >= 0 for item in input_ids),
             "masks", "input_ids must be nonnegative integers")
    _require(all(_is_int(item) and item >= 0 for item in response_ids),
             "masks", "response_ids must be nonnegative integers")
    _require(response_ids[-1] == eos_id, "masks",
             f"response must end with the terminal EOS id {eos_id}")
    _require(list(input_ids[-len(response_ids):]) == list(response_ids),
             "masks", "response_ids must be the suffix of input_ids")
    _require(isinstance(answer_span, tuple) and len(answer_span) == 2
             and all(_is_int(item) for item in answer_span),
             "masks", "answer_span must be a two-integer tuple")
    start, end = answer_span
    _require(0 <= start <= end < len(response_ids), "masks",
             f"answer_span {(start, end)} is outside the response")
    terminal = len(response_ids) - 1
    _require(end <= terminal, "masks",
             "answer_span may not run past the response")
    response_mask = tuple(True for _ in response_ids)
    span = [False] * len(response_ids)
    for index in range(start, end + 1):
        span[index] = True
    span[terminal] = True
    masks = ResponseMasks(
        response_mask=response_mask, answer_span_mask=tuple(span),
        answer_span=(start, end), eos_id=eos_id)
    _require(masks.answer_span_superset, "masks",
             "answer-span mask must be a subset of the response mask")
    return masks


def shift_labels(
        *, prompt_ids: Sequence[int], response_ids: Sequence[int],
        answer_span: tuple[int, int],
        eos_id: int = TERMINAL_EOS_ID) -> tuple[list[int], ResponseMasks]:
    """Return the evaluation-free shifted label row and its masks.

    The caller feeds ``prompt_ids + response_ids[:-1]`` to the model; position
    ``i`` there predicts ``response_ids[i]``.  Labels are response-only with
    ``label_shift == 1`` and no prompt token is ever supervised.
    """
    _require(bool(prompt_ids), "labels", "prompt_ids must not be empty")
    input_ids = list(prompt_ids) + list(response_ids)
    masks = build_response_masks(
        input_ids=input_ids, response_ids=response_ids,
        answer_span=answer_span, eos_id=eos_id)
    response_start = len(prompt_ids) - LABEL_SHIFT
    _require(response_start >= 0
             and response_start + len(masks.response_mask)
             == len(input_ids) - LABEL_SHIFT, "labels",
             "response logits must start at prompt_length-1 and cover every response token")
    return list(response_ids), masks


# ---------------------------------------------------------------------------
# Torch entry point (lazy import; this module imports on CPU-only hosts)


def answer_span_objective(
        logits: Any, label_ids: Any, masks: ResponseMasks, *,
        answer_span_ce_weight: float) -> dict[str, Any]:
    """Native objective.  ``logits`` has shape ``[response_tokens, vocab]``."""
    import torch  # noqa: PLC0415 - deliberate lazy native import
    import torch.nn.functional as functional  # noqa: PLC0415

    _require(isinstance(answer_span_ce_weight, (int, float))
             and not isinstance(answer_span_ce_weight, bool),
             "objective", "answer_span_ce_weight must be numeric")
    weight = float(answer_span_ce_weight)
    _require(0.0 <= weight <= 1.0, "objective",
             "answer_span_ce_weight must lie in [0, 1]")
    _require(isinstance(logits, torch.Tensor) and logits.dim() == 2,
             "objective", "logits must be a rank-2 tensor")
    _require(isinstance(label_ids, torch.Tensor)
             and label_ids.dim() == 1, "objective",
             "label_ids must be a rank-1 tensor")
    _require(int(logits.shape[0]) == len(masks.response_mask), "objective",
             "logits row count differs from the mask length")
    _require(int(label_ids.shape[0]) == len(masks.response_mask), "objective",
             "label count differs from the mask length")
    _require(str(logits.device) == str(label_ids.device), "objective",
             "logits and labels must share a device")

    response_index = torch.tensor(masks.response_mask, dtype=torch.bool,
                                  device=logits.device)
    span_index = torch.tensor(masks.answer_span_mask, dtype=torch.bool,
                              device=logits.device)
    per_token = functional.cross_entropy(
        logits, label_ids, reduction="none").float()
    full_response_ce = per_token[response_index].mean()
    answer_span_ce = per_token[span_index].mean()
    span_term = weight * answer_span_ce
    loss = full_response_ce + span_term
    return {
        "format": OBJECTIVE_FORMAT,
        "loss": loss,
        "full_response_ce": float(full_response_ce.detach()),
        "answer_span_ce": float(answer_span_ce.detach()),
        "span_term": float(span_term.detach()),
        "answer_span_ce_weight": weight,
        "response_tokens": masks.response_tokens,
        "answer_span_tokens": masks.answer_span_tokens,
        "masks_coincide": masks.masks_coincide,
        "applied_term": (CLAIM_GLOBAL_RESCALE if masks.masks_coincide
                         else CLAIM_SPAN_SPECIFIC),
        "masks": masks,
    }


# ---------------------------------------------------------------------------
# Standard-library reference path (CPU tests, offline audits)


def _softmax(row: Sequence[float]) -> list[float]:
    _require(bool(row), "objective", "logit row must not be empty")
    peak = max(row)
    exps = [math.exp(value - peak) for value in row]
    total = sum(exps)
    _require(total > 0.0, "objective", "softmax normalizer is not positive")
    return [value / total for value in exps]


def reference_objective(
        logits: Sequence[Sequence[float]], label_ids: Sequence[int],
        masks: ResponseMasks, *,
        answer_span_ce_weight: float) -> dict[str, float]:
    """Pure-Python copy of the native loss, used for offline audits."""
    _require(len(logits) == len(masks.response_mask), "objective",
             "logit rows differ from the mask length")
    _require(len(label_ids) == len(masks.response_mask), "objective",
             "label count differs from the mask length")
    token_losses: list[float] = []
    for row, label in zip(logits, label_ids):
        probabilities = _softmax([float(value) for value in row])
        _require(_is_int(label) and 0 <= label < len(probabilities),
                 "objective", f"label {label!r} is outside the vocabulary")
        token_losses.append(-math.log(max(probabilities[label], 1e-300)))
    response = [loss for loss, keep in zip(token_losses, masks.response_mask)
                if keep]
    span = [loss for loss, keep in zip(token_losses, masks.answer_span_mask)
            if keep]
    full_response_ce = sum(response) / len(response)
    answer_span_ce = sum(span) / len(span)
    return {
        "full_response_ce": full_response_ce,
        "answer_span_ce": answer_span_ce,
        "span_term": float(answer_span_ce_weight) * answer_span_ce,
        "loss": full_response_ce + float(answer_span_ce_weight) * answer_span_ce,
    }


def token_weights(masks: ResponseMasks, *,
                  answer_span_ce_weight: float) -> tuple[float, ...]:
    """Per-position weight of the position's CE in the total loss."""
    _require(0.0 <= float(answer_span_ce_weight) <= 1.0, "objective",
             "answer_span_ce_weight must lie in [0, 1]")
    response_count = masks.response_tokens
    span_count = masks.answer_span_tokens
    _require(response_count > 0 and span_count > 0, "objective",
             "both masks must supervise at least one token")
    base = 1.0 / response_count
    span = float(answer_span_ce_weight) / span_count
    return tuple(base + (span if keep else 0.0)
                 for keep in masks.answer_span_mask)


def reference_gradients(
        logits: Sequence[Sequence[float]], label_ids: Sequence[int],
        masks: ResponseMasks, *,
        answer_span_ce_weight: float) -> list[list[float]]:
    """Exact gradient of the objective with respect to the logits."""
    weights = token_weights(
        masks, answer_span_ce_weight=answer_span_ce_weight)
    rows: list[list[float]] = []
    for row, label, weight in zip(logits, label_ids, weights):
        index = int(label)
        probabilities = _softmax([float(value) for value in row])
        _require(index == label and 0 <= index < len(probabilities),
                 "objective", f"label {label!r} is outside the vocabulary")
        gradients = [weight * value for value in probabilities]
        gradients[index] -= weight
        rows.append(gradients)
    return rows


def _cosine(left: Sequence[Sequence[float]],
            right: Sequence[Sequence[float]]) -> float:
    flat_left = [value for row in left for value in row]
    flat_right = [value for row in right for value in row]
    _require(len(flat_left) == len(flat_right), "audit",
             "gradient shapes differ")
    dot = sum(a * b for a, b in zip(flat_left, flat_right))
    norm_left = math.sqrt(sum(value * value for value in flat_left))
    norm_right = math.sqrt(sum(value * value for value in flat_right))
    _require(norm_left > 0.0 and norm_right > 0.0, "audit",
             "gradient norm is zero; the signal is not comparable")
    return dot / (norm_left * norm_right)


def audit_objective_signal(
        *, masks: ResponseMasks, answer_span_ce_weight: float,
        logits: Sequence[Sequence[float]] | None = None,
        label_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """Declare whether the weighted span term adds a distinct signal.

    Two outcomes are possible and only two:

    * ``masks_coincide`` -> every supervised position carries the identical
      weight ``(1 + weight) / tokens``; the treatment objective is a global CE
      rescale of the matched objective and the claim must be narrowed.
    * otherwise -> the per-position weights differ, so the added term changes
      the learning signal; when logits and labels are supplied the exact
      gradient cosine is reported as well.
    """
    weight = float(answer_span_ce_weight)
    weights = token_weights(masks, answer_span_ce_weight=weight)
    spread = max(weights) / min(weights)
    coincide = masks.masks_coincide
    report: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "objective_format": OBJECTIVE_FORMAT,
        "answer_span_ce_weight": weight,
        "response_tokens": masks.response_tokens,
        "answer_span_tokens": masks.answer_span_tokens,
        "masks_coincide": coincide,
        "per_token_weight_spread": spread,
        "per_token_weight_constant": spread == 1.0,
        "rescaled_only": coincide,
        "added_term_changes_learning_signal": not coincide,
        "claim": CLAIM_GLOBAL_RESCALE if coincide else CLAIM_SPAN_SPECIFIC,
        "constant_rescale_factor": (1.0 + weight) if coincide else None,
        "gradient_cosine": None,
    }
    if logits is not None and label_ids is not None:
        control = reference_gradients(
            logits, label_ids, masks, answer_span_ce_weight=0.0)
        treatment = reference_gradients(
            logits, label_ids, masks, answer_span_ce_weight=weight)
        report["gradient_cosine"] = _cosine(control, treatment)
        if coincide:
            _require(abs(report["gradient_cosine"] - 1.0) < 1e-12, "audit",
                     "coincident masks must produce a pure rescale")
        else:
            _require(report["gradient_cosine"] < 1.0 - 1e-12, "audit",
                     "distinct masks must not produce a pure rescale")
    return report


def audit_teaching_rows(rows: Sequence[Mapping[str, Any]], *,
                        answer_span_ce_weight: float) -> dict[str, Any]:
    """Audit every declared teaching row; the fixture names the mask spans."""
    audits = []
    for row in rows:
        masks = build_response_masks(
            input_ids=row["input_ids"], response_ids=row["response_ids"],
            answer_span=tuple(row["answer_span"]),
            eos_id=int(row.get("eos_id", TERMINAL_EOS_ID)))
        audits.append(audit_objective_signal(
            masks=masks, answer_span_ce_weight=answer_span_ce_weight))
    coincide = sum(1 for audit in audits if audit["masks_coincide"])
    return {
        "format": AUDIT_FORMAT + ".collection",
        "answer_span_ce_weight": float(answer_span_ce_weight),
        "rows": len(audits),
        "coincident_rows": coincide,
        "span_specific_rows": len(audits) - coincide,
        "rescaled_only": coincide == len(audits),
        "claim": (CLAIM_GLOBAL_RESCALE if coincide == len(audits)
                  else CLAIM_SPAN_SPECIFIC),
        "per_row": audits,
    }


# The names the v2 protocol config binds.  A protocol that renames these
# without updating the config fails validation on purpose.
PUBLIC_API = {
    "module": "replica_lab.dc_response_objective",
    "masks": "build_response_masks",
    "labels": "shift_labels",
    "objective": "answer_span_objective",
    "reference": "reference_objective",
    "audit": "audit_objective_signal",
}
