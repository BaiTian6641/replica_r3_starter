"""Authority-specific invalidation for the v15 successor evaluation path."""
from __future__ import annotations

from dataclasses import dataclass

from .contextual_memory import AuthorityRejection
from .contextual_memory_v5 import AuthorityReadFailure
from copy import deepcopy

import torch

from .intrinsic_memory_acceptance_v13 import (
    InvalidationContractError, assess_response,
    chronological_correction as direct_chronological_correction)
from .intrinsic_memory_coverage_v12 import (
    COMPOUND_FAMILIES, _compound_answer, _render_compound)


@dataclass(frozen=True)
class AuthorityInvalidation:
    kind: str
    authority_reason: str
    error_type: str
    message: str
    cause_type: str | None


def expect_invalidation(kind: str, operation, *, reason: str | None = None) -> AuthorityInvalidation:
    """Require a matching authority rejection; all other failures are operational.

    Generic ContextualMemoryError/ContextualReadFailure values do not prove
    revocation. OSError, BudgetError, malformed payloads and implementation
    failures propagate unchanged and can never satisfy this assertion.
    """
    expected = reason or kind
    try:
        operation()
    except (AuthorityRejection, AuthorityReadFailure) as error:
        if error.authority_reason != expected:
            raise InvalidationContractError(
                f"authority reason {error.authority_reason!r} does not match {expected!r}") from error
        cause = error.__cause__
        return AuthorityInvalidation(kind, error.authority_reason, type(error).__name__,
                                     str(error), type(cause).__name__ if cause else None)
    raise InvalidationContractError(f"{kind} did not invalidate dependent authority")



def sequence_log_likelihood(logits_sequence, token_ids, allowed_stop_token_ids):
    """Sum per-step answer-token log probabilities, excluding the final stop."""
    if not logits_sequence or not token_ids:
        raise ValueError("complete answer-sequence evidence required")
    if len(token_ids) > len(logits_sequence):
        raise ValueError("answer tokens exceed available sequence logits")
    stops = set(allowed_stop_token_ids)
    if any(token in stops for token in token_ids[:-1]):
        raise ValueError("answer contains an early stop token")
    total = 0.0
    for index, token in enumerate(token_ids):
        if index == len(token_ids) - 1 and token in stops:
            continue
        step = logits_sequence[index].detach().float()
        total += float(torch.log_softmax(step, dim=-1)[token])
    return total


def complete_response_assessment(row, response, *, control, sequence_logits=None,
                                 counterfactual_logits=None):
    """Combine content/completion with per-sequence likelihood when available."""
    assessment = assess_response(row["answer"], response["raw_answer"], control=control,
                                 terminated=response["terminated"], truncated=response["truncated"])
    result = assessment.as_dict()
    expected_ll = (sequence_log_likelihood(
        sequence_logits, response["token_ids"], row["allowed_stop_token_ids"])
        if sequence_logits is not None else None)
    other_ll = (sequence_log_likelihood(
        counterfactual_logits, response["token_ids"], row["allowed_stop_token_ids"])
        if counterfactual_logits is not None else None)
    result.update(answer_log_likelihood=expected_ll,
                  counterfactual_log_likelihood=other_ll,
                  answer_minus_counterfactual_log_likelihood=(
                      expected_ll - other_ll if expected_ll is not None and other_ll is not None
                      else None),
                  sequence_evidence_complete=sequence_logits is not None
                  and counterfactual_logits is not None)
    return result


def chronological_correction(row, new_color):
    """Apply the declared direct or compound-family correction oracle."""
    if not isinstance(new_color, str) or not new_color.isalpha():
        raise ValueError("invalid correction color")
    family = row.get("rule_family")
    if family not in COMPOUND_FAMILIES:
        return direct_chronological_correction(row, new_color)
    original = deepcopy(row)
    facts = deepcopy(row.get("facts"))
    query = row.get("query_entity")
    if not isinstance(facts, dict) or not isinstance(query, str):
        raise ValueError("compound structured facts required")
    if _compound_answer(facts, family, query) != row.get("answer"):
        raise ValueError("original compound answer does not follow from its oracle")
    if new_color == row["answer"]:
        raise ValueError("correction must change the intended answer")
    changed = deepcopy(facts)
    changed["answer"] = new_color
    changed["unrecorded_target"] = False
    if family == COMPOUND_FAMILIES[1]:
        # Equality: the anchor shares the target's color by declaration.
        changed["anchor_colour"] = new_color
    allowed = {"answer", "anchor_colour", "unrecorded_target"}
    if any(changed.get(key) != facts.get(key) for key in set(facts) - allowed):
        raise ValueError("correction changed an unrelated compound fact")
    text = _render_compound(changed, family)
    if _compound_answer(changed, family, query) != new_color:
        raise ValueError("corrected answer does not follow from the compound oracle")
    return {**original, "facts": changed, "text": text, "answer": new_color,
            "chronology": 2,
            "correction": {"intended_entity": query, "rule_family": family,
                           "supersedes_revision": 1, "earlier_facts_preserved": True,
                           "unrelated_facts_unchanged": True,
                           "verification": "declared_compound_oracle_and_renderer_replay"}}