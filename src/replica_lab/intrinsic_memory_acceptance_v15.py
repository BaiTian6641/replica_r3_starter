"""Authority-specific invalidation for the v15 successor evaluation path."""
from __future__ import annotations

from dataclasses import dataclass

from .contextual_memory import AuthorityRejection
from .contextual_memory_v5 import AuthorityReadFailure
from copy import deepcopy
import re

import torch

from .intrinsic_memory_acceptance_v13 import (
    InvalidationContractError, assess_response,
    chronological_correction as direct_chronological_correction)
from .contextual_completion_objective_v1 import validate_complete_response
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


def sequence_likelihood_parts(logits, encoding, *, spec):
    """Differentiable answer/EOS log-likelihood under one teacher-forced prefix."""
    counts = validate_complete_response(encoding, spec=spec)
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    labels = encoding.labels[0]
    terminal = encoding.metadata["terminal_prediction_position"]
    begin = encoding.metadata["prompt_tokens"] - 1
    answer_positions = torch.arange(begin, terminal, device=labels.device)
    answer = log_probabilities[0, answer_positions, labels[answer_positions]].sum()
    eos = log_probabilities[0, terminal, labels[terminal]]
    return {
        "answer_log_likelihood": answer,
        "eos_log_likelihood": eos,
        "total_log_likelihood": answer + eos,
        "answer_tokens": counts["word_tokens"],
        "supervised_tokens": counts["supervised_tokens"],
    }


def sequence_margin(expected, alternative):
    """Margin between two independently encoded candidate responses."""
    return {key: expected[key] - alternative[key]
            for key in ("answer_log_likelihood", "eos_log_likelihood", "total_log_likelihood")}


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


def _compound_history(row):
    facts = row.get("facts")
    history = row.get("revision_history")
    if history is None:
        history = ([] if row["answer"] == "unknown" else [{
            "revision": 1, "answer": facts["answer"],
            "anchor_colour": facts["anchor_colour"],
        }])
    if (not isinstance(history, list)
            or any(type(item) is not dict or type(item.get("revision")) is not int
                   or not isinstance(item.get("answer"), str)
                   or not isinstance(item.get("anchor_colour"), str)
                   for item in history)):
        raise ValueError("invalid compound revision history")
    revisions = [item["revision"] for item in history]
    if revisions != sorted(set(revisions)) or revisions != list(range(1, len(revisions) + 1)):
        raise ValueError("compound revisions must be consecutive and increasing")
    return deepcopy(history)


def _render_compound_history(facts, history, family):
    segments = [_render_compound(facts, family)]
    revisions = history[1:] if history and history[0]["revision"] == 1 else history
    for item in revisions:
        changed = dict(facts, answer=item["answer"],
                       anchor_colour=item["anchor_colour"], unrecorded_target=False)
        segments.append(f"Revision {item['revision']}: " + _render_compound(changed, family))
    return " ".join(segments)



def _compound_revision_values(body, facts, family):
    matches = re.findall(
        r"The recorded color of " + re.escape(facts["target"]) + r" is ([A-Za-z]+)\.", body)
    if len(matches) != 1:
        raise ValueError("rendered revision does not identify one target answer")
    answer = matches[0]
    return {"answer": answer,
            "anchor_colour": answer if family == COMPOUND_FAMILIES[1] else facts["anchor_colour"]}

def parse_compound_chronology(text, facts, family):
    """Independently replay every rendered revision against original facts."""
    segments = text.split(" Revision ")
    if not segments or segments[0] != _render_compound(facts, family):
        raise ValueError("original compound source was not preserved")
    history = []
    for segment in segments[1:]:
        prefix, separator, body = segment.partition(": ")
        if not separator or not prefix.isdigit() or not body:
            raise ValueError("malformed compound revision")
        history.append({"revision": int(prefix), "body": body})
    previous = 1
    for item in history:
        if item["revision"] != previous + 1:
            raise ValueError("compound revisions are not monotonic")
        previous = item["revision"]
    if history and not facts["unrecorded_target"]:
        initial = {"revision": 1, "answer": facts["answer"],
                   "anchor_colour": facts["anchor_colour"]}
        history = [initial, *history]
    elif not history and facts["unrecorded_target"]:
        history = []
    elif not history:
        history = [{"revision": 1, "answer": facts["answer"],
                    "anchor_colour": facts["anchor_colour"]}]
    for item in history[1:] if history and history[0]["revision"] == 1 else history:
        values = _compound_revision_values(item["body"], facts, family)
        changed = dict(facts, answer=values["answer"],
                       anchor_colour=values["anchor_colour"], unrecorded_target=False)
        if _render_compound(changed, family) != item["body"]:
            raise ValueError("rendered compound revision differs from its facts")
        item.update(values)
    return [{key: value for key, value in item.items() if key != "body"}
            for item in history]


def chronological_correction(row, new_color):
    """Apply the declared direct or compound-family correction oracle."""
    if not isinstance(new_color, str) or not new_color.isalpha():
        raise ValueError("invalid correction color")
    family = row.get("rule_family")
    if family not in COMPOUND_FAMILIES or not isinstance(row.get("facts"), dict):
        return direct_chronological_correction(row, new_color)
    original = deepcopy(row)
    facts = deepcopy(row.get("facts"))
    query = row.get("query_entity")
    if not isinstance(facts, dict) or not isinstance(query, str):
        raise ValueError("compound structured facts required")
    history = _compound_history(row)
    original_answer = history[0]["answer"] if history else row["answer"]
    if _compound_answer(facts, family, query) != original_answer:
        raise ValueError("original compound answer does not follow from its oracle")
    if new_color == row["answer"]:
        raise ValueError("correction must change the intended answer")
    revision = 1 + max([item["revision"] for item in history] or [1])
    changed = dict(facts, answer=new_color, unrecorded_target=False)
    if family == COMPOUND_FAMILIES[1]:
        changed["anchor_colour"] = new_color
    allowed = {"answer", "anchor_colour", "unrecorded_target"}
    if any(changed.get(key) != facts.get(key) for key in set(facts) - allowed):
        raise ValueError("correction changed an unrelated compound fact")
    if _compound_answer(changed, family, query) != new_color:
        raise ValueError("corrected answer does not follow from the compound oracle")
    history.append({"revision": revision, "answer": new_color,
                    "anchor_colour": changed["anchor_colour"]})
    text = _render_compound_history(facts, history, family)
    replayed = parse_compound_chronology(text, facts, family)
    if replayed != history or replayed[-1]["answer"] != new_color:
        raise ValueError("rendered compound chronology does not reproduce its oracle")
    return {**original, "facts": facts, "revision_history": history,
            "text": text, "answer": new_color, "chronology": revision,
            "correction": {"intended_entity": query, "rule_family": family,
                           "supersedes_revision": revision - 1,
                           "earlier_facts_preserved": True,
                           "unrelated_facts_unchanged": True,
                           "verification": "structured_and_rendered_compound_revision_oracle"}}