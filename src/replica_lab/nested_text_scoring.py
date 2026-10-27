"""Versioned NL1 response generation, stopping and scoring contracts.

Scoring v2 accepts a bare expected answer or a complete declarative containing
that answer as a whole phrase.  Ambiguity, contradiction, competing candidate
answers and visibly incomplete output are rejected.  This is deliberately not a
claim of general semantic equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable, Sequence

SCORING_VERSION = "replica.nested_text.scoring.v2.1"
STOP_POLICY_VERSION = "replica.nested_text.sentence_stop.v1"
STOP_TOKENS = (1, 130073)
DEFAULT_MAX_NEW_TOKENS = 24
_STATUS_ANSWERS = frozenset({"unknown", "revoked"})
_AMBIGUOUS = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|might|could|may|think|believ|guess|"
    r"uncertain|unsure|unknown|not sure|not certain|cannot|can't|unable|sorry|"
    r"apolog\w*|don't know|do not know|or)\b", re.IGNORECASE)
_CONTRADICTORY = re.compile(
    r"\b(?:not|n't|never|no|wrong|incorrect|false|instead|but|however|although|"
    r"unless|except|contradict\w*)\b", re.IGNORECASE)
_INCOMPLETE_WORDS = frozenset({
    "and", "but", "or", "to", "of", "for", "with", "by", "is", "are",
    "was", "were", "has", "have", "had", "the", "a", "an", "that",
    "this", "these", "those", "it", "its", "as", "at", "in", "on",
})


def _normalized(value: str, *, strip_terminal: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("response text must be a string")
    text = value.replace("**", "").replace("__", "").replace("`", "")
    text = " ".join(text.lower().strip().split())
    if strip_terminal:
        text = text.rstrip(" \t.!?")
    return text


def _contains(output: str, answer: str) -> bool:
    return bool(re.search(rf"(?<![\w]){re.escape(answer)}(?![\w])", output))


def _visibly_incomplete(output: str) -> bool:
    if not output or output.endswith((",", ";", ":", "-", "(", "[", "{")):
        return True
    words = re.findall(r"[\w']+", output)
    return not words or words[-1] in _INCOMPLETE_WORDS


@dataclass(frozen=True)
class GenerationResult:
    """Raw IDs, decoded content and the actual stopping disposition."""

    text: str
    generated_ids: tuple[int, ...]
    content_ids: tuple[int, ...]
    stop_reason: str
    stop_token: int | None
    max_new_tokens: int

    @property
    def truncated(self) -> bool:
        return is_truncated_stop_reason(self.stop_reason)


@dataclass(frozen=True)
class ResponseScore:
    correct: bool
    reason: str
    normalized_expected: str
    normalized_output: str
    scoring_version: str = SCORING_VERSION


def is_truncated_stop_reason(reason: str) -> bool:
    return reason in {"token_budget", "missing_stop"}


def sentence_complete_text(text: str) -> bool:
    """Recognize the declared first complete-sentence boundary, not an answer."""
    if not isinstance(text, str):
        raise ValueError("sentence boundary requires text")
    stripped = text.strip()
    if not re.findall(r"[\w']+", stripped) or stripped.endswith("..."):
        return False
    if re.search(r"\d\.\d$", stripped):
        return False
    return stripped.endswith((".", "!", "?"))


def finalize_generation(ids: Sequence[int], *, decode: Callable[[Sequence[int],], str],
                        stop_tokens: Iterable[int] = STOP_TOKENS,
                        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                        completed_by: str | None = None) -> GenerationResult:
    """Validate IDs and separate token, sentence and budget completion."""
    if completed_by not in (None, "sentence_complete"):
        raise ValueError("completed_by must be sentence_complete or null")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    stops = frozenset(int(token) for token in stop_tokens)
    if not stops:
        raise ValueError("at least one stop token is required")
    raw = tuple(int(token) for token in ids)
    if not raw:
        raise ValueError("empty generation is not a completed response")
    if len(raw) > max_new_tokens:
        raise ValueError("generated IDs exceed the declared token budget")
    stop_index = next((index for index, token in enumerate(raw) if token in stops), None)
    if stop_index is None:
        content = raw
        reason = (completed_by or
                  ("token_budget" if len(raw) == max_new_tokens else "missing_stop"))
        stop_token = None
    else:
        if stop_index != len(raw) - 1:
            raise ValueError("generated IDs continue after the first stop token")
        if completed_by is not None:
            raise ValueError("stop token and sentence completion cannot both be reported")
        content = raw[:stop_index]
        reason = "stop_token"
        stop_token = raw[stop_index]
    text = decode(content).strip()
    return GenerationResult(
        text=text, generated_ids=raw, content_ids=tuple(content),
        stop_reason=reason, stop_token=stop_token,
        max_new_tokens=max_new_tokens)


def score_response(expected: str, output: str, *, candidate_answers: Iterable[str] = (),
                   truncated: bool = False) -> ResponseScore:
    """Score one response under the v2 grammar and contradiction controls."""
    expected_norm = _normalized(expected, strip_terminal=True)
    output_norm = _normalized(output)
    if not expected_norm or "\n" in expected or "\r" in expected:
        raise ValueError("expected answer must be nonempty and single-line")
    if truncated:
        return ResponseScore(False, "truncated", expected_norm, output_norm)
    if not output_norm:
        return ResponseScore(False, "empty", expected_norm, output_norm)
    if expected_norm in _STATUS_ANSWERS:
        correct = _normalized(output, strip_terminal=True) == expected_norm
        return ResponseScore(correct, "correct" if correct else "status_mismatch",
                             expected_norm, output_norm)
    if output_norm.endswith("?"):
        return ResponseScore(False, "ambiguous", expected_norm, output_norm)
    if _normalized(output, strip_terminal=True) == expected_norm:
        return ResponseScore(True, "correct", expected_norm, output_norm)
    if _visibly_incomplete(output_norm):
        return ResponseScore(False, "truncated", expected_norm, output_norm)
    if _AMBIGUOUS.search(output_norm):
        return ResponseScore(False, "ambiguous", expected_norm, output_norm)
    if _CONTRADICTORY.search(output_norm):
        return ResponseScore(False, "contradictory", expected_norm, output_norm)
    if not _contains(output_norm, expected_norm):
        return ResponseScore(False, "missing_answer", expected_norm, output_norm)
    for other in candidate_answers:
        other_norm = _normalized(other, strip_terminal=True)
        if other_norm and other_norm != expected_norm and _contains(output_norm, other_norm):
            return ResponseScore(False, "other_candidate", expected_norm, output_norm)
    return ResponseScore(True, "correct", expected_norm, output_norm)

__all__ = [
    "DEFAULT_MAX_NEW_TOKENS", "GenerationResult", "ResponseScore",
    "SCORING_VERSION", "STOP_POLICY_VERSION", "STOP_TOKENS", "finalize_generation",
    "score_response", "is_truncated_stop_reason", "sentence_complete_text",
]
