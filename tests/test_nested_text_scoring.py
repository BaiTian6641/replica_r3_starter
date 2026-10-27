"""CPU contracts for NL1 v2 response stopping and scoring."""
from __future__ import annotations

import pytest

from replica_lab.nested_text_scoring import (
    ANSWER_GRAMMAR_VERSION,
    SCORING_VERSION,
    finalize_generation,
    is_truncated_stop_reason,
    score_response,
    sentence_complete_text,
)


@pytest.mark.parametrize("expected, output", [
    ("43", "43"),
    ("43", "43."),
    ("43", "The answer is 43."),
    ("43", "**43**"),
    ("56", "8 times 7 is **56**."),
    ("Paris", "The capital of France is Paris."),
    ("Paris", "Paris is the capital of France."),
    ("81", "The square of 9 is 81"),
    ("366", "A leap year has **366 days**."),
    ("closes", "The library closes at dusk."),
    ("TevufaRevised", "TevufaRevised is correct"),
])
def test_positive_complete_answers_are_accepted(expected, output):
    score = score_response(expected, output, candidate_answers=["Tevufa", "Mipezu"])
    assert score.correct is True
    assert score.reason == "correct"


@pytest.mark.parametrize("expected, output, reason", [
    ("43", "44", "missing_answer"),
    ("43", "The answer is 44.", "missing_answer"),
    ("43", "The answer is 43 and 56.", "other_candidate"),
    ("Tevufa", "Tevufa is Mipezu", "other_candidate"),
    ("Tevufa", "maybe Tevufa", "ambiguous"),
    ("Tevufa", "Tevufa or Mipezu", "ambiguous"),
    ("Tevufa", "Tevufa?", "ambiguous"),
    ("Tevufa", "I think Tevufa is the answer", "ambiguous"),
    ("Tevufa", "Tevufa is wrong", "contradictory"),
    ("Tevufa", "Tevufa is not correct", "contradictory"),
    ("Tevufa", "Tevufa is not Mipezu", "contradictory"),
    ("43", "To find the sum of 17 and", "truncated"),
    ("12", "To divide 144 by 12,", "truncated"),
    ("run", "The word that is a verb is **", "truncated"),
    ("Paris", "The capital of France is", "truncated"),
])
def test_incorrect_ambiguous_contradictory_and_truncated_are_rejected(
        expected, output, reason):
    score = score_response(
        expected, output,
        candidate_answers=["43", "56", "Tevufa", "Mipezu"])
    assert score.correct is False
    assert score.reason == reason


def test_explicit_truncation_rejects_even_an_exact_answer():
    score = score_response("43", "43", truncated=True)
    assert score.correct is False
    assert score.reason == "truncated"


def test_empty_output_is_rejected():
    assert score_response("43", "").reason == "empty"


@pytest.mark.parametrize("expected, output, correct", [
    ("unknown", "unknown", True),
    ("unknown", "unknown.", True),
    ("unknown", "I don't know", False),
    ("revoked", "revoked", True),
    ("revoked", "unknown", False),
])
def test_status_answers_use_the_exact_status_grammar(expected, output, correct):
    score = score_response(expected, output)
    assert score.correct is correct
    assert score.reason == ("correct" if correct else "status_mismatch")


def test_generation_stops_at_eos_and_im_end_without_counting_them_as_content():
    for stop in (1, 130073):
        result = finalize_generation(
            [15876, stop], decode=lambda ids: "answer" if ids == (15876,) else "",
            max_new_tokens=2)
        assert result.text == "answer"
        assert result.generated_ids == (15876, stop)
        assert result.content_ids == (15876,)
        assert result.stop_reason == "stop_token"
        assert result.stop_token == stop
        assert result.truncated is False


def test_token_budget_and_missing_stop_are_distinct_from_clean_stop():
    budget = finalize_generation([5, 6, 7], decode=lambda ids: "x", max_new_tokens=3)
    assert budget.stop_reason == "token_budget"
    assert budget.truncated is True
    missing = finalize_generation([5, 6], decode=lambda ids: "x", max_new_tokens=3)
    assert missing.stop_reason == "missing_stop"
    assert missing.truncated is True


def test_stop_at_budget_is_not_mislabeled_as_truncated():
    result = finalize_generation([5, 1], decode=lambda ids: "x", max_new_tokens=2)
    assert result.stop_reason == "stop_token"
    assert result.truncated is False


def test_terminal_stop_with_empty_text_is_not_a_valid_answer():
    result = finalize_generation([1], decode=lambda ids: "", max_new_tokens=1)
    assert result.stop_reason == "stop_token"
    assert score_response("43", result.text).reason == "empty"


@pytest.mark.parametrize("ids, kwargs", [
    ([], {}),
    ([1, 5], {}),
    ([5, 6, 7], {"max_new_tokens": 2}),
    ([5], {"stop_tokens": []}),
    ([5], {"max_new_tokens": 0}),
])
def test_invalid_generation_contracts_fail_closed(ids, kwargs):
    with pytest.raises(ValueError):
        finalize_generation(ids, decode=lambda values: "x", **kwargs)


def test_sentence_boundary_is_completion_not_answer_selection():
    assert sentence_complete_text("The answer is 43.")
    assert sentence_complete_text("8 times 7 is **56**.")
    assert sentence_complete_text("Tevufa?")
    assert not sentence_complete_text("17 + 26")
    assert not sentence_complete_text("The value is 3.14")
    assert not sentence_complete_text("The value is...")


def test_sentence_completion_is_not_truncation():
    result = finalize_generation(
        [7, 8], decode=lambda ids: "A leap year has **366 days**.",
        max_new_tokens=24, completed_by="sentence_complete")
    assert result.stop_reason == "sentence_complete"
    assert result.stop_token is None
    assert result.truncated is False
    assert not is_truncated_stop_reason(result.stop_reason)
    with pytest.raises(ValueError, match="cannot both"):
        finalize_generation([7, 1], decode=lambda ids: "x",
                            completed_by="sentence_complete")
    with pytest.raises(ValueError, match="completed_by"):
        finalize_generation([7], decode=lambda ids: "x", completed_by="other")


# ---------------------------------------------------------------------------
# Answer-grammar boundaries (answer_grammar.v2.1.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expected, output", [
    ("Paris", "**Paris**"),
    ("Paris", "*Paris*"),
    ("Paris", "_Paris_"),
    ("Paris", "__Paris__"),
    ("Paris", "~~Paris~~"),
    ("Paris", "`Paris`"),
    ("Paris", "**Paris.**"),
    ("Paris", "### Paris"),
    ("Paris", "- Paris"),
    ("Paris", "1. Paris"),
    ("Paris", "Paris!"),
    ("43", "The answer is 43."),
    ("43", "**43**."),
    ("43", "43."),
])
def test_markdown_flourishes_and_terminal_punctuation_are_accepted(expected, output):
    score = score_response(expected, output)
    assert score.correct is True
    assert score.reason == "correct"


@pytest.mark.parametrize("expected, output", [
    ("43", "43.5"),
    ("43", "The answer is 43.5."),
    ("43", "43,000"),
    ("43", "1,043"),
    ("43", "143"),
    ("43", "430"),
    ("Tevufa", "Tevufa's"),
    ("Tevufa", "**Tevufa's**"),
    ("9", "9.5"),
])
def test_extended_numbers_and_possessives_are_not_the_expected_answer(expected, output):
    score = score_response(expected, output, candidate_answers=["81"])
    assert score.correct is False
    assert score.reason in {"missing_answer", "other_candidate"}


@pytest.mark.parametrize("output", [
    "44",
    "The answer is 44.",
    "Purple",
    "The meaning is unclear.",
    "The meaning is elsewhere.",
    "It is recorded in another entry.",
    "Tevufa is wrong",
    "Tevufa is not correct",
    "maybe Tevufa",
    "I think Tevufa is the answer",
    "Tevufa?",
    "Is it Tevufa?",
    "",
    "   ",
])
def test_arbitrary_wrong_answers_never_pass(output):
    score = score_response("Tevufa", output, candidate_answers=["Mipezu"])
    assert score.correct is False


def test_known_answer_string_must_still_be_present_for_a_non_status_answer():
    assert score_response("Tevufa", "Mipezu").reason == "missing_answer"
    assert score_response("Tevufa", "The answer is Mipezu.").reason == "missing_answer"
    assert score_response("Tevufa", "Tevufa").correct is True


@pytest.mark.parametrize("expected, output, correct", [
    ("unknown", "unknown", True),
    ("unknown", "Unknown.", True),
    ("unknown", "**unknown**", True),
    ("unknown", "The answer is unknown.", False),
    ("unknown", "I don't know", False),
    ("unknown", "revoked", False),
    ("revoked", "revoked", True),
    ("revoked", "Revoked!", True),
    ("revoked", "`revoked`", True),
    ("revoked", "unknown", False),
    ("revoked", "The answer is revoked.", False),
])
def test_unknown_and_revoked_are_exact_status_answers(expected, output, correct):
    score = score_response(expected, output)
    assert score.correct is correct
    assert score.reason == ("correct" if correct else "status_mismatch")


def test_status_answers_still_reject_conflicting_known_answers():
    score = score_response("unknown", "Tevufa", candidate_answers=["Tevufa"])
    assert score.correct is False
    assert score.reason == "status_mismatch"


@pytest.mark.parametrize("expected, output", [
    ("unknown", "unknown"),
    ("revoked", "revoked"),
    ("43", "43"),
])
def test_generation_failure_empty_and_truncated_gates_receive_no_credit(expected, output):
    truncated = score_response(expected, output, truncated=True)
    assert truncated.correct is False
    assert truncated.reason == "truncated"
    empty = score_response(expected, "")
    assert empty.correct is False and empty.reason == "empty"
    whitespace = score_response(expected, "   ")
    assert whitespace.correct is False and whitespace.reason == "empty"


def test_a_budget_stop_with_empty_content_is_an_empty_answer_not_a_pass():
    result = finalize_generation([1], decode=lambda ids: "", max_new_tokens=24)
    assert result.stop_reason == "stop_token"
    assert result.truncated is False
    assert score_response("43", result.text).reason == "empty"
    budget = finalize_generation([9, 10], decode=lambda ids: "4", max_new_tokens=2)
    assert budget.truncated is True
    assert score_response("43", budget.text, truncated=budget.truncated).reason == (
        "truncated")


def test_scorer_rejects_non_text_output_instead_of_scoring_it():
    with pytest.raises(ValueError):
        score_response("43", None)
    with pytest.raises(ValueError):
        score_response("43", 43)


def test_every_score_records_the_declared_scoring_and_grammar_versions():
    for expected, output in (("43", "43"), ("43", "44"), ("43", ""),
                             ("unknown", "unknown"), ("unknown", "revoked")):
        score = score_response(expected, output)
        assert score.scoring_version == SCORING_VERSION
        assert score.grammar_version == ANSWER_GRAMMAR_VERSION
    assert ANSWER_GRAMMAR_VERSION.endswith("v2.1.1")
