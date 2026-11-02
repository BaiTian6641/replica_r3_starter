"""Focused successor acceptance contracts; no native or learned-quality claim."""
from copy import deepcopy

import pytest

from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.intrinsic_memory_acceptance_v13 import (
    InvalidationContractError, assess_response, chronological_correction,
    expect_invalidation, parse_chronological, render_chronological)
from replica_lab.intrinsic_memory_budget_v12 import BudgetError
from replica_lab.intrinsic_memory_worlds_v11 import oracle


@pytest.mark.parametrize("control", ["contextual", "exact_text"])
def test_content_correctness_does_not_hide_truncation(control):
    assessment = assess_response("red", "red", control=control,
                                 terminated=False, truncated=True)
    assert assessment.content_correct and not assessment.complete_correct
    assert assessment.truncated and not assessment.complete
    complete = assess_response("red", "red", control=control,
                               terminated=True, truncated=False)
    assert complete.complete_correct
    wrong = assess_response("red", "blue", control=control,
                            terminated=True, truncated=False)
    assert wrong.complete and not wrong.complete_correct


def test_unknown_false_recall_is_separate_from_completion():
    result = assess_response("unknown", "red", control="contextual",
                             terminated=True, truncated=False)
    assert result.false_recall and result.complete and not result.complete_correct


def test_expected_invalidation_is_typed_and_operational_errors_escape():
    def stale():
        from replica_lab.contextual_memory import AuthorityRejection
        raise AuthorityRejection("stale_snapshot", "stale source epoch")
    outcome = expect_invalidation("stale_snapshot", stale)
    assert outcome.kind == "stale_snapshot" and outcome.error_type == "AuthorityRejection"
    with pytest.raises(InvalidationContractError):
        expect_invalidation("stale_snapshot", lambda: None)
    for error in (ContextualMemoryError("corrupt payload"), BudgetError("cap"), OSError("disk"), RuntimeError("implementation")):
        def fail(error=error):
            raise error
        with pytest.raises(type(error)):
            expect_invalidation("revoked_read", fail)


def row():
    facts = [{"entity": "Alpha", "color": "red", "revision": 1},
             {"entity": "Beta", "color": "blue", "revision": 2}]
    return {"world_id": "structured", "facts": facts, "query_entity": "Alpha",
            "answer": oracle(facts, "Alpha"), "chronology": 2}


def test_correction_changes_only_intended_fact_and_replays_rendered_oracle():
    original = row()
    before = deepcopy(original)
    corrected = chronological_correction(original, "green")
    assert original == before
    assert corrected["facts"][:2] == before["facts"]
    assert corrected["facts"][-1] == {"entity": "Alpha", "color": "green", "revision": 3}
    assert corrected["answer"] == "green" and corrected["chronology"] == 3
    assert oracle(corrected["facts"], "Beta") == "blue"
    parsed = parse_chronological(corrected["text"])
    assert parsed == corrected["facts"] and oracle(parsed, "Alpha") == "green"
    assert corrected["correction"]["unrelated_facts_unchanged"]


def test_second_correction_preserves_entire_chronology():
    first = chronological_correction(row(), "green")
    second = chronological_correction(first, "gold")
    assert [fact["color"] for fact in second["facts"] if fact["entity"] == "Alpha"] == [
        "red", "green", "gold"]
    assert [fact["revision"] for fact in second["facts"]] == [1, 2, 3, 4]
    assert oracle(parse_chronological(second["text"]), "Alpha") == "gold"
    assert oracle(parse_chronological(second["text"]), "Beta") == "blue"


def test_correction_refuses_noop_or_inconsistent_original():
    with pytest.raises(ValueError, match="change the intended answer"):
        chronological_correction(row(), "red")
    damaged = row()
    damaged["answer"] = "blue"
    with pytest.raises(ValueError, match="original answer"):
        chronological_correction(damaged, "green")


def test_rendered_source_is_strict_and_chronological():
    facts = row()["facts"]
    text = render_chronological(facts)
    assert parse_chronological(text) == facts
    with pytest.raises(ValueError, match="incomplete"):
        parse_chronological(text.replace(" No color is recorded for any other entity.", ""))
    with pytest.raises(ValueError):
        parse_chronological("unexpected " + text)
