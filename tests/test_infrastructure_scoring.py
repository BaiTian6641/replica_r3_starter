"""Response validity, decision truth and termination remain separate axes."""
import pytest

from replica_lab.infrastructure_scoring import parse_completion, score_completion


def target(action="answer", response="9"):
    return {"action": action, "response": response, "support_event_ids": ["observed-rule"]}


@pytest.mark.parametrize("completion", ["The answer is 9", "9 because XOR", "9\n10", "9.0", "²", "٩",
    "<think></think>9", "9</s>", "```9```", "", "UNKNOWN then 9", "Please provide MASK. 9"])
def test_parser_rejects_explanations_multiple_values_special_tokens_and_nonascii_numbers(completion):
    result = score_completion(completion, target(), family="register_xor8")
    assert not result["format_valid"] and not result["task_correct"]


def test_numeric_semantic_correctness_and_exact_text_are_distinct():
    result = score_completion(" \n+009\t", target(), family="register_xor8")
    assert result["format_valid"] and result["task_correct"]
    assert not result["exact_text_match"]
    wrong = score_completion("10", target(), family="register_xor8")
    assert wrong["format_valid"] and wrong["action_correct"] and not wrong["value_correct"]
    assert not wrong["task_correct"]


def test_wrong_action_is_valid_format_and_separate_from_wrong_payload():
    result = score_completion("UNKNOWN", target(), family="register_xor8")
    assert result["format_valid"] and result["parsed_action"] == "unknown"
    assert not result["action_correct"] and not result["task_correct"]
    wrong_ask = score_completion("Please provide INPUT.", target("ask", "Please provide MASK."),
                                 family="register_xor8")
    assert wrong_ask["format_valid"] and wrong_ask["action_correct"]
    assert wrong_ask["decision_correct"] and not wrong_ask["value_correct"]


@pytest.mark.parametrize("completion", ["Please provide MASK, INPUT.", "Please provide MASK, MASK.",
    "Please provide SECRET.", "Please provide input.", "Please provide INPUT", "unknown"])
def test_ask_field_names_order_duplicates_and_case_are_locked(completion):
    assert not parse_completion(completion, family="register_xor8")["format_valid"]


def test_multiple_missing_fields_parse_in_declared_order():
    response = "Please provide RULE, A, B."
    result = score_completion(response, target("ask", response), family="affine_mod97")
    assert result["parsed_value"] == ["RULE", "A", "B"] and result["task_correct"]


def test_state_and_recovery_decisions_use_family_grammar():
    assert score_completion("S3", target(response="S3"), family="finite_state")["task_correct"]
    assert not parse_completion("S4", family="finite_state")["format_valid"]
    assert not parse_completion("3", family="finite_state")["format_valid"]
    retry = score_completion("RETRY lookup", target(response="VERIFY receipt"), family="tool_recovery")
    assert retry["format_valid"] and retry["action_correct"]
    assert retry["decision_type"] == "retry" and not retry["decision_correct"]
    assert not retry["task_correct"]
    assert not parse_completion("RETRY delete", family="tool_recovery")["format_valid"]
    assert not parse_completion("STOP", family="register_xor8")["format_valid"]


def test_task_truth_is_independent_of_eos_and_cap_status():
    capped = score_completion("9", target(), family="register_xor8", ended_by_eos=False, hit_token_budget=True)
    assert capped["task_correct"] and capped["hit_token_budget"] and not capped["ended_by_eos"]
    ended = score_completion("10", target(), family="register_xor8", ended_by_eos=True, hit_token_budget=False)
    assert not ended["task_correct"] and ended["ended_by_eos"]
    assert score_completion("9", target(), family="register_xor8")["ended_by_eos"] is None


def test_invalid_gold_unknown_family_and_termination_metadata_fail_closed():
    with pytest.raises(ValueError, match="unknown infrastructure"):
        parse_completion("9", family="unsupported")
    with pytest.raises(ValueError, match="expected action/response"):
        score_completion("9", target("ask", "9"), family="register_xor8")
    with pytest.raises(ValueError, match="must be boolean"):
        score_completion("9", target(), family="register_xor8", ended_by_eos=1)
    with pytest.raises(ValueError, match="cannot both"):
        score_completion("9", target(), family="register_xor8", ended_by_eos=True, hit_token_budget=True)
