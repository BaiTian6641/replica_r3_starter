"""Strict response scoring shared by offline evaluation and training diagnostics."""
import re


def strict_integer(text: str) -> int | None:
    stripped = text.strip()
    if not re.fullmatch(r"[+-]?[0-9]+", stripped):
        return None
    return int(stripped)


def score_answer(completion: str, expected: str) -> dict:
    predicted_integer, expected_integer = strict_integer(completion), strict_integer(expected)
    exact = completion.strip() == expected.strip()
    integer_correct = (predicted_integer == expected_integer
                       if expected_integer is not None and predicted_integer is not None
                       else False)
    return {"exact_text_match": exact, "predicted_integer": predicted_integer,
            "expected_integer": expected_integer,
            "strict_integer_match": integer_correct if expected_integer is not None else None,
            "task_correct": integer_correct if expected_integer is not None else exact}
