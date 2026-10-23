"""Fixed response grammar and independent error dimensions for v2 fixtures."""
from __future__ import annotations

import re

from .infrastructure_solver import FAMILIES, _design


SCORING_VERSION = "replica.infrastructure-generation-scoring.v2"


def parse_completion(completion, *, family):
    """Parse one complete response; never extract an answer from extra prose."""
    if family not in FAMILIES:
        raise ValueError("unknown infrastructure scoring family")
    if not isinstance(completion, str):
        raise ValueError("completion must be text")
    text = completion.strip()
    result = {"format_valid": False, "parsed_action": None, "decision_type": None,
              "parsed_value": None}
    if text == "UNKNOWN":
        return {"format_valid": True, "parsed_action": "unknown", "decision_type": "unknown",
                "parsed_value": None}
    ask = re.fullmatch(r"Please provide ([A-Z]+(?:, [A-Z]+)*)\.", text)
    if ask:
        fields = ask[1].split(", ")
        order = ["RULE", *_design()["families"][family]["fact_fields"]]
        if (len(fields) == len(set(fields)) and all(field in order for field in fields)
                and fields == [field for field in order if field in fields]):
            return {"format_valid": True, "parsed_action": "ask", "decision_type": "ask",
                    "parsed_value": fields}
        return result
    kind = _design()["families"][family]["response_kind"]
    if kind == "integer" and re.fullmatch(r"[+-]?[0-9]+", text):
        try:
            value = int(text)
        except ValueError:
            return result
        return {"format_valid": True, "parsed_action": "answer", "decision_type": "integer",
                "parsed_value": value}
    if kind == "state" and text in _design()["families"][family]["states"]:
        return {"format_valid": True, "parsed_action": "answer", "decision_type": "state",
                "parsed_value": text}
    if kind == "recovery" and text in _design()["families"][family]["allowed_responses"]:
        return {"format_valid": True, "parsed_action": "answer",
                "decision_type": text.split(" ")[0].lower(), "parsed_value": text}
    return result


def score_completion(completion, expected, *, family, ended_by_eos=None, hit_token_budget=None):
    """Score task/format/action separately; EOS and caps never alter task truth."""
    if not isinstance(expected, dict) or not {"action", "response"}.issubset(expected):
        raise ValueError("expected must be an independently solved target")
    gold = parse_completion(expected["response"], family=family)
    if not gold["format_valid"] or gold["parsed_action"] != expected["action"]:
        raise ValueError("expected action/response violates fixed scoring grammar")
    for name, flag in (("ended_by_eos", ended_by_eos), ("hit_token_budget", hit_token_budget)):
        if flag is not None and type(flag) is not bool:
            raise ValueError(f"{name} must be boolean or unavailable")
    if ended_by_eos is True and hit_token_budget is True:
        raise ValueError("terminal EOS and cap hit cannot both be true")
    parsed = parse_completion(completion, family=family)
    action_correct = parsed["format_valid"] and parsed["parsed_action"] == expected["action"]
    decision_correct = parsed["format_valid"] and parsed["decision_type"] == gold["decision_type"]
    value_correct = parsed["format_valid"] and parsed["parsed_value"] == gold["parsed_value"]
    return {**parsed, "action_correct": action_correct, "decision_correct": decision_correct,
            "value_correct": value_correct,
            "task_correct": action_correct and decision_correct and value_correct,
            "exact_text_match": completion.strip() == expected["response"].strip(),
            "ended_by_eos": ended_by_eos, "hit_token_budget": hit_token_budget,
            "scoring_version": SCORING_VERSION}
