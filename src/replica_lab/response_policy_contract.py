"""Render one public query policy without examining labels or observed evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from string import Formatter

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_VERSION = "replica.response_policy_contract.v2"
PROJECTION_VERSION = "canonical_prompt_plus_separator_plus_public_query_policy_instruction_v2"
FIELD_ORDERS = {
    "register_xor8": ["RULE", "INPUT", "MASK"],
    "affine_mod97": ["RULE", "A", "X", "B"],
    "finite_state": ["RULE", "START", "SYMBOLS"],
    "tool_recovery": ["RULE", "STATUS", "ATTEMPT", "LIMIT"],
}
ANSWER_FORMATS = {
    "register_xor8": "only the base-10 integer result.",
    "affine_mod97": "only the base-10 integer result.",
    "finite_state": "only S0, S1, S2, or S3.",
    "tool_recovery": "exactly RETRY lookup, VERIFY receipt, or STOP.",
}
QUERY_BODIES = {
    "register_xor8": "compute INPUT XOR MASK using the observed rule",
    "affine_mod97": "compute the affine result using the observed rule",
    "finite_state": "report the final state after consuming SYMBOLS from START using the observed transition rule",
    "tool_recovery": "recommend the next operation for fictional tool lookup using the observed retry rule",
}
POLICY_TEXT = {"ask": "ask for it", "unknown": "respond UNKNOWN"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _template(value, fields):
    _require(isinstance(value, str) and bool(value.strip()) and "\r" not in value and "\n" not in value,
             "Policy-contract wording must be a nonempty single line")
    try:
        pieces = list(Formatter().parse(value))
    except ValueError as error:
        raise ValueError("Malformed policy-contract template") from error
    _require([field for _, field, _, _ in pieces if field is not None] == fields
             and all(not spec and conversion is None for _, _, spec, conversion in pieces),
             "Policy-contract template has undeclared projection inputs")


def parse_query_policy(family, canonical_query):
    """Accept only the exact public canonical family question and missing policy.

    The returned policy describes the requested fallback even when the evidence
    is complete. It never predicts whether that fallback should actually fire.
    """
    _require(isinstance(family, str) and family in QUERY_BODIES, "Unknown policy-contract family")
    _require(isinstance(canonical_query, str), "Canonical query must be text")
    prefix = r"At observation cutoff (?:0|[1-9][0-9]*), " + re.escape(QUERY_BODIES[family])
    matches = [policy for policy, text in POLICY_TEXT.items()
               if re.fullmatch(prefix + re.escape(". If a required field is unavailable, " + text + "."),
                               canonical_query) is not None]
    _require(len(matches) == 1, "Query does not match the exact canonical family/policy grammar")
    return matches[0]


def validate_contract(contract):
    _require(isinstance(contract, dict) and contract.get("schema_version") == CONTRACT_VERSION,
             "Unsupported response policy contract")
    _require(contract.get("status") == "predeclared_before_native_generation",
             "Policy contract must be predeclared before native generation")
    _require(contract.get("projection") == PROJECTION_VERSION
             and contract.get("append_separator") == "\n\n" and contract.get("joiner") == "\n",
             "Policy-contract projection/separators differ")
    _require(contract.get("projection_inputs") == ["family", "canonical_query", "static_contract_only"],
             "Policy contract declares private example-specific projection inputs")
    _template(contract.get("common_prefix"), [])
    _template(contract.get("evidence_instruction"), [])
    _template(contract.get("required_items_template"), ["field_order"])
    _template(contract.get("answer_template"), ["answer_format"])
    policies = contract.get("missing_policy_templates")
    _require(isinstance(policies, dict) and set(policies) == {"ask", "unknown"},
             "Policy contract must declare exactly ask/unknown templates")
    _template(policies["ask"], [])
    _template(policies["unknown"], [])
    _require("UNKNOWN" not in policies["ask"] and "Please provide" not in policies["unknown"],
             "Each missing-policy instruction must contain only its selected fallback")
    formats = contract.get("family_formats")
    _require(isinstance(formats, dict) and set(formats) == set(FIELD_ORDERS),
             "Policy-contract family inventory differs")
    for family, order in FIELD_ORDERS.items():
        _require(formats[family] == {"field_order": order, "answer_format": ANSWER_FORMATS[family]},
                 "Policy-contract static family grammar differs")
    comparison = contract.get("comparison", {})
    _require(isinstance(comparison, dict) and all(type(comparison.get(key)) is type(value)
        and comparison.get(key) == value for key, value in {
            "max_new_tokens": 32, "max_length": 512, "seed": 17, "do_sample": False,
            "num_beams": 1, "enable_thinking": False, "device": "mps", "dtype": "float32",
            "cpu_fallback": False, "native_jobs": 1,
            "additional_adaptive_runs": "none_in_this_preregistration"}.items()),
        "Policy-contract frozen native settings differ")
    return contract


def load_contract(path=None):
    path = ROOT / "configs/response_policy_contract_v2.json" if path is None else Path(path)
    return validate_contract(json.loads(path.read_text(encoding="utf-8")))


def render_instruction(family, canonical_query, contract):
    """Choose exactly one fallback instruction from public query text only."""
    validate_contract(contract)
    policy = parse_query_policy(family, canonical_query)
    grammar = contract["family_formats"][family]
    return contract["joiner"].join((contract["common_prefix"],
        contract["evidence_instruction"],
        contract["required_items_template"].format(field_order=", ".join(grammar["field_order"])),
        contract["answer_template"].format(answer_format=grammar["answer_format"]),
        contract["missing_policy_templates"][policy]))


instruction_appendix = render_instruction


def project_row(row, *, family, contract):
    """Deep-copy a verified original export and append only the public contract."""
    _require(isinstance(row, dict) and isinstance(row.get("prompt"), str),
             "Projection requires a canonical query")
    instruction = render_instruction(family, row["prompt"], contract)
    result = copy.deepcopy(row)
    result["prompt"] += contract["append_separator"] + instruction
    return result
