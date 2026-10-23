"""Deterministic public-family instruction projection, independent of labels.

The renderer accepts only a family name and the predeclared static contract.
There is deliberately no episode, action, target, value or missingness input.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from string import Formatter

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_VERSION = "replica.response_contract.v1"
PROJECTION_VERSION = "canonical_prompt_plus_separator_plus_family_instruction_v1"
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


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _template(template, field):
    _require(isinstance(template, str) and bool(template.strip())
             and "\r" not in template and "\n" not in template,
             "Response-contract template must be a nonempty single line")
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as error:
        raise ValueError("Malformed response-contract template") from error
    _require([name for _, name, _, _ in parsed if name is not None] == [field]
             and all(not specification and conversion is None for _, _, specification, conversion in parsed),
             "Response-contract template has undeclared projection inputs")


def validate_contract(contract):
    """Reject any v1 projection that depends on example-specific information."""
    _require(isinstance(contract, dict) and contract.get("schema_version") == CONTRACT_VERSION,
             "Unsupported response contract")
    _require(contract.get("status") == "predeclared_before_native_generation",
             "Response contract must be predeclared before native generation")
    _require(contract.get("projection") == PROJECTION_VERSION
             and contract.get("append_separator") == "\n\n" and contract.get("joiner") == "\n",
             "Response-contract projection/separators differ")
    _require(contract.get("projection_inputs") == ["family", "static_contract_only"],
             "Response contract declares example-specific projection inputs")
    prefix = contract.get("common_prefix")
    _require(isinstance(prefix, str) and bool(prefix.strip()) and "\r" not in prefix and "\n" not in prefix,
             "Response-contract prefix must be a nonempty single line")
    _template(contract.get("answer_template"), "answer_format")
    _template(contract.get("missing_template"), "field_order")
    formats = contract.get("family_formats")
    _require(isinstance(formats, dict) and set(formats) == set(FIELD_ORDERS),
             "Response-contract family inventory differs")
    for family, order in FIELD_ORDERS.items():
        _require(formats[family] == {"field_order": order, "answer_format": ANSWER_FORMATS[family]},
                 "Response-contract static family grammar differs")
    comparison = contract.get("comparison", {})
    _require(isinstance(comparison, dict) and all(type(comparison.get(name)) is type(expected)
        and comparison.get(name) == expected for name, expected in {
        "max_new_tokens": 32, "max_length": 512, "seed": 17, "do_sample": False,
        "num_beams": 1, "enable_thinking": False, "device": "mps", "dtype": "float32",
        "cpu_fallback": False, "native_jobs": 1, "additional_adaptive_runs": "none_in_this_preregistration"}.items()),
        "Response-contract frozen native settings differ")
    return contract


def load_contract(path=None):
    path = ROOT / "configs/response_contract_v1.json" if path is None else Path(path)
    return validate_contract(json.loads(path.read_text(encoding="utf-8")))


def render_instruction(family, contract):
    """Render exactly the registered family grammar, without inspecting a row."""
    validate_contract(contract)
    _require(isinstance(family, str) and family in FIELD_ORDERS, "Unknown response-contract family")
    grammar = contract["family_formats"][family]
    return contract["joiner"].join((contract["common_prefix"],
        contract["answer_template"].format(answer_format=grammar["answer_format"]),
        contract["missing_template"].format(field_order=", ".join(grammar["field_order"]))))


instruction_appendix = render_instruction


def project_row(row, *, family, contract):
    """Copy an existing verified export and change only its prompt suffix."""
    _require(isinstance(row, dict) and isinstance(row.get("prompt"), str) and bool(row["prompt"]),
             "Projection requires a nonempty canonical prompt")
    instruction = render_instruction(family, contract)
    result = copy.deepcopy(row)
    result["prompt"] = row["prompt"] + contract["append_separator"] + instruction
    return result
