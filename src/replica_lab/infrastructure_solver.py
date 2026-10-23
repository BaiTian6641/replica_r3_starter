"""Independent raw-source solvers for the synthetic infrastructure v2 release.

The sampled generator and its labels are deliberately not imported. Truth is
reconstructed from archived source bytes; permission to use that truth remains
the episode compiler's separate responsibility.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from .contracts import canonical_json
from .episodes import Episode, USES, validate_episode


SOLVER_VERSION = "replica.infrastructure-source-solver.v2"
DESIGN_PATH = Path(__file__).resolve().parents[2] / "configs/infrastructure_slice_v2_design.json"
FAMILIES = ("register_xor8", "affine_mod97", "finite_state", "tool_recovery")


@lru_cache(maxsize=1)
def _design():
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if design["schema_version"] != "replica.infrastructure_slice_design.v2":
        raise ValueError("unsupported infrastructure design")
    if tuple(design["family_order"]) != FAMILIES:
        raise ValueError("unsupported infrastructure family set")
    return design


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key in source")
        value[key] = item
    return value


def _json(text):
    try:
        return json.loads(text, object_pairs_hook=_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValueError(f"nonfinite JSON value {value}")))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid source JSON") from error


def _safe_source(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source path must remain inside release")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("source path escapes release")
    return path


def _bound_source(payload, release_root, source_record):
    """Authenticate source spans and bind canonical fields to archived bytes."""
    payload = payload.to_dict() if isinstance(payload, Episode) else copy.deepcopy(payload)
    if not isinstance(payload, dict):
        raise ValueError("episode must be an object")
    # Structural validation must not make truth reconstruction depend on a
    # claimed label, including its claimed support. The claim is checked only
    # by independent_check after solving the evidence/query side.
    payload["target"] = {"action": "unknown", "response": "UNKNOWN", "support_event_ids": []}
    payload = validate_episode(payload).to_dict()
    design = _design()
    try:
        raw = _safe_source(release_root, source_record["path"]).read_bytes()
        if _sha256(raw) != source_record["sha256"] or len(raw) != source_record["bytes"]:
            raise ValueError("raw source bytes/hash mismatch")
        raw.decode("utf-8", errors="strict")
        lines = raw.splitlines(keepends=True)
        if not lines or any(not line.endswith(b"\n") for line in lines):
            raise ValueError("raw source requires complete newline-delimited JSON")
        header, events = _json(lines[0]), [_json(line) for line in lines[1:]]
        if set(header) != set(design["source_header_required"]):
            raise ValueError("raw source header fields differ from design")
        if header["schema_version"] != design["source_header_schema"] or header["synthetic"] is not True:
            raise ValueError("not a declared synthetic infrastructure world")
        if (header["family"] not in FAMILIES or header["scenario"] not in design["scenarios"]
                or header["split"] not in design["roles_order"]):
            raise ValueError("unknown source family/scenario/role")
        if (header["branch_id"] != "source" or header["privacy"] != "public"
                or type(header["seed"]) is not int or header["seed"] < 0):
            raise ValueError("source scope/seed differs from design")
        if (not isinstance(header["permissions"], list)
                or header["permissions"] != design["permissions"][header["split"]]
                or not set(header["permissions"]).issubset(USES)):
            raise ValueError("raw source permissions differ from design")
        for key in ("world_id", "group_id", "source_id", "subject_id", "actor_id"):
            if not isinstance(header[key], str) or not header[key].strip():
                raise ValueError("raw source identity must be nonempty")
        for key in ("source_id", "world_id", "group_id", "split", "family", "scenario"):
            if source_record[key] != header[key]:
                raise ValueError("source manifest identity/lineage differs from raw source")
        spans = source_record["event_spans"]
        if len(events) != len(spans):
            raise ValueError("source span inventory differs")
        offset = len(lines[0])
        for event, line, span in zip(events, lines[1:], spans):
            expected = {"event_id": event["event_id"], "byte_start": offset,
                        "byte_end": offset + len(line), "sha256": _sha256(line)}
            if span != expected or raw[span["byte_start"]:span["byte_end"]] != line:
                raise ValueError("raw source span mismatch")
            offset += len(line)
        if len(payload["sources"]) != 1 or payload["events"] != events:
            raise ValueError("episode events differ from bound raw source")
        source = payload["sources"][0]
        for key in ("source_id", "group_id", "split", "subject_id", "branch_id", "actor_id", "privacy"):
            if source[key] != header[key]:
                raise ValueError("episode source identity/role differs from raw source")
        if source["sha256"] != _sha256(raw) or source["kind"] != "synthetic":
            raise ValueError("episode source hash/kind binding differs")
        if any(not set(permissions).issubset(header["permissions"])
               for permissions in (source["permissions"], payload["permissions"])):
            raise ValueError("episode escalates raw source use permissions")
    except (KeyError, TypeError, UnicodeDecodeError, IndexError) as error:
        raise ValueError("malformed source binding") from error
    return payload, header, events


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _validate_rule(family, rule):
    spec = _design()["families"][family]
    if family != "finite_state":
        if canonical_json(rule) != canonical_json(spec["rule"]):
            raise ValueError("observed rule differs from family contract")
        return
    if (not isinstance(rule, dict) or set(rule) != {"operation", "consume", "transitions"}
            or rule["operation"] != "finite_state" or rule["consume"] != spec["rule"]["consume"]):
        raise ValueError("invalid finite-state rule")
    states, alphabet = spec["states"], spec["alphabet"]
    transitions = rule["transitions"]
    if not isinstance(transitions, dict) or set(transitions) != set(states):
        raise ValueError("finite-state transition states differ")
    for state in states:
        if (not isinstance(transitions[state], dict)
                or set(transitions[state]) != set(alphabet)
                or any(value not in states for value in transitions[state].values())):
            raise ValueError("invalid finite-state transition mapping")
    for symbol in alphabet:
        if {transitions[state][symbol] for state in states} != set(states):
            raise ValueError("finite-state symbol transitions must be permutations")


def _validate_fact(family, name, value):
    spec = _design()["families"][family]
    if name not in spec["fact_fields"]:
        raise ValueError("fact field does not belong to family")
    valid = False
    if family == "register_xor8":
        valid = _integer(value, 0, 255)
    elif family == "affine_mod97":
        valid = _integer(value, 1 if name == "A" else 0, 96)
    elif family == "finite_state":
        valid = (isinstance(value, str) and (value in spec["states"] if name == "START"
                 else re.fullmatch(r"[ab]{2,4}", value) is not None))
    elif family == "tool_recovery":
        valid = (isinstance(value, str) and value in {"ok", "transient", "permanent"}
                 if name == "STATUS" else _integer(value, 0 if name == "ATTEMPT" else 1,
                                                   9 if name == "ATTEMPT" else 10))
    if not valid:
        raise ValueError("fact value is outside family domain")


def _read_observation(family, text):
    prefixes = _design()["event_text"]
    if text.startswith(prefixes["rule_prefix"]):
        encoded = text[len(prefixes["rule_prefix"]):]
        rule = _json(encoded)
        if canonical_json(rule) != encoded:
            raise ValueError("observed rule JSON is not canonical")
        _validate_rule(family, rule)
        return "RULE", rule
    if text.startswith(prefixes["fact_prefix"]):
        encoded = text[len(prefixes["fact_prefix"]):]
        fact = _json(encoded)
        if not isinstance(fact, dict) or set(fact) != {"name", "value"} or canonical_json(fact) != encoded:
            raise ValueError("observed fact JSON differs from contract")
        _validate_fact(family, fact["name"], fact["value"])
        return fact["name"], fact["value"]
    raise ValueError("source observation has unknown rule/fact grammar")


def _question_policy(family, prompt, cutoff=None):
    design = _design()
    if family not in FAMILIES or not isinstance(prompt, str):
        raise ValueError("unknown family or invalid question")
    spec = design["families"][family]
    if cutoff is None:
        match = re.match(r"At observation cutoff ([0-9]+), ", prompt)
        if match is None:
            raise ValueError("question/family/cutoff differs from fixed solver grammar")
        try:
            cutoff = int(match[1])
        except ValueError as error:
            raise ValueError("invalid observation cutoff") from error
    for policy, instruction in design["missing_instructions"].items():
        question = design["query_template"].format(cutoff=cutoff, query_body=spec["query_body"],
                                                  missing_instruction=instruction)
        if prompt == question:
            return policy
    raise ValueError("question/family/cutoff differs from fixed solver grammar")


def _solve_fields(family, fields, missing_policy):
    design, spec = _design(), _design()["families"][family]
    missing = [name for name in ["RULE", *spec["fact_fields"]] if name not in fields]
    if missing:
        response = (design["ask_response_template"].format(fields=design["ask_field_separator"].join(missing))
                    if missing_policy == "ask" else "UNKNOWN")
        return {"action": missing_policy, "response": response}
    rule = fields["RULE"]
    if family == "register_xor8":
        response = str(fields["INPUT"] ^ fields["MASK"])
    elif family == "affine_mod97":
        response = str((fields["A"] * fields["X"] + fields["B"]) % rule["modulus"])
    elif family == "finite_state":
        state = fields["START"]
        for symbol in fields["SYMBOLS"]:
            state = rule["transitions"][state][symbol]
        response = state
    else:
        if fields["STATUS"] == "ok":
            response = rule["policy"]["ok"]
        elif fields["STATUS"] == "transient" and fields["ATTEMPT"] < fields["LIMIT"]:
            response = rule["policy"]["transient_below_limit"]
        else:
            response = rule["policy"]["otherwise"]
    return {"action": "answer", "response": response}


def solve_visible(family, prompt, visible_event_texts):
    """Solve an explicitly unauthenticated visible-evidence control condition.

    The caller supplies already selected event texts. This function authenticates
    neither their source/ownership nor chronology/permissions, and returns no
    source receipt. Native release labels must use solve_episode instead.
    """
    policy = _question_policy(family, prompt)
    if not isinstance(visible_event_texts, (list, tuple)) or not all(
            isinstance(text, str) for text in visible_event_texts):
        raise ValueError("visible evidence must be a list of observation texts")
    fields = {}
    for text in visible_event_texts:
        name, value = _read_observation(family, text)
        if name in fields:
            raise ValueError("multiple active observations for one required field")
        fields[name] = value
    return _solve_fields(family, fields, policy)


def solve_episode(payload, release_root, source_record):
    """Recompute the exact target from bound raw observations, never labels."""
    payload, header, events = _bound_source(payload, release_root, source_record)
    family, cutoff = header["family"], payload["query"]["at"]
    missing_policy = _question_policy(family, payload["query"]["prompt"], cutoff)
    active = {}
    # Validate every observed source expression, then independently replay the
    # cutoff. An old event_time does not reveal a later observation early.
    parsed = {event["event_id"]: _read_observation(family, event["text"])
              for event in events if event["channel"] == "observed" and event["operation"] != "tombstone"}
    for event in events:
        if (event["channel"] != "observed" or event["observed_at"] > cutoff
                or event["event_time"] > cutoff):
            continue
        if event["replaces_event_id"] is not None:
            active.pop(event["replaces_event_id"], None)
        if event["operation"] != "tombstone":
            active[event["event_id"]] = parsed[event["event_id"]]
    fields = {}
    for name, value in active.values():
        if name in fields:
            raise ValueError("multiple active observations for one required field")
        fields[name] = value
    return {**_solve_fields(family, fields, missing_policy), "support_event_ids": list(active)}


def independent_check(payload, release_root, source_record):
    """Compare a claimed target with the independently reconstructed result."""
    expected = solve_episode(payload, release_root, source_record)
    try:
        claimed = validate_episode(payload).to_dict()["target"]
    except ValueError:
        return False
    return claimed == expected
