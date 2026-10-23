"""Independent hand-built source worlds, not generator-label round trips."""
import copy
import hashlib
import json

import pytest

from replica_lab.contracts import canonical_json
from replica_lab.episodes import export_sft, input_view, verify_episode
from replica_lab.infrastructure_solver import DESIGN_PATH, independent_check, solve_episode, solve_visible


DESIGN = json.loads(DESIGN_PATH.read_text())


def encoded(value):
    return (canonical_json(value) + "\n").encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def world(tmp_path, family="register_xor8", *, scenario="late_correction", policy="unknown", cutoff=5):
    """Build source bytes directly with fixed, human-computable operands."""
    spec = DESIGN["families"][family]
    source_id = "synthetic://infrastructure-v2/seed41/test-world"
    header = {"schema_version": DESIGN["source_header_schema"], "synthetic": True,
              "world_id": "infrastructure-v2-seed41-test-world", "group_id": "world-group",
              "split": "train", "family": family, "scenario": scenario, "source_id": source_id,
              "subject_id": "synthetic-subject", "branch_id": "source", "actor_id": "synthetic-generator",
              "privacy": "public", "permissions": DESIGN["permissions"]["train"], "seed": 41}
    rule = copy.deepcopy(spec["rule"])
    values = {"register_xor8": {"INPUT": 10, "MASK": 3},
              "affine_mod97": {"A": 3, "X": 8, "B": 5},
              "finite_state": {"START": "S0", "SYMBOLS": "ab"},
              "tool_recovery": {"STATUS": "transient", "ATTEMPT": 1, "LIMIT": 3}}[family]
    if family == "finite_state":
        rule["transitions"] = {f"S{i}": {"a": f"S{(i + 1) % 4}", "b": f"S{(i + 2) % 4}"}
                               for i in range(4)}

    def event(event_id, text, event_time, observed_at, operation="observe", replaces=None):
        return {"event_id": event_id, "source_id": source_id, "event_time": event_time,
                "observed_at": observed_at, "operation": operation, "replaces_event_id": replaces,
                "channel": "observed", "text": text}

    events = [event("world.00-rule", "RULE " + canonical_json(rule), 0, 0)]
    decisive = spec["decisive_field"]
    for index, name in enumerate(spec["fact_fields"]):
        if name != decisive:
            events.append(event(f"world.{10 + index:02}-{name}",
                "FACT " + canonical_json({"name": name, "value": values[name]}), 1, 1))
    late = scenario in {"late_observation_ask", "late_observation_unknown"}
    original_id = "world.80-decisive"
    events.append(event(original_id, "FACT " + canonical_json({"name": decisive, "value": values[decisive]}),
                        2, 8 if late else 2))
    if scenario == "late_correction":
        corrected = {"register_xor8": 2, "affine_mod97": 6,
                     "finite_state": "S1", "tool_recovery": "ok"}[family]
        events.append(event("world.90-correct", "FACT " + canonical_json({"name": decisive, "value": corrected}),
                            3, 8, "correct", original_id))
    elif scenario == "tombstone":
        events.append(event("world.99-delete", "", 9, 10, "tombstone", original_id))
    events.sort(key=lambda item: (item["observed_at"], item["event_id"]))
    episode = {"schema_version": "replica.episode.v1", "id": "query-0", "group_id": header["group_id"],
               "split": "train", "subject_id": header["subject_id"], "branch_id": "source",
               "parent_episode_id": None, "privacy": "public", "permissions": header["permissions"],
               "query": {"at": cutoff, "prompt": DESIGN["query_template"].format(
                   cutoff=cutoff, query_body=spec["query_body"], missing_instruction=DESIGN["missing_instructions"][policy])},
               "sources": [{key: header[key] for key in ("source_id", "group_id", "split", "subject_id",
                                                           "branch_id", "actor_id", "privacy", "permissions")}],
               "events": events, "target": {"action": "answer", "response": "placeholder",
                   "support_event_ids": ["world.00-rule"]}, "evaluator": {"oracle": {"invented_answer": "BAD"}}}
    episode["sources"][0]["kind"] = "synthetic"
    path = tmp_path / "raw_sources/test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"source_id": source_id, "world_id": header["world_id"], "group_id": header["group_id"],
              "split": "train", "family": family, "scenario": scenario, "path": "raw_sources/test.jsonl"}
    rebind(episode, header, record, path)
    episode["target"] = solve_episode(episode, tmp_path, record)
    return episode, header, record, path


def rebind(episode, header, record, path):
    chunks = [encoded(header)]
    spans, offset = [], len(chunks[0])
    for event in episode["events"]:
        chunk = encoded(event)
        spans.append({"event_id": event["event_id"], "byte_start": offset,
                      "byte_end": offset + len(chunk), "sha256": sha(chunk)})
        chunks.append(chunk)
        offset += len(chunk)
    raw = b"".join(chunks)
    path.write_bytes(raw)
    record.update(sha256=sha(raw), bytes=len(raw), event_spans=spans)
    episode["sources"][0]["sha256"] = sha(raw)


@pytest.mark.parametrize("family,before,after", [("register_xor8", "9", "8"),
    ("affine_mod97", "29", "30"), ("finite_state", "S3", "S0"),
    ("tool_recovery", "RETRY lookup", "VERIFY receipt")])
def test_solver_replays_each_mechanism_before_and_after_observed_correction(tmp_path, family, before, after):
    episode, _, source, _ = world(tmp_path, family)
    assert solve_episode(episode, tmp_path, source)["response"] == before
    assert independent_check(episode, tmp_path, source)
    episode["query"]["at"] = 9
    episode["query"]["prompt"] = episode["query"]["prompt"].replace("cutoff 5,", "cutoff 9,")
    # A future correction cannot be used at the old observation cutoff.
    assert not independent_check(episode, tmp_path, source)
    expected = solve_episode(episode, tmp_path, source)
    assert expected["response"] == after
    assert "world.80-decisive" not in expected["support_event_ids"]
    assert expected["support_event_ids"][-1] == "world.90-correct"


@pytest.mark.parametrize("scenario,policy,response,action", [
    ("late_observation_ask", "ask", "Please provide MASK.", "ask"),
    ("late_observation_unknown", "unknown", "UNKNOWN", "unknown"),
    ("tombstone", "ask", "Please provide MASK.", "ask")])
def test_absence_and_tombstone_follow_visible_missing_policy(tmp_path, scenario, policy, response, action):
    cutoff = 11 if scenario == "tombstone" else 5
    episode, _, source, _ = world(tmp_path, scenario=scenario, policy=policy, cutoff=cutoff)
    assert episode["target"]["response"] == response
    assert episode["target"]["action"] == action
    assert "world.80-decisive" not in episode["target"]["support_event_ids"]
    assert independent_check(episode, tmp_path, source)


def test_solver_does_not_read_claimed_target_or_oracle_and_current_revocation_is_separate(tmp_path):
    episode, _, source, _ = world(tmp_path)
    expected = solve_episode(episode, tmp_path, source)
    episode["target"]["response"] = "wrong-but-rehashed-label"
    episode["evaluator"]["oracle"] = {"gold": "wrong-but-rehashed-label", "passed": True}
    assert solve_episode(episode, tmp_path, source) == expected
    assert not independent_check(episode, tmp_path, source)
    episode["target"] = expected
    episode["sources"][0]["permissions"] = []
    assert independent_check(episode, tmp_path, source)
    assert input_view(episode)["memory_texts"] == []
    receipt = verify_episode(episode, verifier_id="independent-test", verifier_sha256="a" * 64,
        verifier=lambda payload: independent_check(payload, tmp_path, source))
    with pytest.raises(ValueError, match="lacks permission"):
        export_sft(episode, verification=receipt)


def test_source_bytes_span_event_binding_and_path_escape_fail_closed(tmp_path):
    episode, _, source, path = world(tmp_path)
    original = path.read_bytes()
    path.write_bytes(original.replace(b"INPUT", b"WRONG", 1))
    with pytest.raises(ValueError, match="bytes/hash"):
        solve_episode(episode, tmp_path, source)
    path.write_bytes(original)
    wrong_span = copy.deepcopy(source)
    wrong_span["event_spans"][0]["byte_start"] += 1
    with pytest.raises(ValueError, match="span mismatch"):
        solve_episode(episode, tmp_path, wrong_span)
    wrong_event = copy.deepcopy(episode)
    wrong_event["events"][1]["observed_at"] = 2
    with pytest.raises(ValueError, match="events differ"):
        solve_episode(wrong_event, tmp_path, source)
    with pytest.raises(ValueError, match="inside release"):
        solve_episode(episode, tmp_path, {**source, "path": "../outside.jsonl"})


def test_rehashed_rule_mutation_and_question_cutoff_mismatch_are_rejected(tmp_path):
    episode, header, source, path = world(tmp_path)
    episode["events"][0]["text"] = "RULE " + canonical_json({"operation": "xor", "width": 16})
    rebind(episode, header, source, path)
    with pytest.raises(ValueError, match="rule differs"):
        solve_episode(episode, tmp_path, source)
    episode, _, source, _ = world(tmp_path)
    episode["query"]["prompt"] = episode["query"]["prompt"].replace("cutoff 5", "cutoff 9")
    with pytest.raises(ValueError, match="question/family/cutoff"):
        solve_episode(episode, tmp_path, source)


def test_rehashed_late_observation_changes_truth_and_does_not_trust_old_label(tmp_path):
    episode, header, source, path = world(tmp_path)
    episode["events"][2]["observed_at"] = 8
    episode["events"].sort(key=lambda event: (event["observed_at"], event["event_id"]))
    rebind(episode, header, source, path)
    assert solve_episode(episode, tmp_path, source)["action"] == "unknown"
    assert not independent_check(episode, tmp_path, source)


def test_source_header_identity_permission_escalation_and_unknown_fields_rejected(tmp_path):
    episode, header, source, path = world(tmp_path)
    with pytest.raises(ValueError, match="identity/lineage"):
        solve_episode(episode, tmp_path, {**source, "family": "finite_state"})
    episode["sources"][0]["permissions"] = [*header["permissions"], "external_teacher"]
    with pytest.raises(ValueError, match="escalates"):
        solve_episode(episode, tmp_path, source)
    episode["sources"][0]["permissions"] = header["permissions"]
    header["hidden_rule"] = "ignored gold is forbidden in headers"
    rebind(episode, header, source, path)
    with pytest.raises(ValueError, match="header fields"):
        solve_episode(episode, tmp_path, source)


def test_finite_state_rule_requires_per_symbol_permutation(tmp_path):
    episode, header, source, path = world(tmp_path, "finite_state")
    rule = json.loads(episode["events"][0]["text"][5:])
    rule["transitions"]["S0"]["a"] = "S0"
    episode["events"][0]["text"] = "RULE " + canonical_json(rule)
    rebind(episode, header, source, path)
    with pytest.raises(ValueError, match="permutations"):
        solve_episode(episode, tmp_path, source)


@pytest.mark.parametrize("status,attempt,limit,response", [("ok", 9, 1, "VERIFY receipt"),
    ("transient", 2, 3, "RETRY lookup"), ("transient", 3, 3, "STOP"), ("permanent", 0, 10, "STOP")])
def test_recovery_policy_is_static_and_uses_strict_limit_boundary(tmp_path, status, attempt, limit, response):
    episode, header, source, path = world(tmp_path, "tool_recovery")
    values = {"STATUS": status, "ATTEMPT": attempt, "LIMIT": limit}
    for event in episode["events"]:
        if event["text"].startswith("FACT "):
            fact = json.loads(event["text"][5:])
            event["text"] = "FACT " + canonical_json({"name": fact["name"], "value": values[fact["name"]]})
    rebind(episode, header, source, path)
    assert solve_episode(episode, tmp_path, source)["response"] == response


def test_visible_controls_recompute_missing_fields_without_claiming_source_authentication(tmp_path):
    episode, _, _, _ = world(tmp_path, policy="ask")
    texts = [event["text"] for event in episode["events"] if event["observed_at"] <= 5]
    assert solve_visible("register_xor8", episode["query"]["prompt"], texts) == {
        "action": "answer", "response": "9"}
    missing = solve_visible("register_xor8", episode["query"]["prompt"], texts[:1])
    assert missing == {"action": "ask", "response": "Please provide INPUT, MASK."}
    assert set(missing) == {"action", "response"}  # no receipt/provenance invented
    unknown_prompt = episode["query"]["prompt"].replace("ask for it", "respond UNKNOWN")
    assert solve_visible("register_xor8", unknown_prompt, []) == {"action": "unknown", "response": "UNKNOWN"}
    with pytest.raises(ValueError, match="multiple active"):
        solve_visible("register_xor8", episode["query"]["prompt"], texts[:1] * 2)
    with pytest.raises(ValueError, match="grammar"):
        solve_visible("register_xor8", episode["query"]["prompt"], ["untrusted prose"])
    with pytest.raises(ValueError, match="question/family/cutoff"):
        solve_visible("register_xor8", "guess anything", texts)
