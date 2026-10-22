"""Fictional solver-grounded episode boundaries; no model or external data."""
import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from replica_lab.contracts import validate_example
from replica_lab.episodes import (
    Episode, VerificationReceipt, audit_episode_families, export_sft, input_view,
    validate_episode, verify_episode,
)


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ["local_retrieval", "local_training", "local_evaluation"]
HASH = hashlib.sha256(b"fictional-world-generator-v1").hexdigest()


def source(source_id="world-source", **changes):
    value = {"source_id": source_id, "group_id": "world-1", "split": "train", "sha256": HASH,
             "subject_id": "fictional-fixture", "branch_id": "source", "actor_id": "solver",
             "kind": "synthetic", "privacy": "public", "permissions": LOCAL.copy()}
    return {**value, **changes}


def event(event_id="event-1", **changes):
    value = {"event_id": event_id, "source_id": "world-source", "event_time": 1,
             "observed_at": 2, "operation": "observe", "replaces_event_id": None,
             "channel": "observed", "text": "The fictional packet transform XORs input with mask 1."}
    return {**value, **changes}


def episode():
    return {"schema_version": "replica.episode.v1", "id": "query-1", "group_id": "world-1",
            "split": "train", "subject_id": "fictional-fixture", "branch_id": "source",
            "parent_episode_id": None, "privacy": "public", "permissions": LOCAL.copy(),
            "query": {"at": 5, "prompt": "Apply the fictional packet transform to input 6."},
            "sources": [source()], "events": [event()],
            "target": {"action": "answer", "response": "7", "support_event_ids": ["event-1"]},
            "evaluator": {"oracle": {"input": 6, "mask": 1, "private_annotation": "ORACLE_SENTINEL"}}}


def xor_solver(payload):
    # Independent numeric computation; never calls the exporter or trusts its pass flag.
    oracle = payload["evaluator"]["oracle"]
    return payload["target"]["response"] == str(oracle["input"] ^ oracle["mask"])


def receipt(payload):
    return verify_episode(payload, verifier_id="test.xor-v1", verifier_sha256=HASH,
                          verifier=xor_solver)


def test_solver_verified_exact_eight_field_export_and_schema():
    raw = episode()
    canonical = Episode.from_dict(raw)
    row = export_sft(canonical, verification=receipt(canonical))
    assert set(row) == {"id", "group_id", "split", "prompt", "response", "memory_texts", "source_ids", "verified"}
    assert row["response"] == "7" and row["verified"] is True
    assert row["source_ids"] == ["world-source"]
    assert "event_time=1 | observed_at=2" in row["memory_texts"][0]
    assert "kind=synthetic" in row["memory_texts"][0]
    validate_example(row, require_verified=True)
    jsonschema.validate(raw, json.loads((ROOT / "schemas/episodes/episode.schema.json").read_text()))
    jsonschema.validate(row, json.loads((ROOT / "schemas/sft_example.schema.json").read_text()))


def test_input_view_excludes_target_and_explicit_oracle_but_keeps_answer_fact():
    raw = episode()
    raw["events"][0]["text"] = "The measured output was 7. This is an ordinary observed fact."
    raw["events"].append(event("event-oracle", observed_at=3, channel="evaluator_only",
                               text="ANNOTATION_ONLY_GOLD"))
    view = input_view(raw)
    rendered = json.dumps(view)
    assert "measured output was 7" in rendered  # Natural answer-bearing evidence is permitted.
    assert "ORACLE_SENTINEL" not in rendered and "ANNOTATION_ONLY_GOLD" not in rendered
    assert set(view) == {"prompt", "memory_texts", "source_ids"}
    assert "response" not in view and "evaluator" not in view


def test_future_observation_stays_hidden_even_when_event_time_is_old():
    raw = episode()
    raw["events"].append(event("late-report", event_time=1, observed_at=10, text="FUTURE_REPORT"))
    assert "FUTURE_REPORT" not in json.dumps(input_view(raw))
    raw["target"]["support_event_ids"] = ["late-report"]
    with pytest.raises(ValueError, match="cutoff"):
        validate_episode(raw)


def test_late_correction_is_available_at_observation_time_not_event_time():
    raw = episode()
    raw["events"].append(event("event-2", event_time=2, observed_at=8, operation="correct",
                               replaces_event_id="event-1", text="The corrected mask is 3."))
    assert "mask 1" in input_view(raw)["memory_texts"][0]
    raw["query"]["at"] = 8
    raw["target"]["support_event_ids"] = ["event-2"]
    view = input_view(raw)
    assert len(view["memory_texts"]) == 1 and "corrected mask is 3" in view["memory_texts"][0]
    raw["target"]["support_event_ids"] = ["event-1"]
    with pytest.raises(ValueError, match="active observed"):
        validate_episode(raw)


def test_tombstone_removes_fact_without_revealing_tombstone_text():
    raw = episode()
    raw["events"].append(event("event-2", observed_at=6, operation="tombstone",
                               replaces_event_id="event-1", text=""))
    assert input_view(raw)["source_ids"] == ["world-source"]
    raw["query"]["at"] = 7
    raw["target"] = {"action": "unknown", "response": "Unknown after deletion.", "support_event_ids": []}
    assert input_view(raw)["memory_texts"] == []
    assert input_view(raw)["source_ids"] == []


def test_ineligible_correction_never_resurrects_old_eligible_fact():
    raw = episode()
    raw["permissions"].append("external_teacher")
    raw["sources"][0]["permissions"].append("external_teacher")
    raw["privacy"] = "private"
    raw["sources"].append(source("correction-source", privacy="private"))
    raw["events"].append(event("event-2", source_id="correction-source", observed_at=4,
                               operation="correct", replaces_event_id="event-1", text="PRIVATE_CORRECTION"))
    raw["target"]["support_event_ids"] = ["event-2"]
    assert input_view(raw, purpose="external_teacher")["memory_texts"] == []
    with pytest.raises(ValueError, match="lacks permission"):
        export_sft(raw, purpose="external_teacher", require_verified=False)


@pytest.mark.parametrize("action,response", [("ask", "Please provide the missing mask."),
                                             ("unknown", "The mask is unknown.")])
def test_explicit_unknown_and_ask_can_have_no_evidence(action, response):
    raw = episode()
    raw["sources"], raw["events"] = [], []
    raw["target"] = {"action": action, "response": response, "support_event_ids": []}
    row = export_sft(raw, require_verified=False)
    assert row["memory_texts"] == [] and row["source_ids"] == [] and row["verified"] is False


def test_answer_without_declared_support_is_rejected():
    raw = episode()
    raw["target"]["support_event_ids"] = []
    with pytest.raises(ValueError, match="requires support"):
        validate_episode(raw)


@pytest.mark.parametrize("split", ["dev", "calibration", "promotion", "audit"])
def test_nontraining_roles_cannot_be_training_or_teacher_exports(split):
    raw = episode()
    raw["split"] = raw["sources"][0]["split"] = split
    raw["permissions"].append("external_teacher")
    for purpose in ["local_training", "external_teacher"]:
        with pytest.raises(ValueError, match="split=train|audit model input"):
            export_sft(raw, purpose=purpose, require_verified=False)
    assert export_sft(raw, purpose="local_evaluation", verification=receipt(raw))["split"] == split


def test_no_public_or_private_implicit_permission_and_private_label_preserved():
    raw = episode()
    with pytest.raises(ValueError, match="does not allow"):
        input_view(raw, purpose="external_teacher")
    raw["sources"][0]["privacy"] = "private"
    with pytest.raises(ValueError, match="remain private"):
        validate_episode(raw)
    raw["privacy"] = "private"
    raw["sources"][0]["permissions"].remove("local_training")
    with pytest.raises(ValueError, match="lacks permission"):
        export_sft(raw, require_verified=False)


def test_current_permission_revocation_blocks_even_older_query_exports():
    raw = episode()
    raw["sources"][0]["permissions"] = []
    assert input_view(raw)["memory_texts"] == []
    with pytest.raises(ValueError, match="lacks permission"):
        export_sft(raw, require_verified=False)


@pytest.mark.parametrize("changes", [
    {"kind": "human_source", "actor_id": "other-person"},
    {"kind": "agent_experience"}, {"kind": "inference"},
    {"subject_id": "other-subject"}, {"branch_id": "unrelated-agent"},
])
def test_invalid_source_or_agent_branch_attribution(changes):
    raw = episode()
    raw["sources"][0].update(changes)
    with pytest.raises(ValueError):
        validate_episode(raw)


def test_parallel_branch_may_read_source_but_cannot_correct_source_ownership():
    raw = episode()
    raw["branch_id"] = "replica-a"
    raw["sources"].append(source("agent", kind="agent_experience", actor_id="replica-agent",
                                  branch_id="replica-a"))
    raw["events"].append(event("event-2", source_id="agent", observed_at=4, operation="correct",
                               replaces_event_id="event-1", text="The agent claims a different source life."))
    with pytest.raises(ValueError, match="ownership"):
        validate_episode(raw)
    raw["events"][-1].update(operation="observe", replaces_event_id=None)
    view = input_view(raw)
    assert "branch=source" in view["memory_texts"][0]
    assert "kind=agent_experience" in view["memory_texts"][1]


def test_oracle_event_cannot_support_answer_or_supersede_observed_history():
    raw = episode()
    raw["events"].append(event("event-2", observed_at=3, channel="evaluator_only", text="GOLD"))
    raw["target"]["support_event_ids"] = ["event-2"]
    with pytest.raises(ValueError, match="never future/oracle"):
        validate_episode(raw)
    raw["target"]["support_event_ids"] = ["event-1"]
    raw["events"][-1].update(operation="correct", replaces_event_id="event-1")
    with pytest.raises(ValueError, match="same-channel"):
        validate_episode(raw)


@pytest.mark.parametrize("mutation", ["reversed", "dangling", "fork", "future_event", "bool_time", "extra_gold"])
def test_strict_timeline_and_no_extra_annotation_fields(mutation):
    raw = episode()
    if mutation == "reversed":
        raw["events"].insert(0, event("future", observed_at=9))
    elif mutation == "dangling":
        raw["events"][0].update(operation="correct", replaces_event_id="missing")
    elif mutation == "fork":
        raw["events"].extend([event("event-2", observed_at=3, operation="correct", replaces_event_id="event-1"),
                              event("event-3", observed_at=4, operation="correct", replaces_event_id="event-1")])
    elif mutation == "future_event":
        raw["events"][0]["event_time"] = 3
    elif mutation == "bool_time":
        raw["events"][0]["observed_at"] = True
    else:
        raw["events"][0]["gold_answer"] = "7"
    with pytest.raises(ValueError):
        validate_episode(raw)


def test_verified_flag_requires_bound_external_receipt_and_numeric_solver():
    raw = episode()
    with pytest.raises(ValueError, match="quarantined"):
        export_sft(raw)
    raw["target"]["response"] = "0"
    failed = receipt(raw)
    assert failed.passed is False
    with pytest.raises(ValueError, match="failed"):
        export_sft(raw, verification=failed)
    assert export_sft(raw, verification=failed, require_verified=False)["verified"] is False
    raw["target"]["response"] = "7"
    correct = receipt(raw)
    raw["query"]["prompt"] += " Changed prompt."
    with pytest.raises(ValueError, match="different episode bytes"):
        export_sft(raw, verification=correct)


def test_verification_cannot_be_self_declared_or_truthy_string():
    raw = episode()
    raw["verified"] = True
    with pytest.raises(ValueError, match="fields"):
        validate_episode(raw)
    with pytest.raises(ValueError, match="boolean"):
        verify_episode(episode(), verifier_id="bad", verifier_sha256=HASH, verifier=lambda _: "true")
    with pytest.raises(ValueError, match="host VerificationReceipt"):
        export_sft(episode(), verification={"passed": True})


def test_immutable_snapshot_and_deterministic_detached_export():
    raw = episode()
    original = copy.deepcopy(raw)
    canonical = validate_episode(raw)
    raw["events"][0]["text"] = "mutation"
    detached = canonical.to_dict()
    detached["target"]["response"] = "wrong"
    assert canonical.to_dict() == original
    checked = receipt(canonical)
    first = export_sft(canonical, verification=checked)
    first["memory_texts"].clear()
    assert export_sft(canonical, verification=checked)["memory_texts"]
    reversed_keys = dict(reversed(list(original.items())))
    assert validate_episode(reversed_keys).sha256 == canonical.sha256


def test_family_parent_source_inheritance_and_cycle_audit():
    parent, child = episode(), episode()
    child["id"], child["parent_episode_id"] = "child", "query-1"
    assert audit_episode_families([child, parent])["episodes"] == 2
    child["group_id"] = child["sources"][0]["group_id"] = "other-family"
    child["sources"][0]["source_id"] = child["events"][0]["source_id"] = "other-source"
    child["sources"][0]["sha256"] = hashlib.sha256(b"different synthetic source").hexdigest()
    with pytest.raises(ValueError, match="inherit parent"):
        audit_episode_families([parent, child])
    child = copy.deepcopy(parent)
    child["id"], child["parent_episode_id"] = "child", "query-1"
    parent["parent_episode_id"] = "child"
    with pytest.raises(ValueError, match="cycle"):
        audit_episode_families([parent, child])


def test_same_family_or_source_cannot_move_to_different_split():
    first, second = episode(), episode()
    second["id"] = "second"
    second["split"] = second["sources"][0]["split"] = "audit"
    with pytest.raises(ValueError, match="family crosses"):
        audit_episode_families([first, second])
    second["group_id"] = second["sources"][0]["group_id"] = "different-family"
    with pytest.raises(ValueError, match="source identity"):
        audit_episode_families([first, second])


def test_descendant_parent_must_be_present_and_source_binding_cannot_drift():
    raw = episode()
    raw["parent_episode_id"] = "missing-parent"
    with pytest.raises(ValueError, match="parent missing"):
        audit_episode_families([raw])
    raw["sources"][0]["split"] = "dev"
    with pytest.raises(ValueError, match="inherit episode"):
        validate_episode(raw)


def test_schema_count_and_reject_non_json_or_unknown_fields():
    assert len(list((ROOT / "schemas").glob("*.schema.json"))) == 3
    for key, value in [("evaluator", {"oracle": {"bad": float("nan")}}), ("split", []),
                       ("permissions", ["local_training", "local_training"])]:
        raw = episode()
        raw[key] = value
        with pytest.raises(ValueError):
            validate_episode(raw)


@pytest.mark.parametrize("purpose", ["local_retrieval", "sharing"])
def test_target_bearing_export_cannot_use_retrieval_or_sharing_permission(purpose):
    raw = episode()
    raw["permissions"] = raw["sources"][0]["permissions"] = [purpose]
    assert input_view(raw, purpose=purpose)["memory_texts"]
    with pytest.raises(ValueError, match="target-bearing export purpose"):
        export_sft(raw, purpose=purpose, verification=receipt(raw))
    with pytest.raises(ValueError, match="target-bearing export purpose"):
        export_sft(raw, purpose=purpose, require_verified=False)


@pytest.mark.parametrize("purpose", ["local_retrieval", "sharing", "local_training", "external_teacher"])
def test_audit_input_requires_explicit_evaluation_even_if_permissions_claim_other_use(purpose):
    raw = episode()
    raw["split"] = raw["sources"][0]["split"] = "audit"
    raw["permissions"] = raw["sources"][0]["permissions"] = [purpose, "local_evaluation"]
    with pytest.raises(ValueError, match="audit model input requires explicit local_evaluation"):
        input_view(raw, purpose=purpose)
    if purpose == "local_retrieval":
        with pytest.raises(ValueError, match="audit model input"):
            input_view(raw)  # The convenient default must also fail closed.
    assert input_view(raw, purpose="local_evaluation")["memory_texts"]


@pytest.mark.parametrize("changed_split", [False, True])
def test_renamed_source_ids_cannot_move_identical_declared_source_bytes_to_new_family(changed_split):
    first, second = episode(), episode()
    second["id"], second["group_id"] = "renamed-query", "renamed-family"
    second["sources"][0].update(source_id="renamed-source", group_id="renamed-family")
    second["events"][0].update(source_id="renamed-source", event_id="renamed-event")
    second["target"]["support_event_ids"] = ["renamed-event"]
    if changed_split:
        second["split"] = second["sources"][0]["split"] = "audit"
    with pytest.raises(ValueError, match="declared source bytes must retain"):
        audit_episode_families([first, second])


@pytest.mark.parametrize("change", [{"text": "The same event now claims mask 9."},
                                    {"observed_at": 3}, {"channel": "evaluator_only"}])
def test_descendant_cannot_rewrite_existing_event_bytes(change):
    parent, child = episode(), episode()
    child["id"], child["parent_episode_id"] = "child", parent["id"]
    child["events"][0].update(change)
    if change.get("channel") == "evaluator_only":
        child["target"] = {"action": "unknown", "response": "Unknown.", "support_event_ids": []}
    with pytest.raises(ValueError, match="source/event bytes are immutable"):
        audit_episode_families([parent, child])


def test_descendant_new_correction_retains_original_event_bytes_and_lineage():
    parent, child = episode(), episode()
    child["id"], child["parent_episode_id"] = "child", parent["id"]
    child["events"].append(event("correction-2", event_time=2, observed_at=4, operation="correct",
                                  replaces_event_id="event-1", text="Corrected mask 3."))
    child["target"].update(response="5", support_event_ids=["correction-2"])
    assert audit_episode_families([parent, child])["episodes"] == 2
    assert "Corrected mask 3" in input_view(child)["memory_texts"][0]
