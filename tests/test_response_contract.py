"""Prompt projection, parent binding and no-truncation lock tests, model-free."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from replica_lab.response_contract import (
    ANSWER_FORMATS, FIELD_ORDERS, PROJECTION_VERSION, load_contract, project_row,
    render_instruction, validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_response_contract_test", ROOT / "scripts/prepare_response_contract.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)
RELEASE = ROOT / "data/diagnostics/infrastructure_slice_v2"
PARENT = ROOT / "reports/infrastructure_preflight_v3_20261024/evaluation_spec_m3.json"
CONTRACT = ROOT / "configs/response_contract_v1_m3.json"
MODEL = ROOT / "checkpoints/minicpm5_source"
LOCK = ROOT / "configs/minicpm5_sealed_lock.local.json"


class FakeTokenizer:
    eos_token_id = 2

    def get_chat_template(self):
        # This small tracked archive is byte-identical to the sealed template;
        # unit tests do not need local model/tokenizer checkpoint payloads.
        return (ROOT / "reports/checkpoint_retirement_metadata_after_D2_20261022/"
                "minicpm5_D2_tiny_chunk01_20261022/checkpoint/base/chat_template.jinja").read_text()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return "<chat>" + messages[0]["content"] + "</chat>"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [7] * max(1, (len(text) + 11) // 12)


class OverlongTokenizer(FakeTokenizer):
    def encode(self, text, *, add_special_tokens):
        if text.startswith("<chat>") and "Response contract v1:" in text:
            return [7] * 513
        return super().encode(text, add_special_tokens=add_special_tokens)


def _prepare(output, tokenizer=None):
    return PREPARE.prepare(RELEASE, PARENT, CONTRACT, output, tokenizer or FakeTokenizer(), MODEL, LOCK)


@pytest.mark.parametrize("family", FIELD_ORDERS)
def test_renderer_is_exact_declared_family_only_instruction(family):
    contract = load_contract()
    actual = render_instruction(family, contract)
    assert actual == "\n".join((contract["common_prefix"],
        "If all required items are observed, output " + ANSWER_FORMATS[family],
        "If a required item is unavailable, follow the task: output exactly UNKNOWN, or Please provide <fields>. "
        "Replace <fields> with missing names only, ordered " + ", ".join(FIELD_ORDERS[family])
        + ", separated by comma-space."))
    assert "No reasoning" in actual and "Do not invent missing values." in actual
    assert actual.count("\n") == 2


def test_projection_changes_only_prompt_and_cannot_branch_on_targets_or_missingness():
    contract = load_contract()
    row = {"id": "unchanged", "group_id": "world", "split": "dev", "prompt": "query",
           "response": "252", "memory_texts": ["fact252"], "source_ids": ["source"], "verified": True}
    snapshot = copy.deepcopy(row)
    projected = project_row(row, family="register_xor8", contract=contract)
    assert row == snapshot
    assert projected["prompt"] == row["prompt"] + "\n\n" + render_instruction("register_xor8", contract)
    assert {key: value for key, value in projected.items() if key != "prompt"} == {
        key: value for key, value in row.items() if key != "prompt"}
    altered = {**copy.deepcopy(row), "response": "UNKNOWN", "memory_texts": [], "target": {"action": "unknown"}}
    alternate = project_row(altered, family="register_xor8", contract=contract)
    assert alternate["prompt"] == projected["prompt"]
    projected["memory_texts"].append("mutation")
    assert row["memory_texts"] == ["fact252"]


@pytest.mark.parametrize("family", ["unknown", "register_xor8.answer", 7, None])
def test_unknown_family_rejected(family):
    with pytest.raises(ValueError, match="Unknown response-contract family"):
        render_instruction(family, load_contract())


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(schema_version="future"),
    lambda c: c.update(status="posthoc"),
    lambda c: c.update(append_separator="\n"),
    lambda c: c.update(joiner=" "),
    lambda c: c.update(projection_inputs=["family", "expected_response"]),
    lambda c: c.update(answer_template="output {expected_response}"),
    lambda c: c.update(answer_template="output {answer_format!r}"),
    lambda c: c.update(answer_template="output {answer_format:>9}"),
    lambda c: c.update(missing_template="output {field_order} then {target}"),
    lambda c: c["family_formats"]["register_xor8"].update(field_order=["INPUT", "MASK", "RULE"]),
    lambda c: c["family_formats"]["finite_state"].update(answer_format="only S3"),
    lambda c: c["comparison"].update(max_new_tokens=64),
    lambda c: c["comparison"].update(cpu_fallback=True),
    lambda c: c["comparison"].update(native_jobs=True),
])
def test_contract_rejects_undeclared_grammar_inputs_or_native_settings(mutation):
    contract = load_contract()
    mutation(contract)
    with pytest.raises(ValueError):
        validate_contract(contract)


def test_prepare_binds_exact_parent_failed_evidence_and_changed_prompt_only(tmp_path):
    output = tmp_path / "new"
    result = _prepare(output)
    assert result["examples"] == 16 and result["preserved_failed_files"] == 9
    spec = PREPARE.read(output / "evaluation_spec.json")
    parent = PREPARE.read(PARENT)
    assert spec["schema_version"] == "replica.response_contract_preflight.v1"
    assert spec["selected_cases"] == parent["selected_cases"]
    assert spec["prompt_projection"] == PROJECTION_VERSION
    assert spec["parent_spec_sha256"] == PREPARE.digest(PARENT)
    assert spec["parent_data_sha256"] == parent["data_sha256"]
    assert spec["response_contract_sha256"] == PREPARE.digest(CONTRACT)
    assert spec["max_new_tokens"] == 32 and spec["max_length"] == 512 and spec["seed"] == 17
    assert spec["device"] == "mps" and spec["dtype"] == "float32" and spec["cpu_fallback"] is False
    assert spec["data_sha256"] == PREPARE.digest(output / "dev_cases.jsonl")
    assert set(parent["implementation_sha256"]).issubset(spec["implementation_sha256"])
    assert PREPARE.NEW_PINS.issubset(spec["implementation_sha256"])
    assert all(spec["implementation_sha256"][key] == value for key, value in parent["implementation_sha256"].items())
    old = PREPARE.lines(ROOT / parent["data_path"])
    new = PREPARE.lines(output / "dev_cases.jsonl")
    assert [row["id"] for row in new] == [row["id"] for row in old]
    for before, after, case in zip(old, new, spec["selected_cases"]):
        assert after == project_row(before, family=case["family"], contract=load_contract())
        assert spec["original_case_sha256"][before["id"]] == hashlib.sha256(PREPARE.canonical_json(before).encode()).hexdigest()
    for path, expected in spec["preserved_failed_files_sha256"].items():
        assert PREPARE.digest(ROOT / path) == expected
    baseline = spec["preserved_baseline"]
    for name in ("generation_report", "generations", "scoring_report", "guard"):
        assert PREPARE.digest(ROOT / baseline[name + "_path"]) == baseline[name + "_sha256"]
    assert "--max-new-tokens" in spec["generation_command"]
    assert spec["scoring_settings"]["unchanged"] is True
    assert spec["scoring_settings"]["task_correctness_separate"] is True
    accounting = PREPARE.read(output / "token_accounting.json")
    assert accounting["truncated_examples"] == 0 and accounting["overlong_ids"] == []
    assert accounting["new_max_full_tokens"] <= 512
    assert accounting["old_prompt_tokens"] < accounting["new_prompt_tokens"]
    assert all(record["old"]["source_writer"] == record["new"]["source_writer"]
               and record["old"]["supervised_answer_plus_eos"] == record["new"]["supervised_answer_plus_eos"]
               for record in accounting["examples_detail"])


def test_prepare_rejects_existing_directory_without_overwriting(tmp_path):
    output = tmp_path / "already"
    output.mkdir()
    sentinel = output / "preserved.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="fresh response-contract directory"):
        _prepare(output)
    assert sentinel.read_text() == "keep"


def test_prepare_rejects_overlong_instruction_without_partial_output(tmp_path):
    output = tmp_path / "overlong"
    with pytest.raises(ValueError, match="no truncation"):
        _prepare(output, OverlongTokenizer())
    assert not output.exists()


def test_prepare_rejects_tokenizer_template_change_before_output(tmp_path):
    class ChangedTemplate(FakeTokenizer):
        def get_chat_template(self):
            return "altered"
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="tokenizer template differs"):
        _prepare(output, ChangedTemplate())
    assert not output.exists()


@pytest.mark.parametrize("change", ["duplicate", "reorder", "response", "evidence", "selection"])
def test_parent_rejects_changed_cases_labels_or_source_rows(tmp_path, change):
    parent = PREPARE.read(PARENT)
    rows = PREPARE.lines(ROOT / parent["data_path"])
    if change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "response":
        rows[0]["response"] = "0"
    elif change == "evidence":
        rows[0]["memory_texts"] = []
    else:
        parent["selected_cases"][0], parent["selected_cases"][1] = parent["selected_cases"][1], parent["selected_cases"][0]
        rows[0], rows[1] = rows[1], rows[0]
    data = tmp_path / "data.jsonl"
    data.write_text("".join(PREPARE.canonical_json(row) + "\n" for row in rows))
    parent["data_path"], parent["data_sha256"] = str(data), PREPARE.digest(data)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="Duplicate|ordered|order/selection|canonical export"):
        PREPARE._parent(RELEASE, spec, MODEL, LOCK)


def test_parent_rejects_changed_frozen_backend_pin(tmp_path):
    parent = PREPARE.read(PARENT)
    parent["implementation_sha256"]["scripts/train_text.py"] = "0" * 64
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="implementation bytes changed"):
        PREPARE._parent(RELEASE, spec, MODEL, LOCK)


def test_parent_rejects_changed_migration_receipt():
    parent = PREPARE.read(PARENT)
    parent["m3_pin_migration"]["parent_spec_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="original parent pin differs"):
        PREPARE._migration_parent(parent, PARENT)
