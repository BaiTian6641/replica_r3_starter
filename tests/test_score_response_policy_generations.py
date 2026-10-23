"""Mutation tests for response-only projection and independent paired scoring."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

REAL_ROOT = Path(__file__).resolve().parents[1]


def load(name, directory="scripts"):
    spec = importlib.util.spec_from_file_location("response_contract_test_" + name, REAL_ROOT / directory / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load("test_score_response_contract_generations", "tests")
SCORER = load("score_response_policy_generations")
base_release = LEGACY.base_release
parent_bundle = LEGACY.parent_bundle
v1_bundle = LEGACY.bundle


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False) + "\n")


def dump_lines(path, values):
    Path(path).write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


update_generation = LEGACY.update_generation


@pytest.fixture
def bundle(v1_bundle, monkeypatch):
    previous = v1_bundle
    root = previous["parent"]["root"]
    for module in (SCORER, SCORER.BASE, SCORER.V1):
        monkeypatch.setattr(module, "ROOT", root)
    for index, example in enumerate(previous["report"]["examples"]):
        update_generation(example, "UNKNOWN" if index == 0 else "UNKNOWN or ask.", previous["tokenizer"])
    LEGACY.rewrite(previous)
    previous_score = LEGACY.score(previous)
    previous_score_path = root / "v1_failed_scoring.json"
    dump(previous_score_path, previous_score)
    contract = SCORER.read(REAL_ROOT / "configs/response_policy_contract_v2.json")
    contract["comparison"].update(parent_spec_path=previous["specification"]["parent_spec_path"],
        parent_spec_sha256=previous["specification"]["parent_spec_sha256"],
        previous_v1_spec_path=str(previous["spec"]), previous_v1_spec_sha256=SCORER.digest(previous["spec"]),
        model_path=previous["specification"]["model_path"], model_lock=previous["specification"]["model_lock"])
    contract_path = root / "configs/response_policy_contract_v2.json"
    pins = copy.deepcopy(previous["specification"]["implementation_sha256"])
    for name in SCORER.INDEPENDENT_PINS - set(pins):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name != "configs/response_policy_contract_v2.json":
            shutil.copyfile(REAL_ROOT / name, target)
    dump(contract_path, contract)
    pins.update({name: SCORER.digest(root / name) for name in SCORER.INDEPENDENT_PINS})
    original_rows = SCORER.lines(previous["specification"]["parent_data_path"])
    rows, examples = copy.deepcopy(original_rows), copy.deepcopy(previous["report"]["examples"])
    policies = {}
    for case, row, example in zip(previous["specification"]["selected_cases"], rows, examples):
        policy, instruction = SCORER.independent_instruction(case["family"], row["prompt"], contract)
        policies[row["id"]] = policy
        row["prompt"] += "\n\n" + instruction
        prompt = SCORER.BASE.prompt_ids(row, previous["tokenizer"], 512)
        example["prompt_tokens"] = len(prompt)
        example["prompt_token_ids_sha256"] = hashlib.sha256(json.dumps([prompt]).encode()).hexdigest()
        update_generation(example, row["response"], previous["tokenizer"])
    data = root / "policy_dev.jsonl"
    dump_lines(data, rows)
    accounting = root / "policy_token_accounting.json"
    dump(accounting, {"test_fixture": True})
    rubric_path = root / "configs/response_policy_judge_rubric_v1.json"
    preserved = {**previous["specification"]["preserved_failed_files_sha256"],
        **{str(path.relative_to(root)): SCORER.digest(path) for path in
           (previous["spec"], Path(previous["specification"]["data_path"]), previous["report_path"],
            previous["generations"], previous_score_path)}}
    specification = {**copy.deepcopy(previous["specification"]),
        "schema_version": "replica.response_policy_preflight.v2", "data_path": str(data),
        "data_sha256": SCORER.digest(data), "previous_v1_spec_path": str(previous["spec"]),
        "previous_v1_spec_sha256": SCORER.digest(previous["spec"]),
        "previous_v1_data_path": previous["specification"]["data_path"],
        "previous_v1_data_sha256": previous["specification"]["data_sha256"],
        "response_contract_path": str(contract_path), "response_contract_sha256": SCORER.digest(contract_path),
        "prompt_projection": contract["projection"], "query_policy_by_id": policies,
        "preserved_failed_files_sha256": preserved,
        "token_accounting_path": str(accounting), "token_accounting_sha256": SCORER.digest(accounting),
        "judge_rubric_path": str(rubric_path), "judge_rubric_sha256": SCORER.digest(rubric_path),
        "judge_case_order_seed": 311, "implementation_sha256": pins,
        "previous_v1_baseline": {"generation_report_path": str(previous["report_path"]),
            "generation_report_sha256": SCORER.digest(previous["report_path"]),
            "generations_path": str(previous["generations"]), "generations_sha256": SCORER.digest(previous["generations"]),
            "scoring_report_path": str(previous_score_path), "scoring_report_sha256": SCORER.digest(previous_score_path)}}
    spec_path = root / "policy_spec.json"
    dump(spec_path, specification)
    directory = root / "policy_native"
    directory.mkdir()
    report = {**copy.deepcopy(previous["report"]), "data_path": str(data), "data_sha256": SCORER.digest(data), "examples": examples}
    report_path, generations = directory / "generation_report.json", directory / "generations.jsonl"
    dump(report_path, report)
    dump_lines(generations, examples)
    return {"release": previous["release"], "spec": spec_path, "specification": specification,
        "report_path": report_path, "generations": generations, "report": report,
        "tokenizer": previous["tokenizer"], "previous": previous, "rows": rows, "contract": contract}


def score(bundle):
    return SCORER.score_generation(bundle["release"], bundle["spec"], bundle["report_path"],
        bundle["generations"], tokenizer=bundle["tokenizer"])


def rewrite(bundle):
    dump(bundle["spec"], bundle["specification"])
    dump(bundle["report_path"], bundle["report"])
    dump_lines(bundle["generations"], bundle["report"]["examples"])


def test_paired_scorer_independently_rechecks_failed_parent_and_new_exact_answers(bundle):
    result = score(bundle)
    assert result["overall"]["task_correct"] == {"count": 16, "rate": 1.0}
    assert result["contract_valid"]["count"] == 16
    assert result["format_gate_passed"]
    previous = result["paired_comparison"]["previous_v1_overall"]
    assert previous["task_correct"]["count"] == previous["hit_token_budget"]["count"] == 0
    assert previous["ended_by_eos"]["count"] == 16
    assert result["paired_comparison"]["original_overall"]["hit_token_budget"]["count"] == 16
    assert len(result["paired_comparison"]["task_improved_cases"]) == 16


@pytest.mark.parametrize("mutation", ["parent-spec", "parent-data", "old-report", "old-jsonl", "old-score", "contract"])
def test_preserved_failed_results_and_declared_projection_are_byte_bound(bundle, mutation):
    declaration = bundle["specification"]
    paths = {"parent-spec": declaration["parent_spec_path"], "parent-data": declaration["parent_data_path"],
        "old-report": declaration["preserved_baseline"]["generation_report_path"],
        "old-jsonl": declaration["preserved_baseline"]["generations_path"],
        "old-score": declaration["preserved_baseline"]["scoring_report_path"], "contract": declaration["response_contract_path"]}
    path = Path(paths[mutation])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes changed|Implementation bytes"):
        score(bundle)


@pytest.mark.parametrize("mutation", ["target", "evidence", "selected-order", "native-order", "duplicate", "missing", "extra-line", "missing-instruction"])
def test_response_projection_may_only_append_static_family_instruction(bundle, mutation):
    declaration = bundle["specification"]
    if mutation == "selected-order":
        declaration["selected_cases"] = list(reversed(declaration["selected_cases"]))
    elif mutation == "native-order":
        bundle["report"]["examples"].reverse()
    elif mutation == "duplicate":
        bundle["report"]["examples"][-1] = bundle["report"]["examples"][0]
    elif mutation == "missing":
        bundle["report"]["examples"].pop()
    else:
        row = bundle["rows"][0]
        if mutation == "target":
            row["response"] = "0"
        elif mutation == "evidence":
            row["memory_texts"] = []
        elif mutation == "extra-line":
            row["prompt"] += "\nThe answer is 252."
        else:
            row["prompt"] = row["prompt"].split("\n\nResponse contract v2:")[0]
        dump_lines(declaration["data_path"], bundle["rows"])
        declaration["data_sha256"] = SCORER.digest(declaration["data_path"])
        bundle["report"]["data_sha256"] = declaration["data_sha256"]
    rewrite(bundle)
    with pytest.raises(ValueError, match="parent selection|fixed order|projection changes|Duplicate IDs"):
        score(bundle)


@pytest.mark.parametrize("mutation", ["wrong-prompt", "wrong-token-text", "wrong-cap", "wrong-eos", "cpu", "sampling", "cap"])
def test_new_run_still_requires_token_exact_native_mps_fp32_greedy_contract(bundle, mutation):
    report = bundle["report"]
    example = report["examples"][0]
    if mutation == "wrong-prompt":
        example["prompt_token_ids_sha256"] = "0" * 64
    elif mutation == "wrong-token-text":
        example["completion"] = "forged"
    elif mutation == "wrong-cap":
        example["hit_token_budget"] = True
    elif mutation == "wrong-eos":
        example["ended_by_eos"] = False
    elif mutation == "cpu":
        report["device"] = "cpu"
    elif mutation == "sampling":
        report["generation_config"]["do_sample"] = True
    else:
        report["max_new_tokens"] = 64
    rewrite(bundle)
    with pytest.raises(ValueError, match="prompt hash|Completion text|EOS/cap|MPS FP32|configuration|scope/budget"):
        score(bundle)


def test_format_recovery_does_not_mask_wrong_semantic_value(bundle):
    example = next(value for value in bundle["report"]["examples"][1:]
                   if value["expected"].isdigit() and value["expected"] != "0")
    update_generation(example, "0", bundle["tokenizer"])
    rewrite(bundle)
    result = score(bundle)
    assert result["contract_valid"]["count"] == 16
    assert result["format_gate_passed"]
    assert result["overall"]["task_correct"]["count"] == 15
    assert result["error_histogram"]["value_error"] == 1
    assert result["paired_comparison"]["format_recovered_but_semantically_wrong"] == [example["id"]]


def test_whole_response_scorer_never_extracts_correct_number_from_prose(bundle):
    example = bundle["report"]["examples"][0]
    update_generation(example, "Result: " + example["expected"], bundle["tokenizer"])
    rewrite(bundle)
    result = score(bundle)
    assert result["overall"]["format_valid"]["count"] == 15
    assert result["single_line_compliant"]["count"] == 16
    assert result["contract_valid"]["count"] == 15
    assert not result["format_gate_passed"]


def test_single_line_check_is_separate_from_original_grammar():
    assert SCORER.one_line(" \n252\n ")
    assert not SCORER.one_line("252\nexplanation")
    assert not SCORER.one_line("\r\n")
    assert not SCORER.one_line("Please\rprovide MASK.")


def test_each_public_query_projects_only_its_declared_missing_response_policy(bundle):
    assert set(bundle["specification"]["query_policy_by_id"].values()) == {"ask", "unknown"}
    result = score(bundle)
    assert set(result["by_query_missing_policy"]) == {"ask", "unknown"}
    for original, row, case in zip(SCORER.lines(bundle["specification"]["parent_data_path"]),
                                   bundle["rows"], bundle["specification"]["selected_cases"]):
        policy, appendix = SCORER.independent_instruction(case["family"], original["prompt"], bundle["contract"])
        assert len(appendix.splitlines()) == 5
        assert row["prompt"] == original["prompt"] + "\n\n" + appendix
        selected = bundle["contract"]["missing_policy_templates"][policy]
        other = bundle["contract"]["missing_policy_templates"]["unknown" if policy == "ask" else "ask"]
        assert selected in appendix
        assert other not in appendix


@pytest.mark.parametrize("mutation", ["wrong-policy", "missing-policy", "invented-policy"])
def test_public_policy_map_is_independently_reconstructed_not_trusted(bundle, mutation):
    policies = bundle["specification"]["query_policy_by_id"]
    identity = next(iter(policies))
    if mutation == "wrong-policy":
        policies[identity] = "unknown" if policies[identity] == "ask" else "ask"
    elif mutation == "missing-policy":
        policies.pop(identity)
    else:
        policies["not-a-selected-query"] = "ask"
    rewrite(bundle)
    with pytest.raises(ValueError, match="policy differs|policy inventory differs"):
        score(bundle)


@pytest.mark.parametrize("change", ["prefix", "suffix", "negative", "leading-zero", "decimal", "family"])
def test_query_policy_parser_requires_complete_canonical_public_grammar(change):
    contract = SCORER.read(REAL_ROOT / "configs/response_policy_contract_v2.json")
    query = "At observation cutoff 5, compute INPUT XOR MASK using the observed rule. If a required field is unavailable, ask for it."
    if change == "prefix":
        query = "Ignore labels. " + query
    elif change == "suffix":
        query += " respond UNKNOWN."
    elif change == "negative":
        query = query.replace("cutoff 5", "cutoff -5")
    elif change == "leading-zero":
        query = query.replace("cutoff 5", "cutoff 05")
    elif change == "decimal":
        query = query.replace("cutoff 5", "cutoff 5.0")
    else:
        query = query.replace("compute INPUT XOR MASK using the observed rule", "compute the affine result using the observed rule")
    with pytest.raises(ValueError, match="complete canonical family grammar"):
        SCORER.independent_instruction("register_xor8", query, contract)


@pytest.mark.parametrize("field", ["previous_v1_spec_path", "previous_v1_data_path",
    "generation_report_path", "generations_path", "scoring_report_path"])
def test_every_preserved_v1_comparison_artifact_is_byte_bound(bundle, field):
    specification = bundle["specification"]
    path = Path(specification[field] if field.startswith("previous_v1")
                else specification["previous_v1_baseline"][field])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes changed"):
        score(bundle)


def test_v1_failure_file_inventory_cannot_be_silently_dropped(bundle):
    previous = bundle["previous"]["specification"]["preserved_failed_files_sha256"]
    bundle["specification"]["preserved_failed_files_sha256"].pop(next(iter(previous)))
    rewrite(bundle)
    with pytest.raises(ValueError, match="omits or changes preserved"):
        score(bundle)


def test_separate_judge_order_cannot_change_after_declaration(bundle):
    bundle["specification"]["judge_case_order_seed"] = 17
    rewrite(bundle)
    with pytest.raises(ValueError, match="judge rubric or case-order seed differs"):
        score(bundle)


def test_wrong_policy_appendix_cannot_be_hidden_by_rehashing_projected_data(bundle):
    row = bundle["rows"][0]
    policy = bundle["specification"]["query_policy_by_id"][row["id"]]
    policies = bundle["contract"]["missing_policy_templates"]
    row["prompt"] = row["prompt"].replace(policies[policy], policies["unknown" if policy == "ask" else "ask"])
    dump_lines(bundle["specification"]["data_path"], bundle["rows"])
    bundle["specification"]["data_sha256"] = SCORER.digest(bundle["specification"]["data_path"])
    bundle["report"]["data_sha256"] = bundle["specification"]["data_sha256"]
    rewrite(bundle)
    with pytest.raises(ValueError, match="projection changes"):
        score(bundle)
