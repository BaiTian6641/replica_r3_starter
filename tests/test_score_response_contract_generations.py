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


LEGACY = load("test_score_infrastructure_generations", "tests")
SCORER = load("score_response_contract_generations")
base_release = LEGACY.base_release
parent_bundle = LEGACY.bundle


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False) + "\n")


def dump_lines(path, values):
    Path(path).write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def update_generation(example, text, tokenizer, *, eos=True):
    ids = tokenizer.encode(text, add_special_tokens=False) + ([1] if eos else [])
    example.update(generated_token_ids=ids, generated_tokens=len(ids), ended_by_eos=eos,
                   hit_token_budget=not eos and len(ids) == 32,
                   raw_completion=tokenizer.decode(ids, skip_special_tokens=False), completion=text)


@pytest.fixture
def bundle(parent_bundle, monkeypatch):
    parent = parent_bundle
    root = parent["root"]
    monkeypatch.setattr(SCORER, "ROOT", root)
    monkeypatch.setattr(SCORER.BASE, "ROOT", root)
    # Retain the original failed native evidence separately from the new run.
    for value in parent["report"]["examples"]:
        update_generation(value, "x" * 32, parent["tokenizer"], eos=False)
    LEGACY.rewrite_report(parent)
    previous = LEGACY.score(parent)
    previous_path = root / "failed_scoring.json"
    dump(previous_path, previous)
    contract = SCORER.read(REAL_ROOT / "configs/response_contract_v1.json")
    contract["comparison"]["parent_spec_path"] = str(parent["spec"])
    contract["comparison"]["parent_spec_sha256"] = SCORER.digest(parent["spec"])
    contract_path = root / "configs/response_contract_v1.json"
    pins = copy.deepcopy(parent["specification"]["implementation_sha256"])
    for name in SCORER.INDEPENDENT_PINS - set(pins):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name != "configs/response_contract_v1.json":
            shutil.copyfile(REAL_ROOT / name, target)
    dump(contract_path, contract)
    pins.update({name: SCORER.digest(root / name) for name in SCORER.INDEPENDENT_PINS})
    rows = copy.deepcopy(SCORER.lines(parent["specification"]["data_path"]))
    examples = copy.deepcopy(parent["report"]["examples"])
    for case, row, example in zip(parent["specification"]["selected_cases"], rows, examples):
        row["prompt"] += "\n\n" + SCORER.independent_instruction(case["family"], contract)
        prompt = SCORER.BASE.prompt_ids(row, parent["tokenizer"], 512)
        example["prompt_tokens"] = len(prompt)
        example["prompt_token_ids_sha256"] = hashlib.sha256(json.dumps([prompt]).encode()).hexdigest()
        update_generation(example, row["response"], parent["tokenizer"])
    data = root / "contract_dev.jsonl"
    dump_lines(data, rows)
    accounting = root / "token_accounting.json"
    dump(accounting, {"test_fixture": True})
    old_cases = SCORER.lines(parent["specification"]["data_path"])
    specification = {**copy.deepcopy(parent["specification"]),
        "schema_version": "replica.response_contract_preflight.v1", "data_path": str(data),
        "data_sha256": SCORER.digest(data), "parent_spec_path": str(parent["spec"]),
        "parent_spec_sha256": SCORER.digest(parent["spec"]),
        "parent_data_path": parent["specification"]["data_path"],
        "parent_data_sha256": parent["specification"]["data_sha256"],
        "response_contract_path": str(contract_path), "response_contract_sha256": SCORER.digest(contract_path),
        "prompt_projection": contract["projection"], "model_lock_sha256": SCORER.digest(parent["specification"]["model_lock"]),
        "preserved_failed_files_sha256": {str(path.relative_to(root)): SCORER.digest(path)
            for path in (previous_path,)},
        "token_accounting_path": str(accounting), "token_accounting_sha256": SCORER.digest(accounting),
        "effective_chat_template_sha256": parent["report"]["effective_chat_template_sha256"],
        "original_case_sha256": {row["id"]: hashlib.sha256(SCORER.canonical_json(row).encode()).hexdigest() for row in old_cases},
        "implementation_sha256": pins,
        "preserved_baseline": {"generation_report_path": str(parent["report_path"]),
            "generation_report_sha256": SCORER.digest(parent["report_path"]),
            "generations_path": str(parent["generations"]), "generations_sha256": SCORER.digest(parent["generations"]),
            "scoring_report_path": str(previous_path), "scoring_report_sha256": SCORER.digest(previous_path)}}
    spec_path = root / "contract_spec.json"
    dump(spec_path, specification)
    directory = root / "contract_native"
    directory.mkdir()
    report = {**copy.deepcopy(parent["report"]), "data_path": str(data), "data_sha256": SCORER.digest(data), "examples": examples}
    report_path, generations = directory / "generation_report.json", directory / "generations.jsonl"
    dump(report_path, report)
    dump_lines(generations, examples)
    return {"release": parent["release"], "spec": spec_path, "specification": specification,
        "report_path": report_path, "generations": generations, "report": report,
        "tokenizer": parent["tokenizer"], "parent": parent, "rows": rows, "contract": contract}


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
    previous = result["paired_comparison"]["previous_overall"]
    assert previous["task_correct"]["count"] == previous["ended_by_eos"]["count"] == 0
    assert previous["hit_token_budget"]["count"] == 16
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
            row["prompt"] = row["prompt"].split("\n\nResponse contract v1:")[0]
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
    example = bundle["report"]["examples"][0]
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
