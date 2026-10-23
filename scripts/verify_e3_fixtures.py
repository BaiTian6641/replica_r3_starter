#!/usr/bin/env python3
"""Offline validator for the E3 evaluation fixtures.

Checks schema exactness, role coverage, layout, leakage, that every case
starts failing and passes under its reference repair, that the reference
repair preserves the two non-target constants, and that destructive rewrites
fail closed. Evaluators run in isolated temporary copies; the committed
fixtures are hashed before and after to prove they are never mutated.

Usage: python scripts/verify_e3_fixtures.py [--cases fixtures/e3/cases]
Prints canonical JSON (per-case evidence + summary) and exits nonzero on
any failure. Designed to run offline in a few seconds.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

FORMAT = "replica.e3.fixture.v1"
ROLES = ("train", "dev", "heldout", "interrupted")
ROLE_COUNTS = {"train": 2, "dev": 1, "heldout": 2, "interrupted": 1}
ALLOWED_TOOLS = ["read_file", "write_file", "run_test"]
CASE_KEYS = {
    "format", "case_id", "role", "goal", "success_condition", "constraints",
    "allowed_tools", "workspace_files", "evaluator", "workspace_entry",
    "reference_repair", "variant",
}
SUCCESS_KEYS = {"condition_id", "description"}
VARIANT_KEYS = {"constant", "description", "actual_value", "expected_value"}
REPAIR_KEYS = {"path", "content"}
# Evaluators must be stdlib-only and must not execute or import the workspace.
EVALUATOR_ALLOWED_IMPORTS = {"ast", "importlib", "importlib.util", "pathlib", "unittest"}
FORBIDDEN_IMPORT_ROOTS = {
    "asyncio", "ftplib", "http", "requests", "runpy", "socket", "ssl",
    "subprocess", "telnetlib", "urllib",
}
FORBIDDEN_CALLS = {"compile", "eval", "exec", "__import__"}
WORKSPACE_PREFIX = "workspace/"
MAX_FILE_BYTES = 2048
MAX_FAMILY_BYTES = 40 * 1024
EVALUATOR_TIMEOUT = 30
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?![\w.])")


def _check(cond: bool, failures: list, msg: str) -> bool:
    if not cond:
        failures.append(msg)
    return bool(cond)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_equal(left, right) -> bool:
    return _is_number(left) and _is_number(right) and float(left) == float(right)


def _mentions_number(text: str, value) -> bool:
    return any(float(token.group(0)) == float(value) for token in NUMBER.finditer(text))


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_case(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"_error": f"unreadable case.json: {exc}"}
    if not isinstance(value, dict):
        return {"_error": "case.json must be a JSON object"}
    return value


def _parse_settings(text: str, label: str, failures: list) -> dict | None:
    """Parse a settings file as a module docstring plus literal assignments."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        failures.append(f"{label}: not valid Python: {exc}")
        return None
    settings: dict = {}
    ok = True
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) \
                and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            failures.append(f"{label}: only simple name = number assignments are allowed")
            ok = False
            continue
        name, value = node.targets[0].id, node.value
        if not (isinstance(value, ast.Constant) and _is_number(value.value)):
            failures.append(f"{label}: {name} must be a numeric literal")
            ok = False
            continue
        if name in settings:
            failures.append(f"{label}: {name} assigned more than once")
            ok = False
            continue
        settings[name] = value.value
    return settings if ok else None


def _resolve_workspace_file(case_dir: pathlib.Path, rel) -> tuple[pathlib.Path | None, str | None]:
    if not isinstance(rel, str) or not rel:
        return None, "path must be a nonempty string"
    pure = pathlib.PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts or not rel.startswith(WORKSPACE_PREFIX):
        return None, "path must be case-root-relative under workspace/"
    if any(part.startswith(".") or part == "__pycache__" for part in pure.parts):
        return None, "hidden/cache paths are not allowed"
    if "evaluator" in pure.parts or pure.name == "case.json":
        return None, "evaluator/case.json are outside model authority"
    target = case_dir.joinpath(*pure.parts)
    workspace_root = (case_dir / "workspace").resolve()
    if not target.resolve().is_relative_to(workspace_root):
        return None, "path resolves outside workspace/"
    if target.is_symlink():
        return None, "symlink targets are not allowed"
    return target, None


def _resolve_evaluator(case_dir: pathlib.Path, rel) -> tuple[pathlib.Path | None, str | None]:
    if not isinstance(rel, str) or not rel:
        return None, "evaluator must be a nonempty string"
    pure = pathlib.PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        return None, "evaluator must be a case-root-relative path"
    if pure.parts[0] == "workspace" or pure.name == "case.json":
        return None, "evaluator must live outside workspace/"
    if len(pure.parts) < 2 or pure.parts[0] != "evaluator" \
            or not pure.name.startswith("test_") or pure.suffix != ".py":
        return None, "evaluator must be evaluator/test_*.py"
    target = case_dir.joinpath(*pure.parts)
    if target.is_symlink():
        return None, "evaluator must not be a symlink"
    if target.resolve().is_relative_to((case_dir / "workspace").resolve()):
        return None, "evaluator must live outside workspace/"
    return target, None


def _run_evaluator(case_root: pathlib.Path, evaluator_rel: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-m", "unittest", evaluator_rel]
    try:
        return subprocess.run(command, cwd=case_root, capture_output=True, text=True,
                              timeout=EVALUATOR_TIMEOUT, env=env)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) \
            else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) \
            else (exc.stderr or "")
        return subprocess.CompletedProcess(command, 124, stdout,
                                           stderr + "\nevaluator timed out")


def _replace_assignment(text: str, name: str, value) -> str:
    pattern = re.compile(rf"^{re.escape(name)}\s*=\s*[^\n]*$", re.MULTILINE)
    updated, count = pattern.subn(f"{name} = {value!r}", text, count=1)
    if count != 1:
        raise ValueError(f"could not rewrite assignment {name}")
    return updated


def check_imports(evaluator_path: pathlib.Path, failures: list) -> None:
    """Evaluator imports must be stdlib-only and must not execute workspace code."""
    try:
        tree = ast.parse(evaluator_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        failures.append(f"evaluator unreadable: {exc}")
        return
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                failures.append("evaluator must not use relative imports")
                continue
            if node.module:
                names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS or root.startswith("replica"):
                failures.append(f"evaluator imports forbidden module {name!r}")
            elif name not in EVALUATOR_ALLOWED_IMPORTS:
                failures.append(f"evaluator import {name!r} outside allowlist")
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in FORBIDDEN_CALLS:
                failures.append(f"evaluator must not call {function.id}")
            if isinstance(function, ast.Attribute) and function.attr == "exec_module":
                failures.append("evaluator must not execute workspace code")


def check_case(case_dir: pathlib.Path, tmp_root: pathlib.Path) -> dict:
    tmp_root.mkdir(parents=True, exist_ok=True)
    evidence: dict = {"case_id": case_dir.name, "checks": [], "failures": [],
                      "destructive_mutations": []}
    fail = evidence["failures"]

    case_path = case_dir / "case.json"
    if not _check(case_path.is_file() and not case_path.is_symlink(), fail,
                  "missing or symlinked case.json"):
        return evidence
    case = _load_case(case_path)
    if "_error" in case:
        fail.append(case["_error"])
        return evidence

    # ---- schema: exact keys and types --------------------------------------
    _check(set(case) == CASE_KEYS, fail,
           f"case.json keys mismatch: extra={sorted(set(case) - CASE_KEYS)} "
           f"missing={sorted(CASE_KEYS - set(case))}")
    _check(case.get("format") == FORMAT, fail, "bad format field")
    _check(case.get("case_id") == case_dir.name, fail, "case_id does not match directory")
    _check(case.get("role") in ROLES, fail, f"bad role {case.get('role')!r}")
    goal = case.get("goal", "")
    _check(isinstance(goal, str) and bool(goal.strip()), fail, "goal must be a nonempty string")
    constraints = case.get("constraints", [])
    _check(isinstance(constraints, list) and bool(constraints)
           and all(isinstance(item, str) and bool(item.strip()) for item in constraints),
           fail, "constraints must be a nonempty list of strings")
    _check(isinstance(constraints, list) and len(set(constraints)) == len(constraints),
           fail, "constraints must be unique")
    _check(case.get("allowed_tools") == ALLOWED_TOOLS, fail,
           "allowed_tools must be exactly [read_file, write_file, run_test]")
    success = case.get("success_condition", {})
    _check(isinstance(success, dict) and set(success) == SUCCESS_KEYS, fail,
           "success_condition must have exactly condition_id/description")
    condition_id = success.get("condition_id") if isinstance(success, dict) else None
    description = success.get("description", "") if isinstance(success, dict) else ""
    _check(condition_id == "evaluator_test_passes", fail,
           "success_condition.condition_id must be evaluator_test_passes")
    _check(isinstance(description, str) and bool(description.strip()), fail,
           "success_condition.description must be a nonempty string")
    workspace_files = case.get("workspace_files", [])
    _check(isinstance(workspace_files, list) and bool(workspace_files)
           and all(isinstance(item, str) for item in workspace_files), fail,
           "workspace_files must be a nonempty list of strings")
    _check(isinstance(workspace_files, list)
           and len(set(workspace_files)) == len(workspace_files), fail,
           "workspace_files must be unique")
    entry = case.get("workspace_entry")
    _check(isinstance(entry, str) and isinstance(workspace_files, list)
           and entry in workspace_files, fail,
           "workspace_entry must name a workspace_files entry")
    repair = case.get("reference_repair", {})
    _check(isinstance(repair, dict) and set(repair) == REPAIR_KEYS, fail,
           "reference_repair must have exactly path/content")
    repair_path = repair.get("path") if isinstance(repair, dict) else None
    repair_content = repair.get("content", "") if isinstance(repair, dict) else ""
    _check(isinstance(repair_content, str) and bool(repair_content), fail,
           "reference_repair.content must be a nonempty string")
    _check(repair_path == entry, fail, "reference_repair.path must equal workspace_entry")
    variant = case.get("variant", {})
    _check(isinstance(variant, dict) and set(variant) == VARIANT_KEYS, fail,
           "variant must have exactly constant/description/actual_value/expected_value")
    constant = variant.get("constant") if isinstance(variant, dict) else None
    actual = variant.get("actual_value") if isinstance(variant, dict) else None
    expected = variant.get("expected_value") if isinstance(variant, dict) else None
    _check(isinstance(constant, str) and constant.isidentifier(), fail,
           "variant.constant must be an identifier")
    _check(isinstance(variant.get("description"), str)
           and bool(variant.get("description", "").strip()), fail,
           "variant.description must be a nonempty string")
    _check(_is_number(actual), fail, "variant.actual_value must be a JSON number")
    _check(_is_number(expected), fail, "variant.expected_value must be a JSON number")
    _check(_is_number(actual) and _is_number(expected) and actual != expected, fail,
           "variant actual and expected values must differ")

    # ---- layout: workspace files under workspace/, evaluator outside -------
    workspace_paths: dict[str, pathlib.Path] = {}
    for rel in workspace_files if isinstance(workspace_files, list) else []:
        target, error = _resolve_workspace_file(case_dir, rel)
        if not _check(error is None, fail, f"workspace file {rel!r}: {error}"):
            continue
        if _check(target.is_file() and not target.is_symlink(), fail,
                  f"workspace file missing or not regular: {rel}"):
            workspace_paths[rel] = target
    evaluator_rel = case.get("evaluator")
    evaluator_path = None
    if isinstance(evaluator_rel, str):
        evaluator_path, error = _resolve_evaluator(case_dir, evaluator_rel)
        _check(error is None, fail, f"evaluator path {evaluator_rel!r}: {error}")
        if error is None:
            _check(evaluator_path.is_file() and not evaluator_path.is_symlink(), fail,
                   f"evaluator missing or not regular: {evaluator_rel}")
            _check(not isinstance(workspace_files, list)
                   or evaluator_rel not in workspace_files, fail,
                   "evaluator must not be listed as a workspace file")
    else:
        fail.append("evaluator must be a string")

    # ---- reference repair semantics ----------------------------------------
    committed_settings: dict = {}
    for rel, path in workspace_paths.items():
        parsed = _parse_settings(path.read_text(encoding="utf-8"), rel, fail)
        if parsed is not None:
            committed_settings.update(parsed)
    repaired_settings = _parse_settings(repair_content, "reference_repair.content", fail) \
        if isinstance(repair_content, str) else None
    if isinstance(constant, str) and committed_settings and repaired_settings is not None:
        _check(set(repaired_settings) == set(committed_settings), fail,
               "reference repair must keep exactly the original setting names")
        _check(constant in committed_settings, fail,
               "variant.constant must exist in the committed workspace")
        if constant in committed_settings and _is_number(actual):
            _check(_numbers_equal(committed_settings[constant], actual), fail,
                   "committed workspace value must equal variant.actual_value")
        if constant in repaired_settings and _is_number(expected):
            _check(_numbers_equal(repaired_settings[constant], expected), fail,
                   "reference repair must set the target to variant.expected_value")
        for name, value in committed_settings.items():
            if name == constant:
                continue
            _check(name in repaired_settings and _numbers_equal(repaired_settings[name], value),
                   fail, f"reference repair must preserve non-target constant {name}")

    # ---- leakage ------------------------------------------------------------
    constraint_text = " ".join(str(item) for item in constraints) \
        if isinstance(constraints, list) else ""
    prompt_text = "\n".join([goal if isinstance(goal, str) else "",
                             constraint_text,
                             description if isinstance(description, str) else ""])
    visible_workspace = "\n".join(path.read_text(encoding="utf-8")
                                 for path in workspace_paths.values())
    if _is_number(expected):
        _check(str(expected) not in prompt_text, fail,
               "expected value leaks as text into goal/constraints/description")
        _check(not _mentions_number(prompt_text, expected), fail,
               "expected value leaks as a number into goal/constraints/description")
        _check(not _mentions_number(visible_workspace, expected), fail,
               "expected value leaks into the current workspace")
    _check(repair_content not in prompt_text + "\n" + visible_workspace, fail,
           "reference repair content reachable from model-visible text")

    if evaluator_path is not None and evaluator_path.is_file():
        check_imports(evaluator_path, fail)

    # ---- failing before, passing after repair (isolated temp copy) ---------
    work = pathlib.Path(tempfile.mkdtemp(prefix="e3fix_", dir=tmp_root))
    try:
        run_dir = work / case_dir.name
        shutil.copytree(case_dir, run_dir, symlinks=False)
        if isinstance(evaluator_rel, str) and isinstance(repair_path, str) \
                and repair_path == entry and entry in workspace_files:
            pre = _run_evaluator(run_dir, evaluator_rel)
            evidence["pre_repair_exit"] = pre.returncode
            pre_output = (pre.stderr or "") + (pre.stdout or "")
            evidence["pre_repair_message"] = pre_output[-300:]
            _check(pre.returncode != 0, fail, "evaluator must fail before the repair")
            if isinstance(constant, str):
                _check(constant in pre_output, fail,
                       "pre-repair failure must name the target constant")
            target_path = run_dir / repair_path
            target_path.write_text(repair_content, encoding="utf-8")
            post = _run_evaluator(run_dir, evaluator_rel)
            evidence["post_repair_exit"] = post.returncode
            _check(post.returncode == 0, fail,
                   f"evaluator must pass after reference repair: {(post.stderr or '')[-200:]}")
            for name, value in (repaired_settings or {}).items():
                if name == constant or not _is_number(value):
                    continue
                try:
                    mutated = _replace_assignment(repair_content, name, value + 1)
                except ValueError as exc:
                    fail.append(str(exc))
                    continue
                target_path.write_text(mutated, encoding="utf-8")
                destroyed = _run_evaluator(run_dir, evaluator_rel)
                output = (destroyed.stderr or "") + (destroyed.stdout or "")
                evidence["destructive_mutations"].append(
                    {"constant": name, "exit": destroyed.returncode,
                     "named": name in output})
                _check(destroyed.returncode != 0, fail,
                       f"evaluator must reject a changed non-target constant {name}")
        if workspace_paths and isinstance(constant, str) and _is_number(actual):
            committed = _parse_settings(pathlib.Path(next(iter(workspace_paths.values())))
                                        .read_text(encoding="utf-8"),
                                        "committed workspace", fail)
            if committed is not None and constant in committed:
                _check(_numbers_equal(committed[constant], actual), fail,
                       "committed workspace must still hold variant.actual_value")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    evidence["checks"] = [
        "schema keys/types exact",
        "workspace files and evaluator layout safe",
        "reference repair changes only the target constant",
        "expected value and reference content absent from model-visible text",
        "pre-repair failure, post-repair pass, destructive rewrites rejected",
    ]
    evidence["passed"] = not fail
    return evidence


def _snapshot(root: pathlib.Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and not path.is_symlink()
            and ".host_tmp" not in path.relative_to(root).parts}


def validate_family(cases_root: pathlib.Path) -> dict:
    report: dict = {"validator": "verify_e3_fixtures.py", "cases": [], "summary": {}}
    global_failures: list[str] = []
    if not cases_root.is_dir():
        global_failures.append(f"cases root missing: {cases_root}")
        report["summary"] = {"cases": 0, "passed": 0, "failures": global_failures,
                             "ok": False}
        return report

    before = _snapshot(cases_root)
    tmp_root = cases_root.parent / ".verify_tmp"
    case_dirs = sorted(path for path in cases_root.iterdir() if path.is_dir())
    loaded: dict[str, dict] = {}
    for case_dir in case_dirs:
        evidence = check_case(case_dir, tmp_root)
        report["cases"].append(evidence)
        loaded[case_dir.name] = _load_case(case_dir / "case.json")
        for failure in evidence["failures"]:
            global_failures.append(f"{evidence['case_id']}: {failure}")

    # ---- family schema, roles, ids, cross-case leakage ---------------------
    roles: Counter = Counter()
    seen_ids: dict[str, str] = {}
    target_pairs: dict[tuple, str] = {}
    target_constants: dict[str, str] = {}
    target_values: dict[float, str] = {}
    parsed_workspaces: dict[str, dict] = {}
    for case_dir in case_dirs:
        case = loaded[case_dir.name]
        if "_error" in case:
            continue
        case_id = case.get("case_id")
        if isinstance(case_id, str):
            _check(case_id not in seen_ids, global_failures,
                   f"duplicate case id {case_id}")
            seen_ids[case_id] = case_dir.name
        if case.get("role") in ROLES:
            roles[case["role"]] += 1
        variant = case.get("variant") if isinstance(case.get("variant"), dict) else {}
        constant = variant.get("constant")
        expected = variant.get("expected_value")
        if isinstance(constant, str) and _is_number(expected):
            pair = (constant, float(expected))
            if pair in target_pairs:
                global_failures.append(
                    f"duplicate target constant/value pair {constant}={expected} "
                    f"in {target_pairs[pair]} and {case_dir.name}")
            target_pairs[pair] = case_dir.name
            if constant in target_constants:
                global_failures.append(
                    f"target constant {constant} reused by "
                    f"{target_constants[constant]} and {case_dir.name}")
            target_constants[constant] = case_dir.name
            if float(expected) in target_values:
                global_failures.append(
                    f"target expected value {expected} reused by "
                    f"{target_values[float(expected)]} and {case_dir.name}")
            target_values[float(expected)] = case_dir.name
        settings: dict = {}
        for rel in case.get("workspace_files") or []:
            path = case_dir / rel if isinstance(rel, str) else None
            if path is not None and path.is_file() and not path.is_symlink():
                parsed = _parse_settings(path.read_text(encoding="utf-8"),
                                         f"{case_dir.name}:{rel}", global_failures)
                if parsed is not None:
                    settings.update(parsed)
        parsed_workspaces[case_dir.name] = settings

    _check(len(case_dirs) == sum(ROLE_COUNTS.values()), global_failures,
           f"fixture family must contain exactly {sum(ROLE_COUNTS.values())} cases")
    for role, count in ROLE_COUNTS.items():
        _check(roles[role] == count, global_failures,
               f"role {role!r} must appear {count} time(s), found {roles[role]}")

    for name, settings in parsed_workspaces.items():
        case = loaded[name]
        variant = case.get("variant") if isinstance(case.get("variant"), dict) else {}
        constant = variant.get("constant")
        expected = variant.get("expected_value")
        if not isinstance(constant, str) or not _is_number(expected):
            continue
        for other, other_settings in parsed_workspaces.items():
            if other == name:
                continue
            if constant in other_settings and _numbers_equal(other_settings[constant], expected):
                global_failures.append(
                    f"{name}: target pair {constant}={expected} also appears in "
                    f"{other} workspace")
            other_case = loaded[other]
            other_variant = other_case.get("variant") \
                if isinstance(other_case.get("variant"), dict) else {}
            other_goal = other_case.get("goal", "")
            other_constraints = other_case.get("constraints", [])
            other_success = other_case.get("success_condition", {})
            other_text = "\n".join([
                other_goal if isinstance(other_goal, str) else "",
                " ".join(other_constraints) if isinstance(other_constraints, list) else "",
                other_success.get("description", "") if isinstance(other_success, dict) else "",
            ])
            if not _mentions_number(other_text, expected):
                continue
            global_failures.append(
                f"{name}: expected value {expected} appears in {other} prompt text")

    # ---- size, no symlinks, no generated files -----------------------------
    total = 0
    for path in sorted(cases_root.rglob("*")):
        rel = path.relative_to(cases_root).as_posix()
        rel_parts = pathlib.PurePosixPath(rel).parts
        if ".host_tmp" in rel_parts:
            continue  # runtime scratch owned by FixtureWorkspace, not fixture content
        if not _check(not path.is_symlink(), global_failures,
                      f"symlink in fixture family: {rel}"):
            continue
        if path.is_dir():
            _check(path.name != "__pycache__", global_failures,
                   f"{rel}: generated directory in fixture family")
            continue
        if not path.is_file():
            continue
        total += path.stat().st_size
        _check(path.stat().st_size <= MAX_FILE_BYTES, global_failures,
               f"{rel}: file exceeds {MAX_FILE_BYTES} bytes")
        parts = pathlib.PurePosixPath(rel).parts
        _check("__pycache__" not in parts and path.name not in (".DS_Store",)
               and path.suffix != ".pyc", global_failures,
               f"{rel}: generated/cache file in fixture family")
    _check(total <= MAX_FAMILY_BYTES, global_failures,
           f"fixture family exceeds {MAX_FAMILY_BYTES} bytes")

    after = _snapshot(cases_root)
    _check(before == after, global_failures, "validator mutated the committed fixtures")
    shutil.rmtree(tmp_root, ignore_errors=True)

    report["family"] = {
        "roles": dict(sorted(roles.items())),
        "target_pairs": {f"{key[0]}={key[1]}": value
                         for key, value in sorted(target_pairs.items())},
        "bytes": total,
        "committed_unchanged": before == after,
    }
    report["summary"] = {
        "cases": len(report["cases"]),
        "passed": sum(1 for case in report["cases"] if case.get("passed")),
        "failures": global_failures,
        "ok": not global_failures,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="fixtures/e3/cases")
    args = parser.parse_args(argv)
    report = validate_family(pathlib.Path(args.cases))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
