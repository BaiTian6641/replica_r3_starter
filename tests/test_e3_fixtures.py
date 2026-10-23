"""Tests for the E3 fixture validator, fixture family and runtime authority."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]
CASES_ROOT = REPO / "fixtures/e3/cases"
VALIDATOR = REPO / "scripts/verify_e3_fixtures.py"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import verify_e3_fixtures as vef  # noqa: E402

EXPECTED_IDS = {
    "train_retry_01", "train_timeout_02", "dev_retry_03",
    "heldout_timeout_04", "heldout_retry_05", "interrupted_retry_06",
}
EXPECTED_ROLE_COUNTS = {"train": 2, "dev": 1, "heldout": 2, "interrupted": 1}


def load_case(case_dir):
    return json.loads((case_dir / "case.json").read_text(encoding="utf-8"))


def copy_case(case_id, parent, name=None):
    destination = pathlib.Path(parent) / (name or case_id)
    shutil.copytree(CASES_ROOT / case_id, destination)
    return destination


def run_evaluator(case_dir):
    case = load_case(case_dir)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "unittest", case["evaluator"]],
        cwd=case_dir, capture_output=True, text=True, timeout=30, env=env)


def file_hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and not path.is_symlink()}


def settings_of(case_dir):
    case = load_case(case_dir)
    text = (case_dir / case["reference_repair"]["path"]).read_text(encoding="utf-8")
    failures = []
    parsed = vef._parse_settings(text, "test", failures)
    assert not failures, failures
    return parsed


class FixtureFamilyTest(unittest.TestCase):
    def test_case_inventory_roles_and_counts(self):
        dirs = {path.name for path in CASES_ROOT.iterdir() if path.is_dir()}
        self.assertEqual(dirs, EXPECTED_IDS)
        roles = Counter(load_case(CASES_ROOT / case_id)["role"] for case_id in EXPECTED_IDS)
        self.assertEqual(dict(roles), EXPECTED_ROLE_COUNTS)

    def test_schema_exactness(self):
        for case_id in EXPECTED_IDS:
            case_dir = CASES_ROOT / case_id
            case = load_case(case_dir)
            self.assertEqual(set(case), vef.CASE_KEYS, case_id)
            self.assertEqual(case["format"], vef.FORMAT)
            self.assertEqual(case["case_id"], case_id)
            self.assertIn(case["role"], vef.ROLES)
            self.assertIsInstance(case["goal"], str)
            self.assertTrue(case["goal"].strip())
            self.assertEqual(case["allowed_tools"], vef.ALLOWED_TOOLS)
            self.assertEqual(case["evaluator"], "evaluator/test_policy.py")
            self.assertIsInstance(case["constraints"], list)
            self.assertTrue(all(isinstance(item, str) and item for item in case["constraints"]))
            condition = case["success_condition"]
            self.assertEqual(set(condition), vef.SUCCESS_KEYS)
            self.assertEqual(condition["condition_id"], "evaluator_test_passes")
            self.assertTrue(condition["description"])
            self.assertEqual(set(case["reference_repair"]), vef.REPAIR_KEYS)
            self.assertEqual(case["reference_repair"]["path"], case["workspace_entry"])
            self.assertIn(case["workspace_entry"], case["workspace_files"])
            for rel in case["workspace_files"]:
                self.assertTrue(rel.startswith("workspace/"), (case_id, rel))
            variant = case["variant"]
            self.assertEqual(set(variant), vef.VARIANT_KEYS)
            self.assertTrue(variant["constant"].isidentifier())
            self.assertIsInstance(variant["actual_value"], (int, float))
            self.assertIsInstance(variant["expected_value"], (int, float))
            self.assertNotIsInstance(variant["actual_value"], bool)
            self.assertNotIsInstance(variant["expected_value"], bool)
            self.assertNotEqual(variant["actual_value"], variant["expected_value"])

    def test_size_bounds_and_no_generated_files(self):
        total = 0
        for path in CASES_ROOT.rglob("*"):
            if ".host_tmp" in path.parts:
                continue
            self.assertFalse(path.is_symlink(), path)
            self.assertNotIn(path.name, ("__pycache__",), path)
            if not path.is_file():
                continue
            self.assertLess(path.stat().st_size, 2048, path)
            self.assertNotIn("__pycache__", path.parts, path)
            self.assertNotEqual(path.name, ".DS_Store", path)
            self.assertNotEqual(path.suffix, ".pyc", path)
            total += path.stat().st_size
        self.assertLess(total, 40 * 1024)

    def test_layout(self):
        for case_id in EXPECTED_IDS:
            case_dir = CASES_ROOT / case_id
            self.assertTrue((case_dir / "case.json").is_file())
            self.assertTrue((case_dir / "workspace/app_settings.py").is_file())
            self.assertTrue((case_dir / "evaluator/test_policy.py").is_file())
            self.assertFalse((case_dir / "workspace/evaluator").exists())
            self.assertFalse((case_dir / "workspace/case.json").exists())

    def test_workspace_is_only_literal_settings(self):
        for case_id in EXPECTED_IDS:
            case = load_case(CASES_ROOT / case_id)
            text = (CASES_ROOT / case_id / case["workspace_entry"]).read_text(encoding="utf-8")
            failures = []
            parsed = vef._parse_settings(text, case_id, failures)
            self.assertEqual(failures, [], case_id)
            self.assertEqual(set(parsed), set(settings_of(CASES_ROOT / case_id)), case_id)
            self.assertEqual(len(parsed), 3, case_id)

    def test_reference_repair_changes_only_target(self):
        for case_id in EXPECTED_IDS:
            case = load_case(CASES_ROOT / case_id)
            workspace = (CASES_ROOT / case_id / case["workspace_entry"]).read_text(encoding="utf-8")
            failures = []
            pre = vef._parse_settings(workspace, "workspace", failures)
            post = vef._parse_settings(case["reference_repair"]["content"], "repair", failures)
            self.assertEqual(failures, [], case_id)
            self.assertEqual(set(pre), set(post), case_id)
            target = case["variant"]["constant"]
            self.assertEqual(pre[target], case["variant"]["actual_value"], case_id)
            self.assertEqual(post[target], case["variant"]["expected_value"], case_id)
            for name, value in pre.items():
                if name != target:
                    self.assertEqual(post[name], value, (case_id, name))

    def test_distinct_constants_values_and_target_pairs(self):
        constants, values, pairs = set(), set(), set()
        for case_id in EXPECTED_IDS:
            variant = load_case(CASES_ROOT / case_id)["variant"]
            constant, expected = variant["constant"], variant["expected_value"]
            self.assertNotIn(constant, constants, case_id)
            self.assertNotIn(expected, values, case_id)
            self.assertNotIn((constant, expected), pairs, case_id)
            constants.add(constant)
            values.add(expected)
            pairs.add((constant, expected))

    def test_leakage(self):
        for case_id in EXPECTED_IDS:
            case_dir = CASES_ROOT / case_id
            case = load_case(case_dir)
            expected = case["variant"]["expected_value"]
            prompt = "\n".join([case["goal"], " ".join(case["constraints"]),
                                case["success_condition"]["description"]])
            workspace = (case_dir / case["workspace_entry"]).read_text(encoding="utf-8")
            self.assertNotIn(str(expected), prompt, case_id)
            self.assertFalse(vef._mentions_number(prompt, expected), case_id)
            self.assertFalse(vef._mentions_number(workspace, expected), case_id)
            self.assertNotIn(case["reference_repair"]["content"],
                             prompt + "\n" + workspace, case_id)
            for rel in case["workspace_files"]:
                self.assertTrue(rel.startswith("workspace/"), case_id)
            self.assertNotIn(case["evaluator"], case["workspace_files"])

    def test_no_cross_case_target_pair_transfer(self):
        cases = {case_id: load_case(CASES_ROOT / case_id) for case_id in EXPECTED_IDS}
        settings = {case_id: settings_of(CASES_ROOT / case_id) for case_id in EXPECTED_IDS}
        for case_id, case in cases.items():
            constant = case["variant"]["constant"]
            expected = case["variant"]["expected_value"]
            for other_id, other_settings in settings.items():
                if other_id == case_id:
                    continue
                self.assertFalse(constant in other_settings
                                 and other_settings[constant] == expected,
                                 (case_id, other_id))
                other = cases[other_id]
                prompt = "\n".join([other["goal"], " ".join(other["constraints"]),
                                    other["success_condition"]["description"]])
                workspace = (CASES_ROOT / other_id / other["workspace_entry"]).read_text(
                    encoding="utf-8")
                self.assertFalse(vef._mentions_number(prompt, expected),
                                 (case_id, other_id, "prompt"))
                self.assertFalse(vef._mentions_number(workspace, expected),
                                 (case_id, other_id, "workspace"))

    def test_evaluator_is_stdlib_only_and_static(self):
        for case_id in EXPECTED_IDS:
            case_dir = CASES_ROOT / case_id
            check = load_case(case_dir)["evaluator"]
            path = case_dir / check
            failures = []
            vef.check_imports(path, failures)
            self.assertEqual(failures, [], case_id)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("exec_module", text, case_id)
            self.assertNotIn("spec_from_file_location", text, case_id)
            self.assertIn("ast.parse", text, case_id)


class RuntimeAuthorityTest(unittest.TestCase):
    def test_runtime_tools_deny_evaluator_and_case_json(self):
        from replica_lab.runtime.tools import FixtureWorkspace

        with tempfile.TemporaryDirectory() as tmp:
            case_dir = copy_case("train_retry_01", tmp)
            case = load_case(case_dir)
            tools = FixtureWorkspace(case_dir)
            settings = tools.read_file({"path": "workspace/app_settings.py"})
            self.assertEqual(settings["status"], "ok")
            self.assertNotIn("policy requires", settings["content"])
            for raw in ("app_settings.py", "evaluator/test_policy.py", "case.json",
                        "../case.json", "/etc/passwd", "workspace/../case.json",
                        "workspace/evaluator/test_policy.py", "workspace/case.json"):
                self.assertNotEqual(tools.read_file({"path": raw})["status"], "ok", raw)
            evaluator_before = (case_dir / "evaluator/test_policy.py").read_bytes()
            case_before = (case_dir / "case.json").read_bytes()
            for raw in ("app_settings.py", "evaluator/test_policy.py", "case.json",
                        "../case.json", "/etc/passwd", "workspace/../case.json"):
                result = tools.write_file({"path": raw, "content": "x"})
                self.assertIn(result["status"], ("denied", "error"), raw)
            self.assertEqual((case_dir / "evaluator/test_policy.py").read_bytes(),
                             evaluator_before)
            self.assertEqual((case_dir / "case.json").read_bytes(), case_before)

            pre = tools.run_test({})
            self.assertEqual(pre["status"], "error")
            self.assertNotEqual(pre["receipt"]["exit_code"], 0)
            self.assertIn(str(case["variant"]["expected_value"]),
                          pre["receipt"]["stderr_excerpt"])
            repaired = tools.write_file(
                {"path": "workspace/app_settings.py",
                 "content": case["reference_repair"]["content"]})
            self.assertEqual(repaired["status"], "ok")
            post = tools.run_test({})
            self.assertEqual(post["status"], "ok")
            self.assertEqual(post["receipt"]["exit_code"], 0)

    def test_committed_cases_load_in_runtime(self):
        from replica_lab.runtime.tools import FixtureWorkspace

        with tempfile.TemporaryDirectory() as tmp:
            for case_id in EXPECTED_IDS:
                tools = FixtureWorkspace(copy_case(case_id, tmp))
                self.assertEqual(tools.case_id, case_id)
                result = tools.read_file({"path": "workspace/app_settings.py"})
                self.assertEqual(result["status"], "ok", case_id)


class ValidatorTest(unittest.TestCase):
    def test_validator_passes_on_committed_fixtures(self):
        proc = subprocess.run(
            [sys.executable, str(VALIDATOR), "--cases", str(CASES_ROOT)],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:])
        report = json.loads(proc.stdout)
        self.assertTrue(report["summary"]["ok"])
        self.assertEqual(report["summary"]["cases"], 6)
        self.assertEqual(report["summary"]["passed"], 6)
        self.assertTrue(report["family"]["committed_unchanged"])
        for case in report["cases"]:
            self.assertNotEqual(case["pre_repair_exit"], 0)
            self.assertEqual(case["post_repair_exit"], 0)
            self.assertTrue(all(item["exit"] != 0 for item in case["destructive_mutations"]))

    def test_validator_does_not_mutate_committed_fixtures(self):
        before = file_hashes(CASES_ROOT)
        proc = subprocess.run(
            [sys.executable, str(VALIDATOR), "--cases", str(CASES_ROOT)],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:])
        self.assertEqual(file_hashes(CASES_ROOT), before)

    def test_rejects_already_passing_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = copy_case("train_retry_01", tmp)
            case = load_case(dst)
            text = (dst / case["workspace_entry"]).read_text(encoding="utf-8")
            (dst / case["workspace_entry"]).write_text(
                text.replace("RETRY_ATTEMPTS = 2", "RETRY_ATTEMPTS = 5"), encoding="utf-8")
            evidence = vef.check_case(dst, pathlib.Path(tmp))
            self.assertFalse(evidence["passed"])
            self.assertTrue(any("fail before" in item for item in evidence["failures"]))

    def test_rejects_evaluator_inside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = copy_case("train_retry_01", tmp)
            case = load_case(dst)
            (dst / "workspace/evaluator").mkdir()
            shutil.copy(dst / "evaluator/test_policy.py",
                        dst / "workspace/evaluator/test_policy.py")
            case["workspace_files"].append("workspace/evaluator/test_policy.py")
            (dst / "case.json").write_text(json.dumps(case), encoding="utf-8")
            evidence = vef.check_case(dst, pathlib.Path(tmp))
            self.assertFalse(evidence["passed"])

    def test_rejects_destructive_rewrite_even_with_target_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = copy_case("train_retry_01", tmp)
            case = load_case(dst)
            lines = [line for line in case["reference_repair"]["content"].splitlines()
                     if not line.startswith("BACKOFF_SECONDS")]
            (dst / case["workspace_entry"]).write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
            evidence = vef.check_case(dst, pathlib.Path(tmp))
            self.assertFalse(evidence["passed"])

    def test_rejects_expected_value_leaked_into_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = copy_case("train_retry_01", tmp)
            case = load_case(dst)
            case["goal"] += " The policy bound is 5."
            (dst / "case.json").write_text(json.dumps(case), encoding="utf-8")
            evidence = vef.check_case(dst, pathlib.Path(tmp))
            self.assertFalse(evidence["passed"])
            self.assertTrue(any("goal" in item or "leak" in item
                                for item in evidence["failures"]))

    def test_rejects_reference_repair_that_changes_non_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = copy_case("train_retry_01", tmp)
            case = load_case(dst)
            case["reference_repair"]["content"] = case["reference_repair"]["content"].replace(
                "BACKOFF_SECONDS = 0.4", "BACKOFF_SECONDS = 1.4")
            (dst / "case.json").write_text(json.dumps(case), encoding="utf-8")
            evidence = vef.check_case(dst, pathlib.Path(tmp))
            self.assertFalse(evidence["passed"])
            self.assertTrue(any("preserve" in item for item in evidence["failures"]))

    def test_rejects_reference_repair_missing_non_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = copy_case("train_retry_01", tmp)
            case = load_case(dst)
            case["reference_repair"]["content"] = "\n".join(
                line for line in case["reference_repair"]["content"].splitlines()
                if not line.startswith("BACKOFF_SECONDS")) + "\n"
            (dst / "case.json").write_text(json.dumps(case), encoding="utf-8")
            evidence = vef.check_case(dst, pathlib.Path(tmp))
            self.assertFalse(evidence["passed"])

    def test_rejects_unsafe_workspace_path(self):
        for rel in ("../case.json", "workspace/../case.json", "/etc/passwd",
                    "workspace/evaluator/test_policy.py"):
            with tempfile.TemporaryDirectory() as tmp:
                dst = copy_case("train_retry_01", tmp)
                case = load_case(dst)
                case["workspace_files"] = [rel]
                case["workspace_entry"] = rel
                (dst / "case.json").write_text(json.dumps(case), encoding="utf-8")
                evidence = vef.check_case(dst, pathlib.Path(tmp))
                self.assertFalse(evidence["passed"], rel)

    def test_rejects_extra_case_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = copy_case("train_retry_01", tmp)
            case = load_case(dst)
            case["unexpected"] = True
            (dst / "case.json").write_text(json.dumps(case), encoding="utf-8")
            evidence = vef.check_case(dst, pathlib.Path(tmp))
            self.assertFalse(evidence["passed"])
            self.assertTrue(any("keys mismatch" in item for item in evidence["failures"]))

    def test_rejects_forbidden_evaluator_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = copy_case("train_retry_01", tmp)
            case = load_case(dst)
            path = dst / case["evaluator"]
            path.write_text(path.read_text(encoding="utf-8") + "import subprocess\n",
                            encoding="utf-8")
            failures = []
            vef.check_imports(path, failures)
            self.assertTrue(any("subprocess" in item for item in failures), failures)

    def test_rejects_symlinked_workspace_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = copy_case("train_retry_01", tmp)
            workspace = dst / "workspace/app_settings.py"
            outside = pathlib.Path(tmp) / "outside.py"
            outside.write_text(workspace.read_text(encoding="utf-8"), encoding="utf-8")
            workspace.unlink()
            os.symlink(outside, workspace)
            evidence = vef.check_case(dst, pathlib.Path(tmp))
            self.assertFalse(evidence["passed"])

    def test_rejects_wrong_role_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            family = pathlib.Path(tmp) / "cases"
            shutil.copytree(CASES_ROOT, family)
            case_path = family / "dev_retry_03/case.json"
            case = json.loads(case_path.read_text(encoding="utf-8"))
            case["role"] = "train"
            case_path.write_text(json.dumps(case), encoding="utf-8")
            report = vef.validate_family(family)
            self.assertFalse(report["summary"]["ok"])
            self.assertTrue(any("role" in item for item in report["summary"]["failures"]))


class ReportTest(unittest.TestCase):
    def test_report_matches_committed_files(self):
        report_path = REPO / "reports/e3_fixture_review_20261023.json"
        self.assertTrue(report_path.is_file(), report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["format"], vef.FORMAT)
        self.assertTrue(report["reviewed_at_utc"])
        self.assertIn("independent", report["audit"]["kind"].lower())
        for entry in report["files"]:
            path = REPO / entry["path"]
            self.assertTrue(path.is_file(), entry["path"])
            data = path.read_bytes()
            self.assertEqual(len(data), entry["bytes"], entry["path"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"],
                             entry["path"])
        not_established = " ".join(report["not_established"]).lower()
        for phrase in ("exposed development fixtures", "generalization", "native mps"):
            self.assertIn(phrase, not_established)


class EvaluatorRobustnessTest(unittest.TestCase):
    def _run_mutation(self, case_id, content):
        with tempfile.TemporaryDirectory() as tmp:
            dst = copy_case(case_id, tmp)
            case = load_case(dst)
            target = dst / case["workspace_entry"]
            if content is None:
                target.unlink()
            else:
                target.write_text(content, encoding="utf-8")
            return run_evaluator(dst)

    def _expect_failure(self, case_id, content, label):
        proc = self._run_mutation(case_id, content)
        output = (proc.stderr or "") + (proc.stdout or "")
        self.assertNotEqual(proc.returncode, 0, (case_id, label, output[-500:]))
        self.assertIn("FAILED (failures=1)", output, (case_id, label, output[-500:]))
        return output

    def test_malformed_workspaces_fail_closed(self):
        for case_id in EXPECTED_IDS:
            case = load_case(CASES_ROOT / case_id)
            target = case["variant"]["constant"]
            expected = case["variant"]["expected_value"]
            base = case["reference_repair"]["content"]
            original = settings_of(CASES_ROOT / case_id)
            original_keys = [name for name in original if name != target]
            first = original_keys[0]
            mutations = {
                "missing": None,
                "syntax": base.replace(f"{target} = {expected}", f"{target} ="),
                "non_numeric": base.replace(f"{target} = {expected}",
                                            f'{target} = "{expected}"'),
                "renamed": base.replace(f"{target} = {expected}",
                                        f"{target}_RENAMED = {expected}"),
                "deleted": "\n".join(line for line in base.splitlines()
                                     if not line.startswith(f"{target} =")) + "\n",
                "extra_constant": base + "EXTRA = 1\n",
                "extra_function": base + "def helper():\n    return 1\n",
                "os_exit": "import os\n" + base + "os._exit(0)\n",
                "object_override": base.replace(
                    f"{target} = {expected}",
                    "class Value:\n    def __ge__(self, other):\n        return True\n"
                    f"{target} = Value()"),
                "non_target_changed": base.replace(
                    f"{first} = {original[first]}", f"{first} = {original[first] + 1}"),
                "duplicate_target": base + f"{target} = {expected + 1}\n",
            }
            for label, content in mutations.items():
                self._expect_failure(case_id, content, label)

    def test_reference_repair_passes_every_case(self):
        for case_id in EXPECTED_IDS:
            with tempfile.TemporaryDirectory() as tmp:
                dst = copy_case(case_id, tmp)
                case = load_case(dst)
                (dst / case["workspace_entry"]).write_text(
                    case["reference_repair"]["content"], encoding="utf-8")
                proc = run_evaluator(dst)
                self.assertEqual(proc.returncode, 0, (case_id, proc.stderr[-500:]))


if __name__ == "__main__":
    unittest.main()
