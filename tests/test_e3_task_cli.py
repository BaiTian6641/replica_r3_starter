"""E3 task CLI state/binding construction; no model or GPU."""

from pathlib import Path
from types import SimpleNamespace
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e3_task_cli", ROOT / "scripts" / "run_e3_task.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def arguments():
    return SimpleNamespace(
        max_decisions=7, max_tool_calls=5, max_new_tokens=128,
        max_prompt_tokens=640, max_total_prompt_tokens=4480,
        max_total_generated_tokens=896, branch="main", purpose="bounded_repair",
        subject=None)


def test_initial_state_binding_is_scalar_and_spends():
    from replica_lab.runtime.tools import FixtureWorkspace
    tools = FixtureWorkspace(ROOT / "fixtures/e3/cases/interrupted_retry_06")
    args = arguments()
    state = CLI.build_initial_state(tools, args)
    assert all(not isinstance(value, dict) for value in state.binding.values())
    assert state.binding["runtime"] == "e3-runtime-v1"
    spent = state.spend(decisions=1, prompt_tokens=10, generated_tokens=2)
    assert spent.budget.used_decisions == 1
    assert spent.budget.used_prompt_tokens == 10


def test_journal_binding_component_versions_are_strings():
    from replica_lab.runtime.tools import FixtureWorkspace
    tools = FixtureWorkspace(ROOT / "fixtures/e3/cases/interrupted_retry_06")
    binding = CLI.journal_binding(tools, arguments())
    assert binding["task_id"] == tools.case_id
    assert binding["branch"] == "main"
    assert all(isinstance(value, str)
               for value in binding["component_versions"].values())
    assert len(binding["component_versions"]["budget"]) == 64
