"""Shared fast fixtures for the E3 runtime tests (no native model work)."""

import json

import torch

from replica_lab.architecture.contracts import CoreState, fingerprint
from replica_lab.runtime.journal import TaskJournal
from replica_lab.runtime.loop import ProposalOutput, TaskRuntime
from replica_lab.runtime.task_graph import (Budget, ExactTaskState, SuccessCondition)
from replica_lab.runtime.tools import FixtureWorkspace

GOAL = 'Make the evaluator pass by setting the required value.'
REPAIR = '{"rate": 0.25}\n'
ORIGINAL = '{"rate": 1.0}\n'


def make_case(root, *, case_id='runtime-case', prefix_paths=True):
    case_root = root / 'cases' / case_id
    (case_root / 'workspace').mkdir(parents=True)
    (case_root / 'evaluator').mkdir(parents=True)
    (case_root / 'workspace' / 'config.json').write_text(ORIGINAL)
    (case_root / 'evaluator' / 'test_rate.py').write_text(
        'import json, pathlib\n'
        'cfg = json.loads(pathlib.Path("workspace/config.json").read_text())\n'
        'assert cfg["rate"] == 0.25, "config fails the bound check"\n'
        'print("ok")\n')
    declared = 'workspace/config.json' if prefix_paths else 'config.json'
    case = {
        'format': 'replica.e3.fixture.v1', 'case_id': case_id, 'role': 'train',
        'goal': GOAL,
        'success_condition': {'condition_id': 'sc-1', 'description': 'evaluator exits 0'},
        'constraints': ['Only workspace/ is writable'],
        'allowed_tools': ['read_file', 'write_file', 'run_test'],
        'workspace_files': [declared], 'evaluator': 'evaluator/test_rate.py',
        'workspace_entry': declared,
        'reference_repair': {'path': 'workspace/config.json', 'content': REPAIR},
        'variant': {'expected_value': 0.25},
    }
    (case_root / 'case.json').write_text(json.dumps(case, indent=2))
    return FixtureWorkspace(case_root)


def make_binding(tools, **changes):
    values = {'task_id': tools.case_id, 'subject': 'subject-1', 'branch': 'e3',
              'purpose': 'test', 'source_epoch': 0, 'goal_digest': fingerprint(tools.goal),
              'component_versions': {'runtime': 'v1'}}
    values.update(changes)
    return values


def make_state(tools, *, budget=None, **changes):
    values = dict(
        task_id=tools.case_id, subject='subject-1', branch='e3', purpose='test',
        source_epoch=0, goal=tools.goal,
        success_conditions=(SuccessCondition('sc-1', 'evaluator exits 0'),),
        constraints=(), subgoals=(), claims=(), evidence=(), obligations=(), actions=(),
        outcomes=(), notes=(), budget=budget or Budget(8, 16, 4096, 512),
        binding={'runtime': 'v1'}, allowed_tools=tuple(tools.allowed_tools),
        revision=0)
    values.update(changes)
    return ExactTaskState(**values)


def make_runtime(tools, tmp_path, *, workspace_store=None, mode='full',
                 state_changes=None, **runtime_changes):
    journal = TaskJournal.open(tmp_path / 'task.sqlite', make_binding(tools))
    runtime = TaskRuntime(tools, journal, workspace_store=workspace_store, mode=mode,
                          **runtime_changes)
    if not journal.initialize_done():
        runtime.initialize(make_state(tools, **(state_changes or {})))
    return runtime


def output(raw, *, workspace=None, prompt_tokens=11, token_ids=(7, 8),
           wall_seconds=0.0, notes=()):
    return ProposalOutput(
        raw_text=raw, prompt_tokens=prompt_tokens,
        generated_token_ids=tuple(token_ids), generated_tokens=len(token_ids),
        frame_digest='frame-' + str(len(raw)), workspace=workspace,
        workspace_stamp=workspace.stamp() if workspace is not None else None,
        wall_seconds=wall_seconds, notes=tuple(notes))


def workspace_for(request, *, width=4, slots=2, producer='producer-1'):
    generator = torch.Generator().manual_seed(3)
    return CoreState(
        torch.randn(1, slots, width, generator=generator),
        torch.ones(1, slots, dtype=torch.bool),
        torch.zeros(1, 24), producer, request.task_digest, request.revision, 2)


class ScriptedProposer:
    """Records every request and returns scripted raw proposals in order."""

    def __init__(self, raws, *, workspace_factory=None):
        self.raws = list(raws)
        self.workspace_factory = workspace_factory
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        raw = self.raws.pop(0) if self.raws else '{"kind": "stop"}'
        workspace = self.workspace_factory(request) if self.workspace_factory else None
        return output(raw, workspace=workspace)

    @property
    def prompts(self):
        return [request.prompt_text for request in self.requests]


def write_action(path='workspace/config.json', content=REPAIR):
    return json.dumps({'kind': 'invoke_tool', 'tool': 'write_file',
                       'arguments': {'path': path, 'content': content}})


def read_action(path='workspace/config.json'):
    return json.dumps({'kind': 'read', 'path': path})


def run_test_action():
    return json.dumps({'kind': 'invoke_tool', 'tool': 'run_test', 'arguments': {}})
