#!/usr/bin/env python3
"""Collect the interrupted-task fresh-process acceptance evidence (CPU only).

Reads the two guarded CLI reports, the correction receipt and the persisted
journal, then verifies that the failure was observed, the correction bumped the
revision, the second process rebuilt incompatible workspace state, the repair
landed once, and only the independent evaluator declared success.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

COMPONENT_VERSIONS = {'runtime': 'e3-runtime-v1', 'tools': 'fixture-workspace-v1',
                      'journal': 'exact-journal-v1', 'task_features': 'e3-layout-v1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt_exit(item):
    outcome = item.get('outcome') or {}
    result = outcome.get('result') or {}
    receipt = result.get('receipt') if isinstance(result, dict) else None
    return (receipt or {}).get('exit_code') if isinstance(receipt, dict) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run1-report', type=Path, required=True)
    parser.add_argument('--run2-report', type=Path, required=True)
    parser.add_argument('--correction', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--original-case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run1 = json.loads(args.run1_report.read_text())
    run2 = json.loads(args.run2_report.read_text())
    correction = json.loads(args.correction.read_text())

    from replica_lab.architecture.contracts import fingerprint
    from replica_lab.runtime.journal import TaskJournal
    from replica_lab.runtime.tools import FixtureWorkspace
    tools = FixtureWorkspace(args.case)
    import sqlite3
    connection = sqlite3.connect(str(args.journal))
    try:
        row = connection.execute(
            "SELECT payload FROM events WHERE kind='binding' ORDER BY seq LIMIT 1").fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError('Journal has no binding event')
    binding = json.loads(row[0])['binding']
    expected = {'task_id': tools.case_id, 'subject': tools.case_id, 'branch': 'main',
                'purpose': 'bounded_repair', 'source_epoch': 0,
                'goal_digest': fingerprint(tools.goal)}
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError('Journal identity does not match this case/branch/purpose')
    journal = TaskJournal.open(args.journal, binding)
    try:
        snapshot = journal.load()
        state = snapshot.state
    finally:
        journal.close()

    trace1 = run1.get('summary', {}).get('trace', [])
    trace2 = run2.get('summary', {}).get('trace', [])
    def tool_name(item):
        action = item.get('action') or {}
        arguments = action.get('args') or {}
        return arguments.get('tool')

    failed_test1 = any(receipt_exit(item) not in (None, 0) for item in trace1
                       if tool_name(item) == 'run_test')
    passed_test2 = any(receipt_exit(item) == 0 for item in trace2
                       if tool_name(item) == 'run_test')
    write_actions = [action for action in state.actions
                     if action.tool == 'write_file' and action.state == 'completed']
    workspace = args.case / 'workspace' / 'app_settings.py'
    original_case = json.loads((args.original_case / 'case.json').read_text())
    repair = original_case['reference_repair']
    repaired = (workspace.read_text(encoding='utf-8') == repair['content'])
    evaluator_untouched = (sha(args.case / original_case['evaluator'])
                           == sha(args.original_case / original_case['evaluator']))
    notes2 = [note for item in trace2 for note in (item.get('notes') or [])]
    conditions = list(state.satisfied_conditions(snapshot.verifier))
    checks = {
        'run1_observed_failing_test': failed_test1,
        'run1_incomplete': run1.get('summary', {}).get('success') is False,
        'run1_stopped_by_process_step_limit': len(trace1) == 2,
        'correction_bumped_revision': correction['revision_after'] > correction['revision_before'],
        'correction_preserved_goal': correction['goal_unchanged'],
        'stored_workspace_stale_after_correction':
            correction['stored_workspace_digest_matches'] is False,
        'run2_completed': run2.get('summary', {}).get('success') is True,
        'run2_saw_independent_pass': passed_test2,
        'run2_rebuilt_incompatible_workspace':
            any('workspace_discarded' in note for note in notes2),
        'repair_landed_exactly_once': len(write_actions) == 1,
        'workspace_matches_reference_repair': repaired,
        'evaluator_unmodified': evaluator_untouched,
        'success_condition_satisfied': bool(conditions),
        'run2_final_state_complete': run2.get('summary', {}).get('success') is True,
    }
    report = {
        'status': 'passed' if all(checks.values()) else 'failed',
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'checks': checks,
        'run1': {'success': run1.get('summary', {}).get('success'),
                 'decisions': len(trace1),
                 'prompt_tokens': run1.get('costs', {}).get('prompt_tokens'),
                 'generated_tokens': run1.get('costs', {}).get('generated_tokens'),
                 'tool_calls': run1.get('costs', {}).get('tool_calls')},
        'correction': correction,
        'run2': {'success': run2.get('summary', {}).get('success'),
                 'decisions': len(trace2),
                 'prompt_tokens': run2.get('costs', {}).get('prompt_tokens'),
                 'generated_tokens': run2.get('costs', {}).get('generated_tokens'),
                 'tool_calls': run2.get('costs', {}).get('tool_calls'),
                 'workspace_notes': notes2},
        'final_state': {'revision': state.revision, 'digest': state.digest,
                        'satisfied_conditions': conditions,
                        'completed_write_actions': [a.action_id for a in write_actions]},
        'journal': {'path': str(args.journal.resolve()), 'head': snapshot.head,
                    'events': snapshot.event_count},
        'hashes': {'workspace': sha(workspace),
                   'evaluator': sha(args.case / original_case['evaluator'])},
        'claims': 'Engineering restart evidence on one exposed fixture; no generalization claim.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'checks': checks}))


if __name__ == '__main__':
    main()
