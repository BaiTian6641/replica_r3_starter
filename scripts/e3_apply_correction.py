#!/usr/bin/env python3
"""Host-only correction for a persisted E3 task journal (CPU, no model).

Appends one authorized correction event, which bumps the exact task revision.
A workspace persisted at the old revision must be rejected and rebuilt by the
next native process. This script runs no neural code and executes no tools.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def component_versions():
    return {"runtime": "e3-runtime-v1", "tools": "fixture-workspace-v1",
            "journal": "exact-journal-v1", "task_features": "e3-layout-v1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--workspace-dir', type=Path, required=True)
    parser.add_argument('--note', default=None)
    parser.add_argument('--constraint-id', default=None)
    parser.add_argument('--constraint-text', default=None)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)

    from replica_lab.architecture.contracts import fingerprint
    from replica_lab.runtime.journal import TaskJournal
    from replica_lab.runtime.loop import TaskRuntime
    from replica_lab.runtime.tools import FixtureWorkspace
    from replica_lab.runtime.workspace_store import WorkspaceStore

    tools = FixtureWorkspace(args.case)
    # Reuse the recorded binding verbatim: the native CLI adds its resolved
    # budget digest to component_versions, and the journal is the authority for
    # the exact task identity. Validate the identity fields we depend on.
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
    expected = {'task_id': tools.case_id, 'subject': tools.case_id,
                'branch': 'main', 'purpose': 'bounded_repair', 'source_epoch': 0,
                'goal_digest': fingerprint(tools.goal)}
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError('Journal identity does not match this case/branch/purpose')
    journal = TaskJournal.open(args.journal, binding)
    try:
        before = journal.load()
        constraint = None
        if args.constraint_id or args.constraint_text:
            if not args.constraint_id or not args.constraint_text:
                raise ValueError('Both --constraint-id and --constraint-text are required')
            constraint = {'constraint_id': args.constraint_id, 'text': args.constraint_text}
        if args.note is None and constraint is None:
            raise ValueError('A note or constraint is required')
        runtime = TaskRuntime(tools, journal, workspace_store=WorkspaceStore(args.workspace_dir))
        state = runtime.correct(note=args.note, constraint=constraint)
        stored = WorkspaceStore(args.workspace_dir).load()
        report = {
            'status': 'corrected',
            'corrected_at_utc': datetime.now(timezone.utc).isoformat(),
            'journal': str(args.journal.resolve()),
            'head_before': before.head,
            'head_after': journal.head,
            'revision_before': before.state.revision,
            'revision_after': state.revision,
            'goal_unchanged': state.goal == before.state.goal,
            'success_conditions_unchanged': [
                c.to_dict() for c in state.success_conditions],
            'stored_workspace_revision': None if stored is None else stored.revision,
            'stored_workspace_digest_matches': bool(
                stored is not None and stored.task_digest == state.digest),
            'note': args.note,
            'constraint': constraint,
            'model_loaded': False,
            'tool_calls': 0,
        }
    finally:
        journal.close()
    (args.output / 'correction.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'revision': report['revision_after'],
                      'stale_workspace': not report['stored_workspace_digest_matches']}))


if __name__ == '__main__':
    main()
