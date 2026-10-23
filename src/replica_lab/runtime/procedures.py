"""Host-executed procedure candidate from verified traces (E5 baseline).

This is deliberately not a neural policy and does not select the repair on
behalf of a model. It compiles a bounded read/test/set_constant/test procedure
from verified train traces, executes it as a host tool sequence on new cases,
and exposes the results for independent evaluation. It is a procedure-library
baseline for comparing model-selected behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .artifacts import ProcedureArtifact, ProcedureStep

FAILURE_BOUND = re.compile(
    r'\b(?P<name>[A-Z][A-Z0-9_]{1,63})\s*=\s*(?P<actual>-?\d+(?:\.\d+)?)\s*'
    r'(?:but policy requires|needs)\s*(?P=name)\s*>=\s*(?P<expected>-?\d+(?:\.\d+)?)')


def parse_failure_bound(text: str):
    """Extract the constant/bound disclosed by an actual evaluator failure."""
    if not isinstance(text, str):
        raise ValueError('failure text required')
    match = FAILURE_BOUND.search(text)
    if match is None:
        raise ValueError('no bounded constant failure was disclosed')
    return {'name': match.group('name'), 'actual': match.group('actual'),
            'expected': match.group('expected')}


@dataclass(frozen=True)
class ProcedureRun:
    status: str
    receipts: tuple[dict, ...]
    parsed: dict | None = None
    final_exit_code: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class RepairProcedure:
    procedure_id: str
    workspace_entry: str
    source_event_ids: tuple[str, ...]
    version: int = 1
    preconditions: tuple[str, ...] = ('the evaluator fails with a disclosed numeric bound',)
    tests: tuple[str, ...] = ('the independent evaluator exits zero',)

    def __post_init__(self):
        for name in ('procedure_id', 'workspace_entry'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f'{name} required')
        if type(self.version) is not int or self.version < 1:
            raise ValueError('procedure version must be positive')
        if not self.source_event_ids:
            raise ValueError('procedure requires verified source events')
        if not self.tests:
            raise ValueError('procedure requires independent tests')

    def artifact(self) -> ProcedureArtifact:
        steps = (ProcedureStep('read_file', {'path': self.workspace_entry}),
                 ProcedureStep('run_test', {}),
                 ProcedureStep('set_constant', {'name': 'from_failure',
                                                'value': 'from_failure'}),
                 ProcedureStep('run_test', {}))
        return ProcedureArtifact(
            f'artifact-{self.procedure_id}', self.procedure_id, self.version,
            self.preconditions, steps, self.tests, self.source_event_ids,
            {'procedure': 'repair-config.v1'}, status='proposed')

    def execute(self, tools) -> ProcedureRun:
        """Execute the fixed host sequence; returns actual receipts only."""
        receipts = []
        read = tools.read_file({'path': self.workspace_entry})
        receipts.append(read)
        if read.get('status') != 'ok':
            return ProcedureRun('failed', tuple(receipts), error='read_failed')
        evaluated = tools.run_test({})
        receipts.append(evaluated)
        receipt = evaluated.get('receipt') or {}
        if receipt.get('exit_code') == 0:
            return ProcedureRun('not_required', tuple(receipts),
                                final_exit_code=0)
        try:
            parsed = parse_failure_bound(
                (receipt.get('stderr_excerpt') or '') + '\n' +
                (receipt.get('stdout_excerpt') or ''))
        except ValueError as error:
            return ProcedureRun('failed', tuple(receipts), error=str(error))
        changed = tools.set_constant({'name': parsed['name'],
                                      'value': float(parsed['expected'])})
        receipts.append(changed)
        if changed.get('status') != 'ok':
            return ProcedureRun('failed', tuple(receipts), parsed=parsed,
                                error=changed.get('error', 'set_constant_failed'))
        verified = tools.run_test({})
        receipts.append(verified)
        final = (verified.get('receipt') or {}).get('exit_code')
        return ProcedureRun('succeeded' if final == 0 else 'failed',
                            tuple(receipts), parsed=parsed, final_exit_code=final,
                            error=None if final == 0 else 'evaluator_failed')
