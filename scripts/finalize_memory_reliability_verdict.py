#!/usr/bin/env python3
"""Finalize a reliability verdict after a scorer-only correction.

The raw native reports are never edited. This utility authenticates those reports,
then links a temporary re-pinned scorer artifact back to the original protocol,
guards and implementation without changing the measured decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_metadata(path):
    guard = json.loads(path.read_text())
    return {'path': str(path), 'sha256': sha(path), 'policy': guard['guard']['policy']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--guard', type=Path, required=True)
    parser.add_argument('--restart-run', type=Path, required=True)
    parser.add_argument('--restart-guard', type=Path, required=True)
    parser.add_argument('--smoke-verdict', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh finalized verdict path')
    protocol = json.loads(args.protocol.read_text())
    run = json.loads((args.run / 'run_report.json').read_text())
    restart = json.loads((args.restart_run / 'run_report.json').read_text())
    smoke = json.loads(args.smoke_verdict.read_text())
    protocol_sha = sha(args.protocol)
    if (run['protocol_sha256'] != protocol_sha or restart['protocol_sha256'] != protocol_sha
            or run['implementation_sha256'] != protocol['implementation_sha256']
            or run['status'] != 'completed' or restart['status'] != 'completed'):
        raise ValueError('Native protocol or implementation lineage differs')
    if not smoke.get('report_sha256') or not smoke['restart'].get('report_sha256'):
        raise ValueError('Scorer smoke is missing report hashes')
    if smoke['guard']['policy'] != guard_metadata(args.guard)['policy']:
        raise ValueError('Scorer smoke guard policy differs')
    if smoke['restart']['guard']['policy'] != guard_metadata(args.restart_guard)['policy']:
        raise ValueError('Scorer smoke restart guard policy differs')
    finalized = dict(smoke)
    finalized.update(
        protocol_sha256=protocol_sha,
        implementation_sha256=protocol['implementation_sha256'],
        guard=guard_metadata(args.guard),
        restart=dict(smoke['restart'], guard=guard_metadata(args.restart_guard)),
        metadata_correction={
            'reason': 'Restored args=parser.parse_args() before output-path validation in the independent scorer.',
            'scorer_sha256': sha(ROOT / 'scripts/score_memory_reliability.py'),
            'scorer_smoke': str(args.smoke_verdict),
            'scorer_smoke_sha256': sha(args.smoke_verdict),
            'decision_unchanged': True,
            'corrected_at_utc': datetime.now(timezone.utc).isoformat(),
            'finalizer_script_sha256': sha(Path(__file__))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finalized, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'decision': finalized['decision'],
                      'reference_decision': finalized['reference_decision'],
                      'candidate_decision': finalized['candidate_decision']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
