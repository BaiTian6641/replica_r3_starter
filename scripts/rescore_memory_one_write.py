#!/usr/bin/env python3
"""Independently rescore one-write raw artifacts on CPU; never reruns the model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from replica_lab.memory_one_write import score_one_write  # noqa: E402
from replica_lab.runtime.persistent_fast_memory import read_persistent_manifest  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Use a fresh score output path')
    protocol = json.loads((args.prepared / 'protocol.json').read_text())
    report = json.loads((args.run / 'run_report.json').read_text())
    rows = [json.loads(line) for line in
            (args.run / 'readouts.jsonl').read_text().splitlines() if line.strip()]
    codebooks = json.loads((args.run / 'codebooks.json').read_text())['codebooks']
    assert sha(args.run / 'readouts.jsonl') == report['readouts_sha256']
    assert sha(args.run / 'codebooks.json') == report['codebooks_sha256']
    assert len(rows) == protocol['planned_workload']['planned_rows']
    mechanical = {'frozen': {
        'optimizer_updates': report['optimizer_updates'],
        'parameters_updated': report['parameters_updated'],
        'parameter_versions_equal': report['frozen']['parameter_versions_equal'],
        'requires_grad_all_false': report['frozen']['requires_grad_all_false']}}
    scored = score_one_write(rows, codebooks,
                             thresholds=protocol['thresholds'], mechanical=mechanical)
    manifests = [path for path in (args.run / 'images').rglob('persistent_manifest.json')]
    for path in manifests:
        read_persistent_manifest(path.parent)
    score = {
        'format': 'replica.memory_one_write_independent_score.v1',
        'native_report_sha256': sha(args.run / 'run_report.json'),
        'protocol_sha256': sha(args.prepared / 'protocol.json'),
        'data_sha256': sha(args.prepared / 'data.json'),
        'evaluated_rows': len(rows),
        'unique_cases': len(rows) // 2,
        'state_images_authenticated': len(manifests),
        'independent_rescore': 'matched' if (
            scored['summary'] == report['summary']
            and scored['gates'] == report['gates']
            and scored['decision'] == report['decision']) else 'differs',
        **scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(score, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: score[key] for key in (
        'independent_rescore', 'decision', 'failed_gates', 'evaluated_rows',
        'unique_cases', 'state_images_authenticated')}, indent=2))
    return 0 if score['independent_rescore'] == 'matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
