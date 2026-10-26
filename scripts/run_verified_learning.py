#!/usr/bin/env python3
"""Prepare and exercise the CPU verified-learning procedure contract.

This command never loads a native model, never runs a native job and never
claims verified learning or RSI.  ``--prepare`` writes deterministic worlds and
protocol pins; ``--verify`` rechecks those pins; ``--fixture`` runs the tiny CPU
objective/control/policy plumbing and writes a procedure-only report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from replica_lab.verified_learning import (  # noqa: E402
    CONFIG_FORMAT,
    FIXTURE_FORMAT,
    _worlds_from_config,
    build_protocol,
    load_config,
    prepare_verified_learning,
    run_fixture,
    verify_prepared,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + '\n')


def fresh_output(path, parser):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        parser.error(f'Use a fresh output path: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'configs/verified_learning_v1.json')
    parser.add_argument('--prepare', action='store_true',
                        help='write deterministic worlds/protocol/manifest')
    parser.add_argument('--verify', action='store_true',
                        help='recompute prepared pins')
    parser.add_argument('--fixture', action='store_true',
                        help='run the tiny CPU procedure fixture')
    parser.add_argument('--hash-protocol', action='store_true',
                        help='print the protocol hash without writing')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'data/verified_learning_20261025')
    parser.add_argument('--prepared', type=Path,
                        default=ROOT / 'data/verified_learning_20261025')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.device != 'cpu':
        parser.error('native execution is separately gated; this script is CPU-only')
    modes = [args.prepare, args.verify, args.fixture, args.hash_protocol]
    if sum(bool(value) for value in modes) != 1:
        parser.error('choose exactly one of --prepare, --verify, --fixture or --hash-protocol')
    config = load_config(args.config)
    if config.get('format') != CONFIG_FORMAT:
        parser.error(f'configuration format must be {CONFIG_FORMAT}')

    if args.prepare:
        output = fresh_output(args.output, parser)
        manifest = prepare_verified_learning(
            output, config_path=args.config.resolve(), config=config)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.verify:
        result = verify_prepared(args.prepared)
        print(json.dumps({'protocol_sha256': result['protocol_sha256'],
                          'manifest': result['manifest'],
                          'status': 'verified'}, indent=2, sort_keys=True))
        return 0
    if args.fixture:
        output = fresh_output(args.output, parser)
        report = run_fixture(config, device=args.device)
        if report.get('format') != FIXTURE_FORMAT:
            raise SystemExit('fixture report format mismatch')
        dump(output / 'report.json', report)
        print(json.dumps({
            'format': report['format'],
            'protocol_sha256': report['protocol_sha256'],
            'status': report['status'],
            'claims': report['claims'],
            'rule_gate': report['rule_learning']['gate']['decision'],
            'rule_gate_reason': report['rule_learning']['gate']['reason'],
            'policy_decision': report['policy']['comparison']['decision'],
            'native_evaluation': report['native_evaluation'],
            'report': str(output / 'report.json'),
        }, indent=2, sort_keys=True))
        return 0
    protocol = build_protocol(_worlds_from_config(config), config)
    print(json.dumps({'protocol_sha256': protocol['protocol_sha256'],
                      'observation_sha256':
                          protocol['worlds']['observation_sha256']},
                     indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
