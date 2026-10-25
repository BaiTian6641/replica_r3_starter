#!/usr/bin/env python3
"""Create a fresh counterfactual multi-query/correction dataset; never overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.memory_reliability import build_episodes, audit_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh dataset directory')
    episodes = build_episodes()
    audit = audit_dataset(episodes)
    args.output.mkdir(parents=True)
    path = args.output / 'episodes.json'
    path.write_text(json.dumps(episodes, indent=2) + '\n')
    manifest = {'format': 'replica.memory_reliability.dataset.v1', 'seed': 20261024,
                'episodes_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'audit': audit,
                'scope': 'Fresh bounded synthetic diagnostic, not a sealed broad-capability audit',
                'split_rule': 'Complete counterfactual family remains in one split; no evaluation training'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
