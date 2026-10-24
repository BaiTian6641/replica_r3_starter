#!/usr/bin/env python3
"""Create a fresh independent memory-reliability dataset; never overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.memory_reliability import build_episodes, audit_dataset

FAMILY_PLAN = [('train', range(1000, 1008)), ('dev', range(1100, 1104)),
               ('heldout', range(1200, 1204))]
VALUES = {
    'train': ('cobalt', 'saffron', 'teal', 'violet', 'amber', 'copper'),
    'dev': ('magenta', 'cerulean', 'orange', 'green', 'purple', 'yellow'),
    'heldout': ('maroon', 'navy', 'onyx', 'gold', 'crimson', 'indigo'),
}

DISTRACTOR_VALUES = {
    'train': ('silver', 'bronze'),
    'dev': ('silver', 'bronze'),
    'heldout': ('obsidian', 'quartz'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh dataset directory')
    episodes = build_episodes(seed=args.seed, family_plan=FAMILY_PLAN,
                              values_by_split=VALUES, query_forms=True,
                              distractor_values_by_split=DISTRACTOR_VALUES)
    audit = audit_dataset(episodes, require_novel_values=True)
    args.output.mkdir(parents=True)
    path = args.output / 'episodes.json'
    path.write_text(json.dumps(episodes, indent=2) + '\n')
    manifest = {
        'format': 'replica.memory_reliability.dataset.v2', 'seed': args.seed,
        'episodes_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'audit': audit,
        'family_plan': [(split, list(families)) for split, families in FAMILY_PLAN],
        'values_by_split': VALUES,
        'distractor_values_by_split': DISTRACTOR_VALUES,
        'query_forms': True,
        'novel_values_enforced': True,
        'scope': 'Fresh bounded synthetic diagnostic; complete counterfactual families, no sealed broad-capability audit',
        'split_rule': 'Complete counterfactual family and both variants remain in one split; no evaluation training',
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
