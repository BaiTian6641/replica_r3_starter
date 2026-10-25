#!/usr/bin/env python3
"""Create the prospective fresh-family, diverse-retention dataset; never overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.memory_reliability import GENERAL_V2, build_episodes, audit_dataset

FAMILY_PLAN = [('train', range(2000, 2008)), ('dev', range(2100, 2104)),
               ('heldout', range(2200, 2204))]
VALUES = {
    'train': ('cobalt', 'saffron', 'teal', 'violet', 'amber', 'copper'),
    'dev': ('magenta', 'cerulean', 'orange', 'green', 'purple', 'yellow'),
    'heldout': ('plum', 'azure', 'ochre', 'chartreuse', 'burgundy', 'slate'),
}
DISTRACTOR_VALUES = {
    'train': ('silver', 'bronze'),
    'dev': ('silver', 'bronze'),
    'heldout': ('pewter', 'alabaster'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261026)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh dataset directory')
    episodes = build_episodes(seed=args.seed, family_plan=FAMILY_PLAN,
                              values_by_split=VALUES, query_forms=True,
                              distractor_values_by_split=DISTRACTOR_VALUES,
                              general_by_split='general_v2')
    audit = audit_dataset(episodes, require_novel_values=True,
                          general_oracle='general_v2', min_unique_general=16)
    args.output.mkdir(parents=True)
    path = args.output / 'episodes.json'
    path.write_text(json.dumps(episodes, indent=2) + '\n')
    manifest = {
        'format': 'replica.memory_reliability.dataset.v4', 'seed': args.seed,
        'episodes_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'audit': audit,
        'family_plan': [(split, list(families)) for split, families in FAMILY_PLAN],
        'values_by_split': VALUES,
        'distractor_values_by_split': DISTRACTOR_VALUES,
        'query_forms': True,
        'general_oracle': 'general_v2',
        'general_pool_sizes': {split: len(pool) for split, pool in GENERAL_V2.items()},
        'min_unique_general': 16,
        'novel_values_enforced': True,
        'scope': 'Fresh bounded synthetic diagnostic; diverse retention and counterfactual families, no sealed broad-capability audit',
        'split_rule': 'Complete counterfactual family and both variants remain in one split; no evaluation training',
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
