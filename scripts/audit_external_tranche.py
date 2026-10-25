#!/usr/bin/env python3
"""Audit a small pinned external tranche without acquiring assets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from replica_lab.dataset_intake import audit  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path,
                        default=ROOT / 'docs/datasets/20261024_Source_Catalog.json')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'reports/external_intake_20261025/audit.json')
    args = parser.parse_args(argv)
    record = audit(args.catalog, args.output)
    print(json.dumps({key: record[key] for key in (
        'catalog_sha256', 'catalogued_external_sources', 'pinned_external_sources',
        'assumption_checks')} | {'stage_counts': record['tranche']['stage_counts']},
        indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
