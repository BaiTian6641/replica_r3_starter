#!/usr/bin/env python3
"""Validate the E2 retrieval dataset, splits, eligibility and chunk metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/e2_retrieval')
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    if manifest['format'] != 'replica.e2_retrieval.dataset.v1':
        raise SystemExit('unexpected dataset format')
    if sha(args.root / 'records.jsonl') != manifest['files']['records_sha256']:
        raise SystemExit('records hash mismatch')
    if sha(args.root / 'queries.jsonl') != manifest['files']['queries_sha256']:
        raise SystemExit('queries hash mismatch')
    records = [json.loads(line) for line in
               (args.root / 'records.jsonl').read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in
               (args.root / 'queries.jsonl').read_text().splitlines() if line.strip()]
    by_id = {item['record_id']: item for item in records}
    if len(by_id) != len(records):
        raise SystemExit('duplicate record id')
    groups = {q['source_group']: q['split'] for q in queries}
    train = {g for g, split in groups.items() if split == 'train'}
    dev = {g for g, split in groups.items() if split == 'dev'}
    held = {g for g, split in groups.items() if split == 'heldout'}
    if train & dev or train & held or dev & held:
        raise SystemExit('source groups are not disjoint')
    if not train or not dev or not held:
        raise SystemExit('all three splits are required')
    from replica_lab.memory_system import EligibilityScope, MemoryRecord, RecordLedger
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        ledger = RecordLedger(Path(directory) / 'data.sqlite')
        try:
            for item in records:
                ledger.admit(MemoryRecord(**item), authorized_subject=item['subject_id'],
                             authorized_branch=item['branch_id'], source_authorized=False)
            for query in queries:
                scope = EligibilityScope(query['subject_id'], query['branch_id'],
                                         query['purpose'], query['as_of'],
                                         'assistant_for_source', None)
                eligible = {r.record_id for r in ledger.snapshot(scope).records}
                expected = query['expected_record_id']
                forbidden = set(query['forbidden_record_ids'])
                if expected is not None and expected not in eligible:
                    raise SystemExit(f'{query["case_id"]}: expected record not eligible')
                if forbidden & eligible:
                    raise SystemExit(f'{query["case_id"]}: forbidden record eligible: '
                                     f'{sorted(forbidden & eligible)}')
                if expected is not None and expected in forbidden:
                    raise SystemExit(f'{query["case_id"]}: expected record is forbidden')
                if query['expected_answer'] != 'UNKNOWN':
                    text = query['query'].casefold()
                    answer = str(query['expected_answer']).casefold()
                    if answer in text:
                        raise SystemExit(f'{query["case_id"]}: answer leaked in query')
                checks.append({'case_id': query['case_id'], 'split': query['split'],
                               'eligible': sorted(eligible),
                               'expected': expected, 'forbidden_absent': True})
        finally:
            ledger.close()
    long_meta = manifest['long_record']
    long_record = by_id[long_meta['record_id']]
    if not 129 <= long_meta['tokens'] <= 256:
        raise SystemExit('long record does not span two or three 128-token chunks')
    if long_meta['fact_token_start'] < 128:
        raise SystemExit('long-record fact starts in the first chunk')
    if long_meta['fact_token_start'] >= long_meta['tokens']:
        raise SystemExit('long-record fact offset is invalid')
    if int(long_meta['tokens']) != len(long_record['text']):
        pass  # token count, not character count; retained in the manifest
    report = {'format': 'replica.e2_retrieval.dataset.validation.v1',
              'ok': True, 'cases': len(queries), 'records': len(records),
              'splits': {'train': sorted(train), 'dev': sorted(dev),
                         'heldout': sorted(held)},
              'long_record': long_meta, 'checks': checks}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
