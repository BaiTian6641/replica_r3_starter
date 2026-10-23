#!/usr/bin/env python3
"""Generate a 16-unit E2 capacity bank with 120-128 token source spans.

The workload is synthetic and used only to measure the existing memory path.
Twelve records occupy one 128-token source chunk each; one long record occupies
four chunks, giving exactly sixteen selectable units. No answer labels or task
semantics are implied.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/e2_capacity'
SUBJECT = 'e2_capacity'
PURPOSE = 'capacity_probe'
BRANCH = 'capacity'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_short(tokenizer, seed):
    sentences = [
        f'Capacity source record {seed} describes a routine configuration note.',
        'It records scheduling, maintenance, review, documentation and audit details.',
        'The note is synthetic and has no external provenance or personal data.',
        'Each statement is deliberately ordinary so the workload is reproducible.',
        'No answer or task label is attached to this capacity-only fixture.',
        'The record remains within one declared 128-token source chunk.',
        'Operators use the note only to exercise the source encoder and writer.',
        'The text contains no instruction to the model and no hidden answer.',
        'Retention is limited to the declared capacity experiment directory.',
        'The same structure is repeated for each short capacity record.',
        'The measurement records actual token counts rather than configured maxima.',
        'A final sentence keeps the record inside the intended token band.',
    ]
    text = ' '.join(sentences)
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if len(ids) > 128:
        text = tokenizer.decode(ids[:128], skip_special_tokens=True)
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
    while len(ids) < 120:
        text += ' Additional synthetic capacity filler text.'
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if not 120 <= len(ids) <= 128:
        raise ValueError(f'short record {seed} is {len(ids)} tokens, expected 120..128')
    return text, len(ids)


def build_long(tokenizer):
    sentence = ('The capacity handbook contains routine notes about scheduling, reviews, '
                'maintenance, documentation, audits, training, and recovery procedures. ')
    text = sentence * 22
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if len(ids) > 4 * 128:
        text = tokenizer.decode(ids[:4 * 128], skip_special_tokens=True)
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if not 3 * 128 + 1 <= len(ids) <= 4 * 128:
        raise ValueError(f'long capacity record is {len(ids)} tokens, expected four chunks')
    return text, len(ids)


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / 'checkpoints/minicpm5_source', local_files_only=True,
        trust_remote_code=False)
    records = []
    token_counts = {}
    for index in range(1, 13):
        text, count = fit_short(tokenizer, index)
        record_id = f'cap_short_{index:02d}'
        records.append({'record_id': record_id, 'subject_id': SUBJECT,
                        'branch_id': BRANCH, 'text': text, 'observed_at': 1,
                        'event_at': 0, 'purpose': PURPOSE, 'kind': 'source',
                        'supersedes': None, 'source_id': 'capacity_fixture',
                        'authenticity': 'unverified'})
        token_counts[record_id] = count
    long_text, long_count = build_long(tokenizer)
    records.append({'record_id': 'cap_long', 'subject_id': SUBJECT,
                    'branch_id': BRANCH, 'text': long_text, 'observed_at': 1,
                    'event_at': 0, 'purpose': PURPOSE, 'kind': 'source',
                    'supersedes': None, 'source_id': 'capacity_fixture',
                    'authenticity': 'unverified'})
    token_counts['cap_long'] = long_count
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'records.jsonl').open('w', encoding='utf-8') as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
    manifest = {
        'format': 'replica.e2_capacity.dataset.v1',
        'created_at_utc': '2026-10-23',
        'subject_id': SUBJECT, 'branch_id': BRANCH, 'purpose': PURPOSE, 'as_of': 10,
        'records': len(records), 'units': 16,
        'short_records': 12, 'long_record_spans': (long_count + 127) // 128,
        'token_counts': token_counts,
        'files': {'records_sha256': sha(OUT / 'records.jsonl')},
        'claims': 'Synthetic capacity bank; no task labels or quality claim.',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'records': len(records), 'units': 16,
                      'long_tokens': long_count,
                      'short_tokens': [token_counts[f'cap_short_{i:02d}'] for i in range(1, 13)]}))


if __name__ == '__main__':
    main()
