#!/usr/bin/env python3
"""Generate the source-separated E2 retrieval/novel-record development set.

All records are synthetic and authored for this diagnostic. Groups are split by
source group: train groups train the reranker; dev groups are development; the
held-out groups are used once for the final assessment. Corrupted, future,
unauthorized and wrong-purpose records exist only to test eligibility filtering.
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
OUT = ROOT / 'data/e2_retrieval'
SUBJECT = 'e2_retrieval'
PURPOSE = 'retrieval_eval'
AS_OF = 10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(case, record_id, text, *, observed=1, supersedes=None, purpose=None,
           branch=None, group=None):
    return {
        'record_id': f'{case}_{record_id}', 'subject_id': SUBJECT,
        'branch_id': branch or case, 'text': text,
        'observed_at': observed, 'event_at': 0,
        'purpose': purpose or PURPOSE, 'kind': 'source',
        'supersedes': f'{case}_{supersedes}' if supersedes else None,
        'source_id': group or case, 'authenticity': 'source_report',
    }


def case(case_id, group, split, kind, query, expected_record, expected_answer,
         records, forbidden=(), answer_token_start=None):
    return {
        'case_id': case_id, 'source_group': group, 'split': split, 'kind': kind,
        'query': query, 'expected_record_id': expected_record,
        'expected_answer': expected_answer,
        'forbidden_record_ids': list(forbidden),
        'answer_token_start': answer_token_start,
        'subject_id': SUBJECT, 'branch_id': case_id, 'purpose': PURPOSE,
        'as_of': AS_OF, 'records': list(records),
    }


def build_long_record(tokenizer, case_id, group):
    filler = ('The Juniper program maintains routine operational notes about scheduling, '
              'reviews, maintenance, documentation, audits, and training. ')
    fact = 'The launch code for project Juniper is quartz. '
    text = filler * 8 + fact + filler * 2
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if not 129 <= len(ids) <= 256:
        raise ValueError(f'Long record must span 2-3 chunks, got {len(ids)} tokens')
    # The fact must begin in the second 128-token chunk, not the first.
    prefix = tokenizer(filler * 8, add_special_tokens=False)['input_ids']
    start = len(prefix)
    if start < 128:
        raise ValueError('Long-record fact must start after the first source chunk')
    return text, start, len(ids)


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / 'checkpoints/minicpm5_source', local_files_only=True,
        trust_remote_code=False)
    long_text, long_start, long_tokens = build_long_record(
        tokenizer, 'heldout_long', 'heldout_d')
    cases = [
        case('train_a1', 'train_a', 'train', 'relevant_among_distractors',
             'What is the launch code for project Aster?', 'train_a1_launch', 'amber',
             [
                 record('train_a1', 'launch', 'The launch code for project Aster is amber.'),
                 record('train_a1', 'review', 'The review code for project Aster is blue.'),
                 record('train_a1', 'owner', 'The owner of project Aster is Dana.'),
                 record('train_a1', 'other', 'The launch code for project Birch is green.'),
             ]),
        case('train_a2', 'train_a', 'train', 'same_entity_different_attribute',
             'What is the backup window for project Aster?', 'train_a2_backup', 'midnight',
             [
                 record('train_a2', 'backup', 'The backup window for project Aster is midnight.'),
                 record('train_a2', 'launch', 'The launch code for project Aster is amber.'),
                 record('train_a2', 'target', 'The target date for project Aster is Friday.'),
                 record('train_a2', 'other', 'The backup window for project Birch is noon.'),
             ]),
        case('train_b1', 'train_b', 'train', 'relevant_among_distractors',
             'What is the release label for project Birch?', 'train_b1_release', 'blue',
             [
                 record('train_b1', 'release', 'The release label for project Birch is blue.'),
                 record('train_b1', 'review', 'The review label for project Birch is gold.'),
                 record('train_b1', 'owner', 'The owner of project Birch is Omar.'),
                 record('train_b1', 'other', 'The release label for project Cedar is red.'),
             ]),
        case('train_b2', 'train_b', 'train', 'same_entity_different_attribute',
             'What is the deploy region for project Cedar?', 'train_b2_region', 'north',
             [
                 record('train_b2', 'region', 'The deploy region for project Cedar is north.'),
                 record('train_b2', 'backup', 'The backup region for project Cedar is south.'),
                 record('train_b2', 'target', 'The target date for project Cedar is Monday.'),
                 record('train_b2', 'other', 'The deploy region for project Birch is east.'),
             ]),
        case('train_b3', 'train_b', 'train', 'correction',
             'What is the release label for project Aster?', 'train_b3_release_new', 'silver',
             [
                 record('train_b3', 'release_old',
                        'The release label for project Aster is amber.', observed=1),
                 record('train_b3', 'release_new',
                        'The release label for project Aster is silver.', observed=5,
                        supersedes='release_old'),
                 record('train_b3', 'review',
                        'The review label for project Aster is violet.', observed=2),
                 record('train_b3', 'other',
                        'The release label for project Birch is blue.', observed=2),
             ], forbidden=('train_b3_release_old',)),
        case('dev_a', 'dev_a', 'dev', 'relevant_among_distractors',
             'What is the launch code for project Dune?', 'dev_a_launch', 'copper',
             [
                 record('dev_a', 'launch', 'The launch code for project Dune is copper.'),
                 record('dev_a', 'review', 'The review code for project Dune is violet.'),
                 record('dev_a', 'owner', 'The owner of project Dune is Priya.'),
                 record('dev_a', 'other', 'The launch code for project Fjord is emerald.'),
             ]),
        case('dev_b', 'dev_b', 'dev', 'same_entity_different_attribute',
             'What is the release label for project Fjord?', 'dev_b_release', 'teal',
             [
                 record('dev_b', 'release', 'The release label for project Fjord is teal.'),
                 record('dev_b', 'review', 'The review label for project Fjord is magenta.'),
                 record('dev_b', 'other', 'The release label for project Garnet is ochre.'),
             ]),
        case('heldout_a', 'heldout_a', 'heldout', 'relevant_among_distractors',
             'What is the launch code for project Garnet?', 'heldout_a_launch', 'crimson',
             [
                 record('heldout_a', 'launch', 'The launch code for project Garnet is crimson.'),
                 record('heldout_a', 'review', 'The review code for project Garnet is ochre.'),
                 record('heldout_a', 'backup', 'The backup code for project Garnet is cobalt.'),
                 record('heldout_a', 'other', 'The launch code for project Harbor is bronze.'),
             ]),
        case('heldout_b', 'heldout_b', 'heldout', 'correction',
             'What is the release label for project Halcyon?', 'heldout_b_release_new', 'indigo',
             [
                 record('heldout_b', 'release_old',
                        'The release label for project Halcyon is orange.', observed=1),
                 record('heldout_b', 'release_new',
                        'The release label for project Halcyon is indigo.', observed=5,
                        supersedes='release_old'),
                 record('heldout_b', 'review',
                        'The review label for project Halcyon is violet.', observed=2),
                 record('heldout_b', 'other',
                        'The release label for project Iris is rose.', observed=2),
             ], forbidden=('heldout_b_release_old',)),
        case('heldout_c', 'heldout_c', 'heldout', 'missing_information',
             'What is the launch code for project Iris?', None, 'UNKNOWN',
             [
                 record('heldout_c', 'owner', 'The owner of project Iris is Nia.'),
                 record('heldout_c', 'review', 'The review code for project Iris is grey.'),
                 record('heldout_c', 'backup', 'The backup window for project Iris is noon.'),
             ]),
        case('heldout_d', 'heldout_d', 'heldout', 'long_record_chunking',
             'What is the launch code for project Juniper?', 'heldout_d_long', 'quartz',
             [
                 record('heldout_d', 'long', long_text),
                 record('heldout_d', 'review', 'The review code for project Juniper is topaz.'),
                 record('heldout_d', 'other', 'The launch code for project Kelp is pearl.'),
             ], answer_token_start=long_start),
        case('heldout_e', 'heldout_e', 'heldout', 'future_unauthorized_wrong_purpose',
             'What is the launch code for project Kestrel?', None, 'UNKNOWN',
             [
                 record('heldout_e', 'review', 'The review code for project Kestrel is olive.'),
                 record('heldout_e', 'future',
                        'The launch code for project Kestrel is saffron.', observed=30),
                 record('heldout_e', 'unauthorized',
                        'The launch code for project Kestrel is ruby.', observed=1,
                        branch='unauthorized_branch'),
                 record('heldout_e', 'wrongpurpose',
                        'The launch code for project Kestrel is onyx.', observed=1,
                        purpose='other_purpose'),
             ], forbidden=('heldout_e_future', 'heldout_e_unauthorized',
                            'heldout_e_wrongpurpose')),
        case('heldout_f', 'heldout_f', 'heldout', 'unseen_entity_value',
             'What is the launch code for project Lumen?', 'heldout_f_launch', 'ivory',
             [
                 record('heldout_f', 'launch', 'The launch code for project Lumen is ivory.'),
                 record('heldout_f', 'review', 'The review code for project Lumen is slate.'),
                 record('heldout_f', 'owner', 'The owner of project Lumen is Tao.'),
                 record('heldout_f', 'other', 'The launch code for project Maple is coral.'),
             ]),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    records = [item for case_row in cases for item in case_row.pop('records')]
    queries = []
    for case_row in cases:
        query = {key: value for key, value in case_row.items()}
        query['records'] = [item['record_id'] for item in records
                            if item['branch_id'] in (case_row['case_id'], 'unauthorized_branch')
                            and item['record_id'].startswith(case_row['case_id'] + '_')]
        queries.append(query)
    # Validate record/source token bounds and eligibility metadata.
    token_counts = {}
    for item in records:
        ids = tokenizer(item['text'], add_special_tokens=False)['input_ids']
        token_counts[item['record_id']] = len(ids)
        if not ids or len(ids) > 256:
            raise ValueError(f'Record token count out of bounds: {item["record_id"]}')
    for query in queries:
        if any(record_id not in token_counts for record_id in query['records']):
            raise ValueError(f'Unknown record for {query["case_id"]}')
    with (OUT / 'records.jsonl').open('w', encoding='utf-8') as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
    with (OUT / 'queries.jsonl').open('w', encoding='utf-8') as stream:
        for query in queries:
            stream.write(json.dumps(query, ensure_ascii=False, sort_keys=True) + '\n')
    groups = {}
    for query in queries:
        groups.setdefault(query['source_group'], {'split': query['split'], 'cases': []})
        groups[query['source_group']]['cases'].append(query['case_id'])
    manifest = {
        'format': 'replica.e2_retrieval.dataset.v1',
        'created_at_utc': '2026-10-23',
        'subject_id': SUBJECT, 'purpose': PURPOSE, 'as_of': AS_OF,
        'tokenizer': str(ROOT / 'checkpoints/minicpm5_source'),
        'splits': {
            'train': sorted({q['source_group'] for q in queries if q['split'] == 'train'}),
            'dev': sorted({q['source_group'] for q in queries if q['split'] == 'dev'}),
            'heldout': sorted({q['source_group'] for q in queries if q['split'] == 'heldout'}),
        },
        'groups': groups,
        'counts': {'cases': len(queries), 'records': len(records),
                   'by_split': {split: sum(q['split'] == split for q in queries)
                                for split in ('train', 'dev', 'heldout')}},
        'token_counts': token_counts,
        'long_record': {'record_id': 'heldout_d_long', 'tokens': long_tokens,
                        'fact_token_start': long_start,
                        'chunks': (long_tokens + 127) // 128},
        'files': {'records_sha256': sha(OUT / 'records.jsonl'),
                  'queries_sha256': sha(OUT / 'queries.jsonl')},
        'claims': 'Authored synthetic retrieval fixtures; held-out groups are source-separated but exposed on disk.',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cases': len(queries), 'records': len(records),
                      'long_tokens': long_tokens, 'long_start': long_start,
                      'splits': manifest['counts']['by_split']}))


if __name__ == '__main__':
    main()
