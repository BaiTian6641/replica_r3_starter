#!/usr/bin/env python3
"""Generate source-separated short-record answer-loss data for an E2 reader candidate.

Each case has one relevant record and distractors that share entity or attribute.
Train/dev/held-out groups are disjoint, and held-out values/entities never enter
the answer-loss training set. All records are synthetic development fixtures.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/e2_reader'
SUBJECT = 'e2_reader'
PURPOSE = 'reader_candidate_eval'
AS_OF = 10

TRAIN_ENTITIES = ['Aster', 'Birch', 'Cedar', 'Dune', 'Ember', 'Fjord', 'Garnet',
                  'Harbor', 'Iris', 'Juniper', 'Kelp', 'Lumen', 'Maple', 'Nimbus',
                  'Onyx', 'Pearl', 'Quartz', 'Raven', 'Sable', 'Topaz', 'Umber',
                  'Vesper', 'Willow', 'Zephyr']
DEV_ENTITIES = ['Alpine', 'Basalt', 'Cinder', 'Delta', 'Fable', 'Glacier']
HELDOUT_ENTITIES = ['Bramble', 'Cobalt', 'Drift', 'Eclipse', 'Flint', 'Grotto',
                    'Hollow', 'Ivory', 'Jasper', 'Kestrel']
ATTRS = ['launch code', 'release label', 'backup window', 'owner', 'deploy region',
         'review code', 'timeout seconds', 'retry limit']
VALUES = ['amber', 'blue', 'silver', 'crimson', 'teal', 'violet', 'ochre',
          'indigo', 'copper', 'emerald', 'bronze', 'saffron', 'slate', 'ivory',
          'quartz', 'topaz', 'onyx', 'coral', 'rose', 'pearl', 'olive', 'ruby',
          'jade', 'opal', 'azure', 'mauve', 'umber', 'lilac', 'flint', 'hazel',
          'cedar', 'maroon', 'mint', 'navy', 'peach', 'plum', 'rust', 'sage',
          'tan', 'wine']
HELDOUT_VALUES = ['vermilion', 'cerulean', 'chartreuse', 'magenta', 'tangerine',
                  'periwinkle', 'turquoise', 'burgundy', 'auburn', 'lavender']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_case(case_id, group, split, entity, attribute, value, index):
    relevant = f'{case_id}_relevant'
    records = [
        {'record_id': relevant, 'subject_id': SUBJECT, 'branch_id': case_id,
         'text': f'The {attribute} for project {entity} is {value}.',
         'observed_at': 1, 'event_at': 0, 'purpose': PURPOSE, 'kind': 'source',
         'supersedes': None, 'source_id': group, 'authenticity': 'source_report'},
    ]
    # Same entity, different attribute.
    other_attr = ATTRS[(ATTRS.index(attribute) + 1) % len(ATTRS)]
    records.append({'record_id': f'{case_id}_same_entity', 'subject_id': SUBJECT,
                    'branch_id': case_id,
                    'text': f'The {other_attr} for project {entity} is {VALUES[(index+3) % len(VALUES)]}.',
                    'observed_at': 1, 'event_at': 0, 'purpose': PURPOSE, 'kind': 'source',
                    'supersedes': None, 'source_id': group, 'authenticity': 'source_report'})
    # Different entity, same attribute.
    other_entity = TRAIN_ENTITIES[(TRAIN_ENTITIES.index(entity) + 5) % len(TRAIN_ENTITIES)] \
        if entity in TRAIN_ENTITIES else 'Other'
    records.append({'record_id': f'{case_id}_same_attr', 'subject_id': SUBJECT,
                    'branch_id': case_id,
                    'text': f'The {attribute} for project {other_entity} is {VALUES[(index+7) % len(VALUES)]}.',
                    'observed_at': 1, 'event_at': 0, 'purpose': PURPOSE, 'kind': 'source',
                    'supersedes': None, 'source_id': group, 'authenticity': 'source_report'})
    records.append({'record_id': f'{case_id}_owner', 'subject_id': SUBJECT,
                    'branch_id': case_id,
                    'text': f'The owner of project {entity} is person{index:02d}.',
                    'observed_at': 1, 'event_at': 0, 'purpose': PURPOSE, 'kind': 'source',
                    'supersedes': None, 'source_id': group, 'authenticity': 'source_report'})
    query = {'case_id': case_id, 'source_group': group, 'split': split,
             'subject_id': SUBJECT, 'branch_id': case_id, 'purpose': PURPOSE,
             'as_of': AS_OF, 'query': f'What is the {attribute} for project {entity}?',
             'expected_record_id': relevant, 'expected_answer': value}
    return records, query


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / 'checkpoints/minicpm5_source', local_files_only=True,
        trust_remote_code=False)
    cases = []
    for index, entity in enumerate(TRAIN_ENTITIES):
        group = f'train_{index % 6}'
        cases.append(('reader_train_%02d' % (index + 1), group, 'train', entity,
                      ATTRS[index % len(ATTRS)], VALUES[index % 24], index))
    for index, entity in enumerate(DEV_ENTITIES):
        group = f'dev_{index % 2}'
        cases.append(('reader_dev_%02d' % (index + 1), group, 'dev', entity,
                      ATTRS[(index + 2) % len(ATTRS)], VALUES[(index + 24) % 40],
                      index + 30))
    for index, entity in enumerate(HELDOUT_ENTITIES):
        group = f'heldout_{index % 5}'
        cases.append(('reader_heldout_%02d' % (index + 1), group, 'heldout', entity,
                      ATTRS[(index + 4) % len(ATTRS)], HELDOUT_VALUES[index],
                      index + 40))
    records, queries = [], []
    for case_id, group, split, entity, attribute, value, index in cases:
        case_records, query = make_case(case_id, group, split, entity, attribute,
                                        value, index)
        records.extend(case_records)
        queries.append(query)
    counts = {}
    for record in records:
        ids = tokenizer(record['text'], add_special_tokens=False)['input_ids']
        if not ids or len(ids) > 128:
            raise ValueError(f'record token length out of bounds: {record["record_id"]}')
        counts[record['record_id']] = len(ids)
    if any(query['expected_answer'].casefold() in query['query'].casefold()
           for query in queries):
        raise ValueError('query leaks expected answer')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'records.jsonl').open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    with (OUT / 'queries.jsonl').open('w', encoding='utf-8') as stream:
        for query in queries:
            stream.write(json.dumps(query, ensure_ascii=False, sort_keys=True) + '\n')
    groups = {}
    for query in queries:
        groups.setdefault(query['source_group'], {'split': query['split'], 'cases': []})
        groups[query['source_group']]['cases'].append(query['case_id'])
    manifest = {
        'format': 'replica.e2_reader.dataset.v1', 'created_at_utc': '2026-10-23',
        'subject_id': SUBJECT, 'purpose': PURPOSE, 'as_of': AS_OF,
        'counts': {'cases': len(queries), 'records': len(records),
                   'by_split': {split: sum(q['split'] == split for q in queries)
                                for split in ('train', 'dev', 'heldout')}},
        'groups': groups, 'token_counts': counts,
        'files': {'records_sha256': sha(OUT / 'records.jsonl'),
                  'queries_sha256': sha(OUT / 'queries.jsonl')},
        'claims': 'Synthetic short-record reader candidate data; held-out entities/values are source-separated but exposed on disk.'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cases': len(queries), 'records': len(records),
                      'splits': manifest['counts']['by_split']}))


if __name__ == '__main__':
    main()
