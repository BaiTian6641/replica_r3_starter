#!/usr/bin/env python3
"""Generate source-separated pages for the E4 delayed-use native arm.

One case is one observed source page containing the relevant record plus
entity/attribute/owner distractors. The page is supplied to fast memory before
the later query; it is never inserted into the query prompt. Held-out entities
and values are absent from training. Synthetic development fixtures only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/e4_delayed'
SUBJECT = 'e4_delayed'
PURPOSE = 'delayed_use_eval'
AS_OF = 12

ENTITIES = [
    'Alder', 'Basin', 'Cobalt', 'Dell', 'Ember', 'Fern', 'Grove', 'Hearth',
    'Inlet', 'Juniper', 'Kite', 'Lark', 'Meadow', 'Nectar', 'Orchid', 'Pike',
    'Quill', 'Reed', 'Sage', 'Thistle', 'Umber', 'Vale', 'Wren', 'Yarrow',
    'Zephyr', 'Aurora', 'Bramble', 'Cinder', 'Dune', 'Elm', 'Flint', 'Glade',
    'Harbor', 'Iris', 'Jasper', 'Kelp', 'Lumen', 'Maple',
]
DEV_ENTITIES = ENTITIES[24:30]
HELDOUT_ENTITIES = ENTITIES[30:38]
ATTRS = ['deploy region', 'launch code', 'backup window', 'owner',
         'review code', 'timeout seconds', 'retry limit', 'release label']
VALUES = [
    'amber', 'blue', 'silver', 'crimson', 'teal', 'violet', 'ochre', 'indigo',
    'copper', 'emerald', 'bronze', 'saffron', 'slate', 'ivory', 'quartz',
    'topaz', 'onyx', 'coral', 'rose', 'pearl', 'olive', 'ruby', 'jade', 'opal',
    'azure', 'mauve', 'umber', 'lilac', 'flint', 'hazel', 'saffron2', 'maroon',
    'mint', 'navy', 'peach', 'plum', 'rust', 'sage',
]
HELDOUT_VALUES = ['vermilion', 'cerulean', 'chartreuse', 'magenta',
                  'tangerine', 'periwinkle', 'turquoise', 'burgundy']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_page(case_id, group, split, entity, attribute, value, index, other_entity):
    relevant = f'{case_id}_relevant'
    other_attr = ATTRS[(ATTRS.index(attribute) + 1) % len(ATTRS)]
    records = [
        {'record_id': relevant, 'subject_id': SUBJECT, 'branch_id': case_id,
         'text': f'The {attribute} for project {entity} is {value}.',
         'observed_at': 1, 'event_at': 0, 'purpose': PURPOSE, 'kind': 'source',
         'supersedes': None, 'source_id': group, 'authenticity': 'source_report'},
        {'record_id': f'{case_id}_same_entity', 'subject_id': SUBJECT,
         'branch_id': case_id,
         'text': f'The {other_attr} for project {entity} is {VALUES[(index + 3) % len(VALUES)]}.',
         'observed_at': 1, 'event_at': 0, 'purpose': PURPOSE, 'kind': 'source',
         'supersedes': None, 'source_id': group, 'authenticity': 'source_report'},
        {'record_id': f'{case_id}_same_attr', 'subject_id': SUBJECT,
         'branch_id': case_id,
         'text': f'The {attribute} for project {other_entity} is {VALUES[(index + 7) % len(VALUES)]}.',
         'observed_at': 1, 'event_at': 0, 'purpose': PURPOSE, 'kind': 'source',
         'supersedes': None, 'source_id': group, 'authenticity': 'source_report'},
        {'record_id': f'{case_id}_owner', 'subject_id': SUBJECT,
         'branch_id': case_id,
         'text': f'The owner of project {entity} is person{index:02d}.',
         'observed_at': 1, 'event_at': 0, 'purpose': PURPOSE, 'kind': 'source',
         'supersedes': None, 'source_id': group, 'authenticity': 'source_report'},
    ]
    page = '\n'.join(record['text'] for record in records)
    query = {'case_id': case_id, 'source_group': group, 'split': split,
             'subject_id': SUBJECT, 'branch_id': case_id, 'purpose': PURPOSE,
             'as_of': AS_OF, 'observation': page,
             'query': f'What is the {attribute} for project {entity}?',
             'expected_record_id': relevant, 'expected_answer': value}
    return records, query


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / 'checkpoints/minicpm5_source', local_files_only=True,
        trust_remote_code=False)
    cases = []
    for index in range(24):
        group = f'train_{index % 6}'
        cases.append(('e4_train_%02d' % (index + 1), group, 'train',
                      ENTITIES[index], ATTRS[index % len(ATTRS)],
                      VALUES[index], index))
    for index in range(6):
        group = f'dev_{index % 2}'
        cases.append(('e4_dev_%02d' % (index + 1), group, 'dev',
                      DEV_ENTITIES[index], ATTRS[(index + 2) % len(ATTRS)],
                      VALUES[(index + 24) % 38], index + 30))
    for index in range(8):
        group = f'heldout_{index % 4}'
        cases.append(('e4_heldout_%02d' % (index + 1), group, 'heldout',
                      HELDOUT_ENTITIES[index], ATTRS[(index + 4) % len(ATTRS)],
                      HELDOUT_VALUES[index], index + 40))
    records, queries = [], []
    for case_id, group, split, entity, attribute, value, index in cases:
        other = ENTITIES[(ENTITIES.index(entity) + 5) % len(ENTITIES)] \
            if entity in ENTITIES else 'Other'
        case_records, query = make_page(case_id, group, split, entity,
                                        attribute, value, index, other)
        records.extend(case_records)
        queries.append(query)
    counts = {}
    for query in queries:
        ids = tokenizer(query['observation'], add_special_tokens=False)['input_ids']
        if not ids or len(ids) > 256:
            raise ValueError(f'observation token length out of bounds: {query["case_id"]}')
        if query['expected_answer'].casefold() in query['query'].casefold():
            raise ValueError(f'query leaks answer: {query["case_id"]}')
        counts[query['case_id']] = len(ids)
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
        'format': 'replica.e4_delayed.dataset.v1', 'created_at_utc': '2026-10-23',
        'subject_id': SUBJECT, 'purpose': PURPOSE, 'as_of': AS_OF,
        'counts': {'cases': len(queries), 'records': len(records),
                   'by_split': {split: sum(q['split'] == split for q in queries)
                                for split in ('train', 'dev', 'heldout')}},
        'groups': groups, 'observation_tokens': counts,
        'files': {'records_sha256': sha(OUT / 'records.jsonl'),
                  'queries_sha256': sha(OUT / 'queries.jsonl')},
        'claims': 'Synthetic source pages; one observation-only fast write before a later query. Held-out entities/values are source-separated but exposed on disk.'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cases': len(queries), 'records': len(records),
                      'splits': manifest['counts']['by_split'],
                      'max_tokens': max(counts.values())}))


if __name__ == '__main__':
    main()
