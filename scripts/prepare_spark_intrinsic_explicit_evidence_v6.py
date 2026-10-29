#!/usr/bin/env python3
"""Prepare one data-only successor to the failed v5 competence attempt.

The v5 execution implementation, scorer, gates, numerical policy and budgets are
unchanged. Explicit entity/color facts remove material and referent ambiguity.
Preparation neither parses any prior development corpus nor launches a model.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.intrinsic_memory_execution_v5 import binding, local_closure, validate_protocol

REPORT = 'reports/intrinsic_memory_explicit_v6'
DATA = 'data/intrinsic_memory_explicit_v6_20261029'
CONFIG = 'configs/spark_intrinsic_explicit_evidence_v6.json'
ENTITIES = {
    'calibration': ('Runa', 'Tavi', 'Meko', 'Zori', 'Neli'),
    'fit': ('Pavo', 'Sela', 'Kito', 'Daro', 'Havi'),
    'development': ('Fena', 'Lumo', 'Vera', 'Noro', 'Juna'),
}
PALETTES = {'calibration': ('copper', 'silver'), 'fit': ('red', 'blue'),
            'development': ('green', 'gold')}


def oracle(facts, query_entity):
    matches = [f['color'] for f in facts if f['entity'] == query_entity]
    if len(matches) > 1:
        raise ValueError('ambiguous entity authority')
    return matches[0] if matches else 'unknown'


def render(fact):
    return (f"The recorded color of {fact['entity']} is {fact['color']}. "
            'No color is recorded for any other entity.')


def worlds(split):
    names, colors = ENTITIES[split], PALETTES[split]
    rows = []
    for i in range(8):
        entity = names[i // 2]
        fact = {'entity': entity, 'color': colors[i % 2]}
        query = names[4] if split != 'fit' and i == 7 else entity
        rows.append({'world_id': f'v6_{split}_{i:02d}', 'family': entity,
                     'rule_family': 'explicit_entity_color_lookup',
                     'facts': [fact], 'query_entity': query, 'text': render(fact),
                     'question': f'What is the recorded color of {query}?',
                     'answer': oracle([fact], query), 'chronology': 1,
                     'source_kind': 'scripted_complete_world',
                     'verification': 'structured_fact_oracle_not_generated_explanation'})
    return {'format': 'replica.intrinsic_memory.explicit_worlds.v6', 'split': split,
            'seed': 2026102906, 'provenance': 'deterministic_explicit_entity_color_worlds_v6',
            'historical_development_data_used': False, 'sealed_final': False,
            'independent_rule_family_claim': False, 'rows': rows}


def validate_corpus(doc):
    split = doc['split']
    if split not in ENTITIES or len(doc['rows']) != 8:
        raise ValueError('incomplete declared split')
    seen = set()
    for i, row in enumerate(doc['rows']):
        facts = row['facts']
        if len(facts) != 1 or row['world_id'] in seen:
            raise ValueError('one unique authoritative world required')
        seen.add(row['world_id'])
        if (row['text'] != render(facts[0]) or len(row['text'].encode()) > 256
                or row['question'] != f"What is the recorded color of {row['query_entity']}?"
                or row['answer'] != oracle(facts, row['query_entity'])):
            raise ValueError('source/question/oracle disagreement')
        absent = split != 'fit' and i == 7
        if (row['answer'] == 'unknown') != absent:
            raise ValueError('missing-entity case differs')
        if facts[0]['entity'] != ENTITIES[split][i // 2]:
            raise ValueError('entity assignment differs')
        if facts[0]['color'] != PALETTES[split][i % 2]:
            raise ValueError('color assignment differs')
        if row['query_entity'] != ENTITIES[split][4 if absent else i // 2]:
            raise ValueError('query entity differs')
    if split == 'fit':
        for i in range(0, 8, 2):
            left, right = doc['rows'][i:i + 2]
            if left['question'] != right['question'] or left['answer'] == right['answer']:
                raise ValueError('counterfactual fitting pair differs')


def save_new(root, relative, value):
    p = root / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def prepare(root=ROOT):
    # No old data is read. Only the already reviewed protocol and closure are
    # reused; all old files stay immutable. A second preparation is refused.
    if (root / CONFIG).exists() or (root / DATA).exists():
        raise ValueError('v6 preparation already exists; no overwrite')
    protocol = deepcopy(json.loads((root / 'configs/spark_intrinsic_diagnostic_v5.json').read_text()))
    original = deepcopy(protocol)
    for split in ENTITIES:
        doc = worlds(split)
        validate_corpus(doc)
        save_new(root, f'{DATA}/{split}.json', doc)
    previous_closure = json.loads((root / protocol['implementation']['path']).read_text())
    entries = [*previous_closure['entries'], 'scripts/prepare_spark_intrinsic_explicit_evidence_v6.py']
    save_new(root, REPORT + '/implementation_v6.json', {
        'format': 'replica.intrinsic_memory.local_execution_closure.v5', 'entries': entries,
        'files': local_closure(root, entries),
        'policy': 'unchanged_v5_execution_plus_explicit_v6_corpus_preparation'})
    protocol['implementation'] = binding(root, REPORT + '/implementation_v6.json')
    protocol['data'] = {s: binding(root, f'{DATA}/{s}.json') for s in ENTITIES}
    protocol['experiment_variant'] = {
        'id': 'explicit_entity_color_evidence_v6',
        'predecessor': binding(root, 'configs/spark_intrinsic_diagnostic_v5.json'),
        'reason': 'v5 competence 6/8; material and missing-referent wording is ambiguous; causal explanation remains unproven',
        'changed': ['fresh_explicit_entity_attribute_corpus', 'bound_output_and_execution_identity'],
        'unchanged': ['official_BF16_source', 'v5_runner', 'prompt_template', 'scorer', '8_of_8_competence',
                      '32_plus_1_schedule', 'fit_memory_dependence_gate', 'utility_gate', 'resource_and_work_limits'],
        'historical_results_rewritten': False,
        'not_a_repeat_of_consumed_v5_packet': True,
        'all_v6_choices_frozen_before_any_v6_model_result': True,
    }
    for arm in protocol['arms']:
        protocol['arms'][arm]['output_root'] = f'{REPORT}/native_bf16_{arm}_v6_attempt01'
        assert protocol['arms'][arm]['budget'] == original['arms'][arm]['budget']
    validate_protocol(protocol)
    save_new(root, CONFIG, protocol)
    for arm in protocol['arms']:
        packet_path = f'{REPORT}/native_packet_{arm}_v6.json'
        authorization = f'{REPORT}/authorization_{arm}_v6.json'
        save_new(root, packet_path, {
            'format': 'replica.intrinsic_memory.launch_packet.v5', 'experiment_version': 6,
            'protocol': binding(root, CONFIG), 'arm': arm, 'attempts': 1, 'retries': 0,
            'output_root': protocol['arms'][arm]['output_root'],
            'authorization_required': authorization,
            'command': ['.venv-spark-bf16/bin/python', 'scripts/run_spark_intrinsic_diagnostic_v5.py',
                        '--packet', packet_path, '--authorization', authorization, '--launch'],
            'authorization_status': 'external_receipt_required',
            'historical_v5_packet_unchanged': True})
    print(json.dumps({'prepared': True, 'model_loaded': False, 'execution_version': 5,
                      'corpus_protocol_version': 6, 'config': CONFIG}, indent=2))


if __name__ == '__main__':
    prepare()
