#!/usr/bin/env python3
"""Prepare a data-only exposure control with fresh entity worlds.

The frozen v7 runner and its complete budget/gates are reused. Fitting covers the
answer vocabulary and unsupported queries; development has new entities, without
withholding the answer vocabulary. This is one synthetic family, not broad
generalization, and it does not replace or rescore any earlier experiment.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.intrinsic_memory_execution_v5 import binding, local_closure
from replica_lab.intrinsic_memory_exposure_v7 import PACKET, validate_protocol
from prepare_spark_intrinsic_explicit_evidence_v6 import oracle, render, save_new

REPORT = 'reports/intrinsic_memory_supported_v8'
DATA = 'data/intrinsic_memory_supported_v8_20261029'
CONFIG = 'configs/spark_intrinsic_supported_values_v8.json'
# (fact entity, color, query entity). Paired fitting questions have different
# verified answers; the last two pairs include unsupported-query teaching.
CASES = {
    'fit': (
        ('Kela', 'red', 'Kela'), ('Kela', 'blue', 'Kela'),
        ('Domi', 'green', 'Domi'), ('Domi', 'gold', 'Domi'),
        ('Rivo', 'red', 'Rivo'), ('Laki', 'blue', 'Rivo'),
        ('Sanu', 'green', 'Sanu'), ('Toro', 'gold', 'Sanu'),
    ),
    'development': (
        ('Hena', 'red', 'Hena'), ('Hena', 'blue', 'Hena'),
        ('Sori', 'green', 'Sori'), ('Sori', 'gold', 'Sori'),
        ('Mavi', 'red', 'Mavi'), ('Mavi', 'gold', 'Mavi'),
        ('Telo', 'green', 'Telo'), ('Nalo', 'blue', 'Telo'),
    ),
}


def worlds(split):
    rows = []
    for i, (entity, color, query) in enumerate(CASES[split]):
        facts = [{'entity': entity, 'color': color}]
        rows.append({'world_id': f'v8_{split}_{i:02d}', 'family': query,
                     'rule_family': 'explicit_entity_color_lookup',
                     'facts': facts, 'query_entity': query, 'text': render(facts[0]),
                     'question': f'What is the recorded color of {query}?',
                     'answer': oracle(facts, query), 'chronology': 1,
                     'source_kind': 'scripted_complete_world',
                     'verification': 'structured_fact_oracle_not_generated_explanation'})
    return {'format': 'replica.intrinsic_memory.supported_worlds.v8', 'split': split,
            'seed': 2026102908, 'provenance': 'deterministic_supported_value_worlds_v8',
            'historical_development_data_used': False, 'sealed_final': False,
            'answer_vocabulary_withheld': False, 'independent_rule_family_claim': False,
            'rows': rows}


def validate_corpus(doc):
    split = doc.get('split')
    if split not in CASES or doc != worlds(split):
        raise ValueError('declared source, oracle, split or exposure schedule differs')
    if any(len(r['text'].encode()) > 256 for r in doc['rows']):
        raise ValueError('evidence span exceeds declared budget')
    if split == 'fit':
        for i in range(0, 8, 2):
            a, b = doc['rows'][i:i+2]
            if a['question'] != b['question'] or a['answer'] == b['answer']:
                raise ValueError('same-question counterfactual pair missing')


def prepare(root=ROOT):
    if (root / CONFIG).exists() or (root / DATA).exists():
        raise ValueError('v8 already prepared; no overwrite')
    previous = 'configs/spark_intrinsic_exposure_curve_v7.json'
    protocol = deepcopy(json.loads((root / previous).read_text()))
    for split in CASES:
        doc = worlds(split)
        validate_corpus(doc)
        save_new(root, f'{DATA}/{split}.json', doc)
        protocol['data'][split] = binding(root, f'{DATA}/{split}.json')
    # Calibration is the unchanged complete-context source/input qualification.
    # No prior development file is read by this preparation.
    old = json.loads((root / protocol['implementation']['path']).read_text())
    entries = [*old['entries'], 'scripts/prepare_spark_intrinsic_supported_values_v8.py']
    save_new(root, REPORT + '/implementation_v8.json', {
        'format': 'replica.intrinsic_memory.local_execution_closure.v7',
        'entries': entries, 'files': local_closure(root, entries),
        'policy': 'unchanged_v7_execution_plus_explicit_v8_corpus_preparation'})
    protocol['implementation'] = binding(root, REPORT + '/implementation_v8.json')
    protocol['experiment_variant'] = {
        'id': 'supported_answer_vocabulary_and_abstention_exposure_v8',
        'predecessor': binding(root, previous),
        'reason': 'v7 learns the fitting memories; separate novel-entity recall from untrained answer vocabulary and absent unsupported-query teaching',
        'changed': ['new_fitting_worlds_cover_four_colors_and_unknown',
                    'fresh_development_entities_with_supported_answer_vocabulary',
                    'execution_and_output_identity'],
        'unchanged': ['official_BF16_source', 'v7_runner', 'memory_mechanism',
                      '128_plus_1_updates', 'fixed_four_probe_schedule', 'initialization',
                      'optimizer', 'lr', 'calibration', 'template', 'scorer',
                      'competence_fit_and_utility_criteria', 'resource_and_work_limits'],
        'historical_results_rewritten': False,
        'prior_development_reused': False,
        'independent_rule_family_claim': False,
        'isolated_causal_attribution_to_one_data_factor': False,
        'automatic_early_stop_or_best_prefix_selection': False,
        'teacher_updates': 0,
    }
    for arm, spec in protocol['arms'].items():
        spec['output_root'] = f'{REPORT}/native_bf16_{arm}_v8_attempt01'
    validate_protocol(protocol)
    save_new(root, CONFIG, protocol)
    for arm, spec in protocol['arms'].items():
        packet = f'{REPORT}/native_packet_{arm}_v8.json'
        authorization = f'{REPORT}/authorization_{arm}_v8.json'
        save_new(root, packet, {
            'format': PACKET, 'experiment_version': 8,
            'protocol': binding(root, CONFIG), 'arm': arm,
            'attempts': 1, 'retries': 0, 'output_root': spec['output_root'],
            'authorization_required': authorization,
            'authorization_status': 'external_receipt_required',
            'command': ['.venv-spark-bf16/bin/python', 'scripts/run_spark_intrinsic_exposure_v7.py',
                        '--packet', packet, '--authorization', authorization, '--launch']})
    print(json.dumps({'prepared': True, 'native_run': False,
                      'execution_version': 7, 'corpus_protocol_version': 8}, indent=2))


if __name__ == '__main__':
    prepare()
