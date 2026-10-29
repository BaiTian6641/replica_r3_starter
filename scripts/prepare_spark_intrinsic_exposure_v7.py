#!/usr/bin/env python3
"""Bind a fixed 32/64/96/128 fitting curve before its first native result."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.intrinsic_memory_execution_v5 import binding, local_closure
from replica_lab.intrinsic_memory_exposure_v7 import FORMAT, PACKET, PREFIXES, TRAINING, plan, totals, validate_protocol
from prepare_spark_intrinsic_explicit_evidence_v6 import save_new

REPORT = 'reports/intrinsic_memory_exposure_v7'
CONFIG = 'configs/spark_intrinsic_exposure_curve_v7.json'


def prepare(root=ROOT):
    if (root / CONFIG).exists():
        raise ValueError('v7 already prepared; no overwrite')
    previous = 'configs/spark_intrinsic_explicit_evidence_v6.json'
    protocol = deepcopy(json.loads((root / previous).read_text()))
    entries = ['scripts/run_spark_intrinsic_exposure_v7.py',
               'scripts/prepare_spark_intrinsic_exposure_v7.py',
               'src/replica_lab/intrinsic_memory_native_v7.py',
               'src/replica_lab/intrinsic_memory_diagnostic_v5.py',
               'scripts/launch_spark_intrinsic_readout_environment_v1.py']
    save_new(root, REPORT + '/implementation_v7.json', {
        'format': 'replica.intrinsic_memory.local_execution_closure.v7',
        'entries': entries, 'files': local_closure(root, entries),
        'policy': 'all_static_local_dependencies_plus_explicit_dynamic_children'})
    protocol['implementation'] = binding(root, REPORT + '/implementation_v7.json')
    protocol['format'] = FORMAT
    protocol['training'] = TRAINING
    protocol['fit_probe_prefixes'] = list(PREFIXES)
    protocol['fit_curve_selection'] = 'all_128_updates_fixed_final_prefix_only_opens_development'
    protocol['experiment_variant'] = {
        'id': 'direct_CE_exposure_curve_v7', 'predecessor': binding(root, previous),
        'reason': 'v6 passed competence and numerical continuation but 32 updates produced immediate EOS for all contextual fitting cases',
        'changed': ['fixed_128_update_exposure', 'four_declared_fitting_probe_prefixes', 'stage_budget_and_execution_identity'],
        'unchanged': ['official_BF16_source', 'memory_mechanism', 'data', 'objective', 'lr', 'seed',
                      'template', 'scorer', 'competence_and_utility_gates', 'resource_ceilings'],
        'no_development_results_observed_before_selection': True,
        'automatic_early_stop_or_best_prefix_selection': False,
        'historical_results_rewritten': False,
        'teacher_updates': 0,
    }
    protocol['budget_derivation'].update(
        fit_probe_forwards='4 fixed prefixes * 448 complete fitting-control forwards = 1792',
        fitting_updates='128 planned main updates plus one numerical continuation; zero P/teacher',
        total_forwards='64 baseline + 2 numerical + 128 fit + 1792 fit probes + 1 checkpoint + 384 development + 24 correction + 2 restart = 2397',
        stage_wall='4*240 fitting + 4*600 probes + 3720 other stages = 7080 seconds; outer 7200 seconds',
    )
    for arm, spec in protocol['arms'].items():
        spec['budget'] = plan(arm)
        spec['total_reservations'] = totals(spec['budget'])
        spec['output_root'] = f'{REPORT}/native_bf16_{arm}_v7_attempt01'
    validate_protocol(protocol)
    save_new(root, CONFIG, protocol)
    for arm, spec in protocol['arms'].items():
        packet = f'{REPORT}/native_packet_{arm}_v7.json'
        auth = f'{REPORT}/authorization_{arm}_v7.json'
        save_new(root, packet, {
            'format': PACKET, 'protocol': binding(root, CONFIG), 'arm': arm,
            'attempts': 1, 'retries': 0, 'output_root': spec['output_root'],
            'authorization_required': auth, 'authorization_status': 'external_receipt_required',
            'command': ['.venv-spark-bf16/bin/python', 'scripts/run_spark_intrinsic_exposure_v7.py',
                        '--packet', packet, '--authorization', auth, '--launch']})
    print(json.dumps({'prepared': True, 'native_run': False, 'per_arm': protocol['arms']['two']['total_reservations']}, indent=2))


if __name__ == '__main__':
    prepare()
