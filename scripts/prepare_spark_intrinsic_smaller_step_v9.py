#!/usr/bin/env python3
"""Bind one smaller-step control before observing its fitting/development results."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.intrinsic_memory_execution_v5 import binding, local_closure
from replica_lab.intrinsic_memory_smaller_step_v9 import FORMAT, PACKET, TRAINING, validate_protocol
from prepare_spark_intrinsic_explicit_evidence_v6 import save_new

REPORT = 'reports/intrinsic_memory_step_v9'
CONFIG = 'configs/spark_intrinsic_smaller_step_v9.json'


def prepare(root=ROOT):
    if (root / CONFIG).exists():
        raise ValueError('v9 already prepared; no overwrite')
    previous = 'configs/spark_intrinsic_supported_values_v8.json'
    protocol = deepcopy(json.loads((root / previous).read_text()))
    predecessor_result = protocol['arms']['two']['output_root'] + '/result.json'
    observed = json.loads((root / predecessor_result).read_text())
    if (observed.get('native_competence') is not True
            or observed.get('main', {}).get('fit_memory_dependence_passed') is not False
            or observed.get('main', {}).get('fit_updates') != 128
            or (root / protocol['arms']['two']['output_root'] / 'development.json').exists()):
        raise ValueError('smaller-step selection requires completed failed fitting and unopened development')
    entries = ['scripts/run_spark_intrinsic_smaller_step_v9.py',
               'scripts/prepare_spark_intrinsic_smaller_step_v9.py',
               'src/replica_lab/intrinsic_memory_native_v7.py',
               'scripts/launch_spark_intrinsic_readout_environment_v1.py',
               'scripts/prepare_spark_intrinsic_supported_values_v8.py']
    save_new(root, REPORT + '/implementation_v9.json', {
        'format': 'replica.intrinsic_memory.local_execution_closure.v9',
        'entries': entries, 'files': local_closure(root, entries),
        'policy': 'versioned_admission_with_unchanged_v7_native_schedule'})
    protocol['implementation'] = binding(root, REPORT + '/implementation_v9.json')
    protocol['format'] = FORMAT
    protocol['training'] = deepcopy(TRAINING)
    protocol['diagnostic_predecessor'] = binding(root, predecessor_result)
    protocol['experiment_variant'] = {
        'id': 'smaller_ordinary_optimizer_step_v9',
        'predecessor': binding(root, previous),
        'reason': 'complete-EOS direct CE can learn the two-color fit but the broader v8 fit remains collapsed; earlier probes show large early reader residuals',
        'changed': ['learning_rate_0.001_to_0.0001', 'execution_and_output_identity'],
        'unchanged': ['official_BF16_source', 'v7_native_engine', 'memory_mechanism',
                      'data_including_unopened_v8_development', '128_plus_1_updates',
                      'fixed_four_probe_schedule', 'initialization', 'NativeFusedAdamW',
                      'template', 'scorer', 'competence_fit_and_utility_gates',
                      'resource_and_work_limits'],
        'historical_results_rewritten': False,
        'development_exposure_required_to_select_rate': False,
        'automatic_early_stop_or_best_prefix_selection': False,
        'teacher_updates': 0,
    }
    for arm, spec in protocol['arms'].items():
        spec['output_root'] = f'{REPORT}/native_bf16_{arm}_v9_attempt01'
    validate_protocol(protocol)
    save_new(root, CONFIG, protocol)
    for arm, spec in protocol['arms'].items():
        packet = f'{REPORT}/native_packet_{arm}_v9.json'
        authorization = f'{REPORT}/authorization_{arm}_v9.json'
        save_new(root, packet, {
            'format': PACKET, 'experiment_version': 9, 'native_engine_version': 7,
            'protocol': binding(root, CONFIG), 'arm': arm, 'attempts': 1, 'retries': 0,
            'output_root': spec['output_root'], 'authorization_required': authorization,
            'authorization_status': 'external_receipt_required',
            'command': ['.venv-spark-bf16/bin/python', 'scripts/run_spark_intrinsic_smaller_step_v9.py',
                        '--packet', packet, '--authorization', authorization, '--launch']})
    print(json.dumps({'prepared': True, 'native_run': False,
                      'native_engine_version': 7, 'admission_protocol_version': 9}, indent=2))


if __name__ == '__main__':
    prepare()
