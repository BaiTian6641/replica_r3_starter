#!/usr/bin/env python3
"""Bind one frozen-teacher comparison before observing its fitting/development results."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.intrinsic_memory_execution_v5 import binding, local_closure
from replica_lab.intrinsic_memory_alignment_v10 import FORMAT, PACKET, TRAINING, TEACHER, plan, totals, validate_protocol
from prepare_spark_intrinsic_explicit_evidence_v6 import save_new

REPORT = 'reports/intrinsic_memory_alignment_v10'
CONFIG = 'configs/spark_intrinsic_alignment_v10.json'


def prepare(root=ROOT):
    if (root / CONFIG).exists():
        raise ValueError('v10 already prepared; no overwrite')
    previous = 'configs/spark_intrinsic_smaller_step_v9.json'
    protocol = deepcopy(json.loads((root / previous).read_text()))
    predecessor_result = protocol['arms']['two']['output_root'] + '/result.json'
    observed = json.loads((root / predecessor_result).read_text())
    if (observed.get('native_competence') is not True
            or observed.get('main', {}).get('fit_memory_dependence_passed') is not False
            or observed.get('main', {}).get('fit_updates') != 128
            or (root / protocol['arms']['two']['output_root'] / 'development.json').exists()):
        raise ValueError('teacher comparison requires completed failed fitting and unopened development')
    entries = ['scripts/run_spark_intrinsic_alignment_v10.py',
               'scripts/prepare_spark_intrinsic_alignment_v10.py',
               'src/replica_lab/intrinsic_memory_native_v10.py',
               'scripts/launch_spark_intrinsic_readout_environment_v1.py',
               'scripts/prepare_spark_intrinsic_explicit_evidence_v6.py']
    save_new(root, REPORT + '/implementation_v10.json', {
        'format': 'replica.intrinsic_memory.local_execution_closure.v10',
        'entries': entries, 'files': local_closure(root, entries),
        'policy': 'separate_teacher_objective_with_fixed_v9_student_schedule_and_extra_teacher_budget'})
    protocol['implementation'] = binding(root, REPORT + '/implementation_v10.json')
    protocol['format'] = FORMAT
    protocol['training'] = deepcopy(TRAINING)
    protocol['teacher'] = deepcopy(TEACHER)
    protocol['diagnostic_predecessor'] = binding(root, predecessor_result)
    protocol['experiment_variant'] = {
        'id': 'frozen_full_context_teacher_alignment_v10',
        'predecessor': binding(root, previous),
        'reason': 'the v9 direct-CE final prefix fails fitting; compare response-aligned soft targets with the same source, data, initialization and student exposure',
        'changed': ['complete_response_CE_to_top64_forward_KL', 'teacher_qualification_and_detached_cache',
                    'additional_teacher_work_and_reallocated_wall_artifact_envelope', 'execution_and_output_identity'],
        'unchanged': ['official_BF16_source', 'contextual_producer_and_reader_mechanism',
                      'v8_fit_and_unopened_development_data', '128_plus_1_student_updates',
                      'four_fixed_probes_no_best_prefix_selection', 'initialization', 'learning_rate_0.0001',
                      'NativeFusedAdamW', 'template', 'scorer', 'competence_fit_utility_gates',
                      '7200_second_outer_wall', '50_GiB_512_MiB_15_percent_guards'],
        'reference': {'url': 'https://arxiv.org/html/2610.26825v2', 'sections': ['3.3', 'E.2.3'],
                      'scope': 'response_alignment_and_KL_equations_only_no_architecture_or_capability_transfer_claim'},
        'historical_results_rewritten': False, 'development_labels_used_to_select_objective': False,
        'teacher_updates': 0, 'persistent_P_updates': 0,
    }
    for arm, spec in protocol['arms'].items():
        spec['output_root'] = f'{REPORT}/native_bf16_{arm}_v10_attempt01'
        spec['budget'] = plan(arm)
        spec['total_reservations'] = totals(spec['budget'])
    validate_protocol(protocol)
    save_new(root, CONFIG, protocol)
    for arm, spec in protocol['arms'].items():
        packet = f'{REPORT}/native_packet_{arm}_v10.json'
        authorization = f'{REPORT}/authorization_{arm}_v10.json'
        save_new(root, packet, {
            'format': PACKET, 'experiment_version': 10, 'native_engine_version': 10,
            'protocol': binding(root, CONFIG), 'arm': arm, 'attempts': 1, 'retries': 0,
            'output_root': spec['output_root'], 'authorization_required': authorization,
            'authorization_status': 'external_receipt_required',
            'command': ['.venv-spark-bf16/bin/python', 'scripts/run_spark_intrinsic_alignment_v10.py',
                        '--packet', packet, '--authorization', authorization, '--launch']})
    print(json.dumps({'prepared': True, 'native_run': False,
                      'native_engine_version': 10, 'admission_protocol_version': 10}, indent=2))


if __name__ == '__main__':
    prepare()
