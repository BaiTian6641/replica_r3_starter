"""Fixed broader-teaching comparison; existing eight-world gates apply per group.

Only standard-library dependencies are imported before source admission. The
producer, reader, objective, placement and original model remain unchanged.
"""
from copy import deepcopy

from .intrinsic_memory_exposure_v7 import stage, totals
from .intrinsic_memory_alignment_v10 import (
    FORMAT as PREVIOUS_FORMAT, TRAINING as PREVIOUS_TRAINING,
    TEACHER as PREVIOUS_TEACHER, plan as previous_plan, validate_protocol as validate_previous,
)

FORMAT = 'replica.intrinsic_memory.coverage.v11'
PACKET = 'replica.intrinsic_memory.launch_packet.v11'
AUTHORIZATION = 'replica.intrinsic_memory.external_authorization.v11'
FIT_CASES = 32
DEVELOPMENT_CASES = 16
GROUP_SIZE = 8
PREFIXES = (128, 256, 384, 512)
TRAINING = {**PREVIOUS_TRAINING, 'updates': 512,
            'schedule': 'fixed_round_robin_32_worlds_16_passes'}
TEACHER = deepcopy(PREVIOUS_TEACHER)
TEACHER['qualification'].update(cases=32, correct_min=32)
TEACHER['target_cache'] = '32_detached_targets_created_once_reused_512_plus_1_updates'
COVERAGE = {
    'calibration_cases': 8, 'fit_cases': FIT_CASES, 'development_cases': DEVELOPMENT_CASES,
    'criterion_group_size': GROUP_SIZE, 'fit_groups': 4, 'development_groups': 2,
    'fitting_criterion': 'all_four_fixed_groups_must_pass_unchanged_eight_world_fit_gate',
    'utility_criterion': 'both_fixed_groups_must_pass_unchanged_eight_world_utility_gate',
    'group_membership': 'contiguous_predeclared_source_order_no_selection_or_regrouping',
    'rule_families': ['explicit_entity_color_lookup', 'latest_recorded_color'],
    'fit_passes_per_world': 16, 'seeds': 1,
    'claim_scope': 'two_synthetic_rules_new_entities_one_seed_feasibility_not_broad_generalization',
    'fresh_development_required': True,
    'historical_exposed_development_is_regression_only': True,
}
BUDGET_DERIVATION = {
    'baseline_forwards': '8*8=64', 'teacher_forwards': '32*8 generation + 32 target forwards = 288',
    'fitting_updates': '32 worlds * 16 passes = 512; one separately charged restart update',
    'fit_probe_forwards': '4 prefixes * 32 worlds * (2 parity + 6 forced + 6*8 generated) = 7168',
    'development_forwards': '16 worlds * 6 controls * 8 tokens = 768',
    'correction_forwards': 'corrected contextual, corrected exact, revoked absent * 8 = 24',
    'total_forwards': '64 + 288 + 2 + 512 + 7168 + 1 + 768 + 24 + 2 = 8829',
    'stage_wall': '7080 seconds inside a 7200-second whole-process-group envelope',
    'artifact_accounting': '126 MiB audit/store + 2 MiB teacher cache + 128/384/1024 MiB checkpoint',
    'failed_or_interrupted_work': 'reserved_before_dispatch_no_refunds',
    'I/O_costs': 'logical/file/serialized bytes and synchronized wall; not physical SSD write amplification',
}


def plan(arm):
    checkpoint = {'two': 128, 'full_attention_nine': 384, 'all': 1024}[arm] * 1024**2
    result = {
        'admission': stage(120, artifact_bytes=126*1024**2, writes=1),
        'load': stage(600, model_loads=1),
        'baseline': stage(240, forwards=64, input_tokens=64*256, generated_tokens=64),
        'numerical': stage(120, forwards=2, input_tokens=2*256, source_encodes=1),
        'prepare': stage(120, source_encodes=32, retrievals=32, writes=64),
        'teacher': stage(300, forwards=288, input_tokens=288*256, generated_tokens=256,
                         writes=33, artifact_bytes=2*1024**2),
        'checkpoint': stage(180, forwards=1, input_tokens=256, source_encodes=1,
                            artifact_bytes=checkpoint, writes=1),
        'development': stage(600, forwards=768, input_tokens=768*256, generated_tokens=768,
                             source_encodes=16, retrievals=16, writes=32),
        'correction': stage(120, forwards=24, input_tokens=24*256, generated_tokens=24,
                            source_encodes=1, retrievals=2, writes=4),
        'restart': stage(120, forwards=2, backwards=1, updates=1, input_tokens=2*256,
                         model_loads=1, source_encodes=2, retrievals=1, writes=1),
    }
    for end in PREFIXES:
        result[f'fit_{end:03d}'] = stage(240, forwards=128, backwards=128, updates=128,
                                        input_tokens=128*256, source_encodes=128)
        result[f'fit_probes_{end:03d}'] = stage(900, forwards=1792, input_tokens=1792*256,
                                              generated_tokens=1536, source_encodes=96,
                                              retrievals=32, writes=64)
    return result


def validate_protocol(protocol):
    if (protocol.get('format') != FORMAT or protocol.get('training') != TRAINING
            or protocol.get('teacher') != TEACHER or protocol.get('coverage') != COVERAGE
            or protocol.get('budget_derivation') != BUDGET_DERIVATION
            or protocol.get('fit_probe_prefixes') != list(PREFIXES)
            or protocol.get('fit_curve_selection') != 'all_512_updates_fixed_final_prefix_all_groups_only_opens_development'):
        raise ValueError('fixed coverage, group criteria or training contract differs')
    for arm, spec in protocol['arms'].items():
        if spec['budget'] != plan(arm) or spec['total_reservations'] != totals(plan(arm)):
            raise ValueError('complete coverage stage budget differs')
    # All original source, numerical, competence, eight-world fit/utility and
    # resource requirements are still checked. Only the strictly verified
    # larger schedule and teacher count project to the historical validator.
    common = deepcopy(protocol)
    common.update(format=PREVIOUS_FORMAT, training=deepcopy(PREVIOUS_TRAINING),
                  teacher=deepcopy(PREVIOUS_TEACHER), fit_probe_prefixes=[32, 64, 96, 128],
                  fit_curve_selection='all_128_updates_fixed_final_prefix_only_opens_development')
    for arm, spec in common['arms'].items():
        spec['budget'] = previous_plan(arm)
        spec['total_reservations'] = totals(spec['budget'])
    validate_previous(common)
