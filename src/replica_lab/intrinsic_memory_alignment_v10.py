"""Predeclared frozen-teacher comparison with the completed v9 direct-CE control."""
from copy import deepcopy

from .intrinsic_memory_exposure_v7 import plan as previous_plan, stage, totals
from .intrinsic_memory_smaller_step_v9 import (
    FORMAT as PREVIOUS_FORMAT, TRAINING, validate_protocol as validate_previous,
)

FORMAT = 'replica.intrinsic_memory.teacher_alignment.v10'
PACKET = 'replica.intrinsic_memory.launch_packet.v10'
AUTHORIZATION = 'replica.intrinsic_memory.external_authorization.v10'
TEACHER = {
    'objective_format': 'replica.contextual_teacher_alignment.v1',
    'top_k': 64, 'temperature': 1.0, 'kl_weight': 1.0, 'ce_weight': 0.0,
    'source': 'same_frozen_original_extensions_disabled_selected_exact_text',
    'qualification': {'cases': 8, 'correct_min': 8, 'false_recall_max': 0,
                      'truncation_max': 0, 'termination_required': True},
    'data': 'fit_only_after_frozen_baseline',
    'alignment': 'same_response_tokens_by_response_index_including_one_EOS',
    'student_normalization': 'full_vocabulary',
    'target_cache': 'eight_detached_targets_created_once_reused_128_plus_1_updates',
    'cache_bytes_max': 2 * 1024**2,
    'base_updates': 0, 'teacher_backward_calls': 0,
    'development_teacher_targets': 0,
}


def plan(arm):
    result = previous_plan(arm)
    # Reallocate existing wall/artifact envelopes; do not multiply the native
    # budget or reduce any competence/generation criterion.
    result['baseline']['wall_seconds'] = 480
    result['admission']['artifact_bytes'] = 30 * 1024**2
    result['teacher'] = stage(120, forwards=72, input_tokens=72*256,
                              generated_tokens=64, writes=9, artifact_bytes=2*1024**2)
    return result


def validate_protocol(protocol):
    if (protocol.get('format') != FORMAT or protocol.get('training') != TRAINING
            or protocol.get('teacher') != TEACHER):
        raise ValueError('fixed teacher-alignment contract differs')
    for arm, spec in protocol['arms'].items():
        if spec['budget'] != plan(arm) or spec['total_reservations'] != totals(plan(arm)):
            raise ValueError('complete teacher stage work/wall/artifact budget differs')
    common = deepcopy(protocol)
    common['format'] = PREVIOUS_FORMAT
    for arm, spec in common['arms'].items():
        spec['budget'] = previous_plan(arm)
        spec['total_reservations'] = totals(spec['budget'])
    validate_previous(common)
