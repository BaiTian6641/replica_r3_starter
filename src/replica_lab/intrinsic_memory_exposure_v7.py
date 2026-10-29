"""Fixed 128-update direct-CE exposure curve; no adaptive stopping or new loss."""
from copy import deepcopy

from .intrinsic_memory_budget_v5 import COUNTERS
from .intrinsic_memory_execution_v5 import validate_protocol as validate_common_v5

FORMAT = 'replica.intrinsic_memory.exposure_curve.v7'
PACKET = 'replica.intrinsic_memory.launch_packet.v7'
AUTHORIZATION = 'replica.intrinsic_memory.external_authorization.v7'
PREFIXES = (32, 64, 96, 128)
TRAINING = {'updates': 128, 'restart_updates': 1, 'lr': 0.001, 'seed': 17,
            'batch_size': 1, 'schedule': 'fixed_round_robin_8_worlds_16_passes',
            'teacher_updates': 0}


def stage(wall, **counts):
    return {**{k: 0 for k in COUNTERS}, 'wall_seconds': wall, **counts}


def v5_plan(arm):
    """The unchanged v5 common-stage arithmetic, for strict compatibility checks."""
    checkpoint = {'two': 128, 'full_attention_nine': 384, 'all': 1024}[arm] * 1024**2
    return {
        'admission': stage(120, artifact_bytes=32*1024**2, writes=1),
        'load': stage(600, model_loads=1),
        'baseline': stage(600, forwards=64, input_tokens=64*256, generated_tokens=64),
        'numerical': stage(120, forwards=2, input_tokens=2*256, source_encodes=1),
        'prepare': stage(120, source_encodes=8, retrievals=8, writes=16),
        'fit': stage(1200, forwards=32, backwards=32, updates=32, input_tokens=32*256, source_encodes=32),
        'fit_probes': stage(2100, forwards=448, input_tokens=448*256, generated_tokens=384,
                            source_encodes=24, retrievals=8, writes=16),
        'checkpoint': stage(180, forwards=1, input_tokens=256, source_encodes=1,
                            artifact_bytes=checkpoint, writes=1),
        'development': stage(900, forwards=384, input_tokens=384*256, generated_tokens=384,
                             source_encodes=8, retrievals=8, writes=16),
        'correction': stage(180, forwards=24, input_tokens=24*256, generated_tokens=24,
                            source_encodes=1, retrievals=2, writes=4),
        'restart': stage(900, forwards=2, backwards=1, updates=1, input_tokens=2*256,
                         model_loads=1, source_encodes=2, retrievals=1, writes=1),
    }


def plan(arm):
    old = v5_plan(arm)
    result = {k: v for k, v in old.items() if k not in ('fit', 'fit_probes')}
    for end in PREFIXES:
        result[f'fit_{end:03d}'] = {**old['fit'], 'wall_seconds': 240}
        result[f'fit_probes_{end:03d}'] = {**old['fit_probes'], 'wall_seconds': 600}
    return result


def totals(budget):
    return {k: sum(s[k] for s in budget.values()) for k in next(iter(budget.values()))}


def validate_protocol(protocol):
    if (protocol.get('format') != FORMAT or protocol.get('training') != TRAINING
            or protocol.get('fit_probe_prefixes') != list(PREFIXES)
            or protocol.get('fit_curve_selection') != 'all_128_updates_fixed_final_prefix_only_opens_development'):
        raise ValueError('fixed exposure curve differs')
    for arm, spec in protocol['arms'].items():
        if spec['budget'] != plan(arm) or spec['total_reservations'] != totals(plan(arm)):
            raise ValueError('complete v7 stage work/wall budget differs')
        if spec['wall_seconds'] != 7200:
            raise ValueError('outer deadline differs')
    # Check all shared source, numerical, ownership, competence and utility
    # contracts using the unchanged v5 validator. Only its known old training
    # schedule and budget are projected; actual v7 values were checked above.
    common = deepcopy(protocol)
    common['format'] = 'replica.intrinsic_memory.experiment.v5'
    common['training'] = {**TRAINING, 'updates': 32, 'schedule': 'fixed_round_robin_8_worlds_4_passes'}
    for arm, spec in common['arms'].items():
        spec['budget'] = v5_plan(arm)
        spec['total_reservations'] = totals(spec['budget'])
    validate_common_v5(common)
