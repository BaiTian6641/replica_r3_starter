"""v12 protocol and envelope tests: derivation, projection, tamper rejection.

The completed v11 configuration is read as the comparison reference; nothing is
written and no native work is started.
"""
from copy import deepcopy
import json
from pathlib import Path

import re

import pytest

from replica_lab.intrinsic_memory_budget_v12 import COUNTERS, validate_plan
from replica_lab.intrinsic_memory_coverage_v12 import (
    ARMS, BUDGET_DERIVATION, COVERAGE, DIFFERENCES_FROM_V11, FEATURE_CACHE_ENTRIES,
    ORIGIN_ARTIFACT_BYTES,
    FORMAT, GROUP_SIZE, OUTER_WALL_SECONDS, PREFIXES, REPRESENTATION_VALUE, SLOT_POLICY_V12,
    SOURCE_FEATURES, SOURCE_MAX_BYTES, SOURCE_MAX_TOKENS, TEACHER, TRAINING,
    UNCHANGED_FROM_V11, declared_tokenizer_identity, plan, tokenizer_identity_digest,
    totals, validate_protocol, validate_source_feature_binding)
from replica_lab.intrinsic_memory_budget_v12 import sha_json
from replica_lab.intrinsic_memory_coverage_v11 import (
    plan as previous_plan, validate_protocol as validate_previous_protocol)
from replica_lab.contextual_source_features_v12 import tokenizer_contract_digest

ROOT = Path(__file__).resolve().parents[1]
V11 = json.loads((ROOT/'configs/spark_intrinsic_coverage_v11.json').read_text())
SOURCE = json.loads((ROOT/'reports/intrinsic_memory_successor_v5/official_bf16_source.json').read_text())


def binding():
    identity = declared_tokenizer_identity(SOURCE['files'])
    return {'tokenizer_identity': identity,
            'tokenizer_sha256': tokenizer_identity_digest(identity),
            'original_parameters_sha256': (
                '39ea150a4d2ed39184f2a83fbd474b67644e4d38f4039177c72cedd069ef4747'),
            'source_files_sha256': sha_json(SOURCE['files'])}


def candidate():
    protocol = deepcopy(V11)
    protocol.update(format=FORMAT, slot_policy_v12=SLOT_POLICY_V12,
                    source_features=deepcopy(SOURCE_FEATURES),
                    source_feature_binding=binding(),
                    training=deepcopy(TRAINING), teacher=deepcopy(TEACHER),
                    coverage=deepcopy(COVERAGE), budget_derivation=deepcopy(BUDGET_DERIVATION),
                    fit_probe_prefixes=list(PREFIXES),
                    fit_curve_selection=('all_512_updates_fixed_final_prefix_all_groups_only_'
                                         'opens_development'),
                    experiment_variant={'changed': list(DIFFERENCES_FROM_V11),
                                        'unchanged': list(UNCHANGED_FROM_V11),
                                        'historical_results_rewritten': False})
    for arm, spec in protocol['arms'].items():
        spec.update(budget=plan(arm), total_reservations=totals(plan(arm)),
                    wall_seconds=OUTER_WALL_SECONDS)
    return protocol


def test_complete_v12_protocol_passes_and_keeps_the_v11_projection():
    protocol = candidate()
    validate_protocol(protocol)
    # The sealed v11 validator still accepts the inherited contracts when the
    # v12-only fields and the re-derived envelope are projected away.
    common = deepcopy(protocol)
    common.update(format='replica.intrinsic_memory.coverage.v11',
                  training=deepcopy(V11['training']), teacher=deepcopy(V11['teacher']),
                  coverage=deepcopy(V11['coverage']),
                  budget_derivation=deepcopy(V11['budget_derivation']),
                  fit_probe_prefixes=list(V11['fit_probe_prefixes']),
                  fit_curve_selection=V11['fit_curve_selection'])
    for name in ('source_features', 'source_feature_binding', 'slot_policy_v12'):
        common.pop(name)
    for arm, spec in common['arms'].items():
        spec.update(budget=previous_plan(arm), total_reservations=totals(previous_plan(arm)))
    validate_previous_protocol(common)
    assert V11['slot_policy'] == protocol['slot_policy'] == 'fixed_eight_masked_padding_v5'
    assert protocol['slot_policy_v12'] == 'fixed_eight_masked_padding_v12'


def test_stage_plan_is_complete_finite_and_inside_the_outer_envelope():
    for arm in ARMS:
        stages = plan(arm)
        validate_plan(stages)
        for stage, caps in stages.items():
            assert set(caps) == set(COUNTERS)|{'wall_seconds'}, stage
            assert all(type(caps[k]) is int and caps[k] >= 0 for k in COUNTERS)
        total = totals(stages)
        assert total['wall_seconds'] == 6925
        assert total['wall_seconds'] <= OUTER_WALL_SECONDS-11
        assert total['updates'] == 513
        assert total['model_loads'] == 2
        assert total['backwards'] == 513
        assert 'finalize_worker' in stages and 'finalize_parent' in stages
        assert 'capacity_main' in stages and 'capacity_restart' in stages
        assert stages['finalize_worker']['writes'] >= 2
        assert stages['restart']['prompt_tokenizer_calls'] >= 2*32 + 3
        assert stages['restart']['wall_seconds'] > stages['checkpoint']['wall_seconds']
    assert plan('two')['checkpoint']['artifact_bytes'] == 96*1024**2
    stages = plan('two')
    origin = 32*ORIGIN_ARTIFACT_BYTES*4 + ORIGIN_ARTIFACT_BYTES   # four probes + restart
    assert (stages['admission']['artifact_bytes']
            + stages['teacher']['artifact_bytes']
            + stages['checkpoint']['artifact_bytes']
            + origin) == totals(stages)['artifact_bytes']


def test_derivation_strings_match_the_computed_totals():
    total = totals(plan('two'))
    for key, field in (('forwards', 'total_forwards'), ('backwards', 'total_backwards'),
                       ('updates', 'total_updates'), ('generated_tokens', 'total_generated_tokens'),
                       ('retrievals', 'total_retrievals'),
                       ('source_encodes', 'total_tokenizer_passes'),
                       ('source_feature_forwards', 'total_source_forwards'),
                       ('input_tokens', 'total_input_tokens'),
                       ('prompt_tokenizer_calls', 'total_prompt_tokenizer_passes')):
        assert str(total[key]) in BUDGET_DERIVATION[field].replace('_', ''), (key, field)
    assert str(total['writes']) in BUDGET_DERIVATION['total_writes']
    assert str(total['artifact_read_bytes']) in BUDGET_DERIVATION['artifact_read_accounting']
    assert str(total['wall_seconds']) in BUDGET_DERIVATION['stage_wall']


def test_source_feature_declaration_matches_the_mechanism_constants():
    assert SOURCE_FEATURES['max_tokens'] == SOURCE_MAX_TOKENS == 256
    assert SOURCE_FEATURES['cache_bytes'] == 32*1024**2
    assert SOURCE_FEATURES['cache_entries'] == FEATURE_CACHE_ENTRIES == 64
    assert SOURCE_FEATURES['input_policy'] == 'canonical_span_only_no_template_no_special_tokens'
    assert SOURCE_FEATURES['output_policy'] == 'last_decoder_hidden_after_final_norm'
    assert SOURCE_FEATURES['feature_dtype'] == 'float32'
    assert SOURCE_FEATURES['original_updates'] == 0
    assert SOURCE_FEATURES['representation'] == REPRESENTATION_VALUE
    assert SOURCE_MAX_BYTES == 256


def test_declared_tokenizer_digest_matches_the_producer_digest():
    identity = declared_tokenizer_identity(SOURCE['files'])

    class FakeTokenizer:
        def identity(self):
            return dict(identity)

    assert tokenizer_contract_digest(FakeTokenizer()) == tokenizer_identity_digest(identity)
    assert validate_source_feature_binding({**binding()}) == binding()
    with pytest.raises(ValueError): validate_source_feature_binding(
        {**binding(), 'unexpected': 'field'})


def test_tokenizer_identity_requires_the_verified_file_set():
    with pytest.raises(ValueError, match='tokenizer files'):
        declared_tokenizer_identity([{'path': 'a/tokenizer.json', 'sha256': '0'*64}])
    with pytest.raises(ValueError, match='vocabulary size'):
        declared_tokenizer_identity(SOURCE['files'], vocab_size=0)


@pytest.mark.parametrize('damage', [
    'format', 'slot_policy', 'slot_policy_v12', 'source_features', 'feature_binding',
    'tokenizer_digest', 'training_updates', 'teacher_cases', 'coverage_groups',
    'prefixes', 'curve_selection', 'budget_forward', 'stage_wall', 'checkpoint_artifact',
    'derivation', 'outer_wall', 'arm_sites', 'differences', 'restart_prompt_cap',
    'restart_wall', 'read_bytes_value',
])
def test_no_silent_contract_or_envelope_change(damage):
    protocol = candidate()
    if damage == 'format':
        protocol['format'] = 'replica.intrinsic_memory.coverage.v11'
    elif damage == 'slot_policy':
        protocol['slot_policy'] = SLOT_POLICY_V12
    elif damage == 'slot_policy_v12':
        protocol['slot_policy_v12'] = 'fixed_eight_masked_padding_v5'
    elif damage == 'source_features':
        protocol['source_features']['cache_entries'] = 32
    elif damage == 'feature_binding':
        protocol['source_feature_binding'].pop('source_files_sha256')
    elif damage == 'tokenizer_digest':
        protocol['source_feature_binding']['tokenizer_sha256'] = 'a'*64
    elif damage == 'training_updates':
        protocol['training']['updates'] = 511
    elif damage == 'teacher_cases':
        protocol['teacher']['qualification']['cases'] = 31
    elif damage == 'coverage_groups':
        protocol['coverage']['fit_groups'] = 2
    elif damage == 'prefixes':
        protocol['fit_probe_prefixes'] = [64, 128, 256, 512]
    elif damage == 'curve_selection':
        protocol['fit_curve_selection'] = 'best_prefix_may_open_development'
    elif damage == 'budget_forward':
        protocol['arms']['two']['budget']['fit_128']['forwards'] -= 1
    elif damage == 'stage_wall':
        protocol['arms']['two']['budget']['fit_128']['wall_seconds'] += 1
    elif damage == 'checkpoint_artifact':
        protocol['arms']['two']['budget']['checkpoint']['artifact_bytes'] = 128*1024**2
    elif damage == 'derivation':
        protocol['budget_derivation']['stage_wall'] = 'inherited 7080 seconds'
    elif damage == 'outer_wall':
        protocol['arms']['two']['wall_seconds'] = 7300
    elif damage == 'arm_sites':
        protocol['arms']['two']['sites'] = [3, 8]
    elif damage == 'differences':
        protocol['experiment_variant']['changed'] = list(DIFFERENCES_FROM_V11)[:-1]
    elif damage == 'restart_prompt_cap':
        # The review's P0: a cap below the 67 real passes must be refused.
        protocol['arms']['two']['budget']['restart']['prompt_tokenizer_calls'] = 4
    elif damage == 'restart_wall':
        protocol['arms']['two']['budget']['restart']['wall_seconds'] = 140
    elif damage == 'read_bytes_value':
        protocol['budget_derivation']['artifact_read_bytes_value'] = 0
    with pytest.raises(ValueError): validate_protocol(protocol)


def test_fresh_development_split_is_independent_deterministic_and_sealed():
    from replica_lab.intrinsic_memory_coverage_v12 import (
        COMPOUND_FAMILIES, DEVELOPMENT_CASES, DEVELOPMENT_ENTITIES, DEVELOPMENT_ROOTS,
        DEVELOPMENT_WORLD_PREFIX, RETIRED_V11_FAMILIES, v12_development_public_identity,
        v12_development_worlds, validate_v12_development)
    from replica_lab.intrinsic_memory_worlds_v11 import ENTITIES as V11_ENTITIES
    rows = v12_development_worlds()
    assert validate_v12_development(v12_development_worlds()) is True
    assert rows == v12_development_worlds()          # deterministic
    assert len(rows) == DEVELOPMENT_CASES == 16
    assert all(r['world_id'].startswith(DEVELOPMENT_WORLD_PREFIX) for r in rows)
    assert {r['family'] for r in rows} == set(DEVELOPMENT_ROOTS)
    v11_entities = {name for names in V11_ENTITIES.values() for name in names}
    assert not set(DEVELOPMENT_ENTITIES) & v11_entities
    assert not {r['query_entity'] for r in rows} & v11_entities
    for start in range(0, DEVELOPMENT_CASES, GROUP_SIZE):
        group = rows[start:start+GROUP_SIZE]
        assert sum(r['answer'] == 'unknown' for r in group) == 2
        assert len({r['family'] for r in group}) == 4
        assert len({r['rule_family'] for r in group}) == 1
    identity = v12_development_public_identity()
    assert identity['label_content_included'] is False
    assert set(identity) == {'split', 'role', 'cases', 'entities', 'decision_roots',
                             'seed', 'generator', 'rule_families', 'plan_sha256',
                             'label_content_included'}
    assert identity['role'] == 'fresh_v12_development'
    assert identity['generator'] == 'seeded_compound_rule_holdout_v12'
    assert identity['rule_families'] == list(COMPOUND_FAMILIES)
    assert not set(identity['rule_families']) & set(RETIRED_V11_FAMILIES)
    assert 'rows' not in identity and 'answer' not in identity and 'colors' not in identity
    # Per-family composition is declared and never the retired single-fact or
    # latest-revision lookup.
    assert {row['criterion_group']: row['rule_family'] for row in rows} == {
        0: COMPOUND_FAMILIES[0], 1: COMPOUND_FAMILIES[1]}
    assert {row['composition'] for row in rows} == {
        'relation_plus_partner_attribute', 'equality_relation_plus_anchor_attribute'}


def test_default_split_is_a_compound_rule_family_holdout():
    """X0: the fresh split holds out compounds, not a same-family redraw.

    The comparison uses only public generator code and the documented retired
    structure; neither development data file is read.
    """
    from replica_lab.intrinsic_memory_coverage_v12 import (
        COMPOUND_FAMILIES, DEVELOPMENT_ROOTS, DEVELOPMENT_SEED, RETIRED_V11_FAMILIES,
        v12_development_case_positions, v12_development_plan)
    plan = v12_development_plan()
    positions = v12_development_case_positions(plan)
    assert len(positions) == 16
    assert [entry['family'] for entry in plan] == list(COMPOUND_FAMILIES)
    assert not set(COMPOUND_FAMILIES) & set(RETIRED_V11_FAMILIES)
    retired = {'pair_order': [0, 0, 1, 1, 2, 2, 3, 3],
               'unknown_positions': {4, 7},
               'rotation': {0: 2, 1: 3},
               'case_order': [0, 1, 2, 3]}
    differences = []
    if [p['pair'] for p in positions[:8]] != retired['pair_order']:
        differences.append('pair_order')
    if {i for i, p in enumerate(positions[:8]) if p['unknown']}             != retired['unknown_positions']:
        differences.append('unknown_positions')
    if {entry['group']: entry['rotation'] for entry in plan} != retired['rotation']:
        differences.append('rotation')
    if any(entry['order'] != retired['case_order'] for entry in plan):
        differences.append('case_order')
    differences.append('rule_semantics')
    assert len(differences) >= 3, differences
    assert DEVELOPMENT_SEED == 2026102912
    assert len(DEVELOPMENT_ROOTS) == 8


def test_unknown_cases_render_no_answer_bearing_text():
    """Compound blocker: an unrecorded target must not print a colour as its answer."""
    from replica_lab.intrinsic_memory_coverage_v12 import (
        COMPOUND_FAMILIES, DEVELOPMENT_PALETTE, v12_development_worlds)
    rows = v12_development_worlds()
    unknown = [row for row in rows if row['answer'] == 'unknown']
    assert len(unknown) == 4                      # two per group
    for row in unknown:
        target = row['facts']['target']
        assert f"recorded color of {target} is" not in row['text']
        assert f"No color is recorded for {target}." in row['text']
        assert row['facts']['unrecorded_target'] is True
        # Word-boundary colour words only: "recorded" is not "red".
        printed = re.findall(r'\b(?:red|blue|green|gold)\b', row['text'])
        assert printed == [row['facts']['anchor_colour']] or printed == []
    # The known sibling of each pair does print its answer, so the pair differs.
    for row in rows:
        if row['answer'] in DEVELOPMENT_PALETTE:
            assert row['facts']['unrecorded_target'] is False
            assert f"The recorded color of {row['facts']['target']} is {row['answer']}." in row['text']
    # False-recall control: an unknown target is scored as `unknown`, never a colour.
    from replica_lab.baseline_competence import score
    assert all(score(row['answer'], 'unknown') for row in unknown)
    assert not any(score(row['answer'], colour) for row in unknown
                   for colour in DEVELOPMENT_PALETTE)
    assert {row['rule_family'] for row in unknown} == set(COMPOUND_FAMILIES)


@pytest.mark.parametrize('damage', ['v11_entity', 'world_prefix', 'group_order',
                                    'root', 'answer', 'duplicate_id', 'length'])
def test_fresh_development_validation_rejects_damage(damage):
    from replica_lab.intrinsic_memory_coverage_v12 import (
        v12_development_worlds, validate_v12_development)
    rows = v12_development_worlds()
    if damage == 'v11_entity':
        rows[0]['query_entity'] = 'Arven'          # a v11 calibration entity
    elif damage == 'world_prefix':
        rows[0]['world_id'] = 'v11_development_00'
    elif damage == 'group_order':
        rows[0]['criterion_group'] = 1
    elif damage == 'root':
        rows[0]['family'] = 'v11_development_decision_00'
    elif damage == 'answer':
        rows[0]['answer'] = 'orange'
    elif damage == 'duplicate_id':
        rows[1]['world_id'] = rows[0]['world_id']
    elif damage == 'length':
        rows[0]['text'] = 'The recorded color of x is red. ' * 20
    with pytest.raises(ValueError):
        validate_v12_development(rows)


def test_fresh_development_never_reuses_the_retired_v11_split():
    """The current exposure contract must retire v11 development, not extend it."""
    from replica_lab.intrinsic_memory_coverage_v12 import v12_development_worlds
    v11_paths = {pin['path'] for pin in V11['data'].values()}
    assert not ({'data/intrinsic_memory_coverage_v12_20261029/development.json'} & v11_paths)
    rows = v12_development_worlds()
    assert all('v11_' not in row['world_id'] for row in rows)
    assert all('v11_' not in row['family'] for row in rows)


def test_dense_arms_and_group_criteria_stay_declared_but_gated():
    protocol = candidate()
    assert protocol['persistent_P_updates'] == 0
    assert protocol['automatic_promotion'] is False
    assert protocol['coverage']['fitting_criterion'] == (
        'all_four_fixed_groups_must_pass_unchanged_eight_world_fit_gate')
    assert protocol['coverage']['utility_criterion'] == (
        'both_fixed_groups_must_pass_unchanged_eight_world_utility_gate')
    assert protocol['arms']['full_attention_nine']['budget'] != plan('two')
    assert plan('two')['checkpoint']['artifact_bytes'] < plan('all')['checkpoint']['artifact_bytes']
