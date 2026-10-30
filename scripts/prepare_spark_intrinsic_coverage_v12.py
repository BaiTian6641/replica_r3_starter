#!/usr/bin/env python3
"""Prepare the separately versioned official-BF16 v12 native packet.

No original weights are loaded, no native execution starts, no development
label is parsed and no authorization receipt is created. Preparation refuses to
overwrite anything and is idempotent only in the sense of failing closed.

What this writes:

* ``configs/spark_intrinsic_coverage_v12.json`` - the complete protocol,
  including the re-derived work envelope and the explicit difference list;
* ``data/intrinsic_memory_coverage_v12_20261029/split_manifest.json`` - the new
  manifest that reaches the same calibration/fitting/development rows as v11
  with an explicit exposure contract (development is lawful to reuse only
  because the v11 receipt still says ``development_opened: false``);
* ``reports/intrinsic_memory_native_v12/implementation_v12.json`` - the local
  executable import closure, resolved statically;
* ``reports/intrinsic_memory_native_v12/native_packet_two_v12.json`` - the
  single two-site packet. Dense arms stay gated and unpacketized;
* ``reports/intrinsic_memory_native_v12/preparation.json`` - what was and was
  not done, with the exact intended commands.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from replica_lab.intrinsic_memory_budget_v12 import sha_json
from replica_lab.intrinsic_memory_coverage_v12 import (
    ARMS, BUDGET_DERIVATION, COVERAGE, DIFFERENCES_FROM_V11, FORMAT, GATED_ARMS,
    OUTER_WALL_SECONDS, PACKET, PREFIXES, SOURCE_FEATURES, SOURCE_HIDDEN_SIZE,
    SOURCE_VOCAB_SIZE, SLOT_POLICY_V12, TEACHER, TRAINING, UNCHANGED_FROM_V11,
    declared_tokenizer_identity, plan, tokenizer_identity_digest, totals,
    validate_protocol)
from replica_lab.intrinsic_memory_execution_v12 import (
    validate_bindings, validate_predecessor, validate_reused_data,
    validate_source_feature_binding)
from replica_lab.intrinsic_memory_execution_v5 import binding, bound_json, local_closure
from replica_lab.intrinsic_memory_worlds_v11 import public_split_metadata

REPORT = 'reports/intrinsic_memory_native_v12'
DATA = 'data/intrinsic_memory_coverage_v12_20261029'
CONFIG = 'configs/spark_intrinsic_coverage_v12.json'
PREVIOUS_CONFIG = 'configs/spark_intrinsic_coverage_v11.json'
STANDING_AUTHORIZATION = ('Continued as planned, no need more approval for testing, '
                          'follow the goal, assume I authorize the tests.')
OUTPUT_ROOTS = {arm: f'{REPORT}/native_bf16_{arm}_v12_attempt01' for arm in ARMS}


def save_new(root, relative, value):
    path = root/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return path


def refuse_existing(root):
    for relative in (CONFIG, DATA, f'{REPORT}/preparation.json',
                     f'{REPORT}/implementation_v12.json',
                     f'{REPORT}/native_packet_two_v12.json',
                     f'{REPORT}/authorization_two_v12.json',
                     *OUTPUT_ROOTS.values()):
        if (root/relative).exists():
            raise ValueError('v12 preparation already exists or an attempt was consumed: '
                             + relative)


def exposure_contract(root, development_pin, exposure_pin):
    return {
        'calibration': {
            'status': 'already_exposed_in_v11',
            'labels': 'already_exposed_in_v11',
            'reuse': 'matched_comparison_only'},
        'fit': {
            'status': 'already_exposed_in_v11_no_tuning_escape',
            'labels': 'already_exposed_in_v11',
            'reuse': 'matched_comparison_only',
            'group_selection': 'fixed_predeclared_groups_only'},
        'development': {
            'status': 'open_only_after_frozen_final_fit_gate',
            'labels': 'unopened_at_preparation',
            'reuse': 'matched_comparison_only',
            'opened_in_v11': False,
            'exposure_receipt': exposure_pin,
            'development_file_sha256': development_pin['sha256'],
            'preparation_parses_development_labels': False},
    }


def prepare(root=ROOT):
    refuse_existing(root)
    previous = json.loads((root/PREVIOUS_CONFIG).read_text())
    predecessor_pin = binding(root, previous['arms']['two']['output_root']+'/result.json')
    observed = bound_json(root, predecessor_pin)
    if (observed.get('native_competence') is not True
            or observed.get('main', {}).get('fit_memory_dependence_passed') is not False
            or observed.get('useful_recall') is not False):
        raise ValueError('v12 requires the completed v11 learned-fitting but failed-utility reference')
    exposure_pin = binding(root, 'reports/intrinsic_memory_coverage_v11/development_exposure.json')
    exposure = bound_json(root, exposure_pin)
    if exposure.get('development_opened') is not False or exposure.get('fresh_for_successor') is not True:
        raise ValueError('v11 development is no longer lawfully reusable; prepare a fresh split')
    source = bound_json(root, previous['source'])
    tokenizer_identity = declared_tokenizer_identity(source['files'],
                                                     vocab_size=SOURCE_VOCAB_SIZE)
    protocol = deepcopy(previous)
    protocol.update(
        format=FORMAT, slot_policy_v12=SLOT_POLICY_V12,
        source_features=deepcopy(SOURCE_FEATURES),
        training=deepcopy(TRAINING), teacher=deepcopy(TEACHER), coverage=deepcopy(COVERAGE),
        budget_derivation=deepcopy(BUDGET_DERIVATION), fit_probe_prefixes=list(PREFIXES),
        fit_curve_selection='all_512_updates_fixed_final_prefix_all_groups_only_opens_development',
        standing_authorization_text=STANDING_AUTHORIZATION,
        diagnostic_predecessor=predecessor_pin,
        prior_development_exposure=exposure_pin,
        source_feature_binding={
            'tokenizer_identity': tokenizer_identity,
            'tokenizer_sha256': tokenizer_identity_digest(tokenizer_identity),
            'original_parameters_sha256': observed['main']['final_original_parameters_sha256'],
            'source_files_sha256': sha_json(source['files'])},
        exposure_contract=exposure_contract(root, previous['data']['development'], exposure_pin))
    protocol['data'] = deepcopy(previous['data'])
    # The same rows reach v12 only through this new manifest.
    manifest = {'format': 'replica.intrinsic_memory.coverage_split_manifest.v12',
                'generator_policy': 'predeclared_source_order_revision_oracle_no_generated_explanations',
                'reuse_policy': 'matched_v11_rows_via_new_v12_manifest_development_labels_unread',
                'exposure_contract': deepcopy(protocol['exposure_contract']),
                'splits': {}}
    for split in ('calibration', 'fit', 'development'):
        pin = protocol['data'][split]
        if (root/pin['path']).is_symlink() or not (root/pin['path']).is_file():
            raise ValueError('reused prepared split is missing: '+split)
        # Only the digest is used here. Development labels are never parsed.
        if binding(root, pin['path'])['sha256'] != pin['sha256']:
            raise ValueError('reused prepared split digest differs: '+split)
        manifest['splits'][split] = {
            'file': pin, 'public_identity': public_split_metadata(split),
            'label_exposure': protocol['exposure_contract'][split]['labels']}
    save_new(root, DATA+'/split_manifest.json', manifest)
    protocol['data_manifest'] = binding(root, DATA+'/split_manifest.json')
    entries = ['scripts/run_spark_intrinsic_coverage_v12.py',
               'scripts/prepare_spark_intrinsic_coverage_v12.py',
               'src/replica_lab/intrinsic_memory_native_v12.py',
               'scripts/launch_spark_intrinsic_readout_environment_v1.py']
    save_new(root, REPORT+'/implementation_v12.json', {
        'format': 'replica.intrinsic_memory.local_execution_closure.v12',
        'entries': entries, 'files': local_closure(root, entries),
        'policy': ('frozen_source_feature_mechanism_new_ledger_and_adapter_contracts_'
                   'no_sealed_execution_mutation')})
    protocol['implementation'] = binding(root, REPORT+'/implementation_v12.json')
    protocol['experiment_variant'] = {
        'id': 'frozen_source_features_contextual_compression_v12',
        'predecessor': binding(root, PREVIOUS_CONFIG),
        'reason': ('v11 learns its fitting memories but does not transfer useful recall; test a '
                   'different source representation (frozen canonical-source hidden states with '
                   'a live trainable compressor) under unchanged reader, exposure and gates'),
        'changed': list(DIFFERENCES_FROM_V11),
        'unchanged': list(UNCHANGED_FROM_V11),
        'historical_results_rewritten': False,
        'prior_development_reused': True,
        'prior_development_reuse_basis': exposure_pin,
        'isolated_causal_attribution': False,
        'scope_limit': ('one mechanism substitution at matched exposure; no claim that the '
                        'source representation alone explains any change'),
        'all_choices_fixed_before_v12_native_answers': True,
        'teacher_updates': 0, 'persistent_P_updates': 0,
        'prelaunch_resource_review': {
            'outer_wall_seconds': OUTER_WALL_SECONDS,
            'allocated_stage_seconds': totals(plan('two'))['wall_seconds'],
            'driver_bytes': 50*1024**3, 'swap_growth_bytes': 512*1024**2,
            'minimum_host_free_percent': 15, 'maximum_workers': 1,
            'source_hidden_size': SOURCE_HIDDEN_SIZE},
    }
    for arm, spec in protocol['arms'].items():
        spec.update(output_root=OUTPUT_ROOTS[arm], budget=plan(arm),
                    total_reservations=totals(plan(arm)),
                    wall_seconds=OUTER_WALL_SECONDS)
    validate_protocol(protocol)
    validate_predecessor(root, protocol)
    validate_source_feature_binding(root, protocol)
    save_new(root, CONFIG, protocol)
    validate_reused_data(root, protocol)
    packet_name = f'{REPORT}/native_packet_two_v12.json'
    authorization = f'{REPORT}/authorization_two_v12.json'
    save_new(root, packet_name, {
        'format': PACKET, 'experiment_version': 12, 'native_engine_version': 12,
        'protocol': binding(root, CONFIG), 'arm': 'two', 'attempts': 1, 'retries': 0,
        'output_root': OUTPUT_ROOTS['two'],
        'authorization_required': authorization,
        'authorization_status': 'external_receipt_required_not_created',
        'gated_arms_not_prepared': list(GATED_ARMS),
        'commands': {
            'focused_tests': ['.venv/bin/python', '-m', 'pytest', '-q',
                              'tests/test_intrinsic_budget_v12.py',
                              'tests/test_intrinsic_coverage_v12.py',
                              'tests/test_intrinsic_native_v12.py'],
            'environment_only': ['.venv-spark-bf16/bin/python',
                                 'scripts/run_spark_intrinsic_coverage_v12.py',
                                 '--packet', packet_name, '--environment-only'],
            'review': ['.venv-spark-bf16/bin/python',
                       'scripts/run_spark_intrinsic_coverage_v12.py',
                       '--packet', packet_name, '--review'],
            'future_launch': ['.venv-spark-bf16/bin/python',
                              'scripts/run_spark_intrinsic_coverage_v12.py',
                              '--packet', packet_name, '--authorization', authorization,
                              '--launch']}})
    _, _, receipt = validate_bindings(root, packet_name, check_source=False)
    save_new(root, REPORT+'/preparation.json', {
        'format': 'replica.intrinsic_memory.native_preparation.v12',
        'native_run': False, 'weights_loaded': False, 'native_authorization_created': False,
        'development_labels_parsed': False, 'history_written': False,
        'protocol': binding(root, CONFIG), 'packet': binding(root, packet_name),
        'data_manifest': binding(root, DATA+'/split_manifest.json'),
        'implementation': binding(root, REPORT+'/implementation_v12.json'),
        'authorization_path_reserved_not_created': authorization,
        'differences_from_v11': list(DIFFERENCES_FROM_V11),
        'unchanged_from_v11': list(UNCHANGED_FROM_V11),
        'budget_derivation': deepcopy(BUDGET_DERIVATION),
        'two_site_total_reservations': protocol['arms']['two']['total_reservations'],
        'two_site_stage_walls': {k: v['wall_seconds']
                                 for k, v in protocol['arms']['two']['budget'].items()},
        'preparation_checks': receipt,
        'next_step': ('independent review of these bytes, then a truthful external '
                      'authorization receipt, then exactly one guarded launch'),
    })
    return {'prepared': True, 'native_run': False, 'native_engine_version': 12,
            'authorization_created': False,
            'two_site_maximums': protocol['arms']['two']['total_reservations']}


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
