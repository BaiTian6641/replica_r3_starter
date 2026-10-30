#!/usr/bin/env python3
"""Prepare the separately versioned official-BF16 v12 native packet.

No original weight is loaded, no native execution starts and no authorization
receipt is created. Preparation refuses to overwrite anything and is idempotent
only in the sense of failing closed.

What this writes:

* ``data/intrinsic_memory_coverage_v12_20261029/development.json`` - a fresh
  v12 development split with new entities, world ids and decision roots. The
  labels are produced deterministically by the sealed structured oracle; the
  preparation never inspects or selects on them, and the file is pinned before
  any observation. Calibration and fitting keep the exposed v11 rows so the
  comparison stays matched;
* ``reports/intrinsic_memory_native_v12/development_exposure_v12.json`` - the
  current exposure receipt: the new split is unopened, and the v11 development
  split is retired with the re-review reading incident recorded. Historical
  exposure receipts are never edited;
* ``reports/intrinsic_memory_native_v12/environment_closure_v12.json`` - the
  executable dependency closure (interpreter, module origins, native
  extensions, torch/lib, wheel RECORDs, critical files) with actual hashes;
* ``configs/spark_intrinsic_coverage_v12.json`` - the complete protocol;
* ``data/intrinsic_memory_coverage_v12_20261029/split_manifest.json`` - the
  role-bound split manifest with an explicit exposure contract;
* ``reports/intrinsic_memory_native_v12/implementation_v12.json`` - the local
  executable import closure, resolved statically;
* ``reports/intrinsic_memory_native_v12/native_packet_two_v12.json`` - the
  single two-site packet, including its durable attempt-marker path. Dense arms
  stay gated and unpacketized;
* ``reports/intrinsic_memory_native_v12/preparation.json`` - what was and was
  not done, with the exact intended commands.
"""
from copy import deepcopy
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from replica_lab.intrinsic_memory_budget_v12 import sha_json
from replica_lab.intrinsic_memory_coverage_v12 import (
    ARMS, BUDGET_DERIVATION, COVERAGE, DIFFERENCES_FROM_V11, ENVIRONMENT_CLOSURE_BYTES,
    ENVIRONMENT_CLOSURE_FILES, FORMAT, GATED_ARMS, OUTER_WALL_SECONDS, PACKET, PREFIXES,
    SOURCE_FEATURES, SOURCE_HIDDEN_SIZE, SOURCE_VOCAB_SIZE, SLOT_POLICY_V12, TEACHER,
    TRAINING, UNCHANGED_FROM_V11, declared_tokenizer_identity, plan,
    tokenizer_identity_digest, totals, v12_development_public_identity,
    v12_development_worlds, validate_protocol, validate_v12_development)
from replica_lab.intrinsic_memory_execution_v12 import (
    ENVIRONMENT_CLOSURE_FORMAT, REUSE_POLICY, SPLIT_ROLES, validate_bindings,
    validate_predecessor, validate_reused_data, validate_source_feature_binding,
    verify_environment_closure)
from replica_lab.intrinsic_memory_execution_v5 import (
    binding, bound_json, hash_file, local_closure)
from replica_lab.intrinsic_memory_worlds_v11 import public_split_metadata

REPORT = 'reports/intrinsic_memory_native_v12'
DATA = 'data/intrinsic_memory_coverage_v12_20261029'
CONFIG = 'configs/spark_intrinsic_coverage_v12.json'
PREVIOUS_CONFIG = 'configs/spark_intrinsic_coverage_v11.json'
CLOSURE = f'{REPORT}/environment_closure_v12.json'
EXPOSURE = f'{REPORT}/development_exposure_v12.json'
ATTEMPT_MARKER = 'runs/intrinsic_memory_native_v12/attempts.jsonl'
STANDING_AUTHORIZATION = ('Continued as planned, no need more approval for testing, '
                          'follow the goal, assume I authorize the tests.')
HISTORICAL_V11_EXPOSURE = 'reports/intrinsic_memory_coverage_v11/development_exposure.json'
OUTPUT_ROOTS = {arm: f'{REPORT}/native_bf16_{arm}_v12_attempt10' for arm in ARMS}


def save_new(root, relative, value):
    """Create durably: temp file in the same directory, fsync, then rename.

    A crash between encode and rename leaves either no file or the complete
    file, never a truncated packet that later reads as a valid receipt.
    """
    path = root/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, sort_keys=True) + '\n'
    handle, temporary = tempfile.mkstemp(dir=str(path.parent),
                                         prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(handle, 'w') as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)          # fails if the target already exists
    finally:
        os.unlink(temporary)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def refuse_existing(root):
    # Existing attempt markers are immutable evidence, not a reason to block a
    # fresh packet with a new output identity. Launch still refuses duplicates.
    for relative in (CONFIG, DATA, CLOSURE, EXPOSURE,
                     f'{REPORT}/preparation.json', f'{REPORT}/implementation_v12.json',
                     f'{REPORT}/native_packet_two_v12.json',
                     f'{REPORT}/authorization_two_v12.json', *OUTPUT_ROOTS.values()):
        if (root/relative).exists():
            raise ValueError('v12 preparation already exists or an attempt was consumed: '
                             + relative)


def build_environment_closure(root, environment):
    """Actual hashes/origins of the executable dependency closure."""
    prefix = Path(environment['environment_directory'])
    if not prefix.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('environment prefix escapes the repository root')
    version = '.'.join(environment['python'].split('.')[:2])
    site_rel = f'lib/python{version}/site-packages'
    site = prefix/site_rel
    interpreter_rel = f'bin/python{version}'
    interpreter_link = prefix/interpreter_rel
    if not interpreter_link.exists():
        raise ValueError('qualified interpreter binary is missing')
    # The venv entry point is normally a symlink into the managed interpreter;
    # bind the resolved executable by content and keep the link path as provenance.
    interpreter_target = interpreter_link.resolve()
    if not interpreter_target.is_file():
        raise ValueError('qualified interpreter target is missing')
    module_origins = {}
    for name, relative in sorted(environment['module_origins'].items()):
        path = prefix/relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('module origin is missing: '+name)
        module_origins[name] = {'path_relative': relative, 'sha256': hash_file(path),
                                'bytes': path.stat().st_size}
    native_extensions = {}
    for name, pattern in (('torch._C', 'torch/_C*.so'),
                          ('safetensors', 'safetensors/_safetensors_rust*.so'),
                          ('tokenizers', 'tokenizers/tokenizers*.so')):
        matches = sorted(site.glob(pattern))
        if len(matches) != 1:
            raise ValueError('native extension missing or ambiguous: '+name)
        relative = matches[0].relative_to(prefix).as_posix()
        native_extensions[name] = {'path_relative': relative,
                                   'sha256': hash_file(matches[0]),
                                   'bytes': matches[0].stat().st_size}
    library_root = site/'torch'/'lib'
    shared_libraries = {}
    for path in sorted(library_root.rglob('*')):
        if path.is_file():
            relative = path.relative_to(prefix).as_posix()
            shared_libraries[relative] = {'sha256': hash_file(path),
                                          'bytes': path.stat().st_size}
    wheel_records = {}
    for path in sorted(site.glob('*.dist-info/RECORD')):
        relative = path.relative_to(prefix).as_posix()
        wheel_records[relative] = {'sha256': hash_file(path), 'bytes': path.stat().st_size}
    critical_files = dict(environment['critical_files'])
    closure = {
        'format': ENVIRONMENT_CLOSURE_FORMAT, 'prefix': str(prefix.resolve()),
        'interpreter': {'path': str(interpreter_link.resolve()),
                        'path_relative': interpreter_rel,
                        'resolved_path': str(interpreter_target),
                        'sha256': hash_file(interpreter_target),
                        'bytes': interpreter_target.stat().st_size},
        'module_origins': module_origins, 'native_extensions': native_extensions,
        'shared_library_root': site_rel+'/torch/lib', 'shared_libraries': shared_libraries,
        'site_packages': site_rel, 'wheel_records': wheel_records,
        'critical_files': critical_files,
        'coverage': ('interpreter binary, declared module origins, loaded native '
                     'extensions, every file in torch/lib, every wheel RECORD and the '
                     'sealed receipt critical files')}
    files = (1 + len(module_origins) + len(native_extensions) + len(shared_libraries)
             + len(wheel_records) + len(critical_files))
    total = (closure['interpreter']['bytes']
             + sum(item['bytes'] for item in module_origins.values())
             + sum(item['bytes'] for item in native_extensions.values())
             + sum(item['bytes'] for item in shared_libraries.values())
             + sum(item['bytes'] for item in wheel_records.values())
             + sum((prefix/relative).stat().st_size for relative in critical_files))
    if files != ENVIRONMENT_CLOSURE_FILES or total != ENVIRONMENT_CLOSURE_BYTES:
        raise ValueError(f'environment closure drifted: {files} files / {total} bytes')
    return closure


def exposure_contract(development_pin, exposure_pin):
    return {
        'calibration': {
            'status': 'already_exposed_in_v11',
            'labels': 'already_exposed_in_v11',
            'role': SPLIT_ROLES['calibration'],
            'reuse': 'matched_comparison_only'},
        'fit': {
            'status': 'already_exposed_in_v11_no_tuning_escape',
            'labels': 'already_exposed_in_v11',
            'role': SPLIT_ROLES['fit'],
            'reuse': 'matched_comparison_only',
            'group_selection': 'fixed_predeclared_groups_only'},
        'development': {
            'status': 'sealed_until_final_fit_gate',
            'labels': 'unopened_at_preparation',
            'role': SPLIT_ROLES['development'],
            'fresh_split': True,
            'retired_v11_development': True,
            'exposure_receipt': exposure_pin,
            'development_file_sha256': development_pin['sha256'],
            'preparation_generated_labels_from_oracle': True,
            'preparation_parsed_development_labels': False},
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
    historical_exposure_pin = binding(root, HISTORICAL_V11_EXPOSURE)
    historical = bound_json(root, historical_exposure_pin)
    if historical.get('development_opened') is not False:
        raise ValueError('historical v11 development receipt no longer says unopened')
    source = bound_json(root, previous['source'])
    tokenizer_identity = declared_tokenizer_identity(source['files'],
                                                     vocab_size=SOURCE_VOCAB_SIZE)

    # A fresh v12 development split, generated before any observation.
    development_rows = v12_development_worlds()
    validate_v12_development(development_rows)
    development_document = {
        'format': 'replica.intrinsic_memory.coverage_worlds.v12',
        'split': 'development', 'seed': v12_development_public_identity()['seed'],
        'role': SPLIT_ROLES['development'], 'sealed_until_final_fit_gate': True,
        'provenance': 'deterministic_predeclared_revision_worlds_v12_fresh_split',
        'prior_development_data_used': False, 'sealed_final': False,
        'independent_seed_replication': False, 'heldout_rule_family_claim': False,
        'rows': development_rows}
    save_new(root, DATA+'/development.json', development_document)
    development_pin = binding(root, DATA+'/development.json')

    exposure_document = {
        'format': 'replica.intrinsic_memory.development_exposure.v12',
        'development_opened': False,
        'fresh_for_successor': True,
        'opened_split': development_pin,
        'opened_at': None,
        'source_attempt': None,
        'scope': ('Fresh v12 development split; this receipt describes runtime label '
                  'opening, not semantic success.'),
        'retired_for_this_successor': [previous['data']['development']],
        'historical_exposure_receipt': historical_exposure_pin,
        'exposure_incident': {
            'labels_read': True,
            'split': previous['data']['development'],
            'reader': 'delegated NativeRereview scout during reports/intrinsic_memory_review_v12/rereview',
            'review_commit': '81df39ae',
            'historical_receipt_claim': ('development_opened=false, which no longer '
                                         'describes the review team'),
            'historical_receipt_left_unedited': True,
            'consequence': ('the v11 development split is retired for this successor and '
                            'cannot support a fresh utility claim informed by that review')},
        'preparation_parsed_development_labels': False,
        'preparation_generated_labels_from_oracle': True,
        'labels_inspected_or_used_for_selection': False,
    }
    save_new(root, EXPOSURE, exposure_document)
    exposure_pin = binding(root, EXPOSURE)

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
        historical_v11_exposure=historical_exposure_pin,
        source_feature_binding={
            'tokenizer_identity': tokenizer_identity,
            'tokenizer_sha256': tokenizer_identity_digest(tokenizer_identity),
            'original_parameters_sha256': observed['main']['final_original_parameters_sha256'],
            'source_files_sha256': sha_json(source['files'])},
        exposure_contract=exposure_contract(development_pin, exposure_pin))
    protocol['data'] = {'calibration': previous['data']['calibration'],
                        'fit': previous['data']['fit'],
                        'development': development_pin}
    manifest = {'format': 'replica.intrinsic_memory.coverage_split_manifest.v12',
                'generator_policy': 'predeclared_source_order_revision_oracle_no_generated_explanations',
                'reuse_policy': REUSE_POLICY,
                'exposure_contract': deepcopy(protocol['exposure_contract']),
                'splits': {}}
    for split in ('calibration', 'fit'):
        pin = protocol['data'][split]
        if previous['data'][split] != pin:
            raise ValueError('reused v11 split identity differs: '+split)
        if binding(root, pin['path'])['sha256'] != pin['sha256']:
            raise ValueError('reused v11 split digest differs: '+split)
        manifest['splits'][split] = {
            'file': pin, 'role': SPLIT_ROLES[split],
            'public_identity': public_split_metadata(split),
            'label_exposure': 'already_exposed_in_v11'}
    manifest['splits']['development'] = {
        'file': development_pin, 'role': SPLIT_ROLES['development'],
        'public_identity': v12_development_public_identity(),
        'label_exposure': 'unopened_at_preparation',
        'generated_by': 'intrinsic_memory_coverage_v12.v12_development_worlds'}
    save_new(root, DATA+'/split_manifest.json', manifest)
    protocol['data_manifest'] = binding(root, DATA+'/split_manifest.json')
    entries = ['scripts/run_spark_intrinsic_coverage_v12.py',
               'scripts/prepare_spark_intrinsic_coverage_v12.py',
               'src/replica_lab/intrinsic_memory_native_v12.py',
               'scripts/launch_spark_intrinsic_readout_environment_v1.py']
    save_new(root, REPORT+'/implementation_v12.json', {
        'format': 'replica.intrinsic_memory.local_execution_closure.v12',
        'entries': entries, 'files': local_closure(root, entries),
        'policy': ('frozen_source_feature_mechanism_fresh_v12_development_new_ledger_'
                   'attempt_marker_and_environment_closure_no_sealed_execution_mutation')})
    protocol['implementation'] = binding(root, REPORT+'/implementation_v12.json')
    environment = bound_json(root, protocol['environment'])
    save_new(root, CLOSURE, build_environment_closure(root, environment))
    protocol['environment_closure'] = binding(root, CLOSURE)
    protocol['experiment_variant'] = {
        'id': 'frozen_source_features_contextual_compression_v12',
        'predecessor': binding(root, PREVIOUS_CONFIG),
        'reason': ('v11 learns its fitting memories but does not transfer useful recall; test a '
                   'different source representation (frozen canonical-source hidden states with '
                   'a live trainable compressor) under unchanged reader, exposure and gates'),
        'changed': list(DIFFERENCES_FROM_V11),
        'unchanged': list(UNCHANGED_FROM_V11),
        'historical_results_rewritten': False,
        'prior_development_reused': False,
        'prior_development_retired': previous['data']['development'],
        'prior_development_reuse_basis': historical_exposure_pin,
        'fresh_development_split': development_pin,
        'isolated_causal_attribution': False,
        'scope_limit': ('one mechanism substitution at matched exposure with a fresh '
                        'development split; no claim that the source representation alone '
                        'explains any change'),
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
    closure_document = bound_json(root, protocol['environment_closure'])
    closure_receipt = verify_environment_closure(
        closure_document, prefix=Path(environment['environment_directory']),
        interpreter=Path(closure_document['interpreter']['path']))
    if (closure_receipt['files_verified'] != ENVIRONMENT_CLOSURE_FILES
            or closure_receipt['bytes_verified'] != ENVIRONMENT_CLOSURE_BYTES):
        raise ValueError('verified environment closure differs from the declared envelope')
    packet_name = f'{REPORT}/native_packet_two_v12.json'
    authorization = f'{REPORT}/authorization_two_v12.json'
    save_new(root, packet_name, {
        'format': PACKET, 'experiment_version': 12, 'native_engine_version': 12,
        'protocol': binding(root, CONFIG), 'arm': 'two', 'attempts': 1, 'retries': 0,
        'output_root': OUTPUT_ROOTS['two'], 'attempt_marker': ATTEMPT_MARKER,
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
        'development_labels_parsed_by_preparation': False,
        'development_labels_generated_from_oracle': True,
        'history_written': False, 'commits_made': False,
        'protocol': binding(root, CONFIG), 'packet': binding(root, packet_name),
        'data_manifest': binding(root, DATA+'/split_manifest.json'),
        'fresh_development': development_pin,
        'development_exposure_receipt': exposure_pin,
        'retired_v11_development': previous['data']['development'],
        'historical_v11_exposure_untouched': historical_exposure_pin,
        'environment_closure': binding(root, CLOSURE),
        'environment_closure_receipt': closure_receipt,
        'attempt_marker': ATTEMPT_MARKER,
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
                      'authorization receipt, then exactly one guarded launch bound to '
                      'the durable attempt marker'),
    })
    return {'prepared': True, 'native_run': False, 'native_engine_version': 12,
            'authorization_created': False,
            'fresh_development': development_pin['path'],
            'environment_closure_files': closure_receipt['files_verified'],
            'environment_closure_bytes': closure_receipt['bytes_verified'],
            'two_site_maximums': protocol['arms']['two']['total_reservations']}


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
