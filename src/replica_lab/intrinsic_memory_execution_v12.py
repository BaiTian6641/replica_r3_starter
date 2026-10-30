"""Strict v12 admission: matched v11 comparison, lawful data reuse, exact bindings.

Only the standard library is imported before source admission (plus the local
admission helpers, which are themselves standard-library-only). Nothing here
loads weights, parses development labels or grants authorization.

Three independent things are checked before a model may be loaded:

1. the *comparison* is the completed v11 experiment, whose development split
   was never opened (verified from its exposure receipt, never by reading
   labels), and whose every shared contract is still byte-identical;
2. the *prepared data* is the same calibration/fitting/development corpus,
   reached only through a new v12 manifest with an explicit exposure contract;
3. the *code and source* are the exact bound bytes, including the declared
   source-feature binding (tokenizer identity, source file digest, original
   parameter digest) and the complete local import closure.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

from .execution_manifest import build_execution_manifest
from .intrinsic_memory_budget_v12 import sha_json, validate_plan
from .intrinsic_memory_coverage_v11 import validate_protocol as validate_previous_protocol
from .intrinsic_memory_coverage_v12 import (
    ARMS, AUTHORIZATION, DIFFERENCES_FROM_V11, GATED_ARMS, OUTER_WALL_SECONDS,
    PACKET, SOURCE_FEATURE_BINDING_KEYS, TEACHER, UNCHANGED_FROM_V11,
    declared_tokenizer_identity, plan, tokenizer_identity_digest, totals,
    validate_protocol)
from .intrinsic_memory_execution_v5 import (
    contained, hash_file, bound_json, local_closure)

MANIFEST_FORMAT = 'replica.intrinsic_memory.coverage_split_manifest.v12'
EXPOSURE_FORMAT = 'replica.intrinsic_memory.development_exposure.v12'
# Fields that must remain identical to the completed v11 comparison. A silent
# change to any of them would make the two-site comparison uninterpretable.
SHARED_FIELDS = (
    'source_kind', 'source_repo', 'source_revision', 'base_dtype', 'extension_dtype',
    'optimizer', 'placement', 'slots', 'slot_policy', 'persistent_P_updates',
    'automatic_promotion', 'competence', 'fit_gate', 'utility', 'training', 'coverage',
    'generation', 'persistent_capacity', 'numerical_restart_on_failed_fit',
    'stages_closed_after_failed_fit', 'density_controls', 'fit_probe_prefixes',
    'fit_curve_selection',
)
ARMS_SHARED = ('sites', 'wall_seconds', 'resource_limits')


def validate_predecessor(root, protocol):
    """The consumed v11 attempt is a comparison reference, never a reusable output."""
    previous_pin = protocol['experiment_variant']['predecessor']
    previous = bound_json(root, previous_pin)
    validate_previous_protocol(previous)
    result_pin = protocol['diagnostic_predecessor']
    if result_pin['path'] != previous['arms']['two']['output_root']+'/result.json':
        raise ValueError('predecessor result is not the completed v11 two-site output')
    observed = bound_json(root, result_pin)
    main = observed.get('main', {})
    if (observed.get('source_kind') != 'official_bf16' or observed.get('arm') != 'two'
            or observed.get('protocol_sha256') != previous_pin['sha256']
            or observed.get('native_competence') is not True
            or observed.get('useful_recall') is not False
            or main.get('teacher_competence_passed') is not True
            or main.get('fit_updates') != 512
            or main.get('fit_memory_dependence_passed') is not False
            or main.get('original_weights_unchanged') is not True
            or main.get('final_original_parameters_sha256')
            != protocol['source_feature_binding']['original_parameters_sha256']
            or main.get('persistent_P_updates') != 0
            or observed.get('restart', {}).get('fresh_process_logits_exact') is not True
            or observed.get('restart', {}).get('actual_continuation_updates') != 1
            or observed.get('capacity', {}).get('stop_reason') is not None):
        raise ValueError('completed v11 reference outcome differs')
    # Every shared contract must still be identical to the sealed comparison.
    for name in SHARED_FIELDS:
        if protocol.get(name) != previous.get(name):
            raise ValueError(f'shared v11 contract changed without declaration: {name}')
    for arm in ARMS:
        for name in ARMS_SHARED:
            if protocol['arms'][arm].get(name) != previous['arms'][arm].get(name):
                raise ValueError(f'shared v11 arm contract changed: {arm}/{name}')
    if protocol['arms'].keys() != previous['arms'].keys():
        raise ValueError('density arm inventory differs from v11')
    # The teacher contract may add only the declared cache-binding fields.
    added = {k: v for k, v in TEACHER.items() if k not in previous['teacher']}
    if ({k: v for k, v in protocol['teacher'].items() if k not in added}
            != previous['teacher'] or {k: protocol['teacher'].get(k) for k in added} != added):
        raise ValueError('teacher contract changed beyond the declared cache binding')
    variant = protocol.get('experiment_variant', {})
    if (variant.get('changed') != list(DIFFERENCES_FROM_V11)
            or variant.get('unchanged') != list(UNCHANGED_FROM_V11)
            or variant.get('historical_results_rewritten') is not False):
        raise ValueError('declared difference list differs from the reviewed v12 differences')
    if protocol['source']['sha256'] != previous['source']['sha256']:
        raise ValueError('source verification receipt changed')
    if protocol['source_execution']['sha256'] != previous['source_execution']['sha256']:
        raise ValueError('source execution manifest changed')
    if protocol['environment']['sha256'] != previous['environment']['sha256']:
        raise ValueError('qualified environment receipt changed')
    return {'v11_comparison': result_pin, 'shared_contracts_identical': True,
            'v11_development_opened': False}


def validate_reused_data(root, protocol):
    """Same rows, new manifest, explicit exposure contract, no label parsing."""
    previous = bound_json(root, protocol['experiment_variant']['predecessor'])
    manifest = bound_json(root, protocol['data_manifest'])
    exposure = bound_json(root, protocol['prior_development_exposure'])
    if (manifest.get('format') != MANIFEST_FORMAT
            or set(manifest.get('splits', {})) != {'calibration', 'fit', 'development'}
            or manifest.get('generator_policy')
            != 'predeclared_source_order_revision_oracle_no_generated_explanations'
            or manifest.get('reuse_policy')
            != 'matched_v11_rows_via_new_v12_manifest_development_labels_unread'
            or manifest.get('exposure_contract') != protocol['exposure_contract']):
        raise ValueError('prepared v12 split manifest differs')
    for split in ('calibration', 'fit', 'development'):
        item = manifest['splits'][split]
        pin = protocol['data'][split]
        if item.get('file') != pin or previous['data'][split] != pin:
            raise ValueError('reused split identity differs from the v11 comparison rows')
        if item.get('public_identity') != previous_data_identity(previous, split):
            raise ValueError('split role, roots or prepared digest differs')
        if item.get('label_exposure') != protocol['exposure_contract'][split]['labels']:
            raise ValueError('split exposure declaration differs')
    if (protocol['exposure_contract']['calibration'].get('status') != 'already_exposed_in_v11'
            or protocol['exposure_contract']['fit'].get('status')
            != 'already_exposed_in_v11_no_tuning_escape'
            or protocol['exposure_contract']['development'].get('status')
            != 'open_only_after_frozen_final_fit_gate'):
        raise ValueError('exposure contract status differs')
    # The v11 development exposure receipt is the sole authority for reuse. It is
    # re-read here; no development label file is parsed at admission.
    if (exposure.get('format') != 'replica.intrinsic_memory.development_exposure.v11'
            or exposure.get('development_opened') is not False
            or exposure.get('fresh_for_successor') is not True
            or exposure.get('opened_split', {}).get('sha256')
            != protocol['data']['development']['sha256']
            or exposure.get('opened_split', {}).get('path')
            != protocol['data']['development']['path']
            or exposure.get('source_attempt') != protocol['diagnostic_predecessor']):
        raise ValueError('v11 development exposure no longer permits lawful reuse')
    return {'same_rows_as_v11': True, 'new_manifest': protocol['data_manifest'],
            'development_opened': exposure['development_opened'],
            'development_labels_parsed_at_admission': False}


def previous_data_identity(previous, split):
    """Public split metadata is deterministic in the sealed v11 worlds module."""
    from .intrinsic_memory_worlds_v11 import public_split_metadata
    return public_split_metadata(split)


def validate_source_feature_binding(root, protocol):
    """The declared feature binding must be derivable from the verified source."""
    declared = protocol['source_feature_binding']
    for name in SOURCE_FEATURE_BINDING_KEYS:
        if name not in declared: raise ValueError('source-feature binding is incomplete')
    source = bound_json(root, protocol['source'])
    files = source['files']
    if declared['source_files_sha256'] != sha_json(files):
        raise ValueError('declared source file digest differs')
    expected_identity = declared_tokenizer_identity(files)
    if declared['tokenizer_identity'] != expected_identity:
        raise ValueError('declared tokenizer identity differs from the verified source')
    if declared['tokenizer_sha256'] != tokenizer_identity_digest(expected_identity):
        raise ValueError('declared tokenizer digest differs from its identity mapping')
    previous = bound_json(root, protocol['experiment_variant']['predecessor'])
    observed = bound_json(root, protocol['diagnostic_predecessor'])
    for item in (previous, observed.get('main', {})):
        recorded = item.get('source_feature_binding')
        if recorded is not None and recorded != declared:
            raise ValueError('declared source-feature binding differs from the recorded evidence')
    return declared


def validate_bindings(root, packet_path, *, check_source=True):
    root = Path(root); packet_path = Path(packet_path)
    packet = json.loads(packet_path.read_text()); protocol = bound_json(root, packet['protocol'])
    if (packet.get('format') != PACKET or packet.get('arm') not in ARMS
            or packet.get('attempts') != 1 or packet.get('retries') != 0
            or packet.get('experiment_version') != 12
            or packet.get('native_engine_version') != 12):
        raise ValueError('invalid single-attempt v12 packet')
    validate_protocol(protocol)
    predecessor = validate_predecessor(root, protocol)
    data_receipt = validate_reused_data(root, protocol)
    feature_binding = validate_source_feature_binding(root, protocol)
    if packet['output_root'] != protocol['arms'][packet['arm']]['output_root']:
        raise ValueError('output/arm binding differs')
    contained(root, packet['output_root'])
    implementation = bound_json(root, protocol['implementation'])
    if local_closure(root, implementation['entries']) != implementation['files']:
        raise ValueError('executable implementation closure differs')
    environment = bound_json(root, protocol['environment'])
    source = bound_json(root, protocol['source'])
    source_root = contained(root, source['source_root'])
    if (source.get('source_identity') != 'official_bf16'
            or source.get('repo_id') != protocol['source_repo']
            or source.get('revision') != protocol['source_revision']
            or source.get('derivative') is not False):
        raise ValueError('official BF16 source lineage differs')
    source_execution = bound_json(root, protocol['source_execution'])
    if build_execution_manifest(source_root, artifact_sha256=sha_json(source['files'])) != source_execution:
        raise ValueError('actual model/configuration/local execution code differs')
    start = time.monotonic(); bytes_checked = 0
    if check_source:
        for item in source['files']:
            path = contained(root, item['path'])
            if path.stat().st_size != item['bytes'] or hash_file(path) != item['sha256']:
                raise ValueError('source file content differs: '+item['path'])
            bytes_checked += item['bytes']
    for pin in protocol['data'].values():
        # Hashing prepared bytes is allowed here; parsing them is not. Development
        # labels are read only after the frozen final fitting prefix passes.
        path = contained(root, pin['path'])
        if hash_file(path) != pin['sha256']: raise ValueError('prepared data differs')
    return packet, protocol, {
        'packet_sha256': hash_file(packet_path), 'source_bytes_hashed': bytes_checked,
        'source_hash_wall_seconds': time.monotonic()-start,
        'source_qualified_for_loading': check_source, 'code_before_model_loading': True,
        'implementation_files': len(implementation['files']),
        'source_execution_files': len(source_execution['files']), 'environment': environment,
        'feature_binding': {'tokenizer_sha256': feature_binding['tokenizer_sha256'],
                            'original_parameters_sha256':
                                feature_binding['original_parameters_sha256'],
                            'source_files_sha256': feature_binding['source_files_sha256']},
        'comparison': predecessor, 'reused_data': data_receipt,
        'native_competence': False, 'useful_memory': False}


def validate_authorization(root, packet_path, receipt_path, *, allow_existing=False):
    packet = json.loads(Path(packet_path).read_text())
    protocol = bound_json(root, packet['protocol']); arm = protocol['arms'][packet['arm']]
    receipt = json.loads(Path(receipt_path).read_text())
    expected = {'format': AUTHORIZATION, 'decision': 'authorized',
                'packet_sha256': hash_file(Path(packet_path)),
                'protocol_sha256': packet['protocol']['sha256'], 'arm': packet['arm'],
                'source_kind': 'official_bf16', 'output_root': packet['output_root'],
                'attempts': 1, 'retries': 0, 'budget': arm['total_reservations'],
                'implementation_sha256': protocol['implementation']['sha256'],
                'environment_sha256': protocol['environment']['sha256'],
                'feature_binding_sha256': sha_json(protocol['source_feature_binding']),
                'slot_policy_v12': protocol['slot_policy_v12'],
                'outer_wall_seconds': arm['wall_seconds']}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('missing or different exact external authorization')
    if not receipt.get('human_approval_reference') or not receipt.get('human_approval_text'):
        raise ValueError('direct human approval evidence required')
    if receipt.get('human_approval_text') != protocol.get('standing_authorization_text'):
        raise ValueError('authorization text differs from the recorded standing instruction')
    if packet['arm'] in GATED_ARMS:
        pin = receipt.get('qualified_two_site_receipt')
        if not pin: raise ValueError('density execution requires useful two-site readout evidence')
        prerequisite = bound_json(root, pin)
        if (pin['path'] != protocol['arms']['two']['output_root']+'/result.json'
                or prerequisite.get('protocol_sha256') != packet['protocol']['sha256']
                or prerequisite.get('arm') != 'two'
                or prerequisite.get('source_kind') != 'official_bf16'
                or prerequisite.get('native_competence') is not True
                or prerequisite.get('useful_recall') is not True
                or not prerequisite.get('main', {}).get('teacher_competence_passed')
                or not prerequisite.get('main', {}).get('fit_memory_dependence_passed')
                or not prerequisite.get('main', {}).get('original_weights_unchanged')
                or prerequisite.get('capacity', {}).get('stop_reason') is not None
                or not prerequisite.get('restart', {}).get('fresh_process_logits_exact')
                or prerequisite.get('restart', {}).get('actual_continuation_updates') != 1):
            raise ValueError('unqualified two-site predecessor; density does not repair a failed gate')
    if not allow_existing and contained(root, packet['output_root']).exists():
        raise ValueError('attempt already consumed; zero retries')
    return receipt


def validate_stage_plan(protocol, arm):
    """Re-check one arm's envelope without trusting the serialized totals."""
    validate_plan(protocol['arms'][arm]['budget'])
    if protocol['arms'][arm]['budget'] != plan(arm):
        raise ValueError('v12 stage plan differs from its derivation')
    if totals(protocol['arms'][arm]['budget']) != protocol['arms'][arm]['total_reservations']:
        raise ValueError('v12 reservation totals differ')
    if protocol['arms'][arm]['wall_seconds'] != OUTER_WALL_SECONDS:
        raise ValueError('outer wall differs')
    return True
