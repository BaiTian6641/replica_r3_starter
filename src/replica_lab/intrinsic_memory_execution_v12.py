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
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from .execution_manifest import build_execution_manifest
from .intrinsic_memory_budget_v12 import sha_json, validate_plan
from .intrinsic_memory_coverage_v11 import validate_protocol as validate_previous_protocol
from .intrinsic_memory_coverage_v12 import (
    ARMS, AUTHORIZATION, DIFFERENCES_FROM_V11, ENVIRONMENT_CLOSURE_BYTES,
    ENVIRONMENT_CLOSURE_FILES, GATED_ARMS, OUTER_WALL_SECONDS,
    PACKET, SOURCE_FEATURE_BINDING_KEYS, SOURCE_FILE_BYTES, TEACHER, UNCHANGED_FROM_V11,
    declared_tokenizer_identity, plan, public_split_metadata,
    tokenizer_identity_digest, totals, v12_development_public_identity,
    validate_protocol)
from .intrinsic_memory_execution_v5 import (
    contained, hash_file, bound_json, local_closure)

MANIFEST_FORMAT = 'replica.intrinsic_memory.coverage_split_manifest.v12'
EXPOSURE_FORMAT = 'replica.intrinsic_memory.development_exposure.v12'
ENVIRONMENT_CLOSURE_FORMAT = 'replica.intrinsic_memory.native_environment_closure.v12'
ATTEMPT_MARKER_FORMAT = 'replica.intrinsic_memory.native_attempt_marker.v12'
REUSE_POLICY = 'exposed_v11_calibration_and_fit_matched_fresh_v12_development'
SPLIT_ROLES = {'calibration': 'reused_v11_calibration', 'fit': 'reused_v11_fit',
               'development': 'fresh_v12_development'}
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
    """Matched v11 rows for calibration/fit, a fresh role-bound development split.

    Calibration and fitting keep the exposed v11 bytes so the comparison stays
    matched. Development is new for this successor: the v11 development split is
    retired because the re-review scout read it, and that fact is recorded in the
    current exposure receipt without editing the historical one. Only hashes are
    computed here; no development label is parsed at admission.
    """
    previous = bound_json(root, protocol['experiment_variant']['predecessor'])
    manifest = bound_json(root, protocol['data_manifest'])
    exposure = bound_json(root, protocol['prior_development_exposure'])
    if (manifest.get('format') != MANIFEST_FORMAT
            or set(manifest.get('splits', {})) != {'calibration', 'fit', 'development'}
            or manifest.get('generator_policy')
            != 'predeclared_source_order_revision_oracle_no_generated_explanations'
            or manifest.get('reuse_policy') != REUSE_POLICY
            or manifest.get('exposure_contract') != protocol['exposure_contract']):
        raise ValueError('prepared v12 split manifest differs')
    for split in ('calibration', 'fit'):
        item = manifest['splits'][split]
        pin = protocol['data'][split]
        if item.get('file') != pin or previous['data'][split] != pin:
            raise ValueError('reused v11 row identity differs from the comparison rows')
        if item.get('role') != SPLIT_ROLES[split]:
            raise ValueError('split role binding differs')
        if item.get('public_identity') != public_split_metadata(split):
            raise ValueError('split roots or prepared digest differ')
        if item.get('label_exposure') != 'already_exposed_in_v11':
            raise ValueError('split exposure declaration differs')
    development = manifest['splits']['development']
    dev_pin = protocol['data']['development']
    if development.get('file') != dev_pin:
        raise ValueError('fresh development identity differs from the protocol')
    if development.get('role') != SPLIT_ROLES['development']:
        raise ValueError('development role binding differs')
    if development.get('public_identity') != v12_development_public_identity():
        raise ValueError('fresh development public identity differs')
    if development.get('label_exposure') != 'unopened_at_preparation':
        raise ValueError('fresh development exposure declaration differs')
    # Fresh means it cannot be any v11 split, including the retired development.
    if dev_pin['sha256'] in {pin['sha256'] for pin in previous['data'].values()}:
        raise ValueError('fresh v12 development reuses a v11 split')
    if dev_pin['path'] == previous['data']['development']['path']:
        raise ValueError('fresh v12 development reuses the retired path')
    calibration = manifest['splits']['calibration']['public_identity']
    fit = manifest['splits']['fit']['public_identity']
    dev_identity = development['public_identity']
    if (set(dev_identity['entities'])
            & (set(calibration['entities']) | set(fit['entities']))):
        raise ValueError('fresh development entity overlaps a reused split')
    if (set(dev_identity['decision_roots'])
            & (set(calibration['decision_roots']) | set(fit['decision_roots']))):
        raise ValueError('fresh development decision root overlaps a reused split')
    # The current v12 exposure receipt is the only reuse authority for this split.
    if (exposure.get('format') != EXPOSURE_FORMAT
            or exposure.get('development_opened') is not False
            or exposure.get('fresh_for_successor') is not True
            or exposure.get('opened_split', {}).get('sha256') != dev_pin['sha256']
            or exposure.get('opened_split', {}).get('path') != dev_pin['path']
            or exposure.get('retired_for_this_successor') != [previous['data']['development']]
            or exposure.get('historical_exposure_receipt')
            != protocol.get('historical_v11_exposure')
            or exposure.get('source_attempt') is not None
            or exposure.get('exposure_incident', {}).get('labels_read') is not True
            or exposure.get('preparation_parsed_development_labels') is not False):
        raise ValueError('current v12 development exposure receipt differs')
    return {'matched_calibration_and_fit': True, 'fresh_development': dev_pin,
            'new_manifest': protocol['data_manifest'],
            'development_opened': exposure['development_opened'],
            'v11_development_retired': True,
            'development_labels_parsed_at_admission': False}


def validate_source_feature_binding(root, protocol):
    """The declared feature binding must be derivable from the verified source."""
    declared = protocol['source_feature_binding']
    for name in SOURCE_FEATURE_BINDING_KEYS:
        if name not in declared: raise ValueError('source-feature binding is incomplete')
    source = bound_json(root, protocol['source'])
    files = source['files']
    # The read-byte envelope is derived from this exact size; it must be the
    # receipt's own total, not a copied constant.
    if (sum(item['bytes'] for item in files) != SOURCE_FILE_BYTES
            or source.get('total_bytes') != SOURCE_FILE_BYTES):
        raise ValueError('declared source byte size differs from the verified receipt')
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


def validate_environment_closure(root, protocol, *, check=True, interpreter=None):
    """Verify actual hashes/origins of the interpreter, native libraries and wheels.

    The sealed environment receipt binds package versions, selected module origins
    and twelve critical runtime files. This closure adds the executable dependency
    set the receipt itself declares it does not cover: the interpreter binary, the
    loaded native extensions, every file in `torch/lib`, and every wheel `RECORD`.
    Declared sets are compared exactly, so an added or removed library is a
    failure rather than an untested difference.

    ``interpreter`` is only for a preparation-time self-check running under a
    different interpreter; admission and launch always use the running one.
    """
    closure = bound_json(root, protocol['environment_closure'])
    if closure.get('format') != ENVIRONMENT_CLOSURE_FORMAT:
        raise ValueError('environment closure format differs')
    if not check:
        return closure
    environment = bound_json(root, protocol['environment'])
    prefix = Path(environment['environment_directory'])
    if not prefix.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('environment prefix escapes the repository root')
    receipt = verify_environment_closure(closure, prefix=prefix, interpreter=interpreter)
    if (receipt['bytes_verified'] != ENVIRONMENT_CLOSURE_BYTES
            or receipt['files_verified'] != ENVIRONMENT_CLOSURE_FILES):
        raise ValueError('verified environment closure size differs from the declared envelope')
    return receipt


def verify_environment_closure(closure, *, prefix=None, interpreter=None):
    """Recompute every declared hash and refuse missing, added or substituted files."""
    if closure.get('format') != ENVIRONMENT_CLOSURE_FORMAT:
        raise ValueError('environment closure format differs')
    declared_prefix = Path(closure['prefix']).resolve()
    if prefix is not None and Path(prefix).resolve() != declared_prefix:
        raise ValueError('environment prefix differs from the qualified environment')
    prefix = declared_prefix
    interpreter = Path(sys.executable) if interpreter is None else Path(interpreter)
    declared_interpreter = closure['interpreter']
    if Path(declared_interpreter['path']).resolve() != interpreter.resolve():
        raise ValueError('running interpreter differs from the closure')
    target = Path(declared_interpreter['resolved_path'])
    if target.is_symlink() or not target.is_file():
        raise ValueError('closure interpreter target missing or substituted')
    if (target.stat().st_size != declared_interpreter['bytes']
            or hash_file(target) != declared_interpreter['sha256']):
        raise ValueError('closure interpreter content differs')
    files, total = 1, target.stat().st_size

    def check_file(relative, expected_sha, expected_bytes):
        nonlocal files, total
        path = prefix/relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('closure file missing or substituted: '+relative)
        size = path.stat().st_size
        if size != expected_bytes or hash_file(path) != expected_sha:
            raise ValueError('closure file content differs: '+relative)
        files += 1; total += size
    for name, item in closure['module_origins'].items():
        check_file(item['path_relative'], item['sha256'], item['bytes'])
    for name, item in closure['native_extensions'].items():
        check_file(item['path_relative'], item['sha256'], item['bytes'])
    library_root = prefix/closure['shared_library_root']
    present = {p.relative_to(prefix).as_posix() for p in library_root.rglob('*') if p.is_file()}
    declared = set(closure['shared_libraries'])
    if present != declared:
        raise ValueError('shared library set differs: added or removed files')
    for relative, item in closure['shared_libraries'].items():
        check_file(relative, item['sha256'], item['bytes'])
    site = prefix/closure['site_packages']
    present_records = {p.relative_to(prefix).as_posix() for p in site.glob('*.dist-info/RECORD')}
    if present_records != set(closure['wheel_records']):
        raise ValueError('wheel RECORD set differs: added or removed distributions')
    for relative, item in closure['wheel_records'].items():
        check_file(relative, item['sha256'], item['bytes'])
    for relative, expected in closure['critical_files'].items():
        path = prefix/relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('closure file missing or substituted: '+relative)
        if hash_file(path) != expected:
            raise ValueError('closure file content differs: '+relative)
        files += 1; total += path.stat().st_size
    return {'files_verified': files, 'bytes_verified': total,
            'interpreter': str(interpreter), 'prefix': str(prefix),
            'closure_sha256': sha_json(closure),
            'coverage': ('interpreter, native extensions, torch/lib shared libraries, '
                         'wheel RECORD files and the sealed receipt critical files')}


def read_attempt_marker(path):
    """Append-only external attempt registry; a torn line blocks the attempt."""
    path = Path(path)
    if not path.exists(): return []
    if path.is_symlink() or not path.is_file():
        raise ValueError('substituted attempt marker')
    payload = path.read_bytes()
    if payload and not payload.endswith(b'\n'):
        raise ValueError('torn attempt marker: operator review required')
    records = []
    for index, line in enumerate(payload.decode('utf-8').splitlines()):
        if not line.strip(): continue
        try: record = json.loads(line)
        except ValueError as error:
            raise ValueError(f'corrupt attempt marker line {index}') from error
        if record.get('format') != ATTEMPT_MARKER_FORMAT:
            raise ValueError('attempt marker format differs')
        records.append(record)
    return records


def record_attempt(path, *, packet_sha256, protocol_sha256, arm, output_root,
                   authorization_sha256):
    """Consume exactly one attempt, durably, outside the deletable output tree."""
    path = Path(path)
    if path.is_symlink(): raise ValueError('substituted attempt marker')
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix('.lock')
    record = {'format': ATTEMPT_MARKER_FORMAT, 'packet_sha256': packet_sha256,
              'protocol_sha256': protocol_sha256, 'arm': arm, 'attempt': 1, 'retries': 0,
              'output_root': output_root, 'authorization_sha256': authorization_sha256,
              'started_unix': time.time()}
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            existing = read_attempt_marker(path)
            if any(item.get('packet_sha256') == packet_sha256 for item in existing):
                raise ValueError('attempt already consumed for this packet')
            if any(item.get('output_root') == output_root for item in existing):
                raise ValueError('attempt already consumed for this output root')
            line = (json.dumps(record, sort_keys=True, separators=(',', ':'))+'\n').encode()
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            try:
                written = 0
                while written < len(line):
                    written += os.write(fd, line[written:])
                os.fsync(fd)
            finally:
                os.close(fd)
            directory = os.open(path.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    return record


def attempt_consumed(path, *, packet_sha256=None, output_root=None):
    """True when the exact packet or output root already has a durable marker."""
    records = read_attempt_marker(path)
    if packet_sha256 is not None and any(r['packet_sha256'] == packet_sha256 for r in records):
        return True
    if output_root is not None and any(r['output_root'] == output_root for r in records):
        return True
    return False


def validate_bindings(root, packet_path, *, check_source=True, closure_interpreter=None):
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
    environment_closure = validate_environment_closure(root, protocol,
                                                       interpreter=closure_interpreter)
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
        'environment_closure': environment_closure,
        'attempt_marker': packet['attempt_marker'],
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
                'environment_closure_sha256': protocol['environment_closure']['sha256'],
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
    marker = contained(root, packet['attempt_marker'])
    output_root = contained(root, packet['output_root'])
    if marker.resolve().is_relative_to(output_root.resolve()):
        raise ValueError('attempt marker must live outside the deletable output tree')
    packet_sha256 = hash_file(Path(packet_path))
    if allow_existing:
        # A supervised worker may only run for an attempt that was durably marked.
        if not attempt_consumed(marker, packet_sha256=packet_sha256):
            raise ValueError('no durable attempt marker for this supervised run')
    else:
        if attempt_consumed(marker, packet_sha256=packet_sha256,
                            output_root=packet['output_root']):
            raise ValueError('attempt already consumed; zero retries')
        if output_root.exists():
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
