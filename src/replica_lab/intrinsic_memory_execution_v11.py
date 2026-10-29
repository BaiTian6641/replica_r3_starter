"""Strict v11 admission: completed predecessor, fresh splits and complete coverage."""
import json
from pathlib import Path
import time

from .execution_manifest import build_execution_manifest
from .intrinsic_memory_budget_v5 import sha_json
from .intrinsic_memory_execution_v5 import (ARMS, contained, hash_file, binding, bound_json, local_closure)
from .intrinsic_memory_coverage_v11 import FORMAT, PACKET, AUTHORIZATION, validate_protocol
from .intrinsic_memory_alignment_v10 import validate_protocol as validate_previous
from .intrinsic_memory_worlds_v11 import public_split_metadata


def validate_predecessor(root, protocol):
    """The completed v10 failure is a reference, never a reusable output."""
    previous_pin = protocol['experiment_variant']['predecessor']
    previous = bound_json(root, previous_pin)
    validate_previous(previous)
    prior = bound_json(root, protocol['diagnostic_predecessor'])
    previous_output = previous['arms']['two']['output_root']
    if (protocol['diagnostic_predecessor']['path'] != previous_output + '/result.json'
            or prior.get('protocol_sha256') != previous_pin['sha256']
            or prior.get('source_kind') != 'official_bf16' or prior.get('arm') != 'two'
            or prior.get('native_competence') is not True
            or prior.get('main', {}).get('teacher_competence_passed') is not True
            or prior.get('main', {}).get('fit_memory_dependence_passed') is not True
            or prior.get('main', {}).get('fit_updates') != 128
            or prior.get('main', {}).get('original_weights_unchanged') is not True
            or prior.get('restart', {}).get('fresh_process_logits_exact') is not True
            or prior.get('restart', {}).get('actual_continuation_updates') != 1
            or prior.get('useful_recall') is not False):
        raise ValueError('completed teacher reference qualification differs')
    excluded = {'format', 'teacher', 'training', 'implementation', 'experiment_variant',
                'diagnostic_predecessor', 'arms', 'data', 'data_manifest', 'prior_development_exposure',
                'coverage', 'fit_probe_prefixes', 'fit_curve_selection', 'budget_derivation', 'density_controls'}
    if ({k: v for k, v in protocol.items() if k not in excluded}
            != {k: v for k, v in previous.items() if k not in excluded}):
        raise ValueError('coverage comparison changed another contract')
    expected_density = {**previous['density_controls'], 'initial_study': 'feasibility_two_predeclared_rules_one_seed'}
    if protocol['density_controls'] != expected_density:
        raise ValueError('density control contract differs')
    for arm, spec in protocol['arms'].items():
        if ({k: v for k, v in spec.items() if k not in ('output_root', 'budget', 'total_reservations')}
                != {k: v for k, v in previous['arms'][arm].items() if k not in ('output_root', 'budget', 'total_reservations')}):
            raise ValueError('matched source, placement or physical resource cap differs')


def validate_fresh_data(root, protocol):
    """Read public split identities and prior exposure only; do not parse labels."""
    manifest = bound_json(root, protocol['data_manifest'])
    exposure = bound_json(root, protocol['prior_development_exposure'])
    previous = bound_json(root, protocol['experiment_variant']['predecessor'])
    previous_output = previous['arms']['two']['output_root']
    if (exposure.get('fresh_for_successor') is not False
            or exposure.get('opened_split') != previous['data']['development']
            or exposure.get('first_measured_opening_in_this_lineage', {}).get('path') != previous_output + '/development.json'
            or exposure.get('source_attempt') != protocol['diagnostic_predecessor']):
        raise ValueError('prior exposed development identity differs')
    for name in ('first_measured_opening_in_this_lineage', 'fitting_frozen_before_open', 'fixed_final_fit_gate'):
        bound_json(root, exposure[name])  # already exposed historical evidence
    if (manifest.get('format') != 'replica.intrinsic_memory.coverage_split_manifest.v11'
            or set(manifest.get('splits', {})) != {'calibration', 'fit', 'development'}
            or manifest.get('generator_policy') != 'predeclared_source_order_revision_oracle_no_generated_explanations'):
        raise ValueError('prepared split manifest differs')
    identities = []
    forbidden = {pin['sha256'] for pin in previous['data'].values()}
    forbidden.add(exposure['opened_split']['sha256'])
    for split, pin in protocol['data'].items():
        item = manifest['splits'][split]
        if item.get('file') != pin or item.get('public_identity') != public_split_metadata(split):
            raise ValueError('split role, roots or prepared digest differs')
        if pin['sha256'] in forbidden:
            raise ValueError('historical or exposed split cannot be fresh successor data')
        identities.append(item['public_identity'])
    if len({p['sha256'] for p in protocol['data'].values()}) != 3:
        raise ValueError('split byte identities overlap')
    for key in ('entities', 'decision_roots'):
        flattened = [value for identity in identities for value in identity[key]]
        if len(flattened) != len(set(flattened)):
            raise ValueError('cross-split decision/source identity overlap')
    return {'fresh_split_digests': True, 'public_split_identities_checked': True,
            'development_labels_parsed': False, 'prior_exposure_bound': True}


def validate_bindings(root, packet_path, *, check_source=True):
    root=Path(root);packet_path=Path(packet_path)
    packet=json.loads(packet_path.read_text());protocol=bound_json(root,packet['protocol'])
    if packet.get('format')!=PACKET or packet.get('arm') not in ARMS or packet.get('attempts')!=1 or packet.get('retries')!=0 or packet.get('experiment_version')!=11 or packet.get('native_engine_version')!=11:
        raise ValueError('invalid single-attempt packet')
    validate_protocol(protocol)
    validate_predecessor(root, protocol)
    data_receipt = validate_fresh_data(root, protocol)
    if packet['output_root']!=protocol['arms'][packet['arm']]['output_root']:
        raise ValueError('output/arm binding differs')
    contained(root,packet['output_root'])
    implementation=bound_json(root,protocol['implementation'])
    if local_closure(root,implementation['entries'])!=implementation['files']:
        raise ValueError('executable implementation closure differs')
    environment=bound_json(root,protocol['environment'])
    source=bound_json(root,protocol['source'])
    source_root=contained(root,source['source_root'])
    if (source.get('source_identity')!='official_bf16' or source.get('repo_id')!=protocol['source_repo']
            or source.get('revision')!=protocol['source_revision'] or source.get('derivative') is not False):
        raise ValueError('official BF16 source lineage differs')
    source_execution=bound_json(root,protocol['source_execution'])
    if build_execution_manifest(source_root,artifact_sha256=sha_json(source['files']))!=source_execution:
        raise ValueError('actual model/configuration/local execution code differs')
    start=time.monotonic();bytes_checked=0
    if check_source:
        for item in source['files']:
            path=contained(root,item['path'])
            if path.stat().st_size!=item['bytes'] or hash_file(path)!=item['sha256']:
                raise ValueError('source file content differs: '+item['path'])
            bytes_checked+=item['bytes']
    for pin in protocol['data'].values():
        # Hashing sealed/fresh labels is allowed here; parsing happens only after
        # the declared gate. No historical attempt03 worlds are consulted.
        path=contained(root,pin['path'])
        if hash_file(path)!=pin['sha256']:raise ValueError('prepared data differs')
    return packet,protocol,{'packet_sha256':hash_file(packet_path),'source_bytes_hashed':bytes_checked,
        'source_hash_wall_seconds':time.monotonic()-start,'source_qualified_for_loading':check_source,
        'code_before_model_loading':True,'implementation_files':len(implementation['files']),
        'source_execution_files':len(source_execution['files']),'environment':environment,
        'native_competence':False,'useful_memory':False,'fresh_data':data_receipt}


def validate_authorization(root, packet_path, receipt_path, *, allow_existing=False):
    packet=json.loads(Path(packet_path).read_text());protocol=bound_json(root,packet['protocol'])
    receipt=json.loads(Path(receipt_path).read_text());arm=protocol['arms'][packet['arm']]
    expected={'format':AUTHORIZATION,'decision':'authorized','packet_sha256':hash_file(Path(packet_path)),
              'protocol_sha256':packet['protocol']['sha256'],'arm':packet['arm'],
              'source_kind':'official_bf16','output_root':packet['output_root'],
              'attempts':1,'retries':0,'budget':arm['total_reservations'],
              'implementation_sha256':protocol['implementation']['sha256'],
              'environment_sha256':protocol['environment']['sha256']}
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('missing or different exact external authorization')
    if not receipt.get('human_approval_reference') or not receipt.get('human_approval_text'):
        raise ValueError('direct human approval evidence required')
    if packet['arm']!='two':
        pin=receipt.get('qualified_two_site_receipt')
        if not pin:raise ValueError('density execution requires useful two-site readout evidence')
        prerequisite=bound_json(root,pin)
        if (pin['path']!=protocol['arms']['two']['output_root']+'/result.json'
                or prerequisite.get('protocol_sha256')!=packet['protocol']['sha256']
                or prerequisite.get('arm')!='two' or prerequisite.get('source_kind')!='official_bf16'
                or prerequisite.get('native_competence') is not True
                or prerequisite.get('useful_recall') is not True
                or not prerequisite.get('main',{}).get('teacher_competence_passed')
                or not prerequisite.get('main',{}).get('fit_memory_dependence_passed')
                or not prerequisite.get('main',{}).get('original_weights_unchanged')
                or prerequisite.get('capacity',{}).get('stop_reason') is not None
                or not prerequisite.get('restart',{}).get('fresh_process_logits_exact')
                or prerequisite.get('restart',{}).get('actual_continuation_updates')!=1):
            raise ValueError('unqualified two-site predecessor; density does not repair a failed gate')
    if not allow_existing and contained(root,packet['output_root']).exists():
        raise ValueError('attempt already consumed; zero retries')
    return receipt
