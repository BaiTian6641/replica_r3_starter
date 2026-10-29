"""Strict v9 admission; shared file/source checks retain the v5 implementation."""
import json
from pathlib import Path
import time

from .execution_manifest import build_execution_manifest
from .intrinsic_memory_budget_v5 import sha_json
from .intrinsic_memory_execution_v5 import (ARMS, contained, hash_file, binding, bound_json, local_closure)
from .intrinsic_memory_smaller_step_v9 import FORMAT, PACKET, AUTHORIZATION, validate_protocol


def validate_predecessor(root, protocol):
    """Bind the isolated rate comparison and preserve unopened labels."""
    previous_pin = protocol['experiment_variant']['predecessor']
    previous = bound_json(root, previous_pin)
    prior = bound_json(root, protocol['diagnostic_predecessor'])
    previous_output = previous['arms']['two']['output_root']
    if (protocol['diagnostic_predecessor']['path'] != previous_output + '/result.json'
            or prior.get('protocol_sha256') != previous_pin['sha256']
            or prior.get('source_kind') != 'official_bf16' or prior.get('arm') != 'two'
            or prior.get('native_competence') is not True
            or prior.get('main', {}).get('fit_memory_dependence_passed') is not False
            or prior.get('main', {}).get('fit_updates') != 128
            or prior.get('main', {}).get('original_weights_unchanged') is not True
            or prior.get('restart', {}).get('fresh_process_logits_exact') is not True
            or prior.get('restart', {}).get('actual_continuation_updates') != 1
            or prior.get('useful_recall') is not False
            or (contained(root, previous_output) / 'development.json').exists()):
        raise ValueError('failed-fit predecessor or unopened development evidence differs')
    excluded = {'format', 'training', 'implementation', 'experiment_variant',
                'diagnostic_predecessor', 'arms'}
    if ({k: v for k, v in protocol.items() if k not in excluded}
            != {k: v for k, v in previous.items() if k not in excluded}
            or protocol['training'] != {**previous['training'], 'lr': 0.0001}):
        raise ValueError('smaller-step comparison changed another contract')
    for arm, spec in protocol['arms'].items():
        if ({k: v for k, v in spec.items() if k != 'output_root'}
                != {k: v for k, v in previous['arms'][arm].items() if k != 'output_root'}):
            raise ValueError('matched arm budget or placement differs')


def validate_bindings(root, packet_path, *, check_source=True):
    root=Path(root);packet_path=Path(packet_path)
    packet=json.loads(packet_path.read_text());protocol=bound_json(root,packet['protocol'])
    if packet.get('format')!=PACKET or packet.get('arm') not in ARMS or packet.get('attempts')!=1 or packet.get('retries')!=0:
        raise ValueError('invalid single-attempt packet')
    validate_protocol(protocol)
    validate_predecessor(root, protocol)
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
        'native_competence':False,'useful_memory':False}


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
                or not prerequisite.get('main',{}).get('fit_memory_dependence_passed')
                or not prerequisite.get('main',{}).get('original_weights_unchanged')
                or prerequisite.get('capacity',{}).get('stop_reason') is not None
                or not prerequisite.get('restart',{}).get('fresh_process_logits_exact')
                or prerequisite.get('restart',{}).get('actual_continuation_updates')!=1):
            raise ValueError('unqualified two-site predecessor; density does not repair a failed gate')
    if not allow_existing and contained(root,packet['output_root']).exists():
        raise ValueError('attempt already consumed; zero retries')
    return receipt
