"""Standard-library execution/source admission for official BF16 v5 packets."""
import ast
import hashlib
import json
from pathlib import Path
import time

from .execution_manifest import build_execution_manifest
from .intrinsic_memory_budget_v5 import validate_plan, sha_json

FORMAT = 'replica.intrinsic_memory.experiment.v5'
PACKET = 'replica.intrinsic_memory.launch_packet.v5'
AUTHORIZATION = 'replica.intrinsic_memory.external_authorization.v5'
PROTECTED = 'src/replica_lab/runtime_pilot.py'
ARMS = {'two': [3,7], 'full_attention_nine': list(range(3,36,4)), 'all': list(range(36))}


def contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or relative.as_posix()==PROTECTED:
        raise ValueError('invalid or protected binding path')
    path = root/relative
    if not path.resolve().is_relative_to(root.resolve()): raise ValueError('binding escapes root')
    if any(p.is_symlink() for p in [path,*path.parents] if p!=root.parent):
        raise ValueError('substituted binding path')
    return path


def hash_file(path):
    if path.is_symlink() or not path.is_file(): raise ValueError('missing/substituted bound file')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def binding(root, relative):
    p = contained(root,relative)
    return {'path':relative,'sha256':hash_file(p)}


def bound_json(root, pin):
    p = contained(root,pin['path'])
    if hash_file(p)!=pin['sha256']:raise ValueError('bound file hash differs: '+pin['path'])
    return json.loads(p.read_text())


def local_closure(root, entries):
    """Resolve static local imports, including package initializers, without imports.

    Explicit entries also bind the environment validator and command children.
    Dynamic third-party loading is separately covered by the environment and
    source manifests. The protected module is rejected before any file read.
    """
    pending=list(entries);result={}
    while pending:
        relative=pending.pop()
        if relative in result:continue
        path=contained(root,relative)
        result[relative]=hash_file(path)
        if path.suffix!='.py':continue
        tree=ast.parse(path.read_text(),filename=relative)
        candidates=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                candidates.extend('src/'+n.name.replace('.','/') for n in node.names if n.name.startswith('replica_lab'))
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=Path(relative).parent
                    for _ in range(node.level-1):base=base.parent
                    candidates.append(str(base/(node.module or '').replace('.','/')))
                elif node.module and node.module.startswith('replica_lab'):
                    candidates.append('src/'+node.module.replace('.','/'))
                elif node.module and (root/'scripts'/(node.module+'.py')).is_file():
                    candidates.append('scripts/'+node.module)
        for value in candidates:
            for candidate in (value+'.py',value+'/__init__.py'):
                if candidate==PROTECTED:raise ValueError('protected module dependency refused')
                if (root/candidate).is_file():pending.append(candidate);break
        for parent in Path(relative).parents:
            if str(parent) in ('.','src','scripts'):break
            candidate=(parent/'__init__.py').as_posix()
            if candidate!=relative and (root/candidate).is_file():pending.append(candidate)
    return dict(sorted(result.items()))


def validate_protocol(protocol):
    if (protocol.get('format')!=FORMAT or protocol.get('source_kind')!='official_bf16'
            or protocol.get('source_repo')!='XHToken/Spark-X2.6-4B'
            or protocol.get('source_revision')!='66e3ae7bee54e0a523d692cb8ef841f35409807b'
            or protocol.get('base_dtype')!='bfloat16' or protocol.get('extension_dtype')!='float32'
            or protocol.get('optimizer')!='NativeFusedAdamW'
            or protocol.get('placement')!='block_input_query_to_post_block_residual'
            or protocol.get('slots')!=8 or protocol.get('slot_policy')!='fixed_eight_masked_padding_v5'
            or protocol.get('persistent_P_updates')!=0 or protocol.get('automatic_promotion') is not False):
        raise ValueError('v5 source/ownership/numerical contract differs')
    if protocol.get('competence')!={'cases':8,'correct_min':8,'false_recall_max':0,'truncation_max':0,'weights':'frozen_original_extensions_disabled'}:
        raise ValueError('competence contract differs')
    if protocol.get('fit_gate')!={'cases':8,'complete_correct_min':8,'first_token_correct_min':8,
            'termination_required':True,'prefix_logits_exact':True,'better_than':['wrong','zero','absent']}:
        raise ValueError('fitting gate differs')
    if protocol.get('utility')!={'cases':8,'contextual_correct_min':6,'false_recall_max':0,'exact_text_deficit_max':1,
            'better_than':['wrong','shuffled','zero','absent']}:
        raise ValueError('utility criteria differ')
    if protocol.get('training')!={'updates':32,'restart_updates':1,'lr':0.001,'seed':17,'batch_size':1,
                               'schedule':'fixed_round_robin_8_worlds_4_passes','teacher_updates':0}:
        raise ValueError('fixed training schedule differs')
    if protocol.get('generation')!={'context_tokens':256,'max_new_tokens':8,'thinking':'none','padding':'right',
             'decode':'greedy_uncached_no_sampling','stop_token_ids':[1],'eos_token_id':1,'pad_token_id':2,
             'label_alignment':'logit_i_predicts_token_i_plus_1_no_additional_shift'}:
        raise ValueError('token/position/stopping contract differs')
    if set(protocol.get('arms',{}))!=set(ARMS):raise ValueError('density arm inventory differs')
    for name,arm in protocol['arms'].items():
        if arm['sites']!=ARMS[name]:raise ValueError('density sites differ')
        validate_plan(arm['budget'])
        totals={k:sum(s[k] for s in arm['budget'].values()) for k in arm['budget']['admission']}
        if totals!=arm['total_reservations'] or totals['wall_seconds']>arm['wall_seconds']-11:
            raise ValueError('budget derivation/outer wall differs')
        if sum(s['updates'] for s in arm['budget'].values())!=33 or totals['model_loads']!=2:
            raise ValueError('updates/model loads differ')
        if arm['resource_limits']!={'driver_bytes':50*1024**3,'swap_growth_bytes':512*1024**2,
             'minimum_host_free_percent':15,'sustained_host_samples':3,'telemetry_timeout_seconds':30}:
            raise ValueError('resource guard differs')


def validate_bindings(root, packet_path, *, check_source=True):
    root=Path(root);packet_path=Path(packet_path)
    packet=json.loads(packet_path.read_text());protocol=bound_json(root,packet['protocol'])
    if packet.get('format')!=PACKET or packet.get('arm') not in ARMS or packet.get('attempts')!=1 or packet.get('retries')!=0:
        raise ValueError('invalid single-attempt packet')
    validate_protocol(protocol)
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
