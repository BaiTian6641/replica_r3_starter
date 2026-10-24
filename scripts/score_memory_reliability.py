#!/usr/bin/env python3
"""CPU-only verification of a bounded memory pilot; never trust cached scores."""
from __future__ import annotations
from dataclasses import asdict

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.answer_audit import audit_rows
from replica_lab.memory_reliability import audit_dataset, cases
from replica_lab.reliability_scoring import metrics_and_checks, regression_inventory, family_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_producer(path, expected_sha):
    """Reconstruct the stable writer/reader producer from actual saved tensors."""
    from safetensors.torch import load_file
    from replica_lab.architecture.contracts import fingerprint
    receipt = json.loads((path / 'manifest.json').read_text())
    if (sha(path / 'extensions.safetensors') != expected_sha or receipt['weights_sha256'] != expected_sha
            or fingerprint(receipt['manifest']) != receipt['manifest_digest']):
        raise ValueError('Extension bytes or component manifest changed')
    weights = load_file(str(path / 'extensions.safetensors'))
    names = receipt['parameters']['new']['names']
    if len(names) != len(set(names)) or set(weights) != set(names):
        raise ValueError('Extension parameter inventory differs')
    tensors = []
    for prefix, source in [('writer', 'fast_depth_write_encoders.'), ('ports', 'fast_ports.')]:
        matching = [name for name in names if name.startswith(source)]
        if not matching:
            raise ValueError('Missing persistent writer/reader parameters')
        for name in matching:
            value = weights[name]
            tensors.append((prefix + '.' + name.removeprefix(source), str(value.dtype), tuple(value.shape),
                            hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()))
    return fingerprint({'manifest': receipt['manifest_digest'], 'tensors': tensors})


def audit_changed_tensors(parent_path, candidate_path, expected_names):
    """Require exactly the protocol-declared trainable tensors to differ."""
    import torch
    from safetensors.torch import load_file
    parent = load_file(str(Path(parent_path) / 'extensions.safetensors'))
    candidate = load_file(str(Path(candidate_path) / 'extensions.safetensors'))
    if set(parent) != set(candidate):
        raise ValueError('Parent/candidate tensor inventory differs')
    for name in parent:
        if parent[name].shape != candidate[name].shape or parent[name].dtype != candidate[name].dtype:
            raise ValueError('Parent/candidate tensor shape or dtype differs')
    changed = {name for name in parent if not torch.equal(parent[name], candidate[name])}
    if changed != set(expected_names):
        raise ValueError('Changed tensors differ from declared reader-only scope')
    return {'changed_tensors': sorted(changed),
            'unchanged_tensor_count': len(parent) - len(changed)}


def audit_images(run, report, episodes, producer, protocol=None):
    from replica_lab.runtime.fast_memory import FastMemoryConfig, FastMemoryScope, load_fast_depth_image
    config = FastMemoryConfig(heads=4, key_size=64, value_size=64, update_rule='sgdn2',
                              max_tokens_per_chunk=256, max_image_bytes=1048576)
    evaluation = [e for e in episodes if e['split'] != 'train']
    transitions = report['transitions']
    if (len(transitions) != len(evaluation)
            or {t['episode_id'] for t in transitions} != {e['episode_id'] for e in evaluation}):
        raise ValueError('Persistent transition inventory differs')
    for episode in evaluation:
        eid = episode['episode_id']
        transition = next(t for t in transitions if t['episode_id'] == eid)
        scope = FastMemoryScope('memory_reliability', eid, 'reliability', 'parallel_replica', 0)
        expected = [(f'{eid}:source:{i}', hashlib.sha256(text.encode()).hexdigest())
                    for i, text in enumerate(episode['observations'])]
        for stage, directory, digest_key in [('initial', 'before', 'initial_digest'), ('corrected', 'corrected', 'corrected_digest')]:
            if stage == 'corrected':
                expected.append((f'{eid}:correction', hashlib.sha256(episode['correction_observation'].encode()).hexdigest()))
            path = run / 'images' / eid / directory
            if stage == 'corrected' and Path(transition['image']).resolve() != path.resolve():
                raise ValueError('Unexpected persisted image path')
            read, metadata = load_fast_depth_image(path, config=config, scope=scope, producer=producer, device='cpu')
            digest = hashlib.sha256(read.weights.float().numpy().tobytes()).hexdigest()
            if (metadata['lineage'] != [list(item) for item in expected]
                    or not metadata['cursor'] == metadata['chunk_index'] == metadata['update_count'] == len(expected)
                    or digest != transition[digest_key]):
                raise ValueError('Persistent image lineage, counters or digest differ')
            rows = [r for r in report['candidate'] if r['episode_id'] == eid and r['stage'] == stage and r['arm'] == 'fast']
            if any(r['state_digest'] != digest for r in rows):
                raise ValueError('Answers did not share the persisted state')
            if protocol and protocol.get('exact_reference', {}).get('enabled', False):
                from replica_lab.architecture.contracts import fingerprint
                from replica_lab.memory_system import EligibilityScope
                from replica_lab.runtime.exact_memory import episode_records
                initial_records, correction_record = episode_records(episode)
                binding = transition.get('initial_source_binding' if stage == 'initial' else 'corrected_source_binding')
                if not binding:
                    raise ValueError('Missing source-view binding')
                sidecar = path / 'source_binding.json'
                if not sidecar.is_file() or json.loads(sidecar.read_text()) != binding:
                    raise ValueError('Image source-binding sidecar differs')
                from replica_lab.runtime.exact_memory import ExactMemoryReference
                with ExactMemoryReference(run / 'ledger.sqlite') as reference:
                    if binding['ledger_id'] != reference.ledger.ledger_id:
                        raise ValueError('Source-binding ledger identity differs')
                    episode_index = evaluation.index(episode)
                    if report.get('evaluation_only', False):
                        expected_epoch = sum(len(episode_records(item)[0]) + 1 for item in evaluation)
                    else:
                        expected_epoch = (sum(len(episode_records(item)[0]) + 1 for item in evaluation)
                                          + sum(len(episode_records(item)[0]) + 1
                                                for item in episodes if item['split'] == 'train'))
                    if binding['epoch'] != expected_epoch or reference.ledger.epoch != expected_epoch:
                        raise ValueError('Source-binding epoch differs from admission sequence')
                if stage == 'initial':
                    current_records, as_of = initial_records, 2
                else:
                    current_records = tuple(r for r in initial_records
                                            if r.record_id != correction_record.supersedes) + (correction_record,)
                    as_of = 3
                current_records = tuple(sorted(current_records,
                                               key=lambda r: (r.observed_at, r.event_at, r.record_id)))
                view_scope = EligibilityScope('memory_reliability', eid, 'reliability', as_of,
                                              'parallel_replica', None)
                expected_view_digest = fingerprint({
                    'scope': asdict(view_scope), 'epoch': binding['epoch'],
                    'records': [(r.record_id, r.record_hash) for r in current_records]})
                if binding['view_digest'] != expected_view_digest:
                    raise ValueError('Source-view digest differs from authenticated dataset')
    return {'images_authenticated': len(evaluation)*2, 'actual_tensor_bytes_checked': True,
            'scope_source_lineage_and_counters_checked': True, 'multiple_queries_per_state': True,
            'source_bindings_authenticated': (len(evaluation)*2
                                               if protocol and protocol.get('exact_reference', {}).get('enabled', False) else 0)}


def audit_guard(path, protocol, run):
    guard = json.loads(path.read_text())
    if (guard['status'] != 'completed_within_declared_sampled_limits'
            or guard['child_exit_code'] != 0 or Path(guard['output']).resolve() != run.resolve()):
        raise ValueError('Native guard did not pass for this run')
    # Bind actual command and resource evidence, not just the protocol declaration.
    command = guard['command']
    if command[command.index('--protocol') + 1] != protocol['protocol_path']:
        raise ValueError('Guard command used another protocol')
    for name, value in guard['implementation_sha256'].items():
        if protocol['implementation_sha256'].get(name) != value:
            raise ValueError('Guard implementation differs from protocol')
    policy = guard['guard']['policy']
    expected = {'driver_limit_bytes': 50*1024**3, 'swap_growth_limit_bytes': 512*1024**2,
                'minimum_free_percent': 15, 'sustained_host_samples': 3}
    if policy != expected:
        raise ValueError('Actual guard policy differs from declared reliability limits')
    return {'path': str(path), 'sha256': sha(path), 'policy': policy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--guard', type=Path, required=True)
    parser.add_argument('--restart-run', type=Path)
    parser.add_argument('--evaluation-only', action='store_true')
    parser.add_argument('--restart-guard', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh verdict output path')
    protocol = json.loads(args.protocol.read_text())
    for name, expected in protocol['implementation_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT/name) != expected:
            raise ValueError(f'Implementation changed: {name}')
    if sha(ROOT/protocol['dataset']['path']) != protocol['dataset']['sha256']:
        raise ValueError('Dataset bytes changed')
    episodes = json.loads((ROOT/protocol['dataset']['path']).read_text())
    dataset_audit = audit_dataset(episodes, require_novel_values=protocol['dataset'].get('require_novel_values', False))
    lock_path = ROOT / 'configs/minicpm5_sealed_lock.local.json'
    if sha(lock_path) != protocol['source_lock_sha256']:
        raise ValueError('Source lock changed')
    lock = json.loads(lock_path.read_text())
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja'):
        if sha(ROOT/'checkpoints/minicpm5_source'/name) != lock['snapshot_files_sha256'][name]:
            raise ValueError('Sealed tokenizer changed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'checkpoints/minicpm5_source', local_files_only=True, trust_remote_code=False)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)
    report_path = args.run / 'run_report.json'
    report = json.loads(report_path.read_text())
    if bool(report.get('evaluation_only', False)) != args.evaluation_only:
        raise ValueError('Scorer evaluation-only mode differs from run report')
    if (report['status'] != 'completed' or report['protocol_sha256'] != sha(args.protocol)
            or report['implementation_sha256'] != protocol['implementation_sha256'] or report['source_lock'] != lock):
        raise ValueError('Run provenance differs from protocol')
    expected_tensors = set(protocol['training'].get('expected_trainable_tensors', ()))
    if not args.evaluation_only:
        training = report['training']
        expected_ids = {q['case_id'] for e in episodes if e['split'] == 'train'
                        for stage in ('initial', 'corrected') for q in cases(e, stage)}
        if (set(training['case_ids']) != expected_ids or len(training['case_ids']) != len(expected_ids)
                or report['optimizer_updates'] != protocol['training']['updates']
                or len(training['updates']) != protocol['training']['updates'] or report['checkpoints_saved'] != 1):
            raise ValueError('Training exposure or update/checkpoint budget differs')
        prefixes = tuple(protocol['training'].get('trainable_prefixes',
                                                ('fast_depth_write_encoders.', 'fast_ports.')))
        audit = training['parameter_audit']
        if (audit['original']['trainable'] != 0 or not audit['new']['names']
                or any(not name.startswith(prefixes) for name in audit['new']['names'])
                or (expected_tensors and set(audit['new']['names']) != expected_tensors)
                or training.get('trainable_prefixes') not in (None, list(prefixes))):
            raise ValueError('Training ownership differs from declared D1 scope')
        probe = training['updates'][1]['gradients']
        if probe['original']['present'] or probe['new']['nonfinite']:
            raise ValueError('Fast-memory gradient ownership failed')
        if expected_tensors:
            if (set(probe['new']['nonzero']) != expected_tensors
                    or any(name.startswith('fast_depth_write_encoders.') for name in probe['new']['present'])):
                raise ValueError('Declared reader gradients differ from frozen-writer scope')
        elif (not any(n.startswith('fast_depth_write_encoders.') for n in probe['new']['nonzero'])
              or not any(n.startswith('fast_ports.') for n in probe['new']['nonzero'])):
            raise ValueError('Fast-memory gradient ownership failed')
    else:
        if report['optimizer_updates'] != 0 or report['checkpoints_saved'] != 0:
            raise ValueError('Evaluation-only report contains training or checkpoint updates')
    if report['parent_extension']['weights_sha256'] != protocol['parent_weights_sha256']:
        raise ValueError('Parent lineage differs')
    checkpoint_producer(ROOT/protocol['parent_extension'], protocol['parent_weights_sha256'])
    candidate_path = ROOT / protocol['candidate_extension'] if args.evaluation_only else args.run / 'candidate_extension'
    producer = checkpoint_producer(candidate_path, report['extension']['weights_sha256'])
    if producer != report['candidate_fast_version']:
        raise ValueError('Saved candidate does not produce the persisted state')
    checkpoint_changes = (audit_changed_tensors(
        ROOT / protocol['parent_extension'], args.run / 'candidate_extension', expected_tensors)
        if expected_tensors and not args.evaluation_only else None)
    metrics, checks = metrics_and_checks(report, episodes, decode, protocol)
    regressions = regression_inventory(report, episodes, decode, protocol.get('primary_fast_arm', 'fast'))
    families = family_metrics(report, episodes, decode)
    images = audit_images(args.run, report, episodes, producer, protocol)
    if images['images_authenticated'] != protocol['budget']['depth_images']:
        raise ValueError('Depth-image count differs from declared budget')
    guard = audit_guard(args.guard, protocol, args.run)
    rows = report['parent'] + report['candidate']
    if (len(rows) != report['generation_answers'] or len(rows) > protocol['budget']['max_generated_answers']
            or any(len(r['token_ids']) > 6 or r['prompt_tokens'] > 256 for r in rows)):
        raise ValueError('Generation budget differs')
    restart_audit = None
    if args.restart_run:
        if not args.restart_guard:
            parser.error('--restart-guard is required with --restart-run')
        path = args.restart_run / 'run_report.json'
        restart = json.loads(path.read_text())
        if (restart['status'] != 'completed' or restart['previous_report_sha256'] != sha(report_path)
                or restart['protocol_sha256'] != sha(args.protocol) or restart['source_reencoded'] is not False
                or restart['optimizer_updates'] != 0 or restart['checkpoints_saved'] != 0):
            raise ValueError('Restart provenance differs')
        episode = next(e for e in episodes if e['split'] == 'heldout')
        expected = {q['case_id']: q for q in cases(episode, 'corrected')}
        verified = audit_rows(restart['answers'], expected, decode)
        primary = protocol.get('primary_fast_arm', 'fast')
        reference = {r['case_id']: r for r in report['candidate'] if r['arm'] == primary}
        checks['fresh_process_tokens_and_state'] = all(r['token_ids'] == reference[r['case_id']]['token_ids']
                                                      and r['state_digest'] == reference[r['case_id']]['state_digest'] for r in verified)
        if protocol.get('exact_reference', {}).get('enabled', False):
            transition = next(t for t in report['transitions'] if t['episode_id'] == episode['episode_id'])
            checks['fresh_process_source_view'] = (
                restart.get('source_binding') == transition.get('corrected_source_binding'))
        restart_audit = {'report_sha256': sha(path), 'answers_decoded': len(verified),
                         'guard': audit_guard(args.restart_guard, protocol, args.restart_run)}
    reference_checks = {name: value for name, value in checks.items() if name.startswith('reference_')}
    candidate_checks = checks
    reference_decision = (('accept_reference' if all(reference_checks.values()) else 'reject_reference')
                          if reference_checks else None)
    decision = ('accept_diagnostic' if all(candidate_checks.values()) and restart_audit
                else 'defer_restart' if all(candidate_checks.values()) else 'reject')
    result = {'format': 'replica.memory_reliability.verdict.v1', 'decision': decision,
              'deployment': 'diagnostic_only_not_published', 'report_sha256': sha(report_path),
              'protocol_sha256': sha(args.protocol), 'dataset_audit': dataset_audit,
              'decoded_answer_rows': len(rows), 'images': images, 'guard': guard, 'restart': restart_audit,
              'metrics': metrics, 'checks': checks,
              'regressions': regressions, 'families': families,
              'failed_checks': [name for name, value in checks.items() if not value],
              'checkpoint_changes': checkpoint_changes,
              'reference_decision': reference_decision, 'candidate_decision': decision,
              'evaluation_only': args.evaluation_only,
              'scope': 'Small exposed synthetic diagnostic; no RSI or broad memory claim'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'decision': decision, 'failed_checks': result['failed_checks']}))


if __name__ == '__main__':
    main()
