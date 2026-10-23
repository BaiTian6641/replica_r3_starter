#!/usr/bin/env python3
"""Train and evaluate an E2 reader candidate on source-separated values (MPS)."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib, json, os, shutil, sys, traceback
from pathlib import Path
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def cap(session, selection, max_units=16):
    from collections import defaultdict
    from replica_lab.memory_system.index import RetrievalSelection
    spans = session.spans(selection.records)
    counts = defaultdict(int)
    for span in spans:
        counts[span.record_id] += 1
    records, scores, used = [], [], 0
    for record, score in zip(selection.records, selection.scores):
        if used + counts[record.record_id] > max_units:
            break
        records.append(record); scores.append(score); used += counts[record.record_id]
    return RetrievalSelection(tuple(records), tuple(scores), selection.epoch,
                              selection.ledger_id, selection.scope, selection.query)


def prepare_args(session, tokenizer, query_row, template, selection=None):
    import torch
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.architecture.loading import encode_prompt
    from replica_lab.memory_system import EligibilityScope
    scope = EligibilityScope(query_row['subject_id'], query_row['branch_id'],
                             query_row['purpose'], query_row['as_of'],
                             'assistant_for_source', None)
    prompt = template.replace('{query}', query_row['query'])
    ids = encode_prompt(tokenizer, prompt, 'mps')
    mask = torch.ones_like(ids, dtype=torch.bool)
    if selection is None:
        selection = cap(session, session.index.retrieve(
            query_row['query'], **{name: getattr(scope, name) for name in
                                   ('subject_id', 'branch_id', 'purpose', 'as_of',
                                    'mode', 'snapshot_at')}, limit=16))
    return ids, mask, scope, TaskGraph(query_row['case_id'], query_row['query']), selection


def evaluate(model, tokenizer, ledger, queries, template):
    from replica_lab.memory_system.session import MemorySession
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    results = []
    for query_row in queries:
        ids, mask, scope, task, selection = prepare_args(session, tokenizer, query_row,
                                                         template)
        request = session.prepare(prompt_ids=ids, prompt_mask=mask,
                                  query=query_row['query'], scope=scope, task=task,
                                  selection=selection)
        try:
            generated = session.generate(request, max_new_tokens=8)
            length = int(request.frame.sense.prompt_ids.shape[1])
            new_ids = [int(x) for x in generated[0, length:].tolist()]
            answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            results.append({'case_id': query_row['case_id'], 'split': query_row['split'],
                            'expected': query_row['expected_answer'], 'answer': answer,
                            'correct': answer.casefold() == query_row['expected_answer'].casefold(),
                            'selected_records': [s.record_id for s in request.frame.memory.spans],
                            'generated_token_ids': new_ids})
        finally:
            request.release()
    return results


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.architecture.minicpm5 import MiniCPM5Adapter
    from replica_lab.memory_system import MemoryRecord, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer
    protocol = json.loads(args.protocol.read_text())
    data = ROOT / protocol['dataset']['root']
    if sha(data / 'records.jsonl') != protocol['dataset']['records_sha256']:
        raise ValueError('reader records changed')
    records = [MemoryRecord(**row) for row in rows(data / 'records.jsonl')]
    queries = rows(data / 'queries.jsonl')
    train = [q for q in queries if q['split'] == 'train']
    dev = [q for q in queries if q['split'] == 'dev']
    heldout = [q for q in queries if q['split'] == 'heldout']
    template = ('Answer the query using the stored record. Reply with only the value.\n\n'
                'Query: {query}')
    torch.manual_seed(protocol['training']['seed'])
    model, tokenizer, lock = load_local_mps(args.model_path, args.model_lock,
                                            profiler=profiler)
    parent = model.load_extension(Path(args.parent_extension))
    report.update(source_lock=lock, manifest_digest=model.manifest.digest,
                  parent=parent, parameters=model.parameter_audit(), checks={})
    ledger = RecordLedger(args.output / 'reader_ledger.sqlite')
    for record in records:
        ledger.admit(record, authorized_subject=record.subject_id,
                     authorized_branch=record.branch_id, source_authorized=False)
    parent_results = evaluate(model, tokenizer, ledger, dev + heldout, template)
    report['parent_results'] = parent_results
    report['parent_metrics'] = {
        split: (sum(x['correct'] for x in parent_results if x['split'] == split)
                / sum(1 for x in parent_results if x['split'] == split))
        for split in ('dev', 'heldout')}
    # Answer-loss D1 candidate: writer + both memory readers only.
    previous = {name: p.requires_grad for name, p in model.new_named_parameters()}
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(name.startswith(('record_writer.', 'memory_ports.')))
    optimizer = build_optimizer(model, 'muon8bit', lr_base=0.0,
                                lr_new=protocol['training']['lr_new'],
                                lr_muon_base=0.0,
                                lr_muon_new=protocol['training']['lr_muon_new'],
                                weight_decay=0.0)
    if optimizer is None:
        raise RuntimeError('No trainable reader parameters')
    model.train()
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    report['training'] = {'optimizer': optimizer.audit(), 'updates': []}
    for update in range(1, int(protocol['training']['updates']) + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for offset in range(int(protocol['training']['examples_per_update'])):
            query = train[(4 * (update - 1) + offset) % len(train)]
            ids, mask, scope, task, selection = prepare_args(
                session, tokenizer, query, template)
            target = tokenizer(query['expected_answer'],
                               add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
            target_ids = torch.tensor([target], dtype=torch.long, device='mps')
            action_ids = torch.cat((ids, target_ids), dim=1)
            action_mask = torch.ones_like(action_ids, dtype=torch.bool)
            labels = action_ids.clone(); labels[:, :ids.shape[1]] = -100
            with session.training_request(
                    prompt_ids=ids, prompt_mask=mask, query=query['query'],
                    scope=scope, task=task, selection=selection) as request:
                output = session.forward(request, action_ids, labels=labels,
                                         attention_mask=action_mask)
                loss = output.loss
                total += float(loss.detach())
                session.backward(request, loss)
            del output, loss
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                       1.0, error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        report['training']['updates'].append({'update': update, 'mean_loss': total})
        profiler.mark('reader_update', update=update, loss=total)
    del optimizer
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(previous[name])
    model.eval()
    extension = args.output / 'selected_extension'
    receipt = model.save_extension(extension)
    replacement = MiniCPM5Adapter(model.base, model.manifest).to('mps').eval()
    replacement.seal_original_identity(); replacement.load_extension(extension)
    candidate_results = evaluate(replacement, tokenizer, ledger, dev + heldout, template)
    report['candidate_results'] = candidate_results
    report['candidate_metrics'] = {
        split: (sum(x['correct'] for x in candidate_results if x['split'] == split)
                / sum(1 for x in candidate_results if x['split'] == split))
        for split in ('dev', 'heldout')}
    report['extension'] = {'path': str(extension), 'format': receipt['format'],
                           'weights_sha256': receipt['weights_sha256'],
                           'manifest_digest': receipt['manifest_digest'],
                           'parameters': receipt['parameters'], 'base_saved': False}
    p = report['parent_metrics']; c = report['candidate_metrics']
    if c['heldout'] < p['heldout'] or c['dev'] < p['dev']:
        decision = 'reject'
    elif c['heldout'] > p['heldout']:
        decision = 'accept'
    else:
        decision = 'defer'
    report['decision'] = decision
    report['deployment'] = 'opt_in_only' if decision == 'accept' else 'not_published'
    report['status'] = 'completed'
    ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'configs/e2_reader_candidate_v1.json')
    parser.add_argument('--model-path', type=Path,
                        default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path,
                        default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    parser.add_argument('--parent-extension', type=Path,
                        default=ROOT / 'runs/e1_e2_native_attempt02_20261023/selected_extension')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha(args.protocol),
              'protocol': json.loads(args.protocol.read_text()),
              'downloads': 0, 'optimizer_updates': 0, 'checkpoints_saved': 0}
    profiler = RunProfiler('mps', args.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            execute(args, profiler, report)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        profile = profiler.stop(); report['profile'] = profile
        memory = (profile or {}).get('memory') or {}
        report['costs'] = {'profiled_wall_seconds': (profile or {}).get('elapsed_seconds'),
                           'sampled_peak_driver_bytes': memory.get('mps_driver_peak_sampled_bytes')}
        report['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        (args.output / ('run_report.json' if report['status'] == 'completed'
                        else 'failure_report.json')).write_text(
            json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'parent': report.get('parent_metrics'),
                      'candidate': report.get('candidate_metrics'),
                      'decision': report.get('decision')}))


if __name__ == '__main__':
    main()
