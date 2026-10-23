#!/usr/bin/env python3
"""Evaluate the existing E2 reranker candidate against its parent (MPS, no training)."""
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


def rank_case(session, tokenizer, query_row, template):
    import torch
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.architecture.loading import encode_prompt
    from replica_lab.memory_system import EligibilityScope
    scope = EligibilityScope(query_row['subject_id'], query_row['branch_id'],
                             query_row['purpose'], query_row['as_of'],
                             'assistant_for_source', None)
    pool = cap(session, session.index.retrieve(
        query_row['query'], **{name: getattr(scope, name) for name in
                               ('subject_id', 'branch_id', 'purpose', 'as_of', 'mode',
                                'snapshot_at')}, limit=16))
    prompt = template.replace('{query}', query_row['query'])
    ids = encode_prompt(tokenizer, prompt, 'mps')
    mask = torch.ones_like(ids, dtype=torch.bool)
    request = session.prepare(prompt_ids=ids, prompt_mask=mask, query=query_row['query'],
                              scope=scope, task=TaskGraph(query_row['case_id'],
                                                          query_row['query']),
                              selection=pool)
    try:
        frame = request.frame
        query_features = dict(frame.sense.summaries)[
            session.adapter.ports.injection_layers[0]][:, 0]
        valid = frame.memory.valid.any(dim=2)
        with torch.no_grad():
            scores = session.adapter.reranker(
                query_features, frame.memory.slots.mean(dim=2), valid)[0]
        record_scores = {}
        for index, span in enumerate(frame.memory.spans):
            if bool(valid[0, index]):
                record_scores[span.record_id] = max(
                    record_scores.get(span.record_id, -1e9), float(scores[index]))
        order = sorted(pool.records, key=lambda r: -record_scores.get(r.record_id, -1e9))
        lexical_order = list(pool.records)
    finally:
        request.release()
    expected = query_row['expected_record_id']
    return {
        'case_id': query_row['case_id'], 'expected': expected,
        'ranked': [record.record_id for record in order],
        'lexical_ranked': [record.record_id for record in lexical_order],
        'rank': ([record.record_id for record in order].index(expected) + 1
                 if expected in [record.record_id for record in order] else None),
        'lexical_rank': ([record.record_id for record in lexical_order].index(expected) + 1
                         if expected in [record.record_id for record in lexical_order] else None),
        'scores': {record.record_id: record_scores.get(record.record_id)
                   for record in pool.records},
    }


def evaluate_arm(path, extension, queries, model_args, profiler, template):
    import torch
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import EligibilityScope, MemoryRecord, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    model, tokenizer, lock = load_local_mps(model_args.model_path, model_args.model_lock,
                                            profiler=profiler)
    receipt = model.load_extension(extension)
    ledger = RecordLedger(path['db'])
    for row in rows(path['records']):
        record = MemoryRecord(**row)
        ledger.admit(record, authorized_subject=record.subject_id,
                     authorized_branch=record.branch_id, source_authorized=False)
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    results = [rank_case(session, tokenizer, query, template) for query in queries]
    ledger.close()
    return {'extension': str(extension), 'receipt': receipt, 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'configs/e5_neural_candidate_eval_v1.json')
    parser.add_argument('--model-path', type=Path,
                        default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path,
                        default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    data = ROOT / protocol['dataset']['root']
    if sha(data / 'records.jsonl') != protocol['dataset']['records_sha256']:
        raise ValueError('dataset records changed')
    all_queries = {row['case_id']: row for row in rows(data / 'queries.jsonl')}
    queries = [all_queries[item] for role in ('dev', 'heldout')
               for item in protocol['evaluation_cases'][role]]
    report = {'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha(args.protocol), 'protocol': protocol,
              'downloads': 0, 'optimizer_updates': 0, 'checkpoints_saved': 0}
    profiler = RunProfiler('mps', args.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            template = ('Return the record that answers the query. Query: {query}')
            parent = evaluate_arm({'db': args.output / 'parent.sqlite',
                                   'records': data / 'records.jsonl'},
                                  ROOT / protocol['parent_extension'], queries, args,
                                  profiler, template)
            candidate = evaluate_arm({'db': args.output / 'candidate.sqlite',
                                      'records': data / 'records.jsonl'},
                                     ROOT / protocol['candidate_extension'], queries, args,
                                     profiler, template)
            parent_h = [r for r, q in zip(parent['results'], queries) if q['split'] == 'heldout']
            candidate_h = [r for r, q in zip(candidate['results'], queries) if q['split'] == 'heldout']
            parent_d = [r for r, q in zip(parent['results'], queries) if q['split'] == 'dev']
            candidate_d = [r for r, q in zip(candidate['results'], queries) if q['split'] == 'dev']
            def top1(values): return sum(item['rank'] == 1 for item in values) / len(values)
            def top4(values): return sum(item['rank'] is not None and item['rank'] <= 4
                                         for item in values) / len(values)
            metrics = {'parent': {'heldout_top1': top1(parent_h), 'heldout_top4': top4(parent_h),
                                  'dev_top1': top1(parent_d)},
                       'candidate': {'heldout_top1': top1(candidate_h),
                                     'heldout_top4': top4(candidate_h),
                                     'dev_top1': top1(candidate_d)},
                       'lexical_reference': {'heldout_top1': sum(item['lexical_rank'] == 1
                                                                 for item in parent_h) / len(parent_h)}}
            if metrics['candidate']['heldout_top1'] < metrics['parent']['heldout_top1'] \
                    or metrics['candidate']['dev_top1'] < metrics['parent']['dev_top1']:
                decision = 'reject'
            elif metrics['candidate']['heldout_top1'] > metrics['parent']['heldout_top1']:
                decision = 'accept'
            else:
                decision = 'defer'
            parent_receipt = parent['receipt']
            candidate_receipt = candidate['receipt']
            from replica_lab.runtime.artifacts import NeuralCandidate
            candidate_artifact = NeuralCandidate(
                'e2-reranker-candidate', parent_receipt['manifest_digest'],
                tuple(name for name, _ in [] ) or ('reranker.query_projection.weight',
                                                    'reranker.record_projection.weight',
                                                    'sense_ports.13.projection.weight'),
                candidate_receipt['weights_sha256'], 'modules_only', 'muon8bit',
                {'e2': parent_receipt['manifest_digest']},
                ('e2_retrieval_train',), evaluation=('passed' if decision == 'accept'
                                                     else 'failed' if decision == 'reject'
                                                     else 'inconclusive'),
                status=('accepted' if decision == 'accept' else
                        'rejected' if decision == 'reject' else 'proposed'))
            deployment = ('opt_in_only; lexical remains the default'
                          if decision == 'accept' else
                          'not_published' if decision == 'reject' else 'deferred')
            report.update(parent=parent, candidate=candidate, metrics=metrics,
                          decision=decision, deployment=deployment,
                          candidate_artifact={'artifact_id': candidate_artifact.artifact_id,
                                              'digest': candidate_artifact.digest,
                                              'status': candidate_artifact.status,
                                              'evaluation': candidate_artifact.evaluation,
                                              'weights_sha256': candidate_artifact.weights_sha256},
                          status='completed')
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
                        else 'failure_report.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'metrics': report.get('metrics'),
                      'decision': report.get('decision')}))


if __name__ == '__main__':
    main()
