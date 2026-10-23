#!/usr/bin/env python3
"""Preregistered E2 retrieval/novel-record qualification on MPS.

Trains only the reranker and block-13 sensor through the explicit ranking
objective, compares lexical/native-dense/learned-rerank arms, and uses an
oracle diagnostic only to isolate reader errors. Eligibility filtering is host
side and precedes every ranking arm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_prompt(template, query):
    return template.replace('{query}', query)


def cap_selection(session, selection, max_units=16):
    """Add records in order until the next record would exceed the unit bank."""
    from replica_lab.memory_system.index import RetrievalSelection
    if not selection.records:
        return selection
    spans = session.spans(selection.records)
    counts = defaultdict(int)
    for span in spans:
        counts[span.record_id] += 1
    records, scores, used = [], [], 0
    for record, score in zip(selection.records, selection.scores):
        count = counts[record.record_id]
        if used + count > max_units:
            break
        records.append(record)
        scores.append(score)
        used += count
    if not records:
        raise ValueError('First candidate record exceeds the unit bank; narrow the source')
    return RetrievalSelection(tuple(records), tuple(scores), selection.epoch,
                              selection.ledger_id, selection.scope, selection.query,
                              selection.dense_producer_id)


def lexical_selection(session, query, scope):
    selection = session.index.retrieve(query, **asdict(scope), limit=16)
    return cap_selection(session, selection)


def dense_selection(session, query, scope):
    selection = session.dense_candidates(query, scope, limit=16, lexical_weight=0.0)
    return cap_selection(session, selection)


def rerank_selection(session, query, scope, prompt_ids, prompt_mask, task):
    from replica_lab.memory_system.index import RetrievalSelection
    import torch
    pool = cap_selection(session, session.index.retrieve(query, **asdict(scope), limit=16))
    if not pool.records:
        return pool, {}
    request = session.prepare(prompt_ids=prompt_ids, prompt_mask=prompt_mask,
                              query=query, scope=scope, task=task, selection=pool)
    try:
        frame = request.frame
        query_features = dict(frame.sense.summaries)[
            session.adapter.ports.injection_layers[0]][:, 0]
        valid = frame.memory.valid.any(dim=2)
        with torch.no_grad():
            scores = session.adapter.reranker(
                query_features, frame.memory.slots.mean(dim=2), valid)[0]
        if not bool(torch.isfinite(scores[valid[0]]).all()):
            raise ValueError('Nonfinite reranker score')
        record_scores = {}
        for index, span in enumerate(frame.memory.spans):
            if not bool(valid[0, index]):
                continue
            value = float(scores[index])
            record_scores[span.record_id] = max(record_scores.get(span.record_id, -1e9), value)
    finally:
        request.release()
    order = sorted(range(len(pool.records)),
                   key=lambda i: (-record_scores.get(pool.records[i].record_id, -1e9), i))
    records = tuple(pool.records[i] for i in order)
    values = tuple(record_scores.get(record.record_id, -1e9) for record in records)
    ranked = RetrievalSelection(records, values, pool.epoch, pool.ledger_id, pool.scope,
                                pool.query, pool.dense_producer_id)
    return cap_selection(session, ranked), record_scores


def oracle_selection(session, query, scope, expected):
    from replica_lab.memory_system.index import RetrievalSelection
    snap = session.ledger.snapshot(scope)
    record = next((item for item in snap.records if item.record_id == expected), None)
    if record is None:
        return RetrievalSelection((), (), snap.epoch, snap.ledger_id, scope, query)
    return RetrievalSelection((record,), (1.0,), snap.epoch, snap.ledger_id, scope, query)


def answer_once(session, tokenizer, query, scope, task, selection, *, max_new_tokens):
    prompt = session._e2_prompt(query)
    ids = session._encode(prompt)
    mask = session._mask(ids)
    request = session.prepare(prompt_ids=ids, prompt_mask=mask, query=query,
                              scope=scope, task=task, selection=selection)
    try:
        generated = session.generate(request, max_new_tokens=max_new_tokens)
        prompt_tokens = int(request.frame.sense.prompt_ids.shape[1])
        new_ids = tuple(int(token) for token in generated[0, prompt_tokens:].tolist())
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        spans = request.frame.memory.spans
        return {
            'answer': text, 'generated_token_ids': list(new_ids),
            'generated_tokens': len(new_ids), 'prompt_tokens': prompt_tokens,
            'selected_records': list(dict.fromkeys(span.record_id for span in spans)),
            'selected_spans': [{'record_id': span.record_id,
                                'token_start': span.token_start,
                                'token_end': span.token_end} for span in spans],
        }
    finally:
        request.release()


def evaluate_case(session, tokenizer, query_row, arms, *, max_new_tokens):
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.memory_system import EligibilityScope
    scope = EligibilityScope(query_row['subject_id'], query_row['branch_id'],
                             query_row['purpose'], query_row['as_of'],
                             'assistant_for_source', None)
    task = TaskGraph(query_row['case_id'], query_row['query'])
    prompt_ids = session._encode(session._e2_prompt(query_row['query']))
    prompt_mask = session._mask(prompt_ids)
    snap = session.ledger.snapshot(scope)
    eligible_ids = {record.record_id for record in snap.records}
    forbidden = set(query_row['forbidden_record_ids'])
    selection_map = {
        'lexical': lexical_selection(session, query_row['query'], scope),
        'dense': dense_selection(session, query_row['query'], scope),
    }
    rerank, rerank_scores = rerank_selection(
        session, query_row['query'], scope, prompt_ids, prompt_mask, task)
    selection_map['rerank'] = rerank
    if query_row['expected_record_id'] is not None:
        selection_map['oracle_diagnostic'] = oracle_selection(
            session, query_row['query'], scope, query_row['expected_record_id'])
    results = []
    for arm in arms:
        if arm not in selection_map:
            continue
        selection = selection_map[arm]
        observation = answer_once(session, tokenizer, query_row['query'], scope, task,
                                  selection, max_new_tokens=max_new_tokens)
        selected = observation['selected_records']
        expected = query_row['expected_record_id']
        rank = selected.index(expected) + 1 if expected in selected else None
        result = {
            **observation,
            'case_id': query_row['case_id'], 'split': query_row['split'],
            'kind': query_row['kind'], 'arm': arm,
            'expected_record_id': expected,
            'expected_answer': query_row['expected_answer'],
            'rank': rank,
            'recall_at_1': bool(rank == 1),
            'recall_at_4': bool(rank is not None and rank <= 4),
            'retrieval_applicable': expected is not None and arm != 'oracle_diagnostic',
            'answer_correct': observation['answer'].casefold()
                              == str(query_row['expected_answer']).casefold(),
            'abstained': observation['answer'].casefold() == 'unknown',
            'forbidden_selected': sorted(set(selected) & forbidden),
            'eligible_forbidden': sorted(forbidden & eligible_ids),
            'distractors_selected': len([item for item in selected if item != expected]),
            'span_count': len(observation['selected_spans']),
            'answer_span_included': (None if query_row.get('answer_token_start') is None
                                     else any(span['record_id'] == expected
                                              and span['token_start'] <= query_row['answer_token_start']
                                              < span['token_end']
                                              for span in observation['selected_spans'])),
            'rerank_score': (rerank_scores.get(expected) if arm == 'rerank'
                             and expected is not None else None),
        }
        results.append(result)
    return results


def summarize(results):
    by_arm = defaultdict(lambda: defaultdict(list))
    for item in results:
        by_arm[item['arm']][item['split']].append(item)
    summary = {}
    for arm, splits in by_arm.items():
        summary[arm] = {}
        for split, rows in splits.items():
            relevant = [r for r in rows if r['retrieval_applicable']]
            summary[arm][split] = {
                'cases': len(rows), 'relevant_cases': len(relevant),
                'recall_at_1': (sum(r['recall_at_1'] for r in relevant)
                                / len(relevant)) if relevant else None,
                'recall_at_4': (sum(r['recall_at_4'] for r in relevant)
                                / len(relevant)) if relevant else None,
                'answer_correct': sum(r['answer_correct'] for r in rows) / len(rows),
                'abstentions': sum(r['abstained'] for r in rows),
                'forbidden_selected': sum(len(r['forbidden_selected']) for r in rows),
                'eligible_forbidden': sum(len(r['eligible_forbidden']) for r in rows),
                'mean_distractors': (sum(r['distractors_selected'] for r in rows)
                                     / len(rows)),
                'long_span_counts': [r['span_count'] for r in rows
                                     if r['kind'] == 'long_record_chunking'],
                'long_answer_span': [r['answer_span_included'] for r in rows
                                     if r['kind'] == 'long_record_chunking'],
            }
    return summary


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.loading import encode_prompt, load_local_mps
    from replica_lab.architecture.minicpm5 import MiniCPM5Adapter
    from replica_lab.memory_system import EligibilityScope, MemoryRecord, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.optimizers import build_optimizer

    protocol = json.loads(args.protocol.read_text())
    if protocol['format'] != 'replica.e2_retrieval.qualification.v1':
        raise ValueError('Unknown retrieval protocol')
    data_root = ROOT / protocol['dataset']['root']
    manifest = json.loads((data_root / 'manifest.json').read_text())
    if sha(data_root / 'records.jsonl') != protocol['dataset']['records_sha256']:
        raise ValueError('Retrieval records differ from the preregistered hash')
    if sha(data_root / 'queries.jsonl') != protocol['dataset']['queries_sha256']:
        raise ValueError('Retrieval queries differ from the preregistered hash')
    if sha(data_root / 'manifest.json') != protocol['dataset']['manifest_sha256']:
        raise ValueError('Retrieval manifest differs from the preregistered hash')
    records = [MemoryRecord(**row) for row in load_rows(data_root / 'records.jsonl')]
    queries = load_rows(data_root / 'queries.jsonl')
    training = protocol['training']
    if training['optimizer'] != 'muon8bit' or training['lr_muon_new'] > 0.01:
        raise ValueError('Protocol training scope/rate mismatch')
    if training['trainable'] != ['reranker', 'sense_ports.13']:
        raise ValueError('Reranker training must declare its exact scope')
    train_rows = [q for q in queries if q['split'] == 'train']
    if not train_rows:
        raise ValueError('No train queries')
    torch.manual_seed(training['seed'])
    model, tokenizer, source_lock = load_local_mps(
        args.model_path, args.model_lock, profiler=profiler)
    report.update(manifest=model.manifest.to_dict(), manifest_digest=model.manifest.digest,
                  source_lock=source_lock, parameters=model.parameter_audit(), checks={},
                  dataset={'manifest': manifest, 'files': {
                      'records': protocol['dataset']['records_sha256'],
                      'queries': protocol['dataset']['queries_sha256'],
                      'manifest': protocol['dataset']['manifest_sha256']}})
    parent = Path(args.parent_extension)
    parent_receipt = model.load_extension(parent)
    report['parent_extension'] = {
        'path': str(parent.resolve()), 'format': parent_receipt['format'],
        'weights_sha256': parent_receipt['weights_sha256'],
        'manifest_digest': parent_receipt['manifest_digest']}
    ledger = RecordLedger(args.output / 'retrieval_ledger.sqlite')
    for record in records:
        ledger.admit(record, authorized_subject=record.subject_id,
                     authorized_branch=record.branch_id, source_authorized=False)
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    session._e2_prompt = lambda query: protocol['generation']['prompt'].replace(
        '{query}', query)
    session._encode = lambda text: encode_prompt(tokenizer, text, 'mps')
    session._mask = lambda ids: torch.ones_like(ids, dtype=torch.bool)

    def mark(name, **fields):
        profiler.mark(name, **fields)
        dump(args.output / 'progress.json', report)
        print(json.dumps({'phase': name, **fields}), flush=True)

    def require(name, value, **details):
        report['checks'][name] = {'passed': bool(value), **details}
        if not value:
            raise AssertionError(name)

    # Host eligibility is separate from ranking and is checked before training/eval.
    eligibility = []
    for query in queries:
        scope = EligibilityScope(query['subject_id'], query['branch_id'],
                                 query['purpose'], query['as_of'],
                                 'assistant_for_source', None)
        eligible = {record.record_id for record in ledger.snapshot(scope).records}
        forbidden = set(query['forbidden_record_ids'])
        eligibility.append({'case_id': query['case_id'], 'split': query['split'],
                            'eligible': sorted(eligible),
                            'forbidden_eligible': sorted(forbidden & eligible),
                            'expected_eligible': (query['expected_record_id'] in eligible
                                                  if query['expected_record_id'] else None)})
    require('eligibility_precedes_ranking',
            all(not item['forbidden_eligible'] for item in eligibility),
            cases=len(eligibility))

    # Explicit ranking objective only: reranker + block-13 sensor.
    previous_flags = {name: parameter.requires_grad
                      for name, parameter in model.new_named_parameters()}
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(name.startswith(('reranker.', 'sense_ports.13.')))
    optimizer = build_optimizer(model, 'muon8bit', lr_base=0.0,
                                lr_new=training['lr_new'],
                                lr_muon_base=0.0,
                                lr_muon_new=training['lr_muon_new'],
                                weight_decay=training['weight_decay'])
    if optimizer is None:
        raise RuntimeError('No trainable reranker parameters')
    model.train()
    report['training'] = {'optimizer': optimizer.audit(), 'updates': [],
                          'trainable': training['trainable'],
                          'objective': training['objective']}
    for update in range(1, int(training['updates']) + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for query_row in train_rows:
            scope = EligibilityScope(query_row['subject_id'], query_row['branch_id'],
                                     query_row['purpose'], query_row['as_of'],
                                     'assistant_for_source', None)
            selection = lexical_selection(session, query_row['query'], scope)
            prompt_ids = session._encode(session._e2_prompt(query_row['query']))
            prompt_mask = session._mask(prompt_ids)
            from replica_lab.architecture.contracts import TaskGraph
            with session.training_request(
                    prompt_ids=prompt_ids, prompt_mask=prompt_mask,
                    query=query_row['query'], scope=scope,
                    task=TaskGraph(query_row['case_id'], query_row['query']),
                    selection=selection, train_sense=True) as request:
                loss = session.ranking_loss(request, query_row['expected_record_id'])
                total += float(loss.detach())
                session.backward(request, loss)
            del loss
        norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            training['gradient_clip'], error_if_nonfinite=True))
        if update == 1:
            report['training']['gradient_audit_update1'] = model.gradient_audit()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report['training']['updates'].append({'update': update, 'mean_loss': total,
                                              'gradient_norm': norm})
        mark('reranker_update', update=update, loss=total)
    report['training']['optimizer_after'] = optimizer.audit()
    del optimizer
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(previous_flags[name])
    model.eval()
    torch.mps.empty_cache()

    extension = args.output / 'selected_extension'
    receipt = model.save_extension(extension)
    replacement = MiniCPM5Adapter(model.base, model.manifest).to('mps').eval()
    replacement.seal_original_identity()
    replacement.load_extension(extension)
    model = replacement
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    session._e2_prompt = lambda query: protocol['generation']['prompt'].replace(
        '{query}', query)
    session._encode = lambda text: encode_prompt(tokenizer, text, 'mps')
    session._mask = lambda ids: torch.ones_like(ids, dtype=torch.bool)
    report['extension'] = {
        'path': str(extension.resolve()), 'format': receipt['format'],
        'weights_sha256': receipt['weights_sha256'],
        'manifest_digest': receipt['manifest_digest'],
        'parameters': receipt['parameters'], 'base_saved': receipt['base_saved'],
        'bytes': sum(path.stat().st_size for path in extension.rglob('*')
                     if path.is_file())}
    require('extension_reload', receipt['base_saved'] is False)

    arms = [arm['id'] for arm in protocol['arms']]
    results = []
    for query_row in queries:
        results.extend(evaluate_case(session, tokenizer, query_row, arms,
                                     max_new_tokens=int(protocol['generation']['max_new_tokens'])))
        mark('retrieval_case', case=query_row['case_id'], split=query_row['split'],
             results=len(results))
    report['results'] = results
    report['metrics'] = summarize(results)
    report['eligibility'] = eligibility
    report['acceptance'] = {
        'engineering': all(check['passed'] for check in report['checks'].values()),
        'eligibility_violations': sum(len(item['forbidden_selected']) for item in results),
        'heldout_split': 'heldout',
        'oracle_diagnostic': 'not a retrieval arm; answers only isolate reader errors',
        'reranker_default': 'unchanged; opt-in pending held-out evidence',
        'claims': 'Exposed synthetic source groups; final held-out assessment is source-separated but not a broad generalization benchmark.'
    }
    # Explicit per-split held-out and correction diagnostics.
    report['diagnostics'] = {
        'heldout': [item for item in results if item['split'] == 'heldout'],
        'correction': [item for item in results if item['kind'] == 'correction'],
        'missing_information': [item for item in results
                                if item['kind'] == 'missing_information'],
        'long_record': [item for item in results
                        if item['kind'] == 'long_record_chunking'],
        'future_unauthorized_wrong_purpose': [
            item for item in results
            if item['kind'] == 'future_unauthorized_wrong_purpose'],
    }
    require('no_forbidden_neural_candidates',
            all(not item['forbidden_selected'] for item in results),
            violations=sum(len(item['forbidden_selected']) for item in results))
    require('long_record_chunking_exercised',
            any(item['span_count'] > 1 for item in results
                if item['kind'] == 'long_record_chunking'))
    report['costs'] = {'profiled_wall_seconds': None,
                       'sampled_peak_driver_bytes': None,
                       'sampled_peak_rss_bytes': None,
                       'new_original_checkpoints': 0, 'downloads': 0}
    ledger.close()
    mark('retrieval_complete', engineering=report['acceptance']['engineering'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'configs/e2_retrieval_qualification_v1.json')
    parser.add_argument('--model-path', type=Path,
                        default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path,
                        default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    parser.add_argument('--parent-extension', type=Path,
                        default=ROOT / 'runs/e1_e2_native_attempt02_20261023/selected_extension')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    if shutil.disk_usage(ROOT).free < 5 * 1024 ** 3:
        parser.error('At least 5 GiB disk headroom required')
    protocol = json.loads(args.protocol.read_text())
    if not Path(args.parent_extension).is_dir():
        parser.error('Parent extension is missing')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started',
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'arguments': {key: str(value) for key, value in vars(args).items()},
              'protocol_sha256': sha(args.protocol), 'protocol': protocol,
              'packages': {name: importlib.metadata.version(name)
                           for name in ('torch', 'transformers', 'safetensors')},
              'implementation_sha256': {
                  str(path.relative_to(ROOT)): sha(path)
                  for path in [Path(__file__),
                               *(ROOT / 'src/replica_lab/memory_system').glob('*.py'),
                               *(ROOT / 'src/replica_lab/architecture').glob('*.py')]},
              'free_disk_before_bytes': shutil.disk_usage(ROOT).free,
              'downloads': 0, 'original_model_loads': 1,
              'environment': {key: os.environ[key] for key in
                              ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                               'HF_DATASETS_OFFLINE', 'PYTORCH_ENABLE_MPS_FALLBACK')}}
    profiler = RunProfiler('mps', args.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            execute(args, profiler, report)
            report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        profile = profiler.stop()
        report['profile'] = profile
        memory = (profile or {}).get('memory') or {}
        report.setdefault('costs', {}).update({
            'profiled_wall_seconds': (profile or {}).get('elapsed_seconds'),
            'sampled_peak_driver_bytes': memory.get('mps_driver_peak_sampled_bytes'),
            'sampled_peak_rss_bytes': memory.get('rss_peak_sampled_bytes')})
        report['free_disk_after_bytes'] = shutil.disk_usage(ROOT).free
        report['artifact_bytes'] = sum(path.stat().st_size for path in args.output.rglob('*')
                                       if path.is_file())
        report['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        dump(args.output / ('run_report.json' if report['status'] == 'completed'
                            else 'failure_report.json'), report)
    print(json.dumps({'status': report['status'],
                      'engineering': report.get('acceptance', {}).get('engineering'),
                      'metrics': report.get('metrics')}, default=str))


if __name__ == '__main__':
    main()
