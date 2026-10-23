#!/usr/bin/env python3
"""Post-hoc E2 answer-path diagnostic on held-out cases (no training).

Compares the trained selected/reranker extension under correct record slots, a
different eligible record's slots, no slots, and exact source text in the
prompt. This isolates reader/writer versus prompt/decoder behavior; it is not a
retrieval arm and does not change any default.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
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
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.contracts import TaskGraph
    from replica_lab.architecture.loading import encode_prompt, load_local_mps
    from replica_lab.memory_system import EligibilityScope, MemoryRecord, RecordLedger
    from replica_lab.memory_system.index import RetrievalSelection
    from replica_lab.memory_system.session import MemorySession

    protocol = json.loads(args.protocol.read_text())
    data_root = ROOT / protocol['dataset']['root']
    if sha(data_root / 'records.jsonl') != protocol['dataset']['records_sha256']:
        raise ValueError('Diagnostic records differ from preregistered hash')
    records = [MemoryRecord(**row) for row in load_rows(data_root / 'records.jsonl')]
    by_id = {record.record_id: record for record in records}
    queries = {row['case_id']: row for row in load_rows(data_root / 'queries.jsonl')}
    model, tokenizer, lock = load_local_mps(
        args.model_path, args.model_lock, profiler=profiler)
    receipt = model.load_extension(Path(args.extension))
    report['extension'] = {'path': str(Path(args.extension).resolve()),
                           'weights_sha256': receipt['weights_sha256'],
                           'format': receipt['format']}
    ledger = RecordLedger(args.output / 'diagnostic_ledger.sqlite')
    for record in records:
        ledger.admit(record, authorized_subject=record.subject_id,
                     authorized_branch=record.branch_id, source_authorized=False)
    session = MemorySession(model, tokenizer, ledger, max_candidate_units=64)
    template = protocol['generation']['prompt']
    results = []
    for case_id in protocol['cases']:
        query_row = queries[case_id]
        scope = EligibilityScope(query_row['subject_id'], query_row['branch_id'],
                                 query_row['purpose'], query_row['as_of'],
                                 'assistant_for_source', None)
        snapshot = ledger.snapshot(scope)
        expected = query_row['expected_record_id']
        correct = next((record for record in snapshot.records
                        if record.record_id == expected), None)
        swap_id = protocol['swaps'][case_id]
        swapped = next((record for record in snapshot.records
                        if record.record_id == swap_id), None)
        if correct is None or swapped is None:
            raise ValueError(f'{case_id}: expected/swap record is not eligible')
        selections = {
            'correct_slots': RetrievalSelection(
                (correct,), (1.0,), snapshot.epoch, snapshot.ledger_id, scope,
                query_row['query']),
            'swapped_slots': RetrievalSelection(
                (swapped,), (1.0,), snapshot.epoch, snapshot.ledger_id, scope,
                query_row['query']),
            'no_slots': RetrievalSelection(
                (), (), snapshot.epoch, snapshot.ledger_id, scope, query_row['query']),
            'exact_text': RetrievalSelection(
                (), (), snapshot.epoch, snapshot.ledger_id, scope, query_row['query']),
        }
        for condition in protocol['conditions']:
            prompt = template.replace('{query}', query_row['query'])
            if condition == 'exact_text':
                prompt = protocol['generation']['exact_text_prefix'].replace(
                    '{text}', correct.text) + prompt
            ids = encode_prompt(tokenizer, prompt, 'mps')
            mask = torch.ones_like(ids, dtype=torch.bool)
            request = session.prepare(
                prompt_ids=ids, prompt_mask=mask, query=query_row['query'],
                scope=scope, task=TaskGraph(case_id, query_row['query']),
                selection=selections[condition])
            try:
                generated = session.generate(
                    request, max_new_tokens=int(protocol['generation']['max_new_tokens']))
                prompt_tokens = int(request.frame.sense.prompt_ids.shape[1])
                new_ids = [int(token) for token in generated[0, prompt_tokens:].tolist()]
                answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                results.append({
                    'case_id': case_id, 'condition': condition,
                    'expected_answer': query_row['expected_answer'],
                    'answer': answer,
                    'answer_correct': answer.casefold()
                                      == str(query_row['expected_answer']).casefold(),
                    'selected_records': [record.record_id for record in
                                         selections[condition].records],
                    'prompt_tokens': prompt_tokens,
                    'generated_token_ids': new_ids,
                    'prompt_text': prompt,
                })
            finally:
                request.release()
    report['results'] = results
    report['diagnosis'] = {
        'correct_slots_vs_no_slots': [
            (a['answer'], b['answer']) for a, b in zip(results[0::4], results[2::4])],
        'correct_slots_vs_swapped_slots': [
            (a['answer'], b['answer']) for a, b in zip(results[0::4], results[1::4])],
        'exact_text': [item['answer'] for item in results[3::4]],
        'note': 'Post-hoc diagnostic; not a retrieval arm and not used to change defaults.'}
    report['status'] = 'completed'
    ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'configs/e2_retrieval_diagnostic_v1.json')
    parser.add_argument('--model-path', type=Path,
                        default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path,
                        default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    parser.add_argument('--extension', type=Path,
                        default=ROOT / 'runs/e2_retrieval_20261023/selected_extension')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started',
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha(args.protocol),
              'protocol': json.loads(args.protocol.read_text()),
              'downloads': 0, 'optimizer_updates': 0, 'checkpoints_saved': 0}
    profiler = RunProfiler('mps', args.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            execute(args, profiler, report)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        profile = profiler.stop()
        report['profile'] = profile
        memory = (profile or {}).get('memory') or {}
        report['costs'] = {'profiled_wall_seconds': (profile or {}).get('elapsed_seconds'),
                           'sampled_peak_driver_bytes': memory.get(
                               'mps_driver_peak_sampled_bytes')}
        report['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        dump(args.output / ('run_report.json' if report['status'] == 'completed'
                            else 'failure_report.json'), report)
    print(json.dumps({'status': report['status'],
                      'diagnosis': report.get('diagnosis')}, default=str))


if __name__ == '__main__':
    main()
