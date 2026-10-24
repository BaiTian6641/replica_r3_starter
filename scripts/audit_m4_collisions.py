#!/usr/bin/env python3
"""Audit distinct-n-gram collisions for a preserved M4 run on CPU only.

This is a post-hoc evidence audit, not a training or generation arm. It reuses
PatternAccessPort.hash_indices and the sealed tokenizer, and never loads weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from run_m2_dynamic import DELAYED_USE_TEMPLATE, rows, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-report', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh collision-audit output path')
    protocol = json.loads(args.protocol.read_text())
    run = json.loads(args.run_report.read_text())
    if run.get('status') != 'completed' or run.get('protocol_sha256') != sha(args.protocol):
        raise ValueError('Collision audit requires a completed matching M4 run')
    data = ROOT / protocol['dataset']['root']
    for filename, key in (('manifest.json', 'manifest_sha256'),
                          ('records.jsonl', 'records_sha256'), ('queries.jsonl', 'queries_sha256')):
        if sha(data / filename) != protocol['dataset'][key]:
            raise ValueError(f'M4 collision-audit dataset changed: {filename}')
    from transformers import AutoTokenizer
    import torch
    from replica_lab.architecture.ports import PatternAccessPort
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                              trust_remote_code=False)
    spec = {key: value for key, value in protocol['pattern']['spec'].items() if key != 'injection_layer'}
    port = PatternAccessPort(hidden_size=1, **spec).eval()
    all_queries = rows(data / 'queries.jsonl')
    queries = [query for query in all_queries
               if query['split'] == 'train' and query.get('case_role') == 'delayed_training']
    if len(queries) != protocol['splits']['train']:
        raise ValueError('M4 collision-audit training split differs')
    eval_queries = [query for query in all_queries if query['split'] in {'dev', 'heldout', 'regression'}]
    replay = [query for query in all_queries if query['split'] == 'retention']
    retention_dev = [query for query in all_queries if query['split'] == 'retention_dev']
    restart_query = next(query for query in all_queries if query['split'] == 'heldout')
    def collect(query_list):
        counts, ngrams = {}, {}
        with torch.no_grad():
            for query in query_list:
                text = DELAYED_USE_TEMPLATE.replace('{query}', query['query']).replace('{observation}', '')
                rendered = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': text}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                token_ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
                ids = torch.tensor([token_ids], dtype=torch.long)
                mask = torch.ones_like(ids, dtype=torch.bool)
                for order, head, indices, available in port.hash_indices(ids, mask):
                    for position in range(ids.shape[1]):
                        if not bool(available[0, position]):
                            continue
                        index = int(indices[0, position])
                        key = (order, head, index)
                        ngram = tuple(int(value) for value in ids[0, position - order + 1:position + 1])
                        counts[key] = counts.get(key, 0) + 1
                        ngrams.setdefault(key, set()).add(ngram)
        return counts, ngrams
    training_counts, training_ngrams = collect(queries)
    router_queries = queries + eval_queries + replay + retention_dev + [restart_query]
    counts, ngrams = collect(router_queries)
    trained = {tuple(item) for item in run['trained_hash_ids']}
    if trained != set(training_counts):
        raise ValueError('Collision-audit training hash inventory differs from the preserved run')
    def summarize(counts, ngrams):
        result = {}
        for order in spec['orders']:
            for head in range(spec['heads']):
                keys = [key for key in counts if key[0] == order and key[1] == head]
                lookups = sum(counts[key] for key in keys)
                distinct = sum(len(ngrams[key]) for key in keys)
                colliding = [key for key in keys if len(ngrams[key]) > 1]
                collision_ngrams = sum(len(ngrams[key]) - 1 for key in colliding)
                repeated = lookups - distinct
                examples = []
                for key in sorted(colliding)[:4]:
                    examples.append({'order': key[0], 'head': key[1], 'index': key[2],
                                     'ngrams': [list(value) for value in sorted(ngrams[key])]})
                result[f'{order}_{head}'] = {
                    'lookups': lookups,
                    'unique_indices': len(keys),
                    'distinct_ngrams': distinct,
                    'repeated_lookups': repeated,
                    'repeat_rate': repeated / lookups if lookups else 0.0,
                    'colliding_indices': len(colliding),
                    'collision_ngrams': collision_ngrams,
                    'collision_rate': collision_ngrams / distinct if distinct else 0.0,
                    'table_rows': spec['table_rows'],
                    'hash_coverage': len(keys) / spec['table_rows'],
                    'touched_bytes': len(keys) * spec['width'] * 4,
                    'collision_examples': examples,
                }
        return result
    result = summarize(counts, ngrams)
    training_result = summarize(training_counts, training_ngrams)
    legacy = run.get('collisions', {})
    for key, value in result.items():
        if key not in legacy:
            continue
        if 'collision_ngrams' in legacy[key]:
            for field in ('lookups', 'unique_indices', 'distinct_ngrams', 'repeated_lookups',
                          'repeat_rate', 'colliding_indices', 'collision_ngrams', 'collision_rate'):
                if legacy[key].get(field) != value[field]:
                    raise ValueError(f'Runner collision field differs for {key}: {field}')
            value['runner_fields_validated'] = True
        else:
            if (legacy[key].get('lookups') != value['lookups']
                    or legacy[key].get('unique_indices') != value['unique_indices']):
                raise ValueError(f'Router hash accounting differs for {key}: '
                                 f'runner={(legacy[key].get("lookups"), legacy[key].get("unique_indices"))} '
                                 f'audit={(value["lookups"], value["unique_indices"])}')
            value['legacy_runner_fields'] = {
                'collisions': legacy[key].get('collisions'),
                'collision_rate': legacy[key].get('collision_rate'),
                'interpretation': 'Legacy runner subtracted unique indices from lookups; this includes repeated same-n-gram lookups and is not a distinct-n-gram collision count.'}
    training_legacy = run.get('training_collisions', {})
    for key, value in training_result.items():
        if key in training_legacy:
            for field in ('lookups', 'unique_indices', 'distinct_ngrams', 'repeated_lookups',
                          'repeat_rate', 'colliding_indices', 'collision_ngrams', 'collision_rate'):
                if training_legacy[key].get(field) != value[field]:
                    raise ValueError(f'Runner training collision field differs for {key}: {field}')
            value['runner_fields_validated'] = True
    output = {
        'status': 'completed',
        'audit_type': 'distinct_ngram_collision_audit',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': str(args.protocol),
        'protocol_sha256': sha(args.protocol),
        'run_report': str(args.run_report),
        'run_report_sha256': sha(args.run_report),
        'audit_script_sha256': sha(Path(__file__)),
        'device': 'cpu_tokenization_and_hash_only',
        'model_weights_loaded': False,
        'native_training_or_generation': False,
        'trained_hash_count': len(trained),
        'collision_definition': 'For each order/head/table index, collect distinct causal token n-grams. collision_ngrams counts distinct n-grams beyond the first at a shared index; collision_rate divides that count by all distinct n-grams. Repeated occurrences of one n-gram are reported separately.',
        'collisions': result,
        'training_collisions': training_result,
        'scopes': {
            'collisions': 'All hash calls made by initial training inventory, candidate/general/development router evaluation, and the same-process restart route.',
            'training_collisions': 'Only the 24 delayed-training prompts used to construct the static table inventory.',
            'router_query_count_including_training': len(router_queries),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'trained_hash_count': len(trained),
                      'collision_ngrams': sum(value['collision_ngrams'] for value in result.values()),
                      'training_collision_ngrams': sum(value['collision_ngrams'] for value in training_result.values()),
                      'distinct_ngrams': sum(value['distinct_ngrams'] for value in result.values())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
