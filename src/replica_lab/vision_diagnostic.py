"""Frozen vision information probe and bounded overfit arm.

Decision question: does the *frozen* encoder representation contain enough
information for a linear readout to answer owned visual questions, and does it
add information beyond the question text? This separates three failure classes
before any connector change:

* visual features carry the answer and the connector/training path is the
  bottleneck;
* visual features carry nothing linear and the encoder/feature path is the
  bottleneck;
* neither features nor question text carry it and supervision/data pairing is
  the bottleneck.

All readout weights are closed-form ridge regressions over frozen features.
No model parameter changes. Oracle captions are never part of this probe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .vision_qualification import require

DIAGNOSTIC_FORMAT = 'replica.vision_diagnostic.protocol.v1'
REPORT_FORMAT = 'replica.vision_diagnostic.report.v1'
RIDGE_LAMBDA = 1.0
FEATURE_SETS = ('image_only', 'question_only', 'combined')
DIAGNOSTIC_IMPLEMENTATION_FILES = (
    'src/replica_lab/vision_diagnostic.py',
    'scripts/run_vision_diagnostic.py',
    'scripts/run_vision_qualification.py',
    'scripts/score_vision_qualification.py',
    'src/replica_lab/vision_qualification.py',
    'src/replica_lab/architecture/vision.py',
    'src/replica_lab/architecture/loading.py',
    'src/replica_lab/architecture/minicpm5.py',
    'src/replica_lab/architecture/contracts.py',
    'src/replica_lab/memory_system/session.py',
    'src/replica_lab/memory_system/ledger.py',
    'scripts/run_capacity_preflight.py',
    'src/replica_lab/capacity_guard.py',
    'src/replica_lab/vision_overfit.py',
    'src/replica_lab/profiling.py',
)
DEFAULT_DECISION = {'min_above_chance': 0.1, 'min_combined_gain': 0.05}


def _protocol_fields():
    return {'format', 'root', 'stage', 'vision', 'vision_digest', 'vision_source',
            'model', 'data', 'implementation_sha256', 'limits', 'seed',
            'train_worlds', 'dev_worlds', 'per_world_cases', 'decision'}


def diagnostic_workload(protocol):
    cases = (int(protocol['train_worlds']) + int(protocol['dev_worlds'])) \
        * int(protocol['per_world_cases'])
    return {'observations': cases, 'encoder_passes': cases,
            'optimizer_updates': 0, 'max_observations':
                protocol['limits']['max_observations']}


def validate_diagnostic_protocol(protocol, *, implementation_files=DIAGNOSTIC_IMPLEMENTATION_FILES):
    """Reject a diagnostic protocol that is unpinned, unbounded or mis-scoped."""
    require(isinstance(protocol, dict) and set(protocol) == _protocol_fields(),
            'Vision diagnostic protocol fields differ')
    require(protocol['format'] == DIAGNOSTIC_FORMAT,
            'Unknown vision diagnostic protocol format')
    require(protocol['stage'] == 'v1',
            'Vision diagnostic requires the v1 data stage')
    require(type(protocol['seed']) is int and protocol['seed'] >= 0,
            'Diagnostic seed must be a nonnegative integer')
    for name in ('train_worlds', 'dev_worlds', 'per_world_cases'):
        require(type(protocol[name]) is int and protocol[name] >= 1,
                f'{name} must be a positive integer')
    require(protocol['train_worlds'] * protocol['per_world_cases'] >= 8
            and protocol['dev_worlds'] * protocol['per_world_cases'] >= 8,
            'Diagnostic requires at least eight train and eight dev cases')
    decision = protocol['decision']
    require(isinstance(decision, dict) and set(decision) == set(DEFAULT_DECISION),
            'Diagnostic decision margins differ')
    for name, value in decision.items():
        require(type(value) in (int, float) and 0.0 <= value <= 1.0,
                f'Decision margin {name} must be in [0, 1]')
    limits = protocol['limits']
    require(isinstance(limits, dict) and set(limits) == {
        'max_observations', 'max_seconds', 'min_free_disk_bytes', 'max_rows'},
        'Diagnostic limits differ')
    for name in limits:
        require(type(limits[name]) is int and limits[name] >= 1,
                f'Diagnostic limit {name} must be a positive integer')
    workload = diagnostic_workload(protocol)
    require(workload['observations'] <= limits['max_observations'],
            'Diagnostic workload exceeds max_observations')
    pins = protocol['implementation_sha256']
    require(isinstance(pins, dict) and set(pins) == set(implementation_files),
            'Diagnostic implementation pins differ')
    for relative, digest in pins.items():
        require(isinstance(digest, str) and len(digest) == 64
                and all(char in '0123456789abcdef' for char in digest),
                f'Implementation pin is not a sha256: {relative}')
    return workload


def _ridge_fit(features: np.ndarray, targets: np.ndarray, *, penalty=RIDGE_LAMBDA):
    """Closed-form multi-output ridge with an unpenalised bias column."""
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    design = np.concatenate([(features - mean) / scale,
                             np.ones((features.shape[0], 1))], axis=1)
    gram = design.T @ design
    regulariser = np.eye(gram.shape[0]) * penalty
    regulariser[-1, -1] = 0.0
    weights = np.linalg.solve(gram + regulariser, design.T @ targets)
    return {'mean': mean, 'scale': scale, 'weights': weights}


def _ridge_predict(model, features, classes):
    design = np.concatenate([(features - model['mean']) / model['scale'],
                             np.ones((features.shape[0], 1))], axis=1)
    logits = design @ model['weights']
    return [classes[int(index)] for index in logits.argmax(axis=1)]


def _accuracy(predicted, expected):
    require(len(predicted) == len(expected) and bool(expected), 'Empty accuracy set')
    return sum(int(left == right) for left, right in zip(predicted, expected)) / len(expected)


def _one_hot(labels, classes):
    index = {name: position for position, name in enumerate(classes)}
    result = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for row, label in enumerate(labels):
        result[row, index[label]] = 1.0
    return result


def deterministic_worlds(dataset, split, *, limit):
    """Choose whole worlds deterministically; related cases stay together."""
    grouped = {}
    for row in dataset.rows_for_split(split):
        grouped.setdefault(dataset.case(row['case_id'])['group_id'], []).append(row)
    worlds = sorted(grouped)[:limit]
    require(bool(worlds), f'No cases for split {split}')
    selected = []
    for world in worlds:
        rows = sorted(grouped[world], key=lambda item: item['case_id'])
        selected.append({'group_id': world, 'rows': rows})
    return selected


def select_cases(dataset, split, *, limit, per_world=2):
    """One case per distinct task kind per world, capped deterministically."""
    chosen = []
    for world in deterministic_worlds(dataset, split, limit=limit):
        seen = set()
        for row in world['rows']:
            case = dataset.case(row['case_id'])
            if case['task'] in seen:
                continue
            seen.add(case['task'])
            chosen.append({'case_id': row['case_id'], 'row': row, 'case': case,
                           'group_id': world['group_id']})
            if len(seen) >= per_world:
                break
    require(bool(chosen), f'No selected cases for {split}')
    return chosen


def image_feature(service, dataset, case_id):
    """Pooled frozen spatial patches for one case image."""
    visual = _observe(service, dataset, case_id)
    patches = visual.features[0].detach().float().cpu().numpy()
    require(patches.shape[0] == service.spec.max_visual_tokens,
            'Visual feature token count differs from the pinned layout')
    require(bool(visual.valid.all()), 'Visual mask has invalid positions')
    return patches.mean(axis=0), patches


def _observe(service, dataset, case_id):
    from .memory_system.ledger import EligibilityScope
    scope = EligibilityScope('vision-diagnostic', 'frozen-readout', 'diagnostic', 0,
                             'parallel_replica', None)
    return service.observe(str(dataset.image_path(case_id)),
                           source_id='vision-diagnostic', task_id=case_id,
                           scope=scope, observed_at=0, source_epoch=0)


def question_embedding(model, tokenizer, question):
    """Frozen decoder final hidden state for the rendered question prompt."""
    from .architecture.loading import encode_prompt
    device = model.base.get_input_embeddings().weight.device
    ids = encode_prompt(tokenizer, question, device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        output = model.base(input_ids=ids, attention_mask=mask,
                            output_hidden_states=True, use_cache=False)
    hidden = output.hidden_states[-1]
    require(hidden is not None and hidden.shape[1] == ids.shape[1],
            'Frozen question hidden state is unavailable')
    return hidden[0, -1].detach().float().cpu().numpy()


def collect_features(model, tokenizer, service, dataset, cases, *, split):
    records = []
    for item in cases:
        image, patches = image_feature(service, dataset, item['case_id'])
        question = question_embedding(model, tokenizer, item['row']['question'])
        label = dataset.label(item['case_id'])
        records.append({
            'case_id': item['case_id'], 'world': item['group_id'],
            'split': split, 'task': item['case']['task'], 'label': label,
            'image': image, 'question': question, 'patches': patches,
        })
    return records


def feature_statistics(records):
    images = np.stack([item['image'] for item in records])
    patches = np.stack([item['patches'] for item in records])
    norms = np.linalg.norm(patches, axis=2)
    same, cross = [], []
    for index, left in enumerate(patches):
        flat = left.reshape(-1)
        same.append(float(np.dot(flat, flat) / (np.linalg.norm(flat) ** 2)))
        other = patches[(index + 1) % len(patches)].reshape(-1)
        denominator = np.linalg.norm(flat) * np.linalg.norm(other)
        cross.append(float(np.dot(flat, other) / denominator) if denominator else 0.0)
    return {
        'cases': len(records),
        'image_dim': int(images.shape[1]),
        'patch_count': int(patches.shape[1]),
        'image_mean': float(images.mean()), 'image_std': float(images.std()),
        'image_norm_mean': float(np.linalg.norm(images, axis=1).mean()),
        'patch_norm_by_position_min': float(norms.mean(axis=0).min()),
        'patch_norm_by_position_max': float(norms.mean(axis=0).max()),
        'patch_self_dot': float(np.mean(same)),
        'patch_adjacent_case_cosine_mean': float(np.mean(cross)),
        'labels_by_task': {task: sorted({item['label'] for item in records
                                         if item['task'] == task})
                           for task in sorted({item['task'] for item in records})},
    }


def run_readout_probe(records, *, seed=17):
    """Fit ridge readouts on train worlds and score fresh dev worlds."""
    train = [item for item in records if item['split'] == 'train']
    dev = [item for item in records if item['split'] == 'dev']
    require(len(train) >= 4 and len(dev) >= 4, 'Too few diagnostic cases')
    classes = sorted({item['label'] for item in train})
    require(len(classes) >= 2, 'Readout needs at least two train label classes')
    dev_labels = [item['label'] for item in dev]
    targets = _one_hot([item['label'] for item in train], classes)
    features = {
        'image_only': np.stack([item['image'] for item in train]),
        'question_only': np.stack([item['question'] for item in train]),
        'combined': np.stack([np.concatenate([item['image'], item['question']])
                              for item in train]),
    }
    dev_features = {
        'image_only': np.stack([item['image'] for item in dev]),
        'question_only': np.stack([item['question'] for item in dev]),
        'combined': np.stack([np.concatenate([item['image'], item['question']])
                              for item in dev]),
    }
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(len(dev))
    shuffled_combined = np.stack([
        np.concatenate([dev_features['image_only'][shuffled_index[index]],
                        dev_features['question_only'][index]])
        for index in range(len(dev))])
    shuffled_image = dev_features['image_only'][shuffled_index]
    results = {}
    predictions = {}
    for name in FEATURE_SETS:
        model = _ridge_fit(features[name], targets)
        train_predictions = _ridge_predict(model, features[name], classes)
        dev_predictions = _ridge_predict(model, dev_features[name], classes)
        predictions[name] = dev_predictions
        results[name] = {
            'train_accuracy': _accuracy(train_predictions,
                                        [item['label'] for item in train]),
            'dev_accuracy': _accuracy(dev_predictions, dev_labels),
            'dev_by_task': {
                task: _accuracy([predicted for predicted, item
                                 in zip(dev_predictions, dev) if item['task'] == task],
                                [item['label'] for item in dev if item['task'] == task])
                for task in sorted({item['task'] for item in dev})},
        }
    image_shuffled = _ridge_predict(_ridge_fit(features['image_only'], targets),
                                    shuffled_image, classes)
    combined_shuffled = _ridge_predict(_ridge_fit(features['combined'], targets),
                                       shuffled_combined, classes)
    results['image_only']['dev_accuracy_shuffled_images'] = _accuracy(
        image_shuffled, dev_labels)
    results['combined']['dev_accuracy_shuffled_images'] = _accuracy(
        combined_shuffled, dev_labels)
    majority = max(set(dev_labels), key=dev_labels.count)
    results['baselines'] = {
        'dev_majority_accuracy': _accuracy([majority] * len(dev), dev_labels),
        'dev_class_count': len(set(dev_labels)),
        'dev_random_chance': 1.0 / len(set(dev_labels)),
        'train_label_count': len(classes),
    }
    return results, predictions


def decide(results, statistics, *, min_above_chance=0.1, min_combined_gain=0.05):
    """Map readout evidence to one justified next change."""
    combined = results['combined']['dev_accuracy']
    image = results['image_only']['dev_accuracy']
    question = results['question_only']['dev_accuracy']
    chance = results['baselines']['dev_random_chance']
    shuffled = results['combined']['dev_accuracy_shuffled_images']
    if combined <= chance + min_above_chance and question <= chance + min_above_chance:
        decision = 'fail_supervision_or_pairing'
        change = ('Audit answer supervision, question/image pairing and label '
                  'vocabulary before any model change.')
    elif image <= chance + min_above_chance and combined <= question + min_combined_gain:
        decision = 'fail_encoder_features_carry_no_linear_signal'
        change = ('Image features add no linear signal beyond question text; '
                  'investigate encoder preprocessing/features/positions rather '
                  'than enlarging the connector.')
    elif combined <= chance + min_above_chance:
        decision = 'fail_features_insufficient'
        change = 'Neither stream separates answers; treat as a data/supervision blocker.'
    elif (combined <= max(image, question) + min_combined_gain
          or combined <= shuffled + min_combined_gain):
        decision = 'inconclusive_visual_contribution'
        change = ('Image features lack the declared incremental gain over the best '
                  'single stream or shuffled-image control. Build a balanced paired '
                  'fresh-world diagnostic before changing the connector.')
    else:
        decision = 'frozen_features_support_readout'
        change = ('A linear readout shows incremental image-dependent signal beyond '
                  'the best single stream and shuffled control. The connector/training '
                  'path is a candidate bottleneck, not a proven one; confirm on a '
                  'balanced fresh-world diagnostic before a bounded connector retry.')
    return {'decision': decision, 'justified_next_change': change,
            'image_minus_question': image - question,
            'combined_minus_best_single': combined - max(image, question),
            'shuffled_image_drop': combined - shuffled,
            'chance': chance}


def run_diagnostic(model, tokenizer, service, dataset, protocol, *, output):
    workload = validate_diagnostic_protocol(protocol)
    train_cases = select_cases(dataset, 'train', limit=protocol['train_worlds'],
                               per_world=protocol['per_world_cases'])
    dev_cases = select_cases(dataset, 'dev', limit=protocol['dev_worlds'],
                             per_world=protocol['per_world_cases'])
    require({case['group_id'] for case in train_cases}.isdisjoint(
        {case['group_id'] for case in dev_cases}),
        'Diagnostic train and dev worlds overlap')
    train_records = collect_features(model, tokenizer, service, dataset,
                                     train_cases, split='train')
    dev_records = collect_features(model, tokenizer, service, dataset,
                                   dev_cases, split='dev')
    records = train_records + dev_records
    pair_mismatches = [item['case_id'] for item in
                       (train_cases + dev_cases)
                       if item['row']['asset_id'] != item['case']['asset_id']]
    require(not pair_mismatches,
            f'Question/image pairing differs from verified cases: {pair_mismatches[:3]}')
    statistics = feature_statistics(records)
    results, predictions = run_readout_probe(records, seed=protocol['seed'])
    verdict = decide(results, statistics,
                     **{name: protocol['decision'][name] for name in DEFAULT_DECISION})
    report = {
        'format': REPORT_FORMAT, 'stage': 'frozen_readout',
        'seed': protocol['seed'],
        'workload': workload,
        'train_worlds': protocol['train_worlds'],
        'dev_worlds': protocol['dev_worlds'],
        'train_cases': len(train_records), 'dev_cases': len(dev_records),
        'feature_statistics': statistics,
        'readout': results,
        'verdict': verdict,
        'oracle_caption_used': False,
        'pairing_mismatches': pair_mismatches,
        'optimizer_updates': 0,
        'parameters_updated': [],
        'quality_claim': ('Frozen-feature information probe only; no perception, '
                          'training or deployment claim'),
        'train_world_groups': sorted({item['world'] for item in train_records}),
        'dev_world_groups': sorted({item['world'] for item in dev_records}),
    }
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
