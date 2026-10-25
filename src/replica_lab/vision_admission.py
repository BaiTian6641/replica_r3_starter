"""Named admission of verified owned visual rows; no downloads or model execution."""
from __future__ import annotations

import calendar
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import random

from .vision_qualification import (
    ADMISSION_FORMAT, CONTROLS, RETENTION_LABELS_FORMAT, RETENTION_PROMPTS_FORMAT,
    TEXT_CONTEXTS_FORMAT, TRAINING_ROWS_FORMAT, VISUAL_QUESTION_TEMPLATE,
    canonical_json, ensure_fresh_directory, load_verified_dataset, require,
    rows_sha256, select_eval_cases, sha256_file, sha256_json, sha256_text,
    verify_admission, verify_v0_prerequisite, visual_question,
)

DATA_CEILING_BYTES = 25 * 1024**3


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _jsonl(path, rows):
    Path(path).write_text(''.join(canonical_json(row) + '\n' for row in rows))


def _pin(root, path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            'sha256': sha256_file(path)}


def retention_cases():
    """Thirty-two protected text cases across seven independently defined domains."""
    items = []
    for index, (a, b, op) in enumerate(((7, 9, '+'), (18, 7, '-'), (6, 8, '*'),
                                      (84, 7, '/'), (13, 19, '+'), (91, 38, '-'),
                                      (12, 7, '*'), (156, 12, '/'))):
        answer = {'+': lambda: a + b, '-': lambda: a - b,
                  '*': lambda: a * b, '/': lambda: a // b}[op]()
        items.append(('arithmetic', f'Calculate {a} {op} {b}.', str(answer)))
    for text in ('darel', 'mivor', 'toben', 'sulak'):
        items.append(('strings', f'Reverse the letters in "{text}".', text[::-1]))
    for a, b, op in ((True, False, 'AND'), (True, False, 'OR'),
                     (False, False, 'OR'), (True, True, 'AND')):
        answer = (a and b) if op == 'AND' else (a or b)
        left = '(2 < 7)' if a else '(7 < 2)'
        right = '(3 < 9)' if b else '(9 < 3)'
        items.append(('logic', f'Evaluate the Boolean expression {left} {op} {right}.',
                      str(answer).lower()))
    for amount, unit, target, multiplier in ((3, 'kilometers', 'meters', 1000),
                                             (7, 'liters', 'milliliters', 1000),
                                             (4, 'hours', 'minutes', 60),
                                             (9, 'weeks', 'days', 7)):
        items.append(('units', f'Convert {amount} {unit} to {target}.', str(amount * multiplier)))
    for year, month, day in ((2024, 2, 29), (2025, 1, 1), (2026, 10, 25), (2030, 6, 15)):
        value = date(year, month, day)
        items.append(('calendar', f'Which weekday is {value.isoformat()}?', calendar.day_name[value.weekday()]))
    for singular, plural in (('child', 'children'), ('mouse', 'mice'), ('goose', 'geese'), ('tooth', 'teeth')):
        items.append(('language', f'Give the standard English plural of "{singular}".', plural))
    for key, value, other in (('amber', 17, 29), ('birch', 23, 41), ('cedar', 31, 53), ('dune', 47, 61)):
        items.append(('structured_text', 'Add the two numerical values in this JSON object: '
                      + json.dumps({key: value, 'other': other}, separators=(',', ':')),
                      str(value + other)))
    prompts, labels = [], []
    for index, (family, question, answer) in enumerate(items):
        case_id = f'retention-{family}-{index:02d}'
        prompts.append({'case_id': case_id,
                        'prompt': VISUAL_QUESTION_TEMPLATE.format(question=question),
                        'family': f'owned_text_{family}', 'group_id': case_id, 'split': 'dev'})
        labels.append({'case_id': case_id, 'answer': answer})
    return prompts, labels


def admit_owned_visual_rows(*, project_root, v0_protocol_path, v0_score_path,
                            output, run_id, seed=17, train_families=64, eval_cases=64):
    """Create a real admission only after a pinned independently accepted V0."""
    root = Path(project_root).resolve()
    require(type(seed) is int and seed >= 0, 'Sampling seed must be an integer')
    require(type(train_families) is int and 1 <= train_families <= 224,
            'Training-family cap is outside the owned source tranche')
    require(type(eval_cases) is int and 2 <= eval_cases <= 256, 'Evaluation cap is invalid')
    require(isinstance(run_id, str) and run_id.strip(), 'A named run is required')
    output = ensure_fresh_directory(output)
    native = json.loads(Path(v0_protocol_path).read_text())
    score_path = Path(v0_score_path).resolve()
    score = json.loads(score_path.read_text())
    native_run = json.loads(Path(score['run_report_path']).read_text())
    extension = native_run['extension']
    initial = Path(extension['path'])
    if not initial.is_absolute():
        initial = root / initial
    context = deepcopy(native)
    context.update(root=str(root), stage='v1')
    context['limits']['max_rows'] = 4096
    data_root = root / 'data/vision_preparation_v1'
    context['data'] = {'root': str(data_root), 'stage': 'v1'}
    for key, filename in (('manifest', 'vision_manifest.json'), ('model_rows', 'model_rows.jsonl'),
                          ('verifier_cases', 'verifier_cases.jsonl'), ('verifier_scenes', 'verifier_scenes.jsonl')):
        path = data_root / 'v1' / filename
        context['data'][key] = {'path': f'v1/{filename}', 'sha256': sha256_file(path)}
    context['v1'] = {
        'run_id': run_id, 'question_template': VISUAL_QUESTION_TEMPLATE,
        'prerequisites': {'v0_protocol_sha256': sha256_file(v0_protocol_path),
                          'v0_score_report': _pin(root, score_path)},
        'initial_checkpoint': {'path': str(initial),
                               'manifest_sha256': score['extension']['manifest_sha256'],
                               'weights_sha256': score['extension']['weights_sha256']},
        'eval': {'splits': ['dev'], 'seed': seed, 'max_cases': eval_cases,
                 'required_controls': list(CONTROLS)},
    }
    verify_v0_prerequisite(context)
    dataset = load_verified_dataset(context)
    grouped = {}
    for row in dataset.rows_for_split('train'):
        grouped.setdefault(dataset.case(row['case_id'])['group_id'], []).append(row)
    families = sorted(grouped)
    random.Random(seed).shuffle(families)
    chosen = set(families[:train_families])
    selected = sorted((row for family in chosen for row in grouped[family]),
                      key=lambda row: row['case_id'])
    training = []
    for row in selected:
        case = dataset.case(row['case_id'])
        training.append({'case_id': row['case_id'],
                         'prompt': visual_question(context, row['question']),
                         'response': dataset.label(row['case_id']),
                         'image': row['image'], 'image_sha256': row['image_sha256'],
                         'family': case['group_id'], 'group_id': case['group_id'], 'split': 'train'})
    evaluation = select_eval_cases(dataset, context['v1']['eval'])
    contexts = []
    for row in evaluation:
        case = dataset.case(row['case_id'])
        scene = dataset.scenes[case['asset_id']]['scene']
        objects = sorted(scene['objects'], key=lambda obj: obj['id'])
        description = '; '.join(f"{obj['id']}: {obj['color']} {obj['shape']} centered at "
                                f"({obj['center'][0]}, {obj['center'][1]})" for obj in objects)
        for kind, text, information in (
                ('caption_only', 'Oracle scene description: ' + description + '.', 'oracle_upper_bound'),
                ('ocr_text_only', 'The owned geometric renderer contains no text.', 'independent')):
            contexts.append({'case_id': row['case_id'], 'kind': kind, 'text': text,
                             'text_sha256': sha256_text(text),
                             'source_id': f'owned_scene_verifier:{case["scene_sha256"]}',
                             'information_bound': information})
    prompts, labels = retention_cases()
    files = {'training_rows': (output / 'training_rows.jsonl', training, TRAINING_ROWS_FORMAT),
             'text_contexts': (output / 'text_contexts.jsonl', contexts, TEXT_CONTEXTS_FORMAT),
             'retention_prompts': (output / 'retention_prompts.jsonl', prompts, RETENTION_PROMPTS_FORMAT),
             'retention_labels': (output / 'retention_labels.jsonl', labels, RETENTION_LABELS_FORMAT)}
    for path, rows, _ in files.values():
        _jsonl(path, rows)
    context['v1']['retention'] = {
        'prompts': _pin(root, files['retention_prompts'][0]),
        'labels': _pin(root, files['retention_labels'][0]), 'splits': ['dev'], 'max_cases': len(prompts)}
    splits = {}
    for split in ('train', 'dev'):
        rows = [dict(row) for row in dataset.rows_for_split(split)]
        splits[split] = {'model_rows_sha256': rows_sha256(rows), 'model_row_count': len(rows),
                         'family_count': len({dataset.case(row['case_id'])['group_id'] for row in rows}),
                         'row_ids_sha256': sha256_json(sorted(row['case_id'] for row in rows))}
    document = {
        'format': ADMISSION_FORMAT, 'stage': 'v1', 'run_id': run_id, 'admitted': True,
        'eligible_rows': sum(item['model_row_count'] for item in splits.values()),
        'admitted_training_rows': len(training),
        'data': dict(dataset.pins), 'splits': splits,
        'sampling_seed': seed, 'admitted_training_families': sorted(chosen),
        'independent_training_worlds': len(chosen),
        'source_families': ['owned_procedural_scene_png_v1'],
        'label_status': 'owned executable scene verifier; responses independently reconstructed',
        'benchmark_training_rows': 0, 'external_training_rows': 0,
        'v0_score_report_sha256': sha256_file(score_path),
        'training_rows': {**_pin(root, files['training_rows'][0]), 'format': TRAINING_ROWS_FORMAT,
                          'rows_sha256': rows_sha256(training), 'count': len(training)},
        'text_contexts': {**_pin(root, files['text_contexts'][0]), 'format': TEXT_CONTEXTS_FORMAT,
                          'rows_sha256': rows_sha256(contexts)},
        'oracle_control': 'caption_only receives scene metadata explicitly as an oracle upper bound; never training/sensing input outside that control',
        'ocr_control': 'independently verified absence of rendered text; not a teacher/OCR-model prediction',
    }
    admission_path = output / 'admission.json'
    pending = output / 'admission.pending.json'
    _json(pending, document)
    context['v1']['prerequisites']['data_admission'] = _pin(root, pending)
    verify_admission(context, dataset, train_splits=['train'], eval_splits=['dev'])
    logical_bytes = sum(path.stat().st_size for path in data_root.rglob('*') if path.is_file())
    logical_bytes += sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    require(logical_bytes <= DATA_CEILING_BYTES, 'Named visual tranche exceeds the 25 GiB data ceiling')
    pending.replace(admission_path)
    context['v1']['prerequisites']['data_admission'] = _pin(root, admission_path)
    report = {
        'format': 'replica.vision_admission_report.v1', 'run_id': run_id,
        'admission': _pin(root, admission_path), 'data': context['data'],
        'training_rows': document['training_rows'], 'text_contexts': document['text_contexts'],
        'retention': context['v1']['retention'], 'v0_prerequisites': context['v1']['prerequisites'],
        'initial_checkpoint': context['v1']['initial_checkpoint'],
        'eligible_rows': document['eligible_rows'], 'admitted_training_rows': len(training),
        'verified_training_assets': len({row['image_sha256'] for row in training}),
        'independent_training_worlds': len(chosen),
        'independent_evaluation_worlds': len({dataset.case(row['case_id'])['group_id'] for row in evaluation}),
        'evaluation_cases': len(evaluation), 'source_family_count': 1,
        'retention_cases': len(prompts), 'retention_domains': len({row['family'] for row in prompts}),
        'logical_dataset_bytes': logical_bytes, 'dataset_disk_ceiling_bytes': DATA_CEILING_BYTES,
        'external_assets_acquired': 0, 'native_execution': False, 'optimizer_updates': 0,
        'preparation_manifests_changed': False,
    }
    _json(output / 'admission_report.json', report)
    return report
