"""Real decoder backward/update through native-runner helpers on tiny CPU fixtures.

Precomputed patches qualify the runner boundary, not the pretrained image encoder.
"""
from pathlib import Path
import hashlib
import importlib.util

import pytest
import torch

from test_visual_adapter import SPEC, SCOPE, observed, visual_session
from replica_lab.vision_qualification import training_schedule, validate_optimizer_route

ROOT = Path(__file__).resolve().parents[1]
_module = importlib.util.spec_from_file_location(
    'visual_runner_contract', ROOT / 'scripts' / 'run_vision_qualification.py')
runner = importlib.util.module_from_spec(_module)
_module.loader.exec_module(runner)


class PatchDataset:
    def __init__(self, root):
        self.path = root / 'observed.patches'
        self.path.write_bytes(bytes(range(32)))
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def asset_for_case(self, case_id):
        return {'sha256': self.digest}

    def image_path(self, case_id):
        return self.path


class PatchObservations:
    spec = SPEC

    def observe(self, path, *, source_id, task_id, scope, observed_at, source_epoch):
        data = Path(path).read_bytes()
        features = torch.tensor(list(data), dtype=torch.float32).reshape(1, 4, 8) / 255
        return observed(features=features, image_sha256=hashlib.sha256(data).hexdigest(),
                        source_id=source_id, task_id=task_id, subject_id=scope.subject_id,
                        branch_id=scope.branch_id, purpose=scope.purpose,
                        observed_at=observed_at, source_epoch=source_epoch)


@pytest.mark.parametrize('update_enabled', [False, True])
def test_v0_runner_backward_lease_and_optional_update_preserve_frozen_tensors(
        visual_session, tmp_path, update_enabled):
    model, session = visual_session
    runner.freeze_for_v0(model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    protocol = {
        'vision_source': {'source_id': 'owned-patch-fixture'},
        'limits': {'max_prompt_tokens': 32},
        'v0': {'probe': {'case_id': 'fixture-color'},
               'backward': {'prompt': 'Inspect observed patches', 'target_text': 'probe',
                            'max_response_tokens': 8},
               'update': {'enabled': update_enabled, 'optimizer': 'sgd', 'lr': .001}},
    }
    result = runner.projector_gradient_probe(
        model, session, session.tokenizer, protocol, PatchObservations(),
        PatchDataset(tmp_path), SCOPE, session.ledger.epoch, tmp_path)
    assert result['decoder_input_gradient_nonzero'] and result['projector_gradients_nonzero']
    assert result['labels'] == [-100] * len(result['prompt_ids']) + result['response_ids']
    assert result['nonprojector_gradient_names'] == []
    changed = {name for name, value in model.named_parameters()
               if not torch.equal(value, before[name])}
    assert changed == set(result['update_changed'])
    assert bool(changed) is update_enabled
    assert all(name.startswith('vision_projector.') for name in changed)
    for name, value in model.named_parameters():
        if not name.startswith('vision_projector.'):
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            assert value.grad is None


def test_v1_runner_accumulates_projector_only_gradients_and_declared_repeats(
        visual_session, tmp_path):
    model, session = visual_session
    dataset = PatchDataset(tmp_path)
    train = dict(seed=31, updates=3, repeats=2, repeat_policy='declared_epochs',
                 schedule_version='seeded_epoch_shuffle.v1', max_grad_norm=1.0,
                 grad_accum=2, lr_new=.0001, momentum=.95, ns_steps=3,
                 block_size=8, weight_decay=0.0, max_seconds=60)
    rows = [{'case_id': f'fixture-{index}', 'prompt': f'Inspect patch question {index}',
             'response': f'observation {index}'} for index in range(4)]
    protocol = {'vision_source': {'source_id': 'owned-patch-fixture'},
                'limits': {'max_prompt_tokens': 32, 'max_seconds': 60},
                'v1': {'train': train, 'retention': {'max_cases': 1},
                       'eval': {'max_new_tokens': 2}}}
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    (tmp_path / 'v1').mkdir()
    retained = {'_retention_prompts': [
        {'case_id': 'held-text', 'prompt': 'Compute a short text reply',
         'family': 'independent-text', 'group_id': 'held-text', 'split': 'dev'}]}
    baseline = runner.evaluate_retention(
        session, session.tokenizer, protocol, retained, SCOPE, tmp_path, baseline=True)
    result = runner.train_projector(
        model, session, session.tokenizer, protocol, PatchObservations(), dataset,
        {'_training_rows': rows}, SCOPE, tmp_path)
    after = runner.evaluate_retention(
        session, session.tokenizer, protocol, retained, SCOPE, tmp_path)
    assert baseline['rows'] == after['rows']
    expected = training_schedule(rows, train)
    assert [entry['case_ids'] for entry in result['trace']] == [
        [row['case_id'] for row in group] for group in expected]
    assert [entry['repeat'] for entry in result['trace']] == [1, 1, 2]
    assert all(entry['grad_norm'] > 0 and entry['nonprojector_gradient_names'] == []
               for entry in result['trace'])
    validate_optimizer_route(result['optimizer_route'], result['optimizer_audit'])
    import copy
    import json
    from replica_lab.vision_qualification import validate_training_trace, QualificationError
    raw = [json.loads(line) for line in (tmp_path / 'v1/training_examples.jsonl').read_text().splitlines()]
    validate_training_trace(result['trace'], raw, rows, train, session.tokenizer,
                            result['trainable_receipt'])
    for mutation in ('duplicate_update', 'boolean_update', 'case_substitution',
                     'missing_backward', 'token_total', 'labels_hash', 'repeat', 'missing_example'):
        trace, examples = copy.deepcopy(result['trace']), copy.deepcopy(raw)
        if mutation == 'duplicate_update':
            trace[-1]['update'] = trace[0]['update']
        elif mutation == 'boolean_update':
            trace[0]['update'] = True
        elif mutation == 'case_substitution':
            trace[0]['case_ids'][0] = 'foreign-case'
        elif mutation == 'missing_backward':
            trace[0]['examples'][0]['backward_called'] = False
        elif mutation == 'token_total':
            trace[0]['prompt_tokens'] += 1
        elif mutation == 'labels_hash':
            trace[0]['examples'][0]['labels_sha256'] = '0' * 64
        elif mutation == 'repeat':
            trace[-1]['repeat'] = 1
        else:
            examples.pop()
        with pytest.raises(QualificationError):
            validate_training_trace(trace, examples, rows, train, session.tokenizer,
                                    result['trainable_receipt'])
    changed = {name for name, value in model.named_parameters()
               if not torch.equal(value, before[name])}
    assert changed and all(name.startswith('vision_projector.') for name in changed)
    for name, value in model.named_parameters():
        if not name.startswith('vision_projector.'):
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            assert value.grad is None


@pytest.mark.parametrize('budget', [0, 1])
def test_capture_releases_visual_request_after_generation_or_failure(
        visual_session, tmp_path, budget):
    from test_visual_adapter import arguments
    from safetensors.torch import load_file
    model, session = visual_session
    request = session.prepare(**arguments(observed()))
    if budget == 0:
        with pytest.raises(ValueError):
            runner.generate_capture(session, request, max_new_tokens=0,
                                    logits_path=tmp_path / 'invalid.safetensors',
                                    eos_token_ids={session.tokenizer.eos_token_id})
    else:
        result = runner.generate_capture(
            session, request, max_new_tokens=1,
            logits_path=tmp_path / 'captured.safetensors',
            eos_token_ids={session.tokenizer.eos_token_id})
        logits = load_file(str(result['logits_path']))['logits']
        assert logits.argmax(dim=-1).tolist() == result['generated_ids']
    assert request.frame is None and request.selection is None
