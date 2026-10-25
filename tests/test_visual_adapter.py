"""Visual prefixes use the existing native decoder/session with isolated ownership."""
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from architecture_helpers import tiny_adapter
from replica_lab.architecture.contracts import TaskGraph, VisionSpec, VisualInput
from replica_lab.memory_system import EligibilityScope, RecordLedger
from replica_lab.memory_system.session import MemorySession
from test_exact_memory import Tokenizer

SPEC = VisionSpec('offline/vision-fixture', '1' * 40, '2' * 64, '3' * 64,
                  encoder_width=8, image_size=32, patch_size=16, max_visual_tokens=4,
                  projector_hidden=16)
SCOPE = EligibilityScope('visual-subject', 'visual-branch', 'visual-contract', 5, 'parallel_replica')
TASK = TaskGraph('visual-task', 'Inspect an observed scene')


def observed(**changes):
    values = dict(features=torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 10,
                  valid=torch.ones(1, 4, dtype=torch.bool), producer=SPEC.digest,
                  encoder_producer='4' * 64, image_sha256='5' * 64,
                  source_id='observed-scene', task_id=TASK.task_id, subject_id=SCOPE.subject_id,
                  branch_id=SCOPE.branch_id, purpose=SCOPE.purpose, observed_at=2,
                  source_epoch=0, original_size=(32, 32), crop=(0, 0, 32, 32), grid=(2, 2))
    values.update(changes)
    return VisualInput(**values)


@pytest.fixture
def visual_session(tmp_path):
    model = tiny_adapter(vision=SPEC).eval()
    ledger = RecordLedger(tmp_path / 'visual.sqlite')
    session = MemorySession(model, Tokenizer(), ledger, max_candidate_units=4)
    try:
        yield model, session
    finally:
        ledger.close()


def arguments(visual=None, *, mask=None, **changes):
    ids = torch.tensor([[1, 2, 3]])
    result = dict(prompt_ids=ids, prompt_mask=torch.ones_like(ids, dtype=torch.bool) if mask is None else mask,
                  query='What is observed?', scope=SCOPE, task=TASK, core_impl='none',
                  enabled_ports=(), enabled_branches=('memory',), visual=visual)
    result.update(changes)
    return result


@pytest.mark.parametrize('visual_enabled', [False, True])
@pytest.mark.parametrize('left_padding', [False, True])
def test_fused_masks_positions_and_text_bypass_match_native(visual_session, visual_enabled, left_padding):
    model, session = visual_session
    visual = observed() if visual_enabled else None
    mask = torch.tensor([[False, True, True]]) if left_padding else torch.ones(1, 3, dtype=torch.bool)
    args = arguments(visual, mask=mask)
    request = session.prepare(**args)
    try:
        with torch.no_grad():
            actual = session.forward(request, args['prompt_ids'], attention_mask=mask).logits
            text = model.base.get_input_embeddings()(args['prompt_ids'])
            if visual is not None:
                embeddings = torch.cat((model.project_visual(visual), text), dim=1)
                full_mask = torch.cat((visual.valid, mask), dim=1)
            else:
                embeddings, full_mask = text, mask
            positions = full_mask.long().cumsum(-1) - 1
            positions = positions.masked_fill(~full_mask, 1)
            expected = model.base(inputs_embeds=embeddings, attention_mask=full_mask.long(),
                                  position_ids=positions, use_cache=False).logits
            if visual is not None:
                expected = expected[:, visual.features.shape[1]:]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert actual.shape[1] == args['prompt_ids'].shape[1]
        assert request.frame.sense.visual is visual
    finally:
        request.release()


@pytest.mark.parametrize('checkpointed', [False, True])
def test_projector_receives_decoder_input_gradient_with_frozen_original(visual_session, monkeypatch, checkpointed):
    model, session = visual_session
    model.train()
    if checkpointed:
        model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    for name, parameter in model.new_named_parameters():
        parameter.requires_grad_(name.startswith('vision_projector.'))
    captured = []
    project = model.project_visual
    def retain_projection(visual):
        value = project(visual)
        if value.requires_grad:
            value.retain_grad()
            captured.append(value)
        return value
    monkeypatch.setattr(model, 'project_visual', retain_projection)
    args = arguments(observed())
    ids = torch.cat((args['prompt_ids'], torch.tensor([[4, 5]])), dim=1)
    labels = ids.clone()
    labels[:, :3] = -100
    with session.training_request(**args) as request:
        before = [summary.detach().clone() for _, summary in request.frame.sense.summaries]
        output = session.forward(request, ids, labels=labels)
        session.backward(request, output.loss)
        assert output.logits.shape[:2] == ids.shape
        assert request.frame.sense.prompt_ids.shape[1] == 3
        for old, (_, summary) in zip(before, request.frame.sense.summaries):
            torch.testing.assert_close(old, summary, atol=0, rtol=0)
    assert len(captured) == 1 and bool(captured[0].grad.ne(0).any())
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert all(parameter.grad is None for name, parameter in model.new_named_parameters()
               if not name.startswith('vision_projector.'))
    assert all(parameter.grad is not None and bool(parameter.grad.ne(0).any())
               for parameter in model.vision_projector.parameters())


@pytest.mark.parametrize('field,value', [('task_id', 'other-task'), ('subject_id', 'other-subject'),
                                         ('branch_id', 'foreign'), ('purpose', 'foreign'),
                                         ('source_epoch', 1), ('observed_at', 6)])
def test_visual_scope_task_epoch_and_future_information_rejected(visual_session, field, value):
    _, session = visual_session
    with pytest.raises(PermissionError, match='another task'):
        session.prepare(**arguments(observed(**{field: value})))


@pytest.mark.parametrize('changes', [dict(core_impl='recurrent'), dict(enabled_ports=(0,)),
                                      dict(enabled_branches=('memory', 'workspace')),
                                      dict(train_sense=True), dict(train_core=True)])
def test_visual_path_cannot_enable_cognitive_or_memory_injections(visual_session, changes):
    _, session = visual_session
    with pytest.raises(ValueError, match='isolated'):
        session.prepare(**arguments(observed(), **changes))


def test_visual_patches_and_projector_versions_are_request_owned(visual_session):
    model, session = visual_session
    visual = observed()
    request = session.prepare(**arguments(visual))
    with torch.no_grad():
        visual.features.add_(1)
    with pytest.raises(RuntimeError, match='mutated|changed'):
        session.generate(request, max_new_tokens=1)
    request.release()
    visual = observed()
    request = session.prepare(**arguments(visual))
    with torch.no_grad():
        next(model.vision_projector.parameters()).add_(.1)
    with pytest.raises((RuntimeError, ValueError), match='changed|producer'):
        session.generate(request, max_new_tokens=1)
    request.release()


def test_visual_sensing_changes_with_observation_but_not_teacher_suffix(visual_session):
    model, session = visual_session
    first, second = observed(), observed(features=torch.randn(1, 4, 8), image_sha256='6' * 64)
    args = arguments(first)
    with torch.no_grad():
        lhs = model.sense(args['prompt_ids'], args['prompt_mask'], visual=first)
        rhs = model.sense(args['prompt_ids'], args['prompt_mask'], visual=second)
    assert any(not torch.equal(a, b) for (_, a), (_, b) in zip(lhs.summaries, rhs.summaries))
    request = session.prepare(**args)
    try:
        labels = args['prompt_ids'].clone()
        with pytest.raises(ValueError, match='Sense prefix'):
            session.forward(request, args['prompt_ids'], labels=labels)
    finally:
        request.release()


@pytest.mark.parametrize('change', [dict(producer='9' * 64), dict(grid=(1, 4)),
                                     dict(features=torch.randn(1, 4, 7))])
def test_visual_producer_layout_and_encoder_width_must_match_spec(visual_session, change):
    _, session = visual_session
    with pytest.raises(ValueError, match='producer|layout'):
        session.prepare(**arguments(observed(**change)))


def test_generation_and_strict_projector_extension_reload(visual_session, tmp_path):
    model, session = visual_session
    model.seal_original_identity()
    args = arguments(observed())
    request = session.prepare(**args)
    try:
        before = session.generate(request, max_new_tokens=3)
    finally:
        request.release()
    receipt = model.save_extension(tmp_path / 'extension')
    assert receipt['format'] == 'replica.minicpm5.extensions.v4' and not receipt['base_saved']
    fresh = tiny_adapter(vision=SPEC).eval()
    fresh.seal_original_identity()
    fresh.load_extension(tmp_path / 'extension')
    second = MemorySession(fresh, Tokenizer(), session.ledger, max_candidate_units=4)
    request = second.prepare(**arguments(observed()))
    try:
        after = second.generate(request, max_new_tokens=3)
    finally:
        request.release()
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    incompatible = tiny_adapter(vision=replace(SPEC, processor_digest='8' * 64)).eval()
    incompatible.seal_original_identity()
    with pytest.raises(ValueError, match='manifest/bytes'):
        incompatible.load_extension(tmp_path / 'extension')
    from safetensors.torch import load_file, save_file
    path = tmp_path / 'extension' / 'extensions.safetensors'
    tensors = load_file(str(path))
    tensors.pop(next(name for name in tensors if name.startswith('vision_projector.')))
    save_file(tensors, str(path))
    manifest = json.loads((tmp_path / 'extension' / 'manifest.json').read_text())
    manifest['weights_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'extension' / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='tensor inventory'):
        fresh.load_extension(tmp_path / 'extension')


def test_visual_layout_copies_mutable_caller_metadata():
    crop, grid, size = [0, 0, 32, 32], [2, 2], [32, 32]
    visual = observed(crop=crop, grid=grid, original_size=size)
    key = visual.cache_key
    crop[2], grid[0], size[0] = 1, 1, 1
    assert visual.crop == (0, 0, 32, 32) and visual.grid == (2, 2)
    assert visual.original_size == (32, 32) and visual.cache_key == key


def test_encoder_feature_gradient_flag_cannot_change_behind_a_frame(visual_session):
    _, session = visual_session
    visual = observed()
    request = session.prepare(**arguments(visual))
    visual.features.requires_grad_(True)
    with pytest.raises(RuntimeError, match='changed'):
        session.generate(request, max_new_tokens=1)
    request.release()
