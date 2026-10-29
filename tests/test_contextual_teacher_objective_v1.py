from dataclasses import replace

import pytest
import torch

from replica_lab.contextual_completion_objective_v1 import (
    CompletionObjectiveSpec, stop_policy_identity, encode_complete_response,
)
from replica_lab.contextual_teacher_objective_v1 import (
    TeacherObjective, TeacherTarget, binding, make_teacher_target, alignment_loss,
)


def fixture(top_k=2):
    spec = CompletionObjectiveSpec(8, 12, 1, 2, (1,), stop_policy_identity((1,)), 'a'*64, 'b'*64, 'none')
    objective = TeacherObjective(top_k=top_k)
    teacher = encode_complete_response([3, 4, 5, 6], [7, 4], spec=spec)
    student = encode_complete_response([3, 5], [7, 4], spec=spec)
    row = {'world_id': 'test', 'text': 'attributed source', 'question': 'question', 'answer': 'answer'}
    logits = torch.arange(96, dtype=torch.float32).reshape(1, 12, 8).requires_grad_()
    target = make_teacher_target(logits, teacher, student, row=row, spec=spec,
        objective=objective, base_identity='c'*64, fit_identity='d'*64)
    return spec, objective, teacher, student, row, logits, target


def test_response_index_alignment_includes_EOS_and_detaches_teacher_only():
    spec, objective, teacher, student, row, teacher_logits, target = fixture()
    assert target.provenance['teacher_prediction_positions'] == [3, 4, 5]
    assert target.provenance['student_prediction_positions'] == [1, 2, 3]
    assert target.provenance['response_token_ids'] == [7, 4, 1]
    assert not target.log_probs.requires_grad and target.log_probs.grad_fn is None
    logits = torch.randn(1, 12, 8, requires_grad=True)
    losses = alignment_loss(logits, student, target, expected=target.provenance, spec=spec, objective=objective)
    losses['loss'].backward()
    assert teacher_logits.grad is None
    assert (logits.grad[0].abs().sum(-1).nonzero().flatten() == torch.tensor([1, 2, 3])).all()
    assert losses['supervised_tokens'] == 3


def test_student_normalizes_over_full_vocabulary_and_matches_manual_KL():
    spec, objective, _, student, _, _, target = fixture()
    logits = torch.zeros(1, 12, 8, requires_grad=True)
    loss = alignment_loss(logits, student, target, expected=target.provenance, spec=spec, objective=objective)['loss']
    expected = (target.log_probs.exp() * (target.log_probs + torch.log(torch.tensor(8.0)))).sum(-1).mean()
    assert torch.allclose(loss, expected)
    changed = logits.detach().clone()
    changed[0, [1, 2, 3], 0] = 20  # outside the teacher's selected support (7, 6)
    larger = alignment_loss(changed, student, target, expected=target.provenance, spec=spec, objective=objective)['loss']
    assert larger > loss + 10


@pytest.mark.parametrize('damage', ['world', 'base', 'fit', 'student_prefix', 'teacher_prefix', 'response', 'EOS'])
def test_wrong_source_or_position_binding_refused(damage):
    spec, objective, teacher, student, row, _, target = fixture()
    base, fit = 'c'*64, 'd'*64
    if damage == 'world': row = {**row, 'text': 'corrected source'}
    elif damage == 'base': base = 'e'*64
    elif damage == 'fit': fit = 'e'*64
    elif damage == 'student_prefix': student = encode_complete_response([3, 6], [7, 4], spec=spec)
    elif damage == 'teacher_prefix': teacher = encode_complete_response([3, 4, 5, 7], [7, 4], spec=spec)
    elif damage == 'response': teacher = encode_complete_response([3, 4, 5, 6], [6, 4], spec=spec)
    else: student.labels[0, 3] = -100
    with pytest.raises(ValueError):
        expected = binding(row, teacher, student, spec, objective, base, fit)
        alignment_loss(torch.zeros(1, 12, 8), student, target, expected=expected, spec=spec, objective=objective)


@pytest.mark.parametrize('damage', ['duplicate', 'nonfinite', 'mass', 'normalization', 'attached', 'format', 'float_id'])
def test_invalid_or_attached_cached_teacher_distribution_refused(damage):
    spec, objective, _, _, _, _, target = fixture()
    if damage == 'duplicate': target.indices[:, 1] = target.indices[:, 0]
    elif damage == 'nonfinite': target.log_probs[0, 0] = float('nan')
    elif damage == 'mass': target.retained_mass[0] = 1.1
    elif damage == 'normalization': target.log_probs[0, 0] += 2
    elif damage == 'attached': target = replace(target, log_probs=target.log_probs.requires_grad_())
    else:
        serialized = target.serialized()
        if damage == 'format': serialized['format'] = 'older_objective'
        else: serialized['indices'][0][0] = 7.5
        with pytest.raises(ValueError): TeacherTarget.from_serialized(serialized)
        return
    with pytest.raises(ValueError): target.validate(target.provenance, spec=spec, objective=objective)


def test_serialized_teacher_targets_round_trip_exactly_with_no_student_graph():
    spec, objective, _, _, _, _, target = fixture()
    restored = TeacherTarget.from_serialized(target.serialized())
    restored.validate(target.provenance, spec=spec, objective=objective)
    assert torch.equal(restored.log_probs, target.log_probs)
    assert torch.equal(restored.indices, target.indices)
    assert torch.equal(restored.retained_mass, target.retained_mass)


@pytest.mark.parametrize('kwargs', [{'top_k': 0}, {'temperature': 0}, {'kl_weight': -1},
                                    {'kl_weight': 0, 'ce_weight': 0}, {'temperature': float('nan')}])
def test_invalid_objective_refused(kwargs):
    with pytest.raises(ValueError): TeacherObjective(**kwargs)


def test_declared_temperature_and_CE_mixture_have_correct_scale():
    spec, _, teacher, student, row, teacher_logits, _ = fixture()
    objective = TeacherObjective(top_k=2, temperature=2, kl_weight=0.25, ce_weight=0.75)
    target = make_teacher_target(teacher_logits, teacher, student, row=row, spec=spec,
        objective=objective, base_identity='c'*64, fit_identity='d'*64)
    logits = torch.randn(1, 12, 8)
    loss = alignment_loss(logits, student, target, expected=target.provenance, spec=spec, objective=objective)
    q = target.log_probs.exp()
    s = (logits[0, [1, 2, 3]]/2).log_softmax(-1).gather(-1, target.indices)
    assert torch.allclose(loss['kl_loss'], 4*(q*(target.log_probs-s)).sum(-1).mean())
    assert torch.allclose(loss['loss'], 0.25*loss['kl_loss']+0.75*loss['ce_loss'])
