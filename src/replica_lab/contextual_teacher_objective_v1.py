"""Response-aligned frozen-teacher targets; never a durable student tensor path.

The teacher sees selected exact source text. The student sees the question and
live contextual values. Their prompt lengths can differ. Predictions align by
the same response-token sequence, including the single bound EOS. Only teacher
targets cross the explicit detach/serialization boundary; student logits and
producer/reader tensors remain in the autograd graph.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .contextual_completion_objective_v1 import (
    complete_response_loss, validate_complete_response, IGNORE_INDEX,
)
from .intrinsic_memory_budget_v5 import sha_json

FORMAT = 'replica.contextual_teacher_targets.v1'
OBJECTIVE = 'replica.contextual_teacher_alignment.v1'


@dataclass(frozen=True)
class TeacherObjective:
    top_k: int = 64
    temperature: float = 1.0
    kl_weight: float = 1.0
    ce_weight: float = 0.0

    def __post_init__(self):
        if (type(self.top_k) is not int or self.top_k < 1
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and torch.isfinite(torch.tensor(v)).item()
                           for v in (self.temperature, self.kl_weight, self.ce_weight))
                or self.temperature <= 0 or min(self.kl_weight, self.ce_weight) < 0
                or self.kl_weight + self.ce_weight <= 0):
            raise ValueError('invalid teacher objective')

    def identity(self, completion_spec):
        if self.top_k > completion_spec.vocab_size:
            raise ValueError('teacher support exceeds bound vocabulary')
        return {'format': OBJECTIVE, 'completion': completion_spec.identity(),
                'top_k': self.top_k, 'temperature': self.temperature,
                'kl_weight': self.kl_weight, 'ce_weight': self.ce_weight,
                'teacher_normalization': 'renormalized_top_k',
                'student_normalization': 'full_vocabulary',
                'alignment': 'response_token_index_including_one_EOS',
                'teacher_boundary': 'frozen_extensions_disabled_detached_targets',
                'student_boundary': 'live_producer_and_reader_no_serialization'}


def response_positions(encoding, spec):
    validate_complete_response(encoding, spec=spec)
    return (encoding.labels[0] != IGNORE_INDEX).nonzero(as_tuple=False).flatten()


def binding(row, teacher_encoding, student_encoding, spec, objective, base_identity, fit_identity):
    """Discrete provenance checks are outside the differentiable student path."""
    for identity in (base_identity, fit_identity):
        if (not isinstance(identity, str) or len(identity) != 64
                or any(c not in '0123456789abcdef' for c in identity)):
            raise ValueError('teacher base and fitting-data digests required')
    teacher_positions = response_positions(teacher_encoding, spec)
    student_positions = response_positions(student_encoding, spec)
    teacher_ids = teacher_encoding.labels[0, teacher_positions]
    student_ids = student_encoding.labels[0, student_positions]
    if not torch.equal(teacher_ids, student_ids):
        raise ValueError('teacher/student response token sequences differ')
    return {'objective': objective.identity(spec), 'base_sha256': base_identity,
            'fit_data_sha256': fit_identity, 'world_sha256': sha_json(row),
            'teacher_input_sha256': teacher_encoding.metadata['input_tokens_sha256'],
            'student_input_sha256': student_encoding.metadata['input_tokens_sha256'],
            'teacher_prediction_positions': teacher_positions.tolist(),
            'student_prediction_positions': student_positions.tolist(),
            'response_token_ids': student_ids.tolist()}


@dataclass(frozen=True)
class TeacherTarget:
    provenance: dict
    indices: torch.Tensor
    log_probs: torch.Tensor
    retained_mass: torch.Tensor

    def validate(self, expected, *, spec, objective):
        if self.provenance != expected or expected['objective'] != objective.identity(spec):
            raise ValueError('teacher target source/objective/response binding differs')
        n = len(expected['response_token_ids'])
        if (self.indices.dtype != torch.long or self.indices.shape != (n, objective.top_k)
                or self.log_probs.dtype != torch.float32 or self.log_probs.shape != self.indices.shape
                or self.retained_mass.dtype != torch.float32 or self.retained_mass.shape != (n,)
                or any(t.requires_grad or t.grad_fn is not None for t in (self.indices, self.log_probs, self.retained_mass))
                or any(t.device != self.indices.device for t in (self.log_probs, self.retained_mass))
                or not torch.isfinite(self.log_probs).all() or not torch.isfinite(self.retained_mass).all()
                or (self.indices < 0).any() or (self.indices >= spec.vocab_size).any()
                or (self.retained_mass <= 0).any() or (self.retained_mass > 1.00001).any()
                or not torch.allclose(self.log_probs.exp().sum(-1), torch.ones(n, device=self.indices.device), atol=2e-6, rtol=0)
                or any(len(set(row)) != objective.top_k for row in self.indices.tolist())):
            raise ValueError('invalid detached teacher support/distribution')

    def serialized(self):
        if any(t.requires_grad or t.grad_fn is not None for t in (self.log_probs, self.retained_mass)):
            raise ValueError('only detached teacher targets may be serialized')
        return {'format': FORMAT, 'provenance': self.provenance,
                'indices': self.indices.cpu().tolist(), 'log_probs': self.log_probs.cpu().tolist(),
                'retained_mass': self.retained_mass.cpu().tolist()}

    @classmethod
    def from_serialized(cls, payload, *, device='cpu'):
        if payload.get('format') != FORMAT or set(payload) != {'format', 'provenance', 'indices', 'log_probs', 'retained_mass'}:
            raise ValueError('teacher target serialization differs')
        if any(type(v) is not int for row in payload['indices'] for v in row):
            raise ValueError('teacher support IDs must be integers')
        return cls(payload['provenance'], torch.tensor(payload['indices'], dtype=torch.long, device=device),
                   torch.tensor(payload['log_probs'], dtype=torch.float32, device=device),
                   torch.tensor(payload['retained_mass'], dtype=torch.float32, device=device))


def make_teacher_target(logits, teacher_encoding, student_encoding, *, row, spec,
                        objective, base_identity, fit_identity):
    expected = binding(row, teacher_encoding, student_encoding, spec, objective, base_identity, fit_identity)
    if (logits.shape != (1, spec.context_tokens, spec.vocab_size) or not logits.is_floating_point()
            or logits.device != teacher_encoding.input_ids.device or not torch.isfinite(logits).all()):
        raise ValueError('finite bound teacher logits required')
    # Intentional teacher-only stop-gradient. No producer, reader or student
    # logit tensor is detached by this function or by alignment_loss below.
    selected = logits[0, expected['teacher_prediction_positions']].detach().float()
    # The caller separately verifies generated teacher answers. Soft targets
    # retain alternatives; their argmax need not equal a canonical answer token
    # (for example, an equally correct capitalization). Both paths still use
    # exactly the same observed response prefix and explicit terminal target.
    scaled = selected / objective.temperature
    values, indices = scaled.topk(objective.top_k, dim=-1)
    log_probs = values.log_softmax(-1)
    retained = (values.logsumexp(-1) - scaled.logsumexp(-1)).exp()
    target = TeacherTarget(expected, indices.detach(), log_probs.detach(), retained.detach())
    target.validate(expected, spec=spec, objective=objective)
    return target


def alignment_loss(logits, student_encoding, target, *, expected, spec, objective):
    target.validate(expected, spec=spec, objective=objective)
    positions = response_positions(student_encoding, spec)
    if (positions.tolist() != expected['student_prediction_positions']
            or student_encoding.metadata['input_tokens_sha256'] != expected['student_input_sha256']
            or student_encoding.labels[0, positions].tolist() != expected['response_token_ids']):
        raise ValueError('student response alignment differs')
    ce = complete_response_loss(logits, student_encoding, spec=spec)
    if target.indices.device != logits.device:
        raise ValueError('teacher targets must be on the student device')
    # Normalize BEFORE gathering. A top-k-only student softmax would silently
    # discard probability mass assigned to unsupported tokens.
    log_student = F.log_softmax(logits[0, positions].float() / objective.temperature, dim=-1)
    gathered = log_student.gather(-1, target.indices)
    per_token = (target.log_probs.exp() * (target.log_probs - gathered)).sum(-1)
    kl = per_token.mean() * objective.temperature**2
    loss = objective.kl_weight * kl + objective.ce_weight * ce.loss
    if not torch.isfinite(loss):
        raise ValueError('nonfinite teacher alignment objective')
    return {'loss': loss, 'kl_loss': kl, 'ce_loss': ce.loss,
            'word_loss': ce.word_loss, 'eos_loss': ce.terminal_loss,
            'supervised_tokens': ce.supervised_tokens,
            'teacher_retained_mass_mean': target.retained_mass.mean()}
