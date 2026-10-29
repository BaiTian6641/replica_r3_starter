"""Frozen-teacher qualification, bounded detached targets and live student updates."""
import json
from pathlib import Path

import torch

from .baseline_competence import score
from .contextual_memory import tensor_digest
from .contextual_teacher_objective_v1 import (
    TeacherObjective, TeacherTarget, binding, make_teacher_target, alignment_loss,
)
from .intrinsic_memory_budget_v5 import atomic, sha_json
from .intrinsic_memory_diagnostic_v5 import DiagnosticEngine
from .intrinsic_memory_execution_v5 import hash_file

CACHE = 'replica.intrinsic_memory.teacher_cache.v10'
CACHE_LIMIT = 2 * 1024**2


class TeacherEngine(DiagnosticEngine):
    def __init__(self, *args, base_identity, fit_identity, objective=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.objective = objective or TeacherObjective()
        self.base_identity, self.fit_identity = base_identity, fit_identity
        if self.path.objective_identity != self.objective.identity(self.spec):
            raise ValueError('extension checkpoint objective is not teacher-aligned')
        self.targets = {}
        self.teacher_evidence = []
        self.teacher_qualified = False

    def freeze(self):
        super().freeze()
        # Evaluation never consumes training targets or teacher responses.
        # A fresh numerical continuation restores its separately bound cache.
        self.targets.clear()
        self.teacher_evidence.clear()
        self.teacher_qualified = False

    def expected(self, row, student_encoding=None):
        student_encoding = student_encoding or self.encoding(row)
        return binding(row, self.encoding(row, exact=True), student_encoding, self.spec,
                       self.objective, self.base_identity, self.fit_identity)

    def _check_teacher_answer(self, row, answer):
        ids = answer.get('token_ids', [])
        return (answer.get('world_id') == row['world_id']
                and bool(ids) and len(ids) <= self.max_new_tokens
                and all(type(v) is int and 0 <= v < self.spec.vocab_size for v in ids)
                and ids[-1] in self.spec.allowed_stop_token_ids
                and not any(v in self.spec.allowed_stop_token_ids for v in ids[:-1])
                and answer.get('terminated') is True and answer.get('truncated') is False
                and answer.get('raw_answer') == self.backend.decode(ids[:-1])
                and score(row['answer'], answer['raw_answer'])
                and answer.get('correct') is True and answer.get('false_recall') is False)

    def prepare_teacher(self, rows, scopes):
        if (self.teacher_qualified or self.targets or self.fitting_closed
                or len(rows) != 8 or len(scopes) != 8
                or len({r['world_id'] for r in rows}) != 8):
            raise ValueError('one complete eight-world fitting teacher stage required')
        for row, scope in zip(rows, scopes):
            with self.measured('teacher_complete_context_qualification'):
                answer = self.generate(row, None, scope, control='exact_text')
            answer.pop('first_logits')
            self.teacher_evidence.append({'world_id': row['world_id'], **answer})
            with self.measured('teacher_qualification_evidence_write', writes=1) as event:
                receipt = self.output/'teacher_qualification.json'
                atomic(receipt, {
                    'rows': self.teacher_evidence, 'complete': len(self.teacher_evidence) == 8,
                    'passed': len(self.teacher_evidence) == 8 and all(
                        self._check_teacher_answer(r, a) for r, a in zip(rows, self.teacher_evidence)),
                    'frozen_original_extensions_disabled': True,
                })
                event['serialized_write_bytes'] = receipt.stat().st_size
        passed = all(self._check_teacher_answer(r, a) for r, a in zip(rows, self.teacher_evidence))
        if not passed:
            return False
        for row, scope in zip(rows, scopes):
            teacher_encoding, student_encoding = self.encoding(row, exact=True), self.encoding(row)
            with self.measured('teacher_response_target_forward') as event:
                with torch.no_grad():
                    logits = self.forward(teacher_encoding.input_ids, teacher_encoding.attention_mask,
                                          None, scope, control='exact_text')
                    target = make_teacher_target(logits, teacher_encoding, student_encoding,
                        row=row, spec=self.spec, objective=self.objective,
                        base_identity=self.base_identity, fit_identity=self.fit_identity)
                event.update(response_tokens=len(target.provenance['response_token_ids']),
                             support_tokens=target.indices.numel(),
                             detached_target_bytes=sum(t.numel()*t.element_size() for t in
                                                       (target.indices, target.log_probs, target.retained_mass)),
                             retained_mass_min=float(target.retained_mass.min()),
                             teacher_gradients=False, teacher_extensions_enabled=False)
            self.targets[row['world_id']] = target
            del logits
        self.teacher_qualified = True
        return True

    def save_teacher_cache(self, path):
        if not self.teacher_qualified or len(self.targets) != 8 or len(self.teacher_evidence) != 8:
            raise ValueError('incomplete teacher targets cannot be published')
        payload = {'format': CACHE, 'objective': self.objective.identity(self.spec),
                   'base_sha256': self.base_identity, 'fit_data_sha256': self.fit_identity,
                   'qualification': self.teacher_evidence,
                   'targets': {key: value.serialized() for key, value in self.targets.items()}}
        encoded_bytes = len((json.dumps(payload, sort_keys=True, indent=2)+'\n').encode())
        if encoded_bytes > CACHE_LIMIT:
            raise ValueError('teacher cache exceeds its declared byte cap')
        if Path(path).exists():
            raise ValueError('teacher cache is immutable; no overwrite')
        with self.measured('detached_teacher_cache_publication', writes=1, artifact_bytes=CACHE_LIMIT) as event:
            atomic(Path(path), payload)
            actual = Path(path).stat().st_size
            if actual > CACHE_LIMIT:
                raise ValueError('serialized teacher cache exceeds its declared byte cap')
            event['actual'] = {'artifact_bytes': actual}
            event['student_tensors_serialized'] = False
        return {'sha256': hash_file(Path(path)), 'bytes': actual}

    def load_teacher_cache(self, path, pin, rows):
        path = Path(path)
        if (self.teacher_qualified or self.targets or path.is_symlink()
                or not path.is_file() or path.stat().st_size != pin['bytes']
                or not 0 < pin['bytes'] <= CACHE_LIMIT or hash_file(path) != pin['sha256']):
            raise ValueError('teacher cache identity, size or lifecycle differs')
        with self.measured('detached_teacher_cache_restore') as event:
            payload = json.loads(path.read_text())
            event['artifact_read_bytes'] = path.stat().st_size
            if (payload.get('format') != CACHE or payload.get('objective') != self.objective.identity(self.spec)
                    or payload.get('base_sha256') != self.base_identity
                    or payload.get('fit_data_sha256') != self.fit_identity
                    or len(rows) != 8 or len(payload.get('qualification', [])) != 8
                    or set(payload.get('targets', {})) != {r['world_id'] for r in rows}
                    or not all(self._check_teacher_answer(r, a) for r, a in zip(rows, payload['qualification']))):
                raise ValueError('teacher cache qualification, source or objective differs')
            targets = {}
            for row in rows:
                target = TeacherTarget.from_serialized(payload['targets'][row['world_id']], device=self.device)
                target.validate(self.expected(row), spec=self.spec, objective=self.objective)
                targets[row['world_id']] = target
            self.targets = targets
            self.teacher_evidence = payload['qualification']
            self.teacher_qualified = True

    def update(self, row, source, scope):
        if self.fitting_closed or not self.teacher_qualified:
            raise ValueError('fitting closed or teacher competence not established')
        if row['world_id'] not in self.targets:
            raise ValueError('unqualified teacher world')
        with self.measured('teacher_aligned_response_update', backwards=1, updates=1) as event:
            self.path.optimizer.zero_grad(set_to_none=True)
            # The live producer path validates current scope/revision/authority.
            # Teacher cache reuse cannot make a revoked source eligible again.
            batch = self.live(source, scope)
            encoding = self.encoding(row)
            with self.measured('teacher_cache_target_revalidation'):
                target = self.targets[row['world_id']]
                expected = self.expected(row, encoding)
                target.validate(expected, spec=self.spec, objective=self.objective)
            logits = self.forward(encoding.input_ids, encoding.attention_mask, batch, scope)
            losses = alignment_loss(logits, encoding, target, expected=expected,
                                    spec=self.spec, objective=self.objective)
            losses['loss'].backward()
            gradients = self.gradients()
            before = {k: tensor_digest(v) for k, v in self.path._tensors().items()}
            self.path.optimizer.step()
            self.path.assert_optimizer_ownership()
            if any(not torch.isfinite(v).all() for v in self.path._tensors().values()):
                raise ValueError('nonfinite updated parameter')
            if any(torch.is_tensor(v) and not torch.isfinite(v).all()
                   for state in self.path.optimizer.state.values() for v in state.values()):
                raise ValueError('nonfinite optimizer moment')
            changed = [k for k, v in self.path._tensors().items() if tensor_digest(v) != before[k]]
            if not changed:
                raise ValueError('no actual parameter update')
            event.update({k: float(v.detach()) if torch.is_tensor(v) else v for k, v in losses.items()})
            event.update(gradients=gradients, changed_tensors=changed,
                         teacher_cache_hit=True, teacher_forward_in_update=False,
                         objective_sha256=sha_json(self.objective.identity(self.spec)))
            return dict(event)
