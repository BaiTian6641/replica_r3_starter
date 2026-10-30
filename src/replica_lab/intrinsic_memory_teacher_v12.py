"""v12 frozen-teacher qualification with a feature-contract-bound target cache.

The teacher is the frozen original with all extensions disabled: its source is
the selected exact canonical text, so it does not use the trainable compressor
and cannot inherit its errors. Only detached soft targets cross the boundary.

The cache now binds more than the objective: the immutable source-feature
contract, the adapter contract and the tokenizer contract are recorded and
re-checked on restore and before every update. A changed feature contract can
therefore never be concealed by a parameter-only mutation stamp. The producer
*release* (which changes on every update) is deliberately not a cache field;
the per-target provenance still binds the student prompt tokens, the world and
the fitting data digest.

Group semantics are unchanged from the sealed v11 comparison: four fixed
fitting groups of eight and two fixed development groups of eight must each
pass independently; no pooling, prefix selection or regrouping.
"""
import json
from pathlib import Path

import torch

from .baseline_competence import score
from .contextual_memory import tensor_digest
from .contextual_teacher_objective_v1 import (
    TeacherObjective, TeacherTarget, binding, make_teacher_target, alignment_loss)
from .intrinsic_memory_budget_v12 import atomic, sha_json
from .intrinsic_memory_coverage_v12 import (
    CODEC_VALUE, DEVELOPMENT_CASES, FIT_CASES, GROUP_SIZE, REPRESENTATION_VALUE,
    TEACHER_CACHE, TEACHER_CACHE_BYTES)
from .intrinsic_memory_diagnostic_v12 import CONTROLS, DiagnosticEngineV12
from .intrinsic_memory_execution_v5 import hash_file

FORMAT = TEACHER_CACHE


class TeacherEngineV12(DiagnosticEngineV12):
    def __init__(self, *args, base_identity, fit_identity, objective=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.objective = objective or TeacherObjective()
        self.base_identity, self.fit_identity = base_identity, fit_identity
        if self.path.objective_identity != self.objective.identity(self.spec):
            raise ValueError('extension checkpoint objective is not teacher-aligned')
        self.targets = {}
        self.teacher_evidence = []
        self.teacher_qualified = False
        self.teacher_feature_identity = None
        self.fit_cases = FIT_CASES

    # ------------------------------------------------------------------
    # Immutable feature contract

    def feature_identity(self):
        provider = self.path.feature_provider
        return {'feature_contract_sha256': sha_json(provider.spec.identity()),
                'adapter_contract': provider.adapter_contract,
                'tokenizer_contract': provider.tokenizer_contract,
                'representation': REPRESENTATION_VALUE,
                'codec': CODEC_VALUE,
                'original_updates': 0}

    def assert_feature_contract(self):
        current = self.feature_identity()
        if self.teacher_feature_identity is not None and current != self.teacher_feature_identity:
            raise ValueError('source feature contract changed after teacher qualification')
        return current

    # ------------------------------------------------------------------
    # Teacher qualification and detached cache

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
        if (self.teacher_qualified or self.targets or self.teacher_evidence or self.fitting_closed
                or len(rows) != FIT_CASES or len(scopes) != FIT_CASES
                or len({r['world_id'] for r in rows}) != FIT_CASES):
            raise ValueError('one complete declared fitting teacher stage required')
        for row, scope in zip(rows, scopes):
            with self.measured('teacher_complete_context_qualification'):
                answer = self.generate(row, None, scope, control='exact_text')
            answer.pop('first_logits')
            self.teacher_evidence.append({'world_id': row['world_id'], **answer})
            with self.measured('teacher_qualification_evidence_write', writes=1) as event:
                receipt = self.output/'teacher_qualification.json'
                atomic(receipt, {
                    'rows': self.teacher_evidence,
                    'complete': len(self.teacher_evidence) == FIT_CASES,
                    'passed': len(self.teacher_evidence) == FIT_CASES and all(
                        self._check_teacher_answer(r, a)
                        for r, a in zip(rows, self.teacher_evidence)),
                    'frozen_original_extensions_disabled': True,
                    'source_feature_contract': self.feature_identity(),
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
                                                       (target.indices, target.log_probs,
                                                        target.retained_mass)),
                             retained_mass_min=float(target.retained_mass.min()),
                             teacher_gradients=False, teacher_extensions_enabled=False,
                             source_feature_provider_used=False)
            self.targets[row['world_id']] = target
            del logits
        self.teacher_qualified = True
        self.teacher_feature_identity = self.feature_identity()
        return True

    def save_teacher_cache(self, path):
        if (not self.teacher_qualified or len(self.targets) != FIT_CASES
                or len(self.teacher_evidence) != FIT_CASES):
            raise ValueError('incomplete teacher targets cannot be published')
        feature = self.assert_feature_contract()
        payload = {'format': FORMAT, 'objective': self.objective.identity(self.spec),
                   'base_sha256': self.base_identity, 'fit_data_sha256': self.fit_identity,
                   'fit_cases': FIT_CASES, 'qualification': self.teacher_evidence,
                   'feature_identity': feature,
                   'targets': {key: value.serialized() for key, value in self.targets.items()}}
        encoded_bytes = len((json.dumps(payload, sort_keys=True, indent=2)+'\n').encode())
        if encoded_bytes > TEACHER_CACHE_BYTES:
            raise ValueError('teacher cache exceeds its declared byte cap')
        if Path(path).exists():
            raise ValueError('teacher cache is immutable; no overwrite')
        with self.measured('detached_teacher_cache_publication', writes=1,
                           artifact_bytes=TEACHER_CACHE_BYTES) as event:
            atomic(Path(path), payload)
            actual = Path(path).stat().st_size
            if actual > TEACHER_CACHE_BYTES:
                raise ValueError('serialized teacher cache exceeds its declared byte cap')
            event['actual'] = {'artifact_bytes': actual}
            event['student_tensors_serialized'] = False
            event['feature_contract_bound'] = True
        return {'sha256': hash_file(Path(path)), 'bytes': actual}

    def load_teacher_cache(self, path, pin, rows):
        path = Path(path)
        if (self.teacher_qualified or self.targets or self.teacher_evidence or self.fitting_closed
                or path.is_symlink() or not path.is_file() or path.stat().st_size != pin['bytes']
                or not 0 < pin['bytes'] <= TEACHER_CACHE_BYTES):
            raise ValueError('teacher cache identity, size or lifecycle differs')
        feature = self.feature_identity()
        # Hashing the file and then reading it are two full reads, both reserved.
        with self.measured('detached_teacher_cache_restore',
                           artifact_read_bytes=2*pin['bytes'],
                           restart_restores=1) as event:
            if hash_file(path) != pin['sha256']:
                raise ValueError('teacher cache content digest differs')
            payload = json.loads(path.read_text())
            event['observed_artifact_read_bytes'] = 2*path.stat().st_size
            if (payload.get('format') != FORMAT
                    or payload.get('objective') != self.objective.identity(self.spec)
                    or payload.get('base_sha256') != self.base_identity
                    or payload.get('fit_data_sha256') != self.fit_identity
                    or payload.get('fit_cases') != FIT_CASES
                    or payload.get('feature_identity') != feature
                    or len(rows) != FIT_CASES
                    or len(payload.get('qualification', [])) != FIT_CASES
                    or set(payload.get('targets', {})) != {r['world_id'] for r in rows}
                    or not all(self._check_teacher_answer(r, a)
                               for r, a in zip(rows, payload['qualification']))):
                raise ValueError('teacher cache feature contract, qualification or objective differs')
            targets = {}
            for row in rows:
                target = TeacherTarget.from_serialized(payload['targets'][row['world_id']],
                                                       device=self.device)
                target.validate(self.expected(row), spec=self.spec, objective=self.objective)
                targets[row['world_id']] = target
            self.targets = targets
            self.teacher_evidence = payload['qualification']
            self.teacher_qualified = True
            self.teacher_feature_identity = feature
            event['feature_contract_bound'] = True

    def update(self, row, source, scope):
        if self.fitting_closed or not self.teacher_qualified:
            raise ValueError('fitting closed or teacher competence not established')
        if row['world_id'] not in self.targets:
            raise ValueError('unqualified teacher world')
        self.assert_feature_contract()
        with self.measured('teacher_aligned_response_update', backwards=1, updates=1) as event:
            self.path.optimizer.zero_grad(set_to_none=True)
            # The live producer path validates current scope/revision/time/
            # authority; re-encoding a revoked source cannot be rescued by the
            # cache. Teacher targets only supply supervised response soft labels.
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
            event.update({k: float(v.detach()) if torch.is_tensor(v) else v
                          for k, v in losses.items()})
            event.update(gradients=gradients, changed_tensors=changed,
                         teacher_cache_hit=True, teacher_forward_in_update=False,
                         objective_sha256=sha_json(self.objective.identity(self.spec)),
                         feature_contract_sha256=self.teacher_feature_identity[
                             'feature_contract_sha256'])
            return dict(event)

    # ------------------------------------------------------------------
    # Fixed group geometry

    @staticmethod
    def _complete_groups(rows, expected, *aligned):
        if (len(rows) != expected or len({r['world_id'] for r in rows}) != expected
                or any(len(items) != expected for items in aligned)):
            raise ValueError('complete predeclared coverage groups required')
        for i, row in enumerate(rows):
            if type(row.get('criterion_group')) is not int or row['criterion_group'] != i // GROUP_SIZE:
                raise ValueError('predeclared group order differs')
            if row['question'] != rows[i ^ 1]['question'] or row['answer'] == rows[i ^ 1]['answer']:
                raise ValueError('counterfactual pair differs')
        return range(0, expected, GROUP_SIZE)

    def fitting_probes(self, rows, sources, scopes, stores):
        groups = []
        for start in self._complete_groups(rows, FIT_CASES, sources, scopes, stores):
            end = start + GROUP_SIZE
            groups.append(DiagnosticEngineV12.fitting_probes(
                self, rows[start:end], sources[start:end], scopes[start:end], stores[start:end]))
        return {'format': 'replica.intrinsic_memory.coverage_fit_probes.v12',
                'rows': [row for group in groups for row in group['rows']],
                'correct': {control: sum(g['correct'][control] for g in groups)
                            for control in CONTROLS},
                'groups': [{k: v for k, v in group.items() if k != 'rows'} for group in groups],
                'same_question_counterfactual_pairs': [pair for g in groups
                                                       for pair in g['same_question_counterfactual_pairs']],
                'counterfactual_pairs_both_correct': sum(
                    g['counterfactual_pairs_both_correct'] for g in groups),
                'fit_memory_dependence_passed': all(g['fit_memory_dependence_passed']
                                                    for g in groups),
                'selection': 'all_four_predeclared_groups_required', 'generalization_measured': False}

    def evaluate(self, rows, batches, scopes):
        if not self.fitting_closed or self.targets or self.teacher_evidence or self.teacher_qualified:
            raise ValueError('development requires frozen fitting and evicted teacher targets')
        groups = []
        for start in self._complete_groups(rows, DEVELOPMENT_CASES, batches, scopes):
            end = start + GROUP_SIZE
            groups.append(DiagnosticEngineV12.evaluate(
                self, rows[start:end], batches[start:end], scopes[start:end]))
        per_rule_family = {}
        for start, group in zip(range(0, DEVELOPMENT_CASES, GROUP_SIZE), groups):
            family = rows[start]['rule_family']
            per_rule_family[family] = {
                'criterion_group': rows[start]['criterion_group'],
                'correct': group['correct'], 'false_recall': group['false_recall'],
                'useful_recall_passed': group['useful_recall_passed']}
        return {'rows': [row for group in groups for row in group['rows']],
                'correct': {control: sum(g['correct'][control] for g in groups)
                            for control in CONTROLS},
                'false_recall': {control: sum(g['false_recall'][control] for g in groups)
                                 for control in CONTROLS},
                'per_rule_family': per_rule_family,
                'groups': [{k: v for k, v in group.items() if k != 'rows'} for group in groups],
                'useful_recall_passed': all(g['useful_recall_passed'] for g in groups),
                'selection': 'both_predeclared_groups_required_no_pooling_escape',
                'generalization_scope': ('two_fixed_synthetic_rules_new_entities_one_seed_'
                                         'feasibility_only')}
