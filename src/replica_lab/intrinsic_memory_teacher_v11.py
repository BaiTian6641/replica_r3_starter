"""Complete broader teacher cache; unchanged eight-world criteria for every group."""
import json
from pathlib import Path

import torch

from .contextual_teacher_objective_v1 import TeacherTarget, make_teacher_target
from .intrinsic_memory_teacher_v10 import TeacherEngine
from .intrinsic_memory_diagnostic_v5 import DiagnosticEngine, CONTROLS
from .intrinsic_memory_budget_v5 import atomic
from .intrinsic_memory_execution_v5 import hash_file
from .intrinsic_memory_coverage_v11 import FIT_CASES, DEVELOPMENT_CASES, GROUP_SIZE

CACHE = 'replica.intrinsic_memory.teacher_cache.v11'
CACHE_LIMIT = 2 * 1024**2


class CoverageTeacherEngine(TeacherEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_cases = FIT_CASES

    def prepare_teacher(self, rows, scopes):
        if (self.teacher_qualified or self.targets or self.teacher_evidence or self.fitting_closed
                or len(rows) != self.fit_cases or len(scopes) != self.fit_cases
                or len({r['world_id'] for r in rows}) != self.fit_cases):
            raise ValueError('one complete declared fitting teacher stage required')
        for row, scope in zip(rows, scopes):
            with self.measured('teacher_complete_context_qualification'):
                answer = self.generate(row, None, scope, control='exact_text')
            answer.pop('first_logits')
            self.teacher_evidence.append({'world_id': row['world_id'], **answer})
            with self.measured('teacher_qualification_evidence_write', writes=1) as event:
                receipt = self.output/'teacher_qualification.json'
                atomic(receipt, {
                    'rows': self.teacher_evidence, 'complete': len(self.teacher_evidence) == self.fit_cases,
                    'passed': len(self.teacher_evidence) == self.fit_cases and all(
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
        if not self.teacher_qualified or len(self.targets) != self.fit_cases or len(self.teacher_evidence) != self.fit_cases:
            raise ValueError('incomplete teacher targets cannot be published')
        payload = {'format': CACHE, 'objective': self.objective.identity(self.spec),
                   'base_sha256': self.base_identity, 'fit_data_sha256': self.fit_identity,
                   'fit_cases': self.fit_cases, 'qualification': self.teacher_evidence,
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
        if (self.teacher_qualified or self.targets or self.teacher_evidence or self.fitting_closed or path.is_symlink()
                or not path.is_file() or path.stat().st_size != pin['bytes']
                or not 0 < pin['bytes'] <= CACHE_LIMIT or hash_file(path) != pin['sha256']):
            raise ValueError('teacher cache identity, size or lifecycle differs')
        with self.measured('detached_teacher_cache_restore') as event:
            payload = json.loads(path.read_text())
            event['artifact_read_bytes'] = path.stat().st_size
            if (payload.get('format') != CACHE or payload.get('objective') != self.objective.identity(self.spec)
                    or payload.get('base_sha256') != self.base_identity
                    or payload.get('fit_data_sha256') != self.fit_identity
                    or payload.get('fit_cases') != self.fit_cases
                    or len(rows) != self.fit_cases or len(payload.get('qualification', [])) != self.fit_cases
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
            groups.append(DiagnosticEngine.fitting_probes(self, rows[start:end], sources[start:end],
                                                          scopes[start:end], stores[start:end]))
        return {'format': 'replica.intrinsic_memory.coverage_fit_probes.v11',
                'rows': [row for group in groups for row in group['rows']],
                'correct': {control: sum(g['correct'][control] for g in groups) for control in CONTROLS},
                'groups': [{k: v for k, v in group.items() if k != 'rows'} for group in groups],
                'same_question_counterfactual_pairs': [pair for g in groups for pair in g['same_question_counterfactual_pairs']],
                'counterfactual_pairs_both_correct': sum(g['counterfactual_pairs_both_correct'] for g in groups),
                'fit_memory_dependence_passed': all(g['fit_memory_dependence_passed'] for g in groups),
                'selection': 'all_four_predeclared_groups_required', 'generalization_measured': False}

    def evaluate(self, rows, batches, scopes):
        if not self.fitting_closed or self.targets or self.teacher_evidence or self.teacher_qualified:
            raise ValueError('development requires frozen fitting and evicted teacher targets')
        groups = []
        for start in self._complete_groups(rows, DEVELOPMENT_CASES, batches, scopes):
            end = start + GROUP_SIZE
            groups.append(DiagnosticEngine.evaluate(self, rows[start:end], batches[start:end], scopes[start:end]))
        return {'rows': [row for group in groups for row in group['rows']],
                'correct': {control: sum(g['correct'][control] for g in groups) for control in CONTROLS},
                'false_recall': {control: sum(g['false_recall'][control] for g in groups) for control in CONTROLS},
                'groups': [{k: v for k, v in group.items() if k != 'rows'} for group in groups],
                'useful_recall_passed': all(g['useful_recall_passed'] for g in groups),
                'selection': 'both_predeclared_groups_required_no_pooling_escape',
                'generalization_scope': 'two_fixed_synthetic_rules_new_entities_one_seed_feasibility_only'}
