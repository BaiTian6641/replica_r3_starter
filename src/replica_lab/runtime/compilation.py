"""Thin runtime compilation for verified procedures (M1).

The compiler binds a verified ledger trajectory to a typed ``ProcedureArtifact``,
executes it through the existing host procedure, then publishes it through the
existing atomic ``ManifestRegistry``. Source and parent state are checked again
at publication. No model is trained and no artifact certifies itself.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import fingerprint
from ..learning import ManifestRegistry
from ..memory_system.ledger import MemoryRecord
from .artifacts import ArtifactRegistry, ProcedureArtifact, ProcedureStep
from .procedures import RepairProcedure


class StaleSourceError(ValueError):
    """The evidence evaluated for a candidate changed before publication."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _nonnegative(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _record_hashes(records: Sequence[MemoryRecord]) -> dict[str, str]:
    result = {}
    for record in records:
        if not isinstance(record, MemoryRecord):
            raise TypeError('compilation sources must be MemoryRecord instances')
        if record.authenticity not in {'verified_source', 'observed_tool'}:
            raise ValueError('procedure sources must be verified observations')
        if record.record_id in result:
            raise ValueError('duplicate source record')
        result[record.record_id] = record.record_hash
    return result


@dataclass(frozen=True)
class CompilationBinding:
    subject_id: str
    branch_id: str
    purpose: str
    cutoff: int
    evidence_epoch: int
    base_revision: str
    identity_version: str
    parent_manifest_id: str | None = None

    def __post_init__(self):
        for name in ('subject_id', 'branch_id', 'purpose', 'base_revision', 'identity_version'):
            _text(getattr(self, name), name)
        _nonnegative(self.cutoff, 'cutoff')
        _nonnegative(self.evidence_epoch, 'evidence_epoch')
        if self.parent_manifest_id is not None:
            _text(self.parent_manifest_id, 'parent_manifest_id')


@dataclass(frozen=True)
class CompilationEvaluation:
    accepted: bool
    candidate_digest: str
    parent_manifest_id: str | None
    artifact_digest: str
    cases: tuple[dict, ...]

    def publication(self) -> dict:
        return {
            'decision': 'accept' if self.accepted else 'reject',
            'candidate_digest': self.candidate_digest,
            'parent_manifest_id': self.parent_manifest_id,
            'artifact_digest': self.artifact_digest,
            'cases': list(self.cases),
            'evaluation_scope': 'independent_fixture_execution',
        }


@dataclass(frozen=True)
class CompiledProcedure:
    artifact: ProcedureArtifact
    binding: CompilationBinding
    source_hashes: dict[str, str]

    def __post_init__(self):
        if not isinstance(self.artifact, ProcedureArtifact):
            raise TypeError('compiled payload must be a ProcedureArtifact')
        if not isinstance(self.binding, CompilationBinding):
            raise TypeError('compiled payload requires a CompilationBinding')
        expected = set(self.artifact.source_event_ids)
        if set(self.source_hashes) != expected:
            raise ValueError('source hashes must cover every source event exactly once')
        for record_id, digest in self.source_hashes.items():
            _text(record_id, 'source record ID')
            _text(digest, 'source record hash')

    @classmethod
    def compile(cls, procedure: RepairProcedure, binding: CompilationBinding,
                records: Sequence[MemoryRecord], artifacts: ArtifactRegistry):
        """Bind exact verified ledger records to a new typed artifact."""
        source_hashes = _record_hashes(records)
        artifact = artifacts.add(procedure.artifact())
        return cls(artifact, binding, source_hashes)

    def manifest_payload(self) -> dict:
        artifact = self.artifact
        return {
            'base_revision': self.binding.base_revision,
            'identity_version': self.binding.identity_version,
            'owner': {
                'subject_id': self.binding.subject_id,
                'branch_id': self.binding.branch_id,
                'purpose': self.binding.purpose,
                'cutoff': self.binding.cutoff,
                'evidence_epoch': self.binding.evidence_epoch,
            },
            'parent_manifest_id': self.binding.parent_manifest_id,
            'source_hashes': dict(self.source_hashes),
            'artifact': {
                'artifact_id': artifact.artifact_id,
                'name': artifact.name,
                'version': artifact.version,
                'preconditions': list(artifact.preconditions),
                'steps': [step.to_dict() for step in artifact.steps],
                'tests': list(artifact.tests),
                'source_event_ids': list(artifact.source_event_ids),
                'producer_versions': dict(artifact.producer_versions),
                'parent_id': artifact.parent_id,
                'status': artifact.status,
                'created_at': artifact.created_at,
                'expires_at': artifact.expires_at,
                'digest': artifact.digest,
            },
        }

    def evaluate(self, artifacts: ArtifactRegistry,
                 workspaces: Sequence[tuple[str, str, Any]]) -> CompilationEvaluation:
        """Execute on caller-provided workspaces and independently verify."""
        if not workspaces:
            raise ValueError('at least one evaluation workspace is required')
        if self.artifact.status != 'proposed':
            raise ValueError('only a proposed procedure can be evaluated')
        procedure = RepairProcedure(self.artifact.name, self.artifact.steps[0].argument_schema['path'],
                                    self.artifact.source_event_ids, version=self.artifact.version)
        cases = []
        for case_id, role, tools in workspaces:
            run = procedure.execute(tools)
            verified = tools.verify_success()
            passed = verified.get('passed') is True
            cases.append({
                'case_id': case_id,
                'role': role,
                'status': run.status,
                'independent_passed': passed,
                'final_exit_code': run.final_exit_code,
                'receipts': list(run.receipts),
                'error': run.error,
            })
        artifacts.transition(self.artifact.artifact_id, 'validated')
        accepted = all(case['status'] == 'succeeded' and case['independent_passed']
                       for case in cases)
        artifacts.transition(self.artifact.artifact_id, 'accepted' if accepted else 'rejected')
        artifact = artifacts.get(self.artifact.artifact_id)
        payload = replace(self, artifact=artifact).manifest_payload()
        return CompilationEvaluation(
            accepted, fingerprint(payload), self.binding.parent_manifest_id,
            artifact.digest, tuple(cases))

    def assert_sources_current(self, current_hashes: Mapping[str, str]) -> None:
        if dict(current_hashes) != self.source_hashes:
            raise StaleSourceError('source evidence changed after evaluation')

    def publish(self, artifacts: ArtifactRegistry, manifests: ManifestRegistry,
                evaluation: CompilationEvaluation,
                current_source_hashes: Mapping[str, str]) -> dict:
        if not evaluation.accepted:
            raise PermissionError('only an independently accepted procedure may be published')
        artifact = artifacts.payload(self.artifact.artifact_id)
        if artifact.status != 'accepted':
            raise PermissionError('procedure is not accepted in the artifact registry')
        self.assert_sources_current(current_source_hashes)
        payload = replace(self, artifact=artifact).manifest_payload()
        if evaluation.candidate_digest != fingerprint(payload):
            raise ValueError('evaluation does not match the compiled payload')
        if evaluation.parent_manifest_id != self.binding.parent_manifest_id:
            raise ValueError('evaluation parent differs from the compiled binding')
        return manifests.publish(payload, evaluation.publication(),
                                 expected_parent=self.binding.parent_manifest_id)

    def rollback(self, artifacts: ArtifactRegistry, manifests: ManifestRegistry,
                 ancestor_manifest_id: str) -> dict:
        restored = manifests.rollback(ancestor_manifest_id)
        current = artifacts.get(self.artifact.artifact_id)
        if current.status == 'accepted':
            artifacts.transition(self.artifact.artifact_id, 'revoked',
                                 reason='manifest rollback')
        return restored

    @classmethod
    def from_manifest(cls, manifest: Mapping, artifacts: ArtifactRegistry) -> 'CompiledProcedure':
        """Reconstruct the exact accepted procedure after a process restart."""
        payload = manifest['payload']
        if manifest['evaluation']['candidate_digest'] != fingerprint(payload):
            raise ValueError('manifest evaluation does not match payload bytes')
        owner = payload['owner']
        binding = CompilationBinding(
            owner['subject_id'], owner['branch_id'], owner['purpose'], owner['cutoff'],
            owner['evidence_epoch'], payload['base_revision'], payload['identity_version'],
            manifest.get('parent'))
        value = payload['artifact']
        artifact = ProcedureArtifact(
            value['artifact_id'], value['name'], value['version'],
            tuple(value['preconditions']),
            tuple(ProcedureStep(step['tool'], step['argument_schema'],
                                step.get('purpose', ''))
                  for step in value['steps']),
            tuple(value['tests']), tuple(value['source_event_ids']),
            value['producer_versions'], parent_id=value['parent_id'], status=value['status'],
            created_at=value['created_at'], expires_at=value['expires_at'])
        if artifact.digest != value['digest']:
            raise ValueError('artifact digest differs from manifest')
        artifacts.add(artifact)
        return cls(artifact, binding, dict(payload['source_hashes']))
