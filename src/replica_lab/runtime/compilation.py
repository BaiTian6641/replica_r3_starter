"""Thin runtime compilation for verified procedures (M1/M2 binding boundary).

The compiler binds a verified ledger trajectory to a typed ``ProcedureArtifact``,
executes it through the existing host procedure, then publishes it through the
existing ``ManifestRegistry``.  A candidate carries authenticated content for:

* the proposal that authorized it (full snapshot, not just an ID),
* the verification world inventory (family, split and per-case definitions),
* the executable procedure payload (typed artifact and its content digest),
* the identity snapshot and the parent lineage (manifest ID and body digest).

Publication and rollback are coordinated transactions across the artifact
catalog and the manifest registry.  A durable journal is written before either
side mutates; every mutation is either compensated in-process or completed by
an explicit :func:`recover_pending_transaction` call after a restart.  No model
is trained and no artifact certifies itself.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..contracts import fingerprint
from ..learning import ManifestRegistry
from ..memory_system.ledger import MemoryRecord
from .artifacts import (
    ArtifactRegistry, ProcedureArtifact, artifact_content_digest,
    artifact_from_dict, artifact_to_dict, atomic_write_text)
from .learning_proposal import LearningProposal, verify_proposal_snapshot
from .procedures import RepairProcedure

MANIFEST_SCHEMA = 'replica.procedure.manifest.v2'
LEGACY_MANIFEST_SCHEMA = 'replica.procedure.manifest.v1'
JOURNAL_FORMAT = 'replica.runtime.transaction.v1'
TRANSACTION_DIRECTORY = 'transactions'
JOURNAL_NAME = 'runtime_journal.json'
CATALOG_NAME = 'artifact_catalog.json'


class StaleSourceError(ValueError):
    """The evidence evaluated for a candidate changed before publication."""


class CandidateIntegrityError(ValueError):
    """A candidate, payload, inventory or lineage digest does not match."""


class ManifestIntegrityError(ValueError):
    """An archived manifest or payload is malformed or self-inconsistent."""


class TransactionError(RuntimeError):
    """Base class for coordinated publish/rollback failures."""


class InterruptedTransactionError(TransactionError):
    """One side committed and the other did not; explicit recovery is required."""


class RecoveryError(TransactionError):
    """A durable transaction journal cannot be replayed safely."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _nonnegative(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
            character not in '0123456789abcdef' for character in value):
        raise ValueError(f'{name} must be a lowercase SHA256')
    return value


def _optional_sha(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _sha(value, name)


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


# ---------------------------------------------------------------------------
# verification world inventory
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerificationCase:
    case_id: str
    role: str
    definition_digest: str
    workspace_digest: str

    def __post_init__(self):
        _text(self.case_id, 'case_id')
        _text(self.role, 'role')
        _sha(self.definition_digest, 'definition_digest')
        _sha(self.workspace_digest, 'workspace_digest')

    def to_dict(self) -> dict:
        return {'case_id': self.case_id, 'role': self.role,
                'definition_digest': self.definition_digest,
                'workspace_digest': self.workspace_digest}


_CASE_KEYS = {'case_id', 'role', 'definition_digest', 'workspace_digest'}
_INVENTORY_KEYS = {'split', 'family_id', 'cases', 'digest'}


@dataclass(frozen=True)
class WorldInventory:
    """The exact verification worlds a candidate was independently run on."""
    split: str
    family_id: str
    cases: tuple[VerificationCase, ...]

    def __post_init__(self):
        _text(self.split, 'split')
        _text(self.family_id, 'family_id')
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError('world inventory requires cases')
        if any(not isinstance(case, VerificationCase) for case in self.cases):
            raise ValueError('world inventory contains non-case values')
        identifiers = [case.case_id for case in self.cases]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('world inventory case IDs must be distinct')

    def _body(self) -> dict:
        return {'split': self.split, 'family_id': self.family_id,
                'cases': [case.to_dict() for case in self.cases]}

    @property
    def digest(self) -> str:
        return fingerprint(self._body())

    def to_dict(self) -> dict:
        return {**self._body(), 'digest': self.digest}

    @property
    def definitions(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(sorted((case.case_id, case.role, case.definition_digest)
                            for case in self.cases))

    def assert_matches_definitions(self, other: 'WorldInventory') -> None:
        """World definitions must be unchanged between verification and publish."""
        if not isinstance(other, WorldInventory):
            raise TypeError('world inventory must be a WorldInventory')
        if (self.split, self.family_id, self.definitions) != (
                other.split, other.family_id, other.definitions):
            raise CandidateIntegrityError(
                'verification world definitions changed after evaluation')

    @classmethod
    def build(cls, split: Any, family_id: Any,
              cases: Sequence[Mapping]) -> 'WorldInventory':
        values = []
        for case in cases:
            if not isinstance(case, Mapping) or not _CASE_KEYS <= set(case) or \
                    not set(case) <= _CASE_KEYS:
                raise ValueError('world case schema mismatch')
            values.append(VerificationCase(
                case['case_id'], case['role'], case['definition_digest'],
                case['workspace_digest']))
        return cls(split, family_id, tuple(values))

    @classmethod
    def from_dict(cls, value: Any) -> 'WorldInventory':
        if not isinstance(value, Mapping) or set(value) != _INVENTORY_KEYS:
            raise CandidateIntegrityError('world inventory schema mismatch')
        cases = value['cases']
        if not isinstance(cases, list):
            raise CandidateIntegrityError('world inventory cases must be a list')
        inventory = cls.build(value['split'], value['family_id'], cases)
        _sha(value['digest'], 'world inventory digest')
        if inventory.digest != value['digest']:
            raise CandidateIntegrityError(
                'world inventory content differs from its digest')
        return inventory


# ---------------------------------------------------------------------------
# compilation binding and payload
# ---------------------------------------------------------------------------
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
    parent_manifest_digest: str | None = None

    def __post_init__(self):
        for name in ('subject_id', 'branch_id', 'purpose', 'base_revision',
                     'identity_version'):
            _text(getattr(self, name), name)
        _nonnegative(self.cutoff, 'cutoff')
        _nonnegative(self.evidence_epoch, 'evidence_epoch')
        if self.parent_manifest_id is not None:
            _text(self.parent_manifest_id, 'parent_manifest_id')
        _optional_sha(self.parent_manifest_digest, 'parent_manifest_digest')
        if self.parent_manifest_id is None and self.parent_manifest_digest is not None:
            raise ValueError('a parent manifest digest requires a parent manifest ID')

    def identity_snapshot(self) -> dict:
        return {'subject_id': self.subject_id, 'branch_id': self.branch_id,
                'purpose': self.purpose, 'cutoff': self.cutoff,
                'evidence_epoch': self.evidence_epoch,
                'base_revision': self.base_revision,
                'identity_version': self.identity_version,
                'parent_manifest_id': self.parent_manifest_id,
                'parent_manifest_digest': self.parent_manifest_digest}

    @property
    def identity_digest(self) -> str:
        return fingerprint(self.identity_snapshot())

    def owner(self) -> dict:
        return {'subject_id': self.subject_id, 'branch_id': self.branch_id,
                'purpose': self.purpose, 'cutoff': self.cutoff,
                'evidence_epoch': self.evidence_epoch}


_IDENTITY_KEYS = set(CompilationBinding.__dataclass_fields__) | {'digest'}


def _proposal_snapshot(proposal: Any) -> dict | None:
    if proposal is None:
        return None
    if isinstance(proposal, LearningProposal):
        value = proposal.to_dict()
    elif isinstance(proposal, Mapping):
        value = dict(proposal)
    else:
        raise TypeError('proposal must be a LearningProposal or its snapshot')
    return verify_proposal_snapshot(value).to_dict()


@dataclass(frozen=True)
class CompilationEvaluation:
    accepted: bool
    candidate_digest: str
    parent_manifest_id: str | None
    artifact_digest: str
    cases: tuple[dict, ...]
    artifact_content_digest: str
    source_digest: str
    proposal_digest: str | None
    worlds_digest: str
    world_inventory: dict
    raw_cases: tuple[dict, ...] = ()

    def __post_init__(self):
        if type(self.accepted) is not bool:
            raise ValueError('evaluation acceptance must be boolean')
        _sha(self.candidate_digest, 'candidate_digest')
        _sha(self.artifact_digest, 'artifact_digest')
        _sha(self.artifact_content_digest, 'artifact_content_digest')
        _sha(self.source_digest, 'source_digest')
        _optional_sha(self.proposal_digest, 'proposal_digest')
        _sha(self.worlds_digest, 'worlds_digest')
        if self.parent_manifest_id is not None:
            _text(self.parent_manifest_id, 'parent_manifest_id')
        if not isinstance(self.cases, tuple) or any(
                not isinstance(case, Mapping) for case in self.cases):
            raise ValueError('evaluation cases must be mappings')
        if not isinstance(self.world_inventory, Mapping):
            raise ValueError('evaluation requires a world inventory')
        inventory = WorldInventory.from_dict(dict(self.world_inventory))
        if inventory.digest != self.worlds_digest:
            raise CandidateIntegrityError('evaluation worlds digest mismatch')
        if not isinstance(self.raw_cases, tuple) or any(
                not isinstance(case, Mapping) for case in self.raw_cases):
            raise ValueError('raw evaluation cases must be mappings')

    def publication(self) -> dict:
        return {
            'decision': 'accept' if self.accepted else 'reject',
            'candidate_digest': self.candidate_digest,
            'parent_manifest_id': self.parent_manifest_id,
            'artifact_digest': self.artifact_digest,
            'artifact_content_digest': self.artifact_content_digest,
            'source_digest': self.source_digest,
            'proposal_digest': self.proposal_digest,
            'worlds_digest': self.worlds_digest,
            'cases': [dict(case) for case in self.cases],
            'evaluation_scope': 'independent_fixture_execution',
        }


def _catalog_commit_state(catalog_path: Any, pre_state: Mapping,
                          post_state: Mapping) -> str:
    """Classify an uncertain catalog write using the durable file itself.

    ``atomic_write_text`` can fail after ``os.replace`` (directory fsync), so the
    only trustworthy source is the file on disk.
    """
    try:
        current = ArtifactRegistry.load(catalog_path).export()
    except (OSError, TypeError, ValueError):
        return 'unknown'
    post = ArtifactRegistry.import_state(post_state).export()
    pre = ArtifactRegistry.import_state(pre_state).export()
    if current == post:
        return 'committed'
    if current == pre:
        return 'pre_state'
    return 'unknown'


def _case_evidence(case: Mapping) -> dict:
    """Stable case fields that may enter a v2 manifest; raw receipts do not."""
    receipts = case.get('receipts') or ()
    last = receipts[-1] if receipts else None
    if not isinstance(last, Mapping):
        last = {}
    nested = last.get('receipt')
    content = nested if isinstance(nested, Mapping) else last
    return {
        'case_id': case.get('case_id'),
        'role': case.get('role'),
        'status': case.get('status'),
        'independent_passed': case.get('independent_passed'),
        'final_exit_code': case.get('final_exit_code'),
        'error': case.get('error'),
        'last_receipt': {
            'kind': last.get('kind') or content.get('kind'),
            'tool_status': last.get('status'),
            'tool_error': last.get('error'),
            'exit_code': content.get('exit_code'),
            'timed_out': content.get('timed_out'),
            'stdout_sha256': content.get('stdout_sha256'),
            'stderr_sha256': content.get('stderr_sha256'),
            'stdout_excerpt': content.get('stdout_excerpt', ''),
            'stderr_excerpt': content.get('stderr_excerpt', ''),
        } if last else None,
    }


def _assert_artifact(artifact: ProcedureArtifact) -> ProcedureArtifact:
    if not isinstance(artifact, ProcedureArtifact):
        raise TypeError('compiled payload must be a ProcedureArtifact')
    record = artifact_to_dict(artifact, content_digest=True)
    rebuilt = artifact_from_dict(record, content_digest=True)
    if rebuilt != artifact:
        raise CandidateIntegrityError('procedure payload is not canonical')
    return artifact


@dataclass(frozen=True)
class CompiledProcedure:
    artifact: ProcedureArtifact
    binding: CompilationBinding
    source_hashes: dict[str, str]
    proposal: Mapping | None = None
    legacy: bool = False

    def __post_init__(self):
        _assert_artifact(self.artifact)
        if not isinstance(self.binding, CompilationBinding):
            raise TypeError('compiled payload requires a CompilationBinding')
        expected = set(self.artifact.source_event_ids)
        if set(self.source_hashes) != expected:
            raise ValueError('source hashes must cover every source event exactly once')
        for record_id, digest in self.source_hashes.items():
            _text(record_id, 'source record ID')
            _sha(digest, 'source record hash')
        if self.proposal is not None:
            snapshot = _proposal_snapshot(self.proposal)
            object.__setattr__(self, 'proposal', MappingProxyType(snapshot))

    @property
    def proposal_digest(self) -> str | None:
        return self.proposal['digest'] if self.proposal is not None else None

    @classmethod
    def compile(cls, procedure: RepairProcedure, binding: CompilationBinding,
                records: Sequence[MemoryRecord], artifacts: ArtifactRegistry,
                *, proposal: Any = None,
                authorized_target: str | None = None) -> 'CompiledProcedure':
        """Bind exact verified ledger records, proposal and lineage to an artifact.

        ``authorized_target`` is the host-owned procedure target the proposal must
        name.  A proposal for another kind or target is never compiled.
        """
        source_hashes = _record_hashes(records)
        snapshot = _proposal_snapshot(proposal)
        if snapshot is not None:
            if snapshot['target_kind'] != 'procedure':
                raise CandidateIntegrityError(
                    'proposal does not authorize a procedure target')
            if authorized_target is None:
                raise CandidateIntegrityError(
                    'compiling a proposal requires the host-authorized target')
            if snapshot['target_id'] != authorized_target:
                raise CandidateIntegrityError(
                    'proposal target differs from the host-authorized target')
            if snapshot['parent_manifest_id'] != binding.parent_manifest_id:
                raise CandidateIntegrityError(
                    'proposal parent differs from the compilation binding')
            if not set(snapshot['source_event_ids']) <= set(source_hashes):
                raise CandidateIntegrityError(
                    'proposal sources are not compiled evidence')
        artifact = artifacts.add(procedure.artifact())
        return cls(artifact, binding, source_hashes, snapshot)

    # -- payload -------------------------------------------------------------
    def manifest_payload(self, *, world_inventory: Any = None,
                         artifact: ProcedureArtifact | None = None) -> dict:
        artifact = _assert_artifact(artifact if artifact is not None else self.artifact)
        worlds = None
        if world_inventory is not None:
            if isinstance(world_inventory, WorldInventory):
                worlds = world_inventory.to_dict()
            else:
                worlds = WorldInventory.from_dict(world_inventory).to_dict()
        sources = {record_id: _sha(digest, 'source record hash')
                   for record_id, digest in self.source_hashes.items()}
        return {
            'schema': MANIFEST_SCHEMA,
            'identity': {**self.binding.identity_snapshot(),
                         'digest': self.binding.identity_digest},
            'owner': self.binding.owner(),
            'base_revision': self.binding.base_revision,
            'identity_version': self.binding.identity_version,
            'parent_manifest_id': self.binding.parent_manifest_id,
            'parent_manifest_digest': self.binding.parent_manifest_digest,
            'source_hashes': sources,
            'source_digest': fingerprint(sources),
            'proposal': dict(self.proposal) if self.proposal is not None else None,
            'proposal_digest': self.proposal_digest,
            'worlds': worlds,
            'artifact': artifact_to_dict(artifact, content_digest=True),
        }

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, artifacts: ArtifactRegistry,
                 workspaces: Sequence[tuple[str, str, Any]], *,
                 inventory: WorldInventory | None = None) -> CompilationEvaluation:
        """Execute on caller-provided workspaces and independently verify."""
        if not workspaces:
            raise ValueError('at least one evaluation workspace is required')
        cases_input = []
        for entry in workspaces:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ValueError('workspaces must be (case_id, role, tools) tuples')
            case_id, role, tools = entry
            _text(case_id, 'case_id')
            _text(role, 'role')
            hashes = getattr(tools, 'workspace_hashes', None)
            if not callable(hashes):
                raise ValueError('evaluation tools must expose workspace hashes')
            cases_input.append((case_id, role, tools,
                                fingerprint(dict(hashes()))))
        identifiers = [case[0] for case in cases_input]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('evaluation case IDs must be distinct')
        if inventory is None:
            inventory = WorldInventory.build(
                'fixture', 'host_supplied',
                [{'case_id': case_id, 'role': role,
                  'definition_digest': fingerprint({'case_id': case_id,
                                                    'role': role}),
                  'workspace_digest': workspace}
                 for case_id, role, _tools, workspace in cases_input])
        if not isinstance(inventory, WorldInventory):
            raise TypeError('inventory must be a WorldInventory')
        if {case.case_id for case in inventory.cases} != set(identifiers):
            raise CandidateIntegrityError(
                'world inventory does not cover the evaluation cases')
        declared = {case.case_id: case for case in inventory.cases}
        actual_workspaces = {case_id: workspace
                             for case_id, _role, _tools, workspace in cases_input}
        for case_id, role, _tools, _workspace in cases_input:
            if declared[case_id].role != role:
                raise CandidateIntegrityError(
                    'world inventory roles differ from the evaluation cases')
            if declared[case_id].workspace_digest != actual_workspaces[case_id]:
                raise CandidateIntegrityError(
                    'world inventory workspace digest differs from the '
                    'executed workspace')
        if self.artifact.status != 'proposed':
            raise ValueError('only a proposed procedure can be evaluated')
        procedure = RepairProcedure(self.artifact.name,
                                    self.artifact.steps[0].argument_schema['path'],
                                    self.artifact.source_event_ids,
                                    version=self.artifact.version)
        cases = []
        for case_id, role, tools, _workspace in cases_input:
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
        evidence = tuple(_case_evidence(case) for case in cases)
        artifacts.transition(self.artifact.artifact_id, 'validated')
        accepted = all(case['status'] == 'succeeded' and case['independent_passed']
                       for case in cases)
        if not accepted:
            artifacts.transition(self.artifact.artifact_id, 'rejected')
        artifact = artifacts.get(self.artifact.artifact_id)
        payload = replace(self, artifact=artifact).manifest_payload(
            world_inventory=inventory)
        return CompilationEvaluation(
            accepted, fingerprint(payload), self.binding.parent_manifest_id,
            artifact.digest, evidence,
            artifact_content_digest(artifact), payload['source_digest'],
            payload['proposal_digest'], inventory.digest, inventory.to_dict(),
            tuple(dict(case) for case in cases))

    def assert_sources_current(self, current_hashes: Mapping[str, str]) -> None:
        if dict(current_hashes) != self.source_hashes:
            raise StaleSourceError('source evidence changed after evaluation')

    # -- publication ---------------------------------------------------------
    def publish(self, artifacts: ArtifactRegistry, manifests: ManifestRegistry,
                evaluation: CompilationEvaluation,
                current_source_hashes: Mapping[str, str], *,
                journal: 'TransactionJournal | None' = None,
                world_inventory: WorldInventory | None = None,
                catalog_path: Any = None) -> dict:
        if not isinstance(evaluation, CompilationEvaluation):
            raise TypeError('publish requires a CompilationEvaluation')
        if not evaluation.accepted:
            raise PermissionError('only an independently accepted procedure may be published')
        journal = journal if journal is not None else TransactionJournal(
            journal_path_for(manifests))
        catalog_path = Path(catalog_path) if catalog_path is not None \
            else catalog_path_for(manifests)
        artifacts.assert_consistent()
        if journal.read() is not None:
            raise TransactionError(
                'an unfinished runtime transaction is durable; call '
                'recover_pending_transaction before publishing again')

        artifact = artifacts.payload(self.artifact.artifact_id)
        if artifact.status != 'validated':
            raise PermissionError(
                'procedure is not in the independently validated state')
        if artifact_content_digest(artifact) != artifact_content_digest(self.artifact):
            raise CandidateIntegrityError(
                'executable procedure payload changed after compilation')
        self.assert_sources_current(current_source_hashes)
        if self.proposal is not None:
            snapshot = verify_proposal_snapshot(self.proposal)
            if snapshot.parent_manifest_id != self.binding.parent_manifest_id:
                raise CandidateIntegrityError(
                    'proposal lineage differs from the compilation binding')
            if not set(snapshot.source_event_ids) <= set(self.source_hashes):
                raise CandidateIntegrityError('proposal sources are not bound evidence')

        inventory = evaluation.world_inventory
        if world_inventory is not None:
            if not isinstance(world_inventory, WorldInventory):
                raise TypeError('world_inventory must be a WorldInventory')
            if world_inventory.digest != evaluation.worlds_digest:
                raise CandidateIntegrityError(
                    'verification world inventory differs from the evaluation')
        else:
            world_inventory = WorldInventory.from_dict(dict(inventory))
        if WorldInventory.from_dict(dict(inventory)).digest != \
                evaluation.worlds_digest:
            raise CandidateIntegrityError(
                'verification world inventory is not self-consistent')
        evaluation_cases = {case.get('case_id') for case in evaluation.cases}
        if evaluation_cases != {case.case_id for case in world_inventory.cases}:
            raise CandidateIntegrityError(
                'evaluation cases do not cover the verified world inventory')

        payload = replace(self, artifact=artifact).manifest_payload(
            world_inventory=inventory)
        if evaluation.candidate_digest != fingerprint(payload):
            raise CandidateIntegrityError(
                'evaluation does not match the compiled payload')
        if evaluation.parent_manifest_id != self.binding.parent_manifest_id:
            raise CandidateIntegrityError(
                'evaluation parent differs from the compiled binding')
        if evaluation.artifact_digest != artifact.digest or \
                evaluation.artifact_content_digest != artifact_content_digest(artifact):
            raise CandidateIntegrityError(
                'evaluation artifact digest differs from the catalog payload')
        if evaluation.source_digest != payload['source_digest'] or \
                evaluation.proposal_digest != payload['proposal_digest'] or \
                evaluation.worlds_digest != payload['worlds']['digest']:
            raise CandidateIntegrityError(
                'evaluation evidence digests differ from the compiled payload')

        active = active_manifest(manifests)
        if self.binding.parent_manifest_id is None:
            if active is not None:
                raise ValueError('active parent changed; reevaluate the candidate')
        else:
            if active is None:
                raise ValueError('the bound parent manifest is not active')
            if active['manifest_id'] != self.binding.parent_manifest_id:
                raise ValueError('active parent changed; reevaluate the candidate')
            if self.binding.parent_manifest_digest is not None and \
                    fingerprint(active) != self.binding.parent_manifest_digest:
                raise CandidateIntegrityError(
                    'active parent body differs from the bound lineage digest')

        publication = evaluation.publication()
        body = {'payload': payload, 'parent': self.binding.parent_manifest_id,
                'evaluation': publication}
        manifest_id = fingerprint(body)
        snapshot = artifacts.snapshot()
        # Durable pre-state baseline: recovery can complete or compensate after a
        # restart even if the catalog file was absent or belonged to another
        # process generation.
        artifacts.save(catalog_path)
        record = _transaction_record(
            'publish', snapshot, self.artifact.artifact_id,
            artifact_content_digest(artifact), 'validated', 'accepted',
            {'active_before': self.binding.parent_manifest_id,
             'expected_manifest_id': manifest_id,
             'published_manifest_id': manifest_id,
             'payload_digest': fingerprint(payload)})
        journal.write(record)
        try:
            manifest = manifests.publish(
                payload, publication, expected_parent=self.binding.parent_manifest_id)
            if manifest.get('manifest_id') != manifest_id:
                raise CandidateIntegrityError(
                    'manifest registry returned a different manifest body')
        except BaseException as error:
            artifacts.restore(snapshot)
            try:
                current = active_manifest(manifests)
            except ManifestIntegrityError:
                current = None
            if current is not None and current.get('manifest_id') == manifest_id:
                # The registry moved before the error surfaced.  The journal and
                # the durable pre-state catalog let recovery finish the commit.
                raise InterruptedTransactionError(
                    'the manifest registry committed before an error was raised; '
                    'call recover_pending_transaction') from error
            try:
                journal.clear()
            except OSError:
                pass
            raise
        try:
            artifacts.transition(self.artifact.artifact_id, 'accepted')
            artifacts.record_publication(
                self.artifact.artifact_id, manifest_id, fingerprint(payload))
            artifacts.assert_consistent()
            artifacts.save(catalog_path)
        except BaseException as error:
            state = _catalog_commit_state(catalog_path, snapshot,
                                          artifacts.snapshot())
            if state == 'committed':
                journal.finish(record)
                return manifest
            raise InterruptedTransactionError(
                'the manifest was published but the artifact catalog commit is '
                f'{state}; call recover_pending_transaction') from error
        journal.finish(record)
        return manifest

    def rollback(self, artifacts: ArtifactRegistry, manifests: ManifestRegistry,
                 ancestor_manifest_id: str, *,
                 journal: 'TransactionJournal | None' = None,
                 catalog_path: Any = None) -> dict:
        journal = journal if journal is not None else TransactionJournal(
            journal_path_for(manifests))
        catalog_path = Path(catalog_path) if catalog_path is not None \
            else catalog_path_for(manifests)
        if journal.read() is not None:
            raise TransactionError(
                'an unfinished runtime transaction is durable; call '
                'recover_pending_transaction before rolling back again')
        artifacts.assert_consistent()
        artifact = artifacts.get(self.artifact.artifact_id)
        if artifact is None or artifact.kind != 'procedure':
            raise KeyError(self.artifact.artifact_id)
        if artifact_content_digest(artifact) != artifact_content_digest(self.artifact):
            raise CandidateIntegrityError(
                'executable procedure payload changed after compilation')
        publication = artifacts.publication(self.artifact.artifact_id)
        if publication is None:
            raise PermissionError('the procedure has no publication to roll back')
        if artifact.status not in ('accepted', 'revoked'):
            raise PermissionError(
                'only an accepted or source-revoked procedure can be rolled back')
        if publication['artifact_content_digest'] != artifact_content_digest(artifact):
            raise CandidateIntegrityError(
                'publication content differs from the catalog artifact')
        if self.binding.parent_manifest_id is None or \
                ancestor_manifest_id != self.binding.parent_manifest_id:
            raise PermissionError('rollback target is not the bound parent')
        active = active_manifest(manifests)
        if active is None:
            raise ManifestIntegrityError('there is no active manifest to roll back')
        if active['manifest_id'] != publication['manifest_id']:
            raise CandidateIntegrityError(
                'active manifest is not the recorded publication')
        target = authenticated_ancestor(manifests, ancestor_manifest_id)
        if self.binding.parent_manifest_digest is not None and \
                fingerprint(target) != self.binding.parent_manifest_digest:
            raise CandidateIntegrityError(
                'rollback target body differs from the bound lineage digest')

        snapshot = artifacts.snapshot()
        # Persist the pre-transaction catalog so recovery can compensate
        # durably even if the process dies immediately after the journal write.
        artifacts.save(catalog_path)
        record = _transaction_record(
            'rollback', snapshot, self.artifact.artifact_id,
            artifact_content_digest(artifact), artifact.status, 'revoked',
            {'active_before': active['manifest_id'],
             'expected_manifest_id': ancestor_manifest_id,
             'published_manifest_id': active['manifest_id'],
             'payload_digest': publication['payload_digest']})
        journal.write(record)
        catalog_moved = artifact.status == 'accepted'
        if catalog_moved:
            try:
                artifacts.transition(self.artifact.artifact_id, 'revoked',
                                     reason='manifest rollback')
                artifacts.assert_consistent()
                artifacts.save(catalog_path)
            except BaseException as error:
                state = _catalog_commit_state(catalog_path, snapshot,
                                              artifacts.snapshot())
                if state == 'committed':
                    pass  # durable catalog already reflects the revocation
                elif state == 'pre_state':
                    artifacts.restore(snapshot)
                    try:
                        artifacts.save(catalog_path)
                    except BaseException as persist_error:
                        raise InterruptedTransactionError(
                            'catalog compensation could not be persisted; call '
                            'recover_pending_transaction') from persist_error
                    try:
                        journal.clear()
                    except OSError:
                        pass
                    raise
                else:
                    artifacts.restore(snapshot)
                    raise InterruptedTransactionError(
                        'rollback catalog commit outcome is unknown; call '
                        'recover_pending_transaction') from error
        try:
            restored = manifests.rollback(ancestor_manifest_id)
        except BaseException as error:
            try:
                current = active_manifest(manifests)
            except ManifestIntegrityError as read_error:
                raise InterruptedTransactionError(
                    'active manifest is unreadable after a failed rollback; '
                    'call recover_pending_transaction') from read_error
            if current is None or current['manifest_id'] != ancestor_manifest_id:
                # The manifest side did not move: undo the catalog mutation if
                # one happened, and keep the journal if compensation fails.
                if catalog_moved:
                    artifacts.restore(snapshot)
                    try:
                        artifacts.save(catalog_path)
                    except BaseException as persist_error:
                        raise InterruptedTransactionError(
                            'catalog compensation could not be persisted; call '
                            'recover_pending_transaction') from persist_error
                try:
                    journal.clear()
                except OSError:
                    pass
                raise
            journal.finish(record)
            return current
        if restored.get('manifest_id') != ancestor_manifest_id:
            raise InterruptedTransactionError(
                'manifest rollback returned an unexpected body; call '
                'recover_pending_transaction')
        authenticate_manifest(restored, expected_id=ancestor_manifest_id)
        journal.finish(record)
        return restored

    # -- reload --------------------------------------------------------------
    @classmethod
    def from_manifest(cls, manifest: Mapping, artifacts: ArtifactRegistry, *,
                      allow_legacy: bool = True,
                      published: bool = True) -> 'CompiledProcedure':
        """Authenticate an archived manifest and reconstruct the procedure.

        A manifest with an accept decision records a published revision, so by
        default the catalog receives the accepted artifact plus the active
        publication record bound to this manifest.  ``published=False`` keeps
        the payload status for audit-only reconstruction.
        """
        if not isinstance(manifest, Mapping):
            raise TypeError('manifest must be a mapping')
        authenticate_manifest(manifest)
        payload = manifest['payload']
        if not isinstance(payload, Mapping):
            raise ManifestIntegrityError('manifest payload must be a mapping')
        schema = payload.get('schema', LEGACY_MANIFEST_SCHEMA)
        try:
            if schema == LEGACY_MANIFEST_SCHEMA:
                if not allow_legacy:
                    raise ManifestIntegrityError(
                        'legacy manifest rejected by policy; migrate it explicitly')
                compiled = _compiled_from_legacy(manifest, payload)
            elif schema == MANIFEST_SCHEMA:
                compiled = _compiled_from_v2(manifest, payload)
            else:
                raise ManifestIntegrityError(
                    f'unknown procedure manifest schema {schema!r}')
        except ManifestIntegrityError:
            raise
        except (TypeError, ValueError) as error:
            raise ManifestIntegrityError(str(error)) from error
        payload_digest = fingerprint(payload)
        if published:
            catalog_artifact = replace(compiled.artifact, status='accepted')
        elif compiled.artifact.status == 'accepted':
            # An audit-only reload must not invent a publication; an accepted
            # procedure without one is not a legal catalog state.
            catalog_artifact = replace(compiled.artifact, status='validated')
        else:
            catalog_artifact = compiled.artifact
        if not published and artifacts.publication(
                compiled.artifact.artifact_id) is not None:
            raise ManifestIntegrityError(
                'audit-only reload cannot reuse a live publication record')
        existing = artifacts.get(catalog_artifact.artifact_id)
        if existing is None:
            artifacts.add(catalog_artifact)
            effective = catalog_artifact
        elif artifact_content_digest(existing) != \
                artifact_content_digest(catalog_artifact):
            raise ManifestIntegrityError(
                'catalog already holds a different artifact with this ID')
        else:
            effective = existing
            if published and effective.status != 'accepted':
                effective = artifacts.restore_status(effective.artifact_id,
                                                     'accepted')
        if published:
            publication = artifacts.publication(effective.artifact_id)
            if publication is None:
                artifacts.record_publication(
                    effective.artifact_id, manifest['manifest_id'],
                    payload_digest, reason='manifest reload', imported=True)
            elif publication['manifest_id'] != manifest['manifest_id'] or \
                    publication['payload_digest'] != payload_digest:
                raise ManifestIntegrityError(
                    'catalog publication differs from the archived manifest')
            elif publication['active'] is not True:
                raise ManifestIntegrityError(
                    'catalog publication was revoked; the manifest is not '
                    'currently publishable')
        artifacts.assert_consistent()
        if effective is compiled.artifact:
            return compiled
        return replace(compiled, artifact=effective)


# ---------------------------------------------------------------------------
# archived manifest authentication
# ---------------------------------------------------------------------------
_MANIFEST_KEYS = {'manifest_id', 'payload', 'parent', 'evaluation'}


def authenticate_manifest(manifest: Any, *, expected_id: str | None = None) -> dict:
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise ManifestIntegrityError('archived manifest body is malformed')
    body = {'payload': manifest['payload'], 'parent': manifest['parent'],
            'evaluation': manifest['evaluation']}
    recomputed = fingerprint(body)
    if manifest['manifest_id'] != recomputed:
        raise ManifestIntegrityError(
            'archived manifest body does not match its manifest ID')
    if expected_id is not None and recomputed != expected_id:
        raise ManifestIntegrityError('archived manifest ID is not the expected ID')
    return dict(manifest)


def load_manifest(manifests: ManifestRegistry, manifest_id: str) -> dict:
    _sha(manifest_id, 'manifest_id')
    path = Path(manifests.root) / f'{manifest_id}.json'
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestIntegrityError(
            f'manifest archive is unreadable: {manifest_id}') from error
    return authenticate_manifest(manifest, expected_id=manifest_id)


def active_manifest(manifests: ManifestRegistry) -> dict | None:
    """Read and authenticate ``active.json``; malformed state is never ignored."""
    try:
        active = manifests.active()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestIntegrityError('active manifest is unreadable') from error
    if active is None:
        return None
    return authenticate_manifest(active)


def authenticated_ancestor(manifests: ManifestRegistry,
                           target_manifest_id: str) -> dict:
    """Walk active -> parent chain, authenticating every body before use."""
    cursor = active_manifest(manifests)
    if cursor is None:
        raise ManifestIntegrityError('no active manifest')
    seen = {cursor['manifest_id']}
    while cursor['manifest_id'] != target_manifest_id:
        parent = cursor.get('parent')
        if not parent:
            raise ManifestIntegrityError('rollback target is not an ancestor')
        cursor = load_manifest(manifests, parent)
        if cursor['manifest_id'] in seen:
            raise ManifestIntegrityError('manifest ancestry contains a cycle')
        seen.add(cursor['manifest_id'])
        if len(seen) > 64:
            raise ManifestIntegrityError('manifest ancestry exceeds bounds')
    return cursor


def publication_artifact(manifest: Mapping) -> ProcedureArtifact:
    """Reconstruct the artifact a published manifest carries."""
    payload = manifest.get('payload')
    if not isinstance(payload, Mapping) or not isinstance(payload.get('artifact'),
                                                         Mapping):
        raise ManifestIntegrityError('manifest payload has no artifact')
    if payload.get('schema') == MANIFEST_SCHEMA:
        return artifact_from_dict(payload['artifact'], content_digest=True,
                                  kind='procedure', require_kind=True)
    return artifact_from_dict(payload['artifact'], kind='procedure')


def verify_catalog_publications(artifacts: ArtifactRegistry,
                                manifests: ManifestRegistry) -> None:
    """Authenticate every catalog publication against its archived manifest."""
    for item in artifacts.list():
        record = artifacts.publication(item.artifact_id)
        if record is None:
            continue
        manifest = load_manifest(manifests, record['manifest_id'])
        if fingerprint(manifest['payload']) != record['payload_digest']:
            raise ManifestIntegrityError(
                'catalog publication payload digest differs from its manifest')
        artifact = publication_artifact(manifest)
        if artifact.artifact_id != item.artifact_id:
            raise ManifestIntegrityError(
                'catalog publication references a different artifact')
        if artifact_content_digest(artifact) != record['artifact_content_digest']:
            raise ManifestIntegrityError(
                'catalog publication content differs from its manifest artifact')
        if record['active'] and item.status != 'accepted':
            raise ManifestIntegrityError(
                'active publication is bound to a non-accepted artifact')


def load_catalog(path: Any, manifests: ManifestRegistry) -> ArtifactRegistry:
    """Load a durable catalog and authenticate its publication records."""
    registry = ArtifactRegistry.load(path)
    verify_catalog_publications(registry, manifests)
    return registry


_LEGACY_PAYLOAD_KEYS = {'base_revision', 'identity_version', 'owner',
                        'parent_manifest_id', 'source_hashes', 'artifact'}
_LEGACY_EVALUATION_KEYS = {'decision', 'candidate_digest', 'parent_manifest_id',
                           'artifact_digest', 'cases', 'evaluation_scope'}
_V2_PAYLOAD_KEYS = {'schema', 'identity', 'owner', 'base_revision',
                    'identity_version', 'parent_manifest_id',
                    'parent_manifest_digest', 'source_hashes', 'source_digest',
                    'proposal', 'proposal_digest', 'worlds', 'artifact'}
_V2_EVALUATION_KEYS = {'decision', 'candidate_digest', 'parent_manifest_id',
                       'artifact_digest', 'artifact_content_digest',
                       'source_digest', 'proposal_digest', 'worlds_digest',
                       'cases', 'evaluation_scope'}
_V2_CASE_KEYS = {'case_id', 'role', 'status', 'independent_passed',
                 'final_exit_code', 'error', 'last_receipt'}
_RECEIPT_KEYS = {'kind', 'tool_status', 'tool_error', 'exit_code', 'timed_out',
                 'stdout_sha256', 'stderr_sha256', 'stdout_excerpt',
                 'stderr_excerpt'}
_OWNER_KEYS = {'subject_id', 'branch_id', 'purpose', 'cutoff', 'evidence_epoch'}


def _schema_mismatch(kind: str, expected: set, value: Mapping) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ManifestIntegrityError(
            f'{kind} schema mismatch; missing={missing} extra={extra}')


def _validate_evaluation(manifest: Mapping, expected: set, *,
                         payload_fingerprint: str) -> dict:
    evaluation = manifest['evaluation']
    if not isinstance(evaluation, Mapping):
        raise ManifestIntegrityError('manifest evaluation must be a mapping')
    _schema_mismatch('manifest evaluation', expected, evaluation)
    if evaluation['decision'] != 'accept':
        raise ManifestIntegrityError('archived manifest is not an acceptance')
    if evaluation['candidate_digest'] != payload_fingerprint:
        raise ManifestIntegrityError(
            'manifest evaluation does not match payload bytes')
    if evaluation['parent_manifest_id'] != manifest['parent']:
        raise ManifestIntegrityError(
            'manifest evaluation parent differs from the archived parent')
    _sha(evaluation['artifact_digest'], 'artifact_digest')
    if not isinstance(evaluation['cases'], list):
        raise ManifestIntegrityError('manifest evaluation cases must be a list')
    _text(evaluation['evaluation_scope'], 'evaluation_scope')
    return dict(evaluation)


def _compiled_from_legacy(manifest: Mapping, payload: Mapping) -> CompiledProcedure:
    _schema_mismatch('legacy manifest payload', _LEGACY_PAYLOAD_KEYS, payload)
    if not isinstance(payload['owner'], Mapping):
        raise ManifestIntegrityError('legacy owner must be a mapping')
    _schema_mismatch('legacy owner', _OWNER_KEYS, payload['owner'])
    artifact = artifact_from_dict(payload['artifact'], kind='procedure')
    if payload['parent_manifest_id'] != manifest['parent']:
        raise ManifestIntegrityError(
            'legacy payload parent differs from the archived parent')
    _text(payload['base_revision'], 'base_revision')
    _text(payload['identity_version'], 'identity_version')
    hashes = payload['source_hashes']
    if not isinstance(hashes, Mapping) or any(
            not isinstance(key, str) for key in hashes):
        raise ManifestIntegrityError('legacy source hashes must be a mapping')
    for record_id, digest in hashes.items():
        _text(record_id, 'source record ID')
        _sha(digest, 'source record hash')
    if set(hashes) != set(artifact.source_event_ids):
        raise ManifestIntegrityError(
            'legacy source hashes do not cover the artifact sources')
    evaluation = _validate_evaluation(
        manifest, _LEGACY_EVALUATION_KEYS,
        payload_fingerprint=fingerprint(dict(payload)))
    if evaluation['artifact_digest'] != artifact.digest:
        raise ManifestIntegrityError('artifact digest differs from the manifest')
    owner = payload['owner']
    binding = CompilationBinding(
        owner['subject_id'], owner['branch_id'], owner['purpose'],
        owner['cutoff'], owner['evidence_epoch'], payload['base_revision'],
        payload['identity_version'], payload['parent_manifest_id'])
    return CompiledProcedure(artifact, binding, dict(hashes), None, legacy=True)


def _validate_v2_case(case: Any) -> None:
    """Require complete, bounded nested evidence for each verified case."""
    if not isinstance(case, Mapping) or set(case) != _V2_CASE_KEYS:
        raise ManifestIntegrityError('evaluation case schema mismatch')
    _text(case['case_id'], 'case_id')
    _text(case['role'], 'role')
    _text(case['status'], 'status')
    if type(case['independent_passed']) is not bool:
        raise ManifestIntegrityError('evaluation acceptance must be boolean')
    if case['final_exit_code'] is not None and \
            type(case['final_exit_code']) is not int:
        raise ManifestIntegrityError('evaluation exit code must be an integer')
    if case['error'] is not None:
        _text(case['error'], 'case error')
    receipt = case['last_receipt']
    if receipt is None:
        if case['independent_passed'] is True:
            raise ManifestIntegrityError(
                'a passed evaluation case requires receipt evidence')
        return
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_KEYS:
        raise ManifestIntegrityError('evaluation receipt schema mismatch')
    for name in ('kind', 'tool_status', 'tool_error', 'stdout_excerpt',
                 'stderr_excerpt'):
        if receipt[name] is not None and not isinstance(receipt[name], str):
            raise ManifestIntegrityError(f'receipt {name} must be text')
    for name in ('exit_code', 'timed_out'):
        if receipt[name] is not None and \
                type(receipt[name]) not in (int, bool):
            raise ManifestIntegrityError(f'receipt {name} has the wrong type')
    for name in ('stdout_sha256', 'stderr_sha256'):
        if receipt[name] is not None:
            _sha(receipt[name], f'receipt {name}')


def _compiled_from_v2(manifest: Mapping, payload: Mapping) -> CompiledProcedure:
    _schema_mismatch('manifest payload', _V2_PAYLOAD_KEYS, payload)
    identity = payload['identity']
    if not isinstance(identity, Mapping):
        raise ManifestIntegrityError('identity snapshot must be a mapping')
    _schema_mismatch('identity snapshot', _IDENTITY_KEYS, identity)
    _sha(identity['digest'], 'identity digest')
    snapshot = {key: identity[key] for key in identity if key != 'digest'}
    if fingerprint(snapshot) != identity['digest']:
        raise ManifestIntegrityError(
            'identity snapshot content differs from its digest')
    binding = CompilationBinding(
        snapshot['subject_id'], snapshot['branch_id'], snapshot['purpose'],
        snapshot['cutoff'], snapshot['evidence_epoch'],
        snapshot['base_revision'], snapshot['identity_version'],
        snapshot['parent_manifest_id'], snapshot['parent_manifest_digest'])
    if binding.identity_digest != identity['digest']:
        raise ManifestIntegrityError('identity snapshot binding differs')
    if not isinstance(payload['owner'], Mapping):
        raise ManifestIntegrityError('owner must be a mapping')
    _schema_mismatch('owner', _OWNER_KEYS, payload['owner'])
    if dict(payload['owner']) != binding.owner():
        raise ManifestIntegrityError('owner differs from the identity snapshot')
    if payload['base_revision'] != binding.base_revision or \
            payload['identity_version'] != binding.identity_version or \
            payload['parent_manifest_id'] != binding.parent_manifest_id or \
            payload['parent_manifest_digest'] != binding.parent_manifest_digest:
        raise ManifestIntegrityError(
            'manifest lineage differs from the identity snapshot')
    if payload['parent_manifest_id'] != manifest['parent']:
        raise ManifestIntegrityError(
            'payload parent differs from the archived parent')
    _optional_sha(payload['parent_manifest_digest'], 'parent_manifest_digest')
    hashes = payload['source_hashes']
    if not isinstance(hashes, Mapping):
        raise ManifestIntegrityError('source hashes must be a mapping')
    for record_id, digest in hashes.items():
        _text(record_id, 'source record ID')
        _sha(digest, 'source record hash')
    _sha(payload['source_digest'], 'source_digest')
    if fingerprint({key: hashes[key] for key in hashes}) != payload['source_digest']:
        raise ManifestIntegrityError('source hashes differ from their digest')
    proposal = payload['proposal']
    if proposal is None:
        if payload['proposal_digest'] is not None:
            raise ManifestIntegrityError(
                'proposal digest is present without a proposal')
    else:
        snapshot_proposal = verify_proposal_snapshot(proposal)
        _optional_sha(payload['proposal_digest'], 'proposal_digest')
        if snapshot_proposal.digest != payload['proposal_digest']:
            raise ManifestIntegrityError(
                'proposal content differs from its digest')
        if snapshot_proposal.parent_manifest_id != binding.parent_manifest_id:
            raise ManifestIntegrityError('proposal lineage differs from the binding')
        if not set(snapshot_proposal.source_event_ids) <= set(hashes):
            raise ManifestIntegrityError('proposal sources are not bound evidence')
    worlds = WorldInventory.from_dict(payload['worlds'])
    if not isinstance(payload['artifact'], Mapping):
        raise ManifestIntegrityError('artifact must be a mapping')
    artifact = artifact_from_dict(payload['artifact'], content_digest=True,
                                  kind='procedure', require_kind=True)
    if set(hashes) != set(artifact.source_event_ids):
        raise ManifestIntegrityError(
            'source hashes do not cover the artifact sources')
    evaluation = _validate_evaluation(
        manifest, _V2_EVALUATION_KEYS,
        payload_fingerprint=fingerprint(dict(payload)))
    if evaluation['artifact_digest'] != artifact.digest or \
            evaluation['artifact_content_digest'] != artifact_content_digest(artifact):
        raise ManifestIntegrityError('artifact digest differs from the manifest')
    _sha(evaluation['source_digest'], 'source_digest')
    _optional_sha(evaluation['proposal_digest'], 'proposal_digest')
    _sha(evaluation['worlds_digest'], 'worlds_digest')
    if evaluation['source_digest'] != payload['source_digest'] or \
            evaluation['proposal_digest'] != payload['proposal_digest'] or \
            evaluation['worlds_digest'] != worlds.digest or \
            evaluation['worlds_digest'] != payload['worlds']['digest']:
        raise ManifestIntegrityError(
            'evaluation evidence digests differ from the payload')
    case_ids = []
    for case in evaluation['cases']:
        _validate_v2_case(case)
        case_ids.append(case['case_id'])
    if len(set(case_ids)) != len(case_ids):
        raise ManifestIntegrityError('evaluation case IDs are duplicated')
    if set(case_ids) != {case.case_id for case in worlds.cases}:
        raise ManifestIntegrityError(
            'evaluation cases do not cover the world inventory')
    return CompiledProcedure(artifact, binding, dict(hashes),
                             payload['proposal'])


# ---------------------------------------------------------------------------
# durable transaction journal and explicit recovery
# ---------------------------------------------------------------------------
def journal_path_for(manifests: ManifestRegistry) -> Path:
    return Path(manifests.root) / TRANSACTION_DIRECTORY / JOURNAL_NAME


def catalog_path_for(manifests: ManifestRegistry) -> Path:
    return Path(manifests.root) / TRANSACTION_DIRECTORY / CATALOG_NAME


_ARTIFACT_TXN_KEYS = {'artifact_id', 'content_digest', 'from_status', 'to_status'}
_MANIFEST_TXN_KEYS = {'active_before', 'expected_manifest_id',
                      'published_manifest_id', 'payload_digest'}
_RECORD_KEYS = {'format', 'transaction_id', 'operation', 'phase',
                'catalog_before', 'artifact', 'manifest'}
_STATUS = {'proposed', 'validated', 'accepted', 'rejected', 'deferred', 'revoked'}


def _transaction_record(operation, catalog_before, artifact_id, content_digest,
                        from_status, to_status, manifest) -> dict:
    body = {
        'format': JOURNAL_FORMAT,
        'operation': operation,
        'phase': 'prepared',
        'catalog_before': catalog_before,
        'artifact': {'artifact_id': artifact_id,
                     'content_digest': content_digest,
                     'from_status': from_status, 'to_status': to_status},
        'manifest': manifest,
    }
    return {'transaction_id': fingerprint(body), **body}


def _validate_record(value: Any) -> dict:
    if not isinstance(value, Mapping):
        raise RecoveryError('transaction journal must be a mapping')
    if set(value) != _RECORD_KEYS:
        raise RecoveryError('transaction journal schema mismatch')
    if value['format'] != JOURNAL_FORMAT:
        raise RecoveryError('unknown transaction journal format')
    if value['operation'] not in ('publish', 'rollback'):
        raise RecoveryError('unknown transaction operation')
    if value['phase'] not in ('prepared', 'committed'):
        raise RecoveryError('unknown transaction phase')
    _sha(value['transaction_id'], 'transaction_id')
    body = {key: value[key] for key in value if key != 'transaction_id'}
    if fingerprint(body) != value['transaction_id']:
        raise RecoveryError('transaction journal content differs from its ID')
    artifact = value['artifact']
    if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_TXN_KEYS:
        raise RecoveryError('journal artifact binding schema mismatch')
    _text(artifact['artifact_id'], 'artifact_id')
    _sha(artifact['content_digest'], 'artifact content digest')
    if artifact['from_status'] not in _STATUS or artifact['to_status'] not in _STATUS:
        raise RecoveryError('journal artifact status is unknown')
    manifest = value['manifest']
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_TXN_KEYS:
        raise RecoveryError('journal manifest binding schema mismatch')
    _optional_sha(manifest['active_before'], 'active_before')
    _optional_sha(manifest['expected_manifest_id'], 'expected_manifest_id')
    _optional_sha(manifest['published_manifest_id'], 'published_manifest_id')
    _optional_sha(manifest['payload_digest'], 'payload_digest')
    try:
        catalog = ArtifactRegistry.import_state(value['catalog_before'])
    except (TypeError, ValueError) as error:
        raise RecoveryError('journal catalog snapshot is invalid') from error
    recorded = catalog.get(artifact['artifact_id'])
    if recorded is None:
        raise RecoveryError('journal artifact is absent from its catalog snapshot')
    if artifact_content_digest(recorded) != artifact['content_digest']:
        raise RecoveryError('journal artifact content differs from its snapshot')
    if recorded.status != artifact['from_status']:
        raise RecoveryError('journal artifact status differs from its snapshot')
    return dict(value)


class TransactionJournal:
    """Single durable slot for the one in-flight catalog/manifest transaction."""

    def __init__(self, path: Any):
        self.path = Path(path)

    def read(self) -> dict | None:
        try:
            raw = self.path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return None
        except OSError as error:
            raise RecoveryError(f'transaction journal is unreadable: {self.path}') from error
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RecoveryError('transaction journal is corrupt') from error
        return _validate_record(value)

    def write(self, record: Mapping) -> None:
        validated = _validate_record(record)
        atomic_write_text(self.path, json.dumps(validated, indent=2, sort_keys=True)
                          + '\n')

    def finish(self, record: Mapping) -> None:
        """Mark the transaction committed, then best-effort remove the journal.

        Journal cleanup is housekeeping: a leftover record is re-validated and
        resolved by recovery, so a cleanup failure must not turn a committed
        transaction into an apparent rollback or duplicate publication.
        """
        body = {key: record[key] for key in record if key != 'transaction_id'}
        body['phase'] = 'committed'
        committed = {'transaction_id': fingerprint(body), **body}
        try:
            atomic_write_text(self.path, json.dumps(
                committed, indent=2, sort_keys=True) + '\n')
        except OSError:
            return
        try:
            self.clear()
        except OSError:
            return

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
        directory = self.path.parent
        if directory.exists():
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _persist_recovered_catalog(artifacts: ArtifactRegistry, catalog_path: Any,
                               journal_state: Mapping) -> None:
    """Persist recovered catalog state, tolerating a write that already landed.

    If the atomic write fails, the durable file is the authority: a matching
    post-state means the write committed, anything else keeps the journal.
    """
    post_state = artifacts.snapshot()
    try:
        artifacts.save(catalog_path)
    except BaseException as error:
        if _catalog_commit_state(catalog_path, journal_state,
                                 post_state) != 'committed':
            raise RecoveryError(
                'recovered catalog state could not be persisted; the journal '
                'is retained') from error


def _assert_publication_binding(artifacts: ArtifactRegistry,
                                artifact_id: str, manifest_id: str,
                                payload_digest: str, *, active: bool) -> dict:
    publication = artifacts.publication(artifact_id)
    if publication is None:
        raise RecoveryError('catalog publication is absent from the journal')
    if publication['manifest_id'] != manifest_id or \
            publication['payload_digest'] != payload_digest:
        raise RecoveryError('catalog publication differs from the journal')
    if publication['active'] is not active:
        raise RecoveryError('catalog publication activity differs from the journal')
    return publication


def _assert_archived_payload(manifest: Mapping, payload_digest: str) -> None:
    if fingerprint(manifest['payload']) != payload_digest:
        raise RecoveryError(
            'archived manifest payload differs from the journal binding')


def recover_pending_transaction(artifacts: ArtifactRegistry,
                                manifests: ManifestRegistry,
                                journal: TransactionJournal, *,
                                catalog_path: Any = None) -> dict | None:
    """Replay or compensate one durable transaction.  Explicit, never silent."""
    record = journal.read()
    if record is None:
        return None
    catalog_path = Path(catalog_path) if catalog_path is not None \
        else catalog_path_for(manifests)
    try:
        verify_catalog_publications(artifacts, manifests)
    except ManifestIntegrityError as error:
        raise RecoveryError(
            'catalog publication authentication failed; the journal is retained'
        ) from error
    operation = record['operation']
    artifact_id = record['artifact']['artifact_id']
    expected = record['manifest']
    try:
        active = active_manifest(manifests)
    except ManifestIntegrityError as error:
        raise RecoveryError(
            'active manifest is unreadable; the journal is retained') from error
    active_id = active['manifest_id'] if active is not None else None

    if record['phase'] == 'committed':
        if active_id != expected['expected_manifest_id']:
            raise RecoveryError(
                'committed journal does not match the active manifest')
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            raise RecoveryError('committed artifact is absent from the catalog')
        if artifact_content_digest(artifact) != record['artifact']['content_digest']:
            raise RecoveryError('committed artifact content changed')
        if operation == 'publish':
            _assert_publication_binding(
                artifacts, artifact_id, expected['expected_manifest_id'],
                expected['payload_digest'], active=True)
            _assert_archived_payload(active, expected['payload_digest'])
            required = 'accepted'
        else:
            _assert_publication_binding(
                artifacts, artifact_id, expected['published_manifest_id'],
                expected['payload_digest'], active=False)
            _assert_archived_payload(
                load_manifest(manifests, expected['published_manifest_id']),
                expected['payload_digest'])
            required = 'revoked'
        if artifact.status != required:
            raise RecoveryError(
                f'committed artifact is {artifact.status}, expected {required}')
        artifacts.assert_consistent()
        post_state = artifacts.snapshot()
        if _catalog_commit_state(catalog_path, record['catalog_before'],
                                 post_state) != 'committed':
            raise RecoveryError(
                'committed journal does not match the durable catalog; the '
                'journal is retained')
        journal.clear()
        return {'transaction_id': record['transaction_id'],
                'operation': operation, 'outcome': 'cleared_committed',
                'manifest_id': active_id, 'artifact_id': artifact_id,
                'artifact_status': artifact.status}

    if operation == 'publish':
        if active_id == expected['expected_manifest_id']:
            _assert_archived_payload(active, expected['payload_digest'])
            artifact = artifacts.get(artifact_id)
            if artifact is None:
                raise RecoveryError('published artifact is absent from the catalog')
            if artifact_content_digest(artifact) != record['artifact']['content_digest']:
                raise RecoveryError('published artifact content changed')
            if artifact.status == 'validated':
                artifacts.transition(artifact_id, 'accepted')
            elif artifact.status != 'accepted':
                raise RecoveryError(
                    f'published artifact is {artifact.status}, expected accepted')
            if artifacts.publication(artifact_id) is None:
                artifacts.record_publication(
                    artifact_id, expected['expected_manifest_id'],
                    expected['payload_digest'], reason='recovered publication')
            _assert_publication_binding(
                artifacts, artifact_id, expected['expected_manifest_id'],
                expected['payload_digest'], active=True)
            artifacts.assert_consistent()
            _persist_recovered_catalog(artifacts, catalog_path,
                                       record['catalog_before'])
            journal.clear()
            return {'transaction_id': record['transaction_id'],
                    'operation': operation, 'outcome': 'completed_publish',
                    'manifest_id': expected['expected_manifest_id'],
                    'artifact_id': artifact_id,
                    'artifact_status': 'accepted'}
        if active_id == expected['active_before']:
            artifacts.restore(record['catalog_before'])
            artifacts.assert_consistent()
            _persist_recovered_catalog(artifacts, catalog_path,
                                       record['catalog_before'])
            journal.clear()
            return {'transaction_id': record['transaction_id'],
                    'operation': operation, 'outcome': 'aborted_publish',
                    'active_manifest_id': active_id,
                    'artifact_id': artifact_id}
        raise RecoveryError(
            'active manifest moved during an interrupted publication')

    if active_id == expected['expected_manifest_id']:
        _assert_archived_payload(
            load_manifest(manifests, expected['published_manifest_id']),
            expected['payload_digest'])
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            raise RecoveryError('rolled-back artifact is absent from the catalog')
        if artifact_content_digest(artifact) != record['artifact']['content_digest']:
            raise RecoveryError('rolled-back artifact content changed')
        if artifact.status == 'accepted':
            artifacts.transition(artifact_id, 'revoked', reason='manifest rollback')
        elif artifact.status != 'revoked':
            raise RecoveryError(
                f'rolled-back artifact is {artifact.status}, expected revoked')
        _assert_publication_binding(
            artifacts, artifact_id, expected['published_manifest_id'],
            expected['payload_digest'], active=False)
        artifacts.assert_consistent()
        _persist_recovered_catalog(artifacts, catalog_path,
                                   record['catalog_before'])
        journal.clear()
        return {'transaction_id': record['transaction_id'],
                'operation': operation, 'outcome': 'completed_rollback',
                'manifest_id': expected['expected_manifest_id'],
                'artifact_id': artifact_id, 'artifact_status': 'revoked'}
    if active_id == expected['published_manifest_id']:
        _assert_archived_payload(active, expected['payload_digest'])
        artifacts.restore(record['catalog_before'])
        artifacts.assert_consistent()
        _persist_recovered_catalog(artifacts, catalog_path,
                                   record['catalog_before'])
        journal.clear()
        return {'transaction_id': record['transaction_id'],
                'operation': operation, 'outcome': 'aborted_rollback',
                'active_manifest_id': active_id,
                'artifact_id': artifact_id}
    raise RecoveryError('active manifest moved during an interrupted rollback')
