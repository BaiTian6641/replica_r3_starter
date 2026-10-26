"""Bounded, isolated procedural-RSI plumbing over owned synthetic worlds.

One cycle practices a host-built repair procedure on a complete source world
family, compiles it with the existing verified-trajectory compiler, verifies it
on a separate world family the candidate has never seen, records a valid
rejection on a third family, publishes through the existing atomic manifest
registry, and finally rolls back to the protected ancestor.  Every world,
evaluator, acceptance rule, identity revision and resource limit is host-owned;
the candidate supplies only a procedure name and workspace entry.

This is deliberately *not* integrated RSI and not a neural cycle.  Neural
promotion is a separate fail-closed gate over independently pinned evidence and
has no software promotion path.  No longitudinal campaign is run here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
import re
from typing import Any, Mapping

from ..contracts import fingerprint
from ..learning import ManifestRegistry
from ..memory_system.ledger import MemoryRecord
from .artifacts import ArtifactRegistry, artifact_content_digest
from .compilation import (
    CandidateIntegrityError, CompilationBinding, CompilationEvaluation,
    CompiledProcedure, ManifestIntegrityError, StaleSourceError,
    TransactionJournal, WorldInventory, active_manifest, authenticate_manifest,
    catalog_path_for, journal_path_for, load_catalog, load_manifest,
    recover_pending_transaction, authenticated_ancestor)
from .director import DirectorBudget, LearningDirector
from .learning_cycle import CycleEvidence, review_cycle
from .learning_proposal import verify_proposal_snapshot
from .procedures import RepairProcedure
from .tools import CASE_FORMAT, FixtureWorkspace

CYCLE_FORMAT = 'replica.bounded_rsi.cycle.v1'
NEURAL_GATE_FORMAT = 'replica.bounded_rsi.neural_gate.v1'
NEURAL_EVIDENCE_FORMAT = 'replica.bounded_rsi.neural_evidence.v1'
PERCEPTION_FORMAT = 'replica.vision_qualification_score.v1'
MEMORY_FORMAT = 'replica.memory_reliability.verdict.v1'
QUALIFICATION_SCOPE = 'procedure_plumbing_only'
SPLITS = ('source', 'verification', 'rejection')
GATES = ('perception', 'memory', 'correction', 'retention')
APP_SETTINGS = 'workspace/app_settings.py'
PROCEDURE_TARGET = 'repair-config'
MAX_WORKSPACE_ENTRY = 512
_IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
_CONSTANT = re.compile(r'^[A-Z][A-Z0-9_]{0,63}$')


class BoundedRsiError(Exception):
    """A bounded cycle invariant was violated."""


class EvidenceIntegrityError(BoundedRsiError):
    """Pinned evidence is missing, changed or does not match its declared schema."""


class CandidateControlError(BoundedRsiError):
    """A candidate attempted to supply host authority or a foreign result."""


class ProtectedRevisionError(BoundedRsiError):
    """The protected base/identity revision or active parent was replaced."""


class ResourceLimitError(BoundedRsiError):
    """A host resource guard rejected the execution."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError(f'{name} must be nonempty text without NUL')
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
            character not in '0123456789abcdef' for character in value):
        raise ValueError(f'{name} must be a lowercase SHA256')
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


# ---------------------------------------------------------------------------
# host policy and resource guards
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResourceGuard:
    """Host-owned per-case execution bounds; never part of a candidate payload."""
    max_tool_calls: int = 8
    max_evaluator_runs: int = 5

    def __post_init__(self):
        for name in ('max_tool_calls', 'max_evaluator_runs'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class HostPolicy:
    """Protected identity/base revisions plus world-family selection."""
    subject_id: str
    branch_id: str
    purpose: str
    base_revision: str
    identity_version: str
    evidence_epoch: int
    source_family: str = 'source.queue_policy.v1'
    verification_family: str = 'verify.retry_policy.v1'
    rejection_family: str = 'reject.dual_constraint.v1'
    guard: ResourceGuard = ResourceGuard()

    def __post_init__(self):
        for name in ('subject_id', 'branch_id', 'purpose', 'base_revision',
                     'identity_version', 'source_family', 'verification_family',
                     'rejection_family'):
            _text(getattr(self, name), name)
        if type(self.evidence_epoch) is not int or self.evidence_epoch < 0:
            raise ValueError('evidence_epoch must be a nonnegative integer')
        if not isinstance(self.guard, ResourceGuard):
            raise TypeError('guard must be a ResourceGuard')
        families = (self.source_family, self.verification_family,
                    self.rejection_family)
        if len(set(families)) != 3:
            raise ValueError('source, verification and rejection families must differ')

    def to_dict(self) -> dict:
        return {
            'subject_id': self.subject_id,
            'branch_id': self.branch_id,
            'purpose': self.purpose,
            'base_revision': self.base_revision,
            'identity_version': self.identity_version,
            'evidence_epoch': self.evidence_epoch,
            'source_family': self.source_family,
            'verification_family': self.verification_family,
            'rejection_family': self.rejection_family,
            'guard': {'max_tool_calls': self.guard.max_tool_calls,
                      'max_evaluator_runs': self.guard.max_evaluator_runs},
        }


class GuardedWorkspace:
    """Delegates to a real FixtureWorkspace while enforcing host limits.

    The candidate procedure receives this object, so it can call tools but
    cannot raise the limits, swap the evaluator or read host-only receipts.
    """
    def __init__(self, workspace: FixtureWorkspace, guard: ResourceGuard):
        if not isinstance(workspace, FixtureWorkspace):
            raise TypeError('workspace must be a FixtureWorkspace')
        if not isinstance(guard, ResourceGuard):
            raise TypeError('guard must be a ResourceGuard')
        self.workspace = workspace
        self.guard = guard
        self.tool_calls = 0
        self.evaluator_runs = 0

    def _charge_tool(self):
        self.tool_calls += 1
        if self.tool_calls > self.guard.max_tool_calls:
            raise ResourceLimitError(
                f'tool-call budget exceeded ({self.guard.max_tool_calls})')

    def _charge_evaluator(self):
        self.evaluator_runs += 1
        if self.evaluator_runs > self.guard.max_evaluator_runs:
            raise ResourceLimitError(
                f'evaluator-run budget exceeded ({self.guard.max_evaluator_runs})')

    def read_file(self, arguments: dict) -> dict:
        self._charge_tool()
        return self.workspace.read_file(arguments)

    def write_file(self, arguments: dict) -> dict:
        self._charge_tool()
        return self.workspace.write_file(arguments)

    def set_constant(self, arguments: dict) -> dict:
        self._charge_tool()
        return self.workspace.set_constant(arguments)

    def run_test(self, arguments: dict) -> dict:
        self._charge_tool()
        self._charge_evaluator()
        return self.workspace.run_test(arguments)

    def verify_success(self, condition_ids=None) -> dict:
        self._charge_evaluator()
        return self.workspace.verify_success(condition_ids)

    def workspace_hashes(self) -> dict:
        return self.workspace.workspace_hashes()

    def counts(self) -> dict:
        return {'tool_calls': self.tool_calls, 'evaluator_runs': self.evaluator_runs}


# ---------------------------------------------------------------------------
# owned synthetic worlds (complete families, never split across phases)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskCase:
    case_id: str
    role: str
    goal: str
    constants: tuple[tuple[str, float, float], ...]

    def __post_init__(self):
        if not _IDENTIFIER.match(self.case_id or ''):
            raise ValueError('case_id must be an identifier')
        if self.role not in ('train', 'dev', 'heldout'):
            raise ValueError('unknown case role')
        _text(self.goal, 'goal')
        if not isinstance(self.constants, tuple) or not self.constants:
            raise ValueError('case requires declared constants')
        names = set()
        for item in self.constants:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError('constants must be (name, initial, required)')
            name, initial, required = item
            if not isinstance(name, str) or not _CONSTANT.match(name):
                raise ValueError('constant name must be an uppercase identifier')
            if name in names:
                raise ValueError('constant names must be distinct per case')
            names.add(name)
            _number(initial, 'initial constant')
            _number(required, 'required constant')
            if initial == required:
                raise ValueError('a practice case must start from a failing bound')

    @property
    def required(self) -> dict:
        return {name: required for name, _, required in self.constants}

    @property
    def initial(self) -> dict:
        return {name: initial for name, initial, _ in self.constants}

    def to_dict(self) -> dict:
        return {'case_id': self.case_id, 'role': self.role, 'goal': self.goal,
                'constants': [{'name': name, 'initial': initial,
                               'required': required}
                              for name, initial, required in self.constants]}


REQUIRED_ROLES = {'source': {'train'},
                  'verification': {'dev', 'heldout'},
                  'rejection': {'heldout'}}


@dataclass(frozen=True)
class WorldFamily:
    family_id: str
    split: str
    cases: tuple[TaskCase, ...]

    def __post_init__(self):
        if not _IDENTIFIER.match(self.family_id or ''):
            raise ValueError('family_id must be an identifier')
        if self.split not in SPLITS:
            raise ValueError('unknown world split')
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError('a world family requires cases')
        if any(not isinstance(case, TaskCase) for case in self.cases):
            raise ValueError('world families contain TaskCase values')
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError('case IDs must be distinct within a family')
        roles = {case.role for case in self.cases}
        required = REQUIRED_ROLES[self.split]
        if not required <= roles or not roles <= required:
            raise ValueError(
                f'{self.split} family requires exactly roles {sorted(required)}')

    def to_dict(self) -> dict:
        return {'family_id': self.family_id, 'split': self.split,
                'cases': [case.to_dict() for case in self.cases]}


@dataclass(frozen=True)
class WorldCatalog:
    """Complete task/world families partitioned before any expansion."""
    worlds: tuple[WorldFamily, ...]

    def __post_init__(self):
        if not isinstance(self.worlds, tuple) or not self.worlds:
            raise ValueError('a catalog requires world families')
        family_ids = [world.family_id for world in self.worlds]
        if len(set(family_ids)) != len(family_ids):
            raise ValueError('world family IDs must be distinct')
        case_ids = [case.case_id for world in self.worlds for case in world.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError('a case may belong to only one complete family')
        counts = {split: 0 for split in SPLITS}
        for world in self.worlds:
            counts[world.split] += 1
        for split in SPLITS:
            if counts[split] != 1:
                raise ValueError(f'exactly one complete {split} family is required')

    def by_split(self, split: str) -> WorldFamily:
        if split not in SPLITS:
            raise ValueError('unknown world split')
        matches = [world for world in self.worlds if world.split == split]
        if len(matches) != 1:
            raise ValueError(f'{split} must map to exactly one family')
        return matches[0]

    def family(self, family_id: str) -> WorldFamily:
        for world in self.worlds:
            if world.family_id == family_id:
                return world
        raise KeyError(family_id)

    def expanded(self, family_id: str, cases: tuple[TaskCase, ...]) -> 'WorldCatalog':
        """Append cases to one existing family without changing its split."""
        if not isinstance(cases, tuple) or not cases:
            raise ValueError('an expansion requires new cases')
        target = self.family(family_id)
        updated = tuple(
            WorldFamily(world.family_id, world.split,
                        world.cases + cases if world.family_id == family_id
                        else world.cases)
            for world in self.worlds)
        return WorldCatalog(updated)

    def assert_isolated(self, policy: HostPolicy | None = None) -> dict:
        source = self.by_split('source')
        verification = self.by_split('verification')
        rejection = self.by_split('rejection')
        if policy is not None:
            expected = (('source', policy.source_family),
                        ('verification', policy.verification_family),
                        ('rejection', policy.rejection_family))
            for split, family_id in expected:
                if self.by_split(split).family_id != family_id:
                    raise ValueError(
                        f'policy {split} family {family_id!r} is not the catalog family')
        source_ids = [case.case_id for case in source.cases]
        verification_ids = [case.case_id for case in verification.cases]
        rejection_ids = [case.case_id for case in rejection.cases]
        shared = (set(source_ids) & set(verification_ids)) | \
                 (set(source_ids) & set(rejection_ids)) | \
                 (set(verification_ids) & set(rejection_ids))
        family_ids = [world.family_id for world in self.worlds]
        if shared:
            raise ValueError('world families share case IDs')
        if len({source.family_id, verification.family_id,
                rejection.family_id}) != 3:
            raise ValueError('world families must be distinct')
        return {
            'families': {world.split: {'family_id': world.family_id,
                                       'cases': len(world.cases),
                                       'roles': sorted({case.role
                                                        for case in world.cases})}
                         for world in self.worlds},
            'family_ids': family_ids,
            'source_family_cases': len(source_ids),
            'verification_family_cases': len(verification_ids),
            'rejection_family_cases': len(rejection_ids),
            'independent_world_families': 2,
            'case_ids_disjoint': True,
            'source_case_ids': source_ids,
            'verification_case_ids': verification_ids,
            'rejection_case_ids': rejection_ids,
        }

    def to_dict(self) -> dict:
        return {'worlds': [world.to_dict() for world in self.worlds]}


def build_owned_worlds() -> WorldCatalog:
    """Three complete, disjoint synthetic world families.

    The source family practices failure-bound repair.  The verification family
    uses different constants and different bounds, so passing it requires
    reading the actual failure rather than the practice values.  The rejection
    family carries a second unsatisfied bound that one repair cannot fix.
    """
    source = WorldFamily('source.queue_policy.v1', 'source', (
        TaskCase('src_train_backoff', 'train',
                 'Restore the queue backoff bound.',
                 (('QUEUE_BACKOFF_MS', 40, 120),)),
        TaskCase('src_train_depth', 'train',
                 'Restore the queue depth bound.',
                 (('QUEUE_MAX_DEPTH', 4, 9),)),
    ))
    verification = WorldFamily('verify.retry_policy.v1', 'verification', (
        TaskCase('ver_holdout_window', 'dev',
                 'Restore the retry window bound.',
                 (('RETRY_WINDOW_SEC', 5, 15),)),
        TaskCase('ver_holdout_ceiling', 'heldout',
                 'Restore the retry ceiling bound.',
                 (('RETRY_CEILING', 2, 7),)),
    ))
    rejection = WorldFamily('reject.dual_constraint.v1', 'rejection', (
        TaskCase('rej_holdout_dual', 'heldout',
                 'Satisfy both lock bounds without host answers.',
                 (('LOCK_TIMEOUT', 1, 5), ('LOCK_CEILING', 1, 5))),
    ))
    return WorldCatalog((source, verification, rejection))


_EVALUATOR_TEMPLATE = '''"""Host-owned evaluator for one owned synthetic world case.

The candidate never sees this file through a tool: only ``workspace/`` is
readable and writable.  Failure text discloses the actual next bound, exactly
as the bounded repair baseline reads it.
"""
import importlib.util
from pathlib import Path
import sys

CASE_ROOT = Path(__file__).resolve().parent
WORKSPACE = CASE_ROOT / 'workspace' / 'app_settings.py'
CONDITIONS = %r
CHECKS = %r


def load_settings():
    spec = importlib.util.spec_from_file_location(
        'bounded_rsi_case_settings', WORKSPACE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    settings = load_settings()
    failures = []
    for name, minimum in CONDITIONS:
        actual = getattr(settings, name, None)
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or actual < minimum):
            failures.append((name, actual, minimum))
    if failures:
        name, actual, minimum = failures[0]
        print(f'{name} = {actual} but policy requires {name} >= {minimum}')
        return 1
    for name in CHECKS:
        if getattr(settings, name, None) is not True:
            print(f'{name} must be True')
            return 1
    print('policy ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
'''


def materialize_case(family: WorldFamily, case: TaskCase, destination) -> Path:
    """Write one host-owned fixture case into a fresh directory."""
    target = Path(destination)
    if target.exists():
        raise BoundedRsiError(f'case destination already exists: {target.name}')
    target.mkdir(parents=True)
    workspace = target / 'workspace'
    workspace.mkdir()
    lines = ['"""Owned synthetic settings module; host-generated."""']
    for name, initial, _ in case.constants:
        value = int(initial) if float(initial).is_integer() else initial
        lines.append(f'{name} = {value!r}')
    (workspace / 'app_settings.py').write_text('\n'.join(lines) + '\n',
                                               encoding='utf-8')
    conditions = tuple((name, float(required))
                       for name, _, required in case.constants)
    (target / 'evaluator.py').write_text(
        _EVALUATOR_TEMPLATE % (conditions, ()), encoding='utf-8')
    case_manifest = {
        'format': CASE_FORMAT,
        'case_id': case.case_id,
        'role': case.role,
        'goal': case.goal,
        'success_condition': {
            'condition_id': f'{case.case_id}-policy-satisfied',
            'description': f'Every declared bound in {family.family_id} holds.',
        },
        'constraints': ['Workspace-only edits.',
                        'The evaluator and case metadata are host-owned.'],
        'allowed_tools': ['read_file', 'set_constant', 'run_test'],
        'evaluator': 'evaluator.py',
        'workspace_files': [APP_SETTINGS],
        'workspace_entry': APP_SETTINGS,
        'variant': {'family': family.family_id, 'split': family.split,
                    'task': case.case_id},
        'reference_repair': {'constants': case.required},
    }
    (target / 'case.json').write_text(
        json.dumps(case_manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    return target


# ---------------------------------------------------------------------------
# episodic observations and stable receipts
# ---------------------------------------------------------------------------
def _receipt_view(result: Mapping) -> dict:
    receipt = result.get('receipt') or {}
    return {
        'exit_code': receipt.get('exit_code'),
        'timed_out': receipt.get('timed_out'),
        'stdout_sha256': receipt.get('stdout_sha256'),
        'stderr_sha256': receipt.get('stderr_sha256'),
        'stdout_excerpt': receipt.get('stdout_excerpt', ''),
        'stderr_excerpt': receipt.get('stderr_excerpt', ''),
    }


def _hashes_digest(before: Mapping, after: Mapping) -> str:
    return fingerprint({'before': dict(sorted(before.items())),
                        'after': dict(sorted(after.items()))})


@dataclass(frozen=True)
class Observation:
    record_id: str
    case_id: str
    world_family: str
    split: str
    stage: str
    status: str
    error: str | None
    final_exit_code: int | None
    before: dict
    after: dict
    tool_calls: int
    evaluator_runs: int
    receipt: dict
    raw_receipts: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            'record_id': self.record_id,
            'case_id': self.case_id,
            'world_family': self.world_family,
            'split': self.split,
            'stage': self.stage,
            'status': self.status,
            'error': self.error,
            'final_exit_code': self.final_exit_code,
            'workspace_hashes_digest': _hashes_digest(self.before, self.after),
            'tool_calls': self.tool_calls,
            'evaluator_runs': self.evaluator_runs,
            'receipt': dict(self.receipt),
        }


class EpisodicLedger:
    """Ordered observed-tool records plus separate raw observations.

    Only record hashes bind a candidate.  The text is the observed receipt, so
    evaluator answers or expected values are never placed into a memory write.
    """

    def __init__(self, policy: HostPolicy):
        if not isinstance(policy, HostPolicy):
            raise TypeError('ledger requires a HostPolicy')
        self.policy = policy
        self.epoch = policy.evidence_epoch
        self._clock = 0
        self._records: list[MemoryRecord] = []
        self._observations: list[Observation] = []

    def _record(self, record_id: str, text: str, *, supersedes=None,
                kind='tool') -> MemoryRecord:
        self._clock += 1
        record = MemoryRecord(
            record_id=record_id, subject_id=self.policy.subject_id,
            branch_id=self.policy.branch_id, text=text, observed_at=self._clock,
            event_at=self._clock, purpose=self.policy.purpose, kind=kind,
            supersedes=supersedes, authenticity='observed_tool')
        self._records.append(record)
        return record

    def append_observation(self, *, case_id: str, world_family: str, split: str,
                           stage: str, status: str, error: str | None,
                           final_exit_code: int | None, before: Mapping,
                           after: Mapping, tool_calls: int, evaluator_runs: int,
                           receipt: Mapping, raw_receipts: tuple = ()) -> Observation:
        if not _IDENTIFIER.match(case_id):
            raise ValueError('case_id must be an identifier')
        if stage not in ('observed_failure', 'observed_repair'):
            raise ValueError('unknown observation stage')
        record_id = f'obs.{self._clock + 1:04d}.{split}.{case_id}.{stage}'
        payload = {
            'case_id': case_id, 'world_family': world_family, 'split': split,
            'stage': stage, 'status': status, 'error': error,
            'final_exit_code': final_exit_code,
            'workspace_hashes_digest': _hashes_digest(before, after),
            'tool_calls': tool_calls, 'evaluator_runs': evaluator_runs,
            'receipt': dict(receipt),
        }
        record = self._record(record_id, json.dumps(
            payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False))
        observation = Observation(
            record_id=record.record_id, case_id=case_id, world_family=world_family,
            split=split, stage=stage, status=status, error=error,
            final_exit_code=final_exit_code, before=dict(before), after=dict(after),
            tool_calls=tool_calls, evaluator_runs=evaluator_runs,
            receipt=dict(receipt),
            raw_receipts=tuple(dict(item) for item in raw_receipts))
        self._observations.append(observation)
        return observation

    def append_correction(self, note: str) -> MemoryRecord:
        """Supersede the latest source record and advance the evidence epoch.

        A candidate compiled before this correction can no longer publish.
        """
        _text(note, 'note')
        if not self._records:
            raise BoundedRsiError('a correction requires prior source evidence')
        prior = self._records[-1]
        self.epoch += 1
        return self._record(f'correction.{self._clock + 1:04d}', note,
                            supersedes=prior.record_id, kind='correction')

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records)

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    def current_hashes(self) -> dict[str, str]:
        return {record.record_id: record.record_hash for record in self._records}


# ---------------------------------------------------------------------------
# practice, candidate compilation and independent verification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PracticeOutcome:
    family_id: str
    case_ids: tuple[str, ...]
    cases: tuple[dict, ...]
    records: tuple[MemoryRecord, ...]
    failure_record_ids: tuple[str, ...]
    repair_record_ids: tuple[str, ...]
    before_successes: int
    after_successes: int
    tool_calls: int
    evaluator_runs: int
    proposal: dict

    def to_dict(self) -> dict:
        return {
            'family_id': self.family_id,
            'case_ids': list(self.case_ids),
            'cases': [dict(case) for case in self.cases],
            'source_records': [{'record_id': record.record_id,
                                'record_hash': record.record_hash,
                                'authenticity': record.authenticity,
                                'kind': record.kind}
                               for record in self.records],
            'failure_record_ids': list(self.failure_record_ids),
            'repair_record_ids': list(self.repair_record_ids),
            'before_successes': self.before_successes,
            'after_successes': self.after_successes,
            'tool_calls': self.tool_calls,
            'evaluator_runs': self.evaluator_runs,
            'proposal': dict(self.proposal),
        }


@dataclass(frozen=True)
class CandidateSubmission:
    """Everything a candidate is allowed to supply: a name and an entry path."""
    name: str = 'repair-config'
    workspace_entry: str = APP_SETTINGS
    claimed_acceptance: bool = False
    claimed_source_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not _IDENTIFIER.match(self.name):
            raise ValueError('candidate name must be a bounded identifier')
        if not isinstance(self.workspace_entry, str) or not self.workspace_entry \
                or len(self.workspace_entry) > MAX_WORKSPACE_ENTRY:
            raise ValueError('candidate workspace_entry must be bounded text')
        if type(self.claimed_acceptance) is not bool:
            raise ValueError('claimed_acceptance must be boolean')
        if not isinstance(self.claimed_source_hashes, Mapping):
            raise ValueError('claimed_source_hashes must be a mapping')
        claims = {}
        for record_id, digest in self.claimed_source_hashes.items():
            _text(record_id, 'claimed source ID')
            _sha(digest, 'claimed source hash')
            claims[record_id] = digest
        object.__setattr__(self, 'claimed_source_hashes',
                           MappingProxyType(claims))

    def ignored_claims(self) -> tuple[str, ...]:
        claims = []
        if self.claimed_acceptance:
            claims.append('claimed_acceptance')
        if self.claimed_source_hashes:
            claims.append('claimed_source_hashes')
        return tuple(claims)


@dataclass(frozen=True)
class CompiledCandidate:
    compiled: CompiledProcedure
    submission: CandidateSubmission
    claims_ignored: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    proposal_id: str
    proposal_digest: str
    proposal: Mapping

    def __post_init__(self):
        if not isinstance(self.proposal, Mapping):
            raise TypeError('candidate requires an authenticated proposal snapshot')
        snapshot = verify_proposal_snapshot(dict(self.proposal))
        if snapshot.digest != self.proposal_digest:
            raise ValueError('candidate proposal content differs from its digest')
        if self.proposal_digest != self.compiled.proposal_digest:
            raise ValueError('candidate proposal differs from the compiled proposal')

    @property
    def artifact_id(self) -> str:
        return self.compiled.artifact.artifact_id

    @property
    def artifact_digest(self) -> str:
        return self.compiled.artifact.digest

    def to_dict(self) -> dict:
        return {
            'name': self.submission.name,
            'workspace_entry': self.submission.workspace_entry,
            'artifact_id': self.artifact_id,
            'artifact_digest_proposed': self.artifact_digest,
            'parent_manifest_id': self.compiled.binding.parent_manifest_id,
            'parent_manifest_digest': self.compiled.binding.parent_manifest_digest,
            'base_revision': self.compiled.binding.base_revision,
            'identity_version': self.compiled.binding.identity_version,
            'identity_digest': self.compiled.binding.identity_digest,
            'evidence_epoch': self.compiled.binding.evidence_epoch,
            'cutoff': self.compiled.binding.cutoff,
            'source_hashes': dict(self.compiled.source_hashes),
            'source_record_ids': list(self.source_record_ids),
            'proposal_id': self.proposal_id,
            'proposal_digest': self.proposal_digest,
            'proposal_target': f"{self.proposal['target_kind']}:{self.proposal['target_id']}",
            'claims_ignored': list(self.claims_ignored),
        }


@dataclass(frozen=True)
class VerificationOutcome:
    candidate: CompiledCandidate
    split: str
    family_id: str
    cases: tuple[dict, ...]
    evaluation: CompilationEvaluation
    tool_calls: int
    evaluator_runs: int
    world_inventory: dict
    raw_cases: tuple[dict, ...] = ()

    def __post_init__(self):
        inventory = WorldInventory.from_dict(self.world_inventory)
        if inventory.digest != self.evaluation.worlds_digest:
            raise ValueError('verification outcome inventory differs from evaluation')
        if inventory.family_id != self.family_id or inventory.split != self.split:
            raise ValueError('verification outcome inventory family differs')

    @property
    def accepted(self) -> bool:
        return self.evaluation.accepted is True

    @property
    def worlds_digest(self) -> str:
        return self.evaluation.worlds_digest

    def to_dict(self) -> dict:
        return {
            'candidate_name': self.candidate.submission.name,
            'artifact_id': self.candidate.artifact_id,
            'split': self.split,
            'family_id': self.family_id,
            'worlds_digest': self.worlds_digest,
            'accepted': self.accepted,
            'candidate_digest': self.evaluation.candidate_digest,
            'parent_manifest_id': self.evaluation.parent_manifest_id,
            'artifact_digest': self.evaluation.artifact_digest,
            'evaluation_scope': self.evaluation.publication()['evaluation_scope'],
            'cases': [dict(case) for case in self.cases],
            'tool_calls': self.tool_calls,
            'evaluator_runs': self.evaluator_runs,
        }


# ---------------------------------------------------------------------------
# neural gate: pinned evidence only, no software promotion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidencePin:
    gate: str
    relative_path: str
    format: str
    sha256: str | None = None

    def __post_init__(self):
        if self.gate not in GATES:
            raise ValueError('unknown neural evidence gate')
        path = Path(self.relative_path)
        if not self.relative_path or path.is_absolute() or '..' in path.parts:
            raise ValueError('evidence path must be a fresh relative path')
        _text(self.format, 'evidence format')
        if self.sha256 is not None:
            _sha(self.sha256, 'evidence sha256')

    @property
    def pinned(self) -> bool:
        return self.sha256 is not None

    def to_dict(self) -> dict:
        return {'gate': self.gate, 'relative_path': self.relative_path,
                'format': self.format, 'sha256': self.sha256,
                'pinned': self.pinned}


@dataclass(frozen=True)
class NeuralEvidencePins:
    perception: EvidencePin
    memory: EvidencePin
    correction: EvidencePin
    retention: EvidencePin

    def __post_init__(self):
        for name in GATES:
            pin = getattr(self, name)
            if not isinstance(pin, EvidencePin) or pin.gate != name:
                raise ValueError(f'{name} pin is missing or mislabelled')

    def items(self) -> tuple[EvidencePin, ...]:
        return tuple(getattr(self, name) for name in GATES)


PINNED_MODEL_REVISION = 'f97400052a43d642bbc6e9975e2397e3ae6a6b52'

# Host-owned pins.  The two existing artifacts are pinned by their actual
# bytes; correction and retention evidence are declared but not yet pinned,
# because no independent artifact passes those gates today.
DEFAULT_NEURAL_PINS = NeuralEvidencePins(
    perception=EvidencePin(
        'perception',
        'runs/minicpm5_vision_v0_20261025/score_restart_replay01/score_report.json',
        PERCEPTION_FORMAT,
        '090c5e9f9e6ee08423efe42b761b528875a00d61e4c243bc649ff6157a229443'),
    memory=EvidencePin(
        'memory',
        'reports/memory_reliability_v5_scorer_fix_eval02/verdict_final.json',
        MEMORY_FORMAT,
        '0a0e040d1102fab1d024b1a9c9300ab3a40eab9c0f281872f347c92479d29286'),
    correction=EvidencePin(
        'correction',
        'reports/bounded_rsi_neural_correction/verdict.json',
        NEURAL_EVIDENCE_FORMAT,
        None),
    retention=EvidencePin(
        'retention',
        'reports/bounded_rsi_neural_retention/verdict.json',
        NEURAL_EVIDENCE_FORMAT,
        None),
)


@dataclass(frozen=True)
class NeuralGate:
    """Fail-closed disposition over independently pinned neural evidence."""
    pins: NeuralEvidencePins = DEFAULT_NEURAL_PINS

    def __post_init__(self):
        if not isinstance(self.pins, NeuralEvidencePins):
            raise TypeError('pins must be NeuralEvidencePins')

    def _load(self, evidence_root: Path) -> dict:
        loaded = {}
        for pin in self.pins.items():
            record = {'gate': pin.gate, 'relative_path': pin.relative_path,
                      'pin_sha256': pin.sha256, 'status': 'not_pinned',
                      'evidence_sha256': None, 'data': None}
            path = evidence_root / pin.relative_path
            record['file_present'] = path.is_file()
            if not pin.pinned:
                loaded[pin.gate] = record
                continue
            if not path.is_file():
                raise EvidenceIntegrityError(
                    f'pinned {pin.gate} evidence is missing: {pin.relative_path}')
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            record['evidence_sha256'] = actual
            if actual != pin.sha256:
                raise EvidenceIntegrityError(
                    f'{pin.gate} evidence hash differs from its host pin')
            try:
                data = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EvidenceIntegrityError(
                    f'{pin.gate} evidence is not valid UTF-8 JSON: {error}')
            if not isinstance(data, dict):
                raise EvidenceIntegrityError(f'{pin.gate} evidence is not an object')
            if data.get('format') != pin.format:
                raise EvidenceIntegrityError(
                    f'{pin.gate} evidence format differs from its host pin')
            record['status'] = 'verified'
            record['data'] = data
            loaded[pin.gate] = record
        return loaded

    def assess(self, evidence_root) -> dict:
        root = Path(evidence_root)
        loaded = self._load(root)
        gates = {}
        blockers: list[str] = []

        perception = loaded['perception']
        observed = perception['data'] or {}
        passed = False
        if perception['status'] != 'verified':
            blockers.append('perception_evidence_not_pinned')
        elif observed.get('quality_pass') is not True:
            if observed.get('quality_pass') is None:
                blockers.append('perception_quality_not_established')
            else:
                blockers.append('perception_quality_failed')
        elif observed.get('accepted') is not True:
            blockers.append('perception_not_accepted')
        elif observed.get('engineering_pass') is not True:
            blockers.append('perception_engineering_failed')
        elif observed.get('model_revision') != PINNED_MODEL_REVISION:
            blockers.append('perception_model_revision_mismatch')
        else:
            passed = True
        gates['perception'] = {
            'status': perception['status'],
            'pin_sha256': perception['pin_sha256'],
            'evidence_sha256': perception['evidence_sha256'],
            'model_revision': observed.get('model_revision'),
            'quality_pass': observed.get('quality_pass'),
            'quality_claim': observed.get('quality_claim'),
            'accepted': observed.get('accepted'),
            'engineering_pass': observed.get('engineering_pass'),
            'passed': passed,
        }

        memory = loaded['memory']
        observed = memory['data'] or {}
        passed = False
        if memory['status'] != 'verified':
            blockers.append('learned_memory_evidence_not_pinned')
        elif observed.get('decision') == 'reject':
            blockers.append('learned_memory_rejected')
        elif observed.get('decision') != 'accept':
            blockers.append('learned_memory_not_accepted')
        elif observed.get('deployment') in ('diagnostic_only_not_published',
                                            'rejected', 'unpublished'):
            blockers.append('learned_memory_not_published')
        elif not isinstance(observed.get('checks'), Mapping) or \
                not observed['checks'] or not all(
                    value is True for value in observed['checks'].values()):
            blockers.append('learned_memory_checks_failed')
        else:
            passed = True
        gates['memory'] = {
            'status': memory['status'],
            'pin_sha256': memory['pin_sha256'],
            'evidence_sha256': memory['evidence_sha256'],
            'decision': observed.get('decision'),
            'deployment': observed.get('deployment'),
            'failed_checks': observed.get('failed_checks'),
            'passed': passed,
        }

        for gate in ('correction', 'retention'):
            item = loaded[gate]
            observed = item['data'] or {}
            passed = False
            checks = observed.get('checks')
            if item['status'] != 'verified':
                blockers.append(f'{gate}_evidence_not_pinned')
            elif observed.get('decision') != 'accept':
                blockers.append(f'{gate}_evidence_not_accepted')
            elif not isinstance(checks, Mapping) or not checks or \
                    not all(value is True for value in checks.values()):
                blockers.append(f'{gate}_checks_failed')
            else:
                passed = True
            gates[gate] = {
                'status': item['status'],
                'pin_sha256': item['pin_sha256'],
                'evidence_sha256': item['evidence_sha256'],
                'decision': observed.get('decision'),
                'checks': dict(checks) if isinstance(checks, Mapping) else None,
                'passed': passed,
            }

        decision = 'deferred' if blockers else 'deferred_host_review'
        return {
            'format': NEURAL_GATE_FORMAT,
            'decision': decision,
            'promotable': False,
            'blockers': blockers,
            'gates': gates,
            'software_promoted': False,
            'promotion_authority': 'host',
            'neural_training_performed': False,
            'neural_artifact_created': False,
            'scope': ('neural RSI promotion requires independently pinned accepted '
                      'perception, useful learned memory, correction and retention '
                      'evidence; no software check may promote a neural candidate'),
        }

    def to_dict(self) -> dict:
        return {'pins': [pin.to_dict() for pin in self.pins.items()]}


# ---------------------------------------------------------------------------
# the cycle
# ---------------------------------------------------------------------------
class BoundedRsiCycle:
    """One isolated practice/verify/reject/publish/rollback plumbing cycle."""

    def __init__(self, root, policy: HostPolicy, *,
                 catalog: WorldCatalog | None = None,
                 neural_gate: NeuralGate | None = None,
                 evidence_root=None):
        if not isinstance(policy, HostPolicy):
            raise TypeError('policy must be a HostPolicy')
        self.root = Path(root)
        if (self.root / 'manifests' / 'active.json').exists():
            raise BoundedRsiError(
                'output registry is already active; use a fresh isolated root')
        self.policy = policy
        self.catalog = catalog if catalog is not None else build_owned_worlds()
        if not isinstance(self.catalog, WorldCatalog):
            raise TypeError('catalog must be a WorldCatalog')
        self.worlds = self.catalog.assert_isolated(policy)
        self.neural_gate = neural_gate if neural_gate is not None else NeuralGate()
        if not isinstance(self.neural_gate, NeuralGate):
            raise TypeError('neural_gate must be a NeuralGate')
        self.evidence_root = Path(evidence_root) if evidence_root is not None \
            else Path(__file__).resolve().parents[3]
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifests = ManifestRegistry(self.root / 'manifests')
        self.transactions = TransactionJournal(journal_path_for(self.manifests))
        self.catalog_path = catalog_path_for(self.manifests)
        if self.transactions.read() is not None:
            raise BoundedRsiError(
                'an interrupted runtime transaction is durable; call '
                'BoundedRsiCycle.recover_interrupted first')
        self.artifacts = ArtifactRegistry()
        self.ledger = EpisodicLedger(policy)
        self._base: Mapping | None = None
        self._practice: PracticeOutcome | None = None
        self._proposal = None
        self._issued: dict[str, dict] = {}
        self._published_candidate: CompiledCandidate | None = None
        self._published_manifest: Mapping | None = None

    # -- explicit restart recovery -------------------------------------------
    @classmethod
    def recover_interrupted(cls, root, policy: HostPolicy, *,
                            catalog_path=None) -> dict:
        """Replay or compensate one durable publish/rollback transaction."""
        if not isinstance(policy, HostPolicy):
            raise TypeError('policy must be a HostPolicy')
        root = Path(root)
        manifests = ManifestRegistry(root / 'manifests')
        journal = TransactionJournal(journal_path_for(manifests))
        catalog = Path(catalog_path) if catalog_path is not None \
            else catalog_path_for(manifests)
        artifacts = load_catalog(catalog, manifests) if catalog.exists() \
            else ArtifactRegistry()
        try:
            result = recover_pending_transaction(
                artifacts, manifests, journal, catalog_path=catalog)
            active = active_manifest(manifests)
        except (CandidateIntegrityError, ManifestIntegrityError) as error:
            raise ProtectedRevisionError(str(error)) from error
        if active is not None:
            payload = active.get('payload') or {}
            if payload.get('base_revision') != policy.base_revision or \
                    payload.get('identity_version') != policy.identity_version:
                raise ProtectedRevisionError(
                    'recovered active manifest differs from the host policy')
        return {
            'recovery': result,
            'active_manifest_id': active['manifest_id'] if active is not None else None,
            'journal_present': journal.read() is not None,
            'catalog_path': str(catalog),
            'artifacts': [{
                'artifact_id': item.artifact_id,
                'kind': item.kind,
                'status': item.status,
                'publication': artifacts.publication(item.artifact_id),
            } for item in artifacts.list()],
        }

    # -- private candidate authority ----------------------------------------
    def _candidate_authority(self, candidate: CompiledCandidate) -> dict:
        """Canonical authority snapshot for one issued candidate.

        Status is deliberately excluded: the artifact registry advances the
        lifecycle, while proposal content, identity, lineage, sources and the
        executable payload stay fixed.
        """
        compiled = candidate.compiled
        artifact = compiled.artifact
        binding = compiled.binding
        submission = candidate.submission
        entry = artifact.steps[0].argument_schema.get('path')
        return {
            'artifact_id': artifact.artifact_id,
            'artifact_content_digest': artifact_content_digest(artifact),
            'artifact': {
                'name': artifact.name,
                'version': artifact.version,
                'preconditions': list(artifact.preconditions),
                'steps': [step.to_dict() for step in artifact.steps],
                'tests': list(artifact.tests),
                'source_event_ids': list(artifact.source_event_ids),
                'producer_versions': dict(artifact.producer_versions),
                'parent_id': artifact.parent_id,
                'created_at': artifact.created_at,
                'expires_at': artifact.expires_at,
            },
            'executable': {'entry': entry,
                           'steps': [step.to_dict() for step in artifact.steps],
                           'content_digest': artifact_content_digest(artifact)},
            'binding': {
                'subject_id': binding.subject_id,
                'branch_id': binding.branch_id,
                'purpose': binding.purpose,
                'cutoff': binding.cutoff,
                'evidence_epoch': binding.evidence_epoch,
                'base_revision': binding.base_revision,
                'identity_version': binding.identity_version,
                'parent_manifest_id': binding.parent_manifest_id,
                'parent_manifest_digest': binding.parent_manifest_digest,
                'identity_digest': binding.identity_digest,
            },
            'source_hashes': dict(compiled.source_hashes),
            'proposal': dict(candidate.proposal),
            'proposal_digest': candidate.proposal_digest,
            'submission': {
                'name': submission.name,
                'workspace_entry': submission.workspace_entry,
                'claimed_acceptance': submission.claimed_acceptance,
                'claimed_source_hashes': dict(submission.claimed_source_hashes),
            },
            'claims_ignored': list(candidate.claims_ignored),
            'source_record_ids': list(candidate.source_record_ids),
            'proposal_id': candidate.proposal_id,
            'payload_digest_proposed': fingerprint(compiled.manifest_payload()),
        }

    def _issue_candidate(self, candidate: CompiledCandidate) -> CompiledCandidate:
        artifact_id = candidate.artifact_id
        if artifact_id in self._issued:
            raise BoundedRsiError('candidate artifact ID was already issued')
        self._issued[artifact_id] = {
            'candidate': candidate,
            'authority_digest': fingerprint(
                self._candidate_authority(candidate)),
        }
        return candidate

    def _authenticate_issued_candidate(
            self, candidate: CompiledCandidate) -> CompiledCandidate:
        """Reject foreign, replaced, or mutated candidates before execution."""
        if not isinstance(candidate, CompiledCandidate):
            raise CandidateControlError('candidate must be a CompiledCandidate')
        issued = self._issued.get(candidate.artifact_id)
        if issued is None or issued['candidate'] is not candidate:
            raise CandidateControlError(
                'candidate was not issued unchanged by this cycle')
        try:
            current_authority = self._candidate_authority(candidate)
        except (TypeError, ValueError, AttributeError) as error:
            raise CandidateControlError(
                'candidate authority is unreadable after issuance') from error
        if fingerprint(current_authority) != issued['authority_digest']:
            raise CandidateControlError(
                'candidate authority changed after issuance')
        if self._practice is None or self._base is None:
            raise BoundedRsiError('candidate has no issued practice/base context')
        binding = candidate.compiled.binding
        expected_binding = {
            'subject_id': self.policy.subject_id,
            'branch_id': self.policy.branch_id,
            'purpose': self.policy.purpose,
            'cutoff': max(item.observed_at
                          for item in self._practice.records),
            'evidence_epoch': self.policy.evidence_epoch,
            'base_revision': self.policy.base_revision,
            'identity_version': self.policy.identity_version,
            'parent_manifest_id': self._base['manifest_id'],
            'parent_manifest_digest': self._base['manifest_digest'],
        }
        actual_binding = {
            'subject_id': binding.subject_id,
            'branch_id': binding.branch_id,
            'purpose': binding.purpose,
            'cutoff': binding.cutoff,
            'evidence_epoch': binding.evidence_epoch,
            'base_revision': binding.base_revision,
            'identity_version': binding.identity_version,
            'parent_manifest_id': binding.parent_manifest_id,
            'parent_manifest_digest': binding.parent_manifest_digest,
        }
        if actual_binding != expected_binding:
            raise ProtectedRevisionError(
                'candidate binding differs from protected policy/base')
        if binding.identity_digest != fingerprint(binding.identity_snapshot()):
            raise CandidateControlError('candidate identity snapshot is not canonical')
        proposal = verify_proposal_snapshot(candidate.proposal)
        if proposal.digest != candidate.proposal_digest or \
                proposal.digest != candidate.compiled.proposal_digest:
            raise CandidateControlError(
                'candidate proposal content differs from its digest')
        if proposal.parent_manifest_id != binding.parent_manifest_id:
            raise ProtectedRevisionError(
                'proposal lineage differs from the protected base')
        if proposal.target_kind != 'procedure':
            raise CandidateControlError(
                'only a procedure proposal may be compiled by this cycle')
        if proposal.target_id != PROCEDURE_TARGET:
            raise ProtectedRevisionError(
                'proposal target differs from the host-authorized procedure target')
        expected_hashes = {record.record_id: record.record_hash
                           for record in self._practice.records}
        if dict(candidate.compiled.source_hashes) != expected_hashes:
            raise StaleSourceError(
                'candidate source hashes differ from practice evidence')
        if not set(proposal.source_event_ids) <= set(expected_hashes):
            raise CandidateControlError(
                'proposal sources are not this cycle\'s evidence')
        if set(candidate.compiled.source_hashes) != set(
                candidate.compiled.artifact.source_event_ids):
            raise CandidateControlError(
                'candidate source hashes do not cover the artifact sources')
        registered = self.artifacts.get(candidate.artifact_id)
        if registered is not None and artifact_content_digest(registered) != \
                artifact_content_digest(candidate.compiled.artifact):
            raise CandidateControlError(
                'registered executable payload differs from the issued candidate')
        return candidate

    # -- archived manifest authentication -----------------------------------
    def _authenticate_manifest_body(self, manifest, *,
                                    expected_id: str | None = None) -> dict:
        try:
            return authenticate_manifest(manifest, expected_id=expected_id)
        except ManifestIntegrityError as error:
            raise ProtectedRevisionError(str(error)) from error

    def _load_manifest_body(self, manifest_id: str) -> dict:
        try:
            _sha(manifest_id, 'manifest_id')
        except ValueError as error:
            raise ProtectedRevisionError(
                'manifest ID is not a SHA256') from error
        try:
            return load_manifest(self.manifests, manifest_id)
        except ManifestIntegrityError as error:
            raise ProtectedRevisionError(str(error)) from error

    def _authenticated_ancestry(self, target_manifest_id: str) -> dict:
        """Walk active -> parent chain, authenticating every body before use."""
        try:
            active = active_manifest(self.manifests)
        except ManifestIntegrityError as error:
            raise ProtectedRevisionError(str(error)) from error
        if active is None:
            raise ProtectedRevisionError('no active manifest')
        if self._published_manifest is None or \
                active['manifest_id'] != self._published_manifest['manifest_id'] or \
                fingerprint(active) != self._published_manifest['manifest_digest']:
            raise ProtectedRevisionError(
                'active manifest is not this cycle\'s publication')
        try:
            return authenticated_ancestor(self.manifests, target_manifest_id)
        except ManifestIntegrityError as error:
            raise ProtectedRevisionError(str(error)) from error

    # -- protected base ------------------------------------------------------
    def prepare_base(self) -> dict:
        if self._base is not None:
            raise BoundedRsiError('the protected base was already prepared')
        payload = {
            'base_revision': self.policy.base_revision,
            'identity_version': self.policy.identity_version,
            'owner': {'subject_id': self.policy.subject_id,
                      'branch_id': self.policy.branch_id,
                      'purpose': self.policy.purpose},
            'guarded_base': {'format': 'replica.bounded_rsi.base.v1',
                             'evidence_epoch': self.policy.evidence_epoch,
                             'source_family': self.policy.source_family},
        }
        evaluation = {
            'decision': 'accept',
            'candidate_digest': fingerprint(payload),
            'parent_manifest_id': None,
            'artifact_digest': fingerprint(payload),
            'cases': [],
            'evaluation_scope': 'host_protected_base',
        }
        manifest = self.manifests.publish(payload, evaluation, expected_parent=None)
        self._base = MappingProxyType({
            'manifest_id': manifest['manifest_id'],
            'manifest_digest': fingerprint(manifest),
            'payload_digest': fingerprint(payload),
            'parent': manifest['parent'],
            'base_revision': self.policy.base_revision,
            'identity_version': self.policy.identity_version,
        })
        # Detached copy: mutating this result cannot rewrite rollback authority.
        return dict(self._base)

    def _assert_active_protected_base(self) -> dict:
        if self._base is None:
            raise BoundedRsiError('the protected base is not prepared')
        try:
            active = active_manifest(self.manifests)
        except ManifestIntegrityError as error:
            raise ProtectedRevisionError(str(error)) from error
        if active is None:
            raise ProtectedRevisionError('the protected base manifest is absent')
        payload = active.get('payload') or {}
        if payload.get('base_revision') != self.policy.base_revision:
            raise ProtectedRevisionError('active base revision differs from policy')
        if payload.get('identity_version') != self.policy.identity_version:
            raise ProtectedRevisionError('active identity version differs from policy')
        return active

    # -- practice ------------------------------------------------------------
    def practice(self) -> PracticeOutcome:
        if self._base is None:
            raise BoundedRsiError('practice requires the protected base')
        if self._practice is not None:
            raise BoundedRsiError('practice already ran in this cycle')
        family = self.catalog.by_split('source')
        if family.family_id != self.policy.source_family:
            raise BoundedRsiError('source family differs from policy')
        before_successes = 0
        after_successes = 0
        cases = []
        failure_ids = []
        repair_ids = []
        total_tools = 0
        total_runs = 0
        for case in family.cases:
            target = self.root / 'owned_worlds' / family.split / family.family_id \
                / case.case_id
            materialize_case(family, case, target)
            workspace = FixtureWorkspace(target)
            guarded = GuardedWorkspace(workspace, self.policy.guard)
            before = workspace.workspace_hashes()
            initial = guarded.run_test({})
            initial_exit = (initial.get('receipt') or {}).get('exit_code')
            if initial_exit == 0:
                before_successes += 1
            initial_counts = guarded.counts()
            failure = self.ledger.append_observation(
                case_id=case.case_id, world_family=family.family_id,
                split=family.split, stage='observed_failure',
                status=initial.get('status', 'error'), error=initial.get('error'),
                final_exit_code=initial_exit, before=before,
                after=workspace.workspace_hashes(),
                tool_calls=initial_counts['tool_calls'],
                evaluator_runs=initial_counts['evaluator_runs'],
                receipt=_receipt_view(initial),
                raw_receipts=(dict(initial),))
            failure_ids.append(failure.record_id)
            procedure = RepairProcedure(
                'repair-config', APP_SETTINGS, (failure.record_id,))
            run = procedure.execute(guarded)
            final_exit = run.final_exit_code
            if run.status in ('succeeded', 'not_required'):
                after_successes += 1
            counts = guarded.counts()
            final_receipt = _receipt_view(run.receipts[-1]) if run.receipts else {}
            repair = self.ledger.append_observation(
                case_id=case.case_id, world_family=family.family_id,
                split=family.split, stage='observed_repair',
                status=run.status, error=run.error, final_exit_code=final_exit,
                before=before, after=workspace.workspace_hashes(),
                tool_calls=counts['tool_calls'],
                evaluator_runs=counts['evaluator_runs'],
                receipt=final_receipt,
                raw_receipts=tuple(run.receipts))
            repair_ids.append(repair.record_id)
            total_tools += counts['tool_calls']
            total_runs += counts['evaluator_runs']
            cases.append({
                'case_id': case.case_id,
                'role': case.role,
                'initial_exit_code': initial_exit,
                'repair_status': run.status,
                'final_exit_code': final_exit,
                'error': run.error,
                'parsed_bound': dict(run.parsed) if run.parsed else None,
            })
        evidence = CycleEvidence(
            task_id=family.family_id, task_family=family.family_id,
            success=False, decisions=len(family.cases),
            tool_calls=total_tools, prompt_tokens=0, generated_tokens=0,
            wall_seconds=0.0, source_event_ids=tuple(failure_ids),
            verified_receipts=len(family.cases), malformed_actions=0,
            false_success_claims=0, repeated_read=False,
            state_digest=fingerprint({'family': family.family_id,
                                      'records': [record.record_hash
                                                  for record in self.ledger.records]}))
        budget = DirectorBudget(max_proposals=1, max_cost_units=10000.0,
                                min_repeated_failures=2,
                                min_verified_receipts_for_procedure=1)
        decision = review_cycle(
            evidence, director=LearningDirector(budget=budget),
            parent_manifest_id=self._base['manifest_id'])
        if decision.status != 'proposed' or decision.proposal is None:
            raise BoundedRsiError(
                f'director did not authorize a procedure: {decision.reason}')
        if decision.proposal.target_kind != 'procedure':
            raise BoundedRsiError('director authorized a non-procedure target')
        self._proposal = decision.proposal
        outcome = PracticeOutcome(
            family_id=family.family_id,
            case_ids=tuple(case.case_id for case in family.cases),
            cases=tuple(cases),
            records=self.ledger.records,
            failure_record_ids=tuple(failure_ids),
            repair_record_ids=tuple(repair_ids),
            before_successes=before_successes, after_successes=after_successes,
            tool_calls=total_tools, evaluator_runs=total_runs,
            proposal=dict(decision.proposal.to_dict(),
                          decision=decision.status, reason=decision.reason,
                          diagnostic=decision.diagnostic or {}))
        self._practice = outcome
        return outcome

    # -- candidate + independent verification --------------------------------
    def compile_candidate(self, practice: PracticeOutcome,
                          base_manifest_id: str,
                          *, submission: CandidateSubmission | None = None
                          ) -> CompiledCandidate:
        if not isinstance(practice, PracticeOutcome) or practice is not self._practice:
            raise CandidateControlError('compile requires this cycle\'s practice')
        if self._proposal is None:
            raise BoundedRsiError('no authorized proposal is present')
        if base_manifest_id != (self._base or {}).get('manifest_id'):
            raise ProtectedRevisionError('candidate parent differs from protected base')
        candidate = submission if submission is not None else CandidateSubmission()
        if not isinstance(candidate, CandidateSubmission):
            raise CandidateControlError('candidate must be a CandidateSubmission')
        record_ids = {record.record_id for record in practice.records}
        if not set(self._proposal.source_event_ids) <= record_ids:
            raise BoundedRsiError('proposal sources are not practice evidence')
        if self._proposal.parent_manifest_id != base_manifest_id:
            raise ProtectedRevisionError('proposal parent differs from protected base')
        if self._proposal.target_kind != 'procedure':
            raise CandidateControlError('only a procedure proposal may be compiled')
        if self._proposal.target_id != PROCEDURE_TARGET:
            raise ProtectedRevisionError(
                'proposal target differs from the host-authorized procedure target')
        procedure = RepairProcedure(
            candidate.name, candidate.workspace_entry,
            tuple(record.record_id for record in practice.records), version=1)
        cutoff = max(record.observed_at for record in practice.records)
        binding = CompilationBinding(
            self.policy.subject_id, self.policy.branch_id, self.policy.purpose,
            cutoff, self.policy.evidence_epoch, self.policy.base_revision,
            self.policy.identity_version, base_manifest_id,
            self._base['manifest_digest'])
        compiled = CompiledProcedure.compile(
            procedure, binding, practice.records, self.artifacts,
            proposal=self._proposal, authorized_target=PROCEDURE_TARGET)
        return self._issue_candidate(CompiledCandidate(
            compiled=compiled, submission=candidate,
            claims_ignored=candidate.ignored_claims(),
            source_record_ids=tuple(record.record_id for record in practice.records),
            proposal_id=self._proposal.proposal_id,
            proposal_digest=self._proposal.digest,
            proposal=MappingProxyType(self._proposal.to_dict())))

    def verify(self, candidate: CompiledCandidate, *, split: str = 'verification'
               ) -> VerificationOutcome:
        if not isinstance(candidate, CompiledCandidate):
            raise CandidateControlError('verify accepts only this cycle\'s candidate')
        self._authenticate_issued_candidate(candidate)
        if split == 'verification':
            family_id = self.policy.verification_family
        elif split == 'rejection':
            family_id = self.policy.rejection_family
        else:
            raise CandidateControlError('verification split must be independent '
                                        'or rejection')
        family = self.catalog.family(family_id)
        workspaces = []
        inventory_cases = []
        for case in family.cases:
            target = self.root / 'verification' / split / candidate.artifact_id \
                / case.case_id
            materialize_case(family, case, target)
            workspace = FixtureWorkspace(target)
            guarded = GuardedWorkspace(workspace, self.policy.guard)
            inventory_cases.append({
                'case_id': case.case_id,
                'role': case.role,
                'definition_digest': fingerprint(case.to_dict()),
                'workspace_digest': fingerprint(dict(guarded.workspace_hashes())),
            })
            workspaces.append((case.case_id, case.role, guarded))
        inventory = WorldInventory.build(family.split, family_id, inventory_cases)
        evaluation = candidate.compiled.evaluate(self.artifacts,
                                                 tuple(workspaces),
                                                 inventory=inventory)
        cases = evaluation.cases
        # The compiler keeps raw receipts out of published evidence so absolute
        # paths and wall durations cannot make replay root-dependent.
        publication_evaluation = evaluation
        tool_calls = sum(ws[2].tool_calls for ws in workspaces)
        evaluator_runs = sum(ws[2].evaluator_runs for ws in workspaces)
        return VerificationOutcome(
            candidate=candidate, split=split, family_id=family_id, cases=cases,
            evaluation=publication_evaluation, tool_calls=tool_calls,
            evaluator_runs=evaluator_runs,
            world_inventory=inventory.to_dict(),
            raw_cases=evaluation.raw_cases)

    def _verified_world_inventory(self,
                                  verification: VerificationOutcome) -> WorldInventory:
        """Re-derive the verified inventory from the current host catalog.

        Any change to the world family between verification and publication
        changes the digest and refuses publication.
        """
        family = self.catalog.family(verification.family_id)
        verified = WorldInventory.from_dict(verification.world_inventory)
        by_case = {case.case_id: case for case in verified.cases}
        if set(by_case) != {case.case_id for case in family.cases}:
            raise ProtectedRevisionError(
                'verification cases are not the current catalog family')
        cases = []
        for case in family.cases:
            previous = by_case[case.case_id]
            if previous.role != case.role:
                raise ProtectedRevisionError(
                    'verification case roles changed after evaluation')
            cases.append({
                'case_id': case.case_id,
                'role': case.role,
                'definition_digest': fingerprint(case.to_dict()),
                'workspace_digest': previous.workspace_digest,
            })
        inventory = WorldInventory.build(family.split, family.family_id, cases)
        if inventory.digest != verified.digest:
            raise ProtectedRevisionError(
                'verification world inventory changed after evaluation')
        return inventory


    # -- publication / rejection / rollback ----------------------------------
    def publish(self, candidate: CompiledCandidate,
                verification: VerificationOutcome) -> dict:
        if not isinstance(candidate, CompiledCandidate):
            raise CandidateControlError('publish accepts only a compiled candidate')
        self._authenticate_issued_candidate(candidate)
        if not isinstance(verification, VerificationOutcome):
            raise CandidateControlError('publish requires host verification')
        if verification.candidate is not candidate:
            raise CandidateControlError(
                'verification does not belong to this issued candidate')
        if verification.candidate.artifact_id != candidate.artifact_id:
            raise CandidateControlError('verification belongs to another candidate')
        if self._base is None:
            raise BoundedRsiError('protected base missing')
        if candidate.compiled.binding.evidence_epoch != self.ledger.epoch:
            raise StaleSourceError('source evidence epoch changed after practice')
        self._assert_active_protected_base()
        if verification.accepted is not True:
            raise PermissionError('independent verification did not accept the candidate')
        if verification.split != 'verification' or \
                verification.family_id != self.policy.verification_family:
            raise CandidateControlError('publication requires the independent '
                                        'verification world family')
        if verification.evaluation.parent_manifest_id != self._base['manifest_id']:
            raise ProtectedRevisionError(
                'verification evaluation parent differs from the protected base')
        inventory = self._verified_world_inventory(verification)
        try:
            manifest = candidate.compiled.publish(
                self.artifacts, self.manifests, verification.evaluation,
                self.ledger.current_hashes(),
                journal=self.transactions, world_inventory=inventory,
                catalog_path=self.catalog_path)
        except CandidateIntegrityError as error:
            raise ProtectedRevisionError(str(error)) from error
        self._published_candidate = candidate
        self._published_manifest = MappingProxyType({
            'manifest_id': manifest['manifest_id'],
            'manifest_digest': fingerprint(manifest),
            'payload_digest': fingerprint(manifest['payload']),
            'artifact_content_digest': artifact_content_digest(
                self.artifacts.get(candidate.artifact_id)),
            'parent': manifest['parent'],
        })
        return manifest


    def rollback(self, ancestor_manifest_id: str) -> dict:
        if self._base is None or self._published_candidate is None or \
                self._published_manifest is None:
            raise BoundedRsiError('rollback requires a published candidate')
        self._authenticate_issued_candidate(self._published_candidate)
        if ancestor_manifest_id != self._base['manifest_id']:
            raise ProtectedRevisionError('rollback target is not the protected base')
        # Authenticate every traversed body and the protected target *before*
        # ManifestRegistry may rewrite active.json.
        target = self._authenticated_ancestry(ancestor_manifest_id)
        if fingerprint(target) != self._base['manifest_digest']:
            raise ProtectedRevisionError(
                'protected ancestor body changed after publication')
        target_payload = target.get('payload') or {}
        if fingerprint(target_payload) != self._base['payload_digest']:
            raise ProtectedRevisionError(
                'protected ancestor payload changed after publication')
        if target_payload.get('base_revision') != self.policy.base_revision or \
                target_payload.get('identity_version') != self.policy.identity_version:
            raise ProtectedRevisionError(
                'protected ancestor identity differs from policy')
        active_before = active_manifest(self.manifests)
        if active_before is None or \
                active_before['manifest_id'] != self._published_manifest['manifest_id']:
            raise ProtectedRevisionError('active manifest changed before rollback')
        current = self.artifacts.get(self._published_candidate.artifact_id)
        if current is None or current.status not in ('accepted', 'revoked'):
            raise BoundedRsiError(
                'published artifact is neither accepted nor source-revoked')
        if artifact_content_digest(current) != \
                self._published_manifest['artifact_content_digest']:
            raise ProtectedRevisionError(
                'published artifact payload changed after publication')
        try:
            restored = self._published_candidate.compiled.rollback(
                self.artifacts, self.manifests, ancestor_manifest_id,
                journal=self.transactions, catalog_path=self.catalog_path)
        except CandidateIntegrityError as error:
            raise ProtectedRevisionError(str(error)) from error
        except ManifestIntegrityError as error:
            raise ProtectedRevisionError(str(error)) from error
        active_after = active_manifest(self.manifests)
        if active_after is None:
            raise ProtectedRevisionError('rollback produced no active manifest')
        if fingerprint(active_after) != self._base['manifest_digest']:
            raise ProtectedRevisionError(
                'restored active body differs from the protected base')
        artifact = self.artifacts.get(self._published_candidate.artifact_id)
        if artifact.status != 'revoked':
            raise BoundedRsiError('published artifact was not revoked by rollback')
        return {
            'ancestor_manifest_id': ancestor_manifest_id,
            'ancestor_manifest_digest': fingerprint(target),
            'protected_payload_digest': self._base['payload_digest'],
            'active_before_rollback': active_before['manifest_id'],
            'active_after_rollback': active_after['manifest_id'],
            'restored_manifest_id': restored['manifest_id'],
            'candidate_artifact_id': self._published_candidate.artifact_id,
            'candidate_artifact_status': artifact.status,
            'revoked': artifact.status == 'revoked',
            'base_restored': active_after['manifest_id'] == ancestor_manifest_id,
        }

    # -- full cycle ----------------------------------------------------------
    def run(self) -> 'CycleReport':
        base = self.prepare_base()
        practice = self.practice()
        accepted_candidate = self.compile_candidate(
            practice, base['manifest_id'],
            submission=CandidateSubmission(name='repair-config'))
        verification = self.verify(accepted_candidate, split='verification')
        if verification.accepted is not True:
            raise BoundedRsiError('the host-owned verification family rejected '
                                  'the practice-derived candidate')
        rejection_candidate = self.compile_candidate(
            practice, base['manifest_id'],
            submission=CandidateSubmission(name='repair-config-rejection-probe'))
        rejection = self.verify(rejection_candidate, split='rejection')
        rejection_active_before = active_manifest(self.manifests)['manifest_id']
        rejection_error = None
        rejection_refused = False
        try:
            self.publish(rejection_candidate, rejection)
        except PermissionError as error:
            rejection_refused = True
            rejection_error = str(error)
        rejection_active_after = active_manifest(self.manifests)['manifest_id']
        rejection_artifact = self.artifacts.get(rejection_candidate.artifact_id)
        rejection_record = {
            'candidate_name': rejection_candidate.submission.name,
            'artifact_id': rejection_candidate.artifact_id,
            'split': rejection.split,
            'family_id': rejection.family_id,
            'evaluation_accepted': rejection.accepted,
            'publish_refused': rejection_refused,
            'publish_error': rejection_error,
            'active_unchanged': rejection_active_before == rejection_active_after,
            'active_before': rejection_active_before,
            'active_after': rejection_active_after,
            'artifact_status': rejection_artifact.status,
            'cases': [dict(case) for case in rejection.cases],
            'tool_calls': rejection.tool_calls,
            'evaluator_runs': rejection.evaluator_runs,
            'claims_ignored': list(rejection_candidate.claims_ignored),
        }
        if not rejection_refused or rejection.accepted or \
                rejection_record['active_unchanged'] is not True:
            raise BoundedRsiError('invalid rejection was not contained')

        publication = self.publish(accepted_candidate, verification)
        rollback = self.rollback(base['manifest_id'])
        neural = self.neural_gate.assess(self.evidence_root)
        if self.artifacts.list(kind='neural_candidate'):
            raise BoundedRsiError('a neural artifact was created without host promotion')
        neural['neural_artifact_created'] = False
        neural['updates'] = []
        neural['trained_parameters'] = []
        neural['pins'] = [pin.to_dict() for pin in self.neural_gate.pins.items()]
        selector_policy = {
            'owner': 'host',
            'changes': [],
            'reason': ('selector-policy updates require a separately evaluated '
                       'candidate and are outside this bounded slice'),
        }
        retrieval = {
            'source_retrieval': {
                'mode': 'exact_record_hash_binding',
                'records': len(practice.records),
                'source_hashes': {record.record_id: record.record_hash
                                  for record in practice.records},
                'epoch': self.ledger.epoch - self.policy.evidence_epoch,
                'evidence_epoch': self.ledger.epoch,
                'corrections': 0,
                'revocations': 1,
            },
            'memory_only_recall': {
                'status': 'not_exercised',
                'reason': ('no learned visual memory passed its gate; the v5 '
                           'verdict is reject and diagnostic-only'),
            },
        }
        unrun_scope = (
            'neural training, fine-tuning or promotion',
            'pinned perception quality on a broad held-out suite (V0 is interface-only)',
            'learned visual memory recall (v5 remains rejected)',
            'independent correction and retention evidence for neural promotion',
            'selector-policy learning and update',
            'longitudinal campaign or repeated-cycle acceleration measurement',
            'integrated RSI beyond isolated host-owned procedural repair',
        )
        gain = {
            'source_cases': len(practice.case_ids),
            'source_before_successes': practice.before_successes,
            'source_after_successes': practice.after_successes,
            'verification_cases': len(verification.cases),
            'verification_passed': sum(
                1 for case in verification.cases if case['independent_passed']),
            'rejection_cases': len(rejection.cases),
            'rejection_passed': sum(
                1 for case in rejection.cases if case['independent_passed']),
            'tool_calls': (practice.tool_calls + verification.tool_calls
                           + rejection.tool_calls),
            'evaluator_runs': (practice.evaluator_runs + verification.evaluator_runs
                               + rejection.evaluator_runs),
            'unit': 'bounded_owned_procedural_cases',
        }
        qualification = {
            'scope': QUALIFICATION_SCOPE,
            'procedure_cycle_verified': True,
            'isolated_output_registry': True,
            'selected_manifest_touched': False,
            'integrated_rsi': False,
            'accelerated': False,
            'acceleration_claim': None,
            'neural_promotions': 0,
            'description': ('This qualifies bounded procedural plumbing only; it '
                            'does not establish integrated RSI or acceleration.'),
        }
        report = CycleReport(
            policy=self.policy.to_dict(),
            worlds=dict(self.worlds),
            base=base,
            practice=practice,
            candidate=accepted_candidate.to_dict(),
            verification=verification.to_dict(),
            publication={
                'manifest_id': publication['manifest_id'],
                'parent': publication['parent'],
                'artifact_id': publication['payload']['artifact']['artifact_id'],
                'candidate_digest': publication['evaluation']['candidate_digest'],
                'evaluation_scope': publication['evaluation']['evaluation_scope'],
                'manifest_schema': publication['payload']['schema'],
                'source_hashes': publication['payload']['source_hashes'],
                'source_digest': publication['payload']['source_digest'],
                'proposal_digest': publication['payload']['proposal_digest'],
                'worlds_digest': publication['payload']['worlds']['digest'],
                'identity_digest': publication['payload']['identity']['digest'],
                'artifact_content_digest':
                    publication['payload']['artifact']['content_digest'],
                'base_revision': publication['payload']['base_revision'],
                'identity_version': publication['payload']['identity_version'],
                'evidence_epoch': publication['payload']['owner']['evidence_epoch'],
            },
            rejection=rejection_record,
            rollback=rollback,
            neural=neural,
            selector_policy=selector_policy,
            retrieval=retrieval,
            unrun_scope=unrun_scope,
            observations=self.ledger.observations,
            gain=gain,
            qualification=qualification,
            raw_traces={
                'practice': [{'record_id': observation.record_id,
                              'case_id': observation.case_id,
                              'stage': observation.stage,
                              'receipts': [dict(item)
                                           for item in observation.raw_receipts]}
                             for observation in self.ledger.observations],
                'verification': [dict(item) for item in verification.raw_cases],
                'rejection': [dict(item) for item in rejection.raw_cases],
            },
        )
        return report


@dataclass(frozen=True)
class CycleReport:
    policy: dict
    worlds: dict
    base: dict
    practice: PracticeOutcome
    candidate: dict
    verification: dict
    publication: dict
    rejection: dict
    rollback: dict
    neural: dict
    selector_policy: dict
    retrieval: dict
    unrun_scope: tuple[str, ...]
    observations: tuple[Observation, ...]
    gain: dict
    qualification: dict
    raw_traces: dict = field(default_factory=dict)

    def core(self) -> dict:
        return {
            'format': CYCLE_FORMAT,
            'policy': dict(self.policy),
            'worlds': dict(self.worlds),
            'base': dict(self.base),
            'practice': self.practice.to_dict(),
            'candidate': dict(self.candidate),
            'verification': dict(self.verification),
            'publication': dict(self.publication),
            'rejection': dict(self.rejection),
            'rollback': dict(self.rollback),
            'neural': dict(self.neural),
            'selector_policy': dict(self.selector_policy),
            'retrieval': dict(self.retrieval),
            'unrun_scope': list(self.unrun_scope),
            'episodic_observations': [observation.to_dict()
                                      for observation in self.observations],
            'procedure_gain': dict(self.gain),
            'qualification': dict(self.qualification),
        }

    @property
    def replay_digest(self) -> str:
        return fingerprint(self.core())

    def as_dict(self) -> dict:
        payload = self.core()
        payload['replay_digest'] = self.replay_digest
        payload['raw_traces'] = dict(self.raw_traces)
        return payload


__all__ = [
    'APP_SETTINGS', 'BoundedRsiCycle', 'BoundedRsiError', 'CYCLE_FORMAT',
    'CandidateControlError', 'CandidateSubmission', 'CompiledCandidate',
    'CycleReport', 'DEFAULT_NEURAL_PINS', 'EpisodicLedger',
    'EvidenceIntegrityError', 'EvidencePin', 'GATES', 'GuardedWorkspace',
    'HostPolicy', 'MEMORY_FORMAT', 'NEURAL_EVIDENCE_FORMAT', 'NEURAL_GATE_FORMAT',
    'NeuralEvidencePins', 'NeuralGate', 'Observation', 'PERCEPTION_FORMAT',
    'PINNED_MODEL_REVISION', 'PROCEDURE_TARGET', 'PracticeOutcome',
    'ProtectedRevisionError',
    'QUALIFICATION_SCOPE', 'ResourceGuard', 'ResourceLimitError', 'SPLITS',
    'TaskCase', 'VerificationOutcome', 'WorldCatalog', 'WorldFamily',
    'build_owned_worlds', 'materialize_case',
]
