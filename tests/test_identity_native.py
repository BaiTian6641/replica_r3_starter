"""Optional native identity reader: null parity, influence, binding and v3 export."""
from dataclasses import replace
import torch

from architecture_helpers import tiny_adapter
from replica_lab.architecture.contracts import (CoreSpec, IdentityRead, IdentitySpec,
                                                fingerprint)
from replica_lab.architecture.minicpm5 import MiniCPM5Adapter
from replica_lab.runtime.workspace_store import WorkspaceStore

SPEC = CoreSpec(width=16, slots=4, steps=2, heads=4, task_features=24,
                ffn_hidden=64, control_hidden=29)
IDENTITY = IdentitySpec(width=16, heads=4)


def adapter_with_identity():
    base = tiny_adapter()
    manifest = replace(base.manifest, core=SPEC, identity=IDENTITY)
    return MiniCPM5Adapter(base.base, manifest, 'modules_only')


def identity_read(valid=None):
    slots = torch.randn(1, 2, 16, generator=torch.Generator().manual_seed(21))
    if valid is None:
        valid = torch.ones(1, 2, dtype=torch.bool)
    return IdentityRead(slots, valid, 'host-identity-encoder',
                        fingerprint({'identity': 'fictional'}), 'fictional-subject',
                        'branch-a', 'source_snapshot', ('ep-1',))


def test_null_parity_disable_and_nonzero_influence():
    adapter = adapter_with_identity().eval()
    # Build a synthetic frame from the same adapter for exact provenance.
    from tests.architecture_helpers import synthetic_frame
    frame = synthetic_frame(adapter)
    features = torch.zeros(1, SPEC.task_features)
    plain = adapter.build_core_state(frame.sense, frame.memory, features,
                                     task_digest='task-digest', revision=2)
    identity = identity_read()
    initial = adapter.build_core_state(frame.sense, frame.memory, features,
                                       task_digest='task-digest', revision=2,
                                       identity=identity)
    assert initial.identity_digest == identity.context_digest
    torch.testing.assert_close(initial.workspace, plain.workspace, atol=0, rtol=0)
    disabled = adapter.build_core_state(frame.sense, frame.memory, features,
                                        task_digest='task-digest', revision=2,
                                        identity=identity, identity_enabled=False)
    torch.testing.assert_close(disabled.workspace, plain.workspace, atol=0, rtol=0)
    with torch.no_grad():
        adapter.identity_reader.output_projection.weight.normal_(0.0, 0.2)
        adapter.identity_reader.gate_logit.fill_(2.0)
    changed = adapter.build_core_state(frame.sense, frame.memory, features,
                                       task_digest='task-digest', revision=2,
                                       identity=identity)
    assert not torch.equal(changed.workspace, plain.workspace)
    masked = adapter.build_core_state(frame.sense, frame.memory, features,
                                      task_digest='task-digest', revision=2,
                                      identity=identity_read(torch.zeros(1, 2, dtype=torch.bool)))
    torch.testing.assert_close(masked.workspace, plain.workspace, atol=0, rtol=0)


def test_identity_binding_and_v3_extension_roundtrip(tmp_path):
    adapter = adapter_with_identity().eval()
    adapter.seal_original_identity()
    from tests.architecture_helpers import synthetic_frame
    frame = synthetic_frame(adapter)
    identity = identity_read()
    core = adapter.build_core_state(frame.sense, frame.memory,
                                    torch.zeros(1, SPEC.task_features),
                                    task_digest='task-digest', revision=3,
                                    identity=identity)
    store = WorkspaceStore(tmp_path / 'workspace')
    store.save(core)
    loaded = store.load(expected_identity_digest=identity.context_digest)
    assert loaded is not None and loaded.identity_digest == identity.context_digest
    assert store.load(expected_identity_digest=fingerprint('other')) is None
    destination = tmp_path / 'candidate'
    receipt = adapter.save_extension(destination)
    assert receipt['format'] == 'replica.minicpm5.extensions.v3'
    assert 'identity_reader' in receipt['components']
    restored = adapter_with_identity().eval()
    restored.seal_original_identity()
    restored.load_extension(destination)
    for name, parameter in adapter.new_named_parameters():
        torch.testing.assert_close(dict(restored.new_named_parameters())[name], parameter,
                                   atol=0, rtol=0)


def test_identity_requires_matching_core_and_slots():
    adapter = adapter_with_identity().eval()
    from tests.architecture_helpers import synthetic_frame
    frame = synthetic_frame(adapter)
    bad = identity_read()
    bad = IdentityRead(bad.slots[:, :, :8], bad.valid, bad.producer,
                       bad.context_digest, bad.subject_id, bad.branch_id, bad.mode,
                       bad.evidence_ids)
    with torch.no_grad():
        try:
            adapter.build_core_state(frame.sense, frame.memory,
                                     torch.zeros(1, SPEC.task_features),
                                     task_digest='t', revision=0, identity=bad)
        except ValueError as error:
            assert 'width' in str(error)
        else:
            raise AssertionError('wrong identity width must be rejected')


def test_memory_session_carries_identity_into_the_workspace(tmp_path):
    from dataclasses import replace as dc_replace
    from architecture_helpers import tiny_adapter
    from replica_lab.architecture.contracts import fingerprint
    from replica_lab.memory_system import EligibilityScope, MemoryRecord, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from tests.test_e3_session_core import CharacterTokenizer, prepare_args

    base = tiny_adapter()
    manifest = dc_replace(base.manifest, core=SPEC, identity=IDENTITY)
    adapter = MiniCPM5Adapter(base.base, manifest, 'modules_only').eval()
    ledger = RecordLedger(tmp_path / 'records.sqlite')
    ledger.admit(MemoryRecord('r1', 'fixture', 'source', 'blue cat', 1, 0, 'test',
                              source_id='fixture', authenticity='source_report'),
                 authorized_subject='fixture', authorized_branch='source',
                 source_authorized=True)
    try:
        session = MemorySession(adapter, CharacterTokenizer(), ledger)
        with torch.no_grad():
            adapter.identity_reader.output_projection.weight.normal_(0.0, 0.2)
            adapter.identity_reader.gate_logit.fill_(2.0)
        args = prepare_args()
        identity = identity_read()
        enabled = session.prepare(identity=identity, **args)
        disabled = session.prepare(identity=identity, identity_enabled=False, **args)
        try:
            assert enabled.frame.core.identity_digest == identity.context_digest
            assert disabled.frame.core.identity_digest == identity.context_digest
            assert not torch.equal(enabled.frame.core.workspace, disabled.frame.core.workspace)
        finally:
            enabled.release(); disabled.release()
    finally:
        ledger.close()
