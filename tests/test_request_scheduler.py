"""C17: one admitted native operation, explicit leases and invalidation."""
import threading

import pytest

from replica_lab.runtime.scheduler import (RequestOwnership, RequestScheduler,
                                           SchedulerBusy)


def owner(request_id='r1', revision=1):
    return RequestOwnership(request_id, 'task-1', 'digest-1', revision, 4,
                            'branch-a', 'producer-v1', 'a' * 64)


def test_one_active_lease_and_explicit_release():
    scheduler = RequestScheduler()
    with scheduler.admit(owner()) as lease:
        assert scheduler.active() == owner()
        lease.assert_current()
        with pytest.raises(SchedulerBusy):
            with scheduler.admit(owner('r2')):
                pass
    assert scheduler.active() is None
    assert [event['event'] for event in scheduler.events()] == ['admitted', 'released']


def test_invalidation_stops_the_lease_and_cannot_be_reused():
    scheduler = RequestScheduler()
    with pytest.raises(RuntimeError, match='invalidated'):
        with scheduler.admit(owner()) as lease:
            scheduler.invalidate('r1', 'source_revoked')
            lease.assert_current()
    assert scheduler.active() is None


def test_second_thread_is_refused_rather_than_mixed():
    scheduler = RequestScheduler()
    entered = threading.Event()
    release = threading.Event()
    result = {}

    def first():
        with scheduler.admit(owner('r1')):
            entered.set()
            release.wait(2)

    def second():
        entered.wait(2)
        try:
            scheduler.run(owner('r2'), lambda lease: None)
        except SchedulerBusy as error:
            result['refused'] = str(error)

    a = threading.Thread(target=first); b = threading.Thread(target=second)
    a.start(); b.start(); entered.wait(2)
    try:
        with pytest.raises(SchedulerBusy):
            scheduler.run(owner('r3'), lambda lease: None)
    finally:
        release.set(); a.join(); b.join()
    assert result.get('refused')
    assert scheduler.active() is None


def test_ownership_binds_task_revision_epoch_branch_identity():
    first = owner()
    second = owner(revision=2)
    assert first.key != second.key
    with pytest.raises(ValueError):
        RequestOwnership('bad id', 'task', 'digest', 0, 0, 'branch', 'p')
    with pytest.raises(ValueError):
        RequestOwnership('r', 'task', 'digest', 0, 0, 'branch', 'p', 'not-sha')
