"""C17 cache manager: exact ownership keys, bounds and source invalidation."""
import pytest
from replica_lab.runtime.cache import CacheKey, OwnedCache

def key(**changes):
    value = dict(producer='model-a', writer='writer-a', tokenizer='tok-a',
                 record_id='r1', record_hash='a'*64, span_hash='b'*64,
                 subject_id='s', branch_id='b', purpose='p', cutoff=10, revision=2,
                 identity_digest=None)
    value.update(changes); return CacheKey(**value)

def test_exact_key_hit_and_mismatch_miss():
    cache=OwnedCache()
    cache.put(key(), {'slot':1}, size_bytes=8)
    assert cache.get(key()) == {'slot':1}
    assert cache.get(key(producer='model-b')) is None
    assert cache.stats['hits']==1 and cache.stats['misses']==1
    with pytest.raises(ValueError):
        key(record_hash='nope')

def test_bounds_and_source_invalidation():
    cache=OwnedCache(max_entries=2, max_bytes=100)
    cache.put(key(record_id='r1'), 'a', size_bytes=10)
    cache.put(key(record_id='r2', record_hash='c'*64), 'b', size_bytes=10)
    cache.put(key(record_id='r3', record_hash='d'*64), 'c', size_bytes=10)
    assert len(cache)==2
    assert cache.get(key(record_id='r1')) is None  # oldest evicted
    assert cache.invalidate_source('c'*64)==1
    assert cache.get(key(record_id='r2', record_hash='c'*64)) is None
    cache.invalidate(lambda item: item.branch_id == 'b')
    assert len(cache)==0

def test_collision_and_clear():
    cache=OwnedCache()
    cache.clear(); assert len(cache)==0
    with pytest.raises(ValueError):
        OwnedCache(max_entries=0)
