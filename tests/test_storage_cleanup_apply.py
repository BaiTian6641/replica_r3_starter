"""Tiny-file tests for authorized cleanup boundaries, COW integrity and rollback."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys

import pytest

SCRIPTS=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('storage_cleanup',SCRIPTS/'apply_checkpoint_storage_cleanup.py')
cleanup=importlib.util.module_from_spec(spec);spec.loader.exec_module(cleanup)


def native():
    header=json.dumps({'model.weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    return struct.pack('<Q',len(header))+header+struct.pack('<f',1.0)


@pytest.fixture
def pair(tmp_path):
    reference,target=tmp_path/'reference',tmp_path/'target'
    for p in (reference,target): p.write_bytes(native())
    return reference,target,hashlib.sha256(native()).hexdigest()


def copy(source,target):
    assert not target.exists()
    shutil.copy2(source,target)


def test_real_apfs_clone_has_independent_inode_and_private_writes(pair):
    reference,target,expected=pair
    old_inode=target.stat().st_ino
    result=cleanup.clone_one(target,reference,expected)
    assert result['sha256_before']==result['sha256_after']==expected
    assert target.stat().st_ino not in (old_inode,reference.stat().st_ino)
    assert reference.stat().st_nlink==target.stat().st_nlink==1
    target.write_bytes(b'private overwrite')
    assert reference.read_bytes()==native()


@pytest.mark.parametrize('method',['failed','hardlink','corrupt'])
def test_bad_clone_preserves_original(pair,method):
    reference,target,expected=pair
    before=target.read_bytes(),target.stat().st_ino
    def bad(src,dst):
        if method=='failed': raise OSError('simulated unavailable clone')
        if method=='hardlink': os.link(src,dst)
        else: dst.write_bytes(b'corrupt')
    with pytest.raises((OSError,ValueError)):
        cleanup.clone_one(target,reference,expected,clone=bad)
    assert (target.read_bytes(),target.stat().st_ino)==before
    assert not list(target.parent.glob('.checkpoint-cleanup-*'))


def test_different_bytes_never_replace_target(pair):
    reference,target,expected=pair
    target.write_bytes(b'different')
    with pytest.raises(ValueError,match='Exact byte'):
        cleanup.clone_one(target,reference,expected,clone=lambda *a:pytest.fail('must not clone'))
    assert target.read_bytes()==b'different'


def test_post_replace_failure_restores_original(pair,monkeypatch):
    reference,target,expected=pair
    inode=target.stat().st_ino
    actual=cleanup.sha
    def fail_installed(path):
        result=actual(path)
        return '0'*64 if path==target and path.stat().st_ino!=inode else result
    monkeypatch.setattr(cleanup,'sha',fail_installed)
    with pytest.raises(ValueError,match='Installed clone'):
        cleanup.clone_one(target,reference,expected,clone=copy)
    assert target.read_bytes()==native()
    assert not list(target.parent.glob('.checkpoint-cleanup-*'))


def test_delete_exact_tree_keeps_sibling_and_metadata_archive(tmp_path):
    obsolete=tmp_path/'runs/run/training_checkpoint_0001'
    obsolete.mkdir(parents=True);(obsolete/'runtime.pt').write_bytes(b'obsolete state')
    selected=obsolete.parent/'checkpoint';selected.mkdir();(selected/'model').write_bytes(b'selected')
    archive=tmp_path/'reports/archive.json';archive.parent.mkdir();archive.write_text('{}')
    before=cleanup.tree_state(obsolete);identity=[obsolete.stat().st_dev,obsolete.stat().st_ino]
    cleanup.delete_one(obsolete,before,identity)
    assert not obsolete.exists() and (selected/'model').read_bytes()==b'selected'
    assert archive.read_text()=='{}' and not list(obsolete.parent.glob('.cleanup-*'))


def test_changed_tree_refuses_deletion(tmp_path):
    p=tmp_path/'old';p.mkdir();(p/'state').write_bytes(b'before')
    state=cleanup.tree_state(p);identity=[p.stat().st_dev,p.stat().st_ino]
    (p/'state').write_bytes(b'after')
    with pytest.raises(ValueError,match='Changed deletion'):
        cleanup.delete_one(p,state,identity)
    assert (p/'state').read_bytes()==b'after'


def test_replaced_directory_identity_refuses_deletion(tmp_path):
    p=tmp_path/'old';p.mkdir();(p/'state').write_bytes(b'data')
    with pytest.raises(ValueError,match='Changed deletion'):
        cleanup.delete_one(p,cleanup.tree_state(p),[p.stat().st_dev,-1])
    assert p.is_dir()


@pytest.mark.parametrize('name',['../outside','/absolute'])
def test_parent_or_absolute_path_refused(tmp_path,name):
    with pytest.raises(ValueError):cleanup.safe_path(name,tmp_path)


def test_symlink_tree_refused(tmp_path):
    target=tmp_path/'keep';target.mkdir();(target/'model').write_bytes(b'protected')
    link=tmp_path/'old';link.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink'):cleanup.tree_state(link)
    assert (target/'model').read_bytes()==b'protected'


def test_changed_pinned_inputs_fail_before_mutation(tmp_path):
    reports=tmp_path/'reports';reports.mkdir()
    for name in cleanup.PINNED:(tmp_path/name).write_text('{}')
    with pytest.raises(ValueError,match='Approved inventory changed'):cleanup.make_plan(tmp_path)


def test_atomic_journal_preserves_real_newlines(tmp_path):
    path=tmp_path/'journal.json'
    cleanup.write_journal(path,{'status':'planned','paths':['a','b']})
    cleanup.write_journal(path,{'status':'complete'})
    assert json.loads(path.read_text())=={'status':'complete'}
    assert not path.with_name('journal.json.tmp').exists()
