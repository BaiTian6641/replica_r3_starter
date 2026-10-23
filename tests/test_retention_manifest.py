"""Retention inventory is read-only, bounded and path-safe."""
import json
from pathlib import Path
import pytest
from replica_lab.retention import build_manifest

def policy(root):
    (root/'preserve').mkdir(exist_ok=True);(root/'preserve'/'a.txt').write_text('a'*10)
    (root/'candidate').mkdir(exist_ok=True);(root/'candidate'/'b.txt').write_text('b'*20)
    return {'format':'replica.retention.policy.v1','entries':[
      {'path':'preserve','action':'preserve','reason':'needed'},
      {'path':'candidate','action':'deletion_candidate','reason':'old'}]}

def test_manifest_measures_and_never_deletes(tmp_path):
    p=policy(tmp_path);m=build_manifest(p,root=tmp_path)
    assert m['total_bytes']==30 and not m['deletion_performed']
    assert (tmp_path/'candidate'/'b.txt').exists()
    assert {x['path']:x['action'] for x in m['entries']}=={'preserve':'preserve','candidate':'deletion_candidate'}

def test_missing_preserve_and_escape_rejected(tmp_path):
    p=policy(tmp_path);p['entries'].append({'path':'missing','action':'preserve','reason':'x'})
    with pytest.raises(FileNotFoundError):build_manifest(p,root=tmp_path)
    p=policy(tmp_path);p['entries']=[{'path':'../escape','action':'preserve','reason':'x'}]
    with pytest.raises(PermissionError):build_manifest(p,root=tmp_path)
