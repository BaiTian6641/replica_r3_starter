#!/usr/bin/env python3
"""Publish the opt-in E2 pointer candidate metadata and test rollback (CPU)."""
from __future__ import annotations
import argparse, hashlib, json, sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def candidate_dict(c):
    value = {f.name: getattr(c, f.name) for f in fields(c)}
    value['changed_parameters'] = list(c.changed_parameters)
    value['source_ids'] = list(c.source_ids)
    value['producer_versions'] = dict(c.producer_versions)
    return value
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,default=ROOT/'configs/e2_pointer_publication_v1.json')
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): p.error('Use a fresh output directory')
    a.output.mkdir(parents=True,exist_ok=True); c=json.loads(a.config.read_text())
    run=json.loads((ROOT/c['run_report']).read_text()); extension=run['extension']; parent=run['parent']
    weights=extension['weights_sha256']; parent_digest=parent['manifest_digest']
    changed=tuple(n for n in extension['parameters']['new']['names'] if n.startswith('copy_head'))
    if not changed: raise ValueError('pointer copy parameters missing')
    data_root = ROOT / run['protocol']['dataset']['root']
    source_ids = tuple(sorted(q['case_id'] for q in
        (json.loads(line) for line in (data_root / 'queries.jsonl').read_text().splitlines() if line.strip())
        if q['split'] == 'train'))
    from replica_lab.runtime.artifacts import NeuralCandidate,ArtifactRegistry
    from replica_lab.learning import ManifestRegistry
    from replica_lab.contracts import fingerprint
    registry=ArtifactRegistry()
    pointer=NeuralCandidate('e2-pointer-copy-1',parent_digest,changed,weights,'modules_only','muon8bit',
        {'e2_parent':parent_digest},source_ids,evaluation='passed',status='proposed')
    registry.add(pointer); registry.transition(pointer.artifact_id,'validated'); pointer=registry.transition(pointer.artifact_id,'accepted')
    harmful=NeuralCandidate('e2-pointer-copy-harmful',parent_digest,changed,weights,'modules_only','muon8bit',
        {'e2_parent':parent_digest},source_ids,evaluation='failed',status='proposed')
    registry.add(harmful); harmful=registry.transition(harmful.artifact_id,'rejected')
    catalog=registry.save(a.output/'artifact_catalog.json')
    basereg=ManifestRegistry(a.output/'manifest_registry')
    baseline_payload={'base_revision':c['source_revision'],'identity_version':'e2_exact_text','candidate':'exact_text_baseline'}
    baseline_eval={'decision':'accept','candidate_digest':fingerprint(baseline_payload),'parent_manifest_id':None,'target':'e2_exact_text'}
    baseline=basereg.publish(baseline_payload,baseline_eval,expected_parent=None)
    pointer_payload={'base_revision':c['source_revision'],'identity_version':'e2_pointer_opt_in','candidate':'e2-pointer-copy-1',
        'extension_path':extension['path'],'weights_sha256':weights,'parent_manifest_digest':parent_digest,
        'default':False,'evaluation':{'dev':run['metrics']['enabled']['dev'],'heldout':run['metrics']['enabled']['heldout'],
        'shuffled_heldout':run['metrics']['shuffled']['heldout']}}
    pointer_eval={'decision':'accept','candidate_digest':fingerprint(pointer_payload),'parent_manifest_id':baseline['manifest_id'],'target':'e2_pointer_copy'}
    published=basereg.publish(pointer_payload,pointer_eval,expected_parent=baseline['manifest_id'])
    rolled=basereg.rollback(baseline['manifest_id'])
    report={'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'run_report':str((ROOT/c['run_report']).resolve()),'pointer_candidate':candidate_dict(pointer),
        'harmful_candidate':{'artifact':candidate_dict(harmful),'rejected':harmful.status=='rejected'},
        'artifact_catalog':catalog,'baseline_manifest':baseline['manifest_id'],
        'pointer_manifest':published['manifest_id'],'rolled_back':rolled['manifest_id'],
        'active_after_rollback':basereg.active()['manifest_id'],
        'claims':'Pointer candidate publication metadata and rollback only; exact text remains default; no neural runtime promotion.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'pointer_manifest':published['manifest_id'],
                      'rolled_back':report['active_after_rollback']==baseline['manifest_id']}))
if __name__=='__main__': main()
