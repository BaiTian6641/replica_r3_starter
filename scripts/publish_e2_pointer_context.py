#!/usr/bin/env python3
"""Publish the context-head candidate metadata and rollback (CPU only)."""
from __future__ import annotations
import argparse, hashlib, json, sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def candidate_dict(c):
    d={f.name:getattr(c,f.name) for f in fields(c)}
    d['changed_parameters']=list(c.changed_parameters); d['source_ids']=list(c.source_ids); d['producer_versions']=dict(c.producer_versions)
    return d
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,default=ROOT/'configs/e2_pointer_context_publication_v1.json')
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): p.error('Use a fresh output directory')
    a.output.mkdir(parents=True, exist_ok=True)
    c=json.loads(a.config.read_text()); run=json.loads((ROOT/c['run_report']).read_text())
    cm=json.loads((ROOT/c['context_manifest']).read_text())
    if sha(ROOT/c['context_head'])!=cm['weights_sha256']: raise ValueError('context head bytes changed')
    pointer_manifest=json.loads((ROOT/cm['parent_pointer_extension']/'manifest.json').read_text())
    if pointer_manifest['weights_sha256']!=cm['parent_pointer_weights_sha256']: raise ValueError('pointer parent changed')
    source_ids=tuple(sorted(q['case_id'] for q in
        (json.loads(line) for line in (ROOT/'data/e2_copy/queries.jsonl').read_text().splitlines() if line.strip())
        if q['split']=='train'))
    from replica_lab.runtime.artifacts import NeuralCandidate,ArtifactRegistry
    from replica_lab.learning import ManifestRegistry
    from replica_lab.contracts import fingerprint
    registry=ArtifactRegistry()
    cand=NeuralCandidate('e2-pointer-context-1',pointer_manifest['manifest_digest'],
        ('linear.weight','linear.bias'),cm['weights_sha256'],'modules_only','adamw',
        {'pointer':pointer_manifest['manifest_digest']},source_ids,evaluation='passed',status='proposed')
    registry.add(cand); registry.transition(cand.artifact_id,'validated'); cand=registry.transition(cand.artifact_id,'accepted')
    harmful=NeuralCandidate('e2-pointer-context-harmful',pointer_manifest['manifest_digest'],
        ('linear.weight','linear.bias'),cm['weights_sha256'],'modules_only','adamw',
        {'pointer':pointer_manifest['manifest_digest']},source_ids,evaluation='failed',status='proposed')
    registry.add(harmful); harmful=registry.transition(harmful.artifact_id,'rejected')
    catalog=registry.save(a.output/'artifact_catalog.json')
    reg=ManifestRegistry(a.output/'manifest_registry')
    base_payload={'base_revision':c['source_revision'],'identity_version':'e2_exact_text','candidate':'exact_text_baseline'}
    base_eval={'decision':'accept','candidate_digest':fingerprint(base_payload),'parent_manifest_id':None,'target':'e2_exact_text'}
    base=reg.publish(base_payload,base_eval,expected_parent=None)
    payload={'base_revision':c['source_revision'],'identity_version':'e2_pointer_context_opt_in',
        'candidate':'e2-pointer-context-1','context_head':str(ROOT/c['context_head']),
        'weights_sha256':cm['weights_sha256'],'pointer_extension':str(ROOT/cm['parent_pointer_extension']),
        'pointer_weights_sha256':pointer_manifest['weights_sha256'],'default':False,
        'features':cm['features'],'evaluation':run['metrics']}
    ev={'decision':'accept','candidate_digest':fingerprint(payload),'parent_manifest_id':base['manifest_id'],'target':'e2_pointer_context'}
    published=reg.publish(payload,ev,expected_parent=base['manifest_id'])
    rolled=reg.rollback(base['manifest_id'])
    report={'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'candidate':candidate_dict(cand),'harmful':{'candidate':candidate_dict(harmful),'rejected':harmful.status=='rejected'},
        'artifact_catalog':catalog,'baseline_manifest':base['manifest_id'],'candidate_manifest':published['manifest_id'],
        'rolled_back':rolled['manifest_id'],'active_after_rollback':reg.active()['manifest_id'],
        'claims':'Context-head publication metadata and rollback only; exact text remains default; not integrated as a default.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'candidate_manifest':published['manifest_id'],
                      'rolled_back':report['active_after_rollback']==base['manifest_id']}))
if __name__=='__main__': main()
