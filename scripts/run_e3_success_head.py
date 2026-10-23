#!/usr/bin/env python3
"""Train/evaluate a pooled E3 outcome head on actual episodes (CPU only)."""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_episodes(paths):
    rows=[]
    for path in paths:
        obj=json.loads(Path(path).read_text())
        for key in ('results','controls','episodes'):
            for item in obj.get(key,[]):
                fv=item.get('summary',{}).get('feature_vector')
                if fv is None or len(fv)!=280: continue
                rows.append({'case_id':item.get('case_id'),'mode':item.get('arm') or item.get('mode'),
                    'success':bool(item.get('success')),'vector':fv,'source':str(path)})
        summary=obj.get('summary')
        if summary and summary.get('feature_vector') and obj.get('case_id'):
            rows.append({'case_id':obj['case_id'],'mode':summary.get('mode'),
                'success':bool(summary.get('success')),'vector':summary['feature_vector'],'source':str(path)})
    # Deduplicate exact report rows by case/mode/vector, retaining the first.
    seen=set(); unique=[]
    for row in rows:
        key=(row['case_id'],row['mode'],tuple(row['vector']),row['success'])
        if key not in seen: seen.add(key); unique.append(row)
    return unique
def train_eval(rows,steps=300,hidden=64):
    from replica_lab.architecture.contracts import CoreSpec
    from replica_lab.architecture.heads import PooledTaskOutcomeHead
    spec=CoreSpec(width=256,slots=8,steps=2,heads=4,task_features=24,ffn_hidden=1024,control_hidden=340)
    cases=sorted({r['case_id'] for r in rows}); folds=[]
    for held in cases:
        train=[r for r in rows if r['case_id']!=held]; test=[r for r in rows if r['case_id']==held]
        if not train or not test: continue
        torch.manual_seed(20261023); head=PooledTaskOutcomeHead(spec,hidden)
        opt=torch.optim.AdamW(head.parameters(),lr=0.01,weight_decay=0.0)
        x=torch.tensor([r['vector'][:24] for r in train],dtype=torch.float32)
        w=torch.tensor([r['vector'][24:] for r in train],dtype=torch.float32)
        y=torch.tensor([float(r['success']) for r in train])
        for _ in range(steps):
            opt.zero_grad(); loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x,w),y)
            loss.backward(); opt.step()
        xt=torch.tensor([r['vector'][:24] for r in test],dtype=torch.float32)
        wt=torch.tensor([r['vector'][24:] for r in test],dtype=torch.float32)
        with torch.no_grad(): prob=head(xt,wt).sigmoid()
        folds.append({'held_case':held,'train':len(train),'test':len(test),
            'test_success':[r['success'] for r in test],'probabilities':[float(v) for v in prob],
            'accuracy':sum((p>=0.5)==r['success'] for p,r in zip(prob.tolist(),test))/len(test)})
    return folds
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--protocol',type=Path,default=ROOT/'configs/e3_success_head_v1.json')
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): p.error('Use a fresh output directory')
    protocol=json.loads(a.protocol.read_text()); paths=[ROOT/x for x in protocol['episode_reports']]
    rowset=load_episodes(paths)
    report={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'protocol_sha256':sha(a.protocol),'protocol':protocol,'downloads':0,
            'optimizer_updates':0,'checkpoints_saved':0,'episodes':len(rowset),
            'positives':sum(r['success'] for r in rowset),'negatives':sum(not r['success'] for r in rowset),
            'cases':sorted({r['case_id'] for r in rowset})}
    a.output.mkdir(parents=True,exist_ok=True)
    folds=train_eval(rowset,int(protocol['steps']),int(protocol['hidden']))
    report['folds']=folds
    report['metrics']={'cases':len(folds),'mean_accuracy':sum(f['accuracy'] for f in folds)/len(folds) if folds else None,
        'brier':sum((p-r)**2 for f in folds for p,r in zip(f['probabilities'],f['test_success']))/sum(len(f['probabilities']) for f in folds) if folds else None,
        'majority_baseline':max(report['positives'],report['negatives'])/len(rowset) if rowset else None,
        'calibration':'none; case-held-out only, no disjoint calibration role'}
    # Save the all-data head as a diagnostic artifact, never an active default.
    from replica_lab.architecture.contracts import CoreSpec
    from replica_lab.architecture.heads import PooledTaskOutcomeHead
    from safetensors.torch import save_file
    spec=CoreSpec(width=256,slots=8,steps=2,heads=4,task_features=24,ffn_hidden=1024,control_hidden=340)
    torch.manual_seed(protocol['seed']); head=PooledTaskOutcomeHead(spec,int(protocol['hidden']))
    opt=torch.optim.AdamW(head.parameters(),lr=protocol['lr'],weight_decay=0.0)
    x=torch.tensor([r['vector'][:24] for r in rowset],dtype=torch.float32); w=torch.tensor([r['vector'][24:] for r in rowset],dtype=torch.float32)
    y=torch.tensor([float(r['success']) for r in rowset])
    for _ in range(int(protocol['steps'])):
        opt.zero_grad(); loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x,w),y); loss.backward(); opt.step()
    save_file({n:p.detach().cpu() for n,p in head.named_parameters()},str(a.output/'success_head.safetensors'))
    manifest={'format':'replica.e3.pooled_success_head.v1','created_at_utc':'2026-10-23',
      'input':'task_features[24] + runtime pooled workspace[256]','hidden':protocol['hidden'],
      'episode_reports':protocol['episode_reports'],'episodes':len(rowset),'positives':report['positives'],'negatives':report['negatives'],
      'weights_sha256':sha(a.output/'success_head.safetensors'),'calibration':'none','authority':'monitor only; never success authority','base_saved':False}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); report['artifact']=manifest
    report['optimizer_updates']=int(protocol['steps']); report['status']='completed'
    (a.output/'run_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'episodes':len(rowset),'metrics':report['metrics']}))
if __name__=='__main__': main()
