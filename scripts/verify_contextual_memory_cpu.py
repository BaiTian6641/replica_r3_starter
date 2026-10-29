#!/usr/bin/env python3
"""Independent CPU numerical/storage diagnostics; never a competence bypass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import torch.nn.functional as F
from replica_lab.contextual_memory import ContextualMemoryPath, canonical_spans, tensor_digest
from replica_lab.contextual_spark_fixture import build_contextual_adapter, ids_and_mask, labels_from_ids
from replica_lab.tiered_memory import MemoryScope, TierCeilings, TieredMemoryStore
from replica_lab.profiling import _ru_maxrss_bytes


def make_path():
    torch.manual_seed(17)
    adapter,*_ = build_contextual_adapter(seed=17,layers=8,spine_sites=(3,7),spine_slots=8,memory_size=256)
    return ContextualMemoryPath(adapter,lr=0.01)


def spans():
    return canonical_spans("The crimson badge means ready.",record_id="rule",source_id="fixture",
                           source_epoch=1,available_at=1)


def step(path):
    ids,mask=ids_and_mask(); scope=MemoryScope("fixture","main","assistant",as_of=10)
    batch=path.training_values(spans(),scope=scope)
    path.optimizer.zero_grad(set_to_none=True)
    logits=path.forward(ids,mask,batch,scope=scope)
    loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels_from_ids(ids).reshape(-1))
    loss.backward()
    gradients={g['name']:sum(float(p.grad.detach().abs().sum()) for p in g['params'] if p.grad is not None)
               for g in path.optimizer.param_groups}
    if not all(torch.isfinite(p.grad).all() for g in path.optimizer.param_groups for p in g['params'] if p.grad is not None):
        raise ValueError("nonfinite diagnostic gradients")
    path.optimizer.step()
    return {'loss':float(loss.detach()),'gradients':gradients}


def identity(path):
    return {'producer':path.encoder.release(),'order':tensor_digest(path.order.weight),
            'reader':{n:tensor_digest(t) for n,t in path.adapter.extension.spine.ports['memory'].state_dict().items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--restart',action='store_true')
    args=parser.parse_args(); output=args.output.resolve(); start=time.monotonic()
    path=make_path(); ids,mask=ids_and_mask(); scope=MemoryScope('fixture','main','assistant',as_of=10)
    if args.restart:
        path.load(output/'extension')
        with torch.no_grad(): logits=path.forward(ids,mask,path.training_values(spans(),scope=scope),scope=scope)
        result={'logits_sha256':tensor_digest(logits),'continuation':step(path),'state_after_update':identity(path)}
        (output/'restart.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    output.mkdir(parents=True,exist_ok=False)
    before=path.forward(ids,mask,None,scope=scope)
    values=path.training_values(spans(),scope=scope)
    parity=torch.equal(before,path.forward(ids,mask,values,scope=scope))
    traces=[step(path) for _ in range(3)]
    path.save(output/'extension')
    store=TieredMemoryStore(output/'store',ceilings=TierCeilings(minimum_volume_free_bytes=0))
    path.admit(store,spans()[0],scope=scope)
    batch=path.read(store,'badge',scope=scope)
    with torch.no_grad(): expected=path.forward(ids,mask,batch,scope=scope)
    store.close()
    expected_logits=tensor_digest(expected)
    step(path); continued=identity(path)
    command=[sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--restart']
    child=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True,timeout=120)
    (output/'child.stdout').write_text(child.stdout);(output/'child.stderr').write_text(child.stderr)
    if child.returncode:raise RuntimeError(child.stderr)
    fresh=json.loads((output/'restart.json').read_text())
    report={'format':'replica.contextual_memory.cpu_diagnostics.v2','native_execution':False,
            'implementation_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__).resolve(), ROOT/'src/replica_lab/contextual_memory.py',
                          ROOT/'src/replica_lab/contextual_readout_training.py', ROOT/'src/replica_lab/tiered_memory.py']},
            'evidence_class':'independent_engineering_diagnostics_not_competence_or_learning_utility',
            'zero_init_parity':parity,'fit_updates':3,'fit':traces,
            'producer_gradient_after_output_opens':traces[1]['gradients']['producer']>0,
            'fresh_process_exact_logits':fresh['logits_sha256']==expected_logits,
            'fresh_process_exact_continuation':fresh['state_after_update']==continued,
            'frozen_base_integrity':path.adapter.base_unchanged(),'cost':path.cost,
            'wall_seconds':time.monotonic()-start,'parent_peak_rss_bytes':_ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sys.platform),
            'child_peak_rss_bytes':_ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,sys.platform),
            'artifact_bytes':sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
            'native_driver_bytes':None,'host_swap_sampled':False,'promotion':False,'learned_capability':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not all(report[k] for k in ('zero_init_parity','producer_gradient_after_output_opens','fresh_process_exact_logits','fresh_process_exact_continuation','frozen_base_integrity')):
        raise RuntimeError('CPU engineering diagnostic failed')


if __name__=='__main__':main()
