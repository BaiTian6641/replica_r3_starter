#!/usr/bin/env python3
"""Explicit integration gate: verified Spark code, tiny random CPU BF16 weights.

No native checkpoint is loaded. Not a unit test dependency, performance estimate
or capability result. The source and qualified environment must be provisioned.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from launch_spark_intrinsic_readout_environment_v1 import validate_environment
from replica_lab.intrinsic_memory_execution_v5 import bound_json,hash_file
from replica_lab.execution_manifest import build_execution_manifest
from replica_lab.intrinsic_memory_budget_v5 import atomic,sha_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    protocol=json.loads((ROOT/'configs/spark_intrinsic_diagnostic_v5.json').read_text())
    environment=validate_environment(bound_json(ROOT,protocol['environment']),root=ROOT)
    source=bound_json(ROOT,protocol['source']);directory=ROOT/source['source_root']
    execution=bound_json(ROOT,protocol['source_execution'])
    if build_execution_manifest(directory,artifact_sha256=sha_json(source['files']))!=execution:raise ValueError('source code differs')
    os.environ['HF_MODULES_CACHE']=str(args.output/'modules')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from replica_lab.intrinsic_memory_native_v5 import verified_source_classes,base_hash
    from replica_lab.architecture.spark2_5 import load_spark2_5_config,official_bf16_precision
    from replica_lab.architecture.spark_adapter import Spark2_5Adapter
    from replica_lab.architecture.intrinsic_memory_layout_v4 import spark_layout
    from replica_lab.contextual_memory_v5 import ContextualMemoryV5
    from replica_lab.contextual_memory import canonical_spans
    from replica_lab.contextual_completion_objective_v1 import CompletionObjectiveSpec,stop_policy_identity,encode_complete_response,complete_response_loss
    from replica_lab.tiered_memory import MemoryScope
    torch.set_num_threads(1);torch.manual_seed(17)
    raw=json.loads((directory/'config.json').read_text());classes=verified_source_classes(directory,raw,execution)
    raw.update(hidden_size=64,intermediate_size=128,num_attention_heads=4,num_key_value_heads=2,
               head_dim=16,vocab_size=64,max_position_embeddings=256,sliding_window=8,use_cache=False)
    tiny_config=classes['AutoConfig'](**raw);tiny_config._attn_implementation='eager'
    base=classes['AutoModelForCausalLM'](tiny_config).to(dtype=torch.bfloat16).eval()
    config=load_spark2_5_config(raw,require_official_geometry=False)
    adapter=Spark2_5Adapter(base,config,official_bf16_precision(config,artifact_sha256=sha_json(execution)),
        device='cpu',require_official_geometry=False,projector_in_dim=64,memory_size=256,heads=4,spine_slots=8)
    adapter.seal_original_identity()
    spec=CompletionObjectiveSpec(64,8,1,2,(1,),stop_policy_identity((1,)),'a'*64,'b'*64,'none')
    path=ContextualMemoryV5(adapter,spark_layout('two',hidden_size=64),source_identity='random_official_code_CPU_v5',objective_identity=spec.identity())
    adapter.extension.spine=None;before=base_hash(base);scope=MemoryScope('cpu','main','probe',1000)
    spans=canonical_spans('The badge is red.',record_id='r',source_id='s',source_epoch=1,available_at=1)
    enc=encode_complete_response([3,5],[7],spec=spec);gradients=[]
    with torch.no_grad():
        b=path.training_values(spans,scope=scope)
        disabled=path.forward(enc.input_ids,enc.attention_mask,None,scope=scope,enabled=False)
        enabled=path.forward(enc.input_ids,enc.attention_mask,b,scope=scope)
        assert torch.equal(disabled,enabled)
    for _ in range(2):
        path.optimizer.zero_grad(set_to_none=True);b=path.training_values(spans,scope=scope)
        logits=path.forward(enc.input_ids,enc.attention_mask,b,scope=scope)
        loss=complete_response_loss(logits,enc,spec=spec);loss.loss.backward()
        norms={name:sum(float(p.grad.float().square().sum()) for p in m.parameters() if p.grad is not None)**0.5 for name,m in path.modules().items()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for m in path.modules().values() for p in m.parameters())
        path.optimizer.step();gradients.append(norms)
    assert gradients[1]['producer']>0 and base_hash(base)==before
    atomic(args.output/'result.json',{'format':'replica.intrinsic_memory.official_code_CPU_probe.v5',
        'source_code':execution['files'],'environment':environment,'passed':True,
        'layers':36,'hidden_size':64,'vocabulary':64,'original_dtype':'bfloat16','disabled_parity_exact':True,
        'optimizer':'NativeFusedAdamW','updates':2,'gradients':gradients,'original_weights_unchanged':True,
        'original_checkpoint_weights_loaded':False,'native_execution':False,'useful_memory':False,
        'scope':'sealed_Spark_code_scaled_random_CPU_configuration_only',
        'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

if __name__=='__main__':main()
