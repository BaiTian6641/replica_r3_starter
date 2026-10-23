#!/usr/bin/env python3
"""Build a read-only retention manifest; never deletes artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from replica_lab.retention import build_manifest
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,default=ROOT/'configs/retention_policy_v1.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();policy=json.loads(a.policy.read_text());manifest=build_manifest(policy,root=ROOT)
    text=json.dumps(manifest,indent=2,sort_keys=True)+'\n';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(json.dumps({'status':'completed','entries':len(manifest['entries']),'total_bytes':manifest['total_bytes'],'deletion_performed':False}))
if __name__=='__main__':main()
