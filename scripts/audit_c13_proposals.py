#!/usr/bin/env python3
"""Validate C13 checked-proposal data sufficiency; never train or publish."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from replica_lab.runtime.director_data import load_registry
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',type=Path,default=ROOT/'configs/c13_checked_proposals_v1.json');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();value=load_registry(a.registry);report={'format':'replica.c13.checked_proposals.audit.v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'registry_sha256':__import__('hashlib').sha256(a.registry.read_bytes()).hexdigest(),**value,'trained':False,'published':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'decision':value['decision'],'groups':value['groups'],'accepted':value['accepted'],'rejected':value['rejected_or_deferred']}))
if __name__=='__main__':main()
