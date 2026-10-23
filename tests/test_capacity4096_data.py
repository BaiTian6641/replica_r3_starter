"""The 4096-unit capacity bank has exactly 4096 eligible units."""
import hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from replica_lab.memory_system import MemoryRecord,RecordLedger
from replica_lab.memory_system.session import MemorySession
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/e2_capacity4096'
def test_capacity4096_data():
 m=json.loads((DATA/'manifest.json').read_text());assert m['units']==4096 and m['records']==4093
 assert hashlib.sha256((DATA/'records.jsonl').read_bytes()).hexdigest()==m['files']['records_sha256']
 records=[MemoryRecord(**json.loads(x)) for x in (DATA/'records.jsonl').read_text().splitlines() if x]
 with tempfile.TemporaryDirectory() as d:
  ledger=RecordLedger(Path(d)/'c.sqlite')
  try:
   for r in records:ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
   from transformers import AutoTokenizer
   tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/minicpm5_source',local_files_only=True,trust_remote_code=False)
   class S(MemorySession):
    def __init__(self):self.ledger=ledger;self.tokenizer=tok;self._cache={};self.max_candidate_units=8192;self.adapter=SimpleNamespace(ports=SimpleNamespace(source_tokens=128))
   assert len(S().spans(records))==4096
  finally:ledger.close()
