"""The 64-unit capacity bank has exactly 64 eligible source units."""
import hashlib,json,tempfile
from pathlib import Path
import pytest
from conftest import require_local_artifacts
from replica_lab.memory_system import MemoryRecord,RecordLedger
from replica_lab.memory_system.session import MemorySession
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/e2_capacity64'
@require_local_artifacts("checkpoints/minicpm5_source")
@pytest.mark.integration
def test_capacity64_manifest_and_spans():
    m=json.loads((DATA/'manifest.json').read_text())
    assert m['units']==64 and m['records']==61
    assert hashlib.sha256((DATA/'records.jsonl').read_bytes()).hexdigest()==m['files']['records_sha256']
    records=[MemoryRecord(**json.loads(x)) for x in (DATA/'records.jsonl').read_text().splitlines() if x]
    with tempfile.TemporaryDirectory() as directory:
        ledger=RecordLedger(Path(directory)/'c.sqlite')
        try:
            for r in records:ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/minicpm5_source',local_files_only=True,trust_remote_code=False)
            from types import SimpleNamespace
            class Session(MemorySession):
                def __init__(self):
                    self.ledger=ledger; self.tokenizer=tok; self._cache={}; self.max_candidate_units=128
                    self.adapter=SimpleNamespace(ports=SimpleNamespace(source_tokens=128))
            assert len(Session().spans(records))==64
        finally:ledger.close()
