"""The 32-unit capacity bank has exactly 32 eligible source units."""
import hashlib,json
from pathlib import Path
from replica_lab.memory_system import MemoryRecord
from replica_lab.memory_system.session import MemorySession

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/e2_capacity32'
def test_capacity32_manifest_and_spans():
    m=json.loads((DATA/'manifest.json').read_text())
    assert m['units']==32 and m['records']==29
    assert hashlib.sha256((DATA/'records.jsonl').read_bytes()).hexdigest()==m['files']['records_sha256']
    records=[MemoryRecord(**json.loads(x)) for x in (DATA/'records.jsonl').read_text().splitlines() if x]
    from replica_lab.memory_system import RecordLedger
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        ledger=RecordLedger(Path(directory)/'c.sqlite')
        try:
            for r in records:ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
            # A tokenizer is needed only to split source spans; use the sealed tokenizer.
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/minicpm5_source',local_files_only=True,trust_remote_code=False)
            from types import SimpleNamespace
            class Session(MemorySession):
                def __init__(self):
                    self.ledger=ledger; self.tokenizer=tok; self._cache={}; self.max_candidate_units=64
                    self.adapter=SimpleNamespace(ports=SimpleNamespace(source_tokens=128))
            session=Session()
            spans=session.spans(records)
            assert len(spans)==32
            assert all(s.token_end>s.token_start for s in spans)
        finally:ledger.close()
