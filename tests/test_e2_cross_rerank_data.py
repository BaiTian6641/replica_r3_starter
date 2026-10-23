"""Cross-template reranker data: splits and eligible candidates."""
import hashlib,json
from pathlib import Path
from replica_lab.memory_system import MemoryRecord,RecordLedger
from replica_lab.memory_system.index import RetrievalIndex
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/e2_cross_rerank'
def test_cross_rerank_manifest(tmp_path):
    m=json.loads((DATA/'manifest.json').read_text())
    assert hashlib.sha256((DATA/'records.jsonl').read_bytes()).hexdigest()==m['files']['records_sha256']
    assert m['counts']['by_split']=={'train':8,'dev':4,'heldout':8}
    groups={}
    for g,v in m['groups'].items():groups.setdefault(v['split'],set()).add(g)
    assert not(groups['train']&groups['dev'] or groups['train']&groups['heldout'] or groups['dev']&groups['heldout'])
    records=[MemoryRecord(**json.loads(x)) for x in (DATA/'records.jsonl').read_text().splitlines() if x]
    queries=[json.loads(x) for x in (DATA/'queries.jsonl').read_text().splitlines() if x]
    led=RecordLedger(tmp_path/'c.sqlite')
    try:
        for r in records:led.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
        idx=RetrievalIndex(led)
        for q in queries:
            assert q['expected_answer'].casefold() not in q['query'].casefold()
            sc={'subject_id':q['subject_id'],'branch_id':q['branch_id'],'purpose':q['purpose'],'as_of':q['as_of'],'mode':'assistant_for_source','snapshot_at':None}
            sel=idx.retrieve(q['query'],**sc,limit=16)
            assert q['expected_record_id'] in [r.record_id for r in sel.records]
    finally:led.close()
