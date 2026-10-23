"""Fresh pointer unit comparison family: manifest and lexical target."""
import hashlib,json
from pathlib import Path
from replica_lab.memory_system import MemoryRecord,RecordLedger
from replica_lab.memory_system.index import RetrievalIndex
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/e2_pointer_fresh2'
def test_pointer_unit_manifest_and_rank(tmp_path):
    m=json.loads((DATA/'manifest.json').read_text())
    assert hashlib.sha256((DATA/'records.jsonl').read_bytes()).hexdigest()==m['files']['records_sha256']
    assert m['counts']=={'cases':12,'records':48,'by_split':{'heldout':12}}
    records=[MemoryRecord(**json.loads(x)) for x in (DATA/'records.jsonl').read_text().splitlines() if x]
    queries=[json.loads(x) for x in (DATA/'queries.jsonl').read_text().splitlines() if x]
    ledger=RecordLedger(tmp_path/'u.sqlite')
    try:
        for r in records: ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
        idx=RetrievalIndex(ledger)
        for q in queries:
            assert q['expected_answer'].casefold() not in q['query'].casefold()
            scope={'subject_id':q['subject_id'],'branch_id':q['branch_id'],'purpose':q['purpose'],'as_of':q['as_of'],'mode':'assistant_for_source','snapshot_at':None}
            sel=idx.retrieve(q['query'],**scope,limit=16)
            assert sel.records[0].record_id==q['expected_record_id']
    finally: ledger.close()
