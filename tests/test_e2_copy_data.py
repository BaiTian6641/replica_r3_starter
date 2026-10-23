"""Fresh E2 copy data: split integrity, lexical target and source answer presence."""
import hashlib,json
from pathlib import Path
from replica_lab.memory_system import MemoryRecord,RecordLedger
from replica_lab.memory_system.index import RetrievalIndex
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/e2_copy'
def test_copy_data_manifest_and_target_rank(tmp_path):
    m=json.loads((DATA/'manifest.json').read_text())
    assert hashlib.sha256((DATA/'records.jsonl').read_bytes()).hexdigest()==m['files']['records_sha256']
    assert m['counts']['by_split']=={'train':24,'dev':6,'heldout':10}
    records=[MemoryRecord(**json.loads(x)) for x in (DATA/'records.jsonl').read_text().splitlines() if x]
    queries=[json.loads(x) for x in (DATA/'queries.jsonl').read_text().splitlines() if x]
    by={r.record_id:r for r in records}
    ledger=RecordLedger(tmp_path/'c.sqlite')
    try:
        for r in records: ledger.admit(r,authorized_subject=r.subject_id,authorized_branch=r.branch_id)
        index=RetrievalIndex(ledger)
        for q in queries:
            expected=by[q['expected_record_id']]
            assert q['expected_answer'].casefold() in expected.text.casefold()
            assert q['expected_answer'].casefold() not in q['query'].casefold()
            scope={'subject_id':q['subject_id'],'branch_id':q['branch_id'],'purpose':q['purpose'],'as_of':q['as_of'],'mode':'assistant_for_source','snapshot_at':None}
            sel=index.retrieve(q['query'],**scope,limit=16)
            assert sel.records[0].record_id==q['expected_record_id']
    finally: ledger.close()
