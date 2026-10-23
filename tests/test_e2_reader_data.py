"""Source-separated E2 reader candidate data: splits, leakage and lexical rank."""

import hashlib
import json
from pathlib import Path

from replica_lab.memory_system import EligibilityScope, MemoryRecord, RecordLedger
from replica_lab.memory_system.index import RetrievalIndex

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/e2_reader'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_reader_dataset_manifest_and_splits():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert sha(DATA / 'records.jsonl') == manifest['files']['records_sha256']
    assert sha(DATA / 'queries.jsonl') == manifest['files']['queries_sha256']
    assert manifest['counts']['by_split'] == {'train': 24, 'dev': 6, 'heldout': 10}
    splits = {}
    for group, value in manifest['groups'].items():
        splits.setdefault(value['split'], set()).add(group)
    assert not (splits['train'] & splits['dev'] or splits['train'] & splits['heldout']
                or splits['dev'] & splits['heldout'])


def test_reader_dataset_lexical_selection_and_no_answer_leak(tmp_path):
    records = [MemoryRecord(**json.loads(line)) for line in
               (DATA / 'records.jsonl').read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in
               (DATA / 'queries.jsonl').read_text().splitlines() if line.strip()]
    ledger = RecordLedger(tmp_path / 'reader.sqlite')
    try:
        for record in records:
            ledger.admit(record, authorized_subject=record.subject_id,
                         authorized_branch=record.branch_id, source_authorized=False)
        index = RetrievalIndex(ledger)
        for query in queries:
            assert query['expected_answer'].casefold() not in query['query'].casefold()
            scope = EligibilityScope(query['subject_id'], query['branch_id'],
                                     query['purpose'], query['as_of'],
                                     'assistant_for_source', None)
            selection = index.retrieve(query['query'], **{
                name: getattr(scope, name) for name in
                ('subject_id', 'branch_id', 'purpose', 'as_of', 'mode', 'snapshot_at')},
                limit=16)
            assert selection.records[0].record_id == query['expected_record_id']
    finally:
        ledger.close()
