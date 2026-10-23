import hashlib,json
from pathlib import Path
from replica_lab.memory_system import MemoryRecord
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/e2_candidate_answers'
def test_candidate_answer_data():
 m=json.loads((DATA/'manifest.json').read_text());assert m['cases']==4 and m['records']==16
 assert hashlib.sha256((DATA/'records.jsonl').read_bytes()).hexdigest()==m['files']['records_sha256']
 records=[MemoryRecord(**json.loads(x)) for x in (DATA/'records.jsonl').read_text().splitlines() if x]
 queries=[json.loads(x) for x in (DATA/'queries.jsonl').read_text().splitlines() if x]
 assert len(records)==16 and len(queries)==4
 for q in queries:
  assert q['expected_answer'] in q['candidates']
  assert len(q['candidates'])==len(set(q['candidates']))
