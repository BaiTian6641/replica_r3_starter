import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parents[1]/'data/e2_candidate_answers2'
def test_data():
 m=json.loads((D/'manifest.json').read_text());assert m['cases']==4 and m['records']==16
 assert hashlib.sha256((D/'records.jsonl').read_bytes()).hexdigest()==m['files']['records_sha256']
 for q in map(json.loads,(D/'queries.jsonl').read_text().splitlines()):
  assert q['expected_answer'] in q['candidates']
