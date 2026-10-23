"""The 8192-unit capacity bank has exactly 8192 eligible units."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/e2_capacity8192'
def test_capacity8192_data():
 m=json.loads((DATA/'manifest.json').read_text());assert m['units']==8192 and m['records']==8189
 assert hashlib.sha256((DATA/'records.jsonl').read_bytes()).hexdigest()==m['files']['records_sha256']
