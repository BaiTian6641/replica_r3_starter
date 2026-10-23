"""Fresh E4 contextual data: source-separated groups and observation isolation."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/e4_contextual'
def test_contextual_data_manifest():
    m=json.loads((DATA/'manifest.json').read_text())
    assert hashlib.sha256((DATA/'records.jsonl').read_bytes()).hexdigest()==m['files']['records_sha256']
    assert m['counts']['by_split']=={'train':24,'dev':6,'heldout':8}
    groups={}
    for g,v in m['groups'].items(): groups.setdefault(v['split'],set()).add(g)
    assert not (groups['train']&groups['dev'] or groups['train']&groups['heldout'] or groups['dev']&groups['heldout'])
    for line in (DATA/'queries.jsonl').read_text().splitlines():
        q=json.loads(line)
        assert q['expected_answer'].casefold() not in q['query'].casefold()
        assert q['expected_answer'] in q['observation']
        assert len(__import__('transformers').AutoTokenizer.from_pretrained(ROOT/'checkpoints/minicpm5_source',local_files_only=True)(q['observation'],add_special_tokens=False)['input_ids'])<=128
