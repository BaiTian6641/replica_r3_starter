"""Source-separated E4 delayed-use pages: splits and observation isolation."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/e4_delayed'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_delayed_pages_are_source_separated_and_do_not_leak_answers():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert sha(DATA / 'records.jsonl') == manifest['files']['records_sha256']
    assert sha(DATA / 'queries.jsonl') == manifest['files']['queries_sha256']
    assert manifest['counts']['by_split'] == {'train': 24, 'dev': 6, 'heldout': 8}
    queries = [json.loads(line) for line in
               (DATA / 'queries.jsonl').read_text().splitlines() if line.strip()]
    groups = {}
    for group, value in manifest['groups'].items():
        groups.setdefault(value['split'], set()).add(group)
    assert not (groups['train'] & groups['dev'] or groups['train'] & groups['heldout']
                or groups['dev'] & groups['heldout'])
    for query in queries:
        assert query['expected_answer'].casefold() not in query['query'].casefold()
        assert query['expected_answer'] in query['observation']
        assert max(manifest['observation_tokens'].values()) <= 256
