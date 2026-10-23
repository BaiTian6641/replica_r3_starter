"""Committed E2 retrieval/capacity datasets match their manifests."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_retrieval_dataset_validator():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/verify_e2_retrieval_data.py')],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['ok'] is True
    assert report['cases'] == 13 and report['records'] == 49
    assert set(report['splits']) == {'train', 'dev', 'heldout'}
    assert report['long_record']['fact_token_start'] >= 128


def test_retrieval_manifest_hashes_and_split_disjointness():
    root = ROOT / 'data/e2_retrieval'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert sha(root / 'records.jsonl') == manifest['files']['records_sha256']
    assert sha(root / 'queries.jsonl') == manifest['files']['queries_sha256']
    splits = [set(manifest['splits'][name]) for name in ('train', 'dev', 'heldout')]
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])


def test_capacity_manifest_declares_exact_units():
    root = ROOT / 'data/e2_capacity'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert sha(root / 'records.jsonl') == manifest['files']['records_sha256']
    assert manifest['units'] == 16
    assert manifest['short_records'] == 12
    assert manifest['long_record_spans'] == 4
    assert all(value == 128 for key, value in manifest['token_counts'].items()
               if key.startswith('cap_short_'))
