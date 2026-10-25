"""Validate the research package; does not load or train a model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import re
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCS = ROOT / 'docs/research/dynamic_engram_20261025'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    documents = sorted(DOCS.glob('*.md'))
    assert len(documents) == 5
    previous_index = ROOT / 'docs/research/sparkx_2_6_4b_20261024/README.md'
    local_links = 0
    external_urls = set()
    word_counts = {}
    for path in [*documents, previous_index]:
        text = path.read_text()
        assert sum(line.startswith('```') for line in text.splitlines()) % 2 == 0, path
        assert all(line.rstrip() == line for line in text.splitlines()), path
        if path in documents:
            word_counts[str(path.relative_to(ROOT))] = len(text.split())
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https://', 'http://')):
                external_urls.add(target)
            elif not target.startswith('#'):
                resolved = (path.parent / target.split('#')[0]).resolve()
                assert resolved.exists(), (str(path), target)
                local_links += 1

    plan = json.loads((ROOT / 'configs/dynamic_engram_research_plan_v1.json').read_text())
    review = json.loads((HERE / 'review.json').read_text())
    costs = json.loads((HERE / 'analytical_costs.json').read_text())
    assert plan['status'] == 'proposal_not_executable_not_training_admitted'
    stages = {stage['id']: stage for stage in plan['stages']}
    assert len(stages) == 10
    visited, active = set(), set()

    def visit(name):
        assert name in stages
        assert name not in active, 'Stage dependency cycle'
        if name in visited:
            return
        active.add(name)
        for dependency in stages[name]['depends_on']:
            visit(dependency)
        assert all(gate in plan['gates'] for gate in stages[name]['gates'])
        active.remove(name)
        visited.add(name)

    for name in stages:
        visit(name)
    assert len(plan['gates']) == 11
    assert not {'DE4', 'DE8'}.intersection(stages['DE9']['depends_on'])
    assert plan['components']['fast_memory']['shared_reducer_is_uttt'] is False

    specification = importlib.util.spec_from_file_location('research_capacity', HERE / 'estimate_capacity.py')
    capacity = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(capacity)
    assert capacity.result == costs
    for label, profile in [('compact', 'compact_pilot'), ('expanded', 'expanded_candidate')]:
        assert plan['cost_and_hardware'][label + '_live_state_payload_bytes'] == costs['profiles'][profile]['live_state_payload_bytes']
    assert plan['components']['generated_adapter']['float32_state_bytes'] == costs['profiles']['compact_pilot']['generated_rank8_two_port_state_fp32_bytes']
    assert plan['components']['static_pattern']['embedding_parameters'] == costs['profiles']['compact_pilot']['static_table_parameters']

    test = ET.parse(HERE / 'focused_tests.xml').getroot()[0]
    assert int(test.attrib['tests']) == 99
    assert all(int(test.attrib[field]) == 0 for field in ['failures', 'errors', 'skipped'])
    assert review['checks_performed']['focused']['source_commit'] == review['head']
    assert review['checks_performed']['focused']['passed'] == int(test.attrib['tests'])
    assert plan['repository_evidence']['reconciled_head'] == review['head']
    source_checks = 0
    for path, expected in review['reviewed_committed_source_sha256'].items():
        content = subprocess.check_output(['git', 'show', review['head'] + ':' + path], cwd=ROOT)
        assert digest(content) == expected, path
        source_checks += 1

    papers = [record for record in review['public_evidence']['sources'] if record['kind'] == 'primary_paper_full_text']
    assert len(papers) == 14
    assert all(record['byte_hash_verified_during_packaging'] and not record['executed'] for record in review['public_evidence']['sources'])
    assert len({record['url'] for record in papers}) == len(papers)
    canonical = 'reports/memory_reliability_v5_scorer_fix_eval02/verdict_final.json'
    assert digest((ROOT / canonical).read_bytes()) == review['working_tree_observed_file_sha256'][canonical]
    protected = ROOT / 'src/replica_lab/runtime_pilot.py'
    if review['protected_runtime_draft_sha256']:
        assert digest(protected.read_bytes()) == review['protected_runtime_draft_sha256']

    changed_since_snapshot = [path for path, expected in review['working_tree_observed_file_sha256'].items()
                              if digest((ROOT / path).read_bytes()) != expected]
    whitespace = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    # A concurrent task's whitespace must be reported rather than silently edited.
    own_files = [*documents, ROOT / 'configs/dynamic_engram_research_plan_v1.json',
                 HERE / 'review.json', HERE / 'analytical_costs.json', HERE / 'estimate_capacity.py',
                 HERE / 'validate_report.py', HERE / 'focused_tests.xml', HERE / 'focused_tests_initial.xml']
    result = {
        'format': 'replica.dynamic_engram.report_validation.v1',
        'validated_at_utc': datetime.now(timezone.utc).isoformat(),
        'package_checks': 'pass',
        'markdown_documents': len(documents),
        'word_counts': word_counts,
        'total_report_words': sum(word_counts.values()),
        'local_markdown_targets_checked': local_links,
        'distinct_external_markdown_urls': len(external_urls),
        'balanced_fences_and_authored_whitespace': 'pass',
        'json_syntax_and_ten_stage_dag': 'pass',
        'gate_count': len(plan['gates']),
        'analytical_cost_recomputation': 'pass',
        'focused_tests': {'passed': 99, 'seconds': float(test.attrib['time']), 'native_execution': False},
        'reviewed_commit': review['head'],
        'reviewed_committed_source_hashes_verified': source_checks,
        'primary_papers_recorded': len(papers),
        'canonical_verdict_and_protected_draft_unchanged': True,
        'working_tree_files_changed_since_packaging': changed_since_snapshot,
        'concurrent_work_note': 'New working-tree edits are outside the focused-test/source snapshot; do not infer their qualification from this report.',
        'git_diff_check': {'exit_code': whitespace.returncode, 'output': whitespace.stdout + whitespace.stderr},
        'artifact_sha256': {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in own_files},
        'limits': ['No proposed architecture implemented.', 'No Spark or vision execution.', 'No inference latency or device-fit measurement.', 'Paper results not reproduced.']
    }
    (HERE / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['package_checks', 'total_report_words', 'local_markdown_targets_checked', 'primary_papers_recorded', 'analytical_cost_recomputation', 'working_tree_files_changed_since_packaging', 'git_diff_check']}, indent=2))


if __name__ == '__main__':
    main()
