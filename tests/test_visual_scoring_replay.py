"""Scorer-only replays retain immutable native parents and cannot alter gates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from replica_lab.vision_qualification import QualificationError, sha256_file

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'visual_scoring_replay', ROOT / 'scripts/score_vision_qualification.py')
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)


def replay_fixture(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    source = scripts / 'score_vision_qualification.py'
    runner = scripts / 'run_vision_qualification.py'
    source.write_text("print('original scorer')\n")
    runner.write_text("print('native runner')\n")
    module = tmp_path / 'src/replica_lab/vision_qualification.py'
    module.parent.mkdir(parents=True)
    module.write_text("NATIVE_GATE_POLICY = 'unchanged'\\n")
    original = {str(path.relative_to(tmp_path)): sha256_file(path)
                for path in (source, runner, module)}
    protocol = {'stage': 'v0', 'root': str(tmp_path / 'untrusted-data-root'),
                'implementation_sha256': original}
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    run, restart = tmp_path / 'run', tmp_path / 'restart'
    run.mkdir(); restart.mkdir()
    (run / 'run_report.json').write_text('{"status":"completed"}\n')
    (restart / 'restart_report.json').write_text('{"zero_updates":true}\n')
    source.write_text("print('repaired scorer')\n")
    replacement = {name: sha256_file(tmp_path / name) for name in original}
    replay = {
        'format': 'replica.vision_scoring_replay.v1', 'evaluation_only': True,
        'retrospective_to_native': True, 'native_code_commit': 'b' * 40,
        'native_protocol_sha256': sha256_file(protocol_path),
        'native_run_report_sha256': sha256_file(run / 'run_report.json'),
        'native_restart_report_sha256': sha256_file(restart / 'restart_report.json'),
        'implementation_overlay': 'Scorer-only infrastructure repair; native gates unchanged.',
        'implementation_sha256': replacement,
    }
    replay_path = tmp_path / 'replay.json'
    replay_path.write_text(json.dumps(replay))
    args = SimpleNamespace(scoring_replay=replay_path, protocol=protocol_path,
                           run=run, restart=restart)
    return args, protocol, replay


def test_scorer_replay_authenticates_repaired_source_and_both_native_parents(tmp_path):
    args, protocol, replay = replay_fixture(tmp_path)
    result = scorer.verify_scoring_source(args, protocol, root=tmp_path)
    assert result['implementation_sha256'] == replay['implementation_sha256']
    assert result['replay_sha256'] == sha256_file(args.scoring_replay)
    (tmp_path / 'scripts/score_vision_qualification.py').write_text('changed again\n')
    with pytest.raises(QualificationError):
        scorer.verify_scoring_source(args, protocol, root=tmp_path)


@pytest.mark.parametrize('parent', ['protocol', 'run', 'restart'])
def test_scorer_replay_rejects_changed_parent_artifacts(tmp_path, parent):
    args, protocol, _ = replay_fixture(tmp_path)
    path = {'protocol': args.protocol, 'run': args.run / 'run_report.json',
            'restart': args.restart / 'restart_report.json'}[parent]
    path.write_text('{"changed":true}\n')
    with pytest.raises(QualificationError):
        scorer.verify_scoring_source(args, protocol, root=tmp_path)


@pytest.mark.parametrize('mutation', ['runner', 'gates'])
def test_scorer_replay_cannot_replace_native_runner_or_override_gates(tmp_path, mutation):
    args, protocol, replay = replay_fixture(tmp_path)
    if mutation == 'runner':
        runner = tmp_path / 'scripts/run_vision_qualification.py'
        runner.write_text("print('different native behavior')\n")
        replay['implementation_sha256']['scripts/run_vision_qualification.py'] = sha256_file(runner)
    else:
        replay['quality_gates'] = {'accept_all': True}
    args.scoring_replay.write_text(json.dumps(replay))
    with pytest.raises(QualificationError):
        scorer.verify_scoring_source(args, protocol, root=tmp_path)


@pytest.mark.parametrize('field,value', [('evaluation_only', False),
                                         ('retrospective_to_native', False),
                                         ('native_code_commit', 'unversioned')])
def test_scorer_replay_requires_explicit_immutable_audit_scope(tmp_path, field, value):
    args, protocol, replay = replay_fixture(tmp_path)
    replay[field] = value
    args.scoring_replay.write_text(json.dumps(replay))
    with pytest.raises(QualificationError):
        scorer.verify_scoring_source(args, protocol, root=tmp_path)
