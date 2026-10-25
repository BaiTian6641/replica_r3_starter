"""Exercise all six control consumers with real tiny decoder generation, CPU only."""
import hashlib
from pathlib import Path

import torch

from test_visual_adapter import SPEC, SCOPE, observed, visual_session
from test_visual_runner_paths import runner
from test_visual_scoring_replay import scorer
from test_vision_qualification import copy_stage, base_protocol, make_v1_admission_fixture
from replica_lab import vision_qualification as vq


class OwnedPatchObservations:
    """Deterministic precomputed patch fixture, not a pretrained image encoder."""
    spec = SPEC

    def observe(self, path, *, source_id, task_id, scope, observed_at, source_epoch):
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).digest()
        return observed(features=torch.tensor(list(digest), dtype=torch.float32).reshape(1, 4, 8) / 255,
                        image_sha256=digest.hex(), source_id=source_id, task_id=task_id,
                        subject_id=scope.subject_id, branch_id=scope.branch_id, purpose=scope.purpose,
                        observed_at=observed_at, source_epoch=source_epoch)


def test_every_control_and_measured_retention_survive_independent_consumption(visual_session, tmp_path):
    model, session = visual_session
    copy_stage(tmp_path, 'v1')
    protocol = base_protocol(tmp_path, 'v1')
    protocol['v1']['eval']['max_new_tokens'] = 2
    dataset = vq.load_verified_dataset(protocol)
    make_v1_admission_fixture(tmp_path, protocol, dataset)
    admission = vq.verify_admission(protocol, dataset, train_splits=['train'], eval_splits=['dev'])
    output = tmp_path / 'actual_tiny_run'
    (output / 'v1').mkdir(parents=True)
    baseline = runner.evaluate_retention(session, session.tokenizer, protocol, admission,
                                         SCOPE, output, baseline=True)
    evaluation = runner.evaluate_v1(model, session, session.tokenizer, protocol,
                                    OwnedPatchObservations(), dataset, admission, SCOPE, output)
    retained = runner.evaluate_retention(session, session.tokenizer, protocol, admission, SCOPE, output)
    report = {'evaluation': {key: value for key, value in evaluation.items() if key != 'rows'},
              'baseline_retention': {key: value for key, value in baseline.items() if key != 'rows'},
              'retention': {key: value for key, value in retained.items() if key != 'rows'}}
    verified = scorer.verify_v1_evaluation(output, report, protocol, dataset, admission,
                                           session.tokenizer)
    assert set(verified['scores']['controls']) == set(vq.CONTROLS)
    assert verified['retention']['tokens_by_case'] == verified['baseline_retention']['tokens_by_case']
    assert len({(row['control'], row['case_id']) for row in evaluation['rows']}) == len(evaluation['rows'])


def test_native_admission_consumes_split_metadata_before_backend_allocation(tmp_path, monkeypatch):
    import pytest
    from types import SimpleNamespace
    copy_stage(tmp_path, 'v1')
    protocol = base_protocol(tmp_path, 'v1')
    dataset = vq.load_verified_dataset(protocol)
    make_v1_admission_fixture(tmp_path, protocol, dataset)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(vq.QualificationError):
        runner.execute_v1(SimpleNamespace(), protocol, '0' * 64, dataset, None, None, None)
