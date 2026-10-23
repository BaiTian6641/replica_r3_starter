"""Runtime entry-point contracts without loading or executing a native model."""
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from replica_lab.memory_system.ledger import EligibilityScope, ExactSpan


def script_module():
    path = Path(__file__).parents[1] / "scripts/run_replica_session.py"
    spec = importlib.util.spec_from_file_location("replica_session_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(module, tmp_path, *extra):
    return module.build_parser().parse_args([
        "--output", str(tmp_path / "report"), "--ledger", str(tmp_path / "records.sqlite"),
        "--query", "What is the badge?", "--subject", "person", "--branch", "replica-a",
        "--purpose", "test", "--as-of", "5", *extra,
    ])


def record(**changes):
    return {"record_id": "fact-a", "subject_id": "person", "branch_id": "source",
            "text": "Badge: blå.\nExact spacing stays.", "observed_at": 1, "event_at": 1,
            "purpose": "test", "authenticity": "source_report", **changes}


def test_cli_rejects_cpu_oversize_and_unqualified_random_reranking(tmp_path):
    module = script_module()
    with pytest.raises(SystemExit):
        arguments(module, tmp_path, "--device", "cpu")
    for extra in (("--max-new-tokens", "129"), ("--max-prompt-tokens", "513"),
                  ("--max-candidate-units", "65"), ("--rerank",)):
        with pytest.raises(ValueError):
            module.validate_arguments(arguments(module, tmp_path, *extra))


def test_cli_preserves_prior_outputs_and_sealed_source_paths(tmp_path):
    module = script_module()
    args = arguments(module, tmp_path)
    args.output.mkdir()
    (args.output / "prior.json").write_text("{}")
    with pytest.raises(ValueError, match="fresh"):
        module.validate_arguments(args)
    args = arguments(module, tmp_path / "new")
    args.model_path = tmp_path / "original"
    args.ledger = args.model_path / "ledger.sqlite"
    with pytest.raises(ValueError, match="sealed original"):
        module.validate_arguments(args)


def test_source_admission_requires_explicit_authority_and_retains_exact_bytes(tmp_path):
    module = script_module()
    path = tmp_path / "records.jsonl"
    item = record()
    path.write_text(json.dumps(item, ensure_ascii=False) + "\n")
    args = arguments(module, tmp_path, "--records", str(path))
    with pytest.raises(PermissionError, match="authorize-source"):
        module.read_records(args)
    args.authorize_source_records = True
    result = module.read_records(args)
    assert result[0].text == item["text"]
    assert result[0].authenticity == "source_report"


@pytest.mark.parametrize("changes", [{"subject_id": "someone-else"}, {"purpose": "unrelated"},
                                     {"branch_id": "other-replica"}])
def test_import_cannot_widen_subject_branch_or_purpose(tmp_path, changes):
    module = script_module()
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(record(**changes)) + "\n")
    args = arguments(module, tmp_path, "--records", str(path), "--authorize-source-records")
    with pytest.raises(PermissionError):
        module.read_records(args)


def test_import_is_fully_parsed_before_admission_and_rejects_duplicate_ids(tmp_path):
    module = script_module()
    path = tmp_path / "records.jsonl"
    args = arguments(module, tmp_path, "--records", str(path), "--authorize-source-records")
    path.write_text(json.dumps(record()) + "\n{" + "\n")
    with pytest.raises(ValueError, match="line 2"):
        module.read_records(args)
    path.write_text(json.dumps(record()) + "\n" + json.dumps(record()) + "\n")
    with pytest.raises(ValueError, match="repeat"):
        module.read_records(args)


@dataclass(frozen=True)
class FakeSpan:
    record_id: str = "fact-a"
    char_start: int = 0
    char_end: int = 12


class FakeRequest:
    def __init__(self, selection, digest):
        self.selection = selection
        self.frame = SimpleNamespace(evidence=SimpleNamespace(digest=digest),
                                     memory=SimpleNamespace(spans=(FakeSpan(),)))
        self.released = False

    def release(self):
        self.released = True
        self.frame = self.selection = None


class FakeSession:
    def __init__(self, changed_pin=False):
        self.adapter = SimpleNamespace(ports=SimpleNamespace(source_tokens=128, max_units=16))
        self.selection = SimpleNamespace(selection_hash="same-selection")
        self.index = SimpleNamespace(retrieve=lambda *args, **kwargs: self.selection)
        self.requests = []
        self.preparations = []
        self.verifications = []
        self.changed_pin = changed_pin

    def prepare(self, **kwargs):
        digest = "changed" if self.changed_pin and self.requests else "pin-a"
        result = FakeRequest(kwargs["selection"], digest)
        self.requests.append(result)
        self.preparations.append(kwargs)
        return result

    def verify(self, request):
        assert not request.released
        self.verifications.append(request)

    def exact_text(self, request, span):
        self.verify(request)
        return ExactSpan("fact-a", "Badge: blue.", 0, 12, "a" * 64, "b" * 64, "c" * 64, 1)


def prepare_exact(module, tmp_path, session, encode=None):
    args = arguments(module, tmp_path, "--exact-text")
    scope = EligibilityScope("person", "replica-a", "test", 5)
    report = {}
    if encode is None:
        encode = lambda *args: torch.tensor([[1, 2]], dtype=torch.long)
    result = module.prepare_request(session, None, args, scope, object(),
                                    SimpleNamespace(mark=lambda *args, **kwargs: None), report, encode)
    return result, report


def test_exact_route_rebuilds_and_rechecks_the_same_source_pin(tmp_path):
    module = script_module()
    session = FakeSession()
    result, report = prepare_exact(module, tmp_path, session)
    assert len(session.requests) == 2
    assert session.requests[0].released and not result.released
    assert session.preparations[0]["selection"] is session.preparations[1]["selection"]
    assert result in session.verifications
    assert report["exact_fetch_pin_rechecked"] is True
    assert report["exact_citations"][0]["text"] == "Badge: blue."
    assert '"text": "Badge: blue."' in report["effective_prompt"]


def test_exact_rebuild_refuses_changed_pin_and_releases_both_requests(tmp_path):
    module = script_module()
    session = FakeSession(changed_pin=True)
    with pytest.raises(ValueError, match="Source selection changed"):
        prepare_exact(module, tmp_path, session)
    assert len(session.requests) == 2
    assert all(request.released for request in session.requests)


def test_exact_prompt_overflow_fails_without_silent_truncation(tmp_path):
    module = script_module()
    session = FakeSession()
    def encode(tokenizer, text, device):
        count = 513 if "Exact authorized source spans" in text else 2
        return torch.ones((1, count), dtype=torch.long)
    with pytest.raises(ValueError, match="rather than truncate"):
        prepare_exact(module, tmp_path, session, encode)
    assert len(session.requests) == 1 and session.requests[0].released


def test_runtime_failure_always_preserves_report_and_stops_profiler(tmp_path, monkeypatch):
    module = script_module()
    import replica_lab.profiling
    import replica_lab.architecture.loading

    class Profiler:
        def __init__(self, *args, **kwargs):
            self.stopped = False
        def start(self):
            return self
        def stop(self):
            self.stopped = True
            return {"stopped": True}

    @contextmanager
    def exclusive(root):
        yield

    def failed(*args):
        raise RuntimeError("native load failed")

    monkeypatch.setattr(replica_lab.profiling, "RunProfiler", Profiler)
    monkeypatch.setattr(replica_lab.architecture.loading, "exclusive_mps_worker", exclusive)
    monkeypatch.setattr(module, "execute", failed)
    for name in module.OFFLINE_ENVIRONMENT:
        monkeypatch.setenv(name, "previous-test-value")
    with pytest.raises(RuntimeError, match="native load failed"):
        module.main(["--output", str(tmp_path / "report"), "--ledger", str(tmp_path / "records.sqlite"),
                     "--query", "badge", "--subject", "person", "--branch", "replica-a",
                     "--purpose", "test", "--as-of", "5"])
    report = json.loads((tmp_path / "report/session_report.json").read_text())
    assert report["status"] == "failed_e1_e2_session"
    assert report["profile"] == {"stopped": True}
    assert report["checkpoints_saved"] == report["downloads"] == 0
    assert report["offline_environment"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
