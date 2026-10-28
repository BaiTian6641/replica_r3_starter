"""CPU-only DC-R1 v2 runner orchestration tests.

The injected backend is a fixture with one-token text mappings.  These tests
exercise authorization, staged evidence, publication and the real scorer; they
do not establish native execution or learned readout quality.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, read_json, read_jsonl, seal_record, sha256_file, sha256_json,
)
import dc_r1_native_value_v2_20261027 as protocol  # noqa: E402
from replica_lab import dc_r1_native_runner as runner  # noqa: E402

CONFIG = ROOT / "configs" / "dc_r1_native_value_protocol_v2.json"
PREPARED = ROOT / "data" / "dc_r1_native_value_v2_20261027"


def _authorization(tmp_path: Path, **overrides) -> Path:
    manifest = read_json(PREPARED / "manifest.json")
    body = {
        "format": runner.AUTHORIZATION_FORMAT,
        "protocol_id": runner.PROTOCOL_ID,
        "config_sha256": sha256_file(CONFIG),
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "authorized_by": "cpu-orchestration-fixture",
        "native_execution_allowed": True,
        "single_attempt": True,
        "restart_allowance": 1,
    }
    body.update(overrides)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(seal_record(body), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _labels() -> dict[str, dict]:
    rows = []
    for name in ("sealed_labels_dev.jsonl", "sealed_labels_final.jsonl"):
        rows.extend(read_jsonl(PREPARED / name))
    return {row["case_id"]: row for row in rows}


class FakeBackend:
    """Explicit CPU fixture; each distinct output is one reversible token."""

    def __init__(self, labels: dict[str, dict], *, fail_after: int | None = None,
                 vocab: tuple[dict[str, int], dict[int, str]] | None = None):
        self.labels = labels
        self.codec = read_json(PREPARED / "codec.json")
        self.reader_state = sha256_json({"fixture": "dc-r1-v2"})
        self.text_to_token, self.token_to_text = vocab or ({}, {})
        self.calls: list[tuple[str, str, int]] = []
        self.startup_calls = 0
        self.closed = False
        self.fail_after = fail_after

    def startup_work(self) -> dict[str, int]:
        self.startup_calls += 1
        return {"model_loads": 1, "tokenizer_loads": 1}

    def close(self) -> None:
        self.closed = True

    def assert_backbone_unchanged(self) -> dict:
        return {"backbone_unchanged": True}

    def _token(self, text: str) -> int:
        if text not in self.text_to_token:
            token = 10000 + len(self.text_to_token)
            self.text_to_token[text] = token
            self.token_to_text[token] = text
        return self.text_to_token[text]

    @property
    def decode_fn(self):
        return lambda ids: "".join(self.token_to_text.get(int(token), "")
                                   for token in ids)

    def estimate_evaluation_work(self, case: dict) -> dict:
        return {
            "prompt_ids": [0],
            "work": {
                "generation_requests": 1,
                "generation_prefill_forwards": 1,
                "generation_decode_forwards": 7,
                "autoregressive_forwards": 8,
                "forward_calls": 8,
                "generated_tokens": 8,
                "prefill_tokens": 1,
                "processed_positions": 36,
                "layer_readout_streams": 1,
            },
        }

    def evaluate(self, case: dict, *, layer: int, prepared: dict) -> dict:
        self.calls.append((case["condition"], case["case_id"], layer))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("injected interruption")
        label = self.labels[case["case_id"]]
        condition = case["condition"]
        if condition in ("exact_context", "selected_literal"):
            text = label["target"]
        elif condition == "compressed_record":
            text = protocol.decode_compressed_literal(
                self.codec, label["compressed_code"])
        elif condition == "wrong_value":
            text = case["evidence"]["literal"]
        else:
            text = "UNKNOWN"
        token = self._token(text)
        return {
            "layer": layer,
            "generated_ids": [token],
            "generated_text": text,
            "empty_generation": False,
            "stopped": True,
            "stop_reason": "eos",
            "generated_tokens": 1,
            "prefill_tokens": 1,
            "decode_forwards": 0,
            "autoregressive_forwards": 1,
            "forward_calls": 1,
            "processed_positions": 1,
            "evaluation_owner": "cpu_fixture",
            "reader_state_sha256": self.reader_state,
        }


def _run_dev(tmp_path: Path, backend: FakeBackend | None = None):
    output = tmp_path / "native01"
    backend = backend or FakeBackend(_labels())
    result = runner.run_dev(
        config_path=CONFIG, prepared=PREPARED,
        authorization_path=_authorization(tmp_path), output=output,
        backend=backend)
    return output, backend, result


def test_dev_runner_uses_real_scorer_and_marks_cpu_fixture_inadmissible(tmp_path):
    output, backend, result = _run_dev(tmp_path)
    report = read_json(output / "dev" / "score_report.json")
    admission = read_json(output / "dev" / "admission.json")
    journal = read_json(output / "dev" / "journal.json")
    assert result["admitted"] is False
    assert admission["evidence_class"] == "cpu_fixture"
    assert report["development_evidence_admissible"] is True
    assert admission["native_execution_admissible"] is False
    assert report["physical_work"]["budget_check"] == "passed"
    assert report["physical_work"]["totals"]["generation_requests"] == 180
    assert result["counters"]["forward_calls"] == 180
    assert result["counters"]["generation_decode_forwards"] == 0
    assert journal["stage_complete"] is True
    assert journal["publication_stage_done"] is True
    assert len(backend.calls) == 180
    assert backend.closed is True
    rows = read_jsonl(output / "dev" / "predictions_all.jsonl")
    assert len(rows) == 60
    assert all(row["source_kv_loaded"] is False for row in rows)
    assert all(row["prompt_source_spans"] == 0 for row in rows)
    assert all(len(row["layer_outputs"]) == 3 for row in rows)


def test_interrupted_stage_resumes_completed_cases_without_repeating_them(tmp_path):
    output = tmp_path / "native01"
    authorization = _authorization(tmp_path)
    first = FakeBackend(_labels(), fail_after=4)
    with pytest.raises(RuntimeError, match="injected interruption"):
        runner.run_dev(
            config_path=CONFIG, prepared=PREPARED,
            authorization_path=authorization, output=output, backend=first)
    journal = read_json(output / "dev" / "journal.json")
    assert journal["stage_complete"] is False
    assert journal["status"] == "failed"
    assert len(journal["predictions"]["exact_context"]) == 1
    resumed = FakeBackend(
        _labels(), vocab=(first.text_to_token, first.token_to_text))
    # This injected CPU backend represents an already-loaded process on resume;
    # no native reload is being claimed or given a free budget exemption.
    resumed.startup_work = lambda: {"model_loads": 0, "tokenizer_loads": 0}
    result = runner.run_dev(
        config_path=CONFIG, prepared=PREPARED,
        authorization_path=authorization, output=output, resume=True,
        backend=resumed)
    assert result["admitted"] is False
    assert len(resumed.calls) == 177
    assert not any(condition == "exact_context" and case_id == first.calls[0][1]
                   for condition, case_id, _ in resumed.calls)
    assert result["counters"]["generation_requests"] == 181
    report = read_json(output / "dev" / "score_report.json")
    assert report["physical_work"]["totals"]["generation_requests"] == 180


def test_development_paths_do_not_hash_sealed_final_labels(tmp_path, monkeypatch):
    output, backend, _ = _run_dev(tmp_path)
    import dc_protocol_common_20261027 as common
    original_sha256 = common.sha256_file

    def guarded_sha256(path):
        if Path(path).name == "sealed_labels_final.jsonl":
            raise AssertionError("sealed-final label bytes opened during development")
        return original_sha256(path)

    monkeypatch.setattr(common, "sha256_file", guarded_sha256)
    monkeypatch.setattr(protocol, "sha256_file", guarded_sha256)
    monkeypatch.setattr(runner, "sha256_file", guarded_sha256)
    report = protocol.score_dev(
        output / "dev" / "predictions_all.jsonl", prepared=PREPARED,
        config_path=CONFIG, decode_fn=backend.decode_fn,
        native_execution=True)
    assert report["development_evidence_admissible"] is True
    context, _ = runner.verify_authorization(
        config_path=CONFIG, prepared=PREPARED,
        authorization_path=tmp_path / "authorization.json",
        output=tmp_path / "other_output")
    assert context.manifest_sha256 == read_json(PREPARED / "manifest.json")["manifest_sha256"]


def test_final_recomputes_dev_boundary_before_generating(tmp_path):
    output, dev_backend, _ = _run_dev(tmp_path)
    final_backend = FakeBackend(
        _labels(), vocab=(dev_backend.text_to_token, dev_backend.token_to_text))
    result = runner.run_sealed_final(
        config_path=CONFIG, prepared=PREPARED,
        authorization_path=_authorization(tmp_path), output=output,
        backend=final_backend)
    report = read_json(output / "sealed_final" / "score_report.json")
    admission = read_json(output / "sealed_final" / "admission.json")
    assert result["admitted"] is False
    assert report["final_labels_opened"] is True
    assert admission["evidence_class"] == "cpu_fixture"
    assert len(final_backend.calls) == 60
    assert len(dev_backend.calls) == 180


def test_final_tampered_dev_report_fails_before_backend_or_final_labels(
        tmp_path, monkeypatch):
    output, _, _ = _run_dev(tmp_path)
    report_path = output / "dev" / "score_report.json"
    report = read_json(report_path)
    report["status"] = "tampered"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    backend = FakeBackend(_labels())
    with pytest.raises(ProtocolError):
        runner.run_sealed_final(
            config_path=CONFIG, prepared=PREPARED,
            authorization_path=_authorization(tmp_path), output=output,
            backend=backend)
    assert backend.startup_calls == 0
    assert backend.calls == []
    assert not (output / "sealed_final" / "score_report.json").exists()

    def forbidden_factory(*args, **kwargs):
        raise AssertionError("backend constructed before development admission")

    monkeypatch.setattr(runner, "build_native_backend", forbidden_factory)
    with pytest.raises(ProtocolError):
        runner.run_sealed_final(
            config_path=CONFIG, prepared=PREPARED,
            authorization_path=_authorization(tmp_path), output=output)


def test_production_factory_seam_is_used_when_no_backend_is_injected(
        tmp_path, monkeypatch):
    backend = FakeBackend(_labels())
    calls = []

    def factory(context, stage):
        calls.append(stage)
        return backend

    monkeypatch.setattr(runner, "build_native_backend", factory)
    output = tmp_path / "native01"
    result = runner.run_dev(
        config_path=CONFIG, prepared=PREPARED,
        authorization_path=_authorization(tmp_path), output=output)
    assert calls == ["dev"]
    assert result["admitted"] is True
    admission = read_json(output / "dev" / "admission.json")
    assert admission["evidence_class"] == "native_mps"
    assert admission["native_execution_admissible"] is True


def test_authorization_is_checked_before_any_backend_work(tmp_path):
    backend = FakeBackend(_labels())
    with pytest.raises(ProtocolError, match="authorization"):
        runner.run_dev(
            config_path=CONFIG, prepared=PREPARED,
            authorization_path=tmp_path / "missing.json",
            output=tmp_path / "native01", backend=backend)
    assert backend.startup_calls == 0
    assert backend.calls == []


def test_actual_work_over_reservation_is_refused(tmp_path):
    class OverrunBackend(FakeBackend):
        def evaluate(self, case, *, layer, prepared):
            result = super().evaluate(case, layer=layer, prepared=prepared)
            result.update({
                "generated_ids": [10000, 10001, 10002, 10003, 10004,
                                  10005, 10006, 10007, 10008],
                "generated_tokens": 9,
                "decode_forwards": 8,
                "autoregressive_forwards": 9,
                "forward_calls": 9,
                "processed_positions": 45,
            })
            return result

    with pytest.raises(ProtocolError, match="exceeds reservation"):
        _run_dev(tmp_path, OverrunBackend(_labels()))


def test_cli_refuses_invalid_authorization_without_loading_model(tmp_path):
    script = ROOT / "scripts" / "run_dc_r1_native_value_v2_20261027.py"
    authorization = tmp_path / "authorization.json"
    authorization.write_text("{}\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script), "--stage", "dev", "--device", "mps",
         "--output", str(tmp_path / "native01"),
         "--authorization", str(authorization)],
        cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "refused"
    assert payload["check"] == "authorization"
    assert not (tmp_path / "native01").exists()


def test_launch_preflight_emits_guarded_external_command(tmp_path):
    script = ROOT / "scripts" / "preflight_dc_r1_native_value_v2_20261027.py"
    receipt = tmp_path / "launch_review.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--launch", "--receipt", str(receipt),
         "--authorization", str(_authorization(tmp_path))],
        cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "launch_admitted"
    assert report["native_execution"] is False
    assert report["executor"] == "scripts/run_dc_r1_native_value_v2_20261027.py"
    assert "run_capacity_preflight.py" in report["next_command"]
    assert "--stage dev" in report["next_command"]
    assert read_json(receipt)["status"] == "launch_admitted"
    assert not (ROOT / "runs/dc_r1_native_value_v2_20261027/native01").exists()


def test_native_backend_real_tiny_llama_forward_decode_and_work(tmp_path):
    from types import SimpleNamespace
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from replica_lab.dc_r1_native_backend import NativeR1Backend

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "checkpoints/minicpm5_source"), local_files_only=True,
        trust_remote_code=False)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=131072, hidden_size=16, intermediate_size=32,
        num_hidden_layers=42, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=512, attention_dropout=0.0))

    class TinyAdapter:
        def __init__(self, base):
            self.base = base

        def parameter_stamp(self, original_only=False):
            return tuple((name, parameter.data_ptr(), parameter._version)
                         for name, parameter in self.base.named_parameters())

    context = SimpleNamespace(
        config=read_json(CONFIG), root=ROOT, output=tmp_path)
    case = read_jsonl(PREPARED / "dev_conditions_exact_context.jsonl")[0]
    backend = NativeR1Backend(
        context, "dev", adapter=TinyAdapter(model), tokenizer=tokenizer)
    try:
        plan = backend.estimate_evaluation_work(case)
        result = backend.evaluate(case, layer=13, prepared=plan)
        count = len(result["generated_ids"])
        prompt_count = len(plan["prompt_ids"])
        assert 0 < prompt_count <= 128
        assert result["forward_calls"] == count
        assert result["generated_tokens"] == count
        assert result["processed_positions"] == (
            count * prompt_count + count * (count - 1) // 2)
        assert not set(result["generated_ids"]) & {1, 130073}
        assert backend.decode_fn(result["generated_ids"]).strip() == result["generated_text"]
        assert backend.assert_backbone_unchanged() == {"backbone_unchanged": True}
    finally:
        backend.close()
