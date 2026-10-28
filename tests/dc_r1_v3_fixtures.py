"""Deterministic DC-R1 v3 orchestration fixtures; never native evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from dc_protocol_common_20261027 import read_json, read_jsonl, seal_record, sha256_file, sha256_json
from tokenizer_fixtures import tiny_tokenizer
import dc_r1_native_value_v2_20261027 as legacy

CONFIG = ROOT / "configs/dc_r1_native_value_protocol_v3.json"
PREPARED = ROOT / "data/dc_r1_native_value_v2_20261027"
COMMIT = "a" * 40


def labels() -> dict[str, dict]:
    rows = []
    for name in ("sealed_labels_dev.jsonl", "sealed_labels_final.jsonl"):
        rows.extend(read_jsonl(PREPARED / name))
    return {row["case_id"]: row for row in rows}


def _texts() -> list[str]:
    codec = read_json(PREPARED / "codec.json")
    values = {row["target"] for row in labels().values()}
    values.update(("UNKNOWN", "wrong"))
    for row in labels().values():
        values.add(legacy.decode_compressed_literal(codec, row["compressed_code"]))
    for prefix in ("dev", "sealed_final"):
        for condition in legacy.CONDITIONS:
            for row in read_jsonl(PREPARED / f"{prefix}_conditions_{condition}.jsonl"):
                evidence = row.get("evidence") or {}
                if isinstance(evidence.get("literal"), str):
                    values.add(evidence["literal"])
    return sorted(values)


def authorization(tmp_path: Path, output: Path, *, attempt_id="attempt-v3-1") -> Path:
    body = {
        "format": "replica.dc_protocol.authorization.v3",
        "protocol_id": "dc_r1_native_value_readout_v3",
        "config_sha256": sha256_file(CONFIG),
        "prepared_manifest_sha256": read_json(PREPARED / "manifest.json")["manifest_sha256"],
        "protocol_commit": COMMIT,
        "attempt_id": attempt_id,
        "output_root": str(output.resolve()),
        "authorized_by": "cpu-orchestration-fixture",
        "native_execution_allowed": True,
        "single_attempt": True,
        "restart_allowance": 1,
    }
    path = tmp_path / "authorization.json"
    path.write_text(__import__("json").dumps(seal_record(body), indent=2, sort_keys=True) + "\n")
    return path


def commit_verifier(commit, relative):
    if commit != COMMIT:
        return None
    return (ROOT / relative).read_bytes()


class FakeBackend:
    def __init__(self, *, native=False, fail_after=None, immediate_eos=False):
        self.native = native
        self.fail_after = fail_after
        self.immediate_eos = immediate_eos
        self.labels = labels()
        self.codec = read_json(PREPARED / "codec.json")
        self.tokenizer = tiny_tokenizer(phrases=_texts())
        self.model_identity = hashlib.sha256(b"dc-r1-v3-fixture-model").hexdigest()
        self.reader_state_sha256 = hashlib.sha256(b"dc-r1-v3-fixture-reader").hexdigest()
        self.calls = []
        self.closed = False

    def startup_work(self):
        return {"model_loads": 1, "tokenizer_loads": 1}

    def close(self):
        self.closed = True

    def assert_backbone_unchanged(self):
        return {"backbone_unchanged": True}

    @property
    def decode_fn(self):
        return self.tokenizer.decode

    def estimate_evaluation_work(self, case):
        prompt_ids = [3]
        return {
            "prompt_ids": prompt_ids,
            "work": {
                "generation_requests": 1,
                "generation_prefill_forwards": 1,
                "generation_decode_forwards": 7,
                "autoregressive_forwards": 8,
                "forward_calls": 8,
                "generated_tokens": 8,
                "prefill_tokens": len(prompt_ids),
                "processed_positions": 36,
                "layer_readout_streams": 1,
            },
        }

    def evaluate(self, case, *, layer, prepared, on_step=None, on_error=None):
        self.calls.append((case["condition"], case["case_id"], layer))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            error = {
                "layer": layer, "raw_index": 0, "forward_calls": 1,
                "generated_tokens": 0, "processed_positions": 1,
                "error": "injected interruption",
            }
            if on_error is not None:
                on_error(error)
            raise RuntimeError("injected interruption")
        label = self.labels[case["case_id"]]
        condition = case["condition"]
        if condition in ("exact_context", "selected_literal"):
            text = label["target"]
        elif condition == "compressed_record":
            text = legacy.decode_compressed_literal(self.codec, label["compressed_code"])
        elif condition == "wrong_value":
            text = case["evidence"]["literal"]
        else:
            text = "UNKNOWN"
        prompt_ids = list(prepared["prompt_ids"])
        if self.immediate_eos:
            content = []
            raw = [1]
        else:
            content = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            raw = content + [1]
        terminal = [1]
        for index, token in enumerate(raw):
            step = {
                "layer": layer, "raw_index": index, "token_id": token,
                "terminal": index == len(raw) - 1,
                "forward_calls": 1, "generated_tokens": 1,
                "processed_positions": len(prompt_ids) + index,
            }
            if on_step is not None:
                on_step(step)
        text = self.decode_fn(content).strip()
        return {
            "layer": layer,
            "raw_generated_ids": raw,
            "terminal_stop_ids": terminal,
            "generated_ids": content,
            "generated_text": text,
            "empty_generation": text == "",
            "stopped": True,
            "stop_reason": "eos",
            "generated_tokens": len(raw),
            "prefill_tokens": len(prompt_ids),
            "prefill_forwards": 1,
            "decode_forwards": len(raw) - 1,
            "autoregressive_forwards": len(raw),
            "forward_calls": len(raw),
            "processed_positions": sum(len(prompt_ids) + index for index in range(len(raw))),
            "evaluation_owner": "native_mps" if self.native else "cpu_fixture",
            "reader_state_sha256": self.reader_state_sha256,
            "model_identity": self.model_identity,
            "normalization": "native_model_norm",
        }
