"""DC-R1 v3 native per-layer readout with raw EOS-complete accounting."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dc_protocol_common_20261027 import require, sha256_json
from replica_lab.model_identity import durable_model_identity
from replica_lab.native_readout import native_layer_logits


class NativeR1Backend:
    """Generate no-cache readouts and retain raw/content/stop token identities."""

    def __init__(self, context: Any, stage: str, *, adapter: Any = None,
                 tokenizer: Any = None) -> None:
        from replica_lab.architecture.loading import (
            exclusive_mps_worker, load_local_mps)
        from replica_lab.profiling import RunProfiler

        self.context = context
        self.stage = stage
        self.layers = tuple(int(layer) for layer in context.config["prepared"]["layer_group"])
        schedule = context.config["schedule"]
        self.max_new = int(schedule["max_generated_tokens_per_request"])
        self.max_prompt = int(schedule["prefill_token_ceiling_per_request"])
        policy = context.config["tokenizer_binding"]["decode_policy"]
        self.stops = frozenset(int(token) for token in policy["stop_tokens"])
        self._worker = None
        self.profiler = None
        if adapter is None:
            self._worker = exclusive_mps_worker(Path(context.root))
            try:
                self._worker.__enter__()
                # The capacity guard watches this single attempt-root stream for
                # both stages; do not hide telemetry in a stage-only directory.
                Path(context.output).mkdir(parents=True, exist_ok=True)
                self.profiler = RunProfiler("mps", Path(context.output)).start()
                model_path = Path(context.root) / context.config["baseline_model"]["model_path"]
                lock_path = Path(context.root) / context.config["baseline_model"]["lock_path"]
                self.adapter, self.tokenizer, self.lock = load_local_mps(
                    model_path, lock_path, profiler=self.profiler)
            except BaseException:
                try:
                    self.close()
                finally:
                    self._worker = None
                    self.profiler = None
                raise
        else:
            self.adapter = adapter
            self.tokenizer = tokenizer
            self.lock = None
        self.device = next(self.adapter.base.parameters()).device
        self.base_identity = self.adapter.parameter_stamp(original_only=True)
        source_identity = getattr(
            getattr(self.adapter, "precision", None), "artifact_sha256", None)
        if source_identity is None and self.lock is not None:
            source_identity = sha256_json({
                "revision": self.lock["revision"],
                "config_sha256": self.lock["config_sha256"],
                "snapshot_files_sha256": self.lock["snapshot_files_sha256"],
            })
        self.reader_state_sha256 = durable_model_identity(
            self.adapter.base,
            source_identity=source_identity,
            config={
                "model_type": getattr(self.adapter.base.config, "model_type", None),
                "hidden_size": getattr(self.adapter.base.config, "hidden_size", None),
                "layers": len(self.adapter.base.model.layers),
            })
        self.model_identity = hashlib.sha256(
            repr((self.adapter.base.config.model_type,
                  getattr(self.adapter.base.config, "hidden_size", None),
                  len(self.adapter.base.model.layers),
                  self.reader_state_sha256)).encode()).hexdigest()

    def startup_work(self) -> dict[str, int]:
        return {"model_loads": 1, "tokenizer_loads": 1}

    def close(self) -> None:
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
        if self._worker is not None:
            self._worker.__exit__(None, None, None)
            self._worker = None

    @property
    def decode_fn(self):
        policy = self.context.config["tokenizer_binding"]["decode_policy"]
        return lambda ids: self.tokenizer.decode(
            list(ids), skip_special_tokens=policy["skip_special_tokens"],
            clean_up_tokenization_spaces=policy["clean_up_tokenization_spaces"])

    def _render(self, text: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        require(isinstance(ids, list) and ids, "tokenizer", "empty prompt tokenization")
        require(len(ids) <= self.max_prompt, "generation",
                f"prompt has {len(ids)} tokens, above {self.max_prompt}")
        return [int(item) for item in ids]

    def estimate_evaluation_work(self, case: Mapping[str, Any]) -> dict:
        prompt_ids = self._render(str(case["prompt"]))
        count = len(prompt_ids)
        return {
            "prompt_ids": prompt_ids,
            "work": {
                "generation_requests": 1,
                "generation_prefill_forwards": 1,
                "generation_decode_forwards": self.max_new - 1,
                "autoregressive_forwards": self.max_new,
                "forward_calls": self.max_new,
                "generated_tokens": self.max_new,
                "prefill_tokens": count,
                "processed_positions": self.max_new * count
                + self.max_new * (self.max_new - 1) // 2,
                "layer_readout_streams": 1,
            },
        }

    def next_logits(self, prompt_ids: list[int], layer: int):
        """Return last-position logits using the ordinary native final path."""
        import torch
        require(layer in self.layers, "layer", f"undeclared readout layer {layer}")
        tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(tensor, dtype=torch.bool)
        logits = native_layer_logits(
            self.adapter.base, tensor, layer, attention_mask=mask)
        require(bool(torch.isfinite(logits).all()), "generation",
                "nonfinite layer logits")
        return logits

    def evaluate(self, case: Mapping[str, Any], *, layer: int,
                 prepared: Mapping[str, Any] | None = None,
                 on_step: Callable[[Mapping[str, Any]], None] | None = None,
                 on_error: Callable[[Mapping[str, Any]], None] | None = None) -> dict:
        require(layer in self.layers, "layer", f"undeclared readout layer {layer}")
        prompt_ids = (list(prepared["prompt_ids"]) if prepared is not None
                      else self._render(str(case["prompt"])))
        raw: list[int] = []
        content: list[int] = []
        terminal: list[int] = []
        stopped = False
        import torch
        with torch.no_grad():
            for _ in range(self.max_new):
                ids = prompt_ids + raw
                try:
                    logits = self.next_logits(ids, layer)
                except BaseException as error:
                    if on_error is not None:
                        on_error({
                            "layer": layer, "raw_index": len(raw),
                            "forward_calls": 1, "generated_tokens": 0,
                            "processed_positions": len(ids),
                            "error": str(error),
                        })
                    raise
                token = int(logits[0, -1].argmax().item())
                raw.append(token)
                is_terminal = token in self.stops
                if is_terminal:
                    terminal.append(token)
                    stopped = True
                else:
                    content.append(token)
                step = {
                    "layer": layer,
                    "raw_index": len(raw) - 1,
                    "token_id": token,
                    "terminal": is_terminal,
                    "forward_calls": 1,
                    "generated_tokens": 1,
                    "processed_positions": len(ids),
                }
                if on_step is not None:
                    on_step(step)
                if is_terminal:
                    break
        policy = self.context.config["tokenizer_binding"]["decode_policy"]
        text = self.tokenizer.decode(
            content, skip_special_tokens=policy["skip_special_tokens"],
            clean_up_tokenization_spaces=policy["clean_up_tokenization_spaces"]).strip()
        require(not any(item in self.stops for item in content), "generation",
                "terminal stop id reached scored content")
        return {
            "layer": layer,
            "raw_generated_ids": raw,
            "terminal_stop_ids": terminal,
            "generated_ids": content,
            "generated_text": text,
            "empty_generation": text == "",
            "stopped": stopped,
            "stop_reason": "eos" if stopped else "max_new_tokens",
            "generated_tokens": len(raw),
            "prefill_tokens": len(prompt_ids),
            "prefill_forwards": 1,
            "decode_forwards": max(len(raw) - 1, 0),
            "autoregressive_forwards": len(raw),
            "forward_calls": len(raw),
            "processed_positions": sum(len(prompt_ids) + index for index in range(len(raw))),
            "evaluation_owner": "native_mps",
            "reader_state_sha256": self.reader_state_sha256,
            "model_identity": self.model_identity,
            "normalization": "native_model_norm",
        }

    def assert_backbone_unchanged(self) -> dict:
        current = self.adapter.parameter_stamp(original_only=True)
        require(current == self.base_identity, "backbone",
                "native backbone identity changed")
        return {"backbone_unchanged": True}
