"""Native DC-R1 per-layer readout backend; no training or checkpoint writes."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dc_protocol_common_20261027 import require


class NativeR1Backend:
    """Generate independent no-cache readouts at declared MiniCPM layers."""

    def __init__(self, context: Any, stage: str, *, adapter: Any = None,
                 tokenizer: Any = None) -> None:
        from replica_lab.architecture.loading import (
            exclusive_mps_worker, load_local_mps)
        from replica_lab.profiling import RunProfiler

        self.context = context
        self.stage = stage
        self.layers = tuple(int(layer) for layer in context.config["prepared"]["layer_group"])
        generation = context.config["budget"]["schedule"]
        self.max_new = int(generation["max_generated_tokens_per_request"])
        self.max_prompt = int(generation["prefill_token_ceiling_per_request"])
        policy = context.config["tokenizer_binding"]["decode_policy"]
        self.stops = tuple(int(token) for token in policy["stop_tokens"])
        self.eos_id = int(policy["target_eos_token"])
        self._worker = None
        self.profiler = None
        if adapter is None:
            self._worker = exclusive_mps_worker(Path(context.root))
            try:
                self._worker.__enter__()
                profile_dir = Path(context.output) / stage
                profile_dir.mkdir(parents=True, exist_ok=True)
                self.profiler = RunProfiler("mps", profile_dir).start()
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
        self.reader_state_sha256 = hashlib.sha256(
            repr(self.base_identity).encode()).hexdigest()

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

    def evaluate(self, case: Mapping[str, Any], *, layer: int,
                 prepared: Mapping[str, Any] | None = None) -> dict:
        require(layer in self.layers, "layer", f"undeclared readout layer {layer}")
        prompt_ids = (list(prepared["prompt_ids"]) if prepared is not None
                      else self._render(str(case["prompt"])))
        raw: list[int] = []
        content: list[int] = []
        stopped = False
        import torch
        with torch.no_grad():
            for _ in range(self.max_new):
                ids = prompt_ids + raw
                tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
                mask = torch.ones_like(tensor, dtype=torch.bool)
                output = self.adapter.base.model(
                    input_ids=tensor, attention_mask=mask, use_cache=False,
                    output_hidden_states=True)
                hidden = output.hidden_states[layer + 1][:, -1:, :]
                hidden = self.adapter.base.model.norm(hidden)
                logits = self.adapter.base.lm_head(hidden)
                require(bool(torch.isfinite(logits).all()), "generation",
                        "nonfinite layer logits")
                token = int(logits[0, -1].argmax().item())
                raw.append(token)
                if token in self.stops:
                    stopped = True
                    break
                content.append(token)
        policy = self.context.config["tokenizer_binding"]["decode_policy"]
        text = self.tokenizer.decode(
            content, skip_special_tokens=policy["skip_special_tokens"],
            clean_up_tokenization_spaces=policy["clean_up_tokenization_spaces"]).strip()
        require(not any(item in self.stops for item in content), "generation",
                "terminal stop id reached generated content")
        # The v2 wire schema excludes terminal stops and derives work from the
        # posted IDs; keep these counters exactly aligned with that schema.
        posted_count = len(content)
        posted_positions = (posted_count * len(prompt_ids)
                            + posted_count * (posted_count - 1) // 2)
        return {
            "layer": layer,
            "generated_ids": content,
            "generated_text": text,
            "empty_generation": text == "",
            "stopped": stopped,
            "stop_reason": "eos" if stopped else "max_new_tokens",
            "generated_tokens": posted_count,
            "prefill_tokens": len(prompt_ids),
            "prefill_forwards": 1,
            "decode_forwards": max(posted_count - 1, 0),
            "autoregressive_forwards": posted_count,
            "forward_calls": posted_count,
            "processed_positions": posted_positions,
            "evaluation_owner": "native_mps",
            "reader_state_sha256": self.reader_state_sha256,
        }

    def assert_backbone_unchanged(self) -> dict:
        current = self.adapter.parameter_stamp(original_only=True)
        require(current == self.base_identity, "backbone",
                "native backbone identity changed")
        return {"backbone_unchanged": True}
