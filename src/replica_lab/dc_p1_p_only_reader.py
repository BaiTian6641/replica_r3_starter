"""Read-only P-only evaluation owner for DC-P1 v2.

The reader clones the selected expert and keeps only sealed model/read-path
bindings, tokenizer files and temporal checkpoint metadata. It owns no optimizer
moments, source rows, source KV, teaching ledger, dedup state or coordinator.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from dc_protocol_common_20261027 import require


class POnlyReader:
    def __init__(self, backend: Any) -> None:
        self.adapter = backend.adapter
        self.tokenizer = backend.tokenizer
        self.expert = deepcopy(backend.bank.expert(0))
        self.projection = backend.projection
        self.sites = tuple(backend.sites)
        self.device = backend.device
        self.stops = tuple(backend.stops)
        self.max_new = int(backend.max_new)
        self.max_prompt = int(backend.max_prompt)
        self.eos_id = int(backend.eos_id)
        self._teaching_positions = {
            group: tuple(values)
            for group, values in backend._teaching_positions.items()}
        self.reader_state_sha256 = backend._state_sha({
            name: parameter.detach().clone()
            for name, parameter in self.expert.named_parameters()})
        self.evidence_class = "native_mps"
        self.owner = "p_only_read_only_no_optimizer_or_source_state"

    def _render(self, text: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        require(isinstance(ids, list) and ids, "tokenizer",
                "tokenizer returned no ids")
        return [int(item) for item in ids]

    def _features(self, hidden_states: Sequence[Any], prompt_length: int) -> Any:
        pooled = []
        for site in self.sites:
            index = site + 1
            require(index < len(hidden_states), "sites",
                    f"hidden state {index} is absent for site {site}")
            value = hidden_states[index][:, :prompt_length, :].float()
            pooled.append(value.mean(dim=1) @ self.projection.T)
        import torch
        return torch.stack(pooled, dim=0).mean(dim=0)

    def _logits(self, ids: Sequence[int], prompt_length: int) -> Any:
        import torch
        tensor = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        mask = torch.ones_like(tensor, dtype=torch.bool)
        with torch.no_grad():
            output = self.adapter.base.model(
                input_ids=tensor, attention_mask=mask, use_cache=False,
                output_hidden_states=True)
        features = self._features(output.hidden_states, prompt_length)
        slot = self.expert(features)
        port = self.adapter.plastic_ports[str(self.sites[0])]
        final = output.last_hidden_state + port(
            output.last_hidden_state, slot, enabled=True)
        return self.adapter.base.lm_head(final)

    def estimate_evaluation_work(self, case: Mapping[str, Any]) -> dict:
        prompt_ids = self._render(str(case["prompt"]))
        require(len(prompt_ids) <= self.max_prompt, "generation",
                f"evaluation prompt {len(prompt_ids)} exceeds {self.max_prompt}")
        count = len(prompt_ids)
        return {
            "prompt_ids": prompt_ids,
            "work": {
                "generation_requests": 1,
                "generation_prefill_forwards": 1,
                "generation_decode_forwards": self.max_new - 1,
                "autoregressive_forwards": self.max_new,
                "generated_tokens": self.max_new,
                "generation_processed_positions": (
                    self.max_new * count + self.max_new * (self.max_new - 1) // 2),
            },
        }

    def _resolved_event(self, case: Mapping[str, Any]) -> int | None:
        mode = case.get("resolution_mode")
        if mode == "historical_explicit_event":
            return int(case["historical_event_position"])
        if mode == "stateless":
            return None
        if "event_position" in case:
            return int(case["event_position"])
        require(mode == "current_at_query_epoch", "temporal_binding",
                f"unsupported resolution mode {mode!r}")
        positions = self._teaching_positions.get(case["source_group"], ())
        applicable = [position for position in positions
                      if position <= int(case["query_epoch"])]
        require(bool(applicable), "temporal_binding",
                f"no event is available for {case['source_group']} at query epoch")
        return max(applicable)

    def evaluate(self, case: Mapping[str, Any], *, arm: str,
                 prepared: Mapping[str, Any] | None = None) -> dict:
        prompt_ids = (list(prepared["prompt_ids"]) if prepared is not None
                      else self._render(str(case["prompt"])))
        require(len(prompt_ids) <= self.max_prompt, "generation",
                f"evaluation prompt {len(prompt_ids)} exceeds {self.max_prompt}")
        raw: list[int] = []
        content: list[int] = []
        processed = 0
        stopped = False
        import torch
        with torch.no_grad():
            for _ in range(self.max_new):
                ids = prompt_ids + raw
                logits = self._logits(ids, len(prompt_ids))
                processed += len(ids)
                token = int(logits[0, -1].argmax().item())
                raw.append(token)
                if token in self.stops:
                    stopped = True
                    break
                content.append(token)
        text = self.tokenizer.decode(
            content, skip_special_tokens=True,
            clean_up_tokenization_spaces=False).strip()
        require(not any(item in self.stops for item in content), "generation",
                "a terminal stop id reached the content ids")
        return {
            "generated_ids": content,
            "generated_text": text,
            "stopped": stopped,
            "stop_reason": "eos" if stopped else "max_new_tokens",
            "generated_tokens": len(raw),
            "prefill_tokens": len(prompt_ids),
            "prefill_forwards": 1,
            "decode_forwards": max(0, len(raw) - 1),
            "processed_positions": processed,
            "query_epoch": case.get("query_epoch"),
            "resolution_mode": case.get("resolution_mode"),
            "resolved_event_position": self._resolved_event(case),
            "reader_state_sha256": self.reader_state_sha256,
            "evaluation_owner": self.owner,
            "source_free_audit": {
                "source_context_present": False,
                "source_kv_loaded": False,
                "source_prefill_tokens": 0,
                "source_kv_reused": False,
                "prompt_source_spans": 0,
                "optimizer_moments_present": False,
                "teaching_rows_present": False,
                "coordinator_present": False,
            },
        }
