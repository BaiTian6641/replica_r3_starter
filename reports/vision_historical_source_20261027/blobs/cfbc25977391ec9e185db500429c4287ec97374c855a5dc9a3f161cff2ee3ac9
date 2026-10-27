"""Native Llama residual adapter; no pretrained performance claims.

One batch owns the sidecar context until backward completes. Call
``release_context`` after backward (or after an evaluation forward). This keeps
checkpoint recomputation on the same immutable support/prompt state. This
``replica_lab.generation`` supplies greedy uncached decoding with one fixed
prompt/source context. This adapter does not implement generation caching,
parallel requests, distributed sharding, online fast-state updates or a VLM port.
"""
from __future__ import annotations

import hashlib
import json
import re
import weakref
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .neural import MemoryReader, PromptWorkspace, RecordEncoder


REGIMES = ("baseline", "full", "modules_only", "full_augmented")


def validate_model_lock(model_path: str | Path, lock: dict[str, Any]) -> dict[str, Any]:
    """Validate local metadata without fetching weights or running remote code."""
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError("model_path must be an existing local snapshot directory")
    if lock.get("repo_id") != "openbmb/MiniCPM5-2B":
        raise ValueError("This trainer supports the pinned MiniCPM5-2B text model only")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(lock.get("revision", ""))):
        raise ValueError("A full 40-character model revision SHA is required")
    raw = (root / "config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != lock.get("config_sha256"):
        raise ValueError("Local config does not match the model lock")
    cfg = json.loads(raw)
    expected = {"model_type": "llama", "hidden_size": 2048,
                "num_hidden_layers": 42, "vocab_size": 130560,
                "tie_word_embeddings": False}
    for name, value in expected.items():
        if cfg.get(name) != value:
            raise ValueError(f"Unexpected MiniCPM5 config field {name}: {cfg.get(name)!r}")
    if cfg.get("architectures") != ["LlamaForCausalLM"] or cfg.get("auto_map"):
        raise ValueError("Only the known native Llama implementation is supported")
    if not list(root.glob("*.safetensors")):
        raise ValueError("A local safetensors checkpoint is required")
    sealed = lock.get("snapshot_files_sha256")
    if not isinstance(sealed, dict) or not sealed:
        raise ValueError("Seal local snapshot bytes with scripts/seal_snapshot.py before training")
    relevant = {p.relative_to(root).as_posix() for p in root.rglob("*")
                if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".jinja"}}
    if set(sealed) != relevant or "config.json" not in sealed:
        raise ValueError("Snapshot file inventory differs from its local byte seal")
    for relative, expected_digest in sealed.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Snapshot manifest contains an invalid path")
        h = hashlib.sha256()
        with (root / path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != expected_digest:
            raise ValueError(f"Snapshot byte mismatch: {relative}")
    return cfg


class _AugmentedLayer(nn.Module):
    def __init__(self, original: nn.Module, owner: "LlamaMemoryAdapter") -> None:
        super().__init__()
        self.original = original
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("original"), name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        result = self.original(*args, **kwargs)
        owner = self._owner_ref()
        if owner is None or owner._context is None:
            raise RuntimeError("Missing owned sidecar context; use adapter.forward")
        hidden = result[0] if isinstance(result, tuple) else result
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("Unsupported native decoder output type")
        memory, memory_mask, workspace = owner._context
        delta_memory = owner.memory_reader(hidden, memory, memory_mask)
        delta_workspace = owner.workspace.read(hidden, workspace)
        updated = hidden + delta_memory + delta_workspace
        return (updated, *result[1:]) if isinstance(result, tuple) else updated


class LlamaMemoryAdapter(nn.Module):
    """Minimal model-internal residual memory/workspace experiment.

    Memory and workspace are built from separate, past-only input embeddings.
    This is a small testable writer, not a reproduction of Titans or Engram.
    The baseline also receives all source records in its textual prompt.
    """

    def __init__(self, base: nn.Module, regime: str = "modules_only", *,
                 memory_size: int = 64, workspace_size: int = 64,
                 slots: int = 4, workspace_steps: int = 2,
                 insertion_layer: int = -1, heads: int = 4) -> None:
        super().__init__()
        if regime not in REGIMES:
            raise ValueError(f"Unknown regime {regime!r}")
        if getattr(base.config, "model_type", None) != "llama":
            raise ValueError("LlamaMemoryAdapter supports only native Llama text models")
        self.base = base
        self.regime = regime
        self.augmented = regime in {"modules_only", "full_augmented"}
        self._original_parameters = dict(base.named_parameters())
        self._context = None
        self.settings = {"memory_size": memory_size, "workspace_size": workspace_size,
                         "slots": slots, "workspace_steps": workspace_steps,
                         "insertion_layer": insertion_layer, "heads": heads}
        base.config.use_cache = False
        if self.augmented:
            width = base.config.hidden_size
            self.support_writer = RecordEncoder(width, memory_size, slots=slots, heads=heads)
            self.memory_reader = MemoryReader(width, memory_size, heads=heads)
            self.workspace = PromptWorkspace(width, workspace_size, slots=slots,
                                             steps=workspace_steps, heads=heads)
            count = len(base.model.layers)
            index = insertion_layer if insertion_layer >= 0 else count + insertion_layer
            if not 0 <= index < count:
                raise ValueError("insertion_layer is outside the native decoder")
            self.insertion_layer = index
            base.model.layers[index] = _AugmentedLayer(base.model.layers[index], self)
        for parameter in self._original_parameters.values():
            parameter.requires_grad_(regime in {"full", "full_augmented"})
        self.assert_scope()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Some tensor/device conversions replace Parameter objects. Refresh the
        # ownership map before constructing an optimizer after such a move.
        prefix = (f"model.layers.{self.insertion_layer}.original."
                  if self.augmented else None)
        self._original_parameters = {
            (name.replace(prefix, f"model.layers.{self.insertion_layer}.", 1)
             if prefix and name.startswith(prefix) else name): parameter
            for name, parameter in self.base.named_parameters()
        }
        return result

    def new_named_parameters(self):
        if self.augmented:
            for prefix in ("support_writer", "memory_reader", "workspace"):
                for name, parameter in getattr(self, prefix).named_parameters():
                    yield f"{prefix}.{name}", parameter

    def parameter_audit(self) -> dict[str, Any]:
        def summarize(items):
            pairs = list(items)
            return {"total": sum(p.numel() for _, p in pairs),
                    "trainable": sum(p.numel() for _, p in pairs if p.requires_grad),
                    "tensor_count": len(pairs),
                    "trainable_dtypes": sorted({str(p.dtype) for _, p in pairs if p.requires_grad})}
        return {"regime": self.regime,
                "original": summarize(self._original_parameters.items()),
                "new": summarize(self.new_named_parameters())}

    def assert_scope(self) -> None:
        expected_original = self.regime in {"full", "full_augmented"}
        if any(p.requires_grad != expected_original for p in self._original_parameters.values()):
            raise RuntimeError("Original parameter freeze scope does not match named regime")
        if any(not p.requires_grad for _, p in self.new_named_parameters()):
            raise RuntimeError("A declared new trainable module was frozen")

    def optimizer_groups(self, base_lr: float, new_lr: float):
        self.assert_scope()
        groups = []
        for name, values, lr in (
            ("original", self._original_parameters.values(), base_lr),
            ("new", (p for _, p in self.new_named_parameters()), new_lr),
        ):
            parameters = [p for p in values if p.requires_grad]
            if parameters:
                groups.append({"params": parameters, "lr": lr, "group_name": name})
        declared = [id(p) for group in groups for p in group["params"]]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(declared) != len(set(declared)) or set(declared) != expected:
            raise RuntimeError("Optimizer parameter ownership is incomplete or duplicated")
        return groups

    def gradient_audit(self) -> dict[str, Any]:
        result = {}
        for group, pairs in (("original", self._original_parameters.items()),
                             ("new", self.new_named_parameters())):
            pairs = [(name, p) for name, p in pairs if p.requires_grad]
            result[group] = {
                "eligible_tensors": len(pairs),
                "gradient_tensors": sum(p.grad is not None for _, p in pairs),
                "nonzero_gradient_tensors": sum(bool(torch.count_nonzero(p.grad).item())
                                                for _, p in pairs if p.grad is not None),
                "missing_gradient_names": [name for name, p in pairs if p.grad is None],
            }
        return result

    def prepare_context(self, *, prompt_ids, prompt_mask=None,
                        memory_ids=None, memory_mask=None) -> None:
        """Build one owned context from already-known prompt/source inputs.

        Training keeps it until backward completes; fixed-context generation
        keeps it through the final emitted token. This method does not decide
        source eligibility. Callers must exclude future targets and release
        ownership before constructing another context or saving the model.
        """
        if self._context is not None:
            raise RuntimeError("Previous batch is still in flight; release context after backward")
        if not self.augmented:
            return
        if prompt_ids is None or memory_ids is None:
            raise ValueError("Separate past-only prompt_ids and memory_ids are required")
        if prompt_ids.ndim != 2 or memory_ids.ndim != 2:
            raise ValueError("Context IDs must have shape [batch, sequence]")
        if not prompt_ids.shape[1] or prompt_ids.shape[0] != memory_ids.shape[0]:
            raise ValueError("Context requires a nonempty prompt and matching batch sizes")
        embed = self.base.get_input_embeddings()
        # Preserve writer and candidate-embedding gradients in training.
        slots, valid = self.support_writer(embed(memory_ids), memory_mask)
        workspace = self.workspace.build(embed(prompt_ids), prompt_mask)
        self._context = (slots, valid, workspace)

    def forward(self, input_ids, attention_mask=None, labels=None, *,
                prompt_ids=None, prompt_mask=None,
                memory_ids=None, memory_mask=None):
        if self._context is not None:
            raise RuntimeError("Previous batch is still in flight; release context after backward")
        if self.augmented:
            if prompt_ids is None or memory_ids is None:
                raise ValueError("Separate past-only prompt_ids and memory_ids are required")
            if prompt_ids.ndim != 2 or memory_ids.ndim != 2:
                raise ValueError("Context IDs must have shape [batch, sequence]")
            if prompt_ids.shape[0] != input_ids.shape[0] or memory_ids.shape[0] != input_ids.shape[0]:
                raise ValueError("Context and query batch sizes must agree")
            length = prompt_ids.shape[1]
            if not 0 < length <= input_ids.shape[1]:
                raise ValueError("Prompt must be a nonempty prefix of the query sequence")
            valid_prompt = (prompt_mask.bool() if prompt_mask is not None
                            else torch.ones_like(prompt_ids, dtype=torch.bool))
            if not torch.equal(prompt_ids[valid_prompt], input_ids[:, :length][valid_prompt]):
                raise ValueError("prompt_ids must match the past-only input prefix")
            if labels is not None and torch.any(labels[:, :length][valid_prompt] != -100):
                raise ValueError("Prompt workspace may not include supervised target positions")
            self.prepare_context(prompt_ids=prompt_ids, prompt_mask=prompt_mask,
                                 memory_ids=memory_ids, memory_mask=memory_mask)
        try:
            return self.base(input_ids=input_ids, attention_mask=attention_mask,
                             labels=labels, use_cache=False, return_dict=True)
        except BaseException:
            self.release_context()
            raise

    def release_context(self) -> None:
        """Call only after backward has finished, or after an eval-only forward."""
        self._context = None

    def save_pretrained(self, directory: str | Path, *, manifest=None) -> None:
        if self._context is not None:
            raise RuntimeError("Finish the current batch before saving")
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        wrapped = None
        if self.augmented:
            wrapped = self.base.model.layers[self.insertion_layer]
            self.base.model.layers[self.insertion_layer] = wrapped.original
        try:
            # Write native key names so AutoModelForCausalLM can reload the base.
            self.base.save_pretrained(directory / "base", safe_serialization=True)
        finally:
            if wrapped is not None:
                self.base.model.layers[self.insertion_layer] = wrapped
        from .checkpoint_storage import deduplicate_native_weights
        checkpoint_storage = deduplicate_native_weights(directory / "base")
        if self.augmented:
            weights = {}
            for prefix in ("support_writer", "memory_reader", "workspace"):
                weights.update({f"{prefix}.{key}": value.detach().cpu().contiguous()
                                for key, value in getattr(self, prefix).state_dict().items()})
            save_file(weights, str(directory / "sidecars.safetensors"))
        record = {"format_version": 1, "regime": self.regime,
                  "adapter_settings": self.settings, "parameters": self.parameter_audit(),
                  "generation_cache_supported": False,
                  "checkpoint_storage": checkpoint_storage,
                  "training_scope": "text_only", "run": manifest or {}}
        (directory / "adapter_manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    def load_sidecars(self, path: str | Path) -> None:
        if not self.augmented:
            raise ValueError("This regime does not have sidecars")
        from safetensors.torch import load_file
        weights = load_file(str(path))
        expected = set()
        for prefix in ("support_writer", "memory_reader", "workspace"):
            local = {key[len(prefix) + 1:]: value for key, value in weights.items()
                     if key.startswith(prefix + ".")}
            getattr(self, prefix).load_state_dict(local, strict=True)
            expected.update(prefix + "." + key for key in local)
        if set(weights) != expected:
            raise ValueError("Unexpected sidecar keys")
