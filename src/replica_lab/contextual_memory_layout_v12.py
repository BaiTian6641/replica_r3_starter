"""Frozen-source contextual producer and configurable intrinsic readers, v12.

This is an independently versioned training candidate. The sealed v3–v11
runners and checkpoints remain unchanged. One canonical source read feeds one
live snapshot shared by all active layers. No projection cache or write exists
inside forward. Durable admission/read/checkpoint detach; training re-encodes
pinned source text through the same contextual producer without crossing
serialization or hashing.

Ownership: ``producer`` (frozen-source compressor), ``order`` and ``reader``
(the two-site or denser ports) each have exactly one NativeFusedAdamW group and
one owner. The frozen original backbone has zero optimizer ownership and is
never exported or serialized here.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from .architecture.intrinsic_memory_layout_v4 import ConfigurableIntrinsicMemoryReader
from .architecture.spine import SpinePort
from .contextual_feature_identity_v12 import (
    CODEC, PRODUCER, REPRESENTATION, SLOTS, WIDTH, ContextualMemoryError,
    FeatureIdentityPath, SourceFeatureValueEncoder, tensor_digest,
)
from .contextual_source_features_v12 import FrozenSourceFeatures
from .optimizers import NativeFusedAdamW

FORMAT = "replica.intrinsic_memory.contextual_training_checkpoint.v12"
IMPLEMENTATION = "replica.intrinsic_memory.contextual_training.v12"


def _sha(path):
    if path.is_symlink() or not path.is_file():
        raise ContextualMemoryError("checkpoint file missing or substituted")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContextualMemoryLayoutV12(FeatureIdentityPath):
    """Reuse the strict contextual v2 record contract, with a new reader owner.

    The caller selects the layout and owns the fitting schedule. No fit, retrieval,
    write or optimizer update is hidden in a decoder hook. This wrapper is not a
    native experiment launcher or a useful-recall qualification.
    """

    def __init__(self, adapter, layout, *, source_identity, feature_provider,
                 lr=1e-3, initialization_seed=17):
        if not isinstance(source_identity, str) or not source_identity:
            raise ContextualMemoryError("source execution identity is required")
        if not math.isfinite(lr) or lr <= 0:
            raise ContextualMemoryError("positive finite learning rate required")
        if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**31:
            raise ContextualMemoryError("bounded integer initialization seed required")
        if not isinstance(feature_provider, FrozenSourceFeatures):
            raise ContextualMemoryError("explicit frozen source-feature provider required")
        if getattr(feature_provider, "adapter", None) is not adapter:
            raise ContextualMemoryError("feature provider is bound to a different adapter")
        if feature_provider.spec.hidden_size != layout.hidden_size:
            raise ContextualMemoryError("feature-provider width differs from the reader layout")
        self.initialization_seed = initialization_seed
        self.adapter, self.spec, self.source_identity = adapter, layout, source_identity
        self.feature_provider = feature_provider
        self.reader = ConfigurableIntrinsicMemoryReader.install(adapter, layout)
        device = next(adapter.base.parameters()).device
        # CPU initialization with separate deterministic streams makes shared
        # producer/order and common sites identical across 2/9/36 layouts.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed)
            self.encoder = SourceFeatureValueEncoder(feature_provider)
            self.order = nn.Embedding(layout.slots, WIDTH)
            nn.init.normal_(self.order.weight, std=0.02)
            self.encoder.to(device)
            self.order.to(device)
            for site in layout.sites:
                torch.random.default_generator.manual_seed(initialization_seed + 1000 + site)
                port = SpinePort(layout, "memory", site)
                port.branch = "intrinsic_memory"
                self.reader.ports[str(site)] = port.to(device)
        groups = []
        for name, module in self.modules().items():
            parameters = list(module.parameters())
            if not parameters:
                raise ContextualMemoryError("declared module has no trainable parameters")
            groups.append({"params": parameters, "lr": lr,
                "weight_decay": 0.0, "betas": (0.9, 0.999), "eps": 1e-8,
                "algorithm": "adamw_fused", "group_name": name,
                "scope": "intrinsic_contextual_phi_v12",
                "param_names": [name + "." + key for key, _ in module.named_parameters()]})
        self.optimizer = NativeFusedAdamW(groups)
        # Reserve the reader against a second reader-only optimizer owner.
        self.reader._optimizer_owner = self.optimizer
        self.assert_optimizer_ownership()
        self.cost = {key: 0 for key in (
            "producer_bytes", "producer_calls", "producer_wall_ns", "value_slots_read",
            "retrieval_wall_ns", "read_file_bytes", "read_file_calls", "revalidations",
            "fallback_bytes", "fallback_tokens", "fallback_calls", "retrieval_calls",
            "admission_calls", "correction_calls", "snapshot_copies", "snapshot_bytes",
            "source_revalidations")}

    # ------------------------------------------------------------------
    # Ownership

    def modules(self):
        return {"producer": self.encoder, "order": self.order, "reader": self.reader}

    def assert_optimizer_ownership(self):
        modules = self.modules()
        intended, names = [], []
        for group_name, module in modules.items():
            for key, parameter in module.named_parameters():
                intended.append(parameter)
                names.append(group_name + "." + key)
        if not intended:
            raise ContextualMemoryError("no intended trainable parameters")
        owned = [p for group in self.optimizer.param_groups for p in group["params"]]
        flags = [p.requires_grad for p in intended]
        if any(flags) and not all(flags):
            raise ContextualMemoryError("partial phi freezing is not a declared training phase")
        if len(owned) != len({id(p) for p in owned}):
            raise ContextualMemoryError("an intended trainable has more than one optimizer owner")
        if {id(p) for p in owned} != {id(p) for p in intended}:
            raise ContextualMemoryError("optimizer ownership differs from intended trainables")
        base_ids = {id(p) for p in self.adapter.base.parameters()}
        if base_ids & {id(p) for p in owned}:
            raise ContextualMemoryError("an original parameter has optimizer ownership")
        if any(p.requires_grad for p in self.adapter.base.parameters()):
            raise ContextualMemoryError("an original parameter is trainable")
        if self.reader._optimizer_owner is not self.optimizer:
            raise ContextualMemoryError("intrinsic reader has a second optimizer owner")
        if ({id(p) for p in self.adapter.extension.parameters() if p.requires_grad}
                != {id(p) for p in self.reader.parameters() if p.requires_grad}):
            raise ContextualMemoryError("extension trainables differ from the reader owner")
        declared = {}
        for group in self.optimizer.param_groups:
            module = modules.get(group["group_name"])
            if module is None:
                raise ContextualMemoryError("optimizer group has no declared module")
            expected_names = [group["group_name"] + "." + key
                              for key, _ in module.named_parameters()]
            if list(group["param_names"]) != expected_names:
                raise ContextualMemoryError("named optimizer ownership differs")
            if len(group["params"]) != len(expected_names):
                raise ContextualMemoryError("optimizer group tensor count differs")
            declared[group["group_name"]] = tuple(group["param_names"])
        flattened = [name for group in declared.values() for name in group]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(names):
            raise ContextualMemoryError("named optimizer ownership is incomplete or duplicated")
        return declared

    # ------------------------------------------------------------------
    # Controlled cost wrappers

    def admit(self, *args, **kwargs):
        self.cost["admission_calls"] += 1
        return super().admit(*args, **kwargs)

    def correct(self, *args, **kwargs):
        self.cost["correction_calls"] += 1
        return super().correct(*args, **kwargs)

    def read(self, *args, **kwargs):
        self.cost["retrieval_calls"] += 1
        return super().read(*args, **kwargs)

    def clear_source_caches(self):
        """Source-free/correction boundary: never retain cache-held source values."""
        self.feature_provider.clear()

    def invalidate_source(self, record_id):
        self.feature_provider.evict_record(record_id)

    # ------------------------------------------------------------------
    # Authority validation

    def validate(self, batch, scope):
        if getattr(batch, "store", None) is not None and len(batch.pins) != len(batch.spans):
            raise ContextualMemoryError("complete durable authority pins required")
        super().validate(batch, scope)
        for index, span in enumerate(batch.spans):
            if batch.store is None:
                continue
            row = batch.store.connection.execute(
                "SELECT source_lineage FROM records WHERE record_id=?",
                (batch.pins[index]["record_id"],)).fetchone()
            self.cost["source_revalidations"] += 1
            if row is None:
                self.feature_provider.evict_record(span.record_id)
                raise ContextualMemoryError("canonical training source authority missing")
            current = None
            for item in json.loads(row["source_lineage"]):
                try:
                    candidate = json.loads(item)
                except (ValueError, TypeError):
                    continue
                if isinstance(candidate, dict) and candidate.get("record_id") == span.record_id:
                    current = candidate
            if current is None:
                self.feature_provider.evict_record(span.record_id)
                raise ContextualMemoryError("canonical training source provenance missing")
            if current != self.provenance(span, current.get("producer_revision")):
                self.feature_provider.evict_record(span.record_id)
                raise ContextualMemoryError("canonical training source provenance differs")
        return True

    def training_from_stored(self, batch, *, scope):
        """Re-encode the admitted canonical spans, never use durable slots as a graph.

        A previous producer release can supply pinned canonical text for fitting.
        Its stored values remain forbidden for inference. Source revision, time,
        scope and generation must still be current at every injection boundary.
        """
        if batch.training or batch.slots.requires_grad:
            raise ContextualMemoryError("training source must be a detached durable batch")
        if batch.store is None or len(batch.pins) != len(batch.spans):
            raise ContextualMemoryError("training source requires complete durable authority pins")
        # Pin payload corruption is rejected even when the old producer release
        # differs from the current trainable producer after an update.
        if any(tensor_digest(batch.slots[i*SLOTS:(i+1)*SLOTS]) != pin["slots_sha256"]
               for i, pin in enumerate(batch.pins)):
            raise ContextualMemoryError("stored training source payload differs")
        self.validate(replace(batch, training=True), scope)
        fresh = self.training_values(batch.spans, scope=scope)
        fresh.pins, fresh.store = batch.pins, batch.store
        self.validate(fresh, scope)
        return fresh

    # ------------------------------------------------------------------
    # One shared live snapshot

    @staticmethod
    def _normalize_active_sites(active_sites):
        if active_sites is None:
            return None
        if isinstance(active_sites, (str, bytes)):
            raise ContextualMemoryError("active sites must be an iterable of integers")
        try:
            sites = tuple(active_sites)
        except TypeError as error:
            raise ContextualMemoryError("active sites must be an iterable of integers") from error
        if any(type(site) is not int for site in sites):
            raise ContextualMemoryError("active sites must be integers")
        return sites

    def snapshot(self, ids, batch, *, scope, active_sites=None, padding_value=0.0):
        active_sites = self._normalize_active_sites(active_sites)
        if active_sites == ():
            raise ContextualMemoryError("a snapshot requires at least one active site")
        self.validate(batch, scope)
        slots = batch.slots + self.order(torch.arange(len(batch.slots), device=ids.device))
        count = len(slots)
        if count > self.spec.slots:
            raise ContextualMemoryError("value slot budget exceeded")
        pad = torch.full((self.spec.slots - count, WIDTH), padding_value,
                         device=slots.device, dtype=slots.dtype)
        values = torch.cat((slots, pad))[None, None].expand(ids.shape[0], ids.shape[1], -1, -1)
        available = (torch.arange(self.spec.slots, device=ids.device) < count)
        available = available[None, None].expand(values.shape[:3])
        context = self.reader.snapshot(values, available,
            validate=lambda: self.validate(batch, scope), active_sites=active_sites)
        self.cost["snapshot_copies"] += 1
        self.cost["snapshot_bytes"] += context.values.numel() * context.values.element_size()
        self.cost["value_slots_read"] += count
        return context

    def forward(self, ids, mask, batch, *, scope, enabled=True, active_sites=None):
        active_sites = self._normalize_active_sites(active_sites)
        if not enabled or batch is None or active_sites == ():
            return self.adapter.forward_text(ids, mask, use_projector=False,
                                              use_ports=False, use_spine=False)
        context = self.snapshot(ids, batch, scope=scope, active_sites=active_sites)
        return self.adapter.forward_text(ids, mask, use_projector=False, use_ports=False,
                                          use_spine=False, intrinsic_memory=context)

    # ------------------------------------------------------------------
    # Training phase and complete extension checkpoint

    def freeze_readout(self):
        """Freeze fitting, retaining moments for a separately declared continuation."""
        for module in self.modules().values():
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None

    def resume_training(self):
        for module in self.modules().values():
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.assert_optimizer_ownership()

    def _tensors(self):
        return {prefix + "." + name: value
                for prefix, module in self.modules().items()
                for name, value in module.state_dict().items()}

    def producer_release(self):
        """Digest of the compressor (producer) state plus the feature contract."""
        return self.encoder.release_from_state(self.encoder.state_dict())

    def representation_release_from_tensors(self, tensors):
        """Digest of the complete contextual representation (producer/order/reader)."""
        digest = hashlib.sha256(self.feature_provider.contract_digest().encode())
        for prefix, module in self.modules().items():
            for name in module.state_dict():
                key = prefix + "." + name
                if key not in tensors:
                    raise ContextualMemoryError("representation tensor inventory differs")
                digest.update(key.encode())
                digest.update(tensor_digest(tensors[key]).encode())
        return "v12-" + digest.hexdigest()

    def representation_release(self):
        return self.representation_release_from_tensors(self._tensors())

    def save(self, directory):
        self.feature_provider.validate_original()
        self.assert_optimizer_ownership()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        save_file({name: value.detach().cpu().contiguous()
                   for name, value in self._tensors().items()}, str(path / "extension.safetensors"))
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        manifest = {"format": FORMAT, "implementation": IMPLEMENTATION,
            "source_identity": self.source_identity, "base": self.adapter.base_snapshot(),
            "initialization_seed": self.initialization_seed,
            "feature_contract": self.feature_provider.spec.identity(),
            "adapter_contract": self.feature_provider.adapter_contract,
            "tokenizer_contract": self.feature_provider.tokenizer_contract,
            "layout": self.spec.identity(), "producer": PRODUCER,
            "producer_release": self.representation_release(),
            "representation_release": self.representation_release(),
            "producer_state_release": self.producer_release(),
            "representation": REPRESENTATION,
            "codec": CODEC, "owner": "intrinsic_memory", "contains_base_weights": False,
            "external_spine_required": False, "projection_cache": "none",
            "feature_cache_serialized": False, "task_state_serialized": False,
            "original_weights_serialized": False,
            "parameter_names": {k: list(v) for k, v in self.assert_optimizer_ownership().items()},
            "fitting_frozen": not any(p.requires_grad for m in self.modules().values() for p in m.parameters()),
            "weights_sha256": _sha(path / "extension.safetensors"),
            "optimizer_sha256": _sha(path / "optimizer.pt")}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest

    def load(self, directory):
        """Validate the complete extension and optimizer before changing weights."""
        if not all(p.requires_grad for m in self.modules().values() for p in m.parameters()):
            raise ContextualMemoryError("checkpoint load requires a fresh or explicitly resumed trainable owner")
        path = Path(directory)
        _sha(path / "manifest.json")
        manifest = json.loads((path / "manifest.json").read_text())
        expected = {"format": FORMAT, "implementation": IMPLEMENTATION,
            "source_identity": self.source_identity, "base": self.adapter.base_snapshot(),
            "initialization_seed": self.initialization_seed,
            "feature_contract": self.feature_provider.spec.identity(),
            "adapter_contract": self.feature_provider.adapter_contract,
            "tokenizer_contract": self.feature_provider.tokenizer_contract,
            "layout": self.spec.identity(), "producer": PRODUCER,
            "representation": REPRESENTATION, "codec": CODEC, "owner": "intrinsic_memory",
            "contains_base_weights": False, "external_spine_required": False,
            "projection_cache": "none", "feature_cache_serialized": False,
            "task_state_serialized": False, "original_weights_serialized": False,
            "parameter_names": {k: list(v) for k, v in self.assert_optimizer_ownership().items()}}
        if any(manifest.get(k) != value for k, value in expected.items()):
            raise ContextualMemoryError("complete v12 checkpoint identity differs; no implicit migration")
        if type(manifest.get("fitting_frozen")) is not bool:
            raise ContextualMemoryError("checkpoint fitting state is missing")
        if (_sha(path / "extension.safetensors") != manifest.get("weights_sha256")
                or _sha(path / "optimizer.pt") != manifest.get("optimizer_sha256")):
            raise ContextualMemoryError("complete v12 checkpoint digest differs")
        tensors = load_file(str(path / "extension.safetensors"))
        current = self._tensors()
        if set(tensors) != set(current) or any(
                value.shape != current[name].shape or value.dtype != torch.float32
                or not torch.isfinite(value).all() for name, value in tensors.items()):
            raise ContextualMemoryError("complete v12 tensor inventory differs or is nonfinite")
        producer_state = {name: tensors["producer." + name] for name in self.encoder.state_dict()}
        if self.encoder.release_from_state(producer_state) != manifest.get("producer_state_release"):
            raise ContextualMemoryError("checkpoint producer state release differs")
        if self.representation_release_from_tensors(tensors) != manifest.get("representation_release"):
            raise ContextualMemoryError("checkpoint representation release differs")
        if manifest.get("producer_release") != manifest.get("representation_release"):
            raise ContextualMemoryError("checkpoint producer/representation release binding differs")
        state = torch.load(path / "optimizer.pt", weights_only=True, map_location="cpu")
        groups = state.get("param_groups", [])
        if len(groups) != len(self.optimizer.param_groups) or any(
                not math.isfinite(group.get("lr", float("nan"))) or group["lr"] <= 0
                for group in groups):
            raise ContextualMemoryError("checkpoint optimizer hyperparameters differ")
        identifiers = [identifier for group in groups for identifier in group.get("params", [])]
        if (any(type(identifier) is not int or identifier < 0 for identifier in identifiers)
                or len(identifiers) != len(set(identifiers))
                or any(len(saved.get("params", [])) != len(current["params"])
                       for saved, current in zip(groups, self.optimizer.param_groups))):
            raise ContextualMemoryError("checkpoint optimizer has aliased or incomplete parameter ownership")
        # NativeFusedAdamW validates named ownership, shape, finite moments and
        # metadata atomically; weights are installed only after that succeeds.
        self.optimizer.load_state_dict(state)
        for prefix, module in self.modules().items():
            module.load_state_dict({name: tensors[prefix + "." + name]
                                    for name in module.state_dict()}, strict=True)
        if manifest["fitting_frozen"]:
            self.freeze_readout()
        else:
            self.resume_training()
        return manifest

    def migrate_legacy(self, *args, **kwargs):
        raise ContextualMemoryError("v12 requires its own complete checkpoint; migration is not defined")
