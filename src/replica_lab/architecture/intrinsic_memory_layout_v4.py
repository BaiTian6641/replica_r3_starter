"""Configurable intrinsic readers; a separate engineering candidate from v3.

Every enabled site reads one incoming snapshot. This module has no retrieval,
writer or optimizer step in forward. It does not change the sealed v3 experiment.
Values retain their autograd graph until the caller's declared detach boundary.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from .spine import SpinePort
from ..optimizers import NativeFusedAdamW

FORMAT = "replica.intrinsic_memory.reader_checkpoint.v4"
PLACEMENT = "block_input_query_to_post_block_residual"
PRE_MLP_PLACEMENT = "block_input_query_to_pre_mlp_residual"


@dataclass(frozen=True)
class IntrinsicMemoryLayout:
    hidden_size: int
    decoder_layers: int = 36
    sites: tuple[int, ...] = (3, 7)
    width: int = 256
    slots: int = 8
    placement: str = PLACEMENT

    def __post_init__(self):
        if (type(self.hidden_size) is not int or self.hidden_size < 1
                or type(self.decoder_layers) is not int or self.decoder_layers < 1
                or type(self.width) is not int or self.width != 256
                or type(self.slots) is not int or self.slots != 8
                or self.placement not in (PLACEMENT, PRE_MLP_PLACEMENT)
                or not isinstance(self.sites, tuple) or not self.sites
                or any(type(s) is not int or not 0 <= s < self.decoder_layers for s in self.sites)
                or tuple(sorted(set(self.sites))) != self.sites):
            raise ValueError("invalid intrinsic v4 layer layout")

    @property
    def workspace_width(self):
        return self.width

    def identity(self):
        result = asdict(self)
        result["sites"] = list(self.sites)
        return result

    def reader_parameter_count(self):
        d, m = self.hidden_size, self.width
        # Q/out, K/V, query norm, slot/output norms, scalar gate.
        return len(self.sites) * (2*d*m + 2*m*m + 2*d + 4*m + 1)


def spark_layout(arm, *, hidden_size=2560, placement=PLACEMENT):
    sites = {"two": (3, 7), "full_attention_nine": tuple(range(3, 36, 4)),
             "all": tuple(range(36))}
    if arm not in sites:
        raise ValueError("unknown Spark memory density arm")
    if placement not in (PLACEMENT, PRE_MLP_PLACEMENT):
        raise ValueError("unknown intrinsic-memory placement")
    return IntrinsicMemoryLayout(hidden_size, sites=sites[arm], placement=placement)


def _sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint file missing or substituted")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConfigurableIntrinsicMemoryReader(nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.spec = layout
        self.ports = nn.ModuleDict({str(site): SpinePort(layout, "memory", site)
                                   for site in layout.sites})
        for port in self.ports.values():
            port.branch = "intrinsic_memory"
        self.float()
        self._optimizer_owner = None

    @classmethod
    def install(cls, adapter, layout):
        geometry = adapter.config.geometry
        if (layout.hidden_size != geometry.hidden_size
                or layout.decoder_layers != geometry.num_hidden_layers):
            raise ValueError("intrinsic layout differs from actual decoder geometry")
        if hasattr(adapter.extension, "intrinsic_memory"):
            raise ValueError("intrinsic memory already has an owner; explicit migration required")
        reader = cls(layout).to(next(adapter.base.parameters()).device)
        # Validate/construct first: a rejected layout must not remove old ports.
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        legacy = getattr(adapter.extension, "spine", None)
        if legacy is not None and "memory" in legacy.ports:
            legacy.ports.pop("memory")
            if hasattr(legacy, "memory_projection"):
                del legacy.memory_projection
        adapter.extension.intrinsic_memory = reader
        return reader

    def make_optimizer(self, *, lr=1e-3):
        if self._optimizer_owner is not None:
            raise ValueError("intrinsic reader already has an optimizer owner")
        parameters = list(self.parameters())
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("aliased reader parameters")
        self._optimizer_owner = NativeFusedAdamW([{
            "params": parameters, "lr": lr, "weight_decay": 0.0,
            "betas": (0.9, 0.999), "eps": 1e-8, "algorithm": "adamw_fused",
            "group_name": "intrinsic_reader", "scope": "intrinsic_phi_v4",
            "param_names": list(dict(self.named_parameters())),
        }])
        return self._optimizer_owner

    def snapshot(self, values, availability, *, validate, active_sites=None):
        return LayerMemorySnapshot(self, values, availability, validate=validate,
                                   active_sites=active_sites)

    def save_checkpoint(self, path, *, source_identity, producer_identity,
                        representation_identity, codec_identity):
        """Reader-only extension plus optional ordinary optimizer, never the base."""
        if not all(isinstance(x, str) and x for x in
                   (source_identity, producer_identity, representation_identity, codec_identity)):
            raise ValueError("explicit source/producer/representation/codec identities required")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        tensors = {name: value.detach().cpu().contiguous()
                   for name, value in self.state_dict().items()}
        save_file(tensors, str(path / "reader.safetensors"))
        optimizer_hash = None
        if self._optimizer_owner is not None:
            torch.save(self._optimizer_owner.state_dict(), path / "optimizer.pt")
            optimizer_hash = _sha(path / "optimizer.pt")
        manifest = {"format": FORMAT, "layout": self.spec.identity(),
                    "source_identity": source_identity, "producer_identity": producer_identity,
                    "representation_identity": representation_identity,
                    "codec_identity": codec_identity, "reader_sha256": _sha(path / "reader.safetensors"),
                    "optimizer_sha256": optimizer_hash,
                    "parameter_names": list(dict(self.named_parameters())),
                    "parameters": sum(p.numel() for p in self.parameters()),
                    "owner": "intrinsic_memory", "state": "one_shared_read_only_snapshot",
                    "native_qualification": False}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        return manifest

    def load_checkpoint(self, path, *, source_identity, producer_identity,
                        representation_identity, codec_identity):
        path = Path(path)
        _sha(path / "manifest.json")
        manifest = json.loads((path / "manifest.json").read_text())
        expected = {"format": FORMAT, "layout": self.spec.identity(),
                    "source_identity": source_identity, "producer_identity": producer_identity,
                    "representation_identity": representation_identity,
                    "codec_identity": codec_identity, "owner": "intrinsic_memory",
                    "state": "one_shared_read_only_snapshot",
                    "parameter_names": list(dict(self.named_parameters())),
                    "parameters": self.spec.reader_parameter_count()}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("reader checkpoint identity/layout differs; no implicit migration")
        if _sha(path / "reader.safetensors") != manifest["reader_sha256"]:
            raise ValueError("reader payload digest differs")
        tensors = load_file(str(path / "reader.safetensors"))
        current = self.state_dict()
        if set(tensors) != set(current) or any(
                value.shape != current[name].shape or value.dtype != torch.float32
                or not torch.isfinite(value).all() for name, value in tensors.items()):
            raise ValueError("reader tensor inventory differs or is nonfinite")
        optimizer_state = None
        if manifest["optimizer_sha256"] is not None:
            if (self._optimizer_owner is None
                    or _sha(path / "optimizer.pt") != manifest["optimizer_sha256"]):
                raise ValueError("optimizer continuation identity differs")
            optimizer_state = torch.load(path / "optimizer.pt", weights_only=True, map_location="cpu")
            # NativeFusedAdamW validates all named ownership and moment tensors.
            # Validate hyperparameters before its final state installation too.
            groups = optimizer_state.get("param_groups", [])
            if (len(groups) != 1 or len(groups[0]["params"]) != len(list(self.parameters()))
                    or not math.isfinite(groups[0]["lr"]) or groups[0]["lr"] < 0):
                raise ValueError("optimizer parameter ownership differs")
        elif self._optimizer_owner is not None and self._optimizer_owner.state:
            raise ValueError("reader-only checkpoint cannot retain unrelated optimizer moments")
        if optimizer_state is not None:
            previous_state = self._optimizer_owner.state
            previous_groups = self._optimizer_owner.param_groups
            try:
                self._optimizer_owner.load_state_dict(optimizer_state)
            except (ValueError, KeyError, TypeError) as error:
                self._optimizer_owner.__setstate__({"state": previous_state,
                                                   "param_groups": previous_groups})
                raise ValueError("optimizer continuation is incompatible") from error
        self.load_state_dict(tensors, strict=True)
        return manifest


class LayerMemorySnapshot:
    """Shared values, independent per-layer access; validation never writes."""
    def __init__(self, reader, values, availability, *, validate, active_sites=None):
        sites = reader.spec.sites if active_sites is None else tuple(active_sites)
        if (not callable(validate) or not sites or any(type(site) is not int for site in sites)
                or tuple(sorted(set(sites))) != sites
                or not set(sites) <= set(reader.spec.sites)):
            raise ValueError("invalid active layer mask or validation callback")
        if (values.ndim != 4 or values.dtype != torch.float32
                or values.shape[-1] != reader.spec.width
                or not 0 < values.shape[2] <= reader.spec.slots
                or availability.shape != values.shape[:3] or availability.dtype != torch.bool
                or not torch.isfinite(values).all()):
            raise ValueError("invalid shared memory snapshot")
        self.reader, self.spec = reader, replace(reader.spec, sites=sites)
        self.values, self.availability = values.clone(), availability.clone()
        self.validate = validate

    def delta(self, site, hidden):
        if site not in self.spec.sites:
            raise ValueError("disabled intrinsic site")
        self.validate()
        return self.reader.ports[str(site)](hidden, self.values, self.availability)
