"""Matched two-group reader control over the reviewed v12 producer.

This is a separately versioned successor to ``intrinsic_memory_matched_reader_v1``
and to ``contextual_memory_v12``.  It does not patch either one.  The v12 path is
constructed with only the two anchor sites, then the existing canonical groups
are wrapped by ``MatchedParameterReader`` and the declared arm sites become
plain dispatch aliases.  No per-site module, state key, parameter copy or
optimizer entry survives construction.

The producer/order, source-feature contract, store validation, placement,
detach boundaries and checkpoint payload remain v12.  The outer checkpoint
identity is distinct; a v12 matched file cannot load into an independent path.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .architecture.intrinsic_memory_layout_v4 import IntrinsicMemoryLayout
from .contextual_memory import ContextualMemoryError
from .contextual_memory_v12 import ContextualMemoryV12
from .intrinsic_memory_matched_reader_v1 import (
    MAPPING, MatchedParameterReader, MatchedReaderLayout,
)

IMPLEMENTATION = "replica.intrinsic_memory.matched_reader.v12"
FORMAT = IMPLEMENTATION + ".checkpoint"
SCOPE = "intrinsic_contextual_phi_matched_reader_v12"
CANONICAL_GROUPS = (0, 1)
INITIALIZATION_SITES = (3, 7)
ARM_SITES = {
    "two": (3, 7),
    "full_attention_nine": (3, 7, 11, 15, 19, 23, 27, 31, 35),
    "all": tuple(range(36)),
}


def matched_v12_layout(arm, *, hidden_size=2560):
    """Declared arm layout with the fixed canonical sharing identity."""
    if arm not in ARM_SITES:
        raise ContextualMemoryError("unknown matched v12 density arm")
    return MatchedReaderLayout(
        hidden_size=hidden_size, decoder_layers=36, sites=ARM_SITES[arm])


def _sha(path):
    if path.is_symlink() or not path.is_file():
        raise ContextualMemoryError("checkpoint file missing or substituted")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_value(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


class MatchedContextualMemoryV12(ContextualMemoryV12):
    """Real v12 producer/store with exactly two canonical reader owners.

    Construction allocates only the two anchor readers.  The full arm's sites
    are dispatch aliases, so increasing 2 -> 9 -> 36 changes access density but
    not unique parameters, reader state tensors or optimizer entries.
    """

    def __init__(self, adapter, layout, *, source_identity, feature_provider,
                 lr=1e-3, initialization_seed=17, objective_identity=None,
                 execution_identity="CPU_engineering_unbound"):
        if isinstance(layout, MatchedReaderLayout):
            spec = layout
        elif isinstance(layout, IntrinsicMemoryLayout):
            spec = MatchedReaderLayout(**vars(layout))
        else:
            raise ContextualMemoryError("declared intrinsic v12 layout required")
        if spec.decoder_layers != 36 or spec.sites not in ARM_SITES.values():
            raise ContextualMemoryError("declared matched v12 2/9/36 layout required")
        if tuple(sorted(set(spec.sites))) != spec.sites:
            raise ContextualMemoryError("ordered unique active sites required")
        anchor_layout = replace(spec, sites=INITIALIZATION_SITES)
        super().__init__(
            adapter, anchor_layout, source_identity=source_identity,
            feature_provider=feature_provider, lr=lr,
            initialization_seed=initialization_seed,
            objective_identity=objective_identity,
            execution_identity=execution_identity)
        old_reader = self.reader
        partner = MatchedParameterReader(
            spec, (old_reader.ports["3"], old_reader.ports["7"]))
        if len(partner.groups) != 2 or len({id(module) for module in partner.groups.values()}) != 2:
            raise ContextualMemoryError("two distinct canonical reader groups required")
        self.reader, self.spec = partner, spec
        self.adapter.extension.intrinsic_memory = partner
        old_reader._optimizer_owner = None
        partner._optimizer_owner = self.optimizer
        reader_group = [group for group in self.optimizer.param_groups
                        if group["group_name"] == "reader"]
        if len(reader_group) != 1:
            raise ContextualMemoryError("one canonical reader optimizer group required")
        group = reader_group[0]
        if [id(parameter) for parameter in group["params"]] != [
                id(parameter) for parameter in partner.parameters()]:
            raise ContextualMemoryError("canonical group parameter order differs")
        group["scope"] = SCOPE
        group["param_names"] = ["reader." + name for name, _ in partner.named_parameters()]
        self.assert_optimizer_ownership()
        self._release_cache = None

    def _sharing_identity(self):
        return {
            "format": FORMAT,
            "implementation": IMPLEMENTATION,
            "layout": self.spec.identity(),
            "reader_sharing": MAPPING,
            "canonical_groups": list(CANONICAL_GROUPS),
            "initialization_sites": list(INITIALIZATION_SITES),
            "unique_reader_groups": 2,
            "unique_reader_parameters": self.spec.reader_parameter_count(),
            "reader_parameter_tensors": 22,
            "site_aliases": len(self.spec.sites),
            "site_to_group": {str(site): (site // 4) % 2 for site in self.spec.sites},
            "optimizer_scope": SCOPE,
            "optimizer_group_names": [group["group_name"]
                                      for group in self.optimizer.param_groups],
            "projection_cache": "none",
            "native_qualification": False,
        }

    def save(self, directory):
        path = Path(directory)
        path.mkdir(exist_ok=False, parents=True)
        inner = super().save(path / "components")
        manifest = {
            **self._sharing_identity(),
            "components_manifest_sha256": _sha(path / "components" / "manifest.json"),
        }
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {**manifest, "components": inner}

    def load(self, directory):
        path = Path(directory)
        _sha(path / "manifest.json")
        manifest = json.loads((path / "manifest.json").read_text())
        expected = self._sharing_identity()
        if (any(manifest.get(key) != value for key, value in expected.items())
                or manifest.get("components_manifest_sha256")
                != _sha(path / "components" / "manifest.json")):
            raise ContextualMemoryError(
                "matched v12 sharing/checkpoint identity differs; no implicit migration")
        result = super().load(path / "components")
        # Only the outer sharing identity is checked again here; the v12 inner
        # loader already validates source, feature contract, layout and owner order.
        if _identity_value(manifest) != _identity_value(
                {**self._sharing_identity(),
                 "components_manifest_sha256": manifest["components_manifest_sha256"]}):
            raise ContextualMemoryError("matched v12 outer manifest differs after load")
        return {**manifest, "components": result}
