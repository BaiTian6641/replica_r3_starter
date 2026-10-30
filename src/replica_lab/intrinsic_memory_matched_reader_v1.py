"""CPU candidate: 2/9/36 access sites share exactly two reader parameter sets.

This is a separate control, never an implicit migration of a sealed checkpoint.
The mapping is fixed before native results: group = (layer // 4) % 2. Groups
inherit the deterministic v4 streams for sites 3 and 7. Only groups are registered
as modules; a site is a dispatch location, not a tensor or optimizer owner.
The v5 producer, store validation, padding, detach boundaries and placement stay
unchanged. Sharing parameters does not match compute or establish useful recall.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from torch import nn

from .architecture.intrinsic_memory_layout_v4 import (
    ConfigurableIntrinsicMemoryReader, IntrinsicMemoryLayout, spark_layout,
)
from .contextual_memory import ContextualMemoryError
from .contextual_memory_v5 import ContextualMemoryV5, atomic_json, file_hash

IMPLEMENTATION = "replica.intrinsic_memory.matched_reader.v1"
FORMAT = IMPLEMENTATION + ".checkpoint"
MAPPING = "alternating_four_layer_groups_anchored_at_3_and_7"
SCOPE = "intrinsic_contextual_phi_matched_reader_v1"


@dataclass(frozen=True)
class MatchedReaderLayout(IntrinsicMemoryLayout):
    def identity(self):
        return {**super().identity(), "reader_sharing": MAPPING,
                "canonical_groups": [0, 1], "initialization_sites": [3, 7],
                "site_to_group": {str(site): (site // 4) % 2 for site in self.sites}}

    def reader_parameter_count(self):
        d, m = self.hidden_size, self.width
        return 2 * (2*d*m + 2*m*m + 2*d + 4*m + 1)


def matched_layout(arm, *, hidden_size=2560):
    layout = spark_layout(arm, hidden_size=hidden_size)
    return MatchedReaderLayout(**vars(layout))


class MatchedParameterReader(ConfigurableIntrinsicMemoryReader):
    def __init__(self, layout, groups):
        # Do not construct len(sites) modules, even temporarily. There is exactly
        # one registered copy of each group and no registered site aliases.
        nn.Module.__init__(self)
        if (not isinstance(layout, MatchedReaderLayout) or len(groups) != 2
                or len({id(group) for group in groups}) != 2):
            raise ContextualMemoryError("two canonical reader groups required")
        self.spec = layout
        self.groups = nn.ModuleDict({str(i): group for i, group in enumerate(groups)})
        self._optimizer_owner = None

    @classmethod
    def install(cls, *args, **kwargs):
        raise ContextualMemoryError("matched reader installation requires the complete contextual wrapper")

    @property
    def ports(self):
        # This plain lookup is not a ModuleDict: aliases must never enter a
        # state_dict, a parameter inventory, or an optimizer group.
        return {str(site): self.groups[str((site // 4) % 2)] for site in self.spec.sites}

    def save_checkpoint(self, *args, **kwargs):
        raise ContextualMemoryError("matched reader requires its complete producer/reader checkpoint")

    def load_checkpoint(self, *args, **kwargs):
        raise ContextualMemoryError("matched reader requires its complete producer/reader checkpoint")


class ContextualMemoryMatchedV1(ContextualMemoryV5):
    """Reuse the real v5 contextual path with a separately bound sharing control.

    Two-site initialization and one optimizer are constructed once, then their
    existing reader tensors are assigned canonical group names. No optimization
    has happened at this point. All later response gradients accumulate into the
    two owners through every active site. Store reads/writes remain caller work.
    """
    def __init__(self, adapter, layout, *, execution_identity="CPU_matched_reader_v1_unbound", **kwargs):
        if not isinstance(layout, MatchedReaderLayout):
            raise ContextualMemoryError("declared matched 2/9/36 layout required")
        allowed = [spark_layout(arm, hidden_size=layout.hidden_size).sites
                   for arm in ("two", "full_attention_nine", "all")]
        if (not isinstance(layout, MatchedReaderLayout) or layout.decoder_layers != 36
                or layout.sites not in allowed):
            raise ContextualMemoryError("declared matched 2/9/36 layout required")
        initial = spark_layout("two", hidden_size=layout.hidden_size)
        super().__init__(adapter, initial, execution_identity=execution_identity, **kwargs)
        old = self.reader
        reader = MatchedParameterReader(layout, (old.ports["3"], old.ports["7"]))
        self.reader, self.spec = reader, layout
        self.adapter.extension.intrinsic_memory = reader
        if self.optimizer.state:
            raise ContextualMemoryError("sharing must be fixed before any update")
        for (name, module), group in zip(self.modules().items(), self.optimizer.param_groups):
            if [id(p) for p in group["params"]] != [id(p) for p in module.parameters()]:
                raise ContextualMemoryError("canonical ownership order differs")
            group["scope"] = SCOPE
            group["param_names"] = [name + "." + key for key, _ in module.named_parameters()]
        old._optimizer_owner = None
        reader._optimizer_owner = self.optimizer
        self.assert_optimizer_ownership()

    def _sharing_identity(self):
        return {"format": FORMAT, "implementation": IMPLEMENTATION,
                "layout": self.spec.identity(), "reader_parameters": self.spec.reader_parameter_count(),
                "optimizer_scope": SCOPE, "projection_cache": "none",
                "native_qualification": False}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        inner = super().save(directory / "contextual_v5")
        manifest = {**self._sharing_identity(),
                    "contextual_manifest_sha256": file_hash(directory / "contextual_v5" / "manifest.json")}
        atomic_json(directory / "manifest.json", manifest)
        return {**manifest, "contextual": inner}

    def load(self, directory):
        directory = Path(directory)
        file_hash(directory / "manifest.json")
        manifest = json.loads((directory / "manifest.json").read_text())
        if (any(manifest.get(k) != value for k, value in self._sharing_identity().items())
                or manifest.get("contextual_manifest_sha256") != file_hash(directory / "contextual_v5" / "manifest.json")):
            raise ContextualMemoryError("matched reader mapping/checkpoint identity differs")
        inner = super().load(directory / "contextual_v5")
        return {**manifest, "contextual": inner}
