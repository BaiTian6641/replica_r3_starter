"""Conditional co-adaptation v2 scope, cache and launch-refusal tests."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab import vision_coadaptation as co  # noqa: E402
import vision_coadaptation_control_runner_v2 as runner  # noqa: E402


class _Parameter:
    def __init__(self, shape, *, trainable=True):
        self.shape = tuple(int(item) for item in shape)
        self.dtype = "float32"
        self.requires_grad = bool(trainable)
        self.grad = None
        self._version = 0

    def numel(self):
        total = 1
        for item in self.shape:
            total *= item
        return total

    def requires_grad_(self, value):
        self.requires_grad = bool(value)
        return self

    def data_ptr(self):
        return 0


def test_scope_owns_exact_projector_last_block_and_post_norm():
    parameters = {
        name: _Parameter(shape)
        for name, shape in co.expected_shapes().items()
    }
    parameters["decoder.weight"] = _Parameter((2,), trainable=True)
    report = co.enable_scope(parameters)
    assert report["trainable_parameter_count"] == 18631424
    assert report["projector_tensors"] == 6
    assert report["encoder_tensors"] == 18
    assert report["adapted_layer_index"] == 11
    assert parameters["decoder.weight"].requires_grad is False
    assert set(report["trainable_names"]) == set(co.SCOPE_TENSORS)


def test_producer_change_invalidates_cached_features():
    version_1 = "a" * 64
    version_2 = "b" * 64
    cache = co.VersionedFeatureCache(max_entries=1)
    cache.put("image", "features", "valid", version=version_1)
    assert cache.lookup("image", version=version_1) == ("features", "valid")
    assert cache.lookup("image", version=version_2) is None
    audit = cache.audit()
    assert audit["entries"] == 0
    assert audit["invalidations"] == ["producer_version_changed"]
    cache.put("image", "new-features", "valid", version=version_2)
    assert cache.lookup("image", version=version_2) == ("new-features", "valid")

    parameter = _Parameter((1,))
    before = co.producer_version({"weight": parameter}.items())
    parameter._version += 1
    assert co.producer_version({"weight": parameter}.items()) != before


def test_protocol_and_real_run_refuse_missing_prerequisites(tmp_path):
    config = runner.load_protocol(ROOT / runner.DEFAULT_CONFIG)
    assert config["name"] == co.NAME
    review = runner.review_launch(ROOT / runner.DEFAULT_CONFIG)
    assert review["launch_allowed"] is False
    assert review["native_execution"] is False
    assert review["model_loads"] == 0
    assert {"smaller_step_report_missing", "frozen_feature_diagnosis_missing",
            "authorization_receipt_missing"} <= set(review["refusals"])
    assert review["isolation"]["trainable_memory_tensors"] == 0
    assert review["isolation"]["trainable_personality_tensors"] == 0
    assert review["isolation"]["memory_optimizer_groups"] == 0
    assert review["isolation"]["personality_components_in_protocol"] is False
    assert review["isolation"]["optimizer_owned_tensors"] == list(co.SCOPE_TENSORS)

    output = tmp_path / "native"
    with pytest.raises(co.CoadaptationError, match="before model load"):
        runner.run_protocol(
            ROOT / runner.DEFAULT_CONFIG,
            output=output,
            model_path=ROOT / "checkpoints/minicpm5_source",
            model_lock=ROOT / "configs/minicpm5_sealed_lock.local.json",
            vision_source=ROOT /
            "checkpoints/siglip2_base_patch16_224_source/source_manifest.json")
    assert not output.exists()
