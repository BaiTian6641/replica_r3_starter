"""Hermetic checks for the sealed SigLIP2 source, cache and service seam.

No real encoder, native MPS job or network is used. The service tests inject a
tiny deterministic callable and a synthetic image reader, so they exercise
source binding, owner metadata, cropping and cache ownership without running
SiglipVisionModel. A processor-only test runs when the isolated PIL overlay is
present.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import require_local_artifacts
import torch

from replica_lab.architecture.contracts import VisionSpec, VisualInput
from replica_lab.architecture.vision import (
    FRAMEWORK_PINS,
    MODEL_DEFAULTS,
    PROCESSOR_FIELDS,
    SOURCE_CAP_BYTES,
    SOURCE_FORMAT,
    FrozenFeatureCache,
    VisionFeatureService,
    VisionSource,
    VisionSourceError,
    _actual_vision_config,
    _inspect_vision_weights,
    _seal_digest,
    _validate_manifest,
    load_pil_image_processor,
    load_vision_source,
    processor_identity,
)
from replica_lab.memory_system import EligibilityScope


PROCESSOR_CONFIG = {
    "do_convert_rgb": None,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "SiglipImageProcessor",
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "SiglipProcessor",
    "resample": 2,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 224, "width": 224},
}
MODEL_CONFIG = {
    "initializer_factor": 1.0,
    "model_type": "siglip",
    "text_config": {"model_type": "siglip_text_model", "vocab_size": 256000},
    "transformers_version": "4.49.0.dev0",
    "vision_config": {"model_type": "siglip_vision_model"},
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_entry(path: Path, role: str) -> dict:
    data = path.read_bytes()
    return {"path": path.name, "role": role, "bytes": len(data), "sha256": _sha(data)}


def _write_safetensors(path: Path) -> None:
    from safetensors.torch import save_file

    hidden, patches, patch = 768, 196, 16
    tensors = {
        "vision_model.embeddings.patch_embedding.weight":
            torch.zeros(hidden, 3, patch, patch),
        "vision_model.embeddings.position_embedding.weight":
            torch.zeros(patches, hidden),
        "vision_model.post_layernorm.weight": torch.zeros(hidden),
        "vision_model.post_layernorm.bias": torch.zeros(hidden),
        "text_model.embeddings.token_embedding.weight": torch.zeros(2, 4),
        "logit_scale": torch.zeros(1),
        "logit_bias": torch.zeros(1),
    }
    for layer in range(12):
        tensors[f"vision_model.encoder.layers.{layer}.layer_norm1.weight"] = torch.zeros(hidden)
    save_file(tensors, str(path))


def _make_source(root: Path, *, mutate_manifest: dict | None = None) -> VisionSource:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(MODEL_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "preprocessor_config.json").write_text(
        json.dumps(PROCESSOR_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        "---\nlicense: apache-2.0\n---\n# Sealed fixture card\n", encoding="utf-8")
    _write_safetensors(root / "model.safetensors")
    files = [
        _file_entry(root / "config.json", "model_config"),
        _file_entry(root / "preprocessor_config.json", "processor_config"),
        _file_entry(root / "README.md", "model_card_terms"),
        _file_entry(root / "model.safetensors", "vision_encoder_weights"),
    ]
    processor_pin = dict(PROCESSOR_FIELDS)
    processor_pin["runtime_requirement"] = FRAMEWORK_PINS["runtime_requirement"]
    manifest = {
        "format": SOURCE_FORMAT,
        "status": "sealed",
        "source_id": "google/siglip2-base-patch16-224@" + "a" * 40,
        "repo_id": "google/siglip2-base-patch16-224",
        "revision": "a" * 40,
        "created_at": "2026-10-25T00:00:00+00:00",
        "license": {
            "spdx": "Apache-2.0", "card_license": "apache-2.0",
            "card_field": "license: apache-2.0", "terms_path": "README.md",
            "commercial_use": True, "redistribution": True,
            "terms_url": "https://huggingface.co/example",
        },
        "acquisition": {
            "max_bytes": SOURCE_CAP_BYTES, "remote_code_executed": False,
            "full_model_loaded": False, "processor_cpu_forward": False,
            "encoder_execution": False, "mps_execution": False,
            "network_used": False, "bytes_downloaded": 0,
            "completed_at": "2026-10-25T00:00:00+00:00",
            "remote_metadata": None,
        },
        "files": files,
        "model": dict(MODEL_DEFAULTS),
        "processor": processor_pin,
        "framework": dict(FRAMEWORK_PINS),
        "vision_config": _actual_vision_config(MODEL_DEFAULTS),
        "tensor_inventory": _inspect_vision_weights(
            root / "model.safetensors", MODEL_DEFAULTS),
    }
    if mutate_manifest is not None:
        manifest.update(mutate_manifest)
    manifest["seal_digest"] = _seal_digest(manifest)
    (root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return load_vision_source(root)


def _spec(width: int = 768, patches: int = 196, image_size: int = 224,
          patch_size: int = 16) -> VisionSpec:
    spec = VisionSpec(
        encoder_repo="google/siglip2-base-patch16-224",
        encoder_revision="a" * 40,
        encoder_source_digest="b" * 64,
        processor_digest="c" * 64,
        encoder_width=width,
        image_size=image_size,
        patch_size=patch_size,
        max_visual_tokens=patches,
    )
    spec.validate()
    return spec


def _scope(as_of: int = 100, purpose: str = "vision-test") -> EligibilityScope:
    return EligibilityScope("subject", "branch", purpose, as_of)


class FakeEncoder(torch.nn.Module):
    def __init__(self, patches: int = 196, width: int = 768):
        super().__init__()
        self.patches = patches
        self.width = width
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, pixel_values):
        assert tuple(pixel_values.shape) == (1, 3, 224, 224)
        value = float(pixel_values.mean()) * float(self.scale.detach())
        features = torch.full((1, self.patches, self.width), value,
                              dtype=torch.float32, requires_grad=False)
        return SimpleNamespace(last_hidden_state=features, pooler_output=None)


class FakePreprocessor:
    def __init__(self, value: float = 0.25):
        self.value = value
        self.calls = 0

    def __call__(self, rgb, crop):
        self.calls += 1
        return torch.full((1, 3, 224, 224), self.value, dtype=torch.float32)


def _reader(size=(64, 48)):
    def read(path):
        data = Path(path).read_bytes()
        return data, _sha(data), size, SimpleNamespace(value=float(len(data)))

    return read


def _service(*, encoder=None, preprocessor=None, max_cache_bytes=8 * 1024**2,
             spec=None):
    encoder = encoder or FakeEncoder()
    return VisionFeatureService._for_testing(
        spec or _spec(), encoder,
        preprocessor=preprocessor or FakePreprocessor(),
        image_reader=_reader(),
        max_cache_bytes=max_cache_bytes,
    )


@pytest.fixture(scope="module")
def sealed_source(tmp_path_factory) -> VisionSource:
    return _make_source(tmp_path_factory.mktemp("sealed_source"))


# ---------------------------------------------------------------------------
# Source pin and seal.
# ---------------------------------------------------------------------------






def test_sealed_source_round_trip(sealed_source):
    source = sealed_source
    assert isinstance(source, VisionSource)
    assert source.source_id.endswith("@" + "a" * 40)
    assert source.encoder_source_digest == source._file("model.safetensors")["sha256"]
    assert source.manifest_sha256
    assert source.tensor_inventory["vision_layer_count"] == 12
    assert source.tensor_inventory["text_tensor_count"] == 1
    assert source.tensor_inventory["projector_tensor_count"] == 0
    assert source.license["spdx"] == "Apache-2.0"




def test_source_rejects_extra_file(tmp_path):
    source = _make_source(tmp_path / "source")
    (source.root / "unknown.bin").write_bytes(b"x")
    with pytest.raises(VisionSourceError, match="Unsealed file"):
        load_vision_source(source.root)


def test_source_rejects_byte_mutation(tmp_path):
    source = _make_source(tmp_path / "source")
    (source.root / "README.md").write_text("changed", encoding="utf-8")
    with pytest.raises(VisionSourceError, match="byte|digest|length"):
        load_vision_source(source.root)


def test_source_rejects_seal_digest_mutation(tmp_path):
    source = _make_source(tmp_path / "source")
    manifest = json.loads((source.root / "source_manifest.json").read_text())
    manifest["seal_digest"] = "0" * 64
    (source.root / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VisionSourceError, match="digest does not match"):
        load_vision_source(source.root)


def test_manifest_rejects_over_cap_inventory(tmp_path):
    source = _make_source(tmp_path / "source")
    manifest = json.loads((source.root / "source_manifest.json").read_text())
    manifest["files"][-1]["bytes"] = SOURCE_CAP_BYTES
    manifest["seal_digest"] = _seal_digest(manifest)
    with pytest.raises(VisionSourceError, match="2 GiB cap"):
        _validate_manifest(manifest, source.manifest_path)


def test_source_rejects_bad_card_license(tmp_path):
    source = _make_source(tmp_path / "source")
    changed = "---\nlicense: mit\n---\n# Changed\n"
    (source.root / "README.md").write_text(changed, encoding="utf-8")
    manifest = json.loads((source.root / "source_manifest.json").read_text())
    encoded = changed.encode("utf-8")
    for item in manifest["files"]:
        if item["path"] == "README.md":
            item["bytes"] = len(encoded)
            item["sha256"] = _sha(encoded)
    manifest["seal_digest"] = _seal_digest(manifest)
    (source.root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(VisionSourceError, match="license"):
        load_vision_source(source.root)


# ---------------------------------------------------------------------------
# Frozen cache.
# ---------------------------------------------------------------------------


def _tiny(features, valid=None):
    if valid is None:
        valid = torch.ones(features.shape[:2], dtype=torch.bool)
    return features, valid


def test_cache_clone_and_hit_accounting():
    cache = FrozenFeatureCache(1024)
    features = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    assert cache.put("key", * _tiny(features))
    first, mask = cache.get("key")
    assert first.shape == (1, 2, 4) and mask.dtype == torch.bool
    first.fill_(99)
    second, _ = cache.get("key")
    assert not torch.equal(first, second)
    audit = cache.audit()
    assert audit["hits"] == 2 and audit["misses"] == 0
    assert audit["bytes"] == features.numel() * 4 + 2


def test_cache_rejects_oversize_and_evicts_lru():
    cache = FrozenFeatureCache(80)
    assert cache.put("a", * _tiny(torch.zeros(1, 2, 4)))
    assert cache.put("b", * _tiny(torch.ones(1, 2, 4)))
    cache.get("a")
    assert cache.put("c", * _tiny(torch.full((1, 2, 4), 2.0)))
    assert cache.get("b") is None
    audit = cache.audit()
    assert audit["evictions"] == 1
    assert audit["entries"] == 2
    assert not cache.put("huge", * _tiny(torch.zeros(1, 64, 64)))


def test_cache_detects_mutation_and_invalidates():
    cache = FrozenFeatureCache()
    assert cache.put("key", * _tiny(torch.zeros(1, 2, 4)))
    entry = cache._entries["key"]
    entry.features.add_(1)
    with pytest.raises(RuntimeError, match="mutated"):
        cache.get("key")
    assert cache.audit()["invalidations"] == 1
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# Service seam (tiny fake tensors; no native encoder).
# ---------------------------------------------------------------------------


def test_observe_binds_owner_layout_and_detaches(tmp_path):
    image = tmp_path / "input.bin"
    image.write_bytes(b"fake image one")
    service = _service()
    scope = _scope()
    visual = service.observe(image, source_id="src-1", task_id="task-1",
                             scope=scope, observed_at=100, source_epoch=7)
    assert isinstance(visual, VisualInput)
    assert visual.features.shape == (1, 196, 768)
    assert visual.valid.shape == (1, 196)
    assert visual.producer == service.spec.digest
    assert visual.encoder_producer == service.encoder_producer
    assert visual.source_id == "src-1"
    assert visual.task_id == "task-1"
    assert visual.subject_id == "subject"
    assert visual.branch_id == "branch"
    assert visual.purpose == "vision-test"
    assert visual.observed_at == 100 and visual.source_epoch == 7
    assert visual.original_size == (64, 48)
    assert visual.crop == (0, 0, 64, 48)
    assert visual.grid == (14, 14)
    assert not visual.features.requires_grad
    visual.assert_immutable()


def test_cache_hit_returns_independent_request_tensors(tmp_path):
    image = tmp_path / "input.bin"
    image.write_bytes(b"fake image two")
    service = _service()
    arguments = dict(source_id="src", task_id="task", scope=_scope(),
                     observed_at=90, source_epoch=1)
    first = service.observe(image, **arguments)
    first.features.fill_(123.0)
    second = service.observe(image, **arguments)
    assert not torch.equal(first.features, second.features)
    audit = service.cache_audit()
    assert audit["hits"] == 1 and audit["misses"] == 1
    assert audit["entries"] == 1


def test_scope_task_and_crop_change_cache_key(tmp_path):
    image = tmp_path / "input.bin"
    image.write_bytes(b"fake image three")
    service = _service()
    service.observe(image, source_id="src", task_id="a", scope=_scope(),
                    observed_at=90, source_epoch=1)
    service.observe(image, source_id="src", task_id="b", scope=_scope(),
                    observed_at=90, source_epoch=1)
    service.observe(image, source_id="src", task_id="a",
                    scope=_scope(purpose="other"), observed_at=90, source_epoch=1)
    service.observe(image, source_id="src", task_id="a", scope=_scope(),
                    observed_at=90, source_epoch=1, crop=(4, 2, 32, 24))
    audit = service.cache_audit()
    assert audit["misses"] == 4 and audit["hits"] == 0


def test_service_rejects_invalid_scope_time_and_crop(tmp_path):
    image = tmp_path / "input.bin"
    image.write_bytes(b"fake image four")
    service = _service()
    with pytest.raises(TypeError):
        service.observe(image, source_id="s", task_id="t", scope="not-a-scope",
                        observed_at=1, source_epoch=0)
    with pytest.raises(VisionSourceError, match="cutoff"):
        service.observe(image, source_id="s", task_id="t", scope=_scope(as_of=10),
                        observed_at=11, source_epoch=0)
    with pytest.raises(VisionSourceError, match="crop"):
        service.observe(image, source_id="s", task_id="t", scope=_scope(),
                        observed_at=1, source_epoch=0, crop=(0, 0, 100, 10))
    with pytest.raises(VisionSourceError, match="crop"):
        service.observe(image, source_id="s", task_id="t", scope=_scope(),
                        observed_at=1, source_epoch=0, crop=(1, 1, 1, 2))
    with pytest.raises(VisionSourceError, match="source_epoch"):
        service.observe(image, source_id="s", task_id="t", scope=_scope(),
                        observed_at=1, source_epoch=-1)


def test_cache_budget_skips_storage_but_serves(tmp_path):
    image = tmp_path / "input.bin"
    image.write_bytes(b"fake image five")
    encoder = FakeEncoder(patches=4, width=4)
    service = _service(encoder=encoder, spec=_spec(width=4, patches=4, image_size=32),
                       max_cache_bytes=1)
    visual = service.observe(image, source_id="s", task_id="t", scope=_scope(),
                             observed_at=1, source_epoch=0)
    assert visual.features.shape == (1, 4, 4)
    assert visual.grid == (2, 2)
    audit = service.cache_audit()
    assert audit["entries"] == 0 and audit["misses"] == 1


def test_manifest_mutation_rejects_and_invalidates(tmp_path):
    image = tmp_path / "input.bin"
    image.write_bytes(b"fake image six")
    service = _service()
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    service.source = SimpleNamespace(root=tmp_path, manifest_path=manifest,
                                     file_stats=(), source_id="synthetic")
    service._manifest_stamp = (manifest.stat().st_dev, manifest.stat().st_ino,
                               manifest.stat().st_size, manifest.stat().st_mtime_ns,
                               manifest.stat().st_ctime_ns, True)
    service._file_stats = ()
    service.observe(image, source_id="s", task_id="t", scope=_scope(),
                    observed_at=1, source_epoch=0)
    manifest.write_text("{\"changed\": true}", encoding="utf-8")
    with pytest.raises(VisionSourceError, match="manifest changed"):
        service.observe(image, source_id="s", task_id="t", scope=_scope(),
                        observed_at=1, source_epoch=0)
    assert service.cache_audit()["invalidations"] >= 1


def test_encoder_parameter_mutation_rejects(tmp_path):
    image = tmp_path / "input.bin"
    image.write_bytes(b"fake image seven")
    encoder = FakeEncoder()
    service = _service(encoder=encoder)
    service._parameter_stamps = service._stamps()
    service.observe(image, source_id="s", task_id="t", scope=_scope(),
                    observed_at=1, source_epoch=0)
    with torch.no_grad():
        encoder.scale.add_(1)
    with pytest.raises(VisionSourceError):
        service.observe(image, source_id="s", task_id="t", scope=_scope(),
                        observed_at=1, source_epoch=0)
    assert service.cache_audit()["invalidations"] >= 1




@require_local_artifacts("PIL")
def test_processor_backend_is_official_pil_when_overlay_present(sealed_source):
    processor = load_pil_image_processor(sealed_source)
    assert type(processor).__name__ == "SiglipImageProcessorPil"
    assert type(processor).__module__.endswith("image_processing_pil_siglip")
    identity = processor_identity(sealed_source, processor)
    assert identity["processor_backend"] == "pil"
    assert identity["processor_class"].endswith("SiglipImageProcessorPil")
    assert identity["processor_config_sha256"] == sealed_source.processor_sha256
    assert identity["pillow_version"] == "12.0.0"
    import numpy as np
    from PIL import Image
    y, x = np.indices((24, 32))
    pixels = np.stack(((x * 43 + y * 17) % 256, (x * 11) % 256, (y * 29) % 256), axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    out = processor(images=image, return_tensors="pt")["pixel_values"]
    resized = np.asarray(image.resize((224, 224), resample=Image.Resampling.BILINEAR)).astype(np.float32)
    expected = torch.from_numpy(((resized / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)).unsqueeze(0)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


def test_load_vision_source_is_regular_file_only(tmp_path, sealed_source):
    # A path that is a directory resolves to its manifest; a missing path fails.
    with pytest.raises(VisionSourceError):
        load_vision_source(tmp_path / "missing")


def test_processor_configuration_mutation_invalidates_cached_observation(tmp_path):
    from replica_lab.architecture.vision import processor_configuration_digest
    image = tmp_path / 'input.bin'
    image.write_bytes(b'processor-cache-observation')
    service = _service()
    configuration = {'resample': 2, 'do_normalize': True}
    service._processor = SimpleNamespace(to_dict=lambda: dict(configuration))
    service._processor_stamp = processor_configuration_digest(service._processor)
    args = dict(source_id='s', task_id='t', scope=_scope(), observed_at=1, source_epoch=0)
    service.observe(image, **args)
    configuration['resample'] = 3
    with pytest.raises(VisionSourceError, match='processor configuration changed'):
        service.observe(image, **args)
    assert service.cache_audit()['invalidations'] >= 1


def test_encoder_position_buffer_mutation_rejects_cached_observation(tmp_path):
    class BufferedEncoder(FakeEncoder):
        def __init__(self):
            super().__init__()
            self.register_buffer('position_ids', torch.arange(self.patches))

        def forward(self, pixel_values):
            output = super().forward(pixel_values)
            output.last_hidden_state += self.position_ids.reshape(1, -1, 1)
            return output

    image = tmp_path / 'position-bound-observation.bin'
    image.write_bytes(b'owned-observed-pixels')
    encoder = BufferedEncoder()
    service = _service(encoder=encoder)
    service._parameter_stamps = service._stamps()
    args = dict(source_id='s', task_id='t', scope=_scope(), observed_at=1, source_epoch=0)
    service.observe(image, **args)
    encoder.position_ids.add_(1)
    with pytest.raises(VisionSourceError):
        service.observe(image, **args)
    assert service.cache_audit()['entries'] == 0
