"""Bounded immutable SigLIP2 source and isolated frozen visual features.

The acquisition half of this module never imports ``torch`` or ``transformers``.
It streams only the four declared source files, checks their pinned byte lengths
and SHA256 digests under a 2 GiB aggregate cap, and writes a sealed manifest that
``load_vision_source`` revalidates from bytes on every load.

The service half is the sole native vision path for the isolated MiniCPM5 V0/V1
work. It loads the standard frozen ``SiglipVisionModel`` and the official PIL
image processor (``SiglipImageProcessorPil``) offline. It never uses the
torchvision backend, never runs a CPU model forward and never offers a silent
substitute processor. Missing Pillow is a hard error naming the pinned overlay
requirement.

Visual features are detached float32 spatial patches from
``SiglipVisionModel.last_hidden_state``. The service caches those frozen patches
under a producer/image/scope key, returns per-request clones, and rejects source
or encoder mutation rather than serving stale features. The projector is not
part of this module and is never cached here.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Callable, Mapping
import urllib.error
import urllib.parse
import urllib.request


SOURCE_FORMAT = "replica.vision_encoder_source.v1"
SOURCE_MANIFEST_NAME = "source_manifest.json"
SOURCE_CAP_BYTES = 2 * 1024**3
HARD_MAX_IMAGE_BYTES = 64 * 1024**2
HARD_MAX_IMAGE_PIXELS = 64 * 1024**2
DEFAULT_PROJECTOR_HIDDEN = 2048
CHUNK_BYTES = 8 * 1024**2
HTTP_TIMEOUT_SECONDS = 120
FEATURE_VERSION = "siglip_patch_last_hidden.v1"
PROCESSOR_API = (
    "transformers.models.siglip.image_processing_pil_siglip."
    "SiglipImageProcessorPil"
)
SERVICE_IMPLEMENTATION = "replica.vision_feature_service.v1"
REQUIRED_ROLES = frozenset(
    {"model_config", "processor_config", "model_card_terms", "vision_encoder_weights"}
)
SOURCES = ("config.json", "preprocessor_config.json", "README.md", "model.safetensors")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
LICENSE_RE = re.compile(r"license:\s*([A-Za-z0-9.+-]+)\s*\Z")
MODEL_DEFAULTS = {
    "architecture": "SiglipVisionModel",
    "model_type": "siglip_vision_model",
    "base_model_prefix": "vision_model",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 16,
    "max_visual_tokens": 196,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-06,
    "feature": "last_hidden_state",
    "spatial_patches": True,
    "compute_dtype": "float32",
    "attn_implementation": "eager",
    "feature_head": "unused",
    "feature_version": FEATURE_VERSION,
}
PROCESSOR_FIELDS = {
    "image_processor_type": "SiglipImageProcessor",
    "processor_class": "SiglipProcessor",
    "image_processor_api": PROCESSOR_API,
    "backend": "pil",
    "image_size": 224,
    "resample": "bilinear",
    "resample_code": 2,
    "rescale_factor": 0.00392156862745098,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "do_resize": True,
    "do_rescale": True,
    "do_normalize": True,
    "do_convert_rgb": True,
}
FRAMEWORK_PINS = {
    "python": ">=3.12",
    "torch": "2.14.1",
    "transformers": "5.20.2",
    "safetensors": "0.8.0",
    "numpy": "2.5.4",
    "runtime_requirement": "Pillow==12.0.0",
}
PROCESSOR_CONFIG_FIELDS = frozenset({
    "do_convert_rgb", "do_normalize", "do_rescale", "do_resize", "image_mean",
    "image_processor_type", "image_std", "processor_class", "resample",
    "rescale_factor", "size",
})
CONFIG_FIELDS = frozenset({
    "initializer_factor", "model_type", "text_config", "transformers_version",
    "vision_config",
})


class VisionSourceError(ValueError):
    """Pinned source or sealed manifest is absent, malformed or mutated."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Bounded streaming digest with an unchanged-file check."""
    before = _stat_tuple(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            value.update(chunk)
    if _stat_tuple(path) != before:
        raise VisionSourceError(f"Source file changed while hashing: {path}")
    return value.hexdigest()


def _stat_tuple(path: Path):
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise VisionSourceError(f"Declared source file is missing: {path}") from error
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, stat.S_ISREG(info.st_mode))


def _regular(path: Path, *, description: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise VisionSourceError(f"{description} is missing: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise VisionSourceError(f"{description} must be a regular non-symlink file: {path}")


def _safe_relative(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise VisionSourceError("Source file path must be a nonempty POSIX string")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != name:
        raise VisionSourceError(f"Unsafe source file path: {name!r}")
    if len(pure.parts) != 1:
        raise VisionSourceError(f"Pinned source files are top-level only: {name!r}")
    return name


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _seal_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "seal_digest"}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict:
    _regular(path, description="JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VisionSourceError(f"Invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise VisionSourceError(f"Expected a JSON object: {path}")
    return value


# ---------------------------------------------------------------------------
# Configuration and manifest validation.
# ---------------------------------------------------------------------------


def load_source_config(path: str | os.PathLike) -> dict:
    """Read and validate the tracked acquisition pin; no network or model import."""
    config_path = Path(path)
    config = _read_json(config_path)
    if config.get("format") != SOURCE_FORMAT:
        raise VisionSourceError("Unsupported vision source configuration format")
    repo = config.get("repo_id")
    revision = config.get("revision")
    if not isinstance(repo, str) or REPO_RE.fullmatch(repo) is None:
        raise VisionSourceError("repo_id must pin one official repository")
    if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
        raise VisionSourceError("revision must be a 40-hex immutable commit")
    source_id = config.get("source_id")
    if source_id != f"{repo}@{revision}":
        raise VisionSourceError("source_id must equal repo_id@revision")
    output = config.get("output_directory")
    if not isinstance(output, str) or not output:
        raise VisionSourceError("output_directory is required")
    if config.get("manifest_name") != SOURCE_MANIFEST_NAME:
        raise VisionSourceError("manifest_name must be the sealed source manifest")
    acquisition = config.get("acquisition")
    if not isinstance(acquisition, dict):
        raise VisionSourceError("acquisition policy is required")
    max_bytes = acquisition.get("max_bytes")
    if max_bytes != SOURCE_CAP_BYTES:
        raise VisionSourceError("Acquisition cap must be exactly 2 GiB")
    if acquisition.get("chunk_bytes", CHUNK_BYTES) != CHUNK_BYTES:
        raise VisionSourceError("Acquisition chunk size differs from the declared stream")
    for name in ("remote_code_executed", "full_model_loaded", "processor_cpu_forward",
                 "encoder_execution", "mps_execution"):
        if acquisition.get(name) is not False:
            raise VisionSourceError(f"Acquisition must not enable {name}")
    if acquisition.get("weights_downloaded") is not True:
        raise VisionSourceError("The selected vision weights are the declared acquisition")
    files = config.get("files")
    if not isinstance(files, list) or len(files) != len(SOURCES):
        raise VisionSourceError("Exactly the four pinned source files are required")
    seen: set[str] = set()
    roles: set[str] = set()
    total = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "role", "bytes", "sha256"}:
            raise VisionSourceError("Each source file requires path/role/bytes/sha256")
        name = _safe_relative(item["path"])
        if name in seen:
            raise VisionSourceError(f"Duplicate source file: {name}")
        seen.add(name)
        if item["role"] not in REQUIRED_ROLES or item["role"] in roles:
            raise VisionSourceError(f"Unknown or duplicate source role: {item['role']!r}")
        roles.add(item["role"])
        if type(item["bytes"]) is not int or item["bytes"] < 1:
            raise VisionSourceError(f"Source byte length is invalid: {name}")
        if not isinstance(item["sha256"], str) or SHA256_RE.fullmatch(item["sha256"]) is None:
            raise VisionSourceError(f"Source SHA256 is invalid: {name}")
        total += item["bytes"]
    if seen != set(SOURCES) or roles != REQUIRED_ROLES:
        raise VisionSourceError("Pinned source inventory differs from the selected files")
    if total > max_bytes:
        raise VisionSourceError("Pinned source exceeds the 2 GiB acquisition cap")
    by_role = {item["role"]: item for item in files}
    if (by_role["model_config"]["path"] != "config.json"
            or by_role["processor_config"]["path"] != "preprocessor_config.json"
            or by_role["model_card_terms"]["path"] != "README.md"
            or by_role["vision_encoder_weights"]["path"] != "model.safetensors"):
        raise VisionSourceError("Pinned source paths are not the selected official files")
    license_info = config.get("license")
    if (not isinstance(license_info, dict)
            or license_info.get("spdx") != "Apache-2.0"
            or license_info.get("card_license") != "apache-2.0"
            or license_info.get("terms_path") != "README.md"
            or license_info.get("commercial_use") is not True
            or license_info.get("redistribution") is not True):
        raise VisionSourceError("Apache-2.0 card/license provenance is required")
    model = config.get("model")
    if not isinstance(model, dict) or model != MODEL_DEFAULTS:
        raise VisionSourceError("Declared SigLIP2 vision configuration differs")
    if model["image_size"] != 224 or model["patch_size"] != 16:
        raise VisionSourceError("Only the fixed 224/16 SigLIP2 vision layout is accepted")
    if model["image_size"] % model["patch_size"]:
        raise VisionSourceError("Image size must divide patch size")
    tokens = (model["image_size"] // model["patch_size"]) ** 2
    if model["max_visual_tokens"] != tokens:
        raise VisionSourceError("max_visual_tokens must equal the spatial patch count")
    processor = config.get("processor")
    if not isinstance(processor, dict):
        raise VisionSourceError("Processor pin is required")
    for name, expected in PROCESSOR_FIELDS.items():
        if processor.get(name) != expected:
            raise VisionSourceError(f"Processor pin differs at {name}")
    if processor.get("runtime_requirement") != FRAMEWORK_PINS["runtime_requirement"]:
        raise VisionSourceError(
            "Processor runtime requirement must pin the isolated overlay Pillow")
    framework = config.get("framework")
    if not isinstance(framework, dict) or framework != FRAMEWORK_PINS:
        raise VisionSourceError("Framework pins differ from the isolated vision environment")
    return config


def _verify_card_license(text: str, expected: str) -> None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise VisionSourceError("Model card has no leading YAML front matter")
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1)
                   if line.strip() == "---")
    except StopIteration as error:
        raise VisionSourceError("Model card front matter is unterminated") from error
    matches = [LICENSE_RE.fullmatch(line.strip()) for line in lines[1:end]]
    licenses = [match.group(1).lower() for match in matches if match is not None]
    if licenses != [expected]:
        raise VisionSourceError(
            f"Model card license differs: expected {expected}, observed {licenses}")


def _verify_model_config(config: Mapping[str, Any], model: Mapping[str, Any]) -> None:
    if set(config) - CONFIG_FIELDS:
        raise VisionSourceError(
            f"Unexpected model config fields: {sorted(set(config) - CONFIG_FIELDS)}")
    if config.get("model_type") != "siglip":
        raise VisionSourceError("Pinned model config is not SigLIP2")
    vision = config.get("vision_config")
    if not isinstance(vision, dict) or vision.get("model_type") != "siglip_vision_model":
        raise VisionSourceError("Pinned model config lacks the SigLIP vision tower")
    comparable = {
        "hidden_size": model["hidden_size"],
        "intermediate_size": model["intermediate_size"],
        "num_hidden_layers": model["num_hidden_layers"],
        "num_attention_heads": model["num_attention_heads"],
        "num_channels": model["num_channels"],
        "image_size": model["image_size"],
        "patch_size": model["patch_size"],
        "hidden_act": model["hidden_act"],
        "layer_norm_eps": model["layer_norm_eps"],
    }
    for name, value in vision.items():
        if name == "model_type":
            continue
        if name not in comparable:
            raise VisionSourceError(f"Unexpected vision config field: {name}")
        if value != comparable[name]:
            raise VisionSourceError(f"Vision config differs at {name}")
    if not isinstance(config.get("text_config"), dict):
        raise VisionSourceError("Pinned full repository config must retain its text tower")


def _verify_processor_config(config: Mapping[str, Any], processor: Mapping[str, Any]) -> None:
    unknown = set(config) - PROCESSOR_CONFIG_FIELDS
    if unknown:
        raise VisionSourceError(f"Unexpected processor config fields: {sorted(unknown)}")
    if config.get("image_processor_type") != processor["image_processor_type"]:
        raise VisionSourceError("Processor class does not match its pinned config")
    if config.get("processor_class") != processor["processor_class"]:
        raise VisionSourceError("Processor wrapper class differs")
    if config.get("size") != {"height": 224, "width": 224}:
        raise VisionSourceError("Processor image size differs from 224x224")
    try:
        rescale = float(config.get("rescale_factor"))
        resample = int(config.get("resample"))
        means = [float(value) for value in config.get("image_mean")]
        deviations = [float(value) for value in config.get("image_std")]
    except (TypeError, ValueError) as error:
        raise VisionSourceError("Processor numeric configuration is malformed") from error
    if rescale != processor["rescale_factor"]:
        raise VisionSourceError("Processor rescale factor differs")
    if resample != processor["resample_code"]:
        raise VisionSourceError("Processor resample filter differs from the pinned bilinear filter")
    for name, observed, expected in (("image_mean", means, processor["image_mean"]),
                                     ("image_std", deviations, processor["image_std"])):
        if observed != expected:
            raise VisionSourceError(f"Processor {name} differs")
    for name in ("do_resize", "do_rescale", "do_normalize"):
        if config.get(name) is not True:
            raise VisionSourceError(f"Processor {name} must be enabled")
    if config.get("do_convert_rgb") is not None and config.get("do_convert_rgb") is not True:
        raise VisionSourceError("Processor RGB conversion must not be disabled")


def _inspect_vision_weights(path: Path, model: Mapping[str, Any]) -> dict:
    """Read only the safetensors header; no tensor is materialized or executed."""
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise VisionSourceError("safetensors is required to seal vision weights") from error
    try:
        with safe_open(path, framework="np") as handle:
            keys = list(handle.keys())
            shapes = {name: tuple(handle.get_slice(name).get_shape()) for name in keys}
    except Exception as error:
        raise VisionSourceError(f"Invalid safetensors vision weights: {error}") from error
    if not keys:
        raise VisionSourceError("Vision weights contain no tensors")
    projector = sorted(name for name in keys
                       if any(token in name.lower()
                              for token in ("projector", "multi_modal", "mm_projector", "resampler")))
    if projector:
        raise VisionSourceError(
            f"Source contains an unsealed multimodal projector: {projector[:4]}")
    vision_keys = [name for name in keys if name.startswith("vision_model.")]
    text_keys = [name for name in keys if name.startswith("text_model.")]
    other_keys = [name for name in keys if name not in vision_keys and name not in text_keys]
    layers = set()
    for name in vision_keys:
        match = re.match(r"vision_model\.encoder\.layers\.(\d+)\.", name)
        if match:
            layers.add(int(match.group(1)))
    if sorted(layers) != list(range(model["num_hidden_layers"])):
        raise VisionSourceError("Vision encoder layer inventory differs")
    hidden = model["hidden_size"]
    patch = model["patch_size"]
    channels = model["num_channels"]
    patches = model["max_visual_tokens"]
    required = {
        "vision_model.embeddings.patch_embedding.weight": [hidden, channels, patch, patch],
        "vision_model.embeddings.position_embedding.weight": [patches, hidden],
        "vision_model.post_layernorm.weight": [hidden],
        "vision_model.post_layernorm.bias": [hidden],
    }
    for name, shape in required.items():
        if list(shapes.get(name, ())) != shape:
            raise VisionSourceError(
                f"Vision tensor shape differs at {name}: {shapes.get(name)} != {shape}")
    elements = sum(_shape_elements(shapes[name]) for name in vision_keys)
    return {
        "tensor_count": len(keys),
        "vision_tensor_count": len(vision_keys),
        "text_tensor_count": len(text_keys),
        "other_tensor_count": len(other_keys),
        "vision_layer_count": len(layers),
        "vision_parameter_elements": elements,
        "required_shapes": required,
        "projector_tensor_count": 0,
        "other_tensor_names": sorted(other_keys),
    }


def _shape_elements(shape) -> int:
    value = 1
    for size in shape:
        value *= int(size)
    return value


def _validate_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    if manifest.get("format") != SOURCE_FORMAT:
        raise VisionSourceError("Not a sealed vision source manifest")
    if manifest.get("status") != "sealed":
        raise VisionSourceError("Vision source is not sealed")
    repo, revision = manifest.get("repo_id"), manifest.get("revision")
    if not isinstance(repo, str) or REPO_RE.fullmatch(repo) is None:
        raise VisionSourceError("Sealed manifest repo_id is invalid")
    if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
        raise VisionSourceError("Sealed manifest revision is invalid")
    if manifest.get("source_id") != f"{repo}@{revision}":
        raise VisionSourceError("Sealed manifest source_id differs")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise VisionSourceError("Sealed manifest has no files")
    seen = set()
    roles = set()
    total = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "role", "bytes", "sha256"}:
            raise VisionSourceError("Sealed file entries require path/role/bytes/sha256")
        name = _safe_relative(item["path"])
        if name in seen:
            raise VisionSourceError(f"Duplicate sealed file: {name}")
        seen.add(name)
        if item["role"] not in REQUIRED_ROLES or item["role"] in roles:
            raise VisionSourceError("Sealed file role is invalid")
        roles.add(item["role"])
        if type(item["bytes"]) is not int or item["bytes"] < 1:
            raise VisionSourceError(f"Sealed byte length is invalid: {name}")
        if not isinstance(item["sha256"], str) or SHA256_RE.fullmatch(item["sha256"]) is None:
            raise VisionSourceError(f"Sealed SHA256 is invalid: {name}")
        total += item["bytes"]
    if seen != set(SOURCES) or roles != REQUIRED_ROLES:
        raise VisionSourceError("Sealed source inventory differs from the selected files")
    acquisition = manifest.get("acquisition")
    if not isinstance(acquisition, dict):
        raise VisionSourceError("Sealed manifest lacks acquisition provenance")
    cap = acquisition.get("max_bytes")
    if type(cap) is not int or cap != SOURCE_CAP_BYTES or total > cap:
        raise VisionSourceError("Sealed source violates the 2 GiB cap")
    if acquisition.get("remote_code_executed") is not False:
        raise VisionSourceError("Sealed acquisition must never execute remote code")
    if acquisition.get("full_model_loaded") is not False:
        raise VisionSourceError("Sealed acquisition must not load the full model")
    if acquisition.get("processor_cpu_forward") is not False:
        raise VisionSourceError("Sealed acquisition must not run a processor forward")
    if acquisition.get("encoder_execution") is not False:
        raise VisionSourceError("Sealed acquisition must not execute the encoder")
    if acquisition.get("mps_execution") is not False:
        raise VisionSourceError("Sealed acquisition must not execute on MPS")
    model = manifest.get("model")
    if model != MODEL_DEFAULTS:
        raise VisionSourceError("Sealed model configuration differs from the pin")
    processor = manifest.get("processor")
    if not isinstance(processor, dict):
        raise VisionSourceError("Sealed manifest lacks the processor pin")
    for name, expected in PROCESSOR_FIELDS.items():
        if processor.get(name) != expected:
            raise VisionSourceError(f"Sealed processor differs at {name}")
    if processor.get("runtime_requirement") != FRAMEWORK_PINS["runtime_requirement"]:
        raise VisionSourceError("Sealed processor runtime requirement differs")
    if manifest.get("framework") != FRAMEWORK_PINS:
        raise VisionSourceError("Sealed framework pins differ")
    license_info = manifest.get("license")
    if (not isinstance(license_info, dict)
            or license_info.get("spdx") != "Apache-2.0"
            or license_info.get("card_license") != "apache-2.0"):
        raise VisionSourceError("Sealed license provenance is not Apache-2.0")
    vision_config = manifest.get("vision_config")
    if not isinstance(vision_config, dict) or vision_config.get("hidden_size") != 768:
        raise VisionSourceError("Sealed actual vision configuration is absent")
    inventory = manifest.get("tensor_inventory")
    if not isinstance(inventory, dict) or inventory.get("projector_tensor_count") != 0:
        raise VisionSourceError("Sealed vision tensor inventory is invalid")
    if not isinstance(manifest.get("seal_digest"), str):
        raise VisionSourceError("Sealed manifest has no seal digest")
    if manifest["seal_digest"] != _seal_digest(manifest):
        raise VisionSourceError("Sealed manifest digest does not match its content")
    if manifest_path.name != SOURCE_MANIFEST_NAME:
        raise VisionSourceError("Sealed source manifest must be source_manifest.json")


@dataclass(frozen=True)
class VisionSource:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    seal_digest: str
    repo_id: str
    revision: str
    source_id: str
    license: Mapping[str, Any]
    files: tuple[Mapping[str, Any], ...]
    model: Mapping[str, Any]
    processor: Mapping[str, Any]
    framework: Mapping[str, Any]
    vision_config: Mapping[str, Any]
    tensor_inventory: Mapping[str, Any]
    acquisition: Mapping[str, Any]
    manifest: Mapping[str, Any] = field(repr=False, compare=False)
    file_stats: tuple[tuple[str, tuple], ...] = field(repr=False, compare=False)

    @property
    def config_sha256(self) -> str:
        return self._file("config.json")["sha256"]

    @property
    def processor_sha256(self) -> str:
        return self._file("preprocessor_config.json")["sha256"]

    @property
    def card_sha256(self) -> str:
        return self._file("README.md")["sha256"]

    @property
    def encoder_source_digest(self) -> str:
        return self._file("model.safetensors")["sha256"]

    def _file(self, name: str) -> Mapping[str, Any]:
        for item in self.files:
            if item["path"] == name:
                return item
        raise VisionSourceError(f"Sealed source lacks {name}")


def _resolve_manifest_path(path: str | os.PathLike) -> Path:
    supplied = Path(path)
    if supplied.is_dir() and not supplied.is_symlink():
        return supplied / SOURCE_MANIFEST_NAME
    return supplied


def load_vision_source(path: str | os.PathLike) -> VisionSource:
    """Load and byte-validate one sealed immutable source; never imports torch."""
    manifest_path = _resolve_manifest_path(path)
    _regular(manifest_path, description="Sealed source manifest")
    supplied_root = manifest_path.parent
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise VisionSourceError("Sealed source root must be a nonsymlink directory")
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest_raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VisionSourceError(f"Invalid sealed source manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise VisionSourceError("Sealed source manifest must be a JSON object")
    _validate_manifest(manifest, manifest_path)
    expected = {item["path"] for item in manifest["files"]}
    allowed = expected | {SOURCE_MANIFEST_NAME, "download.json"}
    for entry in root.iterdir():
        if entry.name not in allowed:
            raise VisionSourceError(f"Unsealed file in source directory: {entry.name}")
        _regular(entry, description="Sealed source entry")
    stats = {"source_manifest.json": _stat_tuple(manifest_path)}
    total = 0
    for item in manifest["files"]:
        candidate = root / item["path"]
        _regular(candidate, description="Sealed source file")
        if _stat_tuple(candidate)[2] != item["bytes"]:
            raise VisionSourceError(f"Sealed byte length differs: {item['path']}")
        if _sha256_file(candidate) != item["sha256"]:
            raise VisionSourceError(f"Sealed byte digest differs: {item['path']}")
        stats[item["path"]] = _stat_tuple(candidate)
        total += item["bytes"]
    if total > manifest["acquisition"]["max_bytes"]:
        raise VisionSourceError("Sealed source exceeds the 2 GiB cap")
    by_path = {item["path"]: item for item in manifest["files"]}
    config = _read_json(root / "config.json")
    processor_config = _read_json(root / "preprocessor_config.json")
    try:
        card = (root / "README.md").read_bytes().decode("utf-8")
    except UnicodeError as error:
        raise VisionSourceError("Model card is not valid UTF-8") from error
    _verify_card_license(card, manifest["license"]["card_license"])
    _verify_model_config(config, manifest["model"])
    _verify_processor_config(processor_config, manifest["processor"])
    actual_vision = dict(manifest["model"])
    actual_vision.pop("architecture", None)
    actual_vision.pop("base_model_prefix", None)
    actual_vision.pop("feature", None)
    actual_vision.pop("spatial_patches", None)
    actual_vision.pop("compute_dtype", None)
    actual_vision.pop("attn_implementation", None)
    actual_vision.pop("feature_head", None)
    actual_vision.pop("feature_version", None)
    actual_vision.pop("max_visual_tokens", None)
    if manifest["vision_config"] != actual_vision:
        raise VisionSourceError("Sealed actual vision config differs from the declared model")
    inventory = _inspect_vision_weights(root / "model.safetensors", manifest["model"])
    if manifest["tensor_inventory"] != inventory:
        raise VisionSourceError("Sealed tensor inventory differs from the weight header")
    for name, expected_stat in tuple(stats.items()):
        if _stat_tuple(root / name) != expected_stat:
            raise VisionSourceError(f"Sealed source changed during validation: {name}")
    return VisionSource(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_bytes(manifest_raw),
        seal_digest=manifest["seal_digest"],
        repo_id=manifest["repo_id"],
        revision=manifest["revision"],
        source_id=manifest["source_id"],
        license=manifest["license"],
        files=tuple(dict(item) for item in manifest["files"]),
        model=dict(manifest["model"]),
        processor=dict(manifest["processor"]),
        framework=dict(manifest["framework"]),
        vision_config=dict(manifest["vision_config"]),
        tensor_inventory=dict(manifest["tensor_inventory"]),
        acquisition=dict(manifest["acquisition"]),
        manifest=manifest,
        file_stats=tuple(sorted(stats.items())),
    )


# ---------------------------------------------------------------------------
# Acquisition. This half never imports torch or transformers.
# ---------------------------------------------------------------------------


def _validate_remote_metadata(metadata: dict, config: dict) -> dict:
    repo, revision = config["repo_id"], config["revision"]
    if not isinstance(metadata, dict):
        raise VisionSourceError("Hugging Face model metadata is not an object")
    if metadata.get("id") != repo or metadata.get("sha") != revision:
        raise VisionSourceError("Remote repository identity differs from the immutable pin")
    if metadata.get("private") is not False or metadata.get("gated") is not False:
        raise VisionSourceError("Pinned source must be public and ungated")
    if metadata.get("disabled") is not False:
        raise VisionSourceError("Pinned source repository is disabled")
    card = metadata.get("cardData")
    license_value = card.get("license") if isinstance(card, dict) else None
    if license_value != config["license"]["card_license"]:
        raise VisionSourceError(
            f"Remote card license differs: {license_value!r} != Apache-2.0")
    siblings = metadata.get("siblings")
    names = ({item.get("rfilename") for item in siblings if isinstance(item, dict)}
             if isinstance(siblings, list) else set())
    if not set(SOURCES) <= names:
        raise VisionSourceError("Remote revision lacks one of the pinned source files")
    return {"id": repo, "sha": revision, "lastModified": metadata.get("lastModified"),
            "card_license": license_value, "observed_files": sorted(names)}


def _fetch_remote_metadata(config: dict) -> dict:
    url = (f"https://huggingface.co/api/models/{config['repo_id']}"
           f"/revision/{config['revision']}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "replica-lab-vision-source/1", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise VisionSourceError(f"Remote metadata HTTP status {response.status}")
            payload = response.read(4 * 1024**2 + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise VisionSourceError(
            f"Vision source metadata request failed: {type(error).__name__}: {error}") from error
    if len(payload) > 4 * 1024**2:
        raise VisionSourceError("Remote metadata response exceeded its bound")
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VisionSourceError(f"Remote metadata is not JSON: {error}") from error
    return _validate_remote_metadata(metadata, config)


def _download_file(config: dict, item: dict, destination: Path, remaining: int) -> dict:
    name = item["path"]
    expected_bytes = item["bytes"]
    if expected_bytes > remaining:
        raise VisionSourceError(
            f"Acquiring {name} would exceed the 2 GiB aggregate cap")
    url = (f"https://huggingface.co/{config['repo_id']}/resolve/"
           f"{config['revision']}/{name}?download=true")
    request = urllib.request.Request(
        url, headers={"User-Agent": "replica-lab-vision-source/1",
                      "Accept": "application/octet-stream"})
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        temporary.unlink()
    digest = hashlib.sha256()
    written = 0
    try:
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                if response.status != 200:
                    raise VisionSourceError(f"{name}: HTTP status {response.status}")
                final = urllib.parse.urlsplit(response.geturl())
                if final.scheme != "https":
                    raise VisionSourceError(f"{name}: redirect left HTTPS")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != expected_bytes:
                    raise VisionSourceError(
                        f"{name}: Content-Length {length} != pinned {expected_bytes}")
                with temporary.open("wb") as stream:
                    while True:
                        chunk = response.read(min(CHUNK_BYTES, expected_bytes - written + 1))
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > expected_bytes:
                            raise VisionSourceError(
                                f"{name}: response exceeded pinned {expected_bytes} bytes")
                        digest.update(chunk)
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise VisionSourceError(
                f"{name}: download failed: {type(error).__name__}: {error}") from error
        actual = digest.hexdigest()
        if written != expected_bytes:
            raise VisionSourceError(
                f"{name}: downloaded {written} bytes != pinned {expected_bytes}")
        if actual != item["sha256"]:
            raise VisionSourceError(
                f"{name}: SHA256 {actual} != pinned {item['sha256']}")
        os.replace(temporary, destination)
        return {"path": name, "bytes": written, "sha256": actual,
                "url": url, "final_scheme": "https", "status": "downloaded"}
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _verify_present(path: Path, item: dict) -> None:
    _regular(path, description=f"Source file {item['path']}")
    if path.stat().st_size != item["bytes"]:
        raise VisionSourceError(
            f"{item['path']}: existing bytes {path.stat().st_size} != pinned {item['bytes']}")
    if _sha256_file(path) != item["sha256"]:
        raise VisionSourceError(f"{item['path']}: existing SHA256 differs from the pin")


def _actual_vision_config(model: Mapping[str, Any]) -> dict:
    keys = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_channels", "image_size", "patch_size",
            "hidden_act", "layer_norm_eps")
    return {name: model[name] for name in keys}


def _build_manifest(config: dict, root: Path, *, network_used: bool,
                    bytes_downloaded: int, remote_metadata: dict | None,
                    completed_at: str) -> dict:
    weights = next(item for item in config["files"] if item["role"] == "vision_encoder_weights")
    inventory = _inspect_vision_weights(root / weights["path"], config["model"])
    acquisition = dict(config["acquisition"])
    acquisition.update({"network_used": network_used,
                        "bytes_downloaded": bytes_downloaded,
                        "completed_at": completed_at,
                        "remote_metadata": remote_metadata})
    manifest = {
        "format": SOURCE_FORMAT,
        "status": "sealed",
        "source_id": config["source_id"],
        "repo_id": config["repo_id"],
        "revision": config["revision"],
        "created_at": completed_at,
        "license": dict(config["license"]),
        "acquisition": acquisition,
        "files": [dict(item) for item in config["files"]],
        "model": dict(config["model"]),
        "processor": dict(config["processor"]),
        "framework": dict(config["framework"]),
        "vision_config": _actual_vision_config(config["model"]),
        "tensor_inventory": inventory,
    }
    manifest["seal_digest"] = _seal_digest(manifest)
    return manifest


def acquire_vision_source(config_path: str | os.PathLike, output: str | os.PathLike | None = None,
                          *, offline: bool = False,
                          report_dir: str | os.PathLike | None = None) -> dict:
    """Acquire or reuse the sealed source under the 2 GiB cap.

    A network failure is written verbatim to ``download.json`` and the optional
    report, then re-raised. Existing mismatched or unsealed files are never
    silently replaced.
    """
    config = load_source_config(config_path)
    root = Path(output if output is not None else config["output_directory"])
    if root.is_symlink():
        raise VisionSourceError("Vision source output must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "download.json"
    report_path = (Path(report_dir) / "acquisition_report.json"
                   if report_dir is not None else None)
    started = _now()
    receipt = {"format": "replica.vision_source_acquisition.v1",
               "source_id": config["source_id"], "repo_id": config["repo_id"],
               "revision": config["revision"], "config_path": str(config_path),
               "config_sha256": _sha256_file(Path(config_path)),
               "requested_at": started, "status": "downloading", "offline": offline,
               "network_used": False, "bytes_downloaded": 0, "files": []}
    _write_json(receipt_path, receipt)
    try:
        manifest_path = root / SOURCE_MANIFEST_NAME
        if manifest_path.exists():
            source = load_vision_source(root)
            if source.manifest["files"] != config["files"]:
                raise VisionSourceError(
                    "Existing sealed source differs from the tracked configuration")
            receipt.update({"status": "reused_sealed_source", "network_used": False,
                            "bytes_downloaded": 0, "completed_at": _now(),
                            "manifest_sha256": source.manifest_sha256,
                            "seal_digest": source.seal_digest, "files": []})
            _write_json(receipt_path, receipt)
            summary = _acquisition_summary(config, root, receipt, source, api=None)
            if report_path is not None:
                _write_json(report_path, summary)
            return summary
        allowed = {SOURCE_MANIFEST_NAME, "download.json"}
        allowed.update(item["path"] for item in config["files"])
        for entry in root.iterdir():
            if entry.name.endswith(".part"):
                entry.unlink()
                continue
            if entry.name not in allowed:
                raise VisionSourceError(
                    f"Unsealed file in source output; choose a new directory: {entry.name}")
        remote_metadata = None
        if not offline:
            remote_metadata = _fetch_remote_metadata(config)
        remaining = config["acquisition"]["max_bytes"]
        downloaded = 0
        file_results = []
        for item in config["files"]:
            destination = root / item["path"]
            if destination.exists():
                _verify_present(destination, item)
                remaining -= item["bytes"]
                if remaining < 0:
                    raise VisionSourceError(
                        "Existing source files exceed the 2 GiB aggregate cap")
                file_results.append({"path": item["path"], "bytes": item["bytes"],
                                     "sha256": item["sha256"], "status": "reused"})
            else:
                if offline:
                    raise VisionSourceError(
                        f"Offline reuse lacks the sealed file {item['path']}")
                result = _download_file(config, item, destination, remaining)
                remaining -= item["bytes"]
                downloaded += item["bytes"]
                file_results.append(result)
                receipt.update({"bytes_downloaded": downloaded, "files": file_results,
                                "network_used": True})
                _write_json(receipt_path, receipt)
        total = 0
        for item in config["files"]:
            _verify_present(root / item["path"], item)
            total += item["bytes"]
        if total > config["acquisition"]["max_bytes"]:
            raise VisionSourceError("Acquired source exceeds the 2 GiB cap")
        completed_at = _now()
        manifest = _build_manifest(config, root, network_used=downloaded > 0 or not offline,
                                   bytes_downloaded=downloaded,
                                   remote_metadata=remote_metadata,
                                   completed_at=completed_at)
        _write_json(manifest_path, manifest)
        source = load_vision_source(root)
        receipt.update({"status": "sealed", "completed_at": completed_at,
                        "network_used": downloaded > 0 or not offline,
                        "bytes_downloaded": downloaded, "files": file_results,
                        "manifest_sha256": source.manifest_sha256,
                        "seal_digest": source.seal_digest})
        _write_json(receipt_path, receipt)
        summary = _acquisition_summary(config, root, receipt, source,
                                       api=remote_metadata)
        if report_path is not None:
            _write_json(report_path, summary)
        return summary
    except Exception as error:
        receipt.update({"status": "failed", "failed_at": _now(),
                        "error_type": type(error).__name__, "error": str(error)})
        try:
            _write_json(receipt_path, receipt)
        except OSError:
            pass
        if report_path is not None:
            failure = {
                "format": "replica.vision_source_acquisition_report.v1",
                "status": "failed",
                "source_id": config["source_id"],
                "config_path": str(config_path),
                "config_sha256": receipt["config_sha256"],
                "output": str(root),
                "offline": offline,
                "started_at": started,
                "failed_at": receipt["failed_at"],
                "error_type": type(error).__name__,
                "error": str(error),
                "network_supported": None,
                "files": receipt.get("files", []),
                "weights_present": (root / "model.safetensors").exists(),
                "manifest_present": (root / SOURCE_MANIFEST_NAME).exists(),
                "network_used": receipt.get("network_used", False),
                "bytes_downloaded": receipt.get("bytes_downloaded", 0),
            }
            try:
                _write_json(report_path, failure)
            except OSError:
                pass
        raise


def _acquisition_summary(config: dict, root: Path, receipt: dict,
                         source: VisionSource, *, api: dict | None) -> dict:
    return {
        "format": "replica.vision_source_acquisition_report.v1",
        "status": receipt["status"],
        "source_id": source.source_id,
        "repo_id": source.repo_id,
        "revision": source.revision,
        "output": str(root),
        "manifest_path": str(source.manifest_path),
        "manifest_sha256": source.manifest_sha256,
        "seal_digest": source.seal_digest,
        "config_path": receipt["config_path"],
        "config_sha256": receipt["config_sha256"],
        "network_used": receipt["network_used"],
        "bytes_downloaded": receipt["bytes_downloaded"],
        "total_source_bytes": sum(item["bytes"] for item in source.files),
        "cap_bytes": source.acquisition["max_bytes"],
        "within_cap": sum(item["bytes"] for item in source.files) <= source.acquisition["max_bytes"],
        "license": dict(source.license),
        "files": [dict(item) for item in source.files],
        "file_actions": receipt.get("files", []),
        "fresh_remote_metadata": api,
        "vision_config": dict(source.vision_config),
        "tensor_inventory": dict(source.tensor_inventory),
        "processor": dict(source.processor),
        "framework": dict(source.framework),
        "completed_at": receipt.get("completed_at"),
        "weights_downloaded": True,
        "remote_code_executed": False,
        "full_model_loaded": False,
        "processor_cpu_forward": False,
        "encoder_execution": False,
        "mps_execution": False,
        "git_staged": False,
        "native_execution": False,
    }


# ---------------------------------------------------------------------------
# Service-facing construction helpers.
# ---------------------------------------------------------------------------


def vision_spec_from_source(source: VisionSource | str | os.PathLike,
                            projector_hidden: int = DEFAULT_PROJECTOR_HIDDEN):
    """Build the parent-owned VisionSpec from one sealed source."""
    from .contracts import VisionSpec

    if not isinstance(source, VisionSource):
        source = load_vision_source(source)
    if type(projector_hidden) is not int or projector_hidden < 1:
        raise VisionSourceError("projector_hidden must be a positive integer")
    model = source.model
    spec = VisionSpec(
        encoder_repo=source.repo_id,
        encoder_revision=source.revision,
        encoder_source_digest=source.encoder_source_digest,
        processor_digest=source.processor_sha256,
        encoder_width=model["hidden_size"],
        image_size=model["image_size"],
        patch_size=model["patch_size"],
        max_visual_tokens=model["max_visual_tokens"],
        projector_hidden=projector_hidden,
        feature_version=model["feature_version"],
        compute_dtype=model["compute_dtype"],
    )
    spec.validate()
    if spec.max_visual_tokens != (spec.image_size // spec.patch_size) ** 2:
        raise VisionSourceError("VisionSpec spatial patch count differs from its grid")
    return spec


def load_pil_image_processor(source: VisionSource):
    """Load the exact official PIL processor offline; refuse any substitute."""
    try:
        from transformers import SiglipImageProcessorPil
    except ImportError as error:
        requirement = source.processor["runtime_requirement"]
        raise VisionSourceError(
            "SiglipImageProcessorPil requires the pinned PIL image backend. "
            f"Use the isolated vision overlay containing {requirement}; "
            "the baseline environment intentionally has no Pillow. "
            f"Original import error: {error}") from error
    try:
        processor = SiglipImageProcessorPil.from_pretrained(
            source.root, local_files_only=True, force_download=False, revision=None)
    except Exception as error:
        raise VisionSourceError(
            f"Official SiglipImageProcessorPil could not load offline: "
            f"{type(error).__name__}: {error}") from error
    actual_api = f"{type(processor).__module__}.{type(processor).__name__}"
    if actual_api != source.processor["image_processor_api"]:
        raise VisionSourceError(
            f"Refusing processor {actual_api}; the seal requires "
            f"{source.processor['image_processor_api']}")
    if type(processor).__name__ != "SiglipImageProcessorPil":
        raise VisionSourceError("Refusing a non-PIL SigLIP image processor backend")
    expected = source.processor
    observed = {
        "image_size": processor.size,
        "resample": int(processor.resample),
        "rescale_factor": float(processor.rescale_factor),
        "image_mean": [float(value) for value in processor.image_mean],
        "image_std": [float(value) for value in processor.image_std],
        "do_resize": bool(processor.do_resize),
        "do_rescale": bool(processor.do_rescale),
        "do_normalize": bool(processor.do_normalize),
        "do_convert_rgb": processor.do_convert_rgb,
    }
    if observed["image_size"] != {"height": expected["image_size"],
                                  "width": expected["image_size"]}:
        raise VisionSourceError("Loaded processor image size differs from the seal")
    if observed["resample"] != expected["resample_code"]:
        raise VisionSourceError("Loaded processor resample differs from the seal")
    if observed["rescale_factor"] != expected["rescale_factor"]:
        raise VisionSourceError("Loaded processor rescale differs from the seal")
    if observed["image_mean"] != expected["image_mean"]:
        raise VisionSourceError("Loaded processor mean differs from the seal")
    if observed["image_std"] != expected["image_std"]:
        raise VisionSourceError("Loaded processor std differs from the seal")
    for name in ("do_resize", "do_rescale", "do_normalize"):
        if observed[name] is not True:
            raise VisionSourceError(f"Loaded processor disabled {name}")
    if observed["do_convert_rgb"] is not None and observed["do_convert_rgb"] is not True:
        raise VisionSourceError("Loaded processor disabled RGB conversion")
    return processor


def _require_runtime_pins(source: VisionSource) -> dict:
    import sys
    import numpy
    import PIL
    import safetensors
    import torch
    import transformers

    if sys.version_info < (3, 12):
        raise VisionSourceError("The sealed vision source requires Python >=3.12")
    observed = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
    }
    expected = {
        "torch": source.framework["torch"],
        "transformers": source.framework["transformers"],
        "safetensors": source.framework["safetensors"],
        "numpy": source.framework["numpy"],
        "pillow": source.processor["runtime_requirement"].split("==", 1)[1],
    }
    for name, value in expected.items():
        if observed[name] != value:
            raise VisionSourceError(
                f"Runtime {name} version {observed[name]} differs from sealed {value}")
    return observed


def processor_configuration_digest(processor) -> str:
    return _sha256_bytes(_canonical(processor.to_dict()).encode('utf-8'))


def processor_identity(source: VisionSource, processor) -> dict:
    """Implementation identity bound into every VisualInput.encoder_producer."""
    import PIL
    import torch
    import transformers

    return {
        "service": SERVICE_IMPLEMENTATION,
        "source_id": source.source_id,
        "seal_digest": source.seal_digest,
        "encoder_source_digest": source.encoder_source_digest,
        "config_sha256": source.config_sha256,
        "processor_config_sha256": source.processor_sha256,
        "processor_runtime_config_sha256": processor_configuration_digest(processor),
        "processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
        "processor_backend": "pil",
        "pillow_version": PIL.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_class": "transformers.SiglipVisionModel",
        "feature": source.model["feature"],
        "feature_version": source.model["feature_version"],
        "compute_dtype": source.model["compute_dtype"],
        "attn_implementation": source.model["attn_implementation"],
    }


def _encoder_producer(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Bounded frozen feature cache.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CacheEntry:
    features: Any
    valid: Any
    stamp: tuple
    bytes: int


def _tensor_stamp(tensor) -> tuple:
    try:
        version = tensor._version
    except RuntimeError as error:
        raise VisionSourceError(
            "Cached features require version tracking; build under no_grad") from error
    return (id(tensor), version, tuple(tensor.shape), str(tensor.device),
            str(tensor.dtype), tensor.data_ptr())


class FrozenFeatureCache:
    """Small LRU cache of detached encoder patches; owned tensors are never served."""

    def __init__(self, max_bytes: int = 8 * 1024**2):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_cache_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self.evictions = 0

    @property
    def bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str):
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.stamp != (_tensor_stamp(entry.features), _tensor_stamp(entry.valid)):
            self._entries.pop(key, None)
            self._bytes -= entry.bytes
            self.invalidations += 1
            raise RuntimeError("Cached frozen visual features were mutated; invalidated")
        self._entries.move_to_end(key)
        self.hits += 1
        return entry.features.clone(), entry.valid.clone()

    def put(self, key: str, features, valid) -> bool:
        import torch

        if features.requires_grad or valid.requires_grad:
            raise ValueError("Cached visual features must be detached")
        if features.dtype != torch.float32:
            raise ValueError("Cached visual features must be float32")
        if (features.ndim != 3 or features.shape[0] != 1
                or features.shape[1] < 1 or features.shape[2] < 1):
            raise ValueError("Cached visual features must be [1, patches, width]")
        if (valid.dtype != torch.bool or valid.shape != features.shape[:2]
                or valid.device != features.device):
            raise ValueError("Cached visual validity must match the patches")
        copy_features = features.detach().clone().contiguous()
        copy_valid = valid.detach().clone().contiguous()
        if not bool(torch.isfinite(copy_features).all()):
            raise ValueError("Cached visual features must be finite")
        size = copy_features.numel() * copy_features.element_size() + copy_valid.numel()
        if size > self.max_bytes:
            return False
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous.bytes
        entry = _CacheEntry(copy_features, copy_valid,
                            (_tensor_stamp(copy_features), _tensor_stamp(copy_valid)), size)
        self._entries[key] = entry
        self._bytes += size
        while self._bytes > self.max_bytes and self._entries:
            _, removed = self._entries.popitem(last=False)
            self._bytes -= removed.bytes
            self.evictions += 1
        return True

    def invalidate(self, reason: str = "unspecified") -> int:
        count = len(self._entries)
        self._entries.clear()
        self._bytes = 0
        self.invalidations += count
        return count

    def clear(self) -> dict:
        return {"cleared_entries": self.invalidate("clear"), "bytes": self._bytes}

    def audit(self) -> dict:
        return {"entries": len(self._entries), "bytes": self._bytes,
                "max_bytes": self.max_bytes, "hits": self.hits, "misses": self.misses,
                "invalidations": self.invalidations, "evictions": self.evictions}


# ---------------------------------------------------------------------------
# Native frozen feature service.
# ---------------------------------------------------------------------------


class VisionFeatureService:
    """Standard frozen SigLIP2 patches for one isolated task request.

    Construction loads the sealed vision model and official PIL processor
    offline. ``observe`` decodes and preprocesses on CPU, runs exactly one
    encoder pass on MPS under ``no_grad``, and returns an immutable
    ``VisualInput`` whose tensors are per-request clones. No projector is
    loaded or cached here.
    """

    def __init__(self, source_path: str | os.PathLike,
                 source_manifest_path: str | os.PathLike, *,
                 device: str = "mps", max_cache_bytes: int = 8 * 1024**2,
                 projector_hidden: int | None = None):
        self.source = load_vision_source(source_manifest_path)
        if Path(source_path).resolve() != self.source.root.resolve():
            raise VisionSourceError("source_path and source_manifest_path disagree")
        self._spec = (vision_spec_from_source(self.source) if projector_hidden is None
                      else vision_spec_from_source(self.source, projector_hidden=projector_hidden))
        if device != "mps":
            raise VisionSourceError("The frozen encoder runs only on the declared MPS device")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._device_name = "mps"
        self._cache = FrozenFeatureCache(max_cache_bytes)
        import torch
        if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
            raise VisionSourceError(
                "Launch the vision service through the capacity guard entry point")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise VisionSourceError("Set PYTORCH_ENABLE_MPS_FALLBACK=0 before importing torch")
        if not torch.backends.mps.is_available():
            raise VisionSourceError("Actual MPS is required; no CPU encoder fallback")
        self._device = torch.device("mps")
        self._processor = load_pil_image_processor(self.source)
        self._preprocessor = None  # Alternate preprocessing is limited to tiny fixtures.
        self._processor_stamp = processor_configuration_digest(self._processor)
        _require_runtime_pins(self.source)
        from transformers import SiglipVisionModel
        try:
            model = SiglipVisionModel.from_pretrained(
                self.source.root, local_files_only=True, force_download=False,
                trust_remote_code=False, use_safetensors=True,
                dtype=torch.float32, attn_implementation="eager")
        except Exception as error:
            raise VisionSourceError(
                f"SiglipVisionModel could not load offline: "
                f"{type(error).__name__}: {error}") from error
        if type(model).__module__ != "transformers.models.siglip.modeling_siglip":
            raise VisionSourceError("Unexpected vision model implementation")
        if type(model).__name__ != "SiglipVisionModel":
            raise VisionSourceError("Refusing a non-SiglipVisionModel encoder")
        observed = {
            "model_type": model.config.model_type,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "num_channels": model.config.num_channels,
            "image_size": model.config.image_size,
            "patch_size": model.config.patch_size,
            "hidden_act": model.config.hidden_act,
            "layer_norm_eps": model.config.layer_norm_eps,
        }
        if observed != dict(self.source.vision_config):
            raise VisionSourceError(
                "Loaded vision configuration differs from the sealed actual config")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise VisionSourceError("Frozen SigLIP2 encoder must remain float32")
        model.to(self._device)
        self._encoder = model
        self._parameter_stamps = self._stamps()
        identity = processor_identity(self.source, self._processor)
        self._identity = identity
        self._encoder_producer = _encoder_producer(identity)
        self._manifest_stamp = self._current_manifest_stamp()
        self._file_stats = self.source.file_stats
        self._image_reader = None

    @classmethod
    def _for_testing(cls, spec, encoder: Callable, *,
                     preprocessor: Callable | None = None,
                     image_reader: Callable | None = None,
                     encoder_producer: str | None = None,
                     max_cache_bytes: int = 8 * 1024**2):
        """Build the service around an injected tiny encoder; never native."""
        self = cls.__new__(cls)
        self.source = None
        self._spec = spec
        self._device_name = "cpu-test"
        self._device = None
        self._cache = FrozenFeatureCache(max_cache_bytes)
        self._processor = None
        self._processor_stamp = None
        self._encoder = encoder
        self._preprocessor = preprocessor
        self._image_reader = image_reader
        self._parameter_stamps = None
        self._manifest_stamp = None
        self._file_stats = ()
        self._identity = {"service": SERVICE_IMPLEMENTATION, "test_encoder": True}
        self._encoder_producer = (encoder_producer
                                  or _encoder_producer(self._identity))
        return self

    @property
    def spec(self):
        return self._spec

    @property
    def encoder_producer(self) -> str:
        return self._encoder_producer

    @property
    def identity(self) -> dict:
        return dict(self._identity)

    @property
    def device(self) -> str:
        return self._device_name

    def _current_manifest_stamp(self):
        return _stat_tuple(self.source.manifest_path)

    def _stamps(self) -> tuple:
        if self._encoder is None or not hasattr(self._encoder, "parameters"):
            return ()
        parameters = (self._encoder.named_parameters()
                      if hasattr(self._encoder, "named_parameters")
                      else enumerate(self._encoder.parameters()))
        buffers = (self._encoder.named_buffers()
                   if hasattr(self._encoder, "named_buffers") else ())
        return tuple((kind, str(name), id(value), value._version, value.data_ptr(),
                      tuple(value.shape), bool(value.requires_grad), str(value.dtype))
                     for kind, values in (("parameter", parameters), ("buffer", buffers))
                     for name, value in values)

    def _assert_source_unchanged(self) -> None:
        if self._processor is not None and self._processor_stamp is not None:
            if processor_configuration_digest(self._processor) != self._processor_stamp:
                self._cache.invalidate("processor_configuration_mutated")
                raise VisionSourceError("Frozen processor configuration changed; refusing stale features")
        if self.source is not None:
            if self._current_manifest_stamp() != self._manifest_stamp:
                self._cache.invalidate("manifest_mutated")
                raise VisionSourceError("Sealed source manifest changed after load")
            for name, expected in self._file_stats:
                if _stat_tuple(self.source.root / name) != expected:
                    self._cache.invalidate("source_file_mutated")
                    raise VisionSourceError(
                        f"Sealed source file changed after load: {name}")
        if self._parameter_stamps is not None:
            current = self._stamps()
            if current != self._parameter_stamps:
                self._cache.invalidate("encoder_tensors_mutated")
                raise VisionSourceError(
                    "Frozen encoder parameters or buffers changed; refusing cached features")

    def _pixels(self, image_path: Path):
        if getattr(self, "_image_reader", None) is not None:
            return self._image_reader(image_path)
        import io
        from PIL import Image, UnidentifiedImageError

        _regular(image_path, description="Visual input image")
        size = image_path.stat().st_size
        if size < 1 or size > HARD_MAX_IMAGE_BYTES:
            raise VisionSourceError("Visual input image byte length is outside its bound")
        with image_path.open("rb") as stream:
            data = stream.read(HARD_MAX_IMAGE_BYTES + 1)
        if len(data) != size:
            raise VisionSourceError("Visual input image changed while reading")
        image_sha = _sha256_bytes(data)
        try:
            with Image.open(io.BytesIO(data)) as opened:
                opened.load()
                original_size = tuple(int(value) for value in opened.size)
                if (len(original_size) != 2 or original_size[0] < 1
                        or original_size[1] < 1):
                    raise VisionSourceError("Visual input image size is invalid")
                if original_size[0] * original_size[1] > HARD_MAX_IMAGE_PIXELS:
                    raise VisionSourceError("Visual input image exceeds the pixel bound")
                rgb = opened.convert("RGB")
        except (UnidentifiedImageError, Image.DecompressionBombError,
                OSError, ValueError) as error:
            raise VisionSourceError(f"CPU image decode failed: {error}") from error
        return data, image_sha, original_size, rgb

    def _validate_crop(self, crop, original_size):
        if crop is None:
            return (0, 0, original_size[0], original_size[1])
        if (not isinstance(crop, (tuple, list)) or len(crop) != 4
                or any(type(value) is not int for value in crop)):
            raise VisionSourceError("crop must be (x0, y0, x1, y1) integers")
        x0, y0, x1, y1 = (int(value) for value in crop)
        if not (0 <= x0 < x1 <= original_size[0] and 0 <= y0 < y1 <= original_size[1]):
            raise VisionSourceError("crop lies outside the decoded image")
        return (x0, y0, x1, y1)

    def _preprocess_cpu(self, rgb, crop):
        import torch

        if self._preprocessor is not None:
            pixels = self._preprocessor(rgb, crop)
        else:
            try:
                pixels = self._processor(images=rgb.crop(crop), return_tensors="pt")
            except Exception as error:
                raise VisionSourceError(
                    f"Official PIL preprocessing failed: {type(error).__name__}: "
                    f"{error}") from error
        values = pixels["pixel_values"] if isinstance(pixels, Mapping) else pixels
        if (not hasattr(values, "shape") or tuple(values.shape) != (1, 3, 224, 224)
                or values.dtype != torch.float32):
            raise VisionSourceError(
                "Processor did not return float32 [1, 3, 224, 224] pixels")
        return values

    def _encode(self, pixels) -> tuple:
        import torch
        if self._device is None:
            pixel_values = pixels.to(dtype=torch.float32)
        else:
            pixel_values = pixels.to(device=self._device, dtype=torch.float32)
        with torch.no_grad():
            output = self._encoder(pixel_values=pixel_values)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, (tuple, list)):
            hidden = output[0] if output else None
        if hidden is None or hidden.ndim != 3:
            raise VisionSourceError("Encoder did not return last_hidden_state")
        expected_width = self._spec.encoder_width
        expected_tokens = self._spec.max_visual_tokens
        if tuple(hidden.shape) != (1, expected_tokens, expected_width):
            raise VisionSourceError(
                f"Encoder patch shape {tuple(hidden.shape)} differs from "
                f"(1, {expected_tokens}, {expected_width})")
        features = hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(features).all()):
            raise VisionSourceError("Encoder returned non-finite visual features")
        valid = torch.ones((1, expected_tokens), dtype=torch.bool)
        return features, valid

    def _scope(self, scope):
        from ..memory_system.ledger import EligibilityScope
        if not isinstance(scope, EligibilityScope):
            raise TypeError("observe requires an EligibilityScope")
        return scope

    def _cache_key(self, image_sha: str, source_id: str, task_id: str, scope,
                   observed_at: int, source_epoch: int, original_size, crop) -> str:
        return _encoder_producer({
            "producer": self._spec.digest,
            "encoder_producer": self._encoder_producer,
            "image_sha256": image_sha,
            "source_id": source_id,
            "task_id": task_id,
            "subject_id": scope.subject_id,
            "branch_id": scope.branch_id,
            "purpose": scope.purpose,
            "observed_at": observed_at,
            "source_epoch": source_epoch,
            "original_size": list(original_size),
            "crop": list(crop),
            "grid": [self._spec.image_size // self._spec.patch_size,
                     self._spec.image_size // self._spec.patch_size],
        })

    def _visual(self, features, valid, *, image_sha, source_id, task_id, scope,
                observed_at, source_epoch, original_size, crop):
        from .contracts import VisualInput
        grid = (self._spec.image_size // self._spec.patch_size,
                self._spec.image_size // self._spec.patch_size)
        device = self._device if self._device is not None else features.device
        return VisualInput(
            features=features.detach().to(device=device, copy=True).contiguous(),
            valid=valid.detach().to(device=device, copy=True).contiguous(),
            producer=self._spec.digest,
            encoder_producer=self._encoder_producer,
            image_sha256=image_sha,
            source_id=source_id,
            task_id=task_id,
            subject_id=scope.subject_id,
            branch_id=scope.branch_id,
            purpose=scope.purpose,
            observed_at=observed_at,
            source_epoch=source_epoch,
            original_size=original_size,
            crop=crop,
            grid=grid,
        )

    def observe(self, image_path: str | os.PathLike, *, source_id: str, task_id: str,
                scope, observed_at: int, source_epoch: int,
                crop: tuple[int, int, int, int] | None = None):
        """Return one immutable, task-owned VisualInput; CPU preprocess + MPS pass."""
        self._assert_source_unchanged()
        scope = self._scope(scope)
        for name, value in (("source_id", source_id), ("task_id", task_id)):
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise VisionSourceError(f"{name} must be a nonempty identifier")
        if type(observed_at) is not int or observed_at < 0:
            raise VisionSourceError("observed_at must be a nonnegative integer")
        if observed_at > scope.cutoff:
            raise VisionSourceError("observed_at exceeds the scope cutoff")
        if type(source_epoch) is not int or source_epoch < 0:
            raise VisionSourceError("source_epoch must be a nonnegative integer")
        _, image_sha, original_size, rgb = self._pixels(Path(image_path))
        crop = self._validate_crop(crop, original_size)
        key = self._cache_key(image_sha, source_id, task_id, scope, observed_at,
                              source_epoch, original_size, crop)
        cached = self._cache.get(key)
        if cached is None:
            pixels = self._preprocess_cpu(rgb, crop)
            features, valid = self._encode(pixels)
            self._cache.put(key, features, valid)
        else:
            features, valid = cached
        visual = self._visual(features, valid, image_sha=image_sha, source_id=source_id,
                              task_id=task_id, scope=scope, observed_at=observed_at,
                              source_epoch=source_epoch, original_size=original_size,
                              crop=crop)
        visual.assert_immutable()
        return visual

    def frozen_audit(self) -> dict:
        """Verify the loaded sealed encoder, not just its declared feature spec."""
        if self.source is None or self._parameter_stamps is None:
            raise VisionSourceError("Frozen parameter evidence requires a sealed encoder")
        self._assert_source_unchanged()
        parameters = tuple(self._encoder.named_parameters())
        trainable = [name for name, value in parameters if value.requires_grad]
        gradients = [name for name, value in parameters if value.grad is not None]
        elements = sum(value.numel() for _, value in parameters)
        inventory = self.source.tensor_inventory
        if (len(parameters) != inventory["vision_tensor_count"]
                or elements != inventory["vision_parameter_elements"]):
            raise VisionSourceError("Loaded encoder parameters differ from the sealed inventory")
        if trainable or gradients:
            raise VisionSourceError("Frozen encoder has trainable parameters or gradients")
        return {"tensor_count": len(parameters), "parameter_elements": elements,
                "trainable_names": trainable, "gradient_names": gradients,
                "parameter_stamp_unchanged": self._stamps() == self._parameter_stamps,
                "dtypes": sorted({str(value.dtype) for _, value in parameters}),
                "devices": sorted({str(value.device) for _, value in parameters}),
                "encoder_producer": self._encoder_producer}

    def cache_audit(self) -> dict:
        return {"encoder_producer": self._encoder_producer,
                "source_id": None if self.source is None else self.source.source_id,
                "device": self._device_name, **self._cache.audit()}

    def clear_cache(self) -> dict:
        return self._cache.clear()
