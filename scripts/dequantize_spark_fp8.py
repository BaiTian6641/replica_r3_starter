#!/usr/bin/env python3
"""Create the single explicitly labelled BF16 derivative of official Spark FP8.

The tool is offline and CPU-only. It validates the pinned payload, the
compressed-tensors quantization group, the FP8 tensor inventory and every
per-output-channel scale before converting one tensor at a time. The result is
a *new* artifact identity: new shard hashes, a new config digest, a derivative
manifest, and explicit activation semantics stating that no per-token F8
quantization happens at runtime.

It never labels the result ``official_native_fp8`` and never substitutes it for
the official FP8 or BF16 release. It refuses an existing output path and writes
at most the one derivative declared by the qualification plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.architecture.spark2_5 import (  # noqa: E402
    DEQUANTIZED_FP8_DERIVATIVE, OFFICIAL_NATIVE_FP8, QUANTIZED_SUFFIXES,
    PrecisionError, canonical_json, dequantized_fp8_derivative, fingerprint,
    load_spark2_5_config, official_native_fp8_precision, sha256_bytes,
    verify_tensor_inventory,
)

DERIVATIVE_SCHEMA = "replica.spark2_5_dequantized_derivative.v1"
PLAN_SCHEMA = "replica.spark2_5_qualification_plan.v1"
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1}


class DequantizationError(RuntimeError):
    """A pinned payload, scale or output identity failed validation."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_digest(files: Sequence[Mapping]) -> str:
    """Deterministic artifact identity over declared filename/hash pairs."""
    entries = sorted(({"filename": str(item["filename"]),
                       "sha256": str(item["sha256"])} for item in files),
                     key=lambda item: item["filename"])
    if not entries:
        raise DequantizationError("an artifact must contain at least one shard")
    for item in entries:
        if not item["sha256"] or len(item["sha256"]) != 64:
            raise DequantizationError("invalid shard digest")
    return sha256_bytes(canonical_json(entries).encode())


def load_plan(path: str | Path, source_role: str) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise DequantizationError(f"plan schema must be {PLAN_SCHEMA}")
    sources = plan.get("sources")
    if not isinstance(sources, dict) or source_role not in sources:
        raise DequantizationError(f"source role {source_role!r} is not in the plan")
    entry = sources[source_role]
    if entry.get("storage") != "fp8_compressed_tensors":
        raise DequantizationError("the dequantizer accepts only official FP8 storage")
    derivative = sources.get("dequantized_fp8_derivative")
    if not isinstance(derivative, dict):
        raise DequantizationError("the plan must declare the derivative role")
    if derivative.get("parent") != source_role:
        raise DequantizationError("the derivative parent differs from the requested source")
    if derivative.get("storage") != "bf16" or derivative.get("max_derivatives") != 1:
        raise DequantizationError("exactly one BF16 derivative is authorized")
    return plan


def verify_source_pins(source_dir: str | Path, entry: Mapping) -> list[dict]:
    """Refuse missing, extra or changed official shards before reading tensors."""
    root = Path(source_dir)
    if not root.is_dir():
        raise DequantizationError(f"missing_official_payload: {root} is not a directory")
    shards = entry.get("shards")
    if not isinstance(shards, list) or not shards:
        raise DequantizationError("the plan declares no shards")
    declared = {item["filename"]: item for item in shards}
    observed = {path.name for path in root.glob("*.safetensors")}
    if observed != set(declared):
        missing = sorted(set(declared) - observed)
        extra = sorted(observed - set(declared))
        raise DequantizationError(
            f"missing_official_payload: shard inventory mismatch; missing={missing} extra={extra}")
    verified = []
    for filename, expected in sorted(declared.items()):
        path = root / filename
        actual_bytes = path.stat().st_size
        if actual_bytes != expected["bytes"]:
            raise DequantizationError(f"{filename}: byte count {actual_bytes} != {expected['bytes']}")
        digest = file_sha256(path)
        if digest != expected["sha256"]:
            raise DequantizationError(f"{filename}: SHA256 differs from the pinned source")
        verified.append({"filename": filename, "bytes": actual_bytes, "sha256": digest})
    config_path = root / "config.json"
    if not config_path.is_file():
        raise DequantizationError("missing_official_payload: config.json is absent")
    if file_sha256(config_path) != entry.get("config_sha256"):
        raise DequantizationError("config.json differs from the pinned source")
    missing_assets = [name for name in entry.get("required_assets", ())
                      if name != "config.json" and not (root / name).is_file()]
    if missing_assets:
        raise DequantizationError(f"missing source assets: {missing_assets}")
    for name, expected in entry.get("asset_sha256", {}).items():
        path = root / name
        if not path.is_file():
            raise DequantizationError(f"missing pinned asset: {name}")
        if file_sha256(path) != expected:
            raise DequantizationError(f"pinned asset SHA256 differs: {name}")
    return verified


def _suffix_matches(name: str) -> bool:
    return any(name.endswith(suffix + ".weight") for suffix in QUANTIZED_SUFFIXES)


def _tensor_metadata(shard: Path) -> list[dict]:
    from safetensors import safe_open

    records = []
    with safe_open(str(shard), framework="pt") as handle:
        for name in handle.keys():
            view = handle.get_slice(name)
            records.append({"name": name, "dtype": str(view.get_dtype()),
                            "shape": list(view.get_shape()), "shard": shard.name})
    return records


def _metadata_for_shards(source_dir: Path) -> list[dict]:
    records: list[dict] = []
    for shard in sorted(source_dir.glob("*.safetensors")):
        records.extend(_tensor_metadata(shard))
    names = [record["name"] for record in records]
    if len(set(names)) != len(names):
        raise DequantizationError("duplicate tensor names across source shards")
    return records


def _load_tensor(source_dir: Path, record: Mapping):
    from safetensors import safe_open

    with safe_open(str(source_dir / record["shard"]), framework="pt") as handle:
        return handle.get_tensor(record["name"])


def _scale_name(weight_name: str) -> str:
    return f"{weight_name}_scale"


def validate_fp8_payload(source_dir: Path, config_raw: Mapping,
                         *, require_official_geometry: bool = True) -> tuple[dict, dict, list[dict], dict]:
    """Validate storage, inventory, scale provenance and value ranges."""
    config = load_spark2_5_config(config_raw,
                                  require_official_geometry=require_official_geometry)
    if "quantization_config" not in config_raw:
        raise DequantizationError("an official FP8 payload must declare quantization_config")
    try:
        parent = official_native_fp8_precision(config)
    except PrecisionError as error:
        raise DequantizationError(f"invalid official FP8 precision: {error}") from error
    records = _metadata_for_shards(source_dir)
    try:
        report = verify_tensor_inventory(records, config, parent)
    except Exception as error:
        raise DequantizationError(f"inventory validation failed: {error}") from error
    by_name = {record["name"]: record for record in records}

    quantized = [record for record in records if record["dtype"] == "F8_E4M3"]
    scales = [record for record in records if record["name"].endswith("weight_scale")]
    if len(quantized) != len(scales):
        raise DequantizationError("FP8 weight and scale counts differ")
    provenance = []
    minimum = None
    maximum = None
    for record in quantized:
        if not _suffix_matches(record["name"]):
            raise DequantizationError(f"{record['name']}: FP8 tensor outside the pinned targets")
        scale_record = by_name.get(_scale_name(record["name"]))
        if scale_record is None:
            raise DequantizationError(f"{record['name']}: missing per-channel weight_scale")
        if scale_record["dtype"] != "BF16":
            raise DequantizationError(f"{scale_record['name']}: scale must be BF16")
        if scale_record["shape"] != [record["shape"][0], 1]:
            raise DequantizationError(f"{scale_record['name']}: scale shape must be [out, 1]")
        scale = _load_tensor(source_dir, scale_record)
        if not bool(_torch_isfinite(scale).all()):
            raise DequantizationError(f"{scale_record['name']}: non-finite scale")
        if not bool((scale > 0).all()):
            raise DequantizationError(f"{scale_record['name']}: non-positive scale")
        value = float(scale.min())
        minimum = value if minimum is None else min(minimum, value)
        maximum = float(scale.max()) if maximum is None else max(maximum, float(scale.max()))
        provenance.append({
            "weight": record["name"], "scale": scale_record["name"],
            "shape": scale_record["shape"], "dtype": scale_record["dtype"],
        })
    quantized_names = {record["name"] for record in quantized}
    expected_scales = {_scale_name(name) for name in quantized_names}
    if {record["name"] for record in scales} != expected_scales:
        raise DequantizationError("orphan scaled tensor without an FP8 weight")
    scale_summary = {
        "strategy": "channel",
        "symmetric": True,
        "scale_dtype": "BF16",
        "formula": "weight_bf16 = round_bf16(e4m3fn_to_f32(byte) * weight_scale[output_channel])",
        "scale_tensor_count": len(scales),
        "scale_tensor_elements": sum(int(record["shape"][0]) for record in scales),
        "scale_name_digest": fingerprint(sorted(expected_scales)),
        "minimum_scale": minimum,
        "maximum_scale": maximum,
        "provenance": provenance,
    }
    inventory = {
        "tensor_count": report.tensor_count,
        "parameter_elements": report.parameter_elements,
        "quantized_tensor_count": report.quantized_tensor_count,
        "scale_tensor_count": report.scale_tensor_count,
        "payload_bytes": report.payload_bytes,
        "dtypes": list(report.dtypes),
        "digest": report.digest,
    }
    return config, inventory, records, scale_summary


def _torch_isfinite(value):
    import torch

    return torch.isfinite(value)


def dequantize_tensor(weight, scale):
    """One channel-wise E4M3FN tensor; never used on MPS."""
    import torch

    if weight.dtype != torch.float8_e4m3fn:
        raise DequantizationError("weight must be torch.float8_e4m3fn")
    if scale.dtype not in (torch.bfloat16, torch.float32):
        raise DequantizationError("scale must be BF16 or FP32")
    if tuple(scale.shape) != (int(weight.shape[0]), 1):
        raise DequantizationError("scale must have one value per output channel")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise DequantizationError("scale must be finite and positive")
    output = weight.to(torch.float32) * scale.to(torch.float32)
    if not bool(torch.isfinite(output).all()):
        raise DequantizationError("dequantized tensor contains NaN/Inf")
    return output.to(torch.bfloat16)


def _output_elements(shape: Sequence[int]) -> int:
    total = 1
    for value in shape:
        total *= int(value)
    return total


def _output_bytes(record: Mapping) -> int:
    return _output_elements(record["shape"]) * 2


def plan_output_groups(records: Sequence[Mapping], shard_bytes: int) -> list[list[dict]]:
    """Group source tensors into bounded BF16 output shards without splitting."""
    if shard_bytes <= 0:
        raise DequantizationError("shard_bytes must be positive")
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    for record in records:
        if record["dtype"] == "BF16" and record["name"].endswith("weight_scale"):
            continue
        size = _output_bytes(record)
        if current and current_bytes + size > shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(dict(record))
        current_bytes += size
    if current:
        groups.append(current)
    if not groups:
        raise DequantizationError("no tensors left after removing scales")
    return groups


def _write_derivative_config(config_raw: Mapping, *, parent, manifest_semantics: str,
                             script_sha256: str) -> dict:
    config = dict(config_raw)
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    config["transformers_version"] = "4.57.1"
    config["replica_precision"] = {
        "kind": DEQUANTIZED_FP8_DERIVATIVE,
        "parent_kind": OFFICIAL_NATIVE_FP8,
        "parent_artifact_sha256": parent.artifact_sha256,
        "dequantization_sha256": script_sha256,
        "activation_semantics": manifest_semantics,
        "equivalent_to_bf16_release": False,
    }
    return config


def dequantize_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    config_raw: Mapping,
    parent,
    script_sha256: str,
    activation_semantics: str,
    asset_names: Sequence[str] = (),
    shard_bytes: int = 1024 ** 3,
    require_official_geometry: bool = True,
    min_free_bytes: int = 0,
    max_output_bytes: int | None = None,
) -> dict:
    """Validate and write one new BF16 derivative; never reuse the parent identity."""
    import torch
    from safetensors.torch import save_file

    if parent.kind != OFFICIAL_NATIVE_FP8 or parent.artifact_sha256 is None:
        raise DequantizationError("parent must be an official native FP8 identity with a digest")
    source = Path(source_dir)
    output = Path(output_dir)
    if not source.is_dir():
        raise DequantizationError(f"missing_official_payload: {source}")
    if output.exists():
        raise DequantizationError(
            f"output already exists; exactly one derivative is authorized: {output}")
    config, inventory, records, scale_summary = validate_fp8_payload(
        source, config_raw, require_official_geometry=require_official_geometry)
    current_parent = official_native_fp8_precision(
        config, artifact_sha256=parent.artifact_sha256)
    if current_parent.digest != parent.digest:
        raise DequantizationError("parent precision identity differs from the source config")

    groups = plan_output_groups(records, shard_bytes)
    total_files = len(groups)
    if total_files > 99999:
        raise DequantizationError("too many output shards")
    expected_output_bytes = sum(sum(_output_bytes(record) for record in group)
                                for group in groups)
    if max_output_bytes is not None and expected_output_bytes > int(max_output_bytes):
        raise DequantizationError(
            f"derivative would use {expected_output_bytes} > {max_output_bytes} bytes")
    free = shutil.disk_usage(output.parent if output.parent.exists() else source).free
    if free < expected_output_bytes + int(min_free_bytes):
        raise DequantizationError(
            f"insufficient free space: need {expected_output_bytes + min_free_bytes}, have {free}")

    output.mkdir(parents=True, exist_ok=False)
    try:
        for name in asset_names:
            if name == "config.json":
                continue
            source_path = source / name
            if not source_path.is_file():
                raise DequantizationError(f"missing source asset: {name}")
            shutil.copyfile(source_path, output / name)
        new_config = _write_derivative_config(
            config_raw, parent=parent, manifest_semantics=activation_semantics,
            script_sha256=script_sha256)
        (output / "config.json").write_text(
            json.dumps(new_config, indent=2, sort_keys=True) + "\n")
        derivative_config = load_spark2_5_config(
            new_config, require_official_geometry=require_official_geometry)

        output_tensors: list[dict] = []
        output_files: list[dict] = []
        weight_map: dict[str, str] = {}
        from safetensors import safe_open

        handles = {shard.name: safe_open(str(shard), framework="pt")
                   for shard in sorted(source.glob("*.safetensors"))}
        try:
            for index, group in enumerate(groups, start=1):
                filename = f"model-{index:05d}-of-{total_files:05d}.safetensors"
                tensors = {}
                for record in group:
                    handle = handles[record["shard"]]
                    name = record["name"]
                    if record["dtype"] == "F8_E4M3":
                        scale_record = next(item for item in records
                                            if item["name"] == _scale_name(name))
                        weight = handle.get_tensor(name)
                        scale = handles[scale_record["shard"]].get_tensor(_scale_name(name))
                        tensor = dequantize_tensor(weight, scale)
                    elif record["dtype"] == "BF16":
                        tensor = handle.get_tensor(name).contiguous()
                    else:
                        raise DequantizationError(f"{name}: unexpected source dtype")
                    tensors[name] = tensor
                    output_tensors.append({"name": name, "dtype": "BF16",
                                           "shape": list(record["shape"])})
                    weight_map[name] = filename
                path = output / filename
                save_file(tensors, str(path), metadata={"format": "pt"})
                del tensors
                output_files.append({"filename": filename,
                                     "bytes": path.stat().st_size,
                                     "sha256": file_sha256(path)})
        finally:
            handles.clear()

        index_payload = {
            "metadata": {"format": "pt", "total_size": expected_output_bytes},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (output / "model.safetensors.index.json").write_text(
            json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
        artifact_sha256 = artifact_digest(output_files)
        derivative_precision = dequantized_fp8_derivative(
            derivative_config, parent=parent, artifact_sha256=artifact_sha256,
            dequantization_sha256=script_sha256,
            activation_semantics=activation_semantics, storage="bf16")
        verify_tensor_inventory(output_tensors, derivative_config, derivative_precision)
        if derivative_precision.kind != DEQUANTIZED_FP8_DERIVATIVE:
            raise DequantizationError("derivative identity was not admitted")

        manifest = {
            "schema": DERIVATIVE_SCHEMA,
            "derivative_index": 1,
            "source": {
                "role": "official_fp8_adaptation_base",
                "repo_id": None,
                "revision": None,
                "precision_kind": parent.kind,
                "parent_precision": parent.to_dict(),
                "artifact_sha256": parent.artifact_sha256,
                "quantization_fingerprint": parent.quantization_fingerprint,
                "config_sha256": None,
                "scale_provenance": scale_summary,
                "inventory": inventory,
            },
            "derivative": {
                "storage": "bf16",
                "weight_dtype": "BF16",
                "artifact_sha256": artifact_sha256,
                "derivative_of": parent.artifact_sha256,
                "dequantization_sha256": script_sha256,
                "activation_semantics": activation_semantics,
                "equivalent_to_bf16_release": False,
                "config_sha256": file_sha256(output / "config.json"),
                "precision": derivative_precision.to_dict(),
                "precision_digest": derivative_precision.digest,
            },
            "files": output_files,
            "tensor_inventory": {
                "tensor_count": len(output_tensors),
                "parameter_elements": config.geometry.parameter_elements,
                "dtypes": ["BF16"],
            },
            "claims": {
                "official_native_fp8": False,
                "official_bf16_release": False,
                "activation_quantization": False,
                "restores_quantization_loss": False,
                "native_execution": False,
            },
            "retention": "single authorized derivative; keep source snapshots and this manifest",
        }
        manifest["fingerprint"] = fingerprint(manifest)
        (output / "derivative_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {"manifest": manifest, "artifact_sha256": artifact_sha256,
                "output_files": output_files, "output_dir": str(output)}
    except BaseException:
        # Preserve the raw partial output; the fresh-path rule refuses reuse.
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/spark_qualification_v1.json")
    parser.add_argument("--source-role", default="official_fp8_adaptation_base")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-bytes", type=int, default=1024 ** 3)
    args = parser.parse_args(argv)

    plan = load_plan(args.plan, args.source_role)
    entry = plan["sources"][args.source_role]
    derivative_entry = plan["sources"]["dequantized_fp8_derivative"]
    authorized_root = (ROOT / derivative_entry["derivative_root"]).resolve()
    output = args.output.resolve()
    if output != authorized_root:
        parser.error(f"the single derivative must be written to {authorized_root}")
    if output.exists():
        parser.error("the authorized derivative already exists; no second variant is permitted")
    source = args.source_dir.resolve()
    source_files = verify_source_pins(source, entry)
    config_raw = json.loads((source / "config.json").read_text())
    parent = official_native_fp8_precision(
        load_spark2_5_config(config_raw, require_official_geometry=True),
        artifact_sha256=artifact_digest(source_files))
    script_sha256 = file_sha256(Path(__file__).resolve())
    storage = plan["storage_policy"]
    result = dequantize_checkpoint(
        source, output, config_raw=config_raw, parent=parent,
        script_sha256=script_sha256,
        activation_semantics=derivative_entry["activation_semantics"],
        asset_names=derivative_entry.get("required_assets", ()),
        shard_bytes=args.shard_bytes,
        require_official_geometry=True,
        min_free_bytes=int(storage["minimum_free_bytes_before_stage"]),
        max_output_bytes=int(storage["max_derivative_bytes"]))
    manifest = result["manifest"]
    manifest["source"].update({
        "role": args.source_role,
        "repo_id": entry["repo_id"],
        "revision": entry["revision"],
        "config_sha256": entry["config_sha256"],
        "source_files": source_files,
    })
    manifest["fingerprint"] = fingerprint(
        {key: value for key, value in manifest.items() if key != "fingerprint"})
    (output / "derivative_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact_sha256": result["artifact_sha256"],
                      "output": str(output),
                      "files": result["output_files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
