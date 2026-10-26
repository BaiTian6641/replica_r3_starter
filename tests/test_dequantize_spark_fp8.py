"""Tiny synthetic tests for the single Spark FP8 -> BF16 derivative tool."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from replica_lab.architecture.spark2_5 import (
    DEQUANTIZED_FP8_DERIVATIVE, OFFICIAL_NATIVE_FP8,
    dequantized_fp8_derivative, load_spark2_5_config,
    official_bf16_precision, official_native_fp8_precision, sha256_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dequantize_spark_fp8.py"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("dequantize_spark_fp8_test", SCRIPT)
dequant = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(dequant)

LAYERS = ["sliding_attention", "full_attention"]


def tiny_config(*, quantized: bool = True) -> dict:
    raw = {
        "architectures": ["Spark2_5ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_spark.Spark2_5Config",
            "AutoModel": "modeling_spark.Spark2_5Model",
            "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
        },
        "model_type": "spark2_5",
        "dtype": "bfloat16",
        "num_hidden_layers": len(LAYERS),
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "sliding_window": 4,
        "max_position_embeddings": 64,
        "tie_word_embeddings": True,
        "use_cache": True,
        "hidden_act": "gelu",
        "attention_bias": False,
        "mlp_bias": False,
        "headwise_attn_output_gate": True,
        "gate_attn_act_mode": "sigmoid",
        "layer_types": LAYERS,
        "rope_parameters": {
            "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
            "full_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
        },
        "transformers_version": "4.57.6",
    }
    if quantized:
        raw["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "version": "0.16.0",
            "kv_cache_scheme": None,
            "config_groups": {"group_0": {
                "format": "float-quantized",
                "targets": ["re:.*self_attn.q_k_v_proj$", "re:.*self_attn.out_proj$",
                            "re:.*mlp.(gate_proj|up_proj|gate_up_proj|down_proj)$"],
                "weights": {"strategy": "channel", "num_bits": 8,
                            "dynamic": False, "symmetric": True, "type": "float"},
                "input_activations": {"strategy": "token", "num_bits": 8,
                                      "dynamic": True, "symmetric": True, "type": "float"},
            }},
            "ignore": [f"model.layers.{index}.self_attn.g_proj"
                       for index in range(len(LAYERS))] + ["lm_head"],
        }
    return raw


def fp8_pair(shape: tuple[int, ...], seed: int):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(shape, generator=generator) * 0.5
    scale = (values.abs().amax(dim=1, keepdim=True) / 400.0).clamp_min(1e-4)
    quantized = (values / scale).to(torch.float8_e4m3fn)
    return quantized, scale.to(torch.bfloat16)


def build_tensors(seed: int = 3) -> dict:
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        "model.embedding.weight": torch.randn(32, 8, generator=generator).to(torch.bfloat16),
        "model.norm.weight": torch.randn(8, generator=generator).to(torch.bfloat16),
    }
    for index in range(len(LAYERS)):
        prefix = f"model.layers.{index}."
        tensors[prefix + "input_layernorm.weight"] = torch.randn(
            8, generator=generator).to(torch.bfloat16)
        tensors[prefix + "post_attention_layernorm.weight"] = torch.randn(
            8, generator=generator).to(torch.bfloat16)
        for name, shape in (
            ("self_attn.q_k_v_proj.weight", (16, 8)),
            ("self_attn.out_proj.weight", (8, 8)),
            ("mlp.gate_proj.weight", (16, 8)),
            ("mlp.up_proj.weight", (16, 8)),
            ("mlp.down_proj.weight", (8, 16)),
        ):
            quantized, scale = fp8_pair(shape, seed + index)
            tensors[prefix + name] = quantized
            tensors[prefix + name + "_scale"] = scale
        tensors[prefix + "self_attn.g_proj.weight"] = torch.randn(
            2, 8, generator=generator).to(torch.bfloat16)
    return tensors


def make_source(tmp_path: Path, *, mutate=None):
    source = tmp_path / "source"
    source.mkdir(parents=True)
    raw = tiny_config()
    tensors = build_tensors()
    if mutate is not None:
        mutate(tensors)
    save_file(tensors, str(source / "model.safetensors"))
    (source / "config.json").write_text(json.dumps(raw))
    files = [{"filename": "model.safetensors",
              "bytes": (source / "model.safetensors").stat().st_size,
              "sha256": dequant.file_sha256(source / "model.safetensors")}]
    config = load_spark2_5_config(raw, require_official_geometry=False)
    parent = official_native_fp8_precision(
        config, artifact_sha256=dequant.artifact_digest(files))
    return source, raw, config, parent, tensors, files


def run(source, raw, parent, output, *, semantics="BF16 activations; no per-token F8 quantization"):
    return dequant.dequantize_checkpoint(
        source, output, config_raw=raw, parent=parent,
        script_sha256=sha256_bytes(b"test-dequantization-script"),
        activation_semantics=semantics, require_official_geometry=False,
        shard_bytes=512)


def test_dequantizes_one_tensor_with_channel_scales():
    quantized, scale = fp8_pair((4, 8), 11)
    output = dequant.dequantize_tensor(quantized, scale)
    assert output.dtype == torch.bfloat16
    assert tuple(output.shape) == (4, 8)
    expected = (quantized.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    known = torch.tensor([[1.0, -2.0, 0.5, 448.0]]).to(torch.float8_e4m3fn)
    known_scale = torch.tensor([[2.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(
        dequant.dequantize_tensor(known, known_scale),
        torch.tensor([[2.0, -4.0, 1.0, 896.0]], dtype=torch.bfloat16),
        rtol=0, atol=0)

    with pytest.raises(dequant.DequantizationError, match="finite and positive"):
        dequant.dequantize_tensor(known, torch.tensor([[-1.0]], dtype=torch.bfloat16))
    nan = torch.tensor([[float("nan"), 1.0]]).to(torch.float8_e4m3fn)
    with pytest.raises(dequant.DequantizationError, match="NaN/Inf"):
        dequant.dequantize_tensor(nan, torch.ones((1, 1), dtype=torch.bfloat16))


def test_dequantized_artifact_has_new_identity_hashes_and_semantics(tmp_path):
    source, raw, config, parent, tensors, source_files = make_source(tmp_path)
    output = tmp_path / "derivative"
    result = run(source, raw, parent, output)
    manifest = result["manifest"]

    assert manifest["schema"] == dequant.DERIVATIVE_SCHEMA
    assert manifest["derivative_index"] == 1
    assert manifest["fingerprint"] == dequant.fingerprint(
        {key: value for key, value in manifest.items() if key != "fingerprint"})
    assert manifest["source"]["artifact_sha256"] == parent.artifact_sha256
    assert manifest["source"]["precision_kind"] == OFFICIAL_NATIVE_FP8
    assert manifest["derivative"]["artifact_sha256"] != parent.artifact_sha256
    assert manifest["derivative"]["derivative_of"] == parent.artifact_sha256
    assert manifest["derivative"]["artifact_sha256"] == result["artifact_sha256"]
    assert manifest["derivative"]["equivalent_to_bf16_release"] is False
    assert manifest["derivative"]["activation_semantics"] == \
        "BF16 activations; no per-token F8 quantization"
    assert manifest["claims"] == {
        "official_native_fp8": False, "official_bf16_release": False,
        "activation_quantization": False, "restores_quantization_loss": False,
        "native_execution": False,
    }
    provenance = manifest["source"]["scale_provenance"]
    assert provenance["scale_tensor_count"] == 10
    assert provenance["scale_tensor_elements"] == 128
    assert provenance["strategy"] == "channel"
    assert provenance["minimum_scale"] > 0
    assert len(provenance["provenance"]) == 10
    assert {item["scale"] for item in provenance["provenance"]} == {
        name for name in tensors if name.endswith("weight_scale")}

    out_config = json.loads((output / "config.json").read_text())
    assert "quantization_config" not in out_config
    assert out_config["dtype"] == "bfloat16"
    assert out_config["replica_precision"]["kind"] == DEQUANTIZED_FP8_DERIVATIVE
    assert out_config["replica_precision"]["equivalent_to_bf16_release"] is False
    assert dequant.file_sha256(output / "config.json") == \
        manifest["derivative"]["config_sha256"]
    assert dequant.file_sha256(source / "config.json") != \
        manifest["derivative"]["config_sha256"]

    verified = []
    for item in manifest["files"]:
        path = output / item["filename"]
        assert path.is_file()
        assert path.stat().st_size == item["bytes"]
        digest = dequant.file_sha256(path)
        assert digest == item["sha256"]
        verified.append({"filename": item["filename"], "sha256": digest})
    assert dequant.artifact_digest(verified) == manifest["derivative"]["artifact_sha256"]
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {
        name for name in tensors if not name.endswith("weight_scale")}
    assert manifest["tensor_inventory"]["tensor_count"] == len(index["weight_map"])

    output_config = load_spark2_5_config(out_config, require_official_geometry=False)
    identity = dequantized_fp8_derivative(
        output_config, parent=parent,
        artifact_sha256=manifest["derivative"]["artifact_sha256"],
        dequantization_sha256=manifest["derivative"]["dequantization_sha256"],
        activation_semantics=manifest["derivative"]["activation_semantics"],
        storage="bf16")
    assert identity.kind == DEQUANTIZED_FP8_DERIVATIVE
    assert identity.digest == manifest["derivative"]["precision_digest"]
    assert identity.derivative_of == parent.artifact_sha256
    assert identity.activation_quantized is False

    quantized_name = "model.layers.0.self_attn.q_k_v_proj.weight"
    expected = (tensors[quantized_name].to(torch.float32) *
                tensors[quantized_name + "_scale"].to(torch.float32)).to(torch.bfloat16)
    located = None
    for item in manifest["files"]:
        with safe_open(str(output / item["filename"]), framework="pt") as handle:
            if quantized_name in handle.keys():
                located = handle.get_tensor(quantized_name)
                assert str(handle.get_slice(quantized_name).get_dtype()) == "BF16"
    assert located is not None
    torch.testing.assert_close(located, expected, rtol=0, atol=0)

    with pytest.raises(dequant.DequantizationError, match="already exists"):
        run(source, raw, parent, output)


def test_missing_and_invalid_scales_are_refused(tmp_path):
    source, raw, _, parent, _, _ = make_source(
        tmp_path / "missing", mutate=lambda values: values.pop(
            "model.layers.0.self_attn.q_k_v_proj.weight_scale"))
    with pytest.raises(dequant.DequantizationError, match="inventory|missing per-channel"):
        run(source, raw, parent, tmp_path / "missing_out")

    source, raw, _, parent, _, _ = make_source(
        tmp_path / "negative",
        mutate=lambda values: values.__setitem__(
            "model.layers.0.self_attn.out_proj.weight_scale",
            -torch.ones(8, 1, dtype=torch.bfloat16)))
    with pytest.raises(dequant.DequantizationError, match="non-positive scale"):
        run(source, raw, parent, tmp_path / "negative_out")

    source, raw, _, parent, _, _ = make_source(
        tmp_path / "shape",
        mutate=lambda values: values.__setitem__(
            "model.layers.0.self_attn.out_proj.weight_scale",
            torch.ones(1, 1, dtype=torch.bfloat16)))
    with pytest.raises(dequant.DequantizationError, match=r"shape|\[out, 1\]"):
        run(source, raw, parent, tmp_path / "shape_out")


def test_wrong_parent_and_payload_pins_are_refused(tmp_path):
    source, raw, config, parent, _, _ = make_source(tmp_path / "source")
    bf16_config = load_spark2_5_config(
        tiny_config(quantized=False), require_official_geometry=False)
    bf16_parent = official_bf16_precision(
        bf16_config, artifact_sha256=parent.artifact_sha256)
    with pytest.raises(dequant.DequantizationError, match="parent must be an official"):
        run(source, raw, bf16_parent, tmp_path / "out")

    missing = tmp_path / "absent"
    with pytest.raises(dequant.DequantizationError, match="missing_official_payload"):
        dequant.verify_source_pins(missing, {
            "shards": [{"filename": "model.safetensors", "bytes": 1, "sha256": "a" * 64}],
            "config_sha256": "b" * 64, "required_assets": ["config.json"]})

    plan = json.loads((ROOT / "configs/spark_qualification_v1.json").read_text())
    entry = plan["sources"]["official_fp8_adaptation_base"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(dequant.DequantizationError, match="shard inventory mismatch"):
        dequant.verify_source_pins(empty, entry)


def test_artifact_digest_is_deterministic_and_hash_bound():
    first = dequant.artifact_digest(
        [{"filename": "b.safetensors", "sha256": "b" * 64},
         {"filename": "a.safetensors", "sha256": "a" * 64}])
    second = dequant.artifact_digest(
        [{"filename": "a.safetensors", "sha256": "a" * 64},
         {"filename": "b.safetensors", "sha256": "b" * 64}])
    assert first == second
    changed = dequant.artifact_digest(
        [{"filename": "a.safetensors", "sha256": "c" * 64},
         {"filename": "b.safetensors", "sha256": "b" * 64}])
    assert changed != first
    with pytest.raises(dequant.DequantizationError):
        dequant.artifact_digest([])
