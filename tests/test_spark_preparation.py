"""Offline tests for bounded Spark source preparation.

No test contacts Hugging Face, imports a tensor library or executes remote code.
Synthetic safetensors headers exercise the real parser and inventory checks.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from replica_lab.spark_preparation import (
    HEADER_BUNDLE_SCHEMA, PLAN_SCHEMA, SCHEMA, BoundedHttpTransport,
    FetchRefused, PlanError, PreparationError, audit_python_source,
    load_header_bundle, load_manifest, parse_safetensors_header,
    run_preparation, validate_plan, validate_spark_inventory,
    _validate_index,
)

OFFICIAL_TOPOLOGY = {
    "num_hidden_layers": 36, "hidden_size": 2560, "intermediate_size": 10240,
    "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
    "vocab_size": 131072, "sliding_window": 512, "tie_word_embeddings": True,
    "layer_pattern": ["sliding_attention"] * 3 + ["full_attention"] +
                     (["sliding_attention"] * 3 + ["full_attention"]) * 8,
}

TINY_TOPOLOGY = {
    "num_hidden_layers": 2, "hidden_size": 8, "intermediate_size": 16,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
    "vocab_size": 32, "sliding_window": 4, "tie_word_embeddings": True,
    "layer_pattern": ["sliding_attention", "full_attention"],
}
SIZES = {"BF16": 2, "F8_E4M3": 1}


# ---------------------------------------------------------------------------
# Synthetic headers


def tensor_specs(storage, topology, *, drop=(), extra=()):
    hidden = topology["hidden_size"]
    inter = topology["intermediate_size"]
    heads = topology["num_attention_heads"]
    kv = topology["num_key_value_heads"]
    head_dim = topology["head_dim"]
    qkv_out = (heads + 2 * kv) * head_dim
    q_dim = heads * head_dim
    specs = [("model.embedding.weight", "BF16", [topology["vocab_size"], hidden])]
    for index in range(topology["num_hidden_layers"]):
        prefix = f"model.layers.{index}."
        specs += [
            (prefix + "input_layernorm.weight", "BF16", [hidden]),
            (prefix + "post_attention_layernorm.weight", "BF16", [hidden]),
            (prefix + "self_attn.q_k_v_proj.weight",
             "F8_E4M3" if storage == "fp8_compressed_tensors" else "BF16",
             [qkv_out, hidden]),
            (prefix + "self_attn.out_proj.weight",
             "F8_E4M3" if storage == "fp8_compressed_tensors" else "BF16",
             [hidden, q_dim]),
            (prefix + "self_attn.g_proj.weight", "BF16", [heads, hidden]),
            (prefix + "mlp.gate_proj.weight",
             "F8_E4M3" if storage == "fp8_compressed_tensors" else "BF16",
             [inter, hidden]),
            (prefix + "mlp.up_proj.weight",
             "F8_E4M3" if storage == "fp8_compressed_tensors" else "BF16",
             [inter, hidden]),
            (prefix + "mlp.down_proj.weight",
             "F8_E4M3" if storage == "fp8_compressed_tensors" else "BF16",
             [hidden, inter]),
        ]
        if storage == "fp8_compressed_tensors":
            specs += [
                (prefix + "self_attn.q_k_v_proj.weight_scale", "BF16", [qkv_out, 1]),
                (prefix + "self_attn.out_proj.weight_scale", "BF16", [hidden, 1]),
                (prefix + "mlp.gate_proj.weight_scale", "BF16", [inter, 1]),
                (prefix + "mlp.up_proj.weight_scale", "BF16", [inter, 1]),
                (prefix + "mlp.down_proj.weight_scale", "BF16", [hidden, 1]),
            ]
    specs += [("model.norm.weight", "BF16", [hidden])]
    specs += list(extra)
    return [spec for spec in specs if spec[0] not in set(drop)]


def build_shard(storage, topology, *, drop=(), extra=(), mutate=None):
    specs = tensor_specs(storage, topology, drop=drop, extra=extra)
    tensors = {}
    offset = 0
    for name, dtype, shape in specs:
        count = math.prod(shape) if shape else 1
        size = count * SIZES[dtype]
        tensors[name] = {"dtype": dtype, "shape": shape,
                         "data_offsets": [offset, offset + size]}
        offset += size
    if mutate is not None:
        tensors = mutate(dict(tensors))
    raw = json.dumps({"__metadata__": {"format": "pt"}, **tensors},
                     separators=(",", ":")).encode()
    return raw, 8 + len(raw) + offset


def parse_shard(raw, total, filename="model.safetensors", max_header=1024 * 1024):
    return parse_safetensors_header(raw, filename=filename, total_bytes=total,
                                    max_header_bytes=max_header)


# ---------------------------------------------------------------------------
# Fake HTTP


class FakeResponse:
    def __init__(self, status, headers, body=b"", url="https://example.test/asset"):
        self.status = status
        self.headers = dict(headers)
        self._body = body
        self._position = 0
        self.url = url
        self.read_calls = 0
        self.closed = False

    def read(self, size=-1):
        self.read_calls += 1
        if size < 0:
            chunk = self._body[self._position:]
        else:
            chunk = self._body[self._position:self._position + size]
        self._position += len(chunk)
        return chunk

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url

    def close(self):
        self.closed = True


class Opener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        return self.responses.pop(0)


def range_headers(start, end, total):
    return {"Content-Range": f"bytes {start}-{end}/{total}",
            "Content-Length": str(end - start + 1),
            "Content-Encoding": "identity", "ETag": '"fixture"'}


def test_range_accepts_exact_partial_content_and_records_evidence():
    opener = Opener([FakeResponse(206, range_headers(0, 7, 1000), b"abcdefgh")])
    transport = BoundedHttpTransport(total_limit=100, opener=opener)
    result = transport.range("https://example.test/model.safetensors", 0, 7,
                             expected_total=1000, max_bytes=8)
    assert result["data"] == b"abcdefgh"
    assert result["bytes_read"] == 8
    assert result["content_range"] == "bytes 0-7/1000"
    assert result["status"] == 206
    assert transport.spent == 8
    assert opener.requests[0].get_header("Range") == "bytes=0-7"
    assert opener.requests[0].get_header("Accept-encoding") == "identity"


def test_range_refuses_http_200_without_reading_the_body():
    response = FakeResponse(200, {"Content-Length": "4451626120"}, b"full weight body")
    transport = BoundedHttpTransport(total_limit=1024, opener=Opener([response]))
    with pytest.raises(FetchRefused) as error:
        transport.range("https://example.test/model.safetensors", 0, 7,
                        expected_total=4451626120, max_bytes=8)
    assert error.value.reason == "server_ignored_range"
    assert error.value.status == 200
    assert error.value.bytes_read == 0
    assert response.read_calls == 0
    assert response.closed is True
    assert transport.spent == 0


@pytest.mark.parametrize("headers,reason", [
    ({"Content-Range": "bytes 1-8/1000", "Content-Length": "8"}, "content_range_mismatch"),
    ({"Content-Range": "bytes 0-7/999", "Content-Length": "8"}, "total_size_mismatch"),
    ({"Content-Range": "bytes 0-7/1000", "Content-Length": "7"}, "content_length_mismatch"),
    ({"Content-Range": "*", "Content-Length": "8"}, "invalid_content_range"),
    ({"Content-Range": "bytes 0-7/1000"}, "missing_content_length"),
])
def test_range_refuses_inconsistent_headers(headers, reason):
    transport = BoundedHttpTransport(total_limit=1024,
                                     opener=Opener([FakeResponse(206, headers, b"abcdefgh")]))
    with pytest.raises(FetchRefused) as error:
        transport.range("https://example.test/x", 0, 7,
                        expected_total=1000, max_bytes=8)
    assert error.value.reason == reason
    assert error.value.bytes_read == 0


def test_range_refuses_short_or_overlong_body():
    transport = BoundedHttpTransport(
        total_limit=1024, opener=Opener([FakeResponse(206, range_headers(0, 7, 1000), b"abc")]))
    with pytest.raises(FetchRefused, match="short_range_body"):
        transport.range("https://example.test/x", 0, 7,
                        expected_total=1000, max_bytes=8)
    transport = BoundedHttpTransport(
        total_limit=1024,
        opener=Opener([FakeResponse(206, range_headers(0, 7, 1000), b"abcdefghij")]))
    with pytest.raises(FetchRefused, match="body_exceeds_bound"):
        transport.range("https://example.test/x", 0, 7,
                        expected_total=1000, max_bytes=8)


def test_range_refuses_before_requesting_an_oversized_header():
    transport = BoundedHttpTransport(total_limit=1024, opener=Opener([]))
    with pytest.raises(FetchRefused, match="range_exceeds_bound"):
        transport.range("https://example.test/x", 0, 99,
                        expected_total=1000, max_bytes=16)
    assert transport.requests == 0


def test_total_budget_is_enforced_across_requests():
    responses = [FakeResponse(206, range_headers(0, 7, 1000), b"abcdefgh"),
                 FakeResponse(206, range_headers(8, 15, 1000), b"ijklmnop")]
    transport = BoundedHttpTransport(total_limit=8, opener=Opener(responses))
    transport.range("https://example.test/x", 0, 7, expected_total=1000, max_bytes=8)
    with pytest.raises(FetchRefused, match="total_fetch_budget"):
        transport.range("https://example.test/x", 8, 15, expected_total=1000, max_bytes=8)
    assert transport.spent == 8


def test_full_fetch_refuses_content_length_over_bound_without_reading():
    response = FakeResponse(200, {"Content-Length": "99"}, b"x" * 99)
    transport = BoundedHttpTransport(total_limit=1024, opener=Opener([response]))
    with pytest.raises(FetchRefused, match="asset_exceeds_bound"):
        transport.get("https://example.test/config.json", max_bytes=10)
    assert response.read_calls == 0


# ---------------------------------------------------------------------------
# Safetensors inventory


def test_parse_header_records_dtypes_scales_and_tied_storage():
    raw, total = build_shard("fp8_compressed_tensors", TINY_TOPOLOGY)
    header = parse_shard(raw, total)
    assert header["tensor_count"] == 28
    assert header["dtype_totals"]["F8_E4M3"]["tensors"] == 10
    assert header["dtype_totals"]["BF16"]["tensors"] == 18
    inventory = validate_spark_inventory([header], topology=TINY_TOPOLOGY,
                                         storage="fp8_compressed_tensors")
    assert inventory["layer_count"] == 2
    assert inventory["scale_tensor_count"] == 10
    assert inventory["zero_point_tensor_count"] == 0
    assert inventory["tied_storage"]["lm_head_serialized"] is False
    assert inventory["parameter_elements_excluding_scales"] == 1480
    assert inventory["parameter_elements_including_scales"] == 1608
    assert inventory["file_bytes"] == total
    assert inventory["payload_bytes"] == header["payload_bytes"]
    assert inventory["quantized_suffixes"] == [
        "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "self_attn.out_proj.weight", "self_attn.q_k_v_proj.weight"]


def test_bf16_inventory_has_no_scales_and_matches_formula():
    raw, total = build_shard("bf16", TINY_TOPOLOGY)
    header = parse_shard(raw, total)
    inventory = validate_spark_inventory([header], topology=TINY_TOPOLOGY, storage="bf16")
    assert inventory["dtype_totals"]["BF16"]["tensors"] == 18
    assert inventory["scale_tensor_count"] == 0
    assert inventory["parameter_elements_excluding_scales"] == 1480
    assert inventory["parameter_elements_including_scales"] == 1480


def test_inventory_rejects_missing_scale_and_wrong_scale_shape():
    raw, total = build_shard(
        "fp8_compressed_tensors", TINY_TOPOLOGY,
        drop=("model.layers.0.mlp.gate_proj.weight_scale",))
    with pytest.raises(PreparationError, match="missing expected tensors"):
        validate_spark_inventory([parse_shard(raw, total)], topology=TINY_TOPOLOGY,
                                 storage="fp8_compressed_tensors")

    # Rebuild offsets after the shape change rather than lying about the bytes.
    specs = tensor_specs("fp8_compressed_tensors", TINY_TOPOLOGY)
    specs = [(name, dtype, [1, 1] if name.endswith("mlp.up_proj.weight_scale") else shape)
             for name, dtype, shape in specs]
    tensors, offset = {}, 0
    for name, dtype, shape in specs:
        size = math.prod(shape) * SIZES[dtype]
        tensors[name] = {"dtype": dtype, "shape": shape,
                         "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps({"__metadata__": {"format": "pt"}, **tensors},
                     separators=(",", ":")).encode()
    with pytest.raises(PreparationError, match="must be BF16"):
        validate_spark_inventory([parse_shard(raw, 8 + len(raw) + offset)],
                                 topology=TINY_TOPOLOGY,
                                 storage="fp8_compressed_tensors")


def test_inventory_rejects_serialized_lm_head_extra_tensor():
    raw, total = build_shard("bf16", TINY_TOPOLOGY,
                             extra=(("model.lm_head.weight", "BF16", [32, 8]),))
    with pytest.raises(PreparationError, match="unexpected tensors"):
        validate_spark_inventory([parse_shard(raw, total)], topology=TINY_TOPOLOGY,
                                 storage="bf16")


def test_header_parser_rejects_overlap_gap_size_and_duplicate_keys():
    overlapping = {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                   "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]}}
    raw = json.dumps(overlapping, separators=(",", ":")).encode()
    with pytest.raises(PreparationError, match="overlaps"):
        parse_shard(raw, 8 + len(raw) + 3)

    gap = {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
           "b": {"dtype": "U8", "shape": [2], "data_offsets": [4, 6]}}
    raw = json.dumps(gap, separators=(",", ":")).encode()
    with pytest.raises(PreparationError, match="unclaimed byte gap"):
        parse_shard(raw, 8 + len(raw) + 6)

    wrong = {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 1]}}
    raw = json.dumps(wrong, separators=(",", ":")).encode()
    with pytest.raises(PreparationError, match="expected 2"):
        parse_shard(raw, 8 + len(raw) + 1)

    duplicate = b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},' \
                b'"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}'
    with pytest.raises(PreparationError, match="duplicate JSON key"):
        parse_shard(duplicate, 8 + len(duplicate) + 2)


def test_header_parser_rejects_oversized_or_unbounded_header():
    raw, total = build_shard("bf16", TINY_TOPOLOGY)
    with pytest.raises(PreparationError, match="out of bounds"):
        parse_safetensors_header(raw, filename="model.safetensors", total_bytes=total,
                                 max_header_bytes=16)
    with pytest.raises(PreparationError, match="larger than the shard"):
        parse_safetensors_header(raw, filename="model.safetensors", total_bytes=8,
                                 max_header_bytes=1024 * 1024)


def test_index_must_match_headers_and_declared_totals():
    raw, total = build_shard("bf16", TINY_TOPOLOGY)
    header = parse_shard(raw, total)
    inventory = validate_spark_inventory([header], topology=TINY_TOPOLOGY, storage="bf16")
    mapping = inventory["tensor_shard_map"]
    index = {"metadata": {"total_parameters": inventory["parameter_elements_excluding_scales"],
                          "total_size": inventory["payload_bytes"]},
             "weight_map": mapping}
    summary = _validate_index(json.dumps(index).encode(), filename="model.safetensors.index.json",
                              headers=[header], inventory=inventory)
    assert summary["tensor_count"] == len(mapping)
    broken = dict(mapping)
    broken[next(iter(broken))] = "other.safetensors"
    with pytest.raises(PreparationError, match="index places"):
        _validate_index(json.dumps({"metadata": index["metadata"], "weight_map": broken}).encode(),
                        filename="model.safetensors.index.json",
                        headers=[header], inventory=inventory)
    index["metadata"]["total_parameters"] += 1
    with pytest.raises(PreparationError, match="total_parameters"):
        _validate_index(json.dumps(index).encode(), filename="model.safetensors.index.json",
                        headers=[header], inventory=inventory)


# ---------------------------------------------------------------------------
# Static review


def test_python_audit_parses_without_executing_and_flags_risky_calls(tmp_path):
    marker = tmp_path / "must_not_exist"
    source = ("import subprocess\n"
              "class Fixture:\n"
              "    def forward(self, value):\n"
              f"        open({str(marker)!r}, 'w').write('executed')\n"
              "        return value\n")
    audit = audit_python_source(source, "remote.py")
    assert audit["executed"] is False
    assert "subprocess" in audit["imports"]
    assert audit["imports_process_or_network"] == ["subprocess"]
    assert audit["module_executes_code_on_import"] is False
    assert {"open", "system", "popen"} & set(audit["risky_symbols"])
    assert marker.exists() is False


# ---------------------------------------------------------------------------
# Plan, local headers and a full offline run


def _write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _config(storage):
    config = {
        "architectures": ["Spark2_5ForCausalLM"],
        "model_type": "spark2_5",
        "num_hidden_layers": 36,
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 131072,
        "sliding_window": 512,
        "tie_word_embeddings": True,
        "use_cache": True,
        "hidden_act": "gelu",
        "attention_bias": False,
        "mlp_bias": False,
        "headwise_attn_output_gate": True,
        "gate_attn_act_mode": "sigmoid",
        "layer_types": OFFICIAL_TOPOLOGY["layer_pattern"],
        "rope_parameters": {
            "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
        },
        "auto_map": {
            "AutoConfig": "configuration_spark.Spark2_5Config",
            "AutoModel": "modeling_spark.Spark2_5Model",
            "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
        },
        "max_position_embeddings": 1048576,
        "transformers_version": "4.57.6",
    }
    if storage == "fp8_compressed_tensors":
        targets = ["re:.*self_attn.q_k_v_proj$", "re:.*self_attn.out_proj$",
                   "re:.*mlp.(gate_proj|up_proj|gate_up_proj|down_proj)$"]
        config["quantization_config"] = {
            "quant_method": "compressed-tensors", "format": "float-quantized",
            "version": "0.16.0", "kv_cache_scheme": None,
            "config_groups": {"group_0": {
                "format": "float-quantized", "targets": targets,
                "weights": {"strategy": "channel", "num_bits": 8, "dynamic": False,
                            "symmetric": True, "observer": "mse"},
                "input_activations": {"strategy": "token", "num_bits": 8,
                                      "dynamic": True, "symmetric": True},
            }},
            "ignore": [f"model.layers.{index}.self_attn.g_proj" for index in range(36)]
                      + ["lm_head"],
        }
    return config


TEMPLATE = (b"{%- if not messages %}{{ raise_exception('No messages provided.') }}{%- endif %}"
            b"{%- set enable_thinking = enable_thinking | default(true) %}"
            b"{%- set reasoning_effort = reasoning_effort | default('high') %}"
            b"{%- if reasoning_effort not in ('high', 'low') %}{{ raise_exception('effort') }}"
            b"{%- endif %}{{ '<think>' }}{{ '</think>' }}"
            b"{%- if add_generation_prompt %}{{ '<think>' }}{%- endif %}")


def asset_files(storage):
    return {
        "config.json": json.dumps(_config(storage)).encode(),
        "LICENSE": b"Apache License 2.0\n",
        "README.md": b"fixture card\n",
        "chat_template.jinja": TEMPLATE,
        "generation_config.json": json.dumps({
            "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
            "max_tokens": 1048576, "temperature": 1.0, "top_p": 0.95,
            "top_k": -1, "do_sample": True}).encode(),
        "tokenizer_config.json": json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 131072, "add_bos_token": False,
            "add_eos_token": False, "chat_template": TEMPLATE.decode()}).encode(),
        "configuration_spark.py": b"class Spark2_5Config:\n    pass\n",
        "modeling_spark.py": (
            b"from .configuration_spark import Spark2_5Config\n"
            b"REGISTERED = []\nREGISTERED.append(Spark2_5Config)\n"),
    }


def source_plan(role, storage, weight_bytes, lfs, assets, quantization=None):
    return {
        "role": role, "repo_id": f"XHToken/{role}", "revision": lfs[0] * 40,
        "storage": storage, "license_claim": "apache-2.0",
        "weight_files": [{"filename": "model.safetensors",
                          "bytes": weight_bytes, "hub_lfs_sha256": lfs * 64}],
        "weight_index": None, "quantization": quantization,
        "assets": [{"filename": name, "required": True, "retain": True,
                    "expected_bytes": len(data),
                    "expected_sha256": hashlib.sha256(data).hexdigest()}
                   for name, data in assets.items()],
    }


def make_plan():
    fp8_raw, fp8_total = build_shard("fp8_compressed_tensors", OFFICIAL_TOPOLOGY)
    bf16_raw, bf16_total = build_shard("bf16", OFFICIAL_TOPOLOGY)
    quantization = {
        "method": "compressed-tensors", "format": "float-quantized",
        "version": "0.16.0",
        "weight_strategy": "channel", "weight_symmetric": True,
        "weight_bits": 8, "weight_dtype": "F8_E4M3",
        "activation_strategy": "token", "activation_dynamic": True,
        "activation_symmetric": True, "activation_bits": 8,
        "activation_dtype": "F8_E4M3", "scale_dtype": "BF16",
        "zero_points": False, "kv_cache_quantized": False,
        "quantized_suffixes": ["self_attn.q_k_v_proj.weight",
                               "self_attn.out_proj.weight", "mlp.gate_proj.weight",
                               "mlp.up_proj.weight", "mlp.down_proj.weight"],
        "unquantized_suffixes_required": ["self_attn.g_proj.weight"],
        "dequantization_equivalent_to_bf16_release": False,
    }
    plan = {
        "schema": PLAN_SCHEMA, "reviewed_on": "2026-10-25", "scope": "offline fixture",
        "weights_downloaded": False, "remote_code_execution": False,
        "topology": OFFICIAL_TOPOLOGY,
        "limits": {"max_header_bytes": 1024 * 1024, "max_asset_bytes": 1024 * 1024,
                   "max_total_fetch_bytes": 16 * 1024 * 1024,
                   "max_api_bytes": 1024 * 1024, "min_free_bytes": 1024,
                   "timeout_seconds": 10},
        "sources": [
            source_plan("official_fp8_adaptation_base", "fp8_compressed_tensors",
                        fp8_total, "a", asset_files("fp8_compressed_tensors"),
                        quantization),
            source_plan("official_bf16_control", "bf16", bf16_total, "b",
                        asset_files("bf16")),
        ],
        "environment": {"isolation": "fixture"},
    }
    return plan, {"official_fp8_adaptation_base": [
                      ("model.safetensors", fp8_raw, fp8_total)],
                  "official_bf16_control": [
                      ("model.safetensors", bf16_raw, bf16_total)]}


def write_capture(root: Path, plan, headers):
    for source in plan["sources"]:
        for asset in source["assets"]:
            _write(root / "assets" / source["role"] / asset["filename"],
                   asset_files(source["storage"])[asset["filename"]])
    bundle = {"schema": HEADER_BUNDLE_SCHEMA, "sources": []}
    for source in plan["sources"]:
        shards = []
        for filename, raw, total in headers[source["role"]]:
            relative = f"{source['role']}/{filename}.header"
            _write(root / "bundle" / relative, raw)
            shards.append({"filename": filename, "total_bytes": total,
                           "source_url": f"https://huggingface.co/{source['repo_id']}/"
                                         f"resolve/{source['revision']}/{filename}",
                           "header_file": relative,
                           "header_sha256": hashlib.sha256(raw).hexdigest(),
                           "header_bytes": len(raw)})
        bundle["sources"].append({"role": source["role"], "repo_id": source["repo_id"],
                                  "revision": source["revision"], "shards": shards})
    bundle_path = root / "bundle" / "headers.json"
    _write(bundle_path, json.dumps(bundle, indent=2).encode())
    return bundle_path


def test_plan_schema_rejects_aliases_unpinned_or_weight_assets():
    plan, _ = make_plan()
    validate_plan(plan)
    broken = json.loads(json.dumps(plan))
    broken["sources"][1]["revision"] = broken["sources"][0]["revision"]
    with pytest.raises(PlanError, match="distinct immutable revisions"):
        validate_plan(broken)
    broken = json.loads(json.dumps(plan))
    broken["sources"][0]["repo_id"] = broken["sources"][1]["repo_id"]
    with pytest.raises(PlanError, match="independent repository"):
        validate_plan(broken)
    broken = json.loads(json.dumps(plan))
    broken["sources"][0]["assets"].append({"filename": "model.safetensors",
                                           "required": True, "retain": False})
    with pytest.raises(PlanError, match="weight payloads"):
        validate_plan(broken)
    broken = json.loads(json.dumps(plan))
    broken["sources"][0]["revision"] = "main"
    with pytest.raises(PlanError, match="40-character commit"):
        validate_plan(broken)
    broken = json.loads(json.dumps(plan))
    broken["limits"]["max_header_bytes"] = 64 * 1024 * 1024
    with pytest.raises(PlanError, match="hard ceiling"):
        validate_plan(broken)


def test_offline_run_is_fresh_bounded_and_writes_provenance(tmp_path):
    plan, headers = make_plan()
    bundle = write_capture(tmp_path / "capture", plan, headers)
    output = tmp_path / "run_01"
    manifest_path = run_preparation(
        plan, output, header_bundle=bundle,
        assets_from=tmp_path / "capture" / "assets", offline=True,
        free_bytes=10**12, now="2026-10-25T00:00:00+00:00")
    manifest = load_manifest(manifest_path)
    assert manifest["schema"] == SCHEMA
    assert manifest["status"] == "preparation_only_unqualified"
    assert manifest["fetch_policy"]["weights_downloaded"] is False
    assert manifest["fetch_policy"]["remote_code_executed"] is False
    assert manifest["fetch_policy"]["bytes_fetched"] == 0
    assert manifest["fetch_policy"]["offline"] is True
    assert manifest["identity"]["independent_repositories"] is True
    assert manifest["identity"]["distinct_weight_digests"] is True
    assert len(manifest["provenance"]["records"]) == 18
    assert all(record["network"] is False for record in manifest["provenance"]["records"])
    assert {source["role"] for source in manifest["sources"]} == {
        "official_fp8_adaptation_base", "official_bf16_control"}
    fp8 = [source for source in manifest["sources"]
           if source["role"] == "official_fp8_adaptation_base"][0]
    assert fp8["tensor_inventory"]["layer_count"] == 36
    assert fp8["tensor_inventory"]["tensor_count"] == 470
    assert fp8["tensor_inventory"]["scale_tensor_count"] == 180
    assert fp8["tensor_inventory"]["parameter_elements_excluding_scales"] == 4112079360
    assert fp8["tensor_inventory"]["parameter_elements_including_scales"] == 4113222144
    assert fp8["tensor_inventory"]["zero_point_tensor_count"] == 0
    assert fp8["config_observed"]["quantization"]["weights"]["strategy"] == "channel"
    assert fp8["config_observed"]["quantization"]["input_activations"]["strategy"] == "token"
    assert fp8["tokenizer_observed"]["chat_template"]["supports_enable_thinking"] is True
    assert (output / "headers" / "official_fp8_adaptation_base" / "tensor_index.json").is_file()
    assert not any(path.suffix == ".safetensors" for path in output.rglob("*"))
    matrix = {row["capability"]: row for row in manifest["capability_matrix"]}
    assert matrix["forward_inference"]["measured"] is False
    assert "Unqualified" in matrix["forward_inference"]["official_fp8"]["status"]
    assert manifest["adapter_abi"]["status"] == "design_only_not_implemented"
    assert manifest["precision_semantics"]["fp8_checkpoint_storage"]["activation"].startswith(
        "dynamic symmetric per-token")
    assert manifest["environment"]["isolation_required"] is True
    assert manifest["unresolved"]
    tampered = json.loads(manifest_path.read_text())
    tampered["scope"] = "tampered"
    tampered_path = tmp_path / "tampered_manifest.json"
    tampered_path.write_text(json.dumps(tampered))
    with pytest.raises(PreparationError, match="fingerprint"):
        load_manifest(tampered_path)

    with pytest.raises(FileExistsError):
        run_preparation(plan, output, header_bundle=bundle,
                        assets_from=tmp_path / "capture" / "assets", offline=True,
                        free_bytes=10**12)


def test_offline_run_refuses_before_creating_output_without_disk(tmp_path):
    plan, headers = make_plan()
    bundle = write_capture(tmp_path / "capture", plan, headers)
    output = tmp_path / "run_01"
    with pytest.raises(PreparationError, match="insufficient free disk"):
        run_preparation(plan, output, header_bundle=bundle,
                        assets_from=tmp_path / "capture" / "assets", offline=True,
                        free_bytes=1024)
    assert not output.exists()


def test_local_header_bundle_rejects_tampered_sidecar(tmp_path):
    plan, headers = make_plan()
    bundle_path = write_capture(tmp_path / "capture", plan, headers)
    loaded = load_header_bundle(bundle_path, validate_plan(plan), max_header_bytes=1024 * 1024)
    assert len(loaded["official_fp8_adaptation_base"]) == 1
    bundle = json.loads(bundle_path.read_text())
    sidecar = bundle_path.parent / bundle["sources"][0]["shards"][0]["header_file"]
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(PreparationError, match="local header SHA256 differs"):
        load_header_bundle(bundle_path, validate_plan(plan), max_header_bytes=1024 * 1024)
