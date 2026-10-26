"""Tiny synthetic tests for the frozen-base Spark2_5 adapter and runner contract.

No Spark weights, tokenizer, remote code or native MPS job is used. The base
model is a tiny BF16 Spark-shaped stub with the audited module names.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from replica_lab.architecture.spark2_5 import (
    BackendAdmissionError, ExtensionStateError, GeometryError,
    OFFICIAL_PARAMETER_ELEMENTS, admit_load, dequantized_fp8_derivative,
    derive_layer_schedule, fingerprint,
    load_spark2_5_config, official_bf16_precision, official_native_fp8_precision,
    sha256_bytes,
)
from replica_lab.architecture.spark_adapter import (
    Spark2_5Adapter, SparkAdapterError,
)
from replica_lab.optimizers import NativeFusedAdamW, build_optimizer

ROOT = Path(__file__).resolve().parents[1]
DEQUANTIZER = ROOT / "scripts" / "dequantize_spark_fp8.py"
SCRIPT = ROOT / "scripts" / "qualify_spark2_5.py"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("qualify_spark2_5_test", SCRIPT)
qualification = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(qualification)

TINY_LAYERS = ["sliding_attention", "sliding_attention", "sliding_attention",
               "full_attention"] * 2


def tiny_config_raw(*, quantized: bool = False, layers: list[str] | None = None) -> dict:
    layer_types = list(layers or TINY_LAYERS)
    raw = {
        "architectures": ["Spark2_5ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_spark.Spark2_5Config",
            "AutoModel": "modeling_spark.Spark2_5Model",
            "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
        },
        "model_type": "spark2_5",
        "dtype": "bfloat16",
        "num_hidden_layers": len(layer_types),
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
        "layer_types": layer_types,
        "rope_parameters": {
            "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
            "full_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
        },
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "transformers_version": "4.57.1",
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
                       for index in range(len(layer_types))] + ["lm_head"],
        }
    return raw


def official_config_raw() -> dict:
    types = (["sliding_attention"] * 3 + ["full_attention"]) * 9
    raw = tiny_config_raw(layers=types)
    raw.update({
        "hidden_size": 2560, "intermediate_size": 10240,
        "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
        "vocab_size": 131072, "sliding_window": 512,
        "max_position_embeddings": 1048576,
    })
    return raw


def official_quant_config_raw() -> dict:
    raw = official_config_raw()
    raw["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "float-quantized",
        "version": "0.16.0", "kv_cache_scheme": None,
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
                   for index in range(36)] + ["lm_head"],
    }
    return raw


class TinyBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
        self.linear = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, cache_position=None, position_ids=None,
                **kwargs):
        return hidden_states + 0.2 * self.linear(self.norm(hidden_states))


class TinySparkModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        geometry = config.geometry
        self.embedding = nn.Embedding(geometry.vocab_size, geometry.hidden_size,
                                      padding_idx=2, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([TinyBlock(geometry.hidden_size)
                                     for _ in range(geometry.num_hidden_layers)])
        self.norm = nn.LayerNorm(geometry.hidden_size, dtype=torch.bfloat16)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                use_cache=False, **kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask,
                           position_ids=position_ids, use_cache=use_cache)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class TinySparkCausalLM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        geometry = config.geometry
        self.config = SimpleNamespace(
            model_type="spark2_5", use_cache=True, _attn_implementation="eager",
            hidden_size=geometry.hidden_size,
            num_hidden_layers=geometry.num_hidden_layers,
            vocab_size=geometry.vocab_size)
        self.model = TinySparkModel(config)
        self.lm_head = nn.Linear(geometry.hidden_size, geometry.vocab_size,
                                 bias=False, dtype=torch.bfloat16)
        self.lm_head.weight = self.model.embedding.weight

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                use_cache=False, **kwargs):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, use_cache=use_cache)
        logits = F.linear(hidden.last_hidden_state, self.model.embedding.weight)
        return SimpleNamespace(logits=logits)

    def get_input_embeddings(self):
        return self.model.embedding

    def get_output_embeddings(self):
        return self.lm_head


def build(seed: int = 7, *, quantized: bool = False,
          require_official_geometry: bool = False, device: str = "cpu"):
    torch.manual_seed(seed)
    raw = tiny_config_raw(quantized=quantized)
    config = load_spark2_5_config(raw, require_official_geometry=require_official_geometry)
    base = TinySparkCausalLM(config)
    artifact = sha256_bytes(b"tiny-spark-artifact")
    precision = (official_native_fp8_precision(config, artifact_sha256=artifact)
                 if quantized else official_bf16_precision(config, artifact_sha256=artifact))
    adapter = Spark2_5Adapter(
        base, config, precision, device=device, require_official_geometry=False,
        projector_layer=3, memory_size=4, heads=2)
    adapter.seal_original_identity()
    return adapter, base, config, precision


def ids_and_mask(sequence: int = 6):
    ids = torch.tensor([[3, 5, 7, 9, 11, 13][:sequence]], dtype=torch.long)
    return ids, torch.ones_like(ids)


def test_geometry_contract_and_official_admission_refusal():
    tiny = load_spark2_5_config(tiny_config_raw(), require_official_geometry=False)
    geometry = tiny.geometry
    assert (geometry.q_dim, geometry.kv_dim, geometry.qkv_dim, geometry.kv_repeat) == (8, 4, 16, 2)
    assert geometry.parameter_elements < OFFICIAL_PARAMETER_ELEMENTS
    assert derive_layer_schedule(tiny).full_indices == (3, 7)

    official = load_spark2_5_config(official_config_raw(), require_official_geometry=True)
    assert official.geometry.is_official_geometry
    assert official.geometry.parameter_elements == OFFICIAL_PARAMETER_ELEMENTS
    assert derive_layer_schedule(official).full_indices == (3, 7, 11, 15, 19, 23, 27, 31, 35)

    adapter, _, _, _ = build()
    assert adapter.describe()["official_geometry"] is False
    with pytest.raises(GeometryError):
        build(require_official_geometry=True)
    with pytest.raises(GeometryError):
        load_spark2_5_config(tiny_config_raw(), require_official_geometry=True)


def test_zero_init_parity_nonzero_influence_and_eager_cache():
    adapter, base, _, _ = build()
    ids, mask = ids_and_mask()
    with torch.no_grad():
        base_logits = base(input_ids=ids, attention_mask=mask, use_cache=False).logits
        adapter_logits, audit = adapter.forward_text(ids, mask, return_audit=True)
    assert torch.equal(adapter_logits, base_logits)
    assert audit.sliding_layers and audit.full_layers
    assert adapter.attention_implementation == "eager"
    assert base.config.use_cache is False

    original = adapter.extension.vision_projector.fc2.weight.detach().clone()
    try:
        with torch.no_grad():
            adapter.extension.vision_projector.fc2.weight.copy_(
                original + torch.randn_like(original) * 0.05)
            perturbed = adapter.forward_text(ids, mask)
    finally:
        with torch.no_grad():
            adapter.extension.vision_projector.fc2.weight.copy_(original)
    assert not torch.equal(perturbed, base_logits)
    restored = adapter.forward_text(ids, mask)
    assert torch.equal(restored, base_logits)

    hidden = adapter.forward_hidden(ids, mask)
    assert hidden.shape == (1, ids.shape[1], 8)


def test_freeze_scope_rejects_wrong_attention_and_fp8():
    adapter, base, _, _ = build()
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(parameter.requires_grad for _, parameter in adapter.new_named_parameters())
    assert all(parameter.dtype == torch.float32 for _, parameter in adapter.new_named_parameters())
    audit = adapter.parameter_audit()
    assert audit["original"]["trainable_parameters"] == 0
    assert audit["new"]["dtypes"] == ["torch.float32"]

    bad, bad_base, bad_config, bad_precision = build()
    bad_base.config._attn_implementation = "flash_attention_2"
    with pytest.raises(SparkAdapterError, match="eager attention"):
        Spark2_5Adapter(bad_base, bad_config, bad_precision, device="cpu",
                        require_official_geometry=False)

    with pytest.raises((BackendAdmissionError, SparkAdapterError)):
        build(quantized=True)


def test_gradient_audit_finite_nonzero_and_base_unchanged():
    adapter, base, _, _ = build()
    adapter.enable_memory_ports((3, 7))
    ids, mask = ids_and_mask()
    memory = torch.linspace(-1, 1, steps=8 * 4).reshape(1, 8, 4)
    memory_mask = torch.ones((1, 8), dtype=torch.bool)
    probe = adapter.backward_probe(ids, mask, memory=memory, memory_mask=memory_mask)
    assert probe["grad_enabled"] is True
    assert probe["loss_finite"] is True
    assert probe["injection"]["sliding_layers"]
    assert probe["injection"]["full_layers"]
    assert probe["input_gradient"]["present"] is True
    assert probe["new_gradient_finite"] is True
    assert probe["new_gradient_nonzero"] is True
    assert probe["gradient_audit"]["new"]["missing_gradient_names"] == []
    assert probe["gradient_audit"]["original"]["gradient_tensors"] == 0
    assert adapter.base_unchanged() is True
    assert all(parameter.grad is None for parameter in base.parameters())
    weight_grad = adapter.extension.vision_projector.fc2.weight.grad
    assert weight_grad is not None and bool(torch.count_nonzero(weight_grad))


def test_single_fp32_adamw_update_only_touches_new_parameters():
    adapter, base, _, _ = build()
    ids, mask = ids_and_mask()
    probe = adapter.backward_probe(ids, mask)
    before = {name: parameter.detach().clone()
              for name, parameter in adapter.new_named_parameters()}
    before_base = adapter.parameter_stamp(original_only=True)
    optimizer = build_optimizer(adapter, "adamw_fused", lr_base=0.0, lr_new=1e-3)
    assert isinstance(optimizer, NativeFusedAdamW)
    optimizer.step()
    after = {name: parameter.detach().clone()
             for name, parameter in adapter.new_named_parameters()}
    deltas = {name: float((after[name] - before[name]).abs().max()) for name in before}
    assert any(value > 0 for value in deltas.values())
    assert adapter.parameter_stamp(original_only=True) == before_base
    assert adapter.base_unchanged() is True
    assert all(bool(torch.isfinite(parameter).all())
               for _, parameter in adapter.new_named_parameters())
    audit = optimizer.audit()
    assert audit["fused"] is True and audit["implementation"] == "native_fused"
    assert audit["state_dtypes"] == ["torch.float32"]
    assert audit["groups"][0]["scope"] == "new"
    post = adapter.backward_probe(ids, mask)
    assert post["input_gradient"]["nonzero"] is True

    # Ownership is complete and disjoint; the frozen base is not a group.
    owned = [id(parameter) for group in optimizer.param_groups
             for parameter in group["params"]]
    assert set(owned) == {id(parameter) for parameter in adapter.parameters()
                          if parameter.requires_grad}
    base_ids = {id(parameter) for parameter in base.parameters()}
    assert not (set(owned) & base_ids)


def test_extension_export_is_complete_extension_only_and_rejects_stale_base(tmp_path):
    adapter, _, config, precision = build()
    adapter.enable_memory_ports((3, 7))
    extension_dir = tmp_path / "extension"
    manifest = adapter.save_extension(extension_dir)
    assert {path.name for path in extension_dir.iterdir()} == {
        "extensions.safetensors", "manifest.json"}
    assert not (extension_dir / "base").exists()
    assert manifest["contains_base_weights"] is False
    assert manifest["base_saved"] is False
    assert manifest["export"]["contains_base_weights"] is False
    names = {item["name"] for item in manifest["export"]["adapter_tensors"]}
    assert "extension.vision_projector.fc2.weight" in names
    assert any(name.startswith("extension.memory_ports.3.") for name in names)
    assert any(name.startswith("extension.memory_ports.7.") for name in names)
    assert [port["layer"] for port in manifest["export"]["extension"]["memory_ports"]] == [3, 7]
    saved = {name: parameter.detach().clone()
             for name, parameter in adapter.new_named_parameters()}

    fresh, _, _, _ = build(seed=11)
    fresh.enable_memory_ports((3, 7))
    receipt = fresh.load_extension(extension_dir)
    assert receipt["extension_only"] is True
    assert receipt["loaded_tensors"] == len(saved)
    for name, parameter in fresh.new_named_parameters():
        torch.testing.assert_close(parameter, saved[name], rtol=0, atol=0)

    stale = copy.deepcopy(manifest)
    stale["export"]["base"]["artifact_sha256"] = "b" * 64
    stale["export"]["fingerprint"] = fingerprint(
        {key: value for key, value in stale["export"].items() if key != "fingerprint"})
    stale["fingerprint"] = fingerprint(
        {key: value for key, value in stale.items() if key != "fingerprint"})
    (extension_dir / "manifest.json").write_text(json.dumps(stale))
    with pytest.raises((ExtensionStateError, SparkAdapterError), match="base"):
        fresh.load_extension(extension_dir)


def test_plan_pins_environment_and_payload_refusals(tmp_path):
    plan = qualification.load_plan(ROOT / "configs/spark_qualification_v1.json")
    assets = ROOT / "reports/spark_preparation_20261025/run_01/assets"
    assert qualification.file_sha256(assets / "official_bf16_control/config.json") == \
        plan["sources"]["official_bf16_control"]["config_sha256"]
    assert qualification.file_sha256(assets / "official_fp8_adaptation_base/config.json") == \
        plan["sources"]["official_fp8_adaptation_base"]["config_sha256"]

    good = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "REPLICA_CAPACITY_GUARD": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    issues = qualification.environment_issues(
        plan, "official_bf16_control", platform_name="Darwin",
        python_version="3.12.14", torch_version="2.14.1",
        transformers_version="4.57.1", compressed_tensors_version=None,
        mps_available=True, environ=good)
    assert issues == []
    bad = dict(good, PYTORCH_ENABLE_MPS_FALLBACK="1")
    issues = qualification.environment_issues(
        plan, "official_bf16_control", platform_name="linux",
        python_version="3.11.9", torch_version="2.13.0",
        transformers_version="5.20.2", compressed_tensors_version=None,
        mps_available=False, environ=bad)
    assert any("platform" in issue for issue in issues)
    assert any("torch" in issue for issue in issues)
    assert any("transformers" in issue for issue in issues)
    assert any("MPS" in issue for issue in issues)
    assert any("PYTORCH_ENABLE_MPS_FALLBACK" in issue for issue in issues)

    with pytest.raises(qualification.QualificationError, match="missing_official_payload"):
        qualification.resolve_source(plan, "official_bf16_control", tmp_path)
    with pytest.raises(qualification.QualificationError, match="missing_derivative"):
        qualification.resolve_source(plan, "dequantized_fp8_derivative", tmp_path)
    with pytest.raises(qualification.QualificationError, match="insufficient free space"):
        qualification.storage_check(plan, free_bytes=1024)

    tampered = copy.deepcopy(plan)
    tampered["claims"]["training_claimed"] = True
    with pytest.raises(qualification.QualificationError):
        qualification.validate_plan(tampered)
    tampered = copy.deepcopy(plan)
    tampered["sources"]["dequantized_fp8_derivative"]["max_derivatives"] = 2
    with pytest.raises(qualification.QualificationError):
        qualification.validate_plan(tampered)
    tampered = copy.deepcopy(plan)
    tampered["sources"]["official_bf16_control"]["shards"][0]["sha256"] = "not-a-sha"
    with pytest.raises(qualification.QualificationError):
        qualification.validate_plan(tampered)

    logits_a = torch.zeros(1, 3, 4)
    logits_b = torch.zeros(1, 3, 4)
    report = qualification.compare_logits(logits_a, logits_b, rtol=1e-5, atol=1e-6)
    assert report["within_tolerance"] is True
    logits_b[0, 0, 0] = 1.0
    assert qualification.compare_logits(logits_a, logits_b, rtol=1e-5,
                                        atol=1e-6)["within_tolerance"] is False


def test_runner_resolves_derivative_manifest_and_rejects_stale_or_unlabelled(tmp_path):
    plan = qualification.load_plan(ROOT / "configs/spark_qualification_v1.json")
    parent_entry = plan["sources"]["official_fp8_adaptation_base"]
    parent_artifact = qualification.artifact_digest(parent_entry["shards"])
    parent_config = load_spark2_5_config(
        json.loads((ROOT / "reports/spark_preparation_20261025/run_01/assets/"
                    "official_fp8_adaptation_base/config.json").read_text()),
        require_official_geometry=True)
    parent = official_native_fp8_precision(
        parent_config, artifact_sha256=parent_artifact)
    derivative_raw = official_config_raw()
    config = load_spark2_5_config(derivative_raw, require_official_geometry=True)
    semantics = plan["sources"]["dequantized_fp8_derivative"]["activation_semantics"]
    source = tmp_path / "derivative"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(derivative_raw))
    asset_pins = {}
    for name in plan["sources"]["dequantized_fp8_derivative"]["asset_sha256"]:
        content = f"synthetic derivative asset {name}".encode()
        (source / name).write_bytes(content)
        asset_pins[name] = sha256_bytes(content)
    plan["sources"]["dequantized_fp8_derivative"]["asset_sha256"] = asset_pins
    shard = source / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"synthetic-derivative")
    shard_sha = sha256_bytes(shard.read_bytes())
    artifact = qualification.artifact_digest(
        [{"filename": shard.name, "sha256": shard_sha}])
    precision = dequantized_fp8_derivative(
        config, parent=parent, artifact_sha256=artifact,
        dequantization_sha256=qualification.file_sha256(DEQUANTIZER),
        activation_semantics=semantics, storage="bf16")
    manifest = {
        "schema": "replica.spark2_5_dequantized_derivative.v1",
        "derivative_index": 1,
        "source": {
            "repo_id": parent_entry["repo_id"], "revision": parent_entry["revision"],
            "config_sha256": parent_entry["config_sha256"],
            "artifact_sha256": parent_artifact,
            "quantization_fingerprint": parent_entry["quantization_fingerprint"],
            "parent_precision": parent.to_dict(),
        },
        "derivative": {
            "artifact_sha256": artifact, "derivative_of": parent_artifact,
            "dequantization_sha256": qualification.file_sha256(DEQUANTIZER),
            "activation_semantics": semantics,
            "equivalent_to_bf16_release": False,
            "config_sha256": sha256_bytes((source / "config.json").read_bytes()),
            "precision_digest": precision.digest,
        },
        "files": [{"filename": shard.name, "bytes": shard.stat().st_size,
                   "sha256": shard_sha}],
        "tensor_inventory": {"parameter_elements": config.geometry.parameter_elements,
                             "tensor_count": 290},
        "claims": {"official_native_fp8": False},
    }
    manifest["fingerprint"] = fingerprint(manifest)
    manifest_path = source / "derivative_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    resolved = qualification.resolve_source(plan, "dequantized_fp8_derivative", source)
    assert resolved["precision"].kind == "dequantized_fp8_derivative"
    assert resolved["derivative_manifest"]["derivative"]["artifact_sha256"] == artifact

    tampered = copy.deepcopy(manifest)
    tampered["source"]["artifact_sha256"] = "c" * 64
    tampered["fingerprint"] = fingerprint(
        {key: value for key, value in tampered.items() if key != "fingerprint"})
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(qualification.QualificationError, match="parent artifact"):
        qualification.resolve_source(plan, "dequantized_fp8_derivative", source)

    bad_config = copy.deepcopy(derivative_raw)
    bad_config["quantization_config"] = official_quant_config_raw()["quantization_config"]
    (source / "config.json").write_text(json.dumps(bad_config))
    manifest["derivative"]["config_sha256"] = sha256_bytes(
        (source / "config.json").read_bytes())
    manifest["fingerprint"] = fingerprint(
        {key: value for key, value in manifest.items() if key != "fingerprint"})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(qualification.QualificationError, match="implicit_fp8_dequantization"):
        qualification.resolve_source(plan, "dequantized_fp8_derivative", source)


def _arm_report(role: str, kind: str, *, tokens, rendered="same"):
    return {
        "status": "completed",
        "source": {"role": role, "artifact_sha256": "a" * 64,
                   "precision": {"kind": kind, "activation": "semantics"}},
        "claims": {"capability_claim": "none", "heldout_accessed": False},
        "stages": {"inference": {"generations": [{
            "prompt_id": "p", "thinking": "none",
            "rendered_prompt_sha256": sha256_bytes(rendered.encode()),
            "prompt_token_count": 4, "token_ids": tokens,
            "decoded_text": "text", "stop_reason": "max_new_tokens",
            "latency_seconds": 1.0, "reasoning_token_count": 0,
        }]}},
    }


def test_arm_comparison_is_paired_and_non_bitwise():
    plan = qualification.load_plan(ROOT / "configs/spark_qualification_v1.json")
    baseline = _arm_report("official_bf16_control", "official_bf16", tokens=[1, 2, 3])
    derivative = _arm_report("dequantized_fp8_derivative",
                             "dequantized_fp8_derivative", tokens=[1, 2, 3])
    comparison = qualification.compare_arms(plan, baseline, derivative)
    assert comparison["comparison"]["non_bitwise"] is True
    assert comparison["comparison"]["all_prompts_paired"] is True
    assert comparison["comparison"]["exact_token_agreement_count"] == 1
    assert comparison["rows"][0]["exact_token_match"] is True
    assert comparison["claims"]["capability_claim"] == "none"
    assert comparison["derivative"]["precision_kind"] == "dequantized_fp8_derivative"

    different = _arm_report("dequantized_fp8_derivative",
                            "dequantized_fp8_derivative", tokens=[9])
    comparison = qualification.compare_arms(plan, baseline, different)
    assert comparison["comparison"]["exact_token_agreement_count"] == 0
    assert comparison["rows"][0]["exact_token_match"] is False

    wrong_role = _arm_report("official_bf16_control", "official_bf16", tokens=[1])
    with pytest.raises(qualification.QualificationError, match="not the labelled derivative"):
        qualification.compare_arms(plan, baseline, wrong_role)
    wrong_template = _arm_report("dequantized_fp8_derivative",
                                 "dequantized_fp8_derivative", tokens=[1],
                                 rendered="different")
    with pytest.raises(qualification.QualificationError, match="template|tokenization"):
        qualification.compare_arms(plan, baseline, wrong_template)
    claimed = copy.deepcopy(derivative)
    claimed["claims"]["capability_claim"] = "trained"
    with pytest.raises(qualification.QualificationError, match="claims a capability"):
        qualification.compare_arms(plan, baseline, claimed)


def test_port_choices_are_full_attention_and_extension_identity_changes():
    adapter, _, _, _ = build()
    with pytest.raises(ExtensionStateError):
        adapter.enable_memory_ports((0,))
    identity = adapter.enable_memory_ports((3, 7))
    assert identity["status"] == "experimental_identity_change"
    assert adapter.port_layers == (3, 7)
    with pytest.raises(SparkAdapterError, match="already enabled"):
        adapter.enable_memory_ports((3, 7))
