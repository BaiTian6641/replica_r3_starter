"""Focused CPU tests for the dedicated Spark2_5 preparation contract.

All configs, tensor indexes and tokenizer payloads here are synthetic. No test
downloads weights, imports a Spark model implementation or executes native code.
"""
from __future__ import annotations

import copy
import json

import pytest

from replica_lab.architecture.spark2_5 import (
    DEFAULT_BACKENDS, DEQUANTIZED_FP8_DERIVATIVE, OFFICIAL_BF16,
    OFFICIAL_NATIVE_FP8, OFFICIAL_PARAMETER_ELEMENTS, BackendAdmissionError,
    BackendCapabilities, ExtensionStateError, GeometryError, InventoryError,
    PrecisionError, TiedEmbeddingError, ThinkingModeError,
    TokenizerIdentityError, admit_load, build_extension_export,
    build_extensions, build_memory_ports, build_projector, classify_precision,
    dequantized_fp8_derivative, derive_layer_schedule, evaluate_admission,
    fingerprint, load_spark2_5_config, load_tokenizer_identity,
    official_bf16_precision, official_native_fp8_precision,
    prepare_spark2_5, resolve_thinking_mode, validate_extension_export,
    verify_tensor_inventory, verify_tied_embeddings,
)

LAYER_TYPES = (["sliding_attention"] * 3 + ["full_attention"] +
               (["sliding_attention"] * 3 + ["full_attention"]) * 8)
TEMPLATE = (
    "{%- if not messages %}{{ raise_exception('No messages provided.') }}{%- endif %}"
    "{%- set enable_thinking = enable_thinking | default(true) %}"
    "{%- set reasoning_effort = reasoning_effort | default('high') %}"
    "{%- if reasoning_effort not in ('high', 'low') %}{{ raise_exception('effort') }}"
    "{%- endif %}{{ '<think>' }}{{ '</think>' }}"
)


def config(quantized: bool = False, *, layer_types=LAYER_TYPES,
           hidden_size: int = 2560, num_hidden_layers: int = 36,
           tie_word_embeddings: bool = True, head_dim: int = 256,
           vocab_size: int = 131072) -> dict:
    raw = {
        "architectures": ["Spark2_5ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_spark.Spark2_5Config",
            "AutoModel": "modeling_spark.Spark2_5Model",
            "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
        },
        "model_type": "spark2_5",
        "dtype": "bfloat16",
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "intermediate_size": 10240,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "sliding_window": 512,
        "max_position_embeddings": 1048576,
        "tie_word_embeddings": tie_word_embeddings,
        "use_cache": True,
        "hidden_act": "gelu",
        "attention_bias": False,
        "mlp_bias": False,
        "headwise_attn_output_gate": True,
        "gate_attn_act_mode": "sigmoid",
        "layer_types": list(layer_types),
        "rope_parameters": {
            "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
        },
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
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
                "targets": ["re:.*self_attn.q_k_v_proj$",
                            "re:.*self_attn.out_proj$",
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


def tokenizer_config(config_raw: dict) -> dict:
    return {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 131072,
        "add_bos_token": False,
        "add_eos_token": False,
        "bos_token": {"content": "<|start|>"},
        "eos_token": {"content": "<|end|>"},
        "pad_token": {"content": "<|pad|>"},
        "unk_token": {"content": "<unk>"},
        "chat_template": TEMPLATE,
    }


def tensor_specs(config_raw: dict, *, quantized: bool) -> list[dict]:
    layers = config_raw["num_hidden_layers"]
    hidden = config_raw["hidden_size"]
    intermediate = config_raw["intermediate_size"]
    heads = config_raw["num_attention_heads"]
    kv = config_raw["num_key_value_heads"]
    head_dim = config_raw["head_dim"]
    q_dim = heads * head_dim
    qkv_dim = (heads + 2 * kv) * head_dim
    specs = [{"name": "model.embedding.weight", "dtype": "BF16",
              "shape": [config_raw["vocab_size"], hidden]}]
    for index in range(layers):
        prefix = f"model.layers.{index}."
        specs += [
            {"name": prefix + "input_layernorm.weight", "dtype": "BF16", "shape": [hidden]},
            {"name": prefix + "post_attention_layernorm.weight", "dtype": "BF16", "shape": [hidden]},
            {"name": prefix + "self_attn.q_k_v_proj.weight",
             "dtype": "F8_E4M3" if quantized else "BF16", "shape": [qkv_dim, hidden]},
            {"name": prefix + "self_attn.out_proj.weight",
             "dtype": "F8_E4M3" if quantized else "BF16", "shape": [hidden, q_dim]},
            {"name": prefix + "self_attn.g_proj.weight", "dtype": "BF16",
             "shape": [heads, hidden]},
            {"name": prefix + "mlp.gate_proj.weight",
             "dtype": "F8_E4M3" if quantized else "BF16", "shape": [intermediate, hidden]},
            {"name": prefix + "mlp.up_proj.weight",
             "dtype": "F8_E4M3" if quantized else "BF16", "shape": [intermediate, hidden]},
            {"name": prefix + "mlp.down_proj.weight",
             "dtype": "F8_E4M3" if quantized else "BF16", "shape": [hidden, intermediate]},
        ]
        if quantized:
            specs += [
                {"name": prefix + "self_attn.q_k_v_proj.weight_scale", "dtype": "BF16",
                 "shape": [qkv_dim, 1]},
                {"name": prefix + "self_attn.out_proj.weight_scale", "dtype": "BF16",
                 "shape": [hidden, 1]},
                {"name": prefix + "mlp.gate_proj.weight_scale", "dtype": "BF16",
                 "shape": [intermediate, 1]},
                {"name": prefix + "mlp.up_proj.weight_scale", "dtype": "BF16",
                 "shape": [intermediate, 1]},
                {"name": prefix + "mlp.down_proj.weight_scale", "dtype": "BF16",
                 "shape": [hidden, 1]},
            ]
    specs.append({"name": "model.norm.weight", "dtype": "BF16", "shape": [hidden]})
    return specs


def refingerprint(payload: dict) -> dict:
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    payload["fingerprint"] = fingerprint(body)
    return payload


# ---------------------------------------------------------------------------
# Geometry and schedule


def test_official_geometry_is_36x2560_and_not_minicpm5():
    loaded = load_spark2_5_config(config(), require_official_geometry=True)
    geometry = loaded.geometry
    assert (geometry.num_hidden_layers, geometry.hidden_size) == (36, 2560)
    assert (geometry.intermediate_size, geometry.head_dim) == (10240, 256)
    assert geometry.head_dim != geometry.hidden_size // geometry.num_attention_heads
    assert geometry.kv_repeat == 4
    assert geometry.parameter_elements == OFFICIAL_PARAMETER_ELEMENTS == 4_112_079_360
    assert geometry.tie_word_embeddings is True

    broken = config(hidden_size=2048)
    with pytest.raises(GeometryError, match="MiniCPM5"):
        load_spark2_5_config(broken)
    broken = config(num_hidden_layers=42,
                    layer_types=list(LAYER_TYPES) + ["full_attention"] * 6)
    with pytest.raises(GeometryError, match="42 layers"):
        load_spark2_5_config(broken)
    broken = config()
    broken.pop("head_dim")
    with pytest.raises(GeometryError, match="head_dim must be explicit"):
        load_spark2_5_config(broken)


def test_schedule_comes_from_config_and_records_the_3_plus_1_block():
    loaded = load_spark2_5_config(config())
    schedule = derive_layer_schedule(loaded)
    assert schedule.full_indices == (3, 7, 11, 15, 19, 23, 27, 31, 35)
    assert schedule.period_runs == (("sliding_attention", 3), ("full_attention", 1))
    assert schedule.period_repeats == 9
    assert len(schedule.sliding_indices) == 27
    assert schedule.to_dict()["source"] == "config.layer_types"

    # A different declared order must be reported as declared, not rewritten
    # into the pinned repeating pattern.
    alternate = ["full_attention", "sliding_attention", "sliding_attention"] * 12
    loaded = load_spark2_5_config(config(layer_types=alternate))
    schedule = derive_layer_schedule(loaded)
    assert schedule.full_indices[:3] == (0, 3, 6)
    assert schedule.period_runs == (("full_attention", 1), ("sliding_attention", 2))
    assert schedule.period_repeats == 12


def test_config_rejects_wrong_layer_count_and_unknown_layer_types():
    with pytest.raises(GeometryError, match="layer_types"):
        load_spark2_5_config(config(layer_types=LAYER_TYPES[:-1]))
    broken = config()
    broken["layer_types"][0] = "cross_attention"
    with pytest.raises(GeometryError, match="layer_types"):
        load_spark2_5_config(broken)


# ---------------------------------------------------------------------------
# Precision identities and backend admission


def _fp8_precision(config_raw=None):
    return official_native_fp8_precision(
        load_spark2_5_config(config_raw or config(quantized=True)),
        artifact_sha256="b" * 64)


def _bf16_precision(config_raw=None):
    return official_bf16_precision(
        load_spark2_5_config(config_raw or config()),
        artifact_sha256="a" * 64)


def test_native_fp8_on_mps_is_refused_without_fallback_or_dequantization():
    precision = _fp8_precision()
    assert precision.kind == OFFICIAL_NATIVE_FP8
    assert precision.activation_quantized is True
    assert precision.quantized_modules == (
        "self_attn.q_k_v_proj", "self_attn.out_proj", "mlp.gate_proj",
        "mlp.up_proj", "mlp.down_proj")
    assert DEFAULT_BACKENDS["mps"].fp8_storage is False

    admission = evaluate_admission(precision, device="mps")
    assert admission.admitted is False
    assert admission.requested_device == admission.effective_device == "mps"
    assert admission.fallback_allowed is False
    assert admission.dequantization_allowed is False
    assert "mps_fp8_unsupported" in admission.reasons[0]

    with pytest.raises(BackendAdmissionError) as error:
        admit_load(precision, device="mps:0")
    assert error.value.reason == "mps_fp8_unsupported"
    evidence = error.value.evidence()
    assert evidence["fallback_used"] is False
    assert evidence["dequantization_performed"] is False
    assert evidence["requested_device"] == "mps:0"

    with pytest.raises(BackendAdmissionError, match="cpu_fp8_unsupported"):
        admit_load(precision, device="cpu")

    capable = BackendCapabilities("cuda", True, True, True, True, "declared capable")
    admitted = admit_load(precision, device="cuda:0", capabilities=capable)
    assert admitted.admitted is True and admitted.effective_device == "cuda:0"


def test_dequantized_derivative_is_a_third_explicit_identity():
    fp8 = _fp8_precision()
    derivative_config = config()
    derivative = dequantized_fp8_derivative(
        load_spark2_5_config(derivative_config), parent=fp8,
        artifact_sha256="c" * 64, dequantization_sha256="d" * 64,
        activation_semantics="bf16 activations; no per-token F8 quantization",
        storage="bf16")
    assert derivative.kind == DEQUANTIZED_FP8_DERIVATIVE
    assert derivative.kind not in (OFFICIAL_BF16, OFFICIAL_NATIVE_FP8)
    assert derivative.dequantized is True
    assert derivative.activation_quantized is False
    assert derivative.derivative_of == fp8.artifact_sha256
    assert derivative.artifact_sha256 != fp8.artifact_sha256
    assert derivative.equivalent_to_bf16_release is False

    # It may run on this backend only because it is explicitly BF16-labelled;
    # the admission report still shows the derivative kind, never native FP8.
    admitted = admit_load(derivative, device="mps")
    assert admitted.precision_kind == DEQUANTIZED_FP8_DERIVATIVE

    with pytest.raises(PrecisionError, match="cannot reuse the parent"):
        dequantized_fp8_derivative(
            load_spark2_5_config(config()), parent=fp8,
            artifact_sha256=fp8.artifact_sha256,
            dequantization_sha256="d" * 64,
            activation_semantics="bf16 activations")
    with pytest.raises(PrecisionError, match="official native FP8 parent"):
        dequantized_fp8_derivative(
            load_spark2_5_config(config()), parent=_bf16_precision(),
            artifact_sha256="c" * 64, dequantization_sha256="d" * 64,
            activation_semantics="bf16 activations")
    with pytest.raises(PrecisionError, match="must not keep float-quantized"):
        dequantized_fp8_derivative(
            load_spark2_5_config(config(quantized=True)), parent=fp8,
            artifact_sha256="c" * 64, dequantization_sha256="d" * 64,
            activation_semantics="bf16 activations")


def test_precision_classification_requires_explicit_derivative_manifest():
    assert classify_precision(load_spark2_5_config(config())).kind == OFFICIAL_BF16
    assert classify_precision(load_spark2_5_config(config(quantized=True))).kind \
        == OFFICIAL_NATIVE_FP8
    with pytest.raises(PrecisionError, match="BF16 identity cannot silently strip"):
        official_bf16_precision(load_spark2_5_config(config(quantized=True)))

    loaded = load_spark2_5_config(config())
    derived = classify_precision(loaded, derivative_manifest={
        "parent": _fp8_precision(),
        "artifact_sha256": "c" * 64,
        "dequantization_sha256": "d" * 64,
        "activation_semantics": "bf16 activations; no per-token F8 quantization",
    })
    assert derived.kind == DEQUANTIZED_FP8_DERIVATIVE
    with pytest.raises(PrecisionError, match="must be a PrecisionIdentity"):
        classify_precision(loaded, derivative_manifest={
            "parent": {"artifact_sha256": "b" * 64},
            "artifact_sha256": "c" * 64,
            "dequantization_sha256": "d" * 64,
            "activation_semantics": "bf16 activations",
        })


# ---------------------------------------------------------------------------
# Tokenizer, template and thinking mode


def test_tokenizer_template_identity_pins_all_three_thinking_modes():
    loaded = load_spark2_5_config(config())
    identity = load_tokenizer_identity(loaded, tokenizer_config(config()), TEMPLATE)
    assert identity.vocab_size == 131072
    assert identity.special_tokens["bos_token"] == "<|start|>"
    assert identity.chat_template_sources_agree is True
    assert set(identity.thinking_modes) == {"none", "low", "high"}
    assert resolve_thinking_mode(identity, "none") == {"enable_thinking": False}
    assert resolve_thinking_mode(identity, "low") == {
        "enable_thinking": True, "reasoning_effort": "low"}
    assert resolve_thinking_mode(identity, "high") == {
        "enable_thinking": True, "reasoning_effort": "high"}
    assert identity.default_thinking_mode == "high"
    assert identity.reasoning_delimiters == ("<think>", "</think>")

    changed = TEMPLATE + " "
    assert load_tokenizer_identity(
        loaded, tokenizer_config(config()), changed).digest != identity.digest


def test_unknown_thinking_mode_is_rejected():
    loaded = load_spark2_5_config(config())
    identity = load_tokenizer_identity(loaded, tokenizer_config(config()), TEMPLATE)
    for mode in ("medium", "auto", "HIGH", "", None, 1):
        with pytest.raises(ThinkingModeError, match="thinking mode must be one of"):
            resolve_thinking_mode(identity, mode)


def test_template_without_the_pinned_thinking_contract_is_rejected():
    loaded = load_spark2_5_config(config())
    with pytest.raises(TokenizerIdentityError, match="missing"):
        load_tokenizer_identity(loaded, tokenizer_config(config()),
                                TEMPLATE.replace("reasoning_effort", "effort"))
    with pytest.raises(TokenizerIdentityError, match="unsupported thinking mode"):
        load_tokenizer_identity(loaded, tokenizer_config(config()),
                                TEMPLATE + "{{ 'medium' }}")


# ---------------------------------------------------------------------------
# Inventory, tied embeddings and full preparation


def test_bf16_inventory_and_tied_embedding_are_complete():
    raw = config()
    loaded = load_spark2_5_config(raw, require_official_geometry=True)
    precision = _bf16_precision(raw)
    tensors = tensor_specs(raw, quantized=False)
    inventory = verify_tensor_inventory(tensors, loaded, precision)
    assert inventory.tensor_count == 290
    assert inventory.parameter_elements == OFFICIAL_PARAMETER_ELEMENTS
    assert inventory.scale_tensor_count == 0
    assert inventory.quantized_tensor_count == 0
    assert inventory.dtypes == ("BF16",)
    tied = verify_tied_embeddings(loaded, tensors)
    assert tied.declared_tied is True and tied.inspected is True
    assert tied.serialized_lm_head is False
    assert tied.unique_parameter_elements == 4_112_079_360
    assert tied.logits_path == "F.linear(hidden_states, model.embedding.weight)"

    with_serialized_head = tensor_specs(raw, quantized=False) + [
        {"name": "lm_head.weight", "dtype": "BF16", "shape": [131072, 2560]}]
    with pytest.raises(TiedEmbeddingError, match="must not be serialized"):
        verify_tensor_inventory(with_serialized_head, loaded, precision)
    with pytest.raises(TiedEmbeddingError, match="must not be serialized"):
        verify_tied_embeddings(loaded, with_serialized_head)


def test_native_fp8_inventory_records_scales_and_unquantized_gates():
    raw = config(quantized=True)
    loaded = load_spark2_5_config(raw, require_official_geometry=True)
    precision = _fp8_precision(raw)
    tensors = tensor_specs(raw, quantized=True)
    inventory = verify_tensor_inventory(tensors, loaded, precision)
    assert inventory.tensor_count == 470
    assert inventory.scale_tensor_count == 180
    assert inventory.quantized_tensor_count == 180
    assert inventory.parameter_elements == OFFICIAL_PARAMETER_ELEMENTS
    assert inventory.dtypes == ("BF16", "F8_E4M3")
    gate = next(item for item in tensors
                if item["name"] == "model.layers.0.self_attn.g_proj.weight")
    assert gate["dtype"] == "BF16"


def test_valid_bf16_preparation_report_keeps_all_native_blockers_explicit():
    raw = config()
    report = prepare_spark2_5(
        raw, tokenizer_config(raw), TEMPLATE,
        tensors=tensor_specs(raw, quantized=False), device="mps",
        require_official_geometry=True, artifact_sha256="a" * 64,
        projector_in_dim=1152, projector_hidden_dim=1024)
    assert report["schema"] == "replica.spark2_5_preparation.v1"
    assert report["status"] == "preparation_only_unqualified"
    assert report["geometry"]["official_geometry"] is True
    assert report["precision"]["kind"] == OFFICIAL_BF16
    assert report["inventory"]["tensor_count"] == 290
    assert report["tied_embeddings"]["serialized_lm_head"] is False
    assert report["admission"]["admitted"] is True
    assert report["admission"]["effective_device"] == "mps"
    assert report["stage_status"]["load"].startswith("blocked_unexecuted")
    for stage in ("inference", "input_gradients", "parameter_gradients",
                  "single_update", "export_reload"):
        assert report["stage_status"][stage] == "blocked_by_load"
    assert report["extensions"]["projector"]["out_dim"] == 2560
    assert [port["layer"] for port in report["extensions"]["memory_ports"]] == [11, 23]
    assert report["weights_downloaded"] is False
    assert report["native_execution"] is False
    assert report["remote_code_executed"] is False
    assert report["capability_claim"] == "none"
    assert report["fallback_allowed"] is False
    assert any("input_gradients" in item for item in report["blockers"])
    assert fingerprint({key: value for key, value in report.items()
                        if key != "fingerprint"}) == report["fingerprint"]


def test_native_fp8_preparation_report_is_blocked_on_mps():
    raw = config(quantized=True)
    report = prepare_spark2_5(
        raw, tokenizer_config(raw), TEMPLATE,
        tensors=tensor_specs(raw, quantized=True), device="mps",
        require_official_geometry=True, artifact_sha256="b" * 64)
    assert report["precision"]["kind"] == OFFICIAL_NATIVE_FP8
    assert report["admission"]["admitted"] is False
    assert report["admission"]["requested_device"] == "mps"
    assert report["admission"]["effective_device"] == "mps"
    assert report["stage_status"]["load"].startswith("blocked_precision_backend")
    assert "mps_fp8_unsupported" in report["stage_status"]["load"]
    assert report["fallback_allowed"] is False
    assert report["dequantization_performed"] is False


# ---------------------------------------------------------------------------
# Extension-only export/reload


def test_extension_inventory_is_spark_width_and_rejects_minicpm5_reuse():
    loaded = load_spark2_5_config(config())
    extensions = build_extensions(loaded, projector_in_dim=1152,
                                  projector_hidden_dim=1024)
    assert extensions.projector.out_dim == 2560
    assert extensions.projector.in_dim == 1152
    assert extensions.projector.hidden_dim == 1024
    assert all(layer is None or layer != 2048
               for _, shape in extensions.trainable_tensors for layer in shape)
    assert [port.layer for port in extensions.memory_ports] == [11, 23]
    assert all(port.width == 2560 for port in extensions.memory_ports)
    assert all(port.layer_type == "full_attention" for port in extensions.memory_ports)
    assert extensions.enabled_for_native_parity is False

    with pytest.raises(ExtensionStateError, match="MiniCPM5"):
        build_projector(loaded, in_dim=1152, hidden_dim=2048)
    with pytest.raises(ExtensionStateError, match="MiniCPM5"):
        build_memory_ports(loaded, layers=(13, 27))
    with pytest.raises(ExtensionStateError, match="not a full-attention block"):
        build_memory_ports(loaded, layers=(10,))


def test_extension_only_export_round_trips_and_rejects_cross_precision_reload():
    loaded = load_spark2_5_config(config())
    bf16 = _bf16_precision()
    fp8 = _fp8_precision()
    extensions = build_extensions(loaded, projector_in_dim=1152,
                                  projector_hidden_dim=1024)
    payload = build_extension_export(loaded, bf16, extensions)
    assert payload["contains_base_weights"] is False
    assert payload["original_checkpoint_bytes_rewritten"] is False
    assert all(tensor["name"].startswith("extension.")
               for tensor in payload["tensors"])
    assert len(payload["tensors"]) == 6

    reloaded = validate_extension_export(json.loads(json.dumps(payload)), loaded, bf16)
    assert reloaded["extension_only"] is True
    assert reloaded["base_unchanged"] is True
    assert reloaded["precision_kind"] == OFFICIAL_BF16
    assert reloaded["tensor_count"] == 6
    assert reloaded["memory_port_layers"] == [11, 23]
    assert reloaded["parameter_elements"] == extensions.projector.parameter_elements

    tampered = copy.deepcopy(payload)
    tampered["base"]["artifact_sha256"] = "e" * 64
    refingerprint(tampered)
    with pytest.raises(ExtensionStateError, match="base.artifact_sha256"):
        validate_extension_export(tampered, loaded, bf16)

    tampered = copy.deepcopy(payload)
    tampered["base"]["storage"] = "fp8_compressed_tensors"
    refingerprint(tampered)
    with pytest.raises(ExtensionStateError, match="base.storage"):
        validate_extension_export(tampered, loaded, bf16)

    with pytest.raises(ExtensionStateError, match="base.precision_kind"):
        validate_extension_export(payload, loaded, fp8)

    tampered = copy.deepcopy(payload)
    tampered["tensors"].append({"name": "model.embedding.weight",
                                "shape": [131072, 2560], "dtype": "FLOAT32"})
    refingerprint(tampered)
    with pytest.raises(ExtensionStateError, match="tensor index"):
        validate_extension_export(tampered, loaded, bf16)


def test_derivative_extension_export_records_its_parent_not_the_official_bf16_release():
    loaded = load_spark2_5_config(config())
    parent = _fp8_precision()
    derivative = dequantized_fp8_derivative(
        loaded, parent=parent, artifact_sha256="c" * 64,
        dequantization_sha256="d" * 64,
        activation_semantics="bf16 activations; no per-token F8 quantization")
    extensions = build_extensions(loaded, projector_in_dim=1152)
    payload = build_extension_export(loaded, derivative, extensions)
    recovered = validate_extension_export(payload, loaded, derivative)
    assert recovered["precision_kind"] == DEQUANTIZED_FP8_DERIVATIVE
    assert payload["base"]["derivative_of"] == parent.artifact_sha256
    assert payload["base"]["precision_kind"] != OFFICIAL_BF16
    with pytest.raises(ExtensionStateError, match="base.precision_kind"):
        validate_extension_export(payload, loaded, _bf16_precision())
    with pytest.raises(ExtensionStateError, match="float32"):
        build_extension_export(loaded, parent, extensions, trainable_dtype="bfloat16")


def test_inventory_rejects_missing_gate_scale_and_unexpected_tensor():
    raw = config(quantized=True)
    loaded = load_spark2_5_config(raw)
    precision = _fp8_precision(raw)
    tensors = tensor_specs(raw, quantized=True)
    missing_gate = [item for item in tensors
                    if item["name"] != "model.layers.0.self_attn.g_proj.weight"]
    with pytest.raises(InventoryError, match="missing expected tensors"):
        verify_tensor_inventory(missing_gate, loaded, precision)

    extra = tensor_specs(raw, quantized=True)
    extra.append({"name": "model.layers.0.mlp.gate_up_proj.weight",
                  "dtype": "BF16", "shape": [10240, 2560]})
    with pytest.raises(InventoryError, match="unexpected tensors"):
        verify_tensor_inventory(extra, loaded, precision)
