"""Focused synthetic tests for the FP8-derived BF16 P1/spine experiment.

No Spark weights, tokenizer, remote code or native job is used. The derivative
artifact is a tiny synthetic tree with eight shards; its manifest, config and
parent lineage follow the real derivative format so the review preflight can be
exercised without reading the sealed checkpoints.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _relative in (ROOT / "scripts", ROOT / "src", ROOT / "tests"):
    if str(_relative) not in sys.path:
        sys.path.insert(0, str(_relative))

from preflight_spark_persistent_learner_derivative_bf16 import (
    CONFIG_PATH, ProtocolError, preflight, validate_config,
    validate_derivative_source,
)
from contextual_spark_fixture import (
    ids_and_mask, labels_from_ids, open_memory_ports,
    populate_contextual_evidence,
)
from replica_lab.architecture.spark2_5 import (
    DEQUANTIZED_FP8_DERIVATIVE, DERIVATIVE_ACTIVATION_SEMANTICS,
    PrecisionIdentity, Spark2_5ContractError, classify_precision,
    dequantized_fp8_derivative, evaluate_derivative_p1_spine,
    derivative_artifact_digest, fingerprint, load_spark2_5_config,
    official_bf16_precision, official_native_fp8_precision,
)
from replica_lab.architecture.spark_adapter import Spark2_5Adapter, SparkAdapterError
from replica_lab.architecture.spine_memory import SpineMemoryExperiment
from replica_lab.architecture.spine_text import SparkTextMemoryPath
from replica_lab.optimizers import NativeFusedAdamW
from replica_lab.p1_response_learning import (
    build_port_optimizer, direct_response_ce_step, prepare_arm_query,
)
from test_spark2_5_adapter import (
    TinySparkCausalLM, official_config_raw, official_quant_config_raw,
    tiny_config_raw,
)

SEMANTICS = DERIVATIVE_ACTIVATION_SEMANTICS
RECORDS = (
    ("answer", "cobalt alpha is the verified response token"),
    ("near", "cobalt beta is a nearby distractor"),
)
ASSET_NAMES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return _sha(path)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_tree(tmp_path: Path, *, official_geometry: bool = False) -> dict:
    """Create one synthetic parent + eight-shard derivative artifact tree."""
    root = tmp_path / ("official" if official_geometry else "tiny")
    root.mkdir(parents=True)
    if official_geometry:
        parent_raw = official_quant_config_raw()
        derivative_raw = official_config_raw()
        geometry = load_spark2_5_config(derivative_raw,
                                        require_official_geometry=True).geometry
        tensor_count = 290
    else:
        parent_raw = tiny_config_raw(quantized=True)
        derivative_raw = tiny_config_raw()
        geometry = load_spark2_5_config(derivative_raw,
                                        require_official_geometry=False).geometry
        tensor_count = 2 + 8 * geometry.num_hidden_layers

    parent_dir = root / "parent"
    parent_dir.mkdir()
    parent_config_path = parent_dir / "config.json"
    parent_config_sha = _write_json(parent_config_path, parent_raw)
    parent_payload = parent_dir / "model.safetensors"
    parent_payload.write_bytes(b"synthetic-official-fp8-payload")
    parent_weight_sha = _sha(parent_payload)
    parent_artifact = derivative_artifact_digest([{
        "filename": "model.safetensors", "sha256": parent_weight_sha}])
    parent_config = load_spark2_5_config(
        parent_raw, require_official_geometry=official_geometry)
    parent_identity = official_native_fp8_precision(
        parent_config, artifact_sha256=parent_artifact)

    derivative_dir = root / "derivative"
    derivative_dir.mkdir()
    files = []
    for index in range(1, 9):
        name = f"model-{index:05d}-of-00008.safetensors"
        payload = derivative_dir / name
        payload.write_bytes(f"synthetic-derivative-shard-{index}".encode())
        files.append({"filename": name, "bytes": payload.stat().st_size,
                      "sha256": _sha(payload)})
    artifact = derivative_artifact_digest(files)
    dequantizer_sha = _sha(ROOT / "scripts/dequantize_spark_fp8.py")

    config_raw = copy.deepcopy(derivative_raw)
    config_raw["dtype"] = "bfloat16"
    config_raw["replica_precision"] = {
        "kind": DEQUANTIZED_FP8_DERIVATIVE,
        "parent_kind": "official_native_fp8",
        "parent_artifact_sha256": parent_artifact,
        "dequantization_sha256": dequantizer_sha,
        "activation_semantics": SEMANTICS,
        "equivalent_to_bf16_release": False,
    }
    config_path = derivative_dir / "config.json"
    config_sha = _write_json(config_path, config_raw)
    derivative_config = load_spark2_5_config(
        config_raw, require_official_geometry=official_geometry)
    precision = dequantized_fp8_derivative(
        derivative_config, parent=parent_identity, artifact_sha256=artifact,
        dequantization_sha256=dequantizer_sha,
        activation_semantics=SEMANTICS, storage="bf16")

    assets = {}
    for name in ASSET_NAMES:
        path = derivative_dir / name
        path.write_bytes(f"synthetic-derivative-{name}".encode())
        assets[name] = _sha(path)

    claims = {key: False for key in (
        "activation_quantization", "native_execution", "official_bf16_release",
        "official_native_fp8", "restores_quantization_loss")}
    manifest = {
        "schema": "replica.spark2_5_dequantized_derivative.v1",
        "derivative_index": 1,
        "claims": claims,
        "source": {
            "repo_id": "synthetic/Spark-FP8",
            "revision": "synthetic-parent-revision",
            "config_sha256": parent_config_sha,
            "artifact_sha256": parent_artifact,
            "precision_kind": "official_native_fp8",
            "quantization_fingerprint": parent_identity.quantization_fingerprint,
            "parent_precision": parent_identity.to_dict(),
            "source_files": [{"filename": "model.safetensors",
                              "bytes": parent_payload.stat().st_size,
                              "sha256": parent_weight_sha}],
        },
        "derivative": {
            "artifact_sha256": artifact,
            "config_sha256": config_sha,
            "dequantization_sha256": dequantizer_sha,
            "derivative_of": parent_artifact,
            "activation_semantics": SEMANTICS,
            "equivalent_to_bf16_release": False,
            "storage": "bf16",
            "weight_dtype": "BF16",
            "precision": precision.to_dict(),
            "precision_digest": precision.digest,
        },
        "files": files,
        "tensor_inventory": {
            "dtypes": ["BF16"],
            "parameter_elements": geometry.parameter_elements,
            "tensor_count": tensor_count,
        },
    }
    manifest["fingerprint"] = fingerprint(manifest)
    manifest_path = derivative_dir / "derivative_manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)

    repository = {
        "source_identities": {
            "official_fp8": {
                "repo_id": "synthetic/Spark-FP8",
                "revision": "synthetic-parent-revision",
                "local_path": str(parent_dir),
                "config_sha256": parent_config_sha,
                "weight_sha256": parent_weight_sha,
                "quantization_fingerprint": parent_identity.quantization_fingerprint,
                "substitution_allowed": False,
            },
            "official_bf16": {"substitution_allowed": False},
            "dequantized_fp8_derivative": {
                "local_path": str(derivative_dir),
                "manifest_path": str(manifest_path),
                "manifest_schema": "replica.spark2_5_dequantized_derivative.v1",
                "manifest_sha256": manifest_sha,
                "derivative_index": 1,
                "artifact_sha256": artifact,
                "config_sha256": config_sha,
                "dequantization_sha256": dequantizer_sha,
                "precision_digest": precision.digest,
                "parent_repo_id": "synthetic/Spark-FP8",
                "parent_revision": "synthetic-parent-revision",
                "parent_config_sha256": parent_config_sha,
                "parent_artifact_sha256": parent_artifact,
                "shard_count": 8,
                "parameter_elements": geometry.parameter_elements,
                "tensor_count": tensor_count,
                "activation_semantics": SEMANTICS,
                "execution_format": "dequantized_fp8_derivative_bf16_storage",
                "substitution_allowed": False,
            },
        },
        "runtime": {
            "tokenizer": {
                "tokenizer_json_sha256": assets["tokenizer.json"],
                "tokenizer_config_sha256": assets["tokenizer_config.json"],
                "chat_template_sha256": assets["chat_template.jinja"],
            },
        },
    }
    return {
        "repository": repository,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "derivative_dir": derivative_dir,
        "parent_dir": parent_dir,
        "parent_identity": parent_identity,
        "config": derivative_config,
        "precision": precision,
        "config_raw": config_raw,
        "files": files,
    }


def _rewrite_manifest(tree: dict) -> None:
    manifest = tree["manifest"]
    manifest["fingerprint"] = fingerprint(
        {key: value for key, value in manifest.items() if key != "fingerprint"})
    manifest_sha = _write_json(tree["manifest_path"], manifest)
    pins = tree["repository"]["source_identities"]["dequantized_fp8_derivative"]
    pins["manifest_sha256"] = manifest_sha
    pins["artifact_sha256"] = manifest["derivative"]["artifact_sha256"]
    pins["config_sha256"] = manifest["derivative"]["config_sha256"]
    pins["dequantization_sha256"] = manifest["derivative"]["dequantization_sha256"]
    pins["activation_semantics"] = manifest["derivative"]["activation_semantics"]
    pins["precision_digest"] = manifest["derivative"]["precision_digest"]


def test_manifest_and_all_eight_shards_validate(tmp_path):
    tree = build_tree(tmp_path)
    source = validate_derivative_source(
        tree["repository"], tmp_path, full_hash=True,
        require_official_geometry=False)
    assert source["source_identity"] == DEQUANTIZED_FP8_DERIVATIVE
    assert source["precision"].kind == DEQUANTIZED_FP8_DERIVATIVE
    assert source["artifact_sha256"] == tree["manifest"]["derivative"]["artifact_sha256"]
    assert source["artifact_sha256"] != source["parent"]["artifact_sha256"]
    assert source["shard_count"] == 8
    assert [entry["filename"] for entry in source["shards"]] == [
        item["filename"] for item in tree["files"]]
    assert source["parameter_elements"] == tree["config"].geometry.parameter_elements
    assert source["activation_quantized"] is False
    assert source["equivalent_to_official_bf16"] is False
    assert source["full_weight_hash_checked"] is True
    assert source["parent"]["revision"] == "synthetic-parent-revision"
    assert source["parent"]["weight_sha256_checked"] is True


def test_precision_substitution_and_manifest_refusals(tmp_path):
    tree = build_tree(tmp_path)
    config = tree["config"]

    # Neither official identity can label the derivative config.
    with pytest.raises(Spark2_5ContractError, match="cannot be labelled official_bf16"):
        official_bf16_precision(config)
    with pytest.raises(Spark2_5ContractError, match="requires its explicit derivative manifest"):
        classify_precision(config)
    with pytest.raises(Spark2_5ContractError):
        official_native_fp8_precision(config)
    with pytest.raises(Spark2_5ContractError, match="must not keep float-quantized"):
        dequantized_fp8_derivative(
            load_spark2_5_config(official_quant_config_raw(),
                                 require_official_geometry=True),
            parent=tree["parent_identity"], artifact_sha256="c" * 64,
            dequantization_sha256="d" * 64, activation_semantics=SEMANTICS)

    # A hand-labelled official FP8 identity is refused by the same binding.
    fp8_labelled = PrecisionIdentity(
        kind="official_native_fp8", storage="fp8_compressed_tensors",
        weight_dtype="F8_E4M3",
        activation="dynamic symmetric per-token F8_E4M3 (observer-free at runtime)",
        quantization_method="compressed-tensors", quantization_version="0.16.0",
        quantization_fingerprint=tree["parent_identity"].quantization_fingerprint,
        artifact_sha256=tree["parent_identity"].artifact_sha256,
        dequantized=False, activation_quantized=True,
        quantized_modules=tree["parent_identity"].quantized_modules,
        unquantized_modules=tree["parent_identity"].unquantized_modules)
    fp8_refusal = evaluate_derivative_p1_spine(fp8_labelled, config, device="mps")
    assert fp8_refusal["admitted"] is False
    assert any("precision_substitution_refused" in reason
               for reason in fp8_refusal["reasons"])

    # The adapter refuses an official BF16 precision on a derivative config and
    # a derivative precision on an official config.
    base = TinySparkCausalLM(config)
    official_precision = official_bf16_precision(
        load_spark2_5_config(tiny_config_raw(), require_official_geometry=False),
        artifact_sha256="a" * 64)
    with pytest.raises(SparkAdapterError, match="cannot load this dequantized"):
        Spark2_5Adapter(base, config, official_precision, device="cpu",
                        require_official_geometry=False)
    official_config = load_spark2_5_config(tiny_config_raw(),
                                           require_official_geometry=False)
    with pytest.raises(SparkAdapterError, match="requires config.replica_precision"):
        Spark2_5Adapter(TinySparkCausalLM(official_config), official_config,
                        tree["precision"], device="cpu",
                        require_official_geometry=False)

    # A derivative manifest cannot reuse the parent artifact.
    manifest = tree["manifest"]
    manifest["derivative"]["artifact_sha256"] = manifest["source"]["artifact_sha256"]
    _rewrite_manifest(tree)
    with pytest.raises(ProtocolError, match="artifact|reused the parent"):
        validate_derivative_source(tree["repository"], tmp_path, full_hash=False,
                                   require_official_geometry=False)


def test_changed_shard_activation_semantics_and_revision_refused(tmp_path):
    tree = build_tree(tmp_path / "shard")
    shard = tree["derivative_dir"] / tree["files"][0]["filename"]
    shard.write_bytes(b"X" * shard.stat().st_size)
    with pytest.raises(ProtocolError, match="SHA256 changed"):
        validate_derivative_source(tree["repository"], tmp_path / "shard",
                                   full_hash=True, require_official_geometry=False)

    tree = build_tree(tmp_path / "semantics")
    tree["manifest"]["derivative"]["activation_semantics"] = "unquantized but claimed FP8"
    _rewrite_manifest(tree)
    with pytest.raises(ProtocolError, match="activation semantics"):
        validate_derivative_source(tree["repository"], tmp_path / "semantics",
                                   full_hash=False, require_official_geometry=False)

    tree = build_tree(tmp_path / "revision")
    tree["manifest"]["source"]["revision"] = "0" * 40
    _rewrite_manifest(tree)
    with pytest.raises(ProtocolError, match="parent revision"):
        validate_derivative_source(tree["repository"], tmp_path / "revision",
                                   full_hash=False, require_official_geometry=False)

    tree = build_tree(tmp_path / "dequantizer")
    tree["manifest"]["derivative"]["dequantization_sha256"] = "e" * 64
    _rewrite_manifest(tree)
    with pytest.raises(ProtocolError, match="dequantizer script hash"):
        validate_derivative_source(tree["repository"], tmp_path / "dequantizer",
                                   full_hash=False, require_official_geometry=False)

    tree = build_tree(tmp_path / "quantized")
    raw = copy.deepcopy(tree["config_raw"])
    raw["quantization_config"] = official_quant_config_raw()["quantization_config"]
    quantized_sha = _write_json(tree["derivative_dir"] / "config.json", raw)
    tree["manifest"]["derivative"]["config_sha256"] = quantized_sha
    _rewrite_manifest(tree)
    with pytest.raises(ProtocolError, match="implicit_fp8_dequantization"):
        validate_derivative_source(tree["repository"], tmp_path / "quantized",
                                   full_hash=False, require_official_geometry=False)


def test_manifest_rejects_extra_or_missing_shards(tmp_path):
    tree = build_tree(tmp_path)
    (tree["derivative_dir"] / "model-00009-of-00008.safetensors").write_bytes(b"extra")
    with pytest.raises(ProtocolError, match="shard inventory"):
        validate_derivative_source(tree["repository"], tmp_path, full_hash=False,
                                   require_official_geometry=False)

    tree = build_tree(tmp_path / "missing")
    (tree["derivative_dir"] / tree["files"][0]["filename"]).unlink()
    with pytest.raises(ProtocolError, match="shard inventory"):
        validate_derivative_source(tree["repository"], tmp_path / "missing",
                                   full_hash=False, require_official_geometry=False)


def test_current_p1_spine_admission_and_official_substitution_refusal(tmp_path):
    tree = build_tree(tmp_path, official_geometry=True)
    admission = evaluate_derivative_p1_spine(
        tree["precision"], tree["config"], device="mps",
        spine_sites=(3, 7), spine_slots=8, memory_size=256, heads=4)
    assert admission["admitted"] is True, admission["reasons"]
    assert admission["p1"]["active_experts"] == 1
    assert admission["p1"]["frozen_original_backbone"] is True
    assert admission["p1"]["optimizer"] == "NativeFusedAdamW"
    assert admission["spine"]["sites"] == [3, 7]
    assert admission["spine"]["candidate_late_sites"] == [23, 31]
    assert admission["adapter_kwargs"]["spine_sites"] == [3, 7]
    assert admission["execution_format"] == "dequantized_fp8_derivative_bf16_storage"
    assert admission["backend_admission"]["fallback_allowed"] is False

    official_precision = official_bf16_precision(
        load_spark2_5_config(official_config_raw(), require_official_geometry=True),
        artifact_sha256="a" * 64)
    refused = evaluate_derivative_p1_spine(
        official_precision, tree["config"], device="mps")
    assert refused["admitted"] is False
    assert any("precision_substitution_refused" in reason
               for reason in refused["reasons"])

    late = evaluate_derivative_p1_spine(
        tree["precision"], tree["config"], device="mps", spine_sites=(23, 31))
    assert late["admitted"] is False
    assert any("late_sites_require" in reason for reason in late["reasons"])

    tiny = build_tree(tmp_path / "tiny", official_geometry=False)
    short = evaluate_derivative_p1_spine(
        tiny["precision"], tiny["config"], device="cpu",
        spine_sites=(3, 7), spine_slots=8, memory_size=256, heads=4)
    assert short["admitted"] is False
    assert any("current_p1_requires_36_layer_source" in reason
               for reason in short["reasons"])


def test_derivative_p1_zero_init_parity_gradients_update_and_reload(tmp_path):
    tree = build_tree(tmp_path / "p1")
    config, precision = tree["config"], tree["precision"]
    base = TinySparkCausalLM(config)
    adapter = Spark2_5Adapter(
        base, config, precision, device="cpu", require_official_geometry=False,
        projector_layer=3, memory_size=4, heads=2, spine_sites=(3, 7),
        spine_slots=4)
    adapter.seal_original_identity()
    ids, mask = ids_and_mask()
    with torch.no_grad():
        base_logits = base(input_ids=ids, attention_mask=mask, use_cache=False).logits
        adapter_logits = adapter.forward_text(ids, mask)
    assert torch.equal(adapter_logits, base_logits)
    assert adapter.precision.kind == DEQUANTIZED_FP8_DERIVATIVE
    assert adapter.admission.admitted is True
    assert adapter.base_unchanged() is True

    experiment = SpineMemoryExperiment(
        spec=adapter.spine_spec, core=adapter.extension.spine,
        bank_experts=1, bank_hidden_width=8, seed=7, read_records=3,
        scope="derivative-fixture")
    experiment.bank.lr = 1e-2
    path = SparkTextMemoryPath(adapter, experiment, response_lr=1e-2)
    open_memory_ports(path)
    populate_contextual_evidence(path, RECORDS)
    labels = labels_from_ids(ids)
    query = prepare_arm_query(
        path, arm="p_only", question="which token follows the prefix",
        input_ids=ids, attention_mask=torch.ones_like(ids), expert_id=0)
    port_optimizer = build_port_optimizer(path, lr=1e-2)
    assert isinstance(port_optimizer, NativeFusedAdamW)
    assert isinstance(path.experiment.bank.optimizer, NativeFusedAdamW)
    result = direct_response_ce_step(
        path, query, labels, event_id="derivative-direct-ce-0",
        port_optimizer=port_optimizer)

    assert math.isfinite(result["loss"]["before"])
    assert all(math.isfinite(value) for value in result["gradients"]["norms"].values())
    assert result["actual_parameter_update"] is True
    assert set(result["changed_parameter_names"]) <= {
        "gate", "up", "down"}
    assert result["moments"]["changed_keys"]
    assert result["clock"]["eligible_events_advanced"] is True
    assert result["clock"]["revision_advanced"] is True
    assert result["optimizer_ownership"]["single_p_owner"] is True
    assert result["port_optimizer"]["stepped"] is True
    assert result["port_optimizer"]["physical_steps"] == 1
    assert result["port_optimizer"]["changed_parameter_names"]
    assert all(name.startswith("extension.spine.ports.")
               for name in result["port_optimizer"]["changed_parameter_names"])
    assert result["non_port_spine_parameter_changes"] == []
    assert adapter.base_unchanged() is True
    assert result["native_execution"] == "not_run"

    export_dir = tmp_path / "export"
    manifest = adapter.save_extension(export_dir)
    assert manifest["contains_base_weights"] is False
    assert manifest["base_saved"] is False
    assert manifest["optimizer_saved"] is False
    assert manifest["export"]["base"]["precision_kind"] == DEQUANTIZED_FP8_DERIVATIVE

    fresh_base = TinySparkCausalLM(config)
    reloaded = Spark2_5Adapter(
        fresh_base, config, precision, device="cpu", require_official_geometry=False,
        projector_layer=3, memory_size=4, heads=2, spine_sites=(3, 7),
        spine_slots=4)
    reloaded.seal_original_identity()
    receipt = reloaded.load_extension(export_dir)
    assert receipt["extension_only"] is True
    assert receipt["fresh_process_comparison_required"] is True
    assert receipt["loaded_tensors"] == len(list(reloaded.new_named_parameters()))


def test_protocol_config_refuses_mutated_substitution_and_guards():
    config = json.loads(CONFIG_PATH.read_text())

    mutated = copy.deepcopy(config)
    mutated["active_direction"]["source_baseline"] = "official_fp8"
    with pytest.raises(ProtocolError, match="active baseline"):
        validate_config(mutated, ROOT, full_hash=False)

    mutated = copy.deepcopy(config)
    mutated["source_identities"]["dequantized_fp8_derivative"][
        "substitution_allowed"] = True
    with pytest.raises(ProtocolError, match="substitution is not allowed"):
        validate_config(mutated, ROOT, full_hash=False)

    mutated = copy.deepcopy(config)
    mutated["runtime"]["spine_sites"] = [11, 23]
    with pytest.raises(ProtocolError, match="spine sites"):
        validate_config(mutated, ROOT, full_hash=False)

    mutated = copy.deepcopy(config)
    mutated["work_and_cost"]["guards"]["driver_limit_bytes"] = 1
    with pytest.raises(ProtocolError, match="resource guard"):
        validate_config(mutated, ROOT, full_hash=False)

    mutated = copy.deepcopy(config)
    mutated["derivative_contract"]["required_checks"]["zero_init_parity"] = False
    with pytest.raises(ProtocolError, match="required derivative check"):
        validate_config(mutated, ROOT, full_hash=False)

    mutated = copy.deepcopy(config)
    mutated["classification"]["official_bf16_substitution"] = True
    with pytest.raises(ProtocolError, match="official_bf16_substitution"):
        validate_config(mutated, ROOT, full_hash=False)


def test_review_preflight_is_preparation_only_and_refuses_launch(tmp_path):
    report = preflight(CONFIG_PATH, full_hash=False)
    assert report["status"] == "review_ready_preparation_only"
    assert report["launch_allowed"] is False
    assert report["native_command"] is None
    assert report["native_execution"] is False
    assert report["model_loaded"] is False
    assert report["tokenizer_loaded"] is False
    assert report["labels_read"] is False
    assert report["downloads"] == 0
    assert report["admission"]["admitted"] is True
    assert report["source"]["shard_count"] == 8
    assert "external_derivative_authorization" in report["missing_bindings"]
    assert "implemented_derivative_runner" not in report["missing_bindings"]
    assert "protocol_commit" in report["missing_bindings"]
    assert report["missing_bindings"] == [
        "protocol_commit", "external_derivative_authorization"]
    assert report["runner"]["implemented"] is True
    assert report["runner"]["hash_matches"] is True
    assert report["authorization"]["present"] is False

    from preflight_spark_persistent_learner_derivative_bf16 import guarded_command
    command = guarded_command(
        json.loads(CONFIG_PATH.read_text()), tmp_path / "native_output",
        tmp_path / "authorization.json")
    assert "--driver-limit-gib 50.0" in command
    assert "--swap-growth-limit-mib 512.0" in command
    assert "--minimum-free-percent 15.0" in command
    assert "--sustained-host-samples 3" in command
    assert "--entry-point run_spark_persistent_learner_derivative_bf16.py" in command
    assert "PYTORCH_ENABLE_MPS_FALLBACK=0" in command

    with pytest.raises(ProtocolError, match="refused before model load") as refusal:
        preflight(CONFIG_PATH, full_hash=False, launch=True)
    assert "external_derivative_authorization" in refusal.value.detail
    assert "protocol_commit" in refusal.value.detail
    assert "implemented_derivative_runner" not in refusal.value.detail

    fabricated = tmp_path / "derivative_authorization.json"
    fabricated.write_text(json.dumps({
        "authorization_id": "not-a-real-receipt",
        "protocol_id": "spark_persistent_learner_derivative_bf16_v1",
    }))
    with pytest.raises(ProtocolError, match="refused before model load"):
        preflight(CONFIG_PATH, authorization=fabricated,
                  full_hash=False, launch=True)



def test_cpu_smoke_entrypoint_runs_and_refuses_existing_output(tmp_path):
    import os
    import subprocess

    output = tmp_path / "smoke"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_spark_persistent_learner_derivative_bf16.py"),
        "--backend", "cpu_synthetic_derivative_fixture",
        "--stage", "smoke", "--device", "cpu",
        "--output", str(output), "--skip-full-hash",
    ]
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}:{ROOT / 'tests'}",
    }
    completed = subprocess.run(command, cwd=ROOT, env=environment,
                               text=True, capture_output=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout)
    assert report["format"] == "replica.spark_persistent_learner_derivative.smoke.v1"
    assert report["status"] == "smoke_passed"
    assert report["native_execution"] is False
    assert report["precision_kind"] == DEQUANTIZED_FP8_DERIVATIVE
    assert report["execution_format"] == "dequantized_fp8_derivative_bf16_storage"
    checks = report["checks"]
    assert checks["zero_init_parity_exact_logits"] is True
    assert checks["finite_gradients_into_P"] is True
    assert checks["finite_gradients_into_reader_ports"] is True
    assert checks["native_fused_adamw_updates"] == 2
    assert checks["moments_changed_keys"] > 0
    assert checks["learning_clock_advanced"] is True
    assert checks["frozen_base_integrity"] is True
    assert checks["extension_only_reload"] is True
    assert checks["extension_reload_logits_exact"] is True
    assert checks["official_fp8_or_bf16_substitution_refused"] is True
    assert checks["stale_hashes_refused"] is True
    assert set(report["substitution_refusals"].values()) == {"refused"}
    assert set(report["stale_hash_refusals"].values()) == {"refused"}
    assert report["site_review"]["sites"] == [3, 7]
    assert "deferred" in report["late_sites"]
    written = json.loads((output / "smoke_report.json").read_text())
    assert written["status"] == "smoke_passed"

    rerun = subprocess.run(command, cwd=ROOT, env=environment,
                           text=True, capture_output=True, timeout=300)
    assert rerun.returncode == 2
    assert "output must be fresh" in rerun.stdout


def test_native_path_refuses_before_any_model_load(tmp_path, monkeypatch):
    import run_spark_persistent_learner_derivative_bf16 as runner

    loaded = {"count": 0}

    def forbidden_load(_source_root):
        loaded["count"] += 1
        raise AssertionError("native model load must not occur before authorization")

    monkeypatch.setattr(runner, "_load_derivative_model", forbidden_load)
    output = tmp_path / "native"
    with pytest.raises(runner.Refused, match="refused before model load") as refusal:
        runner.run_derivative_protocol(
            CONFIG_PATH, ROOT / "data/spark_persistent_learner_v2", output,
            backend_kind=runner.NATIVE_LABEL, device="mps", stage="all", full_hash=False)
    assert "external_derivative_authorization" in refusal.value.detail
    assert "protocol_commit" in refusal.value.detail
    assert loaded["count"] == 0
    assert not output.exists()

    fabricated = tmp_path / "derivative_authorization.json"
    fabricated.write_text(json.dumps({
        "authorization_id": "not-a-real-derivative-receipt",
        "protocol_id": "spark_persistent_learner_derivative_bf16_v1",
        "execution_format": "dequantized_fp8_derivative_bf16_storage",
    }) + "\n")
    with pytest.raises(runner.Refused, match="refused before model load"):
        runner.run_derivative_protocol(
            CONFIG_PATH, ROOT / "data/spark_persistent_learner_v2", output,
            backend_kind=runner.NATIVE_LABEL, device="mps", stage="all",
            authorization_path=fabricated, full_hash=False)
    assert loaded["count"] == 0
    assert not output.exists()


def test_runner_binding_and_stale_hashes_are_refused(tmp_path):
    from preflight_spark_persistent_learner_derivative_bf16 import (
        _missing_bindings, runner_binding,
    )
    import run_spark_persistent_learner_derivative_bf16 as runner

    config = json.loads(CONFIG_PATH.read_text())
    binding = runner_binding(config, ROOT)
    assert binding["implemented"] is True
    assert binding["hash_matches"] is True
    assert binding["status"] == "implemented"
    assert config["launch"]["runner"] == binding["relative"]
    assert config["launch"]["implementation_sha256"][binding["relative"]] == binding["sha256"]

    mutated = copy.deepcopy(config)
    mutated["launch"]["implementation_sha256"][binding["relative"]] = "0" * 64
    with pytest.raises(ProtocolError, match="implementation hashes differ"):
        validate_config(mutated, ROOT, full_hash=False)

    mutated = copy.deepcopy(config)
    mutated["launch"]["runner_status"] = "absent_derivative_orchestration"
    report = validate_config(mutated, ROOT, full_hash=False)
    missing = _missing_bindings(mutated, report, None, ROOT)
    assert "implemented_derivative_runner" in missing
    assert "external_derivative_authorization" in missing

    fabricated = tmp_path / "fabricated_authorization.json"
    fabricated.write_text(json.dumps({
        "authorization_id": "fabricated-not-external",
        "protocol_id": "spark_persistent_learner_derivative_bf16_v1",
        "source_identity": "dequantized_fp8_derivative",
        "execution_format": "dequantized_fp8_derivative_bf16_storage",
    }) + "\n")
    with pytest.raises(runner.Refused, match="refused before model load"):
        runner.verify_derivative_authorization(
            runner.DerivativeProtocol.load(CONFIG_PATH, full_hash=False),
            tmp_path / "fresh_output", fabricated)

    packet = json.loads(
        (ROOT / "reports/spark_persistent_learner_v2/derivative_bf16_packet.json").read_text())
    implemented = packet["implemented"]["implemented_derivative_runner"]
    assert implemented["status"] == "implemented"
    assert implemented["path"] == binding["relative"]
    assert implemented["sha256"] == binding["sha256"]


def test_late_site_comparison_requires_a_completed_matched_early_result():
    import run_spark_persistent_learner_derivative_bf16 as runner

    assert runner.review_site_request((3, 7))["late_comparison"] == "not_requested"
    with pytest.raises(runner.Refused, match="explicit comparison"):
        runner.review_site_request((23, 31))
    with pytest.raises(runner.Refused, match="completed early 3/7"):
        runner.review_site_request((23, 31), compare_late=True)
    early = {"status": "complete", "sites": [3, 7],
             "source": {"artifact_sha256": "c" * 64}}
    with pytest.raises(runner.Refused, match="same 36-layer derivative source"):
        runner.review_site_request((23, 31), early_report=early,
                                   artifact_sha256="d" * 64, compare_late=True)
    justified = runner.review_site_request(
        (23, 31), early_report=early, artifact_sha256="c" * 64, compare_late=True)
    assert justified["late_comparison"] == "matched_same_source_after_early_result"
    with pytest.raises(runner.Refused, match="neither early"):
        runner.review_site_request((11, 23))
