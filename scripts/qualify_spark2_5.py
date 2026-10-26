#!/usr/bin/env python3
"""Qualify the frozen-base Spark2_5 adapter on one exclusive MPS worker.

Stages: ``load``, ``inference``, ``input_gradients``, ``single_update`` and
``export_reload``. The runner refuses non-MPS execution, CPU fallback, online
access, missing or changed source pins, native FP8 on MPS, unlabelled FP8
substitution and any derivative without a complete manifest.

The export stage writes only extension tensors and a base identity. It then
releases the model and starts a *fresh process* that reloads the sealed source
and extension, comparing fixed-case logits and greedy token IDs. No base
checkpoint is saved, no heldout role is opened and no training capability is
claimed.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab.architecture.spark2_5 import (  # noqa: E402
    DEQUANTIZED_FP8_DERIVATIVE, OFFICIAL_BF16, OFFICIAL_NATIVE_FP8, THINKING_MODES,
    PrecisionIdentity, admit_load, canonical_json, dequantized_fp8_derivative,
    fingerprint, load_spark2_5_config, load_tokenizer_identity,
    official_bf16_precision, official_native_fp8_precision,
    sha256_bytes,
)
from dequantize_spark_fp8 import (  # noqa: E402
    DequantizationError, artifact_digest, file_sha256, verify_source_pins,
)

QUALIFICATION_SCHEMA = "replica.spark2_5_qualification.v1"
PLAN_SCHEMA = "replica.spark2_5_qualification_plan.v1"
STAGES = ("load", "inference", "input_gradients", "single_update", "export_reload")
OFFICIAL_ROLES = ("official_bf16_control", "official_fp8_adaptation_base")
DERIVATIVE_ROLE = "dequantized_fp8_derivative"
SHA256_LENGTH = 64


class QualificationError(RuntimeError):
    """A declared precondition or acceptance check failed."""


# ---------------------------------------------------------------------------
# Pure plan/environment validation (unit-testable without tokenizers or weights)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_plan(path: str | Path) -> dict:
    plan = read_json(path)
    validate_plan(plan)
    return plan


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        raise QualificationError(f"{name} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise QualificationError(f"{name} must be a lowercase SHA256")
    return value


def validate_plan(plan: Mapping) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise QualificationError(f"plan schema must be {PLAN_SCHEMA}")
    environment = plan.get("environment")
    if not isinstance(environment, Mapping) or environment.get("device") != "mps":
        raise QualificationError("the local plan must declare device=mps")
    if environment.get("cpu_fallback") != "disabled":
        raise QualificationError("CPU fallback must be disabled")
    if environment.get("pytorch_enable_mps_fallback") != "0":
        raise QualificationError("PYTORCH_ENABLE_MPS_FALLBACK must be pinned to 0")
    sources = plan.get("sources")
    if not isinstance(sources, Mapping):
        raise QualificationError("sources must be an object")
    required = set(OFFICIAL_ROLES) | {DERIVATIVE_ROLE}
    if set(sources) != required:
        raise QualificationError(f"plan source roles must be exactly {sorted(required)}")
    for role in OFFICIAL_ROLES:
        entry = sources[role]
        revision = entry.get("revision")
        if not isinstance(revision, str) or len(revision) != 40:
            raise QualificationError(f"{role}.revision must be a full 40-character SHA")
        _require_sha(entry.get("config_sha256"), f"{role}.config_sha256")
        shards = entry.get("shards")
        if not isinstance(shards, list) or not shards:
            raise QualificationError(f"{role}.shards must be nonempty")
        names = [shard.get("filename") for shard in shards]
        if len(set(names)) != len(names):
            raise QualificationError(f"{role} has duplicate shard names")
        for shard in shards:
            _require_sha(shard.get("sha256"), f"{role}.{shard.get('filename')}.sha256")
            if type(shard.get("bytes")) is not int or shard["bytes"] <= 0:
                raise QualificationError(f"{role} shard bytes must be positive")
        if entry.get("parameter_elements") != 4_112_079_360:
            raise QualificationError(f"{role} must declare the official 4,112,079,360 parameters")
        if not isinstance(entry.get("required_assets"), list):
            raise QualificationError(f"{role}.required_assets must be a list")
        asset_sha256 = entry.get("asset_sha256")
        if not isinstance(asset_sha256, Mapping):
            raise QualificationError(f"{role}.asset_sha256 must pin remote-code/tokenizer bytes")
        for name in ("configuration_spark.py", "modeling_spark.py", "tokenizer.json"):
            _require_sha(asset_sha256.get(name), f"{role}.asset_sha256.{name}")
    if sources["official_fp8_adaptation_base"].get("storage") != "fp8_compressed_tensors":
        raise QualificationError("the FP8 role must keep official compressed-tensors storage")
    derivative = sources[DERIVATIVE_ROLE]
    if derivative.get("parent") != "official_fp8_adaptation_base":
        raise QualificationError("derivative parent must be the official FP8 role")
    if derivative.get("storage") != "bf16" or derivative.get("max_derivatives") != 1:
        raise QualificationError("exactly one BF16 derivative is authorized")
    if derivative.get("equivalent_to_bf16_release") is not False:
        raise QualificationError("a derivative never equals the BF16 release")
    if not isinstance(derivative.get("activation_semantics"), str):
        raise QualificationError("derivative activation semantics must be explicit")
    derivative_assets = derivative.get("asset_sha256")
    if not isinstance(derivative_assets, Mapping):
        raise QualificationError("derivative asset_sha256 must pin copied code/tokenizer bytes")
    for name in ("configuration_spark.py", "modeling_spark.py", "tokenizer.json"):
        _require_sha(derivative_assets.get(name), f"derivative.asset_sha256.{name}")
    prompts = plan.get("qualification", {}).get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise QualificationError("a frozen prompt set is required")
    if len({item.get("id") for item in prompts}) != len(prompts):
        raise QualificationError("prompt ids must be unique")
    if plan["qualification"].get("access_order") != [
            "frozen_prompt_set", "development_smoke", "no_heldout_role_materialized"]:
        raise QualificationError("the frozen development/no-heldout access order is required")
    if not str(plan["qualification"].get("heldout_access", "")).startswith("forbidden"):
        raise QualificationError("heldout access must be explicitly forbidden")
    claims = plan.get("claims", {})
    if claims.get("training_claimed") is not False or claims.get("fp8_training_claimed") is not False:
        raise QualificationError("the qualification plan may not claim training")
    storage = plan.get("storage_policy")
    if not isinstance(storage, Mapping):
        raise QualificationError("storage_policy is required")
    if storage.get("max_derivatives") != 1 or storage.get("spark_specific_budget_bytes") != 40 * 1024**3:
        raise QualificationError("the Spark-specific 40 GiB one-derivative policy is required")
    if storage.get("full_base_saves_during_qualification") is not False:
        raise QualificationError("full-base saves are forbidden during qualification")
    official_bytes = sum(shard["bytes"] for role in OFFICIAL_ROLES
                         for shard in sources[role]["shards"])
    if official_bytes != storage.get("official_payload_bytes"):
        raise QualificationError(
            f"official payload bytes {official_bytes} differ from the storage policy")
    if (official_bytes + int(storage["max_derivative_bytes"])
            >= int(storage["spark_specific_budget_bytes"])):
        raise QualificationError("the derivative does not fit the Spark-specific budget")
    if storage.get("delete_prior_evidence") is not False:
        raise QualificationError("prior evidence must be retained")
    extension = plan.get("extension")
    if not isinstance(extension, Mapping) or extension.get("zero_init_output") is not True:
        raise QualificationError("the projector must be zero initialized")
    if extension.get("projector_out_dim") != 2560:
        raise QualificationError("the projector must target the Spark hidden width")
    if extension.get("memory_ports") not in ([11, 23], (11, 23), []):
        raise QualificationError("memory ports must be the 11/23 hypotheses or empty")


def environment_issues(
    plan: Mapping,
    role: str,
    *,
    platform_name: str,
    python_version: str,
    torch_version: str,
    transformers_version: str | None,
    compressed_tensors_version: str | None,
    mps_available: bool,
    environ: Mapping[str, str],
) -> list[str]:
    expected = plan["environment"]
    entry = plan["sources"][role]
    issues: list[str] = []
    if platform_name.lower() != "darwin":
        issues.append("platform is not macOS")
    if not python_version.startswith(str(expected["python"])):
        issues.append(f"python {python_version} does not start with {expected['python']}")
    if torch_version != expected["torch"]:
        issues.append(f"torch {torch_version} != pinned {expected['torch']}")
    if not mps_available:
        issues.append("MPS is unavailable; CPU fallback is forbidden")
    if transformers_version != entry.get("transformers"):
        issues.append(
            f"transformers {transformers_version} != pinned {entry.get('transformers')} for {role}")
    required_compressed = entry.get("compressed_tensors")
    if required_compressed is not None and compressed_tensors_version != required_compressed:
        issues.append(
            f"compressed_tensors {compressed_tensors_version} != pinned {required_compressed}")
    for name, value in expected.get("offline", {}).items():
        if environ.get(name) != value:
            issues.append(f"{name}={environ.get(name)!r} != {value!r}")
    if environ.get("REPLICA_CAPACITY_GUARD") != "1":
        issues.append("REPLICA_CAPACITY_GUARD must be 1")
    if environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        issues.append("PYTORCH_ENABLE_MPS_FALLBACK must be 0")
    return issues


def verify_environment(plan: Mapping, role: str) -> dict:
    import torch

    try:
        compressed = importlib.metadata.version("compressed_tensors")
    except importlib.metadata.PackageNotFoundError:
        compressed = None
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    issues = environment_issues(
        plan, role, platform_name=platform.system(),
        python_version=platform.python_version(), torch_version=str(torch.__version__),
        transformers_version=transformers_version,
        compressed_tensors_version=compressed,
        mps_available=bool(torch.backends.mps.is_available()), environ=os.environ)
    if issues:
        raise QualificationError("wrong_environment: " + "; ".join(issues))
    return {
        "platform": platform.system(), "python": platform.python_version(),
        "torch": str(torch.__version__), "transformers": transformers_version,
        "compressed_tensors": compressed, "device": "mps",
        "mps_available": True, "fallback": "disabled",
        "offline": {name: os.environ.get(name) for name in
                    plan["environment"]["offline"]},
        "attention_implementation": "eager",
        "use_cache": False,
        "worker_count": 1,
    }


def storage_check(plan: Mapping, *, free_bytes: int | None = None,
                  output: Path | None = None) -> dict:
    policy = plan["storage_policy"]
    path = output if output is not None and output.exists() else ROOT
    free = shutil.disk_usage(path).free if free_bytes is None else int(free_bytes)
    if free < int(policy["minimum_free_bytes_before_stage"]):
        raise QualificationError(
            f"insufficient free space: {free} < {policy['minimum_free_bytes_before_stage']}")
    return {"free_bytes_before": free,
            "minimum_free_bytes": policy["minimum_free_bytes_before_stage"],
            "spark_specific_budget_bytes": policy["spark_specific_budget_bytes"],
            "official_payload_bytes": policy["official_payload_bytes"],
            "max_derivative_bytes": policy["max_derivative_bytes"],
            "max_derivatives": policy["max_derivatives"],
            "full_base_saves": False,
            "extension_only_artifacts": True}


# ---------------------------------------------------------------------------
# Source resolution


def _parent_artifact(plan: Mapping) -> str:
    parent = plan["sources"]["official_fp8_adaptation_base"]
    return artifact_digest(parent["shards"])


def resolve_source(plan: Mapping, role: str, source_dir: str | Path) -> dict:
    """Verify pins and return config/identity/evidence; refuse substitutions."""
    source = Path(source_dir)
    if role in OFFICIAL_ROLES:
        entry = plan["sources"][role]
        try:
            files = verify_source_pins(source, entry)
        except DequantizationError as error:
            raise QualificationError(str(error)) from error
        if not (source / "config.json").is_file():
            raise QualificationError("missing_official_payload: config.json is absent")
        config_raw = read_json(source / "config.json")
        config = load_spark2_5_config(config_raw, require_official_geometry=True)
        artifact = artifact_digest(files)
        if role == "official_bf16_control":
            if entry["storage"] != "bf16":
                raise QualificationError("BF16 role is not plain BF16 storage")
            precision = official_bf16_precision(config, artifact_sha256=artifact)
            if "quantization_config" in config_raw:
                raise QualificationError("BF16 source unexpectedly declares quantization")
        else:
            precision = official_native_fp8_precision(config, artifact_sha256=artifact)
            if "quantization_config" not in config_raw:
                raise QualificationError("official FP8 source lost its quantization_config")
            if precision.quantization_fingerprint != entry.get("quantization_fingerprint"):
                raise QualificationError("FP8 quantization fingerprint differs from the plan")
        return {"role": role, "entry": entry, "source_dir": source,
                "files": files, "config_raw": config_raw, "config": config,
                "precision": precision, "parent_identity": precision,
                "derivative_manifest": None}

    if role != DERIVATIVE_ROLE:
        raise QualificationError(f"unknown source role {role!r}")
    entry = plan["sources"][role]
    manifest_path = source / entry["manifest_name"]
    if not manifest_path.is_file():
        raise QualificationError("missing_derivative_manifest: run the single dequantizer first")
    manifest = read_json(manifest_path)
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if manifest.get("fingerprint") != fingerprint(body):
        raise QualificationError("derivative manifest fingerprint mismatch")
    if manifest.get("derivative_index") != entry.get("derivative_index"):
        raise QualificationError("derivative index differs from the single authorized identity")
    source_block = manifest.get("source", {})
    derivative = manifest.get("derivative", {})
    parent = plan["sources"]["official_fp8_adaptation_base"]
    if source_block.get("repo_id") != parent["repo_id"]:
        raise QualificationError("derivative parent repository differs from the plan")
    if source_block.get("revision") != parent["revision"]:
        raise QualificationError("derivative parent revision differs from the plan")
    if source_block.get("config_sha256") != parent["config_sha256"]:
        raise QualificationError("derivative parent config differs from the plan")
    recorded_parent_files = source_block.get("source_files")
    if recorded_parent_files is not None:
        expected_parent_files = {
            item["filename"]: item["sha256"] for item in parent["shards"]}
        observed_parent_files = {
            item.get("filename"): item.get("sha256")
            for item in recorded_parent_files if isinstance(item, dict)}
        if observed_parent_files != expected_parent_files:
            raise QualificationError("derivative parent shard pins differ from the plan")
    dequantizer_path = ROOT / "scripts" / "dequantize_spark_fp8.py"
    if derivative.get("dequantization_sha256") != file_sha256(dequantizer_path):
        raise QualificationError("derivative script SHA256 differs from the checked-in tool")
    parent_artifact = _parent_artifact(plan)
    if source_block.get("artifact_sha256") != parent_artifact:
        raise QualificationError("derivative parent artifact digest differs from pinned shards")
    if derivative.get("derivative_of") != parent_artifact:
        raise QualificationError("derivative_of does not name the pinned FP8 artifact")
    if derivative.get("activation_semantics") != entry["activation_semantics"]:
        raise QualificationError("derivative activation semantics differ from the plan")
    if derivative.get("equivalent_to_bf16_release") is not False:
        raise QualificationError("derivative cannot equal the BF16 release")
    if manifest.get("claims", {}).get("official_native_fp8") is not False:
        raise QualificationError("derivative cannot claim official native FP8")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise QualificationError("derivative manifest declares no shards")
    observed = {path.name for path in source.glob("*.safetensors")}
    declared = {item["filename"] for item in files}
    if observed != declared:
        raise QualificationError("derivative shard inventory differs from its manifest")
    verified = []
    for item in files:
        path = source / item["filename"]
        if not path.is_file():
            raise QualificationError(f"missing derivative shard {item['filename']}")
        if path.stat().st_size != item["bytes"]:
            raise QualificationError(f"{item['filename']}: byte count changed")
        digest = file_sha256(path)
        if digest != item["sha256"]:
            raise QualificationError(f"{item['filename']}: SHA256 changed")
        verified.append({"filename": item["filename"], "bytes": item["bytes"], "sha256": digest})
    artifact = artifact_digest(verified)
    if artifact != derivative.get("artifact_sha256"):
        raise QualificationError("derivative artifact digest differs from its manifest")
    if artifact == parent_artifact:
        raise QualificationError("derivative reused the parent artifact digest")
    for name, expected in entry.get("asset_sha256", {}).items():
        path = source / name
        if not path.is_file():
            raise QualificationError(f"derivative is missing pinned asset: {name}")
        if file_sha256(path) != expected:
            raise QualificationError(f"derivative asset SHA256 differs: {name}")
    config_path = source / "config.json"
    if not config_path.is_file():
        raise QualificationError("derivative config.json is missing")
    if file_sha256(config_path) != derivative.get("config_sha256"):
        raise QualificationError("derivative config bytes differ from its manifest")
    config_raw = read_json(config_path)
    if "quantization_config" in config_raw:
        raise QualificationError("implicit_fp8_dequantization: derivative still declares FP8 storage")
    config = load_spark2_5_config(config_raw, require_official_geometry=True)
    inventory = manifest.get("tensor_inventory", {})
    if inventory.get("parameter_elements") != config.geometry.parameter_elements:
        raise QualificationError("derivative parameter inventory differs from the config")
    if inventory.get("tensor_count") != 290:
        raise QualificationError("derivative tensor count is not the official BF16 inventory")
    parent_dict = source_block.get("parent_precision")
    if not isinstance(parent_dict, Mapping):
        raise QualificationError("derivative manifest lacks parent precision provenance")
    parent_identity = PrecisionIdentity(**dict(parent_dict))
    if parent_identity.artifact_sha256 != parent_artifact:
        raise QualificationError("parent precision identity names another artifact")
    if parent_identity.kind != OFFICIAL_NATIVE_FP8:
        raise QualificationError("derivative parent is not the official native FP8 identity")
    if parent_identity.quantization_fingerprint != parent.get("quantization_fingerprint"):
        raise QualificationError("derivative parent quantization fingerprint differs from the plan")
    precision = dequantized_fp8_derivative(
        config, parent=parent_identity, artifact_sha256=artifact,
        dequantization_sha256=derivative.get("dequantization_sha256"),
        activation_semantics=derivative["activation_semantics"], storage="bf16")
    if precision.kind != DEQUANTIZED_FP8_DERIVATIVE:
        raise QualificationError("derivative precision identity was not admitted")
    if precision.digest != derivative.get("precision_digest"):
        raise QualificationError("derivative precision digest differs from its manifest")
    if precision.equivalent_to_bf16_release:
        raise QualificationError("derivative cannot be labelled the BF16 release")
    return {"role": role, "entry": entry, "source_dir": source,
            "files": verified, "config_raw": config_raw, "config": config,
            "precision": precision, "parent_identity": parent_identity,
            "derivative_manifest": manifest}


# ---------------------------------------------------------------------------
# Native loading


def _tokenizer_payload(source: Path, config, config_raw: Mapping) -> dict:
    tokenizer_config_raw = read_json(source / "tokenizer_config.json")
    chat_template = (source / "chat_template.jinja").read_bytes()
    tokenizer_json = (source / "tokenizer.json").read_bytes()
    identity = load_tokenizer_identity(
        config, tokenizer_config_raw, chat_template, tokenizer_json=tokenizer_json)
    return identity.to_dict()


def load_adapter(plan: Mapping, resolved: Mapping):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = resolved["source_dir"]
    entry = resolved["entry"]
    config = resolved["config"]
    precision = resolved["precision"]
    admission = admit_load(precision, device="mps")
    if not admission.admitted:
        raise QualificationError("native_fp8_on_mps_refused")

    tokenizer = AutoTokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise QualificationError("the pinned tokenizer must be a fast tokenizer")
    tokenizer_identity = _tokenizer_payload(source, config, resolved["config_raw"])
    if tokenizer_identity["vocab_size"] != config.geometry.vocab_size:
        raise QualificationError("tokenizer vocabulary differs from the validated config")
    base = AutoModelForCausalLM.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True)
    if getattr(base.config, "model_type", None) != "spark2_5":
        raise QualificationError("loaded model is not spark2_5")
    attention = getattr(base.config, "_attn_implementation", None)
    if attention not in (None, "eager"):
        raise QualificationError("loaded attention implementation is not eager")
    if len(base.model.layers) != config.geometry.num_hidden_layers:
        raise QualificationError("loaded decoder layer count differs from the config")
    input_embedding = base.get_input_embeddings()
    output_embedding = base.get_output_embeddings()
    if input_embedding is None or output_embedding is None:
        raise QualificationError("tied input/output embeddings are required")
    if output_embedding.weight.data_ptr() != input_embedding.weight.data_ptr():
        raise QualificationError("output embeddings are not tied to the input embedding")
    base.to("mps")
    base.eval()
    parameter_elements = sum(parameter.numel() for parameter in base.parameters())
    if parameter_elements != entry["parameter_elements"]:
        raise QualificationError(
            f"loaded parameter elements {parameter_elements} != {entry['parameter_elements']}")

    from replica_lab.architecture.spark_adapter import Spark2_5Adapter

    extension = plan["extension"]
    adapter = Spark2_5Adapter(
        base, config, precision, device="mps",
        require_official_geometry=True,
        projector_in_dim=int(extension["projector_in_dim"]),
        projector_hidden_dim=int(extension["projector_hidden_dim"]),
        projector_layer=int(extension["projector_layer"]),
        zero_init_output=True, memory_size=int(extension["memory_size"]),
        heads=int(extension["port_heads"]))
    adapter.eval()
    adapter.seal_original_identity()
    inventory = {
        "parameter_elements": parameter_elements,
        "tensor_count": len(list(base.named_parameters())),
        "dtypes": sorted({str(parameter.dtype) for parameter in base.parameters()}),
        "tied_output_embedding": True,
        "attention_implementation": "eager",
        "use_cache": False,
    }
    return {
        "adapter": adapter, "base": base, "tokenizer": tokenizer,
        "tokenizer_identity": tokenizer_identity, "admission": admission.to_dict(),
        "inventory": inventory, "config": config, "precision": precision,
        "torch": torch,
    }


# ---------------------------------------------------------------------------
# Tokenization and generation


def _prompt_ids(tokenizer, text: str, thinking: str, device: str):
    import torch

    mode = THINKING_MODES[thinking]
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, **mode)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    return rendered, ids, mask


def _greedy_generate(adapter, prompt_ids, prompt_mask, max_new_tokens: int,
                     eos_token_id: int | None) -> dict:
    import torch

    started = time.monotonic()
    generated = []
    stop_reason = "max_new_tokens"
    ids = prompt_ids
    mask = prompt_mask
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = adapter.forward_text(ids, mask, use_projector=True, use_ports=False)
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
            ids = torch.cat((ids, torch.tensor([[next_id]], dtype=torch.long,
                                               device=ids.device)), dim=1)
            mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype,
                                               device=mask.device)), dim=1)
            if eos_token_id is not None and next_id == eos_token_id:
                stop_reason = "eos"
                break
    return {"token_ids": generated, "stop_reason": stop_reason,
            "latency_seconds": time.monotonic() - started}


def _reasoning_count(tokenizer, generated_ids: Sequence[int]) -> dict:
    text = tokenizer.decode(list(generated_ids), skip_special_tokens=False)
    start = text.find("<think>")
    end = text.find("</think>")
    if start < 0 or end <= start:
        return {"reasoning_token_count": 0,
                "method": "no completed <think>...</think> block in decoded output"}
    reasoning_text = text[start + len("<think>"):end]
    return {"reasoning_token_count": len(tokenizer(reasoning_text)["input_ids"]),
            "method": "tokenizer count inside decoded <think>...</think>",
            "reasoning_text": reasoning_text}


def _memory_bank(torch, device: str, *, count: int, width: int, dtype):
    memory = torch.linspace(-1.0, 1.0, steps=count * width, dtype=torch.float32,
                            device=device).reshape(1, count, width)
    return memory.to(dtype), torch.ones((1, count), dtype=torch.bool, device=device)


# ---------------------------------------------------------------------------
# Report writing


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _mps_driver_bytes(torch) -> int:
    return int(torch.mps.driver_allocated_memory())


def _result(stage: str, **fields) -> dict:
    return {"stage": stage, "status": "completed", **fields}


# ---------------------------------------------------------------------------
# Fresh-process reload comparison


def run_reload_compare(args) -> int:
    plan = load_plan(args.config)
    output = Path(args.output)
    report_path = Path(args.reload_report)
    report: dict = {"schema": "replica.spark2_5_reload_compare.v1",
                    "fresh_process": True, "pid": os.getpid()}
    try:
        verify_environment(plan, args.source_role)
        resolved = resolve_source(plan, args.source_role, args.source_dir)
        loaded = load_adapter(plan, resolved)
        adapter = loaded["adapter"]
        extension = plan["extension"]
        if extension["memory_ports"]:
            adapter.enable_memory_ports(
                tuple(extension["memory_ports"]),
                memory_size=int(extension["memory_size"]),
                heads=int(extension["port_heads"]))
        receipt = adapter.load_extension(args.extension_dir)
        expected = loaded["torch"].load(args.expected, map_location="cpu",
                                        weights_only=True)
        input_ids = expected["input_ids"].to("mps")
        attention_mask = expected["attention_mask"].to("mps")
        memory = expected["memory"].to("mps")
        memory_mask = expected["memory_mask"].to("mps")
        with loaded["torch"].no_grad():
            logits = adapter.forward_text(input_ids, attention_mask, memory=memory,
                                          memory_mask=memory_mask,
                                          use_projector=True, use_ports=True)
        comparison = compare_logits(logits, expected["logits"],
                                    rtol=float(expected["rtol"]), atol=float(expected["atol"]))
        tokens = logits[0].argmax(dim=-1)
        expected_tokens = expected["expected_tokens"].to(tokens.device)
        exact_tokens = bool(tokens.equal(expected_tokens))
        report.update({
            "status": "completed",
            "source": {"role": args.source_role,
                       "files": resolved["files"],
                       "precision_kind": resolved["precision"].kind,
                       "artifact_sha256": resolved["precision"].artifact_sha256},
            "extension_receipt": receipt,
            "comparison": comparison,
            "exact_greedy_tokens": exact_tokens,
            "logits_finite": bool(loaded["torch"].isfinite(logits).all()),
            "base_identity_verified": True,
            "environment": {"torch": str(loaded["torch"].__version__),
                            "device": "mps", "fallback": "disabled"},
        })
        if not comparison["within_tolerance"]:
            raise QualificationError("fresh-process logits differ beyond tolerance")
        if not exact_tokens:
            raise QualificationError("fresh-process greedy tokens differ")
        write_report(report_path, report)
        return 0
    except BaseException as error:
        report.update({"status": "failed", "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc()})
        write_report(report_path, report)
        return 1


def compare_arms(plan: Mapping, baseline: Mapping, derivative: Mapping) -> dict:
    """Pair BF16 and derivative arm evidence; bitwise equality is not required."""
    baseline_role = baseline.get("source", {}).get("role")
    derivative_role = derivative.get("source", {}).get("role")
    if baseline_role != "official_bf16_control":
        raise QualificationError("baseline report is not the official BF16 control")
    if derivative_role != DERIVATIVE_ROLE:
        raise QualificationError("comparison arm is not the labelled derivative")
    for name, report in (("baseline", baseline), ("derivative", derivative)):
        if report.get("status") != "completed":
            raise QualificationError(f"{name} arm did not complete")
        if report.get("claims", {}).get("capability_claim") != "none":
            raise QualificationError(f"{name} arm claims a capability")
        if report.get("claims", {}).get("heldout_accessed") is not False:
            raise QualificationError(f"{name} arm accessed heldout data")
    baseline_generations = baseline.get("stages", {}).get("inference", {}).get("generations")
    derivative_generations = derivative.get("stages", {}).get("inference", {}).get("generations")
    if not isinstance(baseline_generations, list) or not isinstance(derivative_generations, list):
        raise QualificationError("both arms need inference generation evidence")

    def index(generations, name):
        result = {}
        for row in generations:
            key = (row.get("prompt_id"), row.get("thinking"))
            if key in result:
                raise QualificationError(f"{name} has duplicate generation rows {key}")
            result[key] = row
        return result

    baseline_index = index(baseline_generations, "baseline")
    derivative_index = index(derivative_generations, "derivative")
    if set(baseline_index) != set(derivative_index):
        raise QualificationError("the arms did not run the same frozen prompt set")
    rows = []
    agreements = 0
    for key in sorted(baseline_index):
        left = baseline_index[key]
        right = derivative_index[key]
        if left.get("rendered_prompt_sha256") != right.get("rendered_prompt_sha256"):
            raise QualificationError(f"prompt template/tokenization differs for {key}")
        same_tokens = left.get("token_ids") == right.get("token_ids")
        agreements += int(same_tokens)
        rows.append({
            "prompt_id": key[0], "thinking": key[1],
            "prompt_token_count": left.get("prompt_token_count"),
            "baseline_token_ids": left.get("token_ids"),
            "derivative_token_ids": right.get("token_ids"),
            "exact_token_match": same_tokens,
            "baseline_token_count": len(left.get("token_ids") or []),
            "derivative_token_count": len(right.get("token_ids") or []),
            "baseline_reasoning_token_count": left.get("reasoning_token_count"),
            "derivative_reasoning_token_count": right.get("reasoning_token_count"),
            "baseline_stop_reason": left.get("stop_reason"),
            "derivative_stop_reason": right.get("stop_reason"),
            "baseline_latency_seconds": left.get("latency_seconds"),
            "derivative_latency_seconds": right.get("latency_seconds"),
            "baseline_decoded_text_sha256": sha256_bytes(
                str(left.get("decoded_text", "")).encode()),
            "derivative_decoded_text_sha256": sha256_bytes(
                str(right.get("decoded_text", "")).encode()),
        })
    return {
        "schema": "replica.spark2_5_arm_comparison.v1",
        "status": "completed",
        "baseline": {
            "role": baseline_role,
            "artifact_sha256": baseline.get("source", {}).get("artifact_sha256"),
            "precision_kind": baseline.get("source", {}).get("precision", {}).get("kind"),
            "declared_budgets": plan["qualification"]["declared_generation_budgets"],
            "executed_budget": plan["qualification"]["smoke_max_new_tokens"],
        },
        "derivative": {
            "role": derivative_role,
            "artifact_sha256": derivative.get("source", {}).get("artifact_sha256"),
            "precision_kind": derivative.get("source", {}).get("precision", {}).get("kind"),
            "activation_semantics": derivative.get("source", {}).get("precision", {}).get(
                "activation"),
            "declared_budgets": plan["qualification"]["declared_generation_budgets"],
            "executed_budget": plan["qualification"]["smoke_max_new_tokens"],
        },
        "comparison": {
            "non_bitwise": True,
            "note": ("Paired smoke rows only; no accuracy tolerance, 512/2048 budget or "
                     "capability claim is established."),
            "paired_rows": len(rows),
            "exact_token_agreement_count": agreements,
            "all_prompts_paired": True,
        },
        "rows": rows,
        "claims": {
            "capability_claim": "none",
            "training_claimed": False,
            "heldout_accessed": False,
            "declared_budgets_executed": False,
        },
    }


def run_compare(args) -> int:
    plan = load_plan(args.config)
    baseline = read_json(args.baseline_report)
    derivative = read_json(args.derivative_report)
    comparison = compare_arms(plan, baseline, derivative)
    comparison["fingerprint"] = fingerprint(comparison)
    Path(args.comparison_report).parent.mkdir(parents=True, exist_ok=True)
    write_report(Path(args.comparison_report), comparison)
    print(json.dumps({"status": comparison["status"],
                      "report": str(args.comparison_report)}))
    return 0


def compare_logits(actual, expected, *, rtol: float, atol: float) -> dict:
    import torch

    left = actual.detach().float().cpu()
    right = expected.detach().float().cpu()
    if left.shape != right.shape:
        raise QualificationError("logit shapes differ")
    difference = (left - right).abs()
    return {
        "rtol": rtol, "atol": atol,
        "max_abs_logit_difference": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs_logit_difference": float(difference.mean()) if difference.numel() else 0.0,
        "within_tolerance": bool(torch.allclose(left, right, rtol=rtol, atol=atol)),
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
    }


# ---------------------------------------------------------------------------
# Main qualification flow


def run_qualification(args) -> int:
    plan = load_plan(args.config)
    source_role = args.source_role
    if source_role not in plan["sources"]:
        raise QualificationError(f"unknown source role {source_role!r}")
    output = Path(args.output)
    if output.exists():
        raise QualificationError("use a fresh output path; existing bytes are never overwritten")
    storage = storage_check(plan, output=output.parent if output.parent.exists() else None)
    resolved = resolve_source(plan, source_role, args.source_dir)
    environment = verify_environment(plan, source_role)

    import torch
    from replica_lab.optimizers import build_optimizer
    from replica_lab.profiling import RunProfiler

    output.mkdir(parents=True, exist_ok=False)
    profiler = RunProfiler("mps", output).start()
    report_path = output / "qualification_report.json"
    report: dict = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "started",
        "source": {
            "role": source_role,
            "entry": {key: resolved["entry"][key] for key in
                      ("repo_id", "revision", "storage", "precision_kind", "transformers")},
            "files": resolved["files"],
            "artifact_sha256": resolved["precision"].artifact_sha256,
            "config_sha256": file_sha256(resolved["source_dir"] / "config.json"),
            "quantization_fingerprint": resolved["precision"].quantization_fingerprint,
            "derivative_manifest": resolved["derivative_manifest"],
            "precision": resolved["precision"].to_dict(),
        },
        "environment": environment,
        "policy": plan["resources"],
        "declared_stop_after": args.stop_after,
        "stages": {},
        "storage": storage,
        "native_execution": False,
        "claims": {
            "weights_downloaded": False,
            "remote_code_executed": "sealed local source only",
            "fallback_used": False,
            "dequantization_performed": source_role == DERIVATIVE_ROLE,
            "native_execution": True,
            "training_claimed": False,
            "fp8_training_claimed": False,
            "capability_claim": "none",
            "heldout_accessed": False,
        },
    }
    write_report(report_path, report)
    loaded = None
    try:
        loaded = load_adapter(plan, resolved)
        report["native_execution"] = True
        report["claims"]["native_execution"] = True
        adapter = loaded["adapter"]
        tokenizer = loaded["tokenizer"]
        base = loaded["base"]
        torch.manual_seed(int(plan["qualification"]["single_update"]["seed"]))
        report["geometry"] = resolved["config"].geometry.to_dict()
        report["admission"] = loaded["admission"]
        report["loaded_inventory"] = loaded["inventory"]
        report["tokenizer"] = loaded["tokenizer_identity"]
        report["stages"]["load"] = _result(
            "load", adapter=adapter.describe(), tokenizer_digest=loaded["tokenizer_identity"]["digest"],
            weight_storage=resolved["precision"].storage,
            attention_implementation="eager", use_cache=False,
            parameter_elements=loaded["inventory"]["parameter_elements"],
            peak_driver_bytes=_mps_driver_bytes(torch))
        profiler.mark("load_complete", peak_driver_bytes=_mps_driver_bytes(torch))
        write_report(report_path, report)
        if args.stop_after == "load":
            return _finish(report, report_path, profiler, torch, output, success=True)

        # ---- inference: zero-init parity, nonzero influence, smoke generation
        prompt = plan["qualification"]["prompts"][0]
        rendered, input_ids, attention_mask = _prompt_ids(
            tokenizer, prompt["text"], "none", "mps")
        with torch.no_grad():
            base_logits = base(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False).logits
            adapter_logits, injection = adapter.forward_text(
                input_ids, attention_mask, use_projector=True, use_ports=False,
                return_audit=True)
        if not torch.equal(adapter_logits, base_logits):
            difference = float((adapter_logits.float() - base_logits.float()).abs().max())
            raise QualificationError(
                f"zero-initialized projector changed initial logits (max {difference})")
        original_fc2 = adapter.extension.vision_projector.fc2.weight.detach().clone()
        try:
            with torch.no_grad():
                noise = torch.randn_like(original_fc2) * float(
                    plan["qualification"]["parity"]["perturbation_scale"])
                adapter.extension.vision_projector.fc2.weight.copy_(original_fc2 + noise)
                perturbed = adapter.forward_text(input_ids, attention_mask,
                                                 use_projector=True, use_ports=False)
        finally:
            with torch.no_grad():
                adapter.extension.vision_projector.fc2.weight.copy_(original_fc2)
        perturbation_difference = float(
            (perturbed.float() - base_logits.float()).abs().max())
        if not perturbation_difference > 0:
            raise QualificationError("perturbed projector did not change logits")
        restored = adapter.forward_text(input_ids, attention_mask,
                                        use_projector=True, use_ports=False)
        if not torch.equal(restored, base_logits):
            raise QualificationError("restoring the zero projector did not restore parity")
        generations = []
        for thinking in plan["qualification"]["thinking_modes"]:
            prompt = plan["qualification"]["prompts"][0]
            rendered, ids, mask = _prompt_ids(tokenizer, prompt["text"], thinking, "mps")
            generated = _greedy_generate(
                adapter, ids, mask,
                int(plan["qualification"]["smoke_max_new_tokens"]),
                getattr(tokenizer, "eos_token_id", None))
            text = tokenizer.decode(generated["token_ids"], skip_special_tokens=False)
            generations.append({
                "prompt_id": prompt["id"], "thinking": thinking,
                "rendered_prompt_sha256": sha256_bytes(rendered.encode()),
                "prompt_token_count": int(ids.shape[1]),
                "token_ids": generated["token_ids"], "decoded_text": text,
                "stop_reason": generated["stop_reason"],
                "latency_seconds": generated["latency_seconds"],
                **{key: value for key, value in
                   _reasoning_count(tokenizer, generated["token_ids"]).items()
                   if key != "reasoning_text"},
            })
        report["stages"]["inference"] = _result(
            "inference",
            parity={"exact_logits": True, "max_abs_difference": 0.0,
                    "projector_zero_init": True},
            perturbation={"scale": plan["qualification"]["parity"]["perturbation_scale"],
                          "max_abs_change": perturbation_difference},
            injected_layers=injection.to_dict(),
            generation_budget=int(plan["qualification"]["smoke_max_new_tokens"]),
            declared_budgets=plan["qualification"]["declared_generation_budgets"],
            generations=generations,
            peak_driver_bytes=_mps_driver_bytes(torch))
        profiler.mark("inference_complete", peak_driver_bytes=_mps_driver_bytes(torch))
        write_report(report_path, report)
        if args.stop_after == "inference":
            return _finish(report, report_path, profiler, torch, output, success=True)

        # ---- input gradients with the optional ports as an explicit identity change
        extension = plan["extension"]
        if extension["memory_ports"]:
            adapter.enable_memory_ports(
                tuple(extension["memory_ports"]),
                memory_size=int(extension["memory_size"]),
                heads=int(extension["port_heads"]))
        memory, memory_mask = _memory_bank(
            torch, "mps", count=8, width=adapter.memory_size,
            dtype=base.model.embedding.weight.dtype)
        probe = adapter.backward_probe(
            input_ids, attention_mask, memory=memory, memory_mask=memory_mask)
        injection_audit = probe["injection"]
        if not injection_audit["sliding_layers"] or not injection_audit["full_layers"]:
            raise QualificationError("both sliding and full-attention blocks were not traced")
        if not probe["input_gradient"]["present"]:
            raise QualificationError("gradient did not reach the projector input")
        if not probe["new_gradient_finite"]:
            raise QualificationError("a new-module gradient is non-finite")
        if not probe["new_gradient_nonzero"]:
            raise QualificationError("no new-module gradient is nonzero")
        if not adapter.base_unchanged():
            raise QualificationError("original Spark parameters changed during backward")
        report["stages"]["input_gradients"] = _result(
            "input_gradients", **probe,
            base_unchanged=True, ports_enabled=adapter.port_layers,
            peak_driver_bytes=_mps_driver_bytes(torch))
        profiler.mark("input_gradients_complete", peak_driver_bytes=_mps_driver_bytes(torch))
        write_report(report_path, report)
        if args.stop_after == "input_gradients":
            return _finish(report, report_path, profiler, torch, output, success=True)

        # ---- exactly one FP32 AdamW update on the new module group
        update_config = plan["qualification"]["single_update"]
        optimizer = build_optimizer(
            adapter, "adamw_fused", lr_base=float(update_config["base_lr"]),
            lr_new=float(update_config["new_lr"]),
            weight_decay=float(update_config["weight_decay"]))
        if optimizer is None:
            raise QualificationError("no new trainable parameters were owned by the optimizer")
        before = {name: parameter.detach().clone()
                  for name, parameter in adapter.new_named_parameters()}
        loss_before = probe["loss"]
        optimizer.step()
        after = {name: parameter.detach().clone()
                 for name, parameter in adapter.new_named_parameters()}
        deltas = []
        for name in before:
            difference = (after[name].float() - before[name].float()).abs()
            deltas.append({"name": name, "max_abs_delta": float(difference.max()),
                           "changed": bool(torch.count_nonzero(difference))})
        if not any(item["changed"] for item in deltas):
            raise QualificationError("the single AdamW step changed no new parameter")
        if not all(bool(torch.isfinite(parameter).all())
                   for _, parameter in adapter.new_named_parameters()):
            raise QualificationError("a new parameter is non-finite after the update")
        if not adapter.base_unchanged():
            raise QualificationError("the frozen base changed during the update")
        post_probe = adapter.backward_probe(
            input_ids, attention_mask, memory=memory, memory_mask=memory_mask)
        if not post_probe["input_gradient"]["nonzero"]:
            raise QualificationError(
                "after one update the projector input gradient is still exactly zero")
        report["stages"]["single_update"] = _result(
            "single_update", optimizer=optimizer.audit(), steps=1,
            loss=loss_before, loss_finite=True,
            named_deltas=deltas, base_unchanged=True,
            all_parameters_finite=True,
            post_update_input_gradient=post_probe["input_gradient"],
            post_update_new_gradient_nonzero=post_probe["new_gradient_nonzero"],
            peak_driver_bytes=_mps_driver_bytes(torch))
        profiler.mark("single_update_complete", peak_driver_bytes=_mps_driver_bytes(torch))
        write_report(report_path, report)
        if args.stop_after == "single_update":
            return _finish(report, report_path, profiler, torch, output, success=True)

        # ---- extension-only export and fresh-process reload
        extension_dir = output / "extension"
        manifest = adapter.save_extension(extension_dir)
        exported_tensors = [item["name"] for item in manifest["export"]["adapter_tensors"]]
        for layer in extension["memory_ports"]:
            prefix = f"extension.memory_ports.{layer}."
            if not any(name.startswith(prefix) for name in exported_tensors):
                raise QualificationError(f"memory port {layer} tensors were not exported")
        if any((extension_dir / name).exists() for name in ("base", "model.safetensors")):
            raise QualificationError("extension export contains base weights")
        with torch.no_grad():
            expected_logits = adapter.forward_text(
                input_ids, attention_mask, memory=memory, memory_mask=memory_mask,
                use_projector=True, use_ports=True)
            expected_tokens = expected_logits[0].argmax(dim=-1)
        expected_path = output / "reload_expected.pt"
        torch.save({
            "input_ids": input_ids.cpu(), "attention_mask": attention_mask.cpu(),
            "memory": memory.cpu(), "memory_mask": memory_mask.cpu(),
            "logits": expected_logits.cpu(), "expected_tokens": expected_tokens.cpu(),
            "rtol": float(plan["qualification"]["export_reload"]["rtol"]),
            "atol": float(plan["qualification"]["export_reload"]["atol"]),
        }, expected_path)
        export_evidence = {
            "manifest_schema": manifest["schema"],
            "extension_tensor_count": len(exported_tensors),
            "contains_base_weights": False,
            "base_saved": False,
            "optimizer_saved": False,
            "weights_sha256": manifest["weights_sha256"],
            "export_fingerprint": manifest["fingerprint"],
        }
        profiler.mark("export_complete", peak_driver_bytes=_mps_driver_bytes(torch))
        del (optimizer, adapter, base, loaded, expected_logits, expected_tokens,
             probe, post_probe, before, after, deltas, base_logits, adapter_logits,
             perturbed, restored, injection, input_ids, attention_mask, memory,
             memory_mask)
        loaded = None
        gc.collect()
        torch.mps.empty_cache()
        profiler.mark("pre_reload_release", peak_driver_bytes=_mps_driver_bytes(torch))
        reload_report = output / "reload_compare.json"
        child_log = output / "reload_compare.child.log"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--stage", "reload_compare",
            "--config", str(Path(args.config).resolve()),
            "--source-role", source_role,
            "--source-dir", str(Path(args.source_dir).resolve()),
            "--output", str(output.resolve()),
            "--extension-dir", str(extension_dir.resolve()),
            "--expected", str(expected_path.resolve()),
            "--reload-report", str(reload_report.resolve()),
            "--device", "mps",
        ]
        child = subprocess.run(command, capture_output=True, text=True,
                               env=os.environ.copy())
        child_log.write_text(
            f"$ {' '.join(command)}\n\nSTDOUT\n{child.stdout}\n\nSTDERR\n{child.stderr}\n")
        if child.returncode != 0:
            raise QualificationError(
                f"fresh-process reload failed with exit {child.returncode}; see {child_log}")
        child_report = read_json(reload_report)
        if child_report.get("status") != "completed":
            raise QualificationError("fresh-process reload report is not completed")
        if not child_report.get("exact_greedy_tokens"):
            raise QualificationError("fresh-process greedy tokens differ")
        if not child_report.get("comparison", {}).get("within_tolerance"):
            raise QualificationError("fresh-process logits differ beyond tolerance")
        report["stages"]["export_reload"] = _result(
            "export_reload", export=export_evidence,
            fresh_process=True, child_exit_code=child.returncode,
            child_log=str(child_log),
            comparison=child_report["comparison"],
            exact_greedy_tokens=True,
            base_identity_verified=child_report.get("base_identity_verified"),
            reload_report_fingerprint=fingerprint(child_report),
            peak_driver_bytes=_mps_driver_bytes(torch))
        profiler.mark("export_reload_complete", peak_driver_bytes=_mps_driver_bytes(torch))
        write_report(report_path, report)
        return _finish(report, report_path, profiler, torch, output, success=True)
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = {"error_type": type(error).__name__, "error": str(error),
                             "traceback": traceback.format_exc(),
                             "stage": next((stage for stage in STAGES
                                            if stage not in report["stages"]), "post_export")}
        return _finish(report, report_path, profiler, torch, output, success=False)
    finally:
        try:
            import torch as _torch
            if _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
        except Exception:
            pass


def _finish(report: dict, report_path: Path, profiler, torch, output: Path,
            *, success: bool) -> int:
    summary = profiler.stop()
    report["profile_summary"] = summary
    policy = report.get("policy", {})
    peak = int((summary.get("memory") or {}).get("mps_driver_peak_sampled_bytes") or 0)
    limit = int(policy.get("driver_limit_bytes", 50 * 1024**3))
    report.setdefault("resources", {})
    report["resources"].update({
        "driver_limit_bytes": limit,
        "swap_growth_limit_bytes": int(policy.get("swap_growth_limit_bytes", 512 * 1024**2)),
        "minimum_free_percent": policy.get("minimum_free_percent", 15),
        "peak_driver_bytes": peak,
        "peak_rss_bytes": (summary.get("memory") or {}).get("rss_peak_sampled_bytes"),
        "stop_reason": None if success else "stage_failure",
        "sampled_driver_within_limit": peak < limit,
        "sustained_host_samples": policy.get("sustained_host_samples"),
    })
    if peak >= limit:
        report["status"] = "failed"
        report["resources"]["stop_reason"] = "sampled_mps_driver_limit"
    else:
        report["status"] = "completed" if success else "failed"
    report["storage"]["free_bytes_after"] = shutil.disk_usage(
        output if output.exists() else ROOT).free
    report["claims"]["native_execution"] = bool(report.get("native_execution"))
    if report["status"] == "completed":
        report["completion_scope"] = report.get("declared_stop_after", "export_reload")
    write_report(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return 0 if report["status"] == "completed" else 1


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/spark_qualification_v1.json")
    parser.add_argument("--source-role",
                        choices=tuple(OFFICIAL_ROLES) + (DERIVATIVE_ROLE,))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--stage", default="all",
                        choices=("all", "reload_compare", "compare"))
    parser.add_argument("--stop-after", default="export_reload", choices=STAGES)
    parser.add_argument("--extension-dir", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--reload-report", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--derivative-report", type=Path)
    parser.add_argument("--comparison-report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device != "mps":
        raise SystemExit("the local Spark qualification requires --device mps")
    if args.stage == "compare":
        for name in ("baseline_report", "derivative_report", "comparison_report"):
            if getattr(args, name) is None:
                raise SystemExit(f"--{name.replace('_', '-')} is required for compare")
        return run_compare(args)
    for name in ("source_role", "source_dir", "output"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required for this stage")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK=0 must be set before importing torch")
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise SystemExit("REPLICA_CAPACITY_GUARD=1 is required")
    if args.stage == "reload_compare":
        for name in ("extension_dir", "expected", "reload_report"):
            if getattr(args, name) is None:
                raise SystemExit(f"--{name.replace('_', '-')} is required for reload_compare")
        return run_reload_compare(args)
    return run_qualification(args)


if __name__ == "__main__":
    raise SystemExit(main())
