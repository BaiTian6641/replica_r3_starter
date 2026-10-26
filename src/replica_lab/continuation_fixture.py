"""Small-model restore-mechanics fixture; never production continuation evidence.

The fixture isolates the direct-target-device resume path (model adoption,
optimizer/RNG/cursor restoration, boundary fingerprint, exact-vs-envelope
comparison) on a pinned 1-2 layer FP32 Llama. It deliberately does not reuse
``validate_model_lock`` or the MiniCPM5 381-tensor inventory and cannot claim
production full-scope continuation. Torch/transformers are imported only when a
model is actually constructed so the receipt/protocol checks stay CPU-light.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

FIXTURE_FORMAT = "replica.fused_continuation_fixture.protocol.v1"
FIXTURE_ARM_NAMES = frozenset(
    {"uninterrupted_a", "uninterrupted_b", "first", "resume"})
REQUIRED_IMPLEMENTATION_PATHS = (
    "scripts/probe_fused_adamw_sustained.py",
    "scripts/compare_fused_continuation.py",
    "src/replica_lab/continuation_fixture.py",
    "src/replica_lab/training_state.py",
    "src/replica_lab/checkpointing.py",
    "src/replica_lab/optimizers.py",
    "src/replica_lab/model_adapter.py",
    "src/replica_lab/training_tools.py",
    "scripts/train_text.py",
)
FAILURE_CLASSES = (
    "boundary_mismatch",
    "numerical_nonfinite",
    "protocol_violation",
    "checkpoint_restore_failure",
    "resource_stop",
    "unclassified",
)


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def policy_digest(policy: dict) -> str:
    """Stable digest binding a declared envelope policy to its exact text."""
    return sha256_bytes(canonical_json(policy).encode("utf-8"))


def fixture_config_digest(config: dict) -> str:
    return policy_digest(config)


def is_fixture_protocol(protocol: dict) -> bool:
    return (protocol.get("format") == FIXTURE_FORMAT
            or protocol.get("production_scope") is False)


def fixture_inventory(config: dict) -> dict:
    """Validate the pinned Llama shape and compute its exact tensor inventory."""
    if not isinstance(config, dict):
        raise ValueError("fixture config must be a JSON object")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"fixture config requires a positive integer {key}")
    if config.get("model_type") != "llama":
        raise ValueError("fixture config must declare model_type=llama")
    if config.get("architectures") != ["LlamaForCausalLM"]:
        raise ValueError("fixture config must declare architectures=[LlamaForCausalLM]")
    if config.get("tie_word_embeddings") is not False:
        raise ValueError("fixture config requires untied embeddings")
    if config.get("attention_dropout") != 0.0:
        raise ValueError("fixture config requires attention_dropout=0.0")
    if config.get("attention_bias") not in (False, None):
        raise ValueError("fixture config requires attention_bias=false")
    if config.get("mlp_bias") not in (False, None):
        raise ValueError("fixture config requires mlp_bias=false")
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    if hidden % heads:
        raise ValueError("hidden_size must divide num_attention_heads")
    if heads % kv_heads:
        raise ValueError("num_attention_heads must divide num_key_value_heads")
    head_dim = config.get("head_dim") or hidden // heads
    if head_dim * heads != hidden:
        raise ValueError("head_dim*num_attention_heads must equal hidden_size")
    layers = config["num_hidden_layers"]
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    kv_dim = kv_heads * head_dim
    attention = hidden * hidden + 2 * hidden * kv_dim + hidden * hidden
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    per_layer_parameters = attention + mlp + norms
    embedding_parameters = vocab * hidden
    parameter_count = (embedding_parameters + layers * per_layer_parameters
                       + hidden + embedding_parameters)
    tensor_count = 1 + 9 * layers + 1 + 1
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "layer_tensor_count": 9,
        "per_layer_parameters": per_layer_parameters,
        "token_embedding_parameters": embedding_parameters,
        "untied_output_head_parameters": embedding_parameters,
        "final_norm_parameters": hidden,
        "head_dim": head_dim,
        "kv_dim": kv_dim,
    }


def validate_policy_declaration(protocol: dict) -> str:
    """Bind ``aa-envelope-v1`` text, name and constraints by canonical hash."""
    comparison = protocol.get("continuation_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("fixture protocol requires continuation_comparison")
    aa = comparison.get("aa_envelope")
    if not isinstance(aa, dict):
        raise ValueError("fixture protocol requires a preregistered aa_envelope")
    policy = aa.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("aa_envelope must carry its normative policy object")
    digest = policy_digest(policy)
    if aa.get("policy_sha256") != digest:
        raise ValueError("aa_envelope policy hash does not match its policy object")
    if aa.get("name") != "aa-envelope-v1" or policy.get("name") != "aa-envelope-v1":
        raise ValueError("fixture protocol requires the aa-envelope-v1 policy")
    if aa.get("historical_envelope_allowed") is not False:
        raise ValueError("historical envelopes are not allowed for the fixture")
    margin = aa.get("extra_margin")
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) \
            or not math.isfinite(float(margin)) or float(margin) != 0.0:
        raise ValueError("aa_envelope extra_margin must be exactly 0")
    for key in ("update_1_loss_and_gradient_norm_exact",
                "boundary_fingerprint_exact",
                "pass_label_forbidden_unless_exact_status_passes"):
        if aa.get(key) is not True:
            raise ValueError(f"aa_envelope requires {key}=true")
    if aa.get("permitted_nonexact_label") != "failed_exact_within_declared_variation":
        raise ValueError("aa_envelope permitted label must stay scoped")
    source = aa.get("source")
    if not isinstance(source, str) or "uninterrupted" not in source:
        raise ValueError("aa_envelope source must be the fresh uninterrupted arms")
    return digest


def validate_fixture_retention(protocol: dict, root: str | Path | None = None) -> dict:
    """Require a separate fixture checkpoint path and one-write retention."""
    storage = protocol.get("storage_policy")
    retention = protocol.get("retention")
    if not isinstance(storage, dict) or not isinstance(retention, dict):
        raise ValueError("fixture protocol requires storage_policy and retention")
    fixture_root = storage.get("fixture_run_root")
    retained = retention.get("retained_full_scope_checkpoint")
    if not isinstance(fixture_root, str) or not fixture_root.strip():
        raise ValueError("storage_policy.fixture_run_root is required")
    if not isinstance(retained, str) or not retained.strip():
        raise ValueError("retention.retained_full_scope_checkpoint is required")
    fixture_path = Path(fixture_root).resolve()
    retained_path = Path(retained).resolve()
    if fixture_path == retained_path or fixture_path in retained_path.parents \
            or retained_path in fixture_path.parents:
        raise ValueError("fixture checkpoint path overlaps preserved full-scope evidence")
    if storage.get("checkpoint_writes") != 1:
        raise ValueError("fixture protocol allows exactly one checkpoint write")
    if storage.get("final_model_exports") != 0:
        raise ValueError("fixture protocol forbids final model exports")
    if storage.get("duplicate_base_saves") != 0:
        raise ValueError("fixture protocol forbids duplicate base saves")
    limit = storage.get("max_checkpoint_bytes")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("storage_policy.max_checkpoint_bytes must be a positive integer")
    if retention.get("preserve_historical") is not True:
        raise ValueError("retention must preserve historical evidence")
    if not isinstance(retention.get("historical_evidence"), str) \
            or not retention["historical_evidence"].strip():
        raise ValueError("retention must name the preserved historical evidence")
    if root is not None and retained_path.exists():
        # Present evidence is never rewritten; only its presence is observed.
        if not retained_path.is_dir():
            raise ValueError("retained full-scope evidence is not a directory")
    return {
        "fixture_run_root": fixture_root,
        "retained_full_scope_checkpoint": retained,
        "max_checkpoint_bytes": limit,
        "checkpoint_writes": 1,
        "final_model_exports": 0,
    }


def validate_fixture_protocol(protocol: dict, *, protocol_path=None,
                              root=None) -> dict:
    """Validate the fixture lock, counts, gates and policy hash binding."""
    if not isinstance(protocol, dict):
        raise ValueError("fixture protocol must be a JSON object")
    if protocol.get("format") != FIXTURE_FORMAT:
        raise ValueError(f"fixture protocol format must be {FIXTURE_FORMAT}")
    if protocol.get("production_scope") is not False:
        raise ValueError("fixture protocol must declare production_scope=false")
    fixture = protocol.get("fixture")
    if not isinstance(fixture, dict):
        raise ValueError("fixture protocol requires a fixture lock section")
    claim = fixture.get("claim_boundary")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("fixture lock requires an explicit claim_boundary")
    config = fixture.get("config")
    if not isinstance(config, dict):
        raise ValueError("fixture lock requires an inline pinned config")
    digest = fixture_config_digest(config)
    if fixture.get("config_sha256") != digest:
        raise ValueError("fixture config hash does not match the pinned config")
    inventory = fixture_inventory(config)
    if fixture.get("tensor_count") != inventory["tensor_count"]:
        raise ValueError("fixture lock tensor_count differs from its config")
    if fixture.get("parameter_count") != inventory["parameter_count"]:
        raise ValueError("fixture lock parameter_count differs from its config")
    seed = fixture.get("init_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("fixture lock requires a nonnegative integer init_seed")
    if protocol.get("expected_tensor_count") != inventory["tensor_count"]:
        raise ValueError("expected_tensor_count must equal the fixture tensor inventory")
    tokenizer = fixture.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("fixture lock requires a sealed tokenizer reference")
    for key in ("path", "lock", "lock_sha256"):
        if not isinstance(tokenizer.get(key), str) or not tokenizer[key].strip():
            raise ValueError(f"fixture tokenizer lock requires {key}")
    if tokenizer.get("vocab_size") != config["vocab_size"]:
        raise ValueError("fixture tokenizer vocab_size must match the config")
    if root is not None:
        root = Path(root)
        tokenizer_path = root / tokenizer["path"]
        if not tokenizer_path.is_dir():
            raise ValueError("sealed fixture tokenizer directory is missing")
        lock_path = root / tokenizer["lock"]
        if not lock_path.is_file():
            raise ValueError("sealed fixture tokenizer lock is missing")
        if sha256_file(lock_path) != tokenizer["lock_sha256"]:
            raise ValueError("sealed fixture tokenizer lock bytes differ")
        lock = json.loads(lock_path.read_text())
        if not isinstance(lock.get("revision"), str) or len(lock["revision"]) != 40:
            raise ValueError("sealed fixture tokenizer lock revision is invalid")

    source = protocol.get("source")
    if not isinstance(source, dict):
        raise ValueError("fixture protocol requires a source block")
    for key in ("data", "data_sha256"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"fixture source requires {key}")
    if root is not None:
        data_path = Path(root) / source["data"]
        if not data_path.is_file():
            raise ValueError("fixture diagnostic data is missing")
        if sha256_file(data_path) != source["data_sha256"]:
            raise ValueError("fixture diagnostic data bytes differ")

    scope = protocol.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("fixture protocol requires a scope block")
    if scope.get("trainable_tensors") != inventory["tensor_count"]:
        raise ValueError("scope.trainable_tensors differs from the fixture inventory")
    if scope.get("trainable_parameters") != inventory["parameter_count"]:
        raise ValueError("scope.trainable_parameters differs from the fixture inventory")
    if scope.get("regime") != "full":
        raise ValueError("fixture scope must be the full original parameter regime")
    if scope.get("attention") != "eager":
        raise ValueError("fixture scope requires eager attention")
    if scope.get("compute_dtype") != "float32":
        raise ValueError("fixture scope requires float32 compute")

    optimizer = protocol.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("fixture protocol requires an optimizer block")
    if optimizer.get("selection") != "adamw_fused":
        raise ValueError("fixture optimizer must select adamw_fused")
    if optimizer.get("class") != "torch.optim.AdamW":
        raise ValueError("fixture optimizer class must be torch.optim.AdamW")
    if optimizer.get("fused") is not True or optimizer.get("fallback") is not False:
        raise ValueError("fixture optimizer requires fused=true and fallback=false")

    numeric = protocol.get("numeric_policy")
    if not isinstance(numeric, dict):
        raise ValueError("fixture protocol requires numeric_policy")
    if numeric.get("torch_use_deterministic_algorithms") is not True:
        raise ValueError("fixture numeric policy requires deterministic algorithms")
    if numeric.get("mps_fast_math") != "disabled" or numeric.get("dtype") != "float32":
        raise ValueError("fixture numeric policy requires disabled fast math and float32")

    resource = protocol.get("resource_policy")
    if not isinstance(resource, dict):
        raise ValueError("fixture protocol requires resource_policy")
    if resource.get("device") != "mps" or resource.get("cpu_fallback") is not False:
        raise ValueError("fixture resource policy requires mps with cpu_fallback=false")
    if resource.get("exclusive_native_worker") is not True:
        raise ValueError("fixture resource policy requires one exclusive native worker")
    if resource.get("native_jobs_at_once") != 1:
        raise ValueError("fixture resource policy allows one native job at a time")

    comparison = protocol.get("continuation_comparison")
    if not isinstance(comparison, dict) or comparison.get("enabled") is not True:
        raise ValueError("fixture protocol requires the continuation comparison")
    if comparison.get("restore_strategy") != "direct_target_device":
        raise ValueError("fixture comparison requires direct_target_device restore")
    arms = comparison.get("arms")
    if not isinstance(arms, list) or len(arms) != 4:
        raise ValueError("fixture comparison requires exactly the four declared arms")
    arm_ids = [arm.get("id") if isinstance(arm, dict) else arm for arm in arms]
    if len(set(arm_ids)) != 4 or set(arm_ids) != FIXTURE_ARM_NAMES:
        raise ValueError("fixture comparison requires exactly the four declared arms")
    accounting = comparison.get("required_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("fixture comparison requires required_accounting")
    if accounting.get("staged_cpu_optimizer_bytes") != 0:
        raise ValueError("fixture accounting requires zero staged CPU optimizer bytes")
    if accounting.get("restored_optimizer_bytes_positive") is not True:
        raise ValueError("fixture accounting requires positive restored optimizer bytes")
    if accounting.get("restore_strategy") != "direct_target_device":
        raise ValueError("fixture accounting requires direct_target_device restore")
    if not comparison.get("stop_conditions"):
        raise ValueError("fixture comparison requires declared stop conditions")

    policy_sha = validate_policy_declaration(protocol)
    retention = validate_fixture_retention(protocol, root)

    implementation = protocol.get("implementation_paths")
    if not isinstance(implementation, list) or not implementation:
        raise ValueError("fixture protocol requires implementation_paths")
    missing = sorted(set(REQUIRED_IMPLEMENTATION_PATHS) - set(implementation))
    if missing:
        raise ValueError(f"implementation_paths omit required files: {missing}")
    for name in implementation:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("implementation_paths must be repo-relative")
        if root is not None and not (Path(root) / name).is_file():
            raise ValueError(f"implementation path is missing: {name}")

    return {
        "format": FIXTURE_FORMAT,
        "production_scope": False,
        "config": config,
        "config_sha256": digest,
        "init_seed": seed,
        "tensor_count": inventory["tensor_count"],
        "parameter_count": inventory["parameter_count"],
        "tokenizer_path": tokenizer["path"],
        "tokenizer_lock": tokenizer["lock"],
        "tokenizer_lock_sha256": tokenizer["lock_sha256"],
        "tokenizer_vocab_size": tokenizer["vocab_size"],
        "claim_boundary": claim,
        "protocol_policy_sha256": policy_sha,
        "retention": retention,
        "protocol_path": str(protocol_path) if protocol_path is not None else None,
        "protocol_sha256": (sha256_file(protocol_path)
                            if protocol_path is not None else None),
    }


def build_fixture_model(config: dict, *, seed: int):
    """Construct the pinned FP32 eager-attention Llama with a fixed init seed."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    inventory = fixture_inventory(config)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("fixture init seed must be a nonnegative integer")
    torch.manual_seed(seed)
    llm_config = LlamaConfig(**config)
    llm_config._attn_implementation = "eager"
    llm_config.use_cache = False
    model = LlamaForCausalLM(llm_config)
    model = model.to(dtype=torch.float32)
    model.config.use_cache = False
    actual_tensors = len(list(model.parameters()))
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if (actual_tensors != inventory["tensor_count"]
            or actual_parameters != inventory["parameter_count"]):
        raise ValueError(
            "built fixture inventory differs from its pinned lock: "
            f"{actual_tensors} tensors / {actual_parameters} parameters")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("fixture model must be FP32")
    if model.config.attention_dropout != 0.0:
        raise ValueError("fixture model must use zero attention dropout")
    return model


def assert_fixture_scope(model, optimizer, binding: dict) -> None:
    """Require the built model and optimizer to match the pinned inventory."""
    import torch

    parameters = list(model.parameters())
    if len(parameters) != binding["tensor_count"]:
        raise ValueError("Fixture parameter tensor count is incomplete")
    if sum(parameter.numel() for parameter in parameters) != binding["parameter_count"]:
        raise ValueError("Fixture parameter count is incomplete")
    if not all(parameter.requires_grad for parameter in parameters):
        raise ValueError("Fixture scope must train every parameter")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("Fixture scope requires FP32 parameters")
    owned = [id(parameter) for group in optimizer.param_groups
             for parameter in group["params"]]
    if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in parameters}:
        raise ValueError("Fixture optimizer does not own every tensor exactly once")
    audit = optimizer.audit()
    if audit.get("optimizer_class") != "torch.optim.AdamW" or audit.get("fused") is not True:
        raise ValueError("Fixture optimizer is not native fused AdamW")


def accounting_gates(accounting: dict | None) -> dict:
    """Evaluate direct-target-device restore accounting as explicit gates."""
    accounting = accounting or {}
    optimizer = accounting.get("optimizer") or {}
    staged = optimizer.get("staged") or {}
    restored = optimizer.get("restored") or {}
    restored_bytes = restored.get("bytes")
    return {
        "staged_cpu_optimizer_bytes": staged.get("bytes"),
        "restored_optimizer_bytes": restored_bytes,
        "optimizer_restore_strategy": optimizer.get("restore_strategy"),
        "full_cpu_model_copy": accounting.get("full_cpu_model_copy"),
        "gates": {
            "accounting_available": bool(accounting),
            "staged_cpu_optimizer_bytes_zero": staged.get("bytes") == 0,
            "restored_optimizer_bytes_positive": (
                isinstance(restored_bytes, int) and not isinstance(restored_bytes, bool)
                and restored_bytes > 0),
            "direct_target_device_restore": (
                optimizer.get("restore_strategy") == "direct_target_device"),
            "no_full_cpu_model_copy": accounting.get("full_cpu_model_copy") is False,
        },
    }


def require_fresh_output(path: str | Path, *, what: str = "output") -> Path:
    """Reject an existing file or nonempty directory before a run writes."""
    destination = Path(path)
    if destination.exists():
        if destination.is_file():
            raise FileExistsError(f"{what} path already exists: {destination}")
        if any(destination.iterdir()):
            raise FileExistsError(f"{what} directory is not fresh: {destination}")
    return destination


def classify_failure(error: BaseException) -> str:
    """Preserve distinct failure classes; never collapse resource and numerics."""
    message = str(error).lower()
    if isinstance(error, MemoryError):
        return "resource_stop"
    if isinstance(error, KeyboardInterrupt):
        return "resource_stop"
    for marker in ("swap", "capacity guard", "driver limit", "memory"):
        if marker in message:
            return "resource_stop"
    if "boundary fingerprint" in message:
        return "boundary_mismatch"
    if "nonfinite" in message:
        return "numerical_nonfinite"
    for marker in ("torch.load", "runtime checkpoint", "checkpoint byte mismatch",
                   "checkpoint", "map_location", "restore"):
        if marker in message:
            return "checkpoint_restore_failure"
    if isinstance(error, (ValueError, KeyError, TypeError, FileExistsError,
                          FileNotFoundError, RuntimeError)):
        return "protocol_violation"
    return "unclassified"
