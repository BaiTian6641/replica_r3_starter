"""CPU-only checks for the small-model continuation fixture diagnostic."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab import continuation_fixture as FIXTURE  # noqa: E402

PROTOCOL_PATH = ROOT / "configs/fused_continuation_fixture_v1.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())

COMPARE_SPEC = importlib.util.spec_from_file_location(
    "fused_compare_fixture", ROOT / "scripts/compare_fused_continuation.py")
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
COMPARE_SPEC.loader.exec_module(COMPARE)

BINDING = COMPARE.protocol_binding(PROTOCOL_PATH)
LEDGER_TENSORS = BINDING["expected_tensor_count"]

MICRO_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 32,
    "rms_norm_eps": 1e-5,
    "initializer_range": 0.02,
    "vocab_size": 64,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_cache": False,
    "torch_dtype": "float32",
}


def _tampered(**changes):
    protocol = copy.deepcopy(PROTOCOL)
    for path, value in changes.items():
        node = protocol
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return protocol


def test_fixture_protocol_lock_counts_and_claim_boundary():
    binding = FIXTURE.validate_fixture_protocol(PROTOCOL, protocol_path=PROTOCOL_PATH)
    inventory = FIXTURE.fixture_inventory(PROTOCOL["fixture"]["config"])
    assert binding["production_scope"] is False
    assert PROTOCOL["production_scope"] is False
    assert binding["tensor_count"] == inventory["tensor_count"] == LEDGER_TENSORS
    assert binding["tensor_count"] == 21
    assert binding["tensor_count"] != 381
    assert binding["parameter_count"] == inventory["parameter_count"] == 16793920
    assert PROTOCOL["expected_tensor_count"] == 21
    assert binding["config_sha256"] == FIXTURE.fixture_config_digest(
        PROTOCOL["fixture"]["config"])
    assert binding["protocol_sha256"] == FIXTURE.sha256_file(PROTOCOL_PATH)
    assert "production" in binding["claim_boundary"].lower()
    assert PROTOCOL["fixture"]["tokenizer"]["vocab_size"] == 130560
    assert PROTOCOL["storage_policy"]["checkpoint_writes"] == 1
    assert PROTOCOL["storage_policy"]["final_model_exports"] == 0


def test_fixture_protocol_rejects_tampering_and_lookalike_scope():
    for changes, match in (
        ({"production_scope": True}, "production_scope"),
        ({"fixture.config.hidden_size": 128}, "config hash"),
        ({"fixture.tensor_count": 381}, "tensor_count"),
        ({"expected_tensor_count": 381}, "expected_tensor_count"),
        ({"scope.trainable_tensors": 381}, "trainable_tensors"),
        ({"continuation_comparison.aa_envelope.policy.rule": "anything"},
         "policy hash"),
        ({"storage_policy.checkpoint_writes": 2}, "checkpoint write"),
        ({"retention.retained_full_scope_checkpoint":
          "runs/fused_continuation_fixture_v1"}, "overlap"),
        ({"resource_policy.cpu_fallback": True}, "cpu_fallback"),
        ({"numeric_policy.mps_fast_math": "enabled"}, "fast math"),
        ({"continuation_comparison.required_accounting."
          "staged_cpu_optimizer_bytes": 1024}, "staged CPU optimizer"),
        ({"continuation_comparison.arms": PROTOCOL["continuation_comparison"]["arms"][:3]},
         "four declared arms"),
    ):
        with pytest.raises(ValueError, match=match):
            FIXTURE.validate_fixture_protocol(_tampered(**changes))


def test_fixture_model_is_pinned_deterministic_and_eager_fp32():
    torch = pytest.importorskip("torch")
    inventory = FIXTURE.fixture_inventory(MICRO_CONFIG)
    first = FIXTURE.build_fixture_model(MICRO_CONFIG, seed=11)
    second = FIXTURE.build_fixture_model(MICRO_CONFIG, seed=11)
    assert len(list(first.parameters())) == inventory["tensor_count"]
    assert sum(p.numel() for p in first.parameters()) == inventory["parameter_count"]
    assert all(p.dtype == torch.float32 for p in first.parameters())
    assert first.config.attention_dropout == 0.0
    assert first.config._attn_implementation == "eager"
    hashes = lambda model: {
        name: FIXTURE.sha256_bytes(p.detach().numpy().tobytes())
        for name, p in model.named_parameters()}
    assert hashes(first) == hashes(second)


def test_fixture_scope_gate_rejects_wrong_inventory():
    from replica_lab.model_adapter import LlamaMemoryAdapter
    from replica_lab.optimizers import build_optimizer

    pytest.importorskip("torch")
    base = FIXTURE.build_fixture_model(MICRO_CONFIG, seed=3)
    model = LlamaMemoryAdapter(base, "full")
    optimizer = build_optimizer(model, "adamw_fused", lr_base=1e-4, lr_new=1e-4,
                                weight_decay=0.0)
    inventory = FIXTURE.fixture_inventory(MICRO_CONFIG)
    FIXTURE.assert_fixture_scope(model, optimizer, inventory)
    with pytest.raises(ValueError, match="tensor count"):
        FIXTURE.assert_fixture_scope(
            model, optimizer, {**inventory, "tensor_count": inventory["tensor_count"] + 1})
    with pytest.raises(ValueError, match="parameter count"):
        FIXTURE.assert_fixture_scope(
            model, optimizer, {**inventory,
                               "parameter_count": inventory["parameter_count"] + 1})


def _norm(l2):
    return {"l1_norm": l2, "l2_norm": l2, "max_abs": l2, "finite": True}


def _update(step, loss, gradient_norm, parameter_l2, gradient_l2, moment_l2):
    return {
        "step": step, "loss": loss, "lr_factor": 1.0,
        "learning_rates": [1e-4],
        "gradient_norm_before_clipping": gradient_norm,
        "processed_tokens": 440, "supervised_tokens": 80,
        "training_seconds": 1.0,
        "parameter_norms": _norm(parameter_l2),
        "gradient_norms_pre_clip": _norm(gradient_l2),
        "optimizer_moment_norms": _norm(moment_l2),
    }


def _final_fingerprint(parameter_l2, moment_l2, parameter_hash):
    entry = {"l1_norm": parameter_l2, "l2_norm": parameter_l2, "max_abs": parameter_l2,
             "value_sha256": parameter_hash, "dtype": "torch.float32",
             "device": "cpu", "shape": [2], "storage_bytes": 8,
             "storage_offset": 0, "stride": [1], "is_contiguous": True,
             "finite": True}
    return {
        "format": "replica.training_fingerprint.v1",
        "parameters": {"weight": dict(entry)},
        "buffers": {"buffers": {}},
        "optimizer": {"state": {"weight": {"exp_avg": {
            **entry, "value_sha256": f"moment-{moment_l2}-{parameter_hash}"}}}},
    }


def _boundary(parameter_hash="boundary", moment_hash="moment"):
    entry = {"l1_norm": 1.0, "l2_norm": 1.0, "max_abs": 1.0, "finite": True,
             "dtype": "torch.float32", "device": "cpu", "shape": [2],
             "storage_bytes": 8, "storage_offset": 0, "stride": [1],
             "is_contiguous": True, "value_sha256": parameter_hash}
    return {
        "format": "replica.training_fingerprint.v1",
        "train_mode": {"model_training": True, "mode_sha256": "mode",
                       "module_count": 1, "training_modules": 1,
                       "modules": [{"name": "", "training": True}]},
        "buffers": {"count": 1, "nonpersistent_names": ["rotary"],
                    "buffers": {"rotary": {**entry, "value_sha256": "rotary"}}},
        "parameters": {"weight": entry},
        "optimizer": {"parameter_count": 1, "group_count": 1,
                      "state_tensors": 1, "state_bytes": 8,
                      "state_bytes_by_dtype": {"torch.float32": 8},
                      "step_values": [1.0],
                      "groups": [{"group_name": "original_adamw_fused",
                                  "algorithm": "adamw_fused", "lr": 1e-4,
                                  "betas": [0.9, 0.999], "eps": 1e-8,
                                  "weight_decay": 0.0, "fused": True,
                                  "tensor_count": 1, "parameter_count": 2}],
                      "state": {"weight": {"exp_avg": {
                          **entry, "value_sha256": moment_hash}}}},
        "rng": {"python_sha256": "python",
                "torch_cpu": {**entry, "value_sha256": "rng"},
                "devices": ["cpu"]},
        "schedule_state": {"completed_steps": 1, "base_rates": [1e-4]},
        "data_state": {"next_step": 1, "batch_plan_sha256": "plan"},
    }


def _accounting(staged=0, restored=2000, strategy="direct_target_device"):
    return {
        "path": "native",
        "full_cpu_model_copy": False,
        "optimizer": {
            "restore_strategy": strategy,
            "staged": {"bytes": staged, "tensor_count": 0 if staged == 0 else 1},
            "restored": {"bytes": restored, "tensor_count": 21},
        },
        "swap_growth_bytes": 0,
        "resident_estimate": {"restored_optimizer_bytes": restored},
    }


def _report(mode, updates, output_loss, output_hash, parameter_hash, *,
            before_hash="initial", checkpoint=None, boundary=None,
            accounting=None, protocol_sha=None, policy_sha=None,
            production_scope=False):
    return {
        "status": "completed",
        "execution_mode": mode,
        "production_scope": production_scope,
        "changed_tensors": LEDGER_TENSORS,
        "model_exported": False,
        "protocol_sha256": protocol_sha or BINDING["protocol_sha256"],
        "protocol_policy_sha256": (
            policy_sha if policy_sha is not None
            else BINDING["aa_envelope_policy_sha256"]),
        "implementation_sha256": {"probe": "implementation",
                                 "fixture": "fixture"},
        "parameter_hashes_before": {"weight": before_hash},
        "parameter_hashes_after": {"weight": parameter_hash},
        "checkpoint": checkpoint,
        "steps": 1 if mode == "first" else 2,
        "start_step": 0 if mode != "resume" else 1,
        "optimizer_state_step_values": [1.0] if mode == "first" else [2.0],
        "updates": updates,
        "output_fingerprint": {"loss": output_loss, "logits_sha256": output_hash},
        "checkpoint_reload_validated": mode == "resume",
        "resume_start_matches_first_process": mode == "resume",
        "boundary_fingerprint": boundary,
        "final_fingerprint": _final_fingerprint(
            updates[-1]["parameter_norms"]["l2_norm"],
            updates[-1]["optimizer_moment_norms"]["l2_norm"],
            f"final-{parameter_hash}"),
        "resume_accounting": accounting,
    }


def _guard():
    return {
        "status": "completed_within_declared_sampled_limits",
        "elapsed_seconds": 10.0,
        "guard": {
            "policy": {"driver_limit_bytes": 50 * 1024**3,
                       "swap_growth_limit_bytes": 512 * 1024**2,
                       "minimum_free_percent": 15.0},
            "stop_reason": None,
            "peak_driver_bytes": 10 * 1024**3,
            "peak_swap_growth_bytes": 0,
            "minimum_observed_free_percent": 30.0,
        },
    }


def _fixture_reports(*, exact=False, resume_loss=1.3497597854584455,
                     resume_output_loss=3.1678595542907715,
                     resume_boundary=None, staged=0, restored=2000,
                     strategy="direct_target_device"):
    uninterrupted = []
    losses = (1.3497576033696532, 1.3497593011707067) if not exact else (
        1.3497597854584455, 1.3497597854584455)
    gradients = (154.01649475097656, 154.0172119140625) if not exact else (
        154.0167694091797, 154.0167694091797)
    outputs = (3.1678543090820312, 3.1678714752197266) if not exact else (
        3.1678595542907715, 3.1678595542907715)
    output_hashes = (("logits-0", "logits-1") if not exact
                     else ("logits-same", "logits-same"))
    hashes = ("final-0", "final-1") if not exact else ("final", "final")
    for index in range(2):
        uninterrupted.append(_report(
            "continuous",
            [_update(1, 14.649566411972046, 1592.486328125, 10.0, 20.0, 30.0),
             _update(2, losses[index], gradients[index],
                     10.1 + index * 1e-6, 20.1 + index * 1e-6, 30.1 + index * 1e-6)],
            outputs[index], output_hashes[index], hashes[index]))
    first = _report(
        "first",
        [_update(1, 14.649566411972046, 1592.486328125, 10.0, 20.0, 30.0)],
        0.06391267478466034, "first-logits",
        "final" if exact else "first",
        checkpoint={"step": 1}, boundary=_boundary())
    resume_updates = [_update(1, 14.649566411972046, 1592.486328125, 10.0, 20.0, 30.0)]
    if exact:
        resume_updates.append(_update(
            2, losses[1], gradients[1], 10.1 + 1e-6, 20.1 + 1e-6, 30.1 + 1e-6))
    else:
        resume_updates.append(_update(
            2, resume_loss, 154.0167694091797, 10.1000001, 20.1000001, 30.1000001))
    resume = _report(
        "resume", resume_updates, resume_output_loss,
        "logits-same" if exact else "resume-logits",
        "final" if exact else "resume", before_hash="final" if exact else "first",
        boundary=resume_boundary or _boundary(), accounting=_accounting(
            staged=staged, restored=restored, strategy=strategy))
    return uninterrupted, first, resume


def _compare(uninterrupted, first, resume, **overrides):
    gates = COMPARE.binding_gates(BINDING)
    gates.update({
        "claim_boundary": BINDING["claim_boundary"],
        "policy": "aa-envelope-v1",
        "reference_index": 1,
    })
    gates.update(overrides)
    return COMPARE.compare_reports(
        uninterrupted, first, resume,
        uninterrupted_guards=[_guard(), _guard()],
        first_guard=_guard(), resume_guard=_guard(), **gates)


def test_fixture_exact_match_passes_only_when_every_exact_check_holds():
    uninterrupted, first, resume = _fixture_reports(exact=True)
    result = _compare(uninterrupted, first, resume)
    assert result["exact_status"] == "passed"
    assert result["status"] == "passed"
    assert result["production_scope"] is False
    assert result["claim_boundary"] == BINDING["claim_boundary"]
    assert result["scope_binding"]["expected_tensor_count"] == 21
    assert result["policy"]["bound_sha256"] == BINDING["aa_envelope_policy_sha256"]


def test_fixture_envelope_labels_variation_without_turning_exact_into_pass():
    uninterrupted, first, resume = _fixture_reports()
    result = _compare(uninterrupted, first, resume)
    assert result["exact_status"] == "failed"
    assert result["variation_status"] == "within_aa_envelope"
    assert result["status"] == "failed_exact_within_declared_variation"
    assert result["status"] != "passed"
    assert result["policy"]["applied"] is True
    assert result["policy"]["does_not_relax_exact_status"] is True
    assert result["final_parameter_hash_difference_count"] == 1


def test_fixture_envelope_divergence_stays_failed():
    uninterrupted, first, resume = _fixture_reports(resume_loss=2.3497597854584455)
    result = _compare(uninterrupted, first, resume)
    assert result["exact_status"] == "failed"
    assert result["variation_status"] == "divergent"
    assert result["status"] == "failed"


def test_fixture_requires_boundary_fingerprint_and_localizes_mismatch():
    uninterrupted, first, resume = _fixture_reports()
    resume["boundary_fingerprint"] = None
    result = _compare(uninterrupted, first, resume)
    assert result["exact_status"] == "failed"
    assert "boundary_fingerprint_present" in result["failures"]

    uninterrupted, first, resume = _fixture_reports(
        resume_boundary=_boundary(parameter_hash="different"))
    result = _compare(uninterrupted, first, resume)
    assert "boundary_fingerprint_identical" in result["failures"]
    assert result["boundary_fingerprint_comparison"]["status"] == "mismatched"
    assert "boundary_fingerprint" in result["ab_variation"]["exceeding_fields"]


def test_direct_restore_accounting_is_a_hard_gate():
    good = FIXTURE.accounting_gates(_accounting())
    assert all(good["gates"].values())

    staged = FIXTURE.accounting_gates(_accounting(staged=4096))
    assert staged["gates"]["staged_cpu_optimizer_bytes_zero"] is False
    assert all(value for key, value in staged["gates"].items()
               if key != "staged_cpu_optimizer_bytes_zero")

    unloaded = FIXTURE.accounting_gates(_accounting(restored=0))
    assert unloaded["gates"]["restored_optimizer_bytes_positive"] is False

    staged_strategy = FIXTURE.accounting_gates(
        _accounting(strategy="cpu_staged"))
    assert staged_strategy["gates"]["direct_target_device_restore"] is False
    assert FIXTURE.accounting_gates(None)["gates"]["accounting_available"] is False

    uninterrupted, first, resume = _fixture_reports(staged=4096)
    result = _compare(uninterrupted, first, resume)
    assert "resume_accounting_staged_cpu_optimizer_bytes_zero" in result["failures"]

    uninterrupted, first, resume = _fixture_reports(restored=0)
    result = _compare(uninterrupted, first, resume)
    assert "resume_accounting_restored_optimizer_bytes_positive" in result["failures"]

    uninterrupted, first, resume = _fixture_reports(strategy="cpu_staged")
    result = _compare(uninterrupted, first, resume)
    assert "resume_accounting_direct_target_device_restore" in result["failures"]


def test_protocol_and_policy_hash_binding_is_enforced():
    assert BINDING["protocol_sha256"] == FIXTURE.sha256_file(PROTOCOL_PATH)
    assert BINDING["aa_envelope_policy_sha256"] == FIXTURE.policy_digest(
        COMPARE.AA_ENVELOPE_POLICY)
    assert BINDING["require_boundary_fingerprint"] is True
    assert BINDING["require_direct_restore_accounting"] is True
    assert BINDING["production_scope"] is False

    uninterrupted, first, resume = _fixture_reports()
    for report in uninterrupted:
        report["protocol_policy_sha256"] = "0" * 64
    result = _compare(uninterrupted, first, resume)
    assert any(name.endswith("policy_hash_bound") for name in result["failures"])

    uninterrupted, first, resume = _fixture_reports()
    first["protocol_sha256"] = "1" * 64
    result = _compare(uninterrupted, first, resume)
    assert "protocol_hash_matches_declared" in result["failures"]


def test_fixture_requires_two_fresh_uninterrupted_arms():
    uninterrupted, first, resume = _fixture_reports()
    result = COMPARE.compare_reports(
        uninterrupted[:1], first, resume,
        uninterrupted_guards=[_guard()],
        first_guard=_guard(), resume_guard=_guard(),
        **COMPARE.binding_gates(BINDING))
    assert result["checks"]["uninterrupted_arm_count_sufficient"] is False
    assert "uninterrupted_arm_count_sufficient" in result["failures"]
    assert result["exact_status"] == "failed"


def test_protocol_binding_rejects_policy_drift(tmp_path):
    drifted = copy.deepcopy(PROTOCOL)
    policy = drifted["continuation_comparison"]["aa_envelope"]["policy"]
    policy["rule"] = "relaxed self-consistent rule"
    drifted["continuation_comparison"]["aa_envelope"]["policy_sha256"] = \
        FIXTURE.policy_digest(policy)
    path = tmp_path / "drifted_protocol.json"
    path.write_text(json.dumps(drifted))
    # Self-consistent but no longer the comparator's declared policy.
    with pytest.raises(ValueError, match="differs from this comparator"):
        COMPARE.protocol_binding(path)

    declared = copy.deepcopy(PROTOCOL)
    declared["continuation_comparison"]["aa_envelope"]["policy_sha256"] = "2" * 64
    path = tmp_path / "bad_hash_protocol.json"
    path.write_text(json.dumps(declared))
    with pytest.raises(ValueError, match="policy hash"):
        COMPARE.protocol_binding(path)


def test_fresh_output_and_fixture_retention_rules(tmp_path):
    fresh = tmp_path / "run"
    assert FIXTURE.require_fresh_output(fresh) == fresh
    fresh.mkdir()
    FIXTURE.require_fresh_output(fresh)
    (fresh / "run_report.json").write_text("{}")
    with pytest.raises(FileExistsError, match="not fresh"):
        FIXTURE.require_fresh_output(fresh)
    file_path = tmp_path / "receipt.json"
    file_path.write_text("{}")
    with pytest.raises(FileExistsError, match="already exists"):
        FIXTURE.require_fresh_output(file_path)

    retention = FIXTURE.validate_fixture_retention(PROTOCOL)
    assert retention["fixture_run_root"] != retention["retained_full_scope_checkpoint"]
    assert retention["checkpoint_writes"] == 1
    assert retention["final_model_exports"] == 0
    with pytest.raises(ValueError, match="overlap"):
        FIXTURE.validate_fixture_retention(_tampered(
            **{"retention.retained_full_scope_checkpoint":
               "runs/fused_continuation_fixture_v1/first/resume_checkpoint"}))
    with pytest.raises(ValueError, match="checkpoint write"):
        FIXTURE.validate_fixture_retention(_tampered(
            **{"storage_policy.checkpoint_writes": 2}))


def test_failure_classes_remain_distinct():
    assert FIXTURE.classify_failure(
        RuntimeError("resume boundary fingerprint mismatch")) == "boundary_mismatch"
    assert FIXTURE.classify_failure(MemoryError("alloc")) == "resource_stop"
    assert FIXTURE.classify_failure(
        RuntimeError("Capacity guard stopped the child")) == "resource_stop"
    assert FIXTURE.classify_failure(
        RuntimeError("Nonfinite training loss")) == "numerical_nonfinite"
    assert FIXTURE.classify_failure(
        RuntimeError("torch.load map_location failed")) == "checkpoint_restore_failure"
    assert FIXTURE.classify_failure(ValueError("Use fresh output")) == "protocol_violation"
    assert set(FIXTURE.FAILURE_CLASSES) >= {
        "boundary_mismatch", "numerical_nonfinite", "protocol_violation",
        "checkpoint_restore_failure", "resource_stop"}
