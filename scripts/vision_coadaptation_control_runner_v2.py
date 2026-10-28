#!/usr/bin/env python3
"""Conditional native runner for the lr 3e-05 co-adaptation control.

The runner is real executable code, not a runner-named stub.  It refuses before
any native import unless all of the following hold:

* the v2 co-adaptation protocol validates on CPU;
* a preserved smaller-step report records an optimization or functional
  failure;
* a preserved frozen-feature diagnosis binds that report and excludes the
  competing causes;
* a separate external authorization receipt binds this exact config SHA-256
  to a full committed protocol;
* the output path is fresh.

Only then does it load the sealed MiniCPM5 parent and the SigLIP2 encoder,
enable gradients on exactly the six projector tensors plus encoder layer 11
and ``vision_model.post_layernorm`` (24 tensors, 18,631,424 parameters), and
execute the identical five-stage schedule with four backward calls and one
NativeFusedAdamW step.  No decoder, backbone or frozen-encoder gradient is
permitted, and encoder drift is bounded by the declared gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab import vision_coadaptation as co  # noqa: E402
from replica_lab import vision_forward_accounting as fwd  # noqa: E402
from replica_lab import vision_smaller_step_accounting_v2 as acc_v2  # noqa: E402

DEFAULT_CONFIG = "configs/vision_coadaptation_control_v2.json"
REPORT_FORMAT = "replica.vision_coadaptation_control.report.v1"
STATUS_PASSED = "coadaptation_optimization_gate_passed"
STATUS_FAILED = "coadaptation_optimization_gate_failed"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise co.CoadaptationError(message)


def load_protocol(path: str | Path = DEFAULT_CONFIG, *,
                  root: str | Path = ROOT) -> dict[str, Any]:
    return co.validate_protocol(_read_json(Path(root) / path))


def review_launch(config_path: str | Path = DEFAULT_CONFIG, *,
                  authorization_path: str | Path | None = None,
                  root: str | Path = ROOT,
                  verify_commit: bool = True) -> dict[str, Any]:
    """Review-only gate; never imports a native runtime."""
    return co.review_launch(
        config_path, authorization_path, root=root,
        verify_commit=verify_commit)


# ---------------------------------------------------------------------------
# Native execution (imports are local and after authorization)
# ---------------------------------------------------------------------------


def _combined_parameters(adapter: Any, service: Any) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for name, parameter in adapter.named_parameters():
        _require(name not in combined,
                 f"Duplicate parameter name in adapter: {name}")
        combined[str(name)] = parameter
    for name, parameter in service.named_parameters():
        _require(name not in combined,
                 f"Duplicate parameter name across adapter/service: {name}")
        combined[str(name)] = parameter
    return combined


def _build_optimizer(config: Mapping[str, Any], combined: Mapping[str, Any]):
    from replica_lab.optimizers import NativeFusedAdamW
    optimizer_config = config["optimizer"]
    names = list(co.PROJECTOR_TENSORS) + list(co.ENCODER_TENSORS)
    parameters = [combined[name] for name in names]
    group = {
        "params": parameters,
        "param_names": names,
        "scope": "coadaptation",
        "group_name": "coadaptation_adamw_fused",
        "algorithm": "adamw_fused",
        "lr": float(optimizer_config["lr"]),
        "weight_decay": float(optimizer_config["weight_decay"]),
        "betas": tuple(optimizer_config["betas"]),
        "eps": float(optimizer_config["eps"]),
    }
    optimizer = NativeFusedAdamW([group])
    _require(len(optimizer.param_groups) == 1
             and len(optimizer.param_groups[0]["params"]) == len(names),
             "Optimizer must own exactly one group of 24 tensors")
    _require(sum(int(parameter.numel()) for parameter in parameters)
             == co.TRAINABLE_PARAMETERS,
             "Optimizer parameter total differs")
    return optimizer


def _coadaptation_update(*, adapter: Any, service: Any, session: Any,
                         tokenizer: Any, protocol: Mapping[str, Any],
                         scope: Any, batch: list[Mapping[str, Any]],
                         optimizer: Any, limits: Mapping[str, Any],
                         stops: Any, guard: Any, source_id: str,
                         config: Mapping[str, Any], index: int
                         ) -> tuple[dict[str, Any], int]:
    import torch
    from replica_lab.vision_balanced_alignment import (
        _verify_fit_tensor_contract)
    from run_vision_curriculum import (
        _example_tensors, curriculum_prompt, task_graph)

    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    role_losses: dict[str, list[float]] = {"answer": [], "caption": []}
    fit_checks: list[Any] = []
    backward_calls = 0
    combined = _combined_parameters(adapter, service)
    before = co.snapshot_tensor_state(combined.items(), co.SCOPE_TENSORS)
    encoder_before = co.snapshot_tensor_state(
        service.named_parameters(), co.ENCODER_TENSORS)

    for row in batch:
        kind = str(row["kind"])
        _require(kind in role_losses, f"Unknown update role: {kind}")
        visual = service.observe(
            row["image_path"], source_id=source_id,
            task_id=str(row["case_id"]), scope=scope, observed_at=0,
            source_epoch=session.ledger.epoch, training=True)
        _require(isinstance(visual, co.MutableVisualInput),
                 "Training observation must carry a grad-enabled visual")
        prompt_tensor, input_ids, labels = _example_tensors(
            session, tokenizer, row,
            maximum_prompt=limits["max_prompt_tokens"],
            maximum_sequence=limits["max_sequence_tokens"])
        fit_check = _verify_fit_tensor_contract(
            tokenizer, row, prompt_tensor, input_ids, labels, protocol,
            stop_tokens=stops)
        _require(fit_check is not None, "The fit contract report is missing")
        fit_checks.append(fit_check)
        prompt_text = curriculum_prompt(row["question"])
        with session.training_request(
                prompt_ids=prompt_tensor,
                prompt_mask=torch.ones_like(prompt_tensor, dtype=torch.bool),
                query=prompt_text, scope=scope,
                task=task_graph(row["case_id"], prompt_text),
                visual=visual, enabled_ports=(), core_impl="none",
                enabled_branches=("memory",), train_sense=True) as request:
            result = session.forward(
                request, input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                labels=labels)
            loss = result.loss
            _require(loss is not None and bool(torch.isfinite(loss)),
                     "Training loss is unavailable or nonfinite")
            _require(loss.requires_grad,
                     "Training loss is detached from autograd")
            session.backward(request, loss / len(batch))
        backward_calls += 1
        value = float(loss.detach().cpu())
        losses.append(value)
        role_losses[kind].append(value)

    ownership = co.scope_gradient_names(combined.items())
    co.assert_gradient_ownership(ownership)
    _require(not ownership["unexpected"],
             "A decoder or frozen encoder tensor received a gradient")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        [combined[name] for name in co.SCOPE_TENSORS],
        float(config["optimizer"]["max_grad_norm"]),
        error_if_nonfinite=True).detach().cpu())
    _require(math.isfinite(grad_norm), "Gradient norm is nonfinite")
    _require(adapter.parameter_stamp(original_only=True)
             == adapter._initial_base_stamp,
             "An update begins with a changed sealed base")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    after = co.snapshot_tensor_state(combined.items(), co.SCOPE_TENSORS)
    drift = co.drift_report(before, after, scope=co.SCOPE_TENSORS)
    changed = set(drift["changed_tensors"])
    _require(changed and changed <= set(co.SCOPE_TENSORS),
             f"Update changed undeclared tensors: {sorted(changed)}")
    encoder_after = co.snapshot_tensor_state(
        service.named_parameters(), co.ENCODER_TENSORS)
    encoder_drift = co.drift_report(
        encoder_before, encoder_after, scope=co.ENCODER_TENSORS)
    co.validate_drift(encoder_drift, limits=co.ENCODER_DRIFT)
    _require(adapter.parameter_stamp(original_only=True)
             == adapter._initial_base_stamp,
             "An update changed the sealed base")
    _require(not any(parameter.requires_grad for name, parameter
                     in adapter.base.named_parameters()),
             "A decoder tensor remains trainable")
    for name, parameter in service.named_parameters():
        expected = name in co.ENCODER_TENSORS
        _require(bool(parameter.requires_grad) is expected,
                 f"Encoder ownership differs: {name}")

    record = {
        "format": "replica.vision_coadaptation.update.v2",
        "label": "seed29_prefix",
        "update": int(index),
        "roles": {"answer": len(role_losses["answer"]),
                  "caption": len(role_losses["caption"])},
        "example_ids": [str(row["example_id"]) for row in batch],
        "case_ids": [str(row["case_id"]) for row in batch],
        "losses": losses,
        "loss": sum(losses) / len(losses),
        "gradient_audit": {
            "within_scope": ownership["within_scope"],
            "present": ownership["present"],
            "nonzero": ownership["nonzero"],
            "nonfinite": ownership["nonfinite"],
            "unexpected": ownership["unexpected"],
        },
        "grad_norm": grad_norm,
        "backward_calls": backward_calls,
        "optimizer_steps": 1,
        "optimizer_owned_tensors": len(co.SCOPE_TENSORS),
        "optimizer_param_groups": 1,
        "changed_scope_tensors": sorted(changed),
        "changed_projector_tensors": sorted(
            name for name in changed if name in co.PROJECTOR_TENSORS),
        "changed_encoder_tensors": sorted(
            name for name in changed if name in co.ENCODER_TENSORS),
        "nonprojector_gradient_names": ownership["unexpected"],
        "decoder_gradient_names": [],
        "encoder_drift": encoder_drift,
        "scope_drift": drift,
        "base_stamp_unchanged": True,
        "response_only_labels": True,
        "fit_contract_checked": True,
        "fit_contract_rows": fit_checks,
    }
    if guard is not None:
        guard.sample(index=int(index), event="coadaptation_update")
    return record, backward_calls


def _evaluate_optimization(config: Mapping[str, Any], *,
                           update: Mapping[str, Any],
                           pre: Mapping[str, Any], post: Mapping[str, Any],
                           prefix_0: Mapping[str, Any],
                           prefix_1: Mapping[str, Any]) -> dict[str, Any]:
    gate = config["gates"]["optimization"]
    violations: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)

    check(int(update["backward_calls"]) == int(gate["backward_calls"]),
          "optimization:backward_calls")
    check(int(update["optimizer_steps"]) == int(gate["optimizer_steps"]),
          "optimization:optimizer_steps")
    changed = set(update["changed_scope_tensors"])
    check(changed and changed <= set(co.SCOPE_TENSORS),
          "optimization:undeclared_change")
    check(not update["decoder_gradient_names"]
          and not update["nonprojector_gradient_names"],
          "optimization:out_of_scope_gradient")
    check(update["base_stamp_unchanged"] is True,
          "optimization:base_weight_change")
    check(update["response_only_labels"] is True,
          "optimization:response_mask")
    losses = [float(value) for value in update["losses"]]
    check(bool(losses) and all(math.isfinite(value) for value in losses),
          "optimization:finite_update_losses")
    check(math.isfinite(float(update["grad_norm"])), "optimization:grad_norm")
    drift = update["encoder_drift"]
    check(drift["max_abs_update"]
          <= float(gate["max_encoder_abs_update"]),
          "optimization:encoder_max_abs_drift")
    check(drift["delta_l2"] <= float(gate["max_encoder_delta_l2"]),
          "optimization:encoder_l2_drift")
    try:
        pre_mean = float(pre["mean_response_nll"])
        post_mean = float(post["mean_response_nll"])
        update_delta = post_mean - pre_mean
        answer_delta = (float(prefix_1["teacher_forced"]["answer"])
                        - float(prefix_0["teacher_forced"]["answer"]))
        caption_delta = (float(prefix_1["teacher_forced"]["caption"])
                         - float(prefix_0["teacher_forced"]["caption"]))
    except (KeyError, TypeError, ValueError):
        return {"ok": False,
                "violations": violations + ["optimization:missing_metrics"],
                "metrics": {}}
    check(math.isfinite(update_delta)
          and update_delta <= float(gate["max_update_batch_mean_nll_delta"]),
          "optimization:update_batch_nll_delta")
    check(math.isfinite(answer_delta)
          and answer_delta <= float(gate["max_wider_answer_mean_nll_delta"]),
          "optimization:wider_answer_nll_delta")
    check(math.isfinite(caption_delta)
          and caption_delta <= float(gate["max_wider_caption_mean_nll_delta"]),
          "optimization:wider_caption_nll_delta")
    return {
        "ok": not violations,
        "violations": violations,
        "metrics": {
            "update_batch_pre_mean_response_nll": pre_mean,
            "update_batch_post_mean_response_nll": post_mean,
            "update_batch_delta": update_delta,
            "wider_answer_delta": answer_delta,
            "wider_caption_delta": caption_delta,
            "encoder_max_abs_update": drift["max_abs_update"],
            "encoder_delta_l2": drift["delta_l2"],
            "max_encoder_abs_update": gate["max_encoder_abs_update"],
            "max_encoder_delta_l2": gate["max_encoder_delta_l2"],
            "loss_movement_is_not_conversion": True,
        },
        "status_on_fail": gate["status_on_fail"],
    }


def _validate_coadaptation_accounting(
        config: Mapping[str, Any], totals: Mapping[str, Any],
        update: Mapping[str, Any], pre: Mapping[str, Any],
        post: Mapping[str, Any],
        readouts: list[Mapping[str, Any]]) -> dict[str, Any]:
    counters = dict(totals["counters"])
    acc_v2.validate_live_counters(
        counters, label="campaign", training_forwards=8)
    campaign = config["accounting"]["campaign_cumulative"]
    for key, cap in campaign.items():
        _require(counters[key] <= int(cap),
                 f"Campaign cap exceeded: {key} {counters[key]} > {cap}")
    expected = config["readouts"]["expected_measured_work"]
    for key in ("prepare_sense_passes", "explicit_logits_forwards",
                "backward_calls", "optimizer_steps"):
        _require(counters[key] == expected[key],
                 f"Schedule accounting differs: {key}")
    _require(counters["decoder_forward_calls"]
             == expected["decoder_forward_calls"],
             "Campaign decoder count differs from the derived identity")
    _require(counters["visual_observation_calls"]
             == expected["visual_observation_calls"]
             and counters["visual_encoder_calls"]
             == expected["visual_encoder_calls_max"],
             "Campaign visual observation/encoder counts differ")
    _require(counters["visual_forward_calls"]
             < counters["decoder_forward_calls"],
             "Visual/projector passes must not equal all decoder passes")
    for name, evaluation in (("pre", pre), ("post", post)):
        measured = dict(evaluation["measured"]["counters"])
        _require(measured["decoder_forward_calls"] == 8
                 and measured["prepare_sense_passes"] == 4
                 and measured["explicit_logits_forwards"] == 4,
                 f"{name}: update-batch decomposition differs")
        _require(measured["visual_observation_calls"] == 4
                 and measured["visual_encoder_calls"] == 4,
                 f"{name}: update-batch visual counts differ")
        _require(measured["generated_tokens"] == 0
                 and measured["generation_prefill_calls"] == 0
                 and measured["generation_decode_calls"] == 0,
                 f"{name}: update-batch evaluation generated text")
        _require(measured["backward_calls"] == 0
                 and measured["optimizer_steps"] == 0,
                 f"{name}: update-batch evaluation changed weights")
    _require(update["backward_calls"] == 4
             and update["optimizer_steps"] == 1,
             "The single update did not make four backward calls and one step")
    _require(set(update["changed_scope_tensors"]) <= set(co.SCOPE_TENSORS),
             "The update changed an out-of-scope tensor")
    _require(update["nonprojector_gradient_names"] == []
             and update["decoder_gradient_names"] == [],
             "The update produced out-of-scope gradients")
    _require(len(readouts) == 2
             and [readout["label"] for readout in readouts]
             == ["prefix_0", "prefix_1"],
             "Wider-cohort readouts differ")
    return {
        "ok": True,
        "counters": counters,
        "expected": expected,
        "update_batch_pre": pre["measured"]["counters"],
        "update_batch_post": post["measured"]["counters"],
        "arithmetic": ("decoder = prepares + logits + generated tokens + "
                       "8 training forwards"),
    }


def run_protocol(config_path: str | Path = DEFAULT_CONFIG, *,
                 output: str | Path, model_path: str | Path,
                 model_lock: str | Path, vision_source: str | Path,
                 authorization_path: str | Path | None = None,
                 root: str | Path = ROOT) -> dict[str, Any]:
    """Authorized native entrypoint.  Authorization precedes native imports."""
    root_path = Path(root)
    review = review_launch(
        config_path, authorization_path=authorization_path, root=root_path,
        verify_commit=True)
    _require(review["launch_allowed"],
             "Native execution refused before model load: "
             f"{review['refusals']}")
    config = load_protocol(config_path, root=root_path)
    _require(not Path(output).exists(),
             f"Fresh output is required; path exists: {output}")
    _require(Path(model_path).resolve()
             == (root_path / config["parent"]["model_path"]).resolve(),
             "CLI model path differs from the pinned MiniCPM5 parent")
    _require(Path(model_lock).resolve()
             == (root_path / config["parent"]["model_lock"]).resolve(),
             "CLI model lock differs from the pinned source lock")
    _require(Path(vision_source).resolve()
             == (root_path / config["parent"]["vision_source_manifest"]
                 ).resolve(),
             "CLI vision source differs from the pinned manifest")

    # Native imports begin only after the review gate and input checks.
    import torch
    from replica_lab import vision_learnability_curve_v2 as curve_v2
    from replica_lab.architecture.loading import (
        exclusive_mps_worker, load_local_mps)
    from replica_lab.profiling import RunProfiler
    from replica_lab.vision_collapse_diagnostic import (
        EXPECTED_STOP_TOKENS, control_plan, resolve_stop_tokens)
    from run_vision_curriculum import (
        Guard, initialize_projector, make_session, protocol_scope,
        verify_source_and_spec)
    import run_vision_balanced_alignment as balanced
    import run_vision_learnability_curve_v2 as runner_v2
    import vision_smaller_step_v2 as ss

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=False)
    protocol_path = root_path / config["prepared"]["protocol"]
    protocol = _read_json(protocol_path)
    runner_v2._write_json(output_path / "config.json", dict(config))
    (output_path / "protocol.json").write_bytes(protocol_path.read_bytes())
    started = time.monotonic()
    guard = None
    profiler = None
    profiler_stopped = False
    accounting_loads = runner_v2.LoadAccounting()
    readouts: list[dict[str, Any]] = []
    try:
        training = balanced._load_training_v2(protocol)
        batch = list(training["batches"][0])
        _require([str(row["example_id"]) for row in batch]
                 == config["update_batch"]["example_ids"],
                 "The first four-row update batch differs")
        _require([row["kind"] for row in batch].count("answer") == 2
                 and [row["kind"] for row in batch].count("caption") == 2,
                 "The update batch role mix differs")
        plan = control_plan(training["training"])
        scope = protocol_scope(protocol)
        budget = ss._ScheduleBudget(config)
        accounting = fwd.ForwardAccounting(budget=budget)
        guard = Guard(protocol)
        with exclusive_mps_worker(root_path):
            profiler = RunProfiler("mps", output_path)
            profiler.start()
            try:
                source, spec, manifest_path = verify_source_and_spec(
                    protocol, "mps")
                adapter, tokenizer, _ = runner_v2._accounted_load(
                    lambda: load_local_mps(
                        model_path, model_lock, vision=spec,
                        profiler=profiler), accounting_loads, "model")
                guard.sample(index=0, event="model_loaded")
                profiler.mark("model_loaded")
                stops, stop_source = resolve_stop_tokens(
                    getattr(adapter.base.config, "eos_token_id", None),
                    getattr(tokenizer, "eos_token_id", None))
                _require(tuple(stops) == tuple(EXPECTED_STOP_TOKENS),
                         f"Model stop set {stops} differs")
                _require(int(tokenizer.eos_token_id) == 1,
                         "Tokenizer EOS differs from the declared target")
                adapter.eval()
                adapter.set_regime("modules_only")
                adapter._initial_base_stamp = adapter.parameter_stamp(
                    original_only=True)
                init_hash = initialize_projector(
                    adapter.vision_projector,
                    int(config["initialization"]["init_seed"]))
                _require(init_hash
                         == config["initialization"]["expected_init_sha256"],
                         "Fresh seed-29 projector differs from the anchor")
                service = co.MutableEncoderFeatureService.load(
                    source.root, manifest_path, device="mps")
                _require(service.spec.digest == spec.digest,
                         "Mutable encoder spec differs from the protocol")
                session = make_session(adapter, tokenizer, output_path)
                combined = _combined_parameters(adapter, service)
                scope_report = co.enable_scope(combined)
                optimizer = _build_optimizer(config, combined)
                service_audit = service.encoder_audit()
                _require(service_audit["tensor_count"] > 0,
                         "Mutable encoder exposes no tensors")
                _require(sorted(service_audit["trainable_names"])
                         == sorted(co.ENCODER_TENSORS),
                         "Mutable encoder trainable scope differs")
                labels = training["train_caption_labels"]
                caption_labels = {str(row["case_id"]): row for row in labels}
                with accounting.wrap_native(adapter), \
                        accounting.wrap_feature_service(service), \
                        accounting.measured_session(session):
                    budget.select("wider")
                    readout_0 = ss._normalise_readout(
                        config, runner_v2._measured_readout(
                            adapter, session, tokenizer, service, protocol,
                            scope, training["training"], plan, labels, stops,
                            prefix=0, label="prefix_0",
                            expected_prefixes=[0, 1], prefix_position=0,
                            output=output_path, guard=guard,
                            started=started, limits=config["limits"],
                            optimizer=optimizer, accounting=accounting))
                    runner_v2._write_json(
                        runner_v2._readout_directory(output_path, "prefix_0")
                        / "readout.json", readout_0)
                    readouts.append(readout_0)

                    budget.select("update_batch")
                    pre = ss._measure_update_batch_loss(
                        adapter=adapter, session=session, tokenizer=tokenizer,
                        service=service, protocol=protocol, scope=scope,
                        batch=batch, labels=caption_labels,
                        limits=config["limits"], accounting=accounting,
                        guard=guard, started=started, output=output_path,
                        prefix=1, label="update_batch_pre", config=config)

                    budget.select("update")
                    runner_v2._check_wall(
                        started, config["limits"]["max_seconds"],
                        "co-adaptation update")
                    with accounting.forced_phase(fwd.PHASE_TRAINING):
                        with accounting.counted_optimizer_step(optimizer):
                            update, calls = _coadaptation_update(
                                adapter=adapter, service=service,
                                session=session, tokenizer=tokenizer,
                                protocol=protocol, scope=scope, batch=batch,
                                optimizer=optimizer, limits=config["limits"],
                                stops=stops, guard=guard,
                                source_id=protocol["vision_source"]["source_id"],
                                config=config, index=1)
                    _require(calls == 4 and update["backward_calls"] == 4,
                             "The single update did not make four backward "
                             "calls")
                    runner_v2._write_jsonl(
                        output_path / "training_trace.jsonl", [update])
                    mask_audit = curve_v2.validate_response_masks(
                        update["fit_contract_rows"])
                    _require(mask_audit["ok"],
                             f"Update mask audit failed: "
                             f"{mask_audit['violations']}")
                    runner_v2._write_json(
                        output_path / "response_mask_audit.json", mask_audit)

                    budget.select("update_batch")
                    post = ss._measure_update_batch_loss(
                        adapter=adapter, session=session, tokenizer=tokenizer,
                        service=service, protocol=protocol, scope=scope,
                        batch=batch, labels=caption_labels,
                        limits=config["limits"], accounting=accounting,
                        guard=guard, started=started, output=output_path,
                        prefix=2, label="update_batch_post", config=config)

                    budget.select("wider")
                    readout_1 = ss._normalise_readout(
                        config, runner_v2._measured_readout(
                            adapter, session, tokenizer, service, protocol,
                            scope, training["training"], plan, labels, stops,
                            prefix=1, label="prefix_1",
                            expected_prefixes=[0, 1], prefix_position=1,
                            output=output_path, guard=guard,
                            started=started, limits=config["limits"],
                            optimizer=optimizer, accounting=accounting))
                    runner_v2._write_json(
                        runner_v2._readout_directory(output_path, "prefix_1")
                        / "readout.json", readout_1)
                    readouts.append(readout_1)

                    totals = accounting.historical_totals()
                    measured_work = _validate_coadaptation_accounting(
                        config, totals, update, pre, post, readouts)
                    runner_v2._write_json(
                        output_path / "accounting.json", totals)
                optimization = _evaluate_optimization(
                    config, update=update, pre=pre, post=post,
                    prefix_0=readout_0, prefix_1=readout_1)
                functional = ss.evaluate_functional_fit_gate(
                    config, prefix_0=readout_0, prefix_1=readout_1)
                profiler_stopped = True
                profiler.stop()
                return _finish_report(
                    config, config_path, review, output=output_path,
                    started=started, guard=guard, profiler=profiler,
                    accounting_loads=accounting_loads,
                    accounting_totals=totals,
                    measured_work=measured_work, readouts=readouts,
                    update=update, pre=pre, post=post,
                    optimization=optimization, functional=functional,
                    init_hash=init_hash, stop_source=stop_source,
                    scope_report=scope_report,
                    service_audit=service_audit,
                    service_cache=service.cache_audit(),
                    stop_tokens=stops)
            finally:
                if profiler is not None and not profiler_stopped:
                    try:
                        profiler.stop()
                    except Exception:  # noqa: BLE001
                        pass
    except Exception as exc:
        failure = {
            "format": REPORT_FORMAT + ".failure",
            "generated_at_utc": runner_v2._now(),
            "status": "stopped_engineering_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "native_execution_attempted": True,
            "development_authorized": False,
            "heldout_authorized": False,
            "v0_authorized": False,
            "promotion": False,
            "retry_authorized": False,
            "more_coadaptation_authorized": False,
        }
        try:
            runner_v2._write_json(output_path / "failure_report.json", failure)
        except Exception:  # noqa: BLE001
            pass
        raise


def _finish_report(config: Mapping[str, Any], config_path: str | Path,
                   review: Mapping[str, Any], *, output: str | Path,
                   started: float, guard: Any, profiler: Any,
                   accounting_loads: Any,
                   accounting_totals: Mapping[str, Any],
                   measured_work: Mapping[str, Any],
                   readouts: list[dict[str, Any]],
                   update: Mapping[str, Any], pre: Mapping[str, Any],
                   post: Mapping[str, Any], optimization: Mapping[str, Any],
                   functional: Mapping[str, Any], init_hash: str,
                   stop_source: str, scope_report: Mapping[str, Any],
                   service_audit: Mapping[str, Any],
                   service_cache: Mapping[str, Any]) -> dict[str, Any]:
    from run_vision_learnability_curve_v2 import _now, _write_json
    report = {
        "format": REPORT_FORMAT,
        "name": config["name"],
        "generated_at_utc": _now(),
        "status": (STATUS_PASSED if optimization["ok"] and functional["ok"]
                   else STATUS_FAILED),
        "command": list(review["launch_command"]),
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "protocol_commit": review["authorization"]["protocol_commit"],
        "scope": dict(scope_report),
        "encoder_audit": dict(service_audit),
        "feature_cache": dict(service_cache),
        "initialization": {"sha256": init_hash,
                           "expected": config["initialization"]
                           ["expected_init_sha256"]},
        "generation_stop_set": {
            "tokens": [int(value) for value in stop_tokens],
            "source": str(stop_source),
        },
        "readouts": readouts,
        "readout_sequence": [item["label"] for item in readouts],
        "training": {
            "update": dict(update),
            "update_batch_pre": pre,
            "update_batch_post": post,
        },
        "accounting": dict(accounting_totals),
        "measured_work": dict(measured_work),
        "optimization": optimization,
        "functional_fit": functional,
        "artifact_bytes": _artifact_bytes(output),
        "artifact_limit_bytes": config["limits"]["artifact_bytes"],
        "wall_seconds": time.monotonic() - started,
        "wall_limit_seconds": config["limits"]["max_seconds"],
        "load_counts": getattr(accounting_loads, "snapshot", lambda: {})(),
        "guard": getattr(guard, "snapshot", lambda: {})(),
        "profiler": getattr(profiler, "snapshot", lambda: {})(),
        "decision": {
            "development_authorized": False,
            "heldout_authorized": False,
            "v0_authorized": False,
            "semantic_audit_authorized": False,
            "promotion": False,
            "retry_authorized": False,
            "more_coadaptation_authorized": False,
            "requires_new_predeclared_protocol": True,
        },
        "claims": {
            "rank_improvement_is_conversion": False,
            "loss_movement_is_conversion": False,
            "generated_answer_conversion_claimed": False,
            "encoder_swap": False,
            "new_loss_introduced": False,
            "budget_increased": False,
        },
        "limitations": list(config["limitations"]),
    }
    _write_json(Path(output) / "report.json", report)
    return report


def _artifact_bytes(output: str | Path) -> int:
    total = 0
    for path in sorted(Path(output).rglob("*")):
        if path.is_file():
            total += path.stat().st_size
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conditional co-adaptation control review and native "
                    "execution")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="CPU protocol validation")
    validate.add_argument("--config", default=DEFAULT_CONFIG)
    validate.add_argument("--json", action="store_true")
    review = sub.add_parser(
        "review-launch",
        help="refuse or review the launch; never imports a native runtime")
    review.add_argument("--config", default=DEFAULT_CONFIG)
    review.add_argument("--authorization", default=None)
    review.add_argument("--no-commit-check", action="store_true",
                        help="test-only: skip the git blob binding")
    run = sub.add_parser("run", help="authorized native execution")
    run.add_argument("--config", default=DEFAULT_CONFIG)
    run.add_argument("--output", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--model-lock", required=True)
    run.add_argument("--vision-source", required=True)
    run.add_argument("--authorization", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            config = load_protocol(args.config)
            summary = {
                "status": "protocol_valid_preparation_only",
                "native_execution": False,
                "model_loads": 0,
                "config": args.config,
                "config_sha256": _sha256_file(args.config),
                "name": config["name"],
                "scope": {
                    "trainable_tensors": len(co.SCOPE_TENSORS),
                    "trainable_parameter_count": co.TRAINABLE_PARAMETERS,
                },
                "budget": config["accounting"]["campaign_cumulative"],
                "eligible": co.eligibility_state(
                    config, root=ROOT)["eligible"],
            }
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                print(f"valid: {args.config} sha256="
                      f"{summary['config_sha256']}")
            return 0
        if args.command == "review-launch":
            result = review_launch(
                args.config, authorization_path=args.authorization,
                verify_commit=not args.no_commit_check)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["launch_allowed"] else 2
        config = load_protocol(args.config)
        run_protocol(
            args.config, output=args.output, model_path=args.model_path,
            model_lock=args.model_lock, vision_source=args.vision_source,
            authorization_path=args.authorization)
        return 0
    except co.AuthorizationRefused as error:
        print(f"launch refused: {error}")
        return 2
    except co.CoadaptationError as error:
        print(f"protocol invalid: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
