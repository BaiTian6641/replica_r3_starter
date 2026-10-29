"""P1 direct response-CE reference for source-free plastic memory.

This module is deliberately an *engineering reference*, not a native capability
claim.  It keeps three owners separate:

* ``SparkTextMemoryPath.response_optimizer`` owns the spine response extension.
  A v2 caller may instead declare it as the port-only owner described below.
* ``ExpertBank.optimizer`` (driven by ``PlasticLearningController``) owns P.
* An optional separate ``NativeFusedAdamW`` owns only
  ``extension.spine.ports.*.output_projection`` tensors.  The v2 integration may
  reuse the declared port-only object as ``path.response_optimizer``.
* The contextual fixture used by the tests owns the tiny frozen decoder.

A direct response cross-entropy backward pass produces gradients for the
selected P expert, and those gradients are handed to the bank's existing
single-owner transaction.  The spine response extension is never stepped here
and ``SparkTextMemoryPath.response_step`` is not called.  When a caller
explicitly declares a port-only optimizer as ``path.response_optimizer``, only
that port-owned object is stepped.  A successful parameter delta, Adam moments
and clock/revision advance are still only evidence that the mechanics executed;
transfer, retention and useful learning remain unevaluated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from replica_lab.architecture.spine_text import (
    SparkTextMemoryPath, TextQuery,
)
from replica_lab.nested_learning import TeachingClock
from replica_lab.optimizers import NativeFusedAdamW

P1_SCHEMA = "replica.p1_direct_response_ce.v1"
ENGINEERING_LABEL = "engineering_cpu_fixture_only"
SEALED_EVALUATION = "closed"
NATIVE_SEALED_LAYERS = 36
LATE_SITE_MIN_LAYERS = 32

DETACH_BOUNDARIES = (
    "prepare_query detaches native_hidden before P/E reads; no decoder "
    "activation gradient crosses the query-sensing pass",
    "plastic_features returns detached query features and is used only for "
    "the controller transaction fingerprint, not for the response loss",
    "P-only packets carry zero evidence payload bytes and empty evidence IDs; "
    "P slots come from the selected expert over current query features",
    "the direct response-CE graph reaches frozen decoder activations and the "
    "selected P expert; spine response gradients are discarded and never stepped",
    "teacher-forced labels are detached token IDs and contribute no gradient",
    "the optional port-only NativeFusedAdamW receives only the already computed "
    "direct-CE gradients of extension.spine.ports.*.output_projection; P, "
    "non-port spine and response parameters never enter that owner",
    "post-update fit evaluation is a fresh forward with no backward or optimizer step",
)

ARMS = (
    "full_eligible", "selected_exact", "e_only", "p_only", "e_plus_p",
    "wrong_memory", "absent_memory",
)


def site_accounting(sites: Sequence[int], num_layers: int, *,
                    native_layers: int = NATIVE_SEALED_LAYERS) -> dict[str, Any]:
    """Charge injection depth and state the native comparison boundary."""
    values = tuple(int(site) for site in sites)
    if not values or len(set(values)) != len(values):
        raise ValueError("sites must be nonempty and distinct")
    if any(site < 0 or site >= num_layers for site in values):
        raise ValueError("a site lies outside the decoder")
    earliest = min(values)
    latest = max(values)
    return {
        "sites": list(values),
        "num_layers": int(num_layers),
        "injection_count_per_forward": len(values),
        "sensing_forward_layers": int(num_layers),
        "port_injections_per_forward": len(values),
        "p_backward_suffix_layers_upper_bound": int(num_layers - earliest),
        "late_site_min_layers": LATE_SITE_MIN_LAYERS,
        "requires_at_least_layers": int(latest + 1),
        "fixture_geometry_valid": bool(latest < num_layers),
        "native_sealed_layers": int(native_layers),
        "native_sites_valid": bool(latest < native_layers),
        "native_comparison": (
            "blocked_geometry" if latest >= native_layers
            else "eligible_geometry_native_backend_absent"),
        "late_site_native_status": (
            "not_a_late_site" if latest < 23
            else "blocked_geometry" if latest >= native_layers
            else "geometry_valid_native_backend_absent"),
        "native_execution": "not_run",
        "native_backend": "absent",
        "engineering_fixture_only": True,
        "native_capability_claim": False,
    }


def selected_parameter_names(path: SparkTextMemoryPath, expert_id: int = 0
                              ) -> tuple[str, ...]:
    expert = path.experiment.bank.expert(int(expert_id))
    return tuple(name for name, _ in expert.named_parameters())


def port_output_parameter_names(path: SparkTextMemoryPath) -> tuple[str, ...]:
    """Return exactly the zero-init port output projections, no other tensor."""
    return tuple(
        name for name, _ in path.adapter.new_named_parameters()
        if name.startswith("extension.spine.ports.")
        and ".output_projection." in name)


def _port_output_parameters(path: SparkTextMemoryPath
                            ) -> tuple[tuple[str, Tensor], ...]:
    names = set(port_output_parameter_names(path))
    return tuple(
        (name, parameter) for name, parameter in path.adapter.new_named_parameters()
        if name in names)


def build_port_optimizer(
        path: SparkTextMemoryPath, *, lr: float = 1e-3,
        betas: Sequence[float] = (0.9, 0.999), eps: float = 1e-8,
        weight_decay: float = 0.0) -> NativeFusedAdamW:
    """Build the separate NativeFusedAdamW owner for port output projections.

    The response optimizer is deliberately not reused.  Only
    ``extension.spine.ports.*.output_projection`` tensors enter this owner.
    """
    parameters = _port_output_parameters(path)
    if not parameters:
        raise ValueError("the path has no port output projections to optimize")
    if not (isinstance(lr, (int, float)) and lr > 0):
        raise ValueError("port optimizer lr must be positive")
    return NativeFusedAdamW([{
        "params": [parameter for _, parameter in parameters],
        "lr": float(lr), "betas": tuple(betas), "eps": float(eps),
        "weight_decay": float(weight_decay),
        "algorithm": "adamw_fused", "fused": True,
        "group_name": "spine_port_output_projection",
        "scope": "p1_port_only",
        "param_names": [name for name, _ in parameters],
    }], optimizer_name="adamw_fused")


def _validate_port_optimizer(path: SparkTextMemoryPath,
                             optimizer: NativeFusedAdamW) -> None:
    if not isinstance(optimizer, NativeFusedAdamW):
        raise TypeError("port_optimizer must be a NativeFusedAdamW")
    # A v2 path may explicitly replace ``response_optimizer`` with this
    # port-only owner and pass the same object here.  Scope, not object
    # identity, decides whether it may receive port gradients.
    expected = {id(parameter) for _, parameter in _port_output_parameters(path)}
    actual = {id(parameter) for group in optimizer.param_groups
              for parameter in group["params"]}
    if actual != expected:
        raise ValueError(
            "port_optimizer must contain only extension.spine.ports.*."
            "output_projection parameters")


def _optimizer_state_by_name(
        optimizer: NativeFusedAdamW,
        parameters: Sequence[tuple[str, Tensor]]) -> dict[str, dict[str, Any] | None]:
    state = {}
    for name, parameter in parameters:
        current = optimizer.state.get(parameter)
        if current is None:
            state[name] = None
        else:
            state[name] = {
                key: (value.detach().clone() if isinstance(value, Tensor) else value)
                for key, value in current.items()}
    return state


def _step_counts(state: Mapping[str, Mapping[str, Any] | None]) -> dict[str, int]:
    return {name: (0 if row is None else int(row.get("step", 0)))
            for name, row in state.items()}


def _clone_parameters(expert) -> dict[str, Tensor]:
    return {name: parameter.detach().clone()
            for name, parameter in expert.named_parameters()}


def _shape(value: Tensor | None) -> list[int] | None:
    return None if value is None else list(value.shape)


def _dtype(value: Tensor | None) -> str | None:
    return None if value is None else str(value.dtype)


def _tensor_digest(value: Tensor) -> str:
    array = value.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()
    return hashlib.sha256(array).hexdigest()


def _state_step(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.detach().item())
    return int(value)


def _moment_summary(state: Mapping[str, Mapping[str, Any] | None]) -> dict:
    rows = {}
    for name, current in state.items():
        if current is None:
            rows[name] = None
            continue
        row = {}
        if "step" in current:
            row["step"] = _state_step(current["step"])
        for key in ("exp_avg", "exp_avg_sq"):
            value = current.get(key)
            if isinstance(value, Tensor):
                row[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "l2": float(value.detach().float().norm()),
                    "sha256": _tensor_digest(value),
                }
        rows[name] = row
    return rows


def _state_equal(left: Mapping[str, Any] | None,
                 right: Mapping[str, Any] | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, Tensor) and isinstance(b, Tensor):
            if a.shape != b.shape or not torch.equal(a, b):
                return False
        elif a != b:
            return False
    return True


def _moments_changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    changed = []
    for name in sorted(set(before) | set(after)):
        if not _state_equal(before.get(name), after.get(name)):
            changed.append(name)
    return changed


def _optimizer_digest(optimizer) -> str:
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    for index, row in sorted(state.get("state", {}).items(),
                             key=lambda item: str(item[0])):
        digest.update(str(index).encode("utf-8"))
        for key in sorted(row):
            digest.update(str(key).encode("utf-8"))
            value = row[key]
            if isinstance(value, Tensor):
                digest.update(_tensor_digest(value).encode("utf-8"))
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PStateSnapshot:
    parameters: dict[str, Tensor]
    optimizer_state: dict[str, dict[str, Any] | None]
    expert_steps: tuple[int, ...]
    clock_events: int
    revision: int

    def to_dict(self) -> dict:
        return {
            "parameter_names": sorted(self.parameters),
            "optimizer_state_keys": sorted(self.optimizer_state),
            "expert_steps": list(self.expert_steps),
            "clock_events": self.clock_events,
            "revision": self.revision,
        }


def capture_p_state(path: SparkTextMemoryPath, expert_id: int = 0) -> PStateSnapshot:
    expert = path.experiment.bank.expert(int(expert_id))
    return PStateSnapshot(
        parameters=_clone_parameters(expert),
        optimizer_state=path.experiment.bank.optimizer_state_for_expert(
            int(expert_id)),
        expert_steps=tuple(path.experiment.bank.expert_steps),
        clock_events=path.experiment.controller.clock.eligible_events,
        revision=path.experiment.controller.revision)


def restore_p_state(path: SparkTextMemoryPath, snapshot: PStateSnapshot, *,
                    expert_id: int = 0, restore_clock: bool = True,
                    restore_revision: bool = True,
                    restore_expert_steps: bool = True) -> dict[str, Any]:
    """Restore P values/moments in place; parameter ownership is unchanged.

    The dedup ledger is intentionally not rewound.  A reset/wrong-state control
    therefore uses a distinct event ID; this keeps duplicate-event protection
    honest instead of silently replaying a committed transaction.
    """
    bank = path.experiment.bank
    expert_id = int(expert_id)
    owner_ids_before = {
        name: id(parameter)
        for name, parameter in bank.expert(expert_id).named_parameters()}
    bank.restore_parameters({expert_id: snapshot.parameters})
    bank.restore_optimizer_state(snapshot.optimizer_state)
    if restore_expert_steps:
        bank.expert_steps[:] = list(snapshot.expert_steps)
    if restore_clock:
        path.experiment.controller.clock = TeachingClock(snapshot.clock_events)
    if restore_revision:
        path.experiment.controller.revision = snapshot.revision
        path.experiment.controller.cache.bind(snapshot.revision)
    path.queries.invalidate()
    path.experiment.invalidate(reason="p1_reference_state_restore")
    current = _clone_parameters(bank.expert(expert_id))
    return {
        "restored": all(torch.equal(current[name], snapshot.parameters[name])
                        for name in snapshot.parameters),
        "expert_id": expert_id,
        "clock_events": path.experiment.controller.clock.eligible_events,
        "revision": path.experiment.controller.revision,
        "dedup_rewound": False,
        "owner_identity_unchanged": owner_ids_before == {
            name: id(parameter)
            for name, parameter in bank.expert(expert_id).named_parameters()},
    }


def _response_ce_gradients(path: SparkTextMemoryPath, query: TextQuery,
                           labels: Tensor, *, expert_id: int = 0
                           ) -> dict[str, Any]:
    """One forward/backward over direct response CE; returns selected P grads."""
    path.adapter.zero_grad(set_to_none=True)
    path.experiment.bank.zero_gradients()
    response = path.respond(query, labels=labels)
    loss = response.loss
    if loss is None or not bool(torch.isfinite(loss)):
        raise ValueError("direct response CE loss is missing or nonfinite")
    loss.backward()
    expert = path.experiment.bank.expert(int(expert_id))
    gradients: dict[str, Tensor] = {}
    gradient_norms: dict[str, float] = {}
    for name, parameter in expert.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"selected P parameter {name} received no gradient")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"selected P parameter {name} has a nonfinite gradient")
        gradients[name] = parameter.grad.detach().clone()
        gradient_norms[name] = float(parameter.grad.detach().float().norm())
    inactive_nonzero = []
    for other in range(path.experiment.bank.expert_count):
        if other == int(expert_id):
            continue
        for name, parameter in path.experiment.bank.expert(other).named_parameters():
            if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)):
                inactive_nonzero.append(f"expert_{other}.{name}")
    adapter_gradients = path.adapter.gradient_audit()
    return {
        "response": response,
        "loss": float(loss.detach()),
        "gradients": gradients,
        "gradient_norms": gradient_norms,
        "gradient_norm_sum": float(sum(gradient_norms.values())),
        "inactive_nonzero_gradient_names": inactive_nonzero,
        "adapter_gradient_audit": adapter_gradients,
    }


def evaluate_response_ce(path: SparkTextMemoryPath, query: TextQuery, *,
                         labels: Tensor | None = None,
                         extra_passes: Iterable[str] = ()) -> dict[str, Any]:
    """No-update response CE/readout for an arm or a control."""
    with torch.no_grad():
        response = path.respond(query, labels=labels, extra_passes=extra_passes)
    return {
        "loss": None if response.loss is None else float(response.loss.detach()),
        "logits": response.logits.detach(),
        "null_logits": {key: value.detach()
                        for key, value in response.null_logits.items()},
        "charges": dict(response.charges),
        "query_id": query.query_id,
        "memory_source": query.memory_source,
        "selected_record_ids": list(query.selected_record_ids),
        "semantic_selection": query.semantic_selection,
    }


def _non_port_spine_snapshot(path: SparkTextMemoryPath) -> dict[str, Tensor]:
    port_names = set(port_output_parameter_names(path))
    return {name: parameter.detach().clone()
            for name, parameter in path.adapter.new_named_parameters()
            if name.startswith("extension.spine.") and name not in port_names}


def _changed_names(before: Mapping[str, Tensor], after: Mapping[str, Tensor]
                   ) -> list[str]:
    return [name for name in before
            if name in after and not torch.equal(before[name], after[name])]


def _parameter_deltas(before: Mapping[str, Tensor], after: Mapping[str, Tensor]
                      ) -> dict[str, dict[str, Any]]:
    deltas = {}
    for name in sorted(before):
        if name not in after:
            continue
        delta = after[name].detach() - before[name].detach()
        deltas[name] = {
            "shape": list(before[name].shape),
            "l2": float(delta.float().norm()),
            "max_abs": float(delta.float().abs().max()) if delta.numel() else 0.0,
            "changed": bool(torch.count_nonzero(delta)),
            "before_sha256": _tensor_digest(before[name]),
            "after_sha256": _tensor_digest(after[name]),
        }
    return deltas


def direct_response_ce_step(
        path: SparkTextMemoryPath, query: TextQuery, labels: Tensor, *,
        event_id: str, expert_id: int = 0, apply: bool = True,
        port_optimizer: NativeFusedAdamW | None = None,
        post_query_factory: Callable[[], TextQuery] | None = None,
        arm: str | None = None) -> dict[str, Any]:
    """Update only the selected P expert from direct response CE.

    ``apply=False`` is the matched non-learning control: it performs the same
    sensing, decoder forward and backward work but discards gradients without an
    optimizer step.  A repeated ``event_id`` with the same objective is handled
    by the existing controller as a replay/retry and must leave P unchanged.

    ``port_optimizer`` is optional and must be a NativeFusedAdamW owner containing
    only ``extension.spine.ports.*.output_projection`` tensors.  A v2 path may
    explicitly replace ``path.response_optimizer`` with this port-only owner and
    pass the same object.  When supplied and a committed P transaction occurs,
    it is stepped once from the already computed direct-CE gradients.  A
    replay/retry steps neither P nor the port owner.  Callers omitting
    ``port_optimizer`` keep prior behavior.
    """
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be nonempty")
    if query.memory_source not in {"P", "E_plus_P"}:
        raise ValueError("direct P response CE requires a P-bearing query")
    if not torch.is_tensor(labels) or labels.shape != query.input_ids.shape:
        raise ValueError("labels must match the query input shape")
    if port_optimizer is not None:
        _validate_port_optimizer(path, port_optimizer)
    bank = path.experiment.bank
    expert_id = int(expert_id)
    expert = bank.expert(expert_id)
    names = tuple(name for name, _ in expert.named_parameters())
    port_parameters = _port_output_parameters(path)
    port_names = tuple(name for name, _ in port_parameters)
    port_before = {name: parameter.detach().clone()
                   for name, parameter in port_parameters}
    port_state_before = (
        _optimizer_state_by_name(port_optimizer, port_parameters)
        if port_optimizer is not None else {})
    before_parameters = _clone_parameters(expert)
    before_moments = bank.optimizer_state_for_expert(expert_id)
    before_steps = tuple(bank.expert_steps)
    clock_before = path.experiment.controller.clock.eligible_events
    revision_before = path.experiment.controller.revision
    response_steps_before = path.response_steps
    response_optimizer_before = _optimizer_digest(path.response_optimizer)
    spine_before = _non_port_spine_snapshot(path)
    inactive_before = {
        other: _clone_parameters(bank.expert(other))
        for other in range(bank.expert_count) if other != expert_id}

    work = _response_ce_gradients(path, query, labels, expert_id=expert_id)
    response = work["response"]
    loss_before = work["loss"]
    gradients = work["gradients"]
    port_gradients: dict[str, Tensor | None] = {}
    port_gradient_norms: dict[str, float] = {}
    port_missing_gradient_names: list[str] = []
    port_zero_gradient_names: list[str] = []
    for name, parameter in port_parameters:
        gradient = parameter.grad
        if gradient is None:
            port_gradients[name] = None
            port_gradient_norms[name] = 0.0
            port_missing_gradient_names.append(name)
            continue
        cloned = gradient.detach().clone()
        port_gradients[name] = cloned
        port_gradient_norms[name] = float(cloned.float().norm())
        if not bool(torch.count_nonzero(cloned)):
            port_zero_gradient_names.append(name)
    features = path.plastic_features(query)
    target = labels.detach().to(torch.float32)
    outcome = None
    port_stepped = False
    if apply:
        def loss_gradients(_expert):
            return torch.tensor(loss_before, dtype=torch.float32), gradients
        outcome = path.experiment.controller.apply_teaching_transaction(
            event_id=event_id, expert_id=expert_id, features=features,
            target=target, steps=1, loss_gradients=loss_gradients)
        if not outcome.retried:
            path.experiment.invalidate(reason="direct_response_ce_p_update")
            path.queries.invalidate()
            if port_optimizer is not None:
                for name, parameter in port_parameters:
                    gradient = port_gradients[name]
                    parameter.grad = (None if gradient is None
                                      else gradient.detach().clone())
                port_optimizer.step()
                port_stepped = True
    # Response/spine gradients not owned by P or the port optimizer are
    # intentionally discarded and never stepped.
    path.adapter.zero_grad(set_to_none=True)
    bank.zero_gradients()

    after_parameters = _clone_parameters(expert)
    after_moments = bank.optimizer_state_for_expert(expert_id)
    changed = _changed_names(before_parameters, after_parameters)
    deltas = _parameter_deltas(before_parameters, after_parameters)
    inactive_changed = {
        other: _changed_names(values, _clone_parameters(bank.expert(other)))
        for other, values in inactive_before.items()}
    spine_changed = _changed_names(spine_before, _non_port_spine_snapshot(path))
    port_after = {name: parameter.detach().clone()
                  for name, parameter in port_parameters}
    port_changed = _changed_names(port_before, port_after)
    port_deltas = _parameter_deltas(port_before, port_after)
    port_state_after = (
        _optimizer_state_by_name(port_optimizer, port_parameters)
        if port_optimizer is not None else {})
    response_optimizer_after = _optimizer_digest(path.response_optimizer)
    retried = bool(outcome is not None and outcome.retried)
    physical_steps = int(outcome.steps) if outcome is not None and not retried else 0
    port_step_counts_before = _step_counts(port_state_before)
    port_step_counts_after = _step_counts(port_state_after)
    port_tensor_steps_advanced = [
        name for name in port_names
        if port_step_counts_after.get(name, 0)
        > port_step_counts_before.get(name, 0)]
    port_state_changed = _moments_changed(port_state_before, port_state_after)
    port_skipped_on_retry = bool(port_optimizer is not None and retried)
    port_physical_steps = 1 if port_stepped else 0
    response_optimizer_is_port_owner = bool(
        port_optimizer is not None and port_optimizer is path.response_optimizer)
    response_optimizer_stepped = bool(
        port_stepped and response_optimizer_is_port_owner)
    clock_after = path.experiment.controller.clock.eligible_events
    revision_after = path.experiment.controller.revision
    sites = tuple(int(site) for site in path.experiment.core.spec.sites)
    sites_meta = site_accounting(sites, path.adapter.config.geometry.num_hidden_layers)

    post_loss = None
    post_charges = None
    if post_query_factory is not None and apply and not retried:
        post_query = post_query_factory()
        post = evaluate_response_ce(path, post_query, labels=labels)
        post_loss = post["loss"]
        post_charges = post["charges"]

    charges = {
        "sensing_forward_calls": int(query.native_query_passes),
        "query_memory_source": query.memory_source,
        "p_read_calls": int(query.p_read_calls_before_injection),
        "e_read_calls": int(query.memory_source in {"E", "E_plus_P"}),
        "p_feature_vectors": int(query.input_ids.numel()),
        "p_backward_feature_vectors": int(query.input_ids.numel()),
        "decoder_forwards_objective": int(response.charges["decoder_passes"]),
        "decoder_forwards_post_update": 1 if post_charges is not None else 0,
        "backward_calls": 1,
        "backward_positions": int((labels != -100).sum().item()),
        "backward_tokens": int(labels.numel()),
        "optimizer_steps": physical_steps,
        "response_optimizer_steps": 0,
        "port_optimizer_parameters": len(port_names),
        "port_optimizer_steps": port_physical_steps,
        "port_gradient_tensors": len(port_names) - len(port_missing_gradient_names),
        "port_parameter_updates": len(port_changed),
        "port_optimizer_state_tensors": sum(
            1 for row in port_state_after.values() if row is not None),
        "port_optimizer_retry_skipped": port_skipped_on_retry,
        "codec_calls": int(response.charges["codec_calls"]),
        "port_calls": int(response.charges["port_calls"]),
        "memory_read_slots": int(response.charges["memory_read_slots"]),
        "e_memory_read_slots": int(response.charges["e_memory_read_slots"]),
        "p_memory_read_slots": int(response.charges["p_memory_read_slots"]),
        "sensing_forward_layers": sites_meta["sensing_forward_layers"],
        "port_injections_per_forward": sites_meta["port_injections_per_forward"],
        "p_backward_suffix_layers_upper_bound":
            sites_meta["p_backward_suffix_layers_upper_bound"],
        "response_charges": dict(response.charges),
        "post_update_charges": post_charges,
    }
    evidence = {
        "schema": P1_SCHEMA,
        "engineering_evidence": True,
        "engineering_label": ENGINEERING_LABEL,
        "native_capability_claim": False,
        "sealed_evaluation": SEALED_EVALUATION,
        "arm": arm or query.memory_source,
        "event_id": event_id,
        "objective": "direct_response_cross_entropy",
        "selected_expert_id": expert_id,
        "selected_parameter_names": list(names),
        "changed_parameter_names": changed,
        "actual_parameter_update": bool(changed),
        "parameter_deltas": deltas,
        "gradients": {
            "names": sorted(work["gradient_norms"]),
            "norms": work["gradient_norms"],
            "sum_norm": work["gradient_norm_sum"],
            "adapter_audit": work["adapter_gradient_audit"],
            "inactive_nonzero_gradient_names":
                work["inactive_nonzero_gradient_names"],
            "spine_gradients_discarded": True,
        },
        "moments": {
            "before": _moment_summary(before_moments),
            "after": _moment_summary(after_moments),
            "changed_keys": _moments_changed(before_moments, after_moments),
            "owner": "expert_bank.optimizer",
        },
        "clock": {
            "before": clock_before,
            "after": clock_after,
            "eligible_events_advanced": clock_after == clock_before + 1,
            "revision_before": revision_before,
            "revision_after": revision_after,
            "revision_advanced": revision_after == revision_before + 1,
            "expert_steps_before": list(before_steps),
            "expert_steps_after": list(bank.expert_steps),
        },
        "optimizer_ownership": {
            "update_owner": "expert_bank.optimizer",
            "controller": "PlasticLearningController",
            "response_optimizer_stepped": response_optimizer_stepped,
            "response_optimizer_state_changed":
                response_optimizer_before != response_optimizer_after,
            "response_optimizer_is_distinct":
                path.response_optimizer is not bank.optimizer,
            "response_optimizer_is_declared_port_owner":
                response_optimizer_is_port_owner,
            "response_optimizer_scope": (
                "port_output_only" if response_optimizer_is_port_owner
                else "spine_response_extension"),
            "response_steps_before": response_steps_before,
            "response_steps_after": path.response_steps,
            "single_p_owner": True,
            "physical_optimizer_steps": physical_steps,
            "port_optimizer_provided": port_optimizer is not None,
            "port_optimizer_stepped": port_stepped,
            "port_optimizer_reuses_response_optimizer":
                response_optimizer_is_port_owner,
            "port_optimizer_is_distinct_from_response_optimizer": (
                None if port_optimizer is None
                else not response_optimizer_is_port_owner),
            "port_optimizer_scope":
                "extension.spine.ports.*.output_projection",
            "spine_response_optimizer_stepped": False,
        },
        "nonlearning_control": {
            "applied": bool(apply),
            "matched_forward_backward_work": True,
            "optimizer_skipped": not apply,
            "parameter_update": bool(changed) and apply,
        },
        "replay": {
            "retried": retried,
            "status": None if outcome is None else outcome.status,
            "parameter_update": bool(changed) and not retried,
        },
        "inactive_expert_changes": inactive_changed,
        "response_spine_parameter_changes": spine_changed,
        "non_port_spine_parameter_changes": spine_changed,
        "port_parameter_deltas": port_deltas,
        "port_changed_parameter_names": port_changed,
        "all_spine_parameter_changes": sorted(
            set(spine_changed) | set(port_changed)),
        "port_optimizer": {
            "provided": port_optimizer is not None,
            "owner": "extension.spine.ports.*.output_projection",
            "scope": "p1_port_only",
            "parameter_names": list(port_names),
            "stepped": port_stepped,
            "skipped_on_retry": port_skipped_on_retry,
            "physical_steps": port_physical_steps,
            "step_counts_before": port_step_counts_before,
            "step_counts_after": port_step_counts_after,
            "tensor_steps_advanced": port_tensor_steps_advanced,
            "state_before": _moment_summary(port_state_before),
            "state_after": _moment_summary(port_state_after),
            "changed_state_keys": port_state_changed,
            "changed_parameter_names": port_changed,
            "parameter_deltas": port_deltas,
            "gradient_norms": port_gradient_norms,
            "missing_gradient_names": port_missing_gradient_names,
            "zero_gradient_names": port_zero_gradient_names,
            "distinct_from_response_optimizer": (
                None if port_optimizer is None
                else not response_optimizer_is_port_owner),
            "reuses_response_optimizer": response_optimizer_is_port_owner,
        },
        "loss": {
            "before": loss_before,
            "after_same_batch": post_loss,
            "same_batch_improved": (
                None if post_loss is None else bool(post_loss < loss_before)),
            "train_fit": "same_batch_ce_measured_only",
        },
        "transfer": "unevaluated",
        "retention": "unevaluated",
        "useful_learning": "unevaluated",
        "learning_success": None,
        "epoch_equivalence": "one_direct_ce_step_not_a_training_schedule",
        "compute_charges": charges,
        "detach_boundaries": list(DETACH_BOUNDARIES),
        "site_accounting": sites_meta,
        "fixture_only_late_sites": max(sites) >= 23,
        "native_execution": "not_run",
    }
    return evidence


def prepare_arm_query(
        path: SparkTextMemoryPath, *, arm: str, question: str,
        input_ids: Tensor, attention_mask: Tensor,
        target_record_id: str | None = None,
        wrong_record_id: str | None = None, expert_id: int = 0,
        query_time: int = 0,
        token_available_at: Sequence[int] | None = None) -> TextQuery:
    """Prepare one readout arm without touching any optimizer."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    records = tuple(path.experiment.evidence.active)
    record_ids = tuple(record.record_id for record in records)
    if arm in {"full_eligible", "e_only", "e_plus_p"}:
        memory_source = "E" if arm != "e_plus_p" else "E_plus_P"
        host: Iterable[str] | None = record_ids
        top_k = None if record_ids else 1
    elif arm == "selected_exact":
        if target_record_id is None or target_record_id not in record_ids:
            raise ValueError("selected_exact needs an active target record")
        memory_source = "E"
        host = (target_record_id,)
        top_k = 1
    elif arm == "wrong_memory":
        if wrong_record_id is None or wrong_record_id not in record_ids:
            raise ValueError("wrong_memory needs an active distractor record")
        memory_source = "E"
        host = (wrong_record_id,)
        top_k = 1
    elif arm == "absent_memory":
        memory_source = "E"
        host = ()
        top_k = 1
    else:  # p_only
        memory_source = "P"
        host = ()
        top_k = None
    return path.prepare_query(
        question, input_ids, attention_mask,
        token_available_at=token_available_at, query_time=query_time,
        memory_source=memory_source,
        plastic_expert_id=(int(expert_id)
                           if memory_source in {"P", "E_plus_P"} else None),
        host_eligible_record_ids=host,
        semantic_top_k=top_k)


def audit_p_only_claim(
        path: SparkTextMemoryPath, query: TextQuery, *,
        answer_text: str | None = None,
        answer_token_ids: Iterable[int] = (),
        labels: Tensor | None = None) -> dict[str, Any]:
    """Audit source/workspace/identity leakage before calling P source-free."""
    if query.memory_source != "P":
        raise ValueError("the audit applies only to a P-only query")
    lineage = query.context.packet.source_lineage
    evidence_ids = list(lineage.get("record_ids", []))
    evidence_ids.extend(lineage.get("evidence_record_ids", []))
    tokens = {int(token) for token in answer_token_ids}
    prompt_tokens = {int(token) for token in query.input_ids.flatten().tolist()}
    prompt_text = query.question.casefold()
    prompt_text_leak = bool(answer_text and answer_text.casefold() in prompt_text)
    prompt_token_leak = bool(tokens & prompt_tokens)

    workspace_text = []
    for workspace in path.tasks.values():
        workspace_text.extend(item.description for item in workspace.subgoals)
        workspace_text.extend(item.observation
                              for item in workspace.observed_completions)
    workspace_tokens = set()
    for text in workspace_text:
        if answer_text and answer_text.casefold() in text.casefold():
            workspace_tokens.add("text")
    workspace_leak = bool(workspace_tokens)
    source_counts = {
        "exact_text_records": len(path.texts),
        "procedures": len(path.procedures),
        "task_workspaces": len(path.tasks),
        "candidate_explanations": len(path.explanations),
        "evidence_records": int(path.experiment.evidence.pending),
    }
    source_state_present = any(source_counts[key] for key in (
        "exact_text_records", "procedures", "task_workspaces",
        "candidate_explanations", "evidence_records"))
    teacher_positions = (0 if labels is None else
                         int((labels != -100).sum().item()))
    evidence_leak = bool(evidence_ids) or query.context.packet.slot_count > 0
    identity_leak = prompt_text_leak or prompt_token_leak
    claim_clean = not (source_state_present or workspace_leak
                       or evidence_leak or identity_leak)
    return {
        "schema": f"{P1_SCHEMA}.p_only_audit",
        "question_sha256": query.question_sha256,
        "memory_source": query.memory_source,
        "selected_record_ids": list(query.selected_record_ids),
        "semantic_selection": query.semantic_selection,
        "packet_slot_count": int(query.context.packet.slot_count),
        "lineage_evidence_ids": evidence_ids,
        "source_counts": source_counts,
        "source_state_present": source_state_present,
        "workspace_texts": workspace_text,
        "workspace_answer_leak": workspace_leak,
        "prompt_contains_answer_text": prompt_text_leak,
        "prompt_contains_answer_token_ids": prompt_token_leak,
        "answer_token_ids": sorted(tokens),
        "teacher_forced_answer_positions": teacher_positions,
        "teacher_forced_labels_declared": labels is not None,
        "evidence_leak": evidence_leak,
        "identity_leak": identity_leak,
        "p_only_claim": "source_free_engineering" if claim_clean
                        else "not_source_free",
        "sealed_evaluation": SEALED_EVALUATION,
        "engineering_evidence": True,
        "native_capability_claim": False,
    }


__all__ = [
    "ARMS", "DETACH_BOUNDARIES", "ENGINEERING_LABEL", "LATE_SITE_MIN_LAYERS",
    "NATIVE_SEALED_LAYERS", "P1_SCHEMA", "PStateSnapshot", "SEALED_EVALUATION",
    "audit_p_only_claim", "build_port_optimizer", "capture_p_state",
    "direct_response_ce_step", "evaluate_response_ce",
    "port_output_parameter_names", "prepare_arm_query", "restore_p_state",
    "selected_parameter_names", "site_accounting",
]
