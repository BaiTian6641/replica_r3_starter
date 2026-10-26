"""Deterministic NL0 software/math qualification tests.

These exercise only the contracts declared in
``configs/nested_learning_nl0_v1.json``.  They do not train a model, run a
native job, qualify a capability, or establish fast/slow timing or adaptive
router utility.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch
from torch import Tensor

from replica_lab.nested_learning import (
    DECLARED_EXPERT_COUNT,
    BackdatedStateError,
    DedupLedger,
    DuplicateObservation,
    ExpertBank,
    FutureReadError,
    NestedLearningError,
    Observation,
    ObservationCoordinator,
    Observer,
    PersonalityIdentity,
    PlasticLearningController,
    RevisionCache,
    RouteControl,
    ScopeViolation,
    StaleCache,
    TransactionHooks,
    TransactionOutcome,
    associative_loss,
    associative_outer_gradient,
    associative_transition,
    bounded_values,
    clip_gradients_to_norm,
    exact_adamw_step,
    exact_adamw_unroll,
    expert_loss_gradients,
    first_order_identity_adaptation,
    load_nl0_config,
    normalized_key,
    validate_nl0_config,
)
from replica_lab.optimizers import NativeFusedAdamW

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "nested_learning_nl0_v1.json"


def tiny_bank(*, expert_count: int = DECLARED_EXPERT_COUNT,
              width: int = 3, hidden_width: int = 4,
              seed: int = 7, lr: float = 0.02) -> ExpertBank:
    return ExpertBank(expert_count=expert_count, width=width,
                      hidden_width=hidden_width, seed=seed, lr=lr,
                      betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def tiny_inputs(width: int = 3, *, seed: int = 11) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.rand(1, width, generator=generator, dtype=torch.float32)
    target = torch.rand(1, width, generator=generator, dtype=torch.float32)
    return features, target


def observation(event_id: str, *, scope: str = "world-a", width: int = 3,
                available_at: float = 1.0, eligible: bool = True,
                source: str = "observation", seed: int = 0) -> Observation:
    generator = torch.Generator(device="cpu").manual_seed(seed + len(event_id))
    key = torch.rand(width, generator=generator, dtype=torch.float64)
    value = torch.rand(width, generator=generator, dtype=torch.float64)
    return Observation(event_id=event_id, scope=scope, key=tuple(key.tolist()),
                       value=tuple(value.tolist()), available_at=available_at,
                       eligible=eligible, source=source)


def finite_difference_gradient(function, tensor: Tensor, *,
                               eps: float = 1e-6) -> Tensor:
    """Central differences of a scalar function over every tensor element."""
    base = tensor.detach().clone()
    flat = torch.zeros(base.numel(), dtype=base.dtype)
    for index in range(base.numel()):
        plus = base.clone().reshape(-1)
        minus = base.clone().reshape(-1)
        plus[index] += eps
        minus[index] -= eps
        with torch.no_grad():
            high = float(function(plus.view_as(base)))
            low = float(function(minus.view_as(base)))
        flat[index] = (high - low) / (2.0 * eps)
    return flat.view_as(base)


def assert_optimizer_state_equal(left, right) -> None:
    assert set(left) == set(right)
    for key in left:
        if left[key] is None:
            assert right[key] is None
            continue
        assert set(left[key]) == set(right[key])
        for name, value in left[key].items():
            if isinstance(value, Tensor):
                assert torch.equal(value, right[key][name])
            else:
                assert value == right[key][name]


def _optimizer_group_signature(controller) -> tuple:
    return tuple(
        (group["group_name"], group["scope"], group["lr"],
         tuple(group["betas"]), group["eps"], group["weight_decay"],
         tuple(group["param_names"]))
        for group in controller.bank.optimizer.param_groups)


def _live_controller_snapshot(controller) -> dict:
    bank = controller.bank
    coordinator = controller.coordinator
    cutoffs = ((coordinator.checkpoint_payload()["latest_sealed_at"])
               if coordinator is not None else None)
    return {
        "ids": [(id(parameter), parameter.data_ptr())
                for parameter in bank.parameters()],
        "parameters": [parameter.detach().clone()
                       for parameter in bank.parameters()],
        "gradients": [None if parameter.grad is None
                      else parameter.grad.detach().clone()
                      for parameter in bank.parameters()],
        "state": bank.optimizer_state_snapshot(range(bank.expert_count)),
        "groups": _optimizer_group_signature(controller),
        "steps": list(bank.expert_steps),
        "clock": controller.clock,
        "dedup": controller.dedup.snapshot(),
        "revision": controller.revision,
        "cache": controller.cache.get("live"),
        "coordinator_fast": (coordinator.fast_state
                             if coordinator is not None else None),
        "coordinator_read": (coordinator.read(scope=coordinator.scope).observations
                             if coordinator is not None else None),
        "coordinator_revision": (coordinator.revision
                                if coordinator is not None else None),
        "coordinator_transitions": (coordinator.transition_count
                                    if coordinator is not None else None),
        "coordinator_latest": cutoffs,
        "coordinator_views": (
            {key: value.view_id
             for key, value in coordinator.last_reader_views.items()}
            if coordinator is not None else {}),
    }


def _assert_live_controller_equal(controller, snapshot: dict) -> None:
    bank = controller.bank
    coordinator = controller.coordinator
    assert [(id(parameter), parameter.data_ptr())
            for parameter in bank.parameters()] == snapshot["ids"]
    for parameter, expected in zip(bank.parameters(), snapshot["parameters"]):
        assert torch.equal(parameter, expected)
    for parameter, expected in zip(bank.parameters(), snapshot["gradients"]):
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected)
    assert_optimizer_state_equal(
        snapshot["state"], bank.optimizer_state_snapshot(range(bank.expert_count)))
    assert _optimizer_group_signature(controller) == snapshot["groups"]
    assert bank.expert_steps == snapshot["steps"]
    assert controller.clock == snapshot["clock"]
    assert controller.dedup.snapshot() == snapshot["dedup"]
    assert controller.revision == snapshot["revision"]
    assert controller.cache.get("live") == snapshot["cache"]
    if coordinator is not None:
        assert torch.equal(coordinator.fast_state, snapshot["coordinator_fast"])
        read = coordinator.read(scope=coordinator.scope)
        assert read.observations == snapshot["coordinator_read"]
        assert coordinator.revision == snapshot["coordinator_revision"]
        assert coordinator.transition_count == snapshot["coordinator_transitions"]
        assert (coordinator.checkpoint_payload()["latest_sealed_at"]
                == snapshot["coordinator_latest"])
        assert {key: value.view_id
                for key, value in coordinator.last_reader_views.items()} == \
            snapshot["coordinator_views"]


def _coordinator_live_snapshot(coordinator) -> dict:
    return {
        "fast": coordinator.fast_state,
        "observations": coordinator.read(scope=coordinator.scope).observations,
        "revision": coordinator.revision,
        "transitions": coordinator.transition_count,
        "latest": coordinator.checkpoint_payload()["latest_sealed_at"],
        "views": {key: value.view_id
                  for key, value in coordinator.last_reader_views.items()},
    }


def _assert_coordinator_live(coordinator, snapshot: dict) -> None:
    assert torch.equal(coordinator.fast_state, snapshot["fast"])
    assert coordinator.read(scope=coordinator.scope).observations == \
        snapshot["observations"]
    assert coordinator.revision == snapshot["revision"]
    assert coordinator.transition_count == snapshot["transitions"]
    assert (coordinator.checkpoint_payload()["latest_sealed_at"]
            == snapshot["latest"])
    assert {key: value.view_id
            for key, value in coordinator.last_reader_views.items()} == \
        snapshot["views"]


# ---------------------------------------------------------------------------
# Declared linear recurrence
# ---------------------------------------------------------------------------


def test_declared_recurrence_matches_formula_and_rejects_bad_inputs():
    generator = torch.Generator(device="cpu").manual_seed(2)
    state = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    key = torch.randn(3, generator=generator, dtype=torch.float64)
    value = torch.randn(3, generator=generator, dtype=torch.float64) * 4.0
    unit_key = key / key.norm()
    bounded = value * (1.0 / value.norm())
    expected = 0.95 * state + 0.1 * torch.outer(bounded - state @ unit_key, unit_key)
    actual = associative_transition(state, key, value, rho=0.95, eta=0.1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    loss = associative_loss(state, key, value)
    manual = 0.5 * (state @ unit_key - bounded).pow(2).sum()
    torch.testing.assert_close(loss, manual, rtol=0, atol=1e-12)
    assert bounded_values(value).norm() <= 1.0 + 1e-12
    assert torch.allclose(normalized_key(key).norm(), torch.tensor(1.0, dtype=torch.float64))

    with pytest.raises(NestedLearningError):
        associative_transition(state, torch.zeros(3), value, rho=0.95, eta=0.1)
    with pytest.raises(NestedLearningError):
        associative_transition(state, key, value, rho=0.5, eta=0.1)
    with pytest.raises(NestedLearningError):
        associative_transition(state, key, value, rho=0.95, eta=1.5)
    nonfinite = state.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(NestedLearningError):
        associative_transition(nonfinite, key, value, rho=0.95, eta=0.1)
    with pytest.raises(NestedLearningError):
        associative_loss(state, key, torch.tensor([math.inf, 0.0, 0.0],
                                                  dtype=torch.float64))


def test_direct_outer_gradient_matches_finite_differences():
    generator = torch.Generator(device="cpu").manual_seed(3)
    state = torch.randn(3, 3, generator=generator, dtype=torch.float64,
                        requires_grad=True)
    key = torch.randn(3, generator=generator, dtype=torch.float64)
    value = torch.randn(3, generator=generator, dtype=torch.float64) * 2.0
    loss = associative_loss(state, key, value)
    autograd_gradient = torch.autograd.grad(loss, state)[0]
    analytic = associative_outer_gradient(state, key, value)
    torch.testing.assert_close(autograd_gradient, analytic, rtol=1e-9, atol=1e-12)
    numeric = finite_difference_gradient(
        lambda candidate: associative_loss(candidate, key, value), state)
    torch.testing.assert_close(autograd_gradient, numeric, rtol=1e-6, atol=1e-7)


def test_later_read_to_earlier_write_gradient_matches_finite_differences():
    generator = torch.Generator(device="cpu").manual_seed(4)
    state = torch.randn(3, 3, generator=generator, dtype=torch.float64,
                        requires_grad=True)
    key1 = torch.randn(3, generator=generator, dtype=torch.float64)
    value1 = torch.randn(3, generator=generator, dtype=torch.float64)
    key2 = torch.randn(3, generator=generator, dtype=torch.float64)
    value2 = torch.randn(3, generator=generator, dtype=torch.float64)

    def rollout(candidate_state, write_value):
        first = associative_transition(candidate_state, key1, write_value,
                                       rho=0.93, eta=0.17)
        second = associative_transition(first, key2, value2, rho=0.93, eta=0.17)
        return associative_loss(second, key2, value2)

    loss = rollout(state, value1)
    state_gradient = torch.autograd.grad(loss, state)[0]
    numeric_state = finite_difference_gradient(
        lambda candidate: rollout(candidate, value1), state)
    torch.testing.assert_close(state_gradient, numeric_state, rtol=1e-6, atol=1e-7)
    assert state_gradient.norm() > 1e-4

    write_value = value1.clone().requires_grad_(True)
    loss = rollout(state.detach(), write_value)
    write_gradient = torch.autograd.grad(loss, write_value)[0]
    numeric_write = finite_difference_gradient(
        lambda candidate: rollout(state.detach(), candidate), write_value)
    torch.testing.assert_close(write_gradient, numeric_write, rtol=1e-6, atol=1e-7)
    assert write_gradient.shape == value1.shape
    assert bool(torch.isfinite(write_gradient).all())
    assert write_gradient.norm() > 1e-4


# ---------------------------------------------------------------------------
# Exact differentiable AdamW and first-order distinction
# ---------------------------------------------------------------------------


def test_exact_adamw_recurrence_matches_closed_form_and_finite_differences():
    parameter = torch.tensor([0.3, -0.2], dtype=torch.float64)
    gradient = torch.tensor([0.5, -0.25], dtype=torch.float64)
    zeros = torch.zeros_like(parameter)
    updated, first, second, state = exact_adamw_step(
        parameter, gradient, zeros, zeros, 1, lr=0.01,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    expected_moments = 0.1 * gradient
    expected_squares = 0.001 * gradient.pow(2)
    expected = parameter - 0.01 * gradient / (gradient.abs() + 1e-8)
    torch.testing.assert_close(first, expected_moments, rtol=0, atol=1e-14)
    torch.testing.assert_close(second, expected_squares, rtol=1e-12, atol=1e-18)
    torch.testing.assert_close(updated, expected, rtol=1e-9, atol=1e-12)
    assert state.step == 1

    generator = torch.Generator(device="cpu").manual_seed(5)
    initial = torch.randn(3, generator=generator, dtype=torch.float64)
    gradients = [torch.randn(3, generator=generator, dtype=torch.float64)
                 for _ in range(4)]
    losses = [
        torch.tensor([0.4, 0.3, 0.2], dtype=torch.float64),
        torch.tensor([-0.1, 0.2, 0.5], dtype=torch.float64),
        torch.tensor([0.25, -0.35, 0.15], dtype=torch.float64),
        torch.tensor([-0.3, 0.1, 0.2], dtype=torch.float64),
    ]
    for horizon in (1, 2, 4):
        source = initial.clone().requires_grad_(True)
        final, states = exact_adamw_unroll(source, gradients[:horizon], lr=0.01)
        assert len(states) == horizon and states[-1].step == horizon
        loss = (final * losses[horizon - 1]).sum()
        autograd_gradient = torch.autograd.grad(loss, source)[0]

        def unrolled(candidate):
            final, _ = exact_adamw_unroll(candidate, gradients[:horizon], lr=0.01)
            return (final * losses[horizon - 1]).sum()

        numeric = finite_difference_gradient(unrolled, source.detach())
        torch.testing.assert_close(autograd_gradient, numeric, rtol=1e-6, atol=1e-7)

    with pytest.raises(NestedLearningError):
        exact_adamw_step(parameter, torch.tensor([math.nan, 0.0], dtype=torch.float64),
                         zeros, zeros, 1, lr=0.01)


def test_first_order_identity_estimator_differs_from_exact_short_adaptation():
    parameter = torch.tensor([0.25, -0.4, 0.1], dtype=torch.float64)
    anchor = torch.tensor([0.2, -0.35, 0.05], dtype=torch.float64)
    query_target = torch.tensor([-0.3, 0.5, 0.2], dtype=torch.float64)
    source = parameter.clone().requires_grad_(True)
    # The teaching gradient depends on the initial parameters, so identity
    # adaptation is not accidentally exact.  A larger eps keeps the
    # first-step AdamW Jacobian observably non-identity on this tiny fixture.
    adapted, _ = exact_adamw_unroll(source, [source - anchor], lr=0.1, eps=1e-3)
    adapted_gradient = adapted - query_target
    exact = torch.autograd.grad(
        0.5 * (adapted - query_target).pow(2).sum(), source)[0]
    first_order = first_order_identity_adaptation(adapted_gradient)

    def query_loss(candidate):
        final, _ = exact_adamw_unroll(candidate, [candidate - anchor],
                                      lr=0.1, eps=1e-3)
        return 0.5 * (final - query_target).pow(2).sum()

    numeric = finite_difference_gradient(query_loss, parameter)
    torch.testing.assert_close(exact, numeric, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(first_order, adapted_gradient, rtol=0, atol=0)
    discrepancy = (exact - first_order).norm()
    assert discrepancy > 1e-3
    # No false gate: the two estimators are not required to agree.
    assert not torch.allclose(exact, first_order, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Selected-only bank and NativeFusedAdamW transaction
# ---------------------------------------------------------------------------


def test_selected_native_fused_update_matches_exact_reference_and_inactive_parity():
    bank = tiny_bank(seed=19, lr=0.02)
    controller = PlasticLearningController(bank, scope="world-a")
    features, target = tiny_inputs(3, seed=21)
    expert_id = 7
    expert = bank.expert(expert_id)
    initial = {name: parameter.detach().clone()
               for name, parameter in expert.named_parameters()}
    loss, gradients = expert_loss_gradients(expert, features, target)
    with torch.no_grad():
        residual = expert(features) - target
    expected_loss = 0.5 * residual.pow(2).sum()
    assert math.isclose(float(loss), float(expected_loss), rel_tol=1e-6)
    assert not math.isclose(float(loss), float(residual.pow(2).mean()))
    inactive_ids = tuple(index for index in range(bank.expert_count)
                         if index != expert_id)
    inactive_parameters = bank.parameter_snapshot(inactive_ids)
    inactive_state = bank.optimizer_state_snapshot(inactive_ids)

    outcome = controller.apply_teaching_transaction(
        event_id="teach-1", expert_id=expert_id,
        features=features, target=target, steps=1)
    assert outcome.status == "committed" and outcome.steps == 1
    assert outcome.eligible_events == 1 and outcome.revision == 1
    assert math.isclose(outcome.loss, float(loss))

    optimizer = bank.optimizer
    assert isinstance(optimizer, NativeFusedAdamW)
    assert all(parameter.dtype == torch.float32
               for parameter in expert.parameters())
    for name, parameter in expert.named_parameters():
        zero = torch.zeros_like(initial[name])
        expected, expected_avg, expected_sq, _ = exact_adamw_step(
            initial[name], gradients[name], zero, zero, 1, lr=0.02,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)
        state = optimizer.state[parameter]
        torch.testing.assert_close(state["exp_avg"], expected_avg, rtol=0, atol=1e-8)
        torch.testing.assert_close(state["exp_avg_sq"], expected_sq, rtol=0, atol=1e-12)
        assert int(float(state["step"])) == 1
    assert bank.expert_steps[expert_id] == 1
    assert [bank.expert_steps[index] for index in inactive_ids] == [0] * len(inactive_ids)
    for index in inactive_ids:
        for parameter in bank.expert(index).parameters():
            assert parameter.grad is None
            assert parameter not in optimizer.state
    bank.assert_inactive_unchanged(inactive_ids, inactive_parameters, inactive_state)
    assert len(optimizer.state) == 3

    selected_state = bank.optimizer_state_snapshot([expert_id])
    selected_parameters = bank.parameter_snapshot([expert_id])
    second_features, second_target = tiny_inputs(3, seed=22)
    controller.apply_teaching_transaction(
        event_id="teach-2", expert_id=2,
        features=second_features, target=second_target, steps=1)
    assert_optimizer_state_equal(
        selected_state, bank.optimizer_state_snapshot([expert_id]))
    for name, parameter in bank.expert(expert_id).named_parameters():
        assert torch.equal(parameter, selected_parameters[expert_id][name])
    assert len(bank.optimizer.state) == 6
    declared = bank.declared_shape
    assert declared["experts"] == 16
    assert declared["width"] == 256
    assert declared["hidden_width"] == 1024
    assert declared["bias"] is False
    assert declared["matrices_per_expert"] == 3


def _exact_two_step_reference(expert, features: Tensor, target: Tensor, *,
                              lr: float, betas: tuple[float, float],
                              eps: float, weight_decay: float,
                              max_norm: float
                              ) -> tuple[dict[str, Tensor], dict[str, Tensor],
                                         dict[str, Tensor]]:
    clone = copy.deepcopy(expert)
    parameters = dict(clone.named_parameters())
    first = {name: torch.zeros_like(parameter)
             for name, parameter in parameters.items()}
    second = {name: torch.zeros_like(parameter)
              for name, parameter in parameters.items()}
    for step in (1, 2):
        _, gradients = expert_loss_gradients(clone, features, target)
        gradients, _, _ = clip_gradients_to_norm(gradients, max_norm=max_norm)
        for name, parameter in parameters.items():
            updated, first[name], second[name], _ = exact_adamw_step(
                parameter, gradients[name], first[name], second[name], step,
                lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            parameter.data.copy_(updated)
    return parameters, first, second


def test_global_gradient_norm_cap_is_applied_before_adamw():
    bank = ExpertBank(expert_count=16, width=2, hidden_width=3, seed=42,
                      lr=0.02, betas=(0.9, 0.999), eps=1e-8,
                      weight_decay=0.0)
    controller = PlasticLearningController(bank, scope="world-a",
                                           gradient_norm_max=1.0)
    features = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    target = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
    expert_id = 3
    expert = bank.expert(expert_id)
    initial = {name: parameter.detach().clone()
               for name, parameter in expert.named_parameters()}
    _, raw_gradients = expert_loss_gradients(expert, features, target)
    clipped, raw_norm, scale = clip_gradients_to_norm(raw_gradients, max_norm=1.0)
    assert raw_norm > 1.0
    assert 0.0 < scale < 1.0

    controller.apply_teaching_transaction(
        event_id="clipped", expert_id=expert_id,
        features=features, target=target, steps=1)
    for name, parameter in expert.named_parameters():
        expected, expected_avg, expected_sq, _ = exact_adamw_step(
            initial[name], clipped[name], torch.zeros_like(initial[name]),
            torch.zeros_like(initial[name]), 1, lr=0.02,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)
        state = bank.optimizer.state[parameter]
        torch.testing.assert_close(state["exp_avg"], expected_avg,
                                   rtol=0, atol=1e-8)
        torch.testing.assert_close(state["exp_avg_sq"], expected_sq,
                                   rtol=0, atol=1e-12)
    # The unclipped first moment would differ measurably on this fixture,
    # even though AdamW's first parameter step is nearly scale-invariant.
    clipped_moment = bank.optimizer.state[expert.gate]["exp_avg"]
    unclipped_moment = 0.1 * raw_gradients["gate"]
    assert not torch.allclose(clipped_moment, unclipped_moment,
                              rtol=0, atol=1e-8)


def test_gradient_clipping_activates_on_second_selected_step():
    generator = torch.Generator(device="cpu").manual_seed(132)
    features = ((torch.rand(1, 2, generator=generator, dtype=torch.float32)
                 - 0.5) * 2.0)
    target = ((torch.rand(1, 2, generator=generator, dtype=torch.float32)
               - 0.5) * 20.0)

    probe_bank = ExpertBank(expert_count=16, width=2, hidden_width=3,
                            seed=131, lr=0.02)
    probe = PlasticLearningController(probe_bank, scope="world-a",
                                      gradient_norm_max=1e9)
    _, first_gradients = expert_loss_gradients(
        probe_bank.expert(0), features, target)
    first_norm = clip_gradients_to_norm(first_gradients, max_norm=1.0)[1]
    probe.apply_teaching_transaction(
        event_id="probe-first", expert_id=0,
        features=features, target=target, steps=1)
    _, second_gradients = expert_loss_gradients(
        probe_bank.expert(0), features, target)
    second_norm = clip_gradients_to_norm(second_gradients, max_norm=1.0)[1]
    assert first_norm < second_norm
    cap = (first_norm + second_norm) / 2.0
    assert first_norm < cap < second_norm

    bank = ExpertBank(expert_count=16, width=2, hidden_width=3,
                      seed=131, lr=0.02)
    controller = PlasticLearningController(bank, scope="world-a",
                                           gradient_norm_max=cap)
    expert = bank.expert(0)
    clipped_parameters, clipped_first, clipped_second = \
        _exact_two_step_reference(
            expert, features, target, lr=bank.lr, betas=bank.betas,
            eps=bank.eps, weight_decay=bank.weight_decay, max_norm=cap)
    unclipped_parameters, unclipped_first, unclipped_second = \
        _exact_two_step_reference(
            expert, features, target, lr=bank.lr, betas=bank.betas,
            eps=bank.eps, weight_decay=bank.weight_decay, max_norm=1e9)

    controller.apply_teaching_transaction(
        event_id="second-step-clipped", expert_id=0,
        features=features, target=target, steps=2)
    for name, parameter in expert.named_parameters():
        torch.testing.assert_close(parameter, clipped_parameters[name],
                                   rtol=1e-5, atol=1e-6)
        state = bank.optimizer.state[parameter]
        torch.testing.assert_close(state["exp_avg"], clipped_first[name],
                                   rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(state["exp_avg_sq"], clipped_second[name],
                                   rtol=1e-5, atol=1e-9)
        assert int(float(state["step"])) == 2
    # The clipped second step is observably distinct from the unclipped path.
    assert not torch.allclose(clipped_first["gate"], unclipped_first["gate"],
                              rtol=0, atol=1e-7)
    assert not torch.allclose(clipped_second["gate"], unclipped_second["gate"],
                              rtol=0, atol=1e-9)
    assert not torch.allclose(clipped_parameters["gate"],
                              unclipped_parameters["gate"],
                              rtol=0, atol=1e-7)


def test_native_fused_two_step_nonzero_decay_matches_exact_reference():
    bank = ExpertBank(expert_count=16, width=2, hidden_width=3, seed=43,
                      lr=0.03, betas=(0.9, 0.999), eps=1e-8,
                      weight_decay=0.05)
    controller = PlasticLearningController(bank, scope="world-a")
    generator = torch.Generator(device="cpu").manual_seed(44)
    features = (torch.rand(1, 2, generator=generator, dtype=torch.float32)
                - 0.5)
    target = (torch.rand(1, 2, generator=generator, dtype=torch.float32)
              - 0.5)
    expert_id = 6
    initial_expert = bank.expert(expert_id)
    expected_parameters, expected_first, expected_second = \
        _exact_two_step_reference(
            initial_expert, features, target, lr=bank.lr, betas=bank.betas,
            eps=bank.eps, weight_decay=bank.weight_decay, max_norm=1.0)

    outcome = controller.apply_teaching_transaction(
        event_id="two-step", expert_id=expert_id,
        features=features, target=target, steps=2)
    assert outcome.steps == 2
    assert bank.expert_steps[expert_id] == 2
    for name, parameter in initial_expert.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name],
                                   rtol=1e-5, atol=1e-6)
        state = bank.optimizer.state[parameter]
        torch.testing.assert_close(state["exp_avg"], expected_first[name],
                                   rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(state["exp_avg_sq"], expected_second[name],
                                   rtol=1e-5, atol=1e-9)
        assert int(float(state["step"])) == 2


def test_transaction_rollback_restores_all_state_and_keeps_parameter_ownership():
    bank = tiny_bank(width=2, hidden_width=3, seed=23)
    controller = PlasticLearningController(bank, scope="world-a")
    features, target = tiny_inputs(2, seed=24)
    with pytest.raises(ScopeViolation):
        controller.apply_teaching_transaction(
            event_id="wrong-world", expert_id=0,
            features=features, target=target, steps=1, scope="world-b")
    with pytest.raises(NestedLearningError):
        controller.apply_teaching_transaction(
            event_id="too-many", expert_id=0,
            features=features, target=target, steps=3)
    assert controller.revision == 0
    assert controller.dedup.event_ids == ()
    controller.apply_teaching_transaction(
        event_id="base", expert_id=4, features=features, target=target, steps=2)
    controller.cache.put("kv", "cached-value")
    assert controller.cache.get("kv") == "cached-value"

    parameter_identity = [(id(parameter), parameter.data_ptr())
                          for parameter in bank.parameters()]
    parameter_values = [parameter.detach().clone()
                        for parameter in bank.parameters()]
    all_state = bank.optimizer_state_snapshot(range(bank.expert_count))
    steps_before = list(bank.expert_steps)
    clock_before = controller.clock
    dedup_before = controller.dedup.snapshot()
    revision_before = controller.revision

    def fail_after_step(event_id: str, step_index: int) -> None:
        raise RuntimeError(f"injected failure at step {step_index} for {event_id}")

    with pytest.raises(RuntimeError, match="injected failure"):
        controller.apply_teaching_transaction(
            event_id="fault", expert_id=4,
            features=features, target=target, steps=2,
            hooks=TransactionHooks(after_optimizer_step=fail_after_step))

    assert [(id(parameter), parameter.data_ptr())
            for parameter in bank.parameters()] == parameter_identity
    for parameter, expected in zip(bank.parameters(), parameter_values):
        assert torch.equal(parameter, expected)
    assert_optimizer_state_equal(
        all_state, bank.optimizer_state_snapshot(range(bank.expert_count)))
    assert bank.expert_steps == steps_before
    assert controller.clock == clock_before
    assert controller.dedup.snapshot() == dedup_before
    assert controller.revision == revision_before
    assert controller.cache.get("kv") == "cached-value"
    assert "fault" not in controller.dedup.event_ids
    for parameter in bank.parameters():
        assert parameter.grad is None

    def fail_before_publish(_payload) -> None:
        raise RuntimeError("injected publication failure")

    expert9_before = {name: parameter.detach().clone()
                      for name, parameter in bank.expert(9).named_parameters()}
    with pytest.raises(RuntimeError, match="publication failure"):
        controller.apply_teaching_transaction(
            event_id="publish-fault", expert_id=9,
            features=features, target=target, steps=1,
            hooks=TransactionHooks(before_publish=fail_before_publish))
    assert controller.revision == revision_before
    assert controller.clock == clock_before
    assert controller.dedup.snapshot() == dedup_before
    assert bank.expert_steps == steps_before
    for name, parameter in bank.expert(9).named_parameters():
        assert torch.equal(parameter, expert9_before[name])
    assert controller.cache.get("kv") == "cached-value"

    committed = controller.apply_teaching_transaction(
        event_id="fault", expert_id=9,
        features=features, target=target, steps=2)
    assert committed.status == "committed"
    assert controller.revision == revision_before + 1
    assert controller.clock.eligible_events == clock_before.eligible_events + 1
    assert bank.expert_steps[9] == 2
    with pytest.raises(StaleCache):
        controller.cache.get("kv")


def test_duplicate_event_retry_returns_original_and_changed_identity_fails():
    bank = tiny_bank(width=2, hidden_width=3, seed=25)
    controller = PlasticLearningController(bank, scope="world-a")
    features, target = tiny_inputs(2, seed=26)
    first = controller.apply_teaching_transaction(
        event_id="event-1", expert_id=1,
        features=features, target=target, steps=2)
    weights = bank.parameter_snapshot([1])
    revision = controller.revision
    retry = controller.apply_teaching_transaction(
        event_id="event-1", expert_id=1,
        features=features, target=target, steps=2)
    assert retry.retried is True
    assert retry.fingerprint == first.fingerprint
    assert retry.revision == revision == first.revision
    assert retry.loss == first.loss
    for name, parameter in bank.expert(1).named_parameters():
        assert torch.equal(parameter, weights[1][name])

    controller.cache.put("kv", "still-valid")
    changed_features, changed_target = tiny_inputs(2, seed=27)
    with pytest.raises(DuplicateObservation):
        controller.apply_teaching_transaction(
            event_id="event-1", expert_id=1,
            features=changed_features, target=changed_target, steps=2)
    assert controller.revision == revision
    assert controller.cache.get("kv") == "still-valid"
    assert controller.dedup.event_ids == ("event-1",)
    assert bank.expert_steps[1] == 2

    second = controller.apply_teaching_transaction(
        event_id="event-2", expert_id=1,
        features=changed_features, target=changed_target, steps=1)
    assert second.revision == revision + 1
    assert bank.expert_steps[1] == 3
    with pytest.raises(StaleCache):
        controller.cache.get("kv")


# ---------------------------------------------------------------------------
# Observation coordination
# ---------------------------------------------------------------------------


def test_observation_chunk_has_one_transition_independent_of_reader_count():
    first = ObservationCoordinator(scope="world-a", width=3, seed=5,
                                   rho=0.91, eta=0.3)
    second = ObservationCoordinator(scope="world-a", width=3, seed=5,
                                    rho=0.91, eta=0.3)
    observations = [observation("o1", available_at=1.0, seed=1),
                    observation("o2", available_at=2.0, seed=2),
                    observation("o3", available_at=3.0, seed=3)]
    initial = first.fast_state.clone()

    one_reader = first.seal_observation_chunk(
        chunk_id="chunk-1", observations=observations,
        reader_ids=("reader-a",), sealed_at=3.0)
    many_reader = second.seal_observation_chunk(
        chunk_id="chunk-1", observations=observations,
        reader_ids=("reader-a", "reader-b", "reader-c", "reader-d"),
        sealed_at=3.0)

    assert one_reader.transition_count == 1
    assert many_reader.transition_count == 1
    assert one_reader.post_revision == many_reader.post_revision == 1
    assert one_reader.pre_revision == many_reader.pre_revision == 0
    assert first.transition_count == second.transition_count == 1
    torch.testing.assert_close(first.fast_state, second.fast_state,
                               rtol=0, atol=0)
    assert not torch.equal(first.fast_state, initial)

    views = second.last_reader_views
    assert set(views) == {"reader-a", "reader-b", "reader-c", "reader-d"}
    snapshots = list(views.values())
    assert all(view is snapshots[0] for view in snapshots)
    assert snapshots[0].revision == 0
    assert snapshots[0].readonly is True
    torch.testing.assert_close(snapshots[0].fast_state, initial,
                               rtol=0, atol=0)
    assert snapshots[0].observations == ()

    second.seal_observation_chunk(
        chunk_id="chunk-2",
        observations=[observation("o4", available_at=4.0, seed=4)],
        reader_ids=("reader-a",), sealed_at=4.0)
    assert second.transition_count == 2
    assert second.revision == 2

    second.cache.put("kv", "before-next-chunk")
    assert second.cache.get("kv") == "before-next-chunk"
    second.seal_observation_chunk(
        chunk_id="chunk-2b",
        observations=[observation("o4b", available_at=4.5, seed=44)],
        reader_ids=("reader-a",), sealed_at=4.5)
    with pytest.raises(StaleCache):
        second.cache.get("kv")
    assert second.transition_count == 3
    assert second.revision == 3
    pre_chunk = second.last_reader_views["reader-a"]
    assert pre_chunk.revision == 2
    assert {item.event_id for item in pre_chunk.observations} == {
        "o1", "o2", "o3", "o4"}

    before = second.fast_state.clone()
    second.cache.put("kv", "before-ineligible-chunk")
    assert second.cache.get("kv") == "before-ineligible-chunk"
    no_eligible = second.seal_observation_chunk(
        chunk_id="chunk-3",
        observations=[observation("o5", available_at=5.0, eligible=False, seed=5)],
        reader_ids=("reader-a",), sealed_at=5.0)
    assert no_eligible.transition_count == 0
    assert no_eligible.pre_revision == 3
    assert no_eligible.post_revision == 4
    assert second.transition_count == 3
    assert second.revision == 4
    assert torch.equal(second.fast_state, before)
    with pytest.raises(StaleCache):
        second.cache.get("kv")
    assert second.read(scope="world-a", event_id="o5") == observation(
        "o5", available_at=5.0, eligible=False, seed=5)

    second.cache.put("empty", "kept")
    empty = second.seal_observation_chunk(
        chunk_id="chunk-3b", observations=[],
        reader_ids=("reader-a",), sealed_at=5.5)
    assert empty.transition_count == 0
    assert empty.pre_revision == empty.post_revision == 4
    assert second.transition_count == 3
    assert second.revision == 4
    assert second.cache.get("empty") == "kept"
    assert second.last_reader_views["reader-a"].revision == 4

    with pytest.raises(FutureReadError):
        second.seal_observation_chunk(
            chunk_id="chunk-4",
            observations=[observation("o6", available_at=9.0, seed=6)],
            reader_ids=("reader-a",), sealed_at=8.0)
    assert second.revision == 4

    with pytest.raises(NestedLearningError):
        second.seal_observation_chunk(
            chunk_id="chunk-5",
            observations=[observation(f"x{index}", width=3, seed=index)
                          for index in range(33)],
            reader_ids=("reader-a",), sealed_at=10.0)


def test_causal_reads_fail_closed_on_wrong_scope_and_future_cutoff():
    coordinator = ObservationCoordinator(scope="world-a", width=3, seed=7)
    early = observation("early", available_at=2.0, seed=1)
    late = observation("late", available_at=6.0, seed=2)
    coordinator.seal_observation_chunk(
        chunk_id="c1", observations=[early], reader_ids=("r",), sealed_at=2.0)
    coordinator.seal_observation_chunk(
        chunk_id="c2", observations=[late], reader_ids=("r",), sealed_at=6.0)

    with pytest.raises(ScopeViolation):
        coordinator.read(scope="world-b")
    with pytest.raises(ScopeViolation):
        coordinator.read(scope="world-b", event_id="early")
    with pytest.raises(ScopeViolation):
        coordinator.read(scope="world-a", event_id="missing")
    with pytest.raises(FutureReadError):
        coordinator.read(scope="world-a", event_id="late", cutoff=5.0)
    assert coordinator.read(scope="world-a", event_id="early", cutoff=2.0) == early
    snapshot = coordinator.read(scope="world-a", cutoff=2.0)
    assert snapshot.scope == "world-a"
    assert snapshot.revision == 2
    assert snapshot.observations == (early,)
    full = coordinator.read(scope="world-a")
    assert full.observations == (early, late)

    with pytest.raises(ScopeViolation):
        coordinator.seal_observation_chunk(
            chunk_id="c3",
            observations=[observation("foreign", scope="world-b", seed=3)],
            reader_ids=("r",), sealed_at=3.0)
    with pytest.raises(NestedLearningError):
        Observation(event_id="query", scope="world-a", key=(1.0,),
                    value=(1.0,), available_at=1.0, source="teacher_forced")


def test_backdated_cutoff_views_and_seals_fail_closed():
    coordinator = ObservationCoordinator(scope="world-a", width=3, seed=17)
    early = observation("early", available_at=2.0, seed=1)
    late = observation("late", available_at=6.0, seed=2)
    coordinator.seal_observation_chunk(
        chunk_id="c1", observations=[early], reader_ids=("r",), sealed_at=2.0)
    coordinator.seal_observation_chunk(
        chunk_id="c2", observations=[late], reader_ids=("r",), sealed_at=6.0)
    state = coordinator.fast_state
    views_before = {key: value.view_id
                    for key, value in coordinator.last_reader_views.items()}

    with pytest.raises(BackdatedStateError):
        coordinator.view(cutoff=2.0)
    with pytest.raises(BackdatedStateError):
        coordinator.view(cutoff=5.0)
    safe = coordinator.view(cutoff=6.0)
    assert safe.cutoff == 6.0
    assert safe.revision == 2
    assert {item.event_id for item in safe.observations} == {"early", "late"}
    assert len(safe.view_id) == 64
    assert coordinator.view(cutoff=6.0).view_id == safe.view_id
    future_safe = coordinator.view(cutoff=7.0)
    assert future_safe.cutoff == 7.0
    assert future_safe.view_id != safe.view_id
    assert coordinator.view().cutoff is None
    assert coordinator.view().view_id != safe.view_id

    # ``read`` exposes observations only, never a mixed fast state.
    read_early = coordinator.read(scope="world-a", cutoff=2.0)
    assert read_early.observations == (early,)

    with pytest.raises(BackdatedStateError):
        coordinator.seal_observation_chunk(
            chunk_id="c3",
            observations=[observation("backdated", available_at=3.0, seed=3)],
            reader_ids=("r",), sealed_at=3.0)
    assert coordinator.revision == 2
    torch.testing.assert_close(coordinator.fast_state, state, rtol=0, atol=0)
    assert {key: value.view_id
            for key, value in coordinator.last_reader_views.items()} == views_before

    # A retry of an already committed earlier chunk remains an idempotent retry.
    retry = coordinator.seal_observation_chunk(
        chunk_id="c1", observations=[early], reader_ids=("r", "r2"), sealed_at=2.0)
    assert retry.retried is True
    assert retry.post_revision == 1
    assert coordinator.revision == 2
    torch.testing.assert_close(coordinator.fast_state, state, rtol=0, atol=0)


def test_public_fast_state_is_mutation_isolated():
    coordinator = ObservationCoordinator(scope="world-a", width=3, seed=18)
    coordinator.seal_observation_chunk(
        chunk_id="c1",
        observations=[observation("o1", available_at=1.0, seed=1)],
        reader_ids=("r",), sealed_at=1.0)
    canonical = coordinator.fast_state
    exposed = coordinator.fast_state
    assert exposed.data_ptr() != canonical.data_ptr()
    with torch.no_grad():
        exposed.add_(123.0)
    torch.testing.assert_close(coordinator.fast_state, canonical, rtol=0, atol=0)
    checkpoint = coordinator.checkpoint_payload()["fast_state"]
    torch.testing.assert_close(checkpoint, canonical, rtol=0, atol=0)


def test_observation_reduction_uses_canonical_event_order():
    def raw(event_id: str, key: tuple[float, ...],
            value: tuple[float, ...], available_at: float) -> Observation:
        return Observation(event_id=event_id, scope="world-a", key=key,
                           value=value, available_at=available_at)

    # Canonical event-id order is a -> c -> b and sums (1e16 - 1e16) + 1.
    # The caller-order A -> B -> C sums (1e16 + 1) - 1e16 and cancels.
    first = raw("a", (1.0, 0.0, 0.0), (1e16, 0.0, 0.0), 1.0)
    third = raw("c", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 3.0)
    second = raw("b", (0.0, 1.0, 0.0), (-1e16, 0.0, 0.0), 2.0)
    caller_one = [first, third, second]
    caller_two = [second, first, third]
    naive_one = 0.0
    for item in caller_one:
        naive_one += item.value[0]
    naive_two = 0.0
    for item in caller_two:
        naive_two += item.value[0]
    assert naive_one != naive_two

    left = ObservationCoordinator(scope="world-a", width=3, seed=19,
                                  rho=0.9, eta=0.25)
    right = ObservationCoordinator(scope="world-a", width=3, seed=19,
                                   rho=0.9, eta=0.25)
    left_outcome = left.seal_observation_chunk(
        chunk_id="c1", observations=caller_one,
        reader_ids=("r",), sealed_at=3.0)
    right_outcome = right.seal_observation_chunk(
        chunk_id="c1", observations=caller_two,
        reader_ids=("r",), sealed_at=3.0)
    assert left_outcome.aggregate_key == right_outcome.aggregate_key
    assert left_outcome.aggregate_value == right_outcome.aggregate_value
    assert left_outcome.aggregate_value[0] == pytest.approx(1.0 / 3.0)
    assert left.reduce_observations(caller_one, width=3) == \
        right.reduce_observations(caller_two, width=3)
    torch.testing.assert_close(left.fast_state, right.fast_state, rtol=0, atol=0)
    assert left.transition_count == right.transition_count == 1


def test_chunk_dedup_retry_and_changed_identity_fail_closed():
    coordinator = ObservationCoordinator(scope="world-a", width=3, seed=8)
    observations = [observation("e1", available_at=1.0, seed=1),
                    observation("e2", available_at=2.0, seed=2)]
    first = coordinator.seal_observation_chunk(
        chunk_id="c1", observations=observations,
        reader_ids=("r",), sealed_at=2.0)
    state = coordinator.fast_state.clone()
    retry = coordinator.seal_observation_chunk(
        chunk_id="c1", observations=observations,
        reader_ids=("r", "r2"), sealed_at=2.0)
    assert retry.retried is True
    assert retry.fingerprint == first.fingerprint
    assert retry.post_revision == first.post_revision
    assert coordinator.revision == 1
    assert coordinator.transition_count == 1
    torch.testing.assert_close(coordinator.fast_state, state, rtol=0, atol=0)

    with pytest.raises(DuplicateObservation):
        coordinator.seal_observation_chunk(
            chunk_id="c1",
            observations=[observation("e3", available_at=3.0, seed=3)],
            reader_ids=("r",), sealed_at=3.0)
    assert coordinator.revision == 1
    torch.testing.assert_close(coordinator.fast_state, state, rtol=0, atol=0)

    with pytest.raises(DuplicateObservation):
        coordinator.seal_observation_chunk(
            chunk_id="c2", observations=[observations[0]],
            reader_ids=("r",), sealed_at=2.0)
    assert coordinator.revision == 1


def test_observer_and_spine_absence_parity_and_read_only_snapshot():
    plain = ObservationCoordinator(scope="world-a", width=3, seed=9,
                                   rho=0.92, eta=0.2)
    observer = Observer()
    spine = Observer()
    assert observer.read_only is True and spine.read_only is True
    observed = ObservationCoordinator(scope="world-a", width=3, seed=9,
                                      rho=0.92, eta=0.2,
                                      observer=observer, spine=spine)
    observations = [observation("o1", available_at=1.0, seed=1),
                    observation("o2", available_at=2.0, seed=2)]
    plain_outcome = plain.seal_observation_chunk(
        chunk_id="c1", observations=observations,
        reader_ids=("r",), sealed_at=2.0)
    observed_outcome = observed.seal_observation_chunk(
        chunk_id="c1", observations=observations,
        reader_ids=("r",), sealed_at=2.0)
    assert plain_outcome.post_revision == observed_outcome.post_revision
    assert plain_outcome.transition_count == observed_outcome.transition_count == 1
    torch.testing.assert_close(plain.fast_state, observed.fast_state, rtol=0, atol=0)
    assert plain.revision == observed.revision
    assert len(observer.reports) == len(spine.reports) == 1
    assert observer.reports[0]["revision"] == 0
    assert observer.reports[0]["cutoff"] == 2.0
    assert len(observer.reports[0]["view_id"]) == 64
    assert observer.reports[0]["readonly"] is True

    view = observed.last_reader_views["r"]
    state_before = observed.fast_state.clone()
    assert view.readonly is True
    assert view.cutoff == 2.0
    assert len(view.view_id) == 64
    assert view.fast_state.requires_grad is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.revision = 9  # type: ignore[misc]
    with torch.no_grad():
        view.fast_state.add_(100.0)
    torch.testing.assert_close(observed.fast_state, state_before, rtol=0, atol=0)
    assert observed.revision == 1

    observed.detach_observer()
    observed.detach_spine()
    plain.seal_observation_chunk(
        chunk_id="c2",
        observations=[observation("o3", available_at=3.0, seed=3)],
        reader_ids=("r",), sealed_at=3.0)
    observed.seal_observation_chunk(
        chunk_id="c2",
        observations=[observation("o3", available_at=3.0, seed=3)],
        reader_ids=("r",), sealed_at=3.0)
    torch.testing.assert_close(plain.fast_state, observed.fast_state, rtol=0, atol=0)
    assert plain.revision == observed.revision == 2
    assert len(observer.reports) == 1


# ---------------------------------------------------------------------------
# Persistence, atomicity, and personality
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_preserves_state_and_dedup(tmp_path: Path):
    bank = tiny_bank(width=2, hidden_width=3, seed=31)
    coordinator = ObservationCoordinator(scope="world-a", width=2, seed=31)
    controller = PlasticLearningController(bank, scope="world-a",
                                           coordinator=coordinator)
    features, target = tiny_inputs(2, seed=32)
    transaction = controller.apply_teaching_transaction(
        event_id="teach", expert_id=5,
        features=features, target=target, steps=2)
    obs = observation("obs", width=2, available_at=4.0, seed=33)
    chunk = coordinator.seal_observation_chunk(
        chunk_id="chunk", observations=[obs], reader_ids=("r",), sealed_at=4.0)
    path = tmp_path / "checkpoint.pt"
    controller.save_checkpoint(path)
    assert path.exists()
    initial_bytes = path.read_bytes()

    restored_bank = tiny_bank(width=2, hidden_width=3, seed=31)
    restored_coordinator = ObservationCoordinator(scope="world-a", width=2,
                                                  seed=31)
    restored = PlasticLearningController(
        restored_bank, scope="world-a", coordinator=restored_coordinator)
    used_features, used_target = tiny_inputs(2, seed=34)
    restored.apply_teaching_transaction(
        event_id="target-used", expert_id=2,
        features=used_features, target=used_target, steps=1)
    restored_coordinator.seal_observation_chunk(
        chunk_id="target-used-chunk",
        observations=[observation("target-used-obs", width=2,
                                  available_at=2.0, seed=35)],
        reader_ids=("r",), sealed_at=2.0)
    assert restored_coordinator.last_reader_views
    restored.load_checkpoint_file(path)
    assert restored_coordinator.last_reader_views == {}

    for left, right in zip(bank.parameters(), restored_bank.parameters()):
        assert torch.equal(left, right)
    assert restored.bank.expert_steps == bank.expert_steps
    assert restored.clock == controller.clock
    assert restored.revision == controller.revision
    assert restored.personality == controller.personality
    assert restored.dedup.event_ids == controller.dedup.event_ids
    assert_optimizer_state_equal(
        bank.optimizer_state_snapshot(range(bank.expert_count)),
        restored_bank.optimizer_state_snapshot(range(restored_bank.expert_count)))
    torch.testing.assert_close(restored_coordinator.fast_state,
                               coordinator.fast_state, rtol=0, atol=0)
    assert restored_coordinator.revision == coordinator.revision
    assert restored_coordinator.transition_count == coordinator.transition_count

    retry = restored.apply_teaching_transaction(
        event_id="teach", expert_id=5,
        features=features, target=target, steps=2)
    assert retry.retried is True
    assert retry.revision == transaction.revision
    chunk_retry = restored_coordinator.seal_observation_chunk(
        chunk_id="chunk", observations=[obs],
        reader_ids=("r", "another"), sealed_at=4.0)
    assert chunk_retry.retried is True
    assert chunk_retry.post_revision == chunk.post_revision
    assert restored_coordinator.transition_count == coordinator.transition_count
    with pytest.raises(DuplicateObservation):
        restored_coordinator.seal_observation_chunk(
            chunk_id="chunk",
            observations=[observation("different", width=2, seed=99)],
            reader_ids=("r",), sealed_at=4.0)
    assert restored_coordinator.transition_count == coordinator.transition_count

    def failing_writer(_payload, destination: Path) -> None:
        Path(destination).write_bytes(b"partial")
        raise RuntimeError("injected checkpoint write failure")

    with pytest.raises(RuntimeError, match="checkpoint write failure"):
        restored.save_checkpoint(path, writer=failing_writer)
    assert path.read_bytes() == initial_bytes
    assert not list(tmp_path.glob(".checkpoint.pt.*.tmp"))

    different_scope_bank = tiny_bank(width=2, hidden_width=3, seed=31)
    different_scope_coordinator = ObservationCoordinator(
        scope="world-b", width=2, seed=31)
    different_scope = PlasticLearningController(
        different_scope_bank, scope="world-b",
        coordinator=different_scope_coordinator)
    with pytest.raises(ScopeViolation):
        different_scope.load_checkpoint_file(path)

    different_identity = PersonalityIdentity.default(snapshot="different")
    identity_bank = tiny_bank(width=2, hidden_width=3, seed=31)
    identity_coordinator = ObservationCoordinator(
        scope="world-a", width=2, seed=31, personality=different_identity)
    identity_controller = PlasticLearningController(
        identity_bank, scope="world-a", coordinator=identity_coordinator,
        personality=different_identity)
    with pytest.raises(NestedLearningError, match="personality"):
        identity_controller.load_checkpoint_file(path)


def _used_target_controller(*, seed: int,
                            betas: tuple[float, float] = (0.9, 0.999)
                            ) -> PlasticLearningController:
    bank = ExpertBank(expert_count=16, width=2, hidden_width=3, seed=seed,
                      lr=0.02, betas=betas, eps=1e-8, weight_decay=0.0)
    coordinator = ObservationCoordinator(scope="world-a", width=2, seed=seed)
    controller = PlasticLearningController(
        bank, scope="world-a", coordinator=coordinator)
    features, target = tiny_inputs(2, seed=seed + 1)
    controller.apply_teaching_transaction(
        event_id=f"target-{seed}", expert_id=4,
        features=features, target=target, steps=2)
    coordinator.seal_observation_chunk(
        chunk_id=f"target-chunk-{seed}",
        observations=[observation(f"target-obs-{seed}", width=2,
                                  available_at=2.0, seed=seed + 2)],
        reader_ids=("r",), sealed_at=2.0)
    controller.cache.put("live", "live-value")
    return controller


def _source_controller_payload(*, seed: int = 61) -> dict:
    bank = ExpertBank(expert_count=16, width=2, hidden_width=3, seed=seed,
                      lr=0.02, betas=(0.9, 0.999), eps=1e-8,
                      weight_decay=0.0)
    coordinator = ObservationCoordinator(scope="world-a", width=2, seed=seed)
    controller = PlasticLearningController(
        bank, scope="world-a", coordinator=coordinator)
    features, target = tiny_inputs(2, seed=seed + 1)
    controller.apply_teaching_transaction(
        event_id=f"source-{seed}", expert_id=5,
        features=features, target=target, steps=2)
    coordinator.seal_observation_chunk(
        chunk_id=f"source-chunk-{seed}",
        observations=[observation(f"source-obs-{seed}", width=2,
                                  available_at=2.0, seed=seed + 2)],
        reader_ids=("r",), sealed_at=2.0)
    return controller.checkpoint_payload()


def test_used_controller_load_failures_are_atomic():
    payload = _source_controller_payload(seed=61)
    target = _used_target_controller(seed=61)
    snapshot = _live_controller_snapshot(target)

    invalid_steps = copy.deepcopy(payload)
    invalid_steps["bank"]["expert_steps"] = (
        [-1] + list(invalid_steps["bank"]["expert_steps"][1:]))
    with pytest.raises(NestedLearningError, match="expert counters"):
        target.load_checkpoint(invalid_steps)
    _assert_live_controller_equal(target, snapshot)

    invalid_revision = copy.deepcopy(payload)
    invalid_revision["revision"] = -1
    with pytest.raises(NestedLearningError):
        target.load_checkpoint(invalid_revision)
    _assert_live_controller_equal(target, snapshot)

    invalid_state = copy.deepcopy(payload)
    invalid_state["observation"]["fast_state"] = \
        invalid_state["observation"]["fast_state"].double()
    with pytest.raises(NestedLearningError, match="FP32"):
        target.load_checkpoint(invalid_state)
    _assert_live_controller_equal(target, snapshot)

    invalid_weight = copy.deepcopy(payload)
    last_parameter = invalid_weight["bank"]["parameters"][-1]
    last_parameter[1] = torch.full_like(last_parameter[1], 1e300,
                                        dtype=torch.float64)
    with pytest.raises(NestedLearningError, match="finite FP32"):
        target.load_checkpoint(invalid_weight)
    _assert_live_controller_equal(target, snapshot)

    nonfinite_weight = copy.deepcopy(payload)
    last_parameter = nonfinite_weight["bank"]["parameters"][-1]
    last_parameter[1] = torch.full_like(last_parameter[1], float("inf"),
                                        dtype=torch.float32)
    with pytest.raises(NestedLearningError, match="finite FP32"):
        target.load_checkpoint(nonfinite_weight)
    _assert_live_controller_equal(target, snapshot)

    invalid_chunk = copy.deepcopy(payload)
    invalid_chunk["observation"]["revision"] = 99
    with pytest.raises(NestedLearningError):
        target.load_checkpoint(invalid_chunk)
    _assert_live_controller_equal(target, snapshot)

    invalid_clock = copy.deepcopy(payload)
    invalid_clock["clock"] = {"eligible_events": -1}
    with pytest.raises(NestedLearningError):
        target.load_checkpoint(invalid_clock)
    _assert_live_controller_equal(target, snapshot)

    invalid_dedup = copy.deepcopy(payload)
    invalid_dedup["dedup"] = {"format_version": 999}
    with pytest.raises(NestedLearningError):
        target.load_checkpoint(invalid_dedup)
    _assert_live_controller_equal(target, snapshot)

    invalid_boundary = copy.deepcopy(payload)
    invalid_boundary["observation"]["latest_sealed_at"] = 99.0
    with pytest.raises(NestedLearningError):
        target.load_checkpoint(invalid_boundary)
    _assert_live_controller_equal(target, snapshot)

    invalid_optimizer = copy.deepcopy(payload)
    state_key = next(iter(invalid_optimizer["bank"]["optimizer"]["state"]))
    invalid_optimizer["bank"]["optimizer"]["state"][state_key]["exp_avg"] = \
        torch.zeros(7, dtype=torch.float32)
    with pytest.raises(ValueError):
        target.load_checkpoint(invalid_optimizer)
    _assert_live_controller_equal(target, snapshot)

    # A coordinator with different optimizer betas fails at late fused-state
    # validation after weights were staged; the whole live state is restored.
    beta_target = _used_target_controller(seed=61, betas=(0.8, 0.991))
    beta_snapshot = _live_controller_snapshot(beta_target)
    with pytest.raises(ValueError):
        beta_target.load_checkpoint(payload)
    _assert_live_controller_equal(beta_target, beta_snapshot)

    # A successful load still replaces the used state and clears ephemeral views.
    target.load_checkpoint(payload)
    assert target.coordinator.last_reader_views == {}
    assert target.dedup.event_ids == ("source-61",)
    assert target.coordinator.transition_count == 1


def test_observation_restore_rejects_non_fp32_and_clears_views():
    source = ObservationCoordinator(scope="world-a", width=2, seed=72)
    source.seal_observation_chunk(
        chunk_id="source",
        observations=[observation("source-obs", width=2,
                                  available_at=3.0, seed=73)],
        reader_ids=("r",), sealed_at=3.0)
    payload = source.checkpoint_payload()

    target = ObservationCoordinator(scope="world-a", width=2, seed=74)
    target.seal_observation_chunk(
        chunk_id="target",
        observations=[observation("target-obs", width=2,
                                  available_at=2.0, seed=75)],
        reader_ids=("r",), sealed_at=2.0)
    snapshot = _coordinator_live_snapshot(target)
    assert target.last_reader_views

    invalid_double = copy.deepcopy(payload)
    invalid_double["fast_state"] = invalid_double["fast_state"].double()
    with pytest.raises(NestedLearningError, match="FP32"):
        target.load_checkpoint(invalid_double)
    _assert_coordinator_live(target, snapshot)

    invalid_integer = copy.deepcopy(payload)
    invalid_integer["fast_state"] = torch.zeros(
        2, 2, dtype=torch.int64)
    with pytest.raises(NestedLearningError, match="FP32"):
        target.load_checkpoint(invalid_integer)
    _assert_coordinator_live(target, snapshot)

    target.load_checkpoint(payload)
    assert target.last_reader_views == {}
    torch.testing.assert_close(target.fast_state, source.fast_state,
                               rtol=0, atol=0)
    assert target.revision == source.revision
    assert target.transition_count == source.transition_count


def test_chunk_accounting_validation_rejects_tampered_payloads():
    eligible_obs = observation("account-eligible", width=2,
                               available_at=1.0, seed=81, eligible=True)
    ineligible_obs = observation("account-ineligible", width=2,
                                 available_at=2.0, seed=82, eligible=False)
    source_bank = tiny_bank(width=2, hidden_width=3, seed=81)
    source_coordinator = ObservationCoordinator(scope="world-a", width=2,
                                                seed=81)
    source = PlasticLearningController(
        source_bank, scope="world-a", coordinator=source_coordinator)
    source_coordinator.seal_observation_chunk(
        chunk_id="account-eligible", observations=[eligible_obs],
        reader_ids=("r",), sealed_at=1.0)
    source_coordinator.seal_observation_chunk(
        chunk_id="account-ineligible", observations=[ineligible_obs],
        reader_ids=("r",), sealed_at=2.0)
    source_coordinator.seal_observation_chunk(
        chunk_id="account-empty", observations=[],
        reader_ids=("r",), sealed_at=3.0)
    assert source_coordinator.transition_count == 1
    assert source_coordinator.revision == 2
    assert source_coordinator.checkpoint_payload()["latest_sealed_at"] == 3.0
    payload = source.checkpoint_payload()
    chunks = {item["chunk_id"]: item
              for item in payload["observation"]["chunks"]}
    assert chunks["account-eligible"]["transition_count"] == 1
    assert chunks["account-ineligible"]["transition_count"] == 0
    assert chunks["account-empty"]["transition_count"] == 0
    assert (chunks["account-empty"]["post_revision"]
            == chunks["account-empty"]["pre_revision"])
    original_fingerprints = {key: item["fingerprint"]
                             for key, item in chunks.items()}

    def tampered(kind: str) -> dict:
        invalid = copy.deepcopy(payload)
        by_id = {item["chunk_id"]: item
                 for item in invalid["observation"]["chunks"]}
        if kind == "eligible-zero":
            by_id["account-eligible"]["transition_count"] = 0
        elif kind == "ineligible-one":
            by_id["account-ineligible"]["transition_count"] = 1
        elif kind == "empty-visible":
            by_id["account-empty"]["post_revision"] = (
                by_id["account-empty"]["pre_revision"] + 1)
        else:
            raise AssertionError(kind)
        assert {key: item["fingerprint"]
                for key, item in by_id.items()} == original_fingerprints
        return invalid

    variants = ("eligible-zero", "ineligible-one", "empty-visible")

    direct = ObservationCoordinator(scope="world-a", width=2, seed=82)
    direct.seal_observation_chunk(
        chunk_id="account-target",
        observations=[observation("account-target-obs", width=2,
                                  available_at=1.5, seed=83)],
        reader_ids=("r",), sealed_at=2.0)
    direct_snapshot = _coordinator_live_snapshot(direct)
    for kind in variants:
        with pytest.raises(NestedLearningError, match="disagrees"):
            direct.load_checkpoint(tampered(kind)["observation"])
        _assert_coordinator_live(direct, direct_snapshot)
    direct.load_checkpoint(payload["observation"])
    assert direct.transition_count == 1
    assert direct.revision == 2
    assert direct.checkpoint_payload()["latest_sealed_at"] == 3.0
    assert direct.last_reader_views == {}

    target = _used_target_controller(seed=81)
    target_snapshot = _live_controller_snapshot(target)
    for kind in variants:
        with pytest.raises(NestedLearningError, match="disagrees"):
            target.load_checkpoint(tampered(kind))
        _assert_live_controller_equal(target, target_snapshot)
    target.load_checkpoint(payload)
    assert target.coordinator.transition_count == 1
    assert target.coordinator.revision == 2
    assert target.coordinator.checkpoint_payload()["latest_sealed_at"] == 3.0
    assert target.coordinator.last_reader_views == {}
    for chunk_id, items, sealed_at in (
            ("account-eligible", [eligible_obs], 1.0),
            ("account-ineligible", [ineligible_obs], 2.0),
            ("account-empty", [], 3.0)):
        retry = target.coordinator.seal_observation_chunk(
            chunk_id=chunk_id, observations=items,
            reader_ids=("r",), sealed_at=sealed_at)
        assert retry.retried is True
    config = load_nl0_config()
    assert config["observation"]["transitions_per_sealed_chunk_max"] == 1
    assert config["observation"]["transitions_when_no_eligible"] == 0
    assert config["observation"]["transitions_when_any_eligible"] == 1


def test_personality_identity_is_stable_frozen_and_versioned():
    identity = PersonalityIdentity.default()
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.version = 2  # type: ignore[misc]
    assert identity.version >= 1
    assert identity.key().startswith("id/")
    bank = tiny_bank(width=2, hidden_width=3, seed=34)
    coordinator = ObservationCoordinator(scope="world-a", width=2, seed=34,
                                         personality=identity)
    controller = PlasticLearningController(bank, scope="world-a",
                                           coordinator=coordinator,
                                           personality=identity)
    controller.apply_teaching_transaction(
        event_id="e", expert_id=0, features=torch.tensor([[0.2, 0.4]]),
        target=torch.tensor([[0.1, 0.3]]), steps=1)
    coordinator.seal_observation_chunk(
        chunk_id="c", observations=[observation("o", width=2, seed=35)],
        reader_ids=("r",), sealed_at=1.0)
    assert controller.personality == identity
    assert coordinator.personality == identity
    assert controller.personality.key() == identity.key()

    inherited = PersonalityIdentity.default(snapshot="coordinator-only")
    inherited_coordinator = ObservationCoordinator(
        scope="world-a", width=2, seed=37, personality=inherited)
    inherited_controller = PlasticLearningController(
        tiny_bank(width=2, hidden_width=3, seed=37), scope="world-a",
        coordinator=inherited_coordinator)
    assert inherited_controller.personality == inherited
    mismatched = PersonalityIdentity.default(snapshot="explicit-mismatch")
    with pytest.raises(NestedLearningError, match="personality"):
        PlasticLearningController(
            tiny_bank(width=2, hidden_width=3, seed=38), scope="world-a",
            personality=mismatched, coordinator=inherited_coordinator)


def test_no_forward_or_backward_hooks_store_state():
    bank = tiny_bank(width=2, hidden_width=3, seed=36)
    controller = PlasticLearningController(bank, scope="world-a")
    controller.apply_teaching_transaction(
        event_id="e", expert_id=0, features=torch.tensor([[0.2, 0.4]]),
        target=torch.tensor([[0.1, 0.3]]), steps=1)
    for module in bank.modules():
        assert not getattr(module, "_forward_hooks", {})
        assert not getattr(module, "_forward_pre_hooks", {})
        assert not getattr(module, "_backward_hooks", {})
        assert not getattr(module, "_backward_pre_hooks", {})
    for expert in bank.experts:
        for parameter in expert.parameters():
            assert not getattr(parameter, "_backward_hooks", {})


def test_dedup_ledger_retry_and_changed_identity_are_explicit():
    ledger = DedupLedger()
    outcome = TransactionOutcome(
        status="committed", event_id="e", fingerprint="a" * 64,
        expert_id=0, revision=1, eligible_events=1, steps=1, loss=0.5)
    ledger.commit("e", "a" * 64, outcome)
    assert ledger.require("e", "a" * 64) == outcome
    assert ledger.lookup("e") == outcome
    with pytest.raises(DuplicateObservation):
        ledger.require("e", "b" * 64)
    snapshot = ledger.snapshot()
    restored = DedupLedger.from_dict(ledger.to_dict())
    assert restored.lookup("e") == outcome
    ledger.restore(snapshot)
    assert ledger.lookup("e") == outcome


# ---------------------------------------------------------------------------
# One-clock mechanical controls and configuration
# ---------------------------------------------------------------------------


def test_fixed_and_shuffled_routing_are_deterministic_mechanical_controls():
    generator = torch.Generator(device="cpu").manual_seed(40)
    keys = [torch.randn(4, generator=generator) for _ in range(24)]
    fixed = RouteControl(mode="fixed", expert_count=16, seed=3)
    fixed_again = RouteControl(mode="fixed", expert_count=16, seed=3)
    shuffled = RouteControl(mode="shuffled", expert_count=16, seed=3)
    shuffled_again = RouteControl(mode="shuffled", expert_count=16, seed=3)
    assert fixed.permutation == tuple(range(16))
    assert sorted(shuffled.permutation) == list(range(16))
    assert shuffled.permutation != fixed.permutation
    fixed_routes = [fixed.route(key) for key in keys]
    assert fixed_routes == [fixed_again.route(key) for key in keys]
    shuffled_routes = [shuffled.route(key) for key in keys]
    assert shuffled_routes == [shuffled_again.route(key) for key in keys]
    assert all(0 <= route < 16 for route in fixed_routes + shuffled_routes)
    assert fixed_routes != shuffled_routes
    with pytest.raises(NestedLearningError):
        RouteControl(mode="learned", expert_count=16)
    with pytest.raises(NestedLearningError):
        RouteControl(mode="fixed", expert_count=0)


def test_config_declares_software_only_nl0_and_not_training_authorization():
    config = load_nl0_config()
    assert config["software_only"] is True
    for key in ("native_execution", "training_admitted", "capability_claim",
                "fast_slow_timing_claimed", "adaptive_router_utility_claimed",
                "multimodal_work"):
        assert config[key] is False
    assert config["bank"]["experts"] == DECLARED_EXPERT_COUNT == 16
    assert config["bank"]["width"] == 256
    assert config["bank"]["hidden_width"] == 1024
    assert config["bank"]["selected_only_updates"] is True
    assert config["bank"]["inactive_moment_decay"] is False
    assert config["optimizer"]["implementation"] == "NativeFusedAdamW"
    assert config["optimizer"]["exact_reference_horizons"] == [1, 2, 4]
    assert config["optimizer"]["gradient_norm_max"] == 1.0
    assert config["optimizer"]["clip_before_adamw"] is True
    assert config["transaction"]["checkpoint_atomic"] is True
    assert config["transaction"]["checkpoint_load_atomic"] is True
    assert config["observation"]["transitions_per_sealed_chunk_max"] == 1
    assert config["observation"]["transitions_when_no_eligible"] == 0
    assert config["observation"]["transitions_when_any_eligible"] == 1
    assert "transitions_per_sealed_chunk" not in config["observation"]
    assert config["observation"]["cutoff_state_views_fail_closed"] is True
    assert config["observation"]["backdated_seals_fail_closed"] is True
    assert config["observation"]["fast_state_dtype"] == "FP32"
    assert config["observation"]["canonical_reduction_order"] == "event_id"
    assert config["observation"]["visible_write_advances_revision"] is True
    assert config["controls"]["controller_inherits_coordinator_personality"] is True
    assert config["observation"]["reader_count_multiplies_updates"] is False
    assert config["recurrence"]["transition"] == "F_next = rho*F + eta*(v-F*k)*k^T"
    assert config["recurrence"]["loss"] == "0.5*||F*k-v||^2"
    assert config["controls"]["routing"] == ["fixed", "shuffled"]
    assert "PlasticLearningController" in config["integration_points"]["plastic_transaction"]
    assert "ObservationCoordinator" in config["integration_points"]["observation_coordinator"]

    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert validate_nl0_config(raw)["status"] == "software_math_qualified"
    unauthorized = copy.deepcopy(raw)
    unauthorized["training_admitted"] = True
    with pytest.raises(NestedLearningError, match="training_admitted"):
        validate_nl0_config(unauthorized)
    incomplete = copy.deepcopy(raw)
    incomplete["transaction"]["rollback_includes"] = ["weights"]
    with pytest.raises(NestedLearningError, match="rollback"):
        validate_nl0_config(incomplete)
    legacy_transition = copy.deepcopy(raw)
    legacy_transition["observation"]["transitions_per_sealed_chunk"] = 1
    with pytest.raises(NestedLearningError, match="fixed one-transition"):
        validate_nl0_config(legacy_transition)


def test_revision_cache_staleness_and_invalidation_are_explicit():
    cache = RevisionCache(revision=3)
    cache.put("a", 1)
    assert cache.get("a") == 1
    cache.bind(4)
    with pytest.raises(StaleCache):
        cache.get("a")
    cache.put("a", 2)
    assert cache.get("a") == 2
    cache.invalidate(5)
    with pytest.raises(StaleCache):
        cache.get("a")
    with pytest.raises(KeyError):
        cache.get("missing")
    with pytest.raises(KeyError):
        RevisionCache(revision=0).get("missing")
    with pytest.raises(NestedLearningError):
        RevisionCache(revision=-1)
