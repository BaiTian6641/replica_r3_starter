"""Unit contracts for the D-only learned local writer fixture.

These tests exercise only the CPU engineering fixture.  They do not load Spark,
do not run native training, do not open sealed evaluation and do not claim
native capability from parameter changes.
"""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

np = pytest.importorskip("numpy")

from replica_lab.learned_local_writer import (
    ARM_ADAMW_REFERENCE,
    ARM_DELTA,
    ARM_SGD,
    DERIVATIVE_EXACT,
    DERIVATIVE_TRUNCATED,
    EVIDENCE_CLASS,
    FUNCTIONAL_ARMS,
    SEALED_SPLIT,
    Episode,
    FixedBackbone,
    SealedEvaluationClosed,
    WriterParams,
    WriterShapes,
    adamw_reference_step,
    cross_entropy,
    declared_meta_ops,
    declared_update_ops,
    evaluate_writer,
    finite_difference_gradient_check,
    functional_delta_step,
    functional_sgd_step,
    make_world_suite,
    one_hot,
    outer_value_and_gradient,
    softmax,
    train_writer,
)


@pytest.fixture(scope="module")
def writer_fixture():
    shapes = WriterShapes()
    backbone = FixedBackbone.from_seed(shapes, seed=11)
    suite = make_world_suite(seed=11, shapes=shapes)
    phi = WriterParams.initial(shapes, seed=5)
    return shapes, backbone, suite, phi


def test_world_suite_is_independent_and_queries_are_distinct(writer_fixture):
    shapes, _backbone, suite, _phi = writer_fixture
    train_worlds = {episode.world_id for episode in suite.train}
    development_worlds = {episode.world_id for episode in suite.development}
    sealed_worlds = set(suite.sealed_world_ids)

    assert train_worlds.isdisjoint(development_worlds)
    assert train_worlds.isdisjoint(sealed_worlds)
    assert development_worlds.isdisjoint(sealed_worlds)
    assert set(suite.family_splits["train"]) == {"direct", "shift"}
    assert set(suite.family_splits["development"]) == {"mirror"}
    assert set(suite.family_splits["train"]).isdisjoint(
        suite.family_splits["development"]
    )

    for episode in suite.train + suite.development:
        assert episode.future
        assert episode.write.query_id not in {
            example.query_id for example in episode.future
        }
        assert all(example.world_id == episode.world_id for example in episode.future)
        assert all(example.split == episode.split for example in episode.future)
        assert all(
            example.query_id != episode.write.query_id
            for example in episode.future
        )
        assert len({example.query_id for example in episode.future}) == len(
            episode.future
        )
        assert all(example.label < shapes.classes for example in episode.future)


def test_sealed_split_is_closed_to_every_training_and_evaluation_path(writer_fixture):
    _shapes, backbone, suite, phi = writer_fixture
    with pytest.raises(SealedEvaluationClosed):
        suite.episodes(SEALED_SPLIT)

    source = suite.train[0]
    sealed_write = replace(
        source.write,
        query_id="sealed-write",
        split=SEALED_SPLIT,
    )
    sealed_future = tuple(
        replace(example, query_id=f"{example.query_id}-sealed", split=SEALED_SPLIT)
        for example in source.future
    )
    sealed_episode = Episode(
        episode_id="sealed-episode",
        world_id=source.world_id,
        family=source.family,
        split=SEALED_SPLIT,
        write=sealed_write,
        future=sealed_future,
        replay=(),
    )
    with pytest.raises(ValueError):
        train_writer(backbone, (sealed_episode,), outer_steps=1)
    with pytest.raises(ValueError):
        evaluate_writer(backbone, phi, (sealed_episode,))

    report = train_writer(
        backbone,
        suite.train,
        outer_steps=1,
        replay_weight=1.0,
        evaluation_episodes=suite.development,
    )
    assert report["execution"]["sealed_opened"] is False
    assert report["sealed"]["opened"] is False
    assert report["sealed"]["evaluated"] is False


def test_sgd_step_matches_its_declared_equation(writer_fixture):
    shapes, backbone, suite, phi = writer_fixture
    episode = suite.train[0]
    step = functional_sgd_step(
        backbone, phi, backbone.D0, episode.write, inner_lr=0.7
    )

    h, _cache = backbone.feature(step.z)
    probabilities = softmax(backbone.readout @ (backbone.D0 @ h))
    target = one_hot(episode.write.label, shapes.classes)
    inner_grad = np.outer(backbone.readout.T @ (probabilities - target), h)
    expected_delta = -0.7 * phi.gain * inner_grad

    assert np.allclose(step.inner_grad, inner_grad)
    assert np.allclose(step.delta, expected_delta)
    assert np.allclose(step.D_prime, backbone.D0 + expected_delta)
    assert step.cost.changed_entries > 0
    assert step.cost.delta_norm > 0.0
    assert step.cost.measured_seconds >= 0.0
    assert step.cost.scalar_ops == declared_update_ops(ARM_SGD, shapes)


def test_delta_step_matches_its_declared_equation(writer_fixture):
    shapes, backbone, suite, phi = writer_fixture
    episode = suite.train[0]
    step = functional_delta_step(
        backbone, phi, backbone.D0, episode.write, inner_lr=0.7
    )

    context = np.concatenate(
        [episode.write.vector(), one_hot(episode.write.label, shapes.classes)]
    )
    z = np.tanh(phi.W_ctx @ context + phi.b_ctx)
    value = phi.W_val @ z + phi.b_val
    h, _cache = backbone.feature(episode.write.vector())
    expected_delta = 0.7 * phi.gain * np.outer(value, h)

    assert np.allclose(step.value, value)
    assert np.allclose(step.delta, expected_delta)
    assert np.allclose(step.D_prime, backbone.D0 + expected_delta)
    assert step.p is None and step.q is None
    assert step.cost.scalar_ops == declared_update_ops(ARM_DELTA, shapes)


def test_sgd_exact_meta_gradient_matches_finite_differences(writer_fixture):
    _shapes, backbone, suite, phi = writer_fixture
    episode = suite.train[0]
    check = finite_difference_gradient_check(
        backbone,
        phi,
        episode,
        arm=ARM_SGD,
        derivative_mode=DERIVATIVE_EXACT,
        epsilon=1e-6,
    )
    assert check["max_abs_error"] < 1e-6
    assert check["max_abs_gradient"] > 0.0
    assert check["second_order_term_included"] is True
    assert check["declared_approximation"] is False
    assert check["fields"]["gain_log"]["max_abs_error"] < 1e-6


def test_truncated_sgd_derivative_is_declared_approximate(writer_fixture):
    _shapes, backbone, suite, phi = writer_fixture
    episode = suite.train[0]
    exact = finite_difference_gradient_check(
        backbone,
        phi,
        episode,
        arm=ARM_SGD,
        derivative_mode=DERIVATIVE_EXACT,
        epsilon=1e-6,
    )
    truncated = finite_difference_gradient_check(
        backbone,
        phi,
        episode,
        arm=ARM_SGD,
        derivative_mode=DERIVATIVE_TRUNCATED,
        epsilon=1e-6,
    )
    assert truncated["max_abs_error"] > 1e-4
    assert truncated["max_abs_error"] > exact["max_abs_error"] * 10.0
    assert truncated["second_order_term_included"] is False
    assert truncated["declared_approximation"] is True

    exact_result = outer_value_and_gradient(
        backbone, phi, episode,
        arm=ARM_SGD, derivative_mode=DERIVATIVE_EXACT,
    )
    truncated_result = outer_value_and_gradient(
        backbone, phi, episode,
        arm=ARM_SGD, derivative_mode=DERIVATIVE_TRUNCATED,
    )
    assert np.allclose(exact_result.value, truncated_result.value)
    assert np.allclose(
        exact_result.step.D_prime, truncated_result.step.D_prime
    )
    assert not np.allclose(
        exact_result.grad.W_ctx, truncated_result.grad.W_ctx
    )


def test_delta_meta_gradient_matches_finite_differences(writer_fixture):
    _shapes, backbone, suite, phi = writer_fixture
    episode = suite.train[0]
    check = finite_difference_gradient_check(
        backbone,
        phi,
        episode,
        arm=ARM_DELTA,
        derivative_mode=DERIVATIVE_EXACT,
        epsilon=1e-6,
    )
    assert check["max_abs_error"] < 1e-6
    assert check["second_order_term_included"] is False
    assert check["declared_approximation"] is False

    with pytest.raises(ValueError):
        outer_value_and_gradient(
            backbone, phi, episode,
            arm=ARM_DELTA, derivative_mode=DERIVATIVE_TRUNCATED,
        )


def test_outer_objective_uses_readout_and_future_query(writer_fixture):
    _shapes, backbone, suite, phi = writer_fixture
    episode = suite.train[0]
    result = outer_value_and_gradient(
        backbone,
        phi,
        episode,
        arm=ARM_SGD,
        inner_lr=0.7,
        replay_weight=0.5,
        cost_weight=0.1,
        cost_reference=0.5,
        derivative_mode=DERIVATIVE_EXACT,
    )
    losses = []
    for example in episode.future:
        h, _cache = backbone.feature(example.vector())
        logits = backbone.readout @ (result.step.D_prime @ h)
        losses.append(cross_entropy(softmax(logits), example.label))
    assert math.isclose(
        result.future_loss, float(np.mean(losses)), rel_tol=1e-12, abs_tol=1e-12
    )
    assert not np.allclose(result.step.D_prime, backbone.D0)
    assert result.cost_norm > 0.0
    expected_value = (
        result.future_loss
        + 0.5 * result.replay_loss
        + 0.1 * result.cost_norm / 0.5
    )
    assert math.isclose(result.value, expected_value, rel_tol=1e-12, abs_tol=1e-12)
    assert np.all(np.isfinite(result.grad.W_ctx))
    assert np.all(np.isfinite(result.grad.b_ctx))


def test_replay_weight_changes_the_retention_gradient(writer_fixture):
    _shapes, backbone, suite, phi = writer_fixture
    episode = suite.train[0]
    low = outer_value_and_gradient(
        backbone, phi, episode,
        arm=ARM_SGD, replay_weight=0.0, derivative_mode=DERIVATIVE_EXACT,
    )
    high = outer_value_and_gradient(
        backbone, phi, episode,
        arm=ARM_SGD, replay_weight=5.0, derivative_mode=DERIVATIVE_EXACT,
    )
    assert high.replay_loss > 0.0
    assert high.value > low.value
    assert not np.allclose(high.grad.W_ctx, low.grad.W_ctx)
    assert not np.allclose(high.grad.gain_log, low.grad.gain_log)


def test_cost_model_and_imperative_adamw_are_explicitly_separate(writer_fixture):
    shapes, backbone, suite, phi = writer_fixture
    episode = suite.train[0]
    assert ARM_ADAMW_REFERENCE not in FUNCTIONAL_ARMS

    sgd_ops = declared_update_ops(ARM_SGD, shapes)
    delta_ops = declared_update_ops(ARM_DELTA, shapes)
    adamw_ops = declared_update_ops(ARM_ADAMW_REFERENCE, shapes)
    assert adamw_ops > sgd_ops > delta_ops

    exact_meta = declared_meta_ops(ARM_SGD, DERIVATIVE_EXACT, shapes)
    truncated_meta = declared_meta_ops(ARM_SGD, DERIVATIVE_TRUNCATED, shapes)
    assert exact_meta > truncated_meta

    D_prime, cost, state = adamw_reference_step(
        backbone,
        phi,
        backbone.D0,
        episode.write,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        step_index=1,
    )
    assert D_prime.shape == backbone.D0.shape
    assert cost.arm == ARM_ADAMW_REFERENCE
    assert cost.meta_scalar_ops == 0
    assert cost.scalar_ops == adamw_ops
    assert state["step_index"] == 2

    context = np.concatenate(
        [episode.write.vector(), one_hot(episode.write.label, shapes.classes)]
    )
    z = np.tanh(phi.W_ctx @ context + phi.b_ctx)
    h, _cache = backbone.feature(z)
    probabilities = softmax(backbone.readout @ (backbone.D0 @ h))
    gradient = np.outer(
        backbone.readout.T
        @ (probabilities - one_hot(episode.write.label, shapes.classes)),
        h,
    )
    expected_m = 0.1 * gradient
    expected_v = 0.001 * gradient * gradient
    expected_update = gradient / (np.abs(gradient) + 1e-8) + 0.01 * backbone.D0
    assert np.allclose(state["m"], expected_m)
    assert np.allclose(state["v"], expected_v)
    assert np.allclose(D_prime, backbone.D0 - 0.01 * expected_update)

    with pytest.raises(ValueError):
        train_writer(
            backbone,
            suite.train,
            arm=ARM_ADAMW_REFERENCE,
            outer_steps=1,
        )


def test_writer_training_reports_separate_sections_and_useful_transfer(
    writer_fixture,
):
    shapes, backbone, suite, _phi = writer_fixture
    report = train_writer(
        backbone,
        suite.train,
        arm=ARM_SGD,
        outer_steps=400,
        outer_lr=0.05,
        inner_lr=0.7,
        replay_weight=5.0,
        cost_weight=0.1,
        cost_reference=0.5,
        derivative_mode=DERIVATIVE_EXACT,
        seed=5,
        evaluation_episodes=suite.development,
    )

    assert report["format"].endswith(".report.v1")
    assert report["evidence_class"] == EVIDENCE_CLASS
    assert report["native_training"] is False
    assert report["execution"]["status"] == "completed"
    assert report["execution"]["outer_steps_completed"] == 400
    assert report["execution"]["inner_steps"] == 400
    assert report["execution"]["evaluation_scope"] == "held_out_development"
    assert set(report["execution"]["evaluation_worlds"]).isdisjoint(
        report["execution"]["training_worlds"]
    )
    assert report["execution"]["native_capability_claim"] is False

    for section in (
        "execution",
        "parameter_updates",
        "transfer",
        "retention",
        "useful_learning",
        "cost",
    ):
        assert section in report

    assert report["transfer"]["future_ce_after"] < (
        report["transfer"]["future_ce_before"] - 0.1
    )
    assert report["transfer"]["future_ce_improvement"] > 0.1
    assert report["transfer"]["controls"]["wrong_world"] > (
        report["transfer"]["future_ce_after"]
    )
    assert report["transfer"]["controls"]["shuffled_label"] > (
        report["transfer"]["future_ce_after"]
    )
    assert report["useful_learning"]["passed"] is True
    assert report["useful_learning"]["parameter_change_alone_is_not_evidence"] is True
    assert report["retention"]["passed"] is True
    assert report["retention"]["replay_ce_change"] <= 0.05

    assert report["parameter_updates"]["online_d_updates"] == 400
    assert report["parameter_updates"]["online_d_parameter_count"] == (
        shapes.latent * shapes.hidden
    )
    assert report["parameter_updates"]["online_d_mean_changed_entries"] > 0
    assert report["parameter_updates"]["online_d_mean_delta_norm"] > 0.0
    assert report["parameter_updates"]["changed_scalars"] > 0
    assert report["parameter_updates"]["total_scalar_updates"] > 0
    assert report["parameter_updates"]["total_scalar_updates"] <= (
        report["parameter_updates"]["total_scalar_slots_visited"]
    )
    assert report["parameter_updates"]["total_scalar_slots_visited"] == (
        report["parameter_updates"]["scalar_parameter_count"] * 400
    )
    assert report["parameter_updates"]["mean_active_scalars_per_step"] > 0
    assert report["parameter_updates"]["phi_norm_after"] > 0.0

    assert report["cost"]["per_update"]["declared_scalar_ops"] == (
        declared_update_ops(ARM_SGD, shapes)
    )
    assert report["cost"]["declared_comparison"][ARM_ADAMW_REFERENCE] > (
        report["cost"]["declared_comparison"][ARM_SGD]
    )
    assert report["cost"]["declared_comparison"][ARM_SGD] > (
        report["cost"]["declared_comparison"][ARM_DELTA]
    )
    assert report["cost"]["training"]["declared_scalar_ops_total"] == (
        400 * declared_update_ops(ARM_SGD, shapes)
    )
    assert report["cost"]["training"]["mean_update_wall_seconds"] is not None

    assert report["clock"]["multi_clock"] is False
    assert report["clock"]["rsi"] is False
    assert report["clock"]["promotion"] is False
    assert report["clock"]["inner_steps_per_online_update"] == 1
    assert report["clock"]["online_d_updates"] == report["execution"]["inner_steps"]
    assert report["sealed"]["opened"] is False
    assert report["sealed"]["access"] == "closed"
    assert any("not native" in limit for limit in report["limits"])

    assert report["derivatives"]["exact_meta_gradient"] is True
    assert report["derivatives"]["second_order_softmax_hessian"] is True
    assert report["derivatives"]["imperative_adamw_differentiable"] is False


def test_truncated_training_is_labeled_as_an_approximation(writer_fixture):
    _shapes, backbone, suite, _phi = writer_fixture
    report = train_writer(
        backbone,
        suite.train,
        arm=ARM_SGD,
        outer_steps=2,
        outer_lr=0.01,
        replay_weight=1.0,
        derivative_mode=DERIVATIVE_TRUNCATED,
        seed=5,
        evaluation_episodes=suite.development,
    )
    assert report["execution"]["status"] == "completed"
    assert report["derivatives"]["first_order_truncated"] is True
    assert report["derivatives"]["exact_meta_gradient"] is False
    assert report["derivatives"]["second_order_softmax_hessian"] is False


def test_delta_training_reports_its_own_first_order_arm(writer_fixture):
    _shapes, backbone, suite, _phi = writer_fixture
    report = train_writer(
        backbone,
        suite.train,
        arm=ARM_DELTA,
        outer_steps=200,
        outer_lr=0.02,
        inner_lr=0.7,
        replay_weight=0.5,
        cost_weight=0.1,
        cost_reference=0.5,
        derivative_mode=DERIVATIVE_EXACT,
        seed=5,
        evaluation_episodes=suite.development,
    )
    assert report["execution"]["status"] == "completed"
    assert report["execution"]["inner_steps"] == 200
    assert report["derivatives"]["delta_first_order_exact"] is True
    assert report["derivatives"]["second_order_softmax_hessian"] is False
    assert report["transfer"]["future_ce_after"] < (
        report["transfer"]["future_ce_before"]
    )
    assert report["useful_learning"]["passed"] is True
    assert report["retention"]["passed"] is True


def test_conflicting_write_does_not_become_useful_learning(writer_fixture):
    shapes, backbone, suite, phi = writer_fixture
    source = suite.train[0]
    conflicting_future = tuple(
        replace(
            example,
            query_id=f"{example.query_id}-conflict",
            label=(example.label + 1) % shapes.classes,
        )
        for example in source.future
    )
    conflict = Episode(
        episode_id="conflict",
        world_id=source.world_id,
        family=source.family,
        split=source.split,
        write=source.write,
        future=conflicting_future,
        replay=source.replay,
    )
    result = evaluate_writer(
        backbone,
        phi,
        (conflict,),
        arm=ARM_SGD,
        inner_lr=0.7,
        replay_weight=5.0,
        cost_weight=0.1,
        cost_reference=0.5,
    )
    assert result["useful_learning"]["passed"] is False
    assert result["useful_learning"]["parameter_change_alone_is_not_evidence"] is True
    assert result["transfer"]["future_ce_after"] > (
        result["transfer"]["future_ce_before"]
    )
    assert result["useful_learning"]["claim"].startswith(
        "future-query transfer"
    )


def test_missing_replay_is_reported_without_faking_retention(writer_fixture):
    shapes, backbone, suite, phi = writer_fixture
    source = suite.train[0]
    no_replay = Episode(
        episode_id="no-replay",
        world_id=source.world_id,
        family=source.family,
        split=source.split,
        write=source.write,
        future=source.future,
        replay=(),
    )
    result = evaluate_writer(
        backbone,
        phi,
        (no_replay,),
        arm=ARM_SGD,
        inner_lr=0.7,
        replay_weight=1.0,
    )
    assert result["retention"]["replay_ce_before"] is None
    assert result["retention"]["replay_ce_after"] is None
    assert result["retention"]["passed"] is False
