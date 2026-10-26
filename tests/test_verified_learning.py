"""Focused CPU contracts for verified learning: transfer, controls, accounting.

No test loads a native model, runs MPS work, performs a native job or claims
verified learning/RSI.  Tiny synthetic worlds exercise the real objective,
gradient, control, audit, policy and resource paths.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

import replica_lab.verified_learning as verified
from replica_lab.architecture.contracts import CoreSpec

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/verified_learning_v1.json'
SPEC = CoreSpec(width=16, slots=2, steps=2, heads=2, task_features=4,
                ffn_hidden=32, control_hidden=32)


@pytest.fixture(scope='module')
def config():
    return verified.load_config(CONFIG_PATH)


@pytest.fixture(scope='module')
def worlds():
    return verified.generate_rule_worlds(
        seed=7, modulus=10,
        families_per_split={'train': 2, 'dev': 1, 'heldout': 2},
        demo_count=3, composition_count=1)


def _transfer_scores(values):
    values = dict(values)
    return {'kind': 'transfer', 'per_world': values,
            'accuracy': sum(values.values()) / len(values)}


def test_worlds_are_deterministic_disjoint_and_observation_bound(worlds):
    again = verified.generate_rule_worlds(
        seed=7, modulus=10,
        families_per_split={'train': 2, 'dev': 1, 'heldout': 2},
        demo_count=3, composition_count=1)
    assert worlds == again
    assert verified.observation_digest(worlds) == verified.observation_digest(again)
    families = [world.family for world in worlds]
    assert len(families) == len(set(families))
    for world in worlds:
        assert set(world.demo_inputs).isdisjoint(world.transfer_inputs)
        assert set(world.demo_inputs).isdisjoint(world.composition_inputs)
        for row in world.rows('demo'):
            assert row['source'] == 'demonstration'
            assert row['target'] == world.target(row['input'])
        for row in world.rows('transfer'):
            assert row['source'] == 'heldout_rule'
            assert row['input'] not in world.demo_inputs
        for row in world.rows('composition'):
            assert row['source'] == 'composition'
            assert row['target'] == world.target(world.target(row['input']))
    other = verified.generate_rule_worlds(
        seed=8, modulus=10,
        families_per_split={'train': 2, 'dev': 1, 'heldout': 2},
        demo_count=3, composition_count=1)
    assert [world.family for world in other] != families
    assert verified.observation_digest(other) != verified.observation_digest(worlds)
    with pytest.raises(ValueError, match='cannot supply disjoint'):
        verified.generate_rule_worlds(
            seed=7, modulus=4,
            families_per_split={'train': 1, 'dev': 1, 'heldout': 1},
            demo_count=3)


def test_config_matches_declared_default(config):
    assert config == verified.default_config()
    assert config['format'] == verified.CONFIG_FORMAT
    assert config['claims'] == {'verified_learning': False, 'rsi': False,
                                'native_capability': False}
    assert verified.ResourceLimits.from_dict(config['limits']).to_dict() == config['limits']


def test_protocol_and_preparation_are_deterministic(tmp_path, config):
    protocol = verified.build_protocol(
        verified.generate_rule_worlds(
            seed=7, modulus=10,
            families_per_split={'train': 2, 'dev': 1, 'heldout': 2},
            demo_count=3, composition_count=1), config)
    assert protocol == verified.build_protocol(
        verified.generate_rule_worlds(
            seed=7, modulus=10,
            families_per_split={'train': 2, 'dev': 1, 'heldout': 2},
            demo_count=3, composition_count=1), config)
    assert protocol['protocol_sha256'] == verified._sha_json(
        {key: value for key, value in protocol.items()
         if key != 'protocol_sha256'})
    assert protocol['objective']['training_rows'] == 'demonstrations_only'
    assert protocol['objective']['transfer_targets_used_in_training'] is False
    assert protocol['controls']['arms'] == list(verified.REQUIRED_CONTROL_ARMS)
    assert protocol['claims'] == config['claims']
    assert protocol['native_evaluation']['status'] == 'not_run'

    first = verified.prepare_verified_learning(
        tmp_path / 'first', config_path=CONFIG_PATH, config=config)
    second = verified.prepare_verified_learning(
        tmp_path / 'second', config_path=CONFIG_PATH, config=config)
    assert first == second
    result = verified.verify_prepared(tmp_path / 'first')
    assert result['protocol_sha256'] == first['protocol_sha256']
    assert result['manifest']['worlds_sha256'] == first['worlds_sha256']
    assert len(result['worlds']) == first['worlds']
    with pytest.raises(ValueError, match='fresh'):
        verified.prepare_verified_learning(
            tmp_path / 'first', config_path=CONFIG_PATH, config=config)


def test_response_objective_reaches_sensors_core_and_reader(worlds):
    model = verified.build_model(SPEC, core_impl='recurrent', steps=2, seed=3)
    encoded = verified.encode_rows(worlds[:1], 'demo')
    for parameter in model.parameters():
        parameter.grad = None
    output = model(encoded['token_ids'], encoded['task_features'])
    loss = verified.response_cross_entropy(output, encoded['targets'])
    loss.backward()
    assert torch.isfinite(loss)
    assert output.producer == 'base_decoder'
    assert output.decoder_frozen is True
    for name, parameter in model.named_parameters():
        if name.startswith(('sensor.', 'reader.', 'core.')):
            assert parameter.grad is not None, name
            assert float(parameter.grad.abs().sum()) > 0, name
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert all(parameter.grad is None for parameter in model.audit.parameters())
    assert all(parameter.requires_grad is False
               for parameter in model.decoder.parameters())

    fresh = verified.build_model(SPEC, core_impl='recurrent', steps=2, seed=3)
    ledger, summary = verified.adapt_adapter(
        fresh, encoded, learning_rate=0.05, update_steps=8,
        limits=verified.ResourceLimits.from_dict(verified.default_config()['limits']),
        seed=3)
    assert summary['objective'] == (
        'response_cross_entropy_through_frozen_base_decoder')
    assert summary['training_kind'] == 'demonstrations_only'
    assert summary['transfer_targets_used_in_training'] is False
    assert summary['final_loss'] < summary['initial_loss']
    assert summary['audit_trainable'] is False
    assert ledger.counters['optimizer_steps'] == 8
    assert len(ledger.update_norms) == 8
    assert float(ledger.update_norms[-1]) >= 0.0


def test_audit_head_is_detached_and_cannot_replace_the_objective(worlds):
    model = verified.build_model(SPEC, core_impl='recurrent', steps=2, seed=4)
    encoded = verified.encode_rows(worlds, 'demo')
    audit_output = model.audit_logits(encoded['token_ids'],
                                      encoded['task_features'])
    with pytest.raises(TypeError, match='audit logits'):
        verified.response_cross_entropy(audit_output, encoded['targets'])
    with pytest.raises(TypeError, match='base-decoder'):
        verified.response_cross_entropy(torch.zeros(2, 15), encoded['targets'][:2])
    for parameter in model.parameters():
        parameter.grad = None
    loss = verified.audit_objective(model, worlds[:2])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None
               for parameter in model.audit.parameters())
    for module in (model.sensor, model.reader, model.core):
        assert all(parameter.grad is None for parameter in module.parameters())
    # The audit input is detached at the head boundary, not by deleting the
    # core's response graph: response CE still reaches those modules above.


def test_control_plan_and_zero_one_two_recurrence_steps(worlds):
    plan = verified.build_control_plan(SPEC)
    counts = plan['parameter_counts']
    assert plan['arms'] == list(verified.REQUIRED_CONTROL_ARMS)
    assert counts['no_core'] == 0
    assert counts['recurrent_0'] == counts['recurrent_1'] == counts['recurrent_2']
    assert abs(plan['feedforward_relative_delta']) <= 0.05
    assert plan['feedforward_matched_within_5_percent'] is True
    assert plan['recurrence_parameter_matched'] is True
    assert counts['recurrent_0'] == plan['analytic_recurrent_parameters']

    encoded = verified.encode_rows(worlds[:1], 'demo')
    outputs = {}
    for name, core_impl, steps in (
            ('no_core', 'none', 0), ('feedforward', 'feedforward', 0),
            ('recurrent_0', 'recurrent', 0), ('recurrent_1', 'recurrent', 1),
            ('recurrent_2', 'recurrent', 2)):
        model = verified.build_model(SPEC, core_impl=core_impl, steps=steps,
                                     seed=5)
        output = model(encoded['token_ids'], encoded['task_features'])
        assert torch.isfinite(output.logits).all()
        actual = (0 if model.core is None
                  else verified.core_parameter_count(model.core))
        assert actual == counts[name]
        outputs[name] = output.logits
    assert not torch.equal(outputs['recurrent_1'], outputs['recurrent_2'])
    assert not torch.equal(outputs['no_core'], outputs['feedforward'])

    zero = verified.build_model(SPEC, core_impl='recurrent', steps=0, seed=5)
    sense = zero.sensor(encoded['token_ids'][:1])
    workspace = zero._workspace(sense, encoded['task_features'][:1])
    assert torch.equal(workspace, zero.core.slot_queries.unsqueeze(0))
    assert zero.steps == 0


def test_resource_accounting_is_exact_repeatable_and_enforced(worlds):
    limits = verified.ResourceLimits.from_dict(verified.default_config()['limits'])
    encoded = verified.encode_rows(worlds[:1], 'demo')
    totals, summaries = [], []
    for _ in range(2):
        model = verified.build_model(SPEC, core_impl='recurrent', steps=2,
                                     seed=6)
        ledger, summary = verified.adapt_adapter(
            model, encoded, learning_rate=0.05, update_steps=3,
            limits=limits, seed=6)
        totals.append(ledger.as_dict())
        summaries.append(summary)
    assert totals[0] == totals[1]
    assert summaries[0]['final_loss'] == summaries[1]['final_loss']
    assert totals[0]['counters'] == {
        'inference_batches': 1, 'inference_examples': 3,
        'inference_positions': 21, 'forward_batches': 3,
        'forward_examples': 9, 'forward_positions': 63,
        'backward_batches': 3, 'optimizer_steps': 3,
        'gradient_clip_calls': 3}
    assert totals[0]['limits'] == limits.to_dict()
    assert totals[0]['optimizer_state_elements'] == 0

    strict = replace(limits, max_optimizer_steps=1)
    model = verified.build_model(SPEC, core_impl='recurrent', steps=2, seed=6)
    with pytest.raises(verified.ResourceBudgetExceeded, match='optimizer_steps'):
        verified.adapt_adapter(model, encoded, learning_rate=0.05,
                               update_steps=2, limits=strict, seed=6)
    tight_delta = replace(limits, max_parameter_delta_norm=1e-9)
    model = verified.build_model(SPEC, core_impl='recurrent', steps=2, seed=6)
    with pytest.raises(ValueError, match='parameter delta norm'):
        verified.adapt_adapter(model, encoded, learning_rate=0.05,
                               update_steps=1, limits=tight_delta, seed=6)
    broken = dict(limits.to_dict())
    broken.pop('max_forward_batches')
    with pytest.raises(ValueError, match='resource limit fields differ'):
        verified.ResourceLimits.from_dict(broken)
    broken = dict(limits.to_dict(), unexpected=1)
    with pytest.raises(ValueError, match='resource limit fields differ'):
        verified.ResourceLimits.from_dict(broken)


def test_rule_transfer_gate_separates_recall_from_transfer():
    gate = {'min_rule_transfer_gain': 0.05, 'family_alpha': 0.05,
            'bootstrap_draws': 500}
    world_ids = [f'w{index}' for index in range(32)]
    low = {world_id: 0.2 for world_id in world_ids}
    high = {world_id: 0.8 for world_id in world_ids}
    recall = {'demo_accuracy': 1.0, 'transfer_accuracy': 0.0}
    accepted = verified.evaluate_rule_learning(
        _transfer_scores(low), _transfer_scores(high),
        _transfer_scores(low), _transfer_scores(low), gate=gate,
        recall_control=recall)
    assert accepted['decision'] == 'accepted'
    assert accepted['reason'] == 'heldout_transfer_gain'
    assert accepted['point_gain'] == pytest.approx(0.6)
    assert accepted['demonstration_recall_alone_sufficient'] is False
    assert accepted['transfer_targets_used_in_training'] is False
    assert all(value['low'] > 0 for value in accepted['comparisons'].values())

    recalled = verified.evaluate_rule_learning(
        _transfer_scores({world_id: 0.0 for world_id in world_ids}),
        _transfer_scores({world_id: 0.0 for world_id in world_ids}),
        _transfer_scores({world_id: 0.0 for world_id in world_ids}),
        _transfer_scores({world_id: 0.0 for world_id in world_ids}), gate=gate,
        recall_control=recall, post_demo={'accuracy': 1.0})
    assert recalled['decision'] == 'rejected'
    assert recalled['reason'] == 'demonstration_recall_without_rule_transfer'
    assert recalled['post_demonstration_recall'] == 1.0
    assert recalled['demonstration_recall_alone_sufficient'] is False

    # The recall control by itself does not condemn an unresolved transfer
    # result; a recall-only verdict needs the evaluated state's own scores.
    unresolved = verified.evaluate_rule_learning(
        _transfer_scores(low), _transfer_scores(low),
        _transfer_scores(low), _transfer_scores(low), gate=gate,
        recall_control=recall)
    assert unresolved['decision'] == 'inconclusive'
    assert unresolved['post_demonstration_recall'] is None


def test_demonstration_recall_control_scores_only_taught_pairs(worlds):
    recall = verified.demonstration_recall_control(worlds)
    assert recall['demo_accuracy'] == 1.0
    assert recall['parameters'] == 0
    assert recall['transfer_accuracy'] < 0.5
    assert recall['exact_training_answer_in_transfer'] is False


def test_policy_arms_match_observations_updates_and_resources(worlds):
    limits = verified.ResourceLimits.from_dict(verified.default_config()['limits'])
    heldout = [world for world in worlds if world.split == 'heldout']
    frozen = verified.LearningPolicy('fixture.frozen', 'frozen', 0.01, 'test')
    updated = verified.LearningPolicy('fixture.updated', 'updated', 0.10, 'test')
    result = verified.compare_policy_arms(
        frozen, updated, heldout, spec=SPEC, update_steps=1, limits=limits,
        seed=11, gate={'min_rule_transfer_gain': 0.05, 'family_alpha': 0.05,
                       'bootstrap_draws': 500})
    assert result['format'] == verified.POLICY_FORMAT
    assert result['all_matched'] is True
    assert result['matched'] == {
        'observations': True, 'update_counts': True,
        'resource_limits': True, 'initial_parameters': True}
    assert result['design']['observations'] == 'identical_example_digest_per_world'
    assert result['design']['update_counts'] == 'identical_optimizer_steps_per_world'
    assert result['design']['resource_limits'] == limits.to_dict()
    assert result['design']['eval_labels_used_for_policy'] is False
    assert result['decision'] in {'accepted', 'inconclusive'}
    for world in heldout:
        frozen_world = result['arms']['frozen']['worlds'][world.world_id]
        updated_world = result['arms']['updated']['worlds'][world.world_id]
        assert frozen_world['counters'] == updated_world['counters']
        assert frozen_world['observations_sha256'] == updated_world['observations_sha256']
        assert frozen_world['initial_parameters_sha256'] == updated_world['initial_parameters_sha256']
        assert frozen_world['update_steps'] == updated_world['update_steps'] == 1
        assert frozen_world['limits'] == updated_world['limits'] == limits.to_dict()

    selection = verified.select_learning_policy(
        [world for world in worlds if world.split == 'train'],
        spec=SPEC, learning_rate_grid=[0.01, 0.05], update_steps=1,
        limits=limits, frozen_learning_rate=0.05, seed=11)
    again = verified.select_learning_policy(
        [world for world in worlds if world.split == 'train'],
        spec=SPEC, learning_rate_grid=[0.01, 0.05], update_steps=1,
        limits=limits, frozen_learning_rate=0.05, seed=11)
    assert selection == again
    assert selection['selected_learning_rate'] in (0.01, 0.05)
    assert selection['meta_total_optimizer_steps'] == 4
    assert selection['policy_update_uses_eval_labels'] is False
    assert selection['frozen']['role'] == 'frozen'
    assert selection['updated']['role'] == 'updated'


def test_personality_fidelity_stays_separate_from_factual_accuracy(worlds):
    reference = [verified.style_trace(world.style) for world in worlds]
    identical = verified.personality_fidelity(reference, reference)
    assert identical['mean'] == pytest.approx(1.0)
    assert identical['factual_accuracy_used'] is False
    assert identical['memory_accuracy_used'] is False
    assert identical['authentic_personality_claim'] is False
    assert 'combined_score' not in identical

    world = worlds[0]
    correct = [row['target'] for row in world.rows('transfer')]
    wrong = [(value + 1) % world.modulus for value in correct]
    assert correct != wrong
    # Style traces are built from synthetic style markers only; changing fact
    # targets or answer accuracy cannot move them.
    unchanged = verified.personality_fidelity(
        [verified.style_trace(world.style)], [verified.style_trace(world.style)])
    assert unchanged == verified.personality_fidelity(
        [verified.style_trace(world.style)], [verified.style_trace(world.style)])
    assert sum(a != b for a, b in zip(correct, wrong)) == len(correct)

    changed = verified.personality_fidelity(
        [verified.style_trace((world.style[0],) * len(world.style))],
        [verified.style_trace(world.style)])
    if len(set(world.style)) > 1:
        assert changed['mean'] < 1.0
    with pytest.raises(ValueError, match='style values outside'):
        verified.style_trace((99,))


def test_fixture_report_is_procedure_only(config):
    report = verified.run_fixture(config)
    assert report['format'] == verified.FIXTURE_FORMAT
    assert report['mode'] == 'cpu_procedure_fixture'
    assert report['status'] == 'procedure_only'
    assert report['claims'] == config['claims']
    assert report['claims']['verified_learning'] is False
    assert report['claims']['rsi'] is False
    assert report['native_evaluation']['status'] == 'not_run'
    assert report['native_evaluation']['required'] is True
    assert report['audit']['detached_from_adapter'] is True
    assert report['audit']['adapter_trainable'] is False
    assert report['audit']['response_objective_rejected'] is True
    assert report['audit']['carries_hidden_answers'] is False
    assert report['audit']['substitutes_for_objective'] is False
    assert report['policy']['comparison']['all_matched'] is True
    assert report['controls']['feedforward_matched_within_5_percent'] is True
    assert report['controls']['recurrence_parameter_matched'] is True
    assert report['fake_training']['ledger_matches_real_counter_shape'] is True
    assert report['rule_learning']['gate']['transfer_targets_used_in_training'] is False
    assert report['fidelity']['combined_score'] is None
    assert report['fidelity']['memory_accuracy'] is None
    assert report['accounting']['exact_counters'] is True
    assert report['accounting']['wall_time_reported'] is False
    assert report['remaining_prerequisites']
    assert len(report['protocol_sha256']) == 64
    with pytest.raises(ValueError, match='CPU-only'):
        verified.run_fixture(config, device='mps')
