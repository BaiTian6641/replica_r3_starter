#!/usr/bin/env python3
"""Bounded, preregistered E3 qualification on the sealed MiniCPM5 source.

Implements the declared protocol: validated E2-parent migration, two sequential
modules-only alignment arms (recurrent and matched feed-forward control),
teacher-trajectory labels with real tool receipts, model-driven dev/held-out
episodes, the declared controls, optional success-head fitting from actual
outcomes, and one final extension export. No base checkpoint is saved.

The interrupted fresh-process task is a separate guarded CLI invocation; this
driver never uses the host to select an evaluation action.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

COMPONENT_VERSIONS = {'runtime': 'e3-runtime-v1', 'tools': 'fixture-workspace-v1',
                      'journal': 'exact-journal-v1', 'task_features': 'e3-layout-v1'}
CONTROL_MODES = ('core_disabled', 'workspace_zero', 'workspace_shuffled',
                 'ff_core', 'extra_inference')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def copy_case(case_dir: Path, destination: Path) -> Path:
    if destination.exists():
        raise ValueError(f'Working case already exists: {destination}')
    shutil.copytree(case_dir, destination)
    return destination


def component_binding(tools, *, subject, branch, purpose):
    from replica_lab.architecture.contracts import fingerprint
    return {'task_id': tools.case_id, 'subject': subject, 'branch': branch,
            'purpose': purpose, 'source_epoch': 0,
            'goal_digest': fingerprint(tools.goal),
            'component_versions': dict(COMPONENT_VERSIONS)}


def initial_state(tools, *, subject, branch, purpose, budget):
    from replica_lab.runtime.task_graph import (Budget, Constraint, ExactTaskState,
                                                SuccessCondition)
    return ExactTaskState(
        task_id=tools.case_id, subject=subject, branch=branch, purpose=purpose,
        source_epoch=0, goal=tools.goal,
        success_conditions=(SuccessCondition(tools.success_condition['condition_id'],
                                             tools.success_condition['description']),),
        constraints=tuple(Constraint(f'con-{index}', text)
                          for index, text in enumerate(tools.constraints)),
        subgoals=(), claims=(), evidence=(), obligations=(), actions=(), outcomes=(),
        notes=(), budget=budget,
        binding={**dict(COMPONENT_VERSIONS), 'case_format': 'replica.e3.fixture.v1'},
        allowed_tools=tuple(tools.allowed_tools), revision=0)


def policy_text(tools) -> str:
    return ('The evaluator tests are authoritative. Only files under workspace/ may be '
            'edited. Run the evaluator test to learn the required policy bound. '
            f'Case: {tools.case_id}.')


def admit_policy(ledger, tools, *, subject, branch, purpose):
    from replica_lab.memory_system import MemoryRecord
    record = MemoryRecord(f'{tools.case_id}_policy', subject, branch, policy_text(tools),
                          observed_at=0, event_at=0, purpose=purpose,
                          source_id='e3_fixture_policy', authenticity='unverified')
    ledger.admit(record, authorized_subject=subject, authorized_branch=branch)
    return record


class TeacherProposer:
    """Queued host-authored actions used only to build alignment labels/receipts."""

    def __init__(self, actions, tokenizer, *, max_prompt_tokens, max_generated_tokens):
        self.actions = tuple(actions)
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.max_generated_tokens = max_generated_tokens
        self.index = 0
        self.examples = []

    def __call__(self, request):
        import torch
        from replica_lab.architecture.loading import encode_prompt
        from replica_lab.runtime.loop import ProposalOutput
        if self.index >= len(self.actions):
            raise RuntimeError('Teacher action sequence exhausted before task completion')
        action = self.actions[self.index]
        self.index += 1
        raw = canonical(action)
        prompt_ids = encode_prompt(self.tokenizer, request.prompt_text, 'cpu')
        if prompt_ids.shape[1] > self.max_prompt_tokens:
            raise ValueError('Teacher prompt exceeds the preregistered prompt budget')
        target = self.tokenizer(raw, add_special_tokens=False)['input_ids']
        target = list(target) + [self.tokenizer.eos_token_id]
        if len(target) > self.max_generated_tokens:
            raise ValueError('Teacher action exceeds the preregistered label budget')
        self.examples.append({
            'prompt_text': request.prompt_text,
            'prompt_tokens': int(prompt_ids.shape[1]),
            'task_features': list(request.task_features),
            'task_digest': request.task_digest,
            'revision': request.revision,
            'target_text': raw,
            'target_token_ids': target,
            'step': self.index,
        })
        return ProposalOutput(raw_text=raw, prompt_tokens=int(prompt_ids.shape[1]),
                              generated_token_ids=tuple(target), generated_tokens=len(target),
                              frame_digest=sha256_bytes(raw.encode()), workspace=None,
                              workspace_stamp=None, wall_seconds=0.0,
                              notes=('teacher_label',))


def teacher_actions(tools, repair_tool='write_file'):
    repair = tools.reference_repair
    if not repair or repair.get('path') != tools.workspace_entry:
        raise ValueError('Fixture reference repair must target the workspace entry')
    if repair_tool == 'set_constant':
        variant = tools.variant
        change = {'kind': 'invoke_tool', 'tool': 'set_constant',
                  'arguments': {'name': variant['constant'],
                                'value': variant['expected_value']}}
    elif repair_tool == 'write_file':
        change = {'kind': 'invoke_tool', 'tool': 'write_file',
                  'arguments': {'path': repair['path'], 'content': repair['content']}}
    else:
        raise ValueError(f'Unsupported repair tool: {repair_tool}')
    return [
        {'kind': 'read', 'path': tools.workspace_entry},
        {'kind': 'invoke_tool', 'tool': 'run_test', 'arguments': {}},
        change,
        {'kind': 'invoke_tool', 'tool': 'run_test', 'arguments': {}},
        {'kind': 'stop'},
    ]


def run_teacher(case_dir, work_dir, *, tokenizer, subject, branch, purpose,
                max_prompt_tokens, max_generated_tokens, repair_tool='write_file'):
    from replica_lab.runtime.journal import TaskJournal
    from replica_lab.runtime.loop import TaskRuntime
    from replica_lab.runtime.tools import FixtureWorkspace
    from replica_lab.runtime.task_graph import Budget
    copy_case(case_dir, work_dir)
    tools = FixtureWorkspace(work_dir)
    budget = Budget(6, 4, 6 * max_prompt_tokens, 6 * max_generated_tokens)
    journal = TaskJournal.open(work_dir / 'teacher_journal.sqlite',
                               component_binding(tools, subject=subject, branch=branch,
                                                 purpose=purpose))
    try:
        runtime = TaskRuntime(tools, journal, mode='full',
                              max_prompt_tokens=max_prompt_tokens,
                              generated_token_budget=max_generated_tokens, budget=budget)
        runtime.initialize(initial_state(tools, subject=subject, branch=branch,
                                         purpose=purpose, budget=budget))
        proposer = TeacherProposer(teacher_actions(tools, repair_tool), tokenizer,
                                   max_prompt_tokens=max_prompt_tokens,
                                   max_generated_tokens=max_generated_tokens)
        summary = runtime.run(proposer, max_decisions=6)
        examples = list(proposer.examples)
        final_state = runtime.state
        return {'summary': summary, 'examples': examples,
                'final_state': final_state.to_dict(),
                'teacher_actions_used': proposer.index}
    finally:
        journal.close()


def tokenize_prompt(tokenizer, text, device):
    from replica_lab.architecture.loading import encode_prompt
    return encode_prompt(tokenizer, text, device)


def build_action_batch(tokenizer, example, device):
    import torch
    prompt_ids = tokenize_prompt(tokenizer, example['prompt_text'], device)
    target = list(example['target_token_ids'])
    target_ids = torch.tensor([target], dtype=torch.long, device=device)
    action_ids = torch.cat((prompt_ids, target_ids), dim=1)
    mask = torch.ones_like(action_ids, dtype=torch.bool)
    labels = action_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100
    return action_ids, mask, labels, int(prompt_ids.shape[1])


def snapshot_new_parameters(model):
    return {name: parameter.detach().clone()
            for name, parameter in model.new_named_parameters()}


def restore_new_parameters(model, snapshot):
    import torch
    current = dict(model.new_named_parameters())
    if set(current) != set(snapshot):
        raise ValueError('Parameter snapshot inventory differs from the model')
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(snapshot[name].to(parameter.device))


def selection_for(session, scope, goal):
    from dataclasses import asdict
    selection = session.index.retrieve(goal, **asdict(scope), limit=1)
    session.index.verify_selection(selection, expected_scope=scope)
    if not selection.records:
        raise ValueError('E3 policy record was not retrievable')
    return selection


def run_alignment_arm(model, session, tokenizer, examples, *, core_impl, protocol,
                      scope_for_case, selection_for_case, mark):
    import torch
    from replica_lab.optimizers import build_optimizer
    from replica_lab.runtime.loop import Budget  # noqa: F401  (import-time coherence)
    alignment = protocol['alignment']
    model.train()
    optimizer = build_optimizer(model, 'muon8bit', lr_base=alignment['lr_base'],
                                lr_new=alignment['lr_new'],
                                lr_muon_base=alignment['lr_muon_base'],
                                lr_muon_new=alignment['lr_muon_new'],
                                weight_decay=alignment['weight_decay'])
    if optimizer is None:
        raise RuntimeError('Alignment requires trainable new parameters')
    audit = {'optimizer': optimizer.audit(), 'updates': [], 'core_impl': core_impl}
    updates = int(alignment['updates'])
    batch_size = int(alignment['examples_per_update'])
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for offset in range(batch_size):
            example = examples[(2 * (update - 1) + offset) % len(examples)]
            case_scope = scope_for_case(example['case_id'])
            selection = selection_for_case(example['case_id'])
            action_ids, mask, labels, prompt_tokens = build_action_batch(
                tokenizer, example, 'mps')
            with session.training_request(
                    prompt_ids=action_ids[:, :prompt_tokens],
                    prompt_mask=mask[:, :prompt_tokens], query=example['goal'],
                    scope=case_scope, task=example['task_graph'],
                    selection=selection, task_features=example['task_features'],
                    task_digest=example['task_digest'], revision=example['revision'],
                    previous_core=None, core_impl=core_impl,
                    enabled_branches=('memory', 'workspace'),
                    train_core=True, train_sense=True) as request:
                output = session.forward(request, action_ids, labels=labels,
                                         attention_mask=mask)
                loss = output.loss / batch_size
                total += float(loss.detach())
                session.backward(request, loss)
            del output, loss, action_ids, mask, labels
        norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            alignment['gradient_clip'], error_if_nonfinite=True))
        ratio = min(1.0, update / alignment['warmup_updates'])
        for group in optimizer.param_groups:
            group['lr'] = (alignment['lr_muon_new']
                           if group['algorithm'] == 'muon8bit' else alignment['lr_new']) * ratio
        if update == 2:
            # Capture the actual accumulated gradients before step/zero_grad.
            gradients = model.gradient_audit()
            audit['gradient_audit_update2'] = gradients
            audit['core_gradient_update2'] = any(
                name.startswith(('core.', 'ff_core.', 'task_head.'))
                for name in gradients['new']['nonzero'])
            audit['sensor_gradient_update2'] = any(
                name.startswith('sense_ports.') for name in gradients['new']['nonzero'])
            audit['workspace_reader_gradient_update2'] = any(
                name.startswith('workspace_ports.') for name in gradients['new']['nonzero'])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        audit['updates'].append({'update': update, 'mean_loss': total,
                                 'gradient_norm': norm})
        mark(f'alignment_{core_impl}_update', update=update, loss=total)
    del optimizer
    model.zero_grad(set_to_none=True)
    model.eval()
    gc.collect()
    torch.mps.empty_cache()
    return audit


def proposer_for_mode(session, tokenizer, tools, *, mode, scope, subject,
                      zero_previous=False):
    from replica_lab.runtime.loop import SessionProposer
    common = dict(task_id=tools.case_id, goal=tools.goal, scope=scope, device='mps')
    if mode == 'core_disabled':
        return SessionProposer(session, tokenizer, core_impl='none',
                               enabled_branches=('memory',), **common)
    if mode == 'workspace_zero':
        return SessionProposer(session, tokenizer, zero_previous_workspace=True, **common)
    if mode == 'ff_core':
        return SessionProposer(session, tokenizer, core_impl='feedforward', **common)
    if mode == 'extra_inference':
        return SessionProposer(session, tokenizer, enabled_branches=('memory',),
                               extra_inference_pass=True, **common)
    if zero_previous:
        return SessionProposer(session, tokenizer, zero_previous_workspace=True,
                               **common)
    return SessionProposer(session, tokenizer, **common)


def plan_change_after_failure(trace):
    """True when the action after an observed failing test/write differs in kind."""
    failed = False
    for item in trace:
        outcome = item.get('outcome') or {}
        result = outcome.get('result') or {}
        receipt = result.get('receipt') if isinstance(result, dict) else None
        exit_code = (receipt or {}).get('exit_code') if isinstance(receipt, dict) else None
        failed_here = (outcome.get('status') in ('failed',)
                       or (exit_code is not None and exit_code != 0))
        if failed and item.get('action') is not None:
            kind = item['action'].get('kind')
            tool = item['action'].get('tool')
            return kind in ('invoke_tool', 'revise', 'read') and tool != 'run_test'
        failed = failed or failed_here
    return False


def run_episode(case_dir, work_dir, *, session, tokenizer, mode, subject, purpose,
                budget, max_prompt_tokens, generated_token_budget, shuffled=None,
                mark=None, zero_previous=False):
    from replica_lab.memory_system import EligibilityScope
    from replica_lab.runtime.journal import TaskJournal
    from replica_lab.runtime.loop import TaskRuntime
    from replica_lab.runtime.tools import FixtureWorkspace
    from replica_lab.runtime.workspace_store import WorkspaceStore
    copy_case(case_dir, work_dir)
    tools = FixtureWorkspace(work_dir)
    branch = tools.case_id
    scope = EligibilityScope(subject, branch, purpose, 0, 'assistant_for_source', None)
    journal = TaskJournal.open(work_dir / 'journal.sqlite',
                               component_binding(tools, subject=subject, branch=branch,
                                                 purpose=purpose))
    started = time.monotonic()
    try:
        store = WorkspaceStore(work_dir / 'workspace_state')
        runtime = TaskRuntime(tools, journal, workspace_store=store, mode=mode,
                              max_prompt_tokens=max_prompt_tokens,
                              generated_token_budget=generated_token_budget,
                              budget=budget, shuffled_workspace=shuffled)
        runtime.initialize(initial_state(tools, subject=subject, branch=branch,
                                         purpose=purpose, budget=budget))
        proposer = proposer_for_mode(session, tokenizer, tools, mode=mode, scope=scope,
                                     subject=subject, zero_previous=zero_previous)
        summary = runtime.run(proposer, max_decisions=budget.max_decisions)
        final = runtime.state
        stored = store.load()
        report = {
            'case_id': tools.case_id, 'mode': mode, 'role': tools.role,
            'summary': summary, 'success': summary['success'],
            'final_revision': final.revision, 'final_digest': final.digest,
            'satisfied_conditions': list(final.satisfied_conditions(runtime.verifier)),
            'workspace_stored_revision': None if stored is None else stored.revision,
            'workspace_stored_matches': bool(stored is not None
                                             and stored.task_digest == final.digest),
            'last_workspace_discard': runtime.last_workspace_discard,
            'workspace_history_disabled': bool(zero_previous),
            'journal': {'path': str((work_dir / 'journal.sqlite').resolve()),
                        'head': journal.head, 'events': len(journal.events())},
            'actions': [a.to_dict() for a in final.actions],
            'outcomes': [o.to_dict() for o in final.outcomes],
            'workspace_hashes': tools.workspace_hashes(),
            'evaluator_pass_after_run': tools.verify_success().get('passed'),
            'wall_seconds': time.monotonic() - started,
            'decision_count': len(summary['trace']),
            'prompt_tokens': final.budget.used_prompt_tokens,
            'generated_tokens': final.budget.used_generated_tokens,
            'tool_calls': final.budget.used_tool_calls,
            'malformed_actions': sum(
                1 for item in summary['trace']
                if item.get('action') is None and item.get('outcome', {}).get('status') == 'denied'),
            'false_success_claims': sum(
                1 for item in summary['trace']
                if item.get('success_claimed') and not item.get('success_condition_met')),
            'plan_changed_after_failure': plan_change_after_failure(summary['trace']),
            'raw_trace': summary['trace'],
            'feature_vector': summary['feature_vector'],
        }
        if mark:
            mark('episode_complete', case=tools.case_id, mode=mode, success=summary['success'])
        return report, work_dir / 'workspace_state'
    finally:
        journal.close()


def train_success_head(model, episodes, protocol, mark):
    import torch
    config = protocol['success_head']
    labels = [int(bool(item['success'])) for item in episodes]
    feature_vectors = [item['feature_vector'] for item in episodes]
    report = {'labels': len(labels), 'positives': int(sum(labels)),
              'negatives': int(len(labels) - sum(labels)),
              'conditions': [{'case_id': item['case_id'], 'mode': item['mode'],
                              'success': item['success']} for item in episodes],
              'trained': False,
              'scope': 'Actual outcomes from this bounded exposed episode set; no held-out calibration role.'}
    if len(labels) < int(config['train_if_labels_at_least']):
        report['reason'] = 'insufficient_outcome_labels'
        return report
    if config.get('require_both_classes') and len(set(labels)) < 2:
        report['reason'] = 'single_outcome_class'
        return report
    width = model.manifest.core.width
    features = torch.tensor(feature_vectors, dtype=torch.float32, device='mps')
    task_part, workspace_part = features[:, :24], features[:, 24:]
    workspace = workspace_part.reshape(-1, model.manifest.core.slots, width)
    target = torch.tensor(labels, dtype=torch.float32, device='mps')
    head = model.task_head
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    count = len(labels)
    losses = []
    for step in range(1, int(config['steps']) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = target.new_zeros(())
        for index in range(count):
            logits = head(task_part[index:index + 1], workspace[index:index + 1])
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target[index:index + 1])
        loss = loss / count
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0:
            mark('success_head_step', step=step, loss=losses[-1])
    probabilities = []
    with torch.no_grad():
        for index in range(count):
            probabilities.append(float(head(task_part[index:index + 1],
                                            workspace[index:index + 1]).sigmoid()[0]))
    probabilities = torch.tensor(probabilities, device='mps')
    brier = float(((probabilities - target) ** 2).mean())
    head.eval()
    report.update(trained=True, initial_loss=losses[0], final_loss=losses[-1],
                  brier=brier, steps=len(losses),
                  probabilities=[float(v) for v in probabilities.tolist()],
                  warning='Not calibrated: no disjoint calibration role; Brier is on the same exposed labels.')
    return report


def extension_parent_regression(model, extension_dir):
    from safetensors.torch import load_file
    artifact = load_file(str(Path(extension_dir) / 'extensions.safetensors'))
    current = dict(model._parent_named_parameters())
    if set(artifact) != set(current):
        return {'passed': False, 'reason': 'inventory_mismatch'}
    identical = all(artifact[name].shape == current[name].shape
                    and torch_equal(artifact[name], current[name].detach().cpu())
                    for name in artifact)
    return {'passed': identical, 'tensors': len(artifact),
            'parent_weights_sha256': hashlib.sha256(
                (Path(extension_dir) / 'extensions.safetensors').read_bytes()).hexdigest()}


def torch_equal(left, right):
    import torch
    return bool(torch.equal(left, right))


def execute(args, profiler, report):
    import torch
    from dataclasses import asdict as dataclass_asdict
    from replica_lab.architecture.contracts import CoreSpec
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import EligibilityScope, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.runtime.task_graph import Budget
    from replica_lab.runtime.workspace_store import WorkspaceStore

    protocol = json.loads(args.protocol.read_text())
    if protocol['format'] != 'replica.e3.qualification.v1':
        raise ValueError('Unknown E3 qualification protocol')
    expected = {'device': 'mps', 'dtype': 'float32', 'native_cache': False,
                'native_cpu_fallback': False, 'sampling': 'greedy', 'thinking': False}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError('Protocol hardware/decoding settings differ from this driver')
    if protocol['source_revision'] != 'f97400052a43d642bbc6e9975e2397e3ae6a6b52':
        raise ValueError('Protocol source revision differs from the sealed source')
    core_spec = CoreSpec(**protocol['core'])
    alignment = protocol['alignment']
    if alignment['regime'] != 'modules_only' or alignment['optimizer'] != 'muon8bit':
        raise ValueError('Protocol training regime/optimizer mismatch')
    if int(alignment['updates']) * int(alignment['examples_per_update']) < 1:
        raise ValueError('Protocol alignment budget is empty')
    evaluation = protocol['evaluation']
    budgets = protocol['budgets']
    repair_tool = protocol.get('repair_tool', 'write_file')
    zero_previous = protocol.get('workspace_history') == 'disabled'
    fixture_manifest = Path(protocol['fixtures']['manifest'])
    if sha256_bytes(fixture_manifest.read_bytes()) != protocol['fixtures']['manifest_sha256']:
        raise ValueError('Fixture manifest differs from the preregistered hash')

    torch.manual_seed(protocol['alignment']['seed'])
    model, tokenizer, source_lock = load_local_mps(
        args.model_path, args.model_lock, profiler=profiler, core=core_spec)
    if source_lock['revision'] != protocol['source_revision']:
        raise ValueError('Loaded source revision differs from protocol')
    report.update(manifest=model.manifest.to_dict(), manifest_digest=model.manifest.digest,
                  source_lock=source_lock, parameters=model.parameter_audit(), checks={},
                  core_spec=dataclass_asdict(core_spec),
                  packages={name: importlib.metadata.version(name)
                            for name in ('torch', 'transformers', 'safetensors')})

    # Null parity belongs to the freshly initialized core adapter: the E2 parent
    # extension is intentionally nonzero, so parity is measured before loading it.
    from replica_lab.architecture.contracts import EvidencePin, MemoryRead
    parity_prompt = tokenizer('E3 core parity probe: configuration policy test.',
                              add_special_tokens=False)['input_ids'][:64]
    parity_ids = torch.tensor([parity_prompt], dtype=torch.long, device='mps')
    parity_mask = torch.ones_like(parity_ids, dtype=torch.bool)
    empty_slots = model.base.get_input_embeddings().weight.new_zeros(
        1, 0, model.ports.slots_per_unit, model.ports.memory_width)
    empty_valid = torch.zeros(1, 0, model.ports.slots_per_unit, dtype=torch.bool,
                              device='mps')
    parity_memory = MemoryRead((), empty_slots, empty_valid, model.memory_producer_key())
    parity_sense = model.sense(parity_ids, parity_mask)
    parity_core = model.build_core_state(
        parity_sense, parity_memory, torch.zeros(1, core_spec.task_features, device='mps'),
        task_digest='parity-probe', revision=0)
    parity_evidence = EvidencePin('parity-ledger', 0, 'probe', 'probe', 'parity', 0,
                                  'assistant_for_source', None, ())
    parity_frame = model.make_frame('parity-decision', parity_evidence, parity_memory,
                                    parity_sense, core=parity_core,
                                    enabled_branches=('memory', 'workspace'))
    with torch.no_grad():
        native = model.base(input_ids=parity_ids, attention_mask=parity_mask,
                            use_cache=False).logits
        augmented = model.forward_with_frame(parity_ids, parity_frame).logits
        difference = float((native - augmented).abs().max())
    report['checks']['initial_null_native_parity'] = {
        'passed': difference <= 1e-5, 'max_absolute_difference': difference}
    if difference > 1e-5:
        raise AssertionError('initial_null_native_parity')
    del native, augmented, parity_frame, parity_core, parity_sense, parity_memory, empty_slots

    parent_extension = Path(args.parent_extension)
    migration = model.load_extension(parent_extension, accept_parent_v1=True)
    report['parent_migration'] = migration.get('migration') or {'accepted_parent_v1': False}
    report['parent_extension_receipt'] = {
        'format': migration['format'], 'weights_sha256': migration['weights_sha256'],
        'manifest_digest': migration['manifest_digest']}
    report['checks']['E2_parent_migration'] = {
        'passed': bool(migration.get('migration', {}).get('accepted_parent_v1')),
        'copied_tensors': len(migration.get('migration', {}).get('copied', [])),
        'fresh_init_tensors': len(migration.get('migration', {}).get('fresh_init', []))}
    report['checks']['E2_parent_regression'] = extension_parent_regression(model, parent_extension)
    parent_snapshot = snapshot_new_parameters(model)

    subject, purpose = 'e3_fixture', 'bounded_repair'
    ledger = RecordLedger(args.output / 'memory_ledger.sqlite')
    session = MemorySession(model, tokenizer, ledger, cache_dir=args.output / 'record_latents')
    scopes, selections, tools_by_case, cases = {}, {}, {}, {}
    for role in ('train', 'dev', 'heldout', 'interrupted'):
        for case_id in protocol['fixtures'][role]:
            case_dir = Path(protocol['fixtures']['root']) / case_id
            from replica_lab.runtime.tools import FixtureWorkspace
            tools = FixtureWorkspace(case_dir)
            if tools.role != role:
                raise ValueError(f'Fixture {case_id} role differs from protocol')
            cases[case_id] = {'dir': case_dir, 'role': role, 'tools': tools}
            tools_by_case[case_id] = tools
            scope = EligibilityScope(subject, case_id, purpose, 0, 'assistant_for_source', None)
            scopes[case_id] = scope
            admit_policy(ledger, tools, subject=subject, branch=case_id, purpose=purpose)
    for case_id, scope in scopes.items():
        selections[case_id] = selection_for(session, scope, tools_by_case[case_id].goal)
    report['memory'] = {'policy_records': len(scopes),
                        'selected_per_case': {k: [r.record_id for r in v.records]
                                              for k, v in selections.items()}}

    def mark(name, **fields):
        profiler.mark(name, **fields)
        dump(args.output / 'progress.json', report)
        print(json.dumps({'phase': name, **fields}), flush=True)

    def require(name, value, **details):
        report['checks'][name] = {'passed': bool(value), **details}
        if not value:
            raise AssertionError(name)

    # ---- independent branch influence after parent migration --------------
    prompt_ids = parity_ids
    prompt_mask = parity_mask
    probe_scope = scopes['train_retry_01']
    probe_task = tools_by_case['train_retry_01']
    from replica_lab.architecture.contracts import TaskGraph
    features = torch.zeros(1, 24, device='mps')
    initial_ports = {name: parameter.detach().clone()
                     for name, parameter in model.memory_ports.named_parameters()}
    initial_workspace = {name: parameter.detach().clone()
                         for name, parameter in model.workspace_ports.named_parameters()}

    def logits_for(branches):
        candidate = session.prepare(prompt_ids=prompt_ids, prompt_mask=prompt_mask,
                                    query=probe_task.goal, scope=probe_scope,
                                    task=TaskGraph('probe', probe_task.goal),
                                    selection=selections['train_retry_01'],
                                    task_features=features, task_digest='probe-digest',
                                    revision=0, enabled_branches=branches)
        try:
            with torch.no_grad():
                return session.forward(candidate, prompt_ids).logits[:, -1].clone()
        finally:
            candidate.release()

    baseline = logits_for(('memory',))
    with torch.no_grad():
        for port in model.workspace_ports.values():
            port.output_projection.weight.normal_(0.0, 0.002)
    workspace_active = logits_for(('memory', 'workspace'))
    workspace_delta = float((workspace_active - baseline).abs().max())
    with torch.no_grad():
        for name, value in initial_workspace.items():
            dict(model.workspace_ports.named_parameters())[name].copy_(value)
        for port in model.memory_ports.values():
            port.output_projection.weight.normal_(0.0, 0.002)
    memory_active = logits_for(('memory',))
    memory_delta = float((memory_active - baseline).abs().max())
    require('independent_workspace_reader_influence', workspace_delta > 1e-6,
            max_logit_difference=workspace_delta)
    require('independent_memory_reader_influence', memory_delta > 1e-6,
            max_logit_difference=memory_delta)
    with torch.no_grad():
        for name, value in initial_ports.items():
            dict(model.memory_ports.named_parameters())[name].copy_(value)
        for name, value in initial_workspace.items():
            dict(model.workspace_ports.named_parameters())[name].copy_(value)
    del baseline, workspace_active, memory_active
    torch.mps.empty_cache()

    # ---- teacher trajectories (labels and real receipts only) ------------
    teacher_examples = []
    teacher_reports = []
    for case_id in protocol['fixtures']['train']:
        tools = tools_by_case[case_id]
        work = args.output / 'teacher' / case_id
        result = run_teacher(cases[case_id]['dir'], work, tokenizer=tokenizer,
                             subject=subject, branch=case_id, purpose=purpose,
                             max_prompt_tokens=int(alignment['max_prompt_tokens']),
                             max_generated_tokens=int(alignment['max_label_tokens']),
                             repair_tool=repair_tool)
        if not result['summary']['success']:
            raise AssertionError(f'Teacher trajectory failed for {case_id}')
        for example in result['examples']:
            example['case_id'] = case_id
            example['goal'] = tools.goal
            example['task_graph'] = TaskGraph(task_id=case_id, goal=tools.goal,
                                              state_digest=example['task_digest'],
                                              revision=example['revision'])
            teacher_examples.append(example)
        teacher_reports.append({'case_id': case_id, 'success': result['summary']['success'],
                                'steps': len(result['examples']),
                                'summary': result['summary']})
        mark('teacher_trajectory', case=case_id, steps=len(result['examples']))
    report['teacher_trajectories'] = teacher_reports
    report['alignment_examples'] = {
        'count': len(teacher_examples),
        'steps': [{'case_id': e['case_id'], 'step': e['step'],
                   'prompt_tokens': e['prompt_tokens'],
                   'target_tokens': len(e['target_token_ids']),
                   'target_text': e['target_text']} for e in teacher_examples]}
    if max(len(e['target_token_ids']) for e in teacher_examples) > int(alignment['max_label_tokens']):
        raise ValueError('Teacher label exceeds the preregistered budget')

    # ---- recurrent and matched feed-forward arms -------------------------
    report['alignment'] = {'recurrent': None, 'feedforward_control': None}
    recurrent_audit = run_alignment_arm(
        model, session, tokenizer, teacher_examples, core_impl='recurrent',
        protocol=protocol, scope_for_case=lambda case_id: scopes[case_id],
        selection_for_case=lambda case_id: selections[case_id], mark=mark)
    report['alignment']['recurrent'] = recurrent_audit
    recurrent_snapshot = snapshot_new_parameters(model)
    require('recurrent_core_gradient_paths',
            recurrent_audit.get('core_gradient_update2')
            and recurrent_audit.get('sensor_gradient_update2')
            and recurrent_audit.get('workspace_reader_gradient_update2'),
            gradient_audit=recurrent_audit.get('gradient_audit_update2'))
    restore_new_parameters(model, parent_snapshot)
    ff_audit = run_alignment_arm(
        model, session, tokenizer, teacher_examples, core_impl='feedforward',
        protocol=protocol, scope_for_case=lambda case_id: scopes[case_id],
        selection_for_case=lambda case_id: selections[case_id], mark=mark)
    report['alignment']['feedforward_control'] = ff_audit
    ff_snapshot = snapshot_new_parameters(model)
    require('feedforward_core_gradient_paths',
            ff_audit.get('core_gradient_update2')
            and ff_audit.get('sensor_gradient_update2')
            and ff_audit.get('workspace_reader_gradient_update2'),
            gradient_audit=ff_audit.get('gradient_audit_update2'))
    restore_new_parameters(model, recurrent_snapshot)
    model.eval()
    gc.collect()
    torch.mps.empty_cache()
    mark('alignment_complete')

    # ---- model-driven dev/held-out episodes and controls -----------------
    episode_budget = Budget(int(budgets['decisions_per_episode']),
                            int(budgets['tool_calls_per_episode']),
                            int(budgets['decisions_per_episode']) * int(budgets['prompt_tokens_per_decision']),
                            int(budgets['decisions_per_episode']) * int(budgets['generated_tokens_per_decision']))
    episodes = []
    dev_store = None
    for case_id in evaluation['full_episodes']:
        work = args.output / 'episodes' / f'{case_id}_full'
        result, store_path = run_episode(
            cases[case_id]['dir'], work, session=session, tokenizer=tokenizer, mode='full',
            subject=subject, purpose=purpose, budget=episode_budget,
            max_prompt_tokens=int(budgets['prompt_tokens_per_decision']),
            generated_token_budget=int(budgets['generated_tokens_per_decision']), mark=mark,
            zero_previous=zero_previous)
        episodes.append(result)
        if case_id == evaluation['control_case']:
            dev_store = store_path
    control_reports = []
    shuffled_core = WorkspaceStore(dev_store).load() if dev_store is not None else None
    if shuffled_core is None:
        raise AssertionError('Control workspace is missing')
    for mode in evaluation['control_modes']:
        work = args.output / 'episodes' / f"{evaluation['control_case']}_{mode}"
        restore_new_parameters(model, ff_snapshot if mode == 'ff_core' else recurrent_snapshot)
        model.eval()
        result, _ = run_episode(
            cases[evaluation['control_case']]['dir'], work, session=session,
            tokenizer=tokenizer, mode=mode, subject=subject, purpose=purpose,
            budget=episode_budget,
            max_prompt_tokens=int(budgets['prompt_tokens_per_decision']),
            generated_token_budget=int(budgets['generated_tokens_per_decision']),
            shuffled=shuffled_core if mode == 'workspace_shuffled' else None, mark=mark,
            zero_previous=zero_previous)
        control_reports.append(result)
    restore_new_parameters(model, recurrent_snapshot)
    model.eval()
    episodes.extend(control_reports)
    report['episodes'] = episodes
    report['evaluation'] = {
        'full_success': {item['case_id']: item['success'] for item in episodes
                         if item['mode'] == 'full'},
        'control_success': {item['mode']: item['success'] for item in control_reports},
        'heldout_success': {item['case_id']: item['success'] for item in episodes
                            if item['role'] == 'heldout'},
        'dev_success': {item['case_id']: item['success'] for item in episodes
                        if item['role'] == 'dev'},
        'plan_changes_after_failure': {f"{i['case_id']}:{i['mode']}":
                                       i['plan_changed_after_failure'] for i in episodes},
        'malformed_actions': sum(i['malformed_actions'] for i in episodes),
        'false_success_claims': sum(i['false_success_claims'] for i in episodes),
        'decision_total': sum(i['decision_count'] for i in episodes),
        'prompt_tokens_total': sum(i['prompt_tokens'] for i in episodes),
        'generated_tokens_total': sum(i['generated_tokens'] for i in episodes),
        'tool_calls_total': sum(i['tool_calls'] for i in episodes),
        'wall_seconds_total': sum(i['wall_seconds'] for i in episodes),
        'scope': 'Model-driven exposed fixture episodes under declared budgets; not a generalization benchmark.'}

    # ---- stale workspace rejection after a host correction ---------------
    correction_case = evaluation['control_case']
    latest = next(item for item in episodes
                  if item['case_id'] == correction_case and item['mode'] == 'full')
    correction_work = args.output / 'correction_probe'
    copy_case(cases[correction_case]['dir'], correction_work)
    from replica_lab.runtime.journal import TaskJournal
    from replica_lab.runtime.loop import TaskRuntime
    from replica_lab.runtime.tools import FixtureWorkspace
    probe_tools = FixtureWorkspace(correction_work)
    probe_store = WorkspaceStore(correction_work / 'probe_workspace')
    probe_journal = TaskJournal.open(
        correction_work / 'probe_journal.sqlite',
        component_binding(probe_tools, subject=subject, branch=correction_case,
                          purpose=purpose))
    try:
        probe_runtime = TaskRuntime(probe_tools, probe_journal, workspace_store=probe_store,
                                    mode='full',
                                    max_prompt_tokens=int(budgets['prompt_tokens_per_decision']),
                                    generated_token_budget=int(budgets['generated_tokens_per_decision']),
                                    budget=episode_budget)
        probe_runtime.initialize(initial_state(probe_tools, subject=subject,
                                               branch=correction_case, purpose=purpose,
                                               budget=episode_budget))
        probe_request = session.prepare(
            prompt_ids=prompt_ids, prompt_mask=prompt_mask, query=probe_tools.goal,
            scope=scopes[correction_case],
            task=probe_runtime.state.task_graph(), selection=selections[correction_case],
            task_features=torch.zeros(1, 24, device='mps'),
            task_digest=probe_runtime.state.digest,
            revision=probe_runtime.state.revision)
        try:
            probe_store.save(probe_request.frame.core)
        finally:
            probe_request.release()
        before = probe_store.load()
        if before is None:
            raise AssertionError('Probe workspace did not persist')
        probe_runtime.correct(note='Host correction probe: revision changed after restart.',
                              constraint={'constraint_id': 'con-correction',
                                          'text': 'A later host correction is authoritative.'})
        discarded, reason = probe_runtime._load_previous_workspace(probe_runtime.state)
        require('stale_workspace_rejected_after_correction',
                discarded is None and reason in ('stale_revision', 'missing_or_incompatible'),
                reason=reason, stored_revision=None if before is None else before.revision,
                current_revision=probe_runtime.state.revision)
        report['checks']['stale_workspace_rejected_after_correction']['goal_unchanged'] = (
            probe_runtime.state.goal == probe_tools.goal)
    finally:
        probe_journal.close()

    # ---- actual-outcome success head -------------------------------------
    head_report = train_success_head(model, episodes, protocol, mark)
    report['success_head'] = head_report
    report['checks']['success_head_scope'] = {
        'passed': True, 'trained': head_report['trained'],
        'labels': head_report['labels'],
        'note': 'Prediction only; never authorizes task success.'}

    # ---- one final extension-only export --------------------------------
    extension_dir = args.output / 'selected_extension'
    receipt = model.save_extension(extension_dir)
    report['extension'] = {
        'path': str(extension_dir.resolve()), 'format': receipt['format'],
        'weights_sha256': receipt['weights_sha256'],
        'manifest_digest': receipt['manifest_digest'],
        'parameters': receipt['parameters'], 'base_saved': receipt['base_saved'],
        'bytes': sum(p.stat().st_size for p in extension_dir.rglob('*') if p.is_file()),
        'migration': report['parent_migration']}
    require('extension_only_export',
            receipt['base_saved'] is False
            and sorted(p.name for p in extension_dir.iterdir())
            == ['extensions.safetensors', 'manifest.json'],
            bytes=report['extension']['bytes'])
    require('original_parameters_unchanged',
            model._original_identity == model.parameter_stamp(original_only=True))
    # Task workspace/latent payloads are tiny tensors; a native base copy would
    # be gigabytes. Only the declared extension may exceed 1 MiB.
    payloads = [p for p in args.output.rglob('*.safetensors')
                if p.name != 'extensions.safetensors'
                and p.stat().st_size > 1_000_000]
    require('no_base_checkpoint_copy', not payloads,
            payloads=[{'path': str(p.relative_to(args.output)),
                       'bytes': p.stat().st_size} for p in payloads])

    report['costs'] = {
        'profiled_wall_seconds': None,
        'sampled_peak_driver_bytes': None,
        'episode_wall_seconds': report['evaluation']['wall_seconds_total'],
        'optimizer_updates': int(alignment['updates']) * len(alignment['arms']),
        'new_original_checkpoints': 0,
        'extension_bytes': report['extension']['bytes'],
        'downloads': 0}
    report['engineering_acceptance'] = bool(
        all(check.get('passed') for check in report['checks'].values()))
    report['learned_outcomes'] = {
        'heldout_success': report['evaluation']['heldout_success'],
        'control_success': report['evaluation']['control_success'],
        'status': 'reported, not promoted; tiny exposed fixtures cannot establish generalization'}
    ledger.close()
    mark('qualification_complete',
         engineering_acceptance=report['engineering_acceptance'],
         heldout=report['evaluation']['heldout_success'])


def preflight(args, profiler, report):
    """Bounded native wiring check: one max-shape backward and one generation.

    No optimizer step, no checkpoint and no evaluation action: this only proves
    the declared core path runs on MPS before the preregistered experiment.
    """
    import torch
    from replica_lab.architecture.contracts import CoreSpec
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import EligibilityScope, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.runtime.task_graph import Budget

    protocol = json.loads(args.protocol.read_text())
    core_spec = CoreSpec(**protocol['core'])
    model, tokenizer, source_lock = load_local_mps(
        args.model_path, args.model_lock, profiler=profiler, core=core_spec)
    migration = model.load_extension(Path(args.parent_extension), accept_parent_v1=True)
    report['parent_migration'] = migration.get('migration')
    report['manifest_digest'] = model.manifest.digest
    report['core_spec'] = asdict(core_spec)
    subject, purpose = 'e3_fixture', 'bounded_repair'
    case_id = protocol['fixtures']['train'][0]
    from replica_lab.runtime.tools import FixtureWorkspace
    tools = FixtureWorkspace(Path(protocol['fixtures']['root']) / case_id)
    ledger = RecordLedger(args.output / 'preflight_ledger.sqlite')
    try:
        admit_policy(ledger, tools, subject=subject, branch=case_id, purpose=purpose)
        scope = EligibilityScope(subject, case_id, purpose, 0, 'assistant_for_source', None)
        session = MemorySession(model, tokenizer, ledger, cache_dir=args.output / 'record_latents')
        selection = selection_for(session, scope, tools.goal)
        work = args.output / 'preflight_teacher'
        teacher = run_teacher(Path(protocol['fixtures']['root']) / case_id, work,
                              tokenizer=tokenizer, subject=subject, branch=case_id,
                              purpose=purpose,
                              max_prompt_tokens=int(protocol['alignment']['max_prompt_tokens']),
                              max_generated_tokens=int(protocol['alignment']['max_label_tokens']),
                              repair_tool=protocol.get('repair_tool', 'write_file'))
        if not teacher['summary']['success']:
            raise AssertionError('Preflight teacher trajectory failed')
        example = max(teacher['examples'], key=lambda item: item['prompt_tokens'])
        from replica_lab.architecture.contracts import TaskGraph
        example['task_graph'] = TaskGraph(
            task_id=case_id, goal=tools.goal,
            state_digest=example['task_digest'], revision=example['revision'])
        action_ids, mask, labels, prompt_tokens = build_action_batch(tokenizer, example, 'mps')
        model.train()
        with torch.no_grad():
            for port in model.workspace_ports.values():
                port.output_projection.weight.normal_(0.0, 0.002)
        with session.training_request(
                prompt_ids=action_ids[:, :prompt_tokens],
                prompt_mask=mask[:, :prompt_tokens], query=tools.goal, scope=scope,
                task=example['task_graph'], selection=selection,
                task_features=example['task_features'],
                task_digest=example['task_digest'], revision=example['revision'],
                previous_core=None, core_impl='recurrent',
                enabled_branches=('memory', 'workspace'),
                train_core=True, train_sense=True) as request:
            output = session.forward(request, action_ids, labels=labels, attention_mask=mask)
            loss_value = float(output.loss.detach())
            session.backward(request, output.loss)
            gradients = model.gradient_audit()
        model.zero_grad(set_to_none=True)
        nonzero = gradients['new']['nonzero']
        report['preflight'] = {
            'status': 'completed', 'teacher_case': case_id,
            'prompt_tokens': example['prompt_tokens'],
            'label_tokens': len(example['target_token_ids']),
            'loss': loss_value,
            'core_gradient': any(name.startswith('core.') for name in nonzero),
            'sensor_gradient': any(name.startswith('sense_ports.') for name in nonzero),
            'workspace_reader_gradient': any(name.startswith('workspace_ports.')
                                             for name in nonzero),
            'gradient_nonfinite': bool(gradients['new']['nonfinite'])
                                  or bool(gradients['original']['nonfinite']),
        }
        if not (report['preflight']['core_gradient']
                and report['preflight']['sensor_gradient']
                and report['preflight']['workspace_reader_gradient']
                and not report['preflight']['gradient_nonfinite']):
            raise AssertionError('Preflight gradient paths failed')
        del output
        model.eval()
        torch.mps.empty_cache()
        request = session.prepare(
            prompt_ids=action_ids[:, :prompt_tokens],
            prompt_mask=mask[:, :prompt_tokens], query=tools.goal, scope=scope,
            task=example['task_graph'], selection=selection,
            task_features=example['task_features'],
            task_digest=example['task_digest'], revision=example['revision'],
            previous_core=None, core_impl='recurrent',
            enabled_branches=('memory', 'workspace'))
        try:
            generated = session.generate(request, max_new_tokens=8)
            new_ids = generated[0, prompt_tokens:].tolist()
            report['preflight']['generated_token_ids'] = new_ids
            report['preflight']['generated_text'] = tokenizer.decode(
                new_ids, skip_special_tokens=False)
            report['preflight']['generated_finite'] = bool(
                torch.isfinite(session.forward(request, generated).logits).all())
        finally:
            request.release()
        if args.preflight_align_steps:
            from copy import deepcopy
            if not 1 <= args.preflight_align_steps <= 12:
                raise ValueError('preflight alignment steps are bounded to 12')
            short = deepcopy(protocol)
            short['alignment']['updates'] = args.preflight_align_steps
            short['alignment']['examples_per_update'] = 2
            short['alignment']['warmup_updates'] = 1
            examples = []
            for item in teacher['examples']:
                item['case_id'] = case_id
                item['goal'] = tools.goal
                item['task_graph'] = TaskGraph(
                    task_id=case_id, goal=tools.goal,
                    state_digest=item['task_digest'], revision=item['revision'])
                examples.append(item)
            audit = run_alignment_arm(
                model, session, tokenizer, examples, core_impl='recurrent',
                protocol=short, scope_for_case=lambda _: scope,
                selection_for_case=lambda _: selection,
                mark=lambda name, **fields: profiler.mark(name, **fields))
            report['preflight']['short_alignment_updates'] = args.preflight_align_steps
            report['preflight']['short_alignment_losses'] = [
                item['mean_loss'] for item in audit['updates']]
            first = examples[0]
            ids = tokenize_prompt(tokenizer, first['prompt_text'], 'mps')
            mask = torch.ones_like(ids, dtype=torch.bool)
            probe = session.prepare(
                prompt_ids=ids, prompt_mask=mask, query=tools.goal, scope=scope,
                task=first['task_graph'], selection=selection,
                task_features=first['task_features'], task_digest=first['task_digest'],
                revision=first['revision'], previous_core=None,
                core_impl='recurrent', enabled_branches=('memory', 'workspace'))
            try:
                generated = session.generate(probe, max_new_tokens=48)
                new_ids = generated[0, ids.shape[1]:].tolist()
                report['preflight']['post_alignment_generated_token_ids'] = new_ids
                report['preflight']['post_alignment_generated_text'] = tokenizer.decode(
                    new_ids, skip_special_tokens=False)
            finally:
                probe.release()
        report['status'] = 'completed_preflight'
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', type=Path,
                        default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path,
                        default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'configs/e3_qualification_v1.json')
    parser.add_argument('--parent-extension', type=Path,
                        default=ROOT / 'runs/e1_e2_native_attempt02_20261023/selected_extension')
    parser.add_argument('--validate-only', action='store_true',
                        help='Check protocol/fixtures/imports without loading the model')
    parser.add_argument('--preflight', action='store_true',
                        help='Load the sealed source on MPS and run one max-shape '
                             'backward plus one generation; no update/save/evaluation')
    parser.add_argument('--preflight-align-steps', type=int, default=0,
                        help='Optional short train-only alignment arm during preflight')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory; prior evidence is preserved')
    if shutil.disk_usage(ROOT).free < 5 * 1024 ** 3:
        parser.error('At least 5 GiB disk headroom required')
    protocol = json.loads(args.protocol.read_text())
    if args.validate_only:
        from replica_lab.runtime.tools import FixtureWorkspace
        checked = {}
        for role in ('train', 'dev', 'heldout', 'interrupted'):
            for case_id in protocol['fixtures'][role]:
                tools = FixtureWorkspace(Path(protocol['fixtures']['root']) / case_id)
                if tools.role != role:
                    raise SystemExit(f'Role mismatch for {case_id}')
                checked[case_id] = {'role': tools.role, 'goal_chars': len(tools.goal),
                                    'allowed_tools': list(tools.allowed_tools)}
        print(json.dumps({'status': 'protocol_valid', 'cases': checked}, indent=2))
        return
    if not Path(args.parent_extension).is_dir():
        parser.error('Parent extension directory is missing')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        'status': 'started',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'arguments': {key: str(value) for key, value in vars(args).items()},
        'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'protocol': protocol,
        'packages': {name: importlib.metadata.version(name)
                     for name in ('torch', 'transformers', 'safetensors')},
        'implementation_sha256': {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),
                         *(ROOT / 'src/replica_lab/architecture').glob('*.py'),
                         *(ROOT / 'src/replica_lab/memory_system').glob('*.py'),
                         *(ROOT / 'src/replica_lab/runtime').glob('*.py'),
                         ROOT / 'src/replica_lab/optimizers.py',
                         ROOT / 'src/replica_lab/neural.py']},
        'free_disk_before_bytes': shutil.disk_usage(ROOT).free,
        'downloads': 0, 'original_model_loads': 1,
        'environment': {key: os.environ[key] for key in
                        ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                         'HF_DATASETS_OFFLINE', 'PYTORCH_ENABLE_MPS_FALLBACK')},
    }
    profiler = RunProfiler('mps', args.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            if args.preflight:
                preflight(args, profiler, report)
            else:
                execute(args, profiler, report)
                report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        profile = profiler.stop()
        report['profile'] = profile
        peaks = (profile or {}).get('memory') or {}
        report.setdefault('costs', {})['profiled_wall_seconds'] = (profile or {}).get(
            'elapsed_seconds')
        report['costs']['sampled_peak_driver_bytes'] = peaks.get(
            'mps_driver_peak_sampled_bytes')
        report['costs']['sampled_peak_rss_bytes'] = peaks.get('rss_peak_sampled_bytes')
        report['free_disk_after_bytes'] = shutil.disk_usage(ROOT).free
        report['artifact_bytes'] = sum(p.stat().st_size for p in args.output.rglob('*')
                                       if p.is_file())
        report['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        dump(args.output / ('run_report.json'
                            if report['status'] in ('completed', 'completed_preflight')
                            else 'failure_report.json'), report)
    print(json.dumps({'report': str(args.output / 'run_report.json'),
                      'engineering_acceptance': report.get('engineering_acceptance'),
                      'heldout': report.get('evaluation', {}).get('heldout_success')}))


if __name__ == '__main__':
    main()
