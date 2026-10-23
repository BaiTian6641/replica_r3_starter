#!/usr/bin/env python3
"""Bounded E3 engineering preflight: native pass/backward/generation cost.

No training update, checkpoint, download or task evidence is produced. The probe
only measures the sealed original at declared prompt lengths so the E3 protocol
can set bounded token/update budgets. It is not part of the E3 evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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


def synchronize(torch):
    torch.mps.synchronize()


def timed(torch, action):
    synchronize(torch)
    start = time.perf_counter()
    value = action()
    synchronize(torch)
    return value, time.perf_counter() - start


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.architecture.contracts import TaskGraph, RequestContext
    from replica_lab.memory_system import RecordLedger, MemoryRecord, EligibilityScope
    from replica_lab.memory_system.session import MemorySession

    torch.manual_seed(args.seed)
    model, tokenizer, lock = load_local_mps(args.model_path, args.model_lock, profiler=profiler)
    report['source_revision'] = lock['revision']
    report['manifest_digest'] = model.manifest.digest
    report['parameters'] = model.parameter_audit()

    filler = ('The project configuration defines retry attempts, backoff seconds, '
              'and request timeout milliseconds. The evaluator test is authoritative. ')
    text = filler * 64
    ids = tokenizer(text, add_special_tokens=False)['input_ids'][:args.prompt_tokens]
    if len(ids) < 64:
        raise ValueError('Tokenizer produced too few probe tokens')
    prompt = torch.tensor([ids], dtype=torch.long, device='mps')
    mask = torch.ones_like(prompt, dtype=torch.bool)
    report['prompt_tokens'] = int(prompt.shape[1])

    sense, sense_seconds = timed(torch, lambda: model.sense(prompt, mask))
    report['sense_seconds'] = sense_seconds

    ledger_path = args.output / 'probe.sqlite'
    ledger = RecordLedger(ledger_path)
    ledger.admit(MemoryRecord('probe', 'probe_subject', 'probe_branch',
                              'The evaluator test is authoritative.', 1, 1, 'probe'),
                 authorized_subject='probe_subject', authorized_branch='probe_branch')
    session = MemorySession(model, tokenizer, ledger)
    scope = EligibilityScope('probe_subject', 'probe_branch', 'probe', 10)
    task = TaskGraph('probe', 'measure native cost')
    request = session.prepare(prompt_ids=prompt, prompt_mask=mask, query='probe',
                              scope=scope, task=task)
    try:
        report['empty_memory_units'] = len(request.frame.memory.spans)
        output, forward_seconds = timed(
            torch, lambda: session.forward(request, prompt))
        report['forward_seconds'] = forward_seconds
        report['forward_logits_shape'] = list(output.logits.shape)
        del output
        suffix = prompt[:, -args.label_tokens:].clone()
        action_ids = torch.cat((prompt, suffix), dim=1)
        action_mask = torch.ones_like(action_ids, dtype=torch.bool)
        labels = action_ids.clone()
        labels[:, :prompt.shape[1]] = -100
        session.adapter.set_regime('modules_only')
        model.train()
        session.adapter.base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        request2 = session.prepare(prompt_ids=prompt, prompt_mask=mask, query='probe',
                                   scope=scope, task=task)
        try:
            result = session.forward(request2, action_ids, labels=labels,
                                     attention_mask=action_mask)
            _, backward_seconds = timed(torch, result.loss.backward)
            report['backward_seconds'] = backward_seconds
            report['probe_loss'] = float(result.loss.detach())
            report['gradient_audit'] = {
                'new_present': len(model.gradient_audit()['new']['present']),
                'new_nonzero': len(model.gradient_audit()['new']['nonzero']),
                'original_present': 0,
            }
        finally:
            request2.release()
            model.zero_grad(set_to_none=True)
        model.eval()
        del result
        torch.mps.empty_cache()
        generated, generation_seconds = timed(
            torch, lambda: session.generate(request, max_new_tokens=args.generate_tokens))
        tokens = int(generated.shape[1] - prompt.shape[1])
        report['generation'] = {'tokens': tokens, 'seconds': generation_seconds,
                                'seconds_per_token': generation_seconds / max(tokens, 1)}
        report['raw_text'] = tokenizer.decode(generated[0, prompt.shape[1]:].tolist(),
                                             skip_special_tokens=False)
    finally:
        request.release()
    ledger.close()
    profile = profiler.summary() if hasattr(profiler, 'summary') else None
    report['profile_peaks'] = profile.get('peaks') if profile else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path, default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    parser.add_argument('--prompt-tokens', type=int, default=384)
    parser.add_argument('--label-tokens', type=int, default=16)
    parser.add_argument('--generate-tokens', type=int, default=12)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if not 128 <= args.prompt_tokens <= 512 or not 1 <= args.label_tokens < args.prompt_tokens:
        parser.error('Probe prompt/label bounds are invalid')
    if not 1 <= args.generate_tokens <= 32:
        parser.error('Generation probe is bounded to 32 tokens')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    report = {'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'device': 'mps', 'dtype': 'float32', 'downloads': 0, 'optimizer_updates': 0,
              'checkpoints_saved': 0,
              'arguments': {k: str(v) for k, v in vars(args).items()},
              'purpose': 'engineering sizing preflight only; not an E3 evaluation result'}
    profiler = RunProfiler('mps', args.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            execute(args, profiler, report)
            report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        report['profile'] = profiler.stop()
        report['free_disk_bytes'] = shutil.disk_usage(ROOT).free
        report['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        (args.output / 'speed_probe.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'sense_seconds', 'forward_seconds',
                                                 'backward_seconds', 'generation')}))


if __name__ == '__main__':
    main()
