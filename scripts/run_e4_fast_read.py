#!/usr/bin/env python3
"""Bounded native fast-memory read-port check; no training or generation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib, json, os, shutil, sys, traceback
from pathlib import Path
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(args, profiler, report):
    import torch
    from replica_lab.architecture.contracts import (EvidencePin, FastMemoryRead,
                                                     FastMemorySpec, MemoryRead)
    from replica_lab.architecture.loading import encode_prompt, load_local_mps
    protocol = json.loads(args.protocol.read_text())
    spec = FastMemorySpec(**protocol['fast_memory'])
    model, tokenizer, lock = load_local_mps(args.model_path, args.model_lock,
                                            profiler=profiler, fast_memory=spec)
    parent = model.load_extension(Path(args.parent_extension), accept_parent_v1=True)
    report.update(source_lock=lock, manifest_digest=model.manifest.digest,
                  parent_migration=parent.get('migration'),
                  fast_spec=protocol['fast_memory'], parameters=model.parameter_audit(),
                  checks={})
    prompt = tokenizer('Native fast-memory read-port check.', add_special_tokens=False)
    ids = torch.tensor([prompt['input_ids'][:int(protocol['prompt_tokens'])]],
                       dtype=torch.long, device='mps')
    mask = torch.ones_like(ids, dtype=torch.bool)
    empty = MemoryRead((), model.base.get_input_embeddings().weight.new_zeros(
        1, 0, model.ports.slots_per_unit, model.ports.memory_width),
        torch.zeros(1, 0, model.ports.slots_per_unit, dtype=torch.bool, device='mps'),
        model.memory_producer_key())
    sense = model.sense(ids, mask)
    evidence = EvidencePin('fast-read-probe', 0, 'e2_capacity', 'capacity',
                           'capacity_probe', 0, 'assistant_for_source', None, ())
    zero_weights = torch.zeros(1, spec.heads, spec.key_size, spec.value_size, device='mps')
    zero_fast = FastMemoryRead(zero_weights, model.fast_producer_key(), 'zero-state')
    zero_frame = model.make_frame('fast-zero', evidence, empty, sense, fast=zero_fast)
    no_fast_frame = model.make_frame('fast-off', evidence, empty, sense)
    with torch.no_grad():
        native = model.base(input_ids=ids, attention_mask=mask, use_cache=False).logits
        zero = model.forward_with_frame(ids, zero_frame).logits
        off = model.forward_with_frame(ids, no_fast_frame).logits
    report['checks']['zero_state_null_parity'] = {
        'passed': float((native - zero).abs().max()) <= 1e-5,
        'max_absolute_difference': float((native - zero).abs().max())}
    report['checks']['branch_off_null_parity'] = {
        'passed': float((native - off).abs().max()) <= 1e-5,
        'max_absolute_difference': float((native - off).abs().max())}
    with torch.no_grad():
        for port in model.fast_ports.values():
            port.output_projection.weight.normal_(0.0, 0.002)
        active_weights = torch.randn(1, spec.heads, spec.key_size, spec.value_size,
                                     device='mps')
        active_fast = FastMemoryRead(active_weights, model.fast_producer_key(), 'active-state')
        active_frame = model.make_frame('fast-active', evidence, empty, sense,
                                        fast=active_fast)
        active = model.forward_with_frame(ids, active_frame).logits
    delta = float((active - zero).abs().max())
    report['checks']['nonzero_state_influence'] = {
        'passed': delta > 1e-6 and bool(torch.isfinite(active).all()),
        'max_absolute_logit_difference': delta}
    report['status'] = 'completed'
    report['claims'] = 'Read-port wiring only; no fast-state training, delayed use or task benefit.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/e4_fast_read_v1.json')
    parser.add_argument('--model-path', type=Path, default=ROOT / 'checkpoints/minicpm5_source')
    parser.add_argument('--model-lock', type=Path,
                        default=ROOT / 'configs/minicpm5_sealed_lock.local.json')
    parser.add_argument('--parent-extension', type=Path,
                        default=ROOT / 'runs/e1_e2_native_attempt02_20261023/selected_extension')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    from replica_lab.profiling import RunProfiler
    from replica_lab.architecture.loading import exclusive_mps_worker
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha(args.protocol),
              'protocol': json.loads(args.protocol.read_text()),
              'downloads': 0, 'optimizer_updates': 0, 'checkpoints_saved': 0}
    profiler = RunProfiler('mps', args.output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            execute(args, profiler, report)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        profile = profiler.stop(); report['profile'] = profile
        memory = (profile or {}).get('memory') or {}
        report['costs'] = {'profiled_wall_seconds': (profile or {}).get('elapsed_seconds'),
                           'sampled_peak_driver_bytes': memory.get('mps_driver_peak_sampled_bytes')}
        report['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        (args.output / ('run_report.json' if report['status'] == 'completed'
                        else 'failure_report.json')).write_text(
            json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'checks': report.get('checks')}))


if __name__ == '__main__':
    main()
