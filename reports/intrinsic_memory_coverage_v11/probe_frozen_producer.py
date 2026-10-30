"""Post-run CPU representation inspection; no native model, fit or release write.

Uses only the already-opened fit worlds and a verified extension checkpoint.
Cosine/RMS diagnostics are not generated-answer or generalization evidence.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from safetensors.torch import load_file
from replica_lab.contextual_memory import ContextualValueEncoder, canonical_spans


def digest(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or substituted diagnostic input')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    protocol_path = ROOT / 'configs/spark_intrinsic_coverage_v11.json'
    protocol = json.loads(protocol_path.read_text())
    data = ROOT / protocol['data']['fit']['path']
    assert digest(data) == protocol['data']['fit']['sha256']
    rows = json.loads(data.read_text())['rows']
    out = {'format': 'replica.intrinsic_memory.frozen_producer_diagnosis.v2',
           'scope': 'CPU_fit_only_no_backbone_no_generation_no_training',
           'data': protocol['data']['fit'], 'arms': {}}
    for version, directory in [(10, 'alignment'), (11, 'coverage')]:
        path = ROOT / f'reports/intrinsic_memory_{directory}_v{version}/native_bf16_two_v{version}_attempt01'
        supervision = json.loads((path / 'supervision/receipt.json').read_text())
        assert supervision['process_group_exit_confirmed'] and supervision['child_exit_code'] == 0
        result = json.loads((path / 'result.json').read_text())
        assert result['main']['original_weights_unchanged']
        checkpoint = path / 'checkpoint'
        wrapper = json.loads((checkpoint / 'manifest.json').read_text())
        request = json.loads((path / 'restart_request.json').read_text())
        assert digest(checkpoint / 'manifest.json') == request['checkpoint_manifest_sha256']
        assert digest(checkpoint / 'components/manifest.json') == wrapper['components_manifest_sha256']
        manifest = json.loads((checkpoint / 'components/manifest.json').read_text())
        weights = checkpoint / 'components/extension.safetensors'
        assert digest(weights) == manifest['weights_sha256']
        tensors = load_file(str(weights), device='cpu')
        encoder = ContextualValueEncoder().eval()
        encoder.load_state_dict({k.removeprefix('producer.'): v for k, v in tensors.items() if k.startswith('producer.')}, strict=True)
        assert encoder.release() == manifest['producer_release']
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        values = []
        with torch.no_grad():
            for row in rows:
                spans = canonical_spans(row['text'], record_id=row['world_id'], source_id=row['world_id'], source_epoch=0, available_at=0)
                assert len(spans) == 1
                values.append(encoder(spans)[0])
        value = torch.stack(values)
        pairs = []
        for i in range(0, len(rows), 2):
            assert rows[i]['question'] == rows[i + 1]['question']
            a, b = value[i], value[i + 1]
            pairs.append({'worlds': [rows[i]['world_id'], rows[i + 1]['world_id']],
                'rms_difference': float((a - b).square().mean().sqrt()),
                'relative_rms_difference': float((a - b).norm() / ((a.norm() + b.norm()) / 2)),
                'same_slot_cosines': torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()})
        slots = []
        for row, item in zip(rows, value):
            normalized = torch.nn.functional.normalize(item, dim=-1)
            similarity = normalized @ normalized.T
            slots.append({'world_id': row['world_id'],
                'slot_rms': item.square().mean(-1).sqrt().tolist(),
                'off_diagonal_slot_cosine_mean': float((similarity.sum() - similarity.diagonal().sum()) / 12)})
        out['arms'][f'v{version}'] = {
            'checkpoint_manifest_sha256': digest(checkpoint / 'manifest.json'),
            'extension_sha256': digest(weights), 'producer_release': encoder.release(),
            'live_CPU_values_shape': list(value.shape), 'finite': bool(torch.isfinite(value).all()),
            'source_bytes': sum(len(row['text'].encode()) for row in rows),
            'source_encodes': len(rows), 'read_checkpoint_bytes': weights.stat().st_size,
            'producer_parameters': sum(p.numel() for p in encoder.parameters()),
            'order_embedding_rms': float(tensors['order.weight'][:4].square().mean().sqrt()),
            'contextual_slot_rms': float(value.square().mean().sqrt()),
            'same_question_counterfactual_pairs': pairs, 'within_span_slots': slots,
            'checkpoint_unchanged': digest(weights) == manifest['weights_sha256']}
    out.update(wall_seconds=time.monotonic()-started,
               peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               native_model_loads=0, updates=0, development_data_opened=False,
               limits=['This CPU analysis is posthoc and reads fitting data only.',
                       'A nonzero slot difference does not prove that a native reader can recover the answer.',
                       'CPU encoder outputs are not claimed bitwise equal to MPS outputs.',
                       'Representation similarity cannot establish a causal explanation for failed recall.',
                       'V10 and V11 have different fitting exposure; this comparison is descriptive and not a matched-training experiment.'])
    destination = ROOT / 'reports/intrinsic_memory_coverage_v11/frozen_producer_diagnosis.json'
    with destination.open('x') as output:
        output.write(json.dumps(out, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: out[k] for k in ('wall_seconds','peak_process_rss_bytes','native_model_loads','updates')}, indent=2))


if __name__ == '__main__':
    main()
