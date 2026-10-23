"""Offline E3 extension-migration contracts: v1 parent adoption and v2 strictness."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from architecture_helpers import action_batch, synthetic_frame, tiny_adapter
from replica_lab.architecture.contracts import CoreSpec
from replica_lab.architecture.minicpm5 import MiniCPM5Adapter

TINY_SPEC = CoreSpec(width=16, slots=4, steps=2, heads=4, task_features=24,
                     ffn_hidden=64, control_hidden=29)


def e3_adapter(regime='modules_only', revision='offline-tiny-native-test'):
    adapter = tiny_adapter(regime)
    manifest = replace(adapter.manifest, core=TINY_SPEC)
    if revision != adapter.manifest.source_revision:
        manifest = replace(manifest, source_revision=revision)
    return MiniCPM5Adapter(adapter.base, manifest, regime)


def rewrite_weights(directory, weights):
    save_file(weights, str(Path(directory) / 'extensions.safetensors'))
    path = Path(directory) / 'manifest.json'
    receipt = json.loads(path.read_text())
    receipt['weights_sha256'] = hashlib.sha256((Path(directory) / 'extensions.safetensors')
                                               .read_bytes()).hexdigest()
    path.write_text(json.dumps(receipt))


def test_parent_v1_extension_migrates_into_core_bearing_adapter(tmp_path):
    parent = tiny_adapter().eval()
    parent.seal_original_identity()
    destination = tmp_path / 'parent'
    saved = parent.save_extension(destination)
    weights = load_file(str(destination / 'extensions.safetensors'))
    child = e3_adapter().eval()
    child.seal_original_identity()
    fresh = {n: p.detach().clone() for n, p in child.new_named_parameters()}
    receipt = child.load_extension(destination, accept_parent_v1=True)
    assert receipt['migration']['accepted_parent_v1'] is True
    assert sorted(receipt['migration']['copied']) == sorted(weights)
    assert 'core.slot_queries' in receipt['migration']['fresh_init']
    for name, parameter in child.new_named_parameters():
        if name in weights:
            torch.testing.assert_close(parameter, weights[name], atol=0, rtol=0)
        else:
            torch.testing.assert_close(parameter, fresh[name], atol=0, rtol=0)
    # The E1/E2 dictionary itself is unchanged by migration.
    assert saved['format'] == 'replica.minicpm5.extensions.v1'


def test_parent_v1_requires_flag_matching_manifest_and_inventory(tmp_path):
    parent = tiny_adapter().eval()
    parent.seal_original_identity()
    destination = tmp_path / 'parent'
    parent.save_extension(destination)

    child = e3_adapter().eval()
    child.seal_original_identity()
    with pytest.raises(ValueError, match='accept_parent_v1'):
        child.load_extension(destination)

    wrong_source = e3_adapter(revision='different-source').eval()
    wrong_source.seal_original_identity()
    with pytest.raises(ValueError, match='Incompatible or changed'):
        wrong_source.load_extension(destination, accept_parent_v1=True)

    weights = load_file(str(destination / 'extensions.safetensors'))
    extra = dict(weights)
    extra['core.slot_queries'] = torch.zeros(1)
    rewrite_weights(destination, extra)
    with pytest.raises(ValueError, match='inventory differs'):
        child.load_extension(destination, accept_parent_v1=True)

    missing = dict(list(weights.items())[:-1])
    rewrite_weights(destination, missing)
    with pytest.raises(ValueError, match='inventory differs'):
        child.load_extension(destination, accept_parent_v1=True)

    name = sorted(weights)[0]
    reshaped = dict(weights)
    reshaped[name] = torch.zeros(weights[name].numel() + 1)
    rewrite_weights(destination, reshaped)
    with pytest.raises(ValueError, match='inventory differs'):
        child.load_extension(destination, accept_parent_v1=True)


def test_v2_roundtrip_is_strict(tmp_path):
    adapter = e3_adapter().eval()
    adapter.seal_original_identity()
    destination = tmp_path / 'e3'
    receipt = adapter.save_extension(destination)
    assert receipt['format'] == 'replica.minicpm5.extensions.v2'
    assert set(receipt['components']) == {'record_writer', 'memory_ports', 'sense_ports',
                                          'reranker', 'core', 'ff_core', 'workspace_ports',
                                          'task_head'}
    weights = load_file(str(destination / 'extensions.safetensors'))
    restored = e3_adapter().eval()
    restored.seal_original_identity()
    restored.load_extension(destination)
    for name, parameter in restored.new_named_parameters():
        torch.testing.assert_close(parameter, weights[name], atol=0, rtol=0)

    path = destination / 'manifest.json'
    original = path.read_text()
    mutated = json.loads(original)
    mutated['manifest_digest'] = '0' * 64
    path.write_text(json.dumps(mutated))
    with pytest.raises(ValueError, match='Incompatible or changed'):
        _reload(destination)
    path.write_text(original)
    with (destination / 'extensions.safetensors').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Incompatible or changed'):
        _reload(destination)
    path.write_text(original)

    mutated = json.loads(original)
    mutated['unexpected_key'] = True
    path.write_text(json.dumps(mutated))
    with pytest.raises(ValueError, match='Incompatible or changed'):
        _reload(destination)
    path.write_text(original)

    extra = dict(weights)
    extra['core.extra'] = torch.zeros(1)
    rewrite_weights(destination, extra)
    with pytest.raises(ValueError, match='inventory differs'):
        _reload(destination)


def _reload(destination):
    adapter = e3_adapter().eval()
    adapter.seal_original_identity()
    return adapter.load_extension(destination)


def test_real_parent_artifact_receipt_shape_is_v1_migration_compatible():
    """Structure-only check; loading the real model is out of scope for tests."""
    receipt = json.loads(Path('runs/e1_e2_native_attempt02_20261023/selected_extension/manifest.json')
                         .read_text())
    assert receipt['format'] == 'replica.minicpm5.extensions.v1'
    assert set(receipt) == {'format', 'manifest', 'manifest_digest', 'regime',
                            'weights_sha256', 'parameters', 'base_saved', 'notes'}
    assert 'core' not in receipt['manifest']
    assert set(receipt['manifest']) == {'source_revision', 'source_seal_digest',
                                        'tokenizer_digest', 'native_config_digest',
                                        'native_abi_digest', 'ports', 'format_version',
                                        'training_scope', 'cached_generation_supported'}
    assert receipt['base_saved'] is False
    assert len(receipt['weights_sha256']) == 64
    # Real-artifact loading needs the sealed 2.5B source; the parent verifies the
    # actual v1 adoption in the native qualification run.


def test_v2_missing_keys_and_component_map_mismatch_are_rejected(tmp_path):
    adapter = e3_adapter().eval()
    adapter.seal_original_identity()
    destination = tmp_path / 'e3'
    adapter.save_extension(destination)
    manifest_path = destination / 'manifest.json'
    original = manifest_path.read_text()
    weights = load_file(str(destination / 'extensions.safetensors'))

    missing = json.loads(original)
    missing.pop('components')
    manifest_path.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match='Incompatible or changed'):
        _reload(destination)
    manifest_path.write_text(original)

    shortened = json.loads(original)
    shortened['components']['core'] = shortened['components']['core'][:-1]
    manifest_path.write_text(json.dumps(shortened))
    with pytest.raises(ValueError, match='component inventory differs'):
        _reload(destination)
    manifest_path.write_text(original)

    ghost = json.loads(original)
    ghost['components']['ghost'] = ['ghost.weight']
    manifest_path.write_text(json.dumps(ghost))
    with pytest.raises(ValueError, match='component inventory differs'):
        _reload(destination)
    manifest_path.write_text(original)

    absent = dict(weights)
    absent.pop(sorted(weights)[0])
    rewrite_weights(destination, absent)
    with pytest.raises(ValueError, match='inventory differs'):
        _reload(destination)
    rewrite_weights(destination, weights)

    wrong_shape = dict(weights)
    name = sorted(weights)[1]
    wrong_shape[name] = torch.zeros(weights[name].numel() + 1, dtype=weights[name].dtype)
    rewrite_weights(destination, wrong_shape)
    with pytest.raises(ValueError, match='inventory differs'):
        _reload(destination)
    rewrite_weights(destination, weights)

    wrong_dtype = dict(weights)
    name = sorted(weights)[2]
    wrong_dtype[name] = weights[name].to(torch.float64)
    rewrite_weights(destination, wrong_dtype)
    with pytest.raises(ValueError, match='inventory differs'):
        _reload(destination)


def test_v1_into_core_less_adapter_is_behavior_identical(tmp_path):
    parent = tiny_adapter().eval()
    parent.seal_original_identity()
    destination = tmp_path / 'parent'
    parent.save_extension(destination)
    weights = load_file(str(destination / 'extensions.safetensors'))

    restored = tiny_adapter().eval()
    with torch.no_grad():
        for _, parameter in restored.new_named_parameters():
            parameter.add_(0.05)
    restored.seal_original_identity()
    receipt = restored.load_extension(destination)
    assert receipt['format'] == 'replica.minicpm5.extensions.v1'
    for name, parameter in restored.new_named_parameters():
        torch.testing.assert_close(parameter, weights[name], atol=0, rtol=0)

    def logits(adapter):
        frame = synthetic_frame(adapter)
        with torch.no_grad():
            return adapter.forward_with_frame(frame=frame, **action_batch(frame)).logits

    torch.testing.assert_close(logits(restored), logits(parent), atol=0, rtol=0)


def test_port_capacity_only_migration_preserves_tensors_and_rejects_other_changes(tmp_path):
    from architecture_helpers import synthetic_frame
    from replica_lab.architecture.contracts import PortSpec

    parent = tiny_adapter().eval()
    parent.seal_original_identity()
    destination = tmp_path / 'capacity-parent'
    parent.save_extension(destination)

    base = tiny_adapter()
    target_ports = PortSpec(**{**base.ports.__dict__, 'max_units': base.ports.max_units + 4})
    target = MiniCPM5Adapter(base.base, replace(base.manifest, ports=target_ports),
                             'modules_only').eval()
    target.seal_original_identity()
    receipt = target.load_extension(destination, accept_parent_v1=True,
                                    port_capacity_migration=True)
    assert receipt['migration']['port_capacity_only'] is True
    assert receipt['migration']['parent_max_units'] == parent.ports.max_units
    assert receipt['migration']['target_max_units'] == target_ports.max_units
    weights = load_file(str(destination / 'extensions.safetensors'))
    for name, parameter in target.new_named_parameters():
        torch.testing.assert_close(parameter, weights[name], atol=0, rtol=0)

    wrong_base = tiny_adapter()
    wrong = MiniCPM5Adapter(wrong_base.base,
                            replace(wrong_base.manifest, source_revision='different-source'),
                            'modules_only').eval()
    wrong.seal_original_identity()
    with pytest.raises(ValueError, match='migration'):
        wrong.load_extension(destination, accept_parent_v1=True,
                             port_capacity_migration=True)

    shrunk = MiniCPM5Adapter(base.base, replace(base.manifest, ports=PortSpec(
        **{**base.ports.__dict__, 'max_units': base.ports.max_units - 1})),
        'modules_only').eval()
    shrunk.seal_original_identity()
    with pytest.raises(ValueError, match='max_units'):
        shrunk.load_extension(destination, accept_parent_v1=True,
                              port_capacity_migration=True)
