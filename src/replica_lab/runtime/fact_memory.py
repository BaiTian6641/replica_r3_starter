"""Observed-fact key/value binding diagnostics and writer representation.

This is an input representation for the existing SGDN-2 writer, not a second
memory engine. Only observed entity/attribute/value fields are accepted; answer
labels, split metadata and future outcomes are rejected. Key and value texts
are encoded independently with the existing native writer, pooled to one write,
and then reduced by the unchanged state update rule.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from ..neural import SGDN2Write

FACT_FIELDS = frozenset(('entity', 'attribute', 'value'))


def validate_fact(fact):
    if not isinstance(fact, Mapping) or set(fact) != FACT_FIELDS:
        raise ValueError('A fact must contain exactly entity, attribute and value')
    for name in FACT_FIELDS:
        value = fact[name]
        if not isinstance(value, str) or not value.strip() or any(c in value for c in '\r\n'):
            raise ValueError(f'Fact {name} must be nonempty single-line text')
    return {name: fact[name] for name in ('entity', 'attribute', 'value')}


def fact_source_text(fact, *, correction=False):
    fact = validate_fact(fact)
    prefix = 'Update: the' if correction else 'The'
    return f"{prefix} {fact['attribute']} for project {fact['entity']} is {fact['value']}."


def _pooled(write, name):
    value = getattr(write, name)
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] < 1:
        raise ValueError('Writer output must be [1, tokens, heads, width]')
    if write.mask is None:
        return value.mean(dim=1, keepdim=True)
    mask = write.mask.to(device=value.device, dtype=value.dtype)
    if mask.ndim != 2 or mask.shape != value.shape[:2] or not bool(mask.sum(dim=1).min() > 0):
        raise ValueError('Writer mask has no valid source token')
    weights = mask.unsqueeze(-1).unsqueeze(-1)
    return (value * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1)


def encode_fact_binding(model, tokenizer, fact, *, layer):
    """Encode one observed fact as one explicit key/value SGDN-2 write.

    The key path sees only entity+attribute; the value path sees only value.
    Mean pooling is parameter-free and preserves gradients through both writer
    calls. No answer target or generated text is accepted by this function.
    """
    fact = validate_fact(fact)
    key_text = f"{fact['entity']} {fact['attribute']}"
    value_text = fact['value']
    for text in (key_text, value_text):
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        if not ids or len(ids) > 256:
            raise ValueError('Fact key/value must fit without silent truncation')
    key_write = model.encode_fast_observation(key_text, tokenizer, layer=layer)
    value_write = model.encode_fast_observation(value_text, tokenizer, layer=layer)
    if (key_write.keys.shape[0] != 1 or value_write.values.shape[0] != 1
            or key_write.keys.shape[2] != value_write.values.shape[2]
            or key_write.keys.shape[3] != key_write.erase.shape[3]
            or key_write.values.shape[3] != value_write.values.shape[3]):
        raise ValueError('Fact writer key/value dimensions differ')
    mask = torch.ones((1, 1), dtype=torch.bool, device=key_write.keys.device)
    return SGDN2Write(
        _pooled(key_write, 'keys'), _pooled(value_write, 'values'),
        _pooled(key_write, 'erase'), _pooled(key_write, 'write_gate'),
        _pooled(key_write, 'decay'), key_write.chunk_index, mask)


def binding_diagnostic(model, tokenizer, facts, *, layer):
    """Measure key/value sensitivity without updating any model parameters."""
    if len(facts) < 2:
        raise ValueError('Binding diagnostic requires two observed facts')
    first, second = (validate_fact(fact) for fact in facts[:2])
    with torch.no_grad():
        base = encode_fact_binding(model, tokenizer, first, layer=layer)
        rebound = encode_fact_binding(
            model, tokenizer, dict(first, entity=second['entity'], attribute=second['attribute']), layer=layer)
        revalued = encode_fact_binding(model, tokenizer, dict(first, value=second['value']), layer=layer)
        key_change = (base.keys - rebound.keys).float().norm().item()
        value_change = (base.values - revalued.values).float().norm().item()
        value_invariant = torch.equal(base.values, rebound.values)
        key_invariant = torch.equal(base.keys, revalued.keys)
    return {
        'scope': 'observed_fact_representation_only',
        'labels_used': False,
        'parameters_updated': False,
        'layer': layer,
        'key_change_norm': key_change,
        'value_change_norm': value_change,
        'value_invariant_under_key_change': value_invariant,
        'key_invariant_under_value_change': key_invariant,
        'write_shape': list(base.keys.shape),
        'hypothesis': 'Explicit entity+attribute keys and value-only values may repair association storage; reader and gate remain fixed.',
    }
