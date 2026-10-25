"""Versioned encodings for derived persistent state, never a memory engine."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re

from ..architecture.contracts import fingerprint
from ..neural import SGDN2Memory
from .fact_memory import encode_fact_binding, fact_source_text, validate_fact

RAW_REPRESENTATION = 'raw_observation.v1'
FACT_REPRESENTATION = 'fact_binding_key_value.v1'
PERSISTENT_FORMAT = 'replica.persistent_fast_memory.v2'


def representation_configuration(name):
    if name == RAW_REPRESENTATION:
        return {'name': 'raw_observation', 'version': 1, 'parser': 'opaque_utf8.v1',
                'normalization': 'identity_utf8.v1', 'pooling': 'native_token_writes.v1'}
    if name == FACT_REPRESENTATION:
        return {'name': 'fact_binding_key_value', 'version': 1,
                'parser': 'canonical_observed_fact.v1', 'normalization': 'identity_utf8.v1',
                'pooling': 'masked_mean_key_and_value.v1'}
    raise ValueError('Unknown or unsupported persistent-memory representation')


def parse_fact_source(text):
    """Parse the exact source grammar without case/whitespace normalization."""
    if not isinstance(text, str):
        raise ValueError('Fact source must be exact UTF-8 text')
    match = re.fullmatch(r'(The|Update: the) (.+) for project (\S+) is (.+)\.', text)
    if match is None:
        raise ValueError('Source is not a canonical observed fact')
    fact = validate_fact({'attribute': match[2], 'entity': match[3], 'value': match[4]})
    correction = match[1] == 'Update: the'
    if fact_source_text(fact, correction=correction) != text:
        raise ValueError('Fact parser would change source bytes')
    return fact, correction


def tokenizer_identity(tokenizer):
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is not None:
        payload = {'backend': backend.to_str(),
                   'special_tokens': getattr(tokenizer, 'special_tokens_map', {}),
                   'chat_template': getattr(tokenizer, 'chat_template', None)}
    else:
        # Tiny software fixtures have no native backend; bind code and state,
        # not just a class name or a handful of probe tokenizations.
        payload = {'implementation': inspect.getsource(type(tokenizer)),
                   'state': vars(tokenizer)}
    try:
        return fingerprint(json.loads(json.dumps(payload, sort_keys=True)))
    except (TypeError, ValueError) as error:
        raise ValueError('Tokenizer needs a serializable, explicit producer identity') from error


def encoding_dependencies_from_types(adapter_type, writer_type, tokenizer, sealed_tokenizer, representation):
    """Also usable by the CPU artifact scorer without loading a backbone."""
    representation_configuration(representation)
    pieces = {'native_observation': inspect.getsource(adapter_type.encode_fast_observation),
              'writer': inspect.getsource(writer_type), 'state_rule': inspect.getsource(SGDN2Memory)}
    if representation == FACT_REPRESENTATION:
        pieces['fact_encoder'] = inspect.getsource(encode_fact_binding)
        pieces['fact_parser'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        pieces['fact_render'] = inspect.getsource(fact_source_text)
        pieces['fact_validation'] = inspect.getsource(validate_fact)
    return {'tokenizer': tokenizer_identity(tokenizer),
            'sealed_tokenizer': sealed_tokenizer,
            'encoder_implementation': fingerprint(pieces)}


def encoding_dependencies(model, tokenizer, representation):
    writer = model.fast_depth_write_encoders[str(model.ports.injection_layers[0])]
    return encoding_dependencies_from_types(type(model), type(writer), tokenizer,
                                            model.manifest.tokenizer_digest, representation)


def encoding_producer(model_key, representation, dependencies, engine_config):
    return fingerprint({'model': model_key, 'representation': representation_configuration(representation),
                        'dependencies': dependencies, 'engine': engine_config.to_dict()})
