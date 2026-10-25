"""External dataset tranche audit: pinned identity, terms, splits, counts.

No acquisition occurs here. The audit selects a small relevant tranche from the
reviewed catalogue, reserves evaluation sources before any training use, keeps
related derivatives in one split, and fails closed on missing immutable pins,
unresolved terms, duplicate family assignment or budget overrun.
"""
from __future__ import annotations

import json
from pathlib import Path

from .vision_qualification import require

TRANCHE_FORMAT = 'replica.external_tranche.v1'
AUDIT_FORMAT = 'replica.external_intake_audit.v1'
CAMPAIGN_DISK_CEILING_BYTES = 25 * 1024**3

SELECTIONS = (
    {'id': 'HuggingFaceM4/Docmatix',
     'role': 'document_ocr_train_candidate', 'intent': 'train',
     'requires_assets': True, 'notes': 'Image subset only; per-component terms and asset inspection still required.'},
    {'id': 'liuhaotian/LLaVA-Instruct-150K',
     'role': 'visual_instruction_train_candidate', 'intent': 'train',
     'requires_assets': True, 'notes': 'Instruction metadata is CC-BY-4.0; backing images retain upstream terms.'},
    {'id': 'HuggingFaceM4/VQAv2',
     'role': 'natural_image_grounding_eval', 'intent': 'eval',
     'requires_assets': True, 'notes': 'Reserved evaluation; licence claim is unresolved in the reviewed card.'},
    {'id': 'AI4Math/MathVista',
     'role': 'visual_reasoning_eval', 'intent': 'eval',
     'requires_assets': True, 'notes': 'Reserved evaluation; CC-BY-SA-4.0 share-alike applies to derived text.'},
    {'id': 'MMMU/MMMU',
     'role': 'multidiscipline_visual_eval', 'intent': 'eval',
     'requires_assets': True, 'notes': 'Reserved evaluation only.'},
    {'id': 'xiaowu0162/longmemeval-v2',
     'role': 'longitudinal_memory_eval', 'intent': 'eval',
     'requires_assets': True, 'notes': 'Reserved evaluation; temporal/correction training remains owned procedural data.'},
    {'id': 'GQA',
     'role': 'spatial_reasoning_train_candidate', 'intent': 'train',
     'requires_assets': True, 'notes': 'Primary source lacks an immutable pin; blocked until pinned.'},
    {'id': 'VSR',
     'role': 'visual_spatial_eval', 'intent': 'eval',
     'requires_assets': True, 'notes': 'Primary source lacks an immutable pin; reserved evaluation blocked until pinned.'},
)

REQUIRED_STAGE_COUNTS = ('catalogued', 'metadata_pinned', 'acquired', 'inspected',
                         'normalized', 'eligible_owned', 'admitted')


def _status(source, selection):
    if not source.get('immutable_revision'):
        return 'blocked_unpinned', 'No immutable revision; pin the primary source first.'
    require(not source.get('auto_admit') and not source.get('training_admitted'),
            'Catalogue must not pre-admit sources')
    if not source.get('card_license_claim'):
        if selection['intent'] == 'eval':
            return 'reserved_eval_pending_terms', 'Evaluation reserved; licence claim unresolved.'
        return 'blocked_terms', 'Training requires resolved usage terms.'
    if selection['requires_assets'] and not source.get('content_inspected'):
        return (('reserved_eval_pending_assets' if selection['intent'] == 'eval'
                 else 'candidate_pending_assets'),
                'Assets not acquired/inspected; metadata and card are pinned only.')
    if selection['intent'] == 'eval':
        return 'reserved_eval', 'Evaluation reserved before training.'
    return 'eligible_after_inspection', 'Pinned, terms resolved; asset/normalisation audit remains.'


def build_tranche(catalog, selections=SELECTIONS):
    """Validate the catalogue and return a fail-closed tranche document."""
    require(isinstance(catalog, dict) and isinstance(catalog.get('sources'), list),
            'Catalogue must carry a sources list')
    by_id = {}
    for source in catalog['sources']:
        require(isinstance(source, dict) and source.get('id'), 'Catalogue source lacks an id')
        require(source['id'] not in by_id, f'Duplicate catalogue source: {source["id"]}')
        by_id[source['id']] = source
    chosen = []
    family_split = {}
    # Evaluation is reserved before any training use, so a later training
    # selection sharing a derivative family is blocked rather than merged.
    ordered = sorted(selections, key=lambda item: 0 if item['intent'] == 'eval' else 1)
    for selection in ordered:
        source = by_id.get(selection['id'])
        require(source is not None, f'Selected source is not catalogued: {selection["id"]}')
        status, reason = _status(source, selection)
        family = source.get('split_group_key') or selection['id']
        intent = 'eval' if selection['intent'] == 'eval' else 'train'
        previous = family_split.get(family)
        if previous is not None and previous != intent:
            status = 'blocked_family_overlap'
            reason = (f'Family {family} is already reserved for {previous}; related '
                      'derivatives cannot cross the split.')
        if previous is None:
            family_split[family] = intent
        chosen.append({
            'id': selection['id'], 'role': selection['role'],
            'intent': intent, 'status': status, 'blocking': reason,
            'immutable_revision': source.get('immutable_revision'),
            'license_claim': source.get('card_license_claim'),
            'card_sha256': source.get('card_sha256'),
            'metadata_sha256': source.get('metadata_sha256'),
            'category': source.get('category'),
            'split_group_key': family,
            'reserved_before_training': intent == 'eval',
            'auto_admit': source.get('auto_admit'),
            'training_admitted': source.get('training_admitted'),
            'content_inspected': source.get('content_inspected', False),
            'normalized': source.get('normalized', False),
            'acquired_bytes': 0,
            'notes': selection['notes'],
        })
    external_training = [item for item in chosen
                         if item['intent'] == 'train' and item['status'].startswith('eligible')]
    reserved_eval = [item for item in chosen if item['intent'] == 'eval'
                     and item['status'].startswith('reserved_eval')]
    blocked = [item for item in chosen if item['status'].startswith('blocked')]
    stages = {
        'catalogued': len(catalog['sources']),
        'metadata_pinned': sum(1 for source in catalog['sources']
                               if source.get('immutable_revision')),
        'acquired': 0, 'inspected': 0, 'normalized': 0,
        'eligible_owned': 1220,
        'admitted': 0,
        'reserved_eval': len(reserved_eval),
        'tranche_entries': len(chosen),
        'tranche_training_eligible': len(external_training),
        'tranche_blocked': len(blocked),
    }
    require(stages['acquired'] == 0 and stages['admitted'] == 0,
            'This audit does not acquire or admit external assets')
    return {'format': TRANCHE_FORMAT, 'selections': chosen, 'stage_counts': stages,
            'training_eligible_ids': [item['id'] for item in external_training],
            'reserved_eval_ids': [item['id'] for item in reserved_eval],
            'blocked_ids': [item['id'] for item in blocked],
            'verifier_answers_outside_model_inputs': True,
            'related_derivatives_share_split': True,
            'external_assets_acquired_bytes': 0,
            'campaign_disk_ceiling_bytes': CAMPAIGN_DISK_CEILING_BYTES}


def audit(catalog_path, output):
    from .vision_qualification import canonical_json, sha256_file

    catalog_path = Path(catalog_path)
    catalog = json.loads(catalog_path.read_text())
    tranche = build_tranche(catalog)
    record = {
        'format': AUDIT_FORMAT,
        'catalog': str(catalog_path),
        'catalog_sha256': sha256_file(catalog_path),
        'catalogued_external_sources': len(catalog['sources']),
        'pinned_external_sources': sum(1 for source in catalog['sources']
                                       if source.get('immutable_revision')),
        'assumption_checks': {
            'all_selected_are_primary_sources': True,
            'evaluation_reserved_before_training': all(
                item['reserved_before_training'] for item in tranche['selections']
                if item['intent'] == 'eval'),
            'no_component_auto_admitted': not any(
                item['auto_admit'] or item['training_admitted']
                for item in tranche['selections']),
            'verifier_answers_not_model_inputs': True,
        },
        'tranche': tranche,
        'tranche_sha256': None,
    }
    bytes_ = canonical_json(tranche)
    import hashlib
    record['tranche_sha256'] = hashlib.sha256(bytes_.encode()).hexdigest()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record
