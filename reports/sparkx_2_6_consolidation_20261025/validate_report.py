"""Validate documentation/protocol evidence, never model capabilities or training."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

from estimate_costs import calculate

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DOC = ROOT / 'docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md'
CONFIG = ROOT / 'configs/sparkx_2_6_4b_consolidated_plan_v2.json'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    checks: list[dict] = []
    draft_drift: list[dict] = []
    def check(name: str, success: bool, detail=None) -> None:
        checks.append({'check': name, 'passed': bool(success), 'detail': detail})
    cfg = json.loads(CONFIG.read_text())
    evidence = json.loads((HERE / 'evidence.json').read_text())
    snapshot = json.loads((HERE / 'repository_snapshot.json').read_text())
    costs = json.loads((HERE / 'analytical_costs.json').read_text())
    text = DOC.read_text()
    check('proposal_only', cfg['status'] == 'proposal_not_execution_protocol'
          and not cfg['training_authorized_by_this_plan']
          and not cfg['runtime_implemented_by_this_review'])
    check('one_canonical_plan', ROOT / cfg['canonical_document'] == DOC)
    check('costs_recomputed_exactly', costs == calculate())
    check('factorial_complete', {a['id'] for a in cfg['factorial']['arms']}
          == {f'{m}{v}{l}' for m in (0, 1) for v in (0, 1) for l in (0, 1)})
    for a in cfg['factorial']['arms']:
        check('factorial_bits_' + a['id'], a['id'] == ''.join(str(int(a[k])) for k in ('M', 'V', 'L')))
    check('churn_exceeds_active_slots_not_semantic_capacity',
          cfg['data']['main_churn_entries'] == 2*cfg['minimal_modules']['vocab']['active_slots']+32
          < cfg['minimal_modules']['bank']['minicpm_entries'])
    check('base_tables_not_cloned', not cfg['minimal_modules']['vocab']['clone_base_tables'])
    check('base_ids_immutable', cfg['minimal_modules']['vocab']['base_ids_immutable'])
    check('identity_not_physical_slot_optimizer', not cfg['minimal_modules']['vocab']['slot_optimizer_state_allowed'])
    check('causal_candidate_availability', cfg['data']['future_answer_candidate_selection_forbidden'])
    check('verifier_separate', cfg['data']['verifier_truth_not_model_input'])
    check('oracle_not_native_result', not cfg['factorial']['oracle_controls_eligible_for_native_acceptance'])
    check('retokenization_not_exact_restart', not cfg['state_contract']['raw_text_base_retokenization_is_exact_restart'])
    required_refs = {'stream_id','slot_index','slot_generation','lexeme_uid','surface_revision','encoder_revision','lexical_state_revision'}
    check('lexical_identity_fields', required_refs <= set(cfg['state_contract']['lexical_ref']))
    check('correct_power_boundary', cfg['study_budget']['power_selection_data'] == 'development_only'
          and cfg['study_budget']['confirmation_sample_size_frozen_before_final_labels'])
    check('inconclusive_allowed', cfg['study_budget']['insufficient_interval_precision_result'] == 'inconclusive')
    check('clusters_not_questions', cfg['statistics']['question_level_iid_forbidden'])
    check('no_compound_interaction_requirement', not cfg['statistics']['require_positive_three_way_interaction'])
    check('single_MPS_worker_no_fallback', cfg['resource_policy']['exclusive_MPS_worker']
          and not cfg['resource_policy']['silent_CPU_fallback'])
    gates = {g['id']: g for g in cfg['gates']}
    check('nine_explicit_gates', set(gates) == {f'G{i}' for i in range(9)})
    check('noninferiority_two_points', gates['G2']['quality_difference_lower_bound_min'] == -0.02
          and gates['G4']['unaffected_and_general_quality_difference_lower_bound_min'] == -0.02)
    check('G5_requires_component_gates', set(gates['G5']['requires']) == {'G1','G2','G3','G4'})
    stages = {s['id']: s for s in cfg['stages']}
    check('nine_explicit_stages', set(stages) == {f'P{i}' for i in range(9)})
    completed = set()
    for _ in range(len(stages)):
        completed |= {sid for sid, s in stages.items() if set(s['depends_on']) <= completed}
    check('stage_graph_complete_and_acyclic', completed == set(stages))
    for sid, stage in stages.items():
        check('stage_gates_' + sid, set(stage['gates']) <= set(gates))
        check('stage_not_claimed_executed_' + sid, stage['status'] == 'not_executed_by_this_review')
    check('joint_requires_components', set(stages['P4']['depends_on']) == {'P1','P2','P3'})
    check('optional_expansion_not_Spark_dependency', 'P6' not in stages['P8']['depends_on']
          and 'P7' not in stages['P8']['depends_on'])
    source_ids = {s['id'] for s in evidence['sources']}
    check('source_ids_unique', len(source_ids) == len(evidence['sources']))
    for source in evidence['sources']:
        sid = source['id']
        check('source_hash_' + sid, bool(re.fullmatch('[0-9a-f]{64}', source['sha256']))
              and source['snapshot_hash_rechecked'] and source['bytes'] > 0)
        check('source_scope_' + sid, bool(source['reading_scope'])
              and not source['upstream_executed'] and not source['author_results_reproduced'])
        if source['kind'] in ('paper', 'paper_abstract_only'):
            check('version_pinned_' + sid, bool(re.search(r'\d{4}\.\d{4,5}v\d+$', source['pinned_url'])))
        if source.get('local_snapshot_path'):
            check('local_upstream_hash_' + sid, digest(ROOT/source['local_snapshot_path']) == source['sha256'])
    for claim in evidence['claims']:
        check('claim_source_ids_' + claim['id'], set(claim['source_ids']) <= source_ids)
    for name, ids in cfg['evidence_basis'].items():
        check('config_source_ids_' + name, set(ids) <= source_ids)
    check('combination_labelled_unproven', any(c['id']=='C08_combination' and c['status']=='proposed_unvalidated'
          and not c['source_ids'] for c in evidence['claims']))
    known_urls = {u for s in evidence['sources'] for u in s['document_urls']}
    links = re.findall(r'\[[^\]]+\]\(([^)]+)\)', text)
    for target in sorted(set(links)):
        if target.startswith('https://'):
            check('citation_manifest_' + target, target in known_urls)
        else:
            local = (DOC.parent / unquote(target.split('#')[0])).resolve()
            check('local_link_' + target, local.exists() or local == HERE/'validation.json')
    for old in ('sparkx_2_6_4b_20261024', 'dynamic_engram_20261025', 'vocabulary_lora_20261025'):
        p = ROOT/'docs/research'/old/'README.md'
        check('historical_pointer_' + old, DOC.name in p.read_text() and 'historical research snapshot' in p.read_text())
    for row in snapshot['reviewed_files']:
        p = ROOT/row['path']
        current = digest(p) if p.exists() else None
        if row['path'] in snapshot.get('unqualified_concurrent_drafts', []):
            check('historical_draft_record_' + row['path'], p.exists()
                  and bool(re.fullmatch('[0-9a-f]{64}', row['sha256'])))
            if current != row['sha256']:
                draft_drift.append({'path':row['path'], 'observed_sha256':row['sha256'],
                                    'current_sha256':current,
                                    'scope':'Unqualified concurrent draft; no native quality or execution claim inherits to current bytes.'})
        else:
            check('snapshot_file_' + row['path'], p.exists() and current == row['sha256'])
    mini_path = ROOT/'reports/checkpoint_retirement_metadata_after_D2_20261022/minicpm5_native_adapter_qualification256_20261022/adapter_export/base/config.json'
    mini = json.loads(mini_path.read_text())
    spark = json.loads((ROOT/'reports/spark_preparation_20261025/run_01/assets/official_bf16_control/config.json').read_text())
    for name, native in (('minicpm5',mini),('sparkx',spark)):
        model = cfg['models'][name]
        for key in ('hidden_size','intermediate_size','vocab_size'):
            check('native_geometry_' + name + '_' + key, model[key] == native[key])
        check('native_layers_' + name, model['layers'] == native['num_hidden_layers'])
        check('native_tied_' + name, model['tied_base_embeddings'] == native['tie_word_embeddings'])
        check('cost_model_geometry_' + name, costs['models'][name]['unique_base_parameters'] == model['unique_parameters'])
    check('vision_v0_not_quality_claim', json.loads((ROOT/'reports/vision_v0_20261025/qualification.json').read_text())['quality_pass'] is None)
    check('memory_rejection_preserved', json.loads((ROOT/'reports/memory_reliability_v5_scorer_fix_eval02/verdict_final.json').read_text())['candidate_decision'] == 'reject')
    check('report_sizes_in_document', '544' in text and '2,621,440' in text and '2,883,584' in text)
    check('markdown_fences_balanced', len(re.findall(r'^```', text, re.M)) % 2 == 0)
    check('no_trailing_spaces', all(line.rstrip() == line for line in text.splitlines()))
    files = [DOC, CONFIG, HERE/'evidence.json', HERE/'repository_snapshot.json', HERE/'analytical_costs.json',
             HERE/'estimate_costs.py', HERE/'validate_report.py']
    failed = [c for c in checks if not c['passed']]
    result = {
        'format':'replica.sparkx_consolidation.artifact_validation.v1',
        'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Documentation links, evidence metadata, source snapshot hashes, protocol consistency and analytical arithmetic only.',
        'status':'pass' if not failed else 'fail', 'checks_passed':len(checks)-len(failed),
        'checks_total':len(checks), 'failures':failed, 'checks':checks,
        'main_document_words':len(text.split()), 'unique_markdown_links':len(set(links)),
        'source_counts':evidence['counts'],
        'repository_head_at_validation':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'model_runs':False, 'training_runs':False, 'upstream_code_executed':False,
        'native_quality_verified':False, 'full_runtime_test_suite_run':False,
        'unqualified_concurrent_draft_drift':draft_drift,
        'artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in files],
    }
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_passed','checks_total','failures','main_document_words','source_counts')}))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
