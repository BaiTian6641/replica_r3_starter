#!/usr/bin/env python3
"""Prepare broader source coverage, fixed group gates and independent per-arm budgets.

No original weights or historical label files are loaded. A completed v10
receipt supplies the comparison; its exposed development split is retired.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from replica_lab.intrinsic_memory_execution_v5 import binding, local_closure, bound_json
from replica_lab.intrinsic_memory_coverage_v11 import (
    FORMAT, PACKET, TRAINING, TEACHER, COVERAGE, BUDGET_DERIVATION, PREFIXES, plan, totals, validate_protocol,
)
from replica_lab.intrinsic_memory_worlds_v11 import worlds, validate_corpus, public_split_metadata
from replica_lab.intrinsic_memory_execution_v11 import validate_predecessor, validate_fresh_data
from prepare_spark_intrinsic_explicit_evidence_v6 import save_new

REPORT = 'reports/intrinsic_memory_coverage_v11'
DATA = 'data/intrinsic_memory_coverage_v11_20261029'
CONFIG = 'configs/spark_intrinsic_coverage_v11.json'


def prepare(root=ROOT):
    if (root/CONFIG).exists() or (root/DATA).exists():
        raise ValueError('v11 already prepared; no overwrite')
    previous = 'configs/spark_intrinsic_alignment_v10.json'
    protocol = deepcopy(json.loads((root/previous).read_text()))
    predecessor_result = protocol['arms']['two']['output_root']+'/result.json'
    observed = json.loads((root/predecessor_result).read_text())
    if (observed.get('native_competence') is not True
            or observed.get('main', {}).get('fit_memory_dependence_passed') is not True
            or observed.get('useful_recall') is not False):
        raise ValueError('coverage study requires completed learned-fitting but failed-utility reference')
    protocol.update(format=FORMAT, training=deepcopy(TRAINING), teacher=deepcopy(TEACHER),
                    coverage=deepcopy(COVERAGE), budget_derivation=deepcopy(BUDGET_DERIVATION),
                    fit_probe_prefixes=list(PREFIXES),
                    fit_curve_selection='all_512_updates_fixed_final_prefix_all_groups_only_opens_development',
                    diagnostic_predecessor=binding(root, predecessor_result),
                    prior_development_exposure=binding(root,'reports/intrinsic_memory_alignment_v10/development_exposure.json'))
    protocol['data'] = {}
    manifest = {'format':'replica.intrinsic_memory.coverage_split_manifest.v11',
                'generator_policy':'predeclared_source_order_revision_oracle_no_generated_explanations','splits':{}}
    for split in ('calibration','fit','development'):
        document = validate_corpus(worlds(split), split)
        name=f'{DATA}/{split}.json';save_new(root,name,document)
        protocol['data'][split]=binding(root,name)
        manifest['splits'][split]={'file':protocol['data'][split], 'public_identity':public_split_metadata(split)}
    save_new(root,DATA+'/split_manifest.json',manifest)
    protocol['data_manifest']=binding(root,DATA+'/split_manifest.json')
    entries=['scripts/run_spark_intrinsic_coverage_v11.py','scripts/prepare_spark_intrinsic_coverage_v11.py',
             'src/replica_lab/intrinsic_memory_native_v11.py',
             'scripts/launch_spark_intrinsic_readout_environment_v1.py']
    save_new(root,REPORT+'/implementation_v11.json',{
        'format':'replica.intrinsic_memory.local_execution_closure.v11','entries':entries,
        'files':local_closure(root,entries),
        'policy':'broader_teacher_corpus_fixed_group_gates_no_old_execution_mutation'})
    protocol['implementation']=binding(root,REPORT+'/implementation_v11.json')
    protocol['density_controls']['initial_study']='feasibility_two_predeclared_rules_one_seed'
    protocol['experiment_variant']={
        'id':'balanced_source_coverage_v11','predecessor':binding(root,previous),
        'reason':'v10 learns all fitting memories but transfers incompletely; test broader balanced source coverage under unchanged model, objective, reader and per-group gates',
        'changed':['32_fitting_worlds_two_predeclared_rules','512_updates_same_16_passes_per_world',
                   'new_calibration_and_16_fresh_development_worlds','all_group_conjunction',
                   'complete_teacher_corpus_cache_and_split_reuse_guard','per_arm_work_and_artifact_budget',
                   'execution_output_identity_and_explicit_environment_validation_receipt_field'],
        'unchanged':['official_BF16_source','producer_and_reader_mechanism','3_7_control',
                     'top64_KL_temperature1_no_CE_mix','lr_0.0001','NativeFusedAdamW','initialization_seed17',
                     'four_fixed_probes_no_best_prefix_selection','per_group_fit_and_utility_criteria',
                     'generation_template_scorer_context_token_limits','physical_resource_limits','7200_second_outer_wall'],
        'historical_results_rewritten':False,'prior_development_reused':False,
        'isolated_causal_attribution_to_coverage_alone':False,
        'scope_limit':'more_worlds_and_two_rules_increase_total_exposure_at_fixed_passes; no matched_total_cost_or_independent_seed_claim',
        'all_choices_fixed_before_v11_native_answers':True,'teacher_updates':0,'persistent_P_updates':0,
    }
    for arm,spec in protocol['arms'].items():
        spec.update(output_root=f'{REPORT}/native_bf16_{arm}_v11_attempt01',
                    budget=plan(arm),total_reservations=totals(plan(arm)))
    validate_protocol(protocol);validate_predecessor(root,protocol);validate_fresh_data(root,protocol)
    save_new(root,CONFIG,protocol)
    for arm,spec in protocol['arms'].items():
        packet=f'{REPORT}/native_packet_{arm}_v11.json';authorization=f'{REPORT}/authorization_{arm}_v11.json'
        save_new(root,packet,{'format':PACKET,'experiment_version':11,'native_engine_version':11,
            'protocol':binding(root,CONFIG),'arm':arm,'attempts':1,'retries':0,
            'output_root':spec['output_root'],'authorization_required':authorization,
            'authorization_status':'external_receipt_required',
            'command':['.venv-spark-bf16/bin/python','scripts/run_spark_intrinsic_coverage_v11.py',
                       '--packet',packet,'--authorization',authorization,'--launch']})
    print(json.dumps({'prepared':True,'native_run':False,'native_engine_version':11,
                      'two_site_maximums':protocol['arms']['two']['total_reservations']},indent=2))


if __name__=='__main__':prepare()
