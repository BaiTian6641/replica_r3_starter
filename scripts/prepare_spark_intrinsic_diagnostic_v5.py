#!/usr/bin/env python3
"""Deterministic, review-only v5 data/protocol/packet preparation; no authorization."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replica_lab.intrinsic_memory_execution_v5 import (FORMAT,PACKET,ARMS,binding,local_closure,validate_protocol)
from replica_lab.intrinsic_memory_budget_v5 import COUNTERS,sha_json
from replica_lab.execution_manifest import build_execution_manifest

REPORT='reports/intrinsic_memory_successor_v5'
DATA='data/intrinsic_memory_v5_20261029'
CONFIG='configs/spark_intrinsic_diagnostic_v5.json'


def save(relative,value):
    path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def data(split):
    # Distinct named worlds, rule families and labels fixed before execution.
    # This is one deterministic feasibility corpus, not independent-seed proof.
    families={'calibration':['badge','lamp','flag','signal'],
              'fit':['seal','ribbon','beacon','marker'],
              'development':['token','panel','lantern','emblem']}[split]
    labels={'calibration':['copper','silver'],'fit':['red','blue'],'development':['green','gold']}[split]
    rows=[]
    for i in range(8):
        noun=families[i//2];word=labels[i%2];unknown=(split!='fit' and i==7)
        text=f'The {noun} is {word}. This report contains the complete known facts.'
        question=f'What color is the {noun}?' if not unknown else f'What color is the missing {noun}?'
        rows.append({'world_id':f'v5_{split}_{i:02d}','family':noun,'text':text,'question':question,
            'answer':'unknown' if unknown else word,'source_kind':'scripted_complete_world',
            'verification':'deterministic_world_oracle_not_generated_explanation','chronology':1})
    return {'format':'replica.intrinsic_memory.fresh_worlds.v5','split':split,'seed':2026102905,
            'provenance':'deterministic_code_defined_worlds_v5','rows':rows,
            'historical_attempt03_data_used':False,'sealed_final':False}


def stage(wall,**counts):return {**{k:0 for k in COUNTERS},'wall_seconds':wall,**counts}


def plan(arm):
    checkpoint={'two':128,'full_attention_nine':384,'all':1024}[arm]*1024**2
    # Each autoregressive token uses one complete uncached 256-position forward.
    # Reservations charge the maximum emitted tokens (EOS included), padded input
    # positions and failed work. No refund for early EOS.
    return {
      'admission':stage(120,artifact_bytes=32*1024**2,writes=1),
      'load':stage(600,model_loads=1),
      'baseline':stage(600,forwards=8*8,input_tokens=8*8*256,generated_tokens=8*8),
      'numerical':stage(120,forwards=2,input_tokens=2*256,source_encodes=1),
      'prepare':stage(120,source_encodes=8,retrievals=8,writes=16),
      'fit':stage(1200,forwards=32,backwards=32,updates=32,input_tokens=32*256,source_encodes=32),
      'fit_probes':stage(2100,forwards=8*(2+6*(1+8)),input_tokens=8*(2+6*(1+8))*256,
           generated_tokens=8*6*8,source_encodes=24,retrievals=8,writes=16),
      'checkpoint':stage(180,forwards=1,input_tokens=256,source_encodes=1,artifact_bytes=checkpoint,writes=1),
      'development':stage(900,forwards=8*6*8,input_tokens=8*6*8*256,generated_tokens=8*6*8,
           source_encodes=8,retrievals=8,writes=16),
      'correction':stage(180,forwards=3*8,input_tokens=3*8*256,generated_tokens=3*8,source_encodes=1,retrievals=2,writes=4),
      'restart':stage(900,forwards=2,backwards=1,updates=1,input_tokens=2*256,model_loads=1,
           source_encodes=2,retrievals=1,writes=1),
    }


def prepare():
    for split in ('calibration','fit','development'):save(DATA+'/'+split+'.json',data(split))
    source=json.loads((ROOT/REPORT/'official_bf16_source.json').read_text())
    save(REPORT+'/source_execution_v5.json',build_execution_manifest(ROOT/source['source_root'],artifact_sha256=sha_json(source['files'])))
    entries=['scripts/run_spark_intrinsic_diagnostic_v5.py','scripts/prepare_spark_intrinsic_diagnostic_v5.py',
             'src/replica_lab/intrinsic_memory_native_v5.py','src/replica_lab/intrinsic_memory_diagnostic_v5.py',
             'scripts/launch_spark_intrinsic_readout_environment_v1.py']
    implementation={'format':'replica.intrinsic_memory.local_execution_closure.v5','entries':entries,
                    'files':local_closure(ROOT,entries),'policy':'all_static_local_dependencies_plus_explicit_dynamic_children'}
    save(REPORT+'/implementation_v5.json',implementation)
    protocol={'format':FORMAT,'source_kind':'official_bf16','source_repo':'XHToken/Spark-X2.6-4B',
      'source_revision':'66e3ae7bee54e0a523d692cb8ef841f35409807b',
      'base_dtype':'bfloat16','extension_dtype':'float32','optimizer':'NativeFusedAdamW',
      'placement':'block_input_query_to_post_block_residual','slots':8,'slot_policy':'fixed_eight_masked_padding_v5',
      'persistent_P_updates':0,'persistent_capacity':'existing_P1_no_expansion','automatic_promotion':False,
      'source':binding(ROOT,REPORT+'/official_bf16_source.json'),
      'source_execution':binding(ROOT,REPORT+'/source_execution_v5.json'),
      'implementation':binding(ROOT,REPORT+'/implementation_v5.json'),
      'environment':binding(ROOT,'configs/spark_intrinsic_execution_environment_v1.json'),
      'data':{s:binding(ROOT,DATA+'/'+s+'.json') for s in ('calibration','fit','development')},
      'competence':{'cases':8,'correct_min':8,'false_recall_max':0,'truncation_max':0,'weights':'frozen_original_extensions_disabled'},
      'fit_gate':{'cases':8,'complete_correct_min':8,'first_token_correct_min':8,'termination_required':True,
           'prefix_logits_exact':True,'better_than':['wrong','zero','absent']},
      'utility':{'cases':8,'contextual_correct_min':6,'false_recall_max':0,'exact_text_deficit_max':1,
           'better_than':['wrong','shuffled','zero','absent']},
      'training':{'updates':32,'restart_updates':1,'lr':0.001,'seed':17,'batch_size':1,
           'schedule':'fixed_round_robin_8_worlds_4_passes','teacher_updates':0},
      'generation':{'context_tokens':256,'max_new_tokens':8,'thinking':'none','padding':'right',
           'decode':'greedy_uncached_no_sampling','stop_token_ids':[1],'eos_token_id':1,'pad_token_id':2,
           'label_alignment':'logit_i_predicts_token_i_plus_1_no_additional_shift'},
      'budget_derivation':{'baseline_forwards':'8 worlds * 8 tokens',
           'fit_probe_forwards':'8 worlds * (2 live/durable parity + 6 controls * (1 teacher-forced + 8 greedy tokens)) = 448',
           'development_forwards':'8 worlds * 6 controls * 8 tokens = 384',
           'correction_forwards':'corrected contextual, corrected exact, revoked absent * 8 tokens = 24',
           'failed_or_interrupted_work':'reserved_before_dispatch_no_refunds',
           'artifact_accounting':'32 MiB audit/store envelope plus separately reserved extension+optimizer checkpoint; guard observes total physical file sizes',
           'I/O_costs':'logical payload bytes, file calls, serialized bytes, physical artifact growth and synchronized wall separately; not physical device write amplification'},
      'density_controls':{'same_producer_initialization':True,'same_common_reader_initialization':True,
           'selected_value_slots':4,'masked_slots':4,'writes_per_forward':0,'retrievals_per_forward':0,
           'independent_native_qualification_per_arm':True,'initial_study':'feasibility_one_family_seed',
           'execution_prerequisite_for_nine_and_all':'qualified_current_two_site_result_bound_in_exact_authorization',
           'matched_parameter_and_total_cost_followups':'required_before_attributing_gains_to_density'},
      'stages_closed_after_failed_fit':['development','correction','formation','P_learning','promotion'],
      'numerical_restart_on_failed_fit':'allowed diagnostic, one update already budgeted; no generalization claim',
      'arms':{}}
    for arm,sites in ARMS.items():
        budget=plan(arm);totals={k:sum(s[k] for s in budget.values()) for k in next(iter(budget.values()))}
        protocol['arms'][arm]={'sites':sites,'budget':budget,'total_reservations':totals,'wall_seconds':7200,
             'output_root':REPORT+'/native_bf16_'+arm+'_v5_attempt01',
             'resource_limits':{'driver_bytes':50*1024**3,'swap_growth_bytes':512*1024**2,
                  'minimum_host_free_percent':15,'sustained_host_samples':3,'telemetry_timeout_seconds':30}}
    validate_protocol(protocol);save(CONFIG,protocol)
    for arm in ARMS:
        relative=REPORT+'/native_packet_'+arm+'_v5.json'
        receipt=REPORT+'/authorization_'+arm+'_v5.json'
        packet={'format':PACKET,'protocol':binding(ROOT,CONFIG),'arm':arm,'attempts':1,'retries':0,
             'output_root':protocol['arms'][arm]['output_root'],'authorization_required':receipt,
             'command':['.venv-spark-bf16/bin/python','scripts/run_spark_intrinsic_diagnostic_v5.py',
                        '--packet',relative,'--authorization',receipt,'--launch'],
             'authorization_status':'absent_prepared_only','all_historical_approvals_consumed':True}
        save(relative,packet)
    print(json.dumps({'prepared':True,'model_loaded':False,'authorized':False,
                     'execution_files':len(implementation['files']),'per_arm':protocol['arms']},indent=2))


if __name__=='__main__':prepare()
