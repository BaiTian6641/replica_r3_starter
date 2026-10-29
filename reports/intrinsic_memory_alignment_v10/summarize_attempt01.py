from pathlib import Path
import json,hashlib,math
root=Path(__file__).resolve().parents[2]
p=root/'reports/intrinsic_memory_alignment_v10/native_bf16_two_v10_attempt01'
def read(name):return json.loads((p/name).read_text())
def pin(path):
 f=root/path
 return {'path':path,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
r=read('result.json');s=read('supervision/receipt.json');l=read('work_ledger.json');b=read('baseline.json')['rows'];f=read('fit_probes_128.json');fit=read('fit.json')['updates'];restart=read('restart_result.json')
assert s['process_group_exit_confirmed'] and s['child_exit_code']==0
assert l['reserved']['updates']==129 and len(fit)==128
refusals=[e for e in l['events'] if e['status']!='completed']
assert all(e['status']=='failed' and e['stage']=='correction' and e['operation']=='retrieval'
           and e['detail']['error']=='ContextualReadFailure' for e in refusals)
assert not refusals or read('correction.json')['revoked_retrieval_rejected']
assert not (p/'failure.json').exists()
telemetry=[]
for name in ['main.telemetry.jsonl','restart.telemetry.jsonl']:
 telemetry += [json.loads(line) for line in (p/name).read_text().splitlines()]
dev=read('development.json') if (p/'development.json').exists() else None
summary={
 'format':'replica.intrinsic_memory.native_attempt_summary.v10',
 'source_kind':'official_bf16',
 'packet':pin('reports/intrinsic_memory_alignment_v10/native_packet_two_v10.json'),
 'result':pin('reports/intrinsic_memory_alignment_v10/native_bf16_two_v10_attempt01/result.json'),
 'approval_consumed':True,'standing_testing_authorization_remains_applicable_to_new_bounded_packets':True,
 'competence':{'cases':len(b),'correct':sum(x['correct'] for x in b),'false_recall':sum(x['false_recall'] for x in b),'truncated':sum(x['truncated'] for x in b),'passed':r['native_competence']},
 'teacher_competence':{'cases':len(read('teacher_qualification.json')['rows']),'correct':sum(x['correct'] for x in read('teacher_qualification.json')['rows']),'false_recall':sum(x['false_recall'] for x in read('teacher_qualification.json')['rows']),'truncated':sum(x['truncated'] for x in read('teacher_qualification.json')['rows']),'terminated':sum(x['terminated'] for x in read('teacher_qualification.json')['rows']),'passed':r['main']['teacher_competence_passed']},
 'teacher_targets':{'cache':pin('reports/intrinsic_memory_alignment_v10/native_bf16_two_v10_attempt01/teacher_targets.json'),'bytes':(p/'teacher_targets.json').stat().st_size,'immutable_detached':True,'evicted_before_development':True,'original_teacher_updates':0,'cache_reused_for_student_updates':sum(u['teacher_cache_hit'] for u in fit)+int(restart['update']['teacher_cache_hit'])},
 'fit':{'updates':128,'curve':read('fit_curve.json')['prefixes'],'fixed_final_prefix':128,'no_early_stop_or_best_prefix_selection':True,
   'memory_dependence_passed':f['fit_memory_dependence_passed'],
   'live_durable_exact':all(x['round_trip_exact'] for x in f['rows']),
   'all_controls_fit_generation_prefix_exact':all(v['fit_generation_prefix_logits_exact'] for x in f['rows'] for v in x['controls'].values()),
   'contextual_first_token_correct':sum(x['controls']['contextual']['first_token_correct'] for x in f['rows']),
   'contextual_terminated':sum(x['controls']['contextual']['terminated'] for x in f['rows']),
   'gradients_after_zero_init':fit[1]['gradients'],
   'loss_prefixes':[{'update':i,**{k:fit[i-1][k] for k in ['loss','kl_loss','ce_loss','word_loss','eos_loss','teacher_retained_mass_mean']}} for i in [1,2,8,16,24,32,64,96,128]]},
 'development_opened':dev is not None,'development':None if dev is None else {k:v for k,v in dev.items() if k!='rows'},
 'correction':read('correction.json') if (p/'correction.json').exists() else None,
 'useful_recall':r['useful_recall'],'density_scientific_gate_passed':r['useful_recall'], 'density_launched':False,
 'restart':{'fresh_logits_exact':restart['fresh_process_logits_exact'],'actual_continuation_updates':restart['actual_continuation_updates'],
   'original_weights_unchanged':restart['original_weights_unchanged'],'uninterrupted_equivalence_measured':False,'teacher_cache_restored':restart['teacher_cache_restored'],'teacher_forward_in_update':restart['update']['teacher_forward_in_update']},
 'original_weights_unchanged':r['main']['original_weights_unchanged'],
 'P_updates':0,'promotion':False,'RSI':False,
 'expected_refusals':[{'stage':e['stage'],'operation':e['operation'],'detail':e['detail']} for e in refusals],
 'resources':{'outer_wall_seconds':s['elapsed_seconds'],'peak_driver_bytes':r['capacity']['peak_driver_bytes'],
   'peak_rss_bytes':max(x['rss_bytes'] for x in telemetry),'peak_swap_growth_bytes':r['capacity']['peak_swap_growth_bytes'],
   'minimum_host_free_percent':r['capacity']['minimum_observed_free_percent'],
   'artifact_bytes':sum(x.stat().st_size for x in p.rglob('*') if x.is_file()),
   'reserved_work':l['reserved'],'audit_serialized_write_bytes':l['audit_io']['serialized_write_bytes'],
   'source_bytes_hashed_inside_attempt':sum(x['detail'].get('source_bytes_hashed',0) for x in l['events']),
   'stages':{k:{kk:vv for kk,vv in v.items() if kk in ['status','elapsed_seconds','reserved']} for k,v in l['stages'].items()},
 },
 'next_experiment':'Review this fixed result before selecting a separately bound successor. Native density requires useful matching two-site evidence; persistent P remains closed.'
}
(root/'reports/intrinsic_memory_alignment_v10/native_attempt01_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:summary[k] for k in ('competence','development_opened','useful_recall','restart')},indent=2));print(json.dumps({k:v for k,v in summary['resources'].items() if k!='stages'},indent=2))

# Post-run analysis uses only already-opened fitting evidence. It never imports
# model code, starts a worker, opens development data or selects a checkpoint.
cache=read('teacher_targets.json')
reference=root/'reports/intrinsic_memory_step_v9/native_bf16_two_v9_attempt01'
v9curve=json.loads((reference/'fit_curve.json').read_text())
v9probe=json.loads((reference/'fit_probes_128.json').read_text())
v10protocol=json.loads((root/'configs/spark_intrinsic_alignment_v10.json').read_text())
v9protocol=json.loads((root/'configs/spark_intrinsic_smaller_step_v9.json').read_text())
assert v10protocol['data']==v9protocol['data']
assert v10protocol['training']==v9protocol['training']
rows=[]
for world,target in cache['targets'].items():
 probabilities=[[math.exp(x) for x in logs] for logs in target['log_probs']]
 rows.append({'world_id':world, 'retained_mass':target['retained_mass'],
   'response_token_ids':target['provenance']['response_token_ids'],
   'response_entropy_nats':[sum(-prob*log for prob,log in zip(probs,logs)) for probs,logs in zip(probabilities,target['log_probs'])],
   'canonical_response_probability':[probs[ids.index(token)] if token in ids else 0 for probs,ids,token in zip(probabilities,target['indices'],target['provenance']['response_token_ids'])]})
probes=[]
assert len(v9probe['rows'])==len(f['rows'])==8
for previous,current in zip(v9probe['rows'],f['rows']):
 assert previous['world_id']==current['world_id']
 probes.append({'world_id':current['world_id'],
   'direct_CE_answer':previous['controls']['contextual']['raw_answer'],
   'teacher_aligned_answer':current['controls']['contextual']['raw_answer'],
   'direct_CE_correct':previous['controls']['contextual']['correct'],
   'teacher_aligned_correct':current['controls']['contextual']['correct'],
   'v10_wrong_answer':current['controls']['wrong']['raw_answer'],
   'v10_shuffled_answer':current['controls']['shuffled']['raw_answer'],
   'v10_shuffled_logit_max_delta':current['controls']['shuffled']['logit_max_delta_from_correct']})
comparison={'format':'replica.intrinsic_memory.teacher_reference_comparison.v10',
 'scope':'posthoc_fitting_only_diagnosis_no_checkpoint_selection_or_development_access',
 'reference':pin('reports/intrinsic_memory_step_v9/native_attempt01_summary.json'),
 'candidate':pin('reports/intrinsic_memory_alignment_v10/native_attempt01_summary.json'),
 'matched_data':v10protocol['data'],'matched_training':v10protocol['training'],
 'direct_CE_curve':v9curve['prefixes'],'teacher_alignment_curve':summary['fit']['curve'],
 'teacher_target_statistics':rows,'final_fitting_answers':probes,
 'independent_world_families':1,'seeds':1,
 'limits':['This small fitting-only comparison does not establish generalization or statistical superiority.',
 'Identical decoded controls can still have different logits.',
 'The target-distribution analysis does not prove the cause of any fitting failure.',
 'Earlier prefixes are diagnostics; only the fixed final update is eligible for the fitting gate.']}
(root/'reports/intrinsic_memory_alignment_v10/direct_ce_comparison.json').write_text(json.dumps(comparison,indent=2,sort_keys=True)+'\n')
