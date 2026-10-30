"""Summarize terminal raw evidence only; never load a model or unopened labels."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/intrinsic_memory_coverage_v11'
OUTPUT=REPORT/'native_bf16_two_v11_attempt01'

def read(name,default=None):
    path=OUTPUT/name
    return json.loads(path.read_text()) if path.exists() else default

def pin(path):
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def answer_counts(rows,passed):
    return {'cases':len(rows),'correct':sum(bool(r['correct']) for r in rows),
            'false_recall':sum(bool(r['false_recall']) for r in rows),
            'truncated':sum(bool(r['truncated']) for r in rows),
            'terminated':sum(bool(r.get('terminated',False)) for r in rows),'passed':passed}

def main():
    supervision=read('supervision/receipt.json');assert supervision['process_group_exit_confirmed']
    result=read('result.json',{});ledger=read('work_ledger.json');main=result.get('main',read('main_result.json',{}))
    baseline=read('baseline.json',{'rows':[]})['rows']
    teacher=read('teacher_qualification.json',{'rows':[]})['rows']
    fit=read('fit.json',{'updates':[]})['updates'];probe=read('fit_probes_512.json')
    restart=read('restart_result.json');dev=read('development.json');correction=read('correction.json')
    telemetry=[]
    for name in ('main.telemetry.jsonl','restart.telemetry.jsonl'):
        path=OUTPUT/name
        if path.exists():telemetry.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    capacity=read('capacity_guard.json',result.get('capacity',{}))
    failures=[e for e in ledger['events'] if e['status']!='completed']
    expected=[e for e in failures if e['status']=='failed' and e['stage']=='correction'
              and e['operation']=='retrieval' and e.get('detail',{}).get('error')=='ContextualReadFailure'
              and correction and correction['revoked_retrieval_rejected']]
    unexpected=[e for e in failures if e not in expected]
    if result and main.get('status')=='completed_frozen_fit_diagnostic':
        assert len(fit)==512 and probe is not None and restart is not None
        assert ledger['reserved']['updates']==513
        assert restart['fresh_process_logits_exact'] and restart['actual_continuation_updates']==1
        assert main['original_weights_unchanged'] and restart['original_weights_unchanged']
        assert len(teacher)==32
    if dev:
        assert main['fit_memory_dependence_passed'] and read('fitting_closed.json')['updates']==512
        assert len(dev['rows'])==16 and len(dev['groups'])==2
    curve=read('fit_curve.json',{'prefixes':[]})['prefixes']
    # A later process interruption must not erase an already completed gate.
    # Fallback requires both its durable passed stage and complete raw evidence.
    def recorded_gate(name, rows, count, flag):
        if flag in main:
            return main[flag]
        return (ledger['stages'].get(name,{}).get('status')=='passed' and len(rows)==count
                and all(r['correct'] and not r['false_recall'] and not r['truncated'] for r in rows))
    fit_summary={'updates_recorded':len(fit),'updates_reserved':sum(v['reserved']['updates'] for k,v in ledger['stages'].items() if k.startswith('fit_')),
                 'curve':curve,'fixed_final_prefix':512,'no_early_stop_or_best_prefix_selection':True,
                 'memory_dependence_passed':main.get('fit_memory_dependence_passed',False)}
    if probe:
        fit_summary.update(groups=probe['groups'],correct=probe['correct'],
            live_durable_exact=all(row['round_trip_exact'] for row in probe['rows']),
            all_controls_fit_generation_prefix_exact=all(v['fit_generation_prefix_logits_exact'] for row in probe['rows'] for v in row['controls'].values()),
            contextual_first_token_correct=sum(row['controls']['contextual']['first_token_correct'] for row in probe['rows']),
            contextual_terminated=sum(row['controls']['contextual']['terminated'] for row in probe['rows']))
    if len(fit)>1:
        fit_summary['gradients_after_zero_init']=fit[1]['gradients']
        fit_summary['loss_prefixes']=[{'update':i,**{k:fit[i-1][k] for k in ('loss','kl_loss','ce_loss','word_loss','eos_loss','teacher_retained_mass_mean')}} for i in (1,2,32,64,128,256,384,512) if i<=len(fit)]
        fit_summary['all_recorded_losses_finite']=all(math.isfinite(u['loss']) for u in fit)
    summary={'format':'replica.intrinsic_memory.native_attempt_summary.v11','source_kind':'official_bf16',
        'packet':pin(REPORT/'native_packet_two_v11.json'),'result':pin(OUTPUT/'result.json') if result else None,
        'execution_completed':bool(result and supervision['child_exit_code']==0),'approval_consumed':True,
        'standing_testing_authorization_remains_applicable_to_new_bounded_packets':True,
        'competence':answer_counts(baseline,recorded_gate('baseline',baseline,8,'competence_passed')),
        'teacher_competence':answer_counts(teacher,recorded_gate('teacher',teacher,32,'teacher_competence_passed')),
        'fit':fit_summary,'development_opened':dev is not None,
        'development':None if dev is None else {k:v for k,v in dev.items() if k!='rows'},
        'correction':correction,'useful_recall':result.get('useful_recall',False),
        'density_scientific_gate_passed':result.get('useful_recall',False),'density_launched':False,
        'restart':None if restart is None else {
            'fresh_logits_exact':restart['fresh_process_logits_exact'],'actual_continuation_updates':restart['actual_continuation_updates'],
            'original_weights_unchanged':restart['original_weights_unchanged'],'uninterrupted_equivalence_measured':False,
            'teacher_cache_restored':restart['teacher_cache_restored'],'teacher_forward_in_update':restart['update']['teacher_forward_in_update']},
        'original_weights_unchanged':main.get('original_weights_unchanged',False),'P_updates':0,'promotion':False,'RSI':False,
        'expected_refusals':[{'stage':e['stage'],'operation':e['operation'],'detail':e['detail']} for e in expected],
        'unexpected_or_uncertain_events':[{'stage':e['stage'],'operation':e['operation'],'status':e['status'],'detail':e.get('detail')} for e in unexpected],
        'resources':{'outer_wall_seconds':supervision['elapsed_seconds'],'peak_driver_bytes':capacity.get('peak_driver_bytes'),
            'peak_rss_bytes':max((x['rss_bytes'] for x in telemetry),default=None),
            'peak_swap_growth_bytes':capacity.get('peak_swap_growth_bytes'),'minimum_host_free_percent':capacity.get('minimum_observed_free_percent'),
            'artifact_bytes':sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file()),
            'reserved_work':ledger['reserved'],'audit_serialized_write_bytes':ledger['audit_io']['serialized_write_bytes'],
            'source_bytes_hashed_inside_attempt':sum((e.get('detail') or {}).get('source_bytes_hashed',0) for e in ledger['events']),
            'stages':{k:{key:value for key,value in v.items() if key in ('status','elapsed_seconds','reserved')} for k,v in ledger['stages'].items()},
            'physical_memory_note':'MPS driver and RSS overlap; sampled maxima and host-wide free/swap are not exclusive allocations.',
            'I/O_note':'Serialized/file byte accounting, not measured physical SSD write amplification.'},
        'claim_limits':['Two fixed scripted rules and one seed; no heldout-rule or broad generalization claim.',
                        'Coverage and total exposure change together; no matched-cost causal attribution.',
                        'Frozen-base/extension continuation is not persistent P learning, identity fidelity or RSI.']}
    cache=OUTPUT/'teacher_targets.json'
    if cache.exists():summary['teacher_targets']={'cache':pin(cache),'bytes':cache.stat().st_size,'immutable_detached':True,
        'evicted_before_development':bool(read('fitting_closed.json')),'original_teacher_updates':0,
        'cache_reused_for_student_updates':sum(u['teacher_cache_hit'] for u in fit)+(int(restart['update']['teacher_cache_hit']) if restart else 0)}
    destination=REPORT/'native_attempt01_summary.json';assert not destination.exists()
    destination.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    names=[p for p in OUTPUT.iterdir() if p.is_file() and p.name!='.DS_Store' and p.suffix!='.lock']
    for sub in ('supervision','checkpoint'):
        names.extend(p for p in (OUTPUT/sub).rglob('*') if p.is_file() and p.suffix=='.json')
    evidence={'format':'replica.intrinsic_memory.raw_evidence_manifest.v11','terminal_only':True,
              'files':[pin(p) for p in sorted(set(names))],'binary_extensions_and_stores':'preserved_locally_not_committed','source_model_weights_exported':False}
    (REPORT/'native_attempt01_evidence.json').write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('execution_completed','competence','teacher_competence','development_opened','useful_recall','restart')},indent=2))
    print(json.dumps({k:v for k,v in summary['resources'].items() if k!='stages'},indent=2))

if __name__=='__main__':main()
