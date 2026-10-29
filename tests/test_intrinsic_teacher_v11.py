"""Broader-cache and controller failures on CPU; no native competence claims."""
import copy
import json
import time
from pathlib import Path

import pytest
import torch

from replica_lab.contextual_memory import canonical_spans, ContextualMemoryError
from replica_lab.intrinsic_memory_budget_v5 import WorkLedger, sha_json
from replica_lab.intrinsic_memory_diagnostic_v5 import DiagnosticEngine, CONTROLS
from replica_lab.intrinsic_memory_execution_v5 import hash_file
from replica_lab.intrinsic_memory_native_v5 import base_hash
from replica_lab.intrinsic_memory_teacher_v11 import CoverageTeacherEngine
from replica_lab.intrinsic_memory_worlds_v11 import worlds
from replica_lab.tiered_memory import MemoryScope
from test_intrinsic_teacher_v10 import make as make_previous
from test_tiered_memory import open_store
from test_intrinsic_coverage_v11 import candidate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def rows():
    return [{'world_id':'coverage_cpu_'+str(i), 'criterion_group':i//8,
             'question':'Badge '+str(i//2)+'?', 'text':'The badge is '+('red' if i%2==0 else 'blue')+'.',
             'answer':'red' if i%2==0 else 'blue'} for i in range(32)]


def scopes(rs=None):
    return [MemoryScope(r['world_id'], 'main', 'engineering', 1000) for r in (rs or rows())]


def make(root, arm='two'):
    old = make_previous(root, arm)
    engine = CoverageTeacherEngine(old.path, old.backend, old.spec, old.ledger,
        max_new_tokens=3, output=root/'out', base_identity=old.base_identity, fit_identity=sha_json(rows()))
    engine.stage='test'
    return engine


def qualify(engine, monkeypatch, fail_at=None):
    def answer(row, batch, scope, **kwargs):
        assert batch is None and kwargs['control']=='exact_text'
        correct = row['world_id'] != fail_at
        label = row['answer'] if correct else 'unknown'
        return {'raw_answer':label, 'token_ids':engine.backend.answer_ids(label)+[1],
                'terminated':True, 'truncated':False, 'correct':correct,
                'false_recall':False, 'first_logits':None, 'synthetic_CPU_qualification':True}
    with monkeypatch.context() as scoped:
        scoped.setattr(engine, 'generate', answer)
        return engine.prepare_teacher(rows(), scopes())


def source(e, root, index):
    r, s = rows()[index], scopes()[index]
    store=open_store(root, name='store'+str(index))
    span=canonical_spans(r['text'], record_id=r['world_id'], source_id=r['world_id'], source_epoch=1, available_at=1)[0]
    e.path.admit(store, span, scope=s)
    return e.path.read(store,'badge',scope=s),s,store


@pytest.mark.parametrize('arm',['two','full_attention_nine','all'])
def test_actual_32_targets_and_late_world_live_gradients(tmp_path, monkeypatch, arm):
    e=make(tmp_path,arm);initial=base_hash(e.path.adapter.base)
    reader_calls=[]
    handles=[p.register_forward_hook(lambda *args:reader_calls.append(True)) for p in e.path.reader.ports.values()]
    assert qualify(e,monkeypatch)
    assert not reader_calls and len(e.targets)==32
    assert e.ledger.snapshot()['reserved']['forwards']==32
    assert all(not t.log_probs.requires_grad for t in e.targets.values())
    b,s,store=source(e,tmp_path,30)
    try:
        first=e.update(rows()[30],b,s);second=e.update(rows()[30],b,s)
        assert first['gradients']['producer']['norm']==0
        assert second['gradients']['producer']['norm']>0
        assert len(second['gradients']['sites'])==len(e.path.spec.sites)
        assert all(v['norm']>0 for v in second['gradients']['sites'].values())
        assert e.ledger.snapshot()['reserved']['forwards']==34
        assert not second['teacher_forward_in_update']
        assert base_hash(e.path.adapter.base)==initial
        assert all(p.grad is None for p in e.path.adapter.base.parameters())
        assert len(e.path.assert_optimizer_ownership())==3
    finally:
        for h in handles:h.remove()
        store.close();e.path.close()


@pytest.mark.parametrize('failure',['incomplete','last_world','duplicate'])
def test_complete_teacher_evidence_required_and_failed_stage_cannot_retry(tmp_path,monkeypatch,failure):
    e=make(tmp_path)
    try:
        if failure=='last_world':
            assert not qualify(e,monkeypatch,rows()[-1]['world_id'])
            with pytest.raises(ValueError,match='complete'):qualify(e,monkeypatch)
        else:
            rs=rows();ss=scopes()
            if failure=='incomplete':rs.pop();ss.pop()
            else:rs[-1]=rs[0]
            with pytest.raises(ValueError,match='complete'):e.prepare_teacher(rs,ss)
        assert not e.targets and not e.teacher_qualified
        with pytest.raises(ValueError,match='teacher competence'):e.update(rows()[0],None,scopes()[0])
        assert e.ledger.snapshot()['reserved']['updates']==0
        assert e.ledger.snapshot()['reserved']['forwards']==0
    finally:e.path.close()


@pytest.mark.parametrize('damage',['old_version','missing_target','partial_qualification','wrong_count','changed_last_source','frozen'])
def test_cache_cannot_truncate_coverage_or_restore_stale_identity(tmp_path,monkeypatch,damage):
    e=make(tmp_path/'parent');other=make(tmp_path/'other')
    assert qualify(e,monkeypatch);cache=tmp_path/'teacher.json';pin=e.save_teacher_cache(cache);rs=rows()
    try:
        payload=json.loads(cache.read_text())
        if damage=='old_version':payload['format']='replica.intrinsic_memory.teacher_cache.v10'
        elif damage=='missing_target':del payload['targets'][rs[-1]['world_id']]
        elif damage=='partial_qualification':payload['qualification'].pop()
        elif damage=='wrong_count':payload['fit_cases']=8
        elif damage=='changed_last_source':rs[-1]['text']='Corrected source'
        else:other.freeze()
        if damage not in ('changed_last_source','frozen'):
            cache.write_text(json.dumps(payload));pin={'bytes':cache.stat().st_size,'sha256':hash_file(cache)}
        with pytest.raises(ValueError):other.load_teacher_cache(cache,pin,rs)
        assert not other.targets and not other.teacher_qualified
    finally:e.path.close();other.path.close()


@pytest.mark.parametrize('event',['correction','revocation'])
def test_current_source_authority_overrides_saved_teacher_targets(tmp_path,monkeypatch,event):
    from dataclasses import replace
    e=make(tmp_path);assert qualify(e,monkeypatch);b,s,store=source(e,tmp_path,30)
    try:
        if event=='correction':
            replacement=canonical_spans('The badge is blue.',record_id='new',source_id=rows()[30]['world_id'],source_epoch=2,available_at=2)[0]
            e.path.correct(store,b.spans[0].record_id,replace(replacement,supersedes=b.spans[0].record_id),scope=s)
        else:store.revoke_record(b.spans[0].record_id,reason='CPU_test',scope=s)
        before={k:v.clone() for k,v in e.path._tensors().items()}
        with pytest.raises(ContextualMemoryError):e.update(rows()[30],b,s)
        assert all(torch.equal(v,e.path._tensors()[k]) for k,v in before.items())
    finally:store.close();e.path.close()


@pytest.mark.parametrize('failed_group',[None,0,3])
def test_every_predeclared_fitting_group_is_required(tmp_path,monkeypatch,failed_group):
    e=make(tmp_path);rs=worlds('fit')['rows'];visited=[]
    def probe(self,group,*aligned):
        index=group[0]['criterion_group'];visited.append(index)
        assert len(group)==8 and all(r['criterion_group']==index for r in group)
        passed=index!=failed_group
        return {'rows':group,'correct':{c:8 if c=='exact_text' or passed else 7 for c in CONTROLS},
                'same_question_counterfactual_pairs':[[group[i]['world_id'],group[i+1]['world_id']] for i in range(0,8,2)],
                'counterfactual_pairs_both_correct':4 if passed else 3,'fit_memory_dependence_passed':passed}
    monkeypatch.setattr(DiagnosticEngine,'fitting_probes',probe)
    try:
        result=e.fitting_probes(rs,[None]*32,[None]*32,[None]*32)
        assert visited==[0,1,2,3]
        assert result['fit_memory_dependence_passed']==(failed_group is None)
    finally:e.path.close()


@pytest.mark.parametrize('damage',['none','second_group_five','shuffled_tie','false_recall','regroup','teacher_not_evicted'])
def test_development_uses_real_unchanged_group_scorer_without_pooling_escape(tmp_path,monkeypatch,damage):
    e=make(tmp_path);rs=worlds('development')['rows'];e.freeze()
    if damage=='regroup':rs[-1]['criterion_group']=0
    if damage=='teacher_not_evicted':e.teacher_evidence=[{}]
    def generate(row,batch,scope,*,control,**kwargs):
        index=int(row['world_id'].split('_')[-1]);group=index//8
        correct=control in ('contextual','exact_text')
        if damage=='second_group_five' and group==1 and control=='contextual':correct=index%8<5
        if damage=='shuffled_tie' and control=='shuffled':correct=True
        return {'correct':correct,'false_recall':damage=='false_recall' and index==15 and control=='contextual','first_logits':None}
    monkeypatch.setattr(e,'generate',generate)
    try:
        if damage in ('regroup','teacher_not_evicted'):
            with pytest.raises(ValueError):e.evaluate(rs,[None]*16,[None]*16)
        else:
            result=e.evaluate(rs,[None]*16,[None]*16)
            assert result['useful_recall_passed']==(damage=='none')
            if damage=='second_group_five':
                assert result['correct']['contextual']==13
                assert [g['useful_recall_passed'] for g in result['groups']]==[True,False]
    finally:e.path.close()


@pytest.mark.parametrize('failed_gate',['baseline','teacher'])
def test_native_failure_closes_fit_and_never_parses_development(tmp_path,monkeypatch,failed_gate):
    from replica_lab import intrinsic_memory_native_v11 as native
    import replica_lab.intrinsic_memory_native_v5 as shared
    e=make(tmp_path/'fixture');p=candidate(json.loads((ROOT/'configs/spark_intrinsic_alignment_v10.json').read_text()))
    opened=[]
    def data(root,pin):
        role=next(k for k,v in p['data'].items() if v==pin)
        assert role!='development';opened.append(role);return worlds(role)
    monkeypatch.setattr(native,'bound_json',data)
    monkeypatch.setattr(shared,'store_at',lambda output,world:open_store(output/'stores',name=world))
    def teacher(self,*args):
        assert failed_gate=='teacher';return False
    monkeypatch.setattr(native.TeacherEngine,'prepare_teacher',teacher)
    monkeypatch.setattr(native.TeacherEngine,'update',lambda *args:pytest.fail('dependent fit entered'))
    monkeypatch.setattr(native.TeacherEngine,'generate',lambda self,row,*args,**kwargs:{'raw_answer':row['answer'],
        'correct':failed_gate!='baseline','false_recall':False,'truncated':False,'first_logits':None})
    output=tmp_path/'controller';output.mkdir()
    ledger=WorkLedger.create(output/'ledger.json',identity='CPU_coverage_failure',plan=p['arms']['two']['budget'],deadline=time.monotonic()+120)
    result=native.run_main(tmp_path,p,'two',ledger,output,e.path,e.backend,e.spec,e.base_identity)
    assert result['status']==failed_gate+'_failed_dependent_stages_closed'
    assert opened==(['calibration','fit'] if failed_gate=='teacher' else ['calibration'])
    assert ledger.snapshot()['reserved']['updates']==0 and not result['restart_requested']
    assert not (output/'development.json').exists()


@pytest.mark.parametrize('final_fit_passed',[False,True])
def test_only_fixed_final_fit_opens_development_after_freeze(tmp_path,monkeypatch,final_fit_passed):
    from replica_lab import intrinsic_memory_native_v11 as native
    import replica_lab.intrinsic_memory_native_v5 as shared
    e=make(tmp_path/'fixture');p=candidate(json.loads((ROOT/'configs/spark_intrinsic_alignment_v10.json').read_text()))
    output=tmp_path/'controller';output.mkdir();opened=[];updates=[];prefixes=[]
    class DevelopmentBoundaryReached(Exception):pass
    def data(root,pin):
        role=next(k for k,v in p['data'].items() if v==pin);opened.append(role)
        if role=='development':
            assert final_fit_passed and len(updates)==512 and prefixes==[128,256,384,512]
            assert json.loads((output/'fitting_closed.json').read_text())['closed']
            assert all(not v.requires_grad for v in e.path._tensors().values())
            raise DevelopmentBoundaryReached()
        return worlds(role)
    monkeypatch.setattr(native,'bound_json',data)
    monkeypatch.setattr(shared,'store_at',lambda output,world:open_store(output/'stores',name=world))
    monkeypatch.setattr(native.TeacherEngine,'generate',lambda self,row,*args,**kwargs:{'raw_answer':row['answer'],
        'correct':True,'false_recall':False,'truncated':False,'first_logits':None})
    monkeypatch.setattr(native.TeacherEngine,'prepare_teacher',lambda *args:True)
    monkeypatch.setattr(native.TeacherEngine,'save_teacher_cache',lambda *args:{'sha256':'a'*64,'bytes':1})
    def update(self,row,*args):
        updates.append(row['world_id']);return {'gradients':{'producer':{'norm':1},'sites':{'3':{'output_norm':1},'7':{'output_norm':1}}}}
    monkeypatch.setattr(native.TeacherEngine,'update',update)
    def probes(self,*args):
        prefixes.append(len(updates))
        return {'correct':{c:32 for c in CONTROLS},'fit_memory_dependence_passed':len(updates)!=512 or final_fit_passed,
                'counterfactual_pairs_both_correct':16}
    monkeypatch.setattr(native.TeacherEngine,'fitting_probes',probes)
    ledger=WorkLedger.create(output/'ledger.json',identity='CPU_coverage_controller',plan=p['arms']['two']['budget'],deadline=time.monotonic()+120)
    if final_fit_passed:
        with pytest.raises(DevelopmentBoundaryReached):native.run_main(tmp_path,p,'two',ledger,output,e.path,e.backend,e.spec,e.base_identity)
    else:
        result=native.run_main(tmp_path,p,'two',ledger,output,e.path,e.backend,e.spec,e.base_identity)
        assert not result['fit_memory_dependence_passed'] and result['restart_requested']
    assert len(updates)==512 and prefixes==[128,256,384,512]
    assert updates==[r['world_id'] for r in worlds('fit')['rows']]*16
    assert opened==(['calibration','fit','development'] if final_fit_passed else ['calibration','fit'])
    assert not (output/'development.json').exists()
