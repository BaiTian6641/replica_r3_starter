"""Fictional C16 fidelity harness: separate metrics and baseline comparison."""
from replica_lab.replica.fidelity import (BASELINES, FidelityCase, FidelityOutput,
                                          evaluate_baselines, score_case)

def case():
    return FidelityCase('c1','choice','Fictional prompt',('e1','e2'),
      expected_choice='recoverable',requires_clarification=False,
      technical_expected=True,style_tag='plain')

def output(choice='recoverable',response='The plan is recoverable.',evidence=('e1','e2'),technical=True):
    return FidelityOutput('c1',response,evidence,choice=choice,technical_passed=technical,
                          style_tags=('plain',),branch_event_ids=('b1',))

def test_metrics_are_separate_and_no_fidelity_claim():
    score=score_case(case(),output())
    assert score['choice_correct'] is True and score['technical_correct'] is True
    assert score['provenance_coverage']==1.0 and score['style_match'] is True
    assert score['authority'].startswith('score_only')
    bad=score_case(case(),output(choice='safe',evidence=('e1',),technical=False))
    assert bad['choice_correct'] is False and bad['technical_correct'] is False
    assert bad['missing_evidence_ids']==['e2']

def test_baselines_require_all_and_fictional_only():
    value=case(); predictions={name:{'c1':output()} for name in BASELINES}
    report=evaluate_baselines([value],predictions)
    assert report['personal_fidelity_claim'] is False and set(report['baselines'])==set(BASELINES)
    try:
        FidelityCase('x','f','p',('e1',),provenance='authentic')
    except ValueError:
        pass
    else:
        raise AssertionError('authentic case must not be silently accepted')
