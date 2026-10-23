from replica_lab.runtime.director_data import load_registry,validate_registry
ROOT=Path=__import__('pathlib').Path
def test_registry_defers_with_current_counts():
    value=load_registry(ROOT('configs/c13_checked_proposals_v1.json'))
    assert value['training_eligible'] is False
    assert value['decision']=='defer_insufficient_checked_proposals'
    assert value['accepted']>0 and value['rejected_or_deferred']>0

def test_duplicate_and_unknown_rejected():
    import json
    base=json.loads(ROOT('configs/c13_checked_proposals_v1.json').read_text())
    base['proposals'].append(dict(base['proposals'][0]))
    try: validate_registry(base)
    except ValueError: pass
    else: raise AssertionError('duplicate accepted')
