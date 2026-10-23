#!/usr/bin/env python3
"""Fresh E3 set_constant family for the no-progress action-guard study."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'fixtures/e3_constrained/cases'
CASES=[
 ('constraint_heldout_drain_01','heldout','constraint_a','DRAIN_TIMEOUT_S',0.4,2.4,
  {'MAX_CONNECTIONS':120,'HEARTBEAT_S':0.5}),
 ('constraint_heldout_shard_02','heldout','constraint_b','SHARD_COUNT',2,8,
  {'DRAIN_TIMEOUT_S':1.1,'HEARTBEAT_S':0.25}),
 ('constraint_heldout_window_03','heldout','constraint_c','WINDOW_MS',300,1800,
  {'MAX_CONNECTIONS':90,'SHARD_COUNT':6}),
 ('constraint_dev_lease_04','dev','constraint_dev','LEASE_TTL_S',0.6,4.0,
  {'MAX_CONNECTIONS':80,'DRAIN_TIMEOUT_S':1.0}),
]
GOAL=('The policy test for workspace/app_settings.py fails. Inspect the configuration, '
      'run the evaluator to learn the required bound, change exactly that one numeric '
      'constant with set_constant, and rerun the test to confirm it passes.')
def file_text(target, value):
    lines=['"""Service client settings. Adjust values to satisfy the deployment policy."""']
    for name,number in value.items():lines.append(f'{name} = {number}')
    return '\n'.join(lines)+'\n'
def evaluator(target, expected, original):
    return '''"""Independent policy evaluator. Read-only; do not modify."""
import ast
import pathlib
import unittest
SETTINGS = pathlib.Path(__file__).resolve().parents[1] / "workspace" / "app_settings.py"
TARGET = "{target}"
MINIMUM = {expected}
ORIGINAL = {original!r}
def settings():
    tree = ast.parse(SETTINGS.read_text(encoding="utf-8")); values = {{}}
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str): continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)): raise AssertionError("only simple name = number assignments are allowed")
        name, value = node.targets[0].id, node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, (int, float)) and not isinstance(value.value, bool)): raise AssertionError(f"{{name}} must be a numeric literal")
        if name in values: raise AssertionError(f"{{name}} must be assigned exactly once")
        values[name] = value.value
    return values
class PolicyTest(unittest.TestCase):
    def test_policy_bound(self):
        try: values = settings()
        except (OSError, UnicodeError, SyntaxError, ValueError, AssertionError) as error: self.fail(f"invalid workspace/app_settings.py: {{error}}")
        self.assertEqual(set(values), set(ORIGINAL) | {{TARGET}}, "original settings must be defined exactly")
        self.assertGreaterEqual(values[TARGET], MINIMUM, f"{{TARGET}}={{values[TARGET]}} needs {{TARGET}} >= {{MINIMUM}}")
        for name, expected in ORIGINAL.items(): self.assertEqual(values[name], expected, f"{{name}} must keep its original value")
if __name__ == "__main__": unittest.main()
'''.format(target=target,expected=expected,original=original)
def main():
    if OUT.exists() and any(OUT.iterdir()): raise SystemExit('Use a fresh fixture directory')
    manifest={'format':'replica.e3_constraint.dataset.v1','created_at_utc':'2026-10-23','goal':GOAL,'cases':[],'splits':{}}
    for cid,role,group,target,actual,expected,others in CASES:
        case=OUT/cid; (case/'workspace').mkdir(parents=True); (case/'evaluator').mkdir()
        originals=dict(others); originals[target]=actual
        (case/'workspace/app_settings.py').write_text(file_text(target,originals))
        (case/'evaluator/test_policy.py').write_text(evaluator(target,expected,others))
        repair=dict(others); repair[target]=expected
        payload={'format':'replica.e3.fixture.v1','case_id':cid,'role':role,'goal':GOAL,
          'success_condition':{'condition_id':'evaluator_test_passes','description':'python -m unittest evaluator/test_policy.py exits with status zero'},
          'constraints':['Only workspace/app_settings.py may be modified.','Change exactly one numeric constant with set_constant; leave every other setting untouched.','Do not modify anything under the evaluator directory.','Use only the allowed tools to inspect, edit, and test.'],
          'allowed_tools':['read_file','set_constant','run_test'],'workspace_files':['workspace/app_settings.py'],
          'evaluator':'evaluator/test_policy.py','workspace_entry':'workspace/app_settings.py',
          'reference_repair':{'path':'workspace/app_settings.py','content':file_text(target,repair)},
          'variant':{'constant':target,'expected_value':expected,'actual_value':actual,'description':'one numeric policy bound below its required minimum'},
          'source_group':group}
        (case/'case.json').write_text(json.dumps(payload,indent=2)+'\n')
        manifest['cases'].append({'case_id':cid,'role':role,'source_group':group,'constant':target,'actual':actual,'expected':expected})
        manifest['splits'].setdefault(role,[]).append(cid)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'cases':len(CASES),'splits':manifest['splits'],
      'manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()}))
if __name__=='__main__':main()
