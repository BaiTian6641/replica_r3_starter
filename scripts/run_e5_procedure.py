#!/usr/bin/env python3
"""Execute and independently evaluate the E5 procedure candidate (CPU only)."""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/e5_procedure_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    from replica_lab.learning import ManifestRegistry
    from replica_lab.contracts import fingerprint
    from replica_lab.runtime.learning_proposal import LearningProposal
    from replica_lab.runtime.procedures import RepairProcedure
    from replica_lab.runtime.tools import FixtureWorkspace
    procedure = RepairProcedure('repair-config', config['procedure']['workspace_entry'],
                                tuple(config['source_event_ids']))
    results = []
    for role, cases in (('dev', config['dev_cases']), ('heldout', config['heldout_cases'])):
        for case_id in cases:
            with tempfile.TemporaryDirectory() as directory:
                copy = Path(directory) / case_id
                shutil.copytree(ROOT / config['fixtures_root'] / case_id, copy)
                tools = FixtureWorkspace(copy)
                run = procedure.execute(tools)
                verified = tools.verify_success()
                workspace = (copy / 'workspace/app_settings.py').read_text()
                results.append({
                    'case_id': case_id, 'role': role, 'status': run.status,
                    'parsed': run.parsed, 'final_exit_code': run.final_exit_code,
                    'error': run.error,
                    'independent_passed': verified.get('passed'),
                    'receipts': list(run.receipts),
                    'workspace_sha256': hashlib.sha256(workspace.encode()).hexdigest(),
                })
    accepted = all(item['status'] == 'succeeded' and item['independent_passed']
                   for item in results)
    payload = {'base_revision': config['source_revision'],
               'identity_version': 'generic_procedure',
               'procedure': procedure.artifact().digest,
               'steps': config['procedure']['steps']}
    evaluation = {'decision': 'accept' if accepted else 'reject',
                  'candidate_digest': fingerprint(payload),
                  'parent_manifest_id': None,
                  'cases': [{key: item[key] for key in
                             ('case_id', 'role', 'status', 'independent_passed')}
                            for item in results],
                  'target': 'procedure'}
    registry = ManifestRegistry(args.output / 'registry')
    manifest = registry.publish(payload, evaluation, expected_parent=None)
    harmful = {'decision': 'reject', 'reason': 'harmful_candidate_probe',
               'candidate_digest': fingerprint(payload),
               'parent_manifest_id': manifest['manifest_id']}
    rejection = None
    try:
        registry.publish(payload, harmful, expected_parent=manifest['manifest_id'])
    except PermissionError as error:
        rejection = {'rejected': True, 'reason': str(error)}
    proposal = LearningProposal('proposal-procedure-repair', None,
                                tuple(config['source_event_ids']), 'procedure',
                                'repair-config', {'train_cases': config['train_cases']},
                                config['budget'], 'host_tools',
                                'source-separated dev/held-out execution and independent evaluator')
    report = {'status': 'completed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'config_sha256': sha(args.config), 'procedure': procedure.artifact().digest,
              'proposal': proposal.to_dict(), 'results': results,
              'evaluation': evaluation, 'active_manifest_id': manifest['manifest_id'],
              'harmful_candidate': rejection,
              'acceptance': {'accepted': accepted,
                             'heldout_passed': sum(item['independent_passed'] is True
                                                   for item in results if item['role'] == 'heldout'),
                             'heldout_cases': len(config['heldout_cases'])},
              'claims': 'Procedure baseline only; no neural update or model-selected action.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'acceptance': report['acceptance'],
                      'manifest': manifest['manifest_id']}))


if __name__ == '__main__':
    main()
