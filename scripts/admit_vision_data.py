#!/usr/bin/env python3
"""Admit a capped owned visual cohort for one named V1 run; CPU-only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replica_lab.vision_admission import admit_owned_visual_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--v0-protocol', type=Path,
                        default=ROOT / 'configs/minicpm5_vision_v0_qualification_v3.json')
    parser.add_argument('--v0-score', type=Path,
                        default=ROOT / 'runs/minicpm5_vision_v0_20261025/score_restart_replay01/score_report.json')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--train-families', type=int, default=64)
    parser.add_argument('--eval-cases', type=int, default=64)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = admit_owned_visual_rows(project_root=ROOT,
        v0_protocol_path=args.v0_protocol, v0_score_path=args.v0_score,
        output=output, run_id=args.run_id, seed=args.seed,
        train_families=args.train_families, eval_cases=args.eval_cases)
    print(json.dumps({key: result[key] for key in (
        'run_id', 'admitted_training_rows', 'eligible_rows', 'verified_training_assets',
        'independent_training_worlds', 'independent_evaluation_worlds', 'evaluation_cases',
        'retention_cases', 'retention_domains', 'logical_dataset_bytes',
        'dataset_disk_ceiling_bytes', 'external_assets_acquired', 'native_execution')}))


if __name__ == '__main__':
    main()
