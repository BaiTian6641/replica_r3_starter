#!/usr/bin/env python3
"""Run the bounded Spark text-memory protocol: fit, development and restart.

The canonical native invocation is wrapped by
``scripts/run_capacity_preflight.py --entry-point run_spark_text_memory_v1.py``.
Native execution is refused unless an exact external authorization and native
FP8 backend receipt exist and the active adapter publishes a native-FP8
admission contract.  ``--backend cpu_fixture`` is a deterministic tiny CPU
orchestration fixture only; it is never protocol evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.spark_text_memory_runner import (  # noqa: E402
    CPU_FIXTURE_LABEL, NATIVE_LABEL, STAGES, Refused, RunnerError, WorkLedger,
    WorkProfile, build_backend, load_fit_state, load_protocol, PreparedDataset,
    run_development, run_fit, run_protocol, run_restart,
)

DEFAULT_CONFIG = ROOT / "configs/spark_text_memory_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "runs/spark_text_memory_v1/native01"


def _profile(args, protocol, *, backend_kind: str) -> WorkProfile:
    return WorkProfile.from_protocol(
        protocol.config, backend_kind=backend_kind,
        fixture_context_tokens=(args.fixture_context_tokens
                                if backend_kind == CPU_FIXTURE_LABEL else None),
        fixture_max_new_tokens=(args.fixture_max_new_tokens
                                if backend_kind == CPU_FIXTURE_LABEL else None))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--backend", choices=(NATIVE_LABEL, CPU_FIXTURE_LABEL),
                        default=NATIVE_LABEL)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared", type=Path, default=None)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--backend-evidence", type=Path, default=None)
    parser.add_argument("--fixture-context-tokens", type=int, default=None)
    parser.add_argument("--fixture-max-new-tokens", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.backend == CPU_FIXTURE_LABEL and args.device != "cpu":
        parser.error("cpu_fixture requires --device cpu")
    if args.backend == NATIVE_LABEL and args.device != "mps":
        parser.error("native requires --device mps")
    try:
        protocol = load_protocol(args.config)
        prepared = args.prepared or protocol.dataset_root
        dataset = PreparedDataset.load(
            prepared,
            expected_sha256=protocol.config["dataset"].get("manifest_sha256"))
        profile = _profile(args, protocol, backend_kind=args.backend)
        if args.stage == "all":
            payload = run_protocol(
                protocol_path=args.config, prepared_root=prepared,
                output_root=args.output, backend_kind=args.backend,
                device=args.device, authorization_path=args.authorization,
                backend_evidence_path=args.backend_evidence, profile=profile,
                restart_in_process=False)
            status = payload.get("status")
            code = 0 if payload.get("execution_complete") else 4
        else:
            output = args.output
            if args.stage == "fit":
                if output.exists() and any(output.iterdir()):
                    raise Refused("output", "fit output must be fresh")
                output.mkdir(parents=True, exist_ok=True)
                ledger = WorkLedger(profile)
                backend = build_backend(
                    args.backend, protocol, dataset, profile,
                    device=args.device, output_root=output,
                    authorization_path=args.authorization,
                    backend_evidence_path=args.backend_evidence)
                ledger.load_backend()
                state = run_fit(protocol, dataset, backend, output, profile, ledger)
                payload = {"status": "fit_executed",
                           "execution_complete": True,
                           "fit_receipt": state.receipt,
                           "work": ledger.report()}
                status = "fit_executed"
                code = 0
            elif args.stage == "development":
                if not (output / "fit/fit_receipt.json").is_file():
                    raise Refused("fit", "a fit checkpoint and receipt are required")
                ledger = WorkLedger(profile)
                backend = build_backend(
                    args.backend, protocol, dataset, profile,
                    device=args.device, output_root=output,
                    authorization_path=args.authorization,
                    backend_evidence_path=args.backend_evidence,
                    allow_existing_output=True)
                ledger.load_backend()
                state = load_fit_state(protocol, dataset, backend, output, ledger)
                report = run_development(
                    protocol, dataset, backend, output, profile, ledger, state)
                payload = {"status": "development_complete",
                           "execution_complete": True,
                           "development": report, "work": ledger.report()}
                status = "development_complete"
                code = 0
            else:  # restart
                if not (output / "fit/fit_receipt.json").is_file():
                    raise Refused("fit", "a fit checkpoint and receipt are required")
                ledger = WorkLedger(profile)
                backend = build_backend(
                    args.backend, protocol, dataset, profile,
                    device=args.device, output_root=output,
                    authorization_path=args.authorization,
                    backend_evidence_path=args.backend_evidence,
                    allow_existing_output=True)
                ledger.load_backend()
                report = run_restart(
                    protocol, dataset, backend, output, profile, ledger,
                    restart_process="fresh_process")
                payload = {"status": "restart_complete",
                           "execution_complete": True,
                           "restart": report, "work": ledger.report()}
                status = "restart_complete"
                code = 0
        if args.report is not None:
            if args.report.exists():
                raise Refused("report", "report path must be fresh")
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(payload, indent=2, sort_keys=True)
                                   + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return code
    except (Refused, RunnerError, OSError, ValueError) as error:
        detail = (error.as_dict() if isinstance(error, Refused)
                  else {"check": "runner", "detail": str(error)})
        print(json.dumps({"status": "refused", **detail, "stage": args.stage},
                         indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
