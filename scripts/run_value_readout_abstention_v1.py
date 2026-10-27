#!/usr/bin/env python3
"""Run or admit the deterministic CPU value-readout integration.

This command never loads MiniCPM5, a tokenizer, a native runtime or a download.
It trains only the declared selected-expert readout on the prepared fixture.  A
fit run writes treatment and separate control checkpoints; development opens
labels only after fit admission, and held-out only after development admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from replica_lab.value_readout_abstention import (  # noqa: E402
    ReadoutError, RunInterrupted, enforce_artifact_budget, run_controls,
    run_development, run_fit, run_heldout,
)
from value_readout_abstention_contract_v1 import (  # noqa: E402
    CONFIG_PATH, PREPARED_DIR, LabelVault, ProtocolError, admit_development,
    admit_fit_receipt, admit_heldout, read_json,
)


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _fit(args) -> int:
    try:
        result = run_fit(args.prepared, args.config, args.output,
                         interrupt_after=args.interrupt_after,
                         resume=args.resume)
    except RunInterrupted as interrupted:
        _print({"status": "interrupted",
                "completed_updates": interrupted.completed_updates,
                "resume": str(interrupted.resume_path)})
        return 3
    _print({"status": result["status"], "output": result["output"],
            "receipt_sha256": result["receipt"]["receipt_sha256"],
            "optimizer_steps": 24,
            "control_fits": result["fit_report"]["control_fits"]})
    return 0 if result["status"] == "passed" else 4


def _dev(args) -> int:
    result = run_development(args.prepared, args.config, args.fit_dir,
                             args.output)
    _print({"status": result["status"], "output": result["output"],
            "receipt_sha256": result["receipt"]["receipt_sha256"],
            "metrics": result["report"]["metrics"]})
    return 0 if result["status"] == "passed" else 4


def _heldout(args) -> int:
    result = run_heldout(args.prepared, args.config, args.fit_dir,
                         args.dev_dir, args.output)
    _print({"status": result["status"], "output": result["output"],
            "receipt_sha256": result["receipt"]["receipt_sha256"],
            "metrics": result["report"]["metrics"]})
    return 0 if result["status"] == "passed" else 4


def _all(args) -> int:
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    fit_root = root / "fit"
    fit = run_fit(args.prepared, args.config, fit_root,
                  interrupt_after=args.interrupt_after, resume=args.resume)
    if fit["status"] != "passed":
        _print({"status": "stopped_at_fit", "fit": fit["status"]})
        return 4
    dev_root = root / "dev_run"
    dev = run_development(args.prepared, args.config, fit_root, dev_root)
    if dev["status"] != "passed":
        _print({"status": "stopped_at_development",
                "fit": fit["status"], "development": dev["status"]})
        return 4
    heldout_root = root / "heldout_run"
    heldout = run_heldout(args.prepared, args.config, fit_root, dev_root,
                          heldout_root)
    controls_root = root / "controls_run"
    controls = run_controls(args.prepared, args.config, fit_root,
                            controls_root, split="dev")
    budget = enforce_artifact_budget(
        root, read_json(args.config), context="all")
    _print({"status": "completed",
            "fit": fit["status"], "development": dev["status"],
            "heldout": heldout["status"], "controls": controls["status"],
            "output": str(root),
            "artifact_budget": budget})
    return 0 if heldout["status"] == "passed" else 4


def _admit(args) -> int:
    fit_receipt = read_json(Path(args.fit_dir) / "fit_receipt.json")
    dev_receipt = read_json(Path(args.dev_dir) / "dev_receipt.json")
    heldout_receipt = read_json(Path(args.heldout_dir) / "heldout_receipt.json")
    vault = LabelVault(args.prepared, config_path=args.config)
    fit = admit_fit_receipt(
        fit_receipt, args.prepared, args.config,
        evidence_dir=Path(args.fit_dir) / "fit")
    development = admit_development(
        dev_receipt, fit_receipt=fit_receipt, prepared_dir=args.prepared,
        config_path=args.config, evidence_dir=Path(args.dev_dir) / "dev",
        fit_evidence_dir=Path(args.fit_dir) / "fit", label_vault=vault)
    heldout = admit_heldout(
        heldout_receipt, development_receipt=dev_receipt,
        fit_receipt=fit_receipt, prepared_dir=args.prepared,
        config_path=args.config,
        evidence_dir=Path(args.heldout_dir) / "heldout",
        dev_evidence_dir=Path(args.dev_dir) / "dev",
        fit_evidence_dir=Path(args.fit_dir) / "fit", label_vault=vault)
    _print({"status": "admitted", "fit": fit, "development": development,
            "heldout": heldout,
            "label_access_order": [row["split"] for row in vault.opened]})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--resume", action="store_true")
    fit.add_argument("--interrupt-after", type=int, default=None)

    dev = sub.add_parser("dev")
    dev.add_argument("--fit-dir", type=Path, required=True)
    dev.add_argument("--output", type=Path, required=True)

    heldout = sub.add_parser("heldout")
    heldout.add_argument("--fit-dir", type=Path, required=True)
    heldout.add_argument("--dev-dir", type=Path, required=True)
    heldout.add_argument("--output", type=Path, required=True)

    controls = sub.add_parser("controls")
    controls.add_argument("--fit-dir", type=Path, required=True)
    controls.add_argument("--dev-dir", type=Path, default=None)
    controls.add_argument("--output", type=Path, required=True)
    controls.add_argument("--split", choices=("dev", "heldout"), default="dev")

    all_run = sub.add_parser("all")
    all_run.add_argument("--output", type=Path, required=True)
    all_run.add_argument("--resume", action="store_true")
    all_run.add_argument("--interrupt-after", type=int, default=None)

    admit = sub.add_parser("admit")
    admit.add_argument("--fit-dir", type=Path, required=True)
    admit.add_argument("--dev-dir", type=Path, required=True)
    admit.add_argument("--heldout-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            return _fit(args)
        if args.command == "dev":
            return _dev(args)
        if args.command == "heldout":
            return _heldout(args)
        if args.command == "controls":
            result = run_controls(args.prepared, args.config, args.fit_dir,
                                  args.output, split=args.split,
                                  dev_dir=args.dev_dir)
            _print({"status": result["status"],
                    "arms": result["report"]["arms"]})
            return 0 if result["status"] == "passed" else 4
        if args.command == "all":
            return _all(args)
        if args.command == "admit":
            return _admit(args)
    except (ReadoutError, ProtocolError) as error:
        _print({"status": "refused", "error": str(error)})
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
