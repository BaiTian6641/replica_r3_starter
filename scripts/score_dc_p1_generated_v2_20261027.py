#!/usr/bin/env python3
"""Score one DC-P1 v2 stage and seal its independent receipt.

Three entry points share this command::

    --stage fit          first stage; reads fit labels, needs no predecessor
    --stage development  requires independently verified fit evidence
    --stage final        requires independently verified fit and development evidence

Earlier score reports, predictions and checkpoints are resolved and re-scored
before any current-stage label bytes are read.  The actual label-file parse is
recorded by the access spy.  Generated ids are checked against the sealed
tokenizer and decode policy.  The command loads a tokenizer but no model,
performs no generation and creates no authorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, normalize_answer, read_json, read_jsonl, require,
    require_fresh_path, sha256_file, sha256_json, tokenizer_binding_sha,
    validate_manifest, validate_tokenizer_binding,
    validate_tokenizer_directory, verify_decoded_text, write_json_atomic,
)
from replica_lab.dc_staged_evaluation import (  # noqa: E402
    STAGE_DEVELOPMENT, STAGE_FINAL, STAGE_FIT, STAGE_ORDER, AccessSpy,
    LabelVault, StageError, StagedAdmission, build_stage_receipt,
    score_development_stage, score_final_stage, score_fit_stage,
    verify_stage_receipt,
)
from replica_lab import dc_p1_evidence as evidence  # noqa: E402
import dc_p1_native_response_v2_20261027 as protocol  # noqa: E402


def _decode_fn(tokenizer_dir: Path, binding: dict):
    validate_tokenizer_directory(binding, tokenizer_dir)
    from transformers import AutoTokenizer  # noqa: PLC0415 - offline load only

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False,
        revision=binding["revision"])
    policy = binding["decode_policy"]

    def decode(ids):
        return tokenizer.decode(
            list(ids), skip_special_tokens=policy["skip_special_tokens"],
            clean_up_tokenization_spaces=policy[
                "clean_up_tokenization_spaces"])

    return decode


def _audit(path: Path | None, config: dict) -> dict | None:
    if path is None:
        require(
            not config["objective"]["audit_required"], "objective_audit",
            "the protocol requires the mask/objective audit artifact")
        return None
    audit = read_json(path)
    require(audit.get("objective_format") is None
            or audit.get("format"), "objective_audit",
            "audit artifact must carry a format")
    audit = dict(audit)
    audit["expected_claim"] = config["objective"]["declared_claim"]
    require(audit.get("claim") == config["objective"]["declared_claim"],
            "objective_audit",
            f"audit claim {audit.get('claim')!r} differs from the declared "
            f"{config['objective']['declared_claim']!r}; the claim may not be "
            "widened after the fact")
    return audit


def _evidence_root(config: dict, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    root = Path(config["launch"]["output_root"])
    return (root if root.is_absolute() else ROOT / root).resolve()


def _verify_prior_stage(stage: str, root: Path, args, config: dict,
                        manifest: dict, binding: dict, decode,
                        admission: StagedAdmission) -> None:
    evidence_result = evidence.verify_stage_evidence(
        stage=stage, output_root=root, prepared=args.prepared,
        config=config, config_path=args.config, manifest=manifest,
        protocol_commit=args.protocol_commit, binding=binding,
        decode_fn=decode, expected_evidence_class="native_mps",
        require_publication=(root / "publication_journal.json").is_file())
    if stage == STAGE_FIT:
        admission.admit_fit(evidence_result["receipt"])
    else:
        admission.admit_development(evidence_result["receipt"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGE_ORDER, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=protocol.CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=protocol.PREPARED_DIR)
    parser.add_argument("--tokenizer-dir", type=Path,
                        default=ROOT / "checkpoints" / "minicpm5_source")
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--objective-audit", type=Path, default=None)
    parser.add_argument("--fit-receipt", type=Path, default=None)
    parser.add_argument("--development-receipt", type=Path, default=None)
    parser.add_argument("--evidence-root", type=Path, default=None,
                        help="attempt root for predictions and checkpoints")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = read_json(args.config)
        protocol.validate_config(config)
        protocol.validate_lineage(config)
        manifest = validate_manifest(
            args.prepared, config, expected_files=protocol.FILES,
            manifest_format=protocol.MANIFEST_FORMAT,
            protocol_id=protocol.PROTOCOL_ID,
            skip_hash_files=evidence.LATE_LABEL_FILES[args.stage])
        binding = validate_tokenizer_binding(config["tokenizer_binding"])
        decode = _decode_fn(args.tokenizer_dir, binding)
        expectation = evidence.receipt_expectation(
            config=config, config_path=args.config, manifest=manifest,
            binding=binding, protocol_commit=args.protocol_commit)
        admit = StagedAdmission(expectation, spy=AccessSpy())
        evidence_root = _evidence_root(config, args.evidence_root)
        report_path = require_fresh_path(
            args.receipt.with_suffix(".report.json"), "score_report")
        receipt_path = require_fresh_path(args.receipt, "stage_receipt")

        if args.stage == STAGE_DEVELOPMENT:
            require(args.fit_receipt is not None, "stage_order",
                    "--fit-receipt is required for the development stage")
            fit_root = args.fit_receipt.resolve().parent.parent
            require(fit_root == evidence_root, "stage_linkage",
                    "fit evidence root differs from --evidence-root")
            _verify_prior_stage(STAGE_FIT, fit_root, args, config, manifest,
                                binding, decode, admit)
        elif args.stage == STAGE_FINAL:
            require(args.fit_receipt is not None
                    and args.development_receipt is not None, "stage_order",
                    "the final stage requires both earlier receipts")
            fit_root = args.fit_receipt.resolve().parent.parent
            development_root = args.development_receipt.resolve().parent.parent
            require(fit_root == development_root == evidence_root,
                    "stage_linkage",
                    "fit/development evidence roots differ from --evidence-root")
            _verify_prior_stage(STAGE_FIT, fit_root, args, config, manifest,
                                binding, decode, admit)
            _verify_prior_stage(STAGE_DEVELOPMENT, development_root, args,
                                config, manifest, binding, decode, admit)

        permit = admit.require(args.stage)
        cases, labels = evidence.load_stage_rows(
            args.prepared, args.stage, manifest, spy=admit.spy)
        specs = evidence.arm_specs(config, args.stage)
        prediction_paths = {spec["arm_id"]: args.predictions for spec in specs}
        predictions = [row for row in read_jsonl(args.predictions)
                       if row.get("stage") == args.stage]
        require(bool(predictions), "scoring",
                f"no predictions for stage {args.stage!r}")
        classes = {row.get("evidence_class") for row in predictions}
        require(len(classes) == 1, "evidence_class",
                f"prediction execution classes differ: {sorted(classes)}")
        evidence_class = classes.pop()
        require(evidence_class in ("native_mps", "cpu_fixture"),
                "evidence_class", f"unknown execution class {evidence_class!r}")
        vault = LabelVault({args.stage: labels}, spy=admit.spy)
        audit = _audit(args.objective_audit, config)
        vault_labels = vault.open(args.stage, permit=permit)
        require(len(vault_labels) == len(labels), "label_access",
                "vault label count differs from the sealed rows")
        common = dict(
            protocol_id=protocol.PROTOCOL_ID, cases=cases,
            labels=vault_labels, predictions=predictions,
            arm_specs=specs, binding=binding,
            binding_sha256=tokenizer_binding_sha(binding), decode_fn=decode,
            verify_decoded=lambda prediction, bound, decoder:
                verify_decoded_text(prediction, bound, decoder),
            gates=config["evaluation"]["gates"], objective_audit=audit,
            normalize=normalize_answer)
        if args.stage == STAGE_FIT:
            report = score_fit_stage(**common)
        elif args.stage == STAGE_DEVELOPMENT:
            report = score_development_stage(
                **common, seed=config["seed"],
                resamples=config["evaluation"]["paired_bootstrap_resamples"],
                control_arms=tuple(config["evaluation"]["control_arms"]))
        else:
            report = score_final_stage(**common)
        report["attempt_id"] = args.attempt_id
        report["scoring_counts"] = dict(report["cases"])
        report["evidence_class"] = evidence_class
        report["native_execution_admissible"] = (
            evidence_class == "native_mps" and report["status"] == "passed")

        report_path = require_fresh_path(
            args.receipt.with_suffix(".report.json"), "score_report")
        receipt_path = require_fresh_path(args.receipt, "stage_receipt")
        write_json_atomic(report_path, report)
        receipt = build_stage_receipt(
            report=report, expectation=expectation,
            attempt_id=args.attempt_id)
        verify_stage_receipt(receipt, stage=args.stage,
                             expectation=expectation)
        write_json_atomic(receipt_path, receipt)
        evidence.verify_stage_evidence(
            stage=args.stage, output_root=evidence_root,
            prepared=args.prepared, config=config, config_path=args.config,
            manifest=manifest, protocol_commit=args.protocol_commit,
            binding=binding, decode_fn=decode,
            expected_evidence_class=evidence_class,
            expected_attempt_id=args.attempt_id,
            report_path=report_path, receipt_path=receipt_path,
            prediction_paths=prediction_paths, require_publication=False)

        admission_state = None
        if evidence_class != "native_mps":
            admission_state = {
                "refused": {
                    "check": "evidence_class",
                    "detail": "CPU fixtures cannot admit native stages",
                },
                "state": admit.state(),
            }
        else:
            try:
                if args.stage == STAGE_FIT:
                    admit.admit_fit(receipt)
                elif args.stage == STAGE_DEVELOPMENT:
                    admit.admit_development(receipt)
                else:
                    admit.admit_final(receipt)
                admission_state = admit.state()
            except StageError as error:
                admission_state = {"refused": error.as_dict(),
                                   "state": admit.state()}
        print(json.dumps({
            "status": report["status"],
            "stage": args.stage,
            "receipt": str(receipt_path),
            "score_report": str(report_path),
            "receipt_sha256": receipt["receipt_sha256"],
            "report_sha256": sha256_file(report_path),
            "gates": {name: row.get("passed")
                      for name, row in sorted(report["gates"].items())},
            "admission": admission_state,
            "access": admit.spy.state(),
            "native_computation": False,
            "generation": False,
        }, indent=2, sort_keys=True))
        return 0 if report["status"] == "passed" else 4
    except (ProtocolError, StageError, OSError, ValueError) as error:
        print(json.dumps({"status": "refused", "error": str(error)},
                         indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
