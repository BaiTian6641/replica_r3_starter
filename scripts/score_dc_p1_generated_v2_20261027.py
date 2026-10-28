#!/usr/bin/env python3
"""Score one DC-P1 v2 stage and seal its independent receipt.

Three entry points share this command::

    --stage fit          first stage; reads fit labels, needs no predecessor
    --stage development  requires the sealed fit receipt
    --stage final        requires the sealed fit and development receipts

Every stage verifies the posted generated ids against the sealed tokenizer and
decode policy, rejects source-retained or KV-retained rows, rejects
answer-bearing fields, binds every prediction to its arm checkpoint identity and
to the sealed prompt, and only then compares the normalized generated answer
with the sealed label.  Labels are opened through a ``LabelVault`` permit; the
access spy records the attempt.

The command loads a tokenizer but no model, performs no generation and creates
no authorization.  It refuses an existing receipt path and never retries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, normalize_answer, read_json, read_jsonl, require,
    require_fresh_path, sha256_file, sha256_json, tokenizer_binding_sha,
    validate_manifest, validate_tokenizer_binding,
    validate_tokenizer_directory, verify_decoded_text, write_json_atomic,
)
from replica_lab.dc_staged_evaluation import (  # noqa: E402
    STAGE_DEVELOPMENT, STAGE_FINAL, STAGE_FIT, STAGE_ORDER, AccessSpy,
    LabelVault, ReceiptExpectation, StageError, StagedAdmission,
    build_stage_receipt, score_development_stage, score_final_stage,
    score_fit_stage, verify_stage_receipt,
)
import dc_p1_native_response_v2_20261027 as protocol  # noqa: E402

STAGE_FILES = {
    STAGE_FIT: (("fit_cases.jsonl", "sealed_labels_fit.jsonl"),),
    STAGE_DEVELOPMENT: (("dev_cases.jsonl", "sealed_labels_dev.jsonl"),),
    STAGE_FINAL: (("sealed_final_fit_cases.jsonl",
                   "sealed_final_labels_fit.jsonl"),
                  ("sealed_final_cases.jsonl", "sealed_labels_final.jsonl")),
}


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


def _stage_rows(prepared: Path, stage: str) -> tuple[list[dict], list[dict]]:
    cases: list[dict] = []
    labels: list[dict] = []
    for cases_name, labels_name in STAGE_FILES[stage]:
        cases.extend(read_jsonl(prepared / cases_name))
        labels.extend(read_jsonl(prepared / labels_name))
    return cases, labels


def _arm_specs(config: dict, stage: str) -> list[dict]:
    if stage == STAGE_FINAL:
        final = config["controls"]["final_stage"]
        return [{
            "arm_id": "treatment",
            "checkpoint_id": final["checkpoint_id"],
            "answer_span_ce_weight": config["objective"][
                "answer_span_ce_weight_treatment"],
        }]
    return [{
        "arm_id": arm["arm_id"],
        "checkpoint_id": arm["checkpoint_id"],
        "answer_span_ce_weight": arm["answer_span_ce_weight"],
    } for arm in config["controls"]["arms"]]


def _expectation(config: dict, config_path: Path, manifest: dict,
                 binding: dict, protocol_commit: str, stage: str
                 ) -> ReceiptExpectation:
    counts = config["prepared"]["counts"]
    cases_expected = {
        STAGE_FIT: counts["fit_cases"],
        STAGE_DEVELOPMENT: counts["dev_cases"],
        STAGE_FINAL: (counts["sealed_final_fit_cases"]
                      + counts["sealed_final_cases"]),
    }
    return ReceiptExpectation(
        protocol_id=protocol.PROTOCOL_ID,
        config_sha256=sha256_file(config_path),
        manifest_sha256=manifest["manifest_sha256"],
        protocol_commit=protocol_commit,
        tokenizer_binding_sha256=tokenizer_binding_sha(binding),
        decode_policy_sha256=sha256_json(binding["decode_policy"]),
        arms=tuple(arm["arm_id"] for arm in config["controls"]["arms"]),
        cases_expected=cases_expected,
    )


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
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = read_json(args.config)
        protocol.validate_config(config)
        protocol.validate_lineage(config)
        manifest = validate_manifest(
            args.prepared, config, expected_files=protocol.FILES,
            manifest_format=protocol.MANIFEST_FORMAT,
            protocol_id=protocol.PROTOCOL_ID)
        binding = validate_tokenizer_binding(config["tokenizer_binding"])
        decode = _decode_fn(args.tokenizer_dir, binding)
        expectation = _expectation(
            config, args.config, manifest, binding, args.protocol_commit,
            args.stage)
        admit = StagedAdmission(expectation, spy=AccessSpy())
        cases, labels = _stage_rows(args.prepared, args.stage)
        predictions = [row for row in read_jsonl(args.predictions)
                       if row.get("stage") == args.stage]
        require(bool(predictions), "scoring",
                f"no predictions for stage {args.stage!r}")
        vault = LabelVault({args.stage: labels}, spy=admit.spy)
        audit = _audit(args.objective_audit, config)

        if args.stage == STAGE_FIT:
            pass
        elif args.stage == STAGE_DEVELOPMENT:
            require(args.fit_receipt is not None, "stage_order",
                    "--fit-receipt is required for the development stage")
            admit.admit_fit(read_json(args.fit_receipt))
        else:
            require(args.fit_receipt is not None
                    and args.development_receipt is not None, "stage_order",
                    "the final stage requires both earlier receipts")
            admit.admit_fit(read_json(args.fit_receipt))
            admit.admit_development(read_json(args.development_receipt))

        permit = admit.require(args.stage)
        vault_labels = vault.open(args.stage, permit=permit)
        require(len(vault_labels) == len(labels), "label_access",
                "vault label count differs from the sealed rows")

        if args.stage == STAGE_FIT:
            report = score_fit_stage(
                protocol_id=protocol.PROTOCOL_ID, cases=cases,
                labels=vault_labels, predictions=predictions,
                arm_specs=_arm_specs(config, args.stage), binding=binding,
                binding_sha256=tokenizer_binding_sha(binding),
                decode_fn=decode,
                verify_decoded=lambda prediction, bound, decoder:
                    verify_decoded_text(prediction, bound, decoder),
                gates=config["evaluation"]["gates"],
                objective_audit=audit, normalize=normalize_answer)
        elif args.stage == STAGE_DEVELOPMENT:
            report = score_development_stage(
                protocol_id=protocol.PROTOCOL_ID, cases=cases,
                labels=vault_labels, predictions=predictions,
                arm_specs=_arm_specs(config, args.stage), binding=binding,
                binding_sha256=tokenizer_binding_sha(binding),
                decode_fn=decode,
                verify_decoded=lambda prediction, bound, decoder:
                    verify_decoded_text(prediction, bound, decoder),
                gates=config["evaluation"]["gates"],
                seed=config["seed"],
                resamples=config["evaluation"]["paired_bootstrap_resamples"],
                control_arms=tuple(config["evaluation"]["control_arms"]),
                objective_audit=audit, normalize=normalize_answer)
        else:
            report = score_final_stage(
                protocol_id=protocol.PROTOCOL_ID, cases=cases,
                labels=vault_labels, predictions=predictions,
                arm_specs=_arm_specs(config, args.stage), binding=binding,
                binding_sha256=tokenizer_binding_sha(binding),
                decode_fn=decode,
                verify_decoded=lambda prediction, bound, decoder:
                    verify_decoded_text(prediction, bound, decoder),
                gates=config["evaluation"]["gates"],
                objective_audit=audit, normalize=normalize_answer)

        report["attempt_id"] = args.attempt_id
        receipt = build_stage_receipt(
            report=report, expectation=expectation,
            attempt_id=args.attempt_id)
        verify_stage_receipt(receipt, stage=args.stage,
                             expectation=expectation)
        target = require_fresh_path(args.receipt, "stage_receipt")
        write_json_atomic(target, receipt)
        admission_state = None
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
            "receipt": str(target),
            "receipt_sha256": receipt["receipt_sha256"],
            "report_sha256": sha256_file(target),
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
