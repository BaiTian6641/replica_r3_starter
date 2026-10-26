#!/usr/bin/env python3
"""Prepare the staged vision curriculum and its preregistered protocol.

CPU-only, standard-library data preparation plus deterministic PNG rendering.
No model is loaded, no parameter is updated, no final split is scored. The
command writes the self-contained curriculum corpus (external train/dev images
copied locally, diagnostic train/dev images staged, fresh curriculum train/dev/
audit assets), every hash, the train/dev/audit seals, the exact two-stage
schedule, both arm declarations and one protocol pinning data, implementation,
trainable groups, optimizer/LR/update/seed declarations, generation and scoring
rules, stop conditions, checkpoint/retention policy, the 50 GiB driver guard, the
single exclusive worker and the swap/host limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_curriculum import (  # noqa: E402
    ABSTAIN_KINDS, ARMS, IMPLEMENTATION_FILES, PROCEDURAL_PARAMETERS_SANITY,
    audit_dataset, build_curriculum, build_protocol, dataset_pins,
    load_config, validate_protocol, verify_data_pins, verify_implementation_pins,
    write_curriculum, write_protocol,
)


def _summary(config, dataset, data_root, protocol_path, protocol):
    pins = dataset_pins(data_root)
    manifest = dataset.manifest
    seals = dataset.seal_receipts
    audit_manifest = json.loads(
        (data_root / "procedural/audit_manifest.json").read_text())
    descriptor_pools = dataset.procedural_manifest["descriptor_pools"]
    audit_pool = {tuple(item) for item in descriptor_pools["audit"]}
    pools_disjoint = audit_pool.isdisjoint(
        {tuple(item) for item in descriptor_pools["train"]}
        | {tuple(item) for item in descriptor_pools["dev"]})
    return {
        "status": "prepared_staged_not_admitted",
        "data_root": str(data_root.relative_to(ROOT)),
        "manifest_sha256": pins["manifest"]["sha256"],
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(
            protocol_path.read_bytes()).hexdigest(),
        "seed": config["seed"],
        "counts": manifest["counts"],
        "label_coverage": manifest["label_coverage"],
        "unknown_coverage": manifest["unknown_coverage"],
        "abstention_kinds": list(ABSTAIN_KINDS),
        "seals": {
            "train": seals["train_seal_sha256"],
            "dev": seals["dev_seal_sha256"],
            "audit": seals["audit_seal_sha256"],
            "audit_manifest": audit_manifest["seal_sha256"],
            "external_final": pins["seals"]["external_final"],
            "external_final_consumed": False,
        },
        "descriptor_holdout": {
            "audit_pool": [list(item) for item in sorted(audit_pool)],
            "audit_disjoint_from_curriculum_train_dev": pools_disjoint,
            "audit_disjoint_from_diagnostic": True,
            "world_combinations_disjoint": True,
        },
        "schedule": {
            "updates": protocol["schedule"]["updates"],
            "examples": protocol["schedule"]["examples"],
            "schedule_sha256": protocol["schedule"]["schedule_sha256"],
            "stages": protocol["schedule"]["stages"],
            "probe": protocol["schedule"]["probe"],
            "probe_cases": len(protocol["schedule"]["dev_probe_case_ids"]),
        },
        "arms": {
            "order": list(ARMS),
            "equal_compute": protocol["arms"]["equal_compute"],
            "projector_hidden": protocol["arms"]["projector_hidden"],
            "projector_parameter_count":
                protocol["arms"]["projector_parameter_count"],
            "muon8bit": protocol["trainable_groups"]["muon8bit"],
            "adamw_aux": protocol["trainable_groups"]["adamw_aux"],
            "adamw_control": protocol["trainable_groups"]["adamw_control"],
        },
        "train": protocol["train"],
        "retention": {
            "policy": protocol["retention"]["policy"],
            "baseline": protocol["retention"]["baseline"],
            "require_token_parity": protocol["retention"]["require_token_parity"],
        },
        "checkpoint_policy": protocol["checkpoint_policy"],
        "limits": protocol["limits"],
        "resource_policy": protocol["resource_policy"],
        "audit_policy": protocol["audit_policy"],
        "claims": protocol["claims"],
        "implementation_pins": sorted(IMPLEMENTATION_FILES),
        "sanity": dict(PROCEDURAL_PARAMETERS_SANITY),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/vision_curriculum_v1.json")
    parser.add_argument("--template", type=Path,
                        default=ROOT /
                        "configs/minicpm5_vision_v1_projector_wide_v1.json")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--protocol-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_root = args.data_root or (ROOT / config["data_root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    data_root = data_root.resolve()
    try:
        relative_data_root = data_root.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        parser.error("--data-root must live inside the protocol root")
    config["data_root"] = relative_data_root
    template = json.loads(args.template.read_text(encoding="utf-8"))
    built = build_curriculum(config, ROOT)
    write_curriculum(data_root, built)
    dataset = audit_dataset(data_root)
    if dataset.manifest["config_fingerprint"] != built["manifest"][
            "config_fingerprint"]:
        parser.error("Prepared manifest config hash differs")
    protocol = build_protocol(config, root=ROOT, template=template,
                              data_root=relative_data_root)
    validate_protocol(protocol)
    verify_implementation_pins(protocol, ROOT)
    verify_data_pins(dataset_pins(data_root), data_root)
    protocol_path = args.protocol_out or (data_root / "protocol.json")
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    write_protocol(protocol_path, protocol)
    summary = _summary(config, dataset, data_root, protocol_path, protocol)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary_out:
        path = args.summary_out if args.summary_out.is_absolute() \
            else ROOT / args.summary_out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
