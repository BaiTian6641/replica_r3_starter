#!/usr/bin/env python3
"""Prepare the balanced paired image-dependence dataset and its protocol.

This command is CPU-only and deterministic. It writes the owned PNG corpus,
model rows, verifier rows/pairs/scenes, a hashed manifest, and one protocol that
pins the data hashes, implementation bytes, seed, split/world/combination
coverage, zero-update policy, MPS guard policy and readout limits. It never
loads a model and never scores a split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.vision_dependence import (  # noqa: E402
    IMPLEMENTATION_FILES, build_protocol, dataset_pins, generate_dataset,
    load_config, validate_protocol, verify_data_pins, write_protocol,
)


def _summary(config, dataset, data_root, protocol_path, protocol):
    counts = dataset.manifest["counts"]
    return {
        "status": "prepared",
        "data_root": str(data_root),
        "manifest_sha256": dataset_pins(data_root)["manifest"]["sha256"],
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(
            protocol_path.read_bytes()).hexdigest(),
        "seed": config["seed"],
        "counts": counts,
        "label_coverage": dataset.manifest["label_coverage"],
        "label_balance": dataset.manifest["label_balance"],
        "descriptor_pools": dataset.manifest["descriptor_pools"],
        "descriptor_pools_disjoint": all(
            {tuple(item) for item in dataset.manifest["descriptor_pools"][left]
             }.isdisjoint({tuple(item) for item in
                           dataset.manifest["descriptor_pools"][right]})
            for left in ("train", "dev", "final")
            for right in ("train", "dev", "final") if left < right),
        "distractor_compositions": {
            "counts": dataset.manifest["distractor_compositions"]["counts"],
            "disjoint": dataset.manifest["distractor_compositions"]["disjoint"],
        },
        "combinations_disjoint": dataset.manifest["combinations"]["disjoint"],
        "combination_key_counts": {
            split: len(keys) for split, keys in
            dataset.manifest["combinations"]["keys"].items()},
        "primary_readout": protocol["readout"]["primary"],
        "zero_updates": protocol["zero_updates"],
        "worlds_per_task_split": config["worlds_per_task"],
        "implementation_pins": sorted(IMPLEMENTATION_FILES),
        "files": dataset.manifest["files"],
        "generation_fingerprint": dataset.manifest["generation_fingerprint"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/vision_dependence_v2.json")
    parser.add_argument("--template", type=Path,
                        default=ROOT / "configs/minicpm5_vision_v0_qualification_v3.json")
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
    dataset = generate_dataset(config, data_root)
    protocol = build_protocol(config, root=ROOT, template=template,
                              data_root=relative_data_root)
    protocol_path = args.protocol_out or (data_root / "protocol.json")
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    validate_protocol(protocol)
    verify_data_pins(protocol, root=ROOT)
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
