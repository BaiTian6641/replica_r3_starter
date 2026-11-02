#!/usr/bin/env python3
"""Prepare a fresh holdout once; never imported by tests or fitting diagnostics.

Only public commitments are printed. Label payload is local, mode 0600 and
excluded from Git. Future access requires a separately reviewed evaluation gate.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    destination = ROOT / "data/sealed_spark_eval_v19_20261031"
    destination.mkdir(mode=0o700, exist_ok=False)
    rows, public_groups = [], []
    palette = ["red", "blue", "green", "gold"]
    for family in ("two_hop_colour_v19", "exception_precedence_colour_v19"):
        for group_index in range(8):
            group = "sealed19-" + secrets.token_hex(16)
            a, b, c = ["N" + secrets.token_hex(6) for _ in range(3)]
            chosen = secrets.SystemRandom().sample(palette, 2)
            public_groups.append({"group_id": group, "rule_family": family, "rows": 2})
            for side, color in enumerate(chosen):
                unknown = group_index >= 6 and side == (group_index % 2)
                if family == "two_hop_colour_v19":
                    text = f"{a} links to {b}. {b} links to {c}. "
                    text += f"{c} has color {color}." if not unknown else f"No color is recorded for {c}."
                    question = f"What color is reached after following exactly two links from {a}?"
                else:
                    default = palette[(palette.index(color)+1) % 4]
                    text = f"Members of {b} default to {default}. {a} belongs to {b}. "
                    text += f"An exception sets {a} to {color}. Exceptions override defaults." if not unknown else f"An exception withdraws the color of {a}. Withdrawals override defaults."
                    question = f"What is the current color of {a} after applying the exception?"
                rows.append({"world_id": group + f"-{side}", "group_id": group,
                             "rule_family": family, "text": text, "question": question,
                             "answer": "unknown" if unknown else color})
    payload = (json.dumps({"format": "replica.sealed_evaluation.v19", "rows": rows}, sort_keys=True) + "\n").encode()
    path = destination / "evaluation.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    # Namespace and rule-family separation are enforced during generation;
    # no old development payload is read to establish this disjoint construction.
    manifest = {"format": "replica.sealed_evaluation_public_commitment.v19",
        "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload), "rows": len(rows), "groups": public_groups,
        "disjointness": "fresh cryptographic group/entity namespaces; two new rule families",
        "retired_rule_families": ["explicit_entity_color_lookup", "latest_recorded_color", "partner_colour", "same_colour"],
        "label_access": "preparation_only_no_labels_printed; no test or fit imports",
        "future_gate": "qualified terminal fitting + independent protocol review + explicit evaluation admission",
        "status": "sealed_unlaunched_not_native_competence_qualified"}
    report = ROOT / "reports/spark_repairs_v18/sealed_evaluation_public.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps({key: manifest[key] for key in ("format", "sha256", "rows", "status")}))


if __name__ == "__main__":
    prepare()
