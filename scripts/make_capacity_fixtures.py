#!/usr/bin/env python3
"""Build exact-length, solver-grounded synthetic capacity workloads offline.

These are deliberately simple capacity fixtures, not quality-training data or
independent benchmark evidence. Complete observed register writes and truthful
audit fields vary native encoded length; no whitespace padding, repeated filler,
gold-answer fields, or evidence truncation is used. No model weights are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab.contracts import audit_splits, validate_example
from replica_lab.curriculum import dataset_diagnostics
from train_text import encode_row


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bitwise_truth(width: int, mask: int, value: int) -> int:
    """Independent per-bit arithmetic, without using a response annotation."""
    if type(width) is not int or not 1 <= width <= 32:
        raise ValueError("width must be in [1, 32]")
    if any(type(item) is not int or not 0 <= item < 2 ** width for item in (mask, value)):
        raise ValueError("Register/input value exceeds unsigned bus width")
    return sum(((((value >> bit) & 1) + ((mask >> bit) & 1)) % 2) << bit
               for bit in range(width))


@lru_cache(maxsize=64)
def audit_variants(write_count: int) -> tuple[str, ...]:
    """Complete truthful status records; their different fields tune length."""
    fields = [f"clock={write_count - 1}", f"accepted={write_count}", "rejected=0",
              "queued=0", "ready=1", "mode=unsigned"]
    variants = ["", "Ready.", "Observation complete.", "All writes completed."]
    for count in range(1, len(fields) + 1):
        for selected in itertools.combinations(fields, count):
            for prefix in ("Audit", "Status", "Readback"):
                for separator in ("; ", ", "):
                    variants.append(prefix + ": " + separator.join(selected) + ".")
    return tuple(variants)


def make_episode(target_length: int, split: str, case: int,
                 distractors: int, audit_record: str = "") -> dict:
    if split not in {"train", "dev"} or type(case) is not int or not 0 <= case < 8:
        raise ValueError("Expected train/dev and an input case in [0, 7]")
    if type(target_length) is not int or target_length < 1:
        raise ValueError("Target length must be positive")
    if type(distractors) is not int or not 4 <= distractors <= 64:
        raise ValueError("Require between 4 and 64 meaningful auxiliary writes")
    width, mask = (4, 1) if split == "train" else (5, 2)
    world = f"capacity-{target_length}-{split}-world-{case}"
    records = [f"World {world}: unsigned {width}-bit registers; writes occur in clock order."]
    operations = [("MASK", (mask + 3) % (2 ** width))]
    for index in range(distractors):
        # The input case and event ordinal determine exogenous distractors.
        # The answer is computed only after the observed episode is complete.
        operations.append((f"R{index % 4}", (index * 3 + case * 5 + width) % (2 ** width)))
        if index == distractors // 2:
            operations.append(("MASK", mask))
    records.extend(f"Clock {clock}: {register} <- {value}."
                   for clock, (register, value) in enumerate(operations))
    if audit_record:
        if audit_record not in audit_variants(len(operations)):
            raise ValueError("Audit record is not a truthful supported episode status")
        records.append(audit_record)
    prompt = (f"Input is {case}. After the last observed write, output input XOR the current MASK register. "
              "Ignore auxiliary registers R0 through R3. Return only the decimal integer.")
    return {"id": world, "group_id": f"capacity-{target_length}-{split}-xor-width-{width}-mask-{mask}",
            "split": split, "prompt": prompt,
            "response": str(case ^ mask), "memory_texts": records,
            "source_ids": [f"fixture://capacity/{target_length}/{split}/world-{case}"],
            "verified": True}


def verify_episode(row: dict) -> dict:
    """Reparse chronological inputs and independently recompute the label."""
    validate_example(row, require_verified=True)
    world = re.fullmatch(r"capacity-([1-9][0-9]*)-(train|dev)-world-([0-7])", row["id"])
    if world is None or world.group(2) != row["split"]:
        raise ValueError("Capacity world ID/split differs")
    length, split, case = int(world.group(1)), world.group(2), int(world.group(3))
    width, intended_mask = (4, 1) if split == "train" else (5, 2)
    expected_header = f"World {row['id']}: unsigned {width}-bit registers; writes occur in clock order."
    if not row["memory_texts"] or row["memory_texts"][0] != expected_header:
        raise ValueError("Capacity world header differs")
    operations = []
    status = []
    for record in row["memory_texts"][1:]:
        match = re.fullmatch(r"Clock ([0-9]+): (MASK|R[0-3]) <- ([0-9]+)\.", record)
        if match is None:
            status.append(record)
            continue
        if status or int(match.group(1)) != len(operations):
            raise ValueError("Observed writes must remain in complete clock order")
        value = int(match.group(3))
        if value >= 2 ** width:
            raise ValueError("Observed register value exceeds bus width")
        operations.append((match.group(2), value))
    masks = [value for register, value in operations if register == "MASK"]
    if len(masks) != 2 or masks[-1] != intended_mask:
        raise ValueError("Capacity episode must preserve both original and current mask facts")
    distractors = len(operations) - len(masks)
    if status and (len(status) != 1 or status[0] not in audit_variants(len(operations))):
        raise ValueError("Observed audit status is unsupported or false")
    expected = bitwise_truth(width, masks[-1], case)
    if row["response"] != str(expected):
        raise ValueError("Capacity response disagrees with the independent per-bit solver")
    # Reconstruct every expected observation from exogenous inputs. This also
    # rejects injected target annotations and silently changed distractors.
    canonical = make_episode(length, split, case, distractors, status[0] if status else "")
    if row != canonical:
        raise ValueError("Capacity episode input/provenance differs from its deterministic world")
    return {"solver": "independent_per_bit_xor_after_chronological_writes",
            "width": width, "input": case, "current_mask": masks[-1],
            "expected": str(expected), "distractor_writes": distractors,
            "observed_write_count": len(operations), "audit_record": status[0] if status else None}


def encoded_counts(row: dict, tokenizer, target_length: int) -> dict:
    batch = encode_row(row, tokenizer, max_length=target_length)
    response = tokenizer.encode(row["response"], add_special_tokens=False)
    return {"actual_sequence_tokens": int(batch["input_ids"].shape[1]),
            "prompt_tokens": int(batch["prompt_ids"].shape[1]),
            "target_answer_tokens": len(response), "target_eos_tokens": 1,
            "supervised_tokens": int((batch["labels"] != -100).sum()),
            "support_tokens": int(batch["memory_ids"].shape[1]),
            "task_tokens": len(tokenizer.encode(row["prompt"], add_special_tokens=False)),
            "source_record_count": len(row["memory_texts"]),
            "source_record_tokens": [len(tokenizer.encode(record, add_special_tokens=False))
                                     for record in row["memory_texts"]]}


def fit_episode(tokenizer, target_length: int, split: str, case: int) -> tuple[dict, dict]:
    """Search complete meaningful episode variants; fail on impossible length."""
    if type(target_length) is not int or target_length < 1:
        raise ValueError("Target length must be positive")
    candidates = []
    for distractors in range(4, 65):
        base = make_episode(target_length, split, case, distractors)
        try:
            base_length = int(encode_row(base, tokenizer, target_length)["input_ids"].shape[1])
        except ValueError as error:
            if "exceeds sequence budget" not in str(error):
                raise
            continue
        candidates.append((base_length, distractors, base))
    # Prefer substantive writes over audit verbosity. The longest complete
    # history below budget usually needs only one short truthful status record.
    for base_length, distractors, base in sorted(candidates, reverse=True):
        gap = target_length - base_length
        if gap == 0:
            return base, {**encoded_counts(base, tokenizer, target_length), **verify_episode(base)}
        if gap > 80:
            continue
        for status in audit_variants(distractors + 2)[1:]:
            candidate = make_episode(target_length, split, case, distractors, status)
            try:
                actual_length = int(encode_row(candidate, tokenizer, target_length)["input_ids"].shape[1])
            except ValueError as error:
                if "exceeds sequence budget" not in str(error):
                    raise
                continue
            if actual_length == target_length:
                return candidate, {**encoded_counts(candidate, tokenizer, target_length),
                                   **verify_episode(candidate)}
    raise ValueError(f"Cannot construct exact {target_length}-token {split} case {case} "
                     "using complete register events and truthful audit fields")


def tokenizer_provenance(model_path: Path, lock_path: Path) -> dict:
    """Validate only config/tokenizer/template seal bytes; never load weights."""
    lock = json.loads(lock_path.read_text())
    if not lock.get("local_bytes_sealed") or not re.fullmatch(r"[0-9a-f]{40}", lock.get("revision", "")):
        raise ValueError("A sealed local source with an immutable revision is required")
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "special_tokens_map.json"]
    files = {}
    for name in names:
        actual = digest(model_path / name)
        if lock.get("snapshot_files_sha256", {}).get(name) != actual:
            raise ValueError(f"Tokenizer/config seal mismatch: {name}")
        files[name] = actual
    return {"source_repo_id": lock["repo_id"], "source_revision": lock["revision"],
            "source_lock_sha256": digest(lock_path), "source_tokenizer_files_sha256": files}


def build_workload(tokenizer, target_length: int) -> tuple[dict, dict]:
    rows = {split: [] for split in ("train", "dev")}
    audits = []
    for split in rows:
        for case in range(8):
            row, audit = fit_episode(tokenizer, target_length, split, case)
            rows[split].append(row)
            audits.append({"id": row["id"], "split": split, **audit})
    audit_splits(rows["train"] + rows["dev"])
    manifest = {"kind": "synthetic_register_capacity_workload", "quality_training": False,
                "authentic_personal_records": 0, "benchmark_examples": 0,
                "target_sequence_tokens": target_length,
                "actual_sequence_token_histogram": dict(Counter(a["actual_sequence_tokens"] for a in audits)),
                "supervised_token_histogram": dict(Counter(a["supervised_tokens"] for a in audits)),
                "splits": {split: dataset_diagnostics(items) for split, items in rows.items()},
                "episodes": audits,
                "length_policy": {
                    "padding_tokens": 0, "truncated_tokens": 0,
                    "method": "complete chronological auxiliary register writes plus truthful audit fields",
                    "distractors": "R0..R3 writes from input case/event ordinal; no response annotation consulted",
                    "relevant_evidence": "Both historical and latest MASK writes remain intact and past-only",
                    "exact_length_failure": "Reject if no complete supported episode reaches the exact native length"}}
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/minicpm5_source"))
    parser.add_argument("--model-lock", type=Path, default=Path("configs/minicpm5_sealed_lock.local.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/diagnostics"))
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512])
    args = parser.parse_args()
    if len(set(args.lengths)) != len(args.lengths) or any(length < 1 for length in args.lengths):
        parser.error("Lengths must be distinct positive integers")
    destinations = [args.output_root / f"capacity_{length}" for length in args.lengths]
    if any(path.exists() for path in destinations):
        parser.error("Use new output directories; existing capacity data is never overwritten")
    provenance = tokenizer_provenance(args.model_path, args.model_lock)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    template = tokenizer.get_chat_template()
    # Build all requested lengths before publishing any directory.
    prepared = []
    for target_length, destination in zip(args.lengths, destinations):
        rows, manifest = build_workload(tokenizer, target_length)
        manifest.update(provenance)
        manifest.update(generator_sha256=digest(Path(__file__)),
                        chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                        enable_thinking=False, add_generation_prompt=True,
                        eos_token_id=tokenizer.eos_token_id)
        prepared.append((destination, rows, manifest))
    for destination, rows, manifest in prepared:
        destination.mkdir(parents=True)
        files = {}
        for split, items in rows.items():
            path = destination / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in items))
            files[path.name] = {"rows": len(items), "sha256": digest(path)}
        manifest["files"] = files
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"path": str(destination), "lengths": manifest["actual_sequence_token_histogram"],
                          "train_answers": manifest["splits"]["train"]["answer_histogram"],
                          "dev_answers": manifest["splits"]["dev"]["answer_histogram"]}))


if __name__ == "__main__":
    main()
