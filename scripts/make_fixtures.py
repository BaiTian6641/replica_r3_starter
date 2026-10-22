#!/usr/bin/env python3
"""Create tiny solver-verified text fixtures. No teacher model or personal records."""
import argparse
import hashlib
import json
from pathlib import Path


def truth(width, mask, x):
    # Independent per-bit reference for the fixture's XOR transformation.
    return sum(((((x >> b) & 1) + ((mask >> b) & 1)) % 2) << b for b in range(width))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/fixtures"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "deterministic_synthetic_plumbing_fixtures",
                "authentic_personal_records": 0, "benchmark_examples": 0, "files": {}}
    for split, width, masks in (("train", 4, (1, 3, 7, 11)), ("dev", 5, (2, 6))):
        rows = []
        for mask in masks:
            for x in range(8):
                expected = truth(width, mask, x)
                assert expected == (x ^ mask), "Independent oracle disagreement"
                source = f"fixture://xor/width-{width}/mask-{mask}"
                rows.append({"id": f"{split}-w{width}-m{mask}-x{x}",
                    "group_id": f"xor-width-{width}-mask-{mask}", "split": split,
                    "prompt": f"Input is {x}. Apply the fixture's bus transformation. Return only the decimal integer.",
                    "response": str(expected),
                    "memory_texts": [f"This synthetic {width}-bit bus transforms x to x XOR {mask}."],
                    "source_ids": [source], "verified": True})
        path = a.output / (split + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["files"][path.name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
