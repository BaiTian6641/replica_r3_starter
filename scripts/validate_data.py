#!/usr/bin/env python3
"""Validate small SFT exports. This does not establish source truth or consent."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replica_lab.contracts import audit_splits, validate_example


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--require-verified", action="store_true")
    a = p.parse_args()
    rows = []
    for name in a.files:
        for line_no, line in enumerate(Path(name).read_text().splitlines(), 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    validate_example(row, require_verified=a.require_verified)
                    rows.append(row)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"{name}:{line_no}: {e}") from e
    report = audit_splits(rows)
    report["semantic_grounding_checked"] = False
    report["consent_checked"] = False
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
