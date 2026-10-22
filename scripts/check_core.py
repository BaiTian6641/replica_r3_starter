#!/usr/bin/env python3
"""Run dependency-free record/data/publication contracts; no neural claims."""
import argparse
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for name in ("test_contracts.py", "test_memory.py", "test_learning.py"):
        suite.addTests(loader.discover(str(ROOT / "tests"), pattern=name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"kind": "stdlib_contract_tests", "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "success": result.wasSuccessful(),
              "model_quality_evaluated": False}
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
