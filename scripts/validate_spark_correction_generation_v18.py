#!/usr/bin/env python3
"""Run the controller's real correction generation from this source root on CPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replica_lab.intrinsic_memory_correction_fixture_v18 import FAMILIES, controller_case, repeated_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {kind: [run(args.output / f"{kind}_{family}_{int(unknown)}", family, unknown)
                      for family in FAMILIES for unknown in (False, True)]
               for kind, run in (("controller", controller_case), ("repeated", repeated_case))}
    # Assert local project imports came from the snapshot when invoked there.
    imported = {name: str(Path(module.__file__).resolve().relative_to(ROOT))
                for name, module in sys.modules.items()
                if name == "replica_lab" or name.startswith("replica_lab.")}
    receipt = {"scope": "synthetic_CPU_real_generation_and_source_validation",
               "source_root": str(ROOT), "project_imports": imported, "cases": results,
               "native_model_loads": 0, "evaluation_payload_access": False,
               "model_competence_claim": False}
    (args.output / "validation.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"controller_cases": len(results["controller"]),
                      "repeated_cases": len(results["repeated"]), "source_root": str(ROOT),
                      "native_model_loads": 0}, indent=2))


if __name__ == "__main__":
    main()
