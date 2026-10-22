#!/usr/bin/env python3
"""Check delivery integrity; this does not run or evaluate a language model.

Archived R2 links can name attachments that were not provided. They are excluded
from the current-document link check, while all seven archived bytes are hashed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
STEMS = (
    "01_full_system", "02_native_llama_integration", "03_memory_timescales",
    "04_causal_support_query", "05_autonomous_learning_loop",
    "06_replica_identity_lineage", "07_two_stage_roadmap", "08_dataset_pipeline",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/bundle_validation.json")
    args = parser.parse_args()
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = ROOT / destination
    failures: list[str] = []
    checks: dict[str, object] = {}
    required = [
        "README.md", "AGENT_START.md", "AGENT_CONTINUE.md", "execution_state.md",
        "pyproject.toml", "requirements-neural.txt", "source_manifest.json",
        "configs/model_roster.json", "configs/experiment_design.json",
        "configs/dataset_registry.json", "reports/validation.json",
        "reports/contract_demo.json", "reports/tests.xml", "reports/test_output.txt",
        "diagrams/Replica_R3_Architecture_Atlas.pdf",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            failures.append(f"Missing required file: {name}")
    checks["required_files"] = len(required)
    for folder, expected in (("docs", 7), ("prompts", 8), ("schemas", 3)):
        suffix = "*.json" if folder == "schemas" else "*.md"
        actual = len(list((ROOT / folder).glob(suffix)))
        checks[f"{folder}_count"] = actual
        if actual != expected:
            failures.append(f"{folder}: expected {expected} files, got {actual}")

    json_paths = sorted(ROOT.rglob("*.json"))
    for path in json_paths:
        try:
            json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            failures.append(f"Invalid JSON {path.relative_to(ROOT)}: {exc}")
    checks["json_files_parsed"] = len(json_paths)
    python_paths = sorted(path for folder in ("src", "scripts", "tests")
                          for path in (ROOT / folder).rglob("*.py"))
    for path in python_paths:
        try:
            ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, OSError) as exc:
            failures.append(f"Invalid Python {path.relative_to(ROOT)}: {exc}")
    checks["python_files_parsed"] = len(python_paths)

    local_links = 0
    for path in sorted(ROOT.rglob("*.md")):
        if "reference" in path.relative_to(ROOT).parts:
            continue
        content = re.sub(r"(?ms)^```.*?^```\s*$|^~~~.*?^~~~\s*$", "", path.read_text())
        for target in re.findall(r"!?\[[^\]]*\]\(([^\n)]+)\)", content):
            target = target.strip().strip("<>")
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            local_links += 1
            linked_path = path.parent / unquote(parts.path)
            if not linked_path.exists() and linked_path.resolve() != destination.resolve():
                failures.append(f"Broken link {path.relative_to(ROOT)}: {target}")
    checks["current_document_local_links"] = local_links

    source = json.loads((ROOT / "source_manifest.json").read_text())
    for row in source["files"]:
        path = ROOT / row["archive_path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            failures.append(f"Archived source changed: {row['archive_path']}")
    checks["archived_sources_hashed"] = len(source["files"])
    validation = json.loads((ROOT / "reports/validation.json").read_text())
    for name, digest in validation["tested_file_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            failures.append(f"Code changed since recorded test run: {name}")
    checks["tested_files_hash_matched"] = len(validation["tested_file_sha256"])
    checks["recorded_test_run"] = validation["pytest"]
    checks["tests_rerun_by_this_script"] = False

    for stem in STEMS:
        for suffix in ("svg", "png", "mmd"):
            path = ROOT / "diagrams" / f"{stem}.{suffix}"
            if not path.is_file() or not path.stat().st_size:
                failures.append(f"Missing diagram: {path.relative_to(ROOT)}")
            elif suffix == "svg":
                try:
                    ET.parse(path)
                except ET.ParseError as exc:
                    failures.append(f"Malformed SVG {stem}: {exc}")
    checks["diagram_sets"] = len(STEMS)
    try:
        from pypdf import PdfReader
    except ImportError:
        checks["pdf_pages"] = "not checked; install pypdf to check page count"
    else:
        pages = len(PdfReader(ROOT / "diagrams/Replica_R3_Architecture_Atlas.pdf").pages)
        checks["pdf_pages"] = pages
        if pages != 8:
            failures.append(f"Expected 8 PDF pages, got {pages}")
    report = {
        "status": "passed" if not failures else "failed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "File integrity, syntax and delivery consistency; no model-quality evaluation",
        "checks": checks,
        "failures": failures,
        "archival_link_check_excluded": "reference/r2; source bytes verified separately",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
