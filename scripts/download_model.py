#!/usr/bin/env python3
"""Download the complete public snapshot at an existing immutable metadata lock.

Does not load weights or execute model code. Re-running resumes the same local
snapshot; choose a separate output directory when changing source revisions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    lock = json.loads(args.model_lock.read_text())
    if lock.get("repo_id") != "openbmb/MiniCPM5-2B" or not re.fullmatch(
            r"[0-9a-f]{40}", lock.get("revision", "")):
        p.error("Requires an immutable MiniCPM5-2B metadata lock")
    receipt_path = args.output.parent / (args.output.name + ".download.json")
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous["revision"] != lock["revision"] or previous["repo_id"] != lock["repo_id"]:
            p.error("Output belongs to another revision; choose a new directory")
    elif args.output.exists() and any(args.output.iterdir()):
        p.error("Existing nonempty output has no acquisition receipt; choose a new directory")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"repo_id": lock["repo_id"], "revision": lock["revision"],
               "requested_at": datetime.now(timezone.utc).isoformat(), "status": "downloading"}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=lock["repo_id"], revision=lock["revision"],
                      local_dir=args.output, cache_dir=root / ".cache" / "huggingface" / "hub",
                      max_workers=2)
    if hashlib.sha256((args.output / "config.json").read_bytes()).hexdigest() != lock["config_sha256"]:
        raise ValueError("Downloaded configuration differs from the pinned metadata lock")
    receipt.update(status="downloaded_not_loaded", completed_at=datetime.now(timezone.utc).isoformat())
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
