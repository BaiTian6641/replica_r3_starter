#!/usr/bin/env python3
"""Hash a local snapshot after acquisition at the resolved model revision.

This records local bytes for reproducibility; it is not an independent signature
from the upstream model publisher. Retain the snapshot download receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--input-lock", required=True, type=Path)
    p.add_argument("--output-lock", required=True, type=Path)
    a = p.parse_args()
    if a.output_lock.exists():
        raise FileExistsError("choose a new output lock; preserve existing provenance")
    lock = json.loads(a.input_lock.read_text())
    if digest(a.snapshot / "config.json") != lock["config_sha256"]:
        raise ValueError("snapshot config differs from resolved revision")
    paths = sorted(x for x in a.snapshot.rglob("*") if x.is_file() and
                   x.suffix in {".json", ".safetensors", ".model", ".jinja"})
    if not any(x.suffix == ".safetensors" for x in paths):
        raise ValueError("no weights in snapshot")
    lock["snapshot_files_sha256"] = {x.relative_to(a.snapshot).as_posix(): digest(x) for x in paths}
    lock["weights_downloaded"] = True
    lock["local_bytes_sealed"] = True
    lock["upstream_authenticity_independently_verified"] = False
    a.output_lock.parent.mkdir(parents=True, exist_ok=True)
    a.output_lock.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"Sealed {len(paths)} local files; no weights executed.")


if __name__ == "__main__":
    main()
