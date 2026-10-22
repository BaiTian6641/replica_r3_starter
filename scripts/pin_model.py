#!/usr/bin/env python3
"""Resolve a public model revision and hash config bytes. Does not fetch weights."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "replica-lab-metadata/0.3"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="openbmb/MiniCPM5-2B")
    p.add_argument("--revision", default="main")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", a.model_id):
        p.error("model-id must be an explicit owner/repository")
    rev = urllib.parse.quote(a.revision, safe="")
    info = json.loads(fetch(f"https://huggingface.co/api/models/{a.model_id}/revision/{rev}"))
    sha = info.get("sha", "")
    if not re.fullmatch("[0-9a-f]{40}", sha):
        raise RuntimeError("provider did not return a 40-character commit; refusing unpinned lock")
    raw = fetch(f"https://huggingface.co/{a.model_id}/resolve/{sha}/config.json")
    config = json.loads(raw)
    lock = {"repo_id": a.model_id, "revision": sha, "requested_revision": a.revision,
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "architectures": config.get("architectures"), "model_type": config.get("model_type"),
            "resolved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "weights_downloaded": False, "integration_tested": False}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("lock already exists; choose a new path for a new revision")
    out.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))
    print("Download the entire snapshot at this exact revision with huggingface_hub.snapshot_download.")
    print("A metadata lock is not a checkpoint hash or an integration pass.")


if __name__ == "__main__":
    main()
