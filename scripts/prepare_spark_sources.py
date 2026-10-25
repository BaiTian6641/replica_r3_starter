#!/usr/bin/env python3
"""Acquire bounded Spark-X2.6 source/header evidence into a fresh output directory.

The command never downloads a weight payload, never executes remote Python and
never imports torch/transformers. Every safetensors header is read with HTTP
range requests; an HTTP 200 response to a range request is refused rather than
consumed. Re-run with ``--header-bundle`` plus ``--assets-from`` (and
``--offline``) to reproduce a manifest from previously captured bytes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.spark_preparation import (  # noqa: E402
    DEFAULT_PLAN, FetchRefused, PreparationError, load_plan, run_preparation,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / DEFAULT_PLAN,
                        help="validated source-preparation plan JSON")
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh output directory; must not already exist")
    parser.add_argument("--header-bundle", type=Path,
                        help="previously captured safetensors headers (JSON bundle)")
    parser.add_argument("--assets-from", type=Path,
                        help="directory with <role>/<filename> captured text assets")
    parser.add_argument("--offline", action="store_true",
                        help="perform no network requests; requires both capture inputs")
    parser.add_argument("--quiet", action="store_true", help="suppress the summary")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        plan = load_plan(args.plan)
        manifest_path = run_preparation(
            plan, args.output,
            header_bundle=args.header_bundle,
            assets_from=args.assets_from,
            offline=args.offline,
        )
        manifest = json.loads(manifest_path.read_text())
    except FetchRefused as error:
        print(json.dumps({"status": "refused", **error.evidence()}, indent=2),
              file=sys.stderr)
        return 2
    except (PreparationError, FileExistsError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2),
              file=sys.stderr)
        return 2
    if not args.quiet:
        summary = {
            "status": manifest["status"],
            "manifest": str(manifest_path),
            "weights_downloaded": manifest["fetch_policy"]["weights_downloaded"],
            "remote_code_executed": manifest["fetch_policy"]["remote_code_executed"],
            "native_execution": manifest["fetch_policy"]["native_execution"],
            "offline": manifest["fetch_policy"]["offline"],
            "bytes_fetched": manifest["fetch_policy"]["bytes_fetched"],
            "network_requests": manifest["fetch_policy"]["network_requests"],
            "sources": [{
                "role": source["role"],
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "storage": source["storage"],
                "shards": len(source["safetensors_headers"]),
                "tensor_count": source["tensor_inventory"]["tensor_count"],
                "parameters_excluding_scales": source["tensor_inventory"][
                    "parameter_elements_excluding_scales"],
                "scale_tensor_count": source["tensor_inventory"]["scale_tensor_count"],
                "retained_assets": sum(1 for asset in source["assets"]
                                       if asset.get("retained_path")),
            } for source in manifest["sources"]],
            "manifest_fingerprint": manifest["manifest_fingerprint"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
