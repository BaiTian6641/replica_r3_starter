#!/usr/bin/env python3
"""Acquire or reuse the sealed SigLIP2 vision source without loading a model.

Exactly four pinned files from ``google/siglip2-base-patch16-224`` are streamed
under a 2 GiB aggregate cap. The tool never constructs or runs a model or
processor, never runs an encoder forward and never touches MPS. A network
failure is preserved verbatim in ``download.json`` and the report; a valid
sealed source is always reused without network access.

The acquired weights live under ``checkpoints/`` and are intentionally not Git
staged. The sealed manifest and its hashes are reported under ``reports/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.architecture.vision import (  # noqa: E402
    VisionSourceError,
    acquire_vision_source,
    load_source_config,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/vision_encoder_source_v1.json",
                        help="tracked immutable source pin")
    parser.add_argument("--output", type=Path,
                        help="source directory; defaults to the pinned output_directory")
    parser.add_argument("--report-dir", type=Path,
                        default=ROOT / "reports/vision_encoder_source_20261025",
                        help="directory for the acquisition report")
    parser.add_argument("--offline", action="store_true",
                        help="reuse only already sealed/complete bytes; never request network")
    parser.add_argument("--quiet", action="store_true", help="suppress the summary")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = load_source_config(args.config)
        output = args.output or (ROOT / config["output_directory"])
        report = acquire_vision_source(
            args.config, output, offline=args.offline, report_dir=args.report_dir)
    except (VisionSourceError, OSError) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "error": str(error)}, indent=2), file=sys.stderr)
        return 2
    if not args.quiet:
        summary = {
            "status": report["status"],
            "source_id": report["source_id"],
            "revision": report["revision"],
            "manifest": report["manifest_path"],
            "manifest_sha256": report["manifest_sha256"],
            "seal_digest": report["seal_digest"],
            "network_used": report["network_used"],
            "bytes_downloaded": report["bytes_downloaded"],
            "total_source_bytes": report["total_source_bytes"],
            "cap_bytes": report["cap_bytes"],
            "within_cap": report["within_cap"],
            "license": report["license"],
            "files": report["files"],
            "processor": report["processor"],
            "tensor_inventory": {
                "tensor_count": report["tensor_inventory"]["tensor_count"],
                "vision_tensor_count": report["tensor_inventory"]["vision_tensor_count"],
                "text_tensor_count": report["tensor_inventory"]["text_tensor_count"],
                "vision_layer_count": report["tensor_inventory"]["vision_layer_count"],
                "vision_parameter_elements":
                    report["tensor_inventory"]["vision_parameter_elements"],
            },
            "report": str(args.report_dir / "acquisition_report.json"),
            "weights_git_staged": False,
            "native_execution": False,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
