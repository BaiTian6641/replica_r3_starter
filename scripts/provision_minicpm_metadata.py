#!/usr/bin/env python3
"""Provision and verify MiniCPM tokenizer/config metadata without model weights.

The source is an existing sealed local snapshot or a previously provisioned
metadata directory.  Only declared metadata is copied; weights are never read,
downloaded, replaced or duplicated.  Existing mismatched output files fail
closed.  This is for CPU tokenizer/integration tests, not native model loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "f97400052a43d642bbc6e9975e2397e3ae6a6b52"
METADATA_SHA256 = {
    "chat_template.jinja": "bc1f78ef8a86b6bdf64a056cf89dd0ac506db4f198ab363d85168e2500ab44dd",
    "config.json": "59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180",
    "generation_config.json": "9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1",
    "model.safetensors.index.json": "6d839cd76e8395de548a0e6cc310386f66d1ecbb2c75d198a8dfd3d70892b756",
    "special_tokens_map.json": "82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db",
    "tokenizer.json": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config.json": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def provision(source: Path, output: Path, receipt: Path | None) -> dict:
    source = source.resolve()
    output = output.resolve()
    require_source = source.is_dir()
    if not require_source:
        raise ValueError(f"metadata source is not a directory: {source}")
    files = {}
    for name, expected in METADATA_SHA256.items():
        source_file = source / name
        if not source_file.is_file():
            raise ValueError(f"missing sealed metadata: {source_file}")
        actual = digest(source_file)
        if actual != expected:
            raise ValueError(f"sealed metadata hash differs for {name}: {actual}")
        output_file = output / name
        if output_file.exists() and digest(output_file) != expected:
            raise ValueError(f"refusing to overwrite mismatched metadata: {output_file}")
        files[name] = actual
    for name in METADATA_SHA256:
        payload = (source / name).read_bytes()
        if source != output:
            atomic_write(output / name, payload)
    report = {
        "format": "replica.minicpm_metadata_provisioning.v1",
        "status": "verified_metadata_only",
        "revision": REVISION,
        "source": str(source),
        "output": str(output),
        "files_sha256": files,
        "weights_copied": False,
        "weights_read": False,
        "downloads": 0,
        "model_loaded": False,
        "native_execution": False,
        "note": "Tokenizer/config integration only; native model loading requires the separately sealed full snapshot and lock.",
    }
    if receipt is not None:
        if receipt.exists():
            raise ValueError(f"receipt already exists: {receipt}")
        atomic_write(receipt, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="existing sealed local snapshot or metadata directory")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/minicpm5_source")
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = provision(args.source, args.output, args.receipt)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "refused", "detail": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
