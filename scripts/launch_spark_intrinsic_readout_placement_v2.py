#!/usr/bin/env python3
"""Bound environment admission before placement execution revision v2.

The S0–S1 checkpoint format and criteria are unchanged; the separately bound
runner places and verifies the frozen base before constructing its adapter.

Standard-library checks only until the environment and human receipt validate.
This launcher neither repairs installed packages nor substitutes model code.
The guarded runner retains source, competence, work and exclusive-worker gates.
"""
import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "replica.spark_intrinsic.execution_environment.v1"


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or substituted file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(root, relative):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("binding path must be relative and contained")
    path = root / value
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("binding escapes its root")
    return path


def validate_environment(manifest, *, root=ROOT, prefix=None, python=None,
                         version=importlib.metadata.version,
                         module_spec=importlib.machinery.PathFinder.find_spec):
    if manifest.get("format") != FORMAT:
        raise ValueError("unknown environment manifest")
    expected_prefix = checked_path(root, manifest["environment_directory"])
    actual_prefix = Path(sys.prefix if prefix is None else prefix)
    if actual_prefix.resolve() != expected_prefix.resolve():
        raise ValueError("wrong interpreter environment; no automatic fallback")
    if (platform.python_version() if python is None else python) != manifest["python"]:
        raise ValueError("Python version differs")
    lock = checked_path(root, manifest["lock"]["path"])
    if digest(lock) != manifest["lock"]["sha256"]:
        raise ValueError("environment lock hash differs")
    locked = dict(line.split("==", 1) for line in lock.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#"))
    if locked != manifest["packages"]:
        raise ValueError("package inventory differs from lock")
    actual = {}
    for name, expected in locked.items():
        try:
            actual[name] = version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"missing dependency: {name}") from error
        if actual[name] != expected:
            raise ValueError(f"dependency version differs: {name} {actual[name]} != {expected}")
    for name, relative in manifest["module_origins"].items():
        spec = module_spec(name)
        expected = checked_path(expected_prefix, relative)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected.resolve():
            raise ValueError(f"module resolution differs: {name}")
    for relative, expected in manifest["critical_files"].items():
        if digest(checked_path(expected_prefix, relative)) != expected:
            raise ValueError(f"runtime file hash differs: {relative}")
    return {"status": "environment_verified", "python": manifest["python"],
            "environment_directory": str(expected_prefix), "packages": actual,
            "critical_files_checked": len(manifest["critical_files"]),
            "model_imported": False, "native_execution": False}


def validate_launch(packet, receipt, *, packet_path, manifest_path, root=ROOT):
    expected = {
        "execution_environment_sha256": digest(manifest_path),
        "launcher_sha256": digest(Path(__file__)),
        "approved_packet_sha256": digest(packet_path),
        "output_root": str(checked_path(root, packet["output_root"])),
        "guard_report": str(checked_path(root, packet["guard_report"])),
    }
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("external authorization lacks exact environment/launcher/packet/output binding")
    if (packet["execution_environment"]["sha256"] != expected["execution_environment_sha256"]
            or packet["launcher"]["sha256"] != expected["launcher_sha256"]):
        raise ValueError("review packet code/environment binding differs")
    for relative in [packet["output_root"], packet["guard_report"], packet["admission_receipt"]]:
        if checked_path(root, relative).exists():
            raise ValueError("attempt output/report must be fresh; no retries")
    protocol = checked_path(root, packet["protocol"]["path"])
    if digest(protocol) != packet["protocol"]["sha256"]:
        raise ValueError("reviewed source protocol differs")


def commands(packet, authorization, *, root=ROOT):
    python = sys.executable  # verified prefix; inherited by the runner's restart child
    output = str(checked_path(root, packet["output_root"]))
    config = str(checked_path(root, packet["protocol"]["path"]))
    common = ["--config", config, "--authorization", str(authorization), "--output", output]
    preflight = [python, str(root / "scripts/preflight_spark_intrinsic_readout_placement_v2.py"),
                 *common, "--launch", "--receipt", str(checked_path(root, packet["admission_receipt"]))]
    guard = [python, str(root / "scripts/run_capacity_preflight_spark_placement_v2.py"),
             "--entry-point", "run_spark_intrinsic_readout_placement_v2.py",
             "--driver-limit-gib", "50.0", "--swap-growth-limit-mib", "512.0",
             "--minimum-free-percent", "15.0", "--sustained-host-samples", "3",
             "--report", str(checked_path(root, packet["guard_report"])), "--",
             "--device", "mps", *common, "--stage", "readout",
             "--prepared", str(checked_path(root, packet["prepared_root"]))]
    return preflight, guard


def run(packet_path, authorization=None, *, launch=False, root=ROOT):
    packet_path = Path(packet_path)
    packet = json.loads(packet_path.read_text())
    manifest_path = checked_path(root, packet["execution_environment"]["path"])
    if (digest(manifest_path) != packet["execution_environment"]["sha256"]
            or digest(Path(__file__)) != packet["launcher"]["sha256"]):
        raise ValueError("reviewed environment or launcher changed")
    environment = validate_environment(json.loads(manifest_path.read_text()), root=root)
    print(json.dumps(environment, sort_keys=True), flush=True)
    if not launch:
        return 0
    if authorization is None:
        raise ValueError("external human authorization required before launch")
    authorization = Path(authorization).resolve()
    validate_launch(packet, json.loads(authorization.read_text()), packet_path=packet_path,
                    manifest_path=manifest_path, root=root)
    preflight, guard = commands(packet, authorization, root=root)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    result = subprocess.run(preflight, cwd=root, env=env)
    if result.returncode != 0:
        return result.returncode
    # Recheck after preflight; changing environment does not inherit admission.
    validate_environment(json.loads(manifest_path.read_text()), root=root)
    return subprocess.run(guard, cwd=root, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    try:
        return run(args.packet, args.authorization, launch=args.launch)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "refused_before_model_loading", "detail": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
