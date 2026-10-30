#!/usr/bin/env python3
"""Derive a current budget/navigation receipt from an exact v12 launch packet.

Host-side JSON/hash work only. This does not import the model, mutate the packet,
protocol, authorization, attempt marker, or native output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

FORMAT = "replica.intrinsic_memory.current_packet_budget.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive(root: Path, packet_path: Path) -> dict:
    packet_bytes = packet_path.read_bytes()
    packet = json.loads(packet_bytes)
    if packet.get("experiment_version") != 12:
        raise ValueError("not a v12 packet")
    binding = packet["protocol"]
    protocol_path = root / binding["path"]
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if protocol_sha256 != binding["sha256"]:
        raise ValueError("protocol digest differs from the packet binding")
    protocol = json.loads(protocol_bytes)
    arm = packet["arm"]
    spec = protocol["arms"][arm]
    budget = spec["budget"]
    if spec["total_reservations"] != {
            key: sum(stage[key] for stage in budget.values())
            for key in spec["total_reservations"]}:
        raise ValueError("protocol reservation total differs from its stages")
    return {
        "format": FORMAT,
        "packet_path": str(packet_path.relative_to(root)),
        "packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "protocol_path": binding["path"],
        "protocol_sha256": protocol_sha256,
        "arm": arm,
        "output_root": packet["output_root"],
        "attempts": packet["attempts"],
        "retries": packet["retries"],
        "stage_count": len(budget),
        "outer_wall_seconds": spec["wall_seconds"],
        "total_reservations": spec["total_reservations"],
        "stage_wall_seconds": {
            name: stage["wall_seconds"] for name, stage in budget.items()},
        "resource_limits": spec["resource_limits"],
        "gated_arms_not_prepared": packet["gated_arms_not_prepared"],
        "derivation": "read packet; verify pinned protocol SHA-256; select packet arm; recompute every reservation total from protocol stages",
        "native_outcome_inferred": False,
    }


def atomic_write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    packet = args.packet if args.packet.is_absolute() else root / args.packet
    document = derive(root, packet.resolve())
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        atomic_write(output, document)
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
