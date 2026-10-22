#!/usr/bin/env python3
"""Build/validate a tiny synthetic source-bound register slice; no model/network."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replica_lab.contracts import canonical_json
from replica_lab.episodes import audit_episode_families, export_sft, validate_episode, verify_episode

ROLES = ("train", "dev", "calibration", "promotion", "audit")
FAMILY = "synthetic-register-mask-xor-v1"
PERMISSIONS = {role: (["local_retrieval", "local_training", "local_evaluation"] if role == "train"
                      else ["local_retrieval", "local_evaluation"] if role == "dev"
                      else ["local_evaluation"]) for role in ROLES}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (canonical_json(value) + "\n").encode("utf-8")


def implementation():
    return {"generator_sha256": digest(Path(__file__).read_bytes()),
            "episode_compiler_sha256": digest((ROOT / "src/replica_lab/episodes.py").read_bytes()),
            "sft_contract_sha256": digest((ROOT / "src/replica_lab/contracts.py").read_bytes())}


def _safe_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source path must remain inside release")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("source path escapes release")
    return path


def _question(cutoff, missing):
    return (f"At observation cutoff {cutoff}, compute INPUT XOR MASK using width=8 registers. "
            + ("If a register is unavailable, ask for it." if missing == "ask"
               else "If a register is unavailable, respond UNKNOWN."))


def _world(assignment, seed):
    world = assignment["world_id"]
    numbers = hashlib.sha256(f"{seed}|{world}".encode()).digest()
    initial, mask = numbers[0], numbers[1]
    corrected = mask ^ (1 << (numbers[2] % 8))
    header = {**assignment, "schema_version": FAMILY, "synthetic": True, "width": 8,
              "source_id": f"synthetic://episode-slice-v1/{world}",
              "subject_id": f"synthetic-fixture:{world}", "branch_id": "source",
              "actor_id": "synthetic-register-generator", "privacy": "public",
              "permissions": PERMISSIONS[assignment["split"]], "seed": seed}

    def event(name, register, value, event_time, observed_at, operation="observe", replaces=None):
        return {"event_id": f"{world}.{name}", "source_id": header["source_id"],
                "event_time": event_time, "observed_at": observed_at, "operation": operation,
                "replaces_event_id": f"{world}.{replaces}" if replaces else None,
                "channel": "observed", "text": f"Register {register} = {value}." if value is not None else ""}

    events = [event("input", "INPUT", initial, 1, 1)]
    if assignment["scenario"] == "late_correction":
        events += [event("mask", "MASK", mask, 2, 2),
                   event("mask-corrected", "MASK", corrected, 3, 8, "correct", "mask")]
        # Labels come from generator parameters, independently of the archived-source solver.
        queries = [(5, "unknown", "answer", str(initial ^ mask), ["input", "mask"]),
                   (9, "unknown", "answer", str(initial ^ corrected), ["input", "mask-corrected"])]
    else:
        events += [event("mask-late", "MASK", mask, 2, 8),
                   event("mask-deleted", "MASK", None, 9, 10, "tombstone", "mask-late")]
        queries = [(5, "ask", "ask", "Please provide MASK.", ["input"]),
                   (11, "unknown", "unknown", "UNKNOWN", ["input"])]
    return header, events, queries


def _archive_source(header, events):
    chunks = [json_bytes(header)]
    spans, offset = [], len(chunks[0])
    for event in events:
        chunk = json_bytes(event)
        spans.append({"event_id": event["event_id"], "byte_start": offset,
                      "byte_end": offset + len(chunk), "sha256": digest(chunk)})
        chunks.append(chunk)
        offset += len(chunk)
    raw = b"".join(chunks)
    return raw, {"source_id": header["source_id"], "world_id": header["world_id"],
                 "group_id": header["group_id"], "split": header["split"],
                 "path": f"raw_sources/{header['world_id']}.jsonl", "sha256": digest(raw),
                 "bytes": len(raw), "event_spans": spans}


def _read_bound_source(payload, release_root, source_record):
    """Bind claimed episode metadata/event text to exact archived source bytes."""
    payload = validate_episode(payload).to_dict()
    raw = _safe_file(release_root, source_record["path"]).read_bytes()
    if digest(raw) != source_record["sha256"] or len(raw) != source_record["bytes"]:
        raise ValueError("raw source bytes/hash mismatch")
    lines = raw.splitlines(keepends=True)
    header, events = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]
    if len(events) != len(source_record["event_spans"]):
        raise ValueError("source span inventory differs")
    offset = len(lines[0])
    for event, line, span in zip(events, lines[1:], source_record["event_spans"]):
        expected = {"event_id": event["event_id"], "byte_start": offset,
                    "byte_end": offset + len(line), "sha256": digest(line)}
        if span != expected or raw[span["byte_start"]:span["byte_end"]] != line:
            raise ValueError("raw source span mismatch")
        offset += len(line)
    if header.get("schema_version") != FAMILY or header.get("synthetic") is not True:
        raise ValueError("not a declared synthetic register world")
    if any(source_record[key] != header[key] for key in ("world_id", "group_id", "split")):
        raise ValueError("source manifest lineage differs from raw source")
    if len(payload["sources"]) != 1 or payload["events"] != events:
        raise ValueError("episode events differ from bound raw source")
    source = payload["sources"][0]
    for key in ("source_id", "group_id", "split", "subject_id", "branch_id", "actor_id", "privacy"):
        if source[key] != header[key]:
            raise ValueError("source identity/role differs from raw source")
    if (source["sha256"] != digest(raw) or source["source_id"] != source_record["source_id"]
            or source["kind"] != "synthetic"):
        raise ValueError("episode source hash/kind binding differs")
    for permissions in (payload["permissions"], source["permissions"]):
        if not set(permissions).issubset(header["permissions"]):
            raise ValueError("episode escalates raw source use permissions")
    return payload, header, events


def independent_check(payload, release_root, source_record):
    """Parse archived observations and solve without generator labels/oracle fields."""
    payload, header, events = _read_bound_source(payload, release_root, source_record)
    match = re.fullmatch(
        r"At observation cutoff (\d+), compute INPUT XOR MASK using width=8 registers\. "
        r"If a register is unavailable, (ask for it|respond UNKNOWN)\.", payload["query"]["prompt"])
    if not match or int(match[1]) != payload["query"]["at"] or header["width"] != 8:
        raise ValueError("question/cutoff does not match the fixed solver family")
    active, registers = {}, {}
    for event in events:
        if event["observed_at"] > payload["query"]["at"]:
            continue
        replaced = event["replaces_event_id"]
        if replaced is not None:
            register = active.pop(replaced)
            registers.pop(register, None)
        if event["operation"] != "tombstone":
            parsed = re.fullmatch(r"Register (INPUT|MASK) = (\d+)\.", event["text"])
            if not parsed or not 0 <= int(parsed[2]) < 256:
                raise ValueError("source register observation is not solver-readable")
            register = parsed[1]
            active[event["event_id"]] = register
            registers[register] = (int(parsed[2]), event["event_id"])
    missing = [name for name in ("INPUT", "MASK") if name not in registers]
    support = [event_id for _, event_id in registers.values()]
    if missing:
        action = "ask" if match[2] == "ask for it" else "unknown"
        response = f"Please provide {', '.join(missing)}." if action == "ask" else "UNKNOWN"
    else:
        action, response = "answer", str(registers["INPUT"][0] ^ registers["MASK"][0])
    return payload["target"] == {"action": action, "response": response, "support_event_ids": support}


def build_slice(output, *, seed=23):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}

    def write(relative, data):
        path = _safe_file(output, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[relative] = digest(data)

    # Role assignment is complete before worlds, queries or targets are generated.
    assignments = [{"world_id": f"{role}-world-{index}", "group_id": f"register-{role}-world-{index}",
                    "split": role, "scenario": "late_correction" if index == 0 else "late_then_deleted"}
                   for role in ROLES for index in range(2)]
    write("world_roles.json", json_bytes(assignments))
    sources, payloads, derivations = [], [], []
    for assignment in assignments:
        header, events, queries = _world(assignment, seed)
        raw, source_record = _archive_source(header, events)
        write(source_record["path"], raw)
        sources.append(source_record)
        parent = None
        for index, (cutoff, missing, action, response, support) in enumerate(queries):
            source = {key: header[key] for key in ("source_id", "group_id", "split", "subject_id",
                                                   "branch_id", "actor_id", "privacy", "permissions")}
            source.update(kind="synthetic", sha256=source_record["sha256"])
            payload = {"schema_version": "replica.episode.v1", "id": f"{header['world_id']}.query-{index}",
                       "group_id": header["group_id"], "split": header["split"],
                       "subject_id": header["subject_id"], "branch_id": "source",
                       "parent_episode_id": parent, "privacy": "public", "permissions": header["permissions"],
                       "query": {"at": cutoff, "prompt": _question(cutoff, missing)},
                       "sources": [source], "events": events,
                       "target": {"action": action, "response": response,
                                  "support_event_ids": [f"{header['world_id']}.{name}" for name in support]},
                       "evaluator": {"oracle": {"family": FAMILY, "raw_source_path": source_record["path"]}}}
            canonical = validate_episode(payload)
            payloads.append(canonical.to_dict())
            derivations.append({"episode_id": payload["id"], "episode_sha256": canonical.sha256,
                                "parent_episode_id": parent, "group_id": header["group_id"],
                                "split": header["split"], "source_id": header["source_id"],
                                "source_sha256": source_record["sha256"], "query_cutoff": cutoff,
                                "source_span_ids": [s["event_id"] for s in source_record["event_spans"]]})
            parent = payload["id"]
    audit = audit_episode_families(payloads)
    write("source_manifest.json", json_bytes(sources))
    write("derivation_manifest.json", json_bytes(derivations))
    source_map = {source["source_id"]: source for source in sources}
    code = implementation()
    for role in ROLES:
        selected = [p for p in payloads if p["split"] == role]
        prefix = "canonical" if role in {"train", "dev"} else "evaluator_only"
        write(f"{prefix}/{role}_episodes.jsonl", b"".join(json_bytes(p) for p in selected))
        receipts = [verify_episode(p, verifier_id=FAMILY + ".source-solver", verifier_sha256=code["generator_sha256"],
                                  verifier=lambda value: independent_check(value, output,
                                      source_map[value["sources"][0]["source_id"]])) for p in selected]
        if not all(receipt.passed for receipt in receipts):
            raise ValueError("independent raw-source solver rejected generated targets")
        write(f"{prefix}/{role}_receipts.jsonl", b"".join(json_bytes(r.to_dict()) for r in receipts))
        if role in {"train", "dev"}:
            purpose = "local_training" if role == "train" else "local_evaluation"
            rows = [export_sft(p, purpose=purpose, verification=r) for p, r in zip(selected, receipts)]
            write(f"exports/{role}.jsonl", b"".join(json_bytes(r) for r in rows))
    manifest = {"schema_version": "replica.episode_slice.v1", "synthetic": True, "family": FAMILY,
                "seed": seed, "worlds_per_role": 2, "queries_per_world": 2, "implementation": code,
                "tokenizer_independent": True, "lineage_audit": audit,
                "export_counts": {"train": 4, "dev": 4},
                "audit_exposure": "Role-designated synthetic contract fixtures; not a sealed scientific benchmark.",
                "files_sha256": dict(sorted(files.items()))}
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    validate_release(output)
    return manifest


def validate_release(release_root):
    release_root = Path(release_root)
    manifest = json.loads((release_root / "manifest.json").read_bytes())
    if manifest.get("schema_version") != "replica.episode_slice.v1" or manifest["implementation"] != implementation():
        raise ValueError("release format/implementation hashes differ")
    actual = {p.relative_to(release_root).as_posix() for p in release_root.rglob("*") if p.is_file()}
    if actual != set(manifest["files_sha256"]) | {"manifest.json"}:
        raise ValueError("release file inventory differs")
    for relative, expected in manifest["files_sha256"].items():
        if digest(_safe_file(release_root, relative).read_bytes()) != expected:
            raise ValueError(f"release bytes mismatch: {relative}")
    sources = json.loads((release_root / "source_manifest.json").read_bytes())
    source_map = {s["source_id"]: s for s in sources}
    assignments = json.loads((release_root / "world_roles.json").read_bytes())
    if len(source_map) != 10 or len(assignments) != 10:
        raise ValueError("release requires ten independent source worlds")
    if {(a["world_id"], a["group_id"], a["split"]) for a in assignments} != {
            (s["world_id"], s["group_id"], s["split"]) for s in sources}:
        raise ValueError("preassigned world roles differ from source lineage")
    payloads = []
    expected_derivations = []
    for role in ROLES:
        prefix = "canonical" if role in {"train", "dev"} else "evaluator_only"
        values = [json.loads(line) for line in (release_root / prefix / f"{role}_episodes.jsonl").read_bytes().splitlines()]
        receipts = []
        for value in values:
            if value["split"] != role or not independent_check(value, release_root, source_map[value["sources"][0]["source_id"]]):
                raise ValueError("episode role/solver verification failed")
            source_record = source_map[value["sources"][0]["source_id"]]
            receipts.append(verify_episode(value, verifier_id=FAMILY + ".source-solver",
                verifier_sha256=manifest["implementation"]["generator_sha256"],
                verifier=lambda v: independent_check(v, release_root, source_record)))
            expected_derivations.append({"episode_id": value["id"],
                "episode_sha256": validate_episode(value).sha256,
                "parent_episode_id": value["parent_episode_id"], "group_id": value["group_id"],
                "split": role, "source_id": source_record["source_id"],
                "source_sha256": source_record["sha256"], "query_cutoff": value["query"]["at"],
                "source_span_ids": [s["event_id"] for s in source_record["event_spans"]]})
        stored_receipts = [json.loads(line) for line in
            (release_root / prefix / f"{role}_receipts.jsonl").read_bytes().splitlines()]
        if stored_receipts != [r.to_dict() for r in receipts]:
            raise ValueError("stored verification receipts differ")
        if role in {"train", "dev"}:
            purpose = "local_training" if role == "train" else "local_evaluation"
            expected = [export_sft(v, purpose=purpose, verification=r) for v, r in zip(values, receipts)]
            stored = [json.loads(line) for line in (release_root / "exports" / f"{role}.jsonl").read_bytes().splitlines()]
            if stored != expected:
                raise ValueError("stored SFT export differs from source-bound compilation")
        payloads.extend(values)
    if json.loads((release_root / "derivation_manifest.json").read_bytes()) != expected_derivations:
        raise ValueError("derivation lineage differs from canonical episodes")
    if audit_episode_families(payloads) != manifest["lineage_audit"]:
        raise ValueError("release lineage counts differ")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if (args.output is None) == (args.validate is None):
        parser.error("choose exactly one of --output or --validate")
    result = validate_release(args.validate) if args.validate else build_slice(args.output, seed=args.seed)
    print(json.dumps({"episodes": result["lineage_audit"]["episodes"], "worlds": result["lineage_audit"]["lineage_groups"],
                      "split_counts": result["lineage_audit"]["split_counts"], "export_counts": result["export_counts"]}, indent=2))


if __name__ == "__main__":
    main()
