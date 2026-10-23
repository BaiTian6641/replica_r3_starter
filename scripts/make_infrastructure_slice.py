#!/usr/bin/env python3
"""Build or revalidate the preregistered 2,000-target synthetic release offline."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replica_lab.contracts import canonical_json
from replica_lab.episodes import audit_episode_families, export_sft, validate_episode, verify_episode
from replica_lab.infrastructure_solver import independent_check

DESIGN_PATH = ROOT / "configs/infrastructure_slice_v2_design.json"
VERSION = "replica.infrastructure_slice.v2"
VERIFIER_ID = "replica.infrastructure.raw-source-solver.v2"
FAMILY_SHORT = {"register_xor8": "reg", "affine_mod97": "aff", "finite_state": "fsm",
                "tool_recovery": "tool"}
ROLE_SHORT = {"train": "tr", "dev": "de", "calibration": "ca", "promotion": "pr", "audit": "au"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (canonical_json(value) + "\n").encode("utf-8")


def _design():
    design = json.loads(DESIGN_PATH.read_bytes())
    if (design["schema_version"] != "replica.infrastructure_slice_design.v2"
            or design["world_count"] != 1000 or design["canonical_target_count"] != 2000
            or design["queries_per_world"] != 2
            or set(design["family_order"]) != set(FAMILY_SHORT)
            or set(design["roles_order"]) != set(ROLE_SHORT)):
        raise ValueError("unsupported infrastructure design")
    return design


def implementation():
    return {"generator_sha256": digest(Path(__file__).read_bytes()),
            "solver_sha256": digest((ROOT / "src/replica_lab/infrastructure_solver.py").read_bytes()),
            "episode_compiler_sha256": digest((ROOT / "src/replica_lab/episodes.py").read_bytes()),
            "sft_contract_sha256": digest((ROOT / "src/replica_lab/contracts.py").read_bytes()),
            "design_sha256": digest(DESIGN_PATH.read_bytes())}


def _safe_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("release path must remain inside release")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("release path escapes release")
    return path


def _assignments(design, seed):
    values = []
    for role in design["roles_order"]:
        for family_index, family in enumerate(design["family_order"]):
            for index in range(design["role_family_world_counts"][role]):
                world = f"i2s{seed}-{ROLE_SHORT[role]}-{FAMILY_SHORT[family]}-{index:03d}"
                values.append({"world_id": world, "group_id": world, "split": role,
                               "family": family, "scenario": design["scenarios"][(index + family_index) % 4]})
    if len(values) != design["world_count"]:
        raise ValueError("design world accounting differs")
    return values


def _question(design, family, cutoff, missing):
    return design["query_template"].format(cutoff=cutoff,
        query_body=design["families"][family]["query_body"],
        missing_instruction=design["missing_instructions"][missing])


def _generated_answer(family, rule, facts):
    """Generator-only label path from sampled values, never the archived-source solver."""
    if family == "register_xor8":
        return str(facts["INPUT"] ^ facts["MASK"])
    if family == "affine_mod97":
        return str((facts["A"] * facts["X"] + facts["B"]) % 97)
    if family == "finite_state":
        state = facts["START"]
        for symbol in facts["SYMBOLS"]:
            state = rule["transitions"][state][symbol]
        return state
    if facts["STATUS"] == "ok":
        return "VERIFY receipt"
    if facts["STATUS"] == "transient" and facts["ATTEMPT"] < facts["LIMIT"]:
        return "RETRY lookup"
    return "STOP"


def _sample_world(design, assignment, seed, attempt):
    family, world = assignment["family"], assignment["world_id"]
    rng = random.Random(int(digest(f"{seed}|{world}|{attempt}".encode()), 16))
    spec = design["families"][family]
    rule = json.loads(json.dumps(spec["rule"]))
    if family == "register_xor8":
        facts = {"INPUT": rng.randrange(256), "MASK": rng.randrange(256)}
        corrected = facts["MASK"] ^ (1 << rng.randrange(8))
    elif family == "affine_mod97":
        facts = {"A": rng.randrange(1, 97), "X": rng.randrange(97), "B": rng.randrange(97)}
        corrected = (facts["B"] + 1) % 97
    elif family == "finite_state":
        states = spec["states"]
        columns = {symbol: rng.sample(states, len(states)) for symbol in spec["alphabet"]}
        rule["transitions"] = {state: {symbol: columns[symbol][index] for symbol in spec["alphabet"]}
                               for index, state in enumerate(states)}
        facts = {"START": rng.choice(states),
                 "SYMBOLS": "".join(rng.choice(spec["alphabet"]) for _ in range(rng.randrange(2, 5)))}
        corrected = states[(states.index(facts["START"]) + 1) % len(states)]
    else:
        facts = {"STATUS": rng.choice(["ok", "transient", "permanent"]),
                 "ATTEMPT": rng.randrange(10), "LIMIT": rng.randrange(1, 11)}
        candidates = [status for status in ["ok", "transient", "permanent"]
                      if _generated_answer(family, rule, {**facts, "STATUS": status})
                      != _generated_answer(family, rule, facts)]
        corrected = rng.choice(candidates)
    header = {**assignment, "schema_version": design["source_header_schema"], "synthetic": True,
              "source_id": f"synthetic://{world}", "subject_id": f"fictional:{world}",
              "branch_id": "source", "actor_id": "infrastructure-generator-v2", "privacy": "public",
              "permissions": design["permissions"][assignment["split"]], "seed": seed}
    if set(header) != set(design["source_header_required"]):
        raise ValueError("source header differs from declared fields")

    def event(name, text, event_time, observed_at, operation="observe", replaces=None):
        return {"event_id": f"{world}.{name}", "source_id": header["source_id"],
                "event_time": event_time, "observed_at": observed_at, "operation": operation,
                "replaces_event_id": f"{world}.{replaces}" if replaces else None,
                "channel": "observed", "text": text}

    timing = design["source_timing"]
    events = [event("00-rule", "RULE " + canonical_json(rule), timing["rule_at"], timing["rule_at"])]
    decisive = spec["decisive_field"]
    decisive_name = None
    for index, field in enumerate(spec["fact_fields"]):
        name = f"{10 + index:02d}-{field}"
        if field == decisive:
            decisive_name = name
            continue
        events.append(event(name, "FACT " + canonical_json({"name": field, "value": facts[field]}),
                            timing["ordinary_facts_at"], timing["ordinary_facts_at"]))
    text = "FACT " + canonical_json({"name": decisive, "value": facts[decisive]})
    scenario = assignment["scenario"]
    initial_at = (timing["initial_decisive_observed_at"] if scenario == "late_correction"
                  else timing["late_decisive_observed_at"])
    events.append(event(decisive_name, text, timing["decisive_fact_event_time"], initial_at))
    if scenario == "late_correction":
        events.append(event("90-correct", "FACT " + canonical_json({"name": decisive, "value": corrected}),
                            timing["correction_event_time"], timing["correction_observed_at"],
                            "correct", decisive_name))
        queries = [(5, "unknown", facts), (9, "unknown", {**facts, decisive: corrected})]
    elif scenario == "tombstone":
        events.append(event("99-delete", "", timing["tombstone_event_time"],
                            timing["tombstone_observed_at"], "tombstone", decisive_name))
        queries = [(5, "ask", None), (11, "unknown", None)]
    else:
        missing = "ask" if scenario == "late_observation_ask" else "unknown"
        queries = [(5, missing, None), (9, "unknown", facts)]
    events.sort(key=lambda e: (e["observed_at"], e["event_id"]))
    generated = []
    for cutoff, missing, available_facts in queries:
        # This replay is independent of the verifier's raw-source parser/replay.
        active = {}
        for item in events:
            if item["observed_at"] > cutoff:
                continue
            if item["replaces_event_id"] is not None:
                active.pop(item["replaces_event_id"], None)
            if item["operation"] != "tombstone":
                active[item["event_id"]] = item
        if available_facts is None:
            action = missing
            response = (design["ask_response_template"].format(fields=decisive)
                        if missing == "ask" else "UNKNOWN")
        else:
            action, response = "answer", _generated_answer(family, rule, available_facts)
        generated.append((cutoff, missing, {"action": action, "response": response,
                                           "support_event_ids": list(active)}))
    return header, events, generated


def semantic_signature(events):
    positions = {item["event_id"]: index for index, item in enumerate(events)}
    normalized = [{"text": item["text"], "event_time": item["event_time"],
                   "observed_at": item["observed_at"], "operation": item["operation"],
                   "replaces_position": positions[item["replaces_event_id"]]
                   if item["replaces_event_id"] is not None else None} for item in events]
    return digest(json_bytes(normalized))


def _archive_source(header, events):
    chunks = [json_bytes(header)]
    spans, offset = [], len(chunks[0])
    for item in events:
        chunk = json_bytes(item)
        spans.append({"event_id": item["event_id"], "byte_start": offset,
                      "byte_end": offset + len(chunk), "sha256": digest(chunk)})
        chunks.append(chunk)
        offset += len(chunk)
    raw = b"".join(chunks)
    record = {key: header[key] for key in ("source_id", "world_id", "group_id", "split", "family", "scenario")}
    record.update(path=f"raw_sources/{header['world_id']}.jsonl", sha256=digest(raw), bytes=len(raw),
                  event_spans=spans, semantic_sha256=semantic_signature(events))
    return raw, record


def _derivation(payload, record):
    return {"episode_id": payload["id"], "episode_sha256": validate_episode(payload).sha256,
            "parent_episode_id": payload["parent_episode_id"], "group_id": payload["group_id"],
            "split": payload["split"], "family": record["family"], "scenario": record["scenario"],
            "source_id": record["source_id"], "source_sha256": record["sha256"],
            "source_semantic_sha256": record["semantic_sha256"], "query_cutoff": payload["query"]["at"],
            "source_span_ids": [span["event_id"] for span in record["event_spans"]]}


def _histograms(payloads, source_map):
    result = {name: Counter() for name in ("roles", "families", "scenarios", "actions", "responses")}
    by_role = defaultdict(lambda: {name: Counter() for name in ("families", "scenarios", "actions")})
    characters = {"query": [], "response": [], "source_event_text": []}
    for payload in payloads:
        record = source_map[payload["sources"][0]["source_id"]]
        role, action = payload["split"], payload["target"]["action"]
        for name, value in [("roles", role), ("families", record["family"]),
                            ("scenarios", record["scenario"]), ("actions", action),
                            ("responses", payload["target"]["response"])]:
            result[name][value] += 1
        for name, value in [("families", record["family"]), ("scenarios", record["scenario"]), ("actions", action)]:
            by_role[role][name][value] += 1
        characters["query"].append(len(payload["query"]["prompt"]))
        characters["response"].append(len(payload["target"]["response"]))
        characters["source_event_text"].append(sum(len(e["text"]) for e in payload["events"]))
    result = {key: dict(sorted(value.items())) for key, value in result.items()}
    result["by_role"] = {role: {name: dict(sorted(counts.items())) for name, counts in data.items()}
                         for role, data in sorted(by_role.items())}
    result["character_lengths"] = {name: {"min": min(values), "max": max(values),
        "sum": sum(values), "count": len(values)} for name, values in characters.items()}
    result["token_lengths"] = "Not measured by generator; external offline native-tokenizer admission required."
    return result


def _load_lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def build_release(output, *, seed=41):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    design, output = _design(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}

    def write(relative, data):
        path = _safe_file(output, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[relative] = digest(data)

    assignments = _assignments(design, seed)
    # Persist the complete role/family/scenario assignment before sampling any world.
    write("world_roles.json", json_bytes(assignments))
    sources, payloads, derivations, rejections, signatures = [], [], [], [], set()
    for assignment in assignments:
        for attempt in range(10000):
            header, events, queries = _sample_world(design, assignment, seed, attempt)
            signature = semantic_signature(events)
            if signature not in signatures:
                break
            rejections.append({**assignment, "attempt": attempt, "reason": "semantic_duplicate_world",
                               "semantic_sha256": signature})
        else:
            raise ValueError("cannot sample a distinct semantic world")
        signatures.add(signature)
        raw, record = _archive_source(header, events)
        write(record["path"], raw)
        sources.append(record)
        parent = None
        for index, (cutoff, missing, target) in enumerate(queries):
            source = {key: header[key] for key in ("source_id", "group_id", "split", "subject_id",
                                                   "branch_id", "actor_id", "privacy", "permissions")}
            source.update(kind="synthetic", sha256=record["sha256"])
            payload = {"schema_version": "replica.episode.v1", "id": f"{header['world_id']}.q{index}",
                       "group_id": header["group_id"], "split": header["split"],
                       "subject_id": header["subject_id"], "branch_id": "source", "parent_episode_id": parent,
                       "privacy": "public", "permissions": header["permissions"],
                       "query": {"at": cutoff, "prompt": _question(design, header["family"], cutoff, missing)},
                       "sources": [source], "events": events, "target": target,
                       "evaluator": {"oracle": {"family": header["family"], "scenario": header["scenario"],
                                                  "raw_source_path": record["path"]}}}
            payload = validate_episode(payload).to_dict()
            payloads.append(payload)
            derivations.append(_derivation(payload, record))
            parent = payload["id"]
    source_map = {source["source_id"]: source for source in sources}
    lineage = audit_episode_families(payloads)
    write("source_manifest.json", json_bytes(sources))
    write("derivation_manifest.json", json_bytes(derivations))
    write("rejected_candidates.jsonl", b"".join(json_bytes(item) for item in rejections))
    code = implementation()
    for role in design["roles_order"]:
        selected = [payload for payload in payloads if payload["split"] == role]
        prefix = "canonical" if role in design["exports"] else "evaluator_only"
        receipts = [verify_episode(payload, verifier_id=VERIFIER_ID, verifier_sha256=code["solver_sha256"],
                    verifier=lambda value: independent_check(value, output,
                        source_map[value["sources"][0]["source_id"]])) for payload in selected]
        if not all(receipt.passed for receipt in receipts):
            rejected = [{"episode": payload, "receipt": receipt.to_dict()}
                        for payload, receipt in zip(selected, receipts) if not receipt.passed]
            write("rejected_episodes.jsonl", b"".join(json_bytes(item) for item in rejected))
            raise ValueError("independent source solver rejected generated targets")
        write(f"{prefix}/{role}_episodes.jsonl", b"".join(json_bytes(payload) for payload in selected))
        write(f"{prefix}/{role}_receipts.jsonl", b"".join(json_bytes(receipt.to_dict()) for receipt in receipts))
        if role in design["exports"]:
            purpose = "local_training" if role == "train" else "local_evaluation"
            rows = [export_sft(payload, purpose=purpose, verification=receipt)
                    for payload, receipt in zip(selected, receipts)]
            write(f"exports/{role}.jsonl", b"".join(json_bytes(row) for row in rows))
    histograms = _histograms(payloads, source_map)
    manifest = {"schema_version": VERSION, "synthetic": True, "seed": seed,
                "implementation": code, "world_count": len(sources), "canonical_target_count": len(payloads),
                "lineage_audit": lineage, "histograms": histograms,
                "export_counts": {role: histograms["roles"][role] for role in design["exports"]},
                "semantic_duplicate_worlds_admitted": 0, "rejected_candidate_count": len(rejections),
                "teacher_used": False, "public_sources_ingested": False,
                "tokenizer_independent": True, "token_admission_status": "pending_external_offline_accounting",
                "generalization_scope": design["generalization_scope"], "audit_exposure": design["exposure"],
                "files_sha256": dict(sorted(files.items()))}
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    validate_release(output)
    return manifest


def validate_release(release_root):
    root, design = Path(release_root), _design()
    manifest = json.loads((root / "manifest.json").read_bytes())
    if manifest.get("schema_version") != VERSION or manifest["implementation"] != implementation():
        raise ValueError("release format/implementation hashes differ")
    declarations = {"synthetic": True, "teacher_used": False, "public_sources_ingested": False,
                    "tokenizer_independent": True,
                    "token_admission_status": "pending_external_offline_accounting",
                    "generalization_scope": design["generalization_scope"],
                    "audit_exposure": design["exposure"]}
    if any(type(manifest.get(key)) is not type(value) or manifest[key] != value
           for key, value in declarations.items()):
        raise ValueError("release scope/exposure/admission declarations differ from design")
    if type(manifest.get("seed")) is not int or manifest["seed"] < 0:
        raise ValueError("release seed must be a nonnegative integer")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(manifest["files_sha256"]) | {"manifest.json"}:
        raise ValueError("release file inventory differs")
    for relative, expected in manifest["files_sha256"].items():
        if digest(_safe_file(root, relative).read_bytes()) != expected:
            raise ValueError(f"release bytes mismatch: {relative}")
    assignments = json.loads((root / "world_roles.json").read_bytes())
    if assignments != _assignments(design, manifest["seed"]):
        raise ValueError("preassigned world roles/families/scenarios differ")
    sources = json.loads((root / "source_manifest.json").read_bytes())
    source_map = {source["source_id"]: source for source in sources}
    if len(sources) != design["world_count"] or len(source_map) != len(sources):
        raise ValueError("source world count/identity differs")
    assignment_fields = ("world_id", "group_id", "split", "family", "scenario")
    if [{key: source[key] for key in assignment_fields} for source in sources] != assignments:
        raise ValueError("source lineage differs from preassigned world roles")
    signatures = []
    for source in sources:
        raw_lines = _load_lines(_safe_file(root, source["path"]))
        if raw_lines[0]["seed"] != manifest["seed"]:
            raise ValueError("raw source seed differs from release")
        events = raw_lines[1:]
        signature = semantic_signature(events)
        if source["semantic_sha256"] != signature:
            raise ValueError("source semantic signature differs")
        signatures.append(signature)
    if len(set(signatures)) != len(signatures):
        raise ValueError("semantic duplicate worlds cross release")
    payloads, derivations, referenced = [], [], Counter()
    for role in design["roles_order"]:
        prefix = "canonical" if role in design["exports"] else "evaluator_only"
        values = _load_lines(root / prefix / f"{role}_episodes.jsonl")
        if len(values) != design["role_target_counts"][role]:
            raise ValueError("canonical role target count differs")
        receipts = []
        for payload in values:
            if payload["split"] != role or len(payload["sources"]) != 1:
                raise ValueError("canonical role/source inventory differs")
            source = source_map[payload["sources"][0]["source_id"]]
            if payload["group_id"] != source["group_id"] or source["split"] != role:
                raise ValueError("canonical source role/group differs")
            index = referenced[source["source_id"]]
            scenario = source["scenario"]
            cutoff = ([5, 11] if scenario == "tombstone" else [5, 9])[index] if index < 2 else None
            missing = "ask" if index == 0 and scenario in {"late_observation_ask", "tombstone"} else "unknown"
            expected_parent = f"{source['world_id']}.q0" if index == 1 else None
            if (payload["id"] != f"{source['world_id']}.q{index}"
                    or payload["parent_episode_id"] != expected_parent
                    or payload["query"] != {"at": cutoff,
                        "prompt": _question(design, source["family"], cutoff, missing)}):
                raise ValueError("canonical queries differ from fixed chronological world derivation")
            receipt = verify_episode(payload, verifier_id=VERIFIER_ID,
                verifier_sha256=manifest["implementation"]["solver_sha256"],
                verifier=lambda value: independent_check(value, root, source))
            if not receipt.passed:
                raise ValueError("independent source solver rejected canonical target")
            receipts.append(receipt)
            derivations.append(_derivation(payload, source))
            referenced[source["source_id"]] += 1
        if _load_lines(root / prefix / f"{role}_receipts.jsonl") != [r.to_dict() for r in receipts]:
            raise ValueError("stored verification receipts differ")
        if role in design["exports"]:
            purpose = "local_training" if role == "train" else "local_evaluation"
            rows = [export_sft(payload, purpose=purpose, verification=receipt)
                    for payload, receipt in zip(values, receipts)]
            if _load_lines(root / "exports" / f"{role}.jsonl") != rows:
                raise ValueError("stored SFT export differs from verified canonical compilation")
        payloads.extend(values)
    if set(referenced) != set(source_map) or set(referenced.values()) != {2}:
        raise ValueError("each source world must bind exactly two canonical queries")
    if json.loads((root / "derivation_manifest.json").read_bytes()) != derivations:
        raise ValueError("derivation lineage differs from canonical episodes")
    if audit_episode_families(payloads) != manifest["lineage_audit"]:
        raise ValueError("release lineage counts differ")
    histograms = _histograms(payloads, source_map)
    if histograms != manifest["histograms"]:
        raise ValueError("release histograms differ")
    for role, targets in design["role_target_counts"].items():
        expected = {action: int(targets * fraction) for action, fraction in design["role_action_proportions"].items()}
        if histograms["by_role"][role]["actions"] != expected:
            raise ValueError("role action balance differs from preregistration")
    if manifest["export_counts"] != {role: histograms["roles"][role] for role in design["exports"]}:
        raise ValueError("export count differs")
    if (manifest["world_count"] != len(sources) or manifest["canonical_target_count"] != len(payloads)
            or manifest["semantic_duplicate_worlds_admitted"] != 0
            or manifest["rejected_candidate_count"] != len(_load_lines(root / "rejected_candidates.jsonl"))):
        raise ValueError("manifest accounting differs")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if (args.output is None) == (args.validate is None):
        parser.error("choose exactly one of --output or --validate")
    result = validate_release(args.validate) if args.validate else build_release(args.output, seed=args.seed)
    print(json.dumps({key: result[key] for key in ("world_count", "canonical_target_count", "export_counts",
                                                 "rejected_candidate_count", "token_admission_status")}, indent=2))


if __name__ == "__main__":
    main()
