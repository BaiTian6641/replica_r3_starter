#!/usr/bin/env python3
"""Build deterministic, role-separated synthetic text episodes for NL1.

This is plumbing only: it creates invented glyph/meaning pairs, corrections,
unaffected/unknown/revoked/wrong-world controls and a legacy text retention
panel.  It creates no authentic personal records, no benchmark items and no
sealed heldout evidence.  Prompts and verified feedback are the only fields an
event may carry; ``route_key`` is not stored and must be derived from prompt
token ids at runtime.  Labels never enter routing or source observations.

Output must be a fresh directory.  The manifest records every file digest so a
runner can fail closed before reading an episode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "nested_text_episodes_v1.json"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _word(rng: random.Random, length: int) -> str:
    consonants = "bdfgklmnprstvz"
    vowels = "aeiou"
    letters = []
    for index in range(length):
        letters.append(rng.choice(consonants if index % 2 == 0 else vowels))
    return "".join(letters).capitalize()


def _invent_fact(rng: random.Random) -> tuple[str, str]:
    return _word(rng, 5), _word(rng, 6)


def _event(*, row_id: str, world: str, session: str, family: str,
           source_group: str, split_role: str, teaching_or_query: str,
           prompt: str, response: str, verified: bool, eligible: bool,
           source: str, feedback_type: str, target_available_at: int,
           causal_cutoff: int, supersedes: str | None = None,
           contamination_group: str | None = None) -> dict:
    if eligible and not verified:
        raise ValueError("eligible events require verified feedback")
    if eligible and split_role != "teaching":
        raise ValueError("only teaching-role rows may be eligible")
    return {
        "id": row_id,
        "world_uid": world,
        "scope_uid": f"{world}/text",
        "session_id": session,
        "concept_or_rule_family": family,
        "source_group": source_group,
        "contamination_group": contamination_group or source_group,
        "split_role": split_role,
        "teaching_or_query": teaching_or_query,
        "prompt": prompt,
        "expected_response": response,
        "prompt_sha256": sha256_text(prompt),
        "response_sha256": sha256_text(response),
        "verified": verified,
        "eligible": eligible,
        "source": source,
        "feedback_type": feedback_type,
        "supersedes": supersedes,
        "target_available_at": target_available_at,
        "causal_cutoff": causal_cutoff,
        "predict_before_reveal": teaching_or_query == "future_query",
        "labels_used_for_routing": False,
        "world_used_for_routing": False,
        "answer_label_used_for_observation": False,
    }


def _facts(seed: int, world: str, count: int) -> list[dict]:
    rng = random.Random(f"{seed}:{world}")
    facts = []
    seen = set()
    while len(facts) < count:
        glyph, meaning = _invent_fact(rng)
        if glyph in seen:
            continue
        seen.add(glyph)
        facts.append({"glyph": glyph, "meaning": meaning})
    return facts


def _teaching_prompt(glyph: str, index: int) -> str:
    forms = (
        "Which meaning is bound to the glyph {glyph}?",
        "What does the glyph {glyph} denote?",
        "State the recorded meaning of glyph {glyph}.",
    )
    return forms[index % len(forms)].format(glyph=glyph)


def _query_prompt(glyph: str, index: int) -> str:
    forms = (
        "What is the meaning of glyph {glyph}?",
        "Recall the value associated with glyph {glyph}.",
        "Name the recorded meaning for glyph {glyph}.",
    )
    return forms[index % len(forms)].format(glyph=glyph)


def build_episode_bundle(config: dict) -> dict[str, list[dict]]:
    required = (
        "seed", "primary_world", "wrong_world", "session_id", "facts_per_world",
        "teaching_events", "development_events", "sealed_final_events",
        "correction_pairs", "unaffected_events", "unknown_events",
        "revoked_events", "retention_events",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"episode config is missing {missing}")
    seed = config["seed"]
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    primary = config["primary_world"]
    wrong = config["wrong_world"]
    if not isinstance(primary, str) or not isinstance(wrong, str) or primary == wrong:
        raise ValueError("primary and wrong worlds must be distinct nonempty strings")
    session = config["session_id"]
    if not isinstance(session, str) or not session:
        raise ValueError("session_id must be nonempty")
    per_world = int(config["facts_per_world"])
    if per_world < 3:
        raise ValueError("at least three facts per world are required")
    primary_facts = _facts(seed, primary, per_world)
    wrong_rng = random.Random(f"{seed}:{wrong}:meanings")
    wrong_by_glyph = {fact["glyph"]: _word(wrong_rng, 6)
                      for fact in primary_facts}
    while any(wrong_by_glyph[fact["glyph"]] == fact["meaning"]
              for fact in primary_facts):
        for fact in primary_facts:
            if wrong_by_glyph[fact["glyph"]] == fact["meaning"]:
                wrong_by_glyph[fact["glyph"]] = _word(wrong_rng, 7)
    # A heldout world uses facts that never appear in teaching or development.
    final_facts = _facts(seed + 7919, "sealed_world_nonce", int(config["sealed_final_events"]))
    correction_count = min(int(config["correction_pairs"]), per_world - 2)
    corrected = {fact["glyph"]: f"{fact['meaning']}Revised" for fact in primary_facts[:correction_count]}
    revoked = {fact["glyph"] for fact in primary_facts[-int(config["revoked_events"]):]}
    unknown_rng = random.Random(f"{seed}:unknown")
    unknown_glyphs: list[str] = []
    used_glyphs = {fact["glyph"] for fact in primary_facts} | {
        fact["glyph"] for fact in final_facts}
    while len(unknown_glyphs) < int(config["unknown_events"]):
        candidate = _word(unknown_rng, 5)
        if candidate in used_glyphs or candidate in unknown_glyphs:
            continue
        unknown_glyphs.append(candidate)

    def base_time(index: int) -> int:
        return 1_000 + index

    rows: dict[str, list[dict]] = {name: [] for name in (
        "teaching", "development", "correction", "unaffected", "unknown",
        "revoked", "wrong_world", "wrong_world_teaching", "legacy_retention",
        "sealed_final")}

    for index in range(int(config["teaching_events"])):
        fact = primary_facts[index % len(primary_facts)]
        row_id = f"teach-{index:04d}"
        rows["teaching"].append(_event(
            row_id=row_id, world=primary, session=session,
            family=fact["glyph"], source_group=f"source-{fact['glyph']}",
            split_role="teaching", teaching_or_query="teaching",
            prompt=_teaching_prompt(fact["glyph"], index),
            response=fact["meaning"], verified=True, eligible=True,
            source="feedback", feedback_type="answer",
            target_available_at=base_time(index),
            causal_cutoff=base_time(index) + 10))

    for index in range(int(config["development_events"])):
        fact = primary_facts[index % len(primary_facts)]
        rows["development"].append(_event(
            row_id=f"dev-{index:04d}", world=primary, session=session,
            family=fact["glyph"], source_group=f"source-{fact['glyph']}",
            split_role="development", teaching_or_query="future_query",
            prompt=_query_prompt(fact["glyph"], index),
            response=fact["meaning"], verified=True, eligible=False,
            source="observation", feedback_type="observed_outcome",
            target_available_at=base_time(index) + 20,
            causal_cutoff=base_time(index) + 30))

    for index, fact in enumerate(primary_facts[:correction_count]):
        source_id = f"teach-{index:04d}"
        rows["correction"].append(_event(
            row_id=f"correction-{index:04d}", world=primary, session=session,
            family=fact["glyph"], source_group=f"source-{fact['glyph']}",
            split_role="teaching", teaching_or_query="teaching",
            prompt=f"Correction: what meaning is now bound to glyph {fact['glyph']}?",
            response=corrected[fact["glyph"]], verified=True, eligible=True,
            source="correction", feedback_type="correction",
            target_available_at=base_time(index) + 40,
            causal_cutoff=base_time(index) + 50, supersedes=source_id,
            contamination_group=f"correction-{fact['glyph']}"))

    unaffected_facts = [fact for fact in primary_facts[correction_count:]
                        if fact["glyph"] not in revoked]
    for index in range(int(config["unaffected_events"])):
        fact = unaffected_facts[index % len(unaffected_facts)]
        rows["unaffected"].append(_event(
            row_id=f"unaffected-{index:04d}", world=primary, session=session,
            family=fact["glyph"], source_group=f"source-{fact['glyph']}",
            split_role="development", teaching_or_query="future_query",
            prompt=_query_prompt(fact["glyph"], index + 2),
            response=fact["meaning"], verified=True, eligible=False,
            source="observation", feedback_type="observed_outcome",
            target_available_at=base_time(index) + 60,
            causal_cutoff=base_time(index) + 70))

    for index, glyph in enumerate(unknown_glyphs):
        rows["unknown"].append(_event(
            row_id=f"unknown-{index:04d}", world=primary, session=session,
            family=glyph, source_group=f"unknown-{glyph}",
            split_role="development", teaching_or_query="future_query",
            prompt=_query_prompt(glyph, index),
            response="unknown", verified=False, eligible=False,
            source="observation", feedback_type="observed_outcome",
            target_available_at=base_time(index) + 80,
            causal_cutoff=base_time(index) + 90))

    for index, glyph in enumerate(sorted(revoked)):
        rows["revoked"].append(_event(
            row_id=f"revoked-{index:04d}", world=primary, session=session,
            family=glyph, source_group=f"revoked-{glyph}",
            split_role="development", teaching_or_query="future_query",
            prompt=_query_prompt(glyph, index + 1),
            response="unknown", verified=False, eligible=False,
            source="observation", feedback_type="observed_outcome",
            target_available_at=base_time(index) + 100,
            causal_cutoff=base_time(index) + 110))

    for index, fact in enumerate(primary_facts[:int(config["development_events"])]):
        meaning = wrong_by_glyph.get(fact["glyph"])
        if meaning is None:
            break
        rows["wrong_world"].append(_event(
            row_id=f"wrong-world-{index:04d}", world=wrong, session=session,
            family=fact["glyph"], source_group=f"wrong-{fact['glyph']}",
            split_role="development", teaching_or_query="future_query",
            prompt=_query_prompt(fact["glyph"], index),
            response=meaning, verified=True, eligible=False,
            source="observation", feedback_type="observed_outcome",
            target_available_at=base_time(index) + 120,
            causal_cutoff=base_time(index) + 130,
            contamination_group=f"wrong-{fact['glyph']}"))

    for index in range(int(config["teaching_events"])):
        fact = primary_facts[index % len(primary_facts)]
        meaning = wrong_by_glyph[fact["glyph"]]
        rows["wrong_world_teaching"].append(_event(
            row_id=f"wrong-teach-{index:04d}", world=wrong, session=session,
            family=fact["glyph"], source_group=f"wrong-source-{fact['glyph']}",
            split_role="teaching", teaching_or_query="teaching",
            prompt=_teaching_prompt(fact["glyph"], index),
            response=meaning, verified=True, eligible=True,
            source="feedback", feedback_type="answer",
            target_available_at=base_time(index) + 140,
            causal_cutoff=base_time(index) + 150,
            contamination_group=f"wrong-{fact['glyph']}"))

    legacy_prompts = (
        ("What is 17 + 26?", "43"),
        ("What is 8 times 7?", "56"),
        ("Complete the sentence: The library closes at dusk.", "closes"),
        ("Name the capital of France.", "Paris"),
        ("What is 144 divided by 12?", "12"),
        ("Which word is a verb: stone, run, blue?", "run"),
        ("What is 9 squared?", "81"),
        ("How many days are in a leap year?", "366"),
    )
    for index in range(int(config["retention_events"])):
        prompt, response = legacy_prompts[index % len(legacy_prompts)]
        rows["legacy_retention"].append(_event(
            row_id=f"legacy-{index:04d}", world=primary, session=session,
            family="legacy-text", source_group="legacy-retention",
            split_role="development", teaching_or_query="future_query",
            prompt=prompt, response=response, verified=True, eligible=False,
            source="observation", feedback_type="observed_outcome",
            target_available_at=base_time(index) + 160,
            causal_cutoff=base_time(index) + 170,
            contamination_group="legacy-retention"))

    for index, fact in enumerate(final_facts[:int(config["sealed_final_events"])]):
        rows["sealed_final"].append(_event(
            row_id=f"final-{index:04d}", world="sealed_world_nonce",
            session=session, family=fact["glyph"],
            source_group=f"final-{fact['glyph']}",
            split_role="sealed_final", teaching_or_query="future_query",
            prompt=_query_prompt(fact["glyph"], index),
            response=fact["meaning"], verified=True, eligible=False,
            source="observation", feedback_type="observed_outcome",
            target_available_at=base_time(index) + 180,
            causal_cutoff=base_time(index) + 190))
    version = config.get("schema_version", 1)
    if version == 1:
        return rows
    if version != 2:
        raise ValueError(f"unsupported episode schema version: {version}")
    corrections = config.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        raise ValueError("schema v2 requires explicit corrections")
    for correction in corrections:
        if set(correction) != {"role", "id", "prompt", "reason"}:
            raise ValueError("each v2 correction must name role, id, prompt and reason")
        role = correction["role"]
        matches = [row for row in rows.get(role, []) if row["id"] == correction["id"]]
        if len(matches) != 1:
            raise ValueError(f"v2 correction does not identify exactly one row: {correction}")
        row = matches[0]
        if not isinstance(correction["prompt"], str) or not correction["prompt"].strip():
            raise ValueError("v2 correction prompt must be nonempty")
        if not isinstance(correction["reason"], str) or not correction["reason"].strip():
            raise ValueError("v2 correction reason must be nonempty")
        row["prompt"] = correction["prompt"]
        row["prompt_sha256"] = sha256_text(row["prompt"])
    return rows


def write_bundle(bundle: dict[str, list[dict]], config: dict, output: Path) -> dict:
    role_files = config["role_files"]
    missing = set(role_files) - set(bundle)
    if missing:
        raise ValueError(f"bundle lacks declared role files {sorted(missing)}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a fresh/empty directory and is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    counts = {}
    for role, name in role_files.items():
        rows = bundle[role]
        payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                          for row in rows)
        path = output / name
        path.write_text(payload, encoding="utf-8")
        files[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        counts[role] = len(rows)
    version = config.get("schema_version", 1)
    manifest = {
        "schema_version": version,
        "format": f"replica.nested_text.episode_manifest.v{version}",
        "deterministic": True,
        "seed": config["seed"],
        "session_id": config["session_id"],
        "primary_world": config["primary_world"],
        "wrong_world": config["wrong_world"],
        "role_files": role_files,
        "counts": counts,
        "files_sha256": files,
        "routing": config["routing"],
        "access_guidance": {
            "teaching": "teaching and correction rows only",
            "development": "development, unaffected, unknown, revoked and retention controls",
            "sealed_final": "open once after development gates pass; never retry after failure",
            "wrong_world": "reserved wrong-world controls and optional wrong-world teaching"
        },
        "non_claims": config["non_claims"],
    }
    if version == 2:
        manifest["parent"] = config["parent"]
        manifest["corrections"] = config["corrections"]
    elif version != 1:
        raise ValueError(f"unsupported episode schema version: {version}")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be a fresh or empty directory")
    try:
        bundle = build_episode_bundle(config)
        manifest = write_bundle(bundle, config, args.output)
    except (ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({"status": "prepared", "output": str(args.output),
                      "counts": manifest["counts"],
                      "manifest_sha256": hashlib.sha256(
                          (args.output / "manifest.json").read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
