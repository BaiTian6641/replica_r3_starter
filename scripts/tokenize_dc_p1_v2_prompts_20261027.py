#!/usr/bin/env python3
"""Measure sealed prompt/teaching token lengths with the sealed tokenizer.

This is the one offline step the preparation package cannot run for itself: the
exact token counts need the MiniCPM5 tokenizer files.  The command

* verifies the sealed tokenizer file hashes against the protocol binding before
  loading anything,
* loads the tokenizer forwarder offline (``local_files_only``,
  ``trust_remote_code=False``),
* tokenizes every sealed prompt and teaching target with the declared prompt
  encoding policy,
* writes ``token_lengths.v2.json`` into the prepared root with its own digest,
* prints the exact config binding to record.

It loads no model, runs no generation, trains nothing and writes no
authorization.  It refuses an existing artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from dc_protocol_common_20261027 import (  # noqa: E402
    ProtocolError, read_json, read_jsonl, require, seal_record, sha256_file,
    tokenizer_binding_sha, validate_manifest, validate_tokenizer_binding,
    validate_tokenizer_directory, write_json_atomic,
)
import dc_p1_native_response_v2_20261027 as protocol  # noqa: E402


def measure(config_path: Path, prepared: Path, tokenizer_dir: Path) -> dict:
    config = read_json(config_path)
    protocol.validate_config(config)
    manifest = validate_manifest(
        prepared, config, expected_files=protocol.FILES,
        manifest_format=protocol.MANIFEST_FORMAT,
        protocol_id=protocol.PROTOCOL_ID)
    binding = validate_tokenizer_binding(config["tokenizer_binding"])
    validate_tokenizer_directory(binding, tokenizer_dir)
    encoding = config.get("generation", {}).get("prompt_encoding")
    require(isinstance(encoding, dict), "tokenize",
            "generation.prompt_encoding is required")
    require(encoding.get("chat_template") is True, "tokenize",
            "prompts must use the native chat template")
    require(encoding.get("add_generation_prompt") is True, "tokenize",
            "prompts must use a generation prompt")
    require(encoding.get("enable_thinking") is False, "tokenize",
            "thinking must be disabled")
    require(encoding.get("add_special_tokens") is False, "tokenize",
            "chat-template special tokens must be disabled")

    from transformers import AutoTokenizer  # noqa: PLC0415 - offline tokenizer only

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False,
        revision=binding["revision"])

    def count(text: str, *, special: bool) -> int:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(rendered, add_special_tokens=False,
                        truncation=False)["input_ids"]
        return len(ids)

    prompts: dict[str, int] = {}
    teaching_prompts: dict[str, int] = {}
    targets: dict[str, int] = {}
    for name in protocol.FILES:
        for row in read_jsonl(prepared / name):
            if "prompt_sha256" not in row:
                continue
            digest = row["prompt_sha256"]
            source_included = "source" in row
            count_for_prompt = count(
                f"{row['source']}\n{row['prompt']}" if source_included
                else row["prompt"], special=True)
            target = teaching_prompts if source_included else prompts
            if digest in target:
                require(target[digest] == count_for_prompt, "tokenize",
                        f"prompt {digest} counted twice with different lengths")
            target[digest] = count_for_prompt
            if "target" in row:
                targets[row["case_id"]] = count(row["target"], special=False)

    teaching = read_jsonl(prepared / "teaching.jsonl") + read_jsonl(
        prepared / "sealed_final_teaching.jsonl")
    teaching_sequences = {}
    for row in teaching:
        teaching_sequences[row["case_id"]] = (
            teaching_prompts[row["prompt_sha256"]]
            + targets[row["case_id"]]
            + 1)  # the terminal EOS supervised by both masks
    evaluation_prompts = {}
    for name in ("fit_cases.jsonl", "dev_cases.jsonl",
                 "sealed_final_fit_cases.jsonl", "sealed_final_cases.jsonl"):
        for row in read_jsonl(prepared / name):
            evaluation_prompts[row["case_id"]] = prompts[row["prompt_sha256"]]
    maxima = {
        "evaluation_prompt_tokens": max(evaluation_prompts.values()),
        "teaching_prompt_tokens": max(teaching_prompts.values()),
        "teaching_sequence_tokens": max(teaching_sequences.values()),
        "retention_prompt_tokens": max(
            evaluation_prompts[row["case_id"]]
            for row in read_jsonl(prepared / "dev_cases.jsonl")
            + read_jsonl(prepared / "sealed_final_cases.jsonl")
            if row["role"] == "retention"),
    }
    body = {
        "format": protocol.TOKEN_LENGTHS_FORMAT,
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_config_sha256": sha256_file(config_path),
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "tokenizer_binding_sha256": tokenizer_binding_sha(binding),
        "tokenizer_files_verified": True,
        "tokenizer_dir": str(tokenizer_dir),
        "prompt_encoding": encoding,
        "counts": {
            "prompts": len(prompts),
            "targets": len(targets),
            "teaching_sequences": len(teaching_sequences),
            "evaluation_prompts": len(evaluation_prompts),
        },
        "maxima": maxima,
        "prompts": prompts,
        "targets": targets,
        "teaching_sequences": teaching_sequences,
        "evaluation_prompts": evaluation_prompts,
        "declared_by": "scripts/tokenize_dc_p1_v2_prompts_20261027.py",
        "native_model_loaded": False,
        "generation": False,
        "downloads": 0,
    }
    artifact = seal_record(body)
    require(maxima["evaluation_prompt_tokens"]
            <= config["generation"]["evaluation_prompt_tokens_max"], "tokenize",
            "measured evaluation prompts exceed the declared ceiling")
    require(maxima["teaching_sequence_tokens"]
            <= config["generation"]["teaching_tokens_per_event_max"],
            "tokenize",
            "measured teaching sequences exceed the declared ceiling")
    schedule = protocol.validate_work_units(config, token_lengths=artifact)
    return {
        "artifact": artifact,
        "schedule": schedule,
        "config_binding": {
            "prepared.token_lengths.status": "sealed",
            "prepared.token_lengths.sha256": "set to the artifact file digest",
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=protocol.CONFIG_PATH)
    parser.add_argument("--prepared", type=Path, default=protocol.PREPARED_DIR)
    parser.add_argument("--tokenizer-dir", type=Path,
                        default=ROOT / "checkpoints" / "minicpm5_source")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or (args.prepared / protocol.TOKEN_LENGTHS_PATH.name)
    try:
        require(not output.exists(), "fresh_output",
                f"refusing to overwrite {output}")
        measured = measure(args.config, args.prepared, args.tokenizer_dir)
        write_json_atomic(output, measured["artifact"])
        digest = sha256_file(output)
        print(json.dumps({
            "status": "sealed",
            "path": str(output),
            "artifact_sha256": digest,
            "receipt_sha256": measured["artifact"]["receipt_sha256"],
            "maxima": measured["artifact"]["maxima"],
            "measured_counters": measured["schedule"]["counters"],
            "next_step": {
                "prepared.token_lengths.status": "sealed",
                "prepared.token_lengths.sha256": digest,
                "note": "record this binding in the protocol config, then "
                        "re-seal the config and re-run the v2 preflight review",
            },
        }, indent=2, sort_keys=True))
    except ProtocolError as error:
        print(json.dumps({"status": "refused", **error.as_dict()},
                         indent=2, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
