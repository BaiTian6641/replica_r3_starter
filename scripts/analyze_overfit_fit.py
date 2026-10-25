#!/usr/bin/env python3
"""Supervised-fit analysis for the overfit cohort (no updates).

Computes teacher-forced response loss and compares greedy decoding against the
independently recomputed response-token argmax for the same trained extension.
This separates a learning failure from a decoding/evaluation failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab.vision_diagnostic import select_cases  # noqa: E402
from replica_lab.vision_overfit import overfit_training_rows  # noqa: E402
from replica_lab.vision_qualification import (  # noqa: E402
    load_verified_dataset, require, resolve_project_path, sha256_file,
    verify_implementation_pins,
)
import run_vision_qualification as v1_runner  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args(argv)
    import torch
    from replica_lab.architecture.contracts import VisionSpec
    from replica_lab.architecture.loading import exclusive_mps_worker, load_local_mps
    from replica_lab.architecture.vision import VisionFeatureService
    from replica_lab.profiling import RunProfiler

    protocol = json.loads(args.protocol.read_text())
    verify_implementation_pins(protocol, ROOT)
    require(torch.backends.mps.is_available(), "MPS required; no CPU fallback")
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise RuntimeError("Launch through scripts/run_capacity_preflight.py")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=0 required")
    require(sha256_file(args.model_lock) == protocol["model"]["lock_sha256"],
            "Model lock bytes differ")
    dataset = load_verified_dataset(protocol)
    source = resolve_project_path(protocol, protocol["vision_source"]["path"])
    manifest_path = resolve_project_path(protocol, protocol["vision_source"]["manifest"]["path"])
    require(sha256_file(manifest_path) == protocol["vision_source"]["manifest"]["sha256"],
            "Vision manifest bytes changed")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Use a fresh analysis output path")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "replica.vision_overfit_fit_analysis.v1",
        "protocol_sha256": sha256_file(args.protocol),
        "extension": str(args.extension),
        "extension_manifest_sha256": sha256_file(args.extension / "manifest.json"),
        "extension_weights_sha256": sha256_file(args.extension / "extensions.safetensors"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "mps", "fallback_disabled": True, "optimizer_updates": 0,
    }
    profiler = RunProfiler("mps", output)
    try:
        with exclusive_mps_worker(ROOT):
            profiler.start()
            adapter, tokenizer, lock = load_local_mps(
                args.model_path, args.model_lock, profiler=profiler,
                vision=VisionSpec(**protocol["vision"]))
            adapter.load_extension(args.extension)
            scope = v1_runner.protocol_scope(protocol)
            service = VisionFeatureService(source, manifest_path, device="mps")
            session = v1_runner.make_session(adapter, tokenizer, v1_runner.empty_ledger(output))
            cases = select_cases(dataset, "train", limit=protocol["train_worlds"],
                                 per_world=protocol["per_world_cases"])
            rows = []
            for item in cases:
                case_id = item["case_id"]
                visual = service.observe(str(dataset.image_path(case_id)),
                                         source_id="vision-overfit", task_id=case_id,
                                         scope=scope, observed_at=0, source_epoch=0)
                prompt = v1_runner.prompt_ids(tokenizer, item["row"]["question"],
                                              adapter.base.get_input_embeddings().weight.device,
                                              256)
                response = tokenizer(item["row"]["question"], add_special_tokens=False)["input_ids"]
                response_ids = tokenizer(dataset.label(case_id), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
                input_ids = torch.cat((prompt, torch.tensor([response_ids], dtype=torch.long,
                                                            device=prompt.device)), dim=1)
                labels = torch.tensor([[-100] * prompt.shape[1] + response_ids],
                                      dtype=torch.long, device=prompt.device)
                with session.training_request(
                        prompt_ids=prompt, prompt_mask=torch.ones_like(prompt, dtype=torch.bool),
                        query=item["row"]["question"], scope=scope,
                        task=v1_runner.task_graph(case_id, item["row"]["question"]),
                        visual=visual, enabled_ports=(), core_impl="none",
                        enabled_branches=("memory",)) as request:
                    result = session.forward(request, input_ids,
                                             attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                                             labels=labels)
                    loss = float(result.loss.detach().cpu())
                    start = prompt.shape[1] - 1
                    argmax = [int(index) for index in result.logits[0, start:start + len(response_ids)].argmax(dim=-1).tolist()]
                request = session.prepare(
                    prompt_ids=prompt, prompt_mask=torch.ones_like(prompt, dtype=torch.bool),
                    query=item["row"]["question"], scope=scope,
                    task=v1_runner.task_graph(case_id, item["row"]["question"]),
                    training=False, visual=visual, enabled_ports=(),
                    core_impl="none", enabled_branches=("memory",))
                try:
                    prompt_length = int(request.frame.sense.prompt_ids.shape[1])
                    generated = session.generate(request, max_new_tokens=args.max_new_tokens)
                    generated_ids = [int(value) for value in generated[0, prompt_length:].tolist()]
                finally:
                    request.release()
                rows.append({
                    "case_id": case_id,
                    "teacher_forced_loss": loss,
                    "response_ids": response_ids,
                    "teacher_forced_argmax_ids": argmax,
                    "generated_ids": generated_ids,
                    "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                    "teacher_forced_text": tokenizer.decode(argmax, skip_special_tokens=True),
                    "expected_text": dataset.label(case_id),
                    "greedy_matches_argmax": generated_ids[:len(argmax)] == argmax[:len(generated_ids)],
                })
            report.update({
                "cases": rows,
                "mean_teacher_forced_loss": sum(row["teacher_forced_loss"] for row in rows) / len(rows),
                "argmax_matches_expected": sum(row["teacher_forced_argmax_ids"] == row["response_ids"] for row in rows),
                "greedy_matches_argmax": sum(row["greedy_matches_argmax"] for row in rows),
                "profile": profiler.stop(), "status": "completed",
            })
            (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({key: report[key] for key in (
                "mean_teacher_forced_loss", "argmax_matches_expected", "greedy_matches_argmax",
                "status")}, indent=2))
            return 0
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      profile=profiler.stop())
        (output / "failure.json").write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
