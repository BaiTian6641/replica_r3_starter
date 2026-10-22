#!/usr/bin/env python3
"""Exercise the selected backend/optimizer with tiny random Llama, without downloads.

This checks D0/D1/D2 updates and model export/reload, not pretrained model fit,
throughput, task quality, or training-run resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    p.add_argument("--optimizer", choices=("muon8bit", "adamw"), default="muon8bit")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new output report path")
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        p.error("Unset PYTORCH_ENABLE_MPS_FALLBACK to test actual MPS operation support")

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from replica_lab.model_adapter import LlamaMemoryAdapter
    from replica_lab.optimizers import build_optimizer

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing backend substitution")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing backend substitution")

    def synchronize():
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()

    def batch():
        return {k: v.to(args.device) for k, v in {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "labels": torch.tensor([[-100, -100, -100, 4, 5]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "prompt_ids": torch.tensor([[1, 2, 3]]),
            "memory_ids": torch.tensor([[7, 8]]),
            "prompt_mask": torch.ones(1, 3, dtype=torch.bool),
            "memory_mask": torch.ones(1, 2, dtype=torch.bool),
        }.items()}

    arms = []
    for regime in ("full", "modules_only", "full_augmented"):
        torch.manual_seed(17)
        cfg = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=64,
                          attention_dropout=0.0, tie_word_embeddings=False)
        cfg._attn_implementation = "eager"
        model = LlamaMemoryAdapter(LlamaForCausalLM(cfg), regime,
                                  memory_size=16, workspace_size=16,
                                  slots=2, workspace_steps=1).to(args.device)
        model.base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        before = {n: x.detach().clone() for n, x in model.named_parameters()}
        optimizer = build_optimizer(model, args.optimizer)
        synchronize()
        started = time.monotonic()
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            try:
                loss = model(**batch()).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite smoke loss")
                loss.backward()
            finally:
                model.release_context()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        synchronize()
        elapsed = time.monotonic() - started
        changed = [n for n, x in model.named_parameters() if not torch.equal(x, before[n])]
        original_ids = {id(x) for x in model._original_parameters.values()}
        changed_original = [n for n, x in model.named_parameters()
                            if id(x) in original_ids and n in changed]
        embedding_updated = any(x is model.base.get_input_embeddings().weight and n in changed
                                for n, x in model.named_parameters())
        head_updated = any(x is model.base.get_output_embeddings().weight and n in changed
                           for n, x in model.named_parameters())
        if regime == "modules_only":
            assert not changed_original, "Frozen original weights changed"
        else:
            assert embedding_updated and head_updated, "Full scope must update embeddings and head"
        assert changed, "No parameters changed"
        model.eval()
        with torch.no_grad():
            expected = model(**batch()).logits.detach().cpu()
            model.release_context()
        with tempfile.TemporaryDirectory(prefix="replica-smoke-") as directory:
            model.save_pretrained(directory)
            base = LlamaForCausalLM.from_pretrained(Path(directory) / "base", local_files_only=True)
            loaded = LlamaMemoryAdapter(base, regime, memory_size=16, workspace_size=16,
                                        slots=2, workspace_steps=1)
            if loaded.augmented:
                loaded.load_sidecars(Path(directory) / "sidecars.safetensors")
            loaded = loaded.to(args.device).eval()
            with torch.no_grad():
                actual = loaded(**batch()).logits.detach().cpu()
                loaded.release_context()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        arms.append({"regime": regime, "steps": 3, "losses": losses,
                     "parameters": model.parameter_audit(), "optimizer": optimizer.audit(),
                     "changed_tensor_names": changed,
                     "changed_original_tensors": len(changed_original),
                     "embedding_updated": embedding_updated, "output_head_updated": head_updated,
                     "export_reload_close": True, "checkpointing": True,
                     "elapsed_update_seconds": elapsed})
    synchronize()
    report = {"status": "passed", "validated_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Tiny randomly initialized Llama backend/optimizer smoke only",
              "device": args.device, "parameter_dtype": "float32", "compute_dtype": "float32",
              "optimizer": args.optimizer, "arms": arms,
              "packages": {name: importlib.metadata.version(name)
                           for name in ("torch", "transformers", "safetensors")},
              "pretrained_weights_loaded": False, "full_model_fit_verified": False,
              "model_quality_evaluated": False, "cuda_experiment_run": args.device == "cuda"}
    if args.device == "mps":
        report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        report["mps_counters_are_peak_measurements"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "device": args.device,
                      "arms": len(arms), "report": str(args.output)}))


if __name__ == "__main__":
    main()
