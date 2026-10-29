"""Explicit provisioned-source integration probe: random tiny CPU weights only."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import resource
import tempfile
import time
import os

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("probe output must be fresh")
    started = time.monotonic()
    source = ROOT / "checkpoints/spark_fp8_derivative_v1"
    manifest = json.loads((ROOT / "configs/spark_derivative_execution_manifest_v1.json").read_text())
    for name, expected in {**manifest["files"], "config.json": manifest["config_sha256"]}.items():
        path = source / name
        assert not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert importlib.metadata.version("transformers") == "4.57.1"
    with tempfile.TemporaryDirectory(prefix="spark_cpu_source_import_") as cache:
        os.environ.update(HF_MODULES_CACHE=cache, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        import torch
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        torch.set_num_threads(1)
        torch.manual_seed(17)
        config = AutoConfig.from_pretrained(str(source), trust_remote_code=True, local_files_only=True)
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_hidden_layers = 8
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 16
        config.vocab_size = 64
        config.max_position_embeddings = 128
        config.sliding_window = 16
        config.layer_types = ["sliding_attention"] * 3 + ["full_attention"]
        config.layer_types *= 2
        config.use_cache = False
        config._attn_implementation = "eager"
        cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"],
                                             str(source), local_files_only=True)
        model = cls(config).to(device="cpu", dtype=torch.float32).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        ids = torch.tensor([[1, 2, 3, 4]])
        embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
        result = model(inputs_embeds=embeds, attention_mask=torch.ones_like(ids), use_cache=False)
        result.logits.square().mean().backward()
        tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
        passed = (tied and bool(torch.isfinite(result.logits).all())
                  and embeds.grad is not None and bool(torch.isfinite(embeds.grad).all())
                  and bool(embeds.grad.abs().sum() > 0))
        report = {"format": "replica.spark_source_environment.cpu_probe.v1", "passed": passed,
                  "scope": "sealed_source_code_with_scaled_random_CPU_configuration_no_checkpoint_weight_loading",
                  "source_code_sha256": manifest["files"], "torch": str(torch.__version__),
                  "transformers": importlib.metadata.version("transformers"),
                  "layers": 8, "hidden_size": 64, "vocab_size": 64,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "logits_shape": list(result.logits.shape), "finite_logits": bool(torch.isfinite(result.logits).all()),
                  "tied_output": tied, "finite_nonzero_input_gradient": passed,
                  "native_execution": False, "native_checkpoint_weights_loaded": False,
                  "tokenizer_loaded": False, "optimizer_steps": 0,
                  "elapsed_seconds": time.monotonic() - started,
                  "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps(report, indent=2))
        if not passed:
            raise RuntimeError("tiny source integration failed")


if __name__ == "__main__":
    main()
