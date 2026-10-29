"""Random tiny CPU source-code padding diagnostic; never loads native weights."""
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_suffix(".json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise ValueError("diagnostic output must be fresh")
    start = time.monotonic()
    manifest_path = ROOT / "configs/spark_derivative_execution_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    source = ROOT / "checkpoints/spark_fp8_derivative_v1"
    for name, expected in {**manifest["files"], "config.json": manifest["config_sha256"]}.items():
        assert not (source / name).is_symlink() and sha(source / name) == expected
    assert importlib.metadata.version("transformers") == "4.57.1"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    import torch
    torch.set_num_threads(1)
    package = types.ModuleType("spark_padding_source_probe")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    modules = {}
    for name in ("configuration_spark", "modeling_spark"):
        spec = importlib.util.spec_from_file_location(package.__name__ + "." + name, source / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    source_config = json.loads((source / "config.json").read_text())
    config_values = {**source_config, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 16, "vocab_size": 64, "max_position_embeddings": 4096,
        "layer_types": source_config["layer_types"][:8], "use_cache": False}
    config_values.pop("quantization_config", None)
    real = torch.tensor([[3, 5, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37]])
    cases = []
    with torch.no_grad():
        for seed in (17, 31):
            for dtype in (torch.float32, torch.bfloat16):
                torch.manual_seed(seed)
                config = modules["configuration_spark"].Spark2_5Config(**config_values)
                config._attn_implementation = "eager"
                model = modules["modeling_spark"].Spark2_5ForCausalLM(config).to(device="cpu", dtype=dtype).eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                base = model(input_ids=real, attention_mask=torch.ones_like(real), use_cache=False).logits.float()
                for length in (32, 2048):
                    for side in ("right", "left"):
                        ids = torch.full((1, length), config.pad_token_id, dtype=torch.long)
                        mask = torch.zeros_like(ids)
                        region = slice(0, real.shape[1]) if side == "right" else slice(length-real.shape[1], length)
                        ids[:, region], mask[:, region] = real, 1
                        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, region].float()
                        difference = (logits-base).abs()
                        cases.append({"seed": seed, "dtype": str(dtype), "padding": side,
                            "length": length, "real_tokens": real.shape[1],
                            "finite": bool(torch.isfinite(logits).all()),
                            "max_absolute_logit_difference_from_unpadded": float(difference.max()),
                            "mean_absolute_logit_difference_from_unpadded": float(difference.mean()),
                            "real_token_argmax_differences": int((logits.argmax(-1)!=base.argmax(-1)).sum()),
                            "last_token_argmax_equal": bool(torch.equal(logits[:,-1].argmax(-1),base[:,-1].argmax(-1)))})
                del model
    report = {"format": "replica.spark_source_padding.cpu_probe.v1",
        "scope": "manifest_bound_source_code_scaled_random_CPU_model_no_native_weights_no_memory_extensions_no_development_data",
        "source_execution_manifest_sha256": sha(manifest_path), "source_code_sha256": manifest["files"],
        "source_config_sha256": manifest["config_sha256"], "probe_script_sha256": sha(Path(__file__)),
        "python": sys.version, "torch": str(torch.__version__), "transformers": importlib.metadata.version("transformers"),
        "command": [str(ROOT/".venv-spark-bf16/bin/python"), str(Path(__file__).resolve())],
        "command_environment": {"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "configuration_overrides": {key: config_values[key] for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings", "layer_types", "use_cache")},
        "preserved_source_properties": {key: source_config[key] for key in ("rope_parameters", "sliding_window", "hidden_act", "headwise_attn_output_gate", "gate_attn_act_mode", "tie_word_embeddings", "pad_token_id")},
        "source_attention_classes": ["eager_softmax_sliding_attention", "eager_softmax_full_attention"],
        "linear_attention_branch_present": False, "optimizer_steps": 0,
        "native_execution": False, "checkpoint_weights_loaded": False, "tokenizer_loaded": False,
        "cases": cases, "all_finite": all(case["finite"] for case in cases),
        "claim_limit": "random_scaled_CPU_diagnostic_cannot_establish_native_padding_equivalence_or_explain_native_failure; no_trained_memory_reader_tested",
        "elapsed_seconds": time.monotonic()-start,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with OUTPUT.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("all_finite", "elapsed_seconds", "process_peak_rss_bytes", "cases")}, indent=2))


if __name__ == "__main__":
    main()
