"""Analytical payloads only. Standard library; no model import or execution."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent


def calculate():
    header = json.loads((ROOT / "reports/spark_preparation_20261025/run_01/headers/official_bf16_control/tensor_index.json").read_text())
    tensors = header["tensors"]
    p = sum(math.prod(t["shape"]) for t in tensors)
    # Explicit candidate assignment, not an executed Spark optimizer inventory.
    aux = sum(math.prod(t["shape"]) for t in tensors
              if len(t["shape"]) != 2 or t["name"] == "model.embedding.weight")
    gib, mib = 2**30, 2**20
    def whisper(width, layers):
        trainable = 3*80*width + width + 3*width*width + width
        trainable += layers*(12*width*width + 12*width) + 2*width
        positional = 1500*width
        return {"trainable_formula_parameters": trainable,
                "fixed_position_elements": positional,
                "stored_elements_including_fixed_positions": trainable+positional,
                "bf16_MiB": (trainable+positional)*2/mib,
                "fp32_MiB": (trainable+positional)*4/mib}
    return {
        "format": "sparkx.product_costs.v1",
        "kind": "analytical_lower_bounds_not_native_runtime_measurements",
        "spark": {
            "unique_parameters": p, "serialized_tensors": len(tensors),
            "bf16_weights_GiB": p*2/gib,
            "fp32_weights_GiB": p*4/gib,
            "training_state_GiB": {
                "fp32_weights_grads_two_fp32_moments_16_bytes": p*16/gib,
                "bf16_weights_grads_fp32_moments_no_master_12_bytes": p*12/gib,
                "bf16_weights_grads_fp32_moments_fp32_master_16_bytes": p*16/gib,
                "hypothetical_bf16_weights_grads_two_int8_moments_no_master_6_bytes": p*6/gib,
                "candidate_fp32_muon_one_int8_momentum_aux_fp32_adam": (9*p+7*aux)/gib,
            },
            "muon_candidate_assignment": {"matrix_parameters": p-aux, "auxiliary_parameters": aux,
                "qualification": "assignment assumption; excludes block scales, orthogonalization scratch, activations and backend overhead"},
            "bf16_KV_examples": [
                {"positions": n, "hybrid_upper_bound_MiB": 2*2*4*256*(9*n+27*min(n,512))/mib,
                 "all_layers_unbounded_MiB": 2*2*4*256*36*n/mib}
                for n in (8192,32768,1048576)
            ],
        },
        "audio": {
            "whisper_base_encoder_formula": whisper(512,6),
            "whisper_small_encoder_formula": whisper(768,12),
            "spear_base_v2_metadata_parameters": 93289476,
            "spear_base_v2_bf16_MiB": 93289476*2/mib,
            "spear_base_v2_fp32_MiB": 93289476*4/mib,
            "bridge_512_to_1024_to_2560_with_bias": 512*1024+1024+1024*2560+2560,
            "bridge_768_to_1024_to_2560_with_bias": 768*1024+1024+1024*2560+2560,
            "tokens_per_hour_at_10_Hz": 3600*10,
            "seconds_per_window": 16,
            "tokens_per_window_at_10_Hz": 160,
        },
        "state": {
            "spark_bank_4096_key128_value256_bf16_MiB": 4096*(128+256)*2/mib,
            "minicpm_bank_1024_key128_value256_bf16_MiB": 1024*(128+256)*2/mib,
            "spark_256_vocab_shared_rows_bf16_MiB": 256*2560*2/mib,
            "workspace_8_by_256_bf16_KiB": 8*256*2/1024,
            "event_64_by_256_bf16_KiB": 64*256*2/1024,
            "two_128_by_256_fp32_fast_matrices_MiB": 2*128*256*4/mib,
        },
        "proposed_16_second_stream_window": {
            "video_frames": 4, "visual_tokens_per_frame": 64,
            "max_additional_crops": 2, "tokens_per_crop": 64,
            "audio_tokens_at_10_Hz": 160, "event_tokens_cap": 64,
            "clock_and_modality_token_reserve": 32,
            "total_token_cap": 4*64+2*64+160+64+32,
            "scope": "post-qualification compression candidate, not perception-control setting",
        },
        "exclusions": ["activations and full logits", "temporary attention and optimizer buffers",
                       "allocator and compiler memory", "optimizer quantization scales/master copies unless listed",
                       "raw media, replay, provenance and representation migration", "per-session replication",
                       "any prediction of MPS latency or convergence"],
    }


if __name__ == "__main__":
    (OUT / "analytical_costs.json").write_text(json.dumps(calculate(), indent=2)+"\n")
    print("Analytical cost inventory generated; no models executed.")
