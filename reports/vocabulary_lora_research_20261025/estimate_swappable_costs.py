"""Analytical capacity examples for the swappable vocabulary proposal."""
from __future__ import annotations

import json
from pathlib import Path


def calculate_swappable() -> dict:
    d, vocab, width, depth, phrase_length = 2560, 131072, 256, 2, 8
    mib = 2 ** 20
    # Illustrative encoder: down projection, two Transformer blocks (MLP 4w),
    # last-state pooling, and one up projection. No second embedding table.
    encoder_matrix_parameters = 2 * d * width + depth * 12 * width * width
    phrase_macs = (phrase_length * d * width
                   + depth * (12 * phrase_length * width * width
                              + 2 * phrase_length * phrase_length * width)
                   + width * d)
    return {
        "kind": "analytical_estimate_not_runtime_measurement",
        "spark_hidden_size": d,
        "base_vocabulary_preserved": vocab,
        "active_rows": [
            {"slots": k, "tied_bf16_rows_mib": 2 * k * d / mib,
             "two_full_row_buffers_mib": 4 * k * d / mib,
             "extra_output_projection_macs_per_position": k * d,
             "extra_output_head_arithmetic_percent": 100 * k / vocab}
            for k in (256, 512, 1024)
        ],
        "illustrative_shared_phrase_encoder": {
            "width": width, "layers": depth, "mlp_ratio": 4,
            "matrix_parameters": encoder_matrix_parameters,
            "bf16_matrix_weights_mib": 2 * encoder_matrix_parameters / mib,
            "fp32_weights_grad_two_adam_moments_mib": 16 * encoder_matrix_parameters / mib,
            "example_phrase_base_tokens": phrase_length,
            "matrix_macs_per_example_phrase": phrase_macs,
            "matrix_macs_for_16_fresh_phrases": 16 * phrase_macs,
            "matrix_macs_for_256_fresh_phrases": 256 * phrase_macs,
        },
        "compact_bank_alternative": {
            "entries": 4096, "value_width": 256, "key_width": 128,
            "bf16_values_mib": 4096 * 256 * 2 / mib,
            "bf16_keys_mib": 4096 * 128 * 2 / mib,
            "reuse_existing_engram_storage_where_possible": True,
        },
        "exclusions": [
            "Norms, biases, positional/type parameters, selector and other small heads",
            "Base model weights, attention/KV, full logits, saved activations and solve/allocator temporaries",
            "Raw spelling, stable identity, provenance, historical revisions and other metadata",
            "Replay, cache invalidation, encoding/selection wall time and persistence I/O",
            "Quality of the proposed small encoder is unestablished",
        ],
    }


if __name__ == "__main__":
    target = Path(__file__).with_name("swappable_costs.json")
    target.write_text(json.dumps(calculate_swappable(), indent=2) + "\n")
    print(target)
