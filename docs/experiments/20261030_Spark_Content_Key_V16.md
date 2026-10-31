# Spark content-only attention keys v16

Status: **selected from the completed fitting-only reader/query diagnostic and predeclared; native attempt01 not launched.** The single mechanism is content-only attention keys with ordered values retained. Useful recall and development generalization remain unestablished.

## Fitting-only evidence and selection

V15 attempt02 completed all 16 pairs with zero generation/updates and no development labels. Its 128 first-token distributions exactly reproduce sealed v13 for the same four controls. Wrong-bank weights match partner-contextual weights at both sites.

- Site 3 attention has mean maximum weight 0.9739; site 7 has 0.9972. This is peaked, not diffuse, attention.
- Site 3 selects the same top slot in 16/16 counterfactual pairs; site 7 selects slot 0 in 32/32 rows.
- Pair-held-out answer decodability is 0/24 for site-3 weights and 5/24 for site-7 weights, versus 21/24 for each site's read and 21/24 for averaged slots. Shuffled-label means are approximately 6/24.
- Actual known-known first-token correctness is 16/24, while known-known pairs both correct remain 1/8. Known-unknown and exact-evidence behavior are separate controls.

The selected failure hypothesis is **learned-order dominance in attention keys causing positional addressing collapse**, not insufficient information in the compressed values. Content-only keys test that hypothesis while ordered values retain the established slot semantics. Source features, four slots per span, value order, sites 3/7, post-block placement, optimizer, objective, data order, initialization seed 17 and terminal gates are unchanged. The extension starts from the same initialization, not trained v12/v14 weights.

## Falsification and fixed gates

1. Frozen complete-context calibration must pass 8/8 and frozen fitting teacher 32/32 with zero false recall/truncation.
2. Exactly 512 updates run in the declared order. No best-prefix selection.
3. The mechanism is falsified if any eight-world final fitting group fails, or if known-known counterfactual pairs both correct do not exceed the sealed 1/8 reference. Only the final probe may open development, and only after every final group passes.
4. Development labels stay closed after failure. Density, formation, persistent-P updates, capacity expansion, promotion and RSI remain closed.
5. Only extension plus optimizer state is saved. A fresh process must restore exact logits and perform one continuation update with original parameter hashes unchanged; this is numerical continuation, not capability evidence.

A less-compressed representation or compression-training arm remains a separate fallback only if later evidence shows compression loses the relevant information. Exact-evidence retrieval remains a charged reference. Neither fallback is combined with content keys, placement or objective changes.

## Exact bindings and envelope

- Protocol: `configs/spark_content_key_v16.json`, SHA-256 `f1cea011ca335515a9b966e4117147711d8b3bbd9f1179a0a1160224f3bc26a4`.
- Packet: `reports/intrinsic_memory_content_key_v16/native_packet_two.json`, SHA-256 `043b745591a64c2515922231b94388a6156df77b39260998b2b3bf3a3a6bdc7b`.
- Implementation closure: 62 files; SHA-256 `3a311f683cefb8f5e610445566f7fa856a079a16ea1c87578484f01e1a1bbf4e`.
- Diagnostic: `reports/intrinsic_memory_reader_query_v15/native_bf16_reader_query_attempt02/result.json`.
- Budget: 23 stages; 7,155 reserved stage seconds inside 7,200 outer seconds; 9,778 forwards; 949 source-feature forwards; 513 updates/backwards; 7,256 generated tokens; 4,669 prompt tokenizations; 3,618 writes; 55,490,098,857 reserved read bytes; 205,537,280 artifact bytes.
- Resources: one exclusive worker, 50 GiB sampled driver, 512 MiB sustained swap growth, at least 15% host free, CPU fallback disabled.
- Attempts: one, zero retries; separate exact authorization required.

CPU verification passes 158 broad content-key, reader, checkpoint, native and contextual contracts. These tests establish key/value separation, gradients, exact default behavior, intervention parity and old/new checkpoint identity—not native learning.
