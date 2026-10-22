# Native sidecar qualification design

Prepared 22 October 2026 for [S2.1](../stages/03_Native_Architecture_Study.md). This package prepares a forward/backward/serialization gate; it does not claim that the native experiment has run or that S2.1 augmented generation is complete.

After capacity jobs finish, run one native process:

```bash
.venv/bin/python scripts/qualify_native_adapter.py \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --data data/fixtures/dev.jsonl --split dev \
  --output runs/minicpm5_native_adapter_qualification_next --device mps
```

The script validates every sealed source file and guards the actual 2,516,756,480-parameter, 42-layer, 2048-wide native model. It uses FP32/eager compute and refuses unavailable accelerators and MPS CPU fallback. Memory sampling runs every 0.25 seconds; RSS and MPS driver counters overlap and must not be added. One backbone remains resident as D1 and D2 wrappers are constructed sequentially; the original is released before export reload. Large exports remain in the ignored `runs/` tree. Copy the completed compact qualification report into `reports/` for review.

The native reference logits are captured before wrapping. Both branch output projections must be exactly zero, and D1/D2 logits must match that reference with default absolute/relative tolerance zero. A controlled nonzero workspace projection must change logits, guarding against an inert test. Scope checks verify frozen originals for D1, all originals for D2 and complete disjoint original/new parameter ownership.

Workspace inputs are the encoder's fixed prompt prefix; support inputs come only from its observed `memory_texts`. The script changes future teacher-forced answer tokens and requires identical support/workspace contexts, rejects placing supervised targets in the workspace, and checks that changing writer weights rebuilds source features. These tensor checks depend on the trusted dataset's observation/source provenance; they do not establish that arbitrary source text is authorized or factually correct.

Default qualification performs one nonreentrant checkpointed backward in each regime without optimizer updates. It checks finite output-projection gradients, D1 frozen gradients/D2 full gradient coverage, native block recomputation, one-time writer/workspace construction and unchanged owned context through backward. Context and gradients are then released. `--skip-backward` explicitly marks this native subgate unrun.

The final D2 export contains a deterministic nonzero sidecar probe, making export/reload equivalence observable rather than hidden behind zero branches. This is a diagnostic artifact, not a trained candidate. CPU tiny-native tests verify parity, perturbation sensitivity, scope, causality, recomputation and nonzero-sidecar roundtrip without model downloads or accelerator work. Full native execution is a separate root-owned run.

Augmented autoregressive context reconstruction and generation remain unsupported by the existing adapter. This script does not implement them. After forward qualification, complete that S2.1 gate before interpreting native D1/D2 task generations or launching a mechanism comparison.
