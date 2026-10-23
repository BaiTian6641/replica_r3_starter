# E3 full-backbone D2 action diagnostic: stopped at residency limit

23 October 2026 UTC. Protocol `configs/e3_d2_action_v1.json` committed before
execution. Fresh 8-train/2-dev/4-held-out action family, initialized from the
accepted E3 extension. Full augmented scope: all 381 original tensors plus new
modules; Muon 8-bit/FP32 AdamW; 8 updates × 4 examples; seed 20261023.

**Result: no optimizer update and no held-out evaluation.** The first backward
peaked at **43,682,725,888 bytes (40.68 GiB)** sampled MPS driver memory, above
the declared **40 GiB** limit. The capacity guard stopped the child; the failure
was a host guard stop, not a learning result. No checkpoint or base copy was
written.

Actual teacher action prompts are up to **609 tokens** plus 6–33 target tokens.
A 512-token rerun would truncate the declared task context, so it was not run.
The existing D2 capacity path fits at 512 tokens because it has a smaller
new-module/action workload; this action arm does not.

Residency options remain explicit separate work: activation offload, bf16
compute under a new numerically declared protocol, or a shorter task interface.
None is claimed. The action-selection negative results from modules-only families
remain unchanged.
Evidence: [failure report](../../reports/e3_d2_action_attempt01/failure_report.json),
[guard](../../reports/e3_d2_action_attempt01/guard.json).

## bf16-compute residency follow-up

A separately declared bf16-compute variant kept FP32 parameters/optimizer state
and used the same fresh family. It completed all **8 updates** and stayed under
the guard: sampled peak **39.10 GiB**. The 640-token evaluation budget was too
small for the runtime prompt, so a separate 768-token evaluation protocol was
declared; no task text was removed.

**Result: 0/4 held-out and 0/2 dev completions.** Every episode selected `read`
seven times; no false success was accepted. Training loss was unstable
(0.016 → 1.16 → … → 0.63), and the model still did not select `set_constant`.
No export was written. This is a full-backbone action-selection negative, not a
residency or parser result.
Evidence: [report](../../reports/e3_d2_action_bf16_eval768_attempt01/run_report.json).
