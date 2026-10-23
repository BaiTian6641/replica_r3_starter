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
