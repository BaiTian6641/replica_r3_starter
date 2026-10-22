# Fixed-context generation and native D1 diagnostic protocol

Declared 22 October 2026 before native generation qualification or D1 updates.
This is the next bounded package in [S2.1–S2.2](../stages/03_Native_Architecture_Study.md).
Results will be recorded separately; this protocol does not assert success.

## Generation correctness gate

Use one resident sealed MiniCPM5-2B FP32 backbone on MPS. Capture native greedy
uncached reference answers before installing the wrapper. Check both D1 and D2
null branches against the same reference, including logits and generated token
IDs on one actual 256-token fixture, with an eight-token answer cap. Build source
slots and workspace once from the original prompt and eligible source text, hold
them fixed for the complete answer, and re-prefill the full growing prefix at
every decode step. Generated tokens never become workspace-builder inputs.

Check context cleanup, unchanged prompt/source inputs, nonzero deterministic
branch sensitivity and sidecar serialization/reload answer parity. CPU tests
must exercise multi-token context reuse even if a native answer ends early.
No cached augmented generation, concurrent requests, architecture benefit or
answer-quality improvement is inferred from this compatibility gate.

## D1 optimizer and tiny-overfit gate

Start a fresh sealed source, seed 17, regime `modules_only`, measured 64-wide
memory/workspace, four slots, two workspace steps and final-layer insertion.
Freeze all 381 original tensors / 2,516,756,480 parameters. Train all 48 added
tensors / 895,490 parameters. The original evidence text remains duplicated in
the native prompt, matching the existing diagnostic information budget.

Use the existing eight balanced mask-1 training rows (answers 0–7 once per
effective update) and original 16 development fixtures. Actual full sequence
length is 55 tokens; 256 is a ceiling. Use FP32, microbatch one, accumulation
eight, nonreentrant activation checkpointing and experimental Muon 8-bit with
its FP32 AdamW exclusions. Both new-module peak rates are **1e-3**, two warmup
updates, constant schedule and zero weight decay. This is an explicit rate for
fresh sidecars, not a matched-rate architecture comparison with D0's 1e-4.

The total schedule is **64 updates**, initially stop the first process at update
8, save completed-update state every eight updates, and continue in fresh
processes as required. Generate all eight training answers every four updates,
greedy, eight-token cap, original-prompt-fixed sidecars. Stop after two consecutive
8/8 checks with eight distinct responses. If no success at 64, preserve a D1
negative; do not unfreeze originals or silently change rates in that run.

Audit gradients on each process's first two updates: two nonzero initial output
projections are expected, followed by upstream signal. Record actual optimizer
ownership/bytes, response/EOS loss, raw answers, memory traces, export/reload
logits and completed-update continuation. Independently compare all saved base
tensors with BF16-source values promoted to FP32; D1 must change **zero** originals.
Evaluate saved sidecars in a fresh process on all training and development rows.
If outputs terminate normally, use the historical 256-token external evaluation
cap; otherwise keep a declared short cap and do not equate unequal-cap scores.

The memory watchdog uses the established 40 GiB driver ceiling, 512 MiB sustained
swap growth ceiling and 15% minimum system free-memory reading, with the same
sampling and stop behavior as the capacity protocol. Keep one native MPS job
resident. Driver allocation and RSS overlap and are reported separately.

An achieved tiny fit establishes memorization and training infrastructure only.
No broad transfer, retention, semi-cognition, Replica fidelity, frozen-base
self-improvement or useful memory advantage is established by eight cases.
D2 optimizer capacity and fair architecture controls remain separate later gates.

## Conditional follow-up rate screen

Declared while the first arm was at update 42 (training generation remained 0/8
through update 40), before its 64-update outcome. If the 1e-3 arm does not attain
two perfect generation checks within its budget, preserve that negative and run
**one** fresh-source arm at new-module Muon and AdamW rates **1e-2**. Keep seed 17,
data, scope, modules, insertion, warmup, schedule, decay, generation and guard
settings identical. Its own cap is 64 updates, first process stops at 8 for a
fresh-process restart, with checkpoints every eight updates and the same stopping
rule. Never resume the 1e-3 optimizer at the higher rate.

This tests whether the first declared rate was too weak for newly initialized
residual sidecars. The hypothesis is unproven; all search costs and outcomes count.
No third rate or scope change belongs to this package. After two failed arms,
record the diagnostic negative and plan an activation-scale/placement investigation
instead of extending the budget silently. Rate selection here uses training-only
memorization, not the later independent architecture or promotion benchmark.
