# Spark memory S0–S1: first native attempt and environment repair

**29 October 2026 · historical attempt01/02 environment and placement failures.**

Current outcome: the separately authorized [attempt03](20261029_Spark_Memory_Native_Readout03.md)
completed with S0 8/8 and failed S1 useful recall. All three single-attempt receipts
are consumed. The earlier failures below do not measure Spark capability.

The corrected environment packet below was subsequently explicitly authorized.
[Attempt02](../../reports/intrinsic_memory_v3/native_readout02/result.json) passed
runtime/source admission and loaded the existing derivative, but its first
calibration forward failed because the backbone stayed CPU while inputs/extensions
were MPS. No answer or update completed. It used 9.712113 s, a 1,082,540,032-byte
sampled driver peak, 700,645,376-byte RSS peak and zero swap growth. The guard did
not trip. Both zero-retry receipts are consumed. See the
[current integration report](20261029_Spark_Memory_Integrated_Access.md) for the
separately versioned placement repair. Its later authorization and completed result
are recorded in the attempt03 report above.

The account below preserves the original environment diagnosis and packet history.

The user explicitly authorized the exact original S0–S1 packet. Its receipt and
full source/implementation admission passed, the exclusive worker lock was free,
and the guarded attempt ran at 14:01:51–14:02:00 America/Chicago. It stopped during
model construction: Transformers 5.20.2 expected `_tied_weights_keys` as a mapping,
while the sealed Spark code supplies a list. The packet selected the general
`.venv`; the preserved derivative qualification used `.venv-spark-bf16` with
Transformers 4.57.1. The original packet failed to bind and verify that dependency
environment. This is an execution defect, not a failed competence result.

The [portable result](../../reports/intrinsic_memory_v3/native_readout01/result.json)
and copied raw logs preserve the original attempt. The model constructor was
attempted once; no model load completed, no answer was generated and no update ran.
The tokenizer had loaded and validated. Calibration, fit, development and restart
were not reached; development and sealed-final labels remained closed. Source
qualification passed, while current numerical qualification and competence remain
unmeasured. The historical qualification/conversion manifests were not changed.

The guard recorded **9.441364 s**, **393,216 bytes** peak sampled MPS driver
allocation, **568,393,728 bytes** process peak RSS, **zero swap growth** and **91%**
minimum host free. Model-output artifacts totalled **10,721 bytes**. The guard
did not trip; the runner returned 2 and the guard returned 1. These are failed-load
costs and say nothing about the memory needed for a successful model run.

## Implemented environment admission

Commit `73a5f9ef` adds a separately versioned
[launcher](../../scripts/launch_spark_intrinsic_readout_environment_v1.py) and
[environment manifest](../../configs/spark_intrinsic_execution_environment_v1.json).
They require the existing qualified interpreter directory and Python version,
all 32 package versions in the preserved BF16 lock, expected module origins and
hashes of 12 selected runtime files. This is explicitly not a full-wheel integrity
claim. Checks use only the standard library and run before source preflight or
model imports. They reject the wrong interpreter instead of changing packages or
model code. The launcher verifies additional human-receipt bindings for its bytes,
the environment, the reviewed packet, output and guard report. It then invokes the
unchanged source preflight and capacity guard with the verified interpreter;
the original runner's restart uses that same `sys.executable`.

The existing `.venv-spark-bf16` passes those checks. The general `.venv` is refused
before any model import. **21 portable CPU tests pass**, including wrong versions,
missing packages/files, changed executable bytes, shadowed module resolution,
missing or stale authorization, reused output, and refusal before subprocess
launch. No environment was installed or modified.

An explicit provisioned-source CPU probe verifies the exact model/config code
hashes, scales the configuration to eight layers, width 64 and vocabulary 64,
and constructs **302,144 random parameters**. It passes tied-output identity,
finite forward values and finite nonzero input gradients in Transformers 4.57.1.
It took **3.660994 s**, peaking at **338,886,656 bytes** process RSS. No checkpoint
weight or tokenizer was loaded, no optimizer update ran and no MPS execution
occurred. This diagnoses source/runtime compatibility; it does not qualify the
4B model or its generated answers.

## Fresh review packet

The [environment-bound S0–S1 packet](../../reports/intrinsic_memory_v3/native_experiment_s0_s1_environment_v2.json)
keeps the original source, precision identity, code hashes, eight-case competence
gate, 32 readout updates plus one restart update, exact/wrong/shuffled/absent controls
and all work/resource limits. It uses fresh `native_readout02` output and
`guard_derivative_native02.json`. Its command names the qualified interpreter and
new launcher. This remains a two-site v3 readout experiment; the v4 2/9/36 density
candidate is separate.

The original authorization allowed zero retries. A fresh explicit receipt was
subsequently supplied for this packet and consumed by attempt02, described above.
Its additional environment/launcher bindings were not inferred from the earlier
approval. Neither consumed receipt permits a further attempt. No useful readout, persistent P, retention, reasoning
cost reduction, capacity expansion or promotion follows from these diagnostics.
