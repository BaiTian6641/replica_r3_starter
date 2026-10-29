# Fixed direct-CE exposure curve v7

The [v6 native result](../intrinsic_memory_explicit_v6/native_attempt01_summary.json)
passes official BF16 competence at 8/8 and numerical continuation, but fails fitting:
contextual/wrong/shuffled/zero memory emits immediate EOS in every fitting world;
selected exact text answers 8/8. Live/durable values and logits are exact, all
fitting/generation prefixes match, gradients are finite, and original parameter
hashes remain unchanged. Development labels are still unopened.

V7 tests **exposure only**. The source, producer/readers, initialization, optimizer,
learning rate, complete-response objective, prompt, scorer and v6 data are unchanged.
It resets the extension to the same initialization and runs exactly 128 updates,
with all fitting controls at fixed prefixes 32/64/96/128. There is no early stop,
best-prefix choice, criterion change or new development access at an early prefix.
Only the final prefix can open development, after freezing fitting. This is a
fitting convergence diagnostic on one exposed synthetic rule family, not a new
independent capability test. The repeated calibration verifies the unchanged source
and input contract; it is not fresh generalization evidence.

V5/v6 execution files and results remain immutable. The separately versioned v7
launcher and native schedule reuse their validated memory, source and continuation
helpers. Protocol, dependency closure, checkpoint identity and output destinations
bind the new schedule. Each attempted operation is reserved before execution;
failed work and every probe count toward the same durable ledger and outer deadline.

Per-arm maxima are 2,397 forwards, 129 backwards/updates including one restart,
613,632 input positions, 2,008 generated tokens, 245 source encodes, 51 retrievals,
103 write transactions and two sequential model loads. The stage wall allocation
is 7,080 seconds inside the 7,200-second outer supervisor. Artifact limits remain
160/416/1,056 MiB for 2/9/36 readers. The exclusive-worker, 50 GiB driver,
512 MiB sustained swap-growth and 15% minimum host-free limits remain unchanged.
There are zero teacher/P updates and no automatic promotion. Nine/all require a
qualified useful two-site result from this exact protocol; they cannot bypass a
failed fit or utility gate.

The user has authorized continuing the planned tests without more approval
questions. Each packet still receives its own truthful authorization receipt.
The two-site command is:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_exposure_v7.py --packet reports/intrinsic_memory_exposure_v7/native_packet_two_v7.json --authorization reports/intrinsic_memory_exposure_v7/authorization_two_v7.json --launch
```

If the fixed curve still fails memory-dependent fitting, preserve the result and
prepare the separately costed teacher-versus-student alignment experiment; do not
expand reader density or P capacity to hide the failure. If fitting passes but
development fails, distinguish exposure/generalization from fitting success.
