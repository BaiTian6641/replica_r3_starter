# Frozen-teacher intrinsic-memory comparison v10

The [experiment specification](../../docs/experiments/20261029_Spark_Memory_Teacher_Alignment_V10.md)
defines the new teacher objective and complete budget. The completed v9 run is
the matched direct-CE reference. V10 preserves its source, data, initialization,
learning rate, student exposure and competence/fitting/utility criteria.

Current preparation at `ad58c14b`: **502/502 clean-checkout tests pass**, zero
skips, in 226.228 s. This includes 58 new teacher objective,
reader and admission checks. Environment-only validation and full source/code/
data/predecessor preflight pass without loading weights. The clean 2/9/36 CPU
workflow passes in 14.439167 s, with 615,071,744 bytes
parent peak RSS and 105,446,031 bytes across its arm directories.
Every active reader and the producer receive gradients; each fresh child restores
exact logits and one optimizer owner, then performs one actual update using the
restored teacher cache. Teacher qualification in this random CPU fixture is
explicitly synthetic; it proves neither pretrained teacher competence nor useful
memory. See the [prelaunch verification receipt](prelaunch_verification.json).

Native v10 is not yet run. The [exact receipt](authorization_two_v10.json) records the user's standing
authorization for this bounded comparison.
No consumed earlier packet or output is reused. The two-site command is:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_alignment_v10.py --packet reports/intrinsic_memory_alignment_v10/native_packet_two_v10.json --authorization reports/intrinsic_memory_alignment_v10/authorization_two_v10.json --launch
```

`--environment-only` checks the qualified interpreter and locked dependencies.
`--review` checks full source, executable closure, prepared-data and predecessor
bindings. Neither mode loads weights or launches a worker. `--launch` has zero
retries and consumes its distinct output directory even if admission later fails.

Per arm: 2,469 forwards, 2,072 generated tokens, 632,064 padded input positions,
129 student updates/backwards, two sequential model loads, 245 source encodes,
51 retrievals and 112 declared writes. Stage time totals 7,080 s inside the
7,200-second outer supervisor. Artifact caps remain 160/416/1,056 MiB for 2/9/36.
Keep one native worker, 50 GiB sampled driver memory, 512 MiB sustained swap
growth and at least 15% host free. Teacher/original-backbone and P updates are zero.

Calibration must pass 8/8 before fitting labels open. Frozen full-context teacher
answers on the eight fitting worlds must also pass 8/8 with termination, zero
false recall and no truncation. Only then are detached targets cached and the
128 student updates run. Only the fixed final fitting prefix can open development.
Teacher caches are evicted from the engine before evaluation. A final 8/8 fitting
score is not sufficient to qualify useful recall: the matched control criteria
still apply. Native 9/36 require useful two-site evidence from this exact protocol.

Formation, source-free P learning, capacity expansion, nested adaptation, Replica
fidelity and RSI remain unqualified. Keep the separate external spine and approved
identity/state ownership. Preserve every failed trajectory and raw resource record.
