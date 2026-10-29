# Supported-value exposure control v8

**Current outcome:** Completed: frozen competence passed 8/8; contextual fitting failed 0/8 at every fixed prefix through 128 updates. Development stayed unopened. Fresh-process continuation and full frozen-weight integrity passed. See the [measured summary](native_attempt01_summary.json) and [current native report](../../docs/experiments/20261029_Spark_Memory_BF16_Native_Diagnostics.md). The two-site output is consumed with zero retries. Commands below document its historical preparation and must not be relaunched. Native 9/36 and persistent learning remain gated.

This separately bound data-only control follows the v7 fitting result. V7 learned
all eight fitting memories at 64/96/128 updates, including four same-question
counterfactual pairs, while wrong/zero/absent memory scored 0/8. Its fitting data
contains only red/blue and no unsupported queries. Its development protocol asks
for green/gold and an unsupported entity, so that test also requires transfer to
answer vocabulary and a response class absent from fitting.

V8 teaches red, blue, green, gold and unknown on fresh fitting entities. Two of its
four same-question counterfactual pairs switch between supported and unsupported
queries by changing the evidence entity. Eight new development worlds use fresh
entities and the same answer vocabulary. The structured oracle derives all answers
from the complete fact table. Generated explanations are never verification.
The corpus remains one lookup family and one seed, with one fact per world.
It is a limited novel-entity recall test, not independent-family generalization.
Both vocabulary coverage and abstention exposure change, so this does not isolate
their separate causal effects.

The v7 runner, official BF16 artifact, calibration, source/producer/reader
mechanism, initialization, NativeFusedAdamW, learning rate, fixed 128+1 updates,
32/64/96/128 probes, template, stopping policy, scoring and all gates are unchanged.
Every prefix runs; only the fixed final fitting gate may open development after
fitting freezes. Shuffled-slot superiority remains required by the utility gate.
No result or consumed packet is overwritten and no earlier development file is
read by corpus preparation. No P/teacher updates or promotion are permitted.

Each arm independently permits 2,397 forwards, 129 backwards/updates, 613,632
padded input positions, 2,008 generated tokens, two sequential model loads,
245 source encodes, 51 retrievals and 103 write transactions. The stage wall
allocation is 7,080 seconds within the 7,200-second outer supervisor. Artifact
caps are 160/416/1,056 MiB for 2/9/36 readers. Keep one exclusive native worker,
50 GiB sampled driver memory, 512 MiB sustained swap growth and at least 15%
host free. Nine/all cannot run without useful current two-site evidence.

The user has authorized continuing bounded tests without additional approval
questions. Record that instruction truthfully in each exact packet receipt.
The two-site environment check, full review and one-attempt launch are:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_exposure_v7.py --packet reports/intrinsic_memory_supported_v8/native_packet_two_v8.json --environment-only
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_exposure_v7.py --packet reports/intrinsic_memory_supported_v8/native_packet_two_v8.json --review
PYTHONDONTWRITEBYTECODE=1 .venv-spark-bf16/bin/python scripts/run_spark_intrinsic_exposure_v7.py --packet reports/intrinsic_memory_supported_v8/native_packet_two_v8.json --authorization reports/intrinsic_memory_supported_v8/authorization_two_v8.json --launch
```

The current protocol and output identities are in
[native_packet_two_v8.json](native_packet_two_v8.json) and
[spark_intrinsic_supported_values_v8.json](../../configs/spark_intrinsic_supported_values_v8.json).
Do not launch a consumed output directory. Report competence, memory-dependent
fitting, novel-entity quality, strict utility acceptance, continuation and resource
limits separately. A failure keeps density, formation and persistent learning
closed; it does not call for another retry or an altered acceptance threshold.
