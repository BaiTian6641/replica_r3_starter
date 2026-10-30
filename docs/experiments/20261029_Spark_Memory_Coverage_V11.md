# Broader source coverage for intrinsic memory v11

Prepared before native v11 answers on 29 October 2026. The source is the existing
**official BF16** XHToken/Spark-X2.6-4B artifact, revision
`66e3ae7bee54e0a523d692cb8ef841f35409807b`, in the qualified environment. It is
separate from official FP8 and the historical FP8-derived BF16 derivative.

V10 learned its eight fitting memories but failed useful recall: fresh contextual
5/8, exact text 8/8 and shuffled values 5/8. Its development set is now exposed.
This successor tests broader balanced teaching with a fresh split. It does not
change the readout mechanism, choose an earlier prefix, or lower a criterion.
[The v10 report](20261029_Spark_Memory_Teacher_Alignment_V10.md) remains historical.

## Fixed comparison

The [v11 protocol](../../configs/spark_intrinsic_coverage_v11.json) binds eight
new calibration worlds, 32 fitting worlds and 16 fresh development worlds. The
scripted source oracle supports two explicit rules: recorded entity-color lookup
and the latest of two ordered revisions. Decisions, entities and counterfactual
pairs stay in one split. No generated explanation is a verified teaching label.
Each rule has sixteen fitting cases: three each of red, blue, green and gold, and
four unknowns. Every span is at most 256 UTF-8 bytes. A public split manifest
exposes identities without labels; admission hashes prepared data. Calibration,
fitting and development parsing occur behind their declared gates.

The producer, four ordered width-256 slots per span, eight-slot masked budget,
reader, tokenizer, template, scorer, generation limits, seed 17, learning rate
0.0001 and teacher objective are unchanged from v10. Original Spark weights stay
frozen. One NativeFusedAdamW owns each intended producer and reader parameter.
Teacher targets detach before caching; student values remain live tensors through
response loss. The block input supplies each query and the reader residual is
added after the block. Pre-MLP fusion is not part of this experiment.

1. Require frozen complete-context calibration 8/8, false recall zero, truncation
   zero. Failure closes dependent work.
2. Require the complete fitting teacher to answer 32/32 correctly, terminate and
   avoid false recall/truncation. Create 32 detached top-64 teacher distributions
   once in a separately bound cache, at most 2 MiB. Original teacher updates: zero.
3. Train exactly 512 student updates: sixteen passes through all 32 worlds in
   fixed order. Record fitting probes at 128/256/384/512; only the final prefix
   can open development. No early stopping or best-prefix selection.
4. Require all four predeclared groups of eight to pass the unchanged fitting
   criterion: contextual 8/8, first token/termination/logit parity, all four pairs
   correct, and better than wrong, zero and absent memory. Shuffling remains a
   fitting diagnostic and a strict development control.
5. Freeze fitting and evict teacher targets before opening fresh development.
   Both fixed groups of eight must independently pass: contextual at least 6/8,
   zero false recall, at most one answer behind exact text, and strictly better
   than wrong, shuffled, zero and absent memory. A pooled score cannot bypass a
   failed group. Record correction, revoked-read and generated-answer behavior.
6. Save extension plus optimizer without original weights. A second process
   restores exact logits and the teacher cache, and performs one actual update.
   Compare original parameter hashes in both processes. This is numerical
   continuation; uninterrupted-update equivalence is not claimed.

The new format binds this entire execution, prepared data, source code and local
executable closure. Teacher caches cannot be retried after a failed qualification,
restored after fitting freezes, or used to revive a corrected/revoked source.
Changed, missing or substituted Spark code is refused before model loading.
Historical execution bytes and receipts are untouched.

## Work and native density limits

Each 2/9/36 arm has a distinct output and exact packet. The two-site control uses
layers 3/7. Native 9/36 remain closed until the matching v11 control establishes
useful recall, competence, integrity, restart and resource compliance. Standing
human testing authorization is recorded for the concrete two-site packet; it
never bypasses scientific gates or reuses a consumed output.

| Maximum per arm | Value |
|---|---:|
| Student updates including restart | 513 |
| Forward passes | 8,829 |
| Generated tokens including stop tokens | 7,256 |
| Padded input tokens | 2,260,224 |
| Sequential model loads | 2 |
| Source encodes / retrievals / writes | 949 / 179 / 392 |
| Reserved stage wall / outer process wall | 7,080 / 7,200 seconds |
| Artifacts, two / nine / all readers | 256 / 512 / 1,152 MiB |

A single exclusive native worker enforces 50 GiB sampled MPS driver memory,
512 MiB sustained swap growth, and at least 15% host free. These are limits,
not predictions of physical fit. Logs, source hashing, synchronized computation,
teacher work, generated tokens, durable reads/writes and restart are accounted.
The two-site artifact envelope is larger than v10's and is explicitly bound.
No download, duplicate original checkpoint or full-backbone export is needed.

The comparison changes coverage and total exposure together at fixed passes per
world. It cannot isolate coverage causally from computation, establish broad
reasoning transfer, or replace independent seeds/rule families. Later density
interpretation still requires matched-parameter and matched-total-cost controls.

## Evidence and next boundary

The initial working-tree suite passes 60 CPU tests. The actual tiny BF16 2/9/36
workflow creates 32 teacher targets per arm, reaches producer and early-reader
gradients, shares one snapshot, preserves exact live/durable logits, restores a
complete extension and optimizer in a fresh process, and makes one continuation
update. It took 18.114 seconds with 567,427,072 bytes parent peak RSS. Teacher
qualification is explicitly synthetic on that fixture; none of this is native
competence, useful recall or persistent learning. See the [runbook](../../reports/intrinsic_memory_coverage_v11/README.md)
for clean-checkout and native results as they become available.

P remains P1 with zero updates. Useful recall must precede completed-interaction
formation, exactly-once chunk commits, hidden-state-conditioned P learning and
source-free future-query tests. Acquisition, retention, correction and rollback
must precede nested adaptation and RSI. Preserve the [Replica successor
contracts](20261029_Spark_Memory_Formation_and_Replica_Successors.md): external spine
for specialists/sensors/actors; separate E/S/P/phi/C/I/K; approved identity outside
task reward; a bounded active core and explicit archival recall. Hot/warm/cold
movement changes residency, not learning clocks. Native multimodal work is deferred.

## Prelaunch resource revision

The first prepared packet was never launched and is preserved in
[preparation revision 1](../../reports/intrinsic_memory_coverage_v11/preparation_revision1/supersession.json).
A new CPU probe of the actual durable audit writer measured median writes of
0.079/0.153/0.308 seconds at 2,036/4,072/8,144 historical-shaped events. This is
isolated CPU audit overhead, not a native runtime forecast. V10's measured
32-update stages grew from 31.2 to 46.0 seconds as that ledger grew. The initial
240-second allowance for each 128-update v11 stage was therefore too tight.

Before any v11 native answer, revision 2 assigns 550 seconds to each fitting
stage, retains 900 seconds per fitting probe, and reduces the generous setup and
development allocations. Total stage reservation remains 7,080 seconds, outer
wall 7,200 seconds; all scientific choices, work counts and physical caps remain
unchanged. Exact implementation, protocol, packet and standing-authorization
bindings were refreshed. No consumed attempt or historical result was changed.

## Recorded native attempt01

The attempt completed. Frozen calibration: 8/8; fitting-teacher qualification:
32/32. Both false-recall counts are zero; inspect the raw receipt for all
termination/truncation details. Recorded fitting updates: 512.

| Updates | Contextual | Exact | Wrong | Shuffled | Zero | Absent | Fit gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 22/32 | 32/32 | 6/32 | 20/32 | 8/32 | 8/32 | false |
| 256 | 25/32 | 32/32 | 4/32 | 25/32 | 8/32 | 8/32 | false |
| 384 | 26/32 | 32/32 | 3/32 | 25/32 | 8/32 | 8/32 | false |
| 512 | 26/32 | 32/32 | 2/32 | 26/32 | 8/32 | 8/32 | false |

Development remains unopened because the final fitting gate did not pass. Useful intrinsic recall remains unqualified.

A fresh process restored exact logits and the teacher cache, then performed one actual continuation update. Original weights remained unchanged in both processes. This is numerical continuation; uninterrupted-update equivalence was not measured.

Outer wall 3656.043 s; peak driver 10.488 GiB;
peak RSS 8610463744 bytes; swap growth 0 bytes;
minimum host free 69.0%; artifacts 61785311 bytes.
See [terminal summary](../../reports/intrinsic_memory_coverage_v11/native_attempt01_summary.json) for exact
work reservations, stage costs and separately charged audit I/O. Source encoding,
teacher targets, retrieval, publication, probes and continuation are included.
Raw answers and all failed controls remain preserved. Native 9/36 remains closed. Review final fitting groups and distinguish source representation from reader alignment before a new two-site mechanism experiment. Keep fixed gates; no best-prefix selection or retry of this output.

New final clean verification is 562/562, zero skips, at `71f7ab4d` in
239.645 s. The clean tiny 2/9/36 workflow at `607ba7a8` took
18.968 s. No nine/all-layer native run, P update or promotion is
claimed by this receipt. The [requirement audit](../../reports/intrinsic_memory_coverage_v11/requirement_audit_terminal.json)
keeps earlier implementation, new engineering checks and unqualified successors separate.

The four final fitting groups score 5/8, 6/8, 7/8 and 8/8; only the last passes.
Contextual false recall is zero. Direct entity lookup totals 11/16 and ordered
revision lookup 15/16. The final prefix is still the only admissible selection;
the earlier group that reached 8/8 does not substitute for the final result.
This is fitting performance, not held-out utility. Native correction generation
was not reached in v11; the older failed correction result remains historical.

Recorded work is 3,189 forwards, 513 backward/update operations, 1,616 generated
tokens, 816,384 padded input tokens, two sequential model loads, 932 source
encodes, 161 retrievals and 356 declared writes. The audit additionally serialized
37,101,387,705 bytes. Source validation hashed 24,716,988,741 bytes inside this
attempt. These byte counters are not measurements of physical SSD traffic.
The [budget receipt](../../reports/intrinsic_memory_coverage_v11/budget_compliance.json)
checks all recorded stage/count limits and sampled resources. The final probe
took 887.769 of its 900 seconds; its budget was not enlarged after observations.

The [post-run CPU producer probe](../../reports/intrinsic_memory_coverage_v11/frozen_producer_diagnosis.json)
uses only the verified extension and already-opened fitting text. It finds
nonzero differences between paired counterfactual sources; mean relative RMS is
0.295 for v11 versus 0.218 for v10 on these same texts. Mean within-span slot
cosine is 0.631 versus 0.583. It took 0.104 seconds, loaded no native backbone and
performed no update. Different exposure prevents a causal comparison, and these
statistics do not demonstrate recoverable answers or explain the recall failure.
The separate [byte-coverage diagnostic](../../reports/intrinsic_memory_coverage_v11/source_byte_coverage.json)
finds five public development-name bytes absent from fitting text. Development
labels remain unopened; that observation cannot explain the observed fitting
failure. Do not alter this experiment or its criteria using either diagnostic.

The next mechanism study should distinguish a source encoder initialized without
pretrained language features from reader alignment. A frozen native source-feature
producer is a candidate, with live trainable compression and unchanged two-site
readout. It requires its own implemented producer/representation identities,
source-feature costs, gradient/durable tests and exact packet before launch.
That successor is not yet native-launchable. Native density remains closed.

Separately, the [fixed-parameter reader control](20261029_Spark_Memory_Matched_Readers_V1.md)
is implemented and passes 29 clean CPU tests. This prepares a later density
control; it does not change v11's reader, scientific result or authorization.
The [prelaunch audit erratum](../../reports/intrinsic_memory_coverage_v11/prelaunch_audit_navigation_errata.json)
resolves three descriptive references to the unused first packet revision. The
actual launch bindings were correct; the historical audit is preserved unchanged.
