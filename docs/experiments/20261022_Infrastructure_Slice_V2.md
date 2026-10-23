# S1.2 infrastructure release and S1.4 development preflight

22 October 2026, America/Chicago. This package advances the independent data and
evaluation prerequisites after the completed D2 diagnostic. It uses the existing
MiniCPM5 original and preserves all selected checkpoints. Architecture comparison
training remains behind S1.3/S1.4/S1.5 admission.

## Released data and independently checked truth

The [frozen design](../../configs/infrastructure_slice_v2_design.json) precedes
release generation. The [release manifest](../../data/diagnostics/infrastructure_slice_v2/manifest.json)
binds raw source files, event byte spans, role assignments, derivations, compiler,
generator and independent solver hashes. Generator labels come from sampled
parameters; the separate verifier reconstructs truth from authenticated source
bytes, observed rules/facts and query cutoffs. Target and oracle annotations do
not supply the verifier's answers or the model's prompt.

| Scope | Worlds | Targets |
| --- | ---: | ---: |
| Train | 600 | 1,200 |
| Dev | 100 | 200 |
| Calibration | 100 | 200 |
| Promotion | 100 | 200 |
| Audit fixtures | 100 | 200 |
| Total | 1,000 | 2,000 |

Each world has two related queries. Register XOR, affine arithmetic modulo 97,
finite-state transitions and static tool-recovery recommendations contribute
500 targets each. Every role contains 50% answer, 25% ask and 25% unknown targets.
Late correction, late observation and tombstone cases exercise the chronology
contract. Tool recommendations are text answers (`RETRY lookup`, `VERIFY receipt`,
`STOP`); no tool operation executes. Roles/worlds are assigned before generation;
59 sampled semantic duplicates were rejected and resampled, with zero duplicate
world signatures admitted across the release. The rejection journal is retained.

All 2,000 claimed targets independently verify. The deterministic operational
[review sample](../../reports/infrastructure_preflight_v2_20261022/stratified_review.json)
covers 20 episodes in each family × scenario stratum: 320 episodes from 160 worlds,
including both related queries. This is an automated semantic/chronology check,
not a human annotation study or proof of zero generator error. The 16 selected
prompt/evidence/answer bundles were also inspected during integration, including
the affine residues and state-transition paths.

The four mechanisms and grammar are shared across roles. Separation supports
unseen-world infrastructure checks; it does not establish unseen-mechanism
generalization. All role packets are public, inspected synthetic fixtures;
promotion/audit are exposed and cannot become a sealed scientific benchmark.
The compiler resolves active eligible observations before constructing the
native prompt. Native evaluation therefore tests reasoning over that supplied
snapshot, rather than autonomous temporal retrieval from full history. No public
corpus, teacher, authentic personal record or biography was used.

## Actual local token accounting

The existing offline original tokenizer/template, with thinking disabled, admits
all 2,000 rows under 512 native full-sequence and source-writer tokens. No row was
truncated or dropped. [Per-example accounting](../../reports/infrastructure_preflight_v2_20261022/token_accounting.json)
records full/prompt/support/answer/EOS views, role/family summaries and lengths.

| Single-pass quantity | Train only | All roles |
| --- | ---: | ---: |
| Native prompt + answer + EOS | 300,369 | 501,800 |
| Supervised answer + EOS | 4,374 | 7,291 |
| Separate augmented source-writer view | 233,295 | 390,009 |
| Maximum native full sequence | 326 | 334 |

The source writer is additional architecture work, not extra native sequence
tokens. These are dataset counts, not measured training compute or permission
to train on non-training roles. Only train/dev have persisted flattened SFT;
other role packets remain evaluator-side canonical fixtures.

## Locked evaluator and controls

The [preflight specification](../../reports/infrastructure_preflight_v2_20261022/evaluation_spec.json)
locks 16 development cases before native generation: four distinct worlds per
family, one per scenario, eight answer/four ask/four unknown targets. Selection
uses the first sorted world in each stratum, with the declared earlier/later
query rule. All selected cases have different worlds. Native settings are the
same sealed original B1 source-text condition, MPS FP32, fallback disabled,
greedy decoding, thinking disabled, seed 17, cap 32 and reference ceiling 512.
This is a bounded descriptive probe, with no optimizer update or checkpoint save.

Whole-response parsing reports format validity, action, decision type, value,
task correctness and exact text separately. EOS/cap status remains a separate
dimension. Extra prose cannot supply a numeric answer through substring
extraction. The unchanged raw backend retains historical XOR descriptions and
legacy aggregates; the new release-aware independent scorer is authoritative
for these four families and rechecks source truth, selected rows, token IDs,
prompt/template hashes and ending metadata.

On all 200 dev targets, the best constant response is `UNKNOWN`: 50/200 (25%).
Constant `0` gets 1/200 (0.5%). No constant can pass a perfect-fit control.
The [semantic controls](../../reports/infrastructure_preflight_v2_20261022/constant_controls.json)
also remove all facts while retaining the rule and recompute expected missing
field/unknown decisions: all 100 originally answerable dev cases become
abstentions. These are CPU evaluator controls over explicitly unauthenticated
altered views; they are not model evidence-sensitivity measurements and do not
receive raw-source verification receipts.

## Native result: execution passes, output contract fails

The [independent native result](../../reports/infrastructure_preflight_v2_20261022/native_B1_scoring.json)
is **0/16 task-correct and 0/16 format-valid**, with **16/16 cap hits and no EOS**.
All four family scores are 0/4. The model emits explanatory prose, producing 15
distinct completions; this is not the old constant-zero collapse. For example,
the first expected answer is `252`, while the completion begins “We need to
compute INPUT XOR MASK” and consumes its budget restating the setup. Extra prose
and unfinished answers remain failures under the frozen parser.

The question contract specifies missing-field behavior but does not explicitly
request the complete canonical one-line grammar. An output-instruction/budget
mismatch is a development hypothesis, not an isolated causal finding. These
capped generations do not establish underlying semantic inability, and cannot
support a useful-gain comparison. The cap and parser were not changed after
seeing the answers. Preserve this negative when versioning the next prompt.

| Measured native execution | Result |
| --- | ---: |
| Generated tokens | 512 |
| Native prompt tokens, selected 16 | 3,946 |
| Synchronized generation time | 13.151 s |
| Profiled process lifetime | 17.786 s |
| Guard lifetime | 20.192 s |
| Sampled peak MPS live / driver | 9.416 / 10.081 GiB |
| Sampled RSS / process lifetime RSS peak | 4.043 / 14.491 GiB |
| Observed guard swap growth | 0 bytes |
| Minimum sampled host free percentage | 71% |

The guard replays exactly against its traces and passes its declared 40 GiB
driver, 512 MiB swap-growth and sustained 15% host-free limits. Driver cadence is
.25 s and host cadence 2 s; samples can miss transients. RSS/MPS counters overlap
and must not be added. MPS executes the model; Neural Engine use is unqualified.
The full [raw archive](../../reports/infrastructure_preflight_v2_20261022/native_archive_manifest.json)
contains answers/token IDs, profiler samples, guard and child log. The original
small run directory also remains for direct guard replay.

The original model's nine sealed files reverify before and after generation;
all 36 file identities and the acquisition receipt remain unchanged. There are
**zero downloads, optimizer updates or new checkpoints**. Selected D0/D1/D2 and
negative exports remain preserved. [Source reuse evidence](../../reports/infrastructure_source_reuse_20261022.json).

The full suite passes **613 tests with one restricted-context MPS skip**. The
native preflight above separately executes on MPS and is not that skipped unit
test. [Code validation](../../reports/validation_infrastructure_v2_20261022.json)
binds tested code, command and xUnit output. All 33 roadmap stage IDs and their
dependencies remain unchanged.

The next development action is to predeclare a versioned explicit one-line
output contract and a fixed-budget B1 probe, preserving this original negative.
Retention, calibration, B0/B1 and measured shuffled/missing-evidence comparisons
remain required before S1.4 completion. No S2.4 architecture training has started.

The [public-source metadata review](20261022_Public_Source_Admission_Review.md)
pins candidate revisions and official schemas but leaves both sources
quarantined. PersonaMem inherited component terms, history versions and identity
overlap need resolution; complete pairs and payload bytes were not admitted.
S1.3 adapters and S1.5 release/budget/exposure analysis remain pending. The
[S2.4 screen proposal](20261022_S2_4_Screen_Proposal.md) records the comparison
controls and unresolved budgets; it is not a final preregistration.

The next smallest step is to close the locked evaluation controls and public
source admission decisions, then freeze the pilot mixture and fair comparison.
The existing D0/D1/D2 diagnostics stay preserved as bounded infrastructure
evidence. Useful architecture gain, retention, longitudinal learning and Replica
fidelity remain unmeasured.
