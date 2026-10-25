# Reproducibility and memory reliability recovery

**Revised 24 October 2026 after the completed v5 writer-repair package.** This is
the immediate work order. It supersedes older next-package prose while
preserving historical protocols, raw answers and rejected candidates. Follow
with [MiniCPM5 vision, visual memory and RSI](18_MiniCPM5_Vision_Training_and_Continual_Learning.md),
then later Spark support. The [loop prompt](16_Loop_Agent_Instructions.md) is the
execution entry point. Documentation itself starts no job; native evidence is
linked below.

## 0. Current v5 outcome and next action

The [v5 fact-binding writer experiment](../experiments/20261024_Memory_Reliability_Writer_Binding_Results.md)
is complete and rejected. One 48-update writer-only arm changed exactly 14
tensors / 5,251,072 parameters. Candidate fast and disabled both score 56/192,
with zero initial/corrected/unaffected facts. Their factual tokens differ on 128
routes, so nonzero influence does not imply useful association. The candidate is
not selected or promoted.

The [canonical corrected-scorer verdict](../../reports/memory_reliability_v5_scorer_fix_eval02/verdict_final.json)
independently decodes 1,920 answers, authenticates 32 images/source bindings and
verifies a seven-answer fresh-process restart. Selective exact context scores
184/192 versus full-context 176/192, but the reference bundle rejects on general
correctness and two full-context non-regressions. Eval01 remains a preserved
failed scoring attempt; eval02 is canonical. Scorer fixes are `eff2079` and
`5aa7e75`. Final clean-export validation is 1,311 tests +21 subtests in 148.69 s,
with 49 focused reliability tests.

**Next:** implement a fresh storage/read-separation probe on observed facts;
compare explicit copy/pointer only when values are recoverable. Do not repeat
writer/reader training on this exposed set. Isolated vision may proceed with
memory/core/identity disabled, but cannot qualify the failed memory path or
integrated RSI.

## 1. Verified starting point

The [reader-only experiment](../experiments/20261024_Memory_Reliability_Reader_Only_Results.md)
completed 48 updates over 192 scheduled training cases, changing eight reader
tensors / 2,105,344 parameters while the original backbone, writers and gate
scalars remained frozen. Across dev and heldout, learned fast memory scores
0/64 facts, 0/16 corrections, 0/48 unaffected facts, 32/32 unknowns and 32/32
general cases. The successful set equals the disabled baseline's successful set.
Selective exact context is 192/192; full context is 190/192. All 32 serialized
memory snapshots/source bindings authenticate, and a fresh process reproduces
7/7 answers and bindings. These snapshots are not visual image inputs.

The [canonical verdict](../../reports/memory_reliability_v2_attempt03/verdict_final_v2.json)
rejects nine gates. The reference bundle rejects because the original protocol
requires correctness equality with the imperfect full-context control. Preserve
that decision; selective retrieval's correct answers remain useful evidence.
The native run used about 11.00 GiB sampled MPS driver memory and 515.48 seconds.
These measurements do not qualify a visual model or a larger training budget.

The review independently decoded the answers, reproduced the failed gates and
reran 37 focused tests successfully. The recorded full integration result is
1,299 tests plus 21 subtests passing in 137.53 seconds; the review did not rerun
that full suite. Passing tests coexist with the open defects below.

| Review finding | Evidence and implication |
|---|---|
| Clean checkout fails scorer import | `reliability_scoring.py` imports untracked `answer_audit.py`; a clean HEAD source export raises `ModuleNotFoundError` |
| Incomplete protocol source package | `scripts/make_memory_reliability_data.py` and `tests/test_memory_reliability.py` are also untracked dependencies named by protocol v4 |
| Exact-context verification gap | Replacing `ExactContext.spans[0].text` while retaining the valid selection/digest still passes `verify_context` |
| Reference gate rejects improvement | Selective 192/192 versus full-context 190/192 fails equality; a prospective correctness/nonregression rule is needed |
| Incorrect routing counter | Native `gate_off_examples` totals 0; 32 unknown + 32 general training examples are gate-off, leaving 128 gate-on factual examples |
| Narrow retention evidence | The 32 evaluation general cases repeat only four distinct arithmetic questions; each split contains four independent counterfactual families |

The context reproduction identifies an API contract defect, not evidence that
the saved native answers were tampered with. Gate-off evaluation parity protects
those tested unrelated answers through bypass; it does not establish learned
noninterference under active memory.

## 2. First package: reproducibility, integrity and scoring

Complete these repairs before new native training:

1. Inventory tracked and untracked runtime, builder, scorer and test dependencies.
   Include the necessary source files in a focused reproducible source commit
   when authorized/permitted; never sweep in unrelated files or the protected
   runtime pilot. Verify imports, dataset/scorer entry points and focused tests
   from a clean tracked export using the declared environment. Model weights
   remain separately sealed local artifacts, not files to add to Git.
2. Authenticate the consumed exact context, not merely its selection receipt.
   Check selected-record membership, current eligibility, expected caller scope,
   record hashes, span offsets/text, duplicates and status. Bind the query if a
   verified context can be reused for another query. Recompute canonical spans
   from the ledger or enforce equivalent byte-level checks. Test modified spans,
   records/status/scope, stale epochs, supersession and revocation at consumption.
3. Repair routing accounting independently of `loss.requires_grad`. Persist
   explicit route, requested versus actually used state, effective trainable
   groups and gradient participation. Keep scheduled examples distinct from
   examples that exercise a learnable memory path. Do not retrofit missing
   per-example gradient evidence into the old native report.
4. For a new versioned protocol, require independently correct exact-reference
   answers and no correct-to-wrong regressions against full context. Agreement
   with full context is a diagnostic, not a requirement to repeat its mistakes.
   Give full-context missing-fact hallucination its own diagnostic. Retain the
   old parity rejection and all native/smoke-fixture provenance unchanged.
5. Expand evaluation diversity and scorer integrity tests. Decode token IDs with
   the sealed tokenizer; derive gold from authenticated data; verify exact
   case/arm membership and reject missing/duplicate rows, changed targets,
   cached-score tampering and inconsistent aggregates. Report family and unique
   question counts alongside answer counts.

Protocol v4's historical scorer pin differs from the current scorer. That is
not permission to repin the old experiment: create new code/data/protocol pins
for the next run. A retrospective smoke fixture must remain explicitly distinct
from native evidence. If Git permissions prevent a commit, report the clean-
commit gate as open, validate an explicit hashed source bundle for independent
work, and never claim clean-checkout reproducibility or a committed pre-run seal.

**Exit:** the intended tracked source imports and its focused tests pass in a
clean environment/export; forged contexts fail; route accounting is consistent;
new scorers pass adversarial fixtures; historical verdicts remain reproducible
under their recorded provenance. Actual native execution also needs a pinned,
reviewable next protocol and valid source artifacts.

## 3. Second package: diagnose storage, readout and correction

Reader-only failure does not prove correct storage. The active fast writer
projects raw input-token embeddings. However, the
[earlier contextual-writer arm](../experiments/20261023_E4_Contextual_Writer_Results.md)
already produced 0/8 heldout recall despite a loss reduction. Do not repeat it
unchanged or assume that contextual features alone solve the failure.

Inspect existing writers, readers and copy/pointer paths before creating more
layers. Use the following discriminating diagnostics on fresh families:

- **Binding sensitivity:** swap entity/attribute/value assignments while keeping
  the token inventory comparable. Identical states for different required
  associations expose lost information; different states alone do not prove
  usable bindings.
- **Storage/read separation:** test whether observed values are recoverable from
  the state through a bounded probe on independent families. An oracle key/value
  representation built only from observed source facts is a labelled diagnostic
  control, never an undeclared deployment mechanism or answer-label write.
- **Decoder/copy separation:** compare the native residual reader with an
  explicit copy/pointer diagnostic when the relevant stored value is known to
  be recoverable. Distinguish failed storage, failed addressing and failed output.
- **Chronological correction:** append observations once, answer multiple queries,
  save/reload without old-source encoding, then append a correction and test
  changed plus unaffected facts. Include missing/conflicting evidence and scopes.
- **Interference:** vary fact count, distractors, delay and query forms; check
  seen values in new bindings and genuinely unseen observed values. Measure
  saturation and unrelated behavior with memory both active and bypassed.

Use the exact-record reference as a practical control. Its current structured
fact parser and lexical coverage gate are narrow mechanisms, not general
semantic retrieval. Exposed v1/v3 evaluations are regressions after inspection,
not fresh final audits. Split complete families before variants, captions or
teacher expansion; audit query-only, first/last-value and template shortcuts.

**Exit:** a report identifies the failing boundary, the diagnostic evidence,
what remains uncertain, and one falsifiable repair hypothesis. A lower loss,
nonzero gradient or changing state digest does not close the quality gate.

## 4. Third package: one targeted MiniCPM5 memory repair

Use the diagnosed failure to select one change: for example an explicit
association objective, a writer representation, query addressing or an output
copy interface. These are alternatives to choose from evidence, not instructions
to activate every mechanism. Preserve the sealed original and keep D1 scope
explicit unless a separate co-adaptation study is declared.

Before native execution, freeze a fresh protocol defining parent/source/data/
code hashes, trainable tensors, optimizer/rates, loss/masks, replay, independent
families, controls and numeric update/token/time/resource caps. Run focused
contracts first. Measure parent, disabled, exact/full-context, shuffled and
last-observation controls. If using teacher distributions, independently check
teacher correctness; the full-context control is not infallible.

Qualify delayed factual gain, incremental correction, unaffected-fact retention,
unknown/conflict handling and broader text retention on both development and
independent evaluation families. Keep per-case regressions and family-level
uncertainty. The four arithmetic queries from the old fixture are insufficient
for a broad retention claim. Report actual scheduled/gate-on examples, gradient
membership and changed tensors, not only optimizer calls.

Save an isolated candidate and independently accept/reject/defer. Run a declared
fresh-process persistence check; a rejected quality candidate may still support
an explicitly labelled mechanical diagnostic. Do not select checkpoints or tune
thresholds on the audit. Preserve negative attempts and count their costs.

Use one exclusive MPS worker with fallback disabled, 50 GiB sampled driver stop,
512 MiB sustained swap-growth and 15% minimum host-free limits. Samples can miss
transients. No guard increase, allocator change, repeated failed recipe or silent
parameter freezing is a recovery action. Preserve `src/replica_lab/runtime_pilot.py`.

**Exit:** a complete, reproducible repair package with measured acceptance or
rejection and a specific next hypothesis. Reliable learned-memory promotion
requires factual gain, correction/retention and restart together. Atomic snapshot
publication is not power-loss durability, and permitted-source rebuild is not
neural unlearning; qualify those claims separately.

## 5. Progression to vision, visual memory and RSI

After the three packages above, follow document 18's V0–V2 MiniCPM5 visual input,
projector alignment and instruction/reasoning training. If learned-memory quality
remains rejected, isolate that vision experiment with memory/core/identity
injections disabled. This preserves priority for memory repairs without turning
an unresolved research result into an indefinite ban on independent perception.

V3 learned visual memory requires both accepted perception and an independently
qualified chosen memory path. Exact visual retrieval is a separate baseline and
must not be relabelled learned memory. Integrated neural RSI promotion requires
useful memory/correction, broad retention, reliable visual behavior where used,
independent outcome verification and tested rejection/rollback. Fixed-practice
and publication plumbing may proceed earlier; C13's 10/10 threshold applies to
learned selection, not to deterministic plumbing.

Keep static Engram, learned multi-rate memory and architecture-changing RSI as
later separate hypotheses. Spark BF16/native/vision support comes after the
MiniCPM5 visual-memory/RSI work has a documented outcome and reusable contracts.
MiniCPM-V is an optional pretrained reference/teacher; it does not replace the
new MiniCPM5 encoder/projector plan. Personality conditioning uses approved
sources and separate fidelity review, without changing factual verification.

## 6. Handoff and preserved evidence

For each package record: changed files; source/protocol/parent/code identities;
clean-export and focused/integration checks; actual trainable groups and updates;
decoded results by split/control and independent family; route/gradient counts;
correction/restart evidence; retention; resource/teacher/verifier cost; candidate
disposition; and the next discriminating action. Mark implemented, tested,
trained, evaluated and unrun separately. Keep historical native artifacts intact.

Preserve the rejected [first recovery pilot](../experiments/20261024_Memory_Reliability_Recovery_Results.md),
[reader-only comparison](../experiments/20261024_Memory_Reliability_Reader_Only_Results.md),
[M3 repair](../experiments/20261024_M3_Retention_Repair_Results.md),
[M4 pattern study](../experiments/20261024_M4_Pattern_Results.md), their protocols
and all failed attempts. Reuse the accepted narrow
[procedure lifecycle](../experiments/20261024_M1_Procedure_Compilation_Results.md).
Do not rebuild passing infrastructure or convert engineering success into a
claim of learned recall, general retention, personality fidelity or RSI.
