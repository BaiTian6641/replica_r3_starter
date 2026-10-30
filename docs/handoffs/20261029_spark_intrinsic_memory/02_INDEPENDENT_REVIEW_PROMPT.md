# Prompt: independent correctness and evidence reviewer

Review the continuation in `/Users/flare/replica_r3_starter`. First read
`docs/handoffs/20261029_spark_intrinsic_memory/STATE_AND_BOUNDARIES.md`, the active
root instructions, architecture chapter 28 and the current v11 receipts. Inspect
HEAD and work after `9e75415e`; do not assume the five handed-off v12 files are
still the candidate being reviewed. Record the exact review commit and dirty
file hashes. Never read, import, edit or stage `src/replica_lab/runtime_pilot.py`.

Be independent of the implementation. Begin with read-only review and isolated
reproductions. Own a new review report and review tests; coordinate any production
fix before editing a file owned by another agent. Do not start native work, open
sealed development labels, or change a scientific threshold to make the review
pass. A missing check is not a successful check.

## Review the real computation

Trace the response-loss graph from output through all later frozen blocks,
active reader sites, value slots and trainable source compression. Confirm the
only intended detach is at frozen original source features and declared durable
inference boundaries. Confirm one NativeFusedAdamW owner for every intended
parameter, no duplicate aliases, no original parameters in the optimizer, and
no full-base checkpoint export. Test initial zero-output behavior followed by
actual updates and expected later producer/early-reader gradients.

Verify disabled extensions skip computation and preserve baseline output.
Inspect query/residual placement: block-input query, post-block residual.
Review masks, slot order, span order and the selected exact-text comparator.
Ensure the correct/wrong/shuffled/zero/absent arms alter the declared memory
channel without introducing unmatched evidence or hidden answer caches.

Check that frozen source extraction sees canonical source text alone and uses
the correct source/tokenizer/positions with all extension branches disabled.
It must not receive the question, fitting answer, development label or a teacher
explanation. A detached teacher target is a separate qualified fitting artifact;
it cannot become a durable evidence record or leak into development.

## Review authority, caching and recovery

Exercise source substitution with identical geometry, changed/missing model or
configuration code, local executable dependency changes, stale producer and
codec identities, wrong scope/cutoff, future evidence, correction, revocation,
mixed generations, corrupt payloads and cold I/O racing with invalidation.
Require refusal before model load or value consumption, whichever boundary
applies. Preserve exact canonical fallback with separate cost and arm labeling.

Check cache keys against source authority, snapshot generation, source epochs,
producer/representation/codec, feature contract, reader release and active site.
Provider-release caching must not conceal a changed contract. Feature caches
are C and must be cleared for source-free learning tests. They are not persistent
P or acceptable answer-bearing checkpoint extras.

Retain dirty-expert recovery coverage: admit r1/r2, publish lineage [r1] at clock
1, commit newer hot state at clock 2 with [r1,r2] and updated encoder, remove
the derived index, reopen. Recover the latest committed hot authority coherently
without conflating it with the latest cold publication. Include multiple
unpublished writes, interrupted writes, missing/corrupt newest authority,
revoked dependencies, publication/restart boundaries and unrelated healthy
state. Do not simply relax lineage checks or discard identity-loss protections.

## Review experimental integrity before any load

Verify a new versioned execution closure includes actual auto_map model and
configuration code, local executable dependencies, adapters, producer, reader,
runner, scorer, template, tokenizer and qualified environment origins. A source
name or Git commit alone is not a sufficient execution binding. Changed, missing
or substituted code must fail before `trust_remote_code` model loading.

Check packet/protocol/source/environment/prepared-data/output/authorization
bindings, unique attempt consumption, pre-dispatch work reservation, resource
guard, restart deadlines and failure charging. Source-feature forwards and cache
recomputation need real budget admission; post-hoc counters cannot authorize work.
No failed run refunds already executed work or makes its output reusable.

Review competence and split gates: complete frozen calibration first; all
required teacher/fitting evidence; fitting frozen before development parsing;
strict final-group criteria; no best-prefix choice or pooled bypass. Missing or
incomplete baseline evidence closes downstream learning. Use public identities
and hashes, not sealed label contents, for preflight review. Confirm v11's
unopened split remains unopened before a successor claims it is fresh.

Separate source qualification, numerical qualification, competence, fitting,
utility, retention, persistent learning and promotion. Historical negative
results must remain negative; historical execution failures must not become
capability failures. A copied success receipt does not validate new code.

## Portable verification procedure

Use an isolated clean source-only checkout of the reviewed commit, with the
protected module absent and no pretrained weights. Configure exclusion before
checkout; do not copy the protected module into an archive or test environment.
Use synthetic source trees and explicit Git/source fixtures. Do not weaken
production checks, skip provisioned-source-dependent tests silently, or download
weights to make unit tests pass. Real-source qualification is a separate gate.

The current v11 `verification_summary.json` contains the exact earlier 32-file
affected regression command. Reuse its test scope where relevant, add the new
v12 tests and the separate matched-reader test file, and write outputs to a new
review directory. Do not rerun a historical command with its old `--junitxml`
destination. Inspect imports/collection so no test imports the protected module.
Report the exact new suite and skip reasons; earlier 562 and 29 counts are
references, not a target to manufacture or a single combined historical suite.

Run a new integrated tiny-BF16 CPU workflow: live producer/reader gradients,
one immutable snapshot, unique optimizer ownership, store/cold invalidation,
complete extension reload and actual fresh-process continuation. Compare
continued weights and moments with uninterrupted CPU execution. Record wall
time, RSS, artifact bytes and what the fixture does not establish.

Established checks, from repository root, are:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python reports/sparkx_v5_1_review_20261027/verify.py --root . --tracked auto --no-write
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python reports/sparkx_final_product_20261026/verify.py --root . --out reports/sparkx_final_product_20261026 --no-write
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/prepare_sparkx_research_bundle.py --profile final-product --check
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/check_git_artifacts.py --scope staged
```

In the clean checkout use the qualified available Python path, `--tracked require`
for migration verification, and artifact scope `tracked`. Check that current
versions honor the protected-file exclusion before executing them. Save new
stdout/receipts elsewhere; do not overwrite generated historical results. The
coordinator owns any legitimate refresh of current review/bundle bindings.

## Deliver the independent verdict

Report findings in severity order with file/line, concrete trigger, consequence,
reproduction and required remedy. Distinguish confirmed defects from suspected
risks and coverage gaps. Re-test repaired findings against the final integrated
commit. State whether CPU engineering is qualified and whether a new native
packet is ready; neither verdict establishes useful recall.

Include reviewed commit/hashes, exact newly run commands and counts, recorded
receipts not rerun, protected-file/source-free conditions, resource measurements,
label exposure and unresolved blockers. A review with no findings must still
state its actual coverage and the native/capability work it did not perform.
