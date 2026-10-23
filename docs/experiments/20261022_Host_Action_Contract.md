# Early S3.1 host action contract

22 October 2026, America/Chicago. This is bounded architecture preparation
under the [explicit early S3.1 scope](../stages/04_Learned_Operating_Loop.md).
That scope may proceed after S1.1 while neural comparison prerequisites remain
open. Full S3.1 still requires S2.5. This package does not complete S3.1 or
establish learned cognition, memory benefit or autonomous improvement.

The implementation is a small, separate
[host action contract](../../src/replica_lab/host_action_contract.py).
It preserves the existing untracked `runtime_pilot.py` draft. It does not import
or depend on that draft, existing model code or a learner.

## Fixed boundary

`HostBinding` declares a model-manifest hash, policy hash, explicit identity
mode, subject, branch, proposing agent, exact goal, evaluation purpose,
observation cutoff, allowed source IDs, and decision/tool budgets. The fixture
runner hashes the existing local sealed-model lock to identify the declared
original-model manifest. That declaration does not load, authenticate or attest
model execution. The policy hash binds both contract-module bytes and fixture
runner bytes. Fixture proposals come from deterministic test code.

Host observations contain exact text and its SHA256, source revision, subject,
branch, observed time, purpose and present eligibility. A host snapshot has a
monotonically increasing epoch. Source-snapshot/current modes expose source
records only; assistance/parallel modes can also expose their declared branch.
The host checks allowed IDs, subject, branch, time, purpose and eligibility before
calling the sole tool. The caller remains responsible for authenticating these
observations and synchronizing current revocation before each submission.

The strict proposal grammar has only three forms:

| Kind | Model-supplied fields beyond `action_id`, `kind`, `epoch` | Host action |
| --- | --- | --- |
| `verify` | `source_ids` | Validate scope/budget, then execute `register_xor8.v1` on exact host observations |
| `answer` | `receipt_id`, integer `value` in 0–255 | Release only a matching successful actual receipt from the current epoch |
| `stop` | bounded `reason` | Stop new work |

Unknown fields and arbitrary tool names are rejected. A model cannot supply
host provenance, observations, receipts, commands, register values for tool input,
or authority. A hypothetical or correctly guessed answer cannot substitute for
an actual verification receipt. The host stamps subject/branch/agent, identity
mode, model/policy hashes and epoch onto recorded new-action outcomes. A stale
duplicate returns an administrative denial without appending a new action.

The actual pure tool parses exact `INPUT=n` and `MASK=n` lines. Missing,
malformed or duplicate registers produce a recorded failure with a null value.
Failures consume a tool call; source/scope rejection does not execute the tool.
Every well-formed new proposal that reaches the available decision budget
consumes a decision, including rejected actions. Further budget denials remain
journaled without executing tools. The journal has a separate 2,048-record bound.

## Replay and invalidation

An action ID binds one exact proposal. A duplicate returns its previous outcome
without executing the tool or spending another decision **only while its epoch
is current**. Reusing an ID with different proposal bytes fails. Epoch checking
precedes returning a cached answer or verification result. Updating/revoking a
source requires a higher host epoch and conservatively clears every live
receipt. Old receipts cannot authorize a new-epoch answer. Historical successful
records stay historical and do not become current authority.

The canonical JSONL journal records creation, host snapshots, proposals and
host-generated outcomes in a SHA256 chain. Restart binds the expected host
configuration, validates canonical bytes and chain order, reconstructs spent
budgets and receipts, and checks recorded pure results against their recorded
input bytes. It does not call the actual tool entry point again. An optional
externally retained trusted head detects truncation relative to that head.
Restart then applies the caller's current snapshot before permitting reuse.
The fixture restart supplies the pre-write session head explicitly. Without an
externally retained trusted head, truncating a journal to an otherwise valid
prefix can pass the chain check. External trust-store persistence and
authentication are not implemented here.

The chain provides integrity relative to an independently trusted head, not
source authentication or a signature. An attacker controlling the entire journal
and its trusted head can rewrite history. Journal snapshots preserve old source
text and receipts. Revocation changes current use; it does not erase historical
bytes, delivered output, backups or learned parameters. The fixture offers
immutable evidence-file writes, not a production crash-safe transaction store.

## Verification and remaining scope

[Focused tests](../../tests/test_host_action_contract.py) exercise actual XOR
success, missing/malformed/duplicate-register failures, exact receipt/answer
binding, model-forged fields, wrong scopes/times/purposes, budgets, restart,
duplicate retries, changed/revoked/removed sources, stale cached answers, and
tampered/truncated journals. Restart tests replace the actual tool entry point
with a failure to verify that replay does not invoke it. These are host software
checks and do not measure neural reasoning.

The [executed fixture report](../../reports/host_action_contract_20261022/report.json)
passes all 12 integration checks, including actual successful/failed tool calls,
restart and source invalidation. The focused contract suite has 38 passing tests.

Run the small fixture package with the existing environment and local lock:

```sh
.venv/bin/python scripts/check_host_action_contract.py \
  --output reports/host_action_contract_20261022
```

The runner creates fresh small journals and `report.json`; it never downloads,
loads a model, trains or saves checkpoints. Host file work and the pure solver
use the CPU. Any future model proposer must execute on the M2 Ultra MPS GPU,
under a separate frozen protocol and the existing resource guards.

Ask/retrieve/revise policies, subgoals/hypotheses and richer task continuation
remain pending. This contract persists the exact goal and actual action trace;
evolving goals, next-action workspace and reconstruction of a learned plan remain
pending. This does not pass the full task-restart gate. Also pending are actual
neural proposals, evidence-store admission, experience-to-learning plumbing,
isolated neural candidates, independent promotion, and selected-system
integration. There is no streaming or concurrent request cancellation, external
side-effect tool, KV/recurrent/fast state, neural cache invalidation, source
expiry service or process isolation. The later selected-system checks remain
those in the [cognitive contracts](20261022_Cognitive_Contracts.md).
