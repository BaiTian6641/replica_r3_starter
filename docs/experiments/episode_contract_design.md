# S1.1 — bounded causal episode contract

**22 October 2026, America/Chicago.** This implements one local canonical intake object and deterministic export for [S1.1](../stages/02_Data_and_Evaluation.md). It does not implement the 2,000/20,000-target data factory, public-source adapters, an authentic Replica archive or production source authentication. Existing trainer contracts and the three top-level schemas are unchanged; the new schema is nested at `schemas/episodes/episode.schema.json`.

## API and trust boundary

`replica_lab.episodes` has no model, numerical or network dependency:

- `Episode.from_dict(payload)` / `validate_episode(payload)` check exact fields, ownership, causal order and declared lineage. The resulting snapshot is immutable; `to_dict()` returns a detached copy and `sha256` binds its canonical JSON bytes.
- `input_view(episode, purpose="local_retrieval")` returns only `prompt`, `memory_texts` and `source_ids`. It never copies `target`, `evaluator.oracle` or `evaluator_only` events. It selects active observations at the query cutoff and then applies current per-source permissions. Audit inputs require an explicit `purpose="local_evaluation"`; default retrieval, sharing, training and teacher views are rejected even if declared permissions claim otherwise.
- `verify_episode(episode, verifier_id=..., verifier_sha256=..., verifier=reviewed_callable)` invokes host-supplied verification code and returns a receipt bound to the complete episode bytes. A wrong/edited episode cannot reuse the receipt. A receipt is a local trusted-host attestation, not a signature, proof that the callback is independent, or a sandbox. The host must supply/review the verifier and correctly bind its declared code hash.
- `export_sft(episode, verification=receipt, purpose="local_training")` requires passing verification by default and emits exactly the existing eight fields. Target-bearing exports accept only `local_training`, `local_evaluation` and `external_teacher`; retrieval or sharing permission cannot authorize response/gold export. Training/teacher exports require `split=train`; local evaluation can export the declared evaluation roles. Audit export is evaluation-only. `require_verified=False` emits a quarantined `verified:false` row without inventing verification, while retaining the same purpose/role restrictions.
- `audit_episode_families(episodes)` checks duplicate IDs, source-identity drift, source/family split inheritance, complete parent lineage and parent cycles across the supplied corpus. Identical declared source SHA values must retain one group/split even when IDs are renamed. Existing `(source_id,event_id)` event bytes are immutable across descendants; changes require a new event with an explicit correction chain. It does not discover undeclared parents or semantic duplicates beyond those declared identities/hashes.

Source identity, consent/use permissions and `source.sha256` are **declared trusted intake metadata**. This API checks hash syntax and consistency, not the actual source bytes, identity or consent. Raw artifact hashing, source authentication, license review and secure storage remain intake/host responsibilities. A `human_source` must bind its subject actor to the `source` branch; agent experience/inference cannot own that source history. A Replica branch may read allowed source history and its own observations, with ownership retained in exported evidence text. It cannot rewrite a source-owned fact through an agent-owned correction.

Corpus auditing must run on the complete proposed release before exporting individual rows; isolated per-episode validation cannot see cross-file hash/lineage conflicts. Treating identical declared bytes as one split/family is a conservative binding rule, not proof that the declared digest matches a raw artifact. The host still owns who can invoke evaluation, audit sealing, exposure logs and permission configuration. An explicit evaluation purpose is a local API boundary, not user authentication or permission to feed audit cases to a learner.

The eight-field export is intentionally a lossy trainer format. Preserve the canonical episodes, verification receipts and declared export purpose beside it in the release manifest. A flattened `verified:true` means the receipt's task check passed; it does not carry consent, authorize a new use, or permit an evaluation export to be reused for training.

An observed source fact can legitimately contain the answer. For example, an earlier measurement saying “the output was 7” is valid evidence for a later question. There is no answer-substring filter. Structurally marked gold/oracle fields are excluded, but **a paraphrased gold label misrepresented as an ordinary observation cannot be detected automatically**. A trusted adapter, source review and an independent task verifier must prevent that semantic laundering. The unknown/ask actions are structural target categories; deciding that an answer is truly unknown still requires the verifier.

## Chronology, corrections, deletion and eligibility

Time is an explicit nonnegative integer logical clock in this first contract. `event_time` says when an event occurred; `observed_at` says when the system learned it. An old event reported at time 8 cannot affect a query at time 5. This version supports occurred events (`event_time <= observed_at`), rather than scheduled future events; real-date adapters need an explicit clock conversion or a later contract revision.

Events are strictly ordered by `(observed_at, event_id)`. Corrections and tombstones reference an earlier active event in the same channel and with the same subject/branch/actor/kind ownership. Forked stale corrections and oracle-to-observation edits are rejected. A correction removes the prior active fact and supplies its replacement only when observed. A tombstone has empty text and removes the active fact after its observation cutoff. Neither an oracle event nor its ID becomes ordinary model evidence.

A **chronological tombstone** changes the episode's as-of-query state after its observation time. **Current privacy/eligibility revocation** is different: current source permissions gate every export, including a historical query before the revocation. An inaccessible correction cannot resurrect its older, formerly eligible predecessor. The tests check both cases. This JSON object does not erase raw files, caches, trained influence or backups; global deletion, derived-artifact invalidation and rebuilding models remain the existing ledger/host's responsibility.

The initial answer contract requires at least one active observed support event. `ask` and `unknown` can have none. Pure prompt-only arithmetic with no evidence record is intentionally outside this evidence-grounded first contract. Support IDs may never reference future, deleted, superseded or evaluator-only observations, and export rejects a target whose declared supporting source lacks permission for that purpose. Missing support annotations or incorrect source classification can still require semantic review.

## Fictional late-correction example and exact export

The following is an API example with a synthetic packet rule, not a personal record. Every example derived from this world keeps `group_id=packet-world-1` and `split=train`. The source hash below is computed for the fictional raw source; the API itself does not fetch or verify it.

```python
import hashlib
from replica_lab.episodes import Episode, export_sft, input_view, verify_episode

uses = ["local_retrieval", "local_training", "local_evaluation"]
payload = {
    "schema_version": "replica.episode.v1",
    "id": "packet-query-after", "group_id": "packet-world-1", "split": "train",
    "subject_id": "fictional-packet", "branch_id": "source", "parent_episode_id": None,
    "privacy": "public", "permissions": uses,
    "query": {"at": 8, "prompt": "Apply the fictional transform to input 6."},
    "sources": [{
        "source_id": "packet-source-1", "group_id": "packet-world-1", "split": "train",
        "sha256": hashlib.sha256(b"mask=1; corrected mask=3").hexdigest(),
        "subject_id": "fictional-packet", "branch_id": "source", "actor_id": "world-generator",
        "kind": "synthetic", "privacy": "public", "permissions": uses,
    }],
    "events": [
        {"event_id": "mask-original", "source_id": "packet-source-1", "event_time": 1,
         "observed_at": 2, "operation": "observe", "replaces_event_id": None,
         "channel": "observed", "text": "The transform XORs input with mask 1."},
        {"event_id": "mask-correction", "source_id": "packet-source-1", "event_time": 2,
         "observed_at": 8, "operation": "correct", "replaces_event_id": "mask-original",
         "channel": "observed", "text": "Correction: the transform uses mask 3."},
    ],
    "target": {"action": "answer", "response": "5", "support_event_ids": ["mask-correction"]},
    "evaluator": {"oracle": {"input": 6, "mask_as_of_query": 3}},
}
episode = Episode.from_dict(payload)

# In a real adapter the reviewed verifier must additionally bind the world,
# prompt and cutoff to raw source truth. This tiny numeric check illustrates
# the callback/receipt interface, not a complete public-dataset verifier.
def xor_check(value):
    oracle = value["evaluator"]["oracle"]
    return value["target"]["response"] == str(oracle["input"] ^ oracle["mask_as_of_query"])

receipt = verify_episode(
    episode, verifier_id="illustrative.xor-v1",
    verifier_sha256=hashlib.sha256(b"illustrative reviewed xor verifier v1").hexdigest(),
    verifier=xor_check,
)
row = export_sft(episode, verification=receipt)
assert row == {
    "id": "packet-query-after", "group_id": "packet-world-1", "split": "train",
    "prompt": "Apply the fictional transform to input 6.", "response": "5",
    "memory_texts": [
        "mask-correction | event_time=2 | observed_at=8 | kind=synthetic | "
        "subject=fictional-packet | branch=source | Correction: the transform uses mask 3."
    ],
    "source_ids": ["packet-source-1"], "verified": True,
}
assert "oracle" not in input_view(episode)
```

For a query at time 5, use `mask-original` as target support and response `7`; the late correction remains unavailable despite its earlier event time. Changing the query/target creates new episode bytes and requires a new verifier receipt. A derived query also needs a new episode ID and declared parent in the corpus audit. There is no automatic rewrite of source-person testimony, no external teacher call and no data download.

## Validation scope

Run the targeted CPU checks with:

```bash
.venv/bin/python -m pytest tests/test_episodes.py -q
```

They cover exact export/schema compatibility, deterministic immutable snapshots, independent numerical label checking, receipt binding, explicit oracle/future exclusion, legitimate answer-bearing observations, late correction, tombstone versus current revocation, permission filtering without stale resurrection, unknown/ask, role/family inheritance, invalid source/agent ownership and chronological/parent cycles. These are bounded S1.1 infrastructure checks. They do not admit a public corpus, authenticate source records, prove semantic nonleakage, or qualify a model training result.
