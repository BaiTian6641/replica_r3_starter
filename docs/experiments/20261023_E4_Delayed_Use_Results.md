# E4 delayed use: state is read, but new associations do not form

23 October 2026 UTC. Protocols and data were committed before each native run.
**Result: rejected for capability; engineering read/write/image contracts pass.**
The learned fast write projection and native readers use the committed state, but
held-out values are not copied and a no-retraining compositional set shows no
new entity/attribute association. No default or fallback was changed.

## 1. Arm and resources

| Item | Value |
|---|---|
| Data | 24 train, 6 dev, 8 held-out source pages; 38 synthetic cases; disjoint groups |
| Observation | one source page containing the relevant record + 3 eligible distractors |
| Query | later prompt; the source page is not in the prompt |
| Trainable | `fast_write_encoder` and `fast_ports` only; base and E2 parent frozen |
| Write feature | frozen native input-token embedding, trainable key/value projection |
| State | one committed page; 4 heads × 64×64 FP32; read at native blocks 13/27 |
| Training | modules-only, Muon 8-bit + FP32 AdamW, 32 updates × 4, lr 2e-2, seed 20261023 |
| Output | extension v2, 57 tensors / **7.25M** new parameters, 29.0 MB weights |
| Cost | **66.9 s**, **10.15 GiB** sampled driver peak, no swap observed, no base copy |

Initial null parity, E2 parent v1 migration, finite writer/read gradients,
extension-only export, zero-state parity with disabled and persisted-image
reload all pass. These qualify wiring and serialization only.

## 2. Primary held-out result

| Arm | Dev exact | Held-out exact |
|---|---:|---:|
| Correct observation | 0/6 | 0/8 |
| Shuffled observation | 0/6 | 0/8 |
| Zero state | 0/6 | 0/8 |
| Fast branch disabled | 0/6 | 0/8 |

Training loss fell **58.74 → 8.43**, but only **7/24 train** answers were exact.
Correct and shuffled states produced different answers on **6/14** eval cases,
so the state is read and affects generation; it is not ignored. Zero state was
byte-identical to disabled on **14/14** cases, confirming the null branch. The
held-out novel values (`vermilion`, `cerulean`, …) were never emitted.

The first held-out fast image was saved and loaded into a fresh
`FastMemorySession`; state digest and generated answer were identical to the
in-memory correct state. This is **image reload, not a new OS process**.

## 3. Compositional diagnostic (no retraining)

A separate eight-case set used held-out entities/attributes with values that
were present in candidate training. Candidate extension loading in a fresh
adapter was validated.

| Arm | Exact |
|---|---:|
| Correct observation | 1/8 |
| Shuffled observation | 1/8 |
| Zero state | 0/8 |
| Fast disabled | 0/8 |

Correct did not exceed shuffled; state influenced four answers, but no new
entity→value binding was learned. The identical `indigo` hit in correct and
shuffled came from prior/training bias, not the query association. This rules
out “only novel token vocabulary” as the primary explanation: the candidate also
fails to bind seen value tokens to new entities.

## 4. Failure classification

| Cause | Evidence |
|---|---|
| Write/read wiring or ownership | Pass: one commit, producer checks, zero parity, image reload, extension reload |
| Data/leakage | Not implicated: source page is absent from prompt; retrieval bank empty; controls use actual pages |
| Optimization | Partial: loss falls and 7/24 train outputs are exact, so a small linear head fits some training associations |
| Representation | **Primary**: a token-embedding linear writer + fixed LM head does not learn a copy/association transform that generalizes to new source pages; both novel values and new entity bindings fail |
| Capacity/budget | Not indicated at 32 updates / 10.15 GiB; further identical updates would be a repeat, not a new test |

The exact-text record fallback remains correct for novel-value precision. A
future fast-memory arm needs a different write representation or an explicit
copy/retrieval objective, plus a fresh evaluation family. Do not tune these
exposed cases.

## 5. Status

- **Implemented:** learned write projection, one-commit page state, native read
  ports, four controls, persisted image reload, extension v2 compatibility.
- **Trained:** 32 modules-only updates on 24 pages.
- **Evaluated:** primary 0/8; compositional 1/8 with no control gain.
- **Rejected:** no promotion/default change; raw answers and failures preserved.
- **Pending:** contextual/associative writer, copy objective, interference and
  long-horizon replay, actual fresh-process restart in the runtime.
