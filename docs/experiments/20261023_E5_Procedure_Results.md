# E5 procedure candidate: accepted repair procedure, not a neural policy

23 October 2026 UTC. Protocol committed before execution (`configs/e5_procedure_v1.json`).
CPU-only host/tool run, no native model load, no optimizer update.

**Result: procedure candidate accepted 3/3 held-out + 1/1 dev, with independent
evaluator receipts; harmful candidate rejected; active manifest published and
rollback tested. This is a procedure-library result, not model-selected action
or a neural parameter improvement.**

## 1. Candidate and lineage

A `RepairProcedure` was compiled from four verified teacher trajectories in the
fresh E3 `set_constant` family:

1. read the workspace configuration;
2. run the evaluator;
3. parse the disclosed numeric bound from the actual failure and call
   `set_constant`;
4. run the evaluator again.

The artifact declares source event IDs
(`teacher:set_train_retry_01:step3`, …, four train cases), preconditions
(“evaluator fails with a disclosed numeric bound”), independent tests
(“evaluator exits zero”), and exact tool steps. A `LearningProposal` binds those
sources, the train manifest, tool-only budget and the evaluation plan. No
teacher output was treated as a verified fact beyond the actual tool receipts.

## 2. Held-out execution

Each case was copied to an isolated temporary directory, executed with the host
tool sequence, then checked by `verify_success()` (a separate evaluator path).

| Case | Role | Procedure | Independent evaluator |
|---|---|---|---|
| `set_heldout_timeout_06` | heldout | succeeded | pass |
| `set_heldout_backoff_07` | heldout | succeeded | pass |
| `set_heldout_poll_08` | heldout | succeeded | pass |
| `set_dev_retry_05` | dev | succeeded | pass |

All non-target constants remained unchanged because `set_constant` edits only
the disclosed literal. The report retains all four tool receipts, parsed
name/value, final exit codes and workspace hashes:
[report](../../reports/e5_procedure_20261023/report.json).

## 3. Publication, rejection and rollback

`ManifestRegistry` published the accepted procedure payload and active pointer
`b24e7722d6bbf0f5f0975901cc8e7e46293e69badd157fb0a888324d92af3941` with no
parent. A deliberately marked harmful candidate was refused by the publisher
(`rejected: harmful_candidate_probe`). The existing manifest registry rollback
path was additionally tested: publish accepted v1 → accepted v2 → rollback to
v1 restores the ancestor active pointer. No model weights or base checkpoint
were saved.

## 4. Interpretation

This procedure solves the fresh task family that the model action policy failed
(0/7 in the E3 `set_constant` arm). That separates two claims:

- **Task solvability:** a bounded host procedure can repair the fixture family
  and satisfy independent verification.
- **Model-selected action:** still negative. The procedure is not selected by
  MiniCPM; it is a compiled baseline and cannot be reported as learned model
  behavior.

The result is useful as the procedure/T0 baseline for later model action
policy and learning comparisons. It does **not** establish neural candidate
training, retention of a trained model, practice selection, or personal
fidelity.

## 5. Status and next

- Implemented: procedure artifact, failure-bound parser, host executor,
  learning proposal, independent evaluation and registry publication/rejection/
  rollback.
- Trained: no neural parameters; no optimizer update.
- Evaluated: 3/3 held-out + 1/1 dev procedure completion; harmful candidate
  rejected; rollback pass.
- Pending: a real neural candidate with declared trainable scope/optimizer,
  independent held-out gain and retention; learned procedure selection; E3
  model action-policy improvement.
