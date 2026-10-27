# SparkX 2.6 4B: nested-learning review

**Current direction:** [the reviewed v5.1 plan](../../architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md) expands learning capacity, corrects optimizer/distillation claims and makes F optional. [The MiniCPM program](../../architecture/25_MiniCPM5_Direction_Correction_and_Learning_Capacity_Plan.md) preserves existing NL IDs and adds new studies. The revision 4.5 tables below are historical controls, not the current expanded capacity profile. Existing executable contracts remain in force.

**26 October 2026 · Architecture revision 4.5 companion · Design review, not a trained capability.**

## Quick reading

**The direction is defensible. The missing part was a precise learning algorithm connecting the components.** Native memory, plastic parameters, visual/audio representations and a controller can support learning over time. Their presence alone does not demonstrate that the system has learned how to learn.

The revised product has three practical learning speeds: observation-level associative memory, experience-level plastic knowledge, and slower consolidation. Offline training prepares these mechanisms to improve later behavior. The original 4B backbone and modality encoders remain frozen during deployment. Full offline fine-tuning still establishes multimodal competence first.

| Decision | Revision 4.5 conclusion |
|---|---|
| Native architecture | Keep memory and plastic experts inside the decoder. The optional spine remains separate. |
| Nested learning | Specify objectives, context, update clocks, transfer edges and reset rules; demonstrate post-adaptation benefit. |
| First concrete candidate | Linear fast memory plus a 16-expert plastic bank partitioned into eight faster and eight slower experts. This partition is a controlled hypothesis, not an established optimum. |
| Offline preparation | Full multimodal training, then chronological teaching episodes with future-query and retention objectives. |
| Optimizer | New parameter training follows the repository's fused AdamW policy. Earlier SGD estimates are historical controls. |
| Meta-gradients | First-order adaptation is an explicit approximation. Exact differentiation through fused AdamW is unsupported; no silent claim of HOPE reproduction. |
| Efficiency | Charge optimizer moments, staging, replay, extra forwards and downstream backward. The previous 128 MiB illustration does not fit the full resident Adam state. |
| Vision/audio | Test preservation of evidence use as well as answer accuracy; acquire knowledge through already-trained perception. |
| Vocab | Swappable surface forms never become the identity of learned concepts or memory keys. |
| Self-improvement | Persistent knowledge acquisition is the initial target. Improving the learning policy itself has a separate later gate. |

## Detailed reading

1. [Research comparison](Research_Review.md): 16 primary papers, what they support, and what they do not establish.
2. [Learning contract](Nested_Learning_Contract.md): losses, clocks, gradient paths, transactions, vocabulary and multimodal connections.
3. [MiniCPM validation and issue ledger](Validation_and_Open_Issues.md): experiments that can falsify the design, current failures and promotion criteria.
4. [Audit report](../../../reports/sparkx_nested_learning_20261026/README.md): repository observations, source hashes and resource corrections.
5. [Machine-readable candidate](../../../configs/sparkx_2_6_4b_nested_learning_v1.json): research specification; not a runnable training configuration.

The [unified v5.1 blueprint](../../architecture/23_SparkX_2_6_4B_Final_Product_Blueprint.md) is the current entry point. [State and RSI contracts](../../architecture/22_State_Mechanism_and_RSI_Contracts.md) continue to govern ownership, transactions and promotion. This companion refines their learning algorithm and evidence requirements; it does not replace them.

## Current evidence

The latest reviewed MiniCPM vision diagnostic still fails learned quality: train04 scores 3/14 train-fit and 2/18 development answers, with zero caption-following accuracy; it stops before audit. Earlier memory recall remains 110/240 with zero optimizer updates, although fresh-process restoration works. These are useful engineering results and failed capability results.

The dataset catalog is metadata-qualified, and the first tranche contains six tiny synthetic protocol fixtures. Neither establishes real-media training readiness. No native nested learner, persistent plastic-knowledge transfer, or recursive learning-policy improvement has been qualified here.

**Design closure means every important claim now has a mechanism, boundary and test. It does not mean the empirical questions have been answered.**
