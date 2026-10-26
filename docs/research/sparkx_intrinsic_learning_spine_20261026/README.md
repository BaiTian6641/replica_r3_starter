# SparkX 2.6: intrinsic learning and a separate neural spine

26 October 2026. Research and design update for architecture revision 4.3. No training, model execution or device operation was performed.

The product has a stable language backbone trained to perceive, reason and learn, together with **permanent, plastic memory components inside its decoder computation**. The Unified Access Port (UAP) is a separate interface for additional sensors, action heads, other models and selected internal readouts. Intrinsic memory does not depend on a peripheral being attached. Full offline fine-tuning establishes multimodal capability; later self-learning acquires knowledge and experience using that trained capability.

Read these together:

- [Research review](Research_Review.md): eight newly downloaded papers, three revisited foundations, and pinned MemT/TTT implementation checks.
- [Intrinsic architecture and spine specification](Architecture_and_Spine.md): what becomes part of the backbone, what changes during use, and how native sensory/action exchange works.
- [Teaching environment and MiniCPM validation](Teaching_and_Validation.md): persistent learning with frozen stable weights, audiovisual transfer, vocabulary turnover and separate intervention tests.
- [Consolidated product plan, revision 4.3](../../architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md).
- [Research configuration](../../../configs/sparkx_2_6_4b_intrinsic_learning_spine_v1.json) and [evidence/resource report](../../../reports/sparkx_intrinsic_learning_spine_20261026/README.md).

This update supersedes two interpretations in revision 4.2: intrinsic memory is not an optional UAP peripheral, and persistent learning does not require changes to the stable 4B weights. Retaining trained dynamic memory parameters while clearing episodes/caches is a valid learning test. Clearing those parameters tests whether learning has moved elsewhere.

The earlier packs remain dated research records. Their receipts validate the document revisions that existed when they were written; they do not qualify the current architecture or learned behavior.
