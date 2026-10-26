# SparkX 2.6: reusable audiovisual experience

**26 October 2026 · Research and proposed architecture; no training or runtime implementation in this review.**

The recommendation is to build a **native, bounded experience system**: modality-specific sensory features, persistent object/event identities, a compact recurrent bridge, an editable associative bank, and two learned memory ports inside the language backbone. Train these together with the full Spark backbone after component warmup. Add fast weights only if they improve on the explicit bank at comparable cost.

The important change is what a memory contains. A caption or answer is insufficient for remembering a previously unseen visual or acoustic pattern. Keep a shared semantic representation **and small modality-specific evidence vectors**, with time, geometry, uncertainty and source lineage. This gives later reasoning and training access to perceptual evidence that a text summary may discard. Whether the proposed compression preserves enough information is an experimental question.

This batch reviewed a mirrored DeepSeek technical report, thirteen other research papers including the revised video-SALMONN S paper, and selected public implementation files. The original DeepSeek repository's absence did not stop the analysis. Two downloaded copies of its report are byte-identical; that establishes consistency between copies, not author authentication or reproducibility. All source identities, retrieval hashes, failures and reading scopes are recorded in the evidence manifest.

## Deliverables

| Document | Purpose |
|---|---|
| [Updated canonical architecture, revision 4](../../architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md) | Consolidated product decisions, including existing vocabulary and learning contracts |
| [Research review](Research_Review.md) | What the papers actually establish, implementation cross-checks, and limitations |
| [Native experience design](Native_Experience_Design.md) | Tensor/state interfaces, backbone integration, audiovisual learning, and full-FT sequence |
| [MiniCPM validation addendum](Validation_Addendum.md) | Distinguishing memory, pattern transfer and parameter learning before Spark |
| [Machine-readable revision 4](../../../configs/sparkx_2_6_4b_product_plan_v4.json) | Proposed dimensions, caps, controls and gates; not an executable training config |
| [Evidence manifest](../../../reports/sparkx_av_experience_20261026/evidence.json) | URLs, exact source hashes and inspection scope |
| [Repository observation](../../../reports/sparkx_av_experience_20261026/repository_snapshot.json) | Local results at the review cut, including failures |
| [Analytical costs](../../../reports/sparkx_av_experience_20261026/analytical_costs.json) | Reproducible payload/projection arithmetic, not device measurements |

## What changed the recommendation

The new local causal-refresh diagnostic improves exact recall from 36/240 to 110/240, but its decision remains **rejected**. Its separate restart attempt reports differing fresh-process outputs. A native explicit-record bank therefore deserves a controlled comparison before further compression into shared fast weights. This does not authorize bypassing existing execution gates.

The research also distinguishes three useful mechanisms that should not be conflated: grounded spatial references, compact state for streaming reasoning, and learning from later verified outcomes. Combining them is a plausible Spark design, not an already demonstrated capability of this repository or of one cited paper.

Keep revision 3's [frozen text](../../../reports/sparkx_av_experience_20261026/architecture_v3_snapshot.txt) and dated reports as historical evidence. Revision 4 supersedes their design recommendations where explicitly stated; it does not rewrite their measurements.
