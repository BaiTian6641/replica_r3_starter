# S1.3 public-source metadata review — 22 October 2026

**Both sources remain quarantined.** This review resolves immutable upstream
revisions and inspects metadata, dataset documentation and official code. It
does not admit data, implement an adapter or complete S1.3. No corpus payload,
model checkpoint or complete history/query pair was downloaded. The dataset
registry and existing local MiniCPM5 original are unchanged.

The HF landing page automatically displayed benchmark query/answer snippets
for persona 521. Those inspected snippets are exposed review material, never
a sealed audit; no linked history payload was retrieved.

Checked **23 October 2026, 00:17–00:21 UTC** (22 October in America/Chicago).
The [machine-readable review](../../reports/public_source_metadata_review_20261022.json)
records retrieval URLs, actual metadata/document response digests, pinned
official-code revisions and remaining admission steps. Advertised payload
hashes below are upstream metadata, **not locally verified payload hashes**.

## Immutable candidates and inventory

| Candidate | Resolved HF revision | Declared card license | Advertised repository inventory |
| --- | --- | --- | ---: |
| PersonaMem-v2 | `ed956dea41521fc4499acbc63f966e0fd3c053ba` | CC-BY-4.0 | 9,001 files; 4,576,126,064 bytes |
| Cleaned LongMemEval | `98d7416c24c778c2fee6e6f3006e7a073259d48f` | MIT | 5 files; 3,029,875,295 bytes |

These are candidates resolved from the official APIs, not a mutable `main`
acquisition instruction. The report hashes all paginated inventory responses
and the canonical file-metadata listing; it does not retain corpus bytes.
[Pinned PersonaMem tree](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/tree/ed956dea41521fc4499acbc63f966e0fd3c053ba),
[pinned LongMemEval tree](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f).

| Selected payload path | Advertised bytes |
| --- | ---: |
| PersonaMem `benchmark/text/train.csv` | 157,116,318 |
| PersonaMem `benchmark/text/val.csv` | 17,474,290 |
| PersonaMem `benchmark/text/benchmark.csv` | 42,426,457 |
| LongMemEval `longmemeval_s_cleaned.json` | 277,383,467 |
| LongMemEval `longmemeval_m_cleaned.json` | 2,737,100,077 |
| LongMemEval `longmemeval_oracle.json` | 15,388,478 |

For cleaned S, the advertised LFS SHA256 is
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
The report includes the six PersonaMem CSV metadata entries and all five
LongMemEval entries, including their advertised LFS hashes where present.

## PersonaMem: source scope and unresolved identity mapping

The pinned card declares synthetic GPT-5-generated material, official
text/multimodal train, validation and benchmark splits, and persona-disjoint
splits. It describes 18,500 text training queries, 2,600 validation queries and
5,000 benchmark queries. These are upstream claims; this review did not count
CSV rows or verify identity intersections. Raw persona files also contain
targets and metadata, so they cannot be copied wholesale into model memory.
[Pinned card](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/blob/ed956dea41521fc4499acbc63f966e0fd3c053ba/README.md).

Actual filename metadata contains **999 raw persona IDs**, with ID **558**
absent from the advertised 0–999 range. Each of the four history directories
contains **1,998 files: two filenames for each of those 999 IDs**. This differs
from the card's simplified 1,000-file description. Filename dates do not prove
which version a CSV references, whether content is duplicated, or why an ID is
absent. Resolve exact row-to-history references before choosing files; do not
merge all versions or infer a broken target from filenames alone.
[Pinned inventory](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/tree/ed956dea41521fc4499acbc63f966e0fd3c053ba/data).

There is a concrete inherited-material question: the card attributes short
personas to PersonaHub, while PersonaHub's resolved revision
`16777e34bf5cb758b925cae5d84e868ee6c2100c` declares **CC-BY-NC-SA-4.0**. That
differs from PersonaMem's card label. This review does not adjudicate derivative
rights; it leaves intended use, lineage and required terms unresolved. The
card also identifies public math/coding filler and multimodal images, which
need component-specific review before inclusion. A text-only 32k candidate
subset could reduce scope, but does not itself resolve inherited persona terms.
[PersonaHub card](https://huggingface.co/datasets/proj-persona/PersonaHub/blob/16777e34bf5cb758b925cae5d84e868ee6c2100c/README.md).

The column specification separates history links and `user_query` from
`correct_answer`, preference annotations, previous preferences, related
snippets and evidence-distance fields. A generative adapter should allowlist
actual history turns plus the query, and keep target/oracle annotations out of
support and workspace. The official inference code joins a referenced history
to the query and has a separate multiple-choice condition; MCQ candidate
answers must not silently become generative answer hints. Cross-modal persona
overlap, chronology/update semantics and complete pair correctness remain
unverified.
[Pinned columns](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/blob/ed956dea41521fc4499acbc63f966e0fd3c053ba/column_descriptions.md),
[pinned official inference](https://github.com/bowen-upenn/PersonaMem-v2/blob/d29d91d016add354e459dfeb0d24af08bc402e2a/inference.py).

## LongMemEval: evaluation-only boundary

The cleaned card states that this release replaces the original and removes
noisy history sessions. Its MIT label is dataset-card metadata; the official
code repository separately carries an MIT software license. This review has
not audited the cleaning spreadsheet, all underlying history-source terms or
payload correctness. LongMemEval remains an **external evaluation candidate**,
never a source of training/teacher/curriculum targets. S, M and oracle variants
must share one question/source exposure lineage rather than being treated as
independent pools.
[Pinned cleaned card](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/blob/98d7416c24c778c2fee6e6f3006e7a073259d48f/README.md),
[official license](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/LICENSE).

Pinned official code supplies a bounded schema review without reading corpus
records: `question`, `question_date`, `haystack_dates`,
`haystack_session_ids` and nested `haystack_sessions` support the input view.
Keep `answer`, `answer_session_ids` and turn-level `has_answer` evaluator-only;
allowlist turn `role`/`content`. The generation code removes `has_answer`, and
the evaluator separately joins gold answers with model hypotheses for judge
prompts. No judge/API execution occurred. Oracle evidence selection must be a
labelled evaluator condition, not ordinary retrieval access.
[Official generation](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/generation/run_generation.py),
[official evaluation](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py).

## Remaining admission steps

1. Resolve intended-use, attribution and inherited component terms; document
   an exact eligible subset rather than promoting the card's license label to
   blanket permission.
2. Specify a minimal pinned payload inspection with allowlisted files, byte
   cap and reuse/retention rules. This review authorizes no corpus clone.
3. Inspect complete linked history/query pairs; audit persona intersections
   across all six PersonaMem splits, duplicated versions, missing ID 558 and
   shared source/semantic lineage. Keep all descendants in their parent role.
4. Define source-specific event/observation chronology and purpose mappings;
   separate natural evidence from gold/evidence-location annotations. Preserve
   unknowns and corrections without inventing personal testimony.
5. Verify acquired payload hashes against the pins; freeze grouped release,
   independent evaluation and exposure manifests. Previously inspected
   benchmark cases cannot be relabelled sealed final audit.
6. Implement and independently review adapters, then measure actual tokenizer,
   retrieval and history-ingestion costs. Long histories require declared
   retrieval/chunk budgets; the Mac's bounded actual-256/512 qualification does
   not permit silent truncation or full-history fit claims.

These steps advance S1.3 admission toward S1.5. Metadata review and the separate
synthetic S1.2/S1.4 preparation do not yet unblock S2.4 training.
