# Integrated architecture component diagrams

28 rendered diagrams for the [current research design](../../docs/architecture/README.md). These show proposed components and execution order, not measured learned behavior. The original eight-page atlas remains unchanged.

The canonical editable sources are the Mermaid blocks in the design chapters. Standalone `.mmd` files and SVGs are extracted/rendered from those blocks. Eight PNGs provide convenient previews; SVGs preserve readable text when zoomed.

| Diagram | Design section | Files |
|---|---|---|
| 01_01 | [1. The decision this design makes](../../docs/architecture/01_System_Blueprint.md) | [SVG](01_01.svg) · [Mermaid](01_01.mmd) · [PNG](01_01.png) |
| 01_02 | [4. Acyclic neural execution and memory access](../../docs/architecture/01_System_Blueprint.md) | [SVG](01_02.svg) · [Mermaid](01_02.mmd) · [PNG](01_02.png) |
| 01_03 | [5. Request and cache ownership](../../docs/architecture/01_System_Blueprint.md) | [SVG](01_03.svg) · [Mermaid](01_03.mmd) |
| 01_04 | [6. Compatibility, publication and storage](../../docs/architecture/01_System_Blueprint.md) | [SVG](01_04.svg) · [Mermaid](01_04.mmd) |
| 02_01 | [2. The backbone port is a complete-block boundary](../../docs/architecture/02_Backbone_Integration.md) | [SVG](02_01.svg) · [Mermaid](02_01.mmd) · [PNG](02_01.png) |
| 02_02 | [3. MiniCPM5 port map and executable scheduling](../../docs/architecture/02_Backbone_Integration.md) | [SVG](02_02.svg) · [Mermaid](02_02.mmd) |
| 02_03 | [4. Native memory encoder and tensor access contract](../../docs/architecture/02_Backbone_Integration.md) | [SVG](02_03.svg) · [Mermaid](02_03.mmd) |
| 02_04 | [5. Request state, cache and gradient ownership](../../docs/architecture/02_Backbone_Integration.md) | [SVG](02_04.svg) · [Mermaid](02_04.mmd) |
| 02_05 | [6. MiniCPM-V 4.7 needs its own adapter](../../docs/architecture/02_Backbone_Integration.md) | [SVG](02_05.svg) · [Mermaid](02_05.mmd) · [PNG](02_05.png) |
| 03_01 | [1. Research findings and the design decision](../../docs/architecture/03_Memory_and_Runtime_Compilation.md) | [SVG](03_01.svg) · [Mermaid](03_01.mmd) · [PNG](03_01.png) |
| 03_02 | [3. Chronological evidence and eligibility](../../docs/architecture/03_Memory_and_Runtime_Compilation.md) | [SVG](03_02.svg) · [Mermaid](03_02.mmd) |
| 03_03 | [4. Persistent record latents and memory access ports](../../docs/architecture/03_Memory_and_Runtime_Compilation.md) | [SVG](03_03.svg) · [Mermaid](03_03.mmd) |
| 03_04 | [5. Private adaptive state and shared-depth commits](../../docs/architecture/03_Memory_and_Runtime_Compilation.md) | [SVG](03_04.svg) · [Mermaid](03_04.mmd) |
| 03_05 | [6. Persistent branch memory image](../../docs/architecture/03_Memory_and_Runtime_Compilation.md) | [SVG](03_05.svg) · [Mermaid](03_05.mmd) |
| 03_06 | [Static lookup branch](../../docs/architecture/03_Memory_and_Runtime_Compilation.md) | [SVG](03_06.svg) · [Mermaid](03_06.mmd) |
| 03_07 | [Procedure and skill memory](../../docs/architecture/03_Memory_and_Runtime_Compilation.md) | [SVG](03_07.svg) · [Mermaid](03_07.mmd) |
| 03_08 | [Identity and task memory](../../docs/architecture/03_Memory_and_Runtime_Compilation.md) | [SVG](03_08.svg) · [Mermaid](03_08.mmd) |
| 03_09 | [8. Runtime compilation and re-compilation](../../docs/architecture/03_Memory_and_Runtime_Compilation.md) | [SVG](03_09.svg) · [Mermaid](03_09.mmd) · [PNG](03_09.png) |
| 03_10 | [9. Deletion, invalidation and migration](../../docs/architecture/03_Memory_and_Runtime_Compilation.md) | [SVG](03_10.svg) · [Mermaid](03_10.mmd) |
| 04_01 | [1. Recovered TAIS lineage and decisions](../../docs/architecture/04_Semi_Cognition_Learning_and_Replica.md) | [SVG](04_01.svg) · [Mermaid](04_01.mmd) |
| 04_02 | [3. Connected operating architecture](../../docs/architecture/04_Semi_Cognition_Learning_and_Replica.md) | [SVG](04_02.svg) · [Mermaid](04_02.mmd) |
| 04_03 | [4. Components C08/C09 — exact task state and latent cognitive workspace](../../docs/architecture/04_Semi_Cognition_Learning_and_Replica.md) | [SVG](04_03.svg) · [Mermaid](04_03.mmd) · [PNG](04_03.png) |
| 04_04 | [5. Components C09/C10 — planning, simulation and action control](../../docs/architecture/04_Semi_Cognition_Learning_and_Replica.md) | [SVG](04_04.svg) · [Mermaid](04_04.mmd) |
| 04_05 | [6. Component C11 — capability model and self-evaluation](../../docs/architecture/04_Semi_Cognition_Learning_and_Replica.md) | [SVG](04_05.svg) · [Mermaid](04_05.mmd) |
| 04_06 | [7. Component C13 — experience diagnosis and learned learning director](../../docs/architecture/04_Semi_Cognition_Learning_and_Replica.md) | [SVG](04_06.svg) · [Mermaid](04_06.mmd) |
| 04_07 | [8. Components C12/C14/C15 — runtime compilation and candidate learning](../../docs/architecture/04_Semi_Cognition_Learning_and_Replica.md) | [SVG](04_07.svg) · [Mermaid](04_07.mmd) |
| 04_08 | [9. Component C16 — Replica source, identity and branch development](../../docs/architecture/04_Semi_Cognition_Learning_and_Replica.md) | [SVG](04_08.svg) · [Mermaid](04_08.mmd) · [PNG](04_08.png) |
| 05_01 | [1. What changes in the work plan](../../docs/architecture/05_Implementation_Sequence.md) | [SVG](05_01.svg) · [Mermaid](05_01.mmd) |

## Component coverage

Every component has a mechanism, integration/use description and diagram. Related components share a diagram where that makes the connection explicit.

| Component | Diagram(s) |
|---|---|
| C01 — native backbone adapter | [02_01](02_01.svg) · [02_05](02_05.svg) |
| C02 — sense and residual ports | [02_01](02_01.svg) · [02_02](02_02.svg) |
| C03 — evidence and episode store | [03_02](03_02.svg) |
| C04 — record encoder and latent store | [02_03](02_03.svg) · [03_03](03_03.svg) |
| C05 — retrieval router and pager | [03_02](03_02.svg) · [03_03](03_03.svg) |
| C06 — fast neural memory | [03_04](03_04.svg) · [03_05](03_05.svg) |
| C07 — conditional lookup memory | [03_06](03_06.svg) |
| C08 — task graph and workspace | [04_03](04_03.svg) · [03_08](03_08.svg) |
| C09 — shared cognitive core | [04_02](04_02.svg) · [04_03](04_03.svg) · [04_04](04_04.svg) |
| C10 — action executor and outcome verifier | [04_04](04_04.svg) |
| C11 — capability and value model | [04_05](04_05.svg) |
| C12 — runtime compiler and skill catalog | [03_07](03_07.svg) · [03_09](03_09.svg) |
| C13 — learning director | [04_06](04_06.svg) |
| C14 — candidate trainer and evaluator | [04_07](04_07.svg) |
| C15 — manifest checkpoint and publication | [01_04](01_04.svg) · [04_07](04_07.svg) |
| C16 — replica identity service | [04_08](04_08.svg) · [03_08](03_08.svg) |
| C17 — request scheduler and cache manager | [01_03](01_03.svg) · [02_04](02_04.svg) |
| C18 — multimodal ingress and writer | [02_05](02_05.svg) |

## Render and verification

The [manifest](manifest.json) records source, SVG and preview hashes. Mermaid **11.12.0** parsed and rendered all 28 diagrams using an isolated headless browser. Page network requests were blocked. The checked renderer bundle has SHA-256 `07e37dfa97b337ccc85365d57eddf99b9706f09db3b59b260d0333b23b343c4b`. Obtain that pinned bundle separately; the script does not install dependencies or download anything.

```sh
node diagrams/architecture_v1/render.mjs /path/to/mermaid-11.12.0.min.js /path/to/playwright /path/to/chromium
```

The optional Chromium argument selects an existing executable. Otherwise Playwright uses its configured binary. Rendering uses a fresh temporary browser profile; it does not access a personal browser session. A JavaScript runtime, Playwright and a compatible browser are rendering dependencies only, not model-training dependencies.

See [render.mjs](render.mjs), [design inventory](../../configs/architecture_design_v1.json) and [review record](../../reports/architecture_research_20261023/review.json).
