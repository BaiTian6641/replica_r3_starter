# Replica R3 architecture atlas

Eight proposed architecture diagrams are supplied as a vector PDF, individual SVGs and PNG previews. Green identifies shipped primitives exercised by CPU contract tests; blue identifies proposed assembled integration; amber identifies optional research branches. Gray denotes native/frozen state or an update contract; red denotes a data boundary. No diagram asserts that pretrained model quality or longitudinal learning has been demonstrated.

Rebuild with `python scripts/render_diagrams.py` from the project directory. Requires ReportLab; PNG rendering additionally uses `pdftoppm`.

The executable text scaffold implements one native decoder insertion and prompt/source-only memory construction. Online fast-memory integration, VLM ports, independent task evaluation and the complete autonomous operating system remain milestones.
