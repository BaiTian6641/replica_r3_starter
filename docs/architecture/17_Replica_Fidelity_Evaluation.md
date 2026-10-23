# C16 fidelity evaluation harness

**Status: fictional scoring harness implemented; authentic personal evaluation
and human source review remain external.**

`replica/fidelity.py` scores three named baselines (`generic`,
`approved_summary`, `archive_retrieval`) over explicitly fictional cases. It
keeps provenance coverage, choice correctness, clarification, unknown
labeling, technical correctness, optional style tags and branch-event counts
separate. It never emits a “percentage replica” or personal-fidelity claim, and
it rejects non-fictional cases.

The harness does not generate model outputs; it scores supplied responses with
executable metrics. Authentic source material, human review and actual baseline
runs remain required before C16 fidelity can be evaluated.
