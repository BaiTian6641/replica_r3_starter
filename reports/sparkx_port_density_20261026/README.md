# Port-density research evidence

**26 October 2026.** This is a research/documentation package, with no new native run, model implementation or capability promotion.

- [Research, recommendation and validation plan](../../docs/research/sparkx_port_density_20261026/Port_Density_Research_and_Plan.md)
- [Supplementary study configuration](../../configs/sparkx_2_6_4b_port_density_study_v1.json)
- [Source manifest and reading scopes](evidence.json)
- [Repository observation](repository_snapshot.json)
- [Exact canonical revision 4 before this update](architecture_v4_snapshot.txt)
- [Analytical cost ledger](analytical_costs.json)
- [Documentation verification](documentation_validation.json)

The review examined ten relevant papers, selected model implementation paths, and the repository's port construction/forward contracts. The manifest distinguishes detailed selected-section reading from configuration metadata. Published results are not reproduced. Public source files were downloaded to temporary scratch storage and read, not executed or copied into runtime code.

Recompute analytical counts with `python3 reports/sparkx_port_density_20261026/estimate_costs.py`. Verify document links, state/config contracts, observed source hashes and arithmetic with `python3 reports/sparkx_port_density_20261026/verify_research.py`. Both scripts use only the standard library; no device, model or network access is involved. Temporary-source checks are reported unavailable if those copies have expired.

Revision 4's baseline configuration is unchanged. The new study deliberately varies reader density independently of writer updates, bank capacity, sensing layers and online-adapter targets. The historical source observations remain dated snapshots even when concurrent work changes the checkout.
