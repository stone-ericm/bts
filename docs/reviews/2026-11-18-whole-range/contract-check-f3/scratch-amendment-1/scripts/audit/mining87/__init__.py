"""#87 leaderboard mechanism mining, execution adapter (season wrap W2.4).

Governed by the original protocol `docs/sota_audit/2026-05-10-leaderboard-mechanism-mining-prereg.md`, its execution
amendment `docs/sota_audit/2026-10-04-mechanism-mining-amendment.md` and the required code changes of Codex review r1
(`docs/audit/2026-10-04-mechanism-mining-codex-r1.md`). The shared historical scripts are imported only for helpers
that review found correct; everything the review found wrong is re-implemented here.

Modules: `registration` (frozen run identity, X-22 gate, drift checks), `production` (W1.1 ledger → locked slots and
per-slot settlement), `consensus` (public picks → legal consensus pairs and conservative settlement), `surfaces`
(served-slate parsing and witness admission), `units` (production-led unit table and decomposition bins), `inference`
(bootstrap, cell inventory, FDR families, five nomination conditions, streams) and `report` (summaries and the
registered report); `run` is the driver.
"""
