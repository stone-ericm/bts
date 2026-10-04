# W3 tracker inventory (Explore agent, 2026-10-04, repo-only; verify citations before the memo)

The tracker was last updated 5/10 (last commit 013389e, 5/09). No SOTA target has been cleared for production. The only live policy change is the 9/03 tail objective (an owner requirement; P-05 validated the mechanism only). Several decisive results are on unmerged local branches: resolution-audit (6/15, db28250 / 339cdb5), kcontact-screen (6/15, 1e93f44), swing-escalation (6/14).

| # | Area | Impl | Disposition | Basis | 2026 outcomes / row | Reopen trigger |
|---|---|---|---|---|---|---|
| 1 | MDP: DR-MDP/CVaR/dist DP | built-and-measured | DR-MDP negative; CVaR retired; dist DP not built; tail deployed (mechanism validated) | actual-PA 5/06 (single seed); est-PA 5/24+; DD guardrail seed 42 | X-04/05/07/08/16; May Gate A/B no row | T:180; Gate B :162-163; P-06 |
| 2 | Calibration | built (isotonic, pre-tracker, off) | inconclusive leaning negative (Gate A n=158 < its own n≥200 floor) | live | 5/23 read no row; DD legs X-05 | n≥200 + Brier CI excludes 0 (gate.md:70-73); P-01 recipe 114+85=199 through 9/13 (agent's arithmetic) |
| 3 | TreeSHAP/ALE | unstarted | parked | — | no | none |
| 4 | e-values | unstarted | parked (F10 used Bonferroni checkpoints instead) | — | X-07 indirectly | none |
| 5 | nested CV + lockbox | built-and-measured (Phase D only) | parked | actual-PA (inferred) | no | T:222 |
| 6 | drift/BOCPD | unstarted; alert-only monitors built | parked; park drag negative | live | X-02; monitors X-01 | none |
| 7 | multiple testing | BH/BY built-and-measured; e-BH unstarted | negative (0 survivors); e-BH parked; #87 never run | live + legacy | 5/05, 5/10 no row | T:244 |
| 8 | streaming calibration | unstarted | parked | — | no | T:254 |
| 9 | sequence/GNN features | unstarted; nearby screens built | parked; nearby negative/inconclusive (embedding NULL branch; swing gate fail; kcontact powered NULL branch; team record underpowered; P-03 no-promote) | mixed | X-02, X-06 | P-03; W4 rank 5 |
| 10 | stacking | pooled policy built-and-measured; stacking unstarted | pooled negative (Phase D −0.063, 0/100 seeds); stacking parked | actual-PA (inferred) | no | T:283 |
| 11 | conformal validation | built-and-measured | negative NO_PRODUCTION_DEPLOY | actual-PA single seed | no | T:294 |
| 12 | proper scoring / realized picks | built-and-measured | inconclusive | live | X-05, X-09, X-19; 5/10 no row | T:309 |
| 13 | OPE | built-and-measured (v1) | negative for 8.17% headline (est-PA ≈ 0.01%) | actual-PA 24 seeds | no | T:322 |
| 14 | rare-event MC | built-and-measured (CE-IS v1) | negative headline; iid undermined by 7/13 run suppression | actual-PA | no | T:335 |
| 15 | dependence | built-and-measured (+8/30 pair lift 0.9995) | negative (no exploitable dependence); 7/13 temporal suppression unmodelled | actual-PA | 7/13 live side-check no row | T:354 |
| 16 | decision-aware learning | built (frozen 5004b1c8; logging 5/09–9/18) | inconclusive (113 < 120; never read; E4_fresh_target_inconclusive) | live | X-11 | prereg :383-384 (2027 continuation) |
| 17 | model-class | built-and-measured (legacy + unmerged resolution audit) | negative; parked ("no reliable top-pick headroom"; actual-PA; untuned RF/ET) | actual-PA | no | T:379; W4 rank 6 |

Gaps: no swing-escalation result doc; no realized-picks rerun after 5/10; no #5 split use after Phase D; X-20/X-21 referenced but not in the register; May reads with no register row: 5/10 realized-picks refresh, 5/23 Gate A/B, 5/24 production-metric read, 7/13 repeat-batter live side-check (a W1.6 corrections item).
