# Resolution audit — RESULT (2026-06-15)

**Verdict (Codex-endorsed wording):** *Within the tested model-class and objective variants, there is no reliable top-pick headroom; ExtraTrees' small AUC gain does not translate into a dependable selection edge, so the next credible ceiling-breaker is data/features rather than another objective or off-the-shelf model class.*

Scope: "tested model classes/objectives" + "top-pick selection." NOT a claim that all possible model classes are exhausted (Codex caveat — avoided overclaiming).

## Evidence chain (every screen, leak-safe, machinery certified by positive controls)
| screen | result |
|---|---|
| Player random-effects (LOSO) | NULL (K-grid 0-100 IC≤+0.005; certified by in-sample IC +0.24, injection recovery +0.80) |
| Opportunity-count / realized-PA oracle | NULL (proper oracle −0.002; realized-PA is reverse-causation; pregame est_pas already in model) |
| Anchored residual reranker (p-anchored, λ=0 fallback) | NULL (residual-IC ~0 OOF; in-sample control +0.47) |
| KNN / LGB 2nd-stage (master bound, top-10) | NULL (all OOF < model-p; in-sample control IC 0.86) |
| **PA-level RF (full corpus, full slate)** | daily-AUC **−0.0035 (worse)**; top-1 +1.9pp = noise |
| **PA-level ExtraTrees** | daily-AUC +0.0042 (sig but tiny); decision-region switched-pick CI [−2.1,+7.5] crosses 0; top-3 identical → no decision edge |
| **Objective: LGBMRanker (day-grouped)** | daily-AUC **−0.0028 (worse)** |
| **Objective: top-of-slate-weighted GBM** | daily-AUC +0.0003 (n.s.) |
| **ET+GBM ensemble vs single-LGB** | +2.8pp top-1 — BUT artifact of single-vs-blend |
| **ET added to the PRODUCTION blend** | **−2.0pp (CI [−3.95, 0.00]) → ET adds nothing; production's 12-model blend already captures ensembling** |
| **Matchup embeddings (batter×starter FM, offset-residual)** | NULL — see `2026-06-15-phase0g-embedding-RESULT.md`. dot−bias top-1 −0.024, decile −0.015 (sig worse); power CERTIFIED by a PA-level diluted-injection control (+0.10–0.12 recovered). Scoped: no decision-useful **low-rank identity interaction** beyond an already matchup-aware offset (`platoon_hr` + `batter_pitcher_shrunk_hr`); does not refute platoon/arsenal/non-low-rank effects. |

Plus prior diagnosis: top-40% realized-hit region flat; Spearman(pred,realized)=0.13; field at the same wall (~P@100 89% best public).

## Honest read
The production model (12-model GBM blend) already extracts the generalizable signal in the available features; no alternative class (RF/ET/KNN), no alternative objective (ranking/weighting), and no ensemble adds a dependable top-pick edge. The binding constraint is **data/feature resolution** — single-game hit/no-hit for ~85-90% batters is near-Bernoulli, and same-looking 0.86 days are genuinely hard to separate. The only credible ceiling-breakers left are NEW DATA modalities (late markets/lineups/injury-news — mostly unavailable pregame or already-rejected) or the deep, season-end-blocked Gate-B scale reconciliation — none a now-win.

## Method caveats
GBM_pa is a single-LGB proxy (production = 12-blend); RF/ET untuned (n_est 150 vs 400); KNN curse-of-dim discounted. The ET-vs-production-blend test is the cleanest (uses the real production p) and is the decisive null. Scripts: `scripts/resolution_audit/phase0*.py`. Codex reviews: `/tmp/codex-{phase0-review,masterbound,challenger,final-signoff}-out.txt`.
