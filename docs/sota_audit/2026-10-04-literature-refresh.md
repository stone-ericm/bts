# W3 literature / SOTA refresh (season wrap, 2026-10-04)

**Plan:** `docs/superpowers/plans/2026-09-14-season-wrap-plan.md` §W3.
**Status:** draft for Codex review (analysis deliverable: at most two rounds, then freeze with the limits stated).
**Method:**
- Each of the 17 tracker areas gets an **implementation state** and an **evidence disposition**, recorded separately, together with its basis, 2026 exposure and a reopening trigger.
- Each nominated paper and each search question maps to a W4 experiment, not applicable (reason) or closed (reason).

**Sources:**
- Every external claim was checked against a primary source by a research pass on 2026-10-03, with labels kept in the working notes. BeyondArena's identity and headline were re-checked by the author on 2026-10-04.
- The prize terms come from the 2026 Official Rules, fetched and quoted on 2026-10-04.
- The repo side comes from a read-only inventory of the docs, scripts and history through 2026-10-03.
- Claims not verified are marked so.

## 1. Summary
- **No tracker area has cleared a production gate.** The tracker itself was last updated on 2026-05-10. Everything since is recorded in §2.
- **Only one policy change went live this season:** the 9/03 tail objective, an owner requirement whose mechanism P-05 validated. No model change was promoted.
  - Park drag, the context shadow, swing features, team record, kcontact and the matchup embedding were all null, inconclusive or underpowered.
- **The literature changes three things for W4:**
  1. **Rank 1 (MLB forecast → one combination rule):**
     - fix the rule in advance as a beta-transformed linear pool or a one-parameter logit pool, never a linear average (Ranjan & Gneiting 2010; Smith & Wallis 2009);
     - gate it on a forecast-encompassing-style residual-information test.
  2. **Rank 4a (calibration map):** if selected, register online or windowed Platt scaling, with an intercept-only variant, as the single map (Gupta & Ramdas, ICML 2023). Theory-optimal recalibration needs orders of magnitude more rounds than about 180 picks a season.
  3. **Rank 6 (model-class challenger):** the one candidate is **TabM** at its frozen default (ICLR 2025), not RealMLP.
     - On BeyondArena (Purucker et al. 2026), the benchmark that includes temporal splits, TabM's default leads RealMLP's and default LightGBM's on the all-dataset leaderboard.
     - Its tuned and ensembled variants lead on temporal data, but the plan forbids sweeps.
     - The gate must be ours: within-slate top-1 and top-bin proper scores. Every benchmark scores ROC AUC.
- **The prize terms matter for D1.** The $10,000 Top Streak Prize goes to the eligible entrant with the **highest** streak of 20 or more, as of the end of the entry period, split equally on ties; it need not be active (Official Rules §7–8).
  - So the late-season objective that pays is relative: P(our best is at least the field's best and at least 20). It is not E[our season best], which is what the tail policy maximises.
  - The 2026 best was 18, below the floor.
  - This is an input to the decision memo, not a recommendation.

## 2. Tracker areas: state and disposition as of 2026-10-03
Basis: actual-PA = the hindsight backtest surface (rank-1 hit about 0.865); est-PA = the serving-realistic surface; live = the at-cutoff production stream. "Seed 42" marks a result measured at one seed only. The CLAUDE.md warning says seed 42 was an outlier.

| # | Area | Implementation | Disposition | Basis | 2026 outcomes consumed | Reopening trigger |
|---|---|---|---|---|---|---|
| 1 | MDP: DR-MDP / CVaR / distributional DP | built-and-measured (DR-MDP screen, Gate B, est-PA A/B, tail DP) | DR-MDP **negative**; CVaR **retired**; distributional DP unbuilt; tail objective deployed (mechanism validated only) | actual-PA (5/06, one seed); est-PA (5/24 onward); DD guardrail seed 42 | X-04, X-05, X-07, X-08, X-16 (the May Gate A/B reads have no row; see §6) | a regenerated 24-seed surface or a materially different bin manifold clears `scripts/dr_mdp_gap_measure.py` (tracker); P-06 follow-up only on the decision_v2 stream |
| 2 | Calibration (Beta / Venn-Abers / spline) | built (isotonic only, off by default) | **inconclusive**, leaning negative: Gate A n=158 is below its own n≥200 floor | live | the 5/23 read has no row; DD legs X-05 | n ≥ 200 and a Brier CI excluding 0; W4 rank 4a |
| 3 | TreeSHAP / ALE | unstarted | **parked** | — | no | none stated |
| 4 | e-values / e-processes | unstarted (F10 used Bonferroni checkpoints instead) | **parked** | — | X-07 indirectly | none stated |
| 5 | Nested purged CV + lockbox | built-and-measured (Phase D only) | **parked** (not adopted after Phase D) | actual-PA (inferred) | no | tracker |
| 6 | Drift / BOCPD | BOCPD unstarted; alert-only monitors built | **parked**; the park-drag screen **negative** | live | X-02; monitors under X-01 | none stated |
| 7 | Multiple testing (e-BH / online FDR) | BH/BY built-and-measured; e-BH unstarted | BH/BY **negative** (0 survivors); e-BH **parked**; #87 never run (W2.4) | live and legacy | the 5/05 and 5/10 reads have no row | tracker |
| 8 | Streaming calibration (ACI / RCPS) | unstarted | **parked** | — | no | when conformal unblocks |
| 9 | Sequence / transformer / GNN features | unstarted; nearby screens built | **parked**; nearby work negative or inconclusive (embedding NULL and kcontact powered NULL, both on unmerged branches; swing gate failed; team record underpowered; P-03 no-promote) | mixed | X-02, X-06 | P-03; W4 rank 5 rules |
| 10 | Predictive stacking | pooled policy built-and-measured; stacking unstarted | pooled **negative** (Phase D −0.063, 0/100 seeds); stacking **parked** | actual-PA (inferred) | no | tracker |
| 11 | Binary-y conformal | built-and-measured | **negative** (`NO_PRODUCTION_DEPLOY`) | actual-PA, one seed | no | tracker |
| 12 | Proper scoring / realized picks | built-and-measured | **inconclusive** | live | X-05, X-09, X-19 (the 5/10 refresh has no row) | tracker; W1.2 (X-21) is the new read |
| 13 | Off-policy evaluation | built-and-measured (v1) | **negative** for the 8.17% headline (est-PA ≈ 0.01%) | actual-PA, 24 seeds | no | realized replay is the trusted evaluator (7/13) |
| 14 | Rare-event Monte Carlo | built-and-measured (CE-IS v1) | **negative** headline; the iid assumption is undermined by 7/13 run suppression | actual-PA | no | tracker |
| 15 | PA / cross-game dependence | built-and-measured (plus the 8/30 pair lift, 0.9995) | **negative** (no exploitable dependence); the 7/13 temporal run suppression is unmodelled | actual-PA | the 7/13 live side-check has no row | tracker |
| 16 | Decision-aware learning | built (frozen `5004b1c8`; logged 5/09–9/18) | **inconclusive** (113 < 120 eligible; never read; `E4_fresh_target_inconclusive`) | live | X-11 | the prereg's 2027 continuation |
| 17 | Model-class challenge | built-and-measured (legacy, plus the unmerged resolution audit) | **negative**; **parked** ("no reliable top-pick headroom"; actual-PA; untuned RF/ET) | actual-PA | no | W4 rank 6 (TabM, §3) |

**Recorded nowhere until now:** decisive results that exist only on unmerged local branches: `resolution-audit` (6/15), `kcontact-screen` (6/15) and `swing-escalation` (6/14, no result doc). Merging or archiving their result docs is a W1.6 item.

## 3. Nominated items
| Item | Disposition | Reason |
|---|---|---|
| **TabM** (Gorishniy, Kotelnikov, Babenko; ICLR 2025; arXiv 2410.24210) | **W4 rank 6 candidate**: frozen default (k=32), the single model-class challenger | <ul><li>On BeyondArena's all-dataset leaderboard (ROC-AUC Elo), TabM default is 1107, RealMLP default 1056, LightGBM default 991 and LightGBM tuned 1149.</li><li>TabReD finds MLP-like models and GBDTs best on time splits.</li><li>Our LightGBM runs on near-default parameters.</li><li>Not verified: defaults on BeyondArena's temporal subset alone (in figures only).</li><li>Daily CPU retraining on millions of rows is a real cost (reasoning only).</li></ul> |
| **RealMLP / "Better by default"** (Holzmüller, Grinsztajn, Steinwart; NeurIPS 2024) | **not selected** (the plan picks one); its LGBM-TD defaults are **closed** by "no LightGBM re-tuning" | its temporal lead needs tuning and ensembling; its default lags TabM's |
| **TabPFN v2** (Hollmann et al.; Nature 2025) and 2.5/3/3.5 | **not applicable** at PA level; the game-level residual idea stays **parked** until W1.2 measures headroom | <ul><li>v2's stated scope is ≤10K samples; 3/3.5 go to ≤1M rows, against millions of PA rows.</li><li>Foundation models lag on temporal and large data (BeyondArena's abstract).</li><li>Weights after v2 are non-commercial, and the 3.5 licence bars "production" use: an owner question before any production use.</li></ul> |
| **TabArena** (Erickson et al.; NeurIPS 2025) | **closed** as evidence (methodology reference only) | IID-only by design, random CV, 500–250K training rows, ROC AUC; BeyondArena is the temporal extension |
| **Online conformal, decaying steps** (Angelopoulos, Barber, Bates; ICML 2024) | **not applicable** (reopen if W4 rank 3 builds a PA-count model) | binary target (uninformative sets), coverage ≠ calibration, and the decaying step adapts more slowly under drift |

## 4. Search questions
| Question | Finding | Disposition |
|---|---|---|
| Public BTS models | <ul><li>Alceo & Henriques: 85% / 81% top-100 pick ratio.</li><li>McKenna 2015: a BTS MDP with skip, iid p, no double-down.</li><li>Pinto's log5 lists; Nickell's 2025 scorecard.</li><li>The README cites "Garnett (2026)" (P@100 85%, P@500 77%), but the article was blocked to automated fetches; a search snippet gives 84% / 81%.</li></ul> | **closed** as a benchmark source; **README hygiene**: the Garnett citation needs a human read (§6) |
| MLB `probabilityStarter` / "% chance to hit" | <ul><li>The only official text is the FAQ: "hit probability estimates from our unique prediction model".</li><li>The field name appears nowhere public.</li><li>The rules define a Hit as needing ≥1 official AB or sacrifice fly, with a Pass otherwise.</li></ul> | **feeds W4 rank 1**: target semantics must be established empirically (W2.3 gate ii) |
| Statcast bat tracking | <ul><li>Miss distance launched 2026-06-09, with data from the second half of 2023, bunts excluded.</li><li>Swing path and attack angle are documented.</li><li>No official availability-lag statement exists.</li><li>Swing metrics are context-confounded (Powers & Yurko 2025).</li></ul> | **supports W4 rank 5**: the registration fixes the swing-conditioned denominator and a **measured** as-of lag |
| Calibration under drift | Online Platt scaling (two parameters, no tuning) fits our sample size | **inside W4 rank 4a** (as §1) |
| MDPs with resets / milestones | State augmentation is the known solution (Xu & Mannor 2011); nothing addresses BTS run structure | **closed** (no new method); the open problem stays the 7/13 run-structure misspecification |
| Forecast combination with a public forecast | <ul><li>A linear pool of calibrated forecasts is uncalibrated; the beta-transformed pool fixes this (Ranjan & Gneiting).</li><li>Fixed simple rules beat estimated weights in small samples (Smith & Wallis).</li></ul> | **W4 rank 1 design** (as §1) |
| 2026 ball drag | MLB confirmed a 2025 drag increase and a sharp June 2026 drop (statements via secondary reports; the primary articles are paywalled) | **closed** for alpha (our 2026 screen was null); keep as regime observability; any variant validates in 2027 (D3) |
| Streak contests and consensus | <ul><li>Contrarian strategies need relative payoffs, and the grand prize is absolute.</li><li>Crowds over-pick favourites.</li></ul> | **closed** as a method; supports #87's framing (a feature hypothesis, never copying) |

## 5. What this means for W4 (reasoning only; selection belongs to the decision memo)
- **Rank 1:** one predeclared combination rule (beta-transformed or logit pool) plus a residual-information test on shared as-of candidates, validated on untouched dates (2027).
- **Rank 4a:** online or windowed Platt, intercept-only variant, gated on held-out proper scores.
- **Rank 5:** miss distance, with a measured lag and a fixed denominator.
- **Rank 6:** TabM at its default only; gate on within-slate top-1 and top-bin proper scores at the identical serving contract.
- **Not candidates:** TabPFN, TabArena-driven selection, online conformal, consensus copying.

## 6. Open items for Eric
- **The external benchmark the README calls "current SOTA"** is a Medium article that blocks automated fetches: https://medium.com/learning-data/chasing-5-6-million-with-machine-learning-my-approach-to-mlbs-impossible-hitting-streak-7f888e1b9d00. Its numbers appear not to match what the README cites.
- **Licence question:** TabPFN weights after v2 are non-commercial; 3.5 bars "production" use.
- **D1 input:** the relative Top Streak Prize (§1).
- **W1.6:** May reads of 2026 outcomes that have no exposure-register row (the 5/10 realized-picks refresh, the 5/23 Gate A/B, the 5/24 production-metric read, the 7/13 repeat-batter live side-check) and the unmerged result branches. These are listed for the corrections pass; whether each actually read 2026 outcomes is checked there.

## 7. Limits
- Benchmarks are aggregates, and the temporal-subset defaults were not extracted.
- Paywalled primaries (The Athletic, Baseball Prospectus) are cited through secondary reports.
- The tracker inventory is repo-derived; the per-area citations, the research notes with evidence labels and the quoted prize terms are in `docs/sota_audit/2026-10-04-literature-refresh-evidence/`.
- No new experiment was run for this memo.
