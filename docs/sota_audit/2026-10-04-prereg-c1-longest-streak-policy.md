# C1 rank 4b registration: a full-season longest-streak policy (D1 Option 2)

**Status:** design rev 1, 2026-10-04, for Codex design review (at most 2 rounds, then freeze).
**Cycle:** C1 (`docs/sota_audit/2026-10-04-c1-cycle-index.md`), approved by Eric 10/04 with 4b added.
**Owner rulings it serves:** D1 = Option 2: accept a specified reduction in the projected chance of 57 for a better expected longest streak. **Gate (Eric):** no 4b change affects picks until he has seen the measured trade table (chance of 57 given up against expected longest streak gained) and approved it under D7.
**Exposure:** row X-31, to be published before any outcome-bearing replay. It reads only the 2021–2025 estimated-PA profiles; no 2026 outcome is read (D3 = RESERVE).

## In plain words
- **The idea.** In 2026 the system played for the chance of 57 all season and switched to "beat our own season best" only once 57 was out of reach (the 9/03 tail policy). The candidate plays for the longest streak from the first day.
- **The test.** We replay real 2021–2025 hit sequences through both policies, fitting on four seasons and testing on the fifth, and measure how much longer the expected best streak gets.
- **What can't be measured.** The chance of 57 can't be measured on those sequences (no trajectory ever reached 57), so it is reported as a model projection with its assumptions stated.
- **Who decides.** Eric sees one table and decides. Nothing changes picks before his approval.

## 1. The objective (ruling R1)
- **Optimized:** the candidate maximizes **our own expected season-best streak**, E[max streak over the season], capped at 57. This is the tail policy's objective (`src/bts/simulate/tail_policy.py`, `solve_emax_season_best`), extended to the whole season.
- **Why not the prize objectives.** The prize objectives (winning the $10,000 Top Streak Prize, or the expected share of it) need a model of the field's best streak and of ties, which we don't have (W3 §1). The frozen decision memo says these can rank policies differently from own E[best]; this registration names own E[best] as its objective.
- **Prize-relevant descriptive columns:** P(reach ≥ 20) (the prize floor) and P(reach ≥ 40) (above 2026's field maximum of 39). Neither is optimized.

## 2. Arms
**The trade table compares** (each arm replays its own streak, season best, saver and days left through the calendar):
- **A0, deployed 2026:** the shipped reach-57 table (`mdp_policy.npz`, sha `66d15471…`) with the 2026 tail routing (tail table `dc5d0c99…` when s + 2d < 57). This is what Eric would be switching away from.
- **A1, matched reach-57 re-solve:** P(57)-optimal, using exactly A2's rates, bins and pairing. A1 − A0 isolates the effect of re-estimating rates and bins; A2 − A1 isolates the objective.
- **A2, the candidate:** the full-season E[best] table.
- **Comparators** (the plan's policy-replay rules): always-single, and legal always-double.

**One candidate.** No second objective or bin scheme is tried in C1 if A2 fails (stopping rule, §7).

## 3. Rates, bins and the fit/test split
- **Profiles:** `data/hetzner_results/mdp_estpa_run`, the serving-realistic estimated-PA profiles: 24 seeds × 2021–2025. Hashed into the manifest at run start.
- **Pairing:** the production double rule, the first lower-ranked candidate in a different game. A day with no partner demotes a double to a single (§4).
- **Bins:** quintiles of rank-1 probability, computed on the fitting seasons and pooled over seeds.
  - **Phase-aware**, as the shipped table is: early bins, plus late bins for the last 30 calendar contest days.
  - A1 and A2 use identical bins and rates.
- **Split:** leave one season out. For each held-out season, rates and bins are fitted on the other four (all 24 seeds), A1 and A2 are solved, and every arm is replayed on the held-out season's 24 profiles. A0 is fixed and never refitted.
- **Primary population:** the 5 held-out seasons, pooled.

## 4. The evaluator: realized-sequence replay with its known shortcuts fixed
- **Base:** `scripts/audit/dd_p_policy_value_sensitivity.py` `replay_vectorized` (the July 7/13 L2 evaluator).
- **Fixes:**
  - **(a) Calendar clock:** days left = the season's remaining contest dates, not 180 minus row position. The old clock dropped 312 of 21,888 rows.
  - **(b) Partnerless double:** demoted to a single, as production does; the old code advanced +2 on rank-1's outcome.
  - **(c) Season best in state:** the policy function receives the running season best m. In replay, m is the running max, which is trusted by construction.
  - **(d) Deployed comparator:** A0 includes the tail routing. The July "deployed" arm used the base table only, which goes all-skip once s + 2d < 57.
- **Parity first.** Before any fixed-evaluator number, the unfixed evaluator must reproduce the July anchors at Δ = 0:
  - deployed 17.73 mean max / 31.7% reach-20;
  - always-single 15.94 / 7.5%;
  - always-double 17.85 / 40.8%.

  The run then reports each fix's effect separately, in the order (a), (b), (c), (d).
- **Stress grid:** the July double-leg haircut Δ ∈ {0, 0.05, 0.10, 0.139}.

## 5. Metrics
- **Primary:**
  - A2 − A0 paired difference in E[season best]: per trajectory, on identical held-out sequences;
  - pooled over the 5 held-out seasons (120 trajectories);
  - at Δ = 0;
  - with its across-profile standard error and each season's difference.
- **Secondary, descriptive, every arm:**
  - E[season best] at every Δ;
  - P(reach ≥ 20), P(reach ≥ 30), P(reach ≥ 40);
  - resets;
  - the A1 − A0 and A2 − A1 decomposition.
- **P(57), a projection, not a measurement.** For each arm, an exact fixed-policy evaluation of P(reach 57) by backward induction over (s, m, d, saver, q) under iid phase-aware rates, computed two ways:
  - (i) on the fitting seasons' rates (in-sample);
  - (ii) on the held-out season's rates.

  It is reported with a calibration stress: every bin's p_hit and p_both lowered by 0.02 and by 0.04. Every P(57) cell is labelled a model projection; the July memo shows iid models get long runs policy-dependently wrong.
  - **Needs a new evaluator:** the repo has none over the m state. It is written test-first, and checked against `solve_mdp`'s own value on A1, where a P(57)-optimal policy's evaluated value must equal the solver's.
- **State-weighted decision changes:** for A2 against A0, the share of replayed days where the action differs, weighted by state-visit frequency; the high-streak (s ≥ 10) changes are reported separately.

## 6. Thresholds, fixed before any result
**Positive** means the engineering recommendation is to switch. All four must hold:
1. the pooled held-out A2 − A0 gain in E[season best] is **≥ +0.25 streaks** at Δ = 0;
2. the gain is ≥ 0 in at least **4 of 5** held-out seasons;
3. the pooled gain is still ≥ 0 at **Δ = 0.10**;
4. P(reach ≥ 20) is not lower than A0's by more than **2 percentage points**.

**Negative:** the pooled gain is ≤ 0 at Δ = 0. **Inconclusive:** anything else.

**Rationale (reasoning only, not a measurement):**
- The smallest policy contrast the July replay measured was +0.12 (always-double against deployed); +0.25 is about twice that.
- The 6/29 single-seed iid comparison found the two objectives' policies nearly identical, so a small effect is plausible, and a swap smaller than this is not worth the integration risk.
- The effective sample is about 5 seasons. The 4-of-5 sign rule is the main robustness check; no formal power claim is made.

**Whatever the disposition, Eric gets the full trade table** (§8) and decides under D7. The disposition is a recommendation only.

## 7. Stopping, family and missingness
- **Family:** one primary comparison (A2 against A0, pooled, Δ = 0). Everything else is descriptive.
- **Stopping:**
  - one candidate, run once on the registered split;
  - no retuning of bins, phase length or thresholds after results;
  - a failed parity check (§4) stops the run before any fixed-evaluator number is read.
- **Missingness:**
  - profile dates absent from a season are skipped, not imputed;
  - partnerless double days are demoted and counted;
  - each season's row and date counts are reported.

## 8. The trade table for Eric
**Rows:** A0, A2, A1, always-single, always-double.

**Columns:**

| Column | Detail |
|---|---|
| Expected longest streak (held-out) | Δ = 0 and Δ = 0.10, with SE |
| Change vs A0 | in streaks |
| Reach-20 / reach-40 | percent |
| Projected chance of 57 | in-sample and held-out rates, plus calibration stress, each labelled a projection |
| Days the action differs from A0 | state-weighted |

**One plain sentence above it:** "Switching to A2 changes the projected chance of 57 from X to Y (a model projection; not measurable on real sequences) and changes the expected longest streak by Z streaks [range across seasons]."

## 9. If Eric approves (integration: a separate engineering step, D7)
- **Artifact:** a new sha-bound one-table artifact covering the whole season (working name `mdp_season_best_policy.npz`). The loader re-solves it and demands equality, like the tail loader, and binds it to its rates manifest.
- **Code touch points** (from the read-only inventory; each needs tests and the deploy-gating review):
  - `tail_policy` horizon fixed at 28 days;
  - objective routing in `strategy.py` / `daily_decision.py`;
  - the hard-coded `SEASON_END_DATE`;
  - reach-57-only consumers: `scheduler.py` skip census, `health/mdp_policy_alignment.py`, the skip-policy shadow, the boundary census.
- **Checklist A2** (artifact pairing) must be extended to the new artifact before activation.

## 10. Compute and execution
- **Cost:** the solves take seconds. Replay covers 5 arms × 120 trajectories × 4 Δ, plus the P(57) evaluations.
- **Box run:** through the C1 launcher (`c1-r4b-*` units), **declared budget 4 CPU-hours, 3 wall-hours**, from `~/projects/bts-c1` pinned to the reviewed code commit.
- **Code:** `scripts/audit/c1_r4b/`, written test-first. The evaluator fixes (a)–(d) and the m-state P(57) evaluator each get unit tests that fail before the code exists.

## Limits stated up front
- **Effective sample:** about 5 seasons. The 120 trajectories reuse the same dates across 24 seeds.
- **Information:** the replay conditions on realized participation (estimated-PA profiles). These are historical sequences, not 2027.
- **P(57):** model-dependent and probably not distinguishable from calibration noise.
- **Optimism:** re-solved arms are optimistic in-sample; the held-out split addresses this only within five seasons.
- **Objective:** own E[best] is not the prize objective.
