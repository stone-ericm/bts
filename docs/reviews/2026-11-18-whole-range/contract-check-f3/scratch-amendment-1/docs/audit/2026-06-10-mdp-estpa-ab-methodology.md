# MDP estimated_pa re-solve — A/B methodology (2026-06-10)

## Problem (the quality-bin collapse)
The shipped MDP policy (`data/models/mdp_policy.npz`) was solved on the
**actual_pa** game-probability basis — quintile boundaries ≈ `[0.796 … 0.841]`.
Production, however, computes **estimated_pa** probabilities at pick time
(starter-matchup representative row, PA count estimated from lineup slot,
starter/reliever split). Live top-1 estimated_pa probs sit ≈ `[0.69, 0.83]`,
**below most of the actual_pa boundaries**. So `np.digitize(live_p, actual_pa_boundaries)`
piles nearly all live picks into the bottom bin(s) and never reaches the top bin —
the policy's quality discrimination is wasted. Fix: re-solve the policy on the
estimated_pa basis so the boundaries land inside the live distribution.

## How the policy is used in production (faithful replay target)
`strategy.py`: `policy_table, boundaries, season_length = load_policy(...)`.
Per candidate day: digitize rank-1 estimated_pa prob through the **single saved
`boundaries`** → bin `qb`; action = `policy_table[streak, days_remaining, saver, qb]`.
(`save_policy` stores one `boundaries` array; phase-awareness is baked into the
day-indexed table, not into separate saved boundaries.)

## Why the stock A/B (`scripts/pooled_policy_ab.py`) is WRONG for this change
That script uses `evaluate_mdp_policy(policy_table, bins)`, whose docstring states
**"quintile position carries the semantic meaning, not absolute confidence."** It
re-bins the estimated_pa profiles into fresh matched quintiles and applies each
policy by **bin index**. That gives BOTH policies the benefit of correct quintile
binning and varies only the action-per-quintile table — but the production benefit
here is almost entirely **boundary-driven** (re-binning the live distribution), not
action-table-driven. So the stock A/B would *under-measure / miss* the real benefit.
(It was written for the original Option-7 work, which pooled more seeds on the SAME
actual_pa basis — a different question.)

## Correct A/B: as-deployed, each policy through its OWN boundaries
The fix is to bin the live (estimated_pa) stream through each policy's **own saved
boundaries** (`compute_bins_with_boundaries`, which retains zero-freq bins → stays
shape-compatible with the 5-bin table), then take the value. Harness:
`scripts/mdp_estpa_ab.py`.

- **PRIMARY = analytic `evaluate_mdp_policy(table, compute_bins_with_boundaries(seed_holdout, own_boundaries))`** — exact backward-induction E[P(57)], no MC noise. Per seed (merge is date-only, so one seed at a time) → paired diff → **bootstrap CI over seeds**.
- **SECONDARY = `_terminal_mc_replay`** (own boundaries). Kept as a cross-check but **structurally ≈0 here**: at P(57)~0.01%, 48 season-seed trajectories yield ~0 successes, so it can't discriminate. Don't rely on it.
- **DECISION SHIFT (the actionable lens)** = bin-occupancy under each policy's boundaries + fraction of decisions that change over a (streak×days×saver) grid weighted by the live pick distribution + skip/single/double mix. This is what makes the near-zero P(57) gap legible.

Build the candidate two ways: in-sample (all profile seasons) and **true OOS**
(`rebuild_mdp_policy_pooled.py --build-seasons 2021,2022,2023`, evaluate 2024,2025).

## Smoke validation (4 seeds: 1,7,42,137 — 2026-06-10, before the 24-seed run finished)
Pulled 4 completed seeds off the live fleet and ran the whole pipeline:
- **Collapse confirmed & quantified**: under prod boundaries `[0.796,0.812,0.825,0.841]`, **79.7% of live rank-1 picks land in bin 0** (top bins ~1%). New estimated_pa boundaries `[0.759,0.772,0.784,0.800]` give a proper ~20%/bin spread.
- **Directional result holds in-sample AND OOS**: analytic mean gap positive, 95% CI excludes 0, candidate wins 4/4 seeds both ways (in-sample +0.0040pp; OOS +0.0004pp — smaller, expected). CIs are wide at n=4; the 24-seed run tightens them.
- **Honesty flag**: as-deployed estimated_pa P(57) ≈ **0.01%**, *orders of magnitude below* the 8.17% actual_pa headline (and below the ~3.3% demoted figure) — consistent with BTS being near-unwinnable. The re-solve's value is **sensible live decisions, not a meaningfully higher (still ~0) win prob.** Frame it that way to Eric.
- **Behavioral change is large, not cosmetic**: **36% of decisions change**; the collapse made prod **over-skip (skip 64%)**; the re-binned policy skips 41%, doubles 26%→37%. NOTE the model-risk: the extra aggression assumes the estimated_pa per-bin hit rates are accurate — worth an adversarial Codex pass on the REAL 24-seed numbers ("is the aggression genuine value or overfit to noisy bin estimates?") before shipping.

## Open decisions (resolve when data lands; consider Codex on the writeup)
1. **Holdout seasons.** Profiles cover 2021–2025; validation split test seasons =
   2024,2025. The pooled policy as built by `rebuild_mdp_policy_pooled.py` pools ALL
   profile seasons, so replaying on 2024–2025 gives the pooled policy a mild in-sample
   edge. For a true OOS read, also build a pooled policy from 2021–2023 only and replay
   on 2024–2025. Report both; do not overstate (cf. the 8.17%→0.0333 honesty correction).
2. **Live policy provenance.** Compare against the policy ACTUALLY deployed on
   bts-hetzner (`/home/bts/projects/bts/data/models/mdp_policy.npz`), not just the local
   copy — fetch it (small, clean over tailnet) for the comparison.
3. **Phase boundaries.** Production digitizes through one saved boundaries array; match
   that in replay (don't introduce phase-aware boundary switching the live path doesn't do).
4. **Ship gate.** Only swap the live policy if pooled ≥ prod on the OOS replay with a CI
   that excludes a meaningful regression. Present numbers to Eric BEFORE any swap.

## Run that produced the profiles
`audit_driver.py --run-kind profiles --game-probability-mode estimated_pa --data-relay
--boxes 12 --seeds 24 --test-seasons 2024,2025 --profile-seasons 2021,2022,2023,2024,2025
--no-log-pa-predictions` → `data/hetzner_results/mdp_estpa_run` (9/12 boxes ready; all 24
seeds launched; `BTS_LGBM_DETERMINISTIC=1` per seed for reproducible pooling).

---

# FINDINGS & DECISION (24-seed run, 2026-06-10)

Scripts: `mdp_estpa_ab.py` (A/B), `mdp_estpa_robustness.py` (ablation/shrinkage/milestones).
Result JSONs in `data/validation/mdp_estpa_ab_*.json`. Recovery note: the on-Mac driver
froze in its poll-sleep (macOS lid-close sleep; `caffeinate -is` doesn't stop lid-close on
battery) — took over manually: retrieved 24 seeds × 5 seasons (120 parquets), verified all
9 boxes torn down + relay key removed. **Pooled candidate: `data/models/mdp_policy_pooled_estpa_v1.npz`** (NOT shipped).

## A/B (analytic as-deployed E[P(57)], 24 seeds)
- In-sample: prod 0.0004% → cand 0.0025%, +0.0021% [CI +0.0013,+0.0031], 23/24.
- True OOS (build 2021-23, eval 2024-25): prod 0.0004% → cand 0.0016%, +0.0013% [CI +0.0008,+0.0018], 23/24.
- Both "SHIP" by CI — but **misleading** (see below). Collapse confirmed: prod buckets 79% of live picks into bin 0; cand restores ~20%/bin; 38-43% of decisions change (skip 64%→34-39%).

## Robustness pass → DECISION: **HOLD (do not ship)**
- **Ablation**: boundaries-only (prod table + new boundaries) captures 44% of cand's analytic gain; the aggressive new table adds ~nothing (+0.0001%).
- **Shrinkage**: at −5pp bin-rate pessimism the cand edge ≈ vanishes (→ prod). Fragile.
- **Milestones (realized replay, 120 trajectories) — decisive**: cand is *worse* than prod — reach20 30.8% (cand) vs 30.8% (prod) but max-streak 18.0=18.0 and **resets 56 vs 36**; the analytic P(57) "win" never materializes (no trajectory reached 57). The aggression just resets more.
- Codex (gpt-5.5) adversarial review concurred: flat bins = "biggest red flag"; the 24-seed bootstrap is correlated-seed variation, not real season/calibration CI; don't ship aggression on a near-0, fragile, model-implied edge.

## The bigger findings (upstream)
1. **Strategy layer is over-engineered.** A one-line "always double" beats the deployed MDP on realized milestones (reach20 42.5% vs 30.8%, max-streak 18.4 vs 18.0). Streak-threshold sweep: no simple rule beats always-double; the MDP's day-quality conditioning adds nothing. Doubling is realistic — forcing a *different*-game rank-2 leaves P(both)=57.3% unchanged (different-game constraint is free).
2. **The model ranks PICKS fine** (rank-1 77.5% → rank-10 71.3%, calibrated); corr(pred,hit)=0.045 was range-restriction + binary noise, NOT a broken model. What's flat is **day-to-day top-pick quality** (~78% every day, pred sd 0.024) — which is why day-conditioning is useless.
3. **…but day-quality flatness is partly a MODEL/AGGREGATION blind spot, not reality.** Opposing-**starter** quality drives a **+9.0pp** realized swing in top-pick hit rate (ace 67.9% → weak starter 76.9%) while the estimated_pa prediction is **flat (+0.5pp)**. Crude full-season starter hit-allowed out-predicts the model (corr 0.073 vs 0.049). All other observables (park/roof/slot/temp/wind/home-away) were flat (≤3.5pp). **NOT a missing feature**: `pitcher_hr_30g` is misnamed — it IS the pitcher rolling hit-rate-allowed (`compute.py:326`), the documented "strongest feature." So the gap is in how the **estimated_pa surface** uses it (starter-blind generic-reliever blend dilutes it; and/or the PA model under-weights it; and/or the 30g-rolling feature is noisier than a full-season proxy).

## NEXT (open threads)
- **Pin the starter blind-spot mechanism**: bucket realized hit by the model's *own* `pitcher_hr_30g` value (flat there → not used; spreads → my proxy just sharper). Compare actual_pa vs estimated_pa starter-residuals (isolates the aggregation). This lives in the production estimated_pa surface → likely higher-value than the MDP.
- **Statcast "miss distance" (queued)**: pitcher contact-suppression signal → candidate feature to strengthen the pitcher dimension. See bts_index.md queue entry. Caveats: Savant leaderboards, coverage **mid-2023+ only** (BTS trains 2019+), swing-conditional. Feature work goes through brainstorming at pickup.
- Eric's objective fork (jackpot P(57) vs longest-streak standing) still informs which simple strategy to adopt if the policy layer is simplified.

## CAPSTONE — quantified the starter blind-spot (2026-06-10, seed 42, n=8,285 top-10 picks)
Mechanism pinned: the model DOES use its pitcher hit-allowed feature (`pitcher_hr_30g`, misnamed; corr(feature,pred)=0.086) but the feature is weak — 30-game-rolling/shifted captures only +3.8pp of the +9.0pp full-season starter signal, and the `1−(1−p)^n` game-aggregation compresses prediction spread to +0.6pp.
- **⚠ LEAKAGE CORRECTION**: the eye-catching "+0.019 AUC / +9pp starter signal" was MOSTLY LEAKAGE. It bucketed by the pitcher's FULL-SEASON hit-allowed (includes games AFTER the one predicted → encodes true quality not knowable at pick time). Per project rule "fix leakage first," the leak-free number is the real one.
- **Leak-free AUC lift from a better opposing-starter feature ≈ +0.004 to +0.006** (consistent across 30/60/100-game rolling + expanding-shrunk; model-alone AUC ~0.530–0.534). Marginal — and partly just un-compressing the `1−(1−p)^n` aggregation (adding the raw rolling feature linearly recovers a sliver), not new info. The model's existing `pitcher_hr_30g` already captures most of the *knowable* leak-free starter signal.
- **Honest conclusion**: the model is near its achievable ceiling for top-pick hit prediction (~0.53 AUC — intrinsically hard). This session rigorously RULED OUT three levers — MDP re-solve, strategy layer, and a better starter feature — as meaningful wins. Negative result; redirects effort.
- **The one live (small) thread**: the estimated_pa aggregation compresses real signal (the ~+0.006 from adding raw rolling features linearly) — possibly across features, not just pitcher. More promising than any single new feature.
- Statcast miss-distance (queued): worth trying ONLY because it's an *orthogonal* signal type (live contact/stuff quality, not recycled historical hit-rate) → could clear the bar a historical feature can't. Tempered expectations: must beat a near-ceiling model.
- Process note: got ahead of the data twice this session (corr=0.045 "model broken" → range-restriction; "+9pp starter" → leakage). Both caught by disciplined checks. Use leak-free measures from the START next time.
