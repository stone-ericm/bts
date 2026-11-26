# Skip threshold, re-solve, and the discrimination ceiling (2026-06-29)

**One line:** the deployed skip threshold is ~cosmetic (it neither protects you from worse days
nor establishably costs you streak), the live-matched re-solve barely changes outcomes, 57 is
effectively unreachable, and the one concrete improvement lever we tested — picking by expected
plate-appearance volume — dissolved under scrutiny. **No production change made; threshold kept as-is.**

Extends [`2026-06-20-skip-policy-shadow.md`](2026-06-20-skip-policy-shadow.md) (which first surfaced the
scale mismatch / ~0.744 breakeven) and [`2026-06-10-mdp-estpa-ab-methodology.md`](2026-06-10-mdp-estpa-ab-methodology.md).
This round actually ran the estimated-PA backtest, settled the objective/play-rate questions, and chased the
discrimination lever to ground. Reviewed across **7 Codex (gpt-5.5) rounds**, which pared back **four**
successive over-confident point estimates — that pattern is itself a finding (see §10).

---

## TL;DR — decisions & durable conclusions

- **Keep the 0.796 threshold as-is.** It is mis-scaled (fit on actual-PA, applied to lower estimated-PA),
  so the bot skips ~75% of live days — but the days it skips hit about as often as the ones it plays, so
  the over-skipping is not establishably costing anything (~0.2 E[max], within noise).
- **Don't re-solve for production right now.** A clean live-matched re-solve lifts the season-best streak
  by an amount indistinguishable from zero and only ~triples a ~0.002% jackpot.
- **57 is a wall.** Even the best identifiable live pick hits ~0.80; 0.80⁵⁷ ≈ 3e-6. Discrimination can only
  move the season-best-streak distribution modestly, never make the jackpot reachable.
- **Don't chase a plate-appearance / lineup-freshness lever.** Tested two ways; not real, and unverifiable
  from any realistic amount of data (~135 seasons to confirm a true +1pp pick-level effect).
- **Open item (unrelated, actionable):** `live_forward_resolution` stalled on 2026-06-17 — quietly degrades
  the realized/calibration data the whole system leans on.

---

## 1. What was verified

- **Production runs the MDP, not the heuristic fallback.** `strategy._load_mdp()` returns a live policy;
  every `decision.json` since the schema launched (06-23) is `source: "mdp"`.
- **The analysis solver is bit-identical to production's.** Re-implemented `solve_reach`/`solve_emax` reproduce
  `bts.simulate.mdp.solve_mdp` exactly: same `optimal_p57=0.0483` on the April bins, 100% policy agreement
  across a 350-state grid. So all comparisons below are on the same dynamics production uses.

## 2. How the skip threshold moves with streak

The implied "min p to act" is a gently **rising staircase**: θ≈0 (take everything) below streak ~8 (build phase);
~0.796 at 8–15; ~0.811 at 15–25; ~0.825 at 30–45; pinned at the top of the band (≈"wait") at 50+. The streak-saver
lowers the bar ~1.5pt in the 10–15 window (a miss is caught there, so acting is cheaper).

Under a pure-P(57) objective the threshold is ≈**flat at the per-pick survival rate q** (~0.80): "a longer streak is
worth more" (push up) is offset by "a longer streak is closer to the prize" (push down). The *sign* of dθ/ds is
governed by **log-concavity of (V(s)−V(0))**, not ordinary concavity of V — the staircase reflects mild log-concavity
plus finite-horizon/saver/double effects. (Earlier framings of "concavity of V" and "the two effects exactly cancel"
were imprecise — Codex r1/r2.)

## 3. The scale mismatch (now quantified)

Bins are equal-frequency quintiles of top-pick `p_game_hit`, built from an **actual_pa** backtest (the model knows each
batter's realized PA count = hindsight). Production predicts from **projected lineups** (estimated PA). Median top-pick:

| | actual_pa (bins built here) | estimated_pa (live-matched) | live 2026 (41d) |
|---|---:|---:|---:|
| median top-pick p | 0.817 | **0.778** | 0.765 |

A clean **estimated_pa backtest** (912 days; profiles on bts-hetzner at
`data/validation/estpa_profiles_2026-06-29/`) lands at 0.778 → **the actual_pa hindsight accounts for ~75% of the
backtest-to-live gap** (same-games A/B; caveat: actual_pa used the April model, estimated_pa the current model — models
verified stable). Consequence: the 0.796 boundary sits near the *top* of the live distribution, so the deployed policy
classifies ~75–80% of live days as bin-0 "skip" and plays only ~20–26% of days. The metadata `optimal_p57=0.0817` is an
actual_pa fantasy; a clean re-solve of the April bins gives 0.0483, and on the live scale P(57) is ~0.002%.

## 4. Does the threshold actually help? (skip vs play)

**No reliable evidence it does.** Splitting the 912-day estimated_pa sample at 0.796:

| | n | actual hit rate |
|---|---:|---:|
| PLAY (p ≥ 0.796) | 209 | 0.766 |
| SKIP (p < 0.796) | 703 | 0.734 |

Difference **+3.2pp, ±6.6pp — not significant.** And the threshold sits in the **flat part** of the hit-rate curve:
band 0.78–0.796 hits 0.755, band 0.796–0.82 hits 0.754 (n=208/167) — identical where it actually cuts. The only real
gradient is at the extremes (sub-0.76 days hit ~0.71). The live record *hints* at a bigger gap, but after deduping a
variant double-count it's **3 production play-days that all hit** (Fisher one-sided p=0.37) — noise. Bayesian pooling
(clean prior + live): posterior gap ~+3.5pp, P(gap>10pp)≈2%. **Honest framing: no reliable evidence the 0.796 threshold
separates good days from bad.**

## 5. Should you play more? (play-rate)

A simple play-threshold sweep from the current state (streak 14, 91 days, saver) shows a gentle hump: outcomes rise from
~23% play (E[max] 18.5) to ~75% play (18.8), then dip at 100%. But a **paired bootstrap that re-estimates the bin rates
per resample** (not the over-confident fixed-rate version) gives the play-more effect as **+0.26 E[max], 90% CI
[−0.65, +1.07]**, positive in only ~70–80% of pools (the bootstrap fraction is itself RNG-sensitive across re-runs; the CI spanning zero is the robust point). So: **play-rate is ~indifferent within the data's precision** — the
point estimate weakly favors playing more, but it cannot be established. Not "genuinely indifferent" (no equivalence
proven); not "you're hurting yourself" (the +0.3 was within noise). (Codex r5.)

## 6. P(57) vs E[max], and the wall

Re-solving under "maximize P(57)" vs "maximize E[season-best streak]" yields **nearly identical policies** on the live
scale (from current state: E[max] 19.2 vs 19.6; the skip thresholds match). So you can pursue both with a single
P(57) policy. From the current state on live-matched bins: **deployed E[max] 19.0 → re-solve 19.2–19.6** (bootstrap CI
[18.9, 20.6]); P(max≥30) 0.019 → 0.027; exact P(57) 0.0007% → 0.0020%. **The wall:** 0.80⁵⁷≈3e-6, 0.85→9.5e-5,
0.90→0.0025; the live top decile is ~0.79 (0.92 only with PA hindsight). Discrimination can shift the season-best
distribution modestly; it cannot make 57 reachable.

## 7. Is discrimination a lever? (the main investigation)

Corrected metrics (an earlier "AUC≈0.50, the model can't discriminate" was a **bug** — labels sorted by score but
indexed with original-order ranks; fixed with `scipy.rankdata`, sanity-checked):

| signal (rank-1, vs actual hit) | AUC |
|---|---:|
| model `p_game_hit` (full) | 0.536 |
| `est_pas` (pre-game PA estimate) | 0.516 |
| `lineup_position` (pure pre-game slot) | 0.516 |
| realized `n_pas` (contaminated) | 0.641 |

So live discrimination is **weak (0.536), not random.** The realized-PA AUC of 0.64 is almost entirely **reverse
causality** — a hit extends the inning → more PAs; the *predictable* PA signal (`est_pas`, lineup slot) is ~random
(0.516). **Why:** ~97% of picks are top-of-order hitters (slots 1–3, ~4.4–4.6 PAs each), so PA volume barely *varies*
among the elite hitters you'd consider — nothing to discriminate with.

**Policy-level test (the right test):** re-ranking each day's slate by `est_pas`/slot *looked* like +1.7pp — but that was
a searched, tie-break-sensitive, in-sample point estimate. The clean paired comparison (McNemar over the ~290 days where
the pick differs) is **146 vs 143, p=0.91** (honest effect +0.33pp); it helps 2 of 5 seasons and hurts 2 of 5. And the
backtest is **wildly underpowered**: detecting a true +1pp pick-level effect at 80% power needs **~25,000 days ≈ 135
seasons**. **Conclusion: no validated PA-volume lever; don't deploy a PA tilt.** (Per Codex r7, the honest scope is "no
validated standalone PA lever" — *not* "the model is globally near-optimal"; matchup/model-class headroom is untested.)

## 8. Streak Saver model — verified correct

Production's `transition_outcomes` (and the analysis code) match the official BTS rules exactly: one-time; auto-applied
the first time the streak reaches 10; saves a No-Hit only while streak ∈ [10,15]; holds the streak at its current value
(no increment); **catches a failed Double-Down including the both-miss case**; an unused saver survives a reset and is
usable on the next ≥10 climb. The one modeling gap is **voids/"Pass" picks** (postponements / no-PA), which the planning
MDP doesn't represent — purely *conservative* (a void never resets you).

## 9. Decisions

- Keep the deployed 0.796 threshold. No re-solve deployed. Nothing changed in production.
- If a re-solve is ever shipped: do it for the **P(57)** objective (serves both goals), on **estimated_pa** bins, and
  expect hygiene (the bot stops idling ~75% of days) rather than a measurable streak/jackpot gain.

## 10. Methodology notes — the four walk-backs

Every headline *point estimate* in this investigation was revised downward by a robustness/significance check or a Codex
round. Recording them because the pattern is the lesson — on this signal, only the rigorous tests held:

1. "Re-solving lifts E[max] ~18→20-21" → on clean 912-day bins it's 19.0→19.2 (the 18→21 was 41-day-bin noise).
2. "AUC≈0.50 / model can't discriminate" → buggy AUC; corrected to 0.536 (weak, not random).
3. "~0.07 AUC recoverable via PA prediction" → the actual_pa "ceiling" is oracle-ish/reverse-causal; recoverable ≈ 0.
4. "Picking by PA tilt gives +1.7pp" → in-sample/tie-break artifact; honest effect +0.33pp, p=0.91.

## 11. Reproducibility

- **Data:** estimated_pa profiles → bts-hetzner `~/projects/bts/data/validation/estpa_profiles_2026-06-29/backtest_*.parquet`
  (912 days; columns incl. `p_game_hit`, `actual_hit`, `n_pas`, `est_pas`). Generated by
  `bts simulate backtest --seasons 2021,2022,2023,2024,2025 --game-probability-mode estimated_pa`.
  actual_pa profiles → `data/simulation/backtest_*.parquet`. Lineup slot → `data/processed/pa_*.parquet` (`lineup_position`).
- **Methods:** generalized P(reach-K) and augmented (streak, running-max) E[max] DP solvers; exact fixed-policy DP
  evaluation; vectorized Monte-Carlo with the saver/reset transitions; paired day-bootstrap with per-pool bin re-estimation;
  `scipy.rankdata` AUC; within-day re-ranking + McNemar. Scratch scripts were session-local — re-derive from this doc or
  ask for them to be committed under `scripts/audit/` if a reproducible artifact is wanted.

## 12. Open item

`live_forward_resolution` has been **stalled on 2026-06-17 for >10 days** (missing outcomes for 2 artifact rows; flagged
CRITICAL in the scheduler log). This degrades the realized-data growth that calibration/gate checks (and any future
re-solve) depend on. Unrelated to the threshold question but worth fixing.

## 13. Codex consultation log (gpt-5.5, high reasoning)

7 rounds. Net effect: confirmed the operational conclusions while killing four over-confident magnitudes. Key catches:
fixed-bin bootstrap froze the main uncertainty (use per-pool re-estimation); the live skip-vs-play CI was an invalid
normal-approx on a 3/3 proportion (use Fisher); AUC is the wrong yardstick for an argmax-one-pick problem (use policy-level
re-ranking); the realized-PA signal and actual_pa "ceiling" are reverse-causally contaminated; and "near the discrimination
ceiling" overclaims what the PA test alone can support.
