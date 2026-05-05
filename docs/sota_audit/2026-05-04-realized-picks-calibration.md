# Realized-picks calibration — SOTA #12 phase 2

**Date**: 2026-05-04 (snapshot 2026-05-05T00:54Z)
**Branch**: `feature/realized-picks-calibration`
**Predecessor**: [Falsification-harness synthesis (2026-05-04)](./2026-05-04-falsification-harness-synthesis.md)
**Tracker**: [SOTA audit tracker](../superpowers/specs/2026-05-01-bts-sota-audit-tracker.md), area #12 phase 2
**Canonical artifact**: `data/validation/realized_picks_canonical_2026-05-04.parquet` (72 rows: 66 resolved, 6 pending)
**Scope**: docs + step-0 canonicalization + analysis. Not a deploy authorization.

> Per Codex agent-bus #154 + #156: strict current-model verdict is honestly underpowered (5 resolved pick-slots over 3 resolved days); the **pre-bpm/post-pooled-MDP stratum** (n=30 pick-slots) shows an exploratory overconfidence signal worth tracking. This stratum is NOT the same as a "broader post-pooled-MDP architecture regime" (which would also include the post_bpm strict-current cells, totaling n=35); the memo deliberately reports them as separate strata and refuses to pool them under one label.

## Headline

| Verdict | n (resolved pick-slots) | Confidence |
|---|---|---|
| **Strict current model (post-bpm-wiring, since 2026-04-30 12:27 ET)** | 5 (3 primary + 2 DD over 3 resolved days) | **inconclusive — sample-size-limited** |
| **Pre-bpm/post-pooled-MDP stratum (2026-04-15 23:21 ET → 2026-04-30 12:27 ET)** | 30 (15 primary + 15 DD) | **exploratory overconfidence signal (gap +10.5pp), concentrated in double_down slot and the 0.75-0.80 bucket** |
| **Pre-pooled-MDP (March 29 → April 15)** | 31 (17 primary + 14 DD) | **roughly calibrated (gap −0.7pp)** — historical context only |

**Action implied**: do NOT propose recalibration or threshold-shift from this pass. The strict current-model n=5 cannot support that decision. The pre-bpm/post-pooled stratum's signal is suggestive enough to set up a follow-up measurement, not deploy work.

## Step 0 — canonicalization

Production picks live on `bts-hetzner` at `/home/bts/projects/bts/data/picks/`. The canonical artifact at `data/validation/realized_picks_canonical_2026-05-04.parquet` captures every primary + double_down pick across 38 non-shadow non-scheduler 2026 pick files, with the following lineage columns: `source_file, date, run_time, slot, batter_id, batter_name, pitcher_id, game_pk, p_game_hit, actual_hit, result_status, projected_lineup, pick_file_result, regime, model_cutoff_label, cutoff_iso, attribution_source`.

The `pick_file_result` column carries the streak-level `result` field straight from the original pick JSON (audit-only — not used for `actual_hit`); future readers can compare PA-frame attribution against the original streak result and distinguish "pending because the game hasn't resolved" from "pending because the PA-frame lookup didn't find the batter."

**Attribution source is the PA frame**, not the pick JSON's `result` field. The pick file's `result` is a streak-level outcome (hit/miss for the day's parlay) that is biased on double-down days — `result=miss` could mean "primary missed" or "DD missed" or both. The PA-frame lookup of `(batter_id, date) → day_had_any_hit` is the unbiased per-pick attribution and is the source of every `actual_hit` value in the canonical artifact.

This already corrects a known bias from prior realized-picks analyses. The 2026-04-25 finding of `~7pp chronic overconfidence` (n=48) was based on the pick-file `result` field for primary picks only — the corrected attribution shows a meaningfully different picture, especially for DD picks (see slot breakdown below).

## Regime cutoffs

Per Codex #154: do **not** use git HEAD timestamp (the existing `_current_deploy_iso` reads HEAD, which now returns docs-only-merge timestamps and would filter out 100% of picks). Use **production-affecting commits on the `deploy` branch** to define regimes:

- **`post_bpm`** (strict current model): cutoff `2026-04-30T16:27Z` (commit `ee4190f`, "fix(predict): wire batter_pitcher_shrunk_hr through inference path"). The bpm feature was added to `FEATURE_COLS` in `7afee63` and wired into the prediction path in `ee4190f`; we use the wiring commit because that is when production prediction behavior actually changed.
- **`post_pooled_mdp_pre_bpm`**: cutoff `2026-04-16T03:21Z` (commit `e1ebde9`, "feat(simulate): ship pooled MDP policy (Option 7) — 24/24 LOO + 8/8 cross-path"). This is the **final** commit in the pooled-MDP change group (after `0528bfd` 2026-04-15 18:14 ET); using the final commit gives the conservative regime boundary.
- **`pre_pooled_mdp`**: catch-all, all earlier picks.

A pick belongs to the first regime whose cutoff timestamp is at-or-before its `run_time`. The post_pooled_mdp_pre_bpm stratum is **not** a "current model" claim; it is a "current MDP policy + earlier feature set" claim, useful for the secondary signal but not for production-deploy decisions. If a "current MDP-policy architecture aggregate" verdict is wanted, it would pool post_bpm + post_pooled_mdp_pre_bpm into n=35 with a `mixed feature regime` label — this memo deliberately does not do that, because the post_bpm subset is too small to anchor an aggregate and pooling would obscure the bpm feature delta.

## Headline metrics

```
regime                         n hits   rate  mean_p     gap   Brier     BSS
------------------------------------------------------------------------------
post_bpm                       5    5  1.000   0.738  -0.262  0.0692     nan
post_pooled_mdp_pre_bpm       30   19  0.633   0.739  +0.105  0.2479  -0.068
pre_pooled_mdp                31   23  0.742   0.735  -0.007  0.1920  -0.003
```

(BSS = Brier skill score against the predict-the-base-rate baseline; positive is informative, negative means worse than the constant predictor. NaN for `post_bpm` because base rate is 1.0 in this n=5 stratum.)

## Fixed-bin reliability (Wilson 95%)

Bins per Codex #154: fixed at `[0.65,0.70), [0.70,0.75), [0.75,0.80)` plus padding bins on either side. The strategic-question target zone is the middle three buckets. Quantile binning would smear across the boundaries we care about.

**Strict current model (post_bpm, n=5)**:

| bin | n | mean_p | mean_y | gap | wilson_lo | wilson_hi |
|---|---|---|---|---|---|---|
| [0.70, 0.75) | 3 | 0.725 | 1.000 | −0.275 | 0.439 | 1.000 |
| [0.75, 0.80) | 2 | 0.757 | 1.000 | −0.243 | 0.342 | 1.000 |

Both buckets show 100% realized hit rate against ~73-76% predicted. **Direction is consistent with under-confidence.** Sample size is too small (Wilson lo bounds of 0.439 and 0.342) to support a verdict — the underpowered finding is the result.

**Pre-bpm/post-pooled-MDP stratum (post_pooled_mdp_pre_bpm, n=30)**:

| bin | n | mean_p | mean_y | gap | wilson_lo | wilson_hi |
|---|---|---|---|---|---|---|
| [0.65, 0.70) | 2 | 0.687 | 0.500 | +0.187 | 0.095 | 0.905 |
| [0.70, 0.75) | 19 | 0.726 | 0.737 | **−0.011** | 0.512 | 0.882 |
| [0.75, 0.80) | 9 | 0.776 | 0.444 | **+0.332** | 0.189 | 0.733 |

The 0.70-0.75 bucket (n=19) is **essentially calibrated** (gap −1.1pp). The 0.75-0.80 bucket (n=9) shows striking overconfidence (gap +33pp, but Wilson CI [0.189, 0.733] is too wide for a confident verdict). The split is suggestive: the model's boldest predictions in this regime are systematically not paying off, while its mid-range predictions are well-calibrated.

**Pre-pooled-MDP (pre_pooled_mdp, n=31, historical context only)**:

| bin | n | mean_p | mean_y | gap | wilson_lo | wilson_hi |
|---|---|---|---|---|---|---|
| [0.65, 0.70) | 3 | 0.683 | 1.000 | −0.317 | 0.439 | 1.000 |
| [0.70, 0.75) | 18 | 0.723 | 0.667 | +0.057 | 0.437 | 0.837 |
| [0.75, 0.80) | 10 | 0.770 | 0.800 | −0.030 | 0.490 | 0.943 |

All three buckets contain calibrated estimates within their wide Wilson CIs. **No bucket-level overconfidence signal in the pre-pooled-MDP regime.** This is a notable contrast against the post-pooled-MDP regime, but neither sample is large enough to attribute the difference to the policy change versus sampling variance.

## Slot breakdown (where the architecture-regime signal lives)

```
post_bpm:
  primary       : n=3, hits=3/3 (1.000), mean_p=0.736, gap=-0.264
  double_down   : n=2, hits=2/2 (1.000), mean_p=0.741, gap=-0.259

post_pooled_mdp_pre_bpm:
  primary       : n=15, hits=11/15 (0.733), mean_p=0.751, gap=+0.018
  double_down   : n=15, hits= 8/15 (0.533), mean_p=0.726, gap=+0.192

pre_pooled_mdp:
  primary       : n=17, hits=13/17 (0.765), mean_p=0.749, gap=-0.016
  double_down   : n=14, hits=10/14 (0.714), mean_p=0.718, gap=+0.004
```

In the post_pooled_mdp_pre_bpm stratum, **primary picks are well-calibrated (gap +1.8pp); the overconfidence signal is concentrated in the double_down slot (gap +19.2pp)**. In the pre_pooled_mdp regime, both slots are calibrated. The DD-slot signal is the most actionable observation in this analysis *if* it survives a larger sample, because:

1. DD-slot predictions average ~0.726 — squarely inside the 0.65-0.80 strategic-target zone.
2. The DD-slot rate of 8/15 = 53.3% at mean predicted ~0.726 is **large enough to track but not large enough to rule out sampling variation** — a one-sided binomial tail (P(X ≤ 8 | n=15, p=0.726) ≈ 9%) is suggestive without being implausible-by-chance.
3. If the DD-slot overconfidence is real and persistent, it would suggest **the model's "second-best for the day" is structurally weaker than the headline `p_game_hit` value would suggest** — possibly a bias from how the optimizer selects DD candidates after primary is fixed.

## Interpretation guardrails

1. **Underpowered current-model verdict is the headline result, per Codex #152's "if the stream is too underpowered, that is the result" framing.** The strict post-bpm-wiring stratum has n=5, and 5/5 hits is consistent with everything from "the model is well-calibrated and we got lucky" to "the model is dramatically under-confident at top-of-slate." We can not distinguish.

2. **The post_pooled_mdp_pre_bpm regime mixes feature surfaces.** The bpm feature wasn't in production from 2026-04-15 to 2026-04-29; it was wired in on 2026-04-30. Picks in this regime came from a feature set that current production no longer uses. Treating its calibration signal as evidence about **current** production is a category error; treating it as evidence about **the previous architecture** is fine, with the explicit understanding that the feature delta could move the calibration in either direction.

3. **The 0.75-0.80 bucket falls inside the strategic-question target zone but moves in the OPPOSITE direction from the under-confidence hypothesis.** The tracker reframe (per `project_bts_skill_vs_park_calibration_2026_05_01.md`) hypothesizes: low-skill park-driven picks at predicted 0.65-0.80 realize **higher** than predicted (under-confidence). The post_pooled_mdp_pre_bpm 0.75-0.80 bucket realizes **lower** than predicted (overconfidence — opposite sign from the hypothesis). Two readings: (a) the picks landing in this bucket weren't predominantly low-skill park-driven, so the hypothesis doesn't apply to this stratum; (b) the hypothesis needs sub-bucket disambiguation by pick-driver (skill vs park) rather than by predicted-P-bucket. The current canonical artifact does not contain pick-driver attribution; this is a follow-up.

4. **The DD-slot overconfidence finding is exploratory.** It survives PA-frame attribution (so it is not the streak-result-bias from 2026-04-25), but n=15 with Wilson CI [0.30, 0.75] is too wide for an actionable signal. Track over the next 30 days of resolved picks.

5. **The 2026-04-25 chronic ~7pp overconfidence finding (n=48) used iteration-contaminated data plus the streak-result attribution bias.** This memo's finding (gap +10.5pp on n=30 in the post_pooled_mdp_pre_bpm stratum, PA-frame-attributed and regime-bounded) is **not** a corroboration of the 2026-04-25 number. It is a **fresh measurement on a partially-disjoint sample with cleaner attribution**, and the two numbers are not directly comparable. The 2026-04-25 number should be treated as historical iteration-contaminated context, not as a baseline to compare against. **The most surprising contradiction**: with PA-frame attribution, primary picks are well-calibrated (gap +1.8pp) and the overconfidence signal lives in the DD slot — the inverse of what the streak-result-based 2026-04-25 method would have surfaced, since `result=miss` on a DD-bias day misattributes a hit-DD as a miss whenever the primary missed. The 2026-05-01 health-fix `b08769d` corrected this in the realized_calibration alert code; this memo is the first retrospective analysis to apply the fix.

## What this memo does NOT say

- It does NOT say current production is overconfident. The strict current-model n=5 cannot support that.
- It does NOT say the strategic-question reframe (under-confidence at top-of-slate, low-skill park-driven) is falsified. The current data lacks pick-driver attribution to test it.
- It does NOT propose a recalibration, threshold shift, or any deploy-side change. The sample size and the methodology stack (no aggregate CIs, no lockbox certification) do not support deploy-grade decisions.
- It does NOT use the 2026-04-25 n=48 result as supporting evidence except as historical iteration-contaminated context.

## What this memo establishes

- A canonical realized-picks dataset with PA-frame attribution and explicit regime labels — reproducible from a single parquet.
- A canonicalization script (`scripts/canonicalize_realized_picks.py`) with a `--summary` flag that prints the headline metrics + fixed-bin reliability + slot breakdown tables cited above. The exact regeneration command for this memo's tables (run from the project root, with current production picks rsynced to `/tmp/realized_picks_input/` and the freshest pa_2026.parquet at `/tmp/pa_2026_fresh.parquet`):
  ```
  UV_CACHE_DIR=/tmp/uv-cache uv run --extra model python scripts/canonicalize_realized_picks.py \
    --picks-dir /tmp/realized_picks_input \
    --pa-path /tmp/pa_2026_fresh.parquet \
    --output data/validation/realized_picks_canonical_2026-05-04.parquet \
    --summary
  ```
- An explicit framing that "current model" is the post-bpm-wiring regime, not the post-pooled-MDP architecture aggregate (which would be n=35 if it included the strict-current cells), and the post_pooled_mdp_pre_bpm stratum cannot be relabeled as the former.
- An exploratory observation that the post_pooled_mdp_pre_bpm stratum's overconfidence signal is concentrated in the DD slot and the 0.75-0.80 bucket, both of which are plausible attribution patterns worth tracking but not yet actionable.

## Next steps (recommendations, not commitments)

1. **Add `model_git_sha` to `save_pick` output.** Production picks currently carry no model-identity field; recovering "which model produced this" requires separately matching `run_time` against `deploy` branch git history. A small `feat(picks)` PR adding `model_git_sha` would make future calibration analyses self-contained. ~30 min of work.

2. **Re-run #12 phase 2 in 30 days.** With ~30 more resolved post-bpm picks, the strict current-model verdict can move from "underpowered" to a real signal. The script + canonical artifact are reusable.

3. **Add pick-driver attribution to the canonical artifact.** Specifically: `is_park_driven` (Coors, hot weather, etc.) and `batter_skill_quartile`. This would let the next cycle test the strategic-question reframe directly (does the under-confidence finding hold for low-skill park-driven picks?). Likely M-effort because it requires re-computing features at pick time.

4. **DO NOT pursue Option B (#5 P1.5 aggregate CIs) yet.** Per Codex #152: build the gate after we know what door we are trying to walk through. We do not have a candidate adjustment yet; aggregate-CI infrastructure is premature. Revisit after step 2 produces a non-underpowered current-model verdict.

5. **Track the DD-slot signal as a secondary surface.** If the DD-slot overconfidence persists across regimes as more picks resolve, it points at a structural issue in DD selection (not a top-of-slate calibration question), which would shift the whole strategic frame.
