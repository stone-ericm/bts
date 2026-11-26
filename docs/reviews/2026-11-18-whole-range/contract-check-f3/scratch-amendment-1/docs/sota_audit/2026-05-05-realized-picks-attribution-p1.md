# Realized-picks attribution — SOTA #12 phase 3, α P1

**Date**: 2026-05-05 (snapshot 2026-05-05T20:11Z)
**Branch**: `feature/realized-picks-attribution-p1`
**Predecessor**: [α P0 attribution memo](./2026-05-05-realized-picks-attribution.md)
**Tracker**: [SOTA audit tracker](../superpowers/specs/2026-05-01-bts-sota-audit-tracker.md), area #12 phase 3
**Canonical artifact**: `data/validation/realized_picks_canonical_2026-05-05_p1.parquet` (74 rows; P0 artifact `2026-05-05.parquet` preserved verbatim)
**Scope**: docs + canonicalize-script extension + analysis. Not a deploy authorization.

> Per Codex agent-bus #213 / #215 / #216: extends P0 with a season-to-date batter-skill-quartile axis so the low-skill × park-env-proxy cut becomes observable. P1 was framed as infrastructure + preflighted support, not as likely current-sample evidence; in practice the cut surfaces one heuristic calibration warning in the post_pooled_mdp_pre_bpm stratum (DD not_park_driven Q4, n=8) that warrants tracking — not a formal calibration test.

## Headline

| Cell | n | hits | rate | mean_p | gap | wilson_95 | Status |
|---|---|---|---|---|---|---|---|
| post_pooled_mdp_pre_bpm × DD × not_park_driven × Q4 | 8 | 3 | 0.375 | 0.731 | **+0.356** | [0.137, 0.694] | mean_p above Wilson upper; calibration warning, track |
| post_pooled_mdp_pre_bpm × primary × not_park_driven × Q4 | 8 | 6 | 0.750 | 0.766 | +0.016 | [0.409, 0.929] | calibrated |

**The strategic-question hypothesis** (low-skill park-driven picks at predicted 0.65-0.80 realize HIGHER) **remains untested on this sample**. The Q1 × park_driven cell that the hypothesis predicts has n=0 in the target stratum — production didn't pick any low-skill batters in park-driven games during 2026-04-15 → 2026-04-30. P0 said "not directionally supported on this sample with this proxy"; P1 sharpens that to "the predicted cell has zero observations, so the hypothesis remains untested."

**A different signal warrants tracking**: the P0 DD not_park_driven gap (+22.7pp on n=14) appears concentrated into Q4 (high-skill batters in non-park environments) at current resolution. With mean_p=0.731 falling outside the binomial Wilson interval [0.137, 0.694] for the observed rate, this cell is a calibration warning — useful as a heuristic flag, but NOT a formal calibration test given that it is a single post-hoc-selected n=8 cell from a multi-way cut with heterogeneous row probabilities and ~24 reported cells. The pattern is also notable in shape: high-skill (Q4), not low-skill (Q1); non-park, not park-driven; DD, not primary. If it survives n growth it points at a candidate structural DD-selection issue rather than a park-leverage calibration question; at n=8 it is a cell to watch, not a verdict.

Caveats kept explicit:
- n=8 with multi-way cut means multiple-comparisons context is unfavorable.
- Same stratum as the P0 DD signal — not independent confirmation, just refinement.
- "Skill" here is season-to-date 2026 hit rate, not career; quartile assignments will shift as more PAs accumulate.
- post_bpm strict-current-model stratum is n=7 with all picks landing in Q3 (n=3) or Q4 (n=4); not enough for a quartile-cut verdict on current production.

## Methodology

### Skill columns

Three new nullable columns added to the canonical artifact:

```
batter_skill_prior_pa        Int64 (nullable)
batter_skill_prior_hit_rate  Float64 (nullable)
batter_skill_quartile        Int64 (nullable; values {1, 2, 3, 4} or pd.NA)
```

For each pick row at `(batter_id, pick.date)`:

1. Filter the PA frame to rows where `batter_id` matches AND `date < pick.date` strictly. **No same-day PAs counted; no future PAs counted.** This is the no-leakage contract; tested explicitly via `test_no_same_day_pa_leakage`.
2. `prior_pa` = count of filtered rows; `prior_hit_rate` = `is_hit.mean()` over those rows (NA when `prior_pa = 0`).
3. If `prior_pa < MIN_PRIOR_PA` (= 50): `quartile = NA`. `prior_pa` and `prior_hit_rate` still populate for audit transparency. Codex bus #215/#216: `False` was already reserved for "observed-and-rule-says-no" in P0; here `NA` is reserved for "below threshold, can't be quartile-assigned."
4. If `prior_pa >= MIN_PRIOR_PA`:
   - Build the eligible league pool: ALL PA-frame batters with `prior_pa >= MIN_PRIOR_PA` as-of `pick.date` (NOT restricted to picked batters; that would condition on production behavior and blur skill with selection — Codex #215 A).
   - Compute the pool's quartile bounds: `q25, q50, q75 = pool["prior_hit_rate"].quantile([0.25, 0.50, 0.75])` via pandas linear interpolation.
   - Assign:
     ```
     prior_hit_rate <= q25  -> 1
     else <= q50            -> 2
     else <= q75            -> 3
     else                   -> 4
     ```
   Deterministic ties (lower-quartile bias). Tested at all three boundaries (`test_assign_quartile_tie_at_q25_low`, `_q50_low`, `_q75_low`).

For performance, the pool snapshot is built once per unique pick.date (36 dates in this artifact). Per-date semantics still hold because the snapshot is keyed on `pick.date`.

### Threshold rationale (preflight)

Threshold = 50 was chosen after a per-cell support preflight against the merged 2026-05-05 P0 artifact and the freshest `pa_2026.parquet`:

```
regime                       slot          env                total  >=50   %50  >=100  %100  prior_pa min/median/max
---------------------------------------------------------------------------------------------------------------------
post_bpm                     primary       park_driven            2     2  100%      2  100%  141 / 146 / 150
post_bpm                     primary       not_park_driven        2     2  100%      2  100%  143 / 144 / 146
post_bpm                     double_down   not_park_driven        3     3  100%      3  100%  132 / 147 / 148
post_pooled_mdp_pre_bpm      primary       park_driven            4     4  100%      1   25%  84 / 87 / 103
post_pooled_mdp_pre_bpm      primary       not_park_driven       11    11  100%      7   64%  69 / 111 / 125
post_pooled_mdp_pre_bpm      double_down   park_driven            1     1  100%      0    0%  83 / 83 / 83
post_pooled_mdp_pre_bpm      double_down   not_park_driven       14    14  100%      8   57%  52 / 106 / 122
pre_pooled_mdp               primary       park_driven            4     1   25%      0    0%  28 / 35 / 72
pre_pooled_mdp               primary       not_park_driven       11     4   36%      0    0%  21 / 47 / 76
pre_pooled_mdp               double_down   not_park_driven       14     7   50%      0    0%  25 / 47 / 69
```

At threshold=50: target stratum (`post_pooled_mdp_pre_bpm`) has 30/30 (100%) support and the n=1 DD park_driven cell (prior_pa=83, the cell flagged in P0 as the most-watched future signal) is preserved. At threshold=100: support drops to 16/30 (53%) and the DD park_driven cell zeros out (prior_pa=83 falls below 100). The chosen threshold preserves the analytical surface; threshold=100 is documented as the future/career-data alternative once prior-season PA history is plumbed in (deferred to a later P).

The pre_pooled_mdp regime has high skill-NA support loss (19/31, see Cut C below) at threshold=50 — early-season picks (March + early April) cannot accumulate 50 prior PAs in 2026 alone. This is honest skill-NA, not a methodology issue, and is reported transparently rather than smoothed.

### What this captures and does NOT capture

`batter_skill_quartile` is **season-to-date current-season skill, NOT career.** With 2026-only PA data, a Q4 batter on 2026-04-15 had at most ~150 prior PAs of 2026 baseball; that is a noisy estimate of the player's underlying skill. Career data (multi-season) would tighten the quartile boundaries and stabilize within-season volatility. Computed `park_factor` (replicating `src/bts/features/compute.py:394-406` against the canonical pick stream) is also deferred. Both belong in a future P that's gated on adding prior-season PA history to the source data.

The cut combined with `is_park_driven` is an **environmental-leverage × current-skill-tier proxy**, not a feature-attribution measure. It does not test "did the model pick this batter because of park-environment features rather than batter-skill features." That counterfactual requires SHAP / model rerun work and remains out of scope.

## Cut C — regime × slot × is_park_driven × skill_quartile

The full table is reproduced in the script's `--summary` output. Key observations from the target stratum (`post_pooled_mdp_pre_bpm`, n=30):

```
slot          env                Q     n hits   rate  mean_p     gap  note
---------------------------------------------------------------------------
primary       park_driven        Q3    2    1  0.500   0.739  +0.239 exploratory
primary       park_driven        Q4    2    1  0.500   0.747  +0.247 exploratory
primary       not_park_driven    Q2    2    2  1.000   0.723  -0.277 exploratory
primary       not_park_driven    Q3    1    1  1.000   0.723  -0.277 exploratory
primary       not_park_driven    Q4    8    6  0.750   0.766  +0.016
double_down   park_driven        Q3    1    1  1.000   0.708  -0.292 exploratory
double_down   not_park_driven    Q1    1    0  0.000   0.730  +0.730 exploratory
double_down   not_park_driven    Q2    3    2  0.667   0.726  +0.059 exploratory
double_down   not_park_driven    Q3    2    2  1.000   0.710  -0.290 exploratory
double_down   not_park_driven    Q4    8    3  0.375   0.731  +0.356
```

(Cells with n=0 or n<5 are reported in the full Cut C output for completeness; only the two n>=5 cells in the target stratum are interpretable per the methodology.)

The two interpretable cells:

- **primary × not_park_driven × Q4 (n=8)**: gap +1.6pp. Wilson 95% on the observed rate is [0.409, 0.929]; mean_p = 0.766 sits inside. The model's primary-slot picks of high-skill batters in non-park games appear calibrated on this cell.
- **double_down × not_park_driven × Q4 (n=8)**: gap +35.6pp. Wilson 95% on the observed rate is [0.137, 0.694]; mean_p = 0.731 sits outside the upper bound. This is a **calibration warning** — useful as a heuristic flag for tracking, but NOT a formal rejection of "the cell is calibrated." Wilson here is a binomial interval on a single post-hoc-selected cell from a multi-way cut with heterogeneous row probabilities; ~24 cells were reported in Cut C, and one cell falling outside its Wilson bound is roughly what we'd expect under multiple comparisons even if every cell were truly calibrated. The same player population (Q4 not_park_driven) appears calibrated in the primary slot — that within-skill, within-env contrast is part of what makes the cell worth tracking, but does not by itself license a formal claim at this n.

The strategic-question target cell — Q1 × park_driven × any slot — has **n=0** in the target stratum. Production did not pick a low-skill batter in a park-driven game between 2026-04-15 and 2026-04-30. The hypothesis cannot be tested with current data.

### Skill-NA exclusion

Cut C reports an explicit `(skill-NA, excluded)` line for cells where rows fell below the threshold:

```
pre_pooled_mdp     primary     park_driven       NA    3
pre_pooled_mdp     primary     not_park_driven   NA    7
pre_pooled_mdp     double_down not_park_driven   NA    7
```

19 of 31 pre_pooled_mdp rows are skill-NA — early-season picks before the picked batter accumulated 50 PAs. post_bpm and post_pooled_mdp_pre_bpm have 0 skill-NA (all picked batters had >= 50 prior 2026 PAs by their pick date).

## Interpretation

### What the DD × Q4 × not_park_driven finding might mean

Two readings:

1. **Candidate structural DD-selection mechanic biased toward high-skill non-park batters**: when the strategy picks a Q4 batter as DD, it may implicitly rely on long-run skill signal (career-or-season hit rate) more than situational factors. For a single-game outcome the long-run signal could overstate the bet. Under this reading the DD-selection logic might eventually benefit from an adjustment for high-skill non-park-driven candidates — but framing this as a candidate hypothesis to track, not as a verdict the current sample supports.

2. **Sampling artifact at n=8 with multiple-comparisons context**: this is one cell out of ~24 reported in Cut C; one cell at the 95% Wilson bound is roughly the false-positive rate we'd expect under multiple comparisons. The signal could vanish at n=20+.

To distinguish: track DD × not_park_driven × Q4 specifically as more post_bpm picks resolve. Currently post_bpm has only 2 not_park_driven Q4 DD picks (both hits, gap −0.296). The post-pooled-MDP stratum is closed; only post_bpm grows.

### Strategic-question status

The original strategic-question hypothesis cannot be evaluated on this sample because the predicted cell has n=0. Two possibilities for moving forward:

1. **Wait for the cell to populate**: as more post-bpm picks resolve, eventually some will be Q1 × park_driven. We currently have zero such picks — the strategy may not be selecting low-skill batters at park-driven venues at all, which would itself be a finding.
2. **Loosen the proxy**: treat "low-skill" as Q1+Q2 combined and "park-driven" as a continuous park_factor (P1.5 work). Coarser cells may appear sooner.

Neither is urgent; the artifact + cut now exists to revisit when the cell populates.

## Interpretation guardrails

1. **`is_park_driven` × `batter_skill_quartile` is a proxy, not feature attribution.** P0's env caveats apply unchanged; P1 adds a season-to-date skill caveat (not career skill).

2. **Single n=8 cell with mean_p outside the Wilson interval is a calibration warning, not a formal test.** Wilson 95% is a binomial interval on the observed rate, asymptotic on small n, computed cell-by-cell without multiple-comparisons adjustment, and applied here to a post-hoc-selected cell from a multi-way cut with heterogeneous row-level probabilities. Treat the cell as a heuristic flag worth tracking — not as evidence the model is overconfident in this cell. Track over n=15-20 before considering any DD-mechanic change.

3. **post_bpm strict-current-model regime is still n=7.** No cell-level claim is supportable under post_bpm in this cut; all 7 rows land in Q3 or Q4 with no Q1/Q2 representation. Re-run when the regime grows.

4. **Skill-NA support loss in pre_pooled_mdp (19/31)** is honest — early-season picks lack 50-prior-PA support. Reported transparently in the `(skill-NA, excluded)` line; not a methodology gap.

5. **Coors hardcode and indoor convention from P0 unchanged.** No changes to env-attribution logic in P1.

## What this memo does NOT say

- It does NOT propose a deploy change. The methodology stack still hasn't authorized a production decision; the DD × Q4 cell is a warning to track at n=8 and does not license a strategy edit.
- It does NOT falsify the strategic-question hypothesis. The hypothesis remains untested because production did not produce the predicted-cell observations.
- It does NOT use career-skill data, computed `park_factor`, or counterfactual feature attribution.
- It does NOT pool primary + DD slots when reporting cells; the slot decomposition is load-bearing for the DD finding.

## What this memo establishes

- A canonical artifact at `data/validation/realized_picks_canonical_2026-05-05_p1.parquet` (74 rows, 25 columns including the 3 new skill columns). Reproducible from the script's `--summary` against synced inputs:
  ```
  UV_CACHE_DIR=/tmp/uv-cache uv run --extra model python scripts/canonicalize_realized_picks.py \
    --picks-dir /tmp/realized_picks_input \
    --pa-path /tmp/pa_2026_fresh.parquet \
    --output data/validation/realized_picks_canonical_2026-05-05_p1.parquet \
    --summary
  ```
- The strategic-question target cell (Q1 × park_driven) demonstrably has n=0 on this sample — the hypothesis is unevaluable, not falsified.
- A first-pass DD × Q4 × not_park_driven calibration-warning observation, with explicit n=8 / multi-comparison / sampling-stratum caveats. The memo does not call this a verdict; it calls it a heuristic flag worth tracking.
- A no-same-day-leakage skill column with deterministic tie-breaking, additive to P0's env attribution. Tests at `tests/scripts/test_canonicalize_realized_picks_p1.py` cover the helper, integration, and dtype round-trip; the P0 file at `tests/scripts/test_canonicalize_realized_picks.py` covers the env attribution unchanged.

## What's next (recommendations, not commitments)

1. **Re-run as more post-bpm picks resolve.** The DD × Q4 × not_park_driven cell will gradually accumulate post-bpm rows; we don't forecast a per-week rate (depends on schedule, lineup-confirm timing, and which DDs fall into Q4 not_park_driven). Once the cell has more support — heuristic threshold n~15 — the calibration warning either survives or doesn't. Until then it stays a watch item, not a verdict.

2. **Defer the P1.5 candidates** (computed `park_factor`, career skill data). The current P1 cut is sufficient to track the DD × Q4 signal; tightening the proxy is only worth it if the signal survives n growth. Build the bigger P only after deciding it's worth investigating.

3. **Track Q1 × park_driven cell occupancy.** It currently has n=0 across all three regimes. If production keeps not picking low-skill batters in park-driven games, the strategic-question hypothesis is "untestable on production behavior" — a different finding than "hypothesis not supported." That finding would itself motivate a separate audit (does the strategy avoid Q1 × park batters because the model under-predicts them, even when warranted?).

4. **Defer DD-selection mechanics audit until n grows.** P0 and P1 have now narrowed the DD signal location twice (slot → not_park_driven → not_park_driven Q4). At n=8 we should not commit to a DD-mechanic investigation; at n=15+ we should.

5. **No methodology-stack PRs needed for this analytical surface.** The validation infrastructure (#9-#15) is sufficient for what we've done; the next big methodology candidate (e.g., #4 SAVI e-values, #1 MDP CVaR) is not gated on more α work and can be sequenced independently.
