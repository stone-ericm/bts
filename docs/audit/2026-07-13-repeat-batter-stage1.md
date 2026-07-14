# 2026-07-13 — Repeat-batter conditioning, stage 1: mechanism unsupported, stage 2 not triggered

Tests the leading mechanism hypothesis from
`2026-07-13-dd-p-policy-value-sensitivity.md` finding 5 (run-structure
anti-persistence): if long runs concentrate on recency-hot batters whose
true rate sits below the form-chasing estimate, rank-1 picks that REPEAT
(batter already rank-1 on a recent prior slate day) should realize below
stated while fresh picks stay calibrated. Tested on every slate day rather
than the thin run tail.

**Pre-registered decision rule (in the script docstring, set before the
first run): proceed to stage 2 (run-conditional decomposition) only if the
repeat-vs-fresh gap difference is negative with a 95% date-cluster
bootstrap CI excluding 0 AND the direction holds in ≥4/5 seasons.
Outcome: NOT MET on any flag → stage 2 not run; the repeat-batter
mechanism is unsupported.**

Script `scripts/audit/repeat_batter_conditioning.py` (tests
`tests/scripts/test_repeat_batter_conditioning.py`; fast suite 1865→1871);
artifact `2026-07-13-repeat-batter-stage1.json` (input fingerprints, full
strata). Same estimated_pa profiles as the sensitivity analysis; clusters =
(season, date) so the 24 seeds' date reuse cannot manufacture precision
(21,888 rows, 912 clusters).

## Results

| flag | share | repeat gap | fresh gap | diff | 95% cluster CI | seasons neg |
|---|---|---|---|---|---|---|
| repeat_1 (prev slate day) | 24.3% | −4.1pp | −0.6pp | **−3.5pp** | [−9.0, +1.7] | 3/5 |
| repeat_3 | 39.6% | −3.5pp | −0.2pp | −3.3pp | [−8.0, +1.3] | 3/5 |
| repeat_7 | 50.3% | −2.6pp | −0.3pp | −2.3pp | [−6.9, +2.1] | 3/5 |

- Point estimates are directionally negative but the CIs all include zero,
  the per-season signs split (2021/2023 positive, 2022/2024/2025 negative —
  2025 dominates at −14.9pp on repeat_1), and the stated-p strata are
  inconsistent (middle strata negative, both extremes positive for
  repeat_1). Not the signature of a clean selection mechanism.
- **Live 2026 side-check does not corroborate the mechanism**: on the 65
  scored production primaries (slot grading via `build_slot_dataset.py`;
  contrast archived in the JSON artifact via `--live-csv`), repeat picks
  OVER-deliver — repeat_1 diff **+7.5pp** (n=21), driven by Arraez (22
  picks, realized 0.864 vs stated 0.797). Directional only at this n and
  one-batter-dominated, so it carries corroboration weight, not
  refutation weight. (The live DD-leg shortfall is a different slot;
  leg-side repeat conditioning was not run — n=42 slices too thin.)

## Honest scope of the negative

- **Unsupported ≠ refuted at small effect sizes.** The CIs exclude a repeat
  effect ≥ ~9pp but not a true −3-4pp. As an all-20-exposure illustration
  only: a uniform −3.5pp on every day of a 20-window gives a survivor ratio
  of (0.729/0.764)^20 ≈ 0.39 vs the observed ×0.28 (74% of the log deficit)
  — but actual tail exposure to repeats was never measured (stage 2 not
  run), so the contribution to finding 5 is unquantified. The single-
  mechanism story is **not supported under the pre-registered stage-1
  rule**: seasons disagree, strata disagree, and the live primaries do not
  corroborate it (r1 review edits).
- **Byproduct worth keeping — but NOT as finding-5 evidence: 2025 is an
  aggregate season calibration miss.** The 2025 profile season realizes
  0.699 on rank-1 vs ~0.77 stated — ~7pp overconfident across the board
  (fresh picks −3.7pp, repeats −18.6pp). Note this says nothing about
  finding 5 (r1#2): the permutation null already fixes each file's
  aggregate rate, so only *within-file temporal* structure could explain
  the window suppression, and that is unmeasured here. 2022 also cautions
  against a calibration story: it is the most window-suppressed season
  (1 observed vs 38.5 expected) while its fresh picks OVER-deliver
  (+3.5pp).

## Disposition

Finding-5 mechanism status: repeat-batter form regression **tested, not
supported under the pre-registered stage-1 rule** (this doc). Remaining
candidates: within-file temporal rate structure (unmeasured here — the 2025
aggregate miss is not evidence for it) and run-conditional miscalibration
not mediated by pick identity. No further mechanism work queued — the
finding's practical consequence (use realized-sequence replay, not iid DP,
for milestone values) stands regardless of mechanism, and the policy
conclusions of the sensitivity analysis did not depend on it. One review
round (gpt-5.6-sol xhigh): machinery validated (cluster resampling proven
algebraically exact, numbers reproduced); 5 wording/rigor findings adopted,
incl. this section's own earlier "dead"/"regime evidence" overclaims.

## Reproduce

```
uv run pytest tests/scripts/test_repeat_batter_conditioning.py -q
uv run python scripts/audit/repeat_batter_conditioning.py
# live side-check: run scripts/audit/build_slot_dataset.py on the box,
# repeat flags over scored primaries in date order
```
