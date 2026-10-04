# C1 rank 4a registration: one calibration map (identity vs a regularized intercept shift)

**Status:** design rev 1, 2026-10-04, for Codex design review (at most 2 rounds, then freeze).
**Cycle:** C1 (`docs/sota_audit/2026-10-04-c1-cycle-index.md`).
**Plan row 4a:** identity vs one regularized intercept map (a slope challenger only with support); the gate is a held-out proper-score improvement; kill if proper scores don't improve.
**Exposure:** row X-32, to be published before any 2027 outcome is read. No 2026 outcome is used (D3 = RESERVE); 2026 only motivated the test.

## In plain words
- **The motivation.** In 2026 our stated chances ran higher than what happened: +3.1 points across served candidates, +6.0 at the top pick (W1.3, descriptive). A calibration map shifts every probability down by one fitted amount.
- **The test.** We fit that one number on the first 30 contest days of 2027, then check on the next 60 days whether it makes the probabilities more accurate.
- **What it doesn't do.** It cannot change which batter ranks first, because the shift is monotone. Whether it should change skip or double decisions is a separate question with its own gate.

## 1. The map (ruling R1)
- **Candidate:** logit(p′) = logit(p) + a. One intercept a, fitted by maximum a posteriori with a Normal(0, 0.5²) prior on a, so small samples shrink toward identity.
- **The fit:** the equal-date-weighted log-likelihood over the fit window.
- **No slope challenger.** W1.3 found D's recalibration slope 1.016 [0.857, 1.179], which covers 1, so there is no support. Its intercept was −0.144 [−0.248, −0.042].
- **Baseline:** identity (p′ = p).

## 2. Population and outcomes
- **Rows:** every served candidate on each 2027 contest date's served slate (`data/picks/slates/<date>.json`, the last write that day, as in W1.2 surface D).
  - The probability is the served one, unchanged.
  - Invalid or non-finite probabilities are excluded and counted.
- **Outcomes:** the W1.2 definitions, with the resumed portion of suspended games excluded.
  - **Known** = hit or no_hit.
  - no_pa and unknown are excluded and counted.
  - Graded by the frozen W1.2 outcome code (`scripts/audit/benchmark_bridge/core.py`), pinned at its commit.
- **Weighting:** equal-date means of within-date means, as in W2.3.
- **Secondary population:** each date's rank-1 row, the population the policy reads.

## 3. Split (fixed now; 2027 dates are counted from the first contest date with a served slate)
- **Fit window:** contest dates 1–30.
- **Test window:** contest dates 31–90.
- **One analysis,** run once the test window's last date is graded (after the 08:00 ET next-day cutoff). No interim looks and no refit.

## 4. Metrics and thresholds (fixed before any 2027 outcome)
- **Primary:** the test-window difference in equal-date mean log loss, map minus identity, with a 95% whole-date bootstrap interval (10,000 draws, seed 20270101).
- **Dispositions:**

| Disposition | Condition |
|---|---|
| **Positive** | difference ≤ **−0.001** nats and the interval's upper bound < 0, **and** the guardrail holds |
| **Negative** | difference ≥ 0 |
| **Inconclusive** | anything else |

- **Guardrail:** the test-window rank-1 rows' equal-date log loss is not worse by more than 0.005 under the map.
- **Why −0.001 (reasoning only, not a measurement).** A persistent +0.03 offset at p ≈ 0.7 is worth about 0.03² / (2 · 0.7 · 0.3) ≈ 0.002 nats per row to correct. The threshold is half that: the smallest gain worth carrying a map.
- **Secondary, descriptive:**
  - Brier difference;
  - stated − realized, before and after the map, on all rows and on rank-1 rows;
  - a reliability table by decile of p;
  - the fitted a, with its fit-window posterior interval.

## 5. Family, stopping and missingness
- **Family:** one primary comparison.
- **Stopping:**
  - one map, fitted once;
  - if the fit window has fewer than 25 graded dates by contest date 40, the study is recorded **inconclusive (insufficient support)** and not extended.
- **Missingness:**
  - dates without a served slate are skipped and counted;
  - row exclusions are counted by reason.

## 6. What a positive result does and does not permit
- **What it establishes:** a better forecast on the test window; that alone.
- **It cannot change ranking.** The map is monotone, so the top pick is unchanged.
- **Why it can still change play.** The skip and double rules use probability thresholds, so the map could alter play. Before it may affect picks, one of two paths goes to Eric under D7:
  - **(i) consistent remapping:** the policy's boundaries move with the map, so behaviour is essentially unchanged and only reported probabilities change;
  - **(ii) the map applied against the existing boundaries:** this is a policy change, evaluated with the 4b-style replay gate before approval.
- **Interaction with 4b:** if Eric adopts the 4b policy, its rates and bins are re-estimated under whichever forecast is live. The two must not be combined in one A/B.

## 7. Compute and execution
- **Code:** `scripts/audit/c1_r4a/`, written test-first: the map fit, the metrics and the bootstrap, against fixtures.
- **Runs on the box** through the C1 launcher (`c1-r4a-*`): one fit job after date 30 and one evaluation job after date 90, **each declared 0.25 CPU-hours.**
- **Inputs:** the served slates and the MLB feeds that production already archives. No new acquisition.

## Limits
- **One season, one window.** The test covers contest dates 31–90 of 2027 only, and a calibration error can drift.
- **Ranking:** the map cannot fix ranking; within-slate AUC was 0.566 in 2026.
- **The motivating gap** was descriptive (W1.3), not a validated bias.
