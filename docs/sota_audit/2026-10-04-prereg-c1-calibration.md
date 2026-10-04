# C1 rank 4a registration: one calibration map (identity vs a regularized intercept shift)

**Status:** FROZEN 2026-10-04 after Codex trio design r1 **SIGN WITH EDITS** (A-E1–A-E4 and X-E1 applied verbatim by script; the first two plain-words items keep their bullet markers; review `docs/audit/2026-10-04-c1-trio-design-codex-r1.md`). Pace rule: no further design rounds.
**Cycle:** C1 (`docs/sota_audit/2026-10-04-c1-cycle-index.md`).
**Plan row 4a:** identity vs one regularized intercept map (a slope challenger only with support); the gate is a held-out proper-score improvement; kill if proper scores don't improve.
**Exposure:** row X-32, to be published before any 2027 outcome is read. No 2026 outcome is used (D3 = RESERVE); 2026 only motivated the test.

## In plain words
- **The motivation.** W1.3's restricted 2026 primary support had a candidate-row-weighted stated-minus-realized point residual of +0.031, and a native rank-1 point residual of +0.060 on a different population/weighting. These descriptive figures motivate testing the registered calibration slot; they do not establish a uniform or stable bias on the new equal-date population.
- **The test.** We fit one logit intercept on fixed 2027 contest dates 1–30 and test it on fixed dates 31–90. The fitted map may raise or lower probabilities; it is not a constant percentage-point correction.
- **What it doesn't do.** It cannot change which batter ranks first, because the shift is monotone. Whether it should change skip or double decisions is a separate question with its own gate.

## 1. The map

Candidate: g_a(p) = logistic(logit(p) + a), with one intercept and a Normal(0, 0.5²) prior. Identity is the comparator. For fitting, p is clipped to [10⁻¹⁵, 1−10⁻¹⁵]; the same clipping convention is used for both arms' log loss. Invalid, nonnumeric, boolean, non-finite or out-of-[0,1] probabilities are excluded and counted before clipping. For displayed mapped probabilities, g_a(0)=0 and g_a(1)=1 by continuity.

The MAP maximizes Σ_d [(1/n_d) Σ_i {y_di log g_a(p_di) + (1−y_di) log(1−g_a(p_di))}] − a²/(2·0.5²), summing over eligible fit dates. Date means are summed, not averaged; the weights are not rescaled to the candidate-row count. This is a fixed weighted likelihood with a regularizing prior, not an independently calibrated posterior uncertainty claim. Report the MAP; any normalized weighted-likelihood posterior interval is descriptive and is labelled as such.

No slope challenger is included. W1.3's D slope interval covers 1 but does not establish absence of slope distortion. Intercept-only is the deliberately restricted candidate specified by plan row 4a. The 2026 residuals motivate this test; they establish neither a uniform offset nor a validated correction. Ranking and tie order are taken from the original eligible forecast pool before outcome exclusions.

## 2. Population and outcomes

The daily source is the last retained served-slate write on each fixed contest date, frozen and hashed before outcome joining. This is a retained forecast surface, not evidence that it produced the final pick or was read at the final decision.

Include a row only when its archived write time is strictly before that game's run-known start minus picks.SUBMISSION_CUTOFF_MIN, and an archived pregame-status observation at or before that write establishes eligibility. Missing start/status evidence, started/postponed games, conflicting or duplicate (date,batter_id,game_pk) identities and invalid probabilities are excluded and counted. Later known game times or starters are not substituted into forecast inputs.

Known hit/no_hit outcomes use the W1.2 PA-event definition, with resumed portions excluded, and require an independently complete settled scoring source. A partial game with PA and zero hits is unknown, not no_hit. no_pa also requires complete scoring support; no_pa and unknown are excluded and counted. The pinned outcome reader is subordinate to these completeness checks.

Each date contributes its mean over known eligible candidate rows; dates contribute equally. Select the original-probability rank-1 row from the eligible pool before joining outcomes, preserving stored order for ties. A no_pa/unknown rank-1 row excludes that date from rank-1 metrics and is never replaced by another batter. This is forecast-pool rank 1, not a claim of contest entry or final policy selection.

## 3. Split and support
The numbered dates are the official 2027 contest calendar dates on which MLB games are scheduled, frozen before the first study capture under checklist A3. Numbering starts at the first such date and advances whether or not a slate is present; missing dates never shift or extend the windows. Fit dates are 1–30; test dates are 31–90.

At 08:00 ET after date 40, freeze fit inputs/outcomes and fit once if at least 25 fit dates have known eligible rows; otherwise record inconclusive (insufficient support). No fit date outside 1–30 is admitted. The fit job is declared 0.25 CPU-hours through the C1 launcher.

At 08:00 ET after date 90, freeze evaluation inputs/outcomes and analyze once. Require at least 50 scoreable test dates for the primary and 50 known original rank-1 dates for the guardrail; otherwise record inconclusive (insufficient support). Unknown rows at the freeze remain unknown; there is no extension, refit or second evaluation. The evaluation job is declared 0.25 CPU-hours through the launcher. No interim held-out metrics are inspected.

Report all scheduled, captured, eligible, primary-scoreable and rank-1-known date counts, plus row exclusions by reason and projected/confirmed state. The calendar-end C1 stop takes precedence if a window is unfinished.

## 4. Metrics and thresholds (fixed before any 2027 outcome)
- Primary: mean of the test dates' within-date log-loss differences, map minus identity. Resample this date-difference vector with replacement 10,000 times at seed 20270101; each sampled occurrence retains its multiplicity. Use the 2.5th/97.5th percentiles as the 95% interval. Condition on the once-fitted map; this interval excludes fit uncertainty and cross-date dependence.
- **Dispositions:**

| Disposition | Condition |
|---|---|
| **Positive** | difference ≤ **−0.001** nats and the interval's upper bound < 0, **and** the guardrail holds |
| **Negative** | difference ≥ 0 |
| **Inconclusive** | anything else |

- Guardrail: the 95th percentile of the corresponding 10,000-draw bootstrap of original rank-1 date differences must be ≤0.005 nats. Rank-1 draws use the fixed known-winner support and seed 20270103. A point estimate alone cannot pass the guardrail. Positive requires the registered primary practical threshold and upper bound plus this guardrail; difference ≥0 is negative; every other case is inconclusive.
- The −0.001-nat practical threshold and 0.005-nat regression limit are fixed judgments, not measured effects. A hypothetical uniform 0.03 correction near p=0.7 yields approximately 0.00214 nats; the descriptive 2026 residual does not establish that assumption on this equal-date population.
- Precision rationale (assumptions only): for 60 independent paired dates, a normal sign-detection calculation at two-sided 5% and 80% power gives MDE ≈2.802·σ_date/√60. Assumed date-difference SDs of 0.005, 0.010 and 0.020 imply MDEs about 0.00181, 0.00362 and 0.00723 nats. These are sensitivities, not measured variance or power. They do not certify power for the full practical-threshold/guardrail gate; an effect exactly at the point threshold has approximately a one-half probability of clearing that point condition under symmetric sampling. The study can be inconclusive despite a useful effect. Do not change thresholds or enlarge the window after seeing outcomes.
- **Secondary, descriptive:**
  - Brier difference;
  - stated − realized, before and after the map, on all rows and on rank-1 rows;
  - a reliability table by decile of p;
  - the fitted a, with its fit-window posterior interval.

## 5. Family, stopping and missingness
- **Family:** one primary comparison.

## 6. What a positive result does and does not permit
- **What it establishes:** a better forecast on the test window; that alone.
- **It cannot change ranking.** The map is monotone, so the top pick is unchanged.
- **Why it can still change play.** The skip and double rules use probability thresholds, so the map could alter play. Before it may affect picks, one of two paths goes to Eric under D7:
  - **(i) consistent remapping:** the policy's boundaries move with the map, so behaviour is essentially unchanged and only reported probabilities change;
  - **(ii) the map applied against the existing boundaries:** this is a policy change, evaluated with the 4b-style replay gate before approval.
- **Interaction with 4b:** if Eric adopts the 4b policy, its rates and bins are re-estimated under whichever forecast is live. The two must not be combined in one A/B.

## 7. Compute and execution
- **Code:** `scripts/audit/c1_r4a/`, written test-first: the map fit, the metrics and the bootstrap, against fixtures.
- **Inputs:** the served slates and the MLB feeds that production already archives. No new acquisition.

## Limits
- **One season, one window.** The test covers contest dates 31–90 of 2027 only, and a calibration error can drift.
- **Ranking:** the map cannot fix ranking; within-slate AUC was 0.566 in 2026.
- **The motivating gap** was descriptive (W1.3), not a validated bias.

## Freeze manifest and cross-design rules (trio review X-E1)
Before each authorized fitting/evaluation/diagnostic run, publish the appropriate exposure row and an outcome-free freeze manifest: reviewed registration/code commit and hashes; old serving recipe, model-training/retraining schedule, active blend and aggregation/fallback definitions; configuration/environment including calibration/deterministic/seed flags; fixed calendar/as-of/eligibility rules; and count/identity/reader artifacts as applicable. Future forecast/model/input hashes are recorded with each immutable capture. Pin/hash the exact consumed fitting/outcome/input bytes at the declared freeze, and parse those same bytes. A missing pin, mismatched hash, unsupported schema or unregistered recipe change refuses acceptance; naming a directory is not a pin. Ordinary model retraining under the frozen schedule is allowed and its artifact hashes are recorded. A recipe change does not trigger a post-result refit, window reset or silent pooling; it is reported and the affected study is inconclusive pending a separately approved prospective registration.

X-32 covers 4a's fit and test; X-34 covers rank 3's historical fitting and prospective evaluation; X-33 covers only the outcome-free broader-field/identity diagnostic. The index distinguishes X-33 from receipt-only capture. These rows are published before their respective reads, with no 2026 candidate outcome test. All new artifacts/readers preserve both plain and gzipped static input support and the declared missingness/fallback rules.

Each forecast study has one fixed primary comparison and its declared regression gate. Their shared dates create dependent evidence, not replication; secondary metrics are descriptive and cannot select another map/count specification or a combination. Keep 4a, rank 3 and 4b comparisons separate under D2. No fitted 4a output is supplied to rank 3 and no rank-3 output is supplied to 4a. Independent result acceptance and Eric's D7 approval precede any named production change; applicable policy replay and D1 trade approval remain separate. The approved C1 launcher, cumulative caps, sleep-window/production-safety and calendar stops apply. An unresolved rank-1 403/429 stop pauses C1 until Eric's recorded resumption and keeps capture separately disabled until his recorded reset.
