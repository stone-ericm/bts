# 2026-07-12 — DD-leg calibration: season analysis + monitoring coverage

Follow-up to the eve-of-break incident (same date, separate doc): the drift
CRITICAL's underlying signal was the double-down leg. Question set: is the
DD-leg shortfall a pipeline defect, a selection-bias property, or variance —
and why did no monitor say anything for two months?

Data: all 2026 pick files (box), per-slot grading identical to the shipped
`predicted_vs_realized` semantics. Dataset builder + analysis scripts in the
session scratchpad; threshold bootstrap versioned at
`scripts/audit/pvr_threshold_bootstrap.py`.

## Findings

**1. The "cold streak" framing was wrong — the shortfall is season-long in
aggregate (and temporally heterogeneous, not provably stationary).**
Season slots: primaries 51/65 = 0.785 realized vs 0.767 stated (−1.8pp,
over-delivering; non-Arraez subset +0.8pp — calibrated; Arraez 19/22 masks
nothing). DD legs: 25/42 = 0.595 vs 0.734 stated → **+13.9pp shortfall**
(z≈−2.0, exact tail P(hits≤25 | stated p) = 0.035). Monthly: May 16/28
(z=−1.9), June 7/7, July 2/7. Scan test: the worst 6-consecutive-DD window
(1 hit) has p=0.12 under stated p's and p=0.52 under the season-empirical
rate — the 7/07-7/12 run is an ordinary cluster inside the chronic level,
not an event. The original "~2.5σ cold streak" framing (post-hoc window) is
retired.

**2. Slot is the leading stratification candidate (exploratory).** In the
same [0.70, 0.75) band: primaries realize 0.762 at 0.734 stated (n=21);
DD legs realize 0.545 at 0.731 (n=33, +18.6pp). The direct two-sample
contrast is only ~1.7σ (Fisher two-sided ≈ 0.15) — directionally against a
band-wide p-scale defect, but not itself significant, and slot composition
(rank, opponent context) differs. Review r2#8 language correction: this is
a stratification lead, not a demonstrated slot-specific mechanism.

**3. The backtest finds no large intrinsic penalty — on the estimated_pa
basis.** Applying the exact production rule (rank-1 + best rank≥2 in a
different game) to all 120 estimated_pa profiles (24 seeds × 5 seasons):
21,787 simulated DD legs, gap +0.98pp (and +0.19pp within p<0.76;
per-profile gap IQR −0.002..+0.025; numbers independently reproduced in
review). Caveats (r2#5): the 21,787 rows are 24 seed-selections over 908
dates (2,927 distinct batter-dates, ~7.4× reuse) — date-clustered SE ≈
1.1pp, not the naive 0.3pp — and estimated_pa conditions on realized
participation, so decision-time lineup/pitcher composition effects present
in production are not fully represented. Strong negative evidence against a
large universal winner's-curse; not a full exoneration of the production
serving path.

**4. No DNP/PA/lineup-slot failure found.** All 17 season DD misses: batter
started, batted 1st/2nd/4th, 3-5 PA (7/12 shows pa=0 only because the
nightly parquet lags a day; live-feed grading is authoritative there). One
projected-lineup delivery among the misses (5/22). This rules out the
players-didn't-play class, NOT probability-serialization, serving-basis, or
stale-grade classes (r2 wording correction).

**Verdict: suggestive (~2σ against every reference), consistent with bad luck
at n=42; no defect found. Response = monitoring coverage + evidence
accumulation, NOT pipeline or policy surgery.** Policy materiality is
**unknown pending a value-function sensitivity** (r2#6: the 7/06
"strategy ≈ wash" replay consumed backtest-calibrated DD probabilities; a
real 59.5% DD marginal could flip Q(double)−Q(single) at streaks 1-2 even
though streak-0 DDs are structurally low-risk). Queued: deployed-policy
value contrast at streaks 0-2 over a DD-p sensitivity range — worth running
BEFORE the monitor escalates if DD density stays high. If the shortfall is
real, ~40 more legs push it past 3σ and the new monitor escalates; if luck,
it regresses.

## Why every monitor was silent (the coverage hole)

| monitor | design | why silent |
|---|---|---|
| `predicted_vs_realized` | drift (14d vs 28d) | chronic spans both windows → drift ≈ 0 (per-slot basis shipped in the incident batch fixes attribution, not chronicity) |
| `dd_pair_realized_shortfall` + residual | drift | same, by construction (docstring now says so) |
| `realized_calibration` | ABSOLUTE ladder — but only [0.75, 0.80) | 38/42 DD legs sit below 0.76 (mean 0.734): the DD band had **no absolute-level monitor at all** |

Pooling wouldn't have worked either: [0.70, 0.75) pooled across slots reads
+10.3pp (below the 15pp WARN bar) because calibrated primaries dilute the
DD signal — the bucket must be slot-aware.

## Changes (TDD; suite 1839 → 1845)

- `realized_calibration`: bucket spec generalized to a list with slot
  filters — `[0.75, 0.80) × both` (unchanged behavior) + **`[0.70, 0.75) ×
  double_down-only`** (new). Per-bucket alerts with distinct `incident_key`s
  (composes with the incident batch's same-day DM dedup).
- Slot grading priority now `slot_results` (production live-feed grading,
  authoritative) → PA-frame join → day-result proxy (primary slot on non-DD
  days only; the knowingly-biased legacy DD fallback is excluded instead).
- **Small-n guard with a catastrophic escape hatch**: CRITICAL requires
  bucket n ≥ 20 (at the 25pp bar, SE ≈ 10pp only from n≈20; a 2σ-ish n=8
  reading must not claim "real signal — investigate distribution shift")
  **OR** an exact Poisson-binomial tail ≤ 1e-3 under the stated p's (r2#4:
  0-for-8 at 0.73 has tail 2.8e-5 — a grading/serving pipeline failure must
  not hide behind the n gate; 2-for-8 at 0.0064 stays WARN). Validated
  against live data: as of 7/12 the DD bucket reads **+36.6pp at n=8, tail
  ≈0.03 → WARN** (was CRITICAL pre-guard — inconsistent with this doc's own
  verdict).
- `realized_calibration` added to `REPEATED_ATTENTION_WARN_SOURCES`: its
  WARNs previously never reached the DM digest regardless of persistence.
  Expected post-deploy behavior: WARN logged on day 1, attention digest DM
  from the second consecutive WARN day.
- `dd_pair`/`same_team_corr` docstring: chronic-blindness property recorded;
  no logic change (its acceleration + residual-correlation jobs stand).

## Threshold simulation (queued item from the incident review, closed)

`scripts/audit/pvr_threshold_bootstrap.py` v3 (v1's day-resampling
randomized the chronological clustering it claimed to preserve and ignored
the 35-calendar-day lookback — review r2#1-2; v2 fixed those but still used
resolved-day adjacency for "consecutive" and counted CRITICALs as WARN
crossings — review r3#1). v3 holds the season's OBSERVED schedule fixed (66
resolved days 03-29..07-12, 107 slots; input sha256 fc9257e39ba3b16c…,
builder versioned at `scripts/audit/build_slot_dataset.py`), simulates
outcomes under the calibrated null with the check's exact window semantics,
and evaluates on the PRODUCTION clock: every calendar day (health runs on
skip days too — windows slide/evict without new data), WARN-band-only
crossings (warn ≤ drift < critical), calendar-adjacent pairs. 20k trials:

| threshold | P(any check-day ≥ thr) | P(2 cal-consec WARN-band) |
|---|---|---|
| 0.08 (pre-review WARN) | 0.823 [0.817, 0.828] | **0.688** [0.682, 0.695] |
| **0.12 (pre-incident CRITICAL)** | **0.447** [0.440, 0.454] | 0.309 |
| **0.15 (new WARN)** | 0.209 | **0.126** [0.122, 0.131] |
| 0.20 | 0.039 | 0.019 |
| **0.25 (CRITICAL)** | **0.0037** [0.0030, 0.0047] | ~0 |

Three consequences: (a) the pre-incident 0.12 CRITICAL would have fired in
~45% of calibrated seasons — the incident-night CRITICAL was likely spurious
even before the DD-day confound; (b) the "0.08 WARN is fine because
attention needs 2 consecutive days" argument was WRONG — overlapping windows
make crossings serially persistent and the digest would false-fire in ~69%
of calibrated seasons, so **drift_warn moves 0.08 → 0.15** (~13%/season
digest rate; INFO at 0.05 keeps low drifts visible in logs); (c) 0.25
CRITICAL ≈ 0.4%/season stands. Caveats recorded in the script docstring
(independent same-day slots — different-game by construction; stated-p null;
one season's schedule; "every calendar day" ≈ the true health calendar).

## Open / deferred

- DD-leg evidence accumulation: the new bucket escalates automatically; no
  scheduled re-analysis needed. If it reaches CRITICAL (≥25pp at n≥20, or
  the exact-tail escape hatch),
  THEN a targeted investigation (opponent-pitcher mix, lineup-slot realized
  vs estimated PA, month structure) is warranted.
- `dd_pair` absolute pair-level tier: not added — pair-level absolute
  shortfall is largely implied by the marginals now monitored; the residual
  (correlation) term remains drift-based and would hide a chronic
  correlation shift. Accepted for now (same-team overlap is rare and
  separately checked); revisit if DD marginals normalize while day results
  stay short.
- Per-bucket threshold ladders (the 8/15/25pp ladder was calibrated on the
  75-80 bucket's typical n): single ladder kept; the n≥20 CRITICAL guard
  (with its exact-tail escape hatch for effectively-impossible readings) is
  the load-bearing protection.
