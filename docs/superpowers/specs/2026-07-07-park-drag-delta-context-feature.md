# Spec: `park_drag_delta` context feature (shadow stack) — v2

**Status: DRAFT v2 — not built, not committed to any branch. Codex gpt-5.5 design
review COMPLETE 2026-07-07 (17 findings; full output:
`~/projects/juiced-ball-analysis/codex-review-2026-07-07.txt`); accepted findings
folded in below, disagreements resolved by adoption where noted. Does NOT displace
the teed-up miss-distance session; slot per Eric.**

## Motivation

The 2026 ball changed mid-season and rolled out **park-by-park over ~8 weeks**
(analysis + data: `~/projects/juiced-ball-analysis`, README + `data/rollout_2026.csv`).
League game-level drag stepped −0.0103 Cd (~+5 ft carry at 100 mph) around May 24;
per-park deltas range from −0.025 (Globe Life, loanDepot, Kauffman) to **0.000**
(Rogers, Progressive, Target, Angel, Camden, GABP — still on old stock as of Jul 6).

The model's only park input, `park_factor` (`features/compute.py:393-410`), is an
expanding venue hit-rate ratio over 2019+ history — effectively frozen against a
40-date regime change. Serving takes `.last()` per venue (`model/predict.py:282`).
`weather_temp`/wind carry no ball information.

**Effect size honesty:** league BABIP flat; carry converts deep flyouts to HRs →
~+2% relative per-PA hit rate at fully-switched parks ≈ **+0.4-0.6pp on P(≥1 hit)**
for a 4-PA hitter as the spread between most/least switched parks — decision-relevant
at pick margins, at the edge of the harness's detection floor. Maintenance of a
stale input, not new alpha. The effect is fly-ball-concentrated: a main park effect
mildly over-adjusts GB/contact hitters and under-adjusts air-ball power hitters
(Codex #15) → the eval must report **stratified by batter air-ball exposure**;
interactions are v2+ with their own multiple-testing plan.

## Feature definition (v2 — revised shape per Codex #7)

Per (venue_id, prediction_date), from four-seam pitch-level Cd (Nathan 9P method,
elevation + game-time-temp adjusted; game-level real SD ~0.009 vs regime steps
0.010-0.025 → detectable in ~3-5 venue-games):

- **Primary candidate: `park_drag_delta` = cd_roll15_asof − cd_anchor_asof**, where
  the anchor is the venue's early-season baseline (mean of its first 10 venue-dates,
  frozen thereafter). Rationale: the v1 expanding baseline *absorbs* a persistent
  regime shift, so the delta decays toward zero while the regime persists — it
  detects "recently changed," not "currently shifted" (Codex #7, accepted). The
  anchored delta holds its level for the rest of the season.
- **Comparison variants carried in the table:** expanding-baseline delta (v1 shape),
  and a CUSUM/EWMA-style state if the eval warrants it.
- **Uncertainty shrinkage in the producer, not a clamp** (Codex #9): delta ×
  n-based shrinkage weight, applied identically for training and serving.
  No serving-only transformations of any kind.
- **Scope v1 = mid-season regime changes only** (Codex #8): early-season rows are
  NaN by construction (LightGBM-native); season-boundary ball changes are explicitly
  out of scope (`drag_delta_xseason` remains a caveated diagnostic column, lean-no
  for v1).
- Known ceiling: the ±4-5 ft box-to-box game lottery is unforecastable; only the
  park rolling state is capturable.

## Data contract (v2 — per Codex #2, #13, #14, #16)

- Producer: `~/projects/juiced-ball-analysis/build_feature_table.py` →
  **BTS-facing export** with the exact serving schema: `venue_id` (int),
  `date` (naive Timestamp aligned to BTS officialDate semantics),
  `park_drag_delta` (+ variant columns), one row per **venue_id × prediction_date**
  (not per venue home-date; see serving below), uniqueness enforced at build.
- **Diagnostics (non-as-of columns like `cd_venue_date`) ship in a separate file**;
  BTS reads an allowlist from the export only (Codex #16).
- **Manifest + freshness metadata** written atomically alongside: max source
  game_date, game_pk coverage through D−1, row counts, per-venue coverage
  (Codex #2, #6). Consumer suppresses (all-NaN) + health-alerts when stale.
- Daily refresh job (if/when box-deployed) follows house scraper hygiene:
  `endpoints.browser_headers`, **403/429 → abort kill-switch**, jittered gaps,
  atomic writes + status file (Codex #13; Savant 403s default UAs from datacenter IPs).

## Integration (v2 — per Codex #1, #3, #4, #5, #17)

- **Production-safety (supersedes v1's "zero production-pick risk" claim, which was
  wrong):** `compute_all_features()` computes CONTEXT_COLS unconditionally, so this
  join sits upstream of the production pick path. Requirements: no network in the
  hot path; optional local artifact with **all-NaN fallback + health alert** on
  missing/stale/schema-drifted table (Codex #3).
- **Serving as-of parity (the critical one):** training joins venue-date row D whose
  shift(1) includes the last completed venue game; a `.last()`-style lookup serves
  the *previous* row and is off by one venue-date (Codex #1). Fix: the export is
  materialized per **prediction_date** (every date × every venue, computed from
  games with `game_date < prediction_date`), so training join and serving lookup
  read the identical value. Hard filter is `source_game_date < prediction_date` —
  never "latest available" — which also kills the doubleheader same-date leak if the
  table refreshes intraday (Codex #5). **Train/serve parity test + same-venue
  doubleheader test required.**
- **Shadow identity:** adding a 5th context col invalidates the context stack —
  bump `SHADOW_MODEL_NAME` (context_stack_shadow_v2), include a feature-set hash in
  the `blend_{date}_shadow.pkl` cache key, reset shadow eval history per
  `shadow_eval.py:642-645` (Codex #4).
- Add a populator test asserting every `FEATURE_COLS + CONTEXT_COLS` column is
  computed and non-degenerate in `predict()` (Codex #17).

## Evaluation gate (v2 — restructured per Codex #10, #11, #12)

The v1 gate was post-selection (2026 motivated the feature) and its falsification
framing was wrong (real park drift exists in stable seasons — our own 2025 control
showed χ²=84/29df — so "helps in 2023-25" ≠ leakage and "no help" ≠ required).

1. **Primary gate = live-forward shadow accumulation** on a frozen, pre-registered
   implementation (the regime variance is forward-live right now). Promotion
   requires its own pre-registration per `shadow_eval.py:645`.
2. **Backtests are supporting evidence, not the gate:** 2021 **and 2022** (both
   multi-ball years) walk-forward; 2026-to-date reported but labeled post-selection.
   Multi-seed (BTS_LGBM_RANDOM_STATE sweep), estimated_pa profiles only.
   ⚠ Expectation-setting (validation recompute, 2026-07-07): the anchored
   within-season feature shows **no elevated park-level variance in 2021/2022** —
   season-start/league-wide changes are absorbed by the per-season anchor by
   construction, so these backtests may be structurally null for this feature
   shape regardless of mechanism. A null there is weak evidence either way; the
   forward shadow carries the decision. Also: 2026's elevated feature variance
   decomposes mostly into temporal (league + within-park time) components; the
   cross-park differential at matched dates is only modestly above 2021 — the
   uniquely-park-informative slice is thinner than the headline SD suggests.
3. **Live-boundary replay** (Codex #12): production-pipeline replay with historical
   data cutoffs (table state as of each morning), projected-lineup path, estimated_pa.
4. **Mechanism-specific nulls** replace naive falsification windows (Codex #11):
   within-date/venue permutation, sign-flip, coverage-matched availability mask.
5. **Head-to-head competitor:** baseline + a *same-recency rolling outcome park
   factor* (rolling-15 venue hit rate). The feature's whole claim is lower-variance
   regime detection than outcomes; if rolling outcomes match it, ship that instead
   (no new data dependency).
6. Metrics: top-k (P@1/P@k) + within-day pair-count-weighted AUC, plus the
   air-ball-stratified readout (Codex #15).

## Codex findings disposition (2026-07-07)

Accepted: #1 (serving off-by-one → prediction-date-materialized export + parity
test), #3 (production hot-path safety), #4 (shadow version/cache/history), #5 (DH
refresh-timing), #6 (freshness manifest), #7 (anchored baseline primary — design
disagreement, adopted), #8 (v1 scope = mid-season), #9 (shrinkage in producer, no
clamp), #10 (forward shadow = primary gate; 2022 added), #11 (mechanism nulls),
#12 (live-boundary replay), #13 (fetch hygiene for the box job), #14 (BTS-facing
schema), #15 (stratified eval), #16 (diagnostics split), #17 (populator test).
Partially moot: #2 (artifacts "absent" — the 2021-2025 build was in flight during
review; its substance — manifest + schema validation — is adopted above).
No findings rejected.

## Remaining open questions

- Rolling window 15 venue-dates vs 8-10 (noise/latency tradeoff) — resolve in eval.
- Anchored delta vs CUSUM/EWMA state as the shipped shape — carried as variants.
- Anchor robustness when a regime change occurs *inside* the anchor window
  (first ~10 venue-dates) — accepted v1 limitation, documented.

## Pre-merge implementation review disposition (Codex gpt-5.5, 2026-07-07)

7 findings on the built branch; fixed in the hardening commit: **#2** in-process
train/serve skew (module cache never invalidated in the days-long daemon while
predict() reloaded fresh) → both paths now share one mtime/size-aware cache
(`get_table`/`get_manifest`); **#3** silent no-coverage serving (prediction date
past the table's last materialized row returned None per venue with no warning
while the manifest freshness check still passed) → explicit coverage guard +
one loud warning per date; **#4** shadow cache hash covered column names only
(a same-day cache trained table-absent would be reused after the table appeared)
→ artifact fingerprint (mtime:size / "absent") folded into the filename hash;
**#5** v1 shadow history would count toward v2 review thresholds →
`shadow_model_version` stamped into shadow pick files by `save_shadow_pick`,
status/backfill loops filter to the current version (legacy unstamped = v1,
excluded); **#6** permissive normalization → tz-aware dates coerced to naive
midnight, non-integral venue_id rejected; **#7** test gaps → 8 new tests
(cache invalidation, table-appears, coverage warning, tz/venue hardening,
fingerprint-in-hash, version stamping/exclusion).

**#1 (artifact not deployable via existing prod paths) — accepted as a
pre-ARMING gate, with a disagreement on framing:** the branch is merge-safe
without the table (all-NaN + warnings; production FEATURE_COLS untouched), so
delivery is not a merge blocker — but DO NOT consider the shadow ARMED until:
(a) table shipped to `data/external/park_drag/` on the box, (b) the daily
producer/refresh job exists (browser-UA + 403/429 kill-switch per house rules),
(c) staleness is surfaced as a real health source, not stderr (the stderr
warnings are invisible in a daemon — Codex is right about that).

## Round-2 implementation review disposition (Codex gpt-5.5, final round)

5 findings, all addressed: **#1** save_shadow_pick's auto-stamp would have
PROMOTED legacy v1 files to v2 when check-results re-saved them after grading
→ stamping moved to creation time (`shadow_eval.stamp_shadow_version`, called
by the scheduler); save never stamps; legacy files keep version=None forever.
**#2/#3** mid-cycle artifact swap could train on table A and serve/hash table B
→ `pinned()` snapshot (reentrant contextmanager + `with_pinned_artifact`
decorator) freezes (table, manifest, fingerprint) for a whole cycle:
`run_pipeline` is pinned (train+serve one snapshot) and the scheduler's
`_run_shadow_prediction` is pinned end-to-end (predict_local_shadow's cache
path and the later provenance path always agree). **#4** `bts shadow-report`
was a third unfiltered glob → same version filter applied. **#5** empty-after-
normalization tables now rejected. Codex's one pushback — no real circular-
import risk in the round-1 lazy import — accepted, and made moot by removing
that import entirely. Review loop converged at 2 rounds per house cap;
remaining gates are operational (merge; arming checklist above).
