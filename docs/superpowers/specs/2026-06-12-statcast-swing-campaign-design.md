# Statcast swing-data campaign: miss distance, timing, mechanics — design

**Status: approved 2026-06-12 (Eric). Codex (gpt-5.5) design review incorporated
(metric hierarchy, screen reframe, period strata, controls).**

## Motivation & posture

Statcast began exposing per-pitch swing data (miss distance on whiffs +
all-swing bat-tracking) with coverage from the 2023 All-Star break. Eric's
direction: **wide and open-minded — throw every plausible use at the wall**
(matchups, slumps, timing, mechanics) with fleet compute as needed, and let
the harness decide. Honest prior: the model is near its top-pick ceiling
(three levers ruled out in June 2026 at ≤ +0.006 AUC each), so candidates are
interesting mainly as *orthogonal* signals, and "all negatives, documented"
is an accepted, valuable outcome. Discipline (selection/confirmation
separation, FDR, negative controls) is what turns wall-throwing into
knowledge.

## Data facts (verified 2026-06-12)

- **Per-pitch** (statcast search CSV; pybaseball — existing dependency —
  wraps it): `miss_distance` (inches from barrel, populated ONLY on whiffs,
  ~6% of pitches, verified mean ~2.9" on swinging strikes, ~9.7" on blocked
  chases), plus all-swing `swing_length`, `attack_angle`, `attack_direction`,
  `swing_path_tilt`, `intercept_ball_minus_batter_pos_{x,y}_inches`.
- **Leaderboard** (`/leaderboard/bat-tracking/swing-timing-miss-distance`,
  `?csv=true`, no auth; `type=batter|pitcher`, `season[]=2023..2026`,
  `pitchType[]`, `dateStart/dateEnd`): timing decomposition
  (early/on_time/late % + magnitudes), directional miss decomposition
  (x: tied_up/centered/flailed; z: over/lined_up/under), `competitive_percent`,
  `flawed_percent`, `n_swings`, `whiff_rate`.
- **Leaderboard aggregates are BANNED as model features** (season-to-date
  scrapes are not date-bounded → leakage). Features come from per-pitch data
  through date-level `shift(1)` machinery only. Leaderboards are reference/QA
  with a **tolerance protocol, never exact equality** (Savant applies its own
  tracking/competitive filters and denominators): identical date/season/
  player-type/pitch-type/minimum filters first, then Spearman ≥ 0.98 on
  n_swings and whiff rate, median absolute percent error ≤ 2% (p95 ≤ 5%),
  mean miss-distance median absolute diff ≤ 0.25–0.5"; proprietary buckets
  (competitive/flawed) are gross-sanity only.
- Density: starters induce ~5–8 measured whiffs/start → 30d pitcher windows
  hold ~30–50 swings. Pair-level (batter × pitcher) features are REJECTED on
  sparsity; zone-profile aggregates stand in for matchups.

## Candidate library (5 families, ~20–25 features before variants)

- **P — pitcher contact-suppression**: induced miss-distance 30g,
  induced competitive-miss %, vertical-attack profile (over/under% induced),
  primary-pitch-type variants.
- **B — batter near-miss quality**: own miss-distance 30g, flail %,
  competitive %, chase-miss magnitude.
- **T — timing/slump leading indicators**: timing drift (7g vs 60g deltas of
  intercept-y / late-miss magnitude), short-window trend slopes. The
  orthogonality bet: leading indicator vs our lagging rolling averages.
- **S — swing mechanics (all-swing, not whiff-conditional)**: swing_length
  drift, attack_angle consistency (std), path-tilt change.
- **M — zone-profile matchups**: batter vertical-miss vulnerability ×
  pitcher vertical-attack tendency (aggregate dot products, not pair history).

**Variant sweep axes (per family, screen stage):** windows 7/15/30/60g +
exponential decay; aggregations mean/slope/std/p90; combinations
(miss × whiff rate "quality-adjusted whiff", batter−pitcher differentials);
raw-vs-compressed (feed raw rolling features linearly — deliberately also
evaluates the est-PA aggregation-compression thread from 2026-06-10).

## Campaign structure

**Period strata (Codex):** mid-2023→end-2023 = burn-in/backfill QA only
(rolling windows warm up; never evaluated). 2024 = screen. 2025 = primary
confirmation. 2026-to-date = separate ABS-era validation stratum (the ABS
challenge system changes whiff ecology — a real regime variable; never pooled
silently with 2025; report the 2026 interaction explicitly).

**Stage 0 — ingest + features + harness scaffold (Codex round-2 amendments).**
- **Ingest**: backfill mid-2023→present via pybaseball (verified dep 2.2.7)
  into a WIDE bronze table `data/processed/swing_{season}.parquet`:
  game_date/game_pk/at_bat_number/pitch_number/sv_id (when present),
  batter/pitcher, events/description/type, pitch_type, game_type, count,
  stand/throws, zone + plate location, ALL swing/bat-tracking columns, plus a
  stored raw-column manifest. Storage is cheap; re-pulls and schema drift are
  not. pybaseball hygiene: project-scoped `PYBASEBALL_CACHE`, bounded
  retries/timeouts (the datasource has none), serial pulls, and the daily
  incremental re-pulls a rolling recent window (stale current-season cache).
- **Integration shape (mandated)**: per-pitch data is NEVER merged into the
  PA frame (no stable pitch keys there). Build `swing_daily_*` aggregate
  tables keyed (entity_id, date) — keeping denominator rows so "no whiffs" /
  "no swings" / "no tracking" remain distinguishable — apply
  `shift(1).rolling(...)` at the date level, then left-join features onto PA
  rows. Matchup profiles: join SHIFTED batter and pitcher profiles at PA
  date, then form dot products (never same-day products before shifting).
- **Metric/control scaffold**: the paired daily NDCG@10 + season-stratified
  block bootstrap harness does not exist in the repo (scorecard/experiment
  runner are P@1/P(57)-based) — Stage 0 builds and smoke-tests it. No Stage-1
  selection until it exists.

**Stage 1 — screen (2024, 3 seeds, 2–3 boxes).** Purpose: prune variants,
catch leakage/coverage bugs, freeze the bundle. Per-family omnibus models +
variant sweep, judged jointly on the metric hierarchy (below). **Families are
only killed for leakage, coverage failure, or consistently negative evidence
across all metrics/variants** — with five families, false-negative kills cost
more than carrying plausible families forward. Output: ONE frozen best
variant per surviving family + a frozen selected bundle. Selection rule
pre-registered here; confirmation data untouched during selection.

**Stage 2 — confirmation (2025 primary, n=10 stratified seeds, 9–12 box
fleet, `BTS_LGBM_DETERMINISTIC=1`, audit_driver pattern).**
- **Primary hypothesis: frozen BUNDLE vs baseline** — the deployable claim.
- Baseline = **current prod features + the raw-feature decompression fix**
  (B1), so new features can't take credit for fixing the known aggregation
  defect; B1-vs-B0 (current prod) is reported as its own arm — closing the
  2026-06-10 live thread. **B1 caveat (Codex round 2): the decompression fix
  is not yet a concrete code path** — it gets defined and validated during
  Stage 0/1; if it doesn't materialize cheaply, the baseline reverts to B0
  and decompression is evaluated as its own arm alongside the families.
- **Deployable surface**: promotion claims attach to the single frozen
  production config (seed/params as served); the n=10 stratified pooling
  quantifies seed-noise robustness around it, it is not the deployed object.
- Secondary attribution: add-one-family / drop-one-family ablations,
  Benjamini–Hochberg FDR across those family claims (labeled exploratory;
  deployment does not depend on them).
- 2026 stratum re-run for the bundle only (ABS-era validation).

**Stage 3 — decision memo.** Ship/hold per decision rules; any shipped winner
carries the shelved M3 serving-freshness fix (synthetic prediction-date rows,
spec'd 2026-06-11) as a rider, plus the M3 discount note (fresh-value deltas
overstate live contribution while serving is stale).

## Metric hierarchy (Codex-resolved)

1. **PRIMARY: paired per-day top-weighted slate ranking score** — NDCG@10
   with the standard log2 rank discount (binary game-level got-a-hit labels)
   over each day's ranked starter slate; candidate-vs-baseline as paired
   daily deltas, season-stratified block bootstrap, day-clustered inference.
   Power from the whole slate; weight where decisions live.
2. **Guardrails:** daily top-1 / top-3 hit-rate point estimates must not be
   meaningfully negative, defined as: top-1 not below baseline by >1.0pp and
   top-3 not below by >0.5pp (point estimates; no significance requirement —
   known underpowered: ±3pp CI / 446 days).
3. **Diagnostics:** starter-slate game-level AUC, top-decile partial AUC,
   PA-AUC (report-only unless collapsed).
4. **MDP P(57): report-only veto** — computed for the final memo under the
   fixed strategy; can block a promotion, never drive one.

## Controls & hygiene (all pre-registered)

- **Missingness placebo:** availability-indicator-only model — **boolean
  has-swing-data flags only, no values and no counts** (counts carry real
  playing-time signal; a coverage-count placebo may be run separately) — must
  show ~nothing, else the eval is confounded by the post-2023 era marker.
- **Negative controls:** within-entity permuted features (must show nothing);
  **one known-strong leaky sentinel** — same-day UNSHIFTED whiff/miss data as
  a feature — which the harness MUST flag as inflated (proves leakage
  detectability; weak sparse families are not individually required to
  inflate).
- **Coverage ablation:** train 2019+ with NaNs vs train post-coverage-only —
  report both for the bundle.
- **Whiff-denominator reliability:** min-sample thresholds, pseudocount
  shrinkage toward league means (K tuned at screen), explicit
  no-whiffs-vs-no-tracking distinction.
- Doubleheader/lineup/probable-starter handling audited per existing
  conventions; leakage_audit + the "nuclear test" run per CLAUDE.md.

## Infra

Hetzner fleet via `scripts/audit_driver.py` (existing: provisioning,
data-relay, teardown verification per [[feedback_audit_post_teardown_verify]]).
Screen 2–3 × CPX62; confirmation 9–12 boxes. Swing parquets ship via the
data-relay path. All runs `BTS_LGBM_DETERMINISTIC=1`. Local Mac is fallback
(lightgbm now installs).

## Decision rules

- SHIP candidate bundle: primary metric positive on 2025 (CI excluding 0
  under the pre-registered test), guardrails non-negative, controls clean,
  2026 stratum not contradicting (point estimate not meaningfully negative).
  A SHIP includes serving integration (`_build_feature_lookups` + `predict()`
  row assembly for the new features) and the M3 serving-freshness rider —
  training-column support alone is not shippable.
- HOLD/document: anything less. Per-family negative results recorded in the
  experiment backlog with effect sizes + CIs.
- No mid-campaign metric changes; deviations require a spec amendment.

## Out of scope

- Pair-level (batter × pitcher) swing-history features (sparsity).
- Leaderboard-scrape features (leakage).
- Strategy-layer changes (ruled out 2026-06-10).
- Serving-freshness fix ships only as a winner's rider, not standalone
  (M3 HOLD stands).
