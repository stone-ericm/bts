# C2 side item (e): catcher-grouped framing screen, stage one — pre-registration

**Status:** DRAFT for review (one Codex review unit, at most 2 rounds). Frozen at the reviewed commit once signed; nothing
runs before a plain SIGN, the exposure row X-35 and the admission record.

**Authority:** Eric, 2026-10-06, his own words relayed by job-search-52 (register row **C2-framing-side-item**; C2 index
item (e)): add this screen to C2 as a scoped side item, with no production change and no deploy. Stage one is 3 paired
seeds. Stage two (to 10 seeds) needs his second go-ahead. If the first walk-forward's measured cost is well above the
estimate, stop and report. C2 step 2a keeps priority.

**What it can support:** the 2021–2025 seasons are already consumed for deployment-grade evidence
(`docs/sota_audit/2026-05-08-fresh-audit-pre-registration.md`). So this screen can conclude at most "worth testing on
untouched 2027 data". It can never conclude "ship it" (decision memo `docs/audit/2026-10-04-2027-decisions.md` D2/D4).

## 1. The question
Production's framing proxy `pitcher_catcher_framing` is computed in `bts.features.compute` as follows:
- **Measure:** each PA's borderline called-strike rate, `pa_borderline_csr`.
- **Borderline pitch:** |pX| between 0.5 and 1.2, or within 0.3 ft of `sz_top` or `sz_bottom`.
- **Grouping and history:** grouped by **pitcher** and date. A per-pitcher expanding mean of the daily means is then taken, shift(1) and min_periods 5.

The pitcher grouping predates the catcher id in the PA schema (`31ead55`, then `38bc71c` and `0bfbb93`). A catcher-grouped version of the same measure has never been tested. The 4/09 `catcher_hit_rate` experiment measured hits allowed per catcher, a different quantity.

## 2. The variants
**The feature:** `catcher_framing` is the same computation with `fielding_catcher_id` as the grouping key, applied to the same `pa_borderline_csr` column. The code is `scripts/audit/c2_framing/screen.py`, `framing_by`.
- **Variant A:** the base feature set with `pitcher_catcher_framing` replaced by `catcher_framing`, at the same position.
- **Variant B:** the base feature set with `catcher_framing` appended.
- **Baseline:** the current production base, `FEATURE_COLS` (16 features).
- **Blend configs:** for each variant, each of the 12 blend configs keeps its Statcast extras. Only its base is rewritten, by the same rule as `bts.experiment.runner`.

**Self-check before any walk-forward:** `framing_by(df, "pitcher_id")` must reproduce production's `pitcher_catcher_framing` exactly, on the real inputs (same rows, same order, NaN equal). If it does not, the run stops (`STOPPED.json`, exit 4).

## 3. Inputs
- **Files:** the PA parquets `pa_2017.parquet` to `pa_2025.parquet` in the production data directory on the box.
  - Each is pinned by sha256 in `scripts/audit/c2_framing/admission.json` (`input_pins`), hashed on the box before the exposure row.
  - Each is read once, from the hashed bytes.
- **2026 is never read:** the driver opens only the pinned names.
- **Coverage of `fielding_catcher_id`** (parquet metadata, read 2026-10-06):
  - 2019–2025: 0 nulls;
  - 2017–2018: no such column, so the catcher feature has no 2017–18 history and NaN there.
  - Training starts in 2019 (`TRAIN_START_YEAR`).
  - The run records the feature's non-null coverage in the test seasons.

## 4. Procedure, per seed (one claimed run each)
1. Admission gate (shared `scripts/audit/c1/admission.py`):
   - the reviewed commit, a plain SIGN of this unit, and X-35 published unchanged;
   - the closure unchanged, with no foreign modules;
   - one durable claim per seed, under `data/hetzner_results/c2/framing_screen/seed_<seed>/`.
2. `BTS_LGBM_DETERMINISTIC=1` (refused otherwise) and `BTS_LGBM_RANDOM_STATE=<seed>`.
3. Load the pinned inputs, run `compute_all_features`, then the self-check, then add `catcher_framing`.
4. For baseline, then A, then B, and each test season 2024 and 2025:
   - `blend_walk_forward(..., retrain_every=7, blend_configs=…, game_probability_mode="estimated_pa")`, with the standard top-10 profiles. This is the serving-realistic basis, not actual_pa hindsight (CLAUDE.md, "PROFILE BASIS").
   - Each profile is saved, and the walk-forward's CPU time is recorded.
5. Score:
   - `compute_full_scorecard` per variant;
   - `diff_scorecards` of A and B against the same seed's baseline;
   - the repo's screening rule `evaluate_pass_fail` per seed (P@1 up in both seasons, or neutral within 0.3pp plus streak metrics up).

**Seeds:** the first three of `data/seed_sets/canonical-n10.json`: 2273360, 260991262, 1746737973. These are positions 0–2 of the April multi-seed standard, chosen by position, not by result.

## 5. Dispositions (fixed now), per variant
For each variant and seed, let d = the mean of that seed's 2024 and 2025 P@1 deltas. Then let m = the mean of the three d values and t = m / (sd / √3); t is ±∞ when sd = 0.
- **positive** ("worth testing on untouched 2027 data") requires all of:
  - the mean 2024 delta and the mean 2025 delta are both above 0;
  - m is at least +0.003 (+0.3pp: the practical threshold of the June feature screens, about half a day per season);
  - t is at least 1.5 (the repo's multi-seed keep rule);
  - the per-seed screening rule holds on at least 2 of 3 seeds.
- **negative:** m is 0 or below, and the per-seed rule holds on fewer than 2 seeds.
- **inconclusive:** anything else. Only an inconclusive variant could justify stage two, and stage two needs Eric's second go-ahead.

**What these are not:** with 3 seeds, t ≥ 1.5 is a screening convention, not a significance test. A and B are reported separately, with no multiplicity adjustment. No disposition approves any production change.

**Reported but without decision weight:**
- the exact P(57) delta and the `mean_max_streak` delta (the latter is a 10,000-trial Monte Carlo at its fixed default seed);
- the per-season P@1 of every variant;
- the feature coverage;
- the CPU cost per walk-forward.

## 6. Compute and stops
- **Estimate (unmeasured):** about 5 CPU-hours per season walk-forward, so about 30 per seed (6 walk-forwards) and about 90 for stage one.
- **Launch:** each seed is one launcher job: `--name c2-framing-seed<k> --cpu-hours 45 --max-hours 16`, run from `~/projects/bts-c1`, re-pinned to the reviewed commit.
- **First-walk-forward stop (Eric's condition):** if the first walk-forward (seed 1, baseline, 2024) uses more than 7.5 CPU-hours (1.5 × the estimate), the run stops (`STOPPED.json`, exit 3). The lead reports before anything else runs.
- **Launcher gates** (combined C1 + C2 ledger, 0.48 CPU-hours on 2026-10-06):
  - At 50 the launcher refuses until Eric's `CHECKPOINT_50_ACK.json` exists. The lead stops and asks him.
  - At 100 it hard-stops.
  - Stage one at the estimate would leave about 9.5 CPU-hours under 100 for C2's remaining planned 11.1. That is recorded for Eric's cap decision.
- **Seed order and the seed-1 gate (Eric, 2026-10-06, relayed by job-search-52):** only seed 1 launches first. Its measured CPU cost, the projected cost of seeds 2–3, and what that leaves under the 100 cap for C2's planned jobs go to Eric (through job-search-52) before anything else launches. He decides on the 50 CPU-hour checkpoint and the remaining seeds then; there is no pre-approval of the 50 gate. A seed-1-only result is not a stage-one disposition: §5 needs all three seeds.

## 7. Reporting
- The per-seed `results.json` files, then `aggregate` across the three run directories (dispositions as in §5).
- A results note, then an independent acceptance (C2 rule), then the result goes to Eric through job-search-52:
  - per-seed and mean P@1 deltas per season for A and B;
  - the dispositions;
  - the measured CPU cost.
  - The fallback copy is `~/projects/job-search/mets-2026-10-06/catcher-experiment-result.md`, with the herdr manager told.

## 8. Limits
- **The catcher identity is approximate:** one catcher per side per game, the first player listing position code 2 in the boxscore (`bts.data.build._get_starting_catchers`). Mid-game substitutions are not represented.
- **No catcher history before 2019.** The pitcher feature has it.
- **Small, consumed evaluation:** 2 seasons and 3 seeds, on consumed data. A positive is a reason to register an untouched 2027 test, nothing more.
- **Not comparable to 2026-03-31's 85.1% → 87.0%:** that was a single model on a 13-feature base, seed 42, and probably the actual_pa basis.
- **No serving input:** production has no pre-game catcher input. A variant could not be served without one: the posted lineup's catcher, or the prior game's for projected lineups.
