# K / contact / expected-AB feature screen — design + plan

**Date:** 2026-06-15
**Branch/worktree:** `kcontact-screen` @ `/Users/eric/projects/bts-kcontact` (off main 90ae06d)
**Status:** spec — implementation pending (subagent-driven)

## Goal

Add the model's missing **pitcher contact/command** dimension and a **batter expected-at-bat** signal, screen them leak-safely, and ship only what clears a hard *un-compression* gate. The model today has `pitcher_hr_30g` (hits-allowed/PA), arsenal entropy, and velo/spin/extension/break levels — but **zero pitcher K/whiff/CSW/zone signal** and **nothing on walks consuming at-bats**.

## Honest prior (read before interpreting any result)

This model is near its ceiling. Prior art (`project_bts.md`, `docs/validation/final-report.md`, `docs/audit/2026-06-10-mdp-estpa-ab-methodology.md`) has rejected ~30 features and ruled out "a better opposing-starter feature" as a meaningful win; the +9pp starter signal was mostly leakage. **The most likely truthful outcome here is a clean null.** That is a valuable result — it would effectively close feature-hunting on this model. The danger is a *false positive* from leakage or aggregation un-compression. The controls below exist to make a null trustworthy and a win real.

**Distinct from already-rejected work** (verified): rejected = BB%-as-feature (redundant w/ `count_tendency`), contact-*composite* (batter batted-ball), batting-order (captured by slot→PA), implied-total, TTO-adjustment, sprint-proxy, ranking-objective. New here = **pitcher** contact/command (never tested) and **expected-AB as a batter feature** (a different mechanism than BB%-as-predictor).

## Grounded vocabulary (from `pa_2024.parquet`, do NOT reuse `compute.py:_whiff_rate`'s incomplete set)

- `pitch_calls` codes: `B *B`=ball, `C`=called strike, `S`=swinging strike, `W`=swinging strike (blocked), `F`=foul, `T`=foul tip, `X`=in-play out, `D`=in-play no-out, `E`=in-play run, `H`=HBP, `L/M/O`=bunt fouls/misses, `P`=pitchout.
  - **whiffs = {S, W}** ; **swings = {S, W, F, T, X, D, E}** ; **called_strike = {C}** ; **CSW numerator = {C, S, W}** ; **in_play = {X, D, E}** ; **total pitches = len(pitch_calls)**.
- `event_type`: **K = {strikeout, strikeout_double_play}**, **free_pass = {walk, hit_by_pitch}**, **push (non-AB, non-decision) = {walk, hit_by_pitch, catcher_interf, sac_bunt}**. ⚠ **VERIFY** the official MLB Beat-the-Streak zero-AB rule and whether `sac_fly` is a decision (reset) or a pass before finalizing `batter_decision_pa_rate`; default assumption = sac_fly is a DECISION (failed hit attempt), pushes are PASS.
- Zone: in-zone iff `|pitch_px| <= 0.83` ft AND `sz_bottom <= pitch_pz <= sz_top` (per-pitch arrays present).

## Features (`src/bts/features/k_contact.py`, mirror `src/bts/features/swing.py`)

All features: per-PA counts → **date-level** aggregation (merge doubleheaders) → **`shift(1).rolling(window, min_periods=k)` ratio-of-rolling-sums** (retain numerator+denominator; NEVER mean-of-daily-means). Reliability gate = min denominator in window (NaN otherwise), mirroring `min_whiffs`.

Pitcher (key `pitcher_id × date`, broadcast to PAs faced; window 30 game-dates):
- `pitcher_k_rate_30g` = K / PA
- `pitcher_bb_rate_30g` = free_pass / PA
- `pitcher_csw_30g` = (C+S+W) / pitches
- `pitcher_whiff_rate_30g` = (S+W) / swings
- `pitcher_zone_rate_30g` = in_zone / pitches

Batter (key `batter_id × date`; window 30 except decision_pa = 60):
- `batter_k_rate_30g` = K / PA  (distinct from `batter_whiff_60g` = whiffs/swing)
- `batter_chase_rate_30g` = out-of-zone swings / out-of-zone pitches (O-Swing%; distinct from whiff/swing)
- `batter_decision_pa_rate_60g` = 1 − push_rate = decision_PAs / PAs  — **the expected-AB lever**: high-walk batters convert fewer PAs into hit chances, which `1-(1-p)^est_pas` (PA-count, predict.py:705) cannot see.

**Dropped:** standalone `batter_bb_rate` (rejected item-7, redundant w/ `count_tendency`; its walk signal lives in `batter_decision_pa_rate`).

## Screen (`src/bts/experiment/k_contact_screen.py`, mirror `swing_screen.py`)

Arms: `baseline` (FEATURE_COLS + availability flags) + one single-variant arm per feature + `omni_pitcher` + `omni_batter` + `omni_ALL`.

**Controls (HARD gate — report reads controls FIRST, hard-stop on failure):**
- `ctl_sentinel_gross` — label injected → AUC≈1.0 every seed (harness sanity).
- `ctl_sentinel_leaky` — **UNSHIFTED same-day** pitcher rate (today's K/whiff) → must fire above the permutation-null band (proves the screen can detect a same-day leak; covers the pitch-array vector).
- `ctl_permuted` — within-date label permutation → null band.
- `ctl_mask_only` — availability flags only, no values → null (catches coverage/era artifacts).
- **`ctl_uncompression` — baseline + a re-expression of EXISTING `pitcher_hr_30g` (within-day rank + monotone transform), no new data.** ★ **THE BAR: any new-feature arm must beat this**, else the "lift" is just the aggregation un-compressing (the MDP audit's ~+0.006), not new information.

**Metric:** per-day NDCG@10 primary (Codex-resolved swing-campaign standard) + P@1/P@3 top-k confirmation. Global AUC = sanity only, never the objective.

**Splits:** train 2019–2023 → screen/select on **2024** → confirm the single pre-registered winning bundle ONCE on **2025** (untouched). 2026 = separate ABS-era stratum, report-only, never pooled. (Note: 2025 has been touched by other analyses; treat the 2025 confirmation as supporting, not sole, evidence; forward-2026 is the cleanest future audit.)

**Eval-integrity fixes (from leakage review):**
- **Remove** the "drop zero-decision-PA (all-walk PASS) games from negatives" diagnostic from anything that informs verdict/bundle selection (PASS is a post-game outcome → eval leak). If contest-accurate scoring is wanted, apply PASS-neutrality symmetrically to baseline and candidate.
- Evaluate the winning bundle through the **production split aggregation** (predict.py:705–720: starter features for ~2.5 PA + league-avg reliever), not only the screen's flat `1-(1-p)^est_pas`.

## Build tasks (subagent-driven, TDD red→green each)

1. `src/bts/features/k_contact.py` — per-PA count helpers (grounded vocab above) + date-level shift(1) ratio-of-rolling-sums + attach. Mirror `swing.py`.
2. `tests/features/test_k_contact.py` (red first): (a) synthetic batter/pitcher whose date-D value would change if today's PA leaked → assert unchanged (date-level shift(1)); (b) doubleheader merge; (c) reliability-gate NaN; (d) vocab correctness (W counted as whiff/swing); (e) decision_pa_rate push classification.
3. `src/bts/experiment/k_contact_screen.py` — ARMS + the 5 controls incl. `ctl_uncompression`; mirror `swing_screen.py` build_arm_frame/run_screen_arm; per-day NDCG@10 + P@1/P@3.
4. `scripts/k_contact_screen_driver.py` — build PA frame via `compute_all_features`, attach k_contact inline, run arms across seeds (mirror swing driver).
5. `scripts/k_contact_screen_report.py` — controls-gate FIRST (hard stop unless gross+leaky sentinels fire above every null AND new arms beat `ctl_uncompression`), then families, then bundle proposal (Eric's freeze call, never auto-freeze).
6. Confirmation: retrain winning bundle on 2019–2024, evaluate ONCE on 2025; report 2026 ABS stratum separately.
7. `scripts/leakage_audit.py` + nuclear test after the feature module lands (CLAUDE.md safety rule); adversarial Codex pass on the writeup.
8. **(gated, only if bundle clears gate+confirmation)** promote winning columns into `compute.py` + FEATURE_COLS **and** extend the production reliever-feature swap (predict.py:698–701) to set league-avg reliever values for every new pitcher feature + a serving-parity replay test.
9. **(separate spec, NOT this build)** coherent expected-AB reparameterization of the predict.py aggregation (higher risk, touches serving path).

## Risks / confounds to keep visible in the report
- New pitcher features are collinear with `pitcher_hr_30g` + `pitcher_catcher_framing` → the un-compression control is the discriminator.
- Pitcher "30g" ≈ 30 starts ≈ ~5 months (vs batter 30g ≈ 5–6 weeks): pitcher rates are near-true-talent, not recent-form.
- Coverage/min-periods NaN correlates with fringe (low-hit) players → `ctl_mask_only` captures it.
- Screen uses actual starter + actual lineup; production uses probable pitcher + projected lineup → offline lift is optimistic; discount accordingly.
