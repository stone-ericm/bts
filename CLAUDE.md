# BTS Project Instructions

## Quick Start
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model   # Mac/Alienware (full)
UV_CACHE_DIR=/tmp/uv-cache uv sync                  # Pi5 (no LightGBM)  [Pi5 BTS DECOMMISSIONED 2026-04-14]
UV_CACHE_DIR=/tmp/uv-cache uv run bts run --date 2026-04-01 --dry-run
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v

# ⚠️ The full local suite GRINDS for hours (LightGBM imports locally now → simulate/model/experiment/
# validate run real backtests/training). For NON-model changes, the fast comprehensive regression is:
UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest -m "not slow" \
  --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q  # ~2043 in ~30s

# ⚠️ FETCH FIRST: other sessions (cloud/phone) push to origin — `git fetch origin && git log main..origin/main --oneline`
# before branching. The 8/30 fix was built on a 4-commit-stale base; the rebase auto-merged with ZERO textual
# conflicts but a semantically WRONG ordering in run_single_check. Conflict-free ≠ correct in overlapping areas.

# Scheduler (Hetzner production — systemd --user unit)
UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config ~/.bts-orchestrator.toml
UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config ~/.bts-orchestrator.toml --dry-run

# Hetzner cron setup (reproducible install of cron jobs).
# Source .env first — cron-setup now REQUIRES HEALTHCHECKS_PING_URL (no hardcoded default):
set -a && . ./.env && set +a
bash scripts/cron-setup-hetzner.sh show      # dry-run
bash scripts/cron-setup-hetzner.sh install   # install to bts user crontab
```

## Deployment
- **🚨 Remote / phone incident response → see [`INCIDENT.md`](INCIDENT.md)** — what a cloud (phone) session can vs. can't fix, exact deploy/rollback commands, and the escalation triggers (esp. the on-box cookie re-capture you can't do from the cloud).
- **Deploys trigger on push to `deploy` branch** (NOT main, since 2026-04-21). Workflow: commit/push to main freely; when ready to ship, `git push origin main:deploy`. Gives explicit control over when scheduler restart fires (avoids disrupting live-game scorecard polling).
- **Canary + auto-rollback**: after deploy, workflow waits 30s then checks `systemctl is-active bts-scheduler bts-dashboard` + dashboard HTTP. On failure, auto-reverts to pre-deploy SHA + restarts services.
- **Deploys restart the scheduler** — before a midday deploy, check the sleep window (`ssh bts-hetzner 'journalctl --user -u bts-scheduler -n 2 | grep "Sleeping until"'`) and land inside it; never deploy during a decisive live observation day (a deploy-transition muddied the 6/22 skip-day read).
- **The 01:00 grading cron can now live until ~06:00 ET** (`--wait-deadline-et`). A deploy in that window leaves the old-code grader running against the new tree — harmless for grading but its status artifact stamps the NEW git SHA while produced by the old process. The cron line holds a `flock -n` singleton, so an overlapping manual run no-ops instead of double-grading.
- **Emergency deploy**: `workflow_dispatch` trigger in the GitHub Actions UI — runs deploy without pushing anything. **Select the `deploy` branch in the dropdown** (defaults to main, which fails at the `production` environment gate — the root SSH key is branch-scoped since 2026-07-10).
- Every push to `deploy` triggers the workflow (no paths filter — `deploy` is a dedicated deploy ref, so any push to it is an intentional deploy). Workflow SSHes as root, runs `git pull --ff-only` + restarts `bts-scheduler` + `bts-dashboard` as user `bts`. **Don't manually `systemctl restart` on bts-mlb after pushing** — workflow does it.
- See `the claude-shared memory file reference_bts_deploy_workflow.md` for full details.

## Box access (bts-hetzner) gotchas
- Non-interactive SSH lands as user `bts` with `uv` NOT on PATH → use `~/.local/bin/uv`. No sudo needed (or available) for the user units: `export XDG_RUNTIME_DIR=/run/user/$(id -u)` then `systemctl --user ...`.
- The dashboard binds the **tailnet IP only** (`tailscale ip -4`:3003) — `curl localhost:3003` returns 000 by design, not an outage.
- Data layout: `data/picks/<date>/decision.json` (per-day dirs) vs `data/picks/<date>.policy_shadow.json` (flat files) — skip-day shadow records are NOT inside the date dirs.

## Testing gotchas
- **pi5 clone (`/home/stonehengee/projects/bts`) has 22 known env-only fast-suite failures** (verified identical with/without diff via stash-compare, 2026-08-14): LightGBM not installed (plain `uv sync`, by design) fails tests/test_lgb_params + predict/calibrate/blend/local_tier/preview files, and the box-ported real `.env` credentials shadow the test fixtures in tests/test_dm.py + test_posting.py. A clean run on pi5 = exactly these 22; anything else is yours.
- **Don't write date-relative tests against `date.today()`-defaulting checks** — health `check()` functions date-filter planted files before reading them, so a hardcoded planted date silently expires out of the lookback window and the test goes vacuous/red weeks later (the F4 projected_lineup tests expired 2026-07-16; root-caused + pinned 2026-08-03, `5bde492`). Always pass an explicit `today=` pinned to the planted data.

## Feature computation env vars (set in production via `.env` or systemd unit; defaults in code)
- `BTS_ROOKIE_GATE_K` (default `20`): rookie shrinkage strength. Rookies (career PAs < 100) get PA-weighted rolling + pseudocount shrinkage toward 0.2195 league prior on `batter_hr_{30,60,120}g`. Veterans untouched. Set to `0` to revert.
- `BTS_PITCHER_HR_30G_MIN_PERIODS` (default `7`): rolling-window min_periods for `pitcher_hr_30g`. Shipped 2026-04-14 after +0.81pp avg P@1 + +0.46pp MDP P(57) on walk-forward (seed=42 measurement — pending multi-seed re-validation). Set to `10` to revert.
- `BTS_LGBM_RANDOM_STATE` (default `42`): LightGBM `random_state` for live prediction (classifier + blend), eval backtests, and the simulate/audit paths (classifier, ranker, regressor, V-REx). Defaults to 42 for backward-compat with shipped policy. Used by multi-seed audit workflows to measure cross-seed variance without code edits. **IMPORTANT: seed=42 was discovered to be a statistical outlier on 2026-04-14 — all historical single-seed experiment deltas are suspect until multi-seed audit completes.**
- `BTS_LGBM_DETERMINISTIC` (default `0`, OFF): when `=1`, sets LightGBM `deterministic=True` + `force_row_wise=True` in `LGB_PARAMS` for bit-exact-reproducible training across providers. Required for cross-provider seed pooling (OCI E5.Flex showed -1.62pp 2024 P@1 drift vs Hetzner CPX51 on identical seed=42 without this flag — see `project_bts_oci_provider_add.md`). Code-prepped on main 2026-04-24 (`498c5a8`); flip in `.env` on bts-hetzner only after a deliberate re-baseline cycle since it slightly shifts P(57) by changing parallel-reduction order in gradient calculation. **Same flag also unblocks the AX102-U pivot** — Zen-4-to-Zen-4 doesn't save you from non-determinism on dedicated cores.

## Required Prefixes
- All `uv` commands: `UV_CACHE_DIR=/tmp/uv-cache`
- LightGBM needs ARM libomp: `arch -arm64 /opt/homebrew/bin/brew install libomp`
- LightGBM is an optional dep: `uv sync --extra model` to install it

## Hetzner debugging (non-interactive SSH)
- `uv` is NOT in PATH over non-interactive `ssh bts-hetzner '...'` — run the venv python directly: `cd ~/projects/bts && .venv/bin/python -c "..."` (prod repo is `~/projects/bts`, NOT `~/bts`). Good for one-off state checks like `load_decision_streak_state`.
- Read-only debugging (cat state files, `journalctl --user -u bts-scheduler`, run a diagnostic) does not disturb the running scheduler; it short-circuits already-locked picks.
- **Dashboard binds the TAILNET interface, not localhost** — probe `http://bts-hetzner:3003` from the tailnet; `curl localhost:3003` on the box refuses (looks down when it isn't).
- **Fresh worktree venvs lack the model extra**: run `UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model` or tests/model fail with ModuleNotFoundError. LightGBM + libomp DO work on this Mac (verified 2026-07-08; older notes saying otherwise are stale).
- **park_drag external table**: `data/external/park_drag/` (gitignored) — export + manifest + producer seed stores; refreshed by cron `45 7 * * *` (`bts park-drag-refresh`, log `~/logs/park_drag.log`); Savant's search index LAGS same-night games, so a late-night manual run correctly gets nothing for yesterday.

## Safety Rules
- **M3 revisit trigger**: after any change that materially improves slate discrimination (rolling realized AUC ≥ ~0.61 vs the ~0.59 baseline — the `slate_auc` health check WARNs on this automatically), re-run `scripts/replay_m3_serving_parity.py`. The serving-staleness HOLD (`docs/audit/2026-06-11-m3-serving-staleness.md`) is only valid while the model can't discriminate adjacent top candidates.
- **Never use features computed on full dataset** without shift(1) temporal guard
- **Never trust feature importance or ablation results** if there's any chance of leakage — fix leakage first, then re-evaluate
- **K-Means clustering is NOT safe** for features — cluster assignments are 90.8% unstable across train/test splits
- Run `scripts/leakage_audit.py` and the nuclear test after any feature changes
- **BTS hit-scoring excludes the resumed portion of suspended games** — read PA via `build.read_pa_for_bts_scoring` / `filter_out_resumed_portion` (NOT raw `pd.read_parquet`) anywhere PA feeds hit/streak/calibration/contest scoring; the production scorer grades via `picks.grade_pick_in_feed`. Model training, features, and the skill-pool prior intentionally KEEP resumed PA (real events). See ARCHITECTURE "Suspended-game scoring" (`is_resumed_portion`).
- **The leaderboard scraper + contest fetch run on Eric's REAL BTS contest account cookies** (`~/.bts-leaderboard-cookies.json`) — a ban costs the actual streak. Deep scraping is authorized but kept low-profile: keep the browser identity (`endpoints.browser_headers`), the jittered gaps, the `profile_top_n` footprint cap, and the **403/429 → `RateLimitedError` kill-switch** (`scraper._get_json`). Don't remove the kill-switch or crank the volume/cadence without a reason. No IP/proxy tricks — this is request hygiene for one authorized account, not evasion.

## Data
- Raw JSON: `data/raw/{season}/{gamePk}.json` (gitignored, ~15GB)
- MiLB raw: `data/raw/milb/{season}/{gamePk}.json` (6,643 games, 2023-2025. Pitch types only available 2023+)
- Processed Parquet: `data/processed/pa_{season}.parquet` (gitignored)
- Regular season only. 7-inning COVID doubleheaders dropped.
- MLB API requires v1.1 (`/api/v1.1/game/{pk}/feed/live`), not v1.

## Strategy Simulation
> **⚠ PROFILE BASIS (2026-07-06 gotcha, cost real time):** for policy/model eval use the
> **`estimated_pa`** profiles (`data/hetzner_results/mdp_estpa_run`, 24 seeds × 5 seasons,
> rank-1 hit ~0.75 = serving-realistic, has `game_pk`). Do NOT use `data/simulation/backtest_*.parquet`
> — those are **`actual_pa`** (compounds per-PA prob over the *realized* PA count → hindsight, rank-1
> hit **0.865**), which INFLATES streaks and can *flip* the policy ranking (MDP looks great on inflated
> data; always-double wins on realistic). See `docs/audit/2026-07-06-strategy-model-lever-investigation.md`
> + reproducible comparator `scripts/audit/confirm_mdp_policy_replay.py`.
>
> **⚠ RUN STRUCTURE (2026-07-13 gotcha):** even on the right profiles, iid day-type solvers
> (`solve_mdp` / any DP over quality bins) get milestone probabilities policy-DEPENDENT-wrong:
> realized rank-1 sequences suppress long all-hit windows (20-windows ×0.10-0.14 vs iid/permutation
> nulls, all 5 seasons) — iid inflates always-single reach-20 ×3.6 while UNDERSTATING doubling
> policies. For milestone/policy comparisons, replay realized sequences (pattern:
> `scripts/audit/dd_p_policy_value_sensitivity.py` L2); treat iid DP values as structure only.
> See `docs/audit/2026-07-13-dd-p-policy-value-sensitivity.md` finding 5.
```bash
# Run blend backtest (5 seasons, ~2-3 hours, needs --extra model)
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate backtest --seasons 2021,2022,2023,2024,2025

# Monte Carlo strategy comparison
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate run --trials 10000

# MDP solver — generates optimal policy file
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate solve --save-policy data/models/mdp_policy.npz

# Exact P(57) for any strategy (no Monte Carlo noise)
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate exact --strategy combined

# Multi-metric scorecard (baseline comparison)
UV_CACHE_DIR=/tmp/uv-cache uv run bts validate scorecard --diff data/validation/scorecard_baseline.json
```

## Architecture
See `ARCHITECTURE.md` for full details. Key points:
- PA-level LightGBM → game-level probability aggregation
- 16 baseline features (FEATURE_COLS) + 5 shadow context features (CONTEXT_COLS, incl. park_drag_delta from the external as-of table) + 9 Statcast features, all provably leak-free (date-level shift(1))
- 12-model blend: baseline + single-Statcast variants. `--no-blend` for single model.
- **Shadow model** (v2 since 2026-07-08): `CONTEXT_COLS` (ump_hr, wind, hardness, indoor, park_drag_delta) run alongside production via `feature_cols_override`. Picks saved to `{date}.shadow.json`. Monitor with `bts shadow-status`; result reconciliation via `bts check-results` now writes `data/validation/context_stack_shadow_status.json` by default when shadow files exist. Use `bts shadow-backfill-results` for reviewed DD-aware recompute/audit manifests before any promotion discussion. **Stranded-result protections (2026-08-09, after the 7/10 shadow sat unresolved a month; hardened in Codex r2):** `check-results` attempts shadow reconciliation on EVERY exit path (incl. no-pick and production-pending), and the cron invokes it with `--wait-deadline-et 06:00` — an in-process 15-min retry loop (hard deadline: no attempt starts after it; capped final sleep) using the grader's own resolvability as the "day over" signal. A **stale-scoring guard** refuses streak-bearing scoring for dates >2 days old (shadow reconciliation still runs), so manual old-date remediation is always safe; `--allow-stale-scoring` overrides it ONLY for a chronologically-ordered backfill after a multi-day outage — `update_streak` applies results against CURRENT streak state, so out-of-order grading corrupts it. Residual stranded dates — shadow AND scoreable-production sides — surface via the `result_resolution` health check (WARN age ≥2d → immediate attention DM, CRITICAL ≥7d, 30-day horizon, version-blind in-memory scan). ⚠ **Activation**: deploys never touch crontab — the wait loop + flock are INACTIVE on the box until `bash scripts/cron-setup-hetzner.sh install` is run there once (next-deploy checklist item).
- **Skip-policy shadow** (`skip_policy_shadow.py`, distinct from the shadow MODEL above): a counterfactual "pick-the-band" shadow POLICY testing whether the deployed skip-at-streak≥8/sub-0.796 rule costs streaks on the production scale. **Ground truth via `decision.json`:** the scheduler writes `data/picks/<date>/decision.json` (`bts_daily_decision_v1`) at each true finalization point — pick commit (delivery branches: `delivered`/`private_locked`/`locked_unconfirmed`), classification-lock (only when genuinely delivered), crash-guard, and end-of-day MDP skip. `committed_pick_written` + `final_skip_candidate` are tracked across the day; all writes are best-effort and never affect the pick. The shadow reads `decision.json` files: only `action=="skip" && source=="mdp"` entries produce a shadow record `{date}.policy_shadow.json` for the executable declined candidate. **`check-results` scoreable gate (GH #144):** scores only `decision.scoreable==True`; fallback `picks.pick_was_delivered(daily)` when no decision file exists (NOT `scheduler_state.pick_locked`). A stale preview/undelivered `<date>.json` on a skip day no longer corrupts the streak. Verdict vs the ~0.744 calibrated breakeven: `below_breakeven`=skip validated, `above_breakeven`=skip costs streaks, else accumulating. CLI `bts skip-policy-shadow-update` (nightly cron 23:30 ET) / `bts skip-policy-shadow-status`; dashboard panel. Why `decision.json` not read-only reconstruction (4 review rounds): `docs/audit/2026-06-20-skip-policy-shadow.md`. **Follow-up (2026-06-29):** threshold kept as-is — estimated_pa backtest run (actual_pa hindsight ≈75% of the scale gap), re-solve barely helps, skip days not reliably worse, and the PA-volume discrimination lever dissolved (no validated lever). Full analysis + reproducible scripts: `docs/audit/2026-06-29-skip-threshold-and-discrimination.md`, `scripts/audit/`.
- **MDP-optimal strategy**: auto-loads `data/models/mdp_policy.npz` for skip/single/double decisions. Falls back to heuristic if absent.
- **Tail policy (2026-09-03)**: once `streak + 2*days_left < 57` the reach-57 table is all-skip by construction (every action values 0 → argmax index 0), which idled production on 9/03. `bts.simulate.tail_policy` switches — from STATE alone, before any artifact is consulted — to exact E[season-best] with the contest `best_streak` carried as state (`data/models/mdp_tail_policy.npz`, one late-season bin, stop rule `skip iff min(57, s+2d) <= best`, so it stops only when the season best can't be beaten). Only a TRUSTED best (auto fetch/unexpired manual, current season, streak ≤ best ≤ 57) may stop it; anything else degrades to best = streak and keeps picking. Every artifact failure → forced rule (skip iff stop, else single), never the zero table or the 0.80 heuristic. decision.json v3 records objective/best/effective_best/tail sha (the PolicyDecision also rides ON the pick file as `policy_decision`, so cached-fallback/restart commits keep it); `tail_policy` health source; rebuild with `scripts/rebuild_tail_policy.py` (also required after ANY base-policy rebuild — the tail is sha-bound to it). `docs/audit/2026-09-03-emax-tail-policy.md`.
- **Phase-aware bins**: early season (Mar-Aug) vs late (Sep only, `late_phase_days=30`)
- **Streak saver tracked**: `saver_available` in `streak.json`, consumed on first miss at streak 10-15
- **Scheduler daemon** (`scheduler.py`): replaces fixed 11am/4pm/7:30pm cron with dynamic game_time−offset lineup checks (`lineup_check_offset_min`: code default 45, **box TOML 60** — don't compute expected check times from the 45); run times are computed once at the morning fetch (singleton-slate / moved-up-game gap backlogged in `docs/optimization-ideas.md`); confirmation-based posting via `early_lock_gap`; 1am cron kept as safety-net fallback. **Cutoff guard + deadline-aware fallback (2026-08-30):** `_deliver_and_lock_pick` refuses at/after first pitch − 5 (`picks.SUBMISSION_CUTOFF_MIN`, one definition); `fallback_deadline_min` is the latest cascade START, floored at `5 + cascade_budget_min (12) + operator_reserve_min (10)`; the in-loop fallback re-decides AFTER its refresh via `plan_fallback_action` and delivers a gap-blocked enterable pick when the contender's window can't finish in time. Intraday cascades skip the season re-pull once it succeeded today (`BTS_REFRESH_ALWAYS=1` forces). See `docs/audit/2026-08-30-late-pick-delivery.md`.
- **`scheduler.pick_delivery`** controls pick delivery: `public` posts to the Bluesky feed, `dm` sends the pick by Bluesky DM, and `private` saves locally only. Legacy `private_mode=true` still maps to local-only delivery when `pick_delivery` is unset. Don't confuse this with `shadow_model` (runs the context stack model alongside production for eval).
- **Fallback deadline** uses `_earliest_pick_game_et(daily)` — earliest of primary + double-down game times, not primary alone. Fixes the case where double-down is in an earlier game than primary.
- Projected lineup fallback for morning predictions
- Train on 2019+ data (2017-18 hurts)
- Starter/reliever PA split in aggregation
- `notna().any()` not `.all()` — LightGBM handles NaN natively for Statcast features

## Statcast swing campaign (experimental, 2026-06-12) — gotchas
- Per-pitch swing data: `scripts/backfill_swing_data.py` → `data/processed/swing_{season}.parquet` (mid-2023+ coverage). Set `PYBASEBALL_CACHE=data/raw/pybaseball_cache`; backfill is resumable + has `--incremental-days`.
- **Savant 403s the default urllib/python User-Agent from datacenter IPs** — pass a browser UA (see `scripts/qa_swing_vs_leaderboard.py`).
- **Leaderboard `n_swings` = bat-TRACKED swings EXCLUDING bunts.** Matching that definition closed a +3.5% count bias to 0.25% in QA. Bunt descriptions are excluded from `bts.features.swing` sets.
- Screen runs on the Hetzner box; the controls gate (sentinels + null arms) is a hard STOP — never read family results before the gate passes. Spec/plan: `docs/superpowers/specs|plans/2026-06-12-statcast-swing-*`.
