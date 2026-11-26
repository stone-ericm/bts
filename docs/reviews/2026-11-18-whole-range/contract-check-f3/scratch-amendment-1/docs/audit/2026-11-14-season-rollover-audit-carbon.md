# 2027 Season-Rollover Codebase Audit

**Disclosures:** Outside allowed set read: none (`~/projects/claude-shared`, `~/bts-runs`, `~/.codex`, `~/.claude`, and other checkouts were untouched). Runtime-injected content: global user rules (`~/.gemini/AGENTS.md`), builtin Antigravity skill headers, and six context-compaction summaries from earlier turns of this session.

## Summary

1. **[WRONG PICK STATE]** `SaverState.is_available` returns `False` for `state="not_earned"`, feeding `saver=0` into `mdp_policy.npz` across `745/9,000` divergent cells at streaks `0..9` (including Week-1 `d=180`) and failing checklist B8's gate.
2. **[WRONG PICK STATE]** Dashboard Streak Saver (`web._saver_season`) and `health/contest_state` read `load_contest_streak_state` directly, displaying and mutating `season=2026` until the first 2027 contest fetch.
3. **[WRONG GRADE OR STREAK]** If MLB publishes 2027 `rounds.json` before resetting `user/{id}/profile`, `derive_source_date` returns `None`, defeating Kickoff S1 and auto-earning 2027 `saver_state.json` from 2026's `seasonBestStreak`.
4. **[FALSE ALARM]** Opening Night EOD (and Day-2 EOD under normal 1-pick MLB ledger lag) fires `CRITICAL` `contest_state` DMs even with Kickoff S2, because `run_result_polling` settles `2027-03-24.json` immediately before `run_all_checks`.
5. **[FALSE ALARM]** `park_drag_freshness` pages `CRITICAL` on Opening Night EOD (`gap_days=178` because March is not in `OFFSEASON_MONTHS`), and an empty reseeded 2027 `pitches_current.csv.gz` crashes `park_drag_producer.refresh()` (`NaN` -> `int`) after Game 1.
6. **[FALSE ALARM]** Restored `entry_intent="research"` crontab still runs 4x/day `fetch-contest-streak --dm-recipient`, sending failure DMs on expired 2026 cookies from `2027-03-03` until A4; `bts-leaderboard.service` also hardcodes `stoneericm.bsky.social`.
7. **[SILENT DATA LOSS]** Opening Night 1-game partial-lineup lock (`9` batters vs `top_n=10`) wedges `live_forward_capture` in `failed_verify_existing` even after the second lineup posts, paging `CRITICAL` at EOD and permanently from `2027-03-31`.
8. **[COSMETIC]** `skip_policy_shadow` and `shadow_eval` glob `data/picks/*.policy_shadow.json` and `*.shadow.json` with no season filter, pooling 2026 outcomes into 2027 checkpoint verdicts and review-day counts.

## Findings

### F1. Early-season `saver_state="not_earned"` feeds `saver=0` into MDP at streaks `0..9`
- **Severity:** `WRONG PICK STATE`
- **State:** Fresh 2027 contest account (`streak=0..9`, `season_best_streak<10`) after `fetch-contest-streak` writes `saver_state.json = {"season": 2027, "state": "not_earned"}`.
- **Wrong outcome:** `SaverState.is_available` returns `False` (built on the false 2026-06-18 spec claim that `saver` has zero effect outside streaks `10..15`), telling the MDP the once-per-season saver was already burned (`transition_outcomes` never transitions `0 -> 1` at streak 10). Across `data/models/mdp_policy.npz`, `saver=0` vs `saver=1` flips `745/9,000` cells in streaks `0..9`, including 7 cells at `d=180` (`2027-03-24..03-30`): `s=5,q=0` (`single` vs `double`), `s=7,q=0` (`single` vs `skip`), `s=8,q=0..3` (`skip`/`double` vs `single`), `s=9,q=4` (`single` vs `double`). It also makes checklist B8's gate ("streak 0 with the saver available") fail (`saver_available=False`).
- **File and line:** [saver_state.py](file:///Users/eric/projects/bts-agy-audit/src/bts/saver_state.py#L26-L28), [contest_state.py](file:///Users/eric/projects/bts-agy-audit/src/bts/contest_state.py#L308-L309), [contest_state.py](file:///Users/eric/projects/bts-agy-audit/src/bts/contest_state.py#L339-L340), [mdp.py](file:///Users/eric/projects/bts-agy-audit/src/bts/simulate/mdp.py#L61-L75).
- **How demonstrated:** Ran `maybe_auto_earn_saver(..., season=2027, best_streak=0)` + `load_decision_streak_state` (`saver_available=False`), diffed `data/models/mdp_policy.npz` (`745/9000` flips), and ran `select_pick` on `2027-03-29` (`streak=5, q=0`), returning `"single"` instead of `"double"`.
- **Smallest fix:** Compute decision `saver_available` as `saver.state == "active" or (saver.state == "not_earned" and streak < 10)` (keeping `uninitialized`, `used`, and `not_earned` at `streak >= 10` fail-closed `False`).

### F2. Dashboard Streak Saver displays and mutates `season=2026` until first 2027 fetch
- **Severity:** `WRONG PICK STATE`
- **State:** `2027-03-03..03-23` (pre-A4) with snapshot `contest_streak.json` (`source_date="2026-09-27"`) and `saver_state.json` (`season=2026`). Kickoff S1 leaves `load_contest_streak_state` unchanged.
- **Wrong outcome:** `web._saver_season` and `health/contest_state.check` call `load_contest_streak_state` directly and pass `source_date=2026-09-27` into `season_for(..., now_year=2027)` -> `2026`. The dashboard renders 2026's saver state/buttons, `POST /saver/transition` mutates `season=2026` while `season=2027` stays `uninitialized`, and `health/contest_state` checks 2026's flag.
- **File and line:** [web.py](file:///Users/eric/projects/bts-agy-audit/src/bts/web.py#L137-L145), [web.py](file:///Users/eric/projects/bts-agy-audit/src/bts/web.py#L180-L183), [contest_state.py](file:///Users/eric/projects/bts-agy-audit/src/bts/health/contest_state.py#L128-L130).
- **How demonstrated:** Ran `saver_dashboard_context` and `saver_transition_response` at `2027-03-10T12:00-04:00`: dashboard showed `state="active", button="mark_used"` and `POST` returned `(200, "saver -> used")`, writing `"season": 2026` while `load_saver_state(..., season=2027)` stayed `"uninitialized"`.
- **Smallest fix:** In `season_for` (or `_saver_season` and `health/contest_state.py:128`), clamp `max(source_date.year, now_year)` or pass `None` when `source_date.year < now_year`.

### F3. Partial MLB rollover (`2027` `rounds.json` before profile reset) defeats Kickoff S1 and corrupts 2027 saver state
- **Severity:** `WRONG GRADE OR STREAK`
- **State:** Cron `fetch-contest-streak` runs 4x/day in March 2027 while static `rounds.json` already lists 2027 round IDs (`9001..`) and `user/{id}/profile` still holds 2026 `activeStreak`/`seasonBestStreak` and 2026 `predictions`.
- **Wrong outcome:** `derive_source_date` skips unknown 2026 `roundId`s and returns `None`. `fetch-contest-streak` writes `source_date=null`, `recorded_at` in `2027`, and 2026's streaks. Kickoff S1 row 5 (`no source_date, recorded_at in now_year -> used`) then feeds 2026's `activeStreak` into Opening Night picks, and `maybe_auto_earn_saver(season=season_for(None, now_year=2027))` sticky-transitions 2027 `saver_state.json` to `"active"` if 2026 `best_streak >= 10`.
- **File and line:** [contest_fetch.py](file:///Users/eric/projects/bts-agy-audit/src/bts/contest_fetch.py#L55-L68), [cli.py](file:///Users/eric/projects/bts-agy-audit/src/bts/cli.py#L2046), [cli.py](file:///Users/eric/projects/bts-agy-audit/src/bts/cli.py#L2104-L2105), [contest_state.py](file:///Users/eric/projects/bts-agy-audit/src/bts/contest_state.py#L97-L98).
- **How demonstrated:** Ran `derive_source_date` with 2027 `rounds` and a settled 2026 prediction (`-> None`), `maybe_auto_earn_saver` (`2027: not_earned -> active`), and `select_pick` on `2027-03-24` (`p=0.77`, `streak=18` -> `"skip"` vs `streak=0` -> `"single"`).
- **Smallest fix:** In `derive_source_date`, raise `ContestFetchError` when `predictions` contains settled entries (`streakResult is not None` or settled pick status) and zero `roundId`s match `rounds`.

### F4. Opening Night and Day-2 EOD fire `CRITICAL` `contest_state` DMs despite Kickoff S2
- **Severity:** `FALSE ALARM`
- **State:** `2027-03-24 ~22:15 ET` after `run_result_polling` grades `2027-03-24.json` (`hit`/`miss`), before `01:10 ET` `fetch-contest-streak` runs; and `2027-03-25 ~22:15 ET` when MLB's `predictions` array lags by 1 settled pick (`source_date=None`).
- **Wrong outcome:** Because `run_result_polling` ([scheduler.py](file:///Users/eric/projects/bts-agy-audit/src/bts/scheduler.py#L3273)) settles today's root-level pick before `run_all_checks` ([scheduler.py](file:///Users/eric/projects/bts-agy-audit/src/bts/scheduler.py#L3311)), a settled 2027 pick already exists at Opening Night EOD. Kickoff S2's suppression ("no settled root-level pick dated in `now_year`") evaluates false, firing a `CRITICAL` DM on Day 1 EOD and again on Day 2 EOD under normal 1-pick ledger lag.
- **File and line:** [contest_state.py](file:///Users/eric/projects/bts-agy-audit/src/bts/health/contest_state.py#L61-L69), [scheduler.py](file:///Users/eric/projects/bts-agy-audit/src/bts/scheduler.py#L3273-L3318).
- **How demonstrated:** Saved settled `2027-03-24.json` (`hit`) alongside `13:30 ET` `contest_streak.json` (`source_date=None, recorded_at=2027-03-24T17:30Z`); `latest_resolved_pick_date` returned `2027-03-24` (disabling S2) and `health.contest_state.check(..., expected=True)` returned `CRITICAL` on both `2027-03-24` and `2027-03-25`.
- **Smallest fix:** When `state.source_date is None` and `recorded_at` is in `now_year`, emit `INFO` (not `CRITICAL`) while settled root-level picks in `now_year` are `<= EXPECTED_OVERNIGHT_LAG_STEPS + 1` (`<= 2` at EOD before overnight fetch).

### F5. `park_drag_freshness` pages `CRITICAL` on Opening Night EOD, and empty reseeded store crashes `refresh()`
- **Severity:** `FALSE ALARM`
- **State:** Opening Night `2027-03-24` EOD (`07:45 ET` cron ran pre-game, so `max_source_game_date` is still `"2026-09-27"`), or `2027-03-25 07:45 ET` after reseeding `pitches_current.csv.gz` header-only for 2027 per checklist B7.
- **Wrong outcome:** (a) `OFFSEASON_MONTHS = {10, 11, 12, 1, 2}` excludes March, so Opening Night EOD computes `gap_days = 178 >= 6` and sends a `CRITICAL` DM. (b) Beyond B7's known 2026-store error, an empty 2027 `pitches_current.csv.gz` crashes `park_drag_producer.refresh()` at `int(store["game_date"].dt.year.max())` (`ValueError: cannot convert float NaN to integer`) and `store["game_date"].max().date()` (`NaT`), preventing post-Game-1 self-bootstrap.
- **File and line:** [park_drag_freshness.py](file:///Users/eric/projects/bts-agy-audit/src/bts/health/park_drag_freshness.py#L36), [park_drag_freshness.py](file:///Users/eric/projects/bts-agy-audit/src/bts/health/park_drag_freshness.py#L74-L90), [park_drag_producer.py](file:///Users/eric/projects/bts-agy-audit/src/bts/features/park_drag_producer.py#L311-L320).
- **How demonstrated:** Ran `park_drag_producer.refresh` on `2027-03-25` with header-only `pitches_current.csv.gz` (`{"ok": false, "error": "cannot convert float NaN to integer"}`) and `park_drag_freshness.check` on `2027-03-24` (`CRITICAL: park_drag table data 178d behind`).
- **Smallest fix:** In `park_drag_producer.py`, default `season = today.year` and `window_start = date(season, 3, 1)` when `store.empty`; in `park_drag_freshness.py`, cap `gap_days` at `(today - season_opener).days` once regular-season games exist in `today.year` (suppressing before then).

### F6. Active `fetch-contest-streak` cron in `research` mode DMs on expired 2026 cookies; wrong handle in `bts-leaderboard.service`
- **Severity:** `FALSE ALARM`
- **State:** Snapshot restored `2027-03-03` with `entry_intent="research"` and expired Oct 2026 cookies before A4 re-auth.
- **Wrong outcome:** `cron-setup-hetzner.sh` gates `check-pick-entered` on `entry_intent="enter"` but installs all four daily `fetch-contest-streak --dm-recipient stonehengee.bsky.social` lines unconditionally, sending `AuthError` DMs every 6h cooldown window starting `2027-03-03 01:10 ET`. Separately (read-only finding), `deploy/systemd/bts-leaderboard.service` passes `--dm-recipient stoneericm.bsky.social` instead of `stonehengee.bsky.social`.
- **File and line:** [cron-setup-hetzner.sh](file:///Users/eric/projects/bts-agy-audit/scripts/cron-setup-hetzner.sh#L64-L66), [cron-setup-hetzner.sh](file:///Users/eric/projects/bts-agy-audit/scripts/cron-setup-hetzner.sh#L89-L92), [cli.py](file:///Users/eric/projects/bts-agy-audit/src/bts/cli.py#L2021-L2022), [bts-leaderboard.service](file:///Users/eric/projects/bts-agy-audit/deploy/systemd/bts-leaderboard.service#L15).
- **How demonstrated:** Invoked `bts fetch-contest-streak` via `CliRunner` with `AuthError("401 expired cookie")`: exited `2` and sent failure DM to `stonehengee.bsky.social`; inspected `cron-setup-hetzner.sh` and `bts-leaderboard.service`.
- **Smallest fix:** Gate `fetch-contest-streak` cron lines (or `--dm-recipient`) on `ENTRY_INTENT="enter"`, and change `stoneericm.bsky.social` to `stonehengee.bsky.social` in `bts-leaderboard.service`.

### F7. Opening Night 1-game partial-lineup lock (`9 < 10` batters) permanently wedges `live_forward_capture`
- **Severity:** `SILENT DATA LOSS`
- **State:** Opening Night `2027-03-24` (1 game): Team A posts its 9-batter lineup at `T-180`/`T-120` while Team B has not posted yet.
- **Wrong outcome:** `_fetch_prior_lineup(team_b, 2027)` returns `[]` (0 completed 2027 games), so `_lock_decision_from_predictions` sees 0 projected contenders, locks, and delivers `2027-03-24.json`. `live_forward_capture_once` exports 9 rows with `manifest.top_n=10`; `verify_candidate_artifact_pair` fails `*_top_n_rows_per_date` (`9 != 10`). On later 15-min ticks after Team B posts (18 batters available), `manifest.json` exists and `stale_pick_snapshot` is `False`, so `capture_once` re-verifies the 9-row dir (`failed_verify_existing`) without re-exporting, paging `CRITICAL` at EOD and permanently via `live_forward_resolution` from `2027-03-31` (`age >= 7d`, no `horizon_days`).
- **File and line:** [artifacts.py](file:///Users/eric/projects/bts-agy-audit/src/bts/experiment/artifacts.py#L212), [artifacts.py](file:///Users/eric/projects/bts-agy-audit/src/bts/experiment/artifacts.py#L1463-L1470), [live_forward_capture_once.py](file:///Users/eric/projects/bts-agy-audit/scripts/live_forward_capture_once.py#L514), [live_forward_capture_once.py](file:///Users/eric/projects/bts-agy-audit/scripts/live_forward_capture_once.py#L648-L667), [live_forward_resolution.py](file:///Users/eric/projects/bts-agy-audit/src/bts/health/live_forward_resolution.py#L106-L126).
- **How demonstrated:** Ran `_fetch_prior_lineup` (`[]`), `_lock_decision_from_predictions` (`should_lock=True`), `capture_once` Tick 1 with 9 batters (`failed_verify` on `*_top_n_rows_per_date`), Tick 2 with 18 batters (`failed_verify_existing`, `export_calls=1`), and `analytics_artifacts_missing.check` (`CRITICAL`).
- **Smallest fix:** Record `effective_top_n = min(top_n, len(scored_slate))` in `materialize_live_candidate_profile_pair` (checking `<= expected_top_n` in `verify_candidate_artifact_pair`), and re-export pre-outcome in `capture_once` when `verification.json` is not `ok`.

### F8. Cross-season pooling in `skip_policy_shadow` and `shadow_eval`
- **Severity:** `COSMETIC`
- **State:** `data/picks/` retains `2026-*.policy_shadow.json` and `2026-*.shadow.json` (`context_stack_shadow_v2`) into 2027.
- **Wrong outcome:** `load_decision_records` and `build_shadow_cycle_status` glob `*.policy_shadow.json` / `*.shadow.json` across all years, pooling 2026 days into 2027's Bonferroni checkpoint looks (`30, 60, 100`) and 30-day shadow review threshold.
- **File and line:** [skip_policy_shadow.py](file:///Users/eric/projects/bts-agy-audit/src/bts/skip_policy_shadow.py#L230-L237), [shadow_eval.py](file:///Users/eric/projects/bts-agy-audit/src/bts/shadow_eval.py#L516), [cli.py](file:///Users/eric/projects/bts-agy-audit/src/bts/cli.py#L2585).
- **How demonstrated:** Ran `build_skip_policy_shadow_status` on 29 `2026-*` + 1 `2027-03-24` records (fired checkpoint `30` `"above_breakeven"` on Day 1) and `build_shadow_cycle_status` on `2026-09-27` + `2027-03-24` (`resolved_paired_days=2`).
- **Smallest fix:** Filter both globs by `date.startswith(f"{season}-")`, defaulting `season` to `now_et().year`.

## Checked and clean

- **Spring-training isolation (`03-03..03-23`):** `scheduler.fetch_schedule`, `lineup_collect.collect_for_date`, and `day_outcome.gather` on `2027-03-10` with `gameType="S"` returned `0` games and `DayOutcome("no_games", ok=True)` (no DM).
- **Pre-season `reconcile_results`:** Running `reconcile_results` at `2027-03-04 02:00 ET` left `2026-09-27.json` untouched and reset `streak.json` from `(14, False)` to `(0, True)`.
- **2027 horizon, tail policy, and `best_streak` guard:** `season_end_date("2027-03-24") == "2027-09-26"` (`186` days -> clamped `d=180`); `health.tail_policy.check` returned `0` alerts on `2027-03-24` and `2027-09-10`; `classify_best_streak` marked 2026 `source_date` `"untrusted"`.
- **Health lookback windows, backups, and R2 sync:** On `2027-03-24` with `2026-09-27` picks on disk, `blend_training`, `calibration`, `predicted_vs_realized`, `realized_calibration`, `mdp_policy_alignment`, `slate_auc`, `result_resolution`, and `backup_freshness` all returned `0` alerts; `sync_to_r2` preserved `pa_2026.parquet` + MDP `.npz` files without tripping the half-manifest guard when `pa_2027.parquet` is absent.

## What I ran

- Static literal/season scan across `src/bts/`, `scripts/`, and `deploy/`: `TZ=America/New_York .venv/bin/python .codex-review/audit/scratch/scan_season_bound.py`.
- Production-function probe suite (faking only HTTP/R2 transport, clocks, and `bts.dm.send_dm`, against real `data/models/mdp_policy.npz` and `mdp_tail_policy.npz`): `TZ=America/New_York MPLCONFIGDIR=.codex-review/audit/scratch/mpl .venv/bin/python .codex-review/audit/scratch/run_probes.py > .codex-review/audit/scratch/probes_output.json`.
