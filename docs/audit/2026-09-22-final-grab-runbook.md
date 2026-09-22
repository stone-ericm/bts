# Final leaderboard grab — one-pass runbook (season wrap W0.6)

**Owner authorization:** Eric, 2026-09-14 ("one bounded end-of-season grab") and plan approval 2026-09-22. **Ceiling decision (Eric, 2026-09-22, verbatim): "no request ceiling. when it comes time, let's carefully and courteously try to grab it all."** → the board walk has NO page ceiling (runs until the board ends); pacing, kill-switch, one-attempt and the 300-profile allocation (D5) are unchanged. Implemented `fed10b6`. **Code:** `scripts/final_leaderboard_grab.py` reviewed by Codex (2 design rounds, 5 code rounds) — **SIGN at `40e952e`**. Offline rehearsal = the wrapper test file `tests/scripts/test_final_leaderboard_grab.py` (58 tests at the signed revision); there is no live rehearsal.

## When
**2026-09-28, after 08:00 ET** (regular season ends 9/27; the ordinary result-correction window closes the next morning per the contest rules §6). Never before the last game is final. Never a second attempt without a fresh owner decision.

## Before (on the box, as `bts`)
1. `cd ~/projects/bts && git rev-parse HEAD` → must be **the SHA recorded as Codex-signed in `docs/audit/2026-season-wrap-index.md` (W0.6 row)**, or a descendant that did not touch `scripts/final_leaderboard_grab.py` or `src/bts/leaderboard/`; `git status --short` clean.
2. Credential health: `data/health_state/contest_streak_fetch_status.json` `last_success_at` within 24 h (the 4×/day `fetch-contest-streak` cron uses the same cookies). If it is failing, STOP — do not re-capture cookies for this.
3. Inputs frozen: `sha256sum docs/audit/2026-09-22-early-cohort-2026-05-01.json` matches the committed file; `uv.lock` unchanged.
4. Free space ≥ 2 GB under `data/leaderboard/` as a STARTING check only — with no page ceiling the archive size is not bounded in advance; watch `df -h` and the run root's size during the walk (each page body ≈ 50–100 KB gz). The run root `data/leaderboard/final_grab_20260928/` must NOT exist.
5. Rounds sanity (no auth): `curl -s https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/json/rounds.json | python3 -c 'import json,sys; r=json.load(sys.stdin)["rounds"]; print([x for x in r if x["date"].startswith("2026-09-27")])'` → exactly one round dated 2026-09-27 (the `yesterday` tab binds to it).
6. Record the accepted ceiling in this file and in `docs/audit/2026-season-wrap-index.md`.

## Run (one command, foreground, in a `tmux`/`screen` so an SSH drop cannot kill it)
```
cd ~/projects/bts && set -a && . ./.env && set +a && \
.venv/bin/python scripts/final_leaderboard_grab.py \
  --date 2026-09-28 --season 2026 \
  --final-round-date 2026-09-27 \
  --board-limit 300 --cohort-a 150 --cohort-b 150 \
  --rng-seed 20260928 \
  --i-have-owner-authorization 2>&1 | tee ~/logs/final_grab_20260928.log
```
Expected footprint: 1 login + 4 static + **every board page until the board ends** (≈317 at ~95k participants if the board lists everyone; unknown in advance — no ceiling) + 3 tabs + ≤300 profiles, jittered 2.0–4.5 s per request. Duration is therefore uncertain; an hour is a guess, not an assurance. The walk also ends on its own on ANY previously seen page or three consecutive full pages that add no new users (reported as incomplete, not silently capped); `board.warnings` flags listed users exceeding 1.5× the reported participant count. `plan.json` records `request_budget: null`, `board_ceiling: null` and the owner's words. `status.json` is rewritten after every request; watch live totals and warnings with `python3 -c 'import json;j=json.load(open("data/leaderboard/final_grab_20260928/status.json"));b=j.get("board") or {};print(j["terminal_state"],"requests",len(j["requests"]),"pages",b.get("pages"),"unique",b.get("unique_user_ids"),"last_rank",b.get("last_rank_reached"),"participants",b.get("all_participants_count"),"warnings",b.get("warnings"))'` alongside `df -h ~/projects/bts/data` and `du -sh data/leaderboard/final_grab_20260928`.

## Exit codes / terminal states
| exit | terminal_state | meaning |
|---|---|---|
| 0 | `complete` | walk exhausted, conflict-free, valid consistent metadata, listed == reported participants (census), every profile clean |
| 2 | `complete_with_population_gap` | walk exhausted but listed < reported (expected: unlisted participants); board is complete for what MLB lists |
| 2 | `complete_with_errors` | truncation (ceiling / contradiction / repeated page / error), profile or tab problems |
| 3 | `aborted_rate_limited` | 403/429 anywhere (incl. login) — **STOP. Do not rerun.** Evidence preserved; report to Eric |
| 4 | `aborted_write_failure` | disk/IO — check the `.STATUS_WRITE_FAILED` sibling marker |
| 5 | refusals / `aborted_budget` / `aborted_static_lookup` / `aborted_other` | nothing or little sent; read `problems` |

## After
1. Reconcile: `terminal_state`/`exit_code` in `status.json` vs the process exit; `requests_accounted == true`; `planned_unattempted` empty (or explained); check for a `final_grab_20260928.STATUS_WRITE_FAILED` marker.
2. Board: `board.walk_exhausted`, `termination_reason`, `population.status`, `all_participants_count`, `unique_user_ids`, `last_rank_reached`, `duplicate_conflicts`.
3. Cohort: `cohort.json` A/B/E_in_A/E_unfetched sizes match `plan.json`; `identity.json` has one record per requested id with a terminal status.
4. Hashes: every `requests[*].archived_sha256` matches `sha256sum` of the decompressed archive; artifact hashes match the parquet files.
5. Isolation: `data/leaderboard/{leaderboard_snapshots,user_picks,season_stats}` and `scrape_status.json` unchanged (compare mtimes / hashes against the 9/22 checkpoint manifest).
6. Back up: run `bts backup run --set archive` (covers `data/leaderboard/`), record the snapshot id; then proceed to the W0.7 final snapshot.
7. Record the outcome (state, counts, snapshot id) in `docs/audit/2026-season-wrap-index.md` and the exposure register.

## Operator cancellation
If you must stop the run (disk, time, doubt): Ctrl-C once. The process exits **130** and `status.json` records `terminal_state = cancelled_by_operator`, `exit_code = 130`, with the in-flight request marked `aborted` and every remaining planned request listed under `planned_unattempted`. Keep everything under the run root as evidence; do not delete, do not rerun without a fresh owner decision.

## What this run is not
Not a live rehearsal, not repeatable, not a census claim unless `population.status == census`. A `complete_with_population_gap` result is the expected honest outcome for a board that does not list every participant.
