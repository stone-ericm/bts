# Final leaderboard grab — one-pass runbook (season wrap W0.6)

**Owner authorization:** Eric, 2026-09-14 ("one bounded end-of-season grab") and plan approval 2026-09-22. **Ceiling decision (Eric, 2026-09-22, verbatim): "no request ceiling. when it comes time, let's carefully and courteously try to grab it all."** → the board walk has NO page ceiling (runs until the board ends); pacing, kill-switch, one-attempt and the 300-profile allocation (D5) are unchanged. Implemented `fed10b6`. **Code:** `scripts/final_leaderboard_grab.py` reviewed by Codex (2 design rounds, 8 code rounds) — **SIGN at `1fdd460`** (2026-09-22; the no-ceiling revision). Offline rehearsal = the wrapper test file `tests/scripts/test_final_leaderboard_grab.py` (58 tests at the signed revision); there is no live rehearsal.

## When
**Owner decision (Eric, 2026-09-27 ~22:05 ET, verbatim): "tonight - and set a reminder to check tomorrow for any scoring changes in any games, not in any bts related scoring. if there are any you can back-fix it"** → the supplemental ran as `final_grab_20260927` (unit `bts-final-grab-20260927`, launched 22:10:45 ET, `--date 2026-09-27 --rng-seed 20260927`, everything else as below) against the board tabulated 18:51:54 ET; the 9/28 timing below is superseded. **Correction coverage:** after 08:00 ET 9/28 run `scripts/audit/bts_scoring_change_check.py --date 2026-09-27 --tabulated-at 2026-09-27T18:51:54-04:00 --out docs/audit/2026-09-28-scoring-change-check.json` — public MLB feed revisions only (no contest requests); it reports every batter whose BTS outcome changed between the tabulation and the cutoff, plus a current-boxscore cross-check. Verified both ways on 9/27: a mid-game window (16:00→18:30 ET) flags 124 changes by both detectors; tabulation→22:25 ET flags 0. Any streak-relevant change is back-fixed as an overlay (never touching raw archives): affected round-1009 pickers found in the capture (profile picks + raw `yesterday` tab predictions) → `back_fix_20260928.json` + `_backfixed` parquet copies + a corrections-index entry; board users whose 9/27 pick is not in the capture cannot be individually corrected. **Outcome (run 2026-09-28 09:46 ET → `docs/audit/2026-09-28-scoring-change-check.json`): 0 feed revisions after the tabulation in all 14 final games (BAL@NYY cancelled), 0 outcome changes, 0 current-boxscore discrepancies → `final_grab_20260927` is the final board; no back-fix.** **Corrected the same day (C-04):** feed revisions cannot show MLB's in-place re-scorings, so the final-board claim was re-checked against the 03:00 ET cached feeds and the capture's own graded picks — `docs/audit/2026-09-28-scoring-snapshot-check.json`: no change from 03:00 on and none in any captured pick; residual = 42 batters in the two games MLB touched between 18:51 and 03:00. Next season: save the box scores at tabulation time and compare those. **Closed 9/28:** a board-only pass after the cutoff (`final_grab_20260928`) matched the 9/27 capture for all 121,852 users (C-04) — the 9/27 capture is the final board.

*Original timing (superseded):* **2026-09-28, start ~08:30 ET** — after the morning tabulation. Official rules: "Sponsor will tabulate Streak scores as soon as practicable by 8:00am ET on the day following each MLB gameday", counting official stat corrections made before then; "Streak scores will not be revised or recalculated to reflect official MLB statistics changes or corrections that occur after 8:00 a.m. ET the day following the impacted game" ([official rules](https://www.mlb.com/apps/beat-the-streak/official-rules)). So the 9/27 board is final only after the 9/28 tabulation. Never before the last game is final. Never a second attempt without a fresh owner decision.

**Board freshness (observed; every board page and tab carries one `updatedAt`):** the 9/22 capture (read 13:39–14:10 ET) saw `2026-09-22T08:11:33-04:00` on all 243 pages and 3 tabs. A one-page probe on 9/27 at 21:58 ET (1 login + 1 season-best page at Eric's request; nothing stored) saw `2026-09-27T18:51:54-04:00` — 43 min after the last 9/27 game ended (HOU@ATH 18:08 ET; BAL@NYY was cancelled for rain and not rescheduled). So the board refreshes after a round completes, and (9/22) around the morning tabulation. After launch, record `board.per_page[0].updated_at` from `status.json` in the outcome; if it still shows the 9/27 evening value, note it — do not cancel and rerun.

## Before (on the box, as `bts`)
1. `cd ~/projects/bts && git rev-parse HEAD` → must be **the SHA recorded as Codex-signed in `docs/audit/2026-season-wrap-index.md` (W0.6 row)**, or a descendant that did not touch `scripts/final_leaderboard_grab.py` or `src/bts/leaderboard/`; `git status --short --untracked-files=no` empty (untracked generated outputs under `data/` are expected — 19 untracked entries on 9/27, none under `scripts/` or `src/`).
2. Credential health: `data/health_state/contest_streak_fetch_status.json` `last_success_at` within 24 h (the 4×/day `fetch-contest-streak` cron uses the same cookies). If it is failing, STOP — do not re-capture cookies for this.
3. Inputs frozen: `sha256sum docs/audit/2026-09-22-early-cohort-2026-05-01.json` matches the committed file; `uv.lock` unchanged.
4. Free space ≥ 2 GB under `data/leaderboard/` as a STARTING check only — with no page ceiling the archive size is not bounded in advance; watch `df -h` and the run root's size during the walk (each page body ≈ 50–100 KB gz). The run root `data/leaderboard/final_grab_20260928/` must NOT exist.
5. Rounds sanity (no auth): `curl -s --compressed https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/json/rounds.json | python3 -c 'import json,sys; r=json.load(sys.stdin)["rounds"]; print([x for x in r if x["date"].startswith("2026-09-27")])'` → exactly one round dated 2026-09-27 (id 1009, `complete`; the `yesterday` tab binds to it). `--compressed` is required: the server sends gzip even unasked, and without it `json.load` dies with `UnicodeDecodeError`. Also expect round 1010 dated 2026-09-28 (`scheduled` since at least 9/24; no MLB games that day): harmless — the script binds `yesterday` to `--final-round-date` by date, not to the latest round.
6. Record the accepted ceiling in this file and in `docs/audit/2026-season-wrap-index.md`.

## Run (one command, as a transient systemd user unit so an SSH drop cannot kill it)
The box has no `tmux`/`screen`, and plain `nohup … &` children die ~1 min after the SSH session ends. The 9/22 capture ran the same way as unit `bts-final-grab-20260922`.
```
cd ~/projects/bts && export XDG_RUNTIME_DIR=/run/user/$(id -u) && \
systemd-run --user --unit=bts-final-grab-20260928 --collect \
  --working-directory=/home/bts/projects/bts \
  -p EnvironmentFile=/home/bts/projects/bts/.env \
  -p StandardOutput=append:/home/bts/logs/final_grab_20260928.log \
  /home/bts/projects/bts/.venv/bin/python scripts/final_leaderboard_grab.py \
    --date 2026-09-28 --season 2026 \
    --final-round-date 2026-09-27 \
    --board-limit 300 --cohort-a 150 --cohort-b 150 \
    --rng-seed 20260928 \
    --i-have-owner-authorization
```
stderr follows stdout into the log (systemd's default `StandardError=inherit`). Follow it with `tail -f ~/logs/final_grab_20260928.log`. Afterwards read the exit status with `journalctl --user -u bts-final-grab-20260928 --no-pager | tail`; `systemctl show` on a `--collect`ed unit reports defaults.

Expected footprint: 1 login + 4 static + **every board page until the board ends** (9/22: 243 pages for 72,600 listed users; unknown in advance — no ceiling) + 3 tabs + ≤300 profiles, jittered 2.0–4.5 s per request. Duration is uncertain; 9/22 took 51 min for 551 requests. The walk also ends on its own on ANY previously seen page or three consecutive full pages that add no new users (reported as incomplete, not silently capped); `board.warnings` flags listed users exceeding 1.5× the reported participant count. `plan.json` records `request_budget: null`, `board_ceiling: null` and the owner's words. `status.json` is rewritten after every request; watch live totals and warnings with `python3 -c 'import json;j=json.load(open("data/leaderboard/final_grab_20260928/status.json"));b=j.get("board") or {};print(j["terminal_state"],"requests",len(j["requests"]),"pages",b.get("pages"),"unique",b.get("unique_user_ids"),"last_rank",b.get("last_rank_reached"),"participants",b.get("all_participants_count"),"warnings",b.get("warnings"))'` alongside `df -h ~/projects/bts/data` and `du -sh data/leaderboard/final_grab_20260928`.

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
6. Back up: `set -a && . ./.env && set +a && ~/.local/bin/uv run bts backup run --set archive` (`uv` is not on PATH over ssh; covers `data/leaderboard/`), record the snapshot id; then proceed to the W0.7 final snapshot.
7. Record the outcome (state, counts, snapshot id) in `docs/audit/2026-season-wrap-index.md` and the exposure register.

## Operator cancellation
If you must stop the run (disk, time, doubt): send ONE SIGINT — the script's Ctrl-C path — with `XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user kill -s SIGINT bts-final-grab-20260928`. Never `systemctl --user stop`: that sends SIGTERM, which the script does not handle, so it would die without writing its terminal state. On SIGINT the process exits **130** and `status.json` records `terminal_state = cancelled_by_operator`, `exit_code = 130`, with the in-flight request marked `aborted` and every remaining planned request listed under `planned_unattempted`. Keep everything under the run root as evidence; do not delete, do not rerun without a fresh owner decision.

## What this run is not
Not a live rehearsal, not repeatable, not a census claim unless `population.status == census`. A `complete_with_population_gap` result is the expected honest outcome for a board that does not list every participant.
