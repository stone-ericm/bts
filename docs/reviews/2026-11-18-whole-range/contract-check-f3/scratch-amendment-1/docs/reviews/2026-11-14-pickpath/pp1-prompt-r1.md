# Adversarial review: pick-path changes of 2026-11-12/13 (tier A)

Round 1 of at most 2. You are a fresh reviewer. Work read-only in `/Users/eric/projects/bts` at `HEAD` (`git log --oneline -1`). Do not read anything under `~/projects/bts-c2-framing-results/` or `~/bts-runs/`.

These changes will run a real contest account's daily pick in 2027: which batter is delivered, whether it is delivered in time, how the result is graded, and what the streak state becomes. Nothing is deployed yet. Assume the author was wrong somewhere and find where.

## The changes (production code about 470 lines; read each commit with `git show <sha>`)
1. `29e10bf` season end per year (`src/bts/strategy.py`, `src/bts/health/tail_policy.py`).
2. `1137cbe` the nightly cron builds the current season (`scripts/cron-setup-hetzner.sh`).
3. `9998bbf` thin-slate early lineup checks (`scheduler.plan_run_times`), then `6467cfc` the post-loop wait for the fallback deadline.
4. `222d18e` `bts day-outcome` (`src/bts/day_outcome.py`, `src/bts/cli.py`, two cron lines).
5. `48f6f01` a BTS Pass graded void (`src/bts/picks.py`: `_boxscore_grade`, `_grade_pick_pre_suspension`, `grade_pick_in_feed`). The contest rules are in `docs/audit/2026-09-29-incident-register-evidence/rules/section6.txt`.
6. `327beae` reconcile's season replay includes today's resolved pick (`picks._replay_season_streak`).
7. `4199360` a date's result applied to the streak at most once (`picks.update_streak`, `save_streak`, `applied_dates`).
8. `b205ae0` the delivery chokepoint refuses a postponed or cancelled game (`scheduler._void_committed_games`, `_refuse_void_game_delivery`).

Context: `CLAUDE.md` (sections "Work policies" and "2027 season"), `docs/superpowers/plans/2026-11-12-2027-season-roadmap.md` (the rulings ledger R1–R10 states each design choice and its cost if wrong; do not re-litigate a ledgered choice unless it produces a wrong outcome the ledger does not name).

## What to look for
For each change: a concrete input or state under which production now does the WRONG thing, especially one that was right before the change. In particular:
- **Grading (5):** a real MLB feed shape where a miss now grades as a Pass or a Pass as a miss; a hit lost; interactions with `check_hit`'s fallback search, `effective_daily_result`, double-down slot results, `shadow_eval`, `reconcile_receipt`, and the daemon's `run_result_polling` (does any path still treat a hitless line as a miss, or now treat a real miss as void and keep a streak that the contest reset?).
- **Streak (6, 7):** an ordering of the daemon scorer, the 01:00 `check-results`, the 02:00/07:40 `reconcile`, a crash, a restart, or `--allow-stale-scoring` backfill where the streak or saver ends wrong; whether the `applied_dates` marker can block an update that should happen (a date scored, then corrected; a doubleheader; year rollover; the 30-date bound).
- **Scheduler (3, 8):** a slate or sequence where the new early checks or the post-loop wait deliver the wrong pick, deliver twice, deliver after the cutoff, skip a delivery that used to happen, or block the rest of the day (doubleheader re-checks, next-day wakeup, result polling); a skip day that now waits or delivers; the guard refusing a playable game or passing a postponed one; what happens to the day after a refusal.
- **Season end (1):** any caller that still sees the wrong days-left; the health check's behaviour.
- **day-outcome (4):** a real day that it reports clean but was not, or a normal day it would DM about every night (think through an ordinary delivered-and-entered day with ~56 receipts, a policy skip day, an off day, a doubleheader day, a day the entry checker confirmed only via `already_confirmed`).
- **False greens:** a test added in these commits that would still pass if the behaviour it names were broken.

You may run focused tests: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest <files> -q -p no:cacheprovider`. Do not run the whole suite. You may write throwaway probes under `/tmp`; do not edit the repository.

## Rules for findings
Tier A under `docs/superpowers/notes/work-policies-2026-11-12.md`. A BLOCK names a concrete input or state, the wrong production outcome, the file and line, and the smallest fix. A finding about test-code quality is a NOTE unless the test makes a false claim about behaviour that nothing else covers. Say which findings you verified by running something and which are from reading.

## Deliverable
Write `/Users/eric/projects/bts/.codex-review/pickpath-2026-11-13/report-r1.md` with: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Blocking findings`, `Notes`, `False greens`, `What I checked`. Under 200 lines.

When the file is complete, print one line `PP1-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
