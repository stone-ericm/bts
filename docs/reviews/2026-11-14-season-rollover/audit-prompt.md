# Season-rollover audit of the whole codebase (read-only)

You are a fresh auditor. This checkout is at the current `main`. Work read-only on tracked files; probes go under `.codex-review/audit/scratch/`. Do not read `~/projects/claude-shared`, `~/bts-runs`, or any other checkout. Do not commit or push. Do not use the network (fake it in probes).

## Context
This code picks one or two batters a day for a real contest account (Beat the Streak). The 2026 season is over; the production box was deleted on 2026-10-21 and will be restored from a snapshot about 21 days before the 2027 opener, with last season's files still on it (`data/picks/`, `data/health_state/`, `data/models/`, caches, the crontab). Opening Night is 2027-03-24 with one game; the season ends 2027-09-26. Read `CLAUDE.md` ("2027 season", "Season 2026 wrap", "Architecture") and the "Progress and rulings ledger" in `docs/superpowers/plans/2026-11-12-2027-season-roadmap.md` first.

## Already known: do not re-report, find what is NOT on these lists
- Roadmap ledger, "Season-start hazards" items 1 to 8, and rulings R1 to R22.
- `docs/reviews/2026-11-14-season-rollover/kickoff-draft.md`, items S1 to S6 (being fixed by other authors right now): last season's contest streak and `streak.json` on Opening Night; first-fetch health alarms; weather sidecar files in the feature scan; unvalidated feed downloads; the pre-season build no-op and an unreadable season directory; a day with no selected pick getting no last attempt.
- Checklist `docs/ops/2027-season-start.md` rows B7 (park-drag reseed) and B8.

## What to find
Anything in `src/bts/`, `scripts/cron-setup-hetzner.sh`, `deploy/`, and configuration defaults that will break, raise a false alarm, or silently do the wrong thing in one of these windows:
1. **Restore to Opening Night** (about 2027-03-03 to 03-23): the scheduler, every cron line and every health check running daily with no regular-season game, last season's state on disk, and spring-training games on the public schedule.
2. **Opening Night and the first seven days:** first picks, first grading, first reconcile, first entry check, the saver, the season-best logic, the tail policy, phase-aware bins, shadow models, live-forward capture, the dashboard, backups and the R2 sync.
3. **The year boundary in general:** literal years, season start and end dates, look-back windows that span the off-season, caches keyed by date or game, "current season" inferred from the clock, files whose meaning changes with the season.

Assume there is something. A finding needs: the state, the wrong outcome, file and line, how you demonstrated it (run the real function with only the network, the clock and the DM transport faked wherever you can; say when you only read), the smallest fix, and one severity: LOSES A PICK, WRONG PICK STATE, WRONG GRADE OR STREAK, FALSE ALARM, SILENT DATA LOSS, COSMETIC.

## Second deliverable: the season-bound table
Every place in `src/bts/` and `scripts/cron-setup-hetzner.sh` that contains a literal year (2025, 2026, 2027) or a season-bound constant or date (month-day literals such as `03-20`, `09-28`, season lengths, late-phase day counts), one row each: file and line, what it is, whether logic depends on it, and whether it is correct for 2027 (yes, no, or needs a decision, with the reason). Comments and docstrings count only when the code beside them is season-bound.

## Deliverables
- `.codex-review/audit/report.md`, under 160 lines: `Summary` (the findings ranked by severity, one line each), `Findings` (each with the fields above), `Checked and clean` (what you ran that came out right), `What I ran`.
- `.codex-review/audit/season-bound-table.md`: the table.
Run Python as `.venv/bin/python`; `TZ=America/New_York`. Do not run the whole test suite. When both files are complete, print `AUDITSR-<first 8 hex characters of the sha256 of report.md>` and then `DONE`.
