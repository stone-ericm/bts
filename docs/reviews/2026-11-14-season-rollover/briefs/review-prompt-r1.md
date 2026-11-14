# Adversarial review, round 1 of at most 2 (tier A): season rollover of the streak state, data-path hardenings, a last attempt for a day with no pick

You are a fresh reviewer with no part in writing this. Work read-only on tracked files in this checkout, which is at the commit under review. The change is everything since the base commit `12dbaaa`: `git diff 12dbaaa HEAD` (three commits by two authors, merged). Do not read `~/projects/claude-shared`, `~/bts-runs`, or any other checkout. Do not commit or push. Do not use the network (fake it in probes).

This code runs a real contest account's daily pick in 2027. The season opens 2027-03-24 with one game. Assume the authors are wrong somewhere and find where. A lost pick, a wrong streak, a false alarm and a missed alarm all count.

## The contracts
The author briefs are the contracts, as tables of states: `.codex-review/sr-review/contracts/kickoff-a.md` (S1 to S5), `kickoff-a2.md` (supersedes kickoff A's definition of an observation's season: the fetch time first), `kickoff-b.md` (S6). A row marked "unchanged" must behave exactly as at the base. Background: the "Season-start hazards" list and rulings R1 to R22 in `docs/superpowers/plans/2026-11-12-2027-season-roadmap.md`; `CLAUDE.md`. Do not re-litigate a ledgered ruling unless it produces a wrong outcome the ledger does not name.

## What to establish
1. **Per item S1 to S6:** a concrete state where production now does the wrong thing, above all one that was right at the base. Compare with the base by running both where you can (`git show 12dbaaa:<path>`).
2. **S1 and S2 (streak state and its health check):** year and time-zone boundaries; manual, expired-manual, corrupt and partial files; `require_contest_state` true and false; the saver and season-best logic beside the new rule; every caller of `load_decision_streak_state` and `load_streak` (scheduler, CLI, dashboard, DM text, `decision.json` state fields, shadow paths) and whether any two now disagree; the first days of a season in order: first fetch, first pick delivered and unsettled, first settlement, first reconcile. Is there any state in which the bot now refuses to pick and cannot recover without a human?
3. **S3, S4, S5 (data path):** partial writes and interrupted downloads; temporary files seen by any `*.json` glob; an existing corrupt file; concurrent runs; every combination of season directory state and season-file state against the base; what the nightly chain `pull && build && { preview ; sync-to-r2 ; }` and the prediction refresh now do in each; a mid-season restore.
4. **S6 (scheduler):** drive the real `run_day` with only the network, the clock and the DM transport faked. A restart in the middle of the day; an extra attempt after a pick was refused or archived; a skip recorded during an extra attempt; games already started; equal start times and doubleheaders; a day that already delivered; the interplay with coalesced checks, the heartbeat and watchdog sleeps, and result polling after a late delivery. The new alert: sent exactly once, never on a skip day or a day without games, what happens across restarts and in private or research mode. The worst-case number of extra prediction cascades, and when.
5. **Tests:** false greens (a test that still passes with the behaviour it names broken). For every assertion changed in an existing test file (`tests/test_contest_state.py`, `tests/health/test_contest_state.py`, `tests/test_contest_state_best_trust.py`, `tests/test_data_build_empty.py`, `tests/test_cold_start_integration.py`): was it rightly updated, or weakened?

Write and run your own adversarial probes against the real functions and the real CLI (`click.testing.CliRunner`). Run Python as `.venv/bin/python`; focused tests: `TZ=America/New_York .venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Do not run the whole suite. Probes go under `.codex-review/sr-review/scratch/`.

## Rules for findings
A BLOCK names a concrete input or state, the wrong production outcome, the file and line, the smallest fix, and how you demonstrated it. A finding about test-code quality is a NOTE unless the test makes a false claim about behaviour that nothing else covers.

## Deliverable
`.codex-review/sr-review/report-codex.md`, under 150 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Blocking findings`, `Notes`, `False greens and changed assertions`, `What I ran`. When the file is complete, print `SRRC-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
