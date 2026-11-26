# Adversarial review: four fixes from the Gemini findings (branch `g2-fix`)

Round 1 of at most 2. You are a fresh reviewer with no part in writing this. Work read-only in this checkout. The change under review is the top commit: `git show HEAD` (branch `g2-fix`, one commit on top of `main`). Do not read `~/projects/claude-shared`, `~/projects/bts-c2-framing-results` or `~/bts-runs`, or any other checkout of this repository.

This code runs a real contest account's daily pick in 2027 and the nightly data chain it depends on. Assume the author is wrong somewhere and find where.

## Background
- The four fixes and the contract each was written to are in the commit message and in the section "The contracts" below.
- Where the findings came from: `docs/audit/2026-11-14-gemini-known-answer-tests.md` (Test 2) and `docs/audit/2026-11-14-gemini-known-answer-pickpath-report.md` (B3, B4, notes 1 and 2).
- Project rules and rulings: `CLAUDE.md` ("Work policies", "2027 season"); the rulings ledger in `docs/superpowers/plans/2026-11-12-2027-season-roadmap.md` (R1–R15). Do not re-litigate a ledgered ruling unless it produces a wrong outcome the ledger does not name.

## The contracts
- **F1** `bts data build`: for a requested season with NO regular-season games available (raw directory missing, or no eligible feed), print one stderr line, write nothing, leave any existing parquet untouched, exit 0; other requested seasons still build; a season that HAS eligible games and fails still raises. The cron line `pull && build && { preview ; sync-to-r2 ; }` is unchanged.
- **F2** `scheduler.plan_run_times`: on a thin slate (one or two games) no two planned runs are within `cluster_min` of each other; an early check near another run merges into one run at the earlier time with the union of games. Larger slates are planned exactly as `compute_run_times` plans them.
- **F3** `bts day-outcome`: (a) new clean state `entered_game_unverified` = a receipt bound to this day exactly as `entered` requires, before the cutoff, with legacy `verifier.reason == "match"` and every qualification slot `confirmed` or `unit_unverified`; `entered` still wins; (b) a schedule whose games are ALL postponed or cancelled is `no_games`, with or without a decision record; zero games is unchanged; (c) both cron runs check yesterday, at 01:30 and 08:30.
- **F4** the `late_delivery` health message for a postponed-game refusal names the label and the game(s), not the batter.

## What to look for
For each fix: a concrete input or state where production now does the WRONG thing, especially one that was right before this commit.
- **F1:** an in-season state where the new no-op hides a failure that used to stop the chain (a season directory whose feeds are all unreadable, truncated or mid-download; a feed-format change that makes every game ineligible; the 7-inning filter; a `--seasons` list); what `eligible_games` actually counts and when. Then follow the chain it now lets run: with NO `pa_<year>.parquet` for the current year (the state from the box's restore until the day after the opener, and on Opening Day morning), what do `bts preview`, `bts data sync-to-r2`, the scheduler's prediction cascade (`bts.orchestrator.run_and_pick` and what it loads) and the health checks do? Was that state reachable before this commit, and is anything now worse than the chain simply aborting?
- **F2:** a thin slate where merging changes WHEN a game gets its last check before first pitch in a way that loses a delivery the unmerged plan would have made, or drops a game from every run; interaction with `early checks already in the past are dropped`, with the post-loop deadline wait, and with `state.runs_completed` / coalescing in `run_day`.
- **F3:** a receipt history where `entered_game_unverified` reports clean but the pick was not entered, or was entered on another game of a doubleheader, beyond what the contract itself accepts; precedence between `entered`, `entered_game_unverified`, `entered_late`, `not_entered`; `already_confirmed` chains; receipts for an earlier version of the decision. For (b): a day wrongly called `no_games` (suspended games, a schedule answer missing `status`, a pick already delivered and scoreable). For (c): which contest date `$YESTERDAY` is at 01:30 and 08:30 box time, around midnight and the DST changes, and what the 01:30 run sees while the scheduler is still polling results or the 01:00 grader holds its lock.
- **F4:** any consumer that parsed the old wording.
- **False greens:** a test added or changed in this commit that would still pass with the behaviour it names broken.

Write and run your own adversarial probes against the real functions and the real CLI (`click.testing.CliRunner`), with only the network, the clock and the DM transport faked. Run Python as `.venv/bin/python`; focused tests: `TZ=America/New_York .venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Do not run the whole suite. Do not edit tracked files.

## Rules for findings
A BLOCK names a concrete input or state, the wrong production outcome, the file and line, and the smallest fix, and says how you demonstrated it. A finding about test-code quality is a NOTE unless the test makes a false claim about behaviour that nothing else covers. A false alarm or a lost delivery matters as much as a wrong delivery.

## Deliverable
Write the report file named below with: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Blocking findings`, `Notes`, `False greens`, `What I ran`. Under 150 lines. When the file is complete, print the marker line named below and then `DONE`.
