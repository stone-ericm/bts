# Kickoff: change set K2, a regular-season game after the stored finale is played as the last day, and the morning start warns that the stored boundary is behind (2026-11-26, version 1; contract check on `main` `a5fa7466`; the author builds on `main` after K1 merges, since both change `src/bts/scheduler.py`)

Finding C3-1 (Carbon slice 3 O1) of the closing test re-run (`CLOSING-TEST-2.md`); ruling R72 (K2). Tier A (pick path: the policy's day count). Session `bts-wr-fresh15`. Branch `fix/wr-k2`. R70 applies. Operator text is judged under R64.

## What is wrong on the base
`bts.strategy.SEASON_END_DATES[2027] = "2027-09-27"` (the day after the scheduled finale, 09-26). `strategy._days_remaining(date)` is `max(0, (end - date).days)`, so a regular-season makeup game on 2027-09-27 (MLB adds such days in ordinary seasons: 2024 had games on Sep 30 after a Sep 29 scheduled finale) counts 0 days left. With 0 days every regime skips: the longest-streak regime needs a positive count (`mdp_objective`), the tail and the reach-57 table at 0 days skip by construction. So the day gets no pick at streak 20, 56, 57 or 120 (`skip`, `reach57`, no reason text), and nothing tells the operator: `CLAUDE.md` says to "recheck the stored boundary if it moves", but no check compares the day's schedule with the stored boundary. The same holds for any year not in the table after September 28 (the default).

## The design
1. **The decision's day count is at least 1.** In `src/bts/strategy.py`, `_regime` and `_mdp_action_from` (the only two places where the decision path turns the slate date into a day count; `effective_pick_bar`, `resolve_policy_decision` and `select_pick`'s early regime check all go through them) use `_decision_days(date) = max(1, _days_remaining(date))`. `_days_remaining` itself is unchanged (the health check `health/tail_policy.py`, the display and the audit scripts that import it keep the raw count).
   - **Why unconditional inside the decision path:** those functions run only for a slate that has candidates, and candidates come only from the prediction slate's schedule fetch, which keeps regular-season games only (`is_regular_season_game`, shared with `scheduler.fetch_schedule`). So "the decision runs for D" already means "MLB's schedule, read for the prediction, lists a regular-season game on D". No new input is plumbed into `select_pick` (it is widely patched; `CLAUDE.md` Testing gotchas). The contract check must confirm, by search, that no caller of `_regime`, `_mdp_action_from`, `effective_pick_bar`, `resolve_policy_decision`, `decide_action` or `select_pick` reaches them for a date with no regular-season game in a way that changes a pick, a decision record or an alert; any such caller is listed with what it now sees.
   - **What "the last-day rule" means here:** with 1 day left, streak 57 or more plays the longest-streak rule (a double when doubles are allowed and a different-game second pick exists, otherwise a single); streak 55 or 56 looks up the reach-57 table at 1 day; lower streaks are in the tail regime at 1 day (its stop rule: skip only if `min(57, streak + 2)` is at most a TRUSTED season best). The contract check records the exact action for streaks 0 to 60 and saver on/off at 1 day.
2. **The morning start warns.** In `scheduler.run_day`, after the schedule fetch found games, after the `--dry-run` return and the state save, and before the morning state check: if `strategy._days_remaining(date) == 0`, print one journal line `SEASON BOUNDARY BEHIND SCHEDULE — …` and send ONE `WARN` DM per slate date, source `season_boundary`, incident key `season_boundary:behind_schedule`, through `dispatch_dm_for_health_alerts(..., status_path=health_state/season_boundary_dm_delivery_status.json, now_et_date=<the slate's date>)` (its OWN status file, the pattern of 4c and 6b-1, so it neither suppresses nor is suppressed by another alert). Sent in every delivery mode when a `dm_recipient` is configured (the stored boundary affects research picks too). Ordinary exceptions in it are contained and printed (`Season boundary check failed: …`); an interrupt is not caught (the morning check's rule, 6b-1). The text:
   `Season boundary behind the schedule: MLB lists {n} regular-season game(s) on {D}, but bts.strategy.SEASON_END_DATES gives {E} as the day after the last game day, so the stored count is 0 days left. The pick for {D} is decided as the season's last day. Ask the BTS lead session or Eric to update the stored boundary to the day after the new last regular-season game day; until then each such day is decided as the last day.`
   (`{E}` is `season_end_date(D)`; `{n}` the number of games the morning fetch returned.)
3. **Where the schedule fact comes from:** for the decision, the prediction slate's own schedule fetch (the candidates); for the warning, the scheduler's morning `fetch_schedule(date)` (`state.games`). **When the schedule is unreadable:** the morning fetch raising is today's behaviour (W5: the scheduler restarts under systemd; no warning, no check); a prediction slate that cannot be read gives no candidates and so no decision (today's behaviour); a schedule that lists no game for D is today's off-day path (no warning, no decision).

## Concrete examples (R70 item 2): must change (RED on the base)
| # | Sequence | Base | K2 |
|---|---|---|---|
| E1 | `select_pick`/`resolve_policy_decision` on 2027-09-27 with a candidate at p 0.80 and a different-game second pick, streak 57, doubles allowed | `skip`, `reach57` | `double`, `longest_streak` (1 day left: three or fewer) |
| E2 | the same at streak 120 | `skip` | `double` |
| E3 | the same at streak 56 and 55, saver on and off | `skip` | the reach-57 table's action at 1 day (recorded by the check) |
| E4 | the same at streak 20 with a trusted best of 18 and of 25 | `skip`, no reason | best 18: the tail at 1 day (plays); best 25: `skip` with the tail's stop reason (`min(57, 22) <= 25`) |
| E5 | a year not in the table (2028) on 2028-09-29 | `skip` | as E1 to E4 at 1 day |
| E6 | `run_day` on 2027-09-27 with one game in the morning fetch (captured transport) | no warning | one journal line and one WARN DM with the text above; a second `run_day` start the same day sends no second DM |
| E7 | the season walk (`scripts/audit/season_walk/`) on 2027-09-27 | every state skips | the actions of E1 to E4 |

## Concrete examples: must still happen (identical to the base)
| # | Sequence | Must still happen |
|---|---|---|
| S1 | every date of the 2027 season with 1 or more stored days left (03-24 to 09-26), every streak 0 to 60 and 120, saver on/off, with the shipped artifacts | every action, objective, `days_effective`, guard and pick bar equal to the base's (the check proves it over the whole grid) |
| S2 | the health check `tail_policy` on any date (in and out of season, 09-27, 2028) | alerts byte for byte the base's (it keeps the raw count) |
| S3 | `run_day` on any date with 1 or more stored days left | no warning, nothing else changed |
| S4 | an off-day (no games) on 09-27 or later | the base's off-day path; no warning |
| S5 | the morning state check, its skip conditions and its DM | unchanged; the new warning comes before it and does not depend on the delivery mode or the contest-state settings |
| S6 | `--dry-run` | no DM (the warning comes after the dry-run return) |
| S7 | the decision record (`decision.json`) and the pick's `policy_decision` | the base's fields; on a floored day `days_effective` is 1 |

## Standing rows
- **An interrupt mid-step:** the warning is outside the pick send; an interrupt there leaves `run_day` as the morning check's does (nothing durable in flight).
- **A malformed or non-regular file:** the warning's status file goes through 6b-2's reader (anything not a regular JSON object is no prior status: the DM is sent again). A malformed slate date makes `_days_remaining` raise as on the base (the warning's failure is contained; the decision reports it where it always did).
- **Midnight ET and the year boundary:** the day count comes from the slate date, never the clock; a `run_day` started after midnight for D uses D; 12-31 and 01-01 have no regular-season games; an unconfigured year (2028) gets the floor and the warning, besides the existing `season_end_alerts` WARN from the health check.
- **A failed or unconfirmed send:** the dispatcher's `failed` record; the next start that day sends again.
- **A restart mid-step:** the status file dedups the warning per slate date; the decision is stateless.

## Known and not fixed
1. Two or more makeup days after the stored finale are each decided as the last day (true count 2 on the first): the reach-57 boundary (streak 53 or 54) and the tail's stop can differ from a correct table.
2. A boundary that is behind by some days while the stored count is still positive (MLB moves the finale later) is not detected; only the zero case is.
3. The audit scripts that replay strategy decisions over 2021 to 2025 (years not in the table, default September 28) would now see 1 day where they saw 0 on games after September 28 if re-run; their frozen results are not re-run.
4. `bts run`, `bts preview` and `bts orchestrate` get the floor (they decide through the same functions) but not the warning (only the scheduler's morning start warns).
5. The fix of the boundary itself is a code change (the table), owner-deployed.

## The author's work (tests first; ONE commit)
Files: `src/bts/strategy.py`, `src/bts/scheduler.py`, a new `tests/test_wr_k2_season_boundary.py`, the `SEASON_END_DATES` comment and `CLAUDE.md`'s "Season-bound values" sentence. Tests first, RED on the base for E1 to E7, through the real `select_pick` / `resolve_policy_decision` with the shipped artifacts, the real `run_day` (captured schedule and DM transports, pinned ET clocks, temporary picks directories), and the season walk. Existing tests that pin the touched functions are found BY SEARCH (`grep -rlE "strategy|_days_remaining|SEASON_END_DATES|season_end_date|_regime|_mdp_action_from|effective_pick_bar|run_day|_morning_state_check" tests scripts`, per `CLAUDE.md`'s Testing gotchas) and pass; any existing assertion that changes is one the contract check lists by running; anything else is a STOP.

## Review
Gate; a fresh Codex xhigh review and a Carbon review; at most two rounds (round 2 bounded, R70); merge on a Codex SIGN or SIGN WITH EDITS applied.
