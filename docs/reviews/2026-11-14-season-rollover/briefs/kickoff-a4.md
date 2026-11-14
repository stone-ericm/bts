# Kickoff A4: the season rule for the streak state, restated in full, plus the saver (replaces the S1 and S2 rows of kickoffs A and A2)

The second reviewer (Gemini 4 Carbon) blocked on the saver's season, and a whole-codebase audit found that the decision feeds a saver that is not earned yet to the policy as "used". Working through both showed that my two earlier definitions of "which season does this observation belong to" were each wrong in one state. This brief replaces them with one table over all the states. S3, S4, S5 and your A3 fix stay exactly as they are.

Reports, if you want the detail: `/Users/eric/projects/bts/docs/audit/2026-11-14-season-rollover-carbon-r1.md` (B1 and the note on `test_contest_state_best_trust.py`), `/Users/eric/projects/bts/docs/audit/2026-11-14-season-rollover-audit-carbon.md` (F1 to F4).

## Definitions
- `now_year`: the America/New_York year of `now`.
- `recorded_year`: the New York year of the observation's `recorded_at` when it parses (the fetch time), else none.
- `source_year`: `source_date.year` (the latest settled round the profile shows), else none.
- `settled_this_year`: at least one root-level pick file dated in `now_year` has a settled result (`_resolved_pick_dates`).

## S1 v3. `load_decision_streak_state`: which streak the decision uses
An unexpired manual override is used exactly as today, whatever its dates. For the auto file or an expired manual file:

| # | `recorded_year` | `source_year` | `settled_this_year` | What it means | Required |
|---|---|---|---|---|---|
| 1 | any | `now_year` | any | a normal in-season observation | used, exactly as at the base (status by the existing freshness rules) |
| 2 | `now_year` | none or earlier | no | fetched this year; the profile shows no settled round of this season and no pick of ours has settled yet: the season has not started for this account | **season start: decision streak 0**, `source="contest"`, `status="season_start"`; the streak value in the file is ignored (an un-reset profile still shows last season's); `require_contest_state` is satisfied |
| 3 | `now_year` | none | yes | fetched this year, the profile has no settled round yet, our first picks have settled locally (the contest settles later) | used, as at the base (status `stale`: no `source_date`) |
| 4 | `now_year` | earlier | yes | fetched this year, our picks have settled, but the profile's latest settled round is still last season's | treated as absent: `ContestStateError` saying so when required, otherwise the model-only fallback |
| 5 | earlier | not `now_year` | any | a file left from last season | treated as absent, as in row 4, the message naming the season |
| 6 | none | earlier | any | a legacy file from last season | treated as absent |
| 7 | none | none | any | unknown | used, status `stale`, as at the base |

Model-only fallback (no observation, or treated as absent and not required): the model streak is `load_streak(picks_dir)` when `settled_this_year`, otherwise 0. (Unchanged from your A1.)

## S7. The saver the decision passes to the policy
The policy's saver input means "still in reserve": in `simulate/mdp.transition_outcomes` it starts true and only a miss at streak 10 to 15 turns it false. Today the decision passes `SaverState.is_available`, which is true only for `active`, so a season's `not_earned` (the state until the streak first reaches 10) reaches the policy as "used", and the tracked policy table chooses differently on that input at streaks 7 to 9.

| Saver state for season `now_year` | Decision streak | Passed to the policy |
|---|---|---|
| `active` | any | true |
| `used` | any | false |
| `not_earned` | below 10 | **true** (it cannot have been used: a save needs a streak of 10 to 15) |
| `not_earned` | 10 or more | false (earned by now, and we cannot tell whether it was used) |
| `uninitialized` | any, and not `settled_this_year` | **true** (the season has not started for this account) |
| `uninitialized` | any, and `settled_this_year` | false |

- The saver's season is always `now_year` in the decision loader, in `health/contest_state.check`, and in `web._saver_season` (today they use `season_for(source_date, …)`, which reads LAST season's flag when the profile's latest round is last season's). Do not change `SaverState.is_available` itself or `season_for`'s signature; change what these three callers pass, and compute the policy input in the decision loader.
- Fetch path (`cli.py`, the `fetch-contest-streak` command's call to `maybe_auto_earn_saver`): the season is the fetch's own year, and the `not_earned` to `active` promotion happens only when the observation's `source_date` is in that year (a `seasonBestStreak` read from an un-reset profile is last season's). The `uninitialized` to `not_earned` initialization is unchanged.

## S2 v2. `health/contest_state.check`, `expected=True`
| State (rows of S1 v3) | Required |
|---|---|
| rows 5 and 6 | CRITICAL naming the season the file is from and telling the operator to run the fetch |
| row 4 | CRITICAL: the contest profile shows no settled round this season although local picks have settled |
| row 2 | no CRITICAL for the missing or earlier `source_date` |
| row 3 with at most two settled root-level picks dated in `now_year` | no CRITICAL (the contest settles a day behind; the first days of a season look like this) |
| row 3 with three or more | CRITICAL as at the base ("investigate the fetch") |
| rows 1 and 7, and every other existing check | unchanged |

## Tests
- RED first. One parametrized test per table, every row, with `require_contest_state` true and false where it applies, auto file and expired manual file, and an unexpired manual override beside rows 2, 4 and 5.
- Row 2 with a file whose streak is 9: the decision streak is 0.
- S7: every row, and one end-to-end case through `select_pick` or `resolve_policy_decision` showing the policy receives true at streak 8 with `not_earned` (the table cell where the two inputs choose differently is days-left 180, streak 8, lowest bin: skip with false, single with true; assert the action).
- Restore end-to-end coverage that the reviewer found weakened: through `load_decision_streak_state`, an observation recorded this year whose `source_date` is last season's and which carries a `best_streak` must not yield a trusted best (`tests/test_contest_state_best_trust.py`).
- The fetch path: with `CliRunner` and a faked profile, a `seasonBestStreak` of 14 with last season's `source_date` leaves this season's saver `not_earned`; with a `source_date` in this year it becomes `active`.
- Mutations: for S1 v3 swap rows 2 and 3's outcomes; for S7 return `is_available`; for S2 v2 remove the two-pick tolerance. Report what each breaks.

## Deliverable
`.codex-review/sr/handover-a4.md`, under 60 lines: what changed per table, RED counts, mutations, the focused command with counts (add `tests/test_strategy.py` and any test file you touch), `git diff --stat` against `8479a96`, and any row you could not match or think is wrong: say so plainly, with the state that worries you. Do not commit. Print `SRA4-<first 8 hex characters of the sha256 of the handover file>` and then `DONE`.
