# Kickoff: change set H2, the contest-state caps above 57 (2026-11-25, version 1; base: the merge candidate `36fd3e35` = `main` with change sets G, Ea and H)

Rulings R65 and R66 (`FINDINGS.md`); Eric 2026-11-24: "after 56, try to achieve the longest streak possible". Tier A, and by the 2026-11-18 rule a change that LOOSENS A VERIFIER: one piece, a contract check, and the "must still catch" table below written before an author starts. Session `bts-wr-fresh11`. Branch `fix/wr-h2`.

## What is wrong after H
H keeps playing from streak 57. The state checks refuse every streak or best above 57, so after a hit at 57 the live state stops being accepted:
| # | Where | Today |
|---|---|---|
| S1 | `contest_state._validate_dated_observation`, ordered checks 5 and 6 | `active_streak` must be 0 to 57; `best_streak` must be `active_streak` to 57 |
| S2 | `contest_state` loader, the automatic file (`contest_streak.json`; change set 1) | refused for every caller when `streak > 57`, or best is below the streak or above 57 |
| S3 | `contest_state.classify_best_streak` | a best above 57 is `untrusted` |
| S4 | `contest_fetch.validate_fetch` | `seasonBestStreak must be <= 57` (and best must be at least active) |
| S5 | `bts set-contest-streak --date` (`cli.py`) | `--best-streak` at most 57 |
| S6 | `picks._valid_chosen_on` | a `dated` or `automatic` record with `streak > 57`, or a dated identity outside `0 <= active <= best <= 57`, loads as absent (so the pick can never be delivered) |
And (H's limit 1) a fetch refused for S4 leaves the last accepted automatic file (57/57) in use: H keeps picking on the stale 57 and nothing alerts until the reading is 14 hours old.

## The design (ONE constant; the lead ruled the value and form on 2026-11-25 01:45: 400, not date-aware; a typo between 58 and 400 is an accepted, written limit)
1. `contest_state.MAX_SEASON_STREAK = 400` replaces 57 in S1 to S6 and in their messages ("0 to 400", "at most 400"). `TARGET_STREAK = 57` stays as the name of the contest's target for everything that means the target (the tail policy's cap, displays). 400 is above anything a season can produce (two hits a day for 200 days; the 2027 season has 187 calendar days, so 374) and still catches a slipped digit that adds one (570, 1570). Not chosen: a date-aware cap (twice the days since Opening Night): there is no table of season starts, and a wrong table would refuse a true reading.
2. Nothing else changes: every other check in S1 to S6 (types, booleans, negative values, best at least the streak, dates, schema, provenance, the switch, the saver) is byte for byte as it is.
3. The policy side needs no change and is not changed: below 57 with a best above 57 the reach-57 table plays as it does today with best 57; in the tail regime `effective_best` is already capped at the target, so a trusted best of 58 or more stops exactly as 57 does; at 57 and above H's rule ignores the best (H's limit 4: after a miss from above 57 the account keeps playing by the reach-57 table and, late in the season, the tail's stop rule; the rule does not try to beat an earlier, longer run). The contract check confirms each of these three statements by running the policy with a trusted best of 58, 100 and 400.

## Must still catch (each row: an input the checks refuse today that they must still refuse, with the same error class and, where the message names no bound, the same message)
| # | Input | Must still happen |
|---|---|---|
| K1 | dated observation: `active_streak` -1, `True`, `"12"`, 12.0, null, missing | `ContestStateError`, check 5 |
| K2 | dated: `best_streak` below `active_streak`; `True`; a string; null; missing | `ContestStateError`, check 6 |
| K3 | dated: `active_streak` 401; `best_streak` 401 with active 57 | `ContestStateError`, checks 5 and 6 (the bound in the message is now 400) |
| K4 | automatic file: streak -1 or non-integer; best below the streak; streak 401; best 401 | refused for every caller (`ContestStateError`; the historical variant in the after-midnight health read) |
| K5 | fetch: `seasonBestStreak` below `activeStreak`; negative; non-integer; missing; 401 | `ContestFetchError`, nothing written, exit and alert as today |
| K6 | CLI: `--best-streak` below `--streak`; `--streak` -1; `--best-streak` 401; `--date` not today | `BadParameter`, nothing written |
| K7 | `chosen_on`: negative or boolean streak; dated identity with `active != streak`, best below active, or a value above 400; unknown kind | loads as absent |
| K8 | `classify_best_streak`: best negative, below the streak, above 400, wrong season, future date, unproven provenance | `untrusted` (never authorises the tail's stop) |
| K9 | every ordered check of the dated observation other than 5 and 6, the switch refusal reasons 1 to 5, the saver rule | unchanged: the full existing test files for change sets 1, 4a, 4b-1, 4b-2 pass with only the bound's assertions changed |
| K10 | **the stale 57 (R66):** MLB shows 58/58; the fetch | the fetch ACCEPTS it and the automatic file reads 58/58; a live decision is made at 58 under H's rule; no path leaves a refused reading above 57 with the old file silently in use. A reading above 400 is still refused, alerted through the existing fetch-failure path |

## What is given up, stated plainly (the contract check completes this list)
A typo between 58 and 400 is no longer refused: `--streak 75` meant as 57, or a best of 157 meant as 57. Today the cap catches it; after H2 the only guards are that best is at least the streak, the confirmation the command prints, `bts season-contract status`, the morning state check's journal line (`… accepted (streak n)`), and the automatic reading once the switch is on. Cost of such a typo: at 57 or more H plays whatever the number (the action does not depend on the streak); below 57 a typed streak above 56 turns a reach-57 decision into H's always-play rule for that day.

## Must change (RED on the base)
| # | State | Base | H2 |
|---|---|---|---|
| M1 | dated observation 58/58/used for today, through the real CLI and the live loader | the CLI refuses `--best-streak 58`; a hand-written file is refused by the loader's check 5 | accepted; the live decision is H's rule at 58; `chosen_on` valid; the pick deliverable |
| M2 | automatic: a fetch returning 58/58 (stubbed MLB), switch on | fetch error, file stays 57/57 | file 58/58, accepted by the live loader, best `trusted` |
| M3 | dated 60/75/used; automatic 12/60 | refused | accepted; below 57 the policy answers as it does for best 57 (show equality with the base's answer at best 57 over the table) |
| M4 | a saved pick whose `chosen_on` is dated 58/58 or automatic 58 | loads as absent, never delivered | valid; `saved_pick_refusal` judges it as any other |
| M5 | the season walk: the dated state `58/60/used` added | every date refused | plays as H's rule; no finding |

## Standing rows
An interrupt mid-step (the fetch's atomic write, unchanged); a malformed or non-regular file at each state path (unchanged refusals); midnight ET and the year boundary (a 58 reading written at 23:59 on 12-31: the switch's season rule, unchanged); a failed or unconfirmed send (not touched); a restart mid-step (a pick chosen at 58 before a restart is deliverable after it: M4); a rollback to the base code (a state file or `chosen_on` holding 58 is REFUSED by the base: after H2 is live a rollback past it at a streak above 57 stops picks; state it in the checklist with H's "never without H2").

## The author's work (tests first; ONE commit)
`src/bts/contest_state.py`, `src/bts/contest_fetch.py`, `src/bts/cli.py` (the one bound), `src/bts/picks.py` (the two bounds), the walk and its tests, a new `tests/test_wr_h2_caps.py`, and only the existing assertions the contract check lists by running them. No policy code, no scheduler, no operator text other than the bound inside the six error messages.

## Review
Gate; a fresh Codex xhigh review and a Carbon review; at most two rounds; merge on a Codex SIGN. With the merge: CLAUDE.md and the checklist say H and H2 deploy together and that a rollback past H2 at a streak above 57 stops picks.

## Known and not fixed
1. A typo between 58 and 400 is no longer refused (above).
2. A rollback past H2 with a state above 57 on disk stops picks (the base refuses the file and the saved pick's `chosen_on`).
3. The fetch still has no check that MLB's reading is plausible for the date; only the cap, non-negativity and best at least active.
4. Everything in change set H's "Known and not fixed" except its limit 1, which this change set removes.
