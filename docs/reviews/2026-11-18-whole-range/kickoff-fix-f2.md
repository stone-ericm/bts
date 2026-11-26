# Kickoff: change set F2, the saver flag against a dated observation that says MLB used the saver (2026-11-25, version 1; base `main` at `a4e8278d`; F2 is independent of F5, which touches only the entry watchdog)

Finding C2-B1 of the closing test, its CODE side (`CLOSING-TEST.md`; `closing-test/carbon-slice2/report.md` B1; ruling R60: "nothing tells the operator that the saver flag disagrees with a dated observation that says the saver was used"). Its TEXT side landed with Ea (the stale-reading DMs now tell the operator to run `bts saver-state --use`). Tier A (saver state feeds the live policy). Session `bts-wr-fresh13` (drafted by `bts-wr-fresh12`). Branch `fix/wr-f2`.

## What is wrong on the base
On a day with a dated observation in effect (`bts set-contest-streak --date <today> … --saver used`), the live decision takes the saver from the OBSERVATION and neither reads nor writes the flag `account_state/saver_state.json` (change set 4a). The next day without a dated observation, the live automatic path reads the FLAG (change set 3: `active`, or `not_earned` with the streak below 10, count as in hand). The automatic fetch never marks a saver used (it can only earn one). So when MLB has used the saver, the operator files `--saver used` on the dated observation, and does not also run `bts saver-state --use`, the next automatic day's policy counts a saver that is spent. Nothing alerts: on a day with a dated observation in effect `health.contest_state.check` returns from its dated branch after the observation's own saver check and never looks at the flag.

## The design
1. **One new WARN in the existing dated branch** of `bts.health.contest_state.check` (the `if dated_observation_in_effect(state, now):` block), source `saver_state` (`SAVER_SOURCE`), evaluated after the existing `not_earned`-in-zone check and before its `return`: when the observation's `saver_state` is `used` AND the flag for the caller's current ET season (`load_saver_state(picks_dir, season=<ET year of now>)`, the read the non-dated branch already uses) is `active` or `not_earned`. Nothing else in the check changes; no other alert is added, removed, reworded or reordered.
2. **The text (R64).** Live mode, with `Y` the ET year of `now`, `F` the flag's state and `DIR` the shell-quoted picks directory:
   `Streak Saver: today's dated observation says MLB used the saver, but the saver flag for season Y reads F. When no dated observation is in effect, the live decision reads the flag and counts this saver as still in hand. Mark it used: \`bts saver-state --use --season Y --picks-dir DIR\`. Run it in Y ET away from midnight. It changes only the flag; the dated observation stays as it is.`
   Historical mode (the after-midnight end-of-day read, change set C1): `The dated observation says the saver was used while the saver flag does not. The saver flag may still count it as in hand.` followed by C1's fixed pointer, exactly as the other historical saver messages are built; no command.
3. **Not covered, written as limits:** the observation says `active` and the flag says `used` (the next automatic day counts an available saver as spent: conservative, not a wrong pick in the dangerous direction); the observation says `active` and the flag `not_earned` (counted in hand below streak 10, spent from 10; conservative from 10); the flag is `uninitialized` or for another season (already counted as spent; the existing uninitialized warning fires on non-dated days). A day with no dated observation is unchanged. The decision itself is unchanged (R40's class: no state is written by health).
4. **Delivery of the alert** is the existing one: the end-of-day health run and its DM dispatcher (`saver_state` is in `attention.py`'s live-policy set); it shares the `saver_state` source and its dedup with the existing dated and flag warnings (CLAUDE.md, change set 4a's known limit).

## Must still catch (each must behave exactly as on the base)
| # | State | Must still happen |
|---|---|---|
| K1 | dated observation `not_earned` with streak 10 to 15 | the existing observation WARN, text byte for byte, and in historical mode the existing historical WARN and the early return |
| K2 | dated observation `used`, flag `used` | no new alert |
| K3 | dated observation `used`, flag `uninitialized`, absent, unreadable, a JSON list, or for another season | no new alert (counted spent already) |
| K4 | dated observation `active` or `not_earned`, any flag | no new alert |
| K5 | no dated observation in effect (another date, legacy override, automatic file only, no file) | the base's alerts, byte for byte, in live and historical mode |
| K6 | every other alert of the check (missing state, coverage, future date, legacy override, the live refusal appended last) | unchanged, same order |
| K7 | a dated observation that is invalid (a `ContestStateError`) | the base's behaviour |

## Must change (RED on the base)
| # | State | Base | F2 |
|---|---|---|---|
| M1 | today's dated observation `used`, flag `active` for this season, live end-of-day run | no saver alert | the WARN of design 2 |
| M2 | the same with flag `not_earned` | none | the WARN, `F` = `not_earned` |
| M3 | the same, after midnight (historical mode for the processed slate) | none | the historical WARN, no command |
| M4 | following the WARN's command exactly | — | the flag reads `used`; the next day's live decision without a dated observation counts the saver as spent; the WARN does not fire again for that state |
| M5 | Carbon's reproduction (`closing-test/carbon-slice2/`, B1: day 2 `saver_available True`, flag `active`) | silent | the WARN on day 1; after the command, day 2 `saver_available False` |

## Standing rows
- **An interrupt mid-step:** health only reads; an interrupted run sends nothing or the base's partial set.
- **A malformed or non-regular file:** the flag read is the existing reader (a non-regular or undecodable flag reads `uninitialized`: no new alert, K3); in historical mode the existing regular-file read.
- **Midnight ET and the year boundary:** the flag's season is the ET year of `now` (as everywhere since change set 3); a dated observation is in effect only on its own ET date; on December 31 the command names that year, and the text says to run it in that year away from midnight; after midnight the historical text names no command.
- **A failed or unconfirmed send:** the existing dispatcher and status file; a failed DM is retried by the next run that presents the alert.
- **A restart mid-step:** not touched.

## The author's work (tests first; ONE commit)
`src/bts/health/contest_state.py` (the one block), a new `tests/test_wr_f2_saver_flag.py`. No other file. The existing test files that pin the check (found by searching the tests for `health.contest_state`, `SAVER_SOURCE`, `saver_state`: `tests/health/test_contest_state.py`, `tests/test_contest_dated_cs4a.py`, `tests/test_contest_dated_cs4a_review.py`, `tests/test_saver_cs3.py`, `tests/test_saver_cs3_review.py`, `tests/test_season_contract_cs4b1_edges.py`, `tests/test_season_contract_cs4b2.py`, `tests/test_season_contract_cs4b2_review.py`, `tests/test_wr_b_contract.py`, `tests/test_wr_b_edges.py`, `tests/test_wr_c1_clock.py`; the contract check confirms the list by running) must pass unchanged; if one must change, STOP and ask.

## Review
Gate; a fresh Codex xhigh review and a Carbon review; at most two rounds; merge on a Codex SIGN.

## Known and not fixed
1. Design 3's three directions are not warned.
2. The warning comes from the end-of-day health run, after the day's pick; the next morning's first automatic decision can come before the operator acts on it.
3. It shares the `saver_state` source and dedup with the other saver warnings.
4. Nothing writes the flag for the operator (by design: health never writes state).
