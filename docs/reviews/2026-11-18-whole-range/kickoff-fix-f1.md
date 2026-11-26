# Kickoff: change set F1, the entry watchdog and the wrong game of a doubleheader (2026-11-25, version 1; base `main` at `1e69d0f0`)

Ruling R60 (`FINDINGS.md`), finding BL-B7 of the closing test (`CLOSING-TEST.md`; `closing-test/codex-blind/report.md` blocking finding 7, fixture `closing-test/codex-blind/evidence/interactions-entry-wrong_game.json`). Tier A (entry). A check is made STRICTER, so the list of what the base accepts and this refuses is written here before an author starts. Session `bts-wr-fresh12`. Branch `fix/wr-f1`.

## What is wrong on the base
`bts check-pick-entered` (`cli._check_pick_entered`) decides "entered" from batter ids alone (`contest_fetch.pick_entry_status`). When the recommended batter is entered in the OTHER game of a doubleheader it writes the marker `confirmed` (`health_state/pick_entry_check.json`), sends `✅ BTS pick entry confirmed for <date> (match).` if the day had been alerted, and every later run that day stops at `already confirmed`. The same run's sealed receipt (`entry_receipt.qualify`, computed at publish) already says `wrong_game` for the slot, from the season's local `units` captures. The operator is told the entry is right while the system holds proof that it is in the wrong game, and a hit in that game does not count for the recommended pick.

## The design
1. **One new read, on the match path only.** After `pick_entry_status` returns `(True, "match")` and `rec.verified(...)` has been called with that result, the run computes the game qualification of the delivered slots with the receipt's own functions and the run's own four responses: `entry_receipt.target_rows(success, pending, rounds, now.date())`, `entry_receipt.load_units(picks, now.year)`, `entry_receipt.qualify(slots, rows, bts_to_mlb, units)`, with `slots` built exactly as `EntryReceipt._selection` builds them (one helper in `entry_receipt.py` shared by both; the receipt's output stays byte for byte what it is). Nothing is computed on any other path.
2. **Only `wrong_game` changes the outcome.** If at least one slot's state is `wrong_game`, the run continues as a NOT-entered run with `reason = "wrong_game"`: no `confirmed` marker, no all-clear DM, exit code 1, and the existing alert path below the match branch (cutoff passed during the fetch; the escalation ladder; the markers `alerted` and `dm_failed`; a failed DM stays retryable), with the texts of item 4. Every other state (`confirmed`, `unit_unverified`, `wrong_round`, `ambiguous`, `missing`, `player_unverified`, `unsupported_duplicate_selection`) leaves the run exactly as on the base (a `confirmed` marker). `unit_unverified` (no capture, or a unit with no game binding yet) is the accepted limit the closing test names; `wrong_round` and `ambiguous` are contradictions in the captured data, not proof of a wrong game.
3. **The first wrong-game alert is its own tier.** A new escalation name `wrong_game`: when the reason is `wrong_game` and `"wrong_game"` is not in the marker's `escalations`, the alert is sent NOW, whatever other tiers are consumed (the closing-test case: the 18:30 missing-entry DM consumed `initial`; at 18:45 the operator has just entered the wrong game and must hear it at once, not at 30 minutes to the cutoff). Sending it consumes `wrong_game` and `initial` and, as today, `t30`/`t15` when the cutoff is that near. After it the existing `t30` and `t15` tiers apply unchanged. A failed first wrong-game DM follows the base's failed-DM rules (marker `dm_failed` when the day has no `alerted` marker yet, otherwise the marker's tiers unchanged), so it is retried on the next run.
4. **Texts (R64 applies).** `W` is the names of the slot or slots whose state is `wrong_game`, joined with ` + `; `T` is the recommended game's first pitch for the first such slot, from that slot's `game_time`, rendered as the base renders first pitch (`%-I:%M %p ET`); `m` is the base's `minutes_to_cutoff` (to the earliest cutoff of the pick's games).
   - First wrong-game alert, delivery confirmed: `⚠️ BTS entry is in the WRONG GAME in MLB app: W is entered, but not in the recommended game (first pitch T). m min to submit. Move the entry to that game.` When `T` cannot be read from the slot, the parenthesis is left out.
   - `t30` / `t15` with reason `wrong_game`: `⏰ Entry STILL in the wrong game: W — m min left to submit!`
   - Decision `locked_unconfirmed` (the base's first branch, which takes precedence in every tier): unchanged structure, with `entry` = `The MLB entry is in a different game than this pick's`.
   - stdout lines: `check-pick-entered: <date> entry is in the wrong game (W); re-verifying each run until cutoff` where the base prints `still not entered (...)`; the base's other lines unchanged.
5. **A game check that cannot run is the base's behaviour.** Any `Exception` from the new read (an undecodable or malformed capture, an unreadable file) is caught: one stderr line `check-pick-entered: game check unavailable (<ExceptionType>); batter match only`, and the run confirms as the base does. `KeyboardInterrupt` and `SystemExit` are not caught (they leave through the base's `BaseException` path: no marker written, the receipt published as `raised …`).
6. **The receipt** is not changed in schema or content: `verifier` stays the batter-only result (`ok: true`, `reason: "match"`, `game_qualified: false`), `qualification` is computed at publish as today. In the wrong-game state the only differences from the base's receipt are `marker_status` (`alerted` or `dm_failed`, not `confirmed`) and `exit_code` (1). `docs/ops/pick-entry-receipt-v1.md` gains one sentence saying so. `bts day-outcome` and `health/pick_entry.py` are not changed.

## What the base accepts and this refuses (complete list; the contract check confirms it by running a grid)
Exactly the runs where `pick_entry_status` is `(True, "match")` AND `qualify` gives `wrong_game` for at least one slot: the batter's single target-date row names a unit that every capture of the season binds to ONE game, in the row's round, and that game is not the slot's `game_pk`. For a double: one leg in the wrong game is enough, whatever the other leg's state. Nothing else moves.

## Must still happen (each row identical to the base: stdout, stderr, DMs, marker bytes apart from `receipt` and times, exit code, and the receipt apart from ids and times)
| # | State | Must still happen |
|---|---|---|
| N1 | no entry (`no_pick`), every tier and prior marker | the base's ladder, texts, markers, exit 1 |
| N2 | wrong batter or a missing second leg (`mismatch`) | as the base |
| N3 | right batter in the right game (bound unit) | `confirmed`; the all-clear DM after an alerted day; later runs `already confirmed` with no fetch |
| N4 | right batter; no capture, a unit absent from the captures, or a null `feedId` | `confirmed` as the base (the accepted limit; day-outcome reads `entered_game_unverified`) |
| N5 | `present_unverified` (crosswalk gap) | marker `present_unverified`, not terminal |
| N6 | a fetch failure of any of the four requests; identity mismatch | quiet skip, no marker |
| N7 | no pick file; not committed; outside the window; marker already `confirmed` | as the base, no fetch |
| N8 | decision `locked_unconfirmed` with `no_pick` or `mismatch` | the base's text byte for byte |
| N9 | a double with both legs in their right games | `confirmed` |
| N10 | wrong game seen, then the entry moved to the right game on a later run | `confirmed` and the all-clear DM (the base's alerted-to-confirmed transition) |
| N11 | `bts day-outcome` and the end-of-day `pick_entry` health check over the files each state leaves | computed by unchanged code; a wrong-game day that ends unresolved now gives the health WARN (`last status=alerted, reason=wrong_game`), which is true |

## Must change (RED on the base)
| # | State | Base | F1 |
|---|---|---|---|
| M1 | the closing-test sequence: 18:30 no entry (initial DM); 18:45 the batter entered in game 2 while the pick is game 1, units bound | marker `confirmed`, all-clear DM; 19:00 `already confirmed` | 18:45 the first wrong-game DM, marker `alerted` reason `wrong_game`, exit 1; 19:00 re-verifies |
| M2 | first run of the day already sees the wrong game | `confirmed`, no DM | the first wrong-game DM, marker `alerted`, escalations include `initial` and `wrong_game` |
| M3 | a double: one leg right, one in the wrong game | `confirmed` | the DM names only the wrong leg |
| M4 | wrong game persists to 30 and 15 minutes before the cutoff | nothing | the `t30` and `t15` texts, each once |
| M5 | wrong game, the DM fails | (confirmed) | `dm_failed` on a day with no alert yet; the alert is retried on the next run |
| M6 | wrong game, decision `locked_unconfirmed` | `confirmed` | the unconfirmed-delivery text with the new `entry` clause |
| M7 | wrong game and the cutoff passes during the fetch | `confirmed` | marker `alerted`, no DM, exit 1 (the base's cutoff branch) |

## Standing rows
- **An interrupt mid-step:** during the new read: no marker, no DM, the receipt says `raised KeyboardInterrupt` (item 5). Between the DM and the marker: as the base (the tier is not consumed; the DM can repeat on the next run).
- **A malformed or non-regular file:** a capture that is not gzip, not JSON, not an object, or whose `units` is not a list; a directory named like a capture: item 5 (stderr line, the base's confirmation). A named pipe at a capture path blocks the read: on the base the same read blocks at publish, after the marker and the DM; with F1 it blocks before the `confirmed` marker, on the match path only (the missing-entry alerts are not affected). Stated limit.
- **Midnight ET and the year boundary:** the target date and the captures' season are `now.date()` and `now.year`, the values the base's verifier and receipt use; a run at 23:50 and one at 00:05 read different days exactly as on the base.
- **A failed or unconfirmed send:** M5; a DM that raises after the transport accepted it can repeat (as the base).
- **A restart mid-step:** the command is one cron run; the marker write is atomic (as the base); an interrupted run leaves the earlier marker.

## The author's work (tests first; ONE commit)
`src/bts/cli.py` (`_check_pick_entered` only), `src/bts/entry_receipt.py` (the shared slots helper and one function that returns the qualification for a run), `docs/ops/pick-entry-receipt-v1.md` (one sentence), a new `tests/test_wr_f1_wrong_game.py`. No other file. If an existing assertion must change, STOP and ask.

## Review
Gate; a fresh Codex xhigh review and a Carbon review; at most two rounds; merge on a Codex SIGN.

## Known and not fixed
1. `confirmed` stays terminal for the day: a unit binding that arrives after a batter-only confirmation, or an entry moved to the wrong game after a confirmation, is not seen by the watchdog (day-outcome reads the receipts next morning).
2. With no capture, or a unit no capture binds, the wrong game is not detectable (the accepted limit); `wrong_round` and `ambiguous` rows confirm as on the base.
3. The run and its receipt read the captures separately: a capture landing between the two reads can make the receipt's `qualification` differ from what the run acted on.
4. The morning `bts day-outcome` is unchanged: a day that ends in the wrong game reads `not_entered` (when an earlier check saw no entry) or `entry_unverifiable`, both problem states with a DM; neither names the wrong game.
5. `m` is the time to the earliest cutoff of the pick's games, which for a double can be earlier than the wrong leg's own cutoff; the text does not know whether the game the entry sits in has already started.
6. A named pipe at a capture path (standing rows).
