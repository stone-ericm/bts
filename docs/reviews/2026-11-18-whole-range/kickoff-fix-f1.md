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

## Amendment 1 (2026-11-25, after the contract check `WRCHKF1-d2c4d503`, BUILD WITH THESE EDITS; `contract-check-f1/`). It overrides the text above wherever they differ.

The check reproduced the fault and ran a 6,480-cell grid on a scratch build. It showed four things the version above gets wrong: the instruction "Move the entry to that game" can make a right entry wrong (a capture can lag a rescheduled game; a double can be named for the wrong leg); a saved slot with no game id reads as `wrong_game`; the first wrong-game alert consumed `initial`, so an entry REMOVED afterwards got no alert before 30 minutes to the cutoff; and after a wrong-game alert a later capture that makes the unit's binding unknown produced a false all-clear. The edits below take the checker's texts. Where the session did NOT take a replacement, it says so and why (the session's ruling, reported to the lead).

### A1. When the game check applies (replaces design items 1, 2 and "What the base accepts and this refuses")
On `(True, "match")`, after `rec.verified(...)`, the run computes `Q = qualify(slots, rows, bts_to_mlb, units)` as in item 1. The run is NOT confirmed, and continues as a not-entered run with `reason = "wrong_game"`, in exactly two cases:
- **(a) a wrong game is recorded now:** the saved slots do not share a batter id, and at least one slot has state `wrong_game` AND a saved `game_pk` that is an exact `int` (not `bool`). A slot with a null or non-integer `game_pk` never counts (the base's confirmation).
- **(b) an earlier wrong-game observation of the same ET date is not yet resolved:** the day's marker (same `date`) has `reason == "wrong_game"` or lists `wrong_game` in `escalations`, and not every slot of `Q` is `confirmed`. (So after a wrong-game alert only full game proof of every leg confirms and sends the all-clear; a capture that turns the binding unknown keeps the day unresolved.)
In every other case the run confirms exactly as the base does. When two saved slots share a batter id the game check is skipped (case (a) cannot arise; case (b) needs `all_confirmed`, which `qualify` cannot give there, so a day already warned stays unresolved).
"Affected legs" are: in case (a) the slots that satisfy (a); in case (b) with none of those, every slot whose state is not `confirmed`.

### A2. Tiers (replaces design item 3)
- The first wrong-game alert is the tier `wrong_game`, due at once when `reason == "wrong_game"` and `wrong_game` is not in the marker's `escalations`, whatever else is consumed. A successful send (or no recipient, as the base) consumes `wrong_game` and, as the base does, `t30`/`t15` when the cutoff is that near. **It does not consume `initial`:** if the entry is later removed or the batter changed, the base's `initial` alert is still due at once.
- After it, `t30` and `t15` apply with the reminder text of A3 (shared with the base ladder: each fires once a day).
- Every marker this change writes keeps the tiers already consumed (`escalations` = the prior list, plus what was consumed now), including a `dm_failed` marker.
- A failed first wrong-game DM writes `dm_failed` when the day's marker is not `alerted`, otherwise `alerted` with the prior tiers, so it is retried on the next run.
- The all-clear is the base's, unchanged in text and conditions (`was_alerted or "initial" in prior_escalations`), plus: it is also sent when `wrong_game` is in the prior `escalations`.

### A3. Texts (replaces design item 4 and known limit 5; the checker's texts, R64)
`LEGS` is the affected legs joined with `; `, each as `<role> <name>: recommended game <game_pk> (first pitch <T>), the entry is recorded in game <observed>` where `<role>` is `pick` or `double-down`; `<name>` is the batter name, or `batter <id>` when the name is not a non-empty string; `(first pitch <T>)` is the slot's `game_time` in ET as the base renders first pitch, left out when it cannot be read; `, the entry is recorded in game <observed>` names the game the captures bind the entered unit to and is left out when the leg is not `wrong_game`. No text gives a countdown or says MLB still permits a change.
- First alert: `⚠️ BTS entry game needs verification. LEGS. Check the delivered pick and the game shown in MLB for each listed leg. Change only a leg whose correct game you can verify and that MLB still lets you edit. Keep any correct leg as entered.`
- `t30`/`t15`: `⏰ BTS entry game verification is still needed. LEGS. Check the delivered pick against the game shown in MLB.`
- Decision `locked_unconfirmed` (takes precedence in every tier): `⚠️ Delivery of this pick was NOT confirmed. The scheduler tried to send <names as the base renders them>. MLB game evidence needs verification. LEGS. Check your DMs and the MLB entry before making any change. Run `bts season-contract status --picks-dir <quoted directory>`.`
- stdout where the base prints `still not entered (...)`: `check-pick-entered: <date> game verification unresolved (<roles of the affected legs>); re-verifying each run until cutoff`. The cutoff branch keeps the base's line.

### A4. Rows changed
- N4, N9: unchanged, and they hold only on a day with no earlier wrong-game observation (A1 (b)).
- N10: the all-clear needs every leg `confirmed` (A1 (b)).
- M2: `escalations` holds `wrong_game` and not `initial`.
- M3: the text lists only the wrong leg; the other leg is not named.
- New M8: wrong game alerted, then the entry removed before 30 minutes to the cutoff: the base's `initial` not-entered DM at once.
- New M9: wrong game alerted, then a capture makes the unit's binding unknown: no all-clear, marker `alerted` reason `wrong_game`, the reminder tiers apply.
- New N12: a saved slot with a null `game_pk`; two slots with the same batter: the base's confirmation on a day with no earlier wrong-game observation.
- New N13: a batter name that is null or empty in a wrong-game leg: the alert is sent with `batter <id>`; nothing raises.

### A5. Not adopted (the session's ruling), each written as a known limit
1. **No bounded, validated capture reader and no `game_check_unavailable` state.** The checker asks that a malformed, unreadable or slow capture inventory keep an otherwise matching entry unconfirmed, with a reader bounded in time and size. Not built: the base confirms on the batter match today, one bad capture file would then page for every correct entry of the season, and it is a second mechanism. Item 5 stands: an `Exception` from the new read gives the stderr line and the base's confirmation (on the base the same exception at publish gives one stderr line and no sealed receipt; with F1 there are two stderr lines and still no sealed receipt; the morning check then reports the evidence gap). A named pipe at a capture path blocks the run before the `confirmed` marker on the match path (the base blocks at publish).
2. **The run and the receipt keep separate capture reads** (limit 3 above; the checker's capture-change cases: a WARN with a morning `entered`, or a confirmation with a morning `entry_unverifiable`).
3. **No separate reminder tiers per kind and no delivered-warning record.** `t30`/`t15` are shared; with no DM recipient tiers are consumed as on the base; an all-clear DM that fails is not retried (the base); a cutoff run rewrites a `dm_failed` marker as `alerted` (the base's cutoff branch).
4. **Inherited from the base, unchanged:** a marker file holding a JSON list or null, invalid UTF-8, or tier values that cannot be iterated raises; a marker path that cannot be replaced raises after the DM; an unreadable saved game time raises before the fetch; numeric-string round or player ids in a response can turn a recorded wrong game into `wrong_round` or `player_unverified`, which confirm; several candidate rows for one batter are `ambiguous` and confirm; overlapping runs can both send.

### A6. Known limits, replacing limits 1 to 6
1. A completed `confirmed` stays terminal for the ET date: a later move of the entry or a later capture is not seen by the watchdog.
2. The captures are a stable local inventory, not the live binding: after a rescheduled game whose new binding is not yet captured, a RIGHT entry can read `wrong_game`. The texts therefore ask for verification and never tell the operator to move an entry; such a day ends with the health WARN unless a later capture proves every leg.
3. With no capture, an unbound unit, conflicting bindings or a round contradiction, a wrong game is not detectable and the batter match confirms (unless the day already has a wrong-game observation).
4. A rollback to the base: the base overwrites an `alerted`/`wrong_game` marker with `confirmed`, sends its all-clear and stops checking. Treat a confirmation from rolled-back code as batter-only.
5. The morning `bts day-outcome` is unchanged. It reads the sealed receipts by priority: timely game proof (`entered`), timely batter-only evidence (`entered_game_unverified`), late game proof, observed absence (`not_entered`), then `entry_unverifiable`. It never names a wrong game, and an earlier clean receipt can outrank a later wrong-game observation. With only wrong-game observations it reads `not_entered` when a bound receipt also saw no entry, else `entry_unverifiable`.
6. Captures are selected by the UTC year in their file name: a run late on December 31 ET can miss captures named for the new UTC year.
7. A5's four items.

### A7. Standing rows, corrected
An interrupt during the new read: no marker written and no DM; the receipt is published best-effort with `raised KeyboardInterrupt` (it needs its own capture read to succeed). `SystemExit` records its code. Midnight: the target date and season are the run's start in ET. The other rows stand as written with A5.

### A8. Existing tests
The check ran all 63 test files its search found against a scratch build: no existing assertion pins a behaviour this change alters (`contract-check-f1/test-findings.txt`; the one failure it lists is an artefact of where its scratch copy sat, the others are this machine's known environment failures). The author's file list gains nothing; an existing assertion that must change is still a STOP.

## Amendment 2 (2026-11-25, the author's first stop: a contract statement was false). It overrides the text above wherever they differ.
The standing row "a malformed or non-regular file" promised the stderr line for every capture whose `units` is not a list. The existing reader `entry_receipt.load_units` is NOT changed by this change set, and it only raises for some shapes. Corrected row: **the stderr line and the base's confirmation (item 5) apply to exactly the captures for which the unchanged `load_units` raises** (for example bytes that are not gzip, a truncated gzip, text that is not JSON or not UTF-8, a top level that is not an object, a `units` value that cannot be iterated, a directory or an unreadable file at a capture path). **Every shape the unchanged reader accepts without raising contributes whatever bindings it yields, usually none** (for example `units` null, false or absent; a `units` mapping, which is iterated by its keys; entries that are not objects or lack an integer `id`): the slot then reads `unit_unverified` and the run confirms as the base does, with NO stderr line. The author's tests pin, for each malformed shape, the outcome the unchanged reader gives (raise or silent) and that the run's result is the base's confirmation in both; they do not require a diagnostic the reader does not produce. Known limit added to A6: a malformed capture that the reader accepts silently can hide a binding (a wrong game is then not detected, as limit 3) and gives no diagnostic.

## Amendment 3 (2026-11-25, the lead's ruling on A5.1: partly overruled). It overrides the text above wherever they differ.
The lead: a plain `confirmed` when the capture cannot be read is a false all-clear in exactly the doubleheader case; a new unconfirmed state that keeps nagging is too much. Ruled middle, replacing design item 5's outcome and A5.1's "the base's confirmation":
- **When the new read raises an `Exception`** (exactly the captures for which the unchanged `load_units`, `target_rows` or `qualify` raises; Amendment 2), the run says the BATTER is confirmed and the game could not be checked, and the day is treated as confirmed for later runs (no repeated check, no nag, exit 0):
  - stderr, as item 5: `check-pick-entered: game check unavailable (<ExceptionType>); batter match only`.
  - the marker is `{"date", "status": "confirmed", "reason": "match_game_unchecked", "checked_at"}` (terminal, as `confirmed` is);
  - where the base sends the all-clear DM (an alerted day, or `initial` or `wrong_game` in the prior tiers), the text is `✅ BTS pick entry: the batter is entered for <date>; the game could not be checked.` It has no "Fix it now", no countdown and no instruction to change the entry;
  - stdout: `check-pick-entered: <date> batter entered; the game could not be checked (match_game_unchecked)` in place of the base's `pick entered (match)`.
  - This holds on a day with an unresolved wrong-game observation too (A1 (b) cannot be evaluated without the read): the day ends as batter-confirmed, game unchecked. Known limit.
- A capture shape the reader accepts SILENTLY (Amendment 2) is unchanged: the base's `confirmed`/`match` and the base's texts.
- New row M10: a capture that makes the read raise, on a batter match, with and without an earlier alert: the marker, the DM (only after an alert), stdout and stderr above; a later run prints `already confirmed` and fetches nothing. The receipt for that run is as the base's (its own read fails at publish: no sealed receipt, one more stderr line).
- A5.1 otherwise stands: no bounded reader, no nagging state; a named pipe at a capture path still blocks.
