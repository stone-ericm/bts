# Verdict

**BLOCK** for `3f725779` against `988708df` under R64. The three replacements match the contract. The later-slate instruction can lose a day's pick.

## Blocking findings

**B1. [B8a, line 31](../../docs/ops/2027-season-start.md#L31), later-slate rule.**

**Harmful outcome:** On a one-game slate, following the new between-checks instruction for a season-best-only discrepancy can lose the day's pick when the final check and fallback return no fresh pick.

- The real scheduler sent a stale warning on July 15. On July 17, the 16:05 and 17:05 checks saved an undelivered single, Batter 0 at 99.0%, using automatic `12/14/used`. At 17:06 ET the prior entries were settled, MLB showed `12/15/used`, the slate was unlocked, and the 18:20 check and 18:30 fallback remained. Eric verifies the saved single and runs `bts set-contest-streak --date 2027-07-17 --streak 12 --best-streak 15 --saver used --picks-dir DIR`.
- The 18:20 check returned no predictions. The 18:30 refresh failed and tried the cached single. The delivery guard refused it because the CLI changed the observation from automatic to dated. The scheduler sent the refusal and missed-pick alerts and finished with zero pick DMs and no committed decision. With the old instruction Eric verifies the same pick and leaves the automatic state in use. In the same failure, fallback sent the single before cutoff. A successful refresh under the dated row selected and sent the same single. [Scheduler and CLI replay](probe-later-slate-final.log)
- Replace “If the state differs while the slate is unlocked and a later lineup check or fallback refresh remains, have Eric run the same command between checks.” with “While the slate is unlocked, if MLB differs from status and a later lineup check or fallback refresh remains, have Eric run the same command when no lineup check, fallback refresh or send is in progress, unless only the season best differs and Eric verifies that the policy would choose the same saved, unsent pick using MLB's state. In that case, wait for confirmed delivery and file only if it is still TODAY ET and MLB still differs from status.”

## Edits

None separate from B1. Its replacement also names the interval before a fallback that follows the last lineup check.

## Notes

- A write forced inside the pick transport succeeded but could not recall the in-flight DM. The send was still part of the first check, outside the new between-checks condition. The old after-start instruction also leaves that transport alone.
- An unlocked slate with all checks and fallback over falls under the accepted no-repick limit. The old and new instructions both leave the already-locked and exhausted cases without a dated write.
- Applied R64 as quoted in the brief and the contract. The checkout's `FINDINGS.md` ends at R63. `nice -n 19` reported `setpriority: Operation not permitted` and the bounded commands still ran.

## 1. Contract rows

| Row | What ran and what happened |
|---|---|
| C28 | Real July 15 delivery warning, July 17 `season-contract status`, scheduled checks and dated CLI. At 09:30 the July 16 entry was pending. The 16:05 check made no pick. After the 16:08 Pass, the new instruction files `15/15/used`. At 17:05 the real selection skipped at streak 15. The old instruction filed nothing and sent a double at streak 0. [Replay](probe-later-slate-final.log) |
| C29 | Real July 14 pick confirmed at July 15 00:00:20 ET. The warning prints `--date 2027-07-15`, while the pick belongs to July 14. B8a's equality rule rules out the warning command. The later-slate rule waits for July 14's still-open entry window, then files July 15 only after MLB confirms absence and status differs. [Replay](probe-delivery-warning-r2.log) |
| C30 | A real skip check persisted `Model ID below themodel`. The journal, captured skip DM and dashboard kept it verbatim. The base rendered `under themodel`. [Replay](probe-skip-display.log) |
| C31 | Restored each old expression separately on a scratch copy of `src/`. Both tests failed on the restored text. [Results](probe-mutations-r2.log) |

## 2. Copy and operator cases

| Text | Exact copy and outcome |
|---|---|
| Later-slate B8a | The rendered contract text and the Markdown source matched character for character after accounting for markup. Before the first check, settled, differing state filed as TODAY and drove the later selection. Matching state left the dated row absent. Between checks a best-only correction made the guard withhold a saved pick when fallback failed, and a successful refresh chose a dated replacement. With the send in flight, the induced write left the old send in progress. With the slate locked, Eric verifies the delivered pick and the saved bytes stay fixed. With the final window over there is no write or delivery. [Copy](probe-delivery-warning-r2.log), [timing replay](probe-later-slate-final.log) |
| Delivery warning and B8a twin | Replacing both B8a contract entries in the base reproduced the entire checklist byte for byte. The real rendered warning matched the replacement exactly, including after midnight. The old warning called the printed D the slate and said entries could be absent. The new warning uses the actual pick slate and requires closure for an empty window. On the pick's own date, a pending entry from the previous slate delays the write under both versions. After settlement, a differing state files for D. A dated `13/14/active` already in effect matches MLB and stops a second write even with automatic streak 12. [Replay](probe-delivery-warning-r2.log) |
| Stored display | The formatter expression matched the contract. Standalone lowercase words, including punctuation boundaries and `below\nthe` after line folding, render as `under the`. Embedded prefixes and suffixes, underscores, digits and capitalized variants retain their wording. K2 still prints `below the streak-10 bar` on all surfaces. [Replay](probe-skip-display.log) |

## 3. Source parity and size

Undoing the `re` import, warning sentence and regex substitution reproduces base `src/bts/scheduler.py` byte for byte. It is the only changed source file. The live warning comparison matched level `WARN`, source `contest_state`, incident key `contest_state:stale_reading`, status file and emission ET date, same-day suppression and next-day resend, and interrupt containment. The 128-character timestamp and 177-character detail produced an **896-character envelope**, versus 892 at base. The status receipt's body hash reflects the wording. [Warning parity](probe-delivery-warning-r2.log)

The live skip comparison matched every stored summary field, selection decisions, no-pick results and the timestamp and dedup of the skip notice. The source comparison leaves selection, delivery and the rest of the alert flow identical. [Skip parity](probe-skip-display.log)

## 4. Tests and scratch changes

- `test_wr_e_texts.py`, `test_stale_delivery_cs4c.py`, `test_morning_state_cs6b1.py` — **188 passed**. The initial 25-second group timed out. The 85-second bounded rerun completed in 36.61 seconds. [Test log](tests-texts-warnings.log)
- `test_scheduler_skip_visibility.py`, `test_scheduler_tail_objective.py`, `tests/scripts/season_walk/` — **66 passed**, including 18 season-walk tests. [Test log](tests-skip-walk.log)
- The restored delivery sentence failed `test_stale_warning_maximum_detail[delivery]`. The restored plain `.replace` failed `test_skip_stored_text_surfaces[longer-word]`. Both imported the scratch scheduler. The first mutation runner needed verbose pytest output to verify the differing words. [Warning mutation](mutation-delivery-slate.log), [display mutation](mutation-stored-substring.log)

ran: C28–C31, the ET timing and settlement probes, source and rendering comparisons, six requested test targets and two isolated restorations; did not run: whole fast suite, merge gate, standalone season walk or live transports.
