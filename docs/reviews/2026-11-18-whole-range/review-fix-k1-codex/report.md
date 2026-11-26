# K1 round 1 review

## Verdict

**BLOCK** under R64. The new live-repair pointer sends a late-start operator to a filing deadline they have already missed. A safe same-slate recovery can still produce the day's pick.

## Blocking findings

- **B1. The B8a timing instruction can cost a recoverable pick.** At 18:35 ET on July 15, the scheduler starts after its 18:25 planned lineup check. The switch is absent, there is no dated observation or saved pick, and prior entries have settled. The scheduler skips the morning DM, runs the overdue check, and sends the refusal. The new instruction at `src/bts/scheduler.py:77` points to B8a at `docs/ops/2027-season-start.md:31`. Its daily dated-filing step says to enter the observation before the first lineup check. Its explanation of late filing starts with an already-chosen pick, which this refusal never produced. The separate stale-warning instructions concern a different alert. Applying the daily filing time leaves the observation absent through the 18:55 check.
- **Demonstration and smallest fix.** With two scheduled checks and first pitch at 19:10 ET, the real scheduler sent zero official picks when the observation stayed absent. With the same schedule, filing MLB's verified 57 active, 60 best and saver used through the real CLI between checks produced an official pick at the second check and a delivered decision record. Amend the B8a timing sentence linked from the DM to read “File before the first lineup check when possible. If that check refused before choosing a pick, confirm this is still today's slate in ET and that MLB has settled every prior entry or confirmed its absence after its window closed. File the verified dated observation between checks while another lineup check or fallback remains.” See `replay_operator.py` and `operator-replay.log`.

## Edits

- **Correct Amendment 1 ruling 3(i)'s base comparison** in `kickoff-fix-k1.md:73`. Replace “(base: both sent, because the wall date had moved to D+1)” with “On the base, the first same-source in-day refusal after midnight sends when its only coverage was before midnight. Further in-day refusals are suppressed while the D+1 receipt covers them. D's end-of-day digest uses D, so a D+1 receipt cannot cover it.” Both builds sent once for two in-day refusals starting after midnight, for all four sources at both boundaries. `dedup-base.log`, `dedup-replay-k1.log`.

## Notes

- The shared receipt rule also applies when a historical D digest comes first. In the replay, a later overnight D lineup refusal had a different cause from the digest and the same source key. The base delivered it and K1 suppressed it. Ruling 3's general same-slate key rule covers this order. The examples in ruling 3 describe the opposite order.
- A fixture used a stale-slate case where another lineup check was still scheduled. The replay clustered those games for the in-loop fallback case and exercised both fallback sites. The operator rehearsal's replay identified the refusal by its opening sentence because the separate morning DM also uses `contest_state`. Initial harness logs are retained beside the successful replays.
- The first dedup harness hit the default heartbeat wrapper before its fixture failed. That touched the ignored `data/.heartbeat` and `data/health_state/cascade_stage_durations.jsonl` in this checkout. The replay substitutes that wrapper. The existing ignored files were left in place.

## 1. Changed scope and base comparison

- Read `git show d4017e6e` and diffed against `22a27d29`. The production changes are the four date arguments, the contest-state formatter and the failure-count display. The existing diffs are the source search and five fixture signatures Amendment 1 lists.
- Ran `replay_callers.py` against the checkout and `git archive 22a27d29 src`. Logs print each import's `bts.__file__`. Compared 36 captured runs across same-day, June midnight and New Year with `callers-compare.log`. Scheduler stdout, full captured stderr including all five refusal journals, returned values, and pick, lock, archive, scheduler and decision files matched after replacing the temporary root in each capture. Each branch's transport followed its journal line.
- Both missed-pick variants, late delivery and void delivery had identical wire bodies. Receipt fields and keys matched. The only observed receipt value differences were the expected sent day and refusal-body hash. The large corrupted count intentionally changes the shared failure banner on any source.

## 2. Refusal length

- Ran `probe_length.py` on both imports. It uses the real loaders, CLI and saved-pick checks for the E3 inventory, with the box path substituted where the triggering error contains the temporary path. The reachable raise sites matched E3. Lexical/parser variants exercised the same diagnostic branches, including a JSON file holding a 16,000-code-point string as the streak. Dated check 1 is intercepted by the reader, the required-missing raise is shadowed by the authority refusal, and both model-only raises were exercised with the requirement off.
- Exercised no receipt, failed counts 1, 999, 1,000 and a 701-digit count, invalid receipt bytes, huge emoji and multi-line reasons, empty and whitespace-only first lines, lone surrogates, a raising string conversion and a 1,800-code-point type name. Invalid date cases included a 100,000-character string, impossible calendar date, NUL, lone surrogate, integer and missing date. Standing checks also exercised a real thrown send followed by retry.
- Absent and revoked sent at 536 and 537 code points, or 620 and 621 after a one-count failure. Base was 1,146 and 1,147, or 1,230 and 1,231, and the captured length-limited transport refused them. K1 peaked at 971 code points with the banner and 936 grapheme clusters across the captures (ICU). Every K1 capture was valid UTF-8 and under 1,000. `length-k1.log`, `length-base.log`.

## 3. Slate dates and deduplication

- Both ET boundaries reached the lineup check, the live-load and stale-saved-pick delivery guards, both actual fallback refreshes, the all-void guard, the initial cutoff guard, the DM and public pre-send cutoff rechecks, and the missed-pick wait starting before midnight. The final fallback hit at 00:40 ET, the in-loop fallback at 00:28, and each alert recorded the old slate. `callers-replay-k1.log`, `callers-base.log`.
- For all four sources on both boundaries, old D then new D+1 sent 1/1 under K1 versus 1/0 on base. D before midnight followed by two D calls after midnight sent 1/0/0 versus 1/1/0. New/old/new sent 1/1/1 versus 1/0/0. Two old calls both after midnight sent 1/0 on both builds. All four helpers used wall ET for a blank, bad or impossible date at dispatch.
- Real historical D received a malformed automatic observation diagnosis, while the old-D lineup refusal came from a malformed manual observation. With historical first, base sent three messages across digest, lineup and digest rerun, and K1 sent one. With lineup first, base sent two and K1 sent one. The matched E8 tests also exercised a late-delivery digest and the new-key sends-then-dedups case. The shared one-slate source rule explains the observed suppressions and repeats. `dedup-base.log`, `dedup-replay-k1.log`, `tests-k1.log`.

## 4. Operator text (R64)

- Late start and missed morning check reproduced B1. After a mid-day revoke, the reason and read-only status both showed the refusal. After a real dated correction from 12 to 13, status accepted current state, the DM and journal gave the saved-pick reason, and the cached pick stayed unmodified and unsent.
- With one final check, a later successful dated write changed status to accepted and produced zero picks. The DM's final sentence is conditional on a later official pick and its open window.
- The real overnight final fallback rendered D. Its journal and the read-only status command explained the current refusal, and the status template used D+1. The real CLI rejected a dated write for D and accepted D+1 in the test control. With prior entries pending, B8a's later-slate verification and the DM's new-slate reminder call for waiting. B8a requires verified absence after the entry window closes. This held across both ET boundaries. `operator-replay.log`.

## 5. Existing tests and scratch changes

- The brief's recursive expression matched 39 files. The dedicated file had 97 passes. The other four bounded groups had 1,683 passes and 12 skips, including the tests that invoke the archived fixtures. I also ran an extra health-directory group (773 passes) before running the exact list. See `tests-*.log`.
- Scratch changes caught by the new test file, in order: full reason (23 failures), no 400-point cut (2), no invalid-date label fallback (1), unbounded failure count (1), and refusal date parsing without the `try` (2).
- Passing the wall date at `scheduler.py:1362` by supplying `slate_date=None` survived all 39 files (1,780 passed, 12 skipped). The K1 test checks date forwarding only on the final fallback. The overnight saved-pick correction in `replay_callers.py` requires D from dispatch and the real receipt. Rerunning it against the scratch source failed at date forwarding. `mutation-saved-wall-date-all.log`, `mutation-saved-replay.log`.

## 6. Standing rows

- Reopened the shared receipt after an interrupt before transport, after transport but before persistence, and after truncation plus a partial write. Both builds propagated the interrupt. A successful transport with missing or torn persistence allowed one repeat on retry. Subsequent retries deduped once the receipt was complete.
- Invalid JSON, invalid UTF-8, an array and a wrong source type left the alert eligible. A FIFO and a directory allowed sends without a receipt or a hang. A throw recorded `failed` and retried with the banner. No recipient recorded `skipped_no_recipient` and left the alert eligible. A non-raising transport return without an ID recorded `sent` and suppressed retries, as on the base. Midnight and New Year cases are above. `standing-base.log`, `standing-k1.log`.

## 7. Other diff checks

- The one-slot overwrite, coverage by historical digest, source-only suppression, and legacy unconfirmed-send behavior follow the recorded design or base. The shared health log's “today's DM” text is preserved and includes the dedup date, which may be the prior slate. Morning, stale-reading and the separate unknown-outcome paths were checked by the diff and matched tests.
- The checkout was at `01896138` and the tracked worktree remained clean. Scratch source lived under `/tmp`. Commands used ET, serial bounded runs and requested priority 19. The host printed a priority warning and the commands continued. The permitted Victoria 3 process check found no match.

ran: diff, both imports, wire lengths and ICU grapheme counts, caller and file comparisons, rollover and historical runner, operator and standing rehearsals, 39 matched test files, extra health group, and six single scratch changes; did not run: live MLB or Bluesky, the full fast suite, or a gate.
