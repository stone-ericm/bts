# W3 review, round 1

## Verdict

**BLOCK** on `f7a8f601` plus `59b72c04`, against `ceab9854`. The save-hook patch restores the original write boundary and preserves exact-byte binding. Three remaining defects lose alerts, and one also loses an enterable pick. Source lines refer to the patched copy.

## Blocking findings

1. **The post-poll decision read can bypass unknown-outcome recovery.** At 18:47 ET, publish the matching attempt marker and scoreable `locked_unconfirmed` record during confirmations, before any alert receipt exists. `src/bts/scheduler.py:3491` breaks before reading the attempt, and `:3154` saves lock flags without alerting. W3 preserves the record and polls game 101 but sends no alert. With the same marker plus an invalid `action: []` record, `:3501` skips recovery and the next legacy reader raises `TypeError`. The base emits the unknown-outcome alert in both cases. Preserve authoritative commits and send the existing receipt-deduplicated alert for `locked_unconfirmed`. Recover a newly observed attempt when the decision read is untrusted. The scratch repairs alert in both cases and retain the valid record bytes. [Patched](durable-patched.log) · [Base](durable-base-final.log) · [Invalid record](attempt-invalid-patched.log) · [Invalid control](attempt-invalid-base.log) · [Repair](repair-alert-patched.log).
2. **The extra status read can expire a gap window before W3 archives the pick.** The 19:05 cutoff, 12-minute cascade budget and 10-minute reserve make 18:43 the latest start. At 18:42:59, game 102's overdue window is still feasible. A two-second status read advances to 18:43:01. `src/bts/scheduler.py:1001` still returns the 18:42:59 plan, and `:3518` checks only the cutoff before `:3529` archives. The later prediction is empty. The base plans at 18:43:01 and delivers the confirmed primary in game 101. With a best-only dated correction, the base emits `contest_state` and `missed_pick`. W3 emits neither and has no pick. Re-run the planner with the final live time after the added reads and before archiving. A scratch change restored delivery or both alerts. [Patched](gap-patched.log) · [Base](gap-base.log) · [Repair](repair-gap-patched.log).
3. **A recoverable error clearing a known-unsent marker skips the late refusal.** A pick reaches the 19:05 cutoff during the attempt write, then the first marker-clear write gets `EIO`. `src/bts/scheduler.py:1443` throws into the transport handler at `:1491` or `:1524`, which clears the marker on retry and returns without calling the refusal. At 19:15, the next check archives the stale preview and returns no pick. W3 sends neither a pick nor `late_delivery`. The base's two-second empty refresh takes the late refusal in both channels. Keep the cleanup retry in the refusal path and finish the existing refusal after it succeeds. The scratch repair produced the refusal archive and alert in both channels. [Patched](clear-error-patched.log) · [Base](clear-error-base.log) · [Repair](repair-clear-error-patched.log).

## Notes

- The archived second-round status-delay script counts a skip notice as a pick DM. Its original assertion fails. Filtering actual pick DMs restores the expected result, including the 19:17 late check with no commit. [Original](replay-r2.log) · [Corrected](replay-r2-status-corrected.log).
- The shared decision parser accepts a schema-tagged scoreable `single` with the date but no primary. W3 preserves it and skips polling for the unmatched slot. The read applies the existing parser's core-field validation. [Decision cases](readers-patched.log).
- A reuse history row can contain `action: defer` on a postponed or late refusal before any defer archive was written. Use the refusal history when interpreting that row. [Lookups](lookup-patched.log).
- The save-hook bypass also made `tests/test_wr_a_review_carbon.py:68` pass in optional mode on `f7a8f601` while correcting the observation after the final read. The patched real writer restores the intended fixture timing and acceptance. Required DM/public and midnight likewise refuse with the restored hook. [Patched](write-seams-patched.log) · [Initial W3](write-seams-head.log) · [Base](write-seams-base.log).

## Required probes

| Row | Replay and outcome |
| --- | --- |
| M1 | Base took a second cascade at 19:00 and refused at 19:06. W3 reused the actual producer result, sent once at 19:00 and committed that batter. `binding-*.log` |
| M2 | Replaced batter, audit-append swap and whitespace-only edits ran legacy refresh. Sent and committed slots agreed. Exact-byte Warmup and probability-mismatched selection used saved rows and policy, and a matching single kept its runner-up. `binding-patched.log`, `policy-metadata-patched.log` |
| M3 | Actual 18:25 to 18:35 sleep refreshed with equivalent output. Equality at 18:35 reused. An overdue check after deferral ran independently with fresh predictions. `binding-patched.log`, `lookup-patched.log` |
| M4 | Earlier genuine MDP skip persisted through Warmup cache and ended as skip. A genuine single cleared it and delivered. `policy-early-patched.log` |
| M5 | Final-state read, pick reload and attempt-write cutoff controls retained `late_delivery`. A transient marker-clear failure suppressed it in both channels as in finding 3. `cutoff-patched.log`, `reload-cutoff-patched.log` |
| M6 | Best-only and saver-only changes refused the saved observation, wrote no attempt or commit and emitted both eligible alerts. `outcomes-patched.log` |
| M7 | Confirmed stale DM and public sends recorded the reading and warned once. Failed send cleared and retried. A missing message ID left the attempt with no stale warning. `outcomes-patched.log` |
| M8 | Direct and would-defer all-void refusals sent `postponed_pick`. The later viable check ran. A one-void-leg double sent and committed both slots. `lookup-patched.log`, `policy-metadata-patched.log`, focused tests |
| M9 | Absence and dangling links yielded no pick, regular links were read through, malformed JSON/shape/time, unreadable files and directories propagated, and the initial FIFO blocked until the process alarm. Extra-reader FIFO, directory and bad data abandoned reuse and left evidence for the next check. `initial-files-patched.log`, `extra-files-patched.log` |
| M10 | Existing attempts recovered with the unknown-outcome alert. Matching and mismatched scoreable records during the poll were kept and gated polling. Simultaneous attempt plus record cases lost the alert as in finding 1. `legacy-patched.log`, `durable-patched.log`, `attempt-invalid-patched.log` |
| M11 | Locked, empty, surfaced skip and contest-error return keys matched the base. A failed atomic candidate write propagated without a reusable return. `policy-early-patched.log` |
| M12 | Both updated consumers passed in the focused run. The incident test supplies fixture bytes. The saved-pick test preserves the real producer and dated mutation. The Kwan slept control and chronological measured-duration test also passed. `focused.log` |
| Check 2, LW3-1 | Both real channels refused a marker write crossing the cutoff and cleared the known-unsent marker in the ordinary case. Finding 3 is the transient-I/O counterexample. `cutoff-patched.log`, `clear-error-patched.log` |
| Check 2, LW3-2 | Ordinary midnight and New Year before the final read, during it and during the marker write refused dated observations with a still-future 00:25 cutoff. Automatic-to-automatic revalidation delivered. `calendar-patched.log` |
| Check 2, decision read | True absence reused. FIFO, directory, unreadable, empty, malformed, bad UTF-8, dangling and wrong-date evidence disabled reuse. A valid regular-target link to a matching record preserved it. `readers-patched.log` |
| Check 2, ordering and new I/O | Initial byte changes took legacy refresh. New attempt in both plan branches and newly delivered pick recovered from durable state. Extra-read failures left later checks runnable. Finding 1 covers simultaneous record/attempt cases. `binding-patched.log`, `durable-patched.log`, `extra-files-patched.log`, focused tests |
| Check 2, shared budget | Slow body, partial and failing lookups, 3-second threshold, all-void result, and late worker completion replayed. Finding 2 shows stale feasibility after the status read. `lookup-patched.log`, `gap-patched.log` |
| Check 2, M9 wording | Initial-reader and extra-reader controls above used separate bounded paths. |
| Keyword ruling | Signature has keyword-only `reused_check_cascade: bool = False`. Only the actual W3 send site passes `True`. Original call sites passed `selection` alone. `legacy-*.log` |
| Post-loop ruling | After abandonment, final fallback still made its own cascade, then the refusal and missed-pick alert ran. Matching authoritative records were polled and mismatched slots were skipped. Findings 1–3 identify other alert losses. `abandoned-patched.log`, `durable-patched.log` |
| Interrupt and restart | Both exceptions before marker, after marker, after durable confirmation, during confirmation poll and during status poll. Before marker there was no send or commit. A fresh invocation used only durable state and did not recover the key even with identical bytes and reuse history. After marker it issued the unknown-outcome alert without resending. After confirmation it recovered delivery. `interrupt-patched.log` |
| Failed or unconfirmed send | Known failure cleared and retried. Missing DM ID left the marker and no confirmed-delivery warning. `outcomes-patched.log` |

## 1. Binding, reuse lifetime and refusal behavior

Tracked actual serialization, byte comparison across both reloads, journal, transport and committed rows with `probe_binding.py`, `probe_policy.py`, `probe_delivery.py` and `probe_durable.py`. The patched writer returned the path and captured the first successful exact serialization before the audit append. Nested and separate-thread saves were isolated, and write failures and interrupts reset the observer. No decision-only result, byte replacement, sleep or restart reused a key. The original A3 key-only and read-back counterexamples reproduced on the archived base. [Binding](binding-patched.log) · [Capture](capture-patched.log) · [Archived core](replay-r1-core.log) · [Archived edges](replay-r1-edges.log).

## 2. Existing paths and no-alert days

Paired `probe_legacy.py` output was byte-identical on all original delivery callers and the existing attempted-pick recovery. Compared transport bodies, normalized journals and file digests for picks, decisions and streak. The real result poll scored one synthetic hit. Ordinary slept and final fallback, late/void refusals, caught failure and the 4c warning behaved as expected in the listed controls. Finding 2's corrected-observation day and finding 3 are paired no-pick/no-alert days. [Base](legacy-base.log) · [Patched](legacy-patched.log) · [Abandoned](abandoned-patched.log).

## 3. Lookups and planner

The confirmation took about 1.2 s and left about 0.8 s for the slow status body. Total wait was about 2.01 s, and the 18:50 check committed game 102. The real parser retained an existing home confirmation, added a timely away confirmation and ignored a late daemon completion after joining it. Failed bodies left prior sides intact. The real planner saw both lock booleans, all contenders, the side set, non-coalesced remaining runs, measured budget and reserve. The second-ranked contender's feasible window deferred when the top contender's window was too late. Finding 2 covers the clock sample after status I/O. [Lookups](lookup-patched.log) · [Planner](policy-metadata-patched.log) · [Coalescing](coalescing-patched.log).

## 4. Operator text

| Sentence | Worst literal reading |
| --- | --- |
| `ET date changed before contest state could be validated` | Two date changes exhaust the reread, with no transport or attempt. A later ordinary fallback can still run. The sentence gives no command. `calendar-patched.log` |
| `ET date changed after the final contest-state read` | Midnight after validation causes a refusal and clears this call's known-unsent marker. The sentence gives no command. `calendar-patched.log` |
| `FALLBACK: reusing this check's fresh cascade decision (no second refresh).` | Prints before the extra checks. An operator could infer one cascade even when abandonment starts the final fallback cascade. Log successful reuse after the check or describe it as pending. `abandoned-patched.log` |
| `FALLBACK — delivering (…) before the … cutoff.` | Announces an intended attempt. Final-state and marker guards can still refuse, so the operator needs the lock or refusal line to know the outcome. `cutoff-patched.log` |
| `Pick DM failed: …` / `Bluesky post failed: …` | In finding 3, the wording suggests a transport failure although transport never started. The late refusal line and alert are absent. `clear-error-patched.log` |

## 5. Focused tests and scratch mutations

The eleven requested files ran in one pytest invocation against the patched sources, with **331 passed and 4 skipped**. The optional-LightGBM test's own collection subprocess and the web saver's synthetic loopback test passed. [Focused output](focused.log) · [Wrapper](focused.log.watch).

| Scratch mutation | Targeted result |
| --- | --- |
| Drop both byte comparisons | Detected, 3 failed |
| Permit reuse after fallback sleep | Detected, 1 failed |
| Drop cutoff check after saving the attempt | Detected, 2 failed |
| Drop ET-day checks | Detected, 4 failed |
| Ignore the scoreable record | Detected, 2 failed |
| Merge late worker confirmations | Survived, 1 passed |
| Pass selection without slot matching | Detected, 1 failed |
| Coalesce after reused deferral | Detected, 1 failed |
| Skip the final decision read | Detected, 1 failed |
| Give statuses a fresh two-second budget | Survived, 2 passed |
| Poll a pick with mismatching committed slots | Detected, 1 failed |
| Skip the final pick read | Detected, 3 failed |
| Count confirmations by game | Detected, 1 failed |
| Capture candidate bytes by reading back after the audit | Detected, 1 failed |

The surviving mutants expose two test gaps. The late-result test releases the worker without checking the confirmation set after the worker returns. The budget cases lack a delayed confirmation followed by a slow status body and a total elapsed-time assertion. The independent probes detected both mutations by joining the late worker and by measuring about 3.2 s for the split budget. [Mutation output](mutations.log) · [Late worker](late-worker-mutation-patched.log) · [Split budget](separate-budget-mutation-patched.log).

## What I ran

- Replayed both archived contract checks and the original A3/slice example against `ceab9854` sources, using short bounded offline processes. The original status-delay assertion and the corrected pick-only replay are retained.
- Independent real-scheduler probes against the patched source and paired base cases, plus scratch verification of the three repairs. I applied the supplied patch to an archive of `f7a8f601` and verified both resulting blob hashes. Fixtures used pinned ET clocks, disposable files and captured transports. Direct `nice -n 19` reported a priority warning and the probe commands ran.
- At 09:45:40 ET, the shared wrapper started the eleven-file run. After that wrapper exited, it started the 14 targeted scratch runs at 09:47:18 ET. Both wrappers exited zero. The test runner guarded against connections outside its synthetic loopback fixture. [Focused runner](run_focused.py) · [Mutation runner](run_mutants.py).

## What I did not run

- The author's separate full fast-suite gate and all live service, account and production jobs.
