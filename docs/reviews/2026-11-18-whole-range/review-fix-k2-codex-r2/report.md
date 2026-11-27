# K2 review, round 2

## Verdict

**BLOCK**

## Round-1 findings

- **B1 — the scheduler-thread warning write is fixed.** Both original probes reached the first-check print on time, then failed their old assertion that the warning writes through their patched `sys.stderr`. The stale/fresh morning cases in `probe_remaining.py` passed. The new pipe probe stalled the warning writer and reached 10:15 ET with an enterable game for both eligible and inside-lead starts. It reached the 18:59:59 restart check with the 19:05 game enterable. No recipient and an empty plan each produced one journal-only line.
- With a failed send and a writable scheduler stream, the first check still ran at 10:15. The failed receipt allowed one acknowledged retry. With transport held past the simulated 60-second deadline, the observation and deadline line appeared in order, the first check ran at 10:15 with the game enterable, and the late acknowledgment was recorded. Expiry before dispatch produced the insufficient-budget line and zero sends. See `startup.log`.
- **The check is still vulnerable to the shared-pipe limit in Amendment 3.** With 510 bytes free at startup and the warning writer going first, the journal-only warning left the ordinary `Running lineup check` write blocked at `src/bts/scheduler.py:3288`. A two-second stall moved the 18:59:59 first check to 19:00:01 with the lone game past cutoff. The control with the warning disabled used the same pipe and kept the game enterable. `shared-journal.log` confirms the stated limit and its cutoff consequence.
- **E1 — deliberately unbuilt.** On 2028-09-29, changing the fallback boundary to an explicitly configured boundary with the same date changed the body and sent one extra acknowledged WARN. A further restart with the same body was deduplicated.
- **E2 — fixed.** Zero lead and past plans use the prescribed “The first planned lineup check is due now or has already passed.”
- **E3 — fixed.** The DM and journal use the exact reported-status wording. A same-date `Final` game counted, and a same-date `Postponed` game did not. The advisory asks the operator to verify the finale for a later run and preserve a sent or entered pick. No R64 text blocker found.
- **E4 — fixed.** Both prescribed sentences appear verbatim at `CLAUDE.md:64`. Fresh selection applies the floor once a primary candidate exists. Standalone calls use the raw count by default, and the health snapshot tests confirm the raw count there.
- **Early floor test gap — fixed.** The two loader cases pass in the checkout. A source copy with only the early `min_days=1` removed at `src/bts/strategy.py:645` passes Sep 26 and fails Sep 27 with the escaping `TypeError: broken loader`. See `floor-intact.log` and `floor-mutant.log`.

## Introduced defects

- **B2 — a failed boundary send can prevent an interrupted scheduler from exiting.** On 2027-09-27, start privately at 10:05 ET with one 13:15 game and a DM recipient. With a full stderr pipe, a transport exception makes the dispatch thread take the buffered stderr lock at `src/bts/health/alert.py:238–243,378–380`, through `src/bts/scheduler.py:3075`. `probe_shutdown.py` sent SIGINT. `KeyboardInterrupt` escaped `run_day`, but the Python process stayed alive past the 1.6-second wait and exited only after the reader drained the pipe. The blocked-send control exited before any drain. The base has no boundary dispatch on this path. Amendment 4 describes a delayed scheduler line. Interpreter shutdown also waits on that lock after `run_day` unwinds.
- **Smallest fix for B2.** Give acknowledgment dispatch an optional failure reporter and pass the boundary queue from K2. Record the same failure line through the daemon using `os.write`. Other callers can use the existing reporter by default. Keep the real-full-pipe SIGINT probe as the regression check.
- **Journal loss on a nonblocking sink.** At the journal-only 18:59:59 restart, a full nonblocking pipe returned `EAGAIN`. `src/bts/scheduler.py:2978–2980` discarded the sole observation, even after the consumer drained the pipe. The first check and game were available. The writer can retain the pending line and retry transient failures off the scheduler thread. `worker-lifetime.log` models this descriptor mode. The deployed journal descriptor was outside the probe.
- The queue preserves its own line order. The shared failure line can precede the observation because it bypasses the queue, as `startup.log` shows. Two timed-out starts in one process left two dispatchers and two writers alive even with a writable journal. Each writer waits for its dispatcher's sentinel at `src/bts/scheduler.py:2995–2997,3084–3085`. All four stopped after release. This can retain a writer per permanently blocked dispatch in a reused process. The service launches one `run_day` per process. Ordinary tested failures stayed contained.
- **Positive-day decisions match the base.** The round-2 strategy diff is empty. The K2 test file matched the archived base over 373,188 positive-day decisions and display bars. The day floor leaves every positive raw day count at its original value.

## Edits

- Qualify Amendment 3's E1 limit: “A restart can send another WARN if the date still qualifies and the body differs from the last acknowledged body. A boundary, boundary source, or wording change can change that body.” Moving the boundary beyond the slate date makes the slate ineligible. The receipt compares the full body hash at `src/bts/health/alert.py:342`.

## Notes

- The 64 K2 tests passed. The two original scripts stopped at the obsolete pipe interception assertions. The replacement probes cover the affected timing branches and use real pipes with a pinned ET clock. The send deadline in the startup probes used a simulated timed join. The shutdown probe used a real SIGINT and subprocess wait.
- The tracked checkout was clean at `ec5966e1`. Probes and logs are in `.codex-review/wr-k2-r2/`. The OS declined the requested priority change on probe runs.

ran: both round-1 probes, four round-2 probe scripts, K2 test file, both loader cases on the checkout and scratch copy, strategy diff and diff check; did not run: the full fast suite, the full H test file, the unrelated trailing helpers in the original remaining probe, or live delivery and deployment
