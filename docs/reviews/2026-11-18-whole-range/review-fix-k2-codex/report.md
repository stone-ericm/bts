# K2 review, round 1

## Verdict

**BLOCK**

## Blocking findings

- **B1. A blocked journal delays lineup work outside the warning deadline.** `src/bts/scheduler.py:3021,3047,3053,3061` writes warning diagnostics to stderr on the scheduler thread. Their exceptions are caught, but a write that waits for space holds the scheduler. The daemon's timed join only bounds the worker wait.
  - In `probe_journal_pipe.py`, the 2027-09-27 private startup is at 10:05 ET for a lone 13:15 game. The stderr pipe has 350 bytes free and its consumer is stalled. The base reaches its 10:15 first check. K2 hits real pipe backpressure writing the boundary observation, reaches the first check at 10:17 after the consumer resumes on the pinned clock, and never sends the WARN.
  - `probe_remaining.py` also restarts at 18:59:59 ET for a 19:05 game. A two-second journal stall on the pinned clock makes K2 pass the lone game as past the 19:00 entry cutoff. The base reaches the same check immediately with that game still available. This case takes the journal-only branch.
  - Put boundary diagnostics, including timeout and failure lines, behind a nonblocking logging path that cannot occupy the scheduler's own stderr writer. Give lineup work priority when the sink is full.

## Edits

- **E1. Honor an acknowledged warning across a boundary update.** In `src/bts/scheduler.py:3002–3008,3034`, the body changes when the configured boundary or its source changes. The acknowledgement reader matches the body hash at `src/bts/health/alert.py:342`. In `probe_receipts.py`, adding 2028 to the table with the same September 28 fallback value sends a second acknowledged warning on September 29. Match K2's valid acknowledgement by canonical slate date and incident even when its explanatory body changes. Scope the change to this receipt.
- **E2. Describe a past check accurately.** At `src/bts/scheduler.py:3047`, a restart at 00:01 ET for the previous slate says the first check is “due within 10 minutes” even when it was due the previous afternoon. For zero or negative lead use “The first planned lineup check is due now or has already passed.”
- **E3. Describe what the warning counted.** `src/bts/scheduler.py:3007,3021` calls a same-date `Final` game eligible. It qualifies for the boundary observation and is unavailable for fresh selection. Use “At scheduler startup MLB listed at least one regular-season game with that official date and a reported status other than postponed or canceled.” In the journal, use “{n} regular-season game(s) with that official date and a reported status other than postponed or canceled.”
- **E4. State the scope of the floor.** Replace the two day-count descriptions in `CLAUDE.md:64` with “Fresh selection uses at least one policy day once it has a primary candidate, even when more than one makeup day remains. Standalone policy calls and health checks use the raw day count from the configured or fallback exclusive boundary.”

## Notes

- The text edits are nonblocking under R64. The captured warning on an already-posted slate asks the operator to retain the pick, consult the schedule and contact the owner. Following those instructions did not expose a wrong pick or state write.
- The candidate type/date trust limit, a daemon send surviving its wait, and an unavailable or unconfirmed receipt allowing a retry are covered by the amendments.
- The walk's raw-day oracle detects the wrong stored boundary. Its makeup action oracle can accept a single at streak 55 after changing the loaded one-day reach-57 table. K2's table tests and the real-artifact selection probe caught the expected action separately.
- The initial selection probe incorrectly expected the longest-streak path to bypass a throwing loader. `probe_selection_followup.py` checked the actual contract behavior of a degraded single and exercised the full set of remaining entry points.

## 1. Diff, positive days and raw callers

- Reviewed `712351a5..26c621c7` and verified the imported `bts.__file__` from the archived `712351a5` source and from this checkout. The base and K2 logs identify their respective paths.
- `probe_grid.py` compared 373,188 decisions over Jan 1–Sep 26, 2027, including every regular-season date, streaks 0–60 and 120, both saver values, quality edges and adjacent values, and trusted and untrusted tail bests. Every serialized decision and display bar matched. The common digest was `66e80cec3da274c88d0a4ba36a306f107c3db3a0fd3d8de62d689ac343d97bf7`.
- The bare context, `decide_action`, standalone bar and eight health snapshots matched byte for byte across raw-zero, in-season and year-boundary dates. A real `run_single_check` with all predicted games unavailable used the bare bar and returned the same no-decision summary.

## 2. Raw-zero selection and walk

- `probe_selection.py` matched the real one-day action table at p=.80 on Sep 27, 2027 and Sep 28–29, 2028, with both saver inputs. With a beatable best and a different-game partner, streaks 0–55 doubled, 56 singled, and 57–60 and 120 doubled. A trusted best of 25 stopped streak 20 at a bound of 22. A stale-low best of 18 still played. Either double clamp forced a single. The shipped base skipped the corresponding raw-zero rows.
- `probe_selection_followup.py` used real `orchestrator.run_and_pick`, `bts run`, `bts preview` and the shadow path with synthetic prediction inputs and captured transport. `bts run` posted locally to the capture, preview and shadow wrote separate local picks, and saved decisions used effective day 1. Shadow uses its existing state inputs and does not see the dated best of 25.
- Real `walk.walk` produced double, skip, double, single, double, double for the six makeup examples. The empty fixture had no decision. Changing the stored boundary to Sep 26 gave raw 0 instead of 1 and still triggered both table and raw oracles. The base also flagged the finale skip. A wrong in-memory one-day reach-57 action passed the walk alone. The existing saved-decision test retained the v3 schema.

## 3. Warning placement, timing, rows, receipts and text

- `probe_startup.py` and `probe_remaining.py` captured real morning refusal and stale-reading DMs byte for byte against the base. The morning DM went first in DM and public modes. An 11-second morning send from 15:54:55 moved the boundary check inside the lead limit while preserving the 16:05 lineup check.
- A lone 13:15 opener used the 10:15 early check. Exactly ten minutes qualified, 9m59s and one second before that check were journal-only, and a restart after it used the next check at 11:15. A blocked daemon worker exhausted the simulated 60-second wait, remained alive at the first check, and wrote an acknowledgment afterward. The next restart suppressed the send. B1 covers stderr backpressure separately.
- Captured WARN in private, DM and public modes. No recipient, no plan and late restart logged their reasons. Positive day, off day, dry run and a zero qualifying count produced no advisory. An injected journal write error left the DM eligible.
- `probe_receipts.py` counted a good R row alongside future/earlier dates, missing type/date, void states and malformed status values. Stable-body starts deduped through a count change and D1 → D2 → D1. E1 covers the different-body case.
- `probe_operator.py` captured the fallback-year, late, final-game and already-posted messages. The posted pick file stayed byte-identical. E2–E4 give the text adjustments.

## 4. Tests and one-change scratch copies

- All 86 search-selected pytest files, including all 59 K2 cases, and the additional tail-policy health file were run. There were 2,826 passes, 17 skips and one expected failure. The audit driver hit 33 failures and 16 setup errors on both the base and K2 because GNU rsync was unavailable. The 22 search-selected script and documentation paths were inventoried, with Python and shell syntax checked. The season walk was exercised as code.
- Only the Amendment 2 walk expectations/fixture, H2 expected result, three stub signatures and H spy changed among existing tests. The direct-call, guard and H fingerprint assertions were retained and run.
- Detected mutations were the unconditional `_regime` floor, missing context floor, warning before morning check, missing `officialDate` predicate, synchronous send and plain dispatcher. Each failed its first targeted assertion.
- Removing the early floor alone survived all runnable search-selected tests. A raw-zero streak-57 probe with two games and a throwing MDP loader catches it. Shipped K2 contains the exception and returns a degraded single. The mutant lets the same exception escape. Extend the existing longest-streak loader regression to raw zero.

## 5. Standing rows

- An operator interrupt during the join escaped and left no receipt. Restart sent and acknowledged once. Failed, blank and missing IDs wrote failed status and remained eligible. A simulated receipt write failure also retried.
- Invalid JSON and schema-less status were replaced. FIFO, directory, symlink, hardlink and a symlinked receipt directory remained intact and could not suppress retries. The aliased general-health receipt was byte-identical.
- After midnight for D, the journal and receipt identity use D and the old planned check is past. Empty Dec 31 and Jan 1 idled. An intentionally injected same-date R game on Dec 31 qualified, and fallback-year dates used September 28. A malformed slate date failed only the K2 advisory parse, with the real scheduler still reaching its check. An unreadable schedule raised on both revisions.

## 6. Other checks

- `git diff --check 712351a5 26c621c7` was clean. The checkout's tracked files and git state were untouched. The tests ran serially with per-command timeouts and pinned ET probe clocks. The OS denied lowering priority despite `nice -n 19`, and the sandbox could not inspect the `victoria3` process name.

ran: base/K2 grid, direct callers and health, public and shadow selection, sampled season walk, morning/startup/transport, physical journal backpressure, receipt and R64 probes, search-selected pytest and seven scratch changes; did not run: full fast suite, full-season walk CLI, GNU-rsync-dependent test bodies blocked at import, or deployment/data-build script entry points
