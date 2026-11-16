# Verdict

**BLOCK** — Row L's totality promise fails for hook-bearing Python objects. Ordinary strings, season isolation, corruption handling, rollover, saver behavior, and scorer serialization passed the comparisons below. The blocking inputs are unusual library inputs, not values obtained by decoding ordinary CLI text or JSON.

# Blocking findings

**B1 — `season_of_date` can introduce read failures and fail to reject a writer with `ValueError`.** `src/bts/picks.py:522` runs `isinstance` outside its handlers; line 528 can execute string-subclass hooks, and line 529 catches only `Exception`.

- Concrete input 1: `NotString` in `counterexamples.py` has `__str__` returning `"junk"` and a `__class__` property raising `RuntimeError`. It is not a string and must yield `None`. With real `junk.json` containing an unresolved delivered pick and `streak.json` containing `{"season":2026,"streak":13,"saver_available":false}`, the base's real `run_result_polling` returns `final`, persists `void`, and leaves the streak bytes alone. HEAD raises `RuntimeError`, leaves the pick unresolved, and preserves the streak. The same new failure occurs in both peer display branches. Shadow selection silently skips its selection after catching this error; the base passes streak 13 to selection. `select_pick` changes an existing downstream `TypeError` into this new read-time `RuntimeError`. Dated `update_streak` propagates `RuntimeError` rather than the required `ValueError`.
- Concrete input 2: `StringHook("2027-3-24")`, a `str` subclass whose `__len__` raises `SystemExit`. The base's three polling read paths return `final`; HEAD propagates `SystemExit` before reading the streak. Shadow also propagates it. The base's dated update returns 14; HEAD propagates `SystemExit`, outside either permitted dated-write outcome. CLI's unchanged age guard already rejects this spelling, and strategy already raises this hook's `SystemExit` downstream; those existing errors are not new findings.
- Smallest fix for these exception regressions: check actual type ancestry without invoking the instance's `__class__` property, and contain parser-hook failures, including `BaseException`, returning `None` when parsing fails. Preserve the original marker D. The stronger no-stall promise needs the separate rule decision R1 below; broadening an exception handler cannot fix a blocking callback.
- Demonstration: `counterexamples.py base/current`, `counterexamples-*.json`, and the `class-trap` / `str-subclass-exit` rows of `probe-*.jsonl` and `matrix.txt`. These are real production calls with real pick/streak files; only clocks, prediction, MLB responses and posting are replaced. Exact before/after bytes are retained.

# Notes

**R1 — Contract concern, separate from the code finding:** “exactly these parsers on the supplied str” and an unconditional “never hangs” are incompatible with unrestricted string subclasses. A subclass can block inside `__len__`, which `strptime` invokes. `extras2.py` demonstrates a 60-second callback, terminating only its own child after a one-second timeout. Specify whether to parse native string contents (`str.__str__(D)`, retaining original D as marker), or narrow the guarantee. Native-content parsing also changes how subclasses with deliberately failing hooks are classified; that decision belongs in the contract. This concern does not excuse B1's ordinary `RuntimeError` escaping for a non-string input.

**N1 — Existing limits, not new blockers:** export/regenerate can drop the season tag; restoration then reads fresh until replay. A saver-zone write with explicit saver can replace corruption introduced after its successful reads, as at the base. The reconcile summary samples the receipt clock again, so it can display a different season if replay crosses ET New Year. The specified mid-season deployment gap remains. Row I forbids changing the export/regenerate files here.

**N2 — Evidence limits:** the first frozen-clock run broke heartbeat timestamp advancement. The first ticking version consumed a test's patched monotonic calls and broke its cutoff test. `tick2_family.py` saves the original monotonic function; both failures disappear. Python datetime/date clocks are replaced after pandas loads; OS time, mtimes and C-extension clocks are not simulated. Initial `extras.py` race fixtures lacked delivery evidence and did not exercise scoring; its correction fixture was past cutoff. Only corrected `extras2.py` results count. Logs and scripts for the discarded runs remain. Wrapper success alone is not a test verdict; child logs/JUnit were inspected.

# Row L against the base

Independent matrix: **43 values × 8 entry paths = 344 comparisons** per tree. Paths: real shadow, polling void/final-peer/early-peer, Click `CliRunner`, typed CLI callback, real `select_pick`, and real `update_streak`. Base source came from `git archive d8eb50d0 src`, never from a tracked-tree checkout. `matrix.txt` summarizes every cell; JSONL records contain outcomes, reader/policy/reply values, and pick/streak bytes before and after.

| Input family | Observed base → HEAD |
|---|---|
| Padded, three unpadded forms, compact, week/compact-week/week-without-day, week-year boundary, years 1/9999, Arabic/fullwidth year digits, plain str subclasses | Reads 13/False → season-fresh 0/True where reached; dated hit 14 → 1, correctly tagged; exact marker retained. `2026-W53-7` resolves to calendar year 2027. |
| Ordinal, whitespace/tab/newline, time/zone suffixes, slash/dot, impossible day/month, years 0/10000, rejected Unicode combinations, junk/empty | Unknown season: reached reads preserve base behavior; dated update raises `ValueError`, unchanged bytes and no state reads. Upstream CLI/strategy restrictions remain. |
| `None`, date, datetime, bytes, int, bool | Helper returns None. Original-value callback restrictions match base. Click converts a date object and compact integer to readable text, so those actual CLI reads legitimately become scoped. `None` update stays dateless and preserves the tag. |
| Two strings over 10,000 characters | Helper returns None promptly. Read paths obstructed by filesystem name limits fail/skip at the same upstream point as base; direct dated update rejects before I/O. |
| Hook-bearing non-string / string subclass | Unexplained exceptions, skipped shadow selection, and changed strategy failure described in B1. A blocking string hook stalls the helper (R1). |

For the 328 cells excluding the two hook inputs: `verify_matrix.py` verifies 122 scoped cells, 205 legacy/read-or-L4-rejection cells, and one dateless tag-preservation cell. No other behavioral difference was found. It compares pick bytes exactly and normalizes scratch-directory names in diagnostics; long diagnostics were captured with a length limit. The generic class-trap fixture's default repr contains a process address; the small counterexample uses stable `junk` paths to remove that artifact.

Every unexplained behavioral difference: non-string hook causes three polling failures, a skipped shadow selection, strategy `TypeError` → `RuntimeError`, and wrong L4 exception; string hook causes three polling failures, shadow `SystemExit`, and L4 `SystemExit`. Strategy's same exception moves before its policy call because the new saver read fails first. No new CLI exception was observed. No tested helper return was non-int/non-None; only the deliberately blocking hook stalled, and only the two hook inputs raised.

# Changed tests

All **848 existing assert statements are structurally identical**, including multiplicity (`assertion-audit.json`). File-by-file review of the whole diff:

- `test_check_results_wait.py`: 11 season-tagged fixtures; pending/error, stale guard, waits, deadlines and retry assertions retained (49 asserts).
- `test_cli_integration.py`: 19 fixture tags; scoring, void, shadow, delivery and contest authority assertions retained (227).
- `test_contest_state.py`: season tags and explicit 2026 decision clocks; model/contest comparison, status and saver assertions retained (40).
- `test_incident_register_2026.py`: exactly 12 `season=2026` fixture arguments; the entire AST otherwise matches base. All 46 asserts, HTTP witnesses, dedicated exception oracles, hit/pass/void, same-day replay and crash/restart checks remain. None targeted absence of a season tag.
- `test_pickpath_review_pp1.py`: one tag; crash/reconcile/restart remains exercised (10).
- `test_pickpath_review_pp2.py`: one tag; delivered hit+void double still increments once (39).
- `test_picks.py`: five reconcile fixture tags; replay/correction/future-preview/partial-void/shadow assertions retained (146).
- `test_reconcile_cutoff.py`: shared fixture gets its date's year; cutoff refusals and timely corrections retained (15).
- `test_reconcile_receipt.py`: fixture tag only; receipt and replay checks retained (78).
- `test_scheduler.py`: two tags; void-plus-hit double assertions retained (166).
- `test_scoring_lock.py`: four tags; peer, vanished file, fresh metadata and replay-under-lock checks retained (18).
- `test_streak_applied_dates.py`: current-season fixtures tagged; bounded-marker dates made valid with timedelta. “legacy file without marker” becomes “tagged file without marker”: its missing-marker assertion remains, while untagged carryover is intentionally replaced by new tagless-reset tests. All 14 asserts retained. No assertion was loosened or made vacuous.
- New `test_streak_season_cs2.py` / `_review.py`: 239 / 16 cases. They cover the normal contract and preservation boundaries, but omit string subclasses and hook-bearing objects; they miss B1/R1. The CLI matrix uses the direct callback for typed values; my matrix additionally runs those through Click.

# False greens

Against actual base src: **92 passed, 163 failed** across the two new files (89/3 passes respectively). Full passing node list: `base-green-tests.json`. Passes comprise 24 corrupt-state preservation cases, 48 unknown-season scheduler/shadow legacy-read cases, 15 CLI upstream-rejection cases, and five dateless/replay/contest-authority/export preservation cases. These are intentional unchanged-behavior witnesses, not proof the seasonal change exists. Season-tagging assertions and fresh-state cases do fail at base. No test-code finding warrants a separate BLOCK.

# Rows A to L

- **A holds:** `extras2.py` rowsABDG/read and corrupt-API/read: exact integer matching; season=None matches base; corrupt exceptions match, markers return []; bytes untouched.
- **B holds:** extras2 save-default/explicit/dateless across ten tags and seven corrupt states: mismatch resets carried saver/markers, explicit saver rebuilds, dateless tag retained.
- **C holds for its state/saver/marker rules; L's input exception defect is B1:** probe dated updates, markers, saver and real scorer sequences; old state 13 → hit 1/miss 0/double 2, no carryover.
- **D holds:** probe saver sequences insert dateless updates between dated ones; base arithmetic preserved, HEAD alone retains tag as required.
- **E holds:** complete real replay stamps 2027; incomplete/today-applied replay preserves bytes; tagless marker protection remains effective at base and HEAD.
- **F holds:** extras2 real CLI reconcile outputs 1 at UTC/ET year boundary and 0 after actual hit→miss correction; both summary arms use receipt-clock ET year (also checked by changed tests).
- **G holds:** extras2 diagnostics over matching/nonmatching/invalid tags and corruption; fresh `(0,True,None)` only for readable mismatch, corrupt model-only fallback still raises.
- **H holds:** reads preserve tagless bytes; first dated hit stamps and starts at 1 (`probe.py`, changed tagless tests).
- **I holds:** zero source diff for all six specified modules; tagged-file health/web/verify/export/backup checks pass. Tag-loss limitation is N1.
- **J holds:** existing assertions/incident AST audit above; fixture and clock adaptations preserve intended behavior, with explicit tagless-test scope replacement noted.
- **K holds by design:** readable untagged/old model state reads 0 before complete replay; replay restores reconstructed season values; no migration on reads.
- **L does not hold:** 328 ordinary-input comparisons pass; two hook-bearing inputs violate totality and preservation of base read behavior (B1); unrestricted no-stall guarantee needs R1.

# What I ran

- Every added/touched test file (14 files) through `~/bts-runs/yield_run.sh`, with `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1`, `.venv/bin/python`, pytest `-p no:cacheprovider -q`; no whole fast suite. Final future-clock runs: **699 passed, 8 skipped** each at 2027-11-16 and 2028-11-16 (`tick2-clock*.log/xml`). Base new-test run: 92/163 as above. No date-sensitive failure remains. Date-relative fixtures still exist in cutoff's post-cutoff test and scheduler `_game`, past-wakeup and sleeping-heartbeat tests; they adapt to the clock instead of asserting a fixed current year.
- `probe.py base/current`: 464/507 records; date matrix, seven corrupt payloads through scoring/display/selection/replay, saver 9–16, canonical→alternate spelling→retry (`6,7,7` at BOTH versions), four ordered season-boundary scenarios. Opening hit/miss/void/double yields HEAD 1/0/unchanged-old-bytes/2; day-two double and 02:00 replay yield 3/2/2/4. Crash→same-day incomplete replay→restart loses/doubles nothing.
- `extras2.py base/current`: 118/125 records; reader/save/diagnostic APIs, actual 200,000-level recursion payload, invalid-date no-I/O guards, five simultaneous CLI/poll races each yielding exactly 6 from 5, real summary correction, bounded hook timeout. Non-peer corrupt picks remain retryable; simulated peer picks are already terminal. Complete history rebuilds all seven corrupt states; incomplete history preserves them, exactly as base.
- `exit_paths.py base/current`: 91 comparisons each across missing/already/pending/error/undelivered/stale/vanished/capped/suspended exits and seven corrupt states; every outcome and file byte matches base.
- `boundary_race.py base/current`: one directory from Dec 31 scoring through winter, truly concurrent Opening Night scorers, day-two double crash, marked-unresolved replay protection, restart, 02:00 replay. HEAD: 13 → winter fresh read 0 → 1 → 3 → 3 → 3 → 3; base carries to 14/16 until replay corrects to 3. Files and all readers captured at every step.
- `counterexamples.py`, `final_audit.py`, `verify_matrix.py`; final fixture assertions pass. `git diff --check`, tracked/index diffs clean. No tracked file changed; all probes/logs/report are inside this review directory. No network, box or contest account used.
