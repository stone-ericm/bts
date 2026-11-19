# Whole-range C3 handover

Commit `c70effe9375fac53e0bb09c782319b2d60e69a81` on `fix/wr-c3`, directly atop `c2039541`. It is the only commit and was not pushed.

## C3 · W14/W23

- `src/bts/picks.py:848` guards pick, decision and streak reads with a nonblocking open and descriptor regularity check. The replay at line 918 visits canonical dates in order through the ET day captured under the scoring lock. Terminal results use the base fold, including legacy hits and misses, partial voids, double-down hits without slot results and the saver window. Past nonterminal picks with an explicit non-scoreable decision or no evidence of a commit or attempt contribute zero. An unresolved scoreable commit, an attempt without a decision, an applied-date marker without a result, and malformed or unreadable dated evidence block replay. A scoreable decision with a missing pick blocks replay too.
- `src/bts/picks.py:1477` guards the unlocked and locked correction reads. It keeps the 08:00 ET cutoff and writes corrections under the scoring lock. Under that lock it also reads the streak snapshot once, preserves matching-season marker spellings, and atomically writes the folded model streak and saver when the full replay succeeds. A readable tagless marker retains the exact-date crash hold. A marker tagged for another season has no effect on this replay.
- `src/bts/cli.py:2883` and `src/bts/reconcile_receipt.py:177` use the locked observation for the displayed streak and the first blocked date and reason. The CLI prints one dated replay warning on stderr, preserves correction lines and labels a readable retained streak. A blocked replay with corrupt streak bytes exits 1 and preserves those bytes. A non-regular streak target fails with its path.

## Base reproductions and controls

- `tests/test_wr_c3_reconcile.py:127` adopts the Carbon correction and corrupt-streak cases and the Codex refused-preview case with a synthetic canonical preview, the real saved-pick refusal and the real `check-results` eligibility gate. At 02:00 ET it exercises a skip leftover, a decision-only skip, private terminal history, delivered hits and both directions of scorer correction after a saver-window miss. Both cases were RED on the base and GREEN on C3.
- `tests/test_wr_c3_reconcile.py:165` was RED on the base for delivered, suspended, private and locked-unconfirmed commits, a bare attempt, a missing root pick with a scoreable decision, malformed and non-regular records, date mismatches and an unknown result. It covers the original date-and-result-only fixture as malformed pick evidence. It checks the first dated blocker when more than one exists and verifies the receipt carries the CLI reason.
- `tests/test_wr_c3_reconcile.py:223` passed on the base and C3 with the same folded streak and saver, `(13, False)`, for synthetic 2026 terminal history. The fixture contains undelivered legacy records with the delivery fields absent, records with no decisions, a saver-window miss, a whole void and a partial void, a double-down hit without slot results, and an unplayed current-day preview. The existing 08:00 cutoff and current-day terminal-score controls passed in the focused run.

## Existing fixture amendments

- `tests/test_streak_replay.py:71` previously used a date-and-result-only object for an unresolved past day. It now uses a valid pending pick with the legacy public-delivery flag. The fail-closed assertion remains.
- `tests/test_streak_season_cs2b_review.py:582` previously planted a pending pick with default legacy post and attempt fields. The pending pick now also has an explicit DM receipt ID. The scorer permission, unchanged mode and complete replay assertions remain.
- `tests/test_streak_season_cs2.py:385` previously mocked only the corrections list while describing a blocked replay, and expected `Streak: 0` or `Streak recalculated: 0`. Its mock now records a dated blocked observation and both variants expect `Streak retained: 0`. The assertion that the CLI requested season 2027 remains.

## Hostile-input checks

- Interrupt and restart. `tests/test_wr_c3_reconcile.py:327` interrupts the streak write after a completed correction, confirms that the correction survives, then retries and verifies a fresh fold. The cross-midnight marker test at line 261 holds a half-written scorer result for tagged and tagless streaks, then completes grading and refolds. The existing pp1 and tagless same-day tests passed.
- Malformed and non-regular input. The dated blocker matrix uses malformed pick and decision objects, a JSON array, a recent corrupt pick, and FIFOs for both dated file types. Line 243 changes the pick to an array between the boxscore lookup and locked correction read. Lines 305 and 317 cover a corrupt streak with blocked replay and a FIFO streak target. The existing chmod-000 control covers a readable history rebuilding a regular streak file that the scorer cannot read.
- Midnight ET and New Year. The test at line 281 moves the clock across New Year as the lock is entered and moves it back for later receipt reads. It verifies that only the locked replay season controls markers, persistence and display. Line 261 tests the earlier day's scorer marker after ET midnight.
- Failed or unconfirmed send. Reconcile adds no send. The refused preview is excluded, an attempt without decision evidence blocks, and an unresolved scoreable `locked_unconfirmed` decision blocks. The eligibility tests use synthetic delivery fields with no live transport.

## Runs

- First base attempt, `.codex-review/wr-c/red.log`. The FIFO streak test used an alarm exception the base code swallowed during its first streak read, then hung on its second read. The wrapper exited 143 after the lead stopped the pytest executable. I had first mistaken its child monitor for pytest when walking the PID tree. The bounded test now raises through those handlers and reports a failure after three seconds.
- Bounded base run, `.codex-review/wr-c/red-bounded.log`. 13 passed and 26 failed. This includes 25 C3 RED cases and the web saver loopback test denied by the sandbox. The legacy 2026 control passed.
- Focused run, `.codex-review/wr-c/green.log`. 520 passed and one failed. All 26 C3 cases, LightGBM optional, scoring and receipt controls, and 11 of 12 web saver tests passed. The remaining web saver test was denied a loopback bind by the sandbox.
- Committed-tree fast suite, `.codex-review/wr-c/fast.log`. 9,360 passed, 37 failed, 16 errors, 25 skipped, six deselected and one xfailed in 25:41. The 53 failure and error IDs matched `~/bts-runs/gate-failed.txt` exactly, with zero extra and zero missing. All 26 C3 cases, LightGBM optional and all 12 web saver tests passed in that process.
