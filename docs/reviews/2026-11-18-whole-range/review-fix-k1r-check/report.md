# Verdict

SIGN WITH EDITS

## 1. Exact revert

- The four dispatch calls match FINAL2. They omit the date override and share the wall-clock ET receipt. Each of the five contest-state callers still passes its requested slate date to the short DM. The formatter uses that date even after midnight and uses “this slate” for a missing or invalid date.
- The source diff against FINAL2 contains the short DM, its label and the `999+` banner, plus an unused `slate_date` keyword on the missed-pick helper and its caller. That keyword cannot affect dispatch. The historical runner still uses the processed date.

## 2. Rollover dispatches

- The original harness on HEAD and archived FINAL2 matched on all 39 dispatches across the June and New Year per-caller loops and both named orders. Each delivered 33 and suppressed 6. Receipt dates and date arguments matched too. [Comparison](rollover-match.log) · [HEAD](rollover-head.log) · [FINAL2](rollover-final2.log)
- DISTINCT delivered both void refusals. DIGEST FIRST delivered the first digest, the live refusal after restart and the second digest. The next-slate and repeated in-loop suppressions matched FINAL2.

## 3. Tests

- The test diff changes the dedup expectations and preserves the DM length, slate label and existing operator-text checks. The inlined overnight replay preserves the old text and CLI assertions and adds a receipt-date assertion. The B8a hash keeps its original expected digest after undoing the two new text changes, which have their own presence checks.
- `_WallClockDedupView` selects the archived wall-clock assertions. The probe still sees the real dispatch arguments and persisted receipt. The 33 alerting cases rejected the opposite date rule in each direction, and the three successful-delivery controls passed. [Revised checks on main](pytest-main-view.log) · [K1 checks on HEAD](pytest-k1-legacy-caller-checks.log)
- Both new tests passed on HEAD. Both failed on the `git archive origin/main src scripts tests` copy of `5cd58a0c`, with its `bts.__file__` printed. The second void refusal and the live refusal after the digest had `wire=False`. [HEAD](pytest-test_wr_k1r_dedup.log) · [Main](pytest-main-dedup.log)
- All 39 requested files finished with 1,816 passed and 12 skipped. The refusal DM, K1 replay, new dedup, saver flag and season-walk files passed 97, 47, 2, 63 and 26 tests respectively.

## 4. Text edits

- B8a and the README match the focused read's wording. The effective-day formula floors candidate slates at one before the 180-day clamp. The makeup and empty fixture tests passed.
- R64 check passed. The B8a replay used the real CLI in the scheduler's picks directory. Filing between checks produced one pick. Filing after the last check with no saved candidate produced zero. The CLI refused the earlier date after midnight and preserved the observation. A changed filing also appends a request audit entry. [B8a replay](b8a.log)
- The `CLAUDE.md` description of the next-slate suppression is accurate while the matching receipt covers the source. Its repeat and first-pitch statements need the conditions below. The delayed status replay used a 23:59 first pitch. At 00:01, the first late refusal delivered, the same-source retry was suppressed, and D's digest delivered. [Receipt replay](c3-receipt-with-state.log)

## Edits

- In B8a and its `CLAUDE.md` summary use “A changed filing records the request and updates the dated observation. The scheduler handles pick selection and delivery.”
- In `CLAUDE.md` and the handover's C3-2 description use “C3-2 is ledgered again. A successful in-day dispatch for slate D evaluated after midnight ET records D+1. While that receipt covers the source, it suppresses same-source in-day alerts evaluated on D+1 for either slate, including the first one for slate D+1. A delayed check or delivery can cross midnight. A same-source in-day alert for D can repeat after midnight when the latest receipt is dated D. D's end-of-day digest can send against a D+1 receipt because the digest uses D.”
- In the handover's test tally, replace “all 36 `test_scheduler_alert_dates`” with “the 33 alerting `test_scheduler_alert_dates` cases (the three successful-delivery controls passed).”

## Notes

- The `origin/main` side added R75 after the branch forked. That accounts for the apparent `FINDINGS.md` deletion in the two-tip diff. The branch commits leave the ledger file untouched.
- The host rejected the `nice` adjustment and ran the commands. Runs started after 16:40 ET and stayed serial and bounded. The initial log parser and delayed-status fixture failed during setup. The comparison was repeated directly from the raw logs and the fixture reran with scheduler state.

ran: both rollover trees, B8a and delayed-receipt probes, all requested test files and the archived negative controls; did not run: 12 pytest-skipped cases, the full fast suite, the standalone full season walk, or live network, box or contest-account checks.
