# C2 side item (e) evidence: the framing screen's mutant ledger
- `mutants.json`: 26 mutants over `scripts/audit/c2_framing/screen.py`. They cover the feature (shift, min_periods, daily mean), the self-check, the variants and blend configs, each disposition rule, the pre-registered constants, the pinned inputs, the run's refusals and stops, the aggregate, and coverage.
- `mutant_runner.py`: the runner used for C2 step 2a.
- `mutants.out`: the first run at `9637a44` gave 25 RED and F12 SURVIVED.
  - F12 swaps "both seasons above zero" for "either". Its test case also missed the practical size, so another rule masked it.
  - After the case was isolated, F12 is RED.
- **Result: 26 of 26 RED.** The tests were written after `screen.py`; this ledger is their strength check.

## Revision 2 (after review r1 BLOCK)
- **Runner revised** (r1 item 8): a mutant counts as RED only when pytest exits 1 with a `FAILED` line. Any other exit is INCONCLUSIVE, and the runner exits 1 if any mutant is not RED.
- **Ledger rebuilt: 54 mutants.**
  - F24 is retired, because its aggregate code was rewritten.
  - G1–G29 cover the new boundaries: aggregate refusals, seed order and release, labels, closed inputs, settings, the launch budget and the zero-variance t.
- **First revision-2 run at `307e7bc`:** 49 RED, 5 SURVIVED (G1, G2, G5, G9, G20).
  - G1, G2 and G9 were masked by later checks that refuse the same input with a different error.
  - G5 had no wrong-root case.
  - G20 was masked because the real park-drag attach also gives NaN when there is no table.
- **The fix:** the tests now require the specific check's message, add a wrong-root run and a consistently changed head, and spy on the table read. The resume run turns all five RED.
- **Result: 54 of 54 RED.**

## Revision 3 (after review r2 BLOCK; row C2-framing-review-r3)
- **Runner revised (r2 R2-4):**
  - RED only when pytest exits 1 with a `FAILED` line, no `ERROR` line, and a summary with no error, interruption, skip, xfail or xpass.
  - A clean pass is SURVIVED; anything else is INCONCLUSIVE.
  - Its classifier has its own regression tests in `test_screen.py`.
- **Ledger: 67 mutants.** 13 were re-anchored to the revised code. H1–H13 cover `validate_run`, the release naming seed 1's run, and the identity-based aggregate.
- **First revision-3 run at `fe5eb85`:** 58 RED, 5 INCONCLUSIVE and 4 SURVIVED.
  - **INCONCLUSIVE (F4, F20, G14, G22, G28):** each failed the test that targets its rule. It also errored in the shared three-run fixture's setup, which the strict runner does not count as clean. Each now names its targeted test, which fails cleanly. F20 and G22 fail `test_run_end_to_end` on their own assertions: the seed env and the feature settings.
  - **G4 (claim binding):** survived; it had no case isolating it. Now covered by a `CLAIM.json` rewritten to other bytes that still names its run and code.
  - **H6 (the ten pin names):** survived; the only pins case also broke the digest. Now covered by nine pins, consistently, with a digest that matches them.
  - **H8 (the admitted identity in the seed-1 release):** survived; there was no case. Now covered by a released seed-1 run made by another identity.
  - **G8: recorded equivalent.** `validate_run` binds each run's `inputs_digest` to its `input_pins`, so runs that agree on the digest agree on the pins (barring a sha256 collision).
- **Resume at `2f43386`:** the eight attributable mutants are RED. G8 survives, as recorded.
- **Result: 66 of 66 attributable RED; G8 equivalent.** The suite holds 75 tests.

## Revision 4 (after review r3 BLOCK; Eric's row C2-framing-review-r4)
**Runner revised (r3 R3-4):**
- Each mutant's intended set is what `pytest --collect-only` selects from its named tests on the unmutated source.
- A mutant is RED only when pytest exits 1 and every intended node reported PASSED or FAILED under `-rA`. At least one must have FAILED, with no `ERROR` line and a clean summary.
- An intended node that never ran makes the mutant INCONCLUSIVE, whatever the summary says. That covers fail-fast, a crash or a changed selection.
- `PYTEST_ADDOPTS` is removed from the subprocess environment, ini `addopts` are cleared, and one `--rootdir` is pinned for both passes.
- A test list holding any option is INVALID before anything is mutated.
- The runner's own tests cover all of this, including the reviewer's two-node case with inherited `-x`, which the old runner marked RED with the second test never run.

**Spec: 88 entries.**
- Every mutant now names node ids: no whole-file lists and no options, and every list collects. Most were narrowed to the tests that failed for them in the revision-3 run.
- G6, H8, H10, H11 and H13 were re-anchored to the revised code.
- The new mutants:
  - N1–N16: card recomputation, scoring settings, season evidence, P@1 coverage, the HEAD check, trusted identity and pins;
  - R1–R5: the runner's error, not-run, environment, option and exact-node rules.
- The parametrized damage and classifier cases have short ids.

**First revision-4 run at `cab2549`:** 84 RED, 4 SURVIVED (G8, H5, H12, R5).
- **H5 (identity completeness):** shadowed in the aggregate cases by the strict trusted-identity check. It is now isolated by a direct call whose admitted identity has the same empty field.
- **R5 (exact node matching):** no case had an executed node whose id merely starts with an intended one. It is now covered: `t.py::ab` ran, `t.py::a` did not.
- **G8 and H12: recorded equivalents.**
  - G8 (unchanged reasoning): each run's digest is bound to its pins.
  - H12: each run must equal the trusted identity, so validated runs cannot disagree on identity.

**Resume at `e1033c5`:** H5 and R5 are RED.

**Result: 86 of 86 attributable RED; G8 and H12 equivalent.** The permitted suite has 122 tests (97 framing, 25 shared admission).

## Revision 5 (after review r4 BLOCK; Eric's row C2-framing-review-r5)
**Runner revised (r4 R4-2 and the sandbox-only failure):**
- Outcomes now come from pytest's own reports, never its text. A plugin loaded with `-p` records each report (node id, phase, outcome) to a private file the runner creates, and writes an end record when the session finishes normally.
- A mutant is RED only when all of these hold:
  - pytest exits 1, and the session ended normally;
  - nothing errored in setup or teardown, and nothing was skipped;
  - every intended node has its own call report, and no unintended node ran;
  - pytest's summary counts agree with the records;
  - at least one intended node failed.
- Printed or captured output cannot add an outcome. The reviewer's case is now a test: a fake `PASSED` line plus a second test stopped before its body gives INCONCLUSIVE.
- **Sandbox fix:** the runner's root is pytest's rootdir and working directory (default: the repository), and the subprocess uses the runner's own interpreter. The runner tests give their scratch files a root of their own, so collection never leaves it. The round-4 failure came from pinning the rootdir to the repository while collecting tests elsewhere, which made pytest list directories outside the scratch area.
- **Stated boundary:** test code that deliberately tampers with the runner's records file or with pytest's internals from inside the run is out of scope.

**Validator (r4 R4-1, R4-3):**
- **`profile_problem`:** every retained row must be one the scorer scores. That means:
  - no missing value in date, rank, season, hit or probability;
  - an integer season equal to the unit's, and dates in the season;
  - ranks 1..n on every day;
  - hits 0/1, and probabilities in [0, 1].
- **Identity:** it must have exactly the five fields.

**Spec: 99 entries.**
- Re-anchored: H5, H13, R1, R2, R5, N7, N8, N9.
- New:
  - H14 (exactly five identity fields);
  - N17–N22 (the scored-row rules);
  - R6–R9 (normal end, summary agreement, the recording plugin, its end record).
- **Recorded equivalents, each over the validator's whole accepted domain:**
  - **G8:** each run's pins must equal the trusted pins, and its digest is bound to them.
  - **H12:** each identity has exactly the five fields and equals the trusted identity.
  - **N10:** every accepted unit is non-empty, of its own season, and ranked 1..n daily, so P@1 always covers both seasons.
- **Message-level kills (named here, not counted as distinct defences):** some mutants are refused by a later check with a different message. N17 is one: without its missing-value check, a nullable season or rank is refused by the dtype or ranks checks instead. The test pins the specific message, so the mutant is RED.

**Revision-5 run at `2bc188f`:** 96 RED, 3 SURVIVED (G8, H12, N10, as recorded).

**Result: 96 of 96 attributable RED; G8, H12 and N10 equivalent.** The permitted suite has 133 tests (108 framing, 25 shared admission).

## Revision 6 (after review r5 BLOCK; Eric's row C2-framing-review-r6, under the full rules)
**Runner revised (r5 R5-1):**
- "Ended normally" is pytest's own session status, never unconfiguration (which pytest also reaches after `pytest.exit`) and never text. The recording plugin also records:
  - the interrupt hook (`pytest.exit` and KeyboardInterrupt), the internal-error hook, and failed collection;
  - a session record written last (`trylast`) at `pytest_sessionfinish`, holding pytest's exit status and its collected and failed counters. If another session-finish hook aborts first, there is no record.
- RED requires all of these:
  - the session record, with no interrupt, internal error or collection error;
  - every intended node's setup, call and teardown reports;
  - the counters agreeing with the reports and the process return code.
- The text cross-check is gone.
- The reviewer's aborted-teardown case is a test, along with a later session-finish abort and an exit after the last teardown report. Run against the revision-5 runner, the first two fail; the aborted-teardown case was certified RED there.

**Validator (r5 R5-2):** probabilities must be numeric, never coerced. Any scorer exception while rescoring the retained profiles is a `RunInvalid` refusal.

**Spec: 108 entries.**
- Re-anchored: H13, R1, R6, R7, R9.
- New:
  - R10–R16: interrupt, its recording, setup/teardown completeness, internal error, collection error, exit status, the session record written last;
  - N23–N24: numeric probabilities, the rescoring refusal.

**Revision-6 run at `750086b`:** 104 RED, 4 SURVIVED (G8, H12, N10 as recorded; R11).
- R11 (the plugin's interrupt record) was masked in the aborted-teardown test by the missing teardown report. It is now isolated by `pytest.exit` raised after the last teardown report, where every report exists and the counters agree.

**Resume at `aa4e72c`:** R11 is RED.

**Result: 105 of 105 attributable RED; G8, H12 and N10 equivalent.** The permitted suite has 143 tests (118 framing, 25 shared admission).

## Revision 7 (after review r6 BLOCK; Eric's row C2-framing-review-r7: full rules, a redesigned runner)
**Runner redesigned (r6 R6-1, R6-2):**
- **Driver process:** the mutated run is a driver that calls `pytest.main()` in-process. It writes its evidence only after `pytest.main` returns, so an abort anywhere, including unconfiguration, leaves no evidence.
- **Sentinel:** at collection finish, after every conftest is loaded, the recorder registers a sentinel whose hooks are `tryfirst` wrappers.
  - Pluggy calls the last-registered tryfirst wrapper outermost. So its session-finish wrapper sees any inner hook's or wrapper's abort (R6-1), and its runtest-call wrapper records what each call itself raised.
  - Any plugin registered after the sentinel is refused.
- **Expected failures (R6-2):** reports keep their `wasxfail` status, and an intended node marked xfail, skip or skipif is never a killer.
- **Cross-checks:** report outcomes must match the sentinel's call observations, and pytest's counters must agree.
- **Stated boundary:** tampering with the runner's own objects, its evidence file or pytest internals is out of scope.
- **New tests,** all red against the revision-6 runner, which certified seven of them RED:
  - a session-finish wrapper abort and a tryfirst wrapper abort;
  - a non-strict XPASS and a runtime xfail;
  - a plugin registered during the run;
  - a makereport wrapper rewriting an outcome;
  - an unconfigure abort;
  - plus a 23-case classifier table over the evidence.

**Spec: 116 entries.**
- Re-anchored: H13, R6, R7, R8, R9, R11, R15, R16; R2 repointed.
- New, R17–R24:
  - sentinel registration;
  - late plugins;
  - xfail/skip markers and `wasxfail` recording;
  - report/observation agreement and the sentinel's call record;
  - exit status and session-finish completion.

**Revision-7 run at `310dfe6`:** 113 RED, 3 SURVIVED (G8, H12, N10 as recorded).

**Result: 113 of 113 attributable RED in one pass; G8, H12 and N10 equivalent.** The permitted suite has 157 tests (132 framing, 25 shared admission).
