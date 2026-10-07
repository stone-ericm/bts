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
