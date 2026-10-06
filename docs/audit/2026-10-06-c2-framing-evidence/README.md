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
