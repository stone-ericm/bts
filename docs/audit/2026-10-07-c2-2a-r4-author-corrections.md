# C2 step 2a, code review r4: author corrections (2026-10-07)

These corrections were written after the round-4 BLOCK (`docs/audit/2026-10-07-c2-2a-code-codex-r4.md`, sha256 `b89eba78…`).
- 2a is stopped for this cycle under Eric's row `C2-2a-review-r4`. These corrections change nothing about that.
- The reviewed branch `c2-capture-2a` stays at `0e32f0d`, exactly as reviewed. By the manager's instruction, the corrections are recorded here on main only.

## 1. The README's retirement claim was wrong
**The claim.** The branch's evidence README (`docs/audit/2026-10-06-c2-2a-evidence/README.md`, revision-4 section, "Ledger") says this of the 71 retired ledger entries: "None of their anchors occurs anywhere in the code (checked)."

**The fact.** At `0e32f0d`, seven of those anchors still occur, exactly once each, in `src/bts/serving_witness.py`:
- C5, C7, C8, C9, C13 and C13b, all originally in `calibrate.py`;
- O8, originally in `orchestrator.py`.

The author rechecked this against every file under `src/bts`, and found 7 of 71. The class repair moved this code. The rules it implements were not removed.

**How the claim got through.** The check behind "(checked)" was never part of `build_spec_r4.py`. It missed code that had moved to another file.

**Test sensitivity.** These results are from the reviewer's r4 report (section "Ledger"); the author has not re-run them:
- The reviewer re-anchored the six non-equivalent entries in a copy of `src`: **C5, C7, C8, C9, C13b and O8**. **All six are RED under the current tests.**
- O8 needs its no-sklearn case to go RED.
- C13 keeps the default-True equivalence it already had.

**Corrected disposition.** For these seven, the retirement reason "the code it mutated was replaced by the r4 class repair" in `mutants_retired_r4.json` is wrong. The correct reason is: moved to `serving_witness.py`; six are covered by the current tests (per the reviewer's re-anchored run), and C13 is equivalent.

## 2. R4-5: a self-repair that introduced a defect
**What happened.** The all-scenario sweep at `9116776` failed 19 runs, all of one case. A faulted run's sibling digest (`samples_sha256`) changed because the part it hashes had lost a field. That was a legitimate loss, which the verdict refused.

Mid-run, the author changed the sweep's verdict oracle in `c0adb12`, red first in `tests/c2_2a/test_sweep_verdict.py`:
- the new rule: a changed `samples_sha256` or `map_sha256` stands only if it is the canonical digest of the faulted run's own published part;
- the review prompt disclosed the change.

**The defect.** The new rule ran only when the digest differed from the plain run's. If the digest was unchanged but its part had changed or was missing, the run still passed (R4-5, which the reviewer reproduced). The new tests covered changed digests only, never an unchanged one.

**What it was.** This was a repair to the checker, made after a failing run, and it introduced a false pass into the checker. The reviewer does not claim that any of the 3,394 retained runs emitted such a stale digest. The defect is in what the oracle could catch.

**For next cycle's method:**
- **A change to an oracle after a failing run is a change under review.** Give it red-first tests in both directions for every value it newly admits or refuses, including the unchanged value. Then re-run the whole sweep under it.
- **Bind, don't except.** The reviewer's required change 5 gives the rule: every non-null digest must match its own non-null published part, whether or not it equals the plain value. A rule that binds every value avoids the conditional exception that let this case through.
