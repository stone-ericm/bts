# C2 step 2a evidence (design `docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md`)

## Mutant ledger (§5.4)
- `mutants.json` — 59 mutants, one per §3 rule, plus the behaviour mutants B1 (attrs lost between the local tier and
  persistence) and D1–D3 (decision changes that leave slate and pick bytes as they were). Pinned at `ed5bebd` (58);
  C13b added after the first run (below).
- `mutant_runner.py` — applies each mutant, runs its named tests (golden scenarios selected with `-k`), prints each
  failing test, restores the file by hash.
- `mutants.out` — the runs:
  - first run at `ed5bebd`: 55 RED, 3 SURVIVED (C9, C11, C13);
  - C9 (a binding's `p` rounded) survived because every fixture probability had two decimals; C11 (`n_fit` from the
    inputs) survived because every fixture input yielded exactly one sample. New tests
    `test_a_binding_carries_the_exact_appended_float` and `test_n_fit_counts_samples_not_files` kill both (resume run);
  - C13 (`increasing` hard-coded True) is **equivalent**: the production constructor keeps sklearn's default
    `increasing=True`, so `increasing_` is True on any data (checked on decreasing data). C13b (the y thresholds taken
    from the X thresholds) replaces it as the map-field mutant and is RED.
- **Result: 58 of 58 attributable mutants RED; 1 equivalent.**
