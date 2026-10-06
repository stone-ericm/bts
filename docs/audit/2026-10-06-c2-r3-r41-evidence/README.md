# C2 (b) step 1 evidence: rank 3's R4-1 fix (author-run, outside any agent sandbox)

Produced on the Mac (Python 3.12.13) in the worktree `~/projects/bts-c2-rank3` at `d143af8`. Each file's first line records its commit and command.

| File | What it shows |
|---|---|
| `mutants.json`, `mutant_runner.py`, `mutants.out` | 9 mutants of the R4-1 rules (typed check skipped, per-outcome completion witnesses, bool or zero game ids, unsupported `kind_of`, unchecked season, unchecked `from_attempt_id`, dict-equality duplicates, unchecked feed game id): all RED. The runner is the kickoff brief's (§6), with `python -B`, a fresh `PYTHONPYCACHEPREFIX` and a hash-checked restore |
| `fast_suite.out` | the fast regression suite at `d143af8`, exit 0 |

The rank-3 tests (`tests/scripts/c1_r3`, `tests/scripts/test_c1_admission.py`) pass 128 at `d143af8`: r4's 115, plus 13 new.
