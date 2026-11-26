# C2 (b) step 1 evidence: rank 3's R4-1 and C2R1-1 fixes (author-run, outside any agent sandbox)

Produced on the Mac (Python 3.12.13) in the worktree `~/projects/bts-c2-rank3`. Each file's first line records its commit and command.

| File | What it shows |
|---|---|
| `mutants.json`, `mutant_runner.py`, `mutants.out` | 14 mutants at `117db05`, all RED: R4-1's typed check, per-outcome witnesses and their network_error / rate_limited selective omissions, bool or zero ids, unsupported `kind_of`, schedule season, `from_attempt_id`, dict-equality duplicates, feed game id; C2R1-1's schedule game id, the join identity check, and a feed carrying a season. The runner is the kickoff brief's (§6), with every named test run (no `-x`) and each failing test printed, so every RED is attributable |
| `fast_suite.out` | the fast regression suite at `117db05`: 3940 passed, 7 skipped, 6 deselected, 22 xfailed, exit 0 |

**History:** at `d143af8` (the R4-1 fix, reviewed as `21a5166`) there were 9 mutants, all RED, and the fast suite passed 3920. C2 review r1 (BLOCK, C2R1-1) led to `c50426b` and `117db05`. The rank-3 tests (`tests/scripts/c1_r3`, `tests/scripts/test_c1_admission.py`, `tests/scripts/test_c1_r3_acquire.py`) pass 184 at `117db05`.
