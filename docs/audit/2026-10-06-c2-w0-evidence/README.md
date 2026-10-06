# C2 (a) unit 1 evidence: the W0 repair (author-run, outside any agent sandbox)

All files were produced by the author on the Mac (macOS 27.2, Python 3.12.13) in the worktree `~/projects/bts-c2-watchdog`. Each file's first line records its commit and command.

| File | What it shows |
|---|---|
| `fork_probe.py`, `fork_probe.out` | Measured at `99a26c9`: with `(deny process-fork (with send-signal SIGKILL))`, every process-creation path tried kills the top interpreter (−9) before a child starts; importing `bts.cli` needs no process; the EPERM form refuses attributably except `os.system`, which swallows it (R3-4's basis) |
| `mutants.json`, `mutant_runner.py`, `mutants.out` | The pinned 36-mutant ledger, at `9e4d808`: every mutant makes its named tests fail (RED). The runner is the kickoff brief's (§6), using `python -B`, a fresh `PYTHONPYCACHEPREFIX` and a hash-checked restore |
| `watchdog_suite.out` | `tests/watchdog` at `9e4d808` with `-rA`: 143 passed, including the 13 kernel gate tests (`gate2`, `r34`), which cannot run inside the reviewer's sandbox |
| `fast_suite.out` | The fast regression suite at `9e4d808`: 4050 passed, 7 skipped, 6 deselected, 22 xfailed, exit 0 |

**History of the counts:** at `4a845e8` (before the chain fix) the ledger had 26 mutants, all RED, and the fast suite passed 4026. The chain fix (`808ae12`) added 2 tests and mutants M27–M28: 28 RED, watchdog 121, fast 4028 (the state Codex review r1 saw at `d4c11ae`). The r1 fixes (`9e4d808`) added 22 tests and mutants M29–M35 and M37 (M29 is the reviewer's token-fencing removal; M27 was re-expressed for the new claim code).
