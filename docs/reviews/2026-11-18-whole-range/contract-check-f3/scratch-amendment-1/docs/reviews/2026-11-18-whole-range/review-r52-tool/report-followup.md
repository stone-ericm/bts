# Verdict

**BLOCK.** The fixed-grid arithmetic matches R52. The comparator still issues PASS or FAIL on synthetic inputs whose coverage or identity it has failed to establish, and an invalid JSON destination can exit 1. [Ruling R52](../../docs/reviews/2026-11-18-whole-range/FINDINGS.md#L105)

## Earlier findings retested

| Earlier finding | Still caught? | Follow-up evidence |
|---|---|---|
| R1.1 / R2.1 Grid overrides and duplicates | Yes | Both flags exit 2, duplicate seeds and seasons are rejected through the Python test seam, and the default CLI checks the eight seeds and 40 pairs. A 15-run grid exits 2. |
| R1.2 / R2.2 Exact floors and equal weights | Yes | 138→137 hits on 200 days gives FAIL with condition 2 true. A single season at exactly −1.0 pt gives PASS on the fixed grid. Equal season weights, sample variance and the one-sided t at 7 df agree with R52. |
| R1.3 / R2.3 Stale files, incomplete attempts and late replacements | Partial | Missing DONE, unsuccessful or later unfinished attempts, pre-start mtime, end+3 s replacement and late DONE are blocked. A replacement retaining an in-window mtime flips PASS to FAIL and its replacement digest is the one scored. |
| R1.4 / R2.4 Truncation, partial coverage and unpaired dates | Partial | Byte truncation, missing columns, mixed years, one readable one-day file, equal-count shifted dates and the same crop in both arms of one seed are caught. Cropping day 138 in every arm, seed and season flips FAIL to PASS. Removing one lower-rank row while retaining every date also passes. Date mismatch reports counts and omits the differing dates. |
| R1.4 / R2.4 Invalid ranks and outcomes | Partial | Duplicate rank 1, rank gaps, strings, booleans, infinite ranks, and null or nonbinary rank-1 hits are rejected. Null and nonbinary rank-2 hits, floating-point rank dtype and numeric timestamps posing as writer dates pass. |
| R2 Overlap, stray records and unreadable logs | Partial | Different-run overlap, orphan success after a well-formed failure, backward time, invalid UTF-8 and a matched impossible date exit 2. Malformed runner records and out-of-grid seeds can still yield PASS. |
| R1.5 / R2.5 JSON publication | Partial | Ordinary JSON writes precede the verdict. The direct Parquet and progress-log targets stay intact and an unwritable ordinary destination exits 2 without a verdict. A path alias bypasses the evidence guard and a symlink loop escapes it. |
| R1.5 / R2.5 Streaks and null metrics | Partial | `--streak` uses the validated frames per season, prints both numeric metrics when available and labels null/nonfinite metrics in stdout and JSON. Nonpositive trials exit 2. The default invocation omits both metrics that R52 says to report beside the verdict. |
| R1.6 / R2.6 Discriminating fixtures | Improved | 42 tests pass. Three in-memory mutants for the floor inequalities and day-weighted seasons are caught. A separate repeated-rank-1 probe confirms the raw scorecard averages by day (0.75) and the comparator rejects that writer-invalid shape. |

## New findings and remaining blockers

- **Completion and bytes still disagree.** A deterministic rename after `stat()` and before `read_bytes()` scored bytes whose actual mtime was later than DONE and flipped PASS to FAIL. Bind metadata and reading to the same open file, check for changes during the snapshot, and verify the snapshot against independent run identity, digests and expected date/row coverage. Matching day sets and hashing at comparison time leave the demonstrated substitutions undetected. [Comparator](../../scripts/audit/bullpen_r52_compare.py#L103) · [Writer](../../src/bts/simulate/backtest_blend.py#L963)
- **The log parser discards malformed records.** Appending `start arm=old seed=1 extra` leaves PASS in place despite a possible later attempt. A malformed `rc=-1` end is skipped and the following stray success counts. A start with an `rc` and an open unknown seed also pass. Recognize runner records before strict parsing, constrain them to the queue and allow the known operator note. [Log parser](../../scripts/audit/bullpen_r52_compare.py#L64)
- **Impossible completion ordering passes.** A directory stamped like DONE, a file stamped exactly at start, DONE stamped before the logged end, and a file stamped at end+1.25 s with DONE at end+1.5 s all pass. The writer saves the files before the runner logs end and touches DONE. Validate marker type and both sides of the timestamp bounds. [Comparator](../../scripts/audit/bullpen_r52_compare.py#L103) · [Runner](/Users/eric/bts-runs/bullpen-bt/run_all.sh:9)
- **Path resolution gives both a false verdict and a false FAIL status.** With a symlinked `--root`, comparison reads the lexical parent's `progress.log` while the JSON guard checks the resolved parent's tree. Publishing to the actual log returned PASS and corrupted the log. A self-symlink JSON target raised uncaught `RuntimeError` and exited 1 in the CLI subprocess. Use the same resolved evidence paths across scoring and publication and return 2 on resolution errors. [CLI](../../scripts/audit/bullpen_r52_compare.py#L257)
- Validate writer-compatible dates and ranks and binary outcomes on every row. Enable streak reporting on the normal R52 invocation, with per-metric unavailability diagnostics when needed. [Cell checks](../../scripts/audit/bullpen_r52_compare.py#L124) · [Streak reporting](../../scripts/audit/bullpen_r52_compare.py#L210)

## Deliberate choices for this queue

- Leaving the live runner alone is acceptable. Its log and file times are useful ordering evidence. Using them as the sole proof of complete, unaltered inputs is unacceptable for this deploy condition, even with SHA-256 values for the bytes scored.
- Same-run supersession is acceptable for the recorded serial restarts of old seeds `2048` and `2207587974`. The log snapshot has 14 successes and no successful end for the last old run or start for its new arm. [Progress log](/Users/eric/bts-runs/bullpen-bt/progress.log:13)
- The intended after-start, through-DONE file interval and two-second maximum DONE slack are reasonable on this host given the supplied 0.03–0.97 s gaps and whole-second logging. The current checks omit ordering constraints demonstrated above, and timestamps alone cannot establish identity.

## What I ran

Read both earlier reports, the ruling, comparator, tests, scorecard, writer, runner and progress log. At verified `nice -n 19`, ran the single test file (42 passed in 3.72 s) and four bounded synthetic probe groups. The writer probe exercised date32, tied scores, doubleheader rows, NaN scores with ranks, and both real streak metrics with one trial. [Reproducible probes](probe_r52fu.py)

## What I did not run

No real comparison or access of any kind to the real `out/`, including DONE mtimes. No backtest, wrapper, full suite or network. Wrote only under this review directory and changed no tracked files.
