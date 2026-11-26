# Verdict

**BLOCK.** The arithmetic is fixed. The recorded restart history supports the log-based completion check for this serial queue. The tool still accepts some replaced inputs, can overwrite its evidence when publishing JSON, and can return process exit 1 for invalid input.

## Round-one must-fixes

| # | Resolved? | Finding |
|---|---|---|
| 1 | Yes | The production CLI uses the eight queue seeds and five seasons, rejects grid flags, and passes a complete synthetic grid. The test seam remains callable only from Python. |
| 2 | Yes | The scorecard agrees on the writer shape. Exact cell fractions, equal season weights, sample variance, the one-sided t at 7 df, and the strict lower and inclusive season floors match R52. |
| 3 | Partial | The last-start check handles the documented serial restarts. A successful log entry plus mtimes cannot attest that the current bytes are the completed writer's files. |
| 4 | Partial | Missing, truncated, mixed-year, and unpaired dates are blocked in the suite. Numeric coercion lets some malformed columns reach unsafe operations, and equal day sets can both be incomplete. |
| 5 | Partial | JSON is atomic and per-season streaks are printed when requested. JSON can replace inputs, and streaks reopen the files and can report a silent `null` for exact P(57). |
| 6 | Partial | All 27 tests pass and the two floor tests discriminate. Add regressions for the cases below, including streaks. The unequal-season-length test's comment says PASS even though its −2 pt season fails condition 2. |

## Queue history and new findings

- **This queue.** Old seed `2048` started at Nov 20 12:51:44, was killed at the reboot, restarted at 13:27:42 and ended `rc=0` at 15:25:46. The newer start supersedes the orphaned one. For old seed `2207587974`, the Nov 22 03:43:48 start supersedes 01:18:32, and neither has a logged end in the snapshot. Its new arm has no start. The brief supplies the partial-directory removal, and the guard would require the later start even without that removal. The runner writes all five seasons before exit 0 and creates DONE afterward. This suffices for the legacy restart history once both remaining runs succeed, assuming the same host time zone, serial execution, and untouched outputs. The 16 successful run windows and markers suffice without a separate `ALL DONE` line. [Progress log](/Users/eric/bts-runs/bullpen-bt/progress.log:13) · [Runner](/Users/eric/bts-runs/bullpen-bt/run_all.sh:9) · [Writer](../../src/bts/simulate/backtest_blend.py#L963)
- **File identity and attempt pairing.** A replacement stamped `end + 3s`, after DONE, changed a synthetic PASS into FAIL. Cropping the same dates from both arms and retaining in-window mtimes changed FAIL into PASS. An overlapping pair of starts with one end and a stray success end after a failed end also produced PASS. Consume each active start on its first end, reject backwards/orphan terminal records, and allow supersession only for the verified serial restart. Bound the file mtime by DONE and by the end's logged second instead of `+5s`, and score an immutable snapshot. Use producer digests or independent writer coverage to reject same-window substitutions. The legacy log alone cannot prove that identity. [Comparator](../../scripts/audit/bullpen_r52_compare.py#L57)
- **Invalid inputs can look like FAIL by status.** Parquet rank strings raised `TypeError` in the raw scorecard, and infinite rank raised `IntCastingNaNError`. Invalid log UTF-8 and a matched impossible date also escaped as exceptions. Enforce writer-compatible finite ranks and binary outcomes before scoring, and convert schema/coercion, marker I/O, and progress decode/parse failures to `Incomplete`. The CLI must return 2 and suppress a verdict for these inputs. [Parser and loader](../../scripts/audit/bullpen_r52_compare.py#L57)
- **Result can destroy its own evidence.** In a full synthetic queue, `--json` naming a season parquet replaced it and returned PASS. Repeating the comparison returned 2. Reject resolved JSON destinations inside the evidence tree or at `progress.log` before writing. Pass the validated frames to streak reporting and explicitly label null or non-finite metrics as unavailable. A synthetic scorecard returned `p_57_exact=None` without raising, and the tool printed it without a diagnostic. [Publication and streaks](../../scripts/audit/bullpen_r52_compare.py#L172)

## What I ran

Read the ruling, revised tool and tests, writer and commit diff, runner and progress history. The single allowed test file passed 27/27. Ran three full-grid synthetic probes with `nice -n 19`, covering validity, provenance and publication. [Probes](audit_synthetic_r2.py)

## What I did not run

No reads, listings, schema inspection or computation on real outputs. No real comparison, backtests, wrapper, full suite, network, or independent mutation rerun. Wrote only review files and synthetic scratch in the review directory.
