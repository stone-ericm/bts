## Verdict

**SIGN** — reviewed `28ae3aa9..861c8aac`, with a same-fault scan of `8421e3cd..HEAD -- src`. No tracked file changed.

## B1 and B2

- **B1 closed.** Real `season-contract status` commands swept calibrated decoder/rendering boundaries, depths through 50,000, arrays/objects/mixed nesting, two displayed fields, and positions first/middle/last among 26 rows. Every HEAD invocation emitted exactly one ledger header. Decoder failures and decoded-but-unrenderable objects count as bad lines; renderable objects count as rows. Their sum equaled the input line count throughout the depth sweep.
- The last ten accepted objects remain in file order, including when an unrenderable object is before the displayed tail. Missing keys print `—`; blank, malformed and non-object lines count as bad. Probes included 4,300/4,301/10,000-digit integers, a million-character string and nested containers.
- **B2 closed.** All JSON season types, current/other integer years, and all three persisted saver states were exercised. Both boolean seasons are invalid; valid integer seasons retain the current/another-season distinction.
- **Row 1 preserved.** Per revision: 15,563 filesystem cases, including every combination of absent/valid/malformed/non-UTF-8/directory/dangling across five files, both default and 2028 inspection seasons, cold directories and injected lstat/open denials. Outputs and exits matched exactly. Another 1,108 edge cases per revision yielded only 48 B1 and 6 B2 differences; no unexplained difference.
- Filesystem snapshots and audit guards found no writes or creation. Failed sections retained later sections; only a corrupt switch produced exit 1.
- The 97,193,890-byte ledger with 5,000 lines × 200 prediction objects completed with identical output: old 0.337 s / 800 MiB peak process RSS; HEAD 0.292 s / 416 MiB. Rendering all rows caused no observed time or memory regression.

## Blocking findings

None.

## Notes

- **N1 — test portability, not an observed flake:** `tests/test_season_contract_cs4b1_review.py:81,97,159` assumes a decoder boundary within 500–15,000, a 40-depth caller offset, and rejection of depth 12,000. These depend on the interpreter's decoder/stack budget. An interpreter with different limits could miss B1's rendering boundary or fail the M2 test for its fixed input. Expand/calibrate those bounds when supporting another build. All cases passed here on CPython 3.12.16, also with Python recursion limits 500, 2,000 and 20,000.
- The added/touched source contains no `isinstance(..., int)` checks. Another 414 real status probes exercised switch/automatic integer validation and deeply nested decoded fields in every section, including undisplayed ledger fields; no further bool-admission or misreported rendering failure was found.

## Tests

- At `28ae3aa9`, the adopted seven cases produce **3 failed, 4 passed**: B1 fails on duplicate ledger headers; B2's true/false cases fail on the wrong classification.
- The four legitimate old passes cover wrong enable season before state I/O, genuine deep-JSON handling, automatic season diagnostic before activation ID, and entry-clock capture across midnight. Those behaviors were already correct before this fix commit.
- On HEAD, the adopted file passed **20 consecutive runs per clock year** (2027 and 2028; 40 runs / 280 cases). Each of the seven cases also passed individually. Running it after the other two CS4b-1 files passed **194 cases per year**. No date or ordering dependence was observed; fixtures explicitly pin the operation clock to 2027.
- Each adopted case can fail: separate scratch-source mutations reverting B1/B2, moving enable's read earlier, dropping RecursionError handling, reversing field-check order, and rereading the observation clock killed all seven corresponding cases.
- The eleven author-test lines cover valid empty predictions on a real not-switched run (no DM, status and exactly one ledger line) and a dry run (neither file); the existing assertion preserves prior automatic bytes. Both controls pass. Bypassing the dry-run return fails the new status-existence assertion at line 312; suppressing real unswitched persistence also fails. An unwanted automatic write fails the prior-byte assertion. Broader existing persistence tests cover write order and saver I/O.

## What I ran

- Three immutable scripts, each through one sequential `~/bts-runs/yield_run.sh .codex-review/cs4b1-r2/<name>.log env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ...` invocation; all wrappers exited 0. Each nice warning was followed by waiting for that run to exit; no duplicate run was started.
- `status_family.py` → `status.log`, `status-summary.json`, per-revision matrix/edge/benchmark JSON and difference records; 33,344 command invocations across archived old source and HEAD.
- `tests_family.py` → `tests.log`, `tests-summary.json`, individual `test-*.log/json`; repeats, isolated and ordered runs, old-source checks, recursion-limit checks and nine production mutations. Expected mutation/old-source failures are recorded separately from passing controls.
- `source_family.py` → `source.log`, `source-audit.json`; diff/AST integer-check scan and 414 further real status commands.
- Every Python process installed a network-denying audit hook. No network, box or contest account was used. `git diff --check` and tracked-source/test identity checks passed; all review artifacts are inside this review directory.
