# Verdict

**BLOCK** at `acea53f8`, against the amended contract. Round-1 B1 is closed. One newly demonstrated contract gap remains: Python's integer-decoding ValueError escapes the new model-isolation helper and both defensive pick scans. This input also failed on main; CS1's required isolation/skip behavior still does not cover it.

# Round-1 findings

**B1 closed by the explicit row-4 amendment.** Empty/malformed recorded_at strings now intentionally refuse required reads. The former `test_row4_contract.py` acceptance tests are superseded and were not treated as regressions. The existing test_unparseable_recorded_at_does_not_date_an_observation now pins the amended rule.

Attacked the ruling with actual writer code: 140 combinations of 2026 dates, timezone/naive inputs, and microsecond precision through both the June 6 historical observation builder and today's builder. All emitted timestamps parsed on Python 3.12.16; timestamp-only automatic files loaded successfully. Also ran the actual May 29 manual-writer function body and today's manual CLI: both write parseable UTC recorded_at plus source_date. The dashboard does not write contest observations. No real-writer timestamp counterexample was found; no actual box/account files were inspected. A damaged recorded_at in an otherwise valid object was overwritten successfully by the real fetch command with synthetic transports. The amendment's stated recovery works for its stated case.

**N6 closed for the reported list/object result values.** e3fd5eef's string guard is sufficient after decoding: nonstrings cannot enter set membership; `_has_unconfirmed_miss` uses equality, which does not hash JSON values. All 20 ordinary/result representations tested, including nested containers and Python's nonfinite floats, behave correctly; a 1,200-level array is skipped too. The three new regression parameters fail main and pass HEAD. Decoder failures before either value check are a separate gap, B2 below.

Reran the complete round-1 probe with separately preserved output (`probe-results-r2.json`, `probe-r2.log`): 55 row/nearby cases, 216 date combinations, 28 precedence cases, and 2,178 model/flag combinations. N1–N5 still hold within their tested inputs:

| Note | Re-established result |
|---|---|
| N1, producer compatibility | Ordinary and null-coverage fetch files still pick; 58/58 is intentionally refused under row 3 and a real manual CLI override restores selection |
| N2, date/freshness disagreement | recorded_at-only and malformed-primary/valid-as_of files still load as stale with source_date=None; health remains CRITICAL; future date behavior unchanged |
| N3, precedence | Active manual wins before bad auto parsing; expired/legacy manual cannot rescue corrupt/invalid auto; required/optional undated behavior matches the amendment |
| N4, model/scorer isolation | Zero differences among the 2,178 object-value/flag combinations; scorer helpers remain byte-unchanged; B2 adds a previously untested decoder boundary |
| N5, optional consumers | Actual selection, full dashboard, captured scheduler DM delivery and decision.json still handle unavailable diagnostics correctly for the tested corruptions |

# Blocking findings

**B2 — Valid nonobject JSON can still block a usable contest observation.** In `src/bts/contest_state.py:314`, json.loads can raise **ValueError**, not JSONDecodeError, when converting an integer longer than Python's configured limit. The new helper's handler at line 318 omits that exception, so the observation is never reached. The same omission remains in the changed scan handlers at lines 230 and 276.

Concrete reproducer on this interpreter (default limit 4,300): `streak.json` is exactly **4,301 ASCII 9 digits**, a valid JSON number and therefore a nonobject body. Alongside it, use `account_state/contest_streak.manual.json` containing `{"active_streak":7,"best_streak":20,"source_date":"2026-06-18","override_expires_at":"2026-06-18T23:59:59Z"}` and now=2026-06-18T16:00:00Z. Both required modes raise raw ValueError instead of returning contest streak 7 with unavailable diagnostics. A valid automatic observation fails identically. Real `orchestrator.run_and_pick` with an eligible .99 prediction also aborts with ValueError before selection, despite that usable manual observation.

This violates row 5's nonobject isolation in a newly added function. With no observation, the same input also violates rows 6/7's required domain-error precedence. Put the scalar in `2026-06-18.json` instead and both scans raise ValueError rather than skipping it, violating row 1. An object containing a similarly large numeric result also fails before the new type guard. JSON validity was checked with json.loads(parse_int=...) without changing the interpreter limit.

Smallest fix: include ValueError in the handlers of `_load_model_diagnostics`, `_resolved_pick_dates`, and `_has_unconfirmed_miss`; retain the scorer helpers and the integer limit. An untracked scratch candidate with exactly that exception-handling change returns the contest observation, supplies both None diagnostics, preserves the missing-contest/model-corrupt domain errors, skips the bad pick, and completes real selection as a single. **Eight new regressions fail HEAD; all eight plus the 81 changed-module tests pass in the candidate package (89 passed).** No tracked implementation was edited.

Evidence: `probe_r2.py:decoder_limits`, `probe-extra-results-r2.json` → decoder_limits.model_cases/manual/required=True (including real_run_and_pick), `test_decoder_limits_r2.py`, and `decoder-head-r2.log` / `decoder-candidate-r2.log`. The failure on main is recorded too; this is an unmet CS1 repair requirement, not a claim of a newly introduced failure on an otherwise healthy input.

# Notes

**(a) Verified.** For automatic 13/10, dated 2026-06-18, model streak 4 and an active saver flag, main's saver_transition_response returns 200 and the real do_POST path returns 303. HEAD raises ContestStateError through web.py:180/_saver_season; do_POST at line 1819 writes no response status. A previously displayed/cached form submission therefore gets a dropped/error response, not a 409 page. Main already did exactly this for malformed JSON or a nonobject automatic file. HEAD's ordinary GET omits saver controls because their rendering catches the error. The newly failing input is a consequence of mandated row-3 rejection; the unhandled HTTP error is an existing defect in untouched web.py. **NOTE for a dashboard error-handling follow-up, not a CS1 blocker.** Tested the actual handler with memory streams, without sockets.

**(b) Verified.** On that row-3 file, HEAD's full rendered dashboard shows “Streak State Error” while its main number is the model replay value 4; main shows contest 13 with “Replay 4.” The same fallback display already occurs on main for malformed automatic files. This newly reached state follows row 3; the misleading number-on-error presentation belongs to the same untouched dashboard rendering follow-up. **Not a blocker here.** Rendered HTML is retained under r2-cases/.

**(c) Verified.** With the active flag, main health returns [] for that 13/10 file with expected=False; HEAD returns CRITICAL. Main already returns CRITICAL for malformed automatic files with expected=False. The flag controls missing-state/date expectations, not the existing malformed-state exception handler. Row 3 deliberately classifies this file as invalid for every caller. **Intended CS1 behavior; no blocking regression or necessary edit.**

**(d) Verified, pre-existing.** Ran main's CLI source and HEAD's CLI, with actual disk bytes and synthetic transports. Prior 0xff raises UnicodeDecodeError; prior [] raises AttributeError at cli.py:2035. Both commands exit 1, never call the profile transport, and leave bytes unchanged. Broken JSON is caught and overwritten successfully. A valid object with only damaged recorded_at is also overwritten successfully. Thus the amendment's specific “next fetch or manual observation” recovery is true; it does not establish next-fetch recovery from every corruption. Real manual commands recover loader state beside both 0xff and [] on both versions. **NOTE for fetch hardening/CS4 follow-up; cli.py is unchanged.** No code change is needed for manual recovery, followed by another selection attempt.

**(e) Verified, unchanged.** `{"streak":null}` without contest state returns streak=None on both versions. The real resolve_policy_decision raises the same TypeError (`'>=' not supported between NoneType and int`). This is an existing model-value validation issue in the scorer/strategy contract, not the new unavailable-diagnostics case. **NOTE for a separate model-value validation follow-up alongside CS2; not a CS1 blocker.**

The new optional fields have no additional readers at this commit. The CLI/web/fetch source files are byte-unchanged against main. No further unapproved state-selection or precedence change was found.

# False greens and changed assertions

Ran all 19 adopted test nodes and the three new unhashable-result nodes against main's loader in a scratch package: **14 failed, 8 passed, 50 deselected**. All pass HEAD. Adopted-file breakdown:

| Test (test_contest_state_cs1_review.py) | Fail/pass on main |
|---|---|
| test_chmod000_unreadable_files_across_rows_1_2_5_6_7 | 1/0; fails at row 5, so later checks are not independent RED evidence |
| test_row3_auto_additional_bounds_rejected_in_both_modes | 5/0 |
| test_row3_auto_exact_boundaries_accepted | 0/3; intentional preservation |
| test_row3_real_unexpired_manual_v2_bounds_remain_accepted | 0/5; intentional sequencing preservation |
| test_corrupt_manual_alone_and_beside_corrupt_auto_raises_manual_domain_error | 1/0 |
| test_fetch_writer_out_of_bounds_best_rejected_and_manual_override_recovers | 1/0 |
| test_expired_or_legacy_manual_does_not_rescue_invalid_or_undated_auto | 2/0; fails at first row-3 assertion before the row-4 portion |
| test_row5_optional_diagnostics_flow_cleanly_through_dm_decision_and_subtitle | 1/0 |

The three new defensive-read parameters each fail main with TypeError. No adopted assertion requires behavior forbidden by the amended contract. Manual impossible bounds remain permissive by the explicit sequencing exception; malformed-primary/as_of stale behavior is intentional preservation. The eight main-passing nodes pin preservation and are appropriate.

Evidence limits are NOTES: the adopted “end-to-end” test composes formatter/writer calls directly rather than driving selection/delivery; my rerun drives the real pipeline. Its header mentions malformed-primary/as_of coverage absent from that file, but the existing defensive-read test and both probes cover it. The combined chmod and expired-manual tests stop early on main, so their later assertions alone do not prove independent RED coverage; separate existing tests/probes supply it. None tests Python's integer-decoder failure, explaining why the repository tests stay green despite B2.

Since round 1, no existing assertion was weakened or deleted: three parameters and the adopted file were added. The two pre-main automatic-untrusted assertions replaced by row-3 rejection remain the deliberate earlier change. My two old string-acceptance tests are superseded by the authorized amendment. The eight new decoder regressions are ready for adoption and preserve the amended requirements.

# What I ran

- Read `main...HEAD` and `7cff54d5..HEAD`, the amendment, all new tests, historical/current writers, and refreshed caller inventories (`callers-r2.txt`, `producer-calls-r2.txt`). No network, box, account access, or tracked writes.
- Reran `probe.py probe-results-r2.json`; ran `probe_r2.py` directly with TZ=America/New_York and PYTHONDONTWRITEBYTECODE=1. Logs/results: probe-r2.log, probe-results-r2.json, probe-extra-r2.log, probe-extra-results-r2.json. Ordinary types, integer-decoder limits, actual handlers, both CLI versions, policy calls, historical producers, 140 timestamp combinations, and scratch repair exercised. Socket connects are forbidden; transports captured.
- All pytest invocations went through `~/bts-runs/yield_run.sh <log> env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 ... .venv/bin/python -m pytest -p no:cacheprovider ... -q`; full commands are in `.log.watch` files.
- Fresh focused inventory plus fetch/strategy/orchestrator/posting/decision/dashboard caller files: **878 passed, 8 skipped** (`focused-r2.log`). No whole fast suite.
- Main scratch negative control for 19 adopted plus 3 new nodes: **14 failed, 8 passed, 50 deselected** (`main-new-tests-r2.log`, XML).
- New contract regressions on HEAD: **8 failed** (`decoder-head-r2.log`). Scratch-only ValueError repair plus all three changed test files: **89 passed** (`decoder-candidate-r2.log`).
- `git diff --check` passed; tracked worktree stayed clean. All probes, scratch packages, logs, HTML, and this report are under `.codex-review/cs1-review/`.
