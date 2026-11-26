# CS1 handover — defensive reads and model-state isolation

Branch: `sb-cs1` in `/Users/eric/projects/bts-cs1`.

Commit: `32b6a74b6a1bf4ed2cbfae6ca752c254339b58e5` — `fix: isolate corrupt model state from contest decisions`.
Base: `6b440ee443ec74acfec46dc770d690d9d8a0f232`.

Three files committed: `src/bts/contest_state.py`, `tests/test_contest_state_best_trust.py`, and `tests/test_contest_state_defensive_reads.py`. Nothing pushed. This handover and `CS1-EVIDENCE/` are untracked and were not committed. No external network, box, contest-account, or paid API operation was performed. Escalated test runs exercised local sockets and process controls only.

**The after-change full fast suite was NOT run by me. The lead declined that command and explicitly assigned this gate to another host. The lead must run it on this branch. A fresh adversarial review also follows this handover; neither that review nor the outstanding full gate is claimed complete here.**

## Contract mapping

All source references below are to the committed files. `T` means `tests/test_contest_state_defensive_reads.py`.

| Row | Implementation | Regression |
|---|---|---|
| 1 — bad date-named pick files are skipped in both scans | `src/bts/contest_state.py:218–237` and `:257–282`: explicit UTF-8, catches read/decode/JSON errors, object guard before `.get`. Other files continue to contribute. | `T:67`, `test_corrupt_pick_is_skipped_by_both_scans`, four corruption cases, each scan checked with the corrupt file alone and then a readable miss; also runs the required decision loader. |
| 2 — invalid contest encoding becomes ContestStateError; manual precedence retained | `src/bts/contest_state.py:135–142`; selected manual returns at `:207–209`, before automatic parsing at `:211`. | `T:83`, `test_corrupt_contest_file_raises_domain_error`, both filenames × four corruptions; `T:93`, `test_manual_precedes_corrupt_auto`, four corruptions beside a real CLI-generated unexpired override. |
| 3 — stored automatic bounds | `src/bts/contest_state.py:155–161`: checks automatic filename, rejects active > 57, best < active, or best > 57; error includes file and both values. Manual bounds unchanged. | `T:102`, `test_u1_bad_best_reachable_after_corruption`: first loads a valid producer artifact, then mutates actual bytes to 13/10, 58/58, or 13/58 and verifies the mutation. `T:117`, `test_manual_bounds_unchanged_in_cs1`, same values remain accepted/untrusted for manual. Updated best-trust tests also pin rejection. |
| 4 — required automatic observation cannot be undated | `src/bts/contest_state.py:163–166`, opt-in propagated at `:185`, `:211`, and `:340–341`. Checks parseable source_date/as_of or recorded_at string. Optional reads and manual parsing retain prior behavior. No v1 rejection. | `T:125`, `test_legacy_fixture_exception_is_live`, bare/invalid-date auto accepted as stale in optional mode but refused in required mode. `T:142`, `test_dated_version_one_auto_still_accepted`, source_date, as_of, malformed primary with valid as_of, recorded_at, and empty recorded_at string. Required and optional returned states are compared. |
| 5 — corrupt model diagnostics cannot block usable contest observation | `src/bts/contest_state.py:303–320`: one UTF-8 model snapshot, object guard, both diagnostics unavailable on corruption, short error plus one `model_state_corrupt` warning. Optional fields at `:57–58`, error field at `:69`, returned at `:399–407`. | `T:156`, `test_manual_not_universal_recovery`, manual/automatic × four corruptions; compares the whole returned decision with the healthy decision except the three diagnostic fields, checks exactly one warning, unchanged file bytes and retained directory for OSError. |
| 6 — corrupt model cannot become fallback | `src/bts/contest_state.py:348–349`: domain error includes model path/error, no invented streak. | `T:183`, `test_corrupt_model_cannot_be_fallback`, four corruptions. |
| 7 — missing required contest error takes precedence over corrupt model fallback | `src/bts/contest_state.py:342–347`, before the model fallback guard. | `T:193`, `test_missing_required_contest_wins_over_corrupt_model`, four corruptions. |
| 8 — JSON object with absent keys keeps existing defaults | `src/bts/contest_state.py:311–316` preserves absent-file/absent-key defaults. Scorer helpers `src/bts/picks.py:516–547` are untouched. | `T:200`, `test_empty_model_object_keeps_existing_defaults`: direct scorer reads of `{}` remain `(0, True, [])`; decision model diagnostics remain `(0, True)` with no error, with and without automatic state. |

The three named design probes are represented by tests with the same names: `manual_not_universal_recovery`, `u1_bad_best_reachable_after_corruption`, and `legacy_fixture_exception_is_live` (each prefixed `test_`).

The OSError fixture is an actual directory at the filename, so it fails on read even under a privileged runner. Invalid UTF-8 is a single `0xff` byte; non-object JSON is `[]`; malformed JSON is `{broken`. These are separate parameters, not substitutes for one another.

The model diagnostic helper intentionally reads one snapshot within `contest_state` instead of calling the two scorer loaders separately. Its `.get` defaults mirror those loaders. Their behavior for all other callers, including malformed-file failures and applied-date handling, is unchanged. No season logic, saver mapping, persistent schema, fetch, CLI, or alert routing was added.

## Existing tests changed

Only `tests/test_contest_state_best_trust.py` was changed among pre-existing test files:

- Old `test_best_below_streak_is_untrusted` → `test_auto_best_below_streak_is_rejected` (`:54`): row 3 deliberately replaces automatic-state acceptance/untrusted-best with `ContestStateError`.
- Old `test_best_above_target_is_untrusted` → `test_auto_best_above_target_is_rejected` (`:60`): same required change for automatic best > 57.

Both changed expectations were run RED before production changes: 2 failed (DID NOT RAISE), 7 passed. No existing fixture needed metadata repairs. Manual untrusted-bound behavior now has explicit additional regression coverage.

## Every reader of the optional diagnostics

A source/test `rg -n 'model_streak|model_saver_available' src tests` audit found:

- `src/bts/contest_state.py:57–58,337,351,359–360,399–400`: declarations, local diagnostics, model fallback and decision construction. Both diagnostics now permit None. A corrupt-model fallback is refused before construction; a usable contest decision carries both None plus `model_state_error`.
- `src/bts/web.py:238–240`, `_streak_subtitle`: the sole production consumer of the model_streak field already guards `model is not None` before comparison/formatting. No production change needed. `T:213`, `test_dashboard_handles_missing_model_diagnostics`, loads actual corrupt model bytes plus valid automatic state and verifies `Contest State`, with no replay label or crash.
- No production reader of the model_saver_available field exists outside its declaration/construction.
- `tests/test_contest_state.py:70,104,220`: healthy numeric model_streak assertions; unchanged and passing.
- `tests/test_web_streak_subtitle.py:10`: helper constructs healthy diagnostics; unchanged and passing.
- `T:168–171,207`: new optional-diagnostic assertions, full decision comparison and healthy legacy defaults; handles both None values explicitly.

Decision-record audit: `src/bts/orchestrator.py:296–314` copies selected live streak/saver and contest provenance to selection results, not model diagnostics. `src/bts/daily_decision.py:62–109` and scheduler record paths likewise do not persist/coerce either model diagnostic; the live streak and saver fields retain their existing contracts. Dashboard state consumers at `src/bts/web.py:1089,1900` use the selected live streak, not model_streak. No serialization or decision-record change was needed.

## Commands and measured results

Every uv run used `UV_CACHE_DIR=/tmp/uv-cache`; `UV_OFFLINE=1` prevented dependency downloads. Gate/focused runs used `TZ=America/New_York`. Test discovery used:

```bash
rg -l 'contest_state|load_streak|load_saver_available' tests -g '*.py'
```

Sorted node-file arguments were passed to `uv run pytest ... -q`. Inventories are `CS1-EVIDENCE/baseline-files.txt` (26 files) and `CS1-EVIDENCE/after-files.txt` (27 files, adding the new regression file).

| Run | Measured result | Log |
|---|---|---|
| Before: discovered focused files | 626 passed, 8 skipped | `CS1-EVIDENCE/before-focused.log` |
| Before: full fast suite below, sandboxed | **9 failed, 4933 passed, 18 skipped, 6 deselected, 1 xfailed** | `CS1-EVIDENCE/before-fast.log` |
| Before: eight sandbox-sensitive failures, exact nodes, outside sandbox | 8 passed | `CS1-EVIDENCE/before-sandbox-recheck.log` |
| Before: audit boundary check with original test files temporarily restored | 1 passed | `CS1-EVIDENCE/baseline-boundary-pristine.log` |
| RED new regressions against original production code | 27 failed, 20 passed | `CS1-EVIDENCE/red.log` |
| RED changed existing expectations | 2 failed, 7 passed | `CS1-EVIDENCE/red-existing.log` |
| Isolated draft: preservation check caught an overbroad as_of freshness change | 1 failed, 4 passed; corrected by keeping the existing date parser and limiting validation to required automatic reads | `CS1-EVIDENCE/candidate-metadata-red.log` |
| Corrected isolated draft: new regressions plus best-trust file | 56 passed | `CS1-EVIDENCE/candidate-green.log` |
| **After: all freshly discovered focused files** | **673 passed, 8 skipped** | `CS1-EVIDENCE/after-focused.log` |
| **After: eight exact sandbox-sensitive nodes, outside sandbox** | **8 passed** | `CS1-EVIDENCE/after-sandbox-recheck.log` |
| After local commit: audit boundary check | 1 passed | `CS1-EVIDENCE/after-commit-boundary.log` |
| **After: full fast suite** | **NOT RUN BY ME — lead will run on another host** | No after-fast log; approval was declined before command execution |

The before-change fast command, and the outstanding command for the lead, is:

```bash
UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest -m "not slow" \
  --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q
```

The nine failing before-change test nodes were:

1. `tests/scripts/c2_framing/test_screen.py::test_the_boundary_admits_the_real_suite`
2. `tests/scripts/test_c1_cycle.py::test_the_guard_runs_the_job_in_the_payload_leaf_and_writes_a_terminal_receipt`
3. `tests/scripts/test_c1_cycle.py::test_the_guard_reports_an_overrun_after_killing_the_payload`
4. `tests/scripts/test_c1_cycle.py::test_a_nested_descendant_is_killed_before_the_measurement`
5. `tests/scripts/test_c1_cycle.py::test_an_unconfirmed_empty_payload_is_an_enforcement_failure`
6. `tests/test_sd_notify.py::test_sends_watchdog_message_when_socket_set`
7. `tests/test_sd_notify.py::test_sends_ready_message_when_socket_set`
8. `tests/test_sd_notify.py::test_non_encodable_message_does_not_raise`
9. `tests/test_web_saver.py::test_do_post_saver_transition_socket_wiring`

Nodes 2–9 were sandbox denials: process jobs could not start in preexec_fn; socket binds raised PermissionError. All eight passed outside the sandbox before and after the production change. They were rerun after the change conservatively because the dashboard/account-state area overlaps; no local socket/process implementation was changed.

Node 1 was affected by my procedure: I edited the test files while the already-collected baseline suite was still running. Production code stayed unchanged until that suite finished, but the repository boundary test inspects current file bytes against HEAD and rejected the modified best-trust file and untracked regression file. This means the raw baseline was not a wholly pristine-worktree run, and I do not claim it was fully green. The specific check passed with those two test files temporarily restored to the original state, and passed again after the requested local commit. The raw nine-failure result remains recorded above. No full baseline repeat was run.

`git diff --check` passed. After committing, tracked files were clean; only this handover and the evidence directory remained untracked.

## Artifact check and limits

The regression files are real on-disk artifacts from the production `build_observation` producer (after `validate_fetch`) and from the real `set-contest-streak` CLI. U1 first loads the valid produced automatic file, mutates its stored values and verifies the mutation, then checks the actual decision loader's refusal. F1 uses the produced manual/automatic observations beside each corrupt model representation, with before/after file snapshots and warning checks. This exercises actual producers, actual disk reads and the real decision object, rather than a transcribed classifier or mocked loader.

All account values are synthetic. These checks do not prove real MLB endpoint behavior at a season boundary or recovery on the real account. No production account artifacts were acquired. No 2026 outcome analysis was performed.

## Outstanding work and contract concerns

No table row remains unmatched and no contract correction is requested. Sequencing exceptions are intentional: manual stored-value bounds remain permissive, optional automatic reads retain undated acceptance, recorded_at is checked as a string rather than as a timestamp, dated v1 remains accepted, and `{}` retains scorer defaults.

The lead owns the after-change full fast-suite gate on the designated host and the fresh adversarial review before any push. There was no fixture-day dry-run mechanism invoked for this narrowly specified defensive-read change. The code is committed for those checks, not represented as having passed a full post-change gate or review.
