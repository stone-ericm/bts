# Verdict

**BLOCK**, for the permitted corrupt-file repair described below. For valid production string dates, the implementation matches all twelve rows. This is a safety objection to the contract's treatment of corrupt state, not a claim that the author violated its season-match definition. Reviewed `d8eb50d0 → 3b4196be`; no tracked file changed.

# Blocking findings

**B1 — A dated scorer silently destroys evidence of corrupt current-season state and blesses an invented counter.** In `src/bts/picks.py:516–525`, unreadable/non-object JSON becomes `{}`; `update_streak` at lines 653–668 then writes fresh state. Concrete input: formerly tagged 2027, streak 12, saver false; `streak.json` now contains `{`, `0xff`, or `[]`; an older pick is unreadable; today's delivered pick resolves hit. Base: CLI exits 1 / polling raises, leaving the pick unresolved and corrupt bytes intact. HEAD: CLI exits 0 / both polling branches finish, saving streak **1**, saver **true**, today's marker and season 2027, and marking the pick hit, without an error or alert. Health returns `[]`; the 02:00 reconcile cannot replay the older pick and leaves that invented state indefinitely. Model-only fallback now trusts 1 with no model-state error; shadow selection also receives it. Live contest and saver-flag authority remain unchanged.

Smallest fix: validate an existing streak file as readable UTF-8 JSON **object** before a dated `update_streak` mutates anything; on failure raise and preserve both files. Continue resetting readable old-season/tagless objects, and allow a complete reconcile replay to rebuild corrupt state. **Contract amendment:** add this distinction to row 3; season mismatch alone cannot authorize destructive recovery from unreadable state. This is the separate contract objection, not an implementation-conformance finding. Demonstration: `corrupt_loss.py`, `head-corrupt-loss.json` versus `base-corrupt-loss.json`; all scorer branches also appear as `corrupt-*-*` in the main probe logs.

# Notes

- **N1, unchanged recovery code:** `state/export.py:60` omits the season; `state/regenerate.py:387–395` writes tagless state; `state/verify.py:65–82` compares only values. Real export → regenerate of tagged 13/false reports no drift, but HEAD model fallback reads the restored file as 0/true diagnostics, versus base 13/false. The next dated hit starts at 1 until a successful replay. Material recovery follow-up; row 12 deliberately leaves these modules unchanged. `probe.py`, event `row12`.
- **N2, unchanged readers:** the full `rg` census and every direct file read/write are in `call-sites.md` / `calls-rg.txt`. Only the replay's marker check (`picks.py:707`) intentionally omits a season in the scorer model API; a full date identifies its year and protects pp1 B1. Dateless API paths intentionally retain raw values. Dashboard fallback shows old 13 on its error path; validation accepts old valid state; export/verify/backup ignore season. These are not evidence of current-season freshness. No direct model-streak access or API call was found in `scripts`.
- **N3, API edge:** `update_streak(..., date=datetime.date(2027,3,24))` raises `TypeError` at `picks.py:650`, rather than the row-3 blanket `ValueError`. It also failed on base, later in JSON serialization. All three production scorer calls receive Click/scheduler ISO strings; none passes this object. Add an `isinstance(date, str)` guard if the stated exception guarantee includes non-string arguments. Malformed, impossible, basic-format and time-bearing strings raise `ValueError` before I/O as required.
- **N4, evidence environment:** the no-network guard deliberately blocked `test_web_saver.py::test_do_post_saver_transition_socket_wiring` at its localhost connection; this is not a product regression. The initial wrapper emitted a sandbox `nice()` warning while its tests passed; subsequent pytest jobs used the approved wrapper outside the sandbox. No external network, box or account access occurred.
- No other unexplained difference emerged from 2,203 paired dateless/current-season matrix cases after excluding intended tag preservation and timestamps (`matrix-differences.json` is empty). The first midseason dated write over a tagless deployment still resets by design; no migration was introduced.

# The corrupt-file question

Quiet repair is unsafe: a fresh zero is a season fact only for a readable mismatching object, not evidence about unreadable state. The former loud failure preserved the corrupt bytes, left the pick retryable, and kept the model diagnostic/health error observable. HEAD's replacement removes all three protections. A warning after overwriting would still lose recovery evidence; prefer the B1 preflight and explicit replay/quarantine recovery.

| Real path, corrupt `{` / non-UTF-8 / non-object | Base | HEAD |
| --- | --- | --- |
| CLI non-void scoring | exits 1, JSONDecodeError / UnicodeDecodeError / AttributeError | exits 0, fresh state overwrites bytes, pick terminal; no scorer warning |
| Scheduler final scoring | raises same errors, pick remains unresolved | returns final, fresh state overwrites bytes; no scorer warning |
| Scheduler early-hit scoring | raises same errors | applies fresh hit, saves terminal pick, finishes without corruption warning |
| CLI/scheduler all-void scoring | read raises | displays 0 and saves void pick, **does not overwrite streak**; health still sees corruption |
| Reconcile, incomplete past history | replay None, preserves corrupt bytes | same; after a scorer's quiet repair it preserves the invented valid state instead |
| Reconcile, complete history | rebuilds streak from real picks | same legitimate rebuild, additionally stamps season |

Before repair, bad JSON yields a validation CRITICAL; non-UTF-8/non-object make `streak_validation.check` raise, which the real health `_safe_run` converts to CRITICAL. After the non-void repair there is no validation alert or model diagnostic error. `corrupt_loss.py` runs that health wrapper directly. Full bytes, outputs, readers and nightly outcomes are retained; this objection is separately identified as a contract problem in B1.

# Changed tests

AST comparison found **848 unchanged assert statements** across the twelve existing files. `changed-tests.md` and `existing-test-audit.json` enumerate **each** changed test/helper, its original purpose, assertion counts and line. No assertion, expected-exception/oracle behavior, strict marker or HTTP/crash proof was removed or loosened.

- `tests/test_check_results_wait.py`: 11 fixtures tagged 2026; pending/shadow/error/stale/deadline/vanished-pick subjects unchanged.
- `tests/test_cli_integration.py`: 19 fixtures tagged 2026; live contest precedence, hit/miss/void, shadow and commitment guards unchanged.
- `tests/test_contest_state.py`: 11 JSON fixtures tagged 2026 to retain nonzero current-season model diagnostics; observation precedence and saver authority assertions unchanged.
- `tests/test_incident_register_2026.py`: exactly 12 save-call keyword additions, all 2026; removing them reproduces the base module AST. Line-by-line audit preserves all 46 assertions, L01 grades/HTTP proof, L02 two-replay checks, L04 death/restart oracles, and E77/L03. Frozen audit scripts unchanged.
- `tests/test_pickpath_review_pp1.py`: crash/reconcile/restart fixture tagged 2026; same-day applied-marker protection still exercised; new suite separately pins the tagless case.
- `tests/test_pickpath_review_pp2.py`: delivered hit-plus-void fixture tagged 2026; one-hit increment assertions unchanged.
- `tests/test_picks.py`: five reconcile fixtures tagged 2026; same correction, preview, partial-void and shadow-exclusion subjects.
- `tests/test_reconcile_cutoff.py`: helper tags from its supplied day's year; cutoff/clock/lock assertions unchanged.
- `tests/test_reconcile_receipt.py`: helper tags 2026, matching every actual caller; receipt/source/correction assertions unchanged.
- `tests/test_scheduler.py`: two void-leg polling fixtures tagged 2026; active-leg scoring still verified.
- `tests/test_scoring_lock.py`: four raw fixtures tagged 2026; peer recheck, vanished-file, fresh-metadata and locked-reconcile proofs unchanged.
- `tests/test_streak_applied_dates.py`: six tests adjusted. The former tagless-carry-over test intentionally becomes a tagged **missing-marker** test; new row-3 test independently asserts tagless 3→1. Bounded-list loop replaces invalid April 31 with real consecutive dates; still 35 updates / 30 markers. All 14 assertions, including dateless idempotency coverage, remain intact.

# False greens

New file against scratch **base src**: **63 failed, 8 passed** (`false-greens-base.log`). All eight passes are legitimate compatibility controls, not evidence that season handling already worked:

- `test_row1_dateless_corruption_behavior_is_unchanged`: three raw-byte/error parameters; invokes only legacy loaders.
- `test_row4_tagless_dateless_update_keeps_legacy_state`: unchanged legacy update.
- `test_row5_tagless_same_day_marker_still_blocks_incomplete_replay`: guard existed before CS2.
- `test_row10_model_diagnostics_reset_without_changing_contest_or_saver_authority`: two current-2027/expected-13 parameters, with and without contest; base ignores the tag but returns the same values.
- `test_row12_tagged_file_still_validates_and_exports`: intentionally unchanged consumers already ignored extra keys.

# Rows 1 to 12

Each line has real base/HEAD outcomes; fuller adversarial/nearby inputs and absence of unsupported counterexamples are in `row-evidence.md`. The corruption objection is B1; the only exception-type discrepancy found is N3 outside the annotated string API.

1. **Holds:** `row1-*`, `tags-*`; base legacy 13/false/old markers, HEAD scoped 0/true/[] for absent/old/bool/string/float/null tags, exact int current tag unchanged; reads preserve bytes and default-None errors.
2. **Holds:** `row2-*`; old-file save 4 preserves false/old markers on base, resets to true/new marker and stamps 2027 here; dateless saves preserve tags verbatim here, including null; explicit-saver corrupt-file legacy tolerance unchanged.
3. **Holds for production strings:** `seq-*`, `date-*`, `midseason-*`; old hit/miss/double 14/0/15 becomes 1/0/2 with fresh saver and one marker; retries unchanged; malformed strings fail before I/O. N3 documents typed-object error; B1 objects to permitted corruption repair.
4. **Holds:** `dateless-between-dated`, differential matrix; 6→7→8 on both, tag retained here; tagless dateless 13→14 unchanged. No production dateless updater.
5. **Holds:** `seq-*-0200`, `tagless-pp1`, `crash-*-reconcile`; replay reconstructs season and resets old markers here, preserves them on base; None replay preserves exact bytes on both, including tagless pp1 B1.
6. **Holds:** real CLI/polling in every snapshot; stale/tagless void display 13→0, original bytes unchanged. New-suite final-peer/early-peer tests cover both other reads; real race/crash sequences count once.
7. **Holds:** `utc-0/1`; real command/replay at 2028-01-01 01:00 UTC still reads ET 2027 and displays 13; at 06:00 UTC replay stamps 2028 and displays 0. Both summary branches additionally tested.
8. **Holds:** real shadow entry in `tags-*`/every snapshot; selection receives old/tagless 13→0, current 13→13, strict impostor rejection; no read rewrites.
9. **Holds:** real strategy selection in each snapshot; old/tagless saver false→true using selection-date year; explicit saver controls intact (`test_strategy.py`).
10. **Holds:** real diagnostics with and without contest in every snapshot; stale model 13→0, contest remains 7 and saver flag remains authoritative; corrupt-file warning/None/fallback error retained until scoring erases it (B1).
11. **Holds:** literal legacy bytes, `tags-0`, opening sequences; fresh reads leave bytes intact; first dated result stamps year; void leaves old bytes until successful replay. Known tagless midseason reset retained.
12. **Holds:** `row12`, real validation/export/regeneration/verify/backup/dashboard calls; tagged validation silent and export succeeds on both. Six modules unchanged; semantic restore gap is N1, not an extra-key parsing failure.

# What I ran

- Every pytest invocation used `~/bts-runs/yield_run.sh`, `TZ=America/New_York`, `PYTHONDONTWRITEBYTECODE=1`, `.venv/bin/python -m pytest -p no:cacheprovider … -q`; logs and `.watch` command records are retained beside this report.
- `targeted-head.log`: **662 passed, 8 skipped** (23 files, including new tests, all twelve changed existing files, CS1 defensive/corrupt tests and direct state/backup consumers).
- `original-base.log`: **591 passed, 8 skipped**, same existing files using their original tests and scratch base package, never a tracked-tree source swap.
- `false-greens-base.log`: **63 failed, 8 passed**, exact new file against scratch base package.
- `remaining-callers-head.log`: **301 passed, 1 failed**; includes strategy and every remaining `rg`-identified test file. Sole failure is the deliberately denied localhost socket test (N4). No whole-fast-suite run.
- `probe.py` against both real packages: **169 recorded events each**; eight one-directory season sequences, three injected write-gap deaths/restarts, a barrier-controlled overlapping scorer race, winter/read snapshots, saver zone 9–16, dateless/dates/replay/UTC edges, all requested tag types, corruption across scorer branches and six unchanged direct consumers. `sequence-evidence.md` indexes state progression; JSONL stores exact bytes and row-6..10 outputs at each step.
- `differential.py`: **2,203 paired cases**, zero unexplained differences; `corrupt_loss.py`: three concrete loss/health/nightly cases on each commit. `audit_tests.py` plus manual complete diff/incident-oracle review; `rg` call census, `git diff --check`, and clean tracked-file status. Scripts, disk fixtures and logs retained in this directory; Python socket guard forbids network connections.
