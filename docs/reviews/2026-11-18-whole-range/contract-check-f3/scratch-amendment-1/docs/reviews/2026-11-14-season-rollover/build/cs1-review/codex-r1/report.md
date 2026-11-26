# Verdict

**BLOCK** against the authoritative eight-row table, reviewing `main...7cff54d5`. One production behavior exceeds row 4. Rows 1–3 and 5–8 hold in the probes. No tracked file changed.

# Blocking findings

**B1 — Required reads reject recorded_at strings that the contract preserves.** `src/bts/contest_state.py:164–166`, specifically `_parse_dt(...) is None` at line 165, requires a parseable timestamp. Row 4 instead requires a `recorded_at` **string**; its rejection condition is absence of both a parseable source_date/as_of and that string.

Concrete input: `account_state/contest_streak.json` containing `{"active_streak":7,"best_streak":20,"recorded_at":""}`, healthy `streak.json` containing `{}`, and `require_contest_state=True`. `"not-a-time"` has the same result. Main returns streak 7, stale; HEAD raises `ContestStateError: automatic contest observation is undated`. With a synthetic eligible .99 prediction, real `orchestrator.run_and_pick` selects a single on main and refuses on HEAD. Real scheduler delivery on HEAD returns False, captures a contest-state failure alert, sends no pick DM, and creates no decision record. The optional dashboard still renders streak 7 as last confirmed. Health is CRITICAL for missing source_date on both versions.

Demonstrated in `probe.py`, `probe-results.json` → `production.empty_timestamp` / `production.malformed_timestamp`, and `test_row4_contract.py` (2 fail on HEAD, 2 pass against main's scratch package). Of 216 metadata combinations, HEAD rejects 27: 9 authorized missing-string cases and **18 additional string cases**.

Smallest fix: restore `not isinstance(data.get("recorded_at"), str)` at line 165 and preserve empty/malformed strings in the row-4 tests. Alternatively, revise the authoritative contract before signing the stricter behavior. Commit `7cff54d5` explicitly describes this as an intent correction, but neither supplied kickoff table nor review brief delegates changing that table to a commit message. A healthy fetch or an unexpired manual override followed by another selection provides operational recovery without a code change.

# Notes

**N1 — Actual producer compatibility.** Ran the real fetch CLI with synthetic auth/profile/round responses; no requests were sent. The unchanged writer (`contest_fetch.py:205–230`, `cli.py:2047,2075`) writes v1, source_date as ISO date or null, and a valid UTC recorded_at. Normal 8/9 and null-coverage 8/9 files both still select picks; null coverage remains stale with CRITICAL health and a last-confirmed dashboard. Git history shows the June 6 builder already wrote valid recorded_at; June 17 added nullable source_date. No rejected ordinary 2026 producer shape was found; no actual account snapshot was inspected.

The real writer also accepts/writes 58/58 because validate_fetch has no upper bound. Main selects on that file; HEAD refuses, health becomes CRITICAL, and the dashboard renders “Streak State Error” with the model fallback number. This is the deliberate row-3 change, not an extra blocker. The real set-contest-streak CLI followed by real selection recovered through an active manual override. A repeated unchanged 58/58 fetch would continue to be rejected.

**N2 — Date and freshness disagreement remains.** Tested each metadata field missing, empty, malformed, date-only, timestamp, and future, in all 216 combinations. Parseable recorded_at alone passes required reads but still yields source_date=None/stale and CRITICAL health. A malformed nonempty source_date plus valid as_of also passes the new dating guard while the returned source_date stays None; the dashboard says last confirmed, and real decision.json stores contest_source_date:null. Main already had this freshness/parser behavior; preserve it here and address it in the later change set. Future source_date remains accepted/fresh, best untrusted, health CRITICAL when sufficiently future; future recorded_at alone remains accepted/stale. No additional date rejection beyond B1 was found.

**N3 — Precedence, 28 disk cases, both required modes.** An active manual override returns 7 before every corrupt/invalid/undated automatic file, identically on both versions. Expired/legacy manual plus corrupt auto fails on both (UTF-8 exception becomes the row-2 domain error); plus invalid integer bounds changes auto acceptance to refusal under row 3; plus truly undated auto changes only required reads under row 4. Empty-string recorded_at adds the unauthorized refusal in B1. Optional undated reads remain accepted. Both corrupt files fail on the manual file first; no fallback is attempted. No other precedence change was found.

**N4 — Model/scorer isolation and caller audit.** `picks.py` is byte-unchanged. Direct helper reads plus 2,178 combinations of missing keys, null/bool/int/string/list/object values, three observation modes, and six saver-flag modes produced zero main/HEAD differences in model diagnostics or selected live state (ignoring the new healthy error=None field). Missing streak.json was also checked separately. Live saver still comes from saver_state.json even when only streak.json records a saver. Wrongly typed object values retain main's behavior; noninteger best values are still discarded as missing. Inventory: `callers-all.txt`.

| Production caller | Required contest state? | Change on previously accepted files |
|---|---|---|
| orchestrator.py:283,362,386,429 | Config helper: scheduler.contest_state_required OR health_checks.contest_state_expected; defaults False | Row 3 always; row 4 when enabled, including B1 |
| scheduler.py:1098 | Same config helper | Delivery blocks and alerts on those refusals |
| scheduler.py:1723 | Explicit False | Skip-summary fallback sees row 3; catches errors |
| cli.py:1161,1206,1247,1321 | Default False | Row 3 can now raise; row-4 acceptance retained |
| health/contest_state.py:45 | Direct contest loader, default undated acceptance | Row 3 becomes CRITICAL; row 4 still uses existing health rules |
| web.py:125,140 | Both default False | Row 3 becomes state error / saver-section omission; row-4 files still display |
| strategy, daily_decision, scripts | No direct calls to either contest loader in strategy/records/scripts | Strategy receives selected streak/saver; records copy selected state; scorer helpers unchanged |

The internal call at contest_state.py:340 forwards require_contest_state to require_dated_auto. No production caller passes a literal True directly; scheduler/orchestrator can pass True through configuration. The scheduler/private-delivery and no-predictions bypasses predate this change and belong to later work.

**N5 — Optional field readers.** The only external production reader of model_streak is web.py:238–240: its None guard prevents comparison/formatting of unavailable diagnostics. There is no external production reader of model_saver_available or model_state_error. Healthy assertions/constructors in test_contest_state.py and test_web_streak_subtitle.py remain valid; new defensive tests check both Nones. Arithmetic/policy selection, DM text, and daily_decision.py:62–109 use the selected live fields instead. With an actual 0xff model file, the real selector, full dashboard renderer, scheduler DM formatter/delivery branch, and decision writer succeed: selected streak 7, live saver True, DM “Streak: 7”, decision.json streak 7/state_source contest. Both diagnostics serialize as JSON null in the probe's dataclass serialization. No new None crash or false persisted state was found; model_state_error is logged, not persisted in decision.json.

**N6 — Nearby pre-existing hole.** A date-named object `{"result":[]}` or `{"result":{}}` still raises TypeError in `_resolved_pick_dates` at contest_state.py:234 on both versions; the result set-membership expression is unchanged. Non-object scalars are now skipped as row 1 requires. This is a follow-up NOTE, not a blocker for this diff. No unauthorized nearby change was found for rows 1–3 or 5–8.

# False greens and changed assertions

Ran every added/changed test against a complete scratch package whose contest_state.py comes from `git show main:src/bts/contest_state.py`: **33 failed, 26 passed**. F/P below are parameter-node counts on main; names are from test_contest_state_defensive_reads.py unless stated otherwise.

| Test | F/P on main | Meaning |
|---|---|---|
| test_corrupt_pick_is_skipped_by_both_scans | 2/2 | UTF-8/nonobject fail; JSON/OSError already skipped |
| test_corrupt_contest_file_raises_domain_error | 2/6 | Only UTF-8 normalization is new |
| test_manual_precedes_corrupt_auto | 0/4 | Existing precedence preservation |
| test_u1_bad_best_reachable_after_corruption | 3/0 | Real stored integer mutations fail old code |
| test_manual_bounds_unchanged_in_cs1 | 0/3 | Intentional manual preservation |
| test_legacy_fixture_exception_is_live | 2/0 | Required undated refusal is new |
| test_unparseable_recorded_at_does_not_date_an_observation | 4/0 | Two legitimate nonstring refusals; two assertions encode B1 |
| test_dated_version_one_auto_still_accepted | 0/4 | Preservation, including stale malformed-primary/as_of case |
| test_manual_not_universal_recovery | 8/0 | Both real observation modes and four corruptions |
| test_corrupt_model_cannot_be_fallback | 4/0 | New domain errors |
| test_missing_required_contest_wins_over_corrupt_model | 4/0 | New missing-state precedence |
| test_empty_model_object_keeps_existing_defaults | 1/0 | Fails ONLY on absent model_state_error attribute; old numeric/default assertions already pass |
| test_dashboard_handles_missing_model_diagnostics | 1/0 | Old model decode crashes |
| best_trust: test_auto_best_below_streak_is_rejected | 1/0 | Replaces old (3,untrusted) assertion under row 3 |
| best_trust: test_auto_best_above_target_is_rejected | 1/0 | Replaces old untrusted assertion under row 3 |

The other seven best-trust tests pass unchanged. No unrelated existing assertion was weakened/deleted. The two removed automatic-untrusted expectations are the intended row-3 change; added manual tests retain that behavior for manual files. Preservation tests passing on main are appropriate, but cannot establish the new behavior. Row 4 lacks the contract's string-acceptance direction: the lead removed the empty-string acceptance parameter and added rejection expectations instead. My two table-based regressions expose this gap. Rows 1/2/5 have actual UTF-8/non-object files; row 5 covers manual and automatic observations. My probes additionally use chmod(0) files and verify real PermissionError.

# Rows 1 to 8

1. **Holds.** `probe.py:rows`, keys row1-*: main UTF-8/nonobject raise; HEAD skips all five corruption forms in both scans and still reads the adjacent settled miss; JSON/OSError preservation confirmed.
2. **Holds.** row2-* for both filenames: main UTF-8 raises UnicodeDecodeError; HEAD ContestStateError names the file; array/JSON/read failures already domain errors; active-manual precedence survives.
3. **Holds for stored integer bounds.** row3-*: valid producer bytes are loaded then mutated to 13/10, 58/58, 13/58; main accepts/untrusts best, HEAD refuses with path/values; equivalent manual files remain accepted.
4. **Does not hold in full: B1.** dates grid and production.empty_timestamp/malformed_timestamp: missing/nonstring undated observations refuse only when required, dated v1 stays accepted, but empty/malformed strings are also newly refused against the table.
5. **Holds.** row5-manual/auto-* on five disk corruptions: main raises; HEAD equals the healthy decision except both diagnostics None and error text, with exactly one warning and unchanged file bytes/modes for readable snapshots.
6. **Holds.** row6-*: main read/type/decode errors; HEAD domain error names corrupt model path, with no fallback value.
7. **Holds.** row7-*: main model read errors; HEAD missing-required-contest domain error takes precedence.
8. **Holds.** row8-* plus model_matrix: {} and unrelated-key objects retain direct-helper (0,True,[]) defaults and decision diagnostics (0,True), with live saver controlled by its flag.

# What I ran

- Read full diff, contract/design references, producers, consumers, tests, and relevant 2026 writer history; searched all callers with rg. No network, box, account, or tracked writes.
- `probe.py` directly under `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python`: 55 row/nearby cases, 216 date combinations, 28 precedence cases, 2,178 model/flag combinations, missing-file cases, real fetch/manual CLI and selection/dashboard/delivery/record probes. Socket connects are forbidden; transports are captured. Results/log: `probe-results.json`, `probe.log`; disk cases and rendered HTML retained in `cases/`.
- Every pytest command used `~/bts-runs/yield_run.sh <log> env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider ... -q`. Commands are recorded in each `.log.watch`.
- Fresh rg inventory's 27 test files plus test_contest_fetch.py: **718 passed, 8 skipped** (`focused-escalated.log`). Initial sandbox attempt warned that nice was denied but also completed 718/8 (`focused.log`); repeated with the approved wrapper to meet the priority requirement.
- Nine further strategy/orchestrator/posting/decision/saver-control test files: **138 passed** (`caller-tests.log`). Unique repository tests across these runs: **856 passed, 8 skipped**. No whole fast suite run.
- Added/changed tests against main's isolated package: **33 failed, 26 passed**, expected negative control (`main-regressions.log`, XML). My two row-4 regressions: **2 failed on HEAD**, **2 passed on main** (`contract-head.log`, `contract-main.log`).
- `git diff --check` passed; tracked worktree remained clean.
