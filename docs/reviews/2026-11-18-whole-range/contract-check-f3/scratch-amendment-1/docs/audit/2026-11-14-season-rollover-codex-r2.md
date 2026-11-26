# Verdict

**BLOCK.** The original JSON-shape cases and scheduler corrections pass, but the new season/saver tables make unsafe inferences from missing local history. Two further production regressions remain: invalid UTF-8 history crashes decisions, and health falsely alarms on an expressly permitted manual override.

# Round-1 findings

- **(a), original B1: closed for the reported JSON shapes.** Forty-eight combinations of old/current dates, six shapes, observation presence and required/optional mode pass. The real preview CLI now succeeds with the historical `[]`. Encoding corruption remains B3 below.
- **(b): season selection fixed.** Decision, health and dashboard read the current New York year's saver flag; fetch promotion is bound to its captured fetch year and requires a current-year round. Positive/negative fetch cases passed through `CliRunner`.
- **(c): not-earned reserve semantics fixed.** The tracked policy receives true below 10 for `not_earned`; its streak-8 action changes as required. The uninitialized case is independently unsafe: B2.
- **(d): implementation matches the supplied row tests, but the table does not establish account-season provenance.** B1 below names two counterexamples.
- **(e), (f): closed.** An unlocked extra cascade delivers from its own decision; early `continue` no longer consults a previous result. My 17-minute cascade delivered at 18:52 without a second cascade. A later-game selection still waited/refreshed and delivered at 19:52. Planned-check behavior remained unchanged, including its inherited double-refresh refusal.
- **(g), (h): closed in the tested states.** Void starts get no extra cascade or alert; status failure fails open. Tests drive both alert forms across midnight/restart and then tomorrow's slate through the real dispatcher. The old alert also had the wall-clock problem at the base.
- **(i), (j): closed.** The day-end wording remains true after archival/refusal; the 8/30 incident test passes with all original protections. The skip fixture now supplies a current contest observation, freezes its clock and asserts contest provenance.

# Blocking findings

## B1 — Local history is not proof that the account's season has or has not started

- **Locations:** `src/bts/contest_state.py:144-149`, `:340`, `:382`; the shared health classification at `src/bts/health/contest_state.py:67-84` inherits the error.
- **Concrete regression:** on 2027-06-01, restore the account-state files without root-level pick history. The observation proves a current round dated 2027-05-31, active streak 13, best 20; the current-year saver flag is active. The decision correctly reads 13. Then run the real fetch CLI with the same counter values but an incomplete rounds mapping, so `derive_source_date` returns None. The new observation overwrites the previous round witness. HEAD now returns streak **0**, `season_start`, satisfies `require_contest_state=True`, and health returns **no alerts**. The exact base loader still returns 13/stale. With the shipped policy, two candidates at .75/.74 produce **double at HEAD instead of skip at the known streak 13**.
- **The opposite transition is also unsafe:** use the brief's un-reset last-season profile, active/best 7, with old predictions absent from the new rounds mapping. Opening Night correctly uses 0; its single is entered and hits. After that local settlement, a fresh fetch of the unchanged profile is row 3 and suddenly uses last year's **7**. At .75/.74 the policy skips instead of the streak-1 double. A seven-day synthetic trace following those decisions remains at one local settlement, keeps skipping and never crosses health's three-pick threshold. This is a constructed upstream-state trace, not a claim about measured MLB settlement delays.
- **Smallest corrective change:** revise rows 2/3 and preserve explicit current-season contest provenance across coverage-only fetch failures. Do not use absence of local files to erase a previously proven season, or presence of one local result to promote an un-reset profile counter to current. Health must report unresolved provenance rather than declaring the restored mid-season state a clean season start. This needs a contract correction, not different assertions around the present table.
- **Demonstrated:** `scratch/probe_r2_state.py`; `r2-state-results.json` keys `midseason_mapping_failure_restore`, `unreset_first_week`, `correct_day_two_action_at_streak_one`.

## B2 — A missing saver flag becomes an available saver during an established season

- **Location:** `src/bts/contest_state.py:162-163`.
- **Concrete state:** 2027-06-01, healthy current-season round dated 2027-05-31, active streak 13/best 20, no retained local picks and no saver flag. This account can already have earned and used its saver; losing local files does not undo that. The base returns saver=false. HEAD returns **true** solely because no local result is retained. This contradicts the fetch hook's deliberate refusal to initialize an unknown saver as active at best >=10.
- **Wrong production outcome:** at probability .80 the actual tracked policy chooses **single with HEAD's invented reserve, versus skip with used/unknown conservative state**. `transition_outcomes` then models a miss as retaining streak 13, although a used saver resets it to 0. Adding one unrelated locally settled void flips the decision saver back to false without changing the account. Dashboard says uninitialized and health says the policy is conservative, while HEAD runs the optimistic branch.
- **Smallest fix:** keep uninitialized conservative whenever the contest season has already started; grant the new-season reserve only from established season-start evidence, not missing local files. Revise the corresponding S7 row. Retain the correct `not_earned` behavior.
- **Demonstrated:** `current_round_no_local_history` in `scratch/r2-state-results.json`, including real policy and transition outcomes.

## B3 — Invalid UTF-8 in last year's pick still crash-loops every decision

- **Location:** new unconditional call at `src/bts/contest_state.py:340`, reaching the incomplete exception handling at `:248-249`.
- **Concrete state:** `data/picks/2026-09-26.json` contains the byte `0xff`. On 2027-03-24 HEAD raises `UnicodeDecodeError`, with no observation or with a healthy required first-fetch observation. The base returns model-only streak 0. Refreshing the account cannot repair the unrelated historical file; the exception is not caught as `ContestStateError`.
- **CLI proof:** real preview, build/features/training/selection, synthetic HTTP: historical `[]` now exits 0; changing only that historical file to `0xff` exits 1; substituting the exact base loader makes the same CLI succeed again.
- **Smallest fix:** skip decoding failures as well as JSON syntax failures in `_resolved_pick_dates` and `_has_unconfirmed_miss` (`UnicodeError`/`UnicodeDecodeError`); add the byte-level loader and CLI regression. This is an integration regression from the newly unconditional scan, despite the older helper's pre-existing omission.
- **Demonstrated:** `non_utf8_old_pick`, `non_utf8_required_first_fetch`, `preview_cli` in `scratch/r2-state-results.json`.

## B4 — Health applies the automatic-file season rejection to an unexpired manual override

- **Location:** `src/bts/health/contest_state.py:67-75`; compare the explicit override exemption at `src/bts/contest_state.py:347-353`.
- **Concrete state:** now 2027-03-24; manual active streak 9, source/recording in 2026, expiration 2027-03-25; healthy auto file also present. These dates are explicitly permitted for an unexpired override. The decision correctly uses 9, but health emits a **CRITICAL to fetch current-season state**. Base health returns no alerts. Another successful auto fetch cannot clear the alert because the unexpired manual file still wins.
- **Smallest fix:** apply the same unexpired-manual exception to the new season classification in health, while retaining the base's independent missing/future-date and coverage checks. Add a health assertion beside the manual-override decision tests.
- **Demonstrated:** `unexpired_manual_health` in `scratch/r2-state-results.json`, comparing the actual base health function with HEAD.

# Notes

- No additional scheduler BLOCK demonstrated. Eight independent scenarios over the real day loop cover due/later-deadline reuse, unchanged planned behavior, a postponed game becoming playable, a same-pair doubleheader, voiding during an attempt, status-read failure and early continuation without a previous result. Only HTTP, clocks and DM outcomes are faked; provenance reads remain real.
- Reuse's extra-attempt-only limit is sound for the promised scope: enabling it for planned checks changes an existing refusal into a delivery. That older double-refresh loss is unchanged code and a follow-up NOTE, as instructed.
- Cost: at most one added check per remaining distinct eligible start. A later own deadline can add an existing refresh and a retained pick after delivery failure can add the final fallback refresh: K+2 is a conservative bound per uninterrupted run. It is not a restart-independent daily cap.
- Saver terminology is intentionally different between policy reserve and the dashboard's physical flag. DM text has no saver field; selection/decision.json carry the policy input. That distinction is valid for `not_earned`; it does not justify B2's optimistic uninitialized state.
- S3–S5 source is byte-identical to round 1; their focused tests remain green. The raw model scorer/shadow rollover limitation from round 1 is unchanged and remains a follow-up NOTE.

# False greens and changed assertions

- The seven-row tests encode B1's unsupported season inference and have no transition from a proven mid-season observation to a coverage-only failure after a partial restore. `test_s7_policy_reserve_table` explicitly expects true for an uninitialized saver at streak 12 with a proven current round and no local history: it encodes B2 rather than detecting it.
- The corruption cases cover decoded JSON shapes, not decoding failures. The health table's `manual=True` means **expired** manual; unexpired-manual tests assert only the decision, leaving B4 green.
- `tests/test_incident_2026_08_30.py`: no protection was lost. No pick DM, CRITICAL late-delivery notice, archive, unlocked state, Steven Kwan and stderr assertions remain. Searching all dispatches is correct; exactly one later day-end alert and its wording are additionally asserted.
- Cold-start assertions are stronger: exact pre-pitch feed-time set, separate polling time, exact additional check. Best-trust coverage is restored through the real decision loader. CLI clock pinning retains the actual loader. Saver expectation changes match S7, but table fidelity alone does not establish their safety (B2).

# What I ran

- Reviewed the full `12dbaaa..HEAD` diff and `b3b1e3c..HEAD` fixes, all superseding contracts and caller code. No network, other checkouts, shared-memory reads, commits, pushes or tracked-file edits. All 2027 states/outcomes in probes are synthetic.
- State/policy/UI focused run: `TZ=America/New_York .venv/bin/python -m pytest tests/test_season_rollover_v3.py tests/test_season_rollover_state.py tests/test_season_rollover_data.py tests/test_contest_state.py tests/test_contest_state_best_trust.py tests/test_contest_state_corrupt_precedence.py tests/health/test_contest_state.py tests/test_decision_saver_fallback.py tests/test_saver_state.py tests/test_tail_policy_r3_fixes.py tests/test_strategy.py tests/test_cli_integration.py tests/test_web_saver.py -q -p no:cacheprovider --deselect=tests/test_web_saver.py::test_do_post_saver_transition_socket_wiring` — **499 passed, 1 deselected**. The socket-wiring test was not run; pure dashboard cases passed.
- Data focused run: `TZ=America/New_York .venv/bin/python -m pytest tests/data/test_pull.py tests/data/test_build.py tests/test_data_build_empty.py -q -p no:cacheprovider` — **145 passed**.
- Scheduler discovery run, same pytest flags, paths in `scratch/r2-scheduler-test-files.txt`, JUnit `scratch/r2-scheduler.xml`, log `scratch/r2-scheduler-tests.log` — **571 passed, 8 skipped**. Supplementary seven paths from B3's broader literal discovery: `tests/health/test_analytics_artifacts_missing.py tests/health/test_runner.py tests/health/test_unit_drift.py tests/scripts/incident_register/test_deploy_runs.py tests/scripts/incident_register/test_records.py tests/scripts/season_ledger/test_acquire.py tests/test_scheduler_watchdog_discipline.py` — **63 passed, 1 skipped**. Combined **634 passed, 9 skipped** across the 41 listed paths.
- `TZ=America/New_York .venv/bin/python .codex-review/sr-review/scratch/probe_r2_state.py` — 48 original-shape cases; normal, failed-fetch and un-reset first-week traces; real fetch/preview CLI; mid-season restore/mapping failure; manual health; saver/MDP counterexample; 35 differing action cells across streaks 0–15 on three dates using both shipped artifacts. Results/log: `scratch/r2-state-results.json`, `scratch/r2-state-run.log`.
- `TZ=America/New_York .venv/bin/python .codex-review/sr-review/scratch/probe_r2_scheduler.py` — eight real-day scenarios; results/log: `scratch/r2-scheduler-results.json`, `scratch/r2-scheduler-run.log`.
- `git diff --check`, `git diff --exit-code` — clean tracked tree. Whole suite not run.
