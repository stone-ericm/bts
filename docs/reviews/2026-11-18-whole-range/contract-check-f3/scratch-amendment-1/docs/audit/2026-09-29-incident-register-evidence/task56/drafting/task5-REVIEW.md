# Task 5 replay-spec review ledger (lead's decisions on the drafted specs)

## b1 (I-004, I-011, I-017, I-018, I-019, I-020)
- I-004 links 1-3, I-017 link 1: unavailable as drafted (the fix touches no test). Agreed.
- I-018 links 1-2, I-019 link 1, I-020 links 1-2: unavailable as drafted (fix outside src/: workflow or cron). Agreed.
- **I-011 link 1 (draft R-I011-1): ruled UNAVAILABLE, not run.** Two reasons, both from the drafter's own trace:
  1. at the parent the only assertion reached is `assert "Already resolved" in result.output`, a message the fix
     itself introduced; it fails on any pre-fix code whether or not the result is re-applied, so it does not witness
     the contract (the double-applied streak). The streak assertion (`load_streak(...) == 2`) is never reached, and
     would even pass at the parent without network;
  2. the red run would call the live MLB stats API (check_hit is not patched), so the run is neither isolated nor
     reproducible.
  Reason recorded: "the fix's test asserts the fix's new skip message; at the parent the streak contract is not
  asserted, and the red run would reach the live MLB API".

## b2 (I-023, I-025, I-027, I-028, I-029, I-032)
- R-I023-1 (I-023 link 1): accepted for running, with test_strategy.py::...::test_for_shadow_ignores_locked_production
  dropped from the selection (new-API-only at the parent; adds nothing).
- R-I023-2 (I-023 link 2), R-I029-1 (I-029 link 1): accepted for running as drafted. R-I029-1's anchors are the first
  line of multi-line asserts; if the runner places the failure elsewhere, the replay is rejected by its own anchor check
  (fail closed) and the anchor is corrected.
- I-025 link 1, I-032 links 1-2: unavailable, new API only (their tests import functions the fix added; no test in the
  fix set exercises the broken scheduler path). I-027 link 1, I-028 links 1-2: unavailable, fix outside src/. Agreed.

## b3 (I-034, I-035, I-040, I-042, I-043, I-047)
Rule applied consistently (from I-011): a symptom must witness the link's own mechanism through the contract's
consequence (state, grade, delivery decision), never only through an output or log message the fix introduced, and
never through a different change shipped in the same commit.
- I-034 link 1, I-035 link 1, I-040 link 2, I-042 link 2: unavailable, new API only. I-040 link 1: unavailable, no test
  in the fix. Agreed.
- **I-042 link 1: UNAVAILABLE.** The only node failing at the parent fails because the same commit raised the alert
  thresholds, not because of link 1's attribution defect; no test passes data_dir, so the attribution fix is never
  exercised.
- **I-043 link 2: UNAVAILABLE.** At the parent the first failing assertion is the fix's new "VOID: ..." output line; the
  streak assertions that would show the postponed slot graded as a hit are never reached. The polling-gate alternative
  witnesses mechanism 3, not 2.
- R-I043-1, R-I043-4, R-I043-5, R-I047-1: accepted for running as drafted (R-I043-5's extra selected node is a
  symptom candidate for the reviewer, not a declared symptom; R-I047-1's mock assertion helpers raise the builtin
  AssertionError and map to the test line).
- **R-I043-3: accepted as a PARTIAL replay** (the cap test only). The main overlay test is not selected: at the parent
  it loops on real 15-minute sleeps until 05:00 ET, and at the fix it fails between 05:00 and 10:00 ET. The record
  entry says the replay covers the cap rule only.

## b5 (plan-named: I-082, I-083, I-084, I-085, I-113)
- **R-I082-1 narrowed to link 1.** Link 2 (the cron's credential-shaped DM) is UNAVAILABLE, new API only: every test of
  the message imports TransientAuthError, added by the fix; the AuthError chain witnesses link 1's retry defect, not the
  message. The undeclared red test_retries_empty_200_then_succeeds (the literal 8/11 shape, raising the incident's
  AuthError) stays selected as a reviewer-visible symptom candidate.
- **R-I084-1 ACCEPTED** (drafted medium). Link 1's recorded mechanism is the 12:50 has_pending_future_window snapshot;
  the only assertion-level red (test_fallback_delivers_when_future_checks_have_no_pending_lineups, line 1540) defers on
  exactly that snapshot at the parent. The fix rewrote the test to stop forcing the parent's own predicate
  _has_pending_future_confirmation_window to False and to patch only check_confirmed_lineups (present at the parent):
  that is the fix's test carrying the contract, audited neutral, not an adapter.
- R-I083-1, R-I083-2, R-I084-2, R-I084-3: accepted as drafted. R-I084-3 imports bts.model.predict (LightGBM): run in a
  venv synced with --extra model. Optional extra symptoms left undeclared as drafted.
- I-084 link 4, I-085 link 1 (0abf503): unavailable, new API only. I-113: unavailable, the fix (947fce8, src/bts/data/
  sync.py) touches no test. Agreed.
- Note for the guide/plan: the "2ff2db9 return shape" lesson is a681e89's (3-tuple -> LockDecision); 2ff2db9's src diff
  only reorders run_single_check.

## b4 (I-056, I-061, I-063, I-071, I-072, I-075) — drafted, then verified by a second agent after the first hit an API limit
- Verification corrections accepted (consistent with the rules above): R-I071-2's output-wording symptom
  (test_dm_countdown_uses_submission_cutoff) undeclared, so the countdown half of I-071 link 2 is uncertified;
  R-I075-1 narrowed to link 1 (link 2 UNAVAILABLE: 9551818 only adds a WARNING log line and its test asserts that new
  message); R-I075-2 narrowed to link 4 (link 3 UNAVAILABLE, new API only: now_et_date= / incident_key= keywords).
- **Exit-code anchors accepted** (R-I063-2 lines 216/819, R-I071-1 line 1049): the exit code is the parent's own
  refusal/alert signal on these inputs, an observable consequence of the contract, not a fix-introduced message.
- **R-I072-1 accepted**: the static trace shows red by assertion with this selection; the plan's "8bceda1 new-API-only"
  lesson came from a selection that collected tests/health/test_pick_entry_source.py (imports the new module) — the
  plan's lesson text is corrected.
- R-I075-1: wall-clock dependent by design (date.today(), the parent's datetime.now(ET)); TZ fixed; anchor at the first
  line of a multi-line assert (fail closed if attributed elsewhere). Accepted.
- R-I063-4: the other failing lines (74, 150, 171) stay undeclared: their gap>=2 half reverses I-056 link 3's contract.
- **Pruned (new API only, never by assertion):** R-I063-2 test_build_observation_allows_none_source_date and
  test_fetch_cli_persists_current_activestreak_despite_lag; R-I056-3 test_future_source_date_is_critical; R-I072-1
  test_initial_near_cutoff_consumes_lower_tiers; R-I075-2 test_new_day_advances_anchor.
- For Task 8 (records): R-I061-2 certifies a contract later reversed by 56f9726 (R-I063-4); I-056 link 2's placement
  (its second symptom fits I-063); the drafts' deploy refs for 68a35d3 and 58c9adc need the first deploy containing them.
- All others accepted as drafted; I-056 link 1 and I-061 link 2 unavailable (new API only); I-061 link 1 and I-072
  link 1 not applicable (no fix).
