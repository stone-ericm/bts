# Task 6 current-defence spec review ledger (lead's decisions on the drafted specs)

Rule carried from Task 5: a certificate must witness the link's own defended decision; a fail-closed elapsed-time or
thread dependence that can only make a witness unattributed or unavailable is acceptable (plan ruling 13).

## d2 (I-020, I-023, I-025, I-027, I-028, I-029)
- I-020 links 1-2, I-027 link 1, I-028 links 1-2: unavailable, the defended decision is outside src/ (deploy.yml,
  cron, pyproject pin); no test covers them (git grep recorded in the manifest). Agreed.
- D-I023-1 (event): accepted. Its killing test joins a real heartbeat_watchdog thread with a 2 s timeout before the
  branch: a thread outliving the join makes the boundary call unattributed (fail closed), never a false witness.
  The incident's other branch (production's game started but its pick not posted) is not covered: noted.
- D-I023-2 (return): accepted; the bad category is value-specific to the fixture (a pick saved posted reads back
  unposted), as return witnesses are.
- **D-I025-1 (return): accepted at level component, tests narrowed to ::TestRefreshPickAtFallback (5 nodes).** Entry
  is the test-only compatibility wrapper _refresh_pick_at_fallback (the decision function's own return is incomplete to
  the observer); mutant = early `return FallbackRefreshResult(cached_daily, None)`, matching the helper's error path.
- D-I029-1 (return): accepted. The backward walk is now the D4 forward replay; without the 1d61908 guard the replay
  fails closed (returns None) rather than resetting the streak, so the certificate witnesses the current guard.
  Multi-line assert anchored at its first line (833): confirm on the first run.
- **Coverage findings (for the memo):** I-025: no current test drives run_day's fallback with an unmocked refresh whose
  fresh pick differs from the cached one; dropping `daily = refresh.daily` (scheduler.py ~2872/2995) would likely
  survive. I-029: the incident's own regression test (test_preview_pick_for_tomorrow_does_not_break_streak_walk) does
  not kill the guard's removal (it pre-saves the expected streak); only the direct replay test does.

## d1 (I-004, I-009, I-011, I-017, I-018, I-019)
- **D-I009-1 (return) accepted**, tests narrowed to ::TestSchedulerRun. Return kind, not absence: the witness is the
  wrong lock decision (should_lock false, block_reason gap) — a positive witness of the decision that ruling 10 permits
  (as I-0813-a); it certifies the decision, not "the pick never posted".
- **D-I011-1 (return) accepted**, the W1.5 fixture killing node dropped (the register's own fixtures cannot be its
  defence) and tests narrowed to the product node test_scoring_lock.py::test_check_results_skips_when_peer_scored_mid_flight
  (killed through the in-lock re-check); the mutant disables both resolved checks (the later re-check masks the first).
- I-004 links 1-3: no current defence (coverage gaps; link 3's cp1252 path cannot fail on this Mac/Linux; whether a
  Windows tier remains is in the box config, outside the repo). I-017: superseded (shadow_mode renamed, then folded into
  pick_delivery). I-018 links 1-2, I-019: outside src/. Agreed.
- **Coverage findings (memo):** I-011: no product test pins the 5e02676 pre-check by its streak effect (its tests use
  undelivered picks, stopped first by the GH #144 scoreable gate). I-004: no test exercises the stdout-purity fixes.
- **Hazard findings (memo, for Eric):** (a) no private-mode test patches bts.posting.post_to_bluesky — a mutant or bug in
  the delivery-mode decision would reach the real Bluesky transport with ambient credentials; never run such a mutant
  until a test patches it. (b) A config that still sets only the legacy `shadow_mode = true` is no longer read and would
  post publicly without warning (the box currently uses pick_delivery = "private", so no live exposure).

## d5 (I-072, I-075, I-113)
- **D-I075-1 (event, component) accepted.** The mutant restores the pre-9551818 step-7 decision but takes its clock from
  the fix's now_et parameter instead of the literal wall clock: the same decision, kept off the real date (ruling 13).
  The witness value carries duration_sec (elapsed) but the classify regex does not read it. The test uses date.today():
  the twins run minutes apart on the same day. Anchor at line 62 of a multi-line assert: confirm on the first run.
- **D-I075-4 (event) accepted with a note:** it binds the stdlib method pathlib:Path.write_text (a first). Every
  write_text call in the call phase is recorded; only the checkpoint-shaped bad category after the branch, inside a live
  entry invocation, can certify. Read the linked events closely in review.
- D-I075-3 (event), D-I075-5 (return via the nested compute_metrics.<locals>.gap_over, since compute_metrics' own
  return is incomplete to the observer at depth 3): accepted.
- I-072 link 1: not applicable (operator step). I-072 link 2, I-075 link 2: absence, unavailable (ruling 10;
  positive wrong-decision witnesses deferred as for I-0830-b). Agreed.
- **I-113 link 1: no current defence (coverage gap):** no test writes mdp_policy.npz through sync_to_r2. Recommendation
  for the owner's queued sync_to_r2 tail-artifact fix: its test can also assert models/mdp_policy.npz in the returned
  manifest, giving this link a return-kind defence.

## d3 (I-032, I-034, I-035, I-040, I-042, I-043)
- I-032 links 1-2, I-034, I-035, I-040 link 2: absence, unavailable (ruling 10). I-040 link 1: no current runtime
  defence (only a source-text scan). Agreed.
- **Coverage findings (memo):** I-032 L1: no test asserts the heartbeat_watchdog wrap in run_single_check or
  _refresh_pick_at_fallback_decision. I-034: the only guard is a static scan; no runtime run_day test checks pings.
  I-035: deleting the final _idle_until_next_wakeup call likely passes the suite (reasoned from greps, not run).
  I-040 L1: no test calls predict() or _build_feature_lookups.
- D-I042-1, D-I042-2 accepted as certifying the CURRENT analogues: link 1's mutant is in the slot_results grading path
  (added 2026-07-12; b08769d's own PA-frame path is unreachable by tests without slot_results); link 2's deploy filter
  is only the fallback at the pin (the regime-fingerprint filter takes precedence on stamped picks).
- D-I043-1, D-I043-2, D-I043-4 accepted; D-I043-1/2 narrowed to their killing nodes (test_scheduler.py's real-time
  tests and the overlay test stay out). Links 1 and 4 share one mutant (both decisions run through the void branch
  since 0799bf2): their certificates are not independent — say so in the records.
- **D-I043-3 accepted as PARTIAL (cap clause), narrowed to its killing node.** The link's main clause (the postponed
  overlay when the live feed still says Preview) has NO current defence that can fail: with the overlay removed its only
  test HANGS (real 900 s sleeps, cap_hour_et=10 so the cap never fires, no pytest timeout) — a coverage finding and a
  test hazard (memo).
- **D-I043-5 WITHDRAWN; I-043 link 5 unavailable**: the strict-mode guard it mutates is disabled on the production path
  (abbfdc5 runs with require_detailed_statuses=False). The production protection is the fail-closed branch in
  _lock_decision_from_predictions (pinned by test_lock_decision_fails_closed_when_detailed_status_unavailable), whose
  pre-fix restoration means adding a coarse-status fallback back — a candidate for a later spec, not certified now.

## d4 (I-047, I-056, I-061, I-063, I-071)
- I-047 link 1, I-056 link 1, I-061 link 2: absence, unavailable. I-061 link 1: not applicable (human step). I-061 link
  3: superseded (56f9726 removed the noon-ET WARN; the opposite behaviour is now pinned). I-063 link 3: unavailable (the
  only killing test asserts a datetime.date return, incomplete to the observer; no other observable boundary). Agreed.
- **D-I047-2 WITHDRAWN; I-047 link 2 unavailable** (consistent with the I-011 replay ruling): under its mutant the
  killing test makes an unmocked live MLB statsapi request (retry sleeps up to 15 s offline) and starts a real watchdog
  thread; its only attributable witness would be the dispatch decision, not the redone work.
- D-I056-2, D-I063-1 accepted: together they restore the two halves of 2b4ff1d's old stale branch (doubles freeze; max).
  D-I063-1 binds the dataclass-generated DecisionStreakState.__init__ through the class namespace (a first): if the
  observer cannot attribute it, the run reads unavailable (fail closed).
- **D-I056-3 accepted as the CURRENT analogue:** 56f9726 deliberately demoted the stale-observation alert from CRITICAL
  to WARN; the certificate shows only that the gap>=2 lag is still flagged.
- **D-I063-2 accepted as PARTIAL:** it certifies the gate's 2+ pick refusal; the incident's own case (a silent one-pick
  discard) is absence-shaped and uncertifiable.
- D-I063-4, D-I071-1 accepted; exit-code kills accepted (as in b4). D-I071-2 accepted: the pin's pass needs < 30 s
  elapsed inside the command (countdown "35"); the mutant's kill does not depend on it; a slow run fails closed.
- For the records/memo: the live check-pick-entered cron (the live defence for I-061/I-071) has been commented out on
  the box since 9/14 (season over; CLAUDE.md, not re-verified). Optional extra specs not drafted: I-063 L4 CRITICAL->WARN
  demotion; I-071 L2 window lower bound; I-063 L2 no-source_date case.
