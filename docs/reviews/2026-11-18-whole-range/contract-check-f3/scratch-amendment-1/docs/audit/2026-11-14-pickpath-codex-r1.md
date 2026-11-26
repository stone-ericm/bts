# Verdict

**BLOCK** — tier A, round 1. Reviewed HEAD `46e4e9e48edad260a8790f5d18d9bde3075b052b`.
Two production interactions need repair. Existing focused tests pass; the counterexamples below execute the affected production functions.

# Blocking findings

## B1 — Replay can remove an applied result while preserving the marker that prevents restoring it

**Location:** `src/bts/picks.py:610–611`, `661–662`, and the replay/save at `1317–1320`.

**Concrete sequence:** complete prior history gives streak 1. Today's scorer applies a hit, atomically saving streak 2 and today's `applied_dates` marker, then dies before saving the terminal pick. A same-day `bts reconcile` runs before the scorer retries. Replay excludes today's unresolved pick and saves streak 1, preserving the marker. The restarted scorer resolves the hit again, sees the marker, skips its streak update, and saves the pick as terminal `hit`.

**Wrong outcome:** a resolved hit with streak 1 instead of 2. The two scorers now consider the date settled; a later successful season replay is needed to repair it. This is a new interaction with the marker: before it, the retry restored the removed increment. R9 names manual result clearing, not replay withdrawing a crash-interrupted update while retaining its marker. This probe uses a same-day reconcile; I am not claiming an ordinary next-morning replay skips an unresolved yesterday.

**Verified by execution:** real `check-results`, injected death at `save_pick`, real `reconcile_results`, real `check-results` retry. Observed `(streak=2, unresolved, marked)` → `(streak=1, unresolved, marked)` → `(streak=1, hit, marked)`.

**Smallest fix:** when today's unresolved pick has an applied-date marker, refuse the replay (`None`) rather than skipping that pick and overwriting the streak. Preserve the applied state until scoring finishes. Add this interleaving to the crash-recovery tests; do not merely clear the marker while retaining the applied increment.

## B2 — A postponed DD leg discards the playable primary after the last opportunity to re-pick

**Location:** `src/bts/scheduler.py:1073–1076`, `2283–2286`, and the final-fallback call at `3176–3177`.

**Concrete slate:** two distinct games start at 19:10. Early checks select a cached double with a playable primary and a second leg subsequently postponed at 18:20. Later prediction cascades, including the 18:35 fallback refresh, return no predictions. All scheduled checks have finished. The new guard sees the postponed second leg, archives and deletes the entire daily pick, and returns false. `run_day` proceeds to tomorrow's wakeup without another selection attempt.

**Wrong outcome:** the healthy primary is never delivered although its cutoff is 19:05. The previous cached delivery retained a scoreable active leg; the new behavior loses that opportunity along with the void leg. The alert's promise that the scheduler "will re-pick from other games if any remain" is false at this final-fallback site. R10 acknowledges delivering a postponed pick during an API outage; it does not acknowledge discarding an independently playable selection when statuses are readable.

**Verified by execution:** adapted the existing L03 harness to two distinct matchups and a cached DD, retaining real `run_day`, `run_single_check`, fallback refresh and delivery guard. At 18:35 the day ends with no live pick file and no pick DM. A control bypassing only `_void_committed_games` delivers the cached pair at 18:35. No real account or DM transport was used.

**Smallest fix:** before removing the candidate, retain the unaffected slot as a single when exactly one leg is void, and deliver it through the normal cutoff/commit path with matching single-selection metadata. Alternatively, perform an immediate replacement selection with a usable cached-leg fallback; leaving `pick_locked=False` alone supplies no next cycle here.

# Notes

- **N1 — New status-request latency bypasses the private cutoff contract (verified).** `src/bts/scheduler.py:1073–1089`: enter the chokepoint at 13:34:59 for a 13:35 cutoff; the status request takes two seconds and reports Scheduled. Private mode locks at 13:35:01 and writes `scoreable=True, delivery_status=private_locked`. DM/public have a later clock check; private does not. Re-check the clock after the status read before the private write. This is a research-mode timing/record defect, not evidence of a late real-account DM.
- **N2 — A postponement refusal is subsequently mislabeled as a late refusal (reading).** The new refusal records share `delivery_refusals` but omit `late_min` and `cutoff_et`. `src/bts/health/late_delivery.py:50–55` renders them as a CRITICAL saying the batter was "None min past the None cutoff", even if refused well before cutoff. Give refusal records a reason and render the appropriate message.
- I did not challenge R1–R4 or R7–R10's stated tradeoffs: approximate dates for unknown years, fixed thin-slate offsets, repeated stateless alerts, unobserved entries remaining unverifiable, empty-feed batting lines, manual result clearing, and unreadable schedules delivering unverified.

# False greens

These are test-coverage notes, not additional BLOCKs.

- `tests/test_streak_applied_dates.py::test_other_writers_of_the_streak_keep_the_marker` saves the same streak value before retrying. It stays green when another writer actually rolls the value back while preserving the marker (B1). The L04 crash tests also retry immediately, so all pass without exercising the intervening replay.
- `tests/test_postponed_delivery_guard.py::test_a_double_down_leg_on_a_postponed_game_refuses_the_whole_pick` asserts refusal and no DM, but never continues the day or checks preservation of the healthy leg. It is green for the lost-day behavior in B2. Its narrow refusal assertion is true; it does not prove day recovery.
- `tests/test_thin_slate_checks.py::test_a_selected_pick_is_delivered_at_the_deadline_when_the_later_checks_produce_nothing` uses `_deliver_spy`, so its green result alone does not establish that the real guard delivers at that deadline. The separate guard tests cover ordinary statuses, but not recovery after a final-fallback refusal.

# What I checked

- Read each requested `git show`: `29e10bf`, `1137cbe`, `9998bbf`, `6467cfc`, `222d18e`, `48f6f01`, `327beae`, `4199360`, `b205ae0`; read the requested CLAUDE.md sections, work-policy note, roadmap/rulings ledger and pinned official §6 text. Consulted project/scoring topic memory, not personal memory files or conversation logs.
- Traced `grade_pick_in_feed` through `check_hit`/fallback, slot resolution, effective results, active streak booleans, daemon early/final scoring, shadow cached/live grading, reconcile receipts and season replay. No additional grading BLOCK established; hit+Pass increments by one and all-Pass holds under the focused fixtures.
- Traced thin-slate planning, in-loop and post-loop fallback, standing skips, DH rechecks, next-day wakeup, delivery refusal and commit finalization. Checked production season-end callers; remaining single-date imports are the documented audit scripts.
- **225 passed:** `tests/test_season_end_date.py`, `test_thin_slate_checks.py`, `test_day_outcome.py`, `test_streak_applied_dates.py`, `test_postponed_delivery_guard.py`, `test_incident_register_2026.py`, `test_check_hit_suspension.py`, `test_reconcile_receipt.py`, `test_entry_intent.py`.
- **123 passed:** `tests/test_picks.py`, `tests/test_entry_receipt.py`, plus two temporary producer-to-consumer integration tests. Each integration drove the real entry-receipt producer for all 56 quarter-hour runs on a synthetic delivered day, then real `day_outcome.gather`: both single and DD returned clean `entered`, including `already_confirmed` receipts. Off-day, policy-skip, missing-evidence and research cases passed in `test_day_outcome.py`.
- **Counterexample probes: 3 failed, 1 passed**, as expected: B1, B2 and N1 fail their required-outcome assertions; B2's guard-bypass control passes. Reproduce with `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest /tmp/test_pickpath_pp1_probes.py -q -s -p no:cacheprovider --tb=short`. Producer integration source: `/tmp/test_pickpath_pp1_receipts.py`.
- Tests used fixture feeds, clocks, files and mocked network/DM leaves. No whole suite, deployment, account access, commit or push. Neither prohibited results directory was read. Repository source/tests were unchanged; this report is the only intended repository write.
