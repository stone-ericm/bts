**Verdict: NOT YET.** Keep R24's daily observation / operator-enabled contract frame. Fix the rollout, decision-table contradictions, and recovery contract below before commissioning CS4–CS6. These are specification findings; no implementation change is proposed by this report.

Reviewed `main` at `2fa1eb3e57d4cfbae3b0d3ad6368623668dcbe30`, R24, the earlier attack, `design-v2.md`, relevant writers/readers/callers/tests, the activation checklist, FRESH-START and the local `sr-lastcheck` diff. Shared context consulted: `project_bts.md` and `project_bts_streak_saver.md`; their older claims are subordinate to current code and R24.
Ran the seven focused contest-state/fetch/saver test files: **105 passed**. Exact invocation: `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/test_contest_state.py tests/test_contest_state_best_trust.py tests/test_contest_state_corrupt_precedence.py tests/health/test_contest_state.py tests/test_saver_state.py tests/test_decision_saver_fallback.py tests/test_contest_fetch.py -q`.
Ran `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .codex-review/srd4-scratch/probe.py`; assertions passed and results are in `results.json` beside it. Real CLI, writer, saver hook, alert dispatcher and scheduler-state writes were exercised with synthetic inputs and blocked network/real DM transports. The decision-table and deadline probes are explicitly **specification transcriptions/arithmetic**, not an implementation of v2 or evidence about MLB's March behavior. No network, box, account, or 2026 outcome analysis; no tracked file changed.

## Findings, most severe first

### C1 — The advertised independent change sets reject their writers before replacements exist
**State:** merge CS1 on this `main`. `contest_fetch.build_observation` and `fetch-contest-streak` still write `bts_contest_streak_auto_v1`; CS1 expressly rejects it. A normal successful fetch cannot restore automatic selection. CS4 introduces its v2 writer; the new manual recovery writer arrives only in CS6. CS3 also introduces the sole Section 2 classifier, which requires contracts and manual v3 before either writer exists.
**Specification:** lines 139–169 call these independent, ready change sets, while attaching final-state reader rules to earlier sets. CS4's fetch and `enable` depend on CS5's continuity representation; the all-files atomic-write rule also changes the main manual writer before its allocated set.
**Required edit:** give each set an explicit compatibility contract. Keep hardening/schema bounds separate from retiring v1; introduce compatible record types and writers before activating strict readers, or combine the switch and manual recovery into one coherent set. Do not migrate v1 into a verified seasonal observation by relabeling it.
**Demonstrated:** real main writer produced auto v1 and real manual CLI produced manual v2. Intermediate rejection follows directly from CS1/CS3, not from running nonexistent implementations. Expected test conflicts are listed below.

### C2 — `retry-pick` has no complete result contract and its deadline spends the entire entry window
**Sequence:** singleton 19:10 ET game; operator follows the prescribed cold latest start, 18:47. A cascade taking the specified 18-minute budget returns at 19:05. `_deliver_and_lock_pick` refuses at that instant. Cached 19:02 plus an assumed three-minute cascade does the same. No time remains for Eric to read the DM and enter the pick.
**Specification:** line 73 invents an 18-minute *default* (main defaults to 12 and self-raises from measurements), asserts an unmeasured cached allowance, and promises exit 0 after selection/delivery. It does not define outcomes for empty predictions, policy skip, status failure, delivery returning False, or a crash-marker lock. `is_last_check` exists in neither main nor the inspected `sr-lastcheck` APIs; what it changes about projected candidates is unspecified.
**Required edit:** use the effective configured/measured cascade budget plus an explicit operator reserve, against the earliest relevant pick cutoff; recompute after selection. State that this is a planning allowance, not a runtime guarantee. Define outputs/exit codes and durable state for every result; never report successful recovery merely because one attempt ran. Specify the projected-lineup rule and the actual API to implement it.
**Demonstrated:** arithmetic reaches the real guard's inclusive cutoff exactly; source inspection verifies defaults and absent parameter. No cascade duration was measured or invented.

### C3 — A selection lock alone does not make an external retry coexist with the daemon
**Sequence:** daemon sleeps holding state A; retry acquires `.selection.lock`, reloads state, commits B, releases it. Daemon later acquires the lock but keeps A and saves it. The lock serializes writes without refreshing their contents. Main's `save_state` merges only `analytics_jobs`; it can overwrite B's `pick_locked` and finalization state with A.
**Specification:** line 73 specifies reload for the retry but only lock acquisition for daemon checks. It does not say whether fallback, final fallback, classification finalization, and post-check saves share the lock. A finalized skip is an exit-0 no-op even after a corrected manual observation, so the advertised recovery can leave the day closed while games remain enterable.
**Required edit:** define the whole check/finalization transaction and reload/merge rules for both processes, including fallback and EOD. Specify what a corrected observation may reopen, or explicitly state that finalized skips require another named operator action. Preserve delivered/unconfirmed-send immutability. Add two-process and skip-then-correction cases, not just retry-after-locked idempotency.
**Demonstrated:** sequential real `save_state` calls representing serialized old-daemon/new-retry writes changed `pick_locked=True` back to False. This demonstrates lost state, **not** duplicate delivery; existing disk-decision checks provide additional protection.

### C4 — Restore safety is promised without the restore procedure that makes it true
**Sequence:** in-place restore overwrites account files while a daemon/fetch is alive, then writes `restore_state.json` afterward. A reader can run between extraction and the marker and use apparently current-season restored bytes. A failed extraction can also leave partially restored files without a marker. `--acknowledge-downtime` does not itself stop writers.
**Specification:** lines 23, 76 and 108 rely on managed restore and claim the runbook mandates it. The actual specified runbook edits cover A4/A5/B8 only; none supplies that restore procedure. Main has a restore **drill**, not a production restore transaction.
**Required edit:** specify quiescence of scheduler and fetch writers, a guard established before any overwrite and preserved on failure, target layout/snapshot selection, validation, acknowledgment, saver reconciliation, and restart/retry. State how direct restic/snapshot restores enter this procedure. Specify recovery from a corrupt marker; current `ack-restore` conditions do not explain how it repairs one.
**Demonstrated:** command/write-order and runbook inspection only; no restore was performed. Rejecting an *unacknowledged managed restore* is consistent with R24's stale-day scope. It must not be expanded to ordinary failed-fetch days.

### C5 — The new 36-hour saver rule is neither R24's decision nor a continuity proof
**Sequence:** accepted streak 9 at day 1, 13:30; that afternoon's double earns the saver. Day 2's early-game miss consumes it; fetching resumes at 16:05, after **26h35m**, with best 11. A still-tracked `not_earned` flag passes the proposed age guard and auto-earns a spent saver. Conversely a 37-hour gap during an off-day/break demotes an untouched reserve. These are synthetic possible sequences, not measured endpoint behavior.
**Specification:** R24 mandates post-restore reconciliation; v2 adds post-gap forced-spent state and presents it as part of the ruling (lines 5–8). It does not justify 36h or bound the shorter ambiguous interval. Its command says reconciliation restores continuity, but `policy_saver` still forces False while auto age exceeds 36h. A successful reconciliation alone therefore cannot do what “until reconciled” promises.
**Required edit:** ledger the gap rule as a new design choice and state its limits, rather than attributing it to Eric's stale-day decision. Define exactly which uncertainty requires confirmation and how a newer explicit reconciliation supersedes an older observation/gap; preserve keep-picking on the accepted streak. Add the sub-36h earning/consumption and off-day cases.
**Demonstrated:** the real retained auto-earn hook promoted `not_earned` to `active`; spec arithmetic did not demote at 26h35m. The account consumption in that trace is synthetic.

### C6 — Section 2 is not a total, internally consistent classifier
**States:** (a) valid auto `recorded_at = now + 1s` through `now + 5m`: R4 rejects only beyond five minutes, P2b does not match, P3 requires nonnegative age; no live row. (b) research with no contract: R3 wins before M1, despite `require_contest_state=False`. (c) fresh auto with `last_accepted_attempt_id != auto.attempt_id`: record rules require stale, but P3 still matches and D3 merely adds a warning.
**Further conflicts:** `policy_saver` requires a reconciled restore record, but absence is not defined as a neutral value; literal `None` would disable the saver on every never-restored installation. P1 takes saver authority from the manual file, while D2 prescribes `saver-state --reconcile`, which changes only the flag; P1 would continue using its original value. D1's “any picking row” introduction disagrees with its P2/P3-only condition.
**Required edit:** make research/live gating explicit before refusal rows; choose tolerance handling for clock skew; include attempt mismatch in the stale predicate; define absent-restore semantics and one saver precedence including manual corrections. Specify which stale reason wins when several apply. Do not fall back to model state for an unmatched live row.
**Demonstrated:** table transcription enumerated all 300 future seconds, research precedence, and mismatch/fresh contradiction. Saver conflicts are textual, not executions of future code.

### C7 — The alert promise contradicts the dispatcher and its own acceptance test
**Sequence:** 10:00 refusal sends `contest_state:season_contract_unverified`; same refusal at 16:10 is deduplicated. The CS6 RED test demands a DM at both moments. Changing instructions but retaining the same code is also suppressed, although R24 says a new refusal with new instructions is a distinct incident.
**Specification:** lines 42–44 promise per-code daily dedup; lines 127/170 promise repeated pre-cascade DMs. A 10:00 startup is only a default: main computes earlier wakes for international games and can restart at any hour. `day-outcome` reports a generic `no_decision`, not the original refusal or its recovery; clean skips/entries disclose neither a stale reading nor a saver issue.
**Required edit:** choose first-notification versus reminder/escalation semantics, make the tests agree, and bind incident identity to materially new required action. State “actual run_day startup,” plus any separately implemented timed reminder. Specify the response/entry reserve in C2; a sent DM is not proof Eric read it. Keep 08:30 next-day `day-outcome` described as a later generic check.
**Demonstrated:** real dispatcher sent one DM, suppressed both the repeated code and changed instructions. Wake/day-outcome behavior checked in source.

### C8 — The switch's evidence and promotion transaction are incomplete
**Sequence:** manual observations from opening dates are overwritten by later dates, but `season-contract status` promises a historical join of those observations. No manual observation history is specified; saver transitions cannot reconstruct streak/best. `enable` promotes an attempt but only the fetch is allowed to write accepted auto in the record table, and `enable` does not specify the fetch-status/saver updates. Old status can immediately classify the promoted observation stale.
**Other gaps:** a verified counter contract may legitimately coexist with retained older prediction rows or a truncated/nonmonotonic rounds calendar. The enable/all-settled-row and round-ID-floor restrictions can prohibit the switch indefinitely; one positive current row still does not itself prove counter semantics. “Latest HTTP-complete” is undefined for failed/partial attempts; mandatory attempt fields are unavailable on auth errors. No locking/failed-append policy covers concurrent fetch, manual, enable and revoke operations.
**Required edit:** persist the manual observations the status UI promises; define an operator attestation based on actual inspection rather than calling settled-count proof. State the supported endpoint/mapping contract and refusal limits. Specify enable's full writer/status/saver ordering, crash behavior, and common mutation locking. Log partial failures with explicit null/unavailable fields. Test crashes after every durable step; saver-before-streak ordering alone is not a multi-file transaction.
**Demonstrated:** record/command/source comparison only. I did not simulate MLB changing its calendar, execute a switch, or claim the retained-history scenario will occur.

## Closure audit of the earlier attack

“Concrete” below means buildable and falsifiable on paper, not implemented or validated by the 105 baseline tests.

| Earlier finding | Closure assessment |
|---|---|
| F1 | Concrete diagnostic isolation, corrupt bytes/shapes and selected-fallback error; keep it. Invalid model diagnostics must not become an activation prerequisite. |
| F2 | Concrete removal of `proven_season`, separate attempts, missing-array error and explicit operator switch; main frame closed. Complete switch transaction/contract limits in C8. |
| F3 | Local-file freshness inference removed; restore guard concrete but operationally incomplete (C4); continuity/gap semantics incomplete (C5/C6). |
| F4 | One-date actual-value observation, idempotency and live invariant concrete; incompatible staging and manual/flag precedence remain (C1/C6). |
| F5 | Pre-prediction readiness and named retry are concrete improvements; promised recovery still lacks feasible deadlines/result/concurrency rules (C2/C3/C7). |
| F6 | Shared classifier, zone warning and existing WARN/incident channels concrete; contradictory classifier/notification rules prevent full closure (C6/C7). |

| Earlier coverage row | Assessment |
|---|---|
| S1a | Concrete same-date observed values; first-wave writer compatibility and usable retry need C1/C2. |
| S1b | Frozen multi-day zero removed; next-date expiration is testable. Reminder claim needs C7. |
| S2 | Pre-switch nonacceptance and post-switch regression preservation concrete; no inherited reset proof. |
| S3 | Counter-only pre-switch refusal and zone warning concrete; post-switch contract must state supported ledger-lag behavior (C8). |
| S4 | Unmapped rows preserve accepted state; stale picking matches Eric's decision. Recovery needs C2/C3. |
| S5 | Old-only mapping preserved as unaccepted; no false one-day cost bound. Switch limitations need C8. |
| S6 | Operator contract replaces self-certifying source year; restore/continuity incomplete (C4/C5). |
| S7 | Weak responses no longer inherit a witness or replace the accepted observation; concrete. |
| S8 | No fabricated restore zero; managed-restore recovery not operationally closed (C4). |
| S9 | Date-file hardening plus corrupt-model isolation concrete; no pick-file freshness inference remains in final classifier. |
| S10 | Real tagless bytes, void day and repeated scoring specified; concrete, with fixture migration needed at CS2. |
| S11 | Explicit season enable blocks Dec-31 early switch; future-clock tolerance has an uncovered interval (C6). |
| S12 | Conservative unknown and warnings specified; missing-restore neutral value and gap reconciliation need C5/C6. |
| S13 | Same-date manual bypass and model isolation concrete; saver correction precedence and reader/writer staging need C1/C6. |
| U1 / partition / witness / crash | Stored counter bounds and classification precedence concrete; final-time bounds and multi-file failure promises incomplete (C6/C8). |

| Earlier operator mistake | Assessment |
|---|---|
| Setup skipped | Concrete readiness check without predictions; actual startup and notification semantics need C7. |
| Setup late | Actual-value manual is specified; deadline, retry result and finalized-skip behavior need C2/C3. |
| Done twice | Tuple idempotency/anti-regression specified; define effects after later saver mutation and changed reason/username/operator (C6/C8). |
| Wrong date / delayed opener | Same-date writer/refusal concrete; “off-day no DM” needs schedule-aware suppression in health/EOD as well as run_day. |
| Left active / renewed | One-date expiry prevents silent reuse; next-decision alert is concrete, repeated reminder claim inconsistent (C7). |
| Restore or downtime | Stale streak picking matches owner decision; restore procedure and saver recovery incomplete (C4/C5). |

Earlier Codex/Carbon r2 and audit rows are addressed by the same entries: season inference F2/F3; reserve/unknown F3/F6; malformed reads F1/S9; manual/health alignment F4/F6; target-year saver S11/S12. No earlier obligation is closed merely by the specification's assertion that it is closed.

## Numbers and realistic boundary days

| Number/rule | Basis and failure direction / required clarification |
|---|---|
| 0..57, reserve below 10, saver zone 10..15 | Contest/policy state bounds and retained R24 mapping; justified by current code. Require exact integer types and distinguish season identifiers from streak bounds. |
| 14h; current ET date | New diagnostic policy, no measurements supplied. At 00:30 after a 13:30 fetch, age is only 11h yet the date rollover says stale; late West Coast settlement may be normal. Off-days/All-Star break can leave an old witness without changed account state. False-positive stale DMs are acceptable only if described as uncertainty and picking continues. |
| 36h | New saver policy, not R24; both false negative (26h35m trace) and false positive (off-day/break) in C5. Never present it as proof continuity was intact below the threshold. |
| 5m future tolerance | New clock-skew allowance; 1..300 future seconds have no table row (C6). Decide reject versus tolerated effective age, consistently for stored timestamps/dates. |
| 01:10/02:10/10:30/13:30 ET | These are the real cron lines, not a settlement SLA. A delayed West Coast game can finish after both overnight fetches; 10:00 readiness precedes 10:30 refresh. A day with no successful fetch between midnight and first pitch must use stale accepted state after the switch, not refuse. |
| 10:00 ET; 08:30 next day | First is configurable/default wake, not guaranteed notification time; early overseas games require earlier action. Second is actual day-outcome cron and cannot rescue the previous date (C7). |
| 6h fetch throttle; per-day reason dedup | Existing transport policies, not acknowledgment guarantees. Repeated unaccepted mappings may produce only an earlier generic incident; specify escalation when new action is needed (C7). |
| 12h enable window; latest attempt; >=1 settled row; round-ID floor | New admission restrictions, not empirical proof. A latest failed/partial response can disqualify an otherwise inspectable response; retained history/truncated calendars can prevent switching. Manual operation remains required with no bounded duration (C8). |
| 23m/8m retry; 18m default; 5m cutoff | 5m is real shared cutoff; 18m is not main's 12m default; cached 3m is unsupported. Both latest starts leave zero entry reserve (C2). Define T for later still-playable games, doubleheaders and double picks; never use the primary leg alone. |
| Manual end-of-date expiry; YYYY==today.year | Explicit date-scoped observation/marker rules, not a season inferred from files. Specify aware-clock requirements, exact expiry equality, and historical research semantics. A delayed opener must receive a new observation on its actual date. |

## Every merge boundary and existing-test implications

These are conflicts inferred from inspected fixture/assertion code, not measured failures of unimplemented change sets. Files outside the CS “Files” bullets must be updated when they encode a deliberately retired contract.

| Boundary | Coherent with only earlier sets? Existing tests affected |
|---|---|
| main -> CS1 | **No** with v1 rejection: current fetch cannot produce usable auto. `test_contest_state.py::test_fresh_contest_state_drives_live_decision`, `test_contest_state_best_trust.py::test_fresh_auto_best_is_trusted`, and bare/dated-auto tests conflict. Bounds intentionally change `test_best_below_streak_is_untrusted` / `test_best_above_target_is_untrusted` to rejection. |
| CS1 -> CS2 | Season APIs/reset can be coherent independently after C1 is fixed. `test_streak_applied_dates.py::test_a_legacy_streak_file_has_no_marker_and_still_works` explicitly expects tagless 3 -> 4 in 2027, contrary to reset -> 1. Its dated-update tests and incident/scheduler scoring fixtures seed tagless nonzero state: tag current-season fixtures rather than changing real expected scoring. Define date-less `update_streak` behavior. |
| CS2 -> CS3 | **No** if “share evaluate_decision_state” means the final Section 2 table: contract/manual/continuity writers absent. Saver mapping alone can be coherent. Health `test_saver_flag_uninitialized_below_zone_no_warn` conflicts with the new any-streak warning. Source-year/season-bound fixtures need target-year pinning; `test_season_for_uses_source_date_year_else_now` can remain unchanged if the utility is retained while live callers stop using it. |
| CS3 -> CS4 | **Not specified:** `enable`/fetch require CS5 continuity; v3 manual writer still absent. `test_build_observation_returns_auto_schema`, `test_fetch_cli_persists_snapshot_when_no_settled_predictions`, ledger and auto-earn fetch tests need contract-enabled/pre-switch variants. CS4 claims local-pick-free final freshness while its removal is assigned CS5. |
| CS4 -> CS5 | **Not specified:** supplies restore/continuity writer but existing CLI manual v2 is still R1; only CS6 repairs it. Add backward-compatible migration or delay strict manual rejection. Existing saver loader/transition/concurrency tests and backup CLI/drill tests must still cover prior format/commands; new restore must not replace the drill. |
| CS5 -> CS6 | Final composition still needs C2–C8. `test_orchestrator.py::TestRunAndPick::test_uses_fresh_contest_streak_for_live_action` uses a legacy manual; health legacy-manual tests must migrate. Entry-intent “enter” fixtures omit both newly required config keys. Live delivery tests without state need valid observations or an explicit loader seam; research must retain M1. CS6's two-DM test contradicts C7. |

## Remaining choices an implementer must not guess

- Exact `DecisionEvaluation` fields, error/reason vocabulary and health `expected=False` semantics; malformed-optional versus absent-optional precedence; account identity source for rejecting a manual username (classifier accepts no expected-account argument).
- Whether `recorded_at`, `recorded_et_date`, `season`, `witness_date`, `contract_verified_at` and account identity are cross-validated; timezone-naive clock behavior; handling of unmapped/unparseable **unsettled** rows and unknown result vocabulary.
- R1 repair from malformed bytes: where anti-regression baselines come from and whether `--replace` alone can bypass unreadable content; whether `--force` is also needed by the advertised repair command. `saver-state --reconcile` currently permits used -> active without `--force`, unlike the new manual writer.
- `reconcile --date <D>` in M1 does not exist on main and is not specified in any CS; real CLI probe exits 2. Plain `reconcile` rebuilds model replay only when history permits it; it is not a universal repair instruction. Give exact quarantine/rebuild commands and their loss limits.
- Whether readiness failure at startup records/alerts and continues future checks, or escapes `run_day` and restarts indefinitely; suppression on off-days/all-void schedules across health/EOD; readiness recheck after a long cascade; selection-time versus delivery-time stale provenance on a cached pick.
- `retry-pick --dry-run` write/transport guarantees, lock acquisition policy, lock coverage during fallback/EOD, absent `last_planned_check`, finalized-skip reopening, and reconciliation with the daemon's in-memory state (C2/C3).
- No command should claim success means a pick was delivered, a saver was restored, or automatic operation is ready unless it reports that actual resulting state. The runbook must state manual-observation and post-restore operator duties, plus the pre-entry response window, rather than relying on “DM at 10:00.”
