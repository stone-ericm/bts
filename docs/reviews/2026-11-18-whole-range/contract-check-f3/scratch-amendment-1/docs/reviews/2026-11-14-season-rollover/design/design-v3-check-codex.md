**CS2 verdict: BUILD WITH THESE EDITS — V9 (season propagation and preservation).**
**CS3 verdict: BUILD WITH THESE EDITS — V9 (bind health/dashboard reads and writes to the same target year).**
**CS4–CS6 verdict: NOT YET — resolve V2–V8 before commissioning these contracts.**

**Take back to the owner:** V1 invalidates the stated cost premise of R25's C4/C5 simplification. The lead should stop acceptance of that residual saver risk and present the actual policy counterexamples to Eric. This does not block CS2/CS3 or require abandoning daily observations, operator attestation, or his keep-picking decision. The other findings need specification edits, not another owner preference. Do not silently reinstate the removed 36-hour rule or restore-state machinery.

Reviewed the six requested documents in order, then code/tests on local `main` at `7ea31cfd6271`. Runtime source/tests/scripts have no diff from the earlier check's `2fa1eb3e`. References `v3:L` below mean `docs/reviews/2026-11-14-season-rollover/design/design-v3.md`.
Ran the seven contest-state/health/saver/fetch files listed in the first check: **105 passed**. Ran `test_fallback_plan.py`, `test_scheduler_fallback_deadline.py`, `test_scheduler_decision_record.py`, `test_private_mode_transport.py`, `test_backup.py`, `test_backup_cli.py`, `test_entry_intent.py`, `test_streak_applied_dates.py`: **124 passed**. Both invocations used `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider ... -q`.
Ran `.codex-review/srd6-scratch/probe.py` with that Python/environment; assertions passed. Script, `results.json`, `run.log`, and the additional baseline log are in that scratch folder. Real policy/CLI/dispatcher/state-writer probes are distinguished from table transcriptions and intermediate-file snapshots. Passing main's tests does not validate unbuilt v3 behavior. No network, box, contest account, tracked-file edit, or outcome study.

## Findings, most severe first

### V1 — The claimed 5–15 saver blast radius is false on the actual artifacts
**State:** 2027-08-23, streak/best 38, 34 days remaining, primary probability .82, an eligible different-game second candidate. The real resolver chooses **skip with spent**, **single with available**. At streak 16 that day it chooses **single versus double**; at streak 4 on Aug 29/.80, **skip versus double**.
**Specification:** v3:12,169,222–223 says saver errors affect only 5–15 and that 0–4/16–57 are invariant, repeating R25's cost statement. Those are claims about policy output, not merely the immediate miss-forgiveness zone.
**Should happen:** correct the risk statement and operator obligations using reachable policy cells; have the owner reconsider the accepted blind spots on that basis. Keep the ruled reserve mapping below 10. Saver availability can affect future paths after a reset even above the immediate forgiveness zone.
**Ran:** `real_policy_bound`, using `resolve_policy_decision` with the checked-in base table (sha `66d15471…`, not a rebuilt policy). Raw base-table differences extend to streak 50; tail-table differences also include 2–4. The three examples above exercise the real production resolver in its reachable reach-57 regime, not unreachable raw-array cells.

### V2 — Automatic admission still has an error bypass and incompatible drift promises
**State:** an HTTP-200 attempt has in-range counters, matching account/year, no prior-year mapped rows, but `outcome="parse_error"` from an unknown result/schema. Alternatively a verified June contract receives a semantically ambiguous response while an accepted Y observation remains.
**Specification:** `enable`'s complete admission list (v3:151) checks HTTP success, account/year, bounds and old mapped rows, but never requires successful parsing/classification. v3:160 promises to detect carried counters/semantic drift and says the contract remains unverified/revoked while unsupported behavior persists; it gives no transition rule for an already verified contract. Supported unmapped/counter-only lag cannot itself prove that no Y hit occurred.
**Should happen:** admit only fully parsed, invariant-valid attempts, including explicitly eligible pre-switch rejections; reject parse/auth/network/invariant failures. List executable fetch invariants, including zero-round rejection, prior-year filtering and future settled mappings. Declare indistinguishable counter-semantic drift undetectable rather than promising rejection. On a new rejected/ambiguous mid-season response, preserve the verified contract and accepted observation and **PICK stale**; only an explicit operator revocation changes that permission. Automatically revoking it would contradict Eric's decision.
**Demonstrated:** specification/source inspection only. No v3 admission implementation exists; no synthetic endpoint is evidence of MLB semantics. Operator attestation is the right frame, but does not cure this missing admission predicate.

### V3 — Revocation does not require a new streak observation, and the saver duty's command can do nothing
**Sequence:** restore a same-date manual observation made at 12:00; revoke at 13:00. Gate 2 still selects that old manual before inspecting the contract. Separately, a tracked `not_earned` saver is earned and burned between accepted fetches; Eric immediately follows §7 and runs `saver-state --use`.
**Specification:** v3:54 allows P1 even with a revoked contract, while v3:113,144 promises revoke forces R2 and the next post-restore pick needs an explicit observation. v3:89 names `--earn`; §7 promises `--use` marks the between-fetch saver spent, with unchanged existing flags.
**Should happen:** invalidate manual observations predating the latest revocation, while allowing a new same-date explicit observation to restore P1. Define timestamp equality/idempotent re-attestation. Give state-dependent executable saver instructions: `bts saver-state --init used --force --season Y` can correct `not_earned`; `--use` alone cannot. The dashboard also has no used button for `not_earned`. `--init` requires a value and `--earn` does not exist; specify a new flag or remove that recovery.
**Ran:** Gate-2 transcription returns P1 for the restored old manual. Real CLI returns **exit 0/no-op**, leaving `not_earned`, for the mandated `--use`; the real next best=10 auto-earn then resurrects `active`. Real `--earn` exits 2. This is failure even when Eric performs the stated duty, beyond the knowingly accepted unreported-consumption limit.

### V4 — A writer lock is not a consistent read or a crash contract for the account files
**Sequence:** `set-contest-streak` writes manual streak 13/saver `used`, then fails before updating the old `active` saver flag. Or `enable --attempt-id B` is run while the contract is already verified for A, then fails after replacing the auto observation but before status/contract replacement.
**Specification:** v3:18 locks writers only; v3:111 names manual → history → saver writes without interruption semantics. All P1 saver decisions use the separate flag. v3:157 assumes the untouched contract is necessarily unverified/revoked, although no enable precondition says so. Fetch mutation order and failed JSONL append behavior are not defined.
**Should happen:** specify a consistent read protocol plus recoverable ordering/generation rules for manual, fetch, enable and revoke, including failed appends and idempotent retries after every durable step. Specify whether enable on verified state rejects, is a no-op, or safely replaces it. Preserve the attempt's original retrieval time when promoting it. A common lock helper must support already-locked saver transitions; simply renaming the inner lock deadlocks nested fetch/enable/manual calls.
**Ran:** specified-order snapshot plus the real saver reader exposes **P1 streak 13 / active** while the new manual explicitly says used; this is not execution of v3. Re-enable crash transcription returns **P2b with B**, not the promised R2/P1. Two separately opened descriptors for the same common flock conflict even in one process. Power-loss durability also needs directory fsyncs or an explicitly narrower process-crash claim; file fsync plus rename alone does not establish the stated ordered durability.

### V5 — Retry's purported main API and result contract still disagree
**State:** retry reaches its projected-lineup fallback; or retries a real `locked_unconfirmed` decision left by a prior send crash.
**Specification:** v3:129 calls `plan_fallback_action(sel, date=..., picks_dir=..., now_utc=...)` and acts on `"lock"`. Main's planner is keyword-only, takes cutoff/windows/confirmation/budget/reserve inputs, and returns `"deliver"`/`"defer"`. `_fallback_min_with_floor` also requires the operator reserve, omitted from the described `5+B_eff` floor. v3:121/132 treats every scoreable commit as exit 0, while v3:137 requires exit 6 for unconfirmed delivery. Main deliberately makes `locked_unconfirmed` scoreable.
**Should happen:** specify the real planner arguments and actual action vocabulary, with R_op in its floor; retain the intentional projected-primary deferral policy. Give unconfirmed delivery precedence over generic scoreability. Separate delivered, private research commit and unconfirmed commit; do not equate scoreability with delivery. C2 required distinct results for policy skip and no predictions; v3 still combines no predictions and deferral in exit 4. Define cutoff-during-cascade and status-failure results, presently 2/6/4 ambiguous.
**Ran:** literal proposed call raises TypeError; the real planner returns `deliver`. The real classification writer produced `scoreable=True, delivery_status="locked_unconfirmed"`; applying the proposed generic guard returns 0 instead of 6. No retry command exists yet.

### V6 — Selection transactions omit other overwrites, and delivery-time reselection has no complete contract
**Sequence:** retry commits while the daemon is doing next-day lookahead; the daemon saves its old state afterward. Or saver/streak changes after a candidate was chosen, and a failed refresh leaves only a cached DailyPick/`selection=None`.
**Specification:** v3:118 lists five transaction sites, excluding `run_day` startup, coalesced-check saves, lookahead/result saves, and `_update_analytics_job_status`; rebinding after a previous transaction cannot refresh state after a later external retry. v3:122 orders a local reselection but does not define a persisted comparison tuple, predictions for cached fallback, changed skip/no-candidate outcomes, or rechecking confirmation/cutoff for newly selected legs.
**Should happen:** serialize/reload every scheduler-state mutation or make nonselection writers merge only their owned fields; define lock ordering, including analytics callbacks invoked while a transaction already holds the lock. Persist the candidate's decision inputs and available ranked-slate reference. Specify cancellation to skip/refusal, no-data behavior, lineup checks and recomputed earliest-leg cutoff after reselection, then stamp the decision that actually authorized delivery.
**Ran:** serialized real `save_state` calls still let the old lookahead save change both retry flags from True to False. This demonstrates lost state, **not duplicate delivery**; disk-decision guards remain. Cached-no-DataFrame and reselection outcomes were source/specification inspection, not a v3 execution.

### V7 — Action-hash dedup is not “whenever action changes”; the no-candidate reminder is undefined
**Sequence:** required action changes A → B → A on the same date. Or readiness refuses before any predictions/pick exist, so main never computes a pick-specific fallback deadline.
**Specification:** v3:95 promises a DM on every action change, but v3:99–100 deduplicates command hashes already sent that day. v3:94 requires a reminder relative to `fallback_deadline` without defining it for no candidate, later playable games or a moved game time. v3:97 incorrectly describes wake as `earliest - offset - 15m`; v3:172 also calls first+reminder+action changes a two-DM cap.
**Should happen:** persist action transitions/reminder identity, or narrow the promise to new distinct actions explicitly; R25 currently requires the former. Define a schedule-derived deadline before selection, recalculation, restart/overdue-slot behavior and when first+reminder coalesce. State the resulting response window before latest retry start, not only the entry reserve. Use main's actual default-10:00/earlier-game wake computation; remove the incompatible cap.
**Ran:** real dispatcher with the proposed hashes sends **True, True, False** for A/B/A. For a 19:10 game `compute_wakeup_time` gives **10:00**, not the claimed 18:10 at offset 45. No-pick fallback dependency verified in main; `sr-lastcheck` is named but not supplied by this design.

### V8 — The claimed total classifier still delegates malformed states and contradictory summaries to guesswork
**States:** a manual v3-shaped object lacks `recorded_at` or `effective_date`, has an unknown schema/saver enum, or has mismatched expiry; a valid fresh auto/status pair has no attempts file; research M1 has a wrong-season model file and a used/missing saver flag.
**Specification:** v3:53 enumerates corrupt manual conditions without all §1 structural failures; P1 requires a “valid” file and the residual branch describes absent/prior/expired files, leaving other malformed objects unspecified. v3:61 says missing optional attempts become stale, but predicate #1 only mentions a newer rejection or corrupt tail: a missing file matches no predicate when status is good. M1's `(0, True)` summary conflicts with §2.2's all-row flag precedence unless True is explicitly diagnostic model state.
**Should happen:** make parsing return a defined invalid classification for every missing/type/schema/date failure before comparisons; define which expired malformed files still block. Make absent/unreadable/mid-file-corrupt attempt history explicit stale cases and define timestamp comparison errors. Spell M1's effective saver separately from model diagnostics. Gate precedence otherwise resolves R1/R2 overlap, and 1–300 future seconds now correctly clamp; no live model fallthrough is intended.
**Demonstrated:** table inspection only; no claim to have run a total v3 implementation. These are disagreements between the gate predicates, record constraints and row summaries, not new evidence that local files prove season or freshness.

### V9 — Early sets need precise year plumbing; later deployment and restore instructions still omit real compatibility work
**Sequence:** a tagged model file is saved through a dateless/default-None writer, then receives a dated update; or CS3 decisions use Y but dashboard/health still use the prior observation's source year. CS4 then deploys over existing legacy manual/auto files with only the old live-mode configuration.
**Specification:** v3:190 preserves default legacy behavior without saying whether a default save preserves an existing season tag. Main's shadow selection, replay marker reads and scoring callers need explicit year propagation. v3:194 binds only `load_decision_streak_state`; main `health/contest_state.py:137` and `web.py:_saver_season` independently call `season_for(source_date)`, including the dashboard mutation target. CS4's classifier/recovery exists before unconditional live gating, readiness, retry and the runbook. Co-shipping new schemas fixes writer availability, not conversion of existing files/configuration.
**Should happen:** specify all model read/write call sites, preserve a tagged season through dateless writes (or reject that operation explicitly), and apply the same target-year rule to health/dashboard mutation and display. Declare a quiescent CS4 migration/activation barrier through CS6, or ship the required guard/config check and usable recovery then. Legacy manual files must be replaced with a real observation or explicitly quarantined; a successful v2 fetch/enable alone cannot bypass Gate 2's legacy-manual R3.
**Operator trap:** v3:144 restores `data/health/`; actual ops backups contain **`data/health_state/`**, nested below the original absolute project path under the drill target. Specify resolved source/destination paths, all operational writers to quiesce (including scoring/reconcile/entry-check/backup cron already running), rollback on partial sync, and a fresh empty drill directory; the fixed `/tmp/bts-restore` fails on reuse. Explicitly retain `contest_ledger.jsonl` or adapt its dashboard/restore verifier consumers: attempts are not its replacement format.
**Demonstrated:** source/specification inspection; no restore performed. Existing test identifiers `test_not_earned_maps_false`, `test_not_earned_is_false`, `test_saver_uninitialized_below_zone_silent`, and `test_corrupt_best_below_current_falls_back` do not exist on this main. Use the actual assertions named below instead of promising a nonexistent test migration.

## Closure audit

“Concrete” means an implementable rule with a falsifiable test, not implemented by v3's prose. “Partial” identifies the remaining finding; a closure-summary assertion alone earns no credit. v3:176 also mislabels the original F3/F4/F6 subjects.

| Finding | Closure |
|---|---|
| F1 | Concrete: CS1 matches kickoff rows 1–8, optional diagnostic fields and manual-before-auto isolation; keep. |
| F2 | Partial: no `proven_season`, separate attempts and operator switch are concrete; admission/drift/mapping claims need V2/V4. |
| F3 | Partial: local-pick freshness removed, age explicit, runbook revocation chosen by R25; old manual/recovery and risk premise remain V1/V3/V9. |
| F4 | Partial: actual-value one-date observation and undated-live rejection concrete; interrupted/idempotent writes and staged delivery enforcement need V3/V4/V9. |
| F5 | Partial: readiness independent of predictions and explicit retry are concrete; API/results/concurrency/reminder windows need V5–V7. |
| F6 | Partial: common classifier and not-earned 10–15 warning concrete; routing/transition semantics and parser gaps need V7/V8. |
| C1 | Substantially fixed: v1 survives CS1–3; CS4 supplies strict readers with writers. Existing-file/config migration and later-only recovery remain V9. |
| C2 | Partial: actual budget + reserve + earliest leg is concrete; invalid API, merged outcomes, false success need V5. |
| C3 | Partial: reload/decide/write and explicit reopen-skip concrete; omitted writers and changed-delivery outcomes need V6. |
| C4 | Partial: stopped scheduler/fetch + revoke is the ruled procedure, no new restore machinery; stale manual and real restore layout need V3/V9. |
| C5 | 36h rule removed, flag authority/guard concrete; wrong risk bound and unusable burn command remain V1/V3. |
| C6 | Research gate, tolerated skew, status mismatch and stale priority concrete; malformed/missing history cases and manual revocation need V3/V8. |
| C7 | Partial: first/reminder/action-change/stale channels named; action-cycle, no-candidate timing and cap conflict need V7. |
| C8 | History and operator attestation concrete; full parsing admission, concurrent reads, repeat-enable and crash behavior need V2/V4. |

| Earlier decision coverage row | Closure |
|---|---|
| S1a | One-date observed values concrete; saver tuple/crash behavior V3/V4. |
| S1b | Frozen multi-day zero removed concretely; renewal/reminder semantics V4/V7. |
| S2 | Pre-switch zero cannot automatically authorize counters; malformed-attempt enable bypass V2. |
| S3 | Counter-only acceptance requires attestation; not-earned warning concrete; supported lag versus carried counters V2. |
| S4 | Rejected mapping preserves accepted streak and P2 picks stale; retry completeness V5/V6. |
| S5 | Prior-only endpoints may require indefinite manual operation; executable admission filters need V2. |
| S6 | Operator contract replaces source-year proof; post-restore old manual/risk V1/V3. |
| S7 | No inherited witness intended; which mixed/future/unmapped mappings may replace accepted state needs V2. |
| S8 | No invented restore zero; next explicit observation and restore paths V3/V9. |
| S9 | CS1 byte/shape hardening and model isolation concrete; no local-history freshness inference in final gates. |
| S10 | Season/reset/idempotency concrete in principle; default writers and replay marker reads V9. |
| S11 | ET target and tolerated skew concrete; enable lacks explicit current-year restriction and must define historical/future season eligibility (V2/V8). |
| S12 | Unknown saver conservative and warned; post-revoke recovery/target readers V3/V9, claimed limited impact V1. |
| S13 | Current manual bypasses corrupt auto/model concretely; saver snapshot/restore ordering V3/V4. |
| U1 | Stored counter bounds concrete; full malformed object handling V8. |
| Partition / witness / crash | Strict stale precedence fixes overlap; missing attempts V8, mapping acceptance V2, multiple-file/crash ordering V4. |

| Earlier counterexample/audit coverage row | Closure |
|---|---|
| Codex r2 B1 / Carbon r2 B2 | Local season/freshness inference removed; explicit contract still needs V2/V3. |
| Codex r2 B2 / Carbon r2 B1 | Unknown remains false; restored current-season flag requires effective revocation/reconciliation V3. |
| Codex r2 B3 | Concrete CS1 defensive scans plus model isolation. |
| Codex r2 B4 / Carbon r2 B3 | Shared classifier/manual season and zone warnings concrete; health/target migration V8/V9. |
| Audit F1 | Required below-10 reserve mapping concrete; do not confuse that rule with the false impact bound V1. |
| Audit F2 | Decision target-year lookup concrete; independent dashboard/health target still needs V9. |
| Audit F3 | Unverified counters cannot be admitted by an old witness; parse-error enable bypass/drift V2. |
| Audit F4 | Zero/empty automatic supported after attestation; absence of predictions cannot suppress readiness once CS6 lands; no-candidate reminder V7. |

| Earlier operator mistake | Closure |
|---|---|
| Setup skipped | No invented streak; pre-prediction readiness specified, staged availability/window V7/V9. |
| Setup late | Correct values and explicit retry specified; actual planner/results and available entry window V5/V7. |
| Done twice | Tuple idempotency asserted; define re-run after saver --use/revocation/crash and changed note, including --force versus no-op precedence (V3/V4). |
| Wrong date / delayed opener | Date binding/expiry concrete; malformed/future writer rejection versus stored-but-refusing observation not specified (V8). Schedule suppression must include health/EOD, as CS6 explicitly promises. |
| Left active / renewed | One-date expiry concrete; no silent zero renewal. Next-date action/reminder semantics V7. |
| Restore or downtime | Ordinary stale auto still picks; managed restore needs effective new-observation requirement V3, real copy/quiescence steps V9. Out-of-runbook restore detection remains explicitly impossible. |

| Earlier alternative-C input row | Closure |
|---|---|
| Valid same-account/season/date manual | Buildable date/value/bounds rule and model isolation; separate saver snapshot and post-revoke validity need V3/V4. |
| Accepted auto under verified contract, fresh, saver continuity intact | Buildable P3 and flag precedence; admission/consistent account snapshot need V2/V4. |
| New ambiguous response, accepted state exists | P2a/P2b keep-picking implements Eric's decision; unsupported-behavior text must not revoke automatically (V2). |
| Old-season/undated/expired/restored-unvalidated evidence only | Live no-model rule concrete; old same-date restored manual escapes (V3); research parser cases V8. |
| Saver missing or continuity unverified | Conservative flag and warnings concrete; actual consumed-state correction command V3, impact claim V1. |
| No predictions yet | Independent readiness concrete in CS6; reminder deadline and executable retry remain V5/V7. |

The earlier eleven-date opening trace (Mar 24–Apr 3) is not bounded away by v3: each date can pick with that date's actual manual values; every date without one may refuse until a supported contract is attested. That recurring duty is now explicit, with no one-day loss promise or zero renewal. Normal settlement, old-only rounds, flashing zero/carried counters, current rows beside carried counters, empty rounds and prolonged auth failure still require the acceptance and rejection rules in V2, not local pick counts.

## Numbers and boundary-day checks

| Number | Basis and realistic wrong-answer day |
|---|---|
| 0..57 / exact ints / best >= active | Current code/policy target and arithmetic, not a verified 2027 rule; A3 must recheck. A new target 60 reaching 58 refuses/rejects. Validate stored and fetched values identically. |
| Earn 10; reserve <10; warning 10–15 | Existing contest/policy rules plus R24. A double goes 9→11 and a later miss consumes the saver before fetch; --use fails exactly then (V3). Forgiveness zone does not bound policy impact (V1). |
| Same ET date + <=14h | New diagnostic policy, not settlement SLA. A late West Coast game unsettled at 02:10 can still appear healthy if no pending row exposes it; a prior-day 23:55 reading is labeled stale at an 11:05 game although <14h old. Off-day/All-Star-break gaps and no post-midnight fetch likewise produce stale; **still pick and warn**, never demote saver by age. |
| 01:10 / 02:10 / 10:30 / 13:30 ET | Actual cron lines, not guaranteed fetches/settlements. A game finishing after 02:10 waits until 10:30; early international first pitch may precede it. There is no actual scheduled 23:55 fetch; that example requires a manual fetch. Healthy cron cadence does not prove an account counter is current. |
| 5m future tolerance | Chosen clock allowance; 1–300s now clamp to zero, >300s refuses. A +10m clock correction can refuse a genuine observation until time catches up; define comparisons for every optional timestamp too (V8). |
| Default wake 10:00; claimed offset+15; next-day 08:30 | Main wakes min(default-10:00, earliest-minus-60m), with configurable early buffer; service restarts may be later. Offset+15 is false (V7). 08:30 day-outcome is after the lost date, never a recovery window. |
| 12m budget; last two durations; ceil(max/60)+2; 10m reserve; 27m retry | Main uses configurable budget floor (not unconditional cold minimum 12); max of last two plus 2 is a planning estimate. A 19m cold cascade leaves only 3m after a default latest start. Unknown target pick pre-cascade uses max-playable warning, which cannot guarantee reserve for an earlier selected leg; print actual leg deadlines afterward. |
| 5m submission cutoff; fallback buffers | Existing code guard, inclusive rejection at cutoff, earliest of both legs. Main standard/morning fallback is 35/25m, floored by **5+B+R**, not 5+B. A fluid doubleheader game 2 or moved-up game needs fresh per-game times and a recalculated cutoff; the spec's ambiguous replanning cannot be solved by a constant. |
| 30s lock wait; no lock across sleep | New retry choice, not a response guarantee. A daemon's long cascade holds selection lock >30s: retry exits 5 even before nominal latest start. Include lock wait/readiness/transport latency in operator planning, or state that these spend the reserve. |
| 6h fetch throttle; 1/date stale; first+reminder+changes | First is existing per-category cooldown; others R25's chosen attention policy. Travel can exhaust the whole response window; “sent” is not read. A/B/A suppresses the third action notification; no fixed two-DM cap can also promise unlimited changes (V7). |
| Date-end 23:59:59.999999, expiry equality, Y=D.year | Explicit daily validity, not local-file season proof. Delayed opener needs a new observation on its actual date. Exact expiry instant is valid under <=; next microsecond is not. Clarify past/future manual writes and non-live historical targets (V8). |
| 36h / 12h enable / settled-row minimum / round-ID floor | First three old admission/gap rules are removed; no empirical proof is lost. Ignore-rounds-through remains an explicit operator filter, not proof counters are seasonal; define filter composition and retained mapping evidence (V2). |

## Merge boundaries and existing tests

These are fixture/assertion conflicts inferred from source, not measured failures of nonexistent change sets. Re-run all affected files; don't edit assertions merely to hide new behavior.

| Boundary | Coherence and existing-test implications |
|---|---|
| main → CS1 | Coherent under kickoff: dated auto_v1/manual_v2 writers remain usable; bounds/undated-required behavior intentionally changes. `test_contest_state_best_trust.py::{test_best_below_streak_is_untrusted,test_best_above_target_is_untrusted}` and undated-required fixtures change. Optional model-field consumers must tolerate None. The v3-named tail test does not exist. |
| CS1 → CS2 | No contest-schema skew; season APIs/reset can land after V9. Tag in-season fixtures in `test_streak_applied_dates.py`, `test_incident_register_2026.py`, `test_cli_integration.py`, `test_scoring_lock.py`, `test_picks.py`, `test_reconcile_cutoff.py`, `test_reconcile_receipt.py`, `test_pickpath_review_pp1.py`; preserve dateless idempotency tests and add tagged→dateless→dated/replay cases. |
| CS2 → CS3 | Compatible file formats; mapping can land after V9. Actual `test_saver_state.py::test_not_earned_and_used_not_available` tests the raw .is_available property, which may remain false when a separate policy helper reserves below 10; don't change it blindly. `test_decision_saver_fallback.py` has no claimed below-10 test: add one. Actual health `test_saver_flag_uninitialized_below_zone_no_warn` stays silent with default expected=False; add expected=True and cross-year dashboard/health/write tests (`test_web_saver.py`). |
| CS3 → CS4 | Replacement writers exist together, fixing old C1. Existing legacy manual on disk still yields R3 until replaced; false-required config can still choose M1 until CS6. Enforce the deployment barrier in V9. Change live/legacy fixtures in the v3 list; also `test_contest_state_corrupt_precedence.py`, `test_web_saver.py`, and CLI manual-schema assertion at `test_cli_integration.py:280`. Preserve or deliberately migrate ledger consumers/tests and backup verification. `daily_decision.py` and `strategy.py` must implement new fields despite omission from the Files list. |
| CS4 → CS5 | After V5/V6, named retry and finalization reopening become usable. “Zero existing scheduler tests break” is unsupported: rechecking/reselection changes test seams; exact metadata assertion in `test_scheduler_tail_objective.py::test_decision_meta_extracts_objective_fields_and_tolerates_absence` needs review after CS4 adds provenance. Preserve `test_scheduler_decision_record.py::test_classification_attempted_writes_locked_unconfirmed`; fix retry's exit logic rather than making its record non-scoreable. Cached fallback/tail and failed-write tests remain required. |
| CS5 → CS6 | Only now full live gating/readiness/reminders/runbook exist. Migrate listed live integration fixtures plus CLI run/orchestrate paths and entry-intent configurations. `test_scheduler_tail_objective.py::test_run_single_check_skip_summary_comes_from_the_selection` forbids a second state read for display; place readiness before orchestration without introducing display rereads. Scheduler call-count/no-prediction tests need explicit readiness seams. Research/private behavior and dry-run no-quarantine tests must remain meaningful. |

## Remaining choices that must be written down

In addition to each required edit: exact refusal/action vocabulary and canonical command normalization; enable eligibility for past/future seasons and optional expected username; the auto writer's target-year source; OR/AND composition of ignore filters and audit retention of the original complete round mapping; malformed/non-object prediction rows and absent round IDs; partial-history/corrupt-tail semantics; preservation of original fetched time on enable; whether an existing manual day's values can be superseded automatically after enable; anti-regression baseline after a forced manual correction; tuple-idempotency after later saver mutation; manual/history/saver failed-write recovery; reader/writer lock nesting and state/selection lock order; old schema/quarantine retention and default-None model tag preservation; canonical fetch-status filename/location (main uses `data/health_state/contest_streak_fetch_status.json`, v3's record table moves it under account_state, its command row abbreviates it to `fetch_status.json`); continued readability of actual schema-tagless saver files; outcome/exit and durable state for cutoff/status failures, dry-run races, candidate deferral and reopened-skip crashes. A dry-run must avoid main's mutating `load_state` quarantine path.
Restore/copy and finalized-skip reopening need explicit recovery from interruption; transport-attempted/scoreable commits remain intentionally immutable to retry, so its output must name manual verification when resending is forbidden. Merely writing a manual observation still does not wake an idle daemon: include retry (and --reopen-skip when applicable) and the latest-start calculation in actual recovery text. No irreversible command was executed in this review.
