# Season-boundary design attack

**Verdict: BUILD WITH THESE CHANGES.** Keep the separation of contest state, model replay and saver flag. Do not build the proposed meaning of `proven_season` or its recovery promises as written.

Required changes, tied to findings below:

1. Make a valid manual observation usable despite corrupt model diagnostics; specify and exercise a recovery that actually runs another selection (F1/F5).
2. Record endpoint evidence without claiming that a calendar, zero response or historical anchor proves the season of a later counter. Keep a weaker attempt separate from the last accepted observation (F2).
3. Treat restoration, observation age and legacy migration explicitly; neither absent local history nor undated fixture compatibility may confer trust (F3).
4. Invalidate saver continuity after an unverified restore; a same-season flag can still resurrect a spent saver (F3).
5. Replace the opening-days frozen zero with an observation valid for one contest date, renewable with the actual observed values; make the runbook and recovery steps mandatory before activation (F4).
6. Close the remaining health cases and specify real notification channels/timing, including the missing `not_earned` warning (F6).

## Evidence and limits

Reviewed `main` at `2fa1eb3e`, the proposal, both designer/attack briefs, FRESH-START, the two round-2 reports, audit F1–F4 and season-bound table, the blocked `sr-state` diff/contract, relevant earlier probes, callers, scoring/reconcile and activation checklist. Also consulted the required shared project context (`project_bts.md`, `project_bts_streak_saver.md`); its old claim that saver has no effect below 10 is superseded by the actual policy/code.

Ran 63 existing tests across `tests/test_contest_state.py`, `test_contest_state_best_trust.py`, `test_contest_state_corrupt_precedence.py`, `health/test_contest_state.py`, `test_saver_state.py`, `test_decision_saver_fallback.py`: **63 passed**.
Ran `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .codex-review/srd2-scratch/probe.py`; assertions passed. Script, `results.json` and `run.log` are in that scratch directory. The probes use real loaders, CLI writers, saver transitions, policy artifacts, alert policy and scheduler loop, with clocks/transports isolated.
The five fetch classification rules are separately transcribed in scratch to evaluate the proposed design; those results are **not** a claim that the design is implemented. All endpoint responses and 2027 outcomes below are synthetic. No contest, box or network service was contacted; no tracked implementation file was changed. This is a design review, not a deployment gate or a measurement of MLB rollover behavior.

## Findings, most severe first

### F1 — The advertised universal manual recovery does not bypass corrupt `streak.json`

**Reachable sequence:** restore a healthy account observation, but a corrupt local `streak.json` (one byte `0xff` suffices). Run the prescribed `set-contest-streak --streak 0 --best-streak 0` with valid 2027 source date/expiry. The command succeeds. Every live decision still raises `UnicodeDecodeError`.
**Ran:** `manual_not_universal_recovery`. The real loader reads `load_streak` / `load_saver_available` before selecting the manual observation (`contest_state.py:293–297`; `picks.py:514–531`). This is not one of CS1's date-named pick-file scans or `_parse_state_file` repairs. A non-object local state similarly fails through `.get`.
**Design behavior:** S1/S9/S13 claim a manual file restores picking; CS2 specifies season migration but never a malformed-model-state contract. The documented command cannot repair this state, and its exception is outside the scheduler's `ContestStateError` recovery.
**Should happen / smallest change:** a valid contest/manual observation must not depend on parsing research diagnostics. Report unavailable model diagnostics separately. If model state is the selected fallback, fail explicitly with a concrete quarantine/repair instruction. Add the corrupt-model-plus-valid-manual integration case to CS1/CS2; do not silently invent a model streak.

### F2 — `proven_season` transfers confidence between observations that prove different things

Three concrete accepted transitions refute “none causes a silent stale-streak pick” (§4):

**A. Temporary zero → carried counter.** March 22: rounds list March 24, profile is `0/0, predictions=[]`; classify `empty_reset`, proven 2027. March 28: same account returns last season's `active=7, best=18, predictions=[]`. `counter_only_unsettled` inherits 2027 from that zero response. The opening override has expired. The true streak can be 5 (for example, an opening Pass, then double/single/double hits); the proposal uses 7. This counter even fits the season's possible advancement bound.
**Ran:** `empty_reset_launders_old_counter`: transcribed rules accept 7; actual tracked policy at .75/.74 selects **skip at 7 versus double at 5**. The proposal acknowledges this flap in S3 but calls an optional, already-expired override a mitigation. It is a counterexample, not a closed state.

**B. A current prediction does not date global counter fields.** March 28: `predictions=[{roundId:9004,result:hit}]` maps to March 27, while `activeStreak=7, seasonBestStreak=7` still carry 2026 values. The true streak can be 1 after an earlier miss. Both 7s fit the calendar's maximum, and real parsing/building accepts the response. The proposed `current_settled` rule calls both counters proven 2027. With larger carried bests this can also falsely promote the saver. A backend can roll its ledger and counters independently, or change `activeStreak` to a carried field while adding a separate season field. Nobody has observed that contract across seasons.
**Ran:** `current_row_is_not_counter_season_proof`. Also `same_or_past_date_id_collision`: reusing an old ID for today's/past 2027 round passes both `derive_source_date` and the proposed future-date guard. The proposal discusses reset-to-1 and a September collision only. Reuse of the same ID range is another possible endpoint behavior; this is not a claim MLB will do it.

**C. Old settled anchor → arbitrary weaker snapshot.** A trusted May 31 observation exists; a June fetch has unmapped predictions and stale cached counters from 2026. `unmapped_settled` carries `last_proven_source_date` and stamps the *new* counter proven 2027. The non-clobber rule rejects only `prior_settled_only`, so it does not prevent this overwrite.
**Ran:** `old_settlement_proof_launders_weaker_snapshot`. Preserving the prior observation on a coverage failure would retain a known current-season value; substituting a different unbound value is a stronger, unjustified operation.

**Should happen / smallest change:** distinguish “what this response contains,” “last accepted counter observation,” and “the basis for accepting this response's counter.” Calendar membership, empty responses and prior-file season are observations, not counter-season attestations. Do not automatically chain A/C. For B, acceptance needs an observed, owner-approved endpoint interpretation or explicit current-state operator observation; merely renaming the field cannot establish it. Preserve ambiguous attempts separately, including the round-ID/date mapping actually used, rather than overwriting accepted evidence. A sanity bound helps detect some contradictions but does not resolve two plausible counters.

Also distinguish an absent `predictions` key from an empty array. The current CLI defaults absence to `[]`; `missing_array_is_silently_empty` runs that path and the proposed rules classify a `0/0` schema with the array missing as `empty_reset`. A missing field must be a schema/coverage failure, not positive reset evidence.

### F3 — Same-season restores defeat both freshness and saver safety

**Streak sequence:** restore March 26 account state (`streak=3`, source March 25) on June 1 without root pick history or fresh fetch status. The true account can now be 13 or zero. Year and settled-round membership still match.
**Ran:** `old_same_year_backup_is_fresh_with_no_history`: real decision returns **3/fresh**, best trusted, health **[]**. The proposal explicitly retains freshness-by-local-pick-gap and S6's “zero newer picks → fresh.” Fetch failure/status loss does not invalidate this file. This is still an inference from missing local files, now about freshness rather than rollover.
**Should happen:** distinguish season membership from observation age and post-restore validation. Mark this unknown/stale and require a fresh accepted observation or explicit override before declaring readiness. At minimum show age and fetch failure independently of pick history; the owner must choose any allowance for operating on last-confirmed state.

**Saver sequence:** save a 2027 `not_earned` backup; reach 12; earn the saver and consume it; restore the backup. A valid current-season fetch with `best=12` now promotes restored `not_earned → active`. Alternatively restoring an earlier `active` flag directly resurrects it. The snapshot's season is correct in both cases.
**Ran:** `saver_same_season_restore` executes guarded transitions, restores bytes and runs the real auto-earn hook: **used → restored not_earned → active**. On June 1 at streak 13/.80 the actual policy chooses **single with the resurrected saver versus skip with used**.
**Design behavior:** S12 protects only missing/wrong-season flags. S6/S7 assume V2 is sound; CS4's current-settled requirement still allows this promotion.
**Smallest change:** the restore procedure must declare saver continuity unverified and require an explicit owner reconciliation before trusting/promoting that flag. Keep unknown conservative. An audit log restored from the same older snapshot cannot prove nothing happened after it. Guard `not_earned → active` on continuity, not just a season-qualified best. A fetch gap long enough to contain earning and consumption has the same issue.

### F4 — “Zero for the opening days” has no safe renewal or late-use contract

**Reachable sequences:** omit the setup; set it before the delayed opener so it expires too early; run the recovery command on Day 5; rerun it twice; renew `0/0` every 48h while endpoints remain ambiguous; restore a legacy non-expiring manual zero. All are permitted by the existing CLI. A 72h cap limits each invocation, not repeated resets or a first invocation made late.
**Ran:** `zero_override_renewal` writes zero on March 24, March 28 and April 3 successfully. On the tracked policy, zero chooses double while 7 or 9 chooses skip at .75/.74. Equal actions at 0–5 do not make the streak observation true, and are not a contract for a future approved 2027 policy.
The proposal says to extend the override until reset/settlement but never says to update it to the **actual current streak/best**. “1 missed Opening Night pick” is not a bound: continued endpoint ambiguity plus expiry can refuse every later date until intervention, or repeated zero renewal can drive wrong picks for the whole opening stretch.
**Additional production exceptions:** §2 explicitly accepts an undated auto “bare fixture” and a current-year legacy manual without expiry. `legacy_fixture_exception_is_live` confirms the real required loader accepts `{"active_streak":14,"best_streak":18}`. A 2026 legacy file restored without dates is observationally identical to that fixture. Production cannot identify fixtures by missing metadata.
**Live-mode gap:** `require_contest_state=True` is not an invariant of live callers. Both scheduler/orchestrator helpers default false (probe `live_requirement_default`), and `cli.run` uses the optional loader before its posting path. No claim here about the absent box's actual TOML. The design must bind enforcement to every authorized delivery path or explicitly verify the required configuration at activation.
**Smallest change:** define a one-contest-date observation with correct current values, explicit expiry before its reuse for a later decision, and an idempotent renewal that never silently reasserts zero. Reject undated automatic state for live use; migrate real legacy files explicitly and pin tests to real metadata. Ship the operator runbook with the guard, not as an optional CS6. The actual checklist's **B5 adds 2028 season-end dates**; B8 is the streak readiness row. Fix those references and specify the commands there.

### F5 — A successful override write is not a same-day retry, and the first alert may not be T−180

**Reachable sequence:** singleton synthetic 19:10 ET game, box offset 60. All three planned cascades have no predictions because the teams have no prior lineup and the first lineup is late. The operator writes a valid override at 18:40, before the 19:05 cutoff. A lineup can now be available; the daemon is already idling until tomorrow.
**Ran:** `override_does_not_wake_main_scheduler` drives real `run_day` and `run_single_check`: cascades at **16:10, 17:10, 18:10**, zero decision-loader calls, zero immediate contest-refusal dispatches, no pick. The real contest health function returns missing-state CRITICAL at EOD **18:10**; the probe substitutes the health runner to record this, without sending. Writing the override during idle causes no further cascade.
**Why:** `orchestrator.run_and_pick:273–274` returns before loading state when predictions are absent; final fallback requires an existing pick (`scheduler.py:3095`). `main` has no no-pick extra-attempt loop. FRESH-START explicitly leaves that work on `sr-lastcheck`.
**Design behavior:** §5 promises immediate T−180 refusal DM and rescue whenever the override is set before T−5, citing a T−20 attempt outside CS1–CS5. Even with that future attempt, an override after the last attempt cannot wake it. A prediction cascade also needs time before cutoff.
**Should happen / smallest change:** perform state readiness checks independently of predictions; make recovery include an explicit re-evaluation command or a supported retry/wake path, with a truthful latest-start deadline. Name the scheduler dependency before claiming automatic rescue. Do not promise pickup from a file write. This is a recoverable lost-day failure, not proof of a permanent technical deadlock.

### F6 — Previous health findings are not fully closed, and alerts are overstated

**Remaining exact case:** a season-accepted counter-only/unmapped observation reaches 11 while saver remains `not_earned` (no `current_settled` fetch to promote it, or a crash after writing the observation). The design correctly maps it false, but only specifies warnings for `uninitialized`. Test 10 checks false at 12 without requiring the missing warning. Carbon r2 B3's `not_earned` 10–15 case remains open.
**Channels actually present:** `contest_state` WARN is in neither attention source set (`health/attention.py`), so S1b/S12 warnings are journal/log warnings, not a DM. This is especially material to Alternative B's proposed “WARN DM.” Normal EOD health runs after the day's polling/finalization, not at a fixed 22:30. `day-outcome` is **08:30 next morning for yesterday**, not 23:00; its receipt failure token is `entry_unverifiable`. A clean `entered`/`deliberate_skip` does not diagnose a wrong streak or saver.
**Ran:** `existing_alert_transport`: WARN produces no DM; two different same-day CRITICAL messages under source `contest_state` produce only one sent message because dedup is by source, not body. An earlier state alarm can therefore suppress the later actionable refusal wording. The scheduler-loop probe above demonstrates variable EOD timing.
**Smallest change:** health and selection must share the effective state classification; warn specifically on `not_earned` at 10–15 and on unverified saver continuity. Name log versus DM deliberately; route readiness/override problems through an existing attention/incident mechanism if human action before cutoff is required. Give distinct incidents when a new refusal needs new instructions. Do not claim an alert proves delivery to the operator.

## Decision-table attack coverage

These are attacks on each row, not a claim to have executed an implementation of that row:

| Proposal row | Assessment / counterexample |
|---|---|
| S1a | Correct for a genuinely observed zero/current saver; wrong if the same command is issued late after advancement (F4). |
| S1b | Explicitly chooses a false zero; WARN is logged at variable EOD, and expiry can expose F2 or refusal rather than cure it (F4/F6). |
| S2 | Temporary/default zero or cleared profile after a restore is indistinguishable from reset; preserve uncertainty rather than overwrite a known non-zero observation (F2/F3). |
| S3 | Counter-only inheritance fails on a reset flap; at ≥10, `not_earned` also needs an actionable warning (F2/F6). |
| S4 | Refusal plus correct manual observation is defensible; zero is not a valid universal mid-season recovery, and a file write alone cannot retry (F1/F4/F5). |
| S5 | Defensible refusal with operator recovery; endpoint lag can continue beyond ten days, so no one-day cost bound (F4 and trace below). |
| S6 | A source year does not prove counter semantics or freshness; restored V2 may be spent (F2/F3). |
| S7 | Historical proof does not season-bind the next counter; preserving the accepted sample is different from accepting a new weak one (F2C). |
| S8 | Correctly avoids fabricating zero on a bare restore; recovery needs correct saver state and an actual selection retry (F1/F3/F5). |
| S9 | Encoding/shape fix is sound for date-named history; it does not cover corrupt `streak.json`, or make missing/unreadable history freshness proof (F1/F3). |
| S10 | Explicit season reset and applied-date reset are sound; test actual tagless 2026 files and repeat scoring, not only an already-tagged fixture. No new counterexample to that specified reset. |
| S11 | ET/target-date binding fixes timezone selection, not arbitrary host skew. The classifier can assign `max(rounds_seasons)=2027` to `prior_settled_only` on Dec 31, 2026 with no settled rows; specify rejection precedence over `proven_season==target_year` when later read in 2027. This forward-year case is in scratch results. |
| S12 | Conservative unknown is sound; restore of a valid-looking older flag escapes it, and WARN is not DM (F3/F6). |
| S13 | Active current-date manual bypassing bad auto is sound; malformed model and undated legacy acceptance violate its blanket recovery/safety claim (F1/F4). |
| U1 | `current_settled` with only old-year round dates cannot arise from a consistent classified fetch. But `best < active` is reachable after valid-JSON corruption: fetch 13/20, change one byte to 13/10, then load it. The real reader still accepts streak 13, marking only best untrusted (`u1_bad_best_reachable_after_corruption`). Validate stored fields too; fetch validation is not a disk invariant. |

The claimed five-way partition also overlaps: old-only rounds plus unmapped settled rows satisfy both `unmapped_settled` and `prior_settled_only`. Specify precedence. A current mapped row plus a later future mapped row makes `current_settled` true while `source_date=max(...)` is future; define which dates populate the accepted witness. Counts/earliest date alone are insufficient to audit these mappings.

## Opening Night and each of the next ten days

Construct endpoint behavior **L**: 2027 rounds publish, but profile returns only unmapped 2026 settled rows and a carried counter throughout April 3. This might be a lag, retained history or changed schema; no timing bound is known. A manual zero is installed March 24 before checks, expires March 26 before checks. Predictions exist for this trace.

| Date | Improved A, no renewal | If its zero recovery is repeatedly renewed: true pre-pick streak on an all-hit trace |
|---|---|---|
| Mar 24 | Manual zero; singleton single | 0 |
| Mar 25 | Manual zero; double at Q0 | 1 |
| Mar 26 | Refuse unproven auto | 3 |
| Mar 27 | Refuse | 5 |
| Mar 28 | Refuse | 7: true-state Q0 skip, override doubles |
| Mar 29 | Refuse | 9: true-state Q0 skip, override doubles |
| Mar 30 | Refuse | 11 |
| Mar 31 | Refuse | 13 |
| Apr 1 | Refuse | 15 |
| Apr 2 | Refuse | 17 |
| Apr 3 | Refuse | 19 |

The renewal column is a constructed trace under the **wrong zero-based choices**, not a simulation of the correct policy or observed results. Each later date requires another real state observation if MLB remains ambiguous. If the operator is absent, all nine later dates can be lost; the design has a human recovery, not an automatic bound on that cost.

Other endpoint behaviors and their full-window consequences:
- **Rounds remain 2026, profile resets:** S5 refuses all eleven dates without manual coverage; neither A nor B can assume eventual rollover. Actual manual state permits picks on each date.
- **Zero/empty profile flashes, then carried counters reappear:** S2 then S3 can authorize stale non-zero picks on any date after the opening override expires (F2A), including all remaining dates if responses persist.
- **Current predictions appear but counters retain old meanings, or IDs map old rows to already-past current rounds:** S6 can accept the wrong counter every remaining date; first settlement does not resolve the counter contract (F2B).
- **Empty/unparseable rounds or expired cookies throughout:** preserve accepted observation and record failed attempts; A refuses after override expiry if none was accepted. Alternative B also refuses and adds no recovery. An old same-year backup must not turn this into fresh state (F3).
- **Normal reset and correctly seasonal counters:** A works for each date once that interpretation is established. A temporary missing ledger is not itself evidence the counter changed seasons; use the accepted interpretation and explicit uncertainty rather than a fresh “proof” each day.

## Evidence and operator obligations

For every evidence source: profile can be partial, cached, empty or change semantics; rounds can be stale, multi-season, ID-reused, truncated or future; a prior accepted file can be old/restored; status can be missing/restored and overwritten by success; manual TTL depends on host clock and can be renewed wrongly; saver flag/audit can both precede consumption; model/applied-dates and pick files describe recommendations, not actual account entries. Entry receipts prove their bounded entry claim, not counter/saver state. None may silently substitute for another.
Persist the attempted observation and its classification, the mapping used, and acceptance/rejection reason even when it cannot replace accepted state. The current CLI writes observation, then status, then ledger, then saver: a crash or write failure between these can leave a failed run's new observation beside an old flag. Specify how readers expose that mismatch; a successful-file write is not a transaction across them.

| Operator mistake | Required behavior / present gap |
|---|---|
| Setup skipped | Readiness alert before selection opportunities, independently of predictions; no fabricated current streak. Existing early alert is conditional (F5). |
| Setup late | Use actual current values and explicitly run/wake selection while its deadline remains; writing a file does not do that. |
| Done twice | Repeat the same date/observation idempotently; never reset an earned/used saver or extend a frozen zero into another day. |
| Wrong date or delayed opener | Reject wrong-target/future observations; expire by intended contest date, and refresh the confirmed opener operationally. A multi-day TTL alone is insufficient. |
| Left active / renewed | Notify before the next decision and require current observed values; do not rely on early-table action equivalence. |
| Restore or downtime | Reconcile current account and saver continuity; require an explicit restoration acknowledgment. Current-season metadata alone is insufficient. |

## Alternative I would build

**C: explicit daily observations during the boundary, then a verified seasonal counter contract.** This improves the lead's operator option without inventing an automatic reset. Keep evidence at the fetch source, a separate last accepted observation, and the existing conservative saver principle. An operator observation covers one contest date, with actual streak/best and explicit saver state/uncertainty; opening zero is one instance. Hand back to automatic operation only after the endpoint interpretation is established from owner-authorized observations. Do not claim that arbitrary endpoint responses can prove their own semantics.

| Input | Streak / policy saver | Picking, notification and recovery |
|---|---|---|
| Valid same-account/season/date manual observation | Observed streak; active=true, used/unknown=false, verified not-earned below 10=true | Normal selection; show expiry and selected source. Local model corruption is diagnostic only. |
| Accepted automatic observation under verified counter contract, within explicit freshness allowance, continuity intact | Accepted counter and separately verified saver reserve | Normal selection; report any mapping/coverage degradation without replacing accepted evidence with weaker data. |
| New ambiguous response, but last accepted state exists | Keep that historical value as historical; do not label the new counter season-proven | Alert before a decision. Operate only under the owner's explicit stale-state allowance; otherwise current-date manual observation + immediate retry. |
| Old-season/undated/expired/restored-unvalidated evidence only | Unknown, no invented zero | Refuse live selection with executable manual recovery and a scheduled/operator retry. Research may use labeled replay. |
| Saver missing or continuity unverified | Conservative false, with explicit uncertainty | Picking can continue on a valid streak; actionable notification and owner initialization, not auto-resurrection. |
| No predictions yet | State readiness still checked | Keep the separately reviewed late-lineup retries; state recovery must not depend on a forecast existing. |

Costs if wrong: **raw lead option** silently trusts a newly fetched carried counter when setup is missed; **A as proposed** adds false proofs, stale restores and potentially many missed dates; **B** adds false zero on a mid-season truncated rounds calendar and still refuses on cookie/rounds failures. **C** adds operator burden and can lose dates if nobody responds; a wrong operator observation still gives wrong picks. C makes that obligation explicit and testable. No design can distinguish a real seasonal 7 from a carried 7 using otherwise identical unqualified responses. The lead/owner must choose how to operate through that uncertainty.

## Earlier counterexamples and false greens

| Earlier finding | Closed by this proposal? |
|---|---|
| Codex r2 B1; Carbon r2 B2 | Original local-pick season inference removed. Restored no-history and late-profile risks remain through freshness/weak-proof inheritance (F2/F3); prolonged refusal needs the real recovery in F4/F5. |
| Codex r2 B2; Carbon r2 B1 | Yes for truly `uninitialized`: always false. Not for restored current-season `active`/`not_earned` (F3). |
| Codex r2 B3 | Yes for the specified date-named history UTF-8 error via CS1; add corrupt model-state recovery (F1). |
| Codex r2 B4; Carbon r2 B3 | Manual source-year rejection changes the old allowed contract rather than merely fixing health. Health must agree with that choice. `not_earned` at 10–15 warning remains unspecified (F6). |
| Audit F1 | Correct reserve mapping below 10; keep it. |
| Audit F2 | Current-year saver readers/dashboard fix is sound as specified. |
| Audit F3 | Exact unmapped-old-rows case refused, but reset chaining/mixed-field acceptance can still import old counters/best into saver promotion (this report F2). |
| Audit F4 | Normal first-fetch missing-date exemption is addressed in S2/S3. Local counts cannot prove reset/freshness, and no-prediction alerts still differ from the promised timing. |

False greens to prevent: tests 1–3 only test *absence* of a current row, never a current row with carried counters; 5 blesses inheritance without a regressing/cached response; 6 omits the counter-only flap that test 8 explicitly accepts; 9 tests only future collisions; 10 omits the missing saver warning; 11 tests loss of the flag, not restoration of an older valid one; 13 tests a loader/TTL, not a day loop, repeated renewal or source-set DM suppression; 14 omits corrupt `streak.json`; 15 must use actual legacy bytes and repeated scoring; 16 does not test a next-season calendar published before the year changes. A green table transcription is not evidence those observations mean what the table claims.

## States I tried and could not break

- Required missing-state refusal plus a correctly dated active manual observation, with readable model diagnostics: CLI/loader recovery works; an active override is selected before corrupt auto parsing (existing tests passed).
- Unknown saver remains false; `not_earned` below 10 means reserve; active/used map true/false. The actual MDP transitions support these meanings. No argument for restoring the old optimistic uninitialized rule.
- Rejecting empty rounds while preserving an accepted observation is preferable to replacing it with null coverage. Its age/continuity still needs F3.
- Explicit season tagging/resetting the local model state and current-target-year saver/dashboard binding address their stated cross-season bugs. ET conversion addresses UTC/ET disagreement; no claim that it establishes a correct physical clock.

## Needs the owner, not another inference

The lead can select the design; Eric must accept the daily-observation/restore duties and the cost of a missed date versus any permitted stale-state operation. Requests to the real account, inspection of its March counter semantics/round mapping, and confirming whether a saver was consumed remain owner-authorized work. No such request was made here. No code-only design can recover a saver's post-backup consumption from a wholly rolled-back local snapshot. The report proposes no new outcome study or account request as a prerequisite to this design decision.
