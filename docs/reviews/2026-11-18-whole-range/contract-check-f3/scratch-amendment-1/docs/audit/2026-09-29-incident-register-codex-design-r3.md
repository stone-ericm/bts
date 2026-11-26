## Verdict

**SIGN WITH EDITS.** Apply findings **1–7** below before freezing the implementation plan. These are bounded schema, predicate and fixture edits; the exposure boundary, discovery routes and certificate structure can stand. The most consequential remaining defect is §6.5: an implementation that returns `unknown` for everything can satisfy the stated discovery gate. Several purported exact field paths also disagree with their writers.

Reviewed baseline: **e05641fa084235fa5cc6e50e587051c826c0788c**. HEAD matched; the design and rules files had no diff against that commit. **D:line** means `docs/superpowers/specs/2026-09-29-incident-register-design.md` at that baseline. I read the archived r2 findings and checked the relevant source writers, consumers and installed pytest implementation. The local `section6.txt` SHA-256 matches **b7986f692cc2967368ae5b9a5af115a1191a9ccebeddd693af0f98cb2ccd7a9e**. The full-page acquisition and bundle custody are described by `docs/audit/2026-09-29-incident-register-evidence/rules/PROVENANCE.md:3–6`; they were not independently verified here.

This is a source-based design review. No production/runtime data, SSH, network, tests or mutants were used. Counterexamples below are source-derived or synthetic specifications, not measured incidents. Only this report was written. This verdict does not authorize a box run or production repair; D:150 retains those gates.

## r2 findings status

| r2 finding | Status | v3 evidence and remaining work |
|---|---|---|
| 1 — exposure leaks | **RESOLVED** | D:41–43 requires typed leaves, closed reason categories and opaque unknown text; D:53 excludes health body/error text; D:62 restricts outcome disclosure; D:66 covers error output and metamorphic leakage. The old free-text/viewer counterexamples are prohibited. Correct the field mappings in finding 1 and test the complete output envelope under finding 2. Implementation safety remains for code review. |
| 2 — invariant inputs / ledger schema | **PARTIAL** | D:48–62 adds identities, timestamps, exact ledger columns and accepted-output binding. But several state paths do not exist, the operational ledger projection excludes necessary day rows/counts, and some eligibility inputs remain unspecified. Findings 1, 3, 4 and 6. |
| 3 — silent missed picks / unknown handling | **PARTIAL** | D:73–90 adds independent V1b, four-valued results, epochs and explicit 9/03 limits. Remaining predicate errors include cutoff equality, both slots unknown, unresolved non-null shadow results and absence of optional log messages. D:96 also permits a vacuous discovery gate. Findings 2–5. |
| 4 — causal path witness | **RESOLVED** | D:109 requires the same production invocation to connect the mutant and symptom, and distinguishes component fixtures. Calling the entry point and helper separately in one test no longer qualifies. Finding 7 clarifies how to certify a missing event without inventing a boundary call. |
| 5 — hybrid historical harness | **RESOLVED** | D:105–107 pins deployed refs, the complete fix set, executed files, clocks and dependencies; audits transitive harness changes; and labels unsupported equivalence as semantic regression replay. Whole-directory replacement also closes the obsolete-file hazard. |
| 6 — excluded runtime closure / late repairs | **RESOLVED** | D:11 separates occurrence and evidence windows; D:30 includes commits through the baseline; D:71 follows the versioned command/import closure without directory-name exclusions. |
| 7 — expected-failure specificity / phases | **RESOLVED** | D:117 requires the declared bad signature, ordinary assertions for other mismatches, call-phase/location verification, no imperative xfail, strict XPASS and setup/teardown meta-tests. This addresses pytest 9.0.2's conversion semantics. |
| 8 — Pass clause C / archive | **RESOLVED** | D:124–137 supplies the excerpt and full digests, explicitly requires `void` for complete pre-suspension AB/no-hit cases, and records conflicting existing coverage. The excerpt digest verifies; no contrary condition appears in the supplied section. |
| 9 — dispositions / independent axes | **RESOLVED** | D:13,19,22,94 separates delivery/objective/stream, evidence strength and disposition, with the same evidence standard for every discovery route. |
| 10 — reviewed execution identity | **RESOLVED** | D:25,102,149–150 binds code/schema/templates/X-20/input identities, renewed review after changes and per-version extraction directories; post-freeze latest state cannot establish earlier coverage. Finding 2 strengthens the test that this gate names. |

## New findings

### 1. SHOULD — replace nonexistent state paths with writer-backed mappings

**Design:** D:45–60.

With D:41's missing-field rule, these errors become nulls and `schema_issue` counts even for healthy, correctly formed artifacts. They affect timelines and joins; they are not evidence that the underlying service lost the fields.

| Declared path | Actual writer / consequence | Smallest replacement |
|---|---|---|
| `fallback_refreshes[*].at` | The writer emits `started`, `finished`, `duration_sec`, not `at`: `src/bts/scheduler.py:2927–2931`. | `fallback_refreshes[*].{started,finished,duration_sec}` |
| Lineup-evolution `ts`, flat `batter_id`, `game_pk` | `src/bts/picks.py:381–399` writes `captured_at`, `date`, `run_time`, then `primary` and `double_down` objects. Flattening without a slot loses the join. | `captured_at`, `date`, `run_time`, `primary.{batter_id,game_pk}`, `double_down.{batter_id,game_pk}`; flattened rows retain a `slot` enum. |
| Entry `escalations[*].at` | It is a list of tier strings. `src/bts/cli.py:1792–1801,1827–1834` consumes `initial`, `t30`, `t15`; it has no per-escalation timestamp. One send can consume several tiers. | `escalations[*]` as enum `{initial,t30,t15}`. Keep `checked_at` separate; never turn it into individual send times. |
| Backup `last_attempt`, `last_success` | Source writer `src/bts/data/backup.py:201–207,274–286` emits `started_at`, `finished_at`, `last_success_at`. The last success survives failures (`:145–156`). | Per set: `started_at`, `finished_at`, `last_success_at`, `ok`, `has_error`, `has_forget_error`, `has_last_success_snapshot_id`. Map the last three to presence of their named source fields; never copy their text. |
| Contest-fetch `status`, unspecified timestamps/category | The failure writer has `last_error_at`, `last_error_category`, `last_alert_at_by_category`, `last_alert_at` (`src/bts/cli.py:1593–1605`); success has `last_success_at`, clears `last_error` and preserves cooldown stamps (`:1952–1965`). There is no raw `status`. | `last_success_at`, `last_error_at`, `last_error_category` as enum `{actionable,transient,rate_limited}`, `last_alert_at`, `last_alert_at_by_category.<category>`, `has_last_error`. An absent legacy field stays unavailable; no synthesized success from absence. |

Also spell out these already-intended mappings: `has_degraded_reason := policy_decision.degraded_reason is non-null` for a pick; attention fields are `sources.<normalized_source>.{last_seen,streak}`, with `last_message` dropped (`src/bts/health/attention.py:124–132`); capture status is `capture_status.json.status`, its observation time is `generated_at` (`scripts/live_forward_capture_once.py:297–303`), D8 publication time is `research_capture.accepted.json.published_at` (`:1224–1230`), and `run_kind` comes from `manifest.json` (`:911–912`). For sidecar validation, emit only the verdict component, never the accompanying explanation string (`:925–997`).

**Verbatim addition after D:60:**

> Each row has a source-path → output-path mapping at each deployed schema era, tested with producer-shaped synthetic input. The mappings above are source paths unless explicitly marked derived. No timestamp, status or receipt is synthesized from an absent field. Nullable legacy fields have explicit applicability rules. A valid fixture from each supported era must extract its expected non-null operational values, with zero unexpected schema issues.

No actual outcome/streak/probability leaf needs to be added to fix these paths.

### 2. SHOULD — §6.5 must require detection, through the real extractor pipeline

**Design:** D:66,96,149–150.

**Counterexample:** a detector returns `unknown: required input absent` for every date. Incident cases satisfy “candidate (or a justified unknown)”; every control satisfies “no candidate.” The path errors in finding 1 can supply that justification. A test starting from hand-built projected rows can also miss a broken raw-source parser entirely. D:73 correctly says unknown is not success, but D:96 does not enforce that in its acceptance test.

**Replace D:96 with:**

> Pre-run discovery tests execute producer-shaped synthetic raw artifacts through the reviewed extractors and then the invariants. For complete-input cases, 7/16 and 8/13 must produce missing-finalization/delivery candidates, 8/30 a late-delivery candidate, 7/12 a restart-burst candidate, and 7/08 an entry-evidence-gap candidate. Each fixture pins the expected invariant id, result and reason; `unknown` fails these complete-input cases. Separate cases remove or corrupt one required source and must return `unknown` with that exact dependency named. Evaluable controls pin `satisfied` or `not_applicable` for the check under test; merely returning no candidate is insufficient. Separate controls for producer eras with no necessary witness pin `unknown` and the precise observational limit, such as healthy EOD with no mandatory completion marker. 9/03 has a separate assertion that metadata alone cannot distinguish the wrong skip. Include a detector that always returns unknown as a rejected negative control. Every required fixture/node and expected output row is counted; missing rows or nodes fail the gate.

Then add to D:66:

> Leakage comparisons include event existence/count, event kind, template id, reason category, schema issues and diagnostics, not just captured values. Vary terminal hit/miss/void values while holding the declared resolution state and other operational facts fixed; test pending/terminal classification separately. Integrity fixtures recompute their bindings when varying otherwise valid payloads, so a deliberately broken hash is tested as integrity failure rather than outcome leakage.

This makes the existing gate meaningful without adding another approval stage. Historical real-data gaps may still be `unknown`; that does not justify an all-unknown complete synthetic case.

### 3. SHOULD — supply the day/schedule/epoch inputs, and fix the exact cutoff

**Design:** D:48,60,62,73,77–82,85–86.

**Evidence and missing inputs:**

- V1 needs game days and explicit skip/unfinalized/unobserved day rows. D:62 says “per selection” and omits `scheduled_games`. The ledger actually has that field (`scripts/audit/season_ledger/compile.py:35–38,171–175`) and emits non-selection day rows (`scripts/audit/season_ledger/rows.py:108–114,183–221`). A selection-only projection discards exactly the rows V1 names.
- The persisted scheduler schedule is initialized from the fetch (`src/bts/scheduler.py:2608–2616`); later updates change `lineup_confirmed`, not the game start (`:2718–2719`). It is not an independent evolved-start source. July 16's historical memo expressly describes the stale morning schedule (`docs/optimization-ideas.md:25–36`). A newer observation must actually be retained and joined, or the evolved-cutoff claim is unknown.
- Postponement controls and V9's live polling window need game-status observations. Neither `Pick.game_time` nor `SchedulerState.games` contains game status (`src/bts/picks.py:189–201`; `src/bts/scheduler.py:351–360`). D:60 also omits the named activation flags behind V6/V8: `health_checks.enabled`, `scheduler.shadow_model`, `scheduler.live_forward_capture_on_lock`, and the runner's `--capture-research-on-skip`/candidate configuration (`src/bts/scheduler.py:2537–2538,3146–3147,3184–3186`; `scripts/live_forward_capture_once.py:67,683,1299`).
- V3 has a DM receipt hash but only presence of a public URI. `DailyPick` actually stores that URI (`src/bts/picks.py:211–215`). Presence cannot distinguish two retained public receipts; a log event id also needs an explicit duplicate-observation rule across overlapping exports.
- **Cutoff equality is a concrete false green:** D:79 permits send time equal to first pitch minus five minutes. The serving guard refuses `now >= cutoff` (`src/bts/scheduler.py:1006–1011`, also `:1047–1048,1075–1076`). The synthetic T−4 case will not test this boundary.

**Minimum edits, verbatim:**

1. In D:62 replace “Operational projection per selection” with “Operational projection over all ledger row kinds, including day rows; selection-only fields remain null on day rows”; add `scheduled_games` to its list.
2. Append to D:60:

   > Project the versioned health/shadow/live-forward enable flags and nonsecret runner arguments needed for eligibility, including the research-capture flag and pinned candidate identity. Unknown activation intervals remain unknown. Schedule/status evidence has a separate typed projection: date, game id, regular-season flag, scheduled first pitch, observed-at time with its provenance, and closed game-status category. Register the retained source in B1/B2 before reading it; no new fetch is implicit. A dated snapshot with no observation time cannot certify the start/status at an earlier send.

3. Add `bluesky_uri_sha256` beside `has_bluesky_uri` in D:48; append to V3:

   > Deduplicate observations of the same receipt across files/log exports before counting sends. If only message observations exist and cannot be reconciled to distinct sends, return unknown.

4. In V2 replace `≤` with `<` and “send after cutoff” with “send at or after cutoff.” Add complete synthetic cases immediately before, exactly at and after cutoff.
5. Append to V2/V5:

   > Record the cutoff's source and observation time. A pick or morning schedule supports only a comparison against that recorded schedule; it does not certify an evolved start. Missing required schedule/status evidence yields unknown for that comparison. V5 counts executed checks, not `runs_completed` entries marked skipped. V9's polling-window check likewise requires an evidenced active polling interval or reports unknown.

### 4. SHOULD — research resolution and D8 eligibility need the actual contracts

**Design:** D:58–59,85,88.

**Resolution:** “result non-null” is not terminality. `DailyPick.result` includes `suspended` and `unresolved` (`src/bts/picks.py:223`); the real shadow consumer accepts only `hit`, `miss`, `void` (`src/bts/cli.py:2171–2195`). Skip-shadow uses **`shadow_pick_result`**, not `result` (`src/bts/skip_policy_shadow.py:84–99,255–271`). Reading a generic `result` either loses all skip-shadow resolutions or accepts nonterminal shadow states.

**D8:** it triggers on a **provisional** current skip with no scoreable decision, before the start cutoff (`scripts/live_forward_capture_once.py:1052–1062,1094–1117`). Its sidecar explicitly records `trigger.kind = provisional_scheduler_skip_state` and defers final classification (`:1200–1206`). Therefore a valid early D8 capture followed by a later production pick is not, by itself, a trigger-contract violation. D:88's blanket “D8 capture on a non-skip day” loses this temporal distinction. Conversely, a final skip first observed after cutoff did not provide an eligible capture opportunity. An end-of-day `has_final_skip_candidate` alone cannot decide either case.

The validator also requires an independent `candidate` argument and distinguishes valid from incompatible (`scripts/live_forward_capture_once.py:925–927,957–960,992–997`). Passing the candidate taken from the artifact itself would erase that check. Its returned verdict tests integrity/acceptance, not the missing historical trigger proof.

**Replace D:59 with:**

> Shadow files: presence, date and `resolution_state` ∈ {terminal, pending, invalid}, derived from `result`. Skip-shadow files: the same projection derived from `shadow_pick_result`. Terminal means exactly hit/miss/void; recognized nonterminal values remain pending; malformed or unrecognized values produce invalid/schema_issue. Never emit the result value. Each stream's age threshold and applicable producer version are pinned before the sweep.

**Append to V8 and replace the D8 clause in V11 with:**

> D8 eligibility is evaluated at the pre-cutoff trigger interval: research capture enabled, current nonempty skip candidate, no scoreable production decision, and valid first-pitch timing. A final skip alone does not establish this opportunity. Project only typed trigger metadata needed for that check: trigger kind, candidate-present boolean, decision action/scoreable boolean, and observed/start/completion/publication times; do not copy the candidate payload. The validator's expected candidate comes from pinned job configuration, independently of the sidecar. A later production pick is recorded as provisional-to-final reclassification; it is a candidate only if an explicit capture-time or stream-isolation contract was violated. Missing trigger history yields unknown. Integrity verdict and trigger eligibility are reported separately.

Add controls for a valid early provisional skip followed by a later pick, a first skip after cutoff, wrong candidate identity, an unresolved non-null shadow result, and a terminal skip-shadow result. These are metadata/contract tests; they require no actual outcomes.

### 5. SHOULD — distinguish missing evidence from missed work in V4/V6/V7/V10

**Design:** D:81,83–84,87.

| Check | Source-derived problem | Verbatim replacement/addition |
|---|---|---|
| V4 | The predicate names a double with one confirmed slot and one unknown, or an unknown single. It omits a delivered double with **both** slots unknown. Ledger matching produces `unknown` independently for each unlinked selection (`scripts/audit/season_ledger/compile.py:197–212`). | “Any delivered selection set with one or more slots whose entry is unknown yields an entry-evidence-gap candidate, preserving whether some or all slots are unknown. Missing/unparseable required input is unknown, not a confirmed lack of entry.” |
| V6 | EOD health can run without a DM or universal success message. `run_all_checks` logs the returned alerts and invokes dispatch (`src/bts/health/runner.py:287–298`); logging iterates the alert list (`src/bts/health/alert.py:39–47`), and dispatch returns without writing status when there is no body (`:189–196`). An intraday refusal can also write the same DM status (`src/bts/scheduler.py:2173–2179`). `sent_day` alone does not identify an EOD invocation. | “EOD completion requires a version-specific positive witness bound to the processed date and EOD invocation. Absence of an optional health/DM message is unknown. A candidate requires positive evidence of failed/bypassed EOD work or absence of a mandatory completion witness within complete coverage. Intraday health DMs do not certify EOD.” |
| V7 | A cron error proves failure somewhere in a run, not failure to start. A file existing for a long-lived shared log does not prove coverage of this run. The configured `flock -n` has no skip-print wrapper, and several jobs discard output (`scripts/cron-setup-hetzner.sh:53,62–67`). | “Track scheduled invocation and successful completion separately. An error marker is a failed-run candidate; a missing required product is a completion-gap candidate only with a known deadline and complete product/source coverage. An unlogged flock conflict or an output-silent job is unknown for invocation unless another retained witness establishes it. Do not call all these cases did-not-run.” |
| V10 | `alerted` expressly means detected/unresolved, not necessarily sent or attempted. It is written when cutoff passes without a nag, when tiers are exhausted, and when no recipient exists (`src/bts/cli.py:1776–1803,1815–1842`). | “A failed-alert-delivery candidate requires evidence that a send was attempted, or that an active notification contract required a send that did not occur. `alerted` without a send event alone is insufficient: distinguish cutoff suppression, exhausted tiers, missing recipient and absent receipt coverage. Preserve the entry anomaly separately. Tier membership is not a send timestamp.” |

Add synthetic controls for healthy EOD with no alerts, an intraday DM without EOD, a silent cron job, both DD entries unknown, and cutoff-suppressed entry notification. An output gap and a service contract deviation should retain their different reason categories through reconciliation.

### 6. SHOULD — nullable comparison provenance must not force an unnecessary X-20 amendment

**Design:** D:62.

The normalized comparison columns are now named correctly. However, `derivation_source` is legitimately null when a pick already has a per-slot result; the helper returns `(None, None)` for that case (`scripts/audit/season_ledger/outcomes.py:25–32`). The compiler still fills `local_norm` from the raw slot and can form a valid disagreement (`scripts/audit/season_ledger/compile.py:197–220`). If “any projected value is null” includes the components of `comparison_basis`, a valid direct-slot comparison stops solely because no day-to-slot derivation was needed. I did not inspect whether either of the two actual rows has that shape.

**Replace the null rule in D:62 with:**

> Stop on a count other than two, duplicate selection keys, or a null/invalid selection identity, date, slot, local_norm, contest_norm, match or match_reason. `derivation_source` is nullable by the ledger contract; retain that null as “no derived day-to-slot provenance” rather than inventing a derivation. Only an unexpected missing required value or schema/identity mismatch triggers the amendment gate. Pin required versus nullable components in the projection schema before reading B3.

Keep the two-row count, accepted-output hashes and exclusive outcome-disclosure scope. This edit neither infers a mechanism nor expands the permitted outcome values.

### 7. SHOULD — a missing-send mutant needs an absence witness, not an invented boundary call

**Design:** D:105,109,119.

The new same-invocation requirement closes r2's disconnected-call counterexample. But the hook “at the observable boundary” cannot execute when the declared symptom is **no send at all**: the mutated refusal/early-return branch is precisely what prevents that boundary. D:119 explicitly requires this kind of characterization, and D:105 includes missed sends/alerts as symptom kinds. Requiring a boundary stack for every failure would either reject valid negative-event fixtures or tempt the fixture to call the boundary itself.

**Append to D:109:**

> For a wrong or extra event, require the linked boundary invocation and the checked event's identity. For a missing event, witness the mutated decision and completion/timeout of the same declared production invocation under the controlled clock, then inspect a boundary recorder covering the full required observation interval. That recorder must remain active across relevant callbacks/tasks and show no qualifying event; the fixture never calls the missing boundary to obtain a stack. Pair it with a baseline positive-event control. Witness hooks are observational only. A trace from another invocation, date or selection does not satisfy either certificate.

This is an implementation requirement for the certificate, not a request for production instrumentation. `component` remains the correct level when internal production functions are mocked. A stack proves reachability; the identical green/red/green oracle and inspected semantic mutant still supply the behavioral comparison.

## False greens remaining

Until the edits are applied:

1. **All unknown:** broken extractors/detectors satisfy every §6.5 case. Require exact outcomes on complete producer-shaped inputs (finding 2).
2. **Dropped day rows:** a selection-only ledger projection erases deliberate skips and unobserved/unfinalized dates. Preserve all row kinds and schedule completeness (finding 3).
3. **Exactly too late:** a send at T−5 passes `≤`; production refuses it. Require `<` and the equality test (finding 3).
4. **Stranded marked resolved:** `result="unresolved"` is non-null and passes V8. Use terminal membership and the correct per-stream field (finding 4).
5. **Both DD slots unknown:** neither listed V4 case fires. Test any missing slot evidence (finding 5).
6. **Intraday alert masks missed EOD:** the shared `sent_day` marker is mistaken for an EOD witness. Bind the invocation and processed date (finding 5).
7. **Artificial missing-event witness:** the test directly calls a boundary that production skipped. Certify the bounded absence instead (finding 7).

False positives/unnecessary stops also matter: cutoff-suppressed `alerted` markers, healthy silent EOD, valid provisional D8 later followed by a pick, and nullable ledger derivation provenance must not automatically become delivery failures, capture violations or exposure amendments.

## Answers

- **(a) Leaf lists / exposure:** The old direct value leaks are closed by the typed/opaque contract. I found no required new raw outcome value. The writer-path mistakes in finding 1, day/schedule/config inputs in finding 3, stream mappings in finding 4, and legitimate ledger null in finding 6 remain. Validate the entire emitted envelope so template selection cannot encode an outcome indirectly.
- **(b) V1–V11 / four values / synthetic gate:** The four-valued model and era-based eligibility are appropriate. The gate is currently too permissive. Findings 2–5 supply exact detection expectations, missing-input cases and the specific predicate corrections. Historical coverage can remain unknown; complete synthetic inputs must demonstrate actual detection. 9/03 correctly remains history-derived.
- **(c) Stack witness:** The old unrelated-entry/direct-helper trick fails D:109's connected-invocation requirement. Retain that requirement and component labels. Add finding 7's bounded absence witness for missed sends/alerts and bind witnesses to the checked invocation/selection.
- **(d) pytest 9.0.2:** D:117 is sound. Under `.venv/lib/python3.12/site-packages/`, `_pytest/skipping.py:282–308` bypasses conversion under `--runxfail`, matches `raises` without restricting the phase, and fails strict XPASS. `_pytest/runner.py:215–223` reports unrelated setup/teardown failures as ERROR. The design now checks exactly these distinctions and rejects a dedicated exception in the wrong phase. Version confirmed from `_pytest/_version.py:31–32`; no pytest execution was needed for this source-level conclusion.
- **(e) Pinned Pass excerpt:** The excerpt and its digest support the revised clause-C row. Current-code expectations agree with `src/bts/picks.py:931–989,1003–1016` for the specified complete feeds. The existing resumed-hit test remains conflicting implementation coverage, as D:133 now says. For the absent-player downstream control, also supply no hit-bearing fallback game, since the real `check_hit` path may search other games; do not feed pending `None` into `update_streak`. The full-page provenance remains an unverified bundle claim in this repo-only review.
- **(f) Gates:** Reviewed code/schema/template/input pins, X-20 before B-source access, owner go-ahead, controlled amendments and separate run directories are adequate. Gate 3/4 must run the strengthened full-pipeline disclosure/discovery tests, with exact supported-schema mappings and expected results. No box read is needed to settle these edits. The same-day replay invariant at D:139 remains resolved.

DONE
