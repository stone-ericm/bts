## Verdict

**BLOCK.** v2 resolves important parts of r1, including the replay invariant, narrower incident claims, and most of the expected-failure scheme. Four blockers remain:

1. The redacting viewer and free-text whitelist fields can expose values forbidden by X-20.
2. Several runtime invariants cannot be evaluated from the declared projections; the ledger projection also names nonexistent columns.
3. The runtime predicates can miss silent non-delivery and treat missing evidence as an incident or a successful check.
4. The coverage witness does not establish that the production entry point reached the mutant through the incident's causal path.

Reviewed input: `docs/superpowers/specs/2026-09-29-incident-register-design.md`, commit **ab9ec202e15e85f490afc75df090cc7d112684ac**. HEAD matched that commit; the design had no working-tree diff. **D:line** below denotes this exact design file. Prior findings were read from `docs/audit/2026-09-29-incident-register-codex-design-r1.md`.

This review is source-based. No `data/` contents, SSH, network, or live measurements were used. No tests or mutants were run. The pytest conclusions come from the installed **9.0.2** implementation (`.venv/lib/python3.12/site-packages/_pytest/_version.py:31–32`). Only this report was written. Counterexamples below are logical/source-derived scenarios, not measured production results.

## r1 findings status

| r1 | Status | v2 disposition |
|---|---|---|
| 1 — exposure | **PARTIAL** | D:106–142 replaces raw grep output with projections and explicitly registers the two outcome rows. But D:113's viewer is still a blacklist, `degraded_reason` is free exception text, and output fields lack complete type/value rules. New finding 1. |
| 2 — sources/completeness | **PARTIAL** | D:79–104 adds the missing producer logs, configuration and coverage limits; D:24 correctly narrows the claim. D:157–171 adds all-date discovery, but its projections, predicates and unknown handling do not yet support that discovery. New findings 2–3. |
| 3 — commit rubric | **PARTIAL** | D:146–155 correctly demotes keywords and the 10% sample, and reviews runtime negatives. However, `experiment/` actually runs the included research streams, and R2 ends at the incident cutoff rather than the review baseline. New finding 6. |
| 4 — xfail specificity | **RESOLVED** | D:251–258 replaces broad `AssertionError` with a dedicated oracle exception, unmarked validation, controls and per-parameter marks. This closes r1's ordinary setup-assertion false green. The meta-test terminology and the exact failure signature need the smaller clarifications in new finding 7. |
| 5 — mechanism-specific reproduction | **PARTIAL** | D:218–249 adds pins, minimal mutants, symptom kinds and survivor classifications. D:237 proves co-execution, not a causal call path; D:227 creates a hybrid historical environment without auditing fixture/config changes. New findings 4–5. |
| 6 — Pass contract | **PARTIAL** | D:271–287 correctly distinguishes present zero-AB/SF cases, absent players, resumed-only PA and pre-suspension hits. But the supplied clause C supports an additional Pass case that D:278/320 defer without a contradictory rule. The claimed archive was not present at the declared repository evidence path. New finding 8. |
| 7 — same-day replay invariant | **RESOLVED** | D:289–292 adopts the qualified invariant, both streak/saver cases, repeated-run assertions, preview exclusion and the fixed midnight control. It no longer treats an empty correction list as a universal no-op promise. |
| 8 — named/design-open fixtures | **RESOLVED** | D:29–31,260–262,294–297 require executable characterization/invariant fixtures or explicit gaps and separate them from W4 acceptance. Historical-path adequacy remains subject to findings 4–5; the previous blanket “design choice” exemption is gone. |
| 9 — scope/evidence/recovery | **PARTIAL** | D:33–77,182–208 fixes scope, tiers, timelines and evidence kinds. But D:178–179 contradicts those dispositions by upgrading runtime-only candidates to incidents and downgrading history-only evidence indiscriminately. `operating_mode` also conflates independent dimensions. New finding 9. |

## New findings

### 1. BLOCKER — §5 still permits forbidden values through allowed fields and the viewer

**Design:** D:108–114,120–140.

**Evidence:** `decision.json.degraded_reason` is not a safe enum. `src/bts/strategy.py:77–93` retains artifact exceptions as strings; `:252–274` embeds those strings or lookup exceptions in the degradation reason; `src/bts/daily_decision.py:85–99` writes it unchanged. Those exceptions can contain rate arrays and boundaries (`src/bts/simulate/tail_policy.py:302–307`). Thus directly emitting the explicitly whitelisted `degraded_reason` can emit probabilities. This is a real writer path, not an assumption that “reason” fields are always dangerous. It does not establish that any particular real decision contained such a value.

The unknown-line viewer cannot enforce D:110 either. Synthetic `WARNING: local_normalized=HOLD`, `WARNING: result=Hit`, and `WARNING: streak is zero` survive the specified case-sensitive token/digit masks. The first two reveal an outcome; the third reveals a streak value without digits. A dictionary, exception or traceback can contain any of them. Existing result prose also includes “Streak reset to 0” (`src/bts/cli.py:2350`); masking its numeral does not remove the result's meaning if that prose reaches an unknown-line wrapper. No finite list of result words makes arbitrary message bodies safe to view.

The whitelist is currently field-name based. `reason`, `status`, `schema`, `timestamps`, `keys`, `locator` and `acceptance fields` have no complete leaf/type/value contract. In particular, scheduler `runs_completed`, `skip_summary` and `final_skip_candidate` contain pick probabilities or state (`src/bts/scheduler.py:355–372`); a broad interpretation of “commit” or “skip” must not copy those objects. Health `sent_sources` can contain probability-band labels such as `70-75% DD-leg` (`src/bts/health/alert.py:198–201`; `src/bts/health/realized_calibration.py:70–72,390`). Those are static category labels, not realized rates, but the proposed universal percentage/number ban would reject them unless the categories are normalized explicitly.

**Smallest edit:**

> Unknown text is never displayed under X-20. Emit an opaque source locator and `unclassified`; derive additional templates from pinned producer code and synthetic inputs. A necessary live-text disclosure requires a scoped exposure amendment first. Every emitted field has an exact JSON path, type, allowed enum or numeric meaning, and missing/type-mismatch behavior. Free-text reasons become closed reason categories plus `has_detail`; arbitrary strings and nested objects are never copied. Normalize known alert categories to stable identifiers.

Add tests that vary forbidden result/streak/probability values while holding operational inputs fixed and require identical visible metadata, apart from declared provenance hashes/locators and the two authorized comparison values. Test extractor errors, stderr, tracebacks, viewer output and agent-bound payloads too. The current check that field names belong to a whitelist is necessary but insufficient. D:140 must expressly exempt the two approved outcome columns; otherwise it contradicts D:130 and its own HIT/NO_HIT/HOLD output.

### 2. BLOCKER — §§5–6 lack a coherent input schema for the runtime checks

**Design:** D:108–109,116–131,157–171.

**Evidence and minimum additions:**

| Requirement | What the current contract lacks | Writer/schema evidence |
|---|---|---|
| V2 delivery time against the correct selection | Pick game times and `delivered_at` are allowed, so those are **not** missing. Selection identity, delivery-mode history, and a join from earlier journal sends to that selection are missing. `notification_id` presence cannot identify a message. | `src/bts/picks.py:205–222`; ledger `selection_id`, `game_time`, `delivered_at` at `scripts/audit/season_ledger/compile.py:36–47`. |
| V3 uniqueness | Slot presence cannot distinguish A+B from A+C, or two successful sends from duplicate copies of one receipt. Preserve a non-outcome selection-set key and stable receipt/event identity, not only `has_msg_id`. | Pick persistence is overwrite-on-write (`src/bts/picks.py:301–318`); lineup evolution records identities at `:370–400`. |
| V4 entry completeness | `entry_status=unknown` is not `not_entered`; the projection drops match/history basis and selection identity needed for ambiguous or revised choices. | Compiler explicitly writes `unknown` when no linked record exists (`scripts/audit/season_ledger/compile.py:197–220`); it does not prove absence. |
| V5 liveness/check schedule | Exact `games[*].game_pk/game_time_et`, schedule observation time, `runs_completed[*].time`, and `next_wakeup` are not listed in the state whitelist. Pick times can be stale after a moved-up game. | `src/bts/scheduler.py:353–361,2608–2615`; July 16 chronology at `docs/optimization-ideas.md:30–36`. |
| V6/V7 expected health/cron activity | Need config activation intervals and explicit run-marker semantics. A cron invocation and its stdout are not automatically a timestamped start/completion receipt. | Commands redirect output without a universal run wrapper (`scripts/cron-setup-hetzner.sh:53–70`); some jobs are silent or gated, and `flock -n` can skip the grader. |
| V8 research status | `result_status` presence cannot distinguish `final` from `unresolved`. Shadow, skip-shadow and lineup-evolution schemas have no explicit leaf contracts. Acceptance-marker presence is not D8 acceptance validity. | State vocabulary at `src/bts/scheduler.py:360`; D8 validates file hashes, provenance and publication timing (`scripts/live_forward_capture_once.py:925–991`). |
| V9 deploy/rollback | Need the actual transition record, not merely job/step conclusion. Canary, deploy and rollback are all branches inside one Actions step. | `.github/workflows/deploy.yml:70–84,159–181`. There is no separate “auto-rollback step” whose conclusion supplies R4's promised field. |
| V10 alert history | Health-DM records use `updated_at`, but entry markers use `date`, `checked_at`, `reason`, `escalations`; preserving only the example health keys loses the latter timeline. Current status is not per-date history. | `src/bts/health/alert.py:65–86`; `src/bts/cli.py:1743–1751,1781–1783,1832–1842`. |
| V11 private/tail | Needs mode/config epoch plus decision objective/status; tail stop correctness cannot be recomputed from presence while state values are forbidden. Reuse the registered P-05 result explicitly instead. | Policy stop depends on state (`src/bts/strategy.py:286–302`); P-05's declared sources include state and artifacts (`docs/audit/2026-09-28-p05-tail-policy-audit.md:5–10`). |

There is also a concrete B3 schema mismatch. D:130 names `local_normalized`, `contest_normalized`, `comparison_basis`; the accepted builder defines **`local_norm`, `contest_norm`, `derivation_source`, `match`, `match_reason`**, with no `comparison_basis` column (`scripts/audit/season_ledger/compile.py:49–54`). A permissive `.get()` projection could emit two null-valued rows and still satisfy the row-count check. D:129's unqualified “anomaly rows” must not mean copying full occurrence records: that table contains `fields_json` and `record_raw_json` (`:63–65`).

**Smallest edit:** insert a V1–V11 dependency table before implementation. Each row must name exact source/field paths, safe types, selection/event join keys, date/mode/config scope, evidence sufficiency, and an `unknown/not_evaluable` result. Extend §5 only with the operational fields needed by that table; forbid unresolved joins from reading extra raw fields opportunistically. Use separate typed event/state projection schemas—§5.1's event-only columns cannot also carry every promised state field without an explicit contract.

For B3, write the exact mapping: `local_norm -> local_normalized`, `contest_norm -> contest_normalized`; define a closed comparison-basis category from `derivation_source` and `match/match_reason`, or retain those named columns explicitly. Require exact input schema/type validation, stable row/selection identity, and a unique selection key for each comparison. Project quarantine metadata to source id/locator/reason category only. Bind the **accepted output manifest/ACCEPTED identity**, not just the input bundle hash and an ellipsized directory suffix.

### 3. BLOCKER — Route R can accept a silent missed pick as finalized

**Design:** D:161–171,178.

**Counterexample:** a pick file exists, was never delivered, and has no genuine commit or deliberate skip. V1 passes because it accepts any pick file. V2 only checks confirmed delivery, V3 has zero sends, and V4 starts from a delivered pick. Those checks do not open the missed-delivery candidate. The ledger itself allows `row_kind=selection`, `finalization=pick_file_only`, `commit_status=unconfirmed` for an uncommitted preview (`scripts/audit/season_ledger/rows.py:67–87,200–204`). File presence is not finalization. This is the shape documented for August 13 (`docs/optimization-ideas.md:62–72`) and for a stale classification lock (`src/bts/scheduler.py:3095–3103`). A blanket sweep of unconfirmed ledger rows may rescue the date, but D:129/161 must require that disposition; V1 as written asserts the opposite.

**Would the six named incidents be discovered?** The following assesses the predicates, not retained real records:

| Incident | v2 Route R assessment |
|---|---|
| **7/16 moved-up singleton** | V5 catches it only if it has the moved-up start/status observation. With the morning 19:10 start retained, the 18:10 check appears before the 19:05 cutoff even though it was actually at first pitch. V1 can pass the undelivered preview. `docs/optimization-ideas.md:30–42`. |
| **8/13 Warmup silent pass** | The missing-delivery shape above escapes the delivery predicates. A generic locked status is not a genuine commit; V1 must inspect that distinction. `docs/optimization-ideas.md:62–72`; `src/bts/scheduler.py:3095–3103`. |
| **8/30 late Kwan send** | V2 is appropriate **if** the retained journal send is bound to the right selected games and their applicable start times. A later fixture's `delivered_at` field cannot retroactively supply the original send timestamp. The incident/fix timing is documented at `docs/audit/2026-08-30-late-pick-delivery.md:3–20,71–78`. |
| **7/12 restart/DM storm** | Burst detection should catch it. However, each bad cycle also ended cleanly after EOD and restarted; “idle → exit → restart” cannot by itself label every such cycle planned. Require the expected date transition and bound planned turnover separately from repeated cycles. `docs/audit/2026-07-12-eve-of-break-restart-loop.md:5–11,24–33`. Latest DM status alone cannot count the storm. |
| **7/08 partial entry** | V4 can raise an **entry-evidence gap** from one confirmed and one unknown slot. It cannot establish that the second slot was absent from the contest merely from `unknown`. The historical operator/incident report is separate evidence. `scripts/audit/season_ledger/compile.py:204–220`; `docs/audit/2026-07-09-gpt56-sol-audit.md:10`. |
| **9/03 all-skip idle** | A properly recorded skip can satisfy V1–V10. V11 starts on 9/14 and excludes the state needed to decide whether stopping was wrong. This incident is discoverable through Route H's recorded requirement and code, not guaranteed by these metadata invariants. `docs/audit/2026-09-03-emax-tail-policy.md:5–15`; D:171. |

V8 also specifies the wrong expected population: “skip-shadow record per MDP skip” includes tail stops, while the actual predicate requires **reach57** (`src/bts/daily_decision.py:53–58`). Official capture “per decision day” likewise needs its eligibility/config era: the capture runner deliberately waits for a production pick and uses a separate D8 root for skip-day research (`scripts/live_forward_capture_once.py:2–7,35–45`). V7's schedule must come from each installed era, not today's cron list. These are false candidates unless explicitly classified as expected/inapplicable.

**Smallest edit:**

> V1 requires an evidenced committed selection or a deliberate final skip for each applicable game day. A preview-only, unconfirmed/conflicted, unresolved-finalization or no-record day is a candidate, not a satisfied finalization check. Evaluate missing delivery independently of `pick_locked` and confirmed-delivery existence. All invariants return `satisfied`, `candidate`, `not_applicable` or `unknown`, with source coverage and reason; unknown is never success or an observed incident.

Declare eligibility/activation epochs for V6–V8; restrict skip-shadow to `is_reach57_mdp_skip`. Preserve evolved schedule/status observations for V2/V5 where available, otherwise report the time comparison unavailable. Treat failed-send retries as uncertain until distinct confirmed receipts are reconciled; “recorded failure” alone is not permission to ignore a later duplicate. Reuse P-05 only for its stated checks and dates and label the original 9/03 case history-derived. Before a box run, synthetic metadata for all six named incidents must produce the intended candidate or a justified `unknown`, with ordinary skip/private/off-day controls producing no false incident. Add explicit coverage limitations for silent wrong grades/streaks, wrong eligibility, or duplicate sends whose overwritten evidence is unavailable.

### 4. BLOCKER — §9.3's coverage witness proves co-execution, not the required call path

**Design:** D:223,234–238.

**Counterexample:** during one killing test, call `run_day` with an empty slate so it returns, then directly call the mutated grading/helper function and fail the observable assertion. Both functions and the mutated lines appear in that node's coverage, satisfying D:237. The production entry point never reached the changed branch. `run_day` genuinely has the early return at `src/bts/scheduler.py:2570–2584`. Restricting coverage to only the mutated module also cannot show an entry point in a different module.

This matters because the existing incident fixtures patch internal production functions, not just HTTP (`tests/test_incident_2026_08_30.py:86–98`; `tests/test_scheduler_eve_of_break.py:40–46`). D:245 correctly limits their claims, but does not make line coverage evidence of the missing connection. D:223's blanket external-boundary-only rule and the legacy fixture exception need separate labels.

**Smallest edit:**

> A path witness records the call stack or invocation-linked trace at the mutated branch and at the observable boundary during the killing node. It must connect the declared production entry invocation, the changed decision and the failed symptom. Entry-point coverage elsewhere in the node is insufficient. Instrument all necessary entry/causal modules without replacing their decisions. Legacy fixtures with internal mocks retain only their stated component-level coverage; they do not satisfy a full production-path certificate.

Keep the observable oracle, minimal mutant and paired green/red/green runs. Coverage remains useful supporting evidence; it is not itself the causal witness.

### 5. SHOULD — §9.2's F/F^ construction needs a harness-compatibility audit

**Design:** D:225–232,240–245.

**Evidence:** keeping tests/conftest/dependencies at F is a legitimate way to share one regression harness, but it is not automatically the pre-fix production environment. F can change fixture payloads, clocks, autouse patches, serializers, script wrappers or dependency semantics. A new fixture value can make old code fail the same visible symptom for a reason that never occurred live. Rejecting new-API TypeErrors does not reject that case. This repository's `ac0ce8d1fc43ad4a19f14db572effc8718b9a01a` commit body explicitly says two test clocks were changed as part of the fix; its stat includes both affected test files. Current incident tests also import helpers from other test modules (`tests/test_incident_2026_08_30.py:18`; `tests/test_scheduler_eve_of_break.py:22`).

Reverting only `src/` cannot reconstruct a historical defect in a cron/script/unit path, all of which the register includes (D:35,153). Multi-commit fixes require the actual defective deployment and the complete fix set, not an unspecified single-parent shorthand.

**Smallest edit:** audit F^→F changes to the transitive harness, fixture data, config, lock/dependencies and scripts before accepting a historical claim. Pin the exact defective deployed ref, fixed ref/fix set and all executed files, including imported test helpers/conftest and script entry points. Document any compatibility adapter and prove that it preserves the incident inputs and old call semantics. Prefer the deployed production closure plus the neutral fixture overlay; otherwise label F-tests/F^-src as a **semantic regression replay** until equivalence is established. If equivalence cannot be established, historical reproduction remains unavailable, with current defence reported separately. Use controlled clocks for both versions; the existing eve-of-break test deliberately depends on `date.today()` (`tests/test_scheduler_eve_of_break.py:28–37`), so a content hash alone does not pin its input date.

### 6. SHOULD — Route H still excludes included runtime code and post-cutoff repairs

**Design:** D:93,146–155.

**Evidence:** `experiment/` is excluded from mandatory negative review, but the live capture runner invokes `experiment export-live-candidate-artifacts` and `verify-candidate-artifacts` (`scripts/live_forward_capture_once.py:327–348,351–374`); the verifier and resolver use `bts.experiment.artifacts` (`src/bts/experiment/cli.py:424–435,469–490`). Those are production research-integrity paths explicitly in scope. The script glob is likewise an example, not a demonstrated transitive runtime closure.

R2 ends git history at the September 28 08:00 incident freeze. The C-03 repair commit is **ce6676d9ce3754a3d6ffc9c4415942930e45192e**, committed **2026-09-28T11:14:43-04:00** (verified with `git show --no-patch`). It would fall outside that sweep despite being a required fix-status/fixture source. Occurrence cutoff and evidence/repair cutoff are different.

**Smallest edit:** inventory commits through the pinned review baseline, while limiting incident occurrence dates to the frozen window. Derive mandatory negative review from the versioned command/import closure of every included service, cron, capture, verifier, resolver and restore job, including its frozen research checkout where different; do not exclude a directory merely because it is named `experiment` or `validate`. Retain late discovery/repair evidence with its own observation time.

### 7. SHOULD — §9.7 is sound in principle, but specify failure signatures and pytest phases

**Design:** D:251–258.

**Verified from pytest 9.0.2 source:** `--runxfail` bypasses marker conversion (`.venv/lib/python3.12/site-packages/_pytest/skipping.py:282–283`). An ordinary AssertionError does not match the dedicated exception and gets `report.outcome="failed"` (`:289–304`). At the terminal/meta-test summary level, a failed **setup or teardown** is **ERROR**, not FAILED (`.venv/lib/python3.12/site-packages/_pytest/runner.py:215–223`; `_pytest/terminal.py:340–345`). A pytester expectation of `failed=1` for the setup case would be wrong; assert `errors=1, xfailed=0` or inspect the report phase/outcome directly.

The dedicated class is not intrinsically call-phase-only. If that same exception is accidentally raised in setup, the marker can still convert it to XFAIL; the branch at `_pytest/skipping.py:289–302` has no phase restriction. D:252's restriction on where the class is raised plus the call-phase validation is therefore essential, not redundant. Imperative `pytest.xfail()` is another special conversion path that precedes `raises` (`:284–287`) and should not appear in these harnesses.

There is one residual oracle risk: `if actual != required: raise IncidentFailure(...)` can accept a *different* defect. For example, a grader that now returns pending `None` instead of the known bad `miss` still differs from `void`; it must not silently inherit the old expected failure.

**Smallest edit:**

> The oracle raises the dedicated exception only for the declared bad-value/failure signature; another mismatch raises an ordinary assertion. The accepted expected failure must be in the call phase at that oracle location. Meta-tests assert FAILED for an unrelated call assertion and ERROR for an unrelated setup/teardown exception, both with zero XFAIL; report-level checks may instead assert `outcome=failed` with the exact phase. A dedicated exception from setup/teardown or an imperative xfail is rejected as reproduction evidence.

Include intended-oracle XFAIL and fixed-behavior XPASS(strict) controls in the meta-test. This strengthens the chosen scheme without introducing a new production mechanism.

### 8. SHOULD — §10.1's quoted clause C does not justify waiting for a 2027 outcome

**Design:** D:265–287,320.

**Evidence:** on the text supplied in D:267, clause C expressly makes a suspended game with no hit at suspension a Pass. D:268's general statement that grading uses only pre-suspension activity does not say an earlier out overrides that Pass clause. Listing “Hit, Pass, or No Hit” is not a contradictory condition. The earlier test's `miss` expectation is implementation evidence, not external-rule authority (`tests/test_check_hit_suspension.py:50–58`; `src/bts/picks.py:960–967`).

The rest of the table's current-code descriptions are correct for its **complete** inputs: present zero-hit normal batters return `miss` (`src/bts/picks.py:976–988,1012–1016`); absent players yield None; resumed-only PA yields void; pre-suspension hits yield hit. For suspended cases, explicitly supply `resumeDateTime` and complete timed plays. Without that boundary, the helper falls back (`:944–947`), and a nonfinal game is held pending by the public fetch path (`:1035–1044`). Do not confuse input insufficiency with ambiguity in the quoted external rule.

The design claims section 6 is archived in the evidence directory, but `docs/audit/2026-09-29-incident-register-evidence/` was absent and the scoped repository file inventory contained no rules artifact. The printed hashes are abbreviated. I could review the supplied quotation, but could not independently verify its completeness or digest. No external fetch was attempted under the no-network constraint.

**Smallest edit:** supply the exact archive path and full digests before accepting fixtures that cite it. On the supplied clause C, classify complete pre-suspension AB-without-hit as required `void` and add a strict expected-failure case after unmarked verification. Retain the old test as explicitly conflicting coverage until separately authorized production repair. If the full rule reveals a real contradiction, quote that competing condition and record a rule-clarification task. Replace “until 2027 evidence … settles it” with resolution from authoritative rule clarification; a single future contest result is not necessary to define this fixture contract. Incomplete feeds may remain characterization-only as D:287 specifies.

### 9. SHOULD — routing must preserve the declared evidence dispositions and independent modes

**Design:** D:41–49,72–77,178–179,184–186.

**Evidence:** D:178 calls runtime-only candidates “silent incidents” before §2's contract-deviation test. An unknown entry match, expected off-day or unclassified warning can be runtime-only without being an incident (findings 2–3). Conversely D:179 requires runtime corroboration for every history-only candidate even though D:45/74 permits a contemporaneous operator report and D:46 intentionally keeps deployed latent defects with no evidenced firing. `reported` is used as both an evidence label and an apparent disposition, but is absent from §2's disposition enum and §3's stated kind list.

`operating_mode = dm/public/private/tail/research-only` also mixes axes. A tail-objective pick can be delivered by DM or saved privately. This is documented by the separate objective and delivery fields (`src/bts/daily_decision.py:95–98`) and the private delivery branch (`src/bts/scheduler.py:1013–1025`). Selecting `tail` instead of `private` would make mode-dependent V2/V11 checks ill-defined.

**Smallest edit:**

> Runtime-only candidates receive priority investigation, not automatic incident status. History-only candidates use the same §2 evidence rules: a qualified operator report may establish an observed incident, code/deployment evidence may establish a deployed latent defect, and unsupported reports remain unresolved candidates. `reported` is an evidence-strength label, not another candidate disposition. Add it explicitly to the evidence schema.

Split delivery mode (`dm/public/private/unknown`), policy objective (`reach57/emax_season_best/unknown`), and stream (`production/shadow/skip-shadow/live-forward/D8/...`) into separate fields. Keep the improved occurrence, latency and recovery fields unchanged.

### 10. SHOULD — §12 must bind the reviewed extractor and evaluability checks to the first read

**Design:** D:80–86,97–99,106–107,113–114,210–215,310–316.

The order now correctly gates B-source reads on X-20 and Eric's box-run go-ahead. The remaining issue is execution identity: D:114 permits adding templates during a live-data session, while §12 does not say when those edits invalidate the earlier code review/X-20 output contract. B2 is acquired after the incident freeze and includes overwritten sources; acquisition time alone cannot turn a later current status into historical evidence. Full accepted-build/source-output hashes and approved schema identities also need to be available before a projection silently adapts to a mismatch (finding 2).

**Smallest edit:** add to gate 4:

> The run pins the reviewed extractor/invariant/viewer commit, output schema and template-table hashes, exact X-20 revision, B1 manifest, B3 accepted-output identity, and declared B2 source list. Synthetic leakage, parser-error, join/unknown and named-incident discovery tests must have passed code review. Inventory, acquisition and error output obey the same disclosure contract. Any template/schema change requires renewed synthetic checks and review before use; any expanded disclosure requires an X-20 amendment before the read. Each B2 record is classified by event time and retention semantics; post-freeze latest state cannot certify a pre-freeze interval. A missing or unclassifiable source yields explicit unavailable/unknown coverage.

Use a separate extraction run directory for each approved code/schema version. O1 remains outside default scope. No new production deployment is needed for these gates.

## False greens remaining

1. **Allowed field, forbidden value:** `degraded_reason` copies numeric exception details; a whitelist-key test still passes. Use closed reason categories and typed leaf validation (finding 1).
2. **Viewer masking accepted as non-disclosure:** `HOLD`, `Hit`, or “streak is zero” remains visible after the specified masks. Keep unknown text opaque (finding 1).
3. **Two rows but no comparison:** wrong B3 column names become nulls under a permissive projection, while count still equals two. Require the exact typed schema and explicit mappings (finding 2).
4. **Presence masquerades as finalization/acceptance:** an undelivered preview satisfies V1; an acceptance filename satisfies V8 despite a wrong sidecar hash. Require the relevant commit or acceptance contract, with unknown preserved (findings 2–3).
5. **Stale clock basis:** the singleton check is before the stale morning cutoff and passes V5 although the game moved up. Bind cutoff calculations to the applicable schedule observation or report them unevaluable (finding 3).
6. **Covered entry and covered helper, disconnected invocations:** both execute in one node but the entry never reaches the mutant. Require an invocation-linked causal trace (finding 4).
7. **Hybrid old code/new harness:** new fixture or dependency semantics make old src fail a visible assertion for a nonhistorical reason. Audit the full historical execution closure and label unsupported historical equivalence explicitly (finding 5).
8. **Dedicated exception wraps any mismatch:** changed bad behavior remains XFAIL even though the old symptom no longer reproduces. Match the declared failure signature, and validate the report phase/location (finding 7).
9. **Candidate or missing evidence becomes verified incidence:** runtime-only means “incident,” or unknown entry means “not entered.” Preserve evidence strength and candidate dispositions until the contract deviation is established (finding 9).

The prior import/setup/skip rejection and mutant-survivor classifications remain useful. None of these counterexamples requires weakening those protections.

## Answers

- **§5 exposure and actual writers:** Still unsafe. The main concrete leaks are free exception text in `degraded_reason` and the unknown-line viewer. Pick times and delivery timestamps are correctly allowed, but identities, receipt keys and several state/health timestamps are missing. Use exact leaf/type/enum contracts, correct B3 column names, and explicit quarantine projection; the two normalized comparison outcomes remain the sole authorized outcome exception (findings 1–2).
- **Route R / V1–V11:** V2 can detect the late send with a trustworthy join; V5 can detect the restart burst with a bounded planned-transition definition; V4 can flag a partial-entry evidence gap. July 16 needs changed-start evidence, August 13 needs an independent missing-delivery predicate, and September 3's policy error is not inferable from allowed presence metadata. V6–V8 need config/eligibility eras and evaluability states; V9 needs within-step deployment transitions; V10 needs retained event history; V11 must reuse P-05 within its actual scope (findings 2–3).
- **§9 fixtures and pinning:** Minimal mutants, recorded values and survivor labels improve the protocol, but same-node line coverage is not a causal witness. F tests/conftest/lock with F^-src is a shared-harness regression experiment until compatibility with the defective deployment is demonstrated. Pin transitive helpers, runtime scripts/config and controlled clock inputs as well as source/test content (findings 4–5).
- **§9.7 / pytest 9.0.2:** The dedicated-exception approach plus `--runxfail` is sound for the intended call-phase oracle and ordinary unrelated exceptions. Setup/teardown summarize as ERROR, and pytest itself does not enforce call-only matching. Validate exact phase/location and the known bad signature; meta-test intended XFAIL, XPASS(strict), unrelated call failure and setup/teardown errors (finding 7).
- **§10.1 Pass table / clause C:** Current-code entries are accurate with complete normal/final or suspended-and-resumed inputs. The supplied clause C supports Pass even after a pre-suspension AB without a hit; the general pre-suspension-window sentence does not contradict it. `contract_ambiguous` is not justified by those quotations alone. Supply the full pinned archive, then encode the rule or cite the actual conflicting clause; do not wait for a 2027 outcome (finding 8).
- **§10.2 replay:** Resolved. With consistent complete history, no admissible changes and today's terminal result already applied, `_replay_season_streak` still excludes today (`src/bts/picks.py:599–601`) and reconcile overwrites streak/saver from that replay (`:1180–1184`). The proposed hit and saver cases expose that defect. Assert against the original expected state after **each** repeated run, not merely equality between two already-wrong runs; keep preview and midnight controls ordinary.
- **§12 gates:** The read/owner gate order is now appropriate. Add exact reviewed code/schema/template/input identities, successful synthetic disclosure/evaluability tests, controlled template amendments, and event-time handling for newly acquired B2 latest-state records before the first box extraction (finding 10).

DONE
