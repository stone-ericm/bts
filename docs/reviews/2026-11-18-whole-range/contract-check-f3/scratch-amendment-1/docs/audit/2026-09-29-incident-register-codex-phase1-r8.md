## Verdict

**BLOCK under ruling 10's narrowed target.** Absence refusal works, and the original r7 escapes no longer obtain absence certificates. Three retained obligations still fail: clearing a watched namespace leaves a former callable falsely current; the certifier can accept unattributed or incomplete witnesses; and safe serialization can execute application formatting while the observation is labelled pure. These are positive-witness or eligibility failures inside the stated model, not demands for complete missing-event coverage.

Reviewed main plan rev 8 and design v3.3 at **4544fd48e92153eb76e5fc042dec677268554025**, and the authorized detached checkout **8ab3490**. The tooling, tooling tests, incident fixtures/meta-tests and registry have an empty diff from **31ef140** to 8ab3490. Code file:line references below refer to that code; plan/design references refer to 4544fd4.

Measured offline with Python 3.12.13, pytest 9.0.2 and jsonschema 4.23.0:

| Check | Measured result |
|---|---|
| Submitted tooling suite, initially | **291 passed** |
| Restored tooling + incident fixtures + meta-tests | **334 passed, 22 xfailed**, zero ordinary skips/errors/failures; 291 tooling, 40 fixture and 3 meta-test passes |
| Independently reconstructed r7 requests | **Five absence requests refused before any evidence subprocess**, rejected artifacts contain `ABSENCE_REFUSAL`; original final-census hook no longer makes `stop()` raise |
| Submitted r7 controls | **14 passed**, included in the restored suite |
| Independent review probes | **11 passed** under defect-demonstrating expectations: five refusal cases, one lifecycle case, and five accepted positive executions discussed below |
| Targeted mutant sensitivity | **29/29 KILLED** by their selected regression nodes; their **22 distinct unmutated nodes pass**; source bytes restored after every mutation |
| Expected-failure driver on a synthetic frozen-input copy | **Accepted 22/22**, 20 `value_match`, 2 `exception_shape`, 40 passed controls; all **44 purity records per run** pure |
| Prepared I-0830-d return-spec pilot on that copy | **Accepted**, no reasons |
| E77/L03 sensitivity script | **All eleven as required**: seven predicate mutants caught, both formatter mutants yield the three required ordinary failures, both repair sketches reach strict XPASS with their controls passing |
| Closure screen | Reproduces the **two reviewed-benign hits**, no additional hits |

The synthetic copy is commit **c7f1d9ace3aec48169e8174f412657deb3de1708**. **563 input files** were compared byte for byte with 31ef140: Python source/scripts/tests, config, lock, registry and the pilot spec. It had its own offline-installed venv. This is an equivalent-input synthetic measurement, not publication evidence at 31ef140 or 8ab3490. I did not rerun the full fast suite or all 161 whole-suite mutant runs.

Reproducers, XML, accepted/rejected artifacts, mutant logs and the byte mapping remain under **`.codex-review/2026-09-29-incident-register/r8-probes/`**. The main probe is `test_r8_review.py`; final results are `final-review.xml`, `restored.xml`, `summary.json`, `final-runs/`, `mutants/` and `frozen-copy/`. The five positive acceptances are invalid or require qualification as explained below; a passing defect-demonstration test is not evidence of correctness.

## r7 findings and consult status

“RESOLVED” below disposes the original shape under ruling 10; it does not assert general recorder soundness. The rebooted probes were reconstructed from the report's mechanisms, not recovered byte for byte from the lost directory.

| Item | Status | Measured rerun and limit |
|---|---|---|
| **r7 #1: module fallback / class inheritance** | **RESOLVED under narrowing** | Both reconstructed absence specs reject before execution, with the required artifact/reason. Both supplied event versions reject for no linked `alert` event. Missing those sends is permitted in this round. This does not dispose the former-binding false attribution in finding 1. |
| **r7 #2: C dispatch changed/restored between Python starts** | **RESOLVED under narrowing** | Reconstructed absence refused; supplied event version witnesses nothing. Removal of the sampled dispatch check is consistent with positive-only coverage. |
| **r7 #3: pre-existing mock `__call__` override** | **RESOLVED for the original shape** | Reconstructed absence refused. Both supplied overrides (`sends-elsewhere`, `rewrites-arguments`) reject the normal `alert` request and record the required gap. Standard MagicMock attribution passes; actual held-class change and later call override yield unavailable identities. O45–O48 independently fail their intended controls. Finding 2 concerns consumption of unavailable identities, not failure of these mock checks. |
| **r7 #4: application audit hooks invoked by observation** | **PARTIAL** | Reconstructed absence refused; supplied event probe has a real linked alert but purity refuses it. Direct audited primitives still run the application hook, as the retained control demonstrates. R16/R17/O42 and the purity checks are sensitive. The claim that accepted observation runs no application formatting/callback remains false: finding 3. |
| **r7 #5: final census makes `stop()` raise** | **RESOLVED for that failure path** | Rebuilt hook raising specifically on `sys._current_frames` no longer raises during stop; tool id is released and `obs_end` exists. The stronger supplied raising-hook test also passes, with both watchers released and observer errors recorded. O49 is an assertion kill, including its child's crash status. These are measured paths, not a proof against every exceptional object. |
| **Wording 1: exact types / actual lookup / sampled starts** | **RESOLVED as revised scope wording** | Ruling 2 now distinguishes raw namespaces from actual lookup and removes the per-start completeness implication. Mock verification and audit refusal have measured controls. New positive attribution/purity defects remain implementation failures. |
| **Wording 2: O22 equivalence** | **RESOLVED** | The requested sentence appears verbatim at plan line 364. It limits the equivalence to reading `__dict__`, not resolving the boundary attribute. |
| **Wording 3: failed-start cleanup versus final census / universal stop claim** | **RESOLVED for the requested census correction** | The final census is removed; failed-start and guarded-stop controls pass. The plan's r8 mapping explains this rather than presenting the old census as safe. Finding 3 also reaches the type-name helper used by `_err`; its general no-application-code claim needs correction. |
| **Consult: disable absence mechanically, retain counterexamples** | **RESOLVED** | Spec refusal precedes ownership/reset/runs; direct `certify` refuses all four submitted shapes with the exact exception and reason. Five independent requests verify rejected artifact equality and no event files. D14/C14 are sensitive. |
| **Consult: written bounded model, whole-closure review, hash binding** | **PARTIAL** | Model and reviewed-precondition wording are present; the actual closure screen reproduces its reviewed hits, and the pair freezes inputs/environment. The machine-enforced current-value, incomplete-value and observational promises fail in findings 1–3. |
| **Consult: eligible pilot, then frozen planned evidence / publication** | **PARTIAL** | The I-0830-d pilot and equivalent-input pair work. Tasks 5–8 and publication bindings remain explicitly pending; this review does not accept those results or the other prepared links. |
| **Consult: inventory affected links, retain regression evidence separately** | **RESOLVED in the plan** | Task 6, Task 8 and the specs README name I-083 link 2 / I-0813-b and I-084 link 1 / I-0830-b as unavailable, with regression evidence outside the accepted-certificate field. No incident-wide verification shortcut is authorized. |
| **Consult: independent receiving boundary or narrower positive decision later** | **RESOLVED as an explicit deferral** | Ruling 10 and design §9.3 preserve both alternatives and say a wrong planner decision cannot establish missing operator delivery. Neither alternative is implemented or certified by this review. |

The event-direction r7 controls are appropriate for this target: unseen fallback/dispatch calls must fail to witness; custom mock calls must be unattributed; an audit-hook interval must be unavailable even when a genuine event is linked. Their success must not be described as complete call coverage.

## New findings

### 1. BLOCKER — a whole-namespace change leaves a former callable falsely current

**Locations:** `scripts/audit/incident_register/observer.py:652`, `:784`, `:843`; `certify.py:52`.

`test_clear_keeps_a_former_callable_falsely_current` reaches the complete defence runner's **accepted** verdict, with `certificate.ok=true`, no reasons and a linked `alert` event. Its frozen helper captures `transport.send`, clears the exact module's namespace, independently asserts that `send` is absent, calls the captured function through `map`, then restores the namespace. The sender keeps its output list in a default argument, so the send genuinely occurs while the binding is absent. Only the allowed production helper argument/return changes under the mutant; helper, transport and test stay frozen.

The watcher sees `DICT_CLEARED` and records `a watched namespace was replaced wholesale`, but it leaves `state[name]["current"]` unchanged. `_current_code` therefore still identifies the captured former function as current. Restoring the namespace before stop also prevents the end snapshot from exposing the transient error. Both twins fail at the declared assertion and restoration is green.

This directly violates `COVERAGE.unattributed_when`'s former-value rule. The module is an admitted exact ModuleType, the mutation is ordinary `dict.clear()`, and no observer/interpreter/stdlib function is replaced. It is inside the model even though the watcher records a gap. A gap ignored by positive certification cannot repair a falsely qualifying identity.

**Evidence:** `final-runs/test_clear_keeps_a_former_call0/out/acceptance.json`, its `mutant.events.jsonl` and frozen `tests/helper.py`. Linked sequence: entry 2, branch 3, event 5.

**Smallest fix:** on clear/clone, invalidate the current value of every affected binding immediately; retained callables may remain observation targets but must be unattributed until a trustworthy post-change resolution re-establishes the current binding. Do not re-read the old contents inside a pre-change watcher callback. A conservative unavailable state is sufficient; do not restore general absence coverage. Retain this full-run probe in the refusal direction.

### 2. BLOCKER — unavailable identities and incomplete values can become positive witnesses

**Locations:** `scripts/audit/incident_register/certify.py:143`, `:147`; `defence.py:48`; `certify.py:52`.

Two complete runs demonstrate missing enforcement at the consumer:

- `test_unavailable_boundary_category_is_certified` binds one exact function to two declared boundaries. The observer correctly gives **both calls unavailable identities**, with the reason `the callable is bound to several boundaries`. Requesting symptom category `unavailable` nevertheless produces **accepted**, `ok=true`, and an event link. No actual category was established.
- `test_an_incomplete_return_value_is_certified` returns 501 characters. The observer correctly records a truncated form with `incomplete=true`, length 501 and category `unavailable`. A return spec whose `value` equals that partial serialization still produces **accepted**, `ok=true`, and a return link. Direct equality at line 150 bypasses completeness entirely.

These violate the promised machine refusal of unattributed/incomplete witnesses. The first spec is a mistaken request, not evidence tampering; the spec checker presently permits it. The second is especially clear: an incomplete observation is accepted as a value witness without any reserved-category trick. The pair's `acceptance.unwrap` already refuses incomplete forms, but current defence does not use that safeguard.

**Evidence:** `final-runs/test_unavailable_boundary_cate0/out/` and `final-runs/test_an_incomplete_return_valu0/out/`; both have clean staged runs, unchanged frozen tests and a declared assertion failure. Their raw records themselves carry the unavailability/incompleteness.

**Smallest fix:** reserve `unavailable` so no symptom or classification rule can request it as a qualifying category, and independently require candidate identity/value completeness before category or value matching. Reject boundary identities carrying an attribution-failure reason. Apply the consumer checks inside `certify`, not only the spec validator, so direct certification cannot bypass them. Add both full-run controls plus a nested-incomplete return case.

### 3. BLOCKER — formatting class metadata executes application code while purity stays clean

**Locations:** `scripts/audit/incident_register/observer.py:132`, `:135`, `_safe`, `:476`, `:766`; `certify.py:12`; design §9.3 at line 126.

`type_name` reads class metadata through C descriptors, then interpolates the results in an f-string. Python permits an arbitrary object in a class's `__module__`; descriptor access does not make that object an exact string. Interpolation therefore calls its application-defined `__format__`.

`test_type_metadata_formats_application_code_inside_observation` returns an ordinary Result instance with a plain instance dictionary. Its class's module-name object increments an application counter in `__format__`. The complete return defence is **accepted**, with no errors/reasons. The observed mutant's captured output shows **`application_formats 1`**, the unobserved twin shows **`application_formats 0`**. Both purity endpoints report census 0, collection off and no signal handler. Both runs fail at the same declared assertion, so conformance accepts the behavioral difference.

There is no audit hook, collection/handler toggle, standard-library replacement, native extension or observer tampering. Application formatting is precisely what the serializer promises not to invoke; it is not listed outside the model. The certificate's no-application-callback qualification is false. `_err` uses the same `type_name`, so reading exception arguments safely does not complete its observational guarantee either.

**Evidence:** `final-runs/test_type_metadata_formats_app0/out/acceptance.json`, observed/unobserved stdout and mutant event records; the probe's source pins the counter side effect. The positive return is linked, but its eligibility is false.

**Smallest fix:** verify that both metadata values are exact strings before concatenating them. Use a fixed safe fallback for error reporting and an unavailable/incomplete representation when a value's required type identity cannot be read safely. Never coerce unsupported metadata with `str`, `repr` or formatting. Add a direct callback-count control and this full-run rejection/control; audit-hook census alone cannot establish serializer purity. The existing twins remain useful but equal states/failure locations must not be described as equal application behavior.

### 4. SHOULD — qualify return attribution as source-location identity

**Locations:** `scripts/audit/incident_register/observer.py:856`, `:862`, `_exit`; `certify.py:45`.

`test_a_cloned_return_function_is_certified_as_the_declared_function` creates a FunctionType clone of `grade` with different globals. `deliver` invokes the clone; the frozen test independently verifies that the original `grade` was not called, then calls the original and verifies its return is still `done`. The clone returns `defer` under the mutant. Defence accepts a linked return labelled `grade`; raw returns are `defer` (clone), then `done` (original outside the entry).

The boundary shared-code check does not apply to entries/returns. I am **not counting this as an additional blocker**: entry identity is explicitly defined by file/qualname, and the shared-code refusal explicitly names boundary code. The execution does establish a return from the declared source code. It does not establish which Python function object, globals or receiver ran that code. The present phrase `return of the declared function` can overstate this distinction.

**Evidence:** `final-runs/test_a_cloned_return_function_0/out/`, including the independent frozen assertions and the two recorded returns.

**Smallest wording fix, applicable verbatim:**

> Entry and return identity are source-file and qualname identities, not Python function-object, globals or receiver identities. The fixture review must establish that the observed code invocation belongs to the declared contract and selection. A claim requiring exact callable or receiver identity needs an additional witness or reads unavailable.

If exact return-function identity is intended instead, add binding/identity enforcement and refuse the clone. Do not infer it from the successful boundary clone control.

### 5. SHOULD — the justification incorrectly says every prior false certificate was absence

**Locations:** plan ruling 10 at line 87; design §9.3 at line 132; `scripts/audit/incident_register/certify.py:4`.

The prior r6 report **lines 66–70** explicitly records an accepted **event** certificate for a distinct FunctionType clone while the frozen test verifies that the declared sender never sent. Thus the repeated `Every false certificate from r5 to r7 was an absence certificate` statement is false. The earlier closure/purity failures also applied to retained modes. This does not invalidate Eric's narrowing decision; it invalidates the inference that positive-only certification automatically removes all earlier categories of risk.

**Smallest replacement, applicable verbatim to the repeated justification:**

> The r5–r7 reviews found absence-coverage failures and a false positive event attribution. Phase 1 defers absence certification; attribution, observational purity and execution closure remain independent obligations for retained positive witnesses.

## False greens

The submitted green suite and 161-row sweep coexist with the new accepted executions. I verified that the retained sweep has **161 unique KILLED rows**, exactly matching the current list, and that every listed mutation anchor is unique and its mutated file compiles. I independently exercised all **24 additions**, the **four re-anchored mutants O2/O10/O18/O28**, and relisted **O4** against their selected regression nodes. All 29 are assertion-shaped kills with the expected targeted inventory. This is targeted sensitivity evidence, not a fresh 161-run whole-suite sweep.

Retiring the twelve absence-only mutants is consistent with code removal and ruling 10. O4 is correctly live again because non-production starts may be disabled. C14 now pins the exact refusal rather than allowing an incidental ValueError; O49 pins the child's exit status rather than letting CalledProcessError be the only failure. The class-change control genuinely changes the mock's runtime type through the object descriptor; it is not a vacuous assignment to Mock's `__class__` property.

The uncovered cases explain the false greens. Existing former-value controls exercise per-key stores, not clear/clone invalidation. Incomplete-value controls test observer classification and pair unwrapping, not the certifier's direct value comparison. Unidentified-event controls request a real category, not the reserved unavailable category. Purity tests cover hooks/collector/handlers but not non-string class metadata. The twins compare node states and failing locations; the formatting counter shows that these can agree while application effects differ.

The eleven E77/L03 sensitivity results support those predicates within their stated component scope. They do not establish recorder soundness, complete absence, real delivery, historical occurrence or published fixture binding. No new fixture-predicate vacuity was found.

## Rulings

**Ruling 10 is a reasonable register boundary, and its absence refusal is implemented.** I am not reopening general in-process absence certification. Task 6/8's two unavailable links and the decision-versus-delivery distinction are appropriate. Other worklist links still need their own symptom choice, run and review; seven eligible prepared specs are not seven accepted certificates.

**The bounded model is presently not enforced as advertised.** Findings 1–2 contradict explicit current-value/completeness obligations. Finding 3 contradicts the retained no-application-formatting/callback requirement. These are inside-model failures, so a reviewed benign real pilot cannot substitute for their refusal. Finding 4 needs precise return-identity wording, rather than an inferred exact-function guarantee.

**The closure screen is an acceptable reviewed precondition, not a proof.** I reproduced and inspected its two sys.modules registration hits. Its alias/reflection limitations are disclosed. Before the census, Python site initialization and the admitted startup hooks execute; I inspected the actual `_virtualenv` hook, whose reviewed source installs the distutils/setuptools import finder and no audit hook. Cache/alternate-form checks remain covered by the restored controls. Pre-bootstrap audit hooks, transient collector/handler changes, stdlib replacement and restored mock-class changes are explicitly excluded. I found no evidence that the eligible I-0830-d pilot uses such an excluded mechanism, and no reason those exclusions alone make this retrospective cooperative-code model unfit. They do not exclude ordinary namespace clearing, partial values or unsafe application metadata.

**Reports and fixtures remain bounded by what they witness.** Ruling 6's per-citation/report-role restrictions and ruling 7's reviewed value-match qualification remain appropriate; no new rule allowing a repair report to establish an unwitnessed incident was found. The machine certificate defects above are the current overstatement. Publication's result review, exact pin/closure equivalence, fixture bindings and unavailable coverage rows remain mandatory. Wording edits alone cannot produce SIGN WITH EDITS while findings 1–3 remain executable.

## Answers

- **Attribution:** ordinary current-value, per-key former-value, several-boundary, shared-boundary-code, receiver and exact-callable controls pass. Whole-namespace invalidation fails. Return attribution is currently source-location identity, as the clone probe measures.
- **Mocks:** raw-MRO effective-call checks, held class, hold-time gaps and call-time unavailability have sensitive controls. Their unavailable result still needs the certifier-level refusal in finding 2.
- **Purity and stop:** post-bootstrap hook census, collection quiescing in both twins, endpoint signal checks and the original stop/error paths pass. They do not prevent application formatting via class metadata. `_err` must use the corrected safe type-name path too.
- **Absence:** five independently rebuilt requests and all retained refusal controls reject; direct refusal uses the exact exception/reason and the runner writes rejected artifacts without event runs. A missed call is not counted as a finding.
- **Pair and pilot:** the byte-equivalent pair accepts 22/22 with 88 clean endpoint records total; I-0830-d accepts an empty-list return witness. These are synthetic tooling measurements, not approval of register publication or the deferred absence links.
- **Mutants and vacuity:** the retained 161/161 census is internally consistent; 29 selected mutations are independently sensitive; the restored 356-node run has 334 passes and 22 strict XFAILs. No fresh full fast-suite or whole-sweep claim is made.

Review boundary and cleanup verified: no `data/` reads, SSH, network, `gh`, herdr control, main tracked-file edits or main commits. Synthetic commits/worktrees were confined to authorized probe repositories. Throwaway tooling/source mutations were restored byte for byte. The required `git worktree remove --force` removed wt8; both its directory and registration are absent. Main remains **4544fd48e92153eb76e5fc042dec677268554025**, with unchanged tracked status. Only the requested report and reviewer-owned probe/evidence outputs were added to the main working directory.

DONE
