## Verdict

**BLOCK.** The original r5 examples now receive effective repairs. Rev 6 nevertheless issues accepted false absence and event certificates, and accepts a pair whose executed external dependency changes. Store-time dict watching does not cover replacement of its root or Python attribute dispatch; a callee's code object does not identify its callable; the reviewed `.pth` source is not necessarily the hook Python executes. These are measured failures of the certificate claims, not requests for additional hypothetical hardening.

Reviewed plan rev 6 at **b48ee11409faed6fce9c8da7bd7779a162e031ae**, and the authorized detached checkout at **cd9456b**. Verified an empty diff from **35bc1fe** to **cd9456b** for tooling, tooling tests, incident fixtures/meta-tests, registry, sweep and predicate-sensitivity script. File:line references below use cd9456b; plan references use b48ee11.

Measured offline with Python 3.12.13, pytest 9.0.2, jsonschema 4.23.0 and TZ=America/New_York:

| Check | Result |
|---|---|
| Submitted tooling + incident fixtures + meta-tests, including final run after restoration | **293 passed, 22 xfailed**: 250 tooling passes, 40 fixture passes, 3 meta-test passes |
| Submitted adapted r5 module | **29 passed**, included above; classifier module **12 passed** |
| Independent final probes | **14 passed**; several intentionally assert false acceptance and are defect witnesses |
| Actual expected-failure driver on a synthetic repository with copied pinned source/scripts/fixtures/config/lock/registry and its own venv | **accepted 22/22**, 20 `value_match`, 2 `exception_shape`, 40 `passed_nodes`, no reasons |
| Submitted targeted sweep mutations | **28** independently fail their intended regressions; unmutated selected regressions pass |
| Both defence/replay `completed and` guards removed together | **40 tests passed**; redundant on those exercised paths |
| Observer end-binding guard removed | New complete probe changes from **rejected** to **accepted false absence**; **not equivalent** |
| Submitted fixture predicate/sensitivity script | **All eleven as required**: seven predicate weakenings caught; two formatter mutants fail three ordinary nodes each; both repair sketches strict XPASS with controls passing |
| Drafts and retained deployment corpus | **119 records, zero draft errors**; 192 runs, 48 observations, zero clean rollbacks |

The synthetic real-pair commit is **fd93917e52740c8f4d5f6c5b12be4465d1d48ab7**. Its copied inputs were compared byte for byte with the restored checkout before execution. This is not a claimed run at a publication commit. The retained pair at 35bc1fe has the same 22/40 counts and connection distribution. The retained sweep has **134 KILLED rows**; I did not rerun all 134 whole-suite mutations or the full fast suite.

Evidence and reproducers are retained under **/private/tmp/bts-incident-register-r6-8eQJEt/**: `test_review_r6.py`, `probes-final.xml`, `probe-verdicts.json`, `submitted-final.xml`, `run_targeted.py`, `target-*.log`, `real-pair/`, and the copied retained-r5 scripts/XML. The complete false certificates are under `probes-final/`; each finding names its executable test. All source mutants were confined to authorized throwaway or synthetic worktrees and restored.

## r5 findings status

“RESOLVED” disposes the original failure shape; it is not a universal approval of the subsystem.

| r5 item | Status | Measured rerun and limit |
|---|---|---|
| **#1 Binding discovery / false absence** | **PARTIAL** | Retained frozen-helper rebinding now rejects with one qualifying send; shared-callable aliasing rejects with unavailable identity. Adapted captured-former-value, in-place `__code__`, unsupported instance, replaced mock call, cleared namespace and watcher-unavailable controls pass. O10–O12/O17–O23/O26–O27 targeted regressions are sensitive. New root/type/attribute-dispatch paths still hide real sends, and shared code falsely witnesses a different callable: findings 1–2. |
| **#2 Nested incomplete classification** | **RESOLVED** | Retained nested sequence identity becomes `unavailable`; the complete return-defence probe rejects. All six sequence/unsupported/depth identity/return controls pass. O24 is killed by a nested unsupported child; O25 by the nested map control. Recursive `unwrap` remains correctly unavailable. |
| **#3 `_resolve` application attribute code** | **RESOLVED for the resolver** | Retained instance path resolves to None with zero attribute callbacks. The adapted module-subclass, instance and `__dict__` property test passes, and O22 is independently killed. Application code can still run in callable registration, a different site: finding 4. |
| **#4 Executable closure / symlink targets** | **PARTIAL** | Retained executable `.pth` example raises `ClosureRefused`; ordinary untracked symlink target bytes change the manifest; the changed-helper pair rejects for the specific untracked-content reason. W10/W11 are independently killed. Reviewed hook source can execute unreviewed cached bytecode that exposes an unhashed external root, and a changed-dependency pair is accepted: finding 3. |
| **#5 Affirmative delivery predicate** | **RESOLVED for the stated fixture scope** | Retained named status, negative instruction and wrong game/date text become ordinary `other`; L03 no longer raises its dedicated exception for named status. Submitted separate previous-day, persisted-at-cutoff and wrong-record-day controls pass. Both actual formatter mutants give **3 ordinary failures, 1 XFAIL** across the four E77/L03 incident/positive nodes. Repair sketches still reach strict XPASS. |
| **#6 Chronology** | **RESOLVED for the requested relationships** | Retained observed-after-restoration and confirmed-before-attempt records now reject with the exact chronology reasons. Additional onset/detectable/observed controls and overlap/unknown preservation pass; V18–V20 are independently killed. All 119 drafts validate, including corrected I-077. Report meaning and the causal meaning of restoration remain review responsibilities. |
| **#7 Deployment precision** | **PARTIAL** | Retained `.100Z`/`.900Z` example now bounds installation by `(.100000Z, .901000Z]`; same-second stamps produce a nonempty interval. Ordering/overlap/rollback regressions pass, and DR1/DR2/DR5–DR8 are independently killed. Precision is still lost in timeline lookups and not checked against non-adjacent overlapping observations: finding 7. |
| **#8 Skipped baseline / inventory** | **RESOLVED** | Retained real all-skipped pytest run now classifies `SKIPPED` after adapting the changed classifier argument from count to node inventory. Assertion-plus-skip and changed-inventory controls pass; S1/S2 are independently killed. Compile rejection and S3 pass in the required direction. This does not establish coverage of the new shapes above. |
| **Task 8 publication pin note** | **RESOLVED as a plan gate; execution pending** | The rev 6 mapping explicitly retains receipt `resolved_ref` plus source/fixture/registry closure comparison, or a reviewed byte-equivalence mapping. `records.validate` still has no publication-ref argument; publication is expressly unfinished. The pin must actually be checked in Task 8. Update its older “five negative controls” wording to the current controls when binding the records. |

I reran the retained r5 file with only its checkout path changed: **18 failed, 1 passed**, deliberately retaining its old defect-asserting expectations. Sixteen failures show changed behavior; two are API-shape failures because `classify` now accepts a node list instead of a count. Rerunning those two with that argument adapted gives the expected **FAILED-OTHER** for the look-alike exception and **SKIPPED** for the skipped suite. The imprecise-claim overlap control remains valid. These exploratory negative expectations are not counted as failures of the submitted suite. Multi-assert legacy probes stop at their first changed behavior; the separately submitted controls exercise the later timestamp cases.

## New findings

### 1. BLOCKER — watched dictionaries do not cover the namespace root or attribute dispatch

**Locations:** `observer.py:333–371,491–499,536–582,662–666`; plan ruling 2 and Task 3.

Four complete `current_defence` probes return **accepted**, no reasons, and `certificate.ok=true` with **zero declared-boundary calls**, although the unchanged test verifies exactly one real send before its declared result assertion:

* `test_temporary_sys_modules_replacement_is_false_absence`: a frozen helper replaces `sys.modules` with a copied exact dict, inserts an alternate module, imports that module and calls its replacement sender through `map`, then restores the original root. Watchers remain attached to the old dictionary. The final binding check sees the restored original and misses the entire interval.
* `test_module_attribute_override_is_false_absence`: a supported ModuleType subclass returns the replacement sender through `__getattribute__` when a flag is set. Its raw `send` slot never changes. The flag store is not a step in the declared binding path.
* `test_class_path_with_custom_metaclass_is_false_absence`: the same mechanism through a supported class namespace and metaclass attribute lookup. Reading a raw class dict is observational, but does not prove that Python's actual lookup consumes that value.
* `test_transient_module_type_change_is_false_absence`: the module starts and ends as exact ModuleType, but temporarily changes `__class__` to the overriding subclass. This is missed too. Merely rejecting subclasses at registration would not close this variant.

The mutation changes only production's helper argument/return. The helper, its actual-send assertion and all other closure files stay frozen. Observed/unobserved stages conform. Corresponding `result.json` and raw events are retained under each final probe directory.

**Smallest fix:** include the owner of `sys.modules` in root-change coverage, and establish supported attribute-dispatch/type semantics for every namespace on the path throughout the interval. Record any unobserved root/type/dispatch transition as unavailable. Add all four complete probes. The supported shape cannot be defined solely as “module or class”; namespace dictionary stores do not observe every operation deciding an attribute's value.

The end guard is also **reachable**, contrary to the declared equivalence: `test_end_guard_has_a_reachable_python_path` leaves the copied root in place until fixture teardown. It currently rejects specifically for **binding changed without a watched store**. Removing only that guard produces **accepted false absence**; see `target-not-equivalent-end-guard.log` and `end-guard-mutant/**/end-guard-result.json`. Retain the guard and add this mutant to the sweep. It catches the persistent form, not the restored-root form above.

### 2. BLOCKER — the same code object is incorrectly treated as the same callable

**Locations:** `observer.py:417,529–534,676–688,706–725,752–758`; `certify.py:90–112`.

`test_same_code_unbound_function_is_false_event` constructs two distinct exact Python functions: declared `transport.send`, and an **unbound clone** created with `types.FunctionType(send.__code__, different_globals)`. The clone writes to an unrelated list; the declared sender's `sent` list stays empty.

A production mutant calls only that clone after the branch and changes the result. The frozen test asserts **no declared send occurred**, checks the clone's unrelated effect, then fails at its declared result assertion. Nevertheless, the complete defence returns **accepted** with an **event** certificate for a qualifying `dm` send. `_on_start` knows the code object and arguments; `_current_code(...) is code` succeeds, so identity is classified `pick`. Neither check establishes which function object ran.

This is false attribution in the event direction, not just conservative refusal of shared or former values. The clone never held the declared binding; the unattributable-call rule never runs. Artifact: `probes-final/test_same_code_unbound_functio0/false-event.json` and its raw stages.

**Smallest fix:** bind a boundary event to the actual callable/receiver invocation, or mark code-sharing ambiguity unavailable. A code object plus receiver and argument values is insufficient for plain functions. Add the full false-event probe, including its independent assertion that the declared sender did not run. Do not solve it only by matching globals: different closure/function objects may share globals as well as code.

### 3. BLOCKER — reviewed `.pth` source can execute unreviewed cached code and expose an unhashed root

**Locations:** `owned.py:42–52,264–287`; `acceptance.py:241–263`.

The allowlist reviews `_virtualenv.py` by **source** hash, while Python may execute its `.pyc`. Hashing that cache freezes its bytes, but does not establish that it implements the reviewed source or enumerate the roots it adds.

Two measured probes:

* `test_reviewed_pth_source_can_execute_unreviewed_cache_outside_closure` installs the exact allowlisted source and `import _virtualenv` line, plus valid timestamp/size-matching cached code that adds a private external path. The actual owned-venv Python **-B** executes that cache (a marker proves it), imports external `local_dep` as VALUE=1, then VALUE=2 after its source changes. Both environment fingerprints are **identical** and no `ClosureRefused` is raised.
* `test_pair_accepts_changes_imported_by_unreviewed_cache_of_reviewed_hook` runs a complete synthetic marked/unmarked pair. The marked test imports VALUE=1 and rewrites that external source; the unmarked test imports VALUE=2. The pinned `.pth`, allowlisted source and cached hook bytes remain unchanged. The pair returns **accepted**, no reasons. Artifact: `probes-final/test_pair_accepts_changes_impo0/pth-cache-pair.json` and its stages.

These operations use only reviewer-owned synthetic environments. This is a closure-discovery bypass; I am not claiming a ctime digest-memo bypass or that the submitted real venv contained this cache.

**Smallest fix:** review/bind the hook code that will actually execute, including applicable cache variants, or refuse/remove applicable hook caches before establishing the closure and starting evidence subprocesses. Reject cached code that is not the reviewed source's compiled form. `-B` alone is insufficient: it stops writes, not reads. Add both direct actual-import and complete-pair controls; an allowlist test that edits only `.py` cannot catch this.

### 4. BLOCKER — callable registration runs application code from a forged type name

**Locations:** `observer.py:472–478,501–527,556–581,752–758`.

`_code_of` chooses its permitted types by **module.qualname strings**, then accesses `obj.__code__` / `obj.__func__`. Those strings are assignable application metadata.

`test_function_name_spoof_runs_application_code_in_watch_callback` uses an ordinary application object whose class is labelled `builtins.function` and whose `__code__` is a property. Assigning it to a watched module's sender runs that property **once inside the real dict-watcher callback**. The callback accepts the returned code and records no observer error. It is not a Python function.

This is outside the repaired `_resolve` site, which does read raw namespaces without those hooks. I measured callback execution, not a complete accepted certificate caused by this side effect. Nonetheless it directly falsifies the trusted observer's no-application-code requirement and permits re-entrancy/application effects during a pending namespace store.

**Smallest fix:** identify supported callable types by exact runtime type identity (`types.FunctionType` / `types.MethodType`), then use their safe native members. Treat other objects as unavailable; display names are not type checks. Apply the same restriction consistently to receiver/current-code checks. Add the watcher-callback probe and an exception/cleanup control.

### 5. BLOCKER — a live `_thread` worker is omitted from absence coverage

**Locations:** `observer.py:632,667–673`; `certify.py:119–127`.

`test_low_level_worker_is_not_counted_and_absence_is_accepted` starts a real worker with `_thread.start_new_thread`. It signals that it started, waits on a threading.Event, and has not registered a Thread/DummyThread via `threading.current_thread`. The production entry completes; the declared result assertion fails while this worker is still waiting. `threading.enumerate()` does not include it.

The complete defence returns **accepted**, with `certificate.ok=true`, completed invocation and zero calls. The raw mutant `obs_end` reports zero outstanding/pre-existing threads. Frozen fixture teardown then releases the worker, waits for completion, and verifies exactly one real send. Thus the certificate accepted absence while declared work was outstanding, under a coverage statement that expressly makes any other live thread unavailable.

Artifact: `probes-final/test_low_level_worker_is_not_c0/raw-thread-result.json` and the mutant events. Observer-off stages have the same test outcome; they do not repair the missing census.

**Smallest fix:** include live interpreter thread states/frames, not only threading's registered objects, when establishing the end-of-interval census; refuse absence if a live thread cannot be accounted for. Add this real worker probe and a pre-existing low-level worker control. Do not require the observer to call `current_thread` as a side effect to make its census complete.

### 6. SHOULD — failed observer startup leaves watchers and its monitoring ID installed

**Locations:** `observer.py:265–276,601–622,625–651`.

`test_failed_start_leaves_real_watchers_and_monitoring_id_installed` makes the callable-registration callback in finding 4 raise RuntimeError during `_track`. `start()` has already reserved monitoring ID 4 and installed both real CPython watchers, but has not set `active`. It propagates the exception. Calling `stop()` returns immediately, leaving both watcher IDs non-None and `sys.monitoring.get_tool(4) == 'w15-observer'`.

The pytest wrapper calls `mon.start()` before its try/finally, so it supplies no cleanup for this path either. The probe explicitly removes its own leaked resources afterwards. I did not claim this failing start produces an accepted certificate; it can contaminate later nodes and continue callbacks beyond the intended lifecycle.

**Smallest fix:** make acquisition/registration transactional and clean partially acquired resources in a finally independent of `active`. Put startup inside the wrapper's protected lifecycle. Unwatch/clear each acquired resource independently so one cleanup failure does not skip the others. Add partial-install, failed-track and callback-error cleanup tests using real successful acquisitions where possible.

### 7. SHOULD — deployment precision is only partly carried through ordering and lookups

**Locations:** `deploy_runs.py:149–157,187–195,198–249`; plan Task 4/Task 3 deploy row.

Two executable under-refusals remain:

1. `test_deploy_nonadjacent_overlapping_disagreement_is_allowed`: run 1 observes A at `00:00:00Z` (the whole second), run 2 A at `.100Z`, run 3 B at `.500Z`. The coarse A interval overlaps B and cannot order their disagreement. Only adjacent pairs are checked; the intervening A point shields the disagreement, and all three observations are returned. A symmetric test on adjacent pairs is not an all-overlap test.
2. `test_live_at_claims_floor_of_uncertain_observation_is_observed`: a deployed B log line at `00:00:01Z` has an actual instant somewhere in `[01,02)` by `_instant`'s own rule. `installed_timeline` stores its floor as the segment boundary and `live_at(..., '00:00:01Z')` returns **B, basis observed**. A log observed at 01.750 does not establish B at 01.000. `first_live` now widens correctly; the timeline's exact-observation shortcut still overstates it.

The retained corpus still yields 48 points and no rollback cases. These synthetic examples do not establish that any of the current 96 draft install bounds is wrong; those bounds use `first_live`, a separate result from the overclaimed `live_at` lookup.

**Smallest fix:** carry each observation's precision interval through timelines/lookups and refuse every cross-run conflicting overlap, including non-adjacent/nested intervals. Within an uncertain observation interval, return a supported bound/unknown/transition rather than an exact `observed` state at its floor. Add both probes. Preserve the effective fractional extraction, nonempty first-live bounds, timestamp ordering and rollback identity controls.

## False greens

| Green claim | Independent result |
|---|---|
| Every store deciding a supported binding is watched | Root replacement, attribute dispatch and transient module type changes hide a real send in four accepted complete defences. |
| Calls that cannot be attributed to a binding are unavailable | A distinct unbound function with the same code gets a qualifying identity and an accepted event certificate. |
| A reviewed executable `.pth` hook has a frozen covered closure | Reviewed source executes unreviewed cached code; an imported external dependency changes across an accepted pair. |
| Watcher callbacks are observational | An application `__code__` property runs during a real dict callback without observer error. |
| `obs_end` counts every other live thread | A low-level worker remains live, is counted as zero, and sends after the accepted interval. |
| The end-binding guard is equivalent/unreachable | A pure-Python root replacement reaches it; disabling it changes rejection to false acceptance. |
| Printed deployment precision is preserved throughout | Non-adjacent conflicts pass and timeline lookup labels an uncertain floor `observed`. |

The retained 134/134 result is real for its listed mutations. It does not include these shapes or the now-proven non-equivalent end guard. My 28 targeted checks support the named controls; they do not prove all 250 tests non-vacuous.

One reviewer test selection needed correction: disabling `_complete`'s sequence recursion does **not** break a 51-item sequence, because its own top-level `incomplete` flag already refuses it. That selected case remained green for the right local reason. I switched O24 to the nested unsupported-child case, which independently fails and specifically exercises recursion. The submitted sweep already includes effective unsupported/depth cases; this is not a D13-style finding against its matrix. O25's map recursion and the receiver-reason O11 check also fail at their intended assertions. No additional submitted vacuous-test defect is established by these targeted checks.

The two declared equivalent guards have different dispositions: removal of both completion conditions still leaves all 40 defence/replay tests green because reason recording rejects aborted paths; retain that limited equivalence. The end-binding guard is not equivalent, and its “every supported change IS a watched store” justification must be removed.

## Rulings

**Ruling 2 — implementation does not satisfy the stated coverage.** The external observer remains the right trust boundary, and the original r5 stores, nested incompleteness and resolver callbacks now have effective controls. But dictionary paths are not full Python binding semantics, and callee code does not identify the function object. An unavailable rule only protects shapes actually discovered; finding 1's shapes are undiscovered and falsely certified. Ordinary caller/receiver, former-value and two-declared-binding controls work. The cloned-function event demonstrates a wrongly allowed legitimate-looking identity, not a need to weaken conservative absence refusal.

**Ruling 8 — the owner's decision is clear; its cost-if-wrong statement is unproved.** Continuing the full implementation is authorized by the plan, but “each [dynamic shape] is answered by a fail-closed rule” is presently false for the complete probes. Keep the objective and label complete coverage as an acceptance requirement until these executable failures close. This review does not substitute a narrower objective for Eric's decision.

**Self-review — specific corrections are effective, but the equivalence claim is false.** I-077 now places onset and first detectable time at the 18:05 submission cutoff; the 18:10 check remains the mechanism. Its record explicitly retains the historical inference limit. All 119 drafts validate. The invalid-mutant compile check and S3, receiver reason O11, symmetric ordering/overlap DR1/DR2/DR7, P7 wrong-record-day control, and O26/O27/DR8 protections have effective regressions; the named target mutations fail them. The namespace-property refusal removal is consistent with the raw ModuleType descriptor test. Finding 7 limits the symmetric overlap repair to the pairs it actually checks. The newly tested end guard contradicts the self-review's equivalence rationale.

**Chronology — accept the r5 repairs at their explicit structural scope.** Certainly reversed onset/detectable/observed/restoration and alert attempt/confirmation/failure relationships reject; date-overlap/unknown controls pass. A separate exploratory probe still validates an alert predating onset when machine detection is unknown. I am not promoting that to a defect without a contract ruling on anticipatory alerts; the plan should avoid claiming validation of every conceivable event relationship. Neither zero errors nor a qualified report proves that a report describes a deviation or that an install ended it. Those remain evidence-review obligations under ruling 6.

**Fixture scope — accept the repaired E77/L03 predicates for this declared single-pick format.** The recommendation template is independent of the formatter under test; current recipient/selection/date/receipt and both timestamp bounds are checked. Named negative traffic is no longer dedicated incident evidence. The E77 noon re-fetch reaches strict XPASS plus one passing control; the L03 delivery guard reaches strict XPASS plus eight passing controls. These are throwaway sensitivity sketches, not proposed production changes or proof that every future formatting variant/repair will fit this oracle. Registry reviews were renewed for the repaired predicate; the pair is still explicitly `exception_shape` for these two incidents.

**Task 8 — retain the publication pin as an actual gate.** An accepted receipt and matching node names/registry are not source equivalence. Check the receipt's resolved ref and source/fixture/registry closure against the publication build, or retain a reviewed byte-equivalence mapping. The rev 6 mapping records this requirement; the binding validator alone still does not implement it. Task 8 remains pending. Its operative control list should use today's expanded controls rather than the old five. The new false certificates must not be attached to published records as accepted reviewer decisions.

Minimum plan edits after fixing the code: remove the end-guard equivalence bullet; replace rulings 2/8's achieved-completeness assertions with an explicit acceptance gate backed by the new root/type/dispatch/callable/thread/cache probes; update the deployment precision claim to include all overlapping observations and timeline queries; update Task 8's control list and make the source pin check reviewable. Code changes for findings 1–5 are prerequisites; prose alone does not close them.

## Answers

1. **Were the eight r5 findings fixed as framed?** Their retained concrete examples are repaired. Findings #1, #4 and #7 are PARTIAL because the broader binding, immutable-closure and precision requirements still have new executable counterexamples. Nested classification, raw resolver access, affirmative delivery, the requested chronology relationships and skip/inventory rejection are effective.

2. **Which stores/calls are missed or misattributed?** Replacement of `sys.modules` itself, supported-module/metaclass dispatch and a transient module type change evade the dictionary path. A distinct FunctionType clone sharing the sender's code is falsely attributed as the sender. A low-level worker evades the thread-object census. Complete results and independent side-effect assertions are retained.

3. **Are installation/callback/removal safe?** The submitted ordinary watcher and fail-closed controls pass, but forged type metadata executes an application property in the real callback, and a failed start leaks acquired watchers and the monitoring ID. These sites need exact type checks and transactional cleanup.

4. **Can repairs still reach the required branch, and can unrelated status reach the dedicated exception?** Both submitted repair sketches reach strict XPASS with controls passing. Both tested status formatter mutants now fail ordinarily in both positive controls and the L03 incident node; E77's incident remains its expected bad path because it never reaches that formatter. No new status-message false green was found.

5. **Does 134/134 establish rev 6 completeness or both equivalences?** No. It establishes the retained listed outcomes; 28 intended regression kills were reproduced. The completion guards remain redundant on 40 tested paths. The end-binding guard is reachable and independently necessary on the new persistent-root example.

6. **Can a rule let evidence establish more than it witnessed?** Yes: accepted absence over actual sends/outstanding work, an event attributed to a different callable, an accepted pair over a changing external dependency, and an `observed` deployment label at an uncertain floor. The recorded reviewer decision remains separate; rejecting these artifacts is required and does not make the runner's false claims correct.

7. **Next action:** fix findings 1–5 and watcher lifecycle, add the complete probes and the end-guard mutant, carry deploy precision through all consumers, rerun the affected evidence, then perform Task 8's pinned publication/result review. No production repair, commit, push, deploy, Route R/box read or network operation is authorized by this verdict.

Final verification: the submitted JUnit has 315 cases, zero failures/errors and 22 expected-failure skips; the independent final JUnit has 14 cases with zero failures/errors/skips. The source mutations were restored before both final runs, and wt6 had a clean status before removal. The authorized wt6 was removed with `git worktree remove --force` and is absent from the filesystem and Git worktree inventory. The real-pair synthetic worktree was also removed; its evidence remains in the scratch directory. Main remains at b48ee11409faed6fce9c8da7bd7779a162e031ae with a clean tracked checkout. No data/ contents, network, SSH, gh, production operations or main-repository commits were used.

DONE
