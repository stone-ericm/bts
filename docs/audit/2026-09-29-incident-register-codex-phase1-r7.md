## Verdict

**BLOCK.** The original r6 counterexamples are repaired, but rev 7 still accepts false absence certificates. Exact module/class types do not establish raw-dictionary lookup semantics; checks at Python function starts miss a dispatch change performed and restored between starts; mock recognition admits an already overridden `__call__`; and the observer invokes application audit hooks that can send an unrecorded pick. These are executable contradictions of ruling 8's acceptance requirement.

Reviewed plan rev 7 at **f6d881f1ad2cb30483afb69f9cdf7859e0daec4a**, code **1c6c1835d4711e9293088c85e3ffef3274d39e0f**, and authorized worktree **b991993447f57712fa3a21ebd158b99eebe41086**. The tooling, tooling tests, incident fixtures/meta-tests and registry have an empty diff from 1c6c183 to b991993. The sweep script and evidence differ: b991993 retires O4. Code file:line references below use b991993; plan references use f6d881f.

Measured offline with Python 3.12.13, pytest 9.0.2 and jsonschema 4.23.0:

| Check | Measured result |
|---|---|
| Submitted tooling + incident fixtures + meta-tests, both initially and after all source mutations were restored | **309 passed, 22 xfailed**: 266 tooling, 40 fixture and 3 meta-test passes; no errors or ordinary skips |
| Retained r6 probes, changing only the checkout path | **13 failed, 1 passed** under their original defect-asserting expectations; dispositions below |
| Submitted adapted r6 controls | **16 passed**, included above |
| New complete-defence counterexamples | **5 passed**; these tests intentionally assert false acceptance, not correct behavior |
| O28–O38, W12–W14, DR9–DR10 | **All 16 fail their intended targeted regressions**; the 13 distinct unmutated selected tests pass |
| Retired O4, retired O22, unlisted end-binding guard, removed separately | **17 targeted tests pass in each case**, including the persistent-root control; baseline also 17 passed |
| Actual expected-failure driver on a synthetic copy of pinned inputs, with its own venv | **accepted 22/22**, 20 `value_match`, 2 `exception_shape`, 40 `passed_nodes`, no reasons |
| Fixture sensitivity script | **All eleven as required**: seven predicate mutants caught; both formatter mutants give 3 ordinary failures and 1 XFAIL; both repair sketches strict XPASS with their controls passing |

The synthetic pair commit is **39dec325c6cd064612b1aedef53dfbe37382bf6b**. I compared **315 copied input files** byte for byte with the restored checkout, including source/scripts/config/lock/fixtures/registry. The reviewed-hook cache was absent after the pair. This is a measured synthetic execution of equivalent inputs, not a receipt at the publication commit. I did not rerun the full fast suite or all 149 whole-suite mutant runs. The retained sweep contains 149 measured rows, **148 KILLED and O4 SURVIVED**; the current sweep list contains **148 mutants** after that retirement.

Reproducers and evidence are retained under **/private/tmp/bts-incident-register-r7-WAemeY/**: `test_retained_r6.py`, `test_review_r7.py`, `submitted.xml`, `restored.xml`, `retained.xml`, `new-final.xml`, `new-final/`, `run_targeted.py`, `target-*.log`, `equivalence.py`, `equivalence.log`, `audit_cost.py`, `cost_without_hook.py`, `stop_audit.py`, `real-pair/`, `fixture_mutants.log` and `evidence-summary.json`. The probe file imports the authorized wt7 path; rerunning it requires recreating that checkout at the pinned commit. All throwaway source edits were restored before cleanup.

## r6 findings status

“RESOLVED” disposes the original failure shape. It does not certify universal coverage.

| r6 item | Status | Measured rerun and limit |
|---|---|---|
| **#1 Root and attribute dispatch** | **PARTIAL** | Both copied-root examples now reject for a qualifying send. Initial custom module/metaclass dispatch rejects for unsupported namespace; the original transient Python override rejects for changed dispatch. All six adapted controls pass, and O28–O32 are sensitive. New fallback/inheritance and C-dispatch variants still obtain accepted absence: findings 1–2. |
| **#2 Same-code function falsely attributed** | **RESOLVED** | The original clone event probe now rejects: no qualifying pick event. The adapted clone control passes; O33 is sensitive. Shared code receives unavailable identity. Its GC scan has a distinct observer-side-effect defect: finding 4. |
| **#3 Reviewed-hook bytecode bypass** | **RESOLVED** | Both retained cache probes now raise `ClosureRefused`. Source/cache purge, complete-pair refusal and alternate-importable-form controls pass; W12–W14 are sensitive. The real synthetic driver purges its cache and accepts 22/40 without recreating it. |
| **#4 Callable type-name spoof** | **RESOLVED** | The original property's call list is now empty. Exact-type rejection records a gap, and O34 is sensitive. The broader no-application-code guarantee still fails through audit hooks, a different mechanism. |
| **#5 Low-level worker omitted** | **RESOLVED** | The retained `_thread` absence probe now rejects for an outstanding worker. New and pre-existing low-level worker controls pass; O35 is sensitive. Application audit hooks around the census remain unsupported: findings 4–5. |
| **#6 Failed-start cleanup** | **RESOLVED for acquired-resource cleanup** | The old forged-type startup no longer raises, so its old `pytest.raises` assertion fails before its leak assertions. The adapted test forces a real resolution failure and verifies monitoring-id/watchers cleanup; O36/O38 fail those assertions, and O37 detects the guarded end-resolution exception. The additional universal “stop never raises” claim is false: finding 5. |
| **#7 Deployment precision** | **RESOLVED** | Retained non-adjacent disagreement now raises `ValueError`; an uncertain observation floor returns `sha=None`, `basis=within_observation_precision`, candidate B. Adapted controls pass; DR9/DR10 are sensitive. The within-unit state remains unknown, and continuity after it remains explicitly assumed. |
| **Ruling edits** | **RESOLVED as requested wording** | Ruling 8 now says complete coverage is an acceptance requirement, not achieved; ruling 9 records Eric's decision. The original end-guard probe is caught earlier by the root watch, including when the guard is removed. Ruling 2's exact-dispatch explanation still overstates what its checks establish. |
| **Task 8 note** | **RESOLVED as a pending publication gate** | It names the twelve current E77 cases plus the record-of-another-day control and retains the published-ref/source/fixture/registry closure check or reviewed byte-equivalence mapping. Current controls pass in the measured pair. Rebinding the draft records and performing publication are expressly pending, not accomplished by this review. |

The retained r6 file has no API incompatibility requiring adaptation. Thirteen old expectations fail because behavior changed; the remaining chronology probe still passes under the plan's explicit lack of a ruling on anticipatory alerts. A multi-assert old probe can stop before later checks: the independently executed adapted controls supply those checks. These failures are not submitted-suite regressions.

## New findings

### 1. BLOCKER — exact modules/classes still have lookup behavior outside their own dictionary

**Locations:** `scripts/audit/incident_register/observer.py:337`, `:350`, `:554`, especially `:560`; plan ruling 2.

Two complete `current_defence` probes return **accepted**, no reasons, `certificate.ok=true`, `calls_observed=0`, `completed=true`. The unchanged test independently asserts exactly one pick send before failing at its declared result assertion:

- `test_exact_module_getattr_is_false_absence`: an exact `ModuleType` defines module-level `__getattr__`. The frozen helper deletes `transport.send`, obtains its replacement through the fallback, calls it through `map`, then restores the original. The deletion callback sets `current=None` and assumes nothing can be called through the binding. `__getattr__` runs while every owner still has the admitted exact type; its returned callable is never held.
- `test_exact_class_inheritance_is_false_absence`: an exact-`type` class has an own `send` and a base class with a different `send`. Deleting the own attribute exposes the inherited sender. Calling it and restoring the own attribute similarly hides the send. The base's lookup contribution is absent from the watched chain.

Only the production helper argument/return is mutated; the helper, transport and independent send assertion stay frozen. Observed/unobserved green, mutant and restored stages conform. Raw mutant events contain no boundary or coverage-gap record. Evidence: the corresponding `new-final/test_*0/result.json` and `out/` stages.

**Smallest fix:** treat deletion/unresolved suffixes as coverage gaps, rather than assuming no callable exists. Until fallback/MRO/descriptor resolution is covered, refuse those shapes explicitly. Exact runtime type alone is insufficient. Add both complete probes in the refusal direction, with their independent send assertions retained.

### 2. BLOCKER — dispatch can change and be restored between two Python starts

**Locations:** `scripts/audit/incident_register/observer.py:574`, `:750`; plan ruling 2 and the dispatch self-review.

`test_c_dispatch_between_python_starts_is_false_absence` gives a module subclass this **C-implemented** override:

```python
__getattribute__ = functools.partial(dict.__getitem__, {'send': replacement})
```

Inside one frozen helper invocation, it sets `transport.__class__` to that subclass, captures `transport.send`, restores exact `ModuleType`, then calls the captured replacement through `map`. Assignment and overridden lookup cause no intervening Python function start. By the replacement's start, the owner's exact type is restored. Its raw `send` entry never changed.

The full defence is **accepted**, with zero observed calls and no gaps, although the frozen test asserts the actual pick send. This uses ordinary Python statements and stdlib C callables; it does not modify the observer, use ctypes to forge object internals, or require a C extension written for the probe.

**Smallest fix:** cover dispatch/type changes at the operation that changes or consumes them, rather than sampling only at `PY_START`. A check covering Python instruction/type-setter transitions can refuse this probe; a broad coverage claim additionally needs refusal of transitions whose lookup cannot be observed. Retain the keep-every-start control, but add this between-starts control. An end snapshot cannot recover a restored transient state.

### 3. BLOCKER — mock recognition does not verify the effective `__call__` already installed

**Locations:** `scripts/audit/incident_register/observer.py:481`, `:528`, `:752`.

`test_initial_custom_mock_call_is_false_absence` constructs a `Mock` subclass with its own Python `__call__` before observation. The mutant binds that existing object to `transport.send`; its custom call invokes the replacement sender. `_is_mock` admits every subclass of `NonCallableMock`, while `_hold` assumes the call will start the standard `mock_call_code`. Watching future class stores cannot detect an override already present when the object is first held.

The complete defence again returns **accepted**, zero observed calls and no gaps. The frozen test verifies exactly one real pick send. This differs from the existing replaced-mock-call control, which changes `__call__` after its class dictionary is watched.

**Smallest fix:** inspect the effective mock `__call__` through trusted raw MRO/descriptor reads before treating the object as a supported mock. Accept only the verified supported implementation, or register its actual implementation with sound receiver identity; otherwise record a gap immediately. Keep the later-store guard and add this pre-existing-override probe.

### 4. BLOCKER — observation executes application audit hooks and hides their effects

**Locations:** `scripts/audit/incident_register/observer.py:493`, `:679`, `:725`, `:783`; `scripts/audit/incident_register/defence.py:114`; observer's no-application-code contract at `observer.py:7`.

The direct `audit_cost.py` control installs a harmless application `sys.addaudithook`. One monitored sender invocation executes that application callback for **`sys._current_frames`, `gc.get_referrers`, `sys._current_frames`**. Thus exact callable/namespace types do not make these observer operations free of application callbacks.

`test_audit_hook_sends_inside_observer_but_absence_is_accepted` demonstrates the consequence through the full defence. The frozen transport's audit hook sends a pick when a GC-referrer audit occurs while armed. The mutant arms it and invokes the declared sender with an alert. During the sender's `PY_START` observation, `_code_shared` emits the audit event; the hook sends the pick inside the monitoring callback, where its Python starts are not recorded. The recorded boundary is only the alert.

An independent frozen assertion verifies **pick then alert** in the observed mutant. The unobserved mutant sends only the alert; its assertion explicitly accounts for that difference, and both runs fail at the same declared result assertion. The defence's conformance comparison therefore accepts them. Final result: **accepted absence of a pick**, `ok=true`, `calls_observed=1` (the alert), no reasons/gaps/errors. Raw mutant events show exactly one `boundary` with category `alert` and no `pick`.

This is both a false certificate for the observed execution and an observer-created behavior difference that the conformance control misses. The test's observation-aware side assertion is disclosed; it supplies the independent witness rather than pretending identical test states prove identical boundary effects. The simpler callback-count control independently disproves observer purity without such a branch.

**Smallest fix:** establish that observation cannot invoke application audit callbacks, or refuse that runtime shape before issuing certificates. Coverage must include hooks installed before the call phase as well as during it. Use non-callback native primitives or a trusted startup hook-installation census that can fail closed; do not rely on the current GC/census APIs plus equal failure type/location. Add the direct purity control and the complete-defence probe. Missing events produced inside observation must never establish absence.

### 5. SHOULD — the final thread census can still make `stop()` raise

**Locations:** `scripts/audit/incident_register/observer.py:725`; plan `:315`, `:324`.

`stop_audit.py` installs an application audit hook that raises `RuntimeError` on the final `sys._current_frames` call. Measured result: **`stop()` raises**, monitoring id and both watcher fields are already cleared, and events contain `obs_start` but no `obs_end`. O37 tests the guarded resolution path, not this unguarded census.

The acquired-resource leak is repaired, and this probe does not yield an accepted certificate. It contradicts the additional “stop never raises” claim and can interrupt pytest's call-phase reporting.

**Smallest fix:** guard the census, record an observer error and unavailable completion/census state, and preserve refusal. Add the independent failure-path control. Do not substitute zero counts after a census failure. Qualify the plan's universal claim until this path is handled.

## False greens

The five final independent probes all reach the full `current_defence` runner's **accepted** verdict. Four falsely report zero boundary calls despite an independently asserted send. The audit-hook probe falsely reports no pick despite an independently asserted pick and one recorded alert. Every acceptance has an observed positive control, unchanged frozen tests/helpers, the declared mutant failure, restored green execution and no recorded closure/conformance reason. They are not hand-authored certificate event lists.

The green submitted suite and sensitive mutants exercise their particular shapes. O31/O32 catch a Python override while the changed type is visible at a Python start; they do not cover C lookup between starts. O21 covers a later mock-class store; it does not verify an override already installed. O33 correctly refuses cloned code, but its new scan invokes audit hooks. Those controls can all pass together with the false certificates above.

I inspected the retained 149-row sweep and independently exercised all 16 additions against their intended nodes. I did not claim a fresh strict whole-suite sweep. All targeted mutant failures were ordinary test assertions; source bytes were restored after each. The current list has 148 entries because O4 is retired. Removing O4, O22 or the end guard separately leaves the 17 selected controls green. This corroborates their exercised local equivalences; it does not prove global recorder completeness. In particular, O22 changes the read of **`__dict__`**, not resolution of `send`: equivalence of those dictionary-access expressions does not establish equivalence of raw dictionary lookup and actual attribute lookup.

The E77/L03 sensitivity results remain credible for the stated component fixture scope. P1–P7 each fail ordinary controls; F1/F2 each fail the E77 positive, L03 positive and L03 incident predicate while E77's known bad path remains XFAIL; both repair sketches reach strict XPASS with controls passing. No new fixture-predicate vacuity was found. Their pair accepts value matches/exception shapes under recorded reviews, not causal dataflow or historical occurrence.

## Rulings

**Ruling 2: implementation claim PARTIAL.** The sys-root watch, exact function types and original low-level-thread/clone/startup corrections are supported. “Exactly standard dispatch” and the implication that exact types make raw namespaces describe actual attribute lookup are false as used here. Per-start checks prove sampled types, not interval-wide dispatch stability. The trusted observational guarantee also remains unfulfilled with application audit hooks.

**Rulings 8 and 9: acceptable acceptance contract; currently unmet.** Ruling 8 now explicitly makes newly found false certificates blockers. It therefore requires BLOCK for findings 1–4. Ruling 9 authorizes continued iteration, not acceptance, repair/merge/deploy operations by this reviewer, or a wider historical reading.

**Self-review: partly supported.** The failed-start assertions and cleanup are sensitive, including safe cleanup after O38 fails. The non-production second-start control closes its stated test gap. The old persistent-root example remains refused with the end guard removed. The retired O4/O22 targeted experiments pass. The stronger dispatch and “stop never raises” conclusions are not supported by those measurements.

Minimum wording corrections, applicable verbatim alongside the required code corrections:

> Exact ModuleType and exact-type classes are eligible only when actual attribute resolution is covered. Exact type alone does not rule out module fallback lookup, class inheritance or descriptor dispatch. Sampling owners at Python function starts does not establish that no transient dispatch change occurred between them. Unsupported or unobserved dispatch, an unresolved binding suffix, an unverified mock call implementation, or application callbacks invoked by observation make coverage unavailable.

> Retired O22 changes only how the __dict__ attribute is read. Its local equivalence does not establish that reading a raw namespace reproduces lookup of the declared boundary attribute.

> Failed-start resource cleanup and the guarded end-resolution path are verified. The final thread census can still raise through an application audit hook; the universal stop-never-raises claim remains unproved until that path is guarded.

These wording changes alone cannot produce SIGN WITH EDITS: the measured false certificates must first be refused. No new rule was found that lets a repair report establish an observed incident beyond its qualified claims. The prior per-citation/witness-role rules and value-match qualification remain appropriate. The affected overstatement is the executable witness's coverage, not a new authorization to promote draft records.

## Answers

- **Sys root:** the retained temporary and persistent replacements are covered; removing the end guard does not reopen the original r6 example. No new root-specific bypass was measured.
- **What changes dispatch between starts:** the exact module's fallback, exact class's inherited attribute, and C override/type-swap probes are concrete answers. Keeping all Python starts does not observe operations that create none.
- **Code sharing and cost:** the original clone is correctly unattributed. In an isolated process without application audit hooks, seven batches of 20 scans measured median **0.349 ms/scan** with no added heap, **0.394 ms** with 10,000 retained tracked lists and **0.710 ms** with 100,000. These are bounded local measurements, not BTS workload timings. The global-heap scan occurs per matched boundary start; the callback side effect is the blocker, while cost merits explicit workload accounting rather than a claim of negligible overhead.
- **Exact callables:** the forged FunctionType name executes no property now. Mock subclasses remain an unverified exception to the otherwise exact recognition rule.
- **Thread census:** raw/pre-existing workers are counted in the submitted controls. The API itself invokes audit callbacks and can raise; census completeness and observer purity are separate obligations.
- **Lifecycle/cache/deploy:** acquired-resource cleanup, reviewed-hook purge/refusal, non-adjacent cross-run overlap refusal and within-observation precision behavior pass their direct controls and mutants. The unguarded final census is a separate lifecycle defect.
- **Mutants and vacuity:** 148 kills plus the disclosed O4 survivor are the retained 149-run result, not 149 kills. All 16 additions are independently sensitive. Local retirement tests pass, but do not cover the new counterexamples. E77/L03 controls remain sensitive and reachable.
- **Publication:** the 22/40 synthetic receipt and current draft wording do not establish the published build's pin or fixture bindings. Task 8 must execute its stated publication check.

Review boundary: no `data/` reads, SSH, network, `gh`, main tracked-file edits or main commits. Synthetic commits/worktrees were confined to reviewer-owned temporary repositories as authorized. Main remains at f6d881f. The report is the sole main-checkout output. The authorized wt7 checkout was removed with the required git worktree remove --force command; its directory and Git registration are absent. Main HEAD and tracked status are unchanged. Report headings, measured XML totals and final sentinel were mechanically verified.

DONE
