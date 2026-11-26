## Verdict

**BLOCK.** The five original r8 executions now behave as requested, and the wording changes are in the right places. However, three new full runs accept positive certificates while the observer executes application-defined equality inside observation. The purity requirement remains unenforced for class-namespace and keyword lookups. These are ordinary pinned Python classes and a standard Mock, inside the stated model; they use none of its excluded mechanisms.

Reviewed plan/design at `f28a0766c2b61adb060e13692965bc15ffff65ce`, code `ee2489db514efa466ad9343a9d2ed55434464b4f`, and evidence `cdd24216be68de3a2e012c15003b2bc3be7bafcc`. Independently checked that `cdd2421` changes no `src`, `scripts` or `tests` bytes relative to `ee2489d`. All source line references below are at `ee2489d`.

Evidence retained under `r8-probes/r9/`, relative to this report. The required `wt9` was removed with `git worktree remove --force`; its path and registration are absent. Main remains at the reviewed pin with clean tracked status. No production/data access, network activity, actual BTS commits, or tracked-file changes were made. Test tooling created synthetic fixture repositories only.

## r8 findings status

I reran the five original full-run probes unchanged. Their defect-asserting wrappers produce **four expected failures and one pass**: four no longer obtain the acceptance they asserted. Raw results are in `original-r8-runs/`; `original-summary.json` records the five distinct results.

| r8 finding | Status | Measured rerun and disposition |
|---|---|---|
| #1 Clear leaves a former callable current | **RESOLVED** | Rejected. The captured call is recorded as `unavailable`, with the held-earlier/current-value attribution reason. The alert cannot witness. Additional module, class and `sys.modules` clear/clone probes reject; per-key restoration controls accept. |
| #2 Incomplete or unattributed records witness | **RESOLVED** | The 501-character return is rejected although its requested serialized value exactly matches the recorded incomplete form. The request for category `unavailable` is refused by spec validation before any stage runs. Consumer tests also reject a nonreserved category carrying an attribution reason and nested incomplete forms. |
| #3 Application code runs during observation | **PARTIAL** | The original metadata formatter runs **zero times in both twins**, its return reads unavailable, and the certificate is rejected. New class-key and keyword-key equality paths still run application code in accepted observations; findings 1–2 below. |
| #4 Clone accepted as a declared return | **RESOLVED** | Still accepted, with return values `['defer', 'done']`. This now characterizes source-file/qualname identity. The new qualification explicitly withholds function-object, globals and receiver identity. |
| #5 False historical justification | **RESOLVED** | The replacement sentence is verbatim in plan ruling 10, design §9.3 and the certify docstring. Positive attribution, purity and closure remain independent obligations. |

The supplied `test_r8_counterexamples.py` contributes **18 passing nodes** to the independently measured **309-passed** review suite. Its per-key control restores the actual namespace key by key before calling the sender, so acceptance is a valid positive control. Its clone test measures both returns and checks the coverage qualification; acceptance does not establish exact callable identity. The complete nested-return control also passes. The supplied tests cover their original counterexamples, rather than establishing observational purity for every namespace access.

## New findings

1. **BLOCKER — C descriptors and plain-instance lookup still compare application namespace keys.** `scripts/audit/incident_register/observer.py:137`, `:163`, `:164`, `:168`, `:212`; relevant promises: `certify.py:13` and design §9.3 at line 126.

   A class constructed normally with `type('Result', (), {Key('extra'): 1})` can retain a `str` subclass key. Its hash can collide with `__dict__` or `__module__`; its `__eq__` is application code. Class metadata itself remains ordinary exact strings, and the returned instance has a standard C dictionary descriptor and a plain `status` field.

   The `__dict__` membership/read and `object.__getattribute__` path execute equality **three times**. More directly relevant to the new code, `_TYPE_MODULE.__get__` executes equality **once**, before the exact-string guard. A C descriptor bypassing a metaclass does not guarantee that its internal dictionary lookup avoids application equality.

   Two full `current_defence` runs mutate only `status='done'` to `status='defer'`. Both return **accepted**, with `ok: true` and linked entry/branch/return sequences **2/3/4**. Both serialize the complete value `{'fields': {'status': 'defer'}, 'type': 'bts.mod.Result'}` and classify it `bad`. Observed/plain application-comparison counts are **3/0** for the dictionary path and **1/0** for the module descriptor. Both endpoints report `{audit_hooks_added: 0, gc_enabled: false, signal_handlers: []}`; neither run records an observer error. The assertion failure remains at the declared line in both twins, so conformance accepts the observer's additional application side effects.

   Reproduce with `test_r9_review.py:36`, parameters `__dict__` and `__module__`. Accepted artifacts: `verified-new-runs/test_full_run_class_namespace_0/out/acceptance.json` and `test_full_run_class_namespace_1/out/acceptance.json`; raw events and both stdout files accompany them. `verified-summary.json` retains the certificates and counts. The direct probe at `test_r9_review.py:78` additionally measures one application comparison each from `type_name`, `_raised`, and `_err` for an exception class with the same module-key collision. The same `type_name` path supplies pair exception-module identity at `observer.py:284`.

   **Smallest fix:** eliminate hashed lookups on application class namespaces. Before any metadata descriptor that internally indexes the namespace, inspect its raw items without application dispatch and refuse unsafe key shapes, or obtain the supported metadata through exact-string item iteration. Find the standard instance-dictionary descriptor by that same iteration and invoke the verified C descriptor directly; avoid the generic attribute lookup. Preserve unavailable/incomplete handling and add the two full-run equality probes plus exception/error-record controls. This requires a code fix to enforce the existing purity condition.

2. **BLOCKER — Keyword identity extraction executes application equality inside a verified standard Mock call.** `scripts/audit/incident_register/observer.py:350`–`:351`, called by `_identity` at `:528`.

   Python keyword dictionaries can retain `str` subclass keys. A standard Mock called with `**{Key('text'): payload}` passes such a dictionary to the observer. The membership check and subscription for `kw:text` execute the key's `__eq__`; checking the payload's exact type afterward is too late.

   The independent full run uses an unchanged standard Mock with a normal delivery side effect and a `kw:text` boundary accessor. It is **accepted**, with linked entry/branch/event sequences **2/3/4** and category `alert` for `BTS health CRITICAL: x`. Application equality runs **three times observed versus once plain**: the two extra calls come from observation. Both purity endpoints are clean and no observer error is recorded. No standard-library function, observer object, evidence, mock class or interpreter behavior is replaced.

   Reproduce with `test_r9_review.py:95`; artifact `verified-new-runs/test_full_run_keyword_accessor0/out/acceptance.json`, its raw events and twin stdout files. The direct nonmatching-key probe at line 120 also shows that merely searching for a missing keyword invokes application equality.

   **Smallest fix:** read keyword identities using exact-string item iteration, as `_lookup` already does, instead of membership/subscription on the application's dictionary. Unsupported key identities must fail unavailable without invoking their equality. Add a full-run standard-Mock control for exact-string keyword keys and the subtype-key refusal/equality control.

3. **SHOULD — Session import records still use application attribute dispatch and hashed namespace lookup.** `scripts/audit/incident_register/observer.py:328`–`:330`.

   Direct session-finish probes measure an application `__getattribute__('__dict__')` call for a ModuleType subclass and an application `__eq__('__file__')` call for a colliding key in an **exact** ModuleType's namespace. The latter needs no overridden module dispatch. Reproduce with `test_r9_review.py:133` and `:149`. These measurements concern session records after the call interval; I have not demonstrated a false import-provenance acceptance from them, so they are not an additional basis for BLOCK.

   **Smallest fix:** require exact-string module names before string operations; read supported module namespaces through the existing C descriptor path; locate `__file__` by exact-string item iteration. An unsupported namespace should produce an explicit refusal/gap rather than be queried through application dispatch.

## False greens

The newly accepted certificates in findings 1–2 violate the advertised observational condition. Their bad return/event is actually observed; the false assurance is that obtaining that witness ran no application callback. Clean audit/collector/signal endpoints and matching assertion failures do not establish that stronger condition. The twins' captured stdout independently exposes the extra callbacks, but the conformance gate does not reject them.

The 309-passed suite and submitted 173-killed sweep coexist with those accepted impure certificates. The r8 metadata test exercises an application metadata **value** formatter; it does not exercise application **keys** used while obtaining that value. The keyword accessor has a separate unguarded dictionary path. The static closure screen also does not test either mechanism.

Independent checks, with their limits:

- Review suite: **309 passed**, with the required offline `jsonschema==4.23.0` dependency; `review.xml` and `review.log`.
- New review probes: **16 passed**, including seven clear/clone/restoration executions and the defect-asserting callback probes; `verified-new.xml` and `verified-new.log`. These passes confirm the measured unsafe behavior where explicitly asserted.
- Targeted mutation rerun: **all 12 new mutants killed**, plus re-anchored **O8** and **O16/O24/O25**, from **12 passing targeted baseline nodes**. Every run has its expected inventory and assertion-shaped kill; `mutants/results.json` and per-run XML/logs. This is a 16-mutant targeted check, not a repeat of the full 173-mutant sweep. The submitted sweep contains 173 unique killed-result labels. The observer-level killers remain active where the consumer completeness guard masks an older end-to-end killer.
- Fresh byte-equivalent synthetic copy: **564 pinned input files**, code at `ee2489d`; pair accepted **22/22**, **20 value_match**, **2 exception_shape**, **40 passed_nodes**, **88 clean purity records**. I-0830-d's return-kind pilot also accepted. `frozen3/equivalence.json`, `frozen3/pair/expected_failures_acceptance.json`, `frozen3/pilot/acceptance.json`, and `fresh-summary.json`. The synthetic evidence worktree was destroyed. Two preceding reviewer launch attempts failed before evidence execution because of the interpreter/path setup; `frozen3` is the completed rerun.
- Predicate: all **11** cases behave as required—seven weakened predicates caught, both non-pick formatters caught with the required failures, and both repair sketches reach strict XPASS with their controls passing; `predicate.log`.
- Closure screen: fresh body agrees with the submitted body, with the same two benign canonicalizer module registrations; `closure.txt`. The full fast suite and the entire 173-mutant sweep were not independently rerun in this review.

## Rulings

Ruling 10 still applies. Missing-event links remain unavailable; none of these findings asks for absence certification or treats a missed call as an error.

The ordinary class constructor, namespace keys, keyword dictionary and standard Mock in findings 1–2 are inside the declared model. They add no audit hook, toggle no collector/signal handler, replace no observer-called library function, and perform no evidence or interpreter tampering. They violate the retained no-application-callback condition, which is an independent obligation expressly preserved by the r8 #5 replacement. Excluding them retroactively would narrow supported value/identity shapes and would need a stated rule and closure disposition; it is not enforcement of the current claim.

The source-identity clone remains a qualified acceptance. Exact callable/globals/receiver identity requires the additional witness or unavailable disposition now stated in coverage. A fixture review must establish the contract and selection for that source invocation; it cannot promote the machine's evidence to function-object identity.

The absence refusal, reserved-category checks and recursive completeness gates stand. I found no new accepted incomplete or reason-bearing witness in the executions examined. Session-record dispatch is a separately measured hardening gap, with the limited disposition in finding 3.

## Answers

1. **r8 disposition:** #1, #2, #4 and #5 resolved; #3 partial. All five original full runs were rerun. The supplied clear/per-key and clone tests correctly capture their intended distinctions.
2. **Changed-code attacks:** chain cutting works at module, class and `sys.modules` levels. Cloning saved contents into the empty namespace remains unattributed and yields the end-check gap; per-key restoration makes the legitimate binding current again. A store in the old child after its parent is cleared does not reconnect it. The witness/reserved-category gates reject the tested partial/unattributed shapes. Type metadata and keyword extraction still execute application equality; `_raised`, `_err` and pair exception identity share the metadata path. Session recording has its own remaining application-dispatch paths.
3. **Plan/design placement:** both r8 sentences are verbatim in the operative Phase 1 boundary paragraph at design §9.3, line 132. The history correction also appears in plan ruling 10, line 90, and certify's opening explanation. The identity qualification travels in `COVERAGE['return_identity']`, so accepted certificates carry it. These placements do their job. The written rules preserve separate machine acceptance and recorded review and require additional identity evidence where necessary; they do not authorize a fixture review to invent a witness. The remaining overstatement is implemented purity, demonstrated above.
4. **Verdict:** BLOCK until the two accepted in-model application-equality paths are removed or refused safely, with full-run regression controls. No absence redesign is needed for these fixes.

DONE
