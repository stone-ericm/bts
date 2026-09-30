## Verdict

**BLOCK.** The specific r4 counterexamples mostly received effective repairs, and I reproduced the submitted expected-failure pair. V5 still accepts false absence, classifies incomplete observations, runs application attribute code while resolving boundaries, misses executable closure changes, and mistakes named status messages for pick delivery. These require code/fixture changes before Tasks 5–6 can produce trustworthy certificates.

Reviewed plan rev 5 at **a0f3ebfc53be7806a250bf0518585d57b01fc5e9** and code/evidence in the authorized detached checkout at **274ceeb**. Verified that the tooling, incident fixture, meta-tests, registry and tooling tests are unchanged between **796046b** and **274ceeb**; tooling/fixture/registry bytes are also unchanged between **aaaeb1f** and **796046b**. Source references below are to 274ceeb; plan references are to a0f3ebf.

Measured offline with Python 3.12.13, pytest 9.0.2, jsonschema 4.23.0 and TZ=America/New_York:

| Check | Result |
|---|---|
| Submitted tooling, incident fixtures and meta-tests, after restoration | **250 passed, 22 xfailed**; tooling alone accounts for 218 passes |
| Adapted r4 counterexamples | **17 passed**, included above; additional record, deploy, classifier and fixture regressions also passed |
| Independent reviewer probes, after restoration | **19 passed**; many deliberately assert false acceptance, so these are defect witnesses |
| Actual pair driver on a synthetic repository containing the pinned production source, fixtures, scheduler helper, configuration, lock, scripts and registry, with its own venv | **accepted 22/22**, 20 `value_match`, 2 `exception_shape`, 29 `passed_nodes`, no reasons |
| Seventeen targeted submitted sweep mutants | Each failed its intended regression; the 16 distinct unmutated regression nodes passed |
| Both completion-guard-only mutations applied together | **40 defence/replay tests passed**; these remain redundant for the exercised paths |
| Draft records | **119**, zero draft-mode errors; 61 observed (30 A, 27 B, 4 pending), 28 latent, 4 near misses, 24 unresolved, 2 pre-ship |
| Retained deploy corpus | 192 runs: 164 expired, 28 retained; 48 observation points, zero clean rollbacks |

I did not rerun the full fast regression or all 111 whole-suite mutations. I inspected the sweep, its tests and retained output, and independently exercised the 17 targeted mutations. The actual pair above uses a synthetic commit, **d595534a6d672799dccbe0ff61b79f55d310cd5d**, whose copied inputs come from the pinned checkout; it is not a claimed publication-commit run.

Reproducers, mutation scripts, individual mutant logs, final JUnit files, real-pair events/acceptance, and context summaries are retained at **/private/tmp/bts-incident-register-r5-dRcbMo/**. `test_review_r5.py` contains the independent probes; `run_fixture_mutants.py`, `run_targeted.py`, `prepare_real.py` and `audit_context.py` preserve the other procedures. An attempted combined collection of repository tests and the external probe file failed before tests when pytest walked an unrelated `/private/tmp` ancestor. The separately scoped final runs reported above completed; that collection failure is not counted as a product finding.

## r4 findings status

“RESOLVED” below disposes the original counterexample, not every possible failure in the same subsystem.

| r4 item | Status | Measured rerun and limit |
|---|---|---|
| **#1 Certificate event set** | **PARTIAL** | All four original cases now reject appropriately: initial C gap inside the interval, production-side temporary C rebinding, pre-existing worker, wrong bound receiver. Their adapted regressions passed, and O10/O11/O12 failed when their protections were disabled. A frozen external helper and a binding reusing another boundary's callable still produce accepted false absence: new finding 1. |
| **#2 Serializer callbacks, truncated equality, conformance** | **PARTIAL** | Collision-key serialization makes zero equality calls; the 51-element return no longer matches the wrong 50-element oracle; long strings/maps are incomplete; differing observed/unobserved exception/frame failures reject. Top-level O15/O16 regressions are sensitive. Nested incompleteness still classifies and certifies, and the boundary resolver still runs application code: findings 2–3. |
| **#3 Unrelated readback called connected** | **RESOLVED under amended ruling 7** | The unrelated-readback pair without a review rejects as `unmatched`; removing the review gate fails its regression. Zip cardinality mismatch rejects, and A13 is killed. An adequate recorded review is now explicitly responsible for invocation/selection dataflow; the machine claim is only `value_match`. |
| **#4 Executable environment and untracked contents** | **PARTIAL** | External file symlink, relative path-line `.pth`, existing dependency bytecode, ordinary untracked content, and replay content-drift probes pass corrected expectations. W9 and D13 are independently killed for their named reasons. Executable `.pth` additions and symlinked harness targets remain outside the freeze: finding 4. |
| **#5 Reports promote occurrences / borrow another claim's timeliness** | **RESOLVED for the three original cases** | Repair-only promotion rejects; an April occurrence and April mitigation cannot borrow July citations; each regression checks its specific error using a valid fixture evidence root. Supported observed times, open-ended refusal, witness roles and report-at-own-instant tests pass. Ruling 6 still lacks coherence against an actually ended condition: finding 6. |
| **#6 Missing expected-failure certificate / impossible latency** | **RESOLVED for the original cases** | Missing pair, wrong hash, wrong registry, absent evidence root, and detection before onset reject. Characterization binding, exact `module.qualname` and control membership regressions pass; V14/V16/V17 and A14/A15/A16 are independently killed. Other certainly impossible chronology is accepted: finding 6. |
| **#7 E77 non-delivery message** | **PARTIAL** | The original formatter returning `No pick today` now fails the positive execution control ordinarily. The five submitted negative controls pass. A named status formatter returns **2 passes, 2 XFAILs** across E77/L03 incident and positive-control nodes: finding 5. |
| **#8 L03 valid early delivery refused** | **RESOLVED for that repair direction** | The 18:15 send with its consistent durable record passes; three inconsistent-history controls fail ordinarily. A production delivery-chokepoint postponement guard yields **one strict XPASS and five passing L03 controls**. The shared named-send predicate still lets unrelated status traffic raise the dedicated exception: finding 5. |
| **#9 Deploy order / rollback SHA** | **RESOLVED for the original cases** | Observation-time ordering, cross-run equal-time disagreement refusal, and actual logged rollback SHA/mismatch regressions pass. The retained corpus still has 48 points and no rollbacks. Fractional same-run timestamps collapse to an empty installation interval: finding 7. |
| **#10 Sweep string classifier** | **PARTIAL** | All nine submitted classifier tests pass: an ordinary `RuntimeError("DID NOT RAISE")`, ordinary look-alike `AssertionError` class, and KeyError are refused; genuine assert/explicit builtin assertion/pytest.raises refusal classify as kills; phase errors and short runs reject. My independent ordinary look-alike class is also refused. Skips are nevertheless accepted as a clean baseline: finding 8. |
| **Task 7: I-063/I-074/I-095 reattachment** | **RESOLVED for the requested edits** | Each has explicit `observed` claims; onset inference/continuity limits are retained. All 119 drafts validate. These are structural and cited-report checks, not new box measurements. |
| **Task 7: I-075/I-084 dating** | **RESOLVED** | Previously open-ended report citations gain latest bounds; I-084's 8/27 recurrence uses later-analysis evidence `ev7`. |
| **Task 7: I-103/I-104, I-116/I-117, I-076 exclusion** | **RESOLVED** | Both retired-host mechanisms are reinstated; WARN routing and threshold noise are separate latent records; I-076 excludes the model-quality question and explicitly lists the separately retained operational pieces. No unsupported live firing is added. |
| **Task 7: I-113 strength** | **RESOLVED for the requested qualification** | The added text distinguishes reported Fly fallback from code-verified R2/bootstrap mechanism and disclaims exact onset, contest-pick impact and verified repair. |

## New findings

### 1. BLOCKER — binding discovery still permits accepted false absence

**Locations:** `scripts/audit/incident_register/observer.py:376,389,451,476,519`; `certify.py:109–130`.

Two independent complete `current_defence` runs return **accepted**, with no reasons and an `ok=true` absence certificate reporting zero declared-boundary calls, despite the frozen test proving exactly one real send:

1. **Rebinding inside a frozen external helper.** Production calls `tests.helper.invoke(transport, rebind)`. That helper optionally replaces `transport.send`, invokes it through `list(map(...))`, then restores the original binding. The source mutant changes only the production argument and return (`False/done` to `True/defer`). The unchanged test verifies `transport.sent == [('eric', 'pick: Turner')]` before failing its declared result assertion. `_on_call` scans bindings only at production CALL events. The replacement's start is already past when its own first production CALL discovers the binding; registration is too late to record this invocation. The restored final binding hides the gap. Observer-off conformance agrees.
2. **A callable already registered for another boundary.** Two declared bindings initially have separate Python functions. The production mutant temporarily assigns boundary B's binding to boundary A's function, calls B, restores it, and returns `defer`. `seen_targets` is global by object identity, so A's existing registration prevents B's registration. The one send is recorded under A's name; B's absence is accepted. Moreover `_on_start` emits only `matched[0]` when multiple specs share a code/receiver.

Artifacts under the scratch root: `final-probes-confirmed/test_rebinding_in_a_frozen_ext0/external-rebinding.json` and `final-probes-confirmed/test_a_callable_seen_for_anoth0/alias-result.json`, with their corresponding `out/` directories.

**Smallest fix:** discover current bindings before relevant callee execution across the supported callback/helper closure, including calls originating outside production; fail closed on shapes whose rebinding cannot be covered. Register per **boundary spec plus target/receiver**, and record every applicable boundary identity or mark ambiguity unavailable. Add both complete defence probes, with a frozen assertion that the actual send occurred. Scanning only production CALL sites cannot establish the advertised full event set.

### 2. BLOCKER — nested incomplete values remain classifiable certificates

**Locations:** `observer.py:130,150,159,357,543`; `acceptance.py:61–70`.

`_safe({'tag':'pick', 'payload':[0]*51})` marks its nested sequence incomplete but leaves the outer map unmarked. `_identity` checks only the outer `incomplete` flag and classifies the value as **pick**. `acceptance.unwrap` correctly rejects the same observation as `UNAVAILABLE`.

This is also exercised through the actual defence runner: production returns `{'tag':'bad','payload':[0]*50}`; a source mutant changes 50 to 51; the frozen test's length assertion kills it. With a declared return classification for `bad`, the runner returns **accepted** and `certificate.ok=true` over the incompletely serialized return. This contradicts both the plan's incomplete-value rule and the certificate's stated coverage limit. The submitted top-level long-value regression cannot see this case.

**Smallest fix:** propagate incompleteness/unavailability through every container, or perform one recursive completeness check before **both** identity and return classification. Keep a separately selected field's completeness explicit where `_field` intentionally extracts it; do not certify the complete enclosing value from a complete child. Add nested sequence, nested unsupported-object and depth-limit controls at both classification sites.

### 3. BLOCKER — resolving a nested binding still runs application code

**Locations:** `observer.py:282–290,414,439,526`.

`_resolve` says it reads namespaces “no getattr hooks,” but executes `getattr(obj, '__dict__', None)` at every nested component. An independently constructed holder overrides `__getattribute__`; resolving `review_holder:holder.send` invokes that application method once for `__dict__`. The probe obtains its expected target through `object.__getattribute__`, so its assertion itself does not cause the recorded call.

This is separate from the fixed collision-key serializer bug. Registration, every production CALL scan, and final resolution can now run application callbacks. I measured the direct callback, not an accepted full defence certificate attributable to this particular side effect. Conformance on exception type/frame does not establish that arbitrary application effects are absent.

**Smallest fix:** use safe raw module/instance namespace access with the same descriptor restrictions as `_plain_instance_dict`, without application `getattr`/properties. Resolve only supported namespace shapes; otherwise record an unavailable binding/coverage gap. Pin overridden instance and module attribute access as negative controls.

### 4. BLOCKER — the execution closure can change while the pair remains accepted

**Locations:** `owned.py:158–174,247–257`; `acceptance.py:241–263`; plan Task 3.

Two measured residuals:

* An executable `.pth` line `import sys; sys.path.insert(0, '<external>')` makes the worktree's actual Python `-B` import `local_dep`. Changing its external source changes the executed value **1 → 2**, while `venv_fingerprint` stays identical. Hashing the `.pth` spelling does not hash the code its import hook exposes. Lines beginning `import ` or `import\t` are simply skipped by closure discovery.
* `_file_state` hashes a tracked or untracked symlink's spelling, not its target bytes. In a complete synthetic `run_pair`, `tests/helper.py` is an untracked symlink to an external source. The frozen marked-run test imports VALUE=1 and rewrites that source to VALUE=2; the unmarked run imports VALUE=2 and verifies it. The symlink remains identical, neither imported-helper source is covered by the venv fingerprint, and the pair returns **accepted** with no drift reasons. `symlink-pair.json` and the marked/unmarked events are retained under the final probe root.

The ordinary dependency symlink and path-line `.pth` repairs are real; their scope does not cover these two alternatives. I found no normal same-size/same-mtime write escaping the **ctime-keyed memo** on this host, and independently killed W9. These are closure-discovery omissions, not a demonstrated ctime-memo bypass.

**Smallest fix:** hash resolved target contents for harness/input symlinks with cycle handling, or refuse them as unbound closure inputs. Account for executable `.pth` import hooks and their external roots/imported modules, or explicitly refuse unsupported hooks before issuing an immutable-closure verdict. Correct `venv_fingerprint`'s stale “bytecode caches excluded” docstring; the implementation now includes dependency bytecode.

### 5. BLOCKER — the identified-send predicate is still only a batter-name match

**Locations:** `tests/test_incident_register_2026.py:592–605,710–722,937–968`; registry E77/L03 recorded reviews.

The five E77 negative controls strengthen record/recipient correlation, but the text test is still “non-alert and contains the batter string.” Independently measured E77 required-branch acceptances include:

* `No pick today for Trea Turner`;
* `Do not enter Trea Turner`;
* `pick: Trea Turner, game 999 on 2026-07-15` with the correct saved game's flags;
* a matching send and record from the **previous day**;
* a send at 18:04 with persisted `delivered_at` **18:05, exactly the cutoff**, accepted by the ±60-second record tolerance.

The actual source formatter mutant `return 'No pick today for ' + daily.pick.batter_name` leaves the four real E77/L03 incident and positive-execution-control nodes at **2 passed, 2 xfailed**, exit 0. Thus this is a production-path false green, not merely a fabricated observation dictionary. The original unqualified `No pick today` formatter now correctly fails the positive controls.

For L03, a named status message at 18:35 plus matching flags reaches **PostponedPickDelivered** in the dedicated oracle. A status message saying no pick today is not the declared postponed-pick delivery. Therefore something other than the incident can still reach its dedicated exception.

**Smallest fix:** independently specify and validate affirmative pick-recommendation content for the declared selection/game/day; a name mention is not delivery evidence. Bind its receipt and durable record, and require **both timestamps** to be within the intended day's interval and strictly before the cutoff. Do not construct the oracle's expected message by calling the same production formatter being tested. Add named status/negative instruction, wrong game/date, previous-day and persisted-at-cutoff controls to E77 and the shared L03 predicate, then renew their recorded fixture reviews.

### 6. SHOULD — ruling 6 and the chronology claim still admit a condition observed after its restoration

**Locations:** `records.py:118–142,145–168,330`; plan ruling 6 and Task 8.

The replacement text is verbatim, and its main structural protections work. However, chronology checks cover only onset→detection, detection→confirmed notification, and onset→restoration. Two schema-valid publication probes still return **zero errors** with valid expected-failure evidence roots:

* one continuing occurrence restored at July 16 12:00, then purportedly **observed still deviating at July 17 09:00**;
* one alert **confirmed at 18:00 but first attempted at 19:00**.

The report qualifies against each timestamp separately; no coherence rule notices that the same condition has already ended or that confirmation precedes its attempt. This does not undermine the repaired original negative-latency probe. It limits the broader claims “impossible chronology is rejected” and “use the earliest supported event that actually ended the cited condition.”

**Smallest fix:** validate certainly reversed, contract-ordered endpoint relationships, including supported observed times against the occurrence's actual restoration and alert attempt against its corresponding confirmation/failure. Split recurrence into a distinct occurrence rather than witnessing it after this occurrence's end. Preserve unknown/overlapping bounds. Do not reinstate code-install-as-restoration: the relationship still needs evidence. Otherwise narrow the plan's chronology claim explicitly to the three implemented pairs.

### 7. SHOULD — fractional deployment observations collapse to an impossible `(t, t]`

**Locations:** `deploy_runs.py:27,54,166–171,240–246`.

The log regex accepts fractional seconds but `_hits` discards them. Synthetic retained output has pre=A at `00:00:00.100Z`, deployed=B at `00:00:00.900Z`, in the same run. Extraction produces both times as `00:00:00Z`; the same-run tie is allowed, and `first_live` returns **not_live_before == live_by**, an empty `(t, t]` interval. Log order orders events; it does not turn the whole second into an exact installation instant.

The original cross-run equal-time disagreement probe now rejects correctly. The 48 retained points contain no rollback case, and this new synthetic example does not disprove their current reported bounds.

**Smallest fix:** preserve fractional timestamps and compare parsed instants. When source precision genuinely cannot separate disagreeing observations, widen to that precision's supported bound or refuse the interval; never emit an empty first-live interval. Add same-run same-second and fractional-second cases.

### 8. SHOULD — the sweep calls a skipped suite a clean baseline

**Locations:** `docs/audit/2026-09-29-incident-register-evidence/tooling/mutation_sweep.py:159–179,210–218`; `tests/scripts/incident_register/test_sweep_classifier.py:31–48`.

A real pytest run with one skipped node exits 0 and supplies one JUnit testcase. `classify(xml, 0, expected_cases=1)` returns **SURVIVED**. `main` consequently accepts it as a clean baseline. Equal testcase count also permits some baseline nodes to become skipped during a mutant while an assertion in another node yields KILLED. The classifier does not compare the actual baseline inventory or prohibit skips/XFAIL-shaped skipped cases.

This is not a claim that the retained 218-pass baselines skipped tests: they did not, and I reproduced 218 tooling passes. It is a reusable gate gap absent from the nine classifier tests and the 111-mutant matrix.

**Smallest fix:** require the baseline's exact node inventory and all-pass outcomes; reject any mutant skip/XFAIL/XPASS or inventory change before deciding KILLED. Add real-pytest all-skipped baseline and assertion-plus-skipped-mutant tests. Keep the now-effective RuntimeError/look-alike and phase-error regressions. For stronger exception/location claims, retain structured in-process outcomes; JUnit message shape alone is the deliberately weaker current implementation.

## False greens

| Green claim | What was actually established |
|---|---|
| Completed invocation, zero boundary calls, accepted absence | One real send can occur inside a frozen helper or through a callable shared with another boundary. |
| Incomplete values classify unavailable | Only top-level incompleteness is checked at classification; an incomplete nested return receives an accepted certificate. |
| Namespace lookup runs no application code | Nested `_resolve` invokes application `__getattribute__`. |
| One frozen pair closure | A symlinked helper changes and is imported differently across an accepted pair; executable `.pth` additions evade the dependency hash. |
| E77 identified delivery / L03 dedicated incident exception | Named status text satisfies both predicates; the actual formatter mutant preserves all four nodes' pass/XFAIL shapes. |
| Zero record errors implies coherent chronology | A continuing deviation may be observed after this occurrence's restoration, and an alert confirmed before its attempt. |
| First-live bounds come from observation times | Fractional observations are truncated into an empty interval. |
| 111/111 killed means all v5 checks are covered | It measures the listed mutations. Helper-side rebinding, per-binding callable registration, nested completeness, raw resolver access, executable `.pth`, harness symlink targets, named negative messages and skip/inventory gates are absent. |

The recorded reviewer decision remains a separate required gate. I have not accepted any of these false certificates. A human review can reject them, but cannot make their runner `accepted`, category or immutable-closure statements accurate.

The submitted self-review was useful in a specific, measured sense: W9 and D13 now fail when their own protections are disabled. V14/V16/V17 and A14/A15/A16 also have effective targeted regressions. I did not find another D13-style dummy-fingerprint false green among the 17 targeted tests; that is a bounded result, not a claim that all 218 tests are non-vacuous. Several controls above omit relevant adversarial inputs despite being sensitive to their listed mutants.

## Rulings

**Ruling 6:** accept the replacement wording and the per-citation/witness-role implementation for the original r4 claims. Reports written at a claim's own exact instant qualify; reports certainly before it reject; a latest-unbounded claim cannot anchor a report. Date-only report times deliberately allow overlap with their whole ET day, rather than claiming an exact instant. `_witnessed` requires occurrence onset, observed time or machine-detection evidence, so awareness/fix steps alone do not promote an incident. It remains a structural check: report meaning, the condition-ending relationship, and continuity assumptions need review. Finding 6 identifies a further detectable coherence gap; implementation is not complete equivalence to every sentence of the ruling.

**Ruling 7:** accept the amended vocabulary and review requirement. `return`/`reads` now publish `value_match`, not machine-proven dataflow; `derived` publishes `exception_shape`; missing reviews, incomplete whole-value equality and unequal zip cardinality reject. The registry's ordinary L01/L02/L04 reviews identify consumed return/readback fields. E77/L03's derived reviews need renewal after finding 5, since their current description overstates what “identified” means. Finding 2 concerns classification certificates; recursive `acceptance.unwrap` itself rejects that nested value correctly.

**Task 7 dispositions:** accept the requested rev 5 changes at their stated source strength, with the following limits retained for the result review:

| Record/change | Judgment |
|---|---|
| I-063 | June 11 onset remains inference; June 16–17 deviation is attached to `observed`. The June 17 spec and 2b4ff1d body explicitly describe the current wrong decision/display state. No new measurement of the earlier interval. |
| I-074 | August 9 stranded result is the report's witnessed continuing condition; the July onset uses deploy endpoints plus the stated continuity assumption. Deployment is not remediation verification. |
| I-095 | The production apply addendum reports the live pre-apply dry run and subsequent apply/idempotency checks. Its observed pre-repair state is separate from the April inferred onset and the later restoration verification. I read the memo, not its underlying box manifests. |
| I-075 / I-084 | The previously open-ended alert/action/mitigation claims are bounded by their reports; the August 27 recurrence is explicitly later analysis. These edits fix dating, not verification of every underlying operational claim. |
| I-103 / I-104 | Retaining latent saver/version-drift mechanisms is justified by 2be445e/5207a09 code/commit evidence. Unsupported historical firing stays open; retirement alone no longer excludes the classes. “Workers run latest main” should not later be upgraded to exact deployed-ref parity without evidence. |
| I-116 / I-117 | a275399's diff adds realized_calibration to WARN attention and raises drift_warn; afbea38 reports the production-clock null simulation. Separate latent routing/noise records are justified. I did not rerun that simulation or infer a live firing. |
| I-076 exclusion | Narrowed to model quality, with operational routing/noise records and new-bucket pre-ship support explicitly accounted for. |
| I-113 | 947fce8 reports missing-policy heuristic fallback on Fly; the R2/bootstrap mechanism is supported in code. Exact cold-restore execution, outage start, contest impact and verified repair remain unproved. Current sync still includes the base policy without the tail policy; that is a residual of this route. |

The counts reconcile and all 119 records pass draft validation. This is not the Task 8 publication/result review.

**Task 8 publication plan:** re-binding I-077/I-201/I-202, adding I-203/I-204 characterization entries, acceptance hashes, exact registry exception identity and passing-control membership are necessary and now structurally supported. Replace the old v3 references as planned. Add the stronger controls required by finding 5, not only the current five negative controls.

Keep “pair accepted at the published commit” as a real publication gate. `_ef_binding_errs` verifies receipt/hash/verdict/registry/node/exception/control membership; it has no publication-ref argument and does not itself compare a receipt's `resolved_ref` or source/fixture closure with the published build. Publication must explicitly retain and check that source/fixture/registry pin, or document a reviewed byte-equivalence mapping when commit identity differs. Matching node names and registry bytes alone is insufficient. The pending Task 8 work can enforce this; the already accepted 279335d pair is not automatically a run at a future publication commit.

## Answers

1. **Are the r4 fixes real?** Yes, for the specifically rerun shapes recorded in the status table. The broader observer-completeness, observational and immutable-closure claims remain false under the new executable probes.

2. **Can E77 and L03 repairs reach the required branch?** Yes. A throwaway noon schedule re-fetch makes the real E77 incident node strict XPASS with its positive execution control passing. A delivery-chokepoint postponement guard makes the real L03 node strict XPASS with all five L03 controls passing. These are sensitivity sketches, not proposed production repairs.

3. **Can unrelated behavior reach a dedicated exception or remain false green?** Yes. L03's named status message reaches PostponedPickDelivered; the actual named-status formatter mutant keeps E77/L03 at 2 passes and 2 XFAILs. A no-op run_day still gives four ordinary failures, closing the earlier no-op counterexample.

4. **Can a change escape the digest memo?** The same-size/same-mtime test and W9 mutation show that ctime contributes effective protection for ordinary writes on this host. I have no measured same-key bypass. Executable `.pth` roots and harness symlink target bytes escape closure discovery instead; one changed-helper pair is actually accepted.

5. **Does the sweep establish v5 coverage?** It establishes that all 111 listed mutants were reported killed in retained runs at 796046b. Seventeen were independently killed here at their intended regressions. It does not cover the new input/alias shapes. The known completion-guard mutant remains redundant for the 40 exercised defence/replay tests; retain the guards as defence in depth.

6. **Can a report or fixture establish more than it witnessed?** The original repair-only report promotion and borrowed-timeliness bugs are repaired. Meaning and continuity remain reviewer judgments, and the validator accepts the incoherent ending/alert claims in finding 6. The fixture problem is concrete: a name-bearing status message establishes a delivery label it did not witness.

7. **Next work:** repair findings 1–5, add their independent regressions, rerun the pair and mutation evidence, address or explicitly narrow findings 6–8, then proceed to the planned record/memo result review. This verdict authorizes no merge, commit, push, deployment, network/box read, Route R access or production repair.

Final verification: the submitted JUnit has 272 cases, zero failures/errors and 22 expected-failure skips; the independent-probe JUnit has 19 cases and zero failures/errors/skips. All throwaway source mutations were restored before those final runs. The authorized wt5 was removed with `git worktree remove --force` and is absent from both the filesystem and Git's worktree inventory. The real-pair synthetic worktree was also removed; its evidence remains in the scratch directory. Main remains at a0f3ebfc53be7806a250bf0518585d57b01fc5e9 with a clean tracked checkout. No data/ contents, network, SSH, gh, production operations or main-repository commits were used.

DONE
