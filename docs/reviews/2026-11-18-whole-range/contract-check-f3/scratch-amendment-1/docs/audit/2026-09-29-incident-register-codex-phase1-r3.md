## Verdict

**BLOCK.** Reviewed plan rev 3 at **9907cb93aa509b0a9c9ad04bd0a588068748e10d**, code at **638a982**, and the changes from **39072e3** against design v3.1.

The ownership guard, hard mutation-path rule, absolute observer loading, structured session reports, historical acceptance artifact, and E77 success branch materially improve r2. The remaining failures concern what an accepted result establishes:

- An accepted absence certificate can coexist with a real qualifying send.
- The observer can change a failing test into a passing test.
- Historical replay accepts a regression manufactured entirely by a changing test helper, with identical parent/fix production trees.
- Expected-failure acceptance permits changed helpers and a literal bad value unrelated to the invoked production function.
- Deployment bounds and record validation permit stronger claims than the evidence supports.

These need implementation changes and fresh evidence runs. A plan-only amendment cannot cure them. This review does not reject the actual 18 fixture failures merely because the reusable acceptance machinery has defects.

**Measured scope.** Python 3.12.13, pytest 9.0.2; offline cached dependencies; authorized detached wt3 plus synthetic repositories. Commands used UV_CACHE_DIR=/tmp/uv-cache, UV_OFFLINE=1 and TZ=America/New_York.

| Check | Result |
|---|---|
| Submitted review suite: tests/scripts/incident_register plus both incident fixture files | **121 passed, 18 xfailed, 1 skipped** |
| Submitted schema module, separately with jsonschema==4.23.0 | **22 passed** |
| Independent execution/observer probes | **11 passed**, where a passing probe asserts the reported false acceptance or conservative rejection |
| Deployment/schema probes | **6 passed**: one deployment counterexample, four invalid-record acceptances, one valid scope control |
| Actual expected-failure driver on a synthetic snapshot of 638a982 source/tests/config/registry | **18/18 accepted**, no session reasons; marked exit 0, unmarked exit 1 |
| Real-source run_day no-op, E77 selection | **2 failed, 5 passed, 28 deselected; no XFAIL** |
| Final standard suite, including the 11 independent probes, after restoring the no-op | **132 passed, 18 xfailed, 1 skipped** |
| Final separate schema suite, including the six review probes | **28 passed** |

The actual expected-failure run used exact archived source/test bytes but a synthetic repository ref, f1aa6c6f592b97f3dd64cf7b43704545847b81ee. It is not a claim that the full original Git repository was its execution environment.

**Environment qualification:** a combined run under uv run --with jsonschema==4.23.0 produced **48 failed, 112 passed, 18 xfailed**. Synthetic inner interpreters could not import pytest. tests/scripts/incident_register/synth.py:58–59 exposes only sysconfig's purelib, which becomes the temporary jsonschema overlay; pytest actually remains in wt3/.venv. An inner stderr confirmed ModuleNotFoundError: No module named 'pytest'. The standard run and separate schema run above avoid that harness-environment problem. Those 48 failures are not evidence of production regressions or the false greens below.

Probe sources, logs and the real pair's raw event files are retained at **/private/tmp/bts-incident-register-r3-uxymn_zh/**, with SHA256.json. The two test_review_r3_*.py files can be copied into a disposable checkout to rerun; they use the submitted synthetic helpers. These are temporary review evidence, not a published incident bundle.

## r2 findings status

Statuses cover each finding's acceptance claim, not merely whether its original example has a named test. The measured reruns below include the submitted regression tests; independent extensions are distinguished explicitly.

| r2 | Status | Measured rerun and remaining issue |
|---|---|---|
| **1. Destructive operations not confined** | **RESOLVED** | All **23 test_owned cases** passed, including primary checkout, subdirectory, symlinked root, unowned/attached worktree, traversal, absolute path, symlinked component and validation-before-write. Defence's traversal/primary-checkout refusals also passed. The private ownership record plus exact-root/linked/detached checks close the demonstrated accidental destruction and escape cases. This is an operational guard, not protection against a same-user adversary forging its own ownership metadata. |
| **2. Allowlist can authorize an oracle mutant / frozen harness** | **PARTIAL** | The exact test-expectation edit, even explicitly allowed by the caller, is rejected. The defence regression that writes a tracked harness file during execution is rejected. Hard src/bts/*.py enforcement is now independent of the allowlist. However, replay and expected-failure pairs do not apply the same stage-by-stage freeze, and the environment fingerprint does not hash installed code. Independent helper-drift acceptances remain: new findings 3–4. |
| **3. Run-wide errors and imperative-XFAIL baseline accepted** | **PARTIAL** | Baseline collection error, unrelated mutant teardown error, imperative-XFAIL baseline, surviving killing node, undeclared assertion and stage-only failures are rejected in the submitted defence/runner tests. **Independent replay with an unrelated teardown error is accepted**, because replay_red skips the phase/return-code accounting block. New finding 3. |
| **4. Oracle suffix match / no production invocation** | **PARTIAL** | Submitted foreign-oracle, wrong exception-module, no-entry literal, missing-value message, imperative/non-strict marker, teardown and changed-test-byte probes all passed their rejection assertions. The real 18-node pair is accepted. **Independent unused production invocation followed by a literal bad oracle is also accepted**, even though production returns the required value; a changed non-collected helper is accepted across the pair. New finding 4. |
| **5. Absence not bounded; unsupported async accepted** | **PARTIAL** | The old synchronous send elsewhere, missing positive control, wrong assertion, branch outside entry and actual async-entry cases are now rejected. Independent C-callback and pending-thread scenarios still obtain accepted absence certificates. Actual joined-thread send is seen and rejects absence. Actual frame-address reuse is correctly separated. New finding 1. |
| **6. Shadow observer / own-venv identity not enforced** | **PARTIAL** | Observer loading now uses an absolute pinned copy and passes the module object to pytest. Submitted identity tampering, planted pytest.py, foreign bts and worktree-pytest cases are rejected. Own prefix/executable/root checks work. The stronger frozen-execution claim remains incomplete: installed dependency code can change without changing the fingerprint, and imports are hashed at session end rather than bound to the code loaded at import time. New finding 4. |
| **7. E77 success branch impossible** | **RESOLVED** | Fixed-direction verdict, dedicated bad-direction exception and three ordinary-error cases pass. The real correct-schedule execution control passes through _e77_verdict. A fresh source no-op gives **2 ordinary failures, 5 control passes, no XFAIL**. Bad-path check-time/classification/alert-only assertions now apply only on the bad outcome. |
| **8. Historical acceptance object / retained audit missing** | **PARTIAL** | Submitted accepted replay writes acceptance.json and retains the audit. Missing reason, missing adapter evidence, unaudited change, new-API failure, undeclared assertion and collection-error cases reject; rename-path coverage passes. Independent run-time harness drift and red-stage teardown errors nevertheless yield **accepted**. The new artifact exists, but its acceptance rule remains unsound. New finding 3. |
| **9. Contradictory plan / inherited pytest root** | **RESOLVED** | Tasks are rewritten in place; old v1/v2 runners are removed and preliminary results are labelled. Standard nested synthetic tests pass with their own config and explicit root. The separate uv overlay issue above is narrower and does not recreate the old ancestor-config/node-ID mismatch. |

## New findings

### 1. BLOCKER — absence certification does not establish a complete event set or the required interval

**Locations:** scripts/audit/incident_register/observer.py:14–25,156–168,318–345,384–398; certify.py:25–39,87–99; plan:18,54–56,99; design §9.3.

**Measured C-callback false absence.** A frozen test patches the real declared transport boundary, invokes production, successfully asserts send.call_count == 1, and then checks the returned status. The production mutant changes:

    transport.send("eric", "pick: Turner")  # BRANCH
    return "done"

to:

    list(map(transport.send, ["eric"], ["pick: Turner"]))  # BRANCH
    return "defer"

The final return-status assertion fails at its registered line. The send genuinely happened during the observed call phase; stdout and the independently asserted mock call count confirm it. Result:

    verdict = "accepted"
    reasons = []
    certificate = {"ok": true, "linked": {"calls_observed": 0}, "reasons": []}

A direct production CALL event discovers the boundary; map invokes it from C, so no pending boundary identity/event is created. The initial callee-code handling does not independently record every boundary entry. The module docstring acknowledges C calls are unobserved, but that limitation neither prevents this certificate nor appears in its output. A limitation note would still be insufficient to certify absence over a fixture that uses the omitted path.

**Measured incomplete interval.** Another production mutant starts a worker at the branch and returns "defer". The worker waits on an Event. The same ordinary return assertion fails; fixture teardown releases and joins the worker, and successfully asserts that exactly one pick was sent. All pytest teardown checks pass. The runner again accepts absence with calls_observed = 0. This send occurs after obs_end, not invisibly inside it. That distinction is the defect: the runner never establishes that the call-phase interval covers the contract's relevant work, invocation completion or controlled timeout.

**Controls:** a joined worker sending within the observed phase is recorded and rejects absence with one qualifying call. The old direct synchronous send outside the declared entry is also rejected. This is not a blanket claim that threads or out-of-entry calls are invisible.

**Smallest fix:** make missing-event certification consume a complete recorder at the actual observable boundary, over a declared controlled interval, with completion/timeout and outstanding-worker/task disposition. Retain qualifying event identities and a same-contract baseline positive control. Support the C callback path or refuse certification when its coverage cannot be established. A failed return-status assertion plus an empty partial trace must not establish a failed delivery contract. Apply the design amendment below without weakening these requirements.

### 2. BLOCKER — the external observer can execute application code and manufacture a green baseline

**Locations:** observer.py:79–84,264–275,403–411; defence.py:122–127; plan:18.

**Measured:** the production return object has:

    seen = False
    class Value:
        def __repr__(self):
            global seen
            seen = True
            return "VALUE"
    def grade():
        return Value()

The test calls grade() and asserts seen is True. Under runner.run without a return observation it **fails**. Under the same runner with grade registered in observe.returns it **passes**, and gate(..., mode="green") returns no reasons. The observer calls repr(retval) before application execution resumes.

Boundary identity formatting also calls _short; a non-string value can be represented more than once. Exception formatting and generic attribute access deserve the same audit. Catching a representation exception does not undo its side effects.

**Smallest fix:** do not invoke arbitrary repr, str, properties or other application-defined operations to observe values. Serialize a reviewed set of exact primitive types; mark unsupported values unavailable. Add observer-off/observer-on conformance probes, including side-effecting and raising representations. Retain a genuinely unobserved green baseline where needed. Source-byte equality and “edits nothing” do not prove observational behavior.

### 3. BLOCKER — historical replay accepts changing harness bytes and unrelated red-stage errors

**Locations:** replay.py:122–157; runner.py:199–214; plan:26,55,58.

**Measured harness-only regression, with identical src trees.** Production grade() returns "void" in both parent and fix. A frozen tracked helper initially contains EXPECTED='void'. The added test does:

    env = {}
    p = Path(__file__).with_name("expected.py")
    exec(p.read_text(), env)
    p.write_text("EXPECTED='miss'\n")
    assert mod.grade() == env["EXPECTED"]  # ASSERT-VOID

The green run passes and changes its helper. The red run uses the changed expected value and fails at the registered assertion. The final reset removes the evidence of drift before manifest comparison. Historical replay returns **accepted**, with both recorded src-tree IDs equal to **04e53a6bd41ae919839215ee82943b48c1da9d0e**. There is no production regression to reproduce.

**Measured unrelated error.** A normal miss-to-void parent/fix replay has a second selected test whose autouse teardown raises ValueError only on the parent. The acceptance object reports:

    verdict = "accepted"
    reasons = []
    classes = {
        "tests/test_grade.py::test_pass_is_void": "symptom_candidate",
        "tests/test_grade.py::test_unrelated": "teardown_error"
    }

The replay_red gate intentionally skips all per-node phase and explained-return-code checks. Classification does not repair that acceptance.

**Smallest fix:** compare frozen working bytes and the execution environment immediately after each subprocess, before any reset or src swap. Share the full phase/session gate across modes: replay may allow classified call failures, but setup/teardown/collection errors, missing/duplicate phases and unexplained return codes must reject certification. Preserve red classifications as diagnostics. A restored tree is evidence of cleanup, not evidence that the preceding runs used the same harness.

### 4. BLOCKER — expected-failure pairs are not bound to the same execution closure or to the production result

**Locations:** acceptance.py:25–29,45–52,73–88; run_expected_failures.py:26–34; owned.py:181–194; observer.py:171–179; runner.py:189–198; design §9.1 and §9.7.

**Measured changed-helper acceptance.** The test passes its production result through tests/inputs.py::choose before invoking the registered oracle. Between marked and unmarked executions, change that tracked helper from returning its argument to returning literal "miss". The collected test file and bts modules stay identical. acceptance.accept returns:

    {"_session": [], "tests/test_incident.py::test_pass_is_void": []}

Only collected test files and bts module maps are compared. Imported test helpers, conftest, config and other execution inputs are not included. The driver resets after both runs, without intermediate drift checks.

**Measured disconnected value acceptance.** Keep production correct: grade() returns "void". Change the test to:

    mod.grade()  # result unused
    _oracle("miss", "void", "miss", GradedAsMiss, "pass")

The exact registered class, file, oracle and message are genuine. Both runs use identical bytes. The pair is accepted with no session or node reasons. Entry presence anywhere in the call phase does not connect the declared bad value to production.

**Measured environment-fingerprint gap.** Create a local site-package plus its dist-info/RECORD, fingerprint the owned venv, then change the package's Python bytes while leaving RECORD untouched. The fingerprint is identical. It hashes metadata contents, not the installed files described by that metadata; .pth target contents are not covered either. Thus even current_defence's repeated fingerprint checks do not establish unchanged dependency code. The submitted synthetic helper itself intentionally exposes an outer site-packages directory.

**Related source limitation:** the imports record reads module files at session finish. Comparing those hashes with the same files after the process exits does not demonstrate which bytes produced the already-loaded code.

**Smallest fix:** freeze and hash the full declared execution closure before and after every stage, including helpers, conftest, scripts/config/inputs and actual executed dependencies. Verify RECORD entries against installed bytes or use an equivalent immutable, content-addressed environment; cover external .pth dependencies. Bind the registry and observer configuration to output hashes. Require reviewed, structured actual/required/bad evidence connected to the production return or persisted state in both runs. Until that connection is implemented, label the generic predicate as an exception-shape check; it cannot independently confer reproduction status. Keep the fixture-specific oracle review.

### 5. BLOCKER — deployment interpolation turns an interval into a falsely exact timestamp

**Locations:** deploy_runs.py:145–179,188–204; .github/workflows/deploy.yml:89–113,160–181; plan:21,67–68.

**Measured:** feed the extractor legitimate-shaped synthetic lines:

    12:00:00Z Pre-deploy SHA: aaaaaaa
    12:02:00Z Deployed bbbbbbb
    12:02:30Z CANARY PASSED — bbbbbbb is live and healthy

For a fix first contained in bbbbbbb, first_live returns not_live_before = live_by = **12:02:00Z**. live_at(12:01:00Z) asserts **aaaaaaa**, basis **log**.

The workflow captures/logs the old SHA, fetches, resets, syncs, restarts, then prints Deployed. Checkout and service activation can therefore happen between these observations. Taking the *end* of the old assumed segment as the latest observation of absence creates the false exact lower bound. Mathematically, (t,t] is empty, not an exact-time interval.

**Additional overstatement:** matching successive pre/post SHAs demonstrates endpoint agreement. It cannot prove the plan's parenthetical “no out-of-band box change”: change-and-revert between observations would leave the same endpoints. Checkout evidence also does not by itself establish every already-running process's code or healthy service state.

**Smallest fix:** retain observed transition endpoints separately from inferred continuous segments. For this example, the supported transition interval is between the pre-deploy observation and the post-deploy observation, subject to the logs' timestamp precision; successful canary is a later health observation. Intervening lookups must be unknown/transition or explicitly conditional, not log-observed old SHA. Preserve rollback/anomaly handling and distinguish earliest observed installation from proof of the first-ever live installation.

Replace plan Task 4's continuity/exactness claim with:

> The retained logs provide SHA observations and deployment-transition bounds. Matching successive pre/post SHAs establishes endpoint agreement; continuity between observations is an explicit assumption, not proof that no out-of-band change occurred. For each fix, retain the latest supported observation without the fix and earliest supported observation containing it, with unknown endpoints where necessary. Keep checkout, service restart and canary health observations separate. Do not assign a single exact installation timestamp from a post-deploy log line.

### 6. BLOCKER — record validation permits unsupported certification and inconsistent evidence

**Locations:** record_schema.json:102–110,133–149,152–182; records.py:75–90,123–141; plan:28,71,82–83; design §3 and §7.

Four independent mutations of the submitted valid-record helpers each return **validate(...) == []**:

| Accepted record | Missing or contradictory fact |
|---|---|
| Historical replay status certified; clean label; fix_set=[]; symptom_nodes=["anything"]; acceptance="" | No nonempty fix set, acceptance artifact or reviewer acceptance. Unlike defence, replay does not even define a reviewer_decision property. |
| deployed={"basis":"log"} | No deployed SHA, time/bound or evidence reference. |
| notification latency min_minutes=50, max_minutes=2; confirmed alert has a timestamp; first_machine_detection remains unknown | Reversed latency bounds and missing subtraction endpoint both pass. |
| Contemporaneous operator report written July 1 for an exact July 16 occurrence | The check only rejects reports too far after onset; a pre-event report qualifies as primary observed-incident evidence. |

The validator does not load and bind claimed acceptance artifacts to their runner verdicts, nodes, hashes and reviewer decisions. Requiring a string field is not enforcing “runner accepted AND reviewer accept.”

**Smallest fix:** require a nonempty, hash-bound acceptance reference and reviewer accept for every certified replay/defence; validate the referenced runner result at publication. Require the SHA/time/evidence appropriate to each non-unknown deployment basis. Require both endpoints of each numeric latency, ordered min/max, and consistency with endpoint bounds; otherwise preserve unknown. Link a contemporaneous report to the occurrence it describes and enforce the stated after-event window, handling date/interval uncertainty explicitly.

**Scope control, not a finding:** a fixed Tier-A deployed_latent_defect with empty replay/defence arrays also validates. Design §1 explicitly mandates those two entries for fixed Tier-A **observed incidents**. I am not treating that accepted latent record as a violation of the current design.

### 7. SHOULD — dynamic rebinding and recursion have narrower support than the plan claims

**Locations:** observer.py:302–315,318–345,393–396; certify.py:42–51,76–82; plan:54,63.

**Measured dynamic-binding refusal:** invoke a normal test helper once, then bind it as the transport boundary and call it from production. Its first PY_START was disabled as non-production code. Adding its code object to callee_codes does not restart that disabled event during the observation. The call is recorded but its identity remains unresolved; certify.interval rejects it. restart_events happens only at stop. This is a conservative refusal, not a false green.

**Measured nested invocation:** outer deliver executes the branch, recursively invokes deliver, and the inner invocation sends. Both frames are on the boundary stack. _linked always chooses the latest/innermost entry; the outer branch and inner event therefore do not match. Direct branch→send control certifies successfully.

**Measured frame reuse control:** two sequential deliver invocations actually reused the same frame address. The first executes the branch, the second sends; certification correctly rejects the disconnected pair. The existing latest-entry rule addresses that case.

**Smallest fix:** reactivate a dynamically discovered boundary's relevant monitoring event, or retain the minimal callee observation needed to avoid disabling it. Define recursive invocation semantics explicitly: if an outer connected stack chain is supported, select a common live invocation for branch and event rather than independently choosing only the latest entry. Otherwise document nested/rebound unsupported cases and preserve the conservative refusal. Do not advertise every dynamic boundary as supported.

### 8. SHOULD — the expected-failure driver lacks the promised failure-path artifact and cleanup

**Locations:** run_expected_failures.py:24–43; acceptance.py:15–16; plan:26.

**Source-verified:** the new driver has no try/finally around either run. A missing/malformed registry, subprocess exception or acceptance exception bypasses the trailing reset and output. Its output omits the registry hash despite acceptance.py saying it is hashed into acceptance output. Its summary also counts a node as accepted from empty per-node reasons even when _session contains rejection reasons.

This is a failure-handling/provenance defect, separate from the measured successful false acceptances above.

**Smallest fix:** construct a rejected result before work starts; after ownership validation, restore in finally and write a rejection artifact for every handled failure. Include resolved ref, registry/config hashes and an overall verdict. Session rejection must make the pair rejected and prevent a success count from being presented as accepted reproductions.

## False greens

| Passing assurance | Independent observation |
|---|---|
| “Every production→boundary call,” zero observed calls and a positive control | C map callback really sends; absence certificate and runner both accept. |
| Full call-phase window | Pending worker sends after obs_end; teardown joins it and passes; absence still accepted without a completion/horizon proof. |
| External observer edits no source files | Observing a return invokes __repr__, changes application state and turns failure into pass. |
| Historical src swap + final restored manifest | Parent and fix src tree IDs are identical; changing test helper creates red; accepted. |
| Shared gate rejects teardown errors | replay_red accepts a run explicitly classified with teardown_error. |
| Identical collected tests + bts imports across an XFAIL pair | Changed imported test helper is accepted. |
| Exact oracle plus production entry invocation | Correct production result is discarded; literal bad result gets accepted as reproduction. |
| Venv fingerprint unchanged | Installed Python code changes while metadata hashes remain equal. |
| Retained pre/post SHA chain and passing extraction tests | first_live creates equal lower/upper bounds and mislabels the transition's earlier instant as observed old code. |
| Validated certified record | Empty acceptance reference, no reviewer decision and empty fix set pass. |
| 22 passing schema tests | Reversed notification latency with unknown detection and a report predating its occurrence both pass validation. |

**What did not falsely pass:** wrong-receiver probe was rejected; joined worker send was seen; actual reused frame addresses did not connect separate invocations; async entry and unresolved dynamic identity fail closed; E77 no-op remains an ordinary failure.

**Mutation sweep:** the saved sweep is useful evidence that selected checks matter, not evidence that all acceptance paths are covered. Its script at tooling/mutation_sweep.py:64–67 labels any nonzero pytest return as KILLED, including collection/import/environment errors, and does not require the intended assertion failure. I did not rerun the entire sweep. Retain per-mutant clean-baseline and intended-failure evidence before using its all-killed summary as an acceptance argument. The independently reproduced false greens above occur on unchanged submitted tooling.

## Ruling 2

**Not approved as implemented.** An external observer can be an acceptable design choice. sys.monitoring does not by itself guarantee a complete boundary recorder, causal value evidence or observational behavior. The measured C omission, unfinished worker and side-effecting repr directly defeat the current equivalence claim. Moving instrumentation outside source files removes one editing risk; it does not discharge §9.3.

**Yes, amend design §9.3 before the memo.** Replace its step (3) and witness-specific concluding requirements with the following text; retain the existing semantic-mutant, restoration and certificate-level requirements:

> (3) A causal path witness may use observational hooks compiled into the mutant or a trusted external observer. Pin the loaded observer, configuration and execution environment. Within the killing node, record the declared production entry invocation, execution of the mutated decision, and the checked observable boundary or returned/persisted value on one connected invocation chain. Match code identity and invocation identity, including process/thread/task where relevant, and bind the observable to the fixture's date, selection and declared contract. Record ordered events; a reused frame address, a separate invocation or an unrelated literal oracle value is insufficient.
>
> Instrumentation must be observational: it must not execute application-defined formatting, properties or callbacks to obtain observations, change control flow, or manufacture application state. Unsupported values or identities are unavailable. Validate observer-on behavior against observer-off controls; unchanged source bytes alone are not proof.
>
> For a wrong or extra event, require the linked invocation and the identity of the event checked by the oracle. For a missing event, witness the mutated decision and completion or controlled timeout of the same declared invocation, then consume an independent recorder at the actual observable boundary covering the full required controlled-clock interval. It must remain active across every relevant callback, worker or task until that interval closes, and establish zero qualifying events. Include the equivalent baseline positive-event control. A call-site trace alone is insufficient unless complete boundary coverage for that execution is independently established.
>
> Declare supported execution modes and their coverage in each certificate. If C callbacks, dynamic binding, recursion, threads, async tasks, subprocesses, pending work, missing identities or observer errors prevent the required evidence from being established, report unavailable with the reason. Never turn an unobserved path into an absence claim. The runner's acceptance and the recorded review decision remain separate requirements.

This is a proposed amendment, not approval of 638a982 under it. Fix findings 1–4 and rerun the relevant probes before claiming equivalence.

## Answers

1. **R2 closure:** the original ownership/path, shadow-loader, phase, suffix-oracle and E77 counterexamples received real repairs. The table distinguishes those measured repairs from their still-open broader claims. No production repair is authorized by this verdict.

2. **Observer and linking:** ordinary synchronous direct boundaries work in the positive controls. Global production-call observation improves on an entry-local tag, but omits some true boundary invocations and does not establish the required completion interval. Dynamic DISABLE handling and recursion currently produce conservative refusals; actual frame-address reuse did not produce a false connection. Process/task coverage and the complete contract identity still require explicit support or unavailability.

3. **Ownership, gate and current defence:** ownership validation precedes destructive work and validates edits before writing. The hard production-only mutation rule is sound for the exercised paths. GREEN→MUTANT→RESTORE, frozen assertion location, whole selected-run checks in defence, and restoration checks all help. They do not prove full execution identity, make arbitrary observer operations harmless, or connect every declared failed assertion to the registered symptom. replay_red and the pair driver must enforce equivalent controls.

4. **E77:** accept the fixture restructuring. The schedule mock uses the controlled instant: 19:10 before the declared noon move, 18:10 afterward; the cascade supplies a confirmed selection with the true start. The required branch can now succeed without satisfying the bad-path timing or alert-only predicates. The positive execution control and no-op probe exercise real run_day behavior. Keep **component** as the ceiling; the mocked cascade/internal control points remain explicitly listed. This is fixture acceptance, not an implemented scheduler repair.

5. **Plan consolidation:** resolved. Tasks 5–8 correctly remain unfinished work followed by result review; old evidence is preliminary. Amend Task 3's completeness/observational claims, Task 4's deployment inference, and Task 8's publication gate as above. The global promise that every stage passes the gate and every failure path writes an artifact must also hold in the drivers.

6. **Task 4:** the checked-in typed corpus contains **192 runs: 164 unavailable_expired, 28 retained**. Fixed output templates and exclusion of echoed shell source are useful. I did not refetch logs. Accept those corpus counts, not the inferred exact installation times or proof of continuous unchanged checkout. Phase 2 deployment-history evidence remains pending.

7. **Task 7 negative review:** the two retained TSVs contain **186 rows each, 372 unique commit hashes**, no empty cells, **169 sensitive-path rows** and **28 possible_fix_of_live_behaviour rows**, matching the plan's counts. The rows identify changed functions/paths and exclusion reasons, materially stronger than subject-only screening. These counts do not independently prove complete membership of the original 1,172-commit universe or all adjudications. Preserve that universe, exclusions, row-to-candidate dispositions and the seeded QC sample for result review. The referenced route_h_candidates.json is not present at the reviewed code pin; Task 7 is explicitly in progress, so this is an outstanding deliverable rather than a fabricated result.

   Source spot checks support treating the additions as candidates: 2be445e adds saver-aware local grading/state; 5207a09 adds worker pulls; b430e45 adds the second matchup to DD posting; 18efce1 changes unconditional heartbeat behavior; 30452eb pins deployment to the tested SHA; e5ef7ca changes unverified/missing-slot entry handling; the saver-proxy changes support E111's mechanism investigation. Code changes establish mechanisms, not occurrence dates, deployed execution or user impact.

   **Keep E106's additional mechanism:** 947fce8 also changes src/bts/data/sync.py::sync_to_r2 to include mdp_policy.npz. Its commit body reports heuristic fallback on Fly; the negative-review row also names the missing policy. The plan's short E106 label mentions only slow O(n²) code. Route the policy omission explicitly to E16/E106 or another related record, with its own mechanism/disposition and evidence strength; do not lose it when folding the commit into one episode. E107 should retain the exposed-ping-URL candidate without copying the URL into the memo. Further record-level adjudication remains for the result review.

8. **Task 8:** accept the direction of a closed schema with separate implemented/deployed/mitigated/verified fields, dispositions, residuals and explicit unavailable reasons. Do not call publication validation complete until finding 6 is repaired. A valid JSON shape is not an evidence certificate. The latent-Tier-A scope control is consistent with the present design and is not a requested expansion of scope.

9. **Remaining boundary:** no production data or live operational evidence was read; no SSH, network or gh call was made. Only authorized throwaway mutants and synthetic commits were used. The next step is a tooling revision and fresh acceptance evidence, followed by the already-planned result review; this report grants no merge, commit, push or deployment approval.

**Final verification:** after restoring the throwaway source mutant, the standard suite with review probes returned **132 passed, 18 xfailed, 1 skipped**; the separate schema run returned **28 passed**. Both tracked and staged diffs in wt3 were empty before removal. The authorized wt3 directory and Git worktree registration are now removed. Main remains at **9907cb93aa509b0a9c9ad04bd0a588068748e10d** with empty git status --short. Other project worktrees were left alone. The report and the temporary, hash-listed probe evidence named above are retained.

DONE
