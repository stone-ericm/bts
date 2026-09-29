# W1.5 Phase 1 plan + code review — r2

## Verdict

**BLOCK.** The L01/L02 fixture repairs work, and the old E77 no-op false green is closed. The v2 tooling still accepts invalid evidence and does not confine destructive operations to a disposable evidence worktree. Do not promote the preliminary runs to certificates yet.

Reviewed pins:

- Plan rev 2 / main: `d8563a72c55ade873de5111347d1e7eef3f4bba0`.
- Code: `39072e3fc31a8b658e24fcbdab7a2c1afbb8bb1a`; r1→r2 diff from `3f6fd63`.
- Governing design: `docs/superpowers/specs/2026-09-29-incident-register-design.md`, v3.1, especially §9.1–9.8 and §10.
- Prior findings: `docs/audit/2026-09-29-incident-register-codex-phase1-r1.md`.

**Measured:** Python 3.12.13, pytest 9.0.2; offline `uv sync --extra model` in the authorized detached `wt2`. The standard submitted review suite passed **57 tests, 18 XFAIL**. The fixture file alone collected 31 nodes: **13 pass, 18 XFAIL**; `--runxfail` produced **13 pass, 18 call-phase failures**, all ending at `_oracle`: 5 `PassGradedAsMiss`, 10 `PassScoredAsNoHit`, 2 `SameDayReplayRollback`, 1 `SingletonSlateUndelivered`. Calling the new acceptance predicate explicitly accepted those 18 marked nodes.

Baseline command, from `wt2`:

```sh
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 TZ=America/New_York uv run pytest \
  tests/test_incident_register_2026.py tests/test_incident_register_2026_meta.py \
  tests/scripts/incident_register -q
```

Fourteen independent temporary probe tests also passed their assertions about the defects and repairs below; those are **counterexample checks, not evidence that the tooling is correct**. All repository-source mutants were restored byte for byte. No production `data/`, remote host, network, or `gh` was used. Synthetic repositories created by the submitted fixtures and review probes were the only repositories in which test commits/reset experiments were performed; BTS main and the submitted branch were not edited or committed.

Test-environment qualification: my first attempt used a missing scratch parent and failed before the temporary-file fixtures could run. After creating it, an in-repository `--basetemp` exposed ancestor-pytest-config/node-id assumptions: 11 failed, 46 passed, 18 XFAIL. The standard command above, using pytest's normal temporary root, produced the clean 57/18 baseline. This distinction is not being counted as a product blocker; see finding 9.

## r1 findings status

| r1 finding | Status | Measured rerun and remaining gap |
|---|---|---|
| 1. E77 no-op false green | **PARTIAL** | An immediate `return None` in real `scheduler.run_day` now yields **2 ordinary failures, 1 pass, zero XFAIL** across the three E77 nodes. The oracle-only control is honestly named; the positive execution control passes on baseline. But the marked test cannot accept its required successful delivery outcome: its preceding assertions require the bad sequence and alert-only DMs. Finding 7. |
| 2. Harness mutation / incomplete restore | **PARTIAL** | Repeated the actual root-conftest replacement of `grade` with entry/branch/boundary hooks. With the normal production-only allowlist, v2 raises `RuntimeError: mutation touches paths outside its allowlist: ['conftest.py']`, and restores the original conftest. Normal in-tree cleanup also passes. New probes nevertheless accept an expected-value mutation when the caller includes the test in `allowed_paths`, and a `../outside.txt` edit bypasses that allowlist entirely and survives restore. `reset_worktree` also destroys unsaved work in a synthetic primary checkout. Findings 1–2. |
| 3. Witness co-membership / absence | **PARTIAL** | Reran the supplied real-hook regressions: boundary-before-branch, boundary-present absence, wrong-file same-name entry, disconnected calls and two entries are rejected. Missing positive control is rejected. A real send after the declared entry returns still receives an accepted absence certificate; an `async def` crossing an actual `await` receives `ok=True` despite the advertised unsupported-path rule. Finding 5. |
| 4. Import / JUnit identity | **PARTIAL** | The foreign-`bts` conftest case is rejected by the new in-process import check. Independently reran `TypeError("first\nE   AssertionError: last")`: exact `TypeError`, module `builtins`, original `test_msg.py::test_msg` retained. All node phases are now available. Collection errors/process status are not enforced; the standalone plugin can be shadowed by a worktree module while the wrong hash is recorded. Findings 3 and 6. |
| 5. Expected-failure acceptance / wiring | **PARTIAL** | The new predicate rejects the old direct-body exception, imperative xfail, teardown-error, setup and non-strict cases in real pytest reports. Explicit invocation accepts all 18 actual fixture XFAILs. But there is **no call to this predicate in the evidence runner**; an imperative-XFAIL baseline killing node is accepted. The new frame check also accepts a different file named `foreign_test_h.py` for `test_file="test_h.py"`. Findings 3–4. |
| 6. Historical harness audit | **PARTIAL** | Script and selected-test changes now remain `semantic_regression_replay_unaudited` without decisions; partial audit remains unaudited; rename output retains both paths. Those specific old false-clean labels are closed. A clean historical result does not retain the supplied audit decision/reason, enforce run acceptance, or produce the promised acceptance object. Finding 8. |
| 7. L01/L02 contract gaps | **RESOLVED** | Explicit bench/zero-stat DNP, after-resumption clock, all required direct/CLI controls, whole outcome tuples, all-Pass and saver cases are present. Unmarked L02 reports contain **both** observations: `[(1, True), (1, True)]` and `[(10, True), (10, True)]`. The preview is genuinely undelivered. An over-broad AB=0→void mutant fails both SF-only controls; changing only saver preservation fails the whole-state oracle ordinarily; corrupting only the second reconcile produces two ordinary `AssertionError` failures, zero XFAIL. A temporary L01/L02 fix sketch produces **17 XPASS(strict) failures, 11 passing controls**. No production repair is authorized by these experiments. |
| 8. Procedure / records | **PARTIAL** | The new diff-level negative-review requirement, SHA/log-bound deployment intervals with unknowns, schema validation, and explicit reviewer decision address the procedural omissions. The acceptance-object amendment does not match the historical runner, and current-defence's accepted verdict is unsafe. Old task text also remains contradictory. Findings 3, 8–9. Candidate counts, retained deployment logs and final records remain future result-review inputs, not verified measurements here. |

Legacy `connected`, `current_defence`, JUnit parsing and `historical_replay` remain preliminary interfaces. I assessed certificate acceptance against their designated v2 replacements; relabeling the legacy interfaces does not repair or validate old certificates. The plan correctly requires rerunning those batches.

## New findings

### 1. BLOCKER — destructive reset and edits are not confined to an owned evidence worktree

**Locations:** `scripts/audit/incident_register/evidence.py:254–261,281–301,313–323,380–389,453–469`.

`reset_worktree` immediately executes `git reset --hard <ref>` and `git clean -fdx -e .venv` in whatever directory it receives. It checks cleanliness **after** destruction. There is no exact Git-root check, detached-worktree check, ownership/creation record, or exclusion of the primary checkout. A subdirectory also lets Git resolve the enclosing repository. `apply_edits` writes `worktree / rel` before validating the changes, and accepts `..`, absolute paths and symlink escapes. `changed_paths` cannot see edits outside that Git worktree.

**Measured, synthetic only:**

1. Called `reset_worktree(primary_repo, "HEAD")` on the fixture's **main** checkout after planting an untracked `keep_me.txt` and editing tracked `src/bts/mod.py`. It returned normally, removed the sentinel and discarded the tracked edit; branch remained `main`. The real BTS checkout was never passed to this function.
2. Appended `["../outside.txt", "untouched", "CHANGED"]` to the normal valid mutant's edits, with `allowed_paths=["src/bts/mod.py"]`. Result:

```json
{"verdict":"accepted","touched":["src/bts/mod.py"],"outside":"CHANGED\n"}
```

The sibling file was inside the synthetic test's temporary directory. The final reset did not restore it.

**Smallest fix:** validate all targets before the first mkdir/reset/write. Require the canonical exact root of a task-created disposable detached worktree, with a recorded repository/worktree identity; reject primary checkouts, subdirectories, symlinked roots and mismatches. Resolve and validate every edit path beneath that root and against the allowlist before opening it. Reject absolute paths and traversal. Keep evidence outputs outside the reset target. Test refusal without modifying sentinels. Preserve exception-safe cleanup, but apply it only after that identity check. `.venv` exclusion also means environment restoration must be checked separately; Git cleanliness cannot establish it.

### 2. BLOCKER — the caller's allowlist can authorize an oracle-only mutant

**Locations:** `scripts/audit/incident_register/evidence.py:291–301,315,325–326,345–353,382–388`; `tests/scripts/incident_register/test_evidence_v2.py:104–118`; governing design §9.3.

The only protection is membership in caller-provided `allowed_paths`. There is no independent prohibition on tests, expected values, conftest, configuration or the pinned harness. The named `test_conftest_only_mutation_is_refused` actually renames a function in `tests/test_mod.py`; it verifies rejection outside one supplied allowlist, not a hard harness-freeze rule.

**Measured:** retained the normal observational hooks in `src/bts/mod.py`; added `tests/test_mod.py` to `allowed_paths`; replaced only the test's semantic expectation:

```python
# mutation_edits; grade itself still returns "void"
["tests/test_mod.py", 'assert grade() == "void"', 'assert grade() == "miss"']
```

The result was `verdict="accepted"`, `reasons=[]`, touched paths `src/bts/mod.py` and `tests/test_mod.py`. Green / witness-only green / red / restored green all occurred. This is not a restored defective production decision.

The manifest comparison hashes `git ls-files -s` before mutation and after reset. That pins index entries at those two points; it does not verify the executed harness's working-file bytes before/after each run. Witness-only outcome equality likewise establishes only equality of those node states, not that the witness patch contains no semantic changes.

**Smallest fix:** freeze the reviewed harness closure independently of the mutant spec. Hard-reject edits to its tests/oracles/helpers/config/plugin/witness implementation, even if listed as allowed. Permit only separately approved runtime mechanism paths. Compare content hashes at execution boundaries, including after each subprocess, and retain both patch identities. Inspect witness hunks as observational additions; preserve the witness-only run as an additional check, not the whole proof. A mandatory human patch review remains necessary, but should not be asked to compensate for a runner that accepts an expressly forbidden expected-value mutation.

### 3. BLOCKER — acceptance ignores run-wide errors and permits a nongreen killing baseline

**Locations:** `scripts/audit/incident_register/evidence.py:237–251,329–362,385–390`; `scripts/audit/incident_register/evidence_plugin.py:47–48,82–90`; `scripts/audit/incident_register/acceptance.py:39–92`.

The plugin records no collection-error event. `run_structured` discards the subprocess return code; the session-end `exitstatus` it happens to record is never used by the runner. Baseline acceptance is derived from the nodes that happened to collect. The mutant checks the selected killing node, not the complete selected run. Witness-only compares states only for baseline nodes; restored imports are not checked.

Three independent **measured accepted** results:

| Probe | Evidence | Runner result |
|---|---|---|
| Add a tracked module raising `ImportError` during collection; run `tests --continue-on-collection-errors` | Baseline stdout contains `ERROR collecting`; session exitstatus **1** | `accepted`, no reasons |
| Autouse teardown raises `ValueError` for `test_unrelated` only when the mutant's grade is wrong | Structured state of that node is `teardown_error` | `accepted`, no reasons |
| Killing node calls `grade`; when baseline grade is `void`, invokes imperative `pytest.xfail`; otherwise asserts grade is `void` | Baseline killing node is `xfail`; mutant call fails. The killing node is **absent from `result["kills"]`** | `accepted`, no reasons |

The baseline condition at lines 334–335 also treats an empty `expected_xfail` set as permission for arbitrary XFAILs. It does not require the declared killing node to have passed the baseline, be present in `kills`, or validate an XFAIL through `accept_expected_failure`. The latter function is only called by tests, despite plan line 193's “used by the runner” claim.

**Smallest fix:** add one session-level acceptance gate shared by every stage. Retain and check subprocess status, session identity/completion, collection failures, exact expected node inventory, all phases, and interpreter/import identity. Require the killing node to be an unmarked baseline pass and a genuine member of the accepted kills. Any selected setup/teardown/collection error or missing/unexpected node rejects the run. Treat an empty expected-XFAIL registry as permitting none. Wire registered expected failures through the dedicated predicate and paired unmarked evidence; do not silently accept arbitrary `xfail` states. Save a rejected acceptance object on failed/error paths as well.

### 4. BLOCKER — the oracle frame rule accepts a different source file and no production invocation

**Locations:** `scripts/audit/incident_register/acceptance.py:95–123`; `scripts/audit/incident_register/evidence_plugin.py:51–58,70–77`.

The frame check uses `path.endswith(test_file)` and a bare function name. Exception and marker matching use `__qualname__` only, ignoring the recorded exception module. There is no registered contract/bad signature, production invocation, or hash-bound marked/`--runxfail` pair in this acceptance API.

**Measured with actual pytest reports:**

```python
# foreign_test_h.py
class Dedicated(Exception):
    pass
def _oracle():
    raise Dedicated("no production, no bad-value check")

# test_h.py
import pytest
from foreign_test_h import Dedicated, _oracle
@pytest.mark.xfail(strict=True, raises=Dedicated, reason="wrong module")
def test_wrong():
    _oracle()
```

`accept_expected_failure(..., "test_h.py::test_wrong", exc_class="Dedicated", oracle_func="_oracle", test_file="test_h.py")` returned **`(True, [])`**. The retained last frame is in **`foreign_test_h.py`**, exception module `foreign_test_h`; nothing invoked production or checked a bad value. This directly defeats the new claim that it was raised inside the oracle of the test file.

**Smallest fix:** compare exact canonical registered paths and code identities, not suffixes; match module plus qualified class/function identity. Bind each node/parameter to a reviewed oracle, required/bad signature and path proof. Preserve actual/required values and join the marked/unmarked runs by node, source/input hashes and phase sequence. The current fixture `_oracle` does check full values correctly; the general acceptance predicate does not establish that property for a submitted run.

### 5. BLOCKER — “no boundary under this frame” is still not bounded absence

**Locations:** `scripts/audit/incident_register/witness.py:33–46,70–127`; `scripts/audit/incident_register/evidence.py:366–379`; governing design §9.3.

Ordering and entry-file checks are real improvements. However, the absence certificate still equates no `boundary` tag under a frame with an empty qualifying event set. It has no controlled start/end clock, event identity, date/selection, complete boundary recorder, or check that the oracle failed because the event was absent. A `done` tag need not be at an actual completion. The positive control checks a boundary tag under the entry but does not establish the full observation interval.

**Measured accepted absence with a real send:** used this synchronous shape in a synthetic production module:

```python
def deliver_hit(send):
    # entry; ready/branch; boundary before the normal send; done before return
    ready = True
    if ready:
        send("pick")
    return "done"

def run(send):
    result = deliver_hit(send)
    if result == "defer":
        send("pick")
    return result
```

The mutant changes `ready=True` to `False` and return `"done"` to `"defer"`. The unchanged test records `sent`, calls `run(sent.append)`, successfully asserts **`sent == ["pick"]`**, then fails `result == "done"`. With declared entry `deliver_hit` and kind `absence`, v2 reports:

```json
{"verdict":"accepted","reasons":[],
 "witness":{"ok":true,"reasons":[]},
 "positive_control":{"ok":true,"reasons":[]}}
```

Stdout confirms `OBSERVED_SENDS ['pick']`. A reviewer should reject this diagnostic return-value kill, but the runner's absence claim is already false. A local callee completing does not bound all event production in the fixture.

**Also measured:** an `async def async_entry()` with `entry`, an actual `await asyncio.sleep(0)`, then `branch` and `boundary` returns `certify(..., kind="event")["ok"] == True`. The implementation records no task/coroutine identity and never enforces its documented “asynchronous paths unavailable” restriction. This proves the restriction is not implemented; it does not claim this specific same-task trace was disconnected.

**Smallest fix:** make certificate acceptance consume the independent boundary recorder over a declared controlled interval, with qualifying date/selection/event identities and the recorded observable failure. Connect the mechanism trace to that evidence. Require positive baseline observation for the same contract. For currently unsupported async/process paths, detect and reject them explicitly. Keep static hook placement review; tag presence alone must not confer absence or event-identity proof.

### 6. BLOCKER — the “standalone” plugin can be shadowed while a different hash is certified

**Locations:** `scripts/audit/incident_register/evidence.py:230–251,319–321`; `scripts/audit/incident_register/evidence_plugin.py:42–44,82–90`; `scripts/audit/incident_register/acceptance.py:126–140`.

Prepending a temporary directory to `PYTHONPATH` does not make `-p w15_evidence_plugin` an absolute import: `python -m pytest` can resolve the module from the worktree cwd first. The recorded `plugin_sha256` is the runner's source file, not necessarily the loaded plugin. Session-start interpreter/cwd values and imported-module hashes are recorded but not bound to the expected environment/code by acceptance. `imports_under` only checks a path prefix, and an empty module map passes.

**Measured:** committed a synthetic worktree `w15_evidence_plugin.py` consisting of the original plugin plus a session-start wrapper that reports its own `__file__`. It preserves every outcome. The current-defence run was **accepted**, but its actual plugin was `<synthetic-wt>/w15_evidence_plugin.py`:

- recorded trusted plugin SHA256: `a415958ab1ebce81d97c361442e7ea9deebf206c2cad2478875f285240abedd4`;
- actual loaded shadow plugin SHA256: `203b104367de8067c98b266b9abd158243cb219c49e8f60c0c448e8d58afa303`.

No reports were forged in this probe. It demonstrates a false provenance claim and the ability of a worktree harness module to replace the observer. Also, every supplied synthetic runner test uses the outer test interpreter rather than an inner worktree venv; the implementation has no own-venv enforcement even though design §9.1 requires it for evidence runs.

**Smallest fix:** load the observer by a pinned absolute file through a trusted bootstrap and pass the loaded plugin explicitly to pytest; verify/report the loaded path and hash. Pin the executable/environment and expected imported module identities/hashes for each stage. Keep observer files outside reset/mutation scope and clean them in `finally`. Check the same identities on witness-only and restored runs. The in-process import fix closes the original foreign-`bts` case, but path-prefix checks are not a complete execution-identity gate.

### 7. SHOULD — E77's marked fixture makes successful delivery impossible to accept

**Locations:** `tests/test_incident_register_2026.py:519–545,561–584`; governing design §9.7–9.8.

Before calling `_oracle`, the marked test requires checks exactly at `18:10`, a started-game lock classification, and **every DM to be an alert**. Yet `_delivery_outcome` can return `delivered_before_cutoff` only if a **pick DM exists**. Thus no observation can satisfy both the preceding alert-only assertion and the required oracle value. A real repair that checks earlier may fail the check-time assertion even sooner. The separate correct-schedule positive execution control does not fix this logical contradiction in the marked node.

**Measured predicate probe:** supplied an observation satisfying all the existing check/classification assertions and carrying a valid pre-cutoff pick DM plus delivered pick flags. `_delivery_outcome` returned `("delivered_before_cutoff",)`, but the marked test raised ordinary `AssertionError` before reaching `_oracle`. This was an injected observation to isolate the predicate, **not** a claimed production repair. Separately, the real-source no-op mutant is correctly rejected, as reported above.

**Smallest fix:** prove actual execution, controlled inputs and observation completion on both branches. Apply the specific stale-schedule/check-at-first-pitch/containment-only assertions only when the observed outcome is the declared bad outcome, before raising its dedicated exception. Let a fully verified successful delivery reach the required-value branch and become `XPASS(strict)`. Permit the component's declared external/cascade response to support that path without presupposing the repair. Add a test of the marked node's fixed-behaviour direction.

### 8. BLOCKER — historical replay does not implement the amended acceptance contract

**Locations:** `scripts/audit/incident_register/evidence.py:420–426,445–475`; plan rev 2 `docs/superpowers/plans/2026-09-29-incident-register-phase1.md:193–199`.

`historical_replay_v2` writes **`replay_v2.json`**, not `acceptance.json`, and returns no `verdict`. It records import-check booleans and per-node classifications without rejecting failed import checks or run-wide errors. The supplied audit dictionary is used only to obtain a label and then discarded. A bare permitted decision string is enough for the clean label; no reason or adapter evidence is required or retained. The old script/selected-test automatic exemptions are fixed, but the review that authorizes the new label is not bound to the output.

**Measured:** ran a synthetic parent/fix replay with a changed test comment and an explicit audit entry `{decision: "neutral", reason: "only a comment; reviewed diff"}`. Output had two `symptom_candidate` nodes, clean `semantic_regression_replay`, and:

```json
{"verdict":null,"audit":null,"acceptance_exists":false}
```

Those `null` values mean the corresponding keys were absent. The real output file was `replay_v2.json`. Consequently, the plan's new rule “runner verdict is accepted AND reviewer decision is accept” cannot currently be satisfied for historical replay. Reading `imports_ok` and a clean label as substitutes would reopen false acceptance.

The runner still performs exactly F-tests / F^`src`; it does not accept a defective deployed closure, complete multi-commit fix set, or script/unit historical closure. That is acceptable as a limited experiment **if** the record explicitly preserves the distinction and unavailable cases. The new plan must not imply that the clean label alone supplies these facts.

**Smallest fix:** implement the promised historical acceptance object, or explicitly keep these results preliminary until a separate implemented acceptance command produces it. Include retained audit entries/reasons/adapter proof, exact raw report identities, observable actual/required assertion evidence, node inventory and session-wide rejection checks, plus deployed-ref/fix-set/closure disposition. Bind the contract sheet and reviewer decision to those hashes. Keep `symptom_candidate` as a classification requiring review; it is not a certificate verdict.

### 9. NIT — consolidate the amended plan and isolate nested pytest roots

**Locations:** plan `:13,33,41–45,73–118,154–158,178–184` versus `:186–206`; `tests/scripts/incident_register/test_acceptance.py:79–99`; `tests/scripts/incident_register/test_evidence_v2.py:68–96`; `scripts/audit/incident_register/evidence.py:247`.

The rev-2 appendix clearly improves the intended procedure, but the executable Tasks 5/6 still name legacy functions, Task 7 still declares branch history authoritative, and old counts remain 16 XFAIL / 27 pass. Rewrite those instructions in place or state explicitly that the rev-2 provisions supersede them. Use the actual 31 fixture nodes and 75 review-suite nodes rather than combining unrelated counts.

The nested synthetic repos and pytester cases also inherit an ancestor `pyproject.toml` if their temporary root is inside this checkout. Their hardcoded short node IDs then mismatch pytest's actual IDs. That explains the 11 failures in the qualified baseline attempt above; it is not evidence that the production fixtures failed. Give each synthetic pytest root its own minimal config or pass an explicit root, and verify the runner's intended root in its structured session evidence.

## False greens

| Passing test / assurance | Measured counterexample or disposition |
|---|---|
| `test_reset_worktree_removes_untracked_and_ignored` | Correct cleanup on a disposable tree, but no refusal on a primary checkout: unsaved synthetic main work is discarded. |
| Allowlist + clean restore | `../outside.txt` is changed and survives an **accepted** run; the Git touched-path list contains only the allowed production module. |
| “A mutant never touches tests” | Add the test to the caller's allowlist; change only its expected value; **accepted**. |
| `test_accepted_event_certificate_and_clean_restore` | Baseline collection error, unrelated mutant teardown error, or imperative-XFAIL killing baseline can coexist with **accepted**. |
| Exact oracle-location acceptance | `foreign_test_h.py::_oracle` satisfies a request for `test_h.py`; **accepted**, with no production call. |
| Absence + positive control | Fixture records `['pick']`; callee trace lacks a boundary; **accepted absence**. |
| Async paths unavailable | Actual `await` inside an async entry receives `certify.ok=True`. |
| Plugin from a temporary directory / pinned plugin SHA | Worktree module wins resolution; run **accepted** with the trusted source's SHA recorded for a different loaded observer. |
| Historical clean label + acceptance-object amendment | Clean label, audit reason discarded, no verdict and no `acceptance.json`. This is an output/acceptance gap, not a measured accepted historical verdict. |
| E77 r1 no-op | **Closed:** 2 failures, 1 oracle-control pass, no XFAIL. New issue is false rejection of the required success branch. |
| L01/L02 fix-direction controls | **Verified:** 17 XPASS(strict), 11 controls pass; changed saver or second-run outcome gets ordinary failure. |

No submitted test failure was required to obtain the accepted false-green results. The standard supplied suite remains 57/18 on the reviewed source. Reviewers can reject some of these cases by reading patches and frames; that is valuable, but it does not make a false runner verdict or witness claim correct.

## Answers

1. **r1 disposition:** all eight findings were revisited with code and measured reruns where executable. Finding 7 is resolved. The specific old no-op, root-conftest allowlist bypass, ordering/wrong-file witness, foreign-import, multiline-exception and simple wrong-location/phase-XFAIL cases are materially improved. The broader findings remain partial for the reasons in the table.
2. **v2 isolation and witness-only stage:** full in-tree restore in `finally` is an improvement. It needs a pre-mutation ownership/path gate, a hard frozen-harness boundary, actual executed-file/environment checks and complete-stage validation. Outcome equality cannot itself prove observational instrumentation. The allowlist is a caller assertion, not a protected boundary.
3. **Witness / absence / positive control:** useful synchronous trace evidence, insufficient for §9.3 certification. Record and join the complete observable interval and identity; reject unsupported execution modes. Do not promote `witness.ok` into “no event occurred.”
4. **Structured reports / expected failures:** original node IDs, exception classes and all phases fix the demonstrated JUnit losses. Wire acceptance into the runner, enforce the whole run, and use exact registered identities plus paired bad-signature evidence. The actual 18 fixture failures were independently checked and are not being rejected merely because the reusable acceptance API has defects.
5. **E77:** baseline now proves the intended stale-plan/check/classification sequence and rejects a no-op. `component` is the correct ceiling; mocks are explicitly listed, the idle wall clock is controlled by a declared mock, and a real positive execution control delivers with the corrected morning schedule. Fix the marked node's impossible success branch before claiming strict fixed-direction coverage.
6. **L01/L02:** the requested source, chronology, downstream-state and two-run repairs are present and mutation-checked. The temporary fix sketch only tests fixture sensitivity; it is not a reviewed production implementation. The conflicting historic suspended-game test remains unchanged.
7. **Plan amendments:** approve the **requirements** for diff-level negative review, retained SHA/install/canary/rollback evidence with unknown intervals, and a checked record schema. The proposed reviewer accept/reject step is also necessary. They do not cure missing or permissive machine acceptance. Complete findings 1–6 and 8 before rerunning certificates; fix E77's success branch and consolidate the task instructions. Preserve Route R / X-20 and owner authorization boundaries. No merge, commit, push, deployment, production read or memory update is authorized by this review.

**Final verification:** after all mutants were restored, the standard review suite again returned **57 passed, 18 xfailed**; both tracked and staged diffs in `wt2` were empty. The authorized `wt2` directory and its Git worktree registration have been removed. Main remains at `d8563a72c55ade873de5111347d1e7eef3f4bba0` with empty `git status --short`; other worktrees were left alone. This report is the retained review deliverable.

DONE
