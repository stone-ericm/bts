## Verdict

**BLOCK.** The submitted fixtures reproduce the stated baseline values, but the evidence machinery does not yet establish the certificates the plan promises. The blocking counterexamples are executable:

- Replacing `run_day` with an immediate return leaves E77 at **1 passed, 1 xfailed**.
- A mutant that adds only root `conftest.py`, leaving production source unchanged, produces green/red/green, unchanged test hashes, and an accepted `connected` witness. The conftest survives the purported restoration.
- The import check can report the intended worktree while pytest actually imports a different `bts` package.
- The expected-failure acceptance helper accepts an unrelated call-phase XFAIL, imperative `pytest.xfail`, and the call report of a node whose teardown failed.
- Changed scripts and helpers in a selected test file can receive the supposedly audited historical-replay label without an audit.

**Pins:** plan `04edfac175d589ea1903674468271917beecbcd2`; code `3f6fd63caec40023cf6782679bfa7c6afc20aaca`; diff base/design edits `04267b66de539e9f5a01ff9b197ee97dc0df78c1`. **P:line** denotes `docs/superpowers/plans/2026-09-29-incident-register-phase1.md`; **D:line** denotes the governing design v3.1. Code citations refer to `3f6fd63`, including production source unchanged by that commit.

**Measured locally:** the four submitted test files collected **43 nodes: 27 passed, 16 xfailed**, both before adversarial mutations and after restoring the production source. `--runxfail` on `tests/test_incident_register_2026.py` gave **16 failed, 10 passed**; the sixteen failures were the dedicated exceptions at the expected call locations. A temporary AB=0 grading mutation that ignored sacrifice flies was rejected by **both** SF-only controls. Eleven additional rejection checks exposed the problems described below; the stronger conftest/witness combination was then checked separately. These were synthetic tests under Python **3.12.13**, pytest **9.0.2**, with the review worktree's own venv, `UV_CACHE_DIR=/tmp/uv-cache`, `TZ=America/New_York` and offline uv operation. This does not independently verify P:14's full-suite count or the historical pilot's results.

No production/runtime data, SSH, network or herdr access was used. Main's tracked files were not edited. Temporary probes and mutations were confined to the authorized review worktree and its test scratch directories. No project commit was created; the submitted harness tests create synthetic commits in their throwaway repositories.

## Findings

### 1. BLOCKER — E77 accepts “the scheduler never ran” as the singleton mechanism

**Locations:** `tests/test_incident_register_2026.py:356–401,415–439`; P:43–46,75–80,182–184; D:136,158.

**Verified baseline:** the real fixture prints the stale 19:10 schedule, the lone 18:10 check, `game_started_or_final`, classification lock, then a missed-pick alert at 19:00. That is useful component evidence. Its marker is also narrower than `AssertionError`.

**Measured counterexample:** in the authorized worktree, I inserted `return None` as the first executable statement of `src/bts/scheduler.py::run_day`, leaving the test file unchanged, and ran:

```text
uv run pytest tests/test_incident_register_2026.py -k e77 -q -rx
1 passed, 24 deselected, 1 xfailed; exit 0
```

The fixture prewrites the preview before calling `run_day`; the oracle sees that unchanged preview and raises `SingletonSlateUndelivered`. The “positive” control prewrites an already-delivered pick, so it also passes when no scheduler code executes. Neither test requires that the clock reached cutoff, the scheduled check ran, the game was classified, or the boundary recorder covered the interval. A no-games early return or equivalent fixture mistake can therefore retain the same XFAIL without the historical mechanism.

**False greens:** `test_e77_singleton_slate_moved_up_is_delivered_before_cutoff` and `test_e77_oracle_control_delivered_before_cutoff` together.

**Smallest fix:** before allowing the dedicated exception, use ordinary assertions and observational spies to prove the declared sequence and input identities: morning plan at 19:10, actual cutoff 18:05, the 18:10 check, classification of the undelivered selection as `game_started_or_final`, and completion of the bounded observation interval. Reject “no entry/check/classifier execution” as a fixture/path failure. Keep the already-delivered test as an **oracle-only** control; add a positive execution control in which this fixture actually sends a pick before cutoff. That control can remain component-level and must identify its mocks.

Also verify delivery flags/receipt identity together with `delivered_at` and the recorded pick-send event. The current success branch accepts any non-null pre-cutoff timestamp even if `notification_sent` is false, while the failure branch labels every recorded DM containment without checking its kind. An alert is containment only after identifying it as an alert.

### 2. BLOCKER — `current_defence` admits harness-only mutants and does not restore the full baseline

**Locations:** `scripts/audit/incident_register/evidence.py:185–212`; `tests/scripts/incident_register/test_evidence.py:129–145`; P:34,87–90,114–118.

The patch guard checks only paths beginning `tests/` (`:196–198`). Root `conftest.py`, pytest configuration, imported helpers elsewhere and environment-controlling files remain writable by the mutant. Only explicitly selected test files are hashed. Restoration removes the witness file and replaces `src/`; it does not reverse other patch paths. There is no `try/finally`, so exceptions after application also leave the worktree mutated.

**Measured counterexample:** using the submitted synthetic-repository fixture, I supplied a patch adding only this root `conftest.py`:

```python
import os

def grade():
    from bts._w15_witness import hook
    hook("entry")
    hook("branch")
    hook("boundary")
    return "miss"

def pytest_configure(config):
    if os.environ.get("W15_WITNESS_PATH"):
        from bts import mod
        mod.grade = grade
```

Production `src/bts/mod.py` remained fixed and returned `void`. The result was:

```text
green_before.exit = 0
mutant.exit = 1; phase = call; AssertionError: 'miss' != 'void'
green_after.exit = 0
tests_unchanged_after_mutant = true
witness_records = 3
root conftest survives restore = true
connected(... entry="grade", branch_tag="branch", boundary_tag="boundary").ok = true
```

This passes every acceptance assertion made by `test_current_defence_red_then_restored`; the direct `tests/` patch test also remains green. A manual patch review should reject it, but the plan explicitly claims that the harness refuses mutations of the test harness and restores the baseline. Those claims are false.

**Smallest fix:** require a per-incident allowlist of semantic mutation paths from the production command/import closure, and reject all harness/config/import-overlay changes outside it before any test run. Pin the complete executed harness, including root/nested conftests and imported helpers. Parse changed paths safely rather than splitting human-formatted `--numstat` on whitespace. Restore **every** touched/created path in `finally`, including scripts/config where intentionally allowed; verify the whole permitted execution closure equals the baseline before the second green. Treat a changed hash as rejection, not merely a boolean in JSON. Pin the copied witness module's source/hash too.

The green/red/green certificate must explicitly reject unsuccessful green runs, missing nodes, unexpected errors and unsuccessful restoration. Capturing those results is useful; returning a result dictionary is not acceptance.

### 3. BLOCKER — `connected` proves tag co-membership, not the complete causal/absence certificate

**Locations:** `scripts/audit/incident_register/witness.py:23–36,54–91`; `tests/scripts/incident_register/test_witness.py:55–86`; P:75–80,115–118; D:126.

**Measured counterexamples:**

1. Real hooks emitted `entry → boundary → branch`. `connected(... boundary_tag="boundary")` returned `ok=True`. It does not enforce the “mutated decision and then boundary” ordering in its own docstring.
2. Real hooks emitted `entry → branch → boundary → done`, and a boundary recorder contained `['sent']`. Calling `connected(... completion_tag="done")` returned `ok=True`. There is no recorder input, interval, clock endpoint, task completion or no-event condition. A completion tag alone is not the bounded absence certificate required by D:126.
3. Finding 2's replacement `grade` in root conftest was accepted as the production entry. The trace contains filenames, but `_entry_frames` discards them and the initial hook checks only the qualname. No declared production file/code identity is verified.

**What is sound:** for a synchronous invocation whose entry hook is genuinely unconditional and first, exactly one entry record rejects the original two-invocation/frame-reuse example. The existing disconnected-direct-call test is meaningful. Setup-tagged records are excluded by `:72`. I did not establish a same-process frame-id collision that defeats those assumptions; do not substitute a speculative collision for the measured failures above.

**Remaining limits:** `PYTEST_CURRENT_TEST` is a process-wide environment value, and the trace has no run/process/thread/task identity or date/selection key. Ordinary `f_back` stacks also do not reliably encode suspended asyncio ancestry. This can conservatively reject legitimate asynchronous paths; raw frame ids cannot be treated as durable cross-run/process identifiers.

**False greens:** `test_connected_boundary_within_one_invocation` does not test order or source identity; `test_connected_completion_for_a_missing_event` tests a completion tag, without validating bounded absence. P:78 overstates the latter as the absence witness.

**Smallest fix:** bind the entry to expected module/path/code identity and allocate an invocation token at entry; record a run id, process/thread/task identity as applicable, sequence number and checked selection/date. Require ordered branch and boundary/completion records in the declared invocation. Keep `connected` explicitly as a reachability check, or extend it with a separate acceptance record proving observation start/end, controlled deadline, completion of relevant tasks, and the boundary event set. Missing-event acceptance must check that set is empty for the required identity and interval, paired with a baseline positive event. Declare unsupported asynchronous/process paths `unavailable` until they have a tested propagation scheme.

### 4. BLOCKER — the recorded import and JUnit summary do not identify the execution being certified

**Locations:** `scripts/audit/incident_register/evidence.py:60–92,112–122,169–181`; P:29–33,84–86,95–104.

**Measured import mismatch:** I ran a synthetic test with a root conftest that put another synthetic `bts` package first on `sys.path`. The test recorded the package it actually imported. `run_pytest` returned the normal worktree path from its separate `python -c` process, while the test recorded the foreign package:

```text
reported imported_bts: <scratch>/wt/src/bts/__init__.py
actual pytest import:  <scratch>/foreign/bts/__init__.py
```

`run_pytest` neither observes imports inside pytest nor rejects an unexpected environment/path. Passing `uv run python` is not enough to establish the module code actually used; conftest, `PYTHONPATH`, preloaded modules and environment selection can change it.

**JUnit limitations, verified:**

- Keys are reconstructed as `classname::name`; a real `test_case.py::test_x` became a dotted path without `.py`. Witness records use pytest's original slash-separated node id. There is no canonical join or mapping in this code. The submitted tests compare only the final function-name suffix, hiding this mismatch.
- `parse_junit` stores a single result per reconstructed key, breaks after the first failure/error/skipped child, and overwrites duplicate testcases. It cannot preserve the full setup/call/teardown sequence needed for acceptance.
- A skipped/XFAIL result loses its phase and uses the XML skip type as `exc_type` (`:75–76`). It does not record the actual exception or oracle location.
- A real `TypeError("first\nE   AssertionError: last")` was parsed with `exc_type="AssertionError"`. `classify` still returned `new_api` in **this** probe because the message retained `TypeError`; this is corrupt exception metadata, not a demonstrated false symptom classification. Conversely its substring search can reject a genuine observable assertion merely because its message mentions a type name.
- Pytest 9.0.2's ordinary setup/teardown prefixes do match the parser's strings (`_pytest/junitxml.py:217–229`). That limited compatibility does not turn JUnit into a lossless phase report.

**False greens:** `test_historical_replay_classifies_nodes`, `test_classify_rules`, and `test_current_defence_red_then_restored` establish simple cases, not the phase/node/import guarantees claimed by the plan.

**Smallest fix:** install an observational pytest reporting plugin for these evidence runs. Emit original node ids, collection counts, every phase, actual exception type, traceback/oracle location, xfail metadata and actual/required values where registered. Capture imported module paths/hashes and interpreter identity **inside that same pytest process**; reject unexpected paths/venvs before accepting evidence. Pin all executed harness files/config/clock inputs, not just selected file names and `uv.lock`. Keep stdout/stderr and JUnit as supporting artifacts. Reject duplicate/missing nodes and error-bearing node aggregates; never infer exact exception classes or call phases from free-form messages.

### 5. BLOCKER — the expected-failure acceptance rule is incomplete and is not wired into the evidence runner

**Locations:** `tests/test_incident_register_2026_meta.py:1–6,66–68,80–85,88–128`; `scripts/audit/incident_register/evidence.py:60–122`; P:59–71; D:134.

The submitted meta-tests correctly show that pytest converts the dedicated exception in setup to XFAIL and that a phase check rejects it. But `accepted_as_reproduction` checks only `when == call`, `outcome == skipped`, and nonempty `wasxfail`.

**Measured with actual pytest 9.0.2 reports:** it returned `True` for both:

```python
@pytest.mark.xfail(strict=True, raises=Dedicated, reason="unrelated")
def test_wrong_location():
    raise Dedicated("fixture body failed before any production invocation")

@pytest.mark.xfail(strict=True, raises=Dedicated, reason="unrelated")
def test_imperative():
    pytest.xfail("no oracle")
```

It also returned `True` for the call XFAIL in a node whose teardown raised unrelated `ValueError`; pytest's real reports were `(setup, passed)`, `(call, skipped/XFAIL)`, `(teardown, failed)` and the run summarized **1 xfailed, 1 error**. The current `_final` helper would select the earlier XFAIL and does not aggregate the later failure.

The function appears only in the meta-test file and its own assertions; there is no use in `scripts/audit/incident_register/`. The module docstring's statement that this is the rule “the evidence runner applies” is therefore unsupported. A `--runxfail` validation that merely looks for the word `Dedicated` also does not establish that it came from the registered oracle after production execution.

**False greens:** `test_meta_outcomes_under_the_marker`, `test_meta_summary_counts` and `test_meta_runxfail_shows_the_dedicated_exception_in_call` omit those rejection cases. The fixed six-case counts do not test the governing acceptance predicate.

**Smallest fix:** move acceptance into the evidence library, using the structured reports from finding 4. Accept only a registered node/parameter, declared exception class, designated oracle location, recorded bad signature and validated production invocation. Pair the marked and `--runxfail` runs by node and content/input hashes. Reject imperative xfail, a matching class from another location, missing reports, and any setup/teardown/collection failure for the accepted run. Add teardown cases and wrong-location/imperative cases to the meta-tests; count accepted nodes independently of pytest's XFAIL summary.

### 6. BLOCKER — the historical harness audit can call changed execution inputs “audited”

**Locations:** `scripts/audit/incident_register/evidence.py:125–156,159–181`; `tests/scripts/incident_register/test_evidence.py:82–101`; P:86,95–104; D:122–124.

`harness_changes` lists scripts as `script` and most configuration as `other`, but `Replay.label_kind` considers neither risky. A changed selected `.py` test file is also automatically exempt when its whole path appears in `self.tests`, even though it can contain fixtures, autouse setup, helper functions or clock changes. Selecting a node instead of its file changes that comparison, making the same audit depend on CLI selection syntax.

**Measured computations:** both of these unaudited changes return `semantic_regression_replay`:

```python
Replay("x", "f", "p", ["tests/test_mod.py"], harness=[
    {"status": "M", "path": "scripts/entry.py", "kind": "script"},
]).label_kind

Replay("x", "f", "p", ["tests/test_mod.py"], harness=[
    {"status": "M", "path": "tests/test_mod.py", "kind": "test"},
]).label_kind
```

P:104 then permits a manually classified `symptom` node under that label to count as historical replay. The code has no recorded review decision showing that the script/helper change preserved the old input and call semantics. Root configs, unit files and rename endpoints have the same problem; `status, path = line.partition(...)` does not normalize both paths of a rename.

`historical_replay` always selects `fix^`, changes only `src/`, and records only selected-test/lock hashes. It cannot by itself identify the defective deployed closure, reconstruct a script/unit defect, or apply a complete multi-commit fix set. Those are correctly required by D:122–124 and must remain separate evidence rather than follow from a red assertion.

**False greens:** `test_harness_audit_flags_conftest_changes` only tests two classified paths; `test_historical_replay_classifies_nodes` positively asserts the clean label while the selected test file changed wholesale.

**Smallest fix:** default relevant outside-`src` changes to unaudited, including changes in selected files. Normalize node selectors to file paths and parse status/rename records unambiguously. Permit the clean label only after a saved audit decision identifies each change as neutral, irrelevant to the executed closure, or handled by an input-preserving adapter. A changed fixture in the selected test must never be exempt solely because the file was selected. Pin the deployed ref, complete fix set and actual command/import closure; use a neutral overlay on that ref where possible. Otherwise retain `semantic_regression_replay` as the limited experiment and leave historical equivalence unavailable.

### 7. SHOULD — complete the L01 input/coverage contract and the repeated-run evidence

**Locations:** `tests/test_incident_register_2026.py:71–137,181–184,200–248,277–322,388–401`; P:41,45–51; D:143–158.

Several positive results are real: the single-pick/Hit+Pass fixtures call Click's actual `check-results` command; they set the CLI clock to the next day and plant delivery evidence, so the baseline runs do not merely exit through stale-scoring or scoreable gates. A skipped grading pass would leave streak 5, which cannot masquerade as the declared bad 0 in these XFAIL nodes. The SF-only controls both failed under the measured over-broad AB=0 mutation. L02's baseline bad tuples were also verified, after the no-fetch/no-file-change assertions.

Remaining gaps:

1. **DNP is encoded as missing stats.** `did_not_play` uses `batting=None`, and `_feed` writes `stats.batting = {}` (`:94–98,114`). That does not positively establish zero AB/SF/PA under D:154's “incomplete evidence is characterization-only” rule. Current `_boxscore_hit` defaults missing hits to zero, so this fixture cannot distinguish DNP from incomplete stats. The same source feeds three DNP-marked nodes.
2. **Suspended chronology is inconsistent in the CLI fixture.** The feed is Final with resumed play at **2026-06-11 17:30Z**, but the CLI clock is **2026-06-11 01:00 ET = 05:00Z** (`:72,76–77,100`). It consumes future final evidence. Use an after-resumption clock still inside the two-day stale guard, or a distinct faithful pre-resumption case. This is a fixture chronology error, not rule ambiguity.
3. **Coverage is short of “every case through both paths.”** `official_ab_no_hit` and `pre_suspension_hit` run only at the direct grader level. There is no Pass+Pass DD case, no downstream assertion that an all-Pass result preserves saver state, and no saver-eligible starting state. The marked downstream oracles check only streak 0 versus 5/6, without requiring the declared per-slot bad grades or the HTTP/path witness. Complete those state/signature checks before the dedicated exception.
4. **L02 does not actually perform two bad baseline runs.** `_oracle` raises inside the loop at run 1 (`:281–287`); both unmarked failures confirm `reconcile run 1`. Collect and validate both observations, rejecting an unexpected second-run shape with an ordinary assertion, then raise the dedicated class at the final oracle. Retain the original expected state for each run. The preview control also carries delivery flags from `_delivered_daily`; add a genuinely undelivered/unplayed preview if that is the exclusion being claimed.
5. **E77 does not control every clock.** Its baseline emitted the current September wall time from `_idle_until_next_wakeup`, which calls `datetime.now(UTC)` directly (`src/bts/scheduler.py:1246–1249`). The test patches `_now_et`, not this clock. Either control this dependency or declare/mock the post-observation idle boundary; do not claim the whole run has a pinned clock.

**Smallest fix:** use explicit zero stats and complete positive DNP evidence; parameterize an as-of clock consistent with every returned feed; complete the downstream/control matrix and assert `(slot_results, day_result, streak, saver)` plus the requested game/HTTP calls. For pending controls, require that the resolver was reached and preserve all state. Gather both reconcile observations before the final expected-failure oracle. Keep the old conflicting suspension test unchanged pending a separately authorized production repair.

The CLI fixture is the real command entry, but it does not execute the shell cron wrapper, `flock`, or `--wait-deadline-et 06:00` (`scripts/cron-setup-hetzner.sh:53`). Describe it as real CLI grading coverage; do not claim cron installation/wait-loop coverage from it.

### 8. SHOULD — Tasks 5–8 need an explicit evidence-to-record acceptance step

**Locations:** P:94–118,120–176,180–184; D:22,71,116,122–134.

The Phase 1/Phase 2 split is reasonable. Reviewing the built code and plan together is also a reasonable disclosed deviation; it does not excuse the false certificate paths above. The priority order puts plan-named incidents first, with E77 already assigned a fixture and L03/L04 explicitly pending. `component` is the correct E77 ceiling: its mocks include `count_new_confirmations`, `run_and_pick`, result polling and capture triggering, not only HTTP boundaries. Alert-versus-pick delivery is the right distinction once observed event identities establish it. Exactly one entry is a usable synchronous limitation, not a complete certificate.

**Remaining procedure gaps:**

- P:154 reviews every runtime-closure negative by subject and cue scan, then reads full bodies only for cue-bearing commits. That repeats the keyword-selection weakness the design removed. A negative with a bland message can contain the relevant code/config change. Review the changed execution closure/diff for every required negative, with an explicit exclusion reason; subject/body review alone is not that review.
- P:158 treats deploy-branch history as authoritative deployment history. Git ancestry establishes the candidate code, not successful installation, activation or recovery. The workflow can install a target and later roll it back (`.github/workflows/deploy.yml:89–113,159–177`); even `Deployed ...` precedes canary success. Bind deploy/rollback intervals to retained run logs or qualified operator evidence and exact SHAs. Preserve unknown intervals if the repo-only evidence cannot establish them. A branch switch date does not remove this distinction.
- P:95/104/116–118 needs a recorded acceptance object connecting the contract sheet, raw run reports, expected node inventory, declared defect signature, witness, unchanged harness, restored baseline, certificate level and reviewer decision. A `symptom_candidate` plus a clean-looking label is not enough. Unexpected collection/setup/teardown errors and missing nodes must not disappear when records are folded into episodes.
- P:160 and P:163 correctly reference evidence kinds/strength and design §7. Enforce that reference in a checked record schema: observations and their occurrence bounds, defective deployed interval, fix/deploy/mitigation/recovery as separate fields, residual gaps, and explicit unavailable historical/current certificates. Neither the unavailable pilot files nor the quoted candidate/deploy counts were independently verified in this review. Attach their pinned source inventory and counting procedure at result review.

**Smallest plan edits:** add the above acceptance object to Tasks 5/6; replace Task 7's subject/cue-only review with full relevant-negative diff review; replace branch-history authority with SHA-bound observed deployment intervals; validate all Phase 1 records against design §7 before publication. Keep Route R pending explicit. The 20% rerun sample is QC, while every certificate still needs its own complete accepted evidence. Preserve the existing owner authorization requirements for later merge/commit/push/memory work; this review authorizes none of those actions.

## False greens

| Listed test / claim | Counterexample or missing discriminator | Disposition |
|---|---|---|
| E77 failure + oracle-control pair | `run_day` is an immediate no-op; still 1 pass + 1 XFAIL | **Measured false green**, finding 1 |
| `test_current_defence_red_then_restored` / tests-only patch guard | Root conftest replaces `grade` only on the mutant run; source unchanged, green/red/green, hashes true, conftest survives | **Measured false green**, finding 2 |
| `test_connected_boundary_within_one_invocation` | Boundary precedes branch; root-conftest function substitutes for declared production entry | **Measured accepted witnesses**, finding 3 |
| `test_connected_completion_for_a_missing_event` | A qualifying send occurred; completion check still returns true | **Measured insufficiency as an absence certificate**, finding 3 |
| `test_historical_replay_classifies_nodes` import assertion | Separate import probe names worktree; pytest uses foreign package | **Measured false identity evidence**, finding 4 |
| Simple JUnit/classifier tests | Original node ids lost; multiline exception message changes parsed exception class | **Measured metadata failures**, finding 4; the TypeError probe still classified `new_api` |
| Meta outcome/count tests | Wrong-location class and imperative xfail accepted; teardown failure ignored by call-only acceptance | **Measured false acceptance**, finding 5 |
| Harness audit tests | Script change / helper-bearing selected test change receives clean label | **Measured false audit label**, finding 6 |
| L01 DNP / downstream coverage claims | Empty batting stats stand in for DNP; DD all-Pass/saver and two direct controls lack the required downstream coverage | **Source-verified gaps**, finding 7 |

A green supplied suite is compatible with all of the measured failures in this table. The baseline restoration run reconfirmed the original 27/16 result; it did not erase the adversarial findings.

## Answers

1. **Fixtures / §9.7 / §10:** The baseline L01/L02 failures reached the declared call-phase helpers, and stale/scoreable gates do not explain their bad values. SF-only controls are meaningful, verified by an over-correction mutation. DNP input, suspended chronology, complete downstream state/signature coverage and actual repeated L02 execution need finding 7's fixes. `check-results` is real CLI grading, with HTTP stubbed; the full production cron wrapper is outside this fixture.
2. **E77:** Baseline source execution follows the intended static-schedule/classification sequence, and `component` is honest. Its acceptance is not mechanism-specific: a no-op scheduler retains the XFAIL and positive control. Prove the sequence and observation interval before raising the dedicated exception, and validate pick delivery separately from identified containment alerts. The real wall-clock idle branch also needs explicit treatment.
3. **Meta-tests:** Their six original pytest behaviors and counts are correct. They do not implement the design's full acceptance rule, omit teardown rejection, and are not integrated with the runner. Findings 4–5 require exact exception/location/signature, aggregate phase validation and registered node matching.
4. **`witness.connected`:** It rejects the tested direct-call and two-entry cases and ignores setup-tagged records. It accepts reversed order and a wrong-file entry and cannot certify bounded absence from `done`. Bind source/invocation/event identity and require a full boundary recorder; declare async/process limits rather than assuming raw stack ids carry those relationships.
5. **`evidence.py`:** `use_src`'s wholesale replacement works in the supplied no-new-module-leak test. The surrounding certificate machinery is blocked: selected-test hashes and a `tests/` prefix are insufficient; restoration is partial; the import probe observes another process; JUnit is lossy; audit labels exempt risky changes. `classify` is only a preliminary classifier and must never confer acceptance. Add structured evidence, full closure pins and exception-safe restoration.
6. **Plan:** The phase split, disclosed build-before-plan review, initial priorities and explicit deferred candidates are acceptable. Tasks 5–8 need the corrected tooling plus full required-negative review, observed deployment intervals and a validated acceptance-to-record join. Counts, pilot evidence and recovery claims remain pending result review; branch ancestry and a passing test summary cannot supply them.

**Final verification:** the authorized `wt` directory and its Git worktree registration were removed, including the temporary probes. Main remains at `04edfac175d589ea1903674468271917beecbcd2` with an empty tracked diff; other worktrees were left alone. The report is the only retained review artifact.

DONE
