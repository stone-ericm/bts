## Verdict

**BLOCK.** The r3 replay, failure-artifact and several observer defects received effective repairs. The revised harness still accepts false absence and false production connections, executes application code while observing, misses executable environment changes, and permits reports to establish occurrences they did not witness. These require implementation changes and new evidence runs.

Reviewed plan rev 4 and design v3.2 at **f6b3d0ff8ee0cc1701325912b6c285864a1fb959**; code in the authorized detached worktree at **74b04e8**. Verified that tooling and fixture bytes are unchanged between 27b260d and 74b04e8, and between 0166e13 and 27b260d. Source line references below refer to 74b04e8; plan/design references refer to f6b3d0f.

Measured, offline, with Python 3.12.13, pytest 9.0.2, jsonschema 4.23.0 and TZ=America/New_York:

| Check | Result |
|---|---|
| Submitted tooling + incident fixtures + meta-tests | **198 passed, 22 xfailed** |
| Eleven adapted independent r3 probes | **11 passed**; assertions require the current accept/reject results described below |
| Nineteen further reviewer probes | **19 passed**; several deliberately assert a false acceptance, so this is evidence of defects, not a clean bill of health |
| Final combined run after restoring all source mutants | **228 passed, 22 xfailed**, exit 0 |
| Actual expected-failure driver, using a synthetic repository containing the pinned source, fixtures, configuration, lock, .gitignore and registry, with its own venv | **accepted, 22/22**; **20 connected, 2 exception_shape**; marked exit 0, unmarked exit 1, no session reasons |
| Eight targeted submitted sweep mutants | O2, D9, W3, W4, R14, R15, D11 and P7 each fail their named regression; the eight-test unmutated baseline passes |
| Completion-guard-only mutants | Removing `completed and` leaves the defence and replay unexpected-exception regressions passing; the reason-recording path still rejects |
| Consolidated drafts | **115**, zero draft-mode errors; disposition/tier totals match the plan; **92** log-basis fix entries |

The actual pair reproduces the submitted result. It does not establish the soundness of the reusable acceptance predicates. I did not rerun the full expensive regression or all 75 whole-suite mutation runs. The retained sweep was inspected and eight mutations independently exercised.

Reproducers, exact mutation scripts, logs, actual pair events and SHA256 inventory are retained at **/private/tmp/bts-incident-register-r4-3h3po_p8/**. The five `test_review_r4_*.py` files contain the 30 independent probes. The first synthetic real-pair preparation omitted the project's .gitignore and correctly rejected generated-file drift; `real-pair-v2.log` and `real-pair-output/` are the corrected, accepted preparation. No production data was read.

## r3 findings status

| r3 item | Status | Measured rerun and remaining limit |
|---|---|---|
| **#1 Incomplete absence recorder / unfinished interval** | **PARTIAL** | The original `map(mock_send, ...)` counterexample now rejects: the actual send is recorded. The original worker created during the call and left alive until teardown also rejects. A joined worker is recorded and rejects absence. New accepted counterexamples use an initial C boundary, temporary C-invoked rebinding, and a worker already alive when observation starts. Finding 1. |
| **#2 Observer invokes application code** | **PARTIAL** | The original side-effecting `repr` probe now fails both with and without observation; `repr` is not invoked. New exact-dict serialization invokes application `__eq__`. Lossy serialization also falsely connects a correct production return to a different oracle value. Finding 2. |
| **#3 Replay harness drift / red teardown error** | **RESOLVED** | Independently reran both old cases: the green-stage helper edit with equal production trees rejects, and the unrelated red teardown error rejects. Drift is checked before source swap/reset, and the red stage uses the full mutant phase gate. |
| **#4 Pair closure / disconnected literals / environment** | **PARTIAL** | The old unused correct production return plus literal bad oracle now rejects. Ordinary package-byte changes now change the environment fingerprint. The old low-level `accept_pair` probe remains permissive if callers run their own changing pair, but `run_pair` is now the documented freeze owner and its submitted helper-drift regression passes. New public `run_pair` counterexamples falsely connect truncated returns and unrelated readback; dependency symlinks, relative .pth targets and bytecode evade the fingerprint. Findings 2–4. |
| **#5 Exact installation time inferred from deployment interpolation** | **PARTIAL** | The original exact-time issue is repaired. For 1b50b78, retained points now give **(2026-08-14T19:04:33Z, 19:04:35Z]**. Timeline assumptions are explicitly labelled. Synthetic out-of-order runs and contradictory rollback output still produce incorrect observation claims; these cases are absent from the retained corpus. Finding 9. |
| **#6 Unsupported certification / inconsistent records** | **PARTIAL** | The four old invalid-record cases are rejected by the submitted executable r3 regression: empty certified replay, bare log basis, reversed/unbounded latency, and pre-event report. Publication hash/verdict binding tests pass. New repair-only occurrence promotion, cross-occurrence/step qualification, missing expected-failure artifact and impossible latency cases validate with zero errors. Findings 5–6. |
| **#7 Dynamic DISABLE / recursive invocation linking** | **RESOLVED** | The warmed dynamically rebound Python helper is now recorded without interval errors. The real recursive outer-branch/inner-boundary case is accepted through the common outer invocation. Actual frame-address reuse across separate invocations still rejects. Wrong receiver identity is a distinct new regression in finding 1. |
| **#8 Pair-driver failure artifact / cleanup** | **RESOLVED** | The submitted executable driver-failure regression passes; the driver initializes a rejected result, restores in finally, and writes a resolved ref, registry hash and overall verdict. My actual successful driver rerun also writes the expected artifact and destroys its synthetic evidence worktree. |
| **Sweep classification** | **PARTIAL** | Whole-suite runs, explicit ERROR rejection and handling genuine pytest `DID NOT RAISE` are improvements. The eight selected mutants are independently killed. The summary-string classifier nevertheless calls an ordinary `RuntimeError("DID NOT RAISE")` a kill. Finding 10. |
| **Ruling 2 / design amendment** | **RESOLVED** for the requested text | Design §9.3 incorporates the supplied witness, observational, full-interval absence and coverage requirements. This resolves the missing amendment; findings 1–4 show that implementation compliance remains blocked. |

## New findings

### 1. BLOCKER — absence and event certificates still misidentify the event set

**Locations:** `scripts/audit/incident_register/observer.py:293,339,347,372,393,427,475`; `certify.py:31,105`; plan ruling 2 and Task 3.

Four independent, executed current-defence probes return **verdict=accepted**, empty reasons and `certificate.ok=true`:

1. **Initial unsupported C boundary.** The baseline calls a Python `send(text)`. The source mutant binds `send = sent.append` at import time and changes the declared return to `defer`. The frozen test verifies `sent == ['pick: Turner']` before failing its declared return assertion. `_register` emits `boundary_gap` **before** `obs_start`; `certify.interval` excludes that event. `_on_call` finds the already-seen target and does not emit another gap. The accepted certificate reports zero calls, even while its coverage list says C-implemented boundaries are unsupported.
2. **Temporary rebinding invoked from C.** Production saves the original binding, rebinds it to a second Python send function, executes `list(map(transport.send, ['eric'], ['pick: Turner']))`, restores the binding and returns `defer`. The frozen test verifies the send occurred. The replacement code was never registered through a direct production CALL, and the restored final binding hides the coverage gap. Accepted absence again reports zero calls.
3. **Already-running worker.** Fixture setup starts a worker waiting on an Event. The mutant queues delivery and returns; fixture teardown releases and joins the worker and verifies exactly one send. Because its thread identity was present in `threads_at_start`, `obs_end.outstanding_threads` is zero. Invocation return is mistaken for completion of relevant work. Accepted absence excludes the later send. Both observer-off stages pass their gates and conformance comparisons.
4. **Wrong receiver.** The declared boundary is `transport.real.send`; the mutant calls `transport.other.send`, whose method shares the same code object. The frozen assertion that the real receiver sent the pick fails. V4 accepts an event certificate for the declared boundary. V3 rejected this counterexample. `boundary_codes` records code-to-spec, without distinguishing receiver identity.

These are actual runner acceptances, not fabricated event tapes. The first three share a passed baseline positive control, an observed mutation, an ordinary failure at the declared assertion, a restored green and the false absence result.

**Smallest fix:** include registration-time coverage failures in the node's certification state; any required unsupported binding must make absence unavailable. Bind boundary identity to the actual callable/receiver as well as code; ambiguous shared-code identities must be unavailable. Either establish coverage across temporary aliases/C dispatch or refuse the execution mode. Close the contract interval only after relevant queued work on both new and existing workers has completed, been cancelled, or reached the declared controlled deadline. Counting newly created live threads is insufficient. Keep the passing direct-C, joined-worker, recursion and rebinding controls.

### 2. BLOCKER — `_safe` can run application code and silently change the value being certified

**Locations:** `observer.py:118–137`; `acceptance.py:52–64,87–93`; `defence.py:89–101`.

**Application callback, measured:** an exact builtin dict contains a custom key whose hash collides with the later string key `"x"`. Its `__eq__` records calls and returns false. After dictionary construction, clear that record and call `_safe`. Copying the string entry with `dict.__getitem__(value, "x")` invokes the custom key's `__eq__` once. The result is `{'map': {'x': 2}}`; obtaining it has executed application code. Exact-container type checks do not make key lookup observational. The same lookup pattern is used for instance fields.

**False production connection, measured:** production `grade()` returns the correct **51-element** list `[0] * 51`. The frozen test ignores that return and gives `_oracle` the wrong **50-element** list, with required=51 elements and declared bad=50. `_safe` retains only 50 elements; `unwrap` treats that prefix as complete. `run_pair` accepts and marks the node **connected**, with no session or per-node reason. The actual production return was correct.

Strings are similarly cut at 500 characters and maps/fields at 50 entries without a completeness marker. A bounded display representation must not become an equality witness for the full value.

**Smallest fix:** traverse builtin dict items directly, without looking keys up again; apply the same rule to raw instance dictionaries. Carry an explicit incomplete/unavailable state for every truncation, omitted field relevant to comparison, unsupported value and depth limit, and refuse semantic equality/identity classification on it. Use complete values only where safely bounded by the contract. Add the collision-key and 50/51-element pair probes. Strengthen observer-off conformance beyond the coarse pass/fail state where symptom equality is claimed: two different failures are both currently `failed`. The direct callback probe establishes the serializer defect; I have not claimed a full accepted defence run from that particular callback alone.

### 3. BLOCKER — `reads` can connect a different selection's state to the incident oracle

**Locations:** `acceptance.py:67–113`; observer return records at `observer.py:486–501`; plan ruling 7; design §9.3.

**Measured `run_pair` acceptance:** production `grade()` returns the required `void`. A real production loader supports `read('correct') -> 'void'` and `read('unrelated') -> 'miss'`. The frozen test invokes `grade()`, then passes `[read('unrelated')]` to the dedicated oracle. The registry declares a `reads` component for that loader. The pair returns **accepted / connected**.

The loader ran outside the entry and after it, and its value matches the oracle. Nothing binds its argument, date, path, selection or state generation to the declared production operation. This verifies value coincidence with a production loader, not the relevant production result. `return` likewise uses the last matching file/qualname return without identifying which declared call the oracle consumes.

**Smallest fix:** record and validate the declared invocation and readback identity: date/selection/path or another contract-specific key, ordering relative to entry completion and the oracle record, and the exact complete value. For multi-read zipped observations, require the intended cardinality rather than allowing `zip` to drop unmatched values. Until that exists, label the present check as a value match requiring fixture review, and do not describe it as machine-proven connection to the incident. This counterexample does not establish that any of the actual 20 registered fixtures reads the wrong selection; those fixtures need their own review.

### 4. BLOCKER — the environment fingerprint omits bytes Python actually executes

**Locations:** `owned.py:181–223`; `runner.py:96–105`; plan Task 3 and self-review R14/W3/W4.

Three independent environment probes use the owned worktree's actual `.venv/bin/python -B`:

| Environment change | Executed value | Fingerprint |
|---|---|---|
| A site-packages module is a symlink to an external file; change its target bytes | 1 → 2 | unchanged |
| A relative .pth line names an external directory; change the imported module there | 1 → 2 | unchanged |
| Keep dependency `.py` bytes and timestamp unchanged, replace its valid cached bytecode with a same-size/timestamp alternate | 1 → 2 | unchanged |

`_tree_hash` hashes a file symlink's spelling and does not traverse directory symlinks. Relative .pth paths are resolved against the review process's working directory instead of the .pth file's directory. All `.pyc` and `__pycache__` content is omitted. `-B`/`PYTHONDONTWRITEBYTECODE=1` prevents bytecode writes, **not bytecode reads**.

The submitted R14 probe correctly prevents a newly created source cache during its clean two-run scenario; it does not cover existing dependency bytecode. W3 and W4 correctly cover normal files and absolute external .pth directories; they do not cover these alternatives. The manifest also stores only untracked names, so its name list must not be described as a content freeze of untracked inputs.

**Smallest fix:** hash the resolved executable dependency closure with cycle handling, or explicitly refuse unresolved/external symlink dependencies. Resolve .pth paths with Python's semantics, and inventory executable import-hook additions or declare them unsupported. Ensure evidence runs cannot consume unhashed existing bytecode, including dependency and external .pth caches; alternatively bind the bytecode actually executed. Hash every declared input's bytes regardless of Git tracking status. Retain the current mutation-path restrictions; this finding does not ask to loosen ownership or edit permissions.

### 5. BLOCKER — repair reports can manufacture observed incidents, and one timely citation qualifies unrelated old claims

**Locations:** `records.py:108–120,185–215`; plan ruling 6 and Task 8.

All three independently constructed records return **`validate(..., evidence_root=tmp_path) == []`** in publication mode:

1. `observed_incident`, **no occurrences**, and one primary operator report saying the repair was verified installed and no failure occurrence was witnessed. The report is cited only by `fix.verified_recovered`. It qualifies the step and also satisfies the global observed-incident primary-evidence check.
2. A July 17 report cites the normal July 16 occurrence **and an April 1 one-shot occurrence**. The timely July occurrence makes `ok=True`, so the April citation also passes.
3. The same report cites a July 17 verification **and an April 1 mitigation**. `any(...)` over steps lets the timely verification cover the old mitigation.

This directly contradicts “establishes that step only, never an occurrence” and “each role is judged on its own.” Correctly distinguishing occurrence and step lists is not enough when validation is existential within each list.

**Smallest fix:** qualify every `(report, occurrence)` and `(report, fix step)` citation independently and report errors at that precise claim. Make observed-incident disposition require a qualified primary witness to a deviation occurrence, not any primary report or installation observation anywhere in the record. A step-only report remains usable for the step, without promoting disposition. Keep unknown occurrence evidence unresolved. Text meaning still requires review; the validator must at least enforce explicit claim roles and prevent these structurally detectable promotions.

### 6. BLOCKER — publication still accepts a nonexistent expected-failure certificate and impossible latency

**Locations:** `records.py:78–84,123–140,240–252`; `record_schema.json:881`; design §7 and §9.7.

**Missing certificate, measured:** take the valid unfixed, plan-named E77 helper record, retain its expected-failure entry pointing to a nonexistent acceptance file, and call publication validation with an evidence root. It returns **zero errors**. Only historical replay and current defence are artifact-bound. A bare expected-failure path satisfies the fixture requirement without any accepted pair, node membership, registry binding or reviewed `exception_shape` decision.

**Impossible latency, measured:** onset is exactly 18:00, first machine detection exactly 17:00 on the same date, and detection latency is `[0,0]`. Validation returns **zero errors**. `_feasible_minutes` produces `[0,-60]`, and the later “not tighter than bounds” comparison never rejects this empty feasible interval.

**Smallest fix:** bind every claimed expected-failure reproduction to a present hash-matched accepted pair, the exact registered nodes and exception, the registry identity, connection status and the required recorded review. Require an evidence root whenever publication contains certified claims. Reject impossible chronological endpoint relationships before constructing latency bounds; do not clamp a wholly negative interval into a valid zero latency. Keep uncertainty as bounds or unknown.

### 7. BLOCKER — E77's required branch can pass without a pick message

**Locations:** `tests/test_incident_register_2026.py:572–577,681–714,727`; registered E77 derived review in `...-evidence/expected_failures.json`.

**Predicate probe:** a production-shaped daily pick has normal notification/delivery flags and a pre-cutoff timestamp. The observed message is **“No pick today”**, with no lineup checks. `_dm_kind` calls it a pick because it contains `pick`; `_e77_verdict` accepts the required branch.

**Actual source mutant:** change only `scheduler._format_pick_delivery_text` to return that message. Run the actual E77 incident and positive-execution-control nodes through their real scheduler harness. Result: **1 passed, 1 xfailed**. The positive control's real delivery path sets the flags, sends a message containing no batter selection and still passes. This is not limited to a hand-constructed observation dict.

Removing the lineup-check requirement is correct: a valid fallback repair need not run that check. The remaining required predicate must establish the delivered selection. It currently establishes only a flag, a timestamp and a substring.

**Smallest fix:** identify the actual pick-bearing DM for the declared batter/game/date and recipient, correlate its send result with the saved delivery record, and require the send and persisted timestamp to fall inside the intended day's pre-cutoff interval. Add negative controls for a skip/status message, wrong batter, wrong recipient and inconsistent delivery record. Keep the required branch independent of the repair's scheduling shape. Until then, the recorded E77 `derived` review overstates its success predicate.

### 8. SHOULD — L03 still refuses a valid delivery before the postponement

**Locations:** `tests/test_incident_register_2026.py:877–904`.

An observation with the selected game's real pick delivered at **18:15**, before the **18:20** postponement, no subsequent delivery, and the completed day through fallback is classified `('other', True, 0)` and raises an ordinary assertion. `_l03_outcome` requires the saved notification flag to be false for the required result. The contract forbids delivery naming the game **after** postponement; a valid earlier delivery need not erase its durable notification flag.

A repair can nevertheless reach the required branch: a temporary status guard at `_deliver_and_lock_pick` produces **one strict XPASS**, while the playable-game control passes. Thus this is a remaining false refusal of another legitimate repair direction, not the old “no repair can succeed” defect.

**Smallest fix:** decide the required result from the complete identified send history relative to `postponed_at`. Retain a consistent earlier delivery record when no forbidden later send occurred. Treat disagreement between the message history and durable record as an ordinary failure. Add the early-valid-delivery control.

### 9. SHOULD — deploy observation order and rollback identity are still assumptions inside the extractor

**Locations:** `deploy_runs.py:66–107,145–159,219–235`.

**Measured synthetic ordering case:** an older-created run executes at 14:00/14:01; a later-created run executes at 12:00/12:01. The first containing observation is 12:01, with the non-containing predecessor at 12:00. `first_live` returns **live_by=14:00, not_live_before=None**, because `observations` follows creation order rather than observation time.

**Measured synthetic rollback case:** fixed-template output says pre=A, deployed=B, canary rolling back to A, and **rolled back cleanly to C**. `extract` returns no anomaly, discards the parsed `rolled_sha`, and `observations` invents an A rollback observation. The observed line actually names C.

**Current-corpus limit:** I rechecked the typed retained corpus: **192 runs, 164 expired, 28 retained, 48 SHA observation points**, monotone in current order, **zero failed canaries and zero rollbacks**. These two probes do not invalidate the present corpus's bounds. They invalidate unconditional reusable-tool claims for those execution shapes.

**Smallest fix:** sort validated observations by their actual timestamps, reject conflicting equal-time/order evidence, retain the observed rollback SHA, and flag disagreement with the intended rollback target. Keep reported checkout/restart points separate from canary health and from assumed continuity. Do not reconstruct an observation from the intended shell command.

### 10. SHOULD — the sweep's string classifier still calls unrelated failures assertion kills

**Locations:** `...-evidence/tooling/mutation_sweep.py:114–119,151–162`.

A real pytest run containing only `raise RuntimeError("DID NOT RAISE")` produces exit 1 and the summary:

`FAILED test_probe.py::test_runtime_error - RuntimeError: DID NOT RAISE`

The exact submitted classifier puts this line in `assertion_kills` and declares **KILLED**, with no ERROR line. `AssertionError` in a node name or unrelated message has the same structural problem. The mutant classifier also does not use the subprocess return code when assigning KILLED, so a terminal assertion line is not proof of complete execution.

**Smallest fix:** obtain structured call-phase exception class, assertion/refusal location, setup/teardown outcomes, complete inventory and session exit status. Count a real pytest expected-refusal failure only when its type and intended assertion location match. Require a completed expected exit; classify other exceptions/aborts separately. Whole-suite execution is useful and should remain. Genuine pytest `Failed: DID NOT RAISE` can be a sound kill; substring occurrence alone cannot.

## False greens

| Green claim | What the independent evidence shows |
|---|---|
| `certificate.ok=true`, completed invocation, zero calls | Actual qualifying sends are omitted through initial C gaps, temporary C rebinding or existing workers. |
| Linked event at the declared Python method code | The event can belong to a different receiver. |
| `_safe` uses only builtin container methods | A dict lookup can call an application key's equality method. |
| Pair `connected` | A correct 51-element result can become a wrong 50-element result; a different selection's readback can be accepted. |
| Frozen venv fingerprint and `-B` | Actual executable symlink/.pth/bytecode content can change without changing the fingerprint. |
| Publication validator returns no errors | A repair-only report can create an observed incident; old claims borrow timely citations; the expected-failure artifact can be missing. |
| E77 positive execution control passes | The real scheduler can send “No pick today,” record notification flags and pass the delivery predicate. |
| 75/75 sweep kills | The retained result has useful regression evidence, but the classifier itself accepts an unrelated runtime error as a kill. My eight targeted kills do not independently reproduce all 75 runs. |

The reviewer decision remains a separate required gate. None of these probes carries my `accept` decision for the false certificate. A careful reviewer can reject it, but that does not make the runner's `accepted` or `connected` claim correct. Conversely, these tooling failures do not prove that the 22 real fixtures misdescribe their current production failures.

## Rulings

### Ruling 2 — external observer

**Accept the design amendment; block implementation equivalence.** The supplied witness requirements are now in §9.3. Keep them. C callbacks, thread lifetime, callable identity and application-defined serialization effects must be handled or explicitly unavailable in the actual execution, rather than listed as unsupported while acceptance proceeds.

### Ruling 6 — continuing conditions and fix-step reports

**Accept the role distinction and earliest relevant end in principle, with these required corrections.** Earliest end and occurrence-specific links repair the previous “latest fix anywhere” rule; the submitted regressions for both pass. An old unresolved condition can be witnessed today even when its first onset was weeks ago. That does not establish every earlier date as directly observed.

The time window is a qualification rule, not evidence that a report actually witnessed an occurrence. “No end is known” cannot by itself establish that a condition continued until the report. Likewise, code installation does not necessarily end a persisted stale-state occurrence: the resolver may need another run, a stranded date may need remediation, or a mitigation may only contain an alert storm. Installation and restoration remain separate facts.

Minimum replacement clarification:

> Qualify each report-to-occurrence and report-to-fix-step citation independently. A fix-step report establishes only the cited mitigation, installation or verification and cannot establish an observed incident. An observed incident needs a qualified primary witness to a contract deviation. For a continuing condition, the report must describe the condition as still observable at a supported time; absence of a known end is not evidence of continuity. Use the earliest supported event that actually ended the cited condition, restricted to its own links. A code install or partial mitigation ends that condition only when the evidence supports that relationship. Preserve inferred onset and uncertain end bounds explicitly.

Apply finding 5's per-claim validation. For an operator-dated installation, validate the report against that installation role, without treating the same report as evidence for all other links or occurrences.

### Ruling 7 — expected-failure connection kinds

**Accept `derived -> exception_shape`, with an adequate recorded fixture review. Amend `return` and `reads`.** They presently prove a value match against selected observer records. They do not enforce the contract identity and data relationship advertised by `connected`; findings 2–3 demonstrate this directly. Either bind the relevant invocation, selection and complete value, or publish the weaker check accurately with explicit fixture review. E77 and L03 correctly remain `exception_shape`; fix their predicate issues before renewing those recorded reviews.

### Task 7 — consolidation and disposition

The mechanical reconciliation is correct: **119 − 2 merges − 5 removals + 3 splits = 115**. I independently loaded and draft-validated all 115. Totals are **61 observed (30 A, 27 B, 4 pending), 24 latent, 4 near misses, 24 unresolved, 2 pre-ship**. This is not a result sign-off on every record's evidence, the complete 1,172-commit census, or all negative adjudications.

| Decision | Ruling |
|---|---|
| I-044 → I-095 | **Accept the merge of the same shadow-result deviation**, keeping onset and repair verification distinct. The retained record carries the shared repair memo and mechanisms. Remove stale pre-consolidation notes/tier text referring to a still-separate I-044 before publication. The repair addendum must witness the pre-repair deviation as well as verification; the current validator cannot establish that distinction. |
| I-096 → I-047 | **Accept.** Both drafts describe shadow prediction starving the watchdog and the `shadow_model=false` containment. The retained record includes mechanism and mitigation. |
| I-026 exclusion | **Accept within the stated scope:** training-frame order / single-seed model-quality evidence. No operational occurrence is established merely by that experiment. |
| I-076 exclusion | **Narrow the recorded rationale.** Exclude the statistical DD shortfall/model-quality question. The a275399 diff also touches alert routing, per-incident attention identity and authoritative slot readback. Those operational pieces need an explicit per-mechanism disposition; the generic “model quality” label is not sufficient. The second-bucket identity change may be pre-ship support for a new feature, not a previously deployed failure. I am not promoting it to an observed incident. |
| I-208 exclusion | **Accept as a coverage-bounded exclusion** on the retained source evidence: no cron/unit/production caller and the cited operational note is unavailable. A later live caller or contemporaneous report would reopen it. |
| I-103 / I-104 exclusions | **Conditionally accept as scope decisions under the latent-defect relevance rule, not as proved non-occurrences.** The retained 2be445e pack establishes that saver support was added, but does not establish the exclusion's stronger claim that the streak stayed below the trigger throughout the earlier interval. Retain a source for that claim or drop it. For both exclusions, record why the historical mechanism has no remaining watchdog/restore relevance; retirement alone must not erase a still-relevant grading/version-drift class. No new occurrence is inferred here. |
| I-113 split from I-106 | **Required and correctly retained.** A policy-artifact omission is a separate restore mechanism from slow prediction code in the same commit. See the source-strength ruling below. |
| I-114 / I-115 split from I-059 | **Accept.** Pick-time NaN exposure and reported daily health DMs are separable operational deviations from the wider audit bundle. Keep I-115's observed DM/basis claim distinct from the stronger claim that every alert was false; the retained record itself notes that recomputation is missing. |

**Three re-dispositions:**

- **I-063:** accept observed status for the continuing deviation described first-hand in the June 17 spec. That document explicitly describes the incorrect decision/display state and the fetch gate refusal. Preserve the draft's distinction between the June 11 inferred onset and the June 16–17 witnessed condition. “Continuing from June 11” must not relabel the whole interval as measured.
- **I-064:** accept observed status from the June 29 memo's current stalled-resolution observation, with the July 1 memo separately reporting alert and restoration verification. Keeping impact **tier_pending** is appropriate. I read the cited operational sections; I did not read the underlying artifacts or independently verify the box measurements.
- **I-074:** accept observed status from 4f0257a's contemporaneous statement that the July 10 shadow pick remained ungraded. This witnesses the continuing stranded result on August 9. Its endpoint deployment observations establish installed SHAs at those endpoints; the draft's claim about the intervening July 11 cron still needs an explicit continuity assumption. A later code install is not, by itself, verification that the stranded result was repaired.

**I-113 / Fly restore path:** accept the separate **observed restore-path incident**, at the report's actual evidentiary strength. Commit 947fce8 explicitly reports missing `mdp_policy.npz` causing heuristic fallback on Fly. Its diff adds that file to `sync_to_r2`; the same ref's `scripts/fly-bootstrap.sh:10–14` invokes `sync-from-r2` for the model directory, and the entrypoint invokes bootstrap on an empty volume. This supports the restore mechanism and keeps the Fly shadow host inside design §2's any-host restore scope.

The exact cold-restore execution is reconstructed from code and the report, not a retained live restore log. Phrase the evidence accordingly: **reported contemporaneous fallback on Fly; R2 omission and bootstrap mechanism verified in code**. Do not convert it into a witnessed contest pick, exact outage start, verified repair or evidence of production impact. Current `src/bts/data/sync.py:240–242` includes the base policy and still has no `mdp_tail_policy.npz` inclusion; that is a concrete residual of this sync path, not proof that every backup/restore route lacks the tail artifact.

The other listed episode exclusions remain explicit records of scope decisions. None of the above authorizes new historical outcome analysis or Route R access.

## Answers

1. **Can L04 repairs reach the required branch?** Yes. Reordering the terminal pick save before the streak update in the CLI and polling paths produces **three strict XPASS results and one passing control**. The injected BaseException still hits its declared save fault point, and the restart checks still execute. This is a sensitivity sketch, not a production repair: reversing two non-atomic writes creates a different crash window. Making `update_streak` a no-op produces **four ordinary failures**, not the dedicated defect exception.

2. **Can L03 repairs reach the required branch?** Yes. The real delivery-chokepoint postponement guard produces **one strict XPASS and one passing playable-game control**. A no-op `run_day` produces ordinary failures in both L03 nodes. The tested independent changes do not reach `PostponedPickDelivered`. The valid early-delivery direction is still falsely refused as finding 8 describes. Keep its `derived` review and component-level limits explicit.

3. **Does E77 accept only real pre-cutoff delivery after removing the check assertion?** No. Finding 7 supplies both a direct predicate probe and an actual production formatter mutant. Removing the scheduling-shape requirement was appropriate; the fix is a stronger delivery identity predicate. A no-op `run_day` gives **four ordinary failures across the E77/L03 incident and positive-control nodes**, with the six E77 verdict controls passing, so the known no-op false green remains closed.

4. **Are the self-review fixes real?** The unexpected-exception reason and completion guard reject the exercised failures and preserve restoration. D11/P7 each fail when the reason append is removed; the completion-guard-only mutations remain equivalent for these paths because the exception reason independently forces rejection. Keep the guard as defence in depth. R15's startup digest regression and R14's clean source-cache regression are sensitive to their named mutations. R14 does not close existing dependency bytecode. O2/D9/W3/W4 each have effective targeted regressions, with the additional coverage limits above.

5. **Is observer-off conformance sufficient?** It catches the submitted observer-dependent pass/fail cases, including dependence visible only under the mutant. It compares node states, not full observable outcomes or failure provenance. All of finding 1's accepted counterexamples pass these controls. It cannot substitute for complete boundary identity/interval evidence or a serializer that runs no application callbacks.

6. **Are replay, ownership and failure-path artifacts still blocked by their old counterexamples?** The rerun replay-drift, red-teardown, ownership and driver-failure checks pass their corrected expectations. I found no need to loosen the production-only mutation rule. The environment closure defects apply across evidence tooling, so those successful gates do not establish an immutable full execution closure.

7. **Is the strict sweep sound?** Executing the whole suite and rejecting setup/teardown/collection errors is sound directionally. Accepting genuine `pytest.raises` non-raises failures is reasonable. The present free-text classifier is not a sound implementation of “assertion kills only”; finding 10 is a real false classification. Retain the eight independently verified kills and the full retained result as measured outputs, with that limitation.

8. **Can the validator now let a report establish more than it witnessed?** Yes, concretely: a repair-only report establishes an observed incident, and one timely occurrence/step qualifies additional old claims. The earliest-end and own-link fixes help but do not repair role qualification or turn unknown continuity into evidence.

9. **Next authorized work:** revise the tooling and fixture predicates, rerun the counterexamples and actual pair, then perform the already-planned record/memo result review. This verdict grants no production repair, merge, commit, push, deployment, remote access or Route R approval.

**Final verification:** After restoring every temporary source mutation, the combined suite returned **228 passed, 22 xfailed** (exit 0). Tracked and staged diffs in wt4 were empty. The authorized wt4 directory and its Git worktree registration have been removed; the retained probe evidence remains at the path above. Main remains at **f6b3d0ff8ee0cc1701325912b6c285864a1fb959** with empty `git status --short`. Other project worktrees were left untouched. No project commit, network/SSH/gh call or production data read was made.

DONE
