# Fresh whole-range review, Part 1

## Verdicts

**(a) Tooling fit to run Task 5/6 certificate runs for the stated model: BLOCK.** F1 accepts a declared wrong value different from the value recorded; F2 does not enforce the claimed mutated-line link. These are blockers under the prompt's literal COVERAGE rule. F3 additionally leaves the promised builtin-assertion killing check unenforced. The smallest fixes are the type-sensitive safe-value comparator, branch-to-edit-span validation, and trusted builtin exception-identity flag specified below. Independent patch/fixture review remains necessary after those changes; it does not turn the current automated claims into enforced facts.

**(b) Evidence complete for the code pin: BLOCK.** Tree equality, pair configuration hashes, predicate input hashes, closure-screen body, and all 243 merged sweep rows bind as detailed below. However, F4 invalidates the strict assertion-kill inference from the sweep classifier, and F6 leaves the prepared cost runs' exact inputs and synthetic baseline unbound. Correct the classifier and retain/revalidate its exception evidence; retain the exact prepared inputs and reconstructible baseline with full run receipts. Changes to the tooling require evidence at the resulting new pin.

F5 is a measured environment-hardening gap, not a demonstrated accepted full defence and not the reason these two verdicts are BLOCK. The inherited prefix was absent in this review environment (`final_binding_receipt.json`). No missed-certificate cost or expressly excluded observation-sensitive shape is being promoted to a blocker. The report distinguishes automated acceptance from the independent reviewer decision required by `defence.py:23–25` and `current_defence/README.md:10–13`.

## Findings

Source locations below refer to `c59ee585e4138fe4b91a9062f261e352245f3d92`; evidence-only files are from `98945f413f0d333a3942589674eef233597afd1b`. Probe paths are relative to `r8-probes/fresh/` beside this report. These are observed failures of the indicated certificate or acceptance gate, not claims that a complete, independently reviewed Task 6 defence was accepted.

### F1 — BLOCKER: the declared wrong return can be a different JSON value

**Location:** `scripts/audit/incident_register/certify.py:265–268`; primitive recording at `observer.py:436–439`; claim at `certify.py:53–56`.

The observer accurately records an exact builtin `False`, but `certify` accepts a requested bad value `0` because it uses Python `==`. The recorded JSON boolean `false` and requested JSON number `0` are different values. The same alias occurs inside safe sequences/maps because their dictionaries/lists use that equality recursively. No application dispatch or excluded observation-sensitive behavior is involved.

**Executable probe:** `initial_probes.py`, `initial_events.json`, `initial_results.json`: actual type `bool`, actual value `false`, request `0`, certificate `ok: true`, empty reasons, live entry/branch/return links. This uses the real `_Monitor` and certificate code, with its direct-probe audit-census/quiescence setup; it does not run the destructive defence orchestrator. `test_fresh_contracts.py::test_distinct_json_boolean_and_number_are_not_the_same_return` is red at this pin (`red-contracts.log`). `trusted_value_probe.py` independently reproduces the mismatch using the trusted runner/bootstrap and the unmodified production `pick_was_delivered` function: genuine recorded `false`, request `0`, clean mutant-mode session gate, certificate `ok: true` (`trusted_value_results.json`, `trusted_value_outputs/observed.events.jsonl`). This corroboration substitutes only the read-only ownership-marker precondition with an exact own-path/HEAD check; it makes no claim that its unmodified branch is a mutation.

**Smallest fix:** use a recursive safe-value comparator that distinguishes booleans from numbers, including nested values, rather than bare Python equality. Define any intended integer/float equivalence explicitly. Add the boolean/number negative regression. Audit analogous equality in `acceptance.py` before extending this fix to pair-data matching.

### F2 — BLOCKER: a branch record need not name any mutated line

**Location:** `scripts/audit/incident_register/defence.py:76–77`, `defence.py:217–219`, `certify.py:253–275`; mutated-line claim at `certify.py:53–56`.

The spec check only requires the branch anchor's **file** to occur in `mutation_edits`. The later anchor lookup requires uniqueness, but never ties the line to the edit. In a single file, changing line 6 in `other()` while declaring unchanged line 2 in `entry()` passes `_check_spec`. The real observer then records the unchanged line and `entry()`'s already-existing bad return, and the certificate accepts it. The claimed link to the mutation has not been established. This can arise from an ordinary misplaced spec anchor; it does not require modifying the observer or its records.

**Executable probe:** `branch_probe.py`, `branch_results.json`: changed lines `[6]`, declared branch `2`, spec check passed, actual return `"bad"`, certificate `ok: true`, empty reasons. `test_fresh_contracts.py::test_accepted_branch_is_actually_in_the_mutation` is red. This establishes the spec-check/certificate gap; it does not claim a full defence acceptance or that any shipped spec has this particular mistake.

**Smallest fix:** validate the resolved post-edit branch location against the actual replacement spans, accounting for shifted line numbers; require an executable location belonging to the changed decision, and reject an unchanged anchor in another function. Retain patch/fixture review for the semantic claim that the edit restores the pre-fix decision. The seven eligible prepared specs put their declared anchor text into a replacement span, so this probe is a tooling counterexample rather than a demonstrated defect in those seven inputs (`prepared_branch_audit.json`).

### F3 — HIGH, blocks the promised killing-failure check: exception metadata is not builtin class identity

**Location:** `scripts/audit/incident_register/observer.py:544–550`, `defence.py:146–157`, `replay.py:171–176`; builtin-AssertionError requirement at `defence.py:11–13`.

An ordinary heap exception class can have module `builtins` and qualname `AssertionError` without being the builtin class. The observer records those writable names, and `_assertion_ok` accepts them. A test that raises this non-assertion error at the declared line therefore passes the killing-failure identity gate. Both conformance twins can raise the same class with the same message, so their comparison supplies no missing identity check.

**Executable probe:** `failure_identity_probe.py`, `failure_identity_results.json`, and retained trusted-bootstrap records in `failure_identity/`. The actual class identity check is false; pytest returns 1; the observer reports `builtins.AssertionError`; `_assertion_ok` returns no reasons. The frame and line are genuine, not fabricated. The probe invokes this gate with the scratch test's own containing path as the worktree argument; it does not assert that the full worktree/session acceptance pipeline accepted it. `test_fresh_contracts.py::test_kill_requires_the_builtin_assertion_class` is red.

**Smallest fix:** have the trusted observer record an owned boolean from `exc.type is AssertionError`, and require it in current-defence and replay symptom checks. Continue recording names for diagnostics. Names alone cannot prove this contract.

### F4 — HIGH, blocks the strict sweep's error classification: a non-assertion error is called KILLED

**Location:** `docs/audit/2026-09-29-incident-register-evidence/tooling/mutation_sweep.py:320–322`, `:340–349`; contrary classifier explanation at `:10–14`.

The classifier treats every JUnit message starting `assert ` as assertion-shaped. A real heap exception named `assert nonassertion`, with module `builtins`, produces `assert nonassertion: the application failed before any oracle`. The real pytest run failed before any assertion, yet the strict classifier returns `KILLED` and a killing node. The retained sweep rows consequently do not establish actual assertion kills merely by passing this classifier.

**Executable probe:** `classifier_probe.py`, `classifier_results.json`, retained `classifier_app/out.xml` and pytest logs. Actual class is not builtin AssertionError; return code 1; classification `KILLED`. `test_fresh_contracts.py::test_strict_sweep_does_not_classify_a_custom_nonassertion_as_killed` is red. This does not show that any particular one of the 243 recorded mutants was misclassified. `extra_probes.py` also confirms the more ordinary dynamically named `AssertionError` case: its actual class is not builtin, but its JUnit message is `AssertionError: lookalike`, and classification is `KILLED` even with the exact node inventory supplied (`extra_results.json`, `classifier_variants/builtin_name/result.xml`).

**Smallest fix:** retain structured exception identity from a trusted pytest reporting hook and classify from that, including an explicit treatment of pytest's failed-expectation errors. Do not infer the exception class from message prefixes. Revalidate the sweep with the corrected classifier and retain the identity evidence.

### F5 — MEDIUM, measured environment gap: inherited relocated bytecode can outrun source checks

**Location:** `scripts/audit/incident_register/runner.py:60–61`, `:121–125`; `owned.py:131–139`, `:291–332`; overbroad regression explanation at `tests/scripts/incident_register/test_runner.py:387–403`.

`PYTHONPYCACHEPREFIX` is neither scrubbed by the runner nor covered by the reset's worktree-local cache removal or venv fingerprint. `PYTHONDONTWRITEBYTECODE=1` prevents new cache writes; it does not prevent reading an existing valid cache. A same-size source edit with preserved mtime can consequently import old bytes from an external prefix while source-only digests see the edited file.

**Executable probe:** `cache_probe.py`, `cache_results.json`: native Python first imports `"old"`; after the same-size/mtime edit to `"new"`, the second import with write-bytecode disabled still returns `"old"`; prefix is absent from `_SCRUB`; altering that external cache leaves the minimal probe venv fingerprint unchanged. This is a native-import plus fingerprint probe, **not** a full defence false-positive and not evidence that the prepared runs inherited this variable.

**Smallest fix:** reject/scrub the inherited prefix and prevent spec environment overrides from reintroducing it, or explicitly clear and bind the relocated cache. Seed a preexisting valid cache in the stale-bytecode regression. Until this is done, narrow the claim that disabling bytecode writes alone proves edited-source execution.

### F6 — HIGH, evidence-completeness blocker: prepared cost inputs cannot be reconstructed from the retained record

**Location:** `docs/audit/2026-09-29-incident-register-evidence/current_defence/prepared-c59ee58.json:2–8`; full input retention available in `scripts/audit/incident_register/defence.py:163–167`.

The summary names an absolute tooling directory, a synthetic baseline SHA and hashes of as-run specs, but does not retain those exact specs, a reconstructible synthetic tree, or the per-run acceptance/events. The synthetic commit is unavailable in the allowed repository object database (return code 128). The shipped specs use `BASELINE`; replacing that with the named synthetic SHA does not reproduce **any** of the nine recorded hashes. A bounded search of routine baseline/optional-field transformations also reproduces none. This does not prove the author ran incorrect inputs; it means the retained evidence cannot establish which inputs were run against the pin.

**Executable computation:** `evidence_audit.py`, `hash_followup.py`, `evidence_binding.json`, including all nine recorded/computed hashes and the synthetic-object return code. No author's worktree was accessed.

**Smallest fix:** retain the exact as-run specs, a synthetic-tree construction recipe and manifest (or reachable synthetic object), resolved tooling SHA and observer hash, and the referenced acceptance/event records. Recompute their hashes and make the summary point to these retained artifacts. Prepared cost runs remain cost/closure evidence, not final Task 6 certificates.


## False greens

The fresh full baseline is green: **502 passed, 22 xfailed**, including jsonschema-dependent tooling tests (`baseline-schema.log`). The first run without the extra schema dependency had six skips and is not the comprehensive baseline (`baseline.log`). Four new negative contract checks fail at the unchanged pin, return code 1 (`red-contracts.log`); they expose F1–F4, rather than repairs to application code.

| Passing regression or recorded kill | What it actually establishes | Fresh counterexample / smallest strengthening |
|---|---|---|
| `tests/scripts/incident_register/test_runner.py:387–403`, sole killer of sweep R14 (`mutation_sweep.txt:40`) | Two runner executions with no previously seeded bytecode run the changed source. | F5's preexisting relocated cache still executes old source with bytecode writes disabled (`cache_results.json`). Seed that cache before the first evidence execution, then make the same-size/mtime edit. The current test's broad edited-source claim exceeds its setup. |
| `tests/scripts/incident_register/test_defence.py:188–191`, sole killer of D8 (`mutation_sweep.txt:61`) | An anchor in a different **file** is refused. | F2 changes another function in the **same** file and certifies through an unchanged anchor (`branch_results.json`). Add the same-file negative; a filename-only kill is no proof of mutated-line linkage. |
| `tests/scripts/incident_register/test_sweep_classifier.py:28–42` | Several representative JUnit strings classify as expected; the locally declared lookalike keeps its original module-qualified name. | Giving the lookalike builtin module/name metadata is a small change to the negative scenario, and it becomes `KILLED`; the `assert ` keyword premise fails as well (`classifier_results.json`, `extra_results.json`). Test actual class identity, not the representative spellings. |
| Existing certificate return tests and all-243-KILLED aggregate | Their chosen values and disabled guards are exercised. | A distinct boolean/number request remains green in the certificate despite accurately recorded return data (F1; `initial_results.json`, `trusted_value_results.json`). Add type-sensitive negative values, including nesting. |

No specific existing sweep row is accused of an actual non-assertion kill: the per-mutant JUnit XML was discarded, and the classifier can falsely label one. The sweep proves coverage against its selected 243 edits under that classifier, not against the fresh counterexamples. Its merged/raw-row checks are recorded below.

Executable checks across the requested trust boundaries:

| Dimension | Measured result and probe |
|---|---|
| Attribution | Real boundary and standard mock calls certify; former bindings, shared-code functions and custom mock calls do not (`model_probes.py`, `model_results.json`). These are direct monitor/certificate probes with a synthetic zero census, not whole-defence acceptance. |
| Arguments / returns | Keyword-only and cell arguments are read; an incomplete argument does not witness. The actual recorded boolean is faithful but matches the wrong declared numeric value (F1). `model_results.json`, `initial_events.json`, `trusted_value_results.json`. |
| Entry / branch | A genuine bad boundary called after entry exit is refused: event order is start, entry, branch, entry-exit, boundary, end (`extra_probes.py`, `extra_results.json`, `linkage_events.json`). The earlier `no_live_witness` battery case only lacked a call; this additional probe actually performs it. Same-file mutation linkage remains F2. |
| Observer side effects | A dict with an application string-subclass key and a value trapping hash/equality/attribute/repr access becomes incomplete; trap hit list stays empty (`model_results.json:safe_purity`). This is a tested representative, not an exhaustive proof of all callbacks. |
| Observed/unobserved twin | Real monitored and quiesced subprocesses reject differing ordinary failure messages (`session_probes.py`, `session_results.json`, `session_outputs/`). The probe deliberately varies fixture input to exercise the comparison; it does not show observation caused a divergence. Address-shaped application text still normalizes to the same digest (`model_results.json:normalization`), exactly the documented scope limitation. |
| Session / environment | A genuine trusted-runner execution passes; copied records with wrong observer hash, prefix, rootdir or source digest reject (`session_results.json`). This read-only probe replaces ownership-marker checking with exact own-path/HEAD checking; it never invokes reset, source edits or destructive orchestration. Native-import cache behavior is separately measured in F5. |

The tests and probes do not settle the explicitly open native-thread attachment interval or prove observer invisibility for excluded liveness, instrumentation, timing or numeric-address decisions. Those are model-scope comments, not newly invented blockers (`certify.py:101–142`). Ordinary fixed clocks, prewritten fixture files and object identity remain permitted inputs.


## Evidence binding

**Pins and execution.** The review worktree was created at `98945f413f0d333a3942589674eef233597afd1b`. Git tree IDs for `src`, `scripts` and `tests` are identical to `c59ee585e4138fe4b91a9062f261e352245f3d92`; all source findings therefore apply to the tooling pin (`evidence_binding.json:trees`). The worktree's tracked status is empty after the tests/probes, and the sweep harness itself has identical bytes at both pins (`final_binding_receipt.json`). The schema-complete baseline used the worktree's own Python 3.12.13 and offline `uv run --with jsonschema==4.23.0`; the retained logs show 502 passed and 22 xfailed. No assertion is made that the whole mutation sweep was rerun: doing its source edits here would violate the task's tracked-file constraint.

| Artifact | Verified binding | What the retained artifact does not establish |
|---|---|---|
| `expected_failures_runs/acceptance-c59ee58.json` | Resolved full tooling SHA matches. All 22 registered nodes are accepted and 40 controls are listed passed. The registry's normalized entries hash recomputes to `8aa988c16b40927c1e8f49a0a230d95fd2f0c666ab15b82e3ebd72dafd6a3dc9`; the declared observation configuration recomputes to `2f705fb9ce344a98710a2957325e74548f8b449a2a58a8ea9508e78e9771deac` using the recorded path **as a string**, without accessing that worktree (`evidence_binding.json:pair`; artifact lines 71–73). | It gives marked/unmarked raw-event digests at lines 127–130, but those raw event files are absent from the retained evidence tree (`evidence_binding.json:tracked_raw_jsonl`). I verified the summary/configuration bindings; I could not independently replay its raw observation history. |
| `current_defence/prepared-c59ee58.json` | Summary contains seven accepted eligible specs and two absence refusals. The seven shipped anchors appear in their declared replacement spans (`prepared_branch_audit.json`). | F6: the synthetic commit cannot be resolved, none of the nine candidate as-run spec hashes match, and full specs/acceptances/events are not retained. The filename and absolute tooling path alone do not resolve executed bytes. The README's “Not yet run” refers to actual Task 6 inputs, not a contradiction of synthetic cost runs (`current_defence/README.md:3–7`); I do not treat these as final incident certificates. |
| `tooling/fixture_predicate_mutants.txt` | Header resolves c59ee58 and names the fixture hash `5c5748ec9b9a66da63e0f17a9fba1355ded0ed91d4f342756110b888e4b7226b` and driver hash `b56d72ebe0e57fb918dcd0145ce0d066df806a03ef3d57d19b7ac501ef2586db`; both match the retained bytes. The output records baseline 40 pass / 22 xfail, P1–P7 and F1–F2 caught, R1–R2 required branches reached, and “ALL AS REQUIRED” (`evidence_binding.json:predicate`, `final_binding_receipt.json`; output lines 1–14). | This establishes the declared predicate sweep record/input binding, not the positive-certificate invariants F1/F2. I did not rerun its tracked-fixture edits. |
| `tooling/closure_screen.txt` | An independent screen rerun has an identical body (`closure-rerun.txt`, `evidence_binding.json:closure`). Its 44 timing/resource hits and five address hits are warnings; zero literal matches in other categories is not an absence proof. | The screen explicitly does not resolve aliases, generated code or outside dependencies (`closure_screen.py:5–10`). Its header refers to earlier per-closure review dispositions; those reports were not opened under the blind rule. The focal source review below is limited, rather than an assertion of exhaustive closure independence. |
| `tooling/mutation_sweep.txt` plus `mutation_sweep-c59ee58-chunks/` | 243 unique labels, all recorded KILLED, no missing/duplicate labels; every mutation anchor is unique and its replacement compiles at this pin. Every merged row matches exactly one non-void raw chunk. All 16 chunk metadata pins equal the full c59 SHA; all end with `DONE` and `exit=0`, and each starts with the recorded clean 462-test baseline (`evidence_binding.json:sweep`, `final_binding_receipt.json`). | Header lines 6–8 transparently void the four initial chunks whose exit lines were lost and say JUnit XML was not retained. The replacement chunks close the recorded intervals; the voids are not counted. Discarded per-mutant exception data plus F4 prevent an independent strict assertion-kill conclusion from the aggregate. |

**Prepared witness paths, independent focal source review.** I inspected the following entries and their killing fixtures, with hashes retained in `final_binding_receipt.json:focal_closure_source_hashes`. No new excluded observation-dependent choice was identified on these specific witness paths. This is source reasoning, not measured observer invisibility or exhaustive review of every imported/native dependency, and it does not clear F6's unavailable as-run closures.

| Eligible spec | Source/fixture disposition |
|---|---|
| I-0811-a | `leaderboard/auth.py:158–265` classifies fixed mocked response shapes and performs retry sleeps; `tests/leaderboard/test_auth.py:161–168` replaces that module's time binding with a sleep recorder. The killing fixtures use fixed blank responses (`:244`, `:400`), not elapsed-duration decisions. |
| I-0811-b | `cli.py:1852` enters the CLI's error classifier; `tests/test_contest_fetch.py:405–416` replaces cookie/uid/login helpers. The declared witness is the bad message at the **start** of `_contest_fetch_alert`, not proof that a DM was sent. That function has a wall-clock cooldown (`cli.py:1554–1592`); do not broaden this boundary-call certificate to delivery or timing independence. |
| I-0813-a | `picks.py:861–914` classifies a fixture pick and mocked detailed statuses. Killing assertions distinguish locked/stale state (`tests/test_warmup_lock_classification.py:44–77`). The returned public-field state uses fixed status data. |
| I-0830-a | `_deliver_and_lock_pick` uses fixture `_now_et` readings, including a predefined crossing sequence, and mocked transport/contest/capture/health helpers (`scheduler.py:926–1090`; `tests/test_late_delivery_guard.py:64–95`, `:141–160`). The witness is the declared mock boundary call, not real delivery. This entry does not execute the scheduler's separate monotonic refresh-budget loop flagged by the screen. |
| I-0830-c | `data/pull.py:83–121` makes a cache-presence decision over prewritten fixture files; tests replace discovery/download transport and `time.sleep` (`tests/data/test_pull.py:180–195`). Sleeping is the witnessed call; there is no measured elapsed-time oracle on this path, and the observer does not call that sleep function. |
| I-0830-d | `health/late_delivery.py:44–85` evaluates prewritten pick/state files with explicit fixture date `D` (`tests/health/test_late_delivery.py:8`, `:38–42`). These date/clock inputs are allowed by the model. |
| I-0903-a | `strategy.py:233–331` resolves the objective from supplied state/date and policy arrays; `simulate/tail_policy.py:97–108` uses arithmetic reachability. `tests/test_tail_policy_strategy.py:24–71` builds deterministic in-memory policies and checks returned decision fields; the contract is that decision, not a season outcome or timing result. |

The absence specs I-0813-b and I-0830-b remain outside Phase 1 certification and refused; I did not recast them as missing-event certificates (`current_defence/README.md:24–28`, prepared summary). I make no new closure clearance for their scheduler timing paths.

**Review boundary and cleanup.** All added probes and outputs live under the authorized `r8-probes/fresh/` root; the only BTS review worktree was `wt-fresh`. No tracked source was edited. No repository `data/`, author's worktree, network, SSH or GitHub access was used; no commit, push, deployment or operational run was performed. Baseline tests create their own synthetic fixture repositories under the probe basetemps. The task worktree was removed (return code 0); its path is absent and a successful Git worktree listing no longer registers it (`cleanup_receipt.json`). This check precedes the final report hash/marker.


## Blind-rule note

Before I had read the blind rule, the initial tool batch searched the memory registry for this project and returned earlier-review topic keywords. The exposed snippets included prior review terms, test counts and a historical code pin (MEMORY.md lines 103-155). The supplied session memory summary and AGENTS instructions also already contained historical W1.5 context. I did not open any prior review, plan, rollout summary or author's worktree, and stopped memory lookup once I read the rule. This review cannot honestly be described as perfectly blind; all findings below require fresh code/evidence and executable probes.

After the report was complete, a registry-line check selected MEMORY.md line 103 solely for the required memory citation; it supplied no review evidence and changed no finding.
