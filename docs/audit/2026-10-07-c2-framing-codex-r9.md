## Verdict
**BLOCK.**
Reviewed-commit: 537e37b71d024196618960299022f3da0d35bbf7

The narrowed threat model does not yet have the claimed enforced boundary. The unchanged permitted suite finishes with **27 failures and 207 passes** because the runner's new empty-configuration invocation allows collection outside its owned root. Separately, the unchanged gate admits test code whose allowed import resolves to an unscanned local module. After adding collection confinement only in an owned scratch copy, that module's public hook-dispatch wrapper aborts session finish and is certified RED. A linked-module control also defeats the new change scan.

The original R8-1 lifecycle is now rejected by the gate. With collection confined, the pluggy watch correctly refuses its before-yield session and call controls. The experiment and validator remain unchanged and the legitimate default-scoring synthetic three-seed replay still validates in a fresh interpreter.

This is Eric's authorized round 9 under the stated narrowed runner boundary and otherwise full review rules. The item stops. A tenth round requires a new ruling. This verdict admits no seed, run or operational action.

The complete prompt was read first; HEAD and the clean tracked tree were verified. All review writes used the two requested deliverables or the owned prefix `/private/tmp/c2-framing-r9-7PLJezt9`. No real evaluation data, operational configuration or credentials, network, box, escalation, tracked repair, walk-forward or model fit was used. Installed pytest source was read intentionally to explain collection. The unchanged runner accidentally attempted to stat `/private/tmp/codex-daemon-501` during ancestor collection; the sandbox denied that access. This was disclosed immediately in commentary. No source from that path was read. The same collection problem affected the permitted subprocess tests. Scratch is deleted at completion.

## Findings

### R9-1 — Empty configuration reopens collection outside the runner root

The new command builder in `scripts/audit/c2_framing/mutant_runner.py` supplies an explicit root directory, an empty configuration at `/dev/null`, and no conftests, but does not set the collection cutoff directory. These are different controls. Installed pytest 9.0.2 walks the named file's ancestors while they remain within its configuration cutoff; an explicit root directory alone does not stop that traversal.

I ran the exact permitted suite from the checkout, routing temporary files under the owned r9 prefix. It collected 234 top-level items and finished **27 failed, 207 passed in 208.55 seconds**, exit 1. Runner tests that should certify ordinary failures or exercise their defensive layers instead encounter collection errors. This is a regression from r8's 162 passing tests in the same sandbox.

An independent unchanged-runner probe at `/private/tmp/c2-framing-r9-7PLJezt9/shadow_numpy` retained the actual collection subprocess. Its command specified that scratch directory as root, yet pytest traversed `/private/tmp` and attempted to stat the protected daemon directory. Collection returned 4 with no intended nodes. Runner `main` returned 1 and printed `X1 INVALID: the named tests do not collect`. The target remained unchanged. The prescribed clean-run and ordinary-mutant tests show the same inability to reach their intended execution paths.

For further diagnosis, I created `/private/tmp/c2-framing-r9-7PLJezt9/confined_runner.py` from the reviewed runner, changing only its command builder to add a collection cutoff equal to the owned root. No tracked file was changed. A runner-focused test subset with that same argument adjustment produced **112 passes and one remaining failure** in 24.85 seconds. This is a conditional diagnostic, not a passing unchanged suite or a complete verified repair. Ordinary owned two-test failures then execute completely and classify RED; an ordinary complete pass classifies SURVIVED, with no foreign plugins or observed hook changes.

The issue is a false refusal and a collection-scope failure in the new invocation. It is not evidence that the later defenses correctly refused the intended hostile lifecycle: those unchanged subprocesses never reached it.

### R9-2 — Allowed import spelling is not a binding to reviewed module bytes

The gate's file set contains the named test modules and their package initializers. It does not bind an allowed import to its actual source file, nor parse a local module reached through that import. The static vocabulary audit starts from installed modules in the author's environment; it does not establish what pytest will import from a test root.

At `/private/tmp/c2-framing-r9-7PLJezt9/shadow_numpy_confined`, I placed a test module, a target, and a pre-existing local `numpy.py` before the ledger started. The test imports an allowed attribute from the module spelled `numpy`, under an allowed fixture name. The unchanged gate returned **no problems** and listed only `test_owned.py` as gated source. Pytest instead imported the local module. That unscanned source supplies a fixture which uses pytest's public request object and pluggy's public hook-call-monitoring API to wrap dispatch. It accesses no recorder, sentinel, evidence file, report attributes or private pytest field.

With only the R9-1 collection argument adjusted in scratch, both intended test bodies genuinely fail their target assertions. Their setup and teardown complete. The public dispatch wrapper runs after the sentinel's session-finish record and aborts before its completion canary. The actual mutated subprocess returns 1 and reports `Exit: owned public hook monitor abort`. The runner returns **0**, prints **`X1 RED`**, and reports `NOT RED: none`. The target is restored.

The authentic bundle has six phase reports, two AssertionError call observations, and matching finish counters: exit status 1, failed 2, collected 2. It also has empty late-plugin, ordering, hook-change and foreign-plugin indications. The wrapper-entry canary exists; its completion canary does not. No interrupt, internal-error or collection-error record is present.

This mechanism changes hook dispatch through a public API without adding or removing a hook implementation. Consequently, watching the two pluggy add/remove functions does not observe it, and no foreign plugin is registered. The missing boundary is the unscanned imported fixture code, not deliberate tampering with the runner's objects or private pytest state.

The unchanged full `main` in this sandbox first hits R9-1 and returns INVALID; I do **not** claim it reached RED here. The measured gate omission is unchanged, and the false certification is measured with collection confinement as the sole adjustment. Repairing collection alone leaves this boundary failure intact. A refusal caused by unrelated ancestor permissions does not enforce the claimed test-code boundary.

### R9-3 — The change scan misses a linked import's physical source

The scan recursively walks the root and interpreter directories, but uses the link's metadata and does not traverse linked directories or bind imported files to their physical targets. A pre-existing link can therefore name source outside those scanned trees.

At `/private/tmp/c2-framing-r9-7PLJezt9/linked_import`, a pre-existing local module link points to an owned sibling directory, `linked_import_external`. Admitted test code starts a child which writes that linked module, then imports it. All paths and writes remain under the authorized scratch prefix. The generated module uses the same public dispatch-wrapper lifecycle as R9-2.

The unchanged gate returns no problems. With collection confined only, the child-written source is imported, session finish aborts after the sentinel record, and runner `main` returns **0 / RED** with no change-scan alarm. The wrapper-entry canary exists and completion does not. The physical module changed after the ledger started, but the link's metadata did not establish that fact.

The independent `direct_import` control writes the same source into an ordinary file under its root. It produces the same inner RED bundle, but the runner's scan overrides that classification to **`INCONCLUSIVE(exit 1, files changed during the run)`**, identifies the changed `numpy.py`, and returns 1. This isolates the physical-source coverage gap from failure of the ordinary write detector.

These are conditional end-to-end diagnostics because of R9-1, with the gate, scan, driver, watch and classifier otherwise unchanged. They contradict the new claim that every file a child writes and the run later imports is covered by the change scan. The child merely writes owned source; it sends no signal, attaches no debugger, and changes no private interpreter state.

### R8 requirements and earlier runner controls

Eric explicitly superseded binding ordering to the active dispatch with an enforced gate and the add/remove watch. I assessed that revised requirement rather than requiring the former design. The stale universal abort sentence is removed; the new docstring correctly distinguishes an escaping abort from one pytest catches, and explicitly acknowledges the live-registry check's before-yield limitation.

I ran the supplied `r8_witness_replay.py` unchanged. It reproduces RED on the r8 runner and gate refusal on r9. Its unchanged r9 dynamic run returns `INCONCLUSIVE(exit 4, collection errors)` because of R9-1; that run does not demonstrate the watch. The same supplied replay, with the collection cutoff added only to the r9 invocation, reaches the lifecycle and returns `INCONCLUSIVE(exit 1, plugin system changed)`. Its bundle records the add and remove operations while the older late-plugin and ordering indications remain clear.

Twenty independent owned cases also ran with this disclosed collection adjustment. The two ordinary in-scope controls produce complete matching evidence: genuine failures are RED; a complete pass is SURVIVED. The other eighteen cases are refused by the gate. Forced through the driver, they remain non-RED:

- Captured-stdout spoof, both final-teardown abort messages, and exit after the final report: interrupted refusal.
- Ordinary session-finish abort and both pre-existing wrapper aborts, including tryfirst: session-finish-incomplete refusal.
- Non-strict XPASS and runtime xfail: collection-marker or report-status refusal respectively.
- Unconfiguration abort: no-normal-end refusal, without a bundle.
- Persistent and briefly registered public-API plugins, and both original r7 self-cleanup controls: late-registration refusal.
- Base-class after-yield wrapper: ordering refusal.
- Base-class before-yield session wrapper with and without abort, and before-yield call wrapper: plugin-system-changed refusal, with both monitored operations retained.

The conftest-based old cases were supplied as explicit plugin modules for these diagnostics, since r9 intentionally suppresses conftests. Every owned target was restored. The prescribed tests cover harmless early foreign plugins, early removal and rewritten reports, but their unchanged subprocesses failed before reaching those controls. The conditional runner-focused rerun improves that coverage; it is not substituted for the failed permitted suite. Complete clean evidence for the full unchanged real runner-test selection is therefore not established in this environment.

The vocabulary gate's existing bytes parsing, package-initializer, mutation-override, fixture-parameter and forbidden-name tests pass in the permitted suite. Empty configuration, no conftests, disabled entry points, scrubbed Python/pytest environment, safe interpreter path flags and disabled bytecode are meaningful controls. They do not bind arbitrary allowed imports to reviewed source, and the static audit explicitly omits instances and call results. The measured imported fixture illustrates that limitation. The current child-process claim also needs the physical-source correction in R9-3.

### Experiment and validator regression checks

`screen.py` remains byte-identical to r8, including all 35 top-level functions. The narrowed runner threat model was not applied to the experiment or result checks.

Three synthetic completed runs at the registered 10,000 trials and 180-day scoring settings pass same-process and fresh-interpreter aggregation: A positive and B negative. The actual existing HEADs are `b9c3c8188fdc6f68e8904fe757afbafb21640050` for seed 1 and `537e37b71d024196618960299022f3da0d35bbf7` for seeds 2–3; real git checks return no problems. Equal reviewed/run HEAD and metadata-only descendants remain accepted. Missing HEAD and an old reviewed closure refuse.

My first synthetic identity used `d3bf225` and correctly refused the runner changes subsequently made at `b9c3c81`. That was a stale harness premise, not an experiment defect. The successful replay binds the last executable change. Admission, input loading, feature loading and walk-forward are stubbed with synthetic trusted pins/identities and an owned release-row mock; claim writing, retained profiles, labels, scorecards, validator and git checks are real. No actual admission or release was created.

The earlier damaged-artifact controls remain closed. Missing nullable season membership, extra identity fields, coherently altered secondary scorecards with unchanged profiles, 2024 profiles substituted for 2025, foreign HEAD and bogus identities all refuse at their relevant consumers. Missing-season, secondary-card, duplicate-year and foreign-HEAD witnesses refuse direct validation, release, launch, child run and aggregation. The fresh damaged namespace creates no seed-2 root. Nonnumeric probabilities refuse; an injected scorer TypeError becomes RunInvalid. Valid numeric probabilities including 0 and 1 remain accepted by the predicate. Damage probes use 200 trials; legitimate replay uses the full registered defaults.

Real feature computation on 48 synthetic PA rows and 24 games reproduces all 16 production features exactly after installing frozen inputs. The closed computation makes no trapped text or table reads; both computations retain 28 non-null bullpen values, park drag is absent as registered, and the pitcher self-check matches. Original-portion labels preserve an original miss despite a resumed hit and drop the resumed-only batter; counts are one changed and one void, with ranks 1 and 2. Registered settings pass and each independently changed setting refuses. Recording classifier constructors receive the three seeds with deterministic and row-wise flags true, without fitting a model.

Exact Eric/30 release parses; Ericsson, zero and overflow budgets refuse. Budget decisions remain over_cap, ok, checkpoint and ok for the four prior boundary vectors. Namespace, repeat-claim, cardinality and shared-admission tests pass. The author's box facts remain stated and unverified; none of this establishes box readiness or real-input coverage.

### Test and mutant strength

The current spec has 188 unique entries with unique source anchors, no missing current-spec result, and no option-bearing named test list. Latest recorded statuses are 185 RED and three SURVIVED: G8, H12 and N10. These are the author's recorded results, not a new tracked-source sweep in this review. The independent permitted suite fails as described above, so the recorded ledger cannot establish a clean local suite under this sandbox.

R31–R38 exercise loading controls; R39–R75 exercise the gate; R76–R88 exercise change scanning; R89 and R91–R97 exercise dynamic plugin evidence. These tests cover explicit forbidden vocabulary, named modules/initializers and ordinary changed paths. They do not test an allowed import resolving to additional local source, a public dispatch wrapper reached through that source, or physical source behind a link. R9-2 and the matched direct/linked controls expose those untested components. The static gadget audit reruns with the recorded 157 visited objects and reported allowed endpoints, but it does not cover this import-binding invariant.

G8, H12 and N10 remain supported as equivalents over validator-accepted evidence. Each accepted run already has complete trusted pins and the exact trusted identity, and each accepted scored unit has valid season membership and ranks. Independent owned copies agree with the original on ordinary three-seed acceptance and on coherent wrong-pin, extra-identity and missing-rank-one refusals. These redundant later checks do not repair any runner finding.

## Required changes

1. Bound collection explicitly to the owned root for both collection and execution while preserving empty configuration and no conftests. Re-run the prescribed suite in this sandbox; resolve the remaining conditional-subset failure rather than treating the argument change as a complete repair.
2. Bind every allowed import used by the suite to reviewed installed/repository bytes, or include the actual additional test/helper source in the bytes gate. An allowed module spelling is insufficient when pytest's import path can resolve local source. Test the imported-fixture control with actual gate and driver evidence. Public hook-dispatch wrapping is inside the newly excluded mechanism set and must be refused before execution.
3. Cover the physical source of linked imports in the change scan, or refuse such import paths. Retain the ordinary-file control and require the linked write-then-import case to refuse. State the child-write guarantee only for coverage actually enforced.

No repair, further review or operational run is authorized by this BLOCK. Round 9 is the last round under the current ruling; a tenth round requires Eric's new authorization.

## What was run

- The exact permitted suite from the checkout, with owned temporary and plotting-cache routing: 234 collected, 207 passed, 27 failed, 208.55 seconds, exit 1; Python 3.12.13, pytest 9.0.2 and pluggy 1.6.0.
- The unchanged independent scratch collection probe, the supplied vocabulary audit and supplied r8 replay. The scratch collection and r9 replay expose the ancestor-collection failure; the audit reproduces its recorded object count.
- Owned diagnostics changing only collection cutoff arguments: the runner-focused subset, twenty prior runner controls, the supplied r8 replay, the pre-existing local-module import, and matched linked/direct child-written module cases. Conditional outcomes are labelled throughout; they are not claimed as unchanged-main successes.
- Default-scoring same-process/fresh-process three-seed replay; earlier damaged-artifact consumers; owned equivalent copies; real synthetic feature/label computation; settings, recording constructors, release grammar, budgets, source hashes and current ledger checks.
- Runs completed before the report was written. Final HEAD, clean tracked tree, report structure and report/receipt hash binding are verified. The receipt retains measured outcomes without exploit source. Only the owned r9 scratch prefix is deleted. No real-data run, model fit, walk-forward, network or escalation occurred.
