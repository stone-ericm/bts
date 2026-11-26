## Verdict
**BLOCK.**
Reviewed-commit: 2bcff1293af625b3ca49ec5fcee5a44d89167af5

All three concrete round-four counterexamples are corrected, and the permitted suite passes **133/133**, including the three runner subprocess tests in this sandbox. The new identity shape closes H12's earlier distinguishing case. The three-seed metadata-HEAD lifecycle and full scorecard replay still pass at the registered scoring defaults in a fresh process.

The new recording plugin nevertheless certifies an **aborted teardown as a normally completed RED session**. Its end marker records unconfiguration, which pytest also performs after interruption. A final abort message containing matching failure counts satisfies the runner's remaining text cross-check. This is within the stated boundary: the reproduction changes neither private records nor pytest internals. Both assertion failures are real; the false certification is that the session finished cleanly.

This is Eric's authorized round 5, the last under the current ruling. The item stops. A sixth review requires a new ruling from Eric. This verdict admits no seed or run.

**Scope:** The complete r5 prompt was read first. HEAD matched the requested detached commit and the tracked tree was clean. No outside context corpus, memory registry, real evaluation data, operational configuration/credentials, box or network was consulted. No accidental external-directory collection touch was observed this round. All independent artifacts and test subprocess scratch were routed under `/private/tmp/c2-framing-r4-r5-xf7dufi6`, following the prompt's literal permitted `c2-framing-r4-*` prefix, and are deleted at completion. No escalation or tracked edit occurred.

## Findings

### R5-1 — Unconfiguration is not a normal-end witness; an interrupted teardown can still be RED

**New runner:** `docs/audit/2026-10-06-c2-framing-evidence/mutant_runner.py:33–44,59–89`. **Relevant measured pytest behavior:** installed pytest 9.0.2, `_pytest/main.py:wrap_session` and `_pytest/config/__init__.py:Config._ensure_unconfigure`.

The plugin writes `{"end": true}` in `pytest_unconfigure`. Pytest calls unconfiguration from its cleanup path after `pytest.exit`, including a requested return code of 1. The marker therefore proves that cleanup reached this hook, not that the test session ended normally. `classify` requires each intended call report but does not require each intended teardown report. It checks only for a *reported* setup/teardown failure; an aborted teardown can have no report at all.

The remaining cross-check takes the last stdout line and searches anywhere in it for `N failed` and `N passed`. It does not require that line to be pytest's statistics summary. An exit diagnostic containing those words can supply the required counts.

**Measured through unchanged runner `main`, its actual recording plugin and actual pytest subprocesses:** An owned scratch mutant changed `value = 1` to `value = 2`. The spec named exactly two nodes. Both bodies asserted the original value and failed. The second used this ordinary yield fixture:

```python
@pytest.fixture
def gate():
    yield
    pathlib.Path(__file__).with_name("cleanup_entered").write_text("entered")
    if val() != 1:
        pytest.exit("2 failed cleanup checks", returncode=1)
    pathlib.Path(__file__).with_name("cleanup_completed").write_text("finished")
```

The second body-entry canary and `cleanup_entered` existed; `cleanup_completed` did not. Pytest's own private records contained:

```text
first:  setup passed, call failed, teardown passed
second: setup passed, call failed, NO teardown report
{"end": true}
```

The actual final stdout line and runner result were:

```text
!!!!!!!!!!!!!!!! _pytest.outcomes.Exit: 2 failed cleanup checks !!!!!!!!!!!!!!!!
X1 RED ... [2 intended]
NOT RED: none
runner return: 0
target restored: true
```

The exit diagnostic's `2 failed` matched the two call failures, so the cross-check accepted. Changing only the diagnostic to `owned teardown stop` gave **INCONCLUSIVE(exit 1, summary mismatch)** and runner return 1, with the same reports and interrupted cleanup. Thus the result depends on the abort's wording, not on a verified normal end.

No report-file content, plugin object, pytest hook or internal state was modified by the tests. The fixture used public `pytest.exit`, just as the round-four before-body reproduction does. The runner was imported directly from the checkout, passed an owned `root`, and used its own interpreter and subprocess arguments without replacement. Its scratch target was restored after every case.

This does not fabricate the assertion failures or demonstrate that the recorded 96 ledger kills used an interrupted session. It demonstrates that the revised runner does not enforce its own clean-completion condition. Recording call outcomes fixes R4's outcome spoof; recording unconfiguration does not establish closure of the session and its final teardown.

### R5-2 — Nonblocking numeric-predicate gap: coercion can create missing probabilities after the missing-value check

**New helper:** `screen.py:506–520`, particularly `pd.to_numeric(..., errors="coerce")` followed by `.all()`.

On a retained profile whose probability column is pandas nullable `string`, I replaced one value with `"bad"` and rendered the remaining probabilities as strings. All raw values were non-missing. Conversion created a nullable numeric missing value; the range comparison's `.all()` skipped it. Direct `profile_problem(part, 2025)` returned **None**, despite the nonnumeric probability.

The enclosing actual `validate_run` then raised **TypeError: unsupported operand type(s) for +: 'float' and 'str'** while recomputing the scorecard. It did not accept the run. This is not an observed seed release, positive disposition or legitimate-producer false refusal, and it is not the reason for BLOCK. It is a narrower defect in the new predicate's promise to establish scoreable bounded probabilities; it also leaves this malformed input outside the advertised `RunInvalid` refusal path. A directly missing nullable probability, infinite probability and duplicate ranks all refused in independent probes.

### Disposition of R4-1 to R4-3 and the sandbox failure

| Round-four requirement | Repeated reproduction on r5 | Assessment |
| --- | --- | --- |
| R4-1: reject nullable missing season membership before completion, release or disposition | In both B unit files of the first two seeds, rank-1 misses again received nullable missing seasons: 22 rows in 2024 and 24 in 2025 per seed. Coherent damaged cards still reported P@1 of 1.0 in both seasons. Actual validation and aggregation now raise `RunInvalid: missing values in ['season']`; release returns false, launch and child raise `SystemExit`. A separate child namespace with no prior seed-2 claim also refuses and creates no seed-2 root. | Concrete counterexample and completion requirement corrected. |
| R4-2: a captured `PASSED` line cannot count as the second test's execution | Actual first test prints the second's exact `PASSED` line, then fails; the second setup calls `pytest.exit` before its body. Runner returns 1 / `INCONCLUSIVE(exit 1, 1 not run)`; the second body canary is absent and the target is restored. Records contain only the first node's reports. | Original outcome spoof corrected. Normal/complete-session requirement remains open in R5-1. |
| R4-3: exact identity shape and H12's domain | Adding `unrecognized_extra` to only seed 3 now raises `RunInvalid: ['identity']` at direct validation, original aggregate and an owned H12-mutant aggregate. | Met. H12 now has the claimed equivalence over validator-accepted manifests. |
| Rootdir/working-directory collection failure | All three prescribed runner subprocess tests pass with their scratch roots. Independent runner cases also collect and execute directly under their owned roots. | Corrected in this sandbox; no repeat of the r4 denied external-directory stat was observed. |

### Previously closed behavior and false-refusal checks

The r3 counterexamples remain closed in independent artifact probes:

- **Unchanged profiles / edited secondary card:** B's exact P(57) was changed to baseline +0.001 in the first two cards, with regenerated diffs/results and all 18 profile SHA-256 values unchanged. Validation and aggregation refuse full-scorecard/profile mismatch.
- **2024 standing in for 2025:** Validation, release, launch, child and aggregate all refuse the unit's wrong season. Metadata names cannot substitute for actual season evidence.
- **Foreign HEAD / bogus identities:** Coherently rebinding seed 1 to `02516cfedc812f047295b6f2ab212fd72a944507` refuses at all those boundaries because the exposure is not its ancestor. All-three literal `bogus` identities refuse the trusted identity comparison. Actual git probes also refuse a nonexistent HEAD and a descendant with a changed executable closure.
- **Inherited `-x`:** Both named bodies execute and fail; the second canary exists, all six setup/call/teardown reports and the end marker are retained, runner returns 0/RED, and the scratch target is restored.

For legitimate local shape/replay, I produced three complete synthetic `run` outputs at **`mc_trials=10000, season_length=180`**. HEADs were existing commits `[9fdc800d2e0adcd8b79de5eb096c8864e977bbef, 2bcff1293af625b3ca49ec5fcee5a44d89167af5, 2bcff1293af625b3ca49ec5fcee5a44d89167af5]`. Same-process aggregation and a fresh interpreter both accepted **A positive / B negative**. Each real git ancestry/closure check returned no problem. Equal HEADs are not required, and no new false refusal of ordinary producer-shaped profiles or deterministic scorecard replay was found.

These orchestration probes used shaped synthetic expected identities and pins, with the reviewed/exposure commit `9fdc800…` and synthetic report/admission hashes. Admission and input/feature loading were stubbed; walk-forward was a fake callback. Claim writing, retained parquet, labels, scorer, validator and git checks were real. No actual SIGN/exposure/admission, committed owner release, live launcher or real model evaluation was created or executed. The real shared admission tests separately passed their disposable-repository checks.

AST comparison against r4 found **25 unchanged functions**, including all 22 previously closed helpers plus `run`, `head_admitted` and `_canon`. The source change is confined to the new predicate and identity/validator checks. Independent regression probes repeated:

- **Closed inputs:** Real feature computation on 48 synthetic PA rows and 24 games, using an owned lookup/cache and 2026 raw canary. Freeze retained exactly 24 games. After changing live canaries and installing the closed inputs, there were zero trapped `Path.read_text` calls and zero table reads. All 16 registered features matched exactly; both computations had 28 non-null bullpen values, park drag was all NaN, and self-check was identical on 48 rows.
- **Labels/settings:** Original miss plus resumed hit remained a miss; resumed-only batter dropped. Counts `changed=1, void_dropped=1`, ranks `[1,2]`, hits `[0,1]`. Effective settings `(20,7)` pass and `(0,10)` refuse. The real training helper with a recording constructor and no fit used all three registered seeds and both deterministic/row-wise flags. The disclosed official-date availability limit and retrospective void drop/re-rank convention remain unchanged.
- **Namespace/release/budgets:** The permitted suite's repeat-claim, foreign-seed, cardinality and canonical-namespace cases pass. Exact Eric/30 release parsing passes; `Ericsson`, zero and decimal overflow to infinity refuse. Independent accounting remains `gate(60.48,45,True)=over_cap`, `gate(30.48,30,False)=ok`, `gate(60.48,30,False)=checkpoint`, `gate(60.48,30,True)=ok`. No actual budget expenditure or owner release was issued.

The author's box inventory/environment facts remain stated and unverified. Local synthetic replay does not establish real-input coverage or exact replay on the box. No regression of the previously closed computation, inputs, labels, settings, namespace, budgets, release grammar or metadata lifecycle was found.

### Runner boundary, test strength and equivalents

The stated exclusion of deliberately edited private records or modified pytest internals is acceptable for a ledger of the author's own trusted tests. This plugin is an execution witness within that trusted interpreter, not isolation from hostile test code. Its claims should stay bounded accordingly. R5-1 requires neither excluded action: authentic reports plus a public fixture abort are sufficient. Broadening the exclusion to every adverse public fixture behavior would abandon the incomplete-execution requirement the runner is supposed to enforce.

The permitted suite is **108 framing + 25 admission = 133 passed in 185.84s** on Python 3.12.13 / pytest 9.0.2. It now covers the real before-body stdout counterexample, exact identity keys and scored-row damages. Its orchestration fixture still uses 200 Monte Carlo trials and stubs admission/input computation/walk-forward; the additional default-scoring and real-feature probes above address those respective local limits.

All **99** current ledger anchors occur exactly once; every test list names node ids and contains no options. Latest recorded statuses are **96 RED, G8/H12/N10 SURVIVED, none missing**. The full recorded runner exit is **1**, with `NOT RED: G8,H12,N10`. “96 of 96 attributable RED” describes the selected non-equivalent mutants, not an overall runner exit 0. No tracked-source mutation sweep was run in this review.

The three equivalence arguments now hold under the current validator/scorer domain, with source reasoning supported by owned-copy probes:

- **G8:** Every accepted run's entire ten-pin dictionary equals the same trusted pins, with its digest bound as well. Removing the later cross-run pin comparison adds no accepted disagreement. A coherent different pin/digest still refuses `admitted pins` in both original and G8 copy.
- **H12:** Every accepted manifest identity has exactly five fields, and every value equals the same trusted five-field projection. Dictionaries therefore cannot disagree. The prior extra-key witness now refuses in both original and H12 copy.
- **N10:** Every unit is nonempty, wholly of its named integer season, has non-missing date/hit/rank and daily ranks 1..n. Each season therefore has actual rank-1 rows with binary labels. The scorer must supply both registered P@1 keys if recomputation succeeds. Removing the redundant key-coverage check accepts no otherwise-valid missing season. Removing all 2025 rank-1 rows still refuses the rank invariant in both original and N10 copy. All three copies accept the ordinary complete aggregate.

The README appropriately identifies some kills as message-level distinctions rather than independent defenses: N17's missing-value mutant can still be refused by later dtype/rank/hit checks, while a specific refusal-message assertion turns RED. Those are assertion kills, not separate acceptance-boundary kills.

R6 tests hand-built records with no end marker; R9 tests deletion of the marker. Neither establishes that the real end marker distinguishes an interrupted session from a normal one. R7 tests mismatched numeric counts but not an abort diagnostic containing matching counts. The new structured-report tests are useful, yet leave precisely the real teardown/closure case in R5-1 uncovered.

## Required changes

No tracked repair was made. These are unresolved requirements and do not authorize reopening or a sixth review.

1. **Make normal completion a structured fact.** Record and reject public pytest interruptions/internal errors, and require complete expected setup/call/teardown reports with permitted outcomes before RED/SURVIVED. Unconfiguration alone cannot attest normal completion. Merely moving the marker to `pytest_sessionfinish` is insufficient: pytest invokes that hook from `wrap_session`'s `finally` after interruption too. Add the actual last-teardown `pytest.exit(..., returncode=1)` witness, including a diagnostic with `2 failed cleanup checks`; it must be INCONCLUSIVE regardless of the wording. Keep the corrected before-body, inherited-`-x`, exact-node and sandbox-root behavior.
2. **Keep stdout as a bounded cross-check.** An exit diagnostic cannot be parsed as the statistics summary, and matching counts cannot override missing teardown or a structured interruption witness. Preserve real report provenance and the accepted private-records/internals boundary; this repair need not attempt to sandbox malicious tests.
3. **Nonblocking predicate hardening:** Explicitly reject missing values created by probability conversion, or require an appropriate numeric probability dtype, before evaluating the range. Add the nullable-string coercion case. It should return a profile problem and a controlled `RunInvalid`, rather than claim scoreability and subsequently escape as TypeError. No full-validator false green was measured for this case.

## What was run

- Read the complete r5 prompt first, the local r4 report, the r4-to-r5 diff, revised registration/tests/README/specification/output, relevant scorer/model code and installed pytest cleanup source. No outside context or memory lookup.
- Confirmed HEAD `2bcff1293af625b3ca49ec5fcee5a44d89167af5`, clean tracked tree and passing `git diff --check 04439f9eb44d527a52c815d552d12a4e6fb5583d 2bcff1293af625b3ca49ec5fcee5a44d89167af5`.
- Permitted command: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run python -B -m pytest tests/scripts/c2_framing tests/scripts/test_c1_admission.py -p no:cacheprovider`, with `TMPDIR`/`MPLCONFIGDIR` routed under the owned prefix. **133 passed, 0 failed, 185.84s, exit 0**. All three runner subprocess tests pass in this sandbox.
- Owned `runner_probes.py` exercised inherited fail-fast, the original captured outcome/before-body abort, and both last-teardown abort diagnostics using actual checkout runner `main` and its unmodified subprocess/plugin path. All scratch targets were restored. Private reports were read only by the runner/reviewer, never edited by a test.
- Owned `profile_probes.py` and its fresh replay exercised R4-1/R4-3, all prior r3 counterexamples, real git checks, direct row predicates and three producer-shaped runs at registered scoring defaults. Owned `equivalents.py` applied G8/H12/N10 only to disposable source copies. Owned `regression.py` verified 25 function ASTs, real synthetic features/closure, labels, recording constructors, settings, release and accounting. Executed via the checkout's `.venv/bin/python -B` with `PYTHONPATH=.`; no real walk-forward or fit.
- Checked all 99 current anchors/test lists and latest recorded ledger statuses. `screen.py`, registration and runner are unchanged after fix commit `9fdc800…`; later commits concern specifications/evidence.
- SHA-256: `screen.py` = `c1f31490d75fb4b6806c751daa3d424f20ca14f383a6d10f3d0dfa88e289f475`; registration = `94221446dbd39a9971fbf221636dc2aab07769ea067abf6458534f60749f1efd`; runner = `d6e3670013137fcada12904da1199bad45d5fc71ae6ae6e47c52999ebac8366c`.
- Only this report and `receipt-r5.json` are written in the checkout. All owned `/private/tmp/c2-framing-r4-r5-xf7dufi6` scratch is deleted at completion. No real evaluation-data read, operational credentials/configuration read, box/network/SSH/gh, escalation, tracked edit, commit or push. Final report structure/hash, receipt binding, scratch deletion, HEAD and clean tracked status are verified before the completion marker; the report is not edited after its final hash.
