## Verdict
**BLOCK.**
Reviewed-commit: 1657fbde2eed9269a3a09750f25765d690eb7a7d

The concrete R5-1 teardown reproduction and R5-2 probability reproduction are corrected. The permitted suite passes **143/143**, including all five sandbox-rooted runner subprocess tests. Complete synthetic three-seed outputs still replay at the registered scoring defaults, with differing admitted HEADs.

The revised runner nevertheless has **two measured false greens introduced by this change**. A public session-finish hook wrapper aborts after the new session record has been written and is certified RED. A non-strict XPASS is also certified RED because the structured projection loses its expected-failure status. The same scratch cases are INCONCLUSIVE against the r5 runner. Both use public pytest behavior; neither tampers with private records or pytest internals.

This is Eric's authorized round 6 under the full rules. The item stops; a seventh round requires a new ruling. This verdict admits no seed, run or operational action.

**Scope:** The complete r6 prompt was read first. HEAD matched the requested detached commit, and the tracked tree was clean. No outside context corpus or memory registry, real evaluation data, operational configuration/credentials, box or network was consulted. No accidental outside-directory collection touch was observed. Independent artifacts and test subprocess scratch were routed under `/private/tmp/c2-framing-r6-6exjcf2x`, which is deleted at completion. No escalation or tracked edit occurred.

## Findings

### R6-1 — `trylast` does not put the session record after hook-wrapper cleanup

**Changed code:** `docs/audit/2026-10-06-c2-framing-evidence/mutant_runner.py:50–59,88–114`. **Overstated ledger rule:** R16 says the session record is written “after every other session-finish hook.” Its test covers an ordinary hook, not a wrapper's post-yield work.

The recording hook is an ordinary `pytest_sessionfinish` implementation with `trylast=True`. Hook wrappers surround ordinary implementations; their post-yield work runs afterwards. The record therefore precedes completion of the entire session-finish hook call. If a wrapper then calls public `pytest.exit(..., returncode=1)`, installed pytest 9.0.2 handles that exit in `wrap_session`'s finalization path, sets the process status and writes an exit diagnostic. That path does **not** invoke `pytest_keyboard_interrupt`. The plugin's record survives, and every new classifier condition can match.

**Measured through unchanged checkout runner `main`, its actual plugin and actual pytest subprocesses:** An owned target contained `value = 1`; the mutant changed it to `value = 2`. Two intended test bodies imported that target and asserted `val() == 1`. Both genuinely failed. The scratch conftest supplied this public hook:

```python
@pytest.hookimpl(wrapper=True)
def pytest_sessionfinish(session, exitstatus):
    result = yield
    mark("hook_entered")
    if val() != 1:
        pytest.exit("owned session cleanup stop", returncode=1)
    mark("hook_completed")
    return result
```

`mark` only writes an owned canary. `val` loads the owned target with `importlib.util`; it accesses no pytest object. The reviewer cleared collection-pass canaries before starting the mutated subprocess, so they witness that subprocess alone. The resulting authentic records were:

```text
first:  setup passed, call failed, teardown passed
second: setup passed, call failed, teardown passed
{"session": {"exitstatus": 1, "testsfailed": 2, "testscollected": 2}}
NO interrupted, internalerror or collecterror record
```

Both body canaries and `hook_entered` existed; `hook_completed` did not. Pytest returned 1, stderr was `Exit: owned session cleanup stop`, and stdout ended at `FF ... [100%]`, without a statistics summary. The runner reported:

```text
X1 RED FF ... [100%] [2 intended]
    FAILED test_owned.py::test_first
    FAILED test_owned.py::test_second
NOT RED: none
runner return: 0
target restored: true
```

The diagnostic contains no failure-count wording. The false green comes from a premature structured record, not an output spoof. Running the identical case against a scratch copy of the r5 runner gives **INCONCLUSIVE(exit 1, summary mismatch)** and runner return 1. Thus this is a regression in the new end evidence and removal of its former cross-check.

The two assertion failures are real. What is falsely certified is normal completion of session cleanup. This does not show that any of the author's recorded ledger kills used this hook; it shows that the runner's claimed completion rule is still false. The new ordinary-hook test and R16 kill cannot establish ordering after wrappers. The installed pytest and pluggy implementations independently explain the observed order and missing interrupt callback.

### R6-2 — Removing the text cross-check loses the existing XPASS refusal

**Changed code:** `mutant_runner.py:50–51,79–86,98–114`, and the deletion of r5's `UNCLEAN`/summary checks. The unchanged review rules retain the clean-run requirement; the revision-3 README explicitly excludes xfail and xpass, and the r5 runner still enforced it.

The plugin retains only node, phase and `outcome`. Pytest represents a non-strict unexpected pass as `outcome="passed"` plus `report.wasxfail`. Dropping that field turns an XPASS into an ordinary pass. The session's failed counter does not recover the distinction.

**Measured in another ordinary scratch test module, through actual runner `main`:** The same value mutant made `test_first` fail and this second test unexpectedly pass:

```python
@pytest.mark.xfail(reason="owned known issue", strict=False)
def test_second():
    mark("second_body")
    assert val() == 2
```

All six phase reports existed. The recorder projected the second call to `passed`; the session record was `{exitstatus: 1, testsfailed: 1, testscollected: 2}`. Pytest's genuine statistics line was `1 failed, 1 xpassed`. The new runner nevertheless printed **`X1 RED ... [2 intended]`**, `NOT RED: none`, and returned **0**. The target was restored. The identical reproduction against the r5 runner returned **1 / INCONCLUSIVE(exit 1)**.

This is semantic information lost at the authentic-report boundary, not fabrication of an outcome. It requires no record editing or internal-state mutation. No XPASS was found in the author's ledger output; the finding concerns the runner's admission rule. The new classifier cases have no expected-failure metadata case, and none of the new subprocess tests exercises it.

### R5 requirements and previously closed counterexamples

| Requirement | Repeated result on r6 | Assessment |
| --- | --- | --- |
| R5-1: final teardown calls `pytest.exit("2 failed cleanup checks", returncode=1)` after two real failures | Runner returns 1 / `INCONCLUSIVE(exit 1, interrupted)`. Both bodies and cleanup-entry execute, final cleanup-completion and final teardown report are absent; authentic interrupt record exists. | Concrete teardown case corrected. General normal-session closure remains false in R6-1. |
| Same teardown with `owned teardown stop` | Same interruption refusal, independent of wording. | Corrected. |
| R5-2: nullable-string probability column contains `"bad"`, all raw values non-missing | `profile_problem` returns `probabilities are not numeric`; actual validator raises controlled `RunInvalid` with that reason. A separately injected TypeError from rescoring becomes `RunInvalid: ... cannot be rescored (TypeError)`. | Met. |
| R4-1: rank-1 misses get nullable missing season membership | In both B unit files of the first two seeds: 22 missing rows in 2024 and 24 in 2025 per seed; coherent damaged cards still have P@1=1.0 in both seasons. Validation/aggregate refuse missing season; release returns false; launch/child refuse. A separate namespace without any seed-2 claim creates no seed-2 root. | Remains closed. |
| R4-2: first test prints second's exact `PASSED` line; second aborts before body | Runner returns 1 / `INCONCLUSIVE(exit 1, interrupted)`; second-body canary absent, target restored. | Remains closed. |
| R4-3: seed 3 identity adds an extra key | Direct validation and aggregation refuse `['identity']`; original and owned H12-copy aggregate both refuse. | Remains closed. |
| Sandbox-root collection | All five prescribed runner subprocess tests and independent runner cases pass/execute in their owned roots. | No recurrence. |

The r3 artifact counterexamples also remain closed:

- **Unchanged profiles / edited secondary scorecard:** In the first two seeds, B's P(57) was changed to baseline +0.001 and diffs/results coherently regenerated. All 18 profile hashes stayed unchanged. Validation, release, launch, child and aggregate refuse scorecard/recomputation mismatch.
- **2024 masquerading as 2025:** Coherent cards and metadata do not rescue the retained wrong-season rows. All five boundaries refuse.
- **Foreign HEAD / bogus identities:** Coherently rebinding seed 1 to `02516cfedc812f047295b6f2ab212fd72a944507` refuses at all five boundaries because the exposure is not its ancestor. All-three literal `bogus` identities refuse trusted identity comparison. Real git probes also refuse a nonexistent HEAD and the r6 HEAD under the r5 reviewed closure, which actually changed.
- **Inherited `-x`:** Both intended bodies fail, all six phases and the session record exist, and runner return is 0/RED. Target restoration succeeds.

### Legitimate producer shape, lifecycle and unchanged computation

Three complete synthetic `run` outputs at **`mc_trials=10000, season_length=180`** passed same-process and fresh-interpreter aggregation: **A positive / B negative**. Their actual existing HEADs were `[a00fac4036a00c172ab8eb2f49b60bb04f3f7bef, 1657fbde2eed9269a3a09750f25765d690eb7a7d, 1657fbde2eed9269a3a09750f25765d690eb7a7d]`. Real git ancestry/closure checks returned no problems. Equal HEADs are not required. Ordinary float probabilities, non-missing nullable floats, integer probabilities and boundaries 0/1 pass the predicate; infinity, missing probability, booleans and strings refuse. No new false refusal of producer-shaped complete three-seed evidence was found.

These are local shape/replay probes. Admission, input/feature loading and walk-forward were stubbed using shaped synthetic trusted identities and ten pins; claim writing, retained parquet, labels, scorer, validator and git checks were real. Synthetic report/admission hashes and a mocked owner-release row were used. No actual SIGN/exposure/admission, committed release, live launcher, real walk-forward or model fit was created. The permitted shared-admission tests separately pass their disposable-repository checks.

AST comparison against r5 found the **25 previously closed functions unchanged**; only `profile_problem` and `validate_run` changed among top-level functions. Independent checks repeated:

- **Closed inputs:** Real production feature computation on 48 synthetic PA rows / 24 games, with owned live lookup and raw canaries, then frozen-input installation after changing those canaries. Exactly 24 lookup games retained; zero trapped text reads and zero external-table reads during closed computation. All 16 registered features exactly match; both computations have 28 non-null bullpen values; park drag is all NaN; the self-check is identical on all 48 rows.
- **Labels/settings:** Original miss plus resumed hit remains a miss, resumed-only batter drops, counts are `changed=1, void_dropped=1`, ranks `[1,2]`, hits `[0,1]`. Settings `(20,7)` pass and `(0,10)` refuse. A recording classifier constructor, with no actual fit, receives all three registered seeds and `deterministic=True, force_row_wise=True` through the real training helper. The disclosed official-date availability limit and retrospective void drop/re-rank convention are unchanged.
- **Namespace/release/budgets:** The permitted repeat-claim, foreign-seed, cardinality and canonical-namespace tests pass. Exact Eric/30 release parsing passes; `Ericsson`, zero and decimal overflow refuse. Accounting remains `gate(60.48,45,True)=over_cap`, `gate(30.48,30,False)=ok`, `gate(60.48,30,False)=checkpoint`, `gate(60.48,30,True)=ok`. No budget expenditure or actual owner release was issued.

The author's box inventory/environment facts remain stated and unverified. These source and synthetic checks do not establish box input coverage or real box replay.

### Test strength, boundary and equivalents

The boundary excluding deliberate private-record or pytest-internal tampering is reasonable for the author's own trusted tests. Both new counterexamples use ordinary public hooks/marks; the tests never access the records file, plugin object or pytest internal state. The observer wrappers retain unmodified subprocess results and authentic records; they do not synthesize outcomes. Expanding the exclusion to these public behaviors would change the full review's completion/cleanliness contract.

The **118 framing + 25 admission = 143** passing tests use Python 3.12.13 / pytest 9.0.2 / pluggy 1.6.0. The new teardown case is a useful real-process reproduction. The isolated R11 test correctly covers interruption after every phase report exists, removing its first-run masking. R16 proves ordering only relative to an ordinary hook. R6/R9 establish the need to retain a record, not its position after wrapper cleanup. R7 checks the three numeric counters, which match in both false greens. N23 and N24 target the two probability/rescoring fixes and pass; those fixes also pass the independent witnesses above.

All **108** current anchors occur exactly once; every named node is present in the passing permitted suite, and no test list contains options. Latest statuses for the current spec are **105 RED, G8/H12/N10 SURVIVED, none missing**. Retired F24 is excluded from these counts. The revision-6 full run exited **1**, with `NOT RED: G8,H12,N10,R11`; the isolated R11 resume exited **0**, `NOT RED: none`. The README's “105 of 105 attributable RED” is the combined selected non-equivalent count, not a full 108-mutant exit 0. No tracked-source mutation sweep was run here. Green kills do not cure the two untested runner assumptions.

The three recorded equivalences remain supported over validator-accepted evidence, by source reasoning and owned-copy probes:

- **G8:** Every accepted run's entire pin dictionary equals the same trusted dictionary; its digest is also bound. Removing the later cross-run pin comparison adds no accepted disagreement. A coherently changed pin/digest refuses `admitted pins` in original and G8 copy.
- **H12:** Every accepted identity has exactly five fields and equals the trusted five-field projection. Accepted identities cannot disagree. An extra-key identity refuses in original and H12 copy.
- **N10:** Each nonempty unit belongs wholly to its named integer season, with non-missing date/hit/rank and daily ranks 1..n. Each season therefore has rank-1 rows with binary hits; successful scoring supplies both P@1 keys. Removing the redundant key check adds no accepted missing-season case. Dropping all 2025 rank-1 rows refuses the rank invariant in original and N10 copy.

All three copies accept the ordinary complete aggregate. These copies were mutated only under owned scratch. The README's distinction between message-level kills such as N17 and independent acceptance defenses remains appropriate.

## Required changes

No tracked repair was made. These unresolved requirements do not authorize another review or a run.

1. **Establish completion after the entire session-finish hook call.** An ordinary `trylast` record is insufficient. The public post-yield wrapper abort above must be INCONCLUSIVE regardless of its message and unchanged return code. Preserve the authentic report/counter checks, interrupted-teardown refusal, ordinary-hook refusal, after-last-report interrupt refusal and sandbox roots. A different marker placement needs a real-process test that proves the claimed ordering; absence of an interrupt callback alone cannot prove closure.
2. **Preserve the expected-failure distinction in structured evidence.** Retain/report the information needed to refuse non-strict XPASS rather than project it to ordinary PASSED. Add the actual one-failure/one-XPASS subprocess witness and a matching classifier case. Removing stdout as outcome authority is sound; removing a formerly enforced semantic refusal without replacing its evidence is not.
3. Update the runner's stated rules and mutation tests to match the completed guarantees. The current R16 wording overstates its test. Re-certification, any seventh review and any seed admission require Eric's applicable new ruling/gates; this BLOCK supplies none of them.

## What was run

- Read the complete r6 prompt first, the local r5 report, the r5-to-r6 changes, fix commit message, revised registration/tests/README/specification/output, relevant scorer/model/feature code and installed pytest/pluggy hook implementations. No outside context/memory lookup.
- Confirmed HEAD `1657fbde2eed9269a3a09750f25765d690eb7a7d`, clean tracked tree, and passing `git diff --check 2bcff1293af625b3ca49ec5fcee5a44d89167af5 1657fbde2eed9269a3a09750f25765d690eb7a7d`.
- Permitted command: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run python -B -m pytest tests/scripts/c2_framing tests/scripts/test_c1_admission.py -p no:cacheprovider`, with `TMPDIR`/`MPLCONFIGDIR` routed under the owned prefix. **143 passed, 0 failed, 193.70s, exit 0**. All five real runner subprocess tests pass.
- Owned `runner_probes.py` exercised inherited fail-fast, captured stdout/before-body abort, both prior final-teardown messages, ordinary session-finish abort, after-last-report interrupt, post-yield session-finish abort and XPASS. Owned r5 runner copy / `old_finish_probe.py` established the two before/after regressions. Actual runner `main`, subprocess argv and plugin were unchanged; every owned target was restored.
- Owned `profile_probes.py` / fresh replay repeated r3/r4/r5 artifact counterexamples, real git checks, direct row predicates, controlled scorer failure, and three producer-shaped runs at registered scoring defaults. Owned `regression.py` checked 25 unchanged ASTs, real synthetic features/closure, labels, recording constructors, settings, release, budgets, anchors/nodes and ledger statuses. Owned `equivalents.py` applied G8/H12/N10 only to disposable copies. Executed with the checkout's `.venv/bin/python -B`, `PYTHONPATH=.` and owned temp routing; no real walk-forward or fit.
- Source hashes: `screen.py` = `7be92720a90ac488cecb03ae75c337145665e8630831c0f07e83b8d19f6bdc94`; registration = `ab44882111d9dba39ddcf3fed6344b449c30ab25109ee365f48d4d31126d128a`; runner = `8203b3ea94c40a812cf2f9eb90267a0111afee96a8db931acbca3cf6e010d7de`. All three are unchanged after fix commit `a00fac4…`; later commits are specifications/evidence/tests.
- Only this report and `receipt-r6.json` are written in the checkout. Owned `/private/tmp/c2-framing-r6-6exjcf2x` scratch is deleted at completion. No real evaluation-data read, operational configuration/credentials read, box/network/SSH/gh, escalation, tracked edit, commit or push. Final report structure/hash, receipt binding, scratch deletion, HEAD and clean tracked status are checked before the completion marker; the report is not edited after its final hash.
