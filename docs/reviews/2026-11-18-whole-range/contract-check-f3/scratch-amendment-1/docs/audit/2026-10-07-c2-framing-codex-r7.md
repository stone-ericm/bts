## Verdict
**BLOCK.**
Reviewed-commit: 49323519c80587cbd38997a6266f14be6c78b6ca

Both concrete r6 counterexamples now refuse. The permitted suite passes **157/157**, including all twelve runner subprocess tests. A separate driver run of the 36 actual runner/classifier nodes produces complete clean evidence with zero late plugins. The three-seed synthetic metadata-HEAD lifecycle and default scorecard replay still pass.

The redesigned runner nevertheless certifies an aborted session as RED when a late plugin unregisters **itself** during cleanup. The late-plugin field measures the final registry difference, not registrations that occurred. A late tryfirst wrapper can therefore run outside the sentinel, abort after its completion record, and disappear before the final census. This defeats the new ordering defense without touching the sentinel, recorder, evidence file, reports or pytest internals.

This is Eric's authorized round 7 under the full rules. The item stops. An eighth round needs a new ruling; this verdict admits no seed, run or operational action.

**Scope:** The complete r7 prompt was read first. HEAD matched the requested detached commit and the tracked tree was clean. No outside context corpus or memory registry, real evaluation data, operational configuration/credentials, box or network was consulted. No accidental outside-directory collection touch was observed. Independent artifacts and subprocess scratch were routed under `/private/tmp/c2-framing-r7-ug6l7145`, which is deleted at completion. No escalation or tracked edit occurred.

## Findings

### R7-1 — A final plugin census cannot establish that no plugin registered late

**New code:** `docs/audit/2026-10-06-c2-framing-evidence/mutant_runner.py:81–86,98–105,148–151`. **Affected guarantees:** outermost sentinel, completed session finish, and refusal of any plugin registered after the sentinel. **Relevant new mutant:** R18.

The recorder snapshots plugin object ids after registering the sentinel. After `pytest.main` returns, the driver computes:

```python
late = len({id(p) for p in rec.config.pluginmanager.get_plugins()}
           - rec.at_sentinel)
```

This finds late plugins still registered at that final instant. It loses a late plugin that removes its own registration during normal lifecycle cleanup. Such a plugin was already able to run outside the sentinel: the last-registered tryfirst wrapper is outermost. The ordering claim is correct, but that makes complete late-registration evidence necessary.

**Measured through unchanged checkout runner `main`, its unchanged driver and actual pytest subprocesses:** An owned scratch target contained `value = 1`; its mutant changed it to `value = 2`. The intended set was exactly two tests whose bodies asserted `val() == 1`. Both bodies genuinely failed. The conftest registered this owned plugin during the first test's setup, after collection and sentinel registration:

```python
class OwnedLate:
    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus):
        result = yield
        mark("hook_entered")
        if val() != 1:
            pytest.exit("owned late plugin cleanup stop", returncode=1)
        mark("hook_completed")
        return result

    def pytest_unconfigure(self, config):
        config.pluginmanager.unregister(self)
        mark("unregistered")

def pytest_runtest_setup(item):
    if item.name == "test_first" and val() != 1:
        item.config.pluginmanager.register(OwnedLate(), "owned-late")
        mark("registered")
```

`val` imports only the owned target through `importlib.util`; `mark` writes only owned canaries. Registration and self-unregistration use the public plugin-manager APIs. There is no access to a runner object, its evidence file, a private pytest attribute or another plugin's registration. In particular, this does **not** unregister the sentinel. Collection-pass canaries were cleared before the mutated subprocess, so the observed canaries belong to that subprocess.

The actual execution order was:

1. The late plugin registered; both test bodies failed with AssertionError and completed their setup/call/teardown reports.
2. Its outer wrapper yielded to the sentinel. The sentinel completed and recorded the matching counters.
3. The late wrapper resumed and called public `pytest.exit` before `hook_completed`.
4. Pytest handled that session-finish exit, printed an exit diagnostic, and continued unconfiguration. That exit path does not invoke `pytest_keyboard_interrupt`.
5. The owned plugin unregistered itself. `pytest.main` returned 1; the driver wrote authentic evidence whose final registry difference was zero.

The retained evidence was:

```text
rc: 1
first:  setup passed, call failed, teardown passed; wasxfail false
second: setup passed, call failed, teardown passed; wasxfail false
calls: both AssertionError
finish: {exitstatus: 1, testsfailed: 2, testscollected: 2}
late_plugins: 0
marked: []
NO interrupted, internalerror or collecterror record
```

Both body-entry canaries, `registered`, `hook_entered` and `unregistered` existed; `hook_completed` did not. Pytest returned **1**, stderr was **`Exit: owned late plugin cleanup stop`**, and the runner printed:

```text
X1 RED 2 failed in 0.01s [2 intended]
    FAILED test_owned.py::test_first
    FAILED test_owned.py::test_second
NOT RED: none
runner return: 0
target restored: true
```

The assertion failures and reports are authentic. What is false is the certification of a normally completed session with no late registration. The sentinel's record is valid for the inner call it surrounded, but it did not surround the new outer wrapper's completion.

Two controls isolate the defect. A late plugin left registered gives **INCONCLUSIVE(exit 1, plugins registered late)**. A plugin with the same registration/self-cleanup but **no abort** still gives **RED with `late_plugins=0`**, violating the explicitly stated any-late-registration rule even without the session abort. Thus the gap is loss of registration history, not a fabricated failure or counter mismatch.

The installed pluggy 1.6.0 registration/unregistration implementations and pytest 9.0.2 `wrap_session` explain the measured ordering and cleanup. The stated boundary is compatible with excluding sentinel removal and private-state edits; ordinary self-cleanup of the newly registered plugin is neither. R18's real-process test registers an object that persists, and its unit case hand-supplies `late_plugins=1`. Neither tests whether the producer of that field retains registrations after the plugin disappears.

This does not show that the author's recorded 113 attributable ledger kills used such a plugin. It demonstrates a new defect in the redesigned registry/ordering evidence. Changing the boundary to exclude public registration/self-unregistration would abandon the specific late-plugin rule rather than establish it.

### R6 requirements and the redesign's other claims

| Required reproduction or claim | Measured result on r7 | Assessment |
| --- | --- | --- |
| R6-1: post-yield session-finish wrapper abort | Runner return 1 / `INCONCLUSIVE(exit 1, session finish incomplete)`; authentic evidence contains `finish={raised: Exit}`. Both failures and all six phase reports exist; cleanup-completion canary absent. | Concrete r6 case corrected. |
| Same wrapper declared tryfirst | Same refusal. | Last registration places the sentinel outside pre-existing tryfirst wrappers, as claimed. Late registrations remain open in R7-1. |
| R6-2: one genuine failure plus one non-strict XPASS | Runner return 1 / `INCONCLUSIVE(exit 1, marked xfail or skip)`; second call remains `passed` with `wasxfail=true`, and its node is marked. | Corrected. |
| Runtime xfail added during the second body | Runner return 1 / `INCONCLUSIVE(exit 1, xfail or skip)`; report retains expected-failure status even though the collection-time marked list is empty. | Corrected at the report boundary. |
| Unconfiguration abort | Driver writes no evidence; runner return 1 / `INCONCLUSIVE(exit 1, no normal end)`. | Corrected for an abort escaping `pytest.main`. |
| Report/observation agreement | Real rewritten-report subprocess test and classifier cases pass, including flipped outcome and unobserved call. Ordinary failures record AssertionError; ordinary passes record null. | The comparison enforces observed raised-versus-not-raised agreement, within the stated trusted-object boundary. |
| Normal execution evidence / false refusals | Independent actual driver run of all 14 runner/classifier test functions, 36 nodes: pytest exit 0, SURVIVED, all phases/call observations and matching finish counters, zero late plugins. Ordinary scratch mutant with inherited `-x`: both failures executed, complete evidence, RED. A complete clean pass is SURVIVED. | No normal-run false refusal found in these probes. |

The prose saying an “abort anywhere” leaves no evidence is too broad. The corrected wrapper and ordinary session-finish aborts **do** return through pytest's handler and produce evidence with `finish={raised: Exit}`; the classifier then refuses it. Only an abort escaping `pytest.main` prevents the subsequent write. This wording is not a second measured false green. Evidence existence is a driver-return witness, not by itself a clean-session witness; correctness depends on the complete sentinel and event observations. R7-1 defeats that latter dependency.

### Earlier closed behavior and legitimate stage-one replay

The prior runner cases remain closed in independent subprocess probes:

- Both final-teardown messages, `2 failed cleanup checks` and `owned teardown stop`, give **INCONCLUSIVE(exit 1, interrupted)**. Both bodies run; final cleanup is incomplete and its teardown report is absent.
- An ordinary session-finish hook abort gives **INCONCLUSIVE(exit 1, session finish incomplete)**.
- Public exit after the final teardown report gives **INCONCLUSIVE(exit 1, interrupted)** despite all phase reports existing.
- The captured-stdout spoof followed by a second before-body abort gives **INCONCLUSIVE(exit 1, interrupted)**; the second body never runs.
- Inherited `-x` is removed; both intended failures and all phases execute. Every owned target is restored. All twelve prescribed runner subprocess tests pass in their own roots in this sandbox.

The artifact counterexamples from r3 to r5 were also repeated:

- **Nullable season membership:** In both B unit files of the first two seeds, 22 rank-1 misses in 2024 and 24 in 2025 per seed received missing nullable seasons. Coherent damaged cards still report P@1=1.0 in both seasons. Validation/aggregate refuse missing season, release returns false, launch/child refuse. A fresh namespace without a seed-2 claim creates no seed-2 root.
- **Extra identity key:** Seed 3's extra key refuses at direct validation, aggregate and the owned H12 copy.
- **Unchanged profiles / edited P(57):** First-two B cards were changed to baseline +0.001 with coherent diffs/results, all 18 profile hashes unchanged. Validation, release, launch, child and aggregate refuse full-scorecard/profile mismatch.
- **2024 standing in for 2025 / foreign HEAD:** Coherent cards cannot rescue wrong-season rows; coherently rebinding seed 1 to `02516cfedc812f047295b6f2ab212fd72a944507` cannot rescue an unadmitted ancestor relationship. All five consumers refuse each witness. All-three bogus identities also refuse trusted identity comparison.
- **R5 probability/scorer refusal:** Nullable strings containing `"bad"` are rejected as nonnumeric by the predicate and actual validator. An injected rescoring TypeError becomes RunInvalid. Missing probabilities and infinity refuse; ordinary float/nullable float/integer probabilities and boundaries 0/1 pass the predicate.

Three complete synthetic `run` outputs at **`mc_trials=10000, season_length=180`** again pass same-process and fresh-interpreter aggregation, **A positive / B negative**. Their actual existing HEADs are `[165929e2ff350ad9dc99b6fe18d7234bc22dd70b, 49323519c80587cbd38997a6266f14be6c78b6ca, 49323519c80587cbd38997a6266f14be6c78b6ca]`; real git checks return no problems. Missing HEAD refuses; the r7 HEAD under the r6 reviewed closure refuses the actual registration-file change. The equal-HEAD refusal has not returned.

These local replay probes stub admission, input/feature loading and walk-forward with synthetic trusted identities/pins and a mocked owner-release row. Claim writing, retained parquet, labels, scorer, validator and git checks are real. No actual SIGN/exposure/admission, committed release, live launcher, real walk-forward or model fit was created. The shared-admission tests separately pass their disposable-repository checks. No new false refusal of complete producer-shaped three-seed evidence was found.

`screen.py` is byte-identical to r6, including all **35** top-level functions. Independent regression probes nevertheless repeated the closed behavior:

- **Closed inputs:** Real feature computation on 48 synthetic PA rows / 24 games, with owned live lookup and raw canaries, followed by frozen-input installation after changing the canaries. Exactly 24 games retained; zero trapped text/table reads in the closed computation; all 16 registered features exactly match. Both computations have 28 non-null bullpen values; park drag is all NaN; self-check identical on all 48 rows.
- **Labels/settings:** Original miss plus resumed hit remains a miss; resumed-only batter drops. Counts `changed=1, void_dropped=1`, ranks `[1,2]`, hits `[0,1]`. `(20,7)` passes; `(0,10)` refuses. Recording classifier constructors through the real training helper receive all three registered seeds with deterministic and row-wise flags true, without an actual fit. The disclosed official-date availability limit and retrospective void drop/re-rank convention are unchanged.
- **Namespace/release/budgets:** The permitted repeat-claim, foreign-seed, cardinality and canonical-namespace cases pass. Exact Eric/30 release parses; Ericsson, zero and decimal overflow refuse. Budget outcomes remain over_cap, ok, checkpoint and ok respectively for `(60.48,45,True)`, `(30.48,30,False)`, `(60.48,30,False)`, `(60.48,30,True)`. No actual expenditure or release was issued.

The author's box facts remain stated and unverified. Local source/synthetic evidence does not establish real-input coverage or box replay.

### Test and mutant strength; equivalents

The permitted suite is **132 framing + 25 admission = 157 passed**, Python 3.12.13 / pytest 9.0.2 / pluggy 1.6.0. The twelve real runner cases now cover both r6 witnesses, tryfirst ordering, runtime expected failures, a persistent late plugin, rewritten reports and unconfiguration. The independent 36-node driver run adds an actual complete-evidence check on these tests under the redesigned execution path. No tracked-source mutation sweep was run here.

All **116** current anchors occur exactly once. Every named node appears in the passing permitted suite; no list contains options. Latest current-spec statuses are **113 RED, G8/H12/N10 SURVIVED, none missing**; retired F24 is excluded. The single revision-7 full recorded run exits **1**, `NOT RED: G8,H12,N10`. “113 of 113 attributable RED” describes the non-equivalent selection, not a full 116-mutant exit 0.

R16/R17 substantiate registration and priority relative to already loaded hooks; R20's runtime-mark witness isolates expected-failure recording; R21/R22 check authentic call/report agreement; R23/R24 check driver status and incomplete finish. R18 lacks the independently varying cleanup component: deleting the refusal branch tests consumption of `late_plugins`, while its producer loses registrations before consumption. A green ledger cannot establish an untested event-history invariant.

The recorded equivalents remain supported over validator-accepted evidence, by source reasoning and owned-copy probes:

- **G8:** Every accepted run's complete pin dictionary equals the same trusted dictionary, with its digest bound. A coherent wrong pin/digest refuses admitted pins in original and G8 copy; removing the later cross-run pin comparison adds no accepted disagreement.
- **H12:** Every accepted identity has exactly five fields and equals the trusted projection. An extra-key identity refuses in original and H12 copy; accepted identities cannot disagree.
- **N10:** Every nonempty unit wholly belongs to its integer season, with non-missing scored values and daily ranks 1..n. Each season has real rank-1 binary-hit rows, so successful scoring supplies both P@1 keys. Dropping all 2025 rank-1 rows refuses ranks in original and N10 copy; the later coverage check is redundant within this domain.

All three owned copies accept the ordinary complete aggregate. The README's distinction between message-level kills such as N17 and separate acceptance defenses remains appropriate.

## Required changes

No tracked repair was made. These requirements authorize neither another review nor a run.

1. **Retain late registration as an event, through cleanup.** A final registry snapshot is insufficient. A plugin registered after the sentinel must remain disqualifying after its own public self-unregistration. The aborted and non-aborted self-cleanup controls above must both refuse the stated late-registration rule; the aborted one must never be RED or SURVIVED. Preserve the normal unmarked driver run with zero late events.
2. **Exercise the producer of the late-plugin evidence.** Add the actual register/outer-wrapper-abort/self-unregister lifecycle, not only a persistent object or hand-built `late_plugins=1`. Test the observation's retention independently from its classifier check. Preserve corrected r6 wrappers/XPASS, interrupts, complete phases, report agreement, status/counter checks and sandbox roots.
3. **State the driver guarantee precisely.** Evidence is written after `pytest.main` returns. Caught pytest aborts may produce evidence and require explicit refusal; only an abort escaping that call prevents the write. The ordering claim is conditional on complete detection of later registrations. Do not broaden the trusted-object exclusion to ordinary cleanup of a plugin's own public registration.

Any eighth review, re-certification or seed admission remains subject to Eric's applicable new ruling and earlier admission/exposure gates. This BLOCK supplies none.

## What was run

- Read the complete r7 prompt first; reviewed the r6-to-r7 diff, fix message, local r6 report/context, revised registration/tests/README/spec/output and relevant installed pytest/pluggy implementations. No outside context or memory lookup.
- Confirmed HEAD `49323519c80587cbd38997a6266f14be6c78b6ca`, clean tracked tree and passing `git diff --check 1657fbde2eed9269a3a09750f25765d690eb7a7d 49323519c80587cbd38997a6266f14be6c78b6ca`.
- Permitted command: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run python -B -m pytest tests/scripts/c2_framing tests/scripts/test_c1_admission.py -p no:cacheprovider`, with TMPDIR/MPLCONFIGDIR under the owned prefix. **157 passed, 0 failed, 194.98s, exit 0**; all twelve runner subprocess tests pass.
- Owned `runner_probes.py`: fifteen actual runner-main cases, including both r6 witnesses, earlier abort/spoof/fail-fast cases, clean pass, persistent late plugin and both self-cleanup controls. Unchanged driver/subprocess arguments and authentic evidence, all targets restored. Owned `normal_driver.py`: all 36 actual runner/classifier nodes, complete evidence / SURVIVED / zero late plugins.
- Owned `profile_probes.py` / fresh interpreter: all artifact witnesses above, default-scoring three-seed replay, real git checks and G8/H12/N10 disposable source copies. Owned `regression.py`: unchanged source, real synthetic features/closure, labels, recording constructors, settings, release, budgets, anchors/nodes/statuses. Executed with checkout `.venv/bin/python -B`, PYTHONPATH and owned temp routing. No real walk-forward or model fit.
- Source SHA-256: screen = `7be92720a90ac488cecb03ae75c337145665e8630831c0f07e83b8d19f6bdc94`; registration = `7f1c6ff6f681242b92a033edeba2daa43e4eca5e7d6679f08f3cad24d7ead5a4`; runner = `1882ce3d903c9ffb03a5cb72f35b51b1d90174c354cfbd50594e3c2185017532`. All three are unchanged after fix `165929e…`; later commits concern specifications/evidence.
- Only this report and `receipt-r7.json` are written in the checkout. Owned `/private/tmp/c2-framing-r7-ug6l7145` scratch is deleted at completion. No real evaluation-data read, operational configuration/credentials read, box/network/SSH/gh, escalation, tracked edit, commit or push. Report structure/hash, receipt binding, source hashes, scratch deletion, HEAD and clean tracked status are verified before the completion marker; the report is not edited after its final hash.
