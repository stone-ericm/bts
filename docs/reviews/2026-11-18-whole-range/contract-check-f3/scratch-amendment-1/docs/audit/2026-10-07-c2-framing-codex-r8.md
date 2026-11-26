## Verdict
**BLOCK.**
Reviewed-commit: dcdbb71aa8d9cc3945bfacab0e6a99abd163042f

The original r7 aborted and non-aborted self-cleanup controls now refuse. The permitted suite passes **162/162**. An independent driver run of the actual runner/classifier tests produces complete clean evidence with zero late plugins. Default-scoring synthetic three-seed replay also validates in a fresh interpreter.

The new outermost check nevertheless certifies an aborted session as RED. A wrapper registered through pluggy's public base-class API can unregister **itself before yielding**. It remains active around the sentinel in the current dispatch, but the sentinel checks the subsequently changed live registry. All three late-registration layers then miss it. This is a defect in the new observation producer, using the same registration API explicitly exercised by the new tests.

This is Eric's authorized round 8 under the unchanged full rules. The item stops; a ninth round needs a new ruling. This verdict admits no seed, run or operational action.

**Scope:** The complete r8 prompt was read first. HEAD matched the requested detached commit, with a clean tracked tree before and after review. Only the r7-to-r8 change is assessed for new defects. No outside context corpus or memory registry, real evaluation data, operational configuration/credentials, box or network was consulted. Installed dependency source was inspected to explain the measured hook dispatch. Independent artifacts and subprocess scratch were routed under `/private/tmp/c2-framing-r8-wgxso1ef`, deleted at completion. No accidental outside-root collection touch was observed. No escalation or tracked edit occurred.

## Findings

### R8-1 — The live hook registry does not prove the ordering of the active dispatch

**New code:** `docs/audit/2026-10-06-c2-framing-evidence/mutant_runner.py:59–62,66,77,175–176`. **Affected claims:** the sentinel was outermost each time it ran, and a completed sentinel finish establishes a completed session finish when the late-registration fields are clear. **New tests/mutants:** `test_an_outer_wrapper_without_a_registration_event_is_still_seen`, `test_a_hidden_call_wrapper_is_seen`; R27–R30.

The new producer calls `caller.get_hookimpls()` from inside the sentinel, then checks that the live list's last implementation belongs to the sentinel. Installed pluggy 1.6.0 dispatches a **copy** of its implementation list. An outer wrapper already entered from that copy can unregister its own plugin before yielding. The current generator still wraps the sentinel, while the live list the sentinel inspects no longer contains it.

**Measured through the unchanged checkout runner `main`, unchanged driver and actual pytest subprocesses:** An owned target contained `value = 1`; the sole mutant changed it to `value = 2`. Two intended test bodies genuinely asserted `val() == 1` and failed. The owned conftest used this lifecycle:

```python
import importlib.util
from pathlib import Path
import pluggy
import pytest

def val():
    sp = importlib.util.spec_from_file_location(
        "owned_target", Path(__file__).with_name("target.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m.value

def mark(name):
    Path(__file__).with_name(name).write_text("entered")

class OwnedBaseLate:
    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus):
        mark("outer_entered")
        pluggy.PluginManager.unregister(session.config.pluginmanager, self)
        mark("unregistered")
        result = yield
        mark("hook_entered")
        if val() != 1:
            pytest.exit("owned active wrapper cleanup stop", returncode=1)
        mark("hook_completed")
        return result

def pytest_runtest_setup(item):
    if item.name == "test_first" and val() != 1:
        pluggy.PluginManager.register(
            item.config.pluginmanager, OwnedBaseLate(), "owned-base-late")
        mark("registered")
```

Each owned test calls `mark` for its body, then asserts `val() == 1`. The runner first collects those two nodes on unmutated source. Collection-pass canaries were cleared before the mutated subprocess. Recording wrappers retained actual subprocess results and the actual returned bundle; they did not alter the driver, evidence or outcomes.

The execution order explains all three missed layers:

1. Setup registers the late plugin through `pluggy.PluginManager.register`, exactly the base-class API in the author's new session/call tests. It bypasses pytest's registration-event hook.
2. Both intended bodies fail and complete their setup/call/teardown reports.
3. The session-finish dispatch starts the late tryfirst wrapper outermost. It unregisters only itself, then yields.
4. The sentinel checks the live registry, now sees itself last, and records no ordering violation. Its inner session finish completes with matching counters.
5. The active outer wrapper resumes and aborts before its completion canary. Pytest handles this session-finish exit and returns 1; no interrupt hook is recorded on this path.
6. The final registry census contains no late plugin. The driver writes authentic evidence and the classifier accepts RED.

The retained bundle is:

```text
rc: 1
first:  setup passed, call failed, teardown passed; wasxfail false
second: setup passed, call failed, teardown passed; wasxfail false
calls: both AssertionError
finish: {exitstatus: 1, testsfailed: 2, testscollected: 2}
late_plugins: 0
not_outermost: []
marked: []
NO interrupted, internalerror or collecterror record
```

Both body canaries, `registered`, `outer_entered`, `unregistered` and `hook_entered` exist; `hook_completed` does not. The actual mutated subprocess returns **1**, stderr is **`Exit: owned active wrapper cleanup stop`**, and the unchanged runner prints:

```text
X1 RED 2 failed in 0.01s [2 intended]
    FAILED test_owned.py::test_first
    FAILED test_owned.py::test_second
NOT RED: none
runner return: 0
target restored: true
```

The failures are real. The false certification is that the sentinel was outermost and session finish completed normally. Evidence written after `pytest.main` returns cannot establish that independently.

Three controls distinguish observation timing from a fabricated outcome:

| Owned wrapper lifecycle | Actual result on r8 |
| --- | --- |
| Same base-class session wrapper; self-unregister **after** yielding, then abort | `INCONCLUSIVE(exit 1, sentinel not outermost)`; `late_plugins=0`, `not_outermost=["sessionfinish"]`; runner return 1. |
| Self-unregister **before** yielding; no abort | RED; `late_plugins=0`, `not_outermost=[]`; wrapper-completion canary exists. This also falsely satisfies the explicitly stated ordering/late-registration rules. |
| Base-class tryfirst **test-call** wrapper; self-unregister before yielding | RED; `late_plugins=0`, `not_outermost=[]`; its outer-entry and self-unregistration canaries exist. This demonstrates the same per-call producer gap, without claiming an interrupted session in this control. |

The witness accesses no sentinel, recorder, evidence file, report attributes or private pytest field. It unregisters only its own plugin through a public pluggy API. The author's new tests explicitly use that same API to bypass the event layer, then self-unregister after yielding. Their tests therefore establish this API/lifecycle as part of the new defense's domain. Moving self-cleanup before the yield does not become runner-object or pytest-internal tampering.

This does not show that any of the author's 119 recorded attributable kills used this lifecycle. It demonstrates that the new fallback observation does not establish its claimed invariant. The registration event layer correctly covers the original r7 API path; the new base-class fallback remains unsound.

### R7-1: disposition of the three required changes

| Required change | Measured disposition |
| --- | --- |
| Retain late registrations as events, even after self-cleanup | **Original r7 witness corrected.** Both aborted and non-aborted versions now return runner 1 / `INCONCLUSIVE(exit 1, plugins registered late)`, with `late_plugins=1` and `not_outermost=["sessionfinish"]`. The plugin disappears before final census. The aborted control lacks the completion canary; the non-aborted control has it. A briefly registered/unregistered public-API object independently produces `late_plugins=1`, `not_outermost=[]`, and refuses. The retained event works for pytest's registration API. It does not cover base-class registration; R8-1 defeats the new fallback. |
| Exercise the evidence producer, including disappearance before final census | **Implemented for the specified original cases.** The brief-plugin test asserts the produced bundle itself; R25/R26 isolate event capture and its inclusion. The two base-class tests separately exercise per-call and session ordering. Their self-unregistration occurs after yielding, so they do not vary the component R8-1 exposes. |
| State the driver guarantee precisely | **Correct new paragraph, incomplete wording cleanup.** The new paragraph correctly separates an abort escaping `pytest.main` from an abort it catches. However, opening lines 4–6 still say an abort anywhere leaves no evidence, contradicting that paragraph and the measured caught-abort cases. Remove that older sentence. This wording issue is not a separate blocking code defect. The further assertion that the three layers establish ordering is disproved by R8-1. |

The final census remains useful for plugins still present, but cannot recover registrations already removed. `late_plugins` now adds events and census membership; it is a nonzero refusal signal, not a count of distinct plugins (one persistent public-API plugin yielded 2). The new ordering check covers an outer wrapper still registered when the sentinel starts. It misses an outer wrapper active in the dispatch copy but absent from the live registry.

### Earlier closed behavior and legitimate stage-one replay

Independent actual-runner probes repeated all earlier runner counterexamples:

- Captured-stdout spoof plus a second before-body abort: `INCONCLUSIVE(exit 1, interrupted)`; the second body does not run.
- Final teardown aborts with both `2 failed cleanup checks` and `owned teardown stop`: interrupted refusal; both bodies ran, cleanup did not complete and the final teardown report is absent.
- Ordinary session-finish abort and both pre-existing post-yield wrapper aborts (ordinary and tryfirst): session-finish-incomplete refusal. Wrapper evidence records `finish={raised: Exit}`.
- Public exit after the final teardown report: interrupted refusal despite all phase reports existing.
- Genuine failure plus non-strict XPASS: marked-xfail refusal. Runtime xfail added during the body: report-status refusal even with an empty collection-time marked list.
- Unconfiguration abort escaping `pytest.main`: no bundle, no-normal-end refusal.
- Ordinary two-failure mutant with inherited `-x`: both bodies and all phases execute, RED. Complete clean pass: SURVIVED. Every owned mutated target is restored.

All sixteen prescribed runner subprocess tests pass rooted in their own temporary directories. Independently, the unchanged real driver ran all **18 actual runner/classifier test functions / 41 nodes**: pytest exit 0, SURVIVED, complete setup/call/teardown reports and observations, matching finish counters, `late_plugins=0`, `not_outermost=[]`. No ordinary-run false refusal was found in these probes.

The artifact counterexamples from r3–r5 were independently repeated:

- **Nullable season membership:** In the first two seeds' B unit files, 22 rank-1 misses in 2024 and 24 in 2025 per seed receive missing nullable seasons. Coherent damaged cards still report P@1=1.0 in both seasons. Direct validation and aggregation refuse missing season; release returns false; launch/child refuse. A fresh namespace creates no seed-2 root.
- **Extra identity key:** Seed 3's extra key refuses direct validation and aggregation.
- **Edited secondary card with unchanged profiles:** First-two B cards change P(57) to baseline +0.001 with coherent diffs/results; all 18 profile hashes remain unchanged. Validation, release, launch, child and aggregation refuse recomputation mismatch.
- **2024 replacing 2025 / foreign HEAD:** Wrong-season profiles and a coherent seed-1 rebind to `02516cfedc812f047295b6f2ab212fd72a944507` each refuse at all five consumers. All-three bogus identities refuse trusted identity comparison.
- **Probability/scorer refusal:** Nullable strings containing `"bad"` refuse as nonnumeric through the predicate and validator; an injected scorer TypeError becomes RunInvalid. Infinity, missing values, booleans and strings refuse. Float, nullable float, integer, 0 and 1 probabilities pass the predicate.

Three synthetic `run` outputs at the registered **`mc_trials=10000, season_length=180`** pass same-process and fresh-interpreter aggregation, **A positive / B negative**. Their actual existing HEADs are `[a6f400dd67b0a946ff0e3ad038a34bc6b3dd2563, dcdbb71aa8d9cc3945bfacab0e6a99abd163042f, dcdbb71aa8d9cc3945bfacab0e6a99abd163042f]`. Real git admission checks return no problems for all three, including equal reviewed/run HEAD and metadata-only descendants. Missing HEAD refuses. Current HEAD under the old r7 review refuses the actual registration-file change. The equal-HEAD refusal has not returned.

These probes stub admission, input/feature loading and walk-forward using synthetic trusted identities/pins and a mocked owner-release row. Claim writing, retained parquet, label handling, scoring, validation and git checks are real. Damage/equivalent probes use 200 trials to reduce synthetic replay cost; the legitimate replay above uses the full registered defaults. No real SIGN/exposure/admission, release, live launcher, walk-forward or model fit was created. Shared-admission tests separately pass their disposable-git checks.

`screen.py` is byte-identical to r7, including all **35** top-level functions. Independent regression probes also repeated:

- **Closed inputs:** Real feature computation on 48 synthetic PA rows / 24 games, with owned live cache/raw canaries, then frozen-input installation after changing those canaries. All 16 registered features match exactly; zero trapped text/table reads in closed computation; 28 non-null bullpen values in each; park drag all NaN; pitcher self-check identical on 48 rows.
- **Labels/settings:** Original miss plus resumed hit remains a miss; resumed-only batter drops. Counts `changed=1, void_dropped=1`, ranks `[1,2]`, hits `[0,1]`. Settings `(20,7)` pass; changing either to 0 or 10 respectively refuses. Real training-helper calls into a recording classifier receive all three registered seeds with deterministic and row-wise flags true; recording `fit` performs no training. Official-date availability and retrospective void drop/re-ranking limits remain unchanged.
- **Namespace/release/budgets:** Permitted repeat-claim, foreign-seed, cardinality and canonical-namespace cases pass. Exact Eric/30 release parses; Ericsson, zero and decimal overflow refuse. Budget decisions remain over_cap, ok, checkpoint and ok for `(60.48,45,True)`, `(30.48,30,False)`, `(60.48,30,False)` and `(60.48,30,True)` respectively.

No new false refusal of legitimate complete three-seed evidence was found. The author's box inventory/cache/environment facts remain stated and unverified; synthetic checks do not establish real-input coverage or box readiness.

### Test and mutant strength; equivalents

The permitted suite is **137 framing + 25 admission = 162 passed**, Python 3.12.13 / pytest 9.0.2 / pluggy 1.6.0. R25/R26 directly exercise retained events; R27/R28 separately exercise the two ordering producers; R29 covers refusal consumption; R30 checks the list end used for ordering. These are useful component tests. Both base-class wrapper producers, however, leave the outer plugin registered until after the sentinel's check. Before-yield self-unregistration is absent from their coverage. A hand-supplied `not_outermost` value cannot establish the producer's truth.

All **122** current anchors occur exactly once; all named selections appear in the passing suite; no selection list contains options. Latest current-spec statuses are **119 RED, G8/H12/N10 SURVIVED, none missing** (retired F24 excluded). The recorded revision-8 full run exits **1**, `NOT RED: G8,H12,N10`. “119 of 119 attributable RED” describes the non-equivalent selection, not a full 122-mutant exit 0. No tracked-source sweep was run in this review.

The three recorded equivalents remain supported over validator-accepted evidence by source reasoning and independent owned-copy probes:

- **G8:** Every accepted run's complete pins equal the same trusted dictionary, with its digest bound. A coherent wrong pin/digest refuses admitted pins in original and G8 copy; the later cross-run comparison adds no accepted disagreement.
- **H12:** Every accepted identity has exactly five fields and equals the trusted projection. Extra-key identity refuses in original and H12 copy; accepted identities cannot disagree.
- **N10:** Every nonempty unit wholly belongs to its integer season, with non-missing scored values and daily ranks 1..n. Successful scoring therefore supplies both seasons' P@1 keys. Removing all 2025 rank-1 rows refuses ranks in original and N10 copy. The later coverage check is redundant in that accepted domain.

Original and all three copies accept the ordinary synthetic stage-one evidence with A positive / B negative. These equivalent dispositions do not repair R8-1 or establish general runner completeness.

## Required changes

1. Bind ordering evidence to the **implementation list of the active dispatch before wrappers execute**, or retain complete registration history covering the base-class path already claimed. A live registry queried inside the sentinel is insufficient. Pluggy's public `add_hookcall_monitoring` exposes the actual hook implementation list before dispatch; that is a possible approach, not a verified repair here. Preserve independent call/report agreement, complete-session checks, event retention and ordinary-run acceptance.
2. Add actual-driver producer tests for base-class session and call wrappers that self-unregister **before yielding**, plus the aborted/non-aborted session controls and the otherwise identical after-yield control. Assert the authentic bundle/refusal and canaries. Mutate each producer separately from the classifier branch; the before-yield abort must never certify RED.
3. Remove the stale opening “abort anywhere” claim and update the ordering guarantee to match the implemented observation. Retain the correct distinction between escaping and caught aborts.

No repair or further review is authorized by this BLOCK. Round 8 is the last round under the current ruling; the item stops. A ninth round requires Eric's new ruling.

## What was run

- Exact permitted suite, with scratch routing through `TMPDIR` and `MPLCONFIGDIR`: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run python -B -m pytest tests/scripts/c2_framing tests/scripts/test_c1_admission.py -p no:cacheprovider`. **162 passed in 202.89 seconds**, exit 0.
- **20 owned actual-runner cases**, using unchanged collection, driver and classification, authentic returned bundles/subprocess output, and canaries cleared after baseline collection. All targets restored. The before-yield session abort falsely returns runner 0/RED; its after-yield control refuses.
- Independent real-driver run of **18 runner/classifier functions / 41 nodes**, complete evidence, pytest 0/SURVIVED, no late plugins or ordering violations.
- Full-default synthetic three-seed `run`/same-process/fresh-interpreter replay; real metadata-HEAD checks; independent r3–r5 damage probes through all relevant consumers; owned G8/H12/N10 source copies. All expected acceptance/refusal outcomes described above.
- Real synthetic feature/label computation, settings refusals, recording classifier constructors, release grammar, budget gate, current anchor/status/node checks and source hashes. Reviewed runner, screen and registration bytes remain unchanged; later commits after `a6f400d` affect only ledger/spec/README evidence.
- Final report structure, HEAD, clean tracked tree, report/receipt binding and deletion of the owned scratch prefix verified. Detailed measured results, reproducer sources and source hashes are retained in `receipt-r8.json`; no real data or operational run occurred.
