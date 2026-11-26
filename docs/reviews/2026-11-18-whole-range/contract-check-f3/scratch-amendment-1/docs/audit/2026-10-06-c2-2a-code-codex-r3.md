## Verdict

**BLOCK.**

Reviewed-commit: dfbf286f77abbd19af21bfa7bd7817c4d83473e2

The two original round-2 counterexamples are repaired. However, their PA repair introduces an uncontained helper invocation: an injected exception there makes an available pipeline forecast disappear, or replaces calibrated probabilities with raw probabilities. The new failure flag also admits complete-looking PA provenance if its update is lost after an append lands. These findings concern the round-3 change only.

The permitted suite passed: **232 passed, 1 skipped in 226.30 seconds**, including all 60 golden scenarios. The touched-phase cost evidence passes the unchanged limits under the authorized method. Neither result closes the additional boundaries reproduced below.

HEAD matches the requested detached commit; the tracked tree was clean before and after review. This is the third and final round authorized by the manager's delegated ruling at `57a6036`, not by a new decision from Eric. **2a returns to Eric. No fourth round is authorized.** This report admits no W1.5 certification, whole-range deployment review, deployment or D7 action.

## Findings

### R3-1 — Blocking: the new PA failure-marker invocation can change the forecast

**Locations:** `src/bts/model/predict.py:255,268,272–277`; callers at `:1054` and `src/bts/orchestrator.py:283`.

Both PA reader paths call `_mark_incomplete(ok)` outside a containment guard when collection reports failure. Its internal `try` protects the slot assignment, but cannot protect invocation/setup before its body. This repeats the boundary problem the revision correctly fixes for `_take_pipeline_provenance`: protecting a helper's body does not contain a failed invocation.

I independently applied the same invocation-failure envelope used by the new provenance-take test to this newly introduced helper. First, one PA record is appended through the real `collect` implementation using an append-then-raise proxy, so `collect` reports failure. Its error append is lost. A trace hook then raises `MemoryError` on the actual `_mark_incomplete` code object's `call` event, before the helper's body guard. Only the first pipeline PA collection, or only calibration's PA collection, is affected. I tested both held-buffer and original-path fallback reads.

The comparison used actual baseline functions extracted from local `git show f882411`, with the same synthetic 40-pick fixture and lower feature/training/prediction stand-ins on both sides. Parsing, calibration fitting/application and the candidate's provenance handling ran through their real functions.

| injected helper invocation | f882411 probabilities | candidate probabilities | cache result |
|---|---|---|---|
| pipeline PA, buffered | `[0.65, 0.65, 0.65]` | no forecast (`None`) | baseline writes 52 bytes; candidate writes no cache |
| pipeline PA, fallback | `[0.65, 0.65, 0.65]` | no forecast (`None`) | baseline writes 52 bytes; candidate writes no cache |
| calibration PA, buffered | `[0.65, 0.65, 0.65]` | raw `[0.86, 0.81, 0.74]` | identical 52 bytes |
| calibration PA, fallback | `[0.65, 0.65, 0.65]` | raw `[0.86, 0.81, 0.74]` | identical 52 bytes |

The trace fires at the candidate helper's entry (`:272`); it never fires at the baseline, which has no such helper. All retained caches have SHA-256 `b089ecc6c5459dc5888bba9aea3ba268c7dec04fcfa5ddf9d2f645264c037ec0`. The exception reaches the original prediction or calibration handler. Those handlers then treat an ancillary failure as a computation failure. The calibration case persists `status=failed` despite an available baseline calibrator; the buffered case even retains the affected hashed PA record.

The essential injection is:

```python
code = P._mark_incomplete.__code__
fired = []
def trace(frame, event, arg):
    if frame.f_code is code and event == "call":
        fired.append(frame.f_lineno)
        raise MemoryError("injected PA flag helper invocation")
    return trace
try:
    sys.settrace(trace)
    out = O.predict_local(DATE, data_dir=data, models_dir=models, picks_dir=picks)
finally:
    sys.settrace(None)
```

Four separate scratch pytest assertions also fail when the helper is replaced by a function raising `MemoryError`, matching the author's whole-helper invocation test technique. The actual-code trace comparisons confirm the missing boundary without replacing the helper.

This is measured behavior under deterministic synthetic injection, **not measured allocator exhaustion, a production incident, or a measured selection/delivery change**. The signed design §1/§3.0 expressly includes ancillary `MemoryError`, requires the available forecast to survive, and forbids substituting raw probabilities because calibration provenance failed. The review therefore cannot accept the broader containment claim. Preallocating the slot removes allocation from an ordinary successful slot assignment; it does not protect the new function call.

### R3-2 — Permanent PA invalidation still depends on the flag update succeeding

**Locations:** `src/bts/model/predict.py:275–277,280–285`; `tests/c2_2a/test_r2_counterexamples.py::test_the_count_still_withholds_when_the_flag_cannot_be_cleared`.

The helper silently catches a failed flag update, leaving `ok[0]` true. If the failed collection already landed its record, the count remains correct as well. With error recording lost, `inputs_complete` then admits that affected part. There is no independent retained indication of the reported collection failure.

I repeated the landed-then-raised collection at only the first pipeline PA append and, separately, at only calibration's PA append. A trace exception at the actual `ok[0] = False` statement (`:275`) is contained by the helper, but prevents the update. Real `save_slate` persists:

```text
pipeline case: serving.inputs = two non-null hashed PA records
calibration case: calibration.pa_input = non-null hashed PA record
both: serving.errors = []; calibration.errors = []
both: calibration.status = applied; probabilities = [0.65, 0.65, 0.65]
```

A separate scratch test that suppresses the marker, lands all three PA appends and loses error recording also fails its persisted withholding assertion. The forecast and 52-byte cache survive these checks.

This is a **bounded multiple-fault completeness counterexample**, not a claim that ordinary assignment into an exact preallocated builtin list allocates or spontaneously fails. The author explicitly introduced a lost-flag-write test and P23 to cover this envelope. That test only uses an append that never lands, leaving the count short. It proves the count protects that particular shape; it does not prove permanent invalidation when an append lands. The landed records in my probe have correct hashes: I did not manufacture an omitted PA file, partial hash or wrong bytes. The failure is re-admitting an affected provenance part after collection explicitly reported failure, contrary to §3.0 and the round-2 requirement.

### Disposition of the three round-2 required changes

| required change | result at the reviewed commit |
|---|---|
| 1. Contain provenance allocation/invocation; remove metadata before pandas calibration, row extraction and selection | **Met for the identified repair.** The allocation is inside the helper's guard; invocation is guarded; a null/failed take clears attrs before calibration. My allocation trace plus pandas provenance-copy fault now returns calibrated `[0.65, 0.65, 0.65]`, with model/inputs null and calibration actually applied. A failed whole-helper invocation likewise survives calibration and real slate persistence. |
| 2. Honor PA collection failure independently of length/errors, including persisted calibration PA | **Original counterexample fixed; full requirement remains incomplete.** All three landed appends with lost error recording now persist both PA parts as null while retaining the calibrated forecast and `n_fit=40`. The flag handles that case, but its new invocation/update boundaries have R3-1/R3-2. |
| 3. Add gate/ledger evidence and rerun affected behavior/cost | **Requested evidence supplied and verified within its coverage.** Three new scenarios, 15 additional mutants, ledger/resume and touched-phase measurements are present. The green evidence does not cover the new invocation counterexample or landed-plus-lost-marker combination. |

For metadata isolation, I inspected the full path: pipeline attrs are removed before calibration; the final witness is attached afterward; unchanged `save_slate` takes it before column/row extraction; unchanged `run_and_pick` drops any remaining witness before selection. My copy-fault check covers both calibration and actual row persistence, and the permitted golden gate exercises the real selection path. No separate new selection defect was found. These results do not establish immunity to every possible attrs-removal failure.

### New gate and ledger: valid retained results, narrower than the claim

All 60 candidate golden comparisons pass, including `fault_provenance_allocation`, `fault_provenance_take` and `fault_pa_append_landed`. I independently checked every golden file against its manifest hash and verified that the 57 prior file hashes are unchanged from `e22a905`. This verifies retained bytes, not an independent rerun of baseline generation.

I checked all **91 unique mutant entries**, their single matching replacement anchors, and the final round-3 output plus resume. The result is **90 attributable RED and C13 equivalent**. Each counted RED has named `FAILED` test evidence. The first P23 survival is disclosed and superseded. O18/P18's initially deselected unit tests are addressed by the exact-node-id resume, which records both unit and gate failures. I found no falsely classified RED in that retained evidence and did not apply tracked-file mutants.

There is nevertheless a coverage gap: the gate's landed-append scenario assumes the marker works; the lost-marker test assumes the append does not land. R3-1 is also absent from the new fault scenarios. A kill of P23 proves sensitivity to removing the count check in its test, not completeness when both admission signals remain apparently successful. The unchanged broader limitations from round 2 remain: C9's retained mutation changes `p`, not independently `y`; C13b tests threshold corruption rather than every map-policy field; filtered stderr is not complete logger/native-output evidence; fixed-clock cutoff checks do not prove unchanged real delivery times.

### Cost: the touched-phase evidence is a valid bounded pass

I read the actual local Git versions of `C2-2a-cost-method` and `C2-2a-review-r3` at `57a6036`. The ruling declares whole-run memory uninformative in both directions, keeps the 250 MB limit and prior time results, and requires same-code controls before counting phases. Round 3 expressly requests the touched-phase rerun.

I independently recomputed `phases_r3_summary.json` from all **60 raw rows** with the unchanged declared driver restricted to `load`, `tail_off`, `tail_on`. The summary matches exactly. All phase/side/repeat identities are unique and complete: five candidate/baseline pairs and five baseline/baseline controls per phase. RSS units are macOS bytes. Production and benchmark source at measurement commit `5523443` are byte-identical to the reviewed commit; the phase benchmark and shared driver are unchanged since round 2.

| phase | maximum / median paired increment, decimal MB | maximum absolute control increment, MB | result |
|---|---:|---:|---|
| load | 20.496 / 19.628 | 3.178 | PASS |
| tail_off | 0.590 / 0.410 | 0.983 | PASS |
| tail_on | 5.964 / 1.409 | 4.309 | PASS |

Every control is below the 125 MB aim, and every counted maximum is below 250 MB. Restricting the rerun is consistent with the change: load reaches PA collection; both tails reach provenance removal; tail_on also reaches calibration PA collection. Cache stops at pipeline entry, save is unchanged `save_blend`, and slate is unchanged `save_slate`; those phases do not reach the changed code. Their earlier passing measurements are carried forward, not freshly measured here.

The instrument is fresh-process **lifetime high-water RSS at phase end**, including allocations before the measured section. It is not an incremental allocator census or a simultaneous whole-cascade peak. The six-parquet, 351-row synthetic world and representative padded cache remain the declared footprint; training leaves are stand-ins except for the separate real cold-training test. Box sizes, baseline-worktree identity during acquisition and the author's no-concurrent-tests statement remain supplied evidence, not independently observed box or historical host measurements.

The author did not rerun whole-run timing in round 3. The earlier passing time results stand under the ruling; the narrow fixes add constant-size bookkeeping, which is reasoning, not fresh whole-run timing evidence at this commit. The phase-time medians (baseline → candidate: load 1.557 → 1.619 s, tail_off 0.000175 → 0.005236 s, tail_on 0.440 → 0.453 s) are diagnostic only. Cost does not cause this BLOCK and does not prove unchanged real cutoff behavior.

### The five author decisions

| decision | effect of this revision |
|---|---|
| 1. Move provenance and drop it before selection | The identified provenance-removal gap is fixed; the original lifetime choice remains supported. R3-1 is a new PA failure-marker boundary, not evidence against removal before pandas/selection. |
| 2. Genuine buffered-read OSError skips without retry | Unchanged and still accepted within the earlier checked read/decoder distinction. |
| 3. Witness v2, v1-reader skip and park-drag allowlist | Unchanged producer boundary. The skipped reader test does not certify the actual step-3 reader; that compatibility/admission gate remains separate. |
| 4. Experiment parquet gains provenance attrs | Unchanged and still bounded outside the pick path. No experiment artifact compatibility/value measurement was performed. |
| 5. Enabled empty predictions may have null status | Unchanged, bounded to the nonpersisted empty-frame case. Empty slates are refused; incomplete/null provenance requires reader refusal, not admission as complete. |

## Required changes

This is BLOCK, not SIGN WITH EDITS. A documentation diff cannot establish the missing behavior. The following are technical conditions for an acceptable implementation, **not authorization to repair, perform a fourth review, commit or ship**:

1. Contain invocation/setup of the new PA failure-marker operation on both reader paths. A failed marker must not prevent the one original parse, lose the pipeline forecast or skip an available calibration. Preserve genuine parser-failure semantics and avoid introducing a retry of computation.
2. Keep a reported PA collection failure permanently disqualifying even when its record lands, error recording is lost and the marker update fails. Withhold the affected part or witness when truthful incompleteness cannot be retained. Add persisted checks for pipeline and calibration, without claiming ordinary builtin slot assignment inherently allocates.
3. Cover those actual boundaries with forecast/cache comparisons and witness admission assertions; update the gate/ledger and affected cost evidence for any implementation Eric elects to pursue. Eric must decide the disposition now: the delegated ruling authorizes no fourth round. Even a future accepted implementation would still require the separate frozen-tooling W1.5 recertifications with runner and reviewer acceptance, whole-range deploy review and Eric's D7.

## What was run

- Read the entire round-3 prompt before other work. Read the round-2 report, the narrow source/test/evidence diff, relevant signed contract, new gate scenarios/tests, ledger, cost code/raw rows/summaries and local Git ruling rows. Confirmed the exact detached HEAD and clean tracked tree. The change has 17 files, with production edits confined to `src/bts/model/predict.py` and `src/bts/orchestrator.py`.
- Ran the permitted suite with temporary directories set before harness entry:

  ```text
  UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York OMP_NUM_THREADS=1 TMPDIR=/private/tmp/c2-2a-r3-review MPLCONFIGDIR=/private/tmp/c2-2a-r3-review/mpl uv run python -B -m pytest tests/c2_2a tests/test_slate.py tests/test_local_tier.py tests/model/test_calibrate.py tests/scripts/c1_r4a/test_provenance_stored.py -p no:cacheprovider --basetemp=/private/tmp/c2-2a-r3-review/pytest
  ```

  **232 passed, 1 skipped in 226.30 seconds, exit 0.** The skip is the disclosed old-reader/new-producer compatibility test. The harness may internally copy the two policy files expressly permitted by the prompt; no separate root `data/` read was performed.
- Ran eight independently written scratch pytest checks: **3 passed, 5 failed in 2.66 seconds, exit 1**. The passing checks independently rerun both round-2 counterexamples and a failed provenance-take invocation with a pandas copy fault through slate persistence. Four failures concern the new PA helper invocation (pipeline/calibration × buffered/fallback); one concerns landed collection plus a suppressed marker and lost errors. These are assertion failures outside the passing author suite.
- Ran **12 actual-source comparisons**: four helper-invocation shapes on baseline/candidate, and two slot-update shapes on baseline/candidate. Checked the fired trace events, returned probabilities, persisted candidate slates and retained cache bytes/digests. Heavy computation leaves use the shared synthetic fixture; no live provider, real outcome corpus or experiment run was used.
- Recomputed the three cost summaries, checked all 60 raw identities/units, all 91 mutant anchors and final/resumed attributable failures, and every retained golden hash. Did not run the tracked-file-mutating runner, regenerate baseline goldens, acquire fresh cost measurements or rerun the broader fast suite. The supplied fast-suite record at `5523443` reports **4057 passed, 7 skipped, 68 deselected, 22 xfailed in 385.73 seconds**, exit 0.

All reviewer runtime/harness work used owned `/private/tmp/c2-2a-r3-review` routing, including pytest basetemp and synthetic park-drag paths; reviewer scratch was deleted at completion. No external memory/history lookup, shared context corpus read, credential/configuration read, tracked edit, escalation, network/box/SSH/gh operation, commit or push occurred. No round-3 scratch-location exception occurred.
