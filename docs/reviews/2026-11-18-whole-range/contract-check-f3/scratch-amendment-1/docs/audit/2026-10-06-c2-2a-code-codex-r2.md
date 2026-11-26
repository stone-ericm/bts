## Verdict

**BLOCK.**

Process disclosure first: my first reviewer timestamp-probe invocation let `golden.scenarios._template` create its synthetic template in the default macOS temp directory, outside the required `/private/tmp/c2-2a-r2-*` prefix. I then read directory creation metadata to identify that invocation's exact template, verified its identity against the copied probe directory, and removed it. Subsequent harness execution set `TMPDIR` inside the permitted scratch directory. This was a scratch-location violation. No external project corpus, memory, configuration, credentials, or real outcome data were consulted for this round.

Reviewed-commit: e22a90542516324687ba762e259fc50b8b9634fa

The original counterexamples are repaired, but the broader containment and permanent-invalidation requirements are not fully met. A fault at the new provenance-removal helper's unguarded dictionary allocation escapes `predict_local` and loses an available forecast. PA collectors also discard the new failure result and can publish apparently clean provenance after a reported append failure. These are in the revised code, not unrelated repository defects.

The permitted suite passed: **221 passed, 1 skipped in 214.38 seconds**. Two additional reviewer-authored contract checks failed. The cost re-runs meet the unchanged time limits and the authorized replacement memory method; cost is not the reason for this BLOCK.

HEAD matches the requested detached commit. The tracked tree was clean before and after review. This is the last authorized code-review round: 2a returns to the owner. It is not admitted to certification or deployment gates by this report.

## Findings

### R2-1 — Blocking: the new provenance-removal helper can still replace the forecast

**Locations:** `src/bts/orchestrator.py:87` and `:244–247`, introduced by `066893c`.

`_take_pipeline_provenance` creates `parts = {}` before its first `try`. `predict_local` calls it after leaving the prediction handler, without a surrounding containment boundary. A `MemoryError` allocating that new metadata dictionary therefore escapes both functions before calibration. The comment claiming preparation is inside the guards does not cover this statement.

This is a residual F4 defect in the newly introduced F1 recovery helper. The signed design's §3.0 explicitly includes `MemoryError`, forbids new witness exceptions escaping `predict_local`, and requires the available forecast to return. The round-1 required change covered record construction itself, not just operations performed after a record exists.

**Independent reproduction:** I ran the real local prediction/calibration functions with the same synthetic 40-pick fixture used in round 1. Baseline functions were extracted from actual `git show f882411` source; both sides shared the fixture's lower feature/training/prediction stand-ins. A trace hook injected one `MemoryError` precisely at the new dictionary-construction statement:

```python
code = O._take_pipeline_provenance.__code__
def trace(frame, event, arg):
    if event == "line" and frame.f_code is code and frame.f_lineno == 87:
        raise MemoryError("injected provenance dictionary allocation")
    return trace
try:
    sys.settrace(trace)
    out = O.predict_local(DATE, data_dir=data, models_dir=models, picks_dir=picks)
finally:
    sys.settrace(None)
```

| side | result | cache |
|---|---|---|
| f882411 | calibrated probabilities `[0.65, 0.65, 0.65]` | 52 bytes |
| reviewed candidate | `MemoryError` escapes; no returned forecast | identical 52 bytes |

The hook never fires in the baseline because that provenance allocation does not exist there. The candidate traceback passes through `predict_local:247` to `_take_pipeline_provenance:87`. A separate scratch pytest check requiring the calibrated forecast **FAILS** at that boundary.

This measures behavior under a deterministic injected allocation fault. It is not a measurement of real allocator exhaustion or evidence of a box incident. It proves the missing exception boundary; it does not establish a measured selection or delivery change. `run_cascade` also calls `predict_local` without an exception handler, so its code does not restore the lost forecast.

A repair must protect this initialization and any call preparation while still removing pipeline provenance before pandas operations. Merely catching the helper at its caller and continuing with the original metadata-bearing frame can reintroduce F1's attrs-copy failure.

### R2-2 — Residual F3: PA collection does not honor the reported failure independently of list length

**Locations:** `src/bts/model/predict.py:246–259,281–285`; `src/bts/orchestrator.py:128–131`.

The revised `collect` returns a failure indication. The pick resolver correctly carries that indication into permanent completeness flags. `_read_pa_parquet` discards it on both buffered and fallback paths. Pipeline PA completeness is inferred solely from `len(inputs) == n_parquets`; calibration PA completeness is inferred solely from `len(pa_inputs) == 1`.

This has the same limitation the author identified when C19/C22 initially survived: an append can land and then report failure, leaving a complete-looking length. The revision adds a regression for that case on sample bindings, but does not extend the rule to PA inputs.

**Persisted reproduction:** route only PA-input appends through this proxy, using the real `collect` failure handling, and make its error append fail too:

```python
class LandThenRaise:
    def __init__(self, target):
        self.target = target
    def append(self, rec):
        self.target.append(rec)
        raise MemoryError("PA append landed and then raised")
```

Every affected `collect` returns false. Both pipeline PA appends and the calibration PA append fail this way. The actual forecast remains `[0.65, 0.65, 0.65]`, and the cache is byte-identical to the baseline. Real `save_slate` persists:

```text
serving.inputs = two non-null, hashed PA records
serving.errors = []
calibration.pa_input = non-null, hashed pa_2026 record
calibration.errors = []
calibration.status = applied; applied = true; n_fit = 40
```

The scratch check requiring permanent withholding of failed PA collection **FAILS**. The candidate's length-based admission ignores the failure signal.

This is a bounded synthetic collector failure, using the same append-then-raise envelope as the author's new binding test. I did not manufacture an omitted PA file or an incorrect hash: the landed records happened to be correct in this probe. The demonstrated defect is failure to invalidate an affected part after a reported ancillary failure, contrary to the explicit §3.0 contract. It is not evidence that ordinary builtin-list appends currently corrupt PA records. Successful list length is insufficient to implement the stronger signed rule.

Carry PA collection failure independently of the error list and retained length, including the calibration PA read; withhold the affected PA provenance when it fails. An unavailable truthful attachment must remain visibly incomplete/null.

### Disposition of F1–F6 and all eight requested counterexamples

All eight original failure shapes were rerun against this candidate. The suite additionally exercised the author's expanded reproductions, genuine-error checks and 57 golden scenarios.

| round-1 required change | disposition at this commit |
|---|---|
| F1: isolate pipeline provenance before calibration and slate row extraction | The original attrs-copy counterexample is fixed. Pipeline attrs leave the frame before calibration; `save_slate` takes `serving` before row extraction. The new helper's allocation boundary remains defective as R2-1 shows. |
| F2: separate genuine read OSError from decoder preparation | **Met.** The genuine read has its own boundary; decoder-preparation OSError takes the original-text fallback. Genuine read/decode/parse tests pass. |
| F3: permanent invalidation independent of error recording; truthful assignment state | **Partially met.** Omitted pick inputs and lost sample bindings are withheld; assignment truth comes from a local boolean and the record is built afterward. PA collection still has R2-2. |
| F4: contain construction/extraction/error formatting at the operation itself; report package failures | **Partially met.** The original undescribable-error case and package-query silence are fixed. New record construction at R2-1 is unguarded. |
| F5: invalidate digest after a short/unsupported successful write | **Met.** The forwarded write is unchanged, with no retry; the digest is withheld. Full writes, genuine partial-write errors and real tiny-LightGBM round-trip checks pass. |
| F6: envelope/inventory visibility, boundary checks, attributable ledger and repaired-candidate cost re-runs | The reproduced timestamp false green is fixed, as are the complete pick-inventory check and runner classification. The new fault scenarios and cost re-runs are present. Coverage remains incomplete for R2-1/R2-2, so the full required change is not met. |

| requested reproduction | current result |
|---|---|
| attrs-copy fault | Baseline and candidate both return `[0.65, 0.65, 0.65]`; candidate calibration says `applied=true`, `status=applied`, `n_fit=40`. No new provenance dict is copied at that calibration boundary. |
| decoder-preparation OSError | Both return the calibrated forecast. Candidate uses 40 samples and records fallback pick inputs with null hashes and preparation errors. The permitted gate checks exactly one additional original-text read per history file. |
| omitted input with lost error | Add consumed out-of-window `2026-04-02.json`, fail its input append and error recording. The persisted candidate now has `pick_inputs=null`, despite empty error lists; calibration remains applied with 40 samples. It no longer publishes the retained 40-file prefix as the complete 41-file inventory. |
| stale `applied` metadata | The old incremental `_cal_set` mechanism is gone. Fault the replacement calibration-record builder after actual assignment: the calibrated forecast still returns, and the persisted calibration record is null. The ordinary and post-assignment genuine-error tests retain `applied=true`. |
| undescribable preparation error | Fault PA `read_bytes` with an exception whose str/repr fail. Baseline and candidate both calibrate and write identical 52-byte caches. Candidate parses the original path, records an undescribable error and nulls the affected byte/hash fields. |
| successful short write | Write `abcdef` to a sink accepting only `ab`: returned count remains 2, persisted bytes remain `ab`, digest is now null, and an error is recorded. |
| `written_at` visibility mutant | Real synthetic day runs persist `2026-07-01T01:10:00+00:00` versus injected `2037-01-01T00:00:00+00:00`. Revised `_slate` observations differ, and an isolated `_compare` of those actual slate observations rejects on `written_at`. |
| package-query silence | Inject a pyarrow version-query exception: package value is null and `packages: pyarrow version unavailable` is recorded. |

For the stale-field reproduction, the replacement builder was faulted because the former field setter no longer exists. For the timestamp check, I converted the normal observed slate to an otherwise equal v2 reference solely to exercise the allowed-difference comparison; this is an observation-sensitivity test, not independently regenerated baseline evidence.

### Revised gate and mutant ledger

The current manifest/harness hashes, baseline commit, golden hashes and package-version equality pass. The revised observation includes `written_at`, envelope keys, the full routed API call log and selected caught-failure stderr lines. Complete calibration pick inputs must equal the resolver inventory. The new held-buffer-allocation and map-only-hash scenarios address distinct boundaries missed in round 1.

I checked **76 unique ledger entries** and every replacement anchor against the reviewed source: all are valid. The last full output section is the hardened-runner run at `57de238`. It contains **75 RED attributable mutants**, each followed by named `FAILED` test evidence, and C13 remains the recorded equivalent. The intermediate invalid O11 replacement is explicitly INCONCLUSIVE and superseded by the final run; it is not counted as a kill. The runner's nonzero final exit is explained by C13 and does not turn that equivalent into RED.

The classifier now requires pytest exit 1, failed-test evidence, and no reported errors, skips, interruptions or related unclean summary states. Its regression checks passed. I found no falsely classified RED in the retained final section. I did not run the tracked-file-mutating runner.

The 75 kills establish sensitivity to those mutants, not every ancillary boundary. R2-1's dictionary preparation and R2-2's ignored PA failure result remain outside the tested fault envelope. The original complete suite is green while reviewer checks for those contracts fail. C9 still mutates `p` under a combined `p and y` label; the exact-binding unit test checks both values, but there is no separate retained y-corruption kill. C13b exercises thresholds, rather than all map policy fields independently. Do not inflate those ledger labels into independent measurements of every subfield.

The stderr observation is filtered handled-failure text, not a receipt of every logger/native stderr byte. The fixed/advancing clocks continue to support the tested cutoff guard, not unchanged real delivery time. These limits are consistent with the design's bounded-evidence language.

### Cost: valid bounded pass under the ruling

I independently recomputed the revised summaries from **50 whole-run rows** and **120 phase rows**. Both summaries match exactly. Each retained pair member is unique and all five repeats are present; RSS is labeled macOS bytes. `src/bts` and the benchmark implementation at measurement commit `340dcef` are byte-identical to the corresponding files at the reviewed commit.

I also read the actual local Git version of exposure-register row `C2-2a-cost-method` at `6676c72`, rather than relying only on the README's attribution. It authorizes the replacement phase method, declares whole-run memory uninformative in both directions, preserves the time results and the 250 MB limit, and requires same-code controls comfortably below it. The README declaration predates the first phase run; its cache-stop explanation and the summarizer's 125 MB control-aim rule are now aligned.

| whole-run case | paired time median / maximum, seconds | time limit |
|---|---:|---|
| cold_small | 0.284 / 1.241 | met |
| warm_off | -0.327 / 0.541 | met |
| warm_on | -0.483 / -0.147 | met |
| warm_unavailable | -0.501 / 0.342 | met |
| identical-baseline control | 0.620 / 1.649 | diagnostic |

All four meet median ≤2 seconds and maximum ≤5 seconds. The re-run README reports that the author ran nothing else during this time series; that absence claim is supplied evidence, not independently observed by me. Whole-run RSS still exceeds 250 MB in warm cases and the identical-code control reaches **1296.056 MB**. Those columns and their original combined `accept=false` flags must not be described as a whole-run memory pass. Under the explicit ruling they carry no memory acceptance weight.

| replacement phase | maximum candidate-minus-baseline peak, decimal MB | control maximum absolute difference, MB |
|---|---:|---:|
| load | 98.599 | 1.130 |
| cache | 10.994 | 0.754 |
| save | 3.490 | 3.211 |
| tail_off | 1.114 | 2.769 |
| tail_on | 5.226 | 3.293 |
| slate | 1.098 | 1.491 |

Every phase meets ≤250 MB and every control meets ≤125 MB. The load outlier and concurrent review-test activity are disclosed; that pair is retained as measured. The method therefore counts all six phases and passes them under the ruling.

This is an arithmetic/method audit of supplied measurements, not a new timing or memory measurement. It accepts the ruled screen for six synthetic parquets, the 9.30 MB representative cache, 156 history picks and a 351-row slate. Process-lifetime `ru_maxrss` in separate phase processes does not directly measure accumulated allocator residency through a full cascade. The representative cache does not reproduce every native LightGBM allocation; real small training is a distinct control. Box footprint/headroom, ten-parquet scaling, real cutoff behavior and fault-path cost remain unmeasured here. These limitations were bounded in round 1 and still hold.

### The five author decisions

| decision | current disposition |
|---|---|
| 1. Move provenance and drop it before selection | The revised timing of removal fixes the original pandas-copy issue: pipeline provenance is taken before calibration, final witness before slate row extraction, with `run_and_pick`'s drop retained as a backstop. Acceptance still depends on repairing the new containment gap. |
| 2. Genuine buffered-read OSError skips without retry | Still accepted. Implementation now distinguishes it from decoder preparation, and tests both. |
| 3. Witness v2, v1-reader skip, park-drag allowlist | Still a reasonable producer boundary. The skipped v1-reader test does not certify v2-reader compatibility; step 3 must realign and validate the actual reader before study admission. |
| 4. Experiment parquet gains provenance attrs | Still disclosed outside the pick path. No experiment run or independent parquet compatibility/value measurement was performed here. The pick-path tests do not settle that artifact compatibility question. |
| 5. Enabled empty predictions can have null status | Still bounded to the nonpersisted empty-frame case: `run_and_pick` and `save_slate` refuse an empty slate. It is not a complete persisted calibration status. A null calibration record after an ancillary build failure is likewise incomplete and must be refused by the reader. |

## Required changes

This is BLOCK, not SIGN WITH EDITS. Code and regression evidence are required; no verbatim documentation edit can establish the missing containment.

1. Contain the new provenance dictionary initialization and its invocation. Preserve the forecast under failure at that statement and keep newly attached provenance off every pandas operation used for calibration, row extraction and selection. Add a deterministic allocation/preparation check at this boundary, not only a helper-access or whole-builder fault.
2. Honor PA collector failure independently of retained length and error recording, for both pipeline and calibration PA reads. Permanently withhold the affected PA provenance after a reported failure. Extend the existing append-then-raise/lost-error coverage to those actual readers and persisted output.
3. Add gate/ledger coverage for these residual contracts and rerun the affected behavior and cost evidence at any repaired candidate. Keep the distinction between fault-injected behavior, author-supplied measurements, and box extrapolation.

Because this is round 2 of 2, further repair/review needs an owner ruling; this report does not authorize a third round. A future plain SIGN would still only admit 2a to the final-diff/call-graph W1.5 certificate identification and re-runs with frozen `f453283` tooling, both runner and reviewer acceptance, the whole-range deployment review, and Eric's D7. None of those gates was performed or approved here.

## What was run

- Read the complete round-2 prompt before any other action, then the round-1 report, changed production code, relevant signed design, revised tests/harness, commit messages, mutant specification/output/classifier, cost declarations/drivers/raw rows/summaries and README. Read the cost-method register row through local `git show 6676c72`; no network or other checkout was queried. Confirmed the five-file baseline-to-candidate production inventory and measured-code equality with the reviewed candidate.
- Ran the permitted command exactly:

  ```text
  UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York OMP_NUM_THREADS=1 uv run python -B -m pytest tests/c2_2a tests/test_slate.py tests/test_local_tier.py tests/model/test_calibrate.py tests/scripts/c1_r4a/test_provenance_stored.py -p no:cacheprovider
  ```

  **221 passed, 1 skipped in 214.38 seconds.** All 57 golden comparisons passed. The single skip is the disclosed v1-reader/v2-producer compatibility test. This permitted harness may internally copy the two policy artifacts expressly allowed by the round-2 prompt; I did not separately open root `data/` artifacts.
- Ran reviewer-authored synthetic scripts under `/private/tmp/c2-2a-r2-probes`, with explicit local fixture directories and scratch park-drag/matplotlib paths. Extracted baseline functions from `git show f882411` into that scratch. Compared the attrs-copy, decoder OSError, omitted-input/lost-error, replacement calibration-record failure, undescribable-error and new helper-allocation cases against actual baseline functions. Compared all seven fixture cache pairs byte-for-byte by SHA-256: every pair matched. Also ran the short-write, package-query and failed-PA-append probes and read their persisted synthetic slates.
- Ran two real synthetic private-day paths with an empty scratch policy-source repo and the injected slate-only timestamp clock. The initial whole-observation diagnostic rejected before its timestamp-specific assertion; the later isolated comparison of the actual persisted slate observations rejected specifically on `written_at`. These are sensitivity probes, not newly generated deployed-policy goldens. The default-template scratch violation and its exact cleanup are disclosed first in the verdict.
- Ran `/private/tmp/c2-2a-r2-probes/test_remaining_gaps.py` with the checkout environment, `-B`, `-p no:cacheprovider`, and `--basetemp`/`TMPDIR` inside reviewer scratch: **2 failed in 3.93 seconds**. One fails because the injected provenance allocation escapes; the other fails because the persisted failed PA collection is not withheld. These negative checks were outside the passing author suite.
- Recomputed both revised cost summaries with their supplied summarizers; checked raw pair completeness/uniqueness and units. Checked all 76 mutant anchors and the 75 final attributable named failures without applying mutants. The classifier's own tests ran in the permitted suite.
- Inspected, but did not rerun, the supplied broader fast-suite record at `340dcef`: **4043 passed, 7 skipped, 65 deselected, 22 xfailed**. Baseline golden generation, full mutation sweep, benchmark timing/RSS, author concurrency claims and box sizes remain supplied evidence. No experiment artifact or installed-box behavior was tested.

Reviewer scratch and the identified accidental template were deleted at completion. No tracked file was edited. No commit, push, SSH, gh, box/network operation, escalation or credential/configuration read was performed. The scope exception was the synthetic default-temp template and its cleanup metadata inspection, as disclosed above.
