## Verdict

**BLOCK.**

Scope disclosure first: my initial tool batch read the complete prompt and concurrently searched `/Users/eric/.codex/memories/MEMORY.md` for `C2|c2-2a|rank-4|C1`. That outside-checkout search happened before I had read the prompt's prohibition. I read its returned excerpts, did not open any referenced memory, and excluded that material from the review evidence. This was a review-process violation, not an authorized exception.

Reviewed-commit: 08fe9c8b34fab9191a2a9c4ffce0197fd39112d9

The observability-only claim is false under the required ancillary-fault contract. I reproduced usable calibrated forecasts becoming raw forecasts, and a usable forecast becoming `None`, solely because of new provenance handling. I also persisted an `applied` witness that omitted a consumed pick input with both error lists empty. The permitted suite nevertheless passed: **184 passed, 1 skipped**.

HEAD is the requested detached commit and the tracked tree was clean before and after the checks. `git diff f882411 9d24d57 -- src/bts` is empty. The baseline-to-candidate production diff contains exactly the five files named in the prompt. This verdict authorizes no production edit, certification acceptance, deployment, or activation.

## Findings

### F1 — Blocking: pandas copies the new provenance attrs inside the calibration computation

**Locations:** `src/bts/model/predict.py:1009–1011`; `src/bts/orchestrator.py:254–259,277–278`.

`run_pipeline` attaches `serving_errors`, `serving_model`, and `serving_inputs` before returning. `predict_local` leaves those attrs on the frame throughout calibration. With the installed pandas, extracting/copying a predictions Series copies the frame's attrs. An allocation failure copying this newly added metadata therefore enters the existing calibration exception handler and bypasses probability assignment. Dropping the attrs after `save_slate` is too late to protect this computation.

**Reproduced against actual baseline source:** I extracted `predict.py`, `orchestrator.py`, and `calibrate.py` from `f882411` into reviewer scratch and executed their functions with the same synthetic fixture and the same lower computation leaves as `tests/c2_2a/test_local_witness.py`. Only copying a dict containing the new `serving_model` key was faulted:

```python
g = pd.DataFrame.__finalize__.__globals__
real = g["deepcopy"]
def failing(obj, *a, **k):
    if isinstance(obj, dict) and "serving_model" in obj:
        raise MemoryError("synthetic: copying newly attached provenance attrs")
    return real(obj, *a, **k)
g["deepcopy"] = failing
```

For the fixture's raw probabilities `[0.86, 0.81, 0.74]` and 40 resolved samples:

| side | returned probabilities | result |
|---|---|---|
| f882411 | `[0.65, 0.65, 0.65]` | calibration assigned |
| candidate | `[0.86, 0.81, 0.74]` | witness says `failed`, `applied=false`, `n_fit=40` |

I separately ran the real pipeline, prediction, cascade, slate, and selection path with the golden harness's synthetic world and actual baseline functions. The baseline's top served probability was `0.6`; the candidate's was `0.7611060665510311`, with calibration `failed`. Both selected skip in that particular world; I do not claim a measured delivery difference. The computed forecasts already contradict §1 and §3.0's prohibition on substituting raw probabilities because provenance failed.

This is a modeled allocation failure at a real pandas propagation boundary, not a claim that the box has experienced memory exhaustion. The existing attrs tests fault an attrs accessor or attachment helper; they do not fault copying successfully attached metadata during calibration. Both the golden suite and those tests pass with this defect present.

### F2 — Blocking: decoder-preparation OSError is misclassified as an unreadable input

**Location:** `src/bts/model/calibrate.py:65–74,130–135`.

The same `try` contains the buffered file read and decoder preparation. Its `except OSError` therefore handles an `OSError` from `io.BytesIO`, `io.text_encoding`, or `io.TextIOWrapper` as though the file read failed. The resolver skips the file instead of taking the original text fallback. The author's decision applies specifically to a genuine buffered-read failure; the implementation applies it to the whole preparation region.

**Reproduction:** replace only `calibrate.io` with this proxy; the original `Path.read_text()` uses its own normal decoder:

```python
class IoProxy:
    def __getattr__(self, name):
        return getattr(io, name)
    @staticmethod
    def TextIOWrapper(*a, **k):
        raise OSError("synthetic: decoder preparation, after successful read")
```

With the same baseline-source comparison and 40 usable picks, f882411 returned `[0.65, 0.65, 0.65]`. The candidate returned `[0.86, 0.81, 0.74]`, recorded `n_fit=0` and `insufficient_support`, and described all successfully read files as “unreadable, skipped.” No text fallback ran. This changes both the samples and the forecast because an ancillary preparation step failed.

The golden decoder scenario injects only `MemoryError`; the read-error unit test injects `PermissionError` at `Path.read_bytes`. Neither tests an `OSError` after a successful buffered read. Separating the genuine read's exception boundary from decoder preparation is necessary.

### F3 — Blocking: collector failure has no durable invalidation, and losing its error can bless incomplete provenance

**Locations:** `src/bts/serving_witness.py:49–66`; `src/bts/model/calibrate.py:223–230`; `src/bts/orchestrator.py:109–120`.

`collect` catches a failed append and calls `note`. `note` silently discards a failed error append. Neither returns a failure indication or invalidates the collector. Fitting subsequently publishes the retained list and hashes it. An ancillary failure thus does not permanently invalidate its affected provenance part, contrary to §3.0. Even when an error is retained, a partial sample list can receive a non-null `samples_sha256`; when error recording also fails, an incomplete input list can look entirely clean.

**Persisted reproduction:** use the local-witness fixture with its 40 usable history files and add `2026-04-02.json`, an out-of-window pick which the unchanged loop must read. Route its input append through an append proxy which raises `MemoryError`, while other appends forward to the real list. Route the resulting error append through a list whose `append` also raises. Both failures run through the real `collect` and `note` helpers. Run real `predict_local`, then real `save_slate`.

The written v3 slate contained:

```text
status=applied; applied=true; n_fit=40; samples=40
41 history files consumed; pick_inputs records only 40
2026-04-02.json omitted
calibration.errors=[]; serving.errors=[]
samples_sha256 and map_sha256 both present
```

The forecast remained correctly calibrated. The provenance was falsely complete. An omitted out-of-window input leaves all selected-sample counts and joins consistent, so this is not rescued by `n_fit == len(samples)`.

I ran the unchanged golden `_assert_calibration_bound` against this persisted calibration record and the actual hashes of all 41 synthetic history files: **PASS**. The assertion iterates retained input records; it never requires the full consumed-input inventory. The golden collector scenario uses fresh always-failing collectors and explicitly accepts empty lists with a positive `n_fit`. The error-recording scenario combines error loss with null parquet hashes, which remain visibly incomplete; it does not exercise a silently omitted input or binding.

The same structural issue affects failed calibration-field assignments: `_cal_set` leaves the old value and records an error instead of invalidating the record. In particular, an unsuccessful metadata assignment of `applied=true` can leave `applied=false` even after probabilities were assigned. Under the design, an unattainable truthful record must become incomplete/null, not retain a stale assertion.

### F4 — Blocking: helper guards do not contain preparation of their arguments or their error descriptions

**Examples:** `src/bts/model/predict.py:208–212,984,994,997`; `src/bts/model/calibrate.py:69–76,85,173`; `src/bts/serving_witness.py:66,74,125,134`; `src/bts/orchestrator.py:144–152,215–217`.

The guarded append/hash functions receive dictionaries, metadata, and formatted exception descriptions prepared outside their guards. Exceptions in an `except` suite also are not caught by that same `try`. For example, `_read_pa_parquet` formats `e!r` before calling the contained `note`. If that formatting fails, the original-path fallback never executes. New model-provenance dict construction is similarly outside a capture guard.

**Concrete reproduction:** make only `Path.read_bytes()` for a PA file raise this ancillary preparation exception:

```python
class BadRepr(MemoryError):
    def __repr__(self):
        raise MemoryError("synthetic: formatting witness error")
```

The original path parser remains usable. f882411 returned the calibrated forecast and wrote its 52-byte fixture cache. The candidate's error-description construction raised; its existing prediction handler returned `None`, printed `Prediction failed: synthetic: formatting witness error`, and no cache was created. This is a newly escaping provenance exception from the pipeline, causing forecast and cache behavior to change.

The custom exception makes the vulnerable boundary deterministic; a failed allocation constructing the error string or record has the same missing containment. A claim that every metadata operation has its own guard is therefore overstated. Catching `MemoryError` inside `note` does not guard work evaluated before `note` is called.

Additionally, `_packages` converts a version-query exception into a null value without recording any error (`serving_witness.py:106–113`). The null remains visibly incomplete, but this is another difference from the stated “every provenance failure” error contract; its unit test currently codifies the silence.

### F5 — Bounded write-fault defect: the digest assumes every successful write consumed the full argument

**Location:** `src/bts/model/predict.py:160–167`.

`HashingWriter` hashes the whole argument before forwarding it and never checks the real write's returned count. With a sink which successfully writes only a prefix:

```python
class ShortSink(io.BytesIO):
    def write(self, b):
        return super().write(b[:2])
sink = ShortSink()
writer = P.HashingWriter(sink, [])
writer.write(b"abcdef")             # returns 2; sink contains b"ab"
```

The digest was `bef57ec7…` (sha256 of `abcdef`), while the actual written bytes hash to `fb8e20fc…` (sha256 of `ab`), with no error. A completed save through such a sink would publish the wrong written-byte digest. Preserve the original forwarded bytes and return behavior, but null/invalidate provenance on a short or unsupported write result.

This is a synthetic lower-I/O fault, not evidence that the normal blocking buffered file writer short-writes. The ordinary writer and error-raising partial-write scenarios passed. Nevertheless, the unconditional “exactly the bytes written” assertion and the permitted fault envelope do not hold for this case. P3/P4 use a full-writing `BytesIO` or normal files; the partial-write fixture raises instead of returning a partial count.

### F6 — Gate coverage: there is a reproduced false green and several missing fault surfaces

`golden/scenarios.py:_slate` removes `written_at` from observation, and `_compare` checks a selected subset of the envelope. The signed gate permits only three slate differences; changing this existing timestamp is a fourth difference.

**Reproduced visibility mutant, without a tracked edit:** I ran the real path twice with the same golden world and patched only the slate writer's `datetime.now` on the second run. The two complete `Harness.observe()` results were equal, while the actual slate timestamps were:

```text
normal:  2026-06-30T14:00:00+00:00
mutant:  2037-01-01T00:00:00+00:00
gate_observations_equal=true
```

The gate also does not observe stderr or the complete API call log, only unknown URLs. A genuine error caught inside `predict_local` is represented by its returned result, so changed caught exception types/messages need not appear in `obs["exception"]`. This limits the claimed genuine-failure comparison. I found no difference in the tested corrupt-parquet error's type/message, or in the safe pickle-truncation corpus described below.

Boundary assessment:

- `fault_parquet_buffer` and the corresponding unit test fault `Path.read_bytes`, not construction of `io.BytesIO` after a successful read. The fallback code exists, but that distinct allocation boundary is not exercised by this scenario.
- `fault_pick_decoder` does reach preparation after held bytes exist. Its exact extra-read assertion in `FALLBACK_REREAD` is appropriate for the signed fallback; it is not a blanket equality exemption. It misses F2's exception class.
- The read counters are wrappers around `read_bytes`/`read_text`, not lower-level I/O receipts. The injected read-byte failure can occur outside the counting wrapper. Counts therefore do not establish every attempted read or every allocation boundary.
- The collector scenario faults fresh replacement collectors, rather than a collector which has already retained a prefix. The error-recording scenario tests the recording failure but not permanent invalidation when the lost error is the only evidence of an omitted input.
- The attrs scenario tests helper attachment/access failure using stand-in objects. It does not cover pandas copying attached provenance on the real calibration frame (F1).
- The advancing-clock scenario advances scheduler observations after prediction and still tests refusal at the guard. It does not delay synchronous witness work across the cutoff. It supports the unchanged guard, not real-time delivery invariance; the design already disclaims that stronger conclusion.
- The training stand-ins are an allowed design choice. The separate real LightGBM save/round-trip/probability test passed. They are bounded evidence, not a proof for every real model's behavior.

**Mutant ledger:** I checked all 59 unique entries and all replacement anchors against the candidate; none was stale. The supplied output names assertion failures for 58 attributable mutants, including resumed C9/C11 and C13b. I did not run the tracked-file-mutating runner. Its general `RED` classification treats any nonzero pytest exit as red, although this supplied ledger includes named failed tests rather than just exit codes. Preserve the named assertion evidence.

C13 is equivalent for this unchanged constructor: the default is `increasing=True`, and a current fit on decreasing outcomes also returned `increasing_=True`. That disposition is sound. It does not establish full rule coverage. Missing independent mutants/fault checks include:

- permanent invalidation of a collector after a partial append failure, including failed error recording;
- metadata/error-description preparation outside helper guards;
- decoder-preparation `OSError` versus genuine read `OSError`;
- pandas attrs propagation during calibration;
- short successful writes;
- metadata assignment failure at the actual `applied`/status updates;
- map extraction and map hashing failed independently of sample hashing; the shared canonicalization-fault test faults both together;
- independent sample `y` corruption (C9 mutates `p`, despite its combined rule label), and map policy fields other than thresholds/direction;
- swallowed package-version-query errors and the omitted envelope timestamp.

### Conformance and the five author decisions

| contract | assessment |
|---|---|
| §2 | v3 tag, `serving`, unchanged `ROW_COLUMNS`, last-write behavior, and swallowed persistence failures implemented. Unsupported other-producer projected values are not globally filled. |
| §3.0 | buffered consumption and ordinary fallbacks implemented; containment, permanent invalidation, and forecast preservation fail as F1–F4 show. |
| §3.1 | truthiness branch and `_model` pop preserved; cached/trained/trained-unsaved origins correct in tested cases; ordinary saved bytes preserved. Metadata preparation and short-write truth require F4/F5. |
| §3.2 | normal ordered inputs/bindings, exact sample floats/integers, `n_fit`, canonical JSON and fitted map implemented. Tested genuine/unavailable status branches and post-assignment handler behavior pass. Incomplete collectors/failed metadata assignment require F3, and preparation failures can change the samples/assignment. |
| §4 | explicit true/false implemented by real `_fetch_game_slots`; the permitted tests exercise persisted projected and posted rows. Selection's flag-derived projected state is unchanged in the tested ordinary cases. |

| author decision | disposition |
|---|---|
| 1. Pop `serving` after `save_slate`; move pipeline attrs into the witness | Sensible normal-path optimization, but insufficient: pipeline attrs must be isolated before calibration computation, and failed removal/transport must leave truthful incomplete provenance. The quoted 100 ms/1.8 ms measurement is supplied evidence, not my measurement. |
| 2. Genuine buffered-read OSError skips without retry | Accept the stated distinction; reject its overbroad implementation across decoder preparation (F2). |
| 3. Witness v2, reader test skip, park-drag env flag | Reasonable producer version boundary and allowlist addition. The skip means reader compatibility is not certified here. Step 3 must realign and validate the reader before study admission. |
| 4. Experiment parquet gains provenance attrs | Disclosed outside-pick-path artifact change. I inspected the propagation-producing code; I did not run `bts experiment` or independently verify its parquet output/unchanged values. Do not count the pick-path gates as an experiment-artifact compatibility test. |
| 5. Enabled empty predictions leave status null | A literal exception to the stated status set. It is bounded: `run_and_pick` exits before `save_slate`, and `save_slate` also refuses empty predictions. No persisted empty-slate violation reproduced. Document this nonpersisted case or omit its witness. |

### Cost: original metric is not met; the authorized replacement screen passes its recorded rules

I recomputed both summaries from the retained JSONL using the supplied summarizers: they match exactly (50 whole-run rows, 120 phase rows). These are arithmetic reproductions, not new timing/RSS measurements.

| case | paired time median / maximum, seconds | paired peak RSS maximum, decimal MB |
|---|---:|---:|
| cold_small | 0.009 / 0.440 | 10.256 |
| warm_off | 1.936 / 3.867 | **1505.673** |
| warm_on | 0.217 / 0.791 | -18.579 |
| warm_unavailable | 0.532 / 1.274 | 226.066 |
| identical-baseline control | 0.247 / 0.902 | **304.775** |

All five paired time samples per case satisfy the recorded median/maximum rules. The disclosed overlapping tests contaminate two warm_off pairs; the numbers remain a measured pass under the supplied ruling, not a clean estimate of intrinsic overhead. I did not rerun the benchmark or infer box timings.

The original predeclared whole-run RSS maximum fails. The identical-code control exceeding 250 MB, together with large same-baseline variation, supports treating this particular five-pair comparison as unresolved in both directions. It does not prove the true increment is small, nor prove that a whole-run metric can never be measured reliably. The prompt supplies authority for the replacement method; I verified its declaration exists in `30f5e5e`, before the phase implementation/results commits. The literal `C2-2a-cost-method` row was not found in the pinned exposure register; the authority itself is supplied by the prompt and README, not independently established from that register.

The replacement phases cover the normal added allocation intervals: PA load through concatenation; held cache through unpickling; hashing save; calibration/witness construction and intermediate attrs propagation; and witness-bearing slate serialization. After successful removal of the witness, selection has no corresponding large provenance object to copy. The padded stand-in cache preserves the specified serialized footprint and model count; it does not model all native LightGBM deserialization allocations. The real small-training control and representative save phase are correctly distinguished.

All six recorded phase maxima are below the unchanged 250 MB threshold: load 21.479 MB, cache 9.208 MB, save 0.426 MB, tail_off 1.294 MB, tail_on 3.490 MB, slate 1.999 MB. Each same-code control's maximum absolute difference is at most 3.179 MB. I accept this as the **explicitly authorized replacement memory screen for the declared synthetic footprint**. It is not a measured pass of the original whole-run peak metric: `ru_maxrss` is a process-lifetime high-water mark, and separate processes do not directly measure accumulated allocator residency in one complete cascade.

The six-parquet/9.30 MB/156-file footprints and 351-row slate are declared and plausibly representative for the bounded witness work. The real box sizes, ten-parquet scaling argument, and available-headroom claims were not independently verified here. Both raw datasets label RSS as macOS bytes, so the summaries' direct byte subtraction and decimal MB conversion are consistent. No result establishes installed-box memory, real-cutoff invariance, or fault-path cost.

One README correction: the cache phase's stop is a `BaseException`, so it escapes `predict_local`'s `except Exception` and is caught by the phase driver; the handler does not return `None` as the README says. This does not defeat its peak measurement. Also align the summary's resolution rule with the README: currently it will count a control below 250 MB even when outside the 125 MB aim. The retained controls all satisfy the aim, so this discrepancy does not change these results.

The cost-method ruling does not cure F1–F4. After changing provenance lifetimes/containment, rerun the affected behavior and cost gates at the final candidate.

## Required changes

This is BLOCK, not SIGN WITH EDITS; the repairs require code and regression evidence, not verbatim documentation edits alone.

1. Isolate pipeline provenance from every pandas operation used to compute or apply calibration. Move/hold it under containment before those operations, and build/attach the final witness only after computation. Prove the real forecast stays calibrated under an attrs-copy allocation fault, including the assignment and post-assignment paths.
2. Give the genuine buffered read its own `OSError` boundary. Decoder/buffer preparation after a successful read must take the signed fallback for any preparation exception. Keep genuine read/decode/JSON failures single-attempt and preserve baseline exception/handler/file semantics.
3. Track collector and metadata invalidation independently of the success of error recording. Never publish a hash of a partial sample collection as complete. If truthful null/error state cannot be attached, omit/null the witness. Preserve actual calibration assignment truth; a failed metadata write must not leave stale `applied`/status assertions. Add a persisted omitted-skipped-input reproduction with failed error recording.
4. Put record construction, metadata extraction, error-description construction, and each other new ancillary operation inside a containment boundary covering the operation itself. Ensure error reporting cannot replace a parser, unpickler, writer, calibrator, or forecast. Record package query failures or explicitly revise that error contract; do not silently call them complete.
5. Null/invalidate the written-byte digest when a successful write result does not certify the entire forwarded argument. Do not change baseline write behavior or add retries. Verify both successful short writes and error-raising partial writes.
6. Compare the full pre-existing slate envelope, including `written_at`; bind completeness assertions to the full consumed-input inventory. Add the missing fault/mutation checks above, retain assertion-level mutant attribution, and rerun the permitted behavior gate plus appropriate cost checks at the repaired candidate. Correct the two phase-method documentation/driver discrepancies noted above. Keep supplied evidence, reproduced checks, and box extrapolations distinct.

The final diff/call graph must still identify the affected W1.5 certificates, rerun them using frozen `f453283` tooling, and retain both runner acceptance and reviewer acceptance. That work, the whole-range deployment review, and Eric's D7 remain separate gates.

## What was run

### Reproduced in this review

- Read the complete prompt, signed d2 design, full production diff, commit messages, new tests/harness, mutant ledger/output/runner, benchmark code/declaration/results and relevant in-checkout register/index references. Verified HEAD, clean tracked tree, unchanged deployed source at the design commit, and the exact five-file production change inventory.
- Ran the permitted command exactly:

  ```text
  UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York OMP_NUM_THREADS=1 uv run python -B -m pytest tests/c2_2a tests/test_slate.py tests/test_local_tier.py tests/model/test_calibrate.py tests/scripts/c1_r4a/test_provenance_stored.py -p no:cacheprovider
  ```

  **184 passed, 1 skipped in 198.91 seconds.** All 46 golden comparisons passed. The skip is the disclosed v1-reader/v2-builder compatibility test.
- Ran reviewer-authored synthetic probes under `/private/tmp/c2-2a-r1-probes`, importing the permitted local fixture/harness. Baseline comparison functions came from `git show f882411:<source-path>` into that scratch. Heavy fixture training/feature leaves were shared; the end-to-end comparison kept feature computation, prediction, cascade, slate and selection real. End-to-end harnesses used an empty scratch repo for their optional policy-copy source, so they did not copy checkout `data/` artifacts. Thus their selection result is deliberately not presented as a reproduction of a production-policy decision. Verified that the imported candidate predict, calibrate, and orchestrator modules point into this checkout's `src/bts`.
- Reproduced F1's attrs-copy forecast change, F2's decoder-preparation forecast change, F3's persisted input omission with empty errors and passing golden binding assertion, F4's missing forecast/cache, F5's short-write hash mismatch, and F6's equal observations despite unequal persisted timestamps. The report retains their fault construction and results; reviewer scratch was removed at completion.
- Compared genuine corrupt-parquet parsing directly: both raised `ArrowInvalid` with the same message for the tested buffer. Compared every truncation of a safe builtin-only dict pickle across protocols 0–5 using stream load and byte load: no type/message/value differences found. This is a bounded corpus, not exhaustive genuine-error equivalence.
- Confirmed C13's current default-direction behavior with a decreasing-outcome fit. Checked all 59 mutant IDs/anchors without applying them. Recomputed both cost summaries and verified their raw RSS unit labels and reported calibration states.

### Author-supplied evidence, not independently rerun

- Generation of the 46 goldens in the detached baseline worktree. The permitted suite checked the retained manifest, harness and golden hashes and current package-version equality; I did not independently regenerate all baseline goldens.
- The 58 attributable mutant kills and C9/C11 resume. I inspected the named failures and current anchors; I did not run the runner, which edits tracked files.
- Whole-run and phase timings/RSS, the corrected load probe's measurements, concurrent-test contamination, box footprint/headroom measurements, and the delegation ruling. I reproduced summary arithmetic and inspected methods, not those measurements or remote authority records.
- The supplied fast-suite result: **4023 passed, 7 skipped, 54 deselected, 22 xfailed**, recorded at `2430ac2`. I did not run that unpermitted broader suite.
- The experiment parquet metadata/value claim and the quoted attrs performance measurement.

No network, box, SSH, gh, escalation, tracked edit, commit, or push was performed. The explicitly permitted pytest run used its harness's default temporary directories. Its `world.write_world` internally copies `data/models/mdp_policy.npz` and `data/models/mdp_tail_policy.npz` when present; these optional policy inputs are in the retained goldens and the passing candidate equality check. This internal behavior of the specifically allowed command conflicts with the prompt's general no-`data/`-reads wording and should be disclosed; I did not issue a separate command to open those artifacts or other checkout data. Reviewer probes used synthetic inputs only and avoided that policy copy. An initial probe invocation failed to import `tests` until `PYTHONPATH` was set; that invocation also caused matplotlib to choose its normal temporary cache outside the reviewer prefix. Later probes explicitly placed that cache and the park-drag path in reviewer scratch. No configuration or credential file was deliberately opened. The outside memory search disclosed in the verdict is excluded from every substantive finding.
