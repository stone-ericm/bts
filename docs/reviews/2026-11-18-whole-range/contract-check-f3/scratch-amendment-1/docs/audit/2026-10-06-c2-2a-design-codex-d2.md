## Verdict

SIGN WITH EDITS

Apply the complete verbatim diff below before building. The remaining issues have concrete corrections and need no further design decision. This approves building the amended design, not the implementation, deployment, study execution or activation. Code review, the declared cost/certification gates and Eric's D7 remain required.

## Findings

Round-1 required-change dispositions refer to the supplied d2 design, before the edits below:

| d1 required change | Disposition | Evidence |
|---|---|---|
| 1. Executable S3 recovery | Partially closed | §3.0 supplies original-path fallback and contained metadata, but retries genuine parsing/unpickling failures and omits an explicit pick-buffer fallback. Findings 1–2. |
| 2. S1 save-operation contract | Closed | §3.1 retains open/truncate order and forwards the serialization stream. Normal, hash-failure, serialization-error and partial-write probes matched the original file state. |
| 3. S2 samples/map/status | Partially closed | §3.2 defines selected samples, actual `n_fit`, canonical map state and optional metadata APIs. It loses the complete pick-input inventory and misclassifies errors after assignment. Finding 3. |
| 4. Full behavior gate | Partially closed | §5 adds the pinned generator/clocks, real transport chain, cache state, delivery traces, skip/private/fallback/restart cases and both requested behavior mutants. Its fault expectations and calibration scenarios still need the corrections in findings 1–4. |
| 5. Cost acceptance/certification | Closed | §6 declares limits before measurement and a stop consequence, and correctly requires final-candidate scope, frozen tooling, runner acceptance and reviewer acceptance. Measurement/statistic and last-round clarifications appear in finding 5. |

1. **HIGH — the recovery table retries computation, not just capture.** Design lines 55–56 say **any** buffered parsing/unpickling exception runs the original path operation. That violates lines 36 and 63's original-failure semantics and d1 required change 1. Parsing/unpickling can have effects before failing; calling it again can mask the failure.

   **Measured synthetic counterexample:** a pickled object whose `__setstate__` fails on its first invocation caused real current `load_blend` to raise `RuntimeError` after **one** invocation. Executing d2's `pickle.loads` → catch-any → `load_blend` mechanism succeeded after **two** invocations. This is a constructed cached object, not an observed production LightGBM defect. Narrowing the fallback guard to capture preparation preserved the original failure and the one-invocation count. Prepare bytes/adapters inside the capture guard; execute the parser/unpickler once outside it. Genuine failures keep their existing boundaries.

2. **HIGH — pick-file buffering still has no explicit recovery operation.** The table's calibration-samples/map row only says a witness failure never changes the fit. Section 3.2 then mandates `read_bytes`/buffered decoding for picks, without stating how a failed capture allocation obtains the original text. An implementer must invent that boundary or fall into the existing raw-probability fallback.

   **Measured:** with a synthetic 30-sample history and real calibration fitting/application, forcing only pick-file `read_bytes` to raise `MemoryError` left current `predict_local` calibrated at **1.0**. Substituting just d2's mandated buffered pick read into the current resolver returned raw **0.8** through the existing calibration handler. Adding original-`read_text` recovery for the capture failure restored **1.0**. The diff makes preparation, strict decoding and JSON parsing separate, so capture failure can recover without retrying genuine decode/JSON errors or dropping samples.

3. **MEDIUM — calibration's sample binding is useful, but its input/status contract remains incomplete.** D2 hashes files only through selected-sample bindings. That does not inventory every pick buffer the resolver actually reads before filtering. In a real-resolver probe, **31 files were read** while **30 samples** were selected; an out-of-window file was still read. Restore an ordered `pick_inputs` collection alongside the selected bindings, including sizes and hashes, as required by the original consumed-buffer contract. This is an inventory of existing reads, not additional acquisition or a new selection rule. The diff also fixes the JSON representation of payload dates and specifies the exact converted `p,y` values that enter fitting.

   The status table at lines 100–110 forces `failed` to mean `applied=false`, but the real outer handler can run **after** successful probability assignment (`src/bts/orchestrator.py:130–145`). **Measured:** an injected stderr failure on the “Applied calibration” log entered that handler while the returned frame still contained calibrated **1.0** and raw **0.8**. Keep status `applied` after successful assignment and record the later error; do not revert the forecast to make the table fit.

   Positive check: the real 30-sample fit had **two** thresholds, and its arrays plus `bool(cal.increasing_)` serialized under the proposed finite canonical-JSON rule. The new `n_fit` definition correctly counts selected samples rather than thresholds. Map and sample metadata must remain observational if extraction or serialization fails.

4. **MEDIUM — the fault gate promises an impossible error carrier, and the timing claim is overstated.** The expanded gate is substantially stronger, but line 162 demands null/error provenance even when the injected fault is the error append or attrs assignment itself. In that case, returning the same frame with an absent/null witness is the safe outcome; step 3 must refuse it. A partial metadata prefix must never be presented as complete when the error record was lost. Also distinguish actual parser/save failures from ancillary faults: preserve the baseline exception/handler/cache state, rather than requiring an available forecast where the baseline has none.

   Line 35's unconditional “or when” conflicts with the newly permitted synchronous overhead. A delay within the **5-second** maximum can cross a cutoff when the baseline had less than five seconds remaining. Fixed clocks test conditional behavior equivalence; they cannot prove identical real-clock delivery or decisions. Keep the submission guard unchanged and add an advancing-clock fixture that crosses the cutoff and still refuses delivery. The exact edits narrow the claim without relaxing the guard or changing production decision rules.

5. **LOW — cost statistics and fixture footprints need precise interpretation.** The declared **2.0-second median**, **5.0-second maximum** and **250-MB** limits are concrete prospective acceptance judgments. They are not measurements or a proof of production headroom. Specify that the RSS limit applies to the **maximum paired delta**, define MB in bytes, and report platform units. Declare representative cache and pick-directory footprints as well as compressed/decoded parquet sizes; a small real-training control alone does not exercise representative artifact hashing.

   Current code confirms a **default** 12-minute cascade budget (`src/bts/scheduler.py:2558`), not the executing box's configuration or remaining deadline headroom. The supplied 14-GB availability statement is author-reported historical context; it was not independently checked here and is not current headroom evidence. Retain the limits and stop consequence. A material capture redesign after a failed measurement goes to the owner under this last-round rule; it must still bind exact consumed/written bytes. Reopening a mutable pathname merely to hash it would reopen S1/S2.

6. **LOW — correct the unsaved-model example and schema wording.** The ordinary shadow-model training path **does save** its cache (`src/bts/orchestrator.py:188–200`); it is not an example of `trained_unsaved`. Use the actual save-path predicate. Explicit lineup booleans belong to known local producer states; other-producer nulls remain unknown. Keeping old v2 tags preserves already supported old files, without promising that strict v2 readers can read new v3 files.

**Buildability:** after the exact edits below, no remaining issue requires another design decision. The forwarding-writer approach is implementable without a full serialization buffer. Its initialization/finalization, collectors and metadata transport still need the specified containment and tests in the actual implementation. The expanded generator and real scheduler-chain tests remain future acceptance work; neither was executed against nonexistent new production code here.

## Required changes

Apply all hunks verbatim to `docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md`. The extracted patch passed `git apply --check` against HEAD; it was not applied to tracked files.

```diff
diff --git a/docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md b/docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md
--- a/docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md
+++ b/docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md
@@ -32,13 +32,14 @@
 
 ## 1. The rule this design keeps
 **Observability only.**
-- **What it never changes:** what the pipeline computes, what the cache file holds, what is selected, locked, deferred or delivered, or when. Every witness failure is contained, and an available forecast is never lost or altered.
+- **Under identical input bytes and external observations (including clocks), capture does not change** what the pipeline computes, what the cache file holds, or what is selected, locked, deferred or delivered. Every witness failure is contained, and an available forecast is never lost or altered by provenance handling.
+- **Timing is separately bounded by §6:** synchronous witness work takes time and may affect decisions near a real cutoff. This design does not establish unchanged real delivery times or deadline invariance.
 - **Genuine failures keep their exact current semantics:** a model load, parse, training, save or prediction error raises, or `predict_local` returns `None` through its existing handler, exactly as today.
 - **When the two conflict, the forecast wins,** and the witness records incomplete provenance.
 
 ## 2. Schema: `bts_slate_v3`
-- **The change:** the slate envelope gains `serving` (the witness, or null), and every row's `projected` is an explicit boolean (§4).
-- **Why v3:** v2 has been deployed since 2026-10-06 (`f882411`) without `serving`. A new version gives 4a's reader an unambiguous boundary: a 2027 capture must be v3 with a witness, and a v2 file is refused for 2027 rather than read as "witness missing". Existing v2 readers are unaffected, since v2 files keep their tag.
+- **The change:** the slate envelope gains `serving` (the witness, or null), and every known local lineup row's `projected` is an explicit boolean (§4). Missing or unsupported states from other producers remain unknown; never globally fill null with false.
+- **Why v3:** v2 has been deployed since 2026-10-06 (`f882411`) without `serving`. A new version gives 4a's reader an unambiguous boundary: a 2027 capture must be v3 with a witness, and a v2 file is refused for 2027 rather than read as "witness missing". Existing v2 files keep their tag and remain readable by readers already supporting v2; this does not assert that strict v2 readers can read new v3 captures.
 - **Unchanged:** `ROW_COLUMNS` (only the `projected` value changes, from null to false, for posted lineups); last-write-wins per date; swallow-and-log persistence.
 
 ## 3. The witness `bts.serving_witness` (from `e66b440`, amended)
@@ -50,16 +51,17 @@
 ### 3.0 The recovery rule (S3)
 Every new capture follows one pattern. A provenance failure never replaces the computation.
 
-| Step | Buffered path | On a failure of the buffered path | Witness |
+| Step | Buffered path | On a capture-preparation failure | Witness |
 |---|---|---|---|
-| **PA parquet** (`run_pipeline`, each file) | `raw = path.read_bytes()`, then `pd.read_parquet(io.BytesIO(raw))` | **any** exception in reading or parsing the buffer → the original `pd.read_parquet(path)`. A genuinely bad file raises the original error there, unchanged | `{file, bytes, sha256}`. sha256 is null with an error when hashing fails (the parse still uses `raw`) or when the original path was used ("parsed from path, not from the hashed buffer") |
-| **cached blend** (`predict_local`) | `raw = cache_path.read_bytes()`, then `pickle.loads(raw)` | any exception → the original `load_blend(cache_path)`, whose errors propagate exactly as today | the cache's sha256, or null with an error |
+| **PA parquet** (`run_pipeline`, each file) | Prepare `raw = path.read_bytes()` and `io.BytesIO(raw)` in the capture guard; call `pd.read_parquet(buffer)` once outside that guard | An exception in capture preparation → the original `pd.read_parquet(path)` once. An exception from the parser is a computation failure, never a reason to parse again | `{file, bytes, sha256}`. sha256 is null with an error when hashing fails (the parse still uses `raw`) or when the original path was used ("parsed from path, not from the hashed buffer") |
+| **cached blend** (`predict_local`) | Prepare `raw = cache_path.read_bytes()` in the capture guard; call `pickle.loads(raw)` once outside that guard | An exception in capture preparation → the original `load_blend(cache_path)` once. An exception from unpickling propagates through its original boundary, never a reason to unpickle again | the cache's sha256, or null with an error |
 | **trained blend** (`run_pipeline`) | §3.1 | — (the save itself is unchanged) | the hash of the bytes written, or null with an error |
-| **calibration PA parquet** (`predict_local`) | as for the PA parquet | the original `pd.read_parquet(current_pa)` | as above |
+| **calibration PA parquet** (`predict_local`) | as for the PA parquet | the original `pd.read_parquet(current_pa)` once on capture-preparation failure; genuine parser errors retain the existing calibration-handler behavior | as above |
+| **calibration pick JSON** (`_resolve_pick_outcomes`, each file) | Prepare held bytes and the text-decoder wrapper in the capture guard, then decode and `json.loads` once outside that guard, using the encoding, strict errors and universal newlines of `bts.picks._read_text_bytes` | An exception in capture preparation → the original `f.read_text()` and `json.loads` path once. Genuine decode/JSON/read errors retain the existing skip or outer-handler behavior, without an added retry | `{file, bytes, sha256}` for every held pick buffer; sha256 is null with an error if fallback text is consumed. Selected samples bind to that input entry |
 | **calibration samples and map** | §3.2 | a witness failure never changes the samples, the fit or its application | null parts plus errors |
 
 - **Containment boundary:** each hash, metadata extraction, list append, canonical serialization, `build()` call and `attrs` assignment runs inside its own `try/except Exception`. `MemoryError` is an `Exception`.
-- **Errors go into a local list;** appending to it is itself wrapped.
+- **Errors go into a local list;** appending to it is itself wrapped. An ancillary failure permanently invalidates the affected provenance part. If even the null/error metadata cannot be attached, omit the witness or persist it as null; never bless a partial hash or metadata prefix as complete. The forecast still returns.
 - **No new exception escapes** `run_pipeline` or `predict_local` from the witness. The only exceptions that can escape are the original computation's.
 - **No retraining because a cache hash failed,** and no substitution of raw for calibrated probabilities because calibration provenance failed.
 - **`calibration.applied`** says whether the returned forecast is calibrated, set from the actual assignment, even when provenance is incomplete.
@@ -69,7 +71,7 @@
 - **The provenance comes from the branch actually taken:**
   - `{"source": "cache", "sha256": cached_blend_sha256}` when the cached object is used;
   - `{"source": "trained", "sha256": <written bytes' hash>}` when it trains and saves;
-  - `{"source": "trained_unsaved", "sha256": null}` when it trains with no save path (the shadow model and backtests).
+  - `{"source": "trained_unsaved", "sha256": null}` when it trains with no save path (callers that explicitly omit a save path; the normal shadow-model training path saves its cache).
 
   So an empty or falsy cache trains and is witnessed as trained (S1's counterexample).
 - **The save is unchanged:** `save_blend(to_save, path)` still makes the parent directory, opens `path` for writing (truncating it **before** serialization) and calls `pickle.dump`.
@@ -78,10 +80,11 @@
 - **The witness is returned as `predictions.attrs["serving_model"]`.** Setting it is wrapped. `predict_local` never re-reads the cache path.
 
 ### 3.2 Calibration (S2)
-**`_resolve_pick_outcomes(..., bindings=None)`:**
+**`_resolve_pick_outcomes(..., bindings=None, inputs=None, errors=None)`:**
 - **The loop is unchanged:** every `2*.json` in sorted order is read before its date or result is checked, with the same filters, the same skips and the same order.
-- **One read per file:** each file is read once (`read_bytes`) and decoded with `bts.picks._read_text_bytes`, which equals `read_text` (measured equal by the d1 reviewer, including Unicode and CRLF).
-- **Bindings:** when `bindings` is a list, the resolver appends, for every sample it appends, in the same order: `{"file", "file_sha256", "date", "slot": "pick" | "double_down", "batter_id", "p", "y"}`.
+- **Normally one read per file:** prepare held bytes and a decoder with the exact text semantics of `bts.picks._read_text_bytes`, then decode once. Only capture-preparation failures use §3.0's original-text fallback; do not retry genuine decoding or JSON errors. Hash failures keep the held bytes and never trigger a reread.
+- **Pick inputs:** when `inputs` is a list, record every held pick buffer as `{file, bytes, sha256}` in read order, including files later skipped by JSON/date/result/slot filters. A fallback read or capture/hash/collector failure makes the relevant binding incomplete and records an error under §3.0. This collection is observational and never selects or drops a sample.
+- **Bindings:** when `bindings` is a list, the resolver appends, for every sample it appends, in the same order: `{"file", "file_sha256", "date", "slot": "pick" | "double_down", "batter_id", "p", "y"}`. `file` is the source basename, `date` is the payload date as ISO `YYYY-MM-DD`, `batter_id` preserves the selected JSON identity, and `p`/`y` are the exact Python `float(p)`/`int(day_hit)` appended to the computation's sample list. Metadata conversion failures affect provenance only.
   - **A hashing failure** gives `file_sha256 = null` plus an error. **It never drops the sample.**
   - **The return value is unchanged:** a list of `(p, y)`.
 
@@ -89,22 +92,23 @@
 - **The return value is unchanged:** the calibrator or `None`, with the same thresholds and fallbacks.
 - **When `witness` is a dict, it is filled with:**
   - `n_fit` = the number of selected `(p, y)` samples;
+  - `pick_inputs` = the ordered input records;
   - `samples` = the bindings;
   - `samples_sha256` = the sha256 of their canonical JSON;
   - `status` ∈ {`fitted`, `insufficient_support`, `no_sklearn`};
   - for a fitted map, `map` = `{"X_thresholds": [...], "y_thresholds": [...], "increasing": bool, "out_of_bounds": "clip", "y_min": 0.0, "y_max": 1.0}` and `map_sha256`.
-- **Canonical JSON** is `json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)`. Floats are written as Python's shortest round-trip `repr`. A non-finite value is an error, never a substitute.
+- **Canonical JSON** is `json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)`, encoded as UTF-8 before hashing. Map arrays use `.tolist()` and `increasing` uses `bool(cal.increasing_)`. Floats are written as Python's shortest round-trip `repr`. A non-finite value is an error, never a substitute.
 
 **The serving record's `calibration`:**
-- **Fields:** `{enabled, applied, status, pa_input, n_fit, samples, samples_sha256, map, map_sha256, errors}`.
-- **`status`** ∈ {`off`, `no_pa_file`, `insufficient_support`, `no_sklearn`, `failed` (the existing outer handler's genuine calibration failure), `applied`}.
+- **Fields:** `{enabled, applied, status, pa_input, pick_inputs, n_fit, samples, samples_sha256, map, map_sha256, errors}`.
+- **`status`** ∈ {`off`, `no_pa_file`, `insufficient_support`, `no_sklearn`, `failed` (a genuine calibration failure before successful probability assignment), `applied`}. Once probability assignment succeeds, status stays `applied` and `applied=true`, even if a later operation enters the existing outer handler; record that error without reverting the returned probabilities.
 - **Valid combinations:**
 
 | status | enabled | applied | required parts |
 |---|---|---|---|
 | `off` | false | false | none |
 | `no_pa_file` | true | false | none |
-| `insufficient_support` | true | false | `pa_input`, `n_fit`, `samples` |
+| `insufficient_support` | true | false | `pa_input`, `pick_inputs`, `n_fit`, `samples`, `samples_sha256` |
 | `no_sklearn` | true | false | none |
 | `failed` | true | false | errors |
 | `applied` | true | true | every part |
@@ -147,19 +151,20 @@
 
 ### 5.3 Scenarios
 - **Model:** cached; cold train; an empty cached dict.
-- **Calibration:** off; on; on but unavailable (insufficient support).
+- **Calibration:** off; on; genuine unavailable states (`no_pa_file`, `insufficient_support`, `no_sklearn`); primary/double-down and void exclusions; malformed, unresolved and out-of-window picks; the changing-historical-probabilities counterexample; and 30 selected samples collapsing to two map thresholds. Check complete ordered pick-input/sample bindings, `n_fit`, canonical hashes and normal-path single reads.
 - **Prediction failure:** the tier returns `None`.
 - **Day shape:**
   - an MDP skip;
   - private mode;
   - the fallback path with a cached pick;
   - a scheduler restart (a state reload mid-day).
-- **Injected faults, one per boundary in §3.0:**
+- **Injected ancillary faults, one per boundary in §3.0:**
   - parquet buffer allocation (`MemoryError`), parquet hash, cache buffer, cache hash;
-  - the `HashingWriter` hash, a save serialization error, a calibration buffer;
-  - a sample-binding append, the map extraction, `build()`, the `attrs` assignment.
-
-  Each must give the baseline forecast, calibration result and selected behaviour, with the provenance part null and its error recorded.
+  - `HashingWriter` construction/hash/finalization, calibration PA and pick buffer/decoder preparation;
+  - pick-input and sample-binding appends, sample canonicalization/hash, map extraction/hash, error recording, `build()`, and `attrs` assignment.
+
+  Each must give the baseline forecast, calibration result and selected behaviour. Return null/error provenance when possible; if the reporting/attachment boundary itself fails, a missing/null witness is acceptable and must be refused by step 3.
+- **Genuine computation faults:** original parquet parsing, cached-model unpickling, JSON/text decoding, model save serialization and partial write errors, and calibration fitting/application failures. Compare their original exceptions, handler results and cache-file state; never require an available forecast where the baseline has none. Include a stateful failing unpickler that must run only once, and an error after calibration probability assignment that must retain calibrated probabilities and `applied=true`.
 
 ### 5.4 Transport and mutants
 - **The `attrs` transport test** keeps `predict_local`, `run_cascade`, `run_and_pick` and `save_slate` real, and patches only lower computation and I/O leaves. It reads the written slate.
@@ -169,7 +174,7 @@
 - **The ledger** is pinned before review, with one mutant per §3 rule: the buffered and fallback branches, each containment, the `HashingWriter` forward and hash, the provenance branch, each sample field and order, `n_fit`, map canonicalisation, the status table, the explicit `False`, the v3 tag.
 
 ### 5.5 What equality establishes
-Golden equality is evidence for the tested deterministic scenarios only. The real-model test, the injected faults and the cost measurement (§6) cover the rest. None of this is a claim about the installed box.
+Golden equality is evidence for the tested deterministic scenarios only. The real-model, injected-fault and cost checks provide additional bounded evidence, not exhaustive proof. Keep the cutoff guard unchanged and add an advancing-clock fixture where an allowed observation delay crosses the cutoff: the guard must still refuse delivery. Passing fixed-clock equality or §6's cost limits does not establish real-clock decision invariance or installed-box behavior.
 
 ## 6. Production safety and certification
 ### 6.1 Cost: predeclared acceptance
@@ -184,14 +189,14 @@
   - the envelope serialization.
 - **The method:**
   - paired baseline (`f882411`) and candidate runs of `run_and_pick` in fresh processes on the Mac, with identical synthetic inputs at realistic sizes: six PA parquets of about 26 MB compressed each, and their decoded frames;
-  - the cases: a cold train (real training, small), a warm cache, calibration off, on and unavailable;
+  - the cases: a cold train (real training, small), a warm cache, calibration off, on and unavailable. Declare and report cache model count/serialized size, matching pick-file count/total bytes and decoded parquet sizes before measurement; use representative artifact/input footprints for the witness work, distinguishing the small real-training control from the representative serving cases;
   - the phases are timed separately (parquet load, cache load, save, calibration, witness, slate, selection), with buffer lifetimes noted;
   - five repeats per case, reporting the median and the maximum of the incremental time and incremental peak RSS.
-- **The acceptance, fixed now:** for every case, the incremental wall time is ≤ 2.0 s median and ≤ 5.0 s maximum, and the incremental peak RSS is ≤ 250 MB.
+- **The acceptance, fixed now:** for every case, paired incremental wall time is ≤ 2.0 s median and ≤ 5.0 s maximum, and the maximum paired incremental peak RSS is ≤ 250,000,000 bytes (250 MB). Report the platform RSS units and conversion.
 - **The relation to production headroom (reasoning):**
   - the scheduler's cascade budget is 12 minutes (`cascade_budget_min`, CLAUDE.md), so 5 s is under 1%;
   - the box showed 14 GB available on 2026-10-06 15:16, so 250 MB is under 2%.
-- **The consequence:** exceeding either limit stops 2a before code review, to revise the capture (for example chunked hashing of the path read). Mac numbers are reported with their method, and are never stated as box behaviour or deadline invariance.
+- **The consequence:** exceeding any acceptance limit stops 2a before code review. Limits are not relaxed after measurement; material capture redesign goes to the owner because this is the last design round. Any replacement mechanism must still bind the exact consumed/written bytes, never a later reread of a mutable pathname. Mac numbers are reported with their method, and are never stated as box behaviour or deadline invariance.
 
 ### 6.2 Certification
 - **Which certificates:** the affected W1.5 current-defence certificates are identified from the final diff and call graph at the final candidate. Do not assume they are the seven re-run on 10/06.
```

## What was checked

- **Identity and scope:** HEAD is `98ee17f15d8794e443841b7520d97a189b033c1b`; tracked status was clean. Both `git diff f882411 HEAD -- src/bts` and the production-code diff from d1 were empty. Read the d2 prompt first, the d1 report, the complete revised design and its diff, then the relevant current functions.
- **Forwarding writer:** a scratch prototype implements d2's hash-then-forward `write` and retains the current directory/open/truncate/`pickle.dump` sequence. A real tiny LightGBM blend produced **5,474 identical bytes**, the correct byte hash and identical round-trip probabilities. Early serialization failure left **0 bytes** on both paths; a late failure left the same **200,028-byte** prefix; a synthetic partial-write failure left the same **2,737 bytes** and exception. Injected hash `MemoryError` left cache bytes unchanged and null provenance. These are synthetic mechanism checks, not implementation acceptance or production cost measurements.
- **Recovery/calibration:** the remaining probes used the real loader, resolver, isotonic fitter, application and `predict_local`, with only the indicated buffered-read substitution, fault injection and base-prediction leaf substituted. All **12 probe cases** completed with their assertions satisfied. Corrected cache recovery preserved one genuine unpickling attempt; corrected pick capture recovery preserved the calibrated forecast. Metadata/input/status findings distinguish static contract gaps from measured results.
- **Gate and safety:** checked the specified deterministic comparison surface, transport functions, mutants, error expectations, predeclared limits and certification rule. No new producer, golden-generator or scheduler-integration implementation exists in this design diff. Existing production code is unchanged, so the d1 suite was not repeated; this round used focused mechanism probes instead. No full-size benchmark or W1.5 recertification was run.
- **Patch and final state:** verified the complete report's fenced diff is exactly the checked design patch; checked report headings, verdict, unchanged production pin and clean tracked tree. Only the requested report and task-owned scratch were written. Scratch was removed before completion.
- **Constraints:** no memory, shared corpus, other repository, real `data/`, `.env`, production configuration or credential contents were read in this round. Runtime caches were directed into the permitted scratch prefix from the first probe. No pytest invocation read repository configuration. Arrow's denied CPU-capability `sysctl` diagnostics were nonfatal; no escalation was requested. No network, SSH, `gh`, commit, push, deployment or live operation occurred.
